"""Train lại phần đầu mô hình tiên lượng trên đặc trưng DenseNet121 đã cache, chọn mô hình chỉ bằng tập val.

Chạy:
    python research/prognosis/train.py --data $DATA --manifest research/prognosis/split_manifest.csv \
        --out research/prognosis/runs --seeds 0 1 2 3 4 [--install]

--install chép checkpoint của seed được chọn và rna_reference.npz vào backend/ai_core/weights/.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_core.architectures.survival_net import CoxPHLoss, MultimodalBrainTumorModel, c_index_score  # noqa: E402
from ai_core.feature_schema import (  # noqa: E402
    MRI_SLICES,
    WSI_PATCHES,
    encode_clinical,
    encode_rna,
    fit_rna_reference,
    save_rna_reference,
    uniform_indices,
)
from feature_model import forward_from_features  # noqa: E402

FEATURE_DIM = 1024
WEIGHTS_DIR = ROOT / "backend" / "ai_core" / "weights"


@dataclass
class TrainConfig:
    """Siêu tham số; mặc định giữ như notebook (AdamW 5e-5 / 5e-2, ReduceLROnPlateau 0.5 / 5)."""

    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 5e-5
    weight_decay: float = 5e-2
    patience: int = 15
    lr_factor: float = 0.5
    lr_patience: int = 5


@dataclass
class Patient:
    """Một bệnh nhân đã mã hóa sẵn: đặc trưng ảnh [n, 1024] (None nếu thiếu), RNA đã chuẩn hóa, lâm sàng 18 chiều."""

    pid: str
    time: float
    event: float
    study: float
    mri: np.ndarray | None
    wsi: np.ndarray | None
    rna: np.ndarray | None
    clinical: list[float]
    has_clinical: float


def select_split(manifest: pd.DataFrame, splits: tuple[str, ...]) -> pd.DataFrame:
    """Lọc manifest theo split; train.py không bao giờ được đọc tập test.

    Raises:
        ValueError: nếu splits có "test".
    """
    if "test" in splits:
        raise ValueError("train.py không được đọc tập test; tập test chỉ dùng trong evaluate.py")
    return manifest[manifest.split.isin(splits)].reset_index(drop=True)


def load_patients(data_dir: Path, rows: pd.DataFrame, rna_matrix: pd.DataFrame, rna_ref) -> list[Patient]:
    """Nạp đặc trưng cache, RNA (encode_rna) và lâm sàng (encode_clinical) cho các dòng manifest.

    rna_ref=None: để rna=None cho mọi ca (cv.py tự mã hóa RNA lại theo từng fold).
    """
    clinical_csv = pd.read_csv(data_dir / "dataset_multimodal_final.csv").set_index("patient_id")
    features = data_dir / "features"
    patients = []
    for row in rows.itertuples():
        def cached(kind, flag):
            path = features / kind / f"{row.patient_id}.npy"
            return np.load(path) if flag == 1 and path.exists() else None

        values, has_clinical = encode_clinical(clinical_csv.loc[row.patient_id].to_dict())
        rna = encode_rna(rna_matrix.loc[row.patient_id], rna_ref) if row.has_rna == 1 and rna_ref is not None else None
        patients.append(Patient(row.patient_id, float(row.time), float(row.event), float(row.study_type),
                                cached("mri", row.has_mri), cached("wsi", row.has_wsi), rna, values, has_clinical))
    return patients


def _pick(n: int, k: int, train: bool, rng: np.random.Generator) -> list[int]:
    # Train: bốc ngẫu nhiên không lặp mỗi epoch (như notebook); val/test/serving: lấy đều.
    if train and n > k:
        return sorted(rng.choice(n, k, replace=False).tolist())
    return uniform_indices(n, k)


def _stack(arrays: list[np.ndarray | None], k: int, train: bool, rng) -> tuple[torch.Tensor, torch.Tensor]:
    chosen = [a[_pick(len(a), k, train, rng)] if a is not None else np.zeros((0, FEATURE_DIM), np.float32) for a in arrays]
    width = max(1, max(len(c) for c in chosen))
    feats = torch.zeros(len(arrays), width, FEATURE_DIM)
    mask = torch.zeros(len(arrays), width)
    for i, c in enumerate(chosen):
        feats[i, : len(c)] = torch.from_numpy(np.asarray(c, dtype=np.float32))
        mask[i, : len(c)] = 1.0
    return feats, mask


def make_batch(patients: list[Patient], num_genes: int, train: bool, rng, device: str) -> dict[str, torch.Tensor]:
    """Gom batch: đặc trưng ảnh pad theo lát dài nhất kèm mask, cờ has_* theo dữ liệu thật."""
    mri, mri_mask = _stack([p.mri for p in patients], MRI_SLICES, train, rng)
    wsi, wsi_mask = _stack([p.wsi for p in patients], WSI_PATCHES, train, rng)
    rna = torch.stack([torch.from_numpy(p.rna) if p.rna is not None else torch.zeros(num_genes) for p in patients])
    batch = {
        "mri_feats": mri, "mri_mask": mri_mask, "wsi_feats": wsi, "wsi_mask": wsi_mask, "rna": rna,
        "clinical": torch.tensor([p.clinical for p in patients], dtype=torch.float32),
        "has_mri": torch.tensor([float(p.mri is not None) for p in patients]),
        "has_wsi": torch.tensor([float(p.wsi is not None) for p in patients]),
        "has_rna": torch.tensor([float(p.rna is not None) for p in patients]),
        "has_clinical": torch.tensor([p.has_clinical for p in patients]),
        "time": torch.tensor([p.time for p in patients]),
        "event": torch.tensor([p.event for p in patients]),
    }
    return {k: v.to(device) for k, v in batch.items()}


def _forward(model, batch):
    keys = ("mri_feats", "wsi_feats", "rna", "clinical", "has_mri", "has_wsi", "has_rna", "has_clinical")
    risk, _ = forward_from_features(model, *(batch[k] for k in keys), mri_mask=batch["mri_mask"], wsi_mask=batch["wsi_mask"])
    return risk


@torch.no_grad()
def predict(model, patients: list[Patient], num_genes: int, device: str) -> np.ndarray:
    """Risk score (eval, lấy lát đều) cho danh sách bệnh nhân."""
    model.eval()
    return _forward(model, make_batch(patients, num_genes, False, None, device)).cpu().numpy()


class _Run:
    """Trạng thái train một seed (mô hình, optimizer, bộ lấy mẫu cân bằng theo nguồn) dùng chung cho mọi cách train."""

    def __init__(self, train_patients: list[Patient], num_genes: int, seed: int, cfg: TrainConfig):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MultimodalBrainTumorModel(num_genes=num_genes).to(self.device)
        # Backbone DenseNet đóng băng: đặc trưng đã cache; chỉ train phần đầu.
        self.params = [p for name, p in self.model.named_parameters() if "feature_extractor" not in name]
        self.optimizer = torch.optim.AdamW(self.params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        self.criterion = CoxPHLoss()
        studies = pd.Series([p.study for p in train_patients])
        self.weights = torch.tensor(studies.map(1.0 / studies.value_counts()).to_numpy(), dtype=torch.float64)
        self.generator = torch.Generator().manual_seed(seed)
        self.train_patients, self.num_genes, self.cfg = train_patients, num_genes, cfg

    def epoch(self) -> list[float]:
        """Một epoch: bốc mẫu có hoàn lại theo trọng số nguồn, bỏ batch không có event. Trả list loss từng batch."""
        self.model.train()
        order = torch.multinomial(self.weights, len(self.train_patients), replacement=True, generator=self.generator).tolist()
        losses = []
        for start in range(0, len(order), self.cfg.batch_size):
            chosen = [self.train_patients[i] for i in order[start:start + self.cfg.batch_size]]
            batch = make_batch(chosen, self.num_genes, True, self.rng, self.device)
            if batch["event"].sum() == 0:
                continue
            loss = self.criterion(_forward(self.model, batch), batch["time"], batch["event"])
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
            self.optimizer.step()
            losses.append(loss.item())
        return losses


def train_fixed_epochs(train_patients, num_genes: int, seed: int, cfg: TrainConfig, epochs: int,
                       eval_patients=None, out_path: Path | None = None) -> np.ndarray:
    """Train đúng `epochs` epoch, learning rate cố định, không early stopping (dùng cho CV và bản cuối v2).

    Args:
        eval_patients: nếu có, dự đoán risk cho các ca này sau mỗi epoch.
        out_path: nếu có, lưu state_dict đầy đủ sau epoch cuối.

    Returns:
        np.ndarray [epochs, len(eval_patients)] (cột rỗng nếu không có eval_patients).
    """
    run = _Run(train_patients, num_genes, seed, cfg)
    preds = []
    for _ in range(epochs):
        run.epoch()
        preds.append(predict(run.model, eval_patients, num_genes, run.device) if eval_patients else np.zeros(0))
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(run.model.state_dict(), out_path)
    return np.stack(preds)


def train_seed(train_patients, val_patients, num_genes: int, seed: int, cfg: TrainConfig, out_dir: Path) -> dict:
    """Train một seed, early stop theo C-index val, lưu state_dict đầy đủ của epoch tốt nhất vào out_dir/best.pth.

    Returns:
        dict gồm seed, best_val_c_index, best_epoch, history (list theo epoch).
    """
    run = _Run(train_patients, num_genes, seed, cfg)
    model, optimizer, criterion, device = run.model, run.optimizer, run.criterion, run.device
    out_dir.mkdir(parents=True, exist_ok=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience)
    val_batch = make_batch(val_patients, num_genes, False, None, device)

    history, best, best_epoch, stale = [], -1.0, -1, 0
    for epoch in range(cfg.epochs):
        losses = run.epoch()

        model.eval()
        with torch.no_grad():
            val_risk = _forward(model, val_batch)
            val_loss = criterion(val_risk, val_batch["time"], val_batch["event"]).item()
        val_c = float(c_index_score(val_risk, val_batch["time"], val_batch["event"]))
        scheduler.step(val_loss)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)) if losses else None,
                        "val_loss": val_loss, "val_c_index": val_c, "lr": optimizer.param_groups[0]["lr"]})
        print(f"seed {seed} epoch {epoch + 1}: train {history[-1]['train_loss']} val_loss {val_loss:.4f} val_c {val_c:.4f}", flush=True)

        if val_c > best:
            best, best_epoch, stale = val_c, epoch + 1, 0
            torch.save(model.state_dict(), out_dir / "best.pth")
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    (out_dir / "history.json").write_text(json.dumps({"config": asdict(cfg), "history": history}, indent=2), encoding="utf-8")
    return {"seed": seed, "best_val_c_index": best, "best_epoch": best_epoch, "history": history}


def install(run_dir: Path, chosen_seed: int) -> None:
    """Đưa checkpoint seed được chọn và rna_reference vào serving; giữ checkpoint cũ làm đối chiếu."""
    current = WEIGHTS_DIR / "best_multimodal_model.pth"
    legacy = WEIGHTS_DIR / "best_multimodal_model_legacy_0.6936.pth"
    if current.exists() and not legacy.exists():
        current.rename(legacy)
    shutil.copy2(run_dir / f"seed{chosen_seed}" / "best.pth", current)
    shutil.copy2(run_dir / "rna_reference.npz", WEIGHTS_DIR / "rna_reference.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    train_rows, val_rows = select_split(manifest, ("train",)), select_split(manifest, ("val",))
    rna_matrix = pd.read_csv(args.data / "PROCESSED_RNA" / "rna_matrix.csv", index_col=0)
    # Thống kê RNA chỉ từ bệnh nhân train (notebook cũ tính trên cả 899 ca).
    rna_ref = fit_rna_reference(rna_matrix.loc[train_rows.patient_id[train_rows.has_rna == 1]])
    args.out.mkdir(parents=True, exist_ok=True)
    save_rna_reference(rna_ref, str(args.out / "rna_reference.npz"))
    num_genes = len(rna_ref.columns)

    train_patients = load_patients(args.data, train_rows, rna_matrix, rna_ref)
    val_patients = load_patients(args.data, val_rows, rna_matrix, rna_ref)
    cfg = TrainConfig(epochs=args.epochs)
    results = [train_seed(train_patients, val_patients, num_genes, seed, cfg, args.out / f"seed{seed}") for seed in args.seeds]

    chosen = max(results, key=lambda r: r["best_val_c_index"])
    selection = {
        "chosen_seed": chosen["seed"], "chosen_epoch": chosen["best_epoch"],
        "val_c_index": {str(r["seed"]): r["best_val_c_index"] for r in results},
        "n_train": len(train_patients), "n_val": len(val_patients), "num_genes": num_genes,
    }
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2))
    if args.install:
        install(args.out, chosen["seed"])


if __name__ == "__main__":
    main()
