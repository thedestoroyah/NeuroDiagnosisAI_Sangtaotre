"""Chọn cấu hình mô hình tiên lượng v2 bằng cross-validation trên train+val, theo docs/cai_tien/ke_hoach_v2.md.

Chạy:
    python research/prognosis/cv.py --data $DATA --manifest research/prognosis/split_manifest.csv --out research/prognosis/runs_v2
    python research/prognosis/cv.py --data $DATA --manifest ... --out research/prognosis/runs_v2 --final [--install]

Bước CV ghi out/cv_results.json. --final đọc file đó, train 5 seed của cấu hình được chọn trên toàn bộ train+val
và ghi out/selection.json, out/rna_reference.npz, out/seed*/final.pth; --install gộp vào weights serving.
Không bước nào đọc tập test.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import WEIGHTS_DIR, Patient, TrainConfig, load_patients, select_split, train_fixed_epochs  # noqa: E402

from ai_core.architectures.survival_net import c_index_score  # noqa: E402
from ai_core.feature_schema import encode_rna, fit_rna_reference, save_rna_reference  # noqa: E402

GENE_CONFIGS: dict[str, int | None] = {"all": None, "5000": 5000, "1000": 1000}
SEEDS = (0, 1, 2, 3, 4)
N_FOLDS = 5
FOLD_SEED = 0
MAX_EPOCHS = 30
MIN_GAIN = 0.01
TIE_MARGIN = 0.005
TRAIN_SPLITS = ("train", "val")


def select_top_variance_genes(matrix: pd.DataFrame, k: int | None) -> list[str]:
    """Chọn k cột gen Ensembl (tên bắt đầu bằng ENSG) có phương sai log1p lớn nhất, giữ thứ tự cột trong file.

    Args:
        matrix: các dòng rna_matrix.csv của phần train (không dùng nhãn sống còn).
        k: số gen giữ lại; None = giữ mọi cột như v1 (kể cả 4 cột N_*).

    Returns:
        list tên cột.
    """
    if k is None:
        return list(matrix.columns)
    genes = [c for c in matrix.columns if str(c).startswith("ENSG")]
    variance = pd.Series(np.log1p(matrix[genes].fillna(0.0).to_numpy(np.float64)).var(axis=0, ddof=1), index=genes)
    keep = set(variance.nlargest(k).index)
    return [c for c in genes if c in keep]


def make_folds(rows: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = FOLD_SEED) -> list[np.ndarray]:
    """Chia k-fold phân tầng theo (study_type, event). Trả list chỉ số vị trí của phần giữ lại ở mỗi fold."""
    strata = rows.study_type.astype(str) + "_" + rows.event.astype(str)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [held for _, held in splitter.split(np.zeros(len(rows)), strata)]


def curve_c_index(preds: np.ndarray, times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """C-index theo epoch của một fold.

    Args:
        preds: [seed, epoch, n] risk trên phần giữ lại.

    Returns:
        (single [epoch]: trung bình C-index của từng seed, ensemble [epoch]: C-index của risk trung bình các seed).
    """
    single = np.array([[c_index_score(p, times, events) for p in seed_preds] for seed_preds in preds]).mean(axis=0)
    ensemble = np.array([c_index_score(p, times, events) for p in preds.mean(axis=0)])
    return single, ensemble


def decide(results: dict, gene_counts: dict[str, int]) -> dict:
    """Áp luật quyết định của ke_hoach_v2.md.

    Args:
        results: {cấu hình: {"single_best", "ensemble_best", ...}}, phải có "all".
        gene_counts: {cấu hình: số cột RNA}.

    Returns:
        dict r0, candidate, candidate_cv, gain, deploy_v2.
    """
    r0 = results["all"]["single_best"]
    ranked = sorted(results, key=lambda c: results[c]["ensemble_best"], reverse=True)
    top = results[ranked[0]]["ensemble_best"]
    # Trong nhóm sát cấu hình tốt nhất (< TIE_MARGIN) thì lấy cấu hình ít gen nhất.
    close = [c for c in ranked if top - results[c]["ensemble_best"] < TIE_MARGIN]
    candidate = min(close, key=lambda c: gene_counts[c])
    gain = results[candidate]["ensemble_best"] - r0
    return {"r0": r0, "candidate": candidate, "candidate_cv": results[candidate]["ensemble_best"],
            "gain": gain, "deploy_v2": bool(gain >= MIN_GAIN - 1e-12)}


def with_rna(patients: list[Patient], rna_matrix: pd.DataFrame, rna_ids: set[str], columns: list[str], fit_ids):
    """Mã hóa lại RNA cho danh sách bệnh nhân bằng thống kê fit trên `fit_ids`; trả (patients mới, RnaReference)."""
    ref = fit_rna_reference(rna_matrix.loc[list(fit_ids), columns])
    sub = rna_matrix[columns]
    encoded = [replace(p, rna=encode_rna(sub.loc[p.pid], ref)) if p.pid in rna_ids else p for p in patients]
    return encoded, ref


def run_cv(data: Path, rows: pd.DataFrame, patients: list[Patient], rna_matrix: pd.DataFrame, out: Path) -> dict:
    """Chạy toàn bộ lưới cấu hình × fold × seed; ghi và trả nội dung cv_results.json."""
    from evaluate import clinical_baseline  # import muộn: evaluate kéo theo matplotlib/sksurv

    rna_ids = set(rows.patient_id[rows.has_rna == 1])
    times, events = np.array([p.time for p in patients]), np.array([p.event for p in patients])
    folds = make_folds(rows)
    cfg = TrainConfig()
    results, gene_counts = {}, {}
    for name, k in GENE_CONFIGS.items():
        single_folds, ensemble_folds = [], []
        for f, held in enumerate(folds):
            fit = np.setdiff1d(np.arange(len(rows)), held)
            fit_rna = [patients[i].pid for i in fit if patients[i].pid in rna_ids]
            columns = select_top_variance_genes(rna_matrix.loc[fit_rna], k)
            gene_counts[name] = len(columns)
            encoded, _ = with_rna(patients, rna_matrix, rna_ids, columns, fit_rna)
            train_p, held_p = [encoded[i] for i in fit], [encoded[i] for i in held]
            preds = np.stack([train_fixed_epochs(train_p, len(columns), s, cfg, MAX_EPOCHS, eval_patients=held_p) for s in SEEDS])
            single, ensemble = curve_c_index(preds, times[held], events[held])
            single_folds.append(single.tolist())
            ensemble_folds.append(ensemble.tolist())
            print(f"[cv] {name} fold {f}: single max {single.max():.4f} @ {single.argmax() + 1}, "
                  f"ensemble max {ensemble.max():.4f} @ {ensemble.argmax() + 1}", flush=True)
        single_mean, ensemble_mean = np.mean(single_folds, axis=0), np.mean(ensemble_folds, axis=0)
        results[name] = {
            "num_genes": gene_counts[name],
            "single_curve": single_mean.tolist(), "ensemble_curve": ensemble_mean.tolist(),
            "single_best": float(single_mean.max()), "single_epoch": int(single_mean.argmax() + 1),
            "ensemble_best": float(ensemble_mean.max()), "ensemble_epoch": int(ensemble_mean.argmax() + 1),
            "single_per_fold": single_folds, "ensemble_per_fold": ensemble_folds,
        }

    ids = rows.patient_id.to_numpy()
    clinical = [clinical_baseline(data, pd.Series(np.delete(ids, held)), pd.Series(ids[held])) for held in folds]
    report = {
        "protocol": "docs/cai_tien/ke_hoach_v2.md", "n": len(rows), "n_folds": N_FOLDS, "seeds": list(SEEDS),
        "max_epochs": MAX_EPOCHS, "configs": results,
        "clinical_cox_cv": {"mean": float(np.mean(clinical)), "per_fold": clinical},
        "decision": decide(results, gene_counts),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "cv_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def train_final(rows: pd.DataFrame, patients: list[Patient], rna_matrix: pd.DataFrame, out: Path) -> dict:
    """Train 5 seed của cấu hình được chọn trên toàn bộ train+val với số epoch chọn bằng CV; ghi selection.json.

    Raises:
        SystemExit: nếu cv_results.json kết luận giữ v1.
    """
    cv = json.loads((out / "cv_results.json").read_text())
    decision = cv["decision"]
    if not decision["deploy_v2"]:
        raise SystemExit(f"CV kết luận giữ v1 (gain {decision['gain']:.4f} < {MIN_GAIN}); không train bản cuối.")
    name = decision["candidate"]
    epochs = cv["configs"][name]["ensemble_epoch"]
    rna_ids = set(rows.patient_id[rows.has_rna == 1])
    fit_rna = sorted(rna_ids)
    columns = select_top_variance_genes(rna_matrix.loc[fit_rna], GENE_CONFIGS[name])
    encoded, ref = with_rna(patients, rna_matrix, rna_ids, columns, fit_rna)
    save_rna_reference(ref, str(out / "rna_reference.npz"))
    for seed in SEEDS:
        train_fixed_epochs(encoded, len(columns), seed, TrainConfig(), epochs, out_path=out / f"seed{seed}" / "final.pth")
        print(f"[final] seed {seed} xong", flush=True)
    selection = {
        "version": 2, "gene_config": name, "num_genes": len(columns), "epochs": epochs,
        "ensemble_seeds": list(SEEDS), "checkpoint": "final.pth", "train_splits": list(TRAIN_SPLITS),
        "n_train": len(rows), "cv_c_index": decision["candidate_cv"], "cv_r0": decision["r0"],
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return selection


def install(out: Path) -> None:
    """Gộp 5 checkpoint thành backend/ai_core/weights/prognosis_ensemble.pth (backbone một lần) và chép rna_reference."""
    import torch

    from ai_core.architectures.ensemble import save_ensemble
    from ai_core.pipeline import ENSEMBLE_WEIGHTS

    selection = json.loads((out / "selection.json").read_text())
    states = [torch.load(out / f"seed{s}" / selection["checkpoint"], map_location="cpu", weights_only=True)
              for s in selection["ensemble_seeds"]]
    save_ensemble(states, WEIGHTS_DIR / ENSEMBLE_WEIGHTS, meta=selection)
    shutil.copy2(out / "rna_reference.npz", WEIGHTS_DIR / "rna_reference.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()

    rows = select_split(pd.read_csv(args.manifest), TRAIN_SPLITS)
    rna_matrix = pd.read_csv(args.data / "PROCESSED_RNA" / "rna_matrix.csv", index_col=0)
    patients = load_patients(args.data, rows, rna_matrix, rna_ref=None)
    if args.final:
        print(json.dumps(train_final(rows, patients, rna_matrix, args.out), indent=2))
        if args.install:
            install(args.out)
    else:
        report = run_cv(args.data, rows, patients, rna_matrix, args.out)
        print(json.dumps({k: v for k, v in report.items() if k != "configs"}, indent=2))


if __name__ == "__main__":
    main()
