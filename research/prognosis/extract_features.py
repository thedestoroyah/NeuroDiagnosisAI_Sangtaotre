"""Tính sẵn đặc trưng DenseNet121 (ImageNet, eval) cho mọi lát MRI và mọi patch WSI.

Chạy:
    python research/prognosis/extract_features.py --data $DATA --manifest research/prognosis/split_manifest.csv

Ghi ra $DATA/features/{mri,wsi}/<pid>.npy (float16 [n_ảnh, 1024]) và $DATA/features/index.json (tên file theo thứ tự).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_core.architectures.survival_net import MultimodalBrainTumorModel  # noqa: E402
from ai_core.feature_schema import image_transform  # noqa: E402
from feature_model import pooled_backbone_features  # noqa: E402


def sort_like_notebook(names: list[str]) -> list[str]:
    """Sắp xếp theo số ở hậu tố `_N` (slice_2 trước slice_10), lỗi thì sắp theo chữ như notebook."""
    try:
        return sorted(names, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    except ValueError:
        return sorted(names)


def mri_files(mri_root: Path, pid: str) -> list[Path]:
    """Ảnh MRI của bệnh nhân: thư mục flair, không có thì t1ce (như notebook)."""
    for sub in ("flair", "t1ce"):
        folder = mri_root / pid / sub
        names = [p.name for p in folder.glob("*") if p.suffix.lower() in (".png", ".jpg")] if folder.is_dir() else []
        if names:
            return [folder / n for n in sort_like_notebook(names)]
    return []


def wsi_files(wsi_root: Path, pid: str) -> list[Path]:
    folder = wsi_root / pid
    return [folder / n for n in sort_like_notebook([p.name for p in folder.glob("patch_*.png")])]


class _Images(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths, self.transform = paths, image_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            return self.transform(img.convert("RGB"))


@torch.no_grad()
def _features(encoder, paths: list[Path], device: str, batch: int, workers: int) -> np.ndarray:
    loader = DataLoader(_Images(paths), batch_size=batch, num_workers=workers)
    chunks = []
    for images in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            chunks.append(pooled_backbone_features(encoder, images.to(device)).float().cpu())
    return torch.cat(chunks).numpy().astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = pd.read_csv(args.manifest)
    model = MultimodalBrainTumorModel(num_genes=None).to(device).eval()
    out = args.data / "features"
    index = {"mri": {}, "wsi": {}}
    jobs = [
        ("mri", model.mri_encoder, manifest.patient_id[manifest.has_mri == 1], lambda pid: mri_files(args.data / "PROCESSED_MRI", pid)),
        ("wsi", model.wsi_encoder, manifest.patient_id[manifest.has_wsi == 1], lambda pid: wsi_files(args.data / "PROCESSED_WSI_v2", pid)),
    ]
    for kind, encoder, pids, list_files in jobs:
        (out / kind).mkdir(parents=True, exist_ok=True)
        for i, pid in enumerate(pids, 1):
            paths = list_files(pid)
            index[kind][pid] = [p.name for p in paths]
            target = out / kind / f"{pid}.npy"
            if not paths or (target.exists() and len(np.load(target, mmap_mode="r")) == len(paths)):
                continue
            np.save(target, _features(encoder, paths, device, args.batch, args.workers))
            print(f"[{kind} {i}/{len(pids)}] {pid}: {len(paths)} ảnh", flush=True)
    (out / "index.json").write_text(json.dumps(index), encoding="utf-8")


if __name__ == "__main__":
    main()
