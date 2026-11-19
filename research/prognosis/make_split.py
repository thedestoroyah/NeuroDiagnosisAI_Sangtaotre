"""Tạo manifest chia train/val/test 70/15/15 theo bệnh nhân, cố định trước khi train (yêu cầu 2).

Chạy:
    python research/prognosis/make_split.py --data $DATA \
        --out research/prognosis/split_manifest.csv --summary docs/cai_tien/split_summary.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 2026
TEST_FRAC = 0.15
VAL_FRAC = 0.15
_MIN_STRATUM = 3


def _strata(df: pd.DataFrame) -> pd.Series:
    """Tầng event × study_type; tầng ít hơn 3 ca gộp về tầng chỉ theo event để vẫn chia được."""
    event = df["event"].astype(int).astype(str)
    strata = event + "_" + df["study_type"].astype(int).astype(str)
    small = strata.map(strata.value_counts()) < _MIN_STRATUM
    return strata.where(~small, event)


def make_split(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Gán mỗi bệnh nhân vào train/val/test, phân tầng theo event × study_type.

    Args:
        df: có cột patient_id (duy nhất), event, study_type.
        seed: seed cố định để chạy lại ra đúng manifest.

    Returns:
        pd.Series cùng index với df, giá trị "train" | "val" | "test".
    """
    assert df["patient_id"].is_unique, "patient_id phải duy nhất (chia theo bệnh nhân)"
    rest, test = train_test_split(df.index, test_size=TEST_FRAC, random_state=seed, stratify=_strata(df))
    rest_df = df.loc[rest]
    train, val = train_test_split(rest, test_size=VAL_FRAC / (1 - TEST_FRAC), random_state=seed, stratify=_strata(rest_df))
    split = pd.Series("train", index=df.index)
    split[val] = "val"
    split[test] = "test"
    return split


def _has_mri(mri_dir: Path, pid: str) -> int:
    # Như notebook: ưu tiên thư mục flair, không có thì t1ce.
    for sub in ("flair", "t1ce"):
        if any((mri_dir / pid / sub).glob("*.png")):
            return 1
    return 0


def build_manifest(data_dir: Path, seed: int = SEED) -> pd.DataFrame:
    """Đọc dataset_multimodal_final.csv, tính lại has_* theo dữ liệu thật đang có trên đĩa, rồi chia tập."""
    df = pd.read_csv(data_dir / "dataset_multimodal_final.csv")
    rna_ids = set(pd.read_csv(data_dir / "PROCESSED_RNA" / "rna_matrix.csv", usecols=[0]).iloc[:, 0])
    wsi_dir = data_dir / "PROCESSED_WSI_v2"
    df["has_mri"] = [_has_mri(data_dir / "PROCESSED_MRI", pid) for pid in df.patient_id]
    df["has_wsi"] = [int(any((wsi_dir / pid).glob("patch_*.png"))) for pid in df.patient_id]
    df["has_rna"] = df.patient_id.isin(rna_ids).astype(int)
    df["split"] = make_split(df, seed)
    return df[["patient_id", "split", "study_type", "event", "time", "has_mri", "has_wsi", "has_rna"]]


def summary_markdown(manifest: pd.DataFrame) -> str:
    """Bảng số ca / số event theo split × nguồn dữ liệu, kèm số ca có từng mô thức."""
    study = manifest.study_type.map({1: "TCGA-LGG", 2: "TCGA-GBM", 3: "CPTAC", 4: "UPENN", 5: "IvyGAP"})
    cases = pd.crosstab(study, manifest.split)[["train", "val", "test"]]
    events = pd.crosstab(study, manifest.split, values=manifest.event, aggfunc="sum")[["train", "val", "test"]].astype(int)
    modal = manifest.groupby("split")[["has_mri", "has_wsi", "has_rna"]].sum().loc[["train", "val", "test"]]
    lines = [
        "# Manifest chia tập (seed %d, 70/15/15 theo bệnh nhân, phân tầng event × nguồn)" % SEED,
        "",
        "Nguồn: `research/prognosis/split_manifest.csv` (commit trước khi train). Không bệnh nhân nào nằm ở hai tập.",
        "",
        "## Số ca (số event) theo nguồn",
        "",
        "| Nguồn | train | val | test |",
        "|---|---:|---:|---:|",
    ]
    for name in cases.index:
        lines.append(f"| {name} | " + " | ".join(f"{cases.loc[name, s]} ({events.loc[name, s]})" for s in ("train", "val", "test")) + " |")
    lines.append("| **Tổng** | " + " | ".join(f"{cases[s].sum()} ({events[s].sum()})" for s in ("train", "val", "test")) + " |")
    lines += ["", "## Số ca có từng mô thức", "", "| Mô thức | train | val | test |", "|---|---:|---:|---:|"]
    for col, label in (("has_mri", "MRI"), ("has_wsi", "WSI (tải lại từ GDC)"), ("has_rna", "RNA")):
        lines.append(f"| {label} | " + " | ".join(str(int(modal.loc[s, col])) for s in ("train", "val", "test")) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.data)
    manifest.to_csv(args.out, index=False)
    text = summary_markdown(manifest)
    args.summary.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
