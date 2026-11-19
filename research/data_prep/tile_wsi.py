"""Cắt slide WSI (.svs) thành patch 256px có mô, đúng logic notebook 02_wsi_preprocessing.

Chạy:
    python research/data_prep/tile_wsi.py --raw $DATA/WSI_Raw \
        --manifest research/data_prep/wsi_gdc_manifest.tsv --out $DATA/PROCESSED_WSI_v2 [--workers 4]
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

PATCH_SIZE = 256
TARGET_LEVEL = 1
SATURATION_THRESHOLD = 40
TISSUE_FRACTION = 0.3
_DONE_MIN_PATCHES = 10  # như notebook: thư mục có trên 10 patch thì coi là đã xử lý


def is_tissue(patch_rgb: np.ndarray, sat_threshold: int = SATURATION_THRESHOLD, min_fraction: float = TISSUE_FRACTION) -> bool:
    """Patch là mô khi tỉ lệ pixel có độ bão hòa HSV > sat_threshold đạt ít nhất min_fraction."""
    saturation = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    return np.count_nonzero(saturation > sat_threshold) / saturation.size >= min_fraction


def tile_slide(svs_path: str, out_dir: Path) -> int:
    """Quét lưới không chồng lấn ở level 1 (không có thì level 0), lưu patch có mô thành patch_<n>.png.

    Args:
        svs_path: đường dẫn slide .svs.
        out_dir: thư mục của bệnh nhân; n đánh số từ 0 theo thứ tự quét (y rồi x).

    Returns:
        Số patch đã lưu.

    Raises:
        openslide.OpenSlideError: slide không mở được.
    """
    import openslide

    out_dir.mkdir(parents=True, exist_ok=True)
    with openslide.OpenSlide(svs_path) as slide:
        level = TARGET_LEVEL if TARGET_LEVEL < slide.level_count else 0
        width, height = slide.level_dimensions[level]
        # read_region luôn nhận tọa độ level 0
        downsample = int(slide.level_downsamples[level])
        saved = 0
        for y in range(0, height, PATCH_SIZE):
            for x in range(0, width, PATCH_SIZE):
                patch = np.array(slide.read_region((x * downsample, y * downsample), level, (PATCH_SIZE, PATCH_SIZE)).convert("RGB"))
                if is_tissue(patch):
                    cv2.imwrite(str(out_dir / f"patch_{saved}.png"), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                    saved += 1
    return saved


def _process(row: dict, raw_dir: Path, out_root: Path) -> dict:
    pid, slide_name = row["patient_id"], row["file_name"]
    out_dir = out_root / pid
    if out_dir.exists() and len(list(out_dir.glob("patch_*.png"))) > _DONE_MIN_PATCHES:
        return {"patient_id": pid, "slide": slide_name, "n_patches": len(list(out_dir.glob("patch_*.png"))), "status": "skipped"}
    try:
        n = tile_slide(str(raw_dir / slide_name), out_dir)
        return {"patient_id": pid, "slide": slide_name, "n_patches": n, "status": "ok" if n else "no_tissue"}
    except Exception as exc:  # ghi lỗi từng slide vào log, không dừng cả batch
        return {"patient_id": pid, "slide": slide_name, "n_patches": 0, "status": f"error: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_process, row, args.raw, args.out) for row in rows]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(f"[{i}/{len(rows)}] {result['patient_id']}: {result['n_patches']} patch ({result['status']})", flush=True)

    with open(args.out / "tiling_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "slide", "n_patches", "status"])
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: r["patient_id"]))


if __name__ == "__main__":
    main()
