"""Kiểm thử cắt patch WSI theo đúng logic notebook 02_wsi_preprocessing."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
from tile_wsi import is_tissue, tile_slide  # noqa: E402

DATA_DIR = Path(os.getenv("NEURO_DATA_DIR", Path(__file__).resolve().parents[3] / "BrainTumor_Research_2025" / "01_Datasets_Zipped"))
SMALL_SLIDE = "TCGA-06-0130-01Z-00-DX1.0391b65f-4e1d-4444-abb0-e5804606d461.svs"


class TissueTest(unittest.TestCase):
    def test_is_tissue(self):
        white = np.full((256, 256, 3), 245, dtype=np.uint8)
        purple = np.zeros((256, 256, 3), dtype=np.uint8)
        purple[:] = (150, 60, 160)  # màu H&E bão hòa
        half = white.copy()
        half[:, :64] = purple[:, :64]  # 25% mô < ngưỡng 30%
        self.assertFalse(is_tissue(white))
        self.assertTrue(is_tissue(purple))
        self.assertFalse(is_tissue(half))


class TileSlideTest(unittest.TestCase):
    def test_tiles_real_slide(self):
        slide = DATA_DIR / "WSI_Raw" / SMALL_SLIDE
        if not slide.exists():
            self.skipTest(f"không có {slide}")
        with tempfile.TemporaryDirectory() as tmp:
            n = tile_slide(str(slide), Path(tmp) / "TCGA-06-0130")
            files = sorted(Path(tmp, "TCGA-06-0130").glob("patch_*.png"))
            self.assertGreater(n, 0)
            self.assertEqual(n, len(files))
            names = {f.name for f in files}
            self.assertEqual(names, {f"patch_{i}.png" for i in range(n)})
            for f in files[:20]:
                with Image.open(f) as im:
                    self.assertEqual((im.size, im.mode), ((256, 256), "RGB"))


if __name__ == "__main__":
    unittest.main()
