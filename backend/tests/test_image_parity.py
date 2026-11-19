"""Đầu vào ảnh của mô hình tiên lượng lúc serving phải giống lúc train: 256px, lát nguyên, lấy đều (yêu cầu 1)."""

import io
import os
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ai_core.feature_schema import image_transform
from ai_core.pipeline import TumorAnalysisPipeline

DATA_DIR = Path(os.getenv("NEURO_DATA_DIR", Path(__file__).resolve().parents[3] / "BrainTumor_Research_2025" / "01_Datasets_Zipped"))


def _pipeline():
    pipeline = object.__new__(TumorAnalysisPipeline)
    pipeline.device = "cpu"
    return pipeline


def _marked_slice(index: int) -> np.ndarray:
    """Ảnh BGR có giá trị pixel bằng chỉ số lát, để kiểm tra lát nào được chọn."""
    return np.full((64, 64, 3), index, dtype=np.uint8)


class SliceSamplingTest(unittest.TestCase):
    def test_slice_tensor_shape_and_uniform(self):
        tensor = _pipeline().preprocess_slices_for_multimodal([_marked_slice(i) for i in range(50)])
        self.assertEqual(tuple(tensor.shape), (1, 6, 3, 256, 256))
        # Khử chuẩn hóa kênh R để đọc lại chỉ số lát đã chọn
        picked = [round(float(tensor[0, s, 0].mean() * 0.229 + 0.485) * 255) for s in range(6)]
        self.assertEqual(picked, [0, 9, 19, 29, 39, 49])

    def test_few_slices_keep_all(self):
        tensor = _pipeline().preprocess_slices_for_multimodal([_marked_slice(i) for i in range(3)])
        self.assertEqual(tuple(tensor.shape), (1, 3, 3, 256, 256))

    def test_same_pixels_as_training_transform(self):
        pngs = sorted((DATA_DIR / "PROCESSED_MRI").glob("*/flair/*.png"))[:1]
        if not pngs:
            self.skipTest("không có PROCESSED_MRI")
        served = _pipeline().preprocess_slices_for_multimodal([cv2.imread(str(pngs[0]))])[0, 0]
        trained = image_transform()(Image.open(pngs[0]).convert("RGB"))
        torch.testing.assert_close(served, trained, atol=1e-5, rtol=0)

    def test_wsi_tiles_capped_at_8(self):
        tiles = []
        for i in range(20):
            buffer = io.BytesIO()
            Image.new("RGB", (256, 256), (i, i, i)).save(buffer, format="PNG")
            tiles.append(buffer.getvalue())
        tensor = _pipeline().preprocess_tiles_for_multimodal(tiles)
        self.assertEqual(tuple(tensor.shape), (1, 8, 3, 256, 256))


if __name__ == "__main__":
    unittest.main()
