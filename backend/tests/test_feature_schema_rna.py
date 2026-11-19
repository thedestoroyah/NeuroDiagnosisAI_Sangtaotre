"""Kiểm thử căn chỉnh và chuẩn hóa RNA, cùng các hằng số tiền xử lý ảnh (yêu cầu 1)."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ai_core.feature_schema import (
    IMAGE_SIZE,
    MRI_SLICES,
    WSI_PATCHES,
    encode_rna,
    fit_rna_reference,
    image_transform,
    load_rna_reference,
    save_rna_reference,
    uniform_indices,
)

DATA_DIR = Path(os.getenv("NEURO_DATA_DIR", Path(__file__).resolve().parents[3] / "BrainTumor_Research_2025" / "01_Datasets_Zipped"))
N_STAR = ["N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"]
GENES = ["ENSG00000000003", "ENSG00000000005", "ENSG00000000419", "ENSG00000000457", "ENSG00000000460", "ENSG00000000938"]


def _toy_matrix() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = rng.uniform(0, 12, size=(5, len(N_STAR) + len(GENES)))
    df = pd.DataFrame(data, columns=N_STAR + GENES, index=[f"P{i}" for i in range(5)])
    df.iloc[:, :4] = np.nan  # các cột N_* trống như trong rna_matrix.csv
    df.iloc[2, 6] = np.nan
    return df


class RnaEncodingTest(unittest.TestCase):
    def test_encode_matches_training_formula(self):
        train = _toy_matrix()
        ref = fit_rna_reference(train)
        # Công thức đúng như cell 16 của notebook: fillna(0) → log1p → (x - mean) / (std + 1e-8)
        logged = np.log1p(train.fillna(0.0))
        mean, std = logged.mean(), logged.std()
        for pid, row in train.iterrows():
            expected = ((np.log1p(row.fillna(0.0)) - mean) / (std + 1e-8)).to_numpy(np.float32)
            np.testing.assert_allclose(encode_rna(row, ref), expected, atol=1e-6, err_msg=pid)

    def test_alignment_by_name(self):
        train = _toy_matrix()
        ref = fit_rna_reference(train)
        row = train.iloc[1]
        aligned = encode_rna(row, ref)

        messy = row.drop(GENES[3]).rename(lambda c: f"{c}.15" if c.startswith("ENSG") else c)
        messy["ENSG99999999999"] = 7.0
        messy = messy[messy.index[::-1]]

        out = encode_rna(messy, ref)
        expected = aligned.copy()
        expected[ref.columns.index(GENES[3])] = 0.0  # gene vắng mặt = trung bình (0 sau z-score)
        np.testing.assert_allclose(out, expected, atol=1e-6)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(len(out), len(ref.columns))

    def test_accepts_plain_mapping(self):
        ref = fit_rna_reference(_toy_matrix())
        row = _toy_matrix().iloc[0]
        np.testing.assert_allclose(encode_rna(dict(row), ref), encode_rna(row, ref), atol=1e-7)

    def test_save_load_roundtrip(self):
        ref = fit_rna_reference(_toy_matrix())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rna_reference.npz")
            save_rna_reference(ref, path)
            loaded = load_rna_reference(path)
        self.assertEqual(loaded.columns, ref.columns)
        np.testing.assert_allclose(loaded.mean, ref.mean)
        np.testing.assert_allclose(loaded.std, ref.std)

    def test_n_star_columns_keep_position(self):
        matrix = DATA_DIR / "PROCESSED_RNA" / "rna_matrix.csv"
        if not matrix.exists():
            self.skipTest(f"không có {matrix}")
        head = pd.read_csv(matrix, index_col=0, nrows=3)
        ref = fit_rna_reference(head)
        self.assertEqual(ref.columns[:4], N_STAR)
        self.assertEqual(len(ref.columns), 60664)


class ImageConstantsTest(unittest.TestCase):
    def test_constants(self):
        self.assertEqual((IMAGE_SIZE, MRI_SLICES, WSI_PATCHES), (256, 6, 8))

    def test_uniform_indices(self):
        self.assertEqual(uniform_indices(0, 6), [])
        self.assertEqual(uniform_indices(3, 6), [0, 1, 2])
        self.assertEqual(uniform_indices(50, 6), [0, 9, 19, 29, 39, 49])

    def test_image_transform_output(self):
        from PIL import Image

        img = Image.new("RGB", (300, 200), (124, 116, 104))
        tensor = image_transform()(img)
        self.assertEqual(tuple(tensor.shape), (3, 256, 256))
        self.assertAlmostEqual(float(tensor[0].mean()), (124 / 255 - 0.485) / 0.229, places=3)


if __name__ == "__main__":
    unittest.main()
