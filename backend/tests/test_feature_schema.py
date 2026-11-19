"""Kiểm thử mã hóa lâm sàng dùng chung so với notebook huấn luyện gốc (yêu cầu 1)."""

import unittest
from pathlib import Path

import pandas as pd

from ai_core.feature_schema import CLINICAL_FIELDS, encode_clinical

FIXTURE = Path(__file__).parent / "fixtures" / "clinical_rows.csv"


def _notebook_reference(row):
    """Chép nguyên văn từ cell 2 của `Bản sao của 07_training_loop.ipynb` (BrainTumorDataset.__getitem__).

    Đây là "đường train gốc" để chứng minh mã hóa mới cho ra cùng tensor.
    """
    clinical_cols = [
        'age', 'gender', 'race', 'grade', 'prior_treatment',
        'radiation_therapy', 'pharmaceutical_therapy', 'idh_mutation', 'mgmt_methylation'
    ]

    clinical_vals = []
    clinical_flags = []

    for col in clinical_cols:
        val = row[col]
        if pd.isna(val):
            clinical_vals.append(0.0) # Zero-padding
            clinical_flags.append(0.0) # Cờ báo hiệu là Không có dữ liệu
        else:
            # Chuẩn hóa sơ bộ để đưa về khoảng [0, 1]
            if col == 'age': val = float(val) / 100.0
            elif col == 'grade': val = float(val) / 4.0
            elif col == 'race': val = float(val) / 3.0
            else: val = float(val) # Các biến còn lại vốn là 0.0 hoặc 1.0

            clinical_vals.append(val)
            clinical_flags.append(1.0) # Cờ báo hiệu Có dữ liệu

    has_clinical = 1.0 if sum(clinical_flags) > 0 else 0.0
    return clinical_vals + clinical_flags, has_clinical


class EncodeClinicalTest(unittest.TestCase):
    def test_field_order_matches_training(self):
        self.assertEqual(
            CLINICAL_FIELDS,
            ("age", "gender", "race", "grade", "prior_treatment",
             "radiation_therapy", "pharmaceutical_therapy", "idh_mutation", "mgmt_methylation"),
        )

    def test_matches_notebook_on_real_rows(self):
        rows = pd.read_csv(FIXTURE)
        self.assertEqual(len(rows), 6)
        for _, row in rows.iterrows():
            expected_vec, expected_has = _notebook_reference(row)
            vec, has = encode_clinical(row.to_dict())
            self.assertEqual(len(vec), 18)
            for i, (got, want) in enumerate(zip(vec, expected_vec)):
                self.assertAlmostEqual(got, want, places=6, msg=f"{row['patient_id']} ô {i}")
            self.assertEqual(has, expected_has)

    def test_gender_encoding(self):
        for raw in ("male", "Male", "M", 0):
            vec, _ = encode_clinical({"gender": raw})
            self.assertEqual((vec[1], vec[10]), (0.0, 1.0), raw)
        for raw in ("female", "F", 1):
            vec, _ = encode_clinical({"gender": raw})
            self.assertEqual((vec[1], vec[10]), (1.0, 1.0), raw)

    def test_grade_always_missing(self):
        vec, has = encode_clinical({"grade": "4"})
        self.assertEqual((vec[3], vec[12]), (0.0, 0.0))
        self.assertEqual(has, 0.0)

    def test_unparseable_is_missing(self):
        vec, has = encode_clinical({"prior_treatment": "unknown", "gender": " "})
        self.assertEqual((vec[4], vec[13], vec[1], vec[10]), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(has, 0.0)

    def test_string_values_from_db(self):
        vec, has = encode_clinical({"age": "65", "idh_mutation": "1", "mgmt_methylation": "no",
                                    "radiation_therapy": "1.0", "race": "2"})
        self.assertAlmostEqual(vec[0], 0.65)
        self.assertEqual((vec[7], vec[16]), (1.0, 1.0))
        self.assertEqual((vec[8], vec[17]), (0.0, 1.0))
        self.assertEqual((vec[5], vec[14]), (1.0, 1.0))
        self.assertAlmostEqual(vec[2], 2 / 3)
        self.assertEqual(has, 1.0)

    def test_empty_record(self):
        self.assertEqual(encode_clinical({}), ([0.0] * 18, 0.0))

    def test_ignores_mri_and_ki67(self):
        base = {"age": 50, "gender": "female"}
        noisy = dict(base, ki67_index=40, classification_confidence=0.9, class_probabilities=[0.1, 0.9])
        self.assertEqual(encode_clinical(base), encode_clinical(noisy))


if __name__ == "__main__":
    unittest.main()
