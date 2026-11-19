"""Kiểm thử phần đối chứng lâm sàng của evaluate.py trên dữ liệu có cột hằng số (như idh_mutation thật chỉ có 0)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prognosis"))
from evaluate import clinical_baseline  # noqa: E402


class ClinicalBaselineTest(unittest.TestCase):
    def test_constant_column_does_not_break_fit(self):
        rng = np.random.default_rng(0)
        n = 300
        age = rng.uniform(20, 80, n)
        df = pd.DataFrame({
            "patient_id": [f"P{i}" for i in range(n)],
            "age": age,
            "gender": rng.choice([0.0, 1.0], n),
            "idh_mutation": np.where(rng.random(n) < 0.5, 0.0, np.nan),  # chỉ có 0 hoặc thiếu
            "mgmt_methylation": np.where(rng.random(n) < 0.3, rng.choice([0.0, 1.0], n), np.nan),
            "time": rng.exponential(30 / np.exp((age - 50) / 20)),
            "event": (rng.random(n) < 0.85).astype(float),
        })
        with tempfile.TemporaryDirectory() as tmp:
            df.to_csv(Path(tmp) / "dataset_multimodal_final.csv", index=False)
            c_index = clinical_baseline(Path(tmp), df.patient_id[:200], df.patient_id[200:])
        self.assertGreater(c_index, 0.55)


if __name__ == "__main__":
    unittest.main()
