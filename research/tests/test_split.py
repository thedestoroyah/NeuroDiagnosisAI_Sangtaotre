"""Kiểm thử chia train/val/test theo bệnh nhân, phân tầng event × study_type (yêu cầu 2)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prognosis"))
from make_split import make_split  # noqa: E402


def _toy(n=200, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "patient_id": [f"P{i:03d}" for i in range(n)],
        "event": rng.choice([0.0, 1.0], n, p=[0.15, 0.85]),
        "study_type": rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], n, p=[0.15, 0.15, 0.05, 0.6, 0.05]),
    })


class SplitTest(unittest.TestCase):
    def test_no_overlap_and_ratios(self):
        df = _toy()
        split = make_split(df)
        groups = {name: set(df.patient_id[split == name]) for name in ("train", "val", "test")}
        self.assertEqual(set(split.unique()), {"train", "val", "test"})
        self.assertFalse(groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"])
        self.assertEqual(sum(len(g) for g in groups.values()), len(df))
        for name, frac in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
            self.assertLessEqual(abs(len(groups[name]) - frac * len(df)), 2, name)

    def test_deterministic(self):
        df = _toy()
        pd.testing.assert_series_equal(make_split(df), make_split(df))

    def test_each_study_in_test(self):
        df = _toy()
        split = make_split(df)
        counts = df.study_type.value_counts()
        for study in counts[counts >= 3].index:
            self.assertGreater(((split == "test") & (df.study_type == study)).sum(), 0, study)

    def test_tiny_strata_do_not_crash(self):
        df = _toy(60)
        df.loc[0, "study_type"] = 9.0  # tầng chỉ có 1 ca
        split = make_split(df)
        self.assertEqual(len(split), 60)


if __name__ == "__main__":
    unittest.main()
