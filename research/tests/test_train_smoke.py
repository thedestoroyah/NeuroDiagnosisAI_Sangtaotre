"""Smoke test train.py trên dữ liệu giả: chạy được, checkpoint nạp strict vào mô hình serving, không đụng tập test."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "research" / "prognosis"))
from ai_core.architectures.survival_net import MultimodalBrainTumorModel  # noqa: E402
from train import Patient, TrainConfig, select_split, train_seed  # noqa: E402

NUM_GENES = 40


def _fake_patients(n, seed):
    rng = np.random.default_rng(seed)
    patients = []
    for i in range(n):
        has_mri, has_wsi, has_rna = i % 5 != 0, i % 3 == 0, i % 2 == 0
        patients.append(Patient(
            pid=f"P{seed}-{i}", time=float(rng.uniform(1, 40)), event=float(rng.random() < 0.8), study=float(1 + i % 3),
            mri=rng.normal(size=(int(rng.integers(3, 12)), 1024)).astype(np.float16) if has_mri else None,
            wsi=rng.normal(size=(int(rng.integers(2, 20)), 1024)).astype(np.float16) if has_wsi else None,
            rna=rng.normal(size=NUM_GENES).astype(np.float32) if has_rna else None,
            clinical=list(rng.random(18)), has_clinical=1.0,
        ))
    return patients


class TrainSmokeTest(unittest.TestCase):
    def test_smoke_two_epochs(self):
        cfg = TrainConfig(epochs=2, batch_size=8, patience=5)
        with tempfile.TemporaryDirectory() as tmp:
            result = train_seed(_fake_patients(30, 0), _fake_patients(10, 1), NUM_GENES, seed=0, cfg=cfg, out_dir=Path(tmp))
            self.assertEqual(len(result["history"]), 2)
            self.assertTrue(0.0 <= result["best_val_c_index"] <= 1.0)
            state = torch.load(Path(tmp) / "best.pth", map_location="cpu")
            model = MultimodalBrainTumorModel(num_genes=NUM_GENES)
            model.load_state_dict(state, strict=True)

    def test_select_split_refuses_test_rows(self):
        manifest = pd.DataFrame({"patient_id": ["a", "b", "c"], "split": ["train", "val", "test"]})
        self.assertEqual(list(select_split(manifest, ("train",)).patient_id), ["a"])
        with self.assertRaises(ValueError):
            select_split(manifest, ("train", "test"))


if __name__ == "__main__":
    unittest.main()
