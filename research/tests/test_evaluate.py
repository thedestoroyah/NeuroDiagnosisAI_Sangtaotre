"""Kiểm thử phần đối chứng lâm sàng của evaluate.py trên dữ liệu có cột hằng số (như idh_mutation thật chỉ có 0)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prognosis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json  # noqa: E402

from ai_core.architectures.survival_net import MultimodalBrainTumorModel  # noqa: E402
from evaluate import clinical_baseline, ensemble_predict, previous_reports  # noqa: E402
from test_train_smoke import NUM_GENES, _fake_patients  # noqa: E402
from train import predict  # noqa: E402


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



class TestLockTest(unittest.TestCase):
    def test_reports_from_every_runs_dir_count_as_previous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, when in (("runs_v2", "2026-11-21T10:00:00"), ("runs", "2026-11-20T02:36:39")):
                (root / name).mkdir()
                (root / name / "test_report.json").write_text(json.dumps({"evaluated_at": when}))
            (root / "runs" / "rehearsal").mkdir()
            (root / "runs" / "rehearsal" / "test_report.json").write_text(json.dumps({"evaluated_at": "x"}))
            self.assertEqual([r["evaluated_at"] for r in previous_reports(root)], ["2026-11-20T02:36:39", "2026-11-21T10:00:00"])


class EnsemblePredictTest(unittest.TestCase):
    def test_ensemble_is_mean_of_member_risks(self):
        patients = _fake_patients(6, 2)
        models = []
        for seed in (0, 1):
            torch.manual_seed(seed)
            models.append(MultimodalBrainTumorModel(num_genes=NUM_GENES).eval())
        expected = (predict(models[0], patients, NUM_GENES, "cpu") + predict(models[1], patients, NUM_GENES, "cpu")) / 2
        np.testing.assert_allclose(ensemble_predict(models, patients, NUM_GENES, "cpu"), expected, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
