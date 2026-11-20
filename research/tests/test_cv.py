"""Kiểm thử các hàm của cv.py (chọn gen, chia fold, C-index đơn/ensemble) và train_fixed_epochs của train.py."""

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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_core.architectures.survival_net import MultimodalBrainTumorModel  # noqa: E402
from cv import curve_c_index, decide, make_folds, select_top_variance_genes  # noqa: E402
from test_train_smoke import NUM_GENES, _fake_patients  # noqa: E402
from train import TrainConfig, train_fixed_epochs  # noqa: E402


class SelectGenesTest(unittest.TestCase):
    def test_keeps_top_variance_ensembl_columns_in_file_order(self):
        rng = np.random.default_rng(0)
        matrix = pd.DataFrame({
            "N_unmapped": rng.uniform(0, 1000, 20),   # phương sai lớn nhưng không phải gen → bỏ
            "ENSG01.1": rng.normal(5, 0.1, 20),
            "ENSG02.3": rng.normal(5, 3.0, 20),
            "ENSG03.2": rng.normal(5, 2.0, 20),
        }).clip(lower=0)
        self.assertEqual(select_top_variance_genes(matrix, 2), ["ENSG02.3", "ENSG03.2"])

    def test_none_keeps_every_column(self):
        matrix = pd.DataFrame({"N_x": [1.0, 2.0], "ENSG01": [0.0, 3.0]})
        self.assertEqual(select_top_variance_genes(matrix, None), ["N_x", "ENSG01"])


class FoldsTest(unittest.TestCase):
    def test_folds_partition_rows_and_balance_strata(self):
        rows = pd.DataFrame({"study_type": [1.0] * 50 + [2.0] * 50, "event": [1.0, 0.0] * 50})
        folds = make_folds(rows, n_folds=5, seed=0)
        self.assertEqual(sorted(np.concatenate(folds).tolist()), list(range(100)))
        for fold in folds:
            self.assertEqual(len(fold), 20)
            self.assertEqual(int((rows.study_type.iloc[fold] == 1.0).sum()), 10)


class CurveCIndexTest(unittest.TestCase):
    def test_single_is_mean_of_seeds_and_ensemble_averages_risk(self):
        times = np.array([1.0, 2.0, 3.0, 4.0])
        events = np.ones(4)
        perfect = -times  # risk cao chết sớm → C-index 1
        preds = np.stack([np.stack([perfect, -perfect]), np.stack([perfect, perfect])])  # [seed=2, epoch=2, n=4]
        single, ensemble = curve_c_index(preds, times, events)
        np.testing.assert_allclose(single, [1.0, 0.5])
        np.testing.assert_allclose(ensemble, [1.0, 0.5])  # epoch 2: trung bình risk = 0 → hòa → 0.5


class DecideTest(unittest.TestCase):
    def test_rule_from_ke_hoach_v2(self):
        results = {
            "all": {"single_best": 0.62, "ensemble_best": 0.63},
            "5000": {"single_best": 0.62, "ensemble_best": 0.641},
            "1000": {"single_best": 0.62, "ensemble_best": 0.637},
        }
        decision = decide(results, gene_counts={"all": 60664, "5000": 5000, "1000": 1000})
        self.assertEqual(decision["candidate"], "1000")  # chênh < 0.005 so với 5000 → lấy ít gen hơn
        self.assertTrue(decision["deploy_v2"])          # 0.637 - 0.62 ≥ 0.01

    def test_small_gain_keeps_v1(self):
        results = {"all": {"single_best": 0.62, "ensemble_best": 0.625}}
        self.assertFalse(decide(results, gene_counts={"all": 60664})["deploy_v2"])


class TrainFixedEpochsTest(unittest.TestCase):
    def test_returns_per_epoch_predictions_and_loadable_checkpoint(self):
        cfg = TrainConfig(batch_size=8)
        eval_patients = _fake_patients(10, 1)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "final.pth"
            preds = train_fixed_epochs(_fake_patients(30, 0), NUM_GENES, seed=0, cfg=cfg, epochs=3,
                                       eval_patients=eval_patients, out_path=out)
            self.assertEqual(preds.shape, (3, 10))
            model = MultimodalBrainTumorModel(num_genes=NUM_GENES)
            model.load_state_dict(torch.load(out, map_location="cpu"), strict=True)


if __name__ == "__main__":
    unittest.main()
