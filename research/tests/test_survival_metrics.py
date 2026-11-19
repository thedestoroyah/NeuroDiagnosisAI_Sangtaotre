"""Kiểm thử baseline Breslow, dự đoán sống còn, bảng calibration và bootstrap (yêu cầu 2)."""

import sys
import unittest
from pathlib import Path

import numpy as np
from lifelines import NelsonAalenFitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prognosis"))
from survival_metrics import bootstrap_ci, breslow_baseline, calibration_table, predict_survival  # noqa: E402


def _exponential_cohort(n=3000, seed=0, base_hazard=0.05):
    """Dữ liệu sinh từ mô hình Cox mũ đã biết: S(t | x) = exp(-λ0 · t · e^x), kiểm duyệt nhẹ."""
    rng = np.random.default_rng(seed)
    risk = rng.normal(0, 0.8, n)
    event_time = rng.exponential(1 / (base_hazard * np.exp(risk)))
    censor_time = rng.uniform(30, 80, n)
    time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(float)
    return risk, time, event, base_hazard


class BreslowTest(unittest.TestCase):
    def test_breslow_zero_risk_matches_nelson_aalen(self):
        rng = np.random.default_rng(3)
        time = rng.uniform(1, 50, 200)
        event = (rng.random(200) < 0.8).astype(float)
        grid, s0 = breslow_baseline(np.zeros(200), time, event)
        na = NelsonAalenFitter(nelson_aalen_smoothing=False).fit(time, event)
        expected = np.exp(-na.cumulative_hazard_at_times(grid).to_numpy())
        np.testing.assert_allclose(s0, expected, atol=1e-6)

    def test_predict_survival_monotone(self):
        risk, time, event, _ = _exponential_cohort(500)
        grid, s0 = breslow_baseline(risk, time, event)
        s_low = [predict_survival(-1.0, grid, s0, t) for t in (6, 12, 24, 36)]
        s_high = [predict_survival(1.0, grid, s0, t) for t in (6, 12, 24, 36)]
        self.assertTrue(all(a >= b for a, b in zip(s_low, s_low[1:])))
        self.assertTrue(all(lo > hi for lo, hi in zip(s_low, s_high)))
        self.assertEqual(predict_survival(0.0, grid, s0, 0.0), 1.0)

    def test_calibration_perfect_model(self):
        risk, time, event, base_hazard = _exponential_cohort()
        groups = np.digitize(risk, np.quantile(risk, [1 / 3, 2 / 3]))
        for t in (12, 24):
            pred = np.exp(-base_hazard * t * np.exp(risk))
            table = calibration_table(pred, time, event, groups, t)
            self.assertEqual(list(table.columns), ["group", "n", "predicted_mean", "km_observed", "abs_diff"])
            self.assertLess(table.abs_diff.max(), 0.05, table)


class BootstrapTest(unittest.TestCase):
    def test_bootstrap_ci_contains_point(self):
        values = np.random.default_rng(0).normal(5, 1, 300)
        lo, hi = bootstrap_ci(lambda idx: float(values[idx].mean()), size=len(values), n=500, seed=0)
        self.assertLess(lo, values.mean())
        self.assertGreater(hi, values.mean())


if __name__ == "__main__":
    unittest.main()
