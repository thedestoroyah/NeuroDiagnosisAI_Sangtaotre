"""Hiển thị tiên lượng trung thực: chỉ có đường sống còn khi đã hiệu chuẩn, còn lại là điểm tương đối (yêu cầu 2)."""

import json
import tempfile
import unittest
from pathlib import Path

from ai_core.risk_reference import RiskReference, missing_reference_meta


def _reference(calibrated: bool) -> RiskReference:
    payload = {
        "schema_version": "clinical18-v1",
        "train_risk_sorted": [float(x) for x in range(-50, 50)],  # 100 bệnh nhân train, risk -50..49
        "tertile_cuts": [-17.0, 16.0],
        "s0": {"t": [1.0, 6.0, 12.0, 24.0, 36.0], "s": [0.98, 0.9, 0.75, 0.5, 0.3]},
        "calibrated": calibrated,
        "test_metrics": {"c_index": 0.66, "c_index_ci": [0.6, 0.72], "ibs": 0.18, "ibs_km": 0.2},
        "n_train": 100, "n_test": 30,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "risk_reference.json"
        path.write_text(json.dumps(payload))
        return RiskReference.load(str(path))


class RiskReferenceTest(unittest.TestCase):
    def test_percentile_clamped(self):
        ref = _reference(False)
        self.assertEqual(ref.percentile(1e6), 100.0)
        self.assertEqual(ref.percentile(-1e6), 0.0)
        self.assertAlmostEqual(ref.percentile(0.0), 51.0)

    def test_group_by_tertiles(self):
        ref = _reference(False)
        self.assertEqual([ref.group(v) for v in (-30.0, 0.0, 30.0)], ["Low", "Medium", "High"])

    def test_curve_hidden_when_not_calibrated(self):
        ref = _reference(False)
        self.assertIsNone(ref.survival_curve(0.0))
        meta = ref.meta(0.0)
        self.assertFalse(meta["calibrated"])
        self.assertEqual(meta["limitation_text"],
                         "Điểm nguy cơ tương đối so với 100 bệnh nhân huấn luyện – không phải xác suất sống còn đã hiệu chuẩn. "
                         "Khả năng xếp hạng trên 30 bệnh nhân độc lập: C-index 0.66 (KTC95% 0.60–0.72). "
                         "Chỉ dùng tham khảo, bác sĩ quyết định.")

    def test_curve_shown_when_calibrated(self):
        ref = _reference(True)
        curve = ref.survival_curve(0.0)
        self.assertEqual([p["time"] for p in curve], [0, 6, 12, 18, 24, 30, 36])
        probs = [p["survival_probability"] for p in curve]
        self.assertEqual(probs[0], 1.0)
        self.assertEqual(probs, sorted(probs, reverse=True))
        self.assertAlmostEqual(probs[2], 0.75, places=3)
        self.assertTrue(ref.meta(0.0)["limitation_text"].startswith("Xác suất ước tính từ 100 bệnh nhân huấn luyện, kiểm chứng trên 30"))

    def test_missing_reference_meta(self):
        meta = missing_reference_meta()
        self.assertFalse(meta["calibrated"])
        self.assertIsNone(meta["percentile"])


if __name__ == "__main__":
    unittest.main()


class PipelinePresentationTest(unittest.TestCase):
    def _pipeline(self, reference):
        from ai_core.pipeline import TumorAnalysisPipeline

        pipeline = object.__new__(TumorAnalysisPipeline)
        pipeline.risk_reference = reference
        return pipeline

    def test_missing_reference_file(self):
        shown = self._pipeline(None).present_risk(0.42)
        self.assertIsNone(shown["survival_curve_data"])
        self.assertIsNone(shown["risk_group"])
        self.assertFalse(shown["prognosis_meta"]["calibrated"])

    def test_uncalibrated_reference_hides_curve(self):
        shown = self._pipeline(_reference(False)).present_risk(30.0)
        self.assertIsNone(shown["survival_curve_data"])
        self.assertEqual(shown["risk_group"], "High")
        self.assertEqual(shown["prognosis_meta"]["percentile"], 81.0)

    def test_narrative_is_relative_not_absolute(self):
        pipeline = self._pipeline(_reference(False))
        result = {"risk_score": 30.0, **pipeline.present_risk(30.0)}
        text = pipeline._generate_xai_narrative(result, 0.0, 0.0, 0.0, 1.0, {"age": 60}, 0)
        self.assertNotIn("tiên lượng sinh tồn rất ngắn", text)
        self.assertIn("81", text)
        self.assertIn("không phải xác suất sống còn đã hiệu chuẩn", text)
