"""MRI lỗi không bao giờ được báo là 'không phát hiện u'; kết quả không chắc phải chờ bác sĩ (yêu cầu 3)."""

import os
import tempfile
import unittest

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import models
from ai_core.pipeline import TumorAnalysisPipeline
from task import persist_mri_failure

GARBAGE = b"\x00garbage"


class _NoTumorDetector:
    def predict(self, image):
        return None, image, None


def _pipeline(readable=None):
    """Pipeline giả: ảnh đọc được khi bytes != GARBAGE (hoặc theo hàm `readable`), detector không thấy u."""
    pipeline = object.__new__(TumorAnalysisPipeline)
    pipeline.device = "cpu"
    pipeline.detector = _NoTumorDetector()
    is_readable = readable or (lambda b: b != GARBAGE)

    def load_image(source):
        if not is_readable(source):
            raise ValueError("Không giải mã được ảnh")
        return np.zeros((64, 64, 3), dtype=np.uint8)

    pipeline.load_image = load_image
    return pipeline


class SeriesSafetyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, slices, pipeline=None):
        return (pipeline or _pipeline()).run_series_inference(image_bytes_list=slices, output_dir=self.tmp.name)

    def test_all_slices_fail_is_error_not_negative(self):
        result = self._run([GARBAGE] * 5)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "mri_unreadable")
        self.assertEqual(result["mri_status"], "mri_unreadable")
        self.assertNotIn("no_tumor_detected", result)
        self.assertEqual((result["slices_total"], result["slices_failed"], result["slices_examined"]), (5, 5, 0))
        self.assertTrue(result["review_required"])
        self.assertIn("mri_unreadable", result["review_reasons"])
        self.assertTrue(result["error_msg"].startswith("Không đọc được ảnh MRI (0/5 lát)"))
        self.assertTrue(result["failed_slice_reasons"])

    def test_empty_series_is_unreadable(self):
        result = self._run([])
        self.assertEqual(result["mri_status"], "mri_unreadable")
        self.assertEqual(result["status"], "error")
        self.assertNotIn("no_tumor_detected", result)

    def test_partial_failure_flags_review(self):
        result = self._run([GARBAGE, b"ok1", GARBAGE, b"ok2", b"ok3"])
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["mri_status"], "no_tumor_detected")
        self.assertTrue(result["no_tumor_detected"])
        self.assertTrue(result["mri_partial"])
        self.assertEqual((result["slices_examined"], result["slices_failed"]), (3, 2))
        self.assertIn("slices_failed", result["review_reasons"])
        self.assertIn("negative_needs_confirmation", result["review_reasons"])

    def test_clean_negative_still_needs_review(self):
        result = self._run([b"ok"] * 5)
        self.assertEqual(result["mri_status"], "no_tumor_detected")
        self.assertFalse(result["mri_partial"])
        self.assertTrue(result["review_required"])
        self.assertEqual(result["review_reasons"], ["negative_needs_confirmation"])

    def test_no_silent_step_two(self):
        result = self._run([b"ok"] * 30)
        self.assertEqual((result["slices_total"], result["slices_examined"]), (30, 30))


class SingleImageSafetyTest(unittest.TestCase):
    def test_unreadable_single_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _pipeline()._run_mri_core(GARBAGE, tmp)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["mri_status"], "mri_unreadable")
        self.assertNotIn("no_tumor_detected", result)
        self.assertEqual((result["slices_total"], result["slices_failed"]), (1, 1))

    def test_single_negative_needs_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _pipeline()._run_mri_core(b"ok", tmp)
        self.assertEqual(result["mri_status"], "no_tumor_detected")
        self.assertTrue(result["review_required"])


class PersistFailureTest(unittest.TestCase):
    def test_task_constant_matches_pipeline(self):
        import task
        from ai_core.pipeline import MRI_UNREADABLE

        self.assertEqual(task.MRI_UNREADABLE, MRI_UNREADABLE)

    def test_task_marks_failed_with_code(self):
        engine = create_engine("sqlite+pysqlite:///:memory:")
        models.Base.metadata.create_all(engine)
        with Session(engine) as db:
            task_record = models.InferenceTask(celery_task_id="t1", task_type="mri_pipeline", target_id=1, status="processing")
            db.add(task_record)
            db.commit()
            with tempfile.TemporaryDirectory() as tmp:
                result = _pipeline().run_series_inference(image_bytes_list=[GARBAGE] * 3, output_dir=tmp)
            persist_mri_failure(db, task_record, result)
            db.refresh(task_record)
            self.assertEqual(task_record.status, "failed")
            self.assertTrue(task_record.error_message.startswith("Không đọc được ảnh MRI"))
            self.assertEqual(task_record.result["mri_status"], "mri_unreadable")
            self.assertEqual(task_record.result["slices_failed"], 3)


if __name__ == "__main__":
    unittest.main()
