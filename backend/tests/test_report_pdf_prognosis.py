"""PDF báo cáo dựng được trang tiên lượng khi mô hình chưa hiệu chuẩn (chỉ percentile, không có đường sống còn)."""

import unittest

from routers.analysis import _build_professional_report_pdf

BASE = dict(
    report_id="R-1", patient_code="P-1", image_id=1, status="done", processing_date="2026-11-20 20:00",
    tumor_label="glioma", classification_confidence=0.9, bbox_confidence=0.8, bbox=[1, 2, 3, 4],
    class_probabilities=None, no_tumor_detected=False, multimodal_available=True,
    original_image=None, bbox_image=None, cropped_roi_image=None, mask_image=None, overlay_image=None,
    risk_score=1.53, risk_group="High", survival_curve_data=None,
)


class ReportPdfPrognosisTest(unittest.TestCase):
    def test_uncalibrated_prognosis_page_renders(self):
        meta = {"calibrated": False, "percentile": 95.9, "group": "High",
                "limitation_text": "Điểm nguy cơ tương đối so với 764 bệnh nhân huấn luyện – không phải xác suất sống còn đã hiệu chuẩn."}
        pdf = _build_professional_report_pdf(**BASE, prognosis_meta=meta)
        self.assertTrue(pdf.startswith(b"%PDF"))

    def test_missing_meta_still_renders(self):
        self.assertTrue(_build_professional_report_pdf(**BASE).startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
