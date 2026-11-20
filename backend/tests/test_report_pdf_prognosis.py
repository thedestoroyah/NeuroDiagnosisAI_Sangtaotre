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


# ── Yêu cầu 3: PDF không bao giờ biến MRI lỗi / âm tính chưa xác nhận thành kết luận "không có u" ──
from routers.analysis import _report_headline  # noqa: E402


def test_negative_needing_review_is_not_a_conclusion():
    title, summary = _report_headline(no_tumor_detected=True, tumor_label=None,
                                      mri_review_required=True, mri_review_reasons=["negative_needs_confirmation"])
    assert "Không phát hiện khối u" in title
    assert "chưa có kết luận" in summary.lower()
    assert "bác sĩ xác nhận" in summary


def test_missing_label_without_negative_is_not_reported_as_no_tumor():
    title, summary = _report_headline(no_tumor_detected=False, tumor_label=None,
                                      mri_review_required=False, mri_review_reasons=[])
    assert "Không phát hiện" not in title


def test_unreadable_mri_report_is_refused():
    from test_mri_safety_api import _client, _db, _image
    import models

    db = _db()
    owner, image = _image(db)
    db.add(models.InferenceTask(
        celery_task_id="c1", task_type="mri_pipeline", target_id=image.id, status="failed",
        error_message="Không đọc được ảnh MRI (0/4 lát) – chưa có kết luận, cần kiểm tra lại dữ liệu.",
        result={"mri_status": "mri_unreadable", "review_required": True, "review_reasons": ["mri_unreadable"]},
    ))
    db.commit()
    with _client(db, {"sub": str(owner.id), "role": "doctor"}) as client:
        response = client.get(f"/records/analysis/image/{image.id}/report")
    assert response.status_code == 409
    assert "Không đọc được MRI" in response.json()["detail"]
