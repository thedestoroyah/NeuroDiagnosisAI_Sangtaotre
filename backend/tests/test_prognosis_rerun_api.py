"""Chạy lại tiên lượng phải thay kết quả cũ trong task MRI, kể cả khi mô hình mới không trả đường sống còn (yêu cầu 2)."""

import datetime as dt
import os

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from routers.analysis import _merge_prognosis_only  # noqa: E402
from test_mri_safety_api import _client, _db, _image  # noqa: E402

import models  # noqa: E402

OLD = {"risk_score": 2.566, "risk_group": "Very High", "tumor_label": "Glioma",
       "survival_curve_data": [{"time": 0, "survival_probability": 1.0}, {"time": 12, "survival_probability": 0.4}]}
NEW = {"risk_score": 0.09, "risk_group": "Medium", "survival_curve_data": None,
       "prognosis_meta": {"calibrated": False, "percentile": 39.8, "limitation_text": "Điểm nguy cơ tương đối"}}


def _tasks(db, image, mri_time, prognosis_time):
    db.add(models.InferenceTask(celery_task_id="m", task_type="mri_pipeline", target_id=image.id, status="done",
                                result=dict(OLD), created_at=mri_time, updated_at=mri_time))
    db.add(models.InferenceTask(celery_task_id="p", task_type="prognosis", target_id=image.patient_id, status="done",
                                result={**NEW, "image_id": image.id}, created_at=prognosis_time, updated_at=prognosis_time))
    db.commit()


def test_merge_overrides_with_none_from_new_prognosis():
    merged = _merge_prognosis_only(OLD, NEW)
    assert merged["survival_curve_data"] is None
    assert merged["risk_group"] == "Medium"
    assert merged["prognosis_meta"]["percentile"] == 39.8
    assert merged["tumor_label"] == "Glioma"


def test_image_detail_shows_newer_prognosis_over_old_mri_payload():
    db = _db()
    owner, image = _image(db)
    _tasks(db, image, mri_time=dt.datetime(2026, 5, 1), prognosis_time=dt.datetime(2026, 11, 20))

    with _client(db, {"sub": str(owner.id), "role": "doctor"}) as client:
        body = client.get(f"/records/analysis/image/{image.id}").json()

    assert body["risk_score"] == 0.09
    assert body["risk_group"] == "Medium"
    assert not body.get("survival_curve_data")
    assert body["prognosis_meta"]["percentile"] == 39.8
    assert body["tumor_label"] == "Glioma"


def test_image_detail_keeps_mri_payload_when_mri_rerun_is_newer():
    db = _db()
    owner, image = _image(db)
    _tasks(db, image, mri_time=dt.datetime(2026, 11, 21), prognosis_time=dt.datetime(2026, 11, 20))

    with _client(db, {"sub": str(owner.id), "role": "doctor"}) as client:
        body = client.get(f"/records/analysis/image/{image.id}").json()

    assert body["risk_group"] == "Very High"
