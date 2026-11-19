"""API chi tiết ảnh phải trả trạng thái MRI không đọc được tách biệt với kết quả âm tính (yêu cầu 3)."""

import os

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import models
from database import Base, get_db
from routers.analysis import router as analysis_router
from utils import get_current_user


def _db():
    engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _client(db, identity):
    app = FastAPI()
    app.include_router(analysis_router)
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: identity
    return TestClient(app)


def _image(db):
    owner = models.User(username="doctor", hashed_password="hash", role="doctor")
    db.add(owner)
    db.flush()
    patient = models.Patient(owner_user_id=owner.id, patient_external_id="P-1", age=50, gender="F")
    db.add(patient)
    db.flush()
    image = models.Image(patient_id=patient.id, modality="MRI", file_path="/medical-data/p1.png")
    db.add(image)
    db.flush()
    return owner, image


def test_unreadable_mri_is_reported_separately_from_negative():
    db = _db()
    owner, image = _image(db)
    db.add(models.InferenceTask(
        celery_task_id="c1", task_type="mri_pipeline", target_id=image.id, status="failed",
        error_message="Không đọc được ảnh MRI (0/5 lát) – chưa có kết luận, cần kiểm tra lại dữ liệu.",
        result={"mri_status": "mri_unreadable", "slices_total": 5, "slices_failed": 5,
                "review_required": True, "review_reasons": ["mri_unreadable"], "error_code": "mri_unreadable"},
    ))
    db.commit()

    with _client(db, {"sub": str(owner.id), "role": "doctor"}) as client:
        response = client.get(f"/records/analysis/image/{image.id}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "failed"
    assert body["mri_status"] == "mri_unreadable"
    assert body["no_tumor_detected"] is not True
    assert body["mri_review_reasons"] == ["mri_unreadable"]
    assert (body["slices_total"], body["slices_failed"]) == (5, 5)


def test_negative_result_requires_review():
    db = _db()
    owner, image = _image(db)
    db.add(models.AnalysisResult(image_id=image.id, patient_id=image.patient_id, no_tumor_detected=True,
                                 mri_status="no_tumor_detected", review_required=True,
                                 review_reasons=["negative_needs_confirmation"]))
    db.commit()

    with _client(db, {"sub": str(owner.id), "role": "doctor"}) as client:
        body = client.get(f"/records/analysis/image/{image.id}").json()

    assert body["mri_status"] == "no_tumor_detected"
    assert body["mri_review_required"] is True
    assert body["mri_review_reasons"] == ["negative_needs_confirmation"]
