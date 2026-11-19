"""Dựng bản ghi lâm sàng cho mô hình tiên lượng từ Patient và ClinicalData trong DB."""

from typing import Any

from ai_core.feature_schema import CLINICAL_FIELDS


def build_clinical_record(patient: Any, clinical: Any | None) -> dict[str, Any]:
    """Gom 9 trường lâm sàng mô hình dùng, đúng tên trong CLINICAL_FIELDS.

    Args:
        patient: đối tượng có `age` và `gender` (models.Patient).
        clinical: models.ClinicalData hoặc None nếu bệnh nhân chưa có bản ghi lâm sàng.

    Returns:
        dict {trường: giá trị thô hoặc None}; feature_schema.encode_clinical lo việc chuẩn hóa.
    """
    record = {field: getattr(clinical, field, None) for field in CLINICAL_FIELDS}
    record["age"] = getattr(patient, "age", None)
    record["gender"] = getattr(patient, "gender", None)
    return record
