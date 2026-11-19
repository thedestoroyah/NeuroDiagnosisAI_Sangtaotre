"""Nguồn duy nhất cho phép mã hóa đầu vào của mô hình tiên lượng, dùng chung cho train (research/) và serving (pipeline).

Khớp với notebook huấn luyện `Bản sao của 07_training_loop.ipynb` (cell 2 và cell 16).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

SCHEMA_VERSION = "clinical18-v1"

# Thứ tự cố định như lúc train: ô 0–8 là giá trị, ô 9–17 là cờ có dữ liệu.
CLINICAL_FIELDS: tuple[str, ...] = (
    "age", "gender", "race", "grade", "prior_treatment",
    "radiation_therapy", "pharmaceutical_therapy", "idh_mutation", "mgmt_methylation",
)

_SCALE = {"age": 100.0, "race": 3.0, "grade": 4.0}
_NUMERIC_FIELDS = {"age", "race", "grade"}

# grade thiếu 100% trong dữ liệu train nên mô hình chưa từng thấy giá trị này: luôn coi là thiếu.
MODEL_IGNORED_FIELDS = frozenset({"grade"})

_GENDER_WORDS = {"male": 0.0, "m": 0.0, "female": 1.0, "f": 1.0}
_BINARY_WORDS = {"yes": 1.0, "true": 1.0, "no": 0.0, "false": 0.0}


def _to_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) or math.isinf(value) else value


def normalize_clinical_value(field: str, raw: Any) -> float | None:
    """Chuẩn hóa một giá trị lâm sàng thô (từ CSV, DB hoặc form) về số, chưa chia scale.

    Args:
        field: tên trường trong CLINICAL_FIELDS.
        raw: giá trị thô; None, NaN, chuỗi rỗng hoặc chuỗi không nhận ra đều nghĩa là thiếu.

    Returns:
        float (ví dụ age 65 → 65.0, gender "Male" → 0.0) hoặc None nếu thiếu/không hợp lệ.
    """
    text = raw.strip().lower() if isinstance(raw, str) else None
    if text == "":
        return None
    if field == "gender" and text in _GENDER_WORDS:
        return _GENDER_WORDS[text]
    if field not in _NUMERIC_FIELDS and text in _BINARY_WORDS:
        return _BINARY_WORDS[text]

    value = _to_float(raw)
    if value is None:
        return None
    if field not in _NUMERIC_FIELDS and value not in (0.0, 1.0):
        return None
    return value


def encode_clinical(record: Mapping[str, Any]) -> tuple[list[float], float]:
    """Mã hóa bản ghi lâm sàng thành vector 18 chiều đúng như lúc train.

    Args:
        record: dict có thể chứa các key trong CLINICAL_FIELDS; key khác (ki67, confidence MRI...) bị bỏ qua.

    Returns:
        (18 giá trị: 9 giá trị đã scale + 9 cờ, has_clinical = 1.0 nếu có ít nhất một cờ bằng 1).
    """
    values: list[float] = []
    flags: list[float] = []
    for field in CLINICAL_FIELDS:
        value = None if field in MODEL_IGNORED_FIELDS else normalize_clinical_value(field, record.get(field))
        values.append(0.0 if value is None else value / _SCALE.get(field, 1.0))
        flags.append(0.0 if value is None else 1.0)
    return values + flags, (1.0 if any(flags) else 0.0)
