"""Nguồn duy nhất cho phép mã hóa đầu vào của mô hình tiên lượng, dùng chung cho train (research/) và serving (pipeline).

Khớp với notebook huấn luyện `Bản sao của 07_training_loop.ipynb` (cell 2 và cell 16).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

SCHEMA_VERSION = "clinical18-v1"

# Tiền xử lý ảnh như lúc train: 256px, chuẩn hóa ImageNet, 6 lát MRI và 8 patch WSI lấy đều.
IMAGE_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MRI_SLICES = 6
WSI_PATCHES = 8
_STD_EPS = 1e-8

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


@dataclass(frozen=True)
class RnaReference:
    """Danh sách cột RNA (đúng thứ tự lúc train, gồm 4 cột N_*) và mean/std sau log1p tính trên tập train."""

    columns: list[str]
    mean: np.ndarray
    std: np.ndarray


def _gene_key(name: str) -> str:
    name = str(name).strip()
    return name.split(".")[0] if name.startswith("ENSG") else name


def fit_rna_reference(train_matrix) -> RnaReference:
    """Tính mean/std theo cột trên các dòng `rna_matrix.csv` của tập train.

    Args:
        train_matrix: pd.DataFrame, index là patient_id, cột theo đúng thứ tự file (gồm N_*).

    Returns:
        RnaReference; std dùng ddof=1 như pandas trong notebook.
    """
    logged = np.log1p(train_matrix.fillna(0.0).to_numpy(np.float64))
    return RnaReference(
        columns=[_gene_key(c) for c in train_matrix.columns],
        mean=logged.mean(axis=0),
        std=logged.std(axis=0, ddof=1),
    )


def save_rna_reference(ref: RnaReference, path: str) -> None:
    """Lưu RnaReference ra file .npz."""
    np.savez(path, columns=np.array(ref.columns), mean=ref.mean, std=ref.std)


def load_rna_reference(path: str) -> RnaReference:
    """Đọc RnaReference từ file .npz do save_rna_reference ghi."""
    with np.load(path, allow_pickle=False) as data:
        return RnaReference(columns=[str(c) for c in data["columns"]], mean=data["mean"], std=data["std"])


def encode_rna(values: Mapping[str, float], ref: RnaReference) -> np.ndarray:
    """Căn RNA theo tên cột lúc train rồi áp log1p và z-score bằng thống kê tập train.

    Args:
        values: dict hoặc pd.Series {tên cột: giá trị đã ở dạng log2(TPM+1) như rna_matrix.csv}.
            Tên Ensembl có hậu tố version (".15") được bỏ hậu tố; cột lạ bị bỏ qua.

    Returns:
        np.ndarray float32 dài len(ref.columns). Giá trị NaN được coi là 0 (như fillna(0) lúc train);
        cột không có trong `values` nhận 0 sau z-score (tức bằng trung bình tập train).
    """
    index = {name: i for i, name in enumerate(ref.columns)}
    raw = np.zeros(len(ref.columns), dtype=np.float64)
    present = np.zeros(len(ref.columns), dtype=bool)
    for name, value in values.items():
        i = index.get(_gene_key(name))
        if i is None:
            continue
        number = _to_float(value)
        raw[i] = 0.0 if number is None else number
        present[i] = True
    z = (np.log1p(raw) - ref.mean) / (ref.std + _STD_EPS)
    return np.where(present, z, 0.0).astype(np.float32)


def uniform_indices(n: int, k: int) -> list[int]:
    """Chọn k chỉ số rải đều trong [0, n) như lúc val/test; trả hết nếu n ≤ k."""
    if n <= k:
        return list(range(n))
    return [int(i) for i in np.linspace(0, n - 1, k, dtype=int)]


def image_transform():
    """Transform ảnh của val/test/serving: Resize 256 → ToTensor → Normalize ImageNet (ảnh PIL RGB)."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
