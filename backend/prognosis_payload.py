"""Gộp kết quả task MRI với task tiên lượng theo một quy tắc duy nhất (API, PDF và chatbox dùng chung).

Quy tắc: task tiên lượng được CHẠY sau task MRI (so created_at, không so updated_at vì mọi lần ghi giải thích/PDF
vào task đều làm updated_at mới) thì thay hẳn các khóa tiên lượng, kể cả giá trị None. Task MRI chưa xong
(lỗi, đang chạy) thì không ghép kết quả tiên lượng cũ vào.
"""

from __future__ import annotations

import datetime

PROGNOSIS_ONLY_KEYS = {
    "risk_score",
    "risk_group",
    "survival_curve_data",
    "fusion_attention",
    "rna_xai",
    "multimodal_xai_explanation",
    "xai_explanation",
    "multimodal_risk_xai_path",
    "multimodal_risk_xai_meta_path",
    "multimodal_gradcam_heatmap_path",
    "multimodal_gradcam_plus_heatmap_path",
    "multimodal_layercam_heatmap_path",
    "prognosis_meta",
}


def merge_prognosis_only(base_payload: dict, prognosis_payload: dict | None) -> dict:
    """Ghi các khóa tiên lượng của prognosis_payload đè lên bản sao base_payload (kể cả None); giữ khóa MRI."""
    merged = dict(base_payload or {})
    for key in PROGNOSIS_ONLY_KEYS:
        if prognosis_payload and key in prognosis_payload:
            merged[key] = prognosis_payload.get(key)
    return merged


def run_time(task) -> datetime.datetime:
    """Thời điểm task được tạo (lúc chạy mô hình); thiếu thì coi là cũ nhất."""
    return getattr(task, "created_at", None) or datetime.datetime.min


def prognosis_is_current(mri_task, prognosis_task) -> bool:
    """True nếu kết quả của prognosis_task phải được hiển thị thay cho tiên lượng nằm trong task MRI."""
    if not (prognosis_task and prognosis_task.result):
        return False
    if mri_task is None:
        return True
    if mri_task.status != "done":
        return False
    return run_time(prognosis_task) >= run_time(mri_task)


def payload_with_prognosis(mri_task, prognosis_task) -> dict:
    """Payload để hiển thị cho một ảnh MRI.

    Args:
        mri_task, prognosis_task: InferenceTask hoặc None (prognosis_task phải là task của đúng ảnh đó).

    Returns:
        dict mới, không sửa result của task:
        - không có task MRI: kết quả tiên lượng;
        - task MRI lỗi/đang chạy: chỉ payload MRI (không ghép tiên lượng cũ);
        - tiên lượng chạy sau MRI: payload MRI với khóa tiên lượng được thay hẳn;
        - ngược lại: payload MRI nguyên vẹn.
    """
    payload = dict(mri_task.result) if mri_task and mri_task.result else {}
    if mri_task is None:
        return dict(prognosis_task.result) if prognosis_task and prognosis_task.result else {}
    if prognosis_is_current(mri_task, prognosis_task):
        return merge_prognosis_only(payload, prognosis_task.result)
    return payload
