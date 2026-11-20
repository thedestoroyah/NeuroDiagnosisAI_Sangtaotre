"""Ensemble nhiều seed của MultimodalBrainTumorModel cho serving (mô hình tiên lượng v2).

Các seed chỉ khác nhau ở phần đầu; backbone DenseNet121 đóng băng giống hệt nhau nên chạy một lần rồi dùng chung.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .feature_forward import forward_from_features, pooled_backbone_features
from .survival_net import ImageEncoder, MultimodalBrainTumorModel

BACKBONE_KEY = "feature_extractor"
FORMAT = "ensemble-heads-v1"


def _image_features(encoder: ImageEncoder, images: torch.Tensor) -> torch.Tensor:
    batch, slices = images.shape[:2]
    if slices == 0:
        return images.new_zeros(batch, 0, encoder.fc.in_features)
    return pooled_backbone_features(encoder, images.flatten(0, 1)).view(batch, slices, -1)


class EnsembleSurvivalModel(nn.Module):
    """Cùng chữ ký forward với MultimodalBrainTumorModel; trả (risk trung bình [B], attention trung bình [B, 4])."""

    def __init__(self, members: list[MultimodalBrainTumorModel]):
        super().__init__()
        if not members:
            raise ValueError("ensemble cần ít nhất một mô hình")
        shared_mri, shared_wsi = members[0].mri_encoder.feature_extractor, members[0].wsi_encoder.feature_extractor
        for member in members[1:]:
            member.mri_encoder.feature_extractor = shared_mri
            member.wsi_encoder.feature_extractor = shared_wsi
        self.members = nn.ModuleList(members)

    @property
    def mri_encoder(self) -> ImageEncoder:
        """Encoder MRI của thành viên đầu (backbone dùng chung): chỗ Grad-CAM móc hook."""
        return self.members[0].mri_encoder

    def forward(self, mri, wsi, rna, clinical, has_mri, has_wsi, has_rna, has_clinical, mri_mask=None, wsi_mask=None):
        mri_feats = _image_features(self.mri_encoder, mri)
        # Như forward gốc: nhánh WSI không lan truyền gradient (Grad-CAM chỉ cho MRI).
        with torch.no_grad():
            wsi_feats = _image_features(self.members[0].wsi_encoder, wsi)
        outputs = [forward_from_features(m, mri_feats, wsi_feats, rna, clinical, has_mri, has_wsi, has_rna, has_clinical,
                                         mri_mask=mri_mask, wsi_mask=wsi_mask) for m in self.members]
        risk = torch.stack([r for r, _ in outputs]).mean(dim=0)
        attn = torch.stack([a for _, a in outputs]).mean(dim=0)
        return risk, attn


def save_ensemble(member_states: list[dict], path: Path, meta: dict) -> None:
    """Lưu backbone một lần + phần đầu của từng seed.

    Args:
        member_states: state_dict đầy đủ của từng seed (như train.py lưu).
        meta: thông tin mô hình (số epoch, số gen, ...), chỉ gồm kiểu JSON đơn giản.

    Raises:
        ValueError: nếu backbone của các seed không giống hệt nhau.
    """
    backbone = {k: v for k, v in member_states[0].items() if BACKBONE_KEY in k}
    for state in member_states[1:]:
        if any(not torch.equal(state[k], v) for k, v in backbone.items()):
            raise ValueError("backbone của các seed khác nhau; không gộp được")
    heads = [{k: v for k, v in state.items() if BACKBONE_KEY not in k} for state in member_states]
    torch.save({"format": FORMAT, "backbone": backbone, "heads": heads, "meta": meta}, path)


def load_ensemble(path: Path, num_genes: int, device: str) -> tuple[EnsembleSurvivalModel, dict]:
    """Nạp file của save_ensemble (strict) thành EnsembleSurvivalModel ở chế độ eval; trả (mô hình, meta)."""
    stored = torch.load(path, map_location="cpu", weights_only=True)
    if stored.get("format") != FORMAT:
        raise ValueError(f"{path} không phải file ensemble ({stored.get('format')})")
    members = []
    for head in stored["heads"]:
        member = MultimodalBrainTumorModel(num_genes=num_genes)
        member.load_state_dict({**stored["backbone"], **head}, strict=True)
        members.append(member)
    return EnsembleSurvivalModel(members).to(device).eval(), stored["meta"]
