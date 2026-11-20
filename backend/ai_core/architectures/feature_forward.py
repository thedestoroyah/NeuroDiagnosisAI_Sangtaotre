"""Forward của MultimodalBrainTumorModel từ đặc trưng DenseNet121 đã pool (backbone đóng băng).

Dùng chung cho research (train trên đặc trưng cache) và serving (EnsembleSurvivalModel chạy backbone một lần).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .survival_net import ImageEncoder, MultimodalBrainTumorModel


def pooled_backbone_features(encoder: ImageEncoder, images: torch.Tensor) -> torch.Tensor:
    """Phần backbone của ImageEncoder.forward: relu(DenseNet features) → global pool → flatten.

    Args:
        encoder: mri_encoder hoặc wsi_encoder (để ở eval() khi tính cache).
        images: tensor [N, 3, 256, 256] đã chuẩn hóa ImageNet.

    Returns:
        tensor [N, 1024].
    """
    features = F.relu(encoder.feature_extractor(images), inplace=False)
    return encoder.global_pool(features).flatten(1)


def _encode_branch(encoder: ImageEncoder, feats: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Phần sau backbone của ImageEncoder.forward: dropout → fc → ELU → SliceAttention."""
    batch, slices, _ = feats.shape
    if slices == 0:
        return torch.zeros(batch, encoder.fc.out_features, device=feats.device)
    projected = encoder.act(encoder.fc(encoder.dropout(feats)))
    return encoder.slice_attention(projected, mask=mask)


def forward_from_features(
    model: MultimodalBrainTumorModel,
    mri_feats: torch.Tensor,
    wsi_feats: torch.Tensor,
    rna: torch.Tensor,
    clinical: torch.Tensor,
    has_mri: torch.Tensor,
    has_wsi: torch.Tensor,
    has_rna: torch.Tensor,
    has_clinical: torch.Tensor,
    mri_mask: torch.Tensor | None = None,
    wsi_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tái hiện MultimodalBrainTumorModel.forward (survival_net.py) từ đặc trưng đã pool.

    Khác duy nhất: nhánh WSI không bị no_grad/detach, để fc và slice attention của WSI được học.
    Ở chế độ eval giá trị đầu ra giống hệt forward gốc.

    Args:
        mri_feats: [B, S, 1024]; wsi_feats: [B, T, 1024]; rna: [B, num_genes]; clinical: [B, 18].
        has_*: [B] gồm 0/1; mri_mask/wsi_mask: [B, S]/[B, T] (1 = lát thật).

    Returns:
        (risk [B], attention của fusion [B, 4]).
    """
    feat_mri = _encode_branch(model.mri_encoder, mri_feats, mri_mask)
    feat_wsi = _encode_branch(model.wsi_encoder, wsi_feats, wsi_mask)
    feat_rna = model.rna_encoder(rna)
    feat_clinical = model.clinical_encoder(clinical)

    features = [F.layer_norm(f, (f.size(-1),)) for f in (feat_mri, feat_wsi, feat_rna, feat_clinical)]
    features_stack = torch.stack(features, dim=1)
    modality_mask = torch.stack([has_mri, has_wsi, has_rna, has_clinical], dim=1)
    fused, attn_weights = model.fusion(features_stack, mask=modality_mask)
    return model.classifier(fused).squeeze(-1), attn_weights
