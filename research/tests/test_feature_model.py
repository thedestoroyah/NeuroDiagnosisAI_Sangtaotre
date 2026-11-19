"""forward từ đặc trưng cache phải bằng forward đầy đủ của mô hình serving (bằng chứng train ≡ serving)."""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "research" / "prognosis"))
from ai_core.architectures.survival_net import MultimodalBrainTumorModel  # noqa: E402
from feature_model import forward_from_features, pooled_backbone_features  # noqa: E402

NUM_GENES = 50


class FeatureForwardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = MultimodalBrainTumorModel(num_genes=NUM_GENES).eval()

    def _inputs(self):
        torch.manual_seed(1)
        mri = torch.randn(2, 3, 3, 256, 256)
        wsi = torch.randn(2, 4, 3, 256, 256)
        rna = torch.randn(2, NUM_GENES)
        clinical = torch.rand(2, 18)
        has = dict(has_mri=torch.tensor([1.0, 1.0]), has_wsi=torch.tensor([1.0, 0.0]),
                   has_rna=torch.tensor([1.0, 1.0]), has_clinical=torch.tensor([1.0, 1.0]))
        masks = dict(mri_mask=torch.tensor([[1.0, 1, 1], [1, 1, 0]]), wsi_mask=torch.tensor([[1.0, 1, 1, 1], [0, 0, 0, 0]]))
        return mri, wsi, rna, clinical, has, masks

    def test_forward_from_features_equals_full_forward(self):
        mri, wsi, rna, clinical, has, masks = self._inputs()
        with torch.no_grad():
            expected_risk, expected_attn = self.model(mri, wsi, rna, clinical, **has, **masks)
            mri_feats = pooled_backbone_features(self.model.mri_encoder, mri.flatten(0, 1)).view(2, 3, -1)
            wsi_feats = pooled_backbone_features(self.model.wsi_encoder, wsi.flatten(0, 1)).view(2, 4, -1)
            risk, attn = forward_from_features(self.model, mri_feats, wsi_feats, rna, clinical, **has, **masks)
        self.assertEqual(tuple(mri_feats.shape), (2, 3, 1024))
        torch.testing.assert_close(risk, expected_risk, atol=1e-4, rtol=0)
        torch.testing.assert_close(attn, expected_attn, atol=1e-4, rtol=0)

    def test_all_missing_no_nan(self):
        zeros = torch.zeros(1, 1, 1024)
        off = torch.zeros(1)
        with torch.no_grad():
            risk, _ = forward_from_features(self.model, zeros, zeros, torch.zeros(1, NUM_GENES), torch.zeros(1, 18),
                                            off, off, off, off, mri_mask=torch.zeros(1, 1), wsi_mask=torch.zeros(1, 1))
        self.assertTrue(torch.isfinite(risk).all())


if __name__ == "__main__":
    unittest.main()
