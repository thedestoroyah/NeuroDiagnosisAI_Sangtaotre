"""EnsembleSurvivalModel: risk = trung bình các thành viên, backbone dùng chung, lưu/nạp gọn, Grad-CAM vẫn bắt được."""

import tempfile
import unittest
from pathlib import Path

import torch

from ai_core.architectures.ensemble import EnsembleSurvivalModel, load_ensemble, save_ensemble
from ai_core.architectures.survival_net import MultimodalBrainTumorModel
from ai_core.architectures.xai_gradcam import GradCAMExplainer

NUM_GENES = 30


def _members(n):
    members = []
    for seed in range(n):
        torch.manual_seed(seed)
        members.append(MultimodalBrainTumorModel(num_genes=NUM_GENES).eval())
    # Backbone đóng băng giống nhau ở mọi seed (ImageNet); test dựng lại điều đó.
    for m in members[1:]:
        m.mri_encoder.feature_extractor.load_state_dict(members[0].mri_encoder.feature_extractor.state_dict())
        m.wsi_encoder.feature_extractor.load_state_dict(members[0].wsi_encoder.feature_extractor.state_dict())
    return members


def _inputs():
    torch.manual_seed(5)
    return dict(
        mri=torch.randn(1, 2, 3, 256, 256), wsi=torch.randn(1, 3, 3, 256, 256),
        rna=torch.randn(1, NUM_GENES), clinical=torch.rand(1, 18),
        has_mri=torch.ones(1), has_wsi=torch.ones(1), has_rna=torch.ones(1), has_clinical=torch.ones(1),
        mri_mask=torch.ones(1, 2), wsi_mask=torch.ones(1, 3),
    )


class EnsembleModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.members = _members(2)
        with torch.no_grad():
            cls.expected = [m(**_inputs()) for m in cls.members]

    def test_risk_is_mean_of_member_forwards(self):
        model = EnsembleSurvivalModel(_members(2)).eval()
        with torch.no_grad():
            risk, attn = model(**_inputs())
        torch.testing.assert_close(risk, (self.expected[0][0] + self.expected[1][0]) / 2, atol=1e-4, rtol=0)
        torch.testing.assert_close(attn, (self.expected[0][1] + self.expected[1][1]) / 2, atol=1e-4, rtol=0)

    def test_save_load_roundtrip_stores_backbone_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ensemble.pth"
            save_ensemble([m.state_dict() for m in self.members], path, meta={"epochs": 3})
            stored = torch.load(path, map_location="cpu", weights_only=True)
            self.assertFalse(any("feature_extractor" in k for head in stored["heads"] for k in head))
            model, meta = load_ensemble(path, NUM_GENES, "cpu")
        self.assertEqual(meta["epochs"], 3)
        with torch.no_grad():
            risk, _ = model(**_inputs())
        torch.testing.assert_close(risk, (self.expected[0][0] + self.expected[1][0]) / 2, atol=1e-4, rtol=0)

    def test_save_refuses_members_with_different_backbones(self):
        states = [m.state_dict() for m in self.members]
        key = next(k for k in states[1] if "feature_extractor" in k and states[1][k].is_floating_point())
        states[1] = {**states[1], key: states[1][key] + 1.0}
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            save_ensemble(states, Path(tmp) / "e.pth", meta={})

    def test_gradcam_target_layer_gets_activations_and_gradients(self):
        model = EnsembleSurvivalModel(_members(2)).eval()
        cam = GradCAMExplainer(model=model, target_layer=model.mri_encoder.feature_extractor.denseblock4.denselayer16.conv2)
        for p in model.parameters():
            p.requires_grad = True
        inputs = _inputs()
        inputs["mri"].requires_grad_(True)
        risk, _ = model(**inputs)
        risk.sum().backward()
        self.assertIsNotNone(cam.activations)
        self.assertIsNotNone(cam.gradients)
        self.assertEqual(cam.activations.shape[0], 2)  # 2 lát MRI, backbone chỉ chạy một lần


if __name__ == "__main__":
    unittest.main()
