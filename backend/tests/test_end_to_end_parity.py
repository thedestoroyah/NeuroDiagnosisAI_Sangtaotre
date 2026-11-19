"""Cùng một bệnh nhân tập test: đường train (đặc trưng cache) và đường serving thật cho cùng risk score (yêu cầu 1)."""

import io
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("NEURO_DATA_DIR", ROOT.parent / "BrainTumor_Research_2025" / "01_Datasets_Zipped"))
MANIFEST = ROOT / "research" / "prognosis" / "split_manifest.csv"
WEIGHTS = ROOT / "backend" / "ai_core" / "weights"
sys.path.insert(0, str(ROOT / "research" / "prognosis"))


def _requirements_missing() -> str | None:
    for path in (MANIFEST, WEIGHTS / "rna_reference.npz", DATA_DIR / "features" / "index.json"):
        if not path.exists():
            return f"thiếu {path}"
    return None


@unittest.skipIf(_requirements_missing(), _requirements_missing() or "")
class EndToEndParityTest(unittest.TestCase):
    def test_same_patient_same_risk(self):
        from ai_core.feature_schema import encode_clinical, encode_rna, load_rna_reference
        from ai_core.pipeline import TumorAnalysisPipeline
        from clinical_inputs import build_clinical_record
        from extract_features import mri_files, wsi_files
        from feature_model import forward_from_features
        from rna_import import read_rna_series
        from train import load_patients, make_batch

        manifest = pd.read_csv(MANIFEST)
        candidates = manifest[(manifest.split == "test") & (manifest.has_mri == 1) & (manifest.has_rna == 1)]
        if candidates.empty:
            self.skipTest("không có bệnh nhân test đủ MRI + RNA")
        row = candidates.iloc[[0]].reset_index(drop=True)
        pid = row.patient_id[0]

        # --- Đường train: đặc trưng cache + mã hóa từ CSV/rna_matrix như train.py
        rna_matrix = pd.read_csv(DATA_DIR / "PROCESSED_RNA" / "rna_matrix.csv", index_col=0)
        ref = load_rna_reference(str(WEIGHTS / "rna_reference.npz"))
        pipeline = TumorAnalysisPipeline(weights_dir=str(WEIGHTS), device="cpu")
        patient = load_patients(DATA_DIR, row, rna_matrix, ref)[0]
        batch = make_batch([patient], len(ref.columns), False, None, "cpu")
        with torch.no_grad():
            train_risk, _ = forward_from_features(
                pipeline.multimodal_model, batch["mri_feats"], batch["wsi_feats"], batch["rna"], batch["clinical"],
                batch["has_mri"], batch["has_wsi"], batch["has_rna"], batch["has_clinical"],
                mri_mask=batch["mri_mask"], wsi_mask=batch["wsi_mask"],
            )

        # --- Đường serving: file gốc → hàm tiền xử lý của pipeline → mô hình đã nạp từ weights
        slices = [cv2.imread(str(p)) for p in mri_files(DATA_DIR / "PROCESSED_MRI", pid)]
        tiles = [p.read_bytes() for p in wsi_files(DATA_DIR / "PROCESSED_WSI_v2", pid)]
        csv_row = pd.read_csv(DATA_DIR / "dataset_multimodal_final.csv").set_index("patient_id").loc[pid]
        rna_file = io.StringIO()
        rna_matrix.loc[[pid]].rename_axis("patient_id").to_csv(rna_file)
        db_patient = SimpleNamespace(age=csv_row.age, gender={0.0: "Male", 1.0: "Female"}.get(csv_row.gender))
        db_clinical = SimpleNamespace(**{k: (None if pd.isna(csv_row[k]) else str(int(csv_row[k]))) for k in
                                         ("race", "prior_treatment", "radiation_therapy", "pharmaceutical_therapy",
                                          "idh_mutation", "mgmt_methylation")}, grade=None)

        mri = pipeline.preprocess_slices_for_multimodal(slices)
        wsi = pipeline.preprocess_tiles_for_multimodal(tiles) if tiles else torch.zeros(1, 1, 3, 256, 256)
        rna, has_rna = pipeline.prepare_rna_tensor(read_rna_series(rna_file.getvalue().encode(), "csv"))
        clinical, has_clinical = pipeline.prepare_clinical_tensor(build_clinical_record(db_patient, db_clinical))
        self.assertEqual(encode_clinical(build_clinical_record(db_patient, db_clinical))[0], patient.clinical)
        torch.testing.assert_close(rna[0], torch.from_numpy(encode_rna(rna_matrix.loc[pid], ref)))
        with torch.no_grad():
            serve_risk, _ = pipeline.multimodal_model(
                mri, wsi, rna, clinical,
                torch.tensor([1.0]), torch.tensor([1.0 if tiles else 0.0]), torch.tensor([has_rna]), torch.tensor([has_clinical]),
                mri_mask=torch.ones(1, mri.shape[1]), wsi_mask=torch.full((1, wsi.shape[1]), 1.0 if tiles else 0.0),
            )
        print(f"\n{pid}: train-path risk {train_risk.item():.6f} | serving-path risk {serve_risk.item():.6f}")
        self.assertAlmostEqual(train_risk.item(), serve_risk.item(), delta=1e-3)


if __name__ == "__main__":
    unittest.main()
