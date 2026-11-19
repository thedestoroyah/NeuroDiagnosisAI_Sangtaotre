"""Chứng minh đường serving (DB → task → pipeline) cho ra đúng vector như đường train (yêu cầu 1)."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from ai_core.feature_schema import encode_clinical
from ai_core.pipeline import TumorAnalysisPipeline
from clinical_inputs import build_clinical_record
from rna_import import read_rna_series
from tests.test_feature_schema import FIXTURE, _notebook_reference

SAMPLE_RNA = Path(__file__).resolve().parents[2] / "sample_data" / "TCGA-12-1093" / "RNA_sequence.csv"


def _db_objects(row):
    """Dựng Patient/ClinicalData như trong DB: gender dạng chữ, các trường 0/1 dạng chuỗi."""
    def as_db_string(value):
        return None if pd.isna(value) else str(int(value))

    gender = None if pd.isna(row["gender"]) else ("Female" if row["gender"] == 1 else "Male")
    patient = SimpleNamespace(age=None if pd.isna(row["age"]) else row["age"], gender=gender)
    clinical = SimpleNamespace(
        grade=None,
        race=as_db_string(row["race"]),
        prior_treatment=as_db_string(row["prior_treatment"]),
        radiation_therapy=as_db_string(row["radiation_therapy"]),
        pharmaceutical_therapy=as_db_string(row["pharmaceutical_therapy"]),
        idh_mutation=as_db_string(row["idh_mutation"]),
        mgmt_methylation=as_db_string(row["mgmt_methylation"]),
        ki67_index=35.0,
    )
    return patient, clinical


class ClinicalParityTest(unittest.TestCase):
    def test_db_path_equals_training_path(self):
        rows = pd.read_csv(FIXTURE)
        for _, row in rows.head(3).iterrows():
            patient, clinical = _db_objects(row)
            vec, has = encode_clinical(build_clinical_record(patient, clinical))
            expected_vec, expected_has = _notebook_reference(row)
            for i, (got, want) in enumerate(zip(vec, expected_vec)):
                self.assertAlmostEqual(got, want, places=6, msg=f"{row['patient_id']} ô {i}")
            self.assertEqual(has, expected_has)

    def test_record_without_clinical_row(self):
        record = build_clinical_record(SimpleNamespace(age=40, gender="Male"), None)
        vec, has = encode_clinical(record)
        self.assertEqual((vec[0], vec[1], vec[9], vec[10]), (0.4, 0.0, 1.0, 1.0))
        self.assertEqual(has, 1.0)

    def test_pipeline_tensor_equals_schema(self):
        pipeline = object.__new__(TumorAnalysisPipeline)
        pipeline.device = "cpu"
        record = {"age": 61, "gender": "Male", "race": "1", "idh_mutation": "0", "grade": "4"}
        tensor, has = pipeline.prepare_clinical_tensor(record)
        expected, expected_has = encode_clinical(record)
        self.assertEqual(tuple(tensor.shape), (1, 18))
        torch.testing.assert_close(tensor[0], torch.tensor(expected))
        self.assertEqual(has, expected_has)


class RnaReadTest(unittest.TestCase):
    def test_rna_series_keeps_n_star(self):
        series = read_rna_series(SAMPLE_RNA.read_bytes(), "csv")
        self.assertIn("N_unmapped", series.index)
        self.assertNotIn("patient_id", series.index)
        self.assertEqual(len(series), 60664)
        self.assertEqual(list(series.index[:5]), ["N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous", "ENSG00000000003"])
        self.assertAlmostEqual(series["ENSG00000000003"], 4.827834264273176)

    def test_rna_series_tsv(self):
        tsv = "patient_id\tN_unmapped\tENSG00000000003.15\nP1\t\t2.5\n".encode()
        series = read_rna_series(tsv, "tsv")
        self.assertEqual(list(series.index), ["N_unmapped", "ENSG00000000003.15"])
        self.assertTrue(pd.isna(series["N_unmapped"]))
        self.assertEqual(series["ENSG00000000003.15"], 2.5)


if __name__ == "__main__":
    unittest.main()
