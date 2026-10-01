import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.capability.generation import structure_evidence as se
from src.capability.core.io import write_json


class StructureEvidence(unittest.TestCase):
    def test_strict_sequences_never_silently_crop_or_clean(self):
        kwargs = dict(min_length=16, max_length=1024)
        for sequence, reason in [("", "empty_sequence"), ("A" * 15, "below_minimum_length"), ("A" * 1025, "above_maximum_length"), ("ACD X" * 4, "noncanonical_residues"), ("a" * 16, "noncanonical_residues")]:
            self.assertEqual(se.eligibility(sequence, **kwargs), reason)
        self.assertIsNone(se.eligibility("A" * 16, **kwargs))
        self.assertIsNone(se.eligibility("A" * 1024, **kwargs))

    def test_cohort_integrity_and_duplicate_denominators(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "cohort.jsonl"
            rows = [{"id": "a", "sequence": "A" * 16}, {"id": "b", "sequence": "A" * 16}, {"id": "invalid", "sequence": ""}]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            self.assertEqual(len(se.load_cohort(path)), 3)
            sha = se.sequence_digest("")
            write_json(root / "objects" / sha / "result.json", {"evaluation_signature": "fixed", "sequence_sha256": sha, "status": "not_evaluable"})
            report = se.write_index(rows, root, "fixed", shard_index=0, num_shards=1)
            self.assertEqual(report["rows"], 3)
            self.assertEqual(report["unique_sequences"], 2)
            self.assertEqual(report["status_counts"], {"pending": 2, "not_evaluable": 1})
            rows[1]["id"] = "a"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                se.load_cohort(path)
            path.write_text(json.dumps({"id": "a", "sequence": "AAA", "sequence_sha256": "wrong"}))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                se.load_cohort(path)

    def test_diffusion_sample_splits_cover_the_replica_axis(self):
        self.assertEqual(se.diffusion_sample_splits(32, 1), [32])
        self.assertEqual(se.diffusion_sample_splits(32, 2), [16, 16])
        self.assertEqual(se.diffusion_sample_splits(32, 3), [11, 11, 10])
        self.assertEqual(se.diffusion_sample_splits(5, 2), [3, 2])
        self.assertEqual(se.diffusion_sample_splits(4, 8), [1, 1, 1, 1])
        self.assertEqual(sum(se.diffusion_sample_splits(32, 7)), 32)
        with self.assertRaisesRegex(ValueError, "positive"):
            se.diffusion_sample_splits(0, 2)

    def test_exact_duplicates_remain_on_one_shard_and_union_is_complete(self):
        sequences = ["A" * i for i in range(1, 100)]
        for size in (1, 4, 8):
            partitions = [{s for s in sequences if se.shard_for(s, size) == shard} for shard in range(size)]
            self.assertEqual(set.union(*partitions), set(sequences))
            self.assertEqual(sum(map(len, partitions)), len(sequences))

    def test_resume_refuses_changed_configuration_and_missing_prediction_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "result.json", {"evaluation_signature": "a", "sequence_sha256": "b", "status": "ok", "files_sha256": {"prediction.pdb": "c"}})
            with self.assertRaisesRegex(ValueError, "incompatible"):
                se.load_result(root, "changed", "b")
            with self.assertRaisesRegex(ValueError, "incomplete/corrupted"):
                se.load_result(root, "a", "b")

    def test_esmfold2_summary_converts_declared_0_1_scale_and_keeps_best_ptm_sample(self):
        from types import SimpleNamespace

        output = SimpleNamespace(
            ptm=np.asarray([0.2, 0.9]),
            plddt_ca=np.asarray([[0.1] * 5, [0.70, 0.70, 0.70, 0.70, 0.69]]),
            pae=np.ones((2, 5, 5)),
        )
        result = se.summarize_esmfold2(output, sequence="ACDEF")
        self.assertEqual(result["diffusion_sample_index"], 1)
        self.assertEqual(result["predictor"], se.PREDICTOR_ID)
        self.assertAlmostEqual(result["mean_ca_plddt"], 69.8)
        self.assertAlmostEqual(result["fraction_ca_plddt_ge70"], 0.8)
        self.assertFalse(result["predicted_confidence_event"])
        output.plddt_ca[1, -1] = 0.70
        self.assertTrue(se.summarize_esmfold2(output, sequence="ACDEF")["predicted_confidence_event"])
        output.plddt_ca[1, -1] = 1.2
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            se.summarize_esmfold2(output, sequence="ACDEF")

    def test_worker_receipt_must_bind_the_contract_it_prints(self):
        contract = {"model_id": se.PREDICTOR_ID, "seed": 20260905,
                    "code_sha256": {"module": "aa", "runner": "bb"}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "worker-000-of-001.json"
            write_json(path, {"contract": contract,
                              "evaluation_signature": se.contract_signature(contract)})
            self.assertEqual(se.manifest_signature(path), se.contract_signature(contract))
            # An operator-pinned signature is exactly the defect this refuses.
            write_json(path, {"contract": contract, "evaluation_signature": "d795500931460f19"})
            with self.assertRaisesRegex(ValueError, "does not bind the contract"):
                se.manifest_signature(path)
            # The full-contract digest is not the signature either: code digests
            # are provenance, so a run may not claim them as the measurement.
            write_json(path, {"contract": contract, "evaluation_signature": se.digest_json(contract)})
            with self.assertRaisesRegex(ValueError, "does not bind the contract"):
                se.manifest_signature(path)
            write_json(path, {"evaluation_signature": "abc"})
            with self.assertRaisesRegex(ValueError, "without a contract"):
                se.manifest_signature(path)

    def test_measurement_signature_ignores_code_digests_and_nothing_else(self):
        contract = {"model_id": se.PREDICTOR_ID, "num_loops": 3, "seed": 20260905,
                    "code_sha256": {"module": "aa", "runner": "bb"}}
        other_code = dict(contract, code_sha256={"module": "cc", "runner": "dd"})
        other_seed = dict(contract, seed=1)
        self.assertEqual(se.contract_signature(contract), se.contract_signature(other_code))
        self.assertNotEqual(se.contract_signature(contract), se.contract_signature(other_seed))
        self.assertNotIn("code_sha256", se.measurement_contract(contract))

    def test_tree_signature_refuses_a_tree_that_mixes_receipts(self):
        first = {"model_id": se.PREDICTOR_ID, "seed": 1}
        second = {"model_id": se.PREDICTOR_ID, "seed": 2}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(FileNotFoundError, "no worker receipt"):
                se.tree_signature(root)
            write_json(root / "worker-000-of-002.json",
                       {"contract": first, "evaluation_signature": se.contract_signature(first)})
            self.assertEqual(se.tree_signature(root), se.contract_signature(first))
            write_json(root / "worker-001-of-002.json",
                       {"contract": second, "evaluation_signature": se.contract_signature(second)})
            with self.assertRaisesRegex(ValueError, "mixes 2 evaluation signatures"):
                se.tree_signature(root)

    def test_verify_replays_the_recorded_row_set_and_refuses_unbound_artifacts(self):
        contract = {"model_id": se.PREDICTOR_ID, "seed": 20260905}
        signature = se.contract_signature(contract)
        folded, refused = "AC" * 12, "A" * 8
        rows = [{"id": "folded", "sequence": folded}, {"id": "refused", "sequence": refused}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "worker-000-of-001.json",
                       {"contract": contract, "evaluation_signature": signature})
            objects = root / "objects" / se.sequence_digest(folded)
            files = se.save_prediction(
                objects, {"ca_plddt_0_100": np.full(len(folded), 80.0)}, "ATOM\n")
            write_json(objects / "result.json", {
                "schema_version": se.SCHEMA_VERSION, "evaluation_signature": signature,
                "sequence_sha256": se.sequence_digest(folded), "length": len(folded),
                "status": "ok", "files_sha256": files, "mean_ca_plddt": 80.0,
                "diffusion_sample_splits": [16, 16]})
            write_json(root / "objects" / se.sequence_digest(refused) / "result.json", {
                "schema_version": se.SCHEMA_VERSION, "evaluation_signature": signature,
                "sequence_sha256": se.sequence_digest(refused), "length": len(refused),
                "status": "not_evaluable", "reason": "below_minimum_length"})

            with self.assertRaisesRegex(FileNotFoundError, "no index receipt"):
                se.verify_evidence(rows, root, shard_index=0, num_shards=1)
            recorded = se.write_index(rows, root, signature, shard_index=0, num_shards=1,
                                      filename="index.jsonl")
            self.assertEqual(recorded["status_counts"], {"ok": 1, "not_evaluable": 1})
            self.assertEqual(recorded["diffusion_sample_splits_counts"], {"[16, 16]": 1, "null": 1})
            replayed = se.verify_evidence(rows, root, shard_index=0, num_shards=1)
            self.assertEqual(replayed["index_sha256"], recorded["index_sha256"])
            self.assertEqual(replayed["verified_receipts"], ["index.summary.json"])

            # A row set that differs from the recorded one must not pass, even
            # when every extra row shares an already folded sequence.
            with self.assertRaisesRegex(ValueError, "disagrees on rows"):
                se.verify_evidence(rows + [{"id": "again", "sequence": folded}], root,
                                   shard_index=0, num_shards=1)
            # A prediction object outside the cohort must not pass.
            stray = root / "objects" / ("0" * 64)
            stray.mkdir()
            with self.assertRaisesRegex(ValueError, "object directories outside this shard's cohort"):
                se.verify_evidence(rows, root, shard_index=0, num_shards=1)
            stray.rmdir()
            # A stored prediction whose bytes moved away from its claimed digest
            # must not pass, even though the row itself is unchanged.
            (objects / "prediction.pdb").write_text("ATOM tampered\n")
            with self.assertRaisesRegex(ValueError, "incomplete/corrupted"):
                se.verify_evidence(rows, root, shard_index=0, num_shards=1)
            se.save_prediction(objects, {"ca_plddt_0_100": np.full(len(folded), 80.0)}, "ATOM\n")
            # A row with no terminal result must not pass.
            (objects / "result.json").unlink()
            with self.assertRaisesRegex(ValueError, "without a terminal result"):
                se.verify_evidence(rows, root, shard_index=0, num_shards=1)

    def test_a_positive_structure_contrast_is_unreachable_without_attained_calibration(self):
        """The calibration gate is a label on the arm, not an exception.

        The structure rung therefore cannot be read from the contrast alone: a
        strictly positive interval with no attained natural-control receipt
        yields ``uncalibrated_predictor`` and never the positive label.
        """
        from src.capability.generation.generation_biology_analysis import analyze

        def source(key, role="generation", **kwargs):
            sequence = kwargs.pop("sequence", "ACDEFGHIKLMNPQRS")
            return dict(id=key, sequence=sequence, length=len(sequence),
                        sequence_sha256=se.sequence_digest(sequence), arm="a", class_key="c",
                        condition="requested", role=role, primary_class=True,
                        near_duplicate_group=key, target_profile_hit=True, any_profile_hit=True,
                        profile_hit_classes=["c"], reference_identity=None,
                        reference_coverage=None, reference_search_status="not_searched",
                        phase="main", stratum="16_128", inclusion_probability=1.0,
                        paired_id=None) | kwargs

        def predicted(row, confidence):
            return dict(row, structure=dict(
                status="ok", sequence_sha256=row["sequence_sha256"], length=row["length"],
                ca_plddt=[confidence] * row["length"], mean_ca_plddt=confidence,
                fraction_ca_plddt_ge70=float(confidence >= 70), ptm=0.7))

        parents = [source(f"p{index}", class_key=f"c{index}") for index in range(8)]
        shuffles = [source(f"p{index}_shuffle", role="composition_shuffle", paired_id=f"p{index}",
                           class_key=f"c{index}", sequence="SRQPNMLKIHGFEDCA") for index in range(8)]
        subset = parents + shuffles
        predictions = [predicted(row, 85.0) for row in parents] + [predicted(row, 40.0) for row in shuffles]
        attained = analyze(parents, subset, predictions, phase="main", resamples=200,
                           calibration={"arms": {"a": {"calibration_attained": True}}})["arms"]["a"]
        self.assertGreater(attained["ci97_5"][0], 0)
        self.assertEqual(attained["interpretation"],
                         "positive_predictor_confidence_evidence_against_composition_shuffle")
        for receipt in ({}, {"arms": {}}, {"arms": {"a": {"calibration_attained": False}}}):
            unattained = analyze(parents, subset, predictions, phase="main", resamples=200,
                                 calibration=receipt)["arms"]["a"]
            self.assertGreater(unattained["ci97_5"][0], 0)
            self.assertFalse(unattained["calibration_attained"])
            self.assertEqual(unattained["interpretation"], "uncalibrated_predictor")

    def test_refuse_retired_esmfold_v1_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "config.json", {"model_type": "esmfold", "architectures": ["EsmForProteinFolding"], "esmfold_config": {}})
            (root / "pytorch_model.bin").write_bytes(b"not-weights")
            with self.assertRaisesRegex(ValueError, "retired"):
                se.require_esmfold2_checkpoint(root)
            write_json(root / "config.json", {"model_type": "esmfold2", "architectures": ["EsmFold2Model"]})
            self.assertEqual(se.require_esmfold2_checkpoint(root)["model_type"], "esmfold2")


if __name__ == "__main__":
    unittest.main()
