"""Reference distance and coverage must come from the same auditable HSP."""

import tempfile
import unittest
from pathlib import Path

from src.capability.generation import generation_annotations as ga
from src.capability.context import homology
from src.capability.generation.generation_evidence import sequence_hash


class NativeLedgerGate(unittest.TestCase):
    def _rows(self, arm: str, n: int = 800, **overrides):
        rows = []
        for index in range(n):
            row = {"arm": arm, "class_key": None, "condition": "unconditioned", "source_sample_index": index}
            row.update(overrides)
            rows.append(row)
        return rows

    def test_admits_expansion_arm_and_progen3(self):
        self.assertEqual(ga.require_native_ledger(self._rows("protgpt2")), "protgpt2")
        self.assertEqual(ga.require_native_ledger(self._rows("progen3-3b")), "progen3-3b")
        self.assertEqual(ga.native_alignment_fields("protgpt2"), homology.ALIGNMENT_FIELDS)
        self.assertEqual(ga.native_alignment_fields("progen3-3b"), homology.DIAMOND_FIELDS)

    def test_rejects_conditioned_unknown_and_short_ledgers(self):
        conditioned = self._rows("protgpt2")
        conditioned[0]["class_key"] = "EC:1"
        with self.assertRaisesRegex(ValueError, "unconditioned"):
            ga.require_native_ledger(conditioned)
        with self.assertRaisesRegex(ValueError, "zymctrl"):
            ga.require_native_ledger(self._rows("zymctrl"))
        with self.assertRaisesRegex(ValueError, "every declared"):
            ga.require_native_ledger(self._rows("protgpt2", n=10))


class UnconditionalDiversityCensus(unittest.TestCase):
    SEQ_A = "ACDEFGHIKLMNPQRSTVWYACDEF"
    SEQ_B = "YWVTRSQLNPMKIHGFEDCAYWVTR"

    def _ledger(self, sequences, hits=None, arm="protgpt2"):
        hits = [False] * len(sequences) if hits is None else list(hits)
        rows = []
        for index, sequence in enumerate(sequences):
            rows.append(
                {
                    "arm": arm,
                    "class_key": None,
                    "condition": "unconditioned",
                    "source_sample_index": index,
                    "id": f"{arm}:{index}",
                    "sequence": sequence,
                    "sequence_sha256": sequence_hash(sequence),
                    "any_profile_hit": hits[index],
                }
            )
        return rows

    def test_identical_nonempty_collapse_to_one_group(self):
        rows = self._ledger([self.SEQ_A] * 800, hits=[True] * 800)
        census = ga.unconditional_diversity_census(rows)
        self.assertEqual(census["n_groups_nonempty"], 1)
        self.assertEqual(census["n_unique_nonempty_hashes"], 1)
        self.assertEqual(census["n_singleton_groups_nonempty"], 0)
        self.assertEqual(census["n_any_profile_groups"], 1)
        self.assertTrue(census["sequence_diversity_readable"])

    def test_empty_decodes_share_one_group_and_zero_hits_are_not_zero_groups(self):
        rows = self._ledger([""] * 800)
        census = ga.unconditional_diversity_census(rows)
        self.assertEqual(census["n_empty"], 800)
        self.assertIsNone(census["n_groups_nonempty"])
        self.assertEqual(census["n_groups_including_empty"], 1)
        self.assertIsNone(census["n_any_profile_groups"])
        self.assertEqual(census["n_any_profile_hits"], 0)
        self.assertFalse(census["sequence_diversity_readable"])

    def test_two_sequences_and_hits_on_one_family(self):
        sequences = [self.SEQ_A] * 400 + [self.SEQ_B] * 400
        hits = [True] * 400 + [False] * 400
        census = ga.unconditional_diversity_census(self._ledger(sequences, hits=hits))
        self.assertEqual(census["n_groups_nonempty"], 2)
        self.assertEqual(census["n_unique_nonempty_hashes"], 2)
        self.assertEqual(census["n_any_profile_hits"], 400)
        self.assertEqual(census["n_any_profile_groups"], 1)

    def test_refuses_missing_profile_field_and_hit_on_empty(self):
        rows = self._ledger([self.SEQ_A] * 800)
        del rows[0]["any_profile_hit"]
        with self.assertRaisesRegex(ValueError, "any_profile_hit"):
            ga.unconditional_diversity_census(rows)
        broken = self._ledger([""] * 800, hits=[True] + [False] * 799)
        with self.assertRaisesRegex(ValueError, "empty decode"):
            ga.unconditional_diversity_census(broken)


class ReferenceCoverage(unittest.TestCase):
    def test_best_identity_hit_retains_its_own_query_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / "hits.tsv"
            table.write_text("q\ts1\t100\t40\t40\t1\t40\t100\t80\t1e-9\t60\n"
                             "q\ts2\t50\t90\t45\t3\t92\t100\t100\t1e-8\t50\n")
            hit = ga.best_hits(table, {"q": "A" * 100})["q"]
            fields = ga.reference_fields(hit, searched=True)
            self.assertEqual(fields["reference_identity"], 45)
            self.assertEqual(fields["reference_coverage"], 0.9)
            self.assertEqual(fields["reference_subject"], "s2")
            self.assertIsNone(fields["reference_target_coverage"])
            with self.assertRaisesRegex(ValueError, "query/length mismatch"):
                ga.best_hits(table, {"q": "A" * 99})

    def test_no_hit_is_not_zero_identity_or_unsearched(self):
        no_hit = ga.reference_fields(None, searched=True)
        missing = ga.reference_fields(None, searched=False)
        self.assertIsNone(no_hit["reference_identity"])
        self.assertIsNone(no_hit["reference_coverage"])
        self.assertNotEqual(no_hit["reference_search_status"], missing["reference_search_status"])

    def test_fresh_alignment_carries_target_coverage_from_the_same_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / "hits.tsv"
            table.write_text("q\ts\t60\t5\t3\t1\t4\t8\t10\t1e-9\t60\tAC-DE\tACGDE\n")
            hit = ga.best_hits(table, {"q": "ACDEACDE"}, fields=homology.ALIGNMENT_FIELDS)["q"]
            fields = ga.reference_fields(hit, searched=True)
            self.assertEqual(fields["reference_identity"], 37.5)
            self.assertEqual(fields["reference_coverage"], 0.5)
            self.assertEqual(fields["reference_target_coverage"], 0.5)
            self.assertEqual(fields["reference_target_coverage_status"], "aligned_subject_residue_fraction")


if __name__ == "__main__":
    unittest.main()
