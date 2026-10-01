"""Full-output denominators, prospective sampling, and immutable evidence records."""

from __future__ import annotations

import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.capability.generation import generation_evidence as ge


def record(index: int, length: int = 80, role: str = "generation", sequence: str | None = None) -> dict:
    sequence = sequence if sequence is not None else ("ACDEFGHIKLMNPQRSTVWY" * 60)[:length]
    row = ge._row(sequence, arm="zymctrl", class_key="class", condition="requested" if role == "generation" else "natural", role=role, source_label="fixture", source_key=f"{role}/{index}", source_sample_index=index, primary_class=True)
    row.update({"near_duplicate_group": f"fixture/{index}", "target_profile_hit": True, "any_profile_hit": True, "pfam_families": ["PF1"], "profile_hit_classes": ["class"]})
    return row


class ProspectiveSelection(unittest.TestCase):
    def test_length_strata_represent_every_nonempty_band_with_exact_probabilities(self):
        rows = [record(i, length) for i, length in enumerate([80] * 60 + [200] * 20 + [400] * 10 + [800] * 2)]
        _, main, selection = ge.select_cohorts(rows)
        selected = [row for row in main if row["role"] == "generation"]
        self.assertEqual(len(selected), 16)
        self.assertEqual({row["stratum"] for row in selected}, {"16_128", "129_256", "257_512", "513_1024"})
        self.assertAlmostEqual(sum(1 / row["inclusion_probability"] for row in selected), len(rows))
        self.assertEqual(selection["generation"]["zymctrl|class|requested"]["selected_n"], 16)

    def test_scores_cannot_change_selection_and_unsupported_attempts_remain(self):
        rows = [record(i, 80 + i) for i in range(40)] + [record(40, sequence=""), record(41, sequence="AX" * 20), record(42, sequence="A" * 1030)]
        other = copy.deepcopy(rows)
        for row in other:
            row["target_profile_hit"] = False
            row["any_profile_hit"] = False
            row["reference_identity"] = 100
            row["model_score"] = 999
        _, first, _ = ge.select_cohorts(rows)
        _, second, _ = ge.select_cohorts(other)
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertEqual(len(rows), 43)
        self.assertEqual([row["structure_exclusion_reason"] for row in rows[-3:]], ["empty_sequence", "noncanonical_residue", "above_length_support"])

    def test_natural_main_excludes_pilot_exact_sequences_across_classes(self):
        rows = []
        for class_key in ("class", "another"):
            for i in range(30):
                row = record(i, role="natural_reference", sequence="A" * (20 + i) + "CGST")
                row["id"] = ge.identifier(class_key, i)
                row["class_key"] = class_key
                rows.append(row)
        pilot, main, _ = ge.select_cohorts(rows)
        first = {row["sequence_sha256"] for row in pilot if row["role"] == "natural_reference"}
        second = {row["sequence_sha256"] for row in main if row["role"] == "natural_reference"}
        self.assertTrue(first.isdisjoint(second))
        self.assertEqual(sum(row["role"] == "natural_reference" for row in main), 16)

    def test_shuffle_preserves_composition_and_does_not_inherit_annotation(self):
        parent = record(0)
        child = ge.composition_shuffle(parent, seed=ge.POLICY["seed"])
        self.assertEqual(Counter(parent["sequence"]), Counter(child["sequence"]))
        self.assertEqual(child["paired_id"], parent["id"])
        self.assertIsNone(child["target_profile_hit"])
        self.assertIsNone(child["any_profile_hit"])
        self.assertEqual(child, ge.composition_shuffle(parent, seed=ge.POLICY["seed"]))

    def test_small_pools_are_not_topped_up_or_sampled_with_replacement(self):
        rows = [record(i) for i in range(3)]
        _, main, selection = ge.select_cohorts(rows)
        self.assertEqual(sum(row["role"] == "generation" for row in main), 3)
        self.assertEqual(selection["generation"]["zymctrl|class|requested"]["shortfall"], 13)
        with self.assertRaisesRegex(ValueError, "represent every"):
            ge.allocate_strata({"a": 3, "b": 4}, 1)


class ImmutableInputs(unittest.TestCase):
    def test_unchanged_resume_is_allowed_but_changed_content_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            ge.write_immutable(path, b"first\n")
            ge.write_immutable(path, b"first\n")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                ge.write_immutable(path, b"second\n")
            self.assertEqual(path.read_bytes(), b"first\n")

    def test_fasta_duplicate_or_whitespace_is_not_silently_cleaned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.fasta"
            path.write_text(">one\nACDE\n>one\nACDE\n")
            with self.assertRaisesRegex(ValueError, "duplicate FASTA"):
                ge.read_fasta(path)
            path.write_text(">one\nAC DE\n")
            with self.assertRaisesRegex(ValueError, "invalid FASTA"):
                ge.read_fasta(path)


if __name__ == "__main__":
    unittest.main()
