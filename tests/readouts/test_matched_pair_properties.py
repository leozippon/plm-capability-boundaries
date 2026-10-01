"""Conditions that must hold before a declared-property table may rank pairs.

Four of them are the reason the table exists. It has to be outcome-blind, or a
separating property found among 33 arms and 8 survivors is a story rather than a
fact. It has to refuse to relate a table whose bytes changed after the digest was
recorded. It has to distinguish a property the two checkpoints are recorded to
differ on from one neither record states, because ranking a pair by how little
the repository knows about it would put the least documented pair first. And its
two transcribed sources -- hidden width from the admitted readout report, depth
from the admitted manifests -- have to be checked against ``arms.py`` rather than
trusted, since a transcription is a second copy of a fact.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import scripts.capability.readouts.assemble_matched_pair_properties as mpp

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The table is built from the admitted expansion archive, a retained result that
#: is not distributed with the code repository.
needs_expansion_archive = unittest.skipUnless(
    mpp.EXPANSION_ARCHIVE.is_file(),
    f'{mpp.EXPANSION_ARCHIVE.relative_to(REPO_ROOT)} is absent: the retained readout '
    'expansion archive is a result, not distributed with the code repository')


def build_once(cache: dict = {}) -> dict:
    """One build shared across tests; reading the admitted archive is the cost."""

    if "payload" not in cache:
        cache["payload"] = mpp.build_properties()
    return copy.deepcopy(cache["payload"])


@needs_expansion_archive
class PropertyTableStructure(unittest.TestCase):
    def test_the_table_covers_the_admitted_panel_and_names_its_exclusion(self):
        payload = build_once()
        self.assertEqual(len(payload["arms"]), 34)
        self.assertEqual(payload["anchor_roster_size"], 33)
        self.assertEqual(payload["excluded_arm"], mpp.EC_CONDITIONED_ARM)
        anchor = [name for name, row in payload["arms"].items() if row["in_anchor_roster"]]
        self.assertEqual(len(anchor), 33)
        self.assertNotIn(mpp.EC_CONDITIONED_ARM, anchor)

    def test_every_arm_carries_every_declared_property(self):
        payload = build_once()
        for arm, row in payload["arms"].items():
            for name in mpp.PROPERTY_NAMES:
                self.assertIn(name, row, f"{arm} lacks {name}")
                self.assertIsNotNone(row[name], f"{arm}'s {name} is None rather than a value")

    def test_the_declared_property_set_is_the_one_the_report_relates(self):
        payload = build_once()
        declared = [entry["name"] for entry in payload["property_set"]]
        self.assertEqual(tuple(declared), mpp.PROPERTY_NAMES)
        groups = {entry["group"] for entry in payload["property_set"]}
        self.assertEqual(groups, {"checkpoint", "execution"})
        self.assertTrue(set(mpp.ORDINAL_PROPERTIES) <= set(mpp.PROPERTY_NAMES))

    def test_the_undeclared_table_neither_overlaps_nor_leaves_a_gap(self):
        from src.capability.core import arms as arms_module

        declared = (
            set(arms_module.PANEL)
            | set(arms_module.STAGED_ARMS)
            | set(arms_module.PROGEN3_ARMS)
        )
        payload = build_once()
        roster = set(payload["arms"])
        self.assertEqual(set(mpp.UNDECLARED_ARMS) & declared, set())
        self.assertTrue(
            roster <= declared | set(mpp.UNDECLARED_ARMS),
            f"unsourced arms: {sorted(roster - declared - set(mpp.UNDECLARED_ARMS))}",
        )


@needs_expansion_archive
class TranscriptionIsChecked(unittest.TestCase):
    """The two transcribed axes must agree with ``arms.py``, not merely exist."""

    def test_a_wrong_transcribed_width_stops_the_build(self):
        original = dict(mpp.HIDDEN_WIDTH)
        mpp.HIDDEN_WIDTH["progen2-base"] = original["progen2-base"] + 1
        try:
            with self.assertRaises(ValueError) as caught:
                mpp.build_properties()
            self.assertIn("width", str(caught.exception))
        finally:
            mpp.HIDDEN_WIDTH.clear()
            mpp.HIDDEN_WIDTH.update(original)

    def test_a_missing_transcribed_width_stops_the_build(self):
        original = dict(mpp.HIDDEN_WIDTH)
        del mpp.HIDDEN_WIDTH["rita-xl"]
        try:
            with self.assertRaises(ValueError):
                mpp.build_properties()
        finally:
            mpp.HIDDEN_WIDTH.clear()
            mpp.HIDDEN_WIDTH.update(original)

    def test_depth_comes_from_the_manifest_and_matches_the_declaration(self):
        from src.capability.core import arms as arms_module

        payload = build_once()
        specs = {}
        for table in (arms_module.PANEL, arms_module.STAGED_ARMS, arms_module.PROGEN3_ARMS):
            specs.update(table)
        checked = 0
        for arm, row in payload["arms"].items():
            spec = specs.get(arm)
            if spec is None:
                continue
            self.assertEqual(row["depth_blocks"], int(spec.n_layer), arm)
            checked += 1
        self.assertGreaterEqual(checked, 25)


@needs_expansion_archive
class OutcomeBlindness(unittest.TestCase):
    def test_the_table_does_not_read_an_outcome(self):
        reference = mpp.digest(mpp.build_properties())
        saved = (dict(mpp.REPRESENTATION_OUTCOME), dict(mpp.LIKELIHOOD_OUTCOME))
        mpp.REPRESENTATION_OUTCOME.clear()
        mpp.LIKELIHOOD_OUTCOME.clear()
        try:
            self.assertEqual(mpp.digest(mpp.build_properties()), reference)
        finally:
            mpp.REPRESENTATION_OUTCOME.update(saved[0])
            mpp.LIKELIHOOD_OUTCOME.update(saved[1])

    def test_the_outcome_classes_are_disjoint_and_name_anchor_arms(self):
        payload = build_once()
        anchor = {name for name, row in payload["arms"].items() if row["in_anchor_roster"]}
        for table in (mpp.REPRESENTATION_OUTCOME, mpp.LIKELIHOOD_OUTCOME):
            seen: set[str] = set()
            for members in table.values():
                self.assertEqual(seen & set(members), set(), "an arm holds two classes")
                seen |= set(members)
            self.assertTrue(seen <= anchor, f"non-anchor arms classified: {seen - anchor}")

    def test_relating_a_changed_table_is_refused(self):
        payload = build_once()
        frozen = mpp.digest(payload)
        report = mpp.separate(copy.deepcopy(payload), frozen)
        self.assertEqual(report["property_table_sha256"], frozen)

        tampered = copy.deepcopy(payload)
        tampered["arms"]["progen2-base"]["pretraining_corpus"] = "uniref90_bfd30"
        with self.assertRaises(ValueError) as caught:
            mpp.separate(tampered, frozen)
        self.assertIn("does not match the declared", str(caught.exception))

    def test_a_wrong_survivor_count_stops_the_report(self):
        payload = build_once()
        saved = mpp.REPRESENTATION_OUTCOME["resolved_positive_all_three"]
        mpp.REPRESENTATION_OUTCOME["resolved_positive_all_three"] = saved[:-1]
        try:
            with self.assertRaises(ValueError) as caught:
                mpp.separate(payload, None)
            self.assertIn("8 representation survivors", str(caught.exception))
        finally:
            mpp.REPRESENTATION_OUTCOME["resolved_positive_all_three"] = saved


class ComparisonRule(unittest.TestCase):
    def test_an_absent_record_is_undecidable_rather_than_a_difference(self):
        left = {"attention_structure": "dense_multi_head"}
        right = {"attention_structure": mpp._NR}
        self.assertEqual(
            mpp.compare_property(left, right, "attention_structure"), "undecidable"
        )
        self.assertEqual(
            mpp.compare_property(right, right, "attention_structure"), "undecidable"
        )

    def test_parameter_counts_match_inside_the_declared_tolerance_and_differ_outside(self):
        near = (
            {"parameter_count_nominal": 764_000_000},
            {"parameter_count_nominal": 765_000_000},
        )
        far = (
            {"parameter_count_nominal": 151_100_000},
            {"parameter_count_nominal": 765_000_000},
        )
        self.assertEqual(mpp.compare_property(*near, "parameter_count_nominal"), "match")
        self.assertEqual(mpp.compare_property(*far, "parameter_count_nominal"), "differ")
        inside = (
            {"parameter_count_nominal": 100},
            {"parameter_count_nominal": 111},
        )
        outside = (
            {"parameter_count_nominal": 100},
            {"parameter_count_nominal": 112},
        )
        self.assertEqual(mpp.compare_property(*inside, "parameter_count_nominal"), "match")
        self.assertEqual(mpp.compare_property(*outside, "parameter_count_nominal"), "differ")

    def test_every_other_property_is_compared_exactly(self):
        left = {"depth_blocks": 32}
        right = {"depth_blocks": 33}
        self.assertEqual(mpp.compare_property(left, right, "depth_blocks"), "differ")
        self.assertEqual(mpp.compare_property(left, left, "depth_blocks"), "match")


@needs_expansion_archive
class SeparationAndPairs(unittest.TestCase):
    def test_pairs_are_only_those_whose_outcome_class_differs(self):
        payload = build_once()
        report = mpp.separate(payload, None)
        self.assertTrue(report["pairs"])
        for entry in report["pairs"]:
            self.assertTrue(
                entry["representation_outcome_differs"]
                or entry["likelihood_outcome_differs"],
                entry["arms"],
            )
            verdicts = len(entry["differing"]) + len(entry["matching"]) + len(entry["undecidable"])
            self.assertEqual(verdicts, len(mpp.CHECKPOINT_PROPERTIES), entry["arms"])

    def test_pairs_are_ranked_by_how_few_properties_differ(self):
        payload = build_once()
        report = mpp.separate(payload, None)
        keys = [
            (entry["n_differing"], entry["n_undecidable"]) for entry in report["pairs"]
        ]
        self.assertEqual(keys, sorted(keys))

    def test_a_property_unrecorded_on_both_sides_does_not_read_as_separating(self):
        payload = build_once()
        report = mpp.separate(payload, None)
        entry = report["representation_separation"]["released_storage_precision"]
        self.assertIn(mpp._NR, entry["inside"])
        self.assertIn(mpp._NR, entry["outside"])
        self.assertFalse(entry["separates"])

    def test_the_two_boundaries_are_reported_separately(self):
        payload = build_once()
        report = mpp.separate(payload, None)
        self.assertIn("representation_separation", report)
        self.assertIn("likelihood_separation", report)
        self.assertEqual(
            set(report["representation_separation"]), set(mpp.PROPERTY_NAMES)
        )
        self.assertEqual(set(report["likelihood_separation"]), set(mpp.PROPERTY_NAMES))


@needs_expansion_archive
class CanonicalForm(unittest.TestCase):
    def test_the_digest_is_of_a_canonical_form_and_is_order_independent(self):
        payload = build_once()
        shuffled = {key: payload[key] for key in reversed(list(payload))}
        self.assertEqual(mpp.digest(payload), mpp.digest(shuffled))
        self.assertEqual(json.loads(mpp.canonical(payload).decode("utf-8")), payload)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
