"""What the near-duplicate grouping must get right for a split to be a split.

The property being protected is narrow and the failure it prevents is silent. A
stage fits a map on one split and reports it on another; if a held-out record has
a near-copy on the training side, the number reported is partly a number about
the training side, and nothing in the artefact looks wrong. On the Swiss-Prot
pool the model-diffing stage draws, that is not a corner case -- 41.4% of held-out
records have a relative at 95% identity or above and only 17.4% are exact, so an
exact-string check leaves most of it in place.

Four things therefore have to hold, and each of them can be false while the
grouping still returns a plausible-looking partition.

**Near-duplicates end up in one group.** Identical records, a record contained in
a longer one, and a record with scattered substitutions are all the same unit.

**Unrelated records do not.** A relation that joins everything is not a relation:
it produces one group, no split exists, and the stage refuses for a reason that
has nothing to do with the corpus. This is the failure mode of shingling *English*
by character rather than by word, and it is tested rather than reasoned about.

**No group is ever divided.** This is the whole point of splitting by group, and
it is checked on the returned mask rather than trusted from the construction.

**A pool that cannot be split refuses.** Whole groups are indivisible, so a pool
dominated by one of them has no split at the requested fraction. The honest
outcome is a loud refusal naming the group's size; a quieter one would report a
fraction it did not achieve.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.capability.generation.near_duplicates import (  # noqa: E402
    NEAR_DUPLICATE_CONTAINMENT,
    boundary_containment,
    group_disjoint_split,
    near_duplicate_groups,
    screen_against_training_stream,
    shingles,
)

AA20 = "ACDEFGHIKLMNPQRSTVWY"


def _sequences(n: int, *, length: int = 120, seed: int = 0) -> list[str]:
    """Unrelated random protein records: no two share a run of any length."""

    rng = np.random.default_rng(seed)
    return [
        "".join(AA20[int(i)] for i in rng.integers(0, len(AA20), size=length))
        for _ in range(n)
    ]


def _mutate(sequence: str, *, fraction: float, seed: int) -> str:
    rng = np.random.default_rng(seed)
    residues = list(sequence)
    for position in rng.choice(
        len(residues), size=max(1, int(fraction * len(residues))), replace=False
    ):
        residues[int(position)] = AA20[int(rng.integers(0, len(AA20)))]
    return "".join(residues)


class TheShinglesAreTheCorpusOwnSymbol(unittest.TestCase):
    def test_a_protein_shingles_in_residues_and_a_document_in_words(self):
        self.assertEqual(shingles("ACDEFG", unit="residues"), {"ACDEF", "CDEFG"})
        self.assertEqual(
            shingles("the cat sat on the mat here", unit="characters"),
            {"the cat sat on the", "cat sat on the mat", "sat on the mat here"},
        )

    def test_a_record_too_short_to_carry_a_shingle_carries_none(self):
        # Not silently joined to everything and not dropped: the count is reported
        # by the grouping and the caller decides.
        self.assertEqual(shingles("ACD", unit="residues"), frozenset())
        self.assertEqual(shingles("two words", unit="characters"), frozenset())

    def test_an_undeclared_symbol_unit_is_refused(self):
        with self.assertRaises(ValueError):
            shingles("ACDEFG", unit="tokens")


class NearDuplicatesMustBecomeOneUnit(unittest.TestCase):
    def test_identical_and_contained_and_mutated_records_group_together(self):
        base = _sequences(1, length=300, seed=1)[0]
        records = [
            base,
            base,  # byte-identical, as Swiss-Prot carries one protein per strain
            base[40:260],  # a fragment: high containment, low Jaccard
            _mutate(base, fraction=0.03, seed=2),  # ~97% identity
            *_sequences(6, length=300, seed=3),  # unrelated
        ]
        groups, summary = near_duplicate_groups(records, unit="residues")
        self.assertEqual(len(set(groups[:4].tolist())), 1)
        self.assertEqual(summary["largest_group_size"], 4)
        self.assertEqual(summary["n_groups"], 7)
        self.assertEqual(summary["n_singleton_groups"], 6)

    def test_unrelated_records_stay_apart_in_both_symbol_units(self):
        proteins = _sequences(24, seed=4)
        groups, summary = near_duplicate_groups(proteins, unit="residues")
        self.assertEqual(summary["n_groups"], 24)
        self.assertEqual(summary["largest_group_size"], 1)
        # English shingled by word. The same documents shingled by CHARACTER
        # would share nearly every five-character run and collapse into one
        # group, which is why the text unit is words.
        documents = [
            "the quick brown fox jumps over the lazy dog again and again",
            "a slow green turtle walks under the busy road at noon today",
            "interpretability methods must be audited before they are trusted",
        ]
        groups, summary = near_duplicate_groups(documents, unit="characters")
        self.assertEqual(summary["n_groups"], 3)

    def test_group_ids_are_a_function_of_the_records_alone(self):
        records = _sequences(12, seed=5)
        records[7] = records[2]
        first, _ = near_duplicate_groups(records, unit="residues")
        second, _ = near_duplicate_groups(list(records), unit="residues")
        self.assertTrue(np.array_equal(first, second))
        # Allocated in record order, so the first record is always group zero.
        self.assertEqual(int(first[0]), 0)

    def test_an_impossible_containment_is_refused(self):
        for value in (0.0, 1.5):
            with self.assertRaises(ValueError):
                near_duplicate_groups(_sequences(4, seed=6), unit="residues", containment=value)


class NoGroupMayBeDivided(unittest.TestCase):
    def test_the_split_never_divides_a_group_and_records_its_achieved_fraction(self):
        records = _sequences(60, seed=7)
        for index in range(50, 60):  # a ten-member near-duplicate group
            records[index] = _mutate(records[50], fraction=0.02, seed=index)
        groups, _ = near_duplicate_groups(records, unit="residues")
        self.assertEqual(int(np.bincount(groups).max()), 10)
        train, summary = group_disjoint_split(
            groups, n_train=48, seed=11, fraction_tolerance=0.2
        )
        self.assertEqual(summary["verdict"], "GROUP_DISJOINT")
        self.assertEqual(
            np.intersect1d(np.unique(groups[train]), np.unique(groups[~train])).size, 0
        )
        self.assertEqual(summary["n_train_records"], int(train.sum()))
        self.assertAlmostEqual(summary["requested_train_fraction"], 48 / 60)

    def test_a_pool_dominated_by_one_group_refuses_and_names_it(self):
        # 45 near-duplicates of one record against 11 unrelated ones. Whichever
        # order the groups are drawn in, the training side is the whole group
        # plus nothing, so no split near the requested fraction exists.
        records = _sequences(11, seed=8)
        base = _sequences(1, length=200, seed=9)[0]
        records += [_mutate(base, fraction=0.02, seed=100 + i) for i in range(45)]
        groups, _ = near_duplicate_groups(records, unit="residues")
        self.assertEqual(int(np.bincount(groups).max()), 45)
        with self.assertRaises(RuntimeError) as caught:
            group_disjoint_split(groups, n_train=40, seed=11)
        message = str(caught.exception)
        self.assertIn("largest near-duplicate group holds 45", message)
        self.assertIn("do not widen the tolerance", message)

    def test_a_pool_that_is_one_group_refuses(self):
        records = [_sequences(1, length=200, seed=10)[0]] * 20
        groups, summary = near_duplicate_groups(records, unit="residues")
        self.assertEqual(summary["n_groups"], 1)
        with self.assertRaises(RuntimeError):
            group_disjoint_split(groups, n_train=16, seed=11)

    def test_an_out_of_range_training_size_is_refused(self):
        groups, _ = near_duplicate_groups(_sequences(10, seed=12), unit="residues")
        for n_train in (0, 10, 11):
            with self.assertRaises(ValueError):
                group_disjoint_split(groups, n_train=n_train, seed=11)


class TheBoundaryIsMeasuredNotAsserted(unittest.TestCase):
    def test_an_unrelated_split_reports_no_containment_and_a_planted_one_does(self):
        records = _sequences(20, seed=13)
        train = np.zeros(20, dtype=bool)
        train[:14] = True
        clean = boundary_containment(records, train, unit="residues")
        self.assertEqual(clean["n_held_out"], 6)
        self.assertLess(clean["max"], NEAR_DUPLICATE_CONTAINMENT)
        self.assertEqual(clean["n_above_threshold"], 0)

        # The audit is a reading taken off the result, not a gate: pointed at a
        # split that a record-level draw would have produced, it has to see the
        # leakage the group split exists to remove.
        leaky = list(records)
        leaky[17] = _mutate(records[3], fraction=0.02, seed=14)
        planted = boundary_containment(leaky, train, unit="residues")
        self.assertGreaterEqual(planted["max"], NEAR_DUPLICATE_CONTAINMENT)
        self.assertEqual(planted["n_above_threshold"], 1)

    def test_a_mask_that_is_not_one_boolean_per_record_is_refused(self):
        with self.assertRaises(ValueError):
            boundary_containment(
                _sequences(5, seed=15), np.ones(4, dtype=bool), unit="residues"
            )


class AStreamedTrainingSideIsScreenedTheSameWay(unittest.TestCase):
    """The pool form and the stream form must agree on the same records.

    `train_transcoder.py` cannot group its training side: it is hundreds of
    thousands of records read once. What it can do is index the few hundred
    held-out candidates and stream the training records past them. That is the
    same relation at the same threshold, and the risk is that "the same" is
    asserted rather than held -- so it is checked against the pool form here.
    """

    def test_a_planted_near_duplicate_is_dropped_and_the_rest_survive(self):
        training = _sequences(40, seed=21)
        candidates = _sequences(10, seed=22)
        candidates[4] = _mutate(training[7], fraction=0.02, seed=23)

        keep, screen = screen_against_training_stream(
            candidates, iter(training), unit="residues"
        )
        self.assertEqual(list(np.flatnonzero(~keep)), [4])
        self.assertEqual(screen["n_dropped"], 1)
        self.assertEqual(screen["n_kept"], 9)
        self.assertEqual(screen["n_training_records_screened"], 40)
        self.assertGreaterEqual(screen["max_containment"], NEAR_DUPLICATE_CONTAINMENT)

    def test_the_stream_form_drops_exactly_what_the_pool_form_would_group(self):
        training = _sequences(30, seed=31)
        candidates = _sequences(8, seed=32)
        # A fragment of a training record and a substituted copy of another:
        # both are one unit with their source under the pool relation, and the
        # first is the length-asymmetric shape a Jaccard threshold misses.
        candidates[1] = training[3][20:90]
        candidates[6] = _mutate(training[11], fraction=0.03, seed=33)

        keep, _ = screen_against_training_stream(
            candidates, iter(training), unit="residues"
        )
        groups, _ = near_duplicate_groups(training + candidates, unit="residues")
        pooled = {
            position
            for position in range(len(candidates))
            if groups[len(training) + position] in set(groups[: len(training)])
        }
        self.assertEqual(set(np.flatnonzero(~keep).tolist()), pooled)
        self.assertEqual(pooled, {1, 6})

    def test_an_unrelated_text_stream_drops_nothing(self):
        """Attainability on the text control, as an executable property.

        The screen is applied to both modes, so if it removed documents from an
        unrelated web corpus it would be silently shrinking the text mode's
        evaluation budget rather than protecting it.
        """

        rng = np.random.default_rng(41)
        vocabulary = [f"w{index}" for index in range(4000)]
        documents = [
            " ".join(vocabulary[int(i)] for i in rng.integers(0, 4000, size=200))
            for _ in range(30)
        ]
        keep, screen = screen_against_training_stream(
            documents[20:], iter(documents[:20]), unit="characters"
        )
        self.assertTrue(keep.all())
        self.assertEqual(screen["n_dropped"], 0)
        self.assertLess(screen["max_containment"], NEAR_DUPLICATE_CONTAINMENT)

    def test_the_statistic_is_reported_threshold_free(self):
        training = _sequences(12, seed=51)
        candidates = _sequences(6, seed=52)
        _, screen = screen_against_training_stream(
            candidates, iter(training), unit="residues"
        )
        for field in ("max_containment", "q99_containment", "median_containment"):
            self.assertIn(field, screen)
        self.assertEqual(screen["containment_threshold"], NEAR_DUPLICATE_CONTAINMENT)
        self.assertEqual(screen["shingle_length"], 5)

    def test_an_empty_candidate_draw_and_an_impossible_threshold_are_refused(self):
        with self.assertRaises(ValueError):
            screen_against_training_stream([], iter([]), unit="residues")
        with self.assertRaises(ValueError):
            screen_against_training_stream(
                _sequences(3, seed=61), iter([]), unit="residues", containment=0.0
            )

    def test_a_candidate_too_short_to_carry_a_shingle_is_kept_and_counted(self):
        keep, screen = screen_against_training_stream(
            ["AC", *_sequences(3, seed=71)], iter(_sequences(5, seed=72)), unit="residues"
        )
        self.assertTrue(keep[0])
        self.assertEqual(screen["n_candidates_without_shingles"], 1)


if __name__ == "__main__":
    unittest.main()
