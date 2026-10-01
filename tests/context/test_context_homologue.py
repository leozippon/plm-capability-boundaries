"""EXP-R2-228: the properties the in-context conditioning measurement rests on.

Written against the registration rather than against the implementation. Four of
these are the negative paths the design fails on if they are not honoured -- a
context item that survives the local-overlap screen it should not, an arm whose
budget cannot fit one context item, a cohort that has drifted from the digest a
scoring run was pinned to, and an identity that belongs to no declared band -- and
two are known-answer validations: one arithmetic, on planted per-unit likelihoods
whose endpoint value is known in advance, and one on a real checkpoint, where the
answer is known because it is published (Kantroo et al.'s self-copy collapse).
"""

from __future__ import annotations

import json
import sys
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.context import context_homologue as ch  # noqa: E402
from src.capability.core.arms import Arm, PANEL  # noqa: E402

PROGEN_SMALL = PANEL["progen2-small"].path
PROTGPT2 = PANEL["protgpt2"].path


def _tokenizer_arm(name: str) -> Arm:
    from transformers import AutoTokenizer

    spec = PANEL[name]
    tokenizer = AutoTokenizer.from_pretrained(str(spec.path), trust_remote_code=True)
    return Arm(spec=spec, model=None, tokenizer=tokenizer, device="cpu", dtype="float32")


_NATURAL: list[str] = []


def _natural_proteins(n: int) -> list[str]:
    """Real Swiss-Prot records of the frozen band, not a uniform draw.

    A uniformly random amino-acid string is not what a BPE learned over natural
    sequences compresses, and measuring a shuffle's cost against one would answer
    a different question from the one the design turns on.
    """

    from src.capability.core.arms import _eligible_protein_records

    if len(_NATURAL) < n:
        _NATURAL.clear()
        for record, _ in _eligible_protein_records(*ch.PROTEIN_BAND, with_ec=False):
            _NATURAL.append(record)
            if len(_NATURAL) >= n:
                break
    return _NATURAL[:n]


def _natural_protein(index: int) -> str:
    return _natural_proteins(index + 1)[index]


def _sequence(seed: int, length: int) -> str:
    generator = np.random.default_rng(seed)
    alphabet = np.array(list(ch.COMPOSITION_ALPHABET))
    return "".join(alphabet[generator.integers(0, alphabet.size, size=length)])


# ------------------------------------------------------- the local-overlap screen


class TheLocalOverlapScreenRemovesShortRangeCopies(unittest.TestCase):
    """Global identity does not screen the copying mechanism, so this must."""

    def test_a_planted_verbatim_run_is_found_and_excluded(self):
        target = _sequence(1, 200)
        needle = target[40:40 + ch.HIGH_LOCAL_OVERLAP_LCS + 5]
        carrier = _sequence(2, 90) + needle + _sequence(3, 90)
        clean = _sequence(4, 200)
        overlap = ch.pair_overlap(target, [carrier, clean], modality="protein")
        self.assertGreaterEqual(int(overlap["lcs"][0]), ch.HIGH_LOCAL_OVERLAP_LCS)
        self.assertLess(int(overlap["lcs"][1]), ch.HIGH_LOCAL_OVERLAP_LCS)
        excluded = ch.high_local_overlap(overlap, modality="protein")
        self.assertTrue(bool(excluded[0]))
        self.assertFalse(bool(excluded[1]))

    def test_shared_kmer_counts_travel_beside_the_substring(self):
        target = _sequence(5, 300)
        overlap = ch.pair_overlap(target, [target[:150] + _sequence(6, 150)], modality="protein")
        self.assertGreater(int(overlap["shared"][0]), 0)

    def test_a_screened_pair_never_reaches_a_primary_context(self):
        """The unit-level consequence: a high-overlap partner is in the other stratum."""

        records = [_sequence(index, 160) for index in range(40)]
        target = 0
        needle = records[target][20:20 + ch.HIGH_LOCAL_OVERLAP_LCS + 4]
        for partner in range(1, 9):
            records[partner] = _sequence(100 + partner, 70) + needle + _sequence(200 + partner, 70)
        hits = {(target, partner): 40.0 for partner in range(1, 25)}
        hits.update({(partner, target): 40.0 for partner in range(1, 25)})
        with unittest.mock.patch.object(ch, "BAND_TARGET_FLOOR", 1), unittest.mock.patch.object(
            ch, "MIN_CONTEXT_ITEMS", 4
        ):
            built = ch.protein_cohort_units(records, hits)
        strata = {unit["stratum"]: unit for unit in built["units"] if unit["target"] == target}
        self.assertIn("retained", strata)
        self.assertIn("high_local_overlap", strata)
        planted = set(range(1, 9))
        self.assertEqual(set(strata["retained"]["partners"]) & planted, set())
        self.assertTrue(set(strata["high_local_overlap"]["partners"]) <= planted)
        self.assertTrue(
            all(value < ch.HIGH_LOCAL_OVERLAP_LCS for value in strata["retained"]["partner_lcs"])
        )
        self.assertTrue(
            all(
                value >= ch.HIGH_LOCAL_OVERLAP_LCS
                for value in strata["high_local_overlap"]["partner_lcs"]
            )
        )


# ------------------------------------------------------------- the band contract


class TheIdentityBandsRefuseWhatTheyDoNotCover(unittest.TestCase):
    def test_every_scored_identity_lands_in_exactly_one_band(self):
        for value in (0.0, 29.9, 30.0, 49.9, 50.0, 69.9, 70.0, 89.9):
            band = ch.assign_identity_band(value)
            low, high = next((lo, hi) for name, lo, hi in ch.IDENTITY_BANDS if name == band)
            self.assertTrue(low <= value < high, f"{value} landed in {band}")

    def test_a_near_duplicate_identity_is_refused_rather_than_banded(self):
        with self.assertRaises(ValueError) as raised:
            ch.assign_identity_band(ch.NEAR_DUPLICATE_IDENTITY)
        self.assertIn("near duplicate", str(raised.exception).lower())

    def test_an_identity_outside_the_scale_is_refused(self):
        with self.assertRaises(ValueError):
            ch.assign_identity_band(-1.0)

    def test_the_stratification_refuses_an_unbanded_pair_rather_than_dropping_it(self):
        records = [_sequence(index, 120) for index in range(12)]
        hits = {(0, partner): 40.0 for partner in range(1, 10)}
        hits[(0, 10)] = float("nan")
        with unittest.mock.patch.object(ch, "BAND_TARGET_FLOOR", 1), unittest.mock.patch.object(
            ch, "MIN_CONTEXT_ITEMS", 4
        ):
            with self.assertRaises(ValueError):
                ch.protein_cohort_units(records, hits)


# ---------------------------------------------------------------- the token budget


@pytest.mark.skipif(not PROGEN_SMALL.is_dir(), reason="the ProGen2 checkpoint is host-local")
class TheBudgetIsTokensAndKIsAnOutcome(unittest.TestCase):
    def setUp(self):
        self.arm = _tokenizer_arm("progen2-small")

    def _cohort(self, records, units, filler_index=None):
        related = {str(unit["target"]): [] for unit in units}
        return {
            "digest": "unused",
            "protein": {
                "records": records,
                "units": units,
                "related": related,
                "filler": {
                    "index": filler_index if filler_index is not None else len(records) - 1,
                    "record": records[filler_index if filler_index is not None else -1],
                },
            },
        }

    def _unit(self, target, partners):
        return {
            "key": f"protein|id_lt_30|retained|{target:06d}",
            "modality": "protein",
            "band": "id_lt_30",
            "stratum": "retained",
            "target": target,
            "group": target,
            "partners": partners,
            "partner_lcs": [0] * len(partners),
            "partner_shared": [0] * len(partners),
        }

    def test_an_arm_that_cannot_fit_one_item_is_refused_and_the_target_is_not_shortened(self):
        records = [_sequence(1, 1010)] + [_sequence(index, 240) for index in range(2, 30)]
        cohort = self._cohort(records, [self._unit(0, list(range(1, 20)))])
        plan = ch.plan_units(self.arm, cohort, modality="protein")
        self.assertEqual(plan["units"], [])
        self.assertEqual(len(plan["refusals"]), 1)
        reason = plan["refusals"][0]["reason"]
        self.assertIn("never shortened", reason)
        self.assertIn(str(ch.POSITION_BUDGET), reason)

    def test_k_is_set_by_the_budget_and_every_condition_matches_it(self):
        records = [_sequence(index, 150) for index in range(40)]
        cohort = self._cohort(records, [self._unit(0, list(range(1, 30)))])
        plan = ch.plan_units(self.arm, cohort, modality="protein")
        self.assertEqual(len(plan["units"]), 1)
        unit = plan["units"][0]
        self.assertGreaterEqual(unit["k"], 1)
        self.assertLess(unit["k"], ch.MAX_CONTEXT_ITEMS)
        self.assertLessEqual(
            unit["context_tokens"][ch.HOMOLOGUE] + unit["target_tokens"], ch.POSITION_BUDGET
        )
        for condition in (ch.MONO_SHUFFLED, ch.POSITION_ONLY):
            self.assertEqual(
                unit["context_tokens"][condition],
                unit["context_tokens"][ch.HOMOLOGUE],
                f"{condition} is not token-length matched",
            )
        for condition in ch.CONDITIONS:
            if condition == ch.NO_CONTEXT:
                self.assertEqual(unit["conditions"][condition], [])
                continue
            self.assertEqual(len(unit["conditions"][condition]), unit["k"])
        gap = abs(unit["context_tokens"][ch.UNRELATED] - unit["context_tokens"][ch.HOMOLOGUE])
        largest = max(
            len(ch.item_ids(self.arm, records[int(recipe[1])], modality="protein"))
            for recipe in unit["conditions"][ch.HOMOLOGUE]
        )
        self.assertLessEqual(gap, largest, "the matched control is off by more than one item")

    def test_the_mono_shuffle_preserves_the_symbol_multiset_and_destroys_the_order(self):
        record = _sequence(9, 200)
        ids = ch.item_ids(self.arm, record, modality="protein")
        shuffled = ch.shuffled_item_ids(self.arm, record, modality="protein", seed=7)
        self.assertEqual(len(ids), len(shuffled))
        offset = ch.content_offset(self.arm)
        self.assertEqual(ids[:offset], shuffled[:offset])
        self.assertEqual(sorted(ids[offset:]), sorted(shuffled[offset:]))
        self.assertNotEqual(ids, shuffled)

    def test_the_target_token_grid_is_identical_under_every_condition(self):
        records = [_sequence(index, 150) for index in range(40)]
        cohort = self._cohort(records, [self._unit(0, list(range(1, 30)))])
        plan = ch.plan_units(self.arm, cohort, modality="protein")
        unit = plan["units"][0]
        spans = set()
        for condition in ch.CONDITIONS:
            row, start = ch.build_row(
                self.arm,
                unit,
                condition,
                records=records,
                filler=records[-1],
                modality="protein",
            )
            spans.add(tuple(row[start:]))
        self.assertEqual(len(spans), 1, "the scored span moved between conditions")


@pytest.mark.skipif(not PROTGPT2.is_dir(), reason="the ProtGPT2 checkpoint is host-local")
class TheMonoShuffleIsTheRegistrationsOwnOnABpeProteinArm(unittest.TestCase):
    """The clause that could have collided, checked against the arm it would collide on.

    A residue permutation preserves composition exactly and destroys order
    exactly; whether it also preserves ProtGPT2's *token* length -- which the same
    registration requires of every control -- is an empirical question, and the
    answer is that it very nearly does. This test holds that measurement in place,
    because the design's fidelity to the registration depends on it.
    """

    def test_a_residue_shuffle_preserves_composition_and_nearly_preserves_token_length(self):
        arm = _tokenizer_arm("protgpt2")
        generator = np.random.default_rng(3)
        ratios = []
        for index in range(30):
            record = _natural_protein(index)
            natural = len(ch.item_ids(arm, record, modality="protein"))
            shuffled = ch.shuffled_item_ids(arm, record, modality="protein", seed=index)
            residues = np.array(list(record))
            permuted = "".join(residues[generator.permutation(residues.size)])
            self.assertEqual(sorted(permuted), sorted(record))
            ratios.append(len(shuffled) / natural)
        self.assertEqual(ch.SHUFFLE_UNITS["protein"], "residues")
        self.assertLess(
            float(np.median(ratios)),
            1.15,
            "a residue shuffle was measured to inflate ProtGPT2's token count by a "
            "few per cent; a large inflation would mean the registration's residue "
            "shuffle and its token-length clause collide",
        )

    def test_the_shuffled_context_matches_the_homologue_context_token_for_token(self):
        """The token-length clause, on the one arm whose shuffle can break it."""

        arm = _tokenizer_arm("protgpt2")
        records = _natural_proteins(400)
        units = [
            {
                "key": "protein|id_lt_30|retained|000000",
                "modality": "protein",
                "band": "id_lt_30",
                "stratum": "retained",
                "target": 0,
                "group": 0,
                "partners": list(range(1, 41)),
                "partner_lcs": [0] * 40,
                "partner_shared": [0] * 40,
            }
        ]
        cohort = {
            "digest": "unused",
            "protein": {
                "records": records,
                "units": units,
                "related": {"0": []},
                "filler": {"index": 399, "record": records[399]},
            },
        }
        plan = ch.plan_units(arm, cohort, modality="protein")
        unit = plan["units"][0]
        self.assertEqual(
            unit["context_tokens"][ch.MONO_SHUFFLED], unit["context_tokens"][ch.HOMOLOGUE]
        )
        self.assertEqual(plan["shuffle_short_items"], 0)
        self.assertGreater(plan["shuffle_exact_items"], 0)

    def test_the_length_matched_shuffle_keeps_the_composition_it_can(self):
        arm = _tokenizer_arm("protgpt2")
        record = _natural_protein(3)
        wanted = len(ch.item_ids(arm, record, modality="protein"))
        ids, outcome, produced = ch.shuffled_item(
            arm, record, modality="protein", seed=17, target_tokens=wanted
        )
        self.assertGreaterEqual(produced, 1)
        self.assertIn(outcome, ch.SHUFFLE_OUTCOMES)
        self.assertEqual(len(ids), wanted)
        if outcome == "exact":
            decoded = arm.tokenizer.decode(ids[ch.content_offset(arm) :]).replace("\n", "")
            self.assertEqual(sorted(decoded), sorted(record))

    def test_the_text_shuffle_is_at_the_token_level_and_exact(self):
        arm = _tokenizer_arm("gpt2-large")
        passage = "The quick brown fox jumps over the lazy dog, repeatedly and at length."
        ids = ch.item_ids(arm, passage, modality="text")
        shuffled = ch.shuffled_item_ids(arm, passage, modality="text", seed=5)
        self.assertEqual(ch.SHUFFLE_UNITS["text"], "tokens")
        self.assertEqual(len(ids), len(shuffled))
        offset = ch.content_offset(arm)
        self.assertEqual(sorted(ids[offset:]), sorted(shuffled[offset:]))


# ---------------------------------------------------------------- frozen artefacts


class AFrozenArtefactRefusesItsOwnDrift(unittest.TestCase):
    def _cohort(self):
        payload = {
            "pre_registration": ch.PRE_REGISTRATION,
            "draw": {"cohort_draw_seed": 1},
            "protein": {"records": ["AAAA"], "units": [], "filler": {"index": 0}, "groups": [0]},
            "text": {"records": ["hello"], "units": [], "filler": {"index": 0}, "groups": [0]},
        }
        payload["digest"] = ch.cohort_digest(payload)
        return payload

    def test_a_cohort_whose_records_moved_is_refused(self):
        import tempfile

        payload = self._cohort()
        payload["protein"]["records"] = ["AAAC"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cohort.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError) as raised:
                ch.load_cohort(path)
        self.assertIn("drifted", str(raised.exception))

    def test_a_cohort_with_no_digest_is_not_a_frozen_cohort(self):
        import tempfile

        payload = self._cohort()
        payload.pop("digest")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cohort.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                ch.load_cohort(path)

    def test_a_plan_pinned_to_another_cohort_is_refused(self):
        import tempfile

        cohort = self._cohort()
        plan = {
            "pre_registration": ch.PRE_REGISTRATION,
            "arm": "progen2-small",
            "cohort_digest": "0" * 64,
            "units": [],
        }
        plan["digest"] = ch.plan_digest(plan)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaises(ValueError) as raised:
                ch.load_plan(path, cohort=cohort)
        self.assertIn("records this cohort does not hold", str(raised.exception))


# --------------------------------------------------------- known-answer validation


def _rows(n: int, *, homologue: float, unrelated: float, position_only: float, mono: float):
    return [
        {
            "key": f"unit{index}",
            "group": index,
            "band": ch.DECISIVE_BAND,
            "stratum": "retained",
            "k": 4,
            "max_lcs": index % 6,
            "shared_kmers": 0,
            ch.HOMOLOGUE: homologue,
            ch.UNRELATED: unrelated,
            ch.POSITION_ONLY: position_only,
            ch.MONO_SHUFFLED: mono,
            ch.NO_CONTEXT: position_only + 0.05,
            **ch.paired_statistics(
                {
                    ch.HOMOLOGUE: homologue,
                    ch.UNRELATED: unrelated,
                    ch.POSITION_ONLY: position_only,
                    ch.MONO_SHUFFLED: mono,
                }
            ),
        }
        for index in range(n)
    ]


class ThePlantedAnswerIsRecovered(unittest.TestCase):
    """The endpoint returns what it must on cases whose answer is known."""

    def test_a_planted_positive_reaches_the_three_clause_compound(self):
        rows = _rows(60, homologue=2.0, unrelated=2.4, position_only=2.5, mono=2.4)
        block = ch.endpoint_block(rows, resamples=200, seed=ch.BOOTSTRAP_SEED)
        self.assertAlmostEqual(block["auroc"]["mean"], 1.0)
        self.assertAlmostEqual(block["fractional_reduction"]["mean"], 0.4 / 2.5, places=6)
        arm_block = {
            "pooled": block,
            "decisive_stratum": ch.endpoint_block(
                rows[:20], resamples=200, seed=ch.BOOTSTRAP_SEED
            ),
        }
        verdict = ch.gate(arm_block)
        self.assertTrue(all(clause["holds"] for clause in verdict["clauses"].values()))
        self.assertEqual(
            verdict["outcome"], "in_context_relatedness_beyond_composition_and_local_copying"
        )

    def test_a_planted_null_returns_one_half_and_closes_the_line(self):
        rows = _rows(60, homologue=2.4, unrelated=2.4, position_only=2.5, mono=2.4)
        block = ch.endpoint_block(rows, resamples=200, seed=ch.BOOTSTRAP_SEED)
        self.assertAlmostEqual(block["auroc"]["mean"], 0.5)
        self.assertAlmostEqual(block["fractional_reduction"]["mean"], 0.0)
        verdict = ch.gate({"pooled": block, "decisive_stratum": block})
        self.assertEqual(verdict["outcome"], "no_gain_at_this_budget")

    def test_a_gain_that_dies_in_the_decisive_stratum_is_read_as_copying(self):
        strong = _rows(40, homologue=2.0, unrelated=2.4, position_only=2.5, mono=2.4)
        flat = _rows(40, homologue=2.4, unrelated=2.4, position_only=2.5, mono=2.4)
        verdict = ch.gate(
            {
                "pooled": ch.endpoint_block(strong, resamples=200, seed=ch.BOOTSTRAP_SEED),
                "decisive_stratum": ch.endpoint_block(
                    flat, resamples=200, seed=ch.BOOTSTRAP_SEED
                ),
            }
        )
        self.assertEqual(verdict["outcome"], "in_context_copying_and_local_overlap")

    def test_the_gate_refuses_the_k_zero_diagnostic(self):
        block = ch.endpoint_block(
            _rows(20, homologue=2.0, unrelated=2.4, position_only=2.5, mono=2.4),
            resamples=100,
            seed=ch.BOOTSTRAP_SEED,
        )
        with self.assertRaises(ValueError) as raised:
            ch.gate({"pooled": block, "decisive_stratum": block, ch.NO_CONTEXT: 1.0})
        self.assertIn("never the effect", str(raised.exception))

    def test_the_row_count_and_the_resampling_unit_count_are_not_one_number(self):
        """They were, and the floor record's copy silently won.

        A stratum of sixty units drawn from fifty-six near-duplicate groups
        reported ``n_units: 56``, which is the number a reader takes for the
        sample size and is not it.
        """

        record = ch.group_bootstrap_mean(
            [1.0] * 20, [index // 2 for index in range(20)], resamples=100
        )
        self.assertEqual(record["n_rows"], 20)
        self.assertEqual(record["n_groups"], 10)
        self.assertEqual(record["n_units"], 10, "the floor counts resampling units")

    def test_the_group_bootstrap_refuses_below_the_declared_unit_floor(self):
        record = ch.group_bootstrap_mean([1.0, 2.0, 3.0], [0, 0, 1], resamples=50)
        self.assertTrue(record["degenerate"])
        self.assertIsNone(record["ci95"])

    def test_the_terciles_split_a_tied_band_into_three_non_empty_parts(self):
        rows = [{"key": f"u{index}", "max_lcs": 5} for index in range(30)]
        split = ch.terciles(rows)
        self.assertEqual(sorted(len(part) for part in split.values()), [10, 10, 10])


@pytest.mark.skipif(not PROGEN_SMALL.is_dir(), reason="the ProGen2 checkpoint is host-local")
class TheScoringPathReproducesAPublishedResult(unittest.TestCase):
    """Kantroo et al.'s self-copy collapse, on the checkpoint they measured's rung.

    The strongest known answer available: appending a copy of a sequence to itself
    collapses its likelihood. If this campaign's rendering, concatenation and
    scored span were wrong, this would not appear -- and it appears for reasons
    that have nothing to do with homology, which is why the whole control
    structure exists.
    """

    def test_a_self_copy_context_collapses_the_target_nll_and_an_unrelated_one_does_not(self):
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer

        spec = PANEL["progen2-small"]
        tokenizer = AutoTokenizer.from_pretrained(str(spec.path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(spec.path), trust_remote_code=True, torch_dtype=torch.float32
        ).eval()
        arm = Arm(spec=spec, model=model, tokenizer=tokenizer, device="cpu", dtype="float32")
        target = _sequence(21, 180)
        other = _sequence(22, 180)

        def score(context: str | None) -> float:
            ids = ch.item_ids(arm, target, modality="protein")
            start = ch.content_offset(arm)
            row = list(ids)
            if context is not None:
                prefix = ch.item_ids(arm, context, modality="protein")
                row = prefix + row
                start += len(prefix)
            tensor = torch.tensor([row], dtype=torch.long)
            with torch.no_grad():
                logits = model(input_ids=tensor).logits
            logprobs = F.log_softmax(logits[0, start - 1 : len(row) - 1].float(), dim=-1)
            targets = tensor[0, start : len(row)]
            return float(-logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean())

        alone = score(None)
        copied = score(target)
        unrelated = score(other)
        self.assertLess(copied, alone - 1.0, "the published self-copy collapse did not appear")
        self.assertLess(copied, unrelated - 1.0)
        self.assertLess(abs(unrelated - alone), 1.0)


if __name__ == "__main__":
    unittest.main()
