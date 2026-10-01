"""Conditions the generation-and-control gate must always hold.

Three of these exist because the gate's conclusion is worthless without them.
A matched generator that does not carry the statistics it declares is not a
control, so :class:`MatchedGeneratorsCarryWhatTheyDeclare` measures that rather
than trusting the code that built it. A denominator that quietly drops the
attempts nothing recognised turns a rate into a survivor rate, so
:class:`TheDenominatorKeepsEveryAttempt` drives an unassigned attempt through
the pipeline and requires it to still be there. And a control chosen after the
model's outcome is known is not a control at all, so
:class:`NothingSelectsOnModelSuccess` permutes every profile outcome in the
input and requires the selection and every generated sequence to be unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.generation import generative_control as gc  # noqa: E402
from src.capability.core.amino_acids import AA20  # noqa: E402


from scripts.capability.entrypoints import stage_path

def _load_stage(filename: str):
    """Import a stage whose module name starts with a digit, as the stages do."""

    path = stage_path(REPO_ROOT, filename)
    spec = importlib.util.spec_from_file_location(f"_gencontrol_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE = _load_stage("generative_control.py")


def _uniform_counts(k: int) -> np.ndarray:
    return np.full(len(AA20) ** k, 1000, dtype=np.int64)


def _biased_counts(k: int, *, favour: str = "A") -> np.ndarray:
    """A k-mer table whose conditional prefers ``favour`` after every context."""

    width = len(AA20)
    table = np.full((width ** (k - 1), width), 10, dtype=np.int64)
    table[:, AA20.index(favour)] = 10_000
    return table.reshape(-1)


def _order_three_corpus(rng: np.random.Generator, *, sequences: int,
                        length: int) -> list[str]:
    """A synthetic corpus with genuine third-order structure.

    ``P(x | a, b, c)`` puts weight 10 on ``(a + b + c) mod 20`` and weight 1 on
    every other residue, so the process depends on all three preceding
    positions and on no fewer. An order-2 sampler fitted to its 3-mer counts can
    match it at order 2 and cannot match it at order 3, which is the property
    the order axis rests on.
    """

    width = len(AA20)
    data = np.empty((sequences, length), dtype=np.int64)
    data[:, :3] = rng.integers(0, width, size=(sequences, 3))
    alpha = 9 / 29
    for position in range(3, length):
        target = (data[:, position - 3] + data[:, position - 2]
                  + data[:, position - 1]) % width
        take = rng.random(sequences) < alpha
        data[:, position] = np.where(take, target, rng.integers(0, width, size=sequences))
    return [gc.decode(row) for row in data]


def _count_kmers(sequences: list[str], k: int) -> np.ndarray:
    width = len(AA20)
    counts = np.zeros(width ** k, dtype=np.int64)
    for sequence in sequences:
        indices = gc.encode(sequence)
        if len(indices) < k:
            continue
        span = len(indices) - k + 1
        key = np.zeros(span, dtype=np.int64)
        for offset in range(k):
            key = key * width + indices[offset:offset + span]
        np.add.at(counts, key, 1)
    return counts


def _fake_samplers() -> dict[int, gc.MarkovSampler]:
    counts = {1: _uniform_counts(1), 3: _uniform_counts(3), 5: _uniform_counts(5)}
    return {order: gc.markov_from_counts(counts, order) for order in gc.MARKOV_ORDERS}


def _attempt(identifier: str, sequence: str, *, group: str, hit: bool,
             valid: bool = True, identity: float | None = 42.0) -> dict:
    return {"id": identifier, "sequence": sequence, "valid_aa20": valid,
            "near_duplicate_group": group, "any_profile_hit": hit,
            "pfam_families": ["PF00001"] if hit else [],
            "reference_identity": identity,
            "reference_search_status": "aligned_query_coverage_available",
            "arm": "testarm", "condition": "unconditioned", "role": "generation",
            "class_key": None, "stop_status": "native_terminal"}


class MatchedGeneratorsCarryWhatTheyDeclare(unittest.TestCase):
    def test_a_shuffle_keeps_the_length_and_the_residue_multiset_exactly(self):
        rng = np.random.default_rng(0)
        parents = ["MKTAYIAKQRQISFVKSHFSRQ", "AAAACCCCDDDD", "M"]
        controls = [gc.composition_shuffle(rng, parent) for parent in parents]
        receipt = gc.qualify_shuffle(parents, controls)
        self.assertEqual(receipt["length_failures"], 0)
        self.assertEqual(receipt["composition_failures"], 0)
        self.assertTrue(receipt["qualified"])

    def test_a_shuffle_that_changed_a_residue_is_refused(self):
        # The negative path: a control whose composition moved is not a
        # composition-matched control, and the receipt must say so rather than
        # pass because the code that made it intended otherwise.
        receipt = gc.qualify_shuffle(["MKTA"], ["MKTG"])
        self.assertEqual(receipt["composition_failures"], 1)
        self.assertFalse(receipt["qualified"])

    def test_an_order_k_sampler_emits_from_the_order_k_conditional_it_was_given(self):
        # The sampler's declared statistic, checked on the sampler itself rather
        # than on the receipt: the realized frequency of each residue after a
        # fixed two-residue context must approach the conditional row the counts
        # define. A sampler that drifted from its own table would make every
        # order label in this gate wrong, and no downstream receipt would see it.
        counts = {3: _biased_counts(3, favour="W")}
        sampler = gc.markov_from_counts(counts, 2)
        expected = sampler.conditional[AA20.index("W") * len(AA20) + AA20.index("W")]
        emitted = sampler.emit(np.random.default_rng(7), 200_000)
        indices = gc.encode(emitted)
        width = len(AA20)
        context = indices[:-2] * width + indices[1:-1]
        following = indices[2:]
        probe = AA20.index("W") * width + AA20.index("W")
        mask = context == probe
        self.assertGreater(int(mask.sum()), 200)
        realized = np.bincount(following[mask], minlength=width) / int(mask.sum())
        self.assertLess(float(np.abs(realized - expected).sum()), 0.1)
        self.assertAlmostEqual(float(expected[AA20.index("W")]),
                               10_000 / (10_000 + 19 * 10), places=6)

    def test_an_order_k_cohort_scores_below_its_reference_one_order_above(self):
        # The order-axis property, end to end on synthetic counts with genuine
        # third-order structure: a cohort emitted from the order-2 conditional
        # scores like a corpus draw at order 2 and strictly below it at order 3.
        rng = np.random.default_rng(5)
        width = len(AA20)
        corpus = _order_three_corpus(rng, sequences=600, length=1500)
        counts = {3: _count_kmers(corpus, 3), 4: _count_kmers(corpus, 4)}
        sampler = gc.markov_from_counts(counts, 2)
        cohort = [sampler.emit(rng, 800) for _ in range(300)]
        reference = _order_three_corpus(rng, sequences=300, length=800)
        profile, reference_profile = {}, {}
        for order in (2, 3):
            table = gc.conditional_table(counts, order)
            profile[order] = gc.conditional_log_likelihood(cohort, table, order)[0]
            reference_profile[order] = gc.conditional_log_likelihood(
                reference, table, order)[0]
        receipt = gc.qualify_markov(profile, reference_profile, 2)
        self.assertLess(abs(receipt["gap_at_declared_order"]), 0.05, receipt)
        self.assertLess(receipt["gap_one_order_above"], -0.05, receipt)
        self.assertEqual(width, 20)

    def test_a_sampler_that_matches_the_reference_at_the_next_order_is_refused(self):
        # A cohort indistinguishable from a corpus draw one order above its
        # declared order is carrying more than it declares; it is not a point on
        # this axis and the receipt refuses it.
        equal = {2: -2.5, 3: -2.4}
        receipt = gc.qualify_markov(equal, equal, 2)
        self.assertFalse(receipt["is_not_a_higher_order_sampler"])
        self.assertFalse(receipt["qualified"])

    def test_the_amendment_reads_the_share_of_the_reference_step_captured(self):
        # The pre-declared condition and the amendment are both recorded, because
        # the declared 0.02-nat margin is larger than one order step of this
        # corpus below order 4 and the record must not hide that the primary
        # condition became unreachable for a reason that is not the sampler's.
        receipt = gc.qualify_markov({0: -2.8909, 1: -2.8973},
                                    {0: -2.8876, 1: -2.8814}, 0)
        self.assertAlmostEqual(receipt["reference_order_step_nats"], 0.0062, places=4)
        self.assertLess(receipt["cohort_order_step_nats"], 0.0)
        self.assertLess(receipt["order_step_share_captured"], 0.0)
        self.assertFalse(receipt["qualified"])
        self.assertTrue(receipt["qualified_under_the_scale_free_amendment"])

    def test_the_amendment_still_refuses_a_sampler_that_captures_the_next_order(self):
        receipt = gc.qualify_markov({0: -2.8876, 1: -2.8814},
                                    {0: -2.8876, 1: -2.8814}, 0)
        self.assertGreater(receipt["order_step_share_captured"], 0.9)
        self.assertFalse(receipt["qualified_under_the_scale_free_amendment"])

    def test_a_sampler_off_its_declared_order_is_refused(self):
        receipt = gc.qualify_markov({2: -3.0, 3: -3.5}, {2: -2.5, 3: -2.4}, 2)
        self.assertGreater(abs(receipt["gap_at_declared_order"]),
                           gc.ORDER_STATISTIC_TOLERANCE)
        self.assertFalse(receipt["qualified"])

    def test_an_unseen_context_is_refused_rather_than_smoothed_for_sampling(self):
        counts = {3: np.zeros(len(AA20) ** 3, dtype=np.int64)}
        counts[3][0] = 5
        with self.assertRaises(ValueError):
            gc.markov_from_counts(counts, 2)

    def test_a_fragment_is_a_contiguous_substring_at_the_exact_length(self):
        record = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
        rng = np.random.default_rng(3)
        fragment = gc.corpus_fragment(rng, record, 12)
        receipt = gc.qualify_fragment([fragment], [record], [12])
        self.assertEqual(receipt["containment_failures"], 0)
        self.assertEqual(receipt["length_failures"], 0)
        self.assertTrue(receipt["qualified"])

    def test_a_fragment_that_is_not_in_its_named_record_is_refused(self):
        receipt = gc.qualify_fragment(["WWWW"], ["MKTAYIAK"], [4])
        self.assertEqual(receipt["containment_failures"], 1)
        self.assertFalse(receipt["qualified"])

    def test_a_donor_shorter_than_the_request_is_a_failure_not_a_pad(self):
        with self.assertRaises(ValueError):
            gc.corpus_fragment(np.random.default_rng(0), "MKT", 10)

    def test_the_hydropathy_generator_tracks_its_target(self):
        background = np.full(len(AA20), 1.0 / len(AA20))
        rng = np.random.default_rng(11)
        targets = [-1.5, -0.4, 0.0, 0.8]
        controls, lengths = [], []
        for target in targets:
            emitted, _, reachable = gc.hydropathy_matched(rng, background, target, 4000)
            self.assertTrue(reachable)
            controls.append(emitted)
            lengths.append(4000)
        receipt = gc.qualify_hydropathy(controls, targets, lengths)
        self.assertTrue(receipt["qualified"], receipt)
        self.assertLess(receipt["mean_absolute_deviation_kd"], gc.HYDROPATHY_TOLERANCE)

    def test_an_unreachable_hydropathy_target_is_declared_not_silently_matched(self):
        background = np.full(len(AA20), 1.0 / len(AA20))
        _, _, reachable = gc.hydropathy_tilt(background, 9.0)
        self.assertFalse(reachable)

    def test_a_natural_draw_outside_its_declared_band_is_refused(self):
        bounds = {"16_128": (16, 128)}
        self.assertTrue(gc.qualify_natural(["A" * 40], ["16_128"], bounds)["qualified"])
        self.assertFalse(gc.qualify_natural(["A" * 400], ["16_128"], bounds)["qualified"])


class TheDenominatorKeepsEveryAttempt(unittest.TestCase):
    def setUp(self):
        self.samplers = _fake_samplers()
        self.background = np.full(len(AA20), 1.0 / len(AA20))
        self.pool = {name: ["M" + "A" * (low + 1)] for name, low, _ in STAGE.STRATA}
        self.donor = lambda rng, length: ("donor", "MKTAYIAKQRQ" * 400)

    def _record(self, row):
        return STAGE.control_record("testarm__unconditioned",
                                    {"arm": "testarm", "condition": "unconditioned",
                                     "campaign": "TEST"}, row,
                                    samplers=self.samplers, composition=self.background,
                                    donor=self.donor, natural_pool=self.pool)

    def test_an_unsearchable_attempt_stays_in_the_row_set_with_empty_controls(self):
        row = _attempt("a1", "", group="g0", hit=False, valid=False)
        record = self._record(row)
        self.assertFalse(record["parent_searchable"])
        self.assertEqual(set(record["sequences"]), set(STAGE.COHORTS))
        self.assertTrue(all(value == "" for value in record["sequences"].values()))
        self.assertIn("reason", record["provenance"])

    def test_a_non_canonical_attempt_is_unsearchable_rather_than_repaired(self):
        record = self._record(_attempt("a2", "MKTXAYI", group="g0", hit=False))
        self.assertFalse(record["parent_searchable"])
        self.assertEqual(record["parent_length"], 0)

    def test_an_unassigned_attempt_counts_against_the_model_and_the_control(self):
        # The negative path the requirement names: three attempts, one of which
        # the oracle recognises on neither side, must give a denominator of
        # three and a model rate of one third -- not a denominator of two.
        model = [True, False, False]
        control = [False, False, False]
        contrast = gc.paired_rate_contrast(model, control, ["g0", "g1", "g2"])
        self.assertEqual(contrast["n_attempts"], 3)
        self.assertAlmostEqual(contrast["model_rate"], 1 / 3)
        self.assertAlmostEqual(contrast["control_rate"], 0.0)
        self.assertAlmostEqual(contrast["difference"], 1 / 3)

    def test_a_wilson_interval_uses_the_all_attempt_denominator(self):
        lower, upper = gc.wilson_interval(0, 800)
        self.assertAlmostEqual(lower, 0.0, places=12)
        self.assertLess(upper, 0.01)
        self.assertGreater(upper, 0.0)

    def test_a_cluster_of_near_duplicates_contributes_once(self):
        # Eight attempts in one group and eight in eight groups must not carry
        # the same weight: the safeguard the conditioned-generation campaign
        # applied is the unit here too.
        one_group = gc.paired_rate_contrast([True] * 8 + [False] * 8,
                                            [False] * 16, ["g0"] * 8 + [f"h{i}" for i in range(8)])
        self.assertEqual(one_group["n_clusters"], 9)
        self.assertEqual(one_group["n_attempts"], 16)

    def test_a_contrast_refuses_misaligned_vectors(self):
        with self.assertRaises(ValueError):
            gc.paired_rate_contrast([True], [True, False], ["g0", "g1"])

    def test_no_interval_is_reported_below_eight_clusters(self):
        contrast = gc.paired_rate_contrast([True] * 4, [False] * 4, [f"g{i}" for i in range(4)])
        self.assertIsNone(contrast["ci95"])
        self.assertEqual(contrast["interval_status"], "fewer_than_eight_sequence_groups")


class TheEndpointKeepsTheLedgerDenominatorEndToEnd(unittest.TestCase):
    """One cell driven through the analysis, with two attempts nothing can search.

    The condition is the whole point of the gate's accounting: the denominator
    the endpoint reports must be the ledger's, not the searchable subset's, and
    an attempt that produced nothing must count against the model and against
    every control alike.
    """

    def setUp(self):
        self.samplers = _fake_samplers()
        self.background = np.full(len(AA20), 1.0 / len(AA20))
        self.pool = {name: ["M" + "A" * (low + 1)] for name, low, _ in STAGE.STRATA}
        donor = lambda rng, length: ("donor", "MKTAYIAKQRQ" * 400)  # noqa: E731
        rows = [_attempt(f"e{index}", "MKTAYIAKQRQISFVKSHFSRQ", group=f"g{index}",
                         hit=index < 3)
                for index in range(8)]
        rows += [_attempt("e8", "", group="g8", hit=False, valid=False),
                 _attempt("e9", "MKTXAYI", group="g9", hit=False)]
        self.records = [
            STAGE.control_record("testarm__unconditioned",
                                 {"arm": "testarm", "condition": "unconditioned",
                                  "campaign": "TEST"}, row,
                                 samplers=self.samplers, composition=self.background,
                                 donor=donor, natural_pool=self.pool)
            for row in rows]
        # The oracle recognises the model's first three products and one
        # fragment, and nothing else; no entry exists for an unsearchable
        # attempt, which is what the real oracle returns for one.
        self.oracle = {}
        for index in range(3):
            self.oracle[f"testarm__unconditioned|generated|e{index}"] = {
                "families": ["PF00001"], "best_profile_coverage": 0.95}
        self.oracle["testarm__unconditioned|fragment|e0"] = {
            "families": ["PF00002"], "best_profile_coverage": 0.10}

    def test_the_denominator_is_the_ledger_and_not_the_searchable_subset(self):
        cell = STAGE_ANALYSIS.analyse_cell(self.records, self.oracle,
                                          {"shuffle", "fragment"},
                                          {"shuffle", "fragment"}, 0.80)
        self.assertEqual(cell["n_attempts"], 10)
        self.assertEqual(cell["n_searchable"], 8)
        block = cell["endpoints"]["any_family"]
        self.assertEqual(block["model_denominator"], 10)
        self.assertEqual(block["model_successes"], 3)
        self.assertAlmostEqual(block["model_rate"], 0.3)
        self.assertEqual(block["controls"]["fragment"]["n_attempts"], 10)
        self.assertEqual(block["controls"]["fragment"]["control_successes"], 1)
        self.assertEqual(block["controls"]["shuffle"]["control_successes"], 0)

    def test_the_complete_domain_endpoint_reads_coverage_and_not_recognition(self):
        cell = STAGE_ANALYSIS.analyse_cell(self.records, self.oracle,
                                          {"shuffle", "fragment"},
                                          {"shuffle", "fragment"}, 0.80)
        block = cell["endpoints"]["complete_domain"]
        self.assertEqual(block["model_successes"], 3)
        self.assertEqual(block["controls"]["fragment"]["control_successes"], 0)

    def test_an_unqualified_cohort_is_reported_but_not_decisive(self):
        cell = STAGE_ANALYSIS.analyse_cell(self.records, self.oracle, {"shuffle"},
                                           {"shuffle", "fragment"}, 0.80)
        block = cell["endpoints"]["any_family"]
        self.assertIn("fragment", block["controls"])
        self.assertFalse(block["controls"]["fragment"]["qualified"])
        self.assertNotIn("fragment", block["verdict"]["decisive_controls"])
        self.assertIn("fragment",
                      block["verdict_under_the_scale_free_amendment"]["decisive_controls"])

    def test_the_oracle_replay_is_checked_against_the_frozen_ledger(self):
        cell = STAGE_ANALYSIS.analyse_cell(self.records, self.oracle,
                                           {"shuffle"}, {"shuffle"}, 0.80)
        self.assertEqual(
            cell["instrument_replay_disagreements_against_the_frozen_ledger"], 0)
        broken = [dict(record) for record in self.records]
        broken[5]["ledger_any_profile_hit"] = True
        cell = STAGE_ANALYSIS.analyse_cell(broken, self.oracle, {"shuffle"},
                                           {"shuffle"}, 0.80)
        self.assertEqual(
            cell["instrument_replay_disagreements_against_the_frozen_ledger"], 1)


class NothingSelectsOnModelSuccess(unittest.TestCase):
    def setUp(self):
        self.samplers = _fake_samplers()
        self.background = np.full(len(AA20), 1.0 / len(AA20))
        self.pool = {name: ["M" + "A" * (low + 1)] for name, low, _ in STAGE.STRATA}
        self.donor = lambda rng, length: ("donor", "MKTAYIAKQRQ" * 400)
        self.rows = [
            _attempt(f"a{index}", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"[: 20 + index],
                     group=f"g{index % 4}", hit=index % 2 == 0)
            for index in range(16)
        ]

    def _build(self, rows):
        return [STAGE.control_record("testarm__unconditioned",
                                     {"arm": "testarm", "condition": "unconditioned",
                                      "campaign": "TEST"}, row,
                                     samplers=self.samplers, composition=self.background,
                                     donor=self.donor, natural_pool=self.pool)
                for row in rows]

    def test_every_control_sequence_is_unchanged_when_the_outcomes_are_permuted(self):
        before = self._build(self.rows)
        permuted = []
        for index, row in enumerate(self.rows):
            copy = dict(row)
            copy["any_profile_hit"] = not row["any_profile_hit"]
            copy["pfam_families"] = ["PF99999"] if copy["any_profile_hit"] else []
            permuted.append(copy)
        after = self._build(permuted)
        self.assertEqual([record["sequences"] for record in before],
                         [record["sequences"] for record in after])
        self.assertNotEqual([record["ledger_any_profile_hit"] for record in before],
                            [record["ledger_any_profile_hit"] for record in after])

    def test_the_conditioned_subsample_is_unchanged_when_the_outcomes_are_permuted(self):
        rows = []
        for index in range(64):
            row = _attempt(f"c{index:03d}", "MKTAYIAKQRQ", group=f"g{index}",
                           hit=index < 8)
            row["arm"] = "zymctrl"
            row["condition"] = "requested"
            row["class_key"] = f"ec{index % 4}"
            rows.append(row)
        path = Path(self.enterContext(_temporary_directory())) / "attempts.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        spec = {"arm": "zymctrl", "condition": "requested", "campaign": "EXP-R2-232",
                "path": path, "population": 64, "subsample": 16}
        selected, accounting = STAGE.select_rows(spec)
        flipped = [{**row, "any_profile_hit": not row["any_profile_hit"]} for row in rows]
        path.write_text("".join(json.dumps(row) + "\n" for row in flipped), encoding="utf-8")
        reselected, _ = STAGE.select_rows(spec)
        self.assertEqual([row["id"] for row in selected], [row["id"] for row in reselected])
        self.assertEqual(accounting["selected"], 16)
        self.assertEqual(accounting["population"], 64)
        self.assertAlmostEqual(accounting["inclusion_probability"], 0.25)

    def test_the_selection_rule_names_what_it_consults(self):
        rows = [_attempt(f"d{index}", "MKTAYIAKQRQ", group=f"g{index}", hit=False)
                for index in range(4)]
        path = Path(self.enterContext(_temporary_directory())) / "ledger.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        selected, accounting = STAGE.select_rows(
            {"arm": "testarm", "condition": "unconditioned", "campaign": "TEST",
             "path": path, "population": None, "subsample": None})
        self.assertEqual(len(selected), 4)
        self.assertEqual(accounting["inclusion_probability"], 1.0)
        self.assertIn("every attempt", accounting["selection_rule"])


class TheOracleIsReadAtOneGranularity(unittest.TestCase):
    DOMAIN_ROW = ("Kinase              PF00069.30   260 q00000001            -"
                  "            300   1e-40  140.0   0.0   1   1   1e-42   1e-38  "
                  "130.0   0.0     1   250    10   270     8   275 0.95 Protein kinase")

    def test_profile_coverage_is_the_aligned_share_of_the_model_length(self):
        path = Path(self.enterContext(_temporary_directory())) / "x.domtbl"
        path.write_text("# header\n" + self.DOMAIN_ROW + "\n", encoding="utf-8")
        domains = gc.parse_domain_table(path)
        self.assertIn("q00000001", domains)
        entry = domains["q00000001"][0]
        self.assertEqual(entry["accession_unversioned"], "PF00069")
        self.assertEqual(entry["target_length"], 260)
        self.assertAlmostEqual(entry["profile_coverage"], 250 / 260)
        self.assertAlmostEqual(gc.best_profile_coverage(domains["q00000001"]), 250 / 260)

    def test_no_assignment_has_no_coverage_rather_than_a_zero(self):
        self.assertIsNone(gc.best_profile_coverage([]))

    def _workspace(self, sequence_accession: str, domain_accession: str) -> Path:
        directory = Path(self.enterContext(_temporary_directory()))
        (directory / "oracle").mkdir()
        (directory / "shards").mkdir()
        fasta = directory / "shards" / "s.fasta"
        fasta.write_text(">q00000001\nMKTAYIAKQRQ\n", encoding="utf-8")
        digest = hashlib.sha256(fasta.read_bytes()).hexdigest()
        (directory / "build_manifest.json").write_text(json.dumps(
            {"shards": [{"shard": 0, "path": str(fasta), "n": 1, "sha256": digest}]}),
            encoding="utf-8")
        (directory / "oracle" / "s.tbl.done").write_text(
            json.dumps({"command": [], "fasta_sha256": digest}), encoding="utf-8")
        (directory / "query_names.json").write_text(
            json.dumps({"cell|generated|a1": "q00000001"}), encoding="utf-8")
        (directory / "oracle" / "s.tbl").write_text(
            f"# header\nKinase {sequence_accession} q00000001 - 1e-40 140.0 0.0 "
            "1e-42 130.0 0.0 1.0 1 1 0 1 1 1 1 kinase\n"
            f"Other PF12728.14 q00000001 - 5.9e-06 27.1 0.0 8.4e-06 26.6 0.0 1.3 "
            "1 1 0 1 1 0 0 other\n", encoding="utf-8")
        (directory / "oracle" / "s.domtbl").write_text(
            "# header\n" + self.DOMAIN_ROW.replace("PF00069.30", domain_accession) + "\n",
            encoding="utf-8")
        return directory

    def test_a_sequence_level_family_without_a_domain_row_is_counted_not_refused(self):
        # HMMER reports a family whose full-sequence score clears GA1 and writes a
        # domain row only when one domain also clears GA2, so the two tables are
        # different sets by construction. The coarser read stays authoritative for
        # the any-family endpoint and the gap is a counted quantity.
        directory = self._workspace("PF00069.30", "PF00069.30")
        result = gc.collect_oracle(directory)["cell|generated|a1"]
        self.assertEqual(result["families"], ["PF00069", "PF12728"])
        self.assertEqual(result["families_with_a_domain_row"], ["PF00069"])
        self.assertEqual(result["n_families_without_a_domain_row"], 1)

    def test_a_domain_family_the_sequence_table_does_not_carry_is_refused(self):
        with self.assertRaises(RuntimeError):
            gc.collect_oracle(self._workspace("PF00069.30", "PF00071.30"))

    def test_a_shard_without_a_terminal_receipt_refuses_the_endpoint(self):
        # A truncated table on disk would read as a low recognition rate, so an
        # unfinished scan must refuse rather than be analysed.
        directory = self._workspace("PF00069.30", "PF00069.30")
        (directory / "oracle" / "s.tbl.done").unlink()
        with self.assertRaises(RuntimeError) as caught:
            gc.collect_oracle(directory)
        self.assertIn("terminal receipt", str(caught.exception))

    def test_a_shard_whose_query_file_moved_refuses_the_endpoint(self):
        directory = self._workspace("PF00069.30", "PF00069.30")
        (directory / "oracle" / "s.tbl.done").write_text(
            json.dumps({"command": [], "fasta_sha256": "0" * 64}), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            gc.collect_oracle(directory)


class TheVerdictExcludesWhatItMustExclude(unittest.TestCase):
    def _contrast(self, lower: float) -> dict:
        return {"ci97_5": [lower, lower + 0.1]}

    def test_an_unqualified_control_does_not_decide_the_verdict(self):
        contrasts = {name: self._contrast(0.05) for name in
                     ("shuffle", "markov_0", "markov_2", "markov_4", "fragment",
                      "hydropathy", "natural")}
        contrasts["markov_4"] = self._contrast(-0.2)
        qualified = {"shuffle", "markov_0", "markov_2", "fragment", "hydropathy", "natural"}
        result = STAGE_ANALYSIS.verdict(contrasts, qualified)
        self.assertEqual(result["label"],
                         "satisfied_beyond_every_qualified_matched_generator")
        self.assertIn("markov_4", result["excluded_unqualified"])

    def test_a_reference_cohort_does_not_decide_the_verdict(self):
        contrasts = {name: self._contrast(0.05) for name in
                     ("shuffle", "markov_0", "markov_2", "markov_4", "fragment",
                      "hydropathy")}
        contrasts["natural"] = self._contrast(-0.5)
        qualified = {"shuffle", "markov_0", "markov_2", "markov_4", "fragment",
                     "hydropathy", "natural"}
        result = STAGE_ANALYSIS.verdict(contrasts, qualified)
        self.assertEqual(result["label"],
                         "satisfied_beyond_every_qualified_matched_generator")
        self.assertEqual(result["excluded_as_reference"], ["natural"])

    def test_a_qualified_control_the_model_does_not_beat_narrows_the_label(self):
        contrasts = {name: self._contrast(0.05) for name in
                     ("shuffle", "markov_0", "markov_2", "markov_4", "hydropathy",
                      "natural")}
        contrasts["fragment"] = self._contrast(-0.3)
        qualified = {"shuffle", "markov_0", "markov_2", "markov_4", "fragment",
                     "hydropathy", "natural"}
        result = STAGE_ANALYSIS.verdict(contrasts, qualified)
        self.assertEqual(
            result["label"],
            "satisfied_beyond_some_but_not_all_qualified_matched_generators")
        self.assertEqual(result["not_beaten"], ["fragment"])

    def test_a_missing_interval_leaves_the_endpoint_unresolved(self):
        contrasts = {name: {"ci97_5": None} for name in
                     ("shuffle", "markov_0", "markov_2", "markov_4", "fragment",
                      "hydropathy", "natural")}
        result = STAGE_ANALYSIS.verdict(
            contrasts, {"shuffle", "markov_0", "markov_2", "markov_4", "fragment",
                        "hydropathy", "natural"})
        self.assertEqual(result["label"], "unresolved_interval_not_estimable")

    def test_an_inherited_calibration_leaves_the_structure_endpoint_unresolved(self):
        directory = Path(self.enterContext(_temporary_directory()))
        path = directory / "generation_biology_analysis.json"
        path.write_text(json.dumps({
            "phase": "main",
            "arms": {
                "own": {"mean": 6.1, "ci95": [3.8, 8.6], "ci97_5": [3.5, 8.9],
                        "unit": "class", "n_classes": 15, "calibration_attained": True,
                        "interpretation": "positive"},
                "inherited": {"mean": 21.9, "ci95": [17.0, 26.7], "ci97_5": [16.4, 27.4],
                              "unit": "sampled_near_duplicate_sequence_group",
                              "n_clusters": 128, "calibration_attained": True,
                              "calibration_scope":
                                  "shared_R232_natural_control_panel_not_native_class_calibration",
                              "interpretation": "positive"},
            }}), encoding="utf-8")
        result = STAGE_ANALYSIS.structure_passthrough([path])
        labels = {cell["arm"]: cell["verdict"] for cell in result["cells"]}
        self.assertEqual(labels["inherited"],
                         "unresolved_calibration_inherited_from_a_two_arm_panel")
        self.assertEqual(labels["own"],
                         "licensed_paired_within_sequence_contrast_on_own_controls")
        self.assertEqual(result["n_inherited"], 1)
        self.assertEqual(result["n_own_controls"], 1)


class TheIdentityStrataAreDeclaredNotInferred(unittest.TestCase):
    def test_no_reported_alignment_is_its_own_stratum_and_not_zero_identity(self):
        self.assertEqual(
            STAGE_ANALYSIS.band_of(None, "no_reported_alignment"), "no_reported_alignment")
        self.assertEqual(STAGE_ANALYSIS.band_of(0.0, "aligned"), "under_30")
        self.assertEqual(STAGE_ANALYSIS.band_of(100.0, "aligned"), "70_and_over")
        self.assertEqual(STAGE_ANALYSIS.band_of(30.0, "aligned"), "30_to_50")

    def test_an_unsearched_attempt_is_not_a_distance_measurement(self):
        self.assertEqual(STAGE_ANALYSIS.band_of(None, "empty_not_searched"), "not_searched")
        self.assertEqual(STAGE_ANALYSIS.band_of(None, None), "reference_search_missing")


def _temporary_directory():
    import tempfile

    return tempfile.TemporaryDirectory()


STAGE_ANALYSIS = _load_stage("analyse_generative_control.py")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
