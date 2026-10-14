"""The evidence-to-manuscript pipeline's refusals, each with its negative path.

The defects this pipeline exists to stop were all produced by hand
transcription: a count copied against the wrong denominator, two estimates on
different supports differenced implicitly, an interval quoted without the unit
it was resampled over, and a point estimate quoted bare. Each of those is a
test here, on fixtures rather than on the live manuscript, so the test says
what the checker refuses and not what today's draft happens to contain.

One test does run against the live manuscript, and it asserts only that the
checker completes and writes a report: the manuscript is another agent's and
its findings change under it.
"""
from pathlib import Path
import importlib.util
import json
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.evidence_ledger import (  # noqa: E402
    Artifacts,
    Ledger,
    LedgerError,
    Quantity,
    checked_sum,
    resolve,
)
from scripts.capability.reporting import publication_paths  # noqa: E402


def _module(name: str):
    from scripts.capability.entrypoints import stage_path
    path = stage_path(REPO_ROOT, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CHECKER = _module("check_manuscript_evidence")
BUILDER = _module("build_derived_numbers")


PREAMBLE = "\\documentclass{sn-jnl}\n\\begin{document}\n"


def write_manuscript(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "main.tex").write_text(PREAMBLE + body + "\n\\end{document}\n")
    return directory


def table(quantities: list[dict], supports: dict[str, str]) -> dict[str, object]:
    return {
        "schema": "manuscript_derived_numbers_v1",
        "scope": "fixture",
        "verdict_vocabulary": ["measurement_limited", "not_detected", "supported", "unresolved"],
        "supports": supports,
        "counts": {"quantities": len(quantities), "by_family": {}, "artifact_backed": len(quantities),
                   "gate_record_backed": 0, "provisional": 0, "gaps": 0},
        "source_sha256": {},
        "quantities": quantities,
        "gaps": [],
    }


def entry(**overrides) -> dict:
    row = {
        "id": "fixture/progen3-3b/profile_contrast", "claim": "a paired profile contrast",
        "family": "fixture", "value": 0.05678, "unit": "dimensionless Spearman", "kind": "estimate",
        "support_id": "support_a", "interval_low": 0.03601, "interval_high": 0.07894,
        "interval_kind": "95% percentile", "resampling_unit": "wild-type cluster",
        "resampling_draws": 2000, "no_interval_reason": None, "seed_set": [20260923],
        "level": "likelihood", "provisional": False, "verdict": "supported",
        "weighting_convention": None, "source_kind": "artifact",
        "source_path": "results/fixture.json", "source_sha256": "0" * 64,
        "source_pointer": "summaries/delta",
    }
    row.update(overrides)
    return row


#: The interval level the convention sentence states is itself a quantity the
#: table carries, so a fixture that quotes that sentence declares it too.
def convention_entry() -> dict:
    return entry(id="conventions/interval_level_percent", claim="the level of every interval",
                 family="conventions", value=95.0, unit="percent", kind="constant",
                 interval_low=None, interval_high=None, interval_kind=None,
                 resampling_unit=None, resampling_draws=None, verdict=None, seed_set=[])


def run(case: Path, quantities: list[dict], supports: dict[str, str], body: str,
        policy: dict | None = None) -> dict[str, object]:
    evidence = case / "results/evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "derived-numbers.json").write_text(
        json.dumps(table(quantities + [convention_entry()], supports)))
    if policy is None:
        policy = CHECKER.default_policy()
        policy['interval_convention_statement'] = (
            "Intervals throughout are 95\\% percentile intervals over the resampling unit named with "
            "each quantity.")
    (evidence / "checker-policy.json").write_text(json.dumps(policy))
    manuscript = write_manuscript(case / "manuscript", body)
    return CHECKER.check(evidence, manuscript)


class TheCheckerRefusesTheDefectsHandTranscriptionProduced(unittest.TestCase):
    """Each negative path is one of the four refusals the pipeline is for."""

    def setUp(self) -> None:
        import tempfile

        self.case = Path(tempfile.mkdtemp(prefix="evidence-checker-"))

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.case, ignore_errors=True)

    def test_a_number_the_table_supports_passes(self):
        report = run(
            self.case, [entry()], {"support_a": "the fixture support"},
            "ProGen3-3B exceeds the profile by $+0.057$ [$+0.036$, $+0.079$]. "
            "Intervals throughout are 95\\% percentile intervals over the resampling unit named with "
            "each quantity.",
        )
        self.assertEqual(report["status"], "pass", report["findings"] + report["untraced"])
        self.assertEqual(report["untraced_numbers"], 0)


    def test_table_caption_units_and_following_interval_row_are_preserved(self):
        body = (r"\begin{table}\caption{Squared error in squared kcal/mol.}"
                r"\begin{tabular}{lr} Arm & Estimate \\ "
                r"ProGen3-3B & $0.05678$ \\ "
                r"95\% interval & $[0.03601,0.07894]$ \\ \end{tabular}\end{table}")
        policy = CHECKER.default_policy()
        policy['interval_convention_statement'] = None
        report = run(self.case, [entry(unit='kcal^2/mol^2')], {'support_a': 'fixture'}, body, policy)
        self.assertEqual(report['status'], 'pass', report)
        # Removing the caption unit must still refuse the physical estimate.
        body = body.replace("squared kcal/mol", "a named scale")
        report = run(self.case, [entry(unit='kcal^2/mol^2')], {'support_a': 'fixture'}, body, policy)
        self.assertIn('unit_not_named', report['findings_by_check'])

        bare = body.replace("a named scale", "squared kcal/mol")
        bare = bare.replace(r"95\% interval & $[0.03601,0.07894]$ \\ ", "")
        report = run(self.case, [entry(unit='kcal^2/mol^2')], {'support_a': 'fixture'}, bare, policy)
        self.assertIn('bare_point_estimate', report['findings_by_check'])


    def test_scientific_notation_survives_math_delimiters_without_binary_tail(self):
        for expression in (r"4.743$\times10^{-13}$", r"$4.743\times10^{-13}$"):
            rendered = CHECKER.strip_markup(expression)
            self.assertIn('0.0000000000004743', rendered)
            self.assertNotIn('times', rendered)
        self.assertEqual(CHECKER.strip_markup(r"$10^{-3}$").strip(), '$ 0.001 $')

    def test_declared_threshold_is_not_a_checkpoint_estimate(self):
        threshold = entry(id='position/alignment_gate', claim='declared minimum alignment fraction gate',
                          family='position', value=.5, kind='constant', interval_low=None, interval_high=None)
        fitted = entry(value=.5001)
        selected, strong = CHECKER.attributable('0.5', [fitted, threshold],
            CHECKER._words('ProGen3-3B falls below the declared alignment fraction gate.'))
        self.assertEqual(selected, [threshold])
        self.assertTrue(strong)
        # A real checkpoint estimate remains subject to interval checks.
        selected, _ = CHECKER.attributable('0.5', [fitted, threshold],
            CHECKER._words('ProGen3-3B profile contrast'))
        self.assertEqual(selected, [fitted])

    def test_a_fabricated_number_is_untraced_and_refused(self):
        report = run(
            self.case, [entry()], {"support_a": "the fixture support"},
            "ProGen3-3B exceeds the profile by $+0.057$ [$+0.036$, $+0.079$] and ProGen2-xlarge by "
            "$+0.392$ [$+0.364$, $+0.418$]. "
            "Intervals throughout are 95\\% percentile intervals over the resampling unit named with "
            "each quantity.",
        )
        self.assertEqual(report["status"], "refused")
        self.assertIn("+0.392", [item["token"] for item in report["untraced"]])

    def test_two_estimates_differing_only_in_support_are_refused_in_one_sentence(self):
        quantities = [
            entry(),
            entry(id="fixture/proteinglm-7b-clm/profile_contrast", value=0.03842,
                  interval_low=0.01751, interval_high=0.05923, support_id="support_b"),
        ]
        report = run(
            self.case, quantities,
            {"support_a": "217 assays in 174 clusters", "support_b": "201 assays in 163 clusters"},
            "ProGen3-3B exceeds the profile by $+0.05678$ and ProteinGLM-7B-CLM by $+0.03842$. "
            "Intervals throughout are 95\\% percentile intervals over the resampling unit named with "
            "each quantity.",
        )
        checks = [finding["check"] for finding in report["findings"]]
        self.assertIn("cross_support_comparison", checks)
        self.assertEqual(report["status"], "refused")

    def test_paired_bounds_disambiguate_points_before_lexical_attribution(self):
        quantities = [
            entry(id="campaign/recognition", family="campaign", claim="recognition", value=.05678),
            entry(id="campaign/yield", family="campaign", claim="yield", value=.03842,
                  interval_low=.01751, interval_high=.05923),
            entry(id="readout/contrast", family="readout", value=.038419,
                  claim="matched decoding recognition and yield differences",
                  interval_low=.02001, interval_high=.06002, support_id="support_b"),
        ]
        words = CHECKER._words("Matched decoding recognition and yield differences")
        selected, strong = CHECKER.attributable(
            '0.0384', CHECKER.Index(quantities).candidates('0.0384'), words)
        self.assertEqual(selected, [quantities[2]])
        self.assertFalse(strong)
        policy = dict(CHECKER.default_policy(), interval_convention_statement=None)
        for point_and_interval in (
                "0.0384 [0.0175,0.0592]",
                "$0.0384$ [ $0.0175$ , $0.0592$ ]",
                "$0.0384 [0.0175,0.0592]$"):
            with self.subTest(expression=point_and_interval):
                report = run(
                    self.case, quantities,
                    {"support_a": "217 assays in 174 clusters",
                     "support_b": "201 assays in 163 clusters"},
                    "Matched decoding recognition and yield differences are "
                    f"0.0568 [0.0360,0.0789] and {point_and_interval}.", policy)
                self.assertEqual(report['status'], 'pass', report)
                self.assertEqual(report['traced_numbers'], 6)

    def test_reversed_or_mismatched_paired_bounds_are_refused(self):
        # All endpoints individually trace, including the wrong high bound.
        quantities = [entry(), entry(id="other/contrast", value=.12345,
                                    interval_low=.01001, interval_high=.08912)]
        policy = dict(CHECKER.default_policy(), interval_convention_statement=None)
        for bounds in ("0.07894,0.03601", "0.03601,0.08912"):
            with self.subTest(bounds=bounds):
                report = run(self.case, quantities, {"support_a": "fixture"},
                             f"The contrast is 0.05678 [{bounds}].", policy)
                self.assertEqual(report['status'], 'refused', report)
                self.assertEqual(report['findings_by_check'], {'point_interval_mismatch': 1})
                self.assertEqual(report['untraced_numbers'], 0)
                self.assertEqual(report['interval_bounds_matching_no_recorded_interval'], 0)

    def test_unpaired_source_value_does_not_borrow_another_records_interval(self):
        quantities = [
            entry(interval_low=None, interval_high=None, kind="artifact_leaf"),
            entry(id="other/contrast", value=.12345),
        ]
        report = run(
            self.case, quantities, {"support_a": "fixture"},
            "The contrast is 0.05678 [0.03601,0.07894].",
            dict(CHECKER.default_policy(), interval_convention_statement=None))
        self.assertEqual(report['status'], 'refused', report)
        self.assertEqual(report['findings_by_check'], {'point_interval_mismatch': 1})
        self.assertEqual(report['untraced_numbers'], 0)
        self.assertEqual(report['interval_bounds_matching_no_recorded_interval'], 0)

    def test_valid_paired_bounds_do_not_permit_a_crossed_support_comparison(self):
        quantities = [entry(), entry(id="other/contrast", value=.03842,
                                    interval_low=.01751, interval_high=.05923,
                                    support_id="support_b")]
        report = run(
            self.case, quantities,
            {"support_a": "217 assays in 174 clusters", "support_b": "201 assays in 163 clusters"},
            "The contrasts are 0.05678 [0.03601,0.07894] and 0.03842 [0.01751,0.05923].",
            dict(CHECKER.default_policy(), interval_convention_statement=None))
        self.assertEqual(report['status'], 'refused', report)
        self.assertEqual(report['findings_by_check'], {'cross_support_comparison': 1})

    def test_matching_paired_bounds_retain_multiple_candidates(self):
        quantities = [entry(), entry(id="other/contrast", family="other")]
        paired = CHECKER.INTERVAL.search('[0.036,0.07894]')
        narrowed = CHECKER.paired_interval_candidates(quantities, paired)
        self.assertEqual(narrowed, quantities)
        selected, strong = CHECKER.attributable('0.05678', narrowed, set())
        self.assertEqual(selected, quantities)
        self.assertFalse(strong)

    def test_the_same_two_estimates_pass_once_the_comparison_is_declared(self):
        quantities = [
            entry(),
            entry(id="fixture/proteinglm-7b-clm/profile_contrast", value=0.03842,
                  interval_low=0.01751, interval_high=0.05923, support_id="support_b"),
        ]
        sentence = ("ProGen3-3B exceeds the profile by $+0.05678$ and ProteinGLM-7B-CLM by "
                    "$+0.03842$ on their own native supports, which are named apart in Methods.")
        report = run(
            self.case, quantities,
            {"support_a": "217 assays in 174 clusters", "support_b": "201 assays in 163 clusters"},
            sentence + " Intervals throughout are 95\\% percentile intervals over the resampling "
            "unit named with each quantity.",
            policy=dict(CHECKER.default_policy(), cross_support_exceptions=[
                {"anchor": "on their own native supports",
                 "reason": "the sentence names the two supports apart"}]),
        )
        self.assertNotIn("cross_support_comparison",
                         [finding["check"] for finding in report["findings"]])

    def test_an_exception_whose_sentence_is_gone_is_itself_refused(self):
        report = run(
            self.case, [entry()], {"support_a": "the fixture support"},
            "ProGen3-3B exceeds the profile by $+0.057$ [$+0.036$, $+0.079$]. "
            "Intervals throughout are 95\\% percentile intervals over the resampling unit named with "
            "each quantity.",
            policy=dict(CHECKER.default_policy(), cross_support_exceptions=[
                {"anchor": "a sentence that was edited away", "reason": "stale"}]),
        )
        self.assertIn("dead_exception_anchor", [finding["check"] for finding in report["findings"]])
        self.assertEqual(report["status"], "refused")

    def test_an_interval_quoted_without_its_unit_is_refused(self):
        quantities = [entry(id="fixture/contact/epsilon", value=0.19161, unit="kcal/mol",
                            interval_low=-0.06083, interval_high=0.41201,
                            resampling_unit="site pair")]
        report = run(
            self.case, quantities, {"support_a": "111 contact and 53 control site pairs"},
            "The contact-minus-matched-control difference reads $+0.19161$ [$-0.06083$, $+0.41201$] "
            "raw. Intervals throughout are 95\\% percentile intervals over the resampling unit named "
            "with each quantity.",
        )
        self.assertIn("unit_not_named", [finding["check"] for finding in report["findings"]])
        self.assertEqual(report["status"], "refused")

    def test_the_same_interval_passes_once_the_unit_is_named(self):
        quantities = [entry(id="fixture/contact/epsilon", value=0.19161, unit="kcal/mol",
                            interval_low=-0.06083, interval_high=0.41201,
                            resampling_unit="site pair")]
        report = run(
            self.case, quantities, {"support_a": "111 contact and 53 control site pairs"},
            "The contact-minus-matched-control difference reads $+0.19161$ [$-0.06083$, $+0.41201$] "
            "kcal/mol raw. Intervals throughout are 95\\% percentile intervals over the resampling "
            "unit named with each quantity.",
        )
        self.assertNotIn("unit_not_named", [finding["check"] for finding in report["findings"]])

    def test_a_bare_point_estimate_is_refused(self):
        report = run(
            self.case, [entry()], {"support_a": "the fixture support"},
            "ProGen3-3B exceeds the profile by $+0.05678$. "
            "Intervals throughout are 95\\% percentile intervals over the resampling unit named with "
            "each quantity.",
        )
        findings = [finding for finding in report["findings"]
                    if finding["check"] == "bare_point_estimate"]
        self.assertTrue(findings, report["findings"])
        self.assertEqual(report["status"], "refused")

    def test_an_effective_count_quoted_without_its_convention_is_refused(self):
        quantities = [entry(id="fixture/pairwise/kish", value=121.6, unit="effective site pairs",
                            kind="effective_count", interval_low=None, interval_high=None,
                            interval_kind=None, resampling_unit=None, resampling_draws=None,
                            weighting_convention="cycle share", verdict=None)]
        report = run(
            self.case, quantities, {"support_a": "64 groups and 217 site pairs"},
            "Power is set by site pairs at a Kish effective $121.6$. "
            "Intervals throughout are 95\\% percentile intervals over the resampling unit named with "
            "each quantity.",
        )
        self.assertIn("effective_count_without_convention",
                      [finding["check"] for finding in report["findings"]])

    def test_a_provisional_quantity_quoted_without_the_marker_is_refused(self):
        quantities = [entry(id="fixture/progen3-3b/representation_increment", value=0.02,
                            interval_low=0.009, interval_high=0.03, level="representation",
                            provisional=True)]
        report = run(
            self.case, quantities, {"support_a": "the fixture support"},
            "ProGen3-3B adds $+0.020$ [$+0.009$, $+0.030$]. "
            "Intervals throughout are 95\\% percentile intervals over the resampling unit named with "
            "each quantity.",
        )
        self.assertIn("provisional_not_marked", [finding["check"] for finding in report["findings"]])


class BothIntervalNotationsCarryTheirInterval(unittest.TestCase):
    """The manuscript writes its intervals two ways and both are intervals.

    Tables and maths use the bracketed pair. The prose writes ``(95\\% CI,
    a--b)``, separating the bounds with an en dash, an em dash or a double hyphen
    interchangeably. A checker that read only the bracketed form accused every
    sentence of the prose of quoting a bare point estimate, and read each bound as
    a second estimate on another support, so the refusals that matter sat buried
    under corrections to sentences written exactly as the convention requires.
    Reading the second notation is not a relaxation: a parenthesis that is not
    the declared interval, and a pair of bounds the table does not carry, are
    still refused below.
    """

    DASHES = {"en dash": "–", "em dash": "—", "double hyphen": "--"}

    def setUp(self) -> None:
        import tempfile

        self.case = Path(tempfile.mkdtemp(prefix="evidence-interval-"))

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.case, ignore_errors=True)

    def _sentence(self, interval: str) -> str:
        return (f"ProGen3-3B exceeds the profile by 0.05678 {interval}. "
                "Intervals throughout are 95\\% percentile intervals over the resampling unit named "
                "with each quantity.")

    def _report(self, interval: str) -> dict:
        return run(self.case, [entry()], {"support_a": "the fixture support"},
                   self._sentence(interval))

    def test_every_dash_the_prose_writes_carries_the_interval(self):
        for name, dash in self.DASHES.items():
            with self.subTest(separator=name):
                report = self._report(f"(95\\% CI, 0.03601{dash}0.07894)")
                self.assertEqual(report["status"], "pass", report["findings"] + report["untraced"])

    def test_the_bracketed_notation_still_carries_the_interval(self):
        report = self._report("[$+0.03601$, $+0.07894$]")
        self.assertEqual(report["status"], "pass", report["findings"] + report["untraced"])

    def test_the_coverage_label_is_not_one_of_the_bounds(self):
        text = "0.05678 (95\\% CI, 0.03601–0.07894)"
        match = CHECKER.INTERVAL.search(text)
        self.assertEqual(CHECKER.interval_bounds(match), ("0.03601", "0.07894"))
        self.assertEqual(text[slice(*CHECKER.interval_extent(match))], "0.03601–0.07894")

    def test_a_bound_is_not_a_second_estimate_on_another_support(self):
        quantities = [
            entry(),
            entry(id="fixture/proteinglm-7b-clm/profile_contrast", value=0.07894,
                  interval_low=0.05101, interval_high=0.09233, support_id="support_b"),
        ]
        report = run(
            self.case, quantities,
            {"support_a": "217 assays in 174 clusters", "support_b": "201 assays in 163 clusters"},
            self._sentence("(95\\% CI, 0.03601–0.07894)"))
        self.assertEqual(report["status"], "pass", report["findings"] + report["untraced"])

    def test_a_lone_value_in_the_parenthesis_is_not_read_as_two_bounds(self):
        report = self._report("(95\\% CI, 0.03601)")
        self.assertIn("bare_point_estimate", report["findings_by_check"])
        self.assertEqual(report["status"], "refused")

    def test_a_bracketed_pair_without_its_comma_is_not_an_interval(self):
        report = self._report("[0.03601 0.07894]")
        self.assertIn("bare_point_estimate", report["findings_by_check"])
        self.assertEqual(report["status"], "refused")

    def test_a_parenthesis_at_another_coverage_is_not_the_declared_interval(self):
        report = self._report("(90\\% CI, 0.03601–0.07894)")
        self.assertIn("bare_point_estimate", report["findings_by_check"])
        self.assertEqual(report["status"], "refused")

    def test_bounds_matching_no_recorded_interval_are_refused_in_either_notation(self):
        quantities = [entry(), entry(id="other/contrast", value=.12345,
                                     interval_low=.01001, interval_high=.08912)]
        policy = dict(CHECKER.default_policy(), interval_convention_statement=None)
        for notation in ("[0.03601, 0.08912]", "(95\\% CI, 0.03601–0.08912)"):
            with self.subTest(notation=notation):
                report = run(self.case, quantities, {"support_a": "fixture"},
                             f"The contrast is 0.05678 {notation}.", policy)
                self.assertEqual(report["findings_by_check"], {"point_interval_mismatch": 1}, report)
                self.assertEqual(report["status"], "refused")


def span(prefix: str, low: float, high: float, **overrides) -> list[dict]:
    """One recorded span: the ``min`` and ``max`` of a reduction over a row set."""
    shared = dict(family="fixture", kind="range_bound", interval_low=None, interval_high=None,
                  interval_kind=None, resampling_unit=None, resampling_draws=None, verdict=None)
    shared.update(overrides)
    return [entry(id=f"{prefix}/min", value=low,
                  claim="range of the fitted increment over the resolving checkpoints (min)",
                  **shared),
            entry(id=f"{prefix}/max", value=high,
                  claim="range of the fitted increment over the resolving checkpoints (max)",
                  **shared)]


class ARangeAcrossModelsIsNotAnEstimateWithAnInterval(unittest.TestCase):
    """``0.007--0.015`` is a span over a set of models, not one estimate.

    The manuscript quotes a quantity's spread across models that way. Reading
    either endpoint as a point estimate of its own attributes it to whichever of
    several thousand rows happens to share the number, and then reports the
    sentence as quoting that other row's estimate without the interval the
    table carries for it --- a correction to a sentence that quotes no estimate
    at all. The pair is matched against the recorded ``min``/``max`` bounds of
    one span instead, so a range is verified rather than merely exempted, and a
    range the table records no span for is reported as uncovered rather than
    passing silently.
    """

    POLICY = dict(interval_convention_statement=None)

    def setUp(self) -> None:
        import tempfile

        self.case = Path(tempfile.mkdtemp(prefix="evidence-range-"))

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.case, ignore_errors=True)

    def _run(self, body: str, quantities: list[dict]) -> dict:
        return run(self.case, quantities, {"support_a": "the fixture support"}, body,
                   dict(CHECKER.default_policy(), **self.POLICY))

    def test_a_dash_between_two_digits_survives_markup_stripping(self):
        # Every other dash stays punctuation, and every offset is preserved.
        rendered = CHECKER.strip_markup("\\begin{document}\nspan 0.007--0.015 on 163--174 strata---19")
        self.assertIn("0.007\u2013 0.015", rendered)
        self.assertIn("163\u2013 174", rendered)
        self.assertIn("strata   19", rendered)
        self.assertEqual(len(rendered.split("document}", 1)[-1]),
                         len("\nspan 0.007--0.015 on 163--174 strata---19"))

    def test_a_quoted_range_is_verified_against_the_span_the_table_records(self):
        report = self._run(
            "Resolved increments span 0.007--0.015 Spearman.",
            span("fixture/increments/resolved", 0.00712, 0.01544))
        self.assertEqual(report["status"], "pass", report["findings"] + report["untraced"])
        self.assertEqual(report["ranges_verified_against_a_recorded_span"], 1)
        self.assertEqual(report["ranges_matching_no_recorded_span"], 0)
        self.assertEqual(report["verified_ranges"][0]["spans"],
                         ["fixture/increments/resolved"])

    def test_a_range_whose_endpoints_are_not_the_recorded_bounds_fails(self):
        report = self._run(
            "Resolved increments span 0.007--0.016 Spearman.",
            span("fixture/increments/resolved", 0.00712, 0.01544))
        self.assertEqual(report["status"], "refused", report)
        self.assertEqual(report["ranges_verified_against_a_recorded_span"], 0)
        self.assertEqual(report["ranges_matching_no_recorded_span"], 1)
        # The endpoint the table does not carry is a number a reader cannot check.
        self.assertEqual([item["token"] for item in report["untraced"]], ["0.016"])

    def test_a_range_endpoint_is_never_reported_as_a_bare_point_estimate(self):
        # The sentence quotes a span and, separately, one estimate bare. Only
        # the bare estimate is refused.
        quantities = span("fixture/increments/resolved", 0.00712, 0.01544) + [entry()]
        report = self._run(
            "Resolved increments span 0.007--0.015 Spearman, and ProGen3-3B exceeds the profile "
            "by 0.05678.", quantities)
        self.assertEqual(report["findings_by_check"], {"bare_point_estimate": 1}, report)
        self.assertEqual([finding["token"] for finding in report["findings"]], ["0.05678"])

    def test_only_the_dash_classifies_a_range(self):
        # Two estimates that merely sit beside each other are two estimates.
        quantities = [entry(), entry(id="fixture/other/profile_contrast", value=0.03842,
                                     interval_low=0.01751, interval_high=0.05923)]
        report = self._run("The fitted increments are 0.03842 and 0.05678.", quantities)
        self.assertEqual(report["findings_by_check"], {"bare_point_estimate": 2}, report)
        self.assertEqual(report["ranges_matching_no_recorded_span"], 0)
        # Joined by a dash they are one span, which the table records no bounds
        # for: reported as uncovered, and not as two bare estimates.
        report = self._run("The fitted increments span 0.03842--0.05678.", quantities)
        self.assertEqual(report["findings_by_check"], {}, report)
        self.assertEqual(report["ranges_matching_no_recorded_span"], 1)
        self.assertEqual(report["ranges_without_a_recorded_span"][0]["range"], "0.03842\u20130.05678")

    def test_a_labelled_interval_is_an_interval_and_not_a_range(self):
        for dash in ("--", "–"):
            with self.subTest(separator=dash):
                report = self._run(
                    f"ProGen3-3B exceeds the profile by 0.05678 (95\\% CI, 0.03601{dash}0.07894).",
                    [entry()])
                self.assertEqual(report["status"], "pass", report["findings"] + report["untraced"])
                self.assertEqual(report["ranges_verified_against_a_recorded_span"], 0)
                self.assertEqual(report["ranges_matching_no_recorded_span"], 0)
                # And the bounds are still checked against the recorded interval.
                report = self._run(
                    f"ProGen3-3B exceeds the profile by 0.05678 (95\\% CI, 0.03601{dash}0.08912).",
                    [entry(), entry(id="fixture/other/contrast", value=0.12345,
                                    interval_low=0.01001, interval_high=0.08912)])
                self.assertEqual(report["findings_by_check"], {"point_interval_mismatch": 1}, report)
                self.assertEqual(report["ranges_matching_no_recorded_span"], 0)

    def test_a_span_missing_one_of_its_two_bounds_verifies_nothing(self):
        rows = span("fixture/increments/resolved", 0.00712, 0.01544)
        index = CHECKER.Index([rows[0]])
        self.assertEqual(index.spans, [])
        self.assertEqual(index.span_candidates("0.007", "0.015"), [])
        index = CHECKER.Index(rows)
        self.assertEqual([prefix for prefix, _, _ in index.spans],
                         ["fixture/increments/resolved"])


class AMagnitudeIsReadAgainstTheSignedDifferenceItStates(unittest.TestCase):
    """The prose states a reduction as the amount reduced.

    "The stage reduces the rate by 0.1146 (95\\% CI, 0.0909--0.1383)" is the
    readable and correct statement of a ledger row holding -0.1146 on
    [-0.1383, -0.0909]; writing the double negative instead would not be. The
    checker therefore reads one orientation against the other, under the same
    triple match it already makes: the point and both bounds have to flip
    together, so negating the stored interval reverses it and the quoted low
    bound can only be the stored high bound's magnitude. A pair that is not
    consistently oriented, a pair with one bound of its own, a stored interval
    that straddles zero, and a sign-flipped point quoted with no interval at
    all are all still refused, and a same-sign entry is never displaced by a
    reflected one.
    """

    #: The stored row: a reduction, signed, with its interval on one side of zero.
    REDUCTION = dict(id="fixture/prollama/any_family_rate", family="generation_replication",
                     claim="second-minus-first stage unconditional model rate",
                     value=-0.1146, interval_low=-0.1383, interval_high=-0.0909,
                     unit="dimensionless rate difference",
                     resampling_unit="generation stream at a fixed checkpoint", verdict=None)

    CONVENTION = ("Intervals throughout are 95\\% percentile intervals over the resampling unit "
                  "named with each quantity.")

    def setUp(self) -> None:
        import tempfile

        self.case = Path(tempfile.mkdtemp(prefix="evidence-reflected-"))

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.case, ignore_errors=True)

    def _report(self, quoted: str, quantities: list[dict] | None = None) -> dict:
        return run(self.case, quantities if quantities is not None else [entry(**self.REDUCTION)],
                   {"support_a": "the fixture support"},
                   f"The second training stage reduces the any-family recognition rate by {quoted}. "
                   + self.CONVENTION)

    def test_a_reduction_quoted_as_a_magnitude_traces_to_its_signed_row(self):
        report = self._report("0.1146 (95\\% CI, 0.0909–0.1383)")
        self.assertEqual(report["status"], "pass", report["findings"] + report["untraced"])
        self.assertEqual(report["untraced_numbers"], 0)
        # Both bounds belong to the entry the point was attributed by, so
        # neither is reported as an endpoint no recorded interval carries.
        self.assertEqual(report["interval_bounds_matching_no_recorded_interval"], 0)

    def test_a_pair_that_is_not_consistently_oriented_is_refused(self):
        report = self._report("0.1146 (95\\% CI, 0.1383–0.0909)")
        self.assertEqual(report["status"], "refused", report)
        self.assertEqual([item["token"] for item in report["untraced"]], ["0.1146"])
        self.assertEqual(report["interval_bounds_matching_no_recorded_interval"], 2)

    def test_a_pair_with_one_bound_of_its_own_is_refused(self):
        report = self._report("0.1146 (95\\% CI, 0.0909–0.1400)")
        self.assertEqual(report["status"], "refused", report)
        self.assertEqual([item["token"] for item in report["untraced"]], ["0.1146"])

    def test_a_sign_flipped_point_quoted_without_an_interval_is_not_matched(self):
        report = self._report("0.1146")
        self.assertEqual(report["status"], "refused", report)
        self.assertEqual([item["token"] for item in report["untraced"]], ["0.1146"])

    def test_an_interval_straddling_zero_has_no_magnitude_to_quote(self):
        straddling = entry(**dict(self.REDUCTION, value=-0.0100,
                                  interval_low=-0.0300, interval_high=0.0200))
        paired = CHECKER.INTERVAL.search("(95\\% CI, -0.0200–0.0300)")
        self.assertEqual(CHECKER.reflected_interval_candidates([straddling], paired), [])

    def test_a_degenerate_interval_carries_no_orientation_to_confirm(self):
        degenerate = entry(**dict(self.REDUCTION, interval_low=-0.1146, interval_high=-0.1146))
        paired = CHECKER.INTERVAL.search("(95\\% CI, 0.1146–0.1146)")
        self.assertEqual(CHECKER.reflected_interval_candidates([degenerate], paired), [])

    def test_a_same_sign_entry_is_never_displaced_by_a_reflected_one(self):
        # The reflected row carries a physical unit the sentence does not name,
        # so attributing it would refuse; the same-sign row is dimensionless.
        increase = entry(id="fixture/prollama/increase", family="generation_replication",
                         claim="first-minus-second stage unconditional model rate",
                         value=0.1146, interval_low=0.0909, interval_high=0.1383,
                         unit="dimensionless rate difference",
                         resampling_unit="generation stream at a fixed checkpoint", verdict=None)
        reflected = entry(**dict(self.REDUCTION, unit="kcal/mol"))
        report = self._report("0.1146 (95\\% CI, 0.0909–0.1383)", [increase, reflected])
        self.assertEqual(report["status"], "pass", report["findings"] + report["untraced"])
        # Without the same-sign row the reflected one is found, and is held to
        # the same refusals as any other attribution.
        report = self._report("0.1146 (95\\% CI, 0.0909–0.1383)", [reflected])
        self.assertIn("unit_not_named", report["findings_by_check"])


class TheTableRefusesARowWithAHoleInIt(unittest.TestCase):
    """A row that cannot be quoted safely must not reach the table at all."""

    def test_a_quantity_without_a_unit_is_refused(self):
        with self.assertRaises(LedgerError):
            Quantity(id="x", claim="c", family="f", value=1.0, unit="", kind="count",
                     support_id="s", source_path="p", source_sha256="d")

    def test_an_interval_without_its_resampling_unit_is_refused(self):
        with self.assertRaises(LedgerError):
            Quantity(id="x", claim="c", family="f", value=1.0, unit="kcal/mol", kind="estimate",
                     support_id="s", interval=(0.0, 2.0), interval_kind="95% percentile",
                     source_path="p", source_sha256="d")

    def test_an_interval_without_its_draw_count_is_refused(self):
        with self.assertRaises(LedgerError):
            Quantity(id="x", claim="c", family="f", value=1.0, unit="kcal/mol", kind="estimate",
                     support_id="s", interval=(0.0, 2.0), interval_kind="95% percentile",
                     resampling_unit="site pair", source_path="p", source_sha256="d")

    def test_a_point_estimate_without_an_interval_states_why(self):
        with self.assertRaises(LedgerError):
            Quantity(id="x", claim="c", family="f", value=1.0, unit="kcal/mol", kind="estimate",
                     support_id="s", source_path="p", source_sha256="d")
        Quantity(id="x", claim="c", family="f", value=1.0, unit="kcal/mol", kind="estimate",
                 support_id="s", no_interval_reason="a median over checkpoints",
                 source_path="p", source_sha256="d")

    def test_an_effective_count_names_its_weighting_convention(self):
        with self.assertRaises(LedgerError):
            Quantity(id="x", claim="c", family="f", value=121.6, unit="effective site pairs",
                     kind="effective_count", support_id="s", source_path="p", source_sha256="d")

    def test_a_verdict_outside_the_four_outcomes_is_refused(self):
        for verdict in ("unresolved_or_not_detected", "negative", "null"):
            with self.assertRaises(LedgerError):
                Quantity(id="x", claim="c", family="f", value=1.0, unit="checkpoints", kind="count",
                         support_id="s", verdict=verdict, source_path="p", source_sha256="d")
        for verdict in ("unresolved", "not_detected"):
            Quantity(id="x", claim="c", family="f", value=1.0, unit="checkpoints", kind="count",
                     support_id="s", verdict=verdict, source_path="p", source_sha256="d")

    def test_an_interval_that_does_not_bracket_its_estimate_is_refused(self):
        with self.assertRaises(LedgerError):
            Quantity(id="x", claim="c", family="f", value=3.0, unit="kcal/mol", kind="estimate",
                     support_id="s", interval=(0.0, 2.0), interval_kind="95% percentile",
                     resampling_unit="site pair", resampling_draws=2000,
                     source_path="p", source_sha256="d")

    def test_an_undeclared_support_is_refused(self):
        ledger = Ledger(Artifacts())
        with self.assertRaises(LedgerError):
            ledger.add(Quantity(id="x", claim="c", family="f", value=1.0, unit="checkpoints",
                                kind="count", support_id="never_declared",
                                source_path="p", source_sha256="d"))

    def test_a_pointer_that_does_not_resolve_raises_rather_than_returning_a_default(self):
        payload = {"summaries": {"delta": {"point": 1.0}}}
        self.assertEqual(resolve(payload, ("summaries", "delta", "point")), 1.0)
        with self.assertRaises(KeyError):
            resolve(payload, ("summaries", "absent"))


class ACountIsNotSummableAcrossImplementations(unittest.TestCase):
    """A control several units appear to share is several controls.

    Most cohorts computed their baseline-identity check while the run happened
    and one recovered an equivalent afterwards from what the run persisted.
    Totalling those counts would state as one measurement what two
    implementations produced differently, so the sum refuses and the caller
    records why the total does not exist.
    """

    def _control(self, provenance: str, value: float) -> Quantity:
        return Quantity(id=f"c/{provenance}/{value}", claim="a control count", family="f",
                        value=value, unit="checks", kind="support_count", support_id="s",
                        control_provenance=provenance, source_path="p", source_sha256="d")

    def test_counts_from_one_implementation_sum(self):
        rows = [self._control("computed_during_the_run", 135.0),
                self._control("computed_during_the_run", 135.0)]
        self.assertEqual(checked_sum(rows, field="control_provenance"), 270.0)

    def test_counts_from_different_implementations_refuse_to_sum(self):
        rows = [self._control("computed_during_the_run", 135.0),
                self._control("derived_after_the_fact", 2025.0)]
        with self.assertRaises(LedgerError) as refusal:
            checked_sum(rows, field="control_provenance")
        self.assertIn("control_provenance", str(refusal.exception))

    def test_an_unknown_control_provenance_is_refused(self):
        with self.assertRaises(LedgerError):
            Quantity(id="x", claim="c", family="f", value=1.0, unit="checks", kind="support_count",
                     support_id="s", control_provenance="assumed", source_path="p",
                     source_sha256="d")


class TheLivePipelineRuns(unittest.TestCase):
    """The table, the contract and the checker, on the repository as it stands.

    The checker is run here rather than its stored report read. Reading the
    report asserted nothing about the checker: the report is the checker's own
    output, so a report left behind by a directory the pipeline had since moved
    off kept answering "pass" for a run nobody had recomputed, and the two
    defects that produced -- a superseded evidence directory and an interval
    notation the pattern could not read -- were invisible from inside this suite.

    What is asserted is the pipeline's own conditions: the table's rows, the
    contract's bindings, that the checker completes against the current
    manuscript, and that the figures and the checker read one ledger. Not a
    finding count: the manuscript is another agent's and its findings move under
    it. Running the checker rewrites its report in the evidence directory, which
    is exactly what the entry point does and where its output belongs.
    """

    def setUp(self) -> None:
        self.directory = BUILDER.EVIDENCE_DIR
        if not (self.directory / "derived-numbers.json").is_file():
            self.skipTest("no derived-number table has been built in this checkout")

    def test_every_row_of_the_table_carries_what_it_needs(self):
        payload = json.loads((self.directory / "derived-numbers.json").read_text())
        self.assertGreater(payload["counts"]["quantities"], 0)
        for row in payload["quantities"]:
            self.assertTrue(row["unit"], row["id"])
            if row["interval_low"] is not None:
                self.assertTrue(row["resampling_unit"], row["id"])
                self.assertTrue(row["resampling_draws"], row["id"])
            if row["kind"] == "effective_count":
                self.assertTrue(row["weighting_convention"], row["id"])
            self.assertIn(row["support_id"], payload["supports"], row["id"])
            self.assertEqual(len(row["source_sha256"]), 64, row["id"])

    def test_every_figure_table_stays_bound_to_the_artifact_it_drew_from(self):
        contract = json.loads((self.directory / "figure-data-contract.json").read_text())
        for name, bound in contract["tables"].items():
            for source, binding in bound["artifact_bindings"].items():
                self.assertEqual(binding["status"], "bound", f"{name} -> {source}")

    def test_the_checker_completes_against_the_current_manuscript(self):
        if not any((CHECKER.MANUSCRIPT / name).is_file() for name in CHECKER.SOURCES):
            self.skipTest("the manuscript is held locally and is not in this checkout")
        report = CHECKER.check(self.directory)
        # The submission location is declared once; the checker reads that one,
        # not a path of its own that the submission moved out from under.
        self.assertEqual(report["manuscript"],
                         (REPO_ROOT / publication_paths.SUBMISSION).as_posix())
        self.assertEqual(
            report["derived_number_table"]["quantities"],
            json.loads((self.directory / "derived-numbers.json").read_text())["counts"]["quantities"])
        # A checker that attributed nothing would report no findings and look
        # like agreement.
        self.assertGreater(report["traced_numbers"], 0)
        self.assertIn(report["status"], ("pass", "refused"))

    def test_the_figures_and_the_checker_read_one_ledger(self):
        # A plotted figure records which ledger it drew from, so the directory
        # the checker reads is checkable rather than a literal taken on trust.
        receipt = BUILDER.FIGURE_DIR / "main/framework_validation.json"
        if not receipt.is_file():
            self.skipTest("the framework composite receipt is not in this checkout")
        cited = [path for path in json.loads(receipt.read_text())["source_sha256"]
                 if path.startswith("results/shared/manuscript_evidence_")]
        self.assertEqual(
            cited,
            [(BUILDER.EVIDENCE_DIR / "derived-numbers.csv").relative_to(REPO_ROOT).as_posix()])


class CurrentEndpointReceiptsKeepTheirEstimands(unittest.TestCase):
    def test_current_sensitivity_rows_keep_support_and_interval_conventions(self):
        if not (REPO_ROOT / 'results/R3/pairwise_residual_nested_20260927/residual_progen3-3b.json').is_file():
            self.skipTest('retained current endpoint receipts are not staged in this checkout')
        from scripts.capability.reporting.current_manuscript_quantities import add_current_quantities
        ledger = Ledger(Artifacts())
        ledger.declare_support('conditional_class_cells', 'conditional cohorts')
        add_current_quantities(ledger)
        rows = {row.id: row for row in ledger.quantities}
        residual = [row for row in rows.values() if row.family == 'nested_residual' and row.kind == 'estimate']
        self.assertEqual(len(residual), 9)
        self.assertEqual(len({row.support_id for row in residual}), 1)
        self.assertTrue(all(row.interval[0] <= 0 <= row.interval[1] for row in residual))
        self.assertEqual(rows['structure_contact/current_primary_intervals'].value, 44)
        for endpoint in ('any_family', 'complete_domain'):
            original = rows[f'generation_length/all/{endpoint}']
            reweighted = rows[f'generation_length/reweighted/{endpoint}']
            self.assertEqual(original.interval_kind, '97.5% percentile')
            self.assertEqual(original.resampling_draws, 4000)
            self.assertIsNone(reweighted.interval)
            self.assertTrue(reweighted.no_interval_reason)
            self.assertNotEqual(original.support_id, reweighted.support_id)
        self.assertEqual(rows['context/intersection/n_assays'].value, 164)
        self.assertEqual(rows['domainome/fitted/variants'].value, 109568)
        self.assertEqual(rows['conditional/zymctrl/eligible_hits'].value, 2488)
        self.assertEqual(rows['position_terms/alignment_gate'].kind, 'constant')


class TheFigureDataContractCatchesDrift(unittest.TestCase):
    """A figure cannot silently redraw from an artifact that moved."""

    def setUp(self) -> None:
        self.artifacts = Artifacts()
        self.live = "scripts/capability/reporting/figure_data/ncs/source-provenance.json"
        if not self.artifacts.exists(self.live):
            self.skipTest("the manuscript's figure provenance is not in this checkout")
        self.digest = self.artifacts.sha256(self.live)

    def test_a_row_bound_to_the_digest_on_disk_is_bound(self):
        bindings = BUILDER.table_bindings(
            self.artifacts, [{"source_path": self.live, "source_sha256": self.digest}])
        self.assertEqual(bindings[self.live]["status"], "bound")

    def test_a_row_bound_to_a_digest_that_moved_is_refused(self):
        bindings = BUILDER.table_bindings(
            self.artifacts, [{"source_path": self.live, "source_sha256": "0" * 64}])
        self.assertEqual(bindings[self.live]["status"], "artifact_changed")

    def test_a_row_bound_to_an_artifact_that_is_gone_is_refused(self):
        bindings = BUILDER.table_bindings(
            self.artifacts, [{"source_path": "results/absent-fixture.json",
                              "source_sha256": "f" * 64}])
        self.assertEqual(bindings["results/absent-fixture.json"]["status"], "artifact_absent")

    def test_a_table_recording_two_digests_for_one_artifact_is_refused(self):
        bindings = BUILDER.table_bindings(self.artifacts, [
            {"source_path": self.live, "source_sha256": self.digest},
            {"source_path": self.live, "source_sha256": "1" * 64},
        ])
        self.assertEqual(bindings[self.live]["status"], "csv_disagrees_with_itself")


if __name__ == "__main__":
    unittest.main()
