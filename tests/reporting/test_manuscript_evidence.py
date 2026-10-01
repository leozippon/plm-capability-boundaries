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

    Two cohorts computed their baseline-identity check while the run happened
    and a third recovered an equivalent afterwards from what the run persisted.
    Totalling those counts would state as one measurement what three
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
    """The generator and the checker run on the repository as it stands.

    The manuscript is another agent's and its findings move under it, so this
    asserts the pipeline's own conditions rather than a finding count.
    """

    def test_the_table_and_the_contract_build_and_the_checker_reports(self):
        directory = BUILDER.OUT_DIR
        if not (directory / "derived-numbers.json").is_file():
            self.skipTest("no derived-number table has been built in this checkout")
        payload = json.loads((directory / "derived-numbers.json").read_text())
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
        report = json.loads((directory / "consistency-report.json").read_text())
        self.assertEqual(report["status"], "pass", report["findings"] + report["untraced"])
        contract = json.loads((directory / "figure-data-contract.json").read_text())
        for name, bound in contract["tables"].items():
            for source, binding in bound["artifact_bindings"].items():
                self.assertEqual(binding["status"], "bound", f"{name} -> {source}")


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
