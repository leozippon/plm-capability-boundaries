"""The evidence checker reads the notations the manuscript is typeset in.

A sign, a spacing macro, a convention word or a ``to`` between bounds changes how
a quantity is written and not what it is. Each is tested to pass where the
ledger supports the number, and the relaxation is bounded by negative tests: a
point estimate with no interval, or with the wrong one, is still refused.
"""
from pathlib import Path
import shutil
import tempfile
import unittest

from tests.reporting.test_manuscript_evidence import CHECKER, entry, run

CONVENTION = ("Intervals throughout are 95\\% percentile intervals over the resampling unit named "
              "with each quantity.")
POSITIVE = entry()
NEGATIVE = entry(value=-0.05678, interval_low=-0.07894, interval_high=-0.03601)
SUPPORTS = {"support_a": "the fixture support"}


class Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.case = Path(tempfile.mkdtemp(prefix="evidence-notation-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.case, ignore_errors=True)

    def report(self, quantity: dict, text: str) -> dict:
        return run(self.case, [quantity], SUPPORTS, f"ProGen3-3B differs from the profile by {text}. {CONVENTION}")

    def assertClean(self, report: dict) -> None:
        self.assertEqual(report["status"], "pass", report["findings"] + report["untraced"])


class TheCheckerReadsTypesetNotation(Fixture):
    def test_every_typeset_minus_is_the_sign_of_a_point_and_of_its_bounds(self):
        for minus in (r"\textminus{}", "$-$", r"\(-\)", "\u2212"):
            with self.subTest(minus=minus):
                self.assertClean(self.report(
                    NEGATIVE, f"{minus}0.05678 [{minus}0.07894, {minus}0.03601]"))

    def test_a_minus_between_words_is_an_operator_and_not_a_sign(self):
        body = CHECKER.strip_markup("\\begin{document}\nProfile $-$ unguided 0.1 \\textminus{}0.2")
        self.assertIn("Profile $-$ unguided 0.1", body)
        self.assertIn("-0.2", body)

    def test_spacing_between_a_point_and_its_interval_is_not_a_gap(self):
        for spacing in (r"\;", r"\ ", "~", r"\,", r"\quad"):
            with self.subTest(spacing=spacing):
                self.assertClean(self.report(
                    POSITIVE, f"$0.05678{spacing}[0.03601,{spacing}0.07894]$"))

    def test_a_unit_keeps_the_spelling_the_unit_table_looks_for(self):
        self.assertIn(r"kcal\,mol", CHECKER.strip_markup("\\begin{document}\n$0.1$ kcal\\,mol"))

    def test_an_interval_may_name_its_convention_and_the_name_is_recorded(self):
        for words, recorded in (("marginal 95\\% CI", ["marginal"]),
                                ("simultaneous 95\\% CI", ["simultaneous"]),
                                ("pointwise 95\\% CI", ["pointwise"]),
                                ("95\\% BCa CI", ["BCa"]),
                                ("bias-corrected accelerated 95\\% CI", ["bias-corrected accelerated"]),
                                ("marginal 95\\% BCa CI", ["marginal", "BCa"])):
            with self.subTest(words=words):
                report = self.report(POSITIVE, f"0.05678 ({words}, 0.03601--0.07894)")
                self.assertClean(report)
                self.assertEqual([item["qualifiers"] for item in report["intervals"]], [recorded])

    def test_an_unqualified_interval_is_recorded_as_such(self):
        report = self.report(POSITIVE, "0.05678 (95\\% CI, 0.03601--0.07894)")
        self.assertEqual(report["intervals_by_qualifier"], {"unqualified": 1})

    def test_to_separates_the_bounds_and_a_reference_may_close_the_parenthesis(self):
        self.assertClean(self.report(NEGATIVE, "$-0.05678$ (95\\% CI, $-0.07894$ to $-0.03601$)"))
        self.assertClean(self.report(POSITIVE, "0.05678 (marginal 95\\% CI, 0.03601--0.07894; Fig.~\\ref{fig:a})"))

    def test_a_whisker_multiplier_is_a_legend_convention_in_any_legend(self):
        for phrase in ("whiskers within 1.5 interquartile ranges", "whiskers beyond 3 IQR"):
            with self.subTest(phrase=phrase):
                report = run(self.case, [POSITIVE], SUPPORTS,
                             f"\\begin{{figure}}\\caption{{Boxes show the median; {phrase}.}}\\end{{figure}}\n"
                             + CONVENTION)
                self.assertEqual(report["untraced"], [])

    def test_an_interquartile_quantity_is_still_a_number(self):
        report = run(self.case, [POSITIVE], SUPPORTS,
                     f"The spread is 1.5 interquartile ranges wide. {CONVENTION}")
        self.assertEqual([item["token"] for item in report["untraced"]], ["1.5"])


class TheRelaxationDoesNotBlindTheChecker(Fixture):

    def test_a_point_with_no_interval_is_still_bare_whatever_its_sign_or_spacing(self):
        for text in (r"\textminus{}0.05678", r"$0.05678\;$", "0.05678 (marginal 95\\% CI, 0.03601)"):
            with self.subTest(text=text):
                quantity = NEGATIVE if "textminus" in text else POSITIVE
                self.assertIn("bare_point_estimate", self.report(quantity, text)["findings_by_check"])

    def test_a_wrong_interval_is_still_refused_in_the_new_notation(self):
        for text in (r"\textminus{}0.05678 [\textminus{}0.07894, \textminus{}0.02000]",
                     r"$0.05678\;[0.03601,\ 0.09000]$",
                     "0.05678 (marginal 95\\% CI, 0.03601 to 0.09000)"):
            with self.subTest(text=text):
                quantity = NEGATIVE if "textminus" in text else POSITIVE
                self.assertIn("point_interval_mismatch", self.report(quantity, text)["findings_by_check"])

