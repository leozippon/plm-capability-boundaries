"""Conditions that must hold when two pairwise baselines are compared on depth.

A depth comparison is only readable if both sides describe the same frozen
cohort, if the band labels it reports use the frozen edges rather than
re-derived ones, and if a background without a fitted model is counted as
unsupported rather than folded into the fitted quantiles. The retrieval
quantity is the retrieval gate's own definition, so it is tested against that
definition and not against the current implementation's convenience.
"""
from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location(
    "compare_pairwise_baseline_depth",
    ROOT / "scripts" / "capability" / "interactions" / "compare_pairwise_baseline_depth.py")
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)


def baseline(names: dict[str, dict]) -> dict:
    return {"schema": "pairwise_sequence_baseline_v1",
            "summary": {}, "backgrounds": list(names.values())}


def fitted(name: str, *, neff: float, rows: int, length: int = 40) -> dict:
    return {"name": name, "status": "fitted", "length": length, "rows": rows,
            "hits_retrieved": rows, "retrieval_ceiling_reached": False,
            "train_rows": rows - 5, "held_out_rows": 5,
            "neff_at_30_percent_identity": neff,
            "effective_depth_per_site": round(neff / length, 4),
            "selected_lambda_e": 0.8, "held_out_gain_over_independent_site": 0.5,
            "final_converged": True,
            "contrasts": [{"positions": [1, 20], "separation": "10+",
                           "pairwise_contrast": 0.25,
                           "independent_site_contrast": 1e-15}]}


def unsupported(name: str, *, rows: int = 0, length: int = 40) -> dict:
    return {"name": name, "status": "insufficient_homolog_support", "length": length,
            "rows": rows, "hits_retrieved": rows, "retrieval_ceiling_reached": False,
            "train_rows": rows, "held_out_rows": 0}


class BandLabels(unittest.TestCase):
    def test_the_depth_decades_are_read_at_their_edges(self):
        self.assertEqual(compare.depth_band(None, has_support=False), "no_homolog_support")
        self.assertEqual(compare.depth_band(0.0, has_support=True), "neff_lt10")
        self.assertEqual(compare.depth_band(9.999, has_support=True), "neff_lt10")
        self.assertEqual(compare.depth_band(10.0, has_support=True), "neff_10_100")
        self.assertEqual(compare.depth_band(99.999, has_support=True), "neff_10_100")
        self.assertEqual(compare.depth_band(100.0, has_support=True), "neff_100_1000")
        self.assertEqual(compare.depth_band(999.999, has_support=True), "neff_100_1000")
        self.assertEqual(compare.depth_band(1000.0, has_support=True), "neff_ge1000")
        self.assertEqual(compare.depth_band(1e9, has_support=True), "neff_ge1000")

    def test_a_reconstructed_row_set_with_zero_depth_is_not_no_support(self):
        """Neff 0.0 with rows retrieved is shallow support, not absent support.

        The two are different facts: one background retrieved homologs that all
        fall below the 30% identity floor, another retrieved none at all.
        """

        self.assertEqual(compare.depth_band(0.0, has_support=True), "neff_lt10")
        self.assertEqual(compare.depth_band(0.0, has_support=False), "no_homolog_support")

    def test_the_coupling_parameter_count_is_the_pair_count_over_twenty_states(self):
        self.assertEqual(compare.coupling_parameters(2), 400)
        self.assertEqual(compare.coupling_parameters(37), 37 * 36 // 2 * 400)
        self.assertEqual(compare.coupling_parameters(72), 72 * 71 // 2 * 400)
        # The range the frozen cohort spans, quoted in the readiness record.
        self.assertEqual(compare.coupling_parameters(37), 266400)
        self.assertEqual(compare.coupling_parameters(72), 1022400)


class RetrievalQuantity(unittest.TestCase):
    def rows(self, directory: Path, lines: list[str]) -> Path:
        path = directory / "hits.tsv"
        path.write_text("".join(line + "\n" for line in lines))
        return path

    def test_identity_over_query_is_nident_over_qlen_not_the_reported_pident(self):
        """pident is over alignment columns; the gate's band is over the query.

        The hit below is 100% identical over its 20 aligned columns but covers
        only half of a 40-residue query, so the quantity must read 50.0.
        """

        line = "\t".join(["q1", "s1", "100.0", "20", "20", "1", "20", "40", "60",
                          "1e-9", "80", "AAAA", "AAAA"])
        with tempfile.TemporaryDirectory() as raw:
            hits = self.rows(Path(raw), [line])
            best = compare.max_identity_over_query(hits, {"q1", "q2"})
        self.assertAlmostEqual(best["q1"], 50.0)
        self.assertEqual(best["q2"], 0.0)

    def test_the_maximum_over_hits_is_taken(self):
        lines = ["\t".join(["q1", s, "100.0", "20", n, "1", "20", "40", "60",
                            "1e-9", "80", "A", "A"])
                 for s, n in (("s1", "10"), ("s2", "34"), ("s3", "12"))]
        with tempfile.TemporaryDirectory() as raw:
            hits = self.rows(Path(raw), lines)
            best = compare.max_identity_over_query(hits, {"q1"})
        self.assertAlmostEqual(best["q1"], 85.0)

    def test_a_row_that_cannot_hold_the_quantity_is_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            hits = self.rows(Path(raw), ["q1\ts1\t100.0"])
            with self.assertRaises(ValueError) as raised:
                compare.max_identity_over_query(hits, {"q1"})
        self.assertIn("nident", str(raised.exception))

    def test_a_non_positive_query_length_is_refused_rather_than_divided_by(self):
        line = "\t".join(["q1", "s1", "100.0", "20", "20", "1", "20", "0", "60",
                          "1e-9", "80"])
        with tempfile.TemporaryDirectory() as raw:
            hits = self.rows(Path(raw), [line])
            with self.assertRaises(ValueError) as raised:
                compare.max_identity_over_query(hits, {"q1"})
        self.assertIn("qlen", str(raised.exception))


class MatchedSupport(unittest.TestCase):
    def run_compare(self, before: dict, after: dict, directory: Path) -> Path:
        line = "\t".join(["a", "s1", "100.0", "10", "10", "1", "10", "40", "60",
                          "1e-9", "80", "A", "A"])
        hits = directory / "hits.tsv"
        hits.write_text(line + "\n")
        (directory / "before.json").write_text(json.dumps(before))
        (directory / "after.json").write_text(json.dumps(after))
        out = directory / "comparison.json"
        sys.argv = ["compare", "--before", str(directory / "before.json"),
                    "--after", str(directory / "after.json"),
                    "--before-hits", str(hits), "--after-hits", str(hits),
                    "--out", str(out)]
        compare.main()
        return out

    def test_two_baselines_over_different_backgrounds_are_refused(self):
        """The negative path that makes every quantity below meaningful.

        Comparing depth across two different background sets would report a
        change in the cohort as a change in depth.
        """

        before = baseline({"a": fitted("a", neff=50.0, rows=100)})
        after = baseline({"a": fitted("a", neff=50.0, rows=100),
                          "b": fitted("b", neff=50.0, rows=100)})
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(ValueError) as raised:
                self.run_compare(before, after, Path(raw))
        self.assertIn("same frozen cohort", str(raised.exception))

    def test_an_input_that_is_not_a_baseline_record_is_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            with self.assertRaises(ValueError) as raised:
                self.run_compare({"schema": "something_else", "backgrounds": []},
                                 baseline({}), directory)
        self.assertIn("not a pairwise sequence baseline", str(raised.exception))

    def test_an_unfitted_background_is_counted_unsupported_not_averaged_in(self):
        before = baseline({"a": fitted("a", neff=50.0, rows=100),
                           "b": unsupported("b", rows=3)})
        after = baseline({"a": fitted("a", neff=400.0, rows=900),
                          "b": fitted("b", neff=30.0, rows=60)})
        with tempfile.TemporaryDirectory() as raw:
            out = self.run_compare(before, after, Path(raw))
            report = json.loads(out.read_text())
        self.assertEqual(report["support_floor_cleared"], {"before": 1, "after": 2})
        self.assertEqual(report["newly_clearing_the_support_floor"], ["b"])
        self.assertEqual(report["no_longer_clearing_the_support_floor"], [])
        self.assertEqual(report["summary"]["before"]["backgrounds_fitted"], 1)
        self.assertEqual(report["summary"]["before"]["neff_quantiles"], [50.0] * 5)
        self.assertEqual(report["summary"]["after"]["neff_quantiles"][0], 30.0)
        self.assertEqual(report["summary"]["after"]["neff_quantiles"][-1], 400.0)

    def test_a_background_that_loses_its_fit_is_named_rather_than_netted_out(self):
        before = baseline({"a": fitted("a", neff=50.0, rows=100),
                           "b": fitted("b", neff=50.0, rows=100)})
        after = baseline({"a": fitted("a", neff=50.0, rows=100),
                          "b": unsupported("b", rows=4)})
        with tempfile.TemporaryDirectory() as raw:
            out = self.run_compare(before, after, Path(raw))
            report = json.loads(out.read_text())
        self.assertEqual(report["no_longer_clearing_the_support_floor"], ["b"])
        self.assertEqual(report["support_floor_cleared"], {"before": 2, "after": 1})

    def test_a_band_move_is_reported_for_the_gate_rather_than_applied(self):
        before = baseline({"a": fitted("a", neff=50.0, rows=100)})
        after = baseline({"a": fitted("a", neff=500.0, rows=900)})
        with tempfile.TemporaryDirectory() as raw:
            out = self.run_compare(before, after, Path(raw))
            report = json.loads(out.read_text())
        consequences = report["frozen_band_consequences"]
        self.assertEqual(consequences["backgrounds_whose_depth_band_would_move"], ["a"])
        entry = report["per_background"][0]
        self.assertEqual(entry["frozen_bands_before"]["depth_band"], "neff_10_100")
        self.assertEqual(entry["bands_on_the_deeper_corpus"]["depth_band"], "neff_100_1000")
        self.assertIn("stay frozen", consequences["owner"])


if __name__ == "__main__":
    unittest.main()
