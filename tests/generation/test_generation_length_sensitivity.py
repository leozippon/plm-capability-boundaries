"""Arithmetic of the length reweighting, on a fixture small enough to check by hand.

Equal lengths sit in one declared bin, so post-stratification has to reproduce
the unweighted rate. A checkpoint that is recognised only inside a bin the
reference barely uses has to be pulled down toward the bin the reference
actually occupies. A bin under the product floor contributes no rate.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.generation.generation_evidence import POLICY  # noqa: E402


def _load():
    path = REPO_ROOT / "scripts" / "capability" / "generation/generation_length_sensitivity.py"
    spec = importlib.util.spec_from_file_location("generation_length_sensitivity", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LENGTH = _load()


class LengthReweighting(unittest.TestCase):
    def test_bins_are_the_declared_generation_evidence_strata(self) -> None:
        self.assertEqual(
            LENGTH.DECLARED_LENGTH_BINS,
            tuple(tuple(pair) for pair in POLICY["length_strata"]),
        )

    def test_equal_lengths_reproduce_the_unweighted_rate(self) -> None:
        lengths = [100] * 12
        masses = LENGTH.reference_bin_masses(lengths)
        self.assertEqual(masses["16_128"], 1.0)
        self.assertEqual(sum(masses.values()), 1.0)
        result = LENGTH.reweight_difference(
            {"16_128": 3},
            {"16_128": 9},
            {"16_128": 12},
            masses,
        )
        self.assertTrue(result["defined"])
        self.assertEqual(result["model_rate"], 3 / 12)
        self.assertEqual(result["fragment_rate"], 9 / 12)
        self.assertEqual(result["model_minus_fragment"], 3 / 12 - 9 / 12)
        self.assertEqual(result["reference_mass_retained"], 1.0)

    def test_success_only_in_a_rare_bin_is_downweighted(self) -> None:
        masses = {
            "16_128": 0.95,
            "129_256": 0.0,
            "257_512": 0.0,
            "513_1024": 0.05,
        }
        counts = {"16_128": 8, "513_1024": 8}
        # Every success sits in the bin that carries 0.05 of the reference.
        result = LENGTH.reweight_difference(
            {"16_128": 0, "513_1024": 8},
            {"16_128": 0, "513_1024": 0},
            counts,
            masses,
        )
        unweighted_difference = 8 / 16
        self.assertTrue(result["defined"])
        self.assertAlmostEqual(result["model_rate"], 0.05)
        self.assertAlmostEqual(result["fragment_rate"], 0.0)
        self.assertAlmostEqual(result["model_minus_fragment"], 0.05)
        self.assertLess(result["model_minus_fragment"], unweighted_difference)
        self.assertAlmostEqual(result["bins_used"]["513_1024"]["renormalized_weight"], 0.05)
        self.assertAlmostEqual(result["bins_used"]["16_128"]["renormalized_weight"], 0.95)

    def test_bin_below_the_minimum_is_not_given_a_rate(self) -> None:
        self.assertGreaterEqual(LENGTH.MINIMUM_PRODUCTS, 2)
        n_products = LENGTH.MINIMUM_PRODUCTS - 1
        masses = {name: 0.0 for name in LENGTH.BIN_NAMES}
        masses["16_128"] = 1.0
        result = LENGTH.reweight_difference(
            {"16_128": n_products},
            {"16_128": 0},
            {"16_128": n_products},
            masses,
        )
        self.assertFalse(result["defined"])
        self.assertIsNone(result["model_rate"])
        self.assertIsNone(result["model_minus_fragment"])
        self.assertEqual(result["bins_used"], {})
        self.assertEqual(result["bins_excluded"][0]["reason"], "fewer_than_minimum_products")
        self.assertEqual(result["bins_excluded"][0]["n_products"], n_products)

    def test_missing_inputs_do_not_invent_rates(self) -> None:
        payload = LENGTH.assemble(Path("/no/such/generation_length_sensitivity_inputs"))
        self.assertEqual(payload["status"], "inputs_missing")
        self.assertIsNone(payload["checkpoints"])
        self.assertIsNone(payload["lineages"])
        self.assertIsNone(payload["panel"])
        self.assertTrue(payload["missing"])


if __name__ == "__main__":
    unittest.main()
