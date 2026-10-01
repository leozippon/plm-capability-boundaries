"""The additive-response residualizer used by the contact-endpoint reaudit.

Three properties, and no more: a held-out family does not train its curve, a
response that is constant inside each additive bin is removed on the held-out
groups, and one two-bin fixture matches arithmetic done by hand. The enrichment
suite is not repeated here.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.interactions.contact_enrichment import (  # noqa: E402
    ADDITIVE_RESPONSE_BINS, cross_fit_binned_residuals, in_sample_binned_adjustment,
    training_curve)


class TheCrossFitDoesNotTrainOnTheHeldOutFamily(unittest.TestCase):
    def test_held_out_families_do_not_train_the_curve(self):
        additive = np.array([0.0, 1.0, 0.0, 1.0])
        epsilon = np.array([1.0, 5.0, 3.0, 7.0])
        groups = np.array(['X', 'X', 'Y', 'Y'])
        curve = training_curve(additive, epsilon, groups, 'Y', n_bins=2)
        self.assertEqual(curve['n_train'], 2)
        self.assertEqual(curve['response'].tolist(), [1.0, 5.0])
        self.assertEqual(curve['edges'].tolist(), [0.0, 0.5, 1.0])

        shifted = epsilon.copy()
        shifted[groups == 'Y'] += 50.0
        again = training_curve(additive, shifted, groups, 'Y', n_bins=2)
        self.assertEqual(again['response'].tolist(), curve['response'].tolist())
        self.assertEqual(again['edges'].tolist(), curve['edges'].tolist())
        self.assertEqual(again['n_train'], len(epsilon) - int((groups == 'Y').sum()))

        original = cross_fit_binned_residuals(additive, epsilon, groups, n_bins=2)
        moved = cross_fit_binned_residuals(additive, shifted, groups, n_bins=2)
        self.assertTrue(np.array_equal(moved[groups == 'Y'] - original[groups == 'Y'],
                                       np.full(2, 50.0)))


class APureAdditiveResponseIsRemovedOnHeldOutGroups(unittest.TestCase):
    def test_a_pure_additive_response_is_removed_on_held_out_groups(self):
        additive = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), 2)
        epsilon = np.tile(np.array([4.0, 4.0, 4.0, 4.0, 9.0, 9.0, 9.0, 9.0]), 2)
        groups = np.array(['A'] * 8 + ['B'] * 8)
        residuals = cross_fit_binned_residuals(additive, epsilon, groups, n_bins=2)
        self.assertTrue(np.allclose(residuals, 0.0))
        self.assertGreater(abs(float(epsilon[groups == 'B'].mean())), 0.0)
        self.assertTrue(np.allclose(residuals[groups == 'B'], 0.0))


class TheTinyFixtureMatchesHandArithmetic(unittest.TestCase):
    def test_the_reproduction_of_one_tiny_fixture_matches_hand_arithmetic(self):
        # Two bins. Group X trains the curve applied to Y, and the reverse.
        # X is (additive 0, epsilon 1) and (1, 5). Its quantile edges are
        # 0, 0.5, 1, so the bin means are 1 and 5. Y is (0, 3) and (1, 7),
        # and the curve fit on Y has bin means 3 and 7.
        # Cross-fit residuals are therefore 1-3, 5-7, 3-1, 7-5.
        # The in-sample bin means on all four cycles are 2 and 6, and those
        # residuals are a different vector. Fitting the full sample must not
        # be what the cross-fit returns.
        self.assertEqual(ADDITIVE_RESPONSE_BINS, 20)
        additive = np.array([0.0, 1.0, 0.0, 1.0])
        epsilon = np.array([1.0, 5.0, 3.0, 7.0])
        groups = np.array(['X', 'X', 'Y', 'Y'])
        residuals = cross_fit_binned_residuals(additive, epsilon, groups, n_bins=2)
        self.assertEqual(residuals.tolist(), [-2.0, -2.0, 2.0, 2.0])
        in_sample, _curve = in_sample_binned_adjustment(additive, epsilon, n_bins=2)
        self.assertEqual(in_sample.tolist(), [-1.0, -1.0, 1.0, 1.0])
        self.assertNotEqual(residuals.tolist(), in_sample.tolist())


if __name__ == '__main__':
    unittest.main()
