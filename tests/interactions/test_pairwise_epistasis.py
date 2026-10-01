"""Conditions that must hold for the measured double-mutant nonadditivity comparison.

Four of them are what make a reported increment readable at all: identical rows
and identical nested held-group folds across every compared fit, no held-out group
label reaching feature scaling, penalty tuning or the nuisance calibration,
control sets that are genuinely nested, and the independent-site double contrast
measured at its construction zero rather than replaced. The remaining tests pin
the roster freeze, the canonical state order, the weighting unit, the tokenisation
descriptors and the nuisance control's negative path.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from src.capability.interactions.pairwise_epistasis import (
    ADDITIONS, ARM_DTYPE, CONTROL_SETS, C_BLOCKS, DESIGN_PLAN, FEATURE_BLOCKS,
    PRODUCTION_BATCH_SIZE, PROFILE_FEATURE_ORDER, PROJECTION_DIM, PROJECTION_SEED,
    ROSTER, SPLIT_SEEDS, TOKENISATION_FEATURE_ORDER, TOKENISATION_STRATUM,
    cycle_contrast, cycle_control_features, cycle_states, design_blocks, design_names,
    extraction_plan, group_errors, group_spearman, interval,
    kish_effective_site_pairs, measured_singles_correction, measured_singles_predict,
    nested_compare, nuisance_cycle, plan_digest, projection_matrices,
    require_recorded_family_folds, row_identity, state_features, tokenisation_features)
from src.capability.readouts.readout_analysis import family_folds, row_weights

ROOT = Path(__file__).resolve().parents[2]


def load_fit_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'fit_pairwise_epistasis', ROOT / 'scripts/capability/interactions/fit_pairwise_epistasis.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COHORT = ROOT / 'archive/logs/R3/pairwise_cohort_20260924/cohort.json'
BASELINE = ROOT / 'archive/logs/R3/pairwise_homologs_20260924/baseline/baseline_q.json'


def synthetic_panel(*, groups=12, pairs=2, cycles=6, width=3, seed=7, response=0.0,
                    observable_level=False):
    """A cycle panel whose absolute states are additive plus a global response.

    ``response`` scales a monotone cubic response applied to the additive state
    level, which is the nuisance class G exists to absorb. At ``response = 0`` the
    states are exactly additive, so epsilon is zero everywhere.

    ``observable_level`` appends the background's own wild-type stability to every
    state descriptor. It exists only so that the first stage's absolute level is
    predictable inside a unit test; the experiment itself never supplies a measured
    value as a predictor input.
    """

    rng = np.random.default_rng(seed)
    row_group, row_pair, epsilon, cycle_states_index = [], [], [], []
    features, y, state_group, state_background = [], [], [], []
    blocks = {'ident': [], 'geom': [], 'comp': [], 'local': [], 'prof': [],
              'T': [], 'M': [], 'M1': [], 'R': [], 'R1': [], 'Q': []}
    for index in range(groups):
        group, background = f'g{index:02d}', f'b{index:02d}'
        base = len(y)
        level = rng.normal(0, 1)
        single = rng.normal(0, 1, size=(pairs, 2))
        states = [('wt', 0.0)]
        for pair in range(pairs):
            for cycle in range(cycles):
                states.append((f'p{pair}c{cycle}a', single[pair, 0] + 0.1 * cycle))
                states.append((f'p{pair}c{cycle}b', single[pair, 1] - 0.1 * cycle))
                states.append((f'p{pair}c{cycle}ab', single[pair, 0] + single[pair, 1]))
        def curve(value):
            return value + response * value ** 3
        for name, additive in states:
            descriptor = rng.normal(0, 1, size=width)
            features.append(np.r_[additive, descriptor])
            y.append(curve(level + additive))
            state_group.append(group)
            state_background.append(background)
        offset = 1
        for pair in range(pairs):
            for cycle in range(cycles):
                wild, low, high, double = base, base + offset, base + offset + 1, base + offset + 2
                offset += 3
                cycle_states_index.append([wild, low, high, double])
                row_group.append(group)
                row_pair.append(f'{background}:{pair}')
                epsilon.append(y[double] - y[low] - y[high] + y[wild])
                for key, size in (('ident', 4), ('geom', 2), ('comp', 2), ('local', 2),
                                  ('prof', 2), ('T', 2), ('M', 1), ('M1', 2), ('R', 3),
                                  ('R1', 4), ('Q', 1)):
                    blocks[key].append(rng.normal(0, 1, size=size))
    panel = {'group': np.asarray(row_group), 'site_pair': np.asarray(row_pair),
             'background': np.asarray([p.split(':')[0] for p in row_pair]),
             'separation': np.asarray(['10+'] * len(row_group)),
             'epsilon': np.asarray(epsilon, float),
             'blocks': {k: np.asarray(v, float) for k, v in blocks.items()},
             'cycle_states': np.asarray(cycle_states_index),
             'states': {'features': np.asarray(features, float), 'y': np.asarray(y, float),
                        'group': np.asarray(state_group), 'background': np.asarray(state_background)}}
    if observable_level:
        labels = panel['states']['group']
        wild = np.asarray([panel['states']['y'][np.flatnonzero(labels == label)[0]]
                           for label in labels])
        panel['states']['features'] = np.column_stack([panel['states']['features'], wild])
    panel['states']['weight'] = row_weights(panel['states']['background'], panel['states']['group'])
    return panel


class RosterIsFrozenAndComplete(unittest.TestCase):
    def test_roster_is_the_unconditioned_panel_without_a_conditioned_arm(self):
        self.assertEqual(len(ROSTER), 33)
        self.assertEqual(len(set(ROSTER)), 33)
        self.assertNotIn('zymctrl', ROSTER)
        self.assertEqual(set(TOKENISATION_STRATUM), set(ROSTER))
        self.assertEqual(set(TOKENISATION_STRATUM.values()), {'amino_acid', 'byte', 'bpe'})

    def test_the_panel_is_extracted_at_batch_size_one_in_the_admitted_precision(self):
        self.assertEqual(PRODUCTION_BATCH_SIZE, 1)
        self.assertEqual(ARM_DTYPE, {'progen3-3b': 'bfloat16', 'progen3-112m': 'bfloat16'})

    def test_the_projection_depends_only_on_width_and_the_recorded_seed(self):
        first = projection_matrices(64)
        second = projection_matrices(64)
        self.assertEqual(len(first), len(FEATURE_BLOCKS))
        for left, right in zip(first, second):
            self.assertEqual(left.shape, (64, PROJECTION_DIM))
            np.testing.assert_array_equal(left, right)
        expected = np.random.default_rng(PROJECTION_SEED + 2).normal(
            0, 1 / np.sqrt(PROJECTION_DIM), size=(64, PROJECTION_DIM)).astype(np.float32)
        np.testing.assert_array_equal(first[2], expected)
        self.assertEqual(projection_matrices(96)[0].shape, (96, PROJECTION_DIM))


class ControlSetsAreGenuinelyNested(unittest.TestCase):
    def test_every_declared_contrast_adds_columns_to_its_own_baseline(self):
        for support, pairs in load_fit_module().CONTRASTS.items():
            for base, augmented in pairs:
                if base == 'ADDITIVE_NULL':
                    continue
                left, right = set(design_blocks(base)), set(design_blocks(augmented))
                self.assertLess(left, right, f'{augmented} does not strictly contain {base}')

    def test_the_control_ladder_is_a_chain(self):
        chain = ('C', 'C_G', 'C_G_T', 'C_G_T_Q')
        for lower, upper in zip(chain[:-1], chain[1:]):
            self.assertLess(set(CONTROL_SETS[lower]), set(CONTROL_SETS[upper]))
        for name in ('C_G_T_M1', 'C_G_T_R1'):
            self.assertLess(set(CONTROL_SETS['C_G_T']), set(CONTROL_SETS[name]))

    def test_likelihood_and_representation_are_added_separately(self):
        self.assertEqual(ADDITIONS[''], ())
        self.assertEqual(ADDITIONS['M'], ('M',))
        self.assertEqual(ADDITIONS['R'], ('R',))
        for support, plan in DESIGN_PLAN.items():
            for control, additions in plan.items():
                self.assertIn('', additions, f'{control} has no unaugmented baseline')
                if 'MR' in additions:
                    self.assertIn('M', additions, f'{control} reports M and R only jointly')
                    self.assertIn('R', additions, f'{control} reports M and R only jointly')
        self.assertEqual(DESIGN_PLAN['all']['C_G_T_M1'], ('', 'M'))
        self.assertEqual(DESIGN_PLAN['all']['C_G_T_R1'], ('', 'R'))

    def test_q_never_appears_outside_its_own_support(self):
        self.assertNotIn('Q', {block for name in design_names('all')
                               for block in design_blocks(name)})
        self.assertIn('Q', {block for name in design_names('q') for block in design_blocks(name)})



class SupportAndFoldsAreIdenticalAcrossComparedFits(unittest.TestCase):
    def test_every_design_shares_one_row_set_and_one_fold_assignment(self):
        panel = synthetic_panel()
        outcome = nested_compare(panel, 'all', seed=SPLIT_SEEDS[0])
        for name, prediction in outcome['predictions'].items():
            self.assertEqual(len(prediction), len(panel['epsilon']), name)
            self.assertTrue(np.isfinite(prediction).all(), name)
        held = [fold['held_groups'] for fold in outcome['folds']]
        self.assertEqual(sorted(group for fold in held for group in fold),
                         sorted(set(panel['group'])))
        for fold in outcome['folds']:
            self.assertEqual(set(fold['alpha']), set(design_names('all')))

    def test_the_fold_assignment_is_a_partition_of_the_declared_group_universe(self):
        groups = np.asarray([f'g{i:02d}' for i in range(20)])
        folds = family_folds(groups, 5, SPLIT_SEEDS[0])
        self.assertEqual(len(folds), 5)
        self.assertEqual(sorted(g for fold in folds for g in fold), sorted(groups))
        self.assertNotEqual(folds, family_folds(groups, 5, SPLIT_SEEDS[1]))

    def test_a_changed_row_set_changes_the_support_identity(self):
        panel = synthetic_panel()
        other = synthetic_panel(seed=8)
        def identity(rows):
            import hashlib
            return hashlib.sha256('\n'.join(
                f'{g}|{p}|{e!r}' for g, p, e in zip(rows['group'], rows['site_pair'],
                                                    rows['epsilon'])).encode()).hexdigest()
        self.assertEqual(identity(panel), identity(synthetic_panel()))
        self.assertNotEqual(identity(panel), identity(other))


class TheRowIdentityDigestDependsOnTheRowsAndNothingElse(unittest.TestCase):
    """The digest is how a refit proves it used identical rows, so a rendering that
    varies with the interpreter would make the check unreliable in both directions."""

    class NumpyTwoScalar(float):
        """A float that reprs the way a NumPy 2 scalar does."""

        def __repr__(self):
            return f'np.float64({float(self)!r})'

    def test_the_digest_is_stable_across_scalar_representations(self):
        groups = ['g0', 'g0', 'g1']
        pairs = ['b0:1-2', 'b0:1-2', 'b1:3-40']
        values = [0.5, -1.25, 0.1234567890123]
        plain = row_identity(groups, pairs, values)
        self.assertEqual(plain, row_identity(groups, pairs, np.asarray(values, dtype=np.float64)))
        self.assertEqual(plain, row_identity(
            groups, pairs, [self.NumpyTwoScalar(value) for value in values]))
        self.assertEqual(plain, row_identity(
            groups, pairs, tuple(np.float64(value) for value in values)))
        with self.assertRaises(TypeError):
            row_identity(groups, pairs, (value for value in values))

    def test_a_numpy_scalar_rendering_would_not_have_been_stable(self):
        scalars = [self.NumpyTwoScalar(0.5)]
        self.assertNotEqual(repr(scalars[0]), repr(0.5))
        self.assertEqual(row_identity(['g0'], ['b0:1-2'], scalars),
                         row_identity(['g0'], ['b0:1-2'], [0.5]))

    def test_the_digest_moves_with_any_row_field(self):
        base = row_identity(['g0', 'g1'], ['b0:1-2', 'b1:1-2'], [0.5, 0.25])
        self.assertNotEqual(base, row_identity(['g0', 'g2'], ['b0:1-2', 'b1:1-2'], [0.5, 0.25]))
        self.assertNotEqual(base, row_identity(['g0', 'g1'], ['b0:1-2', 'b1:1-3'], [0.5, 0.25]))
        self.assertNotEqual(base, row_identity(['g0', 'g1'], ['b0:1-2', 'b1:1-2'], [0.5, 0.26]))
        self.assertNotEqual(base, row_identity(['g1', 'g0'], ['b1:1-2', 'b0:1-2'], [0.25, 0.5]))

    def test_misaligned_rows_are_refused(self):
        with self.assertRaises(ValueError):
            row_identity(['g0', 'g1'], ['b0:1-2'], [0.5, 0.25])


class TheEffectiveSitePairCountUsesTheEstimatorsOwnWeighting(unittest.TestCase):
    def test_it_is_the_kish_count_of_the_weights_the_fits_apply(self):
        panel = synthetic_panel(groups=6, pairs=3, cycles=4)
        weights = row_weights(panel['site_pair'], panel['group'])
        pairs = sorted(set(panel['site_pair']))
        per_pair = np.asarray([weights[panel['site_pair'] == pair].sum() for pair in pairs])
        expected = float(per_pair.sum() ** 2 / (per_pair ** 2).sum())
        self.assertAlmostEqual(
            kish_effective_site_pairs(panel['group'], panel['site_pair']), expected)

    def test_it_differs_from_a_cycle_share_convention_on_unequal_site_pairs(self):
        groups = np.asarray(['g0'] * 11 + ['g1'] * 2)
        pairs = np.asarray(['b0:1-2'] * 10 + ['b0:3-40'] + ['b1:1-2', 'b1:3-40'])
        equal = kish_effective_site_pairs(groups, pairs)
        labels = sorted(set(groups))
        share = []
        for label in labels:
            rows = groups == label
            for pair in sorted(set(pairs[rows])):
                share.append((1 / len(labels)) * (pairs[rows] == pair).sum() / rows.sum())
        share = np.asarray(share)
        self.assertNotAlmostEqual(equal, float(share.sum() ** 2 / (share ** 2).sum()), places=2)
        self.assertAlmostEqual(equal, 4.0)


class HeldOutLabelsNeverReachAnyFit(unittest.TestCase):
    def test_permuting_held_out_group_labels_leaves_every_prediction_unchanged(self):
        panel = synthetic_panel(response=0.4)
        outcome = nested_compare(panel, 'all', seed=SPLIT_SEEDS[0])
        held = set(outcome['folds'][0]['held_groups'])
        permuted = {key: (value.copy() if isinstance(value, np.ndarray) else value)
                    for key, value in panel.items()}
        permuted['blocks'] = panel['blocks']
        permuted['states'] = dict(panel['states'])
        rng = np.random.default_rng(3)
        rows = np.flatnonzero(np.isin(panel['group'], list(held)))
        permuted['epsilon'] = panel['epsilon'].copy()
        permuted['epsilon'][rows] = rng.permutation(panel['epsilon'][rows])
        states = np.flatnonzero(np.isin(panel['states']['group'], list(held)))
        permuted['states']['y'] = panel['states']['y'].copy()
        permuted['states']['y'][states] = rng.permutation(panel['states']['y'][states])
        after = nested_compare(permuted, 'all', seed=SPLIT_SEEDS[0])
        for name in design_names('all'):
            np.testing.assert_array_equal(
                outcome['predictions'][name][rows], after['predictions'][name][rows],
                err_msg=f'{name} responds to a held-out group label')

    def test_a_changed_training_label_does_move_the_held_out_prediction(self):
        panel = synthetic_panel(response=0.4)
        outcome = nested_compare(panel, 'all', seed=SPLIT_SEEDS[0])
        held = set(outcome['folds'][0]['held_groups'])
        rows = np.flatnonzero(np.isin(panel['group'], list(held)))
        training = np.flatnonzero(~np.isin(panel['group'], list(held)))
        moved = dict(panel)
        moved['epsilon'] = panel['epsilon'].copy()
        moved['epsilon'][training] += 1.0
        after = nested_compare(moved, 'all', seed=SPLIT_SEEDS[0])
        self.assertFalse(np.allclose(outcome['predictions']['C'][rows],
                                     after['predictions']['C'][rows]))


class TheNonlinearAdditiveNuisanceBehavesAsDeclared(unittest.TestCase):
    def test_the_uncalibrated_additive_cycle_is_measured_at_its_construction_zero(self):
        panel = synthetic_panel(response=0.4)
        held = family_folds(panel['group'], 5, SPLIT_SEEDS[0])[0]
        train = panel['group'][~np.isin(panel['group'], held)]
        inner = family_folds(train, 4, SPLIT_SEEDS[0] + 100)
        block, diagnostics = nuisance_cycle(panel['states'], panel['cycle_states'],
                                            np.asarray(held), inner)
        self.assertEqual(block.shape, (len(panel['epsilon']), 5))
        self.assertLess(diagnostics['uncalibrated_additive_cycle_max_absolute'], 1e-9)
        self.assertGreater(diagnostics['held_out_states'], 0)

    def test_no_control_set_manufactures_interaction_on_an_exactly_additive_panel(self):
        panel = synthetic_panel(response=0.0)
        self.assertLess(np.abs(panel['epsilon']).max(), 1e-9)
        outcome = nested_compare(panel, 'all', seed=SPLIT_SEEDS[0])
        for name, prediction in outcome['predictions'].items():
            self.assertLess(np.abs(prediction).max(), 1e-9, name)

    def test_an_affine_calibration_leaves_the_calibrated_cycle_at_zero(self):
        panel = synthetic_panel(groups=20, response=0.0, observable_level=True)
        held = family_folds(panel['group'], 5, SPLIT_SEEDS[0])[0]
        train = panel['group'][~np.isin(panel['group'], held)]
        inner = family_folds(train, 4, SPLIT_SEEDS[0] + 100)
        block, diagnostics = nuisance_cycle(panel['states'], panel['cycle_states'],
                                            np.asarray(held), inner)
        self.assertGreater(diagnostics['held_out_weighted_r2'], 0.99)
        self.assertLess(np.abs(block[:, 0]).max(), 0.05)

    def test_a_global_monotone_response_produces_a_nonzero_calibrated_cycle(self):
        panel = synthetic_panel(groups=20, response=0.6, observable_level=True)
        self.assertGreater(np.abs(panel['epsilon']).max(), 0.1)
        held = family_folds(panel['group'], 5, SPLIT_SEEDS[0])[0]
        train = panel['group'][~np.isin(panel['group'], held)]
        inner = family_folds(train, 4, SPLIT_SEEDS[0] + 100)
        block, diagnostics = nuisance_cycle(panel['states'], panel['cycle_states'],
                                            np.asarray(held), inner)
        self.assertGreater(np.abs(block[:, 0]).max(), 1e-3)
        self.assertGreater(diagnostics['calibration_response_range_kcal_mol'], 0.0)
        rows = np.flatnonzero(np.isin(panel['group'], held))
        self.assertGreater(float(np.corrcoef(block[rows, 0], panel['epsilon'][rows])[0, 1]), 0.1)

    def test_the_nuisance_refuses_a_fold_with_an_empty_side(self):
        panel = synthetic_panel()
        held = sorted(set(panel['group']))
        with self.assertRaises(ValueError):
            nuisance_cycle(panel['states'], panel['cycle_states'], np.asarray(held),
                           family_folds(panel['group'], 4, SPLIT_SEEDS[0]))


class WeightingUnitIsTheSitePairInsideAnEquallyWeightedGroup(unittest.TestCase):
    def test_weights_sum_to_one_and_give_every_group_the_same_total(self):
        panel = synthetic_panel(groups=5, pairs=3, cycles=4)
        weights = row_weights(panel['site_pair'], panel['group'])
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        for group in set(panel['group']):
            rows = panel['group'] == group
            self.assertAlmostEqual(float(weights[rows].sum()), 1 / 5)
        for pair in set(panel['site_pair']):
            rows = panel['site_pair'] == pair
            self.assertAlmostEqual(float(weights[rows].sum()), 1 / (5 * 3))

    def test_group_mean_squared_error_weights_site_pairs_equally(self):
        groups = np.asarray(['g0'] * 5)
        pairs = np.asarray(['p0'] * 4 + ['p1'])
        epsilon = np.asarray([1., 1., 1., 1., 3.])
        labels, values = group_errors(epsilon, np.zeros(5), groups, pairs)
        self.assertEqual(list(labels), ['g0'])
        self.assertAlmostEqual(values[0], (1.0 + 9.0) / 2)


class TheAdditiveNullHasNoRankOrdering(unittest.TestCase):
    def test_a_constant_prediction_gives_an_undefined_spearman_not_a_zero(self):
        groups = np.asarray(['g0'] * 6 + ['g1'] * 6)
        epsilon = np.arange(12, dtype=float)
        _, values = group_spearman(epsilon, np.zeros(12), groups)
        self.assertEqual(values, [None, None])
        record = interval(values, draws=50)
        self.assertIsNone(record['point'])
        self.assertIsNone(record['interval'])
        self.assertEqual(record['undefined_groups'], 2)
        self.assertIn('undefined', record)

    def test_a_varying_prediction_gives_a_defined_spearman(self):
        groups = np.asarray(['g0'] * 6)
        epsilon = np.arange(6, dtype=float)
        _, values = group_spearman(epsilon, epsilon * 2, groups)
        self.assertAlmostEqual(values[0], 1.0)


class TheCanonicalStateOrderIsDerivedNotTrusted(unittest.TestCase):
    def test_a_swapped_pair_is_reordered_to_the_lower_site_first(self):
        wild = 'ACDEFG'
        low, high = 'AWDEFG', 'ACDEWG'
        double = 'AWDEWG'
        self.assertEqual(cycle_states(wild, [wild, high, low, double]),
                         (wild, low, high, double))
        self.assertEqual(cycle_states(wild, [wild, low, high, double]),
                         (wild, low, high, double))

    def test_a_double_that_does_not_match_its_singles_is_refused(self):
        with self.assertRaises(ValueError):
            cycle_states('ACDEFG', ['ACDEFG', 'AWDEFG', 'ACDEWG', 'AWDEFG'])
        with self.assertRaises(ValueError):
            cycle_states('ACDEFG', ['AWDEFG', 'AWDEFG', 'ACDEWG', 'AWDEWG'])

    def test_the_four_state_contrast_is_the_declared_combination(self):
        values = np.asarray([[1.0], [2.0], [4.0], [10.0]])
        states = np.asarray([[0, 1, 2, 3]])
        np.testing.assert_allclose(cycle_contrast(values, states), [[10.0 - 2.0 - 4.0 + 1.0]])


class TokenisationDescriptorsSeeSegmentationAndShift(unittest.TestCase):
    def test_a_residue_level_tokenizer_gives_a_zero_token_cycle(self):
        ids = [[1, 2, 3, 4], [1, 9, 3, 4], [1, 2, 3, 9], [1, 9, 3, 9]]
        counts = np.asarray([4, 4, 4, 4])
        values = tokenisation_features(counts, ids, (0, 1, 2, 3), 4)
        self.assertEqual(len(values), len(TOKENISATION_FEATURE_ORDER))
        self.assertAlmostEqual(values[4], 0.0)
        self.assertAlmostEqual(values[5], 1.0)
        self.assertAlmostEqual(values[11], 2.0)

    def test_a_segmentation_change_moves_the_descriptors(self):
        ids = [[1, 2, 3], [1, 7, 8, 3], [1, 2, 9], [1, 7, 8, 9, 5]]
        counts = np.asarray([3, 4, 3, 5])
        values = tokenisation_features(counts, ids, (0, 1, 2, 3), 6)
        self.assertAlmostEqual(values[4], 5.0 - 4.0 - 3.0 + 3.0)
        self.assertGreater(values[11], 2.0)
        self.assertAlmostEqual(values[14], 0.5)
        self.assertAlmostEqual(values[15], 5.0 / 6.0)


class DeclaredBlocksHaveTheDeclaredShape(unittest.TestCase):
    def test_the_control_block_widths_match_their_declaration(self):
        wild = 'ACDEFGHIKLMNPQRSTVWY' * 2
        low = wild[:3] + 'W' + wild[4:]
        high = wild[:30] + 'W' + wild[31:]
        double = low[:30] + 'W' + low[31:]
        values = cycle_control_features(wild, (wild, low, high, double), (4, 31), '10+', None)
        self.assertEqual(values.shape, (400 + 10 + 40 + 400 + len(PROFILE_FEATURE_ORDER),))
        self.assertAlmostEqual(values[-1], 0.0, msg='an absent profile must not read as available')
        self.assertEqual(list(C_BLOCKS), ['ident', 'geom', 'comp', 'local', 'prof'])

    def test_a_state_descriptor_counts_its_own_substitutions(self):
        wild = 'ACDEFGHIKLMNPQRSTVWY'
        self.assertAlmostEqual(state_features(wild, wild, None)[0], 0.0)
        self.assertAlmostEqual(state_features(wild, 'AWDEFGHIKLMNPQRSTVWY', None)[0], 1.0)
        self.assertAlmostEqual(state_features(wild, 'AWDEFGHIKLMNPQRSTVWW', None)[0], 2.0)

    def test_a_noncanonical_residue_is_refused(self):
        with self.assertRaises(ValueError):
            cycle_control_features('ACDXFG', ('ACDXFG', 'AWDXFG', 'ACDXWG', 'AWDXWG'),
                                   (2, 5), '3-9', None)


class FrozenArtifactsCarryTheirDeclaredProperties(unittest.TestCase):
    @unittest.skipUnless(COHORT.exists(), 'frozen cohort not present on this host')
    def test_the_extraction_plan_carries_no_measurement(self):
        cohort = json.loads(COHORT.read_text())
        plan = extraction_plan(cohort, 'f' * 64)
        text = json.dumps(plan)
        self.assertNotIn('epsilon', text)
        self.assertNotIn('measurements', text)
        self.assertNotIn('source_row', text)
        self.assertEqual(plan['summary']['cycles'], cohort['summary']['cycles'])
        self.assertEqual(plan['summary']['distinct_sequences'],
                         cohort['summary']['distinct_sequences'])
        self.assertEqual(plan['summary']['residues'], cohort['summary']['residues'])
        self.assertEqual(plan['summary']['site_pairs'],
                         cohort['summary']['distinct_site_pairs_across_backgrounds'])
        self.assertEqual(len(plan_digest(plan)), 64)

    @unittest.skipUnless(COHORT.exists(), 'frozen cohort not present on this host')
    def test_the_plan_refuses_an_unbound_cohort_digest(self):
        cohort = json.loads(COHORT.read_text())
        with self.assertRaises(ValueError):
            extraction_plan(cohort, 'short')
        with self.assertRaises(ValueError):
            extraction_plan(dict(cohort, schema='other'), 'f' * 64)

    @unittest.skipUnless(BASELINE.exists(), 'fitted baseline Q not present on this host')
    def test_the_independent_site_double_contrast_is_zero_by_construction(self):
        baseline = json.loads(BASELINE.read_text())
        fitted = [row for row in baseline['backgrounds'] if row['status'] == 'fitted']
        self.assertEqual(len(fitted), 45)
        worst = max(abs(entry['independent_site_contrast'])
                    for row in fitted for entry in row['contrasts'])
        self.assertLess(worst, 1e-10)
        self.assertGreater(max(abs(entry['pairwise_contrast'])
                               for row in fitted for entry in row['contrasts']), 1e-3)


def singles_rows(*, groups=20, cycles=40, seed=11, kind='zero'):
    """Cycles whose epsilon is a chosen function of a measured additive."""

    rng = np.random.default_rng(seed)
    labels, additive, epsilon = [], [], []
    for index in range(groups):
        draw = rng.normal(size=cycles)
        if kind == 'zero':
            response = np.zeros(cycles)
        elif kind == 'monotone':
            response = -2.0 * np.tanh(draw)
        elif kind == 'independent':
            response = rng.normal(size=cycles)
        else:
            raise ValueError(kind)
        labels.append(np.full(cycles, f'g{index:02d}'))
        additive.append(draw)
        epsilon.append(response)
    return np.concatenate(labels), np.concatenate(additive), np.concatenate(epsilon)


class TheMeasuredSinglesCorrectionHoldsFamiliesOut(unittest.TestCase):
    def _correction(self, kind, method):
        labels, additive, epsilon = singles_rows(kind=kind)
        folds = family_folds(labels, 5, SPLIT_SEEDS[0])
        outcome = measured_singles_correction(epsilon, additive, labels, folds, method=method)
        return labels, additive, epsilon, outcome

    def test_an_exactly_additive_panel_leaves_a_zero_residual(self):
        for method in ('isotonic', 'quantile_bins'):
            _, _, epsilon, outcome = self._correction('zero', method)
            self.assertLess(float(np.max(np.abs(epsilon))), 1e-12)
            self.assertLess(float(np.max(np.abs(outcome['residual']))), 1e-12, method)

    def test_a_global_monotone_response_is_removed_on_held_out_families(self):
        for method in ('isotonic', 'quantile_bins'):
            _, _, epsilon, outcome = self._correction('monotone', method)
            self.assertGreater(float(epsilon.var()), 0.1)
            self.assertLess(float(outcome['residual'].var()) / float(epsilon.var()), 0.02, method)
            self.assertGreater(float(np.corrcoef(outcome['prediction'], epsilon)[0, 1]), 0.98, method)

    def test_a_held_out_family_does_not_contribute_to_its_curve(self):
        labels, additive, epsilon = singles_rows(kind='monotone')
        folds = family_folds(labels, 5, SPLIT_SEEDS[0])
        held = folds[0][0]
        rows = labels == held
        for method in ('isotonic', 'quantile_bins'):
            original = measured_singles_correction(epsilon, additive, labels, folds, method=method)
            changed = epsilon.copy()
            changed[rows] = 50.0
            after = measured_singles_correction(changed, additive, labels, folds, method=method)
            np.testing.assert_array_equal(original['prediction'][rows], after['prediction'][rows])
            self.assertFalse(np.allclose(original['prediction'][~rows], after['prediction'][~rows]))

    def test_an_epsilon_independent_of_the_additive_is_left_essentially_unchanged(self):
        for method in ('isotonic', 'quantile_bins'):
            _, _, epsilon, outcome = self._correction('independent', method)
            residual = outcome['residual']
            self.assertGreater(float(residual.var()) / float(epsilon.var()), 0.90, method)
            self.assertGreater(float(np.corrcoef(residual, epsilon)[0, 1]), 0.95, method)

    def test_the_twenty_bin_mean_matches_the_label_instrument_on_its_training_rows(self):
        _, additive, epsilon = singles_rows(kind='monotone')
        prediction, info = measured_singles_predict(
            additive, epsilon, additive, method='quantile_bins', bins=20)
        edges = np.quantile(additive, np.linspace(0.0, 1.0, 21))
        index = np.clip(np.searchsorted(edges, additive, side='right') - 1, 0, 19)
        response = np.zeros(20)
        for bin_index in range(20):
            selected = index == bin_index
            self.assertTrue(selected.any())
            response[bin_index] = epsilon[selected].mean()
        np.testing.assert_allclose(prediction, response[index])
        self.assertEqual(info['empty_bins_filled'], 0)

    def test_a_recorded_partition_that_is_not_the_seeded_one_is_refused(self):
        labels, _, _ = singles_rows(groups=10, cycles=4, kind='zero')
        recorded = family_folds(labels, 5, SPLIT_SEEDS[0])
        matched = require_recorded_family_folds(labels, recorded, n_splits=5, seed=SPLIT_SEEDS[0])
        self.assertEqual(matched['held_groups'], recorded)
        swapped = [recorded[1], recorded[0], *recorded[2:]]
        with self.assertRaises(ValueError):
            require_recorded_family_folds(labels, swapped, n_splits=5, seed=SPLIT_SEEDS[0])

    def test_overlapping_or_incomplete_folds_are_refused(self):
        labels, additive, epsilon = singles_rows(groups=10, cycles=4, kind='zero')
        present = sorted(set(labels.tolist()))
        with self.assertRaises(ValueError):
            measured_singles_correction(epsilon, additive, labels, [present, present[:1]])
        with self.assertRaises(ValueError):
            measured_singles_correction(epsilon, additive, labels, [present[:-1]])


if __name__ == '__main__':
    unittest.main()
