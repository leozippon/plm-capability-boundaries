"""Conditions that must hold for the global-but-additive sequence-context gate.

The first is what makes the gate meaningful at all: neither declared block may
contain a term that reads both mutated sites, at any range, because a block that
did would absorb genuine pairwise interaction and a negative verdict would be
unreadable. The second is that the comparator's receptive field really is
bounded. The third is that this gate's rows, folds, ridge recipe and nuisance
control are the admitted pairwise fit's rather than a second implementation of
them, checked against that fit's own function. The fourth is that no held-out
group label reaches feature scaling, penalty selection or the nuisance
calibration. The rest pin the declaration against the document that pre-declared
it, the frozen weighting unit and the negative paths.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import unittest

import numpy as np

from src.capability.context.global_context import (
    BLOCK_DECLARATION, BOUNDED_FEATURE_ORDER, BOUNDED_INDEX, BOUNDED_RADIUS,
    GATE_CONTRASTS, GATE_CONTROL_SETS, GATE_DESIGNS, GLOBAL_FEATURE_ORDER,
    GATE_BLOCKS, RECONSTRUCTION_CONTROLS, SCALES, apply_indel_exclusion, context_blocks,
    cycle_context_features, declaration_digest, design_blocks, evaluate_gate,
    gate_compare, group_r2, site_context_features)
from src.capability.interactions.pairwise_epistasis import C_BLOCKS, cycle_states, nested_compare
from src.capability.readouts.readout_analysis import family_folds, row_weights

ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / 'archive/docs/R5/gate-global-context.md'
PLAN = ROOT / 'archive/logs/R3/pairwise_epistasis_20260924/extraction_plan.json'
EXCLUSION = ROOT / 'archive/logs/R5/gate_global_context_20260924/indel_exclusion.json'

WILD = ('MKVLTAEGWRDFICQNSYHPLGATVMKDEFRWQSLNIYCPGAHTVEDKRWFMLSQNIYCPGAHTVEDKR')


def load_gate_fit_script():
    path = ROOT / 'scripts/capability/context/fit_global_context_gate.py'
    spec = importlib.util.spec_from_file_location('fit_global_context_gate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flagship_fixture():
    """The admitted pairwise test module's synthetic cycle panel, imported rather
    than copied so this gate is exercised on the same fixture the admitted
    comparison is."""

    path = ROOT / 'tests/interactions/test_pairwise_epistasis.py'
    spec = importlib.util.spec_from_file_location('admitted_pairwise_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.synthetic_panel


def gate_panel(**kwargs):
    panel = flagship_fixture()(**kwargs)
    rng = np.random.default_rng(101)
    rows = len(panel['epsilon'])
    panel['blocks']['X'] = rng.normal(0, 1, size=(rows, 6))
    panel['blocks']['W'] = panel['blocks']['X'][:, :2]
    return panel


def substitute(sequence: str, site: int, residue: str) -> str:
    return sequence[:site] + residue + sequence[site + 1:]


def quartet(wild: str, low: int, high: int, low_residue: str, high_residue: str):
    first = substitute(wild, low, low_residue)
    second = substitute(wild, high, high_residue)
    double = substitute(first, high, high_residue)
    return cycle_states(wild, [wild, first, second, double])


class NeitherBlockCarriesACrossSiteTerm(unittest.TestCase):
    """The always-true condition of this gate."""

    def test_the_block_is_exactly_the_sum_of_its_two_per_site_vectors(self):
        states = quartet(WILD, 9, 40, 'A', 'W')
        block = cycle_context_features(WILD, states, (10, 41))
        expected = (site_context_features(WILD, 9, 'A')
                    + site_context_features(WILD, 40, 'W'))
        self.assertTrue(np.array_equal(block, expected))

    def test_a_site_contribution_does_not_depend_on_the_other_substitution(self):
        """Changing the partner substitution moves the block by exactly the
        partner's own per-site difference. The identity is exact in construction
        and is checked here to floating-point rounding, because subtracting two
        sums re-associates the same three terms."""

        left = cycle_context_features(WILD, quartet(WILD, 9, 40, 'A', 'W'), (10, 41))
        right = cycle_context_features(WILD, quartet(WILD, 9, 40, 'A', 'D'), (10, 41))
        difference = (site_context_features(WILD, 40, 'W')
                      - site_context_features(WILD, 40, 'D'))
        np.testing.assert_allclose(left - right, difference, rtol=1e-12, atol=1e-12)

    def test_a_neighbouring_substitution_inside_the_window_changes_nothing_either(self):
        """The strong case: the partner site sits one residue away, inside every
        declared window, and still contributes no cross term, because the context
        is read off the wild-type background rather than a mutant state."""

        base = site_context_features(WILD, 9, 'A')
        for residue in ('W', 'G', 'K'):
            block = cycle_context_features(WILD, quartet(WILD, 9, 10, 'A', residue), (10, 11))
            np.testing.assert_allclose(block - site_context_features(WILD, 10, residue),
                                       base, rtol=1e-12, atol=1e-12)

    def test_recombining_two_pairs_leaves_the_block_sum_unchanged(self):
        pairs = ((9, 40, 'A', 'W'), (20, 55, 'G', 'K'))
        straight = sum(cycle_context_features(
            WILD, quartet(WILD, low, high, first, second), (low + 1, high + 1))
            for low, high, first, second in pairs)
        crossed = (cycle_context_features(WILD, quartet(WILD, 9, 55, 'A', 'K'), (10, 56))
                   + cycle_context_features(WILD, quartet(WILD, 20, 40, 'G', 'W'), (21, 41)))
        np.testing.assert_allclose(straight, crossed, rtol=1e-12, atol=0)

    def test_every_declared_coordinate_names_one_site_worth_of_content(self):
        for name in GLOBAL_FEATURE_ORDER:
            self.assertFalse(re.search(r'(pair|cross|both|joint|product_of_sites)', name), name)


class TheComparatorReceptiveFieldIsBounded(unittest.TestCase):
    def test_distant_background_content_moves_the_global_block_only(self):
        far = 25
        self.assertGreater(min(abs(far - 9), abs(far - 40)), BOUNDED_RADIUS)
        altered = substitute(WILD, far, 'P' if WILD[far] != 'P' else 'G')
        left = context_blocks(WILD, quartet(WILD, 9, 40, 'A', 'W'), (10, 41))
        right = context_blocks(altered, quartet(altered, 9, 40, 'A', 'W'), (10, 41))
        self.assertTrue(np.array_equal(left['W'], right['W']))
        self.assertFalse(np.array_equal(left['X'], right['X']))

    def test_content_inside_the_window_moves_both_blocks(self):
        near = 12
        self.assertLessEqual(abs(near - 9), BOUNDED_RADIUS)
        altered = substitute(WILD, near, 'P' if WILD[near] != 'P' else 'G')
        left = context_blocks(WILD, quartet(WILD, 9, 40, 'A', 'W'), (10, 41))
        right = context_blocks(altered, quartet(altered, 9, 40, 'A', 'W'), (10, 41))
        self.assertFalse(np.array_equal(left['W'], right['W']))


class TheDeclaredBlocksAreNestedAndPreDeclared(unittest.TestCase):
    def test_the_bounded_block_is_a_column_subset_of_the_global_one(self):
        self.assertEqual(BOUNDED_FEATURE_ORDER,
                         GLOBAL_FEATURE_ORDER[:len(BOUNDED_FEATURE_ORDER)])
        self.assertEqual(BOUNDED_INDEX, tuple(range(len(BOUNDED_FEATURE_ORDER))))
        block = context_blocks(WILD, quartet(WILD, 9, 40, 'A', 'W'), (10, 41))
        self.assertTrue(np.array_equal(block['W'], block['X'][list(BOUNDED_INDEX)]))

    def test_every_gate_control_set_extends_the_admitted_matched_baseline(self):
        for name, blocks in GATE_CONTROL_SETS.items():
            self.assertEqual(blocks[:len(C_BLOCKS) + 2], (*C_BLOCKS, 'G', 'T'), name)
        for base, augmented in GATE_CONTRASTS:
            extra = set(design_blocks(augmented)) - set(design_blocks(base))
            dropped = set(design_blocks(base)) - set(design_blocks(augmented))
            self.assertTrue(extra, (base, augmented))
            if dropped:
                # The one step that swaps a block rather than adding one: the
                # bounded comparator's columns are the leading columns of the
                # global block, so the nesting is in the column space.
                self.assertEqual((dropped, extra), ({'W'}, {'X'}))
                self.assertEqual(BOUNDED_FEATURE_ORDER,
                                 GLOBAL_FEATURE_ORDER[:len(BOUNDED_FEATURE_ORDER)])

    def test_the_declaration_digest_is_the_one_the_document_states(self):
        if not DOCUMENT.is_file():
            self.skipTest('the pre-declaration gate document is archived host-locally '
                          'and is not distributed with the code repository')
        stated = set(re.findall(r'\b[0-9a-f]{64}\b', DOCUMENT.read_text(encoding='utf-8')))
        self.assertIn(declaration_digest(), stated)

    def test_the_declaration_records_the_scales_and_the_additivity_reason(self):
        self.assertEqual(tuple(BLOCK_DECLARATION['scales']), SCALES)
        self.assertEqual(BLOCK_DECLARATION['bounded_radius_residues'], BOUNDED_RADIUS)
        self.assertIn('no coordinate is a function of both sites',
                      BLOCK_DECLARATION['additive_by_construction'])
        self.assertEqual(BLOCK_DECLARATION['blocks']['X']['width'], len(GLOBAL_FEATURE_ORDER))


class TheSupportAndFoldsAreTheAdmittedFits(unittest.TestCase):
    def test_a_shared_design_reproduces_the_admitted_comparison_exactly(self):
        panel = gate_panel(groups=10, pairs=2, cycles=4, response=0.4, observable_level=True)
        gate = gate_compare(panel, seed=20260923)
        admitted = nested_compare(panel, 'all', seed=20260923)
        for mine, theirs in (('CGT', 'C_G_T'), ('CGT_M1', 'C_G_T_M1'),
                             ('CGT_M1+M', 'C_G_T_M1+M')):
            self.assertTrue(np.array_equal(gate['predictions'][mine],
                                           admitted['predictions'][theirs]), mine)
        self.assertEqual([f['held_groups'] for f in gate['folds']],
                         [f['held_groups'] for f in admitted['folds']])
        self.assertTrue(np.array_equal(gate['row_weights'], admitted['row_weights']))

    def test_the_nuisance_control_is_the_admitted_one_fold_for_fold(self):
        panel = gate_panel(groups=10, pairs=2, cycles=4, response=0.4, observable_level=True)
        gate = gate_compare(panel, seed=20260924)
        admitted = nested_compare(panel, 'all', seed=20260924)
        self.assertEqual([f['selected_alpha'] for f in gate['nuisance']],
                         [f['selected_alpha'] for f in admitted['nuisance']])
        self.assertEqual([f['held_out_weighted_r2'] for f in gate['nuisance']],
                         [f['held_out_weighted_r2'] for f in admitted['nuisance']])


class NoHeldOutLabelReachesAFit(unittest.TestCase):
    def test_permuting_one_held_fold_moves_no_held_out_prediction(self):
        panel = gate_panel(groups=10, pairs=2, cycles=4, response=0.4, observable_level=True)
        seed = 20260923
        outcome = gate_compare(panel, seed=seed)
        held = family_folds(panel['group'], 5, seed)[0]
        rng = np.random.default_rng(3)
        permuted = {key: value.copy() if isinstance(value, np.ndarray) else value
                    for key, value in panel.items()}
        permuted['states'] = dict(panel['states'])
        rows = np.flatnonzero(np.isin(panel['group'], held))
        states = np.flatnonzero(np.isin(panel['states']['group'], held))
        permuted['epsilon'] = panel['epsilon'].copy()
        permuted['epsilon'][rows] = rng.permutation(panel['epsilon'][rows])
        permuted['states']['y'] = panel['states']['y'].copy()
        permuted['states']['y'][states] = rng.permutation(panel['states']['y'][states])
        self.assertFalse(np.array_equal(permuted['epsilon'], panel['epsilon']))
        self.assertFalse(np.array_equal(permuted['states']['y'], panel['states']['y']))
        after = gate_compare(permuted, seed=seed)
        for name in GATE_DESIGNS:
            moved = np.max(np.abs(after['predictions'][name][rows]
                                  - outcome['predictions'][name][rows]))
            self.assertEqual(moved, 0.0, f'{name} moved by {moved}')
        for key in outcome['reconstruction']:
            moved = np.max(np.abs(after['reconstruction'][key][rows]
                                  - outcome['reconstruction'][key][rows]))
            self.assertEqual(moved, 0.0, f'{key} moved by {moved}')

    def test_the_reconstruction_targets_are_model_quantities_not_labels(self):
        panel = gate_panel(groups=10, pairs=2, cycles=4)
        for control in RECONSTRUCTION_CONTROLS:
            self.assertNotIn('M', GATE_CONTROL_SETS[control])
            self.assertNotIn('M1', GATE_CONTROL_SETS[control])
        self.assertEqual(panel['blocks']['M1'].shape[1], 2)


class TheReportedQuantitiesBehave(unittest.TestCase):
    def test_a_perfect_and_a_mean_prediction_bracket_the_reconstruction_r2(self):
        target = np.asarray([1.0, 2.0, 3.0, 4.0, 10.0, 12.0])
        groups = np.asarray(['a'] * 4 + ['b'] * 2)
        pairs = np.asarray(['a:1', 'a:1', 'a:2', 'a:2', 'b:1', 'b:1'])
        _, perfect = group_r2(target, target.copy(), groups, pairs)
        self.assertEqual(perfect, [1.0, 1.0])
        centres = np.asarray([2.5, 2.5, 2.5, 2.5, 11.0, 11.0])
        _, flat = group_r2(target, centres, groups, pairs)
        np.testing.assert_allclose(flat, [0.0, 0.0], atol=1e-12)

    def test_a_group_with_no_target_spread_is_undefined_rather_than_zero(self):
        target = np.asarray([2.0, 2.0, 5.0, 7.0])
        groups = np.asarray(['a', 'a', 'b', 'b'])
        pairs = np.asarray(['a:1', 'a:1', 'b:1', 'b:1'])
        _, values = group_r2(target, target.copy(), groups, pairs)
        self.assertIsNone(values[0])
        self.assertEqual(values[1], 1.0)

    def test_the_evaluation_reports_every_declared_contrast_and_stratum(self):
        panel = gate_panel(groups=10, pairs=2, cycles=4, response=0.4, observable_level=True)
        outcome = gate_compare(panel, seed=20260923)
        evaluation = evaluate_gate(panel, outcome, draws=50, seed=20260923)
        self.assertEqual(set(evaluation['increments']),
                         {f'{a}|{b}' for b, a in GATE_CONTRASTS})
        for record in evaluation['increments'].values():
            self.assertEqual(set(record['strata']), {'10+'})
            self.assertIn('mse_reduction_kcal2', record)
        self.assertEqual(len(evaluation['reconstruction']), 2 * len(RECONSTRUCTION_CONTROLS))


class TheWeightingUnitIsTheFrozenOne(unittest.TestCase):
    @unittest.skipUnless(PLAN.exists(), 'the frozen extraction plan is retained locally')
    def test_the_frozen_supports_carry_their_declared_effective_site_pairs(self):
        plan = json.loads(PLAN.read_text())
        fit = load_gate_fit_script()
        for support, pairs, fitting, declared in (('all_groups', 217, 129.2, 121.6),
                                                  ('q_groups', 163, 96.1, 88.4)):
            groups = set(plan['supports'][support]['groups'])
            labels, names = [], []
            for background in plan['backgrounds']:
                if background['group'] not in groups:
                    continue
                for cycle in background['cycles']:
                    labels.append(background['group'])
                    names.append(f"{background['name']}:"
                                 f"{cycle['positions'][0]}-{cycle['positions'][1]}")
            panel = {'site_pair': np.asarray(names), 'group': np.asarray(labels)}
            counts = fit.kish_effective_site_pairs(panel)
            self.assertEqual(len(set(names)), pairs)
            # The weighting these fits use, and the cycle-share convention behind
            # the counts the readiness record states, are both pinned: they differ
            # by about 6% and only the first describes a fit reported here.
            self.assertAlmostEqual(counts['fitting_weights'], fitting, places=1)
            self.assertAlmostEqual(counts['cycle_share'], declared, places=1)


def exclusion_fixture():
    """A two-background panel whose state rows are aligned to a plan, the shape
    :func:`apply_indel_exclusion` receives from the admitted fit script."""

    plan = {'backgrounds': []}
    states_y, state_group, state_background, cycle_states = [], [], [], []
    rows = {'group': [], 'site_pair': [], 'background': [], 'separation': [], 'epsilon': []}
    for index, level in enumerate((0.0, 1.0)):
        name, group = f'b{index}', f'g{index}'
        sequences = ['WT', 'A_', '_B', 'AB']
        base = len(states_y)
        values = [level, level - 1.0, level - 0.5, level - 1.7]
        states_y.extend(values)
        state_group.extend([group] * 4)
        state_background.extend([name] * 4)
        plan['backgrounds'].append({'name': name, 'group': group, 'sequences': sequences,
                                    'cycles': [{'states': [0, 1, 2, 3], 'positions': [3, 20],
                                                'separation': '10+'}]})
        cycle_states.append([base, base + 1, base + 2, base + 3])
        rows['group'].append(group)
        rows['site_pair'].append(f'{name}:3-20')
        rows['background'].append(name)
        rows['separation'].append('10+')
        rows['epsilon'].append(values[3] - values[1] - values[2] + values[0])
    panel = {key: np.asarray(value) for key, value in rows.items() if key != 'epsilon'}
    panel['epsilon'] = np.asarray(rows['epsilon'], dtype=float)
    # Every block the gate reads, plus one the admitted fit script assembles and
    # the gate does not, so the filter is exercised on both.
    panel['blocks'] = {name: np.arange(2 * 3, dtype=float).reshape(2, 3)
                       for name in GATE_BLOCKS}
    panel['blocks']['R'] = np.zeros((2, 4))
    panel['cycle_states'] = np.asarray(cycle_states)
    panel['independent_site_max_absolute'] = 0.0
    panel['states'] = {'features': np.arange(8 * 2, dtype=float).reshape(8, 2),
                       'y': np.asarray(states_y, dtype=float),
                       'group': np.asarray(state_group),
                       'background': np.asarray(state_background)}
    panel['states']['weight'] = row_weights(panel['states']['background'],
                                            panel['states']['group'])
    return plan, panel


class TheIndelExclusionIsADeclaredFilterOverTheFrozenSupport(unittest.TestCase):
    """The frozen cohort admitted insertion constructs whose sequence is truncated
    to the wild-type length, so they entered as though they were substitutions.
    The cohort is not re-frozen; the exclusion is applied to the panel and
    reported beside the unfiltered fit."""

    def declaration(self, states):
        return {'schema': 'pairwise_indel_exclusion_v1', 'states': states}

    def test_a_state_with_no_remaining_row_drops_every_cycle_that_uses_it(self):
        plan, panel = exclusion_fixture()
        declaration = self.declaration([
            {'background': 'b0', 'sequence': 'AB', 'value_kcal_mol': -1.7,
             'value_without_indel_rows_kcal_mol': None}])
        filtered, accounting = apply_indel_exclusion(panel, plan, declaration, {'g0', 'g1'})
        self.assertEqual(accounting['cycles_dropped'], 1)
        self.assertEqual(accounting['states_removed'], 1)
        self.assertEqual(list(filtered['background']), ['b1'])
        self.assertEqual(accounting['site_pairs_dropped'], ['b0:3-20'])
        self.assertEqual(accounting['groups_dropped'], ['g0'])
        self.assertEqual(accounting['blocks_not_carried'], ['R'])
        self.assertEqual(set(filtered['blocks']), set(GATE_BLOCKS))
        # The surviving cycle still indexes its own four states after the remap.
        wild, low, high, double = filtered['cycle_states'][0]
        self.assertEqual(list(filtered['states']['background'][[wild, low, high, double]]),
                         ['b1'] * 4)
        self.assertAlmostEqual(
            float(filtered['epsilon'][0]),
            float(filtered['states']['y'][double] - filtered['states']['y'][low]
                  - filtered['states']['y'][high] + filtered['states']['y'][wild]))

    def test_a_corrected_state_moves_exactly_the_cycles_that_use_it(self):
        plan, panel = exclusion_fixture()
        declaration = self.declaration([
            {'background': 'b1', 'sequence': 'WT', 'value_kcal_mol': 1.0,
             'value_without_indel_rows_kcal_mol': 1.25}])
        filtered, accounting = apply_indel_exclusion(panel, plan, declaration, {'g0', 'g1'})
        self.assertEqual(accounting['cycles_dropped'], 0)
        self.assertEqual(accounting['cycles_with_a_moved_target'], 1)
        self.assertAlmostEqual(accounting['largest_target_move_kcal_mol'], 0.25)
        np.testing.assert_allclose(filtered['epsilon'] - panel['epsilon'], [0.0, 0.25])
        self.assertEqual(accounting['by_stratum']['10+']['cycles_with_a_moved_target'], 1)

    def test_a_panel_whose_cycles_disagree_with_its_states_is_refused(self):
        plan, panel = exclusion_fixture()
        panel['epsilon'] = panel['epsilon'] + 0.5
        with self.assertRaises(ValueError):
            apply_indel_exclusion(panel, plan, self.declaration([]), {'g0', 'g1'})

    def test_an_unexpected_declaration_schema_is_refused(self):
        plan, panel = exclusion_fixture()
        with self.assertRaises(ValueError):
            apply_indel_exclusion(panel, plan, {'schema': 'something_else', 'states': []},
                                  {'g0', 'g1'})

    def test_a_cycle_level_record_is_refused_by_the_state_level_consumer(self):
        plan, panel = exclusion_fixture()
        with self.assertRaisesRegex(ValueError, "state-level shape.*excluded_cycles"):
            apply_indel_exclusion(panel, plan, {'schema': 'pairwise_indel_exclusion_v1', 'excluded_cycles': {}, 'summary': {}}, {'g0', 'g1'})

    @unittest.skipUnless(EXCLUSION.exists(), 'the declared exclusion is retained locally')
    def test_the_declared_exclusion_binds_to_the_frozen_cohort(self):
        declaration = json.loads(EXCLUSION.read_text())
        plan = json.loads(PLAN.read_text())
        self.assertEqual(declaration['cohort_sha256'], plan['cohort_sha256'])
        self.assertEqual(declaration['accounting']['states_with_an_indel_row'], 5)
        self.assertEqual(declaration['accounting']['states_absent_after_exclusion'], 3)
        self.assertEqual(declaration['accounting']['states_whose_value_moves'], 2)
        self.assertEqual(declaration['accounting']['backgrounds_touched'],
                         ['1O6X.pdb', '2AMI.pdb', '2D1U.pdb'])
        self.assertLess(declaration['accounting']['largest_state_value_move_kcal_mol'], 0.05)


class TheFeatureBuildersRefuseWhatTheyCannotRepresent(unittest.TestCase):
    def test_a_site_outside_the_background_is_refused(self):
        with self.assertRaises(ValueError):
            site_context_features(WILD, len(WILD), 'A')

    def test_a_site_carrying_no_substitution_is_refused(self):
        with self.assertRaises(ValueError):
            site_context_features(WILD, 9, WILD[9])

    def test_a_noncanonical_residue_is_refused(self):
        with self.assertRaises(ValueError):
            site_context_features(WILD, 9, 'X')
        with self.assertRaises(ValueError):
            site_context_features(substitute(WILD, 30, 'B'), 9, 'A')

    def test_a_cycle_whose_wild_type_is_not_the_background_is_refused(self):
        states = quartet(WILD, 9, 40, 'A', 'W')
        with self.assertRaises(ValueError):
            cycle_context_features(substitute(WILD, 3, 'G'), states, (10, 41))

    def test_unordered_positions_are_refused(self):
        states = quartet(WILD, 9, 40, 'A', 'W')
        with self.assertRaises(ValueError):
            cycle_context_features(WILD, states, (41, 10))

    def test_a_double_state_missing_a_substitution_is_refused(self):
        wild, first, second, _ = quartet(WILD, 9, 40, 'A', 'W')
        with self.assertRaises(ValueError):
            cycle_context_features(WILD, (wild, first, second, first), (10, 41))


if __name__ == '__main__':
    unittest.main()
