"""Outer held labels cannot reach residual sensitivity predictions or penalties."""
import copy
import unittest
import numpy as np
from scripts.capability.interactions.refit_likelihood_on_residual import nested_residual_compare, residual_targets
from src.capability.interactions.pairwise_epistasis import (
    SPLIT_SEEDS, family_folds, indel_excluded_support)
from tests.interactions.test_pairwise_epistasis import synthetic_panel


class ResidualRefit(unittest.TestCase):
    def test_outer_labels_cannot_change_predictions_or_penalties(self):
        panel = synthetic_panel(groups=12, pairs=1, cycles=2, response=0.3)
        states = panel['states']['y'][panel['cycle_states']]
        additive = states[:, 1] + states[:, 2] - states[:, 0]
        seed = SPLIT_SEEDS[0]
        held = family_folds(panel['group'], 5, seed)[0]
        rows = np.isin(panel['group'], held)
        original = nested_residual_compare(panel, additive, seed=seed)
        changed = copy.deepcopy(panel)
        changed['epsilon'][rows] += np.linspace(3, 15, rows.sum())
        state_rows = np.isin(changed['states']['group'], held)
        changed['states']['y'][state_rows] += 17
        perturbed = nested_residual_compare(changed, additive, seed=seed)
        self.assertEqual(original[2][0], perturbed[2][0])
        for name in original[1]:
            np.testing.assert_array_equal(original[1][name][rows], perturbed[1][name][rows])
        self.assertFalse(np.array_equal(original[0][rows], perturbed[0][rows]))

    def test_query_labels_do_not_change_training_residual(self):
        x = np.arange(8, dtype=float)
        y = x ** 2
        train, query = np.arange(6), np.arange(6, 8)
        before = residual_targets(y, x, train, query)
        y[query] += 100
        after = residual_targets(y, x, train, query)
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_allclose(after[1] - before[1], 100)


def restricted_inputs(panel):
    """The plan and an exclusion declaration for a synthetic panel's states."""
    states = panel['states']
    plan = {'backgrounds': []}
    for background in dict.fromkeys(states['background'].tolist()):
        rows = np.flatnonzero(states['background'] == background)
        plan['backgrounds'].append({
            'name': background, 'group': states['group'][rows[0]],
            'sequences': [f's{index}' for index in rows]})
    keys = [(background['name'], sequence) for background in plan['backgrounds']
            for sequence in background['sequences']]
    return plan, keys


class IndelExcludedSupport(unittest.TestCase):
    """The declared exclusion leaves the support before any fit reads it."""

    def panel(self):
        panel = synthetic_panel(groups=4, pairs=1, cycles=3)
        panel['cycle_index'] = np.arange(len(panel['epsilon']), dtype=np.int64)
        panel['independent_site_max_absolute'] = 0.0
        return panel

    def test_an_emptied_state_takes_its_cycles_with_it_and_leaves_the_rest_aligned(self):
        panel = self.panel()
        plan, keys = restricted_inputs(panel)
        emptied = panel['cycle_states'][0, 3]
        declaration = {'schema': 'pairwise_indel_exclusion_v1', 'states': [
            {'background': keys[emptied][0], 'sequence': keys[emptied][1],
             'value_without_indel_rows_kcal_mol': None}]}
        groups = {background['group'] for background in plan['backgrounds']}
        filtered, accounting = indel_excluded_support(panel, plan, declaration, groups)
        keep = ~(panel['cycle_states'] == emptied).any(axis=1)
        self.assertEqual(accounting['cycles_dropped'], int((~keep).sum()))
        self.assertEqual(accounting['states_removed'], 1)
        self.assertEqual(len(filtered['epsilon']), int(keep.sum()))
        self.assertEqual(len(filtered['states']['y']), len(panel['states']['y']) - 1)
        # Re-indexing must survive the removal: every retained cycle still reads
        # its own four states.
        y = filtered['states']['y'][filtered['cycle_states']]
        np.testing.assert_allclose(y[:, 3] - y[:, 1] - y[:, 2] + y[:, 0], filtered['epsilon'],
                                   atol=1e-12)
        np.testing.assert_array_equal(filtered['cycle_index'], panel['cycle_index'][keep])
        for name, block in filtered['blocks'].items():
            np.testing.assert_array_equal(block, panel['blocks'][name][keep])

    def test_a_corrected_median_moves_exactly_the_targets_that_index_it(self):
        panel = self.panel()
        plan, keys = restricted_inputs(panel)
        moved = panel['cycle_states'][0, 0]
        shift = 0.25
        declaration = {'schema': 'pairwise_indel_exclusion_v1', 'states': [
            {'background': keys[moved][0], 'sequence': keys[moved][1],
             'value_without_indel_rows_kcal_mol': panel['states']['y'][moved] + shift}]}
        groups = {background['group'] for background in plan['backgrounds']}
        filtered, accounting = indel_excluded_support(panel, plan, declaration, groups)
        touched = (panel['cycle_states'] == moved).any(axis=1)
        self.assertEqual(accounting['cycles_dropped'], 0)
        self.assertEqual(accounting['states_corrected'], 1)
        self.assertEqual(accounting['cycles_with_a_moved_target'], int(touched.sum()))
        self.assertAlmostEqual(accounting['largest_target_move_kcal_mol'], shift, places=12)
        np.testing.assert_allclose(filtered['epsilon'][~touched], panel['epsilon'][~touched],
                                   atol=0)
        np.testing.assert_allclose(filtered['epsilon'][touched] - panel['epsilon'][touched],
                                   shift, atol=1e-12)

    def test_a_panel_whose_targets_disagree_with_its_states_is_refused(self):
        panel = self.panel()
        plan, _ = restricted_inputs(panel)
        groups = {background['group'] for background in plan['backgrounds']}
        empty = {'schema': 'pairwise_indel_exclusion_v1', 'states': []}
        panel['epsilon'][2] += 1e-6
        with self.assertRaisesRegex(ValueError, 'disagree with their own states'):
            indel_excluded_support(panel, plan, empty, groups)

    def test_a_cycle_level_record_is_refused_by_the_state_level_consumer(self):
        panel = self.panel()
        plan, _ = restricted_inputs(panel)
        groups = {background['group'] for background in plan['backgrounds']}
        with self.assertRaisesRegex(ValueError, "state-level shape.*excluded_cycles"):
            indel_excluded_support(panel, plan, {'schema': 'pairwise_indel_exclusion_v1', 'excluded_cycles': {}, 'summary': {}}, groups)

    def test_a_declaration_of_another_kind_is_refused(self):
        panel = self.panel()
        plan, _ = restricted_inputs(panel)
        groups = {background['group'] for background in plan['backgrounds']}
        with self.assertRaisesRegex(ValueError, 'declaration schema'):
            indel_excluded_support(panel, plan, {'schema': 'something_else', 'states': []},
                                   groups)
