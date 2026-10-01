"""Conditions that must hold for the single-mutant stability gate.

These are the gate's load-bearing invariants, not a tour of its happy path: the
endpoint's construction and sign convention against a known stability change, the
control-qualification rule refusing a control that does not transfer, identical
support and folds across every compared fit, no held-out label reaching any
fitted quantity, censored and indel rows never becoming point measurements, and
the extraction plan refusing a state it cannot re-derive.
"""
from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.readouts.readout_analysis import family_folds
from src.capability.stability.stability_gate import (
    CHEMISTRY_SCALES, PROFILE_BOUNDED_FEATURE_ORDER, PROFILE_LOG_CLIP, secondary_control_set,
    QUALIFICATION_SEEDS, SOURCE_COLUMNS, accept_rows, aggregate_states, build_endpoint,
    chemistry_block, chemistry_width, ddg, draw_order, endpoint_digest, fold_predictions,
    geometry_block, identity_block, paired_increment, profile_bounded_block, qualify,
    single_substitution, substitution_rows, validate_plan_background)

WILD = 'MKVLIAAGDEFGHIKLMNPQ'


def source_row(name: str, sequence: str, mut_type: str, combined: float,
               trypsin: float, chymotrypsin: float, *, width: float = 0.2,
               dg_ml=None) -> dict:
    row = {column: 0.0 for column in SOURCE_COLUMNS}
    row.update({
        'name': f'{name}|{mut_type}', 'dna_seq': 'ACG', 'mut_type': mut_type,
        'WT_name': name, 'WT_cluster': 'c1', 'aa_seq': sequence,
        'dG_ML': f'{combined}' if dg_ml is None else dg_ml,
        'ddG_ML': 0.0, 'deltaG': combined,
        'deltaG_95CI': width, 'deltaG_95CI_low': combined - width / 2,
        'deltaG_95CI_high': combined + width / 2,
        'deltaG_t': trypsin, 'deltaG_t_95CI': width,
        'deltaG_t_95CI_low': trypsin - width / 2, 'deltaG_t_95CI_high': trypsin + width / 2,
        'deltaG_c': chymotrypsin, 'deltaG_c_95CI': width,
        'deltaG_c_95CI_low': chymotrypsin - width / 2,
        'deltaG_c_95CI_high': chymotrypsin + width / 2,
    })
    return row


def mutate(sequence: str, position: int, residue: str) -> str:
    return sequence[:position] + residue + sequence[position + 1:]


class TheEndpointIsTheDeclaredDifference(unittest.TestCase):
    def test_a_known_destabilising_change_has_the_declared_sign_and_magnitude(self):
        destabilised = mutate(WILD, 4, 'D')
        stabilised = mutate(WILD, 9, 'W')
        frame = pd.DataFrame([
            source_row('BG.pdb', WILD, 'wt', 3.50, 3.40, 3.60),
            source_row('BG.pdb', destabilised, 'I5D', 1.25, 1.20, 1.30),
            source_row('BG.pdb', stabilised, 'L10W', 4.10, 4.05, 4.15),
        ])
        mask, _ = accept_rows(frame)
        endpoint, _ = build_endpoint(
            aggregate_states(frame, mask),
            {'BG.pdb': {'kind': 'natural', 'sequence': WILD, 'cluster': 'c1', 'group': 'g1'}})
        values = dict(zip(endpoint.sequence, endpoint.combined))
        self.assertAlmostEqual(values[destabilised], 1.25 - 3.50, places=9)
        self.assertAlmostEqual(values[stabilised], 4.10 - 3.50, places=9)
        self.assertLess(values[destabilised], 0.0,
                        'dG_ML is an unfolding free energy, so a destabilising change is negative')
        self.assertGreater(values[stabilised], 0.0)
        self.assertAlmostEqual(ddg(1.25, 3.50), -2.25, places=9)
        # Both channels are differenced against the same wild-type state.
        row = endpoint[endpoint.sequence == destabilised].iloc[0]
        self.assertAlmostEqual(row.trypsin, 1.20 - 3.40, places=9)
        self.assertAlmostEqual(row.chymotrypsin, 1.30 - 3.60, places=9)

    def test_the_digest_depends_on_the_values_and_not_on_their_order(self):
        records = [{'background': 'a', 'sequence': 'AA', 'position': 1, 'ddg': -1.0},
                   {'background': 'a', 'sequence': 'AB', 'position': 2, 'ddg': 0.5}]
        self.assertEqual(endpoint_digest(records), endpoint_digest(list(reversed(records))))
        moved = [dict(records[0], ddg=-1.5), records[1]]
        self.assertNotEqual(endpoint_digest(records), endpoint_digest(moved))


class CensoredAndIndelRowsNeverBecomeMeasurements(unittest.TestCase):
    def test_censored_strings_are_counted_and_dropped(self):
        frame = pd.DataFrame([
            source_row('BG.pdb', WILD, 'wt', 3.50, 3.40, 3.60),
            source_row('BG.pdb', mutate(WILD, 4, 'D'), 'I5D', 0.0, 0.0, 0.0, dg_ml='<-1'),
            source_row('BG.pdb', mutate(WILD, 5, 'W'), 'A6W', 0.0, 0.0, 0.0, dg_ml='>5'),
            source_row('BG.pdb', mutate(WILD, 6, 'K'), 'A7K', 0.0, 0.0, 0.0, dg_ml='-'),
            source_row('BG.pdb', mutate(WILD, 7, 'E'), 'G8E', 2.00, 1.95, 2.05),
        ])
        mask, accounting = accept_rows(frame)
        self.assertEqual(accounting['rows_censored_dG_ML'], {'<-1': 1, '>5': 1, '-': 1})
        self.assertEqual(accounting['accepted_rows'], 2)
        endpoint, _ = build_endpoint(
            aggregate_states(frame, mask),
            {'BG.pdb': {'kind': 'natural', 'sequence': WILD, 'cluster': 'c1', 'group': 'g1'}})
        self.assertEqual(len(endpoint), 1)
        for boundary in (-1.0, 5.0, -1.0 - 3.50, 5.0 - 3.50):
            self.assertNotIn(boundary, list(endpoint.combined),
                             'a censored bound reached the endpoint as a point measurement')

    def test_an_indel_construct_cannot_enter_the_substitution_support(self):
        # The pinned files truncate an indel construct's aa_seq to the wild-type
        # length, so it is length-matched and looks like a substitution variant.
        truncated = mutate(WILD, 3, 'S')
        frame = pd.DataFrame([
            source_row('BG.pdb', WILD, 'wt', 3.50, 3.40, 3.60),
            source_row('BG.pdb', truncated, 'ins4S', 0.10, 0.05, 0.15),
            source_row('BG.pdb', mutate(WILD, 7, 'E'), 'del8', 0.20, 0.15, 0.25),
            source_row('BG.pdb', mutate(WILD, 9, 'W'), 'L10W', 4.10, 4.05, 4.15),
        ])
        keep = substitution_rows(frame.mut_type.to_numpy())
        self.assertEqual(keep.tolist(), [True, False, False, True])
        mask, accounting = accept_rows(frame)
        self.assertEqual(accounting['rows_indel_construct'], 2)
        endpoint, _ = build_endpoint(
            aggregate_states(frame, mask),
            {'BG.pdb': {'kind': 'natural', 'sequence': WILD, 'cluster': 'c1', 'group': 'g1'}})
        self.assertEqual(list(endpoint.sequence), [mutate(WILD, 9, 'W')])

    def test_a_wild_type_state_carried_only_by_an_indel_row_is_not_a_wild_type(self):
        # A contaminated wild type would propagate into every difference taken
        # against it, so the background must drop out entirely rather than be
        # differenced against an indel construct's truncated sequence.
        frame = pd.DataFrame([
            source_row('BG.pdb', WILD, 'ins1A', 9.00, 9.00, 9.00),
            source_row('BG.pdb', mutate(WILD, 9, 'W'), 'L10W', 4.10, 4.05, 4.15),
            source_row('OK.pdb', WILD, 'wt', 3.50, 3.40, 3.60),
            source_row('OK.pdb', mutate(WILD, 4, 'D'), 'I5D', 1.25, 1.20, 1.30),
        ])
        mask, _ = accept_rows(frame)
        catalogue = {name: {'kind': 'natural', 'sequence': WILD, 'cluster': 'c1',
                            'group': f'g{index}'}
                     for index, name in enumerate(('BG.pdb', 'OK.pdb'))}
        endpoint, support = build_endpoint(aggregate_states(frame, mask), catalogue)
        self.assertEqual(support['backgrounds_without_accepted_wt'], 1)
        self.assertEqual(set(endpoint.WT_name), {'OK.pdb'})
        self.assertAlmostEqual(endpoint.combined.iloc[0], 1.25 - 3.50, places=9)

    def test_a_wide_channel_interval_is_refused_on_either_channel(self):
        for column in ('deltaG_95CI', 'deltaG_t_95CI', 'deltaG_c_95CI'):
            frame = pd.DataFrame([source_row('BG.pdb', WILD, 'wt', 3.50, 3.40, 3.60)])
            frame.loc[0, column] = 0.9
            frame.loc[0, f'{column}_low'] = 3.0
            frame.loc[0, f'{column}_high'] = 3.9
            mask, accounting = accept_rows(frame)
            self.assertEqual(accounting['accepted_rows'], 0, f'{column} was not enforced')


class TheDrawIsLabelIndependent(unittest.TestCase):
    def test_the_draw_order_ignores_every_measurement(self):
        sequences = [mutate(WILD, i, 'W') for i in range(12)]
        first = draw_order('BG.pdb', sequences)
        self.assertEqual(first, draw_order('BG.pdb', list(reversed(sequences))))
        self.assertNotEqual(first, draw_order('OTHER.pdb', sequences))
        self.assertNotEqual(first, draw_order('BG.pdb', sequences, seed=1))
        self.assertEqual(sorted(first), sorted(sequences))


class TheLocalChemistryBlockReadsTheDeclaredScales(unittest.TestCase):
    def test_a_hydrophobic_to_charged_change_lowers_the_window_hydropathy(self):
        block = chemistry_block(WILD, 4, 'D')
        self.assertEqual(len(block), chemistry_width())
        # Layout: per scale, [site wild, site mutant, difference, then per radius
        # (window wild, window mutant, difference)]. Hydropathy is first.
        self.assertEqual(CHEMISTRY_SCALES[0], 'hydropathy')
        self.assertLess(block[2], 0.0, 'isoleucine to aspartate must lower hydropathy')
        for offset in (5, 8):
            self.assertLess(block[offset], 0.0, 'the window mean must fall as well')
        self.assertTrue(np.isfinite(block).all())

    def test_a_terminal_position_window_stays_inside_the_sequence(self):
        for position in (0, len(WILD) - 1):
            block = chemistry_block(WILD, position, 'W')
            self.assertTrue(np.isfinite(block).all())


def synthetic_panel(*, groups: int = 12, sites: int = 4, variants: int = 6,
                    harmful: bool = True, seed: int = 7) -> dict:
    """A panel whose candidate block does not transfer across held-out groups.

    ``signal`` predicts the target the same way everywhere. ``candidate`` is that
    same signal multiplied by a per-group sign, so any coefficient fitted on the
    training groups arrives at a held-out group with the wrong sign for half of
    them. That is the failure mode the qualification rule exists to catch, and it
    is deterministic rather than a hope about ridge shrinkage.
    """

    rng = np.random.default_rng(seed)
    group, site, signal, candidate, target = [], [], [], [], []
    for index in range(groups):
        sign = 1.0 if index % 2 == 0 else -1.0
        for s in range(sites):
            for _ in range(variants):
                value = float(rng.normal())
                group.append(f'g{index:02d}')
                site.append(f'g{index:02d}:{s}')
                signal.append(value)
                candidate.append((sign if harmful else 1.0) * value)
                target.append(2.0 * value + 0.05 * float(rng.normal()))
    group = np.asarray(group)
    site = np.asarray(site)
    signal = np.asarray(signal)[:, None]
    from src.capability.readouts.readout_analysis import row_weights
    states = {'features': np.concatenate([np.zeros((len(group), 1)), signal], axis=1),
              'y': np.asarray(target), 'group': group,
              'weight': row_weights(site, group)}
    return {'target': np.asarray(target), 'group': group, 'site': site,
            'pair_states': np.column_stack([np.zeros(len(group), int),
                                            np.arange(len(group))]),
            'states': states,
            'blocks': {'signal': signal,
                       'candidate': np.asarray(candidate)[:, None]}}


class TheQualificationRuleRefusesAControlThatDoesNotTransfer(unittest.TestCase):
    def test_a_negative_increment_at_any_seed_is_refused(self):
        increments = {seed: 0.01 for seed in QUALIFICATION_SEEDS}
        self.assertTrue(qualify(increments)['qualified'])
        increments[QUALIFICATION_SEEDS[1]] = -1e-9
        self.assertFalse(qualify(increments)['qualified'])
        increments[QUALIFICATION_SEEDS[1]] = 0.0
        self.assertFalse(qualify(increments)['qualified'],
                         'an exactly zero increment is not a competent predictor')
        with self.assertRaises(ValueError):
            qualify({QUALIFICATION_SEEDS[0]: 0.01})

    def test_a_block_that_does_not_transfer_is_measured_as_harmful_and_refused(self):
        panel = synthetic_panel(harmful=True)
        design = {'standing': ('signal',), 'candidate': ('signal', 'candidate')}
        increments = {}
        for seed in QUALIFICATION_SEEDS:
            outcome = fold_predictions(panel, panel['blocks'], design, seed=seed)
            increments[seed] = paired_increment(
                panel, outcome['predictions'], 'candidate', 'standing')['point']
        verdict = qualify(increments)
        self.assertLess(verdict['min_increment_kcal2_mol2'], 0.0)
        self.assertFalse(verdict['qualified'])

    def test_a_block_that_does_transfer_is_measured_as_helpful_and_kept(self):
        panel = synthetic_panel(harmful=False)
        panel['blocks']['candidate'] = panel['blocks']['candidate'] * 0 + 1.0
        rng = np.random.default_rng(11)
        extra = panel['target'] * 0.5 + 0.01 * rng.normal(size=len(panel['target']))
        panel['blocks']['candidate'] = extra[:, None]
        design = {'standing': ('signal',), 'candidate': ('signal', 'candidate')}
        increments = {}
        for seed in QUALIFICATION_SEEDS:
            outcome = fold_predictions(panel, panel['blocks'], design, seed=seed)
            increments[seed] = paired_increment(
                panel, outcome['predictions'], 'candidate', 'standing')['point']
        self.assertTrue(qualify(increments)['qualified'])


class TheSecondaryControlSetFollowsTheCorrelationRule(unittest.TestCase):
    def ladder(self, mse, spearman, qualified):
        return {'base_blocks': ['ident', 'geom'],
                'candidate_order': ['comp', 'chem', 'prof', 'prof2', 'G'],
                'qualified_control_set': ['ident', 'geom', *qualified],
                'ladder': [{'candidate': c,
                            'qualified': c in qualified,
                            'per_seed_increment_kcal2_mol2': {
                                str(s): mse[c][i] for i, s in enumerate(QUALIFICATION_SEEDS)},
                            'spearman_increment': {
                                str(s): {'point': spearman[c][i]}
                                for i, s in enumerate(QUALIFICATION_SEEDS)}}
                           for c in ('comp', 'chem', 'prof', 'prof2', 'G')]}

    def test_a_discarded_block_that_raises_correlation_at_every_seed_is_added_once(self):
        mse = {'comp': [1e-4, -2e-3, 1e-3], 'chem': [4e-3, 7e-3, 5e-3],
               'prof': [0.02, -1.6, 2e-3], 'prof2': [0.02, -1.3, 4e-3], 'G': [0.02, 0.03, 0.02]}
        spearman = {'comp': [8e-4, 1e-3, 4e-4], 'chem': [1e-3, 2e-3, 1e-3],
                    'prof': [0.040, 0.048, 0.031], 'prof2': [0.049, 0.058, 0.040],
                    'G': [0.034, 0.033, 0.034]}
        secondary, record = secondary_control_set(self.ladder(mse, spearman, ('chem', 'G')))
        self.assertEqual(secondary, ('ident', 'geom', 'comp', 'chem', 'prof2', 'G'))
        self.assertEqual(record['added_over_primary'], ['comp', 'prof2'])
        self.assertEqual(record['evidence']['prof2']['offered_as'], 'prof2',
                         'a restatement must carry its own evidence, not the block it supersedes')
        self.assertEqual(len(secondary), len(set(secondary)),
                         'a superseded block and its restatement must not both appear')

    def test_a_discarded_block_that_lowers_correlation_at_any_seed_is_not_added(self):
        mse = {'comp': [1e-4, -2e-3, 1e-3], 'chem': [4e-3, 7e-3, 5e-3],
               'prof': [0.02, -1.6, 2e-3], 'prof2': [0.02, -1.3, 4e-3], 'G': [0.02, 0.03, 0.02]}
        spearman = {'comp': [8e-4, -1e-9, 4e-4], 'chem': [1e-3, 2e-3, 1e-3],
                    'prof': [0.040, 0.048, 0.031], 'prof2': [0.049, -0.01, 0.040],
                    'G': [0.034, 0.033, 0.034]}
        secondary, record = secondary_control_set(self.ladder(mse, spearman, ('chem', 'G')))
        self.assertEqual(secondary, ('ident', 'geom', 'chem', 'G'))
        self.assertEqual(record['added_over_primary'], [])


class EveryComparedFitSeesOneSupportAndOneSplit(unittest.TestCase):
    def test_adding_a_design_does_not_move_another_designs_predictions(self):
        panel = synthetic_panel()
        alone = fold_predictions(panel, panel['blocks'], {'standing': ('signal',)}, seed=20260923)
        together = fold_predictions(
            panel, panel['blocks'],
            {'standing': ('signal',), 'candidate': ('signal', 'candidate')}, seed=20260923)
        np.testing.assert_array_equal(alone['predictions']['standing'],
                                      together['predictions']['standing'])
        self.assertEqual([f['held_groups'] for f in alone['folds']],
                         [f['held_groups'] for f in together['folds']])

    def test_every_design_in_one_call_shares_one_fold_map_and_one_row_set(self):
        panel = synthetic_panel()
        outcome = fold_predictions(
            panel, panel['blocks'],
            {'standing': ('signal',), 'candidate': ('signal', 'candidate')}, seed=20260924)
        held = [set(f['held_groups']) for f in outcome['folds']]
        self.assertEqual(sorted(set().union(*held)), sorted(set(panel['group'])))
        self.assertEqual(sum(len(f) for f in held), len(set(panel['group'])))
        for name in ('standing', 'candidate'):
            self.assertTrue(np.isfinite(outcome['predictions'][name]).all())
        self.assertEqual(len(outcome['predictions']['standing']), len(panel['target']))

    def test_the_fold_map_is_a_function_of_the_seed_alone(self):
        groups = [f'g{i:02d}' for i in range(12)]
        self.assertEqual(family_folds(groups, 5, 20260923), family_folds(groups, 5, 20260923))
        self.assertNotEqual(family_folds(groups, 5, 20260923), family_folds(groups, 5, 20260924))
        self.assertEqual(family_folds(groups, 5, 20260923),
                         family_folds(list(reversed(groups)), 5, 20260923))


class NoHeldOutLabelReachesAFittedQuantity(unittest.TestCase):
    def test_permuting_held_out_labels_moves_no_held_out_prediction(self):
        panel = synthetic_panel()
        design = {'standing': ('signal',), 'nuisance': ('signal', 'G')}
        seed = 20260923
        base = fold_predictions(panel, panel['blocks'], design, seed=seed)
        held = base['folds'][0]['held_groups']
        rows = np.flatnonzero(np.isin(panel['group'], held))
        rng = np.random.default_rng(3)
        target = panel['target'].copy()
        target[rows] = target[rng.permutation(rows)]
        states = dict(panel['states'])
        y = panel['states']['y'].copy()
        y[rows] = y[rng.permutation(rows)]
        states['y'] = y
        permuted = dict(panel, target=target, states=states)
        self.assertFalse(np.array_equal(target, panel['target']),
                         'the permutation must actually change the held-out labels')
        moved = fold_predictions(permuted, panel['blocks'], design, seed=seed)
        for name in design:
            np.testing.assert_array_equal(moved['predictions'][name][rows],
                                          base['predictions'][name][rows])

    def test_the_purge_changes_training_only_and_leaves_the_held_out_rows_intact(self):
        panel = synthetic_panel()
        groups = sorted(set(panel['group']))
        purge = {group: [other] for group, other in zip(groups, groups[1:] + groups[:1])}
        design = {'standing': ('signal',)}
        unpurged = fold_predictions(panel, panel['blocks'], design, seed=20260923)
        purged = fold_predictions(panel, panel['blocks'], design, seed=20260923, purge=purge)
        self.assertEqual([f['held_groups'] for f in unpurged['folds']],
                         [f['held_groups'] for f in purged['folds']])
        self.assertTrue(any(f['purged_training_groups'] for f in purged['folds']))
        for unpurged_fold, purged_fold in zip(unpurged['folds'], purged['folds']):
            self.assertLessEqual(len(purged_fold['training_groups']),
                                 len(unpurged_fold['training_groups']))


class TheExtractionPlanRefusesAStateItCannotRederive(unittest.TestCase):
    def setUp(self):
        self.row = {
            'name': 'BG.pdb', 'wildtype': WILD,
            'sequences': [WILD, mutate(WILD, 4, 'D'), mutate(WILD, 9, 'W')],
            'variants': [{'position': 5, 'mutant': 'D', 'state': 1},
                         {'position': 10, 'mutant': 'W', 'state': 2}],
        }

    def test_a_consistent_background_passes(self):
        validate_plan_background(self.row)

    def test_a_wrong_position_a_wrong_residue_and_a_misplaced_wild_type_are_refused(self):
        for broken in (
            {**self.row, 'variants': [{'position': 6, 'mutant': 'D', 'state': 1}]},
            {**self.row, 'variants': [{'position': 5, 'mutant': 'E', 'state': 1}]},
            {**self.row, 'variants': [{'position': 5, 'mutant': 'D', 'state': 2}]},
            {**self.row, 'sequences': [mutate(WILD, 4, 'D'), WILD, mutate(WILD, 9, 'W')]},
            {**self.row, 'sequences': [WILD, WILD, mutate(WILD, 9, 'W')]},
        ):
            with self.assertRaises(ValueError):
                validate_plan_background(broken)


class TheBoundedProfileBlockStaysInItsDeclaredRange(unittest.TestCase):
    class _Profile:
        def __init__(self, length: int, frequencies, column_weight, neff: float):
            self.length = length
            self.frequencies = frequencies
            self.column_weight = column_weight
            self.log10_neff = float(np.log10(neff))

    def test_an_unobserved_mutant_and_a_deep_column_stay_bounded(self):
        length = len(WILD)
        frequencies = np.full((length, 20), 1e-9)
        for index, residue in enumerate(WILD):
            frequencies[index, 'ACDEFGHIKLMNPQRSTVWY'.index(residue)] = 1.0
        weight = np.full(length, 5985.6)
        profile = self._Profile(length, frequencies, weight, 5985.6)
        block = profile_bounded_block(profile, WILD, 4, 'D')
        self.assertEqual(len(block), len(PROFILE_BOUNDED_FEATURE_ORDER))
        self.assertTrue(np.all(np.abs(block) <= PROFILE_LOG_CLIP + 1e-9),
                        'a bounded profile coordinate left its declared range')
        self.assertEqual(block[2], 1.0, 'an unobserved mutant must raise the indicator')
        self.assertAlmostEqual(block[1], -PROFILE_LOG_CLIP, places=9)
        self.assertLess(block[5], 4.0, 'the column weight must enter on a log scale')

    def test_an_absent_profile_is_a_declared_absence(self):
        block = profile_bounded_block(None, WILD, 4, 'D')
        self.assertEqual(list(block), [0.0] * len(PROFILE_BOUNDED_FEATURE_ORDER))
        self.assertEqual(block[-1], 0.0, 'the availability indicator must read zero')


class TheBlockWidthsAreTheDeclaredOnes(unittest.TestCase):
    def test_identity_and_geometry_widths(self):
        block = identity_block(WILD, 4, 'D')
        self.assertEqual(len(block), 400)
        self.assertEqual(block.sum(), 1.0)
        self.assertEqual(len(geometry_block(4, len(WILD))), 6)

    def test_single_substitution_recognises_only_one_change(self):
        self.assertEqual(single_substitution(WILD, mutate(WILD, 4, 'D')), 4)
        self.assertIsNone(single_substitution(WILD, WILD))
        self.assertIsNone(single_substitution(WILD, mutate(mutate(WILD, 4, 'D'), 9, 'W')))
        self.assertIsNone(single_substitution(WILD, WILD[:-1]))


if __name__ == '__main__':
    unittest.main()
