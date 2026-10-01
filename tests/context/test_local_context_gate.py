"""Conditions that must hold before a local control may bound anything.

Four of them are the reason this experiment exists at all. The qualification
gate has to refuse a candidate that lowers the held-cluster correlation of the
baseline it augments, because that is exactly the failure the declared
dipeptide/tripeptide block already produced and the reason the local-motif rung
carries no verdict. The support, folds and split seeds have to be the ones the
crossed-controls fits used, or the numbers do not compose with that study. No
held-out group label may reach feature construction, feature scaling or penalty
tuning. And the limited-receptive-field comparator has to genuinely see its
window and nothing else, which is a property of the encoder rather than of the
prose describing it.

The remaining tests pin the exactness of the three descriptor blocks and the
refusals every block inherits from the shared substitution validator.
"""
from __future__ import annotations

import json
import math
import unittest

import numpy as np
import torch

from src.capability.core.amino_acids import AA20, BLOSUM62_ORDER, BLOSUM62_ROWS
from src.capability.context.crossed_controls import evaluate_crossed_controls
from src.capability.context.crossed_controls import fold_membership as crossed_fold_membership
from src.capability.core.protein_properties import AA_CLASSES, CLASS_NAMES, CLASS_OF_RESIDUE
from src.capability.context.local_context import (
    ADDITIONS, CANDIDATE_BLOCKS, PRIMARY_RECEPTIVE_FIELD_RADIUS, QUALIFICATION_BASELINES, RADII,
    RECEPTIVE_FIELD_CANDIDATES, RECEPTIVE_FIELD_RADII, SCALE_ORDER, SPLIT_SEEDS, _scale_matrix,
    assemble_designs, candidate_blocks, candidate_feature_names, control_sets, declaration,
    declaration_sha256, evaluate_local_context, fold_membership, qualification_gate,
    receptive_field_features, window_chemistry_features, window_composition_features,
    window_substitution_features)

#: The digest of the declared block set, fixed before any fitting cell was
#: dispatched. A change to any declared radius, scale table, alphabet partition,
#: encoding or gate rule moves it, which is the point: the declaration is part of
#: the run's identity and cannot be revised after an outcome has been seen.
DECLARED_SHA256 = '255b7e3d1e42893e9c05cd201e2ad9685015a24bc14b5aeefbdee102e5aa25c1'


def apply_mutations(wildtype: str, mutant: str) -> str:
    sequence = list(wildtype)
    for token in mutant.split(':'):
        sequence[int(token[1:-1]) - 1] = token[-1]
    return ''.join(sequence)


def synthetic_panel(seed: int = 3, n_assays: int = 12, n_variants: int = 24,
                    widths=(3, 2, 2, 2, 4), local_widths=(2, 5)) -> list[dict]:
    """A small panel with reduced block widths, so many nested fits stay cheap.

    The production widths are 444 for composition, 14 for the profile block,
    1,024 for the representation and 30 to 600 for the candidate local blocks; a
    nested fit at that size is a cluster job rather than a unit test. Nothing in
    the evaluation depends on the widths, which is what makes the substitution
    legitimate here, and the per-block width and exactness tests below cover the
    production encodings separately. ``L`` and ``T`` are carried so that the same
    rows can be handed to the crossed-controls evaluation for the identity check.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_assays):
        mutants = [f'A{position + 1}C' for position in range(n_variants)]
        blocks = {name: rng.normal(size=(n_variants, width))
                  for name, width in zip(('S', 'L', 'P_block', 'T', 'R'), widths)}
        local = {name: rng.normal(size=(n_variants, width))
                 for name, width in zip(('wcomp', 'wsub'), local_widths)}
        signal = blocks['S'][:, 0] + 0.6 * blocks['R'][:, 0]
        rows.append(dict(assay=f'assay{index:02d}', cluster=index, mutants=mutants,
                         measured=signal + rng.normal(size=n_variants) * 0.3,
                         P=rng.normal(size=n_variants),
                         M=signal + rng.normal(size=n_variants) * 0.8,
                         **blocks, **local))
    return rows


def gate_summaries(points, standalone_interval=(0.05, 0.15), candidate='wall'):
    """Crafted ``summarize`` records, one per declared split seed."""
    measured = {}
    for seed, (over_c, over_cp) in zip(SPLIT_SEEDS, points):
        measured[str(seed)] = {
            f'qualification_delta_over_C_{candidate}': dict(point=over_c, interval=[over_c - 0.01,
                                                                                    over_c + 0.01]),
            f'qualification_delta_over_C_P_{candidate}': dict(point=over_cp,
                                                              interval=[over_cp - 0.01,
                                                                        over_cp + 0.01]),
            f'standalone_{candidate}': dict(point=float(np.mean(standalone_interval)),
                                            interval=list(standalone_interval)),
        }
    return measured


class DeclarationTests(unittest.TestCase):
    def test_the_declaration_is_complete_digested_and_frozen(self):
        declared = declaration()
        names = candidate_feature_names()
        self.assertEqual(tuple(names), CANDIDATE_BLOCKS)
        for name, columns in names.items():
            self.assertEqual(len(columns), len(set(columns)), name)
            self.assertEqual(declared['widths'][name], len(columns), name)
        self.assertEqual(declared['widths'],
                         {'wcomp': len(RADII) * (len(CLASS_NAMES) + 2),
                          'wchem': len(SCALE_ORDER) * (1 + 2 * len(RADII)),
                          'wsub': 3 * len(RADII),
                          'wall': len(RADII) * (len(CLASS_NAMES) + 2)
                                  + len(SCALE_ORDER) * (1 + 2 * len(RADII)) + 3 * len(RADII),
                          'rf3': 2 * 7 * len(AA20), 'rf7': 2 * 15 * len(AA20)})
        self.assertEqual(declaration_sha256(), DECLARED_SHA256)
        json.dumps(declared, allow_nan=False)

    def test_the_alphabet_partition_and_scales_are_the_declared_ones(self):
        covered = ''.join(AA_CLASSES[name] for name in CLASS_NAMES)
        self.assertEqual(sorted(covered), sorted(AA20))
        self.assertEqual(len(covered), len(AA20))
        self.assertEqual({CLASS_OF_RESIDUE[residue] for residue in AA20}, set(CLASS_NAMES))
        scales = _scale_matrix()
        self.assertEqual(scales.shape, (len(SCALE_ORDER), len(AA20)))
        np.testing.assert_allclose(scales.mean(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(scales.std(axis=1), 1.0, atol=1e-12)
        self.assertEqual(PRIMARY_RECEPTIVE_FIELD_RADIUS in RECEPTIVE_FIELD_RADII, True)
        self.assertEqual(tuple(f'rf{radius}' for radius in RECEPTIVE_FIELD_RADII),
                         RECEPTIVE_FIELD_CANDIDATES)


class QualificationGateTests(unittest.TestCase):
    """The gate is a refusal, and the refusal is the reason this module exists."""

    def test_a_candidate_that_lowers_either_baseline_at_any_seed_is_refused(self):
        qualified = qualification_gate(gate_summaries([(0.01, 0.01)] * 3), ['wall'])
        self.assertEqual(qualified['verdicts']['wall']['qualified'], True)
        self.assertEqual(qualified['qualified'], ['wall'])
        self.assertEqual(qualified['primary_local_control'], 'wall')
        # One negative seed on the composition baseline is enough, however large
        # the other two are: this is the crossed-controls failure mode exactly,
        # where the block raised nothing and lowered C by 0.04747.
        for points in ([(-0.04747, 0.01), (0.20, 0.20), (0.20, 0.20)],
                       [(0.20, 0.20), (0.20, -0.01684), (0.20, 0.20)],
                       [(0.20, 0.20), (0.20, 0.20), (-1e-9, 0.20)]):
            refused = qualification_gate(gate_summaries(points), ['wall'])
            self.assertEqual(refused['verdicts']['wall']['qualified'], False, points)
            self.assertEqual(refused['discarded'], ['wall'])
            self.assertEqual(refused['qualified'], [])
            self.assertIsNone(refused['primary_local_control'])
            self.assertEqual(refused['carried_forward'], [])
            self.assertTrue(refused['verdicts']['wall']['refusals'])
        # "Raise, or at minimum not lower" -- exactly zero is not a refusal.
        flat = qualification_gate(gate_summaries([(0.0, 0.0)] * 3), ['wall'])
        self.assertEqual(flat['verdicts']['wall']['qualified'], True)

    def test_a_receptive_field_comparator_must_also_generalise_on_its_own(self):
        for interval in ((-0.02, 0.10), (0.0, 0.10)):
            refused = qualification_gate(
                gate_summaries([(0.05, 0.05)] * 3, standalone_interval=interval, candidate='rf7'),
                ['rf7'])
            self.assertEqual(refused['verdicts']['rf7']['qualified'], False, interval)
            self.assertEqual(refused['receptive_field_control'], None)
        admitted = qualification_gate(
            gate_summaries([(0.05, 0.05)] * 3, standalone_interval=(0.02, 0.10), candidate='rf7'),
            ['rf7'])
        self.assertEqual(admitted['verdicts']['rf7']['qualified'], True)
        self.assertEqual(admitted['receptive_field_control'], 'rf7')
        self.assertEqual(admitted['primary_local_control'], None)
        self.assertEqual(admitted['carried_forward'], ['rf7'])

    def test_the_gate_refuses_incomplete_evidence_rather_than_guessing(self):
        measured = gate_summaries([(0.05, 0.05)] * 3)
        del measured[str(SPLIT_SEEDS[-1])]
        with self.assertRaisesRegex(ValueError, 'every declared split seed'):
            qualification_gate(measured, ['wall'])
        measured = gate_summaries([(0.05, 0.05)] * 3)
        del measured[str(SPLIT_SEEDS[0])]['qualification_delta_over_C_P_wall']
        with self.assertRaisesRegex(ValueError, 'was not measured'):
            qualification_gate(measured, ['wall'])
        measured = gate_summaries([(0.05, 0.05)] * 3)
        measured[str(SPLIT_SEEDS[0])]['qualification_delta_over_C_wall']['point'] = None
        with self.assertRaisesRegex(ValueError, 'no point estimate'):
            qualification_gate(measured, ['wall'])
        with self.assertRaisesRegex(ValueError, 'undeclared candidate'):
            qualification_gate(gate_summaries([(0.05, 0.05)] * 3), ['not_declared'])

    def test_the_primary_control_is_chosen_on_control_correlations_alone(self):
        measured = {}
        for seed in SPLIT_SEEDS:
            record = {}
            for candidate, increase in (('wcomp', 0.01), ('wchem', 0.03), ('wsub', 0.02)):
                for baseline in QUALIFICATION_BASELINES:
                    record[f'qualification_delta_over_{baseline}_{candidate}'] = dict(
                        point=increase, interval=[increase - 0.005, increase + 0.005])
            measured[str(seed)] = record
        gate = qualification_gate(measured, ['wcomp', 'wchem', 'wsub'])
        self.assertEqual(gate['primary_local_control'], 'wchem')
        self.assertEqual(sorted(gate['qualified']), ['wchem', 'wcomp', 'wsub'])


class SupportAndFoldTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_control_sets_are_genuinely_nested_on_real_columns(self):
        rng = np.random.default_rng(1)
        blocks = {name: rng.normal(size=(40, width)) for name, width
                  in (('S', 5), ('P', 3), ('M', 1), ('R', 6), ('wcomp', 4), ('wsub', 2))}
        sets = control_sets(['wcomp', 'wsub'])
        designs = assemble_designs(blocks, sets, ADDITIONS)

        def columns(name):
            return {tuple(column) for column in designs[name].T}

        for child, parent in (('C', 'C_wcomp'), ('wcomp', 'C_wcomp'), ('C', 'C_P'),
                              ('C_P', 'C_P_wcomp'), ('C_wcomp', 'C_P_wcomp'),
                              ('C', 'C_wsub'), ('wsub', 'C_wsub')):
            self.assertTrue(columns(child) < columns(parent), f'{child} is not nested in {parent}')
        self.assertFalse(columns('C_wcomp') <= columns('C_wsub'))
        for name in sets:
            for addition, extra in ADDITIONS.items():
                self.assertTrue(columns(name) <= columns(name + addition))
                self.assertEqual(designs[name + addition].shape[1],
                                 designs[name].shape[1]
                                 + sum(blocks[block].shape[1] for block in extra))
        with self.assertRaisesRegex(ValueError, 'missing feature blocks'):
            assemble_designs({'S': blocks['S']}, sets, ADDITIONS)
        with self.assertRaisesRegex(ValueError, 'undeclared local blocks'):
            control_sets(['not_declared'])
        with self.assertRaisesRegex(ValueError, 'at least one distinct local block'):
            control_sets(['wall', 'wall'])

    def test_every_design_shares_one_support_and_one_fold_map(self):
        rows = synthetic_panel()
        report, predictions = evaluate_local_context(rows, local_blocks=['wcomp', 'wsub'],
                                                     bootstrap=32, seed=7, fold_seed=11)
        self.assertTrue(report['folds_identical_across_designs'])
        self.assertEqual(report['n_assays'], len(rows))
        self.assertEqual(report['n_variants'], sum(len(r['mutants']) for r in rows))
        self.assertEqual({len(value) for value in predictions.values()}, {report['n_variants']})
        self.assertEqual(set(predictions), set(report['feature_dimensions']))
        for outer in report['folds']:
            self.assertFalse(set(outer['held_families']) & set(outer['training_families']))
            for inner in outer['inner_folds']:
                self.assertFalse(set(inner['validation_families']) & set(outer['held_families']))
        json.dumps(report, allow_nan=False)

    def test_the_support_folds_and_seeds_are_the_crossed_controls_ones(self):
        """The composition and profile controls must be the same fits, bit for bit."""
        rows = synthetic_panel()
        mine, my_predictions = evaluate_local_context(rows, local_blocks=['wcomp'], bootstrap=16,
                                                      seed=7, fold_seed=20260924)
        theirs, their_predictions = evaluate_crossed_controls(rows, bootstrap=16, seed=7,
                                                             fold_seed=20260924)
        self.assertEqual(fold_membership(mine['folds']),
                         crossed_fold_membership(theirs['folds']))
        self.assertEqual(mine['n_assays'], theirs['n_assays'])
        self.assertEqual(mine['n_variants'], theirs['n_variants'])
        self.assertEqual(mine['n_families'], theirs['n_families'])
        for design in ('C', 'C_P', 'C+M', 'C+R', 'C+M+R', 'C_P+M', 'C_P+R', 'C_P+M+R'):
            self.assertEqual(mine['prediction_digests'][design],
                             theirs['prediction_digests'][design], design)
            np.testing.assert_array_equal(my_predictions[design], their_predictions[design])
            self.assertEqual(mine['selected_alphas'][design], theirs['selected_alphas'][design])
        for control in ('C', 'C_P'):
            for label in ('M', 'R', 'R_after_M'):
                self.assertEqual(mine['summaries'][f'increment_{label}_{control}'],
                                 theirs['summaries'][f'increment_{label}_{control}'])

    def test_held_group_labels_reach_neither_features_scaling_tuning_nor_predictions(self):
        rows = synthetic_panel()
        report, predictions = evaluate_local_context(rows, local_blocks=['wcomp'], bootstrap=16,
                                                     seed=7, fold_seed=11)
        held = set(report['folds'][0]['held_families'])
        rng = np.random.default_rng(99)
        corrupted = []
        for row in rows:
            copy = dict(row)
            if row['cluster'] in held:
                copy['measured'] = rng.normal(size=len(row['mutants'])) * 50.0
            corrupted.append(copy)
        changed, changed_predictions = evaluate_local_context(corrupted, local_blocks=['wcomp'],
                                                             bootstrap=16, seed=7, fold_seed=11)
        self.assertEqual(fold_membership(changed['folds']), fold_membership(report['folds']))
        index = np.concatenate([[row['cluster'] in held] * len(row['mutants']) for row in rows])
        self.assertGreater(index.sum(), 0)
        # Only the fold that holds these families out is invariant. In the other
        # four outer folds the same families are training data, so their tuning
        # is allowed to move and asserting otherwise would test the wrong thing.
        for name in predictions:
            np.testing.assert_allclose(changed_predictions[name][index], predictions[name][index],
                                       atol=1e-12, err_msg=name)
            self.assertEqual(changed['selected_alphas'][name][0],
                             report['selected_alphas'][name][0], name)
        # Feature construction takes sequences only. Every block is built from
        # the wild type, the mutation strings and the retained mutant sequences,
        # so a label cannot enter a column even in principle; this asserts it on
        # the production builders rather than on the reduced-width panel.
        wildtype = 'ACDEFGHIKLMNPQRSTVWY' * 3
        mutants = ['A1W', 'C2Y:D3K', 'G26P']
        sequences = [apply_mutations(wildtype, mutant) for mutant in mutants]
        first = candidate_blocks(wildtype, mutants, sequences)
        second = candidate_blocks(wildtype, mutants, sequences)
        for name in CANDIDATE_BLOCKS:
            np.testing.assert_array_equal(first[name], second[name])

    def test_a_qualification_cell_fits_no_model_quantity(self):
        rows = synthetic_panel(n_assays=8, n_variants=12)
        report, predictions = evaluate_local_context(rows, local_blocks=['wcomp', 'wsub'],
                                                     additions={'': ()}, bootstrap=8, seed=7,
                                                     fold_seed=11)
        self.assertEqual(report['model_quantities_fitted'], [])
        self.assertEqual(set(predictions), set(control_sets(['wcomp', 'wsub'])))
        for name in predictions:
            self.assertNotIn('+M', name)
            self.assertNotIn('+R', name)
        for candidate in ('wcomp', 'wsub'):
            for baseline in QUALIFICATION_BASELINES:
                key = f'qualification_delta_over_{baseline}_{candidate}'
                self.assertIn(key, report['summaries'])
                point = report['summaries'][key]['point']
                measured = report['summaries'][f'{baseline}_{candidate}_spearman']['point']
                base = report['summaries'][f'{baseline}_spearman']['point']
                self.assertAlmostEqual(point, measured - base, places=10)
            self.assertIn(f'standalone_{candidate}', report['summaries'])
        with self.assertRaisesRegex(ValueError, 'empty addition'):
            evaluate_local_context(rows, local_blocks=['wcomp'], additions={'+M': ('M',)},
                                   bootstrap=4)

    def test_unaligned_or_nonfinite_blocks_and_duplicate_assays_are_refused(self):
        rows = synthetic_panel(n_assays=6, n_variants=8)
        broken = [dict(row) for row in rows]
        broken[0]['wcomp'] = broken[0]['wcomp'][:-1]
        with self.assertRaisesRegex(ValueError, 'unaligned or nonfinite wcomp'):
            evaluate_local_context(broken, local_blocks=['wcomp'], bootstrap=4)
        broken = [dict(row) for row in rows]
        broken[1]['P_block'] = broken[1]['P_block'].copy()
        broken[1]['P_block'][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'unaligned or nonfinite P_block'):
            evaluate_local_context(broken, local_blocks=['wcomp'], bootstrap=4)
        with self.assertRaisesRegex(ValueError, 'duplicate assays'):
            evaluate_local_context(rows + [rows[0]], local_blocks=['wcomp'], bootstrap=4)


class ReceptiveFieldTests(unittest.TestCase):
    """The comparator has to see its window and nothing else."""

    wildtype = 'ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY'

    def test_a_residue_outside_every_window_cannot_change_a_row(self):
        radius = PRIMARY_RECEPTIVE_FIELD_RADIUS
        mutants = ['K9W', 'C2Y:D3K']
        sequences = [apply_mutations(self.wildtype, mutant) for mutant in mutants]
        reference = receptive_field_features(self.wildtype, mutants, sequences, radius)
        positions = {int(token[1:-1]) - 1 for mutant in mutants for token in mutant.split(':')}
        outside = [site for site in range(len(self.wildtype))
                   if all(abs(site - position) > radius for position in positions)]
        self.assertTrue(outside)
        for site in outside:
            replacement = AA20[(AA20.index(self.wildtype[site]) + 7) % len(AA20)]
            altered = self.wildtype[:site] + replacement + self.wildtype[site + 1:]
            altered_sequences = [sequence[:site] + replacement + sequence[site + 1:]
                                 for sequence in sequences]
            np.testing.assert_array_equal(
                receptive_field_features(altered, mutants, altered_sequences, radius), reference)
        inside = [site for site in range(len(self.wildtype))
                  if any(abs(site - position) <= radius for position in positions)
                  and site not in positions]
        changed = 0
        for site in inside:
            replacement = AA20[(AA20.index(self.wildtype[site]) + 7) % len(AA20)]
            altered = self.wildtype[:site] + replacement + self.wildtype[site + 1:]
            altered_sequences = [sequence[:site] + replacement + sequence[site + 1:]
                                 for sequence in sequences]
            if not np.array_equal(
                    receptive_field_features(altered, mutants, altered_sequences, radius),
                    reference):
                changed += 1
        self.assertEqual(changed, len(inside))

    def test_the_same_window_at_a_different_position_and_length_gives_the_same_row(self):
        radius = 3
        core = 'ACDEFGH'
        for pad_left, pad_right in ((5, 5), (11, 30)):
            wildtype = 'W' * pad_left + core + 'Y' * pad_right
            centre = pad_left + 3
            mutant = f'{wildtype[centre]}{centre + 1}K'
            sequence = apply_mutations(wildtype, mutant)
            row = receptive_field_features(wildtype, [mutant], [sequence], radius)
            if pad_left == 5:
                reference = row
            else:
                np.testing.assert_array_equal(row, reference)

    def test_the_wild_type_half_carries_the_in_window_position_count(self):
        radius = 2
        span = 2 * radius + 1
        half = span * len(AA20)
        mutants = ['A1W', 'F5W:H7Y', 'Y40W']
        sequences = [apply_mutations(self.wildtype, mutant) for mutant in mutants]
        rows = receptive_field_features(self.wildtype, mutants, sequences, radius)
        for row, mutant in zip(rows, mutants):
            positions = [int(token[1:-1]) - 1 for token in mutant.split(':')]
            expected = sum(len([site for site in range(position - radius, position + radius + 1)
                                if 0 <= site < len(self.wildtype)]) for position in positions)
            self.assertEqual(row[:half].sum(), float(expected))
            # A single substitution changes exactly one window position, so its
            # difference half carries exactly one +1 and one -1, at the centre.
            if len(positions) == 1:
                self.assertEqual(np.count_nonzero(row[half:]), 2)
                self.assertEqual(row[half:].sum(), 0.0)

    def test_the_encoding_is_exact_against_a_hand_built_window(self):
        radius = 1
        wildtype = 'ACDEF'
        mutant = 'D3K'
        sequence = apply_mutations(wildtype, mutant)
        row = receptive_field_features(wildtype, [mutant], [sequence], radius)[0]
        span = 2 * radius + 1
        half = span * len(AA20)
        expected = np.zeros(2 * half)
        for order, residue in enumerate('CDE'):
            expected[order * len(AA20) + AA20.index(residue)] += 1.0
        for order, (wild, mutated) in enumerate(zip('CDE', 'CKE')):
            expected[half + order * len(AA20) + AA20.index(mutated)] += 1.0
            expected[half + order * len(AA20) + AA20.index(wild)] -= 1.0
        np.testing.assert_array_equal(row, expected)
        with self.assertRaisesRegex(ValueError, 'radius must be positive'):
            receptive_field_features(wildtype, [mutant], [sequence], 0)


class DescriptorBlockTests(unittest.TestCase):
    wildtype = 'ACDEFGHIKLMNPQRSTVWY'

    def test_window_composition_matches_a_brute_force_recount(self):
        mutants = ['A1W', 'K9W:L10Y', 'Y20W']
        sequences = [apply_mutations(self.wildtype, mutant) for mutant in mutants]
        rows = window_composition_features(self.wildtype, mutants, sequences)
        self.assertEqual(rows.shape, (3, len(RADII) * (len(CLASS_NAMES) + 2)))
        for row, mutant in zip(rows, mutants):
            positions = [int(token[1:-1]) - 1 for token in mutant.split(':')]
            expected = np.zeros(rows.shape[1])
            for position in positions:
                offset = 0
                for radius in RADII:
                    sites = [site for site in range(position - radius, position + radius + 1)
                             if 0 <= site < len(self.wildtype) and site != position]
                    counts = [sum(1 for site in sites
                                  if CLASS_OF_RESIDUE[self.wildtype[site]] == name)
                              for name in CLASS_NAMES]
                    fractions = [count / len(sites) for count in counts]
                    expected[offset:offset + len(CLASS_NAMES)] += fractions
                    expected[offset + len(CLASS_NAMES)] += -sum(
                        value * math.log(value) for value in fractions if value > 0.0)
                    expected[offset + len(CLASS_NAMES) + 1] += len(sites) / (2 * radius)
                    offset += len(CLASS_NAMES) + 2
            np.testing.assert_allclose(row, expected / len(positions), atol=1e-12)

    def test_window_chemistry_carries_the_declared_delta_and_interaction(self):
        scales = _scale_matrix()
        mutants = ['I8W']
        sequences = [apply_mutations(self.wildtype, mutants[0])]
        row = window_chemistry_features(self.wildtype, mutants, sequences)[0]
        n_scales = len(SCALE_ORDER)
        delta = scales[:, AA20.index('W')] - scales[:, AA20.index('I')]
        np.testing.assert_allclose(row[:n_scales], delta, atol=1e-12)
        for order, radius in enumerate(RADII):
            sites = [site for site in range(7 - radius, 7 + radius + 1)
                     if 0 <= site < len(self.wildtype) and site != 7]
            means = scales[:, [AA20.index(self.wildtype[site]) for site in sites]].mean(axis=1)
            start = n_scales * (1 + order)
            np.testing.assert_allclose(row[start:start + n_scales], means, atol=1e-12)
            start = n_scales * (1 + len(RADII) + order)
            np.testing.assert_allclose(row[start:start + n_scales], delta * means, atol=1e-12)

    def test_window_substitution_scores_match_blosum62_over_the_window(self):
        order = {residue: index for index, residue in enumerate(BLOSUM62_ORDER)}

        def score(left, right):
            return float(BLOSUM62_ROWS[order[left]][order[right]])

        mutants = ['L10K']
        sequences = [apply_mutations(self.wildtype, mutants[0])]
        row = window_substitution_features(self.wildtype, mutants, sequences)[0]
        self.assertEqual(row.shape, (3 * len(RADII),))
        for index, radius in enumerate(RADII):
            sites = [site for site in range(9 - radius, 9 + radius + 1)
                     if 0 <= site < len(self.wildtype) and site != 9]
            mutant_mean = np.mean([score('K', self.wildtype[site]) for site in sites])
            wild_mean = np.mean([score('L', self.wildtype[site]) for site in sites])
            self.assertAlmostEqual(row[3 * index], mutant_mean, places=12)
            self.assertAlmostEqual(row[3 * index + 1], wild_mean, places=12)
            self.assertAlmostEqual(row[3 * index + 2], mutant_mean - wild_mean, places=12)

    def test_every_block_refuses_an_inconsistent_variant(self):
        builders = (window_composition_features, window_chemistry_features,
                    window_substitution_features,
                    lambda w, m, s: receptive_field_features(w, m, s, 2))
        cases = (('C1W', 'WCDEFGHIKLMNPQRSTVWY', 'disagrees with the wild type'),
                 ('A1W:A1Y', 'WCDEFGHIKLMNPQRSTVWY', 'repeats a position'),
                 ('A99W', 'WCDEFGHIKLMNPQRSTVWY', 'disagrees with the wild type'),
                 ('A1X', 'XCDEFGHIKLMNPQRSTVWY', 'canonical alphabet'),
                 ('A1W', 'YCDEFGHIKLMNPQRSTVWY', 'does not match its mutation string'))
        for builder in builders:
            for mutant, sequence, message in cases:
                with self.assertRaisesRegex(ValueError, message):
                    builder(self.wildtype, [mutant], [sequence])
            with self.assertRaisesRegex(ValueError, 'preserve wild-type length'):
                builder(self.wildtype, ['A1W'], ['WCDEFGHIKLMNPQRSTVW'])
            with self.assertRaisesRegex(ValueError, 'unaligned'):
                builder(self.wildtype, ['A1W'], [])
            with self.assertRaisesRegex(ValueError, 'noncanonical residue'):
                builder('AXDEFGHIKLMNPQRSTVWY', ['A1W'], ['WXDEFGHIKLMNPQRSTVWY'])

    def test_the_production_blocks_carry_their_declared_widths(self):
        mutants = ['A1W', 'K9W:L10Y']
        sequences = [apply_mutations(self.wildtype, mutant) for mutant in mutants]
        blocks = candidate_blocks(self.wildtype, mutants, sequences)
        self.assertEqual(tuple(blocks), CANDIDATE_BLOCKS)
        widths = declaration()['widths']
        for name, values in blocks.items():
            self.assertEqual(values.shape, (len(mutants), widths[name]), name)
        np.testing.assert_array_equal(
            blocks['wall'],
            np.column_stack([blocks['wcomp'], blocks['wchem'], blocks['wsub']]))


if __name__ == '__main__':
    unittest.main()
