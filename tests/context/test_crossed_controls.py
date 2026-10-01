"""Conditions that must hold for every crossed-control comparison.

Three of them are the reason this experiment can be read as an increment at all:
identical support and folds across every compared fit, no held-out group label
reaching feature scaling or penalty tuning, and control sets that are genuinely
nested. The remaining tests pin the exactness of the three new feature blocks and
their refusals.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.capability.core.amino_acids import AA20
from src.capability.context.crossed_controls import (
    ADDITIONS, BLOCK_ORDER, CONTROL_SETS, LOCAL_PROJECTION_DIM, PROFILE_FEATURE_ORDER,
    TOKENISATION_FEATURE_ORDER, assemble_designs, evaluate_crossed_controls, fold_membership,
    local_pattern_features, local_projection, profile_features, substitution_table,
    tokenisation_features)
from src.capability.context.profiles import PSEUDOCOUNT_ALPHA, Profile, profile_scores

CODE = {residue: index for index, residue in enumerate(AA20)}
SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/capability/context/analyse_crossed_controls.py'
_spec = importlib.util.spec_from_file_location('crossed_controls_cli', SCRIPT)
assert _spec is not None and _spec.loader is not None
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def brute_force_kmer_difference(wildtype: str, mutant_sequence: str, k: int) -> dict[int, float]:
    counts: dict[int, float] = {}
    for sequence, sign in ((mutant_sequence, 1.0), (wildtype, -1.0)):
        for start in range(len(sequence) - k + 1):
            index = 0
            for offset in range(k):
                index = index * 20 + CODE[sequence[start + offset]]
            counts[index] = counts.get(index, 0.0) + sign
    return {index: value for index, value in counts.items() if value != 0.0}


def synthetic_panel(seed: int = 3, n_assays: int = 12, n_variants: int = 24,
                    widths=(3, 2, 2, 2, 4)) -> list[dict]:
    """A small panel with reduced block widths, so 20 nested fits stay cheap.

    Widths are reduced deliberately: the production widths are 444/656/14/8/1024
    and a nested fit at that size is a cluster job, not a unit test. Nothing in
    the evaluation depends on the widths, which is what makes the substitution
    legitimate here and what the separate per-block width tests confirm.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_assays):
        mutants = [f'A{position + 1}C' for position in range(n_variants)]
        blocks = {name: rng.normal(size=(n_variants, width))
                  for name, width in zip(('S', 'L', 'P_block', 'T', 'R'), widths)}
        signal = blocks['S'][:, 0] + 0.6 * blocks['R'][:, 0]
        rows.append(dict(assay=f'assay{index:02d}', cluster=index, mutants=mutants,
                         measured=signal + rng.normal(size=n_variants) * 0.3,
                         P=rng.normal(size=n_variants), M=signal + rng.normal(size=n_variants) * 0.8,
                         **blocks))
    return rows


class NestingAndSupportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_control_sets_are_genuinely_nested_on_real_columns(self):
        rng = np.random.default_rng(1)
        blocks = {name: rng.normal(size=(40, width))
                  for name, width in zip((*BLOCK_ORDER, 'M', 'R'), (5, 4, 3, 2, 1, 6))}
        designs = assemble_designs(blocks)
        def columns(name):
            return {tuple(column) for column in designs[name].T}
        for child, parent in (('C', 'C_L'), ('C', 'C_P'), ('C_L', 'C_L_P'), ('C_P', 'C_L_P'),
                              ('C_L_P', 'C_L_P_T')):
            self.assertTrue(columns(child) < columns(parent), f'{child} is not nested in {parent}')
        # C+L and C+P are crossed, not ordered: neither contains the other.
        self.assertFalse(columns('C_L') <= columns('C_P'))
        self.assertFalse(columns('C_P') <= columns('C_L'))
        for control in CONTROL_SETS:
            for addition in ADDITIONS:
                self.assertTrue(columns(control) <= columns(control + addition))
                self.assertEqual(designs[control + addition].shape[1],
                                 designs[control].shape[1]
                                 + sum(blocks[name].shape[1] for name in ADDITIONS[addition]))
        with self.assertRaisesRegex(ValueError, 'missing feature blocks'):
            assemble_designs({'S': blocks['S']})

    def test_every_design_shares_one_support_and_one_fold_map(self):
        rows = synthetic_panel()
        report, predictions = evaluate_crossed_controls(rows, bootstrap=32, seed=7, fold_seed=11)
        self.assertTrue(report['folds_identical_across_designs'])
        self.assertEqual(report['n_assays'], len(rows))
        self.assertEqual(report['n_variants'], sum(len(r['mutants']) for r in rows))
        lengths = {name: len(value) for name, value in predictions.items()}
        self.assertEqual(set(lengths.values()), {report['n_variants']})
        self.assertEqual(set(predictions), set(report['feature_dimensions']))
        for outer in report['folds']:
            self.assertFalse(set(outer['held_families']) & set(outer['training_families']))
            for inner in outer['inner_folds']:
                self.assertFalse(set(inner['validation_families']) & set(outer['held_families']))
        json.dumps(report, allow_nan=False)

    def test_held_group_labels_cannot_reach_scaling_tuning_or_their_own_predictions(self):
        rows = synthetic_panel()
        report, predictions = evaluate_crossed_controls(rows, bootstrap=16, seed=7, fold_seed=11)
        held = set(report['folds'][0]['held_families'])
        rng = np.random.default_rng(99)
        corrupted = []
        for row in rows:
            copy = dict(row)
            if row['cluster'] in held:
                copy['measured'] = rng.normal(size=len(row['mutants'])) * 50.0
            corrupted.append(copy)
        changed, changed_predictions = evaluate_crossed_controls(corrupted, bootstrap=16, seed=7,
                                                                fold_seed=11)
        self.assertEqual(fold_membership(changed['folds']), fold_membership(report['folds']))
        index = np.concatenate([[row['cluster'] in held] * len(row['mutants']) for row in rows])
        self.assertGreater(index.sum(), 0)
        for name in predictions:
            np.testing.assert_allclose(changed_predictions[name][index], predictions[name][index],
                                       atol=1e-12, err_msg=name)
            self.assertEqual(changed['selected_alphas'][name][0], report['selected_alphas'][name][0])

    def test_unaligned_or_nonfinite_blocks_and_duplicate_assays_are_refused(self):
        rows = synthetic_panel(n_assays=6, n_variants=8)
        broken = [dict(row) for row in rows]
        broken[0]['L'] = broken[0]['L'][:-1]
        with self.assertRaisesRegex(ValueError, 'unaligned or nonfinite L'):
            evaluate_crossed_controls(broken, bootstrap=4)
        broken = [dict(row) for row in rows]
        broken[1]['R'] = broken[1]['R'].copy()
        broken[1]['R'][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'unaligned or nonfinite R'):
            evaluate_crossed_controls(broken, bootstrap=4)
        with self.assertRaisesRegex(ValueError, 'duplicate assays'):
            evaluate_crossed_controls(rows + [rows[0]], bootstrap=4)


class LocalPatternTests(unittest.TestCase):
    def test_dipeptide_and_tripeptide_differences_are_exact_including_overlaps(self):
        wildtype = 'ACDEFGHIKL'
        projection = local_projection(dim=8, seed=5)
        cases = ['A1W', 'C2W:D3Y', 'F5W:G6Y:H7W', 'A1W:L10Y']
        sequences = []
        for mutant in cases:
            sequence = list(wildtype)
            for token in mutant.split(':'):
                sequence[int(token[1:-1]) - 1] = token[-1]
            sequences.append(''.join(sequence))
        features = local_pattern_features(wildtype, cases, sequences, projection)
        self.assertEqual(features.shape, (len(cases), 400 + 8))
        for row, sequence in zip(features, sequences):
            expected2 = np.zeros(400)
            for index, value in brute_force_kmer_difference(wildtype, sequence, 2).items():
                expected2[index] = value
            np.testing.assert_allclose(row[:400], expected2, atol=0)
            expected3 = np.zeros(8)
            for index, value in brute_force_kmer_difference(wildtype, sequence, 3).items():
                expected3 += value * projection[index]
            np.testing.assert_allclose(row[400:], expected3, atol=1e-12)

    def test_production_width_and_refusals(self):
        projection = local_projection()
        self.assertEqual(projection.shape, (8000, LOCAL_PROJECTION_DIM))
        features = local_pattern_features('ACDE', ['A1W'], ['WCDE'], projection)
        self.assertEqual(features.shape, (1, 400 + LOCAL_PROJECTION_DIM))
        with self.assertRaisesRegex(ValueError, 'unaligned'):
            local_pattern_features('ACDE', ['A1W'], [], projection)
        with self.assertRaisesRegex(ValueError, 'preserve wild-type length'):
            local_pattern_features('ACDE', ['A1W'], ['WCD'], projection)
        with self.assertRaisesRegex(ValueError, '8,000 tripeptide'):
            local_pattern_features('ACDE', ['A1W'], ['WCDE'], np.zeros((10, 4)))
        with self.assertRaisesRegex(ValueError, 'local projection width'):
            local_projection(dim=0)

    def test_substitution_table_refusals(self):
        self.assertEqual(substitution_table('ACDE', 'A1W', 'WCDE'), [(0, CODE['A'], CODE['W'])])
        for mutant, sequence in (('C1W', 'WCDE'), ('A1W:A1Y', 'WCDE'), ('A9W', 'WCDE'),
                                 ('A1X', 'XCDE'), ('A1W', 'YCDE')):
            with self.assertRaises(ValueError):
                substitution_table('ACDE', mutant, sequence)


class ProfileBlockTests(unittest.TestCase):
    def setUp(self):
        self.wildtype = 'ACDE'
        self.frequencies = np.zeros((4, 20))
        self.frequencies[0, CODE['A']] = 0.5
        self.frequencies[0, CODE['W']] = 0.5
        self.frequencies[1, CODE['C']] = 1.0
        self.frequencies[2, CODE['D']] = 0.25
        self.frequencies[2, CODE['E']] = 0.75
        # Column 3 has no qualifying corpus support: every frequency is zero.
        self.background = np.full(20, 0.05)
        self.record = dict(log10_neff=1.5, max_identity_over_query=62.0)

    def test_mutation_local_coordinates_and_declared_order(self):
        mutants = ['A1W', 'E4W', 'A1W:D3E']
        sequences = ['WCDE', 'ACDW', 'WCEE']
        scores = [0.1, -0.2, 0.3]
        rows = profile_features(self.wildtype, mutants, sequences, scores, self.frequencies,
                                self.background, self.record, alpha=PSEUDOCOUNT_ALPHA)
        self.assertEqual(rows.shape, (3, len(PROFILE_FEATURE_ORDER)))
        order = {name: index for index, name in enumerate(PROFILE_FEATURE_ORDER)}
        self.assertAlmostEqual(rows[0, order['mean_wildtype_column_frequency']], 0.5)
        self.assertAlmostEqual(rows[0, order['mean_mutant_column_frequency']], 0.5)
        entropy = -(0.5 * np.log(0.5) * 2)
        self.assertAlmostEqual(rows[0, order['mean_mutated_column_entropy_nats']], entropy)
        self.assertAlmostEqual(rows[0, order['mean_mutated_column_supported']], 1.0)
        # The unsupported column reports zero entropy and zero coverage.
        self.assertAlmostEqual(rows[1, order['mean_mutated_column_entropy_nats']], 0.0)
        self.assertAlmostEqual(rows[1, order['mean_mutated_column_supported']], 0.0)
        self.assertAlmostEqual(rows[0, order['wildtype_supported_column_fraction']], 0.75)
        self.assertAlmostEqual(rows[0, order['wildtype_log10_neff']], 1.5)
        self.assertAlmostEqual(rows[0, order['wildtype_max_identity_over_query_fraction']], 0.62)
        self.assertAlmostEqual(rows[2, order['minimum_mutant_column_frequency']], 0.5)
        self.assertAlmostEqual(rows[2, order['maximum_wildtype_column_frequency']], 0.5)
        self.assertAlmostEqual(rows[1, order['minimum_mutant_column_frequency']], 0.0)

    def test_mean_log_odds_reproduces_the_admitted_lookup_score(self):
        profile = Profile(query_id='q', wildtype=self.wildtype, n_hits=1, n_sequences=1, saturated=False,
                          frequencies=self.frequencies, column_weight=self.frequencies.sum(axis=1),
                          neff=10.0, max_identity_over_query=62.0)
        mutants = ['A1W', 'A1W:D3E']
        sequences = ['WCDE', 'WCEE']
        variants = [tuple((token[0], int(token[1:-1]), token[-1]) for token in mutant.split(':'))
                    for mutant in mutants]
        expected = profile_scores(profile, self.background, variants, alpha=PSEUDOCOUNT_ALPHA)
        rows = profile_features(self.wildtype, mutants, sequences, expected, self.frequencies,
                                self.background, self.record, alpha=PSEUDOCOUNT_ALPHA)
        index = PROFILE_FEATURE_ORDER.index('mean_per_substitution_log_odds')
        counts = np.array([len(variant) for variant in variants], dtype=float)
        np.testing.assert_allclose(rows[:, index] * counts, expected, atol=1e-12)

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, 'wild-type length'):
            profile_features('ACDEF', ['A1W'], ['WCDEF'], [0.0], self.frequencies, self.background,
                             self.record, alpha=PSEUDOCOUNT_ALPHA)
        with self.assertRaisesRegex(ValueError, 'positive frequencies'):
            profile_features(self.wildtype, ['A1W'], ['WCDE'], [0.0], self.frequencies,
                             np.zeros(20), self.record, alpha=PSEUDOCOUNT_ALPHA)
        with self.assertRaisesRegex(ValueError, 'pseudocount'):
            profile_features(self.wildtype, ['A1W'], ['WCDE'], [0.0], self.frequencies,
                             self.background, self.record, alpha=0.0)
        with self.assertRaisesRegex(ValueError, 'unaligned profile inputs'):
            profile_features(self.wildtype, ['A1W'], ['WCDE'], [0.0, 1.0], self.frequencies,
                             self.background, self.record, alpha=PSEUDOCOUNT_ALPHA)


class TokenisationBlockTests(unittest.TestCase):
    def test_residue_tokenizer_has_no_segmentation_degree_of_freedom(self):
        wildtype = 'ACDE'
        wild_tokens = [10, 11, 12, 13]
        mutant_tokens = [[99, 11, 12, 13], [10, 11, 99, 13]]
        rows = tokenisation_features(wild_tokens, mutant_tokens, wildtype, ['WCDE', 'ACWE'], budget=1024)
        self.assertEqual(rows.shape, (2, len(TOKENISATION_FEATURE_ORDER)))
        order = {name: index for index, name in enumerate(TOKENISATION_FEATURE_ORDER)}
        np.testing.assert_allclose(rows[:, order['pooled_token_count_cycle']], [0.0, 0.0])
        np.testing.assert_allclose(rows[:, order['segmentation_changed_tokens']], [2.0, 2.0])
        np.testing.assert_allclose(rows[:, order['common_prefix_token_fraction']], [0.0, 0.5])
        np.testing.assert_allclose(rows[:, order['common_suffix_token_fraction']], [0.75, 0.25])
        np.testing.assert_allclose(rows[:, order['wildtype_tokens_per_residue']], [1.0, 1.0])
        np.testing.assert_allclose(rows[:, order['mutant_tokens_per_residue']], [1.0, 1.0])
        np.testing.assert_allclose(rows[:, order['wildtype_pooled_tokens_over_budget']],
                                   [4 / 1024, 4 / 1024])

    def test_multi_residue_segmentation_change_is_visible(self):
        order = {name: index for index, name in enumerate(TOKENISATION_FEATURE_ORDER)}
        rows = tokenisation_features([1, 2, 3], [[1, 7, 8, 3]], 'ACDEFG', ['ACWEFG'], budget=1024)
        self.assertAlmostEqual(rows[0, order['pooled_token_count_cycle']], 1.0)
        self.assertAlmostEqual(rows[0, order['common_prefix_token_fraction']], 1 / 3)
        self.assertAlmostEqual(rows[0, order['common_suffix_token_fraction']], 1 / 3)
        self.assertAlmostEqual(rows[0, order['segmentation_changed_tokens']], 3.0)
        self.assertAlmostEqual(rows[0, order['wildtype_tokens_per_residue']], 0.5)
        self.assertAlmostEqual(rows[0, order['mutant_tokens_per_residue']], 4 / 6)

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, 'unaligned mutant token spans'):
            tokenisation_features([1, 2], [[1, 2]], 'AC', ['AC', 'AC'], budget=1024)
        with self.assertRaisesRegex(ValueError, 'wild-type pooled token span is empty'):
            tokenisation_features([], [[1]], 'A', ['C'], budget=1024)
        with self.assertRaisesRegex(ValueError, 'mutant pooled token span is empty'):
            tokenisation_features([1], [[]], 'A', ['C'], budget=1024)
        with self.assertRaisesRegex(ValueError, 'token budget'):
            tokenisation_features([1], [[1]], 'A', ['C'], budget=0)


class IncrementBookkeepingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_increments_are_paired_differences_of_the_reported_design_correlations(self):
        rows = synthetic_panel(seed=4, n_assays=10, n_variants=20)
        report, _ = evaluate_crossed_controls(rows, bootstrap=32, seed=13, fold_seed=17)
        for row, control in itertools.product(report['assays'], CONTROL_SETS):
            for label, left, right in (('M', control + '+M', control), ('R', control + '+R', control),
                                       ('R_after_M', control + '+M+R', control + '+M')):
                self.assertAlmostEqual(row[f'increment_{label}_{control}'],
                                       row[left + '_spearman'] - row[right + '_spearman'], places=12)
                self.assertAlmostEqual(row[f'rank_mse_reduction_{label}_{control}'],
                                       row[right + '_rank_mse'] - row[left + '_rank_mse'], places=12)
        for control in CONTROL_SETS:
            summary = report['summaries'][f'increment_R_{control}']
            self.assertEqual(summary['unit'], 'wild-type family at 50% identity')
            self.assertEqual(summary['resamples'], 32)
            self.assertEqual(summary['n_assays'], len(rows))
        # The planted signal is carried by the representation block, so its
        # increment over the composition control is positive on this panel.
        self.assertGreater(report['summaries']['increment_R_C']['point'], 0.0)


def minimal_report(arm: str, fold_seed: int, *, digest: str = 'a' * 64) -> dict:
    """The fields the cross-cell admission actually reads, and nothing else."""
    return dict(
        schema_version='d1_crossed_controls_v1', status='complete', arm=arm, fold_seed=fold_seed,
        cohort_sha256='c' * 64, control_sets={name: list(columns) for name, columns in CONTROL_SETS.items()},
        feature_order=dict(composition='unchanged'), n_assays=12, n_families=12, n_variants=288,
        support_definition='common', anchor_arm='anchor', manifest_arms=[arm, 'anchor'],
        support=dict(assay_ids=[f'assay{i:02d}' for i in range(12)]),
        feature_dimensions={name + addition: 1 for name in CONTROL_SETS for addition in ADDITIONS},
        prediction_digests={**{name + addition: 'b' * 64 for name in CONTROL_SETS for addition in ADDITIONS},
                            **{name: digest for name in cli.MODEL_INDEPENDENT}},
        folds=[dict(fold=0, held_families=[1, 2], training_families=[3, 4],
                    inner_folds=[dict(validation_families=[3]), dict(validation_families=[4])])],
        summaries={'increment_R_C': dict(point=0.1, interval=[0.0, 0.2])},
        tokenisation_interface=dict(stratum='amino-acid', tokenisation='residue',
                                   measured_segmentation='one_token_per_residue', assays=[{'assay': 'x'}]))


class CrossCellAdmissionTests(unittest.TestCase):
    def run_collect(self, reports: list[dict], directory: Path):
        paths = []
        for index, payload in enumerate(reports):
            path = directory / f'report{index}.json'
            path.write_text(json.dumps(payload))
            paths.append(path)
        out = directory / 'admission'
        cli.collect(argparse.Namespace(collect=paths, out=out))
        return json.loads((out / 'crossed_controls_admission.json').read_text())

    def test_matching_cells_are_admitted_and_report_their_shared_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            admission = self.run_collect(
                [minimal_report('armA', 20260923), minimal_report('armB', 20260923),
                 minimal_report('armA', 20260924, digest='d' * 64),
                 minimal_report('armB', 20260924, digest='d' * 64)], Path(directory))
        self.assertEqual(admission['status'], 'admitted')
        self.assertEqual(admission['fold_seeds'], [20260923, 20260924])
        self.assertEqual(len(admission['cells']), 4)
        self.assertEqual(admission['support']['n_variants'], 288)
        self.assertEqual(set(admission['model_independent_designs']), set(cli.MODEL_INDEPENDENT))
        # Cells of different arms carry different manifest sets, so the shared
        # identity is the anchor arm and the realised support, not the set.
        self.assertEqual(admission['anchor_arm'], 'anchor')
        self.assertEqual(admission['manifest_arms']['armA'], ['armA', 'anchor'])

    def test_divergent_support_folds_predictions_or_status_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            base = minimal_report('armA', 20260923)
            other = minimal_report('armB', 20260923)
            shifted = copy.deepcopy(other)
            shifted['support']['assay_ids'] = shifted['support']['assay_ids'][:-1]
            with self.assertRaisesRegex(ValueError, 'assay support differs'):
                self.run_collect([base, shifted], Path(directory))
            refolded = copy.deepcopy(other)
            refolded['folds'][0]['held_families'] = [5, 6]
            with self.assertRaisesRegex(ValueError, 'fold membership differs'):
                self.run_collect([base, refolded], Path(directory))
            drifted = minimal_report('armB', 20260923, digest='e' * 64)
            with self.assertRaisesRegex(ValueError, 'model-independent design'):
                self.run_collect([base, drifted], Path(directory))
            smoke = copy.deepcopy(other)
            smoke['status'] = 'smoke'
            with self.assertRaisesRegex(ValueError, 'not a complete crossed-controls report'):
                self.run_collect([base, smoke], Path(directory))
            duplicate = copy.deepcopy(base)
            with self.assertRaisesRegex(ValueError, 'duplicate arm and split seed'):
                self.run_collect([base, duplicate], Path(directory))
            relabelled = copy.deepcopy(other)
            relabelled['cohort_sha256'] = 'f' * 64
            with self.assertRaisesRegex(ValueError, 'cohort_sha256 differs'):
                self.run_collect([base, relabelled], Path(directory))
            reanchored = copy.deepcopy(other)
            reanchored['anchor_arm'] = 'other-anchor'
            with self.assertRaisesRegex(ValueError, 'anchor_arm differs'):
                self.run_collect([base, reanchored], Path(directory))


class AnchorPanelTests(unittest.TestCase):
    """A cell may load its own manifest plus the anchor's, never a smaller panel."""

    def manifests(self, directory: Path, rosters: dict[str, list[str]]) -> tuple[list[Path], dict]:
        paths, hashes = [], {}
        for arm, assays in rosters.items():
            payload = dict(identity=dict(arm=arm),
                           assays=[dict(assay=assay, max_packed_tokens=100 + index)
                                   for index, assay in enumerate(assays)])
            path = directory / f'manifest_{arm}.json'
            raw = json.dumps(payload).encode()
            path.write_bytes(raw)
            paths.append(path)
            hashes[str(path)] = cli.sha256_bytes(raw)
        return paths, hashes

    def test_anchor_support_is_required_and_token_maxima_are_carried(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            paths, hashes = self.manifests(base, {'arm': ['a1', 'a2', 'a3'], 'anchor': ['a1', 'a2']})
            rows = [dict(assay='a1'), dict(assay='a2')]
            arms = cli.attach_manifest_tokens(rows, paths, 'arm', 'anchor', hashes, ['a1', 'a2'])
            self.assertEqual(sorted(arms), ['anchor', 'arm'])
            self.assertEqual([row['max_packed_tokens'] for row in rows], [100, 101])
            # A realised support smaller than the anchor is refused, not fitted.
            with self.assertRaisesRegex(ValueError, 'is not the 2-assay anchor'):
                cli.attach_manifest_tokens([dict(assay='a1')], paths, 'arm', 'anchor', hashes, ['a1'])
            # The interface check alone may run on a reduced support.
            self.assertEqual(sorted(cli.attach_manifest_tokens(
                [dict(assay='a1')], paths, 'arm', 'anchor', hashes, None)), ['anchor', 'arm'])

    def test_the_anchor_arm_needs_only_its_own_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, hashes = self.manifests(Path(directory), {'anchor': ['a1', 'a2']})
            rows = [dict(assay='a1'), dict(assay='a2')]
            self.assertEqual(cli.attach_manifest_tokens(rows, paths, 'anchor', 'anchor', hashes,
                                                        ['a1', 'a2']), ['anchor'])
            self.assertEqual([row['max_packed_tokens'] for row in rows], [100, 101])

    def test_missing_arm_missing_anchor_and_changed_bytes_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            paths, hashes = self.manifests(base, {'arm': ['a1'], 'anchor': ['a1']})
            with self.assertRaisesRegex(ValueError, 'no manifest declares arm absent'):
                cli.attach_manifest_tokens([dict(assay='a1')], paths, 'absent', 'anchor', hashes, ['a1'])
            with self.assertRaisesRegex(ValueError, 'no manifest declares the anchor arm absent'):
                cli.attach_manifest_tokens([dict(assay='a1')], paths, 'arm', 'absent', hashes, ['a1'])
            paths[0].write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, 'manifest bytes changed'):
                cli.attach_manifest_tokens([dict(assay='a1')], paths, 'arm', 'anchor', hashes, ['a1'])


if __name__ == '__main__':
    unittest.main()
