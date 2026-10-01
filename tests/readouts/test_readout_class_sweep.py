"""Invariants the readout-class sweep must hold against its admitted baseline.

The sweep is only interpretable if it differs from the admitted readout study in
the readout class and in nothing else. These tests pin the three properties that
carry that claim: the new linear solver is the audited solver, the fold maps and
tie rule are the audited ones, and no held-out cluster's labels or rows reach a
fit, its feature scaling or its capacity choice. They also pin that the matched
baseline of every class is genuinely the same fit, and that the loader refuses an
artifact whose identity does not bind.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from threadpoolctl import threadpool_limits

from src.capability.readouts import readout_analysis as audited
from src.capability.readouts.readout_class_sweep import (BANDWIDTHS, CLASSES, DEPTH_CLASSES, EXTENDED_ALPHAS,
                                              FEATURE_NAMES, PERMUTATION_OFFSET,
                                              PROJECTION_BLAS_THREADS, PROJECTION_SEED,
                                              RANDOM_FEATURES, REPRODUCTION_CLASS, Design,
                                              DesignBuilder, blas_thread_counts,
                                              evaluate_class_sweep, load_panel, nested_predict,
                                              projected_block, projection_matrices,
                                              random_feature_map, raw_blocks, required_blocks,
                                              target_vector, weighted_moments,
                                              weighted_ridge_predict)


#: Random features are a declared 2,048 in production; the tests patch the count
#: down because their panels are tiny and the invariants do not depend on it.
TEST_FEATURES = 64


def synthetic_panel(rng, assays=30, variants=8, width=6):
    """A small cohort-shaped panel: three assays per cluster, unique canonical mutations."""
    letters = 'MKVLTGIVCDEFHNPQRSWY'
    wildtype = (letters * (variants // len(letters) + 2))[:variants + 2]
    rows = []
    for index in range(assays):
        mutants = [f'{wildtype[i]}{i + 1}A' for i in range(variants)]
        features = rng.normal(size=(len(mutants), 4, width)).astype(np.float32)
        compression = np.random.default_rng(99).normal(size=(4 * width, 4)).astype(np.float32)
        rows.append(dict(assay=f'assay{index}', cluster=index // 3, mutants=mutants,
                         measured=rng.normal(size=len(mutants)),
                         P=rng.normal(size=len(mutants)), M=rng.normal(size=len(mutants)),
                         full=features,
                         projected=features.reshape(len(mutants), -1) @ compression,
                         S=rng.normal(size=(len(mutants), 3))))
    return rows


class SolverTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_cholesky_solver_reproduces_the_audited_eigendecomposition(self):
        rng = np.random.default_rng(11)
        for rows, columns in ((60, 4), (40, 25), (25, 40)):
            x = rng.normal(size=(rows, columns))
            x[:, 0] = 3.0                      # a constant column takes scale one in both paths
            x[:, -1] = 1e4 + rng.normal(size=rows) * 1e-3   # a large offset stresses centring
            y = rng.normal(size=rows)
            weights = rng.uniform(0.1, 2.0, rows)
            test = rng.normal(size=(9, columns))
            expected = audited.ridge_predict(x, y, weights, test, audited.ALPHAS)
            observed = weighted_ridge_predict(x, y, weights, test, audited.ALPHAS)
            np.testing.assert_allclose(observed, expected, atol=1e-8, rtol=1e-8)

    def test_a_nonpositive_penalty_is_refused(self):
        rng = np.random.default_rng(12)
        x, y = rng.normal(size=(20, 3)), rng.normal(size=20)
        with self.assertRaises(ValueError):
            weighted_ridge_predict(x, y, np.ones(20), x, [0.0])


class FoldAndLeakageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.rng = np.random.default_rng(13)
        self.rows = synthetic_panel(self.rng)
        self.blocks = raw_blocks([dict(r) for r in self.rows], {'B', 'R_proj'})
        self.assays = np.concatenate([[r['assay']] * len(r['mutants']) for r in self.rows])
        self.families = np.concatenate([[r['cluster']] * len(r['mutants']) for r in self.rows])
        self.measured = np.concatenate([r['measured'] for r in self.rows])

    def test_linear_class_reproduces_the_audited_nested_predictor(self):
        builder = DesignBuilder(self.blocks)
        design = Design('B_R', ('B', 'R_proj'))
        x = builder.linear(design.linear)
        target = target_vector(self.measured, self.assays)
        mine, mine_folds = nested_predict(builder, design, target, self.assays, self.families,
                                          fold_seed=PROJECTION_SEED, alphas=audited.ALPHAS,
                                          bandwidths=(None,))
        theirs, their_folds = audited.nested_predict(x, self.measured, self.assays, self.families,
                                                    seed=PROJECTION_SEED)
        np.testing.assert_allclose(mine, theirs, atol=1e-8, rtol=1e-8)
        for left, right in zip(mine_folds, their_folds):
            self.assertEqual(left['held_families'], right['held_families'])
            self.assertEqual(left['alpha'], right['alpha'])
            self.assertEqual([f['validation_families'] for f in left['inner_folds']],
                             [f['validation_families'] for f in right['inner_folds']])
            np.testing.assert_allclose([f['rank_mse'][0] for f in left['inner_folds']],
                                       [f['rank_mse'] for f in right['inner_folds']], atol=1e-8)

    @mock.patch('src.capability.readouts.readout_class_sweep.RANDOM_FEATURES', TEST_FEATURES)
    def test_a_held_cluster_label_cannot_change_that_cluster_prediction(self):
        for design in (Design('B_R', ('B', 'R_proj')), Design('B_R', ('B', 'R_proj'), ('B', 'R_proj'))):
            builder = DesignBuilder(self.blocks)
            target = target_vector(self.measured, self.assays)
            first, folds = nested_predict(builder, design, target, self.assays, self.families,
                                          fold_seed=PROJECTION_SEED, alphas=EXTENDED_ALPHAS,
                                          bandwidths=BANDWIDTHS if design.mapped else (None,))
            held = np.isin(self.families, folds[0]['held_families'])
            changed = self.measured.copy()
            changed[held] = self.rng.normal(size=int(held.sum())) * 5.0
            second, again = nested_predict(builder, design, target_vector(changed, self.assays),
                                           self.assays, self.families, fold_seed=PROJECTION_SEED,
                                           alphas=EXTENDED_ALPHAS,
                                           bandwidths=BANDWIDTHS if design.mapped else (None,))
            np.testing.assert_allclose(first[held], second[held], atol=1e-12)
            self.assertEqual(folds[0]['alpha'], again[0]['alpha'])
            self.assertEqual(folds[0]['bandwidth'], again[0]['bandwidth'])

    def test_mapped_feature_scaling_reads_only_the_fitting_rows(self):
        rng = np.random.default_rng(14)
        block = rng.normal(size=(40, 5))
        fit = np.arange(0, 20)
        weights = rng.uniform(0.2, 1.5, len(fit))
        mean, scale = weighted_moments(block, fit, weights)
        normalized = weights / weights.sum()
        np.testing.assert_allclose(mean, normalized @ block[fit], atol=1e-12)
        np.testing.assert_allclose(scale, np.sqrt(normalized @ (block[fit] - mean) ** 2), atol=1e-12)
        moved = block.copy()
        moved[20:] = 1e6
        again = weighted_moments(moved, fit, weights)
        np.testing.assert_allclose(again[0], mean, atol=1e-12)
        np.testing.assert_allclose(again[1], scale, atol=1e-12)

    def test_random_feature_map_is_fixed_and_group_normalized(self):
        first, phases = random_feature_map((6, 10))
        second, again = random_feature_map((6, 10))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(phases, again)
        self.assertEqual(first.shape, (16, RANDOM_FEATURES))
        self.assertEqual(RANDOM_FEATURES, 2048)
        self.assertAlmostEqual(float(first[:6].std() * np.sqrt(6)), 1.0, delta=0.05)
        self.assertAlmostEqual(float(first[6:].std() * np.sqrt(10)), 1.0, delta=0.05)


class DeclarationTests(unittest.TestCase):
    def test_the_reproduction_class_carries_the_admitted_grid_and_features(self):
        spec = CLASSES[REPRODUCTION_CLASS]
        self.assertEqual(tuple(spec['alphas']), audited.ALPHAS)
        self.assertEqual(tuple(spec['bandwidths']), (None,))
        self.assertEqual(set(EXTENDED_ALPHAS) & set(audited.ALPHAS), set(audited.ALPHAS))

    def test_every_class_matches_its_baseline_on_the_same_columns(self):
        for name, spec in {**CLASSES, **DEPTH_CLASSES}.items():
            designs = {d.name: d for d in spec['designs']}
            self.assertEqual(set(designs), {'B', 'R', 'B_R'}, name)
            self.assertEqual(designs['B'].linear, ('B',), name)
            self.assertEqual(designs['B_R'].linear, designs['B'].linear + designs['R'].linear, name)
            if designs['B_R'].mapped:
                self.assertEqual(designs['B_R'].mapped,
                                 designs['B'].mapped + designs['R'].mapped, name)
            self.assertNotIn('B', designs['R'].linear + designs['R'].mapped, name)

    def test_the_baseline_fit_is_shared_across_classes_with_the_same_grid(self):
        keys = {name: spec['designs'][0].key(tuple(spec['alphas']), tuple(spec['bandwidths']))
                for name, spec in CLASSES.items()}
        self.assertEqual(keys['C1_compressed_linear'], keys['C2_full_linear'])
        self.assertEqual(keys['C3_compressed_random_feature'], keys['C4_full_random_feature'])
        self.assertNotEqual(keys[REPRODUCTION_CLASS], keys['C1_compressed_linear'])

    def test_depth_classes_cover_exactly_the_retained_blocks(self):
        self.assertEqual(sorted(DEPTH_CLASSES), sorted(f'D_{b}' for b in FEATURE_NAMES))
        self.assertEqual(required_blocks(DEPTH_CLASSES), {'B', *FEATURE_NAMES})


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    @mock.patch('src.capability.readouts.readout_class_sweep.RANDOM_FEATURES', TEST_FEATURES)
    def test_sweep_reports_every_declared_endpoint_on_one_support(self):
        rng = np.random.default_rng(15)
        rows = synthetic_panel(rng, assays=30, variants=8, width=5)
        signal = np.concatenate([r['full'][:, 0, 0] for r in rows]).astype(float)
        offset = 0
        for r in rows:
            n = len(r['mutants'])
            r['measured'] = signal[offset:offset + n] + rng.normal(size=n) * 0.1
            offset += n
        specs = {REPRODUCTION_CLASS: CLASSES[REPRODUCTION_CLASS],
                 'C2_full_linear': CLASSES['C2_full_linear'],
                 'C3_compressed_random_feature': CLASSES['C3_compressed_random_feature']}
        report = evaluate_class_sweep(rows, class_specs=specs, fold_seed=PROJECTION_SEED,
                                      bootstrap=64, shuffle_classes=('C2_full_linear',))
        json.dumps(report, allow_nan=False)
        self.assertEqual(report['permutation_seed'], PROJECTION_SEED + PERMUTATION_OFFSET)
        for name in specs:
            for metric in (f'{name}/delta_spearman', f'{name}/delta_rank_mse',
                           f'{name}/R_minus_raw_M_spearman'):
                self.assertIn(metric, report['summaries'], metric)
                self.assertEqual(report['summaries'][metric]['unit'],
                                 'wild-type family at 50% identity')
        self.assertIn('C2_full_linear/permuted_delta_spearman', report['summaries'])
        self.assertNotIn('C3_compressed_random_feature/permuted_delta_spearman', report['summaries'])
        # The recovered signal lives in the uncompressed block, so the class that
        # sees it must not score below the one that only sees a random projection.
        self.assertGreater(report['summaries']['C2_full_linear/R_spearman']['point'],
                           report['summaries'][f'{REPRODUCTION_CLASS}/R_spearman']['point'])


class ProjectionOrderTests(unittest.TestCase):
    """The compressed block must not depend on the caller's BLAS thread count.

    The admitted per-variant predictions are bound at 1e-8 while the float32
    product that forms the compressed block is only reproducible to about 1e-6
    across thread counts, so the pinned reduction order is a condition of the
    reproduction rather than a performance setting.
    """

    def test_the_compressed_block_is_identical_under_any_ambient_thread_count(self):
        rng = np.random.default_rng(23)
        hidden = rng.normal(size=(96, 4, 512)).astype(np.float32)
        projection = projection_matrices(512)
        with threadpool_limits(limits=PROJECTION_BLAS_THREADS, user_api='blas'):
            reference = np.concatenate([hidden[:, i] @ projection[i] for i in range(4)], axis=1)
        for ambient in (1, 2, PROJECTION_BLAS_THREADS, 8, 48):
            with threadpool_limits(limits=ambient, user_api='blas'):
                observed = projected_block(hidden, projection)
            self.assertTrue(np.array_equal(observed, reference),
                            f'the compressed block moved at {ambient} ambient BLAS threads')

    def test_the_pin_is_released_after_the_product(self):
        hidden = np.zeros((4, 4, 8), dtype=np.float32)
        with threadpool_limits(limits=16, user_api='blas'):
            projected_block(hidden, projection_matrices(8))
            self.assertEqual(set(blas_thread_counts()), {16})

    def test_a_process_without_a_blas_pool_is_refused(self):
        hidden = np.zeros((4, 4, 8), dtype=np.float32)
        with mock.patch('src.capability.readouts.readout_class_sweep.threadpool_info', return_value=[]):
            with self.assertRaises(ValueError):
                projected_block(hidden, projection_matrices(8))


class LoaderTests(unittest.TestCase):
    def _fixture(self, directory, *, break_digest=False, break_cluster=False):
        rng = np.random.default_rng(16)
        wildtype = 'MKVLAAGIVGT'
        mutants = ['M1K', 'K2R', 'V3L', 'L4V']
        cohort = dict(assays=[dict(assay='a1', cluster=7, wildtype_id='wt1', wildtype=wildtype,
                                   mutants=mutants, sequences=[wildtype] * 4,
                                   measured=[0.1, -0.2, 0.3, 0.4],
                                   profile_scores=[1.0, 2.0, 3.0, 4.0],
                                   mutant_digest=hashlib.sha256('\n'.join(mutants).encode()).hexdigest())])
        cohort_path = directory / 'cohort.json'
        cohort_path.write_text(json.dumps(cohort))
        identity = dict(arm='arm1', feature_names=list(FEATURE_NAMES),
                        cohort_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest())
        row = cohort['assays'][0]
        npz = directory / 'arm1_a1.npz'
        np.savez_compressed(npz, features=rng.normal(size=(4, 4, 5)).astype(np.float32),
                            likelihood=np.arange(4.0), wt_features=rng.normal(size=(4, 5)),
                            wt_likelihood=0.0, mutants=np.asarray(mutants),
                            measured=np.asarray(row['measured']),
                            profile_scores=np.asarray(row['profile_scores']),
                            metadata=json.dumps(dict(identity=identity, assay='a1',
                                                     mutant_digest=row['mutant_digest'])))
        record = dict(assay='a1', file=npz.name, cluster=99 if break_cluster else 7,
                      wildtype_id='wt1', mutant_digest=row['mutant_digest'], variants=4,
                      sha256='0' * 64 if break_digest
                      else hashlib.sha256(npz.read_bytes()).hexdigest())
        manifest_path = directory / 'manifest_arm1.json'
        manifest_path.write_text(json.dumps(dict(status='complete', identity=identity,
                                                 assays=[record], block_indices=[1, 3])))
        return cohort_path, manifest_path

    def test_loader_accepts_a_bound_artifact_and_refuses_a_broken_one(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            cohort_path, manifest_path = self._fixture(directory)
            rows, provenance = load_panel(cohort_path, manifest_path, 'arm1', ['a1'])
            self.assertEqual(provenance['n_variants'], 4)
            self.assertEqual(provenance['hidden_width'], 5)
            self.assertEqual(rows[0]['cluster'], 7)
            with self.assertRaises(ValueError):
                load_panel(cohort_path, manifest_path, 'arm1', ['a1', 'missing'])
        with tempfile.TemporaryDirectory() as name:
            cohort_path, manifest_path = self._fixture(Path(name), break_digest=True)
            with self.assertRaises(ValueError):
                load_panel(cohort_path, manifest_path, 'arm1', ['a1'])
        with tempfile.TemporaryDirectory() as name:
            cohort_path, manifest_path = self._fixture(Path(name), break_cluster=True)
            with self.assertRaises(ValueError):
                load_panel(cohort_path, manifest_path, 'arm1', ['a1'])


if __name__ == '__main__':
    unittest.main()
