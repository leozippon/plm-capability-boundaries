"""The depth-resolved readout must reproduce the admitted two-depth fit exactly.

The experiment this covers exists to lift one bound on the admitted readout
panel: that extraction hooked two blocks per arm. A depth sweep is only worth
reading if the pipeline that produces it recovers the admitted numbers at the
admitted depths on the admitted folds, so the binding test here builds one
synthetic cohort, writes the same underlying states in both the admitted and
the depth-resolved layout, produces an admitted-format report through the
admitted analysis code, and requires the depth pipeline's refit to agree with
it to the tolerance the admitted final admission receipt used. The remaining
tests cover the conditions that must always hold -- identical supports, folds
and seeds; no held-out label in any fit, scaling or tuning step; a position
summary only where the rendering defines one -- and the refusals.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from threadpoolctl import threadpool_limits
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.capability.readouts import summarise_depth_panel

from src.capability.readouts import readout_depth as rd
from src.capability.readouts.readout_analysis import ALPHAS, evaluate_readouts, family_folds
from src.capability.readouts.readout_class_sweep import PROJECTION_BLAS_THREADS
from src.capability.context.profile_increment import standardized_rank

AA = 'ACDEFGHIKLMNPQRSTVWY'
DEPTH = 6
WIDTH = 12
VARIANTS = 9
ASSAYS = 12
CLUSTERS = 9


def synthetic_cohort(rng):
    """Twelve assays over nine wild-type clusters, with single substitutions."""
    assays = []
    for index in range(ASSAYS):
        length = 40 + index
        wildtype = ''.join(rng.choice(list(AA), size=length))
        positions = rng.choice(np.arange(1, length - 1), size=VARIANTS, replace=False)
        mutants, sequences = [], []
        for position in sorted(positions):
            before = wildtype[position]
            after = next(a for a in AA if a != before)
            mutants.append(f'{before}{position + 1}{after}')
            sequences.append(wildtype[:position] + after + wildtype[position + 1:])
        assays.append(dict(
            assay=f'ASSAY_{index:02d}', cluster=index % CLUSTERS,
            wildtype_id=f'WT_{index % CLUSTERS}', wildtype=wildtype,
            mutants=mutants, sequences=sequences,
            measured=list(rng.normal(size=VARIANTS)),
            profile_scores=list(rng.normal(size=VARIANTS)),
            mutant_digest=hashlib.sha256('\n'.join(mutants).encode()).hexdigest()))
    return dict(assays=assays)


def write_artifacts(directory: Path, cohort, rng):
    """One set of states, written in the admitted layout and the depth layout."""
    directory.mkdir(parents=True, exist_ok=True)
    cohort_path = directory / 'cohort.json'
    cohort_path.write_text(json.dumps(cohort))
    cohort_sha = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
    middle, final = rd.admitted_block_indices(DEPTH)
    admitted_identity = dict(arm='synthetic', cohort_sha256=cohort_sha,
                             feature_names=list(rd.ADMITTED_FEATURE_NAMES),
                             max_score_drift=0.001, max_feature_drift=0.001)
    depth_identity = dict(schema_version=rd.EXTRACTION_SCHEMA, arm='synthetic',
                          cohort_sha256=cohort_sha,
                          summary_names=list(rd.POOLED_SUMMARIES),
                          position_summary_names=list(rd.POSITION_SUMMARIES),
                          position_resolved=True, smoke_variants=0, assay_limit=0,
                          budget=1024, batch_size=1,
                          max_score_drift=0.001, max_feature_drift=0.001)
    admitted_records, depth_records = [], []
    for row in cohort['assays']:
        pooled = rng.normal(size=(VARIANTS, DEPTH, 2, WIDTH)).astype(np.float32)
        position = rng.normal(size=(VARIANTS, DEPTH, 2, WIDTH)).astype(np.float32)
        likelihood = rng.normal(size=VARIANTS)
        admitted_features = np.stack([pooled[:, middle, 0], pooled[:, middle, 1],
                                      pooled[:, final, 0], pooled[:, final, 1]], axis=1)
        admitted_name = f'admitted_{row["assay"]}.npz'
        depth_name = f'depth_{row["assay"]}.npz'
        np.savez(directory / admitted_name, features=admitted_features,
                 likelihood=likelihood, wt_features=np.zeros((4, WIDTH), np.float32),
                 wt_likelihood=np.asarray(0.0), mutants=np.asarray(row['mutants']),
                 measured=np.asarray(row['measured']),
                 profile_scores=np.asarray(row['profile_scores']),
                 metadata=json.dumps(dict(identity=admitted_identity, assay=row['assay'],
                                          mutant_digest=row['mutant_digest'])))
        payload = dict(likelihood=likelihood, wt_likelihood=np.asarray(0.0),
                       mutants=np.asarray(row['mutants']),
                       measured=np.asarray(row['measured']),
                       profile_scores=np.asarray(row['profile_scores']),
                       metadata=json.dumps(dict(identity=depth_identity, assay=row['assay'],
                                                mutant_digest=row['mutant_digest'])),
                       batch_check_likelihood_delta_nats=np.asarray(0.0),
                       batch_check_feature_relative_l2=np.asarray(0.0),
                       batch_check_mutation_feature_relative_l2=np.asarray(0.0),
                       prefix_abs_max=np.zeros((VARIANTS, DEPTH), np.float32),
                       empty_suffix=np.zeros(VARIANTS, bool),
                       n_mutated=np.ones(VARIANTS, np.int32))
        for index in range(DEPTH):
            payload[f'pooled_d{index:03d}'] = pooled[:, index]
            payload[f'wt_pooled_d{index:03d}'] = np.zeros((2, WIDTH), np.float32)
            payload[f'position_d{index:03d}'] = position[:, index]
        np.savez(directory / depth_name, **payload)
        for name, records in ((admitted_name, admitted_records), (depth_name, depth_records)):
            records.append(dict(assay=row['assay'], file=name,
                                sha256=hashlib.sha256((directory / name).read_bytes()).hexdigest(),
                                cluster=row['cluster'], wildtype_id=row['wildtype_id'],
                                max_packed_tokens=len(row['wildtype']) + 2,
                                mutant_digest=row['mutant_digest'], variants=VARIANTS,
                                bytes=(directory / name).stat().st_size))
    admitted_manifest = directory / 'manifest_synthetic.json'
    admitted_manifest.write_text(json.dumps(dict(
        identity=admitted_identity, status='complete', assays=admitted_records, skipped=[],
        block_indices=[middle, final])))
    depth_manifest = directory / 'manifest_depth_synthetic.json'
    depth_manifest.write_text(json.dumps(dict(
        identity=depth_identity, status='complete', assays=depth_records, skipped=[],
        block_count=DEPTH, admitted_block_indices=[middle, final], batch_size=1)))
    return cohort_path, admitted_manifest, depth_manifest


class DepthContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile

        cls._directory = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(11)
        cls.cohort = synthetic_cohort(rng)
        cls.paths = write_artifacts(Path(cls._directory.name), cls.cohort, rng)
        cls.assay_ids = [row['assay'] for row in cls.cohort['assays']]

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def panel(self):
        cohort_path, _, depth_manifest = self.paths
        return rd.DepthPanel(cohort_path, depth_manifest, 'synthetic', self.assay_ids)

    def admitted(self):
        cohort_path, admitted_manifest, _ = self.paths
        return rd.load_admitted_arrays(cohort_path, admitted_manifest, 'synthetic', self.assay_ids)

    @staticmethod
    def with_representation(rows, blocks):
        """The rows the admitted analysis fits: its own projected representation."""
        design = rd.admitted_design(blocks, [len(row['mutants']) for row in rows])
        out, start = [], 0
        for row in rows:
            stop = start + len(row['mutants'])
            out.append(dict(row, R=design[start:stop]))
            start = stop
        return out

    # ---------------------------------------------------------- the binding control

    def test_the_depth_pipeline_reproduces_the_admitted_two_depth_fit(self):
        """The reproduction control must actually bind: same folds, same penalties,
        same held-out predictions as the admitted analysis code produces."""
        rows, blocks, _, _ = self.admitted()
        admitted = evaluate_readouts(self.with_representation(rows, blocks),
                                     seed=rd.BOOTSTRAP_SEED,
                                     fold_seed=rd.PRIMARY_SPLIT_SEED, bootstrap=64,
                                     permutation_control=False)
        admitted['support'] = dict(assay_ids=list(self.assay_ids))
        panel = self.panel()
        self.addCleanup(panel.close)
        # The depth layout must hold the very same admitted four blocks.
        np.testing.assert_array_equal(rd.admitted_blocks_from_depth(panel), blocks)
        baseline = rd.baseline_design(rows)
        representation = rd.admitted_design(rd.admitted_blocks_from_depth(panel),
                                            [len(row['mutants']) for row in rows])
        measured, assays, clusters = rd.row_labels(rows)
        predictions, folds = {}, {}
        for label, design in (('baseline/B', baseline), ('admitted/R', representation),
                              ('admitted/B_R', np.column_stack([baseline, representation]))):
            predictions[label], folds[label] = rd.fit_design(
                design, measured, assays, clusters, fold_seed=rd.PRIMARY_SPLIT_SEED, device='cpu')
        predictions['raw_P'] = baseline[:, 0]
        predictions['raw_M'] = baseline[:, 1]
        records = rd.per_assay_metrics(rows, assays, measured, predictions,
                                       {'admitted': ('admitted/B_R', 'baseline/B')})
        agreement = rd.verify_against_admitted_report(
            admitted, self.assay_ids, rows, records, predictions, folds,
            {'B': 'baseline/B', 'R': 'admitted/R', 'B_R': 'admitted/B_R'})
        for name, deviation in agreement['max_absolute_deviation'].items():
            self.assertLessEqual(deviation, 1e-8, f'{name} deviates by {deviation}')

    def test_the_whole_cell_fits_every_declared_axis_and_binds_to_the_admitted_report(self):
        """End to end through the stage: both reproduction controls pass, every
        declared axis is summarised, and the capacity axes run at the primary seed."""
        from scripts.capability.readouts import analyse_readout_depth as stage
        import tempfile

        if not stage.ADMISSION_RECEIPT.is_file():
            self.skipTest(f'{stage.ADMISSION_RECEIPT} is absent: the admitted admission receipt '
                          'is a retained result, not distributed with the code repository')
        rows, blocks, _, _ = self.admitted()
        panel = self.panel()
        self.addCleanup(panel.close)
        seeds = list(rd.SPLIT_SEEDS[:2])
        reports = {}
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__('shutil').rmtree(directory, ignore_errors=True))
        for seed in seeds:
            admitted = evaluate_readouts(self.with_representation(rows, blocks),
                                         seed=rd.BOOTSTRAP_SEED, fold_seed=seed, bootstrap=16,
                                         permutation_control=False)
            admitted['support'] = dict(assay_ids=list(self.assay_ids))
            path = directory / f'admitted_{seed}.json'
            path.write_text(json.dumps(admitted))
            reports[seed] = path
        out = stage.fit_cell(panel, rows, blocks, reports, seeds=seeds, device='cpu',
                             bootstrap=16, shuffle=True, progress=lambda name: None)
        self.assertEqual(sorted(out), sorted(str(seed) for seed in seeds))
        primary = out[str(rd.PRIMARY_SPLIT_SEED)]
        secondary = out[str(seeds[1])]
        for index in range(DEPTH):
            for payload in (primary, secondary):
                self.assertIn(f'depth{index:03d}_delta_spearman', payload['summaries'])
                self.assertIn(f'pos{index:03d}_delta_spearman', payload['summaries'])
            self.assertNotIn(f'wide{index:03d}_delta_spearman', secondary['summaries'])
        for index in stage.capacity_depths(DEPTH):
            self.assertIn(f'wide{index:03d}_delta_spearman', primary['summaries'])
            self.assertIn(f'full{index:03d}_delta_spearman', primary['summaries'])
        self.assertIn('union_delta_spearman', primary['summaries'])
        self.assertNotIn('union_delta_spearman', secondary['summaries'])
        self.assertIn('permuted/B_spearman', primary['summaries'])
        self.assertNotIn('permuted/B_spearman', secondary['summaries'])
        self.assertEqual(primary['feature_dimensions']['depth000/R'],
                         2 * rd.DEPTH_PROJECTION_DIM)
        self.assertEqual(primary['feature_dimensions']['wide000/R'],
                         2 * rd.WIDE_PROJECTION_DIM)
        self.assertEqual(primary['feature_dimensions']['union/R'],
                         DEPTH * 2 * rd.UNION_PROJECTION_DIM)
        self.assertEqual(primary['feature_dimensions']['admitted/R'],
                         4 * rd.ADMITTED_PROJECTION_DIM)
        for seed, payload in out.items():
            deviations = payload['baseline_agreement']['max_absolute_deviation']
            for name, value in deviations.items():
                self.assertLessEqual(value, 1e-8, f'{seed}/{name} deviates by {value}')
        self.assertTrue(primary['baseline_agreement']['pipeline_identity']['passed'])

    def test_the_reproduction_control_refuses_a_different_support(self):
        rows, _, _, _ = self.admitted()
        admitted = dict(support=dict(assay_ids=self.assay_ids[:-1]), n_assays=len(rows) - 1,
                        n_families=CLUSTERS, n_variants=VARIANTS * (ASSAYS - 1),
                        assays=[], predictions=[], folds={})
        with self.assertRaises(ValueError):
            rd.verify_against_admitted_report(admitted, self.assay_ids, rows, [], {}, {}, {})

    def test_the_reproduction_control_refuses_a_different_fold_map(self):
        rows, blocks, _, _ = self.admitted()
        admitted = evaluate_readouts(self.with_representation(rows, blocks),
                                     seed=rd.BOOTSTRAP_SEED,
                                     fold_seed=rd.PRIMARY_SPLIT_SEED, bootstrap=16,
                                     permutation_control=False)
        admitted['support'] = dict(assay_ids=list(self.assay_ids))
        baseline = rd.baseline_design(rows)
        measured, assays, clusters = rd.row_labels(rows)
        # A different split seed is a different fold map and must be refused.
        prediction, folds = rd.fit_design(baseline, measured, assays, clusters,
                                          fold_seed=rd.SPLIT_SEEDS[1], device='cpu')
        records = rd.per_assay_metrics(rows, assays, measured,
                                       dict(raw_P=baseline[:, 0], raw_M=baseline[:, 1],
                                            **{'baseline/B': prediction}), {})
        with self.assertRaises(ValueError):
            rd.verify_against_admitted_report(admitted, self.assay_ids, rows, records,
                                              {'baseline/B': prediction},
                                              {'baseline/B': folds}, {'B': 'baseline/B'})

    # ------------------------------------------------------- folds, seeds, leakage

    def test_the_fold_map_is_the_admitted_seeded_cluster_partition(self):
        rows, _, _, _ = self.admitted()
        measured, assays, clusters = rd.row_labels(rows)
        _, folds = rd.fit_design(rd.baseline_design(rows), measured, assays, clusters,
                                 fold_seed=rd.PRIMARY_SPLIT_SEED, device='cpu')
        expected = family_folds(clusters, 5, rd.PRIMARY_SPLIT_SEED)
        self.assertEqual([sorted(record['held_families']) for record in folds],
                         [sorted(group) for group in expected])
        for outer, record in enumerate(folds):
            held = set(record['held_families'])
            self.assertFalse(held & set(record['training_families']))
            inner = [sorted(fold['validation_families']) for fold in record['inner_folds']]
            training = np.asarray([c for c in clusters if c not in held])
            self.assertEqual(inner, [sorted(group) for group in family_folds(
                training, 4, rd.PRIMARY_SPLIT_SEED + 100 + outer)])

    def test_a_held_out_cluster_label_reaches_no_fit_scaling_or_tuning_step(self):
        """Altering one cluster's measured effects must leave that cluster's own
        held-out predictions untouched, because nothing about its fold saw them."""
        rows, _, _, _ = self.admitted()
        measured, assays, clusters = rd.row_labels(rows)
        design = rd.baseline_design(rows)
        base, folds = rd.fit_design(design, measured, assays, clusters,
                                    fold_seed=rd.PRIMARY_SPLIT_SEED, device='cpu')
        target_cluster = folds[0]['held_families'][0]
        perturbed = measured.copy()
        index = np.flatnonzero(clusters == target_cluster)
        perturbed[index] = perturbed[index][::-1] * 17.0 + 3.0
        moved, _ = rd.fit_design(design, perturbed, assays, clusters,
                                 fold_seed=rd.PRIMARY_SPLIT_SEED, device='cpu')
        np.testing.assert_allclose(moved[index], base[index], rtol=0, atol=1e-12)
        # The same perturbation does reach other clusters' predictions, so the
        # test above is not vacuously passing on an inert perturbation.
        other = np.flatnonzero(clusters != target_cluster)
        self.assertGreater(float(np.max(np.abs(moved[other] - base[other]))), 1e-8)

    def test_the_resampling_and_penalty_contract_is_the_admitted_one(self):
        self.assertEqual(rd.SPLIT_SEEDS, (20260923, 20260924, 20260925))
        self.assertEqual(rd.PRIMARY_SPLIT_SEED, 20260923)
        self.assertEqual(rd.BOOTSTRAP_DRAWS, 2000)
        self.assertEqual(rd.BOOTSTRAP_SEED, 20260923)
        self.assertEqual(rd.PERMUTATION_OFFSET, 9000)
        self.assertEqual(tuple(ALPHAS), (0.01, 0.1, 1.0, 10.0, 100.0))
        self.assertEqual(rd.ADMITTED_PROJECTION_DIM, 256)
        self.assertEqual(rd.ADMITTED_PROJECTION_SEED, 20260923)
        rows = [dict(assay=f'A{i}', cluster=i, value=float(i)) for i in range(9)]
        summary, sizes = rd.summarise_metrics(
            [dict(assay=r['assay'], cluster=r['cluster'], n_variants=1, value=r['value'])
             for r in rows], bootstrap=32)
        self.assertEqual(sizes, [1] * 9)
        self.assertNotIn('cluster_sizes', summary['value'])
        self.assertEqual(summary['value']['unit'], 'wild-type family at 50% identity')
        self.assertEqual(summary['value']['alpha'], 0.05)

    # ------------------------------------------------------------- declared axes

    def test_every_declared_axis_covers_every_depth_and_is_seed_dependent(self):
        primary = rd.readout_depth_axes = None  # guard against an accidental global
        del primary
        from scripts.capability.readouts import analyse_readout_depth as stage

        repeats = stage.declared_axes(DEPTH, position_resolved=True, primary=True)
        secondary = stage.declared_axes(DEPTH, position_resolved=True, primary=False)
        for index in range(DEPTH):
            self.assertIn(f'depth{index:03d}', secondary)
            self.assertIn(f'pos{index:03d}', secondary)
        # The capacity grid always carries the two admitted blocks and the ends.
        grid = stage.capacity_depths(DEPTH)
        self.assertEqual(grid[0], 0)
        self.assertEqual(grid[-1], DEPTH - 1)
        self.assertLessEqual(set(rd.admitted_block_indices(DEPTH)), set(grid))
        self.assertLessEqual(len(grid), len(stage.CAPACITY_FRACTIONS) + 2)
        for index in grid:
            self.assertIn(f'wide{index:03d}', repeats)
            self.assertIn(f'full{index:03d}', repeats)
        self.assertEqual(sorted(int(name[4:]) for name in repeats if name.startswith('wide')),
                         list(grid))
        self.assertIn('union', repeats)
        self.assertNotIn('union', secondary)
        self.assertEqual(len(repeats['union']['requests']), DEPTH * len(rd.POOLED_SUMMARIES))
        pooled_only = stage.declared_axes(DEPTH, position_resolved=False, primary=True)
        self.assertFalse([name for name in pooled_only if name.startswith(('pos', 'full'))])

    def test_a_projection_depends_only_on_its_declared_seed_width_and_width(self):
        first = rd.gaussian_projection(rd.DEPTH_PROJECTION_SEED, WIDTH, 8)
        again = rd.gaussian_projection(rd.DEPTH_PROJECTION_SEED, WIDTH, 8)
        np.testing.assert_array_equal(first, again)
        other = rd.gaussian_projection(rd.DEPTH_PROJECTION_SEED + 8, WIDTH, 8)
        self.assertGreater(float(np.max(np.abs(first - other))), 0.0)
        self.assertAlmostEqual(float(first.std()), 1 / np.sqrt(8), delta=0.1)

    def test_projected_summaries_follow_the_declared_seed_rule(self):
        panel = self.panel()
        self.addCleanup(panel.close)
        blocks = rd.depth_blocks(panel, 1)
        self.assertEqual(sorted(blocks), sorted(rd.SUMMARY_NAMES))
        design = rd.project_summaries(blocks, ('mean', 'suffix'), width=WIDTH, depth_index=1,
                                      dim=5, base_seed=rd.DEPTH_PROJECTION_SEED)
        expected = np.concatenate([
            panel.block(1, 'mean') @ rd.gaussian_projection(
                rd.DEPTH_PROJECTION_SEED + 8 + rd.SUMMARY_NAMES.index('mean'), WIDTH, 5),
            panel.block(1, 'suffix') @ rd.gaussian_projection(
                rd.DEPTH_PROJECTION_SEED + 8 + rd.SUMMARY_NAMES.index('suffix'), WIDTH, 5)], axis=1)
        np.testing.assert_array_equal(design, expected)
        # A pooled-only arm exposes only the two pooled summaries.
        panel.position_resolved = False
        self.assertEqual(sorted(rd.depth_blocks(panel, 0)), sorted(rd.POOLED_SUMMARIES))
        panel.position_resolved = True

    # ------------------------------------------------- interface and position rules

    def test_admitted_block_indices_follow_the_admitted_rounding_rule(self):
        for depth in range(1, 65):
            middle, final = rd.admitted_block_indices(depth)
            self.assertEqual(final, depth - 1)
            self.assertEqual(middle, (depth - 1) // 2)
        with self.assertRaises(ValueError):
            rd.admitted_block_indices(0)

    def test_position_summaries_localize_on_the_substituted_residues(self):
        delta = torch.zeros(7, 3)
        delta[3] = torch.tensor([1.0, 2.0, 3.0])
        delta[5] = torch.tensor([4.0, 4.0, 4.0])
        mut, suffix, prefix = rd.position_summaries(delta, [3])
        np.testing.assert_allclose(mut, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(suffix, [4.0 / 3, 4.0 / 3, 4.0 / 3])
        self.assertEqual(prefix, 0.0)
        # A substitution at the final residue leaves no suffix at all.
        _, empty, _ = rd.position_summaries(delta, [6])
        np.testing.assert_array_equal(empty, np.zeros(3))
        # A nonzero difference before the first substitution is reported, not hidden.
        delta[1] = torch.tensor([0.0, -9.0, 0.0])
        _, _, seen = rd.position_summaries(delta, [3])
        self.assertEqual(seen, 9.0)
        with self.assertRaises(ValueError):
            rd.position_summaries(delta, [])
        with self.assertRaises(ValueError):
            rd.position_summaries(delta, [2, 2])

    def test_a_multi_residue_rendering_carries_no_position_summary(self):
        panel = self.panel()
        self.addCleanup(panel.close)
        panel.position_resolved = False
        with self.assertRaises(ValueError):
            panel.block(0, 'mut')
        self.assertEqual(panel.prefix_check(), dict(available=False))

    def test_a_stored_mutant_inconsistent_with_its_identifier_is_refused(self):
        wildtype = 'ACDEFGHIK'
        self.assertEqual(rd.verify_mutant_sequence(wildtype, 'D3W', 'ACWEFGHIK'), [2])
        with self.assertRaises(ValueError):
            rd.verify_mutant_sequence(wildtype, 'D3W', 'ACWEFGHIW')
        with self.assertRaises(ValueError):
            rd.verify_mutant_sequence(wildtype, 'A3W', 'ACWEFGHIK')
        with self.assertRaises(ValueError):
            rd.verify_mutant_sequence(wildtype, 'D3W', 'ACWEFGHI')

    # ------------------------------------------------------------------ refusals

    def test_the_panel_refuses_a_checksum_mismatch(self):
        cohort_path, _, depth_manifest = self.paths
        manifest = json.loads(depth_manifest.read_text())
        manifest['assays'][0]['sha256'] = '0' * 64
        broken = depth_manifest.parent / 'broken_manifest.json'
        broken.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            rd.DepthPanel(cohort_path, broken, 'synthetic', self.assay_ids)

    def test_the_panel_refuses_an_assay_the_arm_does_not_cover(self):
        cohort_path, _, depth_manifest = self.paths
        with self.assertRaisesRegex(ValueError, 'does not cover'):
            rd.DepthPanel(cohort_path, depth_manifest, 'synthetic',
                          [*self.assay_ids, 'ASSAY_MISSING'])

    def test_the_panel_refuses_an_admitted_schema_manifest(self):
        cohort_path, admitted_manifest, _ = self.paths
        with self.assertRaisesRegex(ValueError, 'not a depth-resolved extraction'):
            rd.DepthPanel(cohort_path, admitted_manifest, 'synthetic', self.assay_ids)

    def test_block_drift_refuses_a_nonzero_difference_against_a_zero_reference(self):
        observed = np.ones((2, 4, 3), np.float32)
        reference = np.zeros((2, 4, 3), np.float32)
        with self.assertRaises(ValueError):
            rd.block_relative_drift(observed, reference)
        agreement = rd.block_relative_drift(reference, reference)
        self.assertEqual(agreement['max_relative_l2'], 0.0)
        self.assertTrue(agreement['exactly_equal'])

    def test_the_shard_merger_accepts_the_declared_partition_and_refuses_anything_else(self):
        """Two of the 34 arms are extracted as disjoint shards, so the merger is a gate:
        it must reconstruct the same deterministic partition and refuse a shard whose
        identity, coverage or precision receipt differs."""
        import shutil
        import tempfile

        from scripts.capability.readouts import merge_depth_shards as merger
        from scripts.capability.readouts.extract_frozen_readout import partition_assays

        cohort_path, _, depth_manifest = self.paths
        manifest = json.loads(depth_manifest.read_text())
        universe = [dict(assay=record['assay'], max_packed_tokens=record['max_packed_tokens'],
                         variants=record['variants']) for record in manifest['assays']]
        assignment = partition_assays(universe, 2)
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(directory, ignore_errors=True))
        records = {record['assay']: record for record in manifest['assays']}
        shards = []
        for index, assigned in enumerate(assignment):
            payload = dict(manifest)
            payload['assays'] = [dict(records[assay],
                                      file=str((depth_manifest.parent / records[assay]['file'])
                                               .resolve()))
                                 for assay in assigned]
            payload['execution_partition'] = dict(method='greedy_packed_token_work_v1', count=2,
                                                  index=index, universe=universe,
                                                  assigned_assays=assigned)
            path = directory / f'shard_depth_synthetic_{index}.json'
            path.write_text(json.dumps(payload))
            shards.append(path)
        target = merger.merge_shards(cohort_path, shards, directory / 'merged')
        merged = json.loads(Path(target).read_text())
        self.assertEqual(merged['status'], 'complete')
        self.assertEqual([row['assay'] for row in merged['assays']],
                         [row['assay'] for row in universe])
        self.assertEqual(merged['block_count'], DEPTH)
        self.assertEqual(merged['execution_merge']['count'], 2)
        # The merged manifest must load as a panel, with the same states.
        panel = rd.DepthPanel(cohort_path, Path(target), 'synthetic', self.assay_ids)
        self.addCleanup(panel.close)
        np.testing.assert_array_equal(rd.admitted_blocks_from_depth(panel),
                                      self.admitted()[1])
        # One shard alone does not cover the partition.
        with self.assertRaises(ValueError):
            merger.merge_shards(cohort_path, shards[:1], directory / 'one')
        # A shard whose assigned assays differ from the deterministic assignment is refused.
        swapped = json.loads(shards[0].read_text())
        swapped['execution_partition']['assigned_assays'] = list(reversed(assignment[0]))
        broken = directory / 'shard_depth_synthetic_swapped.json'
        broken.write_text(json.dumps(swapped))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            merger.merge_shards(cohort_path, [broken, shards[1]], directory / 'swapped')
        # A shard carrying a different measurement identity is refused.
        altered = json.loads(shards[1].read_text())
        altered['identity'] = dict(altered['identity'], batch_size=99)
        other = directory / 'shard_depth_synthetic_altered.json'
        other.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, 'identity'):
            merger.merge_shards(cohort_path, [shards[0], other], directory / 'altered')

    def test_the_permutation_control_is_within_assay_and_seeded(self):
        rows, _, _, _ = self.admitted()
        measured, assays, _ = rd.row_labels(rows)
        first = rd.permuted_labels(measured, assays, rd.BOOTSTRAP_SEED)
        again = rd.permuted_labels(measured, assays, rd.BOOTSTRAP_SEED)
        np.testing.assert_array_equal(first, again)
        for assay in np.unique(assays):
            index = np.flatnonzero(assays == assay)
            np.testing.assert_array_equal(np.sort(first[index]), np.sort(measured[index]))
        self.assertGreater(float(np.max(np.abs(
            standardized_rank(first) - standardized_rank(measured)))), 0.0)


class PanelAggregation(unittest.TestCase):
    """The panel artifact must carry the cross-cell quantities the record quotes.

    The falsification tally, the seed-consistent blocks, the per-arm breadth reading
    and the ceilings are aggregates over cells, so no per-cell report can hold them.
    An audit that reads only the cell files cannot check them, which is why they are
    written as explicit fields rather than left to be recomputed.
    """

    @staticmethod
    def _cell(arm, panel, stratum, profiles, admitted, direction):
        """One collect()-shaped cell with a named profile per seed."""
        seeds = {}
        for seed, rows in profiles.items():
            seeds[seed] = dict(
                reproduced_admitted_delta_spearman=admitted,
                reproduced_admitted_direction=direction,
                axes=dict(depth=dict(
                    depths=len(rows),
                    profile=[dict(depth=d, delta_spearman=v, interval=[v - 0.01, v + 0.01],
                                  direction=s, delta_rank_mse=0.0, r_spearman=0.1,
                                  b_r_spearman=0.2, r_minus_raw_m=0.0,
                                  r_minus_raw_m_interval=None)
                             for d, v, s in rows],
                    resolved_positive=sum(1 for _, _, s in rows if s == 'resolved_positive'),
                    resolved_negative=sum(1 for _, _, s in rows if s == 'resolved_negative'),
                    unresolved=sum(1 for _, _, s in rows if s == 'unresolved'),
                    resolved_positive_depths=[d for d, _, s in rows if s == 'resolved_positive'],
                    max_delta_spearman=max(v for _, v, _ in rows),
                    max_delta_spearman_depth=0, max_delta_spearman_interval=None,
                    max_delta_spearman_direction='unresolved')),
                summaries={}, baseline_agreement=dict(max_absolute_deviation={'B_prediction': 0.0}))
        return dict(arm=arm, panel=panel, stratum=stratum, depth=3, sha256='0' * 64,
                    hidden_width=8, position_resolved=False, admitted_block_indices=[1, 2],
                    support={}, extraction_agreement={}, prefix_control={}, seeds=seeds)

    def test_the_tally_counts_every_depth_cell_of_a_stratum(self):
        text = self._cell('textarm', 'anchor', 'literal_text_AA', {
            s: [(0, -0.02, 'resolved_negative'), (1, -0.01, 'resolved_negative'),
                (2, -0.005, 'unresolved')] for s in ('a', 'b')}, -0.02, 'resolved_negative')
        tally = summarise_depth_panel.stratum_tally([text])['literal_text_AA']
        self.assertEqual(tally['depth_cells'], 6)
        self.assertEqual(tally['resolved_positive'], 0)
        self.assertEqual(tally['resolved_negative'], 4)
        self.assertEqual(tally['unresolved'], 2)
        self.assertEqual(tally['best_point'], -0.005)
        self.assertEqual(tally['best_cell']['direction'], 'unresolved')

    def test_a_block_counts_as_seed_consistent_only_when_every_seed_resolves_it(self):
        cell = self._cell('protarm', 'anchor', 'native_sequence', {
            'a': [(0, 0.01, 'resolved_positive'), (1, 0.02, 'resolved_positive')],
            'b': [(0, 0.011, 'resolved_positive'), (1, 0.001, 'unresolved')]},
            0.001, 'unresolved')
        rows = summarise_depth_panel.seed_consistent([cell])
        self.assertEqual([row['depth'] for row in rows], [0])
        self.assertEqual(rows[0]['minimum'], 0.01)
        self.assertEqual(rows[0]['maximum'], 0.011)
        reading = summarise_depth_panel.breadth([cell], rows)[0]
        self.assertEqual(reading['selected_block'], 0)
        self.assertEqual(reading['reading'], 'boundary lifted')
        self.assertFalse(reading['selected_at_admitted_depth'])

    def test_an_already_resolved_cell_is_not_called_a_lifted_boundary(self):
        cell = self._cell('protarm', 'anchor', 'native_sequence', {
            'a': [(1, 0.03, 'resolved_positive')], 'b': [(1, 0.02, 'resolved_positive')]},
            0.025, 'resolved_positive')
        rows = summarise_depth_panel.seed_consistent([cell])
        reading = summarise_depth_panel.breadth([cell], rows)[0]
        self.assertEqual(reading['reading'], 'already resolved')
        self.assertTrue(reading['selected_at_admitted_depth'])
        ceiling = summarise_depth_panel.ceilings([cell], rows)['anchor/native_sequence']
        self.assertEqual(ceiling['single_cell_maximum'], 0.03)
        self.assertEqual(ceiling['seed_consistent_maximum'], 0.03)

    def test_two_reports_for_one_cell_are_refused(self):
        cell = self._cell('protarm', 'anchor', 'native_sequence',
                          {'a': [(0, 0.01, 'unresolved')]}, 0.0, 'unresolved')
        other = self._cell('protarm', 'other', 'native_sequence',
                           {'a': [(0, 0.01, 'unresolved')]}, 0.0, 'unresolved')
        summarise_depth_panel.refuse_duplicate_cells([cell, other])
        with self.assertRaises(ValueError) as caught:
            summarise_depth_panel.refuse_duplicate_cells([cell, other, dict(cell)])
        self.assertIn('protarm/anchor', str(caught.exception))
        # A doubled cell would double every count that sums over cells.
        doubled = summarise_depth_panel.stratum_tally([cell, dict(cell)])['native_sequence']
        self.assertEqual(doubled['depth_cells'],
                         2 * summarise_depth_panel.stratum_tally([cell])['native_sequence']['depth_cells'])


class AdmittedDesignConstruction(unittest.TestCase):
    """The admitted design must be blocked by assay, the way the admitted analysis blocks it.

    A float32 matrix product is not invariant to how its rows are blocked, so a
    design formed over a whole panel at once is not the design the admitted
    analysis fitted. That difference has already exceeded the 1e-8 the
    pipeline-identity control is bound at, on the widest arm of the panel.
    """

    def test_the_design_is_the_per_assay_product_at_any_ambient_thread_count(self):
        rng = np.random.default_rng(31)
        counts = [7, 11, 5, 13]
        blocks = rng.normal(size=(sum(counts), 4, 96)).astype(np.float32)
        projection = rd.admitted_projection(96)
        with threadpool_limits(limits=PROJECTION_BLAS_THREADS, user_api='blas'):
            pieces, start = [], 0
            for count in counts:
                chunk = blocks[start:start + count]
                pieces.append(np.concatenate([chunk[:, i] @ projection[i] for i in range(4)],
                                             axis=1))
                start += count
        reference = np.concatenate(pieces, axis=0)
        for ambient in (1, 4, 16, 48):
            with threadpool_limits(limits=ambient, user_api='blas'):
                observed = rd.admitted_design(blocks, counts)
            np.testing.assert_array_equal(
                observed, reference,
                err_msg=f'the admitted design moved at {ambient} ambient BLAS threads')

    def test_assay_row_counts_must_cover_the_block_rows(self):
        blocks = np.zeros((10, 4, 8), dtype=np.float32)
        with self.assertRaises(ValueError):
            rd.admitted_design(blocks, [4, 4])


if __name__ == '__main__':
    unittest.main()
