"""Conditions that must hold for the stability ranking-versus-error recovery.

Four things can make this panel wrong in ways a happy-path run would not show.

The **cohort** could drift. The recomputation is only comparable with the
2026-09-24 record if it sits on the same 25,856 variants over the same 101
family groups and 5,664 sites, with the same endpoint arithmetic and the same
two protease channels present on every variant. Those are pinned against the
frozen bytes, so a changed support fails here rather than quietly producing a
second, incomparable number.

The **per-family calibration** could leak. It fits a scale and an offset from a
family's own labels, which is legitimate only because the labels it uses are at
other sites; if a site's own measurement reached its own calibration, the
"qualified" error reading would be a fitted value dressed as a held-out one. The
leakage is tested directly, by perturbing one part's labels and requiring that
part's calibrated values not to move.

The **simultaneous inference** could be marginal inference with a different
name. The band must come from the project's own maximum-statistic helper, widen
as the declared family grows, and refuse a family with an undefined cell instead
of dropping it.

A **missing arm** could shrink the panel. The published support is 33 arms, and
the panel of 2026-09-24 registered all 33 while 32 of its per-arm fits had
already ceased to exist. An absent, ambiguous or wrong-schema extraction must
stop the run.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.extensions.stability_followups import COHORT_SHA256, EXPECTED
from src.capability.interactions.pairwise_epistasis import ROSTER, SPLIT_SEEDS
from src.capability.stability import ranking_error as contrast
from src.capability.stability.stability_gate import ddg

_SPEC = importlib.util.spec_from_file_location(
    'stability_ranking_panel_test',
    ROOT / 'scripts/capability/stability/fit_stability_ranking_panel.py')
assert _SPEC is not None and _SPEC.loader is not None
stage = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stage)

#: The frozen gate directory, in the published layout. Absent on a checkout
#: without the result tree, which is a reason to skip the cohort pins and not a
#: reason to weaken them.
GATE = ROOT / 'results/R4/gate_stability_20260924'

#: Digest of the frozen extraction plan, recomputed by the extractor's own rule.
PLAN_SHA256 = '283b4503ea166a3d61052bf68818c8aef162692ed3eb436a7a9864759a48763d'


def synthetic_family(groups: int, sites: int, variants: int, *, seed: int):
    """A support with the cohort's shape: families of sites of variants."""

    rng = np.random.default_rng(seed)
    group, site, target = [], [], []
    for family in range(groups):
        for position in range(sites):
            for _ in range(variants):
                group.append(f'nat-{family:03d}')
                site.append(f'nat-{family:03d}:{position}')
                target.append(float(rng.normal()))
    return np.asarray(group), np.asarray(site), np.asarray(target)


# --------------------------------------------------------------------------- #
# The cohort definition this recovery must reproduce.
# --------------------------------------------------------------------------- #

@unittest.skipUnless(GATE.exists(), 'frozen stability gate directory is not staged here')
class FrozenCohortDefinition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = stage.resolve_gate_inputs(GATE)
        cls.cohort = json.loads(cls.paths['cohort.json'].read_bytes())

    def test_resolution_finds_every_declared_input_exactly_once(self):
        self.assertEqual(set(self.paths), set(stage.GATE_INPUTS))
        for name, path in self.paths.items():
            self.assertTrue(path.is_file(), name)

    def test_cohort_bytes_are_the_frozen_ones(self):
        from src.capability.core.io import sha256_file
        self.assertEqual(sha256_file(self.paths['cohort.json']), COHORT_SHA256)
        self.assertEqual(self.cohort['schema'], 'stability_singles_cohort_v1')

    def test_support_is_the_published_one(self):
        backgrounds = self.cohort['backgrounds']
        variants = sum(len(row['variants']) for row in backgrounds)
        sites = {f"{row['name']}:{variant['position']}"
                 for row in backgrounds for variant in row['variants']}
        self.assertEqual(len(backgrounds), 101)
        self.assertEqual(variants, EXPECTED['rows'])
        self.assertEqual(len({row['group'] for row in backgrounds}), EXPECTED['groups'])
        self.assertEqual(len(sites), EXPECTED['sites'])
        self.assertEqual(self.cohort['cap'], 256)

    def test_every_variant_is_a_single_substitution_of_its_own_wild_type(self):
        for row in self.cohort['backgrounds'][:6]:
            wildtype = row['wildtype']
            self.assertEqual(row['sequences'][0], wildtype)
            for variant in row['variants']:
                sequence = row['sequences'][variant['state']]
                differing = [index for index in range(len(wildtype))
                             if sequence[index] != wildtype[index]]
                self.assertEqual(differing, [variant['position'] - 1])
                self.assertEqual(sequence[variant['position'] - 1], variant['mutant'])

    def test_endpoint_is_the_declared_two_estimate_difference(self):
        row = self.cohort['backgrounds'][0]
        wild = row['wildtype_combined_kcal_mol']
        variant = row['variants'][0]
        self.assertAlmostEqual(ddg(wild + variant['ddg'], wild), variant['ddg'], places=12)

    def test_both_protease_channels_are_present_on_every_variant(self):
        targets = stage.channel_targets(self.cohort)
        self.assertEqual(set(targets), {'combined', 'trypsin', 'chymotrypsin'})
        for name, vector in targets.items():
            self.assertEqual(len(vector), EXPECTED['rows'], name)
            self.assertTrue(np.isfinite(vector).all(), name)
        accounting = self.cohort['row_accounting']['accepted_rows_at_channel_fit_bound']
        self.assertEqual(set(accounting), {'trypsin', 'chymotrypsin'})

    def test_extraction_plan_digest_is_the_declared_one(self):
        plan_path = next(iter(GATE.glob('*/extraction_plan.json')), None) \
            or GATE / 'extraction_plan.json'
        plan = json.loads(Path(plan_path).read_bytes())
        digest = hashlib.sha256(json.dumps(
            {key: plan[key] for key in ('schema', 'cohort_sha256', 'projection', 'backgrounds')},
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(plan['cohort_sha256'], COHORT_SHA256)
        self.assertEqual(digest, PLAN_SHA256)
        self.assertEqual(sum(len(row['variants']) for row in plan['backgrounds']),
                         EXPECTED['rows'])


# --------------------------------------------------------------------------- #
# The per-family calibration path.
# --------------------------------------------------------------------------- #

class WithinFamilyCalibration(unittest.TestCase):
    def setUp(self):
        self.group, self.site, self.target = synthetic_family(6, 8, 3, seed=11)

    def test_site_partition_is_balanced_and_label_blind(self):
        sites = [f'nat-000:{index}' for index in range(34)]
        assignment = contrast.site_folds('nat-000', sites)
        counts = [sum(1 for value in assignment.values() if value == fold)
                  for fold in range(contrast.CALIBRATION_FOLDS)]
        self.assertEqual(sum(counts), len(sites))
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertGreaterEqual(min(counts), 8)
        # Repeating the call, and shuffling the input order, cannot move a site.
        self.assertEqual(assignment, contrast.site_folds('nat-000', list(reversed(sites))))
        # A different family gets its own partition: the same ordinal positions
        # do not land in the same parts, so the partition is not a function of
        # the site's index.
        other = contrast.site_folds('nat-001', [s.replace('000', '001') for s in sites])
        self.assertNotEqual([assignment[f'nat-000:{index}'] for index in range(34)],
                            [other[f'nat-001:{index}'] for index in range(34)])
        # Nor of the seed: a different seed repartitions the same family.
        reseeded = contrast.site_folds('nat-000', sites, seed=contrast.CALIBRATION_SEED + 1)
        self.assertNotEqual([assignment[site] for site in sites],
                            [reseeded[site] for site in sites])

    def test_a_family_with_fewer_sites_than_folds_is_refused(self):
        group = np.asarray(['nat-000'] * 6)
        site = np.asarray(['nat-000:0'] * 3 + ['nat-000:1'] * 3)
        with self.assertRaises(ValueError) as caught:
            contrast.family_calibrated(np.arange(6.0), np.arange(6.0), group, site)
        self.assertIn('cannot support', str(caught.exception))

    def test_calibration_recovers_a_family_specific_affine_distortion(self):
        """A score that orders every family but on its own scale is recoverable."""

        prediction = np.empty(len(self.target))
        for index, family in enumerate(sorted(set(self.group.tolist()))):
            rows = self.group == family
            prediction[rows] = (3.0 + 2.0 * index) * self.target[rows] - 5.0 * index
        calibrated, record = contrast.family_calibrated(
            prediction, self.target, self.group, self.site)
        self.assertEqual(record['calibrations'],
                         contrast.CALIBRATION_FOLDS * len(set(self.group.tolist())))
        self.assertEqual(record['degenerate_slope_calibrations'], 0)
        # The within-family affine map is invertible, so the calibrated value is
        # the target itself up to the fit's own numerical error.
        self.assertLess(float(np.max(np.abs(calibrated - self.target))), 1e-8)
        # The uncalibrated prediction is nowhere near it, which is the point.
        self.assertGreater(float(np.max(np.abs(prediction - self.target))), 1.0)

    def test_no_part_uses_its_own_labels(self):
        rng = np.random.default_rng(3)
        prediction = self.target * 2.0 + rng.normal(scale=0.3, size=len(self.target))
        calibrated, _ = contrast.family_calibrated(
            prediction, self.target, self.group, self.site)
        family = 'nat-002'
        rows = np.flatnonzero(self.group == family)
        assignment = contrast.site_folds(family, self.site[rows])
        part = np.asarray([assignment[site] for site in self.site[rows]])
        held = rows[part == 0]
        perturbed = self.target.copy()
        perturbed[held] += 17.0
        moved, _ = contrast.family_calibrated(
            prediction, perturbed, self.group, self.site)
        self.assertTrue(np.allclose(moved[held], calibrated[held], rtol=0, atol=1e-12))
        # Every other part of that family does move, because those rows are in
        # the fitting set of the perturbed part's neighbours.
        others = rows[part != 0]
        self.assertFalse(np.allclose(moved[others], calibrated[others], rtol=0, atol=1e-12))

    def test_a_flat_prediction_collapses_to_the_weighted_mean_and_is_counted(self):
        """A column with no gradation gets the correct limit, and it is counted.

        Each part is then calibrated to the site-weighted mean of the *other*
        parts of its own family, which is the best affine prediction available
        from a constant column, so the calibrated value is constant inside a part
        and generally differs between parts.
        """

        from src.capability.readouts.readout_analysis import row_weights

        prediction = np.zeros(len(self.target))
        calibrated, record = contrast.family_calibrated(
            prediction, self.target, self.group, self.site)
        families = sorted(set(self.group.tolist()))
        self.assertEqual(record['degenerate_slope_calibrations'],
                         contrast.CALIBRATION_FOLDS * len(families))
        self.assertEqual(record['slope_range'], [0.0, 0.0])
        for family in families:
            rows = np.flatnonzero(self.group == family)
            assignment = contrast.site_folds(family, self.site[rows])
            part = np.asarray([assignment[site] for site in self.site[rows]])
            for fold in range(contrast.CALIBRATION_FOLDS):
                fit, evaluate = rows[part != fold], rows[part == fold]
                weights = row_weights(self.site[fit], np.zeros(len(fit), dtype=int))
                expected = float((weights * self.target[fit]).sum() / weights.sum())
                self.assertLess(abs(float(calibrated[evaluate].std())), 1e-12)
                self.assertAlmostEqual(float(calibrated[evaluate][0]), expected, places=12)

    def test_sites_are_weighted_equally_inside_the_fitting_part(self):
        """One crowded site must not outvote the rest of its family."""

        group = np.asarray(['nat-000'] * 23)
        site = np.asarray(['nat-000:0'] * 20 + ['nat-000:1', 'nat-000:2', 'nat-000:3'])
        prediction = np.concatenate([np.zeros(20), [1.0, 2.0, 3.0]])
        target = np.concatenate([np.full(20, 100.0), [1.0, 2.0, 3.0]])
        # Fold assignment puts one site in each part, so the part holding the
        # crowded site is fitted on the three sparse ones and vice versa.
        calibrated, record = contrast.family_calibrated(
            prediction, target, group, site, folds=4)
        self.assertEqual(record['calibrations'], 4)
        crowded = site == 'nat-000:0'
        fitted = float(calibrated[crowded][0])
        # Fitted on the three single-variant sites, the map is the identity, so
        # the crowded site's prediction of zero calibrates to about zero rather
        # than being dragged to its own mean of 100.
        self.assertLess(abs(fitted), 1.0)

    def test_nonfinite_input_is_refused(self):
        prediction = self.target.copy()
        prediction[0] = np.nan
        with self.assertRaises(ValueError):
            contrast.family_calibrated(prediction, self.target, self.group, self.site)

    def test_calibrated_increment_is_paired_on_identical_treatment(self):
        """Both sides are calibrated, so an inert column cannot score."""

        rng = np.random.default_rng(5)
        baseline = self.target * 0.5 + rng.normal(scale=0.5, size=len(self.target))
        labels, values, record = contrast.calibrated_mse_contrast(
            self.target, baseline, baseline, self.group, self.site)
        self.assertEqual(len(labels), len(set(self.group.tolist())))
        self.assertTrue(np.allclose(values, 0.0, rtol=0, atol=1e-12))
        self.assertEqual(set(record), {'baseline', 'augmented'})


# --------------------------------------------------------------------------- #
# The simultaneous-inference wiring.
# --------------------------------------------------------------------------- #

class SimultaneousInference(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(17)
        self.matrix = self.rng.normal(size=(101, 33))
        self.columns = list(ROSTER)

    def test_band_is_the_project_helper_and_not_a_restatement(self):
        from scripts.capability.interactions.run_residual_panel import simultaneous_bands
        family = contrast.simultaneous_family(self.matrix, self.columns, draws=2000)
        expected = simultaneous_bands(self.matrix, draws=2000)
        self.assertEqual(family['bands'], expected)
        self.assertEqual(family['bands']['groups'], 101)
        self.assertEqual(family['bands']['family_size'], 33)
        self.assertTrue(family['bands']['conditional_on_fitted_predictions'])

    def test_the_band_widens_with_the_declared_family(self):
        single = contrast.simultaneous_family(self.matrix[:, :1], self.columns[:1], draws=2000)
        full = contrast.simultaneous_family(self.matrix, self.columns, draws=2000)
        self.assertLess(single['bands']['critical_value'], full['bands']['critical_value'])
        self.assertGreater(full['bands']['critical_value'], 2.5)

    def test_verdicts_follow_the_band_and_not_the_point(self):
        matrix = self.rng.normal(scale=0.01, size=(101, 33))
        matrix[:, 0] += 5.0
        matrix[:, 1] += 0.001
        family = contrast.simultaneous_family(matrix, self.columns, draws=2000)
        self.assertEqual(family['verdicts'][self.columns[0]], 'above_zero')
        self.assertEqual(family['verdicts'][self.columns[1]], 'unresolved')
        self.assertEqual(family['resolved_above_zero'], [self.columns[0]])
        self.assertGreater(float(np.mean(matrix[:, 1])), 0.0)

    def test_an_undefined_cell_is_refused_rather_than_dropped(self):
        matrix = self.matrix.copy()
        matrix[7, 3] = np.nan
        with self.assertRaises(ValueError) as caught:
            contrast.simultaneous_family(matrix, self.columns, draws=100)
        self.assertIn('nonfinite', str(caught.exception))

    def test_column_count_must_match_the_declared_family(self):
        with self.assertRaises(ValueError):
            contrast.simultaneous_family(self.matrix, self.columns[:-1], draws=100)

    def test_marginal_interval_is_labelled_as_marginal(self):
        record = contrast.marginal_interval(self.matrix[:, 0])
        self.assertEqual(record['groups'], 101)
        self.assertIn('not adjusted', record['multiplicity'])

    def test_marginal_resolves_more_than_simultaneous_on_the_same_family(self):
        matrix = self.rng.normal(scale=1.0, size=(101, 33)) + 0.25
        family = contrast.simultaneous_family(matrix, self.columns, draws=4000)
        marginal = [contrast.marginal_interval(matrix[:, index], draws=4000)
                    for index in range(33)]
        resolved_marginal = sum(1 for row in marginal if row['interval'][0] > 0)
        self.assertGreaterEqual(resolved_marginal, len(family['resolved_above_zero']))

    def test_ranking_versus_error_tabulates_the_paired_metric_disagreement(self):
        error = {'verdicts': {'a': 'above_zero', 'b': 'unresolved', 'c': 'unresolved'},
                 'resolved_above_zero': ['a'], 'resolved_below_zero': []}
        rank = {'verdicts': {'a': 'above_zero', 'b': 'above_zero', 'c': 'unresolved'},
                'resolved_above_zero': ['a', 'b'], 'resolved_below_zero': []}
        summary = contrast.ranking_versus_error(
            {'within_family_calibrated_error': error,
             'within_family_ranking_on_error_controls': rank}, ['a', 'b', 'c'])
        cross = summary['paired_metric_comparison']['cross_tabulation']
        self.assertEqual(cross['both'], ['a'])
        self.assertEqual(cross['ranking_only'], ['b'])
        self.assertEqual(cross['error_only'], [])
        self.assertEqual(cross['neither'], ['c'])
        self.assertEqual(summary['resolved_counts']['within_family_ranking_on_error_controls'],
                         {'above_zero': 2, 'below_zero': 0, 'unresolved': 1})

    def test_contrast_definitions_pair_one_metric_per_control_set(self):
        self.assertEqual(
            contrast.CONTRASTS['within_family_calibrated_error']['control'],
            contrast.CONTRASTS['within_family_ranking_on_error_controls']['control'])
        self.assertEqual(contrast.CONTRASTS['within_family_ranking']['control'], 'S2')
        self.assertEqual(contrast.CONTRASTS['transfer_error']['control'], 'S')


# --------------------------------------------------------------------------- #
# A missing arm is a failure, not a smaller panel.
# --------------------------------------------------------------------------- #

class MissingArmFailsLoudly(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def manifest(self, directory: Path, arm: str, *, schema=stage.LIKELIHOOD_SCHEMA,
                 cohort=COHORT_SHA256) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'manifest_{arm}.json').write_text(json.dumps({
            'identity': {'schema': schema, 'arm': arm, 'cohort_sha256': cohort},
            'status': 'complete', 'backgrounds': []}))

    def test_an_absent_extraction_is_refused(self):
        self.manifest(self.root / 'cell', 'gpt2')
        with self.assertRaises(SystemExit) as caught:
            stage.discover_extraction([self.root], 'progen3-3b')
        message = str(caught.exception)
        self.assertIn('do not resolve to exactly one', message)
        # The refusal names the arm and the candidates it did find, so an
        # operator can see whether the extraction is absent or mislabelled.
        self.assertIn('"progen3-3b": []', message)

    def test_two_extractions_of_one_arm_are_refused_rather_than_chosen_between(self):
        self.manifest(self.root / 'first', 'progen3-3b')
        self.manifest(self.root / 'second', 'progen3-3b')
        with self.assertRaises(SystemExit) as caught:
            stage.discover_extraction([self.root], 'progen3-3b')
        message = str(caught.exception)
        self.assertIn('do not resolve to exactly one', message)
        self.assertIn(str((self.root / 'first').resolve()), message)
        self.assertIn(str((self.root / 'second').resolve()), message)

    def test_the_walk_reports_every_unresolved_arm_at_once(self):
        """One refusal names all of them, so a staging fault is fixed in one pass."""

        self.manifest(self.root / 'cell', 'gpt2')
        with self.assertRaises(SystemExit) as caught:
            stage.locate_extractions([self.root], ['gpt2', 'progen3-3b', 'prollama'])
        message = str(caught.exception)
        self.assertIn('2 of 3 arms', message)
        self.assertIn('progen3-3b', message)
        self.assertIn('prollama', message)

    def test_a_representation_extraction_does_not_satisfy_this_stage(self):
        self.manifest(self.root / 'cell', 'progen3-3b',
                      schema='stability_singles_extraction_v1')
        with self.assertRaises(SystemExit):
            stage.discover_extraction([self.root], 'progen3-3b')

    def test_an_extraction_of_another_cohort_does_not_satisfy_this_stage(self):
        self.manifest(self.root / 'cell', 'progen3-3b', cohort='0' * 64)
        with self.assertRaises(SystemExit):
            stage.discover_extraction([self.root], 'progen3-3b')

    def test_a_malformed_manifest_is_skipped_without_crashing_the_search(self):
        (self.root / 'broken').mkdir()
        (self.root / 'broken/manifest_progen3-3b.json').write_text('{not json')
        self.manifest(self.root / 'good', 'progen3-3b')
        self.assertEqual(stage.discover_extraction([self.root], 'progen3-3b'),
                         (self.root / 'good').resolve())

    def test_an_unknown_arm_name_is_refused_before_anything_is_read(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / 'scripts/capability/stability/fit_stability_ranking_panel.py'),
             '--gate-dir', str(self.root), '--extraction-root', str(self.root),
             '--arms', 'not-an-arm', '--out', str(self.root / 'out')],
            capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('not on the frozen roster', completed.stderr)

    @unittest.skipUnless(GATE.exists(), 'frozen stability gate directory is not staged here')
    def test_the_full_panel_refuses_when_one_arm_has_no_extraction(self):
        """The production path asks for all 33 arms and must not settle for fewer."""

        for arm in ROSTER:
            if arm != 'progen3-3b':
                self.manifest(self.root / 'cell', arm)
        completed = subprocess.run(
            [sys.executable, str(ROOT / 'scripts/capability/stability/fit_stability_ranking_panel.py'),
             '--gate-dir', str(GATE), '--extraction-root', str(self.root),
             '--out', str(self.root / 'out')],
            capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('progen3-3b', completed.stderr)
        self.assertFalse((self.root / 'out' / stage.COMPLETION).exists())
        self.assertFalse((self.root / 'out' / 'panel.json').exists())

    def test_a_restricted_run_cannot_produce_the_production_completion_record(self):
        self.assertNotEqual(stage.COMPLETION, stage.SMOKE_COMPLETION)
        self.assertEqual(len(ROSTER), 33)
        self.assertEqual(tuple(SPLIT_SEEDS), (20260923, 20260924, 20260925))


if __name__ == '__main__':
    unittest.main()
