"""Conditions the retrieval and memorization gate rests on.

Four properties have to hold for a stratified gain to be readable at all: the
band edges are the frozen ones rather than newly chosen numbers, every
stratification is a genuine partition of the fitted support with an accounted
residual, no measured outcome reaches a band boundary, and each stratum's
estimate is the original fit's own statistic on a subset of its units rather than
a new fit. The negative paths are tested beside them, because a check that
cannot refuse is not a check.
"""
from pathlib import Path
import json
import math
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.capability.context import homology
from src.capability.context import retrieval_strata as rs
from src.capability.context.profile_increment import summarize
from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS

GATE = ROOT / 'archive/logs/R5/gate_retrieval_20260923'
DECLARATION = GATE / 'strata_declaration.json'
EXCLUSION = GATE / 'indel_exclusion.json'
REFITS = GATE / 'stability'
REPORT = GATE / 'retrieval_strata_report.json'
CROSSED = ROOT / 'archive/logs/shared/crossed_controls_20260923/reports'
READOUT_COHORT = ROOT / 'results/R5/context_mutation_rescue_20260923/cohort.json'
PAIRWISE_HITS = ROOT / 'archive/logs/R3/pairwise_homologs_20260924/search/hits.tsv'

PLAN_SHA = 'e338420f5df70143ccfc8d16ec5479a67b330ec35d36ea7c5965ea31a59e179c'
CONTRACT_SHA = '6a0a76a4ad1f083f7506f8fa29cb809b555fcf64ccbd722972de877cf26759c4'
READOUT_COHORT_SHA = '4093ac34368cd9e7be02e5c84a22655ce78cb9bb4e4d58d626f281868dbe7992'


def declaration() -> dict:
    return json.loads(DECLARATION.read_text())


class BandEdges(unittest.TestCase):
    def test_identity_bands_are_the_frozen_homology_declaration(self):
        self.assertIs(rs.IDENTITY_BANDS, homology.STRATUM_NAMES)
        self.assertEqual(rs.NEAR_DUPLICATE_IDENTITY, homology.STRATUM_EDGES[3])
        self.assertEqual(rs.NEAR_DUPLICATE_IDENTITY, 95.0)
        self.assertEqual(rs.STRATIFICATIONS['identity_band']['edges'],
                         list(homology.STRATUM_EDGES))

    def test_the_coarse_depth_edge_is_a_fine_depth_edge(self):
        self.assertIn(rs.COARSE_DEPTH_EDGE, rs.DEPTH_EDGES)
        self.assertEqual(rs.COARSE_DEPTH_EDGE, rs.DEPTH_EDGES[1])

    def test_every_stratification_declares_named_bands_and_a_definition(self):
        for name, spec in rs.STRATIFICATIONS.items():
            bands = tuple(spec['bands'])
            self.assertGreaterEqual(len(bands), 2, name)
            self.assertEqual(len(set(bands)), len(bands), name)
            self.assertTrue(str(spec['definition']).strip(), name)
            self.assertTrue(str(spec['quantity']).strip(), name)

    def test_source_provenance_declares_which_exclusion_rule_is_primary(self):
        self.assertEqual(rs.SOURCE_EXCLUSION['primary'], 'drop_whole_units')
        self.assertEqual(rs.SOURCE_EXCLUSION['sensitivity'], 'drop_shared_source_assays_only')
        for key in ('primary_reason', 'primary_cost', 'sensitivity_reason'):
            self.assertTrue(rs.SOURCE_EXCLUSION[key].strip(), key)


class BandAssignment(unittest.TestCase):
    def test_identity_and_near_duplicate_agree_at_the_frozen_edge(self):
        self.assertEqual(rs.identity_band(95.0), 'ge95_near_duplicate')
        self.assertEqual(rs.near_duplicate_band(95.0), 'near_duplicate_present')
        self.assertEqual(rs.identity_band(94.999), 'id70_to_95_close_homology')
        self.assertEqual(rs.near_duplicate_band(94.999), 'near_duplicate_absent')
        self.assertEqual(rs.identity_band(0.0), 'lt30_no_detectable_homology')
        self.assertEqual(rs.identity_band(100.0), 'ge95_near_duplicate')

    def test_every_identity_lands_in_exactly_one_band(self):
        for step in range(0, 1001):
            value = step / 10.0
            bands = [band for band in rs.IDENTITY_BANDS if band == rs.identity_band(value)]
            self.assertEqual(len(bands), 1, value)
            self.assertIn(rs.near_duplicate_band(value), rs.NEAR_DUPLICATE_BANDS)

    def test_depth_bands_are_decades_and_absence_is_its_own_band(self):
        self.assertEqual(rs.depth_band(None), 'no_homolog_support')
        self.assertEqual(rs.coarse_depth_band(None), 'support_shallow_lt100')
        self.assertEqual(rs.depth_band(9.99), 'neff_lt10')
        self.assertEqual(rs.depth_band(10.0), 'neff_10_100')
        self.assertEqual(rs.depth_band(99.99), 'neff_10_100')
        self.assertEqual(rs.depth_band(100.0), 'neff_100_1000')
        self.assertEqual(rs.depth_band(999.99), 'neff_100_1000')
        self.assertEqual(rs.depth_band(1000.0), 'neff_ge1000')
        self.assertEqual(rs.coarse_depth_band(99.99), 'support_shallow_lt100')
        self.assertEqual(rs.coarse_depth_band(100.0), 'support_deep_ge100')

    def test_family_and_source_bands(self):
        self.assertEqual(rs.family_band(1), 'remote_singleton_group')
        self.assertEqual(rs.family_band(2), 'close_multi_member_group')
        self.assertEqual(rs.source_band(['GFP_AEQVI_Sarkisyan_2016']), 'source_other')
        self.assertEqual(rs.source_band(['A_Tsuboyama_2023_1AOY']), 'source_tsuboyama_2023')
        self.assertEqual(rs.source_band(['other', 'A_Tsuboyama_2023_1AOY']),
                         'source_tsuboyama_2023')

    def test_a_quantity_outside_its_support_is_refused(self):
        for value in (-0.1, 100.1, float('nan')):
            with self.assertRaises(ValueError):
                rs.identity_band(value)
            with self.assertRaises(ValueError):
                rs.near_duplicate_band(value)
        for value in (0.0, -1.0, float('nan')):
            with self.assertRaises(ValueError):
                rs.depth_band(value)
            with self.assertRaises(ValueError):
                rs.coarse_depth_band(value)
        with self.assertRaises(ValueError):
            rs.family_band(0)
        with self.assertRaises(ValueError):
            rs.source_band([])


class Partition(unittest.TestCase):
    def one_unit(self) -> dict:
        return rs.assign_unit(max_identity_over_query=42.0, neff=50.0, group_members=1,
                              assay_identifiers=['A_Other_2016'])

    def test_a_complete_assignment_passes_and_counts_every_unit(self):
        assignment = {'u1': self.one_unit(), 'u2': self.one_unit()}
        record = rs.verify_partition(assignment)
        for name in rs.STRATIFICATIONS:
            self.assertEqual(record[name]['units'], 2)
            self.assertEqual(record[name]['unassigned_units'], 0)
            self.assertEqual(sum(record[name]['bands'].values()), 2)

    def test_a_missing_stratification_is_refused(self):
        assignment = {'u1': self.one_unit(), 'u2': self.one_unit()}
        del assignment['u2']['depth_band']
        with self.assertRaises(ValueError):
            rs.verify_partition(assignment)

    def test_an_undeclared_band_is_refused(self):
        assignment = {'u1': self.one_unit()}
        assignment['u1']['identity_band'] = 'id40_invented'
        with self.assertRaises(ValueError):
            rs.verify_partition(assignment)

    def test_an_empty_unit_set_is_refused(self):
        with self.assertRaises(ValueError):
            rs.verify_partition({})

    def test_bands_of_refuses_an_undeclared_stratification(self):
        with self.assertRaises(ValueError):
            rs.bands_of({'u1': self.one_unit()}, 'invented')


class Decomposition(unittest.TestCase):
    def test_a_unit_count_weighted_mean_of_bands_returns_the_whole_support(self):
        residual = rs.decomposition_residual(
            (3 * 1.0 + 7 * 2.0) / 10, {'a': 1.0, 'b': 2.0}, {'a': 3, 'b': 7})
        self.assertLess(abs(residual), 1e-15)

    def test_an_empty_band_does_not_need_a_point_estimate(self):
        residual = rs.decomposition_residual(1.0, {'a': 1.0, 'b': None}, {'a': 4, 'b': 0})
        self.assertLess(abs(residual), 1e-15)

    def test_a_populated_band_without_a_point_estimate_is_refused(self):
        with self.assertRaises(ValueError):
            rs.decomposition_residual(1.0, {'a': 1.0, 'b': None}, {'a': 4, 'b': 2})


class Power(unittest.TestCase):
    def test_a_stratum_below_the_unit_floor_carries_no_verdict(self):
        self.assertEqual(rs.resolution(MINIMUM_BOOTSTRAP_UNITS - 1, True),
                         'unresolved_thin_support')
        self.assertEqual(rs.resolution(0, None), 'unresolved_thin_support')
        self.assertEqual(rs.resolution(MINIMUM_BOOTSTRAP_UNITS, None), 'unresolved_no_interval')
        self.assertEqual(rs.resolution(MINIMUM_BOOTSTRAP_UNITS, False),
                         'unresolved_interval_crosses_zero')
        self.assertEqual(rs.resolution(MINIMUM_BOOTSTRAP_UNITS, True), 'resolved')

    def test_the_estimator_itself_refuses_an_interval_below_the_floor(self):
        rows = [{'cluster': index, 'value': 1.0} for index in range(MINIMUM_BOOTSTRAP_UNITS - 1)]
        record = summarize(rows, 'value', bootstrap=64, seed=20260923)
        self.assertTrue(record['degenerate'])
        self.assertIsNone(record['interval'])
        self.assertIsNone(record['excludes_zero'])

    def test_kish_conventions_are_distinct_and_named(self):
        self.assertAlmostEqual(rs.kish_subunits({'g': 2, 'h': 2}), 4.0)
        self.assertAlmostEqual(rs.kish_row_counts([1, 1, 1, 1]), 4.0)
        self.assertLess(rs.kish_row_counts([1, 100]), 2.0)
        self.assertLess(rs.kish_subunits({'g': 1, 'h': 100}), 101.0)
        self.assertEqual(rs.kish_subunits({}), 0.0)
        self.assertEqual(rs.kish_row_counts([]), 0.0)
        with self.assertRaises(ValueError):
            rs.kish_row_counts([0, 1])
        with self.assertRaises(ValueError):
            rs.kish_subunits({'g': 0})


@unittest.skipUnless(DECLARATION.exists(), 'the frozen strata declaration is not on this host')
class DeclaredSupport(unittest.TestCase):
    def setUp(self):
        self.declaration = declaration()

    def test_every_stratification_partitions_both_cohorts(self):
        for cohort, total in (('stability', 'groups'), ('readout_anchor', 'clusters')):
            entry = self.declaration[cohort]
            units = entry['support'][total]
            self.assertEqual(len(entry['assignment']), units)
            for name, bands in entry['band_support'].items():
                self.assertEqual(sum(band[total] for band in bands.values()), units, name)
                self.assertEqual(entry['partition'][name]['unassigned_units'], 0, name)

    def test_the_stability_support_is_the_frozen_one(self):
        support = self.declaration['stability']['support']
        self.assertEqual((support['groups'], support['site_pairs'], support['cycles']),
                         (64, 217, 8192))
        self.assertAlmostEqual(support['kish_effective_site_pairs_cycle_counts'], 121.6, places=1)

    def test_the_readout_support_is_the_frozen_anchor(self):
        support = self.declaration['readout_anchor']['support']
        self.assertEqual((support['clusters'], support['assays'], support['variants']),
                         (163, 201, 25728))

    def test_the_shared_source_share_matches_the_catalogued_limitation(self):
        evidence = self.declaration['readout_anchor']['source_provenance_evidence']
        self.assertEqual(evidence['anchor_assays_with_token'], 64)
        self.assertEqual(evidence['anchor_variants_with_token'], 8192)
        bands = self.declaration['readout_anchor']['band_support']['source_provenance']
        self.assertEqual(bands['source_tsuboyama_2023']['clusters'], 64)
        self.assertEqual(bands['source_other']['clusters'], 99)
        self.assertEqual(len(evidence['mixed_source_clusters']), 2)
        self.assertEqual(evidence['unshared_support_under_primary_rule']['clusters'], 99)
        self.assertEqual(evidence['unshared_support_under_sensitivity_rule']['clusters'], 101)

    def test_the_whole_stability_cohort_sits_in_one_provenance_band(self):
        bands = self.declaration['stability']['band_support']['source_provenance']
        self.assertEqual(bands['source_other']['groups'], 0)
        self.assertEqual(bands['source_tsuboyama_2023']['groups'], 64)


@unittest.skipUnless(EXCLUSION.exists(), 'the declared indel exclusion is not on this host')
class IndelExclusion(unittest.TestCase):
    """The exclusion is a construct annotation, and it names the cycles it removes."""

    def setUp(self):
        self.exclusion = json.loads(EXCLUSION.read_text())

    def test_it_names_130_cycles_in_three_backgrounds_and_loses_one_group(self):
        summary = self.exclusion['summary']
        self.assertEqual(summary['excluded_cycles'], 130)
        self.assertEqual(summary['affected_backgrounds'], 3)
        self.assertEqual(summary['cohort_cycles'], 8192)
        self.assertEqual(summary['retained_cycles'], 8062)
        self.assertEqual(summary['retained_groups'], 63)
        self.assertEqual(summary['retained_site_pairs'], 211)
        self.assertEqual(summary['groups_lost_entirely'], ['nat-014'])

    def test_every_excluded_cycle_carries_an_insertion_or_deletion_construct(self):
        from src.capability.stability.pairwise_stability import INDEL_MUT_TYPE_PREFIXES
        seen = 0
        for name, entries in self.exclusion['evidence'].items():
            for entry in entries:
                self.assertTrue(entry['carriers'], name)
                for carrier in entry['carriers']:
                    self.assertTrue(carrier['mut_type'].lower().startswith(
                        INDEL_MUT_TYPE_PREFIXES), carrier)
                    seen += 1
        self.assertGreater(seen, 0)

    def test_the_rule_reads_no_measured_value(self):
        source = (ROOT / 'scripts/capability/stability/declare_indel_exclusion.py').read_text()
        for field in ("'dG_ML'", "'value'", "['epsilon']", 'deltaG_t_95CI'):
            self.assertNotIn(field, source)
        self.assertIn('mut_type', self.exclusion['rule'])

    def test_the_scope_is_stated_as_an_evaluation_filter(self):
        self.assertIn('evaluation filter', self.exclusion['scope'])
        self.assertIn('training side', self.exclusion['scope'])


@unittest.skipUnless(DECLARATION.exists() and READOUT_COHORT.exists() and PAIRWISE_HITS.exists(),
                     'the declaration inputs are not on this host')
class LabelIndependence(unittest.TestCase):
    """No measured outcome may reach a band, checked by changing every one of them."""

    def test_the_declaration_rebuilds_from_a_cohort_with_no_measurement_in_it(self):
        """Strip every measured field and rebuild: the bands must not move.

        This is the sharp form of the claim. The rebuilt cohort carries the assay
        identifier, its cluster and its wild-type query id and nothing else, so a
        band that survives it cannot be a function of any measurement. The cell
        reports' cohort digest is repointed at the stripped file in a copy, because
        the entry point's own binding check would otherwise refuse the fixture; that
        check is what keeps the real run honest and is exercised elsewhere.
        """

        import tempfile
        import hashlib
        original = declaration()
        cohort = json.loads(READOUT_COHORT.read_text())
        stripped = {'assays': [{key: row[key] for key in ('assay', 'cluster', 'wildtype_id')}
                               for row in cohort['assays']]}
        self.assertNotIn('measured', json.dumps(stripped))
        with tempfile.TemporaryDirectory() as directory:
            corrupted = Path(directory) / 'cohort.json'
            corrupted.write_text(json.dumps(stripped))
            sha = hashlib.sha256(corrupted.read_bytes()).hexdigest()
            reports = Path(directory) / 'reports'
            for path in sorted(CROSSED.glob('*/crossed_controls_*_fold*.json')):
                cell = json.loads(path.read_text())
                cell['cohort_sha256'] = sha
                target = reports / path.parent.name / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(cell))
            out = Path(directory) / 'rebuilt'
            subprocess.run(
                [sys.executable, str(ROOT / 'scripts/capability/context/declare_retrieval_strata.py'),
                 '--pairwise-plan',
                 str(ROOT / 'archive/logs/R3/pairwise_epistasis_20260924/extraction_plan.json'),
                 '--expect-pairwise-plan-sha256', PLAN_SHA,
                 '--pairwise-hits', str(PAIRWISE_HITS),
                 '--pairwise-search-manifest',
                 str(ROOT / 'archive/logs/R3/pairwise_homologs_20260924/search/search_manifest.json'),
                 '--pairwise-profiles',
                 str(ROOT / 'archive/logs/R3/pairwise_epistasis_20260924/profile_features.npz'),
                 '--family-groups',
                 str(ROOT / 'data/pairwise_assets/megascale_family_groups_20260924.json'),
                 '--expect-grouping-contract-sha256', CONTRACT_SHA,
                 '--readout-cohort', str(corrupted),
                 '--expect-readout-cohort-sha256', sha,
                 '--readout-search', str(ROOT / 'results/R5/retrieval_bound_20260807/search.json'),
                 '--readout-profiles', str(ROOT / 'results/R5/retrieval_bound_20260807/profiles.json'),
                 '--crossed-reports', str(reports), '--out', str(out)],
                check=True, capture_output=True, cwd=ROOT)
            rebuilt = json.loads((out / 'strata_declaration.json').read_text())
        for cohort_name in ('stability', 'readout_anchor'):
            self.assertEqual(rebuilt[cohort_name]['assignment'],
                             original[cohort_name]['assignment'], cohort_name)
            self.assertEqual(rebuilt[cohort_name]['band_support'],
                             original[cohort_name]['band_support'], cohort_name)

    def test_the_declaration_entry_point_names_no_measurement_field(self):
        source = (ROOT / 'scripts/capability/context/declare_retrieval_strata.py').read_text()
        for field in ('--pairwise-cohort', "'epsilon'", "'measurements'", "'measured'",
                      "'dG_ML'", "'profile_scores'"):
            self.assertNotIn(field, source)


@unittest.skipUnless(CROSSED.exists(), 'the crossed-control cell reports are not on this host')
class ReadoutReaggregation(unittest.TestCase):
    """A stratum estimate must be the published estimator on a subset of its units."""

    def cells(self):
        for path in sorted(CROSSED.glob('*/crossed_controls_*_fold*.json')):
            yield path, json.loads(path.read_text())

    def test_reaggregating_the_whole_anchor_reproduces_every_published_summary(self):
        from scripts.capability.context.analyse_retrieval_strata import READOUT_CONTRASTS
        checked = 0
        for path, cell in self.cells():
            self.assertEqual((cell['n_assays'], cell['n_families']), (201, 163), path.name)
            for key in READOUT_CONTRASTS:
                published = cell['summaries'][key]['point']
                rebuilt = summarize(cell['assays'], key, bootstrap=16, seed=20260923)['point']
                self.assertEqual(rebuilt, published, f'{path.name}:{key}')
                checked += 1
        self.assertEqual(checked, 15 * len(READOUT_CONTRASTS))

    def test_every_arm_shares_one_held_family_map_at_a_split_seed(self):
        per_seed: dict[int, list] = {}
        for _, cell in self.cells():
            membership = [sorted(map(str, fold['held_families'])) for fold in cell['folds']]
            per_seed.setdefault(cell['fold_seed'], []).append(membership)
        self.assertEqual(sorted(per_seed), [20260923, 20260924, 20260925])
        for seed, maps in per_seed.items():
            self.assertEqual(len(maps), 5, seed)
            for realised in maps[1:]:
                self.assertEqual(realised, maps[0], seed)

    def test_no_held_family_appears_in_its_own_training_side(self):
        for path, cell in self.cells():
            for fold in cell['folds']:
                held = {str(value) for value in fold['held_families']}
                training = {str(value) for value in fold['training_families']}
                self.assertFalse(held & training, path.name)
                for inner in fold['inner_folds']:
                    self.assertFalse(held & {str(value) for value in inner}, path.name)


@unittest.skipUnless(REFITS.exists() and any(REFITS.glob('refit_*.json')),
                     'the stability refit records are not on this host')
class StabilityRefit(unittest.TestCase):
    def records(self):
        for path in sorted(REFITS.glob('refit_*.json')):
            yield path, json.loads(path.read_text())

    def test_every_refit_agrees_with_the_reference_fit(self):
        seen = 0
        for path, record in self.records():
            agreement = record['reference_agreement']
            self.assertGreater(agreement['contrasts_checked'], 0, path.name)
            self.assertLessEqual(agreement['max_absolute_increment_deviation_kcal2'],
                                 agreement['tolerance_kcal2'], path.name)
            self.assertEqual((record['groups'], record['site_pairs'], record['cycles']),
                             (64, 217, 8192), path.name)
            self.assertEqual(sorted(record['seeds']), ['20260923', '20260924', '20260925'])
            seen += 1
        self.assertGreater(seen, 0)

    def test_every_refit_is_bound_to_the_frozen_declaration(self):
        expected = declaration()['declaration_sha256'] if DECLARATION.exists() else None
        for path, record in self.records():
            self.assertEqual(record['inputs']['strata_declaration_sha256'], expected, path.name)

    def test_two_dispatches_of_one_arm_select_the_same_ridge_penalties(self):
        """Tuning is a property of the fold and the rows, not of where the cell ran.

        The selected penalty is the one place a re-dispatch could silently retune,
        and a retuned refit would no longer decompose the admitted number.
        """

        earlier = ROOT / 'archive/logs/R5/gate_retrieval_20260923/stability_wave1'
        if not earlier.exists() or not any(earlier.glob('refit_*.json')):
            self.skipTest('no second dispatch on this host')
        checked = 0
        for path, record in self.records():
            other = earlier / path.name
            if not other.exists():
                continue
            twin = json.loads(other.read_text())
            for seed in record['seeds']:
                left = {fold['fold']: fold['alpha'] for fold in record['seeds'][seed]['folds']}
                right = {fold['fold']: fold['alpha'] for fold in twin['seeds'][seed]['folds']}
                self.assertEqual(left, right, f'{path.name}:{seed}')
                self.assertEqual(record['seeds'][seed]['fold_identity_sha256'],
                                 twin['seeds'][seed]['fold_identity_sha256'])
                checked += 1
        self.assertGreater(checked, 0)

    def test_held_groups_never_enter_their_own_training_side(self):
        for path, record in self.records():
            for seed, entry in record['seeds'].items():
                held = [set(fold['held_groups']) for fold in entry['folds']]
                self.assertEqual(len(held), 5, path.name)
                union = set().union(*held)
                self.assertEqual(len(union), record['groups'], f'{path.name}:{seed}')
                self.assertEqual(sum(len(fold) for fold in held), record['groups'],
                                 f'{path.name}:{seed}')


class ReferenceRefusal(unittest.TestCase):
    """The refit's agreement check has to be able to refuse."""

    def fixture(self):
        entry = {'row_identity_sha256': 'a' * 64, 'groups': 64, 'cycles': 8192,
                 'site_pairs': 217, 'states': 12977, 'design_dimensions': {'C': 864}}
        seeds = {'20260923': {
            'fold_identity_sha256': 'b' * 64,
            'folds': [{'fold': 0, 'held_groups': ['g'], 'alpha': {'C': 1.0}}],
            'increments': {'C|ADDITIVE_NULL': {'mse_reduction_kcal2': {'point': 0.5}}}}}
        reference = {'supports': {'all': {**entry, 'seeds': {'20260923': {
            'fold_identity_sha256': 'b' * 64,
            'folds': [{'fold': 0, 'held_groups': ['g'], 'alpha': {'C': 1.0},
                       'dimensions': {'C': 864}}],
            'increments': {'C|ADDITIVE_NULL': {'mse_reduction_kcal2': {'point': 0.5}}}}}}}}
        return entry, seeds, reference

    def test_a_state_level_exclusion_is_refused_by_the_cycle_level_consumer(self):
        from scripts.capability.interactions.fit_retrieval_strata_pairwise import retained_rows
        state_record = {'schema': 'pairwise_indel_exclusion_v1', 'states': []}
        with self.assertRaisesRegex(SystemExit, "cycle-level shape.*states"):
            retained_rows({'backgrounds': []}, set(), state_record, {'background': []})

    def test_a_matching_refit_passes(self):
        from scripts.capability.interactions.fit_retrieval_strata_pairwise import reference_checks
        entry, seeds, reference = self.fixture()
        record = reference_checks(entry, reference, seeds, 1e-6)
        self.assertEqual(record['contrasts_checked'], 1)
        self.assertEqual(record['max_absolute_increment_deviation_kcal2'], 0.0)

    def test_divergent_rows_folds_alphas_or_increments_are_refused(self):
        from scripts.capability.interactions.fit_retrieval_strata_pairwise import reference_checks
        for mutate in (
            lambda e, s: e.__setitem__('row_identity_sha256', 'c' * 64),
            lambda e, s: e.__setitem__('site_pairs', 216),
            lambda e, s: e.__setitem__('design_dimensions', {'C': 1}),
            lambda e, s: s['20260923'].__setitem__('fold_identity_sha256', 'd' * 64),
            lambda e, s: s['20260923']['folds'][0].__setitem__('alpha', {'C': 10.0}),
            lambda e, s: s['20260923']['increments']['C|ADDITIVE_NULL'][
                'mse_reduction_kcal2'].__setitem__('point', 0.6),
        ):
            entry, seeds, reference = self.fixture()
            mutate(entry, seeds)
            with self.assertRaises(SystemExit):
                reference_checks(entry, reference, seeds, 1e-6)


@unittest.skipUnless(REPORT.exists(), 'the gate report is not on this host')
class GateReport(unittest.TestCase):
    def setUp(self):
        self.report = json.loads(REPORT.read_text())

    def views(self):
        """Every reported (cohort, contrast, seed) view, on both stability supports."""

        for cohort, keys, residual, unit in (
                ('stability', ('contrasts', 'contrasts_indel_filtered'),
                 'decomposition_residual_kcal2', 'groups'),
                ('readout_anchor', ('contrasts',), 'decomposition_residual', 'clusters')):
            for arm in self.report[cohort].values():
                for key in keys:
                    for contrast in arm[key].values():
                        for entry in contrast.values():
                            yield entry, residual, unit

    def test_every_stratification_decomposes_its_whole_support_estimate(self):
        worst = 0.0
        for entry, residual, _ in self.views():
            for stratum in entry['strata'].values():
                worst = max(worst, abs(stratum[residual]))
        self.assertLess(worst, 1e-10, f'largest decomposition residual {worst:.3e}')

    def test_every_thin_stratum_is_reported_as_unresolved_rather_than_passed(self):
        for entry, _, unit in self.views():
            for stratum in entry['strata'].values():
                for band in stratum['bands'].values():
                    if band[unit] < MINIMUM_BOOTSTRAP_UNITS:
                        self.assertEqual(band['resolution'], 'unresolved_thin_support')
                        self.assertIsNone(band['interval'])

    def test_the_indel_filter_is_reported_beside_the_unfiltered_support(self):
        for arm, entry in self.report['stability'].items():
            exclusion = entry['indel_exclusion']
            self.assertEqual(exclusion['excluded_cycles'], 130, arm)
            self.assertEqual(exclusion['groups_lost_entirely'], ['nat-014'], arm)
            self.assertEqual(exclusion['retained_support'],
                             {'groups': 63, 'site_pairs': 211, 'cycles': 8062}, arm)
            self.assertIn('reproduction gate runs first', exclusion['order'])
            for contrast in entry['contrasts']:
                self.assertIn(contrast, entry['contrasts_indel_filtered'], arm)
            for seed in ('20260923', '20260924', '20260925'):
                unfiltered = entry['contrasts'][
                    'C_G_T_M1|C_G_T'][seed]['full_support']
                filtered = entry['contrasts_indel_filtered'][
                    'C_G_T_M1|C_G_T'][seed]['full_support']
                self.assertEqual(unfiltered['groups'], 64, arm)
                self.assertEqual(filtered['groups'], 63, arm)

    def test_the_arm_selection_is_recorded_as_targeted_follow_up(self):
        for arm, entry in self.report['stability'].items():
            self.assertIn('selected because it carries the finding under test',
                          entry['arm_selection'], arm)

    def test_the_readout_strata_are_read_off_the_published_fit(self):
        for arm in self.report['readout_anchor'].values():
            for contrast in arm['contrasts'].values():
                for entry in contrast.values():
                    self.assertEqual(entry['full_support']['point'], entry['published_point'])

    def test_the_provenance_sensitivity_is_reported_beside_the_primary_rule(self):
        for arm in self.report['readout_anchor'].values():
            for contrast in arm['contrasts'].values():
                for entry in contrast.values():
                    sensitivity = entry['source_exclusion_sensitivity']
                    self.assertEqual(sensitivity['rule'], rs.SOURCE_EXCLUSION['sensitivity'])
                    primary = entry['strata']['source_provenance']['bands']['source_other']
                    self.assertEqual(primary['clusters'], 99)
                    self.assertEqual(sensitivity['clusters'], 101)
                    self.assertTrue(math.isfinite(sensitivity['point']))


if __name__ == '__main__':
    unittest.main()
