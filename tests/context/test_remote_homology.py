"""The remote-homology gate's endpoint, its identity strata and its self-calibration.

The tests that matter most here are the ones about the channel disagreement and
the three outcomes. This endpoint's two protease channels disagree at a
level-scale root-mean-square of 0.824 kcal/mol against a combined spread of
1.251, with a systematic mean offset of about -0.48 -- a floor two thirds of the
endpoint's own variation. The whole design rests on two claims about that: a
constant offset cancels in the within-background difference the endpoint takes,
and whatever floor remains, a remote null is uninterpretable unless the close
stratum says the pipeline can resolve anything at all. Both are tested as
properties of the code rather than described in a document.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.context import remote_homology as rh  # noqa: E402
from src.capability.context.homology import STRATUM_NAMES  # noqa: E402
from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS  # noqa: E402

ENTRY_POINTS = (
    'qualify_remote_homology_endpoint.py',
    'declare_remote_homology_support.py',
)


from scripts.capability.entrypoints import stage_path

class EntryPointsResolve(unittest.TestCase):
    def test_every_entry_point_imports_and_answers_help(self):
        for filename in ENTRY_POINTS:
            with self.subTest(filename):
                path = stage_path(REPO_ROOT, filename)
                spec = importlib.util.spec_from_file_location(f'_entry_{path.stem}', path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                self.assertTrue(callable(module.main))
                completed = subprocess.run(
                    [sys.executable, str(path), '--help'], capture_output=True, text=True,
                    cwd=REPO_ROOT,
                    env={'PATH': '/usr/bin:/bin', 'OMP_NUM_THREADS': '4',
                         'PYTHONPATH': str(REPO_ROOT), 'HOME': str(Path.home())})
                self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])

    def test_every_name_the_entry_points_import_from_the_module_is_defined(self):
        for filename in ENTRY_POINTS:
            wanted = set()
            source = (stage_path(REPO_ROOT, filename)).read_text()
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom) and node.module and (
                        node.module.endswith('remote_homology')):
                    wanted.update(alias.name for alias in node.names)
            with self.subTest(filename):
                self.assertTrue(wanted)
                self.assertEqual(sorted(n for n in wanted if not hasattr(rh, n)), [])

    def test_declared_exports_resolve(self):
        self.assertEqual([name for name in rh.__all__ if not hasattr(rh, name)], [])


class SourceReading(unittest.TestCase):
    def _table(self, directory: Path, *, drop_a_field: bool = False,
               short_row: bool = False) -> Path:
        header = list(rh.SOURCE_COLUMNS) + ['extra']
        if drop_a_field:
            header.remove('deltaG_c_95CI')
        rows = [','.join(header),
                ','.join(['bg', 'dms7', 'ACDE', 'True', 'False'] + ['1.0'] * (len(header) - 5))]
        if short_row:
            rows.append('bg_A2G,dms7')
        path = directory / 'table.csv'
        path.write_text('\n'.join(rows) + '\n')
        return path

    def test_a_missing_declared_column_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._table(Path(directory), drop_a_field=True)
            with self.assertRaises(ValueError):
                rh.read_source(path)

    def test_a_truncated_row_is_dropped_rather_than_padded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._table(Path(directory), short_row=True)
            records = rh.read_source(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][0], 'bg')


class BackgroundRule(unittest.TestCase):
    def test_the_longest_proper_prefix_that_is_itself_a_name_wins(self):
        names = {'rocklin_batch2_0', 'rocklin_batch2_0_A5G', 'rocklin_batch2_550891',
                 'orphan_name_with_no_parent'}
        roots = rh.background_map(names)
        self.assertEqual(roots['rocklin_batch2_0_A5G'], 'rocklin_batch2_0')
        self.assertEqual(roots['rocklin_batch2_0'], 'rocklin_batch2_0')
        self.assertEqual(roots['rocklin_batch2_550891'], 'rocklin_batch2_550891')
        self.assertEqual(roots['orphan_name_with_no_parent'], 'orphan_name_with_no_parent')

    def test_a_longer_prefix_beats_a_shorter_one(self):
        names = {'a', 'a_b', 'a_b_c'}
        roots = rh.background_map(names)
        self.assertEqual(roots['a_b_c'], 'a_b')
        self.assertEqual(roots['a_b'], 'a')


class AdmissionRule(unittest.TestCase):
    def test_the_boundary_is_admitted_and_everything_outside_it_is_not(self):
        floor = rh.QC_WIDTH_KCAL_MOL
        self.assertTrue(rh.admitted((str(floor), '0.0', '0.25')))
        self.assertFalse(rh.admitted((str(floor + 1e-9), '0.0', '0.25')))
        self.assertFalse(rh.admitted(('-0.1', '0.0', '0.25')))

    def test_an_unparseable_width_is_not_admitted_and_is_not_treated_as_missing(self):
        for value in ('', 'NA', 'nan', 'inf', None):
            with self.subTest(value):
                self.assertFalse(rh.admitted((value, '0.1', '0.1')))


class SubstitutionVerification(unittest.TestCase):
    def test_a_consistent_token_resolves_to_its_own_position(self):
        self.assertEqual(rh.substitution_at('ACDEFG', 'AGDEFG', 'C2G'), 1)

    def test_every_disagreement_is_refused(self):
        cases = {
            'wrong wild residue': ('ACDEFG', 'AGDEFG', 'D2G'),
            'wrong mutant residue': ('ACDEFG', 'AGDEFG', 'C2W'),
            'length mismatch': ('ACDEFG', 'AGDEF', 'C2G'),
            'two substitutions': ('ACDEFG', 'AGDWFG', 'C2G'),
            'position past the end': ('ACDEFG', 'ACDEFG', 'G9A'),
            'not a substitution token': ('ACDEFG', 'ACDEFG', 'ins3A'),
            'stop token': ('ACDEFG', 'ACDEFG', 'C2*'),
        }
        for label, (background, variant, token) in cases.items():
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    rh.substitution_at(background, variant, token)


class DeclaredDraw(unittest.TestCase):
    def test_the_draw_reads_names_only_and_is_reproducible(self):
        universe = [f'bg{index:04d}' for index in range(100)]
        first = rh.declared_draw(universe, size=20, seed=7)
        self.assertEqual(len(first), 20)
        self.assertEqual(first, sorted(first))
        self.assertEqual(first, rh.declared_draw(list(reversed(universe)), size=20, seed=7))
        self.assertNotEqual(first, rh.declared_draw(universe, size=20, seed=8))

    def test_a_draw_larger_than_the_universe_is_refused(self):
        with self.assertRaises(ValueError):
            rh.declared_draw(['a', 'b'], size=3, seed=1)

    def test_the_declared_size_and_seed_are_the_banded_ones(self):
        # Changing either detaches this cohort's identity bands from the frozen
        # search that measured them, which is why they are constants and not flags.
        self.assertEqual(rh.BACKGROUND_DRAW, 320)
        self.assertEqual(rh.BACKGROUND_DRAW_SEED, 20260924)


class ChannelDisagreement(unittest.TestCase):
    """The claim the whole design rests on: a constant offset cancels in a difference."""

    def test_a_pure_constant_offset_is_all_offset_and_no_discordance(self):
        first = np.array([1.0, 2.0, 3.0, 4.0])
        second = first - 0.5
        record = rh.channel_decomposition(rh.channel_moments(first, second))
        self.assertAlmostEqual(record['channel_offset_mean_kcal_mol'], 0.5)
        self.assertAlmostEqual(record['per_channel_discordance_sd_kcal_mol'], 0.0)
        self.assertAlmostEqual(record['pearson_r'], 1.0)

    def test_the_discordance_is_the_difference_spread_over_root_two(self):
        generator = np.random.default_rng(0)
        first = generator.standard_normal(4000)
        second = first + 0.3 + 0.2 * generator.standard_normal(4000)
        record = rh.channel_decomposition(rh.channel_moments(first, second))
        difference = first - second
        self.assertAlmostEqual(record['per_channel_discordance_sd_kcal_mol'],
                               float(difference.std()) / np.sqrt(2.0), places=6)
        self.assertAlmostEqual(record['channel_offset_mean_kcal_mol'],
                               float(difference.mean()), places=6)

    def test_taking_a_within_background_difference_removes_the_offset(self):
        # Two channels whose levels differ by a constant, differenced against the
        # background's own level in each channel: the offset is gone and the
        # discordance is unchanged. This is why the effect-scale floor and not the
        # level-scale one bounds this gate.
        generator = np.random.default_rng(1)
        background = generator.standard_normal(500) + 2.0
        noise = 0.2 * generator.standard_normal(500)
        variant_first = background - 0.7 + noise
        variant_second = variant_first - 0.5
        level = rh.channel_decomposition(rh.channel_moments(
            variant_first, variant_second))
        effect = rh.channel_decomposition(rh.channel_moments(
            variant_first - background, variant_second - (background - 0.5)))
        self.assertAlmostEqual(level['channel_offset_mean_kcal_mol'], 0.5, places=6)
        self.assertAlmostEqual(effect['channel_offset_mean_kcal_mol'], 0.0, places=6)
        self.assertGreater(level['channel_difference_rms_kcal_mol'],
                           effect['channel_difference_rms_kcal_mol'])

    def test_a_shared_component_is_an_upper_bound_and_is_never_negative(self):
        generator = np.random.default_rng(2)
        first = generator.standard_normal(200)
        second = -first  # perfectly anticorrelated: covariance is negative
        record = rh.channel_decomposition(rh.channel_moments(first, second))
        self.assertEqual(record['shared_component_sd_kcal_mol'], 0.0)

    def test_the_default_keys_are_this_gate_own_channels_and_unit(self):
        # The already-published qualification artefact carries these key names,
        # so the default path has to keep reproducing them exactly.
        record = rh.channel_decomposition(rh.channel_moments(
            np.array([1.0, 2.0]), np.array([0.5, 1.5])))
        self.assertIn('mean_trypsin_kcal_mol', record)
        self.assertIn('per_channel_discordance_sd_kcal_mol', record)
        self.assertEqual(rh.CHANNEL_NAMES, ('trypsin', 'chymotrypsin'))
        self.assertEqual(rh.CHANNEL_UNIT, 'kcal_mol')

    def test_another_channel_pair_publishes_under_its_own_unit(self):
        # The Domainome endpoint reads the same decomposition over two biological
        # replicates in log2 enrichment. Publishing that under kcal/mol would be a
        # quantity reported in another quantity's unit.
        record = rh.channel_decomposition(
            rh.channel_moments(np.array([1.0, 2.0]), np.array([0.5, 1.5])),
            channels=('replicate_1', 'replicate_2'), unit='log2_enrichment')
        self.assertIn('mean_replicate_1_log2_enrichment', record)
        self.assertIn('per_channel_discordance_sd_log2_enrichment', record)
        self.assertNotIn('per_channel_discordance_sd_kcal_mol', record)
        interval = rh.channel_interval(
            [rh.channel_moments(np.array([0.1 * n, 0.2 * n]),
                                np.array([0.1 * n + 0.05, 0.2 * n - 0.05]))
             for n in range(1, MINIMUM_BOOTSTRAP_UNITS + 1)],
            channels=('replicate_1', 'replicate_2'), unit='log2_enrichment',
            draws=200, seed=0)
        self.assertEqual(interval['unit'], 'log2_enrichment')
        self.assertEqual(interval['channels'], ['replicate_1', 'replicate_2'])
        self.assertIn('per_channel_discordance_sd_log2_enrichment', interval)

    def test_misaligned_or_empty_channels_are_refused(self):
        with self.assertRaises(ValueError):
            rh.channel_moments(np.array([1.0]), np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            rh.channel_moments(np.array([]), np.array([]))


class ChannelIntervalRespectsTheSharedFloor(unittest.TestCase):
    def _vectors(self, units: int):
        return [rh.channel_moments(np.array([0.1 * n, 0.2 * n]),
                                   np.array([0.1 * n + 0.05, 0.2 * n - 0.05]))
                for n in range(1, units + 1)]

    def test_below_the_floor_the_points_survive_and_no_interval_is_published(self):
        record = rh.channel_interval(self._vectors(MINIMUM_BOOTSTRAP_UNITS - 1),
                                     draws=100, seed=0)
        self.assertTrue(record['degenerate'])
        self.assertIn(f'below the {MINIMUM_BOOTSTRAP_UNITS}-', record['degenerate_reason'])
        entry = record['per_channel_discordance_sd_kcal_mol']
        self.assertIn('point', entry)
        self.assertNotIn('ci95', entry)

    def test_at_the_floor_an_interval_is_published_and_brackets_its_point(self):
        record = rh.channel_interval(self._vectors(MINIMUM_BOOTSTRAP_UNITS),
                                     draws=400, seed=0)
        self.assertFalse(record['degenerate'])
        self.assertEqual(record['resampling_unit'],
                         'family group of the wild-type background')
        entry = record['shared_component_sd_kcal_mol']
        self.assertLessEqual(entry['ci95'][0], entry['ci95'][1])

    def test_an_empty_population_is_refused_rather_than_reported(self):
        with self.assertRaises(ValueError):
            rh.channel_interval([], draws=10, seed=0)


class Grouping(unittest.TestCase):
    def test_identical_sequences_group_and_unrelated_ones_stay_singletons(self):
        wildtypes = {
            'a': 'ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY',
            'b': 'ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY',
            'c': 'KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK',
        }
        labels, report = rh.family_groups(wildtypes)
        self.assertEqual(labels['a'], labels['b'])
        self.assertNotEqual(labels['a'], labels['c'])
        self.assertEqual(report['groups'], 2)
        self.assertEqual(report['union_sources'], ['exact_sequence', 'alignment'])
        self.assertIn('30.0', report['edge_rule'])


class IdentityBands(unittest.TestCase):
    def test_no_reported_hit_bands_at_the_lowest_stratum(self):
        bands = rh.query_bands({'a': None, 'b': 12.5, 'c': 55.0, 'd': 80.0, 'e': 99.0})
        self.assertEqual(bands['a'], STRATUM_NAMES[0])
        self.assertEqual(bands['b'], STRATUM_NAMES[0])
        self.assertEqual(bands['c'], STRATUM_NAMES[1])
        self.assertEqual(bands['d'], STRATUM_NAMES[2])
        self.assertEqual(bands['e'], STRATUM_NAMES[3])

    def test_identity_is_the_matched_share_of_the_query_and_the_best_hit_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'hits.tsv'
            path.write_text(
                'a\ts1\t25\t100\t90.0\t30\t1e-9\n'
                'a\ts2\t60\t100\t70.0\t80\t1e-20\n'
                'b\ts3\t0\t0\t0.0\t0\t1.0\n'
                'short\n')
            best = rh.best_identity(path, ['a', 'b', 'c'])
        self.assertAlmostEqual(best['a'], 60.0)
        self.assertIsNone(best['b'])
        self.assertIsNone(best['c'])


class GroupStrata(unittest.TestCase):
    def _labels(self):
        return {'u1': 'g1', 'u2': 'g1', 'u3': 'g2', 'u4': 'g3'}

    def test_a_group_is_close_only_when_every_member_is(self):
        bands = {'u1': STRATUM_NAMES[3], 'u2': STRATUM_NAMES[2],
                 'u3': STRATUM_NAMES[1], 'u4': STRATUM_NAMES[0]}
        assignment, report = rh.group_strata(self._labels(), bands)
        self.assertEqual(assignment, {'g1': 'close', 'g2': 'remote', 'g3': 'remote'})
        self.assertEqual(report['close_groups'], 1)
        self.assertEqual(report['remote_groups'], 2)
        self.assertEqual(report['mixed_groups'], 0)

    def test_a_group_that_straddles_the_boundary_enters_neither_stratum(self):
        bands = {'u1': STRATUM_NAMES[3], 'u2': STRATUM_NAMES[0],
                 'u3': STRATUM_NAMES[1], 'u4': STRATUM_NAMES[2]}
        assignment, report = rh.group_strata(self._labels(), bands)
        self.assertEqual(assignment['g1'], 'mixed')
        self.assertEqual(report['mixed_groups'], 1)

    def test_a_background_with_no_band_is_refused(self):
        with self.assertRaises(ValueError):
            rh.group_strata(self._labels(), {'u1': STRATUM_NAMES[0]})

    def test_the_unit_floor_is_reported_per_stratum(self):
        labels = {f'u{index}': f'g{index}' for index in range(rh.STRATUM_UNIT_FLOOR)}
        bands = {name: STRATUM_NAMES[1] for name in labels}
        _, report = rh.group_strata(labels, bands)
        self.assertTrue(report['remote_clears_the_unit_floor'])
        self.assertFalse(report['close_clears_the_unit_floor'])
        self.assertFalse(rh.unit_floor_cleared(rh.STRATUM_UNIT_FLOOR - 1))
        self.assertTrue(rh.unit_floor_cleared(rh.STRATUM_UNIT_FLOOR))


class StratumReadout(unittest.TestCase):
    def test_the_mask_selects_only_that_stratum_and_refuses_an_unknown_one(self):
        group = np.array(['g1', 'g1', 'g2', 'g3'])
        assignment = {'g1': 'close', 'g2': 'remote', 'g3': 'mixed'}
        np.testing.assert_array_equal(rh.stratum_keep(group, assignment, 'close'),
                                      np.array([True, True, False, False]))
        np.testing.assert_array_equal(rh.stratum_keep(group, assignment, 'remote'),
                                      np.array([False, False, True, False]))
        with self.assertRaises(ValueError):
            rh.stratum_keep(group, assignment, 'somewhere_else')


class ThreeOutcomes(unittest.TestCase):
    def test_each_combination_selects_a_declared_outcome(self):
        self.assertEqual(rh.outcome(True, True), rh.OUTCOME_SURVIVES)
        self.assertEqual(rh.outcome(True, False), rh.OUTCOME_HOMOLOGY_DEPENDENT)
        self.assertEqual(rh.outcome(False, False), rh.OUTCOME_UNRESOLVED)

    def test_a_remote_result_whose_positive_control_did_not_fire_is_unresolved(self):
        # The close stratum gates the reading, so remote-only is not survival.
        self.assertEqual(rh.outcome(False, True), rh.OUTCOME_UNRESOLVED)
        record = rh.outcome_record(False, True)
        self.assertEqual(record['outcome'], rh.OUTCOME_UNRESOLVED)
        self.assertFalse(record['positive_control_fired'])
        self.assertTrue(record['remote_resolved_without_its_positive_control'])
        licensed = rh.outcome_record(True, True)
        self.assertEqual(licensed['outcome'], rh.OUTCOME_SURVIVES)
        self.assertFalse(licensed['remote_resolved_without_its_positive_control'])

    def test_the_unresolved_outcome_refuses_to_call_itself_a_null(self):
        self.assertIn('unresolved', rh.OUTCOME_UNRESOLVED)
        self.assertIn('not null', rh.OUTCOME_UNRESOLVED)
        self.assertNotIn('no effect', rh.OUTCOME_UNRESOLVED)


class DeclaredScopeAndConstants(unittest.TestCase):
    def test_the_endpoint_names_its_scale_its_unit_and_its_sign(self):
        self.assertIn('deltaG', rh.ENDPOINT)
        self.assertIn('kcal/mol', rh.ENDPOINT)
        self.assertIn('negative', rh.ENDPOINT)

    def test_the_measured_quantity_records_the_shared_assay_technology(self):
        # This endpoint may not be read as independent replication of the
        # development stability axis, and the reason travels with the quantity.
        self.assertIn('Tsuboyama', rh.MEASURED_QUANTITY)
        self.assertIn('same assay technology', rh.MEASURED_QUANTITY)

    def test_no_variant_cap_is_invented_for_a_support_that_would_not_reach_one(self):
        self.assertEqual(rh.MIN_VARIANTS, 8)
        self.assertFalse(hasattr(rh, 'VARIANT_CAP'))

    def test_the_bands_are_the_frozen_declaration_and_split_at_seventy_percent(self):
        self.assertEqual(rh.BANDS, STRATUM_NAMES)
        self.assertEqual(set(rh.REMOTE_BANDS) | set(rh.CLOSE_BANDS), set(STRATUM_NAMES))
        self.assertEqual(set(rh.REMOTE_BANDS) & set(rh.CLOSE_BANDS), set())


if __name__ == '__main__':
    unittest.main()
