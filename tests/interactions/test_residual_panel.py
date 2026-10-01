"""Simultaneous uncertainty, the declared support and the indel restriction."""
import json
import unittest
from pathlib import Path

import numpy as np

from scripts.capability.interactions import run_residual_panel
from scripts.capability.interactions.run_residual_panel import simultaneous_bands
from src.capability.core.io import sha256_file


class SimultaneousBands(unittest.TestCase):
    def test_duplicate_contrast_is_not_an_independent_sample(self):
        x = np.random.default_rng(7).normal(size=(64, 1))
        single = simultaneous_bands(x, draws=500)
        duplicate = simultaneous_bands(np.repeat(x, 33, axis=1), draws=500)
        np.testing.assert_allclose(duplicate['interval'], np.repeat(single['interval'], 33, axis=0), atol=1e-15)

    def test_family_critical_value_covers_every_marginal_critical_value(self):
        x = np.random.default_rng(8).normal(size=(64, 4))
        joint = simultaneous_bands(x, draws=500)
        marginal = [simultaneous_bands(x[:, i:i+1], draws=500) for i in range(4)]
        self.assertTrue(all(joint['critical_value'] >= r['critical_value'] for r in marginal))
        for bad in (np.array([1, 2]), np.array([[np.nan], [1]]), np.array([[1]])):
            with self.assertRaises(ValueError):
                simultaneous_bands(bad)

class ChannelSourceContract(unittest.TestCase):
    def test_ordered_shards_ignore_non_row_sources_but_validate_bytes(self):
        import tempfile
        from pathlib import Path
        from scripts.capability.interactions.diagnose_residual_channels import validated_channel_sources
        from src.capability.core.io import sha256_file
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root/'raw.parquet'
            shard.write_bytes(b'channel fixture')
            cohort = {'source_files': [
                {'path': 'raw.parquet', 'sha256': sha256_file(shard)},
                {'path': 'query_index.json', 'sha256': 'unused-cohort-builder-input'}],
                'source_row_order': ['raw.parquet']}
            self.assertEqual(validated_channel_sources(root, cohort), [shard])
            shard.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'digest'):
                validated_channel_sources(root, cohort)
            cohort['source_row_order'] = ['unbound.parquet']
            with self.assertRaisesRegex(ValueError, 'provenance'):
                validated_channel_sources(root, cohort)


class DeclaredSupport(unittest.TestCase):
    """A product is bound to the support it names, by digest and by count."""

    frozen = Path('archive/logs/R3/pairwise_cohort_20260924/cohort.json')
    exclusion = Path('results/R3/pairwise_indel_exclusion_20260928/pairwise_indel_exclusion.json')

    def cohort(self, backgrounds):
        return {'backgrounds': [
            {'name': name, 'group': group,
             'cycles': [{'sequences': list(states), 'positions': list(positions)}
                        for states, positions in cycles]}
            for name, group, cycles in backgrounds]}

    def declaration(self, states):
        return {'schema': 'pairwise_indel_exclusion_v1', 'states': [
            {'background': background, 'sequence': sequence,
             'value_without_indel_rows_kcal_mol': value}
            for background, sequence, value in states]}

    def test_a_state_with_no_remaining_row_takes_its_cycles_and_their_site_pairs_out(self):
        cohort = self.cohort([
            ('one', 'g1', [(('wt', 'a', 'b', 'ab'), (1, 2)), (('wt', 'a', 'c', 'ac'), (1, 3))]),
            ('two', 'g2', [(('wt', 'a', 'b', 'ab'), (1, 2))])])
        self.assertEqual(run_residual_panel.support_counts(cohort, None),
                         {'groups': 2, 'cycles': 3, 'site_pairs': 3})
        emptied = self.declaration([('one', 'b', None)])
        self.assertEqual(run_residual_panel.support_counts(cohort, emptied),
                         {'groups': 2, 'cycles': 2, 'site_pairs': 2})
        # A corrected median moves a target; it never moves the support.
        moved = self.declaration([('one', 'b', 1.5)])
        self.assertEqual(run_residual_panel.support_counts(cohort, moved),
                         run_residual_panel.support_counts(cohort, None))
        # A background that loses every cycle loses its group with them.
        self.assertEqual(run_residual_panel.support_counts(
            cohort, self.declaration([('two', 'a', None)])),
            {'groups': 1, 'cycles': 2, 'site_pairs': 2})

    def test_a_cycle_level_record_is_refused_by_the_state_level_consumer(self):
        cohort = self.cohort([('one', 'g1', [(('wt', 'a', 'b', 'ab'), (1, 2))])])
        with self.assertRaisesRegex(ValueError, "state-level shape.*excluded_cycles"):
            run_residual_panel.support_counts(cohort, {'schema': 'pairwise_indel_exclusion_v1', 'excluded_cycles': {}, 'summary': {}})

    @unittest.skipUnless(frozen.exists() and exclusion.exists(),
                         'the frozen cohort and its exclusion are held locally, not distributed')
    def test_each_declared_support_is_the_one_the_pinned_bytes_realise(self):
        cohort = json.loads(self.frozen.read_text())
        self.assertEqual(sha256_file(self.frozen), run_residual_panel.COHORT_SHA256)
        declared = run_residual_panel.SUPPORTS
        self.assertEqual(run_residual_panel.support_counts(cohort, None),
                         declared['all']['counts'])
        exclusion = json.loads(self.exclusion.read_text())
        self.assertEqual(exclusion['cohort_sha256'], run_residual_panel.COHORT_SHA256)
        self.assertEqual(run_residual_panel.support_counts(cohort, exclusion),
                         declared['indel-excluded']['counts'])
        # Negative path: a support that does not realise its declared counts is a
        # different support, and the declared numbers must say so.
        fabricated = json.loads(self.exclusion.read_text())
        first = cohort['backgrounds'][0]
        fabricated['states'].append({'background': first['name'],
                                     'sequence': first['cycles'][0]['sequences'][3],
                                     'value_without_indel_rows_kcal_mol': None})
        self.assertNotEqual(run_residual_panel.support_counts(cohort, fabricated),
                            declared['indel-excluded']['counts'])

    def test_declare_refuses_a_cohort_that_is_not_the_declared_bytes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in run_residual_panel.INPUTS.values():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(
                    {'backgrounds': [], 'cohort_sha256': '0' * 64,
                     'provenance': {'baseline_q': {'sha256': '0' * 64}}}))
            with self.assertRaisesRegex(ValueError, 'declared digest'):
                run_residual_panel.declare(root, root / 'out', 'all', None)
