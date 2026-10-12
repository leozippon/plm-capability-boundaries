"""Conditions that must hold for retrospective measured-label selection.

The selection stage re-ranks already-frozen held-out predictions, so almost
everything that can go wrong is a silent arithmetic or join error rather than a
crash. These tests pin the properties that a happy-path run would not show.

* **Orientation.** A positive measured ddG is stabilising, so a selector that
  ranks by descending measured ddG must have a positive gain and its mirror must
  have a negative one. A sign inversion is a sign flip, not a plausible effect.
* **Rows, not ids.** ``sample_id`` is ``background:position`` and repeats across
  substitutions at one position, so the row identity is the cohort state key. A
  dict keyed by ``sample_id`` would collapse those rows, and the positional join
  refuses a reordered archive instead.
* **Wild type, not pool.** Selection is inside one wild type and is scored
  against that library's own mean; two libraries on different scales must keep
  their own baselines.
* **Seeds, not units.** The three split seeds are averaged inside each wild type
  before the family-group bootstrap.
* **Groups, not rows.** A resampled group drawn twice contributes twice and is
  never deduplicated, and the common draw keeps paired columns paired.
* **Support, not optimism.** Below the eight-group floor no interval is
  published; a hash or allowlist mismatch stops the run rather than selecting.

Substitution-level quantities come from tiny synthetic libraries; the CLI test
builds a synthetic frozen panel and runs the real entry point in a subprocess.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.io import sha256_file, write_json
from src.capability.interactions.pairwise_epistasis import ROSTER, SPLIT_SEEDS
from src.capability.stability.ranking_error import CONTRASTS
from src.capability.stability import measured_selection as selection
from src.capability.stability.measured_selection import (
    REQUIRED_ROLES, SELECTION_DESIGNS)

_SPEC = importlib.util.spec_from_file_location(
    'analyse_measured_selection_test',
    ROOT / 'scripts/capability/stability/analyse_measured_selection.py')
assert _SPEC is not None and _SPEC.loader is not None
stage = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stage)

_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'

#: The frozen E03 measured-selection panel, in the published layout. Absent on a
#: checkout without the result tree, which is a reason to skip the byte-level legacy
#: replay rather than to weaken it.
FROZEN_E03_PANEL = (ROOT / 'results/external_baseline/20261008095614_7e8ae4b18a4d'
                    / 'stability_ranking_error_grid_cpu/panel.json')


# --------------------------------------------------------------------------- #
# Tiny synthetic libraries and a synthetic frozen panel.
# --------------------------------------------------------------------------- #

def synthetic_cohort(backgrounds: int, variants: int, *, seed: int = 7,
                     values=None) -> dict:
    """A minimal cohort in the frozen shape: wild types of single substitutions.

    Each variant carries its own ``state`` index into its background's sequence
    list, so several substitutions at one position remain distinct rows while
    their ``background:position`` sample id repeats. Measured values are either
    supplied or drawn, and the default orders a wild type's values so a selector
    that ranks by measured ddG has a real gain to find.
    """

    rng = np.random.default_rng(seed)
    rows = []
    for index in range(backgrounds):
        name = f'syn{index:03d}.pdb'
        wildtype = _ALPHABET
        sequences = [wildtype]
        record = []
        for variant in range(variants):
            position = (variant % len(wildtype)) + 1
            mutant = _ALPHABET[(variant + 1) % len(_ALPHABET)]
            sequence = wildtype[:position - 1] + mutant + wildtype[position:]
            if values is None:
                ddg = float(np.linspace(-5.0, 5.0, variants)[variant] + rng.normal(scale=0.05))
            else:
                ddg = float(values[variant])
            record.append({'position': position, 'state': variant + 1, 'mutant': mutant,
                           'sequence': sequence, 'rows': 1, 'distinct_dna': 1,
                           'ddg': ddg, 'ddg_trypsin': ddg, 'ddg_chymotrypsin': ddg})
            sequences.append(sequence)
        rows.append({'name': name, 'group': f'nat-{index:03d}', 'cluster': str(index),
                     'length': len(wildtype), 'wildtype': wildtype,
                     'wildtype_combined_kcal_mol': 1.0, 'sequences': sequences,
                     'variants': record, 'sites': sorted({v['position'] for v in record}),
                     'eligible_variants': 10_000})
    return {'schema': 'stability_singles_cohort_v1', 'endpoint_sha256': 'e' * 64,
            'backgrounds': rows}


def _flattened_measured(cohort: dict) -> np.ndarray:
    return np.asarray([variant['ddg'] for row in cohort['backgrounds']
                       for variant in row['variants']], dtype=float)


def _write_allowlist(root: Path, cohort: dict, cohort_path: Path, *,
                     names=None) -> Path:
    """A valid allowlist over the synthetic cohort, with real file hashes."""

    names = names or [row['name'] for row in cohort['backgrounds']]
    sources = root / 'structures'
    sources.mkdir(parents=True, exist_ok=True)
    entries = []
    for row in cohort['backgrounds']:
        if row['name'] not in names:
            continue
        source = sources / f"{row['name']}.cif"
        source.write_text(f"data_{row['name']}\n")
        review_path = sources / f"{row['name']}.review.json"
        write_json(review_path, {
            'schema': 'measured_construct_review_v1', 'name': row['name'],
            'source': {'path': str(source), 'sha256': sha256_file(source)},
            'reference_construct_sha256': hashlib.sha256(row['wildtype'].encode()).hexdigest(),
            'match_rule': 'exact_or_contiguous_full_construct',
            'canonical': {'entity_id': '1',
                          'match_range_1based_inclusive': [1, len(row['wildtype'])]},
            'resolved': {'label_asym_id': 'A', 'model': 1,
                         'match_range_1based_inclusive': [1, len(row['wildtype'])]},
        })
        entries.append({
            'construct_review': {'path': str(review_path), 'sha256': sha256_file(review_path)},
            'name': row['name'], 'group': row['group'], 'protein': 'synthetic domain',
            'organism': 'synthetic', 'function': 'synthetic non-hazardous fixture',
            'construct_match': {'matched': True, 'method': 'exact synthetic match'},
            'sources': [{'path': str(source), 'sha256': sha256_file(source)}],
            'provenance': [{'path': str(cohort_path), 'sha256': sha256_file(cohort_path)}],
        })
    payload = {'schema': selection.ALLOWLIST_SCHEMA,
               'review': {'protocol': 'synthetic protocol', 'reviewed_utc': '2026-10-11'},
               'backgrounds': entries}
    path = root / 'allowlist.json'
    write_json(path, payload)
    return path


def _write_roles(root: Path, *, panel_dir=None, cohort_path=None) -> Path:
    mapping = dict(zip(REQUIRED_ROLES, ('prollama-stage-1', 'prollama', 'protgpt2',
                                        'progen2-xlarge')))
    path = root / 'roles.json'
    protocol = {f'{name}_sha256': 'a' * 64
                for name in ('completion', 'panel', 'cohort', 'allowlist')}
    if panel_dir is not None:
        assert cohort_path is not None
        protocol = {'completion_sha256': sha256_file(panel_dir / 'completion.json'),
                    'panel_sha256': sha256_file(panel_dir / 'panel.json'),
                    'cohort_sha256': sha256_file(cohort_path),
                    'allowlist_sha256': sha256_file(root / 'allowlist.json')}
    write_json(path, {'schema': selection.ROLES_SCHEMA, 'protocol': protocol,
                      'rationale': 'synthetic frozen roles', 'roles': mapping})
    return path


def make_panel(root: Path, cohort: dict, *, arms=ROSTER,
               informative: bool = True, allowed_names=None, contrasts=None) -> dict:
    """A synthetic completion directory that the real CLI must accept.

    ``contrasts`` overrides the published contrast definitions, so a test can
    replay a panel whose explanatory metadata predates the current correction.
    """

    panel_dir = root / 'stability_ranking_error_grid_cpu'
    (panel_dir / 'fits').mkdir(parents=True)
    (panel_dir / 'oof').mkdir(parents=True)
    cohort_path = root / 'cohort.json'
    write_json(cohort_path, cohort)

    measured = _flattened_measured(cohort)
    rng = np.random.default_rng(11)
    oof_by_arm = {}
    fit_by_arm = {}
    for arm in arms:
        retained = {'sample_id': np.asarray(
                        [f"{row['name']}:{variant['position']}"
                         for row in cohort['backgrounds'] for variant in row['variants']],
                        dtype=str),
                    'group': np.asarray(
                        [row['group'] for row in cohort['backgrounds']
                         for _ in row['variants']], dtype=str),
                    'target': measured.copy()}
        for seed in SPLIT_SEEDS:
            for design in SELECTION_DESIGNS:
                if informative and design in ('S_M', 'S2_M'):
                    vector = measured.copy()
                else:
                    vector = rng.normal(scale=3.0, size=len(measured))
                retained[f'{seed}|{design}'] = np.asarray(vector, dtype=float)
        oof_path = panel_dir / 'oof' / f'oof_{arm}.npz'
        np.savez_compressed(oof_path, **retained)
        fit = {'arm': arm, 'matched_baseline': 'S', 'tokenisation_stratum': 'amino_acid',
               'seeds': {str(seed): {} for seed in SPLIT_SEEDS},
               'held_out_predictions': {'file': oof_path.name,
                                        'sha256': sha256_file(oof_path),
                                        'designs': list(SELECTION_DESIGNS)}}
        fit_path = panel_dir / 'fits' / f'fit_{arm}.json'
        write_json(fit_path, fit)
        oof_by_arm[arm] = sha256_file(oof_path)
        fit_by_arm[arm] = sha256_file(fit_path)

    definitions = CONTRASTS if contrasts is None else contrasts
    write_json(panel_dir / 'panel.json',
               {'schema': 'stability_ranking_error_panel_v2', 'arms': list(arms),
                'contrast_definitions': definitions, 'blocked_contrasts': [],
                'families': {name: {'status': 'complete', 'columns': list(arms)}
                             for name in definitions},
                'support': {'cohort_sha256': sha256_file(cohort_path),
                            'endpoint_sha256': cohort['endpoint_sha256']}})
    write_json(panel_dir / 'channels.json', {'schema': 'stability_channel_robustness_panel_v2'})
    write_json(panel_dir / 'extraction_audit.json', {'schema': 'stability_extraction_audit_v1'})
    write_json(panel_dir / 'completion.json', {
        'schema': stage.COMPLETION_SCHEMA, 'status': 'complete', 'arms': list(arms),
        'artifacts': {name: sha256_file(panel_dir / name)
                      for name in stage.COMPLETION_ARTIFACTS},
        'per_arm_fits': fit_by_arm, 'held_out_predictions': oof_by_arm})
    allowlist = _write_allowlist(root, cohort, cohort_path, names=allowed_names)
    roles = _write_roles(root, panel_dir=panel_dir, cohort_path=cohort_path)
    return {'panel_dir': panel_dir, 'cohort': cohort_path, 'allowlist': allowlist,
            'roles': roles, 'out': root / 'analysis'}


def run_cli(fixture: dict, *, draws: int = 200, allowlist=None, roles=None,
            provenance_root=None):
    command = [sys.executable, str(ROOT / 'scripts/capability/stability/'
                                   'analyse_measured_selection.py'),
               '--panel-dir', str(fixture['panel_dir']), '--cohort', str(fixture['cohort']),
               '--allowlist', str(allowlist or fixture['allowlist']),
               '--roles', str(roles or fixture['roles']),
               '--device', 'cpu', '--out', str(fixture['out']), '--draws', str(draws)]
    if provenance_root is not None:
        command.extend(['--provenance-root', str(provenance_root)])
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True)


def strict_json(path: Path) -> dict:
    text = path.read_text()
    assert 'NaN' not in text and 'Infinity' not in text, 'a non-finite token reached the JSON'
    value = json.loads(text)

    def walk(node):
        if isinstance(node, float):
            assert math.isfinite(node)
        elif isinstance(node, dict):
            for item in node.values():
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(value)
    return value


# --------------------------------------------------------------------------- #
# Orientation, ties and the same-wild-type library.
# --------------------------------------------------------------------------- #

class GainOrientation(unittest.TestCase):
    def setUp(self):
        self.cohort = synthetic_cohort(1, 10, values=np.linspace(-4.0, 4.0, 10))
        self.flat = selection.flatten_cohort(self.cohort)
        self.measured = self.flat['measured']
        self.background = self.flat['backgrounds']
        self.keys = self.flat['state_keys']

    def test_descending_prediction_finds_stabilising_variants(self):
        perfect = selection.library_gains(self.measured, self.measured, self.background,
                                          self.keys, 0.2)
        self.assertGreater(float(perfect['gain'][0]), 0.0)
        self.assertGreater(float(perfect['selected_mean'][0]), float(perfect['library_mean'][0]))
        self.assertAlmostEqual(float(perfect['k'][0]), 2.0)
        # The oracle is the same set when the selector is the measurement itself.
        self.assertAlmostEqual(float(perfect['gain'][0]), float(perfect['oracle_gain'][0]))

    def test_mirrored_prediction_has_the_mirrored_sign(self):
        mirrored = selection.library_gains(self.measured, -self.measured, self.background,
                                           self.keys, 0.2)
        self.assertLess(float(mirrored['gain'][0]), 0.0)

    def test_a_constant_prediction_is_random_expectation(self):
        flat = selection.library_gains(self.measured, np.zeros(10), self.background,
                                       self.keys, 0.2)
        # A constant rank is decided by the state key, so the selected mean is a
        # genuine subset mean and the gain is exactly that subset's deviation --
        # never a pooled quantity.
        selected = flat['selected_mean'][0]
        self.assertAlmostEqual(float(flat['gain'][0]),
                               float(selected) - float(flat['library_mean'][0]))


class RepeatedSampleIds(unittest.TestCase):
    def setUp(self):
        cohort = synthetic_cohort(1, 4, values=[0.0, 0.0, 0.0, 0.0])
        # Two substitutions at one position: identical sample id, distinct states.
        variants = cohort['backgrounds'][0]['variants']
        variants[1]['position'] = variants[0]['position']
        variants[1]['state'] = 2
        cohort['backgrounds'][0]['sequences'][2] = (
            cohort['backgrounds'][0]['wildtype'][:variants[0]['position'] - 1]
            + 'C' + cohort['backgrounds'][0]['wildtype'][variants[0]['position']:])
        self.cohort = cohort
        self.flat = selection.flatten_cohort(cohort)

    def test_sample_id_repeats_but_state_keys_do_not(self):
        ids = self.flat['sample_ids']
        self.assertEqual(ids[0], ids[1])
        self.assertNotEqual(self.flat['state_keys'][0], self.flat['state_keys'][1])
        self.assertEqual(len(set(self.flat['state_keys'])), 4)

    def test_both_substitutions_remain_distinct_rows(self):
        gains = selection.library_gains(self.flat['measured'], np.ones(4), self.flat['backgrounds'],
                                        self.flat['state_keys'], 0.25)
        self.assertEqual(int(gains['n_library'][0]), 4)
        # A dict keyed by sample_id would have found three rows.
        self.assertEqual(len(set(self.flat['sample_ids'])), 3)

    def test_a_tie_at_one_position_is_broken_by_state_key(self):
        keys = ['b:1:1', 'b:1:2', 'b:2:1', 'b:3:1']
        self.assertEqual(list(selection.rank_order([1.0, 1.0, 1.0, 1.0], keys)), [0, 1, 2, 3])
        self.assertEqual(list(selection.rank_order([1.0, 1.0, 1.0, 1.0],
                                                   list(reversed(keys)))), [3, 2, 1, 0])

    def test_a_shuffled_archive_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            selection.verify_oof_join(self.flat, np.asarray(self.flat['sample_ids'])[::-1],
                                      self.flat['groups'][::-1], self.flat['measured'][::-1],
                                      arm='arm')
        self.assertIn('frozen row order', str(caught.exception))
        with self.assertRaises(ValueError):
            selection.verify_oof_join(self.flat, self.flat['sample_ids'],
                                      np.asarray(['wrong'] * 4), self.flat['measured'],
                                      arm='arm')
        with self.assertRaises(ValueError):
            selection.verify_oof_join(self.flat, self.flat['sample_ids'], self.flat['groups'],
                                      self.flat['measured'] + 1.0, arm='arm')


class SameWildTypeSelection(unittest.TestCase):
    def test_each_library_keeps_its_own_random_expectation(self):
        cohort = synthetic_cohort(2, 8, values=np.linspace(1.0, 5.0, 8))
        flat = selection.flatten_cohort(cohort)
        gains = selection.library_gains(flat['measured'], flat['measured'],
                                        flat['backgrounds'], flat['state_keys'], 0.25)
        self.assertEqual(len(gains['backgrounds']), 2)
        for index, label in enumerate(gains['backgrounds']):
            rows = np.flatnonzero(flat['backgrounds'] == label)
            self.assertAlmostEqual(float(gains['library_mean'][index]),
                                   float(flat['measured'][rows].mean()))
            self.assertGreater(float(gains['gain'][index]), 0.0)

    def test_two_libraries_with_opposite_offsets_are_not_differenced_together(self):
        """A global ranking would drag one library's baseline onto the other's."""

        first = synthetic_cohort(1, 6, values=np.linspace(-3.0, -1.0, 6))
        second = synthetic_cohort(1, 6, values=np.linspace(1.0, 3.0, 6))
        second['backgrounds'][0]['name'] = 'two'
        second['backgrounds'][0]['group'] = 'nat-001'
        cohort = {'schema': first['schema'], 'endpoint_sha256': 'e' * 64,
                  'backgrounds': first['backgrounds'] + second['backgrounds']}
        flat = selection.flatten_cohort(cohort)
        gains = selection.library_gains(flat['measured'], flat['measured'],
                                        flat['backgrounds'], flat['state_keys'], 0.5)
        # Both libraries have a positive within-library gain even though their
        # pooled means are on opposite sides of zero.
        self.assertGreater(float(gains['gain'][0]), 0.0)
        self.assertGreater(float(gains['gain'][1]), 0.0)
        self.assertLess(float(flat['measured'][:6].mean()), 0.0)
        self.assertGreater(float(flat['measured'][6:].mean()), 0.0)


class SeedAveraging(unittest.TestCase):
    def test_seeds_are_averaged_on_the_wild_type_axis(self):
        averaged = selection.average_seeds([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertTrue(np.allclose(averaged, [3.0, 4.0]))

    def test_seed_count_does_not_change_the_unit_count(self):
        cohort = synthetic_cohort(1, 4)
        flat = selection.flatten_cohort(cohort)
        per_seed = [selection.library_gains(flat['measured'],
                                            np.full(4, value), flat['backgrounds'],
                                            flat['state_keys'], 0.5)['gain']
                    for value in (1.0, 2.0, 3.0)]
        self.assertEqual(selection.average_seeds(per_seed).shape, (1,))

    def test_a_native_constant_offset_is_algebraically_inert(self):
        """Native versus delta likelihood is one ranking inside a fixed wild type."""

        cohort = synthetic_cohort(2, 6)
        flat = selection.flatten_cohort(cohort)
        base = selection.library_gains(flat['measured'], flat['measured'],
                                       flat['backgrounds'], flat['state_keys'], 0.5)['gain']
        shifted = flat['measured'].copy()
        shifted[flat['backgrounds'] == flat['backgrounds'][0]] += 100.0
        after = selection.library_gains(shifted, shifted, flat['backgrounds'],
                                        flat['state_keys'], 0.5)['gain']
        self.assertTrue(np.allclose(base, after))


# --------------------------------------------------------------------------- #
# The bootstrap unit and the two inference readings.
# --------------------------------------------------------------------------- #

class GroupBootstrap(unittest.TestCase):
    def test_a_repeated_group_contributes_in_full(self):
        matrix = np.asarray([[5.0], [1.0]])
        repeated = selection.resample_group_means(matrix, np.asarray([[0, 0, 1]]))
        deduplicated = float((5.0 + 1.0) / 2.0)
        self.assertAlmostEqual(float(repeated[0, 0]), (5.0 + 5.0 + 1.0) / 3.0)
        self.assertNotAlmostEqual(float(repeated[0, 0]), deduplicated)

    def test_common_draws_keep_paired_columns_paired(self):
        matrix = np.column_stack([np.arange(1.0, 6.0), np.arange(1.0, 6.0)])
        bootstrap = selection.joint_group_bootstrap(matrix, draws=200, seed=3)
        self.assertTrue(np.array_equal(bootstrap['draws_matrix'][:, 0],
                                       bootstrap['draws_matrix'][:, 1]))
        self.assertEqual(bootstrap['n_groups'], 5)
        self.assertGreater(bootstrap['standard_error'][0], 0.0)

    def test_marginal_interval_brackets_the_point(self):
        matrix = np.random.default_rng(1).normal(size=(12, 3))
        bootstrap = selection.joint_group_bootstrap(matrix, draws=500, seed=2)
        interval = selection.percentile_interval(bootstrap['draws_matrix'][:, 0])
        self.assertLess(interval[0], float(bootstrap['point'][0]))
        self.assertGreater(interval[1], float(bootstrap['point'][0]))

    def test_simultaneous_band_widens_with_the_family(self):
        matrix = np.random.default_rng(4).normal(size=(12, 6))
        bootstrap = selection.joint_group_bootstrap(matrix, draws=1000, seed=5)
        single = selection.simultaneous_band(bootstrap['draws_matrix'][:, :1],
                                             bootstrap['point'][:1],
                                             bootstrap['standard_error'][:1])
        full = selection.simultaneous_band(bootstrap['draws_matrix'], bootstrap['point'],
                                           bootstrap['standard_error'])
        self.assertLess(single['critical_value'], full['critical_value'])
        self.assertTrue(full['conditional_on_fitted_predictions'])


class SupportFloor(unittest.TestCase):
    def test_fewer_than_eight_groups_is_degenerate(self):
        self.assertTrue(selection.support_floor(7)['degenerate'])
        self.assertFalse(selection.support_floor(8)['degenerate'])
        self.assertIn('floor', selection.support_floor(7)['degenerate_reason'])


# --------------------------------------------------------------------------- #
# The frozen allowlist and role sheet.
# --------------------------------------------------------------------------- #

class AllowlistValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cohort = synthetic_cohort(2, 4)
        self.cohort_path = self.root / 'cohort.json'
        write_json(self.cohort_path, self.cohort)
        self.path = _write_allowlist(self.root, self.cohort, self.cohort_path)

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, path=None):
        return selection.load_allowlist(path or self.path, self.cohort, root=self.root)

    def test_a_complete_allowlist_is_accepted(self):
        record = self.load()
        self.assertEqual(record['background_count'], 2)
        self.assertTrue(record['review_is_supplied_protocol'])
        self.assertTrue(record['safety_not_inferred_from_metadata'])

    def test_duplicate_background_is_refused(self):
        payload = json.loads(self.path.read_bytes())
        payload['backgrounds'].append(dict(payload['backgrounds'][0]))
        duplicate = self.root / 'duplicate.json'
        write_json(duplicate, payload)
        with self.assertRaises(ValueError) as caught:
            self.load(duplicate)
        self.assertIn('duplicated', str(caught.exception))

    def test_unknown_or_mismatched_group_is_refused(self):
        payload = json.loads(self.path.read_bytes())
        payload['backgrounds'][0]['group'] = 'nat-999'
        mismatch = self.root / 'mismatch.json'
        write_json(mismatch, payload)
        with self.assertRaises(ValueError) as caught:
            self.load(mismatch)
        self.assertIn('assigns group', str(caught.exception))
        payload['backgrounds'][0]['name'] = 'absent.pdb'
        payload['backgrounds'][0]['group'] = 'nat-000'
        absent = self.root / 'absent.json'
        write_json(absent, payload)
        with self.assertRaises(ValueError):
            self.load(absent)

    def test_a_false_construct_match_is_refused(self):
        payload = json.loads(self.path.read_bytes())
        payload['backgrounds'][0]['construct_match'] = {'matched': False, 'method': 'x'}
        bad = self.root / 'bad_match.json'
        write_json(bad, payload)
        with self.assertRaises(ValueError):
            self.load(bad)

    def test_a_changed_construct_review_is_refused(self):
        payload = json.loads(self.path.read_bytes())
        Path(payload['backgrounds'][0]['construct_review']['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'construct review'):
            self.load()

    def test_an_empty_allowlist_is_refused(self):
        payload = json.loads(self.path.read_bytes())
        payload['backgrounds'] = []
        empty = self.root / 'empty.json'
        write_json(empty, payload)
        with self.assertRaises(ValueError):
            self.load(empty)

    def test_a_changed_source_file_is_refused(self):
        payload = json.loads(self.path.read_bytes())
        source = Path(payload['backgrounds'][0]['sources'][0]['path'])
        source.write_text('tampered\n')
        with self.assertRaises(ValueError) as caught:
            self.load()
        self.assertIn('hashes to', str(caught.exception))


class RolesValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_four_required_roles_are_accepted(self):
        record = selection.load_roles(_write_roles(self.root))
        self.assertEqual(list(record['roles']), list(REQUIRED_ROLES))
        self.assertEqual(len(set(record['arms'])), 4)

    def test_a_missing_role_is_refused(self):
        payload = {'schema': selection.ROLES_SCHEMA,
                   'roles': {role: 'protgpt2' for role in REQUIRED_ROLES[:-1]}}
        path = self.root / 'missing.json'
        write_json(path, payload)
        with self.assertRaises(ValueError) as caught:
            selection.load_roles(path)
        self.assertIn('missing', str(caught.exception))

    def test_an_extra_role_or_duplicate_arm_is_refused(self):
        payload = {'schema': selection.ROLES_SCHEMA,
                   'roles': dict(zip(REQUIRED_ROLES,
                                     ('prollama-stage-1', 'prollama', 'protgpt2',
                                      'progen2-xlarge')))}
        payload['roles']['extra'] = 'gpt2'
        extra = self.root / 'extra.json'
        write_json(extra, payload)
        with self.assertRaises(ValueError):
            selection.load_roles(extra)
        payload['roles'] = {role: 'protgpt2' for role in REQUIRED_ROLES}
        duplicate = self.root / 'duplicate.json'
        write_json(duplicate, payload)
        with self.assertRaises(ValueError):
            selection.load_roles(duplicate)


# --------------------------------------------------------------------------- #
# The real entry point over a synthetic frozen panel.
# --------------------------------------------------------------------------- #

class EntryPoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_complete_run_publishes_the_two_families(self):
        cohort = synthetic_cohort(14, 256)
        allowed = [row['name'] for row in cohort['backgrounds'][:12]]
        fixture = make_panel(self.root, cohort, allowed_names=allowed)
        result = run_cli(fixture)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = strict_json(fixture['out'] / stage.OUTPUT_FILENAME)
        self.assertEqual(set(record['budgets_by_background']), set(allowed))
        self.assertEqual(record['support']['cohort_backgrounds'], 14)
        self.assertEqual(record['schema'], selection.SCHEMA)
        self.assertEqual(record['status'], 'complete')
        self.assertEqual(record['support']['allowlisted_backgrounds'], 12)
        self.assertEqual(record['support']['allowlisted_groups'], 12)
        self.assertEqual(record['support']['allowlisted_variants'], 12 * 256)
        self.assertEqual(record['support']['frozen_panel_count'], 12 * 256)
        self.assertIsNone(record['withheld_intervals'])
        self.assertEqual(record['inference']['simultaneous']['primary']['status'], 'complete')
        self.assertEqual(record['inference']['simultaneous']['secondary']['status'], 'complete')
        arm = 'prollama-stage-1'
        primary = record['selection'][arm]['primary']['0.02']
        self.assertGreater(primary['point'], 0.0)
        self.assertIsNotNone(primary['marginal_interval'])
        self.assertEqual(primary['simultaneous_verdict'], 'above_zero')
        self.assertTrue(record['label_orientation']['sign_convention'].startswith('dG_ML'))
        self.assertIn('S_T', record['contrasts']['excluded'])
        self.assertTrue(record['native_versus_delta_likelihood']['not_a_separate_experiment'])
        self.assertIn('safety', ' '.join(record['method_limitations']))

    def test_excluded_backgrounds_never_reach_a_ranker(self):
        cohort = synthetic_cohort(3, 4)
        flat = selection.flatten_cohort(cohort)
        allowed = cohort['backgrounds'][1]['name']
        excluded_keys = {key for key, background in zip(flat['state_keys'], flat['backgrounds'])
                         if background != allowed}
        roles = {'arms': ['synthetic-arm']}
        arm_data = {'synthetic-arm': {'predictions': {
            str(seed): {design: flat['measured'].copy() for design in SELECTION_DESIGNS}
            for seed in SPLIT_SEEDS}}}
        original = selection.rank_order

        def checked_rank(score, keys):
            self.assertFalse(set(keys) & excluded_keys)
            self.assertEqual(len(score), 4)
            return original(score, keys)

        with patch.object(selection, 'rank_order', side_effect=checked_rank) as ranker:
            record = stage.build_analysis(
                cohort, {'backgrounds': [{'name': allowed}]}, roles, arm_data, {},
                draws=10, seed=1)
        self.assertGreater(ranker.call_count, 0)
        self.assertEqual(record['support']['allowlisted_variants'], 4)
        self.assertEqual(set(record['budgets_by_background']), {allowed})

    def test_a_tampered_artefact_is_refused(self):
        cohort = synthetic_cohort(12, 20)
        fixture = make_panel(self.root, cohort)
        panel = fixture['panel_dir'] / 'panel.json'
        panel.write_text(panel.read_text() + ' ')
        result = run_cli(fixture)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('hashes to', result.stderr)

    def test_a_tampered_allowlist_source_stops_the_run(self):
        cohort = synthetic_cohort(12, 20)
        fixture = make_panel(self.root, cohort)
        payload = json.loads(fixture['allowlist'].read_bytes())
        source = Path(payload['backgrounds'][0]['sources'][0]['path'])
        source.write_text('tampered\n')
        result = run_cli(fixture)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('hashes to', result.stderr)

    def test_insufficient_groups_publish_no_interval(self):
        cohort = synthetic_cohort(5, 20)
        fixture = make_panel(self.root, cohort)
        # The real campaign queue creates the empty cell directory before launch.
        fixture['out'].mkdir()
        result = run_cli(fixture)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = strict_json(fixture['out'] / stage.OUTPUT_FILENAME)
        self.assertEqual(record['status'], 'descriptive')
        self.assertIsNotNone(record['withheld_intervals'])
        self.assertIn('floor', record['withheld_intervals']['reason'])
        self.assertIsNone(record['inference']['simultaneous']['primary']['bands'])
        arm = 'prollama-stage-1'
        self.assertIsNone(record['selection'][arm]['primary']['0.02']['marginal_interval'])
        self.assertIsNotNone(record['selection'][arm]['primary']['0.02']['point'])

    def test_a_four_arm_completion_is_refused(self):
        cohort = synthetic_cohort(12, 20)
        fixture = make_panel(self.root, cohort, arms=(
            'prollama-stage-1', 'prollama', 'protgpt2', 'progen2-xlarge'))
        result = run_cli(fixture)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('exact completed 33-arm roster', result.stderr)
        self.assertFalse(fixture['out'].exists())

    def test_a_changed_protocol_hash_is_refused(self):
        fixture = make_panel(self.root, synthetic_cohort(8, 4))
        payload = json.loads(fixture['roles'].read_bytes())
        payload['protocol']['completion_sha256'] = '0' * 64
        write_json(fixture['roles'], payload)
        result = run_cli(fixture)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('protocol completion', result.stderr)

    def test_a_missing_final_contrast_is_refused_even_when_rehashed(self):
        fixture = make_panel(self.root, synthetic_cohort(8, 4))
        path = fixture['panel_dir'] / 'panel.json'
        payload = json.loads(path.read_bytes())
        payload['contrast_definitions'].pop('profile_ranking')
        write_json(path, payload)
        completion_path = fixture['panel_dir'] / 'completion.json'
        completion = json.loads(completion_path.read_bytes())
        completion['artifacts']['panel.json'] = sha256_file(path)
        write_json(completion_path, completion)
        _write_roles(self.root, panel_dir=fixture['panel_dir'], cohort_path=fixture['cohort'])
        result = run_cli(fixture)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('authoritative 33-arm/six-contrast contract', result.stderr)

    def test_an_unselected_arm_must_still_retain_its_archive(self):
        fixture = make_panel(self.root, synthetic_cohort(8, 4))
        (fixture['panel_dir'] / 'oof' / 'oof_gpt2.npz').unlink()
        result = run_cli(fixture)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('gpt2 held-out predictions', result.stderr)
        self.assertFalse(fixture['out'].exists())

    def test_relative_review_paths_use_the_explicit_asset_root(self):
        fixture = make_panel(self.root, synthetic_cohort(8, 4))
        allowlist = json.loads(fixture['allowlist'].read_bytes())
        for entry in allowlist['backgrounds']:
            for kind in ('sources', 'provenance'):
                for reference in entry[kind]:
                    reference['path'] = str(Path(reference['path']).relative_to(self.root))
            review_path = Path(entry['construct_review']['path'])
            evidence = json.loads(review_path.read_bytes())
            evidence['source'] = entry['sources'][0]
            write_json(review_path, evidence)
            entry['construct_review'] = {
                'path': str(review_path.relative_to(self.root)),
                'sha256': sha256_file(review_path),
            }
        write_json(fixture['allowlist'], allowlist)
        roles = json.loads(fixture['roles'].read_bytes())
        roles['protocol']['allowlist_sha256'] = sha256_file(fixture['allowlist'])
        write_json(fixture['roles'], roles)
        result = run_cli(fixture, provenance_root=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = strict_json(fixture['out'] / stage.OUTPUT_FILENAME)
        self.assertEqual(record['input_bindings']['provenance_root'], str(self.root.resolve()))

    def test_existing_output_is_never_overwritten(self):
        fixture = make_panel(self.root, synthetic_cohort(8, 4))
        fixture['out'].mkdir()
        sentinel = fixture['out'] / stage.OUTPUT_FILENAME
        sentinel.write_text('untouched')
        result = run_cli(fixture)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('fresh directory', result.stderr)
        self.assertEqual(sentinel.read_text(), 'untouched')


# --------------------------------------------------------------------------- #
# Replaying a frozen panel whose explanatory metadata predates the correction.
# --------------------------------------------------------------------------- #

def legacy_contrast_definitions() -> dict:
    """The frozen panel's contrast definitions before the metadata correction.

    The frozen E03 artifact carries ``primary``/``companion`` baseline roles and no
    ``qualification``/``calibration`` fields, but the same six measurement
    identities the authoritative definitions carry now. This reconstructs that shape
    from the live definitions so the equivalence is exercised without the frozen
    result tree.
    """

    legacy = {}
    for name, current in CONTRASTS.items():
        definition = {field: current[field] for field in stage.MEASUREMENT_IDENTITY_FIELDS}
        definition['baseline_role'] = (
            'primary' if current['baseline'] == 'profile' else 'companion')
        definition['scale'] = current['scale']
        legacy[name] = definition
    return legacy


class ContrastReplayEquivalence(unittest.TestCase):
    """Corrected qualification metadata must not invalidate a frozen measurement.

    The panel digest still binds the original definitions byte for byte; the replay
    only refuses a panel whose numbers were produced by another design, reading or
    metric.
    """

    def replay(self, definitions):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = make_panel(root, synthetic_cohort(8, 4), contrasts=definitions)
            protocol = {
                'completion_sha256': sha256_file(fixture['panel_dir'] / 'completion.json'),
                'panel_sha256': sha256_file(fixture['panel_dir'] / 'panel.json')}
            return stage.load_completion(fixture['panel_dir'], protocol)

    def test_definitions_predating_the_metadata_correction_are_accepted(self):
        legacy = legacy_contrast_definitions()
        self.assertEqual(set(legacy), set(CONTRASTS))
        for name, definition in legacy.items():
            self.assertEqual(stage._measurement_identity(name, definition),
                             stage._measurement_identity(name, CONTRASTS[name]))
        completion, bindings = self.replay(legacy)
        self.assertEqual(completion['status'], 'complete')
        self.assertEqual(bindings['panel_contract']['arms'], list(ROSTER))

    def test_a_changed_measurement_field_is_refused(self):
        for field, value in (('design', 'S_wrong'), ('metric', 'a different metric')):
            legacy = legacy_contrast_definitions()
            legacy['profile_calibrated_error'][field] = value
            with self.assertRaises(SystemExit) as caught:
                self.replay(legacy)
            self.assertIn('profile_calibrated_error', str(caught.exception))
            self.assertIn('cannot predate a measurement change', str(caught.exception))

    def test_a_definition_missing_a_measurement_field_is_refused(self):
        legacy = legacy_contrast_definitions()
        legacy['profile_ranking'].pop('metric')
        with self.assertRaises(SystemExit) as caught:
            self.replay(legacy)
        self.assertIn('missing', str(caught.exception))

    def test_an_unrecognized_definition_field_is_refused(self):
        legacy = legacy_contrast_definitions()
        legacy['matched_ranking']['notes'] = 'descriptive but undeclared'
        with self.assertRaises(SystemExit) as caught:
            self.replay(legacy)
        self.assertIn('unrecognized fields', str(caught.exception))

    @unittest.skipUnless(FROZEN_E03_PANEL.is_file(), 'frozen E03 panel is not staged here')
    def test_the_actual_frozen_definitions_project_to_the_current_measurement(self):
        declared = json.loads(FROZEN_E03_PANEL.read_bytes())['contrast_definitions']
        self.assertEqual(set(declared), set(CONTRASTS))
        for name, current in CONTRASTS.items():
            self.assertEqual(stage._measurement_identity(name, declared[name]),
                             stage._measurement_identity(name, current))
        # The legacy roles and the missing corrected explanatory fields are exactly
        # what the old strict equality rejected.
        self.assertNotEqual(declared, CONTRASTS)
        self.assertEqual(declared['profile_ranking']['baseline_role'], 'primary')


if __name__ == '__main__':
    unittest.main()
