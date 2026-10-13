"""Invariants of the legacy-plus-new substitution-grid score merge.

These are the properties a merged grid must always satisfy: exactly the wild type
plus 19 non-wild-type states per site with the wild type counted once, the merged
quantity is the raw native FULLSUM log likelihood and the response is that value
minus the wild type, legacy scores are preserved rather than averaged, and a
missing or duplicated background, an arm or protocol mismatch, a state outside
the enumerated grid, a divergent source hash, a non-finite score or a wild-type
repeat drift beyond the fixed 1e-3 nats tolerance is refused rather than
reinterpreted. One readiness check binds the reviewed reference when it is present
and reports that the new intake has not arrived.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

_SPEC = importlib.util.spec_from_file_location(
    'substitution_grid_merge',
    ROOT / 'scripts/capability/stability/merge_substitution_grid_scores.py')
assert _SPEC is not None and _SPEC.loader is not None
merge = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(merge)

PLAN = merge.PLAN
ARM = 'progen2-medium'
REVIEWED_REFERENCE = ROOT / 'data/r6_constraint_response_20261013'
REVIEWED_ARCHIVE = merge.DEFAULT_ARCHIVE
REVIEWED_RAW = merge.DEFAULT_RAW


class GridFixture:
    """A minimal but complete legacy + new intake, hash-bound exactly as in production."""

    def __init__(self, root, names=('A.pdb', 'B.pdb'), shard_names=None, arms=(ARM,), reused_per=2,
                 wildtype=None):
        self.root = Path(root)
        self.reference_dir = self.root / 'reference'
        self.archive_dir = self.root / 'archive'
        self.raw_root = self.root / 'raw'
        self.names = list(names)
        self.arms = list(arms)
        self.reused_per = reused_per
        self.wildtype = dict(wildtype or {'A.pdb': 'ACDE', 'B.pdb': 'ACDF'})
        for name in self.names:
            self.wildtype.setdefault(name, 'AC' + merge.AA20[:2])
        self.sub = {name: PLAN.substitution_grid(self.wildtype[name]) for name in self.names}
        self.reused = {name: self.sub[name][:reused_per] for name in self.names}
        self.missing = {name: self.sub[name][reused_per:] for name in self.names}
        self.shard_names = [list(group) for group in
                            (shard_names or [[name] for name in self.names])]
        self.old_payloads, self.old_paths, self.old_manifests, self.old_identity = {}, {}, {}, {}
        self.new_payloads, self.new_locations, self.new_identity = {}, {}, {}
        self.cohort_path = self.root / 'cohort.json'
        self.allow_path = self.root / 'allowlist.json'
        for directory in (self.reference_dir, self.archive_dir, self.raw_root):
            directory.mkdir(parents=True, exist_ok=True)
        self._write_cohort()
        self._write_json(self.allow_path, {'schema': 'measured_selection_allowlist_v1'})
        self.allow_sha = merge.sha256_file(self.allow_path)
        self._write_old()
        self._write_plans()
        self._write_grid_manifest()
        self._write_reuse_maps()
        self._write_new()

    # ------------------------------------------------------------ construction

    @staticmethod
    def _write_json(path, value):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(value, sort_keys=True))

    @staticmethod
    def _save_npz(path, payload):
        with Path(path).open('wb') as stream:
            np.savez_compressed(stream, **payload)

    def _group(self, name):
        return 'nat-%03d' % (self.names.index(name) + 1)

    def _counts(self, states):
        packed = np.asarray([4 + (index % 3) for index in range(states)], dtype=np.int64)
        scored = packed - 1
        pooled = np.maximum(1, scored - 1)
        offsets = np.concatenate([[0], np.cumsum(packed)]).astype(np.int64)
        return np.zeros(int(offsets[-1]), dtype=np.int64), offsets, packed, scored, pooled

    def _payload(self, identity, name, sequences, variants, likelihood):
        identifiers, offsets, packed, scored, pooled = self._counts(len(sequences))
        return {
            'likelihood': np.asarray(likelihood, dtype=np.float64),
            'state_sequence_sha256': np.asarray([PLAN.sha256_text(s) for s in sequences],
                                                dtype='<U64'),
            'variant_states': np.asarray([v['state'] for v in variants], dtype=np.int64),
            'variant_positions': np.asarray([v['position'] for v in variants], dtype=np.int64),
            'token_ids': identifiers, 'token_offsets': offsets, 'packed_token_counts': packed,
            'scored_token_counts': scored, 'pooled_token_counts': pooled,
            'repeat_likelihood_nats': np.asarray(0.0),
            'metadata': json.dumps({'identity': identity, 'background': name,
                                    'group': self._group(name), 'sequences': len(sequences),
                                    'variants': len(variants)})}

    def _write_cohort(self):
        backgrounds = []
        for name in self.names:
            backgrounds.append({
                'name': name, 'group': self._group(name), 'length': len(self.wildtype[name]),
                'wildtype': self.wildtype[name],
                'sequences': [self.wildtype[name]] + [s['sequence'] for s in self.reused[name]],
                'variants': [{'state': index + 1, 'position': s['position'], 'mutant': s['mutant'],
                              'sequence': s['sequence']} for index, s in enumerate(self.reused[name])]})
        self._write_json(self.cohort_path, {'schema': 'stability_singles_cohort_v1',
                                            'backgrounds': backgrounds})
        self.cohort_sha = merge.sha256_file(self.cohort_path)

    def _identity(self, arm):
        return {'schema': PLAN.EXTRACTOR.LIKELIHOOD_SCHEMA, 'arm': arm, 'dtype': 'float32',
                'batch_size': 1, 'budget': PLAN.BUDGET, 'plan_sha256': f'legacy-plan-{arm}',
                'cohort_sha256': self.cohort_sha,
                'likelihood_semantics': PLAN.NATIVE_LIKELIHOOD_SEMANTICS,
                'state_sequence_sha256_semantics': 'SHA256 of exact UTF-8 sequence bytes',
                'checkpoint_path': f'/legacy/{arm}',
                'checkpoint_metadata_sha256': {'config.json': 'a' * 64},
                'checkpoint_tensor_files': [{'name': 'model.safetensors', 'bytes': 1234}],
                'code_sha256': {}, 'precision_gate': 'inapplicable at batch size one'}

    def old_likelihood(self, name):
        return -10.0 + 0.5 * np.arange(1 + self.reused_per)

    def new_likelihood(self, name):
        return -10.0 + 0.25 * np.arange(1 + len(self.missing[name]))

    def _write_old(self):
        for arm in self.arms:
            identity = self._identity(arm)
            self.old_identity[arm] = identity
            rows = []
            for name in self.names:
                variants = [{'state': index + 1, 'position': s['position'], 'mutant': s['mutant']}
                            for index, s in enumerate(self.reused[name])]
                sequences = [self.wildtype[name]] + [s['sequence'] for s in self.reused[name]]
                payload = self._payload(identity, name, sequences, variants,
                                        self.old_likelihood(name))
                path = self.archive_dir / f'{arm}_{name}.npz'
                self._save_npz(path, payload)
                self.old_payloads[(arm, name)], self.old_paths[(arm, name)] = payload, path
                rows.append({'background': name, 'group': self._group(name), 'file': path.name,
                             'sha256': merge.sha256_file(path), 'sequences': len(sequences),
                             'variants': len(variants), 'repeat_likelihood_nats': 0.0})
            manifest = self.archive_dir / f'manifest_{arm}.json'
            self._write_json(manifest, {'identity': identity, 'backgrounds': rows,
                                        'status': 'complete', 'blas': None})
            self.old_manifests[arm] = manifest

    def _plan_row(self, name):
        missing = self.missing[name]
        return {'name': name, 'group': self._group(name), 'length': len(self.wildtype[name]),
                'wildtype': self.wildtype[name],
                'sequences': [self.wildtype[name]] + [s['sequence'] for s in missing],
                'variants': [{'state': index + 1, 'position': s['position'], 'mutant': s['mutant']}
                             for index, s in enumerate(missing)]}

    def _contract(self):
        return {'alphabet': merge.AA20,
                'substitution_rule': 'exact wild type plus all 19 non-wild-type residues per site',
                'dtype': {arm: 'float32' for arm in self.arms}, 'batch_size': 1,
                'budget': PLAN.BUDGET, 'native_format': PLAN.EXTRACTOR.LIKELIHOOD_SCHEMA,
                'response_reduction': 'FULLSUM',
                'likelihood_semantics': PLAN.NATIVE_LIKELIHOOD_SEMANTICS,
                'likelihood_semantics_binding': 'declared, not re-derived',
                'token_closure': 'ragged layout and count bounds only',
                'state_sequence_sha256_semantics': 'SHA256 of exact UTF-8 sequence bytes',
                'wildtype_contract': 'the wild type is the first state of every background'}

    def _write_plans(self):
        projection = {'seed': PLAN.PROJECTION_SEED, 'dim': PLAN.PROJECTION_DIM,
                      'blocks': list(PLAN.FEATURE_BLOCKS)}
        parent = {'schema': PLAN.SCHEMA, 'cohort_sha256': self.cohort_sha,
                  'projection': projection,
                  'backgrounds': [self._plan_row(name) for name in self.names]}
        self.parent_digest = PLAN.plan_digest(parent)
        self._write_json(self.reference_dir / 'plan.json', parent)
        self.shards, count = [], len(self.shard_names)
        for index, names in enumerate(self.shard_names, start=1):
            plan = {'schema': PLAN.SCHEMA, 'cohort_sha256': self.cohort_sha,
                    'projection': projection,
                    'backgrounds': [self._plan_row(name) for name in names],
                    'shard': {'parent_plan_sha256': self.parent_digest, 'index': index,
                              'count': count, 'backgrounds': list(names),
                              'missing_states': sum(len(self.missing[n]) for n in names)},
                    'contract': self._contract()}
            path = self.reference_dir / f'plan_shard_{index:02d}of{count:02d}.json'
            self._write_json(path, plan)
            self.shards.append({'index': index, 'count': count, 'names': list(names), 'path': path,
                                'plan': plan, 'digest': PLAN.plan_digest(plan)})

    def _write_grid_manifest(self):
        rows = []
        for name in self.names:
            full = 1 + 19 * len(self.wildtype[name])
            rows.append({'name': name, 'group': self._group(name), 'length': len(self.wildtype[name]),
                         'wildtype': self.wildtype[name],
                         'wildtype_sha256': PLAN.sha256_text(self.wildtype[name]),
                         'full_states': full, 'reused_states': 1 + self.reused_per,
                         'missing_states': full - 1 - self.reused_per,
                         'missing': [{'position': s['position'], 'mutant': s['mutant'],
                                      'grid_state': s['state'], 'plan_state': index + 1,
                                      'sha256': s['sha256']}
                                     for index, s in enumerate(self.missing[name])]})
        observed = {'backgrounds': len(self.names),
                    'states_per_model': sum(r['full_states'] for r in rows),
                    'reused_per_model': sum(r['reused_states'] for r in rows),
                    'missing_per_model': sum(r['missing_states'] for r in rows),
                    'wildtype_repeat_states_per_model': len(self.names)}
        expected = {key: observed[key] for key in ('backgrounds', 'states_per_model',
                                                   'reused_per_model', 'missing_per_model')}
        provenance = {arm: {'identity': self.old_identity[arm],
                            'manifest': {'path': str(self.old_manifests[arm]),
                                         'sha256': merge.sha256_file(self.old_manifests[arm])},
                            'checkpoint': {}, 'archives': {}} for arm in self.arms}
        grid = {'schema': PLAN.GRID_SCHEMA,
                'cohort': {'path': str(self.cohort_path), 'sha256': self.cohort_sha},
                'allowlist': {'path': str(self.allow_path), 'sha256': self.allow_sha,
                              'backgrounds': len(self.names), 'reviewed_utc': '2026-01-01'},
                'alphabet': merge.AA20, 'substitution_rule': 'exact wild type plus 19 per site',
                'expected': expected, 'observed': observed,
                'parent_plan': {'file': 'plan.json', 'plan_sha256': self.parent_digest},
                'shards': {'count': len(self.shards), 'policy': 'one background per shard',
                           'files': [{'file': s['path'].name,
                                      'sha256': merge.sha256_file(s['path']),
                                      'plan_sha256': s['digest'], 'backgrounds': s['names'],
                                      'missing_states': s['plan']['shard']['missing_states']}
                                     for s in self.shards]},
                'cells': [], 'provenance': {'arms': provenance}, 'backgrounds': rows,
                'reuse': {}, 'limitations': [], 'h200': {}}
        self.grid = grid
        self._write_json(self.reference_dir / 'grid_manifest.json', grid)

    def _write_reuse_maps(self):
        self.reuse_paths = {}
        for arm in self.arms:
            backgrounds = {}
            for name in self.names:
                entries = [{'sha256': PLAN.sha256_text(self.wildtype[name]), 'archive_index': 0,
                            'role': 'wildtype', 'position': 0, 'mutant': None, 'grid_state': 0}]
                for index, state in enumerate(self.reused[name]):
                    entries.append({'sha256': state['sha256'], 'archive_index': index + 1,
                                    'role': 'measured_single', 'position': state['position'],
                                    'mutant': state['mutant'], 'grid_state': state['state']})
                backgrounds[name] = {'file': self.old_paths[(arm, name)].name,
                                     'sha256': merge.sha256_file(self.old_paths[(arm, name)]),
                                     'reused': entries}
            path = self.reference_dir / f'reuse_map_{arm}.json'
            self._write_json(path, {
                'schema': PLAN.REUSE_SCHEMA, 'arm': arm, 'cohort_sha256': self.cohort_sha,
                'parent_plan_sha256': self.parent_digest,
                'manifest': {'path': str(self.old_manifests[arm]),
                             'sha256': merge.sha256_file(self.old_manifests[arm])},
                'identity': self.old_identity[arm], 'response_reduction': 'FULLSUM',
                'likelihood_semantics': PLAN.NATIVE_LIKELIHOOD_SEMANTICS,
                'likelihood_semantics_binding': 'declared', 'token_closure': 'ragged only',
                'backgrounds': backgrounds})
            self.reuse_paths[arm] = path

    def _write_new(self):
        for arm in self.arms:
            for shard in self.shards:
                directory = self.raw_root / arm / f"shard_{shard['index']:02d}of{shard['count']:02d}"
                directory.mkdir(parents=True, exist_ok=True)
                identity = dict(self.old_identity[arm], plan_sha256=shard['digest'])
                self.new_identity[(arm, shard['index'])] = identity
                rows = []
                for name in shard['names']:
                    variants = [{'state': index + 1, 'position': s['position'], 'mutant': s['mutant']}
                                for index, s in enumerate(self.missing[name])]
                    sequences = [self.wildtype[name]] + [s['sequence'] for s in self.missing[name]]
                    payload = self._payload(identity, name, sequences, variants,
                                            self.new_likelihood(name))
                    path = directory / f'{arm}_{name}.npz'
                    self._save_npz(path, payload)
                    self.new_payloads[(arm, name)] = payload
                    self.new_locations[(arm, name)] = {'shard': shard['index'], 'dir': directory,
                                                       'path': path,
                                                       'manifest': directory / f'manifest_{arm}.json'}
                    rows.append({'background': name, 'group': self._group(name), 'file': path.name,
                                 'sha256': merge.sha256_file(path), 'sequences': len(sequences),
                                 'variants': len(variants), 'repeat_likelihood_nats': 0.0})
                self._write_json(directory / f'manifest_{arm}.json',
                                 {'identity': identity, 'backgrounds': rows, 'status': 'complete',
                                  'blas': None})

    # --------------------------------------------------------------- tampering

    def rewrite_new_npz(self, name, mutate, arm=ARM):
        payload = dict(self.new_payloads[(arm, name)])
        mutate(payload)
        path = self.new_locations[(arm, name)]['path']
        self._save_npz(path, payload)
        self.new_payloads[(arm, name)] = payload
        manifest_path = self.new_locations[(arm, name)]['manifest']
        manifest = json.loads(manifest_path.read_text())
        for row in manifest['backgrounds']:
            if row['background'] == name:
                row['sha256'] = merge.sha256_file(path)
        self._write_json(manifest_path, manifest)

    def rewrite_new_manifest_identity(self, name, **fields):
        arm = fields.pop('source_arm', ARM)
        manifest_path = self.new_locations[(arm, name)]['manifest']
        manifest = json.loads(manifest_path.read_text())
        manifest['identity'] = dict(manifest['identity'], **fields)
        self._write_json(manifest_path, manifest)

    def reference(self):
        return merge.load_reference(self.reference_dir, self.archive_dir)

    def roots(self):
        return {arm: self.raw_root / arm for arm in self.arms}

    def args(self, **overrides):
        values = {'reference': self.reference_dir, 'archive_dir': self.archive_dir,
                  'raw_root': self.raw_root, 'raw_maps': {}, 'out': self.root / 'out',
                  'wt_tolerance_nats': merge.WT_TOLERANCE_NATS, 'overwrite': False,
                  'check_readiness': True}
        values.update(overrides)
        return SimpleNamespace(**values)


class SubstitutionGridMergeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, **kwargs):
        return GridFixture(self.root, **kwargs)

    # ------------------------------------------------------------- coverage

    def test_merged_grid_covers_every_state_once_without_wildtype_double_count(self):
        fixture = self.fixture()
        report = merge.run_merge(fixture.args())
        self.assertEqual(report['status'], 'complete')
        expected_states = sum(row['full_states'] for row in fixture.grid['backgrounds'])
        self.assertEqual(report['validation']['states_total'], expected_states)
        self.assertEqual(report['validation']['reused_states_total'], 2 * (1 + fixture.reused_per))
        for arm in fixture.arms:
            for name in fixture.names:
                grid = merge.full_grid_map(fixture.wildtype[name])
                with np.load(fixture.root / 'out' / arm / f'{name}.npz', allow_pickle=False) as data:
                    self.assertEqual(data['score'].shape, (grid['full_states'],))
                    self.assertEqual(int((data['mutant_index'] == -1).sum()), 1)
                    self.assertEqual(len(set(data['state_sequence_sha256'].tolist())),
                                     grid['full_states'])
                    self.assertEqual(data['position'][0], 0)
                    self.assertTrue(np.isfinite(data['score']).all())

    def test_merged_manifest_records_source_hashes_and_finite_validation(self):
        fixture = self.fixture()
        report = merge.run_merge(fixture.args())
        manifest = json.loads((fixture.root / 'out' / 'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(manifest['validation']['all_finite'], True)
        self.assertEqual(manifest['validation']['arms'], sorted(fixture.arms))
        self.assertEqual(manifest['reference']['grid_manifest']['sha256'],
                         merge.sha256_file(fixture.reference_dir / 'grid_manifest.json'))
        for arm in fixture.arms:
            sources = manifest['source_binding']
            self.assertEqual(sources['old_manifests'][arm]['sha256'],
                             merge.sha256_file(fixture.old_manifests[arm]))
            self.assertEqual(len(sources['new_manifests'][arm]), len(fixture.shards))
            for name in fixture.names:
                self.assertEqual(sources['old_archives'][arm][name]['sha256'],
                                 merge.sha256_file(fixture.old_paths[(arm, name)]))
                self.assertEqual(sources['new_archives'][arm][name]['sha256'],
                                 merge.sha256_file(fixture.new_locations[(arm, name)]['path']))
        self.assertEqual(manifest['validation']['states_total'],
                         report['validation']['states_total'])

    def test_two_arm_merge_reports_per_model_counts_and_arm_totals(self):
        fixture = self.fixture(arms=(ARM, 'protgpt2'))
        report = merge.run_merge(fixture.args())
        self.assertEqual(report['status'], 'complete')
        validation = report['validation']
        expected_states = sum(row['full_states'] for row in fixture.grid['backgrounds'])
        self.assertEqual(validation['model_count'], 2)
        self.assertEqual(validation['biological_backgrounds'], len(fixture.names))
        self.assertEqual(validation['background_model_records_total'], 2 * len(fixture.names))
        self.assertEqual(validation['states_total'], 2 * expected_states)
        self.assertEqual(validation['reused_states_total'],
                         2 * report['arms'][ARM]['validation']['reused_states'])
        self.assertEqual(validation['missing_states_total'],
                         2 * report['arms'][ARM]['validation']['missing_states'])
        for arm in fixture.arms:
            counts = report['arms'][arm]['validation']
            self.assertEqual(counts['states'], expected_states)
            self.assertEqual(counts['backgrounds_complete'], len(fixture.names))
        for stale in ('states_per_model', 'reused_per_model', 'missing_per_model'):
            self.assertNotIn(stale, validation)

    def test_missing_new_coverage_is_refused_and_reported_as_readiness(self):
        fixture = self.fixture()
        for path in (fixture.raw_root / ARM).glob('shard_*'):
            path.rename(path.with_name(path.name + '.absent'))
        report = merge.run_merge(fixture.args())
        self.assertEqual(report['status'], 'awaiting-new-intake')
        self.assertFalse(report['ready'])
        self.assertEqual(len(report['missing_new_intake']), len(fixture.shards))
        self.assertFalse((fixture.root / 'out').exists())
        with self.assertRaises(merge.NewIntakeUnavailable):
            merge.collect_new_cells(fixture.reference(), fixture.roots())

    def test_duplicate_background_across_shards_is_refused(self):
        fixture = self.fixture(names=('A.pdb',), shard_names=[['A.pdb'], ['A.pdb']])
        with self.assertRaisesRegex(ValueError, 'more than one shard'):
            merge.collect_new_cells(fixture.reference(), fixture.roots())

    def test_duplicate_background_within_a_new_manifest_is_refused(self):
        fixture = self.fixture()
        location = fixture.new_locations[(ARM, 'A.pdb')]
        manifest = json.loads(location['manifest'].read_text())
        manifest['backgrounds'].append(dict(manifest['backgrounds'][0]))
        fixture._write_json(location['manifest'], manifest)
        with self.assertRaisesRegex(ValueError, 'repeats a background'):
            merge.run_merge(fixture.args())
        self.assertFalse((fixture.root / 'out').exists())

    def test_background_missing_from_every_shard_is_refused(self):
        fixture = self.fixture(names=('A.pdb', 'B.pdb'), shard_names=[['A.pdb']])
        with self.assertRaisesRegex(ValueError, 'does not cover'):
            merge.collect_new_cells(fixture.reference(), fixture.roots())

    # ------------------------------------------------------ state binding

    def test_new_state_hash_outside_the_plan_is_refused(self):
        fixture = self.fixture()

        def mutate(payload):
            hashes = list(payload['state_sequence_sha256'])
            hashes[1] = PLAN.sha256_text('AAAA')
            payload['state_sequence_sha256'] = np.asarray(hashes, dtype='<U64')

        fixture.rewrite_new_npz('A.pdb', mutate)
        with self.assertRaisesRegex(ValueError, 'do not match the shard plan'):
            merge.collect_new_cells(fixture.reference(), fixture.roots())

    def test_mutation_disagreeing_with_its_sequence_is_refused(self):
        fixture = self.fixture()
        reference = fixture.reference()
        name = 'A.pdb'
        plan_row = copy.deepcopy(fixture.shards[0]['plan']['backgrounds'][0])
        plan_row['variants'][0]['mutant'] = merge.AA20[(merge.AA20.index(plan_row['variants'][0]['mutant']) + 1) % 20]
        with self.assertRaisesRegex(ValueError, 'different mutation than declared'):
            merge.validate_new_archive(reference['backgrounds'][name], plan_row,
                                       fixture.new_payloads[(ARM, name)],
                                       fixture.new_identity[(ARM, 1)])

    def test_state_outside_the_enumerated_grid_is_refused(self):
        fixture = self.fixture()
        reference = fixture.reference()
        name = 'A.pdb'
        plan_row = copy.deepcopy(fixture.shards[0]['plan']['backgrounds'][0])
        plan_row['sequences'][1] = 'AAAE'
        data = dict(fixture.new_payloads[(ARM, name)])
        data['state_sequence_sha256'] = np.asarray(
            [PLAN.sha256_text(s) for s in plan_row['sequences']], dtype='<U64')
        with self.assertRaisesRegex(ValueError, 'outside the enumerated full grid'):
            merge.validate_new_archive(reference['backgrounds'][name], plan_row, data,
                                       fixture.new_identity[(ARM, 1)])

    # ------------------------------------------------------ protocol refusals

    def test_arm_protocol_and_checkpoint_mismatches_are_refused(self):
        cases = {'arm': 'protgpt2', 'dtype': 'bfloat16', 'budget': 512,
                 'likelihood_semantics': 'per-token mean', 'schema': 'other_schema',
                 'plan_sha256': '0' * 64}
        for field, value in cases.items():
            with self.subTest(field=field):
                fixture = GridFixture(self.root / field)
                fixture.rewrite_new_manifest_identity('A.pdb', **{field: value})
                with self.assertRaisesRegex(ValueError, 'not the reusable native protocol'):
                    merge.collect_new_cells(fixture.reference(), fixture.roots())
        fixture = GridFixture(self.root / 'checkpoint')
        fixture.rewrite_new_manifest_identity(
            'A.pdb', checkpoint_metadata_sha256={'config.json': 'b' * 64})
        with self.assertRaisesRegex(ValueError, 'different legacy checkpoint'):
            merge.collect_new_cells(fixture.reference(), fixture.roots())

    # ------------------------------------------------------ source hashes

    def test_source_hash_divergence_is_refused(self):
        fixture = self.fixture()
        fixture.old_paths[(ARM, 'A.pdb')].write_bytes(b'not an archive')
        with self.assertRaisesRegex(ValueError, 'hash mismatch|unreadable'):
            fixture.reference()

        fixture = GridFixture(self.root / 'new-npz')
        fixture.new_locations[(ARM, 'A.pdb')]['path'].write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'bytes differ from the manifest digest'):
            merge.collect_new_cells(fixture.reference(), fixture.roots())

        fixture = GridFixture(self.root / 'plan')
        fixture.shards[0]['path'].write_text(fixture.shards[0]['path'].read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'shard plan bytes differ'):
            fixture.reference()

        fixture = GridFixture(self.root / 'reuse')
        reuse = json.loads(fixture.reuse_paths[ARM].read_text())
        reuse['manifest']['sha256'] = '0' * 64
        fixture._write_json(fixture.reuse_paths[ARM], reuse)
        with self.assertRaisesRegex(ValueError, 'legacy manifest bytes differ'):
            fixture.reference()

    # ------------------------------------------------------ numeric refusals

    def test_nonfinite_new_score_is_refused(self):
        fixture = self.fixture()

        def mutate(payload):
            likelihood = np.asarray(payload['likelihood'], dtype=float)
            likelihood[1] = np.inf
            payload['likelihood'] = likelihood

        fixture.rewrite_new_npz('A.pdb', mutate)
        with self.assertRaisesRegex(ValueError, 'missing or nonfinite native scores'):
            merge.collect_new_cells(fixture.reference(), fixture.roots())

    def test_wildtype_repeat_token_count_change_is_refused(self):
        fixture = self.fixture()

        def mutate(payload):
            packed = np.asarray(payload['packed_token_counts'], dtype=np.int64).copy()
            packed[0] += 1
            scored = packed - 1
            pooled = np.maximum(1, scored - 1)
            offsets = np.concatenate([[0], np.cumsum(packed)]).astype(np.int64)
            payload.update(packed_token_counts=packed, scored_token_counts=scored,
                           pooled_token_counts=pooled, token_offsets=offsets,
                           token_ids=np.zeros(int(offsets[-1]), dtype=np.int64))

        fixture.rewrite_new_npz('A.pdb', mutate)
        with self.assertRaisesRegex(ValueError, 'wild-type repeat packed token count'):
            merge.run_merge(fixture.args())

    def test_wildtype_drift_within_tolerance_is_preserved_and_reported(self):
        fixture = self.fixture()

        def mutate(payload):
            likelihood = np.asarray(payload['likelihood'], dtype=float)
            likelihood[0] = fixture.old_likelihood('A.pdb')[0] + 5e-4
            payload['likelihood'] = likelihood

        fixture.rewrite_new_npz('A.pdb', mutate)
        report = merge.run_merge(fixture.args())
        self.assertEqual(report['status'], 'complete')
        drift = report['validation']['wildtype_drifts_nats'][f'{ARM}/A.pdb']
        self.assertAlmostEqual(drift, 5e-4, places=12)
        with np.load(fixture.root / 'out' / ARM / 'A.pdb.npz', allow_pickle=False) as data:
            self.assertEqual(float(data['wildtype_score']), float(fixture.old_likelihood('A.pdb')[0]))

    def test_wildtype_drift_beyond_the_fixed_tolerance_is_refused(self):
        fixture = self.fixture()

        def mutate(payload):
            likelihood = np.asarray(payload['likelihood'], dtype=float)
            likelihood[0] = fixture.old_likelihood('A.pdb')[0] + 2e-3
            payload['likelihood'] = likelihood

        fixture.rewrite_new_npz('A.pdb', mutate)
        with self.assertRaisesRegex(ValueError, 'drift .* exceeds the fixed'):
            merge.run_merge(fixture.args())

    def test_nonfinite_or_relaxed_wildtype_tolerance_is_refused(self):
        fixture = self.fixture()
        for tolerance in (float('nan'), float('inf'), merge.WT_TOLERANCE_NATS * 2):
            with self.subTest(tolerance=tolerance):
                with self.assertRaisesRegex(ValueError, 'cannot be overridden'):
                    merge.run_merge(fixture.args(wt_tolerance_nats=tolerance))
        self.assertFalse((fixture.root / 'out').exists())

    # ------------------------------------------------------ semantics

    def test_raw_fullsum_delta_orientation_and_wildtype_zero(self):
        fixture = self.fixture()
        merge.run_merge(fixture.args())
        grid = merge.full_grid_map(fixture.wildtype['A.pdb'])
        with np.load(fixture.root / 'out' / ARM / 'A.pdb.npz', allow_pickle=False) as data:
            score = data['score']
            response = data['response']
            self.assertTrue(np.array_equal(response, score - score[0]))
            self.assertEqual(float(response[0]), 0.0)
            self.assertEqual(float(data['wildtype_score']), float(score[0]))
            for state in fixture.reused['A.pdb']:
                self.assertEqual(float(score[state['state']]), float(
                    fixture.old_likelihood('A.pdb')[state['state']]))
            for index, state in enumerate(fixture.missing['A.pdb']):
                self.assertEqual(float(score[state['state']]),
                                 float(fixture.new_likelihood('A.pdb')[index + 1]))
                self.assertEqual(float(response[state['state']]),
                                 float(fixture.new_likelihood('A.pdb')[index + 1])
                                 - float(score[0]))
            self.assertEqual(int(np.flatnonzero(data['state_sequence_sha256'] ==
                                                grid['wildtype_sha256'])[0]), 0)

    def test_token_counts_are_carried_not_normalised_and_scores_stay_raw(self):
        fixture = self.fixture()

        def mutate(payload):
            packed = np.asarray(payload['packed_token_counts'], dtype=np.int64).copy()
            packed[1:] += 7
            scored = packed - 1
            pooled = np.maximum(1, scored - 1)
            offsets = np.concatenate([[0], np.cumsum(packed)]).astype(np.int64)
            payload.update(packed_token_counts=packed, scored_token_counts=scored,
                           pooled_token_counts=pooled, token_offsets=offsets,
                           token_ids=np.zeros(int(offsets[-1]), dtype=np.int64))

        fixture.rewrite_new_npz('A.pdb', mutate)
        merge.run_merge(fixture.args())
        location = fixture.new_locations[(ARM, 'A.pdb')]
        with np.load(location['path'], allow_pickle=False) as new:
            with np.load(fixture.root / 'out' / ARM / 'A.pdb.npz', allow_pickle=False) as merged:
                self.assertTrue(np.isfinite(merged['score']).all())
                for index, state in enumerate(fixture.missing['A.pdb']):
                    grid_state = state['state']
                    self.assertEqual(float(merged['score'][grid_state]),
                                     float(new['likelihood'][index + 1]))
                    self.assertEqual(int(merged['packed_token_counts'][grid_state]),
                                     int(new['packed_token_counts'][index + 1]))
                    self.assertGreater(int(merged['packed_token_counts'][grid_state]), 3)
                    self.assertEqual(float(merged['response'][grid_state]),
                                     float(merged['score'][grid_state]) - float(merged['wildtype_score']))
                for field in merged.files:
                    self.assertNotIn('normalized', field)
                    self.assertNotIn('probability', field)
                self.assertNotIn('normalised_probability', str(merged['metadata']))

    def test_overwrite_is_refused_without_the_declared_flag(self):
        fixture = self.fixture()
        merge.run_merge(fixture.args())
        with self.assertRaises(FileExistsError):
            merge.run_merge(fixture.args())
        report = merge.run_merge(fixture.args(overwrite=True))
        self.assertEqual(report['status'], 'complete')

    # ------------------------------------------------------ readiness

    @unittest.skipUnless(all(path.exists() for path in
                             (REVIEWED_REFERENCE, REVIEWED_ARCHIVE, REVIEWED_RAW / '..')),
                         'the reviewed substitution-grid assets are absent')
    def test_reviewed_reference_readiness_is_explicit(self):
        reference = merge.load_reference(REVIEWED_REFERENCE, REVIEWED_ARCHIVE)
        self.assertEqual(reference['grid']['expected'],
                         {'backgrounds': 12, 'states_per_model': 12894,
                          'reused_per_model': 3084, 'missing_per_model': 9810})
        self.assertEqual(reference['grid']['observed']['states_per_model'], 12894)
        self.assertEqual(reference['grid']['observed']['reused_per_model'], 3084)
        self.assertEqual(sorted(reference['arms']), ['progen2-medium', 'protgpt2'])
        self.assertEqual(len(reference['shards']), 7)
        self.assertEqual(len(reference['backgrounds']), 12)
        self.assertEqual(sum(int(row['reused_states']) for row in reference['backgrounds'].values()),
                         3084)
        probe = merge.probe_new_intake(reference, merge.raw_roots(reference, REVIEWED_RAW))
        if not probe['ready']:
            report = merge.readiness_report(reference, probe, merge.WT_TOLERANCE_NATS)
            self.assertEqual(report['status'], 'awaiting-new-intake')
            self.assertFalse(report['ready'])
            self.assertEqual(len(report['missing_new_intake']), len(probe['missing']))
            self.assertIn('readiness, not a completed matrix', report['message'])
        else:
            self.assertEqual(probe['missing'], [])


if __name__ == '__main__':
    unittest.main()