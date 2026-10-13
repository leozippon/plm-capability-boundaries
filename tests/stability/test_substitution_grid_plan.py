"""Invariants of the missing-state substitution-grid preparation.

These are the properties a reused archive and the plan it feeds must always hold:
the grid is the exact wild type plus all 19 non-wild-type canonical residues at
every position with no duplicated state, every declared mutation re-derives from
the exact sequence it is bound to, a reuse archive that disagrees on its bytes,
protocol identity or state-hash-to-mutation map is refused rather than
reinterpreted, the reused and missing sets partition the grid without leaking a
state across backgrounds or shards, and the ragged likelihood archive closes under
the gate's own token-support validator. One readiness check runs the real
preparation against the reviewed ignored assets when they are present.
"""
from __future__ import annotations

import copy
import importlib.util
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

from src.capability.core.amino_acids import AA20
from src.capability.core.io import sha256_file
from src.capability.stability.stability_gate import (
    single_substitution, validate_likelihood_token_support, validate_plan_background)

spec = importlib.util.spec_from_file_location(
    'substitution_grid_plan',
    ROOT / 'scripts/capability/stability/build_substitution_grid_plan.py')
assert spec is not None and spec.loader is not None
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)

WILD = 'ACDE'
ARM = 'progen2-medium'
IDENTITY = {'schema': plan.EXTRACTOR.LIKELIHOOD_SCHEMA, 'arm': ARM, 'dtype': 'float32',
            'batch_size': 1, 'budget': 1024, 'cohort_sha256': 'c' * 64,
            'likelihood_semantics': plan.NATIVE_LIKELIHOOD_SEMANTICS}

REVIEWED = {
    'cohort': ROOT / 'results/R4/gate_stability_20260924/cohort/cohort.json',
    'allowlist': ROOT / 'data/r6_mechanisms_20261011/measured_stability_eligibility.json',
    'archive_dir': ROOT / 'results/external_baseline/20261008022745_762304146fdd/'
                          'stability_likelihood_small_dense_gpu3',
}
COHORT_SHA = '1e42c3cc1d11d479fcbcce23ffcc37b5799fcf7fc79c65780b1a521fd953e9de'
ALLOWLIST_SHA = '840f4a1566cfedaae3a450a82113b28185a01a7daab3a2d352ea212993ecf168'


def cohort_row(name='FIX.pdb', wildtype=WILD, variants=4):
    grid = plan.substitution_grid(wildtype)
    chosen = grid[:variants]
    return {'name': name, 'group': 'nat-000', 'length': len(wildtype), 'wildtype': wildtype,
            'sequences': [wildtype] + [state['sequence'] for state in chosen],
            'variants': [{'state': index + 1, 'position': state['position'],
                          'mutant': state['mutant'], 'sequence': state['sequence']}
                         for index, state in enumerate(chosen)]}


def archive_payload(row, *, states=None, identity=None):
    states = len(row['sequences']) if states is None else states
    return {
        'state_sequence_sha256': np.asarray(
            [plan.sha256_text(sequence) for sequence in row['sequences']], dtype='<U64'),
        'token_ids': np.zeros(3 * states, dtype=np.int64),
        'token_offsets': np.arange(0, 3 * states + 1, 3, dtype=np.int64),
        'packed_token_counts': np.full(states, 3, dtype=np.int64),
        'scored_token_counts': np.full(states, 2, dtype=np.int64),
        'pooled_token_counts': np.full(states, 1, dtype=np.int64),
        'variant_states': np.asarray([v['state'] for v in row['variants']], dtype=np.int64),
        'variant_positions': np.asarray([v['position'] for v in row['variants']], dtype=np.int64),
        'likelihood': np.arange(states, dtype=np.float64),
        'repeat_likelihood_nats': np.asarray(0.0),
        'metadata': json.dumps({'identity': identity or IDENTITY, 'background': row['name'],
                                'group': row['group'], 'sequences': states,
                                'variants': len(row['variants'])}),
    }


class SubstitutionGridTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, payload, name='fixture.npz'):
        path = self.root / name
        with path.open('wb') as stream:
            np.savez_compressed(stream, **payload)
        return path

    def manifest_row(self, path, row):
        return {'background': row['name'], 'group': row['group'],
                'sequences': len(row['sequences']), 'variants': len(row['variants']),
                'file': path.name, 'sha256': sha256_file(path)}

    # ------------------------------------------------------------------ grid

    def test_the_grid_is_nineteen_non_wild_type_states_per_position(self):
        grid = plan.substitution_grid(WILD)
        self.assertEqual(len(grid), 19 * len(WILD))
        for position in range(1, len(WILD) + 1):
            at_site = [state for state in grid if state['position'] == position]
            self.assertEqual(len(at_site), 19)
            residues = {state['mutant'] for state in at_site}
            self.assertEqual(residues, set(AA20) - {WILD[position - 1]})
        digests = {state['sha256'] for state in grid}
        self.assertEqual(len(digests), len(grid))
        self.assertNotIn(plan.sha256_text(WILD), digests)
        for state in grid:
            self.assertEqual(single_substitution(WILD, state['sequence']), state['position'] - 1)
            self.assertEqual(state['sequence'][state['position'] - 1], state['mutant'])
            self.assertNotEqual(state['sequence'], WILD)
        states = [state['state'] for state in grid]
        self.assertEqual(states, list(range(1, len(grid) + 1)))

    def test_plan_row_binds_each_missing_state_to_its_position_and_label(self):
        row = cohort_row()
        background = plan.grid_background(row)
        missing = background['substitutions'][:7]
        plan_row = plan.plan_row(background, missing)
        validate_plan_background(plan_row)
        self.assertEqual(plan_row['sequences'][0], WILD)
        self.assertEqual(len(plan_row['sequences']), 1 + len(missing))
        for index, variant in enumerate(plan_row['variants']):
            self.assertEqual(variant['state'], index + 1)
            sequence = plan_row['sequences'][variant['state']]
            self.assertEqual(sequence, missing[index]['sequence'])
            self.assertEqual(variant['position'], missing[index]['position'])
            self.assertEqual(variant['mutant'], missing[index]['mutant'])
            self.assertEqual(single_substitution(WILD, sequence), variant['position'] - 1)
            self.assertEqual(sequence[variant['position'] - 1], variant['mutant'])
            self.assertNotEqual(sequence, WILD)

    def test_plan_row_keeps_one_wild_type_and_never_a_wild_type_variant(self):
        background = plan.grid_background(cohort_row())
        plan_row = plan.plan_row(background, background['substitutions'])
        self.assertEqual(plan_row['sequences'].count(WILD), 1)
        self.assertTrue(all(variant['mutant'] != WILD[variant['position'] - 1]
                            for variant in plan_row['variants']))

    # -------------------------------------------------------------- reuse

    def test_reuse_binding_maps_archive_state_hashes_to_declared_mutations(self):
        row = cohort_row()
        background = plan.grid_background(row)
        path = self.write(archive_payload(row))
        binding = plan.verify_and_bind_background(
            background, row, self.manifest_row(path, row), path, IDENTITY, ARM)
        self.assertEqual(len(binding['reused']), len(row['sequences']))
        self.assertEqual(len(binding['missing']), len(background['substitutions']) - 4)
        wild = binding['reused'][0]
        self.assertEqual((wild['role'], wild['archive_index'], wild['position']), ('wildtype', 0, 0))
        for entry, variant in zip(binding['reused'][1:], row['variants']):
            self.assertEqual(entry['role'], 'measured_single')
            self.assertEqual(entry['archive_index'], variant['state'])
            self.assertEqual(entry['position'], variant['position'])
            self.assertEqual(entry['mutant'], variant['mutant'])
        missing_digests = {state['sha256'] for state in binding['missing']}
        reused_digests = {entry['sha256'] for entry in binding['reused']}
        self.assertFalse(missing_digests & reused_digests)

    def test_reuse_refuses_a_reordered_or_corrupted_state_binding(self):
        row = cohort_row()
        background = plan.grid_background(row)
        payload = archive_payload(row)
        payload['state_sequence_sha256'] = payload['state_sequence_sha256'][::-1]
        path = self.write(payload)
        with self.assertRaisesRegex(ValueError, 'do not match the frozen cohort order'):
            plan.verify_and_bind_background(
                background, row, self.manifest_row(path, row), path, IDENTITY, ARM)

    def test_reuse_refuses_a_changed_manifest_digest(self):
        row = cohort_row()
        background = plan.grid_background(row)
        path = self.write(archive_payload(row))
        manifest_row = self.manifest_row(path, row)
        manifest_row['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'archive bytes differ'):
            plan.verify_and_bind_background(background, row, manifest_row, path, IDENTITY, ARM)

    def test_reuse_refuses_a_different_embedded_native_protocol_identity(self):
        row = cohort_row()
        background = plan.grid_background(row)
        other = dict(IDENTITY, dtype='bfloat16')
        path = self.write(archive_payload(row, identity=other))
        with self.assertRaisesRegex(ValueError, 'different protocol identity'):
            plan.verify_and_bind_background(
                background, row, self.manifest_row(path, row), path, IDENTITY, ARM)

    def test_reuse_refuses_a_mutation_position_the_cohort_did_not_declare(self):
        row = cohort_row()
        background = plan.grid_background(row)
        payload = archive_payload(row)
        payload['variant_positions'] = payload['variant_positions'] + 1
        path = self.write(payload)
        with self.assertRaisesRegex(ValueError, 'variant states or positions differ'):
            plan.verify_and_bind_background(
                background, row, self.manifest_row(path, row), path, IDENTITY, ARM)

    # ------------------------------------------------------- protocol pin

    def test_archive_identity_pin_requires_the_reusable_protocol(self):
        manifest = {'status': 'complete', 'blas': None, 'identity': dict(IDENTITY)}
        self.assertEqual(plan.archive_identity_pin(manifest, ARM, 'c' * 64), IDENTITY)
        for mutate in ('dtype', 'budget', 'batch_size', 'cohort_sha256', 'schema', 'arm'):
            broken = copy.deepcopy(manifest)
            broken['identity'][mutate] = 'wrong'
            with self.subTest(field=mutate), self.assertRaisesRegex(ValueError, 'not the reusable'):
                plan.archive_identity_pin(broken, ARM, 'c' * 64)
        for key, value in (('status', 'running'), ('blas', '4')):
            broken = copy.deepcopy(manifest)
            broken[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'not the reusable'):
                plan.archive_identity_pin(broken, ARM, 'c' * 64)

    def test_checkpoint_binding_discloses_metadata_and_sizes_not_weight_bytes(self):
        checkpoint = self.root / 'checkpoint'
        checkpoint.mkdir()
        metadata = checkpoint / 'config.json'
        metadata.write_text('{}')
        tensor = checkpoint / 'model.safetensors'
        tensor.write_bytes(b'weights')
        identity = {'checkpoint_path': str(checkpoint),
                    'checkpoint_metadata_sha256': {'config.json': sha256_file(metadata)},
                    'checkpoint_tensor_files': [{'name': 'model.safetensors',
                                                 'bytes': tensor.stat().st_size}]}
        binding = plan.checkpoint_binding(identity)
        self.assertTrue(binding['metadata_and_tensor_sizes_verified_on_host'])
        self.assertFalse(binding['weight_byte_identity_verified'])
        self.assertIn('size equality does not establish weight-byte identity',
                      binding['verification'])
        absent = plan.checkpoint_binding(
            dict(identity, checkpoint_path=str(self.root / 'absent')))
        self.assertFalse(absent['metadata_and_tensor_sizes_verified_on_host'])
        self.assertFalse(absent['weight_byte_identity_verified'])
        self.assertIn('no metadata hash, tensor size or weight byte was checked',
                      absent['verification'])

    # --------------------------------------------- token support (existing)

    def test_token_support_validator_bounds_the_ragged_archive_layout(self):
        row = cohort_row()
        payload = archive_payload(row)
        validate_likelihood_token_support(payload, len(row['sequences']))
        broken = {
            'offsets open': dict(payload, token_offsets=np.zeros(5, dtype=np.int64)),
            'offsets not monotone': dict(payload, token_offsets=np.arange(6, dtype=np.int64)),
            'packed disagrees with widths': dict(
                payload, packed_token_counts=np.full(5, 4, dtype=np.int64)),
            'scored exceeds packed': dict(
                payload, scored_token_counts=np.full(5, 9, dtype=np.int64)),
            'pooled exceeds scored': dict(
                payload, pooled_token_counts=np.full(5, 5, dtype=np.int64)),
        }
        for label, data in broken.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_likelihood_token_support(data, len(row['sequences']))

    # --------------------------------------------------- partition and shards

    def test_partition_refuses_a_state_that_leaks_between_reuse_and_missing(self):
        row = cohort_row()
        background = plan.grid_background(row)
        path = self.write(archive_payload(row))
        binding = plan.verify_and_bind_background(
            background, row, self.manifest_row(path, row), path, IDENTITY, ARM)
        plan.validate_grid_partition([background], {background['name']: binding})
        tampered = copy.deepcopy(binding)
        tampered['reused'].append(tampered['missing'][0])
        with self.assertRaisesRegex(ValueError, 'do not partition'):
            plan.validate_grid_partition([background], {background['name']: tampered})

    def test_partition_refuses_a_state_shared_by_two_backgrounds(self):
        row = cohort_row()
        other = cohort_row(name='OTHER.pdb')
        backgrounds = [plan.grid_background(row), plan.grid_background(other)]
        path = self.write(archive_payload(row))
        binding = plan.verify_and_bind_background(
            backgrounds[0], row, self.manifest_row(path, row), path, IDENTITY, ARM)
        with self.assertRaisesRegex(ValueError, 'shared by'):
            plan.validate_grid_partition(
                backgrounds, {backgrounds[0]['name']: binding, backgrounds[1]['name']: binding})

    def test_shards_cover_every_background_once_without_splitting_a_family(self):
        rows = [cohort_row('A.pdb', 'ACDE', 4), cohort_row('B.pdb', 'ACDF', 3),
                cohort_row('C.pdb', 'ACDG', 2)]
        backgrounds = [plan.grid_background(row) for row in rows]
        counts = {row['name']: 19 * len(row['wildtype']) - 4 for row in rows}
        shards = plan.shard_backgrounds(backgrounds, 2, counts)
        assigned = [background['name'] for shard in shards for background in shard]
        self.assertEqual(sorted(assigned), sorted(counts))
        self.assertEqual(len(assigned), len(set(assigned)))
        first = plan.shard_backgrounds(backgrounds, 2, counts)
        self.assertEqual([[b['name'] for b in shard] for shard in shards],
                         [[b['name'] for b in shard] for shard in first])
        loads = [sum(counts[b['name']] for b in shard) for shard in shards]
        self.assertLessEqual(max(loads) - min(loads), max(counts.values()))

    # --------------------------------------------------------- readiness

    @unittest.skipUnless(all(path.exists() for path in REVIEWED.values()),
                         'the reviewed R6 preparation assets are absent')
    def test_reviewed_grid_readiness_with_the_present_ignored_assets(self):
        out = self.root / 'prepared'
        args = SimpleNamespace(
            cohort=REVIEWED['cohort'], expect_cohort_sha256=COHORT_SHA,
            allowlist=REVIEWED['allowlist'], expect_allowlist_sha256=ALLOWLIST_SHA,
            archive_dir=REVIEWED['archive_dir'],
            archive_manifests=[('progen2-medium', REVIEWED['archive_dir'] / 'manifest_progen2-medium.json'),
                               ('protgpt2', REVIEWED['archive_dir'] / 'manifest_protgpt2.json')],
            out=out, shards=7)
        report = plan.prepare(args)
        self.assertEqual(report['states_per_model'], plan.EXPECTED_STATES_PER_MODEL)
        self.assertEqual(report['reused_per_model'], plan.EXPECTED_REUSED_PER_MODEL)
        self.assertEqual(report['missing_per_model'], plan.EXPECTED_MISSING_PER_MODEL)
        self.assertEqual(report['cells'], 2 * report['shards'])
        parent = json.loads((out / 'plan.json').read_text())
        self.assertEqual(plan.plan_digest(parent), report['parent_plan_sha256'])
        self.assertEqual(sum(len(row['sequences']) for row in parent['backgrounds']),
                         report['missing_per_model'] + report['backgrounds'])
        for row in parent['backgrounds']:
            validate_plan_background(row)
        self.assertFalse(any(field in json.dumps(parent['backgrounds'])
                             for field in plan.EXTRACTOR.MEASUREMENT_FIELDS))
        digests = set()
        covered = []
        for shard in report['shard_files']:
            written = json.loads((out / shard['file']).read_text())
            self.assertEqual(plan.plan_digest(written), shard['plan_sha256'])
            self.assertEqual(written['shard']['parent_plan_sha256'], report['parent_plan_sha256'])
            digests.add(shard['plan_sha256'])
            covered.extend(shard['backgrounds'])
        self.assertEqual(len(digests), report['shards'])
        self.assertEqual(sorted(covered), sorted({row['name'] for row in parent['backgrounds']}))
        self.assertTrue((out / 'grid_manifest.json').is_file())
        self.assertTrue((out / 'reuse_map_progen2-medium.json').is_file())
        self.assertTrue((out / 'reuse_map_protgpt2.json').is_file())
        grid = json.loads((out / 'grid_manifest.json').read_text())
        self.assertIn('token_closure', grid['reuse'])
        self.assertIn('not independently re-derived from count arrays',
                      grid['reuse']['likelihood_semantics_binding'])
        for arm, record in grid['provenance']['arms'].items():
            self.assertFalse(record['checkpoint']['weight_byte_identity_verified'])
            self.assertFalse(record['checkpoint']['metadata_and_tensor_sizes_verified_on_host'])


if __name__ == '__main__':
    unittest.main()