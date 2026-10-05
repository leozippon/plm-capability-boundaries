"""Native likelihood parity and the separate, exact-state-bound archive contract."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import Identity, Module, ModuleList

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.capability.readouts import readout_extraction as readout

spec = importlib.util.spec_from_file_location(
    'stability_extractor_test', ROOT / 'scripts/capability/stability/extract_stability_singles.py')
assert spec is not None and spec.loader is not None
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)
stage_spec = importlib.util.spec_from_file_location(
    'stability_stage_test', ROOT / 'scripts/capability/stages/context_homologue.py')
assert stage_spec is not None and stage_spec.loader is not None
stage = importlib.util.module_from_spec(stage_spec)
stage_spec.loader.exec_module(stage)

SEQUENCES = ['ACDE', 'VCDE', 'AVDE', 'ACVE']


def packed(_arm, sequence):
    ids = [0] + [ord(residue) % 23 + 1 for residue in sequence] + [31]
    return ids, (1, len(ids) - 1), (1, len(ids) - 1)


class TinyModel(Module):
    """Deterministic test-only block hooks; no pretrained model substitution."""
    def __init__(self):
        super().__init__()
        self.layers = ModuleList([Identity(), Identity()])
        self.config = SimpleNamespace()

    def forward(self, ids):
        hidden = torch.stack([ids.float() / 31, ids.float().square() / 961], dim=-1)
        for block in self.layers:
            hidden = block(hidden)
        return hidden[..., :1] * torch.arange(32, dtype=torch.float32).reshape(1, 1, -1) / 32


class LikelihoodOnlyContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / 'checkpoint'
        self.checkpoint.mkdir()
        self.arm = SimpleNamespace(
            name='gpt2', spec=SimpleNamespace(path=self.checkpoint, architecture='gpt2'),
            model=TinyModel(), device='cpu', serving_provenance=None)
        self.arm.blocks = lambda: self.arm.model.layers
        self.forward_calls = 0

        def forward(arm, rows, _stage):
            self.forward_calls += 1
            ids = torch.tensor(rows, dtype=torch.long)
            return arm.model(ids), ids

        self.forward = forward
        self.plan = {
            'schema': extractor.SCHEMA, 'cohort_sha256': 'c' * 64, 'projection': {},
            'backgrounds': [{
                'name': 'test-background', 'group': 'test-group', 'wildtype': SEQUENCES[0],
                'sequences': SEQUENCES,
                'variants': [{'state': i, 'position': i, 'mutant': 'V'} for i in range(1, 4)],
            }],
        }
        self.plan_path = self.root / 'plan.json'
        self.plan_path.write_text(json.dumps(self.plan))

    def invoke(self, out, *flags):
        argv = ['extract', '--plan', str(self.plan_path), '--expect-plan-sha256',
                extractor.plan_digest(self.plan), '--arm', 'gpt2', '--out', str(out),
                '--device', 'cpu', *flags]
        with patch.object(sys, 'argv', argv), \
                patch.object(extractor, 'load_readout_arm', return_value=self.arm), \
                patch.object(extractor.ch, 'require_position_budget'), \
                patch.object(extractor, 'pack_sequence', side_effect=packed), \
                patch.object(readout, 'pack_sequence', side_effect=packed), \
                patch.object(extractor, 'forward_readout_rows', side_effect=self.forward), \
                patch.object(readout, 'forward_readout_rows', side_effect=self.forward):
            extractor.main()

    def test_full_and_likelihood_only_have_exact_score_token_and_variant_parity(self):
        full, only = self.root / 'full', self.root / 'only'
        self.invoke(full)
        self.assertEqual(self.forward_calls, 7)  # four states, three repeats
        with patch.object(extractor, 'representation_blocks', side_effect=AssertionError('hooks')), \
                patch.object(extractor, 'projection_matrices', side_effect=AssertionError('projection')), \
                patch.object(extractor, 'extract_batch', side_effect=AssertionError('pooling')):
            self.invoke(only, '--likelihood-only')
        self.assertEqual(self.forward_calls, 14)
        with np.load(next(full.glob('*.npz')), allow_pickle=False) as original, \
                np.load(next(only.glob('*.npz')), allow_pickle=False) as observed:
            for key in ('likelihood', 'pooled_token_counts', 'scored_token_counts',
                        'packed_token_counts', 'token_ids', 'token_offsets', 'variant_states',
                        'variant_positions', 'repeat_likelihood_nats'):
                np.testing.assert_array_equal(original[key], observed[key], err_msg=key)
            expected = [hashlib.sha256(s.encode('utf-8')).hexdigest() for s in SEQUENCES]
            self.assertEqual(observed['state_sequence_sha256'].dtype, np.dtype('<U64'))
            self.assertEqual(observed['state_sequence_sha256'].tolist(), expected)
            for key in ('projected', 'hidden_width', 'repeat_feature_relative_l2', 'position_nats'):
                self.assertNotIn(key, observed.files)
            identity = json.loads(str(observed['metadata']))['identity']
            self.assertEqual(identity['schema'], extractor.LIKELIHOOD_SCHEMA)
            self.assertEqual(identity['dtype'], 'float32')
            self.assertEqual(identity['budget'], 1024)
            self.assertEqual(identity['batch_size'], 1)
            self.assertEqual(identity['plan_sha256'], extractor.plan_digest(self.plan))
            self.assertEqual(identity['cohort_sha256'], self.plan['cohort_sha256'])
        manifest = json.loads((only / 'manifest_gpt2.json').read_text())
        self.assertEqual(manifest['backgrounds'][0]['sha256'], extractor.sha(next(only.glob('*.npz'))))
        self.assertEqual(manifest['status'], 'complete')
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.invoke(only, '--likelihood-only')
        self.assertEqual(self.forward_calls, 14)  # resume does not rescore
        with self.assertRaisesRegex(SystemExit, 'resume identity differs'):
            self.invoke(only)

    def test_limited_run_never_publishes_manifest_even_when_it_covers_the_plan(self):
        out = self.root / 'limited'
        self.invoke(out, '--likelihood-only', '--background-limit', '1')
        self.assertFalse((out / 'manifest_gpt2.json').exists())
        self.assertEqual(json.loads((out / 'progress_gpt2.json').read_text())['status'], 'running')

    def test_resume_refuses_a_reordered_state_binding_and_preserves_failure(self):
        out = self.root / 'resume'
        self.invoke(out, '--likelihood-only')
        path = next(out.glob('*.npz'))
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
        payload['state_sequence_sha256'] = payload['state_sequence_sha256'][::-1]
        np.savez_compressed(path, **payload)
        with self.assertRaisesRegex(ValueError, 'state sequence binding mismatch'):
            self.invoke(out, '--likelihood-only')
        self.assertEqual(json.loads((out / 'failure_gpt2.json').read_text())['status'], 'failed')

    def test_incompatible_flags_and_noncanonical_budget_are_refused_before_loading(self):
        for flags in [('--keep-full-features',), ('--keep-position-terms',),
                      ('--extra-block-index', '0'), ('--blas-threads', '1'),
                      ('--budget', '8'), ('--background-limit', '-1'), ('--batch-size', '2')]:
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.invoke(self.root / 'bad', '--likelihood-only', *flags)
        self.assertEqual(self.forward_calls, 0)

    def test_native_reducer_nonfinite_and_no_truncation_paths(self):
        with patch.object(extractor, 'pack_sequence', side_effect=packed), \
                patch.object(extractor, 'forward_readout_rows', side_effect=self.forward):
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                extractor.likelihood_score(self.arm, SEQUENCES[0], stage, 2)
            self.assertEqual(self.forward_calls, 0)
            score = extractor.likelihood_score(self.arm, SEQUENCES[0], stage, 1024)
            ids = torch.tensor([packed(self.arm, SEQUENCES[0])[0]])
            self.assertEqual(score, -stage._target_nll(self.arm.model(ids), ids, 1, 5)['nll_sum'])
        bad = torch.full((1, 6, 32), float('nan'))
        with patch.object(extractor, 'pack_sequence', side_effect=packed), \
                patch.object(extractor, 'forward_readout_rows', return_value=(bad, ids)):
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                extractor.likelihood_score(self.arm, SEQUENCES[0], stage, 1024)

    def test_protgpt2_pooled_count_still_excludes_pure_newline_tokens(self):
        arm = SimpleNamespace(name='protgpt2', tokenizer=SimpleNamespace(
            decode=lambda ids, **_: {1: 'A' * 60, 2: '\n', 3: 'C'}[ids[0]]))
        with patch.object(extractor, 'pack_sequence', return_value=([0, 1, 2, 3], (1, 4), (1, 4))):
            record = extractor.token_record(arm, 'A' * 60 + 'C', 1024)
        self.assertEqual(record['pooled_positions'], [1, 3])
        self.assertEqual(record['scored_span'], [1, 4])
        self.assertEqual(record['ids'], [0, 1, 2, 3])


if __name__ == '__main__':
    unittest.main()
