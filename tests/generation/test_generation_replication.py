import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.capability.generation.generation_replication import capture_generate, token_trace


def test_trace_distinguishes_eos_padding_and_budget():
    ended = token_trace([7, 2, 0, 0], [2, 3], 4)
    assert ended['generated_token_ids'] == [7, 2]
    assert ended['padded_output_token_ids'] == [7, 2, 0, 0]
    assert ended['decoder_stop'] == 'eos'
    assert token_trace([7, 8], None, 2)['decoder_stop'] == 'max_new_tokens'
    with pytest.raises(ValueError, match='neither EOS'):
        token_trace([7], 2, 4)


def test_observation_preserves_output_rng_arguments_and_restores_driver():
    class Generator:
        generation_config = SimpleNamespace(eos_token_id=2, max_new_tokens=4)
        def generate(self, **kwargs):
            self.arguments = kwargs
            return torch.cat((kwargs['input_ids'], torch.tensor([[8, 2, 0, 0]])), dim=1)
    model = Generator()
    original = model.generate
    ids = torch.tensor([[1]])
    state = torch.random.get_rng_state().clone()
    traces = []
    with capture_generate(model, traces):
        actual = model.generate(input_ids=ids, use_cache=False)
    assert actual.tolist() == [[1, 8, 2, 0, 0]]
    assert model.generate == original
    assert model.arguments == {'input_ids': ids, 'use_cache': False}
    assert torch.equal(state, torch.random.get_rng_state())
    assert traces[0]['generated_tokens'] == 2


def test_frozen_manifest_complete_streams_and_outcome_independent_subset():
    path = Path(__file__).parents[2] / 'configs/generation_replication_manifest.json'
    manifest = json.loads(path.read_text())
    assert len(manifest['cells']) == 20
    seeds = [c['seed'] for c in manifest['campaigns']]
    assert len(set(seeds)) == 2 and abs(seeds[1] - seeds[0]) > 800
    for cell in manifest['cells']:
        if cell['condition'] == 'requested':
            assert cell['attempts'] == 3200 and cell['batch_size'] == 25
            assert len(cell['classes']) == 16
            for group in cell['classes']:
                chosen = group['comparison_indices']
                assert len(chosen) == len(set(chosen)) == 50
                assert min(chosen) >= 0 and max(chosen) < 200
        else:
            assert cell['attempts'] == 800


def _load_stage(name):
    import importlib.util
    from scripts.capability.entrypoints import stage_path
    path = stage_path(Path(__file__).parents[2], name)
    spec = importlib.util.spec_from_file_location(name.removesuffix('.py'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fragment_draw_is_length_matched_and_campaign_bound():
    stage = _load_stage('annotate_generation_replication.py')
    pool = {'strata': {'short': ['ACD', 'CDE'], 'long': ['ACDEFGHIK', 'LMNPQRSTV']},
            'population': {'short': 1000, 'long': 200}}
    row = {'sequence': 'ACDEF', 'campaign_seed': 20270905, 'id': 'attempt_1'}
    sequence, receipt = stage.fragment_for(row, pool)
    assert len(sequence) == 5 and any(sequence in donor for donor in pool['strata']['long'])
    assert (sequence, receipt) == stage.fragment_for(row, pool)
    _, other = stage.fragment_for({**row, 'campaign_seed': 20280905}, pool)
    assert receipt['seed'] != other['seed']
    assert stage.fragment_for({**row, 'sequence': ''}, pool)[0] == ''


def test_campaign_uncertainty_uses_three_streams_not_attempts():
    stage = _load_stage('summarise_generation_replication.py')
    estimate = stage.campaign_interval([.1, .2, .3])
    assert estimate['n_campaigns'] == 3 and estimate['degrees_of_freedom'] == 2
    assert estimate['mean'] == pytest.approx(.2)
    assert estimate['ci95'][0] < 0 and estimate['ci95'][1] > .4
    assert estimate['direction'] == 'positive'
    assert not estimate['resolved_across_campaigns']
    assert stage.campaign_interval([0, 0, 0])['ci95'] is None


def test_campaign_estimate_reaches_disk_through_the_summary_writer(tmp_path):
    """Interval arithmetic runs in numpy; every retained field must still serialise."""
    from src.capability.generation.generation_replication import write_json
    stage = _load_stage('summarise_generation_replication.py')
    resolved = stage.campaign_interval([.3, .4, .5])
    assert resolved['resolved_across_campaigns'] is True
    assert stage.campaign_interval([.1, .2, .35])['resolved_across_campaigns'] is False
    records = {'resolved': resolved, 'no_variation': stage.campaign_interval([0, 0, 0]),
               'two_streams': stage.campaign_interval([.1, .2], campaigns=2)}
    for name, record in records.items():
        write_json(tmp_path / f'{name}.json', record)
        assert json.loads((tmp_path / f'{name}.json').read_text()) == record
    with pytest.raises(ValueError, match='exactly 3'):
        stage.campaign_interval([.1, .2])


def test_empty_attempt_remains_in_rate_denominator():
    stage = _load_stage('annotate_generation_replication.py')
    rows = []
    for length, success in [(0, False), (100, True)]:
        rows.append({'length': length, 'decoder_stop': 'eos', 'native_delimiter_observed': True,
                     'profile': {'generated': {'any_family': success, 'complete_domain': success},
                                 'fragment': {'any_family': False, 'complete_domain': False}}})
    report = stage.descriptive(rows)
    assert report['n_attempts'] == 2
    assert report['endpoints']['complete_domain']['model_rate'] == .5
    assert sum(s['n_attempts'] for s in report['length_strata']) == 2


@pytest.mark.parametrize('arm', ['progen3-3b', 'progen3-112m'])
def test_official_extractor_preserves_compiled_and_uncompiled_prefixes(arm):
    from src.capability.generation.generation_replication import extract_sequence
    cell = {'arm': arm, 'condition': 'unconditioned'}
    assert extract_sequence(cell, 'ACD2<eos>', 'ACD') == ('ACD', '<eos>')
    assert extract_sequence(cell, 'ACD', None) == ('ACD', '<eos>')
    with pytest.raises(ValueError, match='compilation differs'):
        extract_sequence(cell, 'ACD2<eos>', 'AAA')


def test_real_transformers_sampling_is_unchanged_by_token_capture():
    from transformers import GPT2Config, GPT2LMHeadModel
    from src.capability.generation.conditioned_generation import sample_continuations
    model = GPT2LMHeadModel(GPT2Config(vocab_size=16, n_positions=32, n_embd=16,
        n_layer=1, n_head=1, eos_token_id=2, bos_token_id=1, pad_token_id=0)).eval()

    class Tokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0
        def __call__(self, text, **kwargs):
            return {'input_ids': torch.tensor([[1]])}
        def decode(self, ids, **kwargs):
            return ','.join(str(int(i)) for i in ids)

    arguments = dict(n=4, seed=31415, batch_size=2, max_new_tokens=8,
                     temperature=.85, top_p=.95, top_k=5)
    plain = sample_continuations(model, Tokenizer(), '1', **arguments)
    traces = []
    with capture_generate(model, traces):
        observed = sample_continuations(model, Tokenizer(), '1', **arguments)
    assert plain == observed
    assert len(traces) == 4
    assert [','.join(map(str, r['padded_output_token_ids'])) for r in traces] == observed


def test_preflight_uses_active_safetensors_index_not_unused_bin_index(tmp_path):
    stage = _load_stage('preflight_generation_replication.py')
    (tmp_path / 'model-00001.safetensors').write_bytes(b'header checked by actual loader')
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({
        'metadata': {}, 'weight_map': {'weight': 'model-00001.safetensors'}}))
    (tmp_path / 'pytorch_model.bin.index.json').write_text(json.dumps({
        'metadata': {}, 'weight_map': {'weight': 'unused-missing.bin'}}))
    assert stage.active_weight_files(tmp_path) == [tmp_path / 'model-00001.safetensors']
    (tmp_path / 'model-00001.safetensors').unlink()
    assert not all(p.is_file() for p in stage.active_weight_files(tmp_path))


def test_worker_isolates_native_default_cuda_device():
    stage = _load_stage('generation_replication_worker.py')
    assert stage.isolated_cuda_environment('cuda:3', {})['CUDA_VISIBLE_DEVICES'] == '3'
    assert stage.isolated_cuda_environment('cuda:0', {'CUDA_VISIBLE_DEVICES': '4'})['CUDA_VISIBLE_DEVICES'] == '4'
    assert stage.isolated_cuda_environment('cuda:1', {'CUDA_VISIBLE_DEVICES': '4,6'})['CUDA_VISIBLE_DEVICES'] == '6'
    with pytest.raises(ValueError, match='explicit CUDA'):
        stage.isolated_cuda_environment('cpu', {})


def test_decoder_stop_interval_has_only_two_observed_streams():
    stage = _load_stage('summarise_generation_replication.py')
    result = stage.campaign_interval([.1, .2], campaigns=2)
    assert result['n_campaigns'] == 2 and result['degrees_of_freedom'] == 1
    assert result['ci95'][0] < 0 and result['ci95'][1] > 1 / 2


def test_historical_lengths_require_transitive_digest_and_complete_support(tmp_path):
    from src.capability.generation.generation_replication import sha256
    stage = _load_stage('summarise_generation_replication.py')
    cell = tmp_path / 'example.jsonl'
    rows = [{'attempt_id': str(i), 'parent_length': i % 2,
             'sequences': {'generated': 'A' if i % 2 else ''}} for i in range(800)]
    cell.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    build = tmp_path / 'build_manifest.json'
    build.write_text(json.dumps({'cells': [{'cell': 'example', 'cell_sha256': sha256(cell)}]}))
    historical = {'input_sha256': {'build/build_manifest.json': sha256(build)}}
    lengths, sources = stage.historical_lengths(historical, build, tmp_path)
    assert lengths['example']['mean'] == .5 and len(sources) == 2
    cell.write_text(cell.read_text() + 'corrupted')
    with pytest.raises(ValueError, match='cell digest mismatch'):
        stage.historical_lengths(historical, build, tmp_path)


def test_summary_separates_campaign_units_and_stage_contrast():
    stage = _load_stage('summarise_generation_replication.py')
    names = ['prollama-stage-1__unconditioned', 'prollama__unconditioned']
    manifest = {'cells': [{'cell': name, 'arm': name.split('__')[0], 'condition': 'unconditioned'} for name in names],
                'campaigns': [{'id': 'replicate_1'}, {'id': 'replicate_2'}]}
    historical, profiles, lengths = {'cells': []}, {}, {}
    for index, name in enumerate(names):
        successes = 80 - index * 40
        endpoint = {'model_rate': successes / 800, 'model_successes': successes,
                    'controls': {'fragment': {'control_rate': .25, 'difference': successes / 800 - .25,
                                             'control_successes': 200}}}
        historical['cells'].append({'cell': name, 'n_attempts': 800,
                                    'endpoints': {key: endpoint for key in ['any_family', 'complete_domain']}})
        lengths[name] = {'mean': 100.}
        for campaign in manifest['campaigns']:
            comparison = {'n_attempts': 800, 'length_residues': {'mean': 110.},
                          'decoder_stop_counts': {'eos': 200, 'max_new_tokens': 600},
                          'endpoints': {key: {'model_rate': successes / 800, 'fragment_rate': .25,
                              'model_minus_fragment': successes / 800 - .25,
                              'model_successes': successes, 'fragment_successes': 200}
                              for key in ['any_family', 'complete_domain']}}
            profiles[name, campaign['id']] = {'campaign': campaign['id'], 'comparison': comparison, 'all_attempts': comparison}
    result = stage.summarise(manifest, historical, profiles, lengths)
    contrast = result['stage2_minus_stage1_unconditional']['complete_domain']['model_rate']
    assert contrast['values'] == pytest.approx([-.05, -.05, -.05])
    assert contrast['direction'] == 'negative' and contrast['ci95'] is None
    cell = result['cells'][0]
    assert cell['length_residues']['campaign_mean']['values'] == [100., 110., 110.]
    assert cell['decoder_stop_rates_two_new_campaigns']['eos']['n_campaigns'] == 2
    assert cell['endpoints']['complete_domain']['counts']['denominators'] == [800, 800, 800]
