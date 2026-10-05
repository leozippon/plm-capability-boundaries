"""Replay invariants; all fitting here is synthetic CPU data, never experiment refits."""
import csv
import gzip
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.capability.mutation import external_confirmation as abundance
from src.capability.stability import stability_gate as stability

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/capability/reporting/replay_phenotype_predictions.py'
spec = importlib.util.spec_from_file_location('phenotype_replay', SCRIPT)
assert spec is not None and spec.loader is not None
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def toy_cohort(endpoint='abundance'):
    units = []
    for index in range(10):
        variants: list[dict] = [dict(position=1, mutant='C', sequence='CA', state=1),
                    dict(position=2, mutant='D', sequence='AD', state=2)]
        for j, variant in enumerate(variants):
            variant['target' if endpoint == 'abundance' else 'ddg'] = float(index - j) / 10
        units.append({'name': f'unit-{index}', 'group': f'family-{index}',
                      'wildtype': 'AA', 'length': 2, 'wildtype_combined_kcal_mol': 1.,
                      'variants': variants})
    return {'schema': 'external_confirmation_cohort_v1' if endpoint == 'abundance' else 'stability_singles_cohort_v1',
            'domains' if endpoint == 'abundance' else 'backgrounds': units}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def fixture_inputs(tmp_path, endpoint='abundance'):
    paths = replay.input_paths(tmp_path, endpoint, 'gpt2')
    cohort = toy_cohort(endpoint)
    write(paths['cohort'], cohort)
    paths['profiles'].write_bytes(b'toy profiles; not used for missing input checks')
    controls = {'schema': 'nested_gate_control_qualification_v1' if endpoint == 'abundance' else 'stability_control_qualification_v1',
                'cohort_sha256': replay.digest(paths['cohort']),
                'profile_sha256': replay.digest(paths['profiles']), 'rows': 20,
                'split_seeds': list(replay.SEEDS),
                'qualified_control_set': ['ident', 'geom', 'comp', 'chem', 'prof', 'prof2', 'G']
                if endpoint == 'abundance' else ['ident', 'geom', 'chem', 'G']}
    write(paths['controls'], controls)
    reference = {'arm': 'gpt2', 'cohort_sha256': controls['cohort_sha256']}
    if endpoint == 'abundance':
        reference.update(schema='nested_gate_fit_v1', profile_sha256=controls['profile_sha256'],
                         controls_sha256=replay.digest(paths['controls']), rows=20,
                         qualified_control_set=controls['qualified_control_set'],
                         split_seeds=list(replay.SEEDS), matched_baseline={'primary': 'S_T'})
    else:
        reference.update(baseline='S', seeds={str(s): {'groups': sorted(u['group'] for u in cohort['backgrounds']),
                         'baseline_mse': [0.] * 10, 'augmented_mse': [0.] * 10} for s in replay.SEEDS})
    write(paths['reference'], reference)
    return cohort, paths


def read_rows(path):
    with gzip.open(path, 'rt', newline='') as stream:
        return list(csv.DictReader(stream))


def small_panel(endpoint):
    cohort = toy_cohort(endpoint)
    records = replay.registry(cohort, endpoint, 'toy')
    group = np.asarray([r['family'] for r in records])
    panel: dict = {'group': group, 'site': np.asarray([f'{r["unit"]}:{r["mutation_position"]}' for r in records]),
             'target': np.asarray([r['phenotype'] for r in records]),
             'domain': np.asarray([r['unit'] for r in records])}
    rng = np.random.default_rng(42)
    blocks = {'x': rng.normal(size=(len(records), 2)), 'M': rng.normal(size=(len(records), 1))}
    if endpoint == 'abundance':
        panel['weights'] = abundance.nested_weights(panel['group'], panel['domain'], panel['site'])
    else:
        n = len(records)
        panel['states'] = {'features': rng.normal(size=(n * 2, 2)),
                           'y': rng.normal(size=n * 2), 'group': np.repeat(group, 2),
                           'weight': np.full(n * 2, 1 / (n * 2))}
        panel['pair_states'] = np.arange(n * 2).reshape(n, 2)
    return records, panel, blocks


@pytest.mark.parametrize('endpoint', ['abundance', 'stability'])
def test_original_functions_and_deterministic_serialization(tmp_path, endpoint):
    import torch
    torch.set_num_threads(4)
    records, panel, blocks = small_panel(endpoint)
    module = abundance if endpoint == 'abundance' else stability
    kwargs = {'first_stage': ('x',)} if endpoint == 'abundance' else {'purge': None}
    with replay.capture_partitions(module) as partitions:
        result = module.fold_predictions(panel, blocks, {'S': ('x', 'G'), 'S_M': ('x', 'G', 'M')},
                                         seed=replay.SEEDS[0], device='cpu', **kwargs)
    assert len(partitions) == 6
    assert [p['splits'] for p in partitions] == [5, 4, 4, 4, 4, 4]
    assigned = replay.heldout_join(panel['group'], result['folds'])
    for i, group in enumerate(panel['group']):
        assert group in result['folds'][assigned[i]]['held_groups']
        assert group not in result['folds'][assigned[i]]['training_groups']
    summary = replay.summary_metrics(module, panel, result, 'S', endpoint)
    if endpoint == 'abundance':
        previous = {k: v for k, v in summary.items() if k.startswith('primary_')}
        previous.update(alpha=[f['alpha'] for f in result['folds']],
                        dimensions=result['folds'][0]['dimensions'],
                        held_groups_per_fold=[f['held_groups'] for f in result['folds']],
                        nuisance=result['nuisance'])
        reference = {'per_seed': {str(replay.SEEDS[0]): previous}}
    else:
        previous = dict(summary, folds=result['folds'])
        reference = {'seeds': {str(replay.SEEDS[0]): previous}}
    joined, checks = replay.validate_fit(result, reference, endpoint, replay.SEEDS[0], 'S', panel, partitions)
    assert all(c['agrees'] for c in checks + replay.compare_metrics(summary, previous, endpoint))
    for name in ('one', 'two'):
        with replay.csv_stream(tmp_path / f'{name}.csv.gz') as writer:
            replay.serialize_seed(writer, records, endpoint=endpoint, arm='gpt2', seed=replay.SEEDS[0],
                                  baseline='S', status='reproduced_within_tolerance', cohort_sha256='toy',
                                  result=result, heldout=joined)
    assert (tmp_path / 'one.csv.gz').read_bytes() == (tmp_path / 'two.csv.gz').read_bytes()
    rows = read_rows(tmp_path / 'one.csv.gz')
    assert len(rows) == len(records)
    assert rows[0]['phenotype'] == '0.0'  # zero is a measurement, not a missingness sentinel
    assert all(row['baseline_prediction'] != '' and row['augmented_prediction'] != '' for row in rows)
    assert [r['sample_id'] for r in rows] == [r['sample_id'] for r in records]
    assert [int(r['heldout_fold']) for r in rows] == assigned.tolist()
    # Fold/support mismatches fail closed, even when the numerical metrics agree.
    result['folds'][0]['held_groups'].reverse()
    with pytest.raises(replay.ReplayError, match='partition mismatch|historical fit'):
        replay.validate_fit(result, reference, endpoint, replay.SEEDS[0], 'S', panel, partitions)


def test_same_support_and_zero_prediction(tmp_path):
    records = replay.registry(toy_cohort(), 'abundance', 'toy')
    result = {'predictions': {'S': np.zeros(20), 'S_M': np.zeros(20)}}
    with replay.csv_stream(tmp_path / 'zero.csv.gz') as writer:
        replay.serialize_seed(writer, records, endpoint='abundance', arm='gpt2', seed=20260923,
                              baseline='S', status='reproduced_metric_mismatch', cohort_sha256='toy',
                              result=result, heldout=np.zeros(20, dtype=int))
    assert read_rows(tmp_path / 'zero.csv.gz')[0]['augmented_prediction'] == '0.0'
    result['predictions']['S_M'] = np.zeros(19)
    with replay.csv_stream(tmp_path / 'bad.csv.gz') as writer:
        with pytest.raises(replay.ReplayError, match='support differs'):
            replay.serialize_seed(writer, records, endpoint='abundance', arm='gpt2', seed=20260923,
                                  baseline='S', status='reproduced_within_tolerance', cohort_sha256='toy',
                                  result=result, heldout=np.zeros(20, dtype=int))
    assert read_rows(tmp_path / 'bad.csv.gz') == []
    result['predictions']['S_M'] = np.full(20, np.nan)
    with replay.csv_stream(tmp_path / 'nan.csv.gz') as writer:
        with pytest.raises(replay.ReplayError):
            replay.serialize_seed(writer, records, endpoint='abundance', arm='gpt2', seed=20260923,
                                  baseline='S', status='bad', cohort_sha256='toy', result=result,
                                  heldout=np.zeros(20, dtype=int))


@pytest.mark.parametrize('endpoint', ['abundance', 'stability'])
def test_missing_extraction_retains_all_failed_samples_and_source_hashes(tmp_path, endpoint):
    _, paths = fixture_inputs(tmp_path, endpoint)
    before = {k: replay.digest(p) for k, p in paths.items()}
    out = tmp_path / 'output'
    out.mkdir()
    with patch.object(abundance, 'fold_predictions', side_effect=AssertionError('must not fit')):
        report = replay.replay_arm(tmp_path, out, tmp_path / 'missing-extraction', endpoint, 'gpt2', {})
    assert report['status'] == 'failed'
    assert 'Missing required input' in report['error']
    assert '--extraction/<arm>' in report['action']
    rows = read_rows(out / 'gpt2.csv.gz')
    assert len(rows) == 20 * 3
    assert {int(r['seed']) for r in rows} == set(replay.SEEDS)
    assert all(r['status'] == 'failed' and r['baseline_prediction'] == r['augmented_prediction'] == ''
               and r['heldout_fold'] == '' for r in rows)
    assert {k: replay.digest(p) for k, p in paths.items()} == before
    assert report['original_artifacts_unchanged'] is True
    assert json.loads((out / 'gpt2.json').read_text())['seeds']['20260925']['status'] == 'failed'


def test_numeric_mismatch_flagged_without_tolerance_relaxation():
    assert replay.compare_numeric('zero', 0., 0.)['agrees']
    assert replay.compare_numeric('near', 1. + 1e-10, 1.)['agrees']
    mismatch = replay.compare_numeric('changed', [1., 2.01], [1., 2.])
    assert not mismatch['agrees']
    assert mismatch['max_absolute_difference'] > .009
    assert replay.ATOL == 1e-9 and replay.RTOL == 1e-8
    with pytest.raises(replay.ReplayError, match='support'):
        replay.compare_numeric('wrong support', [0.], [0., 0.])
    with pytest.raises(replay.ReplayError, match='nonfinite'):
        replay.compare_numeric('nan', np.nan, 0.)


def test_stability_compares_every_group_not_only_mean():
    old = {'groups': ['a', 'b'], 'baseline_mse': [1., 2.], 'augmented_mse': [1., 2.]}
    actual = dict(old, augmented_mse=[1.1, 1.9])  # same mean; different per-group values
    checks = replay.compare_metrics(actual, old, 'stability')
    assert checks[0]['agrees'] and not checks[1]['agrees']
    with pytest.raises(replay.ReplayError, match='labels/order'):
        replay.compare_metrics(dict(actual, groups=['b', 'a']), old, 'stability')


def test_held_group_join_rejects_duplicates_missing_and_purge():
    groups = np.asarray(['a', 'a', 'b'])
    folds = [dict(fold=0, held_groups=['a'], training_groups=['b']),
             dict(fold=1, held_groups=['b'], training_groups=['a'])]
    assert replay.heldout_join(groups, folds).tolist() == [0, 0, 1]
    with pytest.raises(replay.ReplayError, match='coverage'):
        replay.heldout_join(groups, folds[:1])
    with pytest.raises(replay.ReplayError, match='multiple folds'):
        replay.heldout_join(groups, [folds[0], dict(folds[0], fold=1)])
    with pytest.raises(replay.ReplayError, match='purge'):
        replay.heldout_join(groups, [dict(folds[0], purged_training_groups=['b']), folds[1]])


def test_original_baseline_not_requalified(tmp_path):
    cohort, paths = fixture_inputs(tmp_path)
    controls, old = replay.read_json(paths['controls']), replay.read_json(paths['reference'])
    hashes = {k: replay.digest(p) for k, p in paths.items()}
    records = replay.registry(cohort, 'abundance', hashes['cohort'])
    with patch.object(abundance, 'qualify', side_effect=AssertionError('do not requalify')):
        baseline, columns = replay.validate_reference('abundance', 'gpt2', records, controls, old, hashes)
    assert baseline == 'S_T' and columns[-1] == 'T'
    old['matched_baseline']['primary'] = 'S'
    assert replay.validate_reference('abundance', 'gpt2', records, controls, old, hashes)[1][-1] == 'G'
    old['controls_sha256'] = 'changed'
    with pytest.raises(replay.ReplayError, match='controls_sha256'):
        replay.validate_reference('abundance', 'gpt2', records, controls, old, hashes)


def test_feature_hash_and_state_checks(tmp_path):
    unit = toy_cohort()['domains'][0]
    path = tmp_path / 'unit.npz'
    values = dict(variant_states=np.array([1, 2]), variant_positions=np.array([1, 2]),
                  likelihood=np.zeros(3), projected=np.zeros((3, 4, 256)),
                  pooled_token_counts=np.ones(3), token_offsets=np.arange(4), token_ids=np.arange(3))
    np.savez(path, **values)
    manifest = {'status': 'complete', 'identity': {'schema': 'stability_singles_extraction_v1',
                'arm': 'gpt2', 'cohort_sha256': 'toy', 'dtype': 'float32', 'batch_size': 1, 'budget': 1024},
                'backgrounds': [{'background': unit['name'], 'file': path.name, 'sha256': replay.digest(path)}]}
    write(tmp_path / 'manifest_gpt2.json', manifest)
    replay.validate_features(tmp_path, 'gpt2', 'toy', [unit])
    # NaNs must fail even with a valid digest; zero likelihoods above are admitted.
    values['likelihood'][1] = np.nan
    np.savez(path, **values)
    manifest['backgrounds'][0]['sha256'] = replay.digest(path)
    write(tmp_path / 'manifest_gpt2.json', manifest)
    with pytest.raises(replay.ReplayError, match='likelihood'):
        replay.validate_features(tmp_path, 'gpt2', 'toy', [unit])
    values['likelihood'][1] = 0.
    values['variant_states'] = np.array([2, 1])
    np.savez(path, **values)
    manifest['backgrounds'][0]['sha256'] = replay.digest(path)
    write(tmp_path / 'manifest_gpt2.json', manifest)
    with pytest.raises(replay.ReplayError, match='order changed'):
        replay.validate_features(tmp_path, 'gpt2', 'toy', [unit])
    path.unlink()
    with pytest.raises(replay.ReplayError, match='Missing archive'):
        replay.validate_features(tmp_path, 'gpt2', 'toy', [unit])


def test_mixed_seed_mismatch_and_failure_are_retained(tmp_path):
    from contextlib import ExitStack
    import torch
    from scripts.capability.gates import fit_nested_singles as loader
    torch.set_num_threads(4)
    cohort, paths = fixture_inputs(tmp_path)
    records, panel, toy_blocks = small_panel('abundance')
    controls = replay.read_json(paths['controls'])
    for key in ('ident', 'geom', 'comp', 'chem', 'prof', 'prof2'):
        panel.setdefault('blocks', {})[key] = toy_blocks['x'][:, :1]
    model_blocks = {'M': toy_blocks['M'], 'T': toy_blocks['x'][:, 1:]}
    blocks = dict(panel['blocks'], **model_blocks)
    columns = tuple(controls['qualified_control_set']) + ('T',)
    designs = {'S_T': columns, 'S_T_M': columns + ('M',)}
    controls['row_identity_sha256'] = abundance.row_identity(panel['group'], panel['site'], panel['target'])
    write(paths['controls'], controls)
    old = replay.read_json(paths['reference'])
    old.update(controls_sha256=replay.digest(paths['controls']),
               row_identity_sha256=controls['row_identity_sha256'], per_seed={})
    results = {}
    for seed in replay.SEEDS:
        result = abundance.fold_predictions(panel, blocks, designs, seed=seed, device='cpu',
                                            first_stage=tuple(c for c in controls['qualified_control_set'] if c != 'G'))
        results[seed] = result
        old['per_seed'][str(seed)] = dict(replay.summary_metrics(abundance, panel, result, 'S_T', 'abundance'),
            alpha=[f['alpha'] for f in result['folds']], dimensions=result['folds'][0]['dimensions'],
            held_groups_per_fold=[f['held_groups'] for f in result['folds']], nuisance=result['nuisance'])
    old['per_seed']['20260924']['primary_likelihood']['point'] += .01
    write(paths['reference'], old)
    out = tmp_path / 'mixed-output'
    out.mkdir()
    original = abundance.fold_predictions

    def fit_or_fail(*args, **kwargs):
        if kwargs['seed'] == 20260925:
            raise replay.ReplayError('Synthetic third-seed fitting failure')
        return original(*args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(replay, 'validate_features', return_value=(
            {'identity': {}}, {'manifest': 'toy-manifest', 'archives': {}})))
        stack.enter_context(patch.object(replay, 'canonical_plan', return_value=({}, 'toy-plan')))
        stack.enter_context(patch.object(replay, 'load_likelihood_blocks', return_value=model_blocks))
        stack.enter_context(patch.object(stability, 'load_profiles', return_value=({}, {})))
        stack.enter_context(patch.object(abundance, 'build_panel', return_value=panel))
        stack.enter_context(patch.object(abundance, 'fold_predictions', side_effect=fit_or_fail))
        report = replay.replay_arm(tmp_path, out, tmp_path / 'features', 'abundance', 'gpt2', {})
    assert report['status'] == 'failed'
    assert report['seeds']['20260923']['status'] == 'reproduced_within_tolerance'
    assert report['seeds']['20260924']['status'] == 'reproduced_metric_mismatch'
    assert report['seeds']['20260925']['status'] == 'failed'
    rows = read_rows(out / 'gpt2.csv.gz')
    assert len(rows) == 60
    assert all(r['baseline_prediction'] and r['augmented_prediction'] for r in rows[:40])
    assert all(r['baseline_prediction'] == r['augmented_prediction'] == '' for r in rows[40:])
    assert all(r['status'] == 'reproduced_metric_mismatch' for r in rows[20:40])
    assert all(r['heldout_fold'] == '' for r in rows[40:])
    for r in rows:
        assert r['sample_id'] == replay.sample_id(report['original_input_sha256']['cohort'], r['unit'], r['mutation'])


def test_shared_sample_id_contract():
    import hashlib
    payload = json.dumps(['cohort-digest', 'unité', 'A12C'], ensure_ascii=False,
                         sort_keys=True, separators=(',', ':'), allow_nan=False)
    assert replay.sample_id('cohort-digest', 'unité', 'A12C') == hashlib.sha256(payload.encode()).hexdigest()
    cohort = toy_cohort()
    rows = replay.registry(cohort, 'abundance', 'cohort-digest')
    assert rows[0]['mutation'] == 'A1C'
    assert rows[0]['sample_id'] == replay.sample_id('cohort-digest', 'unit-0', 'A1C')
    assert len({r['sample_id'] for r in rows}) == len(rows)


def test_cli_refuses_existing_or_non_namespace_output(tmp_path):
    base = ['--endpoint', 'abundance', '--arm', 'gpt2', '--root', str(tmp_path),
            '--extraction', str(tmp_path / 'features')]
    with pytest.raises(SystemExit):
        replay.main([*base, '--out', str(tmp_path / 'unsafe')])
    existing = tmp_path / 'results/shared' / replay.NAMESPACE / 'existing'
    existing.mkdir(parents=True)
    sentinel = existing / 'keep'
    sentinel.write_text('unchanged')
    with pytest.raises(SystemExit):
        replay.main([*base, '--out', str(existing)])
    assert sentinel.read_text() == 'unchanged'


@pytest.mark.parametrize('endpoint', ['abundance', 'stability'])
def test_baseline_only_matches_joint_fit_and_retains_blank_augmented(tmp_path, endpoint):
    import torch
    torch.set_num_threads(4)
    records, panel, blocks = small_panel(endpoint)
    module = abundance if endpoint == 'abundance' else stability
    kwargs = {'first_stage': ('x',)} if endpoint == 'abundance' else {'purge': None}
    joint = module.fold_predictions(panel, blocks, {'S': ('x', 'G'), 'S_M': ('x', 'G', 'M')},
                                    seed=replay.SEEDS[0], device='cpu', **kwargs)
    with replay.capture_partitions(module) as partitions:
        single = module.fold_predictions(panel, {'x': blocks['x']}, {'S': ('x', 'G')},
                                         seed=replay.SEEDS[0], device='cpu', **kwargs)
    np.testing.assert_array_equal(single['predictions']['S'], joint['predictions']['S'])
    previous = dict(replay.summary_metrics(module, panel, joint, 'S', endpoint),
                    folds=joint['folds'], nuisance=joint['nuisance'])
    reference = {'per_seed' if endpoint == 'abundance' else 'seeds': {str(replay.SEEDS[0]): previous}}
    joined, checks = replay.validate_fit(single, reference, endpoint, replay.SEEDS[0], 'S', panel,
                                         partitions, baseline_only=True)
    summary = replay.summary_metrics(module, panel, single, 'S', endpoint, baseline_only=True)
    assert 'augmented_mse' not in summary and 'primary_likelihood' not in summary
    assert all(c['agrees'] for c in checks + replay.compare_metrics(summary, previous, endpoint, baseline_only=True))
    with replay.csv_stream(tmp_path / 'baseline.csv.gz') as writer:
        replay.serialize_seed(writer, records, endpoint=endpoint, arm='gpt2', seed=replay.SEEDS[0],
                              baseline='S', status='baseline_reproduced_within_tolerance_augmented_unavailable',
                              cohort_sha256='toy', result=single, heldout=joined, baseline_only=True)
    rows = read_rows(tmp_path / 'baseline.csv.gz')
    assert all(r['baseline_prediction'] and r['augmented_prediction'] == ''
               and r['heldout_fold'] != '' and 'augmented_unavailable' in r['status'] for r in rows)


def test_likelihood_only_schema_binds_plan_sequences_tokens_without_projected(tmp_path):
    import hashlib
    unit = toy_cohort()['domains'][0]
    sequences = [unit['wildtype'], *[v['sequence'] for v in unit['variants']]]
    tokens = {unit['name']: [dict(ids=[i, i + 1], pooled_positions=[0, 1]) for i in range(3)]}
    path = tmp_path / 'unit.npz'
    values = dict(variant_states=np.array([1, 2]), variant_positions=np.array([1, 2]),
                  likelihood=np.array([5., 7., 2.]), pooled_token_counts=np.full(3, 2),
                  token_offsets=np.arange(0, 7, 2), token_ids=np.array([0, 1, 1, 2, 2, 3]),
                  state_sequence_sha256=np.asarray([hashlib.sha256(s.encode()).hexdigest()
                                                     for s in sequences], dtype='<U64'))
    manifest = {'status': 'complete', 'identity': {'schema': 'stability_singles_likelihood_extraction_v1',
                'arm': 'gpt2', 'cohort_sha256': 'toy', 'plan_sha256': 'canonical-plan',
                'dtype': 'float32', 'batch_size': 1, 'budget': 1024},
                'backgrounds': [{'background': unit['name'], 'file': path.name}]}

    def save():
        np.savez(path, **values)
        manifest['backgrounds'][0]['sha256'] = replay.digest(path)
        write(tmp_path / 'manifest_gpt2.json', manifest)

    def validate():
        return replay.validate_features(tmp_path, 'gpt2', 'toy', [unit], plan={},
                                        expected_plan_sha256='canonical-plan')

    with patch.object(replay, 'tokenizer_records', return_value=(tokens, {})):
        save()
        validate()
        blocks = replay.load_likelihood_blocks(tmp_path, [unit], manifest)
        assert set(blocks) == {'M', 'T'}
        np.testing.assert_array_equal(blocks['M'], [[2.], [-3.]])
        np.testing.assert_array_equal(blocks['T'], replay.token_blocks([unit], tokens)['T'])
        manifest['identity']['plan_sha256'] = 'changed'
        save()
        with pytest.raises(replay.ReplayError, match='plan digest'):
            validate()
        manifest['identity']['plan_sha256'] = 'canonical-plan'
        values['state_sequence_sha256'][1] = '0' * 64
        save()
        with pytest.raises(replay.ReplayError, match='sequence hashes'):
            validate()
        values['state_sequence_sha256'][1] = hashlib.sha256(sequences[1].encode()).hexdigest()
        values['token_ids'][2] = 99
        save()
        with pytest.raises(replay.ReplayError, match='token content'):
            validate()
        values['token_ids'][2] = 1
        values['pooled_token_counts'][1] = 1
        save()
        with pytest.raises(replay.ReplayError, match='token content'):
            validate()


def test_cli_requires_extraction_only_without_baseline_flag(tmp_path):
    with pytest.raises(SystemExit):
        replay.main(['--endpoint', 'stability', '--out', str(tmp_path / 'out')])
