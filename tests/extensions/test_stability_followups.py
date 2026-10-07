"""Exact joins, pairing and objective licensing for the stability CPU extension."""
import copy
import json
from pathlib import Path

import scripts.capability.extensions.prepare_stability_followups as cli
from src.capability.core.io import sha256_file
from src.capability.interactions.pairwise_epistasis import ROSTER

import numpy as np
import pandas as pd
import pytest

import src.capability.extensions.stability_followups as follow
from src.capability.stability import stability_gate as gate
from src.capability.readouts.readout_analysis import family_folds


def tiny_cohort():
    return {'backgrounds': [{'name': 'p', 'group': 'family', 'wildtype': 'AA',
        'sequences': ['AA', 'CA', 'AD'], 'wildtype_combined_kcal_mol': 2.,
        'variants': [{'state': 1, 'sequence': 'CA', 'position': 1, 'mutant': 'C',
                      'ddg': 1., 'ddg_trypsin': 2., 'ddg_chymotrypsin': 3.},
                     {'state': 2, 'sequence': 'AD', 'position': 2, 'mutant': 'D',
                      'ddg': 2., 'ddg_trypsin': 3., 'ddg_chymotrypsin': 4.}]}]}


def samples():
    return pd.DataFrame({'sample_id': ['a', 'b'], 'unit_id': ['p', 'p'],
        'group_id': ['family', 'family'], 'variant_index': [0, 1], 'state': [1, 2],
        'positions_1based': [1, 2], 'source_pointer': ['/backgrounds/0/variants/0', '/backgrounds/0/variants/1'],
        'ddg': [1., 2.], 'ddg_trypsin': [2., 3.], 'ddg_chymotrypsin': [3., 4.],
        'wildtype_combined_kcal_mol': [2., 2.]})


def test_exact_sample_alignment_and_identity():
    follow.registry(tiny_cohort(), samples())
    with pytest.raises(ValueError, match='alignment'):
        follow.registry(tiny_cohort(), samples().iloc[::-1])
    with pytest.raises(ValueError, match='missing'):
        follow.registry(tiny_cohort(), samples().iloc[:1])
    changed = tiny_cohort()
    changed['backgrounds'][0]['variants'][0]['sequence'] = 'AD'
    # The declared sequence must agree with the indexed state, not just site/residue.
    with pytest.raises(ValueError):
        follow.registry(changed, samples())


def test_missing_predictions_fail_not_losses_reconstruction():
    with pytest.raises(ValueError, match='coverage'):
        follow.validate_predictions(['a'], [1.], ['a', 'b'])
    with pytest.raises(ValueError, match='missing'):
        follow.validate_predictions(['a', 'b'], [1., np.nan], ['a', 'b'])
    with pytest.raises(ValueError, match='33-arm'):
        follow.ranking_family(np.arange(3.), np.array(['g'] * 3), ['a', 'b', 'c'], {}, [], 'S2')


def test_channel_exact_join_no_pairwise_endpoint_dependence():
    reg = follow.registry(tiny_cohort(), samples())
    source = pd.DataFrame({'WT_name': ['p'] * 3, 'aa_seq': ['AA', 'CA', 'AD'],
                          'combined': [2., 3., 4.], 'deltaG_t': [4., 6., 7.], 'deltaG_c': [5., 8., 9.]})
    joined = follow.join_channels(reg, source)
    assert joined.trypsin_wt.tolist() == [4., 4.]
    panel = {'target': np.array([1., 2.]), 'states': {'y': np.zeros(3)}, 'pair_states': np.array([[0, 1], [0, 2]])}
    channel = follow.channel_panel(panel, joined, 'trypsin')
    np.testing.assert_array_equal(channel['target'], [2., 3.])
    np.testing.assert_array_equal(channel['states']['y'], [4., 6., 7.])
    np.testing.assert_array_equal(panel['states']['y'], np.zeros(3))
    with pytest.raises(ValueError, match='missing'):
        follow.join_channels(reg, source.iloc[1:])
    with pytest.raises(ValueError, match='duplicate'):
        follow.join_channels(reg, pd.concat([source, source.iloc[:1]]))


def test_dual_control_qualification():
    controls = {'qualified_control_set': ['ident', 'geom', 'chem', 'G'],
                'base_blocks': ['ident', 'geom'], 'candidate_order': list(gate.CANDIDATE_BLOCKS),
                'ladder': []}
    for candidate in gate.CANDIDATE_BLOCKS:
        controls['ladder'].append({'candidate': candidate, 'qualified': candidate in ('chem', 'G'),
            'spearman_increment': {str(s): {'point': .1} for s in gate.SPLIT_SEEDS},
            'per_seed_increment_kcal2_mol2': {str(s): -.1 for s in gate.SPLIT_SEEDS}})
    sets, _ = follow.control_sets(controls)
    assert sets['S'] == ('ident', 'geom', 'chem', 'G')
    assert set(sets['S2']) == set(sets['S']) | {'comp', 'prof2'}
    assert 'prof' not in sets['S2']
    controls['ladder'][0]['spearman_increment'][str(gate.SPLIT_SEEDS[0])]['point'] = 0.
    with pytest.raises(ValueError, match='qualification'):
        follow.control_sets(controls)


def test_rank_ties_and_nonestimable_groups_are_not_dropped():
    y = np.array([1., 1., 2., 1., 1., 1.])
    groups = np.array(['a'] * 3 + ['b'] * 3)
    names, values, missing = follow.rank_contrasts(y, np.arange(6.), -np.arange(6.), groups)
    assert names == ['a', 'b'] and missing == ['b'] and np.isnan(values[1])
    with pytest.raises(ValueError, match='nonestimable'):
        follow.rank_target(y, groups)
    target = follow.rank_target(y[:3], groups[:3])
    assert target[0] == target[1]
    assert abs(target.mean()) < 1e-12


def test_outer_split_record_and_held_label_independence():
    # A small real nested ridge fit, without model scores or mocks.
    groups = np.repeat(np.array([f'g{i:02d}' for i in range(12)]), 3)
    target = np.arange(len(groups), dtype=float) % 7
    rng = np.random.default_rng(1)
    panel = {'group': groups, 'site': np.array([f's{i}' for i in range(len(groups))]),
             'target': target, 'blocks': {'x': rng.normal(size=(len(groups), 2))}}
    frozen = {'seeds': {str(seed): {'folds': [{'held_groups': list(g)} for g in
               family_folds(groups, gate.OUTER_SPLITS, seed)]} for seed in gate.SPLIT_SEEDS}}
    partitions = follow.recorded_partitions(groups, frozen)
    first_seed = gate.SPLIT_SEEDS[0]
    first = partitions[str(first_seed)][0]
    assert not set(first['held_groups']) & set(first['training_groups'])
    assert sorted(g for part in first['inner_held_groups'] for g in part) == first['training_groups']
    original = gate.fold_predictions(panel, panel['blocks'], {'base': ('x',)}, seed=first_seed)
    changed = dict(panel)
    held = np.isin(groups, first['held_groups'])
    changed['target'] = target.copy()
    changed['target'][held] += 1000
    altered = gate.fold_predictions(changed, panel['blocks'], {'base': ('x',)}, seed=first_seed)
    np.testing.assert_array_equal(original['predictions']['base'][held], altered['predictions']['base'][held])
    bad = copy.deepcopy(frozen)
    bad['seeds'][str(first_seed)]['folds'][0]['held_groups'] = ['g99']
    with pytest.raises(ValueError, match='outer partition'):
        follow.recorded_partitions(groups, bad)


def test_G_calibration_excludes_held_absolute_and_WT_labels():
    rng = np.random.default_rng(8)
    groups = np.repeat(np.array([f'g{i:02d}' for i in range(12)]), 4)
    features = rng.normal(size=(len(groups), 3))
    states = {'features': features, 'y': features[:, 0] + rng.normal(size=len(groups)),
              'group': groups, 'weight': np.full(len(groups), .25)}
    pairs = np.array([[4 * group, 4 * group + mutant] for group in range(12) for mutant in range(1, 4)])
    held = np.array(['g00', 'g01'])
    inner = family_folds(groups[~np.isin(groups, held)], gate.INNER_SPLITS, 123)
    first, _ = gate.nuisance_response(states, pairs, held, inner)
    changed = dict(states)
    changed['y'] = states['y'].copy()
    changed['y'][np.isin(groups, held)] += 1000
    second, _ = gate.nuisance_response(changed, pairs, held, inner)
    np.testing.assert_array_equal(first, second)


def test_shared_33_arm_seed_average_ranking_family():
    from src.capability.interactions.pairwise_epistasis import ROSTER
    y = np.tile(np.arange(4.), 4)
    groups = np.repeat(np.array(['a', 'b', 'c', 'd']), 4)
    ids = np.array([str(i) for i in range(len(y))])
    records = {arm: {seed: {'sample_id': ids, 'baseline': -y, 'augmented': y}
                     for seed in gate.SPLIT_SEEDS} for arm in ROSTER}
    primary = follow.ranking_family(y, groups, ids, records, list(ROSTER), 'S2')
    sensitivity = follow.ranking_family(y, groups, ids, records, list(ROSTER), 'S')
    assert primary['role'] == 'primary_rank_qualified'
    assert sensitivity['role'] == 'same_prediction_MSE_comparison_sensitivity'
    assert primary['bands']['draws'] == 10000
    assert primary['bands']['groups'] == 4 and primary['bands']['family_size'] == 33
    np.testing.assert_allclose(primary['bands']['point'], 2.)


def test_mismatch_receipt_cannot_be_promoted_to_ranking_input(tmp_path):
    from scripts.capability.extensions.prepare_stability_followups import load_OOF
    from src.capability.interactions.pairwise_epistasis import ROSTER
    from src.capability.core.io import sha256_file
    archive = tmp_path / 'scores.npz'
    np.savez(archive, sample_id=np.array(['a']))
    receipt = tmp_path / 'receipt.json'
    follow.dump(receipt, {'status': 'reproduced_metric_mismatch'})
    item = {'path': str(archive), 'sha256': sha256_file(archive),
            'receipt': {'path': str(receipt), 'sha256': sha256_file(receipt)}}
    declaration = {'cohort_sha256': follow.COHORT_SHA256, 'fit_objective': 'mse',
                   'arms': {arm: item for arm in ROSTER}}
    with pytest.raises(ValueError, match='unaccepted/mismatched'):
        load_OOF(declaration, 'combined', {})


@pytest.fixture
def channel_oof(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'OUT', tmp_path)
    labels = tmp_path / 'channel-absolute-labels.csv.gz'
    pd.DataFrame({'sample_id': ['a']}).to_csv(labels, index=False)
    sets = {'S': ['ident', 'geom', 'G'], 'S2': ['ident', 'geom', 'G', 'comp']}
    qualification = tmp_path / 'channel-qualification.json'
    follow.dump(qualification, {channel: {
        'cohort_sha256': follow.COHORT_SHA256, 'channel_labels_sha256': sha256_file(labels),
        'qualified_control_set': sets['S'], 'rank_qualified_control_set': sets['S2']}
        for channel in ('trypsin', 'chymotrypsin')})
    archive = tmp_path / 'scores.npz'
    np.savez(archive, sample_id=np.array(['a']), **{
        f'{seed}|{control}{suffix}': np.array([1.]) for seed in gate.SPLIT_SEEDS
        for control in ('S', 'S2') for suffix in ('', '_M')})
    partitions = {str(seed): [{'held_groups': ['family']}] for seed in gate.SPLIT_SEEDS}
    declaration = {'cohort_sha256': follow.COHORT_SHA256, 'fit_objective': 'mse', 'arms': {}}
    for arm in ROSTER:
        receipt = tmp_path / f'{arm}.json'
        follow.dump(receipt, {'status': 'new_CPU_extension_not_frozen_recovery',
            'cohort_sha256': follow.COHORT_SHA256, 'arm': arm, 'objective': 'mse',
            'channel': 'trypsin', 'prediction_sha256': sha256_file(archive),
            'control_sets': sets, 'channel_qualification': {
                'path': str(qualification), 'sha256': sha256_file(qualification)},
            'scalar_provenance': {'verified': True}, 'folds': partitions})
        declaration['arms'][arm] = {'path': str(archive), 'sha256': sha256_file(archive),
            'receipt': {'path': str(receipt), 'sha256': sha256_file(receipt)}}
    return declaration, partitions, qualification


def revise_receipt(declaration, mutate):
    item = declaration['arms'][ROSTER[0]]
    path = Path(item['receipt']['path'])
    row = json.loads(path.read_text())
    mutate(row)
    follow.dump(path, row)
    item['receipt']['sha256'] = sha256_file(path)


@pytest.mark.parametrize('channel', ['trypsin', 'chymotrypsin'])
def test_channel_oof_requires_exact_qualified_metric_sets(channel_oof, channel):
    declaration, partitions, _ = channel_oof
    for item in declaration['arms'].values():
        path = Path(item['receipt']['path'])
        row = json.loads(path.read_text())
        row['channel'] = channel
        follow.dump(path, row)
        item['receipt']['sha256'] = sha256_file(path)
    result = cli.load_OOF(declaration, channel, partitions)
    assert set(result) == {'S', 'S2'} and set(result['S']) == set(ROSTER)
    for control in ('S', 'S2'):
        revise_receipt(declaration, lambda row: row['control_sets'].__setitem__(control, ['ident', 'geom']))
        with pytest.raises(ValueError, match='S/S2 control qualification differs'):
            cli.load_OOF(declaration, channel, partitions)
        revise_receipt(declaration, lambda row: row['control_sets'].__setitem__(control, result_sets(control)))
    revise_receipt(declaration, lambda row: row.pop('control_sets'))
    with pytest.raises(ValueError, match='S/S2 control qualification differs'):
        cli.load_OOF(declaration, channel, partitions)


def result_sets(control):
    return ['ident', 'geom', 'G'] + (['comp'] if control == 'S2' else [])


@pytest.mark.parametrize('defect, message', [
    ('missing_binding', 'missing channel qualification artifact binding'),
    ('wrong_hash', 'qualification artifact hash differs'),
    ('wrong_cohort', 'qualification input binding differs'),
    ('wrong_labels', 'qualification input binding differs'),
    ('wrong_channel', 'endpoint identity differs'),
    ('no_scalar_provenance', 'baseline-only OOF'),
    ('wrong_prediction_hash', 'endpoint identity differs'),
    ('wrong_outer_fold', 'outer fold identity differs'),
])
def test_channel_oof_rejects_binding_and_original_gate_defects(channel_oof, defect, message):
    declaration, partitions, qualification = channel_oof
    if defect in ('wrong_cohort', 'wrong_labels'):
        rows = json.loads(qualification.read_text())
        key = 'cohort_sha256' if defect == 'wrong_cohort' else 'channel_labels_sha256'
        rows['trypsin'][key] = 'wrong'
        follow.dump(qualification, rows)
        revise_receipt(declaration, lambda row: row['channel_qualification'].__setitem__('sha256', sha256_file(qualification)))
    else:
        def mutate(row):
            if defect == 'missing_binding':
                row.pop('channel_qualification')
            elif defect == 'wrong_hash':
                row['channel_qualification']['sha256'] = 'wrong'
            elif defect == 'wrong_channel':
                row['channel'] = 'combined'
            elif defect == 'no_scalar_provenance':
                row['scalar_provenance'] = None
            elif defect == 'wrong_prediction_hash':
                row['prediction_sha256'] = 'wrong'
            else:
                row['folds'][str(gate.SPLIT_SEEDS[0])][0]['held_groups'] = ['wrong']
        revise_receipt(declaration, mutate)
    with pytest.raises(ValueError, match=message):
        cli.load_OOF(declaration, 'trypsin', partitions)


@pytest.mark.parametrize('key', ['sha256', 'receipt'])
def test_channel_oof_preserves_manifest_hash_gate(channel_oof, key):
    declaration, partitions, _ = channel_oof
    item = declaration['arms'][ROSTER[0]]
    if key == 'receipt':
        item['receipt']['sha256'] = 'wrong'
    else:
        item['sha256'] = 'wrong'
    with pytest.raises(ValueError, match='OOF provenance hash differs'):
        cli.load_OOF(declaration, 'trypsin', partitions)


def test_matched_channel_contrast_blocks_different_qualified_baselines(channel_oof):
    declaration, _, _ = channel_oof
    matched = {channel: copy.deepcopy(declaration) for channel in ('combined', 'trypsin', 'chymotrypsin')}
    cli.require_common_channel_controls(matched)
    # A separately valid channel qualification may select a different baseline;
    # it still cannot license a pure channel contrast against this one.
    path = Path(declaration['arms'][ROSTER[0]]['receipt']['path'])
    other = path.parent / 'different-baseline.json'
    row = json.loads(path.read_text())
    row['control_sets']['S2'] = ['ident', 'geom', 'G']
    follow.dump(other, row)
    matched['chymotrypsin']['arms'][ROSTER[0]]['receipt']['path'] = str(other)
    with pytest.raises(ValueError, match='common-baseline qualification required'):
        cli.require_common_channel_controls(matched)


def setup_small_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'OUT', tmp_path)
    labels = tmp_path / 'channel-absolute-labels.csv.gz'
    pd.DataFrame({'sample_id': ['a']}).to_csv(labels, index=False)
    follow.dump(tmp_path / 'partitions.json', {})
    follow.dump(tmp_path / 'support.json', {'prepared_artifacts': {
        p.name: sha256_file(p) for p in (labels, tmp_path / 'partitions.json')}})
    reg = pd.DataFrame({'sample_id': ['a']})
    monkeypatch.setattr(cli, 'inputs', lambda: ({}, {}, reg, {}, {}))
    return labels


def test_generated_noncombined_fit_receipt_binds_qualification(tmp_path, monkeypatch):
    labels = setup_small_cli(tmp_path, monkeypatch)
    qualification = tmp_path / 'qualification.json'
    follow.dump(qualification, {'trypsin': {'cohort_sha256': follow.COHORT_SHA256,
        'channel_labels_sha256': sha256_file(labels),
        'qualified_control_set': result_sets('S'), 'rank_qualified_control_set': result_sets('S2')}})
    monkeypatch.setattr(follow, 'control_sets', lambda _: ({}, {}))
    monkeypatch.setattr(follow, 'channel_panel', lambda *args: {})
    monkeypatch.setattr(follow, 'fit_extension', lambda *args, **kwargs: {})
    out = tmp_path / 'fit-check'
    monkeypatch.setattr('sys.argv', ['prepare_stability_followups.py', 'fit', '--channel', 'trypsin',
                                   '--channel-qualification', str(qualification), '--out', str(out), '--threads', '1'])
    cli.main()
    receipt = json.loads((out / 'fit-receipt.json').read_text())
    assert receipt['channel_qualification'] == {'path': str(qualification.resolve()), 'sha256': sha256_file(qualification)}
    assert receipt['control_sets'] == {'S': result_sets('S'), 'S2': result_sets('S2')}
    assert json.loads((out / 'fit-execution.json').read_text())['status'] == 'complete'


def test_qualification_execution_persists_start_bindings(tmp_path, monkeypatch):
    labels = setup_small_cli(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, 'input_paths', lambda: {'cohort': labels})
    out = tmp_path / 'qualification-check'
    starting_hash = sha256_file(labels)

    def ladder(*args):
        running = json.loads((out / 'qualify-channels-execution.json').read_text())
        assert running['status'] == 'running'
        assert running['input_bindings']['cohort']['sha256'] == starting_hash
        assert running['input_bindings']['channel-absolute-labels.csv.gz']['sha256'] == starting_hash
        script_path = Path(str(cli.__file__))
        assert running['code_sha256'][str(script_path.relative_to(cli.ROOT))] == sha256_file(script_path)
        # Simulate a later input edit in this isolated fixture, not the prepared outputs.
        labels.write_bytes(b'later bytes')
        return {'trypsin': {}}

    monkeypatch.setattr(follow, 'channel_qualification', ladder)
    monkeypatch.setattr('sys.argv', ['prepare_stability_followups.py', 'qualify-channels', '--out', str(out), '--threads', '1'])
    cli.main()
    receipt = json.loads((out / 'qualify-channels-execution.json').read_text())
    assert receipt['status'] == 'complete'
    assert receipt['input_bindings']['channel-absolute-labels.csv.gz']['sha256'] == starting_hash
    qualified = json.loads((out / 'channel-qualification.json').read_text())
    assert qualified['trypsin']['channel_labels_sha256'] == starting_hash
