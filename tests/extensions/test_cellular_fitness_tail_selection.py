"""Tail selection: declared budget, undefined cells stay null, replay gates.

The conditions that must hold whatever arrives: the selection rule is the
declared one and reads no label; a group too small for the budget yields a null
cell that is never imputed; a shared band is refused rather than computed over a
repaired matrix; the baseline is the frozen qualified set and a reference that
declares another set is not accepted; and the frozen-panel gate rejects a replay
that deviates beyond tolerance.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.capability.core.io import sha256_file
from src.capability.extensions import phenotype_breadth as breadth
from scripts.capability.extensions import analyse_cellular_fitness_tail_selection as tail


SEED = breadth.SPLIT_SEEDS[0]
LABELS = np.asarray([0.1, 0.2, 0.3, 0.4])
INVERSE = np.asarray([0.4, 0.3, 0.2, 0.1])
ROW_IDS = np.asarray(['r0', 'r1', 'r2', 'r3'])


def test_the_budget_is_the_declared_two_percent_of_the_admission_cap():
    assert tail.GROUP_CAP == 256
    assert tail.K == 6 == math.ceil(tail.SELECTION_FRACTION * tail.GROUP_CAP)


def test_percentile_is_the_within_group_rank_and_runs_with_higher_fitness():
    percentiles = tail.tail_percentile(LABELS)
    assert list(percentiles) == [0.25, 0.5, 0.75, 1.0]
    assert percentiles.mean() == pytest.approx(0.625)
    # Ties share the average rank rather than being split by order.
    assert list(tail.tail_percentile(np.asarray([1.0, 1.0, 2.0]))) == [0.5, 0.5, 1.0]
    with pytest.raises(ValueError):
        tail.tail_percentile(np.asarray([]))


def test_selection_keeps_the_largest_and_breaks_boundary_ties_by_identity():
    chosen = tail.select_top_k(np.asarray([0.4, 0.3, 0.2, 0.1]), ROW_IDS, 2)
    assert chosen is not None and list(chosen) == [0, 1]
    # Two predictions tie at the boundary; the ascending row id decides and the
    # label is never read.
    tied_ids = np.asarray(['b', 'a', 'c', 'd'])
    tied = tail.select_top_k(np.asarray([1.0, 1.0, 0.0, 0.0]), tied_ids, 2)
    assert tied is not None and sorted(tied_ids[tied].tolist()) == ['a', 'b']


def test_an_undefined_selection_is_none_and_never_a_zero():
    assert tail.select_top_k(np.asarray([1.0]), ROW_IDS[:1], 2) is None
    assert tail.select_top_k(np.asarray([1.0, np.nan, 0.0, 0.0]), ROW_IDS, 2) is None


def groups_vector(*, sizes=(4, 4)):
    return np.asarray([f'g{i}' for i, size in enumerate(sizes) for _ in range(size)])


def test_the_tail_contrast_is_augmented_minus_baseline_on_the_group_mean_percentile():
    groups = groups_vector()
    labels = np.tile(LABELS, 2)
    ids = np.asarray([f'g{i}:r{j}' for i in range(2) for j in range(4)])
    cells, membership = tail.arm_tail_contributions(
        'arm', [SEED], {SEED: np.tile(INVERSE, 2)}, {SEED: np.tile(LABELS, 2)},
        labels, ids, groups, k=2)
    for group in ('g0', 'g1'):
        assert cells[group]['baseline'] == pytest.approx(0.375)
        assert cells[group]['augmented'] == pytest.approx(0.875)
        assert cells[group]['contrast'] == pytest.approx(0.5)
    assert len(membership) == 2 and membership[0]['baseline_rows'] == ['g0:r0', 'g0:r1']
    assert membership[0]['augmented_rows'] == ['g0:r3', 'g0:r2']


def test_a_group_below_the_budget_is_null_and_the_band_is_unavailable():
    groups = np.asarray(['g0', 'g0', 'g1'])
    labels = np.asarray([0.1, 0.2, 0.3])
    ids = np.asarray(['a', 'b', 'c'])
    prediction = np.asarray([0.3, 0.2, 0.1])
    cells, _ = tail.arm_tail_contributions('arm', [SEED], {SEED: prediction}, {SEED: prediction},
                                           labels, ids, groups, k=2)
    assert cells['g0']['contrast'] is not None
    assert cells['g1'] == {'baseline': None, 'augmented': None, 'contrast': None, 'seeds': []}
    band = tail.shared_band({'arm': cells}, ['arm'], ['g0', 'g1'])
    assert band['status'] == 'unavailable'
    assert band['missing_cells'] == [{'arm': 'arm', 'group': 'g1'}]
    assert 'inference' not in band and 'imputed' in band['reason']


def test_a_seed_that_does_not_resolve_nulls_the_whole_group_cell():
    groups = groups_vector(sizes=(4,))
    labels = LABELS
    ids = ROW_IDS
    # The second declared seed carries a nonfinite prediction on this group.
    broken = np.asarray([1.0, np.nan, 0.0, 0.0])
    cells, membership = tail.arm_tail_contributions(
        'arm', [SEED, breadth.SPLIT_SEEDS[1]], {SEED: LABELS, breadth.SPLIT_SEEDS[1]: broken},
        {SEED: LABELS, breadth.SPLIT_SEEDS[1]: broken}, labels, ids, groups, k=2)
    assert cells['g0']['contrast'] is None
    assert membership == []


def test_descriptive_levels_are_the_random_mean_and_the_label_oracle():
    levels = tail.descriptive_levels(LABELS, ROW_IDS, groups_vector(sizes=(4,)), k=2)
    assert levels['random_expected'] == pytest.approx(0.625)
    assert levels['oracle_best_k'] == pytest.approx(0.875)
    assert 'neither level is inferential' in levels['note']
    # A group that cannot fill the budget has no oracle, and the pooled level is
    # then unavailable rather than averaged over the groups that resolved.
    short = tail.descriptive_levels(LABELS[:1], ROW_IDS[:1], np.asarray(['g0']), k=2)
    assert short['oracle_best_k'] is None
    assert short['random_expected'] == pytest.approx(1.0)


def test_a_complete_shared_band_covers_every_arm_and_resolves_the_contrast():
    groups = [f'g{i}' for i in range(9)]
    cells = {arm: {group: {'baseline': 0.5, 'augmented': 0.5 + 0.1 * (index == 0),
                           'contrast': 0.1 * (index == 0), 'seeds': [SEED]}
                   for index, group in enumerate(groups)}
             for arm in ('negative', 'positive')}
    cells['negative'] = {group: {'baseline': 0.5, 'augmented': 0.45, 'contrast': -0.05,
                                 'seeds': [SEED]} for group in groups}
    band = tail.shared_band(cells, ['negative', 'positive'], groups)
    assert band['status'] == 'complete'
    assert band['inference']['groups'] == 9
    assert band['inference']['draws'] == breadth.BOOTSTRAP_DRAWS
    assert [row['arm'] for row in band['arms']] == ['negative', 'positive']
    assert band['resolved_positive'] + band['resolved_negative'] <= len(band['arms'])


def test_preflight_refuses_a_machine_below_the_declared_floors():
    record = dict(tail.resources('cpu'))
    record['memory_available_bytes'] = 1
    with pytest.raises(SystemExit, match='resource preflight failed'):
        tail.preflight_resources(record)
    tail.preflight_resources(dict(tail.resources('cpu')))


def frozen_reference(tmp_path, *, qualified=tail.BASELINE):
    qualification = tmp_path / 'qual'
    qualification.mkdir(exist_ok=True)
    (qualification / 'cohort_qualification.json').write_text('{}', encoding='utf-8')
    profiles = tmp_path / 'profiles.npz'
    profiles.write_bytes(b'profiles')
    reference = {
        'cohort': tail.COHORT, 'endpoint': tail.ENDPOINT, 'status': 'admitted',
        'model_column': 'M_rank', 'metric': 'spearman', 'qualified_controls': list(qualified),
        'arms': ['armA'], 'groups': ['g0'], 'split_seeds': list(breadth.SPLIT_SEEDS),
        'qualification_sha256': sha256_file(qualification / 'cohort_qualification.json'),
        'profiles': {'sha256': sha256_file(profiles)}, 'support': {'rows': 10}}
    path = tmp_path / 'reference.json'
    path.write_text(json.dumps(reference), encoding='utf-8')
    return reference, path, qualification, profiles


def test_the_baseline_is_the_frozen_qualified_set_and_is_never_requalified(tmp_path):
    reference, path, qualification, profiles = frozen_reference(
        tmp_path, qualified=('ident', 'geom', 'chem', 'prof'))
    admission = {'support': {'rows': 10}}
    with pytest.raises(SystemExit, match='frozen evolutionary baseline'):
        tail.load_reference(path, admission, qualification, profiles, ['armA'], ['g0'])
    good, good_path, qualification, profiles = frozen_reference(tmp_path)
    assert tail.load_reference(good_path, admission, qualification, profiles,
                               ['armA'], ['g0']) == good


def test_the_reference_must_be_fitted_on_this_qualification_and_profile_file(tmp_path):
    _, path, qualification, profiles = frozen_reference(tmp_path)
    (qualification / 'cohort_qualification.json').write_text('{"other": true}', encoding='utf-8')
    with pytest.raises(SystemExit, match='different qualification run'):
        tail.load_reference(path, {'support': {'rows': 10}}, qualification, profiles,
                            ['armA'], ['g0'])


def synthetic_replay(arms=('armA',), groups=tuple(f'g{i}' for i in range(9))):
    cells = [{'arm': arm, 'seed': int(SEED), 'background': group, 'group': group, 'rows': 4,
              'control': 0.5, 'control_plus_model': 0.6, 'increment': 0.1}
             for arm in arms for group in groups]
    return {'predictions': {}, 'cells': cells,
            'design_digests': {arm: f'design-{arm}' for arm in arms},
            'target_digests': {arm: {SEED: 'target'} for arm in arms},
            'alphas': {arm: {SEED: {'control': [1.0], 'augmented': [1.0]}} for arm in arms}}


def synthetic_reference(replay, arms, groups):
    band, resolved = breadth.arm_band(replay['cells'], list(arms), list(groups))
    return {'fits': {arm: {str(SEED): {
                 'design_sha256': replay['design_digests'][arm],
                 'target_sha256': replay['target_digests'][arm][SEED],
                 'alphas_control': replay['alphas'][arm][SEED]['control'],
                 'alphas_augmented': replay['alphas'][arm][SEED]['augmented']}} for arm in arms},
            'per_background': copy.deepcopy(replay['cells']),
            'inference': band, 'arm_results': resolved, 'groups': list(groups)}


def test_the_replay_gate_accepts_the_frozen_record_and_rejects_a_deviation():
    arms, groups = ('armA',), tuple(f'g{i}' for i in range(9))
    replay = synthetic_replay(arms, groups)
    reference = synthetic_reference(replay, arms, groups)
    verified = tail.verify_replay(reference, list(arms), [SEED], replay)
    assert verified['verified'] and verified['failures'] == []
    assert verified['digests_compared'] == 4 * len(arms)
    broken = copy.deepcopy(reference)
    broken['per_background'][0]['control'] += 1e-6
    rejected = tail.verify_replay(broken, list(arms), [SEED], replay)
    assert not rejected['verified']
    assert rejected['max_absolute_deviation'] > tail.REPLAY_TOLERANCE
    drifted = copy.deepcopy(reference)
    drifted['fits']['armA'][str(SEED)]['design_sha256'] = 'drifted'
    assert not tail.verify_replay(drifted, list(arms), [SEED], replay)['verified']


def test_cohort_eligibility_admits_only_the_frozen_nine_targets(monkeypatch):
    units = [f'u{i}' for i in range(9)]
    groups = {unit: unit for unit in units}
    rows = [SimpleNamespace(unit=unit) for unit in units]
    admission = {'status': 'admitted', 'ranking_licensed': True, 'cohort': tail.COHORT}
    monkeypatch.setattr(tail, 'group_gene_by_unit',
                        lambda root: {unit: gene for unit, gene in zip(units, tail.FROZEN_GENES)})
    eligibility = tail.cohort_eligibility(Path('.'), admission, groups,
                                          {'assignment': dict(groups)}, rows)
    assert eligibility['genes'] == sorted(tail.FROZEN_GENES)
    monkeypatch.setattr(tail, 'group_gene_by_unit',
                        lambda root: {unit: gene for unit, gene in
                                      zip(units, (*tail.FROZEN_GENES[:8], 'PALB2'))})
    with pytest.raises(SystemExit, match='outside the frozen target set'):
        tail.cohort_eligibility(Path('.'), admission, groups,
                                {'assignment': dict(groups)}, rows)


def test_cohort_eligibility_refuses_a_merged_or_drifting_grouping(monkeypatch):
    units = [f'u{i}' for i in range(9)]
    groups = {unit: unit for unit in units}
    rows = [SimpleNamespace(unit=unit) for unit in units]
    admission = {'status': 'admitted', 'ranking_licensed': True, 'cohort': tail.COHORT}
    monkeypatch.setattr(tail, 'group_gene_by_unit',
                        lambda root: {unit: gene for unit, gene in zip(units, tail.FROZEN_GENES)})
    merged = dict(groups, u1='u0')
    with pytest.raises(SystemExit, match='failed'):
        tail.cohort_eligibility(Path('.'), admission, merged,
                                {'assignment': dict(groups)}, rows)
    drifted = {'assignment': dict(groups, u0='other')}
    with pytest.raises(SystemExit, match='does not match the re-derived one'):
        tail.cohort_eligibility(Path('.'), admission, groups, drifted, rows)