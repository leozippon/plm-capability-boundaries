"""Frozen ladder definitions, strict joins, aggregation and reuse gating.

Every fixture here is synthetic, so these tests run without the ignored
prediction archives or any phenotype data.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.capability.extensions import overlap as o
from src.capability.extensions import progression as p
from src.capability.extensions import statistical_ladder as sl

SEEDS = sl.SEEDS


def signature(groups, seed=SEEDS[0]):
    from src.capability.readouts.readout_analysis import family_folds

    result = []
    for fold, held in enumerate(family_folds(groups, 5, seed)):
        training = sorted(set(groups) - set(held))
        inner = family_folds(training, 4, seed + 100 + fold)
        result.append((fold, tuple(sorted(held)), tuple(training),
                       tuple(tuple(sorted(values)) for values in inner)))
    return result


def fixture_panel(groups=None):
    rng = np.random.default_rng(7)
    groups = list('abcdefghij') if groups is None else list(groups)
    families = np.repeat(np.asarray(groups, object), 3)
    aids = families.copy()
    samples = [dict(assay=str(a), cluster=str(f), mutation=f'A1{index}', variant_index=index % 3,
                    original_index=index, sample_id=f'{a}-{index}')
               for index, (a, f) in enumerate(zip(aids, families))]
    blocks = {name: rng.normal(size=(len(aids), width)) for name, width in sl.WIDTHS.items()}
    return blocks, rng.normal(size=len(aids)), aids, families, samples, rng


def metric_rows(table, models, seeds=SEEDS, designs=None):
    """Expand { (assay, family): {seed: {design: value}} } into metric rows."""
    rows = []
    for model in models:
        for seed in seeds:
            for (assay, family), cells in table.items():
                rows.append(dict(model=model, seed=seed, assay=assay, cluster=family,
                                 spearman=dict(cells[seed]['spearman']),
                                 rank_mse=dict(cells[seed]['rank_mse'])))
    return rows


def test_frozen_registry_stages_and_declared_family():
    registry = sl.registry()
    assert len(sl.CONTROL_SETS) == 6 and len(sl.DIRECTIONS) == 2
    assert registry['control_set_order'] == ['P', 'P+E', 'P+G', 'P+E+G', 'C0', 'C0+E+G']
    assert sl.stage_designs('full') == ('M', 'MP', 'MPE', 'MPG', 'MPEG', 'P', 'PE', 'PG', 'PEG',
                                        'BPL', 'BMPL', 'BPLEG', 'BMPLEG')
    assert sl.stage_designs('profile_only') == ('M', 'MP', 'P', 'BPL', 'BMPL')
    assert sl.reused_designs('full') == ('M', 'MP', 'BPL', 'BMPL')
    assert sl.reused_designs('profile_only') == ('M', 'MP', 'BPL', 'BMPL')
    for name, columns in sl.DESIGNS.items():
        assert ''.join(columns) == name
        assert tuple(sorted(columns, key=sl.BLOCK_ORDER.index)) == columns
    # The full ladder is exactly 6 sets x 33 models x 2 directions per metric.
    assert 6 * 33 * 2 == 396
    for entry in sl.CONTROL_SETS.values():
        assert set(sl.DESIGNS[entry['joint']]) - set(sl.DESIGNS[entry['control']]) <= {'M'}
        assert set(sl.DESIGNS[entry['joint']]) - set(sl.DESIGNS[entry['model']]) <= set(
            sl.DESIGNS[entry['control']])
    assert sl.CONTROL_SETS['C0']['control'] == 'BPL'
    assert sl.CONTROL_SETS['C0+E+G']['joint'] == 'BMPLEG'
    assert sl.BOOTSTRAP_DRAWS == 10000 and sl.BOOTSTRAP_SEED == 20261106


def test_prepare_block_reranks_only_the_declared_first_columns():
    blocks, _, aids, _, _, rng = fixture_panel()
    blocks['M'][:3, 0] = [-1., 0., 0.]
    blocks['P'][:3, 0] = [-9., -8., -8.]
    blocks['E'][:3, 0] = [5., 5., 4.]
    blocks['G'][:3, 0] = [1., 2., 2.]
    for name in ('M', 'P', 'E', 'G'):
        prepared = sl.prepare_block(name, blocks[name], aids)
        np.testing.assert_array_equal(prepared[:, 0], o.rerank(blocks[name][:, 0], aids))
        np.testing.assert_array_equal(prepared[:, 1:], blocks[name][:, 1:])
    assert sl.prepare_block('E', blocks['E'], aids)[1, 0] == sl.prepare_block('E', blocks['E'], aids)[0, 0]
    for name in ('B', 'L'):
        prepared = sl.prepare_block(name, blocks[name], aids)
        np.testing.assert_array_equal(prepared, blocks[name])
    with pytest.raises(ValueError, match='invalid B'):
        sl.prepare_block('B', blocks['B'][:, :-1], aids)
    with pytest.raises(ValueError, match='invalid P'):
        sl.prepare_block('P', np.full_like(blocks['P'], np.nan), aids)
    with pytest.raises(ValueError, match='unknown block'):
        sl.prepare_block('Z', blocks['P'], aids)


def test_design_matrix_concatenates_in_canonical_order_and_refuses_missing_blocks():
    blocks, _, aids, _, _, _ = fixture_panel()
    prepared = {name: sl.prepare_block(name, blocks[name], aids) for name in ('B', 'M', 'P', 'L')}
    matrix = sl.design_matrix(prepared, 'BMPL')
    assert matrix.shape == (30, 444 + 1 + 14 + 111)
    np.testing.assert_array_equal(matrix[:, 445:459], prepared['P'])
    assert sl.design_matrix(prepared, 'MP').shape == (30, 15)
    with pytest.raises(ValueError, match='needs blocks'):
        sl.design_matrix(prepared, 'BPLEG')
    with pytest.raises(ValueError, match='unknown design'):
        sl.design_matrix(prepared, 'NOPE')


def test_real_fits_match_the_frozen_progression_pipeline_and_share_folds():
    torch.set_num_threads(2)
    blocks, measured, aids, families, samples, _ = fixture_panel()
    sig = signature(families)
    expected, records, checks = p.fit_cell(blocks, measured, aids, families, sig, 'full', samples)
    prepared = {name: sl.prepare_block(name, blocks[name], aids) for name in ('B', 'M', 'P', 'L')}
    membership = sl.projected_membership(sig, families)
    assert all(sl.realized_membership(record) == membership for record in records.values())
    for name in ('M', 'MP', 'BPL', 'BMPL'):
        prediction, folds, design_sha256 = sl.fit_design(prepared, measured, aids, families, sig, name)
        np.testing.assert_array_equal(prediction, expected[name])
        assert design_sha256 == checks[name]['design_sha256']
        assert sl.realized_membership(folds) == membership
        assert [fold['fold'] for fold in folds] == [0, 1, 2, 3, 4]


def test_assay_rows_report_undefined_constant_scores_without_dropping_them():
    blocks, measured, aids, families, samples, _ = fixture_panel()
    predictions = {'M': o.rerank(blocks['M'][:, 0], aids), 'P': o.rerank(blocks['P'][:, 0], aids)}
    predictions['MP'] = predictions['M'] + predictions['P']
    rows = sl.assay_rows(predictions, measured, aids, families, model='arm', seed=SEEDS[0],
                         designs=('M', 'P', 'MP'))
    assert len(rows) == 10
    assert {row['assay'] for row in rows} == set(aids)
    assert all(set(row['spearman']) == {'M', 'P', 'MP'} for row in rows)
    constant = {name: np.ones(30) for name in predictions}
    rows = sl.assay_rows(constant, measured, aids, families, model='arm', seed=SEEDS[0],
                         designs=('M', 'MP', 'P'))
    assert all(row['spearman']['M'] is None for row in rows)
    assert all(row['rank_mse']['M'] >= 0 for row in rows)
    with pytest.raises(ValueError, match='missing BMPL'):
        sl.assay_rows(predictions, measured, aids, families, model='arm', seed=SEEDS[0],
                      designs=('M', 'P', 'MP', 'BMPL'))


def test_direction_signs_seed_then_assay_then_family_weighting():
    # x and y share family f, z is family g; joint varies across the three seeds.
    joint = {SEEDS[0]: (0.30, 0.10, 0.00), SEEDS[1]: (0.40, 0.10, 0.20), SEEDS[2]: (0.50, 0.40, 0.40)}
    control = {'x': 0.10, 'y': 0.00, 'z': -0.20}
    model = {'x': 0.20, 'y': 0.10, 'z': 0.00}
    family = {'x': 'f', 'y': 'f', 'z': 'g'}
    table = {}
    for position, assay in enumerate('xyz'):
        table[(assay, family[assay])] = {}
        for seed in SEEDS:
            table[(assay, family[assay])][seed] = dict(
                spearman={'P': control[assay], 'M': model[assay], 'MP': joint[seed][position]},
                rank_mse={'P': 1 - control[assay], 'M': 1 - model[assay],
                          'MP': 1 - joint[seed][position]})
    rows = metric_rows(table, models=['arm'])
    result = sl.ladder_matrix(rows, 'spearman', control_sets={'P': sl.CONTROL_SETS['P']},
                              models=['arm'], seeds=SEEDS)
    assert result['families'] == ['f', 'g']
    assert result['labels'] == [('arm', 'P', 'model_given_controls'),
                                ('arm', 'P', 'controls_given_model')]
    matrix = result['matrix']
    # family f: seed means of the joint arm are .40 and .20, so the paired
    # reductions are .30/.20 (x) and .20/.10 (y), and the family carries .25/.15.
    np.testing.assert_allclose(matrix[:, 0], [0.25, 0.40])
    np.testing.assert_allclose(matrix[:, 1], [0.15, 0.20])
    np.testing.assert_allclose(matrix.mean(axis=0), [0.325, 0.175])
    # rank MSE reverses the subtraction, so improvement keeps the same sign.
    rank = sl.ladder_matrix(rows, 'rank_mse', control_sets={'P': sl.CONTROL_SETS['P']},
                            models=['arm'], seeds=SEEDS)
    np.testing.assert_allclose(rank['matrix'], matrix)
    with pytest.raises(ValueError, match='unknown metric'):
        sl.ladder_matrix(rows, 'r2', models=['arm'])
    with pytest.raises(ValueError, match='declared'):
        sl.ladder_matrix(rows[:-1], 'spearman', control_sets={'P': sl.CONTROL_SETS['P']},
                         models=['arm'], seeds=SEEDS)
    with pytest.raises(ValueError, match='duplicate'):
        sl.ladder_matrix(rows + [rows[0]], 'spearman', control_sets={'P': sl.CONTROL_SETS['P']},
                         models=['arm'], seeds=SEEDS)


def test_undefined_assay_correlation_makes_the_family_not_estimable():
    blocks, measured, aids, families, samples, _ = fixture_panel()
    predictions = {'M': np.ones(30), 'P': o.rerank(blocks['P'][:, 0], aids), 'MP': np.zeros(30)}
    rows = []
    for seed in SEEDS:
        rows.extend(sl.assay_rows(predictions, measured, aids, families, model='arm', seed=seed,
                                  designs=('M', 'P', 'MP')))
    result = sl.ladder_matrix(rows, 'spearman', control_sets={'P': sl.CONTROL_SETS['P']},
                              models=['arm'], seeds=SEEDS)
    assert len(result['undefined']) == 2
    assert not np.isfinite(result['matrix']).any()
    inference = sl.infer(result['matrix'], result['labels'], metric='spearman')
    assert inference['status'] == 'not_estimable'
    assert 'hypotheses' not in inference


def test_absolute_matrix_is_model_independent_for_designs_without_M():
    table = {}
    for assay, family in (('x', 'f'), ('y', 'f'), ('z', 'g')):
        table[(assay, family)] = {seed: dict(spearman={'P': 0.1, 'MP': 0.2, 'M': 0.3},
                                             rank_mse={'P': 0.9, 'MP': 0.8, 'M': 0.7})
                                   for seed in SEEDS}
    rows = metric_rows(table, models=['a', 'b'])
    result = sl.absolute_matrix(rows, 'spearman', designs=('M', 'MP', 'P'),
                                models=['a', 'b'], seeds=SEEDS)
    assert result['labels'] == [('a', 'M'), ('a', 'MP'), ('a', 'P'),
                                ('b', 'M'), ('b', 'MP'), ('b', 'P')]
    matrix = result['matrix']
    np.testing.assert_allclose(matrix.mean(axis=0), [0.3, 0.2, 0.1, 0.3, 0.2, 0.1])


def test_inference_reports_native_primary_and_labelled_standardized_sensitivity():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(12, 4)) + np.array([3.0, 0.3, 0.05, -3.0])
    labels = [('arm', 'P', 'model_given_controls'), ('arm', 'P', 'controls_given_model'),
              ('arm', 'C0', 'model_given_controls'), ('arm', 'C0', 'controls_given_model')]
    result = sl.infer(matrix, labels, metric='spearman')
    assert result['status'] == 'complete'
    native = result['native']
    assert native['draws'] == 10000 and native['family_size'] == 4
    assert 'native units' in native['method'] and 'centered' in native['method']
    critical = native['joint_critical_max_absolute_deviation']
    for position, hypothesis in enumerate(result['hypotheses']):
        assert hypothesis['point'] == pytest.approx(matrix[:, position].mean())
        assert hypothesis['simultaneous'] == pytest.approx(
            [matrix[:, position].mean() - critical, matrix[:, position].mean() + critical])
        low, high = hypothesis['simultaneous']
        expected = 'positive' if low > 0 else 'negative' if high < 0 else 'unresolved'
        assert hypothesis['verdict'] == expected
        assert hypothesis['pointwise'][0] <= hypothesis['pointwise'][1]
        assert hypothesis['metric'] == 'spearman'
    assert sl.verdict_counts(result['hypotheses']) == {
        'positive': 1, 'negative': 1, 'unresolved': 2}
    sensitivity = result['standardized_maxT']
    assert sensitivity['role'] == 'sensitivity' and 'sensitivity' in sensitivity['note']
    assert sensitivity['hypothesis_count'] == 4 and sensitivity['joint_critical_standardized'] > 0
    assert 'fixed SE' in sensitivity['method']
    with pytest.raises(ValueError, match='groups-by-labelled-columns'):
        sl.infer(matrix[:, :3], labels, metric='spearman')


def test_zero_shot_is_descriptive_and_flags_constant_scores():
    blocks, measured, aids, families, _, rng = fixture_panel()
    order = np.argsort(measured)
    ranked = np.empty(30)
    ranked[order] = np.arange(30.0)
    summary = sl.zero_shot({'profile': ranked}, measured, aids, families)
    assert summary['role'] == 'descriptive_zero_shot'
    assert summary['columns']['profile']['status'] == 'complete'
    assert summary['columns']['profile']['point'] == pytest.approx(1.0)
    assert summary['columns']['profile']['families'] == 10
    flat = sl.zero_shot({'profile': ranked, 'flat': np.ones(30)}, measured, aids, families)
    assert flat['status'] == 'partial'
    assert flat['columns']['flat']['status'] == 'not_estimable'
    assert len(flat['columns']['flat']['undefined_assays']) == 10
    with pytest.raises(ValueError, match='invalid raw'):
        sl.zero_shot({'profile': np.full(30, np.nan)}, measured, aids, families)


def potts_document(rows, **metadata):
    """The upstream Potts shape: a document listing per-sample records."""
    document = dict(schema='r1_msa_baseline_scores_v1',
                    inputs={'manifest': {'sha256': 'a' * 64}, 'support': {'sha256': 'b' * 64}},
                    counts={'samples': len(rows), 'samples_with_potts': len(rows)},
                    samples=rows)
    document.update(metadata)
    return document


def potts_rows(values, *, identifiers=None, status=None):
    identifiers = identifiers or [str(index) for index in range(len(values))]
    status = status or ['fitted'] * len(values)
    return [dict(sample_id=str(identifier), assay='a1', wildtype_id='wt',
                 mutation='A1V', n_substitutions=1, profile_delta=float(index),
                 potts_delta=value, status=state)
            for index, (identifier, value, state) in enumerate(zip(identifiers, values, status))]


def test_upstream_potts_document_is_read_and_joined_over_the_frozen_order(tmp_path):
    document = potts_document(potts_rows([3., 1., 2.]))
    path = tmp_path/'scores.json'
    path.write_text(json.dumps(document))
    table = sl.read_score_table(path, metadata_required=True)
    assert table['sample_id'] == ['0', '1', '2']
    assert set(table['metadata']) == {'schema', 'inputs', 'counts'}
    assert 'samples' not in table['metadata']
    assert set(table['metadata_sha256']) == {'schema', 'inputs', 'counts'}
    joined = sl.join_scores(['2', '0', '1'], table,
                            columns=(sl.POTTS_COLUMN, sl.PROFILE_COLUMN))
    np.testing.assert_array_equal(joined[sl.POTTS_COLUMN], [2., 3., 1.])
    np.testing.assert_array_equal(joined[sl.PROFILE_COLUMN], [2., 0., 1.])
    # `status` is textual and therefore never mistaken for a score column.
    assert 'status' not in table['columns']
    # A bare array dump declares no provenance and cannot pose as the table.
    arrays = tmp_path/'scores.npz'
    np.savez(arrays, sample_id=np.array(['0', '1', '2']), potts_delta=np.array([3., 1., 2.]))
    assert set(sl.read_score_table(arrays)['columns']) == {sl.POTTS_COLUMN}
    with pytest.raises(ValueError, match='no upstream provenance'):
        sl.read_score_table(arrays, metadata_required=True)


def test_json_tables_in_every_declared_container_shape(tmp_path):
    records = potts_rows([1., 2.])
    path = tmp_path/'list.json'
    path.write_text(json.dumps([dict(sample_id=r['sample_id'], potts_delta=r['potts_delta'])
                                for r in records]))
    table = sl.read_score_table(path)
    assert table['metadata'] == {}
    np.testing.assert_array_equal(sl.join_scores(['0', '1'], table,
                                                columns=(sl.POTTS_COLUMN,))[sl.POTTS_COLUMN], [1., 2.])
    keyed = tmp_path/'keyed.json'
    keyed.write_text(json.dumps({'0': {'gemme_score': 5.0}, '1': {'gemme_score': 6.0}}))
    table = sl.read_score_table(keyed)
    np.testing.assert_array_equal(sl.join_scores(['1', '0'], table,
                                                columns=(sl.GEMME_COLUMN,))[sl.GEMME_COLUMN], [6., 5.])
    flat = tmp_path/'flat.json'
    flat.write_text(json.dumps({'0': 5.0, '1': 6.0}))
    assert set(sl.read_score_table(flat)['columns']) == {'score'}
    with pytest.raises(ValueError, match='lacks required column'):
        sl.join_scores(['0', '1'], sl.read_score_table(flat), columns=(sl.GEMME_COLUMN,))
    unknown = tmp_path/'unknown.json'
    unknown.write_text(json.dumps({'meta': {'x': 1}, 'stuff': [1, 2]}))
    with pytest.raises(ValueError, match='unrecognised'):
        sl.read_score_table(unknown)


def test_score_join_refuses_duplicates_gaps_and_non_finite_values(tmp_path):
    path = tmp_path/'scores.json'
    path.write_text(json.dumps(potts_document(potts_rows([3., 1., 2.]))))
    table = sl.read_score_table(path)
    with pytest.raises(ValueError, match='missing'):
        sl.join_scores(['0', '1', '2', '3'], table, columns=(sl.POTTS_COLUMN,))
    with pytest.raises(ValueError, match='extra'):
        sl.join_scores(['0', '1'], table, columns=(sl.POTTS_COLUMN,))
    duplicate = dict(table, sample_id=['0', '0', '2'])
    with pytest.raises(ValueError, match='duplicate'):
        sl.join_scores(['0', '2'], duplicate, columns=(sl.POTTS_COLUMN,))
    with pytest.raises(ValueError, match='duplicate'):
        sl.join_scores(['0', '0'], table, columns=(sl.POTTS_COLUMN,))
    short = dict(table, columns=dict(table['columns'], potts_delta=np.array([1., 2.])))
    with pytest.raises(ValueError, match='not aligned'):
        sl.join_scores(['0', '1', '2'], short, columns=(sl.POTTS_COLUMN,))
    nonnumeric = dict(table, columns=dict(table['columns'], potts_delta=np.array(['x', 'y', 'z'])))
    with pytest.raises(ValueError, match='non-numeric'):
        sl.join_scores(['0', '1', '2'], nonnumeric, columns=(sl.POTTS_COLUMN,))


def test_unresolved_upstream_backgrounds_fail_with_the_recorded_status(tmp_path):
    rows = potts_rows([1., None, None], status=['fitted', 'pending', 'failed'])
    path = tmp_path/'scores.json'
    path.write_text(json.dumps(potts_document(rows)))
    table = sl.read_score_table(path)
    with pytest.raises(ValueError, match='2 non-finite scores'):
        sl.join_scores(['0', '1', '2'], table, columns=(sl.POTTS_COLUMN,))
    with pytest.raises(ValueError, match='pending'):
        sl.join_scores(['0', '1', '2'], table, columns=(sl.POTTS_COLUMN,))


def test_malformed_score_documents_are_rejected(tmp_path):
    path = tmp_path/'scores.json'
    path.write_text('{not a table}')
    with pytest.raises(ValueError, match='not valid JSON'):
        sl.read_score_table(path)
    path.write_text(json.dumps([dict(sample_id='a', potts_delta=1.0),
                                dict(sample_id='b')]))
    with pytest.raises(ValueError, match='only some rows carry'):
        sl.read_score_table(path)
    path.write_text(json.dumps([dict(potts_delta=1.0)]))
    with pytest.raises(ValueError, match='non-empty sample_id'):
        sl.read_score_table(path)
    with pytest.raises(ValueError, match='missing'):
        sl.read_score_table(tmp_path/'absent.json')


SUPPORT = 25728


def complete_counts(name, **overrides):
    counts = dict(sl.BASELINE_CONTRACT[name]['counts'])
    counts.update(overrides)
    return counts


def identifiers(rows=SUPPORT):
    return [f'{index:064x}' for index in range(rows)]


def potts_table(*, status='fitted', rows=SUPPORT, metadata=None):
    return dict(path='potts/scores.json', sample_id=identifiers(rows),
                columns={'potts_delta': np.arange(rows, dtype=float),
                         'profile_delta': -np.arange(rows, dtype=float)},
                annotations={'status': np.asarray([status] * rows)},
                metadata=dict(schema='r1_msa_baseline_scores_v1',
                              counts=complete_counts('potts')) if metadata is None else metadata)


def gemme_table(*, status='scored', n_runs=3, repeats=3, incomplete=(), counts=None,
                schema='r1_gemme_scores'):
    return dict(path='gemme/scores.json', sample_id=identifiers(),
                columns={'gemme_score': np.zeros(SUPPORT), 'n_runs': np.full(SUPPORT, n_runs)},
                annotations={'status': np.asarray([status] * SUPPORT)},
                metadata=dict(schema=schema, repeats=repeats,
                              incomplete=list(incomplete),
                              counts=complete_counts('gemme') if counts is None else counts))


def test_the_upstream_schema_is_provenance_and_both_recorded_ids_are_admitted():
    """The ladder's own new table carries the stable ID; the recorded one still reads.

    A score document's schema is provenance bound by digest, not a shape gate, so
    nothing here strips a suffix: both known labels are read and an unknown one is
    simply carried as provenance like any other metadata value.
    """
    for schema in ('r1_gemme_scores', 'r1_gemme_scores_v1'):
        gemme = sl.require_complete_baseline('gemme', gemme_table(schema=schema))
        assert gemme['repeats'] == 3 and gemme['rows'] == SUPPORT


def test_complete_baselines_are_counted_and_summarised():
    potts = sl.require_complete_baseline('potts', potts_table())
    assert potts['rows'] == SUPPORT and potts['row_status'] == 'fitted'
    assert potts['counts']['backgrounds_fitted'] == 174 and potts['repeats'] is None
    gemme = sl.require_complete_baseline('gemme', gemme_table())
    assert gemme['repeats'] == 3 and gemme['counts']['wildtypes_complete'] == 174
    with pytest.raises(ValueError, match='unknown baseline'):
        sl.require_complete_baseline('other', potts_table())


def test_finite_but_unconverged_potts_rows_are_refused():
    # The admission gap: 128 rows of the real table carry a finite energy
    # difference from a background that did not converge.
    table = potts_table()
    table['annotations']['status'] = np.asarray(
        ['fitted_not_converged'] * 128 + ['fitted'] * (SUPPORT - 128))
    assert np.isfinite(table['columns']['potts_delta']).all()
    with pytest.raises(ValueError, match='128 rows are not fitted'):
        sl.require_complete_baseline('potts', table)
    with pytest.raises(ValueError, match='not_fitted'):
        sl.require_complete_baseline('potts', potts_table(status='not_fitted'))


def test_potts_counts_must_declare_a_complete_finished_contract():
    for overrides, message in (
            ({'complete': False}, 'counts.complete'),
            ({'backgrounds_pending': 172, 'backgrounds_resolved': 2}, 'disagree'),
            ({'backgrounds_not_converged': 1}, 'disagree'),
            ({'backgrounds_failed': 1}, 'disagree'),
            ({'backgrounds_fitted': 173}, 'disagree'),
            ({'samples_with_potts': 256}, 'disagree')):
        counts = complete_counts('potts', **overrides)
        with pytest.raises(ValueError, match=message):
            sl.require_complete_baseline('potts', potts_table(metadata=dict(counts=counts)))
    short = potts_table(rows=SUPPORT - 1)
    with pytest.raises(ValueError, match='disagree'):
        sl.require_complete_baseline('potts', short)


def test_missing_completion_information_is_refused():
    with pytest.raises(ValueError, match='declares no counts'):
        sl.require_complete_baseline('potts', potts_table(metadata={}))
    with pytest.raises(ValueError, match='counts.complete is None'):
        sl.require_complete_baseline('potts', potts_table(metadata={'counts': {'samples': SUPPORT}}))
    table = potts_table()
    del table['annotations']['status']
    with pytest.raises(ValueError, match='no status'):
        sl.require_complete_baseline('potts', table)


def test_gemme_requires_three_runs_and_an_empty_incomplete_list():
    with pytest.raises(ValueError, match='exactly 3 official runs'):
        sl.require_complete_baseline('gemme', gemme_table(n_runs=2))
    with pytest.raises(ValueError, match='repeats'):
        sl.require_complete_baseline('gemme', gemme_table(repeats=1))
    with pytest.raises(ValueError, match='incomplete'):
        sl.require_complete_baseline('gemme', gemme_table(incomplete=[{'wildtype_id': 'q00001'}]))
    with pytest.raises(ValueError, match='rows are not scored'):
        sl.require_complete_baseline('gemme', gemme_table(status='wildtype_pending'))
    with pytest.raises(ValueError, match='counts.complete'):
        sl.require_complete_baseline(
            'gemme', gemme_table(counts=complete_counts('gemme', complete=False)))
    with pytest.raises(ValueError, match='disagree'):
        sl.require_complete_baseline(
            'gemme', gemme_table(counts=complete_counts('gemme', samples_scored=256)))
    table = gemme_table()
    del table['columns']['n_runs']
    with pytest.raises(ValueError, match='no n_runs'):
        sl.require_complete_baseline('gemme', table)


def test_a_shallow_msa_zero_score_stays_a_legitimate_scored_baseline():
    # An all-zero official score is legal; it is a constant predictor, not a
    # failure, so admission keeps it and only the descriptive layer says so.
    table = gemme_table()
    assert not table['columns']['gemme_score'].any()
    sl.require_complete_baseline('gemme', table)
    joined = sl.join_scores(list(table['sample_id']), table, columns=(sl.GEMME_COLUMN,))
    np.testing.assert_array_equal(joined[sl.GEMME_COLUMN], np.zeros(SUPPORT))
    rows = 30
    aids = np.repeat(np.asarray(list('abcdefghij'), object), 3)
    measured = np.arange(rows, dtype=float)
    summary = sl.zero_shot({'gemme': np.zeros(rows), 'profile': measured}, measured,
                           aids, np.repeat(np.asarray(list('abcdefghij'), object), 3))
    assert summary['columns']['gemme']['status'] == 'not_estimable'
    assert summary['columns']['profile']['point'] == pytest.approx(1.0)


def write_baseline(directory, name, records, counts, *, extra=None, arrays=None):
    folder = directory/name
    folder.mkdir(parents=True, exist_ok=True)
    document = dict(schema='synthetic_baseline_v1', counts=counts, samples=records)
    document.update(extra or {})
    (folder/'scores.json').write_text(json.dumps(document))
    if arrays is not None:
        np.savez(folder/'scores.npz', **arrays)
    return folder


def potts_records(status='fitted'):
    return [dict(sample_id=value, potts_delta=float(index), profile_delta=-float(index),
                 status=status) for index, value in enumerate(identifiers())]


def gemme_records(status='scored', n_runs=3):
    return [dict(sample_id=value, gemme_score=0.0, n_runs=n_runs, status=status)
            for value in identifiers()]


def test_full_stage_admission_requires_a_complete_and_bound_baseline(tmp_path):
    from scripts.capability.extensions.fit_statistical_ladder import read_baseline_table

    write_baseline(tmp_path, 'potts', potts_records(), complete_counts('potts'),
                   arrays=dict(sample_id=np.asarray(identifiers()),
                               potts_delta=np.arange(SUPPORT, dtype=float)))
    write_baseline(tmp_path, 'gemme', gemme_records(), complete_counts('gemme'),
                   extra=dict(repeats=3, incomplete=[]))
    potts = read_baseline_table(tmp_path, 'potts', sl.POTTS_COLUMN)
    assert potts['completeness']['row_status'] == 'fitted'
    assert potts['arrays_sha256'] and potts['completeness']['counts']['backgrounds_fitted'] == 174
    gemme = read_baseline_table(tmp_path, 'gemme', sl.GEMME_COLUMN)
    assert gemme['completeness']['repeats'] == 3
    # The array dump must still be the document's own numbers.
    np.savez(tmp_path/'potts'/'scores.npz', sample_id=np.asarray(identifiers()),
             potts_delta=np.arange(SUPPORT, dtype=float) + 1.0)
    with pytest.raises(ValueError, match='disagrees'):
        read_baseline_table(tmp_path, 'potts', sl.POTTS_COLUMN)
    # An incomplete document is refused even though every score is finite.
    write_baseline(tmp_path, 'potts', potts_records('fitted_not_converged'),
                   complete_counts('potts', **{'complete': False,
                                               'backgrounds_not_converged': 1,
                                               'backgrounds_fitted': 173,
                                               'backgrounds_pending': 0,
                                               'backgrounds_resolved': 174}))
    with pytest.raises(ValueError, match='counts.complete'):
        read_baseline_table(tmp_path, 'potts', sl.POTTS_COLUMN)
    # Even a document whose counts claim completion is refused when a row is not
    # the finished status: the two declarations must agree.
    write_baseline(tmp_path, 'potts', potts_records('fitted_not_converged'),
                   complete_counts('potts'))
    with pytest.raises(ValueError, match='rows are not fitted'):
        read_baseline_table(tmp_path, 'potts', sl.POTTS_COLUMN)
    (tmp_path/'gemme'/'scores.json').unlink()
    with pytest.raises(ValueError, match='is missing'):
        read_baseline_table(tmp_path, 'gemme', sl.GEMME_COLUMN)


def saved_cell(tmp_path, designs, *, samples, target):
    identifiers = np.asarray([row['sample_id'] for row in samples])
    rng = np.random.default_rng(11)
    predictions = {name: rng.normal(size=len(target)) for name in designs}
    folds = {name: [dict(fold=index, held_families=['f'], training_families=['g'],
                         alpha=1.0, inner_family_rank_mse=[0.0],
                         inner_folds=[dict(validation_families=['g'], rank_mse=[0.0])])
                    for index in range(1)] for name in designs}
    membership = [(0, ('f',), ('g',), (('g',),))]
    path = tmp_path/'cell.npz'
    np.savez_compressed(path, **predictions, original_index=np.arange(len(target)),
                       target=np.asarray(target), sample_id=identifiers)
    checks = dict(designs={name: dict(design=list(sl.DESIGNS[name]), design_sha256='0' * 64,
                                     target_sha256=sl.array_digest(target), support_sha256='s',
                                     membership_sha256=sl.identity_digest(membership),
                                     rows=len(target),
                                     prediction_sha256=sl.array_digest(predictions[name]))
                           for name in designs})
    return path, predictions, checks, folds, membership, identifiers


def test_archive_identity_and_digest_verification(tmp_path):
    samples = [dict(sample_id='a'), dict(sample_id='b'), dict(sample_id='c')]
    target = np.array([0.1, 0.2, 0.3])
    path, predictions, checks, folds, membership, identifiers = saved_cell(
        tmp_path, ('M', 'MP'), samples=samples, target=target)
    stored = sl.read_predictions(path, ('M', 'MP'), sample_ids=identifiers, target=target)
    assert set(sl.read_predictions(path, ('M',), sample_ids=identifiers, target=target)) == {'M'}
    summary = sl.verify_checks(checks, stored, designs=('M', 'MP'), support_sha256='s',
                              membership=membership, target=target, fold_records=folds)
    assert summary['rows'] == 3 and summary['designs'] == ['M', 'MP']
    with pytest.raises(ValueError, match='missing'):
        stranger = tmp_path/'stranger.npz'
        np.savez_compressed(stranger, BM=np.zeros(3), original_index=np.arange(3),
                            target=target, sample_id=identifiers)
        sl.read_predictions(stranger, ('M',), sample_ids=identifiers, target=target)
    with pytest.raises(ValueError, match='missing'):
        sl.read_predictions(path, ('M', 'BMPL'), sample_ids=identifiers, target=target)
    with pytest.raises(ValueError, match='target'):
        sl.read_predictions(path, ('M', 'MP'), sample_ids=identifiers, target=target + 1)
    with pytest.raises(ValueError, match='sample identity'):
        sl.read_predictions(path, ('M', 'MP'), sample_ids=identifiers[::-1], target=target)
    tampered = dict(checks, designs=dict(checks['designs']))
    tampered['designs']['M'] = dict(tampered['designs']['M'], prediction_sha256='1' * 64)
    with pytest.raises(ValueError, match='prediction digest'):
        sl.verify_checks(tampered, stored, designs=('M', 'MP'), support_sha256='s',
                         membership=membership, target=target, fold_records=folds)
    with pytest.raises(ValueError, match='support digest'):
        sl.verify_checks(checks, stored, designs=('M', 'MP'), support_sha256='other',
                         membership=membership, target=target, fold_records=folds)
    with pytest.raises(ValueError, match='folds disagree'):
        sl.verify_checks(checks, stored, designs=('M',), support_sha256='s',
                         membership=membership, target=target,
                         fold_records={'M': [dict(folds['M'][0], held_families=['bogus'])]})
    with pytest.raises(ValueError, match='design blocks'):
        bad = dict(checks, designs=dict(checks['designs'], M=dict(checks['designs']['M'],
                                                                  design=['P'])))
        sl.verify_checks(bad, stored, designs=('M',), support_sha256='s',
                         membership=membership, target=target, fold_records=folds)


def pilot_record(**overrides) -> dict:
    cell = {name: dict(max_absolute_difference=0.0, prediction_identical=True,
                       design_digest_matched=True, alpha_reproduced=True,
                       fold_membership_matched=True) for name in sl.REPRODUCTION_DESIGNS}
    record: dict = dict(schema=sl.PILOT_SCHEMA, status='passed', threads=2,
                        source_contract_sha256='c' * 64, source_completion_sha256='d' * 64,
                        script_sha256='e' * 64, module_sha256='f' * 64,
                        designs=list(sl.REPRODUCTION_DESIGNS), models=['arm'],
                        seeds=list(SEEDS), reproduction={'arm': {str(seed): cell for seed in SEEDS}})
    record.update(overrides)
    return record


def test_reuse_pilot_gate_requires_exact_reproduction_for_every_model():
    gate = sl.validate_pilot(pilot_record(), source_contract_sha256='c' * 64,
                             source_completion_sha256='d' * 64,
                             current_script_sha256='e' * 64, current_module_sha256='f' * 64,
                             models=['arm'], threads=2)
    assert gate == dict(models=1, designs=list(sl.REPRODUCTION_DESIGNS), seeds=3, threads=2)
    cases = [
        (pilot_record(status='partial'), 'not passed'),
        (pilot_record(schema='other'), 'schema'),
        (pilot_record(source_contract_sha256='9' * 64), 'source contract'),
        (pilot_record(threads=1), 'thread count'),
        (pilot_record(designs=['M']), 'designs being reused'),
    ]
    for record, message in cases:
        with pytest.raises(ValueError, match=message):
            sl.validate_pilot(record, source_contract_sha256='c' * 64,
                              source_completion_sha256='d' * 64,
                              current_script_sha256='e' * 64, current_module_sha256='f' * 64,
                              models=['arm'], threads=2)
    with pytest.raises(ValueError, match='other'):
        sl.validate_pilot(pilot_record(), source_contract_sha256='c' * 64,
                          source_completion_sha256='d' * 64,
                          current_script_sha256='e' * 64, current_module_sha256='f' * 64,
                          models=['arm', 'other'], threads=2)
    broken = pilot_record()
    broken['reproduction']['arm'][str(SEEDS[1])]['MP']['max_absolute_difference'] = 1e-9
    with pytest.raises(ValueError, match='differs above zero'):
        sl.validate_pilot(broken, source_contract_sha256='c' * 64,
                          source_completion_sha256='d' * 64,
                          current_script_sha256='e' * 64, current_module_sha256='f' * 64,
                          models=['arm'], threads=2)
    missing_seed = pilot_record()
    del missing_seed['reproduction']['arm'][str(SEEDS[2])]
    with pytest.raises(ValueError, match='three seeds'):
        sl.validate_pilot(missing_seed, source_contract_sha256='c' * 64,
                          source_completion_sha256='d' * 64,
                          current_script_sha256='e' * 64, current_module_sha256='f' * 64,
                          models=['arm'], threads=2)


def test_reuse_pilot_schema_is_the_stable_id_and_the_recorded_spelling_still_reads():
    assert sl.PILOT_SCHEMA == 'statistical_ladder_reuse_pilot'
    assert set(sl.PILOT_SCHEMAS) == {'statistical_ladder_reuse_pilot',
                                     'statistical_ladder_reuse_pilot_v1'}
    for schema in sl.PILOT_SCHEMAS:
        gate = sl.validate_pilot(pilot_record(schema=schema), source_contract_sha256='c' * 64,
                                 source_completion_sha256='d' * 64,
                                 current_script_sha256='e' * 64, current_module_sha256='f' * 64,
                                 models=['arm'], threads=2)
        assert gate == dict(models=1, designs=list(sl.REPRODUCTION_DESIGNS), seeds=3, threads=2)
    # Exact identities only: an unknown label is refused, never stripped into a known one.
    for schema in ('statistical_ladder_reuse_pilot_v999', 'statistical_ladder_reuse_pilot_v2',
                   'Statistical_Ladder_Reuse_Pilot_v1'):
        with pytest.raises(ValueError, match='schema'):
            sl.validate_pilot(pilot_record(schema=schema), source_contract_sha256='c' * 64,
                              source_completion_sha256='d' * 64,
                              current_script_sha256='e' * 64, current_module_sha256='f' * 64,
                              models=['arm'], threads=2)
    # Admitting the recorded spelling relaxes nothing else in the gate.
    unverified = pilot_record(schema='statistical_ladder_reuse_pilot_v1')
    unverified['reproduction']['arm'][str(SEEDS[0])]['M']['prediction_identical'] = False
    with pytest.raises(ValueError, match='did not reproduce'):
        sl.validate_pilot(unverified, source_contract_sha256='c' * 64,
                          source_completion_sha256='d' * 64,
                          current_script_sha256='e' * 64, current_module_sha256='f' * 64,
                          models=['arm'], threads=2)
    unbound = pilot_record(schema='statistical_ladder_reuse_pilot_v1', script_sha256='not-a-digest')
    with pytest.raises(ValueError, match='recorded digest'):
        sl.validate_pilot(unbound, source_contract_sha256='c' * 64,
                          source_completion_sha256='d' * 64,
                          current_script_sha256='e' * 64, current_module_sha256='f' * 64,
                          models=['arm'], threads=2)


@pytest.mark.parametrize('schema', sl.PILOT_SCHEMAS)
@pytest.mark.parametrize('field', ['script_sha256', 'module_sha256'])
def test_reuse_pilot_refuses_different_valid_code_digests(schema, field):
    record = pilot_record(schema=schema, **{field: '0' * 64})
    with pytest.raises(ValueError, match=f'{field} does not match the current code'):
        sl.validate_pilot(record, source_contract_sha256='c' * 64,
                          source_completion_sha256='d' * 64,
                          current_script_sha256='e' * 64, current_module_sha256='f' * 64,
                          models=['arm'], threads=2)


def test_runner_refuses_unspecified_stages_and_prior_outputs(tmp_path):
    from scripts.capability.extensions.fit_statistical_ladder import run
    evidence = {'status': 'passed'}
    with pytest.raises(ValueError, match='tests required'):
        run(tmp_path, tmp_path/'out', stage='profile_only')
    with pytest.raises(ValueError, match='one or two CPU threads'):
        run(tmp_path, tmp_path/'out', stage='profile_only', threads=8, test_evidence=evidence)
    with pytest.raises(ValueError, match='complete Potts and GEMME'):
        run(tmp_path, tmp_path/'out', stage='full', test_evidence=evidence)
    with pytest.raises(ValueError, match='never reads Potts/GEMME'):
        run(tmp_path, tmp_path/'out', stage='profile_only', scores=tmp_path/'s.npz',
            test_evidence=evidence)
    with pytest.raises(ValueError, match='unknown stage'):
        run(tmp_path, tmp_path/'out', stage='partial', test_evidence=evidence)
    out = tmp_path/'out'
    out.mkdir()
    with pytest.raises(ValueError, match='never overwrite'):
        run(tmp_path, out, stage='profile_only', test_evidence=evidence)


def source_stub(tmp_path, models=('armA', 'armB')):
    directory = tmp_path/'source'
    directory.mkdir()
    (directory/'contract.json').write_text(json.dumps(dict(models=list(models), seeds=list(SEEDS))))
    (directory/'completion.json').write_text(json.dumps(dict(status='complete')))
    return directory


def test_declared_scope_gates_the_full_stage_before_any_artifact_is_read(tmp_path):
    from scripts.capability.extensions.fit_statistical_ladder import declare_scope

    source = source_stub(tmp_path)
    columns = dict(potts_column=sl.POTTS_COLUMN, gemme_column=sl.GEMME_COLUMN)
    with pytest.raises(ValueError, match='complete Potts and GEMME'):
        declare_scope(source, stage='full', scores=None, pilot=None, models=None,
                      seeds=None, threads=2, **columns)
    with pytest.raises(ValueError, match='never reads Potts/GEMME'):
        declare_scope(source, stage='profile_only', scores=tmp_path, pilot=None, models=None,
                      seeds=None, threads=2, **columns)
    with pytest.raises(ValueError, match='reuse-pilot'):
        declare_scope(source, stage='full', scores=tmp_path, pilot=None, models=None,
                      seeds=None, threads=2, **columns)
    with pytest.raises(ValueError, match='all 33 models'):
        declare_scope(source, stage='full', scores=tmp_path, pilot=tmp_path/'p.json', models=None,
                      seeds=None, threads=2, **columns)
    models, seeds, preflight = declare_scope(source, stage='profile_only', scores=None, pilot=None,
                                             models=['armB'], seeds=SEEDS[:2], threads=2, **columns)
    assert models == ['armB'] and seeds == SEEDS[:2] and preflight == dict(tables={}, pilot=None)
    with pytest.raises(ValueError, match='outside the frozen roster'):
        declare_scope(source, stage='profile_only', scores=None, pilot=None, models=['other'],
                      seeds=None, threads=2, **columns)
    with pytest.raises(ValueError, match='original three'):
        declare_scope(source, stage='profile_only', scores=None, pilot=None, models=None,
                      seeds=[19990101], threads=2, **columns)


@pytest.mark.parametrize('schema', sl.PILOT_SCHEMAS)
def test_declared_scope_requires_both_tables_and_a_matching_pilot(tmp_path, schema):
    from scripts.capability.extensions import fit_statistical_ladder as runner

    declare_scope = runner.declare_scope

    models = [f'arm{index:02d}' for index in range(33)]
    source = source_stub(tmp_path, models)
    scores = tmp_path/'scores'
    columns = dict(potts_column=sl.POTTS_COLUMN, gemme_column=sl.GEMME_COLUMN)
    with pytest.raises(ValueError, match='potts/scores.json is missing'):
        declare_scope(source, stage='full', scores=scores, pilot=tmp_path/'p.json', models=None,
                      seeds=None, threads=2, **columns)
    write_baseline(scores, 'potts', potts_records(), complete_counts('potts'))
    write_baseline(scores, 'gemme', gemme_records(), complete_counts('gemme'),
                   extra=dict(repeats=3, incomplete=[]))
    contract_sha, completion_sha = sl.digest(source/'contract.json'), sl.digest(source/'completion.json')
    # Independent byte hashes: the real runner must bind both current files,
    # not merely accept any well-formed digest from a passing receipt.
    record = pilot_record(schema=schema, source_contract_sha256=contract_sha,
                          source_completion_sha256=completion_sha,
                          script_sha256=hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest(),
                          module_sha256=hashlib.sha256(Path(sl.__file__).read_bytes()).hexdigest())
    record['reproduction'] = {model: record['reproduction']['arm'] for model in models}
    pilot = tmp_path/'p.json'
    pilot.write_text(json.dumps(record))
    resolved, seeds, preflight = declare_scope(source, stage='full', scores=scores, pilot=pilot,
                                              models=None, seeds=None, threads=2, **columns)
    assert resolved == models and seeds == SEEDS
    assert preflight['pilot']['path'] == str(pilot)
    assert preflight['pilot']['gate']['models'] == 33
    assert set(preflight['tables']) == {'potts', 'gemme'}
    for field in ('script_sha256', 'module_sha256'):
        stale = dict(record, **{field: '0' * 64})
        pilot.write_text(json.dumps(stale))
        with pytest.raises(ValueError, match=f'{field} does not match the current code'):
            declare_scope(source, stage='full', scores=scores, pilot=pilot, models=None,
                          seeds=None, threads=2, **columns)
    # The baseline contract wins over the reuse gate: what the operator cannot
    # repair by re-running a pilot is reported first.
    (scores/'potts'/'scores.json').unlink()
    with pytest.raises(ValueError, match='is missing'):
        declare_scope(source, stage='full', scores=scores, pilot=pilot, models=None,
                      seeds=None, threads=2, **columns)
