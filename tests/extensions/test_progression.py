"""Frozen extension registry, paired inference and real nested CPU fits."""
import json
from typing import Any

import numpy as np
import pytest
import torch

from src.capability.extensions import overlap as o
from src.capability.extensions import progression as p
from src.capability.context.local_context import fold_membership
from src.capability.readouts.readout_analysis import family_folds, nested_predict


def signature(groups, seed=p.SEEDS[0]):
    result = []
    for fold, held in enumerate(family_folds(groups, 5, seed)):
        train = sorted(set(groups)-set(held))
        inner = family_folds(train, 4, seed+100+fold)
        result.append((fold, tuple(sorted(held)), tuple(train), tuple(tuple(sorted(v)) for v in inner)))
    return result


@pytest.mark.parametrize('panel,names,nprimary,nsupp', [
    ('full', 'M MP MPL BM BMP BMPL BPL BML', 2, 5),
    ('structural', 'M MP MPL MPLS BM BMP BMPL BMPLS BPLS BMLS BMPS', 3, 7)])
def test_registry_frozen_union_and_signs(panel, names, nprimary, nsupp):
    assert list(p.DESIGNS[panel]) == names.split()
    registry = p.registry(panel)
    assert sum(r['family']=='primary' for r in registry['contrasts'].values()) == nprimary
    assert sum(r['family']=='supplementary' for r in registry['contrasts'].values()) == nsupp
    pairs = []
    for contrast, entry in registry['contrasts'].items():
        high, low = entry['augmented'], entry['reduced']
        assert set(p.DESIGNS[panel][low]) < set(p.DESIGNS[panel][high])
        pairs.append((high, low))
        scores: dict[str, Any] = dict.fromkeys(p.DESIGNS[panel], 0.)
        scores[high] = 1.
        assert p.contrast_values(scores, 'spearman', panel)[contrast] == 1.
        assert p.contrast_values(scores, 'rank_mse', panel)[contrast] == -1.
        scores[high] = None
        assert p.contrast_values(scores, 'spearman', panel)[contrast] is None
    assert len(set(pairs)) == len(pairs)  # identical adjusted ladder/drop pairs shared
    assert p.DESIGNS[panel]['BM'] == ('B', 'M')
    with pytest.raises(ValueError, match='unknown metric'):
        p.contrast_values({}, 'bogus', panel)


def test_column_inventory_and_exact_label_blind_census():
    inventory = p.column_inventory()
    assert inventory['B']['mutation_count'] == 402
    assert inventory['B']['WT_composition'] == [403, 423]
    assert len(inventory['P']) == 14 and len(inventory['L']) == 111 and len(inventory['S']) == 7
    x = np.array([[0, 2, 1, 1, -0.], [0, 2, 3, 3, 0.]])
    audit = p.census(x)
    assert audit['zero_columns'] == [0, 4]
    assert audit['constant_columns'] == [0, 1, 4]
    assert audit['exact_duplicate_columns'] == [[2, 3], [0, 4]]
    with pytest.raises(ValueError):
        p.census(np.array([[np.nan]]))


def fixture_panel():
    rng = np.random.default_rng(9)
    families = np.repeat(list('abcdefghij'), 3)
    aids = families.copy()
    blocks = {b: rng.normal(size=(30, width)) for b, width in p.WIDTHS.items()}
    measured = rng.normal(size=30)
    samples = [dict(assay=str(a), cluster=str(f), mutation=f'A1{j}', variant_index=j%3,
                    original_index=j, sample_id=f'{a}-{j}') for j, (a, f) in enumerate(zip(aids, families))]
    return blocks, measured, aids, families, samples


def test_subset_reranking_changes_only_M_and_P0():
    blocks, _, aids, _, _ = fixture_panel()
    # Explicit ties and a noncontiguous subset demonstrate reranking in retained support.
    blocks['M'][:3, 0] = [-1, 0, 0]
    blocks['P'][:3, 0] = [-9, -8, -8]
    result = p.prepare_blocks(blocks, aids, 'full')
    np.testing.assert_array_equal(result['M'][:, 0], o.rerank(blocks['M'][:, 0], aids))
    np.testing.assert_array_equal(result['P'][:, 0], o.rerank(blocks['P'][:, 0], aids))
    np.testing.assert_array_equal(result['P'][:, 1:], blocks['P'][:, 1:])
    assert result['M'][1, 0] == result['M'][2, 0]
    index = np.array([0, 2, 3, 5])
    subset = p.prepare_blocks({b: x[index] for b, x in blocks.items()}, aids[index], 'full')
    assert subset['M'][0, 0] != result['M'][0, 0]
    np.testing.assert_array_equal(blocks['M'][:3, 0], [-1, 0, 0])
    with pytest.raises(ValueError, match='invalid B'):
        p.prepare_blocks({**blocks, 'B': blocks['B'][:, :-1]}, aids, 'full')


def test_projection_preserves_memberships_and_explicitly_refuses_unsupported():
    groups = list('abcdefghijklmnopqrst'); sig = signature(groups)
    projected = o.project_folds(sig, groups)
    assert [(f, tuple(h), tuple(t), tuple(tuple(v) for v in inn)) for f, h, t, inn in projected] == sig
    subset = [held[0] for _, held, _, _ in sig]
    with pytest.raises(o.UnsupportedProjection):
        o.project_folds(sig, subset)
    with pytest.raises(ValueError, match='five'):
        o.project_folds(sig[:-1], groups)


def test_duplicate_negative_nonfinite_or_cross_family_inputs_refused():
    blocks, y, aids, families, samples = fixture_panel()
    p.validate_samples(samples, y, aids, families)
    for replacement in (samples[0], dict(samples[1], original_index=-1),
                        dict(samples[1], variant_index=-1), dict(samples[1], cluster='wrong')):
        bad = list(samples); bad[1] = replacement
        with pytest.raises(ValueError):
            p.validate_samples(bad, y, aids, families)
    with pytest.raises(ValueError):
        p.validate_samples(samples, np.full(30, np.nan), aids, families)
    bad = {**blocks, 'M': np.full((30, 1), np.inf)}
    with pytest.raises(ValueError):
        p.prepare_blocks(bad, aids, 'full')
    cross = families.copy(); cross[1] = 'b'
    bad_samples = [dict(s, cluster=str(f)) for s, f in zip(samples, cross)]
    with pytest.raises(ValueError, match='multiple families'):
        p.validate_samples(bad_samples, y, aids, cross)


def inference_rows():
    # Family f has two assays and g one: seed/assay/family weighting differs
    # from treating either seed rows or assay rows as independent units.
    result = []
    for arm in ('a', 'b'):
        for assay, family, value in (('x', 'f', 0.), ('y', 'f', 2.), ('z', 'g', 5.)):
            for seed, delta in zip(p.SEEDS, (-1., 0., 1.)):
                result.append(dict(arm=arm, assay=assay, cluster=family, seed=seed,
                    spearman_contrasts=dict.fromkeys(p.CONTRASTS['full'], value+delta),
                    rank_mse_contrasts=dict.fromkeys(p.CONTRASTS['full'], value+delta)))
    return result


def test_seed_assay_family_equal_weight_and_shared_joint_draws():
    rows = inference_rows()
    summary = p.joint_bootstrap(rows, 'full', bootstrap=100, expected_arms=['a', 'b'])
    assert summary['status'] == 'complete'
    groups = summary['groups']
    assert [g['hypothesis_count'] for g in groups.values()] == [4, 10, 14]
    for g in groups.values():
        assert all(c['point'] == 3. for c in g['contrasts'])  # (mean(0,2)+5)/2
        assert all(c['pointwise']==g['contrasts'][0]['pointwise'] for c in g['contrasts'])
        assert all(c['simultaneous']==g['contrasts'][0]['simultaneous'] for c in g['contrasts'])
    assert len({g['joint_critical_max_absolute_deviation'] for g in groups.values()}) == 1
    assert summary == p.joint_bootstrap(rows, 'full', bootstrap=100, expected_arms=['a', 'b'])


def test_inference_missing_duplicate_and_undefined_are_not_dropped():
    rows = inference_rows()
    for bad in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError):
            p.joint_bootstrap(bad, 'full', bootstrap=100)
    rows[0]['spearman_contrasts']['P|M'] = None
    summary = p.joint_bootstrap(rows, 'full', bootstrap=100)
    assert summary['status'] == 'not_estimable'
    assert summary['groups']['primary_spearman']['status'] == 'not_estimable'
    assert summary['groups']['supplementary_spearman']['status'] == 'complete'
    assert summary['groups']['secondary_rank_mse']['status'] == 'complete'
    with pytest.raises(ValueError, match='100'):
        p.joint_bootstrap(rows, 'full', bootstrap=-1)


def test_real_small_nested_fit_all_designs_shared_membership_and_persistence(tmp_path):
    from scripts.capability.extensions.fit_information_progression import verify_cell
    torch.set_num_threads(2)
    blocks, y, aids, families, samples = fixture_panel(); sig = signature(families)
    pred, records, checks = p.fit_cell(blocks, y, aids, families, sig, 'structural', samples)
    assert set(pred) == set(p.DESIGNS['structural'])
    assert all(fold_membership(r)==sig for r in records.values())
    m = o.rerank(blocks['M'][:, 0], aids)[:, None]
    expected, _ = nested_predict(m, y, aids, families, device='cpu')
    np.testing.assert_allclose(pred['M'], expected, atol=1e-12)
    assert all(c['support_sha256']==p.identity_digest(samples) for c in checks.values())
    path = tmp_path/'oof.npz'
    data: dict[str, Any] = dict(index=np.arange(30), samples=samples, y=y, aids=aids)
    np.savez_compressed(path, **pred, original_index=data['index'], target=o.rerank(y, aids),
                        sample_id=np.asarray([s['sample_id'] for s in samples]))
    assert verify_cell(path, pred, records, checks, data, o.project_folds(sig, families))['identity_verified']
    bad_records = json.loads(json.dumps(records)); bad_records['M'][0]['held_families'] = ['bogus']
    with pytest.raises(ValueError, match='fold'):
        verify_cell(path, pred, bad_records, checks, data, o.project_folds(sig, families))
    np.savez_compressed(path, **pred, original_index=data['index'], target=np.zeros(30),
                        sample_id=np.asarray([s['sample_id'] for s in samples]))
    with pytest.raises(ValueError, match='target'):
        verify_cell(path, pred, records, checks, data, o.project_folds(sig, families))
    rows = p.assay_metrics(pred, y, aids, families, 'arm', p.SEEDS[0], 'structural')
    assert len(rows) == 10
    constant = {name: np.ones(30) for name in pred}
    assert all(r['spearman_contrasts']['P|M'] is None for r in
               p.assay_metrics(constant, y, aids, families, 'arm', p.SEEDS[0], 'structural'))


def test_cli_refuses_prior_outputs_and_unverified_fits(tmp_path):
    from scripts.capability.extensions.fit_information_progression import run, save, archive_path
    assert archive_path(tmp_path/'qwen2.5-7b-seed20260923').name == 'qwen2.5-7b-seed20260923.npz'
    assert archive_path(tmp_path/'qwen2.5-7b-seed20260924').name != archive_path(tmp_path/'qwen2.5-7b-seed20260923').name
    with pytest.raises(ValueError, match='tests required'):
        run(tmp_path, tmp_path/'sites.json', tmp_path/'out')
    out = tmp_path/'out'; out.mkdir()
    with pytest.raises(ValueError, match='never overwrite'):
        run(tmp_path, tmp_path/'sites.json', out, test_evidence={'status': 'passed'})
    with pytest.raises(ValueError, match='threads'):
        run(tmp_path, tmp_path/'sites.json', out, threads=20)
    save(tmp_path/'receipt.json', {'status': 'checked'})
    assert json.loads((tmp_path/'receipt.json').read_text()) == {'status': 'checked'}
    assert not (tmp_path/'receipt.json.tmp').exists()
