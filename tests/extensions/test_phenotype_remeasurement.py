"""Exact-state and common-denominator guards, not simulated phenotype replication."""
import copy

import numpy as np
import pytest

from src.capability.extensions import phenotype_remeasurement as r


def fixture():
    wt = 'MACDE'
    mutations = ['A2G', 'C3A', 'D4E', 'E5A']
    original = [{'mutation': m, 'sample_id': str(i), 'original_index': i,
                 'variant_index': i, 'cluster': 95} for i, m in enumerate(mutations)]
    canonical = [{'protein': 'RHO', **r.state(wt, m)} for m in mutations]
    rows = [{'protein': 'RHO', 'channel': c, 'accepted': True,
             'score': float(i), 'uncertainty': 0.1, 'exclusions': [],
             'mutation': s['mutation'], 'state_id': s['state_id']}
            for c in r.CHANNELS for i, s in enumerate(canonical)]
    return wt, original, canonical, rows


def test_exact_state_hash_join_and_wrong_WT_reject():
    wt, original, canonical, rows = fixture()
    registry, matched, coverage = r.join_measurements(wt, wt, original, canonical, rows)
    assert len(registry) == len(matched) == 4
    assert matched[0]['state_id'] == r.text_sha(wt) + ':' + r.text_sha('MGCDE')
    assert all(coverage[c]['original_accepted_matches'] == 4 for c in r.CHANNELS)
    with pytest.raises(ValueError, match='external WT'):
        r.join_measurements(wt, 'MG CDE', original, canonical, rows)
    bad = copy.deepcopy(canonical)
    bad[0]['mutated_sequence'] = 'MMMMM'
    with pytest.raises(ValueError, match='full protein-state mismatch'):
        r.join_measurements(wt, wt, original, bad, rows)
    bad_rows = copy.deepcopy(rows)
    bad_rows[0]['state_id'] = canonical[1]['state_id']
    with pytest.raises(ValueError, match='canonical state mismatch'):
        r.join_measurements(wt, wt, original, canonical, bad_rows)
    with pytest.raises(ValueError, match='original WT'):
        r.state(wt, 'G2A')


def test_common_intersection_missing_QC_and_blocked():
    wt, original, canonical, rows = fixture()
    rows[0].update(accepted=False, score=None, uncertainty=None, exclusions=['score_missing'])
    registry, matched, coverage = r.join_measurements(wt, wt, original, canonical, rows)
    assert len(matched) == 3
    assert coverage[r.CHANNELS[0]]['original_accepted_matches'] == 3
    assert coverage[r.CHANNELS[1]]['original_accepted_matches'] == 4
    assert registry[0]['methods'][r.CHANNELS[0]]['missing_QC'] == ['score_missing']
    assert not registry[0]['primary_matched']
    rows[1]['accepted'] = False
    with pytest.raises(ValueError, match='blocked'):
        r.join_measurements(wt, wt, original, canonical, rows)
    with pytest.raises(ValueError, match='empty original'):
        r.join_measurements(wt, wt, [], canonical, rows)


def test_accepted_missing_SE_and_duplicate_fail():
    wt, original, canonical, rows = fixture()
    rows[0]['uncertainty'] = None
    with pytest.raises(ValueError, match='score/SE QC'):
        r.join_measurements(wt, wt, original, canonical, rows)
    rows[0]['uncertainty'] = 0.1
    with pytest.raises(ValueError, match='duplicate accepted'):
        r.join_measurements(wt, wt, original, canonical, rows + rows[:1])


def test_paired_contrast_algebra_preserves_predictions_and_ties():
    bpl = np.array([1., 1., 3., 4.])
    bmpl = np.array([4., 3., 2., 1.])
    method1 = np.array([1., 2., 3., 4.])
    method2 = method1[::-1]
    before = bpl.copy()
    result = r.paired_contrasts(bpl, bmpl, method1, method2)
    a, b = [result['methods'][c] for c in r.CHANNELS]
    assert a['difference_BMPL_minus_BPL'] == pytest.approx(-1 - r.rho(bpl, method1))
    assert b['difference_BMPL_minus_BPL'] == pytest.approx(1 - r.rho(bpl, method2))
    assert result['method_contrast_difference_method2_minus_method1'] == pytest.approx(
        b['difference_BMPL_minus_BPL'] - a['difference_BMPL_minus_BPL'])
    np.testing.assert_array_equal(bpl, before)
    with pytest.raises(ValueError, match='constant'):
        r.rho([1, 1, 1], [1, 2, 3])
    with pytest.raises(ValueError, match='unaligned'):
        r.rho([1, 2, 3], [1, 2])


def test_no_fake_independence_or_winner_selection():
    contrast = r.paired_contrasts([1, 2, 3], [3, 2, 1], [1, 2, 3], [3, 2, 1])
    cells = [{'model': model, 'seed': seed, **contrast}
             for model in ['model-a', 'model-b'] for seed in [1, 2, 3]]
    summary = r.descriptive_summary(cells)
    assert len(summary['models']) == 2
    assert summary['experimental_proteins'] == summary['experimental_families'] == 1
    assert summary['inferential_tests'] == summary['population_intervals'] == []
    assert summary['winner_selection'] is False
    assert summary['sign_counts_mean_seed'][r.CHANNELS[0]]['negative'] == 2
    assert summary['sign_counts_mean_seed'][r.CHANNELS[1]]['positive'] == 2
    assert summary['models'][0]['mean_seed_methods'][r.CHANNELS[0]]['difference_BMPL_minus_BPL'] == -2
    with pytest.raises(ValueError, match='three distinct'):
        r.descriptive_summary(cells[:-1])


def test_project_argument_paths_are_normalized_once(tmp_path):
    root = tmp_path / 'repository'
    expected = root / 'results' / 'prepared'
    assert r.project_path(root, 'results/prepared') == expected
    assert r.project_path(root, expected) == expected
    assert r.project_path(root, 'results/../results/prepared') == expected
    assert r.project_path(root, 'results/prepared').relative_to(root).as_posix() == 'results/prepared'


def test_source_paths_do_not_depend_on_preparation_output_location(tmp_path):
    assert r.measurement_source_path(tmp_path, 'catalog_candidates/scores.csv') == (
        tmp_path / 'results/extensions/phenotype_followups_20261007/discovery/catalog_candidates/scores.csv')
    assert r.measurement_source_path(tmp_path, 'data/source.csv') == tmp_path / 'data/source.csv'
    for invalid in ('../source.csv', '/tmp/source.csv', ''):
        with pytest.raises(ValueError, match='unsafe'):
            r.measurement_source_path(tmp_path, invalid)


def test_replication_is_method_specific_metadata_not_invented_counts():
    manifest = [dict(protein='RHO', channel=c, study_replicates=n,
                     per_variant_replicates='not reported') for c, n in zip(r.CHANNELS, (2, 8))]
    result = r.measurement_replication(manifest)
    assert result[r.CHANNELS[0]]['study_replicates'] == 2
    assert result[r.CHANNELS[1]]['study_replicates'] == 8
    assert result[r.CHANNELS[1]]['per_variant_replicates'] == 'not reported'
    with pytest.raises(ValueError, match='coverage'):
        r.measurement_replication(manifest[:1])


def test_original_held_family_folds_reject_leakage_or_changed_memberships():
    projected = [[0, [95], [1], [[1]]], [1, [1], [95], [[95]]]]
    records = [{'fold': f, 'held_families': h, 'training_families': t,
                'inner_folds': [{'validation_families': v} for v in inner]}
               for f, h, t, inner in projected]
    assert r.verify_fold_records(records, projected, 95) == 0
    bad = copy.deepcopy(records)
    bad[0]['training_families'].append(95)
    with pytest.raises(ValueError, match='fold mismatch'):
        r.verify_fold_records(bad, projected, 95)
