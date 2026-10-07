"""Small direct reproductions of fixed-OOF aggregation and joint family draws."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.capability.extensions import phenotype_strata as p


def test_direct_assay_spearman_and_identity_failure():
    samples = [dict(sample_id=str(i), original_index=i, assay='a', mutation=f'A{i+1}V', cluster=7) for i in range(4)]
    saved = dict(sample_id=np.array([str(i) for i in range(4)]), original_index=np.arange(4),
                 target=np.array([1.,2.,3.,4.]), BPL=np.array([1.,3.,2.,4.]), BMPL=np.array([1.,2.,4.,3.]))
    p.validate_archive(saved, samples, saved['target'])
    row = p.assay_scores(saved, samples, 'model', 1)[0]
    assert row['contrast'] == pytest.approx(spearmanr(saved['target'], saved['BMPL'])[0]-spearmanr(saved['target'], saved['BPL'])[0])
    for field in ['sample_id', 'target', 'original_index']:
        bad = dict(saved); bad[field] = saved[field][::-1]
        with pytest.raises(ValueError, match='misaligned'):
            p.validate_archive(bad, samples, saved['target'])
    duplicates = [*samples[:-1], samples[0]]
    with pytest.raises(ValueError, match='duplicate'):
        p.validate_archive(saved, duplicates, saved['target'])


def test_seed_mean_then_equal_assays_not_rows_or_seeds_as_units():
    metadata = pd.DataFrame([dict(assay=a, category=c) for a,c in [('a','Activity'),('b','Activity'),('c','Binding')]])
    scores = []
    for seed in (1,2):
        for assay, value in [('a',0.),('b',.6),('c',.4)]:
            scores.append(dict(arm='m', seed=seed, assay=assay, cluster=7, baseline_spearman=.2,
                               augmented_spearman=.2+value+seed/10, contrast=value+seed/10))
    keys, matrices, _ = p.family_matrix(scores, metadata, ['m'], [7,8], seeds=(1,2))
    assert matrices['contrast'][0, keys.index(('m','Activity'))] == pytest.approx(.45)
    assert matrices['contrast'][0, keys.index(('m','Binding'))] == pytest.approx(.55)
    assert np.isnan(matrices['contrast'][1]).all()
    with pytest.raises(ValueError, match='incomplete'):
        p.family_matrix(scores[:-1], metadata, ['m'], [7,8], seeds=(1,2))


def test_shared_family_overlap_and_no_checkpoint_bootstrap():
    # Overlapping categories and duplicate fixed checkpoints have identical draws.
    values = np.array([[.1,.1,.2], [.2,.2,.4], [.3,.3,np.nan], [.4,.4,np.nan]])
    stats, boot = p.shared_bootstrap(values, draws=1000, seed=12)
    np.testing.assert_array_equal(boot[:,0], boot[:,1])
    rng = np.random.default_rng(12)
    direct = []
    attempted = 0
    while len(direct)<1000:
        for _ in range(min(100,1000-len(direct))):
            indices=rng.integers(4,size=4); attempted+=1
            local=values[indices]
            if np.isfinite(local[:,2]).any():
                direct.append(np.nanmean(local, axis=0))
    np.testing.assert_allclose(boot, direct)
    assert stats['attempted_draws']==attempted
    assert stats['jointly_rejected_draws']>0
    assert stats['empty_draws_by_contrast'][2]==stats['jointly_rejected_draws']
    assert stats['supported_families']==[4,4,2]
    assert stats['family_size']==3  # hypotheses, not resampling units
    assert stats['original_families']==4
    np.testing.assert_allclose(stats['point'], np.nanmean(values,axis=0))


def test_export_groups_are_verified_not_inferred_from_category_counts():
    samples = [dict(sample_id='x', assay='a', mutation='A1V', cluster=7),
               dict(sample_id='y', assay='b', mutation='A2V', cluster=7)]
    exported = pd.DataFrame([dict(sample_id=r['sample_id'], unit_id=r['assay'],
        mutation=r['mutation'], group_id=str(r['cluster']), headline_common_member=True)
        for r in samples])
    result = p.audit_export_groups(samples, exported)
    assert result['biological_families'] == 1
    assert result['rows'] == 2
    bad = exported.copy(); bad.loc[1, 'group_id'] = '8'
    with pytest.raises(ValueError, match='group differs'):
        p.audit_export_groups(samples, bad)
    bad = exported.copy(); bad.loc[1, 'headline_common_member'] = False
    with pytest.raises(ValueError, match='support differs'):
        p.audit_export_groups(samples, bad)


def test_nonestimable_or_infinite_values_fail():
    with pytest.raises(ValueError, match='two families'):
        p.shared_bootstrap([[1,np.nan],[2,3]],draws=10)
    with pytest.raises(ValueError, match='invalid'):
        p.shared_bootstrap([[1,np.inf],[2,3]],draws=10)
