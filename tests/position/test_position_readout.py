"""Required same-support contrast and prediction invariants for position readout."""
import numpy as np
import pytest
from src.capability.position.position_readout import CONTRASTS, evaluate


def test_unique_information_contrasts_are_conditional_and_directional():
    assert CONTRASTS['unique_own'] == ('joint', 'downstream')
    assert CONTRASTS['unique_downstream'] == ('joint', 'own')
    assert CONTRASTS['joint_over_full'] == ('joint', 'full')


def test_real_nested_fits_hold_families_out_and_share_support():
    rng=np.random.default_rng(17)
    rows=[]
    for group in range(10):
        n=12
        own=rng.normal(size=n); downstream=rng.normal(size=n)
        rows.append(dict(assay=f'a{group}',cluster=f'f{group}',mutants=[str(i) for i in range(n)],
                         measured=own+downstream,own=own,downstream=downstream,full=own+downstream,
                         upstream=np.zeros(n),
                         S=rng.normal(size=(n,2)),P_block=rng.normal(size=(n,2)),
                         wall=rng.normal(size=(n,2)),rf3=rng.normal(size=(n,2))))
    report,pred=evaluate(rows,device='cpu',bootstrap=20,numerical_sensitivity=True)
    assert report['variants']==120 and report['families']==10
    assert report['folds_identical']
    assert all(len(x)==120 and np.isfinite(x).all() for x in pred.values())
    for local in ('wall','rf3'):
        np.testing.assert_array_equal(pred[local+'_full'],pred[local+'_upstream_free'])
        np.testing.assert_array_equal(pred[local+'_full'],pred[local+'_term_sum'])
    for control in report['controls'].values():
        for row in control['assays']:
            assert row['unique_own']==pytest.approx(row['joint']-row['downstream'])
            assert row['unique_downstream']==pytest.approx(row['joint']-row['own'])
        assert control['summaries']['joint_over_control']['point']>0.5
