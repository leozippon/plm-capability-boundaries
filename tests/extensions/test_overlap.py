"""Extension-only strict support, historical folds and paired algebra contracts."""
import hashlib
from typing import Any

import numpy as np
import pytest

from src.capability.extensions import overlap as o
from src.capability.readouts.readout_analysis import family_folds, nested_predict


def fixture_rows():
    assay: dict[str, Any] = dict(assay='a',cluster='f',wildtype='ACD',mutants=['A1C','A1D','A1E','C2A:D3A'])
    site = dict(assay_id='a',cluster='f',wt_sequence='ACD',wt_sha256=hashlib.sha256(b'ACD').hexdigest(),
                wt_position=1,residue='A',rsa=.4,coordinate_present=True,contact_atom_present=False,
                status='admitted',mapping_kind='full_entity',method='X-RAY DIFFRACTION',source_sha256='s',
                source_path='s.cif',entry_id='s',entity_position=1,model=1,chain='A')
    return assay,site


def test_strict_single_exact_identity_and_primary_not_degree_limited():
    assay,site = fixture_rows(); rows,excluded = o.select_rows([assay],[site])
    assert len(rows)==3 and excluded['not_strict_single']==1
    assert o.structural_features(rows).shape==(3,7)
    assert np.all(o.structural_features(rows)[:,0]==.4)
    for key,value in [('wt_sha256','bad'),('residue','C'),('cluster','other'),('entity_position',None)]:
        with pytest.raises(ValueError):
            o.select_rows([assay],[dict(site,**{key:value})])
    with pytest.raises(ValueError,match='duplicate structural'):
        o.select_rows([assay],[site,site])
    with pytest.raises(ValueError,match='duplicate mutation'):
        o.select_rows([dict(assay,mutants=['A1C','A1C'])],[site])
    assert not o.select_rows([assay],[dict(site,method='THEORETICAL MODEL')])[0]
    assert not o.select_rows([assay],[dict(site,mapping_kind='approximate')])[0]


def test_rerank_subset_with_ties():
    aids = np.asarray(['a']*3+['b']*3)
    values = np.asarray([-1,-.2,-.2,5,7,9])
    result = o.rerank(values,aids)
    assert not np.array_equal(result[:3],values[:3])
    assert result[1]==result[2]
    for assay in ('a','b'):
        assert abs(result[aids==assay].mean())<1e-12
        assert np.isclose(result[aids==assay].std(),1)


def signature(groups,seed=20260923):
    result=[]
    for fold,held in enumerate(family_folds(groups,5,seed)):
        train=sorted(set(groups)-set(held))
        inner=family_folds(train,4,seed+100+fold)
        result.append((fold,tuple(held),tuple(train),tuple(tuple(v) for v in inner)))
    return result


def test_projection_rejects_empty_never_regenerates():
    groups=list('abcdefghij'); sig=signature(groups)
    projected=o.project_folds(sig,groups)
    assert len(projected)==5
    import json
    numeric=np.arange(10,dtype=np.int64)
    assert json.loads(json.dumps(o.project_folds(signature(numeric),numeric)))
    with pytest.raises(o.UnsupportedProjection,match='empty'):
        o.project_folds(sig,[groups[0]])
    # Unsupported inner validation is explicit even if every outer fold survives.
    subset=[held[0] for _,held,_,_ in sig]
    with pytest.raises(o.UnsupportedProjection,match='inner'):
        o.project_folds(sig,subset)


def test_projected_full_cohort_matches_existing_nested_fit():
    rng=np.random.default_rng(4); groups=np.repeat(list('abcdefghij'),3)
    aids=groups.copy(); x=rng.normal(size=(30,3)); y=rng.normal(size=30)
    expected,records=nested_predict(x,y,aids,groups,device='cpu')
    actual,realized=o.projected_predict(x,y,aids,groups,signature(groups))
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    assert [r['alpha'] for r in realized]==[r['alpha'] for r in records]
    assert all(len(r['inner_folds'])==4 for r in realized)


def test_paired_algebra():
    scores={'B':.1,'B+M':.4,'B+X':.2,'B+M+X':.3}
    values=o.contrast_values(scores,'spearman')
    assert values['model_without_X']==pytest.approx(.3)
    assert values['model_with_X']==pytest.approx(.1)
    assert values['attenuation']==pytest.approx(.2)
    mse=o.contrast_values(scores,'rank_mse')
    assert mse['attenuation']==pytest.approx(-.2)


def test_zero_overlap_and_joint_seed_averaging():
    assay,_=fixture_rows()
    assert o.select_rows([assay],[])[0]==[]
    assert o.structural_features([]).shape==(0,7)
    rows=[]
    for family in ('f','g'):
        for seed,value in ((1,0.),(2,2.)):
            rows.append(dict(assay=family,cluster=family,arm='m',seed=seed,
                spearman_contrasts=dict.fromkeys(o.CONTRASTS,value),rank_mse_contrasts=dict.fromkeys(o.CONTRASTS,value)))
    result=o.joint_bootstrap(rows,bootstrap=100)
    assert result['status']=='complete'
    assert all(r['point']==1 for r in result['contrasts'])
    assert all(r['simultaneous']==[1.,1.] for r in result['contrasts'])


def test_empty_site_table_writes_truthful_receipt(tmp_path):
    import json
    from scripts.capability.extensions.fit_structural_overlap import run
    from scripts.capability.reporting import export_prediction_details as d
    assay,_=fixture_rows(); assay.update(measured=[1,2,3,4])
    cohort=tmp_path/d.MUTATION; cohort.parent.mkdir(parents=True,exist_ok=True)
    cohort.write_text(json.dumps({'assays':[assay]}))
    admission=tmp_path/d.LOCAL; admission.parent.mkdir(parents=True,exist_ok=True)
    admission.write_text(json.dumps(dict(cohort_sha256=hashlib.sha256(cohort.read_bytes()).hexdigest(),support={'assay_ids':['a']})))
    sites=tmp_path/'sites.json'; sites.write_text('[]')
    out=tmp_path/'out';run(tmp_path,sites,out,100)
    receipt=json.loads((out/'completion.json').read_text())
    assert receipt['status']=='not_fitted' and receipt['overlap_rows']==0
    assert not list(out.glob('*oof*'))
