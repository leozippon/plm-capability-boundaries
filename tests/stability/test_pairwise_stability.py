import struct
import numpy as np
import pandas as pd
import pytest
from src.capability.stability.pairwise_stability import (
    QC_WIDTH_COLUMNS, build_cohort, indel_construct, read_plmc, separation_stratum)


def measurement(name, seq, value, width=.2, label='mut'):
    row = dict(WT_name=name, aa_seq=seq, mut_type=label, dG_ML=str(value),
               name=name+'_'+seq)
    # Both proteases carry their own interval; the combined one alone does not
    # exclude a channel pinned at the fit bound.
    for column in QC_WIDTH_COLUMNS:
        row[column] = width
        row[column+'_low'] = value-width/2
        row[column+'_high'] = value+width/2
    return row


def inputs():
    rows=[]
    for name in ['a', 'b', 'c']:
        for seq,value in [('AAA',2),('CAA',1),('ACA',1),('AAC',1),('CCA',.7),('CAC',.8)]:
            rows.append(measurement(name,seq,value,label='wt' if seq=='AAA' else 'mut'))
    return pd.DataFrame(rows), {n:{'kind':'natural','sequence':'AAA'} for n in ['a','b','c']}, {
        'status':'admitted','provenance':{'method':'synthetic fixture'},
        'assignments':{'a':'g1','b':'g1','c':'g2'}}


def test_cohort_group_selection_exact_cycles_and_replicate_retention():
    f,c,g=inputs()
    f=pd.concat([f,pd.DataFrame([measurement('a','CAA',1.4)])],ignore_index=True)
    out=build_cohort(f,c,g,cap=2)
    assert [r['name'] for r in out['backgrounds']]==['a','c']
    a=out['backgrounds'][0]
    assert a['measurements']['CAA']['value']==pytest.approx(1.2)
    assert len(a['measurements']['CAA']['rows'])==2
    assert a['measurements']['CAA']['rows'][1]['high']==pytest.approx(1.5)
    eps={r['sequences'][-1]:r['epsilon'] for r in a['cycles']}
    assert eps==pytest.approx({'CCA':.5,'CAC':.6})
    # Outcome changes never alter which support was sampled.
    f['dG_ML']=pd.to_numeric(f.dG_ML)+10
    again=build_cohort(f,c,g,cap=2)
    assert [[r['sequences'] for r in b['cycles']] for b in again['backgrounds']]==[[r['sequences'] for r in b['cycles']] for b in out['backgrounds']]


def test_qc_censoring_and_final_group_admission():
    f,c,g=inputs();g['status']='provisional'
    with pytest.raises(ValueError,match='admitted'):build_cohort(f,c,g,cap=2)
    g['status']='admitted';del g['assignments']['b']
    with pytest.raises(ValueError,match='cover'):build_cohort(f,c,g,cap=2)
    f,c,g=inputs();f.loc[f.aa_seq=='CAA','dG_ML']='>5'
    assert not build_cohort(f,c,g,cap=2)['backgrounds']
    f,c,g=inputs();mask=f.aa_seq=='CAA'
    f.loc[mask,'deltaG_95CI']=.6;f.loc[mask,'deltaG_95CI_low']=.7;f.loc[mask,'deltaG_95CI_high']=1.3
    assert not build_cohort(f,c,g,cap=2)['backgrounds']
    f,c,g=inputs();f.loc[0,'deltaG_95CI']=.1
    with pytest.raises(ValueError,match='deltaG_95CI differs'):build_cohort(f,c,g,cap=2)
    # A tight combined interval must not admit a row whose protease channel is
    # unidentified; each channel carries the same boundary.
    for channel in ('deltaG_t_95CI','deltaG_c_95CI'):
        f,c,g=inputs();mask=f.aa_seq=='CAA'
        f.loc[mask,channel]=30.;f.loc[mask,channel+'_low']=-14.;f.loc[mask,channel+'_high']=16.
        assert not build_cohort(f,c,g,cap=2)['backgrounds']
    f,c,g=inputs()
    with pytest.raises(ValueError,match='missing measurement columns'):
        build_cohort(f.drop(columns=['deltaG_t_95CI']),c,g,cap=2)


def test_excluded_accounting_and_separation_strata():
    assert [separation_stratum(1,2),separation_stratum(4,1),separation_stratum(1,11)]==['1-2','3-9','10+']
    with pytest.raises(ValueError,match='distinct sites'):separation_stratum(5,5)
    f,c,g=inputs()
    out=build_cohort(f,c,g,cap=2)
    # Background 'b' shares group g1 with 'a'; nothing may vanish unaccounted.
    assert {(e['name'],e['reason']) for e in out['excluded']}=={('b','group_already_represented')}
    assert out['summary']['backgrounds']==2 and out['summary']['groups_covered']==2
    assert out['summary']['cycles']==4
    assert out['summary']['separation_strata']=={'1-2':4}
    assert out['summary']['groups_per_separation_stratum']=={'1-2':2,'3-9':0,'10+':0}
    # Two backgrounds carry the same synthetic sequences, so the workload the
    # extraction pays for is the deduplicated set, not the per-background sum.
    assert out['summary']['sequences_per_background_sum']==12
    assert out['summary']['distinct_sequences']==6 and out['summary']['residues']==18
    assert out['summary']['natural_groups_in_map']==2
    f,c,g=inputs();f=f[f.WT_name!='c']
    assert {(e['name'],e['reason']) for e in build_cohort(f,c,g,cap=2)['excluded']}==\
        {('b','group_already_represented'),('c','no_accepted_wt_row')}
    f,c,g=inputs();f=f[~((f.WT_name=='c')&(f.aa_seq=='CAC'))]
    assert {(e['name'],e['reason'],e['eligible_cycles']) for e in
            build_cohort(f,c,g,cap=2)['excluded']}=={('b','group_already_represented',2),('c','below_cap',1)}


def test_plmc_layout_and_cycle_cancels_fields(tmp_path):
    # Deliberately asymmetric amino-acid matrix detects transposed serialization.
    header=struct.pack('<5i5f',2,2,1,0,7,.2,.01,1,0,1)
    raw=header+b'AC'+np.array([1],'<f4').tobytes()+b'AA'+np.array([4,9],'<i4').tobytes()
    raw+=np.zeros(4,'<f4').tobytes()+np.array([2,7,3,11],'<f4').tobytes()
    raw+=np.zeros(4,'<f4').tobytes()+np.array([1,2,4,9],'<f4').tobytes()
    p=tmp_path/'p';p.write_bytes(raw);m=read_plmc(p)
    assert m.positions.tolist()==[4,9]
    assert m.score('AC')==15
    assert m.score('CA')==14
    assert m.cycle('AA','CA','AC','CC')==4
    p.write_bytes(raw+b'x')
    with pytest.raises(ValueError,match='byte count'):read_plmc(p)
    with pytest.raises(ValueError,match='constituent'):m.cycle('AA','CA','AC','CA')


def test_indel_construct_is_refused_from_the_substitution_support():
    """An indel construct whose aa_seq is truncated to wild-type length is length
    matched to the substitution states, so it would otherwise enter a cycle as a
    substitution. It must be refused, and a state that rests only on such rows
    must disappear with them.
    """
    f, c, g = inputs()
    # 'CCC' reaches the cohort only through an insertion construct whose sequence
    # the source truncated to the wild type's three residues.
    spurious = [measurement('a', 'CCC', .5, label='insA2'),
                measurement('a', 'CCC', .5, label='insA2')]
    # Every constituent single of the CCC double is already measured, so without
    # the refusal this fixture would gain a third eligible cycle on background a.
    f = pd.concat([f, pd.DataFrame(spurious)], ignore_index=True)
    out = build_cohort(f, c, g, cap=2)
    background = next(r for r in out['backgrounds'] if r['name'] == 'a')
    assert 'CCC' not in background['sequences']
    assert 'CCC' not in background['measurements']
    assert all('CCC' not in cycle['sequences'] for cycle in background['cycles'])
    assert background['eligible_cycles'] == 2
    assert out['summary']['rows_accepted'] == 18


def test_an_indel_row_cannot_perturb_a_genuine_state_median():
    """A substitution state whose median mixes indel rows is not a measured
    substitution value either; the indel rows leave the median, not the state."""
    f, c, g = inputs()
    contaminated = [measurement('a', 'CAA', 9.0, label='insG1'),
                    measurement('a', 'CAA', 9.0, label='delG2')]
    out = build_cohort(pd.concat([f, pd.DataFrame(contaminated)], ignore_index=True), c, g, cap=2)
    background = next(r for r in out['backgrounds'] if r['name'] == 'a')
    state = background['measurements']['CAA']
    assert state['value'] == 1.0
    assert [row['name'] for row in state['rows']] == ['a_CAA']


def test_the_indel_predicate_folds_case_and_spares_substitution_codes():
    flags = indel_construct(['wt', 'A2C', 'I25N', 'D10E', 'insA2', 'DEL3', 'Ins_G1', 'delta'])
    assert list(flags) == [False, False, False, False, True, True, True, True]
