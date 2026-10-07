"""CPU retained-response contracts and real small nested-ridge comparisons."""
import copy
import hashlib
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.capability.extensions.responses import (RetainedResponses, mapped_pairs,
    prepare_structure_pairs, structure_sites, analyse_contacts, evaluate_bins, potential_distance_census)


def archive() -> tuple[dict, list[dict], dict]:
    wild = 'ACDEFGHIKLMN'
    mutant = wild[:1]+'A'+wild[2:]
    # Formatting token at end, and one multiresidue token at residues 4..5.
    counts = [1,1,1,1,2,1,1,1,1,1,1,0]
    ids = [90,*range(12)]
    changed = ids.copy(); changed[2] = 50
    wt = np.arange(1,13,dtype=np.float32)/10
    mt = wt.copy(); mt[1:] += np.float32(.05)
    data = dict(position_nats=np.r_[wt,mt],position_offsets=np.array([0,12,24]),
                position_residue_counts=np.array(counts*2),position_residue_offsets=np.array([0,0]),
                position_sum_check_nats=0.,mutants=np.array(['C2A']),
                wt_likelihood=-float(wt.sum(dtype=np.float32)),
                likelihood=np.array([float(wt.sum(dtype=np.float32))-float(mt.sum(dtype=np.float32))]))
    states = [dict(sequence=s,ids=i,span=[1,13],counts=counts,offset=0) for s,i in ((wild,ids),(mutant,changed))]
    identity = dict(wildtype=wild,mutants=['C2A'],sequences=[mutant])
    return data,states,identity


def test_native_sign_closure_and_separate_tokens():
    data,states,identity = archive()
    r = RetainedResponses(data,states,identity).response(0)
    assert r['own'] < 0
    assert r['native'] == data['likelihood'][0]
    assert r['own']+sum(r['bins'])+r['remainder'] == pytest.approx(r['native'])
    assert abs(r['closure_nats']) <= r['closure_bound_nats']
    assert {x['j'] for x in r['receivers']} == {2,3,6,7,8,9,10,11}
    assert r['token_census'] == dict(primary=8,own=1,special=1,multiresidue=1,upstream=1)
    assert r['bin_support'] == [6,2,0,0]
    assert r['separate']['special'] < 0 and r['separate']['multiresidue'] < 0
    # The original float32 native reduction residual is permitted, not zeroed.
    assert r['closure_nats'] != 0


@pytest.mark.parametrize('defect', ['identity','native','counts','dtype','retention','multi','nonsense'])
def test_invalid_sources_fail_explicitly(defect):
    data,states,identity = archive()
    if defect=='identity': identity['mutants']=['C2D']
    if defect=='native': data['likelihood'][0] += 1
    if defect=='counts': states[0]['counts'] = [1]*12
    if defect=='dtype': data['position_nats'] = data['position_nats'].astype(np.float64)
    if defect=='retention': data['position_sum_check_nats'] = 1e-8
    if defect=='multi': data['mutants']=np.array(['C2A:D3A']); identity['mutants']=['C2A:D3A']
    if defect=='nonsense': identity['sequences']=['A'*12]
    with pytest.raises(ValueError): RetainedResponses(data,states,identity)


def sites(wild):
    return [dict(j=j,identity=aa,atom='CA' if aa=='G' else 'CB',xyz=[float(j),0.,0.],rsa=None if j%2 else .3) for j,aa in enumerate(wild)]


def test_contact_geometry_glycine_and_all_receivers():
    data,states,identity=archive()
    r = RetainedResponses(data,states,identity).response(0)
    mapping=sites(identity['wildtype'])
    pairs=mapped_pairs(r,mapping,identity['wildtype'])
    assert [p['j'] for p in pairs] == [6,7,8,9,10,11]
    # j=9 is exactly 8 A from i=1, hence not contact.
    assert [p['j'] for p in pairs if p['contact']] == [6,7,8]
    mapping[6]['rsa']=1.2  # Original structural RSA is deliberately un-clipped.
    assert mapped_pairs(r,mapping,identity['wildtype'])[0]['rsa']==1.2
    mapping[5]['atom']='CB'
    with pytest.raises(ValueError): mapped_pairs(r,mapping,identity['wildtype'])


def test_preparation_without_responses_is_blind_and_has_receipts():
    wild='ACDEFGHIKLMNPQRSTVWY'
    proteins: list[dict] = [dict(assay='a',family='f',protein='p',wildtype=wild,sites=sites(wild),mutations=['C2A'])]
    result=prepare_structure_pairs(proteins)
    unit=result['units'][0]
    assert result['stage']=='prepared_without_NLL'
    assert all('response' not in p for p in unit['pairs'])
    assert unit['eligible_receivers']==16
    assert sum(p['base_weight'] for p in unit['pairs']) == pytest.approx(1)
    assert sum(p['matched_weight'] for p in unit['pairs']) == pytest.approx(1)
    assert unit['strata'][0]['matched']
    assert unit['strata'][0]['distance_imbalance'] != 0
    assert not unit['strata'][1]['matched']
    assert unit['identity_sensitivity']['matched_receivers']==0
    assert unit['rsa_sensitivity']['missing_rsa_excluded']==8
    changed=copy.deepcopy(proteins); changed[0]['phenotype']=-9999
    assert prepare_structure_pairs(changed)==result


def structural_fixture(wild, missing=()):
    provenance=dict(source_sha256='a'*64,source_path='structure.cif',entry_id='TEST',entity_id='1',
                    chain='A',auth_chain='A',model=1,mapping_kind='full_entity')
    rows=[dict(assay_id='a',cluster='f',wt_sequence=wild,wt_sha256=hashlib.sha256(wild.encode()).hexdigest(),
               wt_position=j+1,residue=aa,coordinate_present=j not in missing,contact_atom_present=j not in missing,
               contact_coordinates=[j,0.,0.] if j not in missing else None,contact_atom='CA' if aa=='G' else 'CB',
               status='admitted',rsa=None,**provenance) for j,aa in enumerate(wild)]
    coverage=dict(assays=[dict(assay='a',cluster='f',status='admitted',method='SOLUTION NMR',**provenance)],
                  sources=[dict(sha256='a'*64,path='structure.cif',status='parsed',method='SOLUTION NMR')])
    return rows,coverage


def test_structure_api_exact_identity_and_missingness():
    wild='ACDG'
    rows,coverage=structural_fixture(wild,missing=(2,))
    result,receipt=structure_sites(rows,assay='a',family='f',wildtype=wild,coverage=coverage)
    assert [s['j'] for s in result] == [0,1,3]
    assert receipt['excluded']==[dict(j=2,status='admitted',mapping_kind='full_entity')]
    bad=copy.deepcopy(rows); bad[1]['residue']='A'
    with pytest.raises(ValueError): structure_sites(bad,assay='a',family='f',wildtype=wild,coverage=coverage)
    with pytest.raises(ValueError): structure_sites(rows[:-1],assay='a',family='f',wildtype=wild,coverage=coverage)


def test_distance_adjustment_and_family_not_pair_inference():
    pairs=[]
    # Linear log-distance response: apparent contact effect disappears on adjustment.
    for j,contact in ((3,True),(4,True),(7,False),(8,False)):
        pairs.append(dict(j=j,distance=j,identity='A',response=2*np.log(j),wt_nll=float(j),rsa=.2,
                          stratum='3-8',contact=contact))
    result=analyse_contacts([dict(assay='a',family='f',protein='p',mutation='A1C',pairs=pairs)],draws=20)
    r=result['mutations'][0]['absolute']
    assert r['contrast'] < 0
    assert r['distance_imbalance']==-4
    assert r['adjusted_contrast']==pytest.approx(0,abs=1e-12)
    assert result['inference'] is None
    assert result['families']==['f']
    # Receiver duplication is not silently accepted as additional evidence.
    with pytest.raises(ValueError): analyse_contacts([dict(assay='a',family='f',protein='p',mutation='A1C',pairs=pairs+pairs)])


def test_missing_rsa_sensitivity_does_not_veto_primary():
    pairs=[dict(j=j,distance=j,identity='A',response=float(j),wt_nll=float(j),rsa=None,
                stratum='3-8',contact=j<5) for j in (3,4,7,8)]
    rows=[dict(assay=f'a{f}',family=f'f{f}',protein=f'p{f}',mutation='A1C',pairs=pairs) for f in range(8)]
    result=analyse_contacts(rows,draws=20)
    assert result['inference'] is not None
    assert ('absolute','contrast') in result['inferential_columns']
    assert ('rsa','contrast') in result['unavailable_columns']


def prediction_rows():
    rows=[]
    for f in range(8):
        t=np.linspace(-1,1,9)
        responses=[]
        for j,v in enumerate(t):
            responses.append(dict(mutation=f'A{j+1}C',own=.1*v,bins=[v,0.,0.,0.],
                                  remainder=.2*v,native=1.3*v,bin_support=[3,0,0,0]))
        rows.append(dict(assay=f'a{f}',cluster=f'f{f}',mutants=[r['mutation'] for r in responses],
                         measured=t+.1*np.sin(t+f),B=np.c_[np.cos(t+f)],responses=responses))
    return rows


def test_bins_zero_census_real_shared_nested_fits():
    rows=prediction_rows()
    result=evaluate_bins(rows,seeds=(3,4),draws=20,outer_splits=4,inner_splits=3)
    assert result['tested_bins']==['1-8']
    assert result['support']['129+']==dict(receivers=0,nonempty_rows=0,zero_feature=True)
    assert result['folds_identical']
    assert len(result['folds'])==4
    assert result['inference']['family_universe']==8
    for seed in (3,4):
        full=result['folds'][f'{seed}:full']; drop=result['folds'][f'{seed}:drop_1-8']
        assert [r['held_families'] for r in full]==[r['held_families'] for r in drop]
        assert [[i['validation_families'] for i in r['inner_folds']] for r in full]==[[i['validation_families'] for i in r['inner_folds']] for r in drop]
    bad=copy.deepcopy(rows); bad[0]['responses'][0]['bins'][2]=1
    with pytest.raises(ValueError): evaluate_bins(bad,seeds=(3,),draws=10)


def test_potential_bins_are_full_panel_sequence_only():
    rows=[dict(assay='long_without_structure',cluster='f',wildtype='A'*180,mutants=['A1C','A180C','A2C:A3C'])]
    result=potential_distance_census(rows)
    assert not result['structure_required']
    assert result['units'][0]['potential_residues']==[8,24,96,51]
    assert result['units'][1]['potential_residues']==[0,0,0,0]
    assert result['support']['129+']['potential_nonempty_rows']==1
    assert len(result['excluded_multiple_substitutions'])==1
    assert 'not actual token support' in result['qualification']


def test_all_zero_bins_not_tested():
    rows=prediction_rows()
    for row in rows:
        for r in row['responses']:
            r['remainder'] += sum(r['bins']); r['bins']=[0.,0.,0.,0.]; r['bin_support']=[0,0,0,0]
    result=evaluate_bins(rows,seeds=(3,),draws=10,outer_splits=4,inner_splits=3)
    assert result['tested_bins']==[] and result['inference'] is None


def test_cli_geometry_preparation_and_no_overwrite(tmp_path):
    path=Path(__file__).resolve().parents[2]/'scripts/capability/extensions/analyse_residue_responses.py'
    spec=importlib.util.spec_from_file_location('response_cli',path)
    assert spec is not None and spec.loader is not None
    cli=importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    wild='ACDEFGHIKLMN'
    rows,coverage=structural_fixture(wild)
    source=tmp_path/'input.json'; structure=tmp_path/'sites.json.gz'; out=tmp_path/'receipt.json'
    coverage_path=tmp_path/'coverage.json'; coverage_path.write_text(json.dumps(coverage))
    source.write_text(json.dumps(dict(assays=[dict(assay='a',cluster='f',protein='p',wildtype=wild,mutants=['C2A'])])))
    with gzip.open(structure,'wt') as handle:
        json.dump(rows,handle)
    args=['--mode','prepare','--input',str(source),'--structures',str(structure),'--coverage',str(coverage_path),'--out',str(out)]
    result=cli.main(args)
    assert result['eligible_receivers']==8
    assert result['source_sha256'][str(source)]==hashlib.sha256(source.read_bytes()).hexdigest()
    assert json.loads(out.read_text())['resources']['cpu_only']
    with pytest.raises(SystemExit): cli.main(args)


@pytest.mark.parametrize('defect',['excluded','theoretical','null_source','wrong_atom','wrong_source','nonexact'])
def test_structure_rejects_unbound_or_wrong_atom(defect):
    wild='ACDG'; rows,coverage=structural_fixture(wild)
    if defect=='excluded': rows[0]['status']='excluded_no_admissible_structure'
    if defect=='theoretical':
        coverage['assays'][0]['method']='THEORETICAL MODEL'; coverage['sources'][0]['method']='THEORETICAL MODEL'
    if defect=='null_source': rows[0]['source_sha256']=None
    if defect=='wrong_atom': rows[0]['contact_atom']='CA'
    if defect=='wrong_source': coverage['sources'][0]['path']='other.cif'
    if defect=='nonexact':
        coverage['assays'][0]['mapping_kind']='local_match'
        for r in rows: r['mapping_kind']='local_match'
    with pytest.raises(ValueError): structure_sites(rows,assay='a',family='f',wildtype=wild,coverage=coverage)


def mixed_archive():
    data,states,identity=archive()
    wt=states[0]; wild=identity['wildtype']; terms=data['position_nats'][:12]
    mutants=['C2A','C2A:D3A','D3A','E4A','F5A']
    counts=wt['counts']; starts=np.r_[0,np.cumsum(counts)[:-1]]; ends=starts+counts
    sequences=[]; new_states=[wt]; values=[terms]
    for i,mutation in enumerate(mutants):
        sequence=list(wild); ids=wt['ids'].copy()
        for token in mutation.split(':'):
            j=int(token[1:-1])-1; sequence[j]=token[-1]
            own=int(np.flatnonzero((starts<=j)&(j<ends))[0]); ids[1+own]=100+i+own
        if i==2: ids[10]=999  # Valid packing metadata, biologically shifted grid.
        seq=''.join(sequence); sequences.append(seq)
        new_states.append(dict(wt,sequence=seq,ids=ids))
        values.append(terms+np.float32(.05*(i+1)))
    data.update(position_nats=np.concatenate(values),position_offsets=np.arange(0,73,12),
                position_residue_counts=np.tile(counts,6),position_residue_offsets=np.zeros(6,dtype=int),
                mutants=np.array(mutants),likelihood=np.array([float(terms.sum(dtype=np.float32))-float(v.sum(dtype=np.float32)) for v in values[1:]]))
    identity.update(mutants=mutants,sequences=sequences)
    return data,new_states,identity


def test_mixed_archive_selects_original_indices_and_corruption_is_fatal():
    data,states,identity=mixed_archive()
    retained=RetainedResponses(data,states,identity)
    assert retained.selected_indices==[0,3,4]
    assert retained.selected_state_indices==[0,1,4,5]
    assert [r['reason'] for r in retained.exclusions]==['multiple_substitutions','grid_misalignment']
    with pytest.raises(ValueError): retained.response(1)
    row=dict(identity,B=[[i,10+i] for i in range(5)],measured=[11,22,33,44,55])
    projected=retained.project(row)
    assert projected['mutants']==['C2A','E4A','F5A']
    assert projected['sequences']==[identity['sequences'][i] for i in (0,3,4)]
    assert projected['B']==[[0,10],[3,13],[4,14]] and projected['measured']==[11,44,55]
    assert [r['original_variant_index'] for r in projected['responses']]==[0,3,4]
    bad=copy.deepcopy(data); bad['likelihood'][1]+=1  # Excluded double still validated.
    with pytest.raises(ValueError,match='closure'): RetainedResponses(bad,states,identity)


def test_unscored_mutation_is_explicit_biological_exclusion():
    wild='ACD'; mt='CCD'; wt_nll=np.array([1.,2.],dtype=np.float32)
    data=dict(position_nats=np.r_[wt_nll,wt_nll],position_offsets=np.array([0,2,4]),
              position_residue_counts=np.ones(4,dtype=int),position_residue_offsets=np.ones(2,dtype=int),
              position_sum_check_nats=0.,mutants=np.array(['A1C']),likelihood=np.array([0.]),wt_likelihood=-3.)
    states=[dict(sequence=s,ids=ids,span=[1,3],counts=[1,1],offset=1) for s,ids in ((wild,[1,2,3]),(mt,[4,2,3]))]
    retained=RetainedResponses(data,states,dict(wildtype=wild,mutants=['A1C'],sequences=[mt]))
    assert retained.selected_indices==[] and retained.exclusions[0]['reason']=='unscored_mutation'


def response_cli():
    path=Path(__file__).resolve().parents[2]/'scripts/capability/extensions/analyse_residue_responses.py'
    spec=importlib.util.spec_from_file_location('response_cli_test',path)
    assert spec is not None and spec.loader is not None
    cli=importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    return cli


def test_cli_projects_mixed_rows_into_real_shared_fits(tmp_path):
    cli=response_cli(); rows=[]
    for f in range(5):
        data,states,identity=mixed_archive()
        np.savez(tmp_path/f'a{f}.npz',**data)
        (tmp_path/f'p{f}.json').write_text(json.dumps(states))
        rows.append(dict(assay=f'a{f}',cluster=f'f{f}',protein=f'p{f}',**identity,
                         archive=f'a{f}.npz',packing=f'p{f}.json',B=[[i] for i in range(5)],measured=[1.,9.,8.,3.,5.]))
    path=tmp_path/'input.json'; path.write_text(json.dumps(dict(assays=rows)))
    result=cli.main(['--mode','bins','--input',str(path),'--out',str(tmp_path/'fit.json'),'--seeds','3','--draws','10'])
    assert len(result['assays'])==5 and result['folds_identical']
    assert all(r['original_variant_indices']==[0,3,4] for r in result['response_coverage'])
    assert all(r['original_variant_indices']==[0,3,4] for r in result['archive_admission'])
    path.write_text(json.dumps(dict(assays=rows[:1])))
    result=cli.main(['--mode','bins','--input',str(path),'--out',str(tmp_path/'small.json')])
    assert result['inference_status']=='insufficient assay/family support after biological exclusions'


def test_tokenizer_only_export_uses_original_packer_without_forward(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from src.capability.readouts import readout_extraction as extraction
    cli=response_cli(); wild='ACD'; sequence='AAD'
    class Tokenizer:
        def decode(self,ids,**kwargs): return ''.join(chr(i) for i in ids)
    arm=SimpleNamespace(model=None,tokenizer=Tokenizer(),name='test')
    loads=[]
    def load(name,stage,*,device=None):
        loads.append(device); return arm
    monkeypatch.setattr(extraction,'load_readout_arm',load)
    from scripts.capability.position import analyse_position_terms as original
    pack=lambda a,s: ([0,*map(ord,s)],(1,4),(1,4))
    monkeypatch.setattr(extraction,'pack_sequence',pack)
    monkeypatch.setattr(original,'pack_sequence',pack)
    nll=np.ones(3,dtype=np.float32)
    data=dict(position_nats=np.r_[nll,nll],position_offsets=np.array([0,3,6]),position_residue_counts=np.ones(6,dtype=int),
              position_residue_offsets=np.zeros(2,dtype=int),position_sum_check_nats=0.,mutants=np.array(['C2A']),
              wt_likelihood=-3.,likelihood=np.array([0.]))
    np.savez(tmp_path/'archive.npz',**data)
    source=tmp_path/'input.json'
    source.write_text(json.dumps(dict(assays=[dict(assay='a',cluster='f',wildtype=wild,mutants=['C2A'],sequences=[sequence],archive='archive.npz')])))
    out=tmp_path/'packed-input.json'
    result=cli.main(['--mode','packing','--input',str(source),'--arm','test','--out',str(out)])
    assert loads==[None] and not result['model_loaded'] and not result['forward_performed']
    assert result['packing_receipts'][0]['selected_indices']==[0]
    row=result['assays'][0]; states=json.loads(Path(row['packing']).read_text())
    with np.load(row['archive']) as retained:
        assert RetainedResponses(retained,states,dict(wildtype=wild,mutants=['C2A'],sequences=[sequence])).selected_indices==[0]
