"""Paired measurement must retain variant-specific WT references and exact prefixes."""
import numpy as np
import pytest
from src.capability.position.paired_position import validate_pair_arrays
from src.capability.position.position_terms import mutation_parts,state_parts


def fixture():
    cohort=dict(wildtype='MAL',mutants=['A2C','L3K'],sequences=['MCL','MAK'])
    tokens={'MAL':[9,1,2,3],'MCL':[9,1,7,3],'MAK':[9,1,2,8]}
    class Packer:
        def state(self,sequence):
            return dict(ids=tokens[sequence],span=(1,4),counts=np.ones(3,dtype=np.int16),offset=0)
    wt=np.array([[1,2,3],[1,4,3]],dtype=np.float32)
    mt=np.array([[1,5,7],[1,4,8]],dtype=np.float32)
    nll=np.array([[6,13],[8,13]],dtype=float)
    parts=[mutation_parts(state_parts(w,np.ones(3),0,site),state_parts(m,np.ones(3),0,site),float(n[0]-n[1]))
           for w,m,n,site in zip(wt,mt,nll,[1,2])]
    data={k:np.asarray([p[k] for p in parts]) for k in parts[0]}
    data.update(variant_indices=np.array([0,1]),mutants=np.asarray(cohort['mutants']),
                wild_ids=np.array(tokens['MAL']),mutant_ids=np.array([tokens['MCL'],tokens['MAK']]),
                residue_counts=np.ones(3,dtype=np.int16),residue_offset=np.array(0),scored_span=np.array([1,4]),
                wild_position_nats=wt,mutant_position_nats=mt,nll=nll)
    return data,cohort,Packer()


def test_variant_specific_wildtype_references_are_not_collapsed():
    data,cohort,packer=fixture()
    indices,parts=validate_pair_arrays(data,cohort,packer)
    assert indices==[0,1]
    assert [p['full'] for p in parts]==[-7.,-5.]
    # Reusing the first WT reference changes the second contrast and must fail.
    data['nll'][1,0]=data['nll'][0,0]
    with pytest.raises(ValueError,match='full reconstruction'):
        validate_pair_arrays(data,cohort,packer)


def test_pair_admission_checks_individual_prefix_terms_and_component_identity():
    data,cohort,packer=fixture()
    data['mutant_position_nats'][0,0]+=0.01
    with pytest.raises(ValueError,match='prefix'):
        validate_pair_arrays(data,cohort,packer)
    data,cohort,packer=fixture()
    data['own'][0]+=1.
    with pytest.raises(ValueError,match='own reconstruction'):
        validate_pair_arrays(data,cohort,packer)
