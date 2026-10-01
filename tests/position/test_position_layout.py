"""Label-blind layout selection and shared-support uncertainty invariants."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest


def module(name):
    from scripts.capability.entrypoints import stage_path
    path=stage_path(Path(__file__).resolve().parents[2], f'{name}.py')
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_layout_selection_ignores_labels_and_covers_position_strata():
    m=module('assess_position_layout')
    row=dict(assay='a',cluster='f',wildtype='A'*100,mutants=[f'A{i}C' for i in range(1,101)],measured=list(range(100)))
    cohort=dict(assays=[row]);support=dict(assays=['a'])
    expected=m.selection(cohort,support)
    row['measured']=list(reversed(row['measured']))
    assert m.selection(cohort,support)==expected
    chosen=expected[0]['indices'];assert len(chosen)==16
    assert [sum(i//25==q for i in chosen) for q in range(4)]==[4]*4


def test_simultaneous_bands_preserve_shared_draws_and_missing_support():
    m=module('position_simultaneous');rng=np.random.default_rng(7)
    x=rng.normal(size=20);values=np.c_[x,x,np.zeros(20)];values[:2,:2]=np.nan
    result=m.bands(values,draws=200)
    assert result['available_families']==[18,18,20]
    assert result['interval'][0]==result['interval'][1]
    assert result['point'][0]==pytest.approx(np.nanmean(x[2:]))
    assert result['interval'][2]==[0,0]
    assert result['interval'][0][0]<result['point'][0]<result['interval'][0][1]
    with pytest.raises(ValueError,match='fewer than8'):m.bands(values[:7])
