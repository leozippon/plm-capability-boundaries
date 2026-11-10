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


def test_a_constant_column_does_not_set_the_critical_value():
    """0.3 is a constant, not a measurement of infinite precision.

    A column every draw gives the same value carries nothing about how far the
    family's estimates can move, so it must not decide the family maximum. An
    exactly representable constant always had a zero standard error and was
    dropped; one that is not exactly representable had a rounding-residue
    standard error instead, and dividing a rounding-residue deviation by it put
    noise of order one into the family maximum -- 3.87 here where the three
    varying columns give 2.27, and 13.79 against a correct 2.48 on the 163-family
    fixture the 2026-11-10 band audit measured.
    """
    m=module('position_simultaneous');rng=np.random.default_rng(20261110)
    varying=rng.normal(0.0,0.05,(16,3));free=m.bands(varying,draws=2000)
    for constant in (0.3,0.1,0.0):
        held=m.bands(np.column_stack([varying,np.full(16,constant)]),draws=2000)
        assert held['critical_value']==pytest.approx(free['critical_value'],rel=1e-12)
        assert held['degenerate_columns']==1
        assert held['interval'][3]==pytest.approx([constant,constant],abs=1e-15)
    assert free['degenerate_columns']==0
    # No column to studentise is an ill-posed family, not a zero critical value
    # that reports every column as resolved exactly.
    with pytest.raises(ValueError,match='no studentised maximum'):m.bands(np.full((16,2),0.3),draws=200)
