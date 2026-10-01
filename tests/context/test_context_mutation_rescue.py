"""Invariants for the context rescue design; no checkpoint assets required."""
import importlib.util
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[2] / 'scripts/capability/context/context_mutation_rescue.py'
spec = importlib.util.spec_from_file_location('rescue', PATH)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def make_row():
    return dict(wildtype='ACDEFG', sequences=['ACDEFA'],
        homolog_candidates=[dict(sequence='ACDEFA', subject='h', identity=70)],
        unrelated_candidates=[dict(sequence='VVVVVV', subject='u')])


def test_paired_context_tokens_and_whole_target_budget():
    with patch.object(r.ch, 'item_ids', side_effect=lambda arm,s,modality: list(s)), patch.object(r.ch, 'row_prefix_ids', return_value=[0]):
        plan, reason = r.plan_row(None, make_row(), 13)
        assert reason is None
        assert len(plan['context_ids']['homolog']) == len(plan['context_ids']['unrelated']) == 6
        assert plan['context_ids']['no_context'] == []
        assert r.plan_row(None, make_row(), 12)[0] is None


def test_no_mismatch_or_local_copy_can_enter_control():
    with patch.object(r.ch, 'item_ids', side_effect=lambda arm,s,modality: list(s)), patch.object(r.ch, 'row_prefix_ids', return_value=[0]):
        row = make_row()
        row['unrelated_candidates'][0]['sequence'] = 'VVVVV'
        assert r.plan_row(None, row, 30)[0] is None
        row = make_row()
        row['wildtype'] = 'ACDEFGHIKL'
        row['homolog_candidates'][0]['sequence'] = 'ACDEFGHIKL'
        row['unrelated_candidates'][0]['sequence'] = 'ACDEFGHIKL'
        assert r.plan_row(None, row, 30)[0] is None


def test_undefined_correlation_is_explicit():
    assert r.rho([1,1,1], [1,2,3]) is None
    assert r.rho([3,2,1], [1,2,3]) == -1
