from src.capability.generation.generation_failure import decompose, summarize


def test_censoring_is_not_format_failure_or_completeness():
    r=decompose('ACDEFG','prollama',residue_budget=4,stop_reason='residue_budget')
    assert r['format_valid'] and not r['native_complete']
    assert r['residue_budget_overshoot']==2


def test_native_endpoints_and_format_are_separate():
    assert decompose('ACDE2<eos>','progen3-112m')['native_complete']
    bad=decompose('ACDE<eos>','progen3-112m')
    assert bad['model_eos_observed'] and bad['format_failure'] and not bad['native_complete']
    complete=decompose(' AC D >','prollama')
    assert complete['native_complete'] and not complete['model_eos_observed']
    assert decompose('','prollama')['empty_output']
    assert decompose('Sorry','prollama')['format_failure']


def test_unknown_profiles_not_zero():
    r=decompose('ACDE>','prollama')
    s=summarize([r])
    assert not s['profiles_available'] and 'profile_groups' not in s
