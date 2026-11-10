"""The three lane adapters: what each reads off its own archive, and what it refuses.

These tests deliberately do not fit anything. A real fit of any of these lanes needs
the restored per-arm extractions and runs for minutes to hours per arm, and what can
go wrong at this boundary is not the arithmetic -- that is the pilot's job -- but the
reading: an archive whose ``matched_baseline`` has a different shape in two lanes, a
control set that is a property of the arm rather than the stratum, a join key that
names one corner of a four-corner cycle, and a support whose every group holds one
background so that a careless pilot comparison agrees by construction.

The pilot reproductions themselves are run against the real archives and recorded in
each lane's ``pilot.json``; they are not reproduced here, because a test that refits
a 109,568-row panel is not a test.
"""
import json
import pathlib

import pytest

from scripts.capability.extensions import explicit_statistics_lane_adapters as adapters
from src.capability.extensions import explicit_statistics_lanes as lanes
from src.capability.interactions import pairwise_epistasis as pairwise


# --------------------------------------------------------------------------- #
# The adapter boundary.
# --------------------------------------------------------------------------- #

BOUNDARY = ('inputs', 'arms', 'strata', 'qualified', 'state', 'fit', 'archived')


def test_every_lane_has_an_adapter_exposing_the_whole_boundary():
    assert sorted(adapters.ADAPTERS) == sorted(lanes.LANES)
    for lane, adapter in adapters.ADAPTERS.items():
        assert adapter.lane == lane
        for name in BOUNDARY:
            assert callable(getattr(adapter, name)), (lane, name)


def test_the_qualified_control_set_is_asked_per_arm_in_every_lane():
    import inspect

    for lane, adapter in adapters.ADAPTERS.items():
        parameters = list(inspect.signature(adapter.qualified).parameters)
        assert parameters[:3] == ['self', 'stratum', 'arm'], lane
        assert inspect.signature(adapter.qualified).parameters['arm'].default is None, lane


def test_a_lane_that_names_two_score_directories_is_refused():
    with pytest.raises(SystemExit) as error:
        adapters._one(['a', 'b'], 'scores', 'abundance')
    assert 'names 2 directories' in str(error.value)
    with pytest.raises(SystemExit):
        adapters._one([], 'scores', 'abundance')
    assert adapters._one(['a'], 'scores', 'abundance').name == 'a'


def test_a_missing_lane_input_is_refused_by_name():
    with pytest.raises(SystemExit) as error:
        adapters._require(None, 'stability', 'the frozen stability gate directory')
    assert 'the frozen stability gate directory' in str(error.value)


# --------------------------------------------------------------------------- #
# What each lane's hierarchy is, over its own archived control set.
# --------------------------------------------------------------------------- #

def test_the_stability_profile_rung_is_the_archived_s2_set_column_for_column():
    columns = adapters.StabilityAdapter.ARCHIVED_CONTROL_SET
    assert lanes.control_blocks('stability', lanes.ARCHIVED_RUNG, columns) == columns


def test_the_stability_nuisance_rung_drops_profile_and_keeps_composition():
    columns = adapters.StabilityAdapter.ARCHIVED_CONTROL_SET
    nuisance = lanes.control_blocks('stability', 'nuisance', columns)
    assert nuisance == ('ident', 'geom', 'chem', 'G', 'comp')
    # The archived S baseline drops composition as well as profile, so it is not
    # this hierarchy's nuisance rung and the two must not be confused.
    assert nuisance != ('ident', 'geom', 'chem', 'G')


def test_an_abundance_arm_carries_its_own_tokenisation_block_in_its_nuisance_set():
    frozen = ('ident', 'geom', 'comp', 'chem', 'prof', 'prof2', 'G')
    plain = lanes.role_blocks('abundance', frozen)
    tokenised = lanes.role_blocks('abundance', (*frozen, 'T'))
    assert 'T' not in plain['nuisance']
    assert tokenised['nuisance'] == ('ident', 'geom', 'comp', 'chem', 'G', 'T')
    assert tokenised['profile'] == ('prof', 'prof2')
    # The profile rung stays the archived design, column for column and in order.
    assert lanes.control_blocks('abundance', lanes.ARCHIVED_RUNG,
                                (*frozen, 'T')) == (*frozen, 'T')
    assert lanes.control_blocks('abundance', lanes.ARCHIVED_RUNG, frozen) == frozen


def test_the_interaction_control_set_carries_the_tokenisation_block_for_every_arm():
    columns = (*pairwise.C_BLOCKS, 'G', 'T')
    assert lanes.control_blocks('interaction', lanes.ARCHIVED_RUNG, columns) == columns
    assert lanes.control_blocks('interaction', 'nuisance', columns) == (
        'ident', 'geom', 'comp', 'local', 'G', 'T')
    # GEMME is non-identifiable here, so the hierarchy stops at profile and couplings.
    assert lanes.lane('interaction').rungs == (
        'nuisance', 'nuisance+profile', 'nuisance+profile+couplings')


# --------------------------------------------------------------------------- #
# The stability archive's own shape.
# --------------------------------------------------------------------------- #

def stability_adapter(tmp_path, record):
    instance = adapters.StabilityAdapter.__new__(adapters.StabilityAdapter)
    instance.published = tmp_path
    (tmp_path / 'fits').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'fits' / 'fit_gpt2.json').write_text(json.dumps(record))
    return instance


def archived_record(**overrides):
    record = {'arm': 'gpt2', 'matched_baseline': 'S',
              'tokenisation_verdict': {'qualified': False},
              'held_out_predictions': {'designs': ['S', 'S2', 'S2_M', 'S_M', 'S_T']}}
    record.update(overrides)
    return record


def test_the_stability_archive_records_its_baseline_as_a_bare_string(tmp_path):
    instance = stability_adapter(tmp_path, archived_record())
    assert instance.qualified('ranking', 'gpt2') == \
        adapters.StabilityAdapter.ARCHIVED_CONTROL_SET
    assert instance.qualified('ranking') == adapters.StabilityAdapter.ARCHIVED_CONTROL_SET


def test_a_nested_matched_baseline_is_refused_rather_than_indexed_as_a_string(tmp_path):
    # The abundance archive nests this field; indexing a bare string the same way
    # silently takes its first character, which would read 'S_T' as 'S'.
    instance = stability_adapter(tmp_path, archived_record(matched_baseline={'primary': 'S'}))
    with pytest.raises(SystemExit) as error:
        instance.qualified('ranking', 'gpt2')
    assert 'not the bare string this lane records' in str(error.value)


def test_a_tokenisation_qualifying_stability_arm_is_refused_not_fitted(tmp_path):
    instance = stability_adapter(tmp_path, archived_record(
        matched_baseline='S_T', tokenisation_verdict={'qualified': True}))
    with pytest.raises(SystemExit) as error:
        instance.qualified('ranking', 'gpt2')
    assert 'no S_T_M joint arm' in str(error.value)


def test_a_stability_archive_without_the_reused_designs_is_refused(tmp_path):
    instance = stability_adapter(tmp_path, archived_record(
        held_out_predictions={'designs': ['S', 'S_M', 'S_T']}))
    with pytest.raises(SystemExit) as error:
        instance.qualified('ranking', 'gpt2')
    assert "'S2'" in str(error.value) and "'S2_M'" in str(error.value)


def test_the_stability_strata_read_the_published_control_and_its_contrast():
    # Three readings, each naming the published control vector, the published
    # contrast and the sign that turns the contrast back into the joint arm.
    strata = adapters.StabilityAdapter.STRATA
    assert sorted(strata) == sorted(lanes.lane('stability').strata)
    assert strata['ranking'][2] == 1.
    assert strata['transfer_error'][2] == -1. and strata['calibrated_error'][2] == -1.
    for stratum, (baseline, contrast, _) in strata.items():
        assert baseline.startswith('baseline_') and contrast.startswith('profile_')


# --------------------------------------------------------------------------- #
# The interaction join, and what a pilot comparison exercises.
# --------------------------------------------------------------------------- #

def test_the_interaction_join_key_names_the_background_and_the_cycle():
    import numpy as np

    panel = {'background': np.asarray(['1ENH.pdb', '1ENH.pdb', '2D1U.pdb']),
             'cycle_index': np.asarray([0, 1, 0])}
    # A cycle is four measured corners rather than one mutation, so the index alone
    # would collide across backgrounds.
    assert adapters.InteractionAdapter._cycle_keys(panel) == [
        '1ENH.pdb|0', '1ENH.pdb|1', '2D1U.pdb|0']


def test_a_support_with_one_background_per_group_cannot_exercise_the_aggregation():
    shape = adapters._backgrounds_per_group(['a', 'b', 'c'], ['g1', 'g2', 'g3'])
    assert shape == {'groups': 3, 'backgrounds': 3, 'most_backgrounds_in_one_group': 1,
                     'exercises_aggregation': False,
                     'reading': adapters.ONE_BACKGROUND_PER_GROUP}


def test_a_support_with_several_backgrounds_in_a_group_does_exercise_it():
    shape = adapters._backgrounds_per_group(['a', 'b', 'c'], ['g1', 'g1', 'g2'])
    assert shape['most_backgrounds_in_one_group'] == 2
    assert shape['exercises_aggregation'] is True
    assert 'the whole published aggregation' in shape['reading']


# --------------------------------------------------------------------------- #
# One fit, several strata.
# --------------------------------------------------------------------------- #

def test_one_fit_serves_every_stratum_of_a_lane():
    cache = adapters._Readings()
    state = {'support_sha256': 'abc'}
    designs = {lanes.CONTROL_ARM: ('ident', 'G'), lanes.MODEL_ARM: ('M',),
               lanes.JOINT_ARM: ('ident', 'G', 'M')}
    assert cache.get(state, 'gpt2', designs, 'ranking') is None
    cache.put(state, 'gpt2', designs,
              readings={'ranking': {'U_C': [1.]}, 'transfer_error': {'U_C': [2.]}},
              alphas={'1': {'U_C': [1.]}}, dimensions={'U_C': 2})
    assert cache.get(state, 'gpt2', designs, 'ranking')['values'] == {'U_C': [1.]}
    assert cache.get(state, 'gpt2', designs, 'transfer_error')['values'] == {'U_C': [2.]}
    assert cache.get(state, 'gpt2', designs, 'ranking')['readings_available'] == [
        'ranking', 'transfer_error']
    # A different control set, a different arm or a different support is a different
    # fit and must not be served from this cell.
    other = dict(designs, **{lanes.CONTROL_ARM: ('ident', 'G', 'prof')})
    assert cache.get(state, 'gpt2', other, 'ranking') is None
    assert cache.get(state, 'gpt2-xl', designs, 'ranking') is None
    assert cache.get({'support_sha256': 'def'}, 'gpt2', designs, 'ranking') is None


def test_the_nuisance_rung_leak_through_g_is_recorded_rather_than_removed():
    text = adapters.NUISANCE_RUNG_LIMITATION
    assert 'G' in text and 'profile' in text
    assert 'plus the profile columns' in text


# --------------------------------------------------------------------------- #
# The export join each lane depends on.
# --------------------------------------------------------------------------- #

EXPORTS = {
    'abundance': 'results/extensions/explicit_statistics_abundance_20261109',
    'pairwise': 'results/extensions/explicit_statistics_pairwise_20261109',
}


@pytest.mark.parametrize('cohort', sorted(EXPORTS))
def test_the_export_names_a_row_by_its_assay_and_its_mutation(cohort):
    from src.capability.baselines.cohorts import SAMPLE_SEPARATOR

    document = json.loads((pathlib.Path(EXPORTS[cohort]) / 'inputs'
                           / 'variant_table.json').read_text())
    # The adapters join the coupling and GEMME tables on this key and on nothing
    # else, so the shape of it is a contract rather than an incidental detail.
    for row in document['samples'][:200]:
        assert row['sample_id'] == f"{row['assay']}{SAMPLE_SEPARATOR}{row['mutation']}"
    multi = [row for row in document['samples'] if row['n_substitutions'] > 1]
    if multi:
        assert ':' in multi[0]['mutation']


def test_the_abundance_export_is_keyed_by_domain_and_refused_by_reference():
    document = json.loads((pathlib.Path(EXPORTS['abundance']) / 'inputs'
                           / 'variant_table.json').read_text())
    shared = [row for row in document['samples'] if row['assay'] != row['wildtype_id']]
    # Two Domainome domains are byte-identical in sequence, so the export carries one
    # reference for both. The join is on the assay and the loss on the reference.
    assert shared, 'the collapsed reference is what makes the two identities distinct'
    assert {row['assay'] for row in shared} != {row['wildtype_id'] for row in shared}
    assert all(row['sample_id'].startswith(row['assay']) for row in shared)


# --------------------------------------------------------------------------- #
# The interaction lane's cycle corners.
# --------------------------------------------------------------------------- #

def interaction_adapter(tmp_path, monkeypatch, *, states=()):
    import types

    plan = {'backgrounds': [{'name': 'X', 'group': 'g1', 'wildtype': 'ACDE',
                             'sequences': ['ACDE', 'AWDE', 'ACDW', 'AWDW']}],
            'supports': {'all_groups': {'groups': ['g1']}}}
    declaration = {'schema': 'pairwise_indel_exclusion_v1', 'states': list(states)}
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    (tmp_path / 'exclusion.json').write_text(json.dumps(declaration))
    instance = adapters.InteractionAdapter.__new__(adapters.InteractionAdapter)
    instance.lane = 'interaction'
    monkeypatch.setattr(instance, '_module', lambda: types.SimpleNamespace(
        INPUTS={'plan': tmp_path / 'plan.json'}, EXCLUSION=tmp_path / 'exclusion.json'))
    return instance


def toy_panel(cycle, kept=4):
    import numpy as np

    return {'states': {'background': np.asarray(['X'] * kept)},
            'cycle_states': np.asarray([cycle])}


def test_a_cycles_corners_are_named_by_diffing_each_state_against_its_wild_type(
        tmp_path, monkeypatch):
    instance = interaction_adapter(tmp_path, monkeypatch)
    # The export scores mutants and a row here is a cycle, so the block this lane
    # joins is v(AB) - v(A) - v(B) over these three keys.
    assert instance._corner_keys(toy_panel([0, 1, 2, 3]), None) == [
        ('X::C2W:E4W', 'X::C2W', 'X::E4W')]


def test_a_cycle_whose_reference_corner_is_mutated_is_refused(tmp_path, monkeypatch):
    instance = interaction_adapter(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as error:
        instance._corner_keys(toy_panel([1, 0, 2, 3]), None)
    assert 'wild-type corner carries' in str(error.value)


def test_a_state_table_that_does_not_line_up_with_the_panel_is_refused(tmp_path, monkeypatch):
    # The indel exclusion removed a state, so the adapter's own reading of the mask
    # must agree with the panel's state table or every corner would shift by a row.
    instance = interaction_adapter(tmp_path, monkeypatch, states=[
        {'background': 'X', 'sequence': 'AWDE', 'value_without_indel_rows_kcal_mol': None}])
    with pytest.raises(SystemExit) as error:
        instance._corner_keys(toy_panel([0, 1, 2, 3], kept=4), None)
    assert 'do not line up' in str(error.value)
