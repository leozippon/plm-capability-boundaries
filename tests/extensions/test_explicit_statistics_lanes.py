"""The lane hierarchy, its families, its matched support and its four finding states.

Every fixture is synthetic, so these tests need neither the coupling wave nor the
GEMME wave. One test reads the real producer output where it is present, because the
refusal on a partial wave is the behaviour that matters most while that wave runs.
"""
import dataclasses
import json
import pathlib
from pathlib import Path

import numpy as np
import pytest

from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS
from src.capability.extensions import bidirectional
from src.capability.extensions import explicit_statistics_lanes as lanes
from src.capability.extensions import overlap

ROOT = Path(__file__).resolve().parents[2]

#: Each lane's archived qualified control set, as its own archive records it. These are
#: inputs to the declaration, never re-qualified by it.
QUALIFIED = {
    'abundance': ('ident', 'geom', 'comp', 'chem', 'prof', 'prof2', 'G'),
    'stability': ('ident', 'geom', 'chem', 'G', 'comp', 'prof2'),
    'interaction': ('ident', 'geom', 'comp', 'local', 'prof', 'G', 'T'),
    'phenotype': ('ident', 'geom', 'prof', 'prof2'),
}
#: Only the abundance lane appends its tokenisation nuisance per arm here; the
#: interaction lane's archived control set already carries T for every arm.
TOKENISATION = {'abundance': True, 'stability': False, 'interaction': False,
                'phenotype': False}


def columns(lane, rung):
    return lanes.control_blocks(lane, rung, QUALIFIED[lane], tokenisation=TOKENISATION[lane])


def cell_matrix(lane, stratum, groups, *, forward=0.05, reverse=0.4, spread=0.005, seed=5,
                digest='a' * 64, arms=None, rungs=None, exact_forward=False, level=None):
    """Synthetic per-group statistics with a declared forward and reverse increment.

    The cells carry the stratum's own per-group statistic in its own direction -- an
    error level where the stratum reads an error, a positive rank agreement where it
    reads one -- which the panel converts to a loss with the stratum's declared sign.
    The joint arm is the reference; the control arm is ``forward`` worse and the pure
    readout ``reverse`` worse, in loss units, so both directions have a known sign
    whatever the stratum reads. ``exact_forward`` lets the control arm share the joint
    arm's per-group draw, which fixes the forward reduction at exactly ``forward``.
    """
    declared = lanes.lane(lane)
    sign = lanes.stratum(lane, stratum).to_loss
    level = (1.0 if sign > 0 else 0.6) if level is None else level
    rng = np.random.default_rng(seed)
    cells = {}
    for rung in (rungs if rungs is not None else declared.rungs):
        for arm in (arms if arms is not None else declared.arms):
            joint = rng.normal(level, spread, size=len(groups))
            noise = 0. if exact_forward else rng.normal(0, spread, len(groups))
            cells[rung, arm] = {
                'U_C': joint + sign * (forward + noise),
                'M': joint + sign * (reverse + rng.normal(0, spread, len(groups))),
                'U_C_M': joint,
                'support_sha256': digest}
    return cells


def score_table(role, rows, *, statuses=None, complete=True, refused=None, column_value=None):
    """A minimal upstream score table in the producer's own shape."""
    source = lanes.BLOCK_SOURCE[role]
    statuses = statuses or {}
    samples = []
    for index, row in enumerate(rows):
        status = statuses.get(row['sample_id'], 'fitted' if role == 'couplings' else 'scored')
        available = lanes.ROW_STATUS[role].get(status) == lanes.AVAILABLE
        value = (column_value if column_value is not None
                 else (float(index) + 0.5) if available else None)
        samples.append({'sample_id': row['sample_id'], 'assay': row['assay'],
                        'wildtype_id': row['wildtype_id'], 'mutation': row['mutation'],
                        'n_substitutions': 1, 'status': status, source['column']: value})
    return {'schema': source['schema'], 'counts': {'complete': complete, 'samples': len(samples)},
            'refused': dict(refused or {}), 'samples': samples, 'read_from': f'<{role}>'}


def cohort_rows(references=('r1', 'r2', 'r3'), per_reference=4):
    rows = []
    for reference in references:
        for index in range(per_reference):
            rows.append({'sample_id': f'{reference}::A{index + 1}C', 'assay': reference,
                         'wildtype_id': reference, 'mutation': f'A{index + 1}C',
                         'original_index': len(rows), 'n_substitutions': 1})
    return rows


# --------------------------------------------------------------------------- #
# The role vocabulary and the collision it makes impossible.
# --------------------------------------------------------------------------- #

def test_roles_are_declared_once_and_the_added_keys_collide_with_no_lane_block():
    assert lanes.ROLES == ('nuisance', 'profile', 'couplings', 'gemme')
    for lane in lanes.LANES:
        roles = lanes.role_blocks(lane, QUALIFIED[lane], tokenisation=TOKENISATION[lane])
        assert set(roles) == set(lanes.ROLES)
        assert set(QUALIFIED[lane]) & {lanes.COUPLINGS_BLOCK, lanes.GEMME_BLOCK} == set()
        assert roles['couplings'] == (lanes.COUPLINGS_BLOCK,)
        assert roles['gemme'] == (lanes.GEMME_BLOCK,)


def test_the_letter_g_means_different_things_in_different_lanes():
    # The nonlinear-additive nuisance is spelled G in these lanes; GEMME is a role, and
    # its block carries a key that cannot be mistaken for that nuisance.
    for lane in ('abundance', 'stability', 'interaction'):
        roles = lanes.role_blocks(lane, QUALIFIED[lane], tokenisation=TOKENISATION[lane])
        assert 'G' in roles['nuisance']
        assert 'G' not in roles['gemme'] and roles['gemme'] == (lanes.GEMME_BLOCK,)


def test_role_binding_refuses_a_set_that_already_carries_an_added_block():
    with pytest.raises(ValueError, match='must not collide'):
        lanes.role_blocks('phenotype', (*QUALIFIED['phenotype'], lanes.COUPLINGS_BLOCK))
    with pytest.raises(ValueError, match='repeats a block'):
        lanes.role_blocks('phenotype', (*QUALIFIED['phenotype'], 'prof'))
    with pytest.raises(ValueError, match='no tokenisation nuisance'):
        lanes.role_blocks('phenotype', QUALIFIED['phenotype'], tokenisation=True)
    with pytest.raises(ValueError, match='unknown lane'):
        lanes.role_blocks('variant_effect', ('P',))


def test_a_lane_whose_control_set_is_only_profile_has_no_nuisance_rung():
    with pytest.raises(ValueError, match='nuisance rung would be empty'):
        lanes.role_blocks('phenotype', ('prof', 'prof2'))


# --------------------------------------------------------------------------- #
# One hierarchy, identical in every lane.
# --------------------------------------------------------------------------- #

def test_the_hierarchy_is_one_declaration_of_five_nested_rungs():
    assert list(lanes.RUNGS) == ['nuisance', 'nuisance+profile', 'nuisance+profile+couplings',
                                 'nuisance+profile+gemme', 'nuisance+profile+couplings+gemme']
    for lane in lanes.LANES:
        nuisance = set(columns(lane, 'nuisance'))
        for rung in lanes.lane(lane).rungs:
            assert nuisance <= set(columns(lane, rung))
        assert set(columns(lane, 'nuisance+profile')) - nuisance == \
            set(lanes.lane(lane).profile_blocks)


def test_the_profile_rung_is_the_archived_control_set_column_for_column():
    # This is what lets a lane reuse its archived control and joint fits: the rung is
    # the archived design matrix, not a reordering of it.
    assert columns('abundance', 'nuisance+profile') == (*QUALIFIED['abundance'], 'T')
    assert columns('stability', 'nuisance+profile') == QUALIFIED['stability']
    assert columns('interaction', 'nuisance+profile') == QUALIFIED['interaction']
    assert columns('phenotype', 'nuisance+profile') == QUALIFIED['phenotype']
    assert lanes.ARCHIVED_RUNG == 'nuisance+profile'


def test_the_nuisance_rung_is_the_archived_set_without_profile():
    assert columns('abundance', 'nuisance') == ('ident', 'geom', 'comp', 'chem', 'G', 'T')
    # The stability archive's own S baseline also drops composition, so it is not this
    # hierarchy's nuisance rung and the first step stays a profile increment.
    assert columns('stability', 'nuisance') == ('ident', 'geom', 'chem', 'G', 'comp')
    assert columns('stability', 'nuisance') != ('ident', 'geom', 'chem', 'G')
    assert columns('interaction', 'nuisance') == ('ident', 'geom', 'comp', 'local', 'G', 'T')


def test_profile_is_carried_even_where_the_archived_qualification_dropped_it():
    # The binding quantitative cell qualified no profile block. The hierarchy carries it
    # by declaration, appended rather than inserted, so the archived order survives.
    qualified = ('ident', 'geom', 'comp')
    assert lanes.control_blocks('phenotype', 'nuisance', qualified) == qualified
    assert lanes.control_blocks('phenotype', 'nuisance+profile', qualified) == \
        (*qualified, 'prof', 'prof2')


def test_the_added_blocks_are_appended_in_the_declared_order():
    for lane in lanes.LANES:
        for rung, roles in lanes.RUNGS.items():
            if rung not in lanes.lane(lane).rungs:
                continue
            tail = columns(lane, rung)[len(columns(lane, 'nuisance+profile')):]
            expected = tuple(lanes.NEW_BLOCKS[role] for role in ('couplings', 'gemme')
                             if role in roles)
            assert tail == expected


def test_arms_are_the_shared_directions_three_arms_and_m_is_one_column():
    designs = lanes.arm_designs('abundance', 'nuisance+profile+couplings+gemme',
                                QUALIFIED['abundance'], tokenisation=True)
    assert set(designs) == set(lanes.ARM_NAMES)
    assert designs[lanes.MODEL_ARM] == ('M',)
    assert designs[lanes.JOINT_ARM] == (*designs[lanes.CONTROL_ARM], 'M')
    representation = lanes.arm_designs('abundance', 'nuisance', QUALIFIED['abundance'],
                                       channel='representation', tokenisation=True)
    assert representation[lanes.MODEL_ARM] == ('R',)
    with pytest.raises(ValueError, match='no .representation. channel'):
        lanes.arm_designs('phenotype', 'nuisance', QUALIFIED['phenotype'],
                          channel='representation')


def test_fit_plan_names_the_hierarchy_the_sensitivity_and_the_pilot():
    plan = lanes.fit_plan('stability', QUALIFIED['stability'])
    primary = [entry for entry in plan if entry['role'] == 'primary']
    sensitivity = [entry for entry in plan if entry['role'] == 'sensitivity']
    pilot = [entry for entry in plan if entry['role'] == 'pilot']
    assert [entry['rung'] for entry in primary] == list(lanes.lane('stability').rungs)
    assert [entry['support'] for entry in primary] == ['complete_case'] * 5
    assert len(sensitivity) == 1 and sensitivity[0]['rung'] == 'nuisance'
    assert sensitivity[0]['support'] == 'full'
    assert len(pilot) == 1 and pilot[0]['rung'] == lanes.ARCHIVED_RUNG
    assert pilot[0]['support'] == 'full' and pilot[0]['compared_against_the_archive'] is True
    # No archived fit is reused inside the family: the complete case is a different
    # row set, however exactly the profile rung's columns match the archive.
    assert all(entry['reuses_archived_fit'] is False for entry in plan)
    assert [entry['rung'] for entry in plan
            if entry['reproduces_archived_control_set']] == [lanes.ARCHIVED_RUNG] * 2
    assert 'not reusable inside the complete-case family' in lanes.ARCHIVED_REUSE_RULE
    assert lanes.contract('stability')['archived_reuse_rule'] == lanes.ARCHIVED_REUSE_RULE


# --------------------------------------------------------------------------- #
# GEMME is non-identifiable for the interaction lane.
# --------------------------------------------------------------------------- #

def test_gemme_is_recorded_as_non_identifiable_for_the_cycle_estimand():
    lane = lanes.lane('interaction')
    assert set(lane.non_identifiable) == {'gemme'}
    assert lane.rungs == ('nuisance', 'nuisance+profile', 'nuisance+profile+couplings')
    assert lane.unidentifiable_rungs == ('nuisance+profile+gemme',
                                         'nuisance+profile+couplings+gemme')
    assert lane.required_blocks() == ('couplings',)
    assert 'sum of its per-site scores' in lane.non_identifiable['gemme']
    for rung in lane.unidentifiable_rungs:
        with pytest.raises(ValueError, match='never estimated'):
            lanes.control_blocks('interaction', rung, QUALIFIED['interaction'])


def test_a_non_identifiable_rung_is_recorded_without_an_estimate():
    groups = [f'g{index:02d}' for index in range(14)]
    cells = cell_matrix('interaction', 'cycle_squared_error', groups)
    record = lanes.panel('interaction', 'cycle_squared_error', 'likelihood', cells,
                         groups=groups, support_sha256='a' * 64)
    assert record['family']['size'] == 198
    assert len(record['columns']) == 198
    unidentifiable = record['non_identifiable']
    assert len(unidentifiable) == 2 * 33 * 2
    assert {entry['state'] for entry in unidentifiable} == {lanes.NON_IDENTIFIABLE}
    # No point, no interval and no verdict anywhere in the record for those rungs.
    assert all('point' not in entry and 'interval' not in entry for entry in unidentifiable)
    assert {entry['control_set'] for entry in record['columns']} == set(
        lanes.lane('interaction').rungs)
    assert record['state_counts'][lanes.NON_IDENTIFIABLE] == len(unidentifiable)


# --------------------------------------------------------------------------- #
# The declared families.
# --------------------------------------------------------------------------- #

def test_declared_family_is_control_sets_by_arms_by_directions():
    expected = {'abundance': 330, 'stability': 330, 'interaction': 198, 'phenotype': 100}
    for lane, size in expected.items():
        declared = lanes.lane(lane)
        for channel in declared.channels:
            family = lanes.declared_family(lane, channel)
            assert family['size'] == size == len(declared.rungs) * len(declared.arms) * 2
            assert family['control_sets'] == list(declared.rungs)
            assert family['directions'] == list(bidirectional.DIRECTIONS)
    assert lanes.lane('abundance').channels == ('likelihood', 'representation')
    assert lanes.declared_family('abundance', 'representation')['channel_role'] == 'secondary'
    assert lanes.declared_family('abundance', 'likelihood')['channel_role'] == 'primary'


def test_the_partial_phenotype_family_states_what_it_becomes_at_full_panel():
    family = lanes.declared_family('phenotype', 'likelihood')
    partial = family['partial_panel']
    assert (partial['scored_arms'], partial['panel_arms']) == (10, 33)
    assert partial['family_at_full_panel'] == 330
    assert '10 of 33 arms are scored' in partial['note'] and '330' in partial['note']
    assert 'partial_panel' not in lanes.declared_family('stability', 'likelihood')


def test_a_family_declaration_that_disagrees_with_its_shape_is_refused(monkeypatch):
    lane = lanes.lane('phenotype')
    monkeypatch.setitem(lanes.LANES, 'phenotype',
                        dataclasses.replace(lane, declared_family={'likelihood': 66}))
    with pytest.raises(ValueError, match='the realised likelihood family is 100'):
        lanes.declared_family('phenotype', 'likelihood')
    monkeypatch.setitem(lanes.LANES, 'phenotype',
                        dataclasses.replace(lane, full_panel_family={'likelihood': 66}))
    with pytest.raises(ValueError, match='the full-panel likelihood family would be 330'):
        lanes.declared_family('phenotype', 'likelihood')


def test_the_import_time_check_states_the_full_panel_size(monkeypatch):
    lane = lanes.lane('phenotype')
    monkeypatch.setitem(lanes.LANES, 'phenotype',
                        dataclasses.replace(lane, declared_family={'likelihood': 110}))
    with pytest.raises(RuntimeError) as error:
        lanes._check_declarations()
    assert '5 control sets x 10 arms x 2 directions = 100 columns' in str(error.value)
    assert 'at the full 33-arm panel it would be 330' in str(error.value)


def test_the_sensitivity_family_is_the_nuisance_rung_only():
    family = lanes.declared_family('abundance', 'likelihood', role='sensitivity')
    assert family['control_sets'] == ['nuisance'] and family['size'] == 66
    partial = lanes.declared_family('phenotype', 'likelihood', role='sensitivity')
    assert partial['size'] == 20 and partial['partial_panel']['family_at_full_panel'] == 66


def test_the_full_support_sensitivity_is_its_own_family_beside_the_hierarchy():
    groups = [f'g{index:02d}' for index in range(12)]
    cells = cell_matrix('stability', 'ranking', groups, rungs=('nuisance',), digest='b' * 64)
    record = lanes.panel('stability', 'ranking', 'likelihood', cells, groups=groups,
                         support_sha256='b' * 64, role='sensitivity')
    assert record['role'] == 'sensitivity' and record['support'] == 'full'
    assert record['family']['size'] == 66 == len(record['columns'])
    assert {entry['control_set'] for entry in record['columns']} == {'nuisance'}
    # The hierarchy's family is unchanged by it: the two are never pooled.
    assert lanes.declared_family('stability', 'likelihood')['size'] == 330


def test_the_finding_states_are_read_from_one_source():
    from src.capability.core import statistics
    from src.capability.generation import objective_tuned_analysis as ota

    assert lanes.STATES is statistics.FINDING_STATES
    assert ota.FINDING_STATES is statistics.FINDING_STATES
    assert (lanes.SUPPORTED, lanes.BOUNDED_NEGATIVE, lanes.UNRESOLVED,
            lanes.NON_IDENTIFIABLE) == statistics.FINDING_STATES
    assert (ota.SUPPORTED, ota.BOUNDED_NEGATIVE, ota.UNRESOLVED,
            ota.NON_IDENTIFIABLE) == statistics.FINDING_STATES


def test_the_variant_effect_bridge_is_declared_and_the_lane_is_not_re_run():
    bridge = lanes.VARIANT_EFFECT_BRIDGE
    assert bridge['rerun_at_this_hierarchy'] is False
    assert 'already published verdicts' in bridge['rerun_refused_because']
    # The profile family supplies rungs two through five with an empty nuisance set.
    assert bridge['profile_family']['nuisance_set'] == 'empty'
    assert set(bridge['profile_family']['rungs']) == set(lanes.SHARED_RUNGS)
    assert bridge['profile_family']['rungs']['nuisance+profile'] == 'P'
    # The C0 family supplies rungs two and five only.
    assert set(bridge['c0_family']['rungs']) == {'nuisance+profile',
                                                 'nuisance+profile+couplings+gemme'}
    assert set(bridge['absent_rungs']) == {'nuisance', 'nuisance+profile+couplings',
                                           'nuisance+profile+gemme'}
    assert lanes.SHARED_RUNGS == tuple(name for name, roles in lanes.RUNGS.items()
                                       if 'profile' in roles)
    assert 'rungs two through five of the profile family' in bridge['cross_lane_reading']


def test_the_bridge_is_checked_against_the_hierarchy(monkeypatch):
    monkeypatch.setitem(lanes.VARIANT_EFFECT_BRIDGE, 'absent_rungs', {})
    with pytest.raises(RuntimeError, match='account for every rung'):
        lanes._check_declarations()


def test_every_lane_declares_the_shared_rungs_the_bridge_names():
    for lane in lanes.LANES:
        shared = set(lanes.lane(lane).rungs) & set(lanes.SHARED_RUNGS)
        assert 'nuisance+profile' in shared
        contract = lanes.contract(lane)
        assert contract['shared_control_sets'] == list(lanes.SHARED_RUNGS)
        assert contract['variant_effect_bridge']['rerun_at_this_hierarchy'] is False


def test_the_variant_effect_lane_is_the_collision_witness_and_stays_frozen():
    from src.capability.extensions import statistical_ladder
    assert set(lanes.VARIANT_EFFECT_ROLES) == set(lanes.ROLES)
    assert lanes.VARIANT_EFFECT_ROLES['gemme'] == ('G',)
    assert lanes.VARIANT_EFFECT_ROLES['couplings'] == ('E',)
    letters = {letter for keys in lanes.VARIANT_EFFECT_ROLES.values() for letter in keys}
    assert letters == set(statistical_ladder.BLOCK_ORDER) - {'M'}
    # The ladder's own control sets are not redeclared here, so nothing frozen moves.
    assert 'variant_effect' not in lanes.LANES
    assert set(statistical_ladder.CONTROL_SETS) == {'P', 'P+E', 'P+G', 'P+E+G', 'C0', 'C0+E+G'}


# --------------------------------------------------------------------------- #
# Both directions, one band convention.
# --------------------------------------------------------------------------- #

def test_both_directions_come_from_the_shared_declaration_unchanged():
    groups = [f'g{index:02d}' for index in range(12)]
    cells = cell_matrix('stability', 'transfer_error', groups)
    record = lanes.panel('stability', 'transfer_error', 'likelihood', cells, groups=groups,
                         support_sha256='a' * 64)
    directions = [entry['direction'] for entry in record['columns'][:2]]
    assert directions == list(bidirectional.DIRECTIONS)
    assert record['contrasts'] == {
        'model_given_controls': 'loss(U_C) - loss(U_C_M)',
        'controls_given_model': 'loss(M) - loss(U_C_M)'}
    # The pairing identity: on the same rows the two columns share the joint arm, so
    # their difference is exactly the difference between the two reduced arms.
    rung, arm = record['columns'][0]['control_set'], record['columns'][0]['arm']
    cell = cells[rung, arm]
    expected = float(np.mean(np.asarray(cell['U_C']) - np.asarray(cell['M'])))
    assert record['point'][0] - record['point'][1] == pytest.approx(expected, abs=1e-12)


def test_a_score_stratum_is_signed_so_that_positive_is_always_improvement():
    groups = [f'g{index:02d}' for index in range(12)]
    for stratum in ('transfer_error', 'ranking'):
        cells = cell_matrix('stability', stratum, groups)
        record = lanes.panel('stability', stratum, 'likelihood', cells, groups=groups,
                             support_sha256='a' * 64)
        assert all(value > 0 for value in record['point'])
    assert lanes.stratum('stability', 'ranking').to_loss == -1.
    assert lanes.stratum('stability', 'transfer_error').to_loss == 1.


def test_the_primary_band_is_native_and_the_studentised_family_is_a_sensitivity():
    groups = [f'g{index:02d}' for index in range(16)]
    cells = cell_matrix('phenotype', 'activity_venus-ranking', groups)
    record = lanes.panel('phenotype', 'activity_venus-ranking', 'likelihood', cells,
                         groups=groups, support_sha256='a' * 64)
    assert record['native']['method'] == bidirectional.NATIVE_METHOD
    assert record['native']['draws'] == bidirectional.DRAWS
    assert record['native']['seed'] == bidirectional.SEED
    assert record['standardized_maxT_sensitivity']['role'] == 'sensitivity'
    assert record['standardized_maxT_sensitivity']['method'] == bidirectional.STANDARDIZED_METHOD
    native = np.asarray(record['native']['interval'], dtype=float)
    width = native[:, 1] - native[:, 0]
    # One additive critical value in native units, not one width per column.
    assert np.allclose(width, 2 * record['native']['joint_critical_max_absolute_deviation'])


def test_the_four_states_are_kept_apart():
    groups = [f'g{index:02d}' for index in range(20)]
    lane, stratum = 'phenotype', 'binding_skempi-quantitative'
    declared = lanes.lane(lane)
    rng = np.random.default_rng(11)
    cells = {}
    for position, rung in enumerate(declared.rungs):
        for arm in declared.arms:
            joint = rng.normal(1.0, 0.01, size=len(groups))
            # The first rung is a resolved negative, the second a zero crossing, the rest
            # resolved positives; the reverse direction is always a resolved positive.
            forward = {0: -0.4, 1: 0.0}.get(position, 0.4)
            cells[rung, arm] = {'U_C': joint + forward + rng.normal(0, 0.01, len(groups)),
                                'M': joint + 0.8 + rng.normal(0, 0.01, len(groups)),
                                'U_C_M': joint}
    record = lanes.panel(lane, stratum, 'likelihood', cells, groups=groups,
                         support_sha256='a' * 64)
    states = {}
    for entry, state in zip(record['columns'], record['states']):
        states[entry['control_set'], entry['direction']] = state
    assert states[declared.rungs[0], 'model_given_controls'] == lanes.BOUNDED_NEGATIVE
    assert states[declared.rungs[1], 'model_given_controls'] == lanes.UNRESOLVED
    assert states[declared.rungs[2], 'model_given_controls'] == lanes.SUPPORTED
    assert all(states[rung, 'controls_given_model'] == lanes.SUPPORTED
               for rung in declared.rungs)
    counts = record['state_counts']
    assert sum(counts[name] for name in lanes.STATES) == record['family']['size']
    assert record['state_rule'][lanes.UNRESOLVED].startswith('the interval contains zero')
    assert 'no margin is declared' in record['state_rule'][lanes.UNRESOLVED]


def test_a_narrow_interval_that_contains_zero_is_unresolved_and_not_negative():
    groups = [f'g{index:02d}' for index in range(24)]
    # The control arm shares the joint arm's per-group draw, so the forward reduction is
    # exactly -1e-5 in every group: a tiny resolved-looking point with an interval that
    # still contains zero, because the band is shared with the reverse direction.
    cells = cell_matrix('stability', 'transfer_error', groups, forward=-1e-5, reverse=0.3,
                        spread=1e-3, exact_forward=True)
    record = lanes.panel('stability', 'transfer_error', 'likelihood', cells, groups=groups,
                         support_sha256='a' * 64)
    forward = [(state, point) for entry, state, point
               in zip(record['columns'], record['states'], record['point'])
               if entry['direction'] == 'model_given_controls']
    assert {state for state, _ in forward} == {lanes.UNRESOLVED}
    assert all(point == pytest.approx(-1e-5, abs=1e-12) for _, point in forward)


def test_a_stratum_under_the_floor_is_unresolved_rather_than_estimated():
    groups = [f'g{index}' for index in range(MINIMUM_BOOTSTRAP_UNITS - 1)]
    cells = cell_matrix('phenotype', 'cellular_fitness_sge-ranking', groups)
    record = lanes.panel('phenotype', 'cellular_fitness_sge-ranking', 'likelihood', cells,
                         groups=groups, support_sha256='a' * 64)
    assert record['status'] == lanes.UNRESOLVED
    assert record['native'] is None and record['standardized_maxT_sensitivity'] is None
    assert set(record['states']) == {lanes.UNRESOLVED}
    assert record['independence_floor']['degenerate'] is True
    assert str(MINIMUM_BOOTSTRAP_UNITS) in record['reason']
    assert record['point'] and len(record['point']) == record['family']['size']


def test_a_studentised_column_without_group_spread_carries_no_state():
    groups = [f'g{index:02d}' for index in range(12)]
    cells = cell_matrix('stability', 'transfer_error', groups)
    rung, arm = lanes.lane('stability').rungs[0], lanes.lane('stability').arms[0]
    constant = np.full(len(groups), 2.0)
    cells[rung, arm] = {'U_C': constant + 0.5, 'M': constant + 1.0, 'U_C_M': constant,
                        'support_sha256': 'a' * 64}
    record = lanes.panel('stability', 'transfer_error', 'likelihood', cells, groups=groups,
                         support_sha256='a' * 64)
    sensitivity = record['standardized_maxT_sensitivity']
    assert sensitivity['zero_standard_error_columns'] == [0, 1]
    assert sensitivity['states'][0] is None and sensitivity['states'][1] is None
    assert record['state_counts']['unassigned'] == 0  # the primary family still resolves
    assert record['states'][0] in lanes.STATES


# --------------------------------------------------------------------------- #
# The forward bound: the quantity that carries the overlap claim.
# --------------------------------------------------------------------------- #

def test_the_forward_bound_is_reported_in_native_units_and_as_a_fraction():
    groups = [f'g{index:02d}' for index in range(16)]
    cells = cell_matrix('stability', 'transfer_error', groups, forward=0.02, reverse=0.5,
                        spread=0.02)
    record = lanes.panel('stability', 'transfer_error', 'likelihood', cells, groups=groups,
                         support_sha256='a' * 64)
    bound = record['forward_bound']
    # One entry per forward column only: the reverse direction carries no bound.
    assert len(bound['columns']) == record['family']['size'] // 2
    assert bound['direction'] == 'model_given_controls'
    index = {(entry['control_set'], entry['arm']): entry for entry in bound['columns']}
    for position, (entry, interval) in enumerate(zip(record['columns'],
                                                     record['native']['interval'])):
        if entry['direction'] != 'model_given_controls':
            continue
        column = index[entry['control_set'], entry['arm']]
        assert column['upper_limit'] == pytest.approx(interval[1], abs=0, rel=0)
        assert column['units'] == lanes.stratum('stability', 'transfer_error').units
        expected = column['upper_limit'] / column['control_performance']
        assert column['fraction_of_control_performance'] == pytest.approx(expected)
        assert column['fraction_withheld_because'] is None
    # The control performance is the control arm's own group-equal mean on these groups.
    rung, arm = lanes.lane('stability').rungs[0], lanes.lane('stability').arms[0]
    assert index[rung, arm]['control_performance'] == pytest.approx(
        float(np.mean(cells[rung, arm]['U_C'])))
    summary = bound['per_control_set'][rung]
    assert summary['arms'] == 33 and summary['arms_without_a_fraction'] == 0
    assert summary['largest_upper_limit'] == max(
        entry['upper_limit'] for entry in bound['columns'] if entry['control_set'] == rung)
    assert 'removes at most this fraction of the error' in bound['fraction_reading']


def test_a_score_stratum_reads_the_bound_against_its_own_rank_agreement():
    groups = [f'g{index:02d}' for index in range(16)]
    cells = cell_matrix('phenotype', 'activity_venus-ranking', groups)
    record = lanes.panel('phenotype', 'activity_venus-ranking', 'likelihood', cells,
                         groups=groups, support_sha256='a' * 64)
    bound = record['forward_bound']
    assert 'rank agreement the explicit statistics already achieve' in bound['fraction_reading']
    assert all(entry['control_performance'] > 0 for entry in bound['columns'])
    assert all(entry['fraction_of_control_performance'] > 0 for entry in bound['columns'])


def test_the_fraction_is_withheld_where_the_control_has_no_positive_performance():
    groups = [f'g{index:02d}' for index in range(16)]
    rng = np.random.default_rng(19)
    declared = lanes.lane('phenotype')
    cells = {}
    for rung in declared.rungs:
        for arm in declared.arms:
            joint = rng.normal(-0.2, 0.02, size=len(groups))
            # Each arm carries its own per-group deviation, so the paired contrasts vary
            # across the resampling units. A fixture whose contrasts are exactly constant
            # has no statistic to studentise and the shared band helper refuses it.
            cells[rung, arm] = {'U_C': joint + rng.normal(-0.02, 0.004, size=len(groups)),
                                'M': joint + rng.normal(-0.4, 0.01, size=len(groups)),
                                'U_C_M': joint}
    record = lanes.panel('phenotype', 'activity_venus-ranking', 'likelihood', cells,
                         groups=groups, support_sha256='a' * 64)
    for entry in record['forward_bound']['columns']:
        assert entry['control_performance'] < 0
        assert entry['fraction_of_control_performance'] is None
        assert 'not a positive performance' in entry['fraction_withheld_because']
    assert all(summary['largest_fraction_of_control_performance'] is None
               for summary in record['forward_bound']['per_control_set'].values())
    assert all(summary['arms_without_a_fraction'] == 10
               for summary in record['forward_bound']['per_control_set'].values())


def test_the_forward_direction_never_reports_a_state_for_smallness():
    for lane in lanes.LANES:
        contract = lanes.contract(lane)
        assert contract['forward_bound']['never'] == (
            'the forward direction never reports a state for smallness')
        assert 'no equivalence margin' not in contract['forward_bound']['reading'] or True
        assert 'at most this fraction' in contract['forward_bound']['reading']
    # And a bound is still reported where the stratum cannot be estimated at all.
    groups = [f'g{index}' for index in range(MINIMUM_BOOTSTRAP_UNITS - 1)]
    cells = cell_matrix('phenotype', 'cellular_fitness_sge-ranking', groups)
    record = lanes.panel('phenotype', 'cellular_fitness_sge-ranking', 'likelihood', cells,
                         groups=groups, support_sha256='a' * 64)
    assert record['forward_bound']['status'] == lanes.UNRESOLVED
    assert 'columns' not in record['forward_bound']


# --------------------------------------------------------------------------- #
# Fragility, uncovered readings and the GEMME risk band.
# --------------------------------------------------------------------------- #

def test_the_cellular_fitness_stratum_is_declared_fragile():
    fragile = lanes.stratum('phenotype', 'cellular_fitness_sge-ranking').fragile
    assert fragile and 'nine homology groups' in fragile
    assert 'floor is not lowered' in fragile
    groups = [f'g{index}' for index in range(9)]
    cells = cell_matrix('phenotype', 'cellular_fitness_sge-ranking', groups)
    record = lanes.panel('phenotype', 'cellular_fitness_sge-ranking', 'likelihood', cells,
                         groups=groups, support_sha256='a' * 64)
    assert record['status'] == 'complete' and record['stratum_fragile'] == fragile
    # One group lost and the stratum is still estimable; two and it is unresolved.
    for count, status in ((8, 'complete'), (7, lanes.UNRESOLVED)):
        shrunk = [f'g{index}' for index in range(count)]
        fewer = {key: {name: (np.asarray(value)[:count] if name in lanes.ARM_NAMES else value)
                       for name, value in cell.items()}
                 for key, cell in cells.items()}
        assert lanes.panel('phenotype', 'cellular_fitness_sge-ranking', 'likelihood', fewer,
                           groups=shrunk, support_sha256='a' * 64)['status'] == status
    assert lanes.stratum('abundance', 'ranking').fragile is None


def test_the_arm_conditional_control_is_declared_as_a_design_choice():
    for lane in ('abundance', 'stability'):
        statement = lanes.contract(lane)['arm_conditional_control']
        assert statement and 'by design' in statement
        assert 'would credit segmentation to the model' in statement
    for lane in ('interaction', 'phenotype'):
        assert lanes.contract(lane)['arm_conditional_control'] is None


def test_an_uncovered_reading_is_declared_rather_than_left_out_silently():
    assert lanes.lane('phenotype').not_covered
    uncovered = lanes.contract('phenotype')['not_covered']
    assert 'binding_skempi-quantitative_calibrated' in uncovered
    assert '97-group support' in uncovered['binding_skempi-quantitative_calibrated']
    assert lanes.contract('abundance')['not_covered'] == {}


def test_the_gemme_risk_band_is_counted_from_the_export():
    support = {'schema': lanes.UPSTREAM_SUPPORT_SCHEMA, 'cohort': 'synthetic',
               'wildtypes': {'a': {'rows': 1}, 'b': {'rows': 2}, 'c': {'rows': 14},
                             'd': {'rows': 15}, 'e': {'rows': 4000}}}
    band = lanes.gemme_risk_band(support)
    assert band['band'] == [2, 14] and band['in_band'] == 2
    assert sorted(band['rows_by_reference']) == ['b', 'c']
    assert 'does not exempt it' in band['alignment_reuse_note']
    with pytest.raises(ValueError, match='is not'):
        lanes.gemme_risk_band(dict(support, schema='other'))
    with pytest.raises(ValueError, match='records no retained row count'):
        lanes.gemme_risk_band({'schema': lanes.UPSTREAM_SUPPORT_SCHEMA,
                               'wildtypes': {'a': {}}})


def test_the_exported_cohorts_risk_band_counts():
    expected = {'abundance': 6, 'pairwise': 15, 'phenotype': 16}
    for cohort, count in expected.items():
        path = ROOT / f'results/extensions/explicit_statistics_{cohort}_20261109' / lanes.SUPPORT_FILE
        if not path.is_file():
            pytest.skip('the cohort export is not present in this checkout')
        assert lanes.gemme_risk_band(json.loads(path.read_text()))['in_band'] == count


# --------------------------------------------------------------------------- #
# Matched support, and what the panel refuses.
# --------------------------------------------------------------------------- #

def test_a_cell_on_another_support_is_refused():
    groups = [f'g{index:02d}' for index in range(12)]
    cells = cell_matrix('stability', 'transfer_error', groups)
    key = (lanes.lane('stability').rungs[2], lanes.lane('stability').arms[0])
    cells[key] = dict(cells[key], support_sha256='b' * 64)
    with pytest.raises(ValueError, match='unmatched support'):
        lanes.panel('stability', 'transfer_error', 'likelihood', cells, groups=groups,
                    support_sha256='a' * 64)


def test_an_incomplete_or_undeclared_family_is_refused():
    groups = [f'g{index:02d}' for index in range(12)]
    cells = cell_matrix('phenotype', 'activity_venus-ranking', groups)
    missing = {key: value for key, value in cells.items() if key[1] != 'protgpt2'}
    with pytest.raises(ValueError, match='are absent'):
        lanes.panel('phenotype', 'activity_venus-ranking', 'likelihood', missing, groups=groups,
                    support_sha256='a' * 64)
    extra = dict(cells)
    extra['nuisance', 'rita-xl'] = cells['nuisance', 'protgpt2']
    with pytest.raises(ValueError, match='undeclared'):
        lanes.panel('phenotype', 'activity_venus-ranking', 'likelihood', extra, groups=groups,
                    support_sha256='a' * 64)


def test_a_cell_missing_an_arm_or_a_group_is_refused():
    groups = [f'g{index:02d}' for index in range(12)]
    cells = cell_matrix('phenotype', 'activity_venus-ranking', groups)
    key = ('nuisance', 'gpt2-large')
    broken = dict(cells)
    broken[key] = {name: value for name, value in cells[key].items() if name != 'M'}
    with pytest.raises(ValueError, match="missing \\['M'\\]"):
        lanes.panel('phenotype', 'activity_venus-ranking', 'likelihood', broken, groups=groups,
                    support_sha256='a' * 64)
    short = dict(cells)
    short[key] = dict(cells[key], M=np.asarray(cells[key]['M'])[:-1])
    with pytest.raises(ValueError, match='one finite value for each of the 12 groups'):
        lanes.panel('phenotype', 'activity_venus-ranking', 'likelihood', short, groups=groups,
                    support_sha256='a' * 64)
    nonfinite = dict(cells)
    values = np.asarray(cells[key]['M'], dtype=float).copy()
    values[0] = np.nan
    nonfinite[key] = dict(cells[key], M=values)
    with pytest.raises(ValueError, match='one finite value'):
        lanes.panel('phenotype', 'activity_venus-ranking', 'likelihood', nonfinite,
                    groups=groups, support_sha256='a' * 64)


def test_repeated_resampling_units_are_refused():
    groups = ['g1', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8']
    cells = cell_matrix('stability', 'ranking', groups)
    with pytest.raises(ValueError, match='repeat a label'):
        lanes.panel('stability', 'ranking', 'likelihood', cells, groups=groups,
                    support_sha256='a' * 64)


def test_an_unknown_lane_stratum_or_channel_is_refused():
    with pytest.raises(ValueError, match='unknown lane'):
        lanes.lane('generation')
    with pytest.raises(ValueError, match='unknown stratum'):
        lanes.stratum('abundance', 'ranking_error')
    with pytest.raises(ValueError, match='no .representation. channel'):
        lanes.channel_block('stability', 'representation')


# --------------------------------------------------------------------------- #
# The complete-case support and its losses.
# --------------------------------------------------------------------------- #

def test_complete_case_keeps_the_rows_every_block_covers_and_names_every_loss():
    rows = cohort_rows()
    statuses = {row['sample_id']: 'no_homolog_support' for row in rows
                if row['wildtype_id'] == 'r2'}
    couplings = score_table('couplings', rows, statuses=statuses, refused={
        'r2': {'reason': '1 retained homolog rows is below the 2-row floor of a Potts fit',
               'rows': 1, 'samples': 4}})
    gemme_statuses = {row['sample_id']: 'refused_degenerate' for row in rows
                      if row['wildtype_id'] == 'r3'}
    gemme = score_table('gemme', rows, statuses=gemme_statuses, refused={
        'r3': {'reason': 'every mutation scored 0.0 in every run: JET2 placed no position',
               'alignment_rows': 4, 'samples': 4}})
    groups = {row['sample_id']: f"group-{row['wildtype_id']}" for row in rows}
    record = lanes.complete_case('stability', rows, {'couplings': couplings, 'gemme': gemme},
                                 groups=groups)
    assert record['rows_offered'] == 12 and record['rows_retained'] == 4
    assert record['references_lost'] == ['r2', 'r3']
    assert record['losses']['couplings']['reference_count'] == 1
    assert record['losses']['couplings']['rows'] == 4
    assert 'below the 2-row floor' in record['losses']['couplings']['references']['r2']['reason']
    assert 'JET2 placed no position' in record['losses']['gemme']['references']['r3']['reason']
    assert record['groups']['emptied'] == ['group-r2', 'group-r3']
    assert record['groups']['retained'] == 1
    assert record['groups']['floor']['degenerate'] is True
    assert record['groups']['status'] == lanes.UNRESOLVED
    assert record['support_sha256'] != record['full_support_sha256']
    assert len(record['sample_ids']) == 4


def test_the_interaction_lane_support_is_not_restricted_by_gemme():
    rows = cohort_rows()
    couplings = score_table('couplings', rows)
    record = lanes.complete_case('interaction', rows, {'couplings': couplings})
    assert record['required_blocks'] == ['couplings']
    assert record['rows_retained'] == record['rows_offered']
    assert record['support_sha256'] == record['full_support_sha256']
    assert 'gemme' in record['non_identifiable']
    with pytest.raises(ValueError, match='declared non-identifiable'):
        lanes.complete_case('interaction', rows,
                            {'couplings': couplings, 'gemme': score_table('gemme', rows)})


def test_a_missing_block_table_is_refused_rather_than_dropped():
    rows = cohort_rows()
    with pytest.raises(ValueError, match="needs the \\['gemme'\\] score tables"):
        lanes.complete_case('stability', rows, {'couplings': score_table('couplings', rows)})


def test_an_absent_or_incomplete_wave_is_refused_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match='wave has not landed'):
        lanes.read_scores(tmp_path / 'scores.json', 'couplings')
    path = tmp_path / 'scores.json'
    rows = cohort_rows()
    path.write_text(json.dumps(score_table('couplings', rows, complete=False)))
    with pytest.raises(ValueError, match='counts.complete=False'):
        lanes.read_scores(path, 'couplings')
    path.write_text(json.dumps(dict(score_table('couplings', rows), schema='something_else')))
    with pytest.raises(ValueError, match='is not'):
        lanes.read_scores(path, 'couplings')
    path.write_text(json.dumps({'schema': lanes.BLOCK_SOURCE['couplings']['schema'],
                                'samples': rows}))
    with pytest.raises(ValueError, match='declares no counts'):
        lanes.read_scores(path, 'couplings')


def test_pending_rows_are_an_absent_wave_and_never_a_loss():
    rows = cohort_rows()
    statuses = {rows[0]['sample_id']: 'not_fitted'}
    table = score_table('couplings', rows, statuses=statuses)
    with pytest.raises(ValueError, match='pending rows'):
        lanes.complete_case('interaction', rows, {'couplings': table})


def test_an_unclassified_status_is_refused_rather_than_guessed():
    rows = cohort_rows()
    statuses = {rows[0]['sample_id']: 'cv_not_converged_after_caps'}
    table = score_table('couplings', rows, statuses=statuses)
    with pytest.raises(ValueError, match='resolved upstream, never classified here'):
        lanes.complete_case('interaction', rows, {'couplings': table})


def test_a_status_that_disagrees_with_its_value_is_refused():
    rows = cohort_rows()
    table = score_table('couplings', rows)
    table['samples'][0][lanes.BLOCK_SOURCE['couplings']['column']] = None
    with pytest.raises(ValueError, match='disagrees with itself'):
        lanes.complete_case('interaction', rows, {'couplings': table})
    table = score_table('couplings', rows)
    table['samples'][0][lanes.BLOCK_SOURCE['couplings']['column']] = 'low'
    with pytest.raises(ValueError, match='non-numeric'):
        lanes.complete_case('interaction', rows, {'couplings': table})


def test_a_refused_row_no_one_enumerates_is_refused():
    rows = cohort_rows()
    statuses = {rows[0]['sample_id']: 'no_homolog_support'}
    table = score_table('couplings', rows, statuses=statuses)
    with pytest.raises(ValueError, match='neither it nor the cohort export enumerates'):
        lanes.complete_case('interaction', rows, {'couplings': table})
    # An absent export document narrows what may be accepted and never widens it.
    with pytest.raises(ValueError, match='neither it nor the cohort export enumerates'):
        lanes.complete_case('interaction', rows, {'couplings': table},
                            refusals={'refused': {}})


def test_the_export_is_the_authority_on_a_reference_it_refused_for_both_blocks():
    # The export refuses a reference with too few homolog rows once, for both blocks.
    # Neither block table re-enumerates it, and the loss is still a declared loss.
    rows = cohort_rows()
    statuses = {row['sample_id']: 'no_homolog_support'
                for row in rows if row['wildtype_id'] == 'r1'}
    table = score_table('couplings', rows, statuses=statuses)
    refusals = {'schema': lanes.REFUSALS_SCHEMA, 'count': 1, 'read_from': 'export/refusals.json',
                'refused': {'r1': {'reason': 'no retained alignment hit', 'rows': 0}}}
    record = lanes.complete_case('interaction', rows, {'couplings': table}, refusals=refusals)
    assert record['references_lost'] == ['r1']
    loss = record['losses']['couplings']
    assert loss['references']['r1']['declared_by'] == 'cohort export'
    assert loss['declared_by'] == {'block table': 0, 'cohort export': 1}
    assert record['refusals_read_from'] == 'export/refusals.json'


def test_a_block_tables_own_refusal_is_still_its_own_to_declare():
    rows = cohort_rows()
    statuses = {row['sample_id']: 'refused_degenerate'
                for row in rows if row['wildtype_id'] == 'r1'}
    table = score_table('gemme', rows, statuses=statuses,
                        refused={'r1': {'reason': 'degenerate run'}})
    record = lanes.complete_case('stability', rows, {
        'gemme': table, 'couplings': score_table('couplings', rows)}, refusals={'refused': {}})
    assert record['losses']['gemme']['references']['r1']['declared_by'] == 'block table'


def test_the_refusals_reader_refuses_a_document_that_disagrees_with_itself(tmp_path):
    path = tmp_path / 'refusals.json'
    path.write_text(json.dumps({'schema': lanes.REFUSALS_SCHEMA, 'count': 2,
                                'refused': {'r1': {'reason': 'x'}}}))
    with pytest.raises(ValueError, match='declares 2 refusals and enumerates 1'):
        lanes.read_refusals(path)
    path.write_text(json.dumps({'schema': 'something_else', 'count': 0, 'refused': {}}))
    with pytest.raises(ValueError, match='is not'):
        lanes.read_refusals(path)


# --------------------------------------------------------------------------- #
# What a fitted cell is bound to.
# --------------------------------------------------------------------------- #

def test_a_full_support_identity_excludes_the_added_block_vocabulary():
    full = lanes.fit_identity('abundance', 'full')
    complete = lanes.fit_identity('abundance', 'complete_case')
    for key in ('row_status', 'added_blocks', 'block_source', 'added_block_rendering',
                'cycle_contrast'):
        assert key not in full, key
        assert key in complete, key
    # No full-support rung carries an added block, which is what makes that sound.
    for rung in lanes.FULL_SUPPORT_RUNGS:
        assert not set(lanes.RUNGS[rung]) & set(lanes.NEW_BLOCKS)


def test_the_identity_carries_what_decides_a_number_and_no_prose():
    identity = lanes.fit_identity('stability', 'complete_case')
    assert identity['profile_blocks'] == ['prof2']
    assert identity['strata_to_loss'] == {'ranking': -1., 'transfer_error': 1.,
                                          'calibrated_error': 1.}
    assert identity['arms'] == list(lanes.ARM_NAMES)
    assert identity['row_status']['gemme']['no_homolog_support'] == lanes.REFUSED
    rendered = json.dumps(identity)
    for prose in (lanes.NUISANCE_RUNG_LIMITATION, lanes.GEMME_CYCLE_REASON,
                  lanes.FIT_IDENTITY_RULE, lanes.EXPORT_REFUSAL_AUTHORITY):
        assert prose not in rendered
    for limitation in lanes.LIMITATIONS:
        assert limitation not in rendered


def test_an_unknown_support_has_no_identity():
    with pytest.raises(ValueError, match='unknown support'):
        lanes.fit_identity('abundance', 'archived')


def test_a_table_that_does_not_cover_the_rows_is_refused():
    rows = cohort_rows()
    table = score_table('couplings', rows)
    table['samples'] = table['samples'][:-1]
    with pytest.raises(ValueError, match='does not cover the cohort rows'):
        lanes.complete_case('interaction', rows, {'couplings': table})


def test_duplicate_rows_are_refused():
    rows = cohort_rows()
    table = score_table('couplings', rows + rows[:1])
    with pytest.raises(ValueError, match='duplicate sample_id'):
        lanes.complete_case('interaction', rows + rows[:1], {'couplings': table})


def test_the_real_producer_output_is_refused_while_its_wave_is_partial():
    path = ROOT / ('results/extensions/explicit_statistics_pairwise_20261109/'
                   'potts-pilot/scores.json')
    if not path.is_file():
        pytest.skip('the cohort pilot output is not present in this checkout')
    with pytest.raises(ValueError) as error:
        lanes.read_scores(path, 'couplings')
    assert 'counts.complete=False' in str(error.value)
    assert 'never read as a control' in str(error.value)


# --------------------------------------------------------------------------- #
# How an added block enters, the rows reader and the contract.
# --------------------------------------------------------------------------- #

def test_an_added_block_enters_as_one_assay_reranked_column():
    values = np.asarray([3.0, 1.0, 2.0, 10.0, 20.0])
    assays = np.asarray(['a', 'a', 'a', 'b', 'b'])
    block = lanes.prepare_scalar(values, assays)
    assert block.shape == (5, 1)
    assert np.allclose(block[:, 0], overlap.rerank(values, assays))
    with pytest.raises(ValueError, match='finite on the support'):
        lanes.prepare_scalar([1.0, np.nan], ['a', 'a'])
    with pytest.raises(ValueError, match='one value per row'):
        lanes.prepare_scalar([1.0, 2.0], ['a'])


def test_the_rows_reader_refuses_a_document_of_another_shape(tmp_path):
    path = tmp_path / 'variant_table.json'
    path.write_text(json.dumps({'schema': 'other', 'samples': cohort_rows()}))
    with pytest.raises(ValueError, match='is not'):
        lanes.read_rows(path)
    path.write_text(json.dumps({'schema': lanes.ROWS_SCHEMA, 'samples': []}))
    with pytest.raises(ValueError, match='holds no sample rows'):
        lanes.read_rows(path)
    path.write_text(json.dumps({'schema': lanes.ROWS_SCHEMA, 'samples': cohort_rows()}))
    assert len(lanes.read_rows(path)) == 12


def test_every_lane_declares_a_serialisable_contract():
    for lane in lanes.LANES:
        contract = lanes.contract(lane)
        assert contract['schema'] == lanes.CONTRACT_SCHEMA
        assert list(contract['hierarchy']) == list(lanes.RUNGS)
        assert contract['uncertainty']['primary'] == bidirectional.NATIVE_METHOD
        assert contract['uncertainty']['sensitivity'] == bidirectional.STANDARDIZED_METHOD
        assert contract['uncertainty']['floor'] == MINIMUM_BOOTSTRAP_UNITS
        assert contract['states'] == list(lanes.STATES)
        assert contract['archived_control_set'] == lanes.ARCHIVED_RUNG
        assert set(contract['families']) == set(lanes.lane(lane).channels)
        json.dumps(contract, allow_nan=False)
    assert lanes.contract('interaction')['non_identifiable']['gemme']
    assert lanes.contract('abundance')['non_identifiable'] == {}


def test_the_declaration_check_holds_for_the_declared_lanes():
    lanes._check_declarations()


# --------------------------------------------------------------------------- #
# The entry point.
# --------------------------------------------------------------------------- #

def test_declare_writes_a_contract_without_any_wave(tmp_path, capsys):
    lanes.main(['declare', '--lane', 'interaction'])
    printed = json.loads(capsys.readouterr().out)
    assert printed['lane'] == 'interaction' and printed['families']['likelihood']['size'] == 198
    out = tmp_path / 'contract.json'
    lanes.main(['declare', '--lane', 'phenotype', '--out', str(out)])
    assert json.loads(out.read_text())['families']['likelihood']['size'] == 100


def test_support_refuses_a_lane_whose_tables_are_not_both_named(tmp_path, capsys):
    rows_path = tmp_path / 'variant_table.json'
    rows = cohort_rows()
    rows_path.write_text(json.dumps({'schema': lanes.ROWS_SCHEMA, 'samples': rows}))
    couplings = tmp_path / 'potts.json'
    couplings.write_text(json.dumps(score_table('couplings', rows)))
    with pytest.raises(SystemExit):
        lanes.main(['support', '--lane', 'stability', '--rows', str(rows_path),
                    '--couplings', str(couplings)])
    lanes.main(['support', '--lane', 'interaction', '--rows', str(rows_path),
                '--couplings', str(couplings)])
    record = json.loads(capsys.readouterr().out)
    assert record['schema'] == lanes.SUPPORT_SCHEMA and record['rows_retained'] == 12


def test_panel_reads_a_cells_document_and_writes_the_family(tmp_path):
    groups = [f'g{index:02d}' for index in range(12)]
    cells = cell_matrix('phenotype', 'activity_venus-ranking', groups)
    document = {'schema': lanes.CELLS_SCHEMA, 'lane': 'phenotype',
                'stratum': 'activity_venus-ranking', 'channel': 'likelihood',
                'support_sha256': 'a' * 64, 'groups': groups,
                'cells': [{'rung': rung, 'arm': arm,
                           'support_sha256': value['support_sha256'],
                           **{name: np.asarray(value[name]).tolist() for name in lanes.ARM_NAMES}}
                          for (rung, arm), value in cells.items()]}
    path = tmp_path / 'cells.json'
    path.write_text(json.dumps(document))
    out = tmp_path / 'panel.json'
    lanes.main(['panel', '--lane', 'phenotype', '--cells', str(path), '--out', str(out)])
    record = json.loads(out.read_text())
    assert record['schema'] == lanes.PANEL_SCHEMA and record['status'] == 'complete'
    assert record['family']['size'] == 100 and len(record['states']) == 100
    document['schema'] = 'explicit_statistics_lane_cells_v2'
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='is not one of'):
        lanes.main(['panel', '--lane', 'phenotype', '--cells', str(path)])


# --------------------------------------------------------------------------- #
# The reverse bound: the second half of a two-sided statement.
# --------------------------------------------------------------------------- #

def test_the_reverse_bound_is_two_sided_and_read_against_the_model_arm():
    groups = [f'g{index:02d}' for index in range(24)]
    cells = cell_matrix('interaction', 'cycle_squared_error', groups, forward=0.001,
                        reverse=0.02, spread=0.004, seed=11,
                        digest='c' * 64)
    record = lanes.panel('interaction', 'cycle_squared_error', 'likelihood', cells,
                         groups=groups, support_sha256='c' * 64)
    bound = record['reverse_bound']
    assert bound['direction'] == 'controls_given_model'
    assert 'Overlap is a two-sided claim' in bound['reading']
    for entry in bound['columns']:
        # Both limits, and the denominator is the model arm's own performance.
        assert entry['lower_limit'] < entry['upper_limit']
        assert entry['fraction_of_model_performance'] == pytest.approx(
            entry['upper_limit'] / entry['model_performance'])
        assert entry['resolved_above_zero'] is (entry['lower_limit'] > 0)
    summary = bound['per_control_set']['nuisance']
    assert summary['arms'] == len(lanes.lane('interaction').arms)
    assert summary['smallest_lower_limit'] <= summary['largest_upper_limit']


def test_the_two_bounds_share_one_simultaneous_critical_value():
    groups = [f'g{index:02d}' for index in range(20)]
    cells = cell_matrix('stability', 'transfer_error', groups, forward=0.01, reverse=0.2,
                        digest='d' * 64)
    record = lanes.panel('stability', 'transfer_error', 'likelihood', cells,
                         groups=groups, support_sha256='d' * 64)
    critical = record['native']['joint_critical_max_absolute_deviation']
    # Every column is point +- the one family-wise critical value, so a limit read off
    # any column holds jointly with every other column's.
    for point, (low, high) in zip(record['point'], record['native']['interval']):
        assert high - point == pytest.approx(critical)
        assert point - low == pytest.approx(critical)
    assert 'simultaneous, not pointwise' in record['simultaneity']
    assert record['native']['family_size'] == record['family']['size']


def test_a_stratum_under_the_floor_withholds_both_bounds():
    groups = [f'g{index:02d}' for index in range(4)]
    cells = cell_matrix('phenotype', 'cellular_fitness_sge-ranking', groups, digest='e' * 64)
    record = lanes.panel('phenotype', 'cellular_fitness_sge-ranking', 'likelihood', cells,
                         groups=groups, support_sha256='e' * 64)
    assert record['status'] == lanes.UNRESOLVED
    for name in ('forward_bound', 'reverse_bound'):
        assert record[name]['status'] == lanes.UNRESOLVED
        assert record[name]['reason']


def test_each_panel_states_what_its_controls_are():
    groups = [f'g{index:02d}' for index in range(16)]
    cells = cell_matrix('abundance', 'squared_error', groups, rungs=('nuisance',),
                        digest='f' * 64)
    record = lanes.panel('abundance', 'squared_error', 'likelihood', cells, groups=groups,
                         support_sha256='f' * 64, role='sensitivity')
    # A reader who misses this takes the result for a claim about model quality.
    assert "target protein's own retained alignment" in record['control_provenance']
    assert 'scored zero-shot' in record['control_provenance']
    assert 'not a statement about model quality in general' in record['control_provenance']


def test_the_scope_conditions_are_declared_limitations():
    joined = ' '.join(lanes.LIMITATIONS)
    assert 'as read out by sequence likelihood' in joined
    assert 'not the\ninformation content of the parameters' in joined or \
        'not the information content of the parameters' in joined
    assert 'asymmetry in power' in joined
    assert 'a power asymmetry would matter only if the forward interval were' in joined


def test_a_reverse_fraction_above_one_is_flagged_not_hidden():
    # Where the model readout alone achieves almost nothing, the ratio exceeds one and
    # has stopped being a share of anything. It is flagged and the native limit stands.
    groups = [f'g{index:02d}' for index in range(16)]
    rng = np.random.default_rng(3)
    cells = {}
    for rung in lanes.lane('stability').rungs:
        for arm in lanes.lane('stability').arms:
            joint = rng.normal(0.50, 0.004, size=len(groups))
            cells[rung, arm] = {
                'U_C_M': joint.tolist(),
                'U_C': (joint - rng.normal(0.05, 0.004, len(groups))).tolist(),
                # The pure readout achieves almost no rank agreement of its own.
                'M': rng.normal(0.01, 0.004, size=len(groups)).tolist(),
                'support_sha256': '9' * 64}
    record = lanes.panel('stability', 'ranking', 'likelihood', cells, groups=groups,
                         support_sha256='9' * 64)
    bound = record['reverse_bound']
    flagged = [c for c in bound['columns'] if c['fraction_exceeds_denominator']]
    assert flagged, 'this fixture leaves the model arm almost no rank agreement'
    for entry in flagged:
        assert entry['fraction_of_model_performance'] > 1
        assert entry['upper_limit'] > entry['model_performance']
    per_rung = sum(1 for c in flagged if c['control_set'] == 'nuisance')
    assert bound['per_control_set']['nuisance']['arms_whose_fraction_exceeds_one'] == per_rung
    assert per_rung == len(lanes.lane('stability').arms)
    assert 'stops being a share' in bound['fraction_limitation']


# --------------------------------------------------------------------------- #
# Every number carries the rung it is an increment over.
# --------------------------------------------------------------------------- #

def test_a_bound_names_the_rung_and_its_columns_not_its_role():
    groups = [f'g{index:02d}' for index in range(16)]
    nuisance = lanes.control_blocks('stability', 'nuisance',
                                    ('ident', 'geom', 'chem', 'G', 'comp', 'prof2'))
    cells = cell_matrix('stability', 'ranking', groups, rungs=('nuisance',), digest='7' * 64)
    record = lanes.panel('stability', 'ranking', 'likelihood', cells, groups=groups,
                         support_sha256='7' * 64, role='sensitivity',
                         control_columns={'nuisance': {arm: nuisance
                                                       for arm in lanes.lane('stability').arms}})
    # The rung, its columns and the fact that it is not the archived set travel together.
    label = record['control_sets']['nuisance']
    assert label['control_set'] == 'nuisance' and label['is_archived_control_set'] is False
    assert label['carries_profile'] is False and label['carries_added_blocks'] == []
    assert label['columns'] == {'/'.join(nuisance): list(lanes.lane('stability').arms)}
    assert 'NOT the lane\'s archived control set' in label['reads_as']
    assert record['reads_as'] == label['reads_as']
    for name in ('forward_bound', 'reverse_bound'):
        held = record[name]['per_control_set']['nuisance']
        assert held['control_set'] == 'nuisance'
        assert held['reads_as'] == label['reads_as']


def test_the_archived_rung_says_that_it_is_the_archived_one():
    qualified = ('ident', 'geom', 'chem', 'G', 'comp', 'prof2')
    label = lanes.control_set_label('stability', lanes.ARCHIVED_RUNG,
                                    {'gpt2': lanes.control_blocks('stability',
                                                                  lanes.ARCHIVED_RUNG, qualified)})
    assert label['is_archived_control_set'] is True and label['carries_profile'] is True
    assert 'NOT the lane' not in label['reads_as']


def test_an_arm_conditional_rung_names_the_columns_only_some_arms_carry():
    frozen = ('ident', 'geom', 'comp', 'chem', 'prof', 'prof2', 'G')
    plain = lanes.control_blocks('abundance', lanes.ARCHIVED_RUNG, frozen)
    tokenised = lanes.control_blocks('abundance', lanes.ARCHIVED_RUNG, (*frozen, 'T'))
    label = lanes.control_set_label('abundance', lanes.ARCHIVED_RUNG,
                                    {'protgpt2': plain, 'gpt2': tokenised})
    assert label['arm_conditional_columns']['added_by_some_arms'] == ['T']
    assert 'for the arms whose archive qualified them' in label['reads_as']


def test_a_panel_without_the_columns_says_so_rather_than_guessing():
    groups = [f'g{index:02d}' for index in range(16)]
    cells = cell_matrix('stability', 'ranking', groups, rungs=('nuisance',), digest='8' * 64)
    record = lanes.panel('stability', 'ranking', 'likelihood', cells, groups=groups,
                         support_sha256='8' * 64, role='sensitivity')
    label = record['control_sets']['nuisance']
    assert label['columns'] is None
    assert 'were not supplied to this panel' in label['reads_as']
    assert label['control_set'] == 'nuisance'


def test_a_forward_fraction_above_one_is_flagged_like_the_reverse_one():
    # The model's bound exceeding the rung's own performance is the same situation as the
    # reverse case: still a true ratio, no longer a share. Both bounds flag it.
    groups = [f'g{index:02d}' for index in range(16)]
    rng = np.random.default_rng(17)
    cells = {}
    for arm in lanes.lane('stability').arms:
        joint = rng.normal(0.50, 0.004, size=len(groups))
        cells['nuisance', arm] = {
            'U_C_M': joint.tolist(),
            # The rung itself achieves almost no rank agreement of its own.
            'U_C': rng.normal(0.01, 0.004, size=len(groups)).tolist(),
            'M': rng.normal(0.02, 0.004, size=len(groups)).tolist(),
            'support_sha256': '6' * 64}
    record = lanes.panel('stability', 'ranking', 'likelihood', cells, groups=groups,
                         support_sha256='6' * 64, role='sensitivity')
    bound = record['forward_bound']
    flagged = [c for c in bound['columns'] if c['fraction_exceeds_denominator']]
    assert flagged, 'this fixture leaves the rung almost no rank agreement'
    for entry in flagged:
        assert entry['fraction_of_control_performance'] > 1
        assert entry['upper_limit'] > entry['control_performance']
    held = bound['per_control_set']['nuisance']
    assert held['arms_whose_fraction_exceeds_one'] == len(flagged)
    assert 'stops being a share' in bound['fraction_limitation']
    # And the flag is present and false on a panel whose denominators are healthy.
    healthy = cell_matrix('stability', 'ranking', groups, digest='5' * 64)
    other = lanes.panel('stability', 'ranking', 'likelihood', healthy, groups=groups,
                        support_sha256='5' * 64)
    assert all(c['fraction_exceeds_denominator'] is False
               for c in other['forward_bound']['columns'])


def test_every_added_block_names_a_producer_that_exists():
    # The producer is named in the refusal a reader gets when a wave has not landed, so a
    # name that resolves to nothing sends them looking for a file that never existed.
    root = pathlib.Path(lanes.__file__).resolve().parents[3]
    for role, source in lanes.BLOCK_SOURCE.items():
        producer = root / source['producer']
        assert producer.is_file(), f'{role}: {source["producer"]} does not exist'
