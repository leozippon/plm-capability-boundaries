"""The lane hierarchy, its families, its matched support and its four finding states.

Every fixture is synthetic, so these tests need neither the coupling wave nor the
GEMME wave. One test reads the real producer output where it is present, because the
refusal on a partial wave is the behaviour that matters most while that wave runs.
"""
import dataclasses
import json
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
                digest='a' * 64, arms=None, rungs=None, exact_forward=False):
    """Synthetic per-group statistics with a declared forward and reverse increment.

    The cells carry the stratum's own per-group statistic, which the panel converts to
    a loss with the stratum's declared sign. The joint arm is the reference; the control
    arm is ``forward`` worse and the pure readout ``reverse`` worse, so both directions
    have a known sign whatever the stratum reads. ``exact_forward`` lets the control arm
    share the joint arm's per-group draw, which fixes the forward reduction at exactly
    ``forward`` in every group.
    """
    declared = lanes.lane(lane)
    sign = lanes.stratum(lane, stratum).to_loss
    rng = np.random.default_rng(seed)
    cells = {}
    for rung in (rungs if rungs is not None else declared.rungs):
        for arm in (arms if arms is not None else declared.arms):
            joint = rng.normal(1.0, spread, size=len(groups))
            noise = 0. if exact_forward else rng.normal(0, spread, len(groups))
            cells[rung, arm] = {
                'U_C': sign * (joint + forward + noise),
                'M': sign * (joint + reverse + rng.normal(0, spread, len(groups))),
                'U_C_M': sign * joint,
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


def test_fit_plan_names_the_hierarchy_and_the_full_support_sensitivity():
    plan = lanes.fit_plan('stability', QUALIFIED['stability'])
    primary = [entry for entry in plan if entry['role'] == 'primary']
    sensitivity = [entry for entry in plan if entry['role'] == 'sensitivity']
    assert [entry['rung'] for entry in primary] == list(lanes.lane('stability').rungs)
    assert [entry['support'] for entry in primary] == ['complete_case'] * 5
    assert len(sensitivity) == 1 and sensitivity[0]['rung'] == 'nuisance'
    assert sensitivity[0]['support'] == 'full'
    assert [entry['rung'] for entry in plan if entry['reuses_archived_fit']] == \
        [lanes.ARCHIVED_RUNG]


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


def test_a_refused_row_whose_reference_is_not_enumerated_is_refused():
    rows = cohort_rows()
    statuses = {rows[0]['sample_id']: 'no_homolog_support'}
    table = score_table('couplings', rows, statuses=statuses)
    with pytest.raises(ValueError, match='does not enumerate as refusals'):
        lanes.complete_case('interaction', rows, {'couplings': table})


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
