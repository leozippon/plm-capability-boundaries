"""Conditions the order-k qualification harness must always hold, and its refusals.

The harness exists because two higher-order gates closed on endpoint
qualification rather than on model measurement, so the conditions tested here
are the ones whose failure would have let a noise-limited endpoint reach a GPU:
the independent unit is the site tuple and never the cube, the channel kind's
lower-bound status is carried rather than described, the order-k floor grows
with order, clearance rests on the conservative reading, and the default is to
stop.
"""
from __future__ import annotations


import numpy as np
import pytest

from src.capability.mutation import higher_order_qualification as hoq
from src.capability.mutation.proteingym_higher_order import AssayStates, cube_addresses
from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS


def substitutions(positions, residue='A'):
    return tuple((position, residue) for position in positions)


def complete_block(positions, *, order, channels=('r1', 'r2'), noise=0.0, planted=0.0,
                   seed=0, reference=None):
    """A block whose every subset of ``positions`` is measured, with a planted term.

    The generator is additive apart from a per-state interaction of scale
    ``planted`` added to every state of exactly ``order`` substitutions, so the
    order-k cycle of such a cube is that state's own interaction before noise.
    The interaction has to *vary* across cubes rather than be a constant,
    because the between-channel decomposition measures the reproducible
    variation of the contrast and a constant contrast has none.
    """
    rng = np.random.default_rng(seed)
    letters = reference or 'C' * (max(positions) + 1)
    singles = {position: float(rng.normal()) for position in positions}
    values = {}
    from itertools import combinations
    for size in range(0, len(positions) + 1):
        for chosen in combinations(sorted(positions), size):
            key = substitutions(chosen)
            base = sum(singles[position] for position in chosen)
            if len(chosen) == order:
                base += planted * float(rng.normal())
            values[key] = tuple(base + noise * float(rng.normal()) for _ in channels)
    return hoq.CandidateBlock(block_id='b', reference=letters, channels=tuple(channels),
                              values=values)


# --------------------------------------------------------------------------- #
# The unit is the site tuple, and the ceiling is printed beside it.
# --------------------------------------------------------------------------- #

def test_the_independent_unit_is_the_site_tuple_and_not_the_cube():
    """A four-position library carries C(4, 3) = 4 triples however many cubes."""
    positions = (0, 1, 2, 3)
    states = []
    from itertools import combinations
    for size in range(0, 5):
        for chosen in combinations(positions, size):
            for residue in 'AGS':
                states.append(tuple((position, residue) for position in chosen))
    inventory = hoq.support_inventory(AssayStates.of(states), orders=(3,),
                                      variable_positions=4)
    order3 = inventory['per_order']['3']
    assert order3['cubes'] > order3['site_tuples']
    assert order3['site_tuples'] == 4 == hoq.combinatorial_ceiling(4, 3)
    assert order3['site_tuple_ceiling'] == 4
    assert order3['positions'] == 4
    assert order3['unit_floor']['degenerate'] is True


def test_the_site_tuple_ceiling_is_the_combinatorial_bound():
    assert hoq.combinatorial_ceiling(4, 3) == 4
    assert hoq.combinatorial_ceiling(7, 3) == 35
    assert hoq.combinatorial_ceiling(2, 3) == 0
    with pytest.raises(ValueError):
        hoq.combinatorial_ceiling(-1, 3)


def test_the_support_depth_census_separates_absent_from_one_level_short():
    """"No cube" and "cubes one level short" are different findings about a library."""
    incomplete = AssayStates.of([(), ((0, 'A'),), ((1, 'A'),), ((2, 'A'),),
                                 ((0, 'A'), (1, 'A')), ((0, 'A'), (2, 'A')),
                                 ((0, 'A'), (1, 'A'), (2, 'A'))])
    census = hoq.support_depth_census(incomplete, 3)
    assert census == {'1': 1}
    with pytest.raises(ValueError):
        hoq.support_depth_census(incomplete, 1)


def test_the_support_inventory_reads_no_value():
    """The admitted support cannot be a function of the labels it will carry.

    Asserted rather than described: the inventory is handed only state keys, so
    replacing every value with a constant leaves it byte-identical.
    """
    block = complete_block((0, 1, 2), order=3, planted=1.0, seed=1)
    flat = hoq.CandidateBlock(block_id='b', reference=block.reference,
                              channels=block.channels,
                              values={key: (0.0, 0.0) for key in block.values})
    assert (hoq.support_inventory(block.states(), orders=(2, 3), variable_positions=3)
            == hoq.support_inventory(flat.states(), orders=(2, 3), variable_positions=3))


# --------------------------------------------------------------------------- #
# The channel kind, and the count-derived lower bound.
# --------------------------------------------------------------------------- #

def test_an_undeclared_channel_kind_is_refused():
    with pytest.raises(ValueError, match='undeclared channel kind'):
        hoq.channel_kind('counts_maybe')


def test_a_count_derived_floor_never_grants_clearance():
    """Clearing a lower bound on total noise is not a qualification.

    The Wu 2016 support cleared its counting-noise floor by a factor of 6.49 at
    order three and was still not qualified, because counting noise covers
    sequencing and library sampling only. A harness that granted clearance there
    would schedule a run on an unqualified endpoint.
    """
    block = complete_block(tuple(range(6)), order=2, planted=5.0, noise=0.001, seed=2)
    floor = hoq.order_floor(block, 2, seed=0, resamples=200)
    replicate = hoq.clearance(floor, 'replicate')
    counted = hoq.clearance(floor, 'count')
    assert replicate['clears'] is True
    assert counted['clears'] is False
    assert counted['channel']['floor_is_lower_bound'] is True
    assert any('lower bound' in reason for reason in counted['refusals'])
    for kind in ('dispersion', 'none'):
        assert hoq.clearance(floor, kind)['clears'] is False


# --------------------------------------------------------------------------- #
# The order-k floor.
# --------------------------------------------------------------------------- #

def test_the_propagated_floor_grows_as_the_square_root_of_the_corner_count():
    """An order-k cycle sums 2**k corner estimates where a pairwise cycle sums four."""
    assert hoq.propagated_order_floor(2, 1.0) == pytest.approx(2.0)
    assert hoq.propagated_order_floor(3, 1.0) == pytest.approx(np.sqrt(8))
    assert hoq.propagated_order_floor(5, 0.5) == pytest.approx(0.5 * np.sqrt(32))
    assert (hoq.propagated_order_floor(3, 1.0)
            == pytest.approx(np.sqrt(2) * hoq.propagated_order_floor(2, 1.0)))
    with pytest.raises(ValueError):
        hoq.propagated_order_floor(0, 1.0)
    with pytest.raises(ValueError):
        hoq.propagated_order_floor(2, float('nan'))


def test_the_measured_floor_tracks_the_propagated_one_on_independent_noise():
    """With independent corner errors the measured discordance meets the prediction.

    This is what makes the ratio ``measured_over_propagated`` a statement about
    the correlation of corner errors rather than a free parameter: planting
    independent noise recovers one, and real data comes in below it.
    """
    block = complete_block(tuple(range(7)), order=3, noise=0.05, seed=3)
    floor = hoq.order_floor(block, 3, seed=0, resamples=200)
    assert floor['measured_over_propagated'] == pytest.approx(1.0, abs=0.15)
    assert floor['propagated_floor'] > 0


def test_the_pooled_label_reading_is_the_generous_one_and_scales_by_sqrt_n():
    assert hoq.pooled_label_factor(2) == pytest.approx(np.sqrt(2))
    assert hoq.pooled_label_factor(3) == pytest.approx(np.sqrt(3))
    with pytest.raises(ValueError):
        hoq.pooled_label_factor(1)
    block = complete_block(tuple(range(6)), order=2, planted=0.3, noise=0.2,
                           channels=('r1', 'r2', 'r3'), seed=4)
    record = hoq.clearance(hoq.order_floor(block, 2, seed=0, resamples=200), 'replicate')
    factor = record['pooled_label_reading']['factor']
    assert factor == pytest.approx(np.sqrt(3))
    assert (record['pooled_label_reading']['point']
            == pytest.approx(record['one_channel_reading']['point'] * factor))
    assert (record['pooled_label_reading']['ci95'][0]
            == pytest.approx(record['one_channel_reading']['ci95'][0] * factor))
    assert record['pooled_label_reading']['ci95'][1] >= record['one_channel_reading']['ci95'][1]


def test_the_primary_channel_pair_is_declared_and_not_chosen_on_an_outcome():
    """The reading the verdict rests on is fixed by a digest of the channel names."""
    names = ('alpha', 'beta', 'gamma')
    assert hoq.channel_order(names) == hoq.channel_order(names)
    reordered = hoq.channel_order(('gamma', 'beta', 'alpha'))
    assert sorted(reordered) == [0, 1, 2]
    block = complete_block((0, 1, 2), order=2, noise=0.1, channels=names, seed=5)
    floor = hoq.order_floor(block, 2, seed=0, resamples=100)
    first, second = floor['channel_order'][:2]
    assert floor['primary_pair'] == f'{first}|{second}'
    assert len(floor['channel_pairs']) == 3
    assert floor['primary_pair'] in floor['channel_pairs']


def test_a_floor_needs_two_channels_and_a_complete_cube():
    single = complete_block((0, 1, 2), order=2, channels=('only',), seed=6)
    with pytest.raises(ValueError, match='two channels'):
        hoq.order_floor(single, 2, seed=0, resamples=10)
    block = complete_block((0, 1, 2), order=2, seed=7)
    with pytest.raises(ValueError, match='no complete cube'):
        hoq.order_floor(block, 4, seed=0, resamples=10)


def test_a_block_refuses_a_state_missing_a_channel():
    with pytest.raises(ValueError, match='channels'):
        hoq.CandidateBlock(block_id='b', reference='CC', channels=('r1', 'r2'),
                           values={(): (0.0,)})
    with pytest.raises(ValueError, match='distinct'):
        hoq.CandidateBlock(block_id='b', reference='CC', channels=('r1', 'r1'),
                           values={(): (0.0, 0.0)})


# --------------------------------------------------------------------------- #
# Clearance, and the refusal that is the default.
# --------------------------------------------------------------------------- #

def test_a_thin_support_withholds_its_interval_and_therefore_cannot_clear():
    """Below the shared unit floor there is no interval, so nothing resolves.

    Two site pairs once carried a point estimate of 1.06 and no interval, and
    the harness must not read that as a clearance.
    """
    block = complete_block((0, 1, 2), order=2, planted=9.0, noise=0.0001, seed=8)
    floor = hoq.order_floor(block, 2, seed=0, resamples=200)
    assert floor['units']['site_tuples'] == 3 < MINIMUM_BOOTSTRAP_UNITS
    record = hoq.clearance(floor, 'replicate')
    assert record['one_channel_reading']['ci95'] is None
    assert record['clears'] is False
    assert record['clears_generous'] is False
    assert any('below the 8-' in reason for reason in record['refusals'])


def test_a_noise_limited_endpoint_does_not_clear_and_names_the_limitation():
    block = complete_block(tuple(range(7)), order=3, planted=0.0, noise=0.2, seed=9)
    record = hoq.clearance(hoq.order_floor(block, 3, seed=0, resamples=300), 'replicate')
    assert record['clears'] is False
    assert record['one_channel_reading']['ci95'][0] <= hoq.CLEARANCE_RATIO
    verdict = hoq.candidate_verdict('candidate', [record], declaration_digest='d',
                                    declared_orders=(3,), blocks=[block])
    assert verdict['recommendation'] == 'stop'
    assert verdict['schedule_model_scoring'] is False
    assert 'measurement limitation' in verdict['per_order']['3']['measurement_limitation']
    with pytest.raises(ValueError, match='does not clear'):
        hoq.assert_scheduling_admitted(verdict, order=3)


def distinct_family_blocks(count, *, order, planted, noise, seed):
    """``count`` blocks whose references share no more than chance identity."""
    blocks = []
    for index in range(count):
        reference = ''.join('ACDEFGHIKLMNPQRSTVWY'[(index * 7 + position) % 20]
                            for position in range(40))
        drawn = complete_block(tuple(range(7)), order=order, planted=planted, noise=noise,
                               seed=seed + index, reference=reference)
        blocks.append(hoq.CandidateBlock(block_id=f'fam{index}', reference=reference,
                                         channels=drawn.channels, values=drawn.values))
    return blocks


def test_clearance_at_one_order_licenses_no_scoring_at_another():
    """The distinction both closed gates turned on, enforced rather than described."""
    clearing = distinct_family_blocks(MINIMUM_BOOTSTRAP_UNITS, order=2, planted=2.0,
                                      noise=0.05, seed=10)
    failing = complete_block(tuple(range(7)), order=3, planted=0.0, noise=0.3, seed=11,
                             reference='W' * 40)
    cells = [hoq.clearance(hoq.order_floor(block, 2, seed=0, resamples=200), 'replicate')
             for block in clearing]
    cells.append(hoq.clearance(hoq.order_floor(failing, 3, seed=0, resamples=300),
                               'replicate'))
    verdict = hoq.candidate_verdict('candidate', cells, declaration_digest='d',
                                    declared_orders=(2, 3), blocks=clearing + [failing])
    assert verdict['clearing_orders'] == [2]
    hoq.assert_scheduling_admitted(verdict, order=2)
    with pytest.raises(ValueError, match='order-3'):
        hoq.assert_scheduling_admitted(verdict, order=3)
    with pytest.raises(ValueError, match='declares no order-4'):
        hoq.assert_scheduling_admitted(verdict, order=4)


def test_the_generalisation_unit_is_the_family_and_not_the_site_tuple():
    """Two units govern an order-k endpoint, and conflating them schedules a run.

    858 site pairs inside one domain clears a noise floor and still cannot carry
    a fit: the frozen held-group design has no second group to hold out. The
    census counts families, not tuples, and says when its count is an upper bound.
    """
    one = complete_block(tuple(range(7)), order=2, seed=20, reference='C' * 20)
    twin = hoq.CandidateBlock(block_id='near', reference='C' * 18 + 'AA',
                              channels=one.channels, values=dict(one.values))
    far = hoq.CandidateBlock(block_id='far', reference='G' * 20,
                             channels=one.channels, values=dict(one.values))
    shorter = hoq.CandidateBlock(block_id='short', reference='C' * 12,
                                 channels=one.channels, values=dict(one.values))
    blocks = [one, twin, far, shorter]
    census = hoq.generalisation_unit_census(blocks, ['b', 'near'])
    assert census['family_groups'] == 1
    assert census['family_group_members'] == [['b', 'near']]
    assert census['pairs'][0]['identity'] == pytest.approx(0.9)
    assert census['pairs'][0]['one_family'] is True
    assert census['group_count_is_upper_bound'] is False
    assert census['unit_floor']['degenerate'] is True
    assert census['unit'].startswith('wild-type family group')

    split = hoq.generalisation_unit_census(blocks, ['b', 'far'])
    assert split['family_groups'] == 2 and split['pairs'][0]['one_family'] is False

    # An unalignable pair is counted as distinct, which makes the count an upper
    # bound: an alignment could only merge further, so a refusal on it is sound.
    bounded = hoq.generalisation_unit_census(blocks, ['b', 'short'])
    assert bounded['family_groups'] == 2
    assert bounded['group_count_is_upper_bound'] is True
    assert bounded['pairs'][0]['identity'] is None

    assert hoq.generalisation_unit_census(blocks, [])['family_groups'] == 0
    with pytest.raises(ValueError, match='absent from the candidate'):
        hoq.generalisation_unit_census(blocks, ['nowhere'])
    with pytest.raises(ValueError, match='fraction'):
        hoq.generalisation_unit_census(blocks, ['b'], identity_threshold=0.0)


def test_clearing_the_floor_in_one_family_schedules_nothing():
    """The refusal that decides the Escobedo candidate at order two."""
    first = complete_block(tuple(range(7)), order=2, planted=2.0, noise=0.05, seed=21,
                           reference='C' * 20)
    second = hoq.CandidateBlock(block_id='sibling', reference='C' * 18 + 'AA',
                                channels=first.channels, values=dict(first.values))
    cells = [hoq.clearance(hoq.order_floor(block, 2, seed=0, resamples=300), 'replicate')
             for block in (first, second)]
    assert all(cell['clears'] for cell in cells)
    verdict = hoq.candidate_verdict('candidate', cells, declaration_digest='d',
                                    declared_orders=(2,), blocks=[first, second])
    order = verdict['per_order']['2']
    assert order['noise_floor_cleared'] is True
    assert order['generalisation_units_sufficient'] is False
    assert order['generalisation_units']['family_groups'] == 1
    assert verdict['clearing_orders'] == []
    assert verdict['orders_clearing_the_noise_floor_only'] == [2]
    assert verdict['recommendation'] == 'stop'
    assert 'generalises over' in order['measurement_limitation']
    with pytest.raises(ValueError, match='does not clear'):
        hoq.assert_scheduling_admitted(verdict, order=2)


def test_enough_families_admit_scheduling():
    """The positive path, so the new refusal is not vacuous."""
    blocks = distinct_family_blocks(MINIMUM_BOOTSTRAP_UNITS, order=2, planted=2.0,
                                    noise=0.05, seed=30)
    cells = [hoq.clearance(hoq.order_floor(block, 2, seed=0, resamples=200), 'replicate')
             for block in blocks]
    verdict = hoq.candidate_verdict('candidate', cells, declaration_digest='d',
                                    declared_orders=(2,), blocks=blocks)
    order = verdict['per_order']['2']
    assert order['generalisation_units']['family_groups'] >= MINIMUM_BOOTSTRAP_UNITS
    assert order['generalisation_units_sufficient'] is True
    assert verdict['clearing_orders'] == [2]
    hoq.assert_scheduling_admitted(verdict, order=2)


def test_two_clearing_cells_from_one_duplicated_library_are_refused():
    """Two tables of one library are not two units of evidence."""
    block = complete_block(tuple(range(7)), order=2, planted=2.0, noise=0.05, seed=12)
    twin = hoq.CandidateBlock(block_id='twin', reference=block.reference,
                              channels=block.channels, values=dict(block.values))
    groups = hoq.duplicate_sequence_blocks([block, twin])
    assert groups == [['b', 'twin']]
    first = hoq.clearance(hoq.order_floor(block, 2, seed=0, resamples=300), 'replicate')
    second = dict(first, block_id='twin')
    assert first['clears'] and second['clears']
    with pytest.raises(ValueError, match='duplicated block group'):
        hoq.candidate_verdict('candidate', [first, second], declaration_digest='d',
                              declared_orders=(2,), blocks=[block, twin],
                              duplicate_groups=groups)


def test_the_shared_sequence_census_measures_the_overlap_it_judges_on():
    block = complete_block((0, 1, 2), order=2, seed=13)
    disjoint = hoq.CandidateBlock(
        block_id='other', reference='GGGG', channels=block.channels,
        values={(): (0.0, 0.0), ((3, 'A'),): (1.0, 1.0)})
    census = hoq.shared_sequence_census([block, disjoint])
    assert census == []
    partial = hoq.CandidateBlock(
        block_id='partial', reference=block.reference, channels=block.channels,
        values={key: block.values[key] for key in list(block.values)[:2]})
    census = hoq.shared_sequence_census([block, partial])
    assert len(census) == 1
    assert census[0]['share_of_smaller'] == pytest.approx(1.0)
    assert census[0]['one_library'] is True


# --------------------------------------------------------------------------- #
# The candidate adapter.
# --------------------------------------------------------------------------- #

def test_a_state_key_is_derived_from_the_sequences_and_not_from_an_order_column():
    """A deposition's own order and indel columns are not trusted.

    MegaScale's ``aa_seq`` is truncated to wild-type length, so an insertion
    reads as a run of substitutions; one of this candidate's tables marks every
    one of its 9,973 equal-length rows ``indel``. Both are why the key comes
    from the sequences.
    """
    assert hoq.substitution_key('CCCC', 'CACC') == ((1, 'A'),)
    assert hoq.substitution_key('CCCC', 'CCCC') == ()
    with pytest.raises(ValueError, match='equal-length'):
        hoq.substitution_key('CCCC', 'CCC')
    with pytest.raises(ValueError, match='canonical'):
        hoq.substitution_key('CCCC', 'CXCC')


def test_dimsum_blocks_split_by_length_and_require_a_measured_reference():
    rows = [
        {'aa_seq': 'CCCC', 'WT': 'True', 'f1': '0.0', 'f2': '0.1'},
        {'aa_seq': 'CACC', 'WT': '', 'f1': '1.0', 'f2': '1.1'},
        {'aa_seq': 'CCAC', 'WT': '', 'f1': '2.0', 'f2': '2.1'},
        {'aa_seq': 'CAAC', 'WT': '', 'f1': '3.0', 'f2': '3.1'},
        # A second protein: its own length block, and no measured reference.
        {'aa_seq': 'GGGGG', 'WT': '', 'f1': '1.0', 'f2': '1.0'},
        {'aa_seq': 'AGGGG', 'WT': '', 'f1': '1.0', 'f2': '1.0'},
        {'aa_seq': 'GAGGG', 'WT': '', 'f1': '1.0', 'f2': '1.0'},
    ]
    parsed = hoq.dimsum_blocks(rows, channel_columns=('f1', 'f2'))
    admitted = {block.block_id: block for block in parsed['blocks']}
    assert set(admitted) == {'len4', 'len5'}
    assert admitted['len4'].reference == 'CCCC'
    assert () in admitted['len4'].values
    assert admitted['len4'].values[((1, 'A'),)] == (1.0, 1.1)


def test_dimsum_blocks_exclude_and_count_rather_than_repair():
    rows = [
        {'aa_seq': 'CCCC', 'WT': 'TRUE', 'f1': '0.0', 'f2': '0.0'},
        {'aa_seq': 'CACC', 'WT': '', 'f1': '1.0', 'f2': '1.0'},
        {'aa_seq': 'CACC', 'WT': '', 'f1': '9.0', 'f2': '9.0'},
        {'aa_seq': 'CXCC', 'WT': '', 'f1': '1.0', 'f2': '1.0'},
        {'aa_seq': 'CCAC', 'WT': '', 'f1': 'NA', 'f2': '1.0'},
        {'aa_seq': 'CCCA', 'WT': '', 'f1': '1.0', 'f2': 'nan'},
    ]
    parsed = hoq.dimsum_blocks(rows, channel_columns=('f1', 'f2'))
    block = parsed['blocks'][0]
    assert parsed['excluded_rows'] == {'noncanonical_residue': 1, 'nonfinite_channel': 2,
                                       'duplicate_state': 1}
    assert set(block.values) == {(), ((1, 'A'),)}
    assert block.values[((1, 'A'),)] == (1.0, 1.0)


def test_a_block_whose_reference_is_unmeasured_is_refused_with_its_reason():
    rows = [{'aa_seq': 'AGGG', 'WT': '', 'f1': '1.0'},
            {'aa_seq': 'GAGG', 'WT': '', 'f1': '1.0'},
            {'aa_seq': 'GGAG', 'WT': '', 'f1': '1.0'},
            {'aa_seq': 'GGGA', 'WT': '', 'f1': '1.0'}]
    parsed = hoq.dimsum_blocks(rows, channel_columns=('f1',))
    assert parsed['blocks'] == []
    refused = parsed['block_summary'][0]
    assert refused['admitted'] is False
    assert 'centred-residual' in refused['reason']


def test_a_declaration_carries_the_support_and_the_channel_kind():
    block = complete_block(tuple(range(6)), order=3, noise=0.05, seed=14)
    declaration = hoq.declare_candidate(
        'c', blocks=[block], orders=(2, 3), channel_kind_name='replicate',
        provenance={'source': 'test'})
    assert declaration['channel']['supports_measured_floor'] is True
    assert declaration['reads_no_value'] is True
    assert declaration['blocks']['b']['wild_type_state_measured'] is True
    assert declaration['blocks']['b']['per_order']['3']['cubes'] == 20
    with pytest.raises(ValueError, match='at least one admitted block'):
        hoq.declare_candidate('c', blocks=[], orders=(2,), channel_kind_name='replicate',
                              provenance={})


def test_the_cube_contrast_is_the_shared_one_and_not_a_second_definition():
    """The contrast and its sign convention are reused, not restated here."""
    cube = ((0, 'A'), (1, 'A'), (2, 'A'))
    from src.capability.mutation.proteingym_higher_order import cube_contrast
    additive = complete_block((0, 1, 2), order=3, planted=0.0, seed=16)
    values = {state: additive.values[state][0] for state in cube_addresses(cube).values()}
    assert cube_contrast(cube, values) == pytest.approx(0.0, abs=1e-12)
    planted = dict(values)
    planted[cube] = values[cube] + 0.75
    assert cube_contrast(cube, planted) == pytest.approx(0.75)
