"""Conditions that must always hold for the ProteinGym higher-order cube support.

The contrast itself is :func:`src.capability.mutation.higher_order_cycle.cycle_contrast`,
already fixed by ``tests/mutation/test_higher_order_cycle.py``; what is new here is the
ProteinGym addressing of it, the lower-order completeness requirement that decides
which variants carry a cycle at all, and the two-channel decomposition that
qualifies the endpoint.

One condition the gate would need is not instantiable and is recorded as absent
rather than satisfied: a held-out-label leakage test on a fitted readout. The
endpoint did not clear its measured noise floor, so no readout was fitted and no
model quantity was read. The leakage-adjacent property that *does* exist here is
tested instead -- the admitted support is a function of the measured state keys
and never of their values -- because that is the condition which, if it broke,
would let a label decide which cubes exist.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.capability.mutation.higher_order_cycle import cycle_contrast
from src.capability.mutation.proteingym_higher_order import (
    WILD_TYPE_STATE, AssayStates, census, channel_decomposition_bootstrap,
    channel_moments, complete_cubes, cube_addresses, cube_contrast,
    derive_channel_decomposition, genotype_rank, parse_mutant, split_channels,
)

#: Three substitutions at distinct positions, in position order.
A, B, C = (4, 'W'), (9, 'K'), (20, 'D')


def _additive_states(field: dict, *, offset: float = -3.5) -> dict:
    """Every subset of ``A, B, C``, scored by a strictly additive generator.

    ``offset`` stands in for the wild-type level, which is what an assay-wide
    additive constant is: a cycle of order one or more must not see it.
    """
    states = {}
    for subset in cube_addresses((A, B, C)).values():
        states[subset] = offset + sum(field[sub] for sub in subset)
    return states


def test_a_planted_third_order_term_is_recovered_with_its_sign():
    """The order-3 cycle returns the planted term, and swapping a corner flips it."""

    field = {A: 0.8, B: -1.4, C: 0.25}
    planted = 0.6
    states = _additive_states(field)
    states[(A, B, C)] += planted
    assert cube_contrast((A, B, C), states) == pytest.approx(planted)

    # The all-mutant corner enters positively, so lowering it lowers the contrast.
    states[(A, B, C)] -= 2 * planted
    assert cube_contrast((A, B, C), states) == pytest.approx(-planted)


def test_the_order_two_case_is_the_pairwise_cycle_and_nests_inside_the_cube():
    """omega is the difference of the pair's epsilon with and without the third state."""

    field = {A: 0.8, B: -1.4, C: 0.25}
    states = _additive_states(field)
    states[(A, B, C)] += 0.6
    with_c = cube_contrast((A, B), {key[:0] + tuple(s for s in key if s != C): value
                                    for key, value in states.items() if C in key})
    without_c = cube_contrast((A, B), {key: value for key, value in states.items()
                                       if C not in key})
    assert with_c - without_c == pytest.approx(cube_contrast((A, B, C), states))


def test_a_strictly_additive_generator_returns_the_construction_zero():
    """No additive field, and no wild-type offset, can manufacture a cycle."""

    field = {A: 0.8, B: -1.4, C: 0.25}
    for offset in (0.0, -3.5, 12.25):
        states = _additive_states(field, offset=offset)
        assert abs(cube_contrast((A, B, C), states)) < 1e-12
        assert abs(cube_contrast((A, B), {key: value for key, value in states.items()
                                          if C not in key})) < 1e-12


def test_a_strictly_pairwise_generator_has_no_third_order_cycle():
    """Pairwise couplings of any magnitude leave the order-3 contrast at zero."""

    field = {A: 0.8, B: -1.4, C: 0.25}
    coupling = {(A, B): 1.7, (A, C): -0.9, (B, C): 0.45}
    states = {}
    for subset in cube_addresses((A, B, C)).values():
        value = -3.5 + sum(field[sub] for sub in subset)
        for pair, strength in coupling.items():
            if pair[0] in subset and pair[1] in subset:
                value += strength
        states[subset] = value
    assert abs(cube_contrast((A, B, C), states)) < 1e-12


def test_a_partial_cube_has_no_contrast_rather_than_an_imputed_one():
    """A missing corner refuses; filling it with a prediction would report the prediction."""

    states = _additive_states({A: 0.8, B: -1.4, C: 0.25})
    del states[(B, C)]
    with pytest.raises(KeyError):
        cube_contrast((A, B, C), states)
    with pytest.raises(ValueError, match='requires all 8 corners'):
        cycle_contrast({corner: 0.0 for corner in
                        list(cube_addresses((A, B, C)))[:-1]})


def test_completeness_refuses_a_variant_whose_lower_order_support_is_incomplete():
    """The negative path: one missing pair, and the order-3 variant is not a cube."""

    keys = set(cube_addresses((A, B, C)).values())
    assert AssayStates.of(keys).has_complete_support((A, B, C))

    for missing in ((B, C), (A,)):
        thinned = AssayStates.of(keys - {missing})
        assert thinned.has_complete_support((A, B, C)) is False
        assert complete_cubes(thinned, 3) == []
        # The depth says *where* the support stops, which is the readable fact.
        assert thinned.support_depth((A, B, C)) == (1 if missing == (B, C) else 0)


def test_the_wild_type_corner_is_not_what_completeness_checks():
    """Completeness runs over non-empty subsets; the empty corner is checked separately.

    The processed ProteinGym CSVs carry no wild-type row, so a support rule that
    demanded it would admit nothing at all; a rule that silently imputed it would
    make every contrast report the imputation.
    """
    keys = set(cube_addresses((A, B, C)).values()) - {WILD_TYPE_STATE}
    assert AssayStates.of(keys).has_complete_support((A, B, C)) is True
    assert WILD_TYPE_STATE not in keys


def test_a_high_order_variant_without_its_singles_costs_no_enumeration():
    """Order 44 with no singles must not enumerate 1.76e13 corners.

    A random-mutagenesis library carries variants of that order. The check is exact
    rather than heuristic -- a level of ``comb(order, size)`` distinct subsets
    cannot be complete when the assay holds fewer states of that order in total --
    so the refusal is a measurement rather than a timeout.
    """
    deep = tuple((position, 'W') for position in range(44))
    assay = AssayStates.of({deep})
    assert assay.support_depth(deep) == 0
    assert assay.has_complete_support(deep) is False


def test_the_admitted_support_is_a_function_of_keys_and_never_of_values():
    """Leakage-adjacent condition: no label decides which cubes exist.

    The census and the completeness rule are handed state keys. Replacing every
    value, including with values whose ordering is reversed, cannot change the
    admitted support -- which is why the support could be declared and digested
    before any channel value was read.
    """
    keys = set(cube_addresses((A, B, C)).values())
    assay = AssayStates.of(keys)
    admitted = complete_cubes(assay, 3)
    counts = census(keys)

    for scale in (1.0, -1.0, 1e6):
        values = {key: scale * (len(key) + 0.5) for key in keys}
        assert set(values) == keys
        assert complete_cubes(AssayStates.of(values), 3) == admitted
        assert census(values) == counts


def test_parse_mutant_refuses_what_it_cannot_key():
    """Every refusal here is a row that would otherwise enter a cube silently."""

    key = parse_mutant('R60D:A29H')
    assert key.substitutions == ((28, 'H'), (59, 'D'))
    assert key.wild == ('A', 'R') and key.order == 2

    for bad, message in (
            ('', 'no wild-type row'),
            ('G114*', 'outside the canonical 20'),
            ('A0G', 'non-positive position'),
            ('AG', 'malformed'),
            ('AxG', 'no 1-based position'),
            ('A5A', 'for itself'),
            ('A5G:C5T', 'one position twice')):
        with pytest.raises(ValueError, match=message):
            parse_mutant(bad)


def test_the_decomposition_separates_a_shared_component_from_channel_noise():
    """A planted common signal and independent channel noise are recovered."""

    rng = np.random.default_rng(7)
    signal = rng.normal(0.0, 1.0, size=4000)
    left = signal + rng.normal(0.0, 0.25, size=4000)
    right = signal + rng.normal(0.0, 0.25, size=4000)
    derived = derive_channel_decomposition(channel_moments(left, right))
    assert derived['shared_component_sd'] == pytest.approx(1.0, abs=0.05)
    assert derived['per_channel_discordance_sd'] == pytest.approx(0.25, abs=0.02)
    assert derived['shared_to_discordance_ratio'] > 3.0

    # Two channels of pure independent noise carry no reproducible component.
    noise = derive_channel_decomposition(channel_moments(
        rng.normal(0.0, 1.0, size=4000), rng.normal(0.0, 1.0, size=4000)))
    assert noise['shared_component_sd'] < 0.2
    assert noise['shared_to_discordance_ratio'] < 0.25


def test_the_decomposition_resamples_groups_and_respects_the_unit_floor():
    """Below the package's unit floor the record carries no interval, and says why."""

    rng = np.random.default_rng(11)
    for units, expect_interval in ((4, False), (12, True)):
        groups = [(index, index + 1) for index in range(units) for _ in range(5)]
        signal = rng.normal(0.0, 1.0, size=len(groups))
        record = channel_decomposition_bootstrap(
            signal + rng.normal(0, 0.2, len(groups)),
            signal + rng.normal(0, 0.2, len(groups)),
            groups, seed=3, resamples=200)
        assert record['units'] == units
        assert record['degenerate'] is not expect_interval
        has_interval = record['estimates']['shared_component_sd'].get('ci95') is not None
        assert has_interval is expect_interval
        if not expect_interval:
            assert 'below the 8-unit floor' in record['degenerate_reason']

    with pytest.raises(ValueError, match='channels and groups must align'):
        channel_decomposition_bootstrap([1.0, 2.0], [1.0], [(0, 1), (1, 2)],
                                        seed=1, resamples=10)


def test_a_replicate_channel_split_is_ordered_by_identifier_and_not_by_value():
    """The split must not be a function of the measurement it is used to check.

    Replicates arrive in whatever order a file lists them. Ordering them by a
    digest of their own identifier makes the two channels a property of the
    library; ordering them by value would put the high measurements in one channel
    and manufacture a channel offset out of nothing.
    """
    replicates = [('ttga', 0.9), ('acgt', 0.1), ('ggca', 0.5), ('cctt', 0.3)]
    ordered = [value for _, value in sorted(replicates, key=lambda r: genotype_rank(r[0]))]
    shuffled = [value for _, value in
                sorted(reversed(replicates), key=lambda r: genotype_rank(r[0]))]
    assert ordered == shuffled
    assert split_channels(ordered) == split_channels(shuffled)

    # Ordering by value instead would separate the channels by construction.
    by_value = sorted(value for _, value in replicates)
    left, right, _, _, _ = split_channels(by_value)
    assert left < right
    honest_left, honest_right, _, _, _ = split_channels(ordered)
    assert abs(honest_left - honest_right) < abs(right - left)


def test_the_channel_split_reports_the_spread_that_propagates_the_floor():
    """The returned variance is the individual replicates', not the channel means'.

    A channel mean over ``n`` replicates has variance ``variance / n``; an order-k
    cycle's floor is the sum of those over its ``2 ** k`` corners. Returning the
    channel means' own spread instead would understate the floor by the averaging
    the channels already did.
    """
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    left, right, variance, size_left, size_right = split_channels(values)
    assert left == pytest.approx(3.0) and right == pytest.approx(3.0)
    assert (size_left, size_right) == (3, 2)
    assert variance == pytest.approx(float(np.var(values, ddof=1)))

    with pytest.raises(ValueError, match='at least two replicates'):
        split_channels([1.0])
