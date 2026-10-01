"""Qualifying a candidate dataset for an order-k interaction endpoint, before any model runs.

Why this exists. The higher-order gate has closed twice as *unresolved rather
than not-detected*, and both closures came from qualifying the endpoint rather
than from measuring a model. MegaScale ``dataset2`` carries zero substitution
variants of order three or above, every apparent one being an indel construct
whose ``aa_seq`` is truncated to wild-type length. ProteinGym carries 206,390
complete cubes, and its best-qualified support reaches a shared-to-discordance
ratio of 0.276 [0.000, 0.496] at 119.6 Kish effective site triples on 598 site
triples, against the admitted pairwise endpoint's 2.78 [2.34, 3.26] at a
comparable 121.6 -- while on that same assay the order-2 shared component *is*
resolved above zero at 0.0596 [0.0232, 0.0800] scaled-fitness units. The
third-order failure there is an order property of a well-powered assay's labels,
not an assay property.

Both closures asked the same five questions in the same order, and answered them
with code written twice. This module asks them once, of any candidate:

1. does the candidate carry **complete lower-order support**, and how much;
2. how many **independent site tuples across how many positions** -- never cube
   counts, because 25,035 complete triples once rested on the C(4, 3) = 4 site
   triples a four-position library has;
3. does it carry a **genuine replicate channel** or only counts, with the
   distinction stated, because a count-derived floor bounds sequencing and
   library sampling only and is therefore a *lower bound* on total noise;
4. what is the **order-k noise floor**, measured and propagated; and
5. does the **endpoint clear that floor**.

Stop is the default. :func:`candidate_verdict` schedules model scoring only when
every condition holds, and :func:`assert_scheduling_admitted` raises otherwise,
so a caller cannot read a permissive default off a record that did not qualify.

No model score, likelihood or representation enters anything here. The cube
contrast and its sign convention come from
:mod:`src.capability.mutation.higher_order_cycle`; the completeness rule, the cube
addressing and the two-channel decomposition from
:mod:`src.capability.mutation.proteingym_higher_order`. Neither is reimplemented.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from math import comb, sqrt

import numpy as np

from src.capability.mutation.higher_order_cycle import Substitution, kish_effective_units
from src.capability.mutation.proteingym_higher_order import (
    RESIDUES, AssayStates, channel_decomposition_bootstrap, complete_cubes,
    cube_addresses, cube_contrast, genotype_rank)
from src.capability.core.statistics import bootstrap_unit_floor

#: The ratio a candidate's endpoint has to resolve above before any model
#: quantity is read. The shared-across-channel component is an **upper** bound
#: on reproducible interaction signal, because error common to both channels
#: contributes to their covariance; the per-channel discordance is a **lower**
#: bound on per-channel measurement noise, because error common to both channels
#: cancels in their difference. A ratio of an upper bound over a lower bound
#: below one therefore supports no statement that the contrast carries
#: reproducible signal at all, which is why the threshold is one rather than a
#: tuned value. The admitted pairwise endpoint stands at 2.78 [2.34, 3.26] and
#: the two failed higher-order endpoints at 1.06 (no interval) and
#: 0.276 [0.000, 0.496].
CLEARANCE_RATIO = 1.0

#: The admitted pairwise endpoint, carried as the comparison every order-k
#: qualification is read against rather than restated per artefact. Source:
#: archive/docs/R3/pairwise-epistasis-results.md.
PAIRWISE_REFERENCE: dict[str, object] = {
    'endpoint': 'order-2 cycle on the frozen MegaScale pairwise cohort',
    'shared_to_discordance_ratio': 2.78,
    'ci95': [2.34, 3.26],
    'effective_units_kish': 121.6,
    'unit': 'site pair, weighted equally per wild-type family group',
    'family_groups': 64,
    'comparability': (
        'This Kish count is over site pairs weighted equally within 64 wild-type '
        'family groups, so it is a *within-group* effective count on a 64-group '
        'support. A site-pair count taken inside one domain is not the same '
        'quantity and is not comparable with it: the generalisation unit of the '
        'model comparison is the family group, and its count is what '
        ':func:`generalisation_unit_census` measures.'),
}

#: Identity at or above which two library references are one wild-type family
#: group, matching the pairwise cohort's single-linkage grouping contract.
FAMILY_IDENTITY_THRESHOLD = 0.5

#: What each declared channel kind supports. ``replicate`` is the only kind that
#: supports a measured floor: independent repeats of the same state, whose
#: disagreement is a measurement of the noise the contrast has to clear.
#: ``dispersion`` is a per-variant spread with no second channel, so only a
#: propagated floor is available and no between-channel decomposition exists.
#: ``count`` is a multinomial model of sequencing and library sampling, which
#: says nothing about assay-level systematic error, batch structure or the
#: selection's own nonlinearity: clearing it is not a qualification, so the kind
#: carries ``floor_is_lower_bound`` and never admits scheduling.
CHANNEL_KINDS: dict[str, dict[str, object]] = {
    'replicate': {
        'supports_measured_floor': True,
        'floor_is_lower_bound': False,
        'statement': (
            'independent per-state repeats -- separate cultures, sorts, growth '
            'replicates or synonymous genotypes -- whose between-channel '
            'disagreement measures the noise the contrast must clear'),
    },
    'dispersion': {
        'supports_measured_floor': False,
        'floor_is_lower_bound': False,
        'statement': (
            'one per-variant spread and no second channel, so a floor can be '
            'propagated from it but no between-channel decomposition exists and '
            'the endpoint cannot be compared against its own reproducibility'),
    },
    'count': {
        'supports_measured_floor': False,
        'floor_is_lower_bound': True,
        'statement': (
            'read counts only. Multinomial counting noise covers sequencing and '
            'library sampling and excludes assay-level systematic error, batch '
            'structure and the selection nonlinearity, so the floor it gives is '
            'a lower bound on total measurement noise and clearing it does not '
            'qualify the endpoint'),
    },
    'none': {
        'supports_measured_floor': False,
        'floor_is_lower_bound': True,
        'statement': 'no per-variant uncertainty of any kind; no floor is available',
    },
}


def channel_kind(kind: str) -> dict[str, object]:
    """The declared record for one channel kind, or a refusal.

    Refusing an undeclared kind is the load-bearing behaviour: a candidate whose
    channel is described in free text would otherwise reach a verdict without
    the lower-bound distinction being stated.
    """
    if kind not in CHANNEL_KINDS:
        raise ValueError(f'undeclared channel kind {kind!r}; kinds are {sorted(CHANNEL_KINDS)}')
    return {'kind': kind, **CHANNEL_KINDS[kind]}


# --------------------------------------------------------------------------- #
# 1 and 2: complete lower-order support, and the independent units it rests on.
# --------------------------------------------------------------------------- #

def support_depth_census(states: AssayStates, order: int) -> dict[str, int]:
    """How deep the lower-order support of this order's variants is measured.

    ``how much`` for question 1. A variant of order k whose support reaches
    depth k-1 carries a cube; one that reaches depth 0 is missing a constituent
    single substitution. Reporting the whole distribution rather than only the
    complete count is what separates "no cube exists" from "cubes are one level
    short", which are different findings about a library's design.
    """
    if order < 2:
        raise ValueError('lower-order support is defined from order two upwards')
    out: dict[str, int] = {}
    for key in states.states:
        if len(key) != order:
            continue
        depth = states.support_depth(key)
        out[str(depth)] = out.get(str(depth), 0) + 1
    return dict(sorted(out.items(), key=lambda item: int(item[0])))


def site_tuple_units(cubes: Sequence[tuple[Substitution, ...]]) -> dict[str, object]:
    """The independent units one order's cubes rest on, never the cube count.

    Every cube on one site tuple reuses that tuple's whole lower-order series --
    its wild type, its singles and every intermediate subset -- so cubes on one
    tuple are not independent draws and a cube-level interval is narrow by the
    amount of that reuse. Both levels below the cube are reported, because the
    site tuples of one library themselves reuse the same underlying single
    substitutions: ``positions`` and ``substitutions`` bound how much of a
    sample the tuple count is.
    """
    tuples = sorted({tuple(sorted(position for position, _ in cube)) for cube in cubes})
    sizes: dict[tuple[int, ...], int] = {}
    for cube in cubes:
        key = tuple(sorted(position for position, _ in cube))
        sizes[key] = sizes.get(key, 0) + 1
    positions = sorted({position for key in tuples for position in key})
    substitutions = sorted({substitution for cube in cubes for substitution in cube})
    return {
        'cubes': len(cubes),
        'site_tuples': len(tuples),
        'effective_site_tuples_kish': kish_effective_units([sizes[key] for key in tuples])
        if tuples else 0.0,
        'positions': len(positions),
        'substitutions': len(substitutions),
        'cubes_per_site_tuple_max': max(sizes.values()) if sizes else 0,
    }


def combinatorial_ceiling(n_positions: int, order: int) -> int:
    """``comb(n_positions, order)``: the most site tuples a library can ever carry.

    The trap this exists to print beside every tuple count. A library that
    randomises four positions has C(4, 3) = 4 position triples however many
    variants sit on them, which is how 25,035 complete order-3 cubes came to
    rest on four independent units.
    """
    if n_positions < 0 or order < 1:
        raise ValueError('a combinatorial ceiling needs a non-negative position count and order >= 1')
    return comb(n_positions, order) if order <= n_positions else 0


def support_inventory(states: AssayStates, *, orders: Sequence[int],
                      variable_positions: int) -> dict[str, object]:
    """Questions 1 and 2, per order, from the measured state keys alone.

    Reads keys and never values, so the support a candidate is admitted on
    cannot be a function of the labels that support will carry -- which is what
    lets the declaration be digested before any channel value is opened.
    """
    if not orders or any(order < 2 for order in orders):
        raise ValueError('an order-k support inventory needs orders of two or above')
    per_order: dict[str, object] = {}
    for order in sorted(set(orders)):
        cubes = complete_cubes(states, order)
        units = site_tuple_units(cubes)
        per_order[str(order)] = {
            **units,
            'measured_states_at_order': int(states.by_order.get(order, 0)),
            'support_depth_census': support_depth_census(states, order),
            'site_tuple_ceiling': combinatorial_ceiling(variable_positions, order),
            'unit_floor': bootstrap_unit_floor(units['site_tuples']),
        }
    return {
        'variable_positions': int(variable_positions),
        'wild_type_state_measured': () in states.states,
        'order_census': {str(order): int(count) for order, count in sorted(states.by_order.items())},
        'per_order': per_order,
    }


# --------------------------------------------------------------------------- #
# 3 and 4: the channel, and the order-k noise floor.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CandidateBlock:
    """One measurement block of a candidate: states, channels and a reference.

    A block is the largest set of states that share one measurement scale. Two
    blocks of one deposition -- two proteins, two segment libraries, two sorting
    runs -- are kept apart because a per-block scale difference does not cancel
    from a contrast whose corners came from different blocks.

    ``values`` maps a state key to one value per channel, in ``channels`` order.
    ``reference`` is the sequence the state keys are taken against, recorded so
    that a later reader can tell which sequence corner ``()`` addresses.
    """

    block_id: str
    reference: str
    channels: tuple[str, ...]
    values: Mapping[tuple[Substitution, ...], tuple[float, ...]]

    def __post_init__(self) -> None:
        if len(self.channels) < 1 or len(set(self.channels)) != len(self.channels):
            raise ValueError(f'{self.block_id}: channels must be distinct and non-empty')
        width = len(self.channels)
        for key, row in self.values.items():
            if len(row) != width:
                raise ValueError(f'{self.block_id}: state {key} carries {len(row)} of {width} channels')

    def states(self) -> AssayStates:
        return AssayStates.of(self.values)


def channel_order(channels: Sequence[str]) -> tuple[int, ...]:
    """Channel indices in a label-independent declared order.

    Ordered by a digest of each channel's own identifier, the convention
    :func:`proteingym_higher_order.genotype_rank` sets for the same reason:
    ordering channels by their measured values would put the high measurements
    in one channel and manufacture a channel offset. The **primary pair** is the
    first two under this order, fixed before any value is read, so the reading
    the verdict rests on is not selected on its outcome.
    """
    return tuple(sorted(range(len(channels)), key=lambda index: genotype_rank(channels[index])))


def propagated_order_floor(order: int, sigma: float) -> float:
    """``sqrt(2 ** order) * sigma``: the independent-corner-error prediction.

    An order-k cycle sums ``2 ** k`` corner estimates with unit weights where a
    pairwise cycle sums four, so its channel noise grows with order and has to be
    quantified per order rather than inherited. The prediction assumes the corner
    errors are independent. They are not, because neighbouring cubes share
    corners and a block's cubes share their whole lower-order series, so the
    measured discordance runs consistently *below* this value -- 0.76 to 0.82 of
    it at orders two through eight on the one assay where both were measured.
    Reporting both, and their ratio, is what establishes the propagation model
    instead of assuming it.
    """
    if order < 1:
        raise ValueError('a propagated floor needs an order of at least one')
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError('a propagated floor needs a finite non-negative sigma')
    return float(sqrt(2 ** order) * sigma)


def pooled_label_factor(n_channels: int) -> float:
    """How much smaller the pooled label's noise is than one channel's.

    The ratio the verdict rests on compares the reproducible component against
    *one channel's* noise, which is the admitted label instrument's construction.
    The quantity a model comparison would be fitted to is the pooled label over
    all channels, whose standard deviation is one channel's divided by
    ``sqrt(n)``. Multiplying the ratio through by ``sqrt(n)`` is the generous
    reading, reported beside the conservative one and never in place of it.
    """
    if n_channels < 2:
        raise ValueError('a pooled-label reading needs at least two channels')
    return float(sqrt(n_channels))


def _corner_sigma(block: CandidateBlock, cubes: Sequence[tuple[Substitution, ...]]) -> float:
    """Pooled within-state per-channel SD over exactly the corner states used."""
    corners = {state for cube in cubes for state in cube_addresses(cube).values()}
    rows = np.asarray([block.values[state] for state in sorted(corners)], dtype=np.float64)
    if rows.shape[1] < 2:
        return float('nan')
    return float(np.sqrt(np.mean(np.var(rows, axis=1, ddof=1))))


def order_floor(block: CandidateBlock, order: int, *, seed: int,
                resamples: int = 2000) -> dict[str, object]:
    """Question 4 for one order of one block: the cycle against its own channels.

    The cycle is formed independently in each channel of the declared primary
    pair and decomposed by the admitted between-channel construction, grouped on
    the site tuple because that is the level the estimand lives at. Every other
    channel pair is reported beside it as a consistency check and never pooled
    with it.
    """
    cubes = complete_cubes(block.states(), order)
    if not cubes:
        raise ValueError(f'{block.block_id}: no complete cube at order {order}')
    order_index = channel_order(block.channels)
    if len(order_index) < 2:
        raise ValueError(f'{block.block_id}: a measured floor needs two channels')
    groups = [tuple(sorted(position for position, _ in cube)) for cube in cubes]
    cycles = {
        index: [cube_contrast(cube, {state: block.values[state][index]
                                     for state in cube_addresses(cube).values()})
                for cube in cubes]
        for index in order_index
    }
    pairs: dict[str, object] = {}
    for left in range(len(order_index)):
        for right in range(left + 1, len(order_index)):
            a, b = order_index[left], order_index[right]
            label = f'{block.channels[a]}|{block.channels[b]}'
            pairs[label] = channel_decomposition_bootstrap(
                cycles[a], cycles[b], groups, seed=seed, resamples=resamples)
    primary_label = f'{block.channels[order_index[0]]}|{block.channels[order_index[1]]}'
    sigma = _corner_sigma(block, cubes)
    primary = pairs[primary_label]
    return {
        'block_id': block.block_id,
        'order': int(order),
        'channels': list(block.channels),
        'channel_order': [block.channels[index] for index in order_index],
        'primary_pair': primary_label,
        'units': site_tuple_units(cubes),
        'per_state_per_channel_sd': sigma,
        'propagated_floor': propagated_order_floor(order, sigma) if np.isfinite(sigma) else None,
        'measured_over_propagated': (
            primary['estimates']['per_channel_discordance_sd']['point']
            / propagated_order_floor(order, sigma)
            if np.isfinite(sigma) and sigma > 0 else None),
        'pooled_label_factor': pooled_label_factor(len(block.channels)),
        'channel_pairs': pairs,
    }


# --------------------------------------------------------------------------- #
# 5: whether the endpoint clears, and the refusal that follows from it.
# --------------------------------------------------------------------------- #

def _scaled(interval, factor):
    return None if interval is None else [value * factor for value in interval]


def clearance(floor: dict[str, object], kind: str) -> dict[str, object]:
    """Whether one order of one block clears its floor, conservatively and generously.

    Clearance is the **conservative** reading: the shared-to-discordance ratio's
    lower interval endpoint must exceed :data:`CLEARANCE_RATIO` against one
    channel's noise. The generous reading against the pooled label's noise is
    computed and reported -- the ratio scales linearly in the noise the
    denominator refers to, so the point estimate and both endpoints multiply
    through by :func:`pooled_label_factor` exactly -- and it never grants
    clearance on its own, because a model comparison fitted to a pooled label
    still has to be read against the reproducibility of the thing it was fitted
    to. A count-derived floor never grants clearance at all.
    """
    record = channel_kind(kind)
    primary = floor['channel_pairs'][floor['primary_pair']]
    ratio = primary['estimates']['shared_to_discordance_ratio']
    interval = ratio.get('ci95')
    factor = floor['pooled_label_factor']
    reasons: list[str] = []
    if not record['supports_measured_floor']:
        reasons.append(
            f'the channel kind {kind!r} supports no measured floor: ' + str(record['statement']))
    if record['floor_is_lower_bound']:
        reasons.append(
            'the floor is a lower bound on total measurement noise, so clearing it is '
            'not a qualification')
    if primary['degenerate']:
        reasons.append(str(primary['degenerate_reason']))
    if interval is None:
        reasons.append('no interval was computed, so no ratio can be resolved above the threshold')
    elif interval[0] <= CLEARANCE_RATIO:
        reasons.append(
            f'the shared-to-discordance ratio is {ratio["point"]:.3f} '
            f'[{interval[0]:.3f}, {interval[1]:.3f}] against one channel, whose lower '
            f'endpoint does not resolve above {CLEARANCE_RATIO}')
    generous_interval = _scaled(interval, factor)
    return {
        'block_id': floor['block_id'],
        'order': int(floor['order']),
        'channel': record,
        'units': floor['units'],
        'one_channel_reading': {'point': ratio['point'], 'ci95': interval},
        'pooled_label_reading': {
            'point': None if ratio['point'] is None else ratio['point'] * factor,
            'ci95': generous_interval,
            'factor': factor,
            'construction': (
                'the conservative ratio multiplied by sqrt(n_channels): the pooled '
                'label over n channels has one channel\'s noise divided by sqrt(n), '
                'and the ratio scales linearly in that denominator'),
        },
        'shared_component_sd': primary['estimates']['shared_component_sd'],
        'per_channel_discordance_sd': primary['estimates']['per_channel_discordance_sd'],
        'per_state_per_channel_sd': floor['per_state_per_channel_sd'],
        'propagated_floor': floor['propagated_floor'],
        'measured_over_propagated': floor['measured_over_propagated'],
        'secondary_pair_ratios': {
            label: record['estimates']['shared_to_discordance_ratio']
            for label, record in floor['channel_pairs'].items()
            if label != floor['primary_pair']},
        'primary_pair': floor['primary_pair'],
        'unit_floor': {key: primary[key] for key in ('units', 'effective_units_kish',
                                                     'degenerate', 'degenerate_reason')},
        'clearance_threshold': CLEARANCE_RATIO,
        'clears': not reasons,
        'clears_generous': bool(
            generous_interval is not None and generous_interval[0] > CLEARANCE_RATIO
            and record['supports_measured_floor'] and not record['floor_is_lower_bound']
            and not primary['degenerate']),
        'refusals': reasons,
        'pairwise_reference': PAIRWISE_REFERENCE,
    }


#: Above this share of the smaller block's molecules, two blocks cannot be
#: independent libraries: a majority of one block's measured sequences are the
#: other's. Half is the boundary rather than a tuned figure, and the measured
#: share is reported beside every pair so a reader never has to take the
#: threshold's word for it.
SHARED_SEQUENCE_FRACTION = 0.5


def block_sequences(block: CandidateBlock) -> set[str]:
    """The molecules one block measured, reconstructed from its reference and keys."""
    out = set()
    for key in block.values:
        letters = list(block.reference)
        for position, residue in key:
            letters[position] = residue
        out.add(''.join(letters))
    return out


def shared_sequence_census(blocks: Sequence[CandidateBlock]) -> list[dict[str, object]]:
    """Every pair of blocks that measured some of the same molecules.

    One deposition can carry one library twice, processed differently, and this
    candidate does: the same length-57 sequences appear in two tables, keyed
    against a wild-type-flagged reference in one and against the modal library
    background in the other, which gives 0 complete order-2 cubes in the first
    and 314 in the second *on the same measurements*. Two such blocks are not
    two units of evidence.
    """
    sequences = {block.block_id: block_sequences(block) for block in blocks}
    out: list[dict[str, object]] = []
    identifiers = sorted(sequences)
    for left in range(len(identifiers)):
        for right in range(left + 1, len(identifiers)):
            a, b = identifiers[left], identifiers[right]
            shared = len(sequences[a] & sequences[b])
            if not shared:
                continue
            smaller = min(len(sequences[a]), len(sequences[b]))
            out.append({'blocks': [a, b], 'shared_sequences': shared,
                        'sizes': [len(sequences[a]), len(sequences[b])],
                        'share_of_smaller': shared / smaller,
                        'one_library': shared / smaller > SHARED_SEQUENCE_FRACTION})
    return out


def duplicate_sequence_blocks(blocks: Sequence[CandidateBlock]) -> list[list[str]]:
    """Groups of blocks the shared-sequence census says are one library.

    :func:`candidate_verdict` refuses a verdict that counts more than one member
    of a group as a clearing cell at one order.
    """
    parent: dict[str, str] = {}

    def root(name: str) -> str:
        parent.setdefault(name, name)
        while parent[name] != name:
            name = parent[name] = parent[parent[name]]
        return name

    for pair in shared_sequence_census(blocks):
        if pair['one_library']:
            left, right = (root(name) for name in pair['blocks'])
            if left != right:
                parent[left] = right
    groups: dict[str, list[str]] = {}
    for name in sorted(parent):
        groups.setdefault(root(name), []).append(name)
    return [sorted(members) for members in groups.values() if len(members) > 1]


def generalisation_unit_census(blocks: Sequence[CandidateBlock],
                               block_ids: Sequence[str], *,
                               identity_threshold: float = FAMILY_IDENTITY_THRESHOLD
                               ) -> dict[str, object]:
    """Wild-type family groups behind a set of blocks: the unit a model comparison needs.

    Two different units govern an order-k endpoint and conflating them is how a
    noise-limited support becomes a scheduled run. The **site tuple** is the unit
    the label instrument's noise floor lives at, because every cube on one tuple
    reuses that tuple's lower-order series; it is what :func:`order_floor` groups
    on. The **wild-type family group** is the unit a *model comparison*
    generalises over, because a fit trained on one member of a family and tested
    on another measures family membership rather than the quantity. A support can
    be rich in the first and empty in the second: 858 site pairs inside one
    domain is one family group, and no site-pair count inside one protein is
    comparable with a Kish effective count taken across 64 of them.

    References of equal length are compared position by position, which is exact
    for fixed-length library references. References of different lengths are
    recorded as unalignable here and counted as **distinct**, which makes the
    returned group count an **upper bound**: an alignment could only merge them
    further. A refusal that fires on an upper bound is therefore sound, and a
    pass on one is not — the record says which it is.
    """
    if not 0.0 < identity_threshold <= 1.0:
        raise ValueError('an identity threshold is a fraction in (0, 1]')
    chosen = [block for block in blocks if block.block_id in set(block_ids)]
    missing = sorted(set(block_ids) - {block.block_id for block in chosen})
    if missing:
        raise ValueError(f'blocks absent from the candidate: {missing}')
    if not chosen:
        return {'blocks': [], 'distinct_references': 0, 'family_groups': 0,
                'group_count_is_upper_bound': False, 'pairs': [],
                'unit_floor': bootstrap_unit_floor(0),
                'unit': 'wild-type family group at 50% identity'}
    parent = {block.block_id: block.block_id for block in chosen}

    def root(name: str) -> str:
        while parent[name] != name:
            name = parent[name] = parent[parent[name]]
        return name

    pairs, unalignable = [], 0
    for left in range(len(chosen)):
        for right in range(left + 1, len(chosen)):
            first, second = chosen[left], chosen[right]
            if len(first.reference) != len(second.reference):
                unalignable += 1
                pairs.append({'blocks': [first.block_id, second.block_id],
                              'identity': None, 'one_family': False,
                              'reason': 'references differ in length; not alignable by '
                                        'index, counted as distinct'})
                continue
            matches = sum(1 for a, b in zip(first.reference, second.reference) if a == b)
            identity = matches / len(first.reference)
            merged = identity >= identity_threshold
            pairs.append({'blocks': [first.block_id, second.block_id],
                          'identity': identity, 'one_family': bool(merged)})
            if merged:
                a, b = root(first.block_id), root(second.block_id)
                if a != b:
                    parent[a] = b
    groups: dict[str, list[str]] = {}
    for block in chosen:
        groups.setdefault(root(block.block_id), []).append(block.block_id)
    members = [sorted(value) for value in groups.values()]
    return {
        'blocks': sorted(block.block_id for block in chosen),
        'distinct_references': len({block.reference for block in chosen}),
        'family_groups': len(members),
        'family_group_members': sorted(members),
        'group_count_is_upper_bound': unalignable > 0,
        'unalignable_pairs': unalignable,
        'identity_threshold': identity_threshold,
        'pairs': pairs,
        'unit': 'wild-type family group at 50% identity',
        'unit_floor': bootstrap_unit_floor(len(members)),
    }


def candidate_verdict(candidate_id: str, clearances: Sequence[dict[str, object]], *,
                      declaration_digest: str, declared_orders: Sequence[int],
                      blocks: Sequence[CandidateBlock],
                      duplicate_groups: Sequence[Sequence[str]] = (),
                      notes: Sequence[str] = ()) -> dict[str, object]:
    """The candidate's verdict, per order, with **stop** as the default.

    A clearance is reported per order, because the two failed gates turned on
    exactly that distinction: an order-2 reading that resolves while order three
    does not makes the failure an order property of a well-powered assay's
    labels rather than an assay property. Scheduling is therefore admitted per
    order too, and :func:`assert_scheduling_admitted` requires the order it is
    asked about -- an order-2 clearance licenses no order-3 scoring run.

    Clearing the noise floor is necessary and not sufficient. An order also has
    to carry enough of the unit a model comparison generalises over, and that is
    the wild-type family group rather than the site tuple: a support of 858 site
    pairs inside one domain clears a floor and still cannot carry a fit, because
    the frozen held-group fold design has no second group to hold out and a
    family bootstrap has one unit to draw. ``blocks`` supplies the references
    :func:`generalisation_unit_census` needs; scheduling is refused for any order
    whose clearing blocks fall below the shared unit floor on that count, with
    the two units reported separately so neither can be read as the other.
    """
    if not clearances:
        raise ValueError('a verdict needs at least one qualified order')
    orders = sorted(set(int(order) for order in declared_orders))
    if not orders:
        raise ValueError('a verdict needs at least one declared order')
    group_of = {member: index for index, group in enumerate(duplicate_groups)
                for member in group}
    per_order: dict[str, object] = {}
    for order in orders:
        cells = [record for record in clearances if int(record['order']) == order]
        clearing = [record for record in cells if record['clears']]
        counted = [group_of[record['block_id']] for record in clearing
                   if record['block_id'] in group_of]
        if len(counted) != len(set(counted)):
            raise ValueError(
                f'order {order}: two clearing cells come from one duplicated block group '
                f'{duplicate_groups}; they are the same measurements twice, not two units '
                'of evidence')
        best = max((record['one_channel_reading']['ci95'] or [float('-inf')])[0]
                   for record in cells) if cells else None
        units = generalisation_unit_census(
            blocks, [record['block_id'] for record in clearing]) if clearing else None
        floor_cleared = bool(clearing)
        units_sufficient = bool(units and not units['unit_floor']['degenerate'])
        limitations = []
        if not floor_cleared:
            limitations.append(
                f'No admitted block clears the order-{order} floor against one channel\'s '
                'noise. Any variation at this order is a measurement limitation, not a '
                'model finding, and the order is unresolved rather than "not detected".')
        elif not units_sufficient:
            limitations.append(
                f'The order-{order} floor is cleared on {len(clearing)} block(s), but they '
                f"rest on {units['family_groups']} wild-type family group(s)"
                + (' (an upper bound: some references are not alignable by index)'
                   if units['group_count_is_upper_bound'] else '')
                + f", below the {units['unit_floor']['minimum_units']}-unit floor. The site "
                'tuple is the unit the noise floor lives at and the family group is the '
                'unit a model comparison generalises over; a site-pair count inside one '
                'domain is not a substitute for the second. The frozen held-group fold '
                'design cannot be instantiated and a family bootstrap has too few units '
                'to draw, so no model comparison is scheduled.')
        per_order[str(order)] = {
            'cells': len(cells),
            'clearing_cells': [f"{record['block_id']}" for record in clearing],
            'clearing_cells_generous': [f"{record['block_id']}" for record in cells
                                        if record['clears_generous'] and not record['clears']],
            'best_one_channel_lower_bound': None if best == float('-inf') else best,
            'noise_floor_cleared': floor_cleared,
            'generalisation_units': units,
            'generalisation_units_sufficient': units_sufficient,
            'schedule_model_scoring': bool(floor_cleared and units_sufficient),
            'measurement_limitation': limitations[0] if limitations else None,
        }
    clearing_orders = [order for order in orders if per_order[str(order)]['schedule_model_scoring']]
    return {
        'schema_version': 'd1_higher_order_qualification_v1',
        'candidate': candidate_id,
        'declaration_digest': declaration_digest,
        'clearance_threshold': CLEARANCE_RATIO,
        'declared_orders': orders,
        'duplicate_block_groups': [list(group) for group in duplicate_groups],
        'cells': list(clearances),
        'per_order': per_order,
        'clearing_cells': [f"{record['block_id']}@order{record['order']}"
                           for record in clearances if record['clears']],
        'clearing_orders': clearing_orders,
        'orders_clearing_the_noise_floor_only': [
            order for order in orders
            if per_order[str(order)]['noise_floor_cleared']
            and not per_order[str(order)]['generalisation_units_sufficient']],
        'schedule_model_scoring': bool(clearing_orders),
        'recommendation': ('expand at ' + ', '.join(f'order {order}' for order in clearing_orders)
                           + '; stop at '
                           + ', '.join(f'order {order}' for order in orders
                                       if order not in clearing_orders)
                           if clearing_orders and len(clearing_orders) < len(orders)
                           else 'expand' if clearing_orders else 'stop'),
        'notes': list(notes),
        'reads_no_model_quantity': True,
    }


def assert_scheduling_admitted(verdict: Mapping[str, object], *, order: int) -> None:
    """Raise unless the verdict admits model scoring **at this order**.

    The default is to stop, and the order is required rather than optional: a
    candidate whose order-2 endpoint clears and whose order-3 endpoint does not
    is exactly the state both closed gates were in, and a caller that read one
    boolean off the whole candidate would schedule the run the floor refuses.
    """
    per_order = verdict.get('per_order') or {}
    record = per_order.get(str(int(order)))
    if record is None:
        raise ValueError(
            f"candidate {verdict.get('candidate')!r} declares no order-{order} "
            f"qualification; orders are {sorted(per_order)}")
    if not record.get('schedule_model_scoring'):
        raise ValueError(
            f"candidate {verdict.get('candidate')!r} does not clear its order-{order} "
            f"measurement-noise floor; model scoring at that order is refused. "
            f"{record.get('measurement_limitation')}")


# --------------------------------------------------------------------------- #
# One candidate adapter: DiMSum per-replicate fitness tables.
# --------------------------------------------------------------------------- #

def substitution_key(reference: str, sequence: str) -> tuple[Substitution, ...]:
    """The state key of one equal-length sequence against a reference.

    Derived from the sequences rather than from a declared order column, because
    a deposition's order column is not reliable: MegaScale's ``aa_seq`` is
    truncated to wild-type length so an insertion reads as a run of
    substitutions, and one of this candidate's own tables marks all 9,973 of its
    length-57 rows ``indel``. A length change is refused rather than truncated,
    and a non-canonical residue is refused rather than silently substituted.
    """
    if len(reference) != len(sequence):
        raise ValueError('a substitution key needs an equal-length reference and sequence')
    if set(sequence) - RESIDUES or set(reference) - RESIDUES:
        raise ValueError('a substitution key needs canonical residues on both sides')
    return tuple((index, sequence[index]) for index in range(len(reference))
                 if sequence[index] != reference[index])


def dimsum_blocks(rows: Iterable[Mapping[str, str]], *, channel_columns: Sequence[str],
                  sequence_column: str = 'aa_seq',
                  wild_type_column: str = 'WT') -> dict[str, object]:
    """Measurement blocks from one DiMSum per-replicate fitness table.

    DiMSum writes one row per amino-acid state with a per-replicate fitness
    column per growth replicate, which is a replicate channel in the same sense
    as MegaScale's two proteases: separately grown and separately sequenced
    repeats of the same state.

    Blocking is by sequence length, because a table that pools several proteins
    pools several scales and a contrast whose corners came from two of them is
    not a contrast. Within a block the reference is the row the file flags as
    wild type when exactly one does, and the block's modal residue per position
    otherwise; either way the reference is required to be a measured state, so
    corner ``()`` of every cube is measured and the absolute wild-type-centred
    estimand is available rather than the centred-residual one.

    Rows are excluded and counted, never repaired: a duplicate state, a
    non-canonical residue, and a row without a finite value in **every** declared
    channel each remove the row. The last of those is a selection worth naming --
    a state resolved in some replicates and not others is dropped, which
    conditions the support on resolvability in every channel and, at order k, in
    all ``2 ** k`` corners.
    """
    if not channel_columns:
        raise ValueError('a replicate channel needs at least one column')
    by_length: dict[int, list[Mapping[str, str]]] = {}
    total = {'noncanonical_residue': 0, 'nonfinite_channel': 0, 'duplicate_state': 0}
    for row in rows:
        by_length.setdefault(len(row[sequence_column]), []).append(row)
    blocks: list[CandidateBlock] = []
    summary: list[dict[str, object]] = []
    for length, members in sorted(by_length.items(), key=lambda item: -len(item[1])):
        excluded = {'noncanonical_residue': 0, 'nonfinite_channel': 0, 'duplicate_state': 0}
        flagged = sorted({row[sequence_column] for row in members
                          if str(row.get(wild_type_column, '')).strip().upper() in ('TRUE', 'T')})
        if len(flagged) == 1:
            reference, reference_source = flagged[0], 'wild-type-flagged row'
        else:
            columns = [{} for _ in range(length)]
            for row in members:
                for index, residue in enumerate(row[sequence_column]):
                    columns[index][residue] = columns[index].get(residue, 0) + 1
            reference = ''.join(max(sorted(column), key=lambda residue: column[residue])
                                for column in columns)
            reference_source = f'modal residue per position over {len(members)} rows'
        values: dict[tuple[Substitution, ...], tuple[float, ...]] = {}
        for row in members:
            sequence = row[sequence_column]
            if set(sequence) - RESIDUES:
                excluded['noncanonical_residue'] += 1
                continue
            try:
                channels = tuple(float(row[column]) for column in channel_columns)
            except (TypeError, ValueError):
                excluded['nonfinite_channel'] += 1
                continue
            if not all(np.isfinite(channels)):
                excluded['nonfinite_channel'] += 1
                continue
            key = substitution_key(reference, sequence)
            if key in values:
                excluded['duplicate_state'] += 1
                continue
            values[key] = channels
        for key, count in excluded.items():
            total[key] += count
        if () not in values:
            summary.append({'block_id': f'len{length}', 'rows': len(members),
                            'admitted': False, 'excluded_rows': excluded,
                            'reason': 'the reference sequence is not a measured state, so corner '
                                      '() of every cube is unmeasured and only the '
                                      'centred-residual estimand is available'})
            continue
        block = CandidateBlock(block_id=f'len{length}', reference=reference,
                               channels=tuple(channel_columns), values=values)
        blocks.append(block)
        variable = len({position for key in values for position, _ in key})
        summary.append({'block_id': block.block_id, 'rows': len(members),
                        'admitted': True, 'states': len(values),
                        'sequence_length': length, 'variable_positions': variable,
                        'reference_source': reference_source, 'excluded_rows': excluded})
    return {'blocks': blocks, 'block_summary': summary, 'excluded_rows': total}


def declare_candidate(candidate_id: str, *, blocks: Sequence[CandidateBlock],
                      orders: Sequence[int], channel_kind_name: str,
                      provenance: Mapping[str, object],
                      block_summary: Sequence[Mapping[str, object]] = (),
                      excluded_rows: Mapping[str, int] | None = None) -> dict[str, object]:
    """Questions 1 to 3, digest-bound and written before any channel value is read.

    The support inventory reads state keys only, so this declaration can be
    written and hashed before the qualification stage opens a value -- which is
    what makes the admitted support provably not a function of the labels it
    will carry.
    """
    if not blocks:
        raise ValueError('a declaration needs at least one admitted block')
    record = {
        'schema_version': 'd1_higher_order_declaration_v1',
        'candidate': candidate_id,
        'declared_orders': sorted(set(int(order) for order in orders)),
        'channel': channel_kind(channel_kind_name),
        'provenance': dict(provenance),
        'block_summary': [dict(entry) for entry in block_summary],
        'excluded_rows': dict(excluded_rows or {}),
        'reads_no_value': True,
        'blocks': {},
    }
    for block in blocks:
        states = block.states()
        variable = len({position for key in states.states for position, _ in key})
        record['blocks'][block.block_id] = {
            'channels': list(block.channels),
            'reference_sha256': hashlib.sha256(block.reference.encode()).hexdigest(),
            'sequence_length': len(block.reference),
            'states': len(states.states),
            **support_inventory(states, orders=record['declared_orders'],
                                variable_positions=variable),
        }
    return record
