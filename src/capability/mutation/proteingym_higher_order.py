"""Mutation-order census and complete-cube support over ProteinGym substitution assays.

No model score, likelihood or representation enters anything here. The contrast
itself is :func:`src.capability.mutation.higher_order_cycle.cycle_contrast`, reused rather
than reimplemented; this module supplies the ProteinGym-specific parts: the
``mutant`` column's grammar, the per-assay measured-state keys, and the
lower-order completeness requirement that decides which variants carry a cycle.

ProteinGym substitution CSVs carry no exact wild-type row, so corner ``()`` of
every cube is unmeasured. The wild-type value is an assay constant and enters a
cycle contrast of any order with coefficient ``(-1) ** order``, so it shifts
every contrast in one assay by the same amount and cancels from within-assay
ranks or centred residuals. That is the estimand this support admits; it does
not identify absolute sign or magnitude and does not remove assay nonlinearity.
Accordingly the completeness requirement below asks for every non-empty subset
of a variant's substitutions and treats the empty corner as the assay constant.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, product
from math import comb

import numpy as np

from src.capability.mutation.higher_order_cycle import (
    Substitution, cycle_contrast, kish_effective_units)
from src.capability.core.statistics import bootstrap_unit_floor

#: The 20 canonical residues a ProteinGym substitution token may name. The column
#: is a colon-joined list of ``<wild><1-based position><mutant>`` tokens.
RESIDUES = frozenset('ACDEFGHIKLMNPQRSTVWY')

@dataclass(frozen=True)
class MutantKey:
    """One measured variant, keyed by its substitutions relative to the assay wild type.

    ``substitutions`` is sorted by zero-based position and is the cube address the
    contrast reads. ``wild`` records the residue each token declares as wild at its
    position, which is what the assay-wide wild-type consistency check compares.
    """

    substitutions: tuple[Substitution, ...]
    wild: tuple[str, ...]

    @property
    def order(self) -> int:
        return len(self.substitutions)


def parse_mutant(mutant: str) -> MutantKey:
    """Parse one ``mutant`` cell into its substitution key.

    Refuses rather than repairs: a token that is not ``<wild><position><mutant>``
    over the canonical residues, a non-positive position, a token whose wild and
    mutant residue coincide, and a variant naming one position twice all raise.
    An empty cell is not a wild-type row in this format and also raises; these
    CSVs carry no wild-type row and a blank cell would be a parse failure rather
    than a measured wild type.
    """
    text = mutant.strip()
    if not text:
        raise ValueError('empty mutant cell: these CSVs carry no wild-type row')
    parsed: list[tuple[int, str, str]] = []
    for token in text.split(':'):
        if len(token) < 3:
            raise ValueError(f'malformed substitution token {token!r}')
        wild, mutant_residue, digits = token[0], token[-1], token[1:-1]
        if wild not in RESIDUES or mutant_residue not in RESIDUES:
            raise ValueError(f'token {token!r} names a residue outside the canonical 20')
        if not digits.isdigit():
            raise ValueError(f'token {token!r} carries no 1-based position')
        position = int(digits)
        if position < 1:
            raise ValueError(f'token {token!r} carries a non-positive position')
        if wild == mutant_residue:
            raise ValueError(f'token {token!r} substitutes a residue for itself')
        parsed.append((position - 1, wild, mutant_residue))
    parsed.sort()
    positions = [position for position, _, _ in parsed]
    if len(set(positions)) != len(positions):
        raise ValueError(f'variant {text!r} names one position twice')
    return MutantKey(
        substitutions=tuple((position, mutant) for position, _, mutant in parsed),
        wild=tuple(wild for _, wild, _ in parsed),
    )


@dataclass(frozen=True)
class AssayStates:
    """One assay's measured state keys, with the order buckets completeness needs.

    ``by_order`` is derived from ``states`` in :meth:`of` and never supplied, so a
    count can never disagree with the set it describes. It exists because a level
    of a cube cannot be complete when the assay holds fewer states of that order
    than the level requires, which turns an otherwise combinatorial enumeration
    into a bounded one: a random-mutagenesis library carries variants of order 44,
    and enumerating that cube's 17.6e12 corners to discover its singles are
    missing is not a measurement.
    """

    states: frozenset[tuple[Substitution, ...]]
    by_order: Mapping[int, int]

    @classmethod
    def of(cls, states: Iterable[tuple[Substitution, ...]]) -> 'AssayStates':
        frozen = frozenset(states)
        return cls(states=frozen, by_order=census(frozen))

    def support_depth(self, substitutions: Sequence[Substitution]) -> int:
        """Largest ``k`` for which every ``k``-subset of ``substitutions`` is measured.

        Levels run from 1 upwards and stop at the first incomplete one, because no
        deeper level can restore completeness. A depth of ``len(substitutions) - 1``
        means the cube is complete up to the unmeasured empty corner; a depth of 0
        means at least one constituent single substitution was not measured.
        """
        order = len(substitutions)
        if order < 2:
            raise ValueError('a cube needs at least two substitutions to have a lower order')
        ordered = sorted(substitutions)
        if len(set(position for position, _ in ordered)) != order:
            raise ValueError('a cube cannot carry one position twice')
        for size in range(1, order):
            required = comb(order, size)
            # Exact, not a heuristic: ``required`` distinct states of this order
            # cannot all be present when the assay holds fewer of them in total.
            if self.by_order.get(size, 0) < required:
                return size - 1
            if any(subset not in self.states for subset in combinations(ordered, size)):
                return size - 1
        return order - 1

    def has_complete_support(self, substitutions: Sequence[Substitution]) -> bool:
        """Whether every non-empty proper subset of ``substitutions`` is measured."""
        return self.support_depth(substitutions) == len(substitutions) - 1


def census(states: Iterable[tuple[Substitution, ...]]) -> dict[int, int]:
    """Count of measured states at each mutation order."""
    out: dict[int, int] = {}
    for key in states:
        out[len(key)] = out.get(len(key), 0) + 1
    return out


def complete_cubes(assay: AssayStates, order: int) -> list[tuple[Substitution, ...]]:
    """Variants of exactly ``order`` substitutions whose lower-order support is complete."""
    if order < 2:
        raise ValueError('completeness is defined from order two upwards')
    return sorted(
        key for key in assay.states
        if len(key) == order and assay.has_complete_support(key)
    )


def centred_residuals(contrasts: Sequence[float]) -> list[float]:
    """Contrasts with their own mean removed, the form an unmeasured wild type cancels from.

    Called per assay. The wild-type corner enters an order-k contrast with
    coefficient ``(-1) ** k`` and one value per assay, so when it is unmeasured the
    whole set of that assay's contrasts is shifted by one unknown constant and
    subtracting the assay mean removes it exactly. It also removes any other
    assay-wide additive offset, which is why the resulting estimand identifies
    neither absolute sign nor absolute magnitude. Where the wild-type state is
    measured this is unnecessary: the contrast's signed coefficients already sum
    to zero, so an assay-wide offset cancels without centring.
    """
    if not contrasts:
        raise ValueError('centring needs at least one contrast')
    mean = sum(contrasts) / len(contrasts)
    return [value - mean for value in contrasts]


def genotype_rank(identifier: str) -> bytes:
    """Label-independent ordering key for one replicate's own identifier.

    A replicate channel split must not depend on the value being measured, so
    replicates are ordered by a digest of their own identifier -- a nucleotide
    sequence, a barcode -- rather than by the measured value or by file order.
    A digest rather than the identifier itself so that the order does not follow
    the library's own design, and stable across processes unlike ``hash``.
    """
    return hashlib.sha256(identifier.encode()).digest()


def split_channels(values: Sequence[float]) -> tuple[float, float, float, int, int]:
    """Two channel means from one state's replicate values, and their spread.

    ``values`` must already be in replicate-identifier order, which
    :func:`genotype_rank` supplies. Alternating ranks rather than halves, so a
    state whose replicates differ systematically along that order does not put one
    end of it in one channel. Returns the two channel means, the unbiased variance
    of the individual replicates, and the two channel sizes; a channel mean over
    ``n`` replicates has variance ``variance / n``, which is what propagates a
    per-state dispersion into an order-k cycle's floor.
    """
    if len(values) < 2:
        raise ValueError('two channels need at least two replicates')
    left, right = list(values[0::2]), list(values[1::2])
    variance = float(np.var(np.asarray(values, dtype=np.float64), ddof=1))
    return (float(np.mean(left)), float(np.mean(right)), variance, len(left), len(right))


# --------------------------------------------------------------------------- #
# The cube, and the between-channel decomposition that qualifies it.
# --------------------------------------------------------------------------- #

#: The wild-type corner's state key. Measured in several ProteinGym *raw* assay
#: files and dropped by the processed CSVs, which is the whole reason the absolute
#: estimand is available here at all.
WILD_TYPE_STATE: tuple[Substitution, ...] = ()


def cube_addresses(
    substitutions: Sequence[Substitution]
) -> dict[tuple[int, ...], tuple[Substitution, ...]]:
    """The ``2 ** order`` corners of one cube, as binary keys to measured state keys.

    Bit ``i`` of the key selects the ``i``-th substitution in position order, which
    is the addressing :func:`src.capability.mutation.higher_order_cycle.cycle_contrast`
    expects, so the contrast and its sign convention are reused rather than
    restated.
    """
    order = len(substitutions)
    if order < 2:
        raise ValueError('a cube needs at least two substitutions')
    ordered = sorted(substitutions)
    return {
        corner: tuple(sub for sub, on in zip(ordered, corner) if on)
        for corner in product((0, 1), repeat=order)
    }


def cube_contrast(
    substitutions: Sequence[Substitution], values: Mapping[tuple[Substitution, ...], float]
) -> float:
    """The order-k wild-type-centred cycle of one cube, in the units of ``values``."""
    corners = {corner: values[state] for corner, state in
               cube_addresses(substitutions).items()}
    return cycle_contrast(corners)


def channel_moments(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Sufficient statistics of one unit's two-channel pairs.

    The same seven per-unit means the admitted label instrument's decomposition
    uses, so a bootstrap draw averages unit moments rather than re-reading values.
    """
    if left.shape != right.shape or left.ndim != 1 or left.size < 1:
        raise ValueError('two aligned non-empty channel vectors are required')
    difference = left - right
    return np.array([
        left.sum(), right.sum(), (left * left).sum(), (right * right).sum(),
        (left * right).sum(), (difference * difference).sum(), np.abs(difference).sum(),
    ]) / left.size


def derive_channel_decomposition(moments: np.ndarray) -> dict[str, float | None]:
    """Reproducible and discordant components of a two-channel quantity.

    The shared component is an upper bound on reproducible signal, because error
    common to both channels contributes to the covariance; the per-channel
    discordance is a lower bound on per-channel measurement noise, because error
    common to both channels cancels in the difference. A ratio of the first to the
    second is therefore an upper bound over a lower bound, and only a ratio well
    above one supports a statement that the quantity carries reproducible signal.
    """
    mean_left, mean_right, sq_left, sq_right, cross, sq_diff, abs_diff = moments
    var_left = max(float(sq_left - mean_left * mean_left), 0.0)
    var_right = max(float(sq_right - mean_right * mean_right), 0.0)
    covariance = float(cross - mean_left * mean_right)
    offset = float(mean_left - mean_right)
    var_diff = max(float(sq_diff - offset * offset), 0.0)
    shared = float(np.sqrt(max(covariance, 0.0)))
    discordance = float(np.sqrt(var_diff / 2.0))
    return {
        'mean_left': float(mean_left), 'mean_right': float(mean_right),
        'sd_left': float(np.sqrt(var_left)), 'sd_right': float(np.sqrt(var_right)),
        'covariance': covariance,
        'pearson_r': (float(covariance / np.sqrt(var_left * var_right))
                      if var_left > 0 and var_right > 0 else None),
        'shared_component_sd': shared,
        'channel_offset': offset,
        'channel_difference_sd': float(np.sqrt(var_diff)),
        'channel_difference_mean_abs': float(abs_diff),
        'per_channel_discordance_sd': discordance,
        'shared_to_discordance_ratio': (float(shared / discordance)
                                        if discordance > 0 else None),
    }


def channel_decomposition_bootstrap(
    left: Sequence[float], right: Sequence[float], groups: Sequence,
    *, seed: int, resamples: int = 2000,
) -> dict:
    """Group-bootstrapped two-channel decomposition, resampling whole groups.

    The group is the resampling unit because every contrast on one site tuple
    reuses the same lower-order measurements, so contrasts within a group are not
    independent draws. Below the package's declared unit floor the record carries
    the point estimates and no interval, with the reason named: the count is a
    measured property of the support rather than a configuration choice, so the
    point estimate survives the refusal and the interval does not.
    """
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    # ``groups`` stays a Python sequence: a group label is a site tuple, and
    # converting it to an array would read its coordinates as extra dimensions.
    group_list = list(groups)
    if left_array.shape != right_array.shape or left_array.size != len(group_list):
        raise ValueError('channels and groups must align')
    if left_array.ndim != 1 or left_array.size < 1:
        raise ValueError('a decomposition needs at least one observation')
    if resamples < 1:
        raise ValueError('resamples must be positive')
    unique: dict = {}
    for index, group in enumerate(group_list):
        unique.setdefault(group, []).append(index)
    order = sorted(unique, key=str)
    vectors = np.array([
        channel_moments(left_array[unique[group]], right_array[unique[group]])
        for group in order])
    sizes = [len(unique[group]) for group in order]
    floor = bootstrap_unit_floor(len(order))
    record = {
        'observations': int(left_array.size), 'units': len(order),
        'effective_units_kish': kish_effective_units(sizes),
        'resamples': resamples, 'seed': seed, **floor,
        'estimates': {key: {'point': value}
                      for key, value in derive_channel_decomposition(
                          vectors.mean(axis=0)).items()},
    }
    if floor['degenerate']:
        return record
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(vectors), size=(resamples, len(vectors)))
    samples = [derive_channel_decomposition(vectors[row].mean(axis=0)) for row in draws]
    for key, entry in record['estimates'].items():
        column = np.array([sample[key] for sample in samples
                           if sample[key] is not None], dtype=np.float64)
        column = column[np.isfinite(column)]
        entry['ci95'] = ([float(np.percentile(column, 2.5)),
                          float(np.percentile(column, 97.5))] if column.size else None)
        entry['finite_draws'] = int(column.size)
    return record
