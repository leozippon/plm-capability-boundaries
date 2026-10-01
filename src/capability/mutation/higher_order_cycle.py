"""Complete-cube substitution contrasts and their cross-background construction.

No model score, likelihood or representation enters anything here. The order-2
specialisation of :func:`cycle_contrast` is the wild-type-centred pairwise cycle
epsilon that the admitted label instrument reports; the order-3 case is the
third-order cycle omega this module exists for.

MegaScale ``dataset2`` carries no variant with three substitutions relative to
its own background wild type, so the only third-order measurement available in
those bytes is a cross-background one. The identity that makes it third order is

    omega = epsilon(variant background) - epsilon(reference background)

whenever the two backgrounds differ at exactly one site that the mutation pair
does not touch: the eight states of the cube are then all measured, four in each
background. A per-background additive offset cancels exactly, because the four
reference-background states carry signs -1, +1, +1, -1 and the four
variant-background states carry +1, -1, -1, +1.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass
from itertools import combinations, product

#: A substitution is a zero-based sequence position and the substituted residue.
Substitution = tuple[int, str]
#: A measured state is addressed by the background it was measured in and the
#: substitutions applied to that background's wild type, sorted by position.
StateAddress = tuple[str, tuple[Substitution, ...]]


def cycle_contrast(values: Mapping[tuple[int, ...], float]) -> float:
    """Signed contrast over the complete cube of ``order`` binary substitutions.

    The sign of a corner is ``(-1) ** (order - popcount)``, so the all-mutant
    corner enters positively. At order 2 this is ``y_AB - y_A - y_B + y_WT``, the
    endpoint the pairwise experiment measured; at order 3 it is
    ``y_ABC - y_AB - y_AC - y_BC + y_A + y_B + y_C - y_WT``.

    Every corner must be supplied. A partially observed cube has no contrast, and
    filling a missing corner with an additive prediction would make the contrast
    report the prediction rather than a measurement.
    """
    if not values:
        raise ValueError('a cycle contrast needs at least one substitution')
    order = len(next(iter(values)))
    if order < 1:
        raise ValueError('a cycle contrast needs at least one substitution')
    corners = set(product((0, 1), repeat=order))
    if set(values) != corners:
        raise ValueError(f'order-{order} contrast requires all {len(corners)} corners')
    return float(sum((-1) ** (order - sum(corner)) * values[corner] for corner in corners))


def background_difference(reference: str, variant: str) -> tuple[int, ...]:
    """Zero-based positions at which two equal-length wild types differ."""
    if len(reference) != len(variant):
        raise ValueError('background contrast requires equal-length wild types')
    sites = tuple(i for i in range(len(reference)) if reference[i] != variant[i])
    if not sites:
        raise ValueError('two backgrounds must differ at at least one site')
    return sites


@dataclass(frozen=True)
class BackgroundContrast:
    """One mutation pair whose complete pairwise cycle is measured in two backgrounds.

    ``reference`` is the alphabetically smaller ``WT_name`` of the two, which is a
    label-independent choice and coincides with the documented parent for every
    name-derived parent/child pair, because a parent's name is a prefix of its
    child's. The sign convention follows from it: the third substitution runs
    reference to variant, so ``omega > 0`` places the mutation pair's interaction
    epsilon higher in the variant background than in the reference background.

    ``first`` and ``second`` are the pair's substitutions ordered by position.
    ``background_sites`` are the positions at which the two wild types differ;
    the contrast is an exact third-order cycle when there is exactly one.
    """

    reference: str
    variant: str
    background_sites: tuple[int, ...]
    first: Substitution
    second: Substitution

    @property
    def order(self) -> int:
        """Substitutions separating the all-mutant corner from the reference wild type."""
        return 2 + len(self.background_sites)

    @property
    def site_pair(self) -> tuple[int, int]:
        return self.first[0], self.second[0]

    def addresses(self) -> dict[tuple[int, int, int], StateAddress]:
        """The eight corners, as ``(background, substitutions)`` measurement keys.

        Bit order is ``(first, second, background)``: bit three selects the
        variant background, so corners with it set are measured there with the
        mutation pair applied to that background's own wild type.
        """
        out: dict[tuple[int, int, int], StateAddress] = {}
        for a, b, c in product((0, 1), repeat=3):
            applied = tuple(s for s, on in ((self.first, a), (self.second, b)) if on)
            out[(a, b, c)] = (self.variant if c else self.reference, applied)
        return out

    def sequences(self, wildtypes: Mapping[str, str]) -> dict[tuple[int, int, int], str]:
        """The eight corner sequences, validated against the two wild types."""
        reference, variant = wildtypes[self.reference], wildtypes[self.variant]
        if background_difference(reference, variant) != self.background_sites:
            raise ValueError('wild types do not carry the declared background difference')
        for position, residue in (self.first, self.second):
            if position in self.background_sites:
                raise ValueError('mutation pair overlaps the background difference')
            if reference[position] == residue:
                raise ValueError('a substitution must change its position')
            if reference[position] != variant[position]:
                raise ValueError('mutation pair position differs between backgrounds')
        out = {}
        for corner, (name, applied) in self.addresses().items():
            letters = list(variant if name == self.variant else reference)
            for position, residue in applied:
                letters[position] = residue
            out[corner] = ''.join(letters)
        return out

    def value_cube(self, measured: Mapping[str, Mapping[tuple[Substitution, ...], float]]
                   ) -> dict[tuple[int, int, int], float]:
        """Corner values read from per-background measured states."""
        return {corner: measured[name][applied]
                for corner, (name, applied) in self.addresses().items()}

    def omega(self, measured: Mapping[str, Mapping[tuple[Substitution, ...], float]]) -> float:
        """The declared endpoint, in the units of ``measured``."""
        return cycle_contrast(self.value_cube(measured))


def complete_pairs(wildtype: str, states: Set[tuple[Substitution, ...]]
                   ) -> set[tuple[Substitution, Substitution]]:
    """Mutation pairs whose wild type, both singles and double are all measured.

    Substitutions are checked against ``wildtype`` so that a state carried under a
    different background's numbering cannot enter a pair.
    """
    if () not in states:
        return set()
    singles = {state[0] for state in states if len(state) == 1}
    out = set()
    for state in states:
        if len(state) != 2:
            continue
        first, second = state
        if first[0] >= second[0]:
            raise ValueError('substitutions must be sorted by position')
        if wildtype[first[0]] == first[1] or wildtype[second[0]] == second[1]:
            raise ValueError('a substitution must change its position')
        if first in singles and second in singles:
            out.add((first, second))
    return out


def enumerate_contrasts(
    wildtypes: Mapping[str, str],
    states: Mapping[str, Set[tuple[Substitution, ...]]],
    *,
    max_background_distance: int,
) -> list[BackgroundContrast]:
    """Every mutation pair with a complete pairwise cycle in two related backgrounds.

    Metadata only: the measured keys decide the support and no value is read, so
    the admitted support cannot be a function of the labels it will carry. A
    background pair qualifies when its wild types have equal length, differ at one
    to ``max_background_distance`` sites, and differ at neither of the pair's own
    two positions — otherwise the two cycles are not the same mutation pair.
    """
    if max_background_distance < 1:
        raise ValueError('a background contrast needs a differing site')
    pairs: dict[tuple[Substitution, Substitution], list[str]] = {}
    for name in sorted(states):
        for pair in complete_pairs(wildtypes[name], states[name]):
            pairs.setdefault(pair, []).append(name)
    out: list[BackgroundContrast] = []
    for (first, second), names in pairs.items():
        for reference, variant in combinations(sorted(names), 2):
            if len(wildtypes[reference]) != len(wildtypes[variant]):
                continue
            sites = background_difference(wildtypes[reference], wildtypes[variant])
            if len(sites) > max_background_distance:
                continue
            if first[0] in sites or second[0] in sites:
                continue
            out.append(BackgroundContrast(reference, variant, sites, first, second))
    return sorted(out, key=lambda c: (c.reference, c.variant, c.first, c.second))


def kish_effective_units(sizes: Iterable[int]) -> float:
    """Kish effective count of weighted units, the power statement the cohort uses."""
    weights = [float(size) for size in sizes]
    total = sum(weights)
    if total <= 0:
        return 0.0
    return total ** 2 / sum(weight ** 2 for weight in weights)
