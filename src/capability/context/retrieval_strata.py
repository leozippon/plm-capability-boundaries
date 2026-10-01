"""Prespecified retrieval and memorization strata, and stratified re-estimation.

The gate this module serves asks one question: does a model-side gain established
on a held-group split survive being read separately on units whose nearest
retrieved homolog is close and on units whose nearest retrieved homolog is
remote? A gain that concentrates where retrieval support is dense and vanishes
where it is thin is a retrieval explanation of that gain; a gain that holds flat
across strata is not explained that way.

No band edge here is a free parameter, because a free parameter on a stratum
boundary is a place where an outcome could enter the design.

* Identity bands are :data:`~.homology.STRATUM_NAMES` with
  :data:`~.homology.STRATUM_EDGES`, frozen in that module before its first
  search and imported rather than restated. Near-duplicate presence is the same
  declaration's top edge, 95.0% identity over the query.
* Depth bands are decades of the reweighted alignment depth ``Neff`` that both
  cohorts' profile stores already retain under identical settings, plus one
  explicit band for a unit whose search returns no qualifying homolog at all.
  The coarse two-level depth contrast splits at the same 100-effective-sequence
  decade and exists because a decade band can fall below the package's
  eight-unit percentile-interval floor.
* Family bands are read off the frozen grouping contract of each cohort: a unit
  whose group holds one member is in the remote band, a unit whose group holds
  more than one is in the close band. Nothing is re-clustered here.

Stratum assignment is a function of retrieved-homolog statistics and frozen
group membership only. No measured stability, DMS effect or fitted prediction
reaches it, so a stratum cannot be a function of the gain it is used to read.

Re-estimation reuses the estimator of the study it reads rather than restating
it: a stratum's estimate is the same paired statistic over the same fitted
held-out predictions, taken over the subset of resampling units the stratum
holds. Unit-level weighting inside a retained unit is therefore untouched, and
the full-support estimate is the unit-count-weighted mean of its strata, which
:func:`decomposition_residual` checks rather than assumes.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from .homology import STRATUM_EDGES, STRATUM_NAMES, assign_stratum
from ..core.statistics import MINIMUM_BOOTSTRAP_UNITS

#: Identity bands, imported from the frozen homology declaration.
IDENTITY_BANDS: tuple[str, ...] = STRATUM_NAMES

#: The frozen near-duplicate edge, in percent identity over the query.
NEAR_DUPLICATE_IDENTITY: float = STRATUM_EDGES[3]

NEAR_DUPLICATE_BANDS: tuple[str, ...] = ('near_duplicate_absent', 'near_duplicate_present')

#: Decade edges of the reweighted alignment depth, in effective sequences.
DEPTH_EDGES: tuple[float, ...] = (10.0, 100.0, 1000.0)
DEPTH_BANDS: tuple[str, ...] = (
    'no_homolog_support', 'neff_lt10', 'neff_10_100', 'neff_100_1000', 'neff_ge1000')

#: The coarse depth contrast's edge, the same decade as the middle fine edge.
COARSE_DEPTH_EDGE: float = DEPTH_EDGES[1]
COARSE_DEPTH_BANDS: tuple[str, ...] = ('support_shallow_lt100', 'support_deep_ge100')

FAMILY_BANDS: tuple[str, ...] = ('remote_singleton_group', 'close_multi_member_group')

#: Experimental-source provenance. This is a source-independence sensitivity and
#: not a retrieval or memorization test: the stability cohort is drawn entirely
#: from the Tsuboyama 2023 MegaScale release, so a Readout-anchor assay carrying
#: the same source is not an independent endpoint for a gain measured on that
#: cohort. The axis is reported separately for that reason, and folding it into
#: an identity or family band would attribute a provenance effect to retrieval.
SOURCE_TOKEN = 'Tsuboyama_2023'
SOURCE_BANDS: tuple[str, ...] = ('source_other', 'source_tsuboyama_2023')

#: A unit holding both a shared-source assay and an unshared one has two
#: defensible exclusions, and the choice is declared here rather than left to
#: whichever the code reaches first.
SOURCE_EXCLUSION: dict[str, str] = {
    'primary': 'drop_whole_units',
    'primary_reason': (
        'the resampling unit is the wild-type family and the estimator weights each of that '
        'family\'s assays equally inside it, so dropping only the shared-source assays of a mixed '
        'unit changes that unit\'s internal weighting and makes its retained value a different '
        'statistic from the one the original fit averaged; dropping whole units leaves every '
        'retained unit\'s internal weighting identical to the original fit, which is the condition '
        'the whole comparison rests on'),
    'primary_cost': (
        'the unshared assays of a mixed unit are discarded with it, so the primary support is '
        'smaller than the unshared material available'),
    'sensitivity': 'drop_shared_source_assays_only',
    'sensitivity_reason': (
        'retains a mixed unit on its unshared assays alone, recovering that material at the price '
        'of a changed within-unit composition; reported beside the primary rule rather than '
        'replacing it'),
}

#: One entry per declared stratification: its bands and what the band is a
#: function of. The text travels with the numbers so a band label cannot be read
#: without its definition.
STRATIFICATIONS: dict[str, dict[str, object]] = {
    'identity_band': {
        'bands': IDENTITY_BANDS,
        'quantity': 'maximum percent identity over the query of any retrieved corpus hit',
        'edges': list(STRATUM_EDGES),
        'definition': ('src.capability.context.homology.assign_stratum on the maximum '
                       'identity_over_query = 100 * nident / qlen over the retained DIAMOND '
                       'hit table; a unit with no hit reads 0.0 and lands in the lowest band'),
    },
    'near_duplicate': {
        'bands': NEAR_DUPLICATE_BANDS,
        'quantity': 'maximum percent identity over the query of any retrieved corpus hit',
        'edges': [NEAR_DUPLICATE_IDENTITY],
        'definition': ('present at or above the frozen near-duplicate edge of '
                       f'{NEAR_DUPLICATE_IDENTITY}% identity over the query, absent below it; '
                       'a coarsening of identity_band, not an independent notion'),
    },
    'depth_band': {
        'bands': DEPTH_BANDS,
        'quantity': 'reweighted alignment depth Neff of the fitted independent-site profile',
        'edges': list(DEPTH_EDGES),
        'definition': ('decades of Neff at the profile store\'s own 30% identity floor and 80% '
                       'reweighting threshold; no_homolog_support is a unit whose search '
                       'returns no qualifying homolog, which carries no profile at all'),
    },
    'coarse_depth': {
        'bands': COARSE_DEPTH_BANDS,
        'quantity': 'reweighted alignment depth Neff of the fitted independent-site profile',
        'edges': [COARSE_DEPTH_EDGE],
        'definition': (f'shallow below {COARSE_DEPTH_EDGE} effective sequences, deep at or above '
                       'it; a unit with no qualifying homolog is shallow, which is the direction '
                       'that cannot overstate deep support'),
    },
    'family_band': {
        'bands': FAMILY_BANDS,
        'quantity': 'member count of the unit\'s group under the cohort\'s frozen grouping contract',
        'edges': [2],
        'definition': ('remote when the frozen contract places the unit in a group of one, close '
                       'when it places two or more members together; a conservative held-group '
                       'control, not certified remote-family disjointness'),
    },
    'source_provenance': {
        'bands': SOURCE_BANDS,
        'quantity': f'presence of the {SOURCE_TOKEN} source token in any of the unit\'s assay identifiers',
        'edges': [],
        'definition': (f'a unit is {SOURCE_BANDS[1]} when any assay it holds carries the '
                       f'{SOURCE_TOKEN} token, which is the experimental source of the whole '
                       'measured-stability cohort; the two gains of this gate are therefore not '
                       'independent endpoints on that band, and this axis is a source-independence '
                       'sensitivity rather than a retrieval or memorization test'),
    },
}


def identity_band(max_identity_over_query: float) -> str:
    """Band the nearest retrieved homolog's identity over the query."""

    return assign_stratum(float(max_identity_over_query))


def near_duplicate_band(max_identity_over_query: float) -> str:
    """Presence of a retrieved near-duplicate at the frozen 95% edge."""

    value = float(max_identity_over_query)
    if not 0.0 <= value <= 100.0:
        raise ValueError(f'percent identity {value} is outside [0, 100]')
    return NEAR_DUPLICATE_BANDS[1] if value >= NEAR_DUPLICATE_IDENTITY else NEAR_DUPLICATE_BANDS[0]


def depth_band(neff: float | None) -> str:
    """Band the reweighted alignment depth; ``None`` is no qualifying homolog."""

    if neff is None:
        return DEPTH_BANDS[0]
    value = float(neff)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f'alignment depth {value} is not a positive effective-sequence count')
    for index, edge in enumerate(DEPTH_EDGES):
        if value < edge:
            return DEPTH_BANDS[index + 1]
    return DEPTH_BANDS[-1]


def coarse_depth_band(neff: float | None) -> str:
    """The two-level depth contrast, at the same decade as the fine middle edge."""

    if neff is None:
        return COARSE_DEPTH_BANDS[0]
    value = float(neff)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f'alignment depth {value} is not a positive effective-sequence count')
    return COARSE_DEPTH_BANDS[1] if value >= COARSE_DEPTH_EDGE else COARSE_DEPTH_BANDS[0]


def family_band(group_members: int) -> str:
    """Band a unit by its frozen group's member count."""

    count = int(group_members)
    if count < 1:
        raise ValueError('a unit belongs to a group with at least itself in it')
    return FAMILY_BANDS[1] if count > 1 else FAMILY_BANDS[0]


def source_band(assay_identifiers: Iterable[str]) -> str:
    """Band a unit by whether any assay it holds carries the shared source token."""

    identifiers = list(assay_identifiers)
    if not identifiers:
        raise ValueError('a unit carries at least one assay identifier')
    return SOURCE_BANDS[1] if any(SOURCE_TOKEN in name for name in identifiers) \
        else SOURCE_BANDS[0]


def assign_unit(*, max_identity_over_query: float, neff: float | None,
                group_members: int, assay_identifiers: Iterable[str]) -> dict[str, str]:
    """Every declared stratification's band for one resampling unit."""

    return {
        'identity_band': identity_band(max_identity_over_query),
        'near_duplicate': near_duplicate_band(max_identity_over_query),
        'depth_band': depth_band(neff),
        'coarse_depth': coarse_depth_band(neff),
        'family_band': family_band(group_members),
        'source_provenance': source_band(assay_identifiers),
    }


def verify_partition(assignment: Mapping[str, Mapping[str, str]]) -> dict[str, dict]:
    """Every stratification must be a genuine partition of the same unit set.

    Refuses a unit missing a stratification, a band outside the declaration and
    a stratification whose band counts do not sum to the unit total. The returned
    record carries the per-band counts and the residual, which is zero by
    construction here and reported so a reader does not have to trust that.
    """

    units = sorted(assignment)
    if not units:
        raise ValueError('a partition needs at least one unit')
    record: dict[str, dict] = {}
    for name, declaration in STRATIFICATIONS.items():
        bands = tuple(declaration['bands'])  # type: ignore[arg-type]
        labels = []
        for unit in units:
            if name not in assignment[unit]:
                raise ValueError(f'{unit} carries no {name} band')
            label = assignment[unit][name]
            if label not in bands:
                raise ValueError(f'{unit}: {label!r} is not a declared {name} band')
            labels.append(label)
        counts = Counter(labels)
        assigned = sum(counts.values())
        if assigned != len(units):
            raise ValueError(f'{name} assigns {assigned} of {len(units)} units')
        record[name] = {
            'bands': {band: int(counts.get(band, 0)) for band in bands},
            'units': len(units),
            'unassigned_units': len(units) - assigned,
            'empty_bands': [band for band in bands if not counts.get(band, 0)],
        }
    return record


def kish_effective(weights: Sequence[float]) -> float:
    """Kish effective count of a weight vector, ``(sum w)^2 / sum w^2``."""

    values = np.asarray(list(weights), dtype=float)
    if values.size == 0:
        return 0.0
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Kish effective counts need finite non-negative weights')
    total = values.sum()
    if total <= 0:
        raise ValueError('Kish effective counts need a positive total weight')
    return float(total ** 2 / (values ** 2).sum())


def kish_subunits(subunits_per_unit: Mapping[str, int]) -> float:
    """Kish effective sub-unit count under the estimator's own weights.

    The stability estimator weights each held group equally and each of that
    group's site pairs equally inside it, so a site pair of a group holding
    ``k`` of them carries ``1 / (G k)``; the Readout estimator weights each
    wild-type family equally and each assay equally inside it. This is the Kish
    count of exactly those weights.
    """

    counts = {unit: int(value) for unit, value in subunits_per_unit.items()}
    if not counts:
        return 0.0
    if any(value < 1 for value in counts.values()):
        raise ValueError('every retained unit carries at least one sub-unit')
    groups = len(counts)
    weights = [1.0 / (groups * value) for value in counts.values() for _ in range(value)]
    return kish_effective(weights)


def kish_row_counts(rows_per_subunit: Sequence[int]) -> float:
    """Kish effective sub-unit count from row counts, ``(sum n)^2 / sum n^2``.

    This is the convention the pairwise readiness record reports, under which the
    64-group support carries 121.6 effective site pairs against its 217 nominal
    ones: it measures how unevenly the fitted rows spread over the dependence
    blocks. It is a different quantity from :func:`kish_subunits`, which is the
    Kish count of the weights the estimator actually applies, and the two are
    reported side by side rather than one being passed off as the other.
    """

    counts = [int(value) for value in rows_per_subunit]
    if not counts:
        return 0.0
    if any(value < 1 for value in counts):
        raise ValueError('every sub-unit carries at least one row')
    return kish_effective(counts)


def bands_of(assignment: Mapping[str, Mapping[str, str]], name: str) -> dict[str, list[str]]:
    """Units of each band of one stratification, in sorted unit order."""

    if name not in STRATIFICATIONS:
        raise ValueError(f'{name} is not a declared stratification')
    grouped: dict[str, list[str]] = {band: [] for band in STRATIFICATIONS[name]['bands']}  # type: ignore[index]
    for unit in sorted(assignment):
        grouped[assignment[unit][name]].append(unit)
    return grouped


def decomposition_residual(full_point: float, band_points: Mapping[str, float | None],
                           band_units: Mapping[str, int]) -> float:
    """``full - sum_b (n_b / n) * point_b`` over the bands of one stratification.

    A unit-count-weighted mean of the band estimates has to return the estimate
    on the whole support exactly, because every unit sits in exactly one band and
    its within-unit weighting is unchanged by the restriction. A nonzero residual
    means the strata are not a partition of the fitted support, so this is
    computed and reported rather than assumed.
    """

    total = sum(int(count) for count in band_units.values())
    if total < 1:
        raise ValueError('a decomposition needs at least one unit')
    accumulated = 0.0
    for band, count in band_units.items():
        if not count:
            continue
        point = band_points.get(band)
        if point is None:
            raise ValueError(f'{band} holds {count} units but carries no point estimate')
        accumulated += (int(count) / total) * float(point)
    return float(full_point) - accumulated


def direction(interval: object) -> str | None:
    """Which side of zero a resolved interval lies on, and ``None`` when it spans zero.

    Reported beside :func:`resolution`, because an interval that excludes zero
    from below is a resolved *negative* increment and reading it as a surviving
    gain would invert the finding.
    """

    if interval is None:
        return None
    low, high = float(interval[0]), float(interval[1])
    if low > 0.0:
        return 'above_zero'
    if high < 0.0:
        return 'below_zero'
    return None


def resolution(n_units: int, excludes_zero: bool | None) -> str:
    """One word for what a stratum's interval establishes about its gain.

    ``unresolved_thin_support`` is a stratum below the package's percentile-unit
    floor: it carries no verdict at all, and must not be read as a stratum the
    gain failed on.
    """

    if int(n_units) < MINIMUM_BOOTSTRAP_UNITS:
        return 'unresolved_thin_support'
    if excludes_zero is None:
        return 'unresolved_no_interval'
    return 'resolved' if excludes_zero else 'unresolved_interval_crosses_zero'


def stratum_support(units: Iterable[str], assignment: Mapping[str, Mapping[str, str]],
                    name: str, band: str) -> list[str]:
    """The retained units of one band, refusing an undeclared band name."""

    if name not in STRATIFICATIONS:
        raise ValueError(f'{name} is not a declared stratification')
    if band not in STRATIFICATIONS[name]['bands']:  # type: ignore[operator]
        raise ValueError(f'{band} is not a declared {name} band')
    return [unit for unit in units if assignment[unit][name] == band]
