"""Where likelihood and the independent structural evaluation part company.

The estimand, and why it is stratified
======================================

The question is whether a model's likelihood of a sequence still carries
information about that sequence's independently evaluated structure. The naive
answer -- the rank correlation between likelihood and structure over every
variant in a rung -- is not that question, because both quantities depend
strongly on chain length and on which backbone a variant came from, so a pooled
correlation would be dominated by between-backbone variation that has nothing to
do with modification extent.

The estimand here is therefore the **within-backbone** rank correlation
(:func:`within_stratum_rank_correlation`): ranks are taken inside each backbone
and centred there before they are correlated, so every between-backbone
difference -- length first of all -- is removed by construction. The resampling
unit is the backbone, not the variant: two draws on one backbone share a parent,
a prompt and a window, and are not independent observations.

Two readings, both reported
===========================

The marginal interval of one cell is a statement about that cell. The ladder
makes a panel-wide claim -- *this* arm diverges at *this* rung, and the rung
differs by model class -- so the panel also carries a simultaneous band
(:func:`simultaneous_band`), built from the same bootstrap resamples across every
cell so that the maximum-deviation statistic is defined at all. A marginal
interval is never read as a simultaneous statement.

Locating the divergence
=======================

:func:`divergence_rung` applies two declared criteria to a ladder: the first rung
whose interval admits zero, and the first rung whose point estimate changes sign
relative to the bottom of the ladder. They can disagree, and when they do, both
are reported. :func:`continuous_crossing` goes finer than the rung grid by
treating extent as continuous and solving for the extent at which the fitted
correlation reaches zero, inside the bootstrap so the crossing carries an
interval of its own. A ladder whose correlation never reaches zero inside the
measured range returns no crossing, and that is a result.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Literal, Mapping, Sequence, overload

import numpy as np
from scipy import stats

SCHEMA_VERSION = "ladder_divergence_v1"

#: Resampling units below this floor do not support a percentile interval. The
#: same floor this package applies everywhere else; restated through an import so
#: there is one declaration of it.
from ..core.statistics import MINIMUM_BOOTSTRAP_UNITS, bootstrap_unit_floor  # noqa: E402

DEFAULT_DRAWS = 10000
CONFIDENCE = 0.95
MINIMUM_VALID_FRACTION = 0.95
MINIMUM_VALID_DRAWS = 20


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not bias the correlation.

    Hand-rolled rather than ``scipy.stats.rankdata``, measured: this is called
    once per backbone per bootstrap draw per readout on vectors of about eight
    values, and at that size SciPy's dispatch overhead dominates -- swapping it
    in tripled the test suite's runtime.
    """

    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def _centred_within(values: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Within-stratum average ranks, centred on each stratum's own mean rank.

    Strata with fewer than two observations contribute a zero vector: a single
    observation carries no within-stratum rank information, and dropping it
    silently would change the denominator without saying so. The caller reports
    how many such strata there were.
    """

    centred = np.zeros(values.size, dtype=np.float64)
    for stratum in np.unique(strata):
        rows = np.flatnonzero(strata == stratum)
        if rows.size < 2:
            continue
        ranks = _ranks(values[rows])
        centred[rows] = ranks - ranks.mean()
    return centred


def within_stratum_rank_correlation(
    x: Sequence[float], y: Sequence[float], strata: Sequence[Any]
) -> float:
    """The pooled within-stratum Spearman correlation of ``x`` and ``y``.

    Ranks inside each stratum, centres them there, then correlates the pooled
    centred ranks. Every between-stratum difference is removed by construction,
    which is what makes this a statement about variation among a backbone's own
    draws rather than about the differences between backbones.
    """

    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    groups = np.asarray(strata)
    if xs.shape != ys.shape or xs.ndim != 1 or groups.shape != xs.shape:
        raise ValueError("x, y and strata must be one-dimensional and aligned")
    if not np.isfinite(xs).all() or not np.isfinite(ys).all():
        raise ValueError("a within-stratum rank correlation refuses non-finite values")
    left = _centred_within(xs, groups)
    right = _centred_within(ys, groups)
    denominator = math.sqrt(float((left**2).sum()) * float((right**2).sum()))
    if denominator == 0.0:
        return float("nan")
    return float((left * right).sum() / denominator)


@overload
def partial_within_stratum_rank_correlation(
    x: Sequence[float], y: Sequence[float], z: Sequence[float], strata: Sequence[Any],
    *, return_diagnostics: Literal[False] = False,
) -> float: ...


@overload
def partial_within_stratum_rank_correlation(
    x: Sequence[float], y: Sequence[float], z: Sequence[float], strata: Sequence[Any],
    *, return_diagnostics: Literal[True],
) -> dict[str, Any]: ...


def partial_within_stratum_rank_correlation(
    x: Sequence[float], y: Sequence[float], z: Sequence[float], strata: Sequence[Any],
    *, return_diagnostics: bool = False,
) -> float | dict[str, Any]:
    """The same correlation after linearly removing a covariate's within-stratum rank.

    The covariate this experiment conditions on is composition distance from the
    parent, so that a correlation driven by amino-acid composition drift is not
    reported as a structural effect. Residualising ranks is a rank-partial
    correlation and not a causal adjustment; it says what is left after a linear
    function of the covariate's rank is taken out, and nothing more.
    """

    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    zs = np.asarray(z, dtype=np.float64)
    groups = np.asarray(strata)
    if not (xs.shape == ys.shape == zs.shape == groups.shape) or xs.ndim != 1:
        raise ValueError("x, y, z and strata must be one-dimensional and aligned")
    if not all(np.isfinite(vector).all() for vector in (xs, ys, zs)):
        raise ValueError("a partial within-stratum correlation refuses non-finite values")
    left = _centred_within(xs, groups)
    right = _centred_within(ys, groups)
    covariate = _centred_within(zs, groups)
    original = [float(np.linalg.norm(vector)) for vector in (left, right)]
    norm = float((covariate**2).sum())
    if norm > 0.0:
        left = left - covariate * float((left * covariate).sum()) / norm
        right = right - covariate * float((right * covariate).sum()) / norm
    residual = [float(np.linalg.norm(vector)) for vector in (left, right)]
    tolerances = [64 * np.finfo(np.float64).eps * value for value in original]
    reason = None
    if any(value == 0 for value in original):
        reason = "constant within-backbone ranks"
    elif any(value <= tolerance for value, tolerance in zip(residual, tolerances)):
        reason = "covariate-collinear ranks: residual norm at float64 roundoff scale"
    informative = [str(group) for group in np.unique(groups)
        if all(float(np.linalg.norm(vector[groups == group])) > tolerance
               for vector, tolerance in zip((left, right), tolerances))
    ]
    point = None if reason else float(np.dot(left, right) / (residual[0] * residual[1]))
    diagnostics = {
        "point": point, "reason": reason, "original_rank_norms": original,
        "residual_rank_norms": residual, "residual_tolerances": tolerances,
        "n_informative_backbones": len(informative),
        "informative_backbones": informative,
    }
    return diagnostics if return_diagnostics else (float("nan") if point is None else point)


def per_stratum_spearman(
    x: Sequence[float], y: Sequence[float], strata: Sequence[Any]
) -> dict[str, Any]:
    """One Spearman per stratum, and their unweighted mean.

    Reported beside the pooled statistic because the two answer slightly
    different questions: the pooled one weights a backbone by how much rank
    variance its draws happen to carry, the mean weights every backbone equally.
    A disagreement between them is a finding about the backbone set.
    """

    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    groups = np.asarray(strata)
    values: dict[str, float] = {}
    for stratum in np.unique(groups):
        rows = np.flatnonzero(groups == stratum)
        if rows.size < 3:
            continue
        left = _ranks(xs[rows])
        right = _ranks(ys[rows])
        left = left - left.mean()
        right = right - right.mean()
        denominator = math.sqrt(float((left**2).sum()) * float((right**2).sum()))
        if denominator == 0.0:
            continue
        values[str(stratum)] = float((left * right).sum() / denominator)
    finite = np.asarray(list(values.values()), dtype=np.float64)
    return {
        "per_stratum": values,
        "n_strata_scored": int(finite.size),
        "mean": float(finite.mean()) if finite.size else None,
        "mean_reason": None if finite.size else "no informative backbone correlations",
        "sd": float(finite.std(ddof=1)) if finite.size > 1 else None,
        "sd_reason": None if finite.size > 1 else "fewer than two informative backbone correlations",
    }


def stratum_resamples(units: Sequence[Any], *, seed: int, draws: int = DEFAULT_DRAWS) -> np.ndarray:
    """One set of cluster resamples over the panel's **whole** unit list.

    Shared deliberately, and taken over the whole list rather than over whatever
    units a particular cell happens to contain. A simultaneous band is a
    statement about the maximum deviation across cells, and that maximum is only
    defined if every cell is recomputed on the same resample; a cell in which one
    backbone produced nothing simply contributes no rows for it, which is the
    correct accounting rather than a different bootstrap.
    """

    count = len(list(units))
    floor = bootstrap_unit_floor(count)
    if floor["degenerate"]:
        raise ValueError(f"cluster bootstrap refused: {floor['degenerate_reason']}")
    if draws < 1:
        raise ValueError("draws must be positive")
    generator = np.random.default_rng(int(seed))
    return generator.integers(0, count, size=(int(draws), count))


def cluster_bootstrap_many(
    values: Mapping[str, Sequence[float]],
    strata: Sequence[Any],
    statistics: Mapping[str, Callable[[Mapping[str, np.ndarray], np.ndarray], float]],
    *,
    units: Sequence[Any],
    resamples: np.ndarray,
    confidence: float = CONFIDENCE,
    informative_units: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Every statistic of one cell, in a single pass over the resamples.

    One pass, because the resample's row index and relabelled stratum vector are
    the expensive part and they are the same for every statistic of a cell; a
    pass per statistic multiplied that cost by the number of readouts, which on
    this panel was the difference between minutes and hours.

    A unit drawn twice is relabelled, so it counts as two strata rather than as
    one with twice the draws: collapsing it would halve the effective cluster
    count exactly where the bootstrap is measuring it.

    The draw vector is returned, not just summarised, because the simultaneous
    band needs the raw draws to take a maximum across cells. Intervals require
    the floor of informative independent backbones for each statistic, at least
    20 finite draws and a fixed 95% valid fraction; undefined draws are counted,
    never retried. Callers with residual diagnostics can supply their informative
    unit identities; otherwise support is counted by finite single-unit statistics.
    """

    groups = np.asarray(strata)
    arrays = {name: np.asarray(vector, dtype=np.float64) for name, vector in values.items()}
    for name, vector in arrays.items():
        if vector.shape != groups.shape:
            raise ValueError(f"{name} does not align with the stratum vector")
    unit_list = np.asarray(list(units))
    if resamples.ndim != 2 or resamples.shape[1] != unit_list.size:
        raise ValueError("the resamples must be (draws, units) over the supplied unit list")
    if not statistics:
        raise ValueError("a bootstrap needs at least one statistic")
    rows_by_unit = [np.flatnonzero(groups == unit) for unit in unit_list]
    point = {name: float(function(arrays, groups)) for name, function in statistics.items()}
    support = {
        name: list(informative_units[name]) if informative_units is not None else [
            unit_list[index] for index, rows in enumerate(rows_by_unit)
            if rows.size >= 2 and np.isfinite(function(
                {key: vector[rows] for key, vector in arrays.items()}, groups[rows]
            ))
        ] for name, function in statistics.items()
    }
    draws = {name: np.empty(resamples.shape[0], dtype=np.float64) for name in statistics}
    for index in range(resamples.shape[0]):
        resample = resamples[index]
        selected = [rows_by_unit[unit_index] for unit_index in resample]
        sizes = [block.size for block in selected]
        total = int(sum(sizes))
        if total < 2:
            for name in statistics:
                draws[name][index] = float("nan")
            continue
        row_index = np.concatenate(selected)
        labels = np.repeat(np.arange(len(sizes)), sizes)
        resampled = {name: vector[row_index] for name, vector in arrays.items()}
        for name, function in statistics.items():
            draws[name][index] = function(resampled, labels)
    tail = (1.0 - confidence) / 2.0
    records: dict[str, dict[str, Any]] = {}
    for name in statistics:
        vector = draws[name]
        finite = vector[np.isfinite(vector)]
        interval_reason = None
        if not np.isfinite(point[name]):
            interval_reason = "undefined point statistic"
        elif len(support[name]) < MINIMUM_BOOTSTRAP_UNITS:
            interval_reason = "too few informative independent backbones"
        elif finite.size < MINIMUM_VALID_DRAWS or finite.size / vector.size < MINIMUM_VALID_FRACTION:
            interval_reason = "too few finite bootstrap draws"
        records[name] = {
            "point": point[name] if np.isfinite(point[name]) else None,
            "point_reason": None if np.isfinite(point[name]) else "undefined within-backbone statistic",
            "interval": (
                [float(np.quantile(finite, tail)), float(np.quantile(finite, 1.0 - tail))]
                if interval_reason is None else None
            ),
            "interval_reason": interval_reason,
            "inferential_available": interval_reason is None,
            "informative_units": [str(unit) for unit in support[name]],
            "n_informative_units": len(support[name]),
            "minimum_valid_fraction": MINIMUM_VALID_FRACTION,
            "minimum_valid_draws": MINIMUM_VALID_DRAWS,
            "n_undefined_draws": int(vector.size - finite.size),
            "confidence": float(confidence),
            "n_observations": int(groups.size),
            "n_units": int(np.unique(groups).size),
            "n_panel_units": int(unit_list.size),
            "n_draws": int(resamples.shape[0]),
            "n_finite_draws": int(finite.size),
            "bootstrap_sd": float(finite.std(ddof=1)) if finite.size > 1 else None,
            "draws": vector,
        }
    return records


def cluster_bootstrap(
    values: Mapping[str, Sequence[float]],
    strata: Sequence[Any],
    statistic: Callable[[Mapping[str, np.ndarray], np.ndarray], float],
    *,
    units: Sequence[Any],
    resamples: np.ndarray,
    confidence: float = CONFIDENCE,
) -> dict[str, Any]:
    """One statistic's point estimate, percentile interval and draw vector."""

    return cluster_bootstrap_many(
        values,
        strata,
        {"statistic": statistic},
        units=units,
        resamples=resamples,
        confidence=confidence,
    )["statistic"]


def simultaneous_band(
    cells: Mapping[str, Mapping[str, Any]], *, confidence: float = CONFIDENCE
) -> dict[str, Any]:
    """Sup-t simultaneous intervals over a panel of cells.

    The maximum of the standardised deviations is taken draw by draw across every
    cell, using only draws finite in every declared column, and its ``confidence``
    quantile scales each cell's own bootstrap standard deviation. An unavailable
    cell or insufficient joint-valid draws makes the declared family's band
    unavailable; the family is never silently reduced.
    """

    usable = {
        name: cell
        for name, cell in cells.items()
        if cell.get("point") is not None and np.isfinite(cell["point"])
        and cell.get("bootstrap_sd") is not None and np.isfinite(cell["bootstrap_sd"])
        and cell["bootstrap_sd"] > 0.0 and cell.get("inferential_available", True)
    }
    excluded = sorted(set(cells) - set(usable))
    if excluded or not usable:
        return {
            "available": False,
            "declared_family": list(cells),
            "n_draws": max((len(cell.get("draws", [])) for cell in cells.values()), default=0),
            "n_joint_valid_draws": 0,
            "n_undefined_draws": max((len(cell.get("draws", [])) for cell in cells.values()), default=0),
            "minimum_valid_fraction": MINIMUM_VALID_FRACTION,
            "minimum_valid_draws": MINIMUM_VALID_DRAWS,
            "quantile": None,
            "intervals": {},
            "excluded_cells": excluded,
            "reason": "declared family contains unavailable cells or is empty",
        }
    lengths = {len(cell["draws"]) for cell in usable.values()}
    if len(lengths) != 1:
        raise ValueError(
            "a simultaneous band needs every cell recomputed on the same resamples; "
            f"the cells carry {sorted(lengths)} draws"
        )
    stacked = np.vstack(
        [
            (np.asarray(cell["draws"]) - cell["point"]) / cell["bootstrap_sd"]
            for cell in usable.values()
        ]
    )
    joint_valid = np.isfinite(stacked).all(axis=0)
    finite = np.max(np.abs(stacked[:, joint_valid]), axis=0)
    counts = {
        "n_draws": stacked.shape[1], "n_joint_valid_draws": int(joint_valid.sum()),
        "n_undefined_draws": int((~joint_valid).sum()),
        "minimum_valid_fraction": MINIMUM_VALID_FRACTION,
        "minimum_valid_draws": MINIMUM_VALID_DRAWS,
        "declared_family": list(cells),
    }
    if finite.size < MINIMUM_VALID_DRAWS or finite.size / stacked.shape[1] < MINIMUM_VALID_FRACTION:
        return {
            "quantile": None,
            "intervals": {},
            "excluded_cells": excluded,
            "reason": "too few joint-valid draws for the declared family",
            "available": False,
            **counts,
        }
    quantile = float(np.quantile(finite, confidence))
    return {
        "quantile": quantile,
        "available": True,
        **counts,
        "confidence": float(confidence),
        "n_cells": len(usable),
        "excluded_cells": excluded,
        "intervals": {
            name: [
                cell["point"] - quantile * cell["bootstrap_sd"],
                cell["point"] + quantile * cell["bootstrap_sd"],
            ]
            for name, cell in usable.items()
        },
    }


def _excludes_zero(interval: Sequence[float] | None) -> bool | None:
    if interval is None or len(interval) != 2 or any(value is None for value in interval):
        return None
    low, high = float(interval[0]), float(interval[1])
    if not np.isfinite([low, high]).all() or low > high:
        return None
    return (low > 0.0 and high > 0.0) or (low < 0.0 and high < 0.0)


def divergence_rung(
    ladder: Sequence[Mapping[str, Any]],
    *,
    simultaneous: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """The rung at which a ladder's correlation stops being resolvable.

    ``ladder`` is in ascending-extent order, each entry carrying ``rung``,
    ``point`` and ``interval``. Two declared criteria, both reported:

    ``loses_significance`` is the first rung whose interval admits zero *and*
    which is not followed by a rung that excludes it again -- a single
    non-significant rung in the middle of a resolved ladder is noise, not a
    divergence, and reporting it as one is how a sampling accident becomes a
    finding.

    ``changes_sign`` is the first rung whose point estimate has the opposite sign
    from the bottom rung. Both criteria are descriptive heuristics, not tests of
    a decline or a mechanism; a decline needs a paired rung contrast.

    A ladder whose bottom rung already admits zero has no divergence point to
    find, and that is reported as ``undetermined_at_bottom`` rather than as a
    divergence at the first rung.
    """

    entries = list(ladder)
    if not entries:
        raise ValueError("a divergence needs a ladder")
    resolved = [_excludes_zero(entry["interval"]) for entry in entries]
    points = [float(entry["point"]) if entry.get("point") is not None
              and np.isfinite(entry["point"]) else None for entry in entries]
    result: dict[str, Any] = {
        "rungs": [str(entry["rung"]) for entry in entries],
        "points": points,
        "resolved": resolved,
        "loses_significance": None,
        "changes_sign": None,
        "undetermined_at_bottom": None if resolved[0] is None else not resolved[0],
        "interpretation": "descriptive sign/first-loss heuristics; not evidence of a decline or mechanism",
        "inferential_available": all(value is not None for value in resolved + points),
        "reason": None if all(value is not None for value in resolved + points) else "missing or undefined primary inference",
    }
    if result["inferential_available"] and resolved[0] and all(point is not None for point in points):
        for index in range(1, len(entries)):
            if not resolved[index] and not any(resolved[index + 1 :]):
                result["loses_significance"] = str(entries[index]["rung"])
                break
        bottom = points[0]
        for index in range(1, len(entries)):
            current = points[index]
            if bottom is not None and current is not None and bottom != 0.0 and current * bottom < 0.0:
                result["changes_sign"] = str(entries[index]["rung"])
                break
    simultaneous = simultaneous or {}
    simultaneous_resolved = [_excludes_zero(simultaneous.get(str(entry["rung"]))) for entry in entries]
    result["simultaneous_resolved"] = simultaneous_resolved
    result["loses_significance_simultaneous"] = None
    result["simultaneous_available"] = all(value is not None for value in simultaneous_resolved)
    result["simultaneous_reason"] = None if result["simultaneous_available"] else "missing or undefined simultaneous inference"
    if result["simultaneous_available"] and simultaneous_resolved[0]:
        for index in range(1, len(entries)):
            if not simultaneous_resolved[index] and not any(simultaneous_resolved[index + 1 :]):
                result["loses_significance_simultaneous"] = str(entries[index]["rung"])
                break
    return result


def adjacent_rung_contrasts(
    cells: Mapping[str, Mapping[str, Any]], *, rungs: Sequence[str],
    confidence: float = CONFIDENCE,
) -> list[dict[str, Any]]:
    """Later minus earlier raw-primary correlations on the SAME cluster draws.

    These are marginal paired contrasts, not simultaneous evidence or a mechanism.
    Missing rungs stay missing: a contrast never bridges a gap in the declaration.
    """
    records = []
    for earlier, later in zip(rungs, rungs[1:]):
        left, right = cells.get(earlier), cells.get(later)
        record = {"earlier_rung": earlier, "later_rung": later,
                  "direction": "later minus earlier", "confidence": confidence,
                  "interval": None, "point": None, "supports_decline": None}
        reason = "missing adjacent rung"
        if left is not None and right is not None:
            a, b = np.asarray(left["draws"]), np.asarray(right["draws"])
            if a.shape != b.shape:
                raise ValueError("paired contrasts require the same bootstrap draws")
            draws = b - a
            finite = draws[np.isfinite(draws)]
            support = set(left.get("informative_units", [])) & set(right.get("informative_units", []))
            record.update(n_draws=int(draws.size), n_finite_draws=int(finite.size),
                          n_undefined_draws=int(draws.size - finite.size),
                          n_informative_units=len(support),
                          minimum_valid_fraction=MINIMUM_VALID_FRACTION,
                          minimum_valid_draws=MINIMUM_VALID_DRAWS)
            reason = None
            if left.get("point") is None or right.get("point") is None:
                reason = "undefined primary point"
            else:
                record["point"] = float(right["point"] - left["point"])
            if reason is None and len(support) < MINIMUM_BOOTSTRAP_UNITS:
                reason = "too few jointly informative independent backbones"
            if reason is None and (finite.size < MINIMUM_VALID_DRAWS or finite.size / draws.size < MINIMUM_VALID_FRACTION):
                reason = "too few finite paired bootstrap draws"
            if reason is None:
                tail = (1 - confidence) / 2
                record["interval"] = [float(np.quantile(finite, tail)), float(np.quantile(finite, 1 - tail))]
                record["supports_decline"] = record["interval"][1] < 0
        record.update(reason=reason, inferential_available=reason is None)
        records.append(record)
    return records


def continuous_crossing(
    extents: Sequence[int],
    draw_matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    point_estimates: Sequence[float],
    confidence: float = CONFIDENCE,
    primary_inference_available: bool = True,
) -> dict[str, Any]:
    """The extent at which the fitted correlation reaches zero, with an interval.

    Extent enters as ``log2 k``, because the ladder's rungs are roughly
    geometric and a linear-in-k fit would let the k = 40 rung determine the whole
    line. The fit is ordinary least squares over the rungs of one arm, repeated
    inside each cluster-bootstrap draw, so the crossing carries the same
    backbone-level uncertainty the rung intervals do. If required primary-rung
    inference is unavailable, the point fit remains descriptive but no crossing
    interval is issued, regardless of how many draws cross inside the range.

    The crossing is reported only where it falls inside the measured range. A
    draw whose line never reaches zero between the smallest and largest extent is
    counted in ``fraction_without_crossing``: an extrapolated crossing at k = 900
    would be an artefact of the fit and is not a finding about any model.

    The solve stays in ``log2`` throughout and is exponentiated only after it has
    been found to lie inside the measured range. A ladder that is nearly flat
    fits a slope near zero, which puts the crossing at an exponent of order
    10^300 -- and raising two to that overflows rather than producing a number to
    discard, which is how this function first failed on real data. An
    extrapolated crossing is meaningless anyway, so the unconstrained value is
    reported as its exponent and never as an extent.
    """

    k = np.asarray(extents, dtype=np.float64)
    if k.ndim != 1 or k.size < 3 or not np.isfinite(k).all() or (k <= 0).any():
        raise ValueError("a continuous crossing needs at least three positive extents")
    design = np.vstack([np.ones(k.size), np.log2(k)]).T
    matrix = np.asarray(draw_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != k.size or matrix.shape[0] == 0:
        raise ValueError("the draw matrix must be (draws, rungs) aligned with the extents")
    point = np.asarray(point_estimates, dtype=np.float64)
    if point.shape != k.shape:
        raise ValueError("the point estimates must align with the extents")

    def crossing_log2(values: np.ndarray) -> float:
        """Where the fitted line reaches zero, as an exponent of two."""

        if not np.isfinite(values).all():
            return float("nan")
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        intercept, slope = float(coefficients[0]), float(coefficients[1])
        if slope == 0.0:
            return float("nan")
        return -intercept / slope

    low, high = float(k.min()), float(k.max())
    low_log2, high_log2 = float(np.log2(low)), float(np.log2(high))
    observed_log2 = crossing_log2(point)
    draws_log2 = np.asarray([crossing_log2(row) for row in matrix], dtype=np.float64)
    inside_mask = np.isfinite(draws_log2) & (draws_log2 >= low_log2) & (draws_log2 <= high_log2)
    inside = np.exp2(draws_log2[inside_mask])
    in_range = bool(np.isfinite(observed_log2) and low_log2 <= observed_log2 <= high_log2)
    tail = (1.0 - confidence) / 2.0
    interval_reason = (
        "required primary inference unavailable" if not primary_inference_available
        else "fewer than two in-range crossings" if inside.size < 2 else None
    )
    return {
        "axis": "log2 nominal extent in residues",
        "inferential_available": interval_reason is None,
        "interval_interpretation": "conditional on a fitted crossing inside the measured range",
        "interval_reason": interval_reason,
        "n_undefined_draws": int((~np.isfinite(matrix).all(axis=1)).sum()),
        "extents": [int(value) for value in k],
        "crossing_extent": float(np.exp2(observed_log2)) if in_range else None,
        "crossing_reason": None if in_range else (
            "undefined primary estimates" if not np.isfinite(point).all()
            else "fitted crossing absent or outside measured range"
        ),
        "crossing_log2_extent_unconstrained": (
            float(observed_log2) if np.isfinite(observed_log2) else None
        ),
        "interval": (
            [float(np.quantile(inside, tail)), float(np.quantile(inside, 1.0 - tail))]
            if interval_reason is None
            else None
        ),
        "confidence": float(confidence),
        "n_draws": int(matrix.shape[0]),
        "n_draws_with_crossing_in_range": int(inside.size),
        "fraction_without_crossing": float(1.0 - inside.size / matrix.shape[0]),
        "measured_range": [low, high],
    }


def required_units(
    *, bootstrap_sd: float, n_units: int, target_effect: float, confidence: float = CONFIDENCE
) -> dict[str, Any]:
    """How many backbones would be needed to resolve an effect of ``target_effect``.

    The honest answer when a ladder does not resolve. The bootstrap standard
    deviation of a cluster-resampled statistic scales as one over the square root
    of the number of clusters, so the required count follows from the observed
    one; power 0.8 at the stated two-sided confidence is the convention and is
    declared rather than implied. This is a planning figure under a scaling
    assumption, not a measurement.
    """

    if n_units < MINIMUM_BOOTSTRAP_UNITS:
        raise ValueError("a scaling statement needs an interval that was admissible")
    if not np.isfinite(bootstrap_sd) or bootstrap_sd <= 0.0:
        raise ValueError("a scaling statement needs a positive finite bootstrap sd")
    if target_effect == 0.0:
        raise ValueError("a target effect of zero cannot be resolved at any sample size")


    z_alpha = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    z_power = float(stats.norm.ppf(0.8))
    needed_sd = abs(target_effect) / (z_alpha + z_power)
    units = int(math.ceil(n_units * (bootstrap_sd / needed_sd) ** 2))
    return {
        "observed_bootstrap_sd": float(bootstrap_sd),
        "observed_n_units": int(n_units),
        "target_effect": float(target_effect),
        "power": 0.8,
        "confidence": float(confidence),
        "required_n_units": units,
        "assumption": (
            "the cluster-bootstrap standard deviation scales as the inverse square "
            "root of the number of backbones, and a larger backbone set has the same "
            "per-backbone draw count and the same effect size"
        ),
    }
