"""Estimators for the reward-provenance experiment, under the project's own band.

Declared in ``docs/rl-environment-provenance.md``; every size, seed, family and
threshold this module reads comes from
:mod:`src.capability.extensions.rl_environment_provenance`. Nothing here chooses
a constant.

The central quantity is the best-of-k control. For each pool and each declared
``k``, it asks what the base policy would have scored if ``k`` of its own samples
had been drawn and the best one kept under the identical reward -- and compares
that with what the trained policy scores on average. The smallest ``k`` at which
selection matches the policy is ``k*``. A small ``k*`` says the optimisation
redistributed mass the base already held, because a rejection sampler with no
weight update and three orders of magnitude fewer reward evaluations reached the
same place. A large ``k*`` says it did not.

Two properties of that estimator matter enough to be implemented rather than
assumed.

*It is read off order statistics, not a second Monte Carlo layer.* Given a pool,
the expected maximum over ``k`` draws with replacement is an exact weighted sum
of the pool's order statistics, so the only sampling error in ``R@k`` is the
pool's own, which is what the band resamples.

*It is cross-fitted whenever the reward is noisy.* A reward containing the pLDDT
term is measured with between-realisation spread the project has twice measured,
and the expected maximum of a noisy score is inflated by that noise while a mean
is not -- so a naive control would be flattered, and ``k*`` biased downward,
which is the direction that favours this project's own thesis. Selection is
therefore made on one folded realisation and the selected sample's reward read
from an independent one.

The band is the project's native statistic -- the maximum absolute centered
bootstrap deviation over the family, in the columns' own units, with no standard
error divided out -- taken through the one implementation of it that this
programme has, :func:`..generation.objective_tuned_analysis.simultaneous_band`.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.statistics import MINIMUM_BOOTSTRAP_UNITS
from ..extensions import bidirectional as bd
from ..generation.objective_tuned_analysis import simultaneous_band
from . import rl_environment_provenance as rp

#: The resampling floor, per (pool, stream), from the project's single source.
MIN_UNITS_PER_POOL = MINIMUM_BOOTSTRAP_UNITS

ADMISSIBILITY_RULE = (
    f"every contributing (pool, stream) carries at least {MIN_UNITS_PER_POOL} "
    "sequences; a column below the floor on any one of them is recorded "
    "unresolved with its counts and never reported as an estimate"
)

RESAMPLING_UNIT = (
    "the generated sequence, resampled with replacement within (pool, stream). "
    "This is the decoder's sampling noise at fixed weights. Each arm is one "
    "training run, so nothing here covers the variation between runs"
)

NATIVE_BAND_METHOD = (
    bd.NATIVE_METHOD
    + "; two independent pools per column, resampled within stream, stream-equal "
    "pool means; for a best-of-k column the base side is the pool's own order "
    "statistics recomputed on each draw"
)


# ----------------------------------------------------------- pools and draws

def pool_units(streams: Mapping[int, Sequence[float]]) -> dict[str, int]:
    return {str(stream): int(len(streams[stream])) for stream in sorted(streams)}


def pool_shortfall(streams: Mapping[int, Sequence[float]]) -> list[str]:
    """Streams of one pool below the declared floor, in order."""

    if not streams:
        return ["<no stream>"]
    return [str(stream) for stream in sorted(streams)
            if len(streams[stream]) < MIN_UNITS_PER_POOL]


def streams_by_digest(rows: Sequence[Mapping[str, Any]],
                      scores: Mapping[str, float], *, pool: str,
                      quantity: str) -> dict[int, list[float]]:
    """One pool's values laid out by stream, joined to its attempts on sequence digest.

    Three instruments -- the fold, the likelihood and the homology search -- all
    measure a *distinct sequence* and all have to be read back onto *attempts*,
    because the attempt is the declared resampling unit and the stream is its
    stratum. This is that join, in one place, so the three cannot drift into three
    spellings of it. An empty attempt carries no measurement of any of the three
    and is skipped; an attempt with a sequence and no score is a refusal, because a
    family assembled from a partial pool is not a family over that pool.
    """

    out: dict[int, list[float]] = {}
    missing: list[str] = []
    for row in rows:
        if int(row["length"]) == 0:
            continue
        value = scores.get(str(row["sequence_sha256"]))
        if value is None:
            missing.append(str(row["sequence_sha256"]))
            continue
        out.setdefault(int(row["stream"]), []).append(float(value))
    if missing:
        raise ValueError(
            f"{len(missing)} attempts of pool {pool!r} have no {quantity} in the "
            f"supplied table, the first being {missing[0]}; a family is assembled "
            "from a complete pool and a partial one is not reported as a pool"
        )
    if not out:
        raise ValueError(f"pool {pool!r} carries no attempt with a {quantity}")
    return out


def pool_mean(streams: Mapping[int, Sequence[float]]) -> float:
    """The stream-equal mean of one pool.

    Stream-equal rather than sequence-equal because the streams are the declared
    replication axis and carry equal budgets by construction, so a stream that
    retained fewer non-empty attempts must not thereby weigh less.
    """

    return float(np.mean([np.mean(np.asarray(streams[stream], dtype=float))
                          for stream in sorted(streams)]))


def _stream_resample(rng: np.random.Generator, size: int, count: int) -> np.ndarray:
    return rng.integers(0, size, size=(count, size))


def pool_draws(streams: Mapping[int, Sequence[float]], *, family: str, column: int,
               side: int, draws: int = rp.BAND_DRAWS) -> np.ndarray:
    """``draws`` resampled stream-equal pool means, resampled within stream."""

    per_stream = []
    for index, stream in enumerate(sorted(streams)):
        values = np.asarray(streams[stream], dtype=float).reshape(-1, 1)
        if not np.isfinite(values).all():
            raise ValueError(f"stream {stream} of a pool carries a non-finite value")
        seed = rp.pool_draw_seed(family, column, side, index)
        per_stream.append(bd.resample_means(values, draws=draws, seed=seed)[:, 0])
    return np.mean(np.stack(per_stream, axis=0), axis=0)


# ----------------------------------------------------------- best-of-k

def best_of_k_weights(n: int, k: int) -> np.ndarray:
    """Probability that the best of ``k`` uniform draws with replacement has each rank.

    Ranks run from best to worst. The weight on rank ``i`` (one-based) is
    ``((n - i + 1)/n)**k - ((n - i)/n)**k``: the chance that every one of the
    ``k`` draws lands at rank ``i`` or worse, minus the chance that every one
    lands strictly worse. At ``k = 1`` this is uniform, and as ``k`` grows the
    mass moves onto rank one, which is what selection does.

    :func:`rank_weights` is the same statement for a pool whose members are not
    equally likely to be drawn, which is the case this experiment's declared
    stream-equal weighting puts it in.
    """

    if n < 1 or k < 1:
        raise ValueError("a best-of-k weight needs a non-empty pool and a positive k")
    tail = np.arange(n, -1, -1, dtype=float) / float(n)
    powers = tail ** float(k)
    return powers[:-1] - powers[1:]


def rank_weights(ranked_probability: np.ndarray, k: int) -> np.ndarray:
    """The same weights for a pool whose members carry unequal draw probabilities.

    ``ranked_probability`` is each member's probability of being the one drawn,
    in rank order, best first, summing to one along the last axis. The weight on
    a rank is the chance that all ``k`` draws land at that rank or worse minus
    the chance that all land strictly worse, which for unequal probabilities is a
    difference of powers of the *cumulative tail* rather than of ``(n - i)/n``.

    This exists because the declared pool mean is stream-equal and the best-of-k
    side was pooling sequence-equally. The two sides of one contrast have to be
    the same estimator, and the invariant that makes that checkable is ``R@1``
    equalling the pool mean exactly, so that the ``k = 1`` column is precisely
    minus the reward gain and ``k = 1`` is the no-selection baseline it is
    supposed to be.
    """

    if k < 1:
        raise ValueError("a best-of-k weight needs a positive k")
    tail = np.concatenate(
        [np.cumsum(ranked_probability[..., ::-1], axis=-1)[..., ::-1],
         np.zeros(ranked_probability.shape[:-1] + (1,))], axis=-1
    )
    powers = tail ** float(k)
    return powers[..., :-1] - powers[..., 1:]


def stream_equal_member_probability(sizes: Sequence[int]) -> np.ndarray:
    """Each member's draw probability under the declared stream-equal weighting.

    A draw picks one of the declared streams with equal probability and then one
    of that stream's members with equal probability, which is exactly what a
    stream-equal pool mean is the expectation of. Streams of equal size give the
    uniform weights back.
    """

    counts = [int(size) for size in sizes]
    if not counts or any(size < 1 for size in counts):
        raise ValueError("a stream-equal weighting needs a non-empty size per stream")
    share = 1.0 / len(counts)
    return np.concatenate([np.full(size, share / size, dtype=float) for size in counts])


def _tie_averaged(selection: np.ndarray, evaluation: np.ndarray) -> np.ndarray:
    """Evaluation values with each tie group of ``selection`` replaced by its mean.

    Ties in the selection score are real: a shaped reward is exactly zero for
    every completion far outside its windows, and exactly one for every
    completion inside them. Which member of a tie group a selector returns is
    arbitrary, so the expectation under a uniform tie-break is the group's mean
    evaluation value, and that is what the weight on the group's ranks should
    multiply. Rows are sorted descending by selection score before this is
    called.
    """

    starts = np.empty(selection.shape, dtype=bool)
    starts[:, 0] = True
    starts[:, 1:] = selection[:, 1:] != selection[:, :-1]
    groups = np.cumsum(starts, axis=1) - 1
    out = np.empty_like(evaluation)
    for row in range(selection.shape[0]):
        ids = groups[row]
        counts = np.bincount(ids)
        sums = np.bincount(ids, weights=evaluation[row])
        out[row] = (sums / counts)[ids]
    return out


def best_of_k(selection: Sequence[float], evaluation: Sequence[float],
              ks: Sequence[int] = rp.K_GRID,
              member_probability: Sequence[float] | None = None) -> np.ndarray:
    """``R@k`` for each ``k``: the expected evaluation value of the selected sample.

    ``selection`` is the score the selector ranks by and ``evaluation`` the score
    the selected sample is credited with. They are the same vector for an exactly
    computable reward and two independent folded realisations for a reward that
    contains the pLDDT term.

    ``member_probability`` is each member's chance of being the one drawn, which
    the declared stream-equal weighting makes non-uniform whenever the streams
    retained different numbers of non-empty attempts. ``None`` is the uniform
    case and is what the closed-form checks use.
    """

    rows = _best_of_k_rows(np.asarray([selection], dtype=float),
                           np.asarray([evaluation], dtype=float), ks,
                           member_probability=member_probability)
    return rows[0]


def _best_of_k_rows(selection: np.ndarray, evaluation: np.ndarray,
                    ks: Sequence[int],
                    member_probability: Sequence[float] | None = None) -> np.ndarray:
    """``R@k`` per row of a (rows, pool) pair of score matrices."""

    if selection.shape != evaluation.shape or selection.ndim != 2:
        raise ValueError("best-of-k needs matching (rows, pool) selection and evaluation")
    if not np.isfinite(selection).all() or not np.isfinite(evaluation).all():
        raise ValueError("best-of-k needs a finite score on every pool member")
    pool = selection.shape[1]
    if pool < 1:
        raise ValueError("best-of-k needs a non-empty pool")
    order = np.argsort(-selection, axis=1, kind="stable")
    index = np.arange(selection.shape[0])[:, None]
    ranked_selection = selection[index, order]
    ranked_evaluation = _tie_averaged(ranked_selection, evaluation[index, order])
    if member_probability is None:
        weights = np.stack([best_of_k_weights(pool, int(k)) for k in ks], axis=1)
        return ranked_evaluation @ weights
    probability = np.asarray(member_probability, dtype=float)
    if probability.shape != (pool,):
        raise ValueError(
            f"one draw probability per pool member; got {probability.shape} for {pool}"
        )
    if not math.isclose(float(probability.sum()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("the member draw probabilities sum to one")
    ranked_probability = probability[order]
    out = np.empty((selection.shape[0], len(ks)), dtype=float)
    for position, k in enumerate(ks):
        out[:, position] = (ranked_evaluation
                            * rank_weights(ranked_probability, int(k))).sum(axis=1)
    return out


def best_of_k_draws(selection: Mapping[int, Sequence[float]],
                    evaluation: Mapping[int, Sequence[float]],
                    *, family: str, column: int, ks: Sequence[int] = rp.K_GRID,
                    draws: int = rp.BAND_DRAWS) -> np.ndarray:
    """``(draws, len(ks))`` resampled ``R@k`` values of one pool.

    Each draw resamples every stream with replacement to its own size and
    concatenates them, so the resampled pool carries the declared stream
    composition and the selection is then allowed to range over the whole pool,
    which is how a rejection sampler would use it. ``R@k`` is read off that
    resampled pool's order statistics exactly, so the draw adds no second layer
    of Monte Carlo error.
    """

    order = sorted(selection)
    if order != sorted(evaluation):
        raise ValueError("the selection and evaluation realisations carry different streams")
    blocks_selection, blocks_evaluation, generators = [], [], []
    for index, stream in enumerate(order):
        chosen = np.asarray(selection[stream], dtype=float)
        scored = np.asarray(evaluation[stream], dtype=float)
        if chosen.shape != scored.shape:
            raise ValueError(
                f"stream {stream} carries {chosen.size} selection values against "
                f"{scored.size} evaluation values; the two realisations must cover "
                "the same sequences in the same order"
            )
        blocks_selection.append(chosen)
        blocks_evaluation.append(scored)
        generators.append(np.random.default_rng(rp.pool_draw_seed(family, column, 0, index)))
    # The declared pool mean is stream-equal, so the draw probability a best-of-k
    # curve is taken under has to be stream-equal too. Pooling sequence-equally --
    # which this did until the k = 1 column was checked against the reward gain it
    # is supposed to be the negative of -- gives the two sides of one contrast two
    # different estimators whenever the streams retained different numbers of
    # non-empty attempts.
    probability = stream_equal_member_probability([block.size for block in
                                                   blocks_selection])
    out = np.empty((draws, len(ks)), dtype=float)
    for start in range(0, draws, bd.DRAW_CHUNK):
        stop = min(start + bd.DRAW_CHUNK, draws)
        count = stop - start
        # One index draw per stream, used for both realisations: the selection
        # and the evaluation of a draw have to be the same resampled pool read
        # two ways, or the cross-fit would compare different pools.
        indices = [_stream_resample(rng, block.size, count)
                   for rng, block in zip(generators, blocks_selection)]
        chosen = np.concatenate(
            [block[index] for block, index in zip(blocks_selection, indices)], axis=1
        )
        scored = np.concatenate(
            [block[index] for block, index in zip(blocks_evaluation, indices)], axis=1
        )
        out[start:stop] = _best_of_k_rows(chosen, scored, ks,
                                          member_probability=probability)
    return out


# ---------------------------------------------------------------- the bands

def contrast_bands(columns: Sequence[Mapping[str, Any]], *, family: str,
                   draws: int = rp.BAND_DRAWS,
                   confidence: float = rp.BAND_CONFIDENCE) -> dict[str, Any]:
    """One simultaneous band per column over a declared family of pool contrasts.

    ``columns`` is the family in declared order. Each entry carries ``key`` and
    either two pools, ``left`` and ``right`` as ``{stream: values}``, whose
    difference is the column -- or a precomputed pair of draw vectors under
    ``left_draws``/``right_draws`` with their points, which is how a best-of-k
    column enters: its left side is not a pool mean.

    A column whose pools do not clear the resampling floor is recorded
    unresolved and contributes no draws, so the critical value covers the
    resolved subfamily and both sizes are recorded.
    """

    keys = [str(column["key"]) for column in columns]
    declared = rp.FAMILY_SIZES.get(family)
    if declared is None:
        raise KeyError(f"no declared size for family {family!r}")
    if len(set(keys)) != len(keys):
        raise ValueError(f"family {family} repeats a column key")
    if len(keys) != declared:
        raise ValueError(
            f"family {family} declares {declared} columns and realises {len(keys)}: "
            f"{keys}. Changing a declared family size is an amendment to "
            "docs/rl-environment-provenance.md, not a run-time decision"
        )
    if not 0 < confidence < 1:
        raise ValueError("a simultaneous confidence level lies strictly between zero and one")

    entries: dict[str, dict[str, Any]] = {}
    resolved: list[str] = []
    deviations: list[np.ndarray] = []
    for index, column in enumerate(columns):
        key = keys[index]
        entry: dict[str, Any] = {"minimum_units_per_pool": MIN_UNITS_PER_POOL}
        skip = column.get("skip_reason")
        if skip:
            entry.update(status="unresolved", reason=str(skip), point=None, interval=None)
            entries[key] = entry
            continue
        if "left_draws" in column:
            left_point = float(column["left_point"])
            right_point = float(column["right_point"])
            left_draws = np.asarray(column["left_draws"], dtype=float)
            right_draws = np.asarray(column["right_draws"], dtype=float)
            entry["units"] = column.get("units")
        else:
            left, right = column["left"], column["right"]
            short = {"left": pool_shortfall(left), "right": pool_shortfall(right)}
            entry["units"] = {"left": pool_units(left), "right": pool_units(right)}
            if short["left"] or short["right"]:
                entry.update(
                    status="unresolved",
                    reason="below the resampling floor on stream(s) "
                           f"left={short['left']} right={short['right']}; "
                           + ADMISSIBILITY_RULE,
                    point=None, interval=None,
                )
                entries[key] = entry
                continue
            left_point, right_point = pool_mean(left), pool_mean(right)
            left_draws = pool_draws(left, family=family, column=index, side=0, draws=draws)
            right_draws = pool_draws(right, family=family, column=index, side=1, draws=draws)
        point = left_point - right_point
        entry.update(status="resolved", reason=None, point=point,
                     left_point=left_point, right_point=right_point)
        entries[key] = entry
        resolved.append(key)
        deviations.append((left_draws - right_draws) - point)

    record: dict[str, Any] = {
        "family": family,
        "method": NATIVE_BAND_METHOD,
        "resampling_unit": RESAMPLING_UNIT,
        "draws": int(draws),
        "seed": rp.BAND_SEED,
        "seed_rule": "BAND_SEED + FAMILY_SEED_OFFSETS[family] + 1000*column + 100*side + stream",
        "confidence": float(confidence),
        "declared_family_size": declared,
        "resolved_family_size": len(resolved),
        "simultaneity_scope": (
            "the critical value is the maximum over the resolved columns only; an "
            "unresolved column carries no estimate and no coverage claim is made "
            "for the declared family beyond its resolved subfamily"
        ),
        "admissibility": ADMISSIBILITY_RULE,
        "columns": entries,
    }
    if not resolved:
        record.update(critical_value=None, standardized=None,
                      reason="no column of this family clears the resampling floor")
        return record
    record.update(simultaneous_band(entries, resolved, deviations, confidence=confidence))
    return record


# -------------------------------------------------------------- the readings

def optimisation_finding(interval: Sequence[float] | None) -> dict[str, Any]:
    """Whether a pool's reward gain under its own reward is bounded above zero."""

    if interval is None:
        return {"optimised": None,
                "reason": "the reward-gain column does not resolve, so whether the "
                          "optimisation moved the reward is not measured"}
    low, high = float(interval[0]), float(interval[1])
    if low > 0.0:
        return {"optimised": True,
                "reason": f"the whole interval [{low:+.4f}, {high:+.4f}] lies above zero"}
    if high < 0.0:
        return {"optimised": False,
                "reason": f"the whole interval [{low:+.4f}, {high:+.4f}] lies below zero: the "
                          "optimisation moved the reward the wrong way"}
    return {"optimised": False,
            "reason": f"the interval [{low:+.4f}, {high:+.4f}] contains zero, so no movement "
                      "is established; this is not evidence that the capability is absent"}


def k_star(columns: Mapping[str, Mapping[str, Any]], *, pool: str,
           ks: Sequence[int] = rp.K_GRID) -> dict[str, Any]:
    """The smallest declared ``k`` at which selection over the base pool matches a pool.

    A column of the best-of-k family is ``R@k(base) - mean(pool)``, so selection
    has matched the pool when that column's whole interval lies at or above zero.
    ``k*`` is the first such ``k`` of the declared grid. A grid on which no ``k``
    reaches it returns ``None`` with the ceiling named, never an infinity.
    """

    reached: list[int] = []
    unresolved: list[int] = []
    for k in ks:
        entry = columns.get(f"{pool}|k={k}")
        if entry is None or entry.get("status") != "resolved" or entry.get("interval") is None:
            unresolved.append(int(k))
            continue
        if float(entry["interval"][0]) >= 0.0:
            reached.append(int(k))
    value = min(reached) if reached else None
    out = {
        "pool": pool,
        "k_grid": [int(k) for k in ks],
        "k_star": value,
        "k_values_reached": reached,
        "k_values_unresolved": unresolved,
        "k_ceiling": int(ks[-1]),
        "threshold": rp.K_REDISTRIBUTION,
        "rule": rp.K_STAR_RULE,
    }
    if value is None:
        out["reason"] = (
            f"no declared k up to {ks[-1]} brings selection over the base pool to this "
            "pool's mean reward, so k* is greater than the grid's ceiling"
        )
        out["redistribution"] = False if not unresolved else None
    else:
        out["redistribution"] = value <= rp.K_REDISTRIBUTION
    return out


def reachability_curve(statistics: Mapping[int, Sequence[Any]], *,
                       ladder: Sequence[tuple[float, float]] = rp.CHARGE_LADDER,
                       ks: Sequence[int] = rp.K_GRID,
                       draws: int = rp.BAND_DRAWS,
                       confidence: float = rp.BAND_CONFIDENCE) -> dict[str, Any]:
    """The base pool's expected best-of-k environment reward at each declared rung.

    ``statistics`` maps a stream to that stream's
    :class:`~.sequence_reward.SequenceStatistics`, in attempt order. Each rung's
    reward is recomputed from those same measured statistics against that rung's
    charge window, so the whole curve costs one pass over numbers already in hand
    and no generation and no fold.

    The curve is read off the base pool alone, so it is not a contrast and carries
    no simultaneous contrast band. Each rung and each ``k`` gets a percentile
    interval from the declared draws, which is stated as its construction rather
    than borrowed from the families' statistic. The environment reward is exactly
    computable, so no cross-fit applies and selection and evaluation are the same
    numbers.

    What it decides is declared in :data:`..rl_environment_provenance.LADDER_NOTE`:
    whether the declared target was base-reachable, which is one half of what a
    small ``k*`` leaves ambiguous. It does not decide the other half.
    """

    from . import sequence_reward as sr

    streams = sorted(statistics)
    rungs = []
    for index, window in enumerate(ladder):
        per_stream = {
            stream: [sr.environment_reward_at(stats, charge_window=window)
                     for stats in statistics[stream]]
            for stream in streams
        }
        short = pool_shortfall(per_stream)
        if short:
            rungs.append({"window": list(window), "status": "unresolved",
                          "reason": f"below the resampling floor on stream(s) {short}"})
            continue
        # The shaped reward is the mean of a charge term and a length term, and the
        # length term is the easy one, so a best-of-k curve on the shaped reward
        # stays high at every rung for reasons that have nothing to do with charge.
        # The conjunction indicator is where target difficulty actually bites, and
        # a rejection sampler's own hit probability is read off it directly. Both
        # are reported; the conjunction is the one that answers the question.
        hit = {
            stream: [1.0 if (window[0] <= stats.net_charge <= window[1]
                             and sr.LENGTH_WINDOW[0] <= stats.length <= sr.LENGTH_WINDOW[1]
                             and stats.low_complexity_fraction < sr.LOW_COMPLEXITY_LIMIT)
                     else 0.0
                     for stats in statistics[stream]]
            for stream in streams
        }
        pooled = np.concatenate([np.asarray(per_stream[stream], dtype=float)
                                 for stream in streams])
        pooled_hit = np.concatenate([np.asarray(hit[stream], dtype=float)
                                     for stream in streams])
        probability = stream_equal_member_probability(
            [len(per_stream[stream]) for stream in streams])
        point = best_of_k(pooled, pooled, ks, probability)
        hit_point = best_of_k(pooled_hit, pooled_hit, ks, probability)
        drawn = best_of_k_draws(per_stream, per_stream, family=rp.FAMILY_BEST_OF_K
                                + "__environment", column=50 + index, ks=ks, draws=draws)
        drawn_hit = best_of_k_draws(hit, hit, family=rp.FAMILY_BEST_OF_K
                                    + "__environment", column=70 + index, ks=ks,
                                    draws=draws)
        low = (1.0 - confidence) / 2.0

        def banded(points, draw_matrix):
            return {str(k): {"point": float(points[position]),
                             "interval": [
                                 float(np.quantile(draw_matrix[:, position], low)),
                                 float(np.quantile(draw_matrix[:, position], 1.0 - low))]}
                    for position, k in enumerate(ks)}

        rungs.append({
            "window": list(window),
            "status": "resolved",
            # the share of the base pool that satisfies this rung's conjunction, which
            # is the probability a single base draw hits the target
            # stream-equal, as every mean in this experiment is
            "conjunction_rate": float(pooled_hit @ probability),
            "mean_reward": float(pooled @ probability),
            "r_at_k": banded(point, drawn),
            "conjunction_r_at_k": banded(hit_point, drawn_hit),
        })
    return {
        "diagnostic": "base_pool_reachability_ladder",
        "note": rp.LADDER_NOTE,
        "method": ("expected best-of-k environment reward over the base pool's own "
                   "order statistics at each declared charge window, with percentile "
                   "intervals from the declared resample of the base pool; a level "
                   "curve on one pool, not a contrast, and not read under the "
                   "families' simultaneous band"),
        "k_grid": [int(k) for k in ks],
        "draws": int(draws),
        "confidence": float(confidence),
        "rungs": rungs,
    }


def conjunction_rate_upper_bound(hits: int, n: int, *,
                                 confidence: float = rp.RUNG_CONFIDENCE) -> float:
    """One-sided exact-binomial upper bound on a conjunction rate.

    Clopper-Pearson, which is the Beta quantile at ``(hits + 1, n - hits)``. It is
    here rather than a bootstrap because the quantity it bounds is sometimes zero
    *by observation*: a charge window the base pool never hits has an all-zero
    indicator, and the nonparametric bootstrap of an all-zero vector is a
    zero-width interval at zero. That would assert unreachability with no
    uncertainty, at exactly the point where unreachability is the thing being
    decided. An exact bound sees the rare event the sample missed.
    """

    from scipy.stats import beta

    if n < 1 or not 0 <= hits <= n:
        raise ValueError(f"a rate bound needs 0 <= hits <= n with n positive; got {hits}/{n}")
    if not 0 < confidence < 1:
        raise ValueError("a one-sided confidence level lies strictly between zero and one")
    if hits == n:
        return 1.0
    return float(beta.ppf(confidence, hits + 1, n - hits))


def rejection_hit_probability(rate: float, budget: int) -> float:
    """Probability that a ``budget``-sample rejection sampler hits a rate-``rate`` target."""

    if not 0.0 <= rate <= 1.0 or budget < 1:
        raise ValueError("a hit probability needs a rate in [0, 1] and a positive budget")
    return float(1.0 - (1.0 - rate) ** int(budget))


def choose_hard_rung(ladder_rungs: Sequence[Mapping[str, Any]], *, n: int,
                     budget: int = rp.RUNG_BUDGET,
                     bound: float = rp.RUNG_UNREACHABLE_BOUND,
                     confidence: float = rp.RUNG_CONFIDENCE) -> dict[str, Any]:
    """The first declared rung that a rejection sampler of ``budget`` cannot reach.

    ``ladder_rungs`` is :func:`reachability_curve`'s own ``rungs`` list and ``n``
    the base pool's attempt count the rates were taken over. The rule is declared
    in :data:`..rl_environment_provenance.RUNG_RULE`, and the one outcome it
    refuses to engineer around is named there: if no rung clears the bound, the
    coordinate cannot support a positive control and the ladder is not extended.
    """

    examined = []
    chosen = None
    for rung in ladder_rungs:
        if rung.get("status") != "resolved":
            examined.append({"window": rung.get("window"), "status": rung.get("status"),
                             "reason": rung.get("reason")})
            continue
        rate = float(rung["conjunction_rate"])
        hits = int(round(rate * n))
        upper = conjunction_rate_upper_bound(hits, n, confidence=confidence)
        reach = rejection_hit_probability(upper, budget)
        entry = {"window": list(rung["window"]), "status": "resolved",
                 "conjunction_hits": hits, "conjunction_rate": rate,
                 "rate_upper_bound": upper,
                 f"hit_probability_upper_bound_at_k{budget}": reach,
                 "unreachable": bool(reach < bound)}
        examined.append(entry)
        if chosen is None and entry["unreachable"]:
            chosen = entry
    out = {
        "rule": rp.RUNG_RULE,
        "budget": int(budget),
        "bound": float(bound),
        "confidence": float(confidence),
        "base_pool_attempts": int(n),
        "rungs_examined": examined,
        "chosen_window": None if chosen is None else chosen["window"],
    }
    if chosen is None:
        out["reason"] = (
            f"no rung of the declared ladder is unreachable at k = {budget}: rejection "
            "sampling from the base pool reaches every charge target on this "
            "coordinate at that budget. The ladder is not extended until something "
            "qualifies, which is declared in the rule above. This is a "
            "statement about the base distribution and not about reinforcement "
            "learning, and it means the coordinate cannot support a positive control"
        )
    return out


#: Whether a pool satisfied the objective it was paid to approach, or only
#: approached it. Three states, because the middle one is a finding rather than a
#: shortfall: a rising shaped reward with a flat attainment rate is a policy
#: learning to approach constraints without meeting them.
ATTAINED = "objective_attained"
APPROACHED = "approached_without_attaining"
ATTAINMENT_UNMEASURED = "not_measured"
ATTAINMENT_STATES: tuple[str, ...] = (ATTAINED, APPROACHED, ATTAINMENT_UNMEASURED)


def objective_attainment(gain: Mapping[str, Any] | None,
                         rate: Mapping[str, Any] | None, *,
                         base_rate: float | None = None) -> dict[str, Any]:
    """Whether a pool attained its own objective, beside whether its reward rose.

    ``gain`` is the pool's reward-gain column and ``rate`` its own objective's
    attainment-rate column -- the share of completions satisfying that reward's
    declared conjunction outright. Both are contrasts against the base pool.

    This is reported for every arm and every pool, not derived on request, because
    of what the base pool showed: 0.9963 on the shaped environment reward while
    satisfying that objective zero times in 3,072 draws, missing on charge +7.98
    against a window opening at +8.0 and a low-complexity fraction of 0.252
    against a limit closing at 0.25. A shaped reward prices a constraint and a
    conjunction enforces it, so **a reward gain is not evidence that the objective
    was attained** -- and every inferential step in this experiment runs through
    reward gain.

    The middle state is the one worth having a name for. A pool whose shaped
    reward rose while its attainment rate did not has learned to approach
    constraints without meeting them, which is not a failure of the measurement
    but very nearly this project's own claim about redistribution inside a
    narrower domain.
    """

    def interval(entry: Mapping[str, Any] | None) -> tuple[float, float] | None:
        if entry is None or entry.get("status") != "resolved" or entry.get("interval") is None:
            return None
        return float(entry["interval"][0]), float(entry["interval"][1])

    gain_interval = interval(gain)
    rate_interval = interval(rate)
    out: dict[str, Any] = {
        "reward_gain_interval": list(gain_interval) if gain_interval else None,
        "attainment_rate_contrast": list(rate_interval) if rate_interval else None,
        "attainment_rate_point": None if rate is None else rate.get("point"),
        "base_attainment_rate": base_rate,
        "note": rp.ATTAINMENT_NOTE,
    }
    if rate_interval is None:
        out.update(state=ATTAINMENT_UNMEASURED,
                   reason="the pool's own objective-attainment rate does not resolve, "
                          "so whether a reward gain corresponds to attaining the "
                          "objective is not measured")
        return out
    if rate_interval[0] > 0.0:
        out.update(state=ATTAINED,
                   reason="the attainment rate is bounded above the base pool's: the "
                          "pool satisfies its own objective outright more often, not "
                          "only nearer")
        return out
    rose = gain_interval is not None and gain_interval[0] > 0.0
    out.update(
        state=APPROACHED,
        reason=("the shaped reward rose while the attainment rate did not: the pool "
                "learned to approach its objective's constraints without meeting "
                "them, which is a finding and not a shortfall of the measurement"
                if rose else
                "neither the shaped reward nor the attainment rate is bounded above "
                "the base pool's"),
    )
    return out


def choose_shaped_rung(ladder_rungs: Sequence[Mapping[str, Any]], *,
                       budget: int = rp.RUNG_SHAPED_BUDGET,
                       bound: float = rp.RUNG_SHAPED_BOUND) -> dict[str, Any]:
    """The first declared rung whose best-of-``budget`` shaped reward is below ``bound``.

    The ladder's second rule, posed on the shaped reward because the first was
    posed on the conjunction indicator and the base pool attains that nowhere --
    so the first rule's answer was degenerate by construction rather than merely
    unwelcome, which is the only ground on which a declared rule may be voided.
    :data:`..rl_environment_provenance.RUNG_RULE_VOIDED` carries that argument and
    :data:`..rl_environment_provenance.RUNG_SHAPED_RULE` the replacement.

    A rule has to be posed on a quantity that discriminates, and this one does:
    the pool means run 0.3221 down to 0.1666 across the ladder and the
    best-of-512 values 0.9705 down to 0.6231.
    """

    examined = []
    chosen = None
    for rung in ladder_rungs:
        if rung.get("status") != "resolved":
            examined.append({"window": rung.get("window"), "status": rung.get("status")})
            continue
        value = float(rung["r_at_k"][str(int(budget))]["point"])
        entry = {"window": list(rung["window"]), "status": "resolved",
                 f"r_at_k{budget}": value, "below_bound": bool(value < bound)}
        examined.append(entry)
        if chosen is None and entry["below_bound"]:
            chosen = entry
    out = {"rule": rp.RUNG_SHAPED_RULE, "budget": int(budget), "bound": float(bound),
            "rungs_examined": examined,
            "chosen_window": None if chosen is None else chosen["window"]}
    if chosen is None:
        out["reason"] = (
            f"no rung of the declared ladder has a best-of-{budget} shaped reward "
            f"below {bound}, so selection reaches every charge target on this "
            "coordinate at that budget and the ladder is not extended"
        )
    return out


def support_tail_share(pool_nll: Sequence[float], base_pool_nll: Sequence[float], *,
                       quantile: float = rp.SUPPORT_TAIL_QUANTILE) -> dict[str, Any]:
    """Share of a pool's samples the base model finds less probable than its own tail.

    The threshold is the base pool's own ``quantile`` of per-residue NLL under
    the base model, so the statistic's null value is one minus that quantile and
    a share bounded above it is mass the base model's memory did not place. This
    is the support reading the bounded rewards cannot carry: a reward capped at
    one, which the base pool attains, admits nothing outside its own range.
    """

    base = np.asarray(base_pool_nll, dtype=float)
    pool = np.asarray(pool_nll, dtype=float)
    if base.size < 1 or pool.size < 1:
        raise ValueError("a support share needs a non-empty pool on both sides")
    if not np.isfinite(base).all() or not np.isfinite(pool).all():
        raise ValueError("a support share needs a finite NLL on every sequence")
    threshold = float(np.quantile(base, quantile))
    return {
        "quantile": float(quantile),
        "null_share": float(1.0 - quantile),
        "threshold_nats_per_residue": threshold,
        "share_above": float(np.mean(pool > threshold)),
        "n": int(pool.size),
    }


def reward_range_share(pool_rewards: Sequence[float],
                       base_rewards: Sequence[float]) -> dict[str, Any]:
    """Share of a pool's rewards outside the base pool's observed reward range.

    Reported with its ceiling stated. The declared rewards are bounded above by
    one; where the base pool attains that bound this share is identically zero
    whatever the policy did, which is why it does not carry the support reading
    on its own.
    """

    base = np.asarray(base_rewards, dtype=float)
    pool = np.asarray(pool_rewards, dtype=float)
    low, high = float(base.min()), float(base.max())
    return {
        "base_observed_range": [low, high],
        "share_above": float(np.mean(pool > high)),
        "share_below": float(np.mean(pool < low)),
        "base_saturates_reward_bound": bool(math.isclose(high, 1.0, abs_tol=1e-12)),
        "ceiling_note": rp.SUPPORT_NOTE,
        "n": int(pool.size),
    }


def degeneracy_offsets(base: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Each declared degeneracy bound as a limit on the pool-minus-base contrast.

    The banded quantity is the contrast, so every bound has to be a statement
    about the contrast, and two of the four are declared against a *level* -- the
    base pool's own 0.05 quantile of distinct 5-mer fraction and twice its 0.95
    quantile of longest run. Those become contrast limits by subtracting the base
    pool's mean, which is exactly the same statement about the same quantity and
    uses only the interval that was actually banded. The other two are declared
    as distances from the base pool and are contrast limits already.

    ``side`` says which end of the contrast interval has to clear the limit:
    ``below`` for a bound that degeneracy breaches by falling, ``above`` for the
    one it breaches by rising.
    """

    offsets: dict[str, dict[str, Any]] = {}
    if "mean_ca_plddt" in base:
        offsets[rp.FAMILY_PLDDT] = {
            "limit": -rp.PLDDT_DEGENERACY_MARGIN, "side": "below",
            "reason": f"more than {rp.PLDDT_DEGENERACY_MARGIN:g} pLDDT points below the "
                      "base pool",
        }
    if {"distinct_5mer_fraction_quantile", "distinct_5mer_fraction_mean"} <= set(base):
        offsets[rp.FAMILY_DISTINCT_5MER] = {
            "limit": (float(base["distinct_5mer_fraction_quantile"])
                      - float(base["distinct_5mer_fraction_mean"])),
            "side": "below",
            "reason": f"below the base pool's {rp.DISTINCT_5MER_BASE_QUANTILE:g} quantile "
                      "of distinct 5-mer fraction",
        }
    if {"longest_single_residue_run_quantile",
            "longest_single_residue_run_mean"} <= set(base):
        offsets[rp.FAMILY_LONGEST_RUN] = {
            "limit": (rp.LONGEST_RUN_DEGENERACY_MULTIPLE
                      * float(base["longest_single_residue_run_quantile"])
                      - float(base["longest_single_residue_run_mean"])),
            "side": "above",
            "reason": f"above {rp.LONGEST_RUN_DEGENERACY_MULTIPLE:g} times the base pool's "
                      f"{rp.LONGEST_RUN_BASE_QUANTILE:g} quantile of longest "
                      "single-residue run",
        }
    if "composition_entropy_bits" in base:
        offsets[rp.FAMILY_COMPOSITION_ENTROPY] = {
            "limit": -rp.COMPOSITION_ENTROPY_DEGENERACY_MARGIN, "side": "below",
            "reason": f"more than {rp.COMPOSITION_ENTROPY_DEGENERACY_MARGIN:g} bit below "
                      "the base pool",
        }
    return offsets


def degeneracy_breaches(summaries: Mapping[str, Mapping[str, Any]],
                        base: Mapping[str, Any]) -> dict[str, Any]:
    """Which declared degeneracy bounds a pool breaches, by the declaration's own rules.

    ``summaries`` maps a family name to that pool's banded column -- ``point``
    and ``interval``, both of the pool-minus-base contrast -- and ``base``
    carries the base pool's own means and the two declared quantiles, from which
    :func:`degeneracy_offsets` forms each bound's limit on that contrast. A
    breach needs the *whole* interval past the limit, which is the same rule
    every other reading in this programme uses: a point estimate past a limit
    with an interval reaching back over it has not shown the effect.

    A breach does not make a bias absent. It decides which of the two
    installed-bias readings applies, because a bias reached by leaving the
    protein-like region is this project's own stronger claim rather than a
    refutation of it.
    """

    offsets = degeneracy_offsets(base)
    breaches: dict[str, str] = {}
    measured: dict[str, dict[str, Any]] = {}
    for family, bound in offsets.items():
        entry = summaries.get(family)
        if entry is None or entry.get("interval") is None:
            continue
        low, high = float(entry["interval"][0]), float(entry["interval"][1])
        limit = float(bound["limit"])
        breached = high < limit if bound["side"] == "below" else low > limit
        measured[family] = {"interval": [low, high], "limit": limit,
                            "side": bound["side"], "breached": bool(breached)}
        if breached:
            breaches[family] = (
                f"the whole contrast interval [{low:+.4f}, {high:+.4f}] lies past "
                f"{limit:+.4f}: {bound['reason']}"
            )
    unmeasured = [family for family in rp.DEGENERACY_BOUNDS
                  if family not in measured]
    return {"breaches": breaches, "breached": bool(breaches), "measured": measured,
            "unmeasured_bounds": unmeasured, "bounds": dict(rp.DEGENERACY_BOUNDS)}


def stratified_gain(columns: Mapping[str, Mapping[str, Any]], *, pool: str,
                    strata: Sequence[str] = rp.LENGTH_STRATUM_NAMES) -> dict[str, Any]:
    """Whether a pool's structural gain survives reading it within length strata.

    Mean CA-pLDDT falls with length across the retained structural support, so a
    policy that shortens gains reward without anything having been installed. The
    pooled gain cannot tell the two apart; a gain that holds at matched length can.

    The rule is :data:`..rl_environment_provenance.STRATIFICATION_RULE`: the gain
    survives when at least one stratum's whole interval lies above zero, and the
    strata that carry it are named. A stratum neither pool populates, or that one
    of them populates below the resampling floor, is unresolved and says so --
    which is the case a policy that moved out of the base pool's length range
    produces, and in that case the verdict is withheld for want of a matched
    comparison rather than passed on the pooled gain.
    """

    survived: list[str] = []
    negative: list[str] = []
    unresolved: list[str] = []
    detail: dict[str, Any] = {}
    for stratum in strata:
        entry = columns.get(f"{pool}|length={stratum}")
        if entry is None or entry.get("status") != "resolved" or entry.get("interval") is None:
            unresolved.append(stratum)
            detail[stratum] = {"status": "unresolved",
                               "reason": None if entry is None else entry.get("reason"),
                               "units": None if entry is None else entry.get("units")}
            continue
        low, high = float(entry["interval"][0]), float(entry["interval"][1])
        detail[stratum] = {"status": "resolved", "point": entry.get("point"),
                           "interval": [low, high], "units": entry.get("units")}
        if low > 0.0:
            survived.append(stratum)
        elif high < 0.0:
            negative.append(stratum)
    out = {
        "pool": pool,
        "strata": list(strata),
        "survived_in": survived,
        "negative_in": negative,
        "unresolved_in": unresolved,
        "per_stratum": detail,
        "survives": bool(survived),
        "rule": rp.STRATIFICATION_RULE,
    }
    if survived:
        out["reason"] = (
            "the gain is bounded above zero at matched length in stratum(s) "
            + ", ".join(survived)
        )
    elif len(unresolved) == len(strata):
        out["reason"] = (
            "no length stratum admits the comparison: the pools do not overlap in "
            "length anywhere both of them clear the resampling floor, so the pooled "
            "gain is measured across a length distribution they do not share"
        )
    else:
        out["reason"] = (
            "no length stratum carries the gain: it does not survive being read at "
            "matched length, which is what a gain earned by shortening looks like"
        )
    return out


def reading(*, optimised: Mapping[str, Any], k: Mapping[str, Any],
            support: Mapping[str, Any], degeneracy: Mapping[str, Any],
            stratified: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One pool's reading, by the decision rules declared before anything was trained.

    The rules are in :data:`..rl_environment_provenance.DECISION_RULES` and are
    implemented here as written: a failed optimisation is read as one and nothing
    further; a matched selection at or below the declared ``k`` threshold is
    redistribution; an installed bias needs the gain, a ``k*`` above the
    threshold and mass outside the base model's own tail together, and is then
    read within or outside the protein-like region by the degeneracy bounds.
    """

    moved = optimised.get("optimised")
    if moved is None:
        # Not measured is not a null. A pool that was never generated, or whose
        # reward-gain column does not resolve, has not failed to optimise -- it has
        # not been asked. Reporting it as a failed optimisation would turn an
        # absent measurement into a finding.
        return {"reading": rp.READING_UNRESOLVED,
                "reason": "the reward gain is not measured, so whether this pool "
                          "optimised at all is unknown: " + str(optimised.get("reason"))}
    if moved is not True:
        return {"reading": rp.READING_NO_OPTIMISATION,
                "reason": "the reward gain is not bounded above zero: "
                          + str(optimised.get("reason"))}
    if stratified is not None and not stratified.get("survives"):
        # Before k* is consulted at all: a gain that does not hold at matched
        # length is not a gain this experiment can attribute to anything the
        # optimisation installed, so neither reading is taken.
        return {"reading": rp.READING_UNRESOLVED,
                "reason": "the reward gain does not survive length stratification, so "
                          "no installed-bias or redistribution reading is taken: "
                          + str(stratified.get("reason"))}
    if k.get("redistribution") is True:
        return {"reading": rp.READING_REDISTRIBUTION,
                "reason": f"selection over {k['k_star']} base samples under the identical "
                          f"reward matches this pool's mean reward, at or below the declared "
                          f"threshold of {rp.K_REDISTRIBUTION}"}
    if k.get("redistribution") is None:
        return {"reading": rp.READING_UNRESOLVED,
                "reason": "the best-of-k family leaves k* undetermined: "
                          + str(k.get("reason", "columns unresolved"))}
    outside = float(support["share_above"])
    null = float(support["null_share"])
    if not outside > null:
        return {"reading": rp.READING_UNRESOLVED,
                "reason": f"the gain was not reached by selection at the declared budget, but "
                          f"the pool places {outside:.4f} of its samples beyond the base pool's "
                          f"own tail against a null of {null:.4f}, so no mass outside the base "
                          "model's memory is established"}
    if degeneracy["breached"]:
        return {"reading": rp.READING_BIAS_OFF_MANIFOLD,
                "reason": "a bias was installed and a declared degeneracy bound is breached: "
                          + "; ".join(degeneracy["breaches"].values())}
    return {"reading": rp.READING_BIAS_ON_MANIFOLD,
            "reason": f"the reward gain is bounded above zero, k* exceeds the declared "
                      f"threshold of {rp.K_REDISTRIBUTION}, the pool places {outside:.4f} of its "
                      f"samples beyond the base pool's own tail against a null of {null:.4f}, and "
                      "no declared degeneracy bound is breached"}
