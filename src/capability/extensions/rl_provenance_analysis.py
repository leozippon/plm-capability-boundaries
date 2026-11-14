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
    """

    if n < 1 or k < 1:
        raise ValueError("a best-of-k weight needs a non-empty pool and a positive k")
    tail = np.arange(n, -1, -1, dtype=float) / float(n)
    powers = tail ** float(k)
    return powers[:-1] - powers[1:]


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
              ks: Sequence[int] = rp.K_GRID) -> np.ndarray:
    """``R@k`` for each ``k``: the expected evaluation value of the selected sample.

    ``selection`` is the score the selector ranks by and ``evaluation`` the score
    the selected sample is credited with. They are the same vector for an exactly
    computable reward and two independent folded realisations for a reward that
    contains the pLDDT term.
    """

    rows = _best_of_k_rows(np.asarray([selection], dtype=float),
                           np.asarray([evaluation], dtype=float), ks)
    return rows[0]


def _best_of_k_rows(selection: np.ndarray, evaluation: np.ndarray,
                    ks: Sequence[int]) -> np.ndarray:
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
    weights = np.stack([best_of_k_weights(pool, int(k)) for k in ks], axis=1)
    return ranked_evaluation @ weights


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
        out[start:stop] = _best_of_k_rows(chosen, scored, ks)
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


def reading(*, optimised: Mapping[str, Any], k: Mapping[str, Any],
            support: Mapping[str, Any], degeneracy: Mapping[str, Any]) -> dict[str, Any]:
    """One pool's reading, by the decision rules declared before anything was trained.

    The rules are in :data:`..rl_environment_provenance.DECISION_RULES` and are
    implemented here as written: a failed optimisation is read as one and nothing
    further; a matched selection at or below the declared ``k`` threshold is
    redistribution; an installed bias needs the gain, a ``k*`` above the
    threshold and mass outside the base model's own tail together, and is then
    read within or outside the protein-like region by the degeneracy bounds.
    """

    if optimised.get("optimised") is not True:
        return {"reading": rp.READING_NO_OPTIMISATION,
                "reason": "the reward gain is not bounded above zero: "
                          + str(optimised.get("reason"))}
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
