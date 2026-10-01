"""Interval estimation and group-disjoint resampling for transfer measurements.

Three primitives, each answering a question this programme has got wrong at least
once.

:func:`mean_interval`
    A Student-t interval over a **finite** sample. The finiteness check is not
    defensive noise: a single non-finite value silently turns a mean and its
    interval into ``nan``, and ``nan`` compares false against every gate, so a
    broken arm reads as a failing arm.

:func:`paired_group_bootstrap`
    Resamples **groups**, not rows, and scores both prediction vectors on the
    same resampled rows. Research plan rule 4 -- "seeds and intervals or nothing"
    -- is only worth anything if the resampling unit is the unit of independence.
    The audit's §0.05 records the cost of getting this wrong in the other
    direction: the path-patching matched-pair interval resampled an arm's
    induction *heads*, its entire population, while the probe records that were
    the real sampling unit contributed nothing. The point estimate survived; the
    interval did not mean what it said.

:func:`make_group_splits`
    Identity-group-disjoint cross-validation folds, validated after
    construction rather than trusted. A probe trained on one member of a protein
    family and tested on another measures family membership, not the concept.

Vendored into ``src/capability`` by EXP-R2-066 from ``src/revision/statistics.py``
and ``src/revision/nested_recoverability.py``, whose remaining contents served
the retired CLT / dictionary-qualification scope.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


#: How much of a requested bootstrap may be discarded for non-finiteness before
#: the percentile interval stops describing the distribution it claims to.
#:
#: A resample whose metric is not finite -- a Spearman correlation over a
#: resample with no variance, say -- is a fact about the data, not a glitch to
#: skip. Dropping it silently changes the estimand from "the bootstrap
#: distribution of this metric" to "that distribution conditioned on being
#: finite", which is narrower and biased towards well-conditioned resamples.
#: Below this fraction the two are too different to publish under one name, so
#: :func:`paired_group_bootstrap` refuses instead of returning the conditioned
#: interval under the requested draw count.
MINIMUM_FINITE_DRAW_FRACTION = 0.95


#: Smallest number of bootstrap draws that may fall in a requested percentile
#: tail before the endpoint is an estimate rather than an order statistic of
#: the extreme draws. Ten is the point at which the Monte-Carlo spread of the
#: bound across seeds drops below the smallest effect the PAA census publishes;
#: :func:`prediction_addressed.cluster_bootstrap` and
#: :func:`information_bootstrap._validate_draw_request` both honour it.
#:
#: Declared here rather than in ``prediction_addressed`` because that module
#: pulls in ``torch`` and the information bootstrap is arithmetic over count
#: vectors that must stay importable, and testable, without a GPU stack.
MINIMUM_DRAWS_IN_TAIL = 10.0


#: Smallest number of resampling units at which a percentile bootstrap interval
#: may be published. One declaration, imported by every module that resamples --
#: which is not the same thing as one *honoured* by every module that resamples,
#: and the difference was measured rather than assumed. Seven of the twelve
#: resamplers in this package reach it: :func:`paired_group_bootstrap` here,
#: ``induction_robustness.cluster_bootstrap_fraction``,
#: ``induction_robustness.contrast_ratio_bootstrap``,
#: ``homology.bootstrap_stratum``, ``path_patching.bootstrap_difference``,
#: ``prediction_addressed.cluster_bootstrap`` and
#: ``circuits._case_resampled_interval``. Five do not, and are named in
#: ``../d2/tests/test_transfer_audit_invariants.py`` so the list cannot grow in
#: silence: ``probes.sequence_bootstrap``, the three
#: ``lenses.*_cluster_bootstrap`` and ``pathways.pathway_cluster_bootstrap``.
#:
#: **The derivation this constant used to carry did not support it.** It read:
#: "eight is the point at which the probability of an all-identical resample
#: falls below the discarded tail mass". That probability is ``n**(1-n)``, which
#: is 1.6% at four units and already below the 2.5% each tail discards, so the
#: stated reasoning selects four, not eight. A future reader following it would
#: "correct" the floor downward. The constant is right and the reason was wrong,
#: which is the more dangerous of the two combinations.
#:
#: The property a floor has to defend is coverage, not width. Measured here on
#: 1200 normal samples per point, the realised coverage of a nominal 95%
#: percentile interval over the cluster mean is 0.50 at two units, 0.74 at
#: three, 0.82 at four, 0.86 at six, 0.89 at eight, 0.92 at sixteen and 0.94 at
#: four hundred. Below eight the interval delivers between half and six-sevenths
#: of what it advertises; eight is where the shortfall stops being of the same
#: order as the confidence level itself. That is the derivation.
#:
#: Note also that average width is *not* monotone at the bottom: two units give
#: a mean width of 1.17 against 1.98 at three units, on the same generating
#: distribution, because the two-unit resample distribution has three atoms and
#: the percentile rule trims the extreme ones. So a small stratum can read as
#: the precise one, which is how two ``consistent_with_memorisation`` verdicts in
#: the 2026-07-28 run came to be decided by non-overlap of a four-unit interval
#: against a four-hundred-unit one.
MINIMUM_BOOTSTRAP_UNITS = 8


def bootstrap_unit_floor(
    n_units: int, *, minimum_units: int = MINIMUM_BOOTSTRAP_UNITS
) -> dict[str, object]:
    """The publishability record for a percentile interval over ``n_units`` units.

    Returned rather than raised, because "too few units to bound this" is a
    finding about the stratum, arm or head population and belongs in the
    artefact. Callers merge the record in and null their interval fields when
    ``degenerate`` is true; what they must not do is publish an interval and a
    unit count side by side and leave the reader to notice.
    """

    if minimum_units < 2:
        raise ValueError("a percentile interval needs at least two units")
    if n_units < 0:
        raise ValueError("unit counts are non-negative")
    degenerate = n_units < minimum_units
    return {
        "n_units": int(n_units),
        "minimum_units": int(minimum_units),
        "degenerate": bool(degenerate),
        "degenerate_reason": (
            (
                f"{n_units} units is below the {minimum_units}-unit floor; a nominal "
                "95% percentile interval over so few atoms realises well under 95% "
                "coverage and can come out narrower than one over hundreds of units, "
                "so it must not be compared against another population's"
            )
            if degenerate
            else None
        ),
    }


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(
            f"{name} must be a one-dimensional vector with at least two values"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def mean_interval(values: Sequence[float], confidence: float = 0.95) -> dict:
    """Student-t interval for a finite sample mean."""

    sample = _finite_vector(values, "values")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    mean = float(sample.mean())
    standard_error = float(stats.sem(sample))
    # A zero standard error gives a zero-width interval, which reads as perfect
    # precision and is the one shape a reader will not question. It comes from
    # identical values, and identical values on a cohort of proteins of
    # different lengths mean a degenerate input -- one sequence repeated, a
    # metric that saturated -- not a measurement without uncertainty. The
    # interval is still returned, because refusing would turn a diagnostic into
    # an abort deep inside a run, but it is flagged where it is produced.
    degenerate = standard_error == 0.0
    if degenerate:
        interval = [mean, mean]
    else:
        radius = (
            float(stats.t.ppf((1.0 + confidence) / 2.0, sample.size - 1))
            * standard_error
        )
        interval = [mean - radius, mean + radius]
    return {
        "mean": mean,
        "standard_error": standard_error,
        "confidence": float(confidence),
        "interval": interval,
        "n": int(sample.size),
        "zero_width_from_zero_variance": bool(degenerate),
    }


def paired_group_bootstrap(
    y: Sequence,
    left_predictions: Sequence,
    right_predictions: Sequence,
    groups: Sequence,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    seed: int,
    n_bootstrap: int = 1000,
    derived_statistic: Callable[[float, float], float] | None = None,
) -> dict:
    """Paired group bootstrap for a metric difference, ratio and optional transform.

    Groups are sampled with replacement and all observations in a sampled group
    are retained. The same sampled rows score both prediction vectors, so the
    pairing that makes the difference meaningful is preserved draw by draw.
    ``derived_statistic`` receives the two scores from each draw, which keeps
    composite statistics such as chance-corrected skill inside the bootstrap.

    Refused below :data:`MINIMUM_BOOTSTRAP_UNITS` groups. The group is the
    resampling unit here, so the floor that governs every other percentile
    interval in this package governs this one; it used to guard ``n < 2``, which
    is the guard the floor exists to replace.
    """

    truth = np.asarray(y)
    left = np.asarray(left_predictions)
    right = np.asarray(right_predictions)
    group_ids = np.asarray(groups)
    if truth.ndim != 1 or left.shape != truth.shape or right.shape != truth.shape:
        raise ValueError(
            "truth and prediction vectors must have identical one-dimensional shape"
        )
    if group_ids.shape != truth.shape:
        raise ValueError("groups must align with truth")
    unique_groups = np.unique(group_ids)
    # The unit floor, not ``n < 2``. Two groups admit the whole range where a
    # percentile interval is anti-conservative -- and, worse, not monotone in
    # width: a two-group resample distribution has three atoms and the
    # percentile rule trims the extreme ones, so the smallest design reads as
    # the most precise. Raised rather than flagged, because every caller reads
    # ``difference_ci95`` straight into a report and a degenerate marker beside
    # it would be optional to notice.
    floor = bootstrap_unit_floor(int(unique_groups.size))
    if floor["degenerate"]:
        raise ValueError(f"paired group bootstrap refused: {floor['degenerate_reason']}")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")

    left_score = float(metric(truth, left))
    right_score = float(metric(truth, right))
    if not np.isfinite(left_score) or not np.isfinite(right_score):
        raise ValueError("the metric is non-finite on the full sample")
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    ratios: list[float] = []
    derived_draws: list[float] = []
    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(
            unique_groups, size=unique_groups.size, replace=True
        )
        indices = np.concatenate(
            [np.flatnonzero(group_ids == group) for group in sampled_groups]
        )
        left_boot = float(metric(truth[indices], left[indices]))
        right_boot = float(metric(truth[indices], right[indices]))
        if not np.isfinite(left_boot) or not np.isfinite(right_boot):
            continue
        differences.append(left_boot - right_boot)
        if abs(right_boot) > 1e-12:
            ratio = left_boot / right_boot
            if np.isfinite(ratio):
                ratios.append(ratio)
        if derived_statistic is not None:
            derived = float(derived_statistic(left_boot, right_boot))
            if np.isfinite(derived):
                derived_draws.append(derived)
    if not differences:
        raise RuntimeError("no finite paired bootstrap draws were produced")
    minimum_draws = int(np.ceil(MINIMUM_FINITE_DRAW_FRACTION * n_bootstrap))
    if len(differences) < minimum_draws:
        raise RuntimeError(
            f"only {len(differences)} of {n_bootstrap} paired bootstrap draws were "
            f"finite, below the {MINIMUM_FINITE_DRAW_FRACTION:.0%} floor; the "
            "percentile interval over what survives is conditioned on finiteness "
            "and is not the requested bootstrap distribution"
        )
    point_ratio = left_score / right_score if abs(right_score) > 1e-12 else None
    if point_ratio is not None and not np.isfinite(point_ratio):
        point_ratio = None
    result = {
        "left_score": left_score,
        "right_score": right_score,
        "difference": left_score - right_score,
        "difference_ci95": [
            float(np.percentile(differences, 2.5)),
            float(np.percentile(differences, 97.5)),
        ],
        "ratio": point_ratio,
        "ratio_ci95": (
            [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))]
            if len(ratios) >= minimum_draws
            else None
        ),
        "n_bootstrap": int(n_bootstrap),
        # The requested count is not the count the intervals were taken over.
        # `difference_ci95` spans `n_finite_draws` values and `ratio_ci95` spans
        # `n_ratio_draws`, which is smaller again whenever a resampled right-hand
        # score lands on zero. Publishing only `n_bootstrap` let a percentile over
        # a handful of surviving draws be read as a 1000-draw interval.
        "n_bootstrap_requested": int(n_bootstrap),
        "n_finite_draws": len(differences),
        "n_non_finite_draws": int(n_bootstrap) - len(differences),
        "n_ratio_draws": len(ratios),
        "n_groups": int(unique_groups.size),
        "minimum_groups": int(MINIMUM_BOOTSTRAP_UNITS),
    }
    if derived_statistic is not None:
        derived_score = float(derived_statistic(left_score, right_score))
        if not np.isfinite(derived_score):
            raise RuntimeError("the requested derived statistic is undefined on the full sample")
        if len(derived_draws) < minimum_draws:
            raise RuntimeError(
                f"only {len(derived_draws)} of {n_bootstrap} derived-statistic bootstrap "
                f"draws were finite, below the {MINIMUM_FINITE_DRAW_FRACTION:.0%} floor"
            )
        result.update(
            {
                "derived_score": derived_score,
                "derived_ci95": [
                    float(np.percentile(derived_draws, 2.5)),
                    float(np.percentile(derived_draws, 97.5)),
                ],
                "n_derived_draws": len(derived_draws),
            }
        )
    return result


def make_group_splits(
    y: Sequence,
    groups: Sequence,
    *,
    n_splits: int,
    seed: int,
    task_type: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create and validate identity-group-disjoint folds.

    The post-construction checks are the point. scikit-learn's grouped splitters
    are correct, but a caller that hands them a grouping vector derived from the
    wrong column gets folds that look fine and leak. Every fold is therefore
    re-checked for group intersection and for class coverage, and every sample is
    required to be tested exactly once.
    """

    truth = np.asarray(y)
    group_ids = np.asarray(groups)
    if n_splits < 2 or np.unique(group_ids).size < n_splits:
        raise ValueError("n_splits requires at least that many unique groups")
    if task_type == "classification":
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        iterator = splitter.split(np.zeros((truth.size, 1)), truth, group_ids)
    elif task_type == "regression":
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros((truth.size, 1)), truth, group_ids)
    else:
        raise ValueError("task_type must be classification or regression")
    splits = [(np.asarray(train), np.asarray(test)) for train, test in iterator]
    test_counts = np.zeros(truth.size, dtype=int)
    for train, test in splits:
        if np.intersect1d(group_ids[train], group_ids[test]).size:
            raise RuntimeError("group leakage detected in a generated fold")
        if task_type == "classification":
            if np.unique(truth[train]).size < 2:
                raise ValueError("a training fold contains fewer than two classes")
            # The test fold was never checked. A single-class test fold makes
            # every threshold-free score on it undefined -- AUC has no negative
            # class to rank against -- and the undefined value then propagates
            # as a nan that compares false against every gate, which is the
            # exact failure mode :func:`mean_interval`'s finiteness check exists
            # to stop one step later. Checked on the same footing as the
            # training fold, because a fold that cannot be scored is a fold
            # that cannot be used.
            if np.unique(truth[test]).size < 2:
                raise ValueError(
                    "a test fold contains fewer than two classes; group-disjoint "
                    "folds over this grouping cannot be scored"
                )
        test_counts[test] += 1
    if not np.all(test_counts == 1):
        raise RuntimeError(
            "group fold construction did not test every sample exactly once"
        )
    return splits


#: Heads a layer must contribute before it can supply a within-layer rank
#: correlation. Three points make a Spearman that is +/-1 far too often to average.
MINIMUM_HEADS_PER_LAYER_FOR_WITHIN = 4


def depth_controlled_rank_correlation(
    census_score: np.ndarray, magnitude: np.ndarray, layer: np.ndarray
) -> dict[str, Any]:
    """The same rank correlation, with layer depth held fixed.

    The all-grid Spearman is taken over a flattened ``(layer, head)`` grid with no
    depth term, and the causal readout it correlates against is depth-biased by
    construction: the sender patch and every freeze hook act at the query row, and
    the metric is read there, so a head that writes at ``q`` late in the stack
    lands closer to the unembedding than one that writes early.  Measured on this
    panel, ``r(layer, |effect|)`` runs +0.36 to +0.77 on *every* arm without
    exception.

    That alone would be a nuisance rather than a confound if the census score were
    depth-neutral, and on ten of eleven arms it rises with depth (+0.00 to +0.50).
    **ProtGPT2 is the exception at −0.387**, and an all-grid correlation between a
    depth-rising readout and a depth-falling score must land near zero whatever the
    heads are doing -- which is its published −0.106.  Controlling for depth left
    the modality separation intact in sign in all four conditions and cut it from a
    gap of +0.195…+0.248 to +0.019…+0.069, below the draw dispersion of the arm
    that defines the boundary (EXP-R2-120).

    Both controls are reported because they fail differently.  ``partial`` removes
    a monotone depth trend from both variables and keeps the whole grid, so it is
    exposed to a non-monotone depth profile.  ``within_layer`` assumes nothing
    about the shape but discards every between-layer comparison and needs enough
    heads per layer, so it is reported with the layers it could use.
    """

    finite = np.isfinite(census_score) & np.isfinite(magnitude) & np.isfinite(layer)
    if finite.sum() < 3 or np.unique(layer[finite]).size < 2:
        return {
            "partial": None,
            "within_layer": None,
            "layers_used": 0,
            "r_layer_census_score": None,
            "r_layer_causal_magnitude": None,
            "withheld_reason": (
                "fewer than three finite heads, or a single layer: there is no "
                "depth to control for"
            ),
        }
    x, y, depth = census_score[finite], magnitude[finite], layer[finite]
    rank_x, rank_y, rank_d = (stats.rankdata(x), stats.rankdata(y), stats.rankdata(depth))
    residual_x = rank_x - np.polyval(np.polyfit(rank_d, rank_x, 1), rank_d)
    residual_y = rank_y - np.polyval(np.polyfit(rank_d, rank_y, 1), rank_d)
    # A variable that *is* depth leaves no residual, and `np.corrcoef` answers
    # two all-but-zero vectors with 1.0 -- a perfect correlation between two
    # quantities that carry no information, which is the worst possible way for
    # this to fail. Withheld instead, with the reason, because a score that
    # tracks depth exactly is precisely the case this function exists to detect.
    scale = max(float(np.std(rank_d)), 1.0)
    if np.std(residual_x) < 1e-9 * scale or np.std(residual_y) < 1e-9 * scale:
        partial: float | None = None
        partial_withheld: str | None = (
            "one variable is a function of layer alone, so nothing remains once "
            "depth is held fixed and a correlation of the residuals is undefined"
        )
    else:
        partial = float(np.corrcoef(residual_x, residual_y)[0, 1])
        partial_withheld = None

    transformed: list[float] = []
    weights: list[float] = []
    for value in np.unique(depth):
        block = depth == value
        if block.sum() < MINIMUM_HEADS_PER_LAYER_FOR_WITHIN:
            continue
        rho = stats.spearmanr(x[block], y[block]).statistic
        if not np.isfinite(rho):
            continue
        transformed.append(float(np.arctanh(np.clip(rho, -0.999, 0.999))))
        weights.append(float(block.sum() - 3))
    within = (
        float(np.tanh(np.average(transformed, weights=weights))) if transformed else None
    )
    record = {
        "partial": partial,
        "within_layer": within,
        "layers_used": len(transformed),
        "r_layer_census_score": float(stats.spearmanr(depth, x).statistic),
        "r_layer_causal_magnitude": float(stats.spearmanr(depth, y).statistic),
    }
    if partial_withheld is not None:
        record["partial_withheld_reason"] = partial_withheld
    return record
