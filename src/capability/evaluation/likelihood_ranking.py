"""Ranking diagnostics for E17's frozen pool: does model likelihood order candidate quality?

What this module answers, and what it refuses to claim
======================================================

E17 freezes a candidate pool before any evaluation and asks whether the generating
model's own likelihood ranks the pool by an independent quality endpoint. Two
endpoints are read, and only two here: ESMFold2's mean CA pLDDT (primary) and its
pTM (secondary). Both are *predicted structural confidence*. Neither is a
stability, a folding free energy or a function measurement, and no number in this
module may be read as any of those.

The likelihood is a mean negative log-likelihood per scored token, so a **lower**
value means the model finds the sequence **more** likely. Reported correlations
use high likelihood = negative mean NLL; a negative association means higher
likelihood accompanies worse confidence. Both endpoints are oriented so that larger is better. :func:`selection_diagnostics` and
:func:`tail_spearman` therefore rank by ascending likelihood, and an accidental
sign inversion would show as a sign flip rather than as a plausible-looking gain.

The association is reported three ways:

* a descriptive whole-pool Spearman between high likelihood (-mean NLL) and endpoint;
* a descriptive Spearman *inside* the upper-likelihood tail, at the fractions E17
  already reports (:data:`SELECTION_FRACTIONS`). This is a selection-conditioned
  diagnostic, not evidence of a causal effect: conditioning on a selected set
  changes the estimand, and the tail's support is small by construction;
* a regression-adjusted whole-pool association from a declared control ladder.

The control ladder
==================

Confounding here is structural: likelihood tracks length, composition and how
close a sequence sits to the training corpus, and every structural confidence
rises with length. Three nested designs are declared, in this order:

``length``
    the intercept and sequence length.
``length_composition``
    the above plus the twenty-residue composition with one fixed residue dropped
    (:data:`DROPPED_RESIDUE`), the composition entropy in bits, and the longest
    homopolymer run as a fraction of length.
``length_composition_novelty_family``
    the above plus the nearest-corpus identity from the novelty search and one
    categorical family indicator per row, the lexicographically smallest Pfam
    family or ``__none__`` when the row carries none.

The design is built once, from the declared columns, and never shrunk to fit a
particular pool: a rank-deficient or saturated support is reported unavailable
with its reason rather than silently swapped for a simpler control. A missing
covariate removes the row from that rung's support and the exclusion is counted;
it is never imputed to zero.

Residualization
===============

The adjusted association is Frisch-Waugh-Lovell on rank-transformed variables.
Each variable is replaced by its mid-rank, each is projected onto the design with
a least-squares solve (:func:`numpy.linalg.lstsq`, an SVD-based solver -- the
normal-equation inverse is never formed), and the adjusted association is the
Pearson correlation of the two residual vectors. The projection is repeated
inside every bootstrap replicate, so the controls are refit rather than held at
their whole-pool values. The design's rank, column count and residual degrees of
freedom are reported beside every estimate.

This is a **regression-adjusted association**, not exact matching and not a causal
effect. Fixed effects can exhaust a small support, and when they do the result is
unavailable, which is an honest reading rather than a failure.

Bootstrap
=========

Replicates resample E17's own declared independence units -- the attempt for an
unconditioned pool, the enzyme class key for a conditioned one -- with replacement,
retaining every row of a drawn unit and preserving a unit drawn more than once in
full (:func:`resample_indices_for_chosen`). Rows are never deduplicated by
identity. The share of replicates in which the declared design is full rank and
the adjusted statistic is finite is reported; below
:data:`~src.capability.core.statistics.MINIMUM_FINITE_DRAW_FRACTION` of the
requested draws the interval is withheld with that reason, and a pool with fewer
than :data:`~src.capability.core.statistics.MINIMUM_BOOTSTRAP_UNITS` units is
unavailable rather than pseudoreplicated. Adjusted and unadjusted estimates are
formed on the *same* resampled rows, so their difference is a paired quantity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from ..core.statistics import (
    MINIMUM_BOOTSTRAP_UNITS,
    MINIMUM_FINITE_DRAW_FRACTION,
    bootstrap_unit_floor,
)
from . import generated_phenotype as gp

SCHEMA_VERSION = "d1_likelihood_ranking_v1"

#: The independent quality endpoints, primary first. Both are ESMFold2 structural
#: confidences, oriented so a larger value is better; neither is stability or function.
PRIMARY_ENDPOINT = "mean_ca_plddt"
SECONDARY_ENDPOINT = "ptm"
ENDPOINTS: tuple[str, ...] = (PRIMARY_ENDPOINT, SECONDARY_ENDPOINT)

ENDPOINT_QUANTITY: dict[str, str] = {
    PRIMARY_ENDPOINT: "esmfold2_mean_ca_plddt",
    SECONDARY_ENDPOINT: "esmfold2_ptm",
}

#: The twenty-residue alphabet used for the composition control, taken from the
#: structure instrument's own declaration so the composition and the evaluator
#: cannot drift onto different alphabets.
AA20: str = gp.AA20

#: The one residue dropped from the composition block. The twenty frequencies sum
#: to one, so with an intercept the full block is collinear; dropping one *fixed*
#: residue, rather than the least frequent one, keeps the design the same across
#: pools and replicates. ``V`` is the last residue of :data:`AA20`.
DROPPED_RESIDUE: str = AA20[-1]

#: The label standing for "no Pfam family recognised". When observed it is the
#: reference level; otherwise the sorted first observed family is the reference.
NO_FAMILY = "__none__"

#: Selection fractions, shared with E17's yield curve so a reader compares the same grid.
SELECTION_FRACTIONS: tuple[float, ...] = gp.SELECTION_FRACTIONS

#: Smallest residual degrees of freedom at which the adjusted association is
#: published. A design whose rank leaves zero or one residual degree of freedom
#: is saturated: the residuals are identically zero or lie on a line, and the
#: correlation they produce is a property of the arithmetic rather than the pool.
MINIMUM_RESIDUAL_DF = 2


@dataclass(frozen=True)
class Rung:
    """One rung of the control ladder, and what it adds to the rung below."""

    name: str
    level: int
    uses_composition: bool
    uses_novelty: bool
    uses_family: bool
    statement: str


RUNG_LENGTH = Rung(
    name="length",
    level=1,
    uses_composition=False,
    uses_novelty=False,
    uses_family=False,
    statement="intercept plus sequence length, the first control every structural confidence tracks",
)
RUNG_COMPOSITION = Rung(
    name="length_composition",
    level=2,
    uses_composition=True,
    uses_novelty=False,
    uses_family=False,
    statement=(
        "the length control plus the twenty-residue composition with one fixed "
        "residue dropped, the composition entropy in bits and the longest "
        "homopolymer run as a fraction of length"
    ),
)
RUNG_FULL = Rung(
    name="length_composition_novelty_family",
    level=3,
    uses_composition=True,
    uses_novelty=True,
    uses_family=True,
    statement=(
        "the composition control plus the nearest-corpus identity from the novelty "
        "search and one categorical Pfam-family indicator per row, an observed level being "
        "the fixed reference level"
    ),
)

RUNGS: tuple[Rung, ...] = (RUNG_LENGTH, RUNG_COMPOSITION, RUNG_FULL)

#: The declared method, recorded verbatim in the artefact so the numbers cannot be
#: read apart from how they were produced.
METHOD: dict[str, Any] = {
    "likelihood_input_quantity": "mean_nll_per_token_nats (lower is more likely)",
    "reported_association_quantity": "-mean_nll_per_token_nats (higher is more likely)",
    "negative_association_means": "higher likelihood accompanies worse endpoint quality",
    "likelihood_orientation": (
        "a mean negative log-likelihood per scored token, so lower is more likely. "
        "Selection is ascending NLL; ALL reported correlations and adjusted associations "
        "use high likelihood = -mean NLL, so negative means higher likelihood, worse quality"
    ),
    "endpoint_orientation": (
        "mean CA pLDDT and pTM are oriented so larger is better. They are the structure "
        "predictor's confidence in its own coordinates, not stability and not function"
    ),
    "tail_rule": (
        "the upper-likelihood tail of fraction f is the k = max(1, ceil(f*n)) rows with "
        "the smallest negative log-likelihood, ties broken by the pool's frozen order. "
        "The Spearman inside that set is a selection-conditioned diagnostic, not a causal "
        "effect, and its support count and unit count are reported beside it. Its "
        "interval re-selects the tail inside each cluster-bootstrap replicate"
    ),
    "control_ladder": [rung.statement for rung in RUNGS],
    "residualization": (
        "Frisch-Waugh-Lovell on mid-ranks: each variable is replaced by its average rank, "
        "projected onto the declared design with numpy.linalg.lstsq (SVD), and the "
        "adjusted association is the Pearson correlation of the two residual vectors. No "
        "normal-equation inverse is formed. The projection is recomputed inside every "
        "bootstrap replicate"
    ),
    "adjusted_selection": (
        "rank by the least-squares residual of the raw negative log-likelihood on the "
        "declared design and evaluate the least-squares residual of the raw endpoint on "
        "the same design. The residual endpoint is a different quantity from the raw "
        "endpoint and the two selected means are never compared directly. The selected "
        "tail statistic is Spearman of -NLL residuals and endpoint residuals"
    ),
    "multiplicity": "uncorrected exploratory intervals, not global discoveries",
    "family_reference": (
        "__none__ when observed, otherwise the sorted first observed point-support level; "
        "fixed across bootstrap draws, without auto-tuning missing categories"
    ),
    "bootstrap": (
        "resample E17's declared independence units with replacement, retaining every row "
        "of a drawn unit and preserving repeated units in full; no identity deduplication. "
        "Controls are refit within each replicate, and the adjusted and unadjusted "
        "estimates are formed on the same resampled rows so their difference is paired. "
        "The valid-replicate fraction is reported and the interval withheld below the "
        "package's finite-draw floor"
    ),
    "outcome_regression_is_diagnostic_only": (
        "the control designs exist to describe what the likelihood association survives. "
        "They are not a trained selector, are not fitted for deployment, and must not be "
        "used to choose candidates"
    ),
}

#: What the artefact cannot establish, stated once.
LIMITATIONS: tuple[str, ...] = (
    "intervals are uncorrected exploratory intervals, not multiplicity-adjusted global "
    "discoveries across endpoints, rungs and selection fractions",
    "family coding uses one observed point-support reference fixed across bootstrap draws; "
    "a draw missing a category can be rank-deficient and is invalid, without auto-tuning",
    "every association here is regression-adjusted, not an exact match and not a causal "
    "effect. The endpoint is a structural confidence and the likelihood is a property of "
    "the generating model, so a residual association is not evidence that likelihood "
    "measures protein quality",
    "the tail association is conditioned on a selected set, so it describes that set and "
    "does not generalise to the pool. Small fractions have few independence units; "
    "their point estimates are flagged and their intervals withheld below the unit floor",
    "fixed effects can exhaust a small support. Where the declared design is "
    "rank-deficient or saturated the rung is unavailable with its reason, and no simpler "
    "control is substituted",
    "nearest-corpus identity comes from one reference corpus at one gathering configuration. "
    "A low identity means no close relative there, not that the sequence is unlike anything "
    "in any model's training data",
    "confidence is not stability and not function: pLDDT and pTM are the structure "
    "predictor's own uncertainty and carry no free-energy or activity units",
)


# --------------------------------------------------------------------- features


def aa20_composition(sequence: str) -> np.ndarray:
    """The twenty canonical residue frequencies of ``sequence``.

    Any character outside :data:`AA20` is refused rather than dropped: silently
    ignoring a non-canonical symbol would change the denominator and the design
    without appearing anywhere in the artefact.
    """

    if not isinstance(sequence, str) or not sequence:
        raise ValueError("a composition needs a non-empty sequence")
    counts = np.zeros(len(AA20), dtype=np.float64)
    index = {aa: position for position, aa in enumerate(AA20)}
    for residue in sequence:
        position = index.get(residue)
        if position is None:
            raise ValueError(f"residue {residue!r} is not one of the twenty canonical residues")
        counts[position] += 1.0
    return counts / counts.sum()


def composition_entropy_bits(fractions: Sequence[float] | np.ndarray) -> float:
    """Shannon entropy of a composition, in bits. Zero for a homopolymer."""

    values = np.asarray(fractions, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("an entropy needs a one-dimensional composition")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("a composition carries a negative or non-finite frequency")
    positive = values[values > 0.0]
    return float(-np.sum(positive * np.log2(positive)))


@dataclass(frozen=True)
class RowFeatures:
    """The declared covariates of one pool member, with the sequence-derived ones fixed."""

    length: float
    composition: tuple[float, ...]
    entropy_bits: float
    homopolymer_fraction: float
    nearest_corpus_identity: float | None
    family_label: str


def make_row_features(
    sequence: str,
    length: int | float,
    *,
    nearest_corpus_identity: float | None,
    families: Sequence[str],
) -> RowFeatures:
    """Build one row's declared covariates, refusing non-finite derived values."""

    length_value = float(length)
    if not math.isfinite(length_value) or length_value < 1.0:
        raise ValueError("a pool member needs a positive, finite length")
    composition = aa20_composition(sequence)
    entropy = composition_entropy_bits(composition)
    homopolymer = float(gp.longest_homopolymer_run(sequence)) / length_value
    identity: float | None
    if nearest_corpus_identity is None:
        identity = None
    else:
        identity = float(nearest_corpus_identity)
        if not math.isfinite(identity):
            raise ValueError("a nearest-corpus identity must be finite when it is present")
    labels = sorted({str(family) for family in families})
    for label in labels:
        if not label or label == NO_FAMILY:
            raise ValueError(f"a Pfam family label must be non-empty and not {NO_FAMILY!r}")
    return RowFeatures(
        length=length_value,
        composition=tuple(float(value) for value in composition),
        entropy_bits=float(entropy),
        homopolymer_fraction=float(homopolymer),
        nearest_corpus_identity=identity,
        family_label=labels[0] if labels else NO_FAMILY,
    )


def family_vocabulary(features: Sequence[RowFeatures]) -> tuple[str, ...]:
    """The declared family levels of a support, sorted so the design is deterministic."""

    if not features:
        raise ValueError("a family vocabulary needs at least one row")
    return tuple(sorted({feature.family_label for feature in features}))


def has_family_variation(vocabulary: Sequence[str]) -> bool:
    """Whether the family categorical has at least two levels present.

    A pool in which every row carries the same family label -- including every row
    carrying ``__none__`` -- gives the family control no contrast and therefore no
    content. The rung is then not the declared rung, so it is reported unavailable
    instead of being run with a silently smaller control set.
    """

    return len(set(vocabulary)) > 1


# ---------------------------------------------------------------------- design


def family_reference(vocabulary: Sequence[str]) -> str | None:
    """Observed point-support reference, held fixed when resampling the design."""

    levels = sorted(set(vocabulary))
    return NO_FAMILY if NO_FAMILY in levels else (levels[0] if levels else None)


def rung_column_names(rung: Rung, vocabulary: Sequence[str]) -> tuple[str, ...]:
    """The declared column names of a rung, in build order."""

    names = ["intercept", "length_scaled"]
    if rung.uses_composition:
        names += [f"composition_{aa}" for aa in AA20 if aa != DROPPED_RESIDUE]
        names += ["entropy_bits", "homopolymer_fraction"]
    if rung.uses_novelty:
        names.append("nearest_corpus_identity_scaled")
    if rung.uses_family:
        names += [f"family_{label}" for label in vocabulary if label != family_reference(vocabulary)]
    return tuple(names)


def rung_added_columns(rung: Rung, vocabulary: Sequence[str]) -> tuple[str, ...]:
    """The columns this rung adds over the rung below it."""

    previous = rung_column_names(RUNGS[rung.level - 2], vocabulary) if rung.level > 1 else ()
    return tuple(name for name in rung_column_names(rung, vocabulary) if name not in previous)


def build_design(
    features: Sequence[RowFeatures], rung: Rung, vocabulary: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...]]:
    """The declared design matrix for a rung, in the declared column order.

    Length and identity are scaled by fixed constants (per 100), and composition,
    entropy and the homopolymer fraction already lie on bounded scales, so the
    design is well scaled without a data-dependent standardization that would make
    the column space depend on the pool. A rung that needs a covariate the row does
    not carry raises here rather than receiving a zero.
    """

    names = rung_column_names(rung, vocabulary)
    rows: list[list[float]] = []
    for feature in features:
        row = [1.0, feature.length / 100.0]
        if rung.uses_composition:
            row += [
                feature.composition[position]
                for position, aa in enumerate(AA20)
                if aa != DROPPED_RESIDUE
            ]
            row += [feature.entropy_bits, feature.homopolymer_fraction]
        if rung.uses_novelty:
            if feature.nearest_corpus_identity is None:
                raise ValueError(
                    "the rung requires the nearest-corpus identity and the row does not "
                    "carry one; the row must have been excluded from this support"
                )
            row.append(feature.nearest_corpus_identity / 100.0)
        if rung.uses_family:
            row += [
                1.0 if feature.family_label == label else 0.0
                for label in vocabulary
                if label != family_reference(vocabulary)
            ]
        rows.append(row)
    design = np.asarray(rows, dtype=np.float64)
    if design.ndim != 2 or design.shape[1] != len(names):
        raise AssertionError("the design does not match its declared columns")
    if not np.isfinite(design).all():
        raise ValueError("the design carries a non-finite value")
    return design, names


def design_diagnostics(design: Sequence[Sequence[float]] | np.ndarray) -> dict[str, Any]:
    """Columns, rank and residual degrees of freedom of a design.

    The rank comes from an SVD through :func:`numpy.linalg.matrix_rank`, and the
    residual degrees of freedom are ``n - rank``. A caller reads these before any
    inversion, which is the point: the rank is a property of the support and the
    declared columns, and a deficient one is reported rather than repaired.
    """

    matrix = np.asarray(design, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("a design is a two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("a design carries a non-finite value")
    n_rows, n_columns = matrix.shape
    rank = int(np.linalg.matrix_rank(matrix))
    return {
        "n_rows": int(n_rows),
        "n_columns": int(n_columns),
        "rank": rank,
        "residual_df": int(n_rows - rank),
        "rank_deficiency": int(n_columns - rank),
    }


# --------------------------------------------------------------- residualization


def _as_1d(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} carries a non-finite value")
    return array


def _is_constant(values: np.ndarray) -> bool:
    if values.size == 0:
        return True
    scale = float(np.max(np.abs(values)))
    return bool(np.ptp(values) <= 1e-12 * max(1.0, scale))


def midranks(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Average ranks, ties sharing the mean of the positions they span."""

    return np.asarray(stats.rankdata(_as_1d(values, "values"), method="average"), dtype=np.float64)


def residualize(
    values: Sequence[float] | np.ndarray, design: Sequence[Sequence[float]] | np.ndarray
) -> np.ndarray:
    """Least-squares residual of ``values`` on ``design``.

    The solve is :func:`numpy.linalg.lstsq` on the design and the values; the
    cross-product matrix is never formed, so the conditioning of the solve is
    that of the design rather than its square.
    """

    target = _as_1d(values, "values")
    matrix = np.asarray(design, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != target.size:
        raise ValueError("the design and the values must align")
    coefficients, _residuals, _rank, _singular = np.linalg.lstsq(matrix, target, rcond=None)
    return target - matrix @ coefficients


def _pearson(left: np.ndarray, right: np.ndarray, left_name: str, right_name: str) -> tuple[float | None, str | None]:
    if left.size < 2 or right.size < 2:
        return None, "fewer_than_two_rows"
    if _is_constant(left):
        return None, f"{left_name}_constant"
    if _is_constant(right):
        return None, f"{right_name}_constant"
    value = float(np.corrcoef(left, right)[0, 1])
    if not math.isfinite(value):
        return None, "non_finite_correlation"
    return value, None


def _rank_association(left: np.ndarray, right: np.ndarray) -> tuple[float | None, str | None]:
    if left.size < 2:
        return None, "fewer_than_two_rows"
    if _is_constant(left):
        return None, "likelihood_constant"
    if _is_constant(right):
        return None, "endpoint_constant"
    ranked_left = midranks(left)
    ranked_right = midranks(right)
    return _pearson(ranked_left, ranked_right, "likelihood", "endpoint")


def whole_pool_spearman(
    likelihood: Sequence[float] | np.ndarray, endpoint: Sequence[float] | np.ndarray
) -> dict[str, Any]:
    """Descriptive Spearman between likelihood and endpoint over the whole support."""

    left = _as_1d(likelihood, "likelihood")
    right = _as_1d(endpoint, "endpoint")
    if left.size != right.size:
        raise ValueError("the likelihood and endpoint vectors must align")
    value, reason = _rank_association(-left, right)
    return {
        "status": "measured" if value is not None else "unavailable",
        "reason": reason,
        "spearman": value,
        "n_rows": int(left.size),
    }


def tail_spearman(
    likelihood: Sequence[float] | np.ndarray,
    endpoint: Sequence[float] | np.ndarray,
    units: Sequence[Any] | np.ndarray,
    *,
    fraction: float,
    minimum_units: int = MINIMUM_BOOTSTRAP_UNITS,
) -> dict[str, Any]:
    """Spearman inside the upper-likelihood tail at ``fraction``.

    The tail is the ``ceil(fraction * n)`` rows with the smallest negative
    log-likelihood, ties broken by the frozen pool order (``kind="stable"``). The
    selected-set size, the number of independence units it spans and the reason the
    correlation is undefined, if it is, are all reported. Correlation uses -NLL,
    so negative means higher likelihood accompanies worse quality. A two-unit tail
    and a constant endpoint are different findings.
    """

    left = _as_1d(likelihood, "likelihood")
    right = _as_1d(endpoint, "endpoint")
    labels = [str(unit) for unit in units]
    if not (left.size == right.size == len(labels)):
        raise ValueError("the likelihood, endpoint and units must align")
    selected = int(gp.selection_count(left.size, fraction))
    order = np.argsort(left, kind="stable")[:selected]
    tail_units = len({labels[index] for index in order.tolist()})
    floor = bootstrap_unit_floor(tail_units, minimum_units=minimum_units)
    value, reason = _rank_association(-left[order], right[order])
    if value is None:
        status = "unavailable"
    elif floor["degenerate"]:
        # The point estimate describes the selected set and is reported, but it spans
        # fewer independence units than the package publishes an interval over, so it
        # is flagged rather than read as a pool property.
        status = "measured_below_unit_floor"
    else:
        status = "measured"
    return {
        "fraction": float(fraction),
        "n_pool": int(left.size),
        "n_selected": selected,
        "n_units_in_tail": int(tail_units),
        "unit_floor": floor,
        "status": status,
        "reason": reason,
        "spearman": value,
        "quantile_rule": "ceil(fraction * n), ties broken by frozen pool order",
        "selection_conditioned": True,
    }


def partial_spearman(
    likelihood: Sequence[float] | np.ndarray,
    endpoint: Sequence[float] | np.ndarray,
    design: Sequence[Sequence[float]] | np.ndarray,
    *,
    units: Sequence[Any] | np.ndarray | None = None,
    minimum_units: int = MINIMUM_BOOTSTRAP_UNITS,
    minimum_residual_df: int = MINIMUM_RESIDUAL_DF,
) -> dict[str, Any]:
    """Regression-adjusted association from the declared control ladder.

    Returns the design diagnostics whether or not the association is available, so
    a caller never has to infer the support from a missing number.
    """

    left = _as_1d(likelihood, "likelihood")
    right = _as_1d(endpoint, "endpoint")
    matrix = np.asarray(design, dtype=np.float64)
    if left.size != right.size or matrix.ndim != 2 or matrix.shape[0] != left.size:
        raise ValueError("the likelihood, endpoint and design must align")
    diagnostics = design_diagnostics(matrix)
    record: dict[str, Any] = {
        "estimator": "fwl_midrank_residual_spearman",
        **diagnostics,
        "minimum_residual_df": int(minimum_residual_df),
    }
    if units is not None:
        labels = [str(unit) for unit in units]
        if len(labels) != left.size:
            raise ValueError("the independence units must align with the support")
        n_units = len({*labels})
        record["n_units"] = int(n_units)
        record["unit_floor"] = bootstrap_unit_floor(n_units, minimum_units=minimum_units)
    if diagnostics["rank"] < diagnostics["n_columns"]:
        record.update(status="unavailable", reason="rank_deficient", spearman=None)
        return record
    if diagnostics["residual_df"] < minimum_residual_df:
        record.update(status="unavailable", reason="saturated", spearman=None)
        return record
    residual_likelihood = residualize(midranks(-left), matrix)
    residual_endpoint = residualize(midranks(right), matrix)
    value, reason = _pearson(
        residual_likelihood, residual_endpoint, "likelihood_residual", "endpoint_residual"
    )
    record.update(status="measured" if value is not None else "unavailable", reason=reason, spearman=value)
    return record


# ---------------------------------------------------------------------- bootstrap


def index_by_unit(units: Sequence[Any] | np.ndarray) -> dict[str, np.ndarray]:
    """Map each independence unit to the row positions it spans."""

    labels = np.asarray([str(unit) for unit in units])
    if labels.ndim != 1:
        raise ValueError("the units must be one-dimensional")
    return {label: np.flatnonzero(labels == label) for label in np.unique(labels).tolist()}


def indices_for_unit_draw(
    index_lists: Mapping[str, np.ndarray], chosen: Sequence[Any] | np.ndarray
) -> np.ndarray:
    """Row indices of a resampled cluster draw, preserving repeated units in full.

    ``chosen`` names the units to include, one entry per draw, so a unit named
    twice contributes all of its rows twice. There is no deduplication by row
    identity: collapsing a duplicate would silently down-weight a repeated cluster
    and turn the cluster bootstrap back into a row bootstrap.
    """

    drawn = np.asarray([str(unit) for unit in chosen])
    if drawn.ndim != 1:
        raise ValueError("the draw must be one-dimensional")
    parts = []
    for label in drawn.tolist():
        rows = index_lists.get(label)
        if rows is None:
            raise ValueError(f"the draw names unit {label!r}, which the support does not carry")
        parts.append(rows)
    if not parts:
        return np.empty(0, dtype=int)
    return np.concatenate(parts)


def resample_indices_for_chosen(
    units: Sequence[Any] | np.ndarray, chosen: Sequence[Any] | np.ndarray
) -> np.ndarray:
    """Convenience wrapper: build the unit index once and draw from it.

    The bootstrap calls :func:`index_by_unit` once and :func:`indices_for_unit_draw`
    per replicate, so the multiplicity rule tested here is the multiplicity rule
    the reported interval uses.
    """

    return indices_for_unit_draw(index_by_unit(units), chosen)


def cluster_bootstrap_association(
    likelihood: Sequence[float] | np.ndarray,
    endpoint: Sequence[float] | np.ndarray,
    design: Sequence[Sequence[float]] | np.ndarray,
    units: Sequence[Any] | np.ndarray,
    *,
    seed: int,
    n_bootstrap: int,
    minimum_units: int = MINIMUM_BOOTSTRAP_UNITS,
) -> dict[str, Any]:
    """Cluster bootstrap of the adjusted association, controls refit per replicate.

    Each replicate resamples the declared units with replacement and refits the
    declared control design on the resampled rows before residualizing, so the
    adjustment is not held at its whole-pool value. A replicate is *invalid* when
    the design is rank-deficient there, when it is saturated, or when the adjusted
    statistic is undefined; the invalid reasons are counted and the valid fraction
    is reported. Adjusted and unadjusted estimates are formed on the same rows.
    """

    left = _as_1d(likelihood, "likelihood")
    right = _as_1d(endpoint, "endpoint")
    matrix = np.asarray(design, dtype=np.float64)
    labels = np.asarray([str(unit) for unit in units])
    if left.size != right.size or matrix.ndim != 2 or matrix.shape[0] != left.size:
        raise ValueError("the likelihood, endpoint and design must align")
    if labels.ndim != 1 or labels.size != left.size:
        raise ValueError("the independence units must align with the support")
    if n_bootstrap < 1:
        raise ValueError("a bootstrap needs at least one draw")
    unique_units = sorted(set(labels.tolist()))
    floor = bootstrap_unit_floor(len(unique_units), minimum_units=minimum_units)
    record: dict[str, Any] = {
        "n_units": int(len(unique_units)),
        "unit_floor": floor,
        "n_bootstrap_requested": int(n_bootstrap),
        "controls_refit_within_replicate": True,
        "replicates_preserve_unit_multiplicity": True,
    }
    if floor["degenerate"]:
        record.update(
            status="unavailable",
            reason="below_unit_floor",
            adjusted_ci95=None,
            unadjusted_ci95=None,
            paired_difference_ci95=None,
        )
        return record

    n_columns = matrix.shape[1]
    rng = np.random.default_rng(seed)
    unit_values = np.asarray(unique_units, dtype=object)
    index_lists = index_by_unit(labels)
    adjusted: list[float] = []
    unadjusted: list[float] = []
    paired: list[float] = []
    invalid = {"rank_deficient": 0, "saturated": 0, "undefined_correlation": 0}
    for _ in range(n_bootstrap):
        chosen = rng.choice(unit_values, size=unit_values.size, replace=True)
        indices = indices_for_unit_draw(index_lists, chosen)
        sub_design = matrix[indices]
        sub_rank = int(np.linalg.matrix_rank(sub_design))
        if sub_rank < n_columns:
            invalid["rank_deficient"] += 1
            continue
        if int(indices.size - sub_rank) < MINIMUM_RESIDUAL_DF:
            invalid["saturated"] += 1
            continue
        residual_likelihood = residualize(midranks(-left[indices]), sub_design)
        residual_endpoint = residualize(midranks(right[indices]), sub_design)
        adjusted_value, _ = _pearson(
            residual_likelihood, residual_endpoint, "likelihood_residual", "endpoint_residual"
        )
        unadjusted_value, _ = _rank_association(-left[indices], right[indices])
        if adjusted_value is None or unadjusted_value is None:
            invalid["undefined_correlation"] += 1
            continue
        adjusted.append(adjusted_value)
        unadjusted.append(unadjusted_value)
        paired.append(adjusted_value - unadjusted_value)
    n_valid = len(adjusted)
    valid_fraction = n_valid / n_bootstrap
    record.update(
        n_valid_draws=n_valid,
        n_invalid_draws=n_bootstrap - n_valid,
        invalid_reasons=invalid,
        valid_replicate_fraction=float(valid_fraction),
    )
    minimum_draws = int(math.ceil(MINIMUM_FINITE_DRAW_FRACTION * n_bootstrap))
    record["minimum_valid_draws"] = minimum_draws
    if n_valid < minimum_draws:
        record.update(
            status="unavailable",
            reason="valid_replicate_fraction_below_floor",
            adjusted_ci95=None,
            unadjusted_ci95=None,
            paired_difference_ci95=None,
        )
        return record
    record.update(
        status="measured",
        reason=None,
        adjusted_ci95=[
            float(np.percentile(adjusted, 2.5)),
            float(np.percentile(adjusted, 97.5)),
        ],
        unadjusted_ci95=[
            float(np.percentile(unadjusted, 2.5)),
            float(np.percentile(unadjusted, 97.5)),
        ],
        paired_difference_ci95=[
            float(np.percentile(paired, 2.5)),
            float(np.percentile(paired, 97.5)),
        ],
    )
    return record


# ------------------------------------------------------------- selection curves


def cluster_bootstrap_tail_spearman(
    likelihood: Sequence[float] | np.ndarray,
    endpoint: Sequence[float] | np.ndarray,
    units: Sequence[Any] | np.ndarray,
    *,
    fraction: float,
    seed: int,
    n_bootstrap: int,
    minimum_units: int = MINIMUM_BOOTSTRAP_UNITS,
) -> dict[str, Any]:
    """Cluster bootstrap of the tail Spearman, the tail re-selected in every replicate.

    Each replicate resamples the declared units with replacement and re-selects the
    upper-likelihood tail on the resampled pool, so the interval describes the
    selection rule this artefact reports rather than the tail of one fixed draw. A
    point tail below the independence-unit floor has no interval. In each draw,
    distinct original unit labels in the re-selected tail must also meet the floor;
    repeated rows retain multiplicity but cluster copies do not create new units.
    Below-floor or undefined draws are counted as invalid, never retried.
    """

    left = _as_1d(likelihood, "likelihood")
    right = _as_1d(endpoint, "endpoint")
    labels = np.asarray([str(unit) for unit in units])
    if left.size != right.size or labels.ndim != 1 or labels.size != left.size:
        raise ValueError("the likelihood, endpoint and units must align")
    if n_bootstrap < 1:
        raise ValueError("a bootstrap needs at least one draw")
    unique_units = sorted(set(labels.tolist()))
    floor = bootstrap_unit_floor(len(unique_units), minimum_units=minimum_units)
    record: dict[str, Any] = {
        "fraction": float(fraction),
        "n_units": int(len(unique_units)),
        "unit_floor": floor,
        "n_bootstrap_requested": int(n_bootstrap),
    }
    if floor["degenerate"]:
        record.update(status="unavailable", reason="below_unit_floor", spearman_ci95=None)
        return record
    point = tail_spearman(left, right, labels, fraction=fraction, minimum_units=minimum_units)
    record["n_units_in_point_tail"] = point["n_units_in_tail"]
    record["point_tail_unit_floor"] = point["unit_floor"]
    if point["unit_floor"]["degenerate"]:
        record.update(status="unavailable", reason="point_tail_below_unit_floor", spearman_ci95=None)
        return record
    index_lists = index_by_unit(labels)
    unit_values = np.asarray(unique_units, dtype=object)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    invalid: dict[str, int] = {}
    for _ in range(n_bootstrap):
        chosen = rng.choice(unit_values, size=unit_values.size, replace=True)
        indices = indices_for_unit_draw(index_lists, chosen)
        selected = int(gp.selection_count(indices.size, fraction))
        tail = indices[np.argsort(left[indices], kind="stable")[:selected]]
        # Count ORIGINAL labels, not copies of resampled clusters. Rows keep multiplicity.
        if len(set(labels[tail].tolist())) < minimum_units:
            invalid["tail_below_unit_floor"] = invalid.get("tail_below_unit_floor", 0) + 1
            continue
        value, reason = _rank_association(-left[tail], right[tail])
        if value is None:
            key = reason or "undefined_correlation"
            invalid[key] = invalid.get(key, 0) + 1
            continue
        draws.append(value)
    n_valid = len(draws)
    minimum_draws = int(math.ceil(MINIMUM_FINITE_DRAW_FRACTION * n_bootstrap))
    record.update(
        n_valid_draws=n_valid,
        n_invalid_draws=n_bootstrap - n_valid,
        invalid_reasons=invalid,
        valid_replicate_fraction=float(n_valid / n_bootstrap),
        minimum_valid_draws=minimum_draws,
    )
    if n_valid < minimum_draws:
        record.update(status="unavailable", reason="valid_replicate_fraction_below_floor", spearman_ci95=None)
        return record
    record.update(
        status="measured",
        reason=None,
        spearman_ci95=[
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
    )
    return record


def raw_selection_block(
    likelihood: Sequence[float] | np.ndarray,
    endpoint: Sequence[float] | np.ndarray,
    *,
    fraction: float,
) -> dict[str, Any]:
    """Raw selection at one budget: likelihood ranking, pool mean and oracle bound.

    Ties are broken by the frozen pool order (``kind="stable"``) on both the
    likelihood selection and the empirical oracle. The oracle is the realised best
    ``k`` of this pool and is not a usable method.
    """

    left = _as_1d(likelihood, "likelihood")
    right = _as_1d(endpoint, "endpoint")
    if left.size != right.size:
        raise ValueError("the likelihood and endpoint vectors must align")
    selected = int(gp.selection_count(left.size, fraction))
    chosen = np.argsort(left, kind="stable")[:selected]
    oracle = np.argsort(-right, kind="stable")[:selected]
    pool_mean = float(right.mean())
    return {
        "n_pool": int(left.size),
        "n_selected": selected,
        "selected_mean": float(right[chosen].mean()),
        "pool_expected_random": pool_mean,
        "oracle_bound": float(right[oracle].mean()),
        "gain_over_random": float(right[chosen].mean() - pool_mean),
        "oracle_gap": float(right[oracle].mean() - pool_mean),
        "oracle_is_not_a_method": (
            "the oracle ranks by the endpoint it would be used to predict, so it is "
            "unusable for selection; it is the realised best-k of this frozen pool"
        ),
    }


def selection_diagnostics(
    likelihood: Sequence[float] | np.ndarray,
    endpoint: Sequence[float] | np.ndarray,
    design: Sequence[Sequence[float]] | np.ndarray,
    *,
    fraction: float,
    rung_name: str,
) -> dict[str, Any]:
    """Raw and adjusted selection at one budget.

    The raw block ranks the frozen pool by the negative log-likelihood and reports
    the selected endpoint mean against random selection's expectation (the pool
    mean) and the empirical oracle bound (the mean over the best ``k`` rows by the
    endpoint itself). The adjusted block ranks by the least-squares residual of the
    raw likelihood on the declared design and evaluates the least-squares residual
    of the raw endpoint, which is a different unit from the raw endpoint: the two
    selected means are reported under different names and are never differenced.
    """

    left = _as_1d(likelihood, "likelihood")
    right = _as_1d(endpoint, "endpoint")
    matrix = np.asarray(design, dtype=np.float64)
    if left.size != right.size or matrix.ndim != 2 or matrix.shape[0] != left.size:
        raise ValueError("the likelihood, endpoint and design must align")
    diagnostics = design_diagnostics(matrix)
    selected = int(gp.selection_count(left.size, fraction))
    record: dict[str, Any] = {
        "fraction": float(fraction),
        "n_selected": selected,
        "rung": rung_name,
        "design": {
            key: diagnostics[key]
            for key in ("n_rows", "n_columns", "rank", "residual_df", "rank_deficiency")
        },
        "raw": raw_selection_block(left, right, fraction=fraction),
    }
    if diagnostics["rank"] < diagnostics["n_columns"]:
        record.update(status="unavailable", reason="rank_deficient", adjusted=None)
        return record
    if diagnostics["residual_df"] < MINIMUM_RESIDUAL_DF:
        record.update(status="unavailable", reason="saturated", adjusted=None)
        return record
    residual_likelihood = residualize(left, matrix)
    residual_endpoint = residualize(right, matrix)
    adjusted_order = np.argsort(residual_likelihood, kind="stable")[:selected]
    adjusted_oracle_order = np.argsort(-residual_endpoint, kind="stable")[:selected]
    tail_value, tail_reason = _rank_association(
        -residual_likelihood[adjusted_order], residual_endpoint[adjusted_order]
    )
    record.update(
        status="measured",
        reason=None,
        adjusted={
            "method": (
                "rank by the least-squares residual of the raw negative log-likelihood on "
                "the declared design; evaluate the least-squares residual of the raw "
                "endpoint on the same design"
            ),
            "selected_mean": float(residual_endpoint[adjusted_order].mean()),
            "pool_expected_random": float(residual_endpoint.mean()),
            "oracle_bound": float(residual_endpoint[adjusted_oracle_order].mean()),
            "gain_over_random": float(
                residual_endpoint[adjusted_order].mean() - residual_endpoint.mean()
            ),
            "selected_tail_spearman": tail_value,
            "selected_tail_spearman_reason": tail_reason,
        },
    )
    return record
