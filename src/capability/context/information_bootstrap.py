"""Intervals for the cohort information estimand, resampled at the unit of dependence.

The estimand is one number per scored cohort and arm,

    I = H_baseline - H_model      (nats per scored token)

with the two terms estimated on **exactly the same token multiset**:

    H_model    = (sum_r clean_nll_sum_r) / (sum_r token_count_r)
    H_baseline = -(1/N) sum_v c(v) log q(v),
                 q(v) = (r(v) + a) / (R + a*V)

where ``c(v)`` counts the cohort's scored target tokens, ``r(v)`` counts the
disjoint reference set's, ``R = sum_v r(v)``, ``V`` is the declared vocabulary
and ``a`` is the additive smoothing constant. Both terms are therefore
token-weighted averages over the *same* ``N`` tokens, and

    I = (1/N) sum_i log[ p_model(t_i) / q(t_i) ]

is a mean of per-token contrasts. Three facts follow, and each of them is a
property this module exists to preserve.

**The two terms are paired.** They are averages over one token multiset, so a
resampling scheme that draws the cohort twice -- once for the model term and once
for the baseline term -- estimates the variance of a difference of two
*independent* cohorts, which is far larger than the variance of the paired
difference the estimand actually is. One cohort draw feeds both terms, always.
``tests/test_information_bootstrap.py::test_sharing_the_cohort_draw_narrows_the_interval``
drives :func:`_statistics_from_weights` with the pairing deliberately broken and
asserts the width ordering, because this is the one property whose violation
still produces a plausible-looking interval.

**The baseline is a random quantity too.** ``q`` is fitted on a finite reference
set, so the reference is resampled -- independently of the cohort, since the two
corpora are disjoint by construction -- and ``q`` is refitted inside every draw
with the same smoothing constant. Holding it fixed understates the interval;
the same test file asserts that inequality rather than assuming it.

**Sequences are not the unit of independence, and neither are tokens.** Groups
are, so groups are what is resampled, and the floor that governs every other
percentile interval in this package (:data:`~src.capability.core.statistics.MINIMUM_BOOTSTRAP_UNITS`)
is applied to the **token-weighted Kish effective group count**

    n_eff = (sum_g w_g)^2 / sum_g w_g^2,   w_g = scored tokens in group g

rather than to the raw group count. A cohort of forty groups one of which
carries 95% of the tokens has an effective count near one; its raw count clears
the floor and its interval does not mean what it says. Below the floor the
interval is **refused**. There is no record-level fallback: when dependence
inside groups is maximal, record-level resampling returns a *narrower* interval,
so the fallback would answer a request for an honest bound with a
confidently wrong one.

The floor binds on the **cohort** only, because the cohort is what defines the
estimand. A reference set with few effective groups contributes a variance
component that is itself estimated from few atoms, and that gap is not closed
here: ``reference_n_effective_groups`` is published beside every interval so it
can be read, and a thin reference also announces itself through the bias
described at the end of this docstring.

Everything is computed from per-record sufficient statistics -- a scalar NLL sum,
a token count, a symbol count and a sparse target-token count vector -- through
weighted sums over per-group aggregates, so a draw costs one sparse
matrix-multivector product and no token sequence is ever materialised. Only the
cohort's own token support is carried, because the baseline enters the
arithmetic as

    sum_v c(v) log q(v) = sum_{v in support} c(v) log(r(v) + a) - N log(R + a*V)

and the reference mass outside that support reaches the estimate only through
the scalar ``R``.

**The blind spot this scheme cannot resample away.** A cohort token the
reference never saw has ``r(v) = 0`` in the full reference and in every
resample of it -- resampling removes mass, it cannot create it -- so its
contribution to the baseline is the smoothing constant alone and varies across
draws only through the normaliser. The share of cohort tokens in that position,
and the share whose reference count is five or fewer, are reported on every
result: they measure how much of the baseline is pseudo-count rather than
corpus, and that part carries no sampling uncertainty here even though it
carries plenty of estimation error.

**A point where the estimand is known to be zero.** Every criterion applied to
``I`` -- a floor on it, a sign test on its interval -- is a claim about the
estimator's behaviour near zero, and the panel supplies no arm that sits there.
:func:`unigram_null_control` builds one: an arm whose predictive distribution
*is* a smoothed unigram fitted on a reference from a disjoint corpus block, so
both terms of ``I`` estimate the same population from independent samples and
the true value is zero by construction. It is an ordinary ``ArmStatistics`` and
goes through the same :func:`bootstrap_arms` call as the arms beside it.
:func:`mixed_unigram_arm` extends that one point to a family: mixing the
independent unigram with the model in a known proportion gives an arm whose
true ``I`` is that proportion of the real arm's, so a criterion stated as a
floor on ``I`` can be watched at any distance from zero and not only at zero.

**The percentile interval can sit above the point estimate, and that is not a
bug.** ``H_baseline`` is ``-sum c(v) log q_b(v)`` and ``-log`` is convex, so a
resampled reference gives ``E[H_baseline_b] > H_baseline`` by Jensen. The gap
grows as the reference shrinks against its vocabulary -- the same axis the
smoothing bias lives on. Measured here on a deliberately small reference (2000
tokens over a 256-symbol vocabulary) it reaches +0.047 nats, enough to put the
whole 95% percentile interval for ``I`` above the point estimate at
``z0 = -2.0``. **The published interval is not corrected for it and the point
estimate is never shifted**: this package has no BCa implementation, and a
silently moved interval is indistinguishable from a measurement. What the result
carries instead is ``bootstrap_bias`` and ``median_bias_z0`` on every statistic,
so the condition is visible rather than inferable.

**The displacement is also measured, separately from everything else it is
confounded with.** Every statistic is evaluated a second time at
``(cohort draw d, full reference)``, and the mean gap between that and
``(cohort draw d, reference draw d)`` is ``reference_resampling_displacement``:
the part of ``bootstrap_bias`` that reference resampling alone is responsible
for, with the cohort draw differenced out. ``displacement_corrected_interval``
is the percentile interval with that constant removed, and it moves the
interval's *position* only -- the point estimate, the standard error and the
reference's share of the width are all untouched.
``cohort_only_bootstrap_se`` and ``reference_held_fixed_interval`` report what
the interval would have been had the reference not been resampled at all, which
is a different object and a narrower one, so the two cannot be confused.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse, stats

from ..core.statistics import MINIMUM_BOOTSTRAP_UNITS, MINIMUM_DRAWS_IN_TAIL

SCHEMA_VERSION = "r2_transfer_information_bootstrap_v2"

LN2 = math.log(2.0)

#: Largest ``g = (z * SE(I) / I)^2`` at which a ratio with ``I`` in the
#: denominator may be published.
#:
#: ``g`` is Fieller's quantity, ``z^2`` times the squared reciprocal of the
#: denominator's t-statistic. ``g < 1`` is the exact condition under which
#: Fieller's confidence set for a ratio is a bounded interval rather than the
#: complement of one or the whole line; ``g < 0.05`` is the stricter and
#: conventional working limit, at which the denominator sits about 8.8 standard
#: errors from zero and the ratio's distribution is close enough to well behaved
#: that a percentile interval over it means something. Between the two the set
#: is technically bounded and practically useless, so the working limit is what
#: gates publication -- a percentile interval over draws whose denominator
#: approaches zero is finite and meaningless, and finiteness is exactly what
#: makes it look publishable.
FIELLER_MAXIMUM_G = 0.05

DEFAULT_BOOTSTRAP_DRAWS = 2000

#: Statistics carried on every result. The first three are the reported
#: quantities; the last two are the terms they are built from, published because
#: a reader who sees only ``I`` cannot tell a large baseline from a small model
#: entropy, and ``relative_information`` divides by one of them.
STATISTIC_NAMES: tuple[str, ...] = (
    "information_nats_per_token",
    "relative_information",
    "information_bits_per_symbol",
    "model_entropy_nats_per_token",
    "baseline_entropy_nats_per_token",
)

ESTIMAND = (
    "I = H_baseline - H_model in nats per scored token, both terms token-weighted "
    "over the same scored-target multiset; H_baseline is the cross-entropy of "
    "those targets under an additively smoothed unigram fitted on a disjoint "
    "reference set"
)

_COMPARABILITY_NOTE = (
    "per-token quantities are tokenizer-dependent and are not comparable across "
    "arms with different tokenisations; of the statistics contrasted here only "
    "information_bits_per_symbol is on a shared axis"
)


# --------------------------------------------------------------------------- #
# Input contract
# --------------------------------------------------------------------------- #


def _vector(values: Any, name: str, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return array


@dataclass(frozen=True)
class SparseCounts:
    """Per-record target-token counts in CSR layout.

    Record ``i`` owns ``unique_token_ids[record_offsets[i]:record_offsets[i + 1]]``
    and the aligned slice of ``counts``. The layout is the persisted one: a
    record's scored targets are a handful of distinct ids out of a vocabulary
    that may be fifty thousand wide, and materialising the dense vector per
    record would cost more memory than the whole bootstrap.
    """

    unique_token_ids: np.ndarray
    counts: np.ndarray
    record_offsets: np.ndarray

    def __post_init__(self) -> None:
        ids = _vector(self.unique_token_ids, "unique_token_ids", np.int64)
        counts = _vector(self.counts, "counts", np.int64)
        offsets = _vector(self.record_offsets, "record_offsets", np.int64)
        object.__setattr__(self, "unique_token_ids", ids)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "record_offsets", offsets)
        if ids.size != counts.size:
            raise ValueError("unique_token_ids and counts must align")
        if offsets.size < 2 or offsets[0] != 0 or offsets[-1] != ids.size:
            raise ValueError(
                "record_offsets must start at zero and end at the number of entries"
            )
        if np.any(np.diff(offsets) < 0):
            raise ValueError("record_offsets must be non-decreasing")
        if ids.size and int(ids.min()) < 0:
            raise ValueError("token ids must be non-negative")
        if counts.size and int(counts.min()) < 1:
            raise ValueError("a persisted sparse count must be positive")
        record_index = np.repeat(np.arange(offsets.size - 1), np.diff(offsets))
        order = np.lexsort((ids, record_index))
        sorted_records, sorted_ids = record_index[order], ids[order]
        if sorted_ids.size > 1 and np.any(
            (sorted_records[1:] == sorted_records[:-1])
            & (sorted_ids[1:] == sorted_ids[:-1])
        ):
            raise ValueError("a record repeats a token id in unique_token_ids")

    @property
    def n_records(self) -> int:
        return int(self.record_offsets.size - 1)

    @property
    def record_totals(self) -> np.ndarray:
        """Total tokens per record, from the counts themselves."""

        cumulative = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(self.counts, dtype=np.int64)]
        )
        return np.diff(cumulative[self.record_offsets])

    @classmethod
    def from_records(
        cls,
        token_ids: Sequence[Sequence[int]],
        counts: Sequence[Sequence[int]],
    ) -> "SparseCounts":
        """Build the CSR layout from one id array and one count array per record."""

        if len(token_ids) != len(counts):
            raise ValueError("token_ids and counts must have one entry per record")
        lengths = [len(block) for block in token_ids]
        if any(length != len(block) for length, block in zip(lengths, counts)):
            raise ValueError("a record's ids and counts have different lengths")
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        flat_ids = (
            np.concatenate([np.asarray(b, dtype=np.int64) for b in token_ids])
            if token_ids
            else np.zeros(0, dtype=np.int64)
        )
        flat_counts = (
            np.concatenate([np.asarray(b, dtype=np.int64) for b in counts])
            if counts
            else np.zeros(0, dtype=np.int64)
        )
        return cls(
            unique_token_ids=flat_ids, counts=flat_counts, record_offsets=offsets
        )


@dataclass(frozen=True)
class CohortStatistics:
    """Per-record sufficient statistics for a scored cohort."""

    clean_nll_sum: np.ndarray
    token_count: np.ndarray
    n_symbols: np.ndarray
    targets: SparseCounts
    group_id: np.ndarray

    def __post_init__(self) -> None:
        nll = _vector(self.clean_nll_sum, "clean_nll_sum", np.float64)
        tokens = _vector(self.token_count, "token_count", np.int64)
        symbols = _vector(self.n_symbols, "n_symbols", np.int64)
        groups = _vector(self.group_id, "group_id", np.int64)
        object.__setattr__(self, "clean_nll_sum", nll)
        object.__setattr__(self, "token_count", tokens)
        object.__setattr__(self, "n_symbols", symbols)
        object.__setattr__(self, "group_id", groups)
        n = self.targets.n_records
        for name, array in (
            ("clean_nll_sum", nll),
            ("token_count", tokens),
            ("n_symbols", symbols),
            ("group_id", groups),
        ):
            if array.size != n:
                raise ValueError(
                    f"{name} carries {array.size} records against "
                    f"{n} in the sparse target counts"
                )
        if n < 1:
            raise ValueError("a scored cohort needs at least one record")
        if not np.isfinite(nll).all():
            raise ValueError("clean_nll_sum contains non-finite values")
        if np.any(nll < 0.0):
            raise ValueError("a negative log-likelihood sum cannot be negative")
        if np.any(tokens < 1):
            raise ValueError("every scored record must carry at least one token")
        if np.any(symbols < 0):
            raise ValueError("symbol counts must be non-negative")
        if int(symbols.sum()) < 1:
            raise ValueError(
                "the cohort carries no symbols, so a per-symbol rate is undefined"
            )
        # The sparse counts are a decomposition of the very tokens ``token_count``
        # counts. If they disagree the two terms of the estimand are being taken
        # over different multisets, which is the one error that leaves both of
        # them individually plausible.
        if not np.array_equal(self.targets.record_totals, tokens):
            raise ValueError(
                "the sparse target counts do not sum to token_count on every "
                "record; the baseline and the model term would then be averages "
                "over different token multisets"
            )


@dataclass(frozen=True)
class ReferenceStatistics:
    """Per-record sufficient statistics for the disjoint reference set."""

    token_count: np.ndarray
    targets: SparseCounts
    group_id: np.ndarray

    def __post_init__(self) -> None:
        tokens = _vector(self.token_count, "token_count", np.int64)
        groups = _vector(self.group_id, "group_id", np.int64)
        object.__setattr__(self, "token_count", tokens)
        object.__setattr__(self, "group_id", groups)
        n = self.targets.n_records
        if tokens.size != n or groups.size != n:
            raise ValueError("reference arrays do not align with the sparse counts")
        if n < 1:
            raise ValueError("a reference set needs at least one record")
        if np.any(tokens < 1):
            raise ValueError("every reference record must carry at least one token")
        if not np.array_equal(self.targets.record_totals, tokens):
            raise ValueError(
                "the reference sparse counts do not sum to token_count on every record"
            )


@dataclass(frozen=True)
class ArmStatistics:
    """One arm's cohort, its reference set and the two scalars they are read with.

    ``vocab_size`` and ``smoothing`` carry no defaults. They are part of the
    baseline estimator, not of the call: the smoothing constant contributes an
    upward bias that scales with vocabulary against reference size (see
    ``pathways.LAPLACE_SMOOTHING``), so an arm whose constant arrived by default
    is an arm whose baseline nobody declared.
    """

    name: str
    cohort: CohortStatistics
    reference: ReferenceStatistics
    vocab_size: int
    smoothing: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "vocab_size", int(self.vocab_size))
        object.__setattr__(self, "smoothing", float(self.smoothing))
        if not self.name:
            raise ValueError("an arm needs a name")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must admit at least two symbols")
        if not self.smoothing > 0.0:
            raise ValueError("additive smoothing must be positive")
        for label, counts in (
            ("cohort", self.cohort.targets),
            ("reference", self.reference.targets),
        ):
            ids = counts.unique_token_ids
            if ids.size and int(ids.max()) >= self.vocab_size:
                raise ValueError(
                    f"{self.name}: a {label} token id lies outside the declared "
                    f"vocabulary of {self.vocab_size}"
                )


# --------------------------------------------------------------------------- #
# A control arm whose true information is zero
# --------------------------------------------------------------------------- #


def _dense_reference_counts(
    reference: ReferenceStatistics, vocab_size: int, what: str
) -> np.ndarray:
    """A reference's token counts over the declared inventory, densely."""

    ids = reference.targets.unique_token_ids
    if ids.size and int(ids.max()) >= vocab_size:
        raise ValueError(
            f"a {what} reference token id lies outside the declared vocabulary of "
            f"{vocab_size}; the two references are not counts over one inventory"
        )
    return np.bincount(
        ids, weights=reference.targets.counts.astype(np.float64), minlength=vocab_size
    )


def mixed_unigram_arm(
    arm: ArmStatistics,
    control_reference: ReferenceStatistics,
    *,
    name: str,
    model_weight: float = 0.0,
) -> ArmStatistics:
    """``arm``'s cohort scored by a mixture of an independent unigram and the model.

    The model term of ``I`` is replaced, per record, by

        (1 - w) * nll_unigram + w * nll_model,

    where ``nll_unigram = -sum_v c_record(v) log q_control(v)`` is the
    cross-entropy of the very same scored targets under

        q_control(v) = (r_control(v) + a) / (R_control + a*V)

    fitted on a reference set the caller supplies, ``nll_model`` is the arm's
    persisted value, and ``a`` and ``V`` are the arm's own smoothing constant and
    declared vocabulary. Nothing else moves: the cohort, its grouping, its symbol
    counts, its target counts and the baseline reference are the arm's, so the
    result is an ordinary :class:`ArmStatistics` and goes through
    :func:`bootstrap_arms` beside the real arms of its block, under the same
    resample indices and through the same estimator.

    ``model_weight`` is a randomised predictor's mixing weight: it scores a ``w``
    fraction of the tokens with the model and the remainder with the independent
    unigram. Both terms of ``I`` are token-weighted averages over one token
    multiset and only the model term moves, so the decomposition

        I_mixed = (1 - w) * I_control + w * I_arm

    is exact -- at the point estimate and inside every bootstrap draw, not merely
    in expectation. The control's true information is zero by construction (see
    :func:`unigram_null_control`), so the mixed arm's true information is
    ``w * I_arm``: a family of arms whose true value is known up to the estimate
    of ``I_arm``, at any chosen distance from zero. That is what a criterion
    stated as a floor on ``I`` has to be calibrated against, and ``w = 0``
    recovers the null control itself.

    **The control's reference must not be the baseline's**, at any weight.
    Fitted on the same counts ``q_control`` *is* ``q_baseline``, ``I_control`` is
    zero identically rather than by measurement, and the family's known value is
    then an algebraic identity rather than a calibration point. Equality of the
    two count vectors is the exact condition, so it is what is checked; the check
    does not weaken as ``w`` grows, because a family calibrated at one weight and
    tautological at another calibrates nothing.
    """

    if not 0.0 <= float(model_weight) <= 1.0:
        raise ValueError(
            f"{name}: model_weight must lie in [0, 1]; got {model_weight}. Outside "
            "it the arm is not a mixture of two predictors and its true "
            "information is not a known multiple of the arm's"
        )
    control_counts = _dense_reference_counts(control_reference, arm.vocab_size, "control")
    baseline_counts = _dense_reference_counts(arm.reference, arm.vocab_size, "baseline")
    if np.array_equal(control_counts, baseline_counts):
        raise ValueError(
            f"{name}: the control's unigram is fitted on the same token counts as "
            "the baseline it would be measured against, so q_control is "
            "q_baseline and I is zero identically rather than by measurement. "
            "Fit the control on the reference of a disjoint corpus block"
        )
    total = float(control_reference.token_count.sum())
    log_q = np.log(control_counts + arm.smoothing) - math.log(
        total + arm.smoothing * arm.vocab_size
    )
    targets = arm.cohort.targets
    per_entry = -targets.counts.astype(np.float64) * log_q[targets.unique_token_ids]
    cumulative = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(per_entry, dtype=np.float64)]
    )
    unigram_nll = np.diff(cumulative[targets.record_offsets])
    weight = float(model_weight)
    return ArmStatistics(
        name=name,
        cohort=CohortStatistics(
            clean_nll_sum=(
                (1.0 - weight) * unigram_nll + weight * arm.cohort.clean_nll_sum
            ),
            token_count=arm.cohort.token_count,
            n_symbols=arm.cohort.n_symbols,
            targets=targets,
            group_id=arm.cohort.group_id,
        ),
        reference=arm.reference,
        vocab_size=arm.vocab_size,
        smoothing=arm.smoothing,
    )


def unigram_null_control(
    arm: ArmStatistics,
    control_reference: ReferenceStatistics,
    *,
    name: str,
) -> ArmStatistics:
    """``arm``'s cohort scored by a smoothed unigram fitted on an independent sample.

    The model term of ``I`` is replaced by the cross-entropy of the very same
    scored targets under

        q_control(v) = (r_control(v) + a) / (R_control + a*V),

    where ``r_control`` counts a reference set the caller supplies and ``a`` and
    ``V`` are the arm's own smoothing constant and declared vocabulary. Nothing
    else moves: the cohort, its grouping, its symbol counts and the baseline
    reference are the arm's. The returned object is an ordinary
    :class:`ArmStatistics` and goes through :func:`bootstrap_arms` beside the
    real arms of its block, under the same resample indices, so it is measured
    by the same estimator rather than by a parallel one.

    **Why the true ``I`` is zero.** Both terms of ``I = H_baseline - H_model``
    are then the cross-entropy of one token multiset under a smoothed unigram of
    the same population: the baseline's is fitted on this block's held-out
    reference, the model's on the caller's. Two estimators of the same
    distribution, fitted on independent samples of comparable size, differ only
    by their estimation error, so ``E[I] = 0`` under exchangeability of the two
    samples -- while the *measured* ``I`` fluctuates with realistic noise and
    carries the same vocabulary-dependent smoothing constant every real arm
    carries. That is a point at which an eligibility criterion can be watched
    behaving, or misbehaving, at a known zero. An untrained model is not such a
    point: near-uniform logits cost about ``log V`` nats and land the arm far
    below any floor, several nats from the boundary the criteria live at.

    **The control's reference must not be the baseline's.** Fitted on the same
    counts, ``q_control`` *is* ``q_baseline``, the point estimate of ``I`` is
    zero identically rather than by measurement, and what the interval then
    describes is the noise of refitting the baseline against a model held fixed
    at its full-sample value -- a quantity with no bearing on any criterion.
    That case is refused here rather than reported, because a tautological zero
    and a measured one are indistinguishable once written to an artefact.
    Equality of the two count vectors is the exact condition, so it is what is
    checked; a caller that also knows the two samples' provenance should say so
    in its own record, since equal counts are the symptom and shared provenance
    is the cause.

    The exchangeability the zero rests on is the caller's to justify. Two
    reference sets drawn from different regions of one corpus are independent
    but not necessarily identically distributed, and under such a shift a
    mismatched unigram costs more than the matched one, which biases the
    measured ``I`` *downwards*. The direction is known and the size is not, so a
    control reading below zero bounds the criteria conservatively and a control
    reading above zero cannot be explained away by it.

    It is the ``model_weight = 0`` member of :func:`mixed_unigram_arm`, which is
    where the arithmetic lives; the two are one construction and not two.
    """

    return mixed_unigram_arm(arm, control_reference, name=name, model_weight=0.0)


# --------------------------------------------------------------------------- #
# Preparation: per-group aggregates and the sparse count matrices
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Prepared:
    """Everything a draw needs, aggregated to groups once."""

    name: str
    vocab_size: int
    smoothing: float
    cohort_group_labels: np.ndarray
    cohort_group_tokens: np.ndarray
    cohort_group_nll: np.ndarray
    cohort_group_symbols: np.ndarray
    cohort_counts_t: sparse.csr_matrix  # (support, groups)
    reference_group_labels: np.ndarray
    reference_group_tokens: np.ndarray
    reference_counts_t: sparse.csr_matrix  # (support, groups)
    support: np.ndarray
    diagnostics: dict[str, Any]

    @property
    def n_cohort_groups(self) -> int:
        return int(self.cohort_group_labels.size)

    @property
    def n_reference_groups(self) -> int:
        return int(self.reference_group_labels.size)

    @property
    def chunk(self) -> int:
        """Draws per block, so that a block's dense support matrix stays small."""

        return max(1, min(256, int(4_000_000 // max(int(self.support.size), 1))))


def _group_index(group_id: np.ndarray, labels: np.ndarray, what: str) -> np.ndarray:
    index = np.searchsorted(labels, group_id)
    if index.size and (
        index.max() >= labels.size or not np.array_equal(labels[index], group_id)
    ):
        raise ValueError(f"a {what} record carries a group outside the declared labels")
    return index


def _count_matrix(
    counts: SparseCounts,
    group_index: np.ndarray,
    n_groups: int,
    support: np.ndarray,
) -> sparse.csr_matrix:
    """``(support, group)`` counts, restricted to the cohort's token support."""

    rows = np.repeat(group_index, np.diff(counts.record_offsets))
    ids = counts.unique_token_ids
    position = np.searchsorted(support, ids)
    keep = (position < support.size) & (support[np.minimum(position, support.size - 1)] == ids)
    matrix = sparse.coo_matrix(
        (
            counts.counts[keep].astype(np.float64),
            (position[keep], rows[keep]),
        ),
        shape=(support.size, n_groups),
    )
    return matrix.tocsr()


def _prepare_arm(
    arm: ArmStatistics,
    cohort_group_labels: np.ndarray | None = None,
    reference_group_labels: np.ndarray | None = None,
) -> _Prepared:
    """Aggregate an arm's records to groups, once, in a declared group order."""

    cohort_labels = (
        np.unique(arm.cohort.group_id)
        if cohort_group_labels is None
        else np.asarray(cohort_group_labels, dtype=np.int64)
    )
    reference_labels = (
        np.unique(arm.reference.group_id)
        if reference_group_labels is None
        else np.asarray(reference_group_labels, dtype=np.int64)
    )
    cohort_index = _group_index(arm.cohort.group_id, cohort_labels, "cohort")
    reference_index = _group_index(
        arm.reference.group_id, reference_labels, "reference"
    )
    n_cohort_groups = int(cohort_labels.size)
    n_reference_groups = int(reference_labels.size)

    tokens = arm.cohort.token_count.astype(np.float64)
    group_tokens = np.bincount(cohort_index, weights=tokens, minlength=n_cohort_groups)
    group_nll = np.bincount(
        cohort_index, weights=arm.cohort.clean_nll_sum, minlength=n_cohort_groups
    )
    group_symbols = np.bincount(
        cohort_index,
        weights=arm.cohort.n_symbols.astype(np.float64),
        minlength=n_cohort_groups,
    )
    reference_group_tokens = np.bincount(
        reference_index,
        weights=arm.reference.token_count.astype(np.float64),
        minlength=n_reference_groups,
    )
    if np.any(group_tokens <= 0.0):
        raise ValueError(f"{arm.name}: a declared cohort group holds no scored tokens")
    if np.any(reference_group_tokens <= 0.0):
        raise ValueError(f"{arm.name}: a declared reference group holds no tokens")

    support = np.unique(arm.cohort.targets.unique_token_ids)
    cohort_counts_t = _count_matrix(
        arm.cohort.targets, cohort_index, n_cohort_groups, support
    )
    reference_counts_t = _count_matrix(
        arm.reference.targets, reference_index, n_reference_groups, support
    )

    total_tokens = float(group_tokens.sum())
    cohort_support_counts = np.asarray(cohort_counts_t.sum(axis=1)).ravel()
    reference_support_counts = np.asarray(reference_counts_t.sum(axis=1)).ravel()
    records_per_group = np.bincount(cohort_index, minlength=n_cohort_groups)
    largest_records = np.sort(arm.cohort.token_count)[::-1][:10]

    diagnostics = {
        "n_records": int(arm.cohort.targets.n_records),
        "n_scored_tokens": int(total_tokens),
        "n_groups": n_cohort_groups,
        "n_effective_groups": kish_effective_units(group_tokens),
        "largest_group_token_share": float(group_tokens.max() / total_tokens),
        "n_singleton_groups": int((records_per_group == 1).sum()),
        "top10_record_token_share": float(largest_records.sum() / total_tokens),
        "symbols_per_token": float(group_symbols.sum() / total_tokens),
        "distinct_cohort_tokens": int(support.size),
        "vocab_size": int(arm.vocab_size),
        "smoothing": float(arm.smoothing),
        "n_reference_records": int(arm.reference.targets.n_records),
        "n_reference_tokens": int(reference_group_tokens.sum()),
        "n_reference_groups": n_reference_groups,
        "reference_n_effective_groups": kish_effective_units(reference_group_tokens),
        # The part of the baseline that resampling the reference cannot move.
        "cohort_token_share_unseen_in_reference": float(
            cohort_support_counts[reference_support_counts <= 0].sum() / total_tokens
        ),
        "cohort_token_share_reference_count_at_most_5": float(
            cohort_support_counts[reference_support_counts <= 5].sum() / total_tokens
        ),
    }
    return _Prepared(
        name=arm.name,
        vocab_size=int(arm.vocab_size),
        smoothing=float(arm.smoothing),
        cohort_group_labels=cohort_labels,
        cohort_group_tokens=group_tokens,
        cohort_group_nll=group_nll,
        cohort_group_symbols=group_symbols,
        cohort_counts_t=cohort_counts_t,
        reference_group_labels=reference_labels,
        reference_group_tokens=reference_group_tokens,
        reference_counts_t=reference_counts_t,
        support=support,
        diagnostics=diagnostics,
    )


def kish_effective_units(weights: np.ndarray) -> float:
    """Token-weighted Kish effective number of groups."""

    values = np.asarray(weights, dtype=np.float64)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    return float(total * total / float((values * values).sum()))


# --------------------------------------------------------------------------- #
# The draw
# --------------------------------------------------------------------------- #


def _statistics_from_weights(
    prepared: _Prepared,
    model_weights: np.ndarray,
    baseline_weights: np.ndarray,
    reference_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """Every statistic under explicit group multiplicities, one row per draw.

    Three weight matrices rather than one because the pairing this scheme
    depends on is exactly the statement ``model_weights is baseline_weights``.
    Production passes the same cohort draw to both; the test that guards the
    topology passes two independent draws and asserts the interval gets wider.
    ``reference_weights`` is drawn independently of both, because the reference
    corpus is disjoint from the cohort.
    """

    n_draws = int(model_weights.shape[0])
    for name, weights, expected in (
        ("model_weights", model_weights, prepared.n_cohort_groups),
        ("baseline_weights", baseline_weights, prepared.n_cohort_groups),
        ("reference_weights", reference_weights, prepared.n_reference_groups),
    ):
        if weights.ndim != 2 or weights.shape != (n_draws, expected):
            raise ValueError(
                f"{name} must be ({n_draws}, {expected}); got {weights.shape}"
            )
        if np.any(weights < 0):
            raise ValueError(f"{name} carries a negative multiplicity")

    out = {name: np.empty(n_draws, dtype=np.float64) for name in STATISTIC_NAMES}
    pseudo_mass = prepared.smoothing * prepared.vocab_size
    for begin in range(0, n_draws, prepared.chunk):
        stop = min(begin + prepared.chunk, n_draws)
        w_model = model_weights[begin:stop].astype(np.float64)
        w_baseline = baseline_weights[begin:stop].astype(np.float64)
        w_reference = reference_weights[begin:stop].astype(np.float64)

        model_tokens = w_model @ prepared.cohort_group_tokens
        baseline_tokens = w_baseline @ prepared.cohort_group_tokens
        reference_tokens = w_reference @ prepared.reference_group_tokens
        if np.any(model_tokens <= 0.0) or np.any(baseline_tokens <= 0.0):
            raise RuntimeError(f"{prepared.name}: a cohort draw scored no tokens")
        if np.any(reference_tokens <= 0.0):
            raise RuntimeError(f"{prepared.name}: a reference draw held no tokens")

        # (support, draws): the cohort's target counts and the reference's, both
        # restricted to the cohort's support. Reference mass outside it reaches
        # the baseline only through the scalar normaliser.
        cohort_counts = prepared.cohort_counts_t @ w_baseline.T
        reference_counts = prepared.reference_counts_t @ w_reference.T
        log_q = np.log(reference_counts + prepared.smoothing) - np.log(
            reference_tokens + pseudo_mass
        )
        baseline = -(cohort_counts * log_q).sum(axis=0) / baseline_tokens
        model = (w_model @ prepared.cohort_group_nll) / model_tokens
        information = baseline - model
        symbols_per_token = (w_model @ prepared.cohort_group_symbols) / model_tokens

        out["baseline_entropy_nats_per_token"][begin:stop] = baseline
        out["model_entropy_nats_per_token"][begin:stop] = model
        out["information_nats_per_token"][begin:stop] = information
        out["relative_information"][begin:stop] = information / baseline
        out["information_bits_per_symbol"][begin:stop] = information / (
            LN2 * symbols_per_token
        )
    for name, draws in out.items():
        if not np.isfinite(draws).all():
            raise RuntimeError(
                f"{prepared.name}: {name} is non-finite on at least one draw; the "
                "percentile interval over what remains would be conditioned on "
                "finiteness rather than the requested bootstrap distribution"
            )
    return out


def _group_multiplicities(
    rng: np.random.Generator, n_groups: int, n_draws: int
) -> np.ndarray:
    """Multiplicities of ``n_groups`` groups drawn with replacement, ``n_draws`` times.

    A with-replacement sample of ``n_groups`` groups out of ``n_groups`` has
    exactly multinomial multiplicities, and the multiplicity vector is what the
    weighted sums need, so it is drawn directly rather than reconstructed from a
    list of indices.
    """

    return rng.multinomial(
        n_groups, np.full(n_groups, 1.0 / n_groups), size=n_draws
    ).astype(np.int64)


# --------------------------------------------------------------------------- #
# Summaries, floors and refusals
# --------------------------------------------------------------------------- #


def _quantile_mc_se(sorted_draws: np.ndarray, probability: float) -> float:
    """Monte-Carlo standard error of one percentile endpoint.

    Read off the order statistics rather than a density estimate: the number of
    draws below the population quantile is ``Binomial(B, p)``, so the draws one
    binomial standard deviation either side of the nominal rank bracket one
    standard error of the endpoint, and half that gap is the estimate. It needs
    no smoothing parameter, and it is honest about a flat tail -- where the
    density is low, the bracketed draws are far apart and the reported error is
    large, which is exactly the case a density-based formula understates.
    """

    n_draws = int(sorted_draws.size)
    spread = math.sqrt(probability * (1.0 - probability) * n_draws)
    low = int(min(max(math.floor(probability * n_draws - spread), 0), n_draws - 1))
    high = int(min(max(math.ceil(probability * n_draws + spread), 0), n_draws - 1))
    return float((sorted_draws[high] - sorted_draws[low]) / 2.0)


def _summarise(draws: np.ndarray, point: float, confidence: float) -> dict[str, Any]:
    """Percentile interval and the diagnostics that say how to read it.

    The bias figures are reported and **not** applied. A bootstrap bias estimate
    is itself noisy at these draw counts, and this package has no BCa
    implementation to apply ``z0`` inside; a silently shifted interval would be
    indistinguishable from a measurement. A reader who finds the point estimate
    outside the interval is reading a percentile interval over a biased draw
    distribution and should look at ``median_bias_z0``; the module docstring
    records where that bias comes from.
    """

    tail = (1.0 - confidence) / 2.0
    ordered = np.sort(draws)
    low = float(np.percentile(draws, 100.0 * tail))
    high = float(np.percentile(draws, 100.0 * (1.0 - tail)))
    fraction_below = float((draws < point).mean())
    z0 = float(stats.norm.ppf(fraction_below)) if 0.0 < fraction_below < 1.0 else None
    return {
        "point": float(point),
        "interval": [low, high],
        "confidence": float(confidence),
        "bootstrap_se": float(np.std(draws, ddof=1)),
        "bootstrap_bias": float(draws.mean() - point),
        "median_bias_fraction_below_point": fraction_below,
        "median_bias_z0": z0,
        "interval_mc_se": [
            _quantile_mc_se(ordered, tail),
            _quantile_mc_se(ordered, 1.0 - tail),
        ],
        "fraction_of_draws_positive": float((draws > 0).mean()),
        "n_draws": int(draws.size),
    }


DISPLACEMENT_NOTE = (
    "reference_resampling_displacement is the mean gap between a draw taken at "
    "(cohort draw, reference draw) and the same draw taken at (cohort draw, full "
    "reference). It is the Jensen displacement L34 and L42 catalogue, isolated "
    "from the cohort resampling it is otherwise confounded with in "
    "bootstrap_bias. displacement_corrected_interval removes it as a constant "
    "shift, so the point estimate, the standard error and the reference's share "
    "of the width are unchanged and only the interval's position moves. "
    "reference_held_fixed_interval and cohort_only_bootstrap_se are the "
    "different, narrower object that dropping the reference resampling would "
    "give, reported so that a corrected interval and a shrunken one cannot be "
    "confused"
)


def _reference_displacement(
    draws: np.ndarray, reference_fixed: np.ndarray, confidence: float
) -> dict[str, Any]:
    """One statistic's Jensen displacement, and the interval with it removed.

    ``reference_fixed`` holds the same cohort draws against the reference sample
    actually observed, so differencing the two means removes the cohort
    resampling entirely and leaves the reference refit alone. That is what makes
    this a correction for the diagnosed mechanism rather than for
    ``bootstrap_bias``, which also carries the ratio-estimator bias of the cohort
    draw and would be corrected away with it.

    The correction is a constant shift, so it is applied to the draws and the
    percentile taken afterwards rather than to the endpoints: the two agree
    exactly for a shift, and applying it to the draws is what keeps the
    arithmetic honest if the correction ever stops being one.
    """

    if draws.shape != reference_fixed.shape:
        raise ValueError(
            "the displacement needs one reference-held-fixed draw per resampled "
            f"draw; got {draws.shape} against {reference_fixed.shape}"
        )
    tail = (1.0 - confidence) / 2.0
    displacement = float(draws.mean() - reference_fixed.mean())
    corrected = draws - displacement
    return {
        "reference_resampling_displacement": displacement,
        "displacement_corrected_interval": [
            float(np.percentile(corrected, 100.0 * tail)),
            float(np.percentile(corrected, 100.0 * (1.0 - tail))),
        ],
        "cohort_only_bootstrap_se": float(np.std(reference_fixed, ddof=1)),
        "reference_held_fixed_interval": [
            float(np.percentile(reference_fixed, 100.0 * tail)),
            float(np.percentile(reference_fixed, 100.0 * (1.0 - tail))),
        ],
    }


def effective_unit_floor(
    n_effective_groups: float,
    n_groups: int,
    *,
    minimum_units: int = MINIMUM_BOOTSTRAP_UNITS,
) -> dict[str, Any]:
    """Publishability of a group bootstrap whose groups carry unequal token mass.

    ``statistics.bootstrap_unit_floor`` applies the same floor to a *count* of
    units, which is the right question only when the units carry comparable
    weight. Here they carry scored tokens, and a cohort where one group holds
    most of the tokens has as many effective atoms as its Kish count says, not
    as many as its group count says. The floor is therefore applied to the
    effective count; the shared constant is unchanged and imported, because the
    coverage argument behind it is about atoms and does not care how the atoms
    were counted.

    Returned rather than raised, in the same style and for the same reason as
    the shared floor: "too few effective groups to bound this" is a finding
    about the cohort and belongs in the artefact.
    """

    if minimum_units < 2:
        raise ValueError("a percentile interval needs at least two units")
    degenerate = bool(n_effective_groups < minimum_units)
    return {
        "n_groups": int(n_groups),
        "n_effective_groups": float(n_effective_groups),
        "minimum_units": int(minimum_units),
        "degenerate": degenerate,
        "degenerate_reason": (
            (
                f"{n_effective_groups:.2f} token-weighted effective groups (Kish) "
                f"across {n_groups} groups is below the {minimum_units}-unit floor; "
                "a nominal 95% percentile interval over so few atoms realises well "
                "under 95% coverage. Resampling records instead would not repair "
                "this: when the dependence inside groups is what suppressed the "
                "effective count, a record-level interval comes out narrower than "
                "the group-level one it replaces, so the refusal stands"
            )
            if degenerate
            else None
        ),
    }


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InformationResult:
    """One arm's interval, its draws, and the resample that produced them.

    ``record`` is JSON-safe and is what a stage writes. ``draws`` and
    ``cohort_multiplicities`` are in-memory only: they exist so that a
    downstream numerator can be recomputed *under the same group draw* and
    divided by this denominator inside the iteration, which is the only way a
    ratio of two resampled quantities gets an honest interval.
    """

    record: dict[str, Any]
    draws: dict[str, np.ndarray]
    cohort_multiplicities: np.ndarray | None
    cohort_group_labels: np.ndarray | None

    @property
    def refused(self) -> bool:
        return bool(self.record["refused"])

    @property
    def information(self) -> float:
        """Point estimate of ``I`` in nats/token; raises on a refused result."""

        if self.refused:
            raise ValueError(
                f"{self.record['arm']}: the interval was refused "
                f"({self.record['refusal_reason']})"
            )
        return float(
            self.record["statistics"]["information_nats_per_token"]["point"]
        )


@dataclass(frozen=True)
class ArmPanel:
    """Several arms bootstrapped under one set of resample indices."""

    arms: dict[str, InformationResult]
    contrasts: dict[str, dict[str, Any]]
    record: dict[str, Any]


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def _validate_draw_request(n_bootstrap: int, confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    tail = (1.0 - confidence) / 2.0
    in_tail = tail * n_bootstrap
    if in_tail < MINIMUM_DRAWS_IN_TAIL:
        raise ValueError(
            f"confidence={confidence} puts {in_tail:.2f} of {n_bootstrap} draws "
            f"below the lower percentile, under a floor of {MINIMUM_DRAWS_IN_TAIL:.0f}; "
            "the bound would be an order statistic of the extreme draws rather "
            f"than an estimate. Use at least {int(math.ceil(MINIMUM_DRAWS_IN_TAIL / tail))} "
            "draws, or a less extreme confidence level"
        )


def bootstrap_arms(
    arms: Sequence[ArmStatistics],
    *,
    seed: int,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_DRAWS,
    confidence: float = 0.95,
    contrasts: Sequence[tuple[str, str]] | None = None,
) -> ArmPanel:
    """Bootstrap several arms of one cohort together, under common resample indices.

    Every arm sees the **same** cohort group draw and the same reference group
    draw in every iteration, so a contrast between two arms is formed inside the
    iteration and gets its own percentile interval. That is the only reading of
    "these two arms differ" this module supports. Non-overlap of two
    independently bootstrapped per-arm intervals is not a test of difference --
    it is conservative when the arms are positively correlated, which arms
    sharing a cohort always are, and this repository has decided verdicts that
    way before.

    The arms must share their cohort and reference *group universes*, because
    that is what makes an index common. Their per-record statistics need not
    match: a different tokenisation gives a different token count for the same
    sequence, and that is precisely the difference being contrasted.

    ``contrasts`` defaults to every unordered pair in the given order.
    """

    _validate_draw_request(n_bootstrap, confidence)
    if not arms:
        raise ValueError("at least one arm is required")
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names):
        raise ValueError("arm names must be unique within a panel")

    cohort_labels = np.unique(arms[0].cohort.group_id)
    reference_labels = np.unique(arms[0].reference.group_id)
    for arm in arms[1:]:
        if not np.array_equal(np.unique(arm.cohort.group_id), cohort_labels):
            raise ValueError(
                f"{arm.name}: cohort groups differ from {arms[0].name}'s, so a "
                "common resample index would not address the same units"
            )
        if not np.array_equal(np.unique(arm.reference.group_id), reference_labels):
            raise ValueError(
                f"{arm.name}: reference groups differ from {arms[0].name}'s, so a "
                "common resample index would not address the same units"
            )

    prepared = [_prepare_arm(arm, cohort_labels, reference_labels) for arm in arms]
    rng = np.random.default_rng(seed)
    cohort_weights = _group_multiplicities(rng, int(cohort_labels.size), n_bootstrap)
    reference_weights = _group_multiplicities(
        rng, int(reference_labels.size), n_bootstrap
    )
    identity_cohort = np.ones((1, int(cohort_labels.size)), dtype=np.int64)
    identity_reference = np.ones((1, int(reference_labels.size)), dtype=np.int64)
    # A read-only view rather than a materialised block: it is only ever sliced
    # and cast one chunk at a time inside _statistics_from_weights.
    held_fixed_reference = np.broadcast_to(
        identity_reference, (n_bootstrap, int(reference_labels.size))
    )

    results: dict[str, InformationResult] = {}
    for arm, prep in zip(arms, prepared):
        floor = effective_unit_floor(
            prep.diagnostics["n_effective_groups"], prep.diagnostics["n_groups"]
        )
        base_record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "arm": arm.name,
            "estimand": ESTIMAND,
            "seed": int(seed),
            "n_bootstrap": int(n_bootstrap),
            "confidence": float(confidence),
            "resampling_unit": "group",
            "cohort_draw_shared_between_terms": True,
            "reference_resampled": True,
            "displacement_note": DISPLACEMENT_NOTE,
            "unit_floor": floor,
            "diagnostics": dict(prep.diagnostics),
        }
        if floor["degenerate"]:
            results[arm.name] = InformationResult(
                record={
                    **base_record,
                    "refused": True,
                    "refusal_reason": floor["degenerate_reason"],
                    "statistics": None,
                },
                draws={},
                cohort_multiplicities=None,
                cohort_group_labels=None,
            )
            continue
        point = _statistics_from_weights(
            prep, identity_cohort, identity_cohort, identity_reference
        )
        draws = _statistics_from_weights(
            prep, cohort_weights, cohort_weights, reference_weights
        )
        # The same cohort draws against the reference sample actually observed.
        # Differencing it out of ``draws`` is what isolates the Jensen
        # displacement from the cohort resampling.
        reference_fixed = _statistics_from_weights(
            prep, cohort_weights, cohort_weights, held_fixed_reference
        )
        results[arm.name] = InformationResult(
            record={
                **base_record,
                "refused": False,
                "refusal_reason": None,
                "statistics": {
                    name: {
                        **_summarise(draws[name], float(point[name][0]), confidence),
                        **_reference_displacement(
                            draws[name], reference_fixed[name], confidence
                        ),
                    }
                    for name in STATISTIC_NAMES
                },
            },
            draws=draws,
            cohort_multiplicities=cohort_weights,
            cohort_group_labels=cohort_labels,
        )

    pairs = (
        [
            (names[i], names[j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        ]
        if contrasts is None
        else [(str(left), str(right)) for left, right in contrasts]
    )
    contrast_records: dict[str, dict[str, Any]] = {}
    for left, right in pairs:
        for name in (left, right):
            if name not in results:
                raise ValueError(f"contrast names an arm outside this panel: {name!r}")
        contrast_records[f"{left}_minus_{right}"] = _contrast_record(
            results[left], results[right], confidence
        )

    record = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "n_bootstrap": int(n_bootstrap),
        "confidence": float(confidence),
        "common_resample_indices": True,
        "arms": {name: result.record for name, result in results.items()},
        "contrasts": contrast_records,
    }
    return ArmPanel(arms=results, contrasts=contrast_records, record=record)


def bootstrap_information(
    arm: ArmStatistics,
    *,
    seed: int,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_DRAWS,
    confidence: float = 0.95,
) -> InformationResult:
    """One arm's interval for ``I``, its ratio to the baseline and its bits/symbol.

    A single-arm case of :func:`bootstrap_arms`, and identical to it draw for
    draw at the same seed, so an arm measured alone and the same arm measured
    inside a panel report the same interval.
    """

    panel = bootstrap_arms(
        [arm], seed=seed, n_bootstrap=n_bootstrap, confidence=confidence, contrasts=()
    )
    return panel.arms[arm.name]


def _contrast_record(
    left: InformationResult, right: InformationResult, confidence: float
) -> dict[str, Any]:
    """Within-iteration difference of two arms measured under one resample."""

    refused = [
        result.record["arm"] for result in (left, right) if result.refused
    ]
    common = {
        "left": left.record["arm"],
        "right": right.record["arm"],
        "paired": True,
        "common_resample_indices": True,
        "confidence": float(confidence),
        "comparability_note": _COMPARABILITY_NOTE,
    }
    if refused:
        return {
            **common,
            "refused": True,
            "refusal_reason": (
                "the interval was refused for "
                + ", ".join(refused)
                + "; a contrast cannot be published where a term has none"
            ),
            "statistics": None,
        }
    return {
        **common,
        "refused": False,
        "refusal_reason": None,
        "statistics": {
            name: _summarise(
                left.draws[name] - right.draws[name],
                left.record["statistics"][name]["point"]
                - right.record["statistics"][name]["point"],
                confidence,
            )
            for name in STATISTIC_NAMES
        },
    }


def unpaired_contrast(
    left: InformationResult,
    right: InformationResult,
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """A difference between arms that do **not** share a cohort.

    Pairing is defined within a cohort and nowhere else. A text arm on an
    OpenWebText draw and a protein arm on a Swiss-Prot draw have no common
    resampling unit, so their difference is taken across two independent
    bootstrap distributions and is labelled ``paired: False`` rather than
    presented as the paired contrast :func:`bootstrap_arms` produces. The
    interval is right for genuinely independent cohorts and **too wide** for
    arms that in fact share records; those belong in one ``bootstrap_arms``
    call, which is why arms with different group universes are refused there
    rather than silently handled here.

    The two results must come from different seeds. Two panels run at one seed
    take their multiplicities from the same stream, which couples the sides in a
    way that is neither independence nor pairing, and the resulting interval
    describes nothing.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    common = {
        "left": left.record["arm"],
        "right": right.record["arm"],
        "paired": False,
        "common_resample_indices": False,
        "confidence": float(confidence),
        "comparability_note": _COMPARABILITY_NOTE,
        "independence_note": (
            "the two arms were bootstrapped separately, so this interval carries "
            "the sum of their variances; it is correct for independent cohorts "
            "and conservative for arms that share records"
        ),
    }
    refused = [result.record["arm"] for result in (left, right) if result.refused]
    if refused:
        return {
            **common,
            "refused": True,
            "refusal_reason": (
                "the interval was refused for "
                + ", ".join(refused)
                + "; a contrast cannot be published where a term has none"
            ),
            "statistics": None,
        }
    if left.record["seed"] == right.record["seed"]:
        raise ValueError(
            "an unpaired contrast needs two independent bootstraps, and these two "
            f"were both drawn at seed {left.record['seed']}; their multiplicities "
            "come from one stream and their difference is neither independent nor "
            "paired"
        )
    if left.record["n_bootstrap"] != right.record["n_bootstrap"]:
        raise ValueError(
            "an unpaired contrast pairs draw i of each side and needs the same "
            f"draw count on both; got {left.record['n_bootstrap']} against "
            f"{right.record['n_bootstrap']}"
        )
    return {
        **common,
        "refused": False,
        "refusal_reason": None,
        "statistics": {
            name: _summarise(
                left.draws[name] - right.draws[name],
                left.record["statistics"][name]["point"]
                - right.record["statistics"][name]["point"],
                confidence,
            )
            for name in STATISTIC_NAMES
        },
    }


def ratio_interval(
    numerator_draws: Sequence[float],
    numerator_point: float,
    denominator: InformationResult,
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """A quantity normalised by ``I``, formed inside the iteration and gated.

    ``numerator_draws[b]`` must be the numerator recomputed under draw ``b`` of
    ``denominator`` -- ``denominator.cohort_multiplicities[b]`` over
    ``denominator.cohort_group_labels`` is that draw, and is returned for
    exactly this purpose. Aligning the two by index is what makes ``R_b =
    D_b / I_b`` a draw from the ratio's distribution; dividing a published
    numerator interval by a published denominator interval is not, and gives an
    interval that is wrong in a direction nothing in the artefact reveals.

    Publication is gated on Fieller's ``g = (z * SE(I) / I)^2``, not on how many
    draws happened to land at a non-positive denominator. That count is reported
    -- it is the symptom -- but a bootstrap can produce zero such draws while
    ``I`` is still two standard errors from zero, and the percentile interval it
    yields there is finite, narrow and meaningless. When the gate refuses, the
    unnormalised numerator and denominator are still returned with their own
    intervals, because they remain the measurements that were made.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    common: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "arm": denominator.record["arm"],
        "confidence": float(confidence),
        "fieller_maximum_g": float(FIELLER_MAXIMUM_G),
        "denominator_estimand": ESTIMAND,
    }
    if denominator.refused:
        return {
            **common,
            "published": False,
            "refusal_reason": (
                "the denominator interval was refused: "
                f"{denominator.record['refusal_reason']}"
            ),
            "fieller_g": None,
            "numerator": None,
            "denominator": None,
            "ratio": None,
            "n_draws_non_positive_denominator": None,
        }

    numerator = np.asarray(numerator_draws, dtype=np.float64)
    information = denominator.draws["information_nats_per_token"]
    if numerator.ndim != 1 or numerator.size != information.size:
        raise ValueError(
            f"numerator_draws must carry one value per bootstrap draw "
            f"({information.size}); got {numerator.shape}"
        )
    if not np.isfinite(numerator).all():
        raise ValueError("numerator_draws contains non-finite values")
    numerator_point = float(numerator_point)
    if not math.isfinite(numerator_point):
        raise ValueError("numerator_point must be finite")

    summary = denominator.record["statistics"]["information_nats_per_token"]
    point = float(summary["point"])
    standard_error = float(summary["bootstrap_se"])
    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    g = float("inf") if point == 0.0 else float((z * standard_error / point) ** 2)
    record = {
        **common,
        "fieller_g": g,
        "numerator": _summarise(numerator, numerator_point, confidence),
        "denominator": summary,
        "n_draws_non_positive_denominator": int((information <= 0.0).sum()),
    }
    if point <= 0.0 or not (g < FIELLER_MAXIMUM_G):
        return {
            **record,
            "published": False,
            "refusal_reason": "denominator not identified away from zero",
            "ratio": None,
        }
    ratios = numerator / information
    if not np.isfinite(ratios).all():
        return {
            **record,
            "published": False,
            "refusal_reason": "denominator not identified away from zero",
            "ratio": None,
        }
    return {
        **record,
        "published": True,
        "refusal_reason": None,
        "ratio": _summarise(ratios, numerator_point / point, confidence),
    }
