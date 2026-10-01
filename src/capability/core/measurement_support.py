"""Shared measurement support utilities required by capability measurements."""
from __future__ import annotations

import math
from collections.abc import Sequence, Mapping
from typing import Any
import numpy as np
from .arms import Cohort, Arm

UNIGRAM_ESTIMATORS = ("disjoint", "plugin")


LAPLACE_SMOOTHING = 1.0


SMOOTHING_SWEEP: tuple[float, ...] = (1.0, 0.5, 0.1, 0.01)


def subsample_cohort(cohort: Cohort, size: int, seed: int) -> Cohort:
    """A seeded sub-cohort, carrying its conditioning metadata and its parent's draw.

    A seed is meaningful here because the ``cohort_mean`` baseline is estimated
    on the evaluation cohort itself, so resampling the cohort resamples both the
    intervention and the thing it is scored on. The returned cohort has its own
    content digest, which is what makes two seeds distinguishable in the record.

    **The parent's sampling record travels with the child.** It used to be
    dropped, which meant ``Cohort.sampling`` on every subsampled cohort answered
    ``"unrecorded"`` -- so an artefact could not say whether the pool this was
    drawn from was a seeded draw over the corpus or the head of a file. Losing
    that is losing the one fact the sampling record exists to carry: a seeded
    subsample of a file-order prefix is still a file-order prefix, and it is the
    pool's mode, not the subsample's seed, that decides whether Appendix B rule 1
    was honoured.
    """

    if size < 1 or size > len(cohort):
        raise ValueError(f"cannot draw {size} of {len(cohort)} sequences from {cohort.name!r}")
    generator = np.random.default_rng(seed)
    indices = sorted(int(i) for i in generator.choice(len(cohort), size=size, replace=False))
    metadata: dict[str, Any] = {}
    labels = cohort.metadata.get("ec_labels")
    if labels is not None:
        if len(labels) != len(cohort.records):
            raise ValueError(f"cohort {cohort.name!r}: EC labels do not align with records")
        metadata["ec_labels"] = [labels[i] for i in indices]
    metadata["sampling"] = {
        **cohort.sampling,
        "subsample_of": cohort.name,
        "subsample_seed": int(seed),
        "subsample_size": int(size),
        "subsample_parent_size": len(cohort),
        "subsample_parent_digest": cohort.digest,
    }
    return Cohort(
        name=f"{cohort.name}_n{size}_seed{seed}",
        kind=cohort.kind,
        records=[cohort.records[i] for i in indices],
        min_symbols=cohort.min_symbols,
        max_symbols=cohort.max_symbols,
        metadata=metadata,
    )


def scored_target_entropy_nats(counts: np.ndarray) -> float:
    """Plug-in entropy of the empirical next-token marginal over scored targets.

    This is an in-sample estimate on exactly the positions the measurement
    scores. For large vocabularies it underestimates the true marginal entropy,
    which shrinks the context-information denominator and therefore inflates
    ``share_of_context_information``. It is retained as a diagnostic and as an
    explicit opt-in, never as the default; ``disjoint_unigram_cross_entropy_nats``
    is the estimator a headline number should use.
    """

    array = np.asarray(counts, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or np.any(array < 0):
        raise ValueError("token counts must be a non-negative vector over the vocabulary")
    total = array.sum()
    if total <= 0:
        raise ValueError("token counts are empty")
    probabilities = array[array > 0] / total
    return float(-(probabilities * np.log(probabilities)).sum())


def disjoint_unigram_cross_entropy_nats(
    reference_counts: np.ndarray,
    target_counts: np.ndarray,
    *,
    smoothing: float = LAPLACE_SMOOTHING,
) -> float:
    """Cross-entropy of a held-out unigram model on the scored targets.

    This is the context-free baseline the model has to beat, estimated the same
    way the model itself is scored: a predictor fitted on data it will not be
    evaluated on, then evaluated on the scored targets. Because the reference
    corpus is disjoint, the estimate carries none of the downward bias the
    in-cohort plug-in has.

    It does carry a smaller bias of its own, in the opposite direction and on
    the same axis: the additive smoothing raises the cross-entropy by an amount
    that scales with vocabulary size against reference size. Upwards bias is
    conservative for *one* arm's share and is not conservative for an ordering
    across arms, which is what this panel reports. :func:`smoothing_diagnostics`
    measures it and every caller that publishes a baseline is expected to carry
    that record beside the number; see :data:`LAPLACE_SMOOTHING`.
    """

    reference = np.asarray(reference_counts, dtype=np.float64)
    targets = np.asarray(target_counts, dtype=np.float64)
    if reference.ndim != 1 or reference.shape != targets.shape or reference.size < 2:
        raise ValueError("reference and target counts must be vectors over one vocabulary")
    if np.any(reference < 0) or np.any(targets < 0):
        raise ValueError("token counts must be non-negative")
    if not smoothing > 0:
        raise ValueError("additive smoothing must be positive")
    reference_total = reference.sum()
    target_total = targets.sum()
    if reference_total < 1 or target_total < 1:
        raise ValueError("reference and target count vectors must both be non-empty")
    probabilities = (reference + smoothing) / (reference_total + smoothing * reference.size)
    return float(-(targets * np.log(probabilities)).sum() / target_total)


def smoothing_diagnostics(
    reference_counts: np.ndarray,
    target_counts: np.ndarray,
    *,
    smoothing: float = LAPLACE_SMOOTHING,
    sweep: Sequence[float] = SMOOTHING_SWEEP,
) -> dict[str, Any]:
    """How much of a held-out baseline is the smoothing constant, not the corpus.

    Three numbers, because the smoothing bias has three distinguishable parts
    and a reader who is handed only the baseline can reconstruct none of them.

    ``smoothing_mass_fraction``
        ``s*V / (N + s*V)``: the share of the fitted distribution that is
        pseudo-count rather than observation. It is the whole story in one
        number and it is a function of vocabulary size, which is why it differs
        by three orders of magnitude between a 32-symbol arm and a 50257-piece
        one on the same reference corpus.
    ``normaliser_inflation_nats``
        ``log(1 + s*V/N)``: the exact amount the smoothing adds to the
        normaliser, and therefore an upper bound on the per-token inflation of
        every target token the reference actually saw. This is the term that
        tracks vocabulary size.
    ``target_mass_unseen_in_reference``
        the share of scored targets the reference never saw. Those tokens are
        the reason a smaller constant is not automatically better: their
        contribution is ``-log(s / (N + s*V))``, which *grows* as ``s`` shrinks.

    ``cross_entropy_by_smoothing`` is the same baseline recomputed across
    :data:`SMOOTHING_SWEEP`. The constant cannot be eliminated -- an unsmoothed
    held-out unigram is infinite the moment a target token is unseen -- so
    Appendix B rule 8's remedy for an unavoidable threshold applies: sweep it
    and show the ordering does not turn on it. Recomputation is a handful of
    vector operations over count vectors already in memory, so the sweep costs
    nothing measurable next to the forward passes that produced the counts.
    """

    reference = np.asarray(reference_counts, dtype=np.float64)
    targets = np.asarray(target_counts, dtype=np.float64)
    if reference.ndim != 1 or reference.shape != targets.shape or reference.size < 2:
        raise ValueError("reference and target counts must be vectors over one vocabulary")
    if not smoothing > 0:
        raise ValueError("additive smoothing must be positive")
    if not sweep or any(not value > 0 for value in sweep):
        raise ValueError("every swept smoothing constant must be positive")
    vocabulary = int(reference.size)
    reference_total = float(reference.sum())
    target_total = float(targets.sum())
    if reference_total < 1 or target_total < 1:
        raise ValueError("reference and target count vectors must both be non-empty")
    pseudo = smoothing * vocabulary
    by_smoothing = {
        f"{value:g}": disjoint_unigram_cross_entropy_nats(
            reference, targets, smoothing=float(value)
        )
        for value in sweep
    }
    scored = list(by_smoothing.values())
    return {
        "smoothing": float(smoothing),
        "vocabulary_size": vocabulary,
        "reference_tokens": int(reference_total),
        "target_tokens": int(target_total),
        "smoothing_mass_fraction": float(pseudo / (reference_total + pseudo)),
        "normaliser_inflation_nats": float(math.log1p(pseudo / reference_total)),
        "target_mass_unseen_in_reference": float(
            targets[reference <= 0].sum() / target_total
        ),
        "sweep": [float(value) for value in sweep],
        "cross_entropy_by_smoothing": by_smoothing,
        "cross_entropy_sweep_range_nats": float(max(scored) - min(scored)),
        "note": (
            "normaliser_inflation_nats scales with vocabulary size against "
            "reference size, so it biases the large-vocabulary arms' baselines "
            "upwards relative to the residue-level arms' and therefore biases "
            "their context-information denominators in the same direction. It is "
            "conservative within an arm and not conservative across the panel"
        ),
    }


def held_out_cohort(candidate: Cohort, scored: Cohort) -> tuple[Cohort, dict[str, int]]:
    """``candidate`` with every record whose content also occurs in ``scored`` removed.

    Swiss-Prot and the EC-labelled corpus both carry the same sequence under
    several accessions, so taking a later block of records in file order does
    not by itself produce a held-out corpus. Fitting the context-free baseline
    on content it will then be evaluated on is precisely the leak this estimator
    exists to avoid, so the duplicates are removed by content and the number
    removed is returned for the record rather than absorbed silently.

    **The candidate's sampling record travels with the result**, for the reason
    :func:`subsample_cohort` already documents: dropping it made
    ``Cohort.sampling`` answer ``"unrecorded"`` on every held-out reference this
    package builds, so an artefact could not say whether the block the
    context-free baseline was fitted on was a seeded draw over the corpus or the
    head of a file. That baseline is the denominator of every context-information
    figure, and Appendix B rule 1 applies to it as much as to the scored draw.
    """

    if candidate.kind != scored.kind:
        raise ValueError("a held-out corpus must have the same kind as the cohort it serves")
    excluded = set(scored.records)
    keep = [index for index, record in enumerate(candidate.records) if record not in excluded]
    if not keep:
        raise ValueError(
            f"reference cohort {candidate.name!r} is entirely contained in {scored.name!r}"
        )
    metadata: dict[str, Any] = {}
    labels = candidate.metadata.get("ec_labels")
    if labels is not None:
        if len(labels) != len(candidate.records):
            raise ValueError(f"cohort {candidate.name!r}: EC labels do not align with records")
        metadata["ec_labels"] = [labels[index] for index in keep]
    metadata["sampling"] = {
        **candidate.sampling,
        "held_out_against": scored.name,
        "held_out_against_digest": scored.digest,
        "dropped_sequences_shared_with_cohort": len(candidate.records) - len(keep),
    }
    cohort = Cohort(
        name=candidate.name,
        kind=candidate.kind,
        records=[candidate.records[index] for index in keep],
        min_symbols=candidate.min_symbols,
        max_symbols=candidate.max_symbols,
        metadata=metadata,
    )
    return cohort, {
        "requested_sequences": len(candidate.records),
        "retained_sequences": len(keep),
        "dropped_sequences_shared_with_cohort": len(candidate.records) - len(keep),
    }


def assert_disjoint(scored: Cohort, reference: Cohort) -> None:
    """Refuse a reference corpus that overlaps the cohort it will normalise."""

    overlap = set(scored.records) & set(reference.records)
    if overlap:
        raise ValueError(
            f"reference cohort {reference.name!r} shares {len(overlap)} sequences with "
            f"{scored.name!r}; a held-out baseline must be disjoint"
        )
    if scored.digest == reference.digest:
        raise ValueError(f"reference cohort {reference.name!r} is the scored cohort")


def unigram_baseline(
    arm: Arm,
    *,
    estimator: str,
    target_counts: np.ndarray,
    reference_counts: np.ndarray | None = None,
    reference: Mapping[str, Any] | None = None,
    override_nats: float | None = None,
    smoothing: float = LAPLACE_SMOOTHING,
) -> dict[str, Any]:
    """The context-free baseline for one arm, with the estimator declared.

    There is no fallback path. Asking for the held-out estimator without a
    held-out corpus is a configuration error and raises, because a silent
    downgrade to the plug-in would move the headline share by tens of percent on
    the large-vocabulary arms without changing anything a reader can see.
    """

    if estimator not in UNIGRAM_ESTIMATORS:
        raise ValueError(f"unknown unigram estimator {estimator!r}; known {UNIGRAM_ESTIMATORS}")
    plug_in = scored_target_entropy_nats(target_counts)
    record: dict[str, Any] = {
        "estimator": estimator,
        "cohort_plug_in_entropy_nats": plug_in,
        "smoothing": None,
        "reference": None,
    }
    if override_nats is not None:
        if not math.isfinite(override_nats) or override_nats <= 0:
            raise ValueError(f"{arm.name}: supplied unigram entropy must be finite and positive")
        # ``estimator`` describes how the returned number was produced, and an
        # externally supplied number was not produced by any estimator in this
        # module. The record used to answer ``"disjoint"`` beside
        # ``source: "external_override"``, so an artefact could claim a held-out
        # cross-entropy for a value that arrived on the command line and whose
        # provenance nothing here can check. The requested estimator is kept
        # under its own key, because which estimator the caller *asked* for is
        # also a fact worth having.
        return {
            **record,
            "estimator": "external_override",
            "requested_estimator": estimator,
            "nats": float(override_nats),
            "source": "external_override",
            "provenance_note": (
                "supplied by the caller; this module did not estimate it and cannot "
                "attest to the corpus, the window or the smoothing behind it"
            ),
        }
    if estimator == "plugin":
        return {**record, "nats": plug_in, "source": "cohort_scored_target_plug_in"}
    if reference_counts is None or reference is None:
        raise RuntimeError(
            f"{arm.name}: the disjoint unigram estimator needs a held-out reference corpus; "
            "supply one or opt in to --unigram-estimator plugin explicitly"
        )
    nats = disjoint_unigram_cross_entropy_nats(
        reference_counts, target_counts, smoothing=smoothing
    )
    return {
        **record,
        "nats": nats,
        "source": "disjoint_reference_cross_entropy",
        "smoothing": float(smoothing),
        # The smoothing constant is part of the estimate, not part of the
        # configuration: it contributes a vocabulary-tracking upward bias to
        # this baseline and therefore to every share divided by it. It travels
        # with the number rather than being recoverable only from the source.
        "smoothing_diagnostics": smoothing_diagnostics(
            reference_counts, target_counts, smoothing=smoothing
        ),
        "reference": {
            **dict(reference),
            "tokens": int(np.asarray(reference_counts).sum()),
            "distinct_tokens": int((np.asarray(reference_counts) > 0).sum()),
        },
    }


from .arms import scoring_target_alphabet, rendering_marker_ids, require_scoring_target_ids, conditioning_boundary_ids
from .budget import SparseCounts
from .scoring import target_rule

def scored_target_records(
    arm: Arm, strings: Sequence[str], *, max_len: int
) -> tuple[np.ndarray, SparseCounts]:
    """Next-token-target counts over exactly the multiset ``scored_tokens`` scores.

    Applies :func:`src.capability.core.scoring.target_rule` without a forward pass, so
    that a held-out reference corpus and the scored cohort are counted over the
    same kind of token. Counting the reference over a different span --
    including ZymCTRL's EC tag, say -- would fit the context-free baseline on a
    distribution the model is never scored against.

    The span arithmetic is open-coded rather than routed through
    :func:`src.capability.core.scoring.sequence_target_mask` because this path has no
    attention mask and no batch: it walks one untruncated id list at a time. The
    *rule* and the *boundary ids* still come from the shared declarations, which
    is where the two used to be able to drift apart.

    Returns the dense count vector every unigram estimator consumes and the
    per-record counts it sums from, in one pass, because a caller that persists
    the per-record statistics would otherwise tokenise a four-thousand-record
    reference corpus a second time to get them. Rows are aligned with
    ``strings``: a row with no scored target is an empty span, not a missing
    record.

    The ``budget`` capability is required for the scoring-target alphabet read
    below, for the reason :func:`src.capability.core.budget.arm_power_with_records`
    gives.
    """

    if max_len < 2:
        raise ValueError("max_len must admit at least one next-token target")
    if arm.spec.architecture != "rita":
        arm.require("budget")
    alphabet = scoring_target_alphabet(arm.spec, getattr(arm.model, "config", None))
    vocab = int(alphabet["size"])
    if arm.spec.architecture == "rita":
        from ..models.rita_fitness import native_encode_for_budget

        # The same exclusion :func:`src.capability.core.budget.scored_tokens` applies,
        # from the same declaration: RITA's document rendering prefixes a
        # boundary and its tokenizer appends the same token as a terminator, and
        # a position whose target is one of those ids is not cohort content. A
        # reference counted over the terminator too would fit the context-free
        # baseline on a token the model is never scored on.
        rita_markers = set(rendering_marker_ids(arm))
        rows = []
        for text in strings:
            ids = native_encode_for_budget(arm.tokenizer, text)[:max_len]
            array = (
                np.asarray(
                    [value for value in ids[1:] if value not in rita_markers],
                    dtype=np.int64,
                )
                if len(ids) >= 2
                else np.asarray([], dtype=np.int64)
            )
            require_scoring_target_ids(array, alphabet, arm=arm.name)
            rows.append(array)
        per_record = SparseCounts.from_records(rows)
        counts = per_record.vocabulary_totals(vocab)
        if counts.sum() < 1:
            raise RuntimeError(f"{arm.name}: reference corpus yields no scored targets")
        return counts, per_record
    if arm.spec.architecture == "progen3":
        from ..models.progen3 import n_to_c_target_rows, require_progen3_handle

        rows = n_to_c_target_rows(
            require_progen3_handle(arm), list(strings), max_len=max_len
        )
        for array in rows:
            require_scoring_target_ids(array, alphabet, arm=arm.name)
        per_record = SparseCounts.from_records(rows)
        counts = per_record.vocabulary_totals(vocab)
        if counts.sum() < 1:
            raise RuntimeError(f"{arm.name}: reference corpus yields no scored targets")
        return counts, per_record
    conditioned = target_rule(arm.spec.input_format) == "between_boundaries"
    start_id, end_id = conditioning_boundary_ids(arm)
    # The same exclusion :func:`src.capability.core.budget.scored_tokens` applies, from
    # the same declaration: a position whose target is a marker the rendering
    # itself added is not cohort content, and the held-out reference has to be
    # counted over the span the model is scored on rather than over a wider one.
    markers = set() if conditioned else set(rendering_marker_ids(arm))
    records: list[np.ndarray] = []
    for text in strings:
        ids = arm.tokenizer(text, return_tensors=None)["input_ids"][:max_len]
        targets: list[int] = []
        if len(ids) >= 2:
            if conditioned:
                if ids.count(start_id) != 1 or ids.count(end_id) != 1:
                    raise ValueError(f"{arm.name}: row lacks exactly one <start>/<end> pair")
                targets = ids[ids.index(start_id) + 1 : ids.index(end_id)]
            else:
                targets = [value for value in ids[1:] if value not in markers]
        array = np.asarray(targets, dtype=np.int64)
        require_scoring_target_ids(array, alphabet, arm=arm.name)
        records.append(array)
    per_record = SparseCounts.from_records(records)
    counts = per_record.vocabulary_totals(vocab)
    if counts.sum() < 1:
        raise RuntimeError(f"{arm.name}: reference corpus yields no scored targets")
    return counts, per_record
