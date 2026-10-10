"""Computational phenotype of a generated protein, and selection on the pool it came from.

Why this module is separate from ``generation``
===============================================

Producing a sequence and judging it are different roles, and the judgement has to
be made with instruments that know nothing about the producer. Everything here
reads a frozen generation ledger and an instrument's output; nothing here samples
from a model, and nothing here is allowed to become a second generator or a
second family oracle.

The three phenotype components, and the one that is unavailable
===============================================================

E14 asks whether generated proteins show plausible *structural*, *stability-related*
and *function-related* properties.

**Structure** is ESMFold2 confidence (CA pLDDT, pTM, PAE), read through the
project's existing instrument. It is a prediction about a sequence, not an
observed fold.

**Function-related** is the restored Pfam/HMMER oracle's family assignment at the
release's own gathering thresholds. It is sequence-level homology to a curated
profile, which is weaker than function: a complete-domain assignment says the
sequence looks like a member of a family, not that it does what the family does.

**Stability is not measured.** No stability or ddG predictor exists anywhere in
this project's staged assets (see :data:`STABILITY_UNAVAILABLE`), and predicted
*confidence* is not stability: pLDDT is the predictor's own uncertainty about
coordinates and is driven by length, completeness and composition, all of which
differ between generated and natural sequences. Reporting it under a stability
heading would substitute a quantity the project can compute for the quantity the
question asks about. The component is therefore reported as unavailable, with its
reasons, rather than filled.

The two confounds that decide whether the comparison says anything
==================================================================

**Length and completeness.** Predicted confidence rises steeply with both: a
64-residue fragment of a real protein folds to pLDDT 0.477 while the same protein
at 320 residues reaches 0.970. So every generated sequence here is paired with a
**whole** natural record of the same length (:func:`match_natural_records`), never
with a fragment of one, and the realised match quality is reported beside every
contrast.

**Budget censoring.** Roughly two thirds of the attempts in an unconditioned cell
exhaust the token budget instead of emitting a stop token. A censored attempt is a
truncated continuation -- a fragment -- and folds badly for reasons that have
nothing to do with protein knowledge. Native and censored attempts also differ
systematically in length (censored attempts pile up at the budget), so they cannot
be compared with each other at matched length. They are instead each compared with
*their own* length-matched natural comparator, and the two gaps are reported
separately: see :func:`classify_outcome` and :data:`STRATA`.

Selection (E17)
===============

A selection experiment is only valid if the candidate pool is fixed before any
evaluation and the evaluator is independent of the selector.
:func:`selection_yield` scores a pool member set chosen by a selector score; the
selector scores live in this module (model likelihood, a Swiss-Prot unigram
composition score, their rank average, and a seeded random key) and the evaluator
values come from ESMFold2 and the Pfam oracle, neither of which reads a selector.
Independence here means "not the same quantity and not derived from it"; it does
not mean uncorrelated, and the composition selector exists precisely to price how
much of any likelihood gain a cheaper feature already buys.

Every curve is reported as a function of selection fraction at equal selected-set
size, because a method that wins at one operating point is not a method, and every
curve carries the repertoire of the selected set beside its yield, because a yield
bought by collapsing onto a narrow set of families is not a gain.

Too few good candidates, or a selector that cannot find them?
=============================================================

A selector that barely beats chance admits two readings, and a yield curve alone
cannot tell them apart: generation may have produced too few promising candidates
for any ranking to find, or the promising ones may be there and the ranking may
miss them. :func:`gap_decomposition` separates the two on the frozen pool by
adding the ceiling an unusable selector would reach -- the mean over the best *k*
members by the evaluator itself. The oracle minus random is how much selectable
quality the pool holds; the oracle minus the selector is how much of it the
selector fails to reach; their ratio localises the failure, and where the ratio's
interval spans the split the decomposition says so and reports the number of
independence units that would resolve it instead of picking a side.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..core.statistics import (
    MINIMUM_FINITE_DRAW_FRACTION,
    bootstrap_unit_floor,
    mean_interval,
    paired_group_bootstrap,
)

SCHEMA_VERSION = "d1_generated_phenotype_v1"

#: The 20 canonical residues, in the order the structure instrument declares.
AA20 = "ARNDCQEGHILKMFPSTWYV"

#: The residue band in which this project holds measured *absolute* folding free
#: energies. Both staged sources are small-domain proteolysis assays: the
#: MegaScale backgrounds are 32-74 residues, and the recalibrated MGnify dG table
#: carries 43-80 residue sequences. The band is the support any absolute-stability
#: instrument built here would have, and it is declared because it, not the
#: absence of a predictor, is what actually limits E14's stability component.
SMALL_DOMAIN_STABILITY_BAND: tuple[int, int] = (43, 80)

#: Why E14's stability component is reported as unavailable rather than filled.
#: Reasons, not an apology: each one is checkable, and the last two say what would
#: lift the limitation and why lifting it would still not reach this question.
STABILITY_UNAVAILABLE: dict[str, str] = {
    "status": "unavailable",
    "no_predictor_is_staged": (
        "no stability, ddG or melting-temperature predictor is vendored in external/, "
        "staged under the pod's model root, or reachable from this host, and nothing "
        "may be installed or downloaded from inside a pod"
    ),
    "confidence_is_not_stability": (
        "ESMFold2 pLDDT, pTM and PAE are the structure predictor's confidence in its "
        "own coordinates. They respond to length, completeness and composition, and "
        "they carry no free-energy or melting-temperature units. Reporting them under "
        "a stability heading would answer a different question than the one asked"
    ),
    "ddG_predictors_do_not_apply": (
        "the staged stability measurements (MegaScale cDNA-display proteolysis, "
        "Domainome abundance) and every ddG predictor built on them score a "
        "substitution against a wild type. A generated sequence has no wild type, so "
        "the well-posed quantity for it is an absolute one -- chain dG or Tm -- not a ddG"
    ),
    "what_would_lift_it": (
        "a single-sequence absolute-dG or melting-temperature predictor, validated on "
        "measured absolute dG before any generated sequence is scored with it. The "
        "measurements to validate one are already staged: data/mgnify_stability_cho2026 "
        "is the supplementary release of a small-domain absolute-dG study and carries "
        "recalibrated dG in kcal/mol with the amino-acid sequence and a declared "
        "train/test/validation split. Only the predictor itself is missing"
    ),
    "why_it_would_still_not_answer_this_question": (
        "every staged absolute-dG measurement sits in the small-domain band "
        f"{SMALL_DOMAIN_STABILITY_BAND[0]}-{SMALL_DOMAIN_STABILITY_BAND[1]} residues, so a "
        "predictor built on it supports only the short end of the generated length range. "
        "A budget-censored continuation near the token budget is several times that length "
        "and a two-state folding free energy is not even well defined for it. The "
        "cohort's own support inside the band is recorded in the E14 declaration, per arm "
        "and per stratum, so the reader can see how little of the question a predictor "
        "would reach rather than take this on trust"
    ),
}

#: What the two available instruments do and do not establish.
CEILING: dict[str, str] = {
    "structure_is_predicted": (
        "ESMFold2 confidence is indirect predicted structural feasibility, not an "
        "experimentally verified fold"
    ),
    "function_is_homology": (
        "a Pfam complete-domain assignment is sequence-level homology to a curated "
        "profile at the release's gathering thresholds. It is not folding, not "
        "catalysis, and not novelty"
    ),
    "stability_is_absent": STABILITY_UNAVAILABLE["status"],
    "no_experimental_validation": (
        "nothing here is wet-lab evidence. Whether any of these sequences expresses, "
        "folds or functions is unresolved and awaits experiment"
    ),
    "matched_not_equivalent": (
        "the natural comparator is matched on length and on being a whole record. It "
        "is not matched on composition, family or domain architecture, so a residual "
        "gap is not attributable to any single property"
    ),
}

#: The two generated strata, declared before anything is folded.
#:
#: ``native`` is an attempt that emitted its own stop token and whose product
#: lands in a length band the structure instrument handles well. ``censored`` is
#: an attempt that reached the token budget; its product piles up at the budget
#: length, which is why it gets its own band and its own natural comparator
#: rather than being compared with ``native``.
STRATA: dict[str, dict[str, Any]] = {
    "native": {
        "decoder_stop": "eos",
        "min_length": 50,
        "max_length": 320,
        "meaning": "the model chose to stop; the product is a finished candidate",
    },
    "censored": {
        "decoder_stop": "max_new_tokens",
        "min_length": 321,
        "max_length": 400,
        "meaning": (
            "the token budget stopped the model; the product is a truncated "
            "continuation, i.e. a fragment"
        ),
    },
}

#: A natively terminated attempt shorter than this is not a protein candidate at
#: all. Several checkpoints emit an immediate stop token, and counting a
#: zero-residue or one-residue product as "the model terminated natively" would
#: inflate the native rate with empty output. Such attempts are counted in their
#: own outcome class.
NATIVE_DEGENERATE_BELOW = STRATA["native"]["min_length"]

#: Selection fractions the yield curve is reported at. A single operating point
#: cannot distinguish a method from an artefact of where its threshold fell. The
#: grid reaches 1.00, where every method necessarily selects the whole pool and
#: the difference between methods is identically zero: that point anchors the
#: curve and is reported without a bootstrap, because an interval around an
#: arithmetic identity would be fabricated width.
SELECTION_FRACTIONS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.25, 0.50, 1.00)

#: Independent random keys the random baseline is averaged over, so the baseline
#: that must be beaten is not one lucky or unlucky draw.
RANDOM_BASELINE_KEYS = 64

#: k for the k-mer repertoire distance. Three is short enough that two unrelated
#: proteins of ordinary composition still share most of their alphabet, so the
#: distance moves on repeat structure and compositional collapse, which is what
#: likelihood selection is suspected of producing.
DIVERSITY_KMER = 3


# ------------------------------------------------------------------ attempts


def classify_outcome(row: Mapping[str, Any]) -> str:
    """The outcome class of one generation attempt.

    Four classes, not two. ``native_degenerate`` exists because a stop token
    emitted immediately is a native termination that produced nothing, and
    pooling it with real native products would make truncation look like the only
    failure mode. ``censored_outside_band`` and ``native_outside_band`` are
    attempts whose product falls outside the length band its stratum folds in;
    they stay in the census denominator and out of the folded cohort.
    """

    if not row.get("valid_aa20", False):
        return "non_canonical_residues"
    length = int(row["length"])
    stop = str(row.get("decoder_stop") or "")
    if stop == STRATA["native"]["decoder_stop"]:
        if length < NATIVE_DEGENERATE_BELOW:
            return "native_degenerate"
        if length <= STRATA["native"]["max_length"]:
            return "native"
        return "native_outside_band"
    if stop == STRATA["censored"]["decoder_stop"]:
        if STRATA["censored"]["min_length"] <= length <= STRATA["censored"]["max_length"]:
            return "censored"
        return "censored_outside_band"
    return f"unknown_decoder_stop:{stop}"


def termination_census(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-arm and pooled outcome counts, reported as a primary fact.

    The headline structural number for generated proteins is mostly a statement
    about this table: if most attempts never finished, most "generated proteins"
    are fragments and their folding is a measure of truncation.
    """

    per_arm: dict[str, Counter] = {}
    pooled: Counter = Counter()
    cells: dict[str, Counter] = {}
    stops: Counter = Counter()
    cell_stops: dict[str, Counter] = {}
    for row in rows:
        arm = str(row.get("arm") or "unknown")
        cell = f"{row.get('campaign')}__{arm}__{row.get('condition')}"
        outcome = classify_outcome(row)
        stop = str(row.get("decoder_stop") or "unrecorded")
        per_arm.setdefault(arm, Counter())[outcome] += 1
        cells.setdefault(cell, Counter())[outcome] += 1
        cell_stops.setdefault(cell, Counter())[stop] += 1
        pooled[outcome] += 1
        stops[stop] += 1
    total = sum(pooled.values())
    if total < 1:
        raise ValueError("the termination census was given no attempt")
    censored = stops[STRATA["censored"]["decoder_stop"]]
    per_cell_censored = [
        counter[STRATA["censored"]["decoder_stop"]] for counter in cell_stops.values()
    ]
    return {
        "n_attempts": int(total),
        "n_cells": len(cells),
        # The termination fact is counted on the stop reason alone. The outcome
        # classes below additionally apply composition and length eligibility, so
        # they are a cohort statement and must not be read as a termination rate:
        # a censored attempt carrying a non-canonical residue is censored.
        "decoder_stop": dict(sorted(stops.items())),
        "budget_censored_fraction": censored / total,
        "budget_censored_per_cell_mean": float(np.mean(per_cell_censored)),
        "budget_censored_per_cell_range": [
            int(min(per_cell_censored)),
            int(max(per_cell_censored)),
        ],
        "outcome_classes_pooled": dict(sorted(pooled.items())),
        "outcome_classes_per_arm": {
            arm: dict(sorted(counter.items())) for arm, counter in sorted(per_arm.items())
        },
        "decoder_stop_per_cell": {
            cell: dict(sorted(counter.items())) for cell, counter in sorted(cell_stops.items())
        },
        "interpretation": (
            "a budget-censored attempt is a truncated continuation. It is a fragment, "
            "and fragments fold badly for reasons unrelated to the model's protein "
            "knowledge, so no pooled structural number may be read without this table. "
            "native_degenerate counts attempts that emitted a stop token immediately "
            "and produced nothing, which is a native termination that is not a product"
        ),
    }


# -------------------------------------------------------- natural comparator


def match_natural_records(
    targets: Sequence[Mapping[str, Any]],
    natural: Sequence[Mapping[str, Any]],
    *,
    tolerance: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair each target with one whole natural record of near-identical length.

    Drawn without replacement across the whole request, so no natural record
    prices two generated sequences, and seeded, so the draw is reproducible from
    the artefact. Unmatched targets are returned in the report rather than
    dropped silently: an unmatched target means the length band has no natural
    support, which is a fact about the band.

    ``natural`` records must be *whole* entries. A length-matched fragment of a
    natural protein is the comparator this project already measured and found to
    flatter generation by a mean 0.064 of complete-domain yield, because being a
    fragment is itself a folding penalty.
    """

    if tolerance < 0:
        raise ValueError("the length tolerance is non-negative")
    by_length: dict[int, list[int]] = {}
    for index, record in enumerate(natural):
        by_length.setdefault(int(record["length"]), []).append(index)
    rng = np.random.default_rng(seed)
    for indices in by_length.values():
        rng.shuffle(indices)
    used: set[int] = set()
    pairs: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    # Longest first: long bands are the sparse ones, so they choose before a
    # short band can consume a record that would also have fitted them.
    order = sorted(range(len(targets)), key=lambda i: (-int(targets[i]["length"]), str(targets[i]["id"])))
    for position in order:
        target = targets[position]
        want = int(target["length"])
        chosen: int | None = None
        for offset in sorted(range(-tolerance, tolerance + 1), key=lambda value: (abs(value), value)):
            for candidate in by_length.get(want + offset, ()):
                if candidate not in used:
                    chosen = candidate
                    break
            if chosen is not None:
                break
        if chosen is None:
            unmatched.append({"id": str(target["id"]), "length": want})
            continue
        used.add(chosen)
        pairs.append(
            {
                "pair_id": str(target["id"]),
                "target": dict(target),
                "natural": dict(natural[chosen]),
                "length_difference": int(natural[chosen]["length"]) - want,
            }
        )
    if not pairs:
        raise ValueError(
            "no target could be matched to a whole natural record; an unmatched "
            "comparison is refused rather than reported against nothing"
        )
    differences = np.asarray([pair["length_difference"] for pair in pairs], dtype=np.float64)
    report = {
        "n_targets": len(targets),
        "n_matched": len(pairs),
        "n_unmatched": len(unmatched),
        "unmatched": unmatched[:50],
        "tolerance_residues": int(tolerance),
        "seed": int(seed),
        "mean_absolute_length_difference": float(np.abs(differences).mean()),
        "max_absolute_length_difference": float(np.abs(differences).max()),
        "mean_signed_length_difference": float(differences.mean()),
        "comparator_kind": "whole_natural_record",
        "comparator_note": (
            "whole Swiss-Prot entries, not fragments. A length-matched fragment "
            "comparator is easier to beat and is not used here"
        ),
    }
    return pairs, report


# ---------------------------------------------------------- cheap descriptors


def residue_background(sequences: Iterable[str]) -> dict[str, float]:
    """A smoothed unigram residue distribution over canonical residues.

    Add-one smoothing on all twenty residues, so a background fitted on a small
    draw cannot assign probability zero to a residue a generated sequence uses
    and turn its composition score into an infinity.
    """

    counts = Counter()
    for sequence in sequences:
        counts.update(residue for residue in sequence if residue in AA20)
    total = sum(counts.values())
    if total < 1:
        raise ValueError("the residue background was fitted on no canonical residue")
    denominator = total + len(AA20)
    return {residue: (counts[residue] + 1) / denominator for residue in AA20}


def composition_cross_entropy(sequence: str, background: Mapping[str, float]) -> float:
    """Nats per residue of a sequence's composition under a residue background.

    The cheap feature. It knows nothing about order, structure or family: it
    prices only how unusual the residue mix is against natural protein
    composition. It is here as the control the E17 likelihood arm has to beat --
    a parallel experiment in this programme watched a +0.10 gain collapse to
    +0.008 once a cheaper feature was added to the comparison.
    """

    usable = [residue for residue in sequence if residue in AA20]
    if not usable:
        raise ValueError("a composition score needs at least one canonical residue")
    missing = sorted(set(usable) - set(background))
    if missing:
        raise ValueError(f"the background assigns no probability to {missing}")
    return float(-sum(math.log(background[residue]) for residue in usable) / len(usable))


def kmer_vector(sequence: str, k: int = DIVERSITY_KMER) -> dict[str, int]:
    if k < 1:
        raise ValueError("k must be positive")
    counts: Counter = Counter()
    for start in range(len(sequence) - k + 1):
        counts[sequence[start : start + k]] += 1
    return dict(counts)


def mean_pairwise_kmer_distance(sequences: Sequence[str], k: int = DIVERSITY_KMER) -> float:
    """Mean cosine distance between k-mer count vectors over all distinct pairs.

    One scalar for "how much of a repertoire is this set". It falls when a
    selected set collapses onto near-copies of one thing, which is the failure
    mode a likelihood-selected set is suspected of.
    """

    if len(sequences) < 2:
        raise ValueError("a pairwise distance needs at least two sequences")
    vectors = [kmer_vector(sequence, k) for sequence in sequences]
    norms = [math.sqrt(sum(value * value for value in vector.values())) for vector in vectors]
    if min(norms) <= 0.0:
        raise ValueError("a sequence shorter than k contributes no k-mer")
    total = 0.0
    pairs = 0
    for i in range(len(vectors)):
        left, left_norm = vectors[i], norms[i]
        for j in range(i + 1, len(vectors)):
            right = vectors[j]
            # Iterate the smaller mapping and look the k-mer up in the larger
            # one. Iterating one mapping and looking up in the *same* one
            # computes its squared norm instead of the dot product, which drives
            # the cosine above one and the distance negative.
            if len(left) <= len(right):
                shared = sum(count * right.get(mer, 0) for mer, count in left.items())
            else:
                shared = sum(count * left.get(mer, 0) for mer, count in right.items())
            distance = 1.0 - shared / (left_norm * norms[j])
            if not -1e-9 <= distance <= 1.0 + 1e-9:
                raise AssertionError(
                    f"a cosine distance of {distance} is outside [0, 1]; the k-mer "
                    "vectors or their norms disagree"
                )
            total += min(max(distance, 0.0), 1.0)
            pairs += 1
    return float(total / pairs)


#: A residue repeated at least this many times in a row marks a sequence as
#: carrying a homopolymer run. Eight is well beyond what natural protein
#: composition produces at any appreciable rate and well inside what a degenerate
#: decoder produces, so the fraction of a selected set above it is a direct
#: reading of whether selection is concentrating on repeats.
HOMOPOLYMER_RUN_THRESHOLD = 8

#: How many independent random keys the size-matched diversity reference averages
#: over. Fewer than the yield baseline uses, because each key costs an
#: all-pairs distance over the selected set.
DIVERSITY_REFERENCE_KEYS = 16

#: Identity over query, in percent, at or above which a generated sequence counts
#: as a near-duplicate of a corpus entry. It is the upper stratum edge the
#: novelty search already declares, named here so that the selected-set profile
#: and the collapse check read the same threshold as the census they are compared
#: against. A selected set richer in near-duplicates than a size-matched random
#: draw is retrieving natural sequences, which is the sharpest way an apparent
#: selection gain can fail to be design.
NEAR_DUPLICATE_IDENTITY = 95.0


def longest_homopolymer_run(sequence: str) -> int:
    """The longest run of one repeated residue."""

    if not sequence:
        raise ValueError("a run length needs a sequence")
    best = run = 1
    for previous, current in zip(sequence, sequence[1:]):
        run = run + 1 if current == previous else 1
        best = max(best, run)
    return best


def selected_set_profile(
    sequences: Sequence[str],
    family_sets: Sequence[Sequence[str]],
    *,
    kmer: int = DIVERSITY_KMER,
    corpus_identity: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Everything about a selected set other than its yield.

    One function, because the question the user put hardest -- whether an
    apparent gain comes from concentrating selections in a few families or in
    repetitive sequences -- cannot be answered from a yield and a family count
    read in different places. Repertoire breadth, duplication, low-complexity
    content and length all travel together, and length is here because every
    structural evaluator rises with it, so a selector that quietly prefers long
    sequences would otherwise look like a selector that prefers good ones.
    """

    if len(sequences) != len(family_sets):
        raise ValueError("the sequences and their family sets must align")
    if not sequences:
        raise ValueError("an empty selected set has no profile")
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.float64)
    entropies = np.asarray(
        [
            float(
                -sum(
                    share * math.log(share)
                    for share in (
                        Counter(sequence)[residue] / len(sequence) for residue in set(sequence)
                    )
                    if share > 0.0
                )
            )
            for sequence in sequences
        ],
        dtype=np.float64,
    )
    runs = np.asarray([longest_homopolymer_run(sequence) for sequence in sequences])
    profile: dict[str, Any] = {
        "n_sequences": len(sequences),
        "n_distinct_sequences": len(set(sequences)),
        "duplicate_fraction": 1.0 - len(set(sequences)) / len(sequences),
        "mean_length": float(lengths.mean()),
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "mean_composition_entropy_nats": float(entropies.mean()),
        "min_composition_entropy_nats": float(entropies.min()),
        "longest_homopolymer_run_max": int(runs.max()),
        "homopolymer_run_threshold": int(HOMOPOLYMER_RUN_THRESHOLD),
        "fraction_with_homopolymer_run": float(np.mean(runs >= HOMOPOLYMER_RUN_THRESHOLD)),
        **family_repertoire(family_sets),
    }
    profile["mean_pairwise_kmer_distance"] = (
        mean_pairwise_kmer_distance(list(sequences), kmer) if len(sequences) >= 2 else None
    )
    if corpus_identity is None:
        profile["nearest_corpus_identity"] = None
        profile["max_nearest_corpus_identity"] = None
        profile["fraction_near_duplicate_of_corpus"] = None
        profile["nearest_corpus_identity_note"] = "no homology search was supplied"
    else:
        identity = np.asarray(corpus_identity, dtype=np.float64)
        if identity.size != len(sequences):
            raise ValueError("the corpus identities must align with the sequences")
        profile["nearest_corpus_identity"] = float(identity.mean())
        profile["max_nearest_corpus_identity"] = float(identity.max())
        profile["fraction_near_duplicate_of_corpus"] = float(
            np.mean(identity >= NEAR_DUPLICATE_IDENTITY)
        )
        profile["near_duplicate_identity_threshold"] = float(NEAR_DUPLICATE_IDENTITY)
    return profile


def diversity_reference(
    sequences: Sequence[str],
    family_sets: Sequence[Sequence[str]],
    *,
    fraction: float,
    seed: int,
    n_keys: int = DIVERSITY_REFERENCE_KEYS,
    corpus_identity: Sequence[float] | None = None,
) -> dict[str, Any]:
    """The diversity a *random* set of the same size has, as the reference.

    Size-matched, because every repertoire measure falls as a set shrinks: a
    twelve-sequence selection covers fewer families than a six-hundred-sequence
    pool whatever the selector did. Comparing a method's selected set against the
    pool would therefore report a collapse at every small fraction. The honest
    reference is a random draw of the same size.
    """

    keys = [
        selection_indices(random_key(len(sequences), seed=seed + index), fraction=fraction)
        for index in range(n_keys)
    ]
    profiles = [
        selected_set_profile(
            [sequences[i] for i in chosen],
            [family_sets[i] for i in chosen],
            corpus_identity=None
            if corpus_identity is None
            else [corpus_identity[i] for i in chosen],
        )
        for chosen in keys
    ]
    summary: dict[str, Any] = {"n_keys": int(n_keys), "fraction": float(fraction)}
    for field in (
        "distinct_families",
        "effective_families",
        "mean_pairwise_kmer_distance",
        "mean_composition_entropy_nats",
        "fraction_with_homopolymer_run",
        "mean_length",
        "duplicate_fraction",
        "nearest_corpus_identity",
        "fraction_near_duplicate_of_corpus",
    ):
        values = [profile[field] for profile in profiles if profile[field] is not None]
        summary[field] = (
            {"mean": float(np.mean(values)), "interval": mean_interval(values)["interval"]}
            if len(values) >= 2
            else None
        )
    return summary


#: The diversity axes a yield gain is checked against, and the direction that
#: counts as a collapse on each.
COLLAPSE_AXES: dict[str, str] = {
    "effective_families": "below",
    "mean_pairwise_kmer_distance": "below",
    "mean_composition_entropy_nats": "below",
    "fraction_with_homopolymer_run": "above",
    # A selected set more similar to the corpus than a size-matched random draw is
    # drifting toward retrieval of natural sequences, which is the other way an
    # apparent gain can fail to be a gain. Mean identity and the near-duplicate
    # share are both read, because they fail differently: a set can rise in mean
    # identity by concentrating on remote homologues without acquiring a single
    # near-duplicate, and it can concentrate onto the handful of verbatim corpus
    # members while its mean identity barely moves.
    "nearest_corpus_identity": "above",
    "fraction_near_duplicate_of_corpus": "above",
}


def collapse_check(
    profile: Mapping[str, Any], reference: Mapping[str, Any], *, yield_difference: float | None
) -> dict[str, Any]:
    """Whether a yield gain at this operating point was bought with a collapse.

    A computed verdict, not a remark, because the question is whether to believe
    the gain at all. An axis is flagged when the selected set falls outside the
    size-matched random reference's interval in the collapsing direction. A
    positive yield difference with any axis flagged is reported as
    ``gain_is_not_a_gain``: more recognised or better-folding candidates drawn
    from a narrower, more repetitive repertoire is a different product, not a
    better one.
    """

    flagged: dict[str, Any] = {}
    for axis, direction in COLLAPSE_AXES.items():
        observed = profile.get(axis)
        band = reference.get(axis)
        if observed is None or band is None:
            continue
        low, high = band["interval"]
        if direction == "below" and observed < low:
            flagged[axis] = {"observed": observed, "random_interval": [low, high], "moved": "below"}
        elif direction == "above" and observed > high:
            flagged[axis] = {"observed": observed, "random_interval": [low, high], "moved": "above"}
    gain = yield_difference is not None and yield_difference > 0.0
    return {
        "axes_checked": sorted(COLLAPSE_AXES),
        "axes_flagged": flagged,
        "collapsed": bool(flagged),
        "yield_gain": bool(gain),
        "gain_is_not_a_gain": bool(gain and flagged),
        "verdict": (
            "a positive yield difference accompanied by a repertoire or complexity "
            "collapse relative to a size-matched random selection. The selected set is "
            "a narrower product, not a better one"
            if gain and flagged
            else (
                "yield gain with no collapse detected on the checked axes"
                if gain
                else (
                    "no yield gain at this operating point; the diversity axes are "
                    "reported for completeness"
                )
            )
        ),
    }


def family_repertoire(family_sets: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Breadth of the Pfam repertoire a set of sequences covers.

    ``effective_families`` is the exponential of the Shannon entropy of the
    family-occurrence distribution: the number of equally common families that
    would give the same concentration. It separates "forty families, one of them
    everything" from "forty families, evenly used", which a distinct count
    cannot.
    """

    counts: Counter = Counter()
    n_with_family = 0
    for families in family_sets:
        if families:
            n_with_family += 1
            counts.update(set(families))
    total = sum(counts.values())
    if total < 1:
        return {
            "n_sequences": len(family_sets),
            "n_with_any_family": 0,
            "distinct_families": 0,
            "effective_families": 0.0,
            "top_family_share": None,
        }
    shares = np.asarray([value / total for value in counts.values()], dtype=np.float64)
    entropy = float(-(shares * np.log(shares)).sum())
    return {
        "n_sequences": len(family_sets),
        "n_with_any_family": int(n_with_family),
        "distinct_families": int(len(counts)),
        "effective_families": float(math.exp(entropy)),
        "top_family_share": float(shares.max()),
    }


# -------------------------------------------------------------- the contrasts


def _mean_metric(_truth: np.ndarray, values: np.ndarray) -> float:
    return float(np.mean(values))


def matched_contrast(
    generated: Sequence[float],
    natural: Sequence[float],
    groups: Sequence[Any],
    *,
    seed: int,
    n_bootstrap: int = 10000,
) -> dict[str, Any]:
    """Generated minus its length-matched whole-natural comparator, paired.

    The pairing is the point: the same resampled units score both sides every
    draw, so the interval is about the gap and not about the two populations'
    separate spreads. Below the package's unit floor the contrast is reported as
    unresolved rather than given an interval, because a percentile interval over
    a handful of atoms can come out narrower than one over hundreds and would be
    compared against it.
    """

    left = np.asarray(generated, dtype=np.float64)
    right = np.asarray(natural, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size < 1:
        raise ValueError("the paired vectors must be one-dimensional and the same length")
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise ValueError("a matched contrast was given a non-finite evaluator value")
    units = sorted({str(group) for group in groups})
    floor = bootstrap_unit_floor(len(units))
    record: dict[str, Any] = {
        "n_pairs": int(left.size),
        "generated_mean": float(left.mean()),
        "natural_mean": float(right.mean()),
        "difference": float(left.mean() - right.mean()),
        "unit_floor": floor,
        "resolved": not floor["degenerate"],
    }
    if floor["degenerate"]:
        record["difference_ci95"] = None
        return record
    bootstrap = paired_group_bootstrap(
        np.zeros(left.size),
        left,
        right,
        [str(group) for group in groups],
        _mean_metric,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    record["difference_ci95"] = bootstrap["difference_ci95"]
    record["n_finite_draws"] = bootstrap["n_finite_draws"]
    record["n_groups"] = bootstrap["n_groups"]
    record["excludes_zero"] = bool(
        bootstrap["difference_ci95"][0] > 0.0 or bootstrap["difference_ci95"][1] < 0.0
    )
    return record


def _stratum_gap_metric(stratum_sign: np.ndarray, values: np.ndarray) -> float:
    """Native-stratum mean minus censored-stratum mean of one side of the pairing.

    The stratum label rides in the bootstrap's first argument, which
    :func:`src.capability.core.statistics.paired_group_bootstrap` passes to the
    metric untouched. Taking the difference of this metric between the generated
    and the natural side gives

        (generated_native - natural_native) - (generated_censored - natural_censored)

    which is the gap in the native stratum minus the gap in the censored stratum:
    how much of the generation deficit is truncation. Each stratum keeps its own
    length-matched comparator, so neither mean is compared across lengths.
    """

    native = values[stratum_sign > 0]
    censored = values[stratum_sign < 0]
    if native.size < 1 or censored.size < 1:
        return float("nan")
    return float(native.mean() - censored.mean())


def stratum_gap_contrast(
    generated: Sequence[float],
    natural: Sequence[float],
    strata: Sequence[str],
    groups: Sequence[Any],
    *,
    seed: int,
    n_bootstrap: int = 10000,
) -> dict[str, Any]:
    """How much more of the generated-minus-natural gap the censored stratum carries."""

    left = np.asarray(generated, dtype=np.float64)
    right = np.asarray(natural, dtype=np.float64)
    sign = np.asarray([1.0 if name == "native" else -1.0 for name in strata], dtype=np.float64)
    if not (left.shape == right.shape == sign.shape) or left.ndim != 1:
        raise ValueError("the paired vectors and stratum labels must align")
    unknown = sorted(set(strata) - set(STRATA))
    if unknown:
        raise ValueError(f"unknown stratum label(s) {unknown}")
    if np.all(sign > 0) or np.all(sign < 0):
        return {
            "resolved": False,
            "reason": "only one stratum has support, so no stratum difference exists",
        }
    unit_ids = [str(group) for group in groups]
    floor = bootstrap_unit_floor(len({*unit_ids}))
    if floor["degenerate"]:
        return {"resolved": False, "unit_floor": floor, "reason": floor["degenerate_reason"]}
    bootstrap = paired_group_bootstrap(
        sign, left, right, unit_ids, _stratum_gap_metric, seed=seed, n_bootstrap=n_bootstrap
    )
    return {
        "resolved": True,
        "gap_native_minus_gap_censored": bootstrap["difference"],
        "ci95": bootstrap["difference_ci95"],
        "n_groups": bootstrap["n_groups"],
        "n_finite_draws": bootstrap["n_finite_draws"],
        "excludes_zero": bool(
            bootstrap["difference_ci95"][0] > 0.0 or bootstrap["difference_ci95"][1] < 0.0
        ),
        "interpretation": (
            "positive means the native stratum's generated-minus-natural gap is less "
            "negative than the censored stratum's, i.e. part of the apparent generation "
            "deficit is budget truncation rather than protein knowledge"
        ),
    }


def selection_count(n: int, fraction: float) -> int:
    """How many pool members a fraction selects. Shared by every method."""

    if n < 1:
        raise ValueError("an empty pool selects nothing")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("a selection fraction lies in (0, 1]")
    return max(1, int(math.ceil(fraction * n)))


def selection_yield(
    evaluator: Sequence[float], score: Sequence[float], *, fraction: float
) -> float:
    """Mean evaluator value over the top ``fraction`` of the pool by ``score``.

    Lower ``score`` is better by convention, so every selector is oriented once,
    where it is defined, instead of each call remembering a direction. Ties break
    on the score's own order, which is deterministic for a given pool.
    """

    values = np.asarray(evaluator, dtype=np.float64)
    keys = np.asarray(score, dtype=np.float64)
    if values.shape != keys.shape or values.ndim != 1:
        raise ValueError("the evaluator and selector vectors must align")
    take = selection_count(values.size, fraction)
    chosen = np.argsort(keys, kind="stable")[:take]
    return float(values[chosen].mean())


def selection_indices(score: Sequence[float], *, fraction: float) -> np.ndarray:
    keys = np.asarray(score, dtype=np.float64)
    return np.argsort(keys, kind="stable")[: selection_count(keys.size, fraction)]


def rank_average(*scores: Sequence[float]) -> np.ndarray:
    """Within-pool average rank of several selectors, as a combined selector.

    Averaged ranks rather than averaged scores, because the components are in
    different units -- nats per token and nats per residue -- and a weighted sum
    of them would be an undeclared choice of exchange rate.
    """

    if len(scores) < 2:
        raise ValueError("a combination needs at least two selectors")
    stacked = [np.asarray(score, dtype=np.float64) for score in scores]
    size = stacked[0].size
    if any(item.shape != (size,) for item in stacked):
        raise ValueError("the selectors must align on the pool")
    total = np.zeros(size, dtype=np.float64)
    for item in stacked:
        order = np.argsort(np.argsort(item, kind="stable"), kind="stable")
        total += order.astype(np.float64)
    return total / len(stacked)


def random_key(n: int, *, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(n)


def random_baseline(
    evaluator: Sequence[float], *, fraction: float, seed: int, n_keys: int = RANDOM_BASELINE_KEYS
) -> dict[str, Any]:
    """The yield random selection achieves, over ``n_keys`` independent keys.

    Reported with its own interval, because one random draw is not the baseline;
    the distribution of random draws is. The pool mean is reported beside it as
    the analytic expectation of random selection at any fraction.
    """

    values = np.asarray(evaluator, dtype=np.float64)
    draws = [
        selection_yield(values, random_key(values.size, seed=seed + index), fraction=fraction)
        for index in range(n_keys)
    ]
    summary = mean_interval(draws)
    return {
        "fraction": float(fraction),
        "n_keys": int(n_keys),
        "mean": summary["mean"],
        "interval": summary["interval"],
        "pool_mean": float(values.mean()),
        "note": (
            "random selection's expectation at every fraction is the pool mean; the "
            "interval here is the spread of finite random draws at this fraction"
        ),
    }


def selection_contrast(
    evaluator: Sequence[float],
    score: Sequence[float],
    groups: Sequence[Any],
    *,
    fraction: float,
    seed: int,
    n_bootstrap: int = 10000,
) -> dict[str, Any]:
    """Top-``fraction`` yield under ``score`` minus under a random key, bootstrapped.

    The selection is recomputed inside every bootstrap draw, on the resampled
    pool, under both scores and at the same selected-set size. That is what makes
    the comparison one of methods at equal budget rather than of two fixed
    sequence sets at two thresholds.
    """

    values = np.asarray(evaluator, dtype=np.float64)
    keys = np.asarray(score, dtype=np.float64)
    if values.shape != keys.shape or values.ndim != 1:
        raise ValueError("the evaluator and selector vectors must align")
    if not np.isfinite(values).all() or not np.isfinite(keys).all():
        raise ValueError("a selection contrast was given a non-finite value")
    unit_ids = [str(group) for group in groups]
    if len(unit_ids) != values.size:
        raise ValueError("the independence units must align with the pool")
    floor = bootstrap_unit_floor(len({*unit_ids}))
    def metric(truth: np.ndarray, key: np.ndarray) -> float:
        return selection_yield(truth, key, fraction=fraction)

    record: dict[str, Any] = {
        "fraction": float(fraction),
        "n_pool": int(values.size),
        "n_selected": selection_count(values.size, fraction),
        "selected_yield": selection_yield(values, keys, fraction=fraction),
        "unit_floor": floor,
        "resolved": not floor["degenerate"],
    }
    if floor["degenerate"]:
        record["difference_vs_random_ci95"] = None
        return record
    if record["n_selected"] == values.size:
        # Every method selects the whole pool here, so the difference is zero by
        # arithmetic rather than by measurement and carries no uncertainty.
        record.update(
            random_draw_yield=record["selected_yield"],
            difference_vs_random=0.0,
            difference_vs_random_ci95=[0.0, 0.0],
            excludes_zero=False,
            identity_point=True,
        )
        return record
    bootstrap = paired_group_bootstrap(
        values,
        keys,
        random_key(values.size, seed=seed),
        unit_ids,
        metric,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    record["identity_point"] = False
    record["random_draw_yield"] = bootstrap["right_score"]
    record["difference_vs_random"] = bootstrap["difference"]
    record["difference_vs_random_ci95"] = bootstrap["difference_ci95"]
    record["n_groups"] = bootstrap["n_groups"]
    record["n_finite_draws"] = bootstrap["n_finite_draws"]
    record["excludes_zero"] = bool(
        bootstrap["difference_ci95"][0] > 0.0 or bootstrap["difference_ci95"][1] < 0.0
    )
    return record


#: The largest share of a selected set that may come from under-matched length
#: bins. A bin is under-matched when it holds fewer unselected rows than the
#: selection took from it, so the matched comparator must largely redraw the
#: selected rows themselves and the contrast is pushed toward zero by
#: construction. Above this ceiling a ``beats_length`` of false would carry no
#: information, so the contrast is reported as unresolved instead.
MAX_FORCED_MATCH_SHARE = 0.5

#: How many equal-count length bins the length-conditional comparator matches on.
#: Few enough that each bin holds a usable number of alternatives, many enough
#: that matching is real: with too many bins the matched draw is forced to be the
#: selected set itself and the contrast collapses to zero by construction.
LENGTH_MATCH_BINS = 8


def length_bins(lengths: Sequence[int], *, n_bins: int = LENGTH_MATCH_BINS) -> dict[str, Any]:
    """Equal-count length bins over the pool, and each row's bin.

    Quantile bins rather than equal-width ones, because a pool whose lengths pile
    up at one end would leave equal-width bins almost empty and make the matched
    comparator draw from a handful of rows.
    """

    array = np.asarray(lengths, dtype=np.float64)
    if array.ndim != 1 or array.size < n_bins:
        raise ValueError(f"length matching needs at least {n_bins} rows")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(array, quantiles))
    assignment = np.searchsorted(edges, array, side="right")
    counts = Counter(int(value) for value in assignment)
    return {
        "bin_of_row": assignment.astype(np.int64),
        "edges": [float(edge) for edge in edges],
        "n_bins_realised": int(len(set(assignment.tolist()))),
        "n_bins_requested": int(n_bins),
        "bin_counts": {str(key): int(value) for key, value in sorted(counts.items())},
    }


def length_matched_mean(
    values: np.ndarray,
    bin_of_row: np.ndarray,
    chosen: np.ndarray,
    generator: np.random.Generator,
) -> float:
    """Mean evaluator value of a random set matching ``chosen``'s length bins.

    The comparator every length control in this module is built on, in one place
    so that the matched draw is the same object whether it is read as a contrast
    against a selector or as the floor under an oracle ceiling.

    The draw is with replacement: inside a bootstrap resample a bin can hold
    fewer rows than the selected set took from it, and refusing there would
    condition the interval on the draws that happened to be easy.
    """

    matched: list[int] = []
    for bin_id, needed in Counter(bin_of_row[chosen].tolist()).items():
        available = np.flatnonzero(bin_of_row == bin_id)
        if available.size == 0:
            return float("nan")
        matched.extend(generator.choice(available, size=needed, replace=True).tolist())
    if not matched:
        return float("nan")
    return float(values[np.asarray(matched)].mean())


def forced_match_share(bin_of_row: np.ndarray, chosen: np.ndarray) -> float:
    """Share of ``chosen`` drawn from length bins it leaves too thin to match.

    A bin is under-matched when it holds fewer unselected rows than the selection
    took from it, so the matched comparator has to redraw the selected rows
    themselves and the contrast is pushed toward zero by construction. The share
    is reported, and compared against :data:`MAX_FORCED_MATCH_SHARE`, so a null
    result that is an artefact of the matching can be told from a real one.
    """

    if chosen.size < 1:
        raise ValueError("an empty selection has no forced-match share")
    per_bin_selected = Counter(bin_of_row[chosen].tolist())
    per_bin_available = Counter(bin_of_row.tolist())
    forced = sum(
        count
        for bin_id, count in per_bin_selected.items()
        if per_bin_available[bin_id] - count < count
    )
    return forced / chosen.size


def length_conditional_contrast(
    evaluator: Sequence[float],
    score: Sequence[float],
    lengths: Sequence[int],
    groups: Sequence[Any],
    *,
    fraction: float,
    seed: int,
    n_bootstrap: int = 10000,
    n_bins: int = LENGTH_MATCH_BINS,
) -> dict[str, Any]:
    """Does this selector beat a *length-matched* random draw at the same budget?

    The question that separates protein knowledge from a length proxy. Every
    structural confidence rises with length, so a selector correlated with length
    beats an unrestricted random draw without knowing anything about proteins. The
    comparator here is therefore not a random set of the same size but a random
    set of the same size *and the same length composition*: the selected set's
    per-bin counts are reproduced, drawing from the same bins.

    A positive interval means the selector carries information beyond length. An
    interval containing zero means that, at this budget, what the selector found
    is recoverable from length alone.

    The resampled row indices ride in the bootstrap's first argument, which
    :func:`src.capability.core.statistics.paired_group_bootstrap` passes to the
    metric untouched; the evaluator values and lengths are then read positionally.
    That indirection exists because the metric needs two aligned vectors and the
    bootstrap hands it one.
    """

    values = np.asarray(evaluator, dtype=np.float64)
    keys = np.asarray(score, dtype=np.float64)
    lengths_array = np.asarray(lengths, dtype=np.int64)
    if not (values.shape == keys.shape == lengths_array.shape) or values.ndim != 1:
        raise ValueError("the evaluator, selector and length vectors must align")
    if not np.isfinite(values).all() or not np.isfinite(keys).all():
        raise ValueError("a length-conditional contrast was given a non-finite value")
    unit_ids = [str(group) for group in groups]
    if len(unit_ids) != values.size:
        raise ValueError("the independence units must align with the pool")
    binning = length_bins(lengths_array, n_bins=n_bins)
    bin_of_row = binning["bin_of_row"]
    take = selection_count(values.size, fraction)
    record: dict[str, Any] = {
        "fraction": float(fraction),
        "n_pool": int(values.size),
        "n_selected": take,
        "length_bins": {key: binning[key] for key in ("edges", "n_bins_realised", "bin_counts")},
    }
    floor = bootstrap_unit_floor(len({*unit_ids}))
    record["unit_floor"] = floor

    # How much room the comparator actually has. A bin the selection exhausts
    # leaves nothing else to match against, so those rows contribute a forced
    # zero and a contrast built mostly from them is not a measurement.
    chosen_full = np.argsort(keys, kind="stable")[:take]
    record["forced_match_share"] = forced_match_share(bin_of_row, chosen_full)
    record["max_forced_match_share"] = float(MAX_FORCED_MATCH_SHARE)

    if floor["degenerate"] or take == values.size or record["forced_match_share"] > MAX_FORCED_MATCH_SHARE:
        record["resolved"] = False
        if floor["degenerate"]:
            record["reason"] = floor["degenerate_reason"]
        elif take == values.size:
            record["reason"] = (
                "at full selection every method takes the whole pool, so there is "
                "nothing for a length-matched comparator to differ from"
            )
        else:
            record["reason"] = (
                f"{record['forced_match_share']:.0%} of the selected set comes from "
                "length bins holding fewer unselected rows than the selection took, "
                "above the "
                f"{MAX_FORCED_MATCH_SHARE:.0%} ceiling. A matched comparator has almost "
                "nothing else to draw from, so the contrast would be driven to zero by "
                "construction and a negative verdict would carry no information"
            )
        record["difference_ci95"] = None
        return record

    # One generator for the whole contrast, so the matched draw is independent
    # from one bootstrap iteration to the next. Re-seeding inside the metric
    # would freeze the comparator and integrate over nothing, leaving the
    # interval conditioned on a single arbitrary matched draw.
    generator = np.random.default_rng(seed + 104729)

    def metric(row_indices: np.ndarray, selector: np.ndarray) -> float:
        rows = row_indices.astype(np.int64)
        local_values = values[rows]
        local_bins = bin_of_row[rows]
        size = selector.size
        wanted = selection_count(size, fraction)
        chosen = np.argsort(selector, kind="stable")[:wanted]
        matched = length_matched_mean(local_values, local_bins, chosen, generator)
        if not np.isfinite(matched):
            return float("nan")
        return float(local_values[chosen].mean() - matched)

    bootstrap = paired_group_bootstrap(
        np.arange(values.size, dtype=np.float64),
        keys,
        random_key(values.size, seed=seed + 7919),
        unit_ids,
        metric,
        seed=seed,
        n_bootstrap=n_bootstrap,
        derived_statistic=_left_selector_score,
    )
    low, high = bootstrap["derived_ci95"]
    point = bootstrap["derived_score"]
    outside = bool(point < low or point > high)
    record.update(
        resolved=True,
        n_groups=bootstrap["n_groups"],
        n_finite_draws=bootstrap["n_finite_draws"],
        gain_over_length_matched_random=point,
        difference_ci95=[low, high],
        excludes_zero=bool(low > 0.0 or high < 0.0),
        beats_length=bool(low > 0.0),
        random_selector_control=bootstrap["right_score"],
        point_outside_interval=outside,
        interpretation=(
            "the selected set's mean evaluator value minus that of a random set of "
            "the same size and the same length composition. Positive with an interval "
            "excluding zero means the selector carries information beyond length; an "
            "interval containing zero means that at this budget the gain is "
            "recoverable from length alone. The verdict is read from the interval, "
            "which is the bootstrap distribution, not from the point"
        ),
        random_selector_control_note=(
            "the same quantity computed for an unrestricted random key. It is the "
            "null's realised scale on this pool and carries its own sampling noise, "
            "so it is reported rather than assumed to be zero"
        ),
    )
    if outside:
        record["point_outside_interval_note"] = (
            "the full-sample estimate falls outside its own percentile interval. A "
            "top-fraction mean is an extreme order statistic, and resampling groups "
            "with replacement leaves about two thirds of the rows distinct, so the "
            "resampled selections are systematically less extreme than the original. "
            "The interval is still the inferential object; the point is reported "
            "beside it rather than reconciled by adjusting either"
        )
    return record


def _left_selector_score(left: float, _right: float) -> float:
    return left


# ------------------------------------------------- too few, or not found?

#: Half-width the share interval must reach before a decomposition is read as
#: separating pool content from selector skill at an operating point. Ten points
#: of the attainable gap: wide enough to be reachable at this pool size on the
#: continuous evaluators, narrow enough that a verdict drawn from it is not
#: compatible with both answers.
GAP_SHARE_TARGET_HALF_WIDTH = 0.10

#: The share of the attainable gap above which the pool, and below which the
#: selector, is named as the limiting factor. One half, declared rather than
#: chosen after seeing the numbers: a selector reaching more than half of what
#: its pool holds is limited mainly by what is there to find, and one reaching
#: less than half is limited mainly by its own ranking.
GAP_SHARE_SPLIT = 0.5

#: An attainable gap at or below this is treated as absent rather than divided
#: by. It happens for real: an evaluator that is constant on the pool, or a
#: budget that takes the whole pool, leaves nothing to decompose.
MINIMUM_ATTAINABLE_GAP = 1e-9


def oracle_key(evaluator: Sequence[float]) -> np.ndarray:
    """The selector that *is* the evaluator: the pool's own ceiling at any budget.

    Deliberately circular, and useful for exactly that reason. It cannot pick
    candidates, because it needs the evaluation it would be used to predict, but
    the yield it reaches is the best any selector on this frozen pool could
    reach, so it says what the pool *contains*. The sign flips because every
    selector in this module is oriented so that the lowest scores are taken.
    """

    values = np.asarray(evaluator, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("an oracle key needs a non-empty one-dimensional evaluator")
    return -values


def _interval(draws: Sequence[float]) -> list[float]:
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _limiting_factor(low: float, high: float) -> dict[str, str]:
    """Which of the two explanations the share interval is compatible with."""

    if low > GAP_SHARE_SPLIT:
        return {
            "code": "pool_content",
            "statement": (
                "the selector reaches more than half of the quality its pool holds, so "
                "what limits the yield at this budget is what generation produced, not "
                "the ranking"
            ),
        }
    if high < GAP_SHARE_SPLIT:
        return {
            "code": "selector_skill",
            "statement": (
                "the pool holds quality the selector reaches less than half of, so what "
                "limits the yield at this budget is the ranking, not what generation "
                "produced"
            ),
        }
    return {
        "code": "not_separable_at_this_pool_size",
        "statement": (
            "the share interval spans the half-way split, so pool content and selector "
            "skill are not separated at this budget on this pool. The size that would "
            "separate them is reported beside it"
        ),
    }


def gap_decomposition(
    evaluator: Sequence[float],
    selectors: Mapping[str, Sequence[float]],
    groups: Sequence[Any],
    *,
    fractions: Sequence[float] = SELECTION_FRACTIONS,
    seed: int,
    n_bootstrap: int = 10000,
    lengths: Sequence[int] | None = None,
    n_bins: int = LENGTH_MATCH_BINS,
    target_half_width: float = GAP_SHARE_TARGET_HALF_WIDTH,
) -> dict[str, Any]:
    """Split a selector's shortfall into what the pool lacks and what it misses.

    The question is which of two explanations accounts for a selector that barely
    beats chance: the pool holds too few good candidates, or the selector cannot
    find the good ones that are in it. At each budget *k* the decomposition reads

    * the **oracle** yield -- the mean over the best *k* members by the evaluator
      itself, which is what the pool contains;
    * the **selector** yield at the same *k*;
    * the **random** yield, taken as the pool mean, which is random selection's
      expectation at every budget.

    ``attainable_gap`` is oracle minus random: how much selectable quality is
    there at all. ``achieved_gap`` is selector minus random. Their ratio, the
    ``share_of_attainable_gap``, localises the failure: near one the pool is the
    limit, near zero the selector is, and an interval spanning the split means
    the pool is too small to tell -- in which case the number of independence
    units that would tell is reported.

    With ``lengths`` the whole decomposition is repeated against a quantile-binned
    length-matched draw, so the ceiling is not merely a length ceiling. Both the
    matched ceiling and the matched achieved gain come from the *same* bootstrap
    draws as the unmatched ones, which is what makes their ratio an interval
    rather than a quotient of two separately estimated numbers.

    The random reference is the pool mean rather than a realised random key,
    because random selection's expectation at any budget is exactly the pool mean
    and injecting one key's sampling noise into the denominator would widen every
    share interval for no gain in honesty. The spread of finite random draws is
    reported separately by :func:`random_baseline`.
    """

    values = np.asarray(evaluator, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("a gap decomposition needs a non-empty one-dimensional evaluator")
    if not np.isfinite(values).all():
        raise ValueError("a gap decomposition was given a non-finite evaluator value")
    if not selectors:
        raise ValueError("a gap decomposition needs at least one selector")
    keys: dict[str, np.ndarray] = {}
    for name, score in sorted(selectors.items()):
        array = np.asarray(score, dtype=np.float64)
        if array.shape != values.shape:
            raise ValueError(f"selector {name!r} does not align with the evaluator")
        if not np.isfinite(array).all():
            raise ValueError(f"selector {name!r} carries a non-finite score")
        keys[name] = array
    unit_ids = np.asarray([str(group) for group in groups])
    if unit_ids.shape != values.shape:
        raise ValueError("the independence units must align with the pool")
    grid = [float(fraction) for fraction in fractions]
    if not grid:
        raise ValueError("a gap decomposition needs at least one selection fraction")

    unique_units = np.unique(unit_ids)
    floor = bootstrap_unit_floor(int(unique_units.size))
    pool_mean = float(values.mean())
    oracle = oracle_key(values)
    record: dict[str, Any] = {
        "n_pool": int(values.size),
        "n_units": int(unique_units.size),
        "unit_floor": floor,
        "pool_mean": pool_mean,
        "random_reference": (
            "the pool mean, which is random selection's expectation at every budget"
        ),
        "oracle": (
            "the mean over the best k pool members by the evaluator itself. Circular by "
            "construction and unusable as a selector; it measures what the pool contains"
        ),
        "share_split": float(GAP_SHARE_SPLIT),
        "target_half_width": float(target_half_width),
        "length_matched": lengths is not None,
        "n_bootstrap": int(n_bootstrap),
        "resolved": not floor["degenerate"],
    }
    if floor["degenerate"]:
        record["reason"] = floor["degenerate_reason"]
        record["points"] = []
        return record

    binning = None if lengths is None else length_bins(lengths, n_bins=n_bins)
    bin_of_row = None if binning is None else binning["bin_of_row"]
    if binning is not None:
        record["length_bins"] = {
            key: binning[key] for key in ("edges", "n_bins_realised", "bin_counts")
        }
        record["max_forced_match_share"] = float(MAX_FORCED_MATCH_SHARE)

    # One generator for the whole decomposition, so the matched draw is
    # independent from one bootstrap iteration to the next. Re-seeding inside the
    # loop would freeze the comparator and leave every interval conditioned on a
    # single arbitrary matched draw.
    generator = np.random.default_rng(seed + 104729)
    index_of_unit = {str(unit): np.flatnonzero(unit_ids == unit) for unit in unique_units}

    attainable: dict[int, list[float]] = {index: [] for index in range(len(grid))}
    ceiling_matched: dict[int, list[float]] = {index: [] for index in range(len(grid))}
    achieved: dict[tuple[int, str], list[float]] = {}
    shortfall: dict[tuple[int, str], list[float]] = {}
    share: dict[tuple[int, str], list[float]] = {}
    achieved_matched: dict[tuple[int, str], list[float]] = {}
    share_matched: dict[tuple[int, str], list[float]] = {}
    for index in range(len(grid)):
        for name in keys:
            for store in (achieved, shortfall, share, achieved_matched, share_matched):
                store[(index, name)] = []

    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_units, size=unique_units.size, replace=True)
        rows = np.concatenate([index_of_unit[str(unit)] for unit in sampled])
        local = values[rows]
        local_mean = float(local.mean())
        local_bins = None if bin_of_row is None else bin_of_row[rows]
        # The orders do not depend on the budget, so each is built once per draw
        # and sliced at every fraction.
        orders = {name: np.argsort(key[rows], kind="stable") for name, key in keys.items()}
        oracle_order = np.argsort(-local, kind="stable")
        size = local.size
        for index, fraction in enumerate(grid):
            take = selection_count(size, fraction)
            oracle_chosen = oracle_order[:take]
            oracle_yield = float(local[oracle_chosen].mean())
            gap = oracle_yield - local_mean
            attainable[index].append(gap)
            matched_ceiling = float("nan")
            if local_bins is not None and take < size:
                oracle_matched = length_matched_mean(
                    local, local_bins, oracle_chosen, generator
                )
                if np.isfinite(oracle_matched):
                    matched_ceiling = oracle_yield - oracle_matched
                    ceiling_matched[index].append(matched_ceiling)
            for name, order in orders.items():
                chosen = order[:take]
                selected_yield = float(local[chosen].mean())
                achieved[(index, name)].append(selected_yield - local_mean)
                shortfall[(index, name)].append(oracle_yield - selected_yield)
                if gap > MINIMUM_ATTAINABLE_GAP:
                    share[(index, name)].append((selected_yield - local_mean) / gap)
                if local_bins is None or not np.isfinite(matched_ceiling):
                    continue
                selected_matched = length_matched_mean(local, local_bins, chosen, generator)
                if not np.isfinite(selected_matched):
                    continue
                matched_gain = selected_yield - selected_matched
                achieved_matched[(index, name)].append(matched_gain)
                if matched_ceiling > MINIMUM_ATTAINABLE_GAP:
                    share_matched[(index, name)].append(matched_gain / matched_ceiling)

    minimum_draws = int(np.ceil(MINIMUM_FINITE_DRAW_FRACTION * n_bootstrap))
    record["minimum_draws_for_an_interval"] = minimum_draws

    def share_block(draws: Sequence[float], point: float | None) -> dict[str, Any]:
        """A share with its interval, its verdict and the size that would resolve it."""

        if point is None or len(draws) < minimum_draws:
            return {
                "value": point,
                "ci95": None,
                "resolved": False,
                "n_draws": len(draws),
                "reason": (
                    "the attainable gap is not positive on enough bootstrap draws for a "
                    "share to be a quantity; the gaps themselves are reported instead"
                ),
            }
        low, high = _interval(draws)
        half_width = (high - low) / 2.0
        enough = half_width <= target_half_width
        units_needed = (
            int(record["n_units"])
            if enough
            else int(math.ceil(record["n_units"] * (half_width / target_half_width) ** 2))
        )
        return {
            "value": point,
            "ci95": [low, high],
            "resolved": True,
            "n_draws": len(draws),
            "half_width": half_width,
            "reaches_target_half_width": bool(enough),
            "limiting_factor": _limiting_factor(low, high),
            "units_for_target_half_width": units_needed,
            "units_scaling_assumption": (
                "the half-width is taken to fall as one over the square root of the "
                "number of independence units, which is the ordinary bootstrap rate. It "
                "is an extrapolation from this pool, not a measurement on a larger one"
            ),
        }

    points: list[dict[str, Any]] = []
    for index, fraction in enumerate(grid):
        take = selection_count(values.size, fraction)
        oracle_yield = selection_yield(values, oracle, fraction=fraction)
        gap = oracle_yield - pool_mean
        low, high = _interval(attainable[index])
        point: dict[str, Any] = {
            "fraction": fraction,
            "n_selected": take,
            "oracle_yield": oracle_yield,
            "random_yield": pool_mean,
            "attainable_gap": gap,
            "attainable_gap_ci95": [low, high],
            "pool_holds_selectable_quality": bool(low > 0.0),
            "identity_point": take == values.size,
            "selectors": {},
        }
        if take == values.size:
            point["note"] = (
                "at full selection every method, and the oracle, takes the whole pool. "
                "The attainable gap is zero by arithmetic and there is nothing to "
                "decompose; the point anchors the curve"
            )
        for name in keys:
            selected_yield = selection_yield(values, keys[name], fraction=fraction)
            block: dict[str, Any] = {
                "selected_yield": selected_yield,
                "achieved_gap": selected_yield - pool_mean,
                "achieved_gap_ci95": _interval(achieved[(index, name)]),
                "shortfall_vs_oracle": oracle_yield - selected_yield,
                "shortfall_vs_oracle_ci95": _interval(shortfall[(index, name)]),
                "share_of_attainable_gap": share_block(
                    share[(index, name)],
                    None if gap <= MINIMUM_ATTAINABLE_GAP else (selected_yield - pool_mean) / gap,
                ),
            }
            point["selectors"][name] = block
        if bin_of_row is not None:
            point["length_matched"] = _length_matched_point(
                values,
                keys,
                oracle,
                bin_of_row,
                fraction=fraction,
                generator=generator,
                ceiling_draws=ceiling_matched[index],
                achieved_draws={name: achieved_matched[(index, name)] for name in keys},
                share_draws={name: share_matched[(index, name)] for name in keys},
                share_block=share_block,
            )
        points.append(point)
    record["points"] = points
    record["reading_guide"] = [
        "attainable_gap answers 'does the pool hold anything selectable at this "
        "budget'. A gap whose interval contains zero means the question about the "
        "selector does not arise, because there is nothing to find",
        "share_of_attainable_gap answers 'pool content or selector skill'. Read its "
        "interval, not its point: a point near zero with an interval spanning the "
        "split does not localise the failure",
        "the oracle is the pool's realised best-k under this evaluator, not an "
        "estimate of what a larger pool would hold. It is a ceiling on this pool",
        "the length_matched block repeats the decomposition against a draw of the "
        "same size and length composition. A ceiling that survives it is a ceiling "
        "on protein quality; one that does not is a length ceiling",
        "a share computed where the attainable gap is near zero is a ratio of two "
        "small numbers and is reported as unresolved rather than as a large share",
    ]
    return record


def _length_matched_point(
    values: np.ndarray,
    keys: Mapping[str, np.ndarray],
    oracle: np.ndarray,
    bin_of_row: np.ndarray,
    *,
    fraction: float,
    generator: np.random.Generator,
    ceiling_draws: Sequence[float],
    achieved_draws: Mapping[str, Sequence[float]],
    share_draws: Mapping[str, Sequence[float]],
    share_block: Any,
) -> dict[str, Any]:
    """The same decomposition against a length-matched draw at one budget.

    Separated only to keep :func:`gap_decomposition` readable; it reports the
    full-sample points beside the intervals its caller accumulated, and refuses a
    verdict wherever length matching had no room to work.
    """

    take = selection_count(values.size, fraction)
    if take == values.size:
        return {
            "resolved": False,
            "reason": (
                "at full selection every method takes the whole pool, so a "
                "length-matched comparator has nothing else to draw from"
            ),
        }
    oracle_chosen = np.argsort(oracle, kind="stable")[:take]
    oracle_forced = forced_match_share(bin_of_row, oracle_chosen)
    if oracle_forced > MAX_FORCED_MATCH_SHARE:
        return {
            "resolved": False,
            "oracle_forced_match_share": oracle_forced,
            "reason": (
                f"{oracle_forced:.0%} of the oracle's selection comes from length bins "
                "holding fewer unselected rows than it took, above the "
                f"{MAX_FORCED_MATCH_SHARE:.0%} ceiling. The matched ceiling would be "
                "driven to zero by construction, so no length-matched decomposition is "
                "reported at this budget"
            ),
        }
    oracle_yield = float(values[oracle_chosen].mean())
    oracle_matched = length_matched_mean(values, bin_of_row, oracle_chosen, generator)
    ceiling = oracle_yield - oracle_matched
    block: dict[str, Any] = {
        "resolved": True,
        "n_selected": take,
        "oracle_forced_match_share": oracle_forced,
        "matched_ceiling": ceiling,
        "matched_ceiling_ci95": _interval(ceiling_draws) if ceiling_draws else None,
        "selectors": {},
        "interpretation": (
            "the oracle's gain over a draw of the same size and length composition is "
            "the quality the pool holds that length alone does not already deliver. "
            "Each selector's share of it is how much of that a selector reaches"
        ),
    }
    for name, key in sorted(keys.items()):
        chosen = np.argsort(key, kind="stable")[:take]
        forced = forced_match_share(bin_of_row, chosen)
        entry: dict[str, Any] = {"forced_match_share": forced}
        if forced > MAX_FORCED_MATCH_SHARE:
            entry["resolved"] = False
            entry["reason"] = (
                f"{forced:.0%} of this selector's selection comes from length bins it "
                "leaves too thin to match, above the "
                f"{MAX_FORCED_MATCH_SHARE:.0%} ceiling. A null here would be an artefact "
                "of the matching"
            )
            block["selectors"][name] = entry
            continue
        matched = length_matched_mean(values, bin_of_row, chosen, generator)
        gain = float(values[chosen].mean()) - matched
        entry["resolved"] = True
        entry["matched_gain"] = gain
        entry["matched_gain_ci95"] = (
            _interval(achieved_draws[name]) if achieved_draws[name] else None
        )
        entry["share_of_matched_ceiling"] = share_block(
            share_draws[name],
            None if ceiling <= MINIMUM_ATTAINABLE_GAP else gain / ceiling,
        )
        block["selectors"][name] = entry
    return block


def declared_independence(selectors: Sequence[str], evaluators: Sequence[str]) -> dict[str, Any]:
    """The statement that makes a selection result non-circular, written down.

    Stated rather than assumed. Each selector is named with what it reads, each
    evaluator with what it reads, and the one residual dependence -- both sides
    are functions of the same sequence -- is recorded instead of being claimed
    away as statistical independence.
    """

    return {
        "selectors": list(selectors),
        "evaluators": list(evaluators),
        "claim": (
            "no evaluator reads a selector score, a generating model's parameters or "
            "its likelihood; no selector reads a folded structure or a Pfam assignment"
        ),
        "residual_dependence": (
            "selector and evaluator are both functions of the same sequence, so they "
            "are not statistically independent and are not claimed to be. Independence "
            "here means the evaluator is a different instrument, not an uncorrelated one"
        ),
        "circularity_that_would_invalidate": (
            "selecting on a quantity and evaluating with that same quantity, or with "
            "one computed from it. The composition selector is the live case to watch: "
            "it is correlated with Pfam recognition through natural composition, so its "
            "structural result is the primary one"
        ),
    }


def require_fresh_out(out: Any, completion: str) -> None:
    """Accept an empty output directory, refuse one that already holds work.

    The campaign queue creates the output directory before it launches the cell,
    so refusing on mere existence loses every cell of the slot. What must be
    refused is evidence of *prior work*: a completion record, or any other
    content. A resumable instrument manages its own tree and does not come
    through here.
    """

    from pathlib import Path

    directory = Path(out)
    marker = directory / completion
    if marker.exists():
        raise SystemExit(
            f"{marker} already exists: this output directory holds a completed run. "
            "Refusing to overwrite it; point --out at a new directory"
        )
    if directory.exists():
        contents = sorted(item.name for item in directory.iterdir())
        if contents:
            raise SystemExit(
                f"{directory} is not empty (holds {contents[:8]}): refusing to write "
                "into a directory that already carries work"
            )
    directory.mkdir(parents=True, exist_ok=True)


def cohort_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Identity of a frozen pool: its ids and sequences, in order, and nothing else."""

    payload = "\n".join(f"{record['id']}\t{record['sequence']}" for record in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
