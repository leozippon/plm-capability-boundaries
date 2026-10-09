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
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..core.statistics import bootstrap_unit_floor, mean_interval, paired_group_bootstrap

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
    return profile


def diversity_reference(
    sequences: Sequence[str],
    family_sets: Sequence[Sequence[str]],
    *,
    fraction: float,
    seed: int,
    n_keys: int = DIVERSITY_REFERENCE_KEYS,
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
            [sequences[i] for i in chosen], [family_sets[i] for i in chosen]
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
