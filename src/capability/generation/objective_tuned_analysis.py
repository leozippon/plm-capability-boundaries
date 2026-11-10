"""Declarations and inference for the objective-tuned generation campaign.

The experiment is registered in ``docs/objective-alignment.md`` (declared
2026-11-09, amended 2026-11-10) and its generation half is wired in
``scripts/capability/generation/build_objective_tuned_campaign.py``. This module
holds only what the registration fixes: which pairs carry which reading, the
multiplicity families and their declared sizes, the margin rule, this lane's own
admissibility rule, the three explicit-statistics filters, and the resampling
that turns a pool contrast into a simultaneous band. Reading artefacts off disk
and assembling cells is the entry point's job
(``scripts/capability/generation/analyse_objective_tuned.py``); nothing here
touches the filesystem, loads a model or folds a sequence.

What this lane measures, and against what
-----------------------------------------
Objective attainment is mean per-residue ESMFold2 CA-pLDDT on the 0-100 scale
the structure instrument writes, read *within* an identity stratum. Distance
from the natural record is percent identity over the query against UniRef50,
from ``scripts/capability/evaluation/search_generated_novelty.py`` with masking
off -- not DIAMOND's ``pident``. The four strata are
:data:`~..context.homology.STRATUM_NAMES` with
:data:`~..context.homology.STRATUM_EDGES`, imported unchanged.

Why this lane declares its own admissibility
--------------------------------------------
``fold_recapitulation.IDENTITY_TO_WT_CEILING`` and
``fold_recapitulation.MIN_ADMISSIBLE_PER_CELL`` are the *recapitulation* lane's
rule: a candidate is admissible when it is far enough from a named wild type,
and a cell needs 25 such candidates. Neither transfers. There is no wild type
here -- novelty is measured against a natural-sequence database -- and identity
is the *stratifying variable*, so using it as an admission filter would delete
the axis the experiment is about. What this lane needs instead is a resampling
floor, and the project already has one measured for exactly that purpose:
:data:`~..core.statistics.MINIMUM_BOOTSTRAP_UNITS`. A pool below it is recorded
unresolved and never reported as an estimate.

Why the band is not a call to ``bidirectional.paired_bands``
------------------------------------------------------------
The construction below is the registration's own, by its amendments of
2026-11-10; it is not a local substitution for the declared estimator.
:func:`~..extensions.bidirectional.paired_bands` supplies the statistic, the
draws, the seed, the confidence and the draw machinery, all unchanged. It cannot
be *called* on this family. ``paired_bands`` wants a groups-by-columns matrix and resamples one
row index vector shared by every column, which is what makes its band
simultaneous; a row is a unit that carries a value in every column. This
family's unit is the generated sequence, the base and tuned pools hold different
sequences in a declared 4:1 ratio, and the three pairs are different models that
share no unit at all, so no such matrix exists. Writing one by pooling the two
arms into signed per-sequence contributions reproduces the point estimate and
inflates its variance by the squared pool means, which at pLDDT levels of 60-90
is an order of magnitude larger than the quantity being estimated.
:func:`contrast_bands` therefore forms the identical statistic -- the
``confidence`` quantile of the largest absolute centered deviation over the
family, in native units, with no standard error divided out -- from
:func:`~..extensions.bidirectional.resample_means` applied within stream, and
names the construction in its ``method`` rather than hiding it behind a borrowed
function name. The studentised family is retained only as a labelled
sensitivity, as that module requires. The one estimator that goes further is
:func:`reproduced_fraction_bands`, whose two base-pool means are nested: it
draws indices rather than means, because the filtered set is a function of the
draw, and forms the same statistic from them through
:func:`simultaneous_band`.

The resample is stratified by stream and conditional on the realised stratum
composition. Resampling within stream is what the registration declares; a
sequence belongs to exactly one identity stratum, so resampling inside a
stratum holds that stratum's count fixed. That is deliberate: the primary
contrast is a within-stratum comparison and the shift in the composition itself
is a separate declared endpoint, reported in its own family and never merged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..context.homology import STRATUM_EDGES, STRATUM_NAMES, assign_stratum
from ..core.statistics import MINIMUM_BOOTSTRAP_UNITS
from ..extensions import bidirectional as bd
from ..forcing.backbone_cohort import distinct_shingle_fraction
from . import generative_control as gc
from . import objective_tuned_arms as ot
from .structure_evidence import PREDICTED_CONFIDENCE_PLDDT

#: The stable identity of every record this module's readouts are written into.
SCHEMA = "objective_tuned_analysis"

#: The registration's identity axis, imported rather than restated.
STRATA: tuple[str, ...] = STRATUM_NAMES
#: The stratum a design-relevant bias has to appear in, by name rather than index.
NOVEL_STRATUM: str = STRATUM_NAMES[0]
#: The two strata a narrowing of memorisation would confine a gain to.
RECORD_STRATA: tuple[str, ...] = (STRATUM_NAMES[2], STRATUM_NAMES[3])
#: The identity below which a sequence counts as novel, from the frozen edges.
NOVEL_IDENTITY_EDGE: float = STRATUM_EDGES[1]


# ------------------------------------------------------------------ the pairs

def _pairs_in(lane: str, *, secondary: bool) -> tuple[ot.ObjectiveTunedPair, ...]:
    return tuple(
        pair for pair in ot.pairs_in_lane(lane) if bool(pair.secondary) is secondary
    )


#: The pairs that carry the matched structural contrast: same architecture,
#: tokenizer, prompt and decoding settings, preference optimisation as the only
#: difference. These and only these form the primary family.
PRIMARY_PAIRS: tuple[ot.ObjectiveTunedPair, ...] = _pairs_in(
    ot.LANE_FREE_LENGTH, secondary=False
)
#: The declared narrowing control, reported beside the primary family and never
#: merged into it, because both of its checkpoints carry thin training
#: documentation (``ObjectiveTunedPair.secondary``).
SECONDARY_PAIRS: tuple[ot.ObjectiveTunedPair, ...] = _pairs_in(
    ot.LANE_FREE_LENGTH, secondary=True
)
#: The pair the 2026-11-10 amendment reads within-arm only. Its prompt
#: difference is confounded with its objective difference, so it carries no
#: matched structural contrast at this lane's operating point at all.
WITHIN_ARM_PAIRS: tuple[ot.ObjectiveTunedPair, ...] = ot.pairs_in_lane(
    ot.LANE_EC_CONDITIONED
)


# ------------------------------------------------- declared families and sizes

#: The primary multiplicity family: the matched pairs crossed with the four
#: identity strata. Declared here as a number and asserted against the
#: enumeration below, so that a family size is never inferred from whatever
#: happened to reach the analysis at run time.
PRIMARY_FAMILY = "primary_plddt_by_stratum"
PRIMARY_FAMILY_SIZE = 12

#: The secondary pair's own family, read beside the primary one.
SECONDARY_FAMILY = "secondary_plddt_by_stratum"
SECONDARY_FAMILY_SIZE = 4

#: The identity-distribution shift, as two families of one column per pair. The
#: two endpoints are in different units -- percent identity and a fraction of
#: attempts -- and a maximum-statistic band in native units may not span them,
#: so they are never one family.
IDENTITY_MEAN_FAMILY = "identity_mean_over_query"
IDENTITY_FRACTION_FAMILY = "identity_fraction_below_30"
IDENTITY_FAMILY_SIZE = 3

#: Family recognition, one family per endpoint, both rates over the attempt
#: denominator.
RECOGNITION_ANY_FAMILY = "recognition_any_family"
RECOGNITION_COMPLETE_FAMILY = "recognition_complete_domain"
RECOGNITION_FAMILY_SIZE = 3

#: The three explicit statistical rules, declared by name so that the families
#: they carry can be sized and seeded before any rule is constructed.
FILTER_RULE_NAMES: tuple[str, ...] = (
    "low_complexity_residues", "composition_order0", "markov_order2",
)

#: One family per rule, each the matched pairs crossed with the four strata.
#: Three families, never merged with each other and never with the primary one:
#: the reproduced share of a gain is a different quantity from the gain, and the
#: three rules are three separate questions about the same base pool.
CONTROL_FAMILIES: dict[str, str] = {
    name: f"explicit_statistics_{name}" for name in FILTER_RULE_NAMES
}
CONTROL_FAMILY_SIZE = PRIMARY_FAMILY_SIZE

#: Largest share of draws a control column may lose to a resample in which one
#: of its three pools is empty. A filtered stratum subset at the admissibility
#: floor loses of order ``exp(-8)`` of its draws, so a percent is slack rather
#: than a target; above it the column carries no band and says so.
MAX_INVALID_DRAW_FRACTION = 0.01

#: One declared seed block per family, so no two families read the same draws
#: and every pool's seed is a declared function of its position.
FAMILY_SEED_OFFSETS: dict[str, int] = {
    PRIMARY_FAMILY: 0,
    SECONDARY_FAMILY: 100_000,
    IDENTITY_MEAN_FAMILY: 200_000,
    IDENTITY_FRACTION_FAMILY: 300_000,
    RECOGNITION_ANY_FAMILY: 400_000,
    RECOGNITION_COMPLETE_FAMILY: 500_000,
    **{CONTROL_FAMILIES[name]: 600_000 + 100_000 * index
       for index, name in enumerate(FILTER_RULE_NAMES)},
}

#: Declared size per family name, asserted against the realised column keys.
FAMILY_SIZES: dict[str, int] = {
    PRIMARY_FAMILY: PRIMARY_FAMILY_SIZE,
    SECONDARY_FAMILY: SECONDARY_FAMILY_SIZE,
    IDENTITY_MEAN_FAMILY: IDENTITY_FAMILY_SIZE,
    IDENTITY_FRACTION_FAMILY: IDENTITY_FAMILY_SIZE,
    RECOGNITION_ANY_FAMILY: RECOGNITION_FAMILY_SIZE,
    RECOGNITION_COMPLETE_FAMILY: RECOGNITION_FAMILY_SIZE,
    **{family: CONTROL_FAMILY_SIZE for family in CONTROL_FAMILIES.values()},
}


# ------------------------------------------------------------- the margin rule

#: One quarter of the distance from the base arm's own in-stratum mean to the
#: project's predicted-confidence criterion.
MARGIN_FRACTION = 0.25
#: The floor, in pLDDT points, so the margin cannot collapse where the base arm
#: is already at the criterion.
MARGIN_FLOOR = 2.0
#: The criterion the margin is scaled against, imported from the instrument that
#: defines the predicted-confidence event rather than restated as a literal.
MARGIN_CRITERION = PREDICTED_CONFIDENCE_PLDDT

MARGIN_RULE = (
    "max(MARGIN_FLOOR, MARGIN_FRACTION * max(0, criterion - base in-stratum mean)), "
    "in native pLDDT points, computed from the base pool alone. The margin is "
    "one-sided: it scales with the headroom the base arm still has to the "
    "criterion, and falls to the floor wherever the base arm already clears it. "
    "An unsigned reading of 'distance' would hand the laxest margin to the arm "
    "furthest above the criterion, which inverts what calibrating to headroom is "
    "for; the one-sided form is the registration's, by its second amendment of "
    "2026-11-10"
)


def margin_for(base_mean: float) -> dict[str, float | str]:
    """The declared margin for one cell, and the headroom it is scaled by."""

    value = float(base_mean)
    if not math.isfinite(value):
        raise ValueError("a margin needs a finite base in-stratum mean")
    headroom = max(0.0, MARGIN_CRITERION - value)
    return {
        "margin": max(MARGIN_FLOOR, MARGIN_FRACTION * headroom),
        "headroom_to_criterion": headroom,
        "base_in_stratum_mean": value,
        "criterion": float(MARGIN_CRITERION),
        "rule": MARGIN_RULE,
    }


# --------------------------------------------------------- the finding states

#: The four states this project keeps apart. A wide interval is never a negative
#: result, and an unmeasurable contrast is never a wide interval.
SUPPORTED = "supported"
BOUNDED_NEGATIVE = "bounded_negative"
UNRESOLVED = "unresolved"
NON_IDENTIFIABLE = "non_identifiable"
FINDING_STATES: tuple[str, ...] = (SUPPORTED, BOUNDED_NEGATIVE, UNRESOLVED, NON_IDENTIFIABLE)


def margin_finding(interval: Sequence[float], margin: float) -> dict[str, Any]:
    """The registration's decision rule for one cell, implemented as written.

    Exceeding the margin is a statement about the whole interval, not about the
    point estimate: a point above the margin with an interval reaching below it
    has not shown the effect at that size. An interval wholly inside the margin
    is a bounded negative at the stated size whichever sign it carries. Anything
    else -- including a resolved positive smaller than the margin -- is
    unresolved with respect to the margin, and is not evidence of absence.
    """

    low, high = (float(interval[0]), float(interval[1]))
    if not (math.isfinite(low) and math.isfinite(high)) or high < low:
        raise ValueError("a finding needs a finite ordered interval")
    size = float(margin)
    if low > size:
        return {"finding": SUPPORTED,
                "reason": f"the whole interval lies above the {size:.3f}-point margin"}
    if -size < low and high < size:
        return {"finding": BOUNDED_NEGATIVE,
                "reason": f"the whole interval lies inside +-{size:.3f} points"}
    return {"finding": UNRESOLVED,
            "reason": f"the interval spans the {size:.3f}-point margin; the margin is not resolved"}


# ----------------------------------------------------------- admissibility

#: This lane's admissibility rule. A pool is a resampling unit set, so the floor
#: is the project's measured percentile-bootstrap floor and nothing else: at
#: fewer units a nominal 95% interval delivers between half and six-sevenths of
#: what it advertises (see ``core.statistics.MINIMUM_BOOTSTRAP_UNITS``). The
#: floor applies per stream, because the resample is within stream.
MIN_UNITS_PER_POOL = MINIMUM_BOOTSTRAP_UNITS

ADMISSIBILITY_RULE = (
    f"every contributing (arm, stream) pool carries at least {MIN_UNITS_PER_POOL} "
    "units; a cell below the floor on any one of them is recorded unresolved with "
    "its counts and is never reported as an estimate. This is a resampling floor, "
    "not the recapitulation lane's identity-to-wild-type admissibility: there is "
    "no wild type here and identity is the stratifying variable"
)


def pool_units(streams: Mapping[int, Sequence[float]]) -> dict[str, int]:
    """Per-stream unit counts for one pool, as the record states them."""

    return {str(stream): int(len(streams[stream])) for stream in sorted(streams)}


def pool_shortfall(streams: Mapping[int, Sequence[float]]) -> list[str]:
    """Streams of one pool that sit below the declared floor, in order."""

    if not streams:
        return ["<no stream>"]
    return [str(stream) for stream in sorted(streams)
            if len(streams[stream]) < MIN_UNITS_PER_POOL]


# ------------------------------------------------------------------ resampling

def pool_seed(family: str, column: int, side: int, stream_index: int) -> int:
    """The declared draw seed of one pool: a function of its family position.

    One seed per (family, column, side, stream) keeps two pools from drawing the
    same index pattern whenever they happen to be the same size, and keeps the
    whole analysis reproducible from ``bidirectional.SEED`` alone.
    """

    if family not in FAMILY_SEED_OFFSETS:
        raise KeyError(f"no declared seed block for family {family!r}")
    if not 0 <= column < 100 or side not in (0, 1) or not 0 <= stream_index < 100:
        raise ValueError("a pool seed is declared for at most 100 columns, 2 sides and 100 streams")
    return bd.SEED + FAMILY_SEED_OFFSETS[family] + 1000 * column + 100 * side + stream_index


def pool_mean(streams: Mapping[int, Sequence[float]]) -> float:
    """The stream-equal mean of one pool.

    Stream-equal rather than sequence-equal because the streams are the declared
    replication axis and are equal-budget by construction, so a stream that
    retained fewer non-empty attempts must not thereby weigh less.
    """

    return float(np.mean([np.mean(np.asarray(streams[stream], dtype=float))
                          for stream in sorted(streams)]))


def pool_draws(
    streams: Mapping[int, Sequence[float]],
    *,
    family: str,
    column: int,
    side: int,
    draws: int = bd.DRAWS,
) -> np.ndarray:
    """``draws`` resampled stream-equal pool means, resampled within stream.

    The draw machinery is ``bidirectional.resample_means`` unchanged, called once
    per stream so the resample is stratified by stream as the registration
    declares.
    """

    per_stream = []
    for index, stream in enumerate(sorted(streams)):
        values = np.asarray(streams[stream], dtype=float).reshape(-1, 1)
        if not np.isfinite(values).all():
            raise ValueError(f"stream {stream} of a pool carries a non-finite value")
        seed = pool_seed(family, column, side, index)
        per_stream.append(bd.resample_means(values, draws=draws, seed=seed)[:, 0])
    return np.mean(np.stack(per_stream, axis=0), axis=0)


def require_declared_family(keys: Sequence[str], family: str) -> None:
    """Refuse a family whose realised columns differ from the declared ones.

    The size is declared in :data:`FAMILY_SIZES` and asserted here against the
    keys actually assembled. A campaign that is missing a pair, or that acquired
    one, changes the multiplicity of the whole family, and that is an amendment
    to the registration rather than something an analysis may absorb at run
    time.
    """

    declared = FAMILY_SIZES.get(family)
    if declared is None:
        raise KeyError(f"no declared size for family {family!r}")
    unique = list(dict.fromkeys(keys))
    if len(unique) != len(keys):
        raise ValueError(f"family {family} repeats a column key")
    if len(unique) != declared:
        raise ValueError(
            f"family {family} declares {declared} columns and realises {len(unique)}: "
            f"{unique}. Changing a declared family size is an amendment to "
            "docs/objective-alignment.md, not a run-time decision"
        )


NATIVE_BAND_METHOD = (
    bd.NATIVE_METHOD
    + "; two independent pools per column, resampled within stream, stream-equal "
    "pool means, conditional on the realised stratum composition"
)
STANDARDIZED_BAND_METHOD = (
    bd.STANDARDIZED_METHOD
    + "; the divisor is each column's own bootstrap standard error of the pool "
    "contrast, since a two-sample pool contrast has no exchangeable groups to "
    "take a fixed group standard error over. Sensitivity only"
)


def contrast_bands(
    columns: Sequence[Mapping[str, Any]],
    *,
    family: str,
    draws: int = bd.DRAWS,
    confidence: float = bd.CONFIDENCE,
) -> dict[str, Any]:
    """One simultaneous band per column over a declared family of pool contrasts.

    ``columns`` is the family in declared order; each entry carries ``key`` and
    the two pools ``base`` and ``tuned`` as ``{stream: values}``. A column whose
    pools do not all clear :data:`MIN_UNITS_PER_POOL` is recorded unresolved and
    contributes no draws, so the critical value covers the resolved subfamily;
    both sizes are recorded and the guarantee is stated for the subfamily rather
    than claimed for the declared family. A column may instead carry
    ``skip_reason``, which records it unresolved for a stated reason of the
    caller's own -- an incomplete search, say -- without pretending the reason
    was the resampling floor.
    """

    keys = [str(column["key"]) for column in columns]
    require_declared_family(keys, family)
    if not 0 < confidence < 1:
        raise ValueError("a simultaneous confidence level lies strictly between zero and one")

    entries: dict[str, dict[str, Any]] = {}
    resolved: list[str] = []
    deviations: list[np.ndarray] = []
    for index, column in enumerate(columns):
        key = keys[index]
        base, tuned = column["base"], column["tuned"]
        short = {"base": pool_shortfall(base), "tuned": pool_shortfall(tuned)}
        entry: dict[str, Any] = {
            "units": {"base": pool_units(base), "tuned": pool_units(tuned)},
            "minimum_units_per_pool": MIN_UNITS_PER_POOL,
        }
        skip = column.get("skip_reason")
        if skip:
            entry.update(status=UNRESOLVED, reason=str(skip), point=None, interval=None,
                         base_mean=None, tuned_mean=None)
            entries[key] = entry
            continue
        if short["base"] or short["tuned"]:
            entry.update(
                status=UNRESOLVED,
                reason="below the resampling floor on stream(s) "
                       + f"base={short['base']} tuned={short['tuned']}; "
                       + ADMISSIBILITY_RULE,
                point=None, interval=None, base_mean=None, tuned_mean=None,
            )
            entries[key] = entry
            continue
        base_mean, tuned_mean = pool_mean(base), pool_mean(tuned)
        point = tuned_mean - base_mean
        draw = (pool_draws(tuned, family=family, column=index, side=1, draws=draws)
                - pool_draws(base, family=family, column=index, side=0, draws=draws))
        entry.update(status="resolved", reason=None, point=point,
                     base_mean=base_mean, tuned_mean=tuned_mean)
        entries[key] = entry
        resolved.append(key)
        deviations.append(draw - point)

    record: dict[str, Any] = {
        "family": family,
        "method": NATIVE_BAND_METHOD,
        "multiplicity_family": "the declared pairs crossed with the declared columns of this family",
        "draws": int(draws),
        "seed": bd.SEED,
        "seed_rule": "bidirectional.SEED + FAMILY_SEED_OFFSETS[family] + 1000*column + 100*side + stream",
        "confidence": float(confidence),
        "declared_family_size": FAMILY_SIZES[family],
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


def simultaneous_band(entries: dict[str, dict[str, Any]], resolved: Sequence[str],
                      deviations: Sequence[np.ndarray], *,
                      confidence: float = bd.CONFIDENCE) -> dict[str, Any]:
    """The family's critical value and every resolved column's interval.

    One definition, shared by every estimator in this module, so that a band on a
    pool contrast and a band on a nested ratio cannot drift apart. ``deviations``
    holds one centered draw vector per resolved column, all over the same draws;
    the critical value is the ``confidence`` quantile of the largest absolute
    deviation over the family in a draw, in the columns' own native units, with
    no standard error divided out. ``entries`` is updated in place with each
    column's interval, its zero verdict and the studentised sensitivity.
    """

    matrix = np.stack(deviations, axis=1)
    if matrix.shape[1] != len(resolved):
        raise ValueError("a band needs one deviation vector per resolved column")
    critical = float(np.quantile(np.max(np.abs(matrix), axis=1), confidence))
    errors = matrix.std(axis=0, ddof=1)
    varying = errors > 0
    standardized_critical = (
        float(np.quantile(np.max(np.abs(matrix[:, varying] / errors[varying]), axis=1), confidence))
        if varying.any() else None
    )
    for position, key in enumerate(resolved):
        point = entries[key]["point"]
        entries[key]["interval"] = [point - critical, point + critical]
        error = float(errors[position])
        entries[key]["standardized_sensitivity"] = {
            "standard_error": error,
            "interval": None if standardized_critical is None or error <= 0 else
            [point - standardized_critical * error, point + standardized_critical * error],
        }
    intervals = np.asarray([entries[key]["interval"] for key in resolved], dtype=float)
    zero = bd.verdict(intervals)
    for key, value in zip(resolved, zero):
        entries[key]["zero_verdict"] = value
    return {"critical_value": critical,
            "draws_in_the_statistic": int(matrix.shape[0]),
            "zero_verdict_counts": bd.counts(zero),
            "standardized": {"method": STANDARDIZED_BAND_METHOD,
                             "critical_value": standardized_critical,
                             "role": "sensitivity; never the primary reading"}}


# -------------------------------------------- the explicit-statistics control

#: The fraction of the base pool an explicit rule retains. The base pools are
#: generated at four times the tuned budget so that this quarter matches the
#: tuned pool: the alignment chose its positives from 80,000 of its own samples,
#: so a statistical rule must be allowed the same selection pressure.
TOP_FRACTION = 0.25

#: The share of a Pfam profile's model length a domain must cover to count as
#: complete, imported from the gate that declared it.
COMPLETE_DOMAIN_COVERAGE = gc.COMPLETE_DOMAIN_COVERAGE


@dataclass(frozen=True)
class FilterRule:
    """One explicit statistical rule that ranks a base pool, with its provenance.

    ``statistic`` reads one sequence and returns a score, or ``None`` where the
    rule cannot be evaluated on it. ``retain`` names the end of the ranking the
    rule keeps, so a reader does not have to infer a direction from a sign.
    """

    name: str
    retain: str
    units: str
    provenance: str
    statistic: Callable[[str], float | None]

    def __post_init__(self) -> None:
        if self.retain not in ("highest", "lowest"):
            raise ValueError(f"{self.name}: retain is 'highest' or 'lowest'")


def _corpus_scorer(order: int, table: np.ndarray) -> Callable[[str], float | None]:
    def score(sequence: str) -> float | None:
        if len(sequence) <= order:
            return None
        value, scored = gc.conditional_log_likelihood([sequence], table, order)
        if scored == 0 or not math.isfinite(value):
            return None
        return float(value)

    return score


def _low_complexity(sequence: str) -> float | None:
    if not sequence:
        return None
    return float(distinct_shingle_fraction(sequence))


LOW_COMPLEXITY_PROVENANCE = (
    "forcing.backbone_cohort.distinct_shingle_fraction at "
    "near_duplicates.RESIDUE_SHINGLE = 5: the repository's existing repeat and "
    "low-complexity screen on a generated sequence. The registration describes "
    "this rule as 'the fraction of low-complexity residues' and says all three "
    "rules already exist in the generative-control module; that module holds the "
    "two corpus rules and no residue-level low-complexity mask, and this lane "
    "needs a ranking rather than the alignment's 25% threshold, so the existing "
    "distinct-shingle statistic is used and named for what it is instead of a "
    "SEG residue mask being introduced here"
)


def filter_rules(counts: Mapping[int, np.ndarray]) -> tuple[FilterRule, ...]:
    """The three declared rules, in the registration's order of richness.

    ``counts`` are corpus k-mer counts from
    ``generative_control.load_kmer_counts``; order 0 reads ``k = 1`` and order 2
    reads ``k = 3``. The evaluation tables are the smoothed ones that module
    declares, because a generated sequence may carry a k-mer the corpus never
    contained and minus infinity on one fragment must not decide a ranking.
    """

    zero = gc.conditional_table(counts, 0)
    second = gc.conditional_table(counts, 2)
    rules = (
        FilterRule(
            name="low_complexity_residues",
            retain="highest",
            units="distinct 5-mer fraction",
            provenance=LOW_COMPLEXITY_PROVENANCE,
            statistic=_low_complexity,
        ),
        FilterRule(
            name="composition_order0",
            retain="highest",
            units="nats per residue",
            provenance="generative_control.conditional_table(counts, 0) over the staged "
                       "UniRef50 background; a zero-order amino-acid composition model",
            statistic=_corpus_scorer(0, zero),
        ),
        FilterRule(
            name="markov_order2",
            retain="highest",
            units="nats per residue",
            provenance="generative_control.conditional_table(counts, 2) over the same "
                       "background; a second-order Markov model",
            statistic=_corpus_scorer(2, second),
        ),
    )
    if tuple(rule.name for rule in rules) != FILTER_RULE_NAMES:
        raise RuntimeError("the realised filter rules are not the declared FILTER_RULE_NAMES, "
                           "so their families are not the ones sized and seeded above")
    return rules


def score_pool(records: Sequence[Mapping[str, Any]], rule: FilterRule) -> dict[str, Any]:
    """Score one pool under one rule, keeping the members the rule can rank.

    Each record carries ``key``, a unique handle for one *attempt*, and
    ``sequence``. The rule is a function of the sequence, so two attempts of the
    same sequence score alike; the unit stays the attempt. A record the rule
    cannot score is dropped from the pool and counted, never scored as zero, and
    the pool the control reads is the scorable one throughout -- the realised
    retention and the resample must rank the same members or the retained count
    would not be a quarter of anything.
    """

    scored, unscorable = [], []
    for record in records:
        value = rule.statistic(str(record["sequence"]))
        if value is None:
            unscorable.append(str(record["key"]))
        else:
            scored.append({**record, "score": float(value)})
    return {"scored": scored, "n_unscorable": len(unscorable),
            "unscorable_keys": sorted(unscorable)[:20]}


def _by_score(scored: Sequence[Mapping[str, Any]], retain: str) -> list[int]:
    """Positions of ``scored``, best first, ties broken on the attempt key."""

    sign = -1.0 if retain == "highest" else 1.0
    return sorted(range(len(scored)),
                  key=lambda index: (sign * scored[index]["score"], str(scored[index]["key"])))


def rank_and_retain(
    scored: Sequence[Mapping[str, Any]],
    rule: FilterRule,
    *,
    fraction: float = TOP_FRACTION,
) -> dict[str, Any]:
    """Retain the top ``fraction`` of an already-scored pool, best first."""

    order = _by_score(scored, rule.retain)
    retain = int(len(order) * float(fraction))
    return {
        "rule": rule.name,
        "retain": rule.retain,
        "units": rule.units,
        "provenance": rule.provenance,
        "fraction": float(fraction),
        "n_scored": len(order),
        "n_retained": retain,
        "retained_keys": frozenset(str(scored[index]["key"]) for index in order[:retain]),
    }


NESTED_DRAW_METHOD = (
    "one index draw per (rule, pair, stream) serves both the base pool mean and "
    "the mean of the pool's own top fraction: the rule is re-applied to each "
    "resample, so the filtered set is a function of the draw and the two means "
    "are nested rather than independent. Resampling is within (stream, stratum), "
    "as the primary contrast is, so a draw carries the realised stratum "
    "composition; the top fraction is then taken globally across strata, which "
    "is how the rule is allowed to select"
)


def nested_base_draws(
    scored: Sequence[Mapping[str, Any]],
    *,
    retain: str,
    family: str,
    column: int,
    stream_index: int,
    draws: int = bd.DRAWS,
    fraction: float = TOP_FRACTION,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-stratum draw vectors of a base pool mean and of its filtered mean.

    Returns ``{stratum: {'base': (draws,), 'filtered': (draws,), 'valid': (draws,)}}``.
    The resample is within (stream, stratum), so a stratum present in the pool
    has a base mean on every draw; the top fraction is then taken globally across
    strata, and ``valid`` says whether that fraction held any member of the
    stratum. It can fail at a rate of order ``exp(-f)`` in the realised filtered
    count ``f``, ``filtered`` carries ``nan`` there, and a draw counts only where
    ``valid`` holds -- the numerator needs both means from the one draw, so an
    unusable filtered mean discards the base mean with it. The caller applies
    :data:`MAX_INVALID_DRAW_FRACTION`. A stratum with no member at all is invalid
    on every draw and both its vectors are ``nan``.
    """

    order = _by_score(scored, retain)
    size = len(order)
    retained = int(size * float(fraction))
    if size < MIN_UNITS_PER_POOL or retained < 1:
        raise ValueError(f"a nested resample needs at least {MIN_UNITS_PER_POOL} scorable "
                         f"members and a non-empty top fraction; got {size} and {retained}")
    plddt = np.asarray([float(scored[index]["plddt"]) for index in order], dtype=float)
    codes = {name: position for position, name in enumerate(STRATA)}
    bands = np.asarray([codes[str(scored[index]["stratum"])] for index in order], dtype=np.int64)
    if not np.isfinite(plddt).all():
        raise ValueError("a nested resample needs a finite value on every member")
    ranks = {name: np.flatnonzero(bands == codes[name]) for name in STRATA}
    present = [name for name in STRATA if ranks[name].size]
    out = {name: {"base": np.full(draws, np.nan), "filtered": np.full(draws, np.nan),
                  "valid": np.zeros(draws, dtype=bool)} for name in STRATA}
    rng = np.random.default_rng(pool_seed(family, column, 0, stream_index))
    for start in range(0, draws, bd.DRAW_CHUNK):
        stop = min(start + bd.DRAW_CHUNK, draws)
        chunk = stop - start
        blocks = []
        for name in present:
            members = ranks[name]
            drawn = members[rng.integers(0, members.size, size=(chunk, members.size))]
            out[name]["base"][start:stop] = plddt[drawn].mean(axis=1)
            blocks.append(drawn)
        pool = np.concatenate(blocks, axis=1)
        top = np.sort(pool, axis=1)[:, :retained]
        values, where = plddt[top], bands[top]
        for name in present:
            mask = where == codes[name]
            count = mask.sum(axis=1)
            total = np.where(mask, values, 0.0).sum(axis=1)
            ok = count > 0
            out[name]["filtered"][start:stop] = np.where(ok, total / np.maximum(count, 1), np.nan)
            out[name]["valid"][start:stop] = ok
    return out


def reproduced_fraction(base_mean: float, tuned_mean: float, filtered_mean: float) -> dict[str, Any]:
    """The share of the tuned-minus-base gain an explicit filter reproduces.

    Undefined, and recorded as such, when the alignment's own gain is not
    positive: dividing by a non-positive denominator would turn "the alignment
    did nothing here" into a fraction with a sign. This is the disposition
    ``analyse_e20_confirmatory.captured_fraction`` already uses for a
    non-positive oracle gain.
    """

    gain = float(tuned_mean) - float(base_mean)
    reproduced = float(filtered_mean) - float(base_mean)
    if gain <= 0.0:
        return {"tuned_minus_base_gain": gain, "filter_minus_base_gain": reproduced,
                "reproduced_fraction": None,
                "reason": "the tuned-minus-base gain is not positive, so the share of it "
                          "an explicit rule reproduces is undefined"}
    return {"tuned_minus_base_gain": gain, "filter_minus_base_gain": reproduced,
            "reproduced_fraction": reproduced / gain, "reason": None}


#: The share at which an explicit statistical filter has done everything the
#: alignment did on this endpoint, in the registration's words.
FULL_REPRODUCTION = 1.0

#: Whether the primary endpoint leaves a gain for a reproduced share to be a
#: share of. Three states, because two of them are opposite readings of the
#: experiment and the third is neither.
GAIN_IDENTIFIED = "identified"
GAIN_ABSENT = "absent"
GAIN_UNMEASURED = "unmeasured"
GAIN_STATES: tuple[str, ...] = (GAIN_IDENTIFIED, GAIN_ABSENT, GAIN_UNMEASURED)

CONTROL_MOOT_NOTE = (
    "the control is moot, not merely unresolved: the primary endpoint shows no "
    "in-stratum gain bounded away from zero anywhere in this family, so there is "
    "nothing for an explicit statistical rule to recover and the reproduced share "
    "is undefined rather than imprecise. The reading rests on the primary "
    "endpoint -- preference optimisation produced no in-stratum structural "
    "improvement to explain -- which is a more direct statement of the "
    "programme's claim than any reproduced fraction would be. This is not a "
    "licence to widen what counts as a gain in order to recover a ratio"
)


def primary_gain_disposition(cell: Mapping[str, Any] | None) -> dict[str, Any]:
    """Whether the primary endpoint leaves a gain for an explicit rule to recover.

    Three outcomes, kept apart because the first two mean opposite things about
    the experiment and a reader who sees one ``unresolved`` for both cannot tell
    which. A gain *bounded away from zero* leaves a share to estimate. A gain the
    primary endpoint resolves and finds not bounded away from zero leaves none, so
    the share is not a quantity at all. A cell the primary endpoint does not
    resolve leaves the question open, which is neither of those.
    """

    if cell is None:
        return {"gain": GAIN_UNMEASURED,
                "reason": "no primary-endpoint column was supplied for this cell, so whether "
                          "there is a gain to recover is unknown here"}
    interval = cell.get("interval")
    if cell.get("status") != "resolved" or interval is None:
        return {"gain": GAIN_UNMEASURED,
                "reason": "the primary endpoint does not resolve this cell, so whether there is "
                          f"a gain to recover is not measured: {cell.get('reason')}"}
    low, high = float(interval[0]), float(interval[1])
    if cell.get("zero_verdict") != "positive":
        return {"gain": GAIN_ABSENT, "primary_point": cell.get("point"),
                "primary_interval": [low, high],
                "reason": "the primary endpoint shows no in-stratum gain bounded away from zero "
                          f"({cell.get('point'):+.3f} pLDDT points, interval [{low:+.3f}, "
                          f"{high:+.3f}]), so there is no gain for an explicit rule to recover "
                          "and the share of it is not a quantity. " + CONTROL_MOOT_NOTE}
    return {"gain": GAIN_IDENTIFIED, "primary_point": cell.get("point"),
            "primary_interval": [low, high], "reason": None}


def reproduction_finding(interval: Sequence[float]) -> dict[str, Any]:
    """Whether a banded share reaches full reproduction, in this project's states.

    The claim under test is that an explicit statistical rule reproduces the
    alignment's gain. A band wholly at or above one supports it: statistics did
    everything the alignment did on this endpoint. A band wholly below one bounds
    it away at the width shown, which is what leaves a remainder for the
    alignment to have contributed. A band spanning one is unresolved, and that is
    the reading a point estimate cannot give at all.
    """

    low, high = float(interval[0]), float(interval[1])
    if low > FULL_REPRODUCTION:
        return {"finding": SUPPORTED,
                "reason": "the whole band lies above one: an explicit rule reproduces the gain"}
    if high < FULL_REPRODUCTION:
        return {"finding": BOUNDED_NEGATIVE,
                "reason": f"the whole band lies below one, bounding the reproduced share away "
                          f"from full reproduction by {FULL_REPRODUCTION - high:.3f}"}
    return {"finding": UNRESOLVED,
            "reason": "the band spans one, so reproducing most of the gain and reproducing all of "
                      "it are not distinguished"}


CONTROL_INTERVAL_NOTE = (
    "banded, one declared family per rule, because reproducing most of a gain "
    "and reproducing all of it are different readings and a point estimate "
    "cannot tell them apart. The filtered pool is a subset of the pool it is "
    "differenced against, so the band rests on a nested resample -- one base "
    "index draw serves both means and the rule is re-applied to each draw -- "
    "which is why the project's pooled-mean helper cannot serve"
)


def reproduced_fraction_bands(
    columns: Sequence[Mapping[str, Any]],
    *,
    family: str,
    draws: int = bd.DRAWS,
    confidence: float = bd.CONFIDENCE,
) -> dict[str, Any]:
    """Simultaneous bands on the reproduced share of the gain, one family per rule.

    Each column carries the nested base draws (``base``, ``filtered``, ``valid``
    per stream), the tuned pool (``tuned``, per stream, resampled independently
    because it is a different model's pool), the realised means, and optionally a
    ``skip_reason``. A column is estimable only where the realised gain is
    positive, where the invalid-draw share is within
    :data:`MAX_INVALID_DRAW_FRACTION`, and where **every** retained resample
    still shows a gain: the share of a gain is not a quantity when the resampled
    gain can be zero, and a ratio band taken across a denominator that changes
    sign reports a width rather than an uncertainty. The family's statistic is
    taken over the draws every resolved column retained, so the max statistic is
    read on one common draw set.
    """

    keys = [str(column["key"]) for column in columns]
    require_declared_family(keys, family)
    entries: dict[str, dict[str, Any]] = {}
    resolved: list[str] = []
    fractions: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for index, column in enumerate(columns):
        key = keys[index]
        realised = {name: column.get(name) for name in
                    ("base_realised", "filtered_realised", "tuned_realised")}
        entry: dict[str, Any] = {"units": column.get("units"), "realised": realised,
                                 "minimum_units_per_pool": MIN_UNITS_PER_POOL}
        gain_state = primary_gain_disposition(column.get("primary"))
        entry["primary_gain"] = gain_state
        if gain_state["gain"] == GAIN_ABSENT:
            entries[key] = {**entry, "status": NON_IDENTIFIABLE, "reason": gain_state["reason"],
                            "point": None, "interval": None}
            continue
        if gain_state["gain"] == GAIN_UNMEASURED:
            entries[key] = {**entry, "status": UNRESOLVED, "reason": gain_state["reason"],
                            "point": None, "interval": None}
            continue
        skip = column.get("skip_reason")
        if skip:
            entries[key] = {**entry, "status": UNRESOLVED, "reason": str(skip),
                            "point": None, "interval": None}
            continue
        share = reproduced_fraction(float(realised["base_realised"]),
                                    float(realised["tuned_realised"]),
                                    float(realised["filtered_realised"]))
        entry["gains"] = {name: share[name] for name in
                          ("tuned_minus_base_gain", "filter_minus_base_gain")}
        point = share["reproduced_fraction"]
        if point is None:
            entries[key] = {**entry, "status": UNRESOLVED, "point": None, "interval": None,
                            "reason": share["reason"]}
            continue
        valid = np.ones(draws, dtype=bool)
        base = np.zeros(draws)
        filtered = np.zeros(draws)
        streams = sorted(column["base"])
        for position, stream in enumerate(streams):
            valid &= np.asarray(column["valid"][stream], dtype=bool)
            base += np.nan_to_num(np.asarray(column["base"][stream], dtype=float))
            filtered += np.nan_to_num(np.asarray(column["filtered"][stream], dtype=float))
        base /= len(streams)
        filtered /= len(streams)
        tuned = pool_draws(column["tuned"], family=family, column=index, side=1, draws=draws)
        invalid = 1.0 - float(valid.mean())
        if invalid > MAX_INVALID_DRAW_FRACTION:
            entries[key] = {**entry, "status": UNRESOLVED, "point": point, "interval": None,
                            "invalid_draw_fraction": invalid,
                            "reason": f"{invalid:.4f} of draws produced a filtered set with no "
                                      f"member of this stratum, above the declared "
                                      f"{MAX_INVALID_DRAW_FRACTION:g}"}
            continue
        denominator = (tuned - base)[valid]
        if denominator.size == 0 or float(denominator.min()) <= 0.0:
            entries[key] = {**entry, "status": UNRESOLVED, "point": point, "interval": None,
                            "invalid_draw_fraction": invalid,
                            "reason": "not every resample shows a tuned-minus-base gain, so the "
                                      "share of that gain has no band: the denominator reaches "
                                      "zero and the ratio is unbounded there"}
            continue
        entry.update(status="resolved", reason=None, point=point,
                     invalid_draw_fraction=invalid, nested_draws=NESTED_DRAW_METHOD)
        entries[key] = entry
        resolved.append(key)
        fractions[key] = np.where(valid, (filtered - base) / np.where(valid, tuned - base, 1.0),
                                  np.nan)
        masks[key] = valid

    record: dict[str, Any] = {
        "family": family,
        "method": NATIVE_BAND_METHOD,
        "multiplicity_family": "the declared matched pairs crossed with the four identity strata, "
                               "for this rule alone; never merged with another rule or with the "
                               "primary family",
        "draws": int(draws), "seed": bd.SEED, "confidence": float(confidence),
        "declared_family_size": FAMILY_SIZES[family],
        "resolved_family_size": len(resolved),
        "nested_draws": NESTED_DRAW_METHOD,
        "units": "dimensionless share of the tuned-minus-base pLDDT gain",
        "admissibility": ADMISSIBILITY_RULE,
        "columns": entries,
    }
    states = [entries[key]["primary_gain"]["gain"] for key in keys
              if "primary_gain" in entries[key]]
    record["primary_gain_counts"] = {state: states.count(state) for state in GAIN_STATES}
    moot = bool(states) and all(state == GAIN_ABSENT for state in states)
    record["control_status"] = "moot" if moot else "readable"
    if moot:
        record["moot_note"] = CONTROL_MOOT_NOTE
    if not resolved:
        record.update(critical_value=None, standardized=None,
                      reason=CONTROL_MOOT_NOTE if moot else
                      "no column of this family carries an estimable share")
        return record
    common = np.ones(draws, dtype=bool)
    for key in resolved:
        common &= masks[key]
    discarded = 1.0 - float(common.mean())
    record["common_draw_discard_fraction"] = discarded
    if discarded > MAX_INVALID_DRAW_FRACTION:
        record.update(critical_value=None, standardized=None,
                      reason=f"{discarded:.4f} of draws are unusable in at least one resolved "
                             f"column, above the declared {MAX_INVALID_DRAW_FRACTION:g}; the "
                             "family's max statistic has no common draw set")
        for key in resolved:
            entries[key].update(status=UNRESOLVED, interval=None,
                                reason=record["reason"])
        record["resolved_family_size"] = 0
        return record
    record.update(simultaneous_band(
        entries, resolved, [fractions[key][common] - entries[key]["point"] for key in resolved],
        confidence=confidence))
    for key in resolved:
        entries[key]["statistics_reproduce_the_gain"] = reproduction_finding(
            entries[key]["interval"])
    record["full_reproduction"] = FULL_REPRODUCTION
    record["reproduction_counts"] = {
        state: sum(1 for key in resolved
                   if entries[key]["statistics_reproduce_the_gain"]["finding"] == state)
        for state in (SUPPORTED, BOUNDED_NEGATIVE, UNRESOLVED)}
    return record


# ---------------------------------------------------- composition decomposition

def shift_share(
    base: Mapping[str, Mapping[str, Any]],
    tuned: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Split a pooled pLDDT gain into a within-stratum term and a composition term.

    ``base`` and ``tuned`` carry ``share`` and ``mean`` per stratum. The
    registration's second route to a narrowing finding is that "the identity
    distribution itself shifts toward the record by enough to account for the
    pooled gain", and this is that arithmetic: the pooled gain is the
    share-weighted mean difference, the within-stratum term holds the base
    composition fixed, and the composition term is what the move between strata
    contributes.

    Descriptive only. It combines pLDDT levels across strata, which the
    registration's own bounds forbid for a verdict, because ESMFold's confidence
    falls with novelty and that is the stratifying variable. It is reported so a
    reader can see the size of the composition term, never as a finding.
    """

    strata = [name for name in STRATA if name in base and name in tuned]
    missing = [name for name in STRATA if name not in strata]
    if not strata:
        raise ValueError("a shift-share split needs at least one stratum on both sides")
    pooled = sum(tuned[name]["share"] * tuned[name]["mean"] for name in strata) - sum(
        base[name]["share"] * base[name]["mean"] for name in strata)
    within = sum(base[name]["share"] * (tuned[name]["mean"] - base[name]["mean"])
                 for name in strata)
    composition = sum((tuned[name]["share"] - base[name]["share"]) * tuned[name]["mean"]
                      for name in strata)
    return {
        "strata": strata,
        "strata_without_both_sides": missing,
        "pooled_gain": float(pooled),
        "within_stratum_term": float(within),
        "composition_term": float(composition),
        "identity": "pooled_gain = within_stratum_term + composition_term",
        "descriptive_only": (
            "combines pLDDT levels across strata, which this registration's bounds "
            "forbid for a verdict; reported as a size, never as a finding"
        ),
    }


#: The share of the pooled gain the composition term must exceed for the
#: registration's second narrowing route, by its second amendment of 2026-11-10.
NARROWING_COMPOSITION_SHARE = 0.5

NARROWING_COMPOSITION_RULE = (
    "the composition term exceeds half the pooled gain and the identity-shift "
    "band for the pair excludes zero in the direction of the record -- mean "
    "identity over the query rising, or the fraction below 30% falling. Both "
    "halves are needed: a composition term can be large on a shift the band "
    "cannot resolve, and a resolved shift can be too small to account for the "
    "gain"
)


def narrowing_by_composition(split: Mapping[str, Any],
                             moves: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The registration's second narrowing route: the shift accounts for the gain.

    ``split`` is a :func:`shift_share` record for the pair and ``moves`` carries
    that pair's column from each identity-shift family, by endpoint name. The
    route is unresolved rather than negative wherever the pooled gain is not
    positive: there is then no gain for a composition shift to account for, and
    saying so is not evidence against narrowing.
    """

    pooled = split.get("pooled_gain")
    composition = split.get("composition_term")
    toward: dict[str, Any] = {}
    for name, column in moves.items():
        interval = column.get("interval")
        if column.get("status") != "resolved" or interval is None:
            toward[name] = {"excludes_zero_toward_the_record": False,
                            "reason": column.get("reason") or "unresolved"}
            continue
        low, high = float(interval[0]), float(interval[1])
        record_side = low > 0.0 if name == IDENTITY_MEAN_FAMILY else high < 0.0
        toward[name] = {"excludes_zero_toward_the_record": bool(record_side),
                        "point": column.get("point"), "interval": interval}
    moved = any(entry["excludes_zero_toward_the_record"] for entry in toward.values())
    if pooled is None or composition is None or pooled <= 0.0:
        return {"finding": UNRESOLVED, "rule": NARROWING_COMPOSITION_RULE,
                "identity_moves": toward,
                "reason": "the pooled gain is not positive, so no composition shift has a gain "
                          "to account for"}
    accounts = composition > NARROWING_COMPOSITION_SHARE * pooled
    if accounts and moved:
        return {"finding": SUPPORTED, "rule": NARROWING_COMPOSITION_RULE,
                "identity_moves": toward, "share_of_pooled_gain": composition / pooled,
                "reason": "the composition term exceeds half the pooled gain and the identity "
                          "distribution moved toward the record with its band excluding zero"}
    return {"finding": UNRESOLVED, "rule": NARROWING_COMPOSITION_RULE,
            "identity_moves": toward, "share_of_pooled_gain": composition / pooled,
            "reason": ("the composition term does not exceed half the pooled gain"
                       if not accounts else
                       "no identity-shift band excludes zero toward the record")}


# ------------------------------------------------- selector competence, oracle

#: The declared selection budget for the selector-competence readout.
SELECTION_BUDGET = 10

ORACLE_STATEMENT = (
    "the oracle ranks the pool by the realised endpoint, which no selector may "
    "see. It measures headroom, is never a selector, and its gain is the "
    "denominator of the captured fraction"
)


def oracle_headroom(values: Sequence[float], *, k: int = SELECTION_BUDGET) -> float | None:
    """Top-``k`` mean minus pool mean on the endpoint itself, in native units.

    The same construction as ``analyse_e20_confirmatory.tm_oracle_gain``, read on
    mean CA-pLDDT instead of TM-score. ``None`` where the pool is smaller than
    the budget, because a top-10 of eight sequences is the pool.
    """

    array = np.asarray(values, dtype=float)
    if array.size < k:
        return None
    ordered = np.sort(array)[::-1]
    return float(ordered[:k].mean() - array.mean())


def captured_fraction(
    values: Sequence[float],
    scores: Sequence[float],
    *,
    k: int = SELECTION_BUDGET,
) -> dict[str, Any]:
    """The share of the oracle's pLDDT headroom a selector captures within one pool.

    ``analyse_e20_confirmatory.captured_fraction``'s definition on this lane's
    endpoint: the selector's top-``k`` gain over the pool mean, divided by the
    oracle's. ``None`` when the oracle gain is not positive, as that module does.
    """

    endpoint = np.asarray(values, dtype=float)
    selector = np.asarray(scores, dtype=float)
    if endpoint.shape != selector.shape:
        raise ValueError("a selector score is needed for every pool member")
    oracle = oracle_headroom(endpoint, k=k)
    if oracle is None:
        return {"k": k, "n_pool": int(endpoint.size), "oracle_gain": None,
                "selector_gain": None, "captured_fraction": None,
                "reason": f"the pool holds {endpoint.size} members, below the budget of {k}",
                "oracle_statement": ORACLE_STATEMENT}
    top = np.argsort(-selector, kind="stable")[:k]
    gain = float(endpoint[top].mean() - endpoint.mean())
    if oracle <= 0.0:
        return {"k": k, "n_pool": int(endpoint.size), "oracle_gain": oracle,
                "selector_gain": gain, "captured_fraction": None,
                "reason": "the oracle gain is not positive",
                "oracle_statement": ORACLE_STATEMENT}
    return {"k": k, "n_pool": int(endpoint.size), "oracle_gain": oracle,
            "selector_gain": gain, "captured_fraction": gain / oracle, "reason": None,
            "oracle_statement": ORACLE_STATEMENT}


# ------------------------------------------------------- declared limitations

#: The size asymmetry the within-arm reading has to be read against. Recorded
#: rather than corrected: the floor's two hundred samples are its own design,
#: and a conditioned arm's three thousand are not comparable to them.
WITHIN_ARM_POPULATION_NOTE = (
    "the two arms' populations are not of comparable size. The EC-conditioned arm "
    "carries thousands of samples per condition; the unconditioned ProtGPT2 floor "
    "carries two hundred, which is thin, and its quantiles and stratum shares rest "
    "on that. A within-arm distribution is read per condition and the two arms are "
    "never pooled, so the asymmetry does not enter any contrast -- it bounds how "
    "finely the floor's own distribution can be read"
)

#: Why the ZymCTRL pair carries no matched structural contrast at this lane's
#: operating point, in the amendment's own terms.
ZYMCTRL_NON_IDENTIFIABLE = (
    "ZymCTRL's native input is an EC class tag and ProtGPT2 has no channel to "
    "receive one, so whichever prompt the pair shares puts one arm out of "
    "distribution. The prompt difference is confounded with the objective "
    "difference and no tuned-minus-base pLDDT contrast for this pair is "
    "identifiable; the pair is read within-arm only (amendment of 2026-11-10)"
)

#: Why the prediction side-effect cannot be read from this campaign's artefacts.
LADDER_NOT_MEASURABLE = (
    "the registration's prediction side-effect scores the tuned arms on the R1 "
    "statistical ladder against Profile, Potts and GEMME, but every arm declared "
    "in objective_tuned_arms.ARMS carries an empty 'capabilities' set and no "
    "scoring_target_alphabet_size, which is that module's explicit refusal of a "
    "scoring-lane admission. No artefact of this campaign can carry the endpoint "
    "and none is fabricated here; admitting these checkpoints to a scoring lane "
    "is an amendment to the declarations, not an analysis step"
)

BOUNDS: tuple[str, ...] = (
    "every structural endpoint is predicted; no generated sequence here has a "
    "measured folding, stability or functional property",
    "for the ProtGPT3 pairs the objective axis shares its instrument with the "
    "training signal, so a gain on it is partly agreement with the optimiser's "
    "own scorer; the identity axis carries the interpretation",
    "ESMFold confidence falls as sequences become novel, which is the stratifying "
    "variable: levels of pLDDT are never compared across strata, and a low value "
    "in the novel stratum cannot separate instrument limitation from design failure",
    "identity to a UniRef50 snapshot is similarity to a named snapshot, not "
    "pretraining exposure; the ProtGPT3 mixture of UniRef90 and GigaRef is not "
    "covered by the searched snapshot, so the measured identity is biased and the "
    "bias is named rather than corrected",
    "the resampling unit is the generated sequence at fixed weights: this is "
    "decoder sampling noise, not weight-level uncertainty and not biological "
    "replication",
)


def _check_declarations() -> None:
    """Refuse at import if the declarations this module asserts have drifted."""

    if tuple(STRATA) != tuple(STRATUM_NAMES) or len(STRATA) != 4:
        raise RuntimeError("the identity strata are not the four frozen homology strata")
    if len(PRIMARY_PAIRS) * len(STRATA) != PRIMARY_FAMILY_SIZE:
        raise RuntimeError(
            f"the primary family declares {PRIMARY_FAMILY_SIZE} columns but "
            f"{len(PRIMARY_PAIRS)} matched pairs crossed with {len(STRATA)} strata "
            f"give {len(PRIMARY_PAIRS) * len(STRATA)}"
        )
    if len(SECONDARY_PAIRS) * len(STRATA) != SECONDARY_FAMILY_SIZE:
        raise RuntimeError("the secondary family size does not match the declared pairs")
    if not len(PRIMARY_PAIRS) == IDENTITY_FAMILY_SIZE == RECOGNITION_FAMILY_SIZE:
        raise RuntimeError("the secondary endpoint families are one column per matched pair")
    if not WITHIN_ARM_PAIRS:
        raise RuntimeError("the amendment's within-arm pair is not declared")
    if assign_stratum(NOVEL_IDENTITY_EDGE - 1e-9) != NOVEL_STRATUM:
        raise RuntimeError("the novel stratum does not end at the declared identity edge")
    if MARGIN_CRITERION != 70.0:
        raise RuntimeError("the predicted-confidence criterion is no longer 70 pLDDT points")
    if sorted(FAMILY_SIZES) != sorted(FAMILY_SEED_OFFSETS):
        raise RuntimeError("a declared family is sized without a seed block, or the reverse")
    if len(set(FAMILY_SEED_OFFSETS.values())) != len(FAMILY_SEED_OFFSETS):
        raise RuntimeError("two families share a seed block and would read the same draws")
    if len(CONTROL_FAMILIES) != len(FILTER_RULE_NAMES) or len(FILTER_RULE_NAMES) != 3:
        raise RuntimeError("the registration declares three explicit statistical rules, one "
                           "family each")
    if any(FAMILY_SIZES[family] != PRIMARY_FAMILY_SIZE for family in CONTROL_FAMILIES.values()):
        raise RuntimeError("a control family is not the matched pairs crossed with the strata")
    if not 0.0 < NARROWING_COMPOSITION_SHARE < 1.0:
        raise RuntimeError("the narrowing composition share is a fraction of the pooled gain")
    if len(set(GAIN_STATES)) != 3 or NON_IDENTIFIABLE not in FINDING_STATES:
        raise RuntimeError("the control's gain dispositions no longer map onto the four states")


_check_declarations()
