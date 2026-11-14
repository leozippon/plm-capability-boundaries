"""Pair-level bidirectional ablation against the explicit evolutionary statistics.

The declaration is ``docs/explicit-pair-ablation.md``, dated 2026-11-14 and fixed
before any direction accuracy below was computed. Every constant here is
transcribed from it and none of it is tunable at runtime.

What this adds to E21
=====================

E21's matched same-substitution pair cohort controls the independent-site
Profile and a position-only rule. It controls neither of the two higher-order
explicit statistics the project fits, the Potts full-sequence energy difference
and the official GEMME score, so the thesis clause it carries -- that the
models' evolutionary memory holds genuine function-discriminative information --
is established against the weakest of the project's three explicit-statistics
controls, while the overlap clause is established against all three.

This module builds the pair-level instance of the bidirectional framework
``extensions/bidirectional.py`` declares for regression increments. The forward
direction asks what an arm orders correctly when the three explicit statistics
cannot order the pair; the reverse direction asks what the three statistics
order correctly when the arm's own likelihood cannot. No new model inference is
needed: the R1 statistical ladder holds all three statistics per variant on the
frozen 25,728-variant R1 common support, and the retained 33-arm panel holds
each arm's native likelihood on that same support.

What is reused rather than rebuilt
==================================

The cohort rule, the phenotype gap, the greedy disjointness rule, the tie
convention, the direction definition and the verdict rule come from
:mod:`..mutation.matched_discrimination`. Baseline admission and the strict
``sample_id`` join come from :mod:`..extensions.statistical_ladder`. The draws,
the simultaneous band and the resampling helper come from
:mod:`..extensions.bidirectional`. Only three things are new: the covariate
scale shared by all four quantities, the tolerance rule, and the exact k-vote
balance that generalises the repository's one- and two-vote rules.

The one scale
=============

All four quantities -- Profile, Potts, GEMME and one arm's native likelihood --
are read on their within-assay standardized rank over the assay's 128 retained
support rows, the panel's own ``raw_M``/``raw_P`` scale, whose standard
deviation is exactly 1. That is the scale on which the statistical ladder's
controls actually enter (``statistical_ladder.RERANKED``), the only scale on
which the retained panel stores a likelihood at all, and the scale that lets one
declared tolerance govern the forward and the reverse direction alike.

Two devices, each with the other's blind spot
=============================================

Matching equates the magnitude of a difference and leaves its sign free; exact
sign balance cancels a statistic's vote over the whole tier and leaves
individual magnitudes large. E21's own record is the evidence that neither alone
suffices: its magnitude-matched tier failed its construction check with the
Profile at 0.5500, which is why amendment 3 added exact balance and amendment 4
made the conjunction of the two the decisive tier. Both devices are built here
for the three statistics, and the conjunction tier is the decisive one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..core.statistics import bootstrap_unit_floor
from ..extensions import bidirectional, statistical_ladder
from . import matched_discrimination as md

#: The declaration this module executes, by path and id.
DECLARATION = "docs/explicit-pair-ablation.md"
DECLARATION_ID = "explicit_pair_ablation_20261114"
ANALYSIS_ID = "explicit_pair_ablation_20261114_S1_retained_panel"
SCHEMA = "explicit_pair_ablation"
SUPPORT = "S1_retained_panel"

#: The three explicit statistics, in report order, and the controlled model
#: quantity of the reverse direction.
STATISTICS: tuple[str, ...] = ("profile", "potts", "gemme")
MODEL = "model"

#: Where each statistic is read from. ``profile`` is the panel's own column, so
#: the reading cannot substitute a different profile channel for E21's.
SOURCE_COLUMN = {
    "profile": "profile_score",
    "potts": statistical_ladder.POTTS_COLUMN,
    "gemme": statistical_ladder.GEMME_COLUMN,
}
#: The two statistics that arrive as admitted upstream baseline documents.
BASELINE_STATISTICS: tuple[str, ...] = ("potts", "gemme")

#: Declared tolerance in within-assay standard deviations of a quantity's rank.
PRIMARY_TOLERANCE = 0.20
#: Declared sensitivity ladder: a geometric halving and doubling either side.
TOLERANCE_LADDER: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.80)

#: Expected standardized-rank separation of two variants drawn at random from
#: one assay, ``E|Z1 - Z2| = 2*sqrt(3)/3`` for a standardized uniform rank. The
#: tolerance is declared as a fraction of this and of one standard deviation.
RANDOM_PAIR_SEPARATION = 2.0 * np.sqrt(3.0) / 3.0

#: Inference, from the shared bidirectional family. No second bootstrap exists.
BOOTSTRAP_DRAWS = bidirectional.DRAWS
BOOTSTRAP_SEED = 20261114
BALANCE_SEED = 20261114
CONFIDENCE = bidirectional.CONFIDENCE

#: The canonical model-blind pair order every seeded rule traverses, identical
#: to the one :func:`matched_discrimination.greedy_disjoint` uses.
CANONICAL_ORDER = ("assay", "wildtype_residue", "mutant_residue", "position_a", "position_b")

#: Forward tiers that carry no tolerance, and the votes each one cancels.
BALANCED_FORWARD_TIERS = {"T_bal_P": ("profile",), "T_bal_PEG": STATISTICS}
#: Forward tiers at each declared tolerance, and the quantities each one matches.
MATCHED_FORWARD_TIERS = {"T_tol_P": ("profile",), "T_tol_PEG": STATISTICS}
#: The decisive forward tier: exact balance imposed inside the matched tier.
CONJUNCTION_FORWARD_TIER = "T_tol_bal_PEG"
#: The reverse direction's tiers, all arm-specific because the model is what
#: they control.
REVERSE_BALANCED_TIER = "T_bal_M"
REVERSE_MATCHED_TIER = "T_tol_M"
REVERSE_CONJUNCTION_TIER = "T_tol_bal_M"

TIER_RULES = {
    "T_open": "the pair rule and the phenotype gap alone; the reference tier",
    "T_tol_P": "Profile matched at the declared tolerance",
    "T_tol_PEG": "Profile, Potts and GEMME each matched at the declared tolerance",
    "T_bal_P": "the Profile's vote cancelled exactly",
    "T_bal_PEG": "the Profile's, Potts's and GEMME's votes each cancelled exactly",
    "T_tol_bal_PEG": "all three votes cancelled exactly inside T_tol_PEG; the decisive forward tier",
    "T_tol_M": "the arm's own likelihood rank matched at the declared tolerance",
    "T_bal_M": "the arm's own likelihood vote cancelled exactly",
    "T_tol_bal_M": "the arm's own vote cancelled exactly inside T_tol_M",
}

#: The reading each forward outcome licenses, spelled once so the artefact and
#: the report cannot disagree about it.
READINGS = {
    "forward_supported": (
        "pairwise directional ability attributable to none of Profile, Potts or GEMME at the "
        "matched resolution, by construction rather than by adjustment; the thesis's third clause "
        "holds in its strong form and is sharpened"
    ),
    "forward_bounded_negative": (
        "no such ability beyond the declared 0.05 margin; the functional information the project "
        "attributes to the memory is information the explicit statistics also carry, and the clause "
        "must be restated as a claim about shared rather than additional information"
    ),
    "forward_unresolved": (
        "neither reading; a statement about this cohort's resolution and not about the effect"
    ),
    "reverse_supported": (
        "the statistic orders pairs the arm's likelihood cannot, so the overlap is incomplete in "
        "that direction too and the memory is not a superset of the explicit statistics"
    ),
    "reverse_bounded_negative": (
        "the statistic's unique pair-level contribution is bounded at the declared 0.05 margin"
    ),
    "conjunction": (
        "an arm supported under matching alone may be reading the residual sign the matching leaves "
        "free, and an arm supported under balance alone may be reading the residual magnitude the "
        "balance leaves large; agreement across T_tol_PEG, T_bal_PEG and T_tol_bal_PEG is the "
        "reading that survives both blind spots, and a disagreement is reported as such"
    ),
}

LIMITATIONS = (
    "The tiers are adversarial by construction: they retain pairs on which the neutralised "
    "statistics cannot discriminate but the phenotype plainly does, so an accuracy here is not an "
    "estimate of how well a rule orders a randomly drawn variant pair.",
    "A sign balance is marginal over the tier, not within each cluster or assay, and a tolerance "
    "match equates a magnitude and not an ordering, so a resolved arm is not proof that the three "
    "statistics are inert everywhere.",
    "Only direction is readable on this support. The retained panel stores a within-assay rank, so "
    "no difference in nats and no effect size in likelihood units is computable or reported.",
    "Potts and GEMME are single assay-reranked scalars on the same aligned homolog support as the "
    "Profile. Three explicit statistics are not all evolutionary statistics, and the instrument "
    "limits of docs/statistical-baseline-ladder.md travel with every number here.",
    "The ProteinGym export carries no measurement replicates, so a one-standard-deviation gap is an "
    "observed gap and some pairs may be ordered by noise.",
    "This cohort has been studied before, so this is a pre-specified reanalysis and not external "
    "validation. An interval covering 0.5 is neither equivalence nor the absence of an effect.",
)


def tier_label(name: str, tolerance: float | None = None) -> str:
    """The artefact's name for one tier, with its tolerance when it has one."""

    if name not in TIER_RULES:
        raise ValueError(f"{name} is not a declared tier")
    return name if tolerance is None else f"{name}({tolerance:g})"


# --------------------------------------------------------------------------- #
# the statistics: admission, strict join and the one shared scale
# --------------------------------------------------------------------------- #


def read_baseline(name: str, path) -> dict[str, Any]:
    """One upstream baseline document, admitted only as a complete official run.

    Admission is :func:`statistical_ladder.require_complete_baseline` and nothing
    weaker: declared counts, the complete flag, every row's status and GEMME's
    three-run aggregate are checked, so a finite score from an unconverged Potts
    background or a partly scored wild type is refused rather than used.
    """

    if name not in BASELINE_STATISTICS:
        raise ValueError(f"{name} is not one of the admitted baselines {BASELINE_STATISTICS}")
    table = statistical_ladder.read_score_table(path, metadata_required=True)
    admission = statistical_ladder.require_complete_baseline(name, table)
    return {"table": table, "admission": admission}


def coverage(name: str, support: pd.DataFrame, table: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse a baseline table that does not cover the support exactly.

    Enumerated by wild-type reference, by count and by the row's upstream status,
    the way :func:`explicit_statistics_lanes.complete_case` enumerates a loss: a
    lane that silently loses rows is a defect, and a lane that loses them for a
    whole reference is a finding that has to be nameable.
    """

    identifiers = [str(value) for value in support["sample_id"]]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name}: the support carries duplicate sample_id values")
    offered = [str(value) for value in table["sample_id"]]
    if len(set(offered)) != len(offered):
        raise ValueError(f"{name}: {table['path']} carries duplicate sample_id values")
    reference_of = dict(zip(support["sample_id"], support["protein_id"], strict=True))
    unseen = [identifier for identifier in identifiers if identifier not in set(offered)]
    extra = sorted(set(offered) - set(identifiers))
    if unseen or extra:
        references: dict[str, int] = {}
        for identifier in unseen:
            key = str(reference_of[identifier])
            references[key] = references.get(key, 0) + 1
        raise ValueError(
            f"{name}: {table['path']} does not cover the support: {len(unseen)} rows missing over "
            f"{len(references)} references {sorted(references.items())[:3]}, {len(extra)} extra "
            "sample_id values; nothing is imputed and no row is dropped"
        )
    return {
        "name": name,
        "read_from": table["path"],
        "sha256": table["sha256"],
        "rows_offered": len(offered),
        "rows_retained": len(identifiers),
        "rows_lost": 0,
        "references_lost": [],
    }


#: The upstream status word for a block that produced numbers but no estimate.
#: ``docs/statistical-baseline-ladder.md`` declares it for a GEMME wild type whose
#: every mutation scores 0.0 in all three official runs -- JET2 located no site,
#: so the tool gave no estimate -- and records that the published R1 table was
#: left as it is, with its ``wildtypes_degenerate`` count naming how many.
DEGENERATE_STATUS = "refused_degenerate"


def degenerate_blocks(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Assays whose block of one statistic is constant, hence carries no estimate.

    A column with no within-assay spread cannot order any pair of that assay: its
    rank is a column of zeros, every pair is an exact tie, and a tier that claims
    to control it there controls nothing. That is the ladder's own
    ``refused_degenerate`` condition, so the block is refused here rather than
    silently reduced to a vacuous control.
    """

    refusals: dict[str, dict[str, Any]] = {}
    for name in STATISTICS:
        spread = frame.groupby("unit_id")[name].transform(lambda block: block.std(ddof=0))
        dead = frame.loc[spread.to_numpy(dtype=np.float64) == 0.0]
        if dead.empty:
            continue
        references: dict[str, Any] = {}
        for reference, block in dead.groupby("protein_id"):
            references[str(reference)] = {
                "status": DEGENERATE_STATUS,
                "assays": sorted(str(value) for value in block["unit_id"].unique()),
                "rows": int(len(block)),
                "constant_value": float(block[name].iloc[0]),
                "reason": (
                    f"{name} is constant over every retained row of these assays, so it carries "
                    "no estimate and cannot order any pair of them"
                ),
            }
        refusals[name] = {
            "statistic": name,
            "references": references,
            "reference_count": len(references),
            "assays": int(dead["unit_id"].nunique()),
            "rows": int(len(dead)),
        }
    return refusals


def attach_statistics(
    support: pd.DataFrame, tables: Mapping[str, Mapping[str, Any]]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The support with the three statistics and their one shared rank scale.

    The join is :func:`statistical_ladder.join_scores`, which refuses duplicate
    keys, missing or extra rows and non-finite values rather than imputing any of
    them. Blocks that carry no estimate are refused by :func:`degenerate_blocks`,
    enumerated by wild-type reference the way
    ``explicit_statistics_lanes.complete_case`` enumerates a loss, and removed
    before the scale is built; every control set is then read on the rows where
    all three statistics are available. The rank is
    :func:`matched_discrimination.standardized_rank` within assay over the
    assay's own retained support rows, which is the scale the panel's ``raw_M``
    and ``raw_P`` columns already carry. A degenerate block surviving into the
    scale raises rather than becoming a column of zeros.
    """

    missing = [name for name in BASELINE_STATISTICS if name not in tables]
    if missing:
        raise ValueError(f"the reading needs the {missing} baseline tables")
    frame = support.copy()
    covered: dict[str, Any] = {}
    for name in BASELINE_STATISTICS:
        table = tables[name]
        covered[name] = coverage(name, frame, table)
        column = SOURCE_COLUMN[name]
        joined = statistical_ladder.join_scores(
            frame["sample_id"].tolist(), table, columns=[column]
        )
        frame[name] = joined[column]
    frame["profile"] = frame[SOURCE_COLUMN["profile"]].to_numpy(dtype=np.float64)
    for name in STATISTICS:
        if not np.isfinite(frame[name].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{name} is not finite on every support row")
    refusals = degenerate_blocks(frame)
    refused_assays = sorted({
        assay
        for record in refusals.values()
        for detail in record["references"].values()
        for assay in detail["assays"]
    })
    offered = len(frame)
    frame = frame.loc[~frame["unit_id"].astype(str).isin(refused_assays)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("every assay carries a refused statistic block; there is no support left")
    remaining = degenerate_blocks(frame)
    if remaining:
        raise ValueError(
            f"a refused block survived the restriction: {sorted(remaining)}; the rank scale would "
            "be a column of zeros and the control would be vacuous"
        )
    for name in STATISTICS:
        frame[f"rank_{name}"] = frame.groupby("unit_id")[name].transform(
            lambda block: md.standardized_rank(block.to_numpy())
        )
    record = {
        "rule": (
            "one support for every tier: the rows where all three explicit statistics carry an "
            "estimate. A block with no within-assay spread is refused by the statistical ladder's "
            "own refused_degenerate condition, enumerated by reference, and removed whole-assay"
        ),
        "coverage": covered,
        "rows_offered": int(offered),
        "rows_retained": int(len(frame)),
        "rows_lost": int(offered - len(frame)),
        "assays_refused": refused_assays,
        "references_refused": sorted({
            reference
            for record_ in refusals.values()
            for reference in record_["references"]
        }),
        "by_statistic": refusals,
    }
    return frame, record


def rank_lookup(frame: pd.DataFrame, name: str) -> dict[str, float]:
    """``sample_id -> within-assay standardized rank`` for one quantity."""

    column = f"rank_{name}"
    if column not in frame.columns:
        raise ValueError(f"{name} has no attached rank column")
    return dict(zip(frame["sample_id"], frame[column].to_numpy(dtype=np.float64), strict=True))


def rank_distribution(frame: pd.DataFrame, name: str) -> dict[str, float]:
    """The achieved spread of one quantity's rank, so the scale is auditable."""

    values = frame[f"rank_{name}"].to_numpy(dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


# --------------------------------------------------------------------------- #
# pairs: orientation, covariates, the two devices
# --------------------------------------------------------------------------- #


def orient(pairs: pd.DataFrame) -> pd.DataFrame:
    """Name the better- and worse-phenotype member of every pair.

    Exactly :func:`matched_discrimination.build_cohort`'s orientation: the
    higher-z member is the better-phenotype member, and a pair with no gap is
    refused rather than oriented arbitrarily.
    """

    frame = pairs.copy()
    left = frame["z_a"].to_numpy(dtype=np.float64)
    right = frame["z_b"].to_numpy(dtype=np.float64)
    if (left == right).any():
        raise ValueError("a candidate pair has no phenotype gap and cannot be oriented")
    better_is_a = left > right
    frame["better_sample"] = np.where(better_is_a, frame["sample_a"], frame["sample_b"])
    frame["worse_sample"] = np.where(better_is_a, frame["sample_b"], frame["sample_a"])
    return frame


def attach_covariate(pairs: pd.DataFrame, name: str, ranks: Mapping[str, float]) -> pd.DataFrame:
    """Add one quantity's within-pair rank distance, direction and signed vote.

    The direction is :func:`matched_discrimination.direction_values`, so the tie
    convention has one source, and the vote is derived from it rather than
    recomputed: a quantity that cannot separate the two members casts no vote.
    """

    frame = pairs.copy()
    values = md.direction_values(frame, ranks)
    better = frame["better_sample"].map(ranks).to_numpy(dtype=np.float64)
    worse = frame["worse_sample"].map(ranks).to_numpy(dtype=np.float64)
    frame[f"{name}_distance"] = np.abs(better - worse)
    frame[f"{name}_direction"] = values
    frame[f"{name}_vote"] = np.where(values > 0.5, 1, np.where(values < 0.5, -1, 0)).astype(np.int8)
    return frame


def distance_summary(pairs: pd.DataFrame, names: Sequence[str]) -> dict[str, Any]:
    """The achieved rank distances of a pair set, against the random separation."""

    record: dict[str, Any] = {
        "n_pairs": int(len(pairs)),
        "random_pair_separation": float(RANDOM_PAIR_SEPARATION),
    }
    for name in names:
        values = pairs[f"{name}_distance"].to_numpy(dtype=np.float64)
        record[name] = (
            None
            if values.size == 0
            else {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "max": float(values.max()),
                "share_of_random_separation_at_median": float(
                    np.median(values) / RANDOM_PAIR_SEPARATION
                ),
            }
        )
    return record


def tolerance_matched(
    candidates: pd.DataFrame, names: Sequence[str], tolerance: float
) -> pd.DataFrame:
    """The pairs whose every named quantity is matched within ``tolerance``.

    Every declared tolerance is honoured for every named quantity, so a pair that
    violates one of them is excluded whatever the others do.
    """

    if not names:
        raise ValueError("a tolerance match needs at least one quantity")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("a matching tolerance is finite and strictly positive")
    mask = np.ones(len(candidates), dtype=bool)
    for name in names:
        column = f"{name}_distance"
        if column not in candidates.columns:
            raise ValueError(f"{name} has no attached rank distance; the tolerance is unenforceable")
        mask &= candidates[column].to_numpy(dtype=np.float64) <= tolerance
    return candidates.loc[mask].reset_index(drop=True)


def balance_votes(
    pairs: pd.DataFrame, names: Sequence[str], *, seed: int = BALANCE_SEED
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cancel every named quantity's directional vote exactly, for any ``k``.

    One construction generalising the repository's single-vote and two-vote
    rules. Pairs neutral on *every* balanced vote contribute exactly 0.5 to each
    of them under the declared tie convention and are retained in full; pairs
    neutral on some but not all cast an incomplete vote and are excluded; the
    rest are grouped by their sign vector and each cell is thinned against its
    complement to the smaller of the two counts, by a seeded permutation of the
    canonical model-blind pair order. Taking equal counts from complementary
    cells balances every marginal exactly whatever the split between cells, which
    retains more pairs than levelling all cells to the smallest would.
    """

    if not names:
        raise ValueError("an exact balance needs at least one vote")
    if len(pairs) == 0:
        raise ValueError("an empty pair set cannot be balanced")
    ordered = pairs.sort_values(list(CANONICAL_ORDER), kind="mergesort").reset_index(drop=True)
    votes = np.stack(
        [ordered[f"{name}_vote"].to_numpy(dtype=np.int64) for name in names], axis=1
    )
    neutral = votes == 0
    all_neutral = neutral.all(axis=1)
    partial = neutral.any(axis=1) & ~all_neutral
    cells: dict[tuple[int, ...], list[int]] = {}
    for position in range(len(ordered)):
        if all_neutral[position] or partial[position]:
            continue
        cells.setdefault(tuple(int(value) for value in votes[position]), []).append(position)
    rng = np.random.default_rng(seed)
    keep: list[int] = [int(position) for position in np.flatnonzero(all_neutral)]
    diagonals: list[dict[str, Any]] = []
    handled: set[tuple[int, ...]] = set()
    for signs in sorted(cells):
        if signs in handled:
            continue
        mirror = tuple(-value for value in signs)
        handled.update((signs, mirror))
        size = min(len(cells.get(signs, [])), len(cells.get(mirror, [])))
        diagonals.append(
            {
                "cell": _signature(names, signs),
                "mirror": _signature(names, mirror),
                "available": [len(cells.get(signs, [])), len(cells.get(mirror, []))],
                "retained_per_side": int(size),
            }
        )
        for key in (signs, mirror):
            rows = cells.get(key, [])
            order = rng.permutation(len(rows))
            keep.extend(rows[int(index)] for index in sorted(order[:size]))
    retained = ordered.iloc[sorted(keep)].reset_index(drop=True)
    record = {
        "balanced_votes": list(names),
        "candidates": int(len(ordered)),
        "all_neutral_retained": int(all_neutral.sum()),
        "partially_neutral_excluded": int(partial.sum()),
        "sign_cells_available": {
            _signature(names, signs): len(rows) for signs, rows in sorted(cells.items())
        },
        "diagonals": diagonals,
        "retained_pairs": int(len(retained)),
        "seed": int(seed),
        "rule": (
            "complementary sign cells thinned to equal counts; pairs neutral on every balanced "
            "vote retained in full, pairs neutral on some but not all excluded"
        ),
    }
    if len(retained) == 0:
        # No sign cell has a non-empty complement, so the votes cannot be
        # cancelled on this input at all. That is a fact about the tier and is
        # reported with its cell counts; it is not an exactly balanced tier of
        # zero pairs and must not be certified as one.
        record["status"] = "no_pairs_after_balance"
        record["certified_direction_accuracy"] = None
        return retained, record
    record["status"] = "balanced"
    record["certified_direction_accuracy"] = certify_balance(retained, names)
    return retained, record


def _signature(names: Sequence[str], signs: Sequence[int]) -> str:
    return ",".join(f"{name}{'+' if sign > 0 else '-'}" for name, sign in zip(names, signs))


def certify_balance(pairs: pd.DataFrame, names: Sequence[str]) -> dict[str, float]:
    """Refuse unless every named quantity's direction accuracy is exactly 0.5.

    Verified rather than assumed: a tier that claims a confound is cancelled and
    is not exactly balanced is reporting the confound as part of the arm.
    """

    achieved: dict[str, float] = {}
    for name in names:
        values = pairs[f"{name}_direction"].to_numpy(dtype=np.float64)
        if values.size == 0:
            raise ValueError(f"an empty tier cannot be certified balanced on {name}")
        accuracy = float(values.mean())
        if accuracy != md.NULL_ACCURACY:
            raise ValueError(
                f"the tier is not exactly balanced on {name}: direction accuracy is {accuracy!r} "
                f"on {values.size} pairs"
            )
        achieved[name] = accuracy
    return achieved


def pair_counts(pairs: pd.DataFrame) -> dict[str, Any]:
    """Pair, variant, assay and cluster counts, and the per-cluster distribution."""

    record = dict(md.tier_counts(pairs)) if len(pairs) else {
        "pairs": 0, "variants": 0, "assays": 0, "clusters": 0
    }
    if len(pairs):
        sizes = pairs.groupby("group_id").size().to_numpy(dtype=np.float64)
        record["per_cluster"] = {
            "min": int(sizes.min()),
            "median": float(np.median(sizes)),
            "max": int(sizes.max()),
            "largest_cluster_share": float(sizes.max() / sizes.sum()),
        }
    return record


# --------------------------------------------------------------------------- #
# tier construction
# --------------------------------------------------------------------------- #


def _build(
    name: str,
    tolerance: float | None,
    frame: pd.DataFrame,
    construction: dict[str, Any],
) -> dict[str, Any]:
    return {
        "tier": tier_label(name, tolerance),
        "rule": TIER_RULES[name],
        "tolerance": None if tolerance is None else float(tolerance),
        "counts": pair_counts(frame),
        "construction": construction,
        "pairs": frame,
    }


def _matched_tier(
    name: str,
    candidates: pd.DataFrame,
    names: Sequence[str],
    tolerance: float,
) -> dict[str, Any]:
    subset = tolerance_matched(candidates, names, tolerance)
    frame = md.greedy_disjoint(subset, seed=md.SPLIT_SEED) if len(subset) else subset
    construction = {
        "matched_quantities": list(names),
        "candidates_offered": int(len(candidates)),
        "candidates_matched": int(len(subset)),
        "retained_after_disjointness": int(len(frame)),
        "achieved_distances": distance_summary(frame, names),
    }
    return _build(name, tolerance, frame, construction)


def _balanced_tier(
    name: str,
    tolerance: float | None,
    base: pd.DataFrame,
    names: Sequence[str],
    *,
    seed: int,
) -> dict[str, Any]:
    if len(base) == 0:
        return _build(
            name,
            tolerance,
            base,
            {
                "balanced_votes": list(names),
                "status": "no_pairs",
                "reason": "the tier this balance is imposed inside holds no pair",
            },
        )
    frame, record = balance_votes(base, names, seed=seed)
    return _build(name, tolerance, frame, record)


def forward_tiers(
    candidates: pd.DataFrame,
    *,
    tolerances: Sequence[float] = TOLERANCE_LADDER,
    seed: int = BALANCE_SEED,
) -> dict[str, dict[str, Any]]:
    """Every declared forward tier, keyed by its artefact label.

    ``candidates`` are the gap-satisfying candidate pairs, oriented and carrying
    the three statistics' rank distances and votes. Disjointness runs inside each
    matched tier, as E21 declares, so a tier's composition is never a function of
    another tier's criterion; the balanced tiers are subsets of ``T_open`` and the
    conjunction tier is a subset of its own matched tier.
    """

    built: dict[str, dict[str, Any]] = {}
    open_tier = md.greedy_disjoint(candidates, seed=md.SPLIT_SEED)
    built["T_open"] = _build(
        "T_open",
        None,
        open_tier,
        {
            "candidates_offered": int(len(candidates)),
            "retained_after_disjointness": int(len(open_tier)),
            "achieved_distances": distance_summary(open_tier, STATISTICS),
        },
    )
    for name, names in BALANCED_FORWARD_TIERS.items():
        built[name] = _balanced_tier(name, None, open_tier, names, seed=seed)
    for tolerance in tolerances:
        for name, names in MATCHED_FORWARD_TIERS.items():
            record = _matched_tier(name, candidates, names, tolerance)
            built[record["tier"]] = record
        matched = built[tier_label("T_tol_PEG", tolerance)]["pairs"]
        record = _balanced_tier(
            CONJUNCTION_FORWARD_TIER, tolerance, matched, STATISTICS, seed=seed
        )
        built[record["tier"]] = record
    return built


def reverse_tiers(
    candidates: pd.DataFrame,
    open_tier: pd.DataFrame,
    *,
    tolerances: Sequence[float] = TOLERANCE_LADDER,
    seed: int = BALANCE_SEED,
) -> dict[str, dict[str, Any]]:
    """Every declared reverse tier for one arm, keyed by its artefact label.

    The controlled quantity is the arm's own likelihood, so these tiers are
    arm-specific by construction and a family across arms is not available; the
    declaration states the family as the three statistics on one arm's tier.
    """

    built: dict[str, dict[str, Any]] = {}
    built[REVERSE_BALANCED_TIER] = _balanced_tier(
        REVERSE_BALANCED_TIER, None, open_tier, (MODEL,), seed=seed
    )
    for tolerance in tolerances:
        record = _matched_tier(REVERSE_MATCHED_TIER, candidates, (MODEL,), tolerance)
        built[record["tier"]] = record
        record = _balanced_tier(
            REVERSE_CONJUNCTION_TIER, tolerance, record["pairs"], (MODEL,), seed=seed
        )
        built[record["tier"]] = record
    return built


# --------------------------------------------------------------------------- #
# inference: one family per tier, the shared bidirectional band
# --------------------------------------------------------------------------- #


def pooled_draws(resampler: md.ClusterResampler, blocks: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """The pooled pair fraction of the drawn clusters, one value per draw.

    This is the arithmetic :meth:`matched_discrimination.ClusterResampler.interval`
    reports percentiles of, read out in full because a simultaneous band needs
    the whole draw vector and not only its quantiles. The draw matrix is the
    resampler's own, so every column of a family is paired draw by draw.
    """

    sums, counts = blocks
    drawn_counts = counts[resampler.index].sum(axis=1)
    drawn_sums = sums[resampler.index].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(drawn_counts > 0, drawn_sums / drawn_counts, np.nan)


def family_bands(
    columns: Mapping[str, Sequence[float]],
    clusters: Sequence[Any],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = CONFIDENCE,
    verdict_from: str = "simultaneous",
    role: str = "primary",
    resampler: md.ClusterResampler | None = None,
) -> dict[str, Any]:
    """One declared family: pooled direction accuracy with a shared simultaneous band.

    ``columns`` maps a label to that column's per-pair direction values, all read
    on the same pairs, so every column of the family is resampled on identical
    clusters and the columns are paired draw by draw.

    The endpoint is the pooled pair fraction over the tier's pairs with the family
    cluster as the resampling unit -- E21's own convention, and the quantity the
    exact balance is defined on, so a balanced tier's reference rules sit at
    exactly 0.5 where the construction puts them. Point estimates, pointwise
    percentile intervals and the unit floor come from
    :class:`matched_discrimination.ClusterResampler` unchanged; the family-wise
    critical value is the statistic ``bidirectional.NATIVE_METHOD`` declares, the
    ``confidence`` quantile of the largest absolute centered deviation over the
    family in native accuracy units, formed from those same draws. The sibling
    helper's :func:`bidirectional.paired_bands` cannot be called directly here
    because it forms a group-equal mean, which a pair-weighted endpoint is not.

    ``verdict_from`` is ``'simultaneous'`` for a declared family and
    ``'pointwise'`` for the construction-check reference rules, which the
    declaration keeps outside every verdict family.
    """

    if verdict_from not in ("simultaneous", "pointwise"):
        raise ValueError(f"unknown verdict source {verdict_from!r}")
    labels = list(columns)
    if not labels:
        raise ValueError("a family needs at least one column")
    labelled = np.asarray([int(value) for value in clusters])
    units = sorted(set(labelled.tolist()))
    floor = bootstrap_unit_floor(len(units))
    record: dict[str, Any] = {
        "role": role,
        "labels": labels,
        "family_size": len(labels),
        "n_pairs": int(labelled.size),
        "n_clusters": len(units),
        "endpoint": (
            "pooled pair fraction over the tier's pairs, the convention the exact balance is "
            "defined on; the cluster-equal mean is reported beside it"
        ),
        "unit": "wild-type family cluster at 50% identity",
        "verdict_from": verdict_from,
        "null": md.NULL_ACCURACY,
        "margin": md.PRACTICAL_MARGIN,
        **floor,
        "status": "complete",
        "band": None,
        "columns": {},
    }
    vectors: dict[str, np.ndarray] = {}
    for label in labels:
        vector = np.asarray(columns[label], dtype=np.float64)
        if vector.shape != labelled.shape:
            raise ValueError(f"{label}: one direction value per pair is required")
        if not np.isfinite(vector).all():
            raise ValueError(f"{label}: a pair carries no finite direction value")
        vectors[label] = vector

    def summary(label: str) -> dict[str, Any]:
        vector = vectors[label]
        return {
            "pooled": None if vector.size == 0 else float(vector.mean()),
            "cluster_mean": None if vector.size == 0 else float(
                np.mean([vector[labelled == unit].mean() for unit in units])
            ),
            "n_ties": int(np.count_nonzero(vector == md.NULL_ACCURACY)),
        }

    if floor["degenerate"]:
        record["status"] = "not_estimable"
        record["columns"] = {
            label: summary(label) | {
                "pointwise": None, "simultaneous": None, "verdict": md.level_verdict(None),
            }
            for label in labels
        }
        return record
    if resampler is None:
        resampler = md.ClusterResampler(units, draws=draws, seed=seed)
    elif resampler.units != units:
        raise ValueError("a shared resampler must be built on this tier's own clusters")
    blocks = {label: resampler.blocks(vectors[label], labelled) for label in labels}
    pointwise = {label: resampler.interval(blocks[label], confidence=confidence)
                 for label in labels}
    withheld = sorted(label for label in labels if pointwise[label]["interval"] is None)
    if withheld:
        record["status"] = "not_estimable"
        record["interval_reason"] = pointwise[withheld[0]]["interval_reason"]
        record["columns"] = {
            label: summary(label) | {
                "pointwise": None, "simultaneous": None, "verdict": md.level_verdict(None),
            }
            for label in labels
        }
        return record
    points = np.asarray([pointwise[label]["point"] for label in labels], dtype=np.float64)
    deviation = np.column_stack(
        [pooled_draws(resampler, blocks[label]) for label in labels]
    ) - points
    finite = np.isfinite(deviation).all(axis=1)
    critical = float(np.quantile(np.max(np.abs(deviation[finite]), axis=1), confidence))
    record["band"] = {
        "method": bidirectional.NATIVE_METHOD,
        "statistic_declared_by": "src/capability/extensions/bidirectional.py::paired_bands",
        "applied_to": "the pooled pair fraction of the drawn clusters, not a group-equal mean",
        "draws": int(resampler.draws),
        "finite_draws": int(finite.sum()),
        "seed": int(resampler.seed),
        "confidence": float(confidence),
        "groups": len(units),
        "family_size": len(labels),
        "critical_value": critical,
    }
    for position, label in enumerate(labels):
        point = float(points[position])
        simultaneous = [point - critical, point + critical]
        interval = [float(value) for value in pointwise[label]["interval"]]
        chosen = simultaneous if verdict_from == "simultaneous" else interval
        record["columns"][label] = summary(label) | {
            "pointwise": interval,
            "simultaneous": simultaneous,
            "verdict": md.level_verdict(chosen),
        }
    return record


def declared_constants() -> dict[str, Any]:
    """Every constant of the declaration, echoed into the run contract."""

    return {
        "declaration": DECLARATION,
        "declaration_id": DECLARATION_ID,
        "phenotype_gap": md.PHENOTYPE_GAP,
        "primary_tolerance": PRIMARY_TOLERANCE,
        "tolerance_ladder": list(TOLERANCE_LADDER),
        "tolerance_units": "within-assay standard deviations of the quantity's rank",
        "random_pair_separation": float(RANDOM_PAIR_SEPARATION),
        "covariate_scale": (
            "within-assay standardized rank over the assay's retained support rows, the panel's "
            "own raw_M/raw_P scale, standard deviation exactly 1"
        ),
        "statistics": list(STATISTICS),
        "greedy_seed": md.SPLIT_SEED,
        "balance_seed": BALANCE_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confidence": CONFIDENCE,
        "endpoint": (
            "pooled pair fraction over the tier's pairs, resampled by family cluster; a drawn "
            "cluster contributes all of its pairs. It is E21's own convention and the quantity the "
            "exact balance is defined on, so a balanced tier's reference rules sit at exactly 0.5. "
            "The cluster-equal mean is reported beside it and carries no verdict"
        ),
        "band": (
            "the statistic bidirectional.paired_bands declares -- the 95th percentile of the "
            "largest absolute centered bootstrap deviation over the family in native accuracy "
            "units -- formed from the ClusterResampler draws of the tier"
        ),
        "null_accuracy": md.NULL_ACCURACY,
        "practical_margin": md.PRACTICAL_MARGIN,
        "development_evaluation_split": (
            "not applied; no threshold in this reading is estimated from data, so E21's split has "
            "no function here and every cluster contributing a retained pair is read"
        ),
        "tier_rules": dict(TIER_RULES),
        "readings": dict(READINGS),
    }
