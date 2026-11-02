"""E21 and E22: direction accuracy on composition-matched variant pairs.

The question
===========

Results 1 reports that adding an independent-site evolutionary profile removes
77-86% of the native-likelihood mutation-ranking increment. That subtraction is
a regression adjustment, so it leaves open how much directional ability survives
once the confounds are removed *by construction*. This module builds the
construction.

Two single mutants of the same wild type in the same assay that carry the same
wild-type residue and the same mutant residue at two different positions give
two full-length sequences of identical length and identical amino-acid
composition. Substitution-matrix score, hydropathy change and mutation count are
then equal by construction rather than adjusted, and the pair differs only in
which site was substituted. Requiring a large measured phenotype gap makes the
pair a real discrimination problem; additionally matching the independent-site
profile score removes the evolutionary component whose overlap R1 measured.

E21 asks whether the native sequence likelihood ranks the better-phenotype
member above the worse one (``direction_accuracy``, null 0.5). E22 asks, on the
same pairs, whether the stored supervised out-of-family readouts recover a
direction the native rule misses.

What this support can and cannot carry
======================================

Support ``S1_retained_panel`` reads the retained per-variant native likelihood
from its **within-assay standardized rank** (``raw_M``). A within-assay rank is
a strictly increasing function of the underlying nats, so it preserves the
ordering of two variants drawn from the same assay -- which is the only quantity
a direction test reads. It carries no magnitude: no difference in nats, no
effect size in likelihood units, and no magnitude-stratified reading may be
computed here. Both members of every pair are verified to come from one assay
before any direction is read (:func:`verify_within_assay`).

Declared order of operations
============================

Every threshold is fixed before any model score is read, and in this order:

1. single substitutions of the retained common support only;
2. candidate pairs from the pair rule (same assay, same wild-type residue, same
   mutant residue, different positions);
3. the phenotype-gap threshold, a declared constant of 1.0 within-assay standard
   deviations of the z-scored measured score;
4. the development/evaluation split over wild-type family clusters;
5. the profile-match threshold, the 20th percentile of ``|delta profile_score|``
   over *development-cluster* candidate pairs that already satisfy the gap,
   applied unchanged to evaluation clusters;
6. seeded deterministic disjoint greedy pairing, per tier, reading no model score
   and no phenotype;
7. only then the stored model scores, on evaluation clusters.

Inference
=========

The wild-type family cluster is the resampling unit. One shared resample matrix
over the evaluation clusters scores every endpoint, so the tier contrast
``D_open - D_match`` is cluster-paired draw by draw even though the two tiers'
retained pair sets are not nested (the greedy rule runs inside each tier). Every
endpoint is a mean of per-pair values, so a draw is evaluated exactly from
per-cluster sums and counts rather than by rebuilding row indices.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..core.statistics import (
    MINIMUM_FINITE_DRAW_FRACTION,
    bootstrap_unit_floor,
)

SCHEMA = "matched_discrimination_v1"

#: Declared in ``configs/matched_discrimination_manifest.json``; the analysis is
#: reported under its own id and never overwrites an R1 record.
DECLARATION_ID = "matched_discrimination_20261102"
ANALYSIS_ID = "matched_discrimination_20261102_S1_retained_panel"
SUPPORT = "S1_retained_panel"

#: Cohort constants of the declaration. None of these may be revised after an
#: evaluation-cluster model score has been read.
PHENOTYPE_GAP = 1.0
PROFILE_MATCH_PERCENTILE = 20.0
DEVELOPMENT_FRACTION = 0.3
SPLIT_SEED = 20261102
BOOTSTRAP_DRAWS = 2000
PRACTICAL_MARGIN = 0.05
NULL_ACCURACY = 0.5

#: The three fold seeds of the retained panel. They partition one cohort three
#: ways; they are not three independent confirmations, and the project's
#: resolution convention is the conjunction over all three.
FOLD_SEEDS = (20260923, 20260924, 20260925)

#: The seed of the shared cluster resample matrix. Declared here because an
#: interval needs its draw stream named; it is the cohort seed of the
#: declaration, used on a different stream from the split and the greedy order.
BOOTSTRAP_SEED = 20261102

TIERS = ("T_open", "T_match")

#: What the manifest's readout names bind to among the stored columns of
#: ``manuscript/data/prediction-details/mutation-supporting-predictions.csv.gz``.
#: The binding is recorded in the artefact because one of the three declared
#: readouts does not exist on disk in the form the declaration describes.
READOUT_BINDING = {
    "R_like": {
        "column": "raw_M",
        "features": "none; the unfitted native likelihood on its within-assay rank scale",
        "fitted": False,
        "model_specific": True,
    },
    "R_seqprof": {
        "column": "profile_sequence",
        "features": "ridge on [profile rank, 444 sequence/mutation-identity/position/length descriptors]",
        "fitted": True,
        "model_specific": False,
    },
    "B": {
        "column": "B",
        "features": "ridge on [profile rank, likelihood rank, 444 descriptors]",
        "fitted": True,
        "model_specific": True,
    },
    "B_R": {
        "column": "B_R",
        "features": "ridge on [profile rank, likelihood rank, 444 descriptors, 1024 projected hidden-state dimensions]",
        "fitted": True,
        "model_specific": True,
    },
}

READOUT_BINDING_LIMIT = (
    "The declaration's R_repr is 'the primary frozen-state readout added on top of "
    "R_seqprof', i.e. [profile, sequence, representation] without the likelihood "
    "feature. No such design is stored: the only stored representation-bearing "
    "design is B_R = B + representation, which also contains the likelihood rank. "
    "The representation increment is therefore measurable only on top of B, and "
    "B_R - R_seqprof is reported as a descriptive surrogate for the declared "
    "contrast, confounded by the likelihood feature it also adds."
)

#: The out-of-family readout columns this analysis reads, in report order.
READOUTS = ("R_like", "R_seqprof", "B", "B_R")

#: E22 contrasts. The first two are the author's validation-plan primaries; the
#: third is the declaration's own first primary; the fourth is the surrogate for
#: the declaration's second primary and carries no verdict weight of its own.
E22_CONTRASTS = (
    ("B_minus_R_seqprof", "B", "R_seqprof", "primary", "the likelihood increment under supervision"),
    ("B_R_minus_B", "B_R", "B", "primary", "the representation increment on top of B"),
    ("R_seqprof_minus_R_like", "R_seqprof", "R_like", "primary", "whether supervision alone recovers direction the native rule misses"),
    ("B_R_minus_R_seqprof", "B_R", "R_seqprof", "descriptive_surrogate", "the declaration's R_repr - R_seqprof is not stored; this surrogate adds the likelihood rank as well as the representation"),
)

#: Declared interpretations of wording the manifest leaves open. Fixed before any
#: model score was read and recorded in the artefact so the choice is auditable.
DECLARED_INTERPRETATIONS = {
    "phenotype_z_reference": (
        "measured is z-scored within assay over every retained common-support row of "
        "that assay (128 rows per assay, singles and multiples), population SD. The "
        "assay's standard deviation is a property of the assay's sample in this "
        "support and must not depend on which variant subset the pair rule later "
        "selects."
    ),
    "gap_comparison": "pairs are retained at |delta z| >= 1.0, the declaration's 'at least one assay standard deviation'",
    "profile_match_comparison": "pairs are retained at |delta profile_score| <= the development-cluster 20th percentile",
    "profile_match_candidate_set": (
        "the threshold is the 20th percentile over development-cluster *candidate* "
        "pairs that satisfy the gap, as declared, not over the greedy-retained subset"
    ),
    "tie_rule": "an exact score tie between the two members counts as 0.5 of a correct direction, and every endpoint reports its tie rate",
    "contrast_decision_scale": (
        "the declaration states the decision rule in direction-accuracy units against "
        "the null 0.5 with a practical margin of 0.05. A contrast of two accuracies is "
        "read on the same margin against the null 0: supported when the interval "
        "excludes 0, bounded negative when it lies inside (-0.05, +0.05). No threshold "
        "or margin is changed by this translation."
    ),
    "bootstrap_cluster_universe": (
        "one resample matrix over the evaluation clusters that contribute at least one "
        "retained pair in either tier scores every endpoint, so the tier contrast is "
        "cluster-paired. Each endpoint also reports the number of those clusters that "
        "are informative for it, and the unit floor is applied to that count."
    ),
    "across_seed_rule": (
        "the project's resolution convention is the conjunction over the three fold "
        "seeds: supported only when every seed is supported with the same sign, bounded "
        "negative only when every seed is a bounded negative, unresolved otherwise."
    ),
}

LIMITATIONS = (
    "An interval covering the null is not equivalence, zero information or the absence "
    "of a biological effect. A bounded negative excludes an effect only at the declared "
    "margin of 0.05 direction-accuracy points.",
    "The ProteinGym export carries no measurement replicates or uncertainty, so a large "
    "phenotype gap is an observed gap and is not independently confirmed as a true gap.",
    "Matching the independent-site profile does not match all evolutionary information; "
    "an alignment-based profile is not an upper bound on all sequence statistics.",
    "Family clusters are the resampling unit. Pair counts are not independent sample "
    "size, and the three fold seeds partition one cohort rather than replicating it.",
    "On this support only the direction of a within-assay pair is read. No magnitude, "
    "difference in nats or effect size in likelihood units is computable from the rank "
    "scale, and none is reported.",
    "This cohort has been studied before, so this is a pre-specified reanalysis and not "
    "new external validation.",
)


# --------------------------------------------------------------------------- #
# cohort construction
# --------------------------------------------------------------------------- #


def standardized_rank(values: np.ndarray) -> np.ndarray:
    """Average-tie rank standardized by population SD, the panel's rank scale."""

    rank = rankdata(np.asarray(values, dtype=np.float64)).astype(np.float64)
    sd = rank.std()
    return (rank - rank.mean()) / sd if sd > 0 else rank * 0.0


def common_support(samples: pd.DataFrame) -> pd.DataFrame:
    """The retained R1 common support, with the pair rule's fields derived."""

    required = {
        "sample_id",
        "unit_id",
        "group_id",
        "family_id",
        "mutation",
        "positions_1based",
        "measured",
        "profile_score",
        "rank_target",
        "supporting_common_member",
        "variant_index",
    }
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"sample table lacks {sorted(missing)}")
    rows = samples.loc[samples["supporting_common_member"]].copy()
    if not (rows["family_id"] == rows["group_id"]).all():
        raise ValueError("family_id and group_id disagree on the common support")
    if rows[["measured", "profile_score"]].isna().any().any():
        raise ValueError("measured or profile_score is missing on the common support")
    if rows.groupby("unit_id")["group_id"].nunique().max() != 1:
        raise ValueError("an assay spans more than one family cluster")
    grouped = rows.groupby("unit_id")["measured"]
    sd = grouped.transform(lambda v: v.std(ddof=0))
    if float(sd.min()) <= 0.0:
        raise ValueError("an assay has zero measured variance; its z-score is undefined")
    rows["z"] = (rows["measured"] - grouped.transform("mean")) / sd
    rows["n_substitutions"] = rows["mutation"].str.count(":") + 1
    return rows


def single_substitutions(support: pd.DataFrame) -> pd.DataFrame:
    """Single substitutions with the wild-type residue, position and mutant parsed."""

    singles = support.loc[support["n_substitutions"] == 1].copy()
    singles["wildtype_residue"] = singles["mutation"].str[0]
    singles["mutant_residue"] = singles["mutation"].str[-1]
    singles["position"] = singles["mutation"].str[1:-1].astype(int)
    stated = singles["positions_1based"].astype(str)
    if not (singles["position"].astype(str) == stated).all():
        raise ValueError("the parsed position disagrees with positions_1based")
    if (singles["wildtype_residue"] == singles["mutant_residue"]).any():
        raise ValueError("a single substitution has equal wild-type and mutant residues")
    return singles


def candidate_pairs(singles: pd.DataFrame) -> pd.DataFrame:
    """Every pair the pair rule admits, before any gap or profile criterion.

    Composition, length, substitution-matrix score and chemistry are equal inside
    a pair by construction: the two sequences differ only in which site carries
    the substitution.
    """

    records: list[tuple] = []
    keys = ["unit_id", "wildtype_residue", "mutant_residue"]
    for (assay, wildtype, mutant), block in singles.groupby(keys, sort=True):
        if len(block) < 2:
            continue
        block = block.sort_values("position", kind="mergesort")
        rows = list(block.itertuples(index=False))
        for left, right in itertools.combinations(rows, 2):
            if left.position == right.position:
                raise ValueError("two retained variants share one position")
            records.append(
                (
                    assay,
                    wildtype,
                    mutant,
                    int(left.position),
                    int(right.position),
                    left.sample_id,
                    right.sample_id,
                    left.group_id,
                    float(left.z),
                    float(right.z),
                    float(left.profile_score),
                    float(right.profile_score),
                )
            )
    pairs = pd.DataFrame(
        records,
        columns=[
            "assay",
            "wildtype_residue",
            "mutant_residue",
            "position_a",
            "position_b",
            "sample_a",
            "sample_b",
            "group_id",
            "z_a",
            "z_b",
            "profile_a",
            "profile_b",
        ],
    )
    pairs["phenotype_gap"] = (pairs["z_a"] - pairs["z_b"]).abs()
    pairs["profile_distance"] = (pairs["profile_a"] - pairs["profile_b"]).abs()
    return pairs


def cluster_split(
    clusters: Sequence[Any], *, fraction: float = DEVELOPMENT_FRACTION, seed: int = SPLIT_SEED
) -> dict[str, list]:
    """Seeded deterministic development/evaluation partition of family clusters."""

    unique = [int(value) for value in sorted(set(clusters))]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    n_development = int(round(fraction * len(unique)))
    if not 0 < n_development < len(unique):
        raise ValueError("the development fraction retains no clusters on one side")
    development = sorted(unique[int(index)] for index in order[:n_development])
    evaluation = sorted(unique[int(index)] for index in order[n_development:])
    return {
        "development": development,
        "evaluation": evaluation,
        "n_clusters": len(unique),
        "n_development": len(development),
        "n_evaluation": len(evaluation),
        "realised_development_fraction": len(development) / len(unique),
        "seed": int(seed),
        "declared_fractions": [fraction, 1.0 - fraction],
    }


def match_threshold(
    development_candidates: pd.DataFrame,
    column: str,
    *,
    percentile: float = PROFILE_MATCH_PERCENTILE,
) -> float:
    """A match threshold, fixed on development-cluster candidates and never moved."""

    if development_candidates.empty:
        raise ValueError("no development-cluster candidate pair satisfies the prior criteria")
    return float(np.percentile(development_candidates[column].to_numpy(), percentile))


def profile_match_threshold(
    development_candidates: pd.DataFrame, *, percentile: float = PROFILE_MATCH_PERCENTILE
) -> float:
    """The declared profile-match threshold of the pre-registration."""

    if development_candidates.empty:
        raise ValueError("no development-cluster candidate pair satisfies the phenotype gap")
    return match_threshold(development_candidates, "profile_distance", percentile=percentile)


def greedy_disjoint(candidates: pd.DataFrame, *, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Seeded deterministic greedy pairing in which no variant appears twice.

    The candidate list is first put in one canonical order, then permuted by the
    declared seed, then walked once. The rule reads neither a model score nor the
    phenotype, so which pairs survive cannot depend on the answer. It runs inside
    a tier, as the declaration's disjointness clause states, so the two tiers'
    retained sets are not nested.
    """

    ordered = candidates.sort_values(
        ["assay", "wildtype_residue", "mutant_residue", "position_a", "position_b"],
        kind="mergesort",
    ).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ordered))
    used: set[str] = set()
    keep: list[int] = []
    for index in order:
        row = ordered.iloc[int(index)]
        if row.sample_a in used or row.sample_b in used:
            continue
        used.add(row.sample_a)
        used.add(row.sample_b)
        keep.append(int(index))
    return ordered.iloc[sorted(keep)].reset_index(drop=True)


def tier_counts(pairs: pd.DataFrame) -> dict[str, int]:
    """Pair, variant, assay and cluster counts of one tier and stratum."""

    variants = set(pairs["sample_a"]) | set(pairs["sample_b"])
    return {
        "pairs": int(len(pairs)),
        "variants": int(len(variants)),
        "assays": int(pairs["assay"].nunique()),
        "clusters": int(pairs["group_id"].nunique()),
    }


def _distribution(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def build_cohort(samples: pd.DataFrame) -> dict[str, Any]:
    """The whole declared construction, with every count and threshold recorded."""

    support = common_support(samples)
    singles = single_substitutions(support)
    candidates = candidate_pairs(singles)
    split = cluster_split(support["group_id"].unique())
    gap_satisfying = candidates.loc[candidates["phenotype_gap"] >= PHENOTYPE_GAP].copy()
    development = set(split["development"])
    development_candidates = gap_satisfying.loc[gap_satisfying["group_id"].isin(development)]
    threshold = profile_match_threshold(development_candidates)
    matched = gap_satisfying.loc[gap_satisfying["profile_distance"] <= threshold].copy()
    tiers = {
        "T_open": greedy_disjoint(gap_satisfying),
        "T_match": greedy_disjoint(matched),
    }
    for name, frame in tiers.items():
        frame["tier"] = name
        frame["stratum"] = np.where(
            frame["group_id"].isin(development), "development", "evaluation"
        )
        better_is_a = frame["z_a"] > frame["z_b"]
        frame["better_sample"] = np.where(better_is_a, frame["sample_a"], frame["sample_b"])
        frame["worse_sample"] = np.where(better_is_a, frame["sample_b"], frame["sample_a"])
        frame["better_profile"] = np.where(better_is_a, frame["profile_a"], frame["profile_b"])
        frame["worse_profile"] = np.where(better_is_a, frame["profile_b"], frame["profile_a"])
        if (frame["z_a"] == frame["z_b"]).any():
            raise ValueError("a retained pair has no phenotype gap")
    retained_keys = {
        name: set(zip(frame["sample_a"], frame["sample_b"], strict=True))
        for name, frame in tiers.items()
    }
    counts: dict[str, Any] = {}
    for name, frame in tiers.items():
        counts[name] = {
            "all": tier_counts(frame),
            "development": tier_counts(frame.loc[frame["stratum"] == "development"]),
            "evaluation": tier_counts(frame.loc[frame["stratum"] == "evaluation"]),
        }
    return {
        "support": {
            "variants": int(len(support)),
            "assays": int(support["unit_id"].nunique()),
            "clusters": int(support["group_id"].nunique()),
            "single_substitution_variants": int(len(singles)),
            "single_substitution_assays": int(singles["unit_id"].nunique()),
            "single_substitution_clusters": int(singles["group_id"].nunique()),
        },
        "candidates": {
            "pair_rule_pairs": int(len(candidates)),
            "pair_rule_groups": int(
                singles.groupby(["unit_id", "wildtype_residue", "mutant_residue"]).size().ge(2).sum()
            ),
            "phenotype_gap_pairs": int(len(gap_satisfying)),
            "phenotype_gap_clusters": int(gap_satisfying["group_id"].nunique()),
            "development_gap_pairs": int(len(development_candidates)),
            "profile_matched_pairs": int(len(matched)),
            "equal_measured_pairs": int((candidates["phenotype_gap"] == 0).sum()),
        },
        "split": split,
        "profile_match_threshold": {
            "value": threshold,
            "percentile": PROFILE_MATCH_PERCENTILE,
            "fixed_on": "development-cluster candidate pairs that satisfy the phenotype gap",
            "n_development_candidates": int(len(development_candidates)),
            "development_distance": _distribution(
                development_candidates["profile_distance"].to_numpy()
            ),
        },
        "counts": counts,
        "tier_overlap": {
            "T_match_pairs": len(retained_keys["T_match"]),
            "T_match_pairs_also_in_T_open": len(
                retained_keys["T_match"] & retained_keys["T_open"]
            ),
            "note": (
                "the greedy rule runs inside each tier, so the retained sets are not "
                "nested and the tier contrast is paired at the cluster, not the pair"
            ),
        },
        "achieved": {
            name: {
                stratum: {
                    "phenotype_gap": _distribution(part["phenotype_gap"].to_numpy()),
                    "profile_distance": _distribution(part["profile_distance"].to_numpy()),
                }
                for stratum, part in (
                    ("development", frame.loc[frame["stratum"] == "development"]),
                    ("evaluation", frame.loc[frame["stratum"] == "evaluation"]),
                )
            }
            for name, frame in tiers.items()
        },
        "tiers": tiers,
    }


# --------------------------------------------------------------------------- #
# direction accuracy and its cluster bootstrap
# --------------------------------------------------------------------------- #


def direction_values(
    pairs: pd.DataFrame, scores: Mapping[str, float], *, better: str = "better_sample", worse: str = "worse_sample"
) -> np.ndarray:
    """Per-pair correctness: 1 above, 0 below, 0.5 on an exact tie."""

    better_scores = pairs[better].map(scores).to_numpy(dtype=np.float64)
    worse_scores = pairs[worse].map(scores).to_numpy(dtype=np.float64)
    if not (np.isfinite(better_scores).all() and np.isfinite(worse_scores).all()):
        raise ValueError("a retained pair member has no finite score")
    return np.where(
        better_scores > worse_scores, 1.0, np.where(better_scores < worse_scores, 0.0, 0.5)
    )


def verify_within_assay(pairs: pd.DataFrame, assay_of: Mapping[str, str]) -> int:
    """Both members of every pair come from one assay, or this raises."""

    checked = 0
    for row in pairs.itertuples(index=False):
        if assay_of[row.sample_a] != assay_of[row.sample_b] or assay_of[row.sample_a] != row.assay:
            raise ValueError(f"pair {row.sample_a}/{row.sample_b} spans two assays")
        checked += 1
    return checked


class ClusterResampler:
    """One resample matrix over the evaluation clusters, shared by every endpoint.

    Every endpoint here is a mean of per-pair values, so a draw is scored from
    per-cluster sums and counts: a drawn cluster contributes its whole block, and
    a cluster drawn twice contributes twice. Sharing the matrix is what makes a
    contrast between two different pair sets paired at the cluster level.
    """

    def __init__(self, clusters: Sequence[Any], *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED):
        self.units = [int(value) for value in sorted(set(clusters))]
        if draws < 1:
            raise ValueError("a bootstrap needs at least one draw")
        self.draws = int(draws)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.index = rng.integers(0, len(self.units), size=(self.draws, len(self.units)))
        self._position = {unit: position for position, unit in enumerate(self.units)}

    def blocks(self, values: np.ndarray, clusters: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
        """Per-unit sums and counts of ``values``, aligned to the unit list."""

        values = np.asarray(values, dtype=np.float64)
        labels = np.asarray(clusters)
        if values.shape != labels.shape:
            raise ValueError("values and cluster labels must align")
        sums = np.zeros(len(self.units), dtype=np.float64)
        counts = np.zeros(len(self.units), dtype=np.float64)
        for value, label in zip(values, labels, strict=True):
            position = self._position[int(label)]
            sums[position] += value
            counts[position] += 1.0
        return sums, counts

    def _draw_means(self, sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
        drawn_counts = counts[self.index].sum(axis=1)
        drawn_sums = sums[self.index].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(drawn_counts > 0, drawn_sums / drawn_counts, np.nan)
        return means

    def interval(
        self,
        left: tuple[np.ndarray, np.ndarray],
        right: tuple[np.ndarray, np.ndarray] | None = None,
        *,
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        """Point estimate and percentile interval of a mean, or of a difference."""

        left_sums, left_counts = left
        point_count = float(left_counts.sum())
        if point_count <= 0:
            raise ValueError("an endpoint with no pairs has no point estimate")
        point = float(left_sums.sum() / point_count)
        informative = left_counts > 0
        draws = self._draw_means(left_sums, left_counts)
        if right is not None:
            right_sums, right_counts = right
            right_count = float(right_counts.sum())
            if right_count <= 0:
                raise ValueError("an endpoint with no pairs has no point estimate")
            point -= float(right_sums.sum() / right_count)
            informative = informative | (right_counts > 0)
            draws = draws - self._draw_means(right_sums, right_counts)
        finite = draws[np.isfinite(draws)]
        floor = bootstrap_unit_floor(int(informative.sum()))
        tail = (1.0 - confidence) / 2.0
        minimum = int(np.ceil(MINIMUM_FINITE_DRAW_FRACTION * self.draws))
        record: dict[str, Any] = {
            "point": point,
            "interval": None,
            "interval_reason": None,
            "confidence": float(confidence),
            "n_draws": self.draws,
            "n_finite_draws": int(finite.size),
            "n_panel_clusters": len(self.units),
            "bootstrap_seed": self.seed,
            **floor,
        }
        if floor["degenerate"]:
            record["interval_reason"] = floor["degenerate_reason"]
            return record
        if finite.size < minimum:
            record["interval_reason"] = (
                f"only {finite.size} of {self.draws} draws were defined, below the "
                f"{MINIMUM_FINITE_DRAW_FRACTION:.0%} floor"
            )
            return record
        record["interval"] = [
            float(np.quantile(finite, tail)),
            float(np.quantile(finite, 1.0 - tail)),
        ]
        return record


def level_verdict(interval: Sequence[float] | None) -> dict[str, Any]:
    """The declared decision rule for a direction accuracy against null 0.5."""

    return _verdict(interval, null=NULL_ACCURACY, margin=PRACTICAL_MARGIN)


def contrast_verdict(interval: Sequence[float] | None) -> dict[str, Any]:
    """The same rule on the contrast scale: null 0, the same 0.05 margin."""

    return _verdict(interval, null=0.0, margin=PRACTICAL_MARGIN)


def _verdict(interval: Sequence[float] | None, *, null: float, margin: float) -> dict[str, Any]:
    if interval is None:
        return {
            "verdict": "withheld",
            "null": null,
            "margin": margin,
            "excludes_null": None,
            "within_practical_margin": None,
            "direction": None,
        }
    low, high = float(interval[0]), float(interval[1])
    excludes = low > null or high < null
    within = low > null - margin and high < null + margin
    if excludes:
        verdict = "supported"
    elif within:
        verdict = "bounded_negative"
    else:
        verdict = "unresolved"
    return {
        "verdict": verdict,
        "null": null,
        "margin": margin,
        "excludes_null": bool(excludes),
        "within_practical_margin": bool(within),
        "direction": ("above" if low > null else "below") if excludes else None,
    }


def across_seeds(records: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """The project's conjunction convention over the three fold seeds."""

    verdicts = {str(seed): record["verdict"]["verdict"] for seed, record in records.items()}
    directions = {str(seed): record["verdict"]["direction"] for seed, record in records.items()}
    supported = set(verdicts.values()) == {"supported"} and len(set(directions.values())) == 1
    bounded = set(verdicts.values()) == {"bounded_negative"}
    return {
        "verdict": "supported" if supported else "bounded_negative" if bounded else "unresolved",
        "direction": next(iter(set(directions.values()))) if supported else None,
        "per_seed_verdict": verdicts,
        "points": {str(seed): record["point"] for seed, record in records.items()},
        "intervals": {str(seed): record["interval"] for seed, record in records.items()},
        "point_spread": (
            float(max(record["point"] for record in records.values())
                  - min(record["point"] for record in records.values()))
        ),
        "rule": DECLARED_INTERPRETATIONS["across_seed_rule"],
    }


def endpoint(
    resampler: ClusterResampler,
    values: np.ndarray,
    clusters: Sequence[Any],
    *,
    level: bool = True,
) -> dict[str, Any]:
    """One level endpoint: accuracy, interval, verdict, tie rate and counts."""

    blocks = resampler.blocks(values, clusters)
    record = resampler.interval(blocks)
    record["verdict"] = level_verdict(record["interval"]) if level else contrast_verdict(record["interval"])
    record["n_pairs"] = int(np.asarray(values).size)
    record["n_ties"] = int(np.count_nonzero(np.asarray(values) == 0.5))
    record["tie_rate"] = float(record["n_ties"] / record["n_pairs"]) if record["n_pairs"] else None
    record["blocks"] = blocks
    return record


def contrast(
    resampler: ClusterResampler,
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    """A cluster-paired contrast of two level endpoints under the shared draws."""

    record = resampler.interval(left["blocks"], right["blocks"])
    record["verdict"] = contrast_verdict(record["interval"])
    record["left_point"] = left["point"]
    record["right_point"] = right["point"]
    record["n_pairs_left"] = left["n_pairs"]
    record["n_pairs_right"] = right["n_pairs"]
    return record


def strip_blocks(record: Mapping[str, Any]) -> dict[str, Any]:
    """The serialisable part of an endpoint record."""

    return {key: value for key, value in record.items() if key != "blocks"}


# --------------------------------------------------------------------------- #
# E22 validity: the stored readouts' established out-of-family performance
# --------------------------------------------------------------------------- #


def family_mean_spearman(
    predictions: np.ndarray,
    target: np.ndarray,
    assays: np.ndarray,
    clusters: np.ndarray,
) -> dict[str, Any]:
    """The panel's published metric: per-assay Spearman, averaged by cluster.

    This is exactly what ``readout_analysis.evaluate_readouts`` summarises -- a
    Spearman of the prediction against the within-assay standardized rank target
    per assay, then :func:`profiles.cluster_means` and their mean -- so a
    reproduction of the published value is an identity check on this join.
    """

    per_assay: list[float] = []
    per_cluster: list[int] = []
    excluded = 0
    for assay in np.unique(assays):
        rows = np.flatnonzero(assays == assay)
        left = rankdata(predictions[rows])
        right = target[rows]
        if np.std(left) < 1e-12 or np.std(right) < 1e-12:
            excluded += 1
            continue
        per_assay.append(float(np.corrcoef(left, right)[0, 1]))
        per_cluster.append(int(clusters[rows][0]))
    if not per_assay:
        return {"point": None, "n_assays": 0, "excluded_assays": excluded, "values": [], "clusters": []}
    values = np.asarray(per_assay, dtype=np.float64)
    labels = np.asarray(per_cluster)
    means = np.asarray([values[labels == label].mean() for label in np.unique(labels)])
    return {
        "point": float(means.mean()),
        "n_assays": len(per_assay),
        "excluded_assays": excluded,
        "n_clusters": int(np.unique(labels).size),
        "values": values,
        "clusters": labels,
    }


# --------------------------------------------------------------------------- #
# amendment_03: the position control, the sign-balanced tier, net of profile
# --------------------------------------------------------------------------- #

AMENDMENT_03_ID = "matched_discrimination_20261102_S1_retained_panel_amendment_03"

#: The two position-only rules the amendment declares. Both read the substituted
#: position and the wild-type length and nothing else.
POSITION_RULES = {
    "nearer_terminus": "prefer the member whose substituted site is nearer a sequence terminus",
    "further_from_midpoint": (
        "prefer the member whose site has the larger relative distance from the sequence midpoint"
    ),
}

#: Why the two declared rules cannot disagree on this cohort. Both members of a
#: pair are substitutions of one wild type, so they share a length, and for a
#: fixed length the distance to the nearer terminus and the distance from the
#: midpoint sum to a constant. The rules are therefore one rule here, ties
#: included, and the amendment's two clauses are not two independent controls.
POSITION_RULE_IDENTITY = (
    "min(position - 1, length - position) = (length - 1)/2 - |position - (length + 1)/2|, and the "
    "two members of a pair share one wild-type length, so 'nearer a terminus' and 'further from the "
    "midpoint' induce the same within-pair ordering and the same ties. Agreement is verified rather "
    "than assumed and reported as a construction fact, not as a replication."
)

AMENDMENT_03_INTERPRETATIONS = {
    "position_statistic": (
        "relative midpoint distance r = |position - (length + 1)/2| / ((length - 1)/2), which is 0 at "
        "the midpoint and 1 at either terminus. It is the quantity both declared rules order a pair "
        "by, so it is also the quantity the position-matched tier matches."
    ),
    "position_match_threshold": (
        "the amendment declares a tier 'additionally matched on relative position' without naming a "
        "percentile. It is fixed by the declaration's own convention for the profile match: the 20th "
        "percentile of |delta r| over development-cluster candidate pairs that already satisfy the "
        "phenotype gap and the profile match, applied unchanged to evaluation clusters. It reads no "
        "model score."
    ),
    "balanced_tier_ties": (
        "pairs whose profile scores are exactly equal cast no directional vote and are retained in "
        "full. With equal counts of profile-correct and profile-incorrect pairs, each tie "
        "contributing 0.5, profile direction accuracy on T_balanced is exactly 0.5 rather than 0.5 "
        "up to parity."
    ),
    "balanced_tier_selection": (
        "the majority class is subsampled to the minority count by a seeded permutation of its "
        "canonically ordered pairs, within each of the development and evaluation strata separately "
        "so the construction holds where the numbers are read. The rule reads the phenotype and the "
        "profile score only."
    ),
    "practical_size_status": (
        "the selection-quality block is descriptive. It carries no verdict, enters no multiplicity "
        "family and cannot overturn any endpoint."
    ),
}


def relative_midpoint_distance(positions: Sequence[float], lengths: Sequence[float]) -> np.ndarray:
    """Distance from the sequence midpoint, 0 at the midpoint and 1 at a terminus."""

    position = np.asarray(positions, dtype=np.float64)
    length = np.asarray(lengths, dtype=np.float64)
    if position.shape != length.shape:
        raise ValueError("positions and lengths must align")
    if np.any(length < 2):
        raise ValueError("a relative position needs a wild type of at least two residues")
    if np.any((position < 1) | (position > length)):
        raise ValueError("a substituted position lies outside its wild type")
    return np.abs(position - (length + 1.0) / 2.0) / ((length - 1.0) / 2.0)


def terminal_distance(positions: Sequence[float], lengths: Sequence[float]) -> np.ndarray:
    """Residues from the substituted site to the nearer terminus."""

    position = np.asarray(positions, dtype=np.float64)
    length = np.asarray(lengths, dtype=np.float64)
    return np.minimum(position - 1.0, length - position)


def attach_positions(pairs: pd.DataFrame, lengths: Mapping[str, int]) -> pd.DataFrame:
    """Add the wild-type length, the position statistics and ``position_distance``.

    Works on a candidate frame as well as on a retained tier, because the
    position-match threshold is fixed on development-cluster candidates. The
    better/worse columns appear only once the phenotype has ordered the pair.
    """

    frame = pairs.copy()
    missing = sorted(set(frame["assay"]) - set(lengths))
    if missing:
        raise ValueError(f"no wild-type length for {missing[:3]}")
    frame["length"] = frame["assay"].map(lengths).astype(int)
    frame["midpoint_distance_a"] = relative_midpoint_distance(frame["position_a"], frame["length"])
    frame["midpoint_distance_b"] = relative_midpoint_distance(frame["position_b"], frame["length"])
    frame["position_distance"] = np.abs(
        frame["midpoint_distance_a"] - frame["midpoint_distance_b"]
    )
    if "better_sample" not in frame.columns:
        return frame
    better_is_a = frame["better_sample"].to_numpy() == frame["sample_a"].to_numpy()
    better_position = np.where(better_is_a, frame["position_a"], frame["position_b"])
    worse_position = np.where(better_is_a, frame["position_b"], frame["position_a"])
    frame["better_position"] = better_position
    frame["worse_position"] = worse_position
    frame["better_midpoint_distance"] = relative_midpoint_distance(better_position, frame["length"])
    frame["worse_midpoint_distance"] = relative_midpoint_distance(worse_position, frame["length"])
    frame["better_terminal_distance"] = terminal_distance(better_position, frame["length"])
    frame["worse_terminal_distance"] = terminal_distance(worse_position, frame["length"])
    return frame


def position_direction_values(pairs: pd.DataFrame, rule: str) -> np.ndarray:
    """Per-pair correctness of a position-only rule, on the declared tie rule."""

    if rule not in POSITION_RULES:
        raise ValueError(f"{rule} is not a declared position rule")
    if rule == "nearer_terminus":
        better = -pairs["better_terminal_distance"].to_numpy(dtype=np.float64)
        worse = -pairs["worse_terminal_distance"].to_numpy(dtype=np.float64)
    else:
        better = pairs["better_midpoint_distance"].to_numpy(dtype=np.float64)
        worse = pairs["worse_midpoint_distance"].to_numpy(dtype=np.float64)
    return np.where(better > worse, 1.0, np.where(better < worse, 0.0, 0.5))


def profile_direction_values(pairs: pd.DataFrame) -> np.ndarray:
    """Per-pair correctness of the independent-site profile score."""

    better = pairs["better_profile"].to_numpy(dtype=np.float64)
    worse = pairs["worse_profile"].to_numpy(dtype=np.float64)
    return np.where(better > worse, 1.0, np.where(better < worse, 0.0, 0.5))


def balance_profile_vote(pairs: pd.DataFrame, *, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """T_balanced: the profile's directional vote cancelled by construction.

    Equal counts of pairs whose higher profile score belongs to the better and to
    the worse member, with every exactly-tied pair retained. Profile direction
    accuracy on the result is exactly 0.5, by construction rather than by
    estimation, so an arm above 0.5 there is not carrying the independent-site
    profile. The balance is struck inside each stratum, because the construction
    has to hold on the evaluation clusters where the numbers are read.
    """

    kept: list[pd.DataFrame] = []
    for stratum in sorted(pairs["stratum"].unique()):
        block = pairs.loc[pairs["stratum"] == stratum]
        values = profile_direction_values(block)
        ties = block.loc[values == 0.5]
        correct = block.loc[values == 1.0]
        incorrect = block.loc[values == 0.0]
        size = min(len(correct), len(incorrect))
        kept.append(ties)
        for side in (correct, incorrect):
            ordered = side.sort_values(
                ["assay", "wildtype_residue", "mutant_residue", "position_a", "position_b"],
                kind="mergesort",
            ).reset_index(drop=True)
            order = np.random.default_rng(seed).permutation(len(ordered))
            kept.append(ordered.iloc[sorted(int(index) for index in order[:size])])
    balanced = pd.concat(kept, ignore_index=True)
    balanced["tier"] = "T_balanced"
    return balanced.sort_values(
        ["stratum", "assay", "position_a", "position_b"], kind="mergesort"
    ).reset_index(drop=True)


def selection_quality(correct: np.ndarray, gaps: np.ndarray) -> dict[str, Any]:
    """Descriptive practical size of a pairwise rule, in phenotype units.

    Direction accuracy says how often a rule orders a pair correctly. It does not
    say how much phenotype a user who followed the rule would get, and on a
    cohort whose pairs differ by at least one assay standard deviation that is
    the question a designer asks. Two descriptive translations, both bounded by
    chance below and the oracle above:

    ``gap_capture``
        the share of the available phenotype gap the rule collects when it picks
        one member of every pair. Chance is 0.5 and the oracle 1.0.
    ``selected_advantage``
        the mean phenotype of the selected member relative to the pair's own
        mean, in within-assay standard deviations. Chance is 0.0 and the oracle
        is half the mean gap.

    Neither carries a verdict, and neither is an endpoint.
    """

    correct = np.asarray(correct, dtype=np.float64)
    gaps = np.asarray(gaps, dtype=np.float64)
    if correct.shape != gaps.shape or correct.size == 0:
        raise ValueError("correctness and gaps must align and be non-empty")
    accuracy = float(correct.mean())
    advantage = (2.0 * correct - 1.0) * gaps / 2.0
    return {
        "accuracy": accuracy,
        "expected_correctly_ordered_per_100_pairs": 100.0 * accuracy,
        "share_of_chance_to_oracle": 2.0 * (accuracy - 0.5),
        "gap_capture": float((correct * gaps).sum() / gaps.sum()),
        "gap_capture_chance": 0.5,
        "gap_capture_oracle": 1.0,
        "selected_advantage_sd": float(advantage.mean()),
        "selected_advantage_chance_sd": 0.0,
        "selected_advantage_oracle_sd": float((gaps / 2.0).mean()),
        "mean_phenotype_gap_sd": float(gaps.mean()),
        "n_pairs": int(correct.size),
        "per_pair_advantage": advantage,
        "status": "descriptive; no verdict, no multiplicity family",
    }


def spearman_interval(
    values: np.ndarray, clusters: np.ndarray, *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    """Cluster percentile interval of a family-mean per-assay Spearman."""

    labels = np.unique(clusters)
    means = np.asarray([values[clusters == label].mean() for label in labels])
    floor = bootstrap_unit_floor(int(means.size))
    if floor["degenerate"]:
        return {"interval": None, "interval_reason": floor["degenerate_reason"], **floor}
    rng = np.random.default_rng(seed)
    index = rng.integers(0, means.size, size=(draws, means.size))
    statistic = means[index].mean(axis=1)
    return {
        "interval": [float(np.quantile(statistic, 0.025)), float(np.quantile(statistic, 0.975))],
        "interval_reason": None,
        "n_draws": int(draws),
        "bootstrap_seed": int(seed),
        **floor,
    }
