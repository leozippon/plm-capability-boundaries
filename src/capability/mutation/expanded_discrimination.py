"""Matched same-substitution pair cohort and direction endpoints for support ``S2_expanded``.

The scientific object is a pair of single mutants of one wild type, measured in
one assay, carrying the *same* wild-type residue and the *same* mutant residue
at two different positions. Both full-length sequences then have identical
length and identical amino-acid composition and differ only in which site was
substituted, so composition, chemistry, substitution-matrix score and length are
matched by construction rather than adjusted for. ``configs/matched_discrimination_manifest.json``
is the binding declaration; every constant below is transcribed from it and none
of it is tunable here.

What this module may read
-------------------------
Nothing in cohort construction reads a model score. The pair rule, the
phenotype-gap threshold, the development/evaluation split, the profile-match
threshold and the seeded disjointness rule are all computed from measured
phenotypes and the independent-site profile alone, which is what makes the
pre-registration's ``forbidden`` clause enforceable rather than aspirational.
The endpoint functions take likelihoods as arguments and never load a
checkpoint.

The profile channel
-------------------
The declared profile score is the independent-site log-odds
``s = log f(mutant | position) - log f(wild type | position)``. No ProteinGym
alignment is staged on this host; what exists is the retained UniRef50/DIAMOND
lookup profile of ``results/R5/retrieval_bound_20260807`` (``profiles.npz``,
one ``(length, 20)`` column-normalised matrix per wild type, pseudocount
``alpha = 0.05`` on the corpus background, 80% identity reweighting, up to 5,000
hits). :func:`lookup_profile_scores` reproduces the retained ``profile_score``
column to 8.9e-16 and extends to *any* single substitution of the same 187 wild
types, which is what lets the cohort grow beyond the retained variant list
without substituting a different profile.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..core.amino_acids import AA20
from ..core.statistics import MINIMUM_FINITE_DRAW_FRACTION, bootstrap_unit_floor

# --------------------------------------------------------------- declared constants

#: Pre-registration this module executes, by id.
DECLARATION_ID = "matched_discrimination_20261102"

#: Declared analysis id and artefact schema of support ``S2_expanded``. A later
#: reproduction takes a new analysis id rather than overwriting these artefacts.
ANALYSIS_ID = "matched_discrimination_s2_expanded_20261102"
SCHEMA = "matched_discrimination_s2_v1"

#: The two supports this analysis writes, named on every artefact because
#: ``amendment_01`` forbids pooling them.
RETAINED_SUPPORT = "retained_variants"
EXPANDED_SUPPORT = "expanded_variants"

#: The nine declared arms, as ``configs/matched_discrimination_manifest.json``
#: spells them, mapped onto the ids this repository's loaders take.
#: ``esm2_t33_650M_UR50D`` is the manifest's spelling of the local checkpoint
#: directory; the repository id for that checkpoint is ``esm2-650m`` and the
#: masked loaders expect the directory name ``esm2_650m``, which is resolved by a
#: recorded symlink rather than by renaming anything on disk.
DECLARED_ARMS: dict[str, dict[str, str]] = {
    "progen2-small": {"arm": "progen2-small", "family": "autoregressive_protein"},
    "progen2-medium": {"arm": "progen2-medium", "family": "autoregressive_protein"},
    "progen2-large": {"arm": "progen2-large", "family": "autoregressive_protein"},
    "progen2-xlarge": {"arm": "progen2-xlarge", "family": "autoregressive_protein"},
    "progen3-112m": {"arm": "progen3-112m", "family": "autoregressive_protein"},
    "progen3-3b": {"arm": "progen3-3b", "family": "autoregressive_protein"},
    "RITA_xl": {"arm": "rita-xl", "family": "autoregressive_protein"},
    "ProtGPT2": {"arm": "protgpt2", "family": "autoregressive_protein"},
    "esm2_t33_650M_UR50D": {"arm": "esm2-650m", "family": "different_scoring_rule_contrast"},
}

#: The one declared arm whose scoring rule is not autoregressive, its local
#: checkpoint, and the directory name the repository's masked loaders expect.
MASKED_ARM = "esm2-650m"
MASKED_MANIFEST_ARM = "esm2_t33_650M_UR50D"
MASKED_CHECKPOINT = "/Data/public/esm2_t33_650M_UR50D"
MASKED_LINK_NAME = "esm2_650m"

#: Scoring conventions recorded on every artefact. The ProGen3 rows are the
#: point: the repository carries two different ProGen3 estimands and this
#: analysis uses the readout-extraction one.
SCORING_CONVENTIONS: dict[str, str] = {
    "autoregressive": (
        "summed native log-likelihood over the arm's own scored residue span "
        "(target_only_native_packed_residue_span), mutant minus wild type, in nats, "
        "at the frozen 1024-position budget; markers, prompts and padding excluded "
        "by each arm's own rendering declaration"
    ),
    "progen3_direction": (
        "N-to-C only, which is the direction convention of "
        "src/capability/models/progen3.py::load_progen3_as_arm "
        "(scoring_directions = ('n_to_c',)) and therefore of the frozen-readout "
        "extraction and of the retained panel's raw_M. Stage 20's DMS scorer uses "
        "the bidirectional mean of the two directional sums, which is a different "
        "estimand and is not used here"
    ),
    "masked": (
        "masked-marginal contrast at corruption level 1/L with the substituted "
        "position alone masked; wild type and mutant are read off the same forward. "
        "A masked language model has no sequence log likelihood and none is claimed: "
        "this arm is reported separately and never pooled with the autoregressive arms"
    ),
    "protgpt2": "native FASTA wrapping; formatting newline tokens are inside the scored target",
}

#: Minimum measured phenotype gap, in within-assay standard deviations of the
#: z-scored DMS score. Declared as a constant, not tuned on any data.
GAP_THRESHOLD = 1.0

#: The profile-match threshold is this percentile of ``|delta s|`` over
#: development-cluster candidate pairs that already satisfy the gap.
PROFILE_MATCH_PERCENTILE = 20.0

#: Development/evaluation split over wild-type family clusters at 50% identity.
SPLIT_SEED = 20261102
DEVELOPMENT_FRACTION = 0.3

#: One seed governs every model-blind cohort decision: the split, the greedy
#: disjointness traversal and the per-cluster retention order.
COHORT_SEED = SPLIT_SEED

#: Inference: paired group bootstrap over wild-type family clusters.
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_ALPHA = 0.05
NULL_ACCURACY = 0.5
PRACTICAL_MARGIN = 0.05

#: Pseudocount of the retained lookup profile, from ``profiles.json``.
PROFILE_PSEUDOCOUNT = 0.05

#: Quantile bins of ``|delta log-likelihood|`` for the amendment-02 secondary
#: descriptive endpoint; boundaries are fixed on development clusters.
MAGNITUDE_BINS = 5

VERDICT_SUPPORTED = "supported"
VERDICT_BOUNDED_NEGATIVE = "bounded_negative"
VERDICT_UNRESOLVED = "unresolved"
VERDICT_DEGENERATE = "non_identifiable_below_unit_floor"

TIER_OPEN = "T_open"
TIER_MATCH = "T_match"

#: ``amendment_03`` addition 1. A subset of ``T_match`` in which the profile's
#: directional vote is cancelled by construction rather than by estimation.
TIER_BALANCED = "T_balanced"

#: ``amendment_04``. ``T_position_balanced`` cancels the position-only rule's
#: vote and leaves the profile uncontrolled; ``T_double_balanced`` cancels both
#: at once and is the decisive tier, because an arm resolved above 0.5 there has
#: directional ability attributable to neither site-wise conservation nor
#: sequence position, by construction rather than by adjustment.
TIER_POSITION_BALANCED = "T_position_balanced"
TIER_DOUBLE_BALANCED = "T_double_balanced"

#: ``amendment_03`` addition 3, as one rule rather than two. The amendment names
#: (a) "nearer a sequence terminus" and (b) "larger relative distance from the
#: sequence midpoint". Both members of a pair substitute the same wild type, so
#: they share a length, and for a fixed length
#: ``min(p - 1, L - p) = (L - 1)/2 - |p - (L + 1)/2|``: the two rules are strictly
#: opposite monotone functions of the same quantity and induce the identical
#: within-pair ordering. :func:`position_rule_agreement` verifies that on the
#: actual cohort instead of asserting it, and the artefact records the count.
POSITION_RULE_TERMINUS = "terminus"
POSITION_RULE_MIDPOINT = "midpoint"

#: Spelled once so the artefact and the report cannot disagree about what the
#: numbers mean.
SEMANTICS: dict[str, str] = {
    "phenotype": (
        "ProteinGym DMS_score, z-scored within assay over that assay's own "
        "single-substitution rows in the support being analysed; the release has "
        "already applied raw_DMS_directionality, so a larger value is more of the "
        "measured function and the higher-z member of a pair is the better-phenotype member"
    ),
    "profile": (
        "independent-site UniRef50/DIAMOND lookup log-odds "
        "log[f(m,p) + 0.05*b(m)] - log[f(w,p) + 0.05*b(w)], per-column normalised "
        "frequencies, 80% identity reweighting, <=5,000 hits per query"
    ),
    "direction": (
        "1 when the scoring rule's value is strictly greater for the "
        "better-phenotype member, 0 otherwise; exact ties count 0 and are "
        "reported separately"
    ),
    "resampling": (
        "wild-type family cluster at 50% identity; a drawn cluster contributes all "
        "of its pairs, so the statistic is the pooled pair fraction of the drawn "
        "clusters and not an average of cluster means"
    ),
    "position_rule": (
        "amendment_03 addition 3: prefer the member whose substituted site is nearer a "
        "sequence terminus. Reads only the position and the wild-type length, never the "
        "phenotype, the profile or a model. The amendment's second spelling -- larger "
        "relative distance from the sequence midpoint -- induces the identical "
        "within-pair ordering because the two members share a length, so it is reported "
        "as one rule with the agreement verified, not as two"
    ),
    "balanced_tier": (
        "amendment_03 addition 1: T_match pairs on which the profile casts a strict "
        "directional vote, thinned by a seeded rule to equal numbers of pairs where the "
        "higher profile score belongs to the better-phenotype member and to the "
        "worse-phenotype member. Profile direction accuracy is then exactly 0.5 by "
        "construction. Balanced inside the development and the evaluation stratum "
        "separately, so the reported evaluation balance is exact rather than inherited"
    ),
}


# ------------------------------------------------------------------ the pair object


@dataclass(frozen=True)
class Pair:
    """One matched same-substitution pair, oriented by measured phenotype.

    ``better`` and ``worse`` name the *phenotype*; nothing here reads a model.
    """

    assay: str
    cluster: int
    wildtype_id: str
    wt_residue: str
    mut_residue: str
    position_better: int
    position_worse: int
    z_better: float
    z_worse: float
    profile_better: float
    profile_worse: float

    @property
    def mutation_better(self) -> str:
        return f"{self.wt_residue}{self.position_better}{self.mut_residue}"

    @property
    def mutation_worse(self) -> str:
        return f"{self.wt_residue}{self.position_worse}{self.mut_residue}"

    @property
    def gap(self) -> float:
        """Measured phenotype gap in within-assay standard deviations."""

        return float(self.z_better - self.z_worse)

    @property
    def profile_distance(self) -> float:
        """``|delta s|``, the quantity the profile match thresholds."""

        return abs(float(self.profile_better - self.profile_worse))

    @property
    def key(self) -> tuple[str, str, str, int, int]:
        """Canonical, model-blind ordering key."""

        low, high = sorted((self.position_better, self.position_worse))
        return (self.assay, self.wt_residue, self.mut_residue, low, high)

    def record(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            mutation_better=self.mutation_better,
            mutation_worse=self.mutation_worse,
            gap=self.gap,
            profile_distance=self.profile_distance,
        )
        return payload


# ------------------------------------------------------------- phenotype and profile


def within_assay_z(measured: Sequence[float]) -> np.ndarray | None:
    """Z-score of one assay's measured values, or ``None`` when it is undefined.

    Returned rather than raised: an assay whose retained single-substitution
    population is constant (in the retained support, two assays carry a single
    single-substitution row each) has no within-assay standard deviation, so the
    declared gap threshold cannot be expressed in its units. Such an assay
    contributes no pair and is recorded as skipped with that reason, which is a
    fact about the support rather than an error.
    """

    values = np.asarray(measured, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("an assay's measured vector must be one-dimensional and non-empty")
    if not np.isfinite(values).all():
        raise ValueError("measured phenotypes must be finite")
    spread = float(values.std(ddof=0))
    if spread <= 0.0:
        return None
    return (values - values.mean()) / spread


def lookup_profile_scores(
    frequencies: np.ndarray,
    background: np.ndarray,
    substitutions: Sequence[tuple[str, int, str]],
    *,
    wildtype: str,
    alpha: float = PROFILE_PSEUDOCOUNT,
) -> np.ndarray:
    """Independent-site lookup log-odds for single substitutions of one wild type.

    The wild-type residue of every mutation string is verified against the wild
    type's own sequence. A mismatch means the coordinate frame or the reference
    sequence is wrong, which would silently score a different substitution, so it
    is refused rather than skipped.
    """

    if frequencies.ndim != 2 or frequencies.shape[1] != len(AA20):
        raise ValueError("profile frequencies must be (length, 20)")
    if alpha <= 0:
        raise ValueError("the pseudocount weight must be positive")
    code = {residue: index for index, residue in enumerate(AA20)}
    positions, wild, mutated = [], [], []
    for residue, position, substituted in substitutions:
        if not 1 <= position <= len(wildtype):
            raise ValueError(f"{residue}{position}{substituted}: position outside the wild type")
        if wildtype[position - 1] != residue:
            raise ValueError(
                f"{residue}{position}{substituted}: the wild type carries "
                f"{wildtype[position - 1]!r} at position {position}"
            )
        if residue not in code or substituted not in code:
            raise ValueError(f"{residue}{position}{substituted} is outside AA20")
        if position > frequencies.shape[0]:
            raise IndexError(f"position {position} outside a {frequencies.shape[0]}-column profile")
        positions.append(position - 1)
        wild.append(code[residue])
        mutated.append(code[substituted])
    rows = np.asarray(positions, dtype=np.int64)
    w = np.asarray(wild, dtype=np.int64)
    m = np.asarray(mutated, dtype=np.int64)
    scores = np.log(frequencies[rows, m] + alpha * background[m]) - np.log(
        frequencies[rows, w] + alpha * background[w]
    )
    if not np.isfinite(scores).all():
        raise FloatingPointError("a lookup profile score is not finite")
    return scores.astype(np.float64)


# ------------------------------------------------------------------- the split


def development_clusters(
    clusters: Iterable[int], *, seed: int = SPLIT_SEED, fraction: float = DEVELOPMENT_FRACTION
) -> frozenset[int]:
    """The development share of the declared cluster universe.

    The universe is passed in and permuted once, so the assignment of a cluster
    to development or evaluation is a property of the declared universe and not
    of which support happens to contain that cluster. Both supports of
    ``amendment_01`` therefore place a given family on the same side of the
    split.
    """

    if not 0.0 < fraction < 1.0:
        raise ValueError("the development fraction lies strictly between 0 and 1")
    universe = sorted({int(cluster) for cluster in clusters})
    if len(universe) < 2:
        raise ValueError("a development/evaluation split needs at least two clusters")
    order = np.random.default_rng(seed).permutation(len(universe))
    count = int(round(fraction * len(universe)))
    if not 1 <= count < len(universe):
        raise ValueError("the declared fraction leaves one side of the split empty")
    return frozenset(universe[int(index)] for index in order[:count])


# -------------------------------------------------------------- candidate pairs


def candidate_pairs(rows: Sequence[Mapping[str, Any]]) -> list[Pair]:
    """Every pair satisfying the pair rule and the declared phenotype gap.

    ``rows`` are single-substitution variants of one support, each carrying
    ``assay, cluster, wildtype_id, wt_residue, mut_residue, position, z,
    profile``. Grouping is by (assay, wild-type residue, mutant residue), which
    is exactly the equivalence class in which two members have identical length
    and identical composition.
    """

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["assay"], row["wt_residue"], row["mut_residue"]), []).append(row)
    pairs: list[Pair] = []
    for (assay, wt_residue, mut_residue), members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda row: int(row["position"]))
        for left, right in itertools.combinations(ordered, 2):
            if int(left["position"]) == int(right["position"]):
                raise ValueError(f"{assay}: two rows share position {left['position']}")
            if abs(float(left["z"]) - float(right["z"])) < GAP_THRESHOLD:
                continue
            better, worse = (left, right) if float(left["z"]) > float(right["z"]) else (right, left)
            pairs.append(
                Pair(
                    assay=assay,
                    cluster=int(left["cluster"]),
                    wildtype_id=str(left["wildtype_id"]),
                    wt_residue=wt_residue,
                    mut_residue=mut_residue,
                    position_better=int(better["position"]),
                    position_worse=int(worse["position"]),
                    z_better=float(better["z"]),
                    z_worse=float(worse["z"]),
                    profile_better=float(better["profile"]),
                    profile_worse=float(worse["profile"]),
                )
            )
    pairs.sort(key=lambda pair: pair.key)
    return pairs


def profile_match_threshold(
    candidates: Sequence[Pair],
    development: frozenset[int],
    *,
    percentile: float = PROFILE_MATCH_PERCENTILE,
) -> float:
    """The declared ``|delta s|`` threshold, read off development clusters only.

    Linear-interpolation percentile (NumPy's default), over every candidate pair
    of a development cluster that already satisfies the gap. Evaluation clusters
    are not read, so the threshold cannot be a function of the numbers it will
    later be applied to.
    """

    distances = [pair.profile_distance for pair in candidates if pair.cluster in development]
    if not distances:
        raise ValueError("no development-cluster candidate pair; the threshold is undefined")
    return float(np.percentile(np.asarray(distances, dtype=np.float64), percentile))


def retain_disjoint(
    candidates: Sequence[Pair],
    *,
    seed: int = COHORT_SEED,
    cap_per_cluster: int | None = None,
) -> list[Pair]:
    """Seeded greedy variant-disjoint retention within one tier.

    The canonical pair order is permuted once with ``seed`` and traversed once; a
    pair is retained when neither of its two variants is already used by a
    retained pair of this tier. ``cap_per_cluster`` stops accepting pairs for a
    family cluster once it holds that many, which is the declared way the
    expanded cohort is sized to the endpoint rather than to the maximum
    available. Nothing here reads a model score, so the rule is identical for
    every arm.
    """

    if cap_per_cluster is not None and cap_per_cluster < 1:
        raise ValueError("a per-cluster cap retains at least one pair")
    ordered = sorted(candidates, key=lambda pair: pair.key)
    if len({pair.key for pair in ordered}) != len(ordered):
        raise ValueError("candidate pairs must be unique under the canonical key")
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    used: set[tuple[str, str]] = set()
    counts: dict[int, int] = {}
    retained: list[Pair] = []
    for index in permutation:
        pair = ordered[int(index)]
        if cap_per_cluster is not None and counts.get(pair.cluster, 0) >= cap_per_cluster:
            continue
        left = (pair.assay, pair.mutation_better)
        right = (pair.assay, pair.mutation_worse)
        if left in used or right in used:
            continue
        used.update((left, right))
        counts[pair.cluster] = counts.get(pair.cluster, 0) + 1
        retained.append(pair)
    retained.sort(key=lambda pair: pair.key)
    return retained


def build_tiers(
    candidates: Sequence[Pair],
    development: frozenset[int],
    *,
    seed: int = COHORT_SEED,
    cap_per_cluster: int | None = None,
) -> dict[str, Any]:
    """Both declared tiers, with the threshold the primary tier used.

    ``T_open`` is the pair rule and the gap alone (the validity tier) and
    ``T_match`` adds the profile match (the primary tier). Disjointness is
    applied inside each tier separately, as declared, so a variant may appear in
    one pair of each tier; retaining ``T_match`` greedily inside ``T_open``
    instead would make the validity tier's composition a function of the profile
    match and destroy the ``D_open - D_match`` decomposition it exists to carry.
    """

    threshold = profile_match_threshold(candidates, development)
    matched = [pair for pair in candidates if pair.profile_distance <= threshold]
    return {
        "profile_match_threshold": threshold,
        "candidates": len(candidates),
        "candidates_matched": len(matched),
        TIER_OPEN: retain_disjoint(candidates, seed=seed, cap_per_cluster=cap_per_cluster),
        TIER_MATCH: retain_disjoint(matched, seed=seed, cap_per_cluster=cap_per_cluster),
    }


def variant_digest(mutants: Sequence[str]) -> str:
    """The cohort-format per-assay mutant digest used by the extraction path."""

    return hashlib.sha256("\n".join(mutants).encode()).hexdigest()


# ------------------------------------------------------- amendment_03 additions


def _pair_positions(pair: Mapping[str, Any]) -> tuple[int, int]:
    return int(pair["position_better"]), int(pair["position_worse"])


def position_rule_values(
    pairs: Sequence[Mapping[str, Any]],
    lengths: Mapping[str, int],
    *,
    variant: str = POSITION_RULE_TERMINUS,
) -> tuple[np.ndarray, np.ndarray]:
    """Position-only rule values for both members, oriented so higher is preferred.

    ``variant`` selects the amendment's spelling: ``terminus`` scores the negated
    distance to the nearer sequence end, ``midpoint`` scores the distance from the
    sequence midpoint. Neither reads a phenotype, a profile or a model.
    """

    if variant not in (POSITION_RULE_TERMINUS, POSITION_RULE_MIDPOINT):
        raise ValueError(f"unknown position rule variant {variant!r}")
    better, worse = [], []
    for pair in pairs:
        length = int(lengths[pair["assay"]])
        for position, sink in zip(_pair_positions(pair), (better, worse)):
            if not 1 <= position <= length:
                raise ValueError(
                    f"{pair['assay']}: position {position} outside a {length}-residue wild type"
                )
            if variant == POSITION_RULE_TERMINUS:
                sink.append(-float(min(position - 1, length - position)))
            else:
                sink.append(abs(float(position) - (length + 1) / 2.0))
    return np.asarray(better, dtype=np.float64), np.asarray(worse, dtype=np.float64)


def position_rule_agreement(
    pairs: Sequence[Mapping[str, Any]], lengths: Mapping[str, int]
) -> dict[str, Any]:
    """Verify that the amendment's two position spellings are one rule on this cohort."""

    terminus = position_rule_values(pairs, lengths, variant=POSITION_RULE_TERMINUS)
    midpoint = position_rule_values(pairs, lengths, variant=POSITION_RULE_MIDPOINT)
    left = terminus[0] - terminus[1]
    right = midpoint[0] - midpoint[1]
    strict = (left != 0.0) & (right != 0.0)
    return {
        "n_pairs": int(len(pairs)),
        "ties_terminus": int((left == 0.0).sum()),
        "ties_midpoint": int((right == 0.0).sum()),
        "orderings_disagreeing": int(((left > 0) != (right > 0))[strict].sum()),
        "note": SEMANTICS["position_rule"],
    }


def profile_vote(pair: Mapping[str, Any]) -> int:
    """``+1`` when the higher profile score belongs to the better-phenotype member."""

    difference = float(pair["profile_better"]) - float(pair["profile_worse"])
    return 0 if difference == 0.0 else (1 if difference > 0 else -1)


def position_vote(pair: Mapping[str, Any], lengths: Mapping[str, int]) -> int:
    """``+1`` when the site nearer a terminus belongs to the better-phenotype member."""

    better, worse = position_rule_values([pair], lengths)
    difference = float(better[0] - worse[0])
    return 0 if difference == 0.0 else (1 if difference > 0 else -1)


def balance_single_vote(
    pairs: Sequence[Mapping[str, Any]],
    vote: Callable[[Mapping[str, Any]], int],
    *,
    seed: int = COHORT_SEED,
    name: str,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Thin one stratum so that ``vote`` casts as many votes for as against.

    The two sides are permuted with ``seed`` and truncated to the smaller count.
    Neutral pairs cast no vote and are excluded rather than kept: under the
    declared direction convention an exact tie counts zero, so keeping them would
    pull the rule's direction accuracy *below* 0.5 and the tier would not be
    balanced in the sense the amendments declare.
    """

    concordant = [pair for pair in pairs if vote(pair) > 0]
    discordant = [pair for pair in pairs if vote(pair) < 0]
    keep = min(len(concordant), len(discordant))
    rng = np.random.default_rng(seed)
    selected: list[Mapping[str, Any]] = []
    for side in (concordant, discordant):
        order = rng.permutation(len(side))
        selected.extend(side[int(index)] for index in sorted(order[:keep]))
    record = {
        "balanced_votes": [name],
        "candidates": int(len(pairs)),
        f"{name}_concordant_available": int(len(concordant)),
        f"{name}_discordant_available": int(len(discordant)),
        f"{name}_neutral_excluded": int(len(pairs) - len(concordant) - len(discordant)),
        "retained_per_side": int(keep),
        "retained_pairs": int(2 * keep),
        "seed": int(seed),
    }
    return selected, record


def balance_two_votes(
    pairs: Sequence[Mapping[str, Any]],
    first: Callable[[Mapping[str, Any]], int],
    second: Callable[[Mapping[str, Any]], int],
    *,
    seed: int = COHORT_SEED,
    names: tuple[str, str],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Thin one stratum so that *both* votes are exactly balanced.

    Pairs fall into four sign cells. Taking equal counts from the two *diagonal*
    cells, ``(+,+)`` with ``(-,-)`` and ``(+,-)`` with ``(-,+)``, balances both
    marginals exactly for any split between the diagonals, which keeps more pairs
    than levelling all four cells to the smallest would. A pair that is neutral on
    either vote casts an incomplete vote and is excluded, for the same reason as
    in :func:`balance_single_vote`.
    """

    cells: dict[tuple[int, int], list[Mapping[str, Any]]] = {
        (+1, +1): [], (+1, -1): [], (-1, +1): [], (-1, -1): []
    }
    neutral = 0
    for pair in pairs:
        key = (first(pair), second(pair))
        if 0 in key:
            neutral += 1
            continue
        cells[key].append(pair)
    diagonals = (((+1, +1), (-1, -1)), ((+1, -1), (-1, +1)))
    rng = np.random.default_rng(seed)
    selected: list[Mapping[str, Any]] = []
    kept_per_diagonal = []
    for high, low in diagonals:
        keep = min(len(cells[high]), len(cells[low]))
        kept_per_diagonal.append(int(keep))
        for key in (high, low):
            order = rng.permutation(len(cells[key]))
            selected.extend(cells[key][int(index)] for index in sorted(order[:keep]))
    record = {
        "balanced_votes": list(names),
        "candidates": int(len(pairs)),
        "sign_cells_available": {
            f"{names[0]}{'+' if a > 0 else '-'}_{names[1]}{'+' if b > 0 else '-'}": len(rows)
            for (a, b), rows in cells.items()
        },
        "neutral_on_either_vote_excluded": int(neutral),
        "retained_per_diagonal": kept_per_diagonal,
        "retained_pairs": int(2 * sum(kept_per_diagonal)),
        "seed": int(seed),
    }
    return selected, record


def assert_vote_balanced(
    pairs: Sequence[Mapping[str, Any]], vote: Callable[[Mapping[str, Any]], int], *, name: str
) -> float:
    """Refuse unless ``vote``'s direction accuracy on ``pairs`` is exactly 0.5."""

    votes = [vote(pair) for pair in pairs]
    if not votes:
        raise ValueError(f"an empty stratum cannot be certified {name}-balanced")
    accuracy = sum(1 for value in votes if value > 0) / len(votes)
    if accuracy != 0.5:
        raise ValueError(
            f"the tier is not exactly {name}-balanced: {name} direction accuracy is "
            f"{accuracy!r} on {len(votes)} pairs, with "
            f"{sum(1 for value in votes if value == 0)} neutral pairs present"
        )
    return accuracy


#: ``amendment_05``'s capacity axis, as the manifest's arm list spells it. The
#: manifest's statistic says "five ProGen2 sizes" while the declared arm list
#: carries four, and the amendment may not alter the arm list, so the ladder
#: executed here is the declared four. ``progen2-base`` is on this host but is not
#: a declared S2 arm, and the repository's own
#: ``context_homologue.CAVEATS['progen2-base']`` rules it out of a scale contrast
#: in any case.
CAPACITY_LADDER: tuple[str, ...] = (
    "progen2-small",
    "progen2-medium",
    "progen2-large",
    "progen2-xlarge",
)

#: Published ProGen2 parameter counts. Spearman reads only their *order*, which is
#: unambiguous, so the trend statistic does not depend on these exact values.
CAPACITY_PARAMETERS: dict[str, float] = {
    "progen2-small": 151.1e6,
    "progen2-medium": 764.8e6,
    "progen2-large": 2779.4e6,
    "progen2-xlarge": 6443.6e6,
}

#: Declared limits that travel with every capacity-trend number.
CAPACITY_LIMITS: tuple[str, ...] = (
    "five released checkpoints of one family are releases, not training trials at "
    "controlled data and compute, so this identifies no causal effect of capacity",
    "parameter count is not the only thing that varies across a release ladder",
    "this axis carries its own declared multiplicity and does not enter the primary "
    "endpoint's family",
    "the repository's own arm caveats record 'staged non-member; 1024 budget imposed; "
    "no scale/window contrast' for progen2-large and progen2-xlarge; the manifest "
    "nonetheless declares this ladder a capacity axis, so the number is computed and "
    "the caveat is carried rather than either being dropped",
    "with four ladder points a Spearman coefficient takes only eleven distinct values, "
    "so its percentile interval is granular and the largest-minus-smallest contrast is "
    "the better-resolved reading",
)


def difference_verdict(low: float, high: float, *, margin: float = PRACTICAL_MARGIN) -> str:
    """The declared decision rule on a difference scale, where the null is zero."""

    if low > 0.0 or high < 0.0:
        return VERDICT_SUPPORTED
    if -margin < low and high < margin:
        return VERDICT_BOUNDED_NEGATIVE
    return VERDICT_UNRESOLVED


def _spearman_rows(values: np.ndarray, reference_ranks: np.ndarray) -> np.ndarray:
    """Spearman of each row of ``values`` against fixed ``reference_ranks``."""

    from scipy.stats import rankdata

    ranks = rankdata(values, axis=-1).astype(np.float64)
    ranks = ranks - ranks.mean(axis=-1, keepdims=True)
    target = reference_ranks - reference_ranks.mean()
    denominator = np.sqrt((ranks**2).sum(axis=-1) * (target**2).sum())
    numerator = (ranks * target).sum(axis=-1)
    return np.where(denominator > 0, numerator / np.maximum(denominator, 1e-300), np.nan)


def capacity_trend(
    indicators: Mapping[str, Sequence[float]],
    clusters: Sequence[int],
    *,
    parameters: Mapping[str, float] = CAPACITY_PARAMETERS,
    ladder: Sequence[str] = CAPACITY_LADDER,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int,
    alpha: float = BOOTSTRAP_ALPHA,
    margin: float = PRACTICAL_MARGIN,
) -> dict[str, Any]:
    """``amendment_05``: direction accuracy against log parameter count on one tier.

    Every ladder arm is read on the *same* pairs with the *same* cluster draw
    matrix, built exactly as :func:`group_bootstrap_accuracy` builds it from
    ``seed``, so passing the seed that tier's per-arm endpoints used makes the
    draws literally the same resample and the arms paired draw by draw. The
    largest-minus-smallest contrast is read off those same draws against the
    declared margin.
    """

    from scipy.stats import rankdata

    labels = np.asarray(clusters)
    ordered = [arm for arm in ladder if arm in indicators]
    if len(ordered) < 3:
        raise ValueError("a rank trend needs at least three ladder arms")
    rows = [np.asarray(indicators[arm], dtype=np.float64) for arm in ordered]
    if any(row.shape != (labels.size,) for row in rows):
        raise ValueError(
            "every ladder arm must be read on the same pairs; got lengths "
            f"{[int(row.size) for row in rows]} against {int(labels.size)} clusters"
        )
    matrix = np.vstack(rows)
    unique = np.unique(labels)
    floor = bootstrap_unit_floor(int(unique.size))
    log_parameters = np.log(np.asarray([parameters[arm] for arm in ordered], dtype=np.float64))
    reference = rankdata(log_parameters).astype(np.float64)
    point_accuracy = matrix.mean(axis=1)
    point = float(_spearman_rows(point_accuracy[None, :], reference)[0])
    record: dict[str, Any] = {
        "ladder": ordered,
        "parameters": {arm: float(parameters[arm]) for arm in ordered},
        "parameter_note": (
            "Spearman reads only the order of these counts, which is unambiguous, so the "
            "coefficient does not depend on their exact values"
        ),
        "accuracy_by_arm": {arm: float(value) for arm, value in zip(ordered, point_accuracy)},
        "spearman_vs_log_parameters": point,
        "draws": int(draws),
        "alpha": float(alpha),
        "margin": float(margin),
        "n_pairs": int(labels.size),
        **floor,
        "spearman_interval": None,
        "spearman_excludes_zero": None,
        "largest_minus_smallest": None,
        "limits": list(CAPACITY_LIMITS),
    }
    if floor["degenerate"]:
        return record
    sums = np.stack([[row[labels == c].sum() for c in unique] for row in matrix])
    sizes = np.asarray([float((labels == c).sum()) for c in unique])
    rng = np.random.default_rng(seed)
    index = rng.integers(0, unique.size, size=(draws, unique.size))
    counts = sizes[index].sum(axis=1)
    accuracies = np.stack([block[index].sum(axis=1) / counts for block in sums], axis=-1)
    drawn = _spearman_rows(accuracies, reference)
    finite = np.isfinite(drawn)
    # A draw in which every ladder arm scores identically has no monotone
    # ordering at all, so its coefficient is undefined rather than zero. Such a
    # draw is counted and excluded; imputing zero would assert "no trend" from an
    # absence of variation. If too few draws are defined, no interval is
    # published -- but the largest-minus-smallest contrast still is, because that
    # is the reading the declared margin is applied to.
    if finite.mean() >= MINIMUM_FINITE_DRAW_FRACTION:
        low, high = np.percentile(drawn[finite], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        record.update(
            spearman_interval=[float(low), float(high)],
            spearman_excludes_zero=bool(low > 0.0 or high < 0.0),
        )
    else:
        record["spearman_interval_withheld"] = (
            f"{int(finite.sum())} of {draws} draws had a defined coefficient, below the "
            f"{MINIMUM_FINITE_DRAW_FRACTION:.0%} floor; the ladder arms score too nearly "
            "identically for a rank trend to be defined on the resamples"
        )
    record["undefined_spearman_draws"] = int((~finite).sum())
    difference = accuracies[:, -1] - accuracies[:, 0]
    contrast_low, contrast_high = np.percentile(
        difference, [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )
    record.update(
        spearman_distinct_values_without_ties=int(
            1 + len(ordered) * (len(ordered) ** 2 - 1) // 6
        ),
        finite_draws=int(finite.sum()),
        largest_minus_smallest={
            "arms": [ordered[-1], ordered[0]],
            "difference": float(point_accuracy[-1] - point_accuracy[0]),
            "interval": [float(contrast_low), float(contrast_high)],
            "excludes_zero": bool(contrast_low > 0.0 or contrast_high < 0.0),
            "verdict": difference_verdict(float(contrast_low), float(contrast_high), margin=margin),
            "unit": "wild-type family cluster, same draw matrix as the per-arm endpoints",
        },
    )
    return record


def per_cluster_counts(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pair counts per family cluster, so cluster dominance can be checked.

    The resampling unit is the cluster, so a cluster holding many pairs sharpens
    its own contribution rather than multiplying units; this record is what makes
    that statement auditable instead of assumed.
    """

    if not pairs:
        return {"clusters": 0}
    counts: dict[int, int] = {}
    for pair in pairs:
        counts[int(pair["cluster"])] = counts.get(int(pair["cluster"]), 0) + 1
    values = np.asarray(sorted(counts.values(), reverse=True), dtype=np.float64)
    total = float(values.sum())
    return {
        "clusters": int(values.size),
        "pairs": int(total),
        "min": int(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "max": int(values.max()),
        "largest_cluster_share": float(values[0] / total),
        "largest_five_clusters_share": float(values[:5].sum() / total),
        "counts_by_cluster": {str(key): int(value) for key, value in sorted(counts.items())},
    }


#: ``amendment_04``'s reporting convention, spelled once so the two shares cannot
#: be confused for one another or for a third quantity.
ORACLE_CONVENTION = (
    "the oracle always selects the better-phenotype member: its direction accuracy is "
    "1.0 and its mean selected-member advantage is the full mean pair gap, not half of "
    "it. Chance is direction accuracy 0.5 and a mean selected-member advantage of 0.0"
)

PRACTICAL_SHARE_DEFINITIONS = {
    "accuracy_displacement_share": (
        "(D - 0.5) / (1.0 - 0.5), the fraction of the way from chance to the oracle on "
        "the direction-accuracy scale. Counts every pair equally, whatever its gap"
    ),
    "phenotype_gap_capture_share": (
        "mean(signed gap) / mean(gap), where the signed gap is +gap when the rule "
        "prefers the better member and -gap when it prefers the worse one. This is the "
        "fraction of the available phenotype separation the rule actually realises, so a "
        "rule that errs mainly on wide-gap pairs scores below its accuracy share"
    ),
}


def practical_size(
    pairs: Sequence[Mapping[str, Any]], indicators: Sequence[float]
) -> dict[str, Any]:
    """How much of the available phenotype separation a rule actually captures.

    Two separately named shares, never combined: see
    :data:`PRACTICAL_SHARE_DEFINITIONS` and :data:`ORACLE_CONVENTION`. They
    coincide when a rule's errors are independent of the gap and diverge when they
    are not, which is exactly why ``amendment_04`` requires both.
    """

    values = np.asarray(indicators, dtype=np.float64)
    gaps = np.asarray([float(pair["gap"]) for pair in pairs], dtype=np.float64)
    if values.shape != gaps.shape:
        raise ValueError("one indicator per pair is required")
    if values.size == 0:
        return {"n_pairs": 0}
    accuracy = float(values.mean())
    signed = np.where(values > 0, gaps, -gaps)
    return {
        "n_pairs": int(values.size),
        "mean_pair_gap_sd": float(gaps.mean()),
        "median_pair_gap_sd": float(np.median(gaps)),
        "accuracy_displacement_share": (accuracy - NULL_ACCURACY) / (1.0 - NULL_ACCURACY),
        "phenotype_gap_capture_share": float(signed.mean() / gaps.mean()),
        "selected_member_advantage_sd": float(signed.mean()),
        "oracle_member_advantage_sd": float(gaps.mean()),
        "oracle_convention": ORACLE_CONVENTION,
        "share_definitions": PRACTICAL_SHARE_DEFINITIONS,
    }


# ---------------------------------------------------------------------- endpoints


def direction_indicators(
    pairs: Sequence[Pair], better: Sequence[float], worse: Sequence[float]
) -> tuple[np.ndarray, int]:
    """Per-pair direction indicators and the number of exact ties.

    A tie is not "higher", so it counts zero, exactly as the endpoint is
    written. It is counted and reported rather than folded into a half, because a
    scoring rule that cannot separate two sequences at all is a different finding
    from one that orders them wrongly.
    """

    left = np.asarray(better, dtype=np.float64)
    right = np.asarray(worse, dtype=np.float64)
    if left.shape != (len(pairs),) or right.shape != (len(pairs),):
        raise ValueError("one score per pair member is required")
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise ValueError("pair scores must be finite")
    return (left > right).astype(np.float64), int((left == right).sum())


def group_bootstrap_accuracy(
    indicators: Sequence[float],
    clusters: Sequence[int],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int,
    alpha: float = BOOTSTRAP_ALPHA,
) -> dict[str, Any]:
    """Pooled direction accuracy with a cluster percentile interval and a verdict.

    Clusters are drawn with replacement and every pair of a drawn cluster is
    retained, so the reported statistic in each draw is the pooled pair fraction
    the endpoint defines. Below the package's eight-unit floor no interval is
    published and the record says why.
    """

    values = np.asarray(indicators, dtype=np.float64)
    labels = np.asarray(clusters)
    if values.shape != labels.shape or values.ndim != 1:
        raise ValueError("indicators and clusters must align as one-dimensional vectors")
    if draws < 1 or not 0.0 < alpha < 1.0:
        raise ValueError("invalid bootstrap parameters")
    unique = np.unique(labels)
    floor = bootstrap_unit_floor(int(unique.size))
    record: dict[str, Any] = {
        "n_pairs": int(values.size),
        "accuracy": float(values.mean()) if values.size else None,
        "draws": int(draws),
        "alpha": float(alpha),
        "unit": "wild-type family cluster at 50% identity",
        **floor,
        "interval": None,
        "verdict": VERDICT_DEGENERATE,
    }
    if values.size == 0:
        record["verdict"] = VERDICT_DEGENERATE
        record["degenerate_reason"] = "no retained pair in this stratum"
        record["degenerate"] = True
        return record
    if floor["degenerate"]:
        return record
    by_cluster = [values[labels == label] for label in unique]
    sums = np.asarray([block.sum() for block in by_cluster], dtype=np.float64)
    sizes = np.asarray([block.size for block in by_cluster], dtype=np.float64)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, unique.size, size=(draws, unique.size))
    statistic = sums[index].sum(axis=1) / sizes[index].sum(axis=1)
    low, high = np.percentile(statistic, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    record["interval"] = [float(low), float(high)]
    record["verdict"] = verdict_of(float(low), float(high))
    record["cluster_pair_counts"] = {str(label): int(size) for label, size in zip(unique, sizes)}
    return record


def verdict_of(low: float, high: float) -> str:
    """The declared decision rule, applied to one interval and nothing else."""

    if low > NULL_ACCURACY or high < NULL_ACCURACY:
        return VERDICT_SUPPORTED
    if (
        NULL_ACCURACY - PRACTICAL_MARGIN < low
        and high < NULL_ACCURACY + PRACTICAL_MARGIN
    ):
        return VERDICT_BOUNDED_NEGATIVE
    return VERDICT_UNRESOLVED


def paired_accuracy_contrast(
    left_indicators: Sequence[float],
    left_clusters: Sequence[int],
    right_indicators: Sequence[float],
    right_clusters: Sequence[int],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int,
    alpha: float = BOOTSTRAP_ALPHA,
) -> dict[str, Any]:
    """``D_left - D_right`` with the family cluster resampled once per draw.

    One function serves both declared kinds of contrast. For two *tiers* of
    pairs (``D_open - D_match``) the resampling unit is a cluster that
    contributes at least one pair to each side; a cluster present on one side
    only cannot enter a paired contrast, and including it would make the
    difference partly a difference of populations, so both accuracies inside the
    record are computed on the common cluster set and reported beside the
    unrestricted ones rather than replacing them. For two *rules* on the same
    pairs (``D_model - D_profile``, ``D_model - D_position``) the two cluster
    vectors are identical, the common set is everything, and the pairing is
    exact pair by pair.
    """

    left_values = np.asarray(left_indicators, dtype=np.float64)
    left_labels = np.asarray(left_clusters)
    right_values = np.asarray(right_indicators, dtype=np.float64)
    right_labels = np.asarray(right_clusters)
    if left_values.shape != left_labels.shape or right_values.shape != right_labels.shape:
        raise ValueError("indicators and clusters must align on each side")
    common = np.array(sorted(set(left_labels.tolist()) & set(right_labels.tolist())))
    floor = bootstrap_unit_floor(int(common.size))
    record: dict[str, Any] = {
        "unit": "wild-type family cluster contributing to both sides",
        "draws": int(draws),
        "alpha": float(alpha),
        **floor,
        "difference": None,
        "interval": None,
        "excludes_zero": None,
        "left_accuracy_on_common_clusters": None,
        "right_accuracy_on_common_clusters": None,
    }
    if floor["degenerate"]:
        return record
    left_mask = np.isin(left_labels, common)
    right_mask = np.isin(right_labels, common)
    left_sums = np.asarray([left_values[left_labels == c].sum() for c in common])
    left_sizes = np.asarray([float((left_labels == c).sum()) for c in common])
    right_sums = np.asarray([right_values[right_labels == c].sum() for c in common])
    right_sizes = np.asarray([float((right_labels == c).sum()) for c in common])
    d_left = float(left_values[left_mask].sum() / left_mask.sum())
    d_right = float(right_values[right_mask].sum() / right_mask.sum())
    rng = np.random.default_rng(seed)
    index = rng.integers(0, common.size, size=(draws, common.size))
    drawn = (left_sums[index].sum(axis=1) / left_sizes[index].sum(axis=1)) - (
        right_sums[index].sum(axis=1) / right_sizes[index].sum(axis=1)
    )
    low, high = np.percentile(drawn, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    record.update(
        difference=d_left - d_right,
        interval=[float(low), float(high)],
        excludes_zero=bool(low > 0.0 or high < 0.0),
        left_accuracy_on_common_clusters=d_left,
        right_accuracy_on_common_clusters=d_right,
        n_pairs_left=int(left_mask.sum()),
        n_pairs_right=int(right_mask.sum()),
    )
    return record


def magnitude_boundaries(
    development_magnitudes: Sequence[float], *, bins: int = MAGNITUDE_BINS
) -> list[float]:
    """Interior quantile boundaries of ``|delta log-likelihood|``, fixed on development.

    Returned for one arm at a time: nats are not comparable across arms, so one
    shared boundary set would put different arms' decisive and indecisive pairs
    in the same bin.
    """

    values = np.asarray(development_magnitudes, dtype=np.float64)
    if values.ndim != 1 or values.size < bins:
        raise ValueError(f"{bins} quantile bins need at least {bins} development pairs")
    if not np.isfinite(values).all():
        raise ValueError("magnitudes must be finite")
    cuts = np.percentile(values, [100.0 * k / bins for k in range(1, bins)])
    return [float(value) for value in cuts]


def magnitude_strata(
    magnitudes: Sequence[float],
    indicators: Sequence[float],
    clusters: Sequence[int],
    boundaries: Sequence[float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int,
    alpha: float = BOOTSTRAP_ALPHA,
) -> list[dict[str, Any]]:
    """Direction accuracy within each declared magnitude bin, with its interval.

    Descriptive and secondary by declaration: it carries no verdict of its own
    for the primary question, and the per-bin verdict field is the same interval
    rule applied for readability only.
    """

    values = np.asarray(magnitudes, dtype=np.float64)
    edges = [-np.inf, *[float(value) for value in boundaries], np.inf]
    strata: list[dict[str, Any]] = []
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        mask = (values > low) & (values <= high) if index else values <= high
        record = group_bootstrap_accuracy(
            np.asarray(indicators, dtype=np.float64)[mask],
            np.asarray(clusters)[mask],
            draws=draws,
            seed=seed + index,
            alpha=alpha,
        )
        record.update(
            bin_index=index,
            lower_nats=None if index == 0 else float(low),
            upper_nats=None if index == len(edges) - 2 else float(high),
            magnitude_median_nats=float(np.median(values[mask])) if mask.any() else None,
        )
        strata.append(record)
    return strata


def magnitude_summary(magnitudes: Sequence[float]) -> dict[str, Any]:
    """The ``|delta log-likelihood|`` distribution in nats, which a rank cannot carry."""

    values = np.asarray(magnitudes, dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    quantiles = [1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0]
    return {
        "n": int(values.size),
        "mean_nats": float(values.mean()),
        "sd_nats": float(values.std(ddof=1)) if values.size > 1 else None,
        "min_nats": float(values.min()),
        "max_nats": float(values.max()),
        "percentiles_nats": {
            f"p{q:g}": float(value) for q, value in zip(quantiles, np.percentile(values, quantiles))
        },
        "fraction_below_0.1_nats": float((values < 0.1).mean()),
        "fraction_below_1_nat": float((values < 1.0).mean()),
    }
