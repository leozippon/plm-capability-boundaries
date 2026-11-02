"""The matched-candidate discrimination cells, as ledger rows rather than as print.

The matched-candidate experiment (E21) and its supervised-readout companion
(E22) are reported in Results 1, in the Discussion, in a Methods subsection and
in one Supplementary Note with two tables. None of their numbers had a ledger
row, so the manuscript's own checker could only match each printed value against
whatever unrelated row happened to share the number. This module closes that
gap for exactly the cells the manuscript prints and for nothing else: a row for
a number the manuscript never writes can only take an attribution away from one
it does.

Four things about this experiment decide how its rows are shaped.

The tier is the support, not a qualifier on the endpoint. Each tier is a
different retained pair set, built by a different declared construction, and the
pre-registration's interpretation bounds require every quoted accuracy to name
its tier and to say what the tier neutralises. Putting that on the support
description rather than on the endpoint name is what makes the one-support-per-
comparison condition do useful work here: two accuracies from two tiers are two
populations, and the checker should say so. The same reasoning gives each
magnitude quintile its own support, because the quintile boundaries are fixed
per arm on development clusters and so partition a different pair set for every
arm.

The declaration's two supports are never pooled, and each row names its own. The
expanded support was rescored for the nine declared arms and carries likelihood
magnitudes in nats; the retained 33-arm panel reads within-assay likelihood
ranks that were already held, and carries the supervised comparison. Amendment 1
forbids any magnitude reading on the rank scale, so no row sourced from the
panel carries one.

A quoted span is a ``range_bound`` pair, which the checker verifies as a pair
against the two rows whose values are its ends. Those rows point at the same
artifact fields the per-arm estimate rows point at, because a span over a panel
*is* two of that panel's members. That is the one place this module writes two
identifiers over one artifact field, and it is deliberate: the estimate row is
what a sentence quoting one arm is attributed to, and the ``range_bound`` pair
is what a sentence quoting the panel's extent is verified against. The kinds
keep them apart, and neither can win the other's attribution.

The ledger's verdict vocabulary has four outcomes and the declaration has three.
``supported`` and ``unresolved`` carry over unchanged. The declaration's
``bounded negative`` --- an interval lying inside the declared practical margin
of the null --- is a measurement that resolved against that margin rather than a
statement about the instrument, so it is recorded as ``not_detected`` and the
declaration's own word is kept in the claim. Nothing is recorded as
``measurement_limited``: the measurement floor this cohort carries bounds the
attainable accuracy for every arm alike and is stated on the supports, not as a
verdict on one arm.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.capability.core.evidence_ledger import LedgerError, resolve
# The emit contract --- how a row carries its locator, digest and pointer --- is
# the follow-up module's. A second copy here would be a second place for it to
# drift from the ledger's requirements.
from scripts.capability.reporting.followup_quantities import _emitter

ROOT = "results/R1/matched_discrimination_20261102"
#: The expanded support: nine declared arms rescored on local GPUs, which is the
#: only support that can carry likelihood magnitudes in nats.
EXPANDED = f"{ROOT}/expanded/expanded_variants/analysis/rows.json"
#: What that support's cohort covered and cost to score.
COHORT = f"{ROOT}/expanded/expanded_variants/cohort/cohort_summary.json"
#: The constants that cohort applied, recorded beside the rules that used them.
CONTRACT = f"{ROOT}/expanded/expanded_variants/cohort/contract.json"
#: The retained 33-arm panel, read on the within-assay likelihood rank scale.
PANEL = f"{ROOT}/retained_panel/matched_discrimination.json"
#: The panel's amendment-3 addendum, which holds the position control and the
#: construction check that over-corrected.
PANEL_A3 = f"{ROOT}/retained_panel/amendment_03.json"
#: The pre-registration itself, which is where the split fractions are declared.
MANIFEST = "configs/matched_discrimination_manifest.json"

FAMILY = "matched_discrimination"
ACCURACY = "dimensionless direction accuracy"
POINTS = "direction-accuracy points"
SD = "within-assay standard deviations"
SHARE = "dimensionless fraction"
NATS = "nats"
PAIRS = "pairs"
CLUSTERS = "wild-type clusters"
RHO = "dimensionless Spearman"
CLUSTER_UNIT = "wild-type family cluster at 50% identity"

#: The ledger's four-outcome vocabulary against the declaration's three.
VERDICTS = {"supported": "supported", "unresolved": "unresolved",
            "bounded_negative": "not_detected"}

#: Each tier, with the prose name the manuscript prints it under and what the
#: tier neutralises. The second half travels on the support description, because
#: the interpretation bounds require every quoted accuracy to carry it and an
#: endpoint name cannot.
TIERS = {
    "T_open": ("open tier",
               "the pair rule and the phenotype gap only, with neither confound neutralised"),
    "T_match": ("matched tier",
                "the absolute difference of the independent-site profile log-odds additionally "
                "matched to at or below the declared threshold, which bounds that difference's "
                "magnitude and leaves its sign free"),
    "T_balanced": ("profile-balanced tier",
                   "the independent-site profile's directional vote cancelled exactly by "
                   "construction, with equal counts of pairs whose higher profile score belongs "
                   "to the better-phenotype and to the worse-phenotype member"),
    "T_position_balanced": ("position-balanced tier",
                            "the position-only rule's directional vote cancelled exactly by "
                            "construction, that rule preferring the member substituted nearer a "
                            "sequence terminus"),
    "T_double_balanced": ("doubly balanced tier",
                          "the independent-site profile's and the position-only rule's directional "
                          "votes each cancelled exactly by construction, which is the decisive "
                          "tier of the declaration"),
}
TIER_ORDER = ("T_open", "T_match", "T_balanced", "T_position_balanced", "T_double_balanced")

#: The nine declared arms of the expanded support, keyed by the name the artifact
#: uses, carrying the repository name the manuscript writes and whether the arm
#: is pooled with the autoregressive panel. The masked arm is reported beside
#: them and never pooled, so the roster carries that rather than leaving it to a
#: reader, and the artifact's own flag is checked against it.
ARMS = {
    "ProtGPT2": ("protgpt2", True),
    "progen2-small": ("progen2-small", True),
    "progen2-medium": ("progen2-medium", True),
    "progen2-large": ("progen2-large", True),
    "progen2-xlarge": ("progen2-xlarge", True),
    "progen3-112m": ("progen3-112m", True),
    "progen3-3b": ("progen3-3b", True),
    "RITA_xl": ("rita-xl", True),
    "esm2_t33_650M_UR50D": ("esm2-650m", False),
}

#: The two unfitted reference rules the construction checks and the headline
#: contrast are read against.
RULES = {
    "profile": ("independent-site profile",
                "the unfitted independent-site evolutionary log-odds, which is the confound the "
                "profile-balanced construction cancels and the declaration's construction check"),
    "position": ("position-only rule",
                 "the unfitted rule preferring the member substituted nearer a sequence terminus, "
                 "which the declaration names the strongest alternative explanation of a "
                 "surviving direction"),
}

#: The practical-size readings, each with the definition the reporting
#: convention requires beside it. Amendment 4 forbids combining them into one
#: number, because three different quantities had been called a share.
SHARES = {
    "accuracy_displacement_share": (
        SHARE, "the fraction of the way from chance to the oracle on the direction-accuracy "
               "scale, counting every pair equally whatever its gap"),
    "phenotype_gap_capture_share": (
        SHARE, "the mean signed phenotype gap divided by the mean gap, which is the fraction of "
               "the available phenotype separation the rule realises and falls below the accuracy "
               "share when the rule errs on wide-gap pairs"),
    "selected_member_advantage_sd": (
        SD, "the mean within-assay phenotype advantage of the member the rule selects"),
}

#: The declared constants the manuscript prints, read from the record that
#: applied them rather than from the declaration that states them, so a constant
#: cannot be quoted from a manifest the executed analysis did not use.
CONSTANTS = (
    ("phenotype_gap_threshold_sd", SD,
     "declared minimum measured phenotype gap between the two members of a retained pair, fixed "
     "as a constant and not tuned on any data"),
    ("null_accuracy", ACCURACY,
     "the null value of every matched-candidate direction accuracy, which is the accuracy of a "
     "rule that orders these pairs at chance"),
    ("practical_margin", POINTS,
     "declared practical margin of the matched-candidate decision rule: an interval lying within "
     "this distance of the null is a bounded negative, and no result closer to the null is read "
     "as an effect"),
    ("bootstrap_draws", "bootstrap draws",
     "declared paired cluster-bootstrap draw count behind every matched-candidate interval"),
    ("split_seed", "split seed",
     "declared seed of the cluster permutation that assigns each wild-type family to the "
     "development or the evaluation stratum"),
)


def add_matched_discrimination_quantities(ledger) -> None:
    emit = _emitter(ledger)
    expanded = ledger.artifacts.json(EXPANDED)
    _schema(ledger, expanded)
    supports = _supports(ledger, expanded)
    _declared_constants(ledger, emit)
    _tier_sizes(ledger, emit, expanded, supports)
    _arm_accuracies(ledger, emit, expanded, supports)
    _reference_rules(ledger, emit, expanded, supports)
    _position_margin_span(emit, expanded, supports)
    _practical_size(emit, expanded, supports)
    _capacity_axis(ledger, emit, expanded)
    _magnitude_strata(ledger, emit, expanded)
    _scoring_cohort(ledger, emit, expanded)
    _panel(ledger, emit)


# --------------------------------------------------------------------------- #
# What the artifacts have to say before any row is written.
# --------------------------------------------------------------------------- #

def _schema(ledger, expanded: Mapping) -> None:
    """Refuse an artifact that is not the declared analysis of this cohort.

    Every row below is positioned by tier, arm and bin names that no field
    declares, so an artifact whose declaration, support or decisive tier has
    moved is a different reading and its cells are not the ones the manuscript
    prints.
    """
    if expanded.get("declaration") != "matched_discrimination_20261102":
        raise LedgerError(f"{EXPANDED}: not the declared matched-discrimination analysis")
    if expanded.get("support") != "expanded_variants":
        raise LedgerError(
            f"{EXPANDED}: support is {expanded.get('support')!r}, not the expanded one")
    if expanded.get("decisive_tier") != "T_double_balanced":
        raise LedgerError(
            f"{EXPANDED}: the decisive tier is {expanded.get('decisive_tier')!r}, so the tier the "
            "manuscript reports as decisive is not the one the analysis declares")
    if set(expanded["arms"]) != set(ARMS):
        raise LedgerError(f"{EXPANDED}: arms are {sorted(expanded['arms'])}, not the declared nine")
    if set(expanded["tier_sizes"]) != set(TIERS):
        raise LedgerError(f"{EXPANDED}: tiers are {sorted(expanded['tier_sizes'])}")
    for arm, (name, autoregressive) in ARMS.items():
        pooled = expanded["arms"][arm]["pooled_with_autoregressive_arms"]
        if pooled is not autoregressive:
            raise LedgerError(
                f"{EXPANDED}: arm {name} records pooled_with_autoregressive_arms={pooled}, which "
                "disagrees with the roster this module declares")
    panel = ledger.artifacts.json(PANEL)
    if panel.get("support") != "S1_retained_panel" or len(panel["E22"]["arms"]) != 33:
        raise LedgerError(f"{PANEL}: not the 33-arm retained panel of this declaration")


def _supports(ledger, expanded: Mapping) -> dict[str, str]:
    """One support per tier, carrying the counts and the reading bounds.

    The adversarial-selection bound and the marginal-balance bound are stated
    here rather than in each claim, because they are properties of the measured
    population: a tier is a set of pairs chosen so that the neutralised confound
    cannot discriminate, and no endpoint name can carry that.
    """
    declared = {}
    for tier in TIER_ORDER:
        sizes = expanded["tier_sizes"][tier]
        name, construction = TIERS[tier]
        declared[tier] = ledger.declare_support(
            f"{FAMILY}/expanded/{tier}",
            f"Matched-candidate {name} of the expanded support: {sizes['pairs_evaluation']:,} "
            f"evaluation pairs of single substitutions in {sizes['clusters_evaluation']} wild-type "
            "family clusters at 50% identity. Each pair is two single mutants of one wild type in "
            "one assay carrying the same wild-type residue and the same mutant residue at two "
            "different positions, so length and amino-acid composition are identical by "
            "construction, and the two members' measured within-assay standardized phenotypes "
            f"differ by at least one standard deviation. The tier adds {construction}. Requiring a "
            "large measured gap while neutralising a confound selects the pairs on which that "
            "confound cannot discriminate, so an accuracy here is not an estimate of how a model "
            "orders a randomly drawn pair and is not comparable with a general mutation-ranking "
            "accuracy; a sign-balanced vote is equalised over the whole tier and not within every "
            "cluster, assay or subgroup. The endpoint unit is the pair and the inference unit the "
            "wild-type family cluster. The phenotype gaps are observed without replicates, which "
            "bounds the attainable accuracy below a perfect ordering by an amount this cohort "
            "cannot estimate")
    return declared


# --------------------------------------------------------------------------- #
# The declared constants.
# --------------------------------------------------------------------------- #

def _declared_constants(ledger, emit) -> None:
    support = ledger.declare_support(
        f"{FAMILY}/declaration",
        "Declared design values of the matched-candidate pre-registration, read from the frozen "
        "cohort contract that applied them and from the manifest clause that states the split. "
        "They are declarations rather than readings of any measured population, and no threshold "
        "among them was tuned on data")
    for name, unit, claim in CONSTANTS:
        emit(CONTRACT, ("declared_constants", name),
             identifier=f"{FAMILY}/declaration/{name}", claim=claim, family=FAMILY,
             support=support, unit=unit, kind="constant")
    fractions = resolve(ledger.artifacts.json(MANIFEST),
                        ("cohort", "development_evaluation_split", "fractions"))
    if len(fractions) != 2 or abs(sum(fractions) - 1.0) > 1e-12:
        raise LedgerError(
            f"{MANIFEST}: the declared split fractions are {fractions}, which are not one "
            "complementary pair, so neither can be quoted as the stratum it names")
    for index, (name, stratum) in enumerate((("development_fraction", "development"),
                                             ("evaluation_fraction", "evaluation"))):
        emit(MANIFEST, ("cohort", "development_evaluation_split", "fractions", f"[{index}]"),
             identifier=f"{FAMILY}/declaration/{name}",
             claim=f"declared fraction of wild-type family clusters assigned to the {stratum} "
                   "stratum of the matched-candidate split",
             family=FAMILY, support=support, unit=SHARE, kind="constant")
    emit(COHORT, ("profile_match_threshold_nats",),
         identifier=f"{FAMILY}/declaration/profile_match_threshold_nats",
         claim="the matched tier's profile-match threshold on the expanded support: the declared "
               "percentile of the absolute independent-site profile-score difference over "
               "gap-satisfying development-cluster candidates, fixed there and applied unchanged "
               "to evaluation clusters",
         family=FAMILY, support=support, unit=NATS, kind="artifact_leaf")
    emit(COHORT, ("cap_per_cluster",), identifier=f"{FAMILY}/declaration/cap_per_cluster",
         claim="sizing cap on retained pairs per wild-type family cluster: a variance-control rule "
               "introduced during execution and recorded as such, not a declared threshold",
         family=FAMILY, support=support, unit="pairs per cluster", kind="constant")


# --------------------------------------------------------------------------- #
# Tier sizes and how the pairs sit across clusters.
# --------------------------------------------------------------------------- #

def _tier_sizes(ledger, emit, expanded: Mapping, supports: Mapping[str, str]) -> None:
    sizes = expanded["tier_sizes"]
    for tier in TIER_ORDER:
        name = TIERS[tier][0]
        emit(EXPANDED, ("tier_sizes", tier, "pairs_evaluation"),
             identifier=f"{FAMILY}/expanded/{tier}/evaluation_pairs",
             claim=f"retained evaluation pairs of the {name}", family=FAMILY,
             support=supports[tier], unit=PAIRS, kind="support_count")
        emit(EXPANDED, ("tier_sizes", tier, "clusters_evaluation"),
             identifier=f"{FAMILY}/expanded/{tier}/evaluation_clusters",
             claim=f"wild-type family clusters contributing evaluation pairs to the {name}, which "
                   "is the resampling unit count behind its intervals",
             family=FAMILY, support=supports[tier], unit=CLUSTERS, kind="support_count")
    # Cluster dominance is quoted once, as the worst case over the five tiers,
    # so one row is written for each share rather than five. A row for a tier
    # the manuscript does not quote could only dilute the attribution of the one
    # it does.
    for field, unit, claim in (
        ("largest_cluster_share", SHARE,
         "evaluation pairs any single cluster contributes to its tier, as a share, at the tier "
         "where that share is largest"),
        ("largest_five_clusters_share", SHARE,
         "evaluation pairs the five largest clusters contribute to their tier, as a share, at the "
         "tier where that share is largest"),
    ):
        worst = max(TIER_ORDER, key=lambda tier: sizes[tier][
            "per_cluster_pair_counts_evaluation"][field])
        emit(EXPANDED, ("tier_sizes", worst, "per_cluster_pair_counts_evaluation", field),
             identifier=f"{FAMILY}/expanded/per_cluster/{field}_worst_tier",
             claim=f"largest share of {claim}, which is the {TIERS[worst][0]}. Because the cluster "
                   "is the resampling unit, within-cluster counts affect a cluster's own precision "
                   "rather than the across-cluster inference",
             family=FAMILY, support=supports[worst], unit=unit, kind="artifact_leaf")
    for name, field, unit, claim in (
        ("tier_pairs_span", "pairs_evaluation", PAIRS,
         "retained evaluation pairs over the five matched-candidate tiers of the expanded support"),
        ("tier_clusters_span", "clusters_evaluation", CLUSTERS,
         "wild-type family clusters contributing evaluation pairs over the five matched-candidate "
         "tiers of the expanded support"),
    ):
        _span(emit, f"{FAMILY}/expanded/{name}", claim,
              [(EXPANDED, ("tier_sizes", tier, field), sizes[tier][field], supports[tier])
               for tier in TIER_ORDER], unit)
    _span(emit, f"{FAMILY}/expanded/tier_median_per_cluster_span",
          "median evaluation pairs per contributing cluster over the five matched-candidate tiers "
          "of the expanded support",
          [(EXPANDED, ("tier_sizes", tier, "per_cluster_pair_counts_evaluation", "median"),
            sizes[tier]["per_cluster_pair_counts_evaluation"]["median"], supports[tier])
           for tier in TIER_ORDER], PAIRS)


def _span(emit, prefix: str, claim: str, members: Sequence[tuple], unit: str) -> None:
    """A quoted span, as the ``min``/``max`` pair the checker verifies it against.

    Each end points at the artifact field of the member that *is* that end, so
    neither bound is a number this module computed. The members are the panel
    the span is taken over, and a panel of one would make the span meaningless.
    """
    if len(members) < 2:
        raise LedgerError(f"{prefix}: a span needs at least two members")
    for bound, member in (("min", min(members, key=lambda item: float(item[2]))),
                          ("max", max(members, key=lambda item: float(item[2])))):
        path, pointer, value, support = member
        emit(path, pointer, identifier=f"{prefix}/{bound}",
             claim=f"{'smallest' if bound == 'min' else 'largest'} {claim}",
             family=FAMILY, support=support, unit=unit, kind="range_bound", value=value)


# --------------------------------------------------------------------------- #
# The per-arm, per-tier table the Supplementary Note prints.
# --------------------------------------------------------------------------- #

def _accuracy(emit, path: str, pointer: tuple[str, ...], block: Mapping, identifier: str,
              claim: str, support: str, *, field: str = "accuracy", draws: str = "draws",
              unit: str | None = "unit", verdict: str = "verdict",
              seed: Sequence[int] = ()) -> None:
    """One direction accuracy with the interval and verdict the artifact records.

    ``unit`` names the field the record carries its resampling unit in, or is
    ``None`` where the record states that unit once for the whole analysis
    rather than on every block; both forms are read and neither is written here.
    """
    recorded = block[verdict]
    if isinstance(recorded, Mapping):
        recorded = recorded["verdict"]
    if recorded not in VERDICTS:
        raise LedgerError(
            f"{path}: {identifier} reports verdict {recorded!r}, which is outside the "
            "declaration's three outcomes")
    emit(path, pointer + (field,), identifier=identifier, claim=claim, family=FAMILY,
         support=support, unit=ACCURACY, kind="estimate", interval=block["interval"],
         draws=block[draws], verdict=VERDICTS[recorded], seed=seed,
         resampling_unit=CLUSTER_UNIT if unit is None else block[unit])


def _arm_accuracies(ledger, emit, expanded: Mapping, supports: Mapping[str, str]) -> None:
    del ledger
    for arm, (name, autoregressive) in sorted(ARMS.items(), key=lambda item: item[1][0]):
        aside = ("" if autoregressive else
                 ". Reported beside the autoregressive arms as a contrast across a different "
                 "training objective and scoring rule, never pooled with them and never as an "
                 "attribution to either difference")
        for tier in TIER_ORDER:
            pointer = ("arms", arm, "tiers", tier, "evaluation")
            _accuracy(emit, EXPANDED, pointer, resolve(expanded, pointer),
                      f"{FAMILY}/expanded/{tier}/{name}/direction_accuracy",
                      f"direction accuracy of {name}'s native likelihood on the "
                      f"{TIERS[tier][0]}: the fraction of pairs whose better-phenotype member "
                      f"receives the higher score{aside}",
                      supports[tier])


def _reference_rules(ledger, emit, expanded: Mapping, supports: Mapping[str, str]) -> None:
    del ledger
    for rule, (name, role) in sorted(RULES.items()):
        for tier in TIER_ORDER:
            pointer = ("reference_rules", rule, "tiers", tier, "evaluation")
            _accuracy(emit, EXPANDED, pointer, resolve(expanded, pointer),
                      f"{FAMILY}/expanded/{tier}/rule_{rule}/direction_accuracy",
                      f"direction accuracy of the {name} on the {TIERS[tier][0]}: {role}",
                      supports[tier])
    certified = resolve(expanded, ("balanced_tier_construction", "T_double_balanced",
                                   "balance_per_stratum", "evaluation",
                                   "certified_direction_accuracy"))
    for rule, (name, _) in sorted(RULES.items()):
        if float(certified[rule]) != 0.5:
            raise LedgerError(
                f"{EXPANDED}: the doubly balanced tier certifies the {rule} vote at "
                f"{certified[rule]}, so it is not cancelled exactly and the tier the manuscript "
                "reports as doubly balanced is not one")
        emit(EXPANDED, ("balanced_tier_construction", "T_double_balanced", "balance_per_stratum",
                        "evaluation", "certified_direction_accuracy", rule),
             identifier=f"{FAMILY}/expanded/T_double_balanced/certified_{rule}_vote",
             claim=f"the {name}'s direction accuracy on the doubly balanced tier, certified by the "
                   "construction rather than estimated: the tier retains equal counts of pairs in "
                   "which the rule prefers the better-phenotype and the worse-phenotype member, "
                   "and the analysis refuses unless the balance is exact",
             family=FAMILY, support=supports["T_double_balanced"], unit=ACCURACY,
             kind="artifact_leaf")


def _position_margin_span(emit, expanded: Mapping, supports: Mapping[str, str]) -> None:
    """The margin the autoregressive arms hold over the position-only rule.

    This is the quantity the central claim turns on, and the manuscript quotes
    its extent over the eight autoregressive arms rather than over nine members
    of a pooled panel: the masked arm is never pooled with them, so it is not a
    member of this span.
    """
    key = "D_model_minus_D_position_T_double_balanced"
    members = []
    for arm, (name, autoregressive) in ARMS.items():
        if not autoregressive:
            continue
        contrast = resolve(expanded, ("arms", arm, "contrasts", key))
        if not contrast["excludes_zero"]:
            raise LedgerError(
                f"{EXPANDED}: {name}'s margin over the position-only rule on the doubly balanced "
                "tier does not exclude zero, so the manuscript's statement that every "
                "autoregressive arm exceeds that rule is not what the analysis records")
        members.append((EXPANDED, ("arms", arm, "contrasts", key, "difference"),
                        contrast["difference"], supports["T_double_balanced"]))
    _span(emit, f"{FAMILY}/expanded/T_double_balanced/model_minus_position_span",
          "cluster-paired margin of an autoregressive arm's direction accuracy over the "
          "position-only rule on the doubly balanced tier, where that rule is held at chance by "
          "construction, over the eight autoregressive arms",
          members, POINTS)


# --------------------------------------------------------------------------- #
# Practical size, under the one oracle convention the amendment fixes.
# --------------------------------------------------------------------------- #

def _practical_size(emit, expanded: Mapping, supports: Mapping[str, str]) -> None:
    """The two separately named shares, and the oracle they are read against.

    Amendment 4 records that three different quantities had been called a share
    and requires the accuracy-displacement share and the phenotype-gap-capture
    share to be reported as separately named quantities with the oracle
    convention stated. Each row therefore carries its own definition, and no row
    combines them.
    """
    tier = "T_double_balanced"
    oracle = None
    for arm in ("progen2-xlarge", "esm2_t33_650M_UR50D"):
        name = ARMS[arm][0]
        pointer = ("arms", arm, "tiers", tier, "practical_size_evaluation")
        block = resolve(expanded, pointer)
        for field, (unit, definition) in sorted(SHARES.items()):
            emit(EXPANDED, pointer + (field,),
                 identifier=f"{FAMILY}/expanded/{tier}/{name}/{field}",
                 claim=f"{name} on the doubly balanced tier: {definition}. The oracle always "
                       "selects the better-phenotype member, so it orders every pair correctly and "
                       "its mean selected-member advantage is the full mean pair gap",
                 family=FAMILY, support=supports[tier], unit=unit, kind="artifact_leaf")
        if oracle is None:
            oracle = block["oracle_member_advantage_sd"]
            emit(EXPANDED, pointer + ("oracle_member_advantage_sd",),
                 identifier=f"{FAMILY}/expanded/{tier}/oracle_member_advantage_sd",
                 claim="the mean within-assay phenotype advantage an oracle realises on the doubly "
                       "balanced tier, which is the full mean pair gap because the oracle always "
                       "selects the better-phenotype member. It is a property of the tier and not "
                       "of any arm",
                 family=FAMILY, support=supports[tier], unit=SD, kind="artifact_leaf")
        elif float(block["oracle_member_advantage_sd"]) != float(oracle):
            raise LedgerError(
                f"{EXPANDED}: the doubly balanced tier records two oracle advantages, so the "
                "practical-size readings of its arms are not read against one convention")


# --------------------------------------------------------------------------- #
# The declared capacity axis.
# --------------------------------------------------------------------------- #

def _capacity_axis(ledger, emit, expanded: Mapping) -> None:
    tier = "T_double_balanced"
    axis = resolve(expanded, ("capacity_axis", tier))
    if not axis["status"].startswith("prospective"):
        raise LedgerError(
            f"{EXPANDED}: the capacity axis on the decisive tier records status "
            f"{axis['status']!r}, so the reading the manuscript reports as prospective is not one")
    support = ledger.declare_support(
        f"{FAMILY}/expanded/{tier}/capacity_axis",
        "The declared capacity axis on the matched-candidate doubly balanced tier: the four "
        f"declared ProGen2 sizes {', '.join(axis['ladder'])} over the same {axis['n_pairs']:,} "
        f"evaluation pairs in {axis['n_units']} wild-type family clusters, under the same draw "
        "matrix as every other endpoint of this cohort. Four released checkpoints of one family "
        "are four releases and not four training trials at controlled data and compute, so the "
        "axis identifies no causal effect of capacity, and parameter count is not the only thing "
        "that varies along a release ladder. With four points the rank coefficient takes only "
        f"{axis['spearman_distinct_values_without_ties']} distinct values, so its interval is "
        "coarse and the largest-minus-smallest contrast is the better-resolved reading of the same "
        "axis")
    emit(EXPANDED, ("capacity_axis", tier, "spearman_vs_log_parameters"),
         identifier=f"{FAMILY}/expanded/{tier}/capacity_spearman",
         claim="Spearman correlation between direction accuracy and log parameter count across the "
               "four declared ProGen2 sizes on the doubly balanced tier, declared before that tier "
               "was read; the declaration states in advance that a positive trend contradicts the "
               "hypothesis that capacity buys no directional ability",
         family=FAMILY, support=support, unit=RHO, kind="estimate",
         interval=axis["spearman_interval"], draws=axis["draws"], resampling_unit=CLUSTER_UNIT,
         verdict="supported" if axis["spearman_excludes_zero"] else "unresolved")
    contrast = axis["largest_minus_smallest"]
    emit(EXPANDED, ("capacity_axis", tier, "largest_minus_smallest", "difference"),
         identifier=f"{FAMILY}/expanded/{tier}/capacity_largest_minus_smallest",
         claim="cluster-paired difference in direction accuracy between the largest and the "
               "smallest declared ProGen2 size on the doubly balanced tier, which is the "
               "better-resolved reading of the declared capacity axis",
         family=FAMILY, support=support, unit=POINTS, kind="estimate",
         interval=contrast["interval"], draws=axis["draws"], resampling_unit=contrast["unit"],
         verdict=VERDICTS[contrast["verdict"]])


# --------------------------------------------------------------------------- #
# The declared magnitude stratification.
# --------------------------------------------------------------------------- #

def _magnitude_strata(ledger, emit, expanded: Mapping) -> None:
    """Direction accuracy within declared quantile bins of the likelihood difference.

    The bin boundaries are fixed on development clusters for each arm
    separately, so a bin index is a different pair set for every arm and each
    (arm, bin) carries its own support. Amendment 2 declares the endpoint
    descriptive and declares in advance that a rising profile means the native
    rule is informative where it is decisive; both statements travel on the
    claims below.
    """
    tier = "T_match"
    extremes: dict[int, list[tuple]] = {1: [], 5: []}
    masked: list[tuple] = []
    for arm, (name, autoregressive) in ARMS.items():
        strata = resolve(expanded, ("arms", arm, "tiers", tier, "magnitude_stratified"))
        if not strata["status"].startswith("descriptive"):
            raise LedgerError(f"{EXPANDED}: {name}'s magnitude stratification is not descriptive")
        if len(strata["bins"]) != 5:
            raise LedgerError(
                f"{EXPANDED}: {name} records {len(strata['bins'])} magnitude bins, not five")
        for block in strata["bins"]:
            index = block["bin_index"] + 1
            pointer = ("arms", arm, "tiers", tier, "magnitude_stratified", "bins",
                       f"[{block['bin_index']}]")
            support = ledger.declare_support(
                f"{FAMILY}/expanded/{tier}/{name}/magnitude_q{index}",
                f"Quintile {index} of five of the absolute likelihood difference in nats on the "
                f"matched-candidate matched tier, for {name}: {block['n_pairs']:,} evaluation "
                f"pairs in {block['n_units']} wild-type family clusters, median absolute "
                f"difference {block['magnitude_median_nats']:.4f} nats. The bin boundaries were "
                "fixed on development clusters for this arm, so a bin index is a different pair "
                "set for every arm and bins are not comparable across arms. The tier's own "
                "selection bounds apply unchanged: it matches the magnitude of the "
                "independent-site profile difference and leaves its sign free, so an accuracy here "
                "is not a general mutation-ranking accuracy")
            _accuracy(emit, EXPANDED, pointer, block,
                      f"{FAMILY}/expanded/{tier}/{name}/magnitude_q{index}/direction_accuracy",
                      f"direction accuracy of {name}'s native likelihood within quintile {index} "
                      "of the absolute likelihood difference on the matched tier. The endpoint is "
                      "descriptive and secondary: it carries no verdict in the declaration's "
                      "family and cannot overturn the primary endpoint, and the declaration states "
                      "in advance that a rising profile means the native rule is informative where "
                      "it is decisive",
                      support)
            if block["upper_nats"] is not None:
                emit(EXPANDED, pointer + ("upper_nats",),
                     identifier=f"{FAMILY}/expanded/{tier}/{name}/magnitude_q{index}/upper_nats",
                     claim=f"upper boundary of quintile {index} of the absolute likelihood "
                           f"difference for {name}, fixed on development clusters",
                     family=FAMILY, support=support, unit=NATS, kind="artifact_leaf")
            member = (EXPANDED, pointer + ("accuracy",), block["accuracy"], support)
            if autoregressive and index in extremes:
                extremes[index].append(member)
            if not autoregressive:
                masked.append(member)
    for index, label in ((1, "lowest"), (5, "highest")):
        _span(emit, f"{FAMILY}/expanded/{tier}/autoregressive_magnitude_q{index}_span",
              f"direction accuracy in the {label} quintile of the absolute likelihood difference "
              "on the matched tier, over the eight autoregressive arms",
              extremes[index], ACCURACY)
    _span(emit, f"{FAMILY}/expanded/{tier}/esm2-650m/magnitude_span",
          "direction accuracy of the masked arm over the five quintiles of its own absolute "
          "likelihood difference on the matched tier, reported separately and never pooled with "
          "the autoregressive arms",
          masked, ACCURACY)


# --------------------------------------------------------------------------- #
# What the expanded cohort covered and cost to score.
# --------------------------------------------------------------------------- #

def _scoring_cohort(ledger, emit, expanded: Mapping) -> None:
    cohort = ledger.artifacts.json(COHORT)
    support = ledger.declare_support(
        f"{FAMILY}/expanded/scoring",
        "The expanded matched-candidate cohort as scored: every variant sequence a retained "
        f"candidate pair of either tier needs, drawn from {cohort['variant_assays']} ProteinGym "
        f"substitution assays in {cohort['variant_clusters']} wild-type family clusters and scored "
        "fresh for the nine declared arms, which is what lets this support carry likelihood "
        "magnitudes in nats. An enumeration of the scored set, not a sampled estimate")
    for name, pointer, unit, claim in (
        ("variant_sequences", ("scoring_cost", "unique_variant_sequences"), "sequences",
         "unique variant sequences scored fresh for each declared arm of the expanded support"),
        ("assays", ("scoring_cost", "assays"), "assays",
         "ProteinGym assays contributing a scored variant sequence to the expanded support"),
        ("longest_wild_type", ("scoring_cost", "wildtype_length", "max"), "residues",
         "longest wild-type sequence in the expanded cohort, which is what decides whether the "
         "imposed scoring budget clips anything in this cohort"),
    ):
        emit(COHORT, pointer, identifier=f"{FAMILY}/expanded/scoring/{name}", claim=claim,
             family=FAMILY, support=support, unit=unit, kind="census")
    budgets = {expanded["arms"][arm]["identity"]["budget"] for arm, (_, autoregressive)
               in ARMS.items() if autoregressive}
    if len(budgets) != 1:
        raise LedgerError(
            f"{EXPANDED}: the autoregressive arms record several scoring budgets "
            f"{sorted(budgets)}, so there is no one imposed budget to quote")
    emit(EXPANDED, ("arms", "progen2-large", "identity", "budget"),
         identifier=f"{FAMILY}/expanded/scoring/budget",
         claim="scored-position budget the repository imposes on every autoregressive arm of this "
               "campaign, including the two ProGen2 arms whose recorded caveat names it",
         family=FAMILY, support=support, unit="positions", kind="constant")


# --------------------------------------------------------------------------- #
# The retained 33-arm panel: the check that over-corrected, and the readouts.
# --------------------------------------------------------------------------- #

def _panel(ledger, emit) -> None:
    panel = ledger.artifacts.json(PANEL)
    # The panel states its resampling unit once for the whole analysis rather
    # than on every block, so the rows below read that statement instead of
    # carrying a literal of their own.
    if panel["inference"]["resampling_unit"] != CLUSTER_UNIT:
        raise LedgerError(
            f"{PANEL}: the panel resamples {panel['inference']['resampling_unit']!r} and the "
            f"expanded support resamples {CLUSTER_UNIT!r}, so the two supports' intervals are not "
            "over one kind of unit and the rows below would claim they are")
    supports = {}
    for tier, name in (("T_open", "open tier"), ("T_match", "matched tier")):
        block = resolve(panel, ("cohort", "counts", tier, "evaluation"))
        supports[tier] = ledger.declare_support(
            f"{FAMILY}/retained_panel/{tier}",
            f"Matched-candidate {name} of the retained 33-arm panel: {block['pairs']} evaluation "
            f"pairs in {block['clusters']} wild-type family clusters, built by the same declared "
            "construction on the retained common mutation cohort and read on the within-assay "
            "likelihood rank scale, which preserves the ordering of two variants of one assay and "
            "carries no magnitude. This support supplies the supervised comparison; it is never "
            "pooled with the expanded support, and the selection bounds of a matched tier apply "
            "to it unchanged")
        emit(PANEL, ("cohort", "counts", tier, "evaluation", "pairs"),
             identifier=f"{FAMILY}/retained_panel/{tier}/evaluation_pairs",
             claim=f"retained evaluation pairs of the retained panel's {name}", family=FAMILY,
             support=supports[tier], unit=PAIRS, kind="support_count")
        emit(PANEL, ("cohort", "counts", tier, "evaluation", "clusters"),
             identifier=f"{FAMILY}/retained_panel/{tier}/evaluation_clusters",
             claim="wild-type family clusters contributing evaluation pairs to the retained "
                   f"panel's {name}", family=FAMILY, support=supports[tier], unit=CLUSTERS,
             kind="support_count")
    _failed_position_check(ledger, emit, panel)
    _readout_levels(ledger, emit, panel, supports)
    _readout_counts(ledger, emit, panel, supports)
    _readout_validity(ledger, emit, panel)


def _failed_position_check(ledger, emit, panel: Mapping) -> None:
    """The construction check that over-corrected, recorded as the failure it is."""
    addendum = ledger.artifacts.json(PANEL_A3)
    check = resolve(addendum, ("construction_checks", "T_position_matched"))
    if check["status"] != "failed":
        raise LedgerError(
            f"{PANEL_A3}: the position-matched tier records status {check['status']!r}, so the "
            "check the manuscript reports as failed is not recorded as one")
    construction = addendum["construction"]
    support = ledger.declare_support(
        f"{FAMILY}/retained_panel/T_position_matched",
        "The retired magnitude-matched position tier of the retained panel: a subset of the "
        "matched tier whose relative-position difference is additionally matched below the "
        f"{construction['position_match_threshold']['percentile']:g}th percentile of that "
        "quantity, holding "
        f"{construction['counts']['T_position_matched']['evaluation']['pairs']} evaluation pairs. "
        "Matching a magnitude leaves its sign free, so the tier failed its own construction check "
        "and amendment 4 retired it in favour of a sign-balanced position tier; it is kept in the "
        "artifact with status failed and is not to be rebuilt")
    emit(PANEL_A3, ("construction_checks", "T_position_matched", "point"),
         identifier=f"{FAMILY}/retained_panel/T_position_matched/rule_position",
         claim="direction accuracy of the position-only rule on the magnitude-matched position "
               "tier: the declared construction check this tier failed, because matching the "
               "magnitude of the relative-position difference left its sign free and drove the "
               "rule below chance by more than the declared margin instead of to it",
         family=FAMILY, support=support, unit=ACCURACY, kind="estimate",
         interval=check["interval"], draws=panel["inference"]["draws"],
         resampling_unit=CLUSTER_UNIT, verdict=VERDICTS[check["verdict"]])


def _readout_levels(ledger, emit, panel: Mapping, supports: Mapping[str, str]) -> None:
    """The supervised readout that carries no model information, across fold seeds.

    The design is model-independent by construction --- profile and sequence
    descriptors only --- so the artifact records one set of points under every
    arm. That is checked rather than assumed, because a design that had become
    arm-specific would make the span below a span over arms as well as seeds.
    """
    del ledger
    level = "T_match/R_seqprof"
    points = first = None
    for arm, record in panel["E22"]["arms"].items():
        seen = record["levels"][level]["across_seeds"]["points"]
        if points is None:
            points, first = seen, arm
        elif seen != points:
            raise LedgerError(
                f"{PANEL}: the profile-plus-sequence readout differs between {first} and {arm} on "
                "the matched tier, so it is not the model-independent design the manuscript "
                "reports")
    across = panel["E22"]["arms"][first]["levels"][level]["across_seeds"]
    if across["verdict"] != "unresolved":
        raise LedgerError(
            f"{PANEL}: the profile-plus-sequence readout on the matched tier is "
            f"{across['verdict']!r}, so the manuscript's statement that every interval covers "
            "chance is not what the analysis records")
    members = []
    for seed in sorted(points):
        pointer = ("E22", "arms", first, "levels", level, "per_seed", seed)
        block = resolve(panel, pointer)
        _accuracy(emit, PANEL, pointer, block,
                  f"{FAMILY}/retained_panel/T_match/readout_profile_sequence/{seed}",
                  "direction accuracy on the matched tier of the supervised readout fitted on the "
                  "profile score and the 444 sequence, mutation-identity, position and length "
                  "descriptors without the native-likelihood feature, at one fold seed. The design "
                  "carries no model information, so this is what supervision alone recovers",
                  supports["T_match"], field="point", draws="n_draws", unit=None,
                  seed=(int(seed),))
        members.append((PANEL, pointer + ("point",), block["point"], supports["T_match"]))
    _span(emit, f"{FAMILY}/retained_panel/T_match/readout_profile_sequence_span",
          "direction accuracy on the matched tier of the supervised profile-plus-sequence readout "
          "without the native-likelihood feature, over the three fold seeds",
          members, ACCURACY)


def _readout_counts(ledger, emit, panel: Mapping, supports: Mapping[str, str]) -> None:
    """How many arms resolve a supervised contrast, counted from the verdicts.

    The count is what the manuscript prints, because the per-arm increments
    themselves are not quoted. It is read off the artifact's own across-seed
    verdicts and the pointer names one of the arms it counts, so the count cannot
    drift from the verdicts it summarises.
    """
    del ledger
    arms = panel["E22"]["arms"]
    contrast = "T_open/B_R_minus_B"
    matching = sorted(arm for arm, record in arms.items()
                      if record["contrasts"][contrast]["across_seeds"]["verdict"]
                      == "bounded_negative")
    if not matching:
        raise LedgerError(f"{PANEL}: no arm records {contrast} as a bounded negative")
    emit(PANEL, ("E22", "arms", matching[0], "contrasts", contrast, "across_seeds", "verdict"),
         identifier=f"{FAMILY}/retained_panel/T_open/representation_increment/bounded_negative_arms",
         claim="arms whose added-representation increment on the open tier is a bounded negative "
               "at the declared margin: the frozen representation adds nothing beyond the "
               "likelihood-bearing supervised baseline, resolved against the margin rather than "
               "merely unresolved",
         family=FAMILY, support=supports["T_open"], unit="checkpoints", kind="count",
         value=len(matching), level="representation")


def _readout_validity(ledger, emit, panel: Mapping) -> None:
    validity = panel["E22"]["validity"]
    if not validity["reproduced_established_points"] or \
            validity["cells_whose_unmatched_interval_misses_the_established_point"]:
        raise LedgerError(
            f"{PANEL}: the supervised validity check did not pass, so the matched reading of the "
            "readouts is void and none of its rows may be written")
    support = ledger.declare_support(
        f"{FAMILY}/retained_panel/readout_validity",
        "The declared validity check of the supervised comparison: every retained readout report "
        "of the panel refitted nothing and had to reproduce its own family-mean per-assay Spearman "
        "on the full common support, and its interval on the "
        f"{validity['matched_variants']:,} variants no retained pair uses had to cover that "
        "established point. A failure of either would void the matched reading; both held")
    emit(PANEL, ("E22", "validity", "n_cells"),
         identifier=f"{FAMILY}/retained_panel/readout_validity/cells",
         claim="retained readout reports whose established out-of-family predictive performance "
               "was reproduced before any matched pair was read",
         family=FAMILY, support=support, unit="cells", kind="count")
    emit(PANEL, ("E22", "validity", "max_abs_difference"),
         identifier=f"{FAMILY}/retained_panel/readout_validity/max_abs_difference",
         claim="largest absolute difference between a recomputed and an established family-mean "
               "per-assay Spearman over those reports, which is the agreement the validity check "
               "required",
         family=FAMILY, support=support, unit="dimensionless Spearman", kind="artifact_leaf")
