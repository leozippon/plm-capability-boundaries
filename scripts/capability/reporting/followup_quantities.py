"""Source mappings for the follow-up supports the later manuscript sections cite.

Every row here is read out of a retained artifact, or computed from two of them
by a stated arithmetic. No fit is run, no cohort is drawn and no interval is
resampled; an interval is copied with the resampling unit and draw count the
artifact records beside it.

The main builder owns the shared ledger and the older campaign mappings. This
module covers the exact stability model cohort, the release-lineage robustness
and correlation follow-ups, the stability likelihood replay, the two pairwise
residual full panels and the insertion/deletion exclusion that separates them,
the position-term simultaneous panel and its layout receipts, the generation
replication campaign summary, and the ProGen3-3B unconditional completion census.
"""
from __future__ import annotations

from statistics import median
import glob
import json
import os

from src.capability.core.evidence_ledger import Quantity, resolve

FOLLOWUP = "results/shared/followup_support_20260927"
REPLAY = "results/R4/stability_likelihood_replay_20260927"
UNFILTERED = "results/R3/pairwise_residual_full_panel_20260927"
COMPANION = "results/R3/pairwise_residual_full_panel_indel_excluded_20260928"
EXCLUSION = "results/R3/pairwise_indel_exclusion_20260928/pairwise_indel_exclusion.json"
RESTRICTED = "archive/logs/R3/pairwise_epistasis_20260924/panel/panel_indel_restricted.json"
POSITION = "results/R1/position_terms_20260926"
REPLICATION = "results/R6/generation_replication_20260927/summary/generation_campaign_summary.json"
GATE = "results/R6/generative_control_20260924/gate_endpoints.json"
MANIFEST = "configs/generation_replication_manifest.json"
NATIVE_EXPRESSION = "results/R1/native_expression_strata_20260926/native_expression_conditioned_33arm.json"
LINEAGE_DECLARATION = f"{FOLLOWUP}/lineage_declaration.json"
PROGEN3 = "results/R6/progen3_generation_20260905"

#: The three split seeds every panel here averages over before resampling.
SPLIT_SEEDS = (20260923, 20260924, 20260925)
#: How a field name states its unit in the follow-up artifacts. A name outside
#: this mapping yields no row rather than a guessed unit.
FIELD_UNITS = (
    ("_kcal_mol", "kcal/mol"),
    ("_nats", "nats"),
    ("_rank_correlation", "dimensionless Spearman"),
    ("_spearman", "dimensionless Spearman"),
    ("pearson_r", "dimensionless"),
    ("_ratio", "dimensionless ratio"),
)


def _unit_of(field: str) -> str:
    for suffix, unit in FIELD_UNITS:
        if field.endswith(suffix) or field == suffix:
            return unit
    return ""


def add_followup_quantities(ledger) -> None:
    emit = _emitter(ledger)
    _exact_stability_cohort(ledger, emit)
    _release_lineage(ledger, emit)
    _stability_replay(ledger, emit)
    _pairwise_residual_panels(ledger, emit)
    _indel_exclusion(ledger, emit)
    _position_simultaneous(ledger, emit)
    _position_layout(ledger, emit)
    _generation_replication(ledger, emit)
    _native_expression(ledger, emit)
    _native_completion(ledger, emit)


def _emitter(ledger):
    def emit(path, pointer, *, identifier, claim, family, support, unit, kind="support_count",
             value=None, interval=None, draws=None, resampling_unit=None, reason=None,
             interval_kind="95% percentile", seed=(), level="measurement",
             weighting_convention=None):
        if value is None:
            value = resolve(ledger.artifacts.json(path), pointer)
        ledger.add(Quantity(
            id=identifier, claim=claim, family=family, value=float(value), unit=unit, kind=kind,
            support_id=support, interval=None if interval is None else tuple(interval),
            interval_kind=interval_kind if interval is not None else None,
            resampling_unit=resampling_unit, resampling_draws=draws, no_interval_reason=reason,
            seed_set=tuple(seed), level=level, weighting_convention=weighting_convention,
            source_path=path, source_sha256=ledger.artifacts.sha256(path),
            source_pointer=tuple(pointer)))
    return emit


# --------------------------------------------------------------------------- #
# The exact frozen model-fitting subset of the folding-stability cohort.
# --------------------------------------------------------------------------- #

def _exact_stability_cohort(ledger, emit) -> None:
    path = f"{FOLLOWUP}/exact_stability_qualification.json"
    payload = ledger.artifacts.json(path)
    block = payload["agreement"]["all"]
    support = ledger.declare_support(
        "exact_stability_model_cohort",
        "the exact frozen model-fitting subset of the folding-stability cohort: 25,856 variants at "
        "5,664 sites in 101 backgrounds, 101 family groups and 101 source clusters")
    draws, seed = payload["bootstrap"]["draws"], payload["bootstrap"]["seed"]
    for field, unit in (("variants", "variants"), ("backgrounds", "backgrounds"),
                        ("family_groups", "family groups"), ("source_clusters", "source clusters"),
                        ("sites", "sites")):
        emit(path, ("agreement", "all", field), identifier=f"exact_stability/support/{field}",
             claim=f"exact stability model cohort {field.replace('_', ' ')}",
             family="exact_stability", support=support, unit=unit)
    for field in ("sd_combined_ddg_kcal_mol", "mean_combined_ddg_kcal_mol"):
        emit(path, ("agreement", "all", field), identifier=f"exact_stability/{field}",
             claim=f"exact stability model cohort {field.replace('_', ' ')}",
             family="exact_stability", support=support, unit="kcal/mol", kind="artifact_leaf")
    unit_block = block["family_group"]
    emit(path, ("agreement", "all", "family_group", "units"),
         identifier="exact_stability/family_group/units",
         claim="family groups the channel assessment resamples", family="exact_stability",
         support=support, unit="family groups")
    emit(path, ("agreement", "all", "family_group", "effective_units_kish"),
         identifier="exact_stability/family_group/effective_units",
         claim="effective family groups behind the channel assessment", family="exact_stability",
         support=support, unit="effective units", kind="effective_count",
         weighting_convention="equal family groups")
    for metric, item in unit_block["agreement"].items():
        unit = _unit_of(metric)
        if not unit:
            ledger.gap(id=f"exact_stability/family_group/{metric}",
                       claim=f"the exact-cohort {metric.replace('_', ' ')}", family="exact_stability",
                       reason="the field name states no unit, so no row is written",
                       looked_in=path, searched="the family-group agreement block")
            continue
        emit(path, ("agreement", "all", "family_group", "agreement", metric),
             identifier=f"exact_stability/family_group/{metric}",
             claim=f"exact stability model cohort {metric.replace('_', ' ')}, original "
                   "median-aggregated accepted channel labels",
             family="exact_stability", support=support, unit=unit, kind="estimate",
             value=item["point"], interval=item["ci95"], draws=draws,
             resampling_unit=unit_block["unit"], seed=(seed,))


# --------------------------------------------------------------------------- #
# Release-family robustness and the exact lineage enumeration.
# --------------------------------------------------------------------------- #

def _release_lineage(ledger, emit) -> None:
    path = f"{FOLLOWUP}/lineage_robustness.json"
    payload = ledger.artifacts.json(path)
    support = ledger.declare_support(
        "lineage_release_families",
        "163 shared wild-type clusters at 50% identity, 33 checkpoints in 16 release families, "
        "three split seeds averaged within cluster before resampling; the frozen release grouping "
        "is not protein-exposure status: InstructProtein is protein-adapted and Galactica is a "
        "scientific language–protein release family")
    contrast = ("the equally weighted release-family contrast: protein-specialized or "
                "protein-adapted release families minus general-purpose text and scientific "
                "language–protein release families")
    # Legacy keys, quantity identifiers and source selectors remain for compatibility.
    for name, claim in (
            ("family_bootstrap", "the 16-release-family simultaneous panel"),
            ("protein_minus_text_release_mean", contrast),
            ("exclude_shared_llama2_prollama",
             f"{contrast}, excluding the Llama2 and ProLLaMA release families")):
        block = payload[name]
        emit(path, (name, "draws"), identifier=f"lineage_robustness/{name}/draws",
             claim=f"paired group bootstrap draws behind {claim}", family="lineage_robustness",
             support=support, unit="bootstrap draws", kind="constant")
        emit(path, (name, "groups"), identifier=f"lineage_robustness/{name}/groups",
             claim=f"biological clusters {claim} resamples", family="lineage_robustness",
             support=support, unit="source clusters")
        emit(path, (name, "critical_value"), identifier=f"lineage_robustness/{name}/critical_value",
             claim=f"maximum-statistic critical value of {claim}", family="lineage_robustness",
             support=support, unit="dimensionless", kind="constant")
        emit(path, (name, "family_size"), identifier=f"lineage_robustness/{name}/family_size",
             claim=f"simultaneous family size of {claim}", family="lineage_robustness",
             support=support, unit="release families", kind="count")
        if name == "family_bootstrap":
            continue
        emit(path, (name, "point", "[0]"), identifier=f"lineage_robustness/{name}",
             claim=f"{claim}, equal checkpoints within release family",
             family="lineage_robustness", support=support, unit="dimensionless Spearman",
             kind="estimate", value=block["point"][0], interval=block["interval"][0],
             draws=block["draws"], resampling_unit="wild-type cluster at 50% identity",
             seed=SPLIT_SEEDS, level="likelihood")
    family_block = payload["family_bootstrap"]
    for index, name in enumerate(payload["family_order"]):
        emit(path, ("family_bootstrap", "point", f"[{index}]"),
             identifier=f"lineage_robustness/family/{name}",
             claim=f"{name} release-family mean adjusted-target increment",
             family="lineage_robustness", support=support, unit="dimensionless Spearman",
             kind="estimate", value=family_block["point"][index],
             interval=family_block["interval"][index], draws=family_block["draws"],
             resampling_unit="wild-type cluster at 50% identity", seed=SPLIT_SEEDS,
             level="likelihood")

    path = f"{FOLLOWUP}/lineage_correlation.json"
    payload = ledger.artifacts.json(path)
    support = ledger.declare_support(
        "lineage_correlation_checkpoints",
        "18 checkpoints in 9 release lineages, one mutation-score and one profile-rate reading each; "
        "exact enumeration of the multinomial-weighted lineage compositions")
    for field, unit in (("checkpoint_count", "checkpoints"), ("release_family_count", "release families")):
        emit(path, (field,), identifier=f"lineage_correlation/{field}",
             claim=f"lineage correlation {field.replace('_', ' ')}", family="lineage_correlation",
             support=support, unit=unit)
    for name, claim in (("cluster_resampling", "the lineage-resampled checkpoint correlation"),
                        ("family_means", "the release-family-mean checkpoint correlation"),
                        ("merge_galactica_instructprotein_sensitivity",
                         "the merged Galactica and InstructProtein sensitivity")):
        block = payload[name]
        emit(path, (name, "compositions"), identifier=f"lineage_correlation/{name}/compositions",
             claim=f"multinomial-weighted count compositions {claim} enumerates exactly",
             family="lineage_correlation", support=support, unit="compositions", kind="census",
             reason="a complete enumeration of the compositions")
        emit(path, (name, "clusters"), identifier=f"lineage_correlation/{name}/clusters",
             claim=f"lineages {claim} enumerates over", family="lineage_correlation",
             support=support, unit="checkpoint lineages")
        emit(path, (name, "point"), identifier=f"lineage_correlation/{name}",
             claim=f"{claim} across the 18 checkpoints", family="lineage_correlation",
             support=support, unit="dimensionless Spearman", kind="estimate",
             value=block["point"], interval=block["interval"], draws=block["compositions"],
             resampling_unit="checkpoint lineage; exact multinomial composition enumeration")


# --------------------------------------------------------------------------- #
# The stability likelihood replay: unchanged fits, matched multiplicity.
# --------------------------------------------------------------------------- #

def _stability_replay(ledger, emit) -> None:
    path = f"{REPLAY}/panel.json"
    payload = ledger.artifacts.json(path)
    primary = payload["primary"]
    support = ledger.declare_support(
        "stability_replay_101g",
        "101 folding-stability family groups on the exact frozen model cohort, 33 likelihood arms "
        "refitted from the historical selected baseline with unchanged fits")
    for field, unit, kind in (("draws", "bootstrap draws", "constant"),
                              ("groups", "family groups", "support_count"),
                              ("family_size", "checkpoints", "count"),
                              ("critical_value", "dimensionless", "constant")):
        emit(path, ("primary", field), identifier=f"stability_replay/panel/{field}",
             claim=f"stability likelihood replay {field.replace('_', ' ')}",
             family="stability_replay", support=support, unit=unit, kind=kind)
    for field in ("resolved_positive", "resolved_negative"):
        emit(path, (field,), identifier=f"stability_replay/{field}",
             claim=f"checkpoints the replay resolves {field.split('_')[1]}",
             family="stability_replay", support=support, unit="checkpoints", kind="count")
    for index, arm in enumerate(payload["arms"]):
        emit(path, ("primary", "point", f"[{index}]"), identifier=f"stability_replay/{arm}",
             claim=f"{arm} seed-mean paired squared-error reduction on the stability replay",
             family="stability_replay", support=support, unit="kcal^2/mol^2", kind="estimate",
             value=primary["point"][index], interval=primary["interval"][index],
             draws=primary["draws"], resampling_unit="family group", seed=SPLIT_SEEDS,
             level="likelihood")


# --------------------------------------------------------------------------- #
# The two pairwise residual full panels and what separates them.
# --------------------------------------------------------------------------- #

PANELS = (
    ("unfiltered", UNFILTERED, "pairwise_residual_unfiltered",
     "64 MegaScale family groups, 8,192 wild-type-centred double-mutant cycles over 217 site pairs; "
     "the unfiltered primary support of the pairwise residual refit"),
    ("indel_excluded", COMPANION, "pairwise_residual_indel_excluded",
     "64 MegaScale family groups, 8,189 wild-type-centred double-mutant cycles over 214 site pairs; "
     "every insertion- or deletion-construct row excluded from the state level before any fit"),
)


def _pairwise_residual_panels(ledger, emit) -> None:
    per_group: dict[str, tuple[str, str, float]] = {}
    for label, root, support_id, description in PANELS:
        support = ledger.declare_support(support_id, description)
        family = f"pairwise_residual_{label}"
        declaration = f"{root}/declaration.json"
        for field, unit in (("groups", "family groups"), ("cycles", "double-mutant cycles"),
                            ("site_pairs", "site pairs")):
            emit(declaration, ("support_counts", field),
                 identifier=f"{family}/support/{field}",
                 claim=f"{label.replace('_', '-')} pairwise residual support {field.replace('_', ' ')}",
                 family=family, support=support, unit=unit)
        emit(declaration, ("bootstrap_draws",), identifier=f"{family}/bootstrap_draws",
             claim=f"paired group bootstrap draws behind the {label.replace('_', '-')} panel",
             family=family, support=support, unit="bootstrap draws", kind="constant")

        path = f"{root}/panel.json"
        payload = ledger.artifacts.json(path)
        for block, field, unit, kind in (
                ("primary_bootstrap", "critical_value", "dimensionless", "constant"),
                ("primary_bootstrap", "family_size", "checkpoints", "count"),
                ("split_diagnostic_bootstrap", "critical_value", "dimensionless", "constant"),
                ("split_diagnostic_bootstrap", "family_size", "checkpoint--seed cells", "count")):
            emit(path, (block, field), identifier=f"{family}/{block}/{field}",
                 claim=f"{block.replace('_', ' ')} {field.replace('_', ' ')} of the "
                       f"{label.replace('_', '-')} panel",
                 family=family, support=support, unit=unit, kind=kind)
        for field in ("resolved_positive_arms", "resolved_negative_arms"):
            emit(path, (field,), identifier=f"{family}/{field}",
                 claim=f"arms the {label.replace('_', '-')} panel resolves "
                       f"{field.split('_')[1]}",
                 family=family, support=support, unit="checkpoints", kind="count")
        uppers = []
        for arm, block in payload["arms"].items():
            uppers.append(block["simultaneous95_interval_kcal2"][1])
            emit(path, ("arms", arm, "seed_mean_mse_reduction_kcal2"),
                 identifier=f"{family}/{arm}",
                 claim=f"{arm} seed-mean paired squared-error reduction on the "
                       f"{label.replace('_', '-')} pairwise residual support",
                 family=family, support=support, unit="kcal^2/mol^2", kind="estimate",
                 value=block["seed_mean_mse_reduction_kcal2"],
                 interval=block["simultaneous95_interval_kcal2"],
                 draws=payload["primary_bootstrap"]["draws"], resampling_unit="family group",
                 seed=SPLIT_SEEDS, level="likelihood")
        for bound, value in (("min", min(uppers)), ("max", max(uppers))):
            emit(path, ("arms", f"<{bound} simultaneous95_interval_kcal2 upper limit>"),
                 identifier=f"{family}/simultaneous_upper_limit/{bound}",
                 claim=f"{'smallest' if bound == 'min' else 'largest'} simultaneous upper limit "
                       f"across the {label.replace('_', '-')} panel's checkpoints",
                 family=family, support=support, unit="kcal^2/mol^2", kind="range_bound",
                 value=value, level="likelihood")
        per_group[label] = _largest_per_group_term(ledger, emit, root, family, support, label)
    _panel_comparison(ledger, emit)


def _seed_mean_per_group(payload: dict) -> dict[str, float]:
    """Each group's seed-mean paired squared-error reduction, from the per-group
    mean squared errors the receipt records for the two designs it contrasts."""
    seeds = list(payload["seeds"])
    terms: dict[str, float] = {}
    for index, group in enumerate(payload["groups_order"]):
        values = [payload["seeds"][seed]["per_group_mse"]["C_G_T"][index]
                  - payload["seeds"][seed]["per_group_mse"]["C_G_T+M"][index] for seed in seeds]
        terms[group] = sum(values) / len(values)
    return terms


def _largest_per_group_term(ledger, emit, root: str, family: str, support: str, label: str):
    best = None
    for path in sorted(glob.glob(os.path.join(ledger.artifacts.root, root, "residual_*.json"))):
        relative = os.path.relpath(path, ledger.artifacts.root)
        terms = _seed_mean_per_group(ledger.artifacts.json(relative))
        group = max(terms, key=lambda name: abs(terms[name]))
        if best is None or abs(terms[group]) > abs(best[2]):
            best = (relative, group, terms[group])
    if best is None:
        ledger.gap(id=f"{family}/largest_per_group_term",
                   claim="the largest seed-mean paired per-group term on this support",
                   family=family, reason="no per-arm residual receipt is retained here",
                   looked_in=root, searched="every residual_*.json under the panel directory")
        return None
    relative, group, value = best
    emit(relative, ("seeds", "<seed mean>", "per_group_mse", f"C_G_T minus C_G_T+M at {group}"),
         identifier=f"{family}/largest_per_group_term",
         claim=f"largest seed-mean paired per-group squared-error term on the "
               f"{label.replace('_', '-')} support, family group {group}",
         family=family, support=support, unit="kcal^2/mol^2", kind="range_bound", value=value,
         seed=SPLIT_SEEDS, level="likelihood")
    return best


def _panel_comparison(ledger, emit) -> None:
    """How far the indel-excluded companion moves each arm's point estimate.

    Read from the two panels rather than restated: the count inside the band,
    the median absolute move and the sign changes all follow from the same pair
    of files, so a change in either moves all three.
    """
    unfiltered = ledger.artifacts.json(f"{UNFILTERED}/panel.json")["arms"]
    companion_path = f"{COMPANION}/panel.json"
    companion = ledger.artifacts.json(companion_path)["arms"]
    support = "pairwise_residual_indel_excluded"
    family = "pairwise_residual_indel_excluded"
    inside = sum(1 for arm, block in unfiltered.items()
                 if block["simultaneous95_interval_kcal2"][0]
                 <= companion[arm]["seed_mean_mse_reduction_kcal2"]
                 <= block["simultaneous95_interval_kcal2"][1])
    moves = [abs(companion[arm]["seed_mean_mse_reduction_kcal2"]
                 - block["seed_mean_mse_reduction_kcal2"]) for arm, block in unfiltered.items()]
    signs = sum(1 for arm, block in unfiltered.items()
                if (block["seed_mean_mse_reduction_kcal2"] > 0)
                != (companion[arm]["seed_mean_mse_reduction_kcal2"] > 0))
    pointer = ("arms", "<compared arm by arm with the unfiltered panel>")
    emit(companion_path, pointer + ("points inside the unfiltered band",),
         identifier=f"{family}/points_inside_unfiltered_band",
         claim="companion point estimates lying inside the corresponding unfiltered simultaneous band",
         family=family, support=support, unit="checkpoints", kind="count", value=inside)
    emit(companion_path, pointer + ("arms whose point changes sign",),
         identifier=f"{family}/arms_changing_point_sign",
         claim="arms whose seed-mean point estimate changes sign between the two supports",
         family=family, support=support, unit="checkpoints", kind="count", value=signs)
    emit(companion_path, pointer + ("median absolute point move",),
         identifier=f"{family}/median_point_move",
         claim="median absolute move of the seed-mean point estimate between the unfiltered and "
               "indel-excluded supports",
         family=family, support=support, unit="kcal^2/mol^2", kind="estimate",
         value=median(moves), level="likelihood",
         reason="a median over the 33 arms' point moves; neither panel resamples the difference")


# --------------------------------------------------------------------------- #
# What the insertion/deletion exclusion touches.
# --------------------------------------------------------------------------- #

def _indel_exclusion(ledger, emit) -> None:
    support = "pairwise_all_64g"
    for field, unit, claim in (
            ("indel_rows_in_pinned_files", "rows",
             "contributing rows in the pinned files naming an insertion or deletion construct"),
            ("states_inspected", "distinct measured states",
             "measured states the exclusion record inspected")):
        emit(EXCLUSION, (field,), identifier=f"pairwise/indel_exclusion/{field}",
             claim=claim, family="pairwise", support=support, unit=unit, kind="census",
             reason="a complete enumeration of the pinned files")
    for field, unit, claim in (
            ("states_with_an_indel_row", "distinct measured states",
             "measured states resting on at least one insertion-construct row"),
            ("states_absent_after_exclusion", "distinct measured states",
             "measured states left with no accepted row once insertion-construct rows are excluded"),
            ("states_whose_value_moves", "distinct measured states",
             "measured states whose median value moves once insertion-construct rows are excluded")):
        emit(EXCLUSION, ("accounting", field), identifier=f"pairwise/indel_exclusion/{field}",
             claim=claim, family="pairwise", support=support, unit=unit, kind="census",
             reason="a complete enumeration of the inspected states")
    emit(EXCLUSION, ("accounting", "largest_state_value_move_kcal_mol"),
         identifier="pairwise/indel_exclusion/largest_state_value_move",
         claim="largest median value move among the states the exclusion keeps",
         family="pairwise", support=support, unit="kcal/mol", kind="artifact_leaf")

    restricted = ledger.artifacts.json(RESTRICTED)
    affected = restricted["restricted_evaluation_support"]["all"]["dropped_cycles"]
    dropped = (ledger.artifacts.json(f"{UNFILTERED}/declaration.json")["support_counts"]["cycles"]
               - ledger.artifacts.json(f"{COMPANION}/declaration.json")["support_counts"]["cycles"])
    emit(EXCLUSION, ("accounting", "<affected cycles minus the cycles the companion support drops>"),
         identifier="pairwise/indel_exclusion/cycles_with_a_shifted_target",
         claim="cycles the companion support keeps whose target value the exclusion shifts",
         family="pairwise", support=support, unit="double-mutant cycles", kind="census",
         value=affected - dropped,
         reason="a complete enumeration of the affected cycles the companion support keeps")
    emit(EXCLUSION, ("accounting", "<cycles the companion support drops>"),
         identifier="pairwise/indel_exclusion/cycles_dropped_by_the_companion",
         claim="cycles the indel-excluded companion support drops relative to the unfiltered one",
         family="pairwise", support=support, unit="double-mutant cycles", kind="census",
         value=dropped, reason="a complete enumeration of the dropped cycles")

    full = ledger.declare_support(
        "pairwise_indel_fully_excluded",
        "63 family groups and 8,062 double-mutant cycles over 211 site pairs, every cycle carrying "
        "an insertion- or deletion-construct row removed; one family group loses its only background")
    for field, unit in (("groups", "family groups"), ("cycles", "double-mutant cycles"),
                        ("site_pairs", "site pairs")):
        emit(RESTRICTED, ("restricted_evaluation_support", "all", field),
             identifier=f"pairwise/indel_full_exclusion/{field}",
             claim=f"{field.replace('_', ' ')} left when every cycle carrying an insertion- or "
                   "deletion-construct row is excluded",
             family="pairwise", support=full, unit=unit)
    emit(RESTRICTED, ("restricted_evaluation_support", "all", "dropped_groups", "<count>"),
         identifier="pairwise/indel_full_exclusion/dropped_groups",
         claim="family groups emptied when every cycle carrying an insertion- or deletion-construct "
               "row is excluded",
         family="pairwise", support=full, unit="family groups", kind="census",
         value=len(restricted["restricted_evaluation_support"]["all"]["dropped_groups"]),
         reason="a complete enumeration of the emptied groups")


# --------------------------------------------------------------------------- #
# Position terms: the simultaneous follow-up and its layout receipts.
# --------------------------------------------------------------------------- #

def _position_simultaneous(ledger, emit) -> None:
    path = f"{POSITION}/position_simultaneous.json"
    payload = ledger.artifacts.json(path)
    support = ledger.declare_support(
        "position_aligned_subset",
        "191 ProteinGym assays in 156 wild-type families at 50% identity and 2,991 grid-aligned "
        "single substitutions selected label-blind; 32 arms and three split seeds")
    emit(path, ("arms", "<count>"), identifier="position_terms/simultaneous/arms",
         claim="arms the position-term simultaneous follow-up covers", family="position_terms",
         support=support, unit="checkpoints", kind="count", value=len(payload["arms"]))
    for control, block in payload["controls"].items():
        name = "window" if control == "wall" else "rejoined-suffix"
        contrasts = block["contrasts"]
        resolved = [row for row in contrasts if not row["interval"][0] <= 0 <= row["interval"][1]]
        prefix = f"position_terms/{control}"
        for field, unit, kind in (("critical_value", "dimensionless", "constant"),
                                  ("draws", "bootstrap draws", "constant"),
                                  ("family_size", "contrasts", "count"),
                                  ("family_universe", "source clusters", "support_count")):
            emit(path, ("controls", control, field), identifier=f"{prefix}/{field}",
                 claim=f"{name} control {field.replace('_', ' ')} of the position-term "
                       "simultaneous follow-up",
                 family="position_terms", support=support, unit=unit, kind=kind)
        for bound, value in (("min", min(row["families"] for row in contrasts)),
                             ("max", max(row["families"] for row in contrasts))):
            emit(path, ("controls", control, "contrasts", f"<{bound} families>"),
                 identifier=f"{prefix}/available_families/{bound}",
                 claim=f"{'fewest' if bound == 'min' else 'most'} available families behind a "
                       f"{name} control contrast",
                 family="position_terms", support=support, unit="source clusters",
                 kind="range_bound", value=value)
        joint = [row for row in contrasts if row["key"] == "joint_over_full"]
        for identifier, value, unit, claim in (
                ("resolved_contrasts", len(resolved), "contrasts",
                 f"contrasts the {name} control resolves"),
                ("arms_resolving_a_contrast", len({row["arm"] for row in resolved}), "checkpoints",
                 f"arms resolving at least one contrast under the {name} control"),
                ("arms_resolving_nothing",
                 len(payload["arms"]) - len({row["arm"] for row in resolved}), "checkpoints",
                 f"arms resolving no contrast under the {name} control"),
                ("joint_over_full_covering_zero",
                 sum(1 for row in joint if row["interval"][0] <= 0 <= row["interval"][1]),
                 "contrasts",
                 f"joint-over-full bands covering zero under the {name} control")):
            emit(path, ("controls", control, "contrasts", f"<{identifier}>"),
                 identifier=f"{prefix}/{identifier}", claim=claim, family="position_terms",
                 support=support, unit=unit, kind="count", value=value)
        for key in payload["keys"]:
            emit(path, ("controls", control, "contrasts", f"<resolved {key}>"),
                 identifier=f"{prefix}/resolved/{key}",
                 claim=f"{key.replace('_', ' ')} contrasts resolving under the {name} control",
                 family="position_terms", support=support, unit="contrasts", kind="count",
                 value=sum(1 for row in resolved if row["key"] == key))
        if control == "wall":
            families = ledger.artifacts.json(LINEAGE_DECLARATION)["families"]
            resolving = {row["arm"] for row in resolved}
            in_panel = {name_: [arm for arm in arms if arm in payload["arms"]]
                        for name_, (flag, arms) in families.items() if flag == "yes"}
            in_panel = {name_: arms for name_, arms in in_panel.items() if arms}
            # Legacy identifiers/selectors retain the frozen release grouping, not exposure.
            for identifier, value, claim in (
                    ("protein_families_in_panel", len(in_panel),
                     "protein-specialized or protein-adapted release families with an admitted "
                     "arm in the position panel"),
                    ("protein_families_resolving",
                     sum(1 for arms in in_panel.values() if resolving & set(arms)),
                     "protein-specialized or protein-adapted release families with an arm "
                     "resolving a contrast under the window control"),
                    ("text_arms_resolving",
                     sum(1 for name_, (flag, arms) in families.items() if flag == "no"
                         for arm in arms if arm in resolving),
                     "general-purpose text and scientific language–protein checkpoints resolving "
                     "a contrast under the window control")):
                emit(LINEAGE_DECLARATION, ("families", f"<{identifier}>"),
                     identifier=f"{prefix}/{identifier}", claim=claim, family="position_terms",
                     support=support, unit="release families" if "families" in identifier
                     else "checkpoints", kind="count", value=value)
        for index, row in enumerate(contrasts):
            emit(path, ("controls", control, "contrasts", f"[{index}]"),
                 identifier=f"{prefix}/{row['arm']}/{row['key']}",
                 claim=f"{row['arm']} {row['key'].replace('_', ' ')} held-family Spearman "
                       f"difference under the {name} control",
                 family="position_terms", support=support, unit="dimensionless Spearman",
                 kind="estimate", value=row["point"], interval=row["interval"],
                 draws=block["draws"], resampling_unit="wild-type family at 50% identity",
                 seed=SPLIT_SEEDS, level="likelihood")


def _position_layout(ledger, emit) -> None:
    path = f"{POSITION}/receipts/layout/layout_assessment.json"
    payload = ledger.artifacts.json(path)
    support = ledger.declare_support(
        "position_layout_replay",
        "191 ProteinGym assays in 156 wild-type families at 50% identity on the label-blind "
        "position-layout subset; a four-state pack replayed against a two-state pack")
    for field, block in payload["stability_summaries"].items():
        unit = _unit_of(field)
        if not unit:
            ledger.gap(id=f"position_layout/{field}",
                       claim=f"the ProGen3-3B layout {field.replace('_', ' ')}", family="position_layout",
                       reason="the field name states no unit, so no row is written",
                       looked_in=path, searched="the layout stability summaries")
            continue
        emit(path, ("stability_summaries", field), identifier=f"position_layout/progen3-3b/{field}",
             claim=f"ProGen3-3B {field.replace('_', ' ')} between a four-state and a two-state pack "
                   "of the summands the position decomposition reads",
             family="position_layout", support=support, unit=unit, kind="estimate",
             value=block["point"], interval=block["interval"], draws=block["resamples"],
             resampling_unit=block["unit"], level="likelihood")
    summary = next(iter(payload["stability_summaries"].values()))
    for field, unit in (("n_assays", "assays"), ("n_units", "source clusters")):
        emit(path, ("stability_summaries", "<any metric>", field),
             identifier=f"position_layout/progen3-3b/support/{field}",
             claim=f"position layout subset {field.replace('n_', '').replace('_', ' ')}",
             family="position_layout", support=support, unit=unit, value=summary[field])

    emit(path, ("diagnostics", "<variants>"), identifier="position_layout/progen3-3b/support/variants",
         claim="label-blind single substitutions in the position layout subset",
         family="position_layout", support=support, unit="variants", kind="support_count",
         value=sum(row["variants"] for row in payload["diagnostics"]))
    contrasts = payload["paired4_minus_paired2_contrasts"]
    resolved: list[float] = []
    recorded: list[float] = []
    total = 0
    per_control: dict[str, int] = {}
    keys: set[str] = set()
    for seed, by_control in contrasts.items():
        for control, by_key in by_control.items():
            per_control.setdefault(control, 0)
            for key, block in by_key.items():
                keys.add(key)
                total += 1
                recorded.append(abs(block["point"]))
                emit(path, ("paired4_minus_paired2_contrasts", seed, control, key),
                     identifier=f"position_layout/progen3-3b/shift/{control}/{key}/{seed}",
                     claim=f"ProGen3-3B four-state minus two-state pack shift in the fitted "
                           f"{key.replace('_', ' ')} increment under the "
                           f"{'window' if control == 'wall' else 'rejoined-suffix'} control at split "
                           f"seed {seed}",
                     family="position_layout", support=support, unit="dimensionless Spearman",
                     kind="estimate", value=block["point"], interval=block["interval"],
                     draws=summary["resamples"], resampling_unit=summary["unit"],
                     seed=(int(seed),), level="likelihood")
                if not block["interval"][0] <= 0 <= block["interval"][1]:
                    per_control[control] += 1
                    resolved.append(abs(block["point"]))
    pointer = ("paired4_minus_paired2_contrasts",)
    emit(path, pointer + ("<interval count>",), identifier="position_layout/progen3-3b/intervals",
         claim="pointwise layout intervals over the two controls, eight contrasts and three seeds",
         family="position_layout", support=support, unit="intervals", kind="count", value=total)
    emit(path, pointer + ("<contrast keys>",), identifier="position_layout/progen3-3b/contrast_keys",
         claim="distinct layout contrasts each control carries", family="position_layout",
         support=support, unit="contrasts", kind="count", value=len(keys))
    for control, count in sorted(per_control.items()):
        emit(path, pointer + (f"<{control} resolutions>",),
             identifier=f"position_layout/progen3-3b/resolutions/{control}",
             claim=f"layout intervals resolving under the "
                   f"{'window' if control == 'wall' else 'rejoined-suffix'} control",
             family="position_layout", support=support, unit="intervals", kind="count", value=count)
    alpha = summary["alpha"]
    emit(path, pointer + ("<interval count times alpha>",),
         identifier="position_layout/progen3-3b/expected_spurious_resolutions",
         claim="layout resolutions expected by chance at the pointwise level the receipts use",
         family="position_layout", support=support, unit="intervals", kind="constant",
         value=total * alpha)
    emit(path, pointer + ("<largest recorded absolute shift>",),
         identifier="position_layout/progen3-3b/largest_recorded_shift",
         claim="largest ProGen3-3B four-state minus two-state pack shift in a fitted increment "
               "among all 48 layout intervals, resolved or not",
         family="position_layout", support=support, unit="dimensionless Spearman",
         kind="range_bound", value=max(recorded), level="likelihood")
    if resolved:
        emit(path, pointer + ("<largest resolved absolute shift>",),
             identifier="position_layout/progen3-3b/largest_resolved_shift",
             claim="largest ProGen3-3B four-state minus two-state pack shift in a fitted increment "
                   "among the layout intervals that resolve; the packing-dependent component that "
                   "excludes this checkpoint from the position panel",
             family="position_layout", support=support, unit="dimensionless Spearman",
             kind="range_bound", value=max(resolved), level="likelihood")


# --------------------------------------------------------------------------- #
# Generation replication and the unconditional completion census.
# --------------------------------------------------------------------------- #

def _generation_replication(ledger, emit) -> None:
    payload = ledger.artifacts.json(REPLICATION)
    gate = {cell["cell"]: cell for cell in ledger.artifacts.json(GATE)["cells"]}
    support = ledger.declare_support(
        "generation_campaign_replication",
        "three unconditional generation campaigns at fixed checkpoints and declared decoding over "
        "20 generation cells; between-campaign spread, not training-lineage uncertainty")
    emit(REPLICATION, ("cells", "<count>"), identifier="generation_replication/cells",
         claim="generation cells the replication campaigns cover", family="generation_replication",
         support=support, unit="cells", kind="count", value=len(payload["cells"]))
    emit(REPLICATION, ("campaign_order", "<count>"), identifier="generation_replication/campaigns",
         claim="decoding campaigns the replication compares", family="generation_replication",
         support=support, unit="campaigns", kind="count", value=len(payload["campaign_order"]))
    for index in range(2):
        emit(MANIFEST, ("campaigns", f"[{index}]", "seed"),
             identifier=f"generation_replication/seed/replicate_{index + 1}",
             claim=f"decoding seed of the replication campaign replicate {index + 1}",
             family="generation_replication", support=support, unit="seed", kind="constant",
             level="generation")
    kind = "95% Student-t interval over generation streams"
    unit = "dimensionless rate difference"
    for endpoint, block in payload["stage2_minus_stage1_unconditional"].items():
        for metric, item in block.items():
            emit(REPLICATION, ("stage2_minus_stage1_unconditional", endpoint, metric, "mean"),
                 identifier=f"generation_replication/{endpoint}/{metric}",
                 claim=f"second-minus-first stage unconditional {metric.replace('_', ' ')} on the "
                       f"{endpoint.replace('_', ' ')} endpoint, averaged over campaigns",
                 family="generation_replication", support=support, unit=unit, kind="estimate",
                 value=item["mean"], interval=item["ci95"], interval_kind=kind,
                 draws=item["n_campaigns"], resampling_unit="generation stream at a fixed checkpoint",
                 level="generation")
    tally = {endpoint: {"resolved": 0, "negative": 0, "class_agree": 0, "sign_agree": 0,
                        "new_sign_agree": 0, "new_total": 0}
             for endpoint in ("any_family", "complete_domain")}
    for cell in payload["cells"]:
        arm = cell["arm"] if cell["condition"] != "requested" else f"{cell['arm']}-requested"
        for endpoint, count in tally.items():
            item = cell["endpoints"][endpoint]["model_minus_fragment"]
            low, high = item["ci95"]
            gate_item = gate[cell["cell"]]["endpoints"][endpoint]["controls"]["fragment"]
            resolved = not low <= 0 <= high
            gate_class = 0 if gate_item["ci97_5"][0] <= 0 <= gate_item["ci97_5"][1] else (
                1 if gate_item["difference"] > 0 else -1)
            replication_class = (1 if item["mean"] > 0 else -1) if resolved else 0
            count["resolved"] += resolved
            count["negative"] += resolved and item["mean"] < 0
            count["class_agree"] += gate_class == replication_class
            count["sign_agree"] += (item["mean"] > 0) == (gate_item["difference"] > 0)
            for value in item["values"][1:]:
                count["new_total"] += 1
                count["new_sign_agree"] += (value > 0) == (gate_item["difference"] > 0)
            emit(REPLICATION, ("cells", f"[{cell['cell']}]", "endpoints", endpoint,
                               "model_minus_fragment", "mean"),
                 identifier=f"generation_replication/{arm}/{endpoint}/model_minus_fragment",
                 claim=f"{arm} model-minus-fragment {endpoint.replace('_', ' ')} recognition-rate "
                       "difference over three generation streams",
                 family="generation_replication", support=support, unit=unit, kind="estimate",
                 value=item["mean"], interval=item["ci95"], interval_kind=kind,
                 draws=item["n_campaigns"], resampling_unit="generation stream at a fixed checkpoint",
                 level="generation")
    for endpoint, count in tally.items():
        for field, claim in (
                ("resolved", "cells whose three-stream contrast interval excludes zero"),
                ("negative", "cells whose three-stream contrast resolves negative"),
                ("class_agree", "cells whose resolution class matches the gate's"),
                ("sign_agree", "cells whose three-stream point sign matches the gate's"),
                ("new_sign_agree", "cell-streams of the two new campaigns whose sign matches the gate's"),
                ("new_total", "cell-streams the two new campaigns supply")):
            emit(REPLICATION, ("cells", f"<{field}>"),
                 identifier=f"generation_replication/{endpoint}/count/{field}",
                 claim=f"{endpoint.replace('_', ' ')} {claim}", family="generation_replication",
                 support=support, unit="cells" if "stream" not in claim else "cell-streams",
                 kind="count", value=count[field])
    multiplier = {}
    for label, item in (("three", payload["cells"][0]["endpoints"]["any_family"]["model_minus_fragment"]),
                        ("two", payload["cells"][0]["decoder_stop_rates_two_new_campaigns"]["eos"])):
        low, high = item["ci95"]
        multiplier[label] = (high - low) / 2 / (item["standard_deviation"] / item["n_campaigns"] ** 0.5)
    for label, degrees in (("three", 2), ("two", 1)):
        emit(REPLICATION, ("cells", "[0]", "<t multiplier>"),
             identifier=f"generation_replication/t_multiplier/{label}_streams",
             claim=f"Student-t multiplier of a {label}-stream campaign interval with {degrees} "
                   "degrees of freedom", family="generation_replication", support=support,
             unit="dimensionless", kind="constant", value=multiplier[label], level="generation")


def _native_expression(ledger, emit) -> None:
    """Counts of the native-strength conditioning of the representation increment."""
    payload = ledger.artifacts.json(NATIVE_EXPRESSION)
    families = ledger.artifacts.json(LINEAGE_DECLARATION)["families"]
    protein = {arm for flag, arms in families.values() if flag == "yes" for arm in arms}
    support = ledger.declare_support(
        "native_expression_anchor",
        "201 ProteinGym assays in 163 wild-type clusters, the readout anchor, for 33 checkpoints "
        "at three split seeds; the conditioning reads each checkpoint's held-out cluster ranks")
    arms = payload["arms"]

    def every_seed(arm, increment, name, sign):
        bounds = [arms[arm][seed]["conditioned"][increment][name]["interval"] for seed in arms[arm]]
        return all((high < 0) if sign < 0 else (low > 0) for low, high in bounds)

    emit(NATIVE_EXPRESSION, ("arms", "<count>"), identifier="native_expression/arms",
         claim="checkpoints the native-strength conditioning covers", family="native_expression",
         support=support, unit="checkpoints", kind="count", value=len(arms))
    for increment, name, sign, label, claim in (
            ("increment_R_C_P", "rho_control_vs_increment", -1, "control_vs_increment_negative",
             "checkpoints whose control-competence versus representation-increment rank correlation "
             "is negative at every split"),
            ("increment_R_C_P", "rho_control_vs_increment", 1, "control_vs_increment_positive",
             "checkpoints whose control-competence versus representation-increment rank correlation "
             "is positive at every split"),
            ("increment_R_C_P", "rho_native_vs_increment_given_control", 1, "partial_positive",
             "checkpoints whose native-strength versus representation-increment relation, given the "
             "control, is positive at every split"),
            ("increment_R_after_M_C_P", "rho_native_vs_increment_given_control", -1,
             "partial_negative_after_likelihood",
             "checkpoints whose native-strength versus post-likelihood representation-increment "
             "relation, given the control, is negative at every split")):
        hits = [arm for arm in arms if every_seed(arm, increment, name, sign)]
        for group, members in (("all", hits), ("text", [a for a in hits if a not in protein]),
                               ("protein", [a for a in hits if a in protein])):
            if group != "all" and label != "control_vs_increment_negative":
                continue
            # Legacy group identifiers/selectors remain for compatibility, not exposure.
            group_label = {
                "all": "all",
                "text": "general-purpose text and scientific language–protein",
                "protein": "protein-specialized or protein-adapted",
            }[group]
            emit(NATIVE_EXPRESSION, ("arms", f"<{label}>"),
                 identifier=f"native_expression/{label}/{group}",
                 claim=f"{group_label} {claim}", family="native_expression", support=support,
                 unit="checkpoints", kind="count", value=len(members))


def _native_completion(ledger, emit) -> None:
    path = f"{PROGEN3}/generation_summary.json"
    support = "uncond_census_800"
    for field, claim in (("attempts", "ProGen3-3B unconditional attempts"),
                         ("official_compilation_valid",
                          "ProGen3-3B unconditional attempts the official compiler accepts")):
        emit(path, (field,), identifier=f"generation_census/progen3-3b/native/{field}",
             claim=claim, family="generation_census", support=support, unit="attempts",
             kind="census", level="generation",
             reason="a complete enumeration of the saved attempts")
    ledger_path = f"{PROGEN3}/attempts.jsonl"
    counts: dict[str, int] = {}
    for line in ledger.artifacts.text(ledger_path).splitlines():
        if not line.strip():
            continue
        reason = json.loads(line).get("source_stop_reason")
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
    for reason, value in sorted(counts.items()):
        emit(ledger_path, ("<attempt rows>", "source_stop_reason", reason),
             identifier=f"generation_census/progen3-3b/native/{reason}",
             claim=f"ProGen3-3B unconditional attempts whose recorded stop reason is "
                   f"{reason.replace('_', ' ')}",
             family="generation_census", support=support, unit="attempts", kind="census",
             level="generation", value=value,
             reason="a complete enumeration of the saved attempt ledger")
