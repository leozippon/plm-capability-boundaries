"""How many readout designs resolve, and in how many release families.

The Supplementary Information states how far the readout-design search got: how
many checkpoints showed a design whose increment over the baseline resolved, and
how many released model families those checkpoints came from. Both numbers turn
on a criterion no artifact declares, so the criterion is declared here and
carried on every support description: *a design resolves for an arm when its
increment excludes zero and is positive in every admitted cell the pair carries,
that is at every split seed and panel the gate ran it at.* An arm resolves when at
least one design resolves for it. Nothing weaker is admitted, because an arm that
resolves at one seed and not at another has not resolved, and nothing stronger,
because requiring three seeds of a gate that ran a design at one would report the
gate's design rather than the arm's result.

Two gates ran readout designs and they resolve different sets of nine arms. The
class-axis sweep varies the readout class on one fixed pair of extraction depths
and adds four pooled-depth designs; the depth panel sweeps extraction depth and
position. Neither set contains the other, and the Supplementary sentence names
both design groups, so the union is what it describes: eleven arms over six
release families, where each gate on its own resolves nine arms, over four
families for the class axis and six for the depth panel.

A release-family count is a join. Which arms resolve is fixed by a gate artifact;
which family an arm belongs to is fixed by the canonical released-checkpoint-family
mapping, which is a property of the releases and of no outcome. Neither artifact
fixes the count alone, and a row carries one source digest, so each row's digest
fixes the artifact its own pointer reads, each support description lists every
artifact in that row's join with its digest, and the one reading no artifact
records --- the join itself --- is registered as a gap rather than presented as a
reading. The mapping is read in both places it is persisted and refuses if they
have diverged, because the count turns on it.

Two refusals protect the family counts. An arm that resolves a design and carries
no release family is a refusal, not a silent drop and not an invented family: the
mapping covers the thirty-three released checkpoints and a count over an arm
outside it is a count over something else. An arm that a gate's roster carries and
the mapping does not is recorded as a gap, because it can enter no family count at
all; ZymCTRL is that arm in both gates, and it resolves no design under the
criterion above.

Sixteen release families are not sixteen independent training trials, which the
mapping's own record states, so every claim here says "release families" and
never "independent lineages".
"""
from __future__ import annotations

from src.capability.core.evidence_ledger import LedgerError
# The emit contract --- how a row carries its locator, digest and pointer --- is
# the follow-up module's.
from scripts.capability.reporting.followup_quantities import _emitter

CLASS_SWEEP = "results/R2/readout_sweeps_20260924/sweep_cell_excerpt.json"
CLASS_SCHEMA = "readout_sweep_cell_excerpt"
CELL_SCHEMA = "readout_class_sweep"
DEPTH_PANEL = "results/R2/readout_depth_panel_20260925/readout_depth_panel.json"
DEPTH_SCHEMA = "d1_readout_depth_panel_v1"
FOLLOWUP = "results/shared/followup_support_20260927"
LINEAGE_DECLARATION = f"{FOLLOWUP}/lineage_declaration.json"
LINEAGE_ROBUSTNESS = f"{FOLLOWUP}/lineage_robustness.json"
LINEAGE_SCHEMA = "local_context_release_family_robustness_v1"

#: The designs the class-axis gate runs: five readout classes and four
#: pooled-depth rules. The roster is checked against what the cells carry,
#: because a tenth design would make the reduction a reduction over another set.
DESIGNS = ("C0_compressed_linear_baseline_grid", "C1_compressed_linear", "C2_full_linear",
           "C3_compressed_random_feature", "C4_full_random_feature", "D_final_last",
           "D_final_mean", "D_middle_last", "D_middle_mean")
#: The summary a design's increment is read from, and the pointer suffix the
#: class-axis cells spell it with.
INCREMENT = "delta_spearman"

CHECKPOINTS = "checkpoints"
RELEASE_FAMILIES = "release families"


def add_readout_design_resolution_quantities(ledger) -> None:
    emit = _emitter(ledger)
    mapping = _release_family_mapping(ledger)
    gates = (_class_axis(ledger), _depth_panel(ledger))
    _unmapped_roster_arms(ledger, mapping, gates)
    for gate in gates:
        _gate_counts(ledger, emit, mapping, gate)
    _union_counts(ledger, emit, mapping, gates)
    _joined_count_gap(ledger, gates)


# --------------------------------------------------------------------------- #
# The canonical mapping, read in both places it is persisted.
# --------------------------------------------------------------------------- #

def _release_family_mapping(ledger) -> dict:
    """Arm to release family, with the basis and the ancestry limit it carries.

    The mapping is persisted twice: as the declaration the local-context
    release-family analysis was computed against, and as that analysis's own
    family records. One number turns on it, so both are read and a divergence
    refuses rather than picking one; the record names the declaration's digest,
    which is what makes the pair checkable at all.
    """
    robustness = ledger.artifacts.json(LINEAGE_ROBUSTNESS)
    if robustness["schema"] != LINEAGE_SCHEMA:
        raise LedgerError(f"{LINEAGE_ROBUSTNESS}: not a {LINEAGE_SCHEMA} record")
    recorded, now = robustness["declaration_sha256"], ledger.artifacts.sha256(LINEAGE_DECLARATION)
    if recorded != now:
        raise LedgerError(
            f"{LINEAGE_ROBUSTNESS}: the record was computed against a release-family declaration "
            f"digesting to {recorded} and {LINEAGE_DECLARATION} now digests to {now}, so the two no "
            "longer state one mapping")
    declaration = ledger.artifacts.json(LINEAGE_DECLARATION)
    declared = {name: tuple(members) for name, (_, members) in declaration["families"].items()}
    persisted = {name: tuple(record["checkpoints"])
                 for name, record in robustness["family_records"].items()}
    if declared != persisted:
        raise LedgerError(
            f"the release-family mapping differs between {LINEAGE_DECLARATION} and "
            f"{LINEAGE_ROBUSTNESS} (declaration families {sorted(declared)}, record families "
            f"{sorted(persisted)}), so there is no one mapping a family count can be taken under")
    arms: dict[str, str] = {}
    for name, members in sorted(persisted.items()):
        for arm in members:
            if arm in arms:
                raise LedgerError(
                    f"{LINEAGE_ROBUSTNESS}: {arm} is a checkpoint of both {arms[arm]} and {name}, "
                    "so it has no one release family and a family count over it is ambiguous")
            arms[arm] = name
    return {"arms": arms, "families": persisted, "basis": declaration["mapping_basis"],
            "limit": robustness["training_ancestry_limit"]}


def _families(arms: set[str], mapping: dict, where: str) -> set[str]:
    """The release families a set of resolving arms belongs to, or a refusal."""
    outside = sorted(arms - set(mapping["arms"]))
    if outside:
        raise LedgerError(
            f"{where}: {', '.join(outside)} resolve a readout design and carry no release family in "
            f"{LINEAGE_ROBUSTNESS}, so a family count over them could only drop them silently or "
            "invent a family for them")
    return {mapping["arms"][arm] for arm in arms}


# --------------------------------------------------------------------------- #
# The class-axis sweep: five readout classes and four pooled-depth rules.
# --------------------------------------------------------------------------- #

def _class_axis(ledger) -> dict:
    payload = ledger.artifacts.json(CLASS_SWEEP)
    if payload["schema"] != CLASS_SCHEMA:
        raise LedgerError(f"{CLASS_SWEEP}: not a {CLASS_SCHEMA} excerpt")
    cells = payload["cells"]["class"]
    if len(cells) != payload["counts"]["class"]:
        raise LedgerError(f"{CLASS_SWEEP}: the excerpt holds {len(cells)} class-axis cells and "
                          f"counts {payload['counts']['class']}, so one of the two is incomplete")
    contrasts, conventions, anchors = set(), set(), set()
    arms, panels, seeds, strata = set(), set(), set(), set()
    observed: dict[tuple[str, str], list[dict]] = {}
    design_seeds: dict[str, set[int]] = {}
    carried: set[str] = set()
    for cell in cells:
        block = cell["block"]
        if block["schema_version"] != CELL_SCHEMA or block["status"] != "complete":
            raise LedgerError(
                f"{CLASS_SWEEP}: a class-axis cell of {block['arm']} is "
                f"{block['schema_version']}/{block['status']} and not a complete {CELL_SCHEMA}, so "
                "it is not an admitted fit the criterion can be applied to")
        contrasts.add(block["primary_contrast"])
        arms.add(block["arm"])
        panels.add(block["panel"])
        seeds.add(block["fold_seed"])
        strata.add(block["stratum"])
        anchors.add((block["n_assays"], block["n_families"], block["n_variants"]))
        for key, summary in block["summaries"].items():
            if not key.endswith(f"/{INCREMENT}"):
                continue
            design = key[: -len(f"/{INCREMENT}")]
            carried.add(design)
            if design not in DESIGNS:
                continue
            observed.setdefault((block["arm"], design), []).append(summary)
            design_seeds.setdefault(design, set()).add(block["fold_seed"])
            conventions.add((summary["resamples"], summary["unit"], summary["minimum_units"]))
    if carried != set(DESIGNS):
        raise LedgerError(
            f"{CLASS_SWEEP}: the class-axis cells carry increments for designs {sorted(carried)} "
            f"and not the declared {sorted(DESIGNS)}; a reduction over a different roster is a "
            "different count")
    if len(contrasts) != 1:
        raise LedgerError(f"{CLASS_SWEEP}: the class-axis cells declare contrasts "
                          f"{sorted(contrasts)}, so the increment the criterion is applied to is "
                          "not one quantity")
    if len(conventions) != 1:
        raise LedgerError(f"{CLASS_SWEEP}: the class-axis increments are resampled under "
                          f"{sorted(conventions)}, so there is no one interval convention behind "
                          "the criterion")
    draws, unit, floor = conventions.pop()
    contrast = contrasts.pop()
    resolving: dict[str, set[str]] = {}
    for (arm, design), summaries in observed.items():
        if all(summary["excludes_zero"] and summary["point"] > 0 for summary in summaries):
            resolving.setdefault(arm, set()).add(design)
    single = {arm for arm, designs in resolving.items()
              if all(len(design_seeds[design]) == 1 for design in designs)}
    return {
        "name": "class-axis readout sweep", "path": CLASS_SWEEP, "segment": "class_axis",
        "pointer": ("cells", "class"), "roster": arms, "resolving": set(resolving),
        "single_seed_only": single,
        "population": (
            f"{len(cells)} admitted class-axis fit cells over {len(arms)} released checkpoints, "
            f"{len(panels)} panels, {len(strata)} input strata and the split seeds "
            f"{', '.join(str(seed) for seed in sorted(seeds))}; the largest panel holds "
            f"{max(anchors)[0]} assays in {max(anchors)[1]} wild-type clusters at 50% identity over "
            f"{max(anchors)[2]:,} variants and the smallest {min(anchors)[0]} assays in "
            f"{min(anchors)[1]} clusters over {min(anchors)[2]:,} variants. The increment is the "
            f"design's own {INCREMENT}: {contrast}"),
        "detail": (
            f"each increment is resampled over the {unit} with {draws:,} draws and carries no "
            f"interval below {floor} units. The five readout-class designs are run at all "
            f"{len(seeds)} split seeds and the four pooled-depth designs at one "
            f"({min(design_seeds['D_final_mean'])}), so \"every split seed the cell carries\" is "
            f"{len(seeds)} seeds for a class design and one for a pooled-depth design; "
            f"{len(single)} of the {len(resolving)} arms this gate resolves do so only through a "
            "pooled-depth design the gate ran at a single split seed"),
    }


# --------------------------------------------------------------------------- #
# The depth panel: extraction depth and position.
# --------------------------------------------------------------------------- #

def _depth_panel(ledger) -> dict:
    payload = ledger.artifacts.json(DEPTH_PANEL)
    if payload["schema_version"] != DEPTH_SCHEMA or payload["status"] != "complete":
        raise LedgerError(f"{DEPTH_PANEL}: {payload['schema_version']}/{payload['status']} is not a "
                          f"complete {DEPTH_SCHEMA} panel")
    entries = payload["seed_consistent"]
    arms, axes, panels, strata, seeds = set(), set(), set(), set(), set()
    for entry in entries:
        if not entry["minimum"] > 0:
            raise LedgerError(
                f"{DEPTH_PANEL}: a seed-consistent block of {entry['arm']} records a minimum of "
                f"{entry['minimum']} over its seeds, so it is not positive at every split seed and "
                "the list is not the set the criterion reads")
        if entry["seeds"] < 1:
            raise LedgerError(f"{DEPTH_PANEL}: a seed-consistent block of {entry['arm']} carries "
                              f"{entry['seeds']} seeds, so it rests on no split seed at all")
        arms.add(entry["arm"])
        axes.add(entry["axis"])
        panels.add(entry["panel"])
        strata.add(entry["stratum"])
        seeds.add(entry["seeds"])
    return {
        "name": "readout depth panel", "path": DEPTH_PANEL, "segment": "depth_panel",
        "pointer": ("seed_consistent",), "roster": set(payload["arms"]), "resolving": arms,
        "population": (
            f"{payload['n_cells']} fitted depth-panel cells over {len(payload['arms'])} released "
            f"checkpoints and {len(payload['panels'])} panels, of which {len(entries)} "
            f"depth-and-position blocks resolve positively at every split seed they carry, in "
            f"{len(panels)} panels on the {', '.join(sorted(strata))} stratum over the "
            f"{', '.join(sorted(axes))} axes. The increment is the block's depth-contrast "
            f"{INCREMENT} against the admitted depth"),
        "detail": (
            "the panel's summariser already applied the criterion, listing only blocks whose "
            "increment resolved above zero at every declared seed; what is re-read here is that "
            f"the minimum over those seeds is positive, and that each block rests on "
            f"{' or '.join(str(count) for count in sorted(seeds))} split seeds"),
    }


# --------------------------------------------------------------------------- #
# The supports: the criterion, the population and the artifacts of the join.
# --------------------------------------------------------------------------- #

CRITERION = (
    "A design resolves for an arm when that arm's readout-design increment over the baseline "
    "excludes zero and is positive in every admitted cell the pair carries, that is at every split "
    "seed and panel the gate ran it at; an arm resolves when at least one design resolves for it, "
    "and a release family resolves when at least one of its released checkpoints does")


def _mapping_clause(mapping: dict) -> str:
    checkpoints = sum(len(members) for members in mapping["families"].values())
    return (
        f"The release-family counts on this support reduce the resolving arms through the canonical "
        f"released-checkpoint-family mapping of {len(mapping['families'])} families over "
        f"{checkpoints} released checkpoints, whose declared basis is '{mapping['basis']}'. The "
        f"record states its own ancestry limit: \"{mapping['limit']}\" So these are release "
        "families and never independent lineages")


def _join_clause(ledger, paths: tuple[str, ...]) -> str:
    listed = "; ".join(f"{path} (sha256 {ledger.artifacts.sha256(path)})" for path in paths)
    return (
        f"The counts on this support are fixed jointly by {len(paths)} retained artifacts and by no "
        f"one of them alone: {listed}. A row carries one source digest, so the digest on a row fixes "
        "the artifact its own pointer reads and this list fixes the rest")


def _support(ledger, mapping: dict, support_id: str, population: str, detail: str,
             paths: tuple[str, ...]) -> str:
    return ledger.declare_support(
        support_id,
        f"{population}; {detail}. {CRITERION}. {_mapping_clause(mapping)}. "
        f"{_join_clause(ledger, paths)}")


# --------------------------------------------------------------------------- #
# The counts.
# --------------------------------------------------------------------------- #

def _gate_counts(ledger, emit, mapping: dict, gate: dict) -> None:
    segment, path, pointer = gate["segment"], gate["path"], gate["pointer"]
    support = _support(ledger, mapping, f"readout_design_resolution/{segment}",
                       gate["population"], gate["detail"],
                       (path, LINEAGE_ROBUSTNESS, LINEAGE_DECLARATION))
    families = _families(gate["resolving"], mapping, f"{path}: {segment}")
    emit(path, pointer + ("<arms resolving a readout design>",),
         identifier=f"readout_design_resolution/{segment}/arms_resolving",
         claim=f"released checkpoints for which at least one readout design of the {gate['name']} "
               "resolves: its increment over the baseline excludes zero and is positive at every "
               "split seed the cell carries",
         family="readout_design_resolution", support=support, unit=CHECKPOINTS, kind="count",
         value=len(gate["resolving"]), level="representation")
    emit(LINEAGE_ROBUSTNESS, ("family_records", "<families of the resolving arms>"),
         identifier=f"readout_design_resolution/{segment}/release_families_resolving",
         claim=f"release families holding at least one checkpoint for which a readout design of "
               f"the {gate['name']} resolves under that criterion. The count is a join: which arms "
               f"resolve is fixed by {path} and which family an arm belongs to by the canonical "
               "released-checkpoint-family mapping this row's digest fixes, and neither artifact "
               "fixes it alone",
         family="readout_design_resolution", support=support, unit=RELEASE_FAMILIES, kind="count",
         value=len(families), level="representation")
    if "single_seed_only" not in gate:
        return
    emit(path, pointer + ("<arms resolving only a single-seed design>",),
         identifier=f"readout_design_resolution/{segment}/"
                    "arms_resolving_only_at_one_split_seed",
         claim=f"released checkpoints that resolve a readout design of the {gate['name']} only "
               "through a pooled-depth design the gate ran at a single split seed, so that the "
               "all-seeds criterion rests on one seed for them rather than on three",
         family="readout_design_resolution", support=support, unit=CHECKPOINTS, kind="count",
         value=len(gate["single_seed_only"]), level="representation")


def _union_counts(ledger, emit, mapping: dict, gates: tuple[dict, ...]) -> None:
    arms = set().union(*(gate["resolving"] for gate in gates))
    names = " and the ".join(gate["name"] for gate in gates)
    paths = tuple(gate["path"] for gate in gates)
    support = _support(
        ledger, mapping, "readout_design_resolution/union",
        "the union of two gates' populations. "
        + ". ".join(f"The {gate['name']}: {gate['population']}; {gate['detail']}"
                    for gate in gates),
        "neither gate's resolving set contains the other, so the union is a reading of both and of "
        "neither alone", paths + (LINEAGE_ROBUSTNESS, LINEAGE_DECLARATION))
    families = _families(arms, mapping, "the union of the readout-design gates")
    emit(paths[0], gates[0]["pointer"] + ("<arms resolving in either gate>",),
         identifier="readout_design_resolution/union/arms_resolving",
         claim=f"released checkpoints for which at least one readout design resolves in the {names}"
               " taken together, under the all-seeds criterion. Neither gate's set contains the "
               "other, so this is the union of two gates and not a reading of either; the row's "
               f"digest fixes {paths[0]} and the support description fixes the rest",
         family="readout_design_resolution", support=support, unit=CHECKPOINTS, kind="count",
         value=len(arms), level="representation")
    emit(LINEAGE_ROBUSTNESS, ("family_records", "<families of the union's resolving arms>"),
         identifier="readout_design_resolution/union/release_families_resolving",
         claim=f"release families holding at least one checkpoint for which a readout design "
               f"resolves in the {names} taken together. This is the count the Supplementary "
               "sentence describes, because that sentence names both design groups; it is a join "
               "across both gate artifacts and the canonical released-checkpoint-family mapping, "
               "and no single artifact fixes it",
         family="readout_design_resolution", support=support, unit=RELEASE_FAMILIES, kind="count",
         value=len(families), level="representation")


# --------------------------------------------------------------------------- #
# What no artifact records.
# --------------------------------------------------------------------------- #

def _unmapped_roster_arms(ledger, mapping: dict, gates: tuple[dict, ...]) -> None:
    """Arms a gate ran that the canonical mapping does not cover.

    Such an arm can enter no release-family count, whatever it resolves, so its
    absence from the mapping is recorded rather than left to be read off a count
    that is one smaller than the roster.
    """
    for gate in gates:
        outside = sorted(gate["roster"] - set(mapping["arms"]))
        if not outside:
            continue
        ledger.gap(
            id=f"readout_design_resolution/{gate['segment']}/unmapped_arms",
            claim=f"the release family of {', '.join(outside)}, which the {gate['name']} runs",
            family="readout_design_resolution",
            reason=f"the canonical released-checkpoint-family mapping covers "
                   f"{len(mapping['arms'])} released checkpoints and does not carry "
                   f"{', '.join(outside)}, so no release-family count on this gate can include "
                   "them; under the all-seeds criterion none of them resolves any design, so no "
                   "resolving arm is dropped by this",
            looked_in=f"{LINEAGE_ROBUSTNESS} and {LINEAGE_DECLARATION}",
            reachability="not_recorded",
            searched="every release family of the canonical mapping")


def _joined_count_gap(ledger, gates: tuple[dict, ...]) -> None:
    paths = ", ".join(gate["path"] for gate in gates)
    ledger.gap(
        id="readout_design_resolution/joined_counts",
        claim="the readout-design arm and release-family counts as a reading of one retained "
              "artifact rather than as a join this table composes",
        family="readout_design_resolution",
        reason="no retained artifact applies the all-seeds criterion, maps the resolving arms onto "
               f"release families, or unions the two gates: {paths} each record their own cells and "
               f"{LINEAGE_ROBUSTNESS} records the mapping, and the criterion is declared in this "
               "module and nowhere on disk. Each row therefore carries the digest of the artifact "
               "its own pointer reads and its support description carries the digests of the "
               "others, because one digest cannot fix a number two or three artifacts fix",
        looked_in=f"{paths}, {LINEAGE_ROBUSTNESS}, {LINEAGE_DECLARATION}",
        reachability="not_recorded",
        searched="every reduction block of both gate artifacts and of the release-family record")
