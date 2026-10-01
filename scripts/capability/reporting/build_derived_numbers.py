#!/usr/bin/env python3
"""Derive the manuscript's numbers from the admitted artifacts, once.

Reads the analysis artifacts and gate records the manuscript rests
on and writes one table of every quantity it cites, each carrying its value,
unit, support description, interval, resampling unit, draw count, seed set,
level and the digest of the file it came from. It also writes the figure-data
contract: each manuscript figure bound to its input tables and, through them,
to the artifact digests those tables were extracted from.

No model is loaded, no cohort is drawn, no fit is run and no interval is
resampled. Existing intervals are copied with their resampling unit attached.
A locator that does not resolve is recorded as a gap and produces no row.

Run from the validated ct environment:

    python scripts/capability/reporting/build_derived_numbers.py
"""
from __future__ import annotations

from pathlib import Path
from statistics import median
import argparse
import csv
import hashlib
import json
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.capability.core.evidence_ledger import (  # noqa: E402
    REPO_ROOT,
    Artifacts,
    Ledger,
    LedgerError,
    Quantity,
    checked_sum,
    resolve,
)
# The cross-measure artifact records neither its draw count nor its seed, so
# both are taken from the module that ran the resampling rather than written
# here as literals. A literal drifted from 4,000 to a copied 2,000 once
# already, and a constant no file binds is exactly what that drift needs.
from src.capability.context.cross_measure_association import (  # noqa: E402
    BOOTSTRAP as CROSS_MEASURE_DRAWS,
    SEED as CROSS_MEASURE_SEED,
)

OUT_DIR = REPO_ROOT / 'results/shared/manuscript_evidence_20260927'
FIGURE_DIR = REPO_ROOT / "scripts" / "capability" / "reporting" / "figure_data"

PERCENTILE_95 = "95% percentile"
READOUT_ROOT = "results/R2/readout_expansion_20260923"
EPISTASIS = "archive/logs/R3/pairwise_epistasis_20260924"
SPLIT_SEEDS = (20260923, 20260924, 20260925)

#: Supports more than one family reads, declared once so the two families cannot
#: describe the same support differently.
PAIRWISE_ALL_SUPPORT = (
    "64 held family groups of MegaScale wild-type-centred double-mutant stability cycles, 8192 "
    "cycles over 217 distinct site pairs"
)
ANCHOR_SUPPORT = (
    "common-support anchor, a separate seeded draw of 128 variants per assay: 201 ProteinGym "
    "assays in 163 wild-type clusters over 25,728 variants"
)
CENSUS_SUPPORT = (
    "800 native unconditional attempts per checkpoint under one decoding recipe and one seed "
    "stream, a complete enumeration of the saved batches"
)


def generator_gate_support(cells: int) -> str:
    return (f"{cells} generation cells at 800 attempts each against seven matched generator cohorts, "
            "paired differences resampled over the ledger's frozen near-duplicate sequence groups")


def _block(payload: object, pointer: tuple[str, ...]) -> dict:
    node = resolve(payload, pointer)
    if not isinstance(node, dict):
        raise KeyError("/".join(pointer))
    return node


def _interval(block: dict) -> tuple[float, float] | None:
    for key in ("interval", "ci95", "ci_95", "interval95"):
        if key in block and block[key] is not None:
            low, high = block[key]
            return float(low), float(high)
    return None


def _point(block: dict) -> float:
    for key in ("point", "mean", "difference", "value", "estimate"):
        if key in block and block[key] is not None:
            return float(block[key])
    raise KeyError(f"no point estimate in {sorted(block)}")


#: How an artifact spells the number of resampling draws behind an interval.
DRAW_KEYS = ("resamples", "draws", "bootstrap_draws", "finite_draws", "bootstrap")
#: Where an artifact declares one draw count for its whole bootstrap.
DRAW_DECLARATIONS = (("bootstrap",), ("declaration", "bootstrap"), ("recipe", "bootstrap"),
                     ("resampling",))
#: The draw count of the bootstraps whose artifacts record none. It is a
#: declared design value taken from the analysis modules, not a reading, and
#: every row that carries it is registered as a gap by
#: ``family_unrecorded_draw_counts`` so the substitution is visible rather than
#: silent. Never use it where ``_draws`` can reach a recorded count.
UNRECORDED_DRAWS = 2000


def _draws(*blocks: object) -> int | None:
    """The draw count the artifact records, from the first place that carries one.

    A draw count written into this script is a provenance field with no
    source, and two of them have already disagreed with the run that produced
    the artifact. Reading it keeps the table and the artifact from drifting
    apart; returning ``None`` when nothing records it makes the absence
    explicit at the call site rather than defaulting under it.
    """
    for block in blocks:
        for pointer in ((),) + DRAW_DECLARATIONS:
            try:
                node = resolve(block, pointer) if pointer else block
            except KeyError:
                continue
            if not isinstance(node, dict):
                continue
            for key in DRAW_KEYS:
                value = node.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                    return int(value)
    return None


#: One draw count per artifact, memoised by path and digest so a file that
#: changes between builds is read again.
_ARTIFACT_DRAWS: dict[tuple[str, str], int | None] = {}


def _collect_draws(node: object, found: set[int]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in DRAW_KEYS and isinstance(value, (int, float)) \
                    and not isinstance(value, bool) and value > 0:
                found.add(int(value))
            _collect_draws(value, found)
    elif isinstance(node, list):
        for value in node:
            _collect_draws(value, found)


def _artifact_draws(ledger: Ledger, path: str) -> int | None:
    """The one draw count an artifact records, wherever in the file it records it.

    Some artifacts declare the bootstrap once at the top, some record it beside
    each estimate, and some record it beside the summaries and not beside the
    per-seed verdicts derived from the same resampling. Reading every
    occurrence and requiring them to agree gives the file's own figure and
    returns nothing when the file disagrees with itself, so a disagreement
    surfaces as a refused row rather than as a number picked from one branch.
    """
    key = (path, ledger.artifacts.sha256(path))
    if key not in _ARTIFACT_DRAWS:
        found: set[int] = set()
        _collect_draws(ledger.artifacts.json(path), found)
        _ARTIFACT_DRAWS[key] = found.pop() if len(found) == 1 else None
    return _ARTIFACT_DRAWS[key]


# --------------------------------------------------------------------------- #
# Family 1 --- mutation-effect ranking on each checkpoint's own native support
# --------------------------------------------------------------------------- #

NATIVE_SOURCES = [
    ("results/R1/progen2_scale_20260826/scale_capability.json", ("dms",), "ProGen2 ladder"),
    ("results/R1/progen3_stage_20260917/second_stage_capability/second_stage_capability.json", ("dms",), "ProGen3 pair"),
    ("results/R1/adaptation_lineage_20260813/adaptation_stage_capability.json", ("dms",), "Llama-2--ProLLaMA lineage"),
    ("results/R1/native_dms_20260913/20260915134508_95c515bf6881/rita_xl_independent_analyse/"
     "native_dms_comparison.json", ("groups", "rita-xl"), "RITA-xl"),
    ("results/R1/source_recovery_20260917/galactica_native_dms_comparison.json",
     ("groups", "galactica"), "Galactica ladder"),
]
NATIVE_ENDPOINTS = {
    "raw_spearman": ("equal-cluster mean within-assay Spearman between model score and measured effect",
                     "dimensionless Spearman"),
    "model_minus_lookup": ("paired Spearman difference from the implemented position-specific profile",
                           "dimensionless Spearman"),
    "model_minus_blosum62": ("paired Spearman difference from the BLOSUM62 substitution-matrix control",
                             "dimensionless Spearman"),
}


def native_support_counts(ledger: Ledger, support: str, n_assays: int, n_units: int,
                           path: str, pointer: tuple[str, ...]) -> None:
    for field, value, unit, claim in (("assays", n_assays, "assays", "assays on this native support"),
                                      ("clusters", n_units, "wild-type clusters",
                                       "wild-type clusters on this native support")):
        identifier = f"native_fitness/{support}/{field}"
        if identifier in {quantity.id for quantity in ledger.quantities}:
            continue
        ledger.add(Quantity(
            id=identifier, claim=claim, family="native_fitness", value=float(value), unit=unit,
            kind="support_count", support_id=support, level="likelihood",
            source_path=path, source_sha256=ledger.artifacts.sha256(path), source_pointer=pointer,
        ))


def native_support_id(ledger: Ledger, n_assays: int, n_units: int) -> str:
    return ledger.declare_support(
        f"native_cap1000_{n_assays}a_{n_units}c",
        f"native support, seeded variant sample capped at 1,000 per assay: {n_assays} ProteinGym "
        f"substitution assays in {n_units} wild-type clusters at 50% identity",
    )


def family_native_fitness(ledger: Ledger, strata: dict[str, str]) -> None:
    rows: list[tuple[str, str, float, float | None, int]] = []
    for path, prefix, label in NATIVE_SOURCES:
        payload = ledger.artifacts.json(path)
        for endpoint, (claim, unit) in NATIVE_ENDPOINTS.items():
            try:
                per_rung = _block(payload, prefix + (endpoint, "per_rung"))
            except KeyError:
                ledger.gap(id=f"native_fitness/{label}/{endpoint}", claim=claim, family="native_fitness",
                           reason="endpoint absent from the artifact", looked_in=path)
                continue
            for arm, value in per_rung.items():
                if value.get("degenerate"):
                    ledger.gap(id=f"native_fitness/{arm}/{endpoint}", claim=claim, family="native_fitness",
                               reason=f"degenerate: {value.get('degenerate_reason')}", looked_in=path)
                    continue
                interval = _interval(value)
                support = native_support_id(ledger, int(value["n_assays"]), int(value["n_units"]))
                native_support_counts(ledger, support, int(value["n_assays"]), int(value["n_units"]),
                                      path, prefix + (endpoint, "per_rung", arm))
                ledger.add(Quantity(
                    id=f"native_fitness/{arm}/{endpoint}",
                    claim=f"{arm}: {claim}",
                    family="native_fitness",
                    value=_point(value), unit=unit, kind="estimate", support_id=support,
                    interval=interval, interval_kind=PERCENTILE_95,
                    resampling_unit=value["unit"], resampling_draws=int(value["resamples"]),
                    level="likelihood",
                    source_path=path, source_sha256=ledger.artifacts.sha256(path),
                    source_pointer=prefix + (endpoint, "per_rung", arm),
                ))
                if endpoint == "model_minus_lookup" and interval is not None:
                    rows.append((arm, strata.get(arm, "unknown"), _point(value), interval[0], int(value["n_units"])))

    path = "results/R1/source_recovery_20260917/proteinglm_retrieval_bound.json"
    payload = ledger.artifacts.json(path)
    for endpoint, key in (("model_minus_lookup", "delta_lookup"), ("model_minus_blosum62", "delta_blosum62")):
        value = _block(payload, ("arms", "proteinglm-7b-clm", key))
        interval = _interval(value)
        support = native_support_id(ledger, int(value["n_assays"]), int(value["n_units"]))
        claim, unit = NATIVE_ENDPOINTS[endpoint]
        ledger.add(Quantity(
            id=f"native_fitness/proteinglm-7b-clm/{endpoint}",
            claim=f"proteinglm-7b-clm: {claim}", family="native_fitness",
            value=_point(value), unit=unit, kind="estimate", support_id=support,
            interval=interval, interval_kind=PERCENTILE_95,
            resampling_unit=value["unit"], resampling_draws=int(value["resamples"]),
            level="likelihood",
            source_path=path, source_sha256=ledger.artifacts.sha256(path),
            source_pointer=("arms", "proteinglm-7b-clm", key),
        ))
        if endpoint == "model_minus_lookup" and interval is not None:
            rows.append(("proteinglm-7b-clm", strata.get("proteinglm-7b-clm", "unknown"),
                         _point(value), interval[0], int(value["n_units"])))

    # The paired profile contrast exists for only part of the panel, so the
    # denominator of any count over it is the set of checkpoints carrying it
    # and not the 19 protein-sequence interfaces of the extraction panel.
    carrying = rows
    above = [row for row in carrying if row[3] > 0.0]
    unit_counts = sorted({row[4] for row in carrying})
    support = ledger.declare_support(
        "native_profile_contrast_cohort",
        "the checkpoints carrying a paired profile contrast on their own native support, wild-type "
        f"clusters {min(unit_counts)}--{max(unit_counts)}",
    )
    source = f"{READOUT_ROOT}/final_admission.json"
    ledger.add(Quantity(
        id="native_fitness/panel/checkpoints_carrying_profile_contrast",
        claim="checkpoints carrying the paired profile contrast on their own native support",
        family="native_fitness", value=float(len(carrying)), unit="checkpoints", kind="support_count",
        support_id=support, level="likelihood",
        source_path=source, source_sha256=ledger.artifacts.sha256(source),
        source_pointer=("interfaces",),
    ))
    ledger.add(Quantity(
        id="native_fitness/panel/checkpoints_exceeding_profile",
        claim="checkpoints whose paired profile contrast resolves above zero on their own native support",
        family="native_fitness", value=float(len(above)), unit="checkpoints", kind="count",
        support_id=support, level="likelihood", verdict="supported",
        source_path=source, source_sha256=ledger.artifacts.sha256(source),
        source_pointer=("interfaces",),
    ))
    # The manuscript's denominator of 7 is a narrower grouping than either
    # declaration this repository retains: all 14 checkpoints carrying the
    # contrast are ``native_sequence`` interfaces, and the protein-pretrained
    # grouping the local-context gate uses would include both ProLLaMA stages.
    ledger.gap(
        id="native_fitness/panel/protein_sequence_subset_denominator",
        claim="the protein-sequence-model subset of the checkpoints carrying the paired profile "
              "contrast, which is the denominator of the profile-exceeding count",
        family="native_fitness",
        reason="no retained artifact or panel declaration states this grouping: all "
               f"{len(carrying)} checkpoints carrying the contrast are declared native_sequence "
               "interfaces, and the protein-pretrained grouping the local-context gate applies would "
               "include both ProLLaMA stages, so the denominator is not reproducible from a declaration",
        looked_in=source, reachability="not_measured",
    )


def family_native_paired_lineage(ledger: Ledger) -> None:
    path = "results/R1/adaptation_lineage_20260813/adaptation_stage_capability.json"
    payload = ledger.artifacts.json(path)
    for endpoint in NATIVE_ENDPOINTS:
        try:
            pairs = _block(payload, ("dms", endpoint, "adjacent_delta_rho"))
        except KeyError:
            continue
        for pair, value in pairs.items():
            if value.get("degenerate"):
                continue
            support = native_support_id(ledger, int(value["n_assays"]), int(value["n_units"]))
            ledger.add(Quantity(
                id=f"lineage_paired/{pair}/{endpoint}",
                claim=f"paired difference from {pair.replace('__', ' to ')} in {endpoint}",
                family="native_fitness", value=_point(value), unit="dimensionless Spearman",
                kind="estimate", support_id=support, interval=_interval(value),
                interval_kind=PERCENTILE_95, resampling_unit=value["unit"],
                resampling_draws=int(value["resamples"]), level="likelihood",
                source_path=path, source_sha256=ledger.artifacts.sha256(path),
                source_pointer=("dms", endpoint, "adjacent_delta_rho", pair),
            ))


# --------------------------------------------------------------------------- #
# Family 2 --- the literal amino-acid-string extension
# --------------------------------------------------------------------------- #

def family_text_aa(ledger: Ledger) -> None:
    path = "results/R1/source_recovery_20260917/text_aa_analyse.json"
    payload = ledger.artifacts.json(path)
    per_model = _block(payload, ("common_all13", "per_model"))
    support = None
    for arm, block in per_model.items():
        for endpoint in ("raw", "model_minus_lookup", "model_minus_blosum62"):
            value = block[endpoint]
            if value.get("degenerate"):
                continue
            support = ledger.declare_support(
                f"text_aa_{int(value['n_assays'])}a_{int(value['n_units'])}c",
                "pre-scoring intersection read as literal uppercase amino-acid strings: "
                f"{int(value['n_assays'])} assays in {int(value['n_units'])} wild-type clusters",
            )
            ledger.add(Quantity(
                id=f"text_aa/{arm}/{endpoint}",
                claim=f"{arm} reading sequences as literal amino-acid strings: {endpoint}",
                family="text_aa", value=_point(value), unit="dimensionless Spearman",
                kind="estimate", support_id=support, interval=_interval(value),
                interval_kind=PERCENTILE_95, resampling_unit=value["unit"],
                resampling_draws=int(value["resamples"]), level="likelihood",
                source_path=path, source_sha256=ledger.artifacts.sha256(path),
                source_pointer=("common_all13", "per_model", arm, endpoint),
            ))
    if support is None:
        ledger.gap(id="text_aa/panel", claim="literal-string extension", family="text_aa",
                   reason="no non-degenerate cell", looked_in=path)


# --------------------------------------------------------------------------- #
# Family 3 --- the frozen-state readout on the common-support anchor
# --------------------------------------------------------------------------- #

READOUT_ENDPOINTS = {
    "readout_minus_likelihood": ("R_minus_raw_M_spearman",
                                 "fitted readout minus the checkpoint's own native likelihood", "representation"),
    "representation_increment": ("delta_spearman",
                                 "increment of the 1,024 representation coordinates over the matched "
                                 "supervised baseline", "representation"),
    "matched_baseline": ("B_spearman", "the matched supervised baseline's own held-cluster Spearman", "control"),
    "native_likelihood": ("raw_M_spearman", "the native likelihood's held-cluster Spearman", "likelihood"),
}


def readout_reports() -> list[Path]:
    root = REPO_ROOT / READOUT_ROOT / "complete/results/external_baseline"
    return sorted(root.glob("*/readout_*/readout_*.json"))


def family_readout(ledger: Ledger, strata: dict[str, str]) -> dict[str, list[dict]]:
    support = ledger.declare_support("anchor_128", ANCHOR_SUPPORT)
    cells: dict[str, list[dict]] = {name: [] for name in READOUT_ENDPOINTS}
    seen: set[tuple[str, int]] = set()
    for report in readout_reports():
        relative = report.relative_to(REPO_ROOT).as_posix()
        payload = ledger.artifacts.json(relative)
        if payload.get("n_assays") != 201 or payload.get("n_families") != 163:
            continue
        summaries = payload.get("summaries") or {}
        if "R_minus_raw_M_spearman" not in summaries:
            continue
        arm, seed = payload["arm"], int(payload["fold_seed"])
        if (arm, seed) in seen:
            continue
        seen.add((arm, seed))
        for name, (key, claim, level) in READOUT_ENDPOINTS.items():
            value = summaries[key]
            if value.get("degenerate"):
                ledger.gap(id=f"readout/{arm}/{seed}/{name}", claim=claim, family="readout",
                           reason=f"degenerate: {value.get('degenerate_reason')}", looked_in=relative)
                continue
            quantity = ledger.add(Quantity(
                id=f"readout/{arm}/{seed}/{name}",
                claim=f"{arm} at split seed {seed}: {claim}", family="readout",
                value=_point(value), unit="dimensionless Spearman", kind="estimate",
                support_id=support, interval=_interval(value), interval_kind=PERCENTILE_95,
                resampling_unit=value["unit"], resampling_draws=int(value["resamples"]),
                seed_set=(seed,), level=level,
                conditioning="readout_selection" if level == "representation" else None,
                source_path=relative, source_sha256=ledger.artifacts.sha256(relative),
                source_pointer=("summaries", key),
            ))
            cells[name].append({"arm": arm, "seed": seed, "stratum": strata.get(arm, "unknown"),
                                "value": quantity.value, "low": quantity.interval[0],
                                "excludes_zero": bool(value["excludes_zero"])})

    admission = f"{READOUT_ROOT}/final_admission.json"
    for name, rows in cells.items():
        if not rows:
            continue
        level = READOUT_ENDPOINTS[name][2]
        arms = sorted({row["arm"] for row in rows})
        text_rows = [row for row in rows if row["stratum"] == "literal_text_AA"]
        _range_bounds(ledger, f"readout/panel/{name}", f"panel range of the {name} over "
                      f"{len(rows)} checkpoint--seed cells", "readout", rows, "anchor_128", level,
                      admission, ledger.artifacts.sha256(admission))
        ledger.add(Quantity(
            id=f"readout/panel/{name}/cells",
            claim=f"checkpoint--seed cells carrying the {name} contrast",
            family="readout", value=float(len(rows)), unit="checkpoint--seed cells",
            kind="support_count", support_id="anchor_128", level=level,
            source_path=admission, source_sha256=ledger.artifacts.sha256(admission),
            source_pointer=("interfaces",),
        ))
        ledger.add(Quantity(
            id=f"readout/panel/{name}/cells_above_zero",
            claim=f"checkpoint--seed cells whose {name} resolves above zero",
            family="readout", value=float(sum(row["excludes_zero"] and row["value"] > 0 for row in rows)),
            unit="checkpoint--seed cells", kind="count", support_id="anchor_128", level=level,
            source_path=admission, source_sha256=ledger.artifacts.sha256(admission),
            source_pointer=("interfaces",),
        ))
        ledger.add(Quantity(
            id=f"readout/panel/{name}/checkpoints_without_any_resolved_cell",
            claim=f"unconditioned checkpoints whose {name} resolves above zero in no cell",
            family="readout",
            value=float(len(arms) - sum(any(row["excludes_zero"] and row["value"] > 0
                                            for row in rows if row["arm"] == arm) for arm in arms)),
            unit="checkpoints", kind="count", support_id="anchor_128", level=level,
            verdict="not_detected",
            source_path=admission, source_sha256=ledger.artifacts.sha256(admission),
            source_pointer=("interfaces",),
        ))
        ledger.add(Quantity(
            id=f"readout/panel/{name}/checkpoints_with_any_resolved_cell",
            claim=f"checkpoints whose {name} resolves above zero in at least one of its cells",
            family="readout",
            value=float(sum(any(row["excludes_zero"] and row["value"] > 0
                                for row in rows if row["arm"] == arm) for arm in arms)),
            unit="checkpoints", kind="count", support_id="anchor_128", level=level,
            source_path=admission, source_sha256=ledger.artifacts.sha256(admission),
            source_pointer=("interfaces",),
        ))
        ledger.add(Quantity(
            id=f"readout/panel/{name}/cells_unresolved_against_zero",
            claim=f"checkpoint--seed cells whose {name} interval contains zero",
            family="readout", value=float(sum(not row["excludes_zero"] for row in rows)),
            unit="checkpoint--seed cells", kind="count", support_id="anchor_128", level=level,
            verdict="unresolved",
            source_path=admission, source_sha256=ledger.artifacts.sha256(admission),
            source_pointer=("interfaces",),
        ))
        ledger.add(Quantity(
            id=f"readout/panel/{name}/checkpoints_above_zero_every_seed",
            claim=f"checkpoints whose {name} resolves above zero at every seed it carries",
            family="readout",
            value=float(sum(all(row["excludes_zero"] and row["value"] > 0
                                for row in rows if row["arm"] == arm) for arm in arms)),
            unit="checkpoints", kind="count", support_id="anchor_128", level=level,
            source_path=admission, source_sha256=ledger.artifacts.sha256(admission),
            source_pointer=("interfaces",),
        ))
        if text_rows:
            ledger.add(Quantity(
                id=f"readout/literal_text/{name}/cells",
                claim=f"literal amino-acid-string checkpoint--seed cells carrying the {name} contrast",
                family="readout", value=float(len(text_rows)), unit="checkpoint--seed cells",
                kind="support_count", support_id="anchor_128", level=level,
                source_path=admission, source_sha256=ledger.artifacts.sha256(admission),
                source_pointer=("interfaces", "<stratum=literal_text_AA>"),
            ))
            ledger.add(Quantity(
                id=f"readout/literal_text/{name}/checkpoints_unresolved_against_zero",
                claim=f"literal amino-acid-string checkpoints whose {name} interval contains zero at "
                      "every seed they carry",
                family="readout",
                value=float(sum(all(not row["excludes_zero"] for row in text_rows if row["arm"] == arm)
                                for arm in sorted({row["arm"] for row in text_rows}))),
                unit="checkpoints", kind="count", support_id="anchor_128", level=level,
                verdict="unresolved",
                source_path=admission, source_sha256=ledger.artifacts.sha256(admission),
                source_pointer=("interfaces", "<stratum=literal_text_AA>"),
            ))
            _range_bounds(ledger, f"readout/literal_text/{name}",
                          f"range of the {name} over the {len(text_rows)} literal-string cells",
                          "readout", text_rows, "anchor_128", level, admission,
                          ledger.artifacts.sha256(admission))
    return cells


def _range_bounds(ledger: Ledger, prefix: str, claim: str, family: str, rows: list[dict],
                  support_id: str, level: str, source_path: str, source_sha256: str) -> None:
    values = [row["value"] for row in rows]
    for bound, value in (("min", min(values)), ("max", max(values))):
        ledger.add(Quantity(
            id=f"{prefix}/{bound}",
            claim=f"{claim} ({bound})", family=family, value=value,
            unit="dimensionless Spearman", kind="range_bound", support_id=support_id, level=level,
            source_path=source_path, source_sha256=source_sha256,
            source_pointer=("<reduction>", bound),
        ))


# --------------------------------------------------------------------------- #
# Family 4 --- measured pairwise interaction
# --------------------------------------------------------------------------- #

EPISTASIS_CONTRASTS = {
    "C|ADDITIVE_NULL": ("the sequence and profile control over the zero-interaction additive null", "control"),
    "C_G|C": ("the nonlinear-additive nuisance over the sequence and profile control", "control"),
    "C_G_T|C_G": ("the tokenisation block over the nuisance-adjusted control", "control"),
    "C_G_T_M1|C_G_T": ("the two constituent single-mutant likelihood differences, no interaction term",
                       "likelihood"),
    "C_G_T+M|C_G_T": ("the likelihood interaction over the control", "likelihood"),
    "C_G_T+R|C_G_T": ("the four-state representation interaction contrast over the control", "representation"),
    "C_G_T_M1+M|C_G_T_M1": ("the likelihood interaction over its own first-order control", "likelihood"),
    "C_G_T_R1+R|C_G_T_R1": ("the representation interaction over its own first-order coordinates",
                            "representation"),
}
KCAL2 = "kcal^2/mol^2"


def family_pairwise(ledger: Ledger) -> None:
    path = f"{EPISTASIS}/panel/panel.json"
    payload = ledger.artifacts.json(path)
    supports = {}
    for key, block in payload["supports"].items():
        description = (
            f"{block['groups']} held family groups of MegaScale wild-type-centred double-mutant "
            f"stability cycles, {block['cycles']} cycles over {block['site_pairs']} distinct site pairs"
            + ("" if key == "all" else ", the support on which the pairwise comparator is admissible")
        )
        supports[key] = ledger.declare_support(f"pairwise_{key}_{block['groups']}g", description)
        for field, unit, kind in (("groups", "family groups", "support_count"),
                                  ("cycles", "double-mutant cycles", "support_count"),
                                  ("site_pairs", "site pairs", "support_count"),
                                  ("states", "distinct measured states", "support_count")):
            ledger.add(Quantity(
                id=f"pairwise/{key}/support/{field}",
                claim=f"{field.replace('_', ' ')} on the {key} support", family="pairwise",
                value=float(block[field]), unit=unit, kind=kind, support_id=supports[key],
                level="measurement",
                source_path=path, source_sha256=ledger.artifacts.sha256(path),
                source_pointer=("supports", key, field),
            ))

    draws = int(payload["recipe"]["bootstrap"]["draws"])
    resampling_unit = str(payload["recipe"]["bootstrap"]["unit"])
    for support_key, panel in payload["panel"].items():
        support_id = supports[support_key]
        for contrast, block in panel.items():
            claim, level = EPISTASIS_CONTRASTS.get(contrast, (f"the {contrast} contrast", "measurement"))
            for arm, arm_block in block["arms"].items():
                for seed, cell in arm_block["per_seed"].items():
                    if cell.get("mse_reduction_kcal2") is None:
                        continue
                    ledger.add(Quantity(
                        id=f"pairwise/{support_key}/{contrast}/{arm}/{seed}",
                        claim=f"{arm} at seed {seed}: paired reduction in group-equal squared error from "
                              f"{claim}",
                        family="pairwise", value=float(cell["mse_reduction_kcal2"]), unit=KCAL2,
                        kind="estimate", support_id=support_id,
                        interval=tuple(map(float, cell["interval"])) if cell.get("interval") else None,
                        interval_kind=PERCENTILE_95 if cell.get("interval") else None,
                        resampling_unit=resampling_unit if cell.get("interval") else None,
                        resampling_draws=draws if cell.get("interval") else None,
                        no_interval_reason=None if cell.get("interval") else "no interval in the artifact",
                        seed_set=(int(seed),), level=level,
                        source_path=path, source_sha256=ledger.artifacts.sha256(path),
                        source_pointer=("panel", support_key, contrast, "arms", arm, "per_seed", seed,
                                        "mse_reduction_kcal2"),
                    ))
            resolved = block.get("arms_with_every_seed_interval_above_zero") or []
            below = block.get("arms_with_every_seed_interval_below_zero") or []
            verdict = "supported" if resolved else "not_detected"
            ledger.add(Quantity(
                id=f"pairwise/{support_key}/{contrast}/checkpoints_above_zero_every_seed",
                claim=f"checkpoints whose reduction from {claim} resolves above zero at all three seeds",
                family="pairwise", value=float(len(resolved)), unit="checkpoints", kind="count",
                support_id=support_id, seed_set=SPLIT_SEEDS, level=level, verdict=verdict,
                source_path=path, source_sha256=ledger.artifacts.sha256(path),
                source_pointer=("panel", support_key, contrast, "arms_with_every_seed_interval_above_zero"),
            ))
            ledger.add(Quantity(
                id=f"pairwise/{support_key}/{contrast}/checkpoints_below_zero_every_seed",
                claim=f"checkpoints whose reduction from {claim} resolves below zero at all three seeds",
                family="pairwise", value=float(len(below)), unit="checkpoints", kind="count",
                support_id=support_id, seed_set=SPLIT_SEEDS, level=level,
                source_path=path, source_sha256=ledger.artifacts.sha256(path),
                source_pointer=("panel", support_key, contrast, "arms_with_every_seed_interval_below_zero"),
            ))
            ranking = block.get("seed_mean_ranking") or []
            seed_means = [row["seed_mean_mse_reduction_kcal2"] for row in ranking
                          if row.get("seed_mean_mse_reduction_kcal2") is not None]
            if seed_means:
                ledger.add(Quantity(
                    id=f"pairwise/{support_key}/{contrast}/panel_median",
                    claim=f"panel median over {len(seed_means)} checkpoints of the seed-mean reduction "
                          f"from {claim}",
                    family="pairwise", value=float(median(seed_means)), unit=KCAL2, kind="estimate",
                    support_id=support_id, no_interval_reason="a median over checkpoints, not a resampled "
                    "estimate; the panel's intervals are per checkpoint",
                    seed_set=SPLIT_SEEDS, level=level,
                    source_path=path, source_sha256=ledger.artifacts.sha256(path),
                    source_pointer=("panel", support_key, contrast, "seed_mean_ranking"),
                ))

        nuisance = payload["nuisance_diagnostics"][support_key]
        r2 = nuisance["first_stage_held_out_weighted_r2"]
        for label, value in (("min", r2[0]), ("median", r2[1]), ("max", r2[2])):
            ledger.add(Quantity(
                id=f"pairwise/{support_key}/nuisance/first_stage_r2_{label}",
                claim=f"{label} group-equal held-out weighted R^2 of the nonlinear-additive nuisance's "
                      f"first stage over {nuisance['folds']} outer folds",
                family="pairwise", value=float(value), unit="dimensionless R^2",
                kind="range_bound" if label != "median" else "estimate", support_id=support_id,
                no_interval_reason="a fold-level order statistic, not a resampled estimate",
                level="control",
                source_path=path, source_sha256=ledger.artifacts.sha256(path),
                source_pointer=("nuisance_diagnostics", support_key, "first_stage_held_out_weighted_r2"),
            ))

    _pairwise_additive_null(ledger, supports)
    _pairwise_kish(ledger, supports)
    _pairwise_indel(ledger, supports)


def _pairwise_additive_null(ledger: Ledger, supports: dict[str, str]) -> None:
    path = f"{EPISTASIS}/panel/fit_progen3-3b.json"
    payload = ledger.artifacts.json(path)
    for support_key, support_id in supports.items():
        pointer = ("supports", support_key, "seeds", str(SPLIT_SEEDS[0]), "designs", "ADDITIVE_NULL",
                   "group_equal_mse_kcal2")
        try:
            block = _block(payload, pointer)
        except KeyError:
            ledger.gap(id=f"pairwise/{support_key}/additive_null",
                       claim="group-equal mean squared error of the zero-interaction additive null",
                       family="pairwise", reason="pointer does not resolve", looked_in=path)
            continue
        ledger.add(Quantity(
            id=f"pairwise/{support_key}/additive_null",
            claim="group-equal mean squared error of the zero-interaction additive null",
            family="pairwise", value=_point(block), unit=KCAL2, kind="estimate", support_id=support_id,
            interval=_interval(block), interval_kind=PERCENTILE_95,
            resampling_unit="held family group", resampling_draws=_draws(block, payload),
            seed_set=(SPLIT_SEEDS[0],),
            level="measurement",
            source_path=path, source_sha256=ledger.artifacts.sha256(path), source_pointer=pointer,
        ))


def _pairwise_kish(ledger: Ledger, supports: dict[str, str]) -> None:
    """The effective count depends on the weighting it is computed under.

    The fit records name the weighting beside the count, so the convention is
    read from the artifact and carried on the row. The cycle-share convention
    the readiness record quotes is not recorded in any artifact, and that is a
    recorded gap rather than a number this table supplies.
    """
    path = f"{EPISTASIS}/panel/fits_lost_state_sensitivity/fit_progen3-3b.json"
    if not ledger.artifacts.exists(path):
        ledger.gap(id="pairwise/kish", claim="Kish effective site pairs with their weighting",
                   family="pairwise", reason="no fit record carrying an effective-count field",
                   looked_in=path)
        return
    payload = ledger.artifacts.json(path)
    for support_key, support_id in supports.items():
        pointer = ("supports", support_key, "kish_effective_site_pairs")
        try:
            value = resolve(payload, pointer)
            convention = str(resolve(payload, ("supports", support_key, "kish_weighting")))
        except KeyError:
            ledger.gap(id=f"pairwise/{support_key}/kish",
                       claim="Kish effective site pairs with the weighting they were computed under",
                       family="pairwise", reason="the fit record carries no effective-count field for "
                       "this support", looked_in=path)
            continue
        ledger.add(Quantity(
            id=f"pairwise/{support_key}/kish_effective_site_pairs",
            claim=f"Kish effective site pairs on the {support_key} support",
            family="pairwise", value=float(value), unit="effective site pairs",
            kind="effective_count", support_id=support_id,
            weighting_convention=convention, level="measurement",
            source_path=path, source_sha256=ledger.artifacts.sha256(path), source_pointer=pointer,
        ))
    ledger.gap(
        id="pairwise/kish_cycle_share",
        claim="Kish effective site pairs under the cycle-share convention, which weights a site pair "
              "by its share of its group's cycles",
        family="pairwise",
        reason="no artifact records this convention: the fit records carry only the weighting the "
               "estimator applies, so a cycle-share count cannot be quoted as the power of these "
               "intervals from anything in the evidence",
        looked_in=path, reachability="not_recorded",
        searched="every supports block of every retained pairwise fit record, and the whole epistasis "
                 "results tree for a cycle-share field",
    )


def _pairwise_indel(ledger: Ledger, supports: dict[str, str]) -> None:
    path = f"{EPISTASIS}/indel_affected_cycles.json"
    if not ledger.artifacts.exists(path):
        ledger.gap(id="pairwise/indel_exclusion", claim="the states a post-freeze defect admitted on "
                   "insertion or deletion rows", family="pairwise",
                   reason="artifact absent", looked_in=path)
        return
    payload = ledger.artifacts.json(path)
    ledger.add(Quantity(
        id="pairwise/indel_exclusion/backgrounds",
        claim="backgrounds carrying a measured state the post-freeze insertion or deletion defect touches",
        family="pairwise", value=float(len(payload)), unit="backgrounds", kind="census",
        support_id=supports["all"], level="measurement",
        source_path=path, source_sha256=ledger.artifacts.sha256(path), source_pointer=("<keys>",),
    ))
    ledger.add(Quantity(
        id="pairwise/indel_exclusion/states",
        claim="measured states the post-freeze insertion or deletion defect touches",
        family="pairwise",
        value=float(sum(len(block.get("states", {})) for block in payload.values())),
        unit="distinct measured states", kind="census",
        support_id=supports["all"], level="measurement",
        source_path=path, source_sha256=ledger.artifacts.sha256(path), source_pointer=("<states>",),
    ))



# --------------------------------------------------------------------------- #
# Family 5 --- endpoint reliability and the structural contrast
# --------------------------------------------------------------------------- #

def _agreement_unit(key: str) -> str:
    if key.endswith("_kcal_mol"):
        return "kcal/mol"
    if "ratio" in key or key in {"pearson_r", "within_background_spearman"}:
        return "dimensionless"
    return "dimensionless"


def _emit_agreement(ledger: Ledger, *, prefix: str, family: str, path: str, pointer: tuple[str, ...],
                    agreement: dict, support_id: str, resampling_unit: str) -> None:
    draws = _draws(ledger.artifacts.json(path))
    for key, value in sorted(agreement.items()):
        if not isinstance(value, dict):
            continue
        try:
            point = _point(value)
        except KeyError:
            continue
        interval = _interval(value)
        ledger.add(Quantity(
            id=f"{prefix}/{key}",
            claim=f"{key.replace('_', ' ')} of the {prefix.rsplit('/', 1)[-1].replace('_', ' ')} endpoint",
            family=family, value=point, unit=_agreement_unit(key), kind="estimate",
            support_id=support_id, interval=interval,
            interval_kind=PERCENTILE_95 if interval else None,
            resampling_unit=resampling_unit if interval else None,
            resampling_draws=draws if interval else None,
            no_interval_reason=None if interval else "the artifact retains no interval for this quantity",
            seed_set=(SPLIT_SEEDS[0],), level="measurement",
            source_path=path, source_sha256=ledger.artifacts.sha256(path),
            source_pointer=pointer + (key,),
        ))


def family_endpoint_reliability(ledger: Ledger) -> None:
    path = "results/R4/gate_stability_20260924/endpoint/endpoint_qualification.json"
    payload = ledger.artifacts.json(path)
    pointer = ("agreement", "natural", "family_group")
    block = _block(payload, pointer)
    support = ledger.declare_support(
        "stability_singles_family_group",
        f"MegaScale single substitutions, {block['units']} wild-type family groups",
    )
    _emit_agreement(ledger, prefix="endpoint/single_substitution", family="endpoint", path=path,
                    pointer=pointer + ("agreement",), agreement=block["agreement"],
                    support_id=support, resampling_unit=str(block["unit"]))
    # The same endpoint grouped by background rather than by family, which is
    # the support the manuscript's variant and background counts name.
    background_pointer = ("agreement", "natural", "background")
    try:
        background = _block(payload, background_pointer)
    except KeyError:
        background = None
    if background is not None:
        background_support = ledger.declare_support(
            "stability_singles_background",
            f"MegaScale single substitutions, {background['units']} backgrounds",
        )
        ledger.add(Quantity(
            id="endpoint/single_substitution/backgrounds",
            claim="backgrounds carrying the admitted single-substitution endpoint",
            family="endpoint", value=float(background["units"]), unit="backgrounds",
            kind="support_count", support_id=background_support, level="measurement",
            source_path=path, source_sha256=ledger.artifacts.sha256(path),
            source_pointer=background_pointer + ("units",),
        ))
        _emit_agreement(ledger, prefix="endpoint/single_substitution_by_background", family="endpoint",
                        path=path, pointer=background_pointer + ("agreement",),
                        agreement=background["agreement"], support_id=background_support,
                        resampling_unit=str(background.get("unit", "background")))
    for field, value in sorted(_block(payload, ("agreement", "natural")).items()):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        unit = _leaf_unit(field, "kcal/mol")
        if not unit:
            continue
        ledger.add(Quantity(
            id=f"endpoint/single_substitution/cohort_{field}",
            claim=f"{field.replace('_', ' ')} over the admitted single-substitution rows, as the "
                  "endpoint qualification records it",
            family="endpoint", value=float(value), unit=unit, kind="artifact_leaf",
            support_id=support, level="measurement",
            source_path=path, source_sha256=ledger.artifacts.sha256(path),
            source_pointer=("agreement", "natural", field),
        ))
    ledger.add(Quantity(
        id="endpoint/single_substitution/units",
        claim="wild-type family groups behind the single-substitution reliability estimate",
        family="endpoint", value=float(block["units"]), unit="family groups", kind="support_count",
        support_id=support, level="measurement",
        source_path=path, source_sha256=ledger.artifacts.sha256(path),
        source_pointer=pointer + ("units",),
    ))
    ledger.add(Quantity(
        id="endpoint/single_substitution/effective_units",
        claim="Kish effective wild-type family groups behind the single-substitution reliability estimate",
        family="endpoint", value=float(block["effective_units_kish"]), unit="effective family groups",
        kind="effective_count", support_id=support,
        weighting_convention="the weighting this reliability estimator applies, as the artifact records it "
                             "beside the estimate",
        level="measurement",
        source_path=path, source_sha256=ledger.artifacts.sha256(path),
        source_pointer=pointer + ("effective_units_kish",),
    ))

    path = "archive/logs/R3/pairwise_label_instrument_20260923/qualification.json"
    payload = ledger.artifacts.json(path)
    pointer = ("variants", "both_channel_width", "kinds", "natural", "all", "source_cluster")
    block = _block(payload, pointer)
    support = ledger.declare_support(
        "megascale_double_cycle_source_cluster",
        f"MegaScale double-mutant cycles, {block['units']} source clusters",
    )
    _emit_agreement(ledger, prefix="endpoint/double_mutant_cycle", family="endpoint", path=path,
                    pointer=pointer + ("agreement",), agreement=block["agreement"],
                    support_id=support, resampling_unit=str(block.get("unit", "source cluster")))
    ledger.add(Quantity(
        id="endpoint/double_mutant_cycle/units",
        claim="source clusters behind the double-mutant cycle reliability estimate",
        family="endpoint", value=float(block["units"]), unit="source clusters", kind="support_count",
        support_id=support, level="measurement",
        source_path=path, source_sha256=ledger.artifacts.sha256(path), source_pointer=pointer + ("units",),
    ))
    ledger.add(Quantity(
        id="endpoint/double_mutant_cycle/effective_units",
        claim="Kish effective source clusters behind the double-mutant cycle reliability estimate",
        family="endpoint", value=float(block["effective_units_kish"]), unit="effective source clusters",
        kind="effective_count", support_id=support,
        weighting_convention="the weighting this reliability estimator applies, as the artifact records it "
                             "beside the estimate",
        level="measurement",
        source_path=path, source_sha256=ledger.artifacts.sha256(path),
        source_pointer=pointer + ("effective_units_kish",),
    ))


def family_higher_order(ledger: Ledger) -> None:
    path = "archive/logs/shared/gate_higher_order_extended_20260923/his3_synonymous.json"
    payload = ledger.artifacts.json(path)
    for index, block in enumerate(payload["orders"]):
        order = int(block["order"])
        support = ledger.declare_support(
            f"his3_order{order}",
            f"order-{order} fitness cycles of the {payload['assay']} deep-mutational-scanning assay "
            f"whose synonymous nucleotide genotypes form an independent replicate channel, "
            f"{block['site_tuples']} site tuples",
        )
        value = block["shared_to_discordance_ratio"]
        verdict = "unresolved" if _interval(value)[0] <= 0.0 else "supported"
        ledger.add(Quantity(
            id=f"higher_order/order_{order}/shared_to_discordance_ratio",
            claim=f"reproducible-across-channel component of the order-{order} fitness cycle as a "
                  "fraction of its own per-channel discordance",
            family="higher_order", value=_point(value), unit="dimensionless ratio", kind="estimate",
            support_id=support, interval=_interval(value), interval_kind=PERCENTILE_95,
            resampling_unit=str(payload["resampling_unit"]),
            resampling_draws=_draws(value, block, payload),
            seed_set=(SPLIT_SEEDS[0],), level="measurement", verdict=verdict,
            source_path=path, source_sha256=ledger.artifacts.sha256(path),
            source_pointer=("orders", f"[{index}]", "shared_to_discordance_ratio"),
        ))
        for field, unit, kind, convention in (
            ("site_tuples", "site tuples", "support_count", None),
            ("effective_site_tuples_kish", "effective site tuples", "effective_count",
             "the weighting this reliability estimator applies: site tuples equal"),
        ):
            if field not in block:
                continue
            ledger.add(Quantity(
                id=f"higher_order/order_{order}/{field}",
                claim=f"{field.replace('_', ' ')} behind the order-{order} reliability estimate",
                family="higher_order", value=float(block[field]), unit=unit, kind=kind,
                support_id=support, weighting_convention=convention, level="measurement",
                source_path=path, source_sha256=ledger.artifacts.sha256(path),
                source_pointer=("orders", f"[{index}]", field),
            ))

    path = "archive/logs/shared/gate_higher_order_extended_20260923/proteingym_census.json"
    payload = ledger.artifacts.json(path)
    support = ledger.declare_support("proteingym_local_census",
                                     "the local ProteinGym substitution assays, complete enumeration")
    for pointer, claim, unit in _census_fields(payload):
        ledger.add(Quantity(
            id="higher_order/census/" + "_".join(pointer), claim=claim, family="higher_order",
            value=float(resolve(payload, pointer)), unit=unit, kind="census", support_id=support,
            level="measurement",
            source_path=path, source_sha256=ledger.artifacts.sha256(path), source_pointer=pointer,
        ))


def _census_fields(payload: object) -> list[tuple[tuple[str, ...], str, str]]:
    fields = []
    for key, value in (payload.items() if isinstance(payload, dict) else []):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            fields.append(((key,), f"census field {key.replace('_', ' ')}",
                           "assays" if "assay" in key else "variants"))
    return fields


def family_structure_contact(ledger: Ledger) -> None:
    path = "results/R3/contact_bootstrap_corrected_20260927/epsilon_enrichment.json"
    payload = ledger.artifacts.json(path)
    primary = payload["primary_support"]
    for endpoint, claim in (
        ("mean_abs_epsilon", "contact minus cell-matched non-contact difference in measured nonadditivity"),
        ("mean_abs_epsilon_adjusted", "the same difference once the global response to the additive "
         "prediction is removed"),
        ("mean_abs_half_channel_difference", "the same estimator applied to the half-difference between "
         "the two protease channels, which measures instrument discordance"),
    ):
        pointer = ("supports", primary, "definitions", "heavy_atom", "endpoints", endpoint, "site_pair_equal")
        block = _block(payload, pointer)
        support = ledger.declare_support(
            f"contact_{primary}",
            f"{block['contact_site_pairs']} contact and {block['control_site_pairs']} cell-matched "
            "control site pairs on the insertion/deletion-excluded cycle support",
        )
        interval = _interval(block)
        ledger.add(Quantity(
            id=f"structure_contact/{endpoint}", claim=claim, family="structure_contact",
            value=_point(block), unit=str(block["unit"]), kind="estimate", support_id=support,
            interval=interval, interval_kind=PERCENTILE_95,
            resampling_unit="held group, then site pair inside it", resampling_draws=_draws(block, payload),
            seed_set=(SPLIT_SEEDS[0],),
            level="measurement", verdict="unresolved" if interval[0] <= 0.0 <= interval[1] else "supported",
            source_path=path, source_sha256=ledger.artifacts.sha256(path), source_pointer=pointer,
        ))
        for field, unit in (("contact_site_pairs", "site pairs"), ("control_site_pairs", "site pairs")):
            key = f"structure_contact/{field}"
            if key in {q.id for q in ledger.quantities}:
                continue
            ledger.add(Quantity(
                id=key, claim=f"{field.replace('_', ' ')} in the structural contrast",
                family="structure_contact", value=float(block[field]), unit=unit, kind="support_count",
                support_id=support, level="measurement",
                source_path=path, source_sha256=ledger.artifacts.sha256(path),
                source_pointer=pointer + (field,),
            ))
        if "effective_control_site_pairs" in block and "structure_contact/effective_control_site_pairs" \
                not in {q.id for q in ledger.quantities}:
            ledger.add(Quantity(
                id="structure_contact/effective_control_site_pairs",
                claim="Kish effective control site pairs carried by the contrast's control side",
                family="structure_contact", value=float(block["effective_control_site_pairs"]),
                unit="effective site pairs", kind="effective_count", support_id=support,
                weighting_convention="the weighting the contrast applies: site pairs equal inside the "
                                     "contact arm, controls weighted onto its cells",
                level="measurement",
                source_path=path, source_sha256=ledger.artifacts.sha256(path),
                source_pointer=pointer + ("effective_control_site_pairs",),
            ))


# --------------------------------------------------------------------------- #
# Family 6 --- related homologous context
# --------------------------------------------------------------------------- #

CONTEXT_SOURCES = [
    ("results/R5/context_homologue_20260826/context_homologue.json", set()),
    ("results/R5/context_homologue_expansion_20260919/context_homologue.json",
     {"progen3-112m", "progen3-3b"}),
    ("results/R5/context_homologue_progen3_20260924/context_homologue.json", set()),
]
CONTEXT_ENDPOINTS = {
    "auroc": ("paired_concordance", "within-target paired win probability", "dimensionless probability"),
    "fractional_reduction": ("fractional_reduction",
                             "unrelated-minus-related loss divided by position-only loss within each target",
                             "dimensionless fraction"),
}


def family_context(ledger: Ledger) -> None:
    joint: dict[str, dict[str, bool]] = {}
    for path, withdrawn in CONTEXT_SOURCES:
        payload = ledger.artifacts.json(path)
        for arm, block in payload["arms"].items():
            if arm in withdrawn or block.get("modality") != "protein" or block.get("status") != "scored":
                continue
            for band, block in (block_bands := (block.get("bands") or {})).items():
                if not isinstance(block_bands, dict) or not isinstance(block, dict):
                    continue
                for key, (name, claim, unit) in CONTEXT_ENDPOINTS.items():
                    value = block.get(key)
                    if not isinstance(value, dict) or value.get("degenerate"):
                        continue
                    band_support = ledger.declare_support(
                        f"context_band_{band}",
                        f"the {band} relatedness band of the related-context cohort",
                    )
                    ledger.add(Quantity(
                        id=f"context/{arm}/band_{band}/{name}",
                        claim=f"{arm} in the {band} relatedness band: {claim}", family="context",
                        value=_point(value), unit=unit, kind="estimate", support_id=band_support,
                        interval=_interval(value),
                        interval_kind=PERCENTILE_95 if _interval(value) else None,
                        resampling_unit="target near-duplicate group" if _interval(value) else None,
                        resampling_draws=int(value["resamples"]) if _interval(value) else None,
                        no_interval_reason=None if _interval(value) else
                        "the artifact retains no interval for this band",
                        level="likelihood",
                        source_path=path, source_sha256=ledger.artifacts.sha256(path),
                        source_pointer=("arms", arm, "bands", band, key),
                    ))
            block = payload["arms"][arm]
            for stratum in ("pooled", "decisive_stratum"):
                cell = block[stratum]
                for key, (name, claim, unit) in CONTEXT_ENDPOINTS.items():
                    value = cell[key]
                    if value.get("degenerate"):
                        ledger.gap(id=f"context/{arm}/{stratum}/{name}", claim=claim, family="context",
                                   reason="degenerate", looked_in=path)
                        continue
                    support = ledger.declare_support(
                        f"context_{stratum}_{int(value['n_rows'])}t_{int(value['n_groups'])}g",
                        ("pooled related-context cohort" if stratum == "pooled" else
                         "lowest-overlap stratum, the bottom local-overlap tercile of the below-30%-"
                         "identity band") +
                        f": {int(value['n_rows'])} targets in {int(value['n_groups'])} target "
                        "near-duplicate groups",
                    )
                    for field, count_unit, count_claim in (
                        ("n_rows", "targets", "targets in this cohort"),
                        ("n_groups", "groups", "target near-duplicate groups in this cohort"),
                    ):
                        identifier = f"context/{support}/{field}"
                        if identifier not in {q.id for q in ledger.quantities}:
                            ledger.add(Quantity(
                                id=identifier, claim=count_claim, family="context",
                                value=float(value[field]), unit=count_unit, kind="support_count",
                                support_id=support, level="likelihood",
                                source_path=path, source_sha256=ledger.artifacts.sha256(path),
                                source_pointer=("arms", arm, stratum, key, field),
                            ))
                    ledger.add(Quantity(
                        id=f"context/{arm}/{stratum}/{name}",
                        claim=f"{arm} in the {stratum} cohort: {claim}", family="context",
                        value=_point(value), unit=unit, kind="estimate", support_id=support,
                        interval=_interval(value), interval_kind=PERCENTILE_95,
                        resampling_unit="target near-duplicate group",
                        resampling_draws=int(value["resamples"]), level="likelihood",
                        source_path=path, source_sha256=ledger.artifacts.sha256(path),
                        source_pointer=("arms", arm, stratum, key),
                    ))
                    threshold = 0.5 if key == "auroc" else 0.0
                    joint.setdefault(arm, {})[f"{stratum}/{name}"] = _interval(value)[0] > threshold
    if not joint:
        return
    met = [arm for arm, flags in joint.items() if len(flags) == 4 and all(flags.values())]
    pooled_only = [arm for arm, flags in joint.items()
                   if arm not in met and flags.get("pooled/paired_concordance")
                   and flags.get("pooled/fractional_reduction")]
    path = CONTEXT_SOURCES[0][0]
    support = ledger.declare_support("context_panel",
                                     "the protein-modality checkpoints scored on the related-context cohort")
    for name, value, claim, verdict in (
        ("checkpoints_scored", len(joint), "checkpoints scored on the related-context cohort", None),
        ("checkpoints_meeting_joint_criterion", len(met),
         "checkpoints meeting the joint criterion in both the pooled cohort and the low-overlap stratum",
         "supported"),
        ("checkpoints_meeting_pooled_only", len(pooled_only),
         "checkpoints meeting the pooled criterion only", None),
    ):
        ledger.add(Quantity(
            id=f"context/panel/{name}", claim=claim, family="context", value=float(value),
            unit="checkpoints", kind="count" if verdict else "support_count", support_id=support,
            level="likelihood", verdict=verdict,
            source_path=path, source_sha256=ledger.artifacts.sha256(path),
            source_pointer=("arms", "<reduction>"),
        ))


# --------------------------------------------------------------------------- #
# Family 7 --- generation: the attempt census and the matched generators
# --------------------------------------------------------------------------- #

CENSUS_SOURCES = [
    "results/R6/unconditional_diversity_20260919/unconditional_diversity.json",
    "results/R6/unconditional_diversity_3b_20260919/unconditional_diversity.json",
]
CENSUS_FIELDS = {
    "n_attempts": ("native unconditional attempts", "attempts"),
    "n_empty": ("empty outputs", "attempts"),
    "n_nonempty": ("nonempty outputs", "attempts"),
    "n_unique_nonempty_hashes": ("exactly unique nonempty sequences", "sequences"),
    "n_any_profile_hits": ("attempts carrying any Pfam profile", "attempts"),
    "n_any_profile_groups": ("recognized near-duplicate groups", "groups"),
}


def family_generation_census(ledger: Ledger) -> None:
    support = ledger.declare_support("uncond_census_800", CENSUS_SUPPORT)
    census: dict[str, tuple[dict, str]] = {}
    for path in CENSUS_SOURCES:
        payload = ledger.artifacts.json(path)
        for arm, block in payload["checkpoints"].items():
            census[arm] = (block, path)
    for arm, (block, path) in census.items():
        for field, (claim, unit) in CENSUS_FIELDS.items():
            if block.get(field) is None:
                ledger.gap(id=f"generation_census/{arm}/{field}", claim=claim,
                           family="generation_census",
                           reason="the artifact carries no value, which is not a zero",
                           looked_in=path, reachability="not_measured")
                continue
            ledger.add(Quantity(
                id=f"generation_census/{arm}/{field}", claim=f"{arm}: {claim}",
                family="generation_census", value=float(block[field]), unit=unit, kind="census",
                support_id=support,
                no_interval_reason="a complete enumeration of the saved batches, so no resampling unit "
                                   "or interval applies",
                level="generation",
                source_path=path, source_sha256=ledger.artifacts.sha256(path),
                source_pointer=("checkpoints", arm, field),
            ))


def family_generative_control(ledger: Ledger) -> None:
    path = "results/R6/generative_control_20260924/gate_endpoints.json"
    payload = ledger.artifacts.json(path)
    resamples = int(payload["resamples"])
    support = ledger.declare_support("generator_gate_cells",
                                     generator_gate_support(len(payload["cells"])))
    for index, cell in enumerate(payload["cells"]):
        for endpoint, block in (cell.get("endpoints") or {}).items():
            for cohort, contrast in (block.get("controls") or {}).items():
                if not isinstance(contrast, dict):
                    continue
                interval = _interval(contrast)
                if interval is None:
                    ledger.gap(id=f"generative_control/{cell['cell']}/{endpoint}/{cohort}",
                               claim=f"{cell['cell']}: {endpoint} rate minus the {cohort} rate",
                               family="generative_control", reason="no interval in the artifact",
                               looked_in=path)
                    continue
                ledger.add(Quantity(
                    id=f"generative_control/{cell['cell']}/{endpoint}/{cohort}",
                    claim=f"{cell['cell']}: {endpoint} rate minus the {cohort} generator's rate",
                    family="generative_control", value=_point(contrast), unit="dimensionless rate difference",
                    kind="estimate", support_id=support, interval=interval,
                    interval_kind=PERCENTILE_95, resampling_unit="attempt near-duplicate sequence group",
                    resampling_draws=resamples, seed_set=(int(payload["seed"]),), level="generation",
                    source_path=path, source_sha256=ledger.artifacts.sha256(path),
                    source_pointer=("cells", f"[{index}]", "endpoints", endpoint, "controls", cohort,
                                    "difference"),
                    verdict="supported" if interval[0] > 0.0 else
                            ("not_detected" if interval[1] < 0.0 else "unresolved"),
                ))
                if contrast.get("ci97_5"):
                    ledger.add(Quantity(
                        id=f"generative_control/{cell['cell']}/{endpoint}/{cohort}/ci97_5",
                        claim=f"{cell['cell']}: {endpoint} rate minus the {cohort} generator's rate, "
                              "at the wider interval the gate also records",
                        family="generative_control", value=_point(contrast),
                        unit="dimensionless rate difference", kind="estimate", support_id=support,
                        interval=tuple(map(float, contrast["ci97_5"])),
                        interval_kind="97.5% percentile",
                        resampling_unit="attempt near-duplicate sequence group",
                        resampling_draws=resamples, seed_set=(int(payload["seed"]),),
                        level="generation",
                        source_path=path, source_sha256=ledger.artifacts.sha256(path),
                        source_pointer=("cells", f"[{index}]", "endpoints", endpoint, "controls",
                                        cohort, "ci97_5"),
                    ))
                for field, unit in (("model_rate", "dimensionless rate"),
                                    ("control_rate", "dimensionless rate"),
                                    ("control_successes", "attempts")):
                    if contrast.get(field) is None:
                        continue
                    key = (f"generative_control/{cell['cell']}/{endpoint}/{cohort}/{field}"
                           if field != "model_rate"
                           else f"generative_control/{cell['cell']}/{endpoint}/model_rate")
                    if key in {q.id for q in ledger.quantities}:
                        continue
                    ledger.add(Quantity(
                        id=key,
                        claim=f"{cell['cell']}: {field.replace('_', ' ')} on the {endpoint} endpoint"
                              + ("" if field == "model_rate" else f" for the {cohort} generator"),
                        family="generative_control", value=float(contrast[field]), unit=unit,
                        kind="census", support_id=support,
                        no_interval_reason="a complete enumeration of the cell's 800 attempts",
                        seed_set=(int(payload["seed"]),), level="generation",
                        source_path=path, source_sha256=ledger.artifacts.sha256(path),
                        source_pointer=("cells", f"[{index}]", "endpoints", endpoint, "controls",
                                        cohort, field),
                    ))
    for index, cell in enumerate(payload["cells"]):
        for band, block in (cell.get("identity_strata") or {}).items():
            if not isinstance(block, dict):
                continue
            for endpoint, inner in block.items():
                if not isinstance(inner, dict):
                    continue
                controls = inner.get("controls") or {}
                if not controls:
                    continue
                sample = next(iter(controls.values()))
                if sample.get("model_rate") is None:
                    continue
                identifier = f"generative_control/{cell['cell']}/{band}/{endpoint}/model_rate"
                ledger.add(Quantity(
                    id=identifier,
                    claim=f"{cell['cell']}: {endpoint} recognition rate in the {band} identity band "
                          "against the searched snapshot",
                    family="generative_control", value=float(sample["model_rate"]),
                    unit="dimensionless rate", kind="census", support_id=support,
                    no_interval_reason="a complete enumeration of the band's attempts",
                    seed_set=(int(payload["seed"]),), level="generation",
                    source_path=path, source_sha256=ledger.artifacts.sha256(path),
                    source_pointer=("cells", f"[{index}]", "identity_strata", band, endpoint,
                                    "controls", "<any>", "model_rate"),
                ))

    ceiling = payload["profile_sampler_ceiling"]
    ceiling_support = ledger.declare_support(
        "profile_sampler_ceiling",
        f"one sequence emitted from each of {ceiling['n_profiles_requested']} Pfam-A profiles and read "
        "back through the same oracle call",
    )
    for field, wilson, claim in (
        ("any_family_rate", "any_family_wilson95", "the profile sampler's own any-family recognition rate"),
        ("complete_domain_rate", "complete_domain_wilson95",
         "the profile sampler's own complete-domain recognition rate"),
    ):
        ledger.add(Quantity(
            id=f"generative_control/ceiling/{field}", claim=claim, family="generative_control",
            value=float(ceiling[field]), unit="dimensionless rate", kind="estimate",
            support_id=ceiling_support,
            interval=tuple(map(float, ceiling[wilson])) if ceiling.get(wilson) else None,
            interval_kind="95% Wilson" if ceiling.get(wilson) else None,
            resampling_unit="emitted sequence" if ceiling.get(wilson) else None,
            resampling_draws=int(ceiling["n_sequences"]) if ceiling.get(wilson) else None,
            no_interval_reason=None if ceiling.get(wilson) else "the artifact retains no interval",
            level="generation",
            source_path=path, source_sha256=ledger.artifacts.sha256(path),
            source_pointer=("profile_sampler_ceiling", field),
        ))


def family_conditional_generation(ledger: Ledger) -> None:
    path = ("results/R6/generation_evidence_20260905/analysis/20260904232436_00129607e3c7/main/"
            "generation_biology_analysis.json")
    payload = ledger.artifacts.json(path)
    support = ledger.declare_support(
        "conditional_class_cells",
        "conditional generation cells at 200 attempts each, requested and mismatched conditions over "
        "the eligible enzyme-commission and superfamily classes",
    )
    rows = 0
    for entry in payload.get("profile_ledger", []):
        if entry.get("role") != "generation" or entry.get("condition") not in ("requested", "mismatched"):
            continue
        rows += 1
        pointer = ("profile_ledger", f"<arm={entry['arm']},class={entry['class_key']},"
                                     f"condition={entry['condition']}>", "target_profile_rate")
        ledger.add(Quantity(
            id=f"conditional/{entry['arm']}/{entry['class_key']}/{entry['condition']}/target_profile_rate",
            claim=f"{entry['arm']} under the {entry['condition']} condition for {entry['class_key']}: "
                  "share of attempts carrying a target profile of that class",
            family="conditional_generation", value=float(entry["target_profile_rate"]),
            unit="dimensionless rate", kind="census", support_id=support,
            no_interval_reason="a complete enumeration of the cell's 200 attempts",
            level="generation",
            source_path=path, source_sha256=ledger.artifacts.sha256(path), source_pointer=pointer,
        ))
    if not rows:
        ledger.gap(id="conditional/panel", claim="conditional target-profile assignment",
                   family="conditional_generation", reason="no generation rows in the profile ledger",
                   looked_in=path)
    ledger.gap(
        id="conditional/panel/requested_minus_mismatched",
        claim="requested-minus-mismatched target-profile assignment aggregated over the eligible "
              "classes, with its interval over classes",
        family="conditional_generation",
        reason="the retained analysis artifact carries per-cell rates and the pairing's sufficient "
               "statistics, and no run computes the class-resampled paired difference the manuscript "
               "quotes; the cluster store holds the same schema",
        looked_in=path, reachability="not_measured",
    )


# --------------------------------------------------------------------------- #
# Family 8 --- cross-measure association and the structural instrument
# --------------------------------------------------------------------------- #

def family_cross_measure(ledger: Ledger) -> None:
    path = "results/shared/cross_measure_20260918/cross_measure_association.json"
    payload = ledger.artifacts.json(path)
    for index, pair in enumerate(payload.get("pooled", [])):
        if pair.get("spearman") is None:
            ledger.gap(id=f"cross_measure/{pair['pair_id']}",
                       claim=f"cross-checkpoint Spearman between {pair['x']} and {pair['y']}",
                       family="cross_measure",
                       reason=pair.get("reason") or "the artifact reports no coefficient for this pair",
                       looked_in=path, reachability="not_measured")
            continue
        interval = pair.get("interval") or {}
        low, high = interval.get("lo"), interval.get("hi")
        support = ledger.declare_support(
            f"cross_measure_n{pair['n']}",
            f"{pair['n']} checkpoints holding both records of the pair, pooled across lineages",
        )
        ledger.add(Quantity(
            id=f"cross_measure/{pair['pair_id']}",
            claim=f"cross-checkpoint Spearman between {pair['x']} and {pair['y']}",
            family="cross_measure", value=float(pair["spearman"]), unit="dimensionless Spearman",
            kind="estimate", support_id=support,
            interval=(float(low), float(high)) if low is not None and high is not None else None,
            interval_kind=PERCENTILE_95 if low is not None else None,
            resampling_unit="checkpoint" if low is not None else None,
            resampling_draws=CROSS_MEASURE_DRAWS if low is not None else None,
            no_interval_reason=None if low is not None else (pair.get("reason") or "no interval reported"),
            seed_set=(CROSS_MEASURE_SEED,),
            level="measurement",
            source_path=path, source_sha256=ledger.artifacts.sha256(path),
            source_pointer=("pooled", f"[{index}]", "spearman"),
        ))


def family_structure_instrument(ledger: Ledger) -> None:
    path = "results/R6/structure_instrument_20260923/structure_instrument_reconciliation.json"
    if not ledger.artifacts.exists(path):
        ledger.gap(id="structure_instrument/calibration",
                   claim="checkpoints on which the folding predictor's calibration is attained",
                   family="structure_instrument", reason="artifact absent", looked_in=path)
        return
    payload = ledger.artifacts.json(path)
    support = ledger.declare_support(
        "structure_instrument_panel",
        "the checkpoints reporting a paired within-sequence structural contrast",
    )
    for pointer, claim, unit in _instrument_counts(payload):
        ledger.add(Quantity(
            id="structure_instrument/" + "_".join(pointer), claim=claim, family="structure_instrument",
            value=float(resolve(payload, pointer)), unit=unit, kind="support_count", support_id=support,
            level="measurement",
            source_path=path, source_sha256=ledger.artifacts.sha256(path), source_pointer=pointer,
        ))


def _instrument_counts(payload: object) -> list[tuple[tuple[str, ...], str, str]]:
    wanted = ("n_own_controls", "n_inherited", "n_cells", "calibration_attained", "n_arms")
    found = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in wanted and isinstance(value, (int, float)) and not isinstance(value, bool):
                found.append(((key,), f"structural instrument count: {key.replace('_', ' ')}",
                              "checkpoints"))
    return found


# --------------------------------------------------------------------------- #
# Family 9 --- the qualified local-context control
# --------------------------------------------------------------------------- #

LOCAL_CONTEXT_RECORD = "archive/docs/R1/gate-local-context.md"
LOCAL_CONTEXT_QUALIFICATION = ("results/R5/local_context_20260923/20260923221324_c1bb5dc48e2e/"
                               "lcg_qualify_gate/local_context_qualification.json")
LOCAL_CONTEXT_MEASUREMENT = ("results/R5/local_context_20260923/20260923233257_e429ce7f31e4/"
                             "lcgp_admission/local_context_measurement.json")
LOCAL_CONTEXT_PRIMARY = ("results/R5/local_context_20260923/20260923223301_305478e124be/"
                         "lcgm_admission/local_context_measurement.json")

#: The two cell-level increments the gate reports, and the level each sits at.
#: ``M`` is the checkpoint's native likelihood entering as one scalar column;
#: ``R`` is its 1,024-coordinate representation block, which is provisional.
LOCAL_CONTEXT_INCREMENTS = {
    "increment_M_C_P_wall": ("the native likelihood's increment over the qualified local control",
                             "likelihood"),
    "increment_R_C_P_wall": ("the representation residual over the qualified local control",
                             "representation"),
    "increment_M_C": ("the native likelihood's increment over the composition control", "likelihood"),
    "increment_M_C_P": ("the native likelihood's increment over composition plus profile", "likelihood"),
    "increment_R_C": ("the representation increment over the composition control", "representation"),
    "increment_R_C_P": ("the representation increment over composition plus profile", "representation"),
    "increment_R_C_wall": ("the representation increment over composition plus the local control",
                           "representation"),
}


def family_local_context(ledger: Ledger) -> None:
    """The qualified local-context gate, read from the receipts it declares.

    Both receipts were retrieved from the cluster store and hash to the digests
    the gate record declares for them, so the numbers here are artifact-backed
    rather than transcribed from the record's prose.
    """
    if not ledger.artifacts.exists(LOCAL_CONTEXT_MEASUREMENT):
        ledger.gap(
            id="local_context/qualification_receipt",
            claim="every per-cell quantity of the qualified local-context control: the 99 measurement "
                  "cells' likelihood and representation increments with their intervals, the control's "
                  "own contributions, and the absorbed shares",
            family="local_context",
            reason="the qualification receipt and the 99 measurement cells are retained in the "
                   "cluster store and are not staged on this host",
            looked_in=LOCAL_CONTEXT_MEASUREMENT, reachability="cluster_only",
        )
        return

    measurement = ledger.artifacts.json(LOCAL_CONTEXT_MEASUREMENT)
    support = ledger.declare_support(
        "local_context_gate_cohort",
        "the frozen readout anchor's 201 assays in 163 wild-type clusters, refitted with the qualified "
        "111-column mutation-centred local control at three split seeds",
    )
    resolved: dict[str, dict[str, list[dict]]] = {name: {} for name in LOCAL_CONTEXT_INCREMENTS}
    for cell, summaries in measurement["summaries"].items():
        arm, seed = cell.split("/")
        for key, (claim, level) in LOCAL_CONTEXT_INCREMENTS.items():
            value = summaries.get(key)
            if value is None or value.get("degenerate"):
                ledger.gap(id=f"local_context/{arm}/{seed}/{key}", claim=claim,
                           family="local_context",
                           reason="the cell is absent or degenerate in the admission receipt",
                           looked_in=LOCAL_CONTEXT_MEASUREMENT)
                continue
            ledger.add(Quantity(
                id=f"local_context/{arm}/{seed}/{key}",
                claim=f"{arm} at split seed {seed}: {claim}", family="local_context",
                value=_point(value), unit="dimensionless Spearman", kind="estimate",
                support_id=support, interval=_interval(value), interval_kind=PERCENTILE_95,
                resampling_unit=value["unit"], resampling_draws=int(value["resamples"]),
                seed_set=(int(seed),), level=level,
                source_path=LOCAL_CONTEXT_MEASUREMENT,
                source_sha256=ledger.artifacts.sha256(LOCAL_CONTEXT_MEASUREMENT),
                source_pointer=("summaries", cell, key),
            ))
            resolved[key].setdefault(arm, []).append(
                {"value": _point(value), "excludes_zero": bool(value["excludes_zero"])})

    for key, (claim, level) in LOCAL_CONTEXT_INCREMENTS.items():
        arms = resolved[key]
        if not arms:
            continue
        above = [arm for arm, cells in arms.items()
                 if len(cells) == len(SPLIT_SEEDS)
                 and all(cell["excludes_zero"] and cell["value"] > 0 for cell in cells)]
        ledger.add(Quantity(
            id=f"local_context/panel/{key}/checkpoints_above_zero_every_seed",
            claim=f"checkpoints for which {claim} resolves above zero at all three split seeds",
            family="local_context", value=float(len(above)), unit="checkpoints", kind="count",
            support_id=support, seed_set=SPLIT_SEEDS, level=level,
            verdict="supported" if above else "not_detected",
            source_path=LOCAL_CONTEXT_MEASUREMENT,
            source_sha256=ledger.artifacts.sha256(LOCAL_CONTEXT_MEASUREMENT),
            source_pointer=("summaries", "<reduction>", key, "excludes_zero"),
        ))
        ledger.add(Quantity(
            id=f"local_context/panel/{key}/checkpoints_measured",
            claim=f"checkpoints carrying {claim}", family="local_context",
            value=float(len(arms)), unit="checkpoints", kind="support_count", support_id=support,
            seed_set=SPLIT_SEEDS, level=level,
            source_path=LOCAL_CONTEXT_MEASUREMENT,
            source_sha256=ledger.artifacts.sha256(LOCAL_CONTEXT_MEASUREMENT),
            source_pointer=("summaries", "<reduction>", key),
        ))
        if above:
            rows = [{"value": cell["value"]} for arm in above for cell in arms[arm]]
            _range_bounds(ledger, f"local_context/panel/{key}/resolved",
                          f"range of {claim} over the checkpoints resolving at all three seeds",
                          "local_context", rows, support, level, LOCAL_CONTEXT_MEASUREMENT,
                          ledger.artifacts.sha256(LOCAL_CONTEXT_MEASUREMENT))
            # The nine-checkpoint primary panel is a different set from the 33
            # the generalizability check covers, and a range over one read
            # beside a count over the other is a support the reader cannot see.
            if ledger.artifacts.exists(LOCAL_CONTEXT_PRIMARY):
                primary = {cell.split("/")[0]
                           for cell in ledger.artifacts.json(LOCAL_CONTEXT_PRIMARY)["cells"]}
                inside = [{"value": cell["value"]} for arm in above if arm in primary
                          for cell in arms[arm]]
                if inside and len(inside) < len(rows):
                    primary_support = ledger.declare_support(
                        "local_context_primary_panel",
                        "the gate's primary panel of stratum-covering checkpoints at three split "
                        f"seeds, {len(primary)} checkpoints of the cohort, on which the control was "
                        "qualified before the 33-checkpoint generalizability check",
                    )
                    _range_bounds(ledger, f"local_context/primary_panel/{key}/resolved",
                                  f"range of {claim} over the primary panel's checkpoints resolving "
                                  "at all three seeds",
                                  "local_context", inside, primary_support, level,
                                  LOCAL_CONTEXT_PRIMARY,
                                  ledger.artifacts.sha256(LOCAL_CONTEXT_PRIMARY))

    for name, path in (("primary_panel", LOCAL_CONTEXT_PRIMARY),
                       ("full_panel", LOCAL_CONTEXT_MEASUREMENT)):
        if not ledger.artifacts.exists(path):
            continue
        ledger.add(Quantity(
            id=f"local_context/{name}/cells",
            claim=f"measurement cells admitted in the gate's {name.replace('_', ' ')}",
            family="local_context",
            value=float(len(ledger.artifacts.json(path)["cells"])), unit="checkpoint--seed cells",
            kind="support_count", support_id=support, seed_set=SPLIT_SEEDS, level="control",
            source_path=path, source_sha256=ledger.artifacts.sha256(path),
            source_pointer=("cells",),
        ))

    _local_context_absorbed_shares(ledger, measurement, support)
    _local_context_qualification(ledger, support)


def _local_context_absorbed_shares(ledger: Ledger, measurement: dict, support: str) -> None:
    """How much of each representation increment the local control absorbs.

    The manuscript quotes these as percentages, so they are computed from the
    two increments rather than restated: a change in either moves the share.
    """
    for cell, summaries in measurement["summaries"].items():
        arm, seed = cell.split("/")
        for name, over, under, claim in (
            ("composition", "increment_R_C", "increment_R_C_wall",
             "share of the composition-relative representation increment the local control absorbs"),
            ("composition_profile", "increment_R_C_P", "increment_R_C_P_wall",
             "share of the profile-relative representation increment the local control absorbs"),
        ):
            before, after = summaries.get(over), summaries.get(under)
            if not before or not after or not _point(before):
                continue
            ledger.add(Quantity(
                id=f"local_context/{arm}/{seed}/absorbed_share_{name}",
                claim=f"{arm} at split seed {seed}: {claim}", family="local_context",
                value=100.0 * (_point(before) - _point(after)) / _point(before),
                unit="percent", kind="ratio", support_id=support, seed_set=(int(seed),),
                level="representation",
                source_path=LOCAL_CONTEXT_MEASUREMENT,
                source_sha256=ledger.artifacts.sha256(LOCAL_CONTEXT_MEASUREMENT),
                source_pointer=("<ratio>", "summaries", cell, over, under),
            ))


def _local_context_qualification(ledger: Ledger, support: str) -> None:
    """The gate's own control-versus-control fits, which carry no model quantity."""
    if not ledger.artifacts.exists(LOCAL_CONTEXT_QUALIFICATION):
        ledger.gap(id="local_context/qualification",
                   claim="the gate's control-versus-control qualification fits",
                   family="local_context",
                   reason="the receipt is retained in the cluster store and is not staged on this host",
                   looked_in=LOCAL_CONTEXT_QUALIFICATION, reachability="cluster_only")
        return
    payload = ledger.artifacts.json(LOCAL_CONTEXT_QUALIFICATION)
    for seed, summaries in payload["summaries"].items():
        for key, value in sorted(summaries.items()):
            if not key.endswith("_spearman") or not isinstance(value, dict) or value.get("degenerate"):
                continue
            ledger.add(Quantity(
                id=f"local_context/qualification/{seed}/{key}",
                claim=f"held-cluster Spearman of the {key[:-9].replace('_', '+')} design at split "
                      f"seed {seed}",
                family="local_context", value=_point(value), unit="dimensionless Spearman",
                kind="estimate", support_id=support, interval=_interval(value),
                interval_kind=PERCENTILE_95, resampling_unit=value["unit"],
                resampling_draws=int(value["resamples"]), seed_set=(int(seed),), level="control",
                source_path=LOCAL_CONTEXT_QUALIFICATION,
                source_sha256=ledger.artifacts.sha256(LOCAL_CONTEXT_QUALIFICATION),
                source_pointer=("summaries", seed, key),
            ))
    gate = payload["gate"]
    for candidate, verdict in sorted(gate["verdicts"].items()):
        for seed, block in sorted((verdict.get("per_seed") or {}).items()):
            for baseline, label in (("C", "the composition control"),
                                    ("C_P", "composition plus profile"),
                                    ("standalone", "held-out clusters on its own")):
                value = block.get(baseline)
                if not isinstance(value, dict) or value.get("point") is None:
                    continue
                ledger.add(Quantity(
                    id=f"local_context/qualification/{seed}/{candidate}/{baseline}",
                    claim=f"the {candidate} candidate block's own contribution over {label} at split "
                          f"seed {seed}",
                    family="local_context", value=float(value["point"]),
                    unit="dimensionless Spearman", kind="estimate", support_id=support,
                    interval=tuple(map(float, value["interval"])) if value.get("interval") else None,
                    interval_kind=PERCENTILE_95 if value.get("interval") else None,
                    resampling_unit="wild-type family at 50% identity" if value.get("interval") else None,
                    resampling_draws=(_draws(value, payload)
                                      or _artifact_draws(ledger, LOCAL_CONTEXT_QUALIFICATION))
                                     if value.get("interval") else None,
                    no_interval_reason=None if value.get("interval") else
                    "the receipt records this contribution without an interval",
                    seed_set=(int(seed),), level="control",
                    source_path=LOCAL_CONTEXT_QUALIFICATION,
                    source_sha256=ledger.artifacts.sha256(LOCAL_CONTEXT_QUALIFICATION),
                    source_pointer=("gate", "verdicts", candidate, "per_seed", seed, baseline),
                ))
    # What a bounded window recovers of the composition control, which the
    # manuscript quotes as a percentage and the receipt records as two fitted
    # correlations per seed.
    for seed, summaries in payload["summaries"].items():
        for candidate in ("rf3", "rf7", "wall"):
            window, composition = summaries.get(f"{candidate}_spearman"), summaries.get("C_spearman")
            if not window or not composition or not _point(composition):
                continue
            ledger.add(Quantity(
                id=f"local_context/qualification/{seed}/{candidate}/recovery_of_composition",
                claim=f"share of the composition control's held-cluster ranking the {candidate} "
                      f"window predictor reaches on its own at split seed {seed}",
                family="local_context", value=100.0 * _point(window) / _point(composition),
                unit="percent", kind="ratio", support_id=support, seed_set=(int(seed),),
                level="control",
                source_path=LOCAL_CONTEXT_QUALIFICATION,
                source_sha256=ledger.artifacts.sha256(LOCAL_CONTEXT_QUALIFICATION),
                source_pointer=("<ratio>", "summaries", seed, f"{candidate}_spearman", "C_spearman"),
            ))
    for name, field, claim in (
        ("candidates", "candidates", "declared candidate local controls"),
        ("qualified", "qualified", "candidate local controls that qualified"),
        ("discarded", "discarded", "candidate local controls discarded by the refusal rule"),
        ("carried_forward", "carried_forward", "candidate local controls carried into the comparison"),
    ):
        ledger.add(Quantity(
            id=f"local_context/gate/{name}", claim=claim, family="local_context",
            value=float(len(gate[field])), unit="candidate controls",
            kind="count" if name != "candidates" else "support_count", support_id=support,
            seed_set=SPLIT_SEEDS, level="control",
            source_path=LOCAL_CONTEXT_QUALIFICATION,
            source_sha256=ledger.artifacts.sha256(LOCAL_CONTEXT_QUALIFICATION),
            source_pointer=("gate", field),
        ))



# --------------------------------------------------------------------------- #
# The figure-data contract
# --------------------------------------------------------------------------- #

def table_bindings(artifacts: Artifacts, rows: list[dict]) -> dict[str, dict[str, object]]:
    """Bind a figure input table's rows to the artifacts they were read from.

    Each row names the artifact path and digest its value was extracted from.
    A digest that no longer matches the file on disk, a file that is gone, or a
    table that records two digests for one artifact each break the binding
    rather than being read past.
    """
    bindings: dict[str, dict[str, object]] = {}
    for row in rows:
        source = (row.get("source_path") or "").strip()
        if not source:
            continue
        recorded = (row.get("source_sha256") or "").strip()
        entry = bindings.setdefault(source, {"recorded_sha256": recorded, "rows": 0,
                                             "status": "unchecked"})
        entry["rows"] = int(entry["rows"]) + 1
        if entry["recorded_sha256"] != recorded:
            entry["status"] = "csv_disagrees_with_itself"
    for source, entry in bindings.items():
        if entry["status"] != "unchecked":
            continue
        if not artifacts.exists(source):
            entry["status"] = "artifact_absent"
        elif artifacts.sha256(source) != entry["recorded_sha256"]:
            entry["status"] = "artifact_changed"
        else:
            entry["status"] = "bound"
    return bindings


def figure_data_contract(artifacts: Artifacts, table: dict[str, object]) -> dict[str, object]:
    """Bind each manuscript figure to its input tables and their artifacts.

    A figure's input CSV already carries the artifact path and digest of every
    row it was extracted from. This reads those columns back, checks each
    artifact still hashes to the recorded digest, and records the binding, so a
    figure cannot drift from the evidence without the contract failing. The
    renderer is untouched.
    """
    main = (REPO_ROOT / "manuscript/main.tex").read_text()
    supplement = (REPO_ROOT / "manuscript/supplementary-information.tex").read_text()
    included = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", main + supplement)

    tables: dict[str, dict[str, object]] = {}
    for csv_path in sorted(FIGURE_DIR.rglob("*.csv")):
        relative = csv_path.relative_to(REPO_ROOT).as_posix()
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        bindings = table_bindings(artifacts, rows)
        tables[relative] = {
            "sha256": artifacts.sha256(relative),
            "rows": len(rows),
            "artifact_bindings": dict(sorted(bindings.items())),
            "unbound_rows": sum(1 for row in rows if not (row.get("source_path") or "").strip()),
        }

    figures: dict[str, object] = {}
    table_root = FIGURE_DIR.relative_to(REPO_ROOT).as_posix()
    for figure in sorted(set(included)):
        stem = Path(figure).with_suffix("")
        # A figure is included from manuscript/figures/ or manuscript/figures/ncs/
        # while its input tables sit under figure_data/main/ or figure_data/ncs/.
        # The directory the figure is written to therefore selects the table
        # directory; it does not hold the tables itself. A figure whose stem
        # matches no table keeps an empty input list and stays unbound, because
        # claiming every table in the directory would bind it to evidence it was
        # not drawn from.
        leaf = stem.parent.name
        directory = f"{table_root}/{'main' if leaf == 'figures' else leaf}"
        inputs = sorted(name for name in tables
                        if Path(name).parent.as_posix() == directory
                        and Path(name).name.startswith(stem.name.split(".")[0]))
        figures[figure] = {
            "input_tables": inputs,
            "bound": bool(inputs) and all(
                entry["status"] == "bound"
                for name in inputs
                for entry in tables[name]["artifact_bindings"].values()
            ),
        }
    unbound = sorted(figure for figure, entry in figures.items() if not entry["bound"])
    claimed = {name for entry in figures.values() for name in entry["input_tables"]}
    return {
        "schema": "manuscript_figure_data_contract_v1",
        "rule": "each included figure is bound to the input tables beside it, and each table row to the "
                "artifact path and digest it was extracted from; a changed or absent artifact breaks the "
                "binding rather than silently redrawing the figure",
        "derived_number_table_sha256": None,
        "quantities_in_table": table["counts"]["quantities"],
        "figures": figures,
        "tables": tables,
        "figures_without_a_complete_binding": unbound,
        "tables_claimed_by_no_figure": sorted(set(tables) - claimed),
    }


# --------------------------------------------------------------------------- #
# Family 10 --- the panel roster, the support counts and the shared conventions
# --------------------------------------------------------------------------- #

def family_roster_and_conventions(ledger: Ledger, strata: dict[str, str]) -> None:
    admission = f"{READOUT_ROOT}/final_admission.json"
    support = ledger.declare_support(
        "extraction_panel",
        "the frozen extraction panel of released checkpoints, in interface strata that are never pooled",
    )
    counts = {"panel": len(strata)}
    for arm, stratum in strata.items():
        counts[stratum] = counts.get(stratum, 0) + 1
    counts["unconditioned"] = len(strata) - counts.get("EC_conditioned", 0)
    for name, value in sorted(counts.items()):
        ledger.add(Quantity(
            id=f"roster/{name}",
            claim=f"checkpoints in the {name.replace('_', ' ')} stratum of the extraction panel"
                  if name not in ("panel", "unconditioned") else
                  f"checkpoints in the {name} of the extraction panel",
            family="roster", value=float(value), unit="checkpoints", kind="support_count",
            support_id=support, level="measurement",
            source_path=admission, source_sha256=ledger.artifacts.sha256(admission),
            source_pointer=("interfaces",),
        ))

    reports = [report for report in readout_reports()]
    if not reports:
        ledger.gap(id="conventions/anchor", claim="the common-support anchor's counts and conventions",
                   family="conventions", reason="no readout report found", looked_in=READOUT_ROOT)
        return
    relative = reports[0].relative_to(REPO_ROOT).as_posix()
    report = ledger.artifacts.json(relative)
    summary = report["summaries"]["R_minus_raw_M_spearman"]
    conventions = [
        ("interval_level_percent", 100.0 * (1.0 - float(summary["alpha"])), "percent",
         "the level of every percentile interval in the study",
         ("summaries", "R_minus_raw_M_spearman", "alpha")),
        ("bootstrap_draws", float(summary["resamples"]), "bootstrap draws",
         "paired bootstrap draws behind each readout and ranking interval",
         ("summaries", "R_minus_raw_M_spearman", "resamples")),
        ("anchor_assays", float(report["n_assays"]), "assays",
         "assays on the common-support anchor", ("n_assays",)),
        ("anchor_clusters", float(report["n_families"]), "wild-type clusters",
         "wild-type clusters on the common-support anchor", ("n_families",)),
        ("anchor_variants", float(report["n_variants"]), "variants",
         "variants on the common-support anchor", ("n_variants",)),
        ("anchor_variants_per_assay", float(report["n_variants"]) / float(report["n_assays"]),
         "variants per assay", "the anchor's variant draw per assay, which is what makes it a support "
         "distinct from the native ones", ("n_variants",)),
    ]
    identity = re.search(r"(\d+)% identity", str(summary.get("unit", "")))
    if identity:
        conventions.append(("family_identity_percent", float(identity.group(1)), "percent",
                            "the identity at which wild types are grouped into resampling families",
                            ("summaries", "R_minus_raw_M_spearman", "unit")))
    for name, value, unit, claim, pointer in conventions:
        ledger.add(Quantity(
            id=f"conventions/{name}", claim=claim, family="conventions", value=value, unit=unit,
            kind="constant", support_id="anchor_128", level="measurement",
            source_path=relative, source_sha256=ledger.artifacts.sha256(relative),
            source_pointer=pointer,
        ))
    for index, seed in enumerate(SPLIT_SEEDS):
        ledger.add(Quantity(
            id=f"conventions/split_seed_{seed}", claim=f"declared split seed {seed}",
            family="conventions", value=float(seed), unit="split seed", kind="constant",
            support_id="anchor_128", seed_set=(seed,), level="measurement",
            source_path=f"{EPISTASIS}/panel/panel.json",
            source_sha256=ledger.artifacts.sha256(f"{EPISTASIS}/panel/panel.json"),
            source_pointer=("recipe", "split_seeds", f"[{index}]"),
        ))

    path = "results/R6/generative_control_20260924/gate_endpoints.json"
    gate = ledger.artifacts.json(path)
    cells = gate["cells"]
    gate_support = ledger.declare_support("generator_gate_cells",
                                          generator_gate_support(len(cells)))
    searchable = sum(int(cell["n_searchable"]) for cell in cells)
    attempts = sum(int(cell["n_attempts"]) for cell in cells)
    for name, value, unit, claim, pointer in (
        ("gate_cells", float(len(cells)), "cells",
         "generation cells read against the matched generators", ("cells",)),
        ("gate_attempts", float(attempts), "attempts",
         "attempts read against the matched generators", ("cells", "<sum>", "n_attempts")),
        ("gate_near_duplicate_groups", float(sum(int(cell["n_clusters"]) for cell in cells)), "groups",
         "frozen near-duplicate sequence groups the paired differences resample over",
         ("cells", "<sum>", "n_clusters")),
        ("gate_unsearchable_attempts", float(attempts - searchable), "attempts",
         "attempts carrying nothing searchable, which stay in every denominator as measured "
         "non-recognitions", ("cells", "<sum>", "n_searchable")),
        ("gate_resamples", float(gate["resamples"]), "bootstrap draws",
         "paired resamples behind each matched-generator difference", ("resamples",)),
        ("gate_qualified_cohorts", float(len(gate["qualified_cohorts"])), "generator cohorts",
         "qualified matched generator cohorts", ("qualified_cohorts",)),
        ("gate_cohorts_under_amendment",
         float(len(gate["qualified_cohorts_under_the_scale_free_amendment"])), "generator cohorts",
         "matched generator cohorts under the scale-free amendment",
         ("qualified_cohorts_under_the_scale_free_amendment",)),
        ("complete_domain_coverage_threshold",
         float(gate["profile_sampler_ceiling"]["complete_domain_coverage_threshold"]),
         "dimensionless fraction",
         "profile-length coverage a single domain instance must reach to count as complete",
         ("profile_sampler_ceiling", "complete_domain_coverage_threshold")),
        ("profiles_emitted_for_the_ceiling",
         float(gate["profile_sampler_ceiling"]["n_profiles_requested"]), "profiles",
         "Pfam-A profiles emitted once each to measure the oracle's own ceiling",
         ("profile_sampler_ceiling", "n_profiles_requested")),
    ):
        ledger.add(Quantity(
            id=f"conventions/{name}", claim=claim, family="conventions", value=value, unit=unit,
            kind="constant" if "threshold" in name or "resamples" in name else "support_count",
            support_id=gate_support, level="generation",
            source_path=path, source_sha256=ledger.artifacts.sha256(path), source_pointer=pointer,
        ))

    census_path = CENSUS_SOURCES[0]
    census = ledger.artifacts.json(census_path)
    first = next(iter(census["checkpoints"].values()))
    for name, field, unit, claim in (
        ("census_attempts_per_checkpoint", "n_attempts", "attempts",
         "native unconditional attempts per checkpoint"),
        ("near_duplicate_shingle_length", "shingle_length", "residues",
         "shingle length of the near-duplicate relation on generated sequences"),
    ):
        ledger.add(Quantity(
            id=f"conventions/{name}", claim=claim, family="conventions", value=float(first[field]),
            unit=unit, kind="constant", support_id="uncond_census_800", level="generation",
            source_path=census_path, source_sha256=ledger.artifacts.sha256(census_path),
            source_pointer=("checkpoints", "<any>", field),
        ))
    ledger.add(Quantity(
        id="conventions/census_procedures",
        claim="native unconditional procedures contributing to the attempt census",
        family="conventions", value=float(len(census["checkpoints"])
                                          + len(ledger.artifacts.json(CENSUS_SOURCES[1])["checkpoints"])),
        unit="procedures", kind="support_count", support_id="uncond_census_800", level="generation",
        source_path=census_path, source_sha256=ledger.artifacts.sha256(census_path),
        source_pointer=("checkpoints",),
    ))


# --------------------------------------------------------------------------- #
# Family 11 --- accounting subtrees an artifact records as scalar fields
# --------------------------------------------------------------------------- #

#: Subtrees whose scalar fields the manuscript quotes as accounting counts,
#: declared thresholds or fitted magnitudes that the artifact states without an
#: interval beside them. Each is emitted under its own field name, so the claim
#: is the artifact's own vocabulary rather than a restatement of it.
LEAF_SUBTREES = [
    ("archive/logs/shared/gate_higher_order_extended_20260923/proteingym_census.json", ("summary",),
     "proteingym_local_census", "higher_order", "measurement", "variants"),
    ("results/R4/gate_stability_20260924/endpoint/endpoint_qualification.json", ("row_accounting",),
     "stability_singles_family_group", "endpoint", "measurement", "rows"),
    ("results/R4/gate_stability_20260924/endpoint/endpoint_qualification.json", ("support_accounting",),
     "stability_singles_family_group", "endpoint", "measurement", "units"),
    ("results/R4/gate_stability_20260924/controls/controls_qualification.json", ("ladder",),
     "stability_singles_family_group", "stability_controls", "control", "kcal^2/mol^2"),
    ("results/R4/gate_stability_20260924/controls/controls_qualification.json", ("discarded",),
     "stability_singles_family_group", "stability_controls", "control", "kcal^2/mol^2"),
    ("archive/logs/R5/gate_retrieval_20260923/strata_declaration.json", (),
     "pairwise_all_64g", "retrieval_strata", "measurement", "units"),
]

#: Design constants the manuscript states that no declaration or receipt in
#: this repository records. Each names the artifact the locator was pointed at,
#: so the gap is a searched-for absence rather than an unexamined one. The
#: manuscript is not a source, so these stay absent from the table.
UNRECORDED_CONSTANTS = [
    ("constants/context/target_length_range",
     "the declared target length range of the related-context cohort",
     "context", "results/R5/context_homologue_20260826/context_homologue.json"),
    ("constants/readout/matched_baseline_groups",
     "the matched-baseline groups the admission reconstructed independently",
     "readout", f"{READOUT_ROOT}/final_admission.json"),
    ("constants/generative_control/corpus_records_per_length_stratum",
     "records retained per length stratum in the staged reference corpus",
     "generative_control", "archive/logs/R6/gate_generative_control/build/build_manifest.json"),
    ("constants/generative_control/pfam_library_size",
     "profiles in the retained Pfam library the oracle searches",
     "generative_control", "archive/logs/R6/gate_generative_control/build/qualification.json"),
    ("constants/crossed/profile_recomputation_deviation",
     "the maximum deviation on recomputing the cohort's profile scores from the retained "
     "column-frequency arrays",
     "local_context", "archive/logs/shared/crossed_controls_20260923/reports"),
    ("constants/retrieval/decomposition_residual",
     "the residual by which the whole-support estimate reproduces the unit-count-weighted mean of "
     "its strata",
     "retrieval_strata", "archive/logs/R5/gate_retrieval_20260923/retrieval_strata_report.json"),
]


#: Artifacts whose intervals carry no draw count anywhere in the file. Their
#: rows take ``UNRECORDED_DRAWS`` from the analysis modules, which is a
#: declared value and not a reading, so each is registered as a gap: a draw
#: count this table states and no file supports is exactly the provenance
#: field that drifted before.
UNRECORDED_DRAW_COUNTS = [
    ("constants/remote_homology/interval_draws", "remote_homology",
     "results/R5/remote_homology_20260924/panel/panel.json"),
    ("constants/readout_sweeps/depth_panel_interval_draws", "readout_sweeps",
     "results/R2/readout_depth_panel_20260925/readout_depth_panel.json"),
]


def family_unrecorded_draw_counts(ledger: Ledger) -> None:
    for identifier, family, looked_in in UNRECORDED_DRAW_COUNTS:
        ledger.gap(
            id=identifier,
            claim="the resampling draw count behind every interval this artifact reports",
            family=family,
            reason=f"the artifact records no draw count, so the {UNRECORDED_DRAWS}-draw figure its "
                   "rows carry is taken from the analysis module that ran the resampling and is a "
                   "declared value rather than a reading",
            looked_in=looked_in, reachability="not_recorded",
            searched="every mapping of the artifact for resamples, draws, bootstrap_draws or "
                     "finite_draws, and its bootstrap, declaration and recipe blocks")


def family_unrecorded_constants(ledger: Ledger) -> None:
    for identifier, claim, family, looked_in in UNRECORDED_CONSTANTS:
        ledger.gap(id=identifier, claim=claim, family=family,
                   reason="the locator was pointed at the declaring artifact and it records no such "
                          "constant; the manuscript is not a source for this table",
                   looked_in=looked_in, reachability="not_recorded",
                   searched="every scalar leaf of the artifact to four levels, and its declaration "
                            "strings")


CONTROLS_QUALIFICATION = "results/R4/gate_stability_20260924/controls/controls_qualification.json"
INDEL_RESTRICTED = f"{EPISTASIS}/panel/panel_indel_restricted.json"


def family_discard_magnitudes(ledger: Ledger) -> None:
    """The size of the failure that discards a control candidate.

    The receipt records a candidate's contribution as a signed increment, and a
    discarded candidate's is negative. A reader quotes the magnitude of the
    degradation, so the magnitude is emitted as its own row, derived from that
    one field and saying what it is, rather than left to a sign convention.
    """
    if not ledger.artifacts.exists(CONTROLS_QUALIFICATION):
        ledger.gap(id="stability_controls/discard_magnitudes",
                   claim="the magnitude of the degradation that discards each control candidate",
                   family="stability_controls", reason="the controls receipt is not retained here",
                   looked_in=CONTROLS_QUALIFICATION, reachability="cluster_only")
        return
    payload = ledger.artifacts.json(CONTROLS_QUALIFICATION)
    support = ledger.declare_support(
        "stability_singles_family_group",
        f"MegaScale single substitutions, {payload['groups']} wild-type family groups",
    )
    for entry in payload["ladder"]:
        if entry.get("qualified") or entry.get("min_increment_kcal2_mol2") is None:
            continue
        increment = float(entry["min_increment_kcal2_mol2"])
        if increment >= 0:
            continue
        candidate = entry["candidate"]
        ledger.add(Quantity(
            id=f"stability_controls/ladder/{candidate}/discard_magnitude",
            claim=f"magnitude of the held-out group-equal squared-error degradation that discards the "
                  f"{candidate} block, at the seed where its contribution is worst",
            family="stability_controls", value=abs(increment), unit="kcal^2/mol^2",
            kind="artifact_leaf", support_id=support, seed_set=SPLIT_SEEDS, level="control",
            source_path=CONTROLS_QUALIFICATION,
            source_sha256=ledger.artifacts.sha256(CONTROLS_QUALIFICATION),
            source_pointer=("<magnitude>", "ladder", candidate, "min_increment_kcal2_mol2"),
        ))


def family_failed_block_cost(ledger: Ledger) -> None:
    """What the declared dipeptide-and-tripeptide block costs the set it extends.

    Both correlations are recorded per design, so the cost is their difference
    rather than a quantity no run measured. The designs are model-independent,
    so one report per split seed carries it.
    """
    reports = sorted((REPO_ROOT / "archive/logs/shared/crossed_controls_20260923/reports").glob(
        "*/crossed_controls_*_fold*.json"))
    by_seed: dict[str, str] = {}
    for report in reports:
        relative = report.relative_to(REPO_ROOT).as_posix()
        seed = str(ledger.artifacts.json(relative).get("fold_seed"))
        by_seed.setdefault(seed, relative)
    if not by_seed:
        ledger.gap(id="local_context/failed_dipeptide_block",
                   claim="the held-cluster Spearman the declared dipeptide-and-tripeptide block costs "
                         "the sets it augments",
                   family="local_context", reason="no crossed-control report is retained here",
                   looked_in="archive/logs/shared/crossed_controls_20260923/reports",
                   reachability="cluster_only")
        return
    support = ledger.declare_support("anchor_128", ANCHOR_SUPPORT)
    for seed, relative in sorted(by_seed.items()):
        summaries = ledger.artifacts.json(relative)["summaries"]
        for base, extended, label in (("C", "C_L", "the composition control"),
                                      ("C_P", "C_L_P", "composition plus profile")):
            before, after = summaries.get(f"{base}_spearman"), summaries.get(f"{extended}_spearman")
            if not before or not after:
                continue
            for design in (base, extended):
                value = summaries[f"{design}_spearman"]
                identifier = f"local_context/crossed_design/{seed}/{design}"
                if identifier in {quantity.id for quantity in ledger.quantities}:
                    continue
                ledger.add(Quantity(
                    id=identifier,
                    claim=f"held-cluster Spearman of the {design.replace('_', '+')} design in the "
                          f"crossed-control study at split seed {seed}",
                    family="local_context", value=_point(value), unit="dimensionless Spearman",
                    kind="estimate", support_id=support, interval=_interval(value),
                    interval_kind=PERCENTILE_95, resampling_unit=value["unit"],
                    resampling_draws=int(value["resamples"]), seed_set=(int(seed),), level="control",
                    source_path=relative, source_sha256=ledger.artifacts.sha256(relative),
                    source_pointer=("summaries", f"{design}_spearman"),
                ))
            ledger.add(Quantity(
                id=f"local_context/failed_dipeptide_block/{seed}/{base}",
                claim=f"held-cluster Spearman the declared dipeptide-and-tripeptide block costs "
                      f"{label} at split seed {seed}",
                family="local_context", value=_point(before) - _point(after),
                unit="dimensionless Spearman", kind="artifact_leaf", support_id=support,
                seed_set=(int(seed),), level="control",
                source_path=relative, source_sha256=ledger.artifacts.sha256(relative),
                source_pointer=("<difference>", "summaries", f"{base}_spearman",
                                f"{extended}_spearman"),
            ))


def family_indel_evaluation(ledger: Ledger) -> None:
    """The cycles the post-freeze insertion-or-deletion defect admitted."""
    if not ledger.artifacts.exists(INDEL_RESTRICTED):
        ledger.gap(id="pairwise/indel_exclusion/cycles",
                   claim="cycles the insertion or deletion defect admitted, and their share",
                   family="pairwise", reason="the restricted-evaluation panel is not retained here",
                   looked_in=INDEL_RESTRICTED, reachability="cluster_only")
        return
    payload = ledger.artifacts.json(INDEL_RESTRICTED)
    support = ledger.declare_support("pairwise_all_64g", PAIRWISE_ALL_SUPPORT)
    pointer = ("excluded_evaluation_cycles", "cycles")
    try:
        cycles = float(resolve(payload, pointer))
    except KeyError:
        ledger.gap(id="pairwise/indel_exclusion/cycles",
                   claim="cycles the insertion or deletion defect admitted",
                   family="pairwise", reason="the panel records no excluded-cycle count",
                   looked_in=INDEL_RESTRICTED, reachability="not_recorded")
        return
    ledger.add(Quantity(
        id="pairwise/indel_exclusion/cycles",
        claim="cycles the post-freeze insertion or deletion defect admitted on truncated rows",
        family="pairwise", value=cycles, unit="double-mutant cycles", kind="census",
        support_id=support, level="measurement",
        no_interval_reason="a complete enumeration of the affected cycles",
        source_path=INDEL_RESTRICTED, source_sha256=ledger.artifacts.sha256(INDEL_RESTRICTED),
        source_pointer=pointer,
    ))
    total = next((quantity.value for quantity in ledger.quantities
                  if quantity.id == "pairwise/all/support/cycles"), None)
    if total:
        ledger.add(Quantity(
            id="pairwise/indel_exclusion/share",
            claim="share of the support the insertion or deletion defect touches",
            family="pairwise", value=100.0 * cycles / total, unit="percent", kind="ratio",
            support_id=support, level="measurement",
            source_path=INDEL_RESTRICTED, source_sha256=ledger.artifacts.sha256(INDEL_RESTRICTED),
            source_pointer=("<ratio>",) + pointer,
        ))


#: Leaf keys that name no quantity of their own and take their subtree's unit.
GENERIC_LEAF_KEYS = frozenset({"point", "value", "mean", "median", "low", "high", "estimate",
                               "difference", "augmented_mse", "baseline_mse"})

#: Branches that carry design widths, digests and fold bookkeeping rather than
#: quantities the manuscript can cite. Emitting them would fill the table with
#: incidental integers and make an integer match meaningless.
SKIPPED_LEAF_BRANCHES = frozenset({"dimensions", "code_sha256", "analysis_code_sha256",
                                   "prediction_digests", "selected_alphas", "assay_ids",
                                   "cluster_sizes", "held_families", "training_families",
                                   "inner_folds", "inner_family_rank_mse", "folds", "units",
                                   "group_members_under_contract", "members"})


def _leaf_unit(field: str, default: str) -> str:
    """The unit a leaf's own field name states, or nothing.

    An unrecognised field name yields no unit and the leaf is not emitted: a
    guessed unit would put a false fact on a row the manuscript is checked
    against.
    """
    if field in GENERIC_LEAF_KEYS:
        return default
    lowered = field.lower()
    if lowered.endswith("_kcal2_mol2") or "kcal2" in lowered:
        return "kcal^2/mol^2"
    if "kcal_mol" in lowered or lowered.endswith("_kcal"):
        return "kcal/mol"
    if "spearman" in lowered or "pearson" in lowered or lowered.endswith("_r2"):
        return "dimensionless"
    if "rate" in lowered or "fraction" in lowered or "share" in lowered:
        return "dimensionless fraction"
    for token, unit in (("kish", "effective units"), ("group", "family groups"), ("assay", "assays"),
                        ("variant", "variants"), ("cluster", "source clusters"),
                        ("site", "site pairs"), ("row", "rows"),
                        ("state", "distinct measured states"),
                        ("cycle", "double-mutant cycles"), ("attempt", "attempts"),
                        ("unit", "units"), ("threshold", "dimensionless fraction")):
        if token in lowered:
            return unit
    return ""


def family_leaf_subtrees(ledger: Ledger) -> None:
    for path, pointer, support_id, family, level, default_unit in LEAF_SUBTREES:
        if not ledger.artifacts.exists(path):
            ledger.gap(id=f"{family}/{'_'.join(pointer) or 'root'}",
                       claim=f"the scalar fields of {'/'.join(pointer) or 'the artifact root'}",
                       family=family, reason="artifact absent", looked_in=path)
            continue
        payload = ledger.artifacts.json(path)
        try:
            node = resolve(payload, pointer) if pointer else payload
        except KeyError:
            ledger.gap(id=f"{family}/{'_'.join(pointer)}",
                       claim=f"the scalar fields of {'/'.join(pointer)}", family=family,
                       reason="pointer does not resolve", looked_in=path)
            continue
        for leaf_pointer, field, value in _numeric_leaves(node, pointer):
            unit = _leaf_unit(field, default_unit)
            if not unit:
                continue
            ledger.add(Quantity(
                id=f"{family}/{'/'.join(leaf_pointer)}",
                claim=f"{field.replace('_', ' ')}, as {path.rsplit('/', 1)[-1]} records it",
                family=family, value=float(value), unit=unit, kind="artifact_leaf",
                support_id=support_id, level=level,
                source_path=path, source_sha256=ledger.artifacts.sha256(path),
                source_pointer=leaf_pointer,
            ))


def _numeric_leaves(node: object, prefix: tuple[str, ...],
                    depth: int = 0) -> list[tuple[tuple[str, ...], str, float]]:
    if depth > 4:
        return []
    found: list[tuple[tuple[str, ...], str, float]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, bool):
                continue
            if key in SKIPPED_LEAF_BRANCHES and isinstance(value, (dict, list)):
                continue
            if isinstance(value, (int, float)):
                # A leaf keyed by a split seed names no quantity of its own; the
                # field that does is the block it sits in.
                naming = prefix[-1] if re.fullmatch(r"\d{8}", str(key)) and prefix else key
                found.append((prefix + (key,), naming, value))
            else:
                found += _numeric_leaves(value, prefix + (key,), depth + 1)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            label = None
            if isinstance(value, dict):
                for naming in ("name", "block", "candidate", "step", "label", "id"):
                    if isinstance(value.get(naming), str):
                        label = value[naming]
                        break
            found += _numeric_leaves(value, prefix + (label or f"[{index}]",), depth + 1)
    return found


def family_derived_shares(ledger: Ledger) -> None:
    """Shares the manuscript quotes as percentages of a quantity already in the
    table. Each is computed from two rows rather than restated, so a change in
    either moves the share."""
    by_id = {quantity.id: quantity for quantity in ledger.quantities}
    pairs = [
        ("pairwise/all/control_share_of_additive_null",
         "share of the additive null's squared error the sequence and profile control removes",
         "pairwise", "pairwise/all/C|ADDITIVE_NULL/progen3-3b/20260923", "pairwise/all/additive_null"),
    ]
    for identifier, claim, family, numerator, denominator in pairs:
        top, bottom = by_id.get(numerator), by_id.get(denominator)
        if top is None or bottom is None or not bottom.value:
            ledger.gap(id=identifier, claim=claim, family=family,
                       reason="one of the two quantities the share divides is not in the table",
                       looked_in=numerator if top is None else denominator)
            continue
        ledger.add(Quantity(
            id=identifier, claim=claim, family=family,
            value=100.0 * top.value / bottom.value, unit="percent", kind="ratio",
            support_id=top.support_id, seed_set=top.seed_set, level=top.level,
            source_path=top.source_path, source_sha256=top.source_sha256,
            source_pointer=("<ratio>",) + top.source_pointer,
        ))


def family_generator_totals(ledger: Ledger) -> None:
    """Panel totals over the matched-generator cells, which the manuscript
    quotes as how often each cohort reached a curated family at all."""
    path = "results/R6/generative_control_20260924/gate_endpoints.json"
    payload = ledger.artifacts.json(path)
    support = ledger.declare_support("generator_gate_cells",
                                     generator_gate_support(len(payload["cells"])))
    totals: dict[tuple[str, str], float] = {}
    model_totals: dict[str, float] = {}
    for cell in payload["cells"]:
        for endpoint, block in (cell.get("endpoints") or {}).items():
            for cohort, contrast in (block.get("controls") or {}).items():
                if contrast.get("control_successes") is None:
                    continue
                totals[(endpoint, cohort)] = totals.get((endpoint, cohort), 0.0) \
                    + float(contrast["control_successes"])
                if contrast.get("model_rate") is not None and contrast.get("n_attempts"):
                    model_totals[endpoint] = model_totals.get(endpoint, 0.0)
            if block.get("model_successes") is not None:
                model_totals[endpoint] = model_totals.get(endpoint, 0.0) + float(block["model_successes"])
            elif (block.get("controls") or {}):
                sample = next(iter(block["controls"].values()))
                if sample.get("model_rate") is not None and sample.get("n_attempts"):
                    model_totals[endpoint] = model_totals.get(endpoint, 0.0) + \
                        float(sample["model_rate"]) * float(sample["n_attempts"])
    for (endpoint, cohort), value in sorted(totals.items()):
        ledger.add(Quantity(
            id=f"generative_control/totals/{endpoint}/{cohort}",
            claim=f"attempts of the {cohort} generator reaching the {endpoint} endpoint over all cells",
            family="generative_control", value=value, unit="attempts", kind="census",
            support_id=support,
            no_interval_reason="a complete enumeration over the gate's cells",
            seed_set=(int(payload["seed"]),), level="generation",
            source_path=path, source_sha256=ledger.artifacts.sha256(path),
            source_pointer=("cells", "<sum>", "endpoints", endpoint, "controls", cohort,
                            "control_successes"),
        ))
    for endpoint, value in sorted(model_totals.items()):
        ledger.add(Quantity(
            id=f"generative_control/totals/{endpoint}/model",
            claim=f"attempts of the models reaching the {endpoint} endpoint over all cells",
            family="generative_control", value=round(value), unit="attempts", kind="census",
            support_id=support,
            no_interval_reason="a complete enumeration over the gate's cells",
            seed_set=(int(payload["seed"]),), level="generation",
            source_path=path, source_sha256=ledger.artifacts.sha256(path),
            source_pointer=("cells", "<sum>", "endpoints", endpoint, "model_rate"),
        ))


# --------------------------------------------------------------------------- #

RECOMPUTATION_DIR = "archive/logs/shared/recomputation_20260924"
RECOMPUTATION_RECORD = "archive/docs/shared/recomputation-pipeline.md"

#: Departures of one design-product formation from another on a retained
#: extraction. These are point measurements of a deterministic computation on
#: fixed arrays: there is no population to draw from and a repeat returns the
#: same number, so they carry no interval and no resampling unit.
RECOMPUTATION_CLAIMS = [
    ("per_assay_thread_departure", 1.0,
     r"per-assay product is identical at 4 and at 48 BLAS threads at a maximum absolute difference "
     r"of (\d+\.\d+)",
     "largest absolute departure between the per-assay design formation at 4 and at 48 BLAS threads"),
    ("whole_panel_thread_departure", 1e-6,
     r"whole-panel product moves between those two thread counts by up to (\d+\.\d+)",
     "largest absolute departure of the whole-panel design formation between 4 and 48 BLAS threads"),
    ("whole_panel_blocking_departure", 1e-6,
     r"whole-panel product departs from the per-assay one by up to (\d+\.\d+)",
     "largest absolute departure of the whole-panel design formation from the per-assay formation"),
    ("largest_design_entry", 1.0,
     r"against a largest absolute entry of (\d+\.\d+)",
     "largest absolute design entry the two departures are read against"),
]


def family_recomputation(ledger: Ledger) -> None:
    """The recomputation pipeline's own identity measurements.

    These are the quantitative basis for the pipeline-identity limitations the
    catalogue already carries, so they belong in the same table as the numbers
    they qualify.
    """
    contract = f"{RECOMPUTATION_DIR}/gate_contract.json"
    retention = f"{RECOMPUTATION_DIR}/gate_retention.json"
    support = ledger.declare_support(
        "recomputation_gate_inventory",
        "the gates of the capability map, as the recomputation pipeline's retention inventory and "
        "gate contract enumerate them",
    )
    if ledger.artifacts.exists(contract):
        payload = ledger.artifacts.json(contract)
        gates = payload["gates"]
        for name, value, unit, kind, claim, pointer in (
            ("declared_quantities",
             float(sum(gate["transcription"]["checked"] for gate in gates.values())), "quantities",
             "census",
             "quantities the gate contract checks against the gate records it transcribes from",
             ("gates", "<sum>", "transcription", "checked")),
            ("gates_declared", float(len(gates)), "gates", "support_count",
             "gates the contract declares", ("gates",)),
            ("gates_conforming", float(len(payload["conforming"])), "gates", "count",
             "gates conforming to the contract", ("conforming",)),
            ("gates_diverging", float(len(payload["diverging"])), "gates", "count",
             "gates diverging from the contract", ("diverging",)),
        ):
            ledger.add(Quantity(
                id=f"recomputation/contract/{name}", claim=claim, family="recomputation",
                value=value, unit=unit, kind=kind, support_id=support, level="measurement",
                no_interval_reason=("a complete enumeration of the contract's declared quantities"
                                    if kind == "census" else None),
                source_path=contract, source_sha256=ledger.artifacts.sha256(contract),
                source_pointer=pointer,
            ))
    else:
        ledger.gap(id="recomputation/contract",
                   claim="the gate contract's declared quantities and conforming gates",
                   family="recomputation", reason="artifact absent", looked_in=contract)

    if ledger.artifacts.exists(retention):
        counts = ledger.artifacts.json(retention)["counts"]
        for field, unit, claim in (
            ("entries", "retention entries", "entries in the retention declaration"),
            ("with_representation_cells", "retention entries",
             "retention entries carrying representation cells"),
            ("representation_cells", "cells",
             "representation cells the retention declaration accounts for"),
        ):
            ledger.add(Quantity(
                id=f"recomputation/retention/{field}", claim=claim, family="recomputation",
                value=float(counts[field]), unit=unit, kind="support_count", support_id=support,
                level="measurement",
                source_path=retention, source_sha256=ledger.artifacts.sha256(retention),
                source_pointer=("counts", field),
            ))
    else:
        ledger.gap(id="recomputation/retention",
                   claim="the retention declaration's entry and cell counts",
                   family="recomputation", reason="artifact absent", looked_in=retention)

    record = ledger.artifacts.text(RECOMPUTATION_RECORD)
    formation = ledger.declare_support(
        "recomputation_formation_probe",
        "one retained extraction this host holds: 3 assays and 9 variants of proteinglm-7b-clm at "
        "hidden width 4,096, the arrays the product-formation departures are measured on",
    )
    for name, scale, pattern, claim in RECOMPUTATION_CLAIMS:
        matches = re.findall(pattern, record)
        if len(matches) != 1:
            ledger.gap(id=f"recomputation/formation/{name}", claim=claim, family="recomputation",
                       reason=f"the anchor matches {len(matches)} times in the pipeline record, so "
                              "the number cannot be bound to it",
                       looked_in=RECOMPUTATION_RECORD,
                       searched="the whole pipeline record for the declared anchor")
            continue
        ledger.add(Quantity(
            id=f"recomputation/formation/{name}", claim=claim, family="recomputation",
            value=float(matches[0]) * scale, unit="design matrix entry", kind="artifact_leaf",
            support_id=formation, level="measurement", source_kind="gate_record",
            source_path=RECOMPUTATION_RECORD,
            source_sha256=ledger.artifacts.sha256(RECOMPUTATION_RECORD),
            source_pointer=("<anchor>", pattern),
        ))
    ledger.gap(
        id="recomputation/formation/report",
        claim="the product-formation departures and the admitted cell's identity digest as a "
              "machine-readable report rather than as prose anchored in the pipeline record",
        family="recomputation",
        reason="no JSON report for the formation probe or the cell-identity replay is retained under "
               f"{RECOMPUTATION_DIR}, which holds the gate contract and the retention declaration "
               "only, so these quantities are record-backed and not artifact-backed",
        looked_in=RECOMPUTATION_DIR, reachability="not_recorded",
        searched="every file of the recomputation log directory",
    )


# --------------------------------------------------------------------------- #
# Family 13 --- declared design constants
# --------------------------------------------------------------------------- #

#: Blocks whose every scalar is a design constant: a width, a count, a seed, a
#: radius or a fold budget that a declaration or receipt fixes before any
#: measurement is read. A constant takes no interval and no resampling unit --
#: there is no population behind it -- and its source is the declaration that
#: fixes it, never the manuscript. A constant no declaration states is a gap.
CONSTANT_SUBTREES = [
    ("readout", ("feature_dimensions",), "design columns", "anchor_128",
     "width of the {field} design on the readout anchor"),
    ("readout", ("projection",), "projection coordinates", "anchor_128",
     "the frozen projection's {field}"),
    ("readout", ("alpha_grid",), "ridge penalty", "anchor_128",
     "a penalty on the declared ridge grid"),
    ("crossed", ("block_dimensions",), "design columns", "anchor_128",
     "width of the {field} block in the crossed-control designs"),
    ("lcg", ("feature_dimensions",), "design columns", "local_context_gate_cohort",
     "width of the {field} design in the local-context gate"),
    ("lcg", ("declaration", "widths"), "design columns", "local_context_gate_cohort",
     "declared width of the {field} candidate local control"),
    ("lcg", ("declaration", "radii"), "residues", "local_context_gate_cohort",
     "a declared window radius of the local control"),
    ("epistasis", ("recipe", "projection"), "projection coordinates", "pairwise_all_64g",
     "the interaction panel's projection {field}"),
    ("epistasis", ("recipe", "bootstrap"), "bootstrap draws", "pairwise_all_64g",
     "the interaction panel's bootstrap {field}"),
    ("contact", ("declaration",), "dimensionless fraction", "contact_indel_excluded",
     "the structural contrast's declared {field}"),
]

#: Constants an artifact states inside its own declaration prose. The regex
#: reads the artifact, not the manuscript, so the check stays non-circular.
CONSTANT_PATTERNS = [
    ("contact", ("declaration", "contact_primary"), r"([\d.]+)\s*A",
     "angstrom", "contact_indel_excluded",
     "the primary contact definition's minimum heavy-atom distance"),
    ("contact", ("declaration", "contact_secondary"), r"([\d.]+)\s*A",
     "angstrom", "contact_indel_excluded",
     "the secondary contact definition's C-beta distance"),
    ("crossed", ("feature_order", "local"), r"^(\d+) exact dipeptide",
     "design columns", "anchor_128",
     "exact dipeptide count differences in the block that failed its own qualification"),
]


def _constant_sources(ledger: Ledger) -> dict[str, str]:
    reports = readout_reports()
    crossed = sorted((REPO_ROOT / "archive/logs/shared/crossed_controls_20260923/reports").glob(
        "*/crossed_controls_*_fold*.json"))
    return {
        "readout": reports[0].relative_to(REPO_ROOT).as_posix() if reports else "",
        "crossed": crossed[0].relative_to(REPO_ROOT).as_posix() if crossed else "",
        "lcg": LOCAL_CONTEXT_QUALIFICATION,
        "epistasis": f"{EPISTASIS}/panel/panel.json",
        "contact": "results/R3/contact_bootstrap_corrected_20260927/epsilon_enrichment.json",
    }


def family_design_constants(ledger: Ledger) -> None:
    sources = _constant_sources(ledger)
    for key, pointer, unit, support_id, claim in CONSTANT_SUBTREES:
        path = sources.get(key)
        if not path or not ledger.artifacts.exists(path):
            ledger.gap(id=f"constants/{key}/{'_'.join(pointer)}",
                       claim=f"the design constants of {'/'.join(pointer)}",
                       family="constants", reason="the declaring artifact is not retained here",
                       looked_in=path or key, reachability="cluster_only")
            continue
        try:
            node = resolve(ledger.artifacts.json(path), pointer)
        except KeyError:
            ledger.gap(id=f"constants/{key}/{'_'.join(pointer)}",
                       claim=f"the design constants of {'/'.join(pointer)}",
                       family="constants",
                       reason="the declaring artifact carries no such block",
                       looked_in=path, reachability="not_recorded")
            continue
        for leaf_pointer, field, value in _numeric_leaves(node, pointer):
            ledger.add(Quantity(
                id=f"constants/{key}/{'/'.join(leaf_pointer)}",
                claim=claim.format(field=field), family="constants", value=float(value),
                unit="seed" if "seed" in field.lower() else unit, kind="constant",
                support_id=support_id, level="control",
                source_path=path, source_sha256=ledger.artifacts.sha256(path),
                source_pointer=leaf_pointer,
            ))
    for key, pointer, pattern, unit, support_id, claim in CONSTANT_PATTERNS:
        path = sources.get(key)
        if not path or not ledger.artifacts.exists(path):
            continue
        try:
            text = str(resolve(ledger.artifacts.json(path), pointer))
        except KeyError:
            ledger.gap(id=f"constants/{key}/{'_'.join(pointer)}", claim=claim, family="constants",
                       reason="the declaring artifact carries no such declaration",
                       looked_in=path, reachability="not_recorded")
            continue
        found = re.search(pattern, text)
        if not found:
            ledger.gap(id=f"constants/{key}/{'_'.join(pointer)}", claim=claim, family="constants",
                       reason="the declaration does not state this constant in a readable form",
                       looked_in=path, reachability="not_recorded")
            continue
        ledger.add(Quantity(
            id=f"constants/{key}/{'_'.join(pointer)}", claim=claim, family="constants",
            value=float(found.group(1)), unit=unit, kind="constant", support_id=support_id,
            level="control",
            source_path=path, source_sha256=ledger.artifacts.sha256(path),
            source_pointer=pointer,
        ))
    for path, pointer, unit, claim in (
        (f"{READOUT_ROOT}/final_admission.json", ("baseline_tolerance",), "dimensionless tolerance",
         "the admission's {field} for reconstructing every nested fold"),
        ("archive/logs/R5/gate_retrieval_20260923/retrieval_strata_report.json", ("estimators",),
         "dimensionless tolerance", "the strata report's {field}"),
    ):
        if not ledger.artifacts.exists(path):
            continue
        try:
            node = resolve(ledger.artifacts.json(path), pointer)
        except KeyError:
            continue
        for leaf_pointer, field, value in _numeric_leaves(node, pointer):
            ledger.add(Quantity(
                id=f"constants/admission/{'/'.join(leaf_pointer)}",
                claim=claim.format(field=field), family="constants", value=float(value), unit=unit,
                kind="constant", support_id="anchor_128", level="control",
                source_path=path, source_sha256=ledger.artifacts.sha256(path),
                source_pointer=leaf_pointer,
            ))
    _constant_reductions(ledger, sources)
    _burial_control(ledger)
    _identity_band_rates(ledger)


def _burial_control(ledger: Ledger) -> None:
    """The structural positive control, which resolves where the contrast does not."""
    path = "results/R3/contact_bootstrap_corrected_20260927/epsilon_enrichment.json"
    if not ledger.artifacts.exists(path):
        return
    payload = ledger.artifacts.json(path)
    primary = payload["primary_support"]
    pointer = ("supports", primary, "structure_positive_control")
    try:
        block = _block(payload, pointer)
    except KeyError:
        ledger.gap(id="structure_contact/positive_control",
                   claim="the burial control on the same structures", family="structure_contact",
                   reason="the artifact carries no positive-control block", looked_in=path,
                   reachability="not_recorded")
        return
    support = ledger.declare_support(
        "structure_positive_control",
        "the backgrounds of the structural cohort carrying at least four annotated positions, on "
        "which accessibility is read against mean measured effect",
    )
    for field, value in sorted(block.items()):
        if isinstance(value, dict) and value.get("point") is not None:
            interval = _interval(value)
            ledger.add(Quantity(
                id=f"structure_contact/positive_control/{field}",
                claim=f"{field.replace('_', ' ')} of the structural positive control",
                family="structure_contact", value=_point(value),
                unit="dimensionless" if "spearman" in field else "kcal/mol", kind="estimate",
                support_id=support, interval=interval,
                interval_kind=PERCENTILE_95 if interval else None,
                resampling_unit="background" if interval else None,
                resampling_draws=_draws(value, payload) if interval else None,
                no_interval_reason=None if interval else "the artifact retains no interval",
                seed_set=(SPLIT_SEEDS[0],), level="measurement",
                source_path=path, source_sha256=ledger.artifacts.sha256(path),
                source_pointer=pointer + (field,),
            ))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            unit = _leaf_unit(field, "backgrounds") or "backgrounds"
            ledger.add(Quantity(
                id=f"structure_contact/positive_control/{field}",
                claim=f"{field.replace('_', ' ')} of the structural positive control",
                family="structure_contact", value=float(value), unit=unit, kind="artifact_leaf",
                support_id=support, level="measurement",
                source_path=path, source_sha256=ledger.artifacts.sha256(path),
                source_pointer=pointer + (field,),
            ))


def _identity_band_rates(ledger: Ledger) -> None:
    """Recognition pooled over the generation gate's cells within each identity band.

    The gate records the band rates per cell; the pooled rate is the attempt-
    weighted mean over cells, so it is a reduction over the artifact rather than
    a number read from the sentence that quotes it.
    """
    path = "results/R6/generative_control_20260924/gate_endpoints.json"
    if not ledger.artifacts.exists(path):
        return
    payload = ledger.artifacts.json(path)
    support = ledger.declare_support("generator_gate_cells",
                                     generator_gate_support(len(payload["cells"])))
    pooled: dict[tuple[str, str], list[float]] = {}
    for cell in payload["cells"]:
        for band, block in (cell.get("identity_strata") or {}).items():
            if not isinstance(block, dict):
                continue
            for endpoint, inner in block.items():
                if not isinstance(inner, dict):
                    continue
                controls = inner.get("controls") or {}
                if not controls:
                    continue
                sample = next(iter(controls.values()))
                rate, attempts = sample.get("model_rate"), sample.get("n_attempts")
                if rate is None or not attempts:
                    continue
                totals = pooled.setdefault((band, endpoint), [0.0, 0.0])
                totals[0] += float(rate) * float(attempts)
                totals[1] += float(attempts)
    for (band, endpoint), (hits, attempts) in sorted(pooled.items()):
        if not attempts:
            continue
        ledger.add(Quantity(
            id=f"generative_control/pooled_bands/{band}/{endpoint}/rate",
            claim=f"{endpoint} recognition rate pooled over the gate's cells in the {band} identity "
                  "band against the searched snapshot",
            family="generative_control", value=hits / attempts, unit="dimensionless rate",
            kind="census", support_id=support,
            no_interval_reason="a complete enumeration of the band's attempts across the gate's cells",
            seed_set=(int(payload["seed"]),), level="generation",
            source_path=path, source_sha256=ledger.artifacts.sha256(path),
            source_pointer=("cells", "<sum>", "identity_strata", band, endpoint, "model_rate"),
        ))
        ledger.add(Quantity(
            id=f"generative_control/pooled_bands/{band}/{endpoint}/attempts",
            claim=f"attempts in the {band} identity band across the gate's cells",
            family="generative_control", value=attempts, unit="attempts", kind="census",
            support_id=support,
            no_interval_reason="a complete enumeration of the gate's cells",
            seed_set=(int(payload["seed"]),), level="generation",
            source_path=path, source_sha256=ledger.artifacts.sha256(path),
            source_pointer=("cells", "<sum>", "identity_strata", band, endpoint, "n_attempts"),
        ))
    # The identity bands are declared boundaries, and the manuscript quotes
    # them; the artifact states them as the band keys it aggregates under.
    for band in sorted({band for cell in payload["cells"]
                        for band in (cell.get("identity_strata") or {})}):
        found = re.findall(r"\d+", band)
        if not found or band.startswith("not_") or band.startswith("no_"):
            continue
        for index, bound in enumerate(found):
            ledger.add(Quantity(
                id=f"constants/generative_control/identity_band/{band}/{index}",
                claim=f"a declared identity boundary of the {band} recognition band",
                family="constants", value=float(bound), unit="percent", kind="constant",
                support_id=support, level="generation",
                source_path=path, source_sha256=ledger.artifacts.sha256(path),
                source_pointer=("cells", "<any>", "identity_strata", band),
            ))
    ledger.gap(
        id="generation_census/native_completion_counts",
        claim="ProGen3-3B's native completion accounting: compiler-accepted products, budget-censored "
              "continuations, and how many of each carry a profile",
        family="generation_census",
        reason="the figure table that states them carries no source binding, and no retained primary "
               "record states them: the census artifacts hold the attempt totals and the generation "
               "analysis holds per-sequence rows, neither with a compiler or stop-status accounting",
        looked_in="results/R6/unconditional_diversity_3b_20260919/unconditional_diversity.json",
        reachability="not_recorded",
        searched="every numeric leaf of the census artifacts and of the generation analysis, and the "
                 "column names of the retained per-attempt ledgers",
    )


def _constant_reductions(ledger: Ledger, sources: dict[str, str]) -> None:
    """Constants a declaration fixes as a sum or a per-gate entry."""
    census = "archive/logs/shared/gate_higher_order_extended_20260923/proteingym_census.json"
    if ledger.artifacts.exists(census):
        order_census = ledger.artifacts.json(census)["summary"]["order_census"]
        for floor in (2, 3):
            ledger.add(Quantity(
                id=f"higher_order/census/variants_order_ge{floor}",
                claim=f"variants carrying {floor} or more substitutions across the local assays",
                family="higher_order",
                value=float(sum(count for order, count in order_census.items() if int(order) >= floor)),
                unit="variants", kind="census", support_id="proteingym_local_census",
                no_interval_reason="a complete enumeration of the local assays",
                level="measurement",
                source_path=census, source_sha256=ledger.artifacts.sha256(census),
                source_pointer=("summary", "order_census", f"<sum over orders >= {floor}>"),
            ))
    retention = f"{RECOMPUTATION_DIR}/gate_retention.json"
    if ledger.artifacts.exists(retention):
        for gate in ledger.artifacts.json(retention)["gates"]:
            cells = gate.get("representation_cells")
            if not cells:
                continue
            ledger.add(Quantity(
                id=f"recomputation/retention/{gate['gate']}/representation_cells",
                claim=f"representation cells the {gate['gate'].replace('_', ' ')} gate retains for "
                      "recomputation",
                family="recomputation", value=float(cells), unit="cells", kind="support_count",
                support_id="recomputation_gate_inventory", level="measurement",
                source_path=retention, source_sha256=ledger.artifacts.sha256(retention),
                source_pointer=("gates", gate["gate"], "representation_cells"),
            ))
    text_aa = "results/R1/source_recovery_20260917/text_aa_analyse.json"
    if ledger.artifacts.exists(text_aa):
        block = next(iter(ledger.artifacts.json(text_aa)["common_all13"]["per_model"].values()))["raw"]
        support = f"text_aa_{int(block['n_assays'])}a_{int(block['n_units'])}c"
        for field, unit, claim in (("n_assays", "assays", "assays in the literal-string intersection"),
                                   ("n_units", "wild-type clusters",
                                    "wild-type clusters in the literal-string intersection")):
            ledger.add(Quantity(
                id=f"text_aa/support/{field}", claim=claim, family="text_aa",
                value=float(block[field]), unit=unit, kind="support_count", support_id=support,
                level="likelihood",
                source_path=text_aa, source_sha256=ledger.artifacts.sha256(text_aa),
                source_pointer=("common_all13", "per_model", "<any>", "raw", field),
            ))
    conditional = ("results/R6/generation_evidence_20260905/analysis/20260904232436_00129607e3c7/main/"
                   "generation_biology_analysis.json")
    if ledger.artifacts.exists(conditional):
        rows = [entry for entry in ledger.artifacts.json(conditional)["profile_ledger"]
                if entry.get("role") == "generation"
                and entry.get("condition") in ("requested", "mismatched")]
        if rows:
            attempts = sorted({int(entry["n_attempts"]) for entry in rows})
            ledger.add(Quantity(
                id="conventions/conditional_attempts_per_cell",
                claim="attempts per conditional generation cell",
                family="conventions", value=float(attempts[0]), unit="attempts", kind="constant",
                support_id="conditional_class_cells", level="generation",
                source_path=conditional, source_sha256=ledger.artifacts.sha256(conditional),
                source_pointer=("profile_ledger", "<any>", "n_attempts"),
            ))
            ledger.add(Quantity(
                id="conventions/conditional_classes_per_checkpoint",
                claim="initial classes offered per conditioned checkpoint",
                family="conventions",
                value=float(max(len({entry["class_key"] for entry in rows if entry["arm"] == arm})
                                for arm in {entry["arm"] for entry in rows})),
                unit="classes", kind="support_count", support_id="conditional_class_cells",
                level="generation",
                source_path=conditional, source_sha256=ledger.artifacts.sha256(conditional),
                source_pointer=("profile_ledger", "<reduction>", "class_key"),
            ))


# --------------------------------------------------------------------------- #
# Family 14 --- the remote-homology gate and the Domainome endpoint
# --------------------------------------------------------------------------- #

REMOTE_ROOT = "results/R5/remote_homology_20260924"
REMOTE_QUALIFICATION = f"{REMOTE_ROOT}/endpoint_qualification.json"
REMOTE_PANEL = f"{REMOTE_ROOT}/panel/panel.json"
DOMAINOME_QUALIFICATION = "results/R1/external_confirmation_20260924/endpoint_qualification.json"

#: The unit a channel-decomposition field is measured in, read from its own
#: name. A ratio is dimensionless and its components are not, which is the
#: distinction this table exists to keep: a shared component that shrinks is
#: not noise that rises, and only separate rows make the two impossible to
#: restate as one another.
DECOMPOSITION_UNITS = (
    ("_log2_enrichment", "log2 enrichment"),
    ("_kcal_mol", "kcal/mol"),
    ("ratio", "dimensionless ratio"),
    ("pearson_r", "dimensionless"),
)


def _decomposition_unit(field: str) -> str:
    for suffix, unit in DECOMPOSITION_UNITS:
        if field.endswith(suffix) or suffix == field:
            return unit
    return ""


def _emit_decomposition(ledger: Ledger, *, prefix: str, claim_prefix: str, family: str, path: str,
                        pointer: tuple[str, ...], block: dict, support_id: str,
                        resampling_unit: str, level: str = "measurement") -> None:
    draws = _draws(block)
    for field, value in sorted(block.items()):
        if not isinstance(value, dict) or value.get("point") is None:
            continue
        unit = _decomposition_unit(field)
        if not unit:
            continue
        interval = _interval(value)
        ledger.add(Quantity(
            id=f"{prefix}/{field}",
            claim=f"{claim_prefix}: {field.replace('_', ' ')}", family=family,
            value=_point(value), unit=unit, kind="estimate", support_id=support_id,
            interval=interval, interval_kind=PERCENTILE_95 if interval else None,
            resampling_unit=resampling_unit if interval else None,
            resampling_draws=draws if interval else None,
            no_interval_reason=None if interval else "the artifact retains no interval",
            seed_set=(SPLIT_SEEDS[0],), level=level,
            source_path=path, source_sha256=ledger.artifacts.sha256(path),
            source_pointer=pointer + (field,),
        ))


def family_remote_homology(ledger: Ledger) -> None:
    """The remote-homology gate, from its qualification and panel receipts."""
    if not ledger.artifacts.exists(REMOTE_QUALIFICATION):
        ledger.gap(id="remote_homology/endpoint", claim="the remote-homology endpoint's qualification",
                   family="remote_homology", reason="the qualification receipt is not staged here",
                   looked_in=REMOTE_QUALIFICATION, reachability="cluster_only")
        return
    payload = ledger.artifacts.json(REMOTE_QUALIFICATION)
    grouping = payload["grouping"]
    cohort = ledger.declare_support(
        "remote_homology_cohort",
        f"the staged remote-homology cohort: {grouping['backgrounds']} backgrounds in "
        f"{grouping['groups']} family groups at 30% identity and 80% coverage, on a combined "
        "free-energy scale in kcal/mol",
    )
    _emit_decomposition(
        ledger, prefix="remote_homology/level_scale", family="remote_homology",
        claim_prefix="the endpoint's level-scale channel disagreement",
        path=REMOTE_QUALIFICATION, pointer=("level_scale_channel_disagreement",),
        block={key: {"point": value} for key, value in
               payload["level_scale_channel_disagreement"].items()
               if isinstance(value, (int, float)) and not isinstance(value, bool)},
        support_id=cohort, resampling_unit="family group",
    )
    for stratum, block in sorted(payload["effect_scale_agreement"].items()):
        support = ledger.declare_support(
            f"remote_homology_{stratum}",
            f"the {stratum.replace('stratum_', '')} identity stratum of the remote-homology cohort: "
            f"{block['backgrounds']} backgrounds in {block['units']} {block['unit']}s over "
            f"{block['variants']} variants",
        )
        _emit_decomposition(
            ledger, prefix=f"remote_homology/effect_scale/{stratum}", family="remote_homology",
            claim_prefix=f"the effect-scale channel decomposition on the {stratum} support",
            path=REMOTE_QUALIFICATION,
            pointer=("effect_scale_agreement", stratum, "decomposition"),
            block=block["decomposition"], support_id=support,
            resampling_unit=str(block["unit"]),
        )
        for field, unit, kind in (("backgrounds", "backgrounds", "support_count"),
                                  ("units", "resampling units", "support_count"),
                                  ("sites", "sites", "support_count"),
                                  ("variants", "variants", "support_count")):
            ledger.add(Quantity(
                id=f"remote_homology/effect_scale/{stratum}/{field}",
                claim=f"{field} on the {stratum} support of the remote-homology cohort",
                family="remote_homology", value=float(block[field]), unit=unit, kind=kind,
                support_id=support, level="measurement",
                source_path=REMOTE_QUALIFICATION,
                source_sha256=ledger.artifacts.sha256(REMOTE_QUALIFICATION),
                source_pointer=("effect_scale_agreement", stratum, field),
            ))
    for field, count in sorted(payload["identity_bands"]["backgrounds_per_band"].items()):
        ledger.add(Quantity(
            id=f"remote_homology/identity_band/{field}",
            claim=f"backgrounds in the {field.replace('_', ' ')} identity band",
            family="remote_homology", value=float(count), unit="backgrounds", kind="support_count",
            support_id=cohort, level="measurement",
            source_path=REMOTE_QUALIFICATION,
            source_sha256=ledger.artifacts.sha256(REMOTE_QUALIFICATION),
            source_pointer=("identity_bands", "backgrounds_per_band", field),
        ))
    _remote_homology_panel(ledger, cohort)


def _remote_homology_panel(ledger: Ledger, cohort: str) -> None:
    if not ledger.artifacts.exists(REMOTE_PANEL):
        ledger.gap(id="remote_homology/panel", claim="the remote-homology panel's per-arm increments",
                   family="remote_homology", reason="the panel receipt is not staged here",
                   looked_in=REMOTE_PANEL, reachability="cluster_only")
        return
    payload = ledger.artifacts.json(REMOTE_PANEL)
    effective = payload["effective_units"]
    for field in ("kish_effective_domains", "kish_effective_groups", "kish_effective_sites"):
        ledger.add(Quantity(
            id=f"remote_homology/panel/{field}",
            claim=f"{field.replace('kish_effective_', 'Kish effective ')} behind the "
                  "remote-homology panel",
            family="remote_homology", value=float(effective[field]),
            unit=f"effective {field.rsplit('_', 1)[-1]}", kind="effective_count",
            support_id=cohort, weighting_convention=str(effective["weighting"]),
            level="measurement",
            source_path=REMOTE_PANEL, source_sha256=ledger.artifacts.sha256(REMOTE_PANEL),
            source_pointer=("effective_units", field),
        ))
    for contrast, block in sorted(payload["panel"].items()):
        median = block.get("panel_median_three_seed_mean")
        if median is None:
            continue
        level = "representation" if "representation" in contrast else "likelihood"
        ledger.add(Quantity(
            id=f"remote_homology/panel/{contrast}/median",
            claim=f"panel median of the three-seed mean {contrast.replace('_', ' ')} increment",
            family="remote_homology", value=float(median),
            unit="dimensionless Spearman" if contrast.endswith("spearman")
                 else "squared normalised effect", kind="estimate", support_id=cohort,
            no_interval_reason="a median over arms, not a resampled estimate",
            seed_set=SPLIT_SEEDS, level=level,
            source_path=REMOTE_PANEL, source_sha256=ledger.artifacts.sha256(REMOTE_PANEL),
            source_pointer=("panel", contrast, "panel_median_three_seed_mean"),
        ))
    flagged: dict[str, list[str]] = {}
    for arm, block in sorted(payload["arms"].items()):
        for side, outcome in sorted((block.get("outcome") or {}).items()):
            if not isinstance(outcome, dict):
                continue
            flag = bool(outcome.get("remote_resolved_without_its_positive_control"))
            for field in ("close_resolved", "remote_resolved", "positive_control_fired",
                          "remote_resolved_without_its_positive_control"):
                if field not in outcome:
                    continue
                ledger.add(Quantity(
                    id=f"remote_homology/outcome/{arm}/{side}/{field}",
                    claim=f"{arm}, {side} control set: {field.replace('_', ' ')} on the "
                          "remote-homology gate",
                    family="remote_homology", value=float(bool(outcome[field])),
                    unit="flag", kind="constant", support_id=cohort,
                    verdict="unresolved" if flag else None, level="likelihood",
                    source_path=REMOTE_PANEL, source_sha256=ledger.artifacts.sha256(REMOTE_PANEL),
                    source_pointer=("arms", arm, "outcome", side, field),
                ))
            if flag:
                flagged.setdefault(side, []).append(arm)
        for contrast, cell in sorted((block.get("cells") or {}).items()):
            level = "representation" if "representation" in contrast else "likelihood"
            for seed, per_seed in sorted((cell.get("per_seed") or {}).items()):
                if per_seed.get("point") is None:
                    continue
                remote_only = any(arm in arms for arms in flagged.values())
                ledger.add(Quantity(
                    id=f"remote_homology/{arm}/{contrast}/{seed}",
                    claim=f"{arm} at split seed {seed}: {cell['description']}"
                          + (", on an arm whose remote stratum resolves without its own positive "
                             "control firing, which is not a resolution" if remote_only else ""),
                    family="remote_homology", value=float(per_seed["point"]),
                    unit=str(per_seed.get("unit") or "squared normalised effect"), kind="estimate",
                    support_id=cohort,
                    interval=tuple(map(float, per_seed["interval"])) if per_seed.get("interval")
                             else None,
                    interval_kind=PERCENTILE_95 if per_seed.get("interval") else None,
                    resampling_unit="family group" if per_seed.get("interval") else None,
                    resampling_draws=(_draws(per_seed, cell, payload) or UNRECORDED_DRAWS)
                                     if per_seed.get("interval") else None,
                    no_interval_reason=None if per_seed.get("interval") else
                    "the panel records no interval for this cell",
                    seed_set=(int(seed),), level=level,
                    verdict="unresolved" if remote_only else None,
                    source_path=REMOTE_PANEL, source_sha256=ledger.artifacts.sha256(REMOTE_PANEL),
                    source_pointer=("arms", arm, "cells", contrast, "per_seed", seed, "point"),
                ))
    for side, arms in sorted(flagged.items()):
        ledger.add(Quantity(
            id=f"remote_homology/panel/{side}/arms_remote_resolved_without_positive_control",
            claim=f"arms whose remote stratum resolves while their own positive control does not "
                  f"fire, on the {side} control set; these are flagged and are not resolutions",
            family="remote_homology", value=float(len(arms)), unit="checkpoints", kind="count",
            support_id=cohort, verdict="unresolved", seed_set=SPLIT_SEEDS, level="likelihood",
            source_path=REMOTE_PANEL, source_sha256=ledger.artifacts.sha256(REMOTE_PANEL),
            source_pointer=("arms", "<reduction>", "outcome", side,
                            "remote_resolved_without_its_positive_control"),
        ))
    for field, unit, claim in (("qualified_control_set", "control blocks",
                                "blocks in the gate's qualified control set"),
                               ("secondary_control_set", "control blocks",
                                "blocks in the gate's secondary control set"),
                               ("split_seeds", "split seeds", "declared split seeds"),
                               ("roster_arms", "checkpoints", "arms on the declared roster"),
                               ("arms_fitted", "checkpoints", "arms fitted")):
        value = payload.get(field)
        if value is None:
            continue
        ledger.add(Quantity(
            id=f"remote_homology/panel/{field}",
            claim=claim, family="remote_homology",
            value=float(len(value) if isinstance(value, list) else value), unit=unit,
            kind="support_count", support_id=cohort, level="measurement",
            source_path=REMOTE_PANEL, source_sha256=ledger.artifacts.sha256(REMOTE_PANEL),
            source_pointer=(field,),
        ))


def family_domainome_endpoint(ledger: Ledger) -> None:
    """The Domainome endpoint's qualification, whose channel unit is log2 enrichment.

    No likelihood or confirmation result exists on this endpoint, so only the
    qualification is registered. Its unit differs from every other support in
    the table, which is what makes a cross-unit comparison refusable.
    """
    if not ledger.artifacts.exists(DOMAINOME_QUALIFICATION):
        ledger.gap(id="domainome/endpoint", claim="the Domainome endpoint's qualification",
                   family="domainome", reason="the qualification receipt is not staged here",
                   looked_in=DOMAINOME_QUALIFICATION, reachability="cluster_only")
        return
    payload = ledger.artifacts.json(DOMAINOME_QUALIFICATION)
    declared = payload["declared_cohort"]["effective_units_over_replicate_rows"]
    support = ledger.declare_support(
        "domainome_declared_cohort",
        f"the declared Domainome cohort: {declared['domains']} domains in {declared['groups']} "
        f"family groups over {declared['variants']} substitutions, measured as log2 enrichment in a "
        "protein-fragment complementation assay and not on any free-energy scale",
    )
    for field, unit, kind in (("domains", "domains", "support_count"),
                              ("groups", "family groups", "support_count"),
                              ("sites", "sites", "support_count"),
                              ("variants", "variants", "support_count")):
        ledger.add(Quantity(
            id=f"domainome/declared_cohort/{field}",
            claim=f"{field} in the declared Domainome cohort", family="domainome",
            value=float(declared[field]), unit=unit, kind=kind, support_id=support,
            level="measurement",
            source_path=DOMAINOME_QUALIFICATION,
            source_sha256=ledger.artifacts.sha256(DOMAINOME_QUALIFICATION),
            source_pointer=("declared_cohort", "effective_units_over_replicate_rows", field),
        ))
    for field in ("kish_effective_domains", "kish_effective_groups", "kish_effective_sites"):
        ledger.add(Quantity(
            id=f"domainome/declared_cohort/{field}",
            claim=f"{field.replace('kish_effective_', 'Kish effective ')} in the declared "
                  "Domainome cohort",
            family="domainome", value=float(declared[field]),
            unit=f"effective {field.rsplit('_', 1)[-1]}", kind="effective_count",
            support_id=support, weighting_convention=str(declared["weighting"]),
            level="measurement",
            source_path=DOMAINOME_QUALIFICATION,
            source_sha256=ledger.artifacts.sha256(DOMAINOME_QUALIFICATION),
            source_pointer=("declared_cohort", "effective_units_over_replicate_rows", field),
        ))
    for scope, block in sorted(payload["replicate_floor"].items()):
        if not isinstance(block, dict) or scope == "between_replicate_sd_over_three_replicates":
            continue
        _emit_decomposition(
            ledger, prefix=f"domainome/replicate_floor/{scope}", family="domainome",
            claim_prefix=f"the Domainome replicate floor on the {scope.replace('_', ' ')}",
            path=DOMAINOME_QUALIFICATION, pointer=("replicate_floor", scope), block=block,
            support_id=support, resampling_unit="family group",
        )
    for field, value in sorted(payload["clean_support"].get("declared", {}).items()):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        ledger.add(Quantity(
            id=f"domainome/clean_support/{field}",
            claim=f"{field.replace('_', ' ')} on the Domainome clean support",
            family="domainome", value=float(value), unit=_leaf_unit(field, "domains") or "domains",
            kind="support_count", support_id=support, level="measurement",
            source_path=DOMAINOME_QUALIFICATION,
            source_sha256=ledger.artifacts.sha256(DOMAINOME_QUALIFICATION),
            source_pointer=("clean_support", "declared", field),
        ))


# --------------------------------------------------------------------------- #
# Family 15 --- the readout-class and depth sweeps that settle the readout
# --------------------------------------------------------------------------- #

SWEEP_EXCERPT = "results/R2/readout_sweeps_20260924/sweep_cell_excerpt.json"

#: The three endpoints the class axis reports per class, and what each answers.
#: They are separate rows because they answer different questions and a count on
#: one is not a count on another: how well the representation predicts on its
#: own, how it compares with the checkpoint's own likelihood, and what it adds
#: over the matched supervised baseline. Only the last is the biological one.
SWEEP_ENDPOINTS = {
    "R_spearman": ("the representation's own held-cluster Spearman as a predictor",
                   "dimensionless Spearman"),
    "R_minus_raw_M_spearman": ("the representation read against the checkpoint's own likelihood",
                               "dimensionless Spearman"),
    "delta_spearman": ("the representation's increment over the matched supervised baseline",
                       "dimensionless Spearman"),
}


def family_readout_sweeps(ledger: Ledger) -> None:
    """The reassessment that settles which boundary the readout null belongs to.

    It licenses strengthening a statement about breadth --- how many cells carry
    resolved information --- and licenses none about magnitude. The ceilings are
    therefore registered as ceilings, with the admitted maximum beside them and
    the difference as its own row, so no reading can turn them into a rise.
    """
    if not ledger.artifacts.exists(SWEEP_EXCERPT):
        ledger.gap(id="readout_sweeps/cells", claim="the readout-class and depth sweeps' fitted cells",
                   family="readout_sweeps",
                   reason="the sweep cells are retained in the cluster store and no excerpt of them "
                          "is staged here",
                   looked_in=SWEEP_EXCERPT, reachability="cluster_only")
        return
    payload = ledger.artifacts.json(SWEEP_EXCERPT)
    digest = ledger.artifacts.sha256(SWEEP_EXCERPT)
    rows: list[dict] = []
    seen: dict[str, float] = {}
    for cell in payload["cells"]["class"]:
        block = cell["block"]
        summaries = block.get("summaries") or {}
        classes = sorted({key.split("/")[0] for key in summaries if "/" in key})
        support = ledger.declare_support(
            f"sweep_panel_{block['panel']}",
            f"the {block['panel']} panel of the readout reassessment, refitted on the same frozen "
            "states with the cohort, supports, fold maps, seeds and resampling contract held fixed",
        )
        for readout_class in classes:
            for endpoint, (claim, unit) in SWEEP_ENDPOINTS.items():
                value = summaries.get(f"{readout_class}/{endpoint}")
                if not isinstance(value, dict) or value.get("point") is None \
                        or value.get("degenerate"):
                    continue
                interval = _interval(value)
                seed = block.get("fold_seed")
                identifier = (f"readout_sweeps/class/{block['arm']}/{block['panel']}/{seed}/"
                              f"{readout_class}/{endpoint}")
                # The retained set holds more cells than the admitted 132: a
                # re-run writes a second file for the same (arm, panel, seed).
                # The first is kept and the repeat is checked against it rather
                # than silently dropped or silently doubled.
                existing = seen.get(identifier)
                if existing is not None:
                    if abs(existing - _point(value)) > 1e-9:
                        ledger.add(Quantity(
                            id=f"{identifier}/superseded_snapshot_difference",
                            claim="how far this cell's value in a superseded snapshot sits from the "
                                  "admitted one; the admission receipt resolves every repeated cell "
                                  "to the admitted snapshot, so this is a measured difference "
                                  "between two runs and not an ambiguity about which was published",
                            family="readout_sweeps", value=abs(existing - _point(value)),
                            unit="dimensionless Spearman", kind="artifact_leaf",
                            support_id=("class_sweep_admitted_cells"
                                        if "class_sweep_admitted_cells" in ledger.supports
                                        else support), level="representation",
                            conditioning="readout_selection",
                            source_path=SWEEP_EXCERPT, source_sha256=digest,
                            source_pointer=("<difference>", "cells", "class", cell["path"]),
                        ))
                    continue
                seen[identifier] = _point(value)
                ledger.add(Quantity(
                    id=identifier,
                    claim=f"{block['arm']} on the {block['panel']} panel at split seed {seed} under "
                          f"the {readout_class} readout class: {claim}",
                    family="readout_sweeps", value=_point(value), unit=unit, kind="estimate",
                    support_id=support, interval=interval,
                    interval_kind=PERCENTILE_95 if interval else None,
                    resampling_unit=str(value.get("unit")) if interval else None,
                    resampling_draws=int(value["resamples"]) if interval and value.get("resamples")
                                     else None,
                    no_interval_reason=None if interval else "the cell records no interval",
                    seed_set=(int(seed),) if seed else (), level="representation",
                    conditioning="readout_selection",
                    source_path=SWEEP_EXCERPT, source_sha256=digest,
                    source_pointer=("cells", "class", cell["path"], "summaries",
                                    f"{readout_class}/{endpoint}"),
                ))
                rows.append({"arm": block["arm"], "panel": block["panel"], "seed": seed,
                             "class": readout_class, "endpoint": endpoint,
                             "value": _point(value),
                             "resolved": bool(value.get("excludes_zero")) and _point(value) > 0})
    _sweep_reductions(ledger, rows, digest)
    _sweep_depth_receipts(ledger, payload, digest)


def _sweep_axis(readout_class: str) -> str:
    """Which axis a fitted class belongs to.

    The readout-class axis varies the fitted function; the pooled-depth axis
    varies which retained block and pooling rule the representation comes from.
    A count over one is not a count over the other, so they are never mixed in a
    reduction.
    """
    return "class" if readout_class.startswith("C") else "pooled_depth"


def _sweep_reductions(ledger: Ledger, rows: list[dict], digest: str) -> None:
    """Breadth counts per class, and magnitude ceilings that stay ceilings."""
    if not rows:
        return
    _sweep_arm_panel_counts(ledger, rows, digest)
    _sweep_admitted_comparisons(ledger, rows, digest)
    _sweep_breadth_comparisons(ledger, rows, digest)
    panels = sorted({row["panel"] for row in rows})
    for endpoint in SWEEP_ENDPOINTS:
        for panel in panels:
            support = f"sweep_panel_{panel}"
            scoped = [row for row in rows if row["endpoint"] == endpoint and row["panel"] == panel]
            if not scoped:
                continue
            per_class = {}
            for readout_class in sorted({row["class"] for row in scoped}):
                cells = [row for row in scoped if row["class"] == readout_class]
                resolved = sum(row["resolved"] for row in cells)
                per_class[readout_class] = resolved
                ledger.add(Quantity(
                    id=f"readout_sweeps/breadth/{panel}/{endpoint}/{readout_class}/resolved_cells",
                    claim=f"cells resolving above zero on the {panel} panel for {endpoint} under the "
                          f"{readout_class} readout class",
                    family="readout_sweeps", value=float(resolved), unit="fit cells", kind="count",
                    support_id=support, level="representation", conditioning="readout_selection",
                    verdict="supported" if resolved else "not_detected",
                    source_path=SWEEP_EXCERPT, source_sha256=digest,
                    source_pointer=("cells", "class", "<reduction>", endpoint, readout_class),
                ))
                ledger.add(Quantity(
                    id=f"readout_sweeps/breadth/{panel}/{endpoint}/{readout_class}/cells",
                    claim=f"cells carrying {endpoint} on the {panel} panel under the "
                          f"{readout_class} readout class",
                    family="readout_sweeps", value=float(len(cells)), unit="fit cells",
                    kind="support_count", support_id=support, level="representation",
                    conditioning="readout_selection",
                    source_path=SWEEP_EXCERPT, source_sha256=digest,
                    source_pointer=("cells", "class", "<reduction>", endpoint, readout_class),
                ))
            for bound, value in (("min", min(per_class.values())), ("max", max(per_class.values()))):
                ledger.add(Quantity(
                    id=f"readout_sweeps/breadth/{panel}/{endpoint}/across_classes/{bound}",
                    claim=f"{bound} over the swept readout classes of the cells resolving above zero "
                          f"on the {panel} panel for {endpoint}",
                    family="readout_sweeps", value=float(value), unit="fit cells",
                    kind="range_bound", support_id=support, level="representation",
                    conditioning="readout_selection",
                    source_path=SWEEP_EXCERPT, source_sha256=digest,
                    source_pointer=("cells", "class", "<reduction>", endpoint, "<across classes>"),
                ))
            # Arms resolving at every seed they carry, which is the breadth
            # statement a panel count cannot make.
            arms = sorted({row["arm"] for row in scoped})
            at_all_seeds = sum(
                all(row["resolved"] for row in scoped if row["arm"] == arm and row["class"] == cls)
                and any(row["arm"] == arm and row["class"] == cls for row in scoped)
                for arm in arms for cls in sorted({row["class"] for row in scoped})
            )
            ledger.add(Quantity(
                id=f"readout_sweeps/breadth/{panel}/{endpoint}/arm_class_cells_at_every_seed",
                claim=f"arm and class combinations on the {panel} panel whose {endpoint} resolves "
                      "above zero at every seed they carry",
                family="readout_sweeps", value=float(at_all_seeds), unit="arm--class cells",
                kind="count", support_id=support, level="representation",
                conditioning="readout_selection",
                verdict="supported" if at_all_seeds else "not_detected",
                source_path=SWEEP_EXCERPT, source_sha256=digest,
                source_pointer=("cells", "class", "<reduction>", endpoint, "<every seed>"),
            ))

    # Magnitude: a ceiling over the swept classes, registered as a ceiling, with
    # the admitted maximum beside it and the difference as its own row. A reading
    # that the magnitude rose would have to contradict a row rather than infer.
    admitted = next((quantity.value for quantity in ledger.quantities
                     if quantity.id == "readout/panel/representation_increment/max"), None)
    for panel in panels:
        scoped = [row for row in rows if row["endpoint"] == "delta_spearman" and row["panel"] == panel]
        if not scoped:
            continue
        support = f"sweep_panel_{panel}"
        for axis in ("class", "pooled_depth"):
            axis_rows = [row for row in scoped if _sweep_axis(row["class"]) == axis]
            if not axis_rows:
                continue
            ceilings = {readout_class: max(row["value"] for row in axis_rows
                                           if row["class"] == readout_class)
                        for readout_class in sorted({row["class"] for row in axis_rows})}
            for bound, value in (("min", min(ceilings.values())),
                                 ("max", max(ceilings.values()))):
                ledger.add(Quantity(
                    id=f"readout_sweeps/magnitude/{panel}/{axis}/ceiling_across_classes/{bound}",
                    claim=f"{bound} over the {axis} axis's classes of the largest representation "
                          f"increment any cell of the {panel} panel reaches; this is a ceiling on "
                          "magnitude and not a measured rise",
                    family="readout_sweeps", value=float(value), unit="dimensionless Spearman",
                    kind="range_bound", support_id=support, level="representation",
                    conditioning="readout_selection",
                    source_path=SWEEP_EXCERPT, source_sha256=digest,
                    source_pointer=("cells", "class", "<reduction>", "delta_spearman", axis,
                                    "<ceiling>"),
                ))
        ceilings = {readout_class: max(row["value"] for row in scoped
                                       if row["class"] == readout_class)
                    for readout_class in sorted({row["class"] for row in scoped})}
        for bound, value in (("min", min(ceilings.values())), ("max", max(ceilings.values()))):
            ledger.add(Quantity(
                id=f"readout_sweeps/magnitude/{panel}/all_axes/ceiling_across_classes/{bound}",
                claim=f"{bound} over every swept class of the largest representation increment any "
                      f"cell of the {panel} panel reaches; this is a ceiling on magnitude and not a "
                      "measured rise",
                family="readout_sweeps", value=float(value), unit="dimensionless Spearman",
                kind="range_bound", support_id=support, level="representation",
                conditioning="readout_selection",
                source_path=SWEEP_EXCERPT, source_sha256=digest,
                source_pointer=("cells", "class", "<reduction>", "delta_spearman", "<ceiling>"),
            ))
        if admitted is not None and panel.startswith("anchor"):
            ledger.add(Quantity(
                id=f"readout_sweeps/magnitude/{panel}/ceiling_minus_admitted_maximum",
                claim="the swept classes' largest representation increment minus the admitted "
                      "panel's largest, on the same frozen states: the magnitude question stated as "
                      "a difference rather than left to be inferred from two maxima",
                family="readout_sweeps", value=float(max(ceilings.values())) - float(admitted),
                unit="dimensionless Spearman", kind="artifact_leaf", support_id=support,
                level="representation", conditioning="readout_selection",
                source_path=SWEEP_EXCERPT, source_sha256=digest,
                source_pointer=("<difference>", "cells", "class", "<ceiling>",
                                "readout/panel/representation_increment/max"),
            ))


def _sweep_arm_panel_counts(ledger: Ledger, rows: list[dict], digest: str) -> None:
    """Arm-panel cells whose increment resolves at every seed they carry.

    This is the breadth statement the reassessment licenses: how many cells
    carry resolved information, per fitted class, within one axis.
    """
    by_cell: dict[tuple[str, str, str, str], dict[object, bool]] = {}
    for row in rows:
        key = (row["arm"], row["panel"], row["class"], row["endpoint"])
        by_cell.setdefault(key, {})[row["seed"]] = row["resolved"]
    for endpoint in SWEEP_ENDPOINTS:
        for axis in ("class", "pooled_depth"):
            classes = sorted({key[2] for key in by_cell
                              if key[3] == endpoint and _sweep_axis(key[2]) == axis})
            if not classes:
                continue
            support = ledger.declare_support(
                f"sweep_arm_panel_cells_{axis}",
                f"the fitted arm-panel cells of the readout reassessment's {axis} axis, each read at "
                "every split seed its panel declares",
            )
            counts = {}
            for readout_class in classes:
                cells = [seeds for key, seeds in by_cell.items()
                         if key[2] == readout_class and key[3] == endpoint]
                counts[readout_class] = sum(1 for seeds in cells if seeds and all(seeds.values()))
                ledger.add(Quantity(
                    id=f"readout_sweeps/arm_panel/{axis}/{endpoint}/{readout_class}/resolved",
                    claim=f"arm-panel cells whose {endpoint} resolves above zero at every seed under "
                          f"the {readout_class} class of the {axis} axis",
                    family="readout_sweeps", value=float(counts[readout_class]),
                    unit="arm--panel cells", kind="count", support_id=support,
                    level="representation", conditioning="readout_selection",
                    verdict="supported" if counts[readout_class] else "not_detected",
                    source_path=SWEEP_EXCERPT, source_sha256=digest,
                    source_pointer=("cells", "class", "<reduction>", endpoint, readout_class),
                ))
                identifier = f"readout_sweeps/arm_panel/{axis}/{endpoint}/cells"
                if identifier not in {quantity.id for quantity in ledger.quantities}:
                    ledger.add(Quantity(
                        id=identifier,
                        claim=f"arm-panel cells carrying {endpoint} on the {axis} axis",
                        family="readout_sweeps", value=float(len(cells)), unit="arm--panel cells",
                        kind="support_count", support_id=support, level="representation",
                        conditioning="readout_selection",
                        source_path=SWEEP_EXCERPT, source_sha256=digest,
                        source_pointer=("cells", "class", "<reduction>", endpoint),
                    ))
            for bound, value in (("min", min(counts.values())), ("max", max(counts.values()))):
                ledger.add(Quantity(
                    id=f"readout_sweeps/arm_panel/{axis}/{endpoint}/across_classes/{bound}",
                    claim=f"{bound} over the {axis} axis's classes of the arm-panel cells whose "
                          f"{endpoint} resolves above zero at every seed",
                    family="readout_sweeps", value=float(value), unit="arm--panel cells",
                    kind="range_bound", support_id=support, level="representation",
                    conditioning="readout_selection",
                    source_path=SWEEP_EXCERPT, source_sha256=digest,
                    source_pointer=("cells", "class", "<reduction>", endpoint, "<across classes>"),
                ))


def _sweep_admitted_comparisons(ledger: Ledger, rows: list[dict], digest: str) -> None:
    """A range across classes and a comparison against the admitted class are
    different quantities, so each gets its own row and the comparison is a
    difference rather than two endpoints a reader must subtract.

    Which class reproduces the admitted readout is read from the numbers: it is
    the class whose value equals the admitted row for the same arm and seed. No
    class is asserted to be the admitted one.
    """
    admitted = {(quantity.id.split("/")[1], quantity.seed_set[0] if quantity.seed_set else None):
                quantity.value
                for quantity in ledger.quantities
                if quantity.id.startswith("readout/") and
                quantity.id.endswith("/representation_increment") and quantity.seed_set}
    if not admitted:
        ledger.gap(id="readout_sweeps/comparison/admitted_reference",
                   claim="the admitted readout's own increment, against which a class is compared",
                   family="readout_sweeps",
                   reason="no admitted readout row is in the table to compare a class against",
                   looked_in=SWEEP_EXCERPT, reachability="not_recorded")
        return
    scoped = [row for row in rows
              if row["endpoint"] == "delta_spearman" and _sweep_axis(row["class"]) == "class"]
    reproducing: dict[str, int] = {}
    for row in scoped:
        reference = admitted.get((row["arm"], row["seed"]))
        if reference is None:
            continue
        if abs(row["value"] - reference) <= 1e-9:
            reproducing[row["class"]] = reproducing.get(row["class"], 0) + 1
    if not reproducing:
        ledger.gap(id="readout_sweeps/comparison/reproducing_class",
                   claim="the swept class that reproduces the admitted readout, identified by its "
                         "values rather than asserted",
                   family="readout_sweeps",
                   reason="no swept class reproduces an admitted cell to within 1e-9",
                   looked_in=SWEEP_EXCERPT, reachability="not_recorded")
        return
    reference_class = max(reproducing, key=lambda name: reproducing[name])
    for panel in sorted({row["panel"] for row in scoped}):
        support = f"sweep_panel_{panel}"
        for seed in sorted({row["seed"] for row in scoped if row["panel"] == panel and row["seed"]}):
            cells = [row for row in scoped if row["panel"] == panel and row["seed"] == seed]
            reference = [row for row in cells if row["class"] == reference_class]
            if not reference:
                continue
            best = max(cells, key=lambda row: row["value"])
            admitted_value = max(row["value"] for row in reference)
            for name, value, claim, kind in (
                ("admitted_class_value", admitted_value,
                 f"the largest increment the admitted readout class ({reference_class}) reaches on "
                 f"the {panel} panel at split seed {seed}", "artifact_leaf"),
                ("best_class_value", best["value"],
                 f"the largest increment any swept class reaches on the {panel} panel at split seed "
                 f"{seed}, attained by {best['class']} on {best['arm']}", "artifact_leaf"),
                ("best_minus_admitted_class", best["value"] - admitted_value,
                 f"how much more than the admitted readout class the best swept class reaches on the "
                 f"{panel} panel at split seed {seed}: the comparison as one quantity, not two "
                 "endpoints to subtract", "artifact_leaf"),
            ):
                ledger.add(Quantity(
                    id=f"readout_sweeps/comparison/{panel}/{seed}/{name}",
                    claim=claim, family="readout_sweeps", value=float(value),
                    unit="dimensionless Spearman", kind=kind, support_id=support,
                    seed_set=(int(seed),), level="representation", conditioning="readout_selection",
                    source_path=SWEEP_EXCERPT, source_sha256=digest,
                    source_pointer=("cells", "class", "<comparison>", panel, str(seed), name),
                ))
        # Classes reaching less than the admitted one, which a range endpoint
        # states and a comparison does not.
        below = sorted({row["class"] for row in scoped if row["panel"] == panel
                        and row["value"] < min(
                            (other["value"] for other in scoped
                             if other["panel"] == panel and other["class"] == reference_class),
                            default=float("inf"))})
        ledger.add(Quantity(
            id=f"readout_sweeps/comparison/{panel}/classes_below_the_admitted_class",
            claim=f"swept classes reaching less on the {panel} panel than the admitted readout class "
                  "reaches at its weakest seed",
            family="readout_sweeps", value=float(len(below)), unit="readout classes", kind="count",
            support_id=support, level="representation", conditioning="readout_selection",
            verdict="not_detected" if below else "supported",
            source_path=SWEEP_EXCERPT, source_sha256=digest,
            source_pointer=("cells", "class", "<comparison>", panel, "<below admitted>"),
        ))


def _sweep_breadth_comparisons(ledger: Ledger, rows: list[dict], digest: str) -> None:
    """The same distinction for the breadth counts.

    A range over the classes and a comparison against the admitted class are
    separate rows, so a range endpoint cannot be quoted as a comparison.
    """
    admitted_resolved = {
        (quantity.id.split("/")[1], quantity.seed_set[0] if quantity.seed_set else None)
        for quantity in ledger.quantities
        if quantity.id.startswith("readout/") and
        quantity.id.endswith("/representation_increment") and quantity.seed_set and
        quantity.interval is not None and quantity.interval[0] > 0
    }
    by_cell: dict[tuple[str, str, str], dict[object, bool]] = {}
    for row in rows:
        if row["endpoint"] != "delta_spearman" or _sweep_axis(row["class"]) != "class":
            continue
        by_cell.setdefault((row["arm"], row["panel"], row["class"]), {})[row["seed"]] = row["resolved"]
    if not by_cell:
        return
    support = ledger.declare_support(
        "sweep_arm_panel_cells_class",
        "the fitted arm-panel cells of the readout reassessment's class axis, each read at every "
        "split seed its panel declares",
    )
    counts = {}
    for readout_class in sorted({key[2] for key in by_cell}):
        counts[readout_class] = sum(1 for key, seeds in by_cell.items()
                                    if key[2] == readout_class and seeds and all(seeds.values()))
    admitted_arms = len({arm for arm, seed in admitted_resolved})
    best_class = max(counts, key=lambda name: counts[name])
    for name, value, claim in (
        ("admitted_resolved_checkpoints", float(admitted_arms),
         "checkpoints whose admitted readout increment resolves above zero in at least one seed, "
         "which is the breadth the reassessment is read against"),
        ("best_class_resolved_cells", float(counts[best_class]),
         f"arm-panel cells resolving at every seed under the best swept class ({best_class})"),
        ("best_minus_admitted_breadth", float(counts[best_class]) - float(admitted_arms),
         "how many more cells the best swept class resolves than the admitted readout does "
         "checkpoints: the comparison as one quantity, and not a range endpoint"),
    ):
        ledger.add(Quantity(
            id=f"readout_sweeps/comparison/breadth/{name}",
            claim=claim, family="readout_sweeps", value=value,
            unit="arm--panel cells" if "cells" in name or "breadth" in name else "checkpoints",
            kind="count" if "resolved" in name else "artifact_leaf", support_id=support,
            seed_set=SPLIT_SEEDS, level="representation", conditioning="readout_selection",
            source_path=SWEEP_EXCERPT, source_sha256=digest,
            source_pointer=("cells", "class", "<comparison>", "breadth", name),
        ))


DEPTH_PANEL = ("results/R2/readout_depth_panel_20260925/readout_depth_panel.json")
CLASS_ADMISSION = ("results/R2/readout_class_sweep_admission_20260925/"
                   "class_sweep_admission.json")


def _sweep_depth_cells(ledger: Ledger, payload: dict, digest: str, support: str) -> None:
    """The per-block fitted increments, which sit under ``seeds`` in each cell.

    They are one level below the cell's top-level keys. An enumeration that does
    not descend there reads as an absence, which is how this family was first
    recorded as a gap; the locator now names the nested path.
    """
    depth_seen: dict[str, float] = {}
    for cell in payload["cells"]["depth"]:
        block = cell["block"]
        for seed, per_seed in sorted((block.get("seeds") or {}).items()):
            summaries = (per_seed or {}).get("summaries") or {}
            for name, value in sorted(summaries.items()):
                if not name.endswith("delta_spearman") or not isinstance(value, dict):
                    continue
                if value.get("point") is None or value.get("degenerate"):
                    continue
                axis = name.rsplit("_delta_spearman", 1)[0].rstrip("/")
                identifier = (f"readout_sweeps/depth/{block['arm']}/{block['panel']}/{seed}/{axis}")
                # gpt2-large's depth cell was re-extracted, so two files carry
                # the same (arm, panel, seed). The first is kept and the repeat
                # checked against it, as on the class axis.
                existing = depth_seen.get(identifier)
                if existing is not None:
                    if abs(existing - _point(value)) > 1e-9:
                        ledger.gap(
                            id=f"{identifier}/repeat_disagreement",
                            claim="two retained depth cells for one (arm, panel, seed, axis) that do "
                                  "not agree",
                            family="readout_sweeps",
                            reason=f"{existing!r} against {_point(value)!r} in {cell['path']}",
                            looked_in=SWEEP_EXCERPT, reachability="not_recorded",
                            searched="both files' seeds/<seed>/summaries blocks")
                    continue
                depth_seen[identifier] = _point(value)
                interval = _interval(value)
                ledger.add(Quantity(
                    id=identifier,
                    claim=f"{block['arm']} on the {block['panel']} panel at split seed {seed}, "
                          f"{axis}: the representation's increment over the matched supervised "
                          "baseline at that extraction depth",
                    family="readout_sweeps", value=_point(value), unit="dimensionless Spearman",
                    kind="estimate", support_id=support, interval=interval,
                    interval_kind=PERCENTILE_95 if interval else None,
                    resampling_unit=str(value.get("unit")) if interval else None,
                    resampling_draws=int(value["resamples"]) if interval and value.get("resamples")
                                     else None,
                    no_interval_reason=None if interval else "the cell records no interval",
                    seed_set=(int(seed),), level="representation",
                    conditioning="readout_selection",
                    source_path=SWEEP_EXCERPT, source_sha256=digest,
                    source_pointer=("cells", "depth", cell["path"], "seeds", seed, "summaries", name),
                ))


def family_depth_panel(ledger: Ledger) -> None:
    """The depth panel aggregate: the falsification tally, breadth and ceilings.

    A cross-cell tally is a quantity no per-cell file can hold, which is why it
    needed its own artifact rather than a reduction over cells.
    """
    if not ledger.artifacts.exists(DEPTH_PANEL):
        ledger.gap(id="readout_sweeps/depth_panel", claim="the depth panel aggregate",
                   family="readout_sweeps", reason="the aggregate is not staged here",
                   looked_in=DEPTH_PANEL, reachability="cluster_only",
                   searched="the declared path only")
        return
    payload = ledger.artifacts.json(DEPTH_PANEL)
    digest = ledger.artifacts.sha256(DEPTH_PANEL)
    support = ledger.declare_support(
        "readout_depth_reextraction",
        "the depth re-extraction: every transformer block of the extraction roster, refitted on the "
        "same admitted panels, seeds and fold maps as the two pooled depths",
    )
    for stratum, block in sorted(payload["falsification"].items()):
        stratum_support = ledger.declare_support(
            f"depth_stratum_{stratum}",
            f"the {stratum} interface stratum of the depth re-extraction, over "
            f"{len(block.get('arms', []))} arms and {len(block.get('panels', []))} panels",
        )
        for field, unit, kind, claim in (
            ("depth_cells", "depth cells", "support_count", "depth cells fitted in this stratum"),
            ("resolved_positive", "depth cells", "count",
             "depth cells whose increment resolves above zero"),
            ("resolved_negative", "depth cells", "count",
             "depth cells whose increment resolves below zero"),
            ("unresolved", "depth cells", "count", "depth cells whose interval contains zero"),
            ("degenerate", "depth cells", "count", "degenerate depth cells"),
        ):
            if block.get(field) is None:
                continue
            verdict = None
            if field == "resolved_positive":
                verdict = "supported" if block[field] else "not_detected"
            elif field == "unresolved":
                verdict = "unresolved"
            ledger.add(Quantity(
                id=f"readout_sweeps/depth_panel/{stratum}/{field}",
                claim=f"{claim}, in the {stratum} stratum", family="readout_sweeps",
                value=float(block[field]), unit=unit, kind=kind, support_id=stratum_support,
                seed_set=SPLIT_SEEDS, level="representation", conditioning="readout_selection",
                verdict=verdict,
                source_path=DEPTH_PANEL, source_sha256=digest,
                source_pointer=("falsification", stratum, field),
            ))
        best = block.get("best_cell") or {}
        if block.get("best_point") is not None:
            ledger.add(Quantity(
                id=f"readout_sweeps/depth_panel/{stratum}/best_point",
                claim=f"the largest increment any depth cell of the {stratum} stratum reaches, "
                      f"attained by {best.get('arm')} at depth {best.get('depth')} and itself "
                      f"{best.get('direction', 'unspecified').replace('_', ' ')}",
                family="readout_sweeps", value=float(block["best_point"]),
                unit="dimensionless Spearman", kind="estimate", support_id=stratum_support,
                interval=tuple(map(float, best["interval"])) if best.get("interval") else None,
                interval_kind=PERCENTILE_95 if best.get("interval") else None,
                resampling_unit="wild-type family at 50% identity" if best.get("interval") else None,
                resampling_draws=(_draws(best, payload) or UNRECORDED_DRAWS)
                                 if best.get("interval") else None,
                no_interval_reason=None if best.get("interval") else "the aggregate records no interval",
                seed_set=SPLIT_SEEDS, level="representation", conditioning="readout_selection",
                verdict="unresolved" if best.get("direction") == "unresolved" else None,
                source_path=DEPTH_PANEL, source_sha256=digest,
                source_pointer=("falsification", stratum, "best_point"),
            ))
    lifted = [entry for entry in payload.get("breadth", [])
              if entry.get("reading") == "boundary lifted"]
    outside = [entry for entry in lifted if entry.get("selected_at_admitted_depth") is False]
    ledger.add(Quantity(
        id="readout_sweeps/depth_panel/boundary_lifted_cells",
        claim="arm-panel cells where a depth resolves above zero at every seed while the admitted "
              "pooled pair resolves at none",
        family="readout_sweeps", value=float(len(lifted)), unit="arm--panel cells", kind="count",
        support_id=support, seed_set=SPLIT_SEEDS, level="representation",
        conditioning="readout_selection", verdict="supported" if lifted else "not_detected",
        source_path=DEPTH_PANEL, source_sha256=digest,
        source_pointer=("breadth", "<reading=boundary lifted>"),
    ))
    ledger.add(Quantity(
        id="readout_sweeps/depth_panel/boundary_lifted_outside_the_admitted_pair",
        claim="of those cells, the ones whose selected block is not a block the admitted extraction "
              "hooked; the remainder select a block the admitted extraction did hook, so for them the "
              "limit was the pooling across depths rather than the depth",
        family="readout_sweeps", value=float(len(outside)), unit="arm--panel cells", kind="count",
        support_id=support, seed_set=SPLIT_SEEDS, level="representation",
        conditioning="readout_selection", verdict="supported" if outside else "not_detected",
        source_path=DEPTH_PANEL, source_sha256=digest,
        source_pointer=("breadth", "<selected_at_admitted_depth=false>"),
    ))
    for key, block in sorted(payload.get("ceilings", {}).items()):
        for field in ("single_cell_maximum", "seed_consistent_maximum"):
            if block.get(field) is None:
                continue
            cell = block.get(field.replace("_maximum", "")) or {}
            ledger.add(Quantity(
                id=f"readout_sweeps/depth_panel/ceiling/{key}/{field}",
                claim=f"the {field.replace('_', ' ')} increment any depth cell of {key} reaches; a "
                      "ceiling on magnitude and not a measured rise",
                family="readout_sweeps", value=float(block[field]), unit="dimensionless Spearman",
                kind="range_bound", support_id=support, seed_set=SPLIT_SEEDS,
                level="representation", conditioning="readout_selection",
                source_path=DEPTH_PANEL, source_sha256=digest,
                source_pointer=("ceilings", key, field),
            ))
            if cell.get("interval") and field == "single_cell_maximum":
                ledger.add(Quantity(
                    id=f"readout_sweeps/depth_panel/ceiling/{key}/attaining_cell",
                    claim=f"the cell attaining {key}'s largest single-cell increment, "
                          f"{cell.get('arm')} on the {cell.get('axis')} axis at depth "
                          f"{cell.get('depth')}",
                    family="readout_sweeps", value=float(block[field]),
                    unit="dimensionless Spearman", kind="estimate", support_id=support,
                    interval=tuple(map(float, cell["interval"])), interval_kind=PERCENTILE_95,
                    resampling_unit="wild-type family at 50% identity",
                    resampling_draws=_draws(cell, payload) or UNRECORDED_DRAWS,
                    seed_set=SPLIT_SEEDS, level="representation", conditioning="readout_selection",
                    source_path=DEPTH_PANEL, source_sha256=digest,
                    source_pointer=("ceilings", key, "single_cell"),
                ))
    for field, unit, claim in (("n_cells", "fit cells", "fitted cells the aggregate covers"),
                               ("arms", "checkpoints", "arms the aggregate covers"),
                               ("panels", "panels", "panels the aggregate covers"),
                               ("strata", "interface strata", "strata the aggregate covers")):
        value = payload.get(field)
        if value is None:
            continue
        ledger.add(Quantity(
            id=f"readout_sweeps/depth_panel/{field}",
            claim=claim, family="readout_sweeps",
            value=float(len(value) if isinstance(value, list) else value), unit=unit,
            kind="support_count", support_id=support, level="measurement",
            source_path=DEPTH_PANEL, source_sha256=digest, source_pointer=(field,),
        ))
    worst = [entry["worst_admitted_prediction_deviation"] for entry in payload.get("reproduction", [])
             if entry.get("worst_admitted_prediction_deviation") is not None]
    if worst:
        ledger.add(Quantity(
            id="readout_sweeps/depth_panel/worst_prediction_deviation",
            claim="the worst deviation between a refitted prediction and the admitted one over every "
                  "reproduction cell",
            family="readout_sweeps", value=float(max(worst)), unit="dimensionless deviation",
            kind="artifact_leaf", support_id=support, level="measurement",
            source_path=DEPTH_PANEL, source_sha256=digest,
            source_pointer=("reproduction", "<max>", "worst_admitted_prediction_deviation"),
        ))


def family_class_admission(ledger: Ledger) -> None:
    """The class sweep's admission receipt, which resolves the repeated cells."""
    if not ledger.artifacts.exists(CLASS_ADMISSION):
        ledger.gap(id="readout_sweeps/class_admission",
                   claim="the class sweep's admission receipt", family="readout_sweeps",
                   reason="the receipt is not staged here", looked_in=CLASS_ADMISSION,
                   reachability="cluster_only", searched="the declared path only")
        return
    payload = ledger.artifacts.json(CLASS_ADMISSION)
    digest = ledger.artifacts.sha256(CLASS_ADMISSION)
    support = ledger.declare_support(
        "class_sweep_admitted_cells",
        f"the {payload['admitted_cells']} admitted class-sweep fit cells, taken from snapshot "
        f"{payload['admitted_snapshot']} where a cell appears in more than one",
    )
    for field, unit, kind, claim in (
        ("admitted_cells", "fit cells", "support_count", "admitted class-sweep fit cells"),
        ("expected_cells", "fit cells", "support_count", "fit cells the roster expects"),
        ("duplicate_cells", "fit cells", "count",
         "cells appearing in more than one snapshot, resolved to the admitted one"),
        ("worst_prediction_deviation", "dimensionless deviation", "artifact_leaf",
         "the worst deviation between a refitted prediction and the admitted one"),
        ("baseline_tolerance", "dimensionless tolerance", "constant",
         "the tolerance the admission verifies the reproducing class against"),
    ):
        value = payload.get(field)
        if value is None:
            continue
        ledger.add(Quantity(
            id=f"readout_sweeps/class_admission/{field}",
            claim=claim, family="readout_sweeps", value=float(value), unit=unit, kind=kind,
            support_id=support, level="measurement",
            source_path=CLASS_ADMISSION, source_sha256=digest, source_pointer=(field,),
        ))
    ledger.add(Quantity(
        id="readout_sweeps/class_admission/cells_only_in_superseded",
        claim="cells existing only in a superseded snapshot, which would have to be taken from it",
        family="readout_sweeps", value=float(len(payload.get("cells_only_in_superseded", []))),
        unit="fit cells", kind="count", support_id=support, level="measurement",
        verdict="not_detected",
        source_path=CLASS_ADMISSION, source_sha256=digest,
        source_pointer=("cells_only_in_superseded",),
    ))


def _sweep_depth_receipts(ledger: Ledger, payload: dict, digest: str) -> None:
    """The depth axis's reproduction receipts, and a gap for its fitted cells."""
    cells = payload["cells"]["depth"]
    if not cells:
        return
    support = ledger.declare_support(
        "readout_depth_reextraction",
        "the depth re-extraction: every transformer block of the extraction roster, refitted on the "
        "same admitted panels, seeds and fold maps as the two pooled depths",
    )
    agreements: list[float] = []
    for cell in cells:
        block = cell["block"]
        agreement = block.get("extraction_agreement") or {}
        for field, value in sorted(agreement.items()):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if not any(token in field for token in ("_l2", "difference", "_nats")):
                continue
            agreements.append(float(value))
            identifier = (f"readout_sweeps/depth/{block['arm']}/{block['panel']}/agreement/{field}")
            if identifier in {quantity.id for quantity in ledger.quantities}:
                continue
            ledger.add(Quantity(
                id=identifier,
                claim=f"{block['arm']} on the {block['panel']} panel: {field.replace('_', ' ')} "
                      "between the re-extraction and the admitted extraction",
                family="readout_sweeps", value=float(value),
                unit="dimensionless deviation", kind="artifact_leaf", support_id=support,
                level="measurement",
                source_path=SWEEP_EXCERPT, source_sha256=digest,
                source_pointer=("cells", "depth", cell["path"], "extraction_agreement", field),
            ))
    if agreements:
        ledger.add(Quantity(
            id="readout_sweeps/depth/extraction_agreement_worst",
            claim="the worst agreement between the depth re-extraction and the admitted extraction "
                  "over every cell that reports one",
            family="readout_sweeps", value=max(abs(value) for value in agreements),
            unit="dimensionless deviation", kind="artifact_leaf", support_id=support,
            level="measurement",
            source_path=SWEEP_EXCERPT, source_sha256=digest,
            source_pointer=("cells", "depth", "<reduction>", "extraction_agreement"),
        ))
    _sweep_depth_cells(ledger, payload, digest, support)
    ledger.add(Quantity(
        id="readout_sweeps/depth/cells",
        claim="fitted (arm, panel) cells of the depth re-extraction",
        family="readout_sweeps", value=float(len(cells)), unit="fit cells", kind="support_count",
        support_id=support, level="measurement",
        source_path=SWEEP_EXCERPT, source_sha256=digest,
        source_pointer=("cells", "depth"),
    ))




CONSOLIDATION_DIR = "archive/logs/shared/recomputation_20260924"
CONSOLIDATION_FILES = (
    ("class_axis_consolidation.json", None),
    ("class_axis_consolidation_residue_interactions.json", "residue_interactions"),
    ("class_axis_consolidation_remote_homology.json", "remote_homology"),
    ("class_axis_consolidation_local_context.json", "local_context"),
    ("class_axis_consolidation_crossed_controls.json", "crossed_controls"),
)


DEPTH_CONTRAST_FILES = (
    ("depth_contrast/consolidation_folding_stability.json", "folding_stability"),
    ("depth_contrast/consolidation_remote_homology.json", "remote_homology"),
    ("depth_contrast/consolidation_residue_interactions.json", "residue_interactions"),
)

#: What the crossed-controls panel's reports bind that cannot be reached, and
#: what had to be recovered by reproducing a number rather than by reading a
#: record. Both are L55's territory and neither is a quantity.
CROSSED_CONTROL_PROVENANCE = [
    ("readout_sweeps/crossed_controls/analysis_code_version",
     "the exact analysis code the crossed-controls panel's published cells ran under",
     "the published reports record an analysis_code_sha256 that is not the committed script's, "
     "because the commit creating that file postdates the run; the digest survives and the version "
     "it binds does not, so what differs between them is knowable only by its numerical effect, "
     "which is inert across all 1,080 compared nodes",
     "archive/logs/shared/crossed_controls_20260923/reports",
     "every retained crossed-control report's analysis_code_sha256 against the committed script"),
    ("readout_sweeps/crossed_controls/anchor_arm",
     "the anchor arm the crossed-controls panel's published cells ran under",
     "the published reports record no anchor_arm at all; passing progen3-3b explicitly reproduces "
     "every quantity exactly, so the default in force is established by reproduction rather than by "
     "record, which is a weaker kind of provenance than a recorded parameter",
     "archive/logs/shared/crossed_controls_20260923/reports",
     "every retained crossed-control report's top-level fields for an anchor declaration"),
]


def family_depth_contrast(ledger: Ledger) -> None:
    """The depth-at-equal-capacity contrast, which is a new measurement.

    Its representation is 512 coordinates at one depth against the published
    1,024 spread over two depths and two pooling rules, so its increments are
    not comparable with the published ones and the support says so. The finding
    lives in the comparison between the two gates, so the per-arm difference
    between the selected depth and the best admitted one is registered for each.
    """
    for name, gate in DEPTH_CONTRAST_FILES:
        path = f"{CONSOLIDATION_DIR}/{name}"
        if not ledger.artifacts.exists(path):
            ledger.gap(id=f"depth_contrast/{gate}", claim="the depth-at-equal-capacity contrast",
                       family="depth_contrast", reason="the consolidation is not staged here",
                       looked_in=path, reachability="cluster_only",
                       searched="the declared path only")
            continue
        payload = ledger.artifacts.json(path)
        digest = ledger.artifacts.sha256(path)
        measured = payload.get("arms_measured")
        if measured is None:
            measured = (payload.get("populations") or {}).get("arms_measured", len(payload["arms"]))
        support = ledger.declare_support(
            f"depth_contrast_{gate}",
            f"the {gate.replace('_', ' ')} gate's depth-at-equal-capacity contrast over "
            f"{measured} arms: {payload['coordinates']} representation coordinates at "
            f"one block, against the published {payload['admitted_published_width']} spread over two "
            "blocks and two pooling rules, so its increments are a separate measurement and not "
            "comparable with the published ones",
        )
        for field, unit, kind, claim in (
            ("arms_measured", "checkpoints", "support_count", "arms the contrast measures"),
            ("coordinates", "projection coordinates", "constant",
             "representation coordinates the contrast fits at one depth"),
            ("admitted_published_width", "projection coordinates", "constant",
             "representation coordinates the published increment spreads over two depths"),
        ):
            value = payload.get(field)
            if value is None and field == "arms_measured":
                value = measured
            if value is None:
                ledger.gap(id=f"depth_contrast/{gate}/{field}",
                           claim=f"{claim}, on the {gate} gate", family="depth_contrast",
                           reason="the consolidation does not record this field",
                           looked_in=path, reachability="not_recorded",
                           searched="every top-level field of the consolidation")
                continue
            ledger.add(Quantity(
                id=f"depth_contrast/{gate}/{field}", claim=f"{claim}, on the {gate} gate",
                family="depth_contrast", value=float(value), unit=unit, kind=kind,
                support_id=support, level="representation",
                conditioning="readout_selection",
                source_path=path, source_sha256=digest, source_pointer=(field,),
            ))
        resolved_at_every_seed = 0
        highest_unselected = 0
        for arm, block in sorted(payload["arms"].items()):
            admitted = set(block.get("admitted_depths") or [])
            if block.get("selected_resolved_seeds") == 3:
                resolved_at_every_seed += 1
            if block.get("highest_depth") is not None and \
                    block["highest_depth"] not in admitted and \
                    block["highest_depth"] != block.get("selected_depth"):
                highest_unselected += 1
            for field, unit, kind, claim in (
                ("selected_mean_point", "dimensionless increment", "artifact_leaf",
                 "the seed-mean increment at the depth the contrast selects"),
                ("best_admitted_mean_point", "dimensionless increment", "artifact_leaf",
                 "the best seed-mean increment at a depth the admitted extraction hooked"),
                ("selected_minus_best_admitted", "dimensionless increment", "artifact_leaf",
                 "the selected depth's increment minus the best admitted one: the comparison as one "
                 "quantity"),
                ("highest_mean_point", "dimensionless increment", "artifact_leaf",
                 "the highest seed-mean increment any depth reaches"),
                ("selected_depth", "block index", "constant", "the block the contrast selects"),
                ("highest_depth", "block index", "constant", "the highest-reading block"),
                ("selected_resolved_seeds", "split seeds", "count",
                 "split seeds at which the selected depth's increment resolves above zero"),
            ):
                if block.get(field) is None:
                    continue
                ledger.add(Quantity(
                    id=f"depth_contrast/{gate}/{arm}/{field}",
                    claim=f"{arm} on the {gate} gate: {claim}", family="depth_contrast",
                    value=float(block[field]), unit=unit, kind=kind, support_id=support,
                    seed_set=SPLIT_SEEDS, level="representation", conditioning="readout_selection",
                    source_path=path, source_sha256=digest,
                    source_pointer=("arms", arm, field),
                ))
        declared = payload.get("populations") or {}
        recorded = {
            "arms_resolving_at_every_seed":
                (declared.get("selected_resolves_at_all_three_seeds") or {}).get("over_arms_measured"),
            "arms_whose_highest_depth_is_unselected_and_unadmitted":
                declared.get("highest_depth_is_unselected_and_unadmitted"),
        }
        for name_, computed in (("arms_resolving_at_every_seed", resolved_at_every_seed),
                                ("arms_whose_highest_depth_is_unselected_and_unadmitted",
                                 highest_unselected)):
            stated = recorded.get(name_)
            if stated is not None and int(stated) != int(computed):
                ledger.gap(
                    id=f"depth_contrast/{gate}/{name_}/disagreement",
                    claim=f"whether {name_.replace('_', ' ')} on the {gate} gate is what the "
                          "consolidation states or what its own per-arm blocks give",
                    family="depth_contrast",
                    reason=f"the populations block states {int(stated)} and the per-arm blocks give "
                           f"{int(computed)}; the artifact's own figure is registered and the "
                           "disagreement is recorded rather than resolved here",
                    looked_in=path, reachability="not_recorded",
                    searched="the populations block against every per-arm block")
        for name_, value, claim, verdict in (
            ("arms_resolving_at_every_seed",
             recorded["arms_resolving_at_every_seed"]
             if recorded["arms_resolving_at_every_seed"] is not None else resolved_at_every_seed,
             "arms whose selected depth's increment resolves above zero at all three seeds; the "
             "differences on the other arms are between quantities that do not separate from zero",
             "supported" if resolved_at_every_seed else "not_detected"),
            ("arms_whose_highest_depth_is_unselected_and_unadmitted",
             recorded["arms_whose_highest_depth_is_unselected_and_unadmitted"]
             if recorded["arms_whose_highest_depth_is_unselected_and_unadmitted"] is not None
             else highest_unselected,
             "arms whose highest-reading depth is neither the selected block nor one the admitted "
             "extraction hooked, so the reading rests on a block no declaration picked",
             "unresolved" if highest_unselected else "not_detected"),
        ):
            ledger.add(Quantity(
                id=f"depth_contrast/{gate}/{name_}", claim=claim, family="depth_contrast",
                value=float(value), unit="checkpoints", kind="count", support_id=support,
                seed_set=SPLIT_SEEDS, level="representation", conditioning="readout_selection",
                verdict=verdict,
                source_path=path, source_sha256=digest,
                source_pointer=("arms", "<reduction>", name_),
            ))
        # Whether the quantities a difference is taken between are themselves
        # above or below zero. An ordering between two negative increments is
        # not the same statement as an ordering between two positive ones, and
        # quoting it without the sign is the same error as a zero without its
        # denominator.
        points = [seed_block["point"]
                  for arm_block in payload["arms"].values()
                  for depth_block in (arm_block.get("depths") or {}).values()
                  for seed_block in (depth_block.get("seeds") or {}).values()
                  if isinstance(seed_block, dict) and seed_block.get("point") is not None]
        if points:
            negative = sum(1 for value in points if value < 0)
            for name_, value, unit, kind, claim in (
                ("underlying_points", float(len(points)), "fitted points", "support_count",
                 "fitted depth-and-seed points behind this cohort's differences"),
                ("underlying_points_below_zero", float(negative), "fitted points", "count",
                 "of those, the points at which the representation worsens the held-out error "
                 "rather than improving it"),
                ("underlying_points_below_zero_share", 100.0 * negative / len(points), "percent",
                 "ratio",
                 "the share of this cohort's fitted points that sit below zero, which is what a "
                 "difference between two of its increments is a difference between"),
            ):
                ledger.add(Quantity(
                    id=f"depth_contrast/{gate}/{name_}", claim=claim, family="depth_contrast",
                    value=value, unit=unit, kind=kind, support_id=support, seed_set=SPLIT_SEEDS,
                    level="representation", conditioning="readout_selection",
                    source_path=path, source_sha256=digest,
                    source_pointer=("arms", "<reduction>", "depths", "seeds", "point"),
                ))
        for verdict_name in sorted({value for value in payload["verdicts"].values()}):
            ledger.add(Quantity(
                id=f"depth_contrast/{gate}/verdict/{verdict_name}",
                claim=f"arms whose selected depth is {verdict_name.replace('_', ' ')} on the {gate} "
                      "gate",
                family="depth_contrast",
                value=float(sum(1 for value in payload["verdicts"].values()
                                if value == verdict_name)),
                unit="checkpoints", kind="count", support_id=support, seed_set=SPLIT_SEEDS,
                level="representation", conditioning="readout_selection",
                source_path=path, source_sha256=digest, source_pointer=("verdicts", verdict_name),
            ))


def family_depth_contrast_controls(ledger: Ledger) -> None:
    """Each cohort's control, with the provenance that makes it its own quantity.

    Two cohorts checksum each arm's baseline predictions across its depths while
    the run happens. The third never did: its adapter recorded increments and a
    fold identity only, so its control is an equivalent recovered afterwards
    from what the run persisted. Three controls, not one, and the table refuses
    to total them.
    """
    controls: list[Quantity] = []
    for name, gate in DEPTH_CONTRAST_FILES:
        path = f"{CONSOLIDATION_DIR}/{name}"
        if not ledger.artifacts.exists(path):
            continue
        payload = ledger.artifacts.json(path)
        digest = ledger.artifacts.sha256(path)
        support = f"depth_contrast_{gate}"
        checks = payload.get("baseline_checks")
        if checks is not None:
            controls.append(ledger.add(Quantity(
                id=f"depth_contrast/{gate}/control/baseline_identity_checks",
                claim=f"baseline-identity checks the {gate} adapter computed while the run happened, "
                      "each confirming an arm's control ladder is identical across its depths",
                family="depth_contrast", value=float(checks), unit="checks",
                kind="support_count", support_id=support, seed_set=SPLIT_SEEDS,
                level="representation", conditioning="readout_selection",
                control_provenance="computed_during_the_run",
                source_path=path, source_sha256=digest, source_pointer=("baseline_checks",),
            )))
            continue
        block = payload.get("depth_invariance_control")
        if not isinstance(block, dict):
            ledger.gap(id=f"depth_contrast/{gate}/control",
                       claim=f"the control that establishes the {gate} cohort's depths share one "
                             "baseline", family="depth_contrast",
                       reason="the adapter recorded no baseline checksum and the consolidation "
                              "carries no derived equivalent",
                       looked_in=path, reachability="not_measured",
                       searched="every top-level field of the consolidation")
            continue
        for field, unit, kind, claim in (
            ("contrast_comparisons", "comparisons", "support_count",
             "contrast comparisons showing every design that omits the representation is unchanged "
             "across an arm's depths"),
            ("contrast_mismatches", "comparisons", "count",
             "of those, the comparisons that disagree"),
            ("largest_absolute_difference", "dimensionless difference", "artifact_leaf",
             "the largest absolute difference any of those comparisons shows"),
            ("fold_identity_comparisons", "comparisons", "support_count",
             "fold-identity comparisons showing every depth of an arm and seed carries one fold map"),
            ("fold_identity_mismatches", "comparisons", "count",
             "of those, the comparisons that disagree"),
        ):
            if block.get(field) is None:
                continue
            controls.append(ledger.add(Quantity(
                id=f"depth_contrast/{gate}/control/{field}",
                claim=f"{claim}, on the {gate} cohort; recovered after the fact from the retained "
                      "outputs because this adapter never computed a baseline checksum, so it is an "
                      "equivalent of the other cohorts' control and not the same measurement",
                family="depth_contrast", value=float(block[field]), unit=unit, kind=kind,
                support_id=support, seed_set=SPLIT_SEEDS, level="representation",
                conditioning="readout_selection",
                control_provenance="derived_after_the_fact",
                verdict="not_detected" if field.endswith("mismatches") else None,
                source_path=path, source_sha256=digest,
                source_pointer=("depth_invariance_control", field),
            )))
    if len({quantity.control_provenance for quantity in controls}) > 1:
        try:
            checked_sum(controls, field="control_provenance")
        except LedgerError as refusal:
            ledger.gap(
                id="depth_contrast/controls/total",
                claim="a single count of the control checks across the three depth cohorts",
                family="depth_contrast",
                reason=f"{refusal}. Two cohorts' counts were computed while the run happened and the "
                       "third's was recovered afterwards from what the run persisted, so a total "
                       "would state as one measurement what three implementations produced "
                       "differently",
                looked_in=CONSOLIDATION_DIR, reachability="not_measured",
                searched="every cohort's consolidation for a control the same implementation "
                         "computed",
            )


def family_depth_contrast_population(ledger: Ledger) -> None:
    """The cross-gate comparison, on the population it can be made over.

    An arm whose selected depth is one the admitted extraction already hooked
    contributes no comparison, so a count over all measured arms and a count
    over the arms that can be compared are different quantities and both are
    registered.
    """
    gates = {}
    for name, gate in DEPTH_CONTRAST_FILES:
        path = f"{CONSOLIDATION_DIR}/{name}"
        if ledger.artifacts.exists(path):
            gates[gate] = (ledger.artifacts.json(path), path, ledger.artifacts.sha256(path))
    if len(gates) < 2:
        return
    names = sorted(gates)
    shared = sorted(set.intersection(*(set(gates[g][0]["arms"]) for g in names)))
    population = [arm for arm in shared
                  if all(gates[g][0]["verdicts"].get(arm) != "selected_is_admitted" for g in names)]
    path, digest = gates[names[0]][1], gates[names[0]][2]
    support = ledger.declare_support(
        "depth_contrast_comparison_population",
        "the arms measured on both depth-contrast cohorts whose selected depth is not one the "
        "admitted extraction already hooked, which are the arms a cross-cohort comparison can be "
        "made over",
    )
    def difference(gate: str, arm: str) -> float:
        return float(gates[gate][0]["arms"][arm]["selected_minus_best_admitted"])
    exceeding = [arm for arm in population if all(difference(g, arm) > 0 for g in names)]
    falling = [arm for arm in population if all(difference(g, arm) < 0 for g in names)]
    cohorts = len(names)
    for name_, value, claim, verdict in (
        ("cohorts", float(cohorts), "depth-contrast cohorts compared", None),
        ("arms_measured_on_every_cohort", float(len(shared)),
         "arms measured on every depth-contrast cohort", None),
        ("comparison_population", float(len(population)),
         "arms of those whose selected depth is not one the admitted extraction hooked, so a "
         "cross-cohort comparison can be made over them", None),
        (f"exceeding_on_all_{cohorts}", float(len(exceeding)),
         f"arms of the comparison population whose selected depth exceeds the best admitted one on "
         f"all {cohorts} cohorts", "supported" if exceeding else "not_detected"),
        (f"falling_short_on_all_{cohorts}", float(len(falling)),
         f"arms of the comparison population whose selected depth falls short of the best admitted "
         f"one on all {cohorts} cohorts", "supported" if falling else "not_detected"),
    ):
        ledger.add(Quantity(
            id=f"depth_contrast/comparison/{name_}", claim=claim, family="depth_contrast",
            value=value, unit="checkpoints",
            kind="support_count" if "population" in name_ or "measured" in name_ else "count",
            support_id=support, seed_set=SPLIT_SEEDS, level="representation",
            conditioning="readout_selection", verdict=verdict,
            source_path=path, source_sha256=digest,
            source_pointer=("<cross-gate reduction>", "verdicts", name_),
        ))
    for gate in names:
        payload = gates[gate][0]
        ledger.add(Quantity(
            id=f"depth_contrast/comparison/{gate}/exceeding_both_admitted",
            claim=f"arms of the comparison population whose selected depth exceeds both admitted "
                  f"depths on the {gate} cohort alone",
            family="depth_contrast",
            value=float(sum(1 for arm in population if difference(gate, arm) > 0)),
            unit="checkpoints", kind="count", support_id=support, seed_set=SPLIT_SEEDS,
            level="representation", conditioning="readout_selection",
            source_path=gates[gate][1], source_sha256=gates[gate][2],
            source_pointer=("<cross-gate reduction>", "arms", "selected_minus_best_admitted"),
        ))
        ledger.add(Quantity(
            id=f"depth_contrast/comparison/{gate}/arms_resolving_at_every_seed",
            claim=f"arms of the comparison population whose selected depth resolves above zero at "
                  f"all three seeds on the {gate} cohort; the count over every measured arm is a "
                  "different quantity and is registered separately",
            family="depth_contrast",
            value=float(sum(1 for arm in population
                            if payload["arms"][arm]["selected_resolved_seeds"] == 3)),
            unit="checkpoints", kind="count", support_id=support, seed_set=SPLIT_SEEDS,
            level="representation", conditioning="readout_selection",
            source_path=gates[gate][1], source_sha256=gates[gate][2],
            source_pointer=("<cross-gate reduction>", "arms", "selected_resolved_seeds"),
        ))


def family_crossed_control_provenance(ledger: Ledger) -> None:
    for identifier, claim, reason, looked_in, searched in CROSSED_CONTROL_PROVENANCE:
        ledger.gap(id=identifier, claim=claim, family="readout_sweeps", reason=reason,
                   looked_in=looked_in, reachability="not_recorded", searched=searched)


def _consolidation_integrity(ledger: Ledger, gate: str, block: dict, support: str, path: str,
                             digest: str) -> None:
    """The recomputation's own integrity counts, which are per gate and not per walk."""
    for field, unit, claim in (
        ("cells_missing", "fit cells", "cells the recomputation could not reach"),
        ("published_digest_failures", "fit cells",
         "cells whose published artifact failed its own digest check"),
    ):
        value = block.get(field)
        if value is None:
            continue
        ledger.add(Quantity(
            id=f"class_axis/{gate}/{field}", claim=f"{claim}, on the {gate} gate",
            family="class_axis", value=float(len(value) if isinstance(value, list) else value),
            unit=unit, kind="count", support_id=support, level="measurement",
            verdict="not_detected",
            source_path=path, source_sha256=digest, source_pointer=(gate, field),
        ))
    check = block.get("published_digest_check")
    if isinstance(check, dict):
        for field, unit, claim in (
            ("checked", "fit cells", "published reports verified against the digest their admission "
                                     "binds before being compared"),
            ("failures", "fit cells", "published reports failing that digest check"),
        ):
            if check.get(field) is None:
                continue
            ledger.add(Quantity(
                id=f"class_axis/{gate}/published_digest_check/{field}",
                claim=f"{claim}, on the {gate} gate", family="class_axis",
                value=float(check[field]), unit=unit,
                kind="count" if field == "failures" else "support_count", support_id=support,
                level="measurement", verdict="not_detected" if field == "failures" else None,
                source_path=path, source_sha256=digest,
                source_pointer=(gate, "published_digest_check", field),
            ))
    for side, count in sorted((block.get("nodes_unpaired") or {}).items()):
        ledger.add(Quantity(
            id=f"class_axis/{gate}/nodes_unpaired/{side}",
            claim=f"summary nodes present on the {side.replace('_', ' ')} side with no counterpart to "
                  f"compare, on the {gate} gate",
            family="class_axis", value=float(count), unit="summary nodes", kind="count",
            support_id=support, level="measurement", verdict="not_detected",
            source_path=path, source_sha256=digest,
            source_pointer=(gate, "nodes_unpaired", side),
        ))


def family_class_axis_total(ledger: Ledger) -> None:
    """The programme-level line across every gate on one convention.

    Each total is registered beside the totals that give it meaning: a zero
    sign-change count over 28,611 compared nodes with 7,122 at risk of crossing
    says something a bare zero does not.
    """
    scoped = [quantity for quantity in ledger.quantities
              if quantity.family == "class_axis" and "/every_node/" in quantity.id]
    if not scoped:
        return
    gates = sorted({quantity.id.split("/")[1] for quantity in scoped})
    support = ledger.declare_support(
        "class_axis_recomputation_all_gates",
        f"every gate whose class-axis recomputation has closed --- {', '.join(gates)} --- read on one "
        "convention, the depth-agnostic walk over every compared node",
    )
    def total(suffix: str) -> float:
        return sum(quantity.value for quantity in scoped if quantity.id.endswith(suffix))
    cells = total("/every_node/cells")
    nodes = total("/every_node/nodes_compared")
    at_risk = sum(quantity.value for quantity in scoped if "/every_node/at_risk/" in quantity.id)
    sign_changes = total("/every_node/resolved_sign_changes")
    for name, value, unit, kind, claim in (
        ("gates", float(len(gates)), "gates", "support_count",
         "gates whose class-axis recomputation has closed"),
        ("cells", cells, "fit cells", "support_count", "fitted cells recompared across those gates"),
        ("nodes_compared", nodes, "summary nodes", "support_count",
         "summary nodes compared across those gates"),
        ("at_risk", at_risk, "summary nodes", "count",
         "nodes close enough to zero that a movement of the size measured here could have crossed it"),
        ("resolved_sign_changes", sign_changes, "summary nodes", "count",
         f"nodes whose resolved sign changes, out of {int(nodes)} compared across {int(cells)} cells "
         f"with {int(at_risk)} of them at risk of crossing"),
    ):
        ledger.add(Quantity(
            id=f"class_axis/all_gates/{name}", claim=claim, family="class_axis", value=value,
            unit=unit, kind=kind, support_id=support, level="measurement",
            verdict="not_detected" if name == "resolved_sign_changes" else None,
            source_path=f"{CONSOLIDATION_DIR}/class_axis_consolidation.json",
            source_sha256=ledger.artifacts.sha256(
                f"{CONSOLIDATION_DIR}/class_axis_consolidation.json"),
            source_pointer=("<sum over gates>", "every_node", name),
        ))


def family_class_axis_consolidation(ledger: Ledger) -> None:
    """The class-axis recomputation's own quantities, gate by gate.

    A count of zero is registered beside the count it is zero out of, because a
    zero over 9,801 compared nodes and a zero over nothing read identically once
    the denominator is dropped.
    """
    found = False
    for name, single in CONSOLIDATION_FILES:
        path = f"{CONSOLIDATION_DIR}/{name}"
        if not ledger.artifacts.exists(path):
            continue
        found = True
        payload = ledger.artifacts.json(path)
        digest = ledger.artifacts.sha256(path)
        gates = {single: payload} if single else payload
        for gate, block in sorted(gates.items()):
            if not isinstance(block, dict) or block.get("verdict") is None:
                continue
            walks = [("every_node", block)]
            licensed = block.get("licensed_quantities")
            if isinstance(licensed, dict):
                # The rewrite keeps the superseded scope beside the new one and
                # reproduces it exactly, so both are registered and neither is a
                # correction of the other.
                walks.append(("licensed_four", licensed))
            support = ledger.declare_support(
                f"class_axis_recomputation_{gate}",
                f"the {gate.replace('_', ' ')} gate's fitted cells, refitted under the settled "
                "readout class on the same states, cohorts, folds, seeds and resampling contract",
            )
            _consolidation_integrity(ledger, gate, block, support, path, digest)
            for walk, scoped in walks:
              nodes = scoped.get("total_nodes_compared", scoped.get("total_nodes"))
              for field, value, unit, kind, claim in (
                  ("cells", block.get("cells"), "fit cells", "support_count",
                   "fitted cells the recomputation covers"),
                  ("arms", block.get("arms"), "checkpoints", "support_count",
                   "arms the recomputation covers"),
                  ("nodes_compared", nodes, "summary nodes", "support_count",
                   "summary nodes compared between the admitted fit and the recomputation"),
                  ("threads", block.get("threads"), "BLAS threads", "constant",
                   "the BLAS thread count the recomputation pinned"),
              ):
                  if value is None:
                      continue
                  identifier = f"class_axis/{gate}/{walk}/{field}"
                  if identifier in {quantity.id for quantity in ledger.quantities}:
                      continue
                  ledger.add(Quantity(
                      id=identifier,
                      claim=f"{claim}, on the {gate} gate over the {walk.replace('_', ' ')} walk",
                      family="class_axis", value=float(value), unit=unit, kind=kind,
                      support_id=support, level="measurement",
                      source_path=path, source_sha256=digest, source_pointer=(gate, walk, field),
                  ))
              for group, count in sorted((scoped.get("summary_nodes_compared") or {}).items()):
                ledger.add(Quantity(
                    id=f"class_axis/{gate}/{walk}/nodes_compared/{group}",
                    claim=f"{group} summary nodes compared on the {gate} gate over the "
                          f"{walk.replace('_', ' ')} walk",
                    family="class_axis", value=float(count), unit="summary nodes",
                    kind="support_count", support_id=support, level="measurement",
                    source_path=path, source_sha256=digest,
                    source_pointer=(gate, "summary_nodes_compared", group),
                ))
              for group, count in sorted((scoped.get("at_risk") or {}).items()):
                ledger.add(Quantity(
                    id=f"class_axis/{gate}/{walk}/at_risk/{group}",
                    claim=f"{group} cells close enough to zero that a movement of the size measured "
                          f"here could have crossed it, on the {gate} gate: the control that makes a "
                          "zero sign-change count informative rather than silent",
                    family="class_axis", value=float(count), unit="fit cells", kind="count",
                    support_id=support, level="measurement",
                    source_path=path, source_sha256=digest,
                    source_pointer=(gate, "at_risk", group),
                ))
              for group, movement in sorted((scoped.get("max_point_change") or {}).items()):
                ledger.add(Quantity(
                    id=f"class_axis/{gate}/{walk}/max_point_change/{group}",
                    claim=f"the largest movement of a {group} point estimate between the admitted fit "
                          f"and the recomputation, on the {gate} gate",
                    family="class_axis", value=float(movement), unit="dimensionless movement",
                    kind="artifact_leaf", support_id=support, level="measurement",
                    source_path=path, source_sha256=digest,
                    source_pointer=(gate, "max_point_change", group),
                ))
              for group, movement in sorted((scoped.get("max_interval_endpoint_change") or {}).items()):
                ledger.add(Quantity(
                    id=f"class_axis/{gate}/{walk}/max_interval_endpoint_change/{group}",
                    claim=f"the largest movement of a {group} interval endpoint between the admitted "
                          f"fit and the recomputation, on the {gate} gate",
                    family="class_axis", value=float(movement), unit="dimensionless movement",
                    kind="artifact_leaf", support_id=support, level="measurement",
                    source_path=path, source_sha256=digest,
                    source_pointer=(gate, "max_interval_endpoint_change", group),
                ))
              ledger.add(Quantity(
                id=f"class_axis/{gate}/{walk}/resolved_sign_changes",
                claim=f"cells whose resolved sign changes between the admitted fit and the "
                      f"recomputation on the {gate} gate, out of "
                      + (f"{int(nodes)} summary nodes compared across "
                         if nodes else "the licensed nodes of ")
                      + f"{int(block.get('cells') or 0)} cells, with "
                      f"{sum(int(v) for v in (block.get('at_risk') or {}).values())} of them at risk "
                      "of crossing zero under a movement of the size measured here",
                family="class_axis", value=float(block["resolved_sign_changes"]), unit="fit cells",
                kind="count", support_id=support, level="measurement",
                verdict="not_detected",
                source_path=path, source_sha256=digest,
                source_pointer=(gate, "resolved_sign_changes"),
            ))
              ledger.add(Quantity(
                id=f"class_axis/{gate}/{walk}/arms_missing",
                claim=f"arms the recomputation could not cover on the {gate} gate",
                family="class_axis", value=float(len(block.get("arms_missing") or [])),
                unit="checkpoints", kind="count", support_id=support, level="measurement",
                verdict="not_detected",
                source_path=path, source_sha256=digest, source_pointer=(gate, "arms_missing"),
            ))
    if not found:
        ledger.gap(id="class_axis/consolidation",
                   claim="the class-axis recomputation's per-gate cell, node, at-risk and movement "
                         "counts and its resolved sign-change count",
                   family="class_axis", reason="no consolidation artifact is staged here",
                   looked_in=CONSOLIDATION_DIR, reachability="cluster_only",
                   searched="every file of the recomputation log directory")


#: Which recomputation each family's representation rows wait on. Assigning this
#: in one place rather than at every call site makes the release auditable and
#: makes a new representation family refuse rather than inherit a settled
#: condition by default: a row must not stop being provisional because nobody
#: named what it was waiting for.
REPRESENTATION_CONDITIONS = {
    "depth_contrast": "readout_selection",
    "readout": "readout_selection",
    "readout_sweeps": "readout_selection",
    "pairwise": "gate_representation_recomputation",
    "remote_homology": "gate_representation_recomputation",
    "local_context": "local_context_gate_recomputation",
}


def assign_representation_conditions(ledger: Ledger) -> None:
    unmapped = sorted({quantity.family for quantity in ledger.quantities
                       if quantity.level == "representation"
                       and quantity.family not in REPRESENTATION_CONDITIONS})
    if unmapped:
        raise LedgerError(
            "representation families with no declared condition: " + ", ".join(unmapped) +
            "; name what each is waiting on before it can be read as settled"
        )
    for quantity in ledger.quantities:
        if quantity.level != "representation":
            continue
        quantity.conditioning = REPRESENTATION_CONDITIONS[quantity.family]


def build(out_dir: Path = OUT_DIR) -> dict[str, object]:
    artifacts = Artifacts()
    ledger = Ledger(artifacts)
    admission = f"{READOUT_ROOT}/final_admission.json"
    strata = {arm: block["stratum"]
              for arm, block in artifacts.json(admission)["interfaces"].items()}

    ledger.declare_support("anchor_128", ANCHOR_SUPPORT)
    ledger.declare_support("uncond_census_800", CENSUS_SUPPORT)
    family_roster_and_conventions(ledger, strata)
    family_native_fitness(ledger, strata)
    family_native_paired_lineage(ledger)
    family_text_aa(ledger)
    family_readout(ledger, strata)
    family_pairwise(ledger)
    family_endpoint_reliability(ledger)
    family_higher_order(ledger)
    family_structure_contact(ledger)
    family_context(ledger)
    family_generation_census(ledger)
    family_generative_control(ledger)
    family_conditional_generation(ledger)
    family_cross_measure(ledger)
    family_structure_instrument(ledger)
    family_local_context(ledger)
    family_recomputation(ledger)
    family_design_constants(ledger)
    family_discard_magnitudes(ledger)
    family_failed_block_cost(ledger)
    family_indel_evaluation(ledger)
    family_unrecorded_constants(ledger)
    family_unrecorded_draw_counts(ledger)
    family_remote_homology(ledger)
    family_domainome_endpoint(ledger)
    family_class_admission(ledger)
    family_class_axis_consolidation(ledger)
    family_class_axis_total(ledger)
    family_depth_contrast(ledger)
    family_depth_contrast_controls(ledger)
    family_depth_contrast_population(ledger)
    family_crossed_control_provenance(ledger)
    family_readout_sweeps(ledger)
    family_depth_panel(ledger)
    family_leaf_subtrees(ledger)
    family_generator_totals(ledger)
    family_derived_shares(ledger)

    from scripts.capability.reporting.current_manuscript_quantities import add_current_quantities
    from scripts.capability.reporting.followup_quantities import add_followup_quantities
    add_current_quantities(ledger)
    add_followup_quantities(ledger)
    assign_representation_conditions(ledger)
    payload = ledger.write(out_dir, scope=(
        "Every quantity the manuscript cites that a retained artifact or gate record "
        "supports, read out of that file rather than transcribed. No model run, no cohort draw, no fit "
        "and no resampling: existing intervals are copied with their resampling unit attached."
    ))
    contract = figure_data_contract(artifacts, payload)
    contract["derived_number_table_sha256"] = hashlib.sha256(
        (out_dir / "derived-numbers.json").read_bytes()).hexdigest()
    (out_dir / "figure-data-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    return {"table": payload, "contract": contract}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="where the derived-number table and figure-data contract are written")
    options = parser.parse_args()
    result = build(options.out_dir)
    print(json.dumps({
        "quantities": result["table"]["counts"],
        "supports": len(result["table"]["supports"]),
        "figures": len(result["contract"]["figures"]),
        "figures_without_a_complete_binding": result["contract"]["figures_without_a_complete_binding"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
