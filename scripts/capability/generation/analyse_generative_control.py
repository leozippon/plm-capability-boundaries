#!/usr/bin/env python3
"""Endpoints and verdicts for the generation-and-control gate.

Reads only completed artefacts: the frozen generation ledgers, the matched
control cohorts and their oracle tables, the competence receipts, and the
already reported structure contrasts. It runs no model, folds nothing and makes
no draw of its own beyond the declared bootstrap.

Three endpoints, reported separately because their instruments license
different things.

``any_family``       Does the oracle assign any curated Pfam-A family to the
                     product, at the release's gathering thresholds? Per-arm
                     licensed: the instrument is deterministic in the sequence
                     and identical across arms.

``complete_domain``  Does one assigned family's best domain instance cover at
                     least the declared fraction of that profile's model
                     length? The same oracle call read on the profile side.

``structure``        The paired within-sequence CA-pLDDT contrast, passed
                     through from the reconciled structure instrument together
                     with its calibration scope. Reported as unresolved on
                     every arm whose calibration is inherited rather than
                     attained in its own sequence regime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.capability.generation import generative_control as gc  # noqa: E402

#: Reference-identity bands to the searched corpus snapshot. Declared before the
#: stratified reading. ``no_reported_alignment`` is its own stratum and is not
#: an observed zero-percent identity.
IDENTITY_BANDS: tuple[tuple[str, float, float], ...] = (
    ("under_30", 0.0, 30.0),
    ("30_to_50", 30.0, 50.0),
    ("50_to_70", 50.0, 70.0),
    ("70_and_over", 70.0, 100.0001),
)

CONTROL_COHORTS: tuple[str, ...] = (
    "shuffle", "markov_0", "markov_2", "markov_4", "fragment", "hydropathy", "natural",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def band_of(identity: float | None, status: str | None) -> str:
    """The declared stratum of one attempt's distance to the searched corpus.

    Four states are kept apart because collapsing any two of them would
    manufacture a result. An attempt with no reported alignment is its own
    stratum and is **not** an observed zero-percent identity; an attempt that
    was never searched, because its output carried nothing searchable, is
    neither of those and is not a distance measurement at all; and a missing
    status is a gap in the ledger rather than a distance.
    """

    if identity is None:
        if status is None:
            return "reference_search_missing"
        if "not_searched" in status:
            return "not_searched"
        return "no_reported_alignment"
    for name, low, high in IDENTITY_BANDS:
        if low <= float(identity) < high:
            return name
    return "70_and_over"


def endpoint_flags(oracle: dict, key: str, threshold: float) -> tuple[bool, bool, float | None]:
    """``(any_family, complete_domain, best_coverage)`` for one cohort sequence."""

    entry = oracle.get(key)
    if entry is None:
        return False, False, None
    coverage = entry.get("best_profile_coverage")
    return (bool(entry["families"]),
            coverage is not None and float(coverage) >= threshold,
            None if coverage is None else float(coverage))


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "median": None, "q25": None, "q75": None}
    array = np.asarray(values, dtype=float)
    return {"n": len(values), "median": float(np.median(array)),
            "q25": float(np.quantile(array, 0.25)),
            "q75": float(np.quantile(array, 0.75))}


def verdict(contrasts: dict[str, dict], qualified: set[str]) -> dict:
    """The gate verdict for one endpoint on one arm.

    A contrast against an unqualified generator is reported and excluded from
    the verdict, because a control that does not reproduce the statistics it
    declares licenses nothing whichever way its contrast points. ``natural`` is
    a reference rather than a control and is likewise excluded: a model that
    does not beat real proteins has not failed this gate.
    """

    decisive = [name for name in CONTROL_COHORTS
                if name != "natural" and name in qualified and name in contrasts]
    beaten: list[str] = []
    not_beaten: list[str] = []
    undetermined: list[str] = []
    for name in decisive:
        interval = contrasts[name].get("ci97_5")
        if interval is None:
            undetermined.append(name)
        elif interval[0] > 0:
            beaten.append(name)
        else:
            not_beaten.append(name)
    if undetermined:
        label = "unresolved_interval_not_estimable"
    elif not decisive:
        label = "unresolved_no_qualified_control"
    elif not not_beaten:
        label = "satisfied_beyond_every_qualified_matched_generator"
    elif beaten:
        label = "satisfied_beyond_some_but_not_all_qualified_matched_generators"
    else:
        label = "not_detected_beyond_any_qualified_matched_generator"
    return {"label": label, "beaten": beaten, "not_beaten": not_beaten,
            "interval_not_estimable": undetermined,
            "decisive_controls": decisive,
            "excluded_unqualified": [name for name in CONTROL_COHORTS
                                     if name not in qualified],
            "excluded_as_reference": ["natural"]}


def analyse_cell(records: list[dict], oracle: dict, qualified: set[str],
                 amended: set[str], threshold: float) -> dict:
    groups = [record["near_duplicate_group"] for record in records]
    flags: dict[str, dict[str, list]] = {}
    for cohort in ("generated", *CONTROL_COHORTS):
        any_family: list[bool] = []
        complete: list[bool] = []
        coverage: list[float] = []
        families: Counter = Counter()
        for record in records:
            key = f"{record['cell']}|{cohort}|{record['attempt_id']}"
            hit, full, best = endpoint_flags(oracle, key, threshold)
            any_family.append(hit)
            complete.append(full)
            if best is not None:
                coverage.append(best)
            entry = oracle.get(key)
            if entry:
                families.update(entry["families"])
        flags[cohort] = {"any_family": any_family, "complete_domain": complete,
                         "coverage": coverage, "families": families}

    replay_disagreements = sum(
        1 for record, hit in zip(records, flags["generated"]["any_family"])
        if bool(record["ledger_any_profile_hit"]) != bool(hit))

    out: dict = {
        "cell": records[0]["cell"], "arm": records[0]["arm"],
        "condition": records[0]["condition"], "campaign": records[0]["campaign"],
        "n_attempts": len(records),
        "n_searchable": sum(1 for record in records if record["parent_searchable"]),
        "n_clusters": len(set(groups)),
        "instrument_replay_disagreements_against_the_frozen_ledger": replay_disagreements,
        "complete_domain_coverage_threshold": threshold,
        "endpoints": {},
    }
    for endpoint in ("any_family", "complete_domain"):
        model = flags["generated"][endpoint]
        successes = int(sum(model))
        lower, upper = gc.wilson_interval(successes, len(records))
        block: dict = {
            "model_successes": successes, "model_denominator": len(records),
            "model_rate": successes / len(records),
            "model_wilson95": [lower, upper],
            "denominator_rule": ("every attempt of the frozen ledger, including the "
                                 "attempts whose output no oracle could search and the "
                                 "attempts the oracle assigned no family"),
            "controls": {},
        }
        for cohort in CONTROL_COHORTS:
            control = flags[cohort][endpoint]
            contrast = gc.paired_rate_contrast(model, control, groups)
            contrast["control_successes"] = int(sum(control))
            contrast["qualified"] = cohort in qualified
            block["controls"][cohort] = contrast
        block["verdict"] = verdict(block["controls"], qualified)
        block["verdict_under_the_scale_free_amendment"] = verdict(block["controls"], amended)
        out["endpoints"][endpoint] = block

    out["best_profile_coverage_distribution"] = {
        cohort: quantiles(flags[cohort]["coverage"]) for cohort in ("generated", *CONTROL_COHORTS)}
    out["distinct_families"] = {
        cohort: len(flags[cohort]["families"]) for cohort in ("generated", *CONTROL_COHORTS)}

    strata: dict[str, dict] = {}
    bands = [band_of(record["reference_identity"], record["reference_search_status"])
             for record in records]
    for band in sorted(set(bands)):
        mask = [index for index, value in enumerate(bands) if value == band]
        stratum: dict = {"n_attempts": len(mask)}
        for endpoint in ("any_family", "complete_domain"):
            model = [flags["generated"][endpoint][index] for index in mask]
            stratum[endpoint] = {
                "model_successes": int(sum(model)),
                "model_rate": sum(model) / len(mask),
                "model_wilson95": list(gc.wilson_interval(int(sum(model)), len(mask))),
                "controls": {},
            }
            for cohort in CONTROL_COHORTS:
                control = [flags[cohort][endpoint][index] for index in mask]
                stratum[endpoint]["controls"][cohort] = gc.paired_rate_contrast(
                    model, control, [groups[index] for index in mask])
        strata[band] = stratum
    out["identity_strata"] = strata
    return out


def structure_passthrough(paths: list[Path]) -> dict:
    """The reported paired CA-pLDDT contrasts, with each arm's calibration scope.

    Nothing is recomputed. The narrowed licence on the structure instrument is
    applied here as a label: an arm whose natural-control calibration was
    attained on another arm's panel reports the contrast and an
    ``unresolved_calibration_inherited`` verdict, because a positive contrast
    there is evidence that the predictor responds to residue order in that arm's
    outputs and not evidence that it was shown to separate natural from shuffled
    sequences in that arm's own class and length regime.
    """

    cells: list[dict] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for arm, record in (payload.get("arms") or {}).items():
            scope = record.get("calibration_scope")
            inherited = scope is not None
            cells.append({
                "arm": arm, "phase": payload.get("phase"),
                "source": str(path),
                "unit": record.get("unit"),
                "n_units": record.get("n_classes") or record.get("n_clusters"),
                "mean_ca_plddt_delta": record.get("mean"),
                "ci95": record.get("ci95"), "ci97_5": record.get("ci97_5"),
                "calibration_attained_flag": record.get("calibration_attained"),
                "calibration_scope": scope,
                "reported_interpretation": record.get("interpretation"),
                "verdict": ("unresolved_calibration_inherited_from_a_two_arm_panel"
                            if inherited else
                            "licensed_paired_within_sequence_contrast_on_own_controls"),
                "generation_seed_uncertainty": (
                    "absent: the interval resamples the structure subsample's units "
                    "with the 800-attempt generation ledger held fixed"),
                "per_row_repeat_noise_plddt_points": {"median": 0.081, "maximum": 0.929,
                                                      "n_rows": 8},
            })
    return {"cells": cells,
            "licence": ("the licensed quantity is the paired within-sequence contrast in "
                        "pLDDT points; predictor confidence is not observed folding, "
                        "expression or function"),
            "n_inherited": sum(1 for cell in cells if cell["calibration_scope"]),
            "n_own_controls": sum(1 for cell in cells if not cell["calibration_scope"])}


def write_report(result: dict, path: Path) -> None:
    """A compact per-arm table beside the machine-readable record."""

    lines = ["# Generation-and-control gate: measured endpoints", ""]
    lines.append(f"Seed {result['seed']}, {result['resamples']} resamples. "
                 f"Qualified controls (pre-declared): "
                 f"{', '.join(result['qualified_cohorts'])}. "
                 f"Under the amendment: "
                 f"{', '.join(result['qualified_cohorts_under_the_scale_free_amendment'])}.")
    lines.append("")
    for endpoint in ("any_family", "complete_domain"):
        lines += [f"## {endpoint}", "",
                  "| arm | condition | n | model rate [Wilson 95%] | "
                  + " | ".join(f"− {cohort}" for cohort in CONTROL_COHORTS)
                  + " | verdict |",
                  "| --- | --- | ---: | --- | " + " | ".join("---" for _ in CONTROL_COHORTS)
                  + " | --- |"]
        for cell in result["cells"]:
            block = cell["endpoints"][endpoint]
            cells = []
            for cohort in CONTROL_COHORTS:
                contrast = block["controls"][cohort]
                interval = contrast["ci97_5"]
                text = f"{contrast['difference']:+.4f}"
                if interval:
                    text += f" [{interval[0]:+.4f}, {interval[1]:+.4f}]"
                cells.append(text)
            lines.append(
                f"| {cell['arm']} | {cell['condition']} | {cell['n_attempts']} | "
                f"{block['model_rate']:.4f} [{block['model_wilson95'][0]:.4f}, "
                f"{block['model_wilson95'][1]:.4f}] | " + " | ".join(cells)
                + f" | {block['verdict']['label']} |")
        lines.append("")
    lines += ["## Structure endpoint (passed through)", "",
              "| arm | phase | unit | n | delta mean CA-pLDDT | 97.5% interval | verdict |",
              "| --- | --- | --- | ---: | ---: | --- | --- |"]
    for cell in result["structure_endpoint"]["cells"]:
        interval = cell["ci97_5"]
        delta = cell["mean_ca_plddt_delta"]
        lines.append(
            f"| {cell['arm']} | {cell['phase']} | {cell['unit']} | {cell['n_units']} | "
            + (f"{delta:+.3f}" if delta is not None else "not measured") + " | "
            + (f"[{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval else "not estimable")
            + f" | {cell['verdict']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--structure-analysis", type=Path, nargs="*", default=[])
    parser.add_argument("--ceiling", type=Path,
                        help="the profile-sampler ceiling record, written by the "
                             "stage's own ceiling phase in its own workspace")
    parser.add_argument("--coverage-threshold", type=float,
                        default=gc.COMPLETE_DOMAIN_COVERAGE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    qualification = json.loads((workspace / "qualification.json").read_text(encoding="utf-8"))
    qualified = set(qualification["qualified_cohorts"])
    amended = set(qualification.get("qualified_cohorts_under_the_scale_free_amendment",
                                    qualification["qualified_cohorts"]))
    oracle = gc.collect_oracle(workspace)

    cells = []
    for path in sorted((workspace / "cells").glob("*.jsonl")):
        records = read_jsonl(path)
        cells.append(analyse_cell(records, oracle, qualified, amended,
                                  args.coverage_threshold))

    ceiling_path = Path(args.ceiling) if args.ceiling else workspace / "ceiling.json"
    result = {
        "schema": "d1_gate_generative_control_v1",
        "seed": gc.GATE_SEED, "resamples": gc.RESAMPLES,
        "qualified_cohorts": sorted(qualified),
        "unqualified_cohorts": qualification["unqualified_cohorts"],
        "qualified_cohorts_under_the_scale_free_amendment": sorted(amended),
        "amendment": qualification.get("amendment"),
        "order_profile_nats_per_residue": qualification["order_profile_nats_per_residue"],
        "competence_receipts": qualification["receipts"],
        "oracle": {
            "instrument": "HMMER hmmscan against the staged Pfam-A at --cut_ga",
            "read": "sequence table for family assignment, domain table for profile coverage",
            "noise": "deterministic in the sequence; no seed and no repeat spread",
        },
        "cells": cells,
        "profile_sampler_ceiling": (json.loads(ceiling_path.read_text(encoding="utf-8"))
                                    if ceiling_path.is_file() else None),
        "structure_endpoint": structure_passthrough(list(args.structure_analysis)),
    }
    inputs = [workspace / "qualification.json", workspace / "build_manifest.json",
              ceiling_path, *sorted(args.structure_analysis)]
    result["input_sha256"] = {
        str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in inputs if Path(path).is_file()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                        encoding="utf-8")
    write_report(result, args.out.with_suffix(".md"))
    print(json.dumps({
        "cells": len(cells), "qualified": sorted(qualified),
        "unqualified": qualification["unqualified_cohorts"],
        "out": str(args.out),
        "replay_disagreements": sum(
            cell["instrument_replay_disagreements_against_the_frozen_ledger"]
            for cell in cells),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
