#!/usr/bin/env python3
"""The two tests that decide whether the ProGen2 pair's novel-stratum positive survives.

Separating the three generation streams moved this pair's own declared family to
``design_relevant_bias: supported`` below 30% identity. Two things then have to
be asked of that positive, and neither is answerable from the run that produced
it.

1. *Does it survive its own attribution?* Every one of the 130 novel-stratum
   recognitions of ``progen2-small-mix7`` falls inside the seven Pfam families it
   was fine-tuned on, and its base arm hits those families in none of its 1,535
   novel-stratum samples. The point estimate without those rows is known to fall
   from +6.36 to +4.26; the *interval* is not. So this driver recomputes the
   pair's whole declared four-column contrast over pools from which every row the
   oracle places in one of those seven families has been removed -- from both
   arms, by one rule, in every stratum -- and reads the result against the same
   margin rule, the same per-(arm, stream) resampling floor and the same native
   band construction. If a column falls below the floor after removal it is
   recorded unresolved with its counts: the floor is not lowered to keep a
   column, because a column kept below its floor is not evidence.

2. *The control the pair never had.* The explicit-statistics control was declared
   over the primary pairs only. Extending it is an amendment, declared as one in
   :data:`ota.SECONDARY_CONTROL_AMENDMENT` and seeded and sized in that module
   beside the families it adds, and run here with the rule, the statistic, the
   top quarter, the ranking scope, the floor, the nested draws and the band
   construction identical to the three primary pairs. Nothing is tuned to this
   pair.

Both tests read the stream-separated table of
``run_stream_endpoints.py`` -- the frozen 2026-11-11 table builder, split by the
recovered labels -- and call the analysis module's own endpoint functions. The
primary pairs are present in the table and are never read from it here: this
driver reports the secondary pair only, and the attribution filter touches that
pair's two arms alone.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/capability/generation"))

import analyse_objective_tuned as aot  # noqa: E402

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation import generative_control as gc  # noqa: E402
from src.capability.generation import objective_tuned_analysis as ota  # noqa: E402
from src.capability.generation import objective_tuned_arms as ot  # noqa: E402

HERE = Path(__file__).resolve().parent


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"{path} cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The stream-separated run's own table builder and table emitters, reused so
#: that this driver assembles no second table and prints no second shape.
streamed = _load(HERE / "run_stream_endpoints.py", "run_stream_endpoints")

#: The seven fine-tuning families, taken from the arm declaration that names
#: them rather than listed again here.
SEVEN: tuple[str, ...] = tuple(sorted(set(
    re.findall(r"PF\d{5}", ot.OBJECTIVE_PFAM_SEVEN_FAMILIES))))

ATTRIBUTION_RULE = (
    "every folded row the Pfam oracle places in at least one of the seven "
    "families progen2-small-mix7 was fine-tuned on is removed, from both arms of "
    "the pair and in every stratum, under one rule. The declared four-column "
    "family, the margin rule, the per-(arm, stream) resampling floor, the nested "
    "draws and the native band are unchanged; a column that falls below the "
    "floor after removal is unresolved with its counts, not rescued"
)


def read_annotations(path: Path, arms: set[str]) -> dict[str, list[str]]:
    """Per-digest Pfam families from the oracle's own sidecar, for these arms."""

    out: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("arm") not in arms:
            continue
        digest = str(row["sequence_sha256"])
        if row.get("profile_search_status") != "searched":
            raise SystemExit(f"{digest}: the oracle records "
                             f"{row.get('profile_search_status')!r}, so whether it hits a "
                             "fine-tuning family is not measured and it cannot be classified")
        if digest in out:
            raise SystemExit(f"{path}: {digest} is annotated twice")
        out[digest] = [str(family) for family in (row.get("pfam_families") or [])]
    return out


def remove_in_seven(table: dict[str, Any], arms: set[str],
                    annotations: dict[str, list[str]]) -> dict[str, Any]:
    """The same table with the pair's in-seven-families rows removed."""

    pools: dict[Any, dict[str, Any]] = {}
    for key, pool in table["pools"].items():
        if pool["arm"] not in arms:
            pools[key] = pool
            continue
        kept = []
        for row in pool["rows"]:
            found = annotations.get(row["digest"])
            if found is None:
                raise SystemExit(f"{row['digest']} ({pool['arm']}) carries no oracle annotation, "
                                 "so whether it sits in a fine-tuning family is unknown and it is "
                                 "neither kept nor dropped by guess")
            if not set(found) & set(SEVEN):
                kept.append(row)
        pools[key] = {**pool, "rows": kept}
    return {**table, "pools": pools}


def exclusion_rows(full: dict[str, Any], corrected: dict[str, Any],
                   arms: set[str]) -> list[dict[str, Any]]:
    """Per (arm, stream, stratum): units before, removed, and left against the floor."""

    out = []
    for (arm, stream), pool in sorted(full["pools"].items()):
        if arm not in arms:
            continue
        after = {row["digest"] for row in corrected["pools"][(arm, stream)]["rows"]}
        for stratum in (*ota.STRATA, "all_strata"):
            members = [row for row in pool["rows"]
                       if stratum == "all_strata" or row["stratum"] == stratum]
            left = sum(1 for row in members if row["digest"] in after)
            out.append({
                "arm": arm, "stream": stream, "stratum": stratum,
                "units_before": len(members), "units_in_seven_families": len(members) - left,
                "units_left": left, "resampling_floor": ota.MIN_UNITS_PER_POOL,
                "below_floor_after_removal": left < ota.MIN_UNITS_PER_POOL,
                "thin_under_30_after_removal": left < 30,
            })
    return out


def contrast_comparison(full: dict[str, Any], corrected: dict[str, Any]) -> list[dict[str, Any]]:
    """The declared contrast per stratum, with and without the fine-tuning families."""

    out = []
    for key, column in full["bands"]["columns"].items():
        pair, stratum = key.split("|")
        other = corrected["bands"]["columns"][key]
        cells = (full["per_pair"][pair]["per_stratum"][stratum],
                 corrected["per_pair"][pair]["per_stratum"][stratum])
        row: dict[str, Any] = {"pair": pair, "stratum": stratum}
        for name, band, entry, cell in (("full", full, column, cells[0]),
                                        ("corrected", corrected, other, cells[1])):
            n_base, base_streams, base_min = streamed.units(entry["units"]["base"])
            n_tuned, tuned_streams, tuned_min = streamed.units(entry["units"]["tuned"])
            interval = entry.get("interval") or [None, None]
            differences, exceeding = streamed.streams_of(cell)
            row.update({
                f"{name}_status": entry["status"],
                f"{name}_n_base": n_base, f"{name}_n_tuned": n_tuned,
                f"{name}_base_units_by_stream": base_streams,
                f"{name}_tuned_units_by_stream": tuned_streams,
                f"{name}_thin_under_30": streamed.thin(base_min, tuned_min),
                f"{name}_base_mean_plddt": entry.get("base_mean"),
                f"{name}_tuned_mean_plddt": entry.get("tuned_mean"),
                f"{name}_diff_plddt": entry.get("point"),
                f"{name}_band_lo": interval[0], f"{name}_band_hi": interval[1],
                f"{name}_critical_value": band["bands"].get("critical_value"),
                f"{name}_margin": (cell.get("margin") or {}).get("margin"),
                f"{name}_stream_differences": differences,
                f"{name}_streams_exceeding_margin": exceeding,
                f"{name}_finding": cell["finding"],
                f"{name}_reason": entry.get("reason") or cell.get("reason"),
            })
        out.append(row)
    return sorted(out, key=lambda row: (row["pair"], row["stratum"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--neighbour", type=Path, required=True)
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--fold-structures", type=Path, default=None)
    parser.add_argument("--stream-labels", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True,
                        help="the Pfam oracle's own per-sequence sidecar, family_recognition.jsonl")
    parser.add_argument("--kmer-background", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if len(SEVEN) != 7:
        raise SystemExit(f"the arm declaration names {SEVEN}, not seven fine-tuning families")
    pair = ota.SECONDARY_PAIRS[0]
    arms = set(pair.arms)

    one_stream, worst = streamed.pooled.build_table(args.neighbour, args.composition, args.cohort,
                                                    args.fold_structures)
    labels = streamed.read_labels(args.stream_labels)
    annotations = read_annotations(args.annotations, arms)
    full = streamed.split_streams(one_stream, labels)
    corrected = streamed.split_streams(remove_in_seven(one_stream, arms, annotations), labels)
    counts = gc.load_kmer_counts(args.kmer_background, aot.KMER_ORDERS)

    # Test 1: the pair's declared family, with and without the fine-tuning families.
    secondary = aot.stratified_contrast(full, ota.SECONDARY_PAIRS, ota.SECONDARY_FAMILY)
    attributed = aot.stratified_contrast(corrected, ota.SECONDARY_PAIRS, ota.SECONDARY_FAMILY)

    # Test 2: the three declared rules over this pair, under the 2026-11-12
    # amendment's own families. The full pools, because the control asks what an
    # explicit rule over the base pool reproduces of the gain as measured.
    control = aot.explicit_statistics_control(full, ota.SECONDARY_PAIRS, counts, secondary,
                                              ota.SECONDARY_CONTROL_FAMILIES)

    record = {
        "schema": ota.SCHEMA,
        "status": "complete",
        "run": args.out.name,
        "registration": "docs/objective-alignment.md, declared 2026-11-09, amended 2026-11-10/11 "
                        "and 2026-11-12",
        "campaign": ot.CAMPAIGN,
        "pair": {"base": pair.base, "tuned": pair.tuned, "family": ota.SECONDARY_FAMILY,
                 "objective": pair.objective},
        "reads": "the stream-separated table of run_stream_endpoints.py, over the frozen "
                 "2026-11-11 table builder. The primary pairs are in the table and are not read",
        "seven_finetuning_families": list(SEVEN),
        "attribution_rule": ATTRIBUTION_RULE,
        "control_amendment": ota.SECONDARY_CONTROL_AMENDMENT,
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in
                   (("neighbour", args.neighbour), ("composition", args.composition),
                    ("cohort", args.cohort), ("stream_labels", args.stream_labels),
                    ("annotations", args.annotations),
                    *((("fold_structures", args.fold_structures),)
                      if args.fold_structures is not None else ()))
                   },
        "cross_table_agreement": {"columns": worst},
        "kmer_background": {"path": str(args.kmer_background),
                            "orders_read": list(aot.KMER_ORDERS)},
        "cells": {f"{pool['arm']}|s{pool['stream']}": pool["counts"]
                  for pool in sorted(full["pools"].values(),
                                     key=lambda pool: (pool["arm"], pool["stream"]))
                  if pool["arm"] in arms},
        "cells_after_removal": {f"{pool['arm']}|s{pool['stream']}": pool["counts"]
                                for pool in sorted(corrected["pools"].values(),
                                                   key=lambda pool: (pool["arm"], pool["stream"]))
                                if pool["arm"] in arms},
        "secondary_pair": secondary,
        "secondary_pair_without_the_finetuning_families": attributed,
        "explicit_statistics_control_secondary": control,
        "finding_states": ota.FINDING_STATES,
        "bounds": ota.BOUNDS,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "progen2_attribution.json", record)
    tables = args.out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    streamed.write_tsv(tables / "attribution_corrected_contrast.tsv",
                       contrast_comparison(secondary, attributed))
    streamed.write_tsv(tables / "attribution_exclusion_accounting.tsv",
                       exclusion_rows(full, corrected, arms))
    streamed.write_tsv(tables / "explicit_statistics_control_secondary.tsv",
                       streamed.control_rows(control))
    novel = ota.NOVEL_STRATUM
    decisive = attributed["bands"]["columns"][f"{pair.tuned}|{novel}"]
    cell = attributed["per_pair"][pair.tuned]["per_stratum"][novel]
    print(json.dumps({
        "attribution": {
            "stratum": novel,
            "status": decisive["status"],
            "point": decisive.get("point"),
            "interval": decisive.get("interval"),
            "margin": (cell.get("margin") or {}).get("margin"),
            "finding": cell["finding"],
            "design_relevant_bias":
                attributed["per_pair"][pair.tuned]["pair_decision"]["design_relevant_bias"],
            "resolved_family_size": attributed["bands"]["resolved_family_size"],
            "critical_value": attributed["bands"].get("critical_value"),
        },
        "control": {rule: {"control_status": block["bands"]["control_status"],
                           "resolved": block["bands"]["resolved_family_size"],
                           "family": block["family"]}
                    for rule, block in control["rules"].items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
