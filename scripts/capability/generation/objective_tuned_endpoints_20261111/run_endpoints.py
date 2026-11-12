#!/usr/bin/env python3
"""Run the declared endpoints of ``scripts/capability/generation/analyse_objective_tuned.py``
on the eighteen-shard production fold set.

Why this driver exists, and what it is not
------------------------------------------
The analysis module's own entry point joins the campaign's sixty cell ledgers,
the fold predictions and the novelty-stage records. Only three of those four
things are on Compute: the cell ledgers and the per-structure fold records live
in the cluster tree, and what was pulled from it is the per-structure table
(arm, sequence digest, pLDDT, pTM, composition) plus the nearest-natural search
over the same digests and the frozen 47,778-sequence fold cohort that carries
the sequences. So the module's ``read_campaign`` / ``read_structure`` /
``read_novelty`` readers cannot be fed, and this driver assembles the one table
they would have produced and calls the module's endpoint functions unchanged.

Every estimator, family, seed, margin, floor, filter rule and finding state is
the module's own: this file computes no statistic. It is a reader, not a second
implementation of the analysis.

The departures the missing ledgers force, declared here rather than hidden
-------------------------------------------------------------------------
1. One pooled stream per arm. The per-structure table carries no stream label,
   so the pools are read as a single stream. The declared three-stream
   conjunction therefore cannot be evaluated at all, and no "design-relevant
   bias" verdict can be reached from this cohort whatever the numbers say. The
   resample is a within-pool bootstrap at the declared draws, seed and
   statistic, and the control's ranking scope becomes within-arm rather than
   within (arm, stream).
2. The denominator is the folded eighteen-shard structure, not the retained
   attempt. The attempt ledgers are not here, so empty and duplicate attempts
   cannot be counted; one row is one folded sequence. The eighteen shards are
   ``sha256(sequence) % 68 in {0..17}``, which is independent of arm, so the
   declared four-to-one base-to-tuned budget survives in the sample and the
   control's top quarter stays size-matched to the tuned pool.
3. ``stop_status`` is recovered from the table's ``capped`` flag under the
   lane's own vocabulary: capped rows are ``budget_censored``, the rest
   ``native_terminal``. Nothing downstream of the three endpoints reads it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/Data/lzp/plm-capability-boundaries")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/capability/generation"))

import analyse_objective_tuned as aot  # noqa: E402

from src.capability.context.homology import assign_stratum  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation import generative_control as gc  # noqa: E402
from src.capability.generation import objective_tuned_analysis as ota  # noqa: E402
from src.capability.generation import objective_tuned_arms as ot  # noqa: E402
from src.capability.generation.structure_evidence import sequence_digest  # noqa: E402

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
STREAM = 0
CELL = "eighteen_shard_18of68"

#: Columns the composition table and the nearest-neighbour table must agree on
#: exactly, because they are two joins of one fold and a disagreement would mean
#: the two tables describe different structures.
SHARED_EXACT = ("arm", "L", "capped", "shard_index_of_68", "diffusion_sample_index")

#: The float columns the two tables carry at different printed precision -- the
#: neighbour table at five significant digits, the composition table at five
#: decimals -- compared within a tolerance and with the realised maximum
#: recorded. The tolerance is formatting slack, orders of magnitude below the
#: declared fold-reproduction tolerance of 2.824 pLDDT points.
SHARED_FLOAT = ("plddt", "ptm", "zlib_ratio", "distinct_4mer_frac")
PRINT_TOLERANCE = 1e-2

#: pLDDT is read from the composition table, which prints five decimals, and is
#: checked against the fold emitter's own per-structure table beside it.
PLDDT_COLUMN = "mean_ca_plddt_0_100"

DEPARTURES = {
    "streams": "one pooled stream per arm; the per-structure table carries no stream label, so "
               "the declared three-stream conjunction cannot be evaluated and no "
               "design-relevant-bias verdict is reachable from this cohort. The resample is a "
               "within-pool bootstrap at the module's declared draws, seed rule and statistic",
    "denominator": "one folded eighteen-shard structure per row, not one retained attempt: the "
                   "cell ledgers are not on this host, so empty and duplicate attempts cannot be "
                   "counted. Rates are over the folded sample",
    "support": "the eighteen shards are sha256(sequence) %% 68 in {0..17}, independent of arm, so "
               "the declared four-to-one base-to-tuned budget survives in the sample",
    "stop_status": "recovered from the table's capped flag under the lane's vocabulary: capped is "
                   "budget_censored, the rest native_terminal",
}


def read_tsv(path: Path, *, comment: bool = False) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        lines = [line for line in handle if not (comment and line.startswith("#"))]
    return list(csv.DictReader(lines, delimiter="\t"))


def build_table(neighbour: Path, composition: Path, cohort: Path,
                structures: Path | None = None) -> tuple[dict[str, Any], dict[str, float]]:
    """The one table the module's own readers would have produced, from what is here.

    Returns the table and the realised maximum disagreement per cross-checked
    float column, so the printed-precision slack is reported rather than assumed.
    """

    rows = read_tsv(neighbour)
    comp = {row["sequence_sha256"]: row for row in read_tsv(composition)}
    if set(comp) != {row["sequence_sha256"] for row in rows}:
        raise SystemExit("the composition and nearest-neighbour tables cover different digests")
    emitted: dict[str, dict[str, str]] = {}
    if structures is not None:
        emitted = {row["sequence_sha256"]: row for row in read_tsv(structures, comment=True)}
        if set(emitted) != set(comp):
            raise SystemExit("the fold emitter's table and the composition table cover "
                             "different digests")
    worst: dict[str, float] = {column: 0.0 for column in SHARED_FLOAT}
    worst["plddt_vs_fold_emitter"] = 0.0
    for row in rows:
        other = comp[row["sequence_sha256"]]
        for column in SHARED_EXACT:
            if row[column] != other[column]:
                raise SystemExit(f"{row['sequence_sha256']}: {column} is {row[column]!r} in the "
                                 f"neighbour table and {other[column]!r} in the composition table")
        for column in SHARED_FLOAT:
            gap = abs(float(row[column]) - float(other[column]))
            worst[column] = max(worst[column], gap)
            if gap > PRINT_TOLERANCE:
                raise SystemExit(f"{row['sequence_sha256']}: {column} differs by {gap} between "
                                 "the neighbour and composition tables, beyond printed precision")
        if emitted:
            source = emitted[row["sequence_sha256"]]
            gap = abs(float(other["plddt"]) - float(source[PLDDT_COLUMN]))
            worst["plddt_vs_fold_emitter"] = max(worst["plddt_vs_fold_emitter"], gap)
            if gap > PRINT_TOLERANCE or source["arm"] != other["arm"]:
                raise SystemExit(f"{row['sequence_sha256']}: the composition table disagrees with "
                                 "the fold emitter's own per-structure table")
    sequences: dict[str, dict[str, Any]] = {}
    for line in cohort.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            sequences[str(record["id"])] = record

    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda row: (row["arm"], row["sequence_sha256"])):
        digest = row["sequence_sha256"]
        record = sequences.get(digest)
        if record is None:
            raise SystemExit(f"{digest} is not in the fold cohort, so its sequence is unknown")
        sequence = str(record["sequence"])
        if sequence_digest(sequence) != digest:
            raise SystemExit(f"{digest}: the cohort sequence digests to {sequence_digest(sequence)}")
        if int(record["length"]) != int(row["L"]) or len(sequence) != int(row["L"]):
            raise SystemExit(f"{digest}: the table reports {row['L']} residues and the cohort "
                             f"{record['length']}/{len(sequence)}")
        plddt = float(comp[digest]["plddt"])
        if not 0.0 <= plddt <= 100.0:
            raise SystemExit(f"{digest}: mean CA-pLDDT {plddt} is outside the 0-100 scale")
        identity = float(row["identity_over_query"])
        if not 0.0 <= identity <= 100.0:
            raise SystemExit(f"{digest}: identity over the query {identity} is outside 0-100")
        capped = row["capped"] == "True"
        by_arm.setdefault(row["arm"], []).append({
            "key": f"{row['arm']}|s{STREAM}|{len(by_arm.get(row['arm'], []))}",
            "cell": CELL, "digest": digest, "sequence": sequence, "length": int(row["L"]),
            "stop_status": "budget_censored" if capped else "native_terminal",
            "empty": False, "valid_aa20": set(sequence) <= AA20,
            "plddt": plddt, "identity": identity, "stratum": assign_stratum(identity),
        })

    declared = {arm: pair for pair in ot.pairs_in_lane(ot.LANE_FREE_LENGTH)
                for arm in pair.arms}
    if set(by_arm) != set(declared):
        raise SystemExit(f"the table holds arms {sorted(by_arm)} against the declared free-length "
                         f"arms {sorted(declared)}")
    pools: dict[tuple[str, int], dict[str, Any]] = {}
    for arm, members in sorted(by_arm.items()):
        pair = declared[arm]
        pools[(arm, STREAM)] = {
            "arm": arm, "stream": STREAM,
            "role": "base" if arm == pair.base else "tuned", "pair": pair.tuned,
            "cells": [CELL], "seeds": [], "checkpoints": [],
            "rows": members,
            "counts": {
                "attempts": len(members), "non_empty": len(members),
                "budget_censored": sum(1 for row in members
                                       if row["stop_status"] == "budget_censored"),
                "empty_or_noncanonical": 0,
                "searched": len(members), "folded": len(members),
                "distinct_sequences": len({row["digest"] for row in members}),
                "identity_search_coverage": 1.0, "fold_coverage": 1.0,
            },
        }
    return ({"pools": pools, "join_rule": aot.JOIN_RULE,
             "novelty_digests_outside_the_campaign": 0,
             "denominator_rule": DEPARTURES["denominator"]}, worst)


def thin(*counts: Any, floor: int = 30) -> bool:
    """Whether any realised cell behind an estimate is thin, at the task's 30-row mark."""

    return any(isinstance(value, int) and value < floor for value in counts)


def primary_rows(band: dict[str, Any], decisions: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for key, column in band["columns"].items():
        pair, stratum = key.split("|")
        units = column["units"]
        n_base = units["base"].get(str(STREAM))
        n_tuned = units["tuned"].get(str(STREAM))
        cell = decisions[pair]["per_stratum"][stratum]
        interval = column.get("interval") or [None, None]
        out.append({
            "pair": pair, "stratum": stratum, "status": column["status"],
            "n_base": n_base, "n_tuned": n_tuned,
            "thin_under_30": thin(n_base, n_tuned),
            "base_mean_plddt": column.get("base_mean"), "tuned_mean_plddt": column.get("tuned_mean"),
            "diff_plddt": column.get("point"),
            "band_lo": interval[0], "band_hi": interval[1],
            "critical_value": band.get("critical_value"),
            "margin": (cell.get("margin") or {}).get("margin"),
            "finding": cell["finding"], "zero_verdict": column.get("zero_verdict"),
            "reason": column.get("reason") or cell.get("reason"),
        })
    return sorted(out, key=lambda row: (row["pair"], row["stratum"]))


def shift_rows(shift: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for endpoint in ("mean_identity", "fraction_below_30"):
        band = shift[endpoint]
        for key, column in band["columns"].items():
            units = column["units"]
            n_base = units["base"].get(str(STREAM))
            n_tuned = units["tuned"].get(str(STREAM))
            interval = column.get("interval") or [None, None]
            out.append({
                "endpoint": endpoint, "pair": key, "status": column["status"],
                "n_base": n_base, "n_tuned": n_tuned,
                "thin_under_30": thin(n_base, n_tuned),
                "base": column.get("base_mean"), "tuned": column.get("tuned_mean"),
                "diff": column.get("point"),
                "band_lo": interval[0], "band_hi": interval[1],
                "critical_value": band.get("critical_value"),
                "zero_verdict": column.get("zero_verdict"), "reason": column.get("reason"),
            })
    return out


def control_rows(control: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for rule, block in control["rules"].items():
        band = block["bands"]
        for key, column in band["columns"].items():
            pair, stratum = key.split("|")
            units = column.get("units") or {}
            counts = {side: (units.get(side) or {}).get(str(STREAM)) for side in
                      ("base", "filtered", "tuned")}
            realised = column.get("realised") or {}
            gains = column.get("gains") or {}
            interval = column.get("interval") or [None, None]
            identity = (block["identity_axis"].get(pair) or {}).get(stratum) or {}
            finding = column.get("statistics_reproduce_the_gain") or {}
            out.append({
                "rule": rule, "pair": pair, "stratum": stratum, "status": column["status"],
                "n_base": counts["base"], "n_filtered": counts["filtered"],
                "n_tuned": counts["tuned"],
                "thin_under_30": thin(*counts.values()),
                "base_mean_plddt": realised.get("base_realised"),
                "filtered_mean_plddt": realised.get("filtered_realised"),
                "tuned_mean_plddt": realised.get("tuned_realised"),
                "tuned_minus_base_gain": gains.get("tuned_minus_base_gain"),
                "filter_minus_base_gain": gains.get("filter_minus_base_gain"),
                "reproduced_fraction": column.get("point"),
                "band_lo": interval[0], "band_hi": interval[1],
                "critical_value": band.get("critical_value"),
                "finding": finding.get("finding"),
                "primary_gain": column["primary_gain"]["gain"],
                "base_mean_identity": identity.get("base_mean_identity"),
                "filtered_mean_identity": identity.get("filtered_mean_identity"),
                "tuned_mean_identity": identity.get("tuned_mean_identity"),
                "reason": column.get("reason"),
            })
    return out


def recognition_rows(block: dict[str, Any]) -> list[dict[str, Any]]:
    if block.get("status") != "measured":
        return []
    out = []
    for endpoint in ("any_family", "complete_domain"):
        band = block[endpoint]
        for key, column in band["columns"].items():
            units = column["units"]
            n_base = units["base"].get(str(STREAM))
            n_tuned = units["tuned"].get(str(STREAM))
            interval = column.get("interval") or [None, None]
            out.append({
                "endpoint": endpoint, "pair": key, "status": column["status"],
                "n_base": n_base, "n_tuned": n_tuned,
                "thin_under_30": thin(n_base, n_tuned),
                "base_rate": column.get("base_mean"), "tuned_rate": column.get("tuned_mean"),
                "diff": column.get("point"),
                "band_lo": interval[0], "band_hi": interval[1],
                "critical_value": band.get("critical_value"),
                "zero_verdict": column.get("zero_verdict"), "reason": column.get("reason"),
            })
    return out


def selector_rows(block: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"arm": arm, "n_pool": entry["n_pool"], "budget": block["budget"],
             "pool_mean_plddt": entry["pool_mean_plddt"],
             "oracle_gain_plddt": entry["oracle_gain"],
             "thin_under_30": thin(entry["n_pool"]),
             "captured_status": entry["captured"].get("status"),
             "captured_reason": entry["captured"].get("reason")}
            for arm, entry in block["per_arm"].items()]


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t",
                                extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--neighbour", type=Path, required=True)
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--fold-structures", type=Path, default=None,
                        help="the fold emitter's own per-structure table, cross-checked on pLDDT")
    parser.add_argument("--kmer-background", type=Path, required=True)
    parser.add_argument("--recognition", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    table, worst = build_table(args.neighbour, args.composition, args.cohort,
                               args.fold_structures)
    counts = gc.load_kmer_counts(args.kmer_background, aot.KMER_ORDERS)

    # The identity shift is read first, because the registration's second
    # narrowing route is a statement about both it and the primary contrast, and
    # the module's own entry point passes it into the primary's decision rule.
    shift = aot.identity_shift(table, ota.PRIMARY_PAIRS)
    primary = aot.stratified_contrast(table, ota.PRIMARY_PAIRS, ota.PRIMARY_FAMILY, shift=shift)
    secondary = aot.stratified_contrast(table, ota.SECONDARY_PAIRS, ota.SECONDARY_FAMILY)
    control = aot.explicit_statistics_control(table, ota.PRIMARY_PAIRS, counts, primary)
    recognised = aot.recognition(table, ota.PRIMARY_PAIRS, args.recognition)
    selector = aot.selector_competence(table, None)

    layout = {f"{pool['arm']}|s{pool['stream']}": {**pool["counts"], "role": pool["role"],
                                                   "pair": pool["pair"]}
              for pool in table["pools"].values()}
    record = {
        "schema": ota.SCHEMA,
        "status": "complete",
        "run": "objective_tuned_endpoints_20261111",
        "registration": "docs/objective-alignment.md, declared 2026-11-09, amended 2026-11-10/11",
        "campaign": ot.CAMPAIGN,
        "endpoints_run": ["explicit_statistics_control", "recognition", "selector_competence"],
        "supporting": ["primary stratified contrast (the control's own input)",
                       "identity shift (the primary decision rule's second narrowing route)",
                       "secondary pair's stratified contrast"],
        "departures": DEPARTURES,
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in
                   (("neighbour", args.neighbour), ("composition", args.composition),
                    ("cohort", args.cohort),
                    *((("fold_structures", args.fold_structures),)
                      if args.fold_structures is not None else ()))
                   },
        "cross_table_agreement": {
            "columns": worst,
            "note": "maximum absolute disagreement between the two joins of one fold, and "
                    "between the composition table and the fold emitter's own table. pLDDT is "
                    "read from the composition table; the neighbour table prints five "
                    "significant digits and supplies identity over the query. The realised gaps "
                    "are printed-precision slack, far below the declared fold tolerance",
        },
        "kmer_background": {"path": str(args.kmer_background),
                            "orders_read": list(aot.KMER_ORDERS)},
        "recognition_table": None if args.recognition is None else
        {"path": str(args.recognition), "sha256": sha256_file(args.recognition)},
        "cells": {key: layout[key] for key in sorted(layout)},
        "join_rule": table["join_rule"],
        "denominator_rule": table["denominator_rule"],
        "primary": primary,
        "identity_shift": shift,
        "secondary_pair": secondary,
        "explicit_statistics_control": control,
        "recognition": recognised,
        "selector_competence": selector,
        "finding_states": ota.FINDING_STATES,
        "bounds": ota.BOUNDS,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "objective_tuned_endpoints.json", record)
    tables = args.out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_tsv(tables / "primary_contrast.tsv", primary_rows(primary["bands"], primary["per_pair"]))
    write_tsv(tables / "secondary_pair_contrast.tsv",
              primary_rows(secondary["bands"], secondary["per_pair"]))
    write_tsv(tables / "identity_shift.tsv", shift_rows(shift))
    write_tsv(tables / "explicit_statistics_control.tsv", control_rows(control))
    write_tsv(tables / "recognition.tsv", recognition_rows(recognised))
    write_tsv(tables / "selector_competence.tsv", selector_rows(selector))
    print(json.dumps({
        "out": str(args.out / "objective_tuned_endpoints.json"),
        "primary_resolved": primary["bands"]["resolved_family_size"],
        "secondary_resolved": secondary["bands"]["resolved_family_size"],
        "control": {rule: {"control_status": block["bands"]["control_status"],
                           "resolved": block["bands"]["resolved_family_size"]}
                    for rule, block in control["rules"].items()},
        "recognition_status": recognised.get("status"),
        "identity_shift_resolved": {
            "mean_identity": shift["mean_identity"]["resolved_family_size"],
            "fraction_below_30": shift["fraction_below_30"]["resolved_family_size"]},
        "selector_arms": len(selector["per_arm"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
