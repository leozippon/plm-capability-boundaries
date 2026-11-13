#!/usr/bin/env python3
"""Re-run the declared endpoints with the two inputs the 2026-11-11 run lacked.

What is different, and nothing else is
--------------------------------------
``scripts/capability/generation/objective_tuned_endpoints_20261111/run_endpoints.py``
assembled the one table the analysis module's readers would have produced, from
the per-structure fold tables and the frozen fold cohort, and declared three
departures from the registration. Two of them have since been closed and this
driver closes them:

1. *The streams.* The per-structure tables carry no stream label, so that run
   read one pooled stream per arm. The labels were recovered from the campaign's
   sixty generation ledgers into
   ``results/extensions/objective_tuned_streams_20261111/stream_labels.tsv``, one
   per folded structure. This driver reads them and splits each arm's pool into
   its three declared streams. The module is already stream-aware throughout --
   it resamples within stream, ranks the control's base pool within
   (arm, stream), and holds every contributing (arm, stream) pool to the
   resampling floor -- so nothing in it changes; what changes is the table it is
   handed.

   Two consequences are stated rather than discovered later. The declared pool
   mean is *stream-equal*, the mean of the three stream means, so a point
   estimate over three streams is not the pooled sequence-equal mean of one; and
   the resampling floor now binds per (arm, stream), so a thin cell that carried
   an estimate pooled can be unresolved per stream. Both are the declaration's
   own, and the second is a loss of reach bought with a denominator the
   declaration actually asks for.

2. *The selector.* That run passed no ``--selector-score``, so every arm recorded
   ``captured: {"status": "not_measured"}``. ``build_selector_scores.py`` beside
   this file reads each arm's own likelihood of its own products out of the frozen
   own-model cells of ``score_cohort_likelihood.py`` and writes them in the shape
   the module declares; this driver passes them.

The third departure is unchanged: the denominator is the folded eighteen-shard
structure, not the retained attempt, because the attempt ledgers are not on this
host. ``stop_status`` is still recovered from the table's ``capped`` flag.

This driver computes no statistic. The table builder, every cross-check on it,
every estimator, family, seed, margin, floor, filter rule and finding state is
the 2026-11-11 driver's or the analysis module's, imported and called unchanged.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
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

#: The frozen 2026-11-11 driver, imported by path so that its table builder and
#: all of its cross-table checks are reused rather than restated. Its directory
#: is not on the import path and its module name is not a package, so the import
#: is explicit.
_PRODUCER = (REPO_ROOT / "scripts/capability/generation/objective_tuned_endpoints_20261111"
             / "run_endpoints.py")


def _load_producer() -> Any:
    spec = importlib.util.spec_from_file_location("objective_tuned_endpoints_20261111", _PRODUCER)
    if spec is None or spec.loader is None:
        raise SystemExit(f"{_PRODUCER} cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pooled = _load_producer()

#: The declared streams of the free-length lane, as the recovered labels carry
#: them. A pool that does not hold exactly these is refused: a missing stream
#: would silently change the estimand from a three-stream mean to a one- or
#: two-stream mean.
STREAMS = (0, 1, 2)

DEPARTURES = {
    "streams": "closed. The three declared streams are read per arm from the labels recovered "
               "from the sixty generation ledgers, so the pool mean is the stream-equal mean the "
               "registration declares, the resample is stratified by stream, the control ranks "
               "within (arm, stream), and the resampling floor binds per (arm, stream)",
    "selector": "closed. Each arm's own likelihood of its own products is supplied from the "
                "frozen own-model cells of score_cohort_likelihood.py; see "
                "selector/selector_score.json for the selector, the join and the coverage checks",
    "denominator": pooled.DEPARTURES["denominator"],
    "support": pooled.DEPARTURES["support"],
    "stop_status": pooled.DEPARTURES["stop_status"],
}


def read_labels(path: Path) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for row in pooled.read_tsv(path):
        digest = row["sequence_sha256"]
        if digest in labels:
            raise SystemExit(f"{path}: {digest} carries two labels")
        labels[digest] = row
    return labels


def split_streams(table: dict[str, Any], labels: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Split each one-stream pool into the three streams the labels name.

    The rows are the pooled table's own, re-keyed within their stream; nothing is
    recomputed. A digest with no label, a label naming another arm, a cell label
    that does not spell its own (arm, stream, replicate), or an arm whose rows do
    not cover exactly the declared streams is refused.
    """

    pools: dict[tuple[str, int], dict[str, Any]] = {}
    for (arm, _), pool in sorted(table["pools"].items()):
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in pool["rows"]:
            label = labels.get(row["digest"])
            if label is None:
                raise SystemExit(f"{row['digest']} ({arm}) carries no recovered stream label; an "
                                 "unlabelled structure cannot be assigned to a stream and is "
                                 "not pooled into one")
            if label["arm"] != arm:
                raise SystemExit(f"{row['digest']}: the fold table says {arm} and the recovered "
                                 f"label says {label['arm']}")
            stream, replicate = int(label["stream"]), int(label["replicate"])
            if label["cell_label"] != f"ot_{arm}_s{stream}_r{replicate}":
                raise SystemExit(f"{row['digest']}: cell label {label['cell_label']!r} does not "
                                 f"spell arm {arm}, stream {stream}, replicate {replicate}")
            grouped.setdefault(stream, []).append({**row, "cell": label["cell_label"]})
        if tuple(sorted(grouped)) != STREAMS:
            raise SystemExit(f"{arm} holds streams {sorted(grouped)} against the declared "
                             f"{list(STREAMS)}")
        for stream, members in sorted(grouped.items()):
            rows = [{**row, "key": f"{arm}|s{stream}|{index}"}
                    for index, row in enumerate(sorted(members, key=lambda row: row["digest"]))]
            censored = sum(1 for row in rows if row["stop_status"] == "budget_censored")
            pools[(arm, stream)] = {
                "arm": arm, "stream": stream, "role": pool["role"], "pair": pool["pair"],
                "cells": sorted({row["cell"] for row in rows}),
                "seeds": [], "checkpoints": [], "rows": rows,
                "counts": {
                    "attempts": len(rows), "non_empty": len(rows),
                    "budget_censored": censored, "empty_or_noncanonical": 0,
                    "searched": len(rows), "folded": len(rows),
                    "distinct_sequences": len({row["digest"] for row in rows}),
                    "identity_search_coverage": 1.0, "fold_coverage": 1.0,
                },
            }
    moved = sum(len(pool["rows"]) for pool in pools.values())
    held = sum(len(pool["rows"]) for pool in table["pools"].values())
    if moved != held:
        raise SystemExit(f"the split holds {moved} rows against the pooled table's {held}")
    return {**table, "pools": pools}


# ------------------------------------------------------------------- the tables
#
# One emitter per endpoint, differing from the 2026-11-11 driver's only in that a
# per-stream count is a dict rather than a single stream's number: the totals are
# printed beside the per-stream counts, and thinness is read of the thinnest
# stream rather than of a pooled total that no estimator uses.

def units(side: dict[str, Any] | None) -> tuple[int | None, str, int | None]:
    """A pool's total units, its per-stream counts as printed, and its thinnest stream."""

    if not side:
        return None, "", None
    counts = {int(stream): int(value) for stream, value in side.items()}
    printed = ",".join(f"{stream}:{counts[stream]}" for stream in sorted(counts))
    return sum(counts.values()), printed, min(counts.values())


def thin(*minima: int | None, floor: int = 30) -> bool:
    """Whether any realised (arm, stream) cell behind an estimate is thin, at 30 rows."""

    return any(value is not None and value < floor for value in minima)


def streams_of(cell: dict[str, Any]) -> tuple[str, str]:
    """The per-stream point differences and which of them clear the margin."""

    per_stream = cell.get("streams") or {}
    order = sorted(per_stream, key=int)
    return (",".join(f"{stream}:{per_stream[stream]['difference']:+.3f}" for stream in order),
            ",".join(stream for stream in order if per_stream[stream]["exceeds_margin"]))


def primary_rows(band: dict[str, Any], decisions: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for key, column in band["columns"].items():
        pair, stratum = key.split("|")
        n_base, base_streams, base_min = units(column["units"]["base"])
        n_tuned, tuned_streams, tuned_min = units(column["units"]["tuned"])
        cell = decisions[pair]["per_stratum"][stratum]
        interval = column.get("interval") or [None, None]
        differences, exceeding = streams_of(cell)
        out.append({
            "pair": pair, "stratum": stratum, "status": column["status"],
            "n_base": n_base, "n_tuned": n_tuned,
            "base_units_by_stream": base_streams, "tuned_units_by_stream": tuned_streams,
            "thin_under_30": thin(base_min, tuned_min),
            "base_mean_plddt": column.get("base_mean"), "tuned_mean_plddt": column.get("tuned_mean"),
            "diff_plddt": column.get("point"),
            "band_lo": interval[0], "band_hi": interval[1],
            "critical_value": band.get("critical_value"),
            "margin": (cell.get("margin") or {}).get("margin"),
            "stream_differences": differences, "streams_exceeding_margin": exceeding,
            "finding": cell["finding"], "zero_verdict": column.get("zero_verdict"),
            "reason": column.get("reason") or cell.get("reason"),
        })
    return sorted(out, key=lambda row: (row["pair"], row["stratum"]))


def shift_rows(shift: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for endpoint in ("mean_identity", "fraction_below_30"):
        band = shift[endpoint]
        for key, column in band["columns"].items():
            n_base, base_streams, base_min = units(column["units"]["base"])
            n_tuned, tuned_streams, tuned_min = units(column["units"]["tuned"])
            interval = column.get("interval") or [None, None]
            out.append({
                "endpoint": endpoint, "pair": key, "status": column["status"],
                "n_base": n_base, "n_tuned": n_tuned,
                "base_units_by_stream": base_streams, "tuned_units_by_stream": tuned_streams,
                "thin_under_30": thin(base_min, tuned_min),
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
            side = column.get("units") or {}
            printed, minima = {}, []
            totals = {}
            for name in ("base", "filtered", "tuned"):
                totals[name], printed[name], smallest = units(side.get(name))
                minima.append(smallest)
            realised = column.get("realised") or {}
            gains = column.get("gains") or {}
            interval = column.get("interval") or [None, None]
            identity = (block["identity_axis"].get(pair) or {}).get(stratum) or {}
            finding = column.get("statistics_reproduce_the_gain") or {}
            out.append({
                "rule": rule, "pair": pair, "stratum": stratum, "status": column["status"],
                "n_base": totals["base"], "n_filtered": totals["filtered"],
                "n_tuned": totals["tuned"],
                "base_units_by_stream": printed["base"],
                "filtered_units_by_stream": printed["filtered"],
                "tuned_units_by_stream": printed["tuned"],
                "thin_under_30": thin(*minima),
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
            n_base, base_streams, base_min = units(column["units"]["base"])
            n_tuned, tuned_streams, tuned_min = units(column["units"]["tuned"])
            interval = column.get("interval") or [None, None]
            out.append({
                "endpoint": endpoint, "pair": key, "status": column["status"],
                "n_base": n_base, "n_tuned": n_tuned,
                "base_units_by_stream": base_streams, "tuned_units_by_stream": tuned_streams,
                "thin_under_30": thin(base_min, tuned_min),
                "base_rate": column.get("base_mean"), "tuned_rate": column.get("tuned_mean"),
                "diff": column.get("point"),
                "band_lo": interval[0], "band_hi": interval[1],
                "critical_value": band.get("critical_value"),
                "zero_verdict": column.get("zero_verdict"), "reason": column.get("reason"),
            })
    return out


def selector_rows(block: dict[str, Any]) -> list[dict[str, Any]]:
    """The selector endpoint, with the captured share this run can finally print.

    ``captured_status`` is the module's own ``status`` where it sets one --
    ``not_measured`` with no score, ``unresolved`` where a score misses a folded
    attempt -- and ``measured`` where the module computed the share instead of
    declining to.
    """

    out = []
    for arm, entry in block["per_arm"].items():
        captured = entry["captured"]
        out.append({
            "arm": arm, "n_pool": entry["n_pool"], "budget": block["budget"],
            "pool_mean_plddt": entry["pool_mean_plddt"],
            "oracle_gain_plddt": entry["oracle_gain"],
            "thin_under_30": thin(entry["n_pool"]),
            "captured_status": captured.get("status", "measured"),
            "selector_gain_plddt": captured.get("selector_gain"),
            "captured_fraction": captured.get("captured_fraction"),
            "captured_reason": captured.get("reason"),
        })
    return out


def control_family_comparison(control: dict[str, Any],
                              frozen: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Per rule: the control's family-level disposition pooled against per stream."""

    out = []
    for rule, block in control["rules"].items():
        band = block["bands"]
        before = (((frozen or {}).get("explicit_statistics_control") or {}).get("rules")
                  or {}).get(rule, {}).get("bands") or {}
        out.append({
            "rule": rule,
            "declared_family_size": band["declared_family_size"],
            "pooled_control_status": before.get("control_status"),
            "pooled_resolved_family_size": before.get("resolved_family_size"),
            "pooled_critical_value": before.get("critical_value"),
            "streams_control_status": band["control_status"],
            "streams_resolved_family_size": band["resolved_family_size"],
            "streams_critical_value": band.get("critical_value"),
            "verdict_moved": None if not before else
            bool(before.get("control_status") != band["control_status"]),
        })
    return out


def control_cell_comparison(control: dict[str, Any],
                            frozen: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Per cell: the control's status, share and finding, pooled against per stream."""

    rules = (((frozen or {}).get("explicit_statistics_control") or {}).get("rules") or {})
    out = []
    for rule, block in control["rules"].items():
        before = (rules.get(rule, {}).get("bands") or {}).get("columns") or {}
        for key, column in block["bands"]["columns"].items():
            pair, stratum = key.split("|")
            old = before.get(key) or {}
            old_finding = (old.get("statistics_reproduce_the_gain") or {}).get("finding")
            new_finding = (column.get("statistics_reproduce_the_gain") or {}).get("finding")
            out.append({
                "rule": rule, "pair": pair, "stratum": stratum,
                "pooled_status": old.get("status"),
                "streams_status": column["status"],
                "pooled_reproduced_fraction": old.get("point"),
                "streams_reproduced_fraction": column.get("point"),
                "pooled_finding": old_finding, "streams_finding": new_finding,
                "status_moved": None if not old else bool(old.get("status") != column["status"]),
                "finding_moved": None if not old else bool(old_finding != new_finding),
            })
    return out


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
    parser.add_argument("--stream-labels", type=Path, required=True,
                        help="one recovered (arm, stream, replicate) label per folded structure")
    parser.add_argument("--kmer-background", type=Path, required=True)
    parser.add_argument("--recognition", type=Path, default=None)
    parser.add_argument("--selector-score", type=Path, required=True,
                        help="per-sequence selector score, one row per folded attempt")
    parser.add_argument("--compare", type=Path, default=None,
                        help="the pooled-stream record this run is read against")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    one_stream, worst = pooled.build_table(args.neighbour, args.composition, args.cohort,
                                           args.fold_structures)
    table = split_streams(one_stream, read_labels(args.stream_labels))
    counts = gc.load_kmer_counts(args.kmer_background, aot.KMER_ORDERS)

    shift = aot.identity_shift(table, ota.PRIMARY_PAIRS)
    primary = aot.stratified_contrast(table, ota.PRIMARY_PAIRS, ota.PRIMARY_FAMILY, shift=shift)
    secondary = aot.stratified_contrast(table, ota.SECONDARY_PAIRS, ota.SECONDARY_FAMILY)
    control = aot.explicit_statistics_control(table, ota.PRIMARY_PAIRS, counts, primary)
    recognised = aot.recognition(table, ota.PRIMARY_PAIRS, args.recognition)
    selector = aot.selector_competence(table, args.selector_score)

    frozen = None if args.compare is None else json.loads(
        args.compare.read_text(encoding="utf-8"))
    layout = {f"{pool['arm']}|s{pool['stream']}": {**pool["counts"], "role": pool["role"],
                                                   "pair": pool["pair"], "cells": pool["cells"]}
              for pool in table["pools"].values()}
    record = {
        "schema": ota.SCHEMA,
        "status": "complete",
        "run": args.out.name,
        "registration": "docs/objective-alignment.md, declared 2026-11-09, amended 2026-11-10/11",
        "campaign": ot.CAMPAIGN,
        "reads": "the 2026-11-11 endpoints run's own table builder, with the recovered stream "
                 "labels and the arms' own likelihoods added",
        "endpoints_run": ["explicit_statistics_control", "recognition", "selector_competence"],
        "supporting": ["primary stratified contrast (the control's own input)",
                       "identity shift (the primary decision rule's second narrowing route)",
                       "secondary pair's stratified contrast"],
        "departures": DEPARTURES,
        "streams": list(STREAMS),
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in
                   (("neighbour", args.neighbour), ("composition", args.composition),
                    ("cohort", args.cohort), ("stream_labels", args.stream_labels),
                    ("selector_score", args.selector_score),
                    *((("fold_structures", args.fold_structures),)
                      if args.fold_structures is not None else ()))
                   },
        "cross_table_agreement": {
            "columns": worst,
            "note": "the 2026-11-11 driver's own check, re-run here: the maximum absolute "
                    "disagreement between the two joins of one fold, and between the composition "
                    "table and the fold emitter's own table. The realised gaps are "
                    "printed-precision slack, far below the declared fold tolerance",
        },
        "kmer_background": {"path": str(args.kmer_background),
                            "orders_read": list(aot.KMER_ORDERS)},
        "recognition_table": None if args.recognition is None else
        {"path": str(args.recognition), "sha256": sha256_file(args.recognition)},
        "compared_against": None if args.compare is None else
        {"path": str(args.compare), "sha256": sha256_file(args.compare),
         "run": frozen.get("run")},
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
    write_tsv(tables / "control_family_pooled_vs_streams.tsv",
              control_family_comparison(control, frozen))
    write_tsv(tables / "control_cells_pooled_vs_streams.tsv",
              control_cell_comparison(control, frozen))
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
        "selector": {arm: entry["captured"].get("captured_fraction")
                     for arm, entry in selector["per_arm"].items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
