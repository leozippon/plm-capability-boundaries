#!/usr/bin/env python3
"""What the 5-mer filter rule is on the base pools, and what length does to its share.

The rule ``low_complexity_residues`` keeps the top quarter of a base pool by
distinct 5-mer fraction. This run measures two things the control's own records
did not carry, from the stream-separated table the 2026-11-12 run read.

1. *Ties.* The statistic is capped at 1.0, so a pool in which more than a quarter
   of the sequences contain no repeated 5-mer cuts at 1.0 and the retained
   quarter lies wholly inside that group, ordered by the attempt key. The run
   reads the cut score and the tie counts from ``rank_and_retain`` at the
   retained fractions of the selection-strength curve (k of 2, 4 and 8), for every
   base arm and stream, and checks that the key-ordered subsample of the cut group is
   representative of the group on length and pLDDT (finite-population z), which is
   what makes the subsample content-independent rather than assumed so.
2. *Length.* The retained subset is shorter than its parent, as is the tuned pool,
   and the control's share is a raw difference of means. The share is re-read with the length
   adjustment the project already uses, ``bin_match`` and ``nn_match`` of
   ``scripts/capability/generation/length_adjusted_contrasts_20261110/analyse.py``,
   taken from that file's own source and not restated. The matcher is applied to
   every draw of the declared nested resample, so the band is the declared
   simultaneous band over the declared family and seed block, read off the same
   draws as the raw share; nothing about the resampling unit, the draw count, the
   floor or the band construction is chosen here.

The tuned arm's pool is shorter than its base too, so the gain the share divides by
has a length component of its own. The same two matchers are applied to the
tuned-minus-base gain as realised point estimates, with no band, because a band
would need a resample of the tuned pool's lengths that the declared machinery does
not draw. They are reported beside the banded share and labelled unbanded.

The raw share is computed again through the same assembly and must equal the
stored k of 4 cell of ``selection_curve.json`` exactly, in status, point and both
ends of the band, or the run refuses. That is what licenses reading the matched
columns beside it.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/capability/generation"))

import analyse_objective_tuned as aot  # noqa: E402

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.extensions import bidirectional as bd  # noqa: E402
from src.capability.generation import generative_control as gc  # noqa: E402
from src.capability.generation import objective_tuned_analysis as ota  # noqa: E402

RULE = "low_complexity_residues"
KS = (2, 4, 8)
PRIMARY_K = 4
MATCHERS = ("bin", "nn")
MATCHER_SOURCE = REPO_ROOT / ("scripts/capability/generation/length_adjusted_contrasts_20261110/"
                              "analyse.py")
#: The 30-to-70% stratum is the one stratum in which the rule's three primary cells resolve.
REPORTED_STRATUM = ota.STRATA[1]
#: Length bins of the within-bin reading. The upper bin is the budget-censored one:
#: every sequence generated to the 400-residue cap is 399 or 400 residues.
LENGTH_BINS = ((0, 223), (224, 319), (320, 398), (399, 400))
PERCENTILES = (5, 25, 50, 75, 95)
#: A sequence of at least this length ran to the generation budget.
CAP_LENGTH = LENGTH_BINS[-1][0]

MATCHED_METHOD = (
    "the declared nested resample, with a length matcher applied to each draw. Within "
    "(stream, stratum) the draw's resampled base members are the parent and the draw's "
    "retained members of that stratum are the filtered side; bin_match (16-residue bins, "
    "weighted by the retained side) or nn_match (nearest parent length, caliper 5) gives that "
    "draw's matched filtered-minus-base difference, and the filtered mean the declared band "
    "machinery reads is the draw's base mean plus that difference. The point is the same "
    "matcher on the realised pools, averaged stream-equal. The tuned side and the denominator "
    "are the declared ones, unmatched. The band is the declared simultaneous band of the "
    "rule's own family and seed block, one per matcher, never merged with the raw band or "
    "with each other"
)
GAIN_METHOD = (
    "the tuned-minus-base gain under the same two matchers, realised, stream-equal and "
    "unbanded: bin_match and nn_match with the base pool as the parent and the tuned pool as "
    "the matched side, as length_adjusted_contrasts_20261110 applies them. Each side of the "
    "ratio is standardised to its own matched group's length mix, so the ratio is not a "
    "declared estimand"
)


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


streamed = _load(REPO_ROOT / "scripts/capability/generation/objective_tuned_endpoints_20261112"
                 / "run_stream_endpoints.py", "objective_tuned_stream_endpoints")


def load_matchers() -> dict[str, Any]:
    """``bin_match`` and ``nn_match`` from the frozen producer's source.

    That file runs its whole analysis on import, so its two matchers are compiled
    from their own definitions rather than imported or copied.
    """

    wanted = {"bin_match", "nn_match"}
    tree = ast.parse(MATCHER_SOURCE.read_text(encoding="utf-8"))
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in wanted]
    if {node.name for node in found} != wanted:
        raise SystemExit(f"{MATCHER_SOURCE} no longer defines {sorted(wanted)}")
    namespace: dict[str, Any] = {"np": np}
    exec(compile(ast.Module(body=found, type_ignores=[]), str(MATCHER_SOURCE), "exec"), namespace)
    return {"bin": namespace["bin_match"], "nn": namespace["nn_match"]}


def stream_equal(matcher, parent: dict[int, list[dict]], other: dict[int, list[dict]],
                 stratum: str) -> tuple[float, int]:
    """The stream-equal mean matched difference of ``other`` against ``parent`` in a stratum.

    Also the number of ``other`` rows the matcher kept, summed over streams.
    """

    diffs, kept = [], 0
    for stream in sorted(parent):
        arrays = [np.asarray([row[field] for row in rows[stream] if row["stratum"] == stratum],
                             dtype=float)
                  for rows in (parent, other) for field in ("length", "plddt")]
        result = matcher(*arrays)
        diffs.append(result[0])
        kept += int(result[1])
    return float(np.mean(diffs)), kept


def percentiles(values: list[int]) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {"n": len(values)}
    for point in PERCENTILES:
        out[f"p{point}"] = float(np.percentile(values, point)) if values else None
    out["share_at_cap"] = float(np.mean(np.asarray(values) >= CAP_LENGTH)) if values else None
    return out


def representativeness(rows: list[dict[str, Any]], keys: set[str], cut: float) -> dict[str, Any]:
    """Whether the key-ordered subsample of the cut group looks like the whole group.

    The finite-population z of the retained members' mean against the mean of the
    group tied at the cut, for length and for pLDDT. ``None`` where the whole group
    is retained or too little of it is drawn for a standard error.
    """

    group = [row for row in rows if row["score"] == cut]
    drawn = [row for row in group if row["key"] in keys]
    out: dict[str, Any] = {}
    for field in ("length", "plddt"):
        values = np.asarray([row[field] for row in group], dtype=float)
        chosen = np.asarray([row[field] for row in drawn], dtype=float)
        n, size = len(drawn), len(group)
        spread = float(values.var(ddof=1)) if size > 1 else 0.0
        if n < 2 or n == size or spread == 0.0:
            out[f"z_{field}_retained_in_group"] = None
        else:
            out[f"z_{field}_retained_in_group"] = float(
                (chosen.mean() - values.mean()) / np.sqrt(spread / n * (1 - n / size)))
    return out


def retain_pool(pair, base, rule, label: str, stored: dict[str, Any],
                tie_rows: list[dict[str, Any]]):
    """Each stream's scored pool and top quarter, and the tie record at every k of the curve."""

    scored: dict[int, list[dict[str, Any]]] = {}
    retained: dict[int, set[str]] = {}
    for stream in sorted(base):
        scored[stream] = ota.score_pool(
            [row for row in base[stream]["rows"] if row["plddt"] is not None], rule)["scored"]
        for k in KS:
            retention = ota.rank_and_retain(scored[stream], rule, fraction=1 / k)
            keys = retention.pop("retained_keys")
            stored_n = stored["controls"][f"{label}|k{k}|statistical"][RULE]["retained"][
                pair.tuned][str(stream)]
            if retention["n_retained"] != stored_n:
                raise SystemExit(f"{pair.base} stream {stream} k={k}: retains "
                                 f"{retention['n_retained']}, the stored run {stored_n}")
            tie_rows.append({
                "arm": pair.base, "stream": stream, "k": k, "n_scored": retention["n_scored"],
                "n_retained": retention["n_retained"], "cut_score": retention["cut_score"],
                "n_tied_at_cut": retention["n_tied_at_cut"],
                "tie_group_share_of_pool": retention["n_tied_at_cut"] / retention["n_scored"],
                "n_retained_at_cut": retention["n_retained_at_cut"],
                "retained_decided_by_key_share": retention["n_retained_at_cut"]
                / retention["n_retained"],
                **representativeness(scored[stream], keys, retention["cut_score"])})
            if k == PRIMARY_K:
                retained[stream] = set(keys)
    return scored, retained


def draw_matched(scored, rule, family, index, needed, matchers, codes):
    """The declared nested draws of every stream, with each matcher read off each draw."""

    draws: dict[int, dict[str, Any]] = {}
    matched: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for position, stream in enumerate(sorted(scored)):
        order = ota._by_score(scored[stream], rule.retain)
        length = np.asarray([scored[stream][i]["length"] for i in order], dtype=float)
        plddt = np.asarray([scored[stream][i]["plddt"] for i in order], dtype=float)
        band = np.asarray([codes[scored[stream][i]["stratum"]] for i in order])
        store = matched[stream] = {stratum: {name: np.full(bd.DRAWS, np.nan) for name in matchers}
                                   for stratum in needed}

        def readout(start, stop, blocks, top, *, length=length, plddt=plddt, band=band,
                    store=store):
            for stratum in needed:
                if stratum not in blocks:
                    continue
                for offset in range(stop - start):
                    drawn = blocks[stratum][offset]
                    kept = top[offset][band[top[offset]] == codes[stratum]]
                    arrays = (length[drawn], plddt[drawn], length[kept], plddt[kept])
                    for name, matcher in matchers.items():
                        store[stratum][name][start + offset] = matcher(*arrays)[0]

        draws[stream] = ota.nested_base_draws(
            scored[stream], retain=rule.retain, family=family, column=index,
            stream_index=position, readout=readout)
    return draws, matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--neighbour", type=Path, required=True)
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--fold-structures", type=Path, default=None)
    parser.add_argument("--stream-labels", type=Path, required=True)
    parser.add_argument("--kmer-background", type=Path, required=True)
    parser.add_argument("--stored-curve", type=Path, required=True,
                        help="selection_curve.json, whose k = 4 cells are the stored raw shares")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    matchers = load_matchers()
    estimators = ("raw", *MATCHERS)
    one_stream, worst = streamed.pooled.build_table(args.neighbour, args.composition, args.cohort,
                                                    args.fold_structures)
    table = streamed.split_streams(one_stream, streamed.read_labels(args.stream_labels))
    counts = gc.load_kmer_counts(args.kmer_background, aot.KMER_ORDERS)
    rule = ota.filter_rules(counts)[0]
    if rule.name != RULE:
        raise SystemExit(f"the first declared rule is {rule.name}, not {RULE}")
    stored = json.loads(args.stored_curve.read_text(encoding="utf-8"))
    codes = {name: position for position, name in enumerate(ota.STRATA)}

    shift = aot.identity_shift(table, ota.PRIMARY_PAIRS)
    sets = (
        ("primary", ota.PRIMARY_PAIRS, ota.CONTROL_FAMILIES[RULE],
         aot.stratified_contrast(table, ota.PRIMARY_PAIRS, ota.PRIMARY_FAMILY, shift=shift)),
        ("secondary", ota.SECONDARY_PAIRS, ota.SECONDARY_CONTROL_FAMILIES[RULE],
         aot.stratified_contrast(table, ota.SECONDARY_PAIRS, ota.SECONDARY_FAMILY)),
    )

    tie_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    share_rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    records: dict[str, Any] = {}
    reproduced = 0

    for label, pairs, family, contrast in sets:
        columns = {name: [] for name in estimators}
        points: dict[tuple[str, str], dict[str, Any]] = {}
        pools = {}
        for index, pair in enumerate(pairs):
            base, tuned = aot.require_pair(table, pair)
            scored, retained = retain_pool(pair, base, rule, label, stored, tie_rows)
            pools[pair.base] = (scored, retained)
            kept_rows = {stream: [row for row in scored[stream] if row["key"] in retained[stream]]
                         for stream in scored}
            tuned_rows = {stream: [row for row in tuned[stream]["rows"]
                                   if row["plddt"] is not None] for stream in tuned}

            # Pools, floor and gain disposition decide which cells can reach a share at all.
            cells: dict[str, dict[str, Any]] = {}
            needed = set()
            for stratum in ota.STRATA:
                key = f"{pair.tuned}|{stratum}"
                cell = {"key": key, "primary": contrast["bands"]["columns"].get(key),
                        "base_pool": aot.scored_by_stratum(scored, stratum),
                        "filtered_pool": aot.scored_by_stratum(scored, stratum, retained),
                        "tuned_pool": aot.plddt_by_stratum(tuned, stratum)}
                cell["short"] = {side: ota.pool_shortfall(cell[f"{side}_pool"])
                                 for side in ("base", "filtered", "tuned")}
                cells[stratum] = cell
                identified = ota.primary_gain_disposition(cell["primary"])["gain"] \
                    == ota.GAIN_IDENTIFIED
                if identified and not any(cell["short"].values()):
                    needed.add(stratum)

            draws, matched = draw_matched(scored, rule, family, index, needed, matchers, codes)

            for stratum in ota.STRATA:
                cell = cells[stratum]
                column = {
                    "key": cell["key"], "primary": cell["primary"], "tuned": cell["tuned_pool"],
                    "units": {side: ota.pool_units(cell[f"{side}_pool"])
                              for side in ("base", "filtered", "tuned")},
                    "retained": sum(len(v) for v in cell["filtered_pool"].values()),
                }
                if any(cell["short"].values()):
                    column["skip_reason"] = (f"below the resampling floor on stream(s) "
                                             f"{cell['short']}; " + ota.ADMISSIBILITY_RULE)
                else:
                    column.update(base_realised=ota.pool_mean(cell["base_pool"]),
                                  filtered_realised=ota.pool_mean(cell["filtered_pool"]),
                                  tuned_realised=ota.pool_mean(cell["tuned_pool"]))
                columns["raw"].append({
                    **column, **{field: {s: draws[s][stratum][field] for s in draws}
                                 for field in ("base", "filtered", "valid")}})
                if stratum not in needed:
                    for name in MATCHERS:
                        columns[name].append(column)
                    continue
                for name, matcher in matchers.items():
                    difference, kept = stream_equal(matcher, scored, kept_rows, stratum)
                    gain, gain_kept = stream_equal(matcher, scored, tuned_rows, stratum)
                    columns[name].append({
                        **column,
                        "base": {s: draws[s][stratum]["base"] for s in draws},
                        "filtered": {s: draws[s][stratum]["base"] + matched[s][stratum][name]
                                     for s in draws},
                        "valid": {s: draws[s][stratum]["valid"]
                                  & np.isfinite(matched[s][stratum][name]) for s in draws},
                        "filtered_realised": column["base_realised"] + difference,
                        "matched_retained": kept})
                    points[(cell["key"], name)] = {
                        "filter_minus_base_matched": difference, "n_tuned_matched": gain_kept,
                        "tuned_minus_base_matched": gain,
                        "tuned_minus_base_raw": column["tuned_realised"] - column["base_realised"]}

        bands = {name: ota.reproduced_fraction_bands(columns[name], family=family)
                 for name in estimators}
        records[label] = {"family": family, "bands": bands}

        # The raw share, again, must be the stored one.
        stored_cells = stored["cells"][RULE]
        for key, cell in bands["raw"]["columns"].items():
            old = stored_cells[key][str(PRIMARY_K)]
            interval = cell.get("interval") or [None, None]
            fresh = (cell["status"], cell.get("point"), interval[0], interval[1])
            if fresh != (old["status"], old["share"], old["lo"], old["hi"]):
                raise SystemExit(f"{key}: the raw share {fresh} is not the stored k = 4 cell "
                                 f"{(old['status'], old['share'], old['lo'], old['hi'])}")
            reproduced += 1
        for name in estimators:
            inputs = {column["key"]: column for column in columns[name]}
            for key, cell in bands[name]["columns"].items():
                if bands["raw"]["columns"][key]["status"] != "resolved":
                    continue
                interval = cell.get("interval") or [None, None]
                gains = cell.get("gains") or {}
                pair_name, stratum = key.split("|")
                share_rows.append({
                    "pairs": label, "pair": pair_name, "stratum": stratum, "estimator": name,
                    "status": cell["status"], "retained": inputs[key]["retained"],
                    "matched_retained": inputs[key].get("matched_retained"),
                    "filter_minus_base": gains.get("filter_minus_base_gain"),
                    "tuned_minus_base": gains.get("tuned_minus_base_gain"),
                    "share": cell.get("point"), "band_lo": interval[0], "band_hi": interval[1],
                    "critical_value": bands[name].get("critical_value"),
                    "finding": (cell.get("statistics_reproduce_the_gain") or {}).get("finding"),
                    "reason": cell.get("reason")})
                if name in MATCHERS:
                    point = points[(key, name)]
                    gain_rows.append({
                        "pairs": label, "pair": pair_name, "stratum": stratum, "estimator": name,
                        "tuned_minus_base_raw": point["tuned_minus_base_raw"],
                        "tuned_minus_base_matched": point["tuned_minus_base_matched"],
                        "n_tuned_matched": point["n_tuned_matched"],
                        "filter_minus_base_matched": point["filter_minus_base_matched"],
                        "share_filter_matched_over_raw_gain": cell.get("point"),
                        "share_both_sides_matched": point["filter_minus_base_matched"]
                        / point["tuned_minus_base_matched"]})

        # Length of the parent and of what the rule keeps, and the within-bin reading.
        for pair in pairs:
            scored, retained = pools[pair.base]
            members = [row for stream in sorted(scored) for row in scored[stream]]
            kept = [row for stream in sorted(scored) for row in scored[stream]
                    if row["key"] in retained[stream]]
            clean = [row for row in members if row["score"] == 1.0]
            _, tuned = aot.require_pair(table, pair)
            tuned_pool = [row for stream in sorted(tuned) for row in tuned[stream]["rows"]
                          if row["plddt"] is not None]
            for scope, test in (("all_strata", lambda row: True),
                                (REPORTED_STRATUM, lambda row: row["stratum"] == REPORTED_STRATUM)):
                for arm, group, rows in ((pair.base, "parent", members),
                                         (pair.base, "retained_top_quarter", kept),
                                         (pair.base, "no_repeated_5mer", clean),
                                         (pair.tuned, "tuned_pool", tuned_pool)):
                    length_rows.append({"arm": arm, "scope": scope, "group": group,
                                        **percentiles([row["length"] for row in rows
                                                       if test(row)])})
            if label != "primary":
                continue
            for low, high in LENGTH_BINS:
                cut = [row for row in members if row["stratum"] == REPORTED_STRATUM
                       and low <= row["length"] <= high]
                cut_kept = [row for row in kept if row["stratum"] == REPORTED_STRATUM
                            and low <= row["length"] <= high]
                parent_mean = float(np.mean([row["plddt"] for row in cut])) if cut else None
                kept_mean = float(np.mean([row["plddt"] for row in cut_kept])) if cut_kept else None
                bin_rows.append({
                    "arm": pair.base, "stratum": REPORTED_STRATUM, "length_bin": f"{low}-{high}",
                    "n_parent": len(cut), "parent_mean_plddt": parent_mean,
                    "n_retained": len(cut_kept), "retained_mean_plddt": kept_mean,
                    "retained_minus_parent": None if not (cut and cut_kept)
                    else kept_mean - parent_mean})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables").mkdir(exist_ok=True)
    write_json(args.out / "five_mer_rule.json", {
        "schema": ota.SCHEMA,
        "reads": "the stream-separated table of objective_tuned_endpoints_20261112, by its own "
                 "table builder and stream split",
        "rule": RULE, "retained_fraction": 1 / PRIMARY_K, "curve_k": list(KS),
        "matched_method": MATCHED_METHOD, "gain_method": GAIN_METHOD,
        "matchers": {"source": str(MATCHER_SOURCE.relative_to(REPO_ROOT)),
                     "sha256": sha256_file(MATCHER_SOURCE), "bin": "bin_match, 16-residue bins",
                     "nn": "nn_match, caliper 5"},
        "raw_cells_checked_against_stored": reproduced,
        "cross_table_agreement": {"columns": worst},
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in
                   (("neighbour", args.neighbour), ("composition", args.composition),
                    ("cohort", args.cohort), ("stream_labels", args.stream_labels),
                    ("stored_curve", args.stored_curve),
                    *((("fold_structures", args.fold_structures),)
                      if args.fold_structures is not None else ()))},
        "bands": records, "length_matched_gain": gain_rows})
    for name, rows in (("tie_saturation", tie_rows), ("length_of_retained_set", length_rows),
                       ("plddt_by_length_bin", bin_rows), ("length_matched_share", share_rows),
                       ("length_matched_gain", gain_rows)):
        streamed.write_tsv(args.out / "tables" / f"{name}.tsv", rows)
    print(json.dumps({"raw_cells_checked": reproduced, "share_rows": len(share_rows)}))


if __name__ == "__main__":
    main()
