#!/usr/bin/env python3
"""The filter controls read at three selection strengths, per the amendment of 2026-11-14 (twentieth).

Retains the top 1/2, 1/4 and 1/8 of each base pool (k = 2, 4, 8) for the three
explicit statistics, the base-likelihood rule and its summed sensitivity, over
the primary and the secondary pairs, through the analysis module's own
``explicit_statistics_control``. The k = 4 points use the families already
declared for those rules and must reproduce the stored results exactly; the
other points read the same families' seed blocks. Reads completed artefacts
only.
"""

from __future__ import annotations

import argparse
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

FRACTIONS = (0.5, 0.25, 0.125)
PRIMARY_K = 4


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


streamed = _load(REPO_ROOT / "scripts/capability/generation/objective_tuned_endpoints_20261112"
                 / "run_stream_endpoints.py", "objective_tuned_stream_endpoints")

RULES = (*ota.FILTER_RULE_NAMES, *ota.LIKELIHOOD_RULE_NAMES)


def match_point(by_k: dict[int, dict[str, Any]]) -> int | None:
    """Smallest k whose band is not wholly below one; ``None`` if no point is."""

    for k in sorted(by_k):
        cell = by_k[k]
        if cell["status"] == "resolved" and cell["finding"] != ota.BOUNDED_NEGATIVE:
            return k
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--neighbour", type=Path, required=True)
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--fold-structures", type=Path, default=None)
    parser.add_argument("--stream-labels", type=Path, required=True)
    parser.add_argument("--kmer-background", type=Path, required=True)
    parser.add_argument("--likelihood", type=Path, required=True)
    parser.add_argument("--stored-primary", type=Path, required=True)
    parser.add_argument("--stored-secondary", type=Path, required=True)
    parser.add_argument("--stored-likelihood", type=Path, required=True,
                        help="base_likelihood_control.json of the nineteenth amendment's run")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base_arms = sorted({pair.base for pair in (*ota.PRIMARY_PAIRS, *ota.SECONDARY_PAIRS)})
    likelihood = ota.merge_likelihood({
        arm: ota.read_own_likelihood(args.likelihood / f"{arm}__on__{arm}.jsonl")
        for arm in base_arms})
    one_stream, worst = streamed.pooled.build_table(args.neighbour, args.composition, args.cohort,
                                                    args.fold_structures)
    table = streamed.split_streams(one_stream, streamed.read_labels(args.stream_labels))
    counts = gc.load_kmer_counts(args.kmer_background, aot.KMER_ORDERS)
    shift = aot.identity_shift(table, ota.PRIMARY_PAIRS)
    primary = aot.stratified_contrast(table, ota.PRIMARY_PAIRS, ota.PRIMARY_FAMILY, shift=shift)
    secondary = aot.stratified_contrast(table, ota.SECONDARY_PAIRS, ota.SECONDARY_FAMILY)

    sets = (
        ("primary", ota.PRIMARY_PAIRS, primary, ota.CONTROL_FAMILIES,
         ota.LIKELIHOOD_CONTROL_FAMILIES),
        ("secondary", ota.SECONDARY_PAIRS, secondary, ota.SECONDARY_CONTROL_FAMILIES,
         ota.SECONDARY_LIKELIHOOD_CONTROL_FAMILIES),
    )
    # cells[rule][key][k] = summary
    cells: dict[str, dict[str, dict[int, dict[str, Any]]]] = {rule: {} for rule in RULES}
    controls: dict[str, Any] = {}
    for label, pairs, contrast, statistical, liked in sets:
        for fraction in FRACTIONS:
            k = round(1 / fraction)
            for families, extra in ((statistical, None), (liked, likelihood)):
                control = aot.explicit_statistics_control(
                    table, pairs, counts, contrast, families, extra, fraction)
                controls[f"{label}|k{k}|{'likelihood' if extra else 'statistical'}"] = {
                    rule: {"family": block["family"],
                           "resolved_family_size": block["bands"]["resolved_family_size"],
                           "critical_value": block["bands"].get("critical_value"),
                           "retained": {pair: {stream: s["n_retained"] for stream, s in per.items()}
                                        for pair, per in block["selection"].items()}}
                    for rule, block in control["rules"].items()}
                for rule, block in control["rules"].items():
                    for key, column in block["bands"]["columns"].items():
                        interval = column.get("interval") or [None, None]
                        cells[rule].setdefault(key, {})[k] = {
                            "status": column["status"], "share": column.get("point"),
                            "lo": interval[0], "hi": interval[1],
                            "finding": (column.get("statistics_reproduce_the_gain") or {}).get("finding"),
                            "reason": column.get("reason"),
                            "gain": (column.get("gains") or {}).get("tuned_minus_base_gain"),
                        }

    # The k = 4 points are the stored results, read again.
    stored = {
        **{name: columns for name, columns in aot_cols(args.stored_primary,
                                                       "explicit_statistics_control").items()},
    }
    stored_secondary = aot_cols(args.stored_secondary, "explicit_statistics_control_secondary")
    stored_likelihood = {
        rule: {key: column for block in ("base_likelihood_control_primary",
                                         "base_likelihood_control_secondary")
               for key, column in json.loads(args.stored_likelihood.read_text())[block]
               ["rules"][rule]["bands"]["columns"].items()}
        for rule in ota.LIKELIHOOD_RULE_NAMES}
    checked = 0
    for rule in RULES:
        for key, by_k in cells[rule].items():
            old = (stored_likelihood.get(rule) or {}).get(key) or stored.get(rule, {}).get(key) \
                or stored_secondary.get(rule, {}).get(key)
            if old is None:
                raise SystemExit(f"{rule} {key}: no stored k=4 cell to compare with")
            fresh = by_k[PRIMARY_K]
            if (old.get("point") != fresh["share"] or old["status"] != fresh["status"]):
                raise SystemExit(f"{rule} {key}: the k=4 point is not the stored result "
                                 f"({old.get('point')} vs {fresh['share']})")
            checked += 1

    rows = []
    for rule in RULES:
        for key, by_k in sorted(cells[rule].items()):
            pair, stratum = key.split("|")
            row = {"rule": rule, "pair": pair, "stratum": stratum}
            for k in sorted(by_k):
                c = by_k[k]
                row.update({f"k{k}_status": c["status"], f"k{k}_share": c["share"],
                            f"k{k}_lo": c["lo"], f"k{k}_hi": c["hi"], f"k{k}_finding": c["finding"]})
            row["tuned_minus_base_gain"] = by_k[PRIMARY_K]["gain"]
            row["match_k"] = match_point(by_k)
            rows.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "selection_curve.json", {
        "schema": ota.SCHEMA, "registration": "docs/objective-alignment.md, twentieth amendment "
        "of 2026-11-14", "fractions": list(FRACTIONS), "primary_k": PRIMARY_K,
        "k4_cells_checked_against_stored": checked, "cross_table_agreement": {"columns": worst},
        "inputs": {name: {"path": str(p), "sha256": sha256_file(p)} for name, p in
                   (("neighbour", args.neighbour), ("composition", args.composition),
                    ("cohort", args.cohort), ("stream_labels", args.stream_labels),
                    ("stored_primary", args.stored_primary),
                    ("stored_secondary", args.stored_secondary),
                    ("stored_likelihood", args.stored_likelihood))},
        "controls": controls,
        "cells": {rule: {key: {str(k): v for k, v in by_k.items()} for key, by_k in c.items()}
                  for rule, c in cells.items()},
        "match_k": {f"{r['rule']}|{r['pair']}|{r['stratum']}": r["match_k"] for r in rows}})
    (args.out / "tables").mkdir(exist_ok=True)
    streamed.write_tsv(args.out / "tables" / "selection_curve.tsv", rows)
    print(json.dumps({"k4_checked": checked, "rows": len(rows)}))


def aot_cols(path: Path, block: str) -> dict[str, dict[str, Any]]:
    rules = json.loads(path.read_text(encoding="utf-8"))[block]["rules"]
    return {name: rules[name]["bands"]["columns"] for name in ota.FILTER_RULE_NAMES}


if __name__ == "__main__":
    main()
