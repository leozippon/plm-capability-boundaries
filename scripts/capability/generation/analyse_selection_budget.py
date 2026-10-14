#!/usr/bin/env python3
"""Does a selector's gain grow with candidate budget? Development-only, novelty-restricted, against a TM ceiling.

For the cells carried to 500 candidates (N, Pr, C per model and domain) the first d candidates of each cell,
in generation-seed order, form the depth-d pool (d = 100 ... 500). Inside the novelty-restricted stratum
(identity_to_wild_type < 20, at least 25 such candidates) every selector of the selection-factorial registry
picks the top k, and its gain is the selected-set TM minus the restricted pool mean.

THE TM ORACLE IS NOT A SELECTOR. ``tm_oracle`` picks the top k by the realised TM-score, the primary endpoint,
which is prohibited for any method. It exists only to measure the ceiling: the headroom a perfect recogniser of
the endpoint would have in the same pool. The captured fraction is a selector's gain over the oracle's gain.

The budget-scaling comparison is development-only. The sealed evaluation domains carry 200 and 100 candidates by
design, not 500, so this comparison cannot be repeated on them; the captured fraction itself is computable at
their depths and is what the confirmatory secondary reports.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyse_selection_factorial as sf  # noqa: E402
from src.capability.context.family_potts import load_family_model  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation import fold_recapitulation as fr  # noqa: E402

SCHEMA = "selection_budget_v1"
DEPTHS = (100, 200, 300, 400, 500)
KS = (1, 10, 25)
ORACLE = "tm_oracle"
ORACLE_STATEMENT = (
    "tm_oracle selects the top k candidates by the realised TM-score, the primary endpoint, which is prohibited "
    "for any method. It is not a selector, not a proposed method and not available to any design procedure; it "
    "exists only to measure the ceiling (the headroom a perfect recogniser of the endpoint would have in the same "
    "pool). identity_oracle, listed with the selectors, is likewise a diagnostic reference that needs the target "
    "sequence.")
DEVELOPMENT_ONLY = (
    "The budget-scaling comparison is development-only. The sealed evaluation domains carry 200 candidates in the "
    "primary-family cells and 100 in the descriptive frontier cells, not 500, so budget scaling cannot be tested "
    "on them; only the captured fraction at their depths can be.")
DETERMINISM_NOTE = (
    "The oracle cannot be exploiting per-fold stochasticity if the endpoint is effectively deterministic per "
    "sequence: the bound below is the spread of TM between records of the same (domain, sequence).")


def tm_oracle_top(tm: np.ndarray, values: dict, k: int) -> dict:
    """Metric means of the k candidates with the highest realised TM. A ceiling measurement, never a selector."""

    return sf.topk_means(tm, values, k)


def determinism(records) -> dict:
    """How far TM varies between records of one (domain, sequence): the evidence that the endpoint is deterministic."""

    tms = defaultdict(list)
    for r in records:
        tms[(r["domain"], r["sequence"])].append(r["tm_score"])
    spreads = sorted(float(np.ptp(v)) for v in tms.values() if len(v) > 1 and np.ptp(v) > 0.0)
    return {"note": DETERMINISM_NOTE, "n_records": len(records), "n_distinct_domain_sequence_pairs": len(tms),
            "n_pairs_with_more_than_one_record": sum(len(v) > 1 for v in tms.values()),
            "n_pairs_with_two_distinct_tm_values": len(spreads),
            "max_abs_tm_difference": spreads[-1] if spreads else 0.0,
            "median_abs_tm_difference_among_those": float(np.median(spreads)) if spreads else 0.0}


def cell_depths(rows, model, donor, permutation_seed) -> dict:
    """Per depth: restricted counts, pool mean/max TM, and per k each selector's gain, the oracle's, the fractions."""

    rows = sorted(rows, key=lambda r: r["seed"])
    scores, values = sf.pool_scores(model, donor, permutation_seed, rows)
    out = {}
    for depth in DEPTHS:
        if depth > len(rows):
            continue
        keep = np.flatnonzero(values["identity_to_wild_type"][:depth] < sf.NOVELTY_IDENTITY)
        entry = {"n_pool": depth, "n_admissible": int(keep.size), "restricted": None}
        if keep.size >= sf.MIN_ADMISSIBLE:
            v = {m: values[m][keep] for m in sf.METRICS}
            pool_mean = float(v["tm_score"].mean())
            restricted = {"pool_mean_tm": pool_mean, "pool_max_tm": float(v["tm_score"].max()), "k": {}}
            for k in KS:
                oracle = tm_oracle_top(v["tm_score"], v, k)
                oracle_gain = oracle["tm_score"] - pool_mean
                sels = {}
                for name, score in scores.items():
                    top = sf.topk_means(score[keep], v, k)
                    gain = top["tm_score"] - pool_mean
                    sels[name] = {"gain": gain, "selected_tm": top["tm_score"],
                                  "identity_to_wild_type": top["identity_to_wild_type"],
                                  "captured_fraction": gain / oracle_gain if oracle_gain > 0 else None}
                restricted["k"][str(k)] = {ORACLE: {"gain": oracle_gain, "selected_tm": oracle["tm_score"],
                                                    "identity_to_wild_type": oracle["identity_to_wild_type"]},
                                           "selectors": sels}
            entry["restricted"] = restricted
        out[str(depth)] = entry
    return out


def summarise_arm(cells) -> dict:
    """Per depth and k: mean gains over model-by-domain cells; captured fraction is the ratio of the mean gains.

    ``all_valid`` uses every cell with a restricted value at that depth; ``matched`` uses only cells valid at
    every depth, so the trend is not confounded by cells dropping out as the threshold is crossed.
    """

    valid_everywhere = [c for c in cells if all(c["depths"][str(d)]["restricted"] for d in DEPTHS if str(d) in c["depths"])]
    out = {}
    for tag, group in (("all_valid", cells), ("matched", valid_everywhere)):
        out[tag] = {}
        for depth in DEPTHS:
            used = [c["depths"][str(depth)]["restricted"] for c in group
                    if str(depth) in c["depths"] and c["depths"][str(depth)]["restricted"]]
            block = {"n_cells": len(used), "n_cells_total": len(group)}
            if used:
                block["mean_pool_tm"] = float(np.mean([u["pool_mean_tm"] for u in used]))
                block["mean_pool_max_tm"] = float(np.mean([u["pool_max_tm"] for u in used]))
                block["k"] = {}
                for k in KS:
                    og = float(np.mean([u["k"][str(k)][ORACLE]["gain"] for u in used]))
                    # a selector missing in any used cell (no admissible cross-domain donor) is not summarised
                    names = set.intersection(*(set(u["k"][str(k)]["selectors"]) for u in used))
                    sels = {}
                    for name in sorted(names):
                        g = float(np.mean([u["k"][str(k)]["selectors"][name]["gain"] for u in used]))
                        sels[name] = {"mean_gain": g, "captured_fraction": g / og if og > 0 else None}
                    block["k"][str(k)] = {ORACLE: {"mean_gain": og}, "selectors": sels}
            out[tag][str(depth)] = block
    return out


def analyse(dev_records, pool_records, family_models, permutation_seed) -> dict:
    keys = {(r["model"], r["domain"], r["arm"], r["lambda_fields"], r["lambda_coupling"]) for r in pool_records}
    by_cell = defaultdict(list)
    for r in (*dev_records, *pool_records):
        key = (r["model"], r["domain"], r["arm"], r["lambda_fields"], r["lambda_coupling"])
        if key in keys:
            by_cell[key].append(r)
    ids = [r["id"] for rows in by_cell.values() for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("a candidate appears in both the development and the pool records")
    donors = sf.cross_domain_donors(family_models)
    cells = []
    for (model, domain, arm, lf, lc), rows in sorted(by_cell.items(), key=str):
        cells.append({"model": model, "domain": domain, "arm": arm, "lambda_fields": lf, "lambda_coupling": lc,
                      "n_candidates": len(rows),
                      "depths": cell_depths(rows, family_models[domain], donors[domain], permutation_seed)})
    arms = sorted({c["arm"] for c in cells})
    return {"schema": SCHEMA, "oracle_statement": ORACLE_STATEMENT, "oracle": ORACLE,
            "development_only": DEVELOPMENT_ONLY,
            "definition": "within identity_to_wild_type < 20 (at least 25 candidates); gain = selected-set TM minus "
                          "restricted pool mean; captured fraction = selector gain / oracle gain, and in the "
                          "summaries the ratio of the cross-cell mean gains; depth d = the first d candidates of a "
                          "cell in generation-seed order",
            "depths": list(DEPTHS), "ks": list(KS), "selectors": [*sf.SELECTORS], "permutation_seed": permutation_seed,
            "determinism": determinism([*dev_records, *pool_records]),
            "summary": {arm: summarise_arm([c for c in cells if c["arm"] == arm]) for arm in arms},
            "cells": cells}


def render(result: dict, selectors=("potts_full", "potts_fields", "potts_coupling", "coupling_permuted")) -> str:
    lines = [ORACLE_STATEMENT, DEVELOPMENT_ONLY, f"determinism: {result['determinism']}"]
    for arm, tags in result["summary"].items():
        for tag in ("all_valid", "matched"):
            lines.append(f"== arm {arm} ({tag}); k=10 restricted; mean gain over cells, captured = ratio of means ==")
            for depth, b in tags[tag].items():
                if not b.get("k"):
                    lines.append(f"depth {depth}: no cell with a restricted value")
                    continue
                k10 = b["k"]["10"]
                cols = "  ".join(f"{n} {k10['selectors'][n]['mean_gain']:+.4f} ({k10['selectors'][n]['captured_fraction']:.2f})"
                                 for n in selectors if k10["selectors"][n]["captured_fraction"] is not None)
                lines.append(f"depth {depth} n={b['n_cells']}/{b['n_cells_total']} pool {b['mean_pool_tm']:.4f} "
                             f"max {b['mean_pool_max_tm']:.4f}  {ORACLE} {k10[ORACLE]['mean_gain']:+.4f}  {cols}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", type=Path, required=True, help="development per_sequence.jsonl (the first 100 per cell)")
    ap.add_argument("--pool-records", type=Path, required=True, help="expanded-pool per_sequence.jsonl")
    ap.add_argument("--references", type=Path, required=True, help="directory of family-reference <DOMAIN>.npz/.json")
    ap.add_argument("--out", type=Path, required=True, help="output JSON artefact")
    ap.add_argument("--permutation-seed", type=int, required=True,
                    help="the declared permuted_coupling seed of the generation protocol")
    args = ap.parse_args(argv)
    fr.require_asset(args.records, "--records")
    fr.require_asset(args.pool_records, "--pool-records")
    fr.require_asset(args.references, "--references", directory=True)
    dev, pool = sf.load_records(args.records), sf.load_records(args.pool_records)
    models = {d: load_family_model(d, args.references) for d in sorted({r["domain"] for r in pool})}
    result = analyse(dev, pool, models, args.permutation_seed)
    result["inputs"] = {"records": str(args.records), "pool_records": str(args.pool_records),
                        "references": str(args.references)}
    write_json(args.out, result)
    print(render(result))
    print(f"sha256 {sha256_file(args.out)}  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
