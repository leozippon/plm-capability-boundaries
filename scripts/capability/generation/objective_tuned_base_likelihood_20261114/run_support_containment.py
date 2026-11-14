#!/usr/bin/env python3
"""Support containment of the tuned arms in their base model, per the twenty-first amendment of 2026-11-14.

Reads the recomputed cross-direction cells (the base checkpoint's NLL of the
base pool and of the tuned pool), the 2026-11-11 own-model cells for the entropy
change, and the frozen fold cohort for lengths. Reports per pair and length
stratum the mixture contrast through ``analyse_cross_likelihood.contrast``
(unchanged), the entropy change, and the share of tuned samples outside the
base pool's own base-model NLL range. Nothing is scored or generated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/capability/generation"))

import analyse_cross_likelihood as acl  # noqa: E402

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation import objective_tuned_analysis as ota  # noqa: E402

EDGES = ((16, 128), (129, 256), (257, 512), (513, 10**9))
MIN_N = 30
SEED = 20261114
DRAWS = 2000
INSIDE_UPPER, OUTSIDE_LOWER = 0.01, 0.05


def load(path: Path, lengths: dict[str, int]) -> dict[str, tuple[int, float]]:
    rows = ota.read_own_likelihood(path)
    for digest, (count, _) in rows.items():
        if digest not in lengths or lengths[digest] != count:
            raise SystemExit(f"{path}: {digest} is not in the cohort or its token count is not its length")
    return rows


def arrays(rows, digests):
    return (np.array([lengths_of[d] for d in digests]),
            np.array([rows[d][1] / rows[d][0] for d in digests]))


def outside(base: np.ndarray, tuned: np.ndarray, rng) -> dict:
    low, high = float(base.min()), float(base.max())
    below, above = tuned < low, tuned > high
    flags = (below | above).astype(float)
    draws = [flags[rng.integers(0, flags.size, flags.size)].mean() for _ in range(DRAWS)]
    ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    reading = ("inside" if ci[1] <= INSIDE_UPPER else "outside" if ci[0] > OUTSIDE_LOWER
               else "partly_outside")
    return {"n_tuned": int(tuned.size), "n_base": int(base.size), "range": [low, high],
            "n_below_range": int(below.sum()), "n_above_range": int(above.sum()),
            "fraction": float(flags.mean()), "ci": ci,
            "exchangeable_expectation": 2.0 / (base.size + 1), "reading": reading}


lengths_of: dict[str, int] = {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cross", type=Path, required=True)
    parser.add_argument("--own-h200", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    for line in args.cohort.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            lengths_of[record["id"]] = len(record["sequence"])

    result: dict = {"seed": SEED, "draws": DRAWS, "edges": [list(e) for e in EDGES], "min_n": MIN_N,
                    "thresholds": {"inside_upper": INSIDE_UPPER, "outside_lower": OUTSIDE_LOWER},
                    "pairs": {}}
    inputs: dict = {"cohort": {"sha256": sha256_file(args.cohort)}}
    for pair in (*ota.PRIMARY_PAIRS, *ota.SECONDARY_PAIRS):
        b, t = pair.base, pair.tuned
        files = {"base_on_base": args.cross / f"{b}__on__{b}.jsonl",
                 "base_on_tuned": args.cross / f"{b}__on__{t}.jsonl",
                 "h200_base": args.own_h200 / f"{b}__on__{b}.jsonl",
                 "h200_tuned": args.own_h200 / f"{t}__on__{t}.jsonl"}
        rows = {name: load(path, lengths_of) for name, path in files.items()}
        inputs[pair.tuned] = {name: sha256_file(path) for name, path in files.items()}
        bd, td = sorted(rows["base_on_base"]), sorted(rows["base_on_tuned"])
        if set(rows["h200_base"]) != set(bd) or set(rows["h200_tuned"]) != set(td):
            raise SystemExit(f"{pair.tuned}: the two hosts' score files do not cover the same digests")
        lb, xb = arrays(rows["base_on_base"], bd)
        lt, xt = arrays(rows["base_on_tuned"], td)
        _, hb = arrays(rows["h200_base"], bd)
        _, ht = arrays(rows["h200_tuned"], td)
        diff = np.abs(hb - xb)
        rng = np.random.default_rng(SEED)
        entry = {
            "n_base": len(bd), "n_tuned": len(td),
            "host_agreement_base_pool": {"max_abs_diff": float(diff.max()),
                                         "n_within_1e-6": int((diff <= 1e-6).sum()),
                                         "n_above_1e-3": int((diff > 1e-3).sum())},
            "pooled": {
                "raw_difference_of_means": float(xt.mean() - xb.mean()),
                "mixture": acl.contrast(xt, lt, xb, lb, rng),
                "entropy_change_h200": acl.contrast(ht, lt, hb, lb, rng),
                "outside_whole_pool": outside(xb, xt, rng)},
            "strata": {}}
        for low, high in EDGES:
            mb, mt = (lb >= low) & (lb <= high), (lt >= low) & (lt <= high)
            if min(mb.sum(), mt.sum()) < MIN_N:
                continue
            entry["strata"][f"{low}_{high}"] = {
                "n": [int(mt.sum()), int(mb.sum())],
                "mixture": acl.contrast(xt[mt], lt[mt], xb[mb], lb[mb], rng),
                "entropy_change_h200": acl.contrast(ht[mt], lt[mt], hb[mb], lb[mb], rng),
                "outside_within_stratum": outside(xb[mb], xt[mt], rng)}
        result["pairs"][pair.tuned] = entry
    result["inputs"] = inputs
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "support_containment.json", result)
    for tuned, e in result["pairs"].items():
        o = e["pooled"]
        print(tuned, round(o["raw_difference_of_means"], 4), o["outside_whole_pool"]["n_below_range"],
              o["outside_whole_pool"]["n_above_range"], o["outside_whole_pool"]["reading"])


if __name__ == "__main__":
    main()
