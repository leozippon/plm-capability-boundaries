#!/usr/bin/env python3
"""Does a base model's likelihood prefer repetitive sequence? The twentieth amendment's second reading.

Within each base arm's full scored pool, Spearman correlation of the base
model's own mean NLL per residue token with the distinct-5-mer fraction and
with the longest single-residue run, overall and within length strata, with 95%
percentile bootstraps over sequences at a declared seed. For context the means
of both statistics in each base arm, its tuned sibling, and the base arm's top
quarter under the distinct-5-mer rule. Reads the own-model score cells and the
frozen fold cohort's sequences; nothing is scored or generated.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.forcing.backbone_cohort import distinct_shingle_fraction  # noqa: E402
from src.capability.generation import objective_tuned_analysis as ota  # noqa: E402

SEED = 20261114
DRAWS = 2000
EDGES = ((16, 128), (129, 256), (257, 512), (513, 10**9))
MIN_N = 30


def longest_run(sequence: str) -> int:
    return max((len(list(group)) for _, group in itertools.groupby(sequence)), default=0)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def with_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    point = spearman(x, y)
    draws = []
    for _ in range(DRAWS):
        index = rng.integers(0, x.size, x.size)
        draws.append(spearman(x[index], y[index]))
    draws = np.asarray(draws)
    draws = draws[~np.isnan(draws)]
    return {"n": int(x.size), "spearman": point,
            "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--likelihood", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sequences = {}
    for line in args.cohort.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            sequences[record["id"]] = str(record["sequence"])

    def pool(arm: str):
        path = args.likelihood / f"{arm}__on__{arm}.jsonl"
        table = ota.read_own_likelihood(path)
        missing = [digest for digest in table if digest not in sequences]
        if missing:
            raise SystemExit(f"{arm}: {len(missing)} scored sequences are not in the fold cohort")
        digests = sorted(table)
        seqs = [sequences[digest] for digest in digests]
        if any(len(seq) != table[digest][0] for digest, seq in zip(digests, seqs)):
            raise SystemExit(f"{arm}: a score's token count is not its sequence's length")
        return {
            "digest": digests,
            "length": np.array([len(seq) for seq in seqs]),
            "nll": np.array([table[digest][1] / table[digest][0] for digest in digests]),
            "distinct5": np.array([distinct_shingle_fraction(seq) for seq in seqs]),
            "run": np.array([longest_run(seq) for seq in seqs], dtype=float),
            "path": path,
        }

    out: dict = {"seed": SEED, "draws": DRAWS, "length_strata": [list(e) for e in EDGES],
                 "min_n": MIN_N, "arms": {}}
    inputs = {"cohort": {"path": str(args.cohort), "sha256": sha256_file(args.cohort)}}
    for pair in (*ota.PRIMARY_PAIRS, *ota.SECONDARY_PAIRS):
        base, tuned = pool(pair.base), pool(pair.tuned)
        inputs[pair.base] = {"sha256": sha256_file(base["path"])}
        rng = np.random.default_rng(SEED)
        entry = {"n": int(base["nll"].size), "tuned": pair.tuned,
                 "overall": {"distinct5": with_ci(base["nll"], base["distinct5"], rng),
                             "longest_run": with_ci(base["nll"], base["run"], rng)},
                 "within_length_stratum": {}}
        for low, high in EDGES:
            mask = (base["length"] >= low) & (base["length"] <= high)
            if mask.sum() >= MIN_N:
                entry["within_length_stratum"][f"{low}_{high}"] = {
                    "distinct5": with_ci(base["nll"][mask], base["distinct5"][mask], rng),
                    "longest_run": with_ci(base["nll"][mask], base["run"][mask], rng)}
        order = sorted(range(base["nll"].size),
                       key=lambda i: (-base["distinct5"][i], base["digest"][i]))
        top = order[: base["nll"].size // 4]
        entry["context_means"] = {
            name: {"distinct5": float(arm["distinct5"][idx].mean()),
                   "longest_run": float(arm["run"][idx].mean()),
                   "mean_nll_under_own_model": float(arm["nll"][idx].mean()),
                   "mean_length": float(arm["length"][idx].mean()), "n": int(arm["nll"][idx].size)}
            for name, arm, idx in (
                ("base", base, slice(None)), ("tuned", tuned, slice(None)),
                ("base_top_quarter_by_distinct5", base, top))}
        out["arms"][pair.base] = entry
    out["inputs"] = inputs
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "likelihood_repetitiveness.json", out)
    print(json.dumps({arm: {k: round(v["spearman"], 3) for k, v in e["overall"].items()}
                      for arm, e in out["arms"].items()}, indent=1))


if __name__ == "__main__":
    main()
