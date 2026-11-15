#!/usr/bin/env python3
"""What the instrument's mean CA-pLDDT does with sequence length, in the retained free-length sets.

A comparison of mean pLDDT between two sets whose lengths differ is partly a
comparison of their lengths, so the shape of the dependence decides which way a
length gap can bias a gap in confidence. This producer measures that shape on
every retained set that carries a length and a mean CA-pLDDT per sequence and in
which length is free to vary within one generator: the objective-tuned cohort's
ProtGPT3 base and tuned arms and its ProGen2 pair, the 768-residue companion
cohort's naturally terminated base and tuned sequences, the two ProLLaMA stages'
pools, and the 17 checkpoints of the composition-shuffle census with the shuffled
copy of every one of their sequences as a control. Whole natural Swiss-Prot
entries (the E14 cohort's natural comparator, length-matched to its generated attempts)
and those generated attempts are read from the recovered fold indexes.

For each set it reports the mean in fixed length bins, the bin of lowest mean among
30-149 residues and how often a bootstrap over sequences puts it in each bin, and
three bootstrap contrasts: the mean at 200-299 residues minus the mean at 60-99
(the trough), and the mean below 50 residues minus the trough. Nothing here is
fitted: the bins, the candidate range and the minimum bin count are declared below
before any number is read.

Reads retained tables only; computes no fold and loads no model.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation.length_matching import (  # noqa: E402
    EVALUATION_FOLD_INDEXES,
    EVALUATION_RECORD,
    read_evaluation_folds,
)

SEED = 20261114
DRAWS = 2000
MIN_BIN_N = 20
MIN_SHORT_N = 8
BINS = ((0, 29, "<30"), (30, 39, "30-39"), (40, 49, "40-49"), (50, 59, "50-59"),
        (60, 69, "60-69"), (70, 79, "70-79"), (80, 89, "80-89"), (90, 99, "90-99"),
        (100, 109, "100-109"), (110, 119, "110-119"), (120, 129, "120-129"),
        (130, 149, "130-149"), (150, 169, "150-169"), (170, 199, "170-199"),
        (200, 249, "200-249"), (250, 299, "250-299"), (300, 399, "300-399"),
        (400, 400, "400"), (401, 10**9, ">400"))
LABELS = [b[2] for b in BINS]
LOWS = np.array([b[0] for b in BINS])
CANDIDATES = [i for i, b in enumerate(BINS) if 30 <= b[0] and b[1] <= 149]
TROUGH = [i for i, b in enumerate(BINS) if 60 <= b[0] and b[1] <= 99]
HIGH = [i for i, b in enumerate(BINS) if 200 <= b[0] and b[1] <= 299]
SHORT = [i for i, b in enumerate(BINS) if b[1] < 50]

OT_DIR = "results/external_baseline_analysis/length_adjusted_contrasts_20261110/eighteen_shard"
PINNED = {
    "fold_table": (f"{OT_DIR}/checkpoint_structures_18of68.tsv",
                   "2fd2ee4b91b67ef7bcef91241ccfddc61353b6f51e5b788270be8a2df8aabf06"),
    "companion_per_structure": ("results/long_cap_termination_20261110/analysis/per_structure_joined.tsv",
                                "b5e7713d3a2be8f5b5430f1d5c562199f1a77371c8e824b97483f3cfa4c18a20"),
    "stage_pair_index": ("results/external_baseline/20261008025237_2306ce46fb82/"
                         "condgen_stage_pair_structures_cpu/stage_pair_structure_index.jsonl",
                         "556a5d2fe4f89f5f2f6bcf6ed7106ad9dc8af481ce01502232d81fc07b3cd0f5"),
    "structure_census": ("results/R6/structure_census_20260920/structure_census.json",
                         "578ed1663648f9ae20784b896390ed3626b8a6874494629d1f06e5ffe8531694"),
    "evaluation_record": (EVALUATION_RECORD,
                          "5a7535a7f11d3ecd9f87d67b272fdec8530c25bc963a9b7ee203d14e6ce3b7c5"),
}
CENSUS_PREDICTIONS = ("results/R6/generation_evidence_20260905/structure/esmfold2_20260920012718/"
                      "s48_*/predictions.jsonl")


def bin_index(lengths: np.ndarray) -> np.ndarray:
    return np.searchsorted(LOWS, lengths, side="right") - 1


def bin_table(name: str, kind: str, lengths: np.ndarray, values: np.ndarray) -> list[dict]:
    index = bin_index(lengths)
    rows = []
    for i, label in enumerate(LABELS):
        x = values[index == i]
        if x.size:
            rows.append({"corpus": name, "kind": kind, "bin": label, "n": int(x.size),
                         "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if x.size > 1 else None})
    return rows


def _means(index: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(index, minlength=len(BINS))
    sums = np.bincount(index, weights=values, minlength=len(BINS))
    return sums, counts


def region_mean(sums: np.ndarray, counts: np.ndarray, region: list[int]) -> float:
    n = counts[region].sum()
    return float(sums[region].sum() / n) if n else float("nan")


def locate_minimum(sums: np.ndarray, counts: np.ndarray) -> int | None:
    eligible = [i for i in CANDIDATES if counts[i] >= MIN_BIN_N]
    if not eligible:
        return None
    return eligible[int(np.argmin([sums[i] / counts[i] for i in eligible]))]


def analyse_corpus(name: str, kind: str, lengths: np.ndarray, values: np.ndarray,
                   rng: np.random.Generator) -> dict:
    index = bin_index(lengths)
    sums, counts = _means(index, values)
    out = {"kind": kind, "n": int(lengths.size), "n_below_50": int(counts[SHORT].sum()),
           "minimum_bin": None, "minimum_bin_bootstrap_share": {}}
    point = locate_minimum(sums, counts)
    argmin_draws, rise, short = [], [], []
    for _ in range(DRAWS):
        pick = rng.integers(0, lengths.size, lengths.size)
        s, c = _means(index[pick], values[pick])
        found = locate_minimum(s, c)
        if found is not None:
            argmin_draws.append(found)
        trough = region_mean(s, c, TROUGH)
        rise.append(region_mean(s, c, HIGH) - trough)
        if c[SHORT].sum() >= 1:
            short.append(region_mean(s, c, SHORT) - trough)

    def interval(draws):
        draws = np.asarray([d for d in draws if not np.isnan(d)])
        return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))] if draws.size else None

    trough_mean = region_mean(sums, counts, TROUGH)
    out["trough_60_99"] = {"n": int(counts[TROUGH].sum()), "mean": trough_mean}
    if point is not None:
        out["minimum_bin"] = LABELS[point]
        shares = {LABELS[i]: argmin_draws.count(i) / len(argmin_draws) for i in set(argmin_draws)}
        out["minimum_bin_bootstrap_share"] = dict(sorted(shares.items(), key=lambda kv: -kv[1]))
    high_n = int(counts[HIGH].sum())
    if high_n >= MIN_BIN_N and counts[TROUGH].sum() >= MIN_BIN_N:
        out["rise_200_299_minus_trough"] = {
            "n_high": high_n, "estimate": region_mean(sums, counts, HIGH) - trough_mean,
            "ci95": interval(rise)}
    if counts[SHORT].sum() >= MIN_SHORT_N and counts[TROUGH].sum() >= MIN_BIN_N:
        out["short_below_50_minus_trough"] = {
            "n_short": int(counts[SHORT].sum()),
            "estimate": region_mean(sums, counts, SHORT) - trough_mean, "ci95": interval(short)}
    return out


def load_corpora(repo: Path) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    fold = pd.read_csv(repo / PINNED["fold_table"][0], sep="\t", comment="#")
    arm = fold["arm"]
    corpora = []
    for name, mask in (
            ("objective-tuned ProtGPT3 base arms", arm.isin(["protgpt3-112m", "protgpt3-1.3b", "protgpt3-10b"])),
            ("objective-tuned ProtGPT3 tuned arms", arm.isin(["protgpt3-112m-dpo", "protgpt3-1.3b-dpo", "protgpt3-10b-dpo"])),
            ("objective-tuned ProGen2 pair", arm.str.startswith("progen2"))):
        part = fold[mask]
        corpora.append((name, "free-length generated", part["length_residues"].to_numpy(),
                        part["mean_ca_plddt_0_100"].to_numpy()))
    companion = pd.read_csv(repo / PINNED["companion_per_structure"][0], sep="\t")
    for name, population in (("companion 10B base, naturally terminated", "base-natural"),
                             ("companion 10B tuned, naturally terminated", "tuned-natural")):
        part = companion[companion["population"] == population]
        corpora.append((name, "free-length generated", part["L"].to_numpy(), part["plddt"].to_numpy()))
    stages = [json.loads(line) for line in
              (repo / PINNED["stage_pair_index"][0]).read_text(encoding="utf-8").splitlines() if line.strip()]
    for stage in ("stage_1", "stage_2"):
        part = [r for r in stages if r["stage"] == stage]
        corpora.append((f"ProLLaMA {stage.replace('_', ' ')} pool", "free-length generated",
                        np.array([r["length"] for r in part]),
                        np.array([r["mean_ca_plddt"] for r in part])))
    generated, shuffled = [], []
    for path in sorted(glob.glob(str(repo / CENSUS_PREDICTIONS))):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            structure = record.get("structure") or {}
            if structure.get("mean_ca_plddt") is None:
                continue
            target = generated if record["role"] == "generation" else shuffled
            target.append((record["length"], structure["mean_ca_plddt"]))
    folds = read_evaluation_folds(repo)
    for name, kind, role in (("E14 generated attempts, nine arms", "free-length generated", "generated"),
                             ("E14 whole natural Swiss-Prot entries, length-matched",
                              "natural whole entries", "natural")):
        part = folds[folds["role"] == role]
        corpora.append((name, kind, part["length"].to_numpy(), part["mean_ca_plddt"].to_numpy()))
    corpora.append(("census generated sequences, 17 checkpoints", "free-length generated",
                    np.array([x[0] for x in generated]), np.array([x[1] for x in generated])))
    corpora.append(("census composition shuffles of those sequences", "composition-shuffled control",
                    np.array([x[0] for x in shuffled]), np.array([x[1] for x in shuffled])))
    return corpora


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path,
                        default=Path("results/extensions/length_confound_audit_20261114"))
    args = parser.parse_args()
    repo = args.repo
    inputs = {}
    for name, (relative, expected) in PINNED.items():
        digest = sha256_file(repo / relative)
        if digest != expected:
            raise SystemExit(f"{relative}: sha256 {digest} is not the retained {expected}")
        inputs[name] = {"path": relative, "sha256": digest}
    inputs["evaluation_fold_indexes"] = [
        {"path": relative, "sha256": sha256_file(repo / relative)} for relative in EVALUATION_FOLD_INDEXES]
    inputs["census_predictions"] = [
        {"path": str(Path(p).relative_to(repo)), "sha256": sha256_file(Path(p))}
        for p in sorted(glob.glob(str(repo / CENSUS_PREDICTIONS)))]
    if len(inputs["census_predictions"]) != 17:
        raise SystemExit("the composition-shuffle census should carry 17 checkpoint prediction files")

    rng = np.random.default_rng(SEED)
    table, corpora = [], {}
    for name, kind, lengths, values in load_corpora(repo):
        if lengths.size != values.size or np.isnan(values).any():
            raise SystemExit(f"{name}: lengths and confidences do not line up")
        table += bin_table(name, kind, lengths, values)
        corpora[name] = analyse_corpus(name, kind, lengths, values, rng)
    result = {"seed": SEED, "bootstrap_draws": DRAWS, "bins": LABELS,
              "candidate_bins_for_minimum": [LABELS[i] for i in CANDIDATES],
              "min_bin_n": MIN_BIN_N, "min_short_n": MIN_SHORT_N,
              "trough_bins": [LABELS[i] for i in TROUGH], "high_bins": [LABELS[i] for i in HIGH],
              "corpora": corpora, "inputs": inputs}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables").mkdir(exist_ok=True)
    write_json(args.out / "length_dependence_of_confidence.json", result)
    pd.DataFrame(table).to_csv(args.out / "tables/confidence_by_length_bin.tsv", sep="\t",
                               index=False, float_format="%.6g")
    for name, entry in corpora.items():
        top = list(entry["minimum_bin_bootstrap_share"].items())[:2]
        rise = entry.get("rise_200_299_minus_trough")
        print(f"{name:52s} n={entry['n']:5d} lowest bin {entry['minimum_bin']} {top}"
              + (f" rise {rise['estimate']:+.1f} {np.round(rise['ci95'], 1).tolist()}" if rise else ""))


if __name__ == "__main__":
    main()
