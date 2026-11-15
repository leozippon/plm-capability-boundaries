#!/usr/bin/env python3
"""The length of every set a retained pLDDT comparison compares, and what each tree does not carry.

For each retained comparison outside the objective-tuned and companion cohorts that
compares a mean CA-pLDDT across sets, this producer reports the length distribution
of the compared sets (count, median, quartiles, minimum, number below 50 residues)
beside the confidence difference the retained result claims: E14 (generated
attempts against length-matched whole natural records), E17 (the likelihood
selector's selected sets against the pool), E18 (ProLLaMA stage 2 against stage 1),
E11 (homologue-conditioned generation), the two decoding sweeps, the
composition-shuffle census and the generation main subset, and the objective-tuned
selector-competence endpoint (top ten by own likelihood against the arm's pool).

Wherever the per-sequence rows are retained, every recomputed figure is compared
with the figure the retained result states, and a mismatch stops the producer, so a
set rebuilt here is the retained set and not a lookalike: E14's generated means
(14 cells), E17's selected yields and selected-set lengths at every retained budget,
the E17 oracle's yield, the selector endpoint's pool means and gains, and every
pair length in the shuffle census. Where they are not retained, nothing is
recomputed and the gap is recorded as a verified fact at run time (the final block),
so that it flips on its own if a later change supplies what is missing.

Reads retained tables only; computes no fold and loads no model.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation.length_matching import (  # noqa: E402
    EVALUATION_FOLD_INDEXES,
    EVALUATION_RECORD,
    check_values,
    length_summary,
    read_evaluation_folds,
)

OT_DIR = "results/external_baseline_analysis/length_adjusted_contrasts_20261110/eighteen_shard"
STRUCTURE = "results/R6/generation_evidence_20260905"
PINNED = {
    "evaluation_record": (EVALUATION_RECORD,
                          "5a7535a7f11d3ecd9f87d67b272fdec8530c25bc963a9b7ee203d14e6ce3b7c5"),
    "evaluation_cohort": ("results/R6/generation_evaluation_20261008/evaluation_cohort.jsonl",
                          "dd282e3c7fc6ddcf479574608c7755f31998eb69e440bbcecf2d7d7fb3d111f2"),
    "e17_likelihood_protgpt2": (
        "results/external_baseline/20261008091454_70777906ad8c/geneval_e17_likelihood_protgpt2_gpu2/"
        "generated_likelihood.jsonl",
        "f2b6b6ef6670d30756afc2e0d32b64e92cfeb127e7f6114b9244473c9aa18012"),
    "e17_likelihood_zymctrl": (
        "results/external_baseline/20261008091454_70777906ad8c/geneval_e17_likelihood_zymctrl_gpu3/"
        "generated_likelihood.jsonl",
        "410948a18b8899a0f5de2bf97f7641aff16f165236bc9505d1bd755862b38e4a"),
    "stage_pair_index": ("results/external_baseline/20261008025237_2306ce46fb82/"
                         "condgen_stage_pair_structures_cpu/stage_pair_structure_index.jsonl",
                         "556a5d2fe4f89f5f2f6bcf6ed7106ad9dc8af481ce01502232d81fc07b3cd0f5"),
    "stage_pair_phenotype": ("results/R6/stage_pair_phenotype_20261008/stage_pair_phenotype.json",
                             "e1dd140842b6e3ff825d08eb0ad2ab9ccaf214c02b004e2a381bd803ef1a5aaa"),
    "e11_comparison": ("results/R6/e11_completed_20261011/comparison/conditioned_generation_comparison.json",
                       "96baf8c8284072ecb6bf81c10f0f5f3dbd11b92a47a7645c9edb045be3d1d4ad"),
    "decoding_sweep": ("results/R6/decoding_sweep_20261010/decoding_sweep.json",
                       "cd3d38ec358c7e7c4660d4ddd920295fdd985042447eb1c81e6a317195703b87"),
    "decoding_frontier": ("results/R6/decoding_frontier_20261010/decoding_sweep.json",
                          "21334b4dadfc3a14d4ab6f213320b6fc374056ded5c193eb7d0fe576e111f089"),
    "structure_census": ("results/R6/structure_census_20260920/structure_census.json",
                         "578ed1663648f9ae20784b896390ed3626b8a6874494629d1f06e5ffe8531694"),
    "generation_main_subset": (f"{STRUCTURE}/main_subset.jsonl",
                               "fa178339ff7fc6763e6f5e78785f3754a7e9bfe77864ead8803e240d21571cc6"),
    "fold_table": (f"{OT_DIR}/checkpoint_structures_18of68.tsv",
                   "2fd2ee4b91b67ef7bcef91241ccfddc61353b6f51e5b788270be8a2df8aabf06"),
    "selector_scores": ("results/extensions/objective_tuned_endpoints_20261112/selector/selector_score.jsonl",
                        "1f01da2eee66f9484be338f5f7f16920115dc3622f9ca0a7d2eef055b8ddcd52"),
    "selector_competence": ("results/extensions/objective_tuned_endpoints_20261112/tables/selector_competence.tsv",
                            "b39de74ee8f0fd89a48d17118070d0adcdc11ef1c8a34f1d2d9e4c6978246981"),
}
CENSUS_PREDICTIONS = f"{STRUCTURE}/structure/esmfold2_20260920012718/s48_*/predictions.jsonl"
CONTROL_TABLES = (
    "results/extensions/objective_tuned_endpoints_20261111/tables/explicit_statistics_control.tsv",
    "results/extensions/objective_tuned_endpoints_20261112/tables/explicit_statistics_control.tsv",
    "results/extensions/objective_tuned_base_likelihood_20261114/tables/base_likelihood_control.tsv",
    "results/extensions/objective_tuned_selection_curve_20261114/tables/selection_curve.tsv")
REWARD_POOLS = "results/extensions/rl_environment_provenance_20261114/pools"
E11_PANEL_FOLDS = "results/external_baseline/20261010123239_2b2e25246fdd"
E14_ORIGINAL_SHARDS = ("results/external_baseline/20261008091454_70777906ad8c/geneval_e14_fold_shard0_gpu2",
                       "results/external_baseline/20261008091454_70777906ad8c/geneval_e14_fold_shard1_gpu3")
YIELD_TOLERANCE = 1e-6
# The selector endpoint's retained means and gains were computed from this same fold table
# in single precision; the differences are rounding, 1.9e-6 at most.
SELECTOR_TOLERANCE = 1.0e-5
SELECTOR_BUDGET = 10
LENGTH_BINS = ((0, 49, "<50"), (50, 99, "50-99"), (100, 199, "100-199"), (200, 299, "200-299"),
               (300, 399, "300-399"), (400, 10**9, ">=400"))


def prefixed(prefix: str, summary: dict) -> dict:
    return {f"{prefix}_{k}": v for k, v in summary.items()}


def require(checks: dict, name: str, rows: list[dict], tolerance: float, expected: int | None = None) -> None:
    verdict = check_values(rows, tolerance, expected)
    checks[name] = verdict
    if not verdict["passed"]:
        raise SystemExit(f"{name}: a recomputed value does not match the retained one: "
                         f"{json.dumps(verdict)}")


# ---------------------------------------------------------------------- E14, E17


def e14_block(record: dict, folds: pd.DataFrame, checks: dict) -> list[dict]:
    generated = folds[folds["role"] == "generated"]
    natural = folds[folds["role"] == "natural"]
    if sorted(generated["length"]) != sorted(natural["length"]):
        raise SystemExit("E14: the generated and natural length multisets are not identical")
    rows, comparisons = [], []
    for arm, block in sorted(record["e14"]["arms"].items()):
        for stratum, entry in sorted(block["strata"].items()):
            contrast = (entry.get("contrasts") or {}).get("mean_ca_plddt")
            part = generated[(generated["arm"] == arm) & (generated["stratum"] == stratum)]
            row = {"arm": arm, "stratum": stratum, "status": entry.get("status", "estimated"),
                   **prefixed("generated_length", length_summary(part["length"]))}
            if contrast is not None:
                recomputed = float(part["mean_ca_plddt"].mean())
                comparisons.append({"abs_difference": abs(recomputed - contrast["generated_mean"]),
                                    "n_equal": len(part) == contrast["n_pairs"]})
                row.update({
                    "n_pairs": contrast["n_pairs"],
                    "retained_pair_length_difference": entry["length"]["mean_absolute_difference"],
                    "generated_mean_retained": contrast["generated_mean"],
                    "generated_mean_recomputed": recomputed,
                    "natural_mean_retained": contrast["natural_mean"],
                    "difference": contrast["difference"],
                    "difference_ci95_low": contrast["difference_ci95"][0],
                    "difference_ci95_high": contrast["difference_ci95"][1]})
            rows.append(row)
    if not all(c["n_equal"] for c in comparisons):
        raise SystemExit("E14: a generated set's size is not the retained pair count")
    require(checks, "e14_generated_means", comparisons, YIELD_TOLERANCE, expected=14)
    return rows


def e17_block(record: dict, folds: pd.DataFrame, likelihood: dict[str, Path],
              checks: dict) -> list[dict]:
    rows, selected_checks, oracle_checks = [], [], []
    for arm in ("protgpt2", "zymctrl"):
        evaluator = record["e17"]["arms"][arm]["evaluators"]["mean_ca_plddt"]
        pool = folds[folds["roles"].map(lambda r: "pool" in r) & (folds["arm"] == arm)]
        scores = pd.DataFrame([json.loads(line) for line in
                               likelihood[arm].read_text(encoding="utf-8").splitlines() if line.strip()])
        merged = scores.merge(pool[["id", "length", "mean_ca_plddt"]], on="id", suffixes=("_scored", ""))
        if len(merged) != len(pool) or (merged["length"] != merged["length_scored"]).any():
            raise SystemExit(f"E17 {arm}: the likelihood file does not cover the pool record for record")
        if abs(merged["mean_ca_plddt"].mean() - evaluator["pool_mean"]) > YIELD_TOLERANCE:
            raise SystemExit(f"E17 {arm}: the pool mean is not the retained one")
        oracle_points = {p["fraction"]: p for p in
                         record["e17"]["arms"][arm]["evaluators"]["mean_ca_plddt"]["gap_decomposition"]["points"]}
        for selector, key in (("likelihood", "mean_nll_per_token_nats"),
                              ("likelihood_per_residue", "mean_nll_per_residue_nats")):
            for point in evaluator["curves"][selector]:
                k = point["n_selected"]
                if k >= len(merged):
                    continue
                top = merged.sort_values([key, "id"]).iloc[:k]
                retained = point["selected_set"]
                selected_checks.append({"abs_difference": max(
                    abs(top["mean_ca_plddt"].mean() - point["selected_yield"]),
                    abs(top["length"].mean() - retained["mean_length"]),
                    abs(top["length"].min() - retained["min_length"]),
                    abs(top["length"].max() - retained["max_length"]))})
                length_matched = (point.get("length_conditional") or {}).get("gain_over_length_matched_random")
                oracle = merged.sort_values(["mean_ca_plddt", "id"], ascending=[False, True]).iloc[:k]
                if selector == "likelihood":
                    oracle_checks.append({"abs_difference": abs(
                        oracle["mean_ca_plddt"].mean() - (evaluator["pool_mean"]
                                                          + oracle_points[point["fraction"]]["attainable_gap"]))})
                rows.append({
                    "arm": arm, "selector": selector, "k": k, "pool_n": len(merged),
                    "selected_mean_plddt": float(top["mean_ca_plddt"].mean()),
                    "pool_mean_plddt": evaluator["pool_mean"],
                    "raw_difference": float(top["mean_ca_plddt"].mean() - evaluator["pool_mean"]),
                    "length_matched_gain_retained": length_matched,
                    **prefixed("selected_length", length_summary(top["length"])),
                    "selected_length_mean": float(top["length"].mean()),
                    **prefixed("pool_length", length_summary(merged["length"])),
                    "pool_length_mean": float(merged["length"].mean()),
                    **prefixed("oracle_length", length_summary(oracle["length"])),
                    "oracle_mean_plddt": float(oracle["mean_ca_plddt"].mean())})
    require(checks, "e17_selected_sets", selected_checks, YIELD_TOLERANCE)
    require(checks, "e17_oracle_yields", oracle_checks, YIELD_TOLERANCE)
    return rows


# ---------------------------------------------------------------- E18 and E11


def e18_block(repo: Path, checks: dict) -> tuple[list[dict], list[dict], list[dict]]:
    index = pd.DataFrame([json.loads(line) for line in (repo / PINNED["stage_pair_index"][0])
                          .read_text(encoding="utf-8").splitlines() if line.strip()])
    phenotype = json.loads((repo / PINNED["stage_pair_phenotype"][0]).read_text(encoding="utf-8"))
    one, two = (index[index["stage"] == s] for s in ("stage_1", "stage_2"))
    sets = [{"set": s, **length_summary(p["length"]), "mean_plddt": float(p["mean_ca_plddt"].mean()),
             "mean_plddt_below_50": float(p[p["length"] < 50]["mean_ca_plddt"].mean())}
            for s, p in (("stage 1", one), ("stage 2", two))]
    contrasts = phenotype["contrasts"]["mean_ca_plddt"]
    pools = [{"abs_difference": abs(sets[0]["mean_plddt"] - 59.0613), "label": "stage 1 mean"},
             {"abs_difference": abs(sets[1]["mean_plddt"] - 60.4935), "label": "stage 2 mean"}]
    # The registry quotes the two pool means to four decimals.
    require(checks, "e18_pool_means", pools, 5.0e-5, expected=2)
    ks = float(ks_2samp(one["length"], two["length"]).statistic)
    estimands = [{"estimand": name, "mean": contrasts[name]["mean"], "ci95_low": contrasts[name]["interval"][0],
                  "ci95_high": contrasts[name]["interval"][1]}
                 for name in ("generation_distribution", "adjusted", "matched_length",
                              "matched_length_hydropathy")]
    pair = {"ks": ks, "median_gap": float(two["length"].median() - one["length"].median()),
            "mean_gap": float(two["length"].mean() - one["length"].mean())}
    by_bin = []
    for low, high, name in LENGTH_BINS:
        a = one[(one["length"] >= low) & (one["length"] <= high)]
        b = two[(two["length"] >= low) & (two["length"] <= high)]
        by_bin.append({"length_bin": name, "n_stage_1": len(a), "n_stage_2": len(b),
                       "mean_stage_1": float(a["mean_ca_plddt"].mean()) if len(a) else None,
                       "mean_stage_2": float(b["mean_ca_plddt"].mean()) if len(b) else None})
    return [{**s, **pair} for s in sets], estimands, by_bin


def e11_block(repo: Path) -> tuple[list[dict], list[dict]]:
    record = json.loads((repo / PINNED["e11_comparison"][0]).read_text(encoding="utf-8"))
    sets, contrasts = [], []
    for panel in ("length_matched", "unmatched"):
        block = record["structure"][panel]
        for stratum, conditions in block["per_stratum"].items():
            for condition, v in conditions.items():
                sets.append({"panel": panel, "stratum": stratum, "condition": condition,
                             "n_folded": v["folded"], "length_median": v["median_residues"],
                             "length_mean": v["mean_residues"], "length_min": v["residue_range"][0],
                             "length_max": v["residue_range"][1], "budget_censored": v["budget_censored"],
                             "mean_plddt_x100": 100 * v["mean_plddt"]})
        for column in block["contrasts"]["plddt"]["columns"]:
            contrasts.append({"panel": panel, "stratum": column["stratum"], "contrast": column["contrast"],
                              "point_x100": 100 * column["point"],
                              "pointwise_low_x100": 100 * column["pointwise_interval"][0],
                              "pointwise_high_x100": 100 * column["pointwise_interval"][1],
                              "simultaneous_low_x100": 100 * column["simultaneous_interval"][0],
                              "simultaneous_high_x100": 100 * column["simultaneous_interval"][1],
                              "n_units": column["n_units"], "below_unit_floor": column["degenerate"]})
    return sets, contrasts


# ------------------------------------------------------------ decoding and census


def decoding_block(repo: Path) -> list[dict]:
    rows = []
    for name in ("decoding_sweep", "decoding_frontier"):
        sweep = json.loads((repo / PINNED[name][0]).read_text(encoding="utf-8"))
        for arm, entry in sorted(sweep["arms"].items()):
            verdict = sweep["verdict"][arm]
            for config, census in sorted(entry["census"].items()):
                comparison = entry["natural_band"][config]["mean_ca_plddt"]
                standardised = comparison.get("standardised") or {}
                paired = comparison.get("paired_whole_natural") or {}
                band = census.get("length_distribution_in_band") or {}
                eligibility = census.get("band_eligibility") or {}
                rows.append({
                    "sweep": name, "arm": arm, "config": config,
                    "is_reference": config == sweep["reference_configuration"],
                    "is_best": config == verdict["best_configuration"],
                    "in_band_n": band.get("n", 0), "in_band_min": band.get("min"),
                    "in_band_q1": band.get("p25"), "in_band_median": band.get("p50"),
                    "in_band_q3": band.get("p75"), "in_band_mean": band.get("mean"),
                    "in_band_max": band.get("max"),
                    "below_band": eligibility.get("below_band", 0), "above_band": eligibility.get("above_band", 0),
                    "evaluation_band": sweep["evaluation_band"],
                    "standardised_gap": standardised.get("gap"),
                    "gap_ci95_low": standardised["gap_interval"][0] if standardised.get("gap_interval") else None,
                    "gap_ci95_high": standardised["gap_interval"][1] if standardised.get("gap_interval") else None,
                    "gap_resolved": standardised.get("resolved"),
                    "paired_mean_abs_length_difference": paired.get("mean_absolute_length_difference")})
    return rows


def census_block(repo: Path, checks: dict) -> tuple[list[dict], list[dict]]:
    census = json.loads((repo / PINNED["structure_census"][0]).read_text(encoding="utf-8"))["checkpoints"]
    rows, pair_checks = [], []
    for path in sorted(glob.glob(str(repo / CENSUS_PREDICTIONS))):
        arm = Path(path).parent.name.removeprefix("s48_")
        records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        generated = {r["id"]: r["length"] for r in records if r["role"] == "generation"}
        shuffles = [r for r in records if r["role"] == "composition_shuffle"]
        differences = [abs(r["length"] - generated[r["paired_id"]]) for r in shuffles
                       if r["paired_id"] in generated]
        pair_checks.append({"abs_difference": float(max(differences)) if differences else float("inf"),
                            "paired": len(differences) == len(shuffles) == len(generated)})
        entry = census[arm]
        rows.append({"checkpoint": arm, "pairs": len(differences),
                     **prefixed("generated_length", length_summary(generated.values())),
                     "max_pair_length_difference": max(differences) if differences else None,
                     "difference_generated_minus_shuffle": entry["u3_mean"],
                     "ci95_low": entry["u3_ci95"][0] if entry.get("u3_ci95") else None,
                     "ci95_high": entry["u3_ci95"][1] if entry.get("u3_ci95") else None})
    if not all(c["paired"] for c in pair_checks):
        raise SystemExit("census: a shuffle is not paired to exactly one generated sequence")
    require(checks, "census_pair_lengths", pair_checks, 0.0, expected=17)
    main = [json.loads(line) for line in (repo / PINNED["generation_main_subset"][0])
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    generated = {r["id"]: r["length"] for r in main if r["role"] == "generation"}
    shuffles = [r for r in main if r["role"] == "composition_shuffle" and r["paired_id"] in generated]
    main_row = [{"set": "generation main subset (ProLLaMA, ZymCTRL)",
                 **length_summary(generated.values()), "pairs_to_a_generation": len(shuffles),
                 "max_pair_length_difference": max(abs(r["length"] - generated[r["paired_id"]]) for r in shuffles),
                 "natural_reference": length_summary([r["length"] for r in main if r["role"] == "natural_reference"])}]
    return rows, main_row


# ----------------------------------------------------------------------- selector


def selector_block(repo: Path, checks: dict) -> list[dict]:
    fold = pd.read_csv(repo / PINNED["fold_table"][0], sep="\t", comment="#")
    scores = pd.DataFrame([json.loads(line) for line in (repo / PINNED["selector_scores"][0])
                           .read_text(encoding="utf-8").splitlines() if line.strip()])
    merged = fold.merge(scores[["arm", "sequence_sha256", "mean_nll_per_residue_nats", "n_tokens"]],
                        on=["arm", "sequence_sha256"], validate="one_to_one")
    if len(merged) != len(fold) or (merged["n_tokens"] != merged["length_residues"]).any():
        raise SystemExit("selector: the scores do not cover the folded sample, token for residue")
    retained = pd.read_csv(repo / PINNED["selector_competence"][0], sep="\t").set_index("arm")
    rows, comparisons = [], []
    for arm, pool in merged.groupby("arm"):
        top = pool.sort_values(["mean_nll_per_residue_nats", "sequence_sha256"]).iloc[:SELECTOR_BUDGET]
        oracle = pool.sort_values(["mean_ca_plddt_0_100", "sequence_sha256"],
                                  ascending=[False, True]).iloc[:SELECTOR_BUDGET]
        r = retained.loc[arm]
        comparisons.append({"abs_difference": max(
            abs(pool["mean_ca_plddt_0_100"].mean() - r["pool_mean_plddt"]),
            abs(top["mean_ca_plddt_0_100"].mean() - pool["mean_ca_plddt_0_100"].mean() - r["selector_gain_plddt"]),
            abs(oracle["mean_ca_plddt_0_100"].mean() - pool["mean_ca_plddt_0_100"].mean() - r["oracle_gain_plddt"]))})
        rows.append({"arm": arm, **prefixed("pool_length", length_summary(pool["length_residues"])),
                     **prefixed("top_ten_length", length_summary(top["length_residues"])),
                     **prefixed("oracle_ten_length", length_summary(oracle["length_residues"])),
                     "pool_mean_plddt": float(pool["mean_ca_plddt_0_100"].mean()),
                     "top_ten_mean_plddt": float(top["mean_ca_plddt_0_100"].mean()),
                     "selector_gain": float(r["selector_gain_plddt"]),
                     "captured_fraction": float(r["captured_fraction"])})
    require(checks, "selector_competence", comparisons, SELECTOR_TOLERANCE, expected=8)
    return rows


# ------------------------------------------------------------------- limitations


def limitations(repo: Path) -> list[dict]:
    """The four things the retained trees do not carry, checked against the trees at run time."""

    control_columns = {t: pd.read_csv(repo / t, sep="\t", nrows=0).columns.tolist() for t in CONTROL_TABLES}
    carries_length = {t: any("length" in c or "residue" in c for c in cols)
                      for t, cols in control_columns.items()}
    e11 = json.loads((repo / PINNED["e11_comparison"][0]).read_text(encoding="utf-8"))
    e11_keys = sorted({k for panel in ("length_matched", "unmatched")
                       for conditions in e11["structure"][panel]["per_stratum"].values()
                       for v in conditions.values() for k in v})
    pools = sorted((repo / REWARD_POOLS).glob("rlp_base_stream*/attempts.jsonl"))
    pool_keys = sorted({k for p in pools for k in json.loads(p.read_text(encoding="utf-8").splitlines()[0])})
    return [
        {"item": "explicit-statistics, base-likelihood and selection-curve control tables",
         "carries": "counts, three pLDDT means, mean identities, reproduced shares",
         "missing": "the length of the filtered base pool, and which sequences it holds",
         "forecloses": "reading any filtered-pool contrast as length-matched, or measuring the length "
                       "shift the filter makes, from the tree",
         "holds_at_run_time": not any(carries_length.values()), "tables": control_columns},
        {"item": "E11 final panels' per-record folds",
         "carries": "per-condition count, mean, median and range of length, and the pLDDT contrasts",
         "missing": "the per-record fold files of the matched and unmatched panels, which live on cluster "
                    "storage and were never pulled",
         "forecloses": "interquartile ranges, length-distribution comparisons and any length-matched "
                       "recomputation of E11's contrasts; every E11 pLDDT contrast is unresolved, so no "
                       "claim depends on it",
         "holds_at_run_time": not (repo / E11_PANEL_FOLDS).exists()
                            and not any("quartile" in k or "p25" in k for k in e11_keys),
         "per_stratum_keys": e11_keys},
        {"item": "the recorded location of the E14 and E17 per-sequence folds",
         "carries": "the two fold-shard indexes, one structure per record of the 4,716-record cohort, in "
                    "the audit recovery tree, with digests equal to the ones the retained evaluation "
                    "record carries",
         "missing": "the original run directory the evaluation record names, which holds only the "
                    "aggregate and the likelihood cells",
         "forecloses": "nothing the audit needs: E14's generated means, E17's selected yields and the E17 "
                       "oracle's yields are all reproduced from the recovered indexes. A reader who follows "
                       "the evaluation record's own paths will not find them",
         "holds_at_run_time": not any((repo / p).exists() for p in E14_ORIGINAL_SHARDS)
                            and all((repo / p).exists() for p in EVALUATION_FOLD_INDEXES)},
        {"item": "the reward-provenance pools",
         "carries": "attempt ledgers: sequence, length, stop status, no fold",
         "missing": "any fold of any pool; the base pool's folds are pending and no trained arm has been "
                    "sampled for evaluation",
         "forecloses": "per-set length distributions and every pool-minus-base pLDDT family, stratified or "
                       "not, until the pools are folded; training-step records hold group mean, minimum and "
                       "maximum length only",
         "holds_at_run_time": not any("plddt" in k.lower() for k in pool_keys),
         "pools_present": [str(p.relative_to(repo)) for p in pools], "attempt_keys": pool_keys}]


# ------------------------------------------------------------------------- main


def write_table(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, float_format="%.6g")


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
        {"path": p, "sha256": sha256_file(repo / p)} for p in EVALUATION_FOLD_INDEXES]
    inputs["census_predictions"] = [{"path": str(Path(p).relative_to(repo)), "sha256": sha256_file(Path(p))}
                                    for p in sorted(glob.glob(str(repo / CENSUS_PREDICTIONS)))]

    checks: dict = {}
    record = json.loads((repo / EVALUATION_RECORD).read_text(encoding="utf-8"))
    folds = read_evaluation_folds(repo)
    likelihood = {"protgpt2": repo / PINNED["e17_likelihood_protgpt2"][0],
                  "zymctrl": repo / PINNED["e17_likelihood_zymctrl"][0]}
    e14 = e14_block(record, folds, checks)
    e17 = e17_block(record, folds, likelihood, checks)
    e18_sets, e18_estimands, e18_by_bin = e18_block(repo, checks)
    e11_sets, e11_contrasts = e11_block(repo)
    decoding = decoding_block(repo)
    census, main_subset = census_block(repo, checks)
    selector = selector_block(repo, checks)
    gaps = limitations(repo)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables").mkdir(exist_ok=True)
    for name, rows in (("e14_generated_against_natural", e14), ("e17_selected_against_pool", e17),
                       ("e18_stage_pools", e18_sets), ("e18_estimands", e18_estimands),
                       ("e18_by_length_bin", e18_by_bin), ("e11_sets", e11_sets),
                       ("e11_contrasts", e11_contrasts), ("decoding_sweeps", decoding),
                       ("composition_shuffle_census", census), ("selector_competence", selector)):
        write_table(args.out / f"tables/register_{name}.tsv", rows)
    write_json(args.out / "length_register_retained_comparisons.json", {
        "checks": checks, "e14": e14, "e17": e17, "e18_sets": e18_sets, "e18_estimands": e18_estimands,
        "e18_by_length_bin": e18_by_bin, "e11_sets": e11_sets, "e11_contrasts": e11_contrasts,
        "decoding": decoding, "composition_shuffle_census": census,
        "generation_main_subset": main_subset, "selector_competence": selector,
        "limitations": gaps, "inputs": inputs})
    print(json.dumps({name: {k: v for k, v in c.items() if k in ("n_values", "max_abs_difference", "passed")}
                      for name, c in checks.items()}, indent=1))
    print(json.dumps({g["item"]: g["holds_at_run_time"] for g in gaps}, indent=1))


if __name__ == "__main__":
    main()
