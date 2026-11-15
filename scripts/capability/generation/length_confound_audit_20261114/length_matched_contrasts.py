#!/usr/bin/env python3
"""Length-matched readings of the objective-alignment contrasts, and the check that licenses them.

The campaign reads tuned-minus-base mean CA-pLDDT inside identity strata. The
instrument's confidence depends on sequence length, and the tuned arms are shorter
than their bases in several cells, so a gap in such a cell can be partly a gap in
length. The campaign already retains length-matched readings for the length
strata, the pooled finished cells and the 768-residue companion cohort, computed
with an exact sixteen-residue-bin estimator: bins of floor(length / 16), each
tuned bin's difference weighted by its tuned count, tuned bins with no base row
dropped. It retains none for the identity-stratified cells, which are the declared
primary. This producer supplies them, and reports for every compared cell the
length distribution of both sets beside the claimed difference.

Two checks come first and are the licence for everything after them, so the
producer makes them instead of citing them. The estimator is rebuilt here and must
reproduce every retained sixteen-bin value (the 11 length-stratum and 4 pooled
finished values of the objective-tuned cohort and the 7 values of the companion
cohort) to within a declared tolerance; and every cell definition used below must
reproduce the retained count and raw difference of the cell it names. If either
check fails the producer writes the failing check and stops with a non-zero exit,
and no matched reading is written.

A cell is "material" under this audit's own working rule, not the project's: the
median lengths differ by 20 residues or more, or the two-sample Kolmogorov-Smirnov
distance between the length distributions is 0.20 or more. The length-attributable
share of a gap is (raw - matched) / raw under the sixteen-bin estimator, reported
where the raw gap is at least one point.

Reads retained tables only; computes no fold and loads no model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.capability.context import homology as hm  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation.length_matching import (  # noqa: E402
    BIN_WIDTH,
    MATERIAL_KS,
    MATERIAL_MEDIAN_GAP,
    MIN_RAW_FOR_SHARE,
    MIN_ROWS,
    attributable_share,
    bin_match,
    check_values,
    is_material,
    length_summary,
)

# The retained sixteen-bin values are printed to four decimals (so a value is known
# to +-5e-5) and the companion table stores pLDDT to three decimals; twice the
# first figure covers both and is far below any difference between estimators.
ESTIMATOR_TOLERANCE = 1.0e-4
# Retained sixteen-bin values the producer must find and reproduce: 11 length-stratum and
# 4 pooled-finished values of the objective-tuned cohort, 7 of the companion cohort.
EXPECTED_REPRODUCTIONS = {"objective_tuned": 15, "companion": 7}
# The identity-stratified table was computed from the nearest-neighbour table,
# whose pLDDT carries three decimals, while this producer reads the fold table's
# full-precision values; per-row differences reach 5.05e-4.
IDENTITY_SET_TOLERANCE = 5.0e-4
SET_TOLERANCE = 1.0e-4

OT_DIR = "results/external_baseline_analysis/length_adjusted_contrasts_20261110/eighteen_shard"
COMPANION_DIR = "results/long_cap_termination_20261110/analysis"
# Retained inputs and their digests. The first six tables carry the digest the
# registry records; the fold and neighbour tables are the ones the 2026-11-12
# endpoints run pins.
PINNED = {
    "fold_table": (f"{OT_DIR}/checkpoint_structures_18of68.tsv",
                   "2fd2ee4b91b67ef7bcef91241ccfddc61353b6f51e5b788270be8a2df8aabf06"),
    "neighbour_table": (f"{OT_DIR}/natural_neighbour/per_structure_nearest_neighbour.tsv",
                        "b99e34e0e77408cb012761989c7911297c3c74afd4e584ec50c8f456f671be47"),
    "contrasts_by_stratum": (f"{OT_DIR}/contrasts_by_stratum.tsv",
                             "9f446194796f540fac252bed2136d230675062a38dd4902df31533f30a8e664f"),
    "uncapped_pooled_contrasts": (f"{OT_DIR}/uncapped_pooled_contrasts.tsv",
                                  "417f8578461930f58daaf3ed75d1f2c57abe5a9b80914a08323297fc20e7557a"),
    "pooled_slope_and_cap_split": (f"{OT_DIR}/pooled_slope_and_cap_split.tsv",
                                   "b63f3b35a2367a9b2306fb3d180b216b438e6e7b2ddeb2ed94d5d48844f15dbf"),
    "identity_stratified_contrast": (f"{OT_DIR}/identity_stratified_contrast.tsv",
                                     "ec440436b1a77a74e7acfdd551ae4f69b1dda5668fa87710b69d4a021ab7430c"),
    "companion_per_structure": (f"{COMPANION_DIR}/per_structure_joined.tsv",
                                "b5e7713d3a2be8f5b5430f1d5c562199f1a77371c8e824b97483f3cfa4c18a20"),
    "companion_primary_natural": (f"{COMPANION_DIR}/primary_natural_16_512.tsv",
                                  "43b9effbdf10a2cf2ea57f5ddbadddd21a1336b411ae01283429dea8652bed5c"),
    "companion_capped_contrasts": (f"{COMPANION_DIR}/capped_contrasts.tsv",
                                   "ed5bc398b7b6a9172cb4640d53d4e730de6e88ccd4dbc7a0c6ebd1c3d550af83"),
}
# Recorded, not pinned: the registry quotes no digest for these three.
RECORDED = {
    "companion_primary_identity": f"{COMPANION_DIR}/primary_identity_stratified.tsv",
    "companion_primary_identity_length_matched":
        f"{COMPANION_DIR}/primary_identity_stratified_length_matched.tsv",
    "companion_mid_natural": f"{COMPANION_DIR}/mid_natural_513_767.tsv",
}

PAIRS = {"112m": ("protgpt3-112m", "protgpt3-112m-dpo"),
         "1.3b": ("protgpt3-1.3b", "protgpt3-1.3b-dpo"),
         "10b": ("protgpt3-10b", "protgpt3-10b-dpo"),
         "mix7": ("progen2-small-hugohrban", "progen2-small-mix7")}
LENGTH_STRATA = ((16, 128), (129, 256), (257, 512))
CAP = 400


# ------------------------------------------------------------------- the cohorts


def load_objective_tuned(fold_table: Path, neighbour_table: Path) -> pd.DataFrame:
    fold = pd.read_csv(fold_table, sep="\t", comment="#").rename(
        columns={"length_residues": "L", "mean_ca_plddt_0_100": "plddt"})
    neighbour = pd.read_csv(neighbour_table, sep="\t")
    frame = fold[["arm", "sequence_sha256", "L", "plddt"]].merge(
        neighbour[["sequence_sha256", "capped", "identity_over_query"]],
        on="sequence_sha256", validate="one_to_one")
    if len(frame) != len(fold):
        raise SystemExit("the neighbour table does not cover every folded structure")
    frame["capped"] = frame["capped"].astype(bool)
    if not ((frame["L"] == CAP) == frame["capped"]).all():
        raise SystemExit("`capped` is not exactly L == 400 in this table")
    frame["stratum"] = frame["identity_over_query"].map(hm.assign_stratum)
    return frame


def objective_tuned_cells(frame: pd.DataFrame):
    """(family, pair, cell, base rows, tuned rows) for every comparison the campaign reads."""

    for pair, (base, tuned) in PAIRS.items():
        b, t = frame[frame["arm"] == base], frame[frame["arm"] == tuned]
        for low, high in LENGTH_STRATA:
            pick = lambda d, low=low, high=high: d[(d["L"] >= low) & (d["L"] <= high)]  # noqa: E731
            yield "length_stratum", pair, f"{low}-{high}", pick(b), pick(t)
        inside = lambda d: d[(d["L"] >= 257) & (d["L"] <= 512)]  # noqa: E731
        yield "cap_split", pair, "257-512 capped", inside(b)[inside(b)["capped"]], \
            inside(t)[inside(t)["capped"]]
        yield "cap_split", pair, "257-512 finished", inside(b)[~inside(b)["capped"]], \
            inside(t)[~inside(t)["capped"]]
        yield "pooled_finished", pair, "all lengths, L < 400", b[~b["capped"]], t[~t["capped"]]
        for stratum in hm.STRATUM_NAMES:
            sb, st = b[b["stratum"] == stratum], t[t["stratum"] == stratum]
            yield "identity", pair, f"{stratum} pooled", sb, st
            yield "identity", pair, f"{stratum} finished", sb[~sb["capped"]], st[~st["capped"]]
            yield "identity", pair, f"{stratum} capped", sb[sb["capped"]], st[st["capped"]]


def companion_cells(frame: pd.DataFrame):
    natural = frame["population"].isin(["base-natural", "tuned-natural"])
    pooled = natural & (frame["L"] >= 16) & (frame["L"] <= 512)
    specs = [("length_stratum", "natural 16-512 pooled", pooled)]
    specs += [("length_stratum", f"natural {lo}-{hi}", natural & (frame["L"] >= lo) & (frame["L"] <= hi))
              for lo, hi in LENGTH_STRATA]
    specs += [("identity", f"{name} (natural 16-512)", pooled & (frame["stratum_id"] == name))
              for name in hm.STRATUM_NAMES[:3]]
    specs += [("longest_finished", "natural 513-767",
               natural & (frame["L"] >= 513) & (frame["L"] <= 767)),
              ("capped", "capped at 768", frame["population"].isin(["base-capped", "tuned-capped"]))]
    for family, name, mask in specs:
        yield family, "10b", name, frame[mask & (frame["tuned"] == 0)], frame[mask & (frame["tuned"] == 1)]


# --------------------------------------------------------------------- retained


def retained_objective_tuned(repo: Path) -> dict:
    """Retained numbers keyed by (family, pair, cell): raw, n's, and any matched value."""

    out: dict = {}
    table = pd.read_csv(repo / PINNED["contrasts_by_stratum"][0], sep="\t")
    for _, r in table[table["endpoint"] == "plddt"].iterrows():
        if pd.isna(r["raw"]):
            continue
        out[("length_stratum", r["pair"], r["stratum"])] = {
            "raw": r["raw"], "n": (r["n_base"], r["n_tuned"]), "table": "contrasts_by_stratum",
            "matched": {"16-bin": r["bin"], "linear": r["lin"], "spline": r["flex"], "nearest": r["nn"]}}
    table = pd.read_csv(repo / PINNED["uncapped_pooled_contrasts"][0], sep="\t")
    for _, r in table[table["endpoint"] == "plddt"].iterrows():
        out[("pooled_finished", r["pair"], "all lengths, L < 400")] = {
            "raw": r["raw"], "n": (r["n_base"], r["n_tuned"]), "table": "uncapped_pooled_contrasts",
            "matched": {"16-bin": r["bin"], "linear": r["lin"], "spline": r["spline"]}}
    table = pd.read_csv(repo / PINNED["pooled_slope_and_cap_split"][0], sep="\t")
    for _, r in table[(table["endpoint"] == "plddt") & table["kind"].str.startswith("L")].iterrows():
        cell = "257-512 capped" if r["kind"].startswith("L==400") else "257-512 finished"
        out[("cap_split", r["pair"], cell)] = {"raw": r["est"], "n": (r["n_base"], r["n_tuned"]),
                                              "table": "pooled_slope_and_cap_split", "matched": {}}
    table = pd.read_csv(repo / PINNED["identity_stratified_contrast"][0], sep="\t")
    for _, r in table.iterrows():
        if pd.isna(r["diff"]):
            continue
        out[("identity", r["pair"], f"{r['stratum']} {r['termination']}")] = {
            "raw": r["diff"], "n": (r["n_base"], r["n_tuned"]),
            "table": "identity_stratified_contrast", "matched": {}}
    return out


def retained_companion(repo: Path) -> dict:
    out: dict = {}
    table = pd.read_csv(repo / PINNED["companion_primary_natural"][0], sep="\t")
    for _, r in table[table["endpoint"] == "plddt"].iterrows():
        name = ("natural 16-512 pooled" if r["stratum"] == "16-512 pooled"
                else f"natural {r['stratum']}")
        out[("length_stratum", "10b", name)] = {
            "raw": r["raw"], "n": (r["n_base"], r["n_tuned"]), "table": "primary_natural_16_512",
            "matched": {"16-bin": r["bin16"], "linear": r["linear"], "spline": r["spline"]}}
    raw = pd.read_csv(repo / RECORDED["companion_primary_identity"], sep="\t")
    matched = pd.read_csv(repo / RECORDED["companion_primary_identity_length_matched"], sep="\t")
    for stratum in hm.STRATUM_NAMES[:3]:
        a = raw[(raw["identity_stratum"] == stratum) & (raw["endpoint"] == "plddt")].iloc[0]
        m = matched[(matched["identity_stratum"] == stratum) & (matched["endpoint"] == "plddt")].iloc[0]
        out[("identity", "10b", f"{stratum} (natural 16-512)")] = {
            "raw": a["diff"], "n": (a["n_base"], a["n_tuned"]), "table": "primary_identity_stratified",
            "matched": {"16-bin": m["bin16"]}}
    mid = pd.read_csv(repo / RECORDED["companion_mid_natural"], sep="\t")
    mid = mid[mid["endpoint"] == "plddt"].iloc[0]
    out[("longest_finished", "10b", "natural 513-767")] = {
        "raw": mid["diff"], "n": (mid["n_base"], mid["n_tuned"]), "table": "mid_natural_513_767",
        "matched": {}}
    capped = pd.read_csv(repo / PINNED["companion_capped_contrasts"][0], sep="\t")
    capped = capped[(capped["endpoint"] == "plddt") & (capped["contrast"] == "tuned capped - base capped")]
    c = capped.iloc[0]
    out[("capped", "10b", "capped at 768")] = {"raw": c["diff"], "n": (c["n_base"], c["n_tuned"]),
                                              "table": "capped_contrasts", "matched": {}}
    return out


# ---------------------------------------------------------------------- the work


def describe_cell(cohort: str, family: str, pair: str, cell: str, base: pd.DataFrame,
                  tuned: pd.DataFrame, retained: dict | None) -> tuple[dict, dict | None]:
    """One row of the register, and the sixteen-bin value if it was recomputed."""

    raw = float(tuned["plddt"].mean() - base["plddt"].mean()) if len(base) and len(tuned) else None
    both_ok = len(base) >= 2 and len(tuned) >= 2
    ks = float(ks_2samp(base["L"], tuned["L"]).statistic) if both_ok else None
    median_gap = (float(tuned["L"].median() - base["L"].median())) if both_ok else None
    row = {"cohort": cohort, "family": family, "pair": pair, "cell": cell,
           "n_base": int(len(base)), "n_tuned": int(len(tuned)),
           **{f"base_{k}": v for k, v in length_summary(base["L"]).items() if k != "n"},
           **{f"tuned_{k}": v for k, v in length_summary(tuned["L"]).items() if k != "n"},
           "median_gap": median_gap, "mean_gap": float(tuned["L"].mean() - base["L"].mean()) if both_ok else None,
           "ks": ks, "material": bool(is_material(median_gap, ks)) if both_ok else None,
           "thin": bool(min(len(base), len(tuned)) < MIN_ROWS), "raw_difference": raw,
           "retained_difference": None, "matched_16bin": None, "matched_source": None,
           "matched_linear": None, "matched_spline": None, "matched_nearest": None}
    recomputed = None
    if retained is not None:
        row["retained_difference"] = float(retained["raw"])
        matched = retained["matched"]
        if "16-bin" in matched and not pd.isna(matched["16-bin"]):
            row["matched_16bin"], row["matched_source"] = float(matched["16-bin"]), "retained"
        for key, name in (("linear", "matched_linear"), ("spline", "matched_spline"),
                          ("nearest", "matched_nearest")):
            if key in matched and not pd.isna(matched[key]):
                row[name] = float(matched[key])
    if row["matched_16bin"] is None:
        one_length = (len(base) and len(tuned) and base["L"].nunique() == 1
                      and tuned["L"].nunique() == 1 and base["L"].iloc[0] == tuned["L"].iloc[0])
        if one_length:
            row["matched_source"] = "not needed: every row of both sets sits at one length"
        elif min(len(base), len(tuned)) >= MIN_ROWS:
            value, _, _ = bin_match(base["L"].to_numpy(), base["plddt"].to_numpy(),
                                    tuned["L"].to_numpy(), tuned["plddt"].to_numpy())
            row["matched_16bin"], row["matched_source"] = float(value), "recomputed"
            recomputed = {"family": family, "pair": pair, "cell": cell, "raw": raw,
                          "matched": float(value)}
        else:
            row["matched_source"] = f"not computed: an arm has fewer than {MIN_ROWS} rows"
    row["length_attributable_share"] = attributable_share(raw, row["matched_16bin"])
    return row, recomputed


def reproduction_rows(cohort: str, frame: pd.DataFrame, cells, retained: dict) -> list[dict]:
    """Every retained sixteen-bin value, recomputed from the rows of its own cell."""

    rows = []
    for family, pair, cell, base, tuned in cells(frame):
        info = retained.get((family, pair, cell))
        if info is None or "16-bin" not in info["matched"] or pd.isna(info["matched"]["16-bin"]):
            continue
        value, _, _ = bin_match(base["L"].to_numpy(), base["plddt"].to_numpy(),
                                tuned["L"].to_numpy(), tuned["plddt"].to_numpy())
        rows.append({"cohort": cohort, "source_table": info["table"], "pair": pair, "cell": cell,
                     "retained": float(info["matched"]["16-bin"]), "recomputed": float(value),
                     "abs_difference": abs(float(value) - float(info["matched"]["16-bin"]))})
    return rows


def set_check_rows(cohort: str, frame: pd.DataFrame, cells, retained: dict) -> list[dict]:
    """Every retained raw difference and count, recomputed from the cell this producer builds."""

    rows = []
    for family, pair, cell, base, tuned in cells(frame):
        info = retained.get((family, pair, cell))
        if info is None:
            continue
        raw = float(tuned["plddt"].mean() - base["plddt"].mean())
        tolerance = IDENTITY_SET_TOLERANCE if family == "identity" and cohort == "objective_tuned" \
            else SET_TOLERANCE
        rows.append({"cohort": cohort, "source_table": info["table"], "pair": pair, "cell": cell,
                     "retained_n_base": int(info["n"][0]), "n_base": int(len(base)),
                     "retained_n_tuned": int(info["n"][1]), "n_tuned": int(len(tuned)),
                     "retained": float(info["raw"]), "recomputed": raw,
                     "abs_difference": abs(raw - float(info["raw"])), "tolerance": tolerance,
                     "counts_equal": bool(int(info["n"][0]) == len(base) and int(info["n"][1]) == len(tuned))})
    return rows


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
    for name, relative in RECORDED.items():
        inputs[name] = {"path": relative, "sha256": sha256_file(repo / relative)}

    ot = load_objective_tuned(repo / PINNED["fold_table"][0], repo / PINNED["neighbour_table"][0])
    companion = pd.read_csv(repo / PINNED["companion_per_structure"][0], sep="\t")
    retained_ot, retained_co = retained_objective_tuned(repo), retained_companion(repo)

    # The two checks. Everything below is written only if both pass.
    reproduction = (reproduction_rows("objective_tuned", ot, objective_tuned_cells, retained_ot)
                    + reproduction_rows("companion", companion, companion_cells, retained_co))
    set_checks = (set_check_rows("objective_tuned", ot, objective_tuned_cells, retained_ot)
                  + set_check_rows("companion", companion, companion_cells, retained_co))
    estimator = check_values(reproduction, ESTIMATOR_TOLERANCE, expected=sum(EXPECTED_REPRODUCTIONS.values()))
    estimator["by_cohort"] = {c: sum(1 for r in reproduction if r["cohort"] == c)
                              for c in EXPECTED_REPRODUCTIONS}
    estimator["passed"] = bool(estimator["passed"] and estimator["by_cohort"] == EXPECTED_REPRODUCTIONS)
    cell_check = {"n_cells": len(set_checks),
                  "max_abs_difference": max(r["abs_difference"] for r in set_checks),
                  "counts_all_equal": all(r["counts_equal"] for r in set_checks),
                  "passed": all(r["abs_difference"] <= r["tolerance"] and r["counts_equal"]
                                for r in set_checks)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables").mkdir(exist_ok=True)
    pd.DataFrame(reproduction).to_csv(args.out / "tables/estimator_reproduction.tsv", sep="\t",
                                      index=False, float_format="%.8g")
    pd.DataFrame(set_checks).to_csv(args.out / "tables/set_definition_checks.tsv", sep="\t",
                                    index=False, float_format="%.8g")
    verdict = {"estimator_reproduction": estimator, "set_definition_check": cell_check,
               "inputs": inputs}
    write_json(args.out / "length_matched_contrasts_checks.json", verdict)
    if not (estimator["passed"] and cell_check["passed"]):
        print(json.dumps(verdict, indent=1))
        raise SystemExit("a check failed: no matched reading was written (see tables/)")

    rows, recomputed = [], []
    for family, pair, cell, base, tuned in objective_tuned_cells(ot):
        row, new = describe_cell("objective_tuned", family, pair, cell, base, tuned,
                                 retained_ot.get((family, pair, cell)))
        rows.append(row)
        recomputed += [new] if new else []
    companion_rows = []
    for family, pair, cell, base, tuned in companion_cells(companion):
        row, _ = describe_cell("companion", family, pair, cell, base, tuned,
                               retained_co.get((family, pair, cell)))
        companion_rows.append(row)
    pd.DataFrame(rows).to_csv(args.out / "tables/cells_objective_tuned.tsv", sep="\t", index=False,
                              float_format="%.6g")
    pd.DataFrame(companion_rows).to_csv(args.out / "tables/cells_companion.tsv", sep="\t",
                                        index=False, float_format="%.6g")
    material = [r for r in rows + companion_rows if r["material"] and not r["thin"]
                and r["matched_16bin"] is not None]
    result = {
        **verdict,
        "rules": {"bin_width": BIN_WIDTH, "min_rows": MIN_ROWS,
                  "material_median_gap": MATERIAL_MEDIAN_GAP, "material_ks": MATERIAL_KS,
                  "min_raw_for_share": MIN_RAW_FOR_SHARE},
        "recomputed_readings": recomputed,
        "n_recomputed_by_family": {f: sum(1 for r in recomputed if r["family"] == f)
                                   for f in sorted({r["family"] for r in recomputed})},
        "material_cells_both_arms_at_least_min_rows": [
            {k: r[k] for k in ("cohort", "family", "pair", "cell", "n_base", "n_tuned", "median_gap",
                               "ks", "raw_difference", "matched_16bin", "matched_source",
                               "length_attributable_share")} for r in material],
    }
    write_json(args.out / "length_matched_contrasts.json", result)
    print(json.dumps({"estimator_reproduction": estimator, "set_definition_check": cell_check,
                      "recomputed_readings_by_family": result["n_recomputed_by_family"],
                      "material_cells": len(material)}, indent=1))


if __name__ == "__main__":
    main()
