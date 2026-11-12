"""Evaluate the declared three-stream conjunction of the R6 primary identity-stratified pLDDT contrast.

The rule is declared in docs/objective-alignment.md ("The primary contrast and how it decides",
with the second amendment of 2026-11-10) and indexed in docs/experiment-registry.md. It has two
conjuncts and this script changes neither:

  1. the pooled tuned-minus-base difference in the stratum exceeds its margin, meaning the whole
     simultaneous interval of the declared family lies beyond the margin;
  2. the three per-stream point differences agree, meaning each one exceeds the same margin.

The same three point differences also answer the producer's own opposite reading, beyond the
negative margin, and that answer is tabulated beside the declared one. It is labelled an
observation throughout, because the declaration states the conjunction for design-relevant bias
only and says nothing about agreement in the negative direction.

Only the second conjunct is new work. The first, the estimand, the declared family (pairs crossed
with strata, pooled over termination), its native maximum-absolute-centred-deviation band and the
per-cohort margins are taken as given from the frozen eighteen-shard output of the unchanged
producer scripts/capability/generation/length_adjusted_contrasts_20261110/identity_stratified_contrast.py,
passed as --declared and recorded by digest in the provenance this script writes. That table is the
one source of the band; the README records the command that re-ran the producer and reproduced it
byte for byte. Nothing about the band is recomputed here: a per-stream family of thirty-six columns
is refused by the declaration itself, and the per-stream condition is declared on point differences.

The margin is the declared per-cohort one, computed from the pooled base pool. The margin each
stream's own base pool would give is carried beside it as transparency, never as the criterion.

Streams enter only as a stratification of the same rows the declared contrast reads. Each folded
structure carries the (arm, stream, replicate) cell recovered from the campaign's attempt ledgers
by recover_stream_labels.py; a structure without a recovered label would be excluded and counted,
and the coverage assertion below states how many there are.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.context import homology as hm  # noqa: E402

# The declared pairs, in the producer's own order and naming.
PAIRS = [
    ("112m", "protgpt3-112m", "protgpt3-112m-dpo"),
    ("1.3b", "protgpt3-1.3b", "protgpt3-1.3b-dpo"),
    ("10b", "protgpt3-10b", "protgpt3-10b-dpo"),
    ("mix7", "progen2-small-hugohrban", "progen2-small-mix7"),
]
#: The stratum the decision rule reads.
DECISION_STRATUM = "lt30_no_detectable_homology"
#: The producer's estimability rule, reused unchanged so a per-stream cell is estimable
#: on the same condition as a pooled one.
MIN_ROWS_PER_ARM = 2
#: The declared primary family, as the producer labels it.
FAMILY = "A_declared_pairs_x_strata"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures", type=Path, required=True, help="per_structure_nearest_neighbour.tsv the declared producer reads")
    ap.add_argument("--labels", type=Path, required=True, help="stream_labels.tsv from recover_stream_labels.py")
    ap.add_argument("--declared", type=Path, required=True, help="the frozen eighteen-shard identity_stratified_contrast.tsv")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    declared = pd.read_csv(args.declared, sep="\t")
    pooled = declared[declared.termination == "pooled"].copy()
    # One band and one set of margins, both read rather than recomputed. A table carrying more
    # than one family-A half-width is not the declared family's band and would silently change
    # the rule, so it is fatal here.
    halves = pooled.loc[pooled.family == FAMILY, "band_half_width"].dropna().unique()
    if len(halves) != 1:
        raise SystemExit(f"{args.declared}: {len(halves)} distinct {FAMILY} band half-widths; this is not the declared band")
    band_half_width = float(halves[0])

    t = pd.read_csv(args.structures, sep="\t")
    if not t.identity_over_query.between(0, 100).all():
        raise SystemExit("identity_over_query outside [0, 100]")
    t["stratum"] = t.identity_over_query.map(hm.assign_stratum)
    labels = pd.read_csv(args.labels, sep="\t")
    if labels.sequence_sha256.duplicated().any():
        raise SystemExit("a digest carries more than one cell label; the per-stream pools would not be disjoint")
    joined = t.merge(labels[["sequence_sha256", "arm", "stream", "replicate"]], on="sequence_sha256", how="left", suffixes=("", "_cell"))
    if len(joined) != len(t):
        raise SystemExit("the label join changed the row count")
    labelled = joined.stream.notna()
    disagree = labelled & (joined.arm != joined.arm_cell)
    if disagree.any():
        raise SystemExit(f"{int(disagree.sum())} rows: the ledger's arm disagrees with the structure table's")
    coverage = {
        "structures": len(joined),
        "labelled": int(labelled.sum()),
        "unlabelled": int((~labelled).sum()),
        "streams": sorted(int(s) for s in joined.loc[labelled, "stream"].unique()),
        "cells": int(joined.loc[labelled].groupby(["arm", "stream", "replicate"]).ngroups),
    }
    work = joined[labelled].copy()
    work["stream"] = work.stream.astype(int)
    streams = coverage["streams"]

    per_stream = []
    for pair, base, tuned in PAIRS:
        for stratum in hm.STRATUM_NAMES:
            g = work[work.stratum == stratum]
            for stream in streams:
                gb = g[(g.arm == base) & (g.stream == stream)]
                gt = g[(g.arm == tuned) & (g.stream == stream)]
                row = dict(pair=pair, stratum=stratum, stream=stream, n_base=len(gb), n_tuned=len(gt))
                if len(gb) < MIN_ROWS_PER_ARM or len(gt) < MIN_ROWS_PER_ARM:
                    row["status"] = f"not estimable (an arm has <{MIN_ROWS_PER_ARM} rows)"
                else:
                    row.update(
                        status="estimable",
                        base_mean=gb.plddt.mean(),
                        tuned_mean=gt.plddt.mean(),
                        diff=gt.plddt.mean() - gb.plddt.mean(),
                        margin_from_this_stream_base=max(2.0, 0.25 * max(0.0, 70 - gb.plddt.mean())),
                    )
                per_stream.append(row)
    S = pd.DataFrame(per_stream)

    # The declared margin is the per-cohort one; carry it onto every stream of the cell.
    S = S.merge(pooled[["pair", "stratum", "margin", "diff", "band_half_width", "sim_lo", "sim_hi", "family", "reading"]]
                .rename(columns={"diff": "pooled_diff", "margin": "margin_declared", "reading": "pooled_reading"}),
                on=["pair", "stratum"], how="left")
    S["exceeds_declared_margin"] = [
        bool(d > m) if status == "estimable" else pd.NA
        for status, d, m in zip(S.status, S["diff"], S.margin_declared)
    ]
    # The producer already reads a cell the other way round, as "beyond margin (negative)".
    # The same three point differences answer that direction too, so the column is carried
    # here rather than left to be recomputed; it is an observation, not a declared verdict,
    # because the declaration states the three-stream conjunction only for design-relevant bias.
    S["below_negative_declared_margin"] = [
        bool(d < -m) if status == "estimable" else pd.NA
        for status, d, m in zip(S.status, S["diff"], S.margin_declared)
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    cols = ["pair", "stratum", "stream", "status", "n_base", "n_tuned", "base_mean", "tuned_mean", "diff",
            "margin_declared", "margin_from_this_stream_base", "exceeds_declared_margin",
            "below_negative_declared_margin", "pooled_diff", "pooled_reading"]
    S[cols].to_csv(args.out / "stream_point_differences.tsv", sep="\t", index=False, float_format="%.4f")

    verdicts = []
    for pair, _, _ in PAIRS:
        for stratum in hm.STRATUM_NAMES:
            cell = S[(S.pair == pair) & (S.stratum == stratum)]
            pcell = pooled[(pooled.pair == pair) & (pooled.stratum == stratum)]
            row = dict(pair=pair, stratum=stratum,
                       pooled_family=pcell.family.iloc[0] if len(pcell) else "none",
                       pooled_reading=pcell.reading.iloc[0] if len(pcell) else "missing")
            estimable = cell[cell.status == "estimable"]
            row["streams_estimable"] = len(estimable)
            row["streams_declared"] = len(streams)
            for stream in streams:
                one = cell[cell.stream == stream]
                row[f"diff_s{stream}"] = one["diff"].iloc[0] if len(one) and one.status.iloc[0] == "estimable" else float("nan")
            if len(pcell) and pd.notna(pcell["diff"].iloc[0]):
                p = pcell.iloc[0]
                row.update(pooled_diff=p["diff"], margin=p["margin"], band_half_width=p["band_half_width"],
                           sim_lo=p["sim_lo"], sim_hi=p["sim_hi"])
                row["interval_beyond_margin_positive"] = bool(p["sim_lo"] > p["margin"])
            else:
                row["interval_beyond_margin_positive"] = pd.NA
            if len(estimable) == len(streams):
                row["streams_all_below_negative_margin"] = bool(estimable.below_negative_declared_margin.all())
            else:
                row["streams_all_below_negative_margin"] = pd.NA
            if len(estimable) < len(streams):
                row["streams_conjunction"] = "undecidable (a stream is not estimable)"
            elif bool(estimable.exceeds_declared_margin.all()):
                row["streams_conjunction"] = "holds (all streams exceed the margin)"
            else:
                row["streams_conjunction"] = "fails (%d of %d streams exceed the margin)" % (
                    int(estimable.exceeds_declared_margin.sum()), len(streams))
            if pd.isna(row["interval_beyond_margin_positive"]):
                row["design_relevant_bias"] = "not estimable: the pooled cell carries no interval"
            elif row["streams_conjunction"].startswith("undecidable"):
                row["design_relevant_bias"] = "undecidable"
            elif row["interval_beyond_margin_positive"] and row["streams_conjunction"].startswith("holds"):
                row["design_relevant_bias"] = "holds"
            else:
                failed = []
                if not row["interval_beyond_margin_positive"]:
                    failed.append("the simultaneous interval does not lie beyond the margin")
                if not row["streams_conjunction"].startswith("holds"):
                    failed.append("the three-stream conjunction fails")
                row["design_relevant_bias"] = "fails: " + "; ".join(failed)
            verdicts.append(row)
    V = pd.DataFrame(verdicts)
    V.to_csv(args.out / "stream_conjunction.tsv", sep="\t", index=False, float_format="%.4f")

    decision = V[V.stratum == DECISION_STRATUM]
    lines = [
        "declared three-stream conjunction, R6 primary identity-stratified pLDDT contrast",
        "band: family %s, half-width %.3f pLDDT, read from %s" % (FAMILY, band_half_width, args.declared.name),
        "label coverage: %d of %d folded structures carry a recovered cell label (%d unlabelled), %d cells, streams %s" % (
            coverage["labelled"], coverage["structures"], coverage["unlabelled"], coverage["cells"],
            ",".join(str(s) for s in coverage["streams"])),
        "",
        "decision stratum %s, per pair:" % DECISION_STRATUM,
    ]
    for _, r in decision.iterrows():
        lines.append(
            "  %-5s pooled diff %+7.3f  margin %.3f  interval [%+7.3f, %+7.3f] (%s)  streams %s  -> %s" % (
                r.pair, r.pooled_diff, r.margin, r.sim_lo, r.sim_hi, r.pooled_reading,
                " ".join("%+.3f" % r[f"diff_s{s}"] for s in streams), r.design_relevant_bias))
    holds = decision[decision.design_relevant_bias == "holds"]
    undec = decision[decision.design_relevant_bias == "undecidable"]
    agree_neg = decision[decision.streams_all_below_negative_margin == True]  # noqa: E712
    lines += [
        "",
        "overall: design-relevant bias holds for %d of %d pairs; %d undecidable" % (len(holds), len(decision), len(undec)),
        "observation, not a declared verdict: all three streams lie below the negative margin for %s" % (
            ", ".join(agree_neg.pair) if len(agree_neg) else "no pair"),
        "",
        "the rest of the declared family, where the rule reads narrowing rather than design bias:",
    ]
    for _, r in V[V.stratum != DECISION_STRATUM].iterrows():
        lines.append(
            "  %-5s %-28s streams %s  conjunction: %s; pooled interval: %s" % (
                r.pair, r.stratum,
                " ".join("%+8.3f" % r[f"diff_s{s}"] for s in streams),
                r.streams_conjunction, r.pooled_reading))
    text = "\n".join(lines) + "\n"
    (args.out / "verdict.txt").write_text(text, encoding="utf-8")
    provenance = {
        "declared_contrast": {"path": str(args.declared), "sha256": sha256_file(args.declared)},
        "structures": {"path": str(args.structures), "sha256": sha256_file(args.structures)},
        "labels": {"path": str(args.labels), "sha256": sha256_file(args.labels)},
        "code_sha256": sha256_file(Path(__file__)),
        "family": FAMILY,
        "band_half_width": band_half_width,
        "decision_stratum": DECISION_STRATUM,
        "coverage": coverage,
    }
    (args.out / "stream_conjunction_provenance.json").write_text(
        json.dumps(provenance, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
