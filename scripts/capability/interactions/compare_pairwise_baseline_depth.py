#!/usr/bin/env python3
"""Compare two fitted pairwise sequence baselines on depth, admissibility and bands.

The pairwise baseline Q is prior-dominated: a 37-72 residue background carries
2.6e5 to 1.0e6 coupling parameters against a median reweighted depth of a few
hundred effective sequences. This stage reads two `baseline_q.json` records
fitted by the same entry point on two corpora and reports what changed as
quantities: Neff and effective-depth-per-site quantiles, how many backgrounds
clear the fit's declared support floor, the selected regularization, convergence,
and the independent-site double contrast that must read its construction zero.

It also reports, without re-deriving anything, which backgrounds would move
between the retrieval gate's frozen identity and depth bands on the deeper
corpus. Those bands belong to that gate and stay frozen here; a move is a
flagged consequence for a later re-estimation, not a reclassification.

No measurement, phenotype, DMS score or model output is read. Both inputs
already exist; this stage draws nothing and fits nothing.
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

from src.capability.context.homology import STRATUM_EDGES, STRATUM_NAMES, assign_stratum

#: The retrieval gate's frozen depth decades, read from its own declaration and
#: restated here only to label a move. Editing these would not re-band anything
#: in that gate; it would only mislabel this report.
DEPTH_BANDS = ("no_homolog_support", "neff_lt10", "neff_10_100",
               "neff_100_1000", "neff_ge1000")
DEPTH_EDGES = (10.0, 100.0, 1000.0)
QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)


def depth_band(neff: float | None, *, has_support: bool) -> str:
    if not has_support:
        return DEPTH_BANDS[0]
    value = float(neff or 0.0)
    for index, edge in enumerate(DEPTH_EDGES):
        if value < edge:
            return DEPTH_BANDS[index + 1]
    return DEPTH_BANDS[-1]


def max_identity_over_query(hits: Path, wanted: set[str]) -> dict[str, float]:
    """Highest ``100 * nident / qlen`` per query, streamed.

    The definition is the retrieval gate's own, and a query with no hit reads
    0.0, which is the direction that cannot overstate retrieval.
    """

    best = {name: 0.0 for name in wanted}
    with hits.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                raise ValueError(f"{hits}:{number} carries {len(parts)} fields, "
                                 "which cannot hold nident and qlen")
            query = parts[0]
            if query not in best:
                continue
            nident, qlen = float(parts[4]), float(parts[7])
            if qlen <= 0:
                raise ValueError(f"{hits}:{number} reports a non-positive qlen")
            best[query] = max(best[query], 100.0 * nident / qlen)
    return best


def coupling_parameters(length: int, states: int = 20) -> int:
    return length * (length - 1) // 2 * states * states


def side(report: Path) -> dict[str, dict]:
    payload = json.loads(report.read_text())
    if payload.get("schema") != "pairwise_sequence_baseline_v1":
        raise ValueError(f"{report}: not a pairwise sequence baseline record")
    return {row["name"]: row for row in payload["backgrounds"]}


def quantiles(values: list[float], digits: int) -> list[float] | None:
    if not values:
        return None
    return np.quantile(values, QUANTILES).round(digits).tolist()


def fitted_summary(rows: list[dict]) -> dict:
    neff = [row["neff_at_30_percent_identity"] for row in rows]
    depth = [row["effective_depth_per_site"] for row in rows]
    gains = [row["held_out_gain_over_independent_site"] for row in rows]
    ratio = [coupling_parameters(row["length"]) / row["neff_at_30_percent_identity"]
             for row in rows if row["neff_at_30_percent_identity"] > 0]
    zero = [abs(contrast["independent_site_contrast"])
            for row in rows for contrast in row["contrasts"]]
    pairwise = [abs(contrast["pairwise_contrast"])
                for row in rows for contrast in row["contrasts"]]
    return {
        "backgrounds_fitted": len(rows),
        "neff_quantiles": quantiles(neff, 1),
        "effective_depth_per_site_quantiles": quantiles(depth, 3),
        "reconstructed_rows_quantiles": quantiles([row["rows"] for row in rows], 0),
        "train_rows_quantiles": quantiles([row["train_rows"] for row in rows], 0),
        "held_out_rows_quantiles": quantiles([row["held_out_rows"] for row in rows], 0),
        "coupling_parameters_per_effective_sequence_quantiles": quantiles(ratio, 0),
        "held_out_gain_over_independent_site": {
            "min": round(min(gains), 4), "median": round(float(np.median(gains)), 4),
            "max": round(max(gains), 4),
            "backgrounds_with_positive_gain": sum(gain > 0 for gain in gains)} if gains else None,
        "selected_lambda_e_counts": {
            str(value): sum(1 for row in rows if row["selected_lambda_e"] == value)
            for value in sorted({row["selected_lambda_e"] for row in rows})},
        "final_refits_converged": sum(bool(row.get("final_converged")) for row in rows),
        "independent_site_double_contrast_max_absolute": max(zero) if zero else None,
        "independent_site_double_contrasts_measured": len(zero),
        "pairwise_double_contrast_absolute_quantiles": (
            np.quantile(pairwise, [0, .5, .9, 1]).round(6).tolist() if pairwise else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--before-hits", type=Path, required=True)
    parser.add_argument("--after-hits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    before, after = side(args.before), side(args.after)
    if set(before) != set(after):
        raise ValueError(
            "the two baselines cover different backgrounds; a depth comparison "
            "requires the same frozen cohort on both sides")
    names = sorted(before)
    identity = {"before": max_identity_over_query(args.before_hits, set(names)),
                "after": max_identity_over_query(args.after_hits, set(names))}

    per_background, moved_identity, moved_depth = [], [], []
    for name in names:
        entry: dict[str, object] = {"name": name, "length": before[name]["length"],
                                    "coupling_parameters": coupling_parameters(
                                        before[name]["length"])}
        bands: dict[str, dict[str, str]] = {}
        for label, record in (("before", before[name]), ("after", after[name])):
            has_support = record.get("rows", 0) > 0
            entry[label] = {
                "status": record["status"], "hits_retrieved": record["hits_retrieved"],
                "retrieval_ceiling_reached": record["retrieval_ceiling_reached"],
                "rows": record.get("rows"), "train_rows": record.get("train_rows"),
                "held_out_rows": record.get("held_out_rows"),
                "neff": record.get("neff_at_30_percent_identity"),
                "effective_depth_per_site": record.get("effective_depth_per_site"),
                "selected_lambda_e": record.get("selected_lambda_e"),
                "held_out_gain_over_independent_site": record.get(
                    "held_out_gain_over_independent_site"),
                "final_converged": record.get("final_converged"),
                "max_identity_over_query": round(identity[label][name], 2),
            }
            bands[label] = {
                "identity_band": assign_stratum(identity[label][name]),
                "depth_band": depth_band(record.get("neff_at_30_percent_identity"),
                                         has_support=has_support),
            }
        entry["frozen_bands_before"] = bands["before"]
        entry["bands_on_the_deeper_corpus"] = bands["after"]
        if bands["before"]["identity_band"] != bands["after"]["identity_band"]:
            moved_identity.append(name)
        if bands["before"]["depth_band"] != bands["after"]["depth_band"]:
            moved_depth.append(name)
        per_background.append(entry)

    fitted = {label: [row for row in source.values() if row["status"] == "fitted"]
              for label, source in (("before", before), ("after", after))}
    newly = sorted(name for name in names
                   if before[name]["status"] != "fitted" and after[name]["status"] == "fitted")
    lost = sorted(name for name in names
                  if before[name]["status"] == "fitted" and after[name]["status"] != "fitted")
    report = {
        "schema": "pairwise_baseline_depth_comparison_v1",
        "inputs": {"before": str(args.before), "after": str(args.after),
                   "before_hits": str(args.before_hits), "after_hits": str(args.after_hits)},
        "backgrounds": len(names),
        "support_floor_cleared": {label: len(rows) for label, rows in fitted.items()},
        "newly_clearing_the_support_floor": newly,
        "no_longer_clearing_the_support_floor": lost,
        "failures": {label: {name: row["status"] for name, row in source.items()
                             if row["status"] != "fitted"}
                     for label, source in (("before", before), ("after", after))},
        "summary": {label: fitted_summary(rows) for label, rows in fitted.items()},
        "frozen_band_consequences": {
            "owner": "the retrieval and memorization gate; its bands stay frozen and "
                     "are not re-derived here",
            "identity_band_edges": list(STRATUM_EDGES),
            "identity_band_names": list(STRATUM_NAMES),
            "depth_band_edges": list(DEPTH_EDGES),
            "depth_band_names": list(DEPTH_BANDS),
            "backgrounds_whose_identity_band_would_move": moved_identity,
            "backgrounds_whose_depth_band_would_move": moved_depth,
        },
        "per_background": per_background,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "per_background"}, indent=1))


if __name__ == "__main__":
    main()
