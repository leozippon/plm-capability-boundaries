#!/usr/bin/env python3
"""Print the decoding sweep's result table from its own artefact.

The analysis stage writes one JSON holding every number. This renders the tables
a reader actually needs from it, and nothing else: it computes no statistic, so
a number on screen can always be found unchanged in the artefact. Every row
carries its draw count, and an unresolved reading is printed as ``--`` with its
reason rather than being left out, because a table that quietly drops the
configurations it could not evaluate is the shape a false positive takes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.decoding import decoding_sweep as ds  # noqa: E402

DASH = "--"


def _number(value: Any, digits: int = 2) -> str:
    return DASH if value is None else f"{float(value):.{digits}f}"


def _interval(value: Any, digits: int = 2) -> str:
    if not value:
        return DASH
    return f"[{float(value[0]):.{digits}f}, {float(value[1]):.{digits}f}]"


def _census_table(block: Mapping[str, Any]) -> None:
    print(
        f"  {'configuration':26s} {'axis':26s} {'draws':>5s} {'inband':>6s} {'native':>6s} "
        f"{'tok/dr':>6s} {'medlen':>6s} {'p95len':>6s}"
    )
    for key in ds.CONFIG_KEYS:
        census = block["census"][key]
        length = census["length_distribution_all_products"]
        print(
            f"  {key:26s} {ds.config(key).axis:26s} {census['n_draws']:5d} "
            f"{census['n_in_band']:6d} {census['termination']['fraction_terminated_natively']:6.3f} "
            f"{census['generation_tokens']['mean_per_draw']:6.0f} "
            f"{length['p50']:6.0f} {length['p95']:6.0f}"
        )


def _per_candidate_table(block: Mapping[str, Any], field: str) -> None:
    print(
        f"  {'configuration':26s} {'nfold':>5s} {'ncl':>4s} {'mean':>7s} {'cluster CI95':>18s} "
        f"{'gap':>7s} {'gapCI95 (std)':>18s} {'paired gap':>10s}"
    )
    for key in ds.CONFIG_KEYS:
        candidate = block["per_candidate"][key].get(field, {})
        if "mean" not in candidate:
            print(f"  {key:26s} {candidate.get('n_folded', 0):5d} {DASH:>4s} {DASH:>7s} "
                  f"{DASH:>18s} {DASH:>7s} {DASH:>18s} {DASH:>10s}")
            continue
        cluster = candidate.get("cluster_level") or {}
        band = block["natural_band"][key].get(field, {})
        standardised = band.get("standardised") or {}
        paired = band.get("paired_whole_natural") or {}
        contrast = paired.get("contrast") or {}
        print(
            f"  {key:26s} {candidate['n_draws']:5d} {candidate['n_clusters']:4d} "
            f"{_number(candidate['mean']):>7s} {_interval(cluster.get('interval')):>18s} "
            f"{_number(standardised.get('gap')):>7s} "
            f"{_interval(standardised.get('gap_interval')):>18s} "
            f"{_number(contrast.get('difference')):>10s}"
        )


def _length_table(block: Mapping[str, Any], field: str) -> None:
    standardised = block.get("length_standardised", {}).get(field)
    if not standardised:
        print("  no configuration had enough in-band products to standardise")
        return
    print(f"  bin edges: {[round(edge) for edge in standardised['bin_edges']]}")
    print(f"  {'configuration':26s} {'nband':>5s} {'raw':>7s} {'standardised':>13s} {'CI95':>18s} {'empty bins':>12s}")
    for key, value in sorted(standardised["configurations"].items()):
        print(
            f"  {key:26s} {value['n_in_band']:5d} {_number(value['raw_mean']):>7s} "
            f"{_number(value['standardised_mean']):>13s} "
            f"{_interval(value['standardised_interval']):>18s} "
            f"{len(value['empty_bins']):12d}"
        )


def _selection_table(block: Mapping[str, Any], field: str, keys: list[str]) -> None:
    for key in keys:
        curves = block["selection"][key]["curves"].get(field, {})
        print(f"  {key} ({block['selection'][key]['draws_per_cluster']} draws per cluster)")
        for selector, curve in sorted(curves.items()):
            print(f"    selector {selector} [{ds.SELECTORS[selector]['kind']}]")
            print(
                f"      {'k':>3s} {'blocks':>6s} {'yield':>6s} {'mean':>7s} {'CI95':>18s} "
                f"{'tok/blk':>8s} {'folds/blk':>9s} {'sel len':>8s}"
            )
            for point in curve:
                print(
                    f"      {point['k']:3d} {point['n_blocks_with_candidate']:6d} "
                    f"{point['block_yield']:6.3f} {_number(point['mean']):>7s} "
                    f"{_interval(point['interval']):>18s} "
                    f"{_number(point['generation_tokens_per_block'], 0):>8s} "
                    f"{_number(point['folds_per_block'], 2):>9s} "
                    f"{_number(point['selected_length_mean'], 0):>8s}"
                )


def _matched_compute_table(block: Mapping[str, Any], field: str) -> None:
    for row in block.get("matched_compute", {}).get(field, []):
        budget = row["generation_tokens_per_selected_candidate"]
        print(f"  budget {budget:.0f} generation tokens per kept candidate "
              f"(tolerance {row['tolerance_fraction']:.0%}); best = {row['best_configuration']}")
        print(f"    {'configuration':26s} {'k':>3s} {'tok/blk':>8s} {'blocks':>6s} {'yield':>6s} "
              f"{'mean':>7s} {'CI95':>18s} {'sel len':>8s}")
        entries = sorted(
            row["entries"], key=lambda entry: (entry["mean"] is None, -(entry["mean"] or 0.0))
        )
        for entry in entries:
            print(
                f"    {entry['configuration']:26s} {entry['k']:3d} "
                f"{entry['generation_tokens_per_block']:8.0f} "
                f"{entry['n_blocks_with_candidate']:6d} {entry['block_yield']:6.3f} "
                f"{_number(entry['mean']):>7s} {_interval(entry['interval']):>18s} "
                f"{_number(entry['selected_length_mean'], 0):>8s}"
            )
        if row["configurations_absent"]:
            print(f"    absent at this budget: {len(row['configurations_absent'])} configurations")


def _degeneracy_table(block: Mapping[str, Any]) -> None:
    print(
        f"  {'configuration':26s} {'dup':>6s} {'kmerd':>6s} {'entropy':>7s} {'homopoly':>8s} "
        f"{'identity':>8s} {'neardup':>7s} {'flagged'}"
    )
    for key in ds.CONFIG_KEYS:
        value = block["degeneracy"].get(key, {})
        profile = value.get("profile")
        if profile is None:
            print(f"  {key:26s} {'(no in-band product to profile)'}")
            continue
        verdict = value.get("verdict") or {}
        flagged = sorted(verdict.get("axes_flagged", {}))
        print(
            f"  {key:26s} {profile['duplicate_fraction']:6.3f} "
            f"{_number(profile['mean_pairwise_kmer_distance'], 3):>6s} "
            f"{profile['mean_composition_entropy_nats']:7.3f} "
            f"{profile['fraction_with_homopolymer_run']:8.3f} "
            f"{_number(profile['nearest_corpus_identity'], 1):>8s} "
            f"{_number(profile['fraction_near_duplicate_of_corpus'], 3):>7s} "
            f"{','.join(flagged) if flagged else '-'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="the analysis stage's decoding_sweep.json")
    parser.add_argument("--field", default=ds.PRIMARY_EVALUATOR, help="which evaluator to table")
    parser.add_argument("--arm", default=None, help="restrict to one arm")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    field = args.field
    print(f"question: {report['question']}")
    print(f"evaluator: {field} -- {ds.EVALUATORS[field]}")
    print(f"evaluation band: {report['evaluation_band']} residues; "
          f"grid: {report['n_configurations']} configurations; "
          f"reference: {report['reference_configuration']}")
    print(f"fold contract: {report['join']['evaluation_signatures']} "
          f"(reference band {report['join'].get('reference_evaluation_signatures')})")
    print(f"novelty search attached: {report['novelty_available']}")
    for name, block in sorted(report["arms"].items()):
        if args.arm and name != args.arm:
            continue
        print()
        print("=" * 118)
        print(f"ARM {name} ({'conditioned' if block['conditioned'] else 'unconditioned'}): "
              f"{block['n_attempts']} attempts, {block['n_folded']} folded")
        natural = block["natural_pool"]["per_record"][field]
        print(f"  natural pool (unmatched): n={natural['n']} mean={natural['mean']:.2f} "
              f"CI95={_interval(natural['interval'])}")
        existing = block.get("existing_generation_results")
        if existing:
            marginal = existing["per_candidate"][field]
            gap = existing["length_matched_gap"][field]
            print(f"  existing generation results: n={marginal['n']} roles={existing['n_by_role']} "
                  f"mean={marginal['mean']:.2f} CI95={_interval(marginal['interval'])} "
                  f"mean length={existing['length_mean']:.0f}")
            if gap:
                print(f"    length-matched gap to natural: {gap['mean']:+.2f} "
                      f"CI95={_interval(gap['interval'])} (n={gap['n']})")
        print()
        print("-- census (every attempt, in band or not)")
        _census_table(block)
        print()
        print(f"-- per candidate, and the length-matched gap to the natural band [{field}]")
        _per_candidate_table(block, field)
        print()
        print(f"-- length-standardised across configurations [{field}]")
        _length_table(block, field)
        print()
        print(f"-- best-of-k on the deep-budget configurations [{field}]")
        _selection_table(block, field, [key for key in ds.DEEP_BUDGET_KEYS])
        print()
        print(f"-- matched compute [{field}]")
        _matched_compute_table(block, field)
        print()
        print("-- degeneracy diagnostics")
        _degeneracy_table(block)
        print()
        print("-- simultaneous statement")
        simultaneous = block.get("simultaneous", {}).get(field, {}).get("simultaneous", {})
        bound = simultaneous.get("max_over_configurations")
        if bound:
            print(f"  largest length-matched gap attained: {bound['point']:+.2f}; "
                  f"one-sided 97.5% upper bound {bound['upper_bound']:+.2f} over "
                  f"{simultaneous['n_configurations']} eligible configurations")
        verdict = report["verdict"][name]
        print("-- verdict")
        print(f"  reference gap {_number(verdict['reference_configuration_gap'])} "
              f"CI95={_interval(verdict['reference_configuration_interval'])}")
        print(f"  best configuration {verdict['best_configuration']} gap "
              f"{_number(verdict['best_configuration_gap'])} "
              f"CI95={_interval(verdict['best_configuration_interval'])}")
        print(f"  improvement over reference: {_number(verdict['improvement_over_reference'])}")
        print(f"  any configuration reaches the natural band: "
              f"{verdict['any_configuration_reaches_natural_band']}")
        degeneracy = verdict.get("degeneracy_of_best") or {}
        print(f"  best configuration degenerate: {degeneracy.get('degenerate')} "
              f"flagged={sorted(degeneracy.get('axes_flagged', {}))}")


if __name__ == "__main__":
    main()
