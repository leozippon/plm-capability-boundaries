#!/usr/bin/env python3
"""Length sensitivity of model products against length-matched corpus fragments.

The generation protocol caps ``max_new_tokens`` at 400. That cap is a token
budget, not a residue budget, so two checkpoints run under it can still put
different residue-length distributions on the table. This script does not
generate sequences and does not load a model. It reads the frozen attempt
ledgers and the oracle tables of the generation-and-control gate.

Length bins are the strata already declared in
``generation_evidence.POLICY``. A bin with fewer than
``MINIMUM_BOOTSTRAP_UNITS`` products is omitted rather than given a rate.

The reference distribution is the fragment control's residue-length
distribution, pooled over attempts inside those bins. It is computed from
lengths before any recognition flag is read. Each checkpoint is then
post-stratified onto that one distribution. The fragment is a contiguous
substring of one staged UniRef50 record at the attempt's exact residue length,
and it is the binding reference. The paired difference is the model rate minus
the fragment rate, in successes per attempt. A whole natural record stays a
reference and stays outside this comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.generation import generative_control as gc  # noqa: E402
from src.capability.context.cross_measure_association import LINEAGES  # noqa: E402
from src.capability.generation.generation_evidence import POLICY  # noqa: E402
from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS  # noqa: E402

SCHEMA = "generation_length_sensitivity_v1"
DECLARED_LENGTH_BINS: tuple[tuple[int, int], ...] = tuple(
    (int(low), int(high)) for low, high in POLICY["length_strata"]
)
BIN_NAMES: tuple[str, ...] = tuple(f"{low}_{high}" for low, high in DECLARED_LENGTH_BINS)
MINIMUM_PRODUCTS = int(MINIMUM_BOOTSTRAP_UNITS)
REFERENCE_NAME = (
    "fragment_control_residue_length_distribution_pooled_over_attempts_inside_declared_bins"
)
ENDPOINTS = ("any_family", "complete_domain")
DESCRIPTOR_JSON = (
    REPO_ROOT / "results/R6/generation_followup_20260923/generation_descriptors"
    / "generation_product_descriptors.json"
)
INTERVAL_REASON = (
    "The saved tables carry a resample unit, the frozen near-duplicate sequence "
    "group. paired_rate_contrast resamples that unit at equal weight. The "
    "length-reweighted difference is a post-stratified enumeration, and that "
    "bootstrap does not apply a residue-length weight, so the reweighted "
    "difference has no interval. The equal-weight paired difference on the same "
    "attempts is reported beside it."
)


def bin_of(length: int) -> str | None:
    """The declared residue-length bin containing ``length``, if it has one."""

    for low, high in DECLARED_LENGTH_BINS:
        if low <= int(length) <= high:
            return f"{low}_{high}"
    return None


def reference_bin_masses(lengths: Sequence[int]) -> dict[str, float]:
    """Empirical bin shares of a residue-length sample, on the declared bins.

    Lengths outside the declared bins are outside this distribution. Every
    declared bin is present, including a bin with mass zero.
    """

    counts = {name: 0 for name in BIN_NAMES}
    for length in lengths:
        name = bin_of(int(length))
        if name is not None:
            counts[name] += 1
    total = sum(counts.values())
    if total == 0:
        raise ValueError(
            "the reference length distribution has no attempt inside the declared bins"
        )
    return {name: counts[name] / total for name in BIN_NAMES}


def reweight_rates(
    successes_by_bin: Mapping[str, int],
    counts_by_bin: Mapping[str, int],
    reference_masses: Mapping[str, float],
    *,
    minimum_products: int = MINIMUM_PRODUCTS,
) -> dict[str, Any]:
    """Post-stratify one recognition count onto the reference bin masses.

    A bin with fewer than ``minimum_products`` products is omitted. The
    reference mass that remains on the other bins is renormalized. A bin that
    is omitted contributes no rate.
    """

    unknown = (set(successes_by_bin) | set(counts_by_bin) | set(reference_masses)) - set(BIN_NAMES)
    if unknown:
        raise ValueError(f"unknown length bins: {sorted(unknown)}")
    missing = [name for name in BIN_NAMES if name not in reference_masses]
    if missing:
        raise ValueError(f"reference masses omit {missing}")
    total_mass = float(sum(float(reference_masses[name]) for name in BIN_NAMES))
    if total_mass <= 0 or not math.isfinite(total_mass):
        raise ValueError("reference masses must sum to a positive finite value")

    used: list[str] = []
    excluded: list[dict[str, Any]] = []
    for name in BIN_NAMES:
        mass = float(reference_masses[name])
        if mass < 0:
            raise ValueError(f"{name}: negative reference mass")
        n_products = int(counts_by_bin.get(name, 0))
        successes = int(successes_by_bin.get(name, 0))
        if n_products < 0 or successes < 0 or successes > n_products:
            raise ValueError(f"{name}: {successes} successes in {n_products} products")
        if mass == 0:
            if n_products:
                excluded.append({
                    "bin": name, "n_products": n_products, "reference_mass": mass,
                    "reason": "reference_mass_zero",
                })
            continue
        if n_products < minimum_products:
            excluded.append({
                "bin": name, "n_products": n_products, "reference_mass": mass,
                "reason": "fewer_than_minimum_products",
            })
            continue
        used.append(name)

    retained = float(sum(float(reference_masses[name]) for name in used))
    base = {
        "bins_excluded": excluded,
        "reference_mass_retained": retained / total_mass if total_mass else 0.0,
        "reference_mass_excluded": 1.0 - (retained / total_mass if total_mass else 0.0),
        "minimum_products": int(minimum_products),
    }
    if retained <= 0:
        return {"defined": False, "rate": None, "bins_used": {}, **base}

    bins_used: dict[str, dict[str, Any]] = {}
    rate = 0.0
    for name in used:
        n_products = int(counts_by_bin[name])
        successes = int(successes_by_bin.get(name, 0))
        bin_rate = successes / n_products
        weight = float(reference_masses[name]) / retained
        rate += weight * bin_rate
        bins_used[name] = {
            "n_products": n_products,
            "successes": successes,
            "rate": bin_rate,
            "reference_mass": float(reference_masses[name]),
            "renormalized_weight": weight,
        }
    return {"defined": True, "rate": float(rate), "bins_used": bins_used, **base}


def reweight_difference(
    model_successes: Mapping[str, int],
    fragment_successes: Mapping[str, int],
    counts_by_bin: Mapping[str, int],
    reference_masses: Mapping[str, float],
    *,
    minimum_products: int = MINIMUM_PRODUCTS,
) -> dict[str, Any]:
    """Model-minus-fragment difference after both rates share one reweighting.

    Eligibility depends only on product counts and the reference masses, so the
    two rates keep or drop the same bins.
    """

    model = reweight_rates(
        model_successes, counts_by_bin, reference_masses, minimum_products=minimum_products,
    )
    fragment = reweight_rates(
        fragment_successes, counts_by_bin, reference_masses, minimum_products=minimum_products,
    )
    if model["defined"] != fragment["defined"] or model["bins_excluded"] != fragment["bins_excluded"]:
        raise RuntimeError("model and fragment reweighting dropped different bins")
    shared = {
        "defined": model["defined"],
        "bins_excluded": model["bins_excluded"],
        "reference_mass_retained": model["reference_mass_retained"],
        "reference_mass_excluded": model["reference_mass_excluded"],
        "minimum_products": model["minimum_products"],
    }
    if not model["defined"]:
        return {
            "model_rate": None, "fragment_rate": None, "model_minus_fragment": None,
            "bins_used": {}, **shared,
        }
    bins_used: dict[str, dict[str, Any]] = {}
    for name, block in model["bins_used"].items():
        other = fragment["bins_used"][name]
        bins_used[name] = {
            "n_products": block["n_products"],
            "reference_mass": block["reference_mass"],
            "renormalized_weight": block["renormalized_weight"],
            "model_successes": block["successes"],
            "fragment_successes": other["successes"],
            "model_rate": block["rate"],
            "fragment_rate": other["rate"],
        }
    return {
        "model_rate": model["rate"],
        "fragment_rate": fragment["rate"],
        "model_minus_fragment": float(model["rate"] - fragment["rate"]),
        "bins_used": bins_used,
        **shared,
    }


def missing_inputs(workspace: Path) -> list[str]:
    """Paths the report needs and that are absent. An empty list means readable."""

    required = (
        workspace / "cells",
        workspace / "query_names.json",
        workspace / "build_manifest.json",
        workspace / "oracle",
    )
    missing = [str(path) for path in required if not path.exists()]
    cells = workspace / "cells"
    if cells.is_dir() and not list(cells.glob("*.jsonl")):
        missing.append(str(cells / "*.jsonl"))
    return missing


def _flags(entry: Mapping[str, Any] | None, threshold: float) -> tuple[bool, bool]:
    """``(any_family, complete_domain)`` for one oracle row.

    A query absent from the sequence table was assigned no family. That is the
    same reading ``analyse_generative_control.endpoint_flags`` uses.
    """

    if not entry:
        return False, False
    coverage = entry.get("best_profile_coverage")
    complete = coverage is not None and float(coverage) >= threshold
    return bool(entry.get("families")), complete


def _load_attempts(cells_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(cells_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            fragment = record["sequences"].get("fragment") or ""
            length = int(record["parent_length"])
            searchable = bool(record["parent_searchable"])
            if searchable and len(fragment) != length:
                raise RuntimeError(
                    f"{record['cell']} {record['attempt_id']}: fragment length "
                    f"{len(fragment)} != attempt residue length {length}"
                )
            rows.append({
                "cell": record["cell"],
                "arm": record["arm"],
                "condition": record["condition"],
                "campaign": record["campaign"],
                "attempt_id": record["attempt_id"],
                "group": record["near_duplicate_group"],
                "residue_length": length,
                "searchable": searchable,
                "bin": bin_of(length),
            })
    if not rows:
        raise RuntimeError(f"no attempts under {cells_dir}")
    return rows


def _attach_flags(rows: list[dict[str, Any]], oracle: Mapping[str, Mapping], threshold: float) -> None:
    for row in rows:
        for cohort, side in (("generated", "model"), ("fragment", "fragment")):
            if row["searchable"]:
                key = f"{row['cell']}|{cohort}|{row['attempt_id']}"
                any_family, complete = _flags(oracle.get(key), threshold)
            else:
                any_family, complete = False, False
            row[f"any_family_{side}"] = any_family
            row[f"complete_domain_{side}"] = complete


def _paired(rows: Sequence[Mapping[str, Any]], endpoint: str) -> dict[str, Any]:
    model = [bool(row[f"{endpoint}_model"]) for row in rows]
    fragment = [bool(row[f"{endpoint}_fragment"]) for row in rows]
    groups = [str(row["group"]) for row in rows]
    contrast = gc.paired_rate_contrast(model, fragment, groups)
    contrast["model_successes"] = int(sum(model))
    contrast["fragment_successes"] = int(sum(fragment))
    contrast["control"] = "fragment"
    return contrast


def _outside(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    zero = below = above = 0
    floor = DECLARED_LENGTH_BINS[0][0]
    for row in rows:
        if row["bin"] is not None:
            continue
        length = int(row["residue_length"])
        if length == 0:
            zero += 1
        elif length < floor:
            below += 1
        else:
            above += 1
    return {
        "n_attempts": zero + below + above,
        "n_residue_length_zero": zero,
        "n_below_shortest_declared_bin": below,
        "n_above_longest_declared_bin": above,
        "rate": None,
        "reason": (
            "Attempts outside the declared bins are counted here and are not "
            "given a recognition rate."
        ),
    }


def _length_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lengths = [int(row["residue_length"]) for row in rows if row["searchable"]]
    summary: dict[str, Any] = {
        "population": "searchable_products",
        "n": len(lengths),
        "unit": "residues",
        "quantile_method": "linear",
        "bin_counts": {name: sum(1 for row in rows if row["bin"] == name) for name in BIN_NAMES},
    }
    if not lengths:
        summary.update({"minimum": None, "q25": None, "median": None, "q75": None, "maximum": None})
        return summary
    array = np.asarray(lengths, dtype=float)
    summary.update({
        "minimum": int(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": int(array.max()),
    })
    return summary


def _endpoint_block(
    rows: Sequence[Mapping[str, Any]], endpoint: str, reference_masses: Mapping[str, float],
) -> dict[str, Any]:
    by_bin: dict[str, Any] = {}
    for name in BIN_NAMES:
        subset = [row for row in rows if row["bin"] == name]
        if len(subset) < MINIMUM_PRODUCTS:
            by_bin[name] = {
                "n_attempts": len(subset),
                "n_clusters": len({row["group"] for row in subset}),
                "rate_status": "dropped",
                "reason": "fewer_than_minimum_products",
                "minimum_products": MINIMUM_PRODUCTS,
            }
        else:
            by_bin[name] = _paired(subset, endpoint)

    counts = {name: sum(1 for row in rows if row["bin"] == name) for name in BIN_NAMES}
    model_successes = {
        name: sum(1 for row in rows if row["bin"] == name and row[f"{endpoint}_model"])
        for name in BIN_NAMES
    }
    fragment_successes = {
        name: sum(1 for row in rows if row["bin"] == name and row[f"{endpoint}_fragment"])
        for name in BIN_NAMES
    }
    reweighted = reweight_difference(
        model_successes, fragment_successes, counts, reference_masses,
    )
    reweighted["unit"] = "successes_per_attempt"
    reweighted["interval"] = None
    reweighted["resample_unit"] = "frozen_near_duplicate_sequence_group_of_the_attempt_ledger"
    if reweighted["defined"]:
        used = set(reweighted["bins_used"])
        support = [row for row in rows if row["bin"] in used]
        reweighted["interval_status"] = "enumeration_only"
        reweighted["interval_reason"] = INTERVAL_REASON
        reweighted["unweighted_paired_difference_on_the_same_attempts"] = _paired(support, endpoint)
        reweighted["unweighted_enumeration_on_used_bins"] = {
            "model_successes": sum(block["model_successes"] for block in reweighted["bins_used"].values()),
            "fragment_successes": sum(
                block["fragment_successes"] for block in reweighted["bins_used"].values()
            ),
            "n_attempts": sum(block["n_products"] for block in reweighted["bins_used"].values()),
        }
    else:
        reweighted["interval_status"] = "undefined_no_bin_met_the_minimum"
        reweighted["interval_reason"] = (
            "No declared bin contains enough products to carry a rate, so the "
            "reweighted difference is undefined and has no interval."
        )
        reweighted["unweighted_paired_difference_on_the_same_attempts"] = None
    return {
        "all_attempts": _paired(rows, endpoint),
        "by_bin": by_bin,
        "length_reweighted": reweighted,
    }


def _assert_self_weighted_identity(
    rows: Sequence[Mapping[str, Any]],
    reference_masses: Mapping[str, float],
    endpoints: Mapping[str, Mapping[str, Any]],
) -> None:
    """When the reference is this cohort's own bin shares, reweighting changes nothing."""

    in_bin_lengths = [int(row["residue_length"]) for row in rows if row["bin"] is not None]
    if not in_bin_lengths:
        return
    empirical = reference_bin_masses(in_bin_lengths)
    if any(
        not math.isclose(float(reference_masses[name]), empirical[name], abs_tol=1e-12)
        for name in BIN_NAMES
    ):
        return
    for endpoint in ENDPOINTS:
        block = endpoints[endpoint]["length_reweighted"]
        if not block["defined"] or block["bins_excluded"]:
            continue
        companion = block["unweighted_paired_difference_on_the_same_attempts"]
        for left, right in (
            (block["model_rate"], companion["model_rate"]),
            (block["fragment_rate"], companion["control_rate"]),
            (block["model_minus_fragment"], companion["difference"]),
        ):
            if not math.isclose(float(left), float(right), abs_tol=1e-9):
                raise RuntimeError(
                    f"{endpoint}: post-stratifying onto this cohort's own length "
                    f"distribution moved the rate from {right} to {left}"
                )


def summarise(
    rows: Sequence[Mapping[str, Any]], reference_masses: Mapping[str, float],
) -> dict[str, Any]:
    """Bin rates, equal-weight paired differences, and the reweighted enumeration."""

    if not rows:
        raise ValueError("a summary needs at least one attempt")
    endpoints = {
        endpoint: _endpoint_block(rows, endpoint, reference_masses) for endpoint in ENDPOINTS
    }
    _assert_self_weighted_identity(rows, reference_masses, endpoints)
    return {
        "n_attempts": len(rows),
        "n_searchable": sum(1 for row in rows if row["searchable"]),
        "n_clusters": len({row["group"] for row in rows}),
        "residue_length": _length_summary(rows),
        "outside_declared_bins": _outside(rows),
        "endpoints": endpoints,
    }


def _lineage_of(arm: str) -> str | None:
    matches = [name for name, members in LINEAGES.items() if arm in members]
    if len(matches) > 1:
        raise RuntimeError(f"{arm} belongs to more than one lineage in LINEAGES: {matches}")
    return matches[0] if matches else None


def _group(rows: Sequence[Mapping[str, Any]], reference_masses: Mapping[str, float]) -> dict[str, Any]:
    summary = summarise(rows, reference_masses)
    summary["cells"] = sorted({str(row["cell"]) for row in rows})
    summary["arms"] = sorted({str(row["arm"]) for row in rows})
    summary["conditions"] = sorted({str(row["condition"]) for row in rows})
    summary["campaigns"] = sorted({str(row["campaign"]) for row in rows})
    return summary


def assemble(workspace: Path) -> dict[str, Any]:
    """The length-sensitivity report, or a missing-input record with no rates."""

    workspace = Path(workspace)
    missing = missing_inputs(workspace)
    if missing:
        return {
            "schema": SCHEMA,
            "status": "inputs_missing",
            "missing": missing,
            "checkpoints": None,
            "lineages": None,
            "panel": None,
        }

    print(f"reading attempt ledgers in {workspace / 'cells'}", file=sys.stderr)
    rows = _load_attempts(workspace / "cells")
    # Lengths only. Recognition flags are attached after this distribution is fixed.
    reference_masses = reference_bin_masses(
        [int(row["residue_length"]) for row in rows if row["bin"] is not None]
    )
    print(f"reading oracle tables in {workspace / 'oracle'}", file=sys.stderr)
    reference_counts = {
        name: sum(1 for row in rows if row["bin"] == name) for name in BIN_NAMES
    }
    oracle = gc.collect_oracle(workspace)
    _attach_flags(rows, oracle, gc.COMPLETE_DOMAIN_COVERAGE)
    del oracle

    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[row["cell"]].append(row)

    checkpoints = []
    for cell in sorted(by_cell):
        print(f"summarising {cell}", file=sys.stderr)
        cell_rows = by_cell[cell]
        record = _group(cell_rows, reference_masses)
        record["cell"] = cell
        record["arm"] = cell_rows[0]["arm"]
        record["condition"] = cell_rows[0]["condition"]
        record["lineage"] = _lineage_of(cell_rows[0]["arm"])
        checkpoints.append(record)

    lineages = []
    present_arms = {row["arm"] for row in rows}
    for name in sorted(LINEAGES):
        members = LINEAGES[name]
        member_rows = [row for row in rows if row["arm"] in members]
        entry: dict[str, Any] = {
            "lineage": name,
            "source": "src.capability.context.cross_measure_association.LINEAGES",
            "members_in_map": list(members),
            "members_present": [member for member in members if member in present_arms],
            "members_absent": [member for member in members if member not in present_arms],
        }
        if member_rows:
            entry.update(_group(member_rows, reference_masses))
        else:
            entry["status"] = "no_attempts_for_the_mapped_checkpoints"
        lineages.append(entry)

    manifest = workspace / "build_manifest.json"
    return {
        "schema": SCHEMA,
        "status": "complete",
        "method": {
            "length_bins_source": "src.capability.generation.generation_evidence.POLICY['length_strata']",
            "length_bins": [list(pair) for pair in DECLARED_LENGTH_BINS],
            "minimum_products": MINIMUM_PRODUCTS,
            "minimum_products_source": "src.capability.core.statistics.MINIMUM_BOOTSTRAP_UNITS",
            "drop_rule": (
                "A declared bin with fewer than minimum_products attempts has no "
                "recognition rate."
            ),
            "reference": REFERENCE_NAME,
            "reference_rule": (
                "The fragment control's residue-length distribution, pooled over "
                "every attempt in this workspace whose residue length falls in a "
                "declared bin. Computed from lengths before recognition flags are read."
            ),
            "reweighting": (
                "Each kept bin's recognition rate is weighted by that bin's "
                "reference mass. Mass on a dropped bin is removed and the "
                "remaining mass is renormalized."
            ),
            "paired_difference": "model rate minus fragment rate, in successes per attempt",
            "interval": INTERVAL_REASON,
            "fragment": (
                "A contiguous substring of one staged UniRef50 record at the "
                "attempt's exact residue length. It is the binding reference."
            ),
            "natural_records": (
                "A whole natural record in the length stratum remains a reference "
                "and stays outside this comparison."
            ),
            "token_cap": (
                "The generation protocol's max_new_tokens=400 is a token cap. "
                "The bins are in residues."
            ),
            "complete_domain_coverage_threshold": gc.COMPLETE_DOMAIN_COVERAGE,
            "bootstrap_seed": gc.GATE_SEED,
            "bootstrap_resamples": gc.RESAMPLES,
        },
        "reference_distribution": {
            "name": REFERENCE_NAME,
            "counts": reference_counts,
            "masses": reference_masses,
            "n_attempts_inside_bins": sum(reference_counts.values()),
            "fragment_length_equals_attempt_residue_length": True,
            "n_searchable_attempts_checked": sum(1 for row in rows if row["searchable"]),
        },
        "lineage_map": {
            "source": "src.capability.context.cross_measure_association.LINEAGES",
            "map": {name: list(members) for name, members in LINEAGES.items()},
            "checkpoints_without_a_declared_lineage": sorted(
                {row["arm"] for row in rows if _lineage_of(row["arm"]) is None}
            ),
        },
        "inputs": {
            "workspace": str(workspace),
            "build_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "n_cell_ledgers": len(list((workspace / "cells").glob("*.jsonl"))),
            "descriptor_json": {
                "path": str(DESCRIPTOR_JSON),
                "present": DESCRIPTOR_JSON.is_file(),
                "used": False,
                "reason": (
                    "That file has per-product residue length and any_profile_hit "
                    "for four checkpoints. It has no complete-domain flag and no "
                    "fragment control, so the fragment reweighting cannot be "
                    "computed from it."
                ),
            },
        },
        "checkpoints": checkpoints,
        "lineages": lineages,
        "panel": _group(rows, reference_masses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path,
        default=REPO_ROOT / "archive/logs/R6/gate_generative_control/build",
    )
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "results/R6/generation_length_sensitivity_20260926/length_sensitivity.json",
    )
    args = parser.parse_args()
    payload = assemble(args.workspace)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if payload["status"] != "complete":
        print(json.dumps({"status": payload["status"], "missing": payload["missing"],
                          "out": str(args.out)}, indent=2))
        return
    panel = payload["panel"]["endpoints"]
    print(json.dumps({
        "status": "complete",
        "out": str(args.out),
        "n_attempts": payload["panel"]["n_attempts"],
        "any_family_model_successes": panel["any_family"]["all_attempts"]["model_successes"],
        "any_family_fragment_successes": panel["any_family"]["all_attempts"]["fragment_successes"],
        "complete_domain_model_successes": panel["complete_domain"]["all_attempts"]["model_successes"],
        "complete_domain_fragment_successes": (
            panel["complete_domain"]["all_attempts"]["fragment_successes"]
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
