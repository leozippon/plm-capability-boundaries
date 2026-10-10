#!/usr/bin/env python3
"""Read the decoding sweep: per candidate, at matched compute, at matched length.

Inputs are all frozen artefacts. The sweep ledger comes from
``build_decoding_cohort.py``; the structural evidence comes from this project's
ESMFold2 instrument; and the natural band and the existing generation results
are read from the fold trees that **already exist** on GPFS, by literal path, at
one evaluation signature, rather than recomputed.

The four readings this stage produces, and why each is necessary.

**Per candidate.** Each configuration's mean evaluator value with an interval
whose unit is the cluster. This is what a sweep usually reports and on its own
it cannot answer the question, because a configuration that draws more
candidates is not comparable with one that draws fewer.

**At matched compute.** Every best-of-k point carries the generation tokens and
the folds it spent, and the sweep is re-read at equal tokens per kept candidate.
Without this, "sample more and keep the best" wins by construction.

**At matched length.** Predicted confidence rises steeply with length, so every
comparison has a length-matched counterpart: each candidate's value minus the
natural mean in its *own* length stratum, the natural band standardised to each
configuration's own length distribution, the project's paired whole-natural
comparator where the folded natural pool can supply one, and a
length-standardised configuration-versus-configuration reading.

**Against degeneracy.** Low-temperature sampling wins by collapsing onto
repeats and onto near-copies of natural sequences. Duplication, composition
entropy, homopolymer content, k-mer repertoire and corpus identity are reported
for every configuration against a size-matched draw from the programme's current
operating point, so a gain bought that way is visible as such.

A panel-wide claim -- "no configuration reaches the natural band" -- is read off
a joint bootstrap over the clusters every configuration of an arm shares, not off
a pile of marginal intervals, which are not a simultaneous statement.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.core.statistics import mean_interval  # noqa: E402
from src.capability.decoding import decoding_sweep as ds  # noqa: E402
from src.capability.evaluation import generated_phenotype as gp  # noqa: E402

COMPLETION = "decoding_sweep.json"

#: The evaluators the whole table is built on. The primary is CA pLDDT because
#: it is the quantity the existing generation results are stated in; the event is
#: carried beside it because a mean can move without any candidate crossing the
#: project's operational threshold.
REPORTED = (ds.PRIMARY_EVALUATOR, "predicted_confidence_event")


def _read_index_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        target = Path(path)
        files = sorted(target.glob("index-*.jsonl")) if target.is_dir() else [target]
        if not files:
            raise SystemExit(f"{target} holds no index-*.jsonl written by the structure instrument")
        for item in files:
            rows.extend(ds.read_jsonl(item))
    return rows


def _attach(rows: Sequence[Mapping[str, Any]], indices: Sequence[Path]) -> tuple[list[dict], dict]:
    """Join the ledger to its folds by candidate identity, keeping every attempt."""

    folded: dict[str, dict[str, Any]] = {}
    signatures: set[str] = set()
    for row in _read_index_rows(indices):
        lifted = ds.lift_structure(row)
        if lifted is None:
            continue
        identifier = str(row["id"])
        if identifier in folded:
            raise SystemExit(f"the structure indices carry {identifier} more than once")
        folded[identifier] = lifted
        if lifted["evaluation_signature"]:
            signatures.add(str(lifted["evaluation_signature"]))
    joined = []
    for row in rows:
        merged = dict(row)
        structure = folded.get(str(row["id"]))
        merged["folded"] = structure is not None
        if structure is not None:
            merged.update(structure)
        joined.append(merged)
    report = {
        "n_rows": len(joined),
        "n_folded": sum(1 for row in joined if row["folded"]),
        "n_in_band": sum(1 for row in joined if row.get("in_band")),
        "evaluation_signatures": sorted(signatures),
    }
    if len(signatures) > 1:
        raise SystemExit(
            f"the swept folds mix {len(signatures)} evaluation signatures; a single "
            "measurement contract is what makes these configurations comparable"
        )
    missing = report["n_in_band"] - report["n_folded"]
    if missing:
        raise SystemExit(
            f"{missing} in-band products have no fold. The structural reading would be "
            "conditioned on whichever candidates happened to be folded; fold them or "
            "narrow the cohort"
        )
    return joined, report


def _natural_pool(
    paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], dict[str, list[dict]], set[str]]:
    """The folded natural records, and the existing generation results beside them."""

    natural: list[dict[str, Any]] = []
    existing: dict[str, list[dict[str, Any]]] = {}
    signatures: set[str] = set()
    for row in _read_index_rows(paths):
        lifted = ds.lift_structure(row)
        if lifted is None:
            continue
        record = {
            "id": str(row["id"]),
            "length": int(row["length"]),
            "sequence": str(row["sequence"]),
            "role": str(row.get("role")),
            "arm": row.get("arm"),
            "stratum": row.get("stratum"),
            **lifted,
        }
        if lifted["evaluation_signature"]:
            signatures.add(str(lifted["evaluation_signature"]))
        if record["role"] == "natural":
            natural.append(record)
        elif record["role"] == "generated" and record["arm"] in ds.ARM_NAMES:
            existing.setdefault(str(record["arm"]), []).append(record)
    if not natural:
        raise SystemExit("the supplied natural indices carry no folded natural record")
    return natural, existing, signatures


def _stratum_means(natural: Sequence[Mapping[str, Any]], field: str) -> dict[int, dict[str, Any]]:
    buckets: dict[int, list[float]] = {}
    for row in natural:
        buckets.setdefault(ds._length_stratum(int(row["length"])), []).append(float(row[field]))
    return {
        key: {"mean": float(np.mean(values)), "n": len(values)}
        for key, values in sorted(buckets.items())
    }


def _delta_rows(
    rows: Sequence[Mapping[str, Any]], strata: Mapping[int, Mapping[str, Any]], field: str
) -> list[dict[str, Any]]:
    """Each folded candidate's value minus the natural mean in its own length stratum.

    The per-candidate length-matched gap. Candidates in a stratum the natural
    pool barely populates are dropped here and counted, because a gap against a
    comparator built from a handful of records is not a length control.
    """

    out: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("folded"):
            continue
        key = ds._length_stratum(int(row["length"]))
        block = strata.get(key)
        if block is None or block["n"] < ds.MIN_NATURAL_PER_BIN:
            continue
        item = dict(row)
        item["delta"] = float(row[field]) - float(block["mean"])
        item["natural_stratum_mean"] = float(block["mean"])
        item["natural_stratum_n"] = int(block["n"])
        out.append(item)
    return out


def _novelty(path: Path | None) -> dict[str, dict[str, Any]]:
    """Identity over query and its declared stratum, keyed by candidate.

    The stratum is read off the novelty artefact rather than recomputed, so the
    band edges here are the ones that stage declares.
    """

    if path is None:
        return {}
    return {
        str(row["id"]): {
            "identity": float(row["nearest_corpus_identity"]),
            "stratum": str(row["identity_stratum"]),
        }
        for row in ds.read_jsonl(path)
    }


def _arm_report(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    natural: Sequence[Mapping[str, Any]],
    existing: Sequence[Mapping[str, Any]],
    identity: Mapping[str, float],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    by_config: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_config.setdefault(str(row["config_key"]), []).append(row)
    report: dict[str, Any] = {
        "arm": name,
        "note": ds.arm(name).note,
        "conditioned": ds.arm(name).conditioned,
        "n_attempts": len(rows),
        "n_folded": sum(1 for row in rows if row.get("folded")),
        "census": {},
        "per_candidate": {},
        "natural_band": {},
        "selection": {},
        "degeneracy": {},
        "novelty": {},
    }

    strata = {field: _stratum_means(natural, field) for field in REPORTED}
    report["natural_pool"] = {
        "n_records": len(natural),
        "comparator_kind": "whole Swiss-Prot entries, already folded at one evaluation signature",
        "length_strata": {
            field: {str(key): block for key, block in strata[field].items()} for field in REPORTED
        },
        "per_record": {
            field: mean_interval([float(row[field]) for row in natural]) for field in REPORTED
        },
    }

    # The existing generation results, read through exactly this arithmetic so
    # the sweep can be placed beside them rather than described as beating them.
    if existing:
        report["existing_generation_results"] = {
            "n_records": len(existing),
            "source": "the generation evaluation cohort's folded products for this arm",
            "length_mean": float(np.mean([int(row["length"]) for row in existing])),
            "per_candidate": {
                field: mean_interval([float(row[field]) for row in existing])
                for field in REPORTED
            },
            "length_matched_gap": {
                field: (
                    mean_interval(
                        [
                            row["delta"]
                            for row in _delta_rows(
                                [dict(item, folded=True) for item in existing], strata[field], field
                            )
                        ]
                    )
                    if _delta_rows(
                        [dict(item, folded=True) for item in existing], strata[field], field
                    )
                    else None
                )
                for field in REPORTED
            },
        }

    reference_rows = [
        row
        for row in by_config.get(ds.REFERENCE_KEY, ())
        if row.get("folded")
    ]

    per_cluster_delta: dict[str, dict[str, dict[str, float]]] = {field: {} for field in REPORTED}
    curves: dict[str, dict[str, list[dict[str, Any]]]] = {field: {} for field in REPORTED}
    standardisable: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        field: {} for field in REPORTED
    }

    for key in ds.CONFIG_KEYS:
        setting = ds.config(key)
        cell = by_config.get(key, [])
        if not cell:
            raise SystemExit(f"{name}: the ledger carries no draw for configuration {key}")
        report["census"][key] = ds.config_census(cell)
        folded = [row for row in cell if row.get("folded")]
        block: dict[str, Any] = {"config": setting.record(), "n_folded": len(folded)}
        for field in REPORTED:
            if len(folded) >= 2:
                block[field] = ds.cluster_level_mean(folded, field=field)
                standardisable[field][key] = folded
                deltas = _delta_rows(folded, strata[field], field)
                block[field]["length_matched"] = {
                    "n_candidates": len(deltas),
                    "n_dropped_unsupported_stratum": len(folded) - len(deltas),
                    "mean_gap": float(np.mean([row["delta"] for row in deltas])) if deltas else None,
                    "estimand": (
                        "the candidate's confidence minus the folded natural pool's mean "
                        "confidence in the same 20-residue length stratum. The natural "
                        "side is treated as known here; its own sampling error is carried "
                        "by the standardised band instead"
                    ),
                }
                if deltas:
                    per_cluster_delta[field][key] = ds.cluster_means(deltas, field="delta")
            else:
                block[field] = {
                    "n_folded": len(folded),
                    "resolved": False,
                    "unresolved_reason": (
                        "fewer than two in-band products: this configuration produced "
                        "almost nothing the structure instrument can evaluate, which is "
                        "the result for it"
                    ),
                }
        report["per_candidate"][key] = block

        band: dict[str, Any] = {}
        for field in REPORTED:
            if len(folded) >= 2:
                band[field] = {
                    "standardised": ds.natural_standardised_band(
                        folded, natural, field=field, seed=seed + 11, n_bootstrap=resamples
                    ),
                    "paired_whole_natural": ds.natural_band_contrast(
                        folded, natural, field=field, seed=seed + 23, n_bootstrap=resamples
                    ),
                }
            else:
                band[field] = {"resolved": False, "unresolved_reason": "fewer than two in-band products"}
        report["natural_band"][key] = band

        budgets = ds.k_values(setting.draws_per_cluster)
        report["selection"][key] = {
            "k_values": list(budgets),
            "draws_per_cluster": setting.draws_per_cluster,
            "curves": {},
        }
        for field in REPORTED:
            for selector in ("model_logprob_per_token", "model_logprob_total", "esmfold2_confidence_oracle"):
                curve = ds.selection_curve(
                    cell,
                    selector=selector,
                    field=field,
                    k_values=budgets,
                    draws_per_cluster=setting.draws_per_cluster,
                )
                report["selection"][key]["curves"].setdefault(field, {})[selector] = [
                    {item: point[item] for item in point if item != "selected_ids"}
                    for point in curve
                ]
                if selector == "model_logprob_per_token":
                    curves[field][key] = curve

        in_band_rows = [row for row in cell if row.get("in_band")]
        sequences = [str(row["sequence"]) for row in in_band_rows]
        if len(sequences) >= 2:
            found = [identity.get(str(row["id"])) for row in in_band_rows]
            usable = (
                None if any(value is None for value in found) else [item["identity"] for item in found]
            )
            identity_strata = (
                None if any(value is None for value in found) else [item["stratum"] for item in found]
            )
            profile = ds.set_profile(sequences, corpus_identity=usable)
            reference_sequences = [str(row["sequence"]) for row in reference_rows]
            reference_found = [identity.get(str(row["id"])) for row in reference_rows]
            reference_identity = (
                None
                if any(value is None for value in reference_found) or not reference_found
                else [item["identity"] for item in reference_found]
            )
            size = min(len(sequences), len(reference_sequences))
            comparison = (
                ds.size_matched_reference(
                    reference_sequences,
                    size=size,
                    seed=seed + 41,
                    corpus_identity=reference_identity,
                )
                if size >= 2 and len(reference_sequences) >= size
                else None
            )
            natural_reference = (
                ds.size_matched_reference(
                    [str(row["sequence"]) for row in natural], size=min(size, len(natural)), seed=seed + 57
                )
                if size >= 2
                else None
            )
            gain = None
            if key != ds.REFERENCE_KEY:
                here = report["per_candidate"][key].get(ds.PRIMARY_EVALUATOR, {}).get("mean")
                there = (
                    report["per_candidate"]
                    .get(ds.REFERENCE_KEY, {})
                    .get(ds.PRIMARY_EVALUATOR, {})
                    .get("mean")
                )
                if here is not None and there is not None:
                    gain = float(here) - float(there)
            report["degeneracy"][key] = {
                "profile": profile,
                "size_matched_reference_configuration": comparison,
                "size_matched_natural_reference": natural_reference,
                "reference_configuration": ds.REFERENCE_KEY,
                "gain_over_reference_configuration": gain,
                "verdict": (
                    ds.degeneracy_verdict(profile, comparison, gain=gain)
                    if comparison is not None
                    else None
                ),
            }
            if usable is not None:
                report["novelty"][key] = {
                    "n_searched": len(usable),
                    "mean_identity_over_query": float(np.mean(usable)),
                    "max_identity_over_query": float(np.max(usable)),
                    "identity_interval": mean_interval(usable)["interval"],
                    "fraction_near_duplicate": float(
                        np.mean(np.asarray(usable) >= gp.NEAR_DUPLICATE_IDENTITY)
                    ),
                    "near_duplicate_threshold": float(gp.NEAR_DUPLICATE_IDENTITY),
                    "strata": dict(sorted(Counter(identity_strata).items())),
                    "definition": (
                        "percent of the query identically matched by its closest UniRef50 "
                        "relative, from the project's own novelty scan. Reference-corpus "
                        "coverage, not pretraining exposure"
                    ),
                }
        else:
            report["degeneracy"][key] = {
                "resolved": False,
                "unresolved_reason": "fewer than two in-band products to profile",
            }

    report["simultaneous"] = {
        field: ds.joint_cluster_bootstrap(
            per_cluster_delta[field], seed=seed + 71, n_bootstrap=resamples
        )
        for field in REPORTED
        if per_cluster_delta[field]
    }
    report["length_standardised"] = {
        field: ds.length_standardised(
            standardisable[field], field=field, seed=seed + 83, n_bootstrap=max(500, resamples // 4)
        )
        for field in REPORTED
        if standardisable[field]
    }
    reference_tokens = (
        report["census"][ds.REFERENCE_KEY]["generation_tokens"]["mean_per_draw"]
    )
    report["matched_compute"] = {
        field: ds.matched_compute_table(
            curves[field],
            budgets=[reference_tokens * factor for factor in (1.0, 4.0, 16.0)],
        )
        for field in REPORTED
        if curves[field]
    }
    report["matched_compute_budgets"] = {
        "reference_configuration": ds.REFERENCE_KEY,
        "reference_tokens_per_draw": float(reference_tokens),
        "factors": [1.0, 4.0, 16.0],
        "selector": "model_logprob_per_token",
    }
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    gp.require_fresh_out(args.out, COMPLETION)
    ledger = ds.read_jsonl(args.ledger)
    joined, join_report = _attach(ledger, args.structure)
    natural, existing, reference_signatures = _natural_pool(args.natural_structure)
    # One measurement contract across the sweep and the band it is read against.
    # A reference band folded under a different contract is a different
    # instrument, and the gap to it would be partly a gap between instruments.
    if set(join_report["evaluation_signatures"]) != reference_signatures:
        raise SystemExit(
            "the swept folds carry evaluation signature(s) "
            f"{sorted(join_report['evaluation_signatures'])} while the reference band "
            f"carries {sorted(reference_signatures)}. The structural comparison would "
            "cross two measurement contracts and is refused"
        )
    join_report["reference_evaluation_signatures"] = sorted(reference_signatures)
    identity = _novelty(args.novelty)

    arms = sorted({str(row["arm"]) for row in joined})
    report = {
        "schema_version": ds.SCHEMA_VERSION,
        "status": "complete",
        "campaign": ds.CAMPAIGN,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "question": (
            "with weights completely fixed, can a better decoding strategy recover "
            "generative performance?"
        ),
        "grid": [setting.record() for setting in ds.GRID],
        "n_configurations": len(ds.GRID),
        "reference_configuration": ds.REFERENCE_KEY,
        "historical_unconditioned_configuration": ds.HISTORICAL_UNCONDITIONED_KEY,
        "deep_budget_configurations": list(ds.DEEP_BUDGET_KEYS),
        "evaluation_band": list(ds.EVALUATION_BAND),
        "evaluators": dict(ds.EVALUATORS),
        "reported_evaluators": list(REPORTED),
        "selectors": {key: dict(value) for key, value in ds.SELECTORS.items()},
        "join": join_report,
        "sources": {
            "ledger": {"path": str(args.ledger), "sha256": sha256_file(args.ledger)},
            "structure": [str(path) for path in args.structure],
            "natural_structure": [str(path) for path in args.natural_structure],
            "novelty": None if args.novelty is None else str(args.novelty),
        },
        "novelty_available": bool(identity),
        "seed": int(args.seed),
        "resamples": int(args.resamples),
        "arms": {},
        "ceiling": dict(ds.CEILING),
    }
    for name in arms:
        report["arms"][name] = _arm_report(
            name,
            [row for row in joined if row["arm"] == name],
            natural,
            existing.get(name, []),
            identity,
            seed=args.seed,
            resamples=args.resamples,
        )
    report["verdict"] = _verdict(report)
    write_json(args.out / COMPLETION, report)
    return report


def _verdict(report: Mapping[str, Any]) -> dict[str, Any]:
    """One answer per arm, stated in the terms the question was asked in."""

    out: dict[str, Any] = {}
    for name, block in report["arms"].items():
        field = ds.PRIMARY_EVALUATOR
        simultaneous = block.get("simultaneous", {}).get(field, {}).get("simultaneous", {})
        bound = simultaneous.get("max_over_configurations")
        marginal = block.get("simultaneous", {}).get(field, {}).get("marginal", {})
        resolved = {
            key: value for key, value in marginal.items() if value.get("interval") is not None
        }
        best = max(resolved, key=lambda key: resolved[key]["mean"], default=None)
        reference = marginal.get(ds.REFERENCE_KEY, {})
        out[name] = {
            "primary_evaluator": field,
            "quantity": (
                "the configuration's mean length-matched gap: candidate confidence minus "
                "the folded natural pool's confidence in the same length stratum. Zero is "
                "the natural band"
            ),
            "reference_configuration_gap": reference.get("mean"),
            "reference_configuration_interval": reference.get("interval"),
            "best_configuration": best,
            "best_configuration_gap": None if best is None else resolved[best]["mean"],
            "best_configuration_interval": None if best is None else resolved[best]["interval"],
            "improvement_over_reference": (
                None
                if best is None or reference.get("mean") is None
                else float(resolved[best]["mean"] - reference["mean"])
            ),
            "any_configuration_reaches_natural_band": (
                None if bound is None else bool(bound["upper_bound"] > 0.0)
            ),
            "simultaneous_upper_bound_on_best_gap": None if bound is None else bound["upper_bound"],
            "simultaneous_meaning": (
                "a one-sided 97.5% bound on the largest length-matched gap any eligible "
                "configuration attains, from one joint bootstrap over the clusters every "
                "configuration shares. An upper bound below zero is a simultaneous "
                "statement that no configuration in this grid reaches the natural band"
            ),
            "degeneracy_of_best": (
                None
                if best is None
                else block.get("degeneracy", {}).get(best, {}).get("verdict")
            ),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--structure", nargs="+", type=Path, required=True)
    parser.add_argument("--natural-structure", nargs="+", type=Path, required=True)
    parser.add_argument("--novelty", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--device", default="cpu", help="accepted because the campaign queue injects it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "verdict": report["verdict"]}, sort_keys=True))


if __name__ == "__main__":
    main()
