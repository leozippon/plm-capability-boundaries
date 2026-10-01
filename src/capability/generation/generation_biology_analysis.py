"""Reconstructable, score-independent generated-output evidence reporting.

This module does no model inference. Missing structural outcomes are not failures
of biology; partial result files are refused, and event coverage is explicit.
"""

from __future__ import annotations

import hashlib
import json
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..core.amino_acids import AA20 as _AA20_ALPHABET
from ..core.statistics import MINIMUM_BOOTSTRAP_UNITS

SEED = 20260905
RESAMPLES = 4000
AA20 = frozenset(_AA20_ALPHABET)
TERMINAL = {"ok", "failed", "not_evaluable"}
KEYS = ("arm", "class_key", "condition", "role")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def index_rows(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        key = row["id"]
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f"Missing or duplicate row ID: {key!r}")
        result[key] = row
    return result


def merge_reference_annotations(rows: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join recoverable reference metadata without rewriting frozen input files."""
    index = index_rows(annotations)
    result = []
    for row in rows:
        extra = index.get(row["id"])
        if extra is None:
            result.append(dict(row))
            continue
        if extra["sequence_sha256"] != row["sequence_sha256"]:
            raise ValueError("Reference annotation belongs to another sequence")
        before, after = row.get("reference_identity"), extra.get("reference_identity")
        if (before is None) != (after is None) or (before is not None and not np.isclose(before, after, atol=1e-9, rtol=0)):
            raise ValueError("Reference sidecar changed the original identity estimate")
        coverage = extra.get("reference_coverage")
        if coverage is not None and (not np.isfinite(coverage) or not 0 <= coverage <= 1):
            raise ValueError("Reference query coverage outside 0–1")
        result.append(dict(row) | {k: v for k, v in extra.items() if k.startswith("reference_")})
    return result


def supported(row: dict[str, Any]) -> bool:
    seq = row["sequence"]
    return 16 <= len(seq) <= 1024 and set(seq) <= AA20


def cell_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in KEYS)


def weighted_mean(values: list[float], weights: list[float]) -> float | None:
    if not values:
        return None
    if len(values) != len(weights) or any(w <= 0 for w in weights):
        raise ValueError("Weights must be positive and align with values")
    return float(np.average(values, weights=weights))


def class_interval(values: list[float], *, seed: int = SEED,
                   resamples: int = RESAMPLES) -> dict[str, Any]:
    if not values or not np.isfinite(values).all():
        raise ValueError("Class estimates must be nonempty and finite")
    result = {"mean": float(np.mean(values)), "n_classes": len(values),
              "ci95": None, "ci97_5": None, "resamples": resamples,
              "seed": seed, "unit": "class"}
    # The class is the resampling unit, so the shared floor governs this
    # interval; imported rather than restated. The status string is a frozen
    # artefact key and stays spelled out.
    if len(values) < MINIMUM_BOOTSTRAP_UNITS:
        result["interval_status"] = "fewer_than_eight_classes"
        return result
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    means = array[rng.integers(0, len(array), (resamples, len(array)))].mean(axis=1)
    result.update(ci95=np.quantile(means, [.025, .975]).tolist(),
                  ci97_5=np.quantile(means, [.0125, .9875]).tolist(),
                  interval_status="declared_class_cohort")
    return result


def sequence_cluster_interval(records: list[dict[str, Any]], *,
                              seed: int = SEED, resamples: int = RESAMPLES) -> dict[str, Any]:
    """One native task: resample paired sequence groups with fixed survey weights."""
    grouped = defaultdict(lambda: [0., 0.])
    for row in records:
        if row["paired_delta"] is not None:
            weight = 1 / row["inclusion_probability"]
            group = grouped[row["near_duplicate_group"]]
            group[0] += weight * row["paired_delta"]
            group[1] += weight
    if not grouped:
        return {"mean": None, "ci95": None, "ci97_5": None,
                "n_clusters": 0, "unit": "sampled_near_duplicate_sequence_group"}
    values = np.array(list(grouped.values()), dtype=float)
    result = {"mean": float(values[:, 0].sum() / values[:, 1].sum()),
              "ci95": None, "ci97_5": None, "n_clusters": len(values),
              "unit": "sampled_near_duplicate_sequence_group", "seed": seed,
              "resamples": resamples,
              "scope": "fixed_native_task_and_sampling_configuration_not_family_generalization"}
    if len(values) < MINIMUM_BOOTSTRAP_UNITS:
        result["interval_status"] = "fewer_than_eight_sequence_groups"
        return result
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (resamples, len(values)))].sum(axis=1)
    means = draws[:, 0] / draws[:, 1]
    result.update(ci95=np.quantile(means, [.025, .975]).tolist(),
                  ci97_5=np.quantile(means, [.0125, .9875]).tolist(),
                  interval_status="weighted_sequence_cluster_bootstrap")
    return result


def validate_predictions(subset: list[dict[str, Any]],
                         predictions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    expected, actual = index_rows(subset), index_rows(predictions)
    if expected.keys() != actual.keys():
        raise ValueError(f"Prediction IDs incomplete or unexpected: "
                         f"missing={len(expected.keys() - actual.keys())}, "
                         f"extra={len(actual.keys() - expected.keys())}")
    result = {}
    for key, row in expected.items():
        sequence = row["sequence"]
        digest = hashlib.sha256(sequence.encode()).hexdigest()
        if row["sequence_sha256"] != digest or row["length"] != len(sequence):
            raise ValueError(f"Input sequence identity mismatch: {key}")
        prediction = actual[key]
        body = prediction["structure"]
        if body["status"] not in TERMINAL:
            raise ValueError(f"Nonterminal structural outcome: {key}")
        if (prediction.get("sequence_sha256", digest) != digest
                or body["sequence_sha256"] != digest or body["length"] != len(sequence)):
            raise ValueError(f"Prediction sequence identity mismatch: {key}")
        if body["status"] == "ok":
            ca = np.asarray(body["ca_plddt"], dtype=float)
            if ca.shape != (len(sequence),) or not np.isfinite(ca).all():
                raise ValueError(f"CA confidence coverage mismatch: {key}")
            if not supported(row) or np.any(ca < 0) or np.any(ca > 100):
                raise ValueError(f"Unsupported input or confidence outside 0–100: {key}")
            mean, frac = float(ca.mean()), float(np.mean(ca >= 70))
            if not np.isclose(mean, body["mean_ca_plddt"], atol=1e-4, rtol=0):
                raise ValueError(f"Mean confidence cannot be reconstructed: {key}")
            if not np.isclose(frac, body["fraction_ca_plddt_ge70"], atol=1e-6, rtol=0):
                raise ValueError(f"Confidence fraction cannot be reconstructed: {key}")
            ptm = float(body["ptm"])
            if not np.isfinite(ptm) or not 0 <= ptm <= 1:
                raise ValueError(f"Invalid pTM: {key}")
            body = dict(body, confidence_event=bool(mean >= 70 and frac >= .8))
        result[key] = body
    return result


def profile_ledger(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells = defaultdict(list)
    for row in attempts:
        if row["role"] in {"generation", "unconditioned_floor", "natural_reference"}:
            cells[cell_key(row)].append(row)
    records = []
    for key, rows in sorted(cells.items()):
        n = len(rows)
        target = [r for r in rows if r.get("target_profile_hit") is True]
        any_hit = [r for r in rows if r.get("any_profile_hit") is True]
        def group(record):
            return str(record.get("near_duplicate_group", record["sequence_sha256"]))
        confusion = Counter(label for r in rows for label in (r.get("profile_hit_classes") or []))
        confusion["__no_profile__"] = sum(r.get("any_profile_hit") is False for r in rows)
        confusion["__profile_unknown__"] = sum(r.get("any_profile_hit") is None for r in rows)
        confusion["__multiple_classes__"] = sum(len(r.get("profile_hit_classes") or []) > 1 for r in rows)
        searches = Counter(str(r.get("reference_search_status", "missing")) for r in rows)
        known_target = sum(isinstance(r.get("target_profile_hit"), bool) for r in rows)
        records.append(dict(zip(KEYS, key)) | {
            "primary_class": all(bool(r.get("primary_class")) for r in rows),
            "n_attempts": n, "n_supported": sum(supported(r) for r in rows),
            "n_empty": sum(not r["sequence"] for r in rows),
            "n_exact_sequences": len({r["sequence_sha256"] for r in rows}),
            "n_near_duplicate_groups": len({group(r) for r in rows}),
            "n_target_profile": len(target), "target_profile_known": known_target,
            "target_profile_rate": len(target) / n if known_target == n else None,
            "any_profile_rate": len(any_hit) / n if all(isinstance(r.get("any_profile_hit"), bool) for r in rows) else None,
            "n_distinct_target_groups": len({group(r) for r in target}) if known_target == n else None,
            "distinct_target_groups_per_attempt": len({group(r) for r in target}) / n if known_target == n else None,
            "confusion_counts_multilabel": dict(confusion),
            "reference_search_status_counts": dict(searches),
            "reference_coverage_missing": sum(r.get("reference_coverage") is None for r in rows),
            "length_quantiles": np.quantile([r["length"] for r in rows], [0, .25, .5, .75, 1]).tolist(),
        })
    return records


def _validate_sampling(population: list[dict[str, Any]], selected: list[dict[str, Any]]) -> None:
    if not selected or selected[0]["role"] != "generation":
        return
    sizes, samples = Counter(), Counter()
    for row in population:
        if supported(row):
            sizes[row["stratum"]] += 1
    for row in selected:
        if not supported(row):
            raise ValueError("Unsupported generation selected for structure")
        samples[row["stratum"]] += 1
    if sizes.keys() != samples.keys():
        raise ValueError("A supported generation stratum was not sampled")
    for row in selected:
        expected = samples[row["stratum"]] / sizes[row["stratum"]]
        if not np.isclose(row["inclusion_probability"], expected, atol=1e-12, rtol=0):
            raise ValueError("Incorrect generation inclusion probability")


def _joint_bounds(population: list[dict[str, Any]], selected: list[dict[str, Any]],
                  scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Calibrate the joint event to known profile marginals, never exceed them."""
    if not all(isinstance(r.get("target_profile_hit"), bool) for r in population):
        return {"status": "profile_marginal_unknown"}
    targets = defaultdict(list)
    for row in population:
        if row["target_profile_hit"]:
            targets[row["stratum"]].append(row)
    lower = upper = 0.
    unknown_strata = []
    for stratum, target_rows in targets.items():
        sampled = [r for r in selected if r["stratum"] == stratum and r["target_profile_hit"]]
        share = len(target_rows) / len(population)
        if not sampled:
            upper += share
            unknown_strata.append(stratum)
            continue
        weights = [1 / r["inclusion_probability"] for r in sampled]
        lo, hi = [], []
        for row in sampled:
            score = scores[row["id"]]
            if score["status"] == "ok":
                lo.append(float(score["confidence_event"]))
                hi.append(float(score["confidence_event"]))
            else:
                lo.append(0.)
                hi.append(1.)
        lower += share * weighted_mean(lo, weights)
        upper += share * weighted_mean(hi, weights)
    return {"status": "profile_calibrated_sample_estimate",
            "lower_estimate": lower, "upper_estimate": upper,
            "known_profile_rate": sum(len(v) for v in targets.values()) / len(population),
            "unsampled_target_strata": unknown_strata,
            "is_exact_census_bound": False}


def analyze(attempts: list[dict[str, Any]], subset: list[dict[str, Any]],
            predictions: list[dict[str, Any]], *, phase: str,
            calibration: dict[str, Any] | None = None,
            resamples: int = RESAMPLES, primary_condition: str = "requested",
            uncertainty_unit: str = "class") -> dict[str, Any]:
    if uncertainty_unit not in {"class", "sequence_cluster"}:
        raise ValueError("Unknown uncertainty unit")
    full = index_rows(attempts)
    scores = validate_predictions(subset, predictions)
    parents, pairs = [], {}
    for row in subset:
        if row["phase"] != phase:
            raise ValueError("Input phase does not match analysis phase")
        if row["role"] == "composition_shuffle":
            parent = row["paired_id"]
            if parent in pairs:
                raise ValueError("Multiple shuffled controls for one parent")
            pairs[parent] = row
        else:
            if row["id"] not in full:
                raise ValueError("Selected parent absent from complete ledger")
            if full[row["id"]]["sequence_sha256"] != row["sequence_sha256"]:
                raise ValueError("Selected parent changed sequence")
            probability = row["inclusion_probability"]
            if probability is None or not 0 < probability <= 1:
                raise ValueError("Invalid parent inclusion probability")
            parents.append(row)
    if {r["id"] for r in parents} != pairs.keys():
        raise ValueError("Every selected parent needs exactly one shuffle")
    if phase == "pilot" and any(r["role"] != "natural_reference" for r in parents):
        raise ValueError("Calibration pilot must contain only natural controls")
    if phase == "main" and calibration is None:
        raise ValueError("Main analysis needs the frozen pilot report")
    pilot_hashes = set(calibration.get("natural_control_sequence_sha256", [])) if calibration else set()
    if phase == "main" and any(r["role"] == "natural_reference" and r["sequence_sha256"] in pilot_hashes for r in parents):
        raise ValueError("Main natural controls overlap the pilot")
    populations, selected_cells = defaultdict(list), defaultdict(list)
    for row in attempts:
        if phase == "main" and row["role"] == "natural_reference" and row["sequence_sha256"] in pilot_hashes:
            continue
        populations[cell_key(row)].append(row)
    for row in parents:
        selected_cells[cell_key(row)].append(row)
    sufficient, cells = [], []
    for key, selected in sorted(selected_cells.items()):
        population = populations[key]
        _validate_sampling(population, selected)
        weighted, weights, confidence = [], [], []
        fraction_deltas, ptm_deltas = [], []
        completed_weight = pair_weight = event_weight = 0.
        all_weight = sum(1 / r["inclusion_probability"] for r in selected)
        group_values = defaultdict(list)
        observed_qualified_groups = set()
        for row in selected:
            other = pairs[row["id"]]
            if Counter(row["sequence"]) != Counter(other["sequence"]):
                raise ValueError("Shuffled control changed sequence composition")
            original_score, shuffled_score = scores[row["id"]], scores[other["id"]]
            weight = 1 / row["inclusion_probability"]
            parent_ok = original_score["status"] == "ok"
            pair_ok = parent_ok and shuffled_score["status"] == "ok"
            delta = None
            if parent_ok:
                completed_weight += weight
                event_weight += weight * original_score["confidence_event"]
                confidence.append((original_score["mean_ca_plddt"], weight))
                if original_score["confidence_event"] and row.get("target_profile_hit"):
                    observed_qualified_groups.add(str(row["near_duplicate_group"]))
            if pair_ok:
                pair_weight += weight
                delta = original_score["mean_ca_plddt"] - shuffled_score["mean_ca_plddt"]
                weighted.append(delta)
                weights.append(weight)
                fraction_deltas.append(original_score["fraction_ca_plddt_ge70"] - shuffled_score["fraction_ca_plddt_ge70"])
                ptm_deltas.append(original_score["ptm"] - shuffled_score["ptm"])
                group_values[str(row["near_duplicate_group"])].append(delta)
            sufficient.append({k: row.get(k) for k in (
                "id", "sequence_sha256", "arm", "class_key", "condition", "role", "length",
                "stratum", "inclusion_probability", "near_duplicate_group", "target_profile_hit",
                "reference_identity", "reference_coverage", "reference_search_status")}
                | {"paired_id": other["id"], "status": original_score["status"],
                   "shuffle_status": shuffled_score["status"], "paired_delta": delta,
                   "mean_ca_plddt": original_score.get("mean_ca_plddt"),
                   "fraction_ca_plddt_ge70": original_score.get("fraction_ca_plddt_ge70"),
                   "ptm": original_score.get("ptm"),
                   "shuffle_mean_ca_plddt": shuffled_score.get("mean_ca_plddt"),
                   "shuffle_fraction_ca_plddt_ge70": shuffled_score.get("fraction_ca_plddt_ge70"),
                   "shuffle_ptm": shuffled_score.get("ptm"),
                   "confidence_event": original_score.get("confidence_event")})
        population_n = len(population)
        supported_n = sum(supported(r) for r in population)
        unknown_weight = all_weight - completed_weight + population_n - supported_n
        event_lower = event_weight / population_n
        event_upper = min(1., (event_weight + unknown_weight) / population_n)
        complete_pairs = np.isclose(pair_weight, all_weight, atol=1e-9, rtol=0)
        cells.append(dict(zip(KEYS, key)) | {
            "n_population": population_n, "n_supported": supported_n,
            "n_selected": len(selected), "selected_weight": all_weight,
            "complete_pair_weight": pair_weight, "complete_pairs": bool(complete_pairs),
            "paired_mean_ca_plddt_delta": weighted_mean(weighted, weights),
            "paired_fraction_delta": weighted_mean(fraction_deltas, weights),
            "paired_ptm_delta": weighted_mean(ptm_deltas, weights),
            "near_duplicate_balanced_delta": float(np.mean([np.mean(v) for v in group_values.values()])) if group_values else None,
            "mean_ca_plddt": weighted_mean([x[0] for x in confidence], [x[1] for x in confidence]),
            "confidence_event_all_attempt_lower_estimate": event_lower,
            "confidence_event_all_attempt_upper_estimate": event_upper,
            "confidence_event_support_bounds_are_exact_census": False,
            "joint_profile_confidence": _joint_bounds(population, selected, scores),
            "observed_distinct_joint_groups": len(observed_qualified_groups) if all(isinstance(r.get("target_profile_hit"), bool) for r in population) else None,
            "joint_group_count_is_lower_bound_not_extrapolated": True,
        })
    arms = {}
    for arm in sorted({r["arm"] for r in parents}):
        candidates = [r for r in cells if r["arm"] == arm and (
            r["role"] == "natural_reference" if phase == "pilot" else
            r["role"] == "generation" and r["condition"] == primary_condition)]
        if not candidates:
            continue
        complete = all(r["complete_pairs"] and r["paired_mean_ca_plddt_delta"] is not None for r in candidates)
        values = [r["paired_mean_ca_plddt_delta"] for r in candidates if r["paired_mean_ca_plddt_delta"] is not None]
        if uncertainty_unit == "sequence_cluster":
            records = [r for r in sufficient if r["arm"] == arm and r["role"] == "generation" and r["condition"] == primary_condition]
            summary = sequence_cluster_interval(records, resamples=resamples)
        else:
            summary = class_interval(values, resamples=resamples) if values else {"mean": None, "ci95": None, "ci97_5": None}
        if phase == "pilot":
            attained = bool(complete and summary["ci95"] is not None and summary["ci95"][0] > 0)
            interpretation = "calibrated" if attained else "uncalibrated"
        else:
            if uncertainty_unit == "sequence_cluster":
                controls = calibration.get("arms", {})
                attained = bool(controls and all(r.get("calibration_attained", False) for r in controls.values()))
                summary["calibration_scope"] = "shared_R232_natural_control_panel_not_native_class_calibration"
            else:
                attained = bool(calibration.get("arms", {}).get(arm, {}).get("calibration_attained", False))
            if not attained:
                interpretation = "uncalibrated_predictor"
            elif not complete:
                interpretation = "incomplete_paired_endpoint"
            elif summary["ci95" if uncertainty_unit == "sequence_cluster" else "ci97_5"] is not None and summary["ci95" if uncertainty_unit == "sequence_cluster" else "ci97_5"][0] > 0:
                interpretation = "positive_predictor_confidence_evidence_against_composition_shuffle"
            else:
                interpretation = "no_positive_composition_margin_at_declared_uncertainty"
        arms[arm] = summary | {"complete_pairs": complete,
                              "calibration_attained": attained,
                              "interpretation": interpretation,
                              "class_values": {r["class_key"]: r["paired_mean_ca_plddt_delta"] for r in candidates}}
    return {"schema_version": "generation_biology_analysis", "phase": phase, "primary_condition": primary_condition,
            "uncertainty_unit": uncertainty_unit,
            "terminal_predictions_complete": True,
            "prediction_status_counts": dict(Counter(x["status"] for x in scores.values())),
            "n_attempt_records": len(attempts), "n_prediction_records": len(subset),
            "natural_control_sequence_sha256": sorted({r["sequence_sha256"] for r in parents if r["role"] == "natural_reference"}),
            "profile_ledger": profile_ledger(attempts), "structural_cells": cells,
            "pair_sufficient_statistics": sufficient, "arms": arms,
            "limitations": ["Predicted confidence is not observed folding or function.",
                "All-attempt event bounds use sampled estimates, not an exact structural census.",
                "Uncertainty intervals do not create independent retraining or family-generalization evidence.",
                "Reference distance does not certify full-training disjointness.",
                "Negative results do not prove absence of learned biological information."]}


def write_report(report: dict[str, Any], output: str | Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "generation_biology_analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Generated-output computational biological evidence", "",
             f"Phase: {report['phase']}. Complete terminal structural records: {report['n_prediction_records']}.",
             "", "These are computational predictor outcomes; no corresponding experimental function is established.",
             "", "| Arm | Mean paired CA-pLDDT difference | 95% interval | 97.5% interval | Interpretation |",
             "| --- | --- | --- | --- | --- |"]
    for arm, result in report["arms"].items():
        lines.append(f"| {arm} | {result['mean']} | {result['ci95']} | {result['ci97_5']} | {result['interpretation']} |")
    lines.extend(["", "## Boundaries", ""] + [f"- {x}" for x in report["limitations"]])
    lines.extend(["", "Per-class populations, inclusion weights, missingness, profile confusion and pair-level sufficient statistics are retained in the JSON beside this report.", ""])
    (output / "generation_biology_report.md").write_text("\n".join(lines))
    for name in ("profile_ledger", "structural_cells", "pair_sufficient_statistics"):
        rows = report[name]
        if not rows:
            continue
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                                 for key, value in row.items()})


def write_figures(report: dict[str, Any], output: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arms = sorted(report["arms"])
    condition = report.get("primary_condition", "requested")
    if not arms:
        return
    fig, axes = plt.subplots(1, len(arms), figsize=(6 * len(arms), 4), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        values = report["arms"][arm]["class_values"]
        ordered = sorted(values.items(), key=lambda item: (item[1] is None, item[1] or 0))
        ax.axvline(0, color="black", lw=.8)
        ax.scatter([v for _, v in ordered], range(len(ordered)), s=20)
        ax.set_yticks(range(len(ordered)), [k for k, _ in ordered], fontsize=7)
        ax.set_title(arm)
        ax.set_xlabel("Mean CA-pLDDT: original − own composition shuffle")
    fig.suptitle("Natural-control calibration" if report["phase"] == "pilot" else "Declared native task: predictor contrasts")
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"figure_structural_contrasts.{extension}", dpi=220)
    plt.close(fig)
    if report["phase"] != "main":
        return
    fig, axes = plt.subplots(1, len(arms), figsize=(6 * len(arms), 4), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        rows = [r for r in report["profile_ledger"] if r["arm"] == arm and r["role"] == "generation"
                and r["condition"] == condition and r["primary_class"]]
        x = np.arange(len(rows))
        ax.bar(x - .18, [r["target_profile_rate"] if r["target_profile_rate"] is not None else np.nan for r in rows], width=.35, label="Target recognition / attempt")
        ax.bar(x + .18, [r["distinct_target_groups_per_attempt"] if r["distinct_target_groups_per_attempt"] is not None else np.nan for r in rows], width=.35, label="Distinct target groups / attempt")
        ax.set_xticks(x, [r["class_key"] for r in rows], rotation=90, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(arm)
        ax.legend(fontsize=7)
    fig.suptitle("All-attempt profile outcomes; distinct label systems are not ranked")
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"figure_profile_yields.{extension}", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(arms), figsize=(6 * len(arms), 4), squeeze=False)
    colors = {"not_searched": "gray", "no_reported_alignment": "#e69f00",
              "aligned_coverage_unavailable": "#0072b2", "aligned": "#009e73"}
    for ax, arm in zip(axes[0], arms):
        rows = [r for r in report["pair_sufficient_statistics"] if r["arm"] == arm
                and r["role"] == "generation" and r["condition"] == condition and r["status"] == "ok"]
        for state in sorted({str(r["reference_search_status"]) for r in rows}):
            part = [r for r in rows if str(r["reference_search_status"]) == state]
            ax.scatter([r["length"] for r in part], [r["mean_ca_plddt"] for r in part],
                       s=14, alpha=.6, label=state.replace("_", " "), color=colors.get(state))
        ax.set(xlabel="Full generated sequence length (residues)", ylabel="Mean CA-pLDDT", ylim=(0, 100), title=arm)
        ax.legend(fontsize=6)
    fig.suptitle("Random structural sample: reference-search support and confidence")
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"figure_length_reference_support.{extension}", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(arms), figsize=(7 * len(arms), 5), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        rows = [r for r in report["profile_ledger"] if r["arm"] == arm and r["role"] == "generation"
                and r["condition"] == condition and r["primary_class"]]
        labels = sorted({key for row in rows for key in row["confusion_counts_multilabel"]})
        if not rows or not labels:
            ax.set_visible(False)
            continue
        matrix = np.array([[r["confusion_counts_multilabel"].get(label, 0) / r["n_attempts"] for label in labels] for r in rows])
        display = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(rows)), [r["class_key"] for r in rows], fontsize=6)
        ax.set_title(arm)
        fig.colorbar(display, ax=ax, label="Fraction of all attempts")
    fig.suptitle("Profile assignments: multi-label counts need not sum to one")
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"figure_profile_confusion.{extension}", dpi=220)
    plt.close(fig)
