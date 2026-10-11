#!/usr/bin/env python3
"""The ladder table, and where likelihood and structure part company.

Joins the three independent products -- the frozen variants, each arm's own
likelihood of them, and ESMFold2's evaluation of them -- and reports, per arm and
per rung, the within-backbone rank correlation between likelihood and the
independent structural evaluation, with a cluster-bootstrap interval, a
panel-wide simultaneous band, the mean and spread of each quantity, and the rung
at which the correlation stops being resolvable or changes sign.

Why the correlation is within-backbone. Both likelihood and predicted confidence
depend strongly on chain length, and the backbones span 150 to 300 residues, so a
pooled correlation over every variant of a rung would be dominated by
between-backbone variation that has nothing to do with modification extent.
Ranking inside each backbone and centring there removes every between-backbone
difference, length first of all. The resampling unit is the backbone for the same
reason: two draws on one backbone share a parent, a prompt and a window.

The sign convention is fixed once here. The likelihood axis is the *log*
likelihood per scored symbol, so a positive correlation means "the sequences this
arm finds more likely are the ones the structure predictor finds more similar to
the parent fold", which is the direction the hypothesis is about.

Three readings are reported side by side and none replaces another: the raw
within-backbone correlation, the same correlation after linearly removing the
within-backbone rank of composition distance from the parent, and the
model-free composition-matched extent reference. A correlation that survives
only in the raw reading is a composition result, not a structural one.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.core.statistics import mean_interval  # noqa: E402
from src.capability.ladder import design, divergence  # noqa: E402

COMPLETION = "ladder_divergence.json"
REPORT = "ladder_divergence.md"
SCHEMA_VERSION = "ladder_divergence_v1"

#: Structural readouts the ladder correlates likelihood against. The first is the
#: primary; the others are reported beside it and are not substitutes for it.
READOUTS: tuple[str, ...] = (
    design.PRIMARY_STRUCTURE_READOUT,
    design.LOCAL_STRUCTURE_READOUT,
    "mean_ca_plddt",
)


def sanitise(value: Any, *, path: str = "", replaced: list[str] | None = None) -> Any:
    """Replace every non-finite float with ``null`` and name where it happened.

    A statistic can legitimately be undefined -- a readout that is constant
    inside every backbone carries no within-backbone rank information, and the
    correlation against it is not a number. Writing ``NaN`` into JSON is not an
    option and quietly coercing it to zero would turn "not computable" into "no
    effect", which is the one reading that must never be produced by accident.
    So it becomes ``null`` and its field is listed in ``non_finite_fields``.
    """

    if replaced is None:
        replaced = []
    if isinstance(value, float):
        if not np.isfinite(value):
            replaced.append(path)
            return None
        return value
    if isinstance(value, dict):
        return {key: sanitise(item, path=f"{path}.{key}", replaced=replaced) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitise(item, path=f"{path}[{index}]", replaced=replaced) for index, item in enumerate(value)]
    if isinstance(value, np.generic):
        return sanitise(value.item(), path=path, replaced=replaced)
    return value


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return {"n": 0}
    record: dict[str, Any] = {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else None,
        "min": float(array.min()),
        "max": float(array.max()),
        "quartiles": [float(value) for value in np.percentile(array, [25, 50, 75])],
    }
    if array.size >= 2:
        record["mean_interval"] = mean_interval(array.tolist())["interval"]
    return record


def build_table(
    variants: list[dict[str, Any]],
    likelihood: list[dict[str, Any]],
    structure: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    """Every joined observation, keyed by (scoring arm, product arm, condition, rung)."""

    by_digest: dict[tuple[str, str], dict[str, Any]] = {
        (str(row["scored_by"]), str(row["sequence_sha256"])): row for row in likelihood
    }
    by_id = {str(row["id"]): row for row in structure}
    scoring_arms = sorted({str(row["scored_by"]) for row in likelihood})
    cells: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for variant in variants:
        if variant["status"] != "filled" or not variant["sequence"]:
            continue
        folded = by_id.get(str(variant["id"]))
        if folded is None or folded.get("fold_status") != "ok":
            continue
        parent_digest = str(variant["parent_sequence_sha256"])
        for scorer in scoring_arms:
            scored = by_digest.get((scorer, str(variant["sequence_sha256"])))
            parent_scored = by_digest.get((scorer, parent_digest))
            if scored is None:
                continue
            observation = {
                "backbone_id": str(variant["backbone_id"]),
                "draw": int(variant["draw"]),
                "length": int(folded["length"]),
                "log_likelihood_per_symbol": -float(scored["mean_nll_per_token_nats"]),
                "delta_log_likelihood_per_symbol": (
                    float(parent_scored["mean_nll_per_token_nats"])
                    - float(scored["mean_nll_per_token_nats"])
                    if parent_scored is not None
                    else float("nan")
                ),
                "composition_distance_to_parent": float(
                    variant["composition_distance_to_parent"]
                    if variant.get("composition_distance_to_parent") is not None else float("nan")
                ),
                "realised_substitutions": variant.get("realised_substitutions"),
                "longest_homopolymer": variant.get("longest_homopolymer"),
                "likelihood_kind": str(scored.get("likelihood_kind", "")),
            }
            for readout in READOUTS:
                observation[readout] = float(
                    folded[readout] if folded.get(readout) is not None else float("nan")
                )
            cells[
                (scorer, str(variant["arm"]), str(variant["condition"]), str(variant["rung"]))
            ].append(observation)
    return cells


def _statistic(readout: str, adjusted: bool):
    def inner(values, strata):
        if adjusted:
            return cast(float, divergence.partial_within_stratum_rank_correlation(
                values["likelihood"], values[readout], values["composition"], strata
            ))
        return divergence.within_stratum_rank_correlation(
            values["likelihood"], values[readout], strata
        )

    return inner


def run(args: argparse.Namespace) -> dict[str, Any]:
    design.require_fresh_out(args.out, COMPLETION)
    backbones = design.read_jsonl(args.backbones)
    units = [str(row["backbone_id"]) for row in backbones]
    variants = [row for path in args.variants for row in design.read_jsonl(path)]
    likelihood = [row for path in args.likelihood for row in design.read_jsonl(path)]
    structure = [row for path in args.structure for row in design.read_jsonl(path)]
    scoring_arms = {str(row["scored_by"]) for row in likelihood}
    expected_ladders = {
        (scorer, str(row["arm"]), str(row["condition"]))
        for row in variants if row["rung"] != design.FULL_GENERATION
        for scorer in scoring_arms
    }
    cells = build_table(variants, likelihood, structure)
    if not cells:
        raise SystemExit(
            "nothing joined: no variant carries both a likelihood and a completed fold. "
            "Refusing to report an empty ladder as a null result"
        )
    resamples = divergence.stratum_resamples(units, seed=args.seed, draws=args.draws)

    results: dict[str, Any] = {}
    bootstrap_cells: dict[str, dict[str, Any]] = {}
    for key, observations in sorted(cells.items()):
        scorer, product_arm, condition, rung = key
        name = f"{scorer}|{product_arm}|{condition}|{rung}"
        strata = [row["backbone_id"] for row in observations]
        vectors = {
            "likelihood": [row["log_likelihood_per_symbol"] for row in observations],
            "composition": [
                row["composition_distance_to_parent"]
                for row in observations
            ],
        }
        entry: dict[str, Any] = {
            "scoring_arm": scorer,
            "product_arm": product_arm,
            "model_class": design.arm(product_arm).model_class,
            "scoring_model_class": design.arm(scorer).model_class,
            "condition": condition,
            "rung": rung,
            "nominal_extent": design.rung_extent(rung),
            "n_observations": len(observations),
            "n_backbones": len(set(strata)),
            "likelihood_kind": observations[0]["likelihood_kind"],
            "distributions": {
                "log_likelihood_per_symbol": _summary(vectors["likelihood"]),
                "delta_log_likelihood_per_symbol_vs_parent": _summary(
                    [row["delta_log_likelihood_per_symbol"] for row in observations]
                ),
                "length": _summary([float(row["length"]) for row in observations]),
                "composition_distance_to_parent": _summary(
                    [row["composition_distance_to_parent"] for row in observations]
                ),
                "longest_homopolymer": _summary(
                    [
                        float(row["longest_homopolymer"])
                        for row in observations
                        if row["longest_homopolymer"] is not None
                    ]
                ),
                "realised_substitutions": _summary(
                    [
                        float(row["realised_substitutions"])
                        for row in observations
                        if row["realised_substitutions"] is not None
                    ]
                ),
                **{readout: _summary([row[readout] for row in observations]) for readout in READOUTS},
            },
            "correlations": {},
        }
        payload = dict(vectors)
        usable: list[str] = []
        for readout in READOUTS:
            column = [row[readout] for row in observations]
            if not np.isfinite(column).all():
                continue
            payload[readout] = column
            usable.append(readout)
        if usable:
            # One pass over the resamples for every readout and both adjustments
            # of this cell: the resample's row index is the expensive part and it
            # does not depend on which column is being correlated.
            functions = {f"raw|{readout}": _statistic(readout, False) for readout in usable}
            composition_available = bool(np.isfinite(payload["composition"]).all())
            if composition_available:
                functions |= {f"adjusted|{readout}": _statistic(readout, True) for readout in usable}
            diagnostics = {}
            for readout in usable:
                diagnostics[f"raw|{readout}"] = divergence.partial_within_stratum_rank_correlation(
                    payload["likelihood"], payload[readout], [0.0] * len(strata), strata,
                    return_diagnostics=True,
                )
                if composition_available:
                    diagnostics[f"adjusted|{readout}"] = divergence.partial_within_stratum_rank_correlation(
                        payload["likelihood"], payload[readout], payload["composition"], strata,
                        return_diagnostics=True,
                    )
            bootstrapped = divergence.cluster_bootstrap_many(
                payload, strata, functions, units=units, resamples=resamples,
                informative_units={key: value["informative_backbones"] for key, value in diagnostics.items()},
            )
            for readout in usable:
                raw = bootstrapped[f"raw|{readout}"]
                adjusted = bootstrapped.get(f"adjusted|{readout}", {
                    "point": None, "interval": None, "inferential_available": False,
                    "point_reason": "missing or non-finite composition covariate",
                    "interval_reason": "missing or non-finite composition covariate",
                })
                if composition_available:
                    adjusted["diagnostics"] = diagnostics[f"adjusted|{readout}"]
                    adjusted["point_reason"] = adjusted["diagnostics"]["reason"]
                raw["diagnostics"] = diagnostics[f"raw|{readout}"]
                raw["point_reason"] = raw["diagnostics"]["reason"]
                entry["correlations"][readout] = {
                    "raw": {key2: value for key2, value in raw.items() if key2 != "draws"},
                    "composition_adjusted": {
                        key2: value for key2, value in adjusted.items() if key2 != "draws"
                    },
                    "per_backbone_spearman": divergence.per_stratum_spearman(
                        payload["likelihood"], payload[readout], strata
                    ),
                }
                if readout == design.PRIMARY_STRUCTURE_READOUT and rung != design.FULL_GENERATION:
                    bootstrap_cells[name] = raw
        results[name] = entry

    family = dict(bootstrap_cells)
    for scorer, product_arm, condition in sorted(expected_ladders):
        for extent in design.WINDOW_EXTENTS:
            family.setdefault(f"{scorer}|{product_arm}|{condition}|k{extent}", {
                "point": None, "bootstrap_sd": None, "inferential_available": False,
            })
    band = divergence.simultaneous_band(family)

    ladders: dict[str, Any] = {}
    grouped: dict[tuple[str, str, str], list[str]] = {key: [] for key in expected_ladders}
    for name, entry in results.items():
        if entry["rung"] == design.FULL_GENERATION:
            continue
        grouped[(entry["scoring_arm"], entry["product_arm"], entry["condition"])].append(name)
    for (scorer, product_arm, condition), names in sorted(grouped.items()):
        ordered = sorted(names, key=lambda item: results[item]["nominal_extent"])
        rungs = [results[name]["rung"] for name in ordered]
        declared_rungs = [f"k{extent}" for extent in design.WINDOW_EXTENTS]
        missing_rungs = [rung for rung in declared_rungs if rung not in rungs]
        primary = [
            results[name]["correlations"].get(design.PRIMARY_STRUCTURE_READOUT) for name in ordered
        ]
        contrasts = divergence.adjacent_rung_contrasts(
            {results[name]["rung"]: bootstrap_cells[name] for name in ordered if name in bootstrap_cells},
            rungs=declared_rungs,
        )
        if missing_rungs or any(item is None for item in primary):
            ladders[f"{scorer}|{product_arm}|{condition}"] = {
                "status": "incomplete",
                "rungs_present": rungs,
                "reason": "missing declared window rung or primary structural readout",
                "declared_rungs": declared_rungs,
                "missing_rungs": missing_rungs,
                "computational_status": "computed",
                "ladder_complete": False,
                "inferential_available": False,
                "adjacent_rung_contrasts": contrasts,
            }
            continue
        ladder = [
            {
                "rung": rung,
                "point": item["raw"]["point"],
                "interval": item["raw"]["interval"],
            }
            for rung, item in zip(rungs, primary)
        ]
        simultaneous = {
            results[name]["rung"]: band["intervals"][name]
            for name in ordered
            if name in band.get("intervals", {})
        }
        extents = [results[name]["nominal_extent"] for name in ordered]
        matrix = np.vstack(
            [np.asarray(bootstrap_cells[name]["draws"]) for name in ordered if name in bootstrap_cells]
        ).T if all(name in bootstrap_cells for name in ordered) else None
        entry: dict[str, Any] = {
            "status": "complete",
            "computational_status": "computed",
            "ladder_complete": True,
            "declared_rungs": declared_rungs,
            "missing_rungs": [],
            "inferential_available": all(item["raw"].get("inferential_available", False) for item in primary),
            "adjacent_rung_contrasts": contrasts,
            "ladder": ladder,
            "divergence": divergence.divergence_rung(ladder, simultaneous=simultaneous),
            "simultaneous_intervals": simultaneous,
        }
        if matrix is not None and matrix.shape[1] >= 3:
            entry["continuous_crossing"] = divergence.continuous_crossing(
                extents, matrix.tolist(), point_estimates=[item["raw"]["point"] for item in primary],
                primary_inference_available=entry["inferential_available"],
            )
        resolved = [
            index
            for index, item in enumerate(primary)
            if divergence._excludes_zero(item["raw"]["interval"]) is True
        ]
        if not resolved:
            bottom = primary[0]["raw"]
            if bottom.get("inferential_available") and bottom.get("bootstrap_sd") is not None and bottom["bootstrap_sd"] > 0:
                entry["required_units_to_resolve_bottom_rung"] = divergence.required_units(
                    bootstrap_sd=bottom["bootstrap_sd"],
                    n_units=bottom["n_informative_units"],
                    target_effect=bottom["point"] if bottom["point"] != 0.0 else 0.2,
                )
        ladders[f"{scorer}|{product_arm}|{condition}"] = entry

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "declaration": design.declaration(),
        "likelihood_axis": (
            "log likelihood per scored symbol, so a positive correlation means the "
            "sequences the arm finds more likely are the ones the structure predictor "
            "finds more similar to the parent fold"
        ),
        "estimand": (
            "within-backbone rank correlation; ranks are taken inside each backbone "
            "and centred there, and the backbone is the cluster-bootstrap unit"
        ),
        "n_backbones": len(units),
        "draws": int(args.draws),
        "seed": int(args.seed),
        "cells": results,
        "ladders": ladders,
        "simultaneous_band": {
            key: value for key, value in band.items() if key != "intervals"
        }
        | {"intervals": band.get("intervals", {})},
        "sources": {
            "backbones": {"path": str(args.backbones), "sha256": sha256_file(args.backbones)},
            "variants": {str(path): sha256_file(path) for path in args.variants},
            "likelihood": {str(path): sha256_file(path) for path in args.likelihood},
            "structure": {str(path): sha256_file(path) for path in args.structure},
        },
        "ceiling": dict(design.CEILING),
    }
    replaced: list[str] = []
    payload = sanitise(raw_payload, replaced=replaced)
    payload["non_finite_fields"] = sorted(replaced)
    write_json(out / COMPLETION, payload)
    (out / REPORT).write_text(render_report(payload), encoding="utf-8")
    return payload


def _signed(value: Any) -> str:
    return "undefined" if value is None else f"{float(value):+.4f}"


def _plain(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _interval(interval: Any) -> str:
    if not interval or any(item is None for item in interval):
        return "undefined"
    return f"[{float(interval[0]):+.4f}, {float(interval[1]):+.4f}]"


def render_report(payload: dict[str, Any]) -> str:
    """A reader's view of the ladder: one table per arm, intervals and sample sizes."""

    lines = [
        "# Modification-extent ladder",
        "",
        f"Campaign `{payload['declaration']['campaign']}`, "
        f"{payload['n_backbones']} backbones, "
        f"{payload['declaration']['sampling']['draws_per_cell']} draws per cell, "
        f"{payload['draws']} cluster-bootstrap resamples over backbones.",
        "",
        payload["estimand"] + ".",
        "",
        payload["likelihood_axis"] + ".",
        "",
    ]
    for name, ladder in sorted(payload["ladders"].items()):
        scorer, product_arm, condition = name.split("|")
        lines.append(f"## {product_arm} products, scored by {scorer} ({condition})")
        lines.append("")
        if ladder.get("status") != "complete":
            lines.append(f"Incomplete: {ladder.get('reason')}")
            lines.append("")
            continue
        lines.append(
            "| rung | k | n | backbones | rho(likelihood, TM to parent) | 95% CI | simultaneous CI | mean TM | mean length |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in ladder["ladder"]:
            cell = payload["cells"][f"{name}|{row['rung']}"]
            primary = cell["correlations"][payload["declaration"]["structure_readouts"]["primary"]]
            simultaneous = ladder["simultaneous_intervals"].get(row["rung"])
            tm = cell["distributions"][payload["declaration"]["structure_readouts"]["primary"]]
            lines.append(
                "| {rung} | {k} | {n} | {b} | {rho} | {ci} | {sim} | {tm} | {length} |".format(
                    rung=row["rung"],
                    k=cell["nominal_extent"],
                    n=primary["raw"]["n_observations"],
                    b=primary["raw"]["n_units"],
                    rho=_signed(row["point"]),
                    ci=_interval(row["interval"]),
                    sim=_interval(simultaneous),
                    tm=_plain(tm.get("mean")),
                    length=_plain(cell["distributions"]["length"].get("mean")),
                )
            )
        lines.append("")
        divergence_record = ladder["divergence"]
        lines.append(
            f"Descriptive first-loss/sign heuristics (not evidence of decline or mechanism). "
            f"Marginal inference available: {divergence_record['inferential_available']}; "
            f"simultaneous inference available: {divergence_record['simultaneous_available']}. "
            f"Loses significance at: {divergence_record['loses_significance'] or 'no identified rung'}; "
            f"simultaneously at: {divergence_record.get('loses_significance_simultaneous') or 'no rung'}; "
            f"changes sign at: {divergence_record['changes_sign'] or 'no rung'}; "
            f"bottom rung unresolved: {divergence_record['undetermined_at_bottom']}."
        )
        lines.append("Paired adjacent-rung raw-primary contrasts (later minus earlier; marginal CIs):")
        for contrast in ladder["adjacent_rung_contrasts"]:
            lines.append(
                f"- {contrast['earlier_rung']} to {contrast['later_rung']}: "
                f"{_signed(contrast['point'])}, {_interval(contrast['interval'])}; "
                f"supports decline: {contrast['supports_decline']}; "
                f"reason: {contrast['reason'] or 'available'}."
            )
        crossing = ladder.get("continuous_crossing")
        if crossing:
            lines.append(
                f"Continuous crossing at k = "
                f"{crossing['crossing_extent'] if crossing['crossing_extent'] is not None else 'outside the measured range'}"
                + (
                    f", 95% interval conditional on an inside-range crossing {crossing['interval']}"
                    if crossing.get("interval")
                    else ""
                )
                + f"; {crossing['fraction_without_crossing']:.3f} of draws have no crossing inside "
                f"{crossing['measured_range']}."
            )
        if "required_units_to_resolve_bottom_rung" in ladder:
            needed = ladder["required_units_to_resolve_bottom_rung"]
            lines.append(
                f"No rung resolves. Resolving an effect of {needed['target_effect']:+.4f} at "
                f"{needed['confidence']:.2f} confidence and {needed['power']:.1f} power would need "
                f"about {needed['required_n_units']} backbones under the stated scaling assumption."
            )
        lines.append("")
    lines.append("## Ceiling")
    lines.append("")
    for key, value in sorted(payload["ceiling"].items()):
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="accepted and unused; this stage is CPU-only")
    parser.add_argument("--backbones", type=Path, required=True)
    parser.add_argument("--variants", type=Path, nargs="+", required=True)
    parser.add_argument("--likelihood", type=Path, nargs="+", required=True)
    parser.add_argument("--structure", type=Path, nargs="+", required=True)
    parser.add_argument("--draws", type=int, default=divergence.DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=design.CAMPAIGN_SEED)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"n_cells": len(payload["cells"]), "n_ladders": len(payload["ladders"])}))


if __name__ == "__main__":
    main()
