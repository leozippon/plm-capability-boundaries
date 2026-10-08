#!/usr/bin/env python3
"""Assemble E14's matched phenotype contrasts and E17's selection curves.

Reads only frozen artefacts: the cohort and its declarations, the ESMFold2
shards' indexes, the Pfam oracle's annotations, and the recomputed likelihoods.
It loads no model, predicts no structure and recognises no family; it accepts
``--device`` because the campaign queue injects it.

E14 is reported per arm and per termination stratum, never pooled over strata,
because a budget-censored product is a fragment and pooling it with a finished
one makes the headline number a measure of truncation. Each stratum's generated
sequences are compared with *their own* length-matched whole natural records, and
the difference between the two strata's gaps is reported as its own quantity.

E17 is reported as a yield curve over selection fraction at equal selected-set
size, with the repertoire of each selected set beside its yield, and with the
cheap composition selector as the control the model-likelihood selector has to
beat rather than as an afterthought.

Stability appears in the output exactly once, as unavailable, with its reasons.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import write_json  # noqa: E402
from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS, mean_interval  # noqa: E402
from src.capability.evaluation import generated_phenotype as gp  # noqa: E402

COMPLETION = "generation_evaluation.json"
SCHEMA_VERSION = "d1_generation_evaluation_v1"

#: The evaluators, every one oriented so that a larger value is better. The
#: structural ones are ESMFold2's own confidences and the operational
#: confidence event it declares; predicted aligned error is negated for that
#: orientation, because a selection curve in which one evaluator runs the other
#: way is exactly the trap this experiment is supposed to avoid. The Pfam ones
#: are the oracle's family calls. None of them is a stability quantity.
EVALUATORS: tuple[str, ...] = (
    "mean_ca_plddt",
    "ptm",
    "confident_fold",
    "neg_mean_pae_angstrom",
    "complete_domain",
    "any_family",
)

EVALUATOR_ORIENTATION = (
    "every evaluator is oriented so that a larger value is better, including "
    "neg_mean_pae_angstrom, which is the negated mean predicted aligned error in angstrom"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"{path} carries no record")
    return rows


def structure_values(directories: list[Path]) -> dict[str, dict[str, Any]]:
    """Per cohort identifier, the selected diffusion sample's confidences.

    Shards are sequence-hash disjoint by construction, so a repeated identifier
    across shards means two shard trees describing the same row and is refused:
    silently keeping one would make the denominator depend on directory order.
    """

    values: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for directory in directories:
        indexes = sorted(directory.glob("index-*-of-*.jsonl"))
        if not indexes:
            raise SystemExit(f"{directory} holds no ESMFold2 index; nothing was folded")
        summary = directory / "structure_evidence.json"
        if not summary.is_file():
            raise SystemExit(
                f"{summary} is absent: the folding cell did not reach its completion "
                "record, so its index is a partial tree and is not read"
            )
        receipts.append(
            {"directory": str(directory), "summary": json.loads(summary.read_text(encoding="utf-8"))}
        )
        for index in indexes:
            for row in read_jsonl(index):
                identifier = str(row["id"])
                if identifier in values:
                    raise SystemExit(f"identifier {identifier!r} appears in two ESMFold2 shards")
                structure = row.get("structure") or {}
                values[identifier] = {
                    "status": structure.get("status"),
                    "reason": structure.get("reason"),
                    "mean_ca_plddt": structure.get("mean_ca_plddt"),
                    "fraction_ca_plddt_ge70": structure.get("fraction_ca_plddt_ge70"),
                    "ptm": structure.get("ptm"),
                    "mean_pae_angstrom": structure.get("mean_pae_angstrom"),
                    # The instrument decides its own confidence event; this
                    # module reads it rather than re-deriving the threshold.
                    "predicted_confidence_event": structure.get("predicted_confidence_event"),
                    "object_directory": structure.get("object_directory"),
                }
    if not values:
        raise SystemExit("the ESMFold2 shards carried no row")
    return {"values": values, "receipts": receipts}


def evaluator_vector(
    identifiers: list[str],
    structure: dict[str, dict[str, Any]],
    recognition: dict[str, dict[str, Any]],
    name: str,
) -> tuple[np.ndarray, list[str]]:
    """One evaluator's values over ``identifiers``, with the dropped rows named.

    A row the structure instrument could not evaluate is dropped from the
    structural evaluators and reported, never defaulted to zero: a failed
    prediction is not a low-confidence prediction. The Pfam evaluators keep every
    row, because "no profile recognised it" is a measurement, not a failure.
    """

    values: list[float] = []
    kept: list[str] = []
    dropped: list[str] = []
    for identifier in identifiers:
        if name in ("complete_domain", "any_family"):
            block = recognition.get(identifier)
            if block is None:
                dropped.append(identifier)
                continue
            key = "complete_domain" if name == "complete_domain" else "any_profile_hit"
            values.append(1.0 if block.get(key) else 0.0)
            kept.append(identifier)
            continue
        block = structure.get(identifier)
        if block is None or block.get("status") != "ok":
            dropped.append(identifier)
            continue
        if name == "confident_fold":
            event = block.get("predicted_confidence_event")
            if event is None:
                dropped.append(identifier)
                continue
            values.append(1.0 if event else 0.0)
        else:
            source = "mean_pae_angstrom" if name == "neg_mean_pae_angstrom" else name
            value = block.get(source)
            if value is None or not np.isfinite(float(value)):
                dropped.append(identifier)
                continue
            values.append(-float(value) if name == "neg_mean_pae_angstrom" else float(value))
        kept.append(identifier)
    if len(values) != len(kept):
        raise AssertionError("evaluator bookkeeping lost a row")
    return np.asarray(values, dtype=np.float64), dropped


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    """Spearman correlation, or ``None`` when one side carries no ranking.

    A rank correlation against a constant vector is undefined, not zero. It
    happens for real here: a pool whose every member carries a complete domain
    makes that evaluator constant, and ``scipy`` returns NaN, which this
    package's serialiser refuses. Recording ``None`` says "undefined on this
    support" instead of inventing a correlation of zero.
    """

    if left.size < 2 or right.size < 2:
        return None
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    value = float(stats.spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def paired_vectors(
    pairs: list[dict[str, Any]],
    structure: dict[str, dict[str, Any]],
    recognition: dict[str, dict[str, Any]],
    evaluator: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, int]]:
    """Generated and natural evaluator values over the pairs both sides survive.

    A pair is kept only if the evaluator is defined on *both* members. Dropping
    one side alone would break the pairing that makes the contrast a matched
    one, and keeping it with a substituted value would invent a measurement.
    """

    left: list[float] = []
    right: list[float] = []
    usable: list[dict[str, Any]] = []
    dropped = {"generated": 0, "natural": 0}
    for pair in pairs:
        generated, missing_g = evaluator_vector(
            [pair["generated_id"]], structure, recognition, evaluator
        )
        natural, missing_n = evaluator_vector(
            [pair["natural_id"]], structure, recognition, evaluator
        )
        dropped["generated"] += len(missing_g)
        dropped["natural"] += len(missing_n)
        if missing_g or missing_n:
            continue
        left.append(float(generated[0]))
        right.append(float(natural[0]))
        usable.append(pair)
    return (
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        usable,
        dropped,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    gp.require_fresh_out(args.out, COMPLETION)
    cohort = {row["id"]: row for row in read_jsonl(args.cohort_dir / "evaluation_cohort.jsonl")}
    e14 = json.loads((args.cohort_dir / "e14_declaration.json").read_text(encoding="utf-8"))
    e17 = json.loads((args.cohort_dir / "e17_declaration.json").read_text(encoding="utf-8"))
    census = json.loads((args.cohort_dir / "termination_census.json").read_text(encoding="utf-8"))
    background = json.loads((args.cohort_dir / "residue_background.json").read_text(encoding="utf-8"))

    folded = structure_values(args.structure)
    structure = folded["values"]
    recognition = {
        str(row["id"]): row for row in read_jsonl(args.recognition / "family_recognition.jsonl")
    }
    control = json.loads((args.recognition / "oracle_control.json").read_text(encoding="utf-8"))
    if not control.get("passed"):
        raise SystemExit(
            "the family-recognition oracle's control did not pass, so its recognition "
            "set is not evidence and no function-related number is computed from it"
        )
    likelihood: dict[str, dict[str, Any]] = {}
    likelihood_receipts: list[dict[str, Any]] = []
    for directory in args.likelihood:
        receipt = json.loads((directory / "generated_likelihood.json").read_text(encoding="utf-8"))
        likelihood_receipts.append(receipt)
        for row in read_jsonl(directory / "generated_likelihood.jsonl"):
            identifier = str(row["id"])
            if identifier in likelihood:
                raise SystemExit(f"identifier {identifier!r} was scored twice for likelihood")
            likelihood[identifier] = row

    # ------------------------------------------------------------------- E14
    pairs_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in e14["pairs"]:
        pairs_by_arm[str(pair["arm"])].append(pair)
    e14_results: dict[str, Any] = {}
    for arm, pairs in sorted(pairs_by_arm.items()):
        per_stratum: dict[str, Any] = {}
        for stratum in sorted(gp.STRATA):
            members = [pair for pair in pairs if pair["stratum"] == stratum]
            if not members:
                per_stratum[stratum] = {"status": "no_support"}
                continue
            block: dict[str, Any] = {
                "status": "measured",
                "n_pairs_declared": len(members),
                "length": {
                    "generated_mean": float(np.mean([p["generated_length"] for p in members])),
                    "natural_mean": float(np.mean([p["natural_length"] for p in members])),
                    "mean_absolute_difference": float(
                        np.mean([abs(p["natural_length"] - p["generated_length"]) for p in members])
                    ),
                },
                "contrasts": {},
            }
            for evaluator in EVALUATORS:
                left, right, usable, dropped = paired_vectors(
                    members, structure, recognition, evaluator
                )
                if len(usable) < 2:
                    block["contrasts"][evaluator] = {"status": "no_usable_pair", "dropped": dropped}
                    continue
                contrast = gp.matched_contrast(
                    left, right, [p["pair_id"] for p in usable], seed=args.seed, n_bootstrap=args.draws
                )
                contrast["dropped"] = dropped
                contrast["generated_standard_error"] = mean_interval(left)["standard_error"]
                block["contrasts"][evaluator] = contrast
            per_stratum[stratum] = block

        # The difference of the two strata's gaps: how much is truncation.
        gap_difference: dict[str, Any] = {}
        for evaluator in EVALUATORS:
            left, right, usable, _ = paired_vectors(pairs, structure, recognition, evaluator)
            if len(usable) < 2:
                gap_difference[evaluator] = {"resolved": False, "reason": "no usable pair"}
                continue
            gap_difference[evaluator] = gp.stratum_gap_contrast(
                left,
                right,
                [pair["stratum"] for pair in usable],
                [pair["pair_id"] for pair in usable],
                seed=args.seed,
                n_bootstrap=args.draws,
            )
        e14_results[arm] = {"strata": per_stratum, "truncation_share": gap_difference}

    # ------------------------------------------------------------------- E17
    e17_results: dict[str, Any] = {}
    for arm, declaration in sorted(e17["pools"].items()):
        unit = declaration["independence_unit"]
        members = [
            row
            for row in cohort.values()
            if row.get("arm") == arm and "pool" in row.get("roles", [])
        ]
        members.sort(key=lambda row: str(row["id"]))
        missing = [row["id"] for row in members if row["id"] not in likelihood]
        if missing:
            raise SystemExit(
                f"{len(missing)} pool members of {arm!r} carry no recomputed likelihood "
                f"(first {missing[:3]}); the selector is incomplete and no curve is reported"
            )
        selectors = {
            "likelihood": np.asarray(
                [likelihood[row["id"]]["mean_nll_per_token_nats"] for row in members],
                dtype=np.float64,
            ),
            "composition": np.asarray(
                [
                    gp.composition_cross_entropy(row["sequence"], background["background"])
                    for row in members
                ],
                dtype=np.float64,
            ),
        }
        selectors["combined"] = gp.rank_average(selectors["likelihood"], selectors["composition"])
        selectors["random"] = gp.random_key(len(members), seed=args.seed + 991)
        correlations = {
            "likelihood_vs_composition_spearman": spearman(
                selectors["likelihood"], selectors["composition"]
            )
        }

        per_evaluator: dict[str, Any] = {}
        for evaluator in EVALUATORS:
            values, dropped = evaluator_vector(
                [row["id"] for row in members], structure, recognition, evaluator
            )
            missing = set(dropped)
            kept = [row for row in members if row["id"] not in missing]
            if len(kept) < MINIMUM_BOOTSTRAP_UNITS:
                per_evaluator[evaluator] = {
                    "status": "no_usable_pool",
                    "n_dropped": len(dropped),
                }
                continue
            mask = np.asarray([row["id"] not in missing for row in members], dtype=bool)
            units = [str(row.get(unit) if unit != "id" else row["id"]) for row in kept]
            families = [recognition.get(row["id"], {}).get("pfam_families", []) for row in kept]
            curves: dict[str, Any] = {}
            for name, score in sorted(selectors.items()):
                scored = score[mask]
                points: list[dict[str, Any]] = []
                for fraction in gp.SELECTION_FRACTIONS:
                    contrast = gp.selection_contrast(
                        values, scored, units, fraction=fraction, seed=args.seed, n_bootstrap=args.draws
                    )
                    chosen = gp.selection_indices(scored, fraction=fraction)
                    contrast["repertoire"] = gp.family_repertoire([families[i] for i in chosen])
                    if len(chosen) >= 2:
                        contrast["repertoire"]["mean_pairwise_kmer_distance"] = (
                            gp.mean_pairwise_kmer_distance([kept[i]["sequence"] for i in chosen])
                        )
                    points.append(contrast)
                curves[name] = points
            per_evaluator[evaluator] = {
                "status": "measured",
                "n_pool": int(mask.sum()),
                "n_dropped": len(dropped),
                "pool_mean": float(values.mean()),
                "random_baseline": [
                    gp.random_baseline(values, fraction=fraction, seed=args.seed + 991)
                    for fraction in gp.SELECTION_FRACTIONS
                ],
                "curves": curves,
                "selector_evaluator_spearman": {
                    name: spearman(score[mask], values)
                    for name, score in sorted(selectors.items())
                },
            }
        e17_results[arm] = {
            "declaration": declaration,
            "n_members": len(members),
            "selector_correlations": correlations,
            "evaluators": per_evaluator,
        }

    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": {
            "cohort_dir": str(args.cohort_dir),
            "structure": [str(path) for path in args.structure],
            "recognition": str(args.recognition),
            "likelihood": [str(path) for path in args.likelihood],
        },
        "structure_receipts": folded["receipts"],
        "likelihood_receipts": likelihood_receipts,
        "oracle_control": {key: control[key] for key in ("passed", "n_positive", "n_negative")},
        "termination_census": census,
        "e14": {
            "arms": e14_results,
            "match_quality": e14["match_quality"],
            "strata": e14["strata"],
            "per_arm_stratum_support": e14["per_arm_stratum"],
            "stability_component": gp.STABILITY_UNAVAILABLE,
            "ceiling": gp.CEILING,
        },
        "e17": {
            "arms": e17_results,
            "pool_digest": e17["pool_digest"],
            "selection_fractions": list(gp.SELECTION_FRACTIONS),
            "independence": e17["independence"],
            "validity_conditions": e17["validity_conditions"],
        },
        "seed": int(args.seed),
        "n_bootstrap": int(args.draws),
        "evaluator_orientation": EVALUATOR_ORIENTATION,
        "reading_guide": [
            "no number here is experimental. Structure is a prediction, function is "
            "sequence homology, and stability is absent",
            "E14 contrasts are read per arm and per stratum. Two intervals that do not "
            "overlap are not a test of their difference; the truncation_share block is "
            "the quantity for that comparison",
            "an E17 method that beats random at one fraction and not across the curve "
            "is not a method, and a yield gain accompanied by a repertoire collapse is "
            "not a gain",
        ],
    }
    write_json(args.out / COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True, help="the frozen cohort directory")
    parser.add_argument("--structure", type=Path, nargs="+", required=True, help="ESMFold2 shard output dirs")
    parser.add_argument("--recognition", type=Path, required=True, help="the Pfam oracle output dir")
    parser.add_argument("--likelihood", type=Path, nargs="+", required=True, help="likelihood output dirs")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--device", default="cpu", help="accepted because the campaign queue injects it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.draws < 1000:
        parser.error("--draws below 1000 gives percentile intervals this package will not publish")
    record = run(args)
    print(
        json.dumps(
            {
                "status": record["status"],
                "e14_arms": sorted(record["e14"]["arms"]),
                "e17_arms": sorted(record["e17"]["arms"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
