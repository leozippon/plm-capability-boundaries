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
import collections
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
    "predicted_delta_g",
)

EVALUATOR_ORIENTATION = (
    "every evaluator is oriented so that a larger value is better, including "
    "neg_mean_pae_angstrom, which is the negated mean predicted aligned error in angstrom"
)

#: Which declared quantity each evaluator reports. Folding confidence and
#: predicted free energy are different physical quantities and never stand in for
#: one another; the mapping is here so the artefact states which is which.
EVALUATOR_QUANTITY: dict[str, str] = {
    "mean_ca_plddt": "esmfold2_mean_ca_plddt",
    "ptm": "esmfold2_ptm",
    "confident_fold": "esmfold2_confidence_event",
    "neg_mean_pae_angstrom": "esmfold2_predicted_aligned_error",
    "complete_domain": "pfam_complete_domain",
    "any_family": "pfam_any_family",
    "predicted_delta_g": "predicted_delta_g",
}

#: The evaluators the length-conditional contrast is computed for. Not all of
#: them: it is the most expensive quantity here and the question it answers is
#: about candidate quality, so it runs on the structural confidence, the
#: structural event and the family call, which are the three endpoints a reader
#: would act on.
LENGTH_CONDITIONAL_EVALUATORS: tuple[str, ...] = (
    "mean_ca_plddt",
    "confident_fold",
    "complete_domain",
)

#: Evaluators that exist for only part of the pool, with the reason. A curve on a
#: sub-pool is a selection experiment on that sub-pool; it is reported as such and
#: never read as the pool's result.
SUB_POOL_EVALUATORS: dict[str, str] = {
    "predicted_delta_g": (
        "the validated stability head is licensed only inside the residue band its "
        "training fold covers, so pool members outside that band carry no free-energy "
        "prediction at all. The band was not chosen to suit this pool and must not be "
        "allowed to select it silently"
    ),
}

#: What each selector reads. The two controls are the point of the design: a gain
#: that a composition score or a length score reproduces is not evidence that the
#: model's likelihood carries usable information.
SELECTOR_DECLARATION: dict[str, str] = {
    "random": "a seeded uniform key; the baseline that must be beaten",
    "likelihood": (
        "the generating model's own mean negative log-likelihood per scored token on "
        "its own product, recomputed under the arm's native rendering"
    ),
    "composition": (
        "nats per residue of the sequence's composition under a Swiss-Prot unigram "
        "background. The cheap external control"
    ),
    "length": (
        "sequence length, longest first. The second control: every structural "
        "confidence rises with length, so a length selector prices how much of any "
        "gain is a length effect"
    ),
    "combined": "the within-pool average rank of likelihood and composition",
}

#: Prior measurements from this repository that bound how much a rendering mistake
#: could have moved the likelihood selector. They are quoted, not re-measured
#: here, and they are reported beside the result because the selector would be
#: meaningless if the rendering were wrong by more than the effect being sought.
RENDERING_RISK: dict[str, Any] = {
    "protgpt2_unwrapped_penalty_nats_per_token": 1.42,
    "protgpt2_note": (
        "ProtGPT2 scored as one unwrapped line instead of the 60-column FASTA layout "
        "it was pretrained on costs 1.42 nats/token, measured on 80 Swiss-Prot records "
        "(8.046 raw versus 6.652 wrapped). The rendering used here is the wrapped one, "
        "resolved from the arm declaration rather than spelled at the call site"
    ),
    "zymctrl_tag_leak_nats": 1.73,
    "zymctrl_note": (
        "ZymCTRL's EC conditioning prompt leaks 1.73 nats if it is scored as cohort "
        "content instead of as a prompt (EXP-R2-034). The scoring here masks the span "
        "between the declared <start> and <end> boundary ids, so the tag is not scored"
    ),
    "why_this_is_reported": (
        "both numbers are larger than any selection effect this experiment could find, "
        "so they bound the damage a rendering error would do. The per-arm rendering "
        "actually used is recorded in each likelihood stage's own artefact"
    ),
}


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
    stability: dict[str, float] | None = None,
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
        if name == "predicted_delta_g":
            # Absent means "outside the licensed band", which is a statement about
            # the instrument's support and never a low free energy.
            value = (stability or {}).get(identifier)
            if value is None:
                dropped.append(identifier)
                continue
            values.append(float(value))
            kept.append(identifier)
            continue
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
    stability: dict[str, float] | None = None,
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
            [pair["generated_id"]], structure, recognition, evaluator, stability
        )
        natural, missing_n = evaluator_vector(
            [pair["natural_id"]], structure, recognition, evaluator, stability
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
    novelty: dict[str, float] = {}
    novelty_receipt: dict[str, Any] | None = None
    if args.novelty is not None:
        novelty_receipt = json.loads(
            (args.novelty / "generated_novelty.json").read_text(encoding="utf-8")
        )
        for row in read_jsonl(args.novelty / "generated_novelty.jsonl"):
            novelty[str(row["id"])] = float(row["nearest_corpus_identity"])

    stability: dict[str, float] = {}
    stability_receipt: dict[str, Any] | None = None
    if args.stability is not None:
        stability_receipt = json.loads(
            (args.stability / "domain_stability_evaluation.json").read_text(encoding="utf-8")
        )
        gate = (stability_receipt.get("gates") or {}).get("plm") or {}
        if gate.get("passed"):
            sidecar = args.stability / "generated_stability.jsonl"
            for row in read_jsonl(sidecar):
                stability[str(row["id"])] = float(row["predicted_delta_g_kcal_per_mol"])
        else:
            # A failed gate blocks the *use* of the instrument, not the rest of the
            # experiment. The free-energy evaluator then reports itself unavailable
            # with the gate's reason, and every other evaluator is unaffected.
            stability_withheld = {
                "withheld": True,
                "reason": gate.get("reason"),
                "consequence": (
                    "no free-energy evaluator is reported. The structural and family "
                    "evaluators are unaffected"
                ),
            }
            stability_receipt = dict(stability_receipt, stability_withheld=stability_withheld)
            print(json.dumps({"stability": "withheld", "reason": gate.get("reason")}), flush=True)

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
                    members, structure, recognition, evaluator, stability
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
            left, right, usable, _ = paired_vectors(
                pairs, structure, recognition, evaluator, stability
            )
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
        absent = [row["id"] for row in members if row["id"] not in likelihood]
        if absent:
            raise SystemExit(
                f"{len(absent)} pool members of {arm!r} carry no recomputed likelihood "
                f"(first {absent[:3]}); the selector is incomplete and no curve is reported"
            )
        lengths = np.asarray([int(row["length"]) for row in members], dtype=np.float64)
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
            # The length control. Every structural confidence rises with length,
            # so a selector that quietly prefers long sequences would look like a
            # selector that prefers good ones. Selecting by length alone prices
            # exactly that, and any likelihood gain it reproduces is a length gain.
            "length": -lengths,
        }
        selectors["combined"] = gp.rank_average(selectors["likelihood"], selectors["composition"])
        selectors["random"] = gp.random_key(len(members), seed=args.seed + 991)
        correlations = {
            "likelihood_vs_composition_spearman": spearman(
                selectors["likelihood"], selectors["composition"]
            ),
            "likelihood_vs_length_spearman": spearman(selectors["likelihood"], lengths),
            "composition_vs_length_spearman": spearman(selectors["composition"], lengths),
        }
        termination = collections.Counter(str(row.get("decoder_stop")) for row in members)
        strata = collections.Counter(str(row.get("stratum")) for row in members)

        # Diversity is a property of a selected set, not of an evaluator, so each
        # distinct evaluator support set is profiled once and shared.
        profile_cache: dict[tuple[str, str, float], dict[str, Any]] = {}
        reference_cache: dict[tuple[str, float], dict[str, Any]] = {}

        def profiles_for(support_key, kept_rows, families, identities, name, fraction, scored):
            key = (support_key, name, fraction)
            if key not in profile_cache:
                chosen = gp.selection_indices(scored, fraction=fraction)
                profile_cache[key] = gp.selected_set_profile(
                    [str(kept_rows[i]["sequence"]) for i in chosen],
                    [families[i] for i in chosen],
                    corpus_identity=None
                    if identities is None
                    else [identities[i] for i in chosen],
                )
            reference_key = (support_key, fraction)
            if reference_key not in reference_cache:
                reference_cache[reference_key] = gp.diversity_reference(
                    [str(row["sequence"]) for row in kept_rows],
                    families,
                    fraction=fraction,
                    seed=args.seed + 991,
                    corpus_identity=identities,
                )
            return profile_cache[key], reference_cache[reference_key]

        per_evaluator: dict[str, Any] = {}
        for evaluator in EVALUATORS:
            values, dropped = evaluator_vector(
                [row["id"] for row in members], structure, recognition, evaluator, stability
            )
            missing = set(dropped)
            kept = [row for row in members if row["id"] not in missing]
            block: dict[str, Any] = {
                "n_dropped": len(dropped),
                "quantity": EVALUATOR_QUANTITY[evaluator],
            }
            if evaluator in SUB_POOL_EVALUATORS:
                block["sub_pool"] = {
                    "n_pool": len(members),
                    "n_evaluable": len(kept),
                    "coverage_fraction": len(kept) / len(members),
                    "why": SUB_POOL_EVALUATORS[evaluator],
                    "selection_is_re_ranked_within_the_sub_pool": True,
                    "warning": (
                        "this evaluator exists for only part of the pool, so its curve is "
                        "a selection experiment on that sub-pool and not on the pool the "
                        "other evaluators use. The two are not interchangeable"
                    ),
                }
                if kept:
                    sub_lengths = np.asarray([int(row["length"]) for row in kept])
                    block["sub_pool"]["length"] = {
                        "min": int(sub_lengths.min()),
                        "max": int(sub_lengths.max()),
                        "mean": float(sub_lengths.mean()),
                    }
            if len(kept) < MINIMUM_BOOTSTRAP_UNITS:
                block["status"] = "unavailable_on_this_pool"
                block["reason"] = (
                    f"{len(kept)} of {len(members)} pool members are evaluable, below the "
                    f"{MINIMUM_BOOTSTRAP_UNITS}-unit floor. Reported as unavailable rather "
                    "than estimated on a remnant"
                )
                per_evaluator[evaluator] = block
                continue
            mask = np.asarray([row["id"] not in missing for row in members], dtype=bool)
            units = [str(row.get(unit) if unit != "id" else row["id"]) for row in kept]
            families = [recognition.get(row["id"], {}).get("pfam_families", []) for row in kept]
            identities = (
                [novelty[row["id"]] for row in kept]
                if novelty and all(row["id"] in novelty for row in kept)
                else None
            )
            kept_lengths = [int(row["length"]) for row in kept]
            support_key = f"{evaluator}:{len(kept)}" if evaluator in SUB_POOL_EVALUATORS else f"shared:{len(kept)}"
            curves: dict[str, Any] = {}
            for name, score in sorted(selectors.items()):
                scored = score[mask]
                points: list[dict[str, Any]] = []
                for fraction in gp.SELECTION_FRACTIONS:
                    contrast = gp.selection_contrast(
                        values, scored, units, fraction=fraction, seed=args.seed, n_bootstrap=args.draws
                    )
                    profile, reference = profiles_for(
                        support_key, kept, families, identities, name, fraction, scored
                    )
                    contrast["selected_set"] = profile
                    contrast["size_matched_random_reference"] = reference
                    contrast["collapse"] = gp.collapse_check(
                        profile, reference, yield_difference=contrast.get("difference_vs_random")
                    )
                    # The number that separates protein knowledge from a length
                    # proxy. Skipped for the random arm, where it has no meaning,
                    # and for evaluators outside the declared set, where it would
                    # cost more than it tells.
                    if name != "random" and evaluator in LENGTH_CONDITIONAL_EVALUATORS:
                        contrast["length_conditional"] = gp.length_conditional_contrast(
                            values,
                            scored,
                            kept_lengths,
                            units,
                            fraction=fraction,
                            seed=args.seed,
                            n_bootstrap=args.draws,
                        )
                    points.append(contrast)
                curves[name] = points
            block.update(
                status="measured",
                n_pool=int(mask.sum()),
                pool_mean=float(values.mean()),
                pool_profile=gp.selected_set_profile(
                    [str(row["sequence"]) for row in kept], families
                ),
                random_baseline=[
                    gp.random_baseline(values, fraction=fraction, seed=args.seed + 991)
                    for fraction in gp.SELECTION_FRACTIONS
                ],
                curves=curves,
                selector_evaluator_spearman={
                    name: spearman(score[mask], values) for name, score in sorted(selectors.items())
                },
            )
            per_evaluator[evaluator] = block
        e17_results[arm] = {
            "declaration": declaration,
            "n_members": len(members),
            "selector_correlations": correlations,
            "selectors": dict(SELECTOR_DECLARATION),
            "termination": {
                "decoder_stop": dict(termination),
                "strata": dict(strata),
                "statement": (
                    "the pool is natively terminated by construction, so selection here "
                    "cannot be rescuing budget-censored continuations and no part of any "
                    "gain is a truncation effect"
                ),
            },
            "length": {
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "mean": float(lengths.mean()),
            },
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
            "stability": None if args.stability is None else str(args.stability),
            "novelty": None if args.novelty is None else str(args.novelty),
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
            "rendering_risk": RENDERING_RISK,
            "length_conditional_evaluators": list(LENGTH_CONDITIONAL_EVALUATORS),
            "length_conditional_question": (
                "a selector correlated with length beats an unrestricted random draw "
                "without knowing anything about proteins, because every structural "
                "confidence rises with length. The length_conditional block beside each "
                "point compares the selected set against a random set of the same size "
                "AND the same length composition. A selector that beats that carries "
                "information beyond length; one that does not is a length proxy"
            ),
            "novelty_receipt": novelty_receipt
            and {
                key: novelty_receipt[key]
                for key in (
                    "n_searched",
                    "n_with_any_hit",
                    "identity_over_query",
                    "identity_strata",
                    "masking_rationale",
                    "database",
                    "ceiling",
                )
                if key in novelty_receipt
            },
            "evaluator_quantities": dict(EVALUATOR_QUANTITY),
            "sub_pool_evaluators": dict(SUB_POOL_EVALUATORS),
            "stability_receipt": stability_receipt
            and {
                key: stability_receipt[key]
                for key in ("licensed_band", "gates", "limitations", "support", "stability_withheld")
                if key in stability_receipt
            },
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
            "not a gain: the collapse verdict beside each point is the computed form of "
            "that judgement",
            "the composition and length selectors are controls, not competitors. A "
            "likelihood gain either of them reproduces is not evidence that the model's "
            "likelihood carries usable information",
            "predicted free energy covers only the sub-pool inside the stability "
            "instrument's licensed band and is a prediction in kcal/mol, never a "
            "measurement and never interchangeable with a folding confidence",
            "read the length_conditional block before the difference_vs_random one: "
            "the second answers whether a selector beats chance, the first whether it "
            "beats length, and only the first bears on whether the model's likelihood "
            "carries protein knowledge",
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
    parser.add_argument(
        "--novelty",
        type=Path,
        default=None,
        help="the homology-search output dir supplying each sequence's nearest corpus identity",
    )
    parser.add_argument(
        "--stability",
        type=Path,
        default=None,
        help=(
            "the validated stability application dir; its gate must have passed and its "
            "predictions are read only inside the licensed residue band"
        ),
    )
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
