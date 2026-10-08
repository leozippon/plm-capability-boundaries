#!/usr/bin/env python3
"""E09: mutation-effect ranking as a function of the context homologue's identity.

One frozen checkpoint, one target at a time, every condition of that target built
to the same item count and the same per-item token length, and nothing varying
across conditions but what the context *is*. The scored quantity is the native
log-likelihood difference between a variant and its own wild type, read over the
target's own tokens only; the endpoint is the within-assay Spearman correlation of
that difference with the measured effect.

Three phases:

``plan``
    Tokeniser only, no weights: resolve each target's item-token window, its item
    count and which conditions it can supply. Cheap enough to run anywhere, and
    it is what makes the support of this experiment inspectable before a GPU is
    occupied.
``score``
    One GPU. Scores every admitted assay under every present condition and writes
    the per-assay correlations.
``analyse``
    CPU, across arms: the gain-versus-identity curve with its per-bin power, read
    on the complete-case panel the declaration names as primary, and the
    identity at which the gain vanishes.

Everything frozen is in :mod:`src.capability.context.homology_context`; the
packing, the scored span and the batch-invariance check are the context-homologue
stage's own, imported rather than restated.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.capability.entrypoints import stage_path  # noqa: E402
from src.capability.context import homology_context as H  # noqa: E402
from src.capability.context import context_homologue as ch  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.models.fitness import parse_mutant  # noqa: E402

EXPECT_PLAN = "context_identity_plan.json"
EXPECT_SCORE = "context_identity_scores.json"
EXPECT_ANALYSE = "context_identity_curve.json"

#: Batch invariance tolerances, the context-homologue stage's own: a packed row's
#: target NLL must agree with the same row scored alone, and the mutant-minus-wild
#: differences -- the quantity the endpoint is built from -- far more tightly.
MAX_BATCH_SINGLE_NATS_PER_TOKEN = 0.02
MAX_MUTANT_DELTA_NATS = 0.001


def stage_module(filename: str):
    """Load another stage by basename, the way the rescue diagnostic already does."""

    spec = importlib.util.spec_from_file_location(filename, stage_path(ROOT, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime(device: str) -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "device": device,
        "peak_rss_bytes": usage.ru_maxrss * 1024,
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
        "threads_requested": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def rho(left, right) -> float | None:
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def variant_sequences(wildtype: str, mutants: list[str]) -> list[str]:
    """Rebuild the scored variants from the wild type and the mutation strings."""

    out = []
    for mutant in mutants:
        residues = list(wildtype)
        for wild, position, mutated in parse_mutant(mutant):
            if residues[position - 1] != wild:
                raise ValueError(
                    f"{mutant}: the wild type carries {residues[position - 1]!r} at "
                    f"position {position}"
                )
            residues[position - 1] = mutated
        out.append("".join(residues))
    return out


def load_contexts(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    H.require_declaration(record, scope="context")
    if record.get("stage") != "bin":
        raise SystemExit(f"{path} is not a binned retrieval artefact")
    return record


def build_plans(arm, contexts: dict, *, budget: int) -> tuple[dict, dict]:
    """Per-target condition plans under one arm's own tokenisation.

    The item-token referent is the target's own rendered length, so neither the
    item count nor the length window can depend on which bin supplied an item.
    """

    prefix = len(ch.row_prefix_ids(arm))

    def tokens_of(sequence: str) -> int:
        return len(ch.item_ids(arm, sequence, modality="protein"))

    assays_by_target: dict[str, list[dict]] = {}
    for assay in contexts["assays"]:
        assays_by_target.setdefault(assay["target_id"], []).append(assay)

    plans: dict[str, dict] = {}
    statuses: dict[str, dict[str, str]] = {}
    for target in contexts["targets"]:
        identifier = target["target_id"]
        wildtype = target["wildtype"]
        referent = tokens_of(wildtype)
        widest = referent
        for assay in assays_by_target.get(identifier, []):
            for sequence in variant_sequences(wildtype, assay["mutants"]):
                widest = max(widest, tokens_of(sequence))
        room = budget - prefix - widest
        conditions, budget_record = H.plan_conditions(
            wildtype=wildtype,
            bins=target["bins"],
            donors=target["unrelated_pool"],
            self_copy=target.get("self_copy"),
            token_length=tokens_of,
            referent=referent,
            room=room,
        )
        plans[identifier] = {
            "target_id": identifier,
            "cluster": target["cluster"],
            "budget": {**budget_record, "row_prefix_tokens": prefix, "widest_target_tokens": widest},
            "conditions": {name: plan.record() for name, plan in conditions.items()},
            "items": {
                name: [
                    {"sequence": item["sequence"], "tokens": item["tokens"], "subject": item["subject"]}
                    for item in plan.items
                ]
                for name, plan in conditions.items()
            },
        }
        statuses[identifier] = {name: plan.status for name, plan in conditions.items()}
    support = plan_support(plans, statuses, contexts)
    return plans, support


def plan_support(plans: dict, statuses: dict, contexts: dict) -> dict:
    cluster_of = {identifier: plan["cluster"] for identifier, plan in plans.items()}
    present = {
        name: [identifier for identifier, status in statuses.items() if status.get(name) == "present"]
        for name in H.CONDITIONS + (H.CEILING_CONDITION,)
    }
    clusters = {
        name: len({cluster_of[identifier] for identifier in targets})
        for name, targets in present.items()
    }
    admitted = H.admitted_bins(
        {name: len({cluster_of[identifier] for identifier in
                    set(present[name]) & set(present[H.PRIMARY_REFERENT])})
         for name in H.BIN_NAMES}
    )
    balanced = H.balanced_targets(statuses, bins=admitted)
    assays_by_target: dict[str, list[str]] = {}
    for assay in contexts["assays"]:
        assays_by_target.setdefault(assay["target_id"], []).append(assay["assay"])
    paired = {}
    overlap = {}
    for name in H.BIN_NAMES:
        shared = set(present[name]) & set(present[H.PRIMARY_REFERENT])
        paired[name] = {
            "targets": len(shared),
            "clusters": len({cluster_of[identifier] for identifier in shared}),
            "assays": sum(len(assays_by_target.get(identifier, ())) for identifier in shared),
        }
        overlap[name] = {
            other: len(set(present[name]) & set(present[other])) for other in H.BIN_NAMES
        }
    return {
        "targets": len(plans),
        "item_counts": {
            str(count): sum(1 for plan in plans.values() if plan["budget"]["item_count"] == count)
            for count in sorted({plan["budget"]["item_count"] for plan in plans.values()})
        },
        "targets_present_by_condition": {name: len(value) for name, value in present.items()},
        "clusters_present_by_condition": clusters,
        "assays_present_by_condition": {
            name: sum(len(assays_by_target.get(identifier, ())) for identifier in targets)
            for name, targets in present.items()
        },
        "absent_reasons": {
            name: sorted(
                {
                    plan["conditions"][name]["reason"]
                    for plan in plans.values()
                    if plan["conditions"][name]["status"] == "absent"
                    and plan["conditions"][name]["reason"]
                }
            )[:6]
            for name in H.CONDITIONS + (H.CEILING_CONDITION,)
        },
        "paired_support_by_bin": paired,
        "paired_referent": H.PRIMARY_REFERENT,
        "bin_target_overlap": overlap,
        "group_floor": H.GROUP_FLOOR,
        "admitted_bins": list(admitted),
        "refused_bins": [name for name in H.CURVE_ORDER if name not in admitted],
        "primary_panel": H.PRIMARY_PANEL,
        "balanced_targets": len(balanced),
        "balanced_clusters": len({cluster_of[identifier] for identifier in balanced}),
        "balanced_assays": sum(len(assays_by_target.get(identifier, ())) for identifier in balanced),
        "balanced_target_ids": balanced,
    }


def plan(args: argparse.Namespace) -> None:
    contexts = load_contexts(args.homologs)
    arm = ch.tokenizer_arm(args.arm)
    plans, support = build_plans(arm, contexts, budget=args.budget)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out / EXPECT_PLAN,
        {
            "schema_version": H.SCHEMA_VERSION,
            "stage": "plan",
            "arm": args.arm,
            **H.declaration_digests(),
            "declaration": H.declaration(),
            "homologs_sha256": sha256_file(args.homologs),
            "corpus_identity": contexts["corpus_identity"],
            "budget": args.budget,
            "caveat": ch.CAVEATS.get(args.arm),
            "support": support,
            "plans": plans,
            "runtime": runtime("cpu"),
        },
    )
    print(
        f"plan {args.arm}: {support['targets']} targets, item counts {support['item_counts']}, "
        f"admitted bins {support['admitted_bins']}, complete-case targets "
        f"{support['balanced_targets']} in {support['balanced_clusters']} clusters",
        flush=True,
    )


def score(args: argparse.Namespace) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    contexts = load_contexts(args.homologs)
    stage = stage_module("context_homologue.py")
    arm = stage.load_scorable_arm(args.arm, device=args.device, dtype=args.dtype)
    ch.require_position_budget(arm.model.config, arm=args.arm)
    plans, support = build_plans(arm, contexts, budget=args.budget)
    args.out.mkdir(parents=True, exist_ok=True)

    targets = {target["target_id"]: target for target in contexts["targets"]}
    assays = sorted(contexts["assays"], key=lambda row: row["assay"])
    if args.assay_limit:
        assays = assays[: args.assay_limit]
    started = time.monotonic()
    checks: dict[str, dict] = {}
    results: list[dict] = []
    conditions = tuple(H.CONDITIONS) + (H.CEILING_CONDITION,)

    for number, assay in enumerate(assays, start=1):
        target = targets[assay["target_id"]]
        plan_record = plans[assay["target_id"]]
        wildtype = target["wildtype"]
        sequences = [wildtype] + variant_sequences(wildtype, assay["mutants"])
        packed_targets = [ch.item_ids(arm, sequence, modality="protein") for sequence in sequences]
        spans = [
            ch.target_span(arm, ids, record=sequence)
            for ids, sequence in zip(packed_targets, sequences)
        ]
        row_prefix = ch.row_prefix_ids(arm)
        scored: dict[str, list[float]] = {}
        wild: dict[str, float] = {}
        absent: dict[str, str] = {}
        for condition in conditions:
            record = plan_record["conditions"][condition]
            if record["status"] != "present":
                absent[condition] = record["reason"] or "absent"
                continue
            context_ids: list[int] = []
            for item in plan_record["items"][condition]:
                context_ids.extend(ch.item_ids(arm, item["sequence"], modality="protein"))
            prefix = row_prefix + context_ids
            rows = [prefix + ids for ids in packed_targets]
            if max(len(row) for row in rows) > args.budget:
                raise RuntimeError(
                    f"{assay['assay']} {condition}: packed row exceeds the {args.budget}-position budget"
                )
            sums: list[float] = []
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start : start + args.batch_size]
                logits, ids = stage._forward_rows(arm, batch)
                for index in range(len(batch)):
                    left, right = spans[start + index]
                    value = stage._target_nll(
                        logits[index : index + 1],
                        ids[index : index + 1],
                        len(prefix) + left,
                        len(prefix) + right,
                    )
                    sums.append(-value["nll_sum"])
                del logits, ids
            if not np.isfinite(sums).all():
                raise RuntimeError(f"{assay['assay']} {condition}: non-finite likelihood")
            if condition not in checks:
                single = []
                for index in range(min(3, len(rows))):
                    logits, ids = stage._forward_rows(arm, [rows[index]])
                    left, right = spans[index]
                    value = stage._target_nll(
                        logits, ids, len(prefix) + left, len(prefix) + right
                    )
                    single.append(-value["nll_sum"])
                    del logits, ids
                gaps = [
                    abs(sums[index] - single[index]) / (spans[index][1] - spans[index][0])
                    for index in range(len(single))
                ]
                deltas = [
                    (sums[index] - sums[0]) - (single[index] - single[0])
                    for index in range(1, len(single))
                ]
                checks[condition] = {
                    "max_batch_single_nats_per_token": max(gaps),
                    "mutant_delta_difference_nats": deltas,
                }
                if max(gaps) > MAX_BATCH_SINGLE_NATS_PER_TOKEN or (
                    deltas and max(abs(value) for value in deltas) > MAX_MUTANT_DELTA_NATS
                ):
                    raise RuntimeError(f"batch/single check failed: {checks[condition]}")
            wild[condition] = sums[0]
            scored[condition] = (np.asarray(sums[1:]) - sums[0]).tolist()

        measured = assay["measured"]
        results.append(
            {
                "assay": assay["assay"],
                "target_id": assay["target_id"],
                "cluster": assay["cluster"],
                "mutant_digest": assay["mutant_digest"],
                "variants": len(measured),
                "item_count": plan_record["budget"]["item_count"],
                "wt_scored_tokens": spans[0][1] - spans[0][0],
                "context_tokens": {
                    condition: plan_record["conditions"][condition]["context_tokens"]
                    for condition in scored
                },
                "context_identities": {
                    condition: plan_record["conditions"][condition]["identities"]
                    for condition in scored
                },
                "context_max_lcs": {
                    condition: plan_record["conditions"][condition]["max_lcs_to_target"]
                    for condition in scored
                },
                "context_subjects": {
                    condition: plan_record["conditions"][condition]["subjects"]
                    for condition in scored
                },
                "wt_log_likelihood": wild,
                "spearman": {
                    condition: rho(values, measured) for condition, values in scored.items()
                },
                "lookup_spearman": rho(assay["profile_scores"], measured),
                "absent_conditions": absent,
            }
        )
        # Written every assay: an interrupted run leaves a readable partial record
        # whose status says so, and never a success record on partial data.
        write_json(
            args.out / EXPECT_SCORE,
            {
                "schema_version": H.SCHEMA_VERSION,
                "stage": "score",
                "status": "complete" if number == len(assays) else "running",
                "arm": args.arm,
                "dtype": args.dtype,
                "device": args.device,
                **H.declaration_digests(),
                "declaration": H.declaration(),
                "homologs_sha256": sha256_file(args.homologs),
                "corpus_identity": contexts["corpus_identity"],
                "caveat": ch.CAVEATS.get(args.arm),
                "scoring_stratum": "target_only_native_packed_residue_span",
                "budget": args.budget,
                "batch_size": args.batch_size,
                "batch_single_checks": checks,
                "batch_single_tolerances": {
                    "nats_per_token": MAX_BATCH_SINGLE_NATS_PER_TOKEN,
                    "mutant_delta_nats": MAX_MUTANT_DELTA_NATS,
                },
                "support": support,
                "assay_limit": args.assay_limit or None,
                "assays_scored": number,
                "assays_declared": len(assays),
                "elapsed_seconds": time.monotonic() - started,
                "assays": results,
                "runtime": runtime(args.device),
            },
        )
        print(
            f"{number}/{len(assays)} {assay['assay']} "
            f"{ {key: None if value is None else round(value, 4) for key, value in results[-1]['spearman'].items()} }",
            flush=True,
        )
    if not results:
        raise RuntimeError("no assay was scored")


# ------------------------------------------------------------------ the analysis


def referent_value(assay: dict, referent: str) -> float | None:
    """The referent correlation of one assay, whichever kind of referent it is.

    The two context referents are conditions this run scored; the
    evolutionary-profile referent is the frozen lookup score the cohort carries,
    so it lives beside the conditions rather than among them.
    """

    if referent == H.PROFILE_REFERENT:
        value = assay.get("lookup_spearman")
        return None if value is None else float(value)
    value = assay["spearman"].get(referent)
    return None if value is None else float(value)


def family_matrix(records: list[dict], *, referent: str, bins: list[str], targets: set[str] | None):
    """``(families, columns, matrix)`` of per-family mean contrasts.

    One column per (arm, bin). A family's value is the mean over its assays of
    that assay's Spearman under the bin minus its Spearman under the referent, so
    assays carry equal weight inside a family and families are the resampled unit.
    ``NaN`` marks a family that supplies no assay for that column: structural
    absence, never a zero.
    """

    rows: dict[tuple[str, str], dict[int, list[float]]] = {}
    families: set[int] = set()
    for record in records:
        arm = record["arm"]
        for assay in record["assays"]:
            if targets is not None and assay["target_id"] not in targets:
                continue
            reference = referent_value(assay, referent)
            if reference is None:
                continue
            families.add(int(assay["cluster"]))
            for name in bins:
                value = assay["spearman"].get(name)
                if value is None:
                    continue
                rows.setdefault((arm, name), {}).setdefault(int(assay["cluster"]), []).append(
                    float(value) - float(reference)
                )
    order = sorted(families)
    columns = [(record["arm"], name) for record in records for name in bins]
    matrix = np.full((len(order), len(columns)), np.nan)
    for index, column in enumerate(columns):
        per_family = rows.get(column, {})
        for position, family in enumerate(order):
            values = per_family.get(family)
            if values:
                matrix[position, index] = float(np.mean(values))
    return order, columns, matrix


def panel_statistics(records: list[dict], *, referent: str, bins: list[str], targets, label: str):
    from src.capability.extensions.phenotype_strata import shared_bootstrap

    families, columns, matrix = family_matrix(
        records, referent=referent, bins=bins, targets=targets
    )
    keep = [index for index in range(matrix.shape[1]) if np.isfinite(matrix[:, index]).sum() >= 2]
    dropped = [
        {"arm": columns[index][0], "bin": columns[index][1], "reason": "fewer than two families"}
        for index in range(matrix.shape[1])
        if index not in keep
    ]
    if not keep:
        return {
            "panel": label,
            "referent": referent,
            "bins": bins,
            "families": len(families),
            "status": "no estimable contrast",
            "dropped_columns": dropped,
        }
    statistics, _ = shared_bootstrap(
        matrix[:, keep], draws=H.BOOTSTRAP_DRAWS, seed=H.BOOTSTRAP_SEED
    )
    critical = float(statistics["critical_value"])
    per_arm: dict[str, list[dict]] = {}
    for position, index in enumerate(keep):
        arm, name = columns[index]
        shared = {
            "point": statistics["point"][position],
            "standard_error": statistics["se"][position],
            "groups": statistics["supported_families"][position],
            "interval": statistics["simultaneous_interval"][position],
            "critical": critical,
        }
        # The ceiling column is a condition, not an identity bin, so it carries no
        # identity edges; everything else about its estimate reads the same way.
        record = (
            H.bin_power_record(bin_name=name, **shared)
            if name in H.BIN_NAMES
            else {"bin": name, **H.power_record(**shared)}
        )
        per_arm.setdefault(arm, []).append(
            record | {"pointwise_interval": statistics["pointwise_interval"][position]}
        )
    return {
        "panel": label,
        "referent": referent,
        "bins": bins,
        "families": len(families),
        "contrasts": len(keep),
        "dropped_columns": dropped,
        "bootstrap": {
            key: statistics[key]
            for key in (
                "draws",
                "seed",
                "confidence",
                "critical_value",
                "family_size",
                "original_families",
                "jointly_rejected_draws",
                "method",
                "conditional_on_fitted_predictions",
            )
        },
        "curve": {
            arm: sorted(values, key=lambda row: -row.get("identity_low", -1.0))
            for arm, values in per_arm.items()
        },
        "vanishing_point": {
            arm: H.vanishing_point([row for row in values if row["bin"] in H.BIN_NAMES])
            for arm, values in per_arm.items()
            if any(row["bin"] in H.BIN_NAMES for row in values)
        },
    }


def analyse(args: argparse.Namespace) -> None:
    records = []
    for path in sorted(args.scores):
        record = json.loads(path.read_text(encoding="utf-8"))
        H.require_declaration(record, scope="context")
        if record.get("status") != "complete":
            raise SystemExit(f"{path} reports status {record.get('status')!r}; refusing a partial fit")
        record["_path"] = str(path)
        record["_sha256"] = sha256_file(path)
        records.append(record)
    if not records:
        raise SystemExit("no score record given")
    arms = [record["arm"] for record in records]
    if len(set(arms)) != len(arms):
        raise SystemExit(f"two score records carry the same arm: {arms}")
    support = records[0]["support"]
    admitted = [name for name in support["admitted_bins"]]
    balanced = set(support["balanced_target_ids"])
    for record in records[1:]:
        if record["support"]["admitted_bins"] != support["admitted_bins"]:
            raise SystemExit(
                "arms disagree on the admitted bin set; they were planned against "
                "different retrieval artefacts"
            )
    args.out.mkdir(parents=True, exist_ok=True)
    panels = []
    for referent in H.REFERENTS:
        panels.append(
            panel_statistics(
                records, referent=referent, bins=admitted, targets=None, label=H.PRIMARY_PANEL
            )
        )
        if support["balanced_clusters"] >= H.GROUP_FLOOR:
            panels.append(
                panel_statistics(
                    records,
                    referent=referent,
                    bins=admitted,
                    targets=balanced,
                    label=H.SECONDARY_PANEL,
                )
            )
        else:
            panels.append(
                {
                    "panel": H.SECONDARY_PANEL,
                    "referent": referent,
                    "bins": admitted,
                    "status": "unsupported",
                    "clusters": support["balanced_clusters"],
                    "group_floor": H.GROUP_FLOOR,
                    "reason": (
                        "a complete-case target must supply every admitted bin and both "
                        "controls at a matched token total; this retrieval leaves too few "
                        "to clear the independence floor"
                    ),
                }
            )
    ceiling = panel_statistics(
        records,
        referent=H.PRIMARY_REFERENT,
        bins=[H.CEILING_CONDITION],
        targets=None,
        label="ceiling_self_copy",
    )
    realised = {}
    for record in records:
        per_bin: dict[str, list[float]] = {}
        for assay in record["assays"]:
            for name, values in assay["context_identities"].items():
                per_bin.setdefault(name, []).extend(
                    float(value) for value in values if np.isfinite(value)
                )
        realised[record["arm"]] = {
            name: {
                "n_items": len(values),
                "mean_identity": float(np.mean(values)) if values else None,
                "min_identity": float(np.min(values)) if values else None,
                "max_identity": float(np.max(values)) if values else None,
            }
            for name, values in sorted(per_bin.items())
        }
    write_json(
        args.out / EXPECT_ANALYSE,
        {
            "schema_version": H.SCHEMA_VERSION,
            "stage": "analyse",
            "status": "complete",
            **H.declaration_digests(),
            "declaration": H.declaration(),
            "inputs": [
                {"path": record["_path"], "sha256": record["_sha256"], "arm": record["arm"]}
                for record in records
            ],
            "corpus_identity": records[0]["corpus_identity"],
            "support": support,
            "realised_identity_by_bin": realised,
            "panels": panels,
            "ceiling": ceiling,
            "limitations": list(H.LIMITATIONS),
            "runtime": runtime("cpu"),
        },
    )
    for panel in panels:
        print(
            f"{panel['panel']} vs {panel['referent']}: "
            f"{panel.get('contrasts')} contrasts over {panel.get('families')} families; "
            f"vanishing {panel.get('vanishing_point')}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["plan", "score", "analyse"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arm")
    parser.add_argument("--homologs", type=Path)
    parser.add_argument("--scores", type=Path, nargs="*", default=[])
    parser.add_argument("--budget", type=int, default=H.POSITION_BUDGET)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--assay-limit", type=int, default=0)
    args = parser.parse_args()
    if args.budget != H.POSITION_BUDGET:
        raise SystemExit(
            f"the declaration fixes the position budget at {H.POSITION_BUDGET}; a run at "
            f"{args.budget} is a different experiment"
        )
    if args.phase in {"plan", "score"}:
        if args.arm is None or args.homologs is None:
            raise SystemExit("--arm and --homologs are required")
        (plan if args.phase == "plan" else score)(args)
    else:
        if not args.scores:
            raise SystemExit("--scores needs at least one score record")
        analyse(args)


if __name__ == "__main__":
    main()
