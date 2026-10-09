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

#: The batch extent the startup probe measures the spread at. Production never
#: scores at it; see
#: :data:`~src.capability.context.homology_context.SCORING_ROWS_PER_FORWARD` for
#: why the spread is measured and published rather than tolerated.


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


def numerics_preflight(args, arm, stage, targets, plans, assays, *, conditions) -> dict:
    """Gate on repeat determinism; measure and publish the batch-extent spread.

    Production scores one row per forward
    (:data:`~src.capability.context.homology_context.SCORING_ROWS_PER_FORWARD`), so
    the gate is the one the extraction lane declares for batch-size-one scoring:
    the same row scored twice must give the identical number, because at one row
    per forward the repeat *is* the identical computation and any difference is
    non-determinism rather than rounding.

    The batched-versus-single spread is still measured, on the first assay that
    supplies each condition, and published -- not gated on. It is the evidence for
    the protocol: at eight rows per forward this quantity reached 1.5e-3 and 4.6e-3
    nats on the two larger ProGen2 rungs, which is what a tolerance would have had
    to be widened past.
    """

    checks: dict[str, dict] = {}
    for condition in conditions:
        assay = next(
            (
                row
                for row in assays
                if plans[row["target_id"]]["conditions"][condition]["status"] == "present"
            ),
            None,
        )
        if assay is None:
            continue
        target = targets[assay["target_id"]]
        plan_record = plans[assay["target_id"]]
        sequences = [target["wildtype"]] + variant_sequences(
            target["wildtype"], assay["mutants"][:2]
        )
        packed = [ch.item_ids(arm, sequence, modality="protein") for sequence in sequences]
        spans = [
            ch.target_span(arm, ids, record=sequence)
            for ids, sequence in zip(packed, sequences)
        ]
        context_ids: list[int] = []
        for item in plan_record["items"][condition]:
            context_ids.extend(ch.item_ids(arm, item["sequence"], modality="protein"))
        prefix = ch.row_prefix_ids(arm) + context_ids
        rows = [prefix + ids for ids in packed]

        def scored(batch: list[list[int]], offset: int) -> list[float]:
            logits, ids = stage._forward_rows(arm, batch)
            out = []
            for index in range(len(batch)):
                left, right = spans[offset + index]
                value = stage._target_nll(
                    logits[index : index + 1],
                    ids[index : index + 1],
                    len(prefix) + left,
                    len(prefix) + right,
                )
                out.append(-value["nll_sum"])
            del logits, ids
            return out

        single = [scored([row], index)[0] for index, row in enumerate(rows)]
        repeat = [scored([row], index)[0] for index, row in enumerate(rows)]
        drift = [abs(single[index] - repeat[index]) for index in range(len(rows))]
        probe = scored(rows[: H.BATCH_EXTENT_PROBE_ROWS], 0)
        extent = [
            (probe[index] - probe[0]) - (single[index] - single[0])
            for index in range(1, len(probe))
        ]
        checks[condition] = {
            "assay": assay["assay"],
            "rows": len(rows),
            "rows_per_forward": H.SCORING_ROWS_PER_FORWARD,
            "max_repeat_difference_nats": max(drift),
            "repeat_tolerance_nats": H.REPEAT_TOLERANCE_NATS,
            "measured_batch_extent_spread_nats": {
                "probe_rows": len(probe),
                "max_mutant_delta_difference": (
                    max(abs(value) for value in extent) if extent else 0.0
                ),
                "mutant_delta_differences": extent,
                "gated_on": False,
                "note": (
                    "recorded as the evidence for scoring one row per forward; production "
                    "never scores at this batch extent"
                ),
            },
            "finite": bool(np.isfinite(single).all() and np.isfinite(repeat).all()),
        }
        if not checks[condition]["finite"] or max(drift) > H.REPEAT_TOLERANCE_NATS:
            raise RuntimeError(
                f"{args.arm}: a row scored twice at one row per forward differs by "
                f"{max(drift):.3e} nats on condition {condition} (assay "
                f"{assay['assay']}), above the {H.REPEAT_TOLERANCE_NATS:.1e} tolerance. "
                "At one row per forward the repeat is the identical computation, so this "
                "is non-determinism and nothing is scored under it."
            )
    return checks


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
    preflight = {
        "arm": args.arm,
        "device": args.device,
        "dtype": args.dtype,
        "budget": args.budget,
        "rows_per_forward": H.SCORING_ROWS_PER_FORWARD,
        "homologs": str(args.homologs),
        "homologs_sha256": sha256_file(args.homologs),
        "assays_declared": len(assays),
        "first_assay": assays[0]["assay"],
        "last_assay": assays[-1]["assay"],
        "targets": len(targets),
        "item_counts": support["item_counts"],
        "cuda_free_bytes": (
            torch.cuda.mem_get_info(torch.device(args.device))[0]
            if args.device.startswith("cuda")
            else None
        ),
    }
    write_json(args.out / "preflight.json", preflight)
    print(f"preflight: {json.dumps(preflight)}", flush=True)
    results: list[dict] = []
    conditions = tuple(H.CONDITIONS) + (H.CEILING_CONDITION,)

    # The batch-invariance check runs here, on the first assay that supplies each
    # condition, rather than when the scoring loop first reaches that condition.
    # The check is a refusal: if a packed row's target NLL depends on the batch it
    # was scored in, nothing this arm produces under this packing is usable.
    # Discovering that hours into a 201-assay cell throws away every assay already
    # scored, so it is discovered in the first minute instead.
    checks = numerics_preflight(
        args, arm, stage, targets, plans, assays, conditions=conditions
    )
    print(
        f"numerics verified on {len(checks)} conditions at "
        f"{H.SCORING_ROWS_PER_FORWARD} row(s) per forward",
        flush=True,
    )

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
            try:
                for start in range(0, len(rows), H.SCORING_ROWS_PER_FORWARD):
                    batch = rows[start : start + H.SCORING_ROWS_PER_FORWARD]
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
            except Exception as error:
                # A cell that dies mid-loop must say which assay, which condition
                # and which row shapes it died on; the log tail is the only thing
                # a pod failure leaves behind.
                raise RuntimeError(
                    f"{args.arm}: scoring {assay['assay']} under {condition} failed after "
                    f"{len(sums)} of {len(rows)} rows "
                    f"(prefix {len(prefix)}, target span {spans[0]}, widest row "
                    f"{max(len(row) for row in rows)}, budget {args.budget}, assay "
                    f"{number} of {len(assays)}): {type(error).__name__}: {error}"
                ) from error
            if not np.isfinite(sums).all():
                raise RuntimeError(f"{assay['assay']} {condition}: non-finite likelihood")
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
                "rows_per_forward": H.SCORING_ROWS_PER_FORWARD,
                "numerics_checks": checks,
                "numerics_protocol": {
                    "rows_per_forward": H.SCORING_ROWS_PER_FORWARD,
                    "repeat_tolerance_nats": H.REPEAT_TOLERANCE_NATS,
                    "gate": "a row scored twice must give the identical number",
                    "reason": (
                        "the endpoint is a difference of two scored states, so a "
                        "batch-extent dependent shift lands on it directly"
                    ),
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
    depth: dict[tuple[str, str], dict[str, int]] = {}
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
                # Depth, not only group count: the phenotype programme found that
                # variants per unit, rather than number of units, decided which of
                # its cohorts resolved. Here depth is fixed at the cohort's own
                # 128-variant draw, so what varies is how many assays and variants
                # stand behind each bin.
                cell = depth.setdefault(
                    (arm, name), {"assays": 0, "variants": 0, "clusters": 0}
                )
                cell["assays"] += 1
                cell["variants"] += int(assay["variants"])
    order = sorted(families)
    columns = [(record["arm"], name) for record in records for name in bins]
    matrix = np.full((len(order), len(columns)), np.nan)
    for index, column in enumerate(columns):
        per_family = rows.get(column, {})
        for position, family in enumerate(order):
            values = per_family.get(family)
            if values:
                matrix[position, index] = float(np.mean(values))
    return order, columns, matrix, depth


def columns_for(referent: str, admitted: list[str]) -> list[str]:
    """Every condition read against one referent: the bins, the other control, the ceiling.

    The other control is a column rather than a footnote because that is the
    estimate that separates a homology-specific effect from a general prefix cost:
    read against the empty context, the matched-unrelated condition *is* the price
    of having a prefix at all.
    """

    others = [name for name in H.CONTROL_CONDITIONS if name != referent]
    return [*admitted, *others, H.CEILING_CONDITION]


def panel_statistics(records: list[dict], *, referent: str, bins: list[str], targets, label: str):
    from src.capability.extensions.phenotype_strata import shared_bootstrap

    families, columns, matrix, depth = family_matrix(
        records, referent=referent, bins=bins, targets=targets
    )
    for column, cell in depth.items():
        cell["clusters"] = int(np.isfinite(matrix[:, columns.index(column)]).sum())
        cell["variants_per_cluster"] = (
            cell["variants"] / cell["clusters"] if cell["clusters"] else None
        )
        cell["assays_per_cluster"] = (
            cell["assays"] / cell["clusters"] if cell["clusters"] else None
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
            record
            | {
                "pointwise_interval": statistics["pointwise_interval"][position],
                "support": depth.get((arm, name), {}),
            }
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


def mechanism(records: list[dict]) -> dict:
    """Separate a copying/identity-driven degradation from a general prefix cost.

    Two candidate explanations of a context effect, tested with numbers rather
    than asserted:

    (a) **copying or prefix-induced bias toward the prompted sequence**, which
        predicts that the per-target effect tracks how close the context is to the
        target -- its realised identity, and the length of the longest verbatim run
        it shares with the target -- and that it is largest for the verbatim
        ceiling;
    (b) **a general prefix cost**, where any long prefix moves the likelihood
        surface, which predicts an effect roughly independent of identity and
        equally present in the matched-unrelated condition.

    The association is the Spearman correlation, over (assay, bin) cells, between
    the per-cell effect against the empty context and each covariate, with a
    bootstrap over wild-type family clusters so the unit of dependence is the
    family and not the cell. The estimator is the package's own paired group
    bootstrap, whose ``derived_statistic`` hook returns each side's own interval;
    its difference interval says which covariate tracks the effect better.

    The general prefix cost itself is not estimated here: it is the
    matched-unrelated column of the no-context panel.
    """

    from src.capability.core.statistics import (
        MINIMUM_BOOTSTRAP_UNITS,
        paired_group_bootstrap,
    )

    def rank_correlation(truth, values):
        return float(spearmanr(truth, values).statistic)

    out: dict[str, dict] = {}
    for record in records:
        cells = []
        for assay in record["assays"]:
            empty = assay["spearman"].get(H.NO_CONTEXT)
            if empty is None:
                continue
            for name in H.BIN_NAMES:
                value = assay["spearman"].get(name)
                identities = [
                    float(v) for v in assay["context_identities"].get(name, ()) if np.isfinite(v)
                ]
                overlaps = assay["context_max_lcs"].get(name, ())
                if value is None or not identities or not overlaps:
                    continue
                cells.append(
                    {
                        "assay": assay["assay"],
                        "bin": name,
                        "cluster": int(assay["cluster"]),
                        "effect": float(value) - float(empty),
                        "mean_identity": float(np.mean(identities)),
                        "max_lcs": float(max(overlaps)),
                    }
                )
        clusters = {cell["cluster"] for cell in cells}
        summary = {
            "cells": len(cells),
            "clusters": len(clusters),
            "unit": "wild-type family cluster",
            "effect": f"within-assay Spearman under a bin minus the same under {H.NO_CONTEXT}",
        }
        if len(clusters) < MINIMUM_BOOTSTRAP_UNITS or len(cells) < 3:
            out[record["arm"]] = {
                **summary,
                "status": "unresolved_thin_support",
                "reason": "fewer clusters than the independence floor",
            }
            continue
        effect = np.asarray([cell["effect"] for cell in cells])
        identity = np.asarray([cell["mean_identity"] for cell in cells])
        overlap = np.asarray([cell["max_lcs"] for cell in cells])
        groups = np.asarray([cell["cluster"] for cell in cells])
        left = paired_group_bootstrap(
            effect, identity, overlap, groups, rank_correlation,
            seed=H.BOOTSTRAP_SEED, n_bootstrap=H.BOOTSTRAP_DRAWS,
            derived_statistic=lambda first, second: first,
        )
        right = paired_group_bootstrap(
            effect, identity, overlap, groups, rank_correlation,
            seed=H.BOOTSTRAP_SEED, n_bootstrap=H.BOOTSTRAP_DRAWS,
            derived_statistic=lambda first, second: second,
        )
        out[record["arm"]] = {
            **summary,
            "status": "estimated",
            "effect_versus_context_identity": {
                "spearman": left["derived_score"],
                "interval": left["derived_ci95"],
                "reads": "negative means the effect is more damaging the closer the context",
            },
            "effect_versus_longest_common_substring": {
                "spearman": right["derived_score"],
                "interval": right["derived_ci95"],
                "reads": "negative means the effect is more damaging the longer the shared run",
            },
            "identity_minus_lcs_association": {
                "difference": left["difference"],
                "interval": left["difference_ci95"],
            },
            "bootstrap": {
                "draws": left["n_bootstrap"],
                "seed": H.BOOTSTRAP_SEED,
                "groups": left["n_groups"],
                "method": "paired group bootstrap over family clusters",
            },
            "candidate_hypotheses": {
                "copying_or_prefix_bias": (
                    "predicts a monotone association with identity and with the longest "
                    "shared run, and the largest effect at the verbatim ceiling"
                ),
                "general_prefix_cost": (
                    "predicts an effect independent of identity and present in the "
                    "matched-unrelated condition; read that from the matched-unrelated "
                    "column of the no-context panel, not from here"
                ),
            },
        }
    return out


def protocol_comparison(records: list[dict], priors: list[Path]) -> dict:
    """Measure what a numerics change did, on the assays both protocols scored.

    When the scoring protocol changes, the numbers move for two reasons at once --
    the arithmetic and the support -- and a record that does not separate them
    invites the reader to attribute the whole move to whichever one is being
    discussed. So the overlap is compared directly: for every assay and condition
    both protocols scored, the difference in the per-assay rank correlation and in
    the wild-type summed log likelihood. A change in a headline larger than what
    this measures cannot be the arithmetic.
    """

    current = {record["arm"]: {row["assay"]: row for row in record["assays"]} for record in records}
    out = []
    for path in priors:
        prior = json.loads(Path(path).read_text(encoding="utf-8"))
        arm = prior["arm"]
        if arm not in current:
            continue
        theirs = {row["assay"]: row for row in prior["assays"]}
        shared = sorted(set(theirs) & set(current[arm]))
        spearman, wild = [], []
        for assay in shared:
            mine, other = current[arm][assay], theirs[assay]
            for condition, value in other["spearman"].items():
                observed = mine["spearman"].get(condition)
                if value is not None and observed is not None:
                    spearman.append(abs(float(value) - float(observed)))
            for condition, value in other["wt_log_likelihood"].items():
                observed = mine["wt_log_likelihood"].get(condition)
                if observed is not None:
                    wild.append(abs(float(value) - float(observed)))
        out.append(
            {
                "arm": arm,
                "prior_record": str(path),
                "prior_rows_per_forward": prior.get("rows_per_forward", prior.get("batch_size")),
                "current_rows_per_forward": H.SCORING_ROWS_PER_FORWARD,
                "prior_assays": prior["assays_scored"],
                "shared_assays": len(shared),
                "compared_condition_cells": len(spearman),
                "max_per_assay_spearman_difference": max(spearman) if spearman else None,
                "mean_per_assay_spearman_difference": (
                    float(np.mean(spearman)) if spearman else None
                ),
                "max_wild_type_log_likelihood_difference_nats": max(wild) if wild else None,
                "reads": (
                    "an upper bound on how much of any change between the two readings the "
                    "arithmetic can account for; a larger move is support, not protocol"
                ),
            }
        )
    return {
        "note": (
            "the prior records are not pooled with these; the analysis refuses that. They "
            "are read only to bound the size of the protocol change."
        ),
        "arms": out,
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
    protocols = {record["arm"]: record.get("rows_per_forward") for record in records}
    if set(protocols.values()) != {H.SCORING_ROWS_PER_FORWARD}:
        raise SystemExit(
            f"these records were not all scored at {H.SCORING_ROWS_PER_FORWARD} row(s) per "
            f"forward: {protocols}. Scores taken at different batch extents are different "
            "arithmetic and are never pooled or compared across arms; rescore the records "
            "that differ."
        )
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
    # The reading referent first, then the others: the order the artefact is read in.
    ordered = [H.READING_REFERENT] + [r for r in H.REFERENTS if r != H.READING_REFERENT]
    for referent in ordered:
        panels.append(
            panel_statistics(
                records,
                referent=referent,
                bins=columns_for(referent, admitted),
                targets=None,
                label=H.PRIMARY_PANEL,
            )
            | {"is_reading_referent": referent == H.READING_REFERENT}
        )
        if support["balanced_clusters"] >= H.GROUP_FLOOR:
            panels.append(
                panel_statistics(
                    records,
                    referent=referent,
                    bins=columns_for(referent, admitted),
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
            "reading_referent": H.READING_REFERENT,
            "protocol_comparison": (
                protocol_comparison(records, args.prior_protocol_scores)
                if args.prior_protocol_scores
                else None
            ),
            "mechanism": mechanism(records),
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
    parser.add_argument(
        "--prior-protocol-scores",
        type=Path,
        nargs="*",
        default=[],
        help="records from a superseded numerics protocol, read only to bound how much "
        "of any change between the two readings the arithmetic can account for",
    )
    parser.add_argument("--budget", type=int, default=H.POSITION_BUDGET)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=H.SCORING_ROWS_PER_FORWARD,
        help="rows per forward; the declaration fixes it at one and refuses any other value",
    )
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--assay-limit", type=int, default=0)
    args = parser.parse_args()
    if args.budget != H.POSITION_BUDGET:
        raise SystemExit(
            f"the declaration fixes the position budget at {H.POSITION_BUDGET}; a run at "
            f"{args.budget} is a different experiment"
        )
    if args.batch_size != H.SCORING_ROWS_PER_FORWARD:
        raise SystemExit(
            f"the declaration fixes scoring at {H.SCORING_ROWS_PER_FORWARD} row(s) per "
            f"forward and was given {args.batch_size}. Scores taken at a different batch "
            "extent are different arithmetic: the mutant-minus-wild differences this "
            "endpoint is built from moved by 1.5e-3 and 4.6e-3 nats at eight rows per "
            "forward on the two larger ProGen2 rungs."
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
