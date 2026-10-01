#!/usr/bin/env python3
"""Stage-01 design context information for a joint checkpoint, native rendering.

A joint decoder is not an arm: ``joint_mode_qualification.py`` keeps it out of
``arms.py``. This wrapper reuses that family's rendering and scorer, and stage
01's cohort construction (8 blocks × 200, 4000-record held-out unigram), so the
sidecars are 01-compatible for ``context_information_bootstrap.py``.

It is a new campaign, not a reinterpretation of stage 21's 64/128-record cells.
Reversed/naive controls are omitted: identification is the declared-mode
estimand. A lower bound that includes zero is a completed measurement.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_STAGE_DIR = str(Path(__file__).resolve().parent)
if _STAGE_DIR not in sys.path:
    sys.path.insert(0, _STAGE_DIR)

from src.capability.models import joint_modes  # noqa: E402
from src.capability.core.arms import DEFAULT_CORPUS_DRAW_SEED, MODEL_ROOT, REPO  # noqa: E402
from src.capability.core.budget import (  # noqa: E402
    SCREENING_CONTEXT_INFORMATION_NATS,
    SparseCounts,
    write_power_records,
)
from src.capability.core.io import write_json  # noqa: E402
from src.capability.core.measurement_support import LAPLACE_SMOOTHING, UNIGRAM_ESTIMATORS, disjoint_unigram_cross_entropy_nats, held_out_cohort


def _load_stage(filename: str):
    path = Path(__file__).resolve().parent / filename
    name = path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STAGE01 = _load_stage("cohort_power.py")
STAGE21 = _load_stage("joint_mode_qualification.py")

SCHEMA_VERSION = "r2_transfer_joint_context_information_v1"
DEFAULT_OUT = REPO / "results/R5/joint_context_information"

#: Directory name under MODEL_ROOT and the joint_modes family. Not ArmSpecs.
JOINT_TARGETS: dict[str, tuple[str, str]] = {
    "galactica-125m": ("galactica-125m", "galactica"),
    "galactica-1.3b": ("galactica-1.3b", "galactica"),
    "galactica-6.7b": ("galactica-6.7b", "galactica"),
    "galactica-30b": ("galactica-30b", "galactica"),
    "instructprotein": ("InstructProtein", "instructprotein"),
    "llama-2-7b": ("Llama-2-7b-hf", "prollama"),
    "prollama-stage-1": ("ProLLaMA_Stage_1", "prollama"),
    "prollama": ("ProLLaMA", "prollama"),
}


def resolve_checkpoint(args: argparse.Namespace) -> tuple[Path, str, str]:
    """``(checkpoint, rendering, arm_name)`` from --joint-target or --checkpoint."""

    if args.joint_target:
        if args.joint_target not in JOINT_TARGETS:
            raise ValueError(
                f"unknown --joint-target {args.joint_target!r}; "
                f"declared {sorted(JOINT_TARGETS)}"
            )
        directory, rendering = JOINT_TARGETS[args.joint_target]
        if args.rendering and args.rendering != rendering:
            raise ValueError(
                f"--joint-target {args.joint_target} takes rendering {rendering!r}, "
                f"not {args.rendering!r}"
            )
        checkpoint = Path(args.checkpoint) if args.checkpoint else MODEL_ROOT / directory
        stem = args.joint_target
    else:
        if args.checkpoint is None or args.rendering is None:
            raise ValueError("pass --joint-target, or both --checkpoint and --rendering")
        checkpoint = Path(args.checkpoint)
        rendering = args.rendering
        stem = checkpoint.name
    arm_name = args.arm_name or f"{stem}_{args.joint_mode}"
    if "::" in arm_name:
        raise ValueError(f"arm name {arm_name!r} collides with the sidecar key separator")
    return checkpoint, rendering, arm_name


def score_text(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    vocab_size: int,
    records: list[str],
    arm_name: str,
):
    nll_sums: list[float] = []
    tokens: list[int] = []
    blocks: list[np.ndarray] = []
    for document in records:
        token_ids = STAGE21.text_token_ids(
            tokenizer, document, max_tokens=args.max_len
        )
        if len(token_ids) < 2:
            raise ValueError("a text record tokenised to fewer than two tokens")
        positions = list(range(1, len(token_ids)))
        nll, _ = STAGE21.score_positions(
            model, token_ids, positions, device=args.device
        )
        nll_sums.append(float(nll.sum()))
        tokens.append(len(positions))
        array = np.asarray(token_ids[1:], dtype=np.int64)
        if array.size and (int(array.min()) < 0 or int(array.max()) >= vocab_size):
            raise ValueError("a token id fell outside the checkpoint vocabulary")
        blocks.append(array)
    statistics = STAGE21.record_statistics(
        arm_name,
        support_size=vocab_size,
        clean_nll_sum=nll_sums,
        token_count=tokens,
        n_symbols=tokens,
        targets=SparseCounts.from_records(blocks),
    )
    return statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-target", choices=tuple(JOINT_TARGETS), default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--rendering",
        choices=joint_modes.RENDERING_NAMES,
        default=None,
    )
    parser.add_argument("--joint-mode", choices=("protein", "text"), required=True)
    parser.add_argument("--arm-name", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--n-seq", type=int, default=200)
    parser.add_argument("--res-min", type=int, default=64)
    parser.add_argument("--res-max", type=int, default=246)
    parser.add_argument("--min-chars", type=int, default=800)
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--cohort-name", default=None)
    parser.add_argument("--cohort-skip", type=int, default=0)
    parser.add_argument("--cohort-pool-size", type=int, default=4000)
    parser.add_argument("--cohort-draw-seed", type=int, default=DEFAULT_CORPUS_DRAW_SEED)
    parser.add_argument("--unigram-estimator", default="disjoint", choices=list(UNIGRAM_ESTIMATORS))
    parser.add_argument("--unigram-reference-size", type=int, default=4000)
    parser.add_argument("--protein-context", default=None)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--record-statistics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.unigram_estimator != "disjoint":
        raise ValueError("this wrapper writes 01-style held-out sidecars; use disjoint")

    checkpoint, rendering, arm_name = resolve_checkpoint(args)
    args.kind = args.joint_mode
    if args.cohort_name is None:
        args.cohort_name = (
            f"swissprot_{arm_name}" if args.kind == "protein" else f"openwebtext_{arm_name}"
        )
    args.with_ec = False
    args.skip_truncation = True
    args.truncation_contexts = [1]
    args.token_shuffle_control_seed = None

    declaration = joint_modes.rendering(rendering)
    resolved, tokenizer = STAGE21.load_tokenizer(checkpoint)
    tokenisation = joint_modes.resolve(tokenizer, declaration)
    model, checkpoint_facts = STAGE21.load_model(
        resolved, tokenizer, device=args.device, dtype=args.dtype
    )
    checkpoint_facts["requested_path"] = str(checkpoint)

    cohort, sampling = STAGE01.build_cohort(args)
    digest = cohort.digest
    candidate = STAGE01.draw_records(
        args,
        args.unigram_reference_size,
        int(sampling["records_consumed"]),
        f"{args.cohort_name}_unigram_reference",
    )
    reference, reference_overlap = held_out_cohort(candidate, cohort)
    write_json(
        args.out / f"cohort_{cohort.name}_{digest[:12]}.json",
        {
            "schema_version": SCHEMA_VERSION,
            "artifact": "frozen_cohort",
            "cohort_digest": digest,
            "cohort_name": cohort.name,
            "cohort_kind": cohort.kind,
            "min_symbols": cohort.min_symbols,
            "max_symbols": cohort.max_symbols,
            "n_records": len(cohort),
            "records": cohort.records,
            "metadata": cohort.metadata,
        },
    )
    reference_path = STAGE01.write_reference_records(args.out, cohort, reference)

    if args.kind == "protein":
        _, declared_statistics = STAGE21.score_protein_records(
            model,
            tokenisation,
            cohort.records,
            device=args.device,
            context=args.protein_context,
            variant=joint_modes.DECLARED,
            max_tokens=max(args.max_len, 512),
            condition=arm_name,
        )
        assert declared_statistics is not None
        reference_counts, reference_per_record = STAGE21.scored_target_records(
            tokenisation, reference.records, context=args.protein_context
        )
        scored_counts = STAGE21.scored_target_counts(
            tokenisation, cohort.records, context=args.protein_context
        )
        statistics = replace(declared_statistics, arm=arm_name, reference_counts=reference_per_record)
        unit = (
            "nats per scored residue"
            if declaration.symbol_unit == joint_modes.RESIDUE_UNIT
            else "nats per scored token"
        )
        clean = float(statistics.clean_nll_sum.sum()) / float(statistics.n_symbols.sum())
        if declaration.symbol_unit != joint_modes.RESIDUE_UNIT:
            clean = float(statistics.clean_nll_sum.sum()) / float(statistics.token_count.sum())
    else:
        vocab_size = int(checkpoint_facts["vocab_size"])
        statistics = score_text(
            args, model, tokenizer, vocab_size, cohort.records, arm_name
        )
        reference_counts, reference_per_record = STAGE21.text_target_records(
            tokenizer,
            reference.records,
            vocab_size=vocab_size,
            max_tokens=args.max_len,
        )
        scored_counts = STAGE21.text_target_counts(
            tokenizer,
            cohort.records,
            vocab_size=vocab_size,
            max_tokens=args.max_len,
        )
        statistics = replace(statistics, reference_counts=reference_per_record)
        unit = "nats per scored token"
        clean = float(statistics.clean_nll_sum.sum()) / float(statistics.token_count.sum())

    baseline = float(disjoint_unigram_cross_entropy_nats(reference_counts, scored_counts))
    context_information = baseline - clean
    artifact, stem = STAGE01.artifact_names(cohort.name, digest, None)
    destination = args.out / f"{stem}.json"
    sufficient_statistics = None
    if args.record_statistics:
        sufficient_statistics = write_power_records(
            destination.with_suffix(".records.npz"),
            {arm_name: statistics},
            cohort_digest=digest,
            reference_digest=reference.digest,
            smoothing=float(LAPLACE_SMOOTHING),
            seeds={"cohort_draw": int(args.cohort_draw_seed)},
            max_len=int(args.max_len),
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact": artifact,
        "not_panel_admission": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_digest": digest,
        "arm_name": arm_name,
        "rendering": rendering,
        "joint_mode": args.kind,
        "checkpoint": checkpoint_facts,
        "cohort": {
            "name": cohort.name,
            "kind": cohort.kind,
            "n_records": len(cohort),
        },
        # Always present, and named the same way cohort_power.py and
        # component_swap.py name it. ``cohort_sampling`` below carries a
        # ``draw_seed`` only on the permuted branch, and the sufficient-statistics
        # sidecar exists only under --record-statistics, so neither is where a
        # reader can go to learn which draw produced this report: a
        # ``--cohort-draw-seed 0`` run recorded no seed at all.
        "seeds": {"cohort_draw": int(args.cohort_draw_seed)},
        "cohort_sampling": sampling,
        "unigram_baseline": {
            "estimator": "disjoint",
            "reference_digest": reference.digest,
            "reference_overlap_removed": reference_overlap,
            "cross_entropy_nats": baseline,
        },
        "arms": {
            arm_name: {
                "arm": arm_name,
                "clean_ce_nats": clean,
                "unigram_entropy_used_for_verdict_nats": baseline,
                "context_information_nats": context_information,
                "context_information_unit": unit,
                "power_verdict": (
                    "PASS"
                    if context_information >= SCREENING_CONTEXT_INFORMATION_NATS
                    else "FAIL"
                ),
                "identification_evaluable_here": False,
                "note": (
                    "point screen only; identification is the displacement-corrected "
                    "interval from context_information_bootstrap.py. A FAIL here "
                    "is a completed measurement, not a reason to drop the arm"
                ),
            }
        },
        "sufficient_statistics": sufficient_statistics,
        "settings": {
            "device": args.device,
            "dtype": args.dtype,
            "max_len": int(args.max_len),
        },
    }
    write_json(destination, payload)
    print(
        f"{arm_name:24s} unigram {baseline:7.4f}  clean {clean:7.4f}  "
        f"context_info {context_information:+8.4f} {unit}"
    )
    print(f"wrote {destination}")
    if sufficient_statistics is not None:
        print(f"wrote {args.out / sufficient_statistics['path']}")
    if reference_path is not None:
        print(f"wrote {reference_path}  (pass to 41 as --reference-json)")


if __name__ == "__main__":
    main()
