#!/usr/bin/env python3
"""R5 generative expression runner for general PLMs:
1. Modification-extent ladder (``k = 1, 2, 5, 10, 20, 40``) on the frozen 16
   high-confidence backbones (``ladder_backbones.jsonl``) for all pMLM,
   diffusion, and new causal arms.
2. Declared ``full_length_matched`` 100%-masked cosine-schedule rung AND
   800-sequence length-matched de novo generation for absorbing-state discrete
   diffusion arms (``dplm-150m``, ``dplm-650m``, ``dplm-3b``).
3. Unconditional free-length 800-sequence generation for newer causal arms
   (``protgpt3-112m/10b``, ``progen3-*``, ``dayhoff-*``).

Outputs both an atomic JSON summary AND FASTA/JSONL sequence manifests ready
for downstream ESMFold2 + Pfam HMMER + UniRef50 DIAMOND evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("TRANSFER_DATA_ROOT") or ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.general_plm import list_arms, load_general_plm
from src.capability.general_plm.eval_utils import atomic_write_json, sha256_file
from src.capability.ladder import design as ladder_design

SCHEMA = "general_plm_r5_infill_and_diffusion_gen_v1"
CONDITION_DIFFUSION_LENGTH_SUPPLIED_FULL = "diffusion_length_supplied_full"
DEFAULT_EXTENTS = (1, 2, 5, 10, 20, 40)


def _derive_seed(base_seed: int, *parts: Any) -> int:
    key = "|".join([str(base_seed), *(str(p) for p in parts)])
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def _centered_window_start(length: int, extent: int) -> int:
    """Strictly interior centered window start matching ``ladder/design.py``."""
    start = max(1, (length - extent) // 2)
    if start + extent >= length:
        start = max(1, length - extent - 1)
    return start


def _sample_empirical_lengths(
    n_draws: int, *, seed: int, min_len: int = 40, max_len: int = 400
) -> list[int]:
    """Deterministic length distribution matched to R4/Swiss-Prot natural chains (40-400 aa)."""
    r4_path = DATA_ROOT / "results/R5/context_homologue_20260826/cohort.json"
    rng = np.random.default_rng(int(seed))
    if r4_path.is_file():
        records = json.loads(r4_path.read_text(encoding="utf-8"))["protein"]["records"]
        pool = [len(s) for s in records if min_len <= len(s) <= max_len]
        if len(pool) >= n_draws:
            chosen = rng.choice(pool, size=n_draws, replace=False)
            return [int(x) for x in chosen]
    return [int(x) for x in rng.integers(min_len, max_len + 1, size=n_draws)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", required=True, choices=list_arms())
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--saprot-mode", default="seq_only", choices=("seq_only", "foldseek_3di"))
    parser.add_argument("--extents", type=int, nargs="+", default=list(DEFAULT_EXTENTS))
    parser.add_argument("--draws-per-cell", type=int, default=ladder_design.DRAWS_PER_CELL)
    parser.add_argument("--temperature", type=float, default=ladder_design.TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=ladder_design.TOP_P)
    parser.add_argument("--seed", type=int, default=ladder_design.CAMPAIGN_SEED)
    parser.add_argument("--score-batch-size", type=int, default=32)
    parser.add_argument("--diffusion-steps", type=int, default=64)
    parser.add_argument(
        "--diffusion-sampling-mode",
        default="max_confidence",
        choices=("max_confidence", "uniform_random", "gumbel_annealed"),
        help=(
            "Unmasking-order rule for 100%-masked diffusion sampling: "
            "'uniform_random' (canonical DPLM reverse process, prevents homopolymer collapse), "
            "'gumbel_annealed' (MaskGIT annealed Gumbel), or 'max_confidence' (legacy default)."
        ),
    )
    parser.add_argument("--denovo-draws", type=int, default=800)
    parser.add_argument("--max-residues", type=int, default=ladder_design.FULL_GENERATION_MAX_RESIDUES)
    parser.add_argument("--skip-ladder", action="store_true")
    parser.add_argument("--skip-denovo", action="store_true")
    parser.add_argument("--skip-pll-scoring", action="store_true")
    parser.add_argument("--max-backbones", type=int, default=0)
    parser.add_argument(
        "--backbones",
        type=Path,
        default=DATA_ROOT / "results/audit_20261011_h200/ladder_original/ladder_backbones.jsonl",
    )
    parser.add_argument("--expect-file", default="")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    adapter = load_general_plm(
        args.arm,
        device=args.device,
        dtype=args.dtype,
        model_root=args.model_root,
        saprot_mode=args.saprot_mode,
    )

    t0 = time.monotonic()
    ladder_records: list[dict[str, Any]] = []
    fasta_entries: list[tuple[str, str]] = []

    if not args.skip_ladder:
        backbones = [
            json.loads(line)
            for line in args.backbones.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.max_backbones > 0:
            backbones = backbones[: args.max_backbones]

        condition = (
            ladder_design.CONDITION_CAUSAL_PREFIX
            if adapter.spec.is_causal
            else ladder_design.CONDITION_BIDIRECTIONAL_INFILL
        )
        for bb_idx, bb in enumerate(backbones, 1):
            bb_id = bb["backbone_id"]
            parent = bb["sequence"]
            if len(parent) > adapter.spec.max_residues:
                continue
            parent_score = (
                None
                if args.skip_pll_scoring
                else float(adapter.exact_pll(parent, batch_size=args.score_batch_size).nll_sum)
            )
            for extent in args.extents:
                if extent >= len(parent) - 2:
                    continue
                start = _centered_window_start(len(parent), extent)
                for draw_i in range(args.draws_per_cell):
                    draw_seed = _derive_seed(args.seed, args.arm, bb_id, f"k{extent}", draw_i)
                    res = adapter.infill_window(
                        parent,
                        start=start,
                        extent=extent,
                        seed=draw_seed,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    seq = res.get("sequence")
                    seq_nll = (
                        float(adapter.exact_pll(seq, batch_size=args.score_batch_size).nll_sum)
                        if (seq and not args.skip_pll_scoring)
                        else None
                    )
                    rec_id = f"{args.arm}__{bb_id}__k{extent}__d{draw_i}"
                    if seq:
                        fasta_entries.append((rec_id, seq))
                    ladder_records.append(
                        {
                            "record_id": rec_id,
                            "backbone_id": bb_id,
                            "rung": f"k{extent}",
                            "extent": int(extent),
                            "condition": condition,
                            "window_start": start,
                            "draw_index": draw_i,
                            "seed": draw_seed,
                            "status": res["status"],
                            "sequence": seq,
                            "length": len(seq) if seq else 0,
                            "nll_sum_nats": seq_nll,
                            "parent_nll_sum_nats": parent_score,
                        }
                    )

            # Full-length 100%-masked rung strictly for diffusion arms (DPLM ladder).
            if adapter.spec.supports_r5_diffusion_full:
                for draw_i in range(args.draws_per_cell):
                    draw_seed = _derive_seed(args.seed, args.arm, bb_id, "full_length_matched", draw_i)
                    res = adapter.sample_diffusion_full(
                        len(parent),
                        seed=draw_seed,
                        steps=args.diffusion_steps,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        sampling_mode=args.diffusion_sampling_mode,
                    )
                    seq = res["sequence"]
                    seq_nll = (
                        float(adapter.exact_pll(seq, batch_size=args.score_batch_size).nll_sum)
                        if not args.skip_pll_scoring
                        else None
                    )
                    rec_id = f"{args.arm}__{bb_id}__full_length_matched__d{draw_i}"
                    fasta_entries.append((rec_id, seq))
                    ladder_records.append(
                        {
                            "record_id": rec_id,
                            "backbone_id": bb_id,
                            "rung": "full_length_matched",
                            "extent": len(parent),
                            "condition": CONDITION_DIFFUSION_LENGTH_SUPPLIED_FULL,
                            "window_start": 0,
                            "draw_index": draw_i,
                            "seed": draw_seed,
                            "status": res["status"],
                            "sequence": seq,
                            "length": len(seq),
                            "nll_sum_nats": seq_nll,
                            "parent_nll_sum_nats": parent_score,
                        }
                    )
            print(f"[R5-ladder:{args.arm}] {bb_idx}/{len(backbones)} backbones", flush=True)

    denovo_records: list[dict[str, Any]] = []
    denovo_mode = "none"
    if not args.skip_denovo and args.denovo_draws > 0:
        if adapter.spec.supports_r5_diffusion_full:
            denovo_mode = "diffusion_length_matched_denovo"
            lengths = _sample_empirical_lengths(
                args.denovo_draws, seed=args.seed + 77, min_len=40, max_len=args.max_residues
            )
            for idx, target_len in enumerate(lengths):
                d_seed = _derive_seed(args.seed, args.arm, "denovo_diff", idx)
                res = adapter.sample_diffusion_full(
                    target_len,
                    seed=d_seed,
                    steps=args.diffusion_steps,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    sampling_mode=args.diffusion_sampling_mode,
                )
                rec_id = f"{args.arm}__denovo__{idx:04d}"
                fasta_entries.append((rec_id, res["sequence"]))
                denovo_records.append({"record_id": rec_id, "draw_index": idx, "seed": d_seed, **res})
                if (idx + 1) % 100 == 0 or (idx + 1) == len(lengths):
                    print(f"[R5-denovo:{args.arm}] {idx + 1}/{len(lengths)} diffusion draws", flush=True)
        elif adapter.spec.supports_r5_causal_free:
            denovo_mode = "causal_free_length_denovo"
            raw_draws = adapter.generate_causal_free(
                args.denovo_draws,
                seed=_derive_seed(args.seed, args.arm, "denovo_causal"),
                max_residues=args.max_residues,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            for d in raw_draws:
                rec_id = f"{args.arm}__denovo__{d['draw_index']:04d}"
                if d["sequence"]:
                    fasta_entries.append((rec_id, d["sequence"]))
                denovo_records.append({"record_id": rec_id, **d})
            print(f"[R5-denovo:{args.arm}] {len(denovo_records)} causal free-length draws", flush=True)

    # Emit FASTA + JSONL sequence manifests for downstream ESMFold2 / Pfam / DIAMOND.
    fasta_path = args.out / f"r5_sequences_{args.arm}.fasta"
    with fasta_path.open("w", encoding="utf-8") as fh:
        for header, seq in fasta_entries:
            fh.write(f">{header}\n{seq}\n")

    jsonl_path = args.out / f"r5_draws_{args.arm}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in ladder_records + denovo_records:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    elapsed = time.monotonic() - t0
    out_name = args.expect_file or f"r5_infill_and_gen_{args.arm}.json"
    atomic_write_json(
        args.out / out_name,
        {
            "schema": SCHEMA,
            "roster": "general_plm_20261120_exploratory",
            "arm": args.arm,
            "backbones_sha256": sha256_file(args.backbones) if args.backbones.is_file() else None,
            "provenance": adapter.provenance(),
            "ladder_summary": {
                "n_draws": len(ladder_records),
                "n_filled": int(sum(1 for r in ladder_records if r["status"] == "filled")),
                "rungs": sorted({r["rung"] for r in ladder_records}),
                "diffusion_sampling_mode": args.diffusion_sampling_mode,
            },
            "denovo_summary": {
                "mode": denovo_mode,
                "diffusion_sampling_mode": (
                    args.diffusion_sampling_mode
                    if denovo_mode == "diffusion_length_matched_denovo"
                    else None
                ),
                "n_draws": len(denovo_records),
                "mean_length": (
                    float(np.mean([r["length"] for r in denovo_records]))
                    if denovo_records
                    else None
                ),
            },
            "sequence_manifest_fasta": fasta_path.name,
            "draws_jsonl": jsonl_path.name,
            "elapsed_seconds": elapsed,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"[R5:{args.arm}] wrote {args.out / out_name}", flush=True)


if __name__ == "__main__":
    main()
