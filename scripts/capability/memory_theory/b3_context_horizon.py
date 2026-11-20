#!/usr/bin/env python3
"""Track B3 GPU Stage: Prefix Mutual Information Horizon Delta H(L_ctx).

Compatible with h200/run_external_baseline_h200.sh (--device, --out, --expect).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("TRANSFER_DATA_ROOT") or REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.memory_theory.horizon import DEFAULT_CONTEXT_GRID, evaluate_prefix_horizon
from src.capability.memory_theory.io_utils import load_json_file, write_json_with_sha256
from src.capability.memory_theory.model_hooks import format_sequence_for_arm, load_model_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--modality", choices=["protein", "text"], default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-name", default="b3_context_horizon.json")
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=DATA_ROOT / "results/extensions/memory_theory_20261120",
    )
    parser.add_argument("--max-seqs", type=int, default=256)
    parser.add_argument(
        "--context-grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_CONTEXT_GRID),
        help="Comma-separated prefix context lengths L_ctx.",
    )
    parser.add_argument("--seed", type=int, default=20261120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = tuple(int(x.strip()) for x in args.context_grid.split(",") if x.strip())
    bundle = load_model_bundle(
        args.arm,
        device=args.device,
        dtype=args.dtype,
        model_dir=args.model_dir,
        modality=args.modality,
    )

    cohort = load_json_file(args.cohort_dir / "text_kv_and_horizon_cohort.json")
    if bundle.modality == "text":
        sequences = [a["text"] for a in cohort["wikitext_articles"][: args.max_seqs]]
    else:
        raw_seqs = [p["aa_seq"] for p in cohort["long_proteins"][: args.max_seqs]]
        # For BPE protein models (ProtGPT2, ~2.8 AA/tok), 512 AA yields ~180 tokens; automatically cap grid
        # if needed or filter sequences so the stage never fails silently
        sequences = [format_sequence_for_arm(s, args.arm) for s in raw_seqs]
        if args.arm.lower().startswith("protgpt2"):
            # Ensure max grid fits BPE token length of long Swiss-Prot chains (up to 1024 AA -> ~360 tokens)
            max_bpe_len = min(len(bundle.tokenizer.encode(s, add_special_tokens=False)) for s in sequences[: min(8, len(sequences))])
            valid_grid = tuple(g for g in grid if g < max_bpe_len)
            if len(valid_grid) >= 3:
                grid = valid_grid

    result = evaluate_prefix_horizon(
        bundle,
        sequences,
        context_grid=grid,
        anchor_offsets=(max(grid),),
        seed=args.seed,
    )

    payload = {
        "schema": "memory_theory_b3_context_horizon_v1",
        "config": {
            "arm": args.arm,
            "modality": bundle.modality,
            "is_masked": bundle.is_masked,
            "dtype": args.dtype,
            "max_seqs": args.max_seqs,
            "context_grid": list(grid),
            "seed": args.seed,
            "model_dir": str(bundle.model_dir),
        },
        "horizon": result,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / args.out_name
    digest = write_json_with_sha256(out_file, payload)
    print(f"Wrote {out_file} (sha256={digest})")


if __name__ == "__main__":
    main()
