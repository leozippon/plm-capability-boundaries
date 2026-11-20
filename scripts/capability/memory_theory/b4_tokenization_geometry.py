#!/usr/bin/env python3
"""Track B4 GPU Stage: Controlled Tokenization Geometry Probe (AA20 vs BPE vs SaProt 3Di/AA-only).

Evaluates gradient effective rank ratio r_eff/d and <25% remote-homolog vs unrelated-matched-SS
KV constructive alignment on the identical CATH-3D domain cohort across tokenization regimes.
Compatible with h200/run_external_baseline_h200.sh (--device, --out, --expect).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("TRANSFER_DATA_ROOT") or REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.memory_theory.gradient_probes import measure_residual_gradient_rank_and_snr
from src.capability.memory_theory.io_utils import load_json_file, write_json_with_sha256
from src.capability.memory_theory.kv_interference import evaluate_kv_interference_pairs
from src.capability.memory_theory.model_hooks import format_sequence_for_arm, load_model_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, help="e.g. progen2-medium, protgpt2, esm2-650m, saprot-650m.")
    parser.add_argument(
        "--tokenization-mode",
        default="native",
        choices=["native", "saprot_3di", "saprot_aa_only"],
        help="For SaProt, selects Foldseek-3Di+AA (structure-privileged) vs '#'-masked AA-only (sequence-only).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-name", default="b4_tokenization_geometry.json")
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=DATA_ROOT / "results/extensions/memory_theory_20261120",
    )
    parser.add_argument("--max-grad-seqs", type=int, default=128)
    parser.add_argument("--max-pairs-per-stratum", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-sampled-positions", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20261120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(
        args.arm,
        device=args.device,
        dtype=args.dtype,
        model_dir=args.model_dir,
        modality="protein",
    )
    cath = load_json_file(args.cohort_dir / "cath_3d_kv_cohort.json")

    saprot_mode = "aa_only" if args.tokenization_mode == "saprot_aa_only" else "3di"
    is_structure_privileged = args.arm.lower().startswith("saprot") and saprot_mode == "3di"

    seq_map: dict[str, str] = {}
    token_ratios: list[float] = []
    for dom_id, d in sorted(cath["domains"].items()):
        formatted = format_sequence_for_arm(
            d["aa_seq"],
            args.arm,
            saprot_seq=d["saprot_seq"],
            saprot_mode=saprot_mode,
        )
        seq_map[dom_id] = formatted
        if len(token_ratios) < 64:
            ids = bundle.tokenizer.encode(formatted, add_special_tokens=False)
            token_ratios.append(float(len(ids) / max(len(d["aa_seq"]), 1)))

    grad_seqs = [seq_map[dom_id] for dom_id in sorted(seq_map.keys())[: args.max_grad_seqs]]
    grad_geo = measure_residual_gradient_rank_and_snr(
        bundle,
        grad_seqs,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    pairs_by_stratum = {
        stratum: [(p["domain_a"], p["domain_b"]) for p in plist[: args.max_pairs_per_stratum]]
        for stratum, plist in cath["pairs"].items()
    }
    kv_geo = evaluate_kv_interference_pairs(
        bundle,
        seq_map,
        pairs_by_stratum,
        max_tokens=args.max_tokens,
        max_sampled_positions=args.max_sampled_positions,
        seed=args.seed,
    )

    payload: dict[str, Any] = {
        "schema": "memory_theory_b4_tokenization_geometry_v1",
        "config": {
            "arm": args.arm,
            "tokenization_mode": args.tokenization_mode,
            "structure_privileged": is_structure_privileged,
            "vocab_size": bundle.vocab_size,
            "mean_tokens_per_residue": float(sum(token_ratios) / max(len(token_ratios), 1)),
            "dtype": args.dtype,
            "max_grad_seqs": args.max_grad_seqs,
            "max_pairs_per_stratum": args.max_pairs_per_stratum,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "model_dir": str(bundle.model_dir),
        },
        "gradient_geometry": grad_geo,
        "kv_interference_geometry": kv_geo,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / args.out_name
    digest = write_json_with_sha256(out_file, payload)
    print(f"Wrote {out_file} (sha256={digest})")


if __name__ == "__main__":
    main()
