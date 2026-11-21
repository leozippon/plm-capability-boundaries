#!/usr/bin/env python3
"""Track B2 GPU Stage: Residual Gradient Effective Rank, Minibatch Gradient SNR, and KV Interference.

Compatible with h200/run_external_baseline_h200.sh:
  run_wrapped_external_stage passes: --device cuda:<G> --out <OUT_DIR> [extra args...]
and verifies <OUT_DIR>/<expect.json>.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("TRANSFER_DATA_ROOT") or REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.memory_theory.capacity_scaling import classify_conservation_tier, shannon_entropy_bits
from src.capability.memory_theory.gradient_probes import measure_residual_gradient_rank_and_snr
from src.capability.memory_theory.io_utils import load_json_file, write_json_with_sha256
from src.capability.memory_theory.kv_interference import evaluate_kv_interference_pairs
from src.capability.memory_theory.model_hooks import format_sequence_for_arm, load_model_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, help="Model arm name (e.g. gpt2, progen2-small, protgpt2, esm2-650m).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--modality", choices=["protein", "text"], default=None)
    parser.add_argument("--saprot-mode", choices=["3di", "aa_only"], default="3di")
    parser.add_argument("--out", type=Path, required=True, help="Output directory (per run_external_baseline_h200.sh).")
    parser.add_argument("--out-name", default="b2_gradient_and_kv_geometry.json", help="Output JSON basename inside --out.")
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=DATA_ROOT / "results/extensions/memory_theory_20261120",
        help="Directory holding cath_3d_kv_cohort.json and text_kv_and_horizon_cohort.json.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DATA_ROOT,
        help="Repository root for family_conditionals / family_distribution inputs.",
    )
    parser.add_argument("--max-grad-seqs", type=int, default=128)
    parser.add_argument("--max-pairs-per-stratum", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-sampled-positions", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20261120)
    return parser.parse_args()


def _load_protein_gradient_batches(
    repo_root: Path,
    arm: str,
    max_seqs: int,
    saprot_mode: str,
    cath_cohort: dict[str, Any],
) -> dict[str, Any]:
    """Prepare protein sequences for gradient rank + conserved/variable + within/cross-family SNR."""
    fc_path = repo_root / "results/extensions/family_conditionals_20261107/conditional-inputs.json"
    fd_path = repo_root / "results/extensions/family_distribution_20261106/queue.json"
    is_saprot = arm.lower().startswith("saprot")

    if is_saprot and saprot_mode == "3di":
        # SaProt 3Di requires structures -> use CATH 3D cohort domains directly
        doms = list(cath_cohort["domains"].values())[:max_seqs]
        seqs = [format_sequence_for_arm(d["aa_seq"], arm, saprot_seq=d["saprot_seq"], saprot_mode="3di") for d in doms]
        # Group by CATH superfamily for within-family vs cross-family SNR
        by_h: dict[str, list[str]] = defaultdict(list)
        for d in cath_cohort["domains"].values():
            by_h[d["cath_superfamily"]].append(
                format_sequence_for_arm(d["aa_seq"], arm, saprot_seq=d["saprot_seq"], saprot_mode="3di")
            )
        within_fams = [v[: min(16, max_seqs)] for _, v in sorted(by_h.items()) if len(v) >= 4][:8]
        return {
            "primary_sequences": seqs,
            "conserved_sites": None,
            "variable_sites": None,
            "within_family_batches": within_fams,
            "cross_family_batch": seqs[: min(64, max_seqs)],
        }

    # Standard AA / BPE PLMs: use exact frozen family_conditionals + family_distribution cohorts
    conserved_seqs: list[str] = []
    conserved_pos: list[int] = []
    variable_seqs: list[str] = []
    variable_pos: list[int] = []

    if fc_path.is_file():
        ci = load_json_file(fc_path)
        for unit in ci["units"]:
            if unit["partition"] != "evaluation":
                continue
            for site in unit["sites"]:
                pos_key = str(site["query_position"])
                h_prof = shannon_entropy_bits(unit["baselines"]["profile_frequencies"][pos_key])
                tier = classify_conservation_tier(h_prof)
                ctx = site["contexts"][0]
                seq = format_sequence_for_arm(ctx["native"], arm, saprot_mode=saprot_mode)
                # Offset by prefix token count (+3 for ProteinGLM CLM, +1 for ProGen2/ESM/SaProt)
                low_arm = arm.lower()
                if low_arm.startswith("proteinglm") and low_arm.endswith("clm"):
                    offset = 3
                elif low_arm.startswith(("progen2", "esm2", "dplm", "esmc", "amplify", "saprot")):
                    offset = 1
                else:
                    offset = 0
                tok_pos = int(ctx["native_index"]) + offset
                # BPE (ProtGPT2) does not map 1:1 to AA residue index -> evaluate whole sequence for BPE
                if arm.lower().startswith("protgpt2"):
                    tok_pos = None  # type: ignore[assignment]
                if tier == "conserved" and len(conserved_seqs) < max_seqs:
                    conserved_seqs.append(seq)
                    conserved_pos.append(tok_pos)
                elif tier == "variable" and len(variable_seqs) < max_seqs:
                    variable_seqs.append(seq)
                    variable_pos.append(tok_pos)

    within_fams: list[list[str]] = []
    cross_fam: list[str] = []
    if fd_path.is_file():
        q = load_json_file(fd_path)
        eval_u = [u for u in q["units"] if u["partition"] == "evaluation"]
        for u in eval_u:
            wt = u["wildtypes"][0]
            rows = [r["native"] for r in wt["rows"] if r["split"] == "eval" and r.get("representative", True)]
            if not rows:
                rows = [r["native"] for r in wt["rows"]][:16]
            formatted = [format_sequence_for_arm(s, arm, saprot_mode=saprot_mode) for s in rows]
            if len(formatted) >= 4 and len(within_fams) < 8:
                within_fams.append(formatted[: min(16, max_seqs)])
            if formatted and len(cross_fam) < max_seqs:
                cross_fam.append(formatted[0])

    primary = cross_fam[:max_seqs] if cross_fam else [
        format_sequence_for_arm(d["aa_seq"], arm, saprot_mode=saprot_mode)
        for d in list(cath_cohort["domains"].values())[:max_seqs]
    ]
    return {
        "primary_sequences": primary,
        "conserved_sites": (conserved_seqs, conserved_pos) if conserved_seqs else None,
        "variable_sites": (variable_seqs, variable_pos) if variable_seqs else None,
        "within_family_batches": within_fams,
        "cross_family_batch": cross_fam[:max_seqs],
    }


def _load_text_gradient_batches(text_cohort: dict[str, Any], max_seqs: int) -> dict[str, Any]:
    articles = text_cohort["wikitext_articles"][:max_seqs]
    primary = [a["text"] for a in articles]

    # Within-article batches: split each of first 8 articles into 4 non-overlapping word chunks
    within_art: list[list[str]] = []
    for art in articles[: min(8, len(articles))]:
        words = art["text"].split()
        chunk_sz = max(40, len(words) // 4)
        chunks = [" ".join(words[i * chunk_sz : (i + 1) * chunk_sz]) for i in range(4)]
        chunks = [c for c in chunks if len(c.split()) >= 20]
        if len(chunks) >= 3:
            within_art.append(chunks)

    return {
        "primary_sequences": primary,
        "within_article_batches": within_art,
        "cross_article_batch": primary[: min(64, len(primary))],
    }


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(
        args.arm,
        device=args.device,
        dtype=args.dtype,
        model_dir=args.model_dir,
        modality=args.modality,
    )

    grad_results: dict[str, Any] = {}
    if bundle.modality == "protein":
        cath = load_json_file(args.cohort_dir / "cath_3d_kv_cohort.json")
        batches = _load_protein_gradient_batches(
            args.repo_root, args.arm, args.max_grad_seqs, args.saprot_mode, cath
        )
        grad_results["primary_rank_and_snr"] = measure_residual_gradient_rank_and_snr(
            bundle, batches["primary_sequences"], max_tokens=args.max_tokens, seed=args.seed
        )
        if batches["conserved_sites"] is not None:
            c_seqs, c_pos = batches["conserved_sites"]
            grad_results["conserved_site_snr"] = measure_residual_gradient_rank_and_snr(
                bundle, c_seqs, max_tokens=args.max_tokens, target_positions=c_pos, seed=args.seed + 1
            )
        if batches["variable_sites"] is not None:
            v_seqs, v_pos = batches["variable_sites"]
            grad_results["variable_site_snr"] = measure_residual_gradient_rank_and_snr(
                bundle, v_seqs, max_tokens=args.max_tokens, target_positions=v_pos, seed=args.seed + 2
            )
        within_snr = []
        for fi, fseqs in enumerate(batches["within_family_batches"]):
            if len(fseqs) >= 2:
                within_snr.append(
                    measure_residual_gradient_rank_and_snr(
                        bundle, fseqs, max_tokens=args.max_tokens, seed=args.seed + 10 + fi
                    )
                )
        grad_results["within_family_snr_batches"] = within_snr

        # KV Interference on CATH 3D pair cohort
        seq_map = {
            dom_id: format_sequence_for_arm(
                d["aa_seq"], args.arm, saprot_seq=d["saprot_seq"], saprot_mode=args.saprot_mode
            )
            for dom_id, d in cath["domains"].items()
        }
        pairs_by_stratum = {
            stratum: [(p["domain_a"], p["domain_b"]) for p in plist[: args.max_pairs_per_stratum]]
            for stratum, plist in cath["pairs"].items()
        }
        kv_results = evaluate_kv_interference_pairs(
            bundle,
            seq_map,
            pairs_by_stratum,
            max_tokens=args.max_tokens,
            max_sampled_positions=args.max_sampled_positions,
            seed=args.seed,
        )
    else:
        text_cohort = load_json_file(args.cohort_dir / "text_kv_and_horizon_cohort.json")
        batches = _load_text_gradient_batches(text_cohort, args.max_grad_seqs)
        grad_results["primary_rank_and_snr"] = measure_residual_gradient_rank_and_snr(
            bundle, batches["primary_sequences"], max_tokens=args.max_tokens, seed=args.seed
        )
        within_snr = []
        for ai, aseqs in enumerate(batches["within_article_batches"]):
            within_snr.append(
                measure_residual_gradient_rank_and_snr(
                    bundle, aseqs, max_tokens=args.max_tokens, seed=args.seed + 10 + ai
                )
            )
        grad_results["within_article_snr_batches"] = within_snr

        # KV Interference on OpenWebText BM25 related vs random pairs
        seq_map = {}
        pairs_by_stratum = {}
        for stratum, plist in text_cohort["text_pairs"].items():
            pairs_s = []
            for p in plist[: args.max_pairs_per_stratum]:
                ka = f"{stratum}_{p['pair_id']}_a"
                kb = f"{stratum}_{p['pair_id']}_b"
                seq_map[ka] = p["text_a"]
                seq_map[kb] = p["text_b"]
                pairs_s.append((ka, kb))
            pairs_by_stratum[stratum] = pairs_s

        kv_results = evaluate_kv_interference_pairs(
            bundle,
            seq_map,
            pairs_by_stratum,
            max_tokens=args.max_tokens,
            max_sampled_positions=args.max_sampled_positions,
            seed=args.seed,
        )

    payload = {
        "schema": "memory_theory_b2_gradient_and_kv_geometry_v1",
        "config": {
            "arm": args.arm,
            "modality": bundle.modality,
            "is_masked": bundle.is_masked,
            "dtype": args.dtype,
            "saprot_mode": args.saprot_mode if args.arm.lower().startswith("saprot") else None,
            "max_grad_seqs": args.max_grad_seqs,
            "max_pairs_per_stratum": args.max_pairs_per_stratum,
            "max_tokens": args.max_tokens,
            "max_sampled_positions": args.max_sampled_positions,
            "seed": args.seed,
            "model_dir": str(bundle.model_dir),
        },
        "gradient_geometry": grad_results,
        "kv_interference_geometry": kv_results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / args.out_name
    digest = write_json_with_sha256(out_file, payload)
    print(f"Wrote {out_file} (sha256={digest})")


if __name__ == "__main__":
    main()
