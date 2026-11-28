#!/usr/bin/env python3
"""Run the Open-Source Foundation PLM Quantitative Physical Exam on one or more models."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.open_model_audit.cohort import OpenAuditBundle, atomic_write_json_with_sha256  # noqa: E402
from src.capability.open_model_audit.evaluators import (  # noqa: E402
    load_open_audit_evaluator,
    run_audit_for_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bundle-dir",
        type=Path,
        default=REPO_ROOT / "output/open_audit_bundle",
        help="Path to the built open_audit_bundle directory.",
    )
    p.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated model IDs or local paths (e.g. 'esm2-8m,esm2-35m,saprot-35m,proteinmpnn-v-020').",
    )
    p.add_argument(
        "--family",
        type=str,
        default="auto",
        choices=["auto", "ar", "mlm", "mdm", "saprot", "proteinmpnn", "dplm2", "esm3", "prot_t5"],
        help="Optional explicit model family override (default: auto-detect).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output/open_audit_results",
        help="Directory to write per-model JSON reports and combined summary.",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--model-root", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-mlm15-draws", type=int, default=8)
    p.add_argument("--n-loo-probes", type=int, default=16)
    p.add_argument("--n-mdm-strata", type=int, default=16)
    p.add_argument("--max-seqs-per-bucket", type=int, default=0)
    p.add_argument("--max-seqs-per-split", type=int, default=0)
    p.add_argument("--max-panel-per-group", type=int, default=0)
    p.add_argument("--progen2-bidirectional", action="store_true")
    p.add_argument("--seq-reference-bits", type=float, default=None)
    p.add_argument("--seed", type=int, default=20261129)
    return p.parse_args()


def _compact_headline(report: dict[str, Any]) -> dict[str, Any]:
    m = report["model"]
    prim = m["primary_readout"]
    row: dict[str, Any] = {
        "model_id": m["model_id"],
        "family_kind": m["family_kind"],
        "primary_readout": prim,
        "P_non_emb_M": round(m["n_params_non_emb"] / 1e6, 3),
        "wall_seconds": report.get("wall_seconds"),
    }
    ht_all = (report.get("horvitz_thompson_all_uniref50") or {}).get(prim)
    if ht_all:
        row["B_total_Gbits"] = round(ht_all["B_total_gbits"], 3)
        row["c_app_bits_per_param"] = round(ht_all["c_app_bits_per_non_emb_param"], 3)
    t3_dn = (report.get("three_tier_eq3a_denovo_unseen_fold") or {}).get(prim)
    if t3_dn:
        row["dh_univ_denovo"] = round(t3_dn["delta_h_univ"], 4)
        row["dh_spec_denovo"] = round(t3_dn["delta_h_spec"], 4)
        row["dh_res_denovo"] = round(t3_dn["delta_h_res"], 4)
        if "calibrated" in t3_dn:
            row["delta_cal_denovo"] = round(t3_dn["calibrated"]["delta_cal_unseen_fold"], 4)
    # Check MDM-16 high-mask OOD jump on unseen_fold_denovo if present
    mdm_dn = (report.get("sequence_splits") or {}).get("unseen_fold_denovo_zero_hit", {}).get("mdm_16")
    if mdm_dn and "high_mask_ood_jump_bits" in mdm_dn:
        row["mdm16_bits_denovo"] = round(mdm_dn["bits_per_res"], 4)
        row["mdm16_high_mask_ood_jump"] = round(mdm_dn["high_mask_ood_jump_bits"], 4)
    p3d = report.get("paired_3d_panel")
    if p3d:
        ov = p3d.get("overall", {})
        if "ar_3d_perm_avg" in ov:
            row["H_3d_overall"] = round(ov["ar_3d_perm_avg"]["bits_per_res"], 4)
            row["delta_cal_3d"] = round(ov["ar_3d_perm_avg"]["delta_cal_bits"], 4)
            by_g = p3d.get("by_domain_group", {})
            if "natural_cath_s40" in by_g:
                row["H_3d_natural"] = round(by_g["natural_cath_s40"]["ar_3d_perm_avg"]["bits_per_res"], 4)
            if "de_novo_zero_hit" in by_g:
                row["H_3d_denovo"] = round(by_g["de_novo_zero_hit"]["ar_3d_perm_avg"]["bits_per_res"], 4)
        elif "b_ext_full_3di" in ov:
            row["b_ext_mlm15_full3di"] = round(ov["b_ext_full_3di"]["mlm_15"]["b_ext_bits_per_res"], 4)
            row["b_ext_mlm15_ctx3di"] = round(ov["b_ext_context_only_3di"]["mlm_15"]["b_ext_bits_per_res"], 4)
        elif "b_ext" in ov:
            row["b_ext_mdm16"] = round(ov["b_ext"]["mdm_16"]["b_ext_bits_per_res"], 4)
    return row


def main() -> None:
    args = parse_args()
    bundle = OpenAuditBundle(args.bundle_dir)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    headlines: list[dict[str, Any]] = []
    inferred_seq_ref = args.seq_reference_bits

    for model_spec in model_list:
        print(f"[open-audit] Loading {model_spec} on {args.device} ({args.dtype}) ...", flush=True)
        ev = load_open_audit_evaluator(
            model_spec,
            family=args.family,
            device=args.device,
            dtype=args.dtype,
            model_root=args.model_root,
            progen2_bidirectional=args.progen2_bidirectional,
        )
        report = run_audit_for_model(
            ev,
            bundle,
            batch_size=args.batch_size,
            n_mlm15_draws=args.n_mlm15_draws,
            n_loo_probes=args.n_loo_probes,
            n_mdm_strata=args.n_mdm_strata,
            max_seqs_per_bucket=args.max_seqs_per_bucket,
            max_seqs_per_split=args.max_seqs_per_split,
            max_panel_per_group=args.max_panel_per_group,
            eval_seed=args.seed,
            seq_reference_bits=inferred_seq_ref,
        )
        # If this model evaluated unseen_fold_denovo_zero_hit and no seq_reference_bits was supplied,
        # record it as a convenient reference anchor for any subsequent structure-only model (e.g. ProteinMPNN).
        if inferred_seq_ref is None and ev.supports_sequence_splits:
            prim = ev.primary_readout
            dn = (report.get("sequence_splits") or {}).get("unseen_fold_denovo_zero_hit", {}).get(prim)
            if dn and isinstance(dn.get("bits_per_res"), (int, float)):
                inferred_seq_ref = float(dn["bits_per_res"])

        slug = model_spec.replace("/", "_").replace(":", "_")
        atomic_write_json_with_sha256(out_dir / f"audit_{slug}.json", report)
        hl = _compact_headline(report)
        headlines.append(hl)
        print(f"[open-audit] Finished {model_spec}: {json.dumps(hl)}", flush=True)

        del ev
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_payload = {
        "bundle_dir": str(bundle.bundle_dir),
        "n_models": len(headlines),
        "headlines": headlines,
    }
    atomic_write_json_with_sha256(out_dir / "audit_summary.json", summary_payload)
    print(json.dumps(summary_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
