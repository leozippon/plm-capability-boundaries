#!/usr/bin/env python3
"""Audit and summarize all Open-PLM physical exam JSONs with intra-family scaling laws & b_ext(Z) synthesis."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.memory_theory.io_utils import write_json_with_sha256
from src.capability.supply_demand.production_criteria import (
    compute_structural_parameter_equivalence,
    default_supply_profiles,
    load_calibrated_supply_profiles,
)
from src.capability.supply_demand.spec import assert_no_forbidden_infra_strings

SCHEMA_VERSION = "open_model_audit_phase2_summary_v1"
UNIREF50_ELIGIBLE_RESIDUES = 5_378_061_546
UNIREF50_ALL_RESIDUES = 17_282_055_792


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _family_group(model_id: str) -> str:
    low = model_id.lower()
    if low.startswith("esm2"):
        return "ESM-2"
    if low.startswith("esmc"):
        return "ESMC"
    if low.startswith("esm3"):
        return "ESM-3"
    if low.startswith("dplm2"):
        return "DPLM-2"
    if low.startswith("dplm"):
        return "DPLM"
    if low.startswith("amplify"):
        return "AMPLIFY"
    if low.startswith("protgpt3"):
        return "ProtGPT3"
    if low.startswith("progen2"):
        return "ProGen2"
    if low.startswith("progen3"):
        return "ProGen3"
    if low.startswith("galactica"):
        return "Galactica"
    if low.startswith("prot-t5") or low.startswith("prot_t5"):
        return "Prot-T5"
    if low.startswith("saprot"):
        return "SaProt"
    if low.startswith("proteinmpnn"):
        return "ProteinMPNN"
    return "Other"


def _fit_family_scaling_laws(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit B_total(P) = a + b * log10(P) and c_app(P) = c0 * P^(-gamma) within one model family."""
    valid = [
        (float(r["n_params_non_emb"]), float(r["B_total_gbits_all"]), float(r["c_app_bpp_all"]))
        for r in rows
        if r.get("n_params_non_emb") and r.get("B_total_gbits_all") and r.get("c_app_bpp_all")
        and float(r["B_total_gbits_all"]) > 0.0 and float(r["c_app_bpp_all"]) > 0.0
    ]
    valid.sort(key=lambda t: t[0])
    n = len(valid)
    if n < 2:
        return {"n_rungs": n, "status": "insufficient_rungs"}

    log10_p = np.asarray([math.log10(t[0]) for t in valid], dtype=np.float64)
    b_gbits = np.asarray([t[1] for t in valid], dtype=np.float64)
    log10_c = np.asarray([math.log10(t[2]) for t in valid], dtype=np.float64)

    # Linear fit B_total(Gbit) = a + b * log10(P)
    b_slope, b_intercept = np.polyfit(log10_p, b_gbits, 1)
    b_pred = b_intercept + b_slope * log10_p
    ss_res_b = float(np.sum((b_gbits - b_pred) ** 2))
    ss_tot_b = float(np.sum((b_gbits - float(np.mean(b_gbits))) ** 2))
    r2_b = 1.0 - ss_res_b / ss_tot_b if ss_tot_b > 1e-15 else 1.0

    # Power-law fit c_app = c0 * P^(-gamma)  <=>  log10(c_app) = log10(c0) - gamma * log10(P)
    c_slope, c_intercept = np.polyfit(log10_p, log10_c, 1)
    c_pred = c_intercept + c_slope * log10_p
    ss_res_c = float(np.sum((log10_c - c_pred) ** 2))
    ss_tot_c = float(np.sum((log10_c - float(np.mean(log10_c))) ** 2))
    r2_c = 1.0 - ss_res_c / ss_tot_c if ss_tot_c > 1e-15 else 1.0

    # Also compute P where fitted c_app(P) crosses 3.69 (AR c_inf) and 1.0 bpp
    gamma = -float(c_slope)
    p_at_c_3p69 = (10.0 ** ((c_intercept - math.log10(3.69)) / gamma)) if gamma > 1e-6 else None
    p_at_c_1p0 = (10.0 ** (c_intercept / gamma)) if gamma > 1e-6 else None

    return {
        "n_rungs": n,
        "status": "fitted",
        "P_range": [int(valid[0][0]), int(valid[-1][0])],
        "B_total_gbits_range": [round(valid[0][1], 4), round(valid[-1][1], 4)],
        "c_app_bpp_range": [round(valid[0][2], 4), round(valid[-1][2], 4)],
        "B_total_vs_log10P": {
            "slope_gbits_per_decade": round(float(b_slope), 4),
            "intercept_gbits": round(float(b_intercept), 4),
            "r2": round(float(r2_b), 5),
        },
        "c_app_power_law": {
            "gamma_exponent": round(gamma, 5),
            "log10_c0": round(float(c_intercept), 5),
            "r2_log_log": round(float(r2_c), 5),
            "implied_P_at_c_app_3p69": int(round(p_at_c_3p69)) if p_at_c_3p69 and p_at_c_3p69 < 1e15 else None,
            "implied_P_at_c_app_1p0": int(round(p_at_c_1p0)) if p_at_c_1p0 and p_at_c_1p0 < 1e15 else None,
        },
    }


def summarize_open_model_audit(
    audit_dir: Path,
    *,
    stage1_summary_path: Path | None = None,
) -> dict[str, Any]:
    files = sorted(p for p in audit_dir.glob("audit_*.json") if not p.name.endswith(".sha256"))
    if not files:
        raise FileNotFoundError(f"No audit_*.json files found in {audit_dir}")

    profiles = (
        load_calibrated_supply_profiles(stage1_summary_path)
        if stage1_summary_path is not None and stage1_summary_path.is_file()
        else default_supply_profiles()
    )

    models_out: list[dict[str, Any]] = []
    protocol_flagged: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for path in files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, Mapping) or doc.get("schema_version") != "open_model_audit_report_v1":
            continue
        mdl = doc.get("model") or {}
        mid = str(mdl.get("model_id", path.stem))
        fam_group = _family_group(mid)
        prim = str(mdl.get("primary_readout", ""))
        p_ne = int(mdl.get("n_params_non_emb", 0))
        p_tot = int(mdl.get("n_params_total", 0))

        ht_all = (doc.get("horvitz_thompson_all_uniref50") or {}).get(prim) or {}
        ht_elig = (doc.get("horvitz_thompson_eligible_uniref50") or {}).get(prim) or {}
        tt_dn = (doc.get("three_tier_eq3a_denovo_unseen_fold") or {}).get(prim) or {}
        tt_or = (doc.get("three_tier_eq3a_orphan_unseen_fold") or {}).get(prim) or {}
        seq_splits = doc.get("sequence_splits") or {}

        # Primary & MDM-16 OOD jump on de novo zero-hit folds
        dn_split = seq_splits.get("unseen_fold_denovo_zero_hit") or {}
        prim_dn = dn_split.get(prim) or {}
        mdm_dn = dn_split.get("mdm_16") or {}
        mlm_dn = dn_split.get("mlm_15") or {}
        mdm_ood_jump = None
        if _safe_float(mdm_dn.get("bits_per_res")) is not None and _safe_float(mlm_dn.get("bits_per_res")) is not None:
            mdm_ood_jump = float(mdm_dn["bits_per_res"]) - float(mlm_dn["bits_per_res"])

        # Flag protocol mismatches cleanly (e.g. Prot-T5 released head not trained with <extra_id_0> decoder prefix)
        dcal_dn = _safe_float(prim_dn.get("delta_cal_bits"))
        hcal_dn = _safe_float(prim_dn.get("calibrated_bits_per_res"))
        is_protocol_mismatch = bool(
            fam_group == "Prot-T5" and dcal_dn is not None and dcal_dn > 10.0 and hcal_dn is not None and hcal_dn > 4.25
        )

        # 3D panel / external supply b_ext(Z)
        p3d = (doc.get("paired_3d_panel") or {}).get("overall") or {}
        p3d_groups = (doc.get("paired_3d_panel") or {}).get("by_domain_group") or {}
        struct_summary: dict[str, Any] | None = None
        if p3d:
            if "b_ext_full_3di" in p3d:
                bf = _safe_float((p3d["b_ext_full_3di"].get(prim) or {}).get("b_ext_bits_per_res"))
                bc = _safe_float((p3d["b_ext_context_only_3di"].get(prim) or {}).get("b_ext_bits_per_res"))
                bf_mdm = _safe_float((p3d["b_ext_full_3di"].get("mdm_16") or {}).get("b_ext_bits_per_res"))
                bc_mdm = _safe_float((p3d["b_ext_context_only_3di"].get("mdm_16") or {}).get("b_ext_bits_per_res"))
                struct_summary = {
                    "kind": "saprot_3di_paired_ablation",
                    "b_ext_full_3di_primary": round(bf, 5) if bf is not None else None,
                    "b_ext_context_only_3di_primary": round(bc, 5) if bc is not None else None,
                    "b_ext_full_3di_mdm16": round(bf_mdm, 5) if bf_mdm is not None else None,
                    "b_ext_context_only_3di_mdm16": round(bc_mdm, 5) if bc_mdm is not None else None,
                }
            elif "b_ext" in p3d:
                b_prim = _safe_float((p3d["b_ext"].get(prim) or {}).get("b_ext_bits_per_res"))
                b_mdm = _safe_float((p3d["b_ext"].get("mdm_16") or {}).get("b_ext_bits_per_res"))
                struct_summary = {
                    "kind": "multitrack_3d_ablation",
                    "b_ext_primary": round(b_prim, 5) if b_prim is not None else None,
                    "b_ext_mdm16": round(b_mdm, 5) if b_mdm is not None else None,
                    "uncond_primary_bits": _safe_float(((p3d.get("unconditioned_null_Z") or {}).get(prim) or {}).get("bits_per_res")),
                    "cond_primary_bits": _safe_float(((p3d.get("conditioned_3d") or {}).get(prim) or {}).get("bits_per_res")),
                }
            elif "ar_3d_perm_avg" in p3d:
                h_ar = _safe_float(p3d["ar_3d_perm_avg"].get("bits_per_res"))
                h_loo = _safe_float((p3d.get("loo_3d") or {}).get("bits_per_res"))
                nat_h = _safe_float(((p3d_groups.get("natural_cath_s40") or {}).get("ar_3d_perm_avg") or {}).get("bits_per_res"))
                dn_h = _safe_float(((p3d_groups.get("de_novo_zero_hit") or {}).get("ar_3d_perm_avg") or {}).get("bits_per_res"))
                struct_summary = {
                    "kind": "inverse_folding_3d_only",
                    "H_3d_ar_overall": round(h_ar, 5) if h_ar is not None else None,
                    "H_3d_loo_overall": round(h_loo, 5) if h_loo is not None else None,
                    "H_3d_ar_natural_cath_s40": round(nat_h, 5) if nat_h is not None else None,
                    "H_3d_ar_denovo_zero_hit": round(dn_h, 5) if dn_h is not None else None,
                    "delta_h_vs_H0": round(math.log2(20.0) - h_ar, 5) if h_ar is not None else None,
                }
            # Attach parameter-equivalence on UniRef50 eligible 5.38B residues when b_ext > 0
            rep_b_ext = (
                struct_summary.get("b_ext_full_3di_primary")
                or struct_summary.get("b_ext_primary")
                or struct_summary.get("delta_h_vs_H0")
            )
            if rep_b_ext is not None and rep_b_ext > 0.0:
                b_ext_total_elig = float(rep_b_ext) * float(UNIREF50_ELIGIBLE_RESIDUES)
                struct_summary["param_equivalence_uniref50_elig"] = {
                    obj: round(
                        compute_structural_parameter_equivalence(
                            b_ext_total_elig, obj, supply_profiles=profiles
                        )["Delta_P_saved_asymptotic"]
                        / 1e9,
                        3,
                    )
                    for obj in ("ar", "mlm_15", "oa_uniform")
                }

        b_tot_all = _safe_float(ht_all.get("B_total_bits"))
        b_tot_elig = _safe_float(ht_elig.get("B_total_bits"))
        c_app_all = _safe_float(ht_all.get("c_app_bits_per_non_emb_param"))
        c_app_elig = _safe_float(ht_elig.get("c_app_bits_per_non_emb_param"))

        entry: dict[str, Any] = {
            "model_id": mid,
            "family_group": fam_group,
            "family_kind": mdl.get("family_kind"),
            "primary_readout": prim,
            "n_params_total": p_tot,
            "n_params_non_emb": p_ne,
            "protocol_mismatch_flag": is_protocol_mismatch,
            "B_total_gbits_all": round(b_tot_all / 1e9, 4) if b_tot_all is not None else None,
            "B_total_se_gbits_all": round(float(ht_all["B_total_se_bits"]) / 1e9, 4)
            if _safe_float(ht_all.get("B_total_se_bits")) is not None
            else None,
            "c_app_bpp_all": round(c_app_all, 4) if c_app_all is not None else None,
            "B_total_gbits_elig": round(b_tot_elig / 1e9, 4) if b_tot_elig is not None else None,
            "c_app_bpp_elig": round(c_app_elig, 4) if c_app_elig is not None else None,
            "eq3a_denovo": {
                "delta_h_univ": round(float(tt_dn["delta_h_univ"]), 5) if _safe_float(tt_dn.get("delta_h_univ")) is not None else None,
                "delta_h_spec": round(float(tt_dn["delta_h_spec"]), 5) if _safe_float(tt_dn.get("delta_h_spec")) is not None else None,
                "delta_h_res": round(float(tt_dn["delta_h_res"]), 5) if _safe_float(tt_dn.get("delta_h_res")) is not None else None,
                "H_cal_denovo": round(hcal_dn, 5) if hcal_dn is not None else None,
                "Delta_cal_denovo": round(dcal_dn, 5) if dcal_dn is not None else None,
            }
            if tt_dn
            else None,
            "eq3a_orphan": {
                "delta_h_univ": round(float(tt_or["delta_h_univ"]), 5) if _safe_float(tt_or.get("delta_h_univ")) is not None else None,
                "delta_h_spec": round(float(tt_or["delta_h_spec"]), 5) if _safe_float(tt_or.get("delta_h_spec")) is not None else None,
                "delta_h_res": round(float(tt_or["delta_h_res"]), 5) if _safe_float(tt_or.get("delta_h_res")) is not None else None,
            }
            if tt_or
            else None,
            "mdm16_denovo_bits": round(float(mdm_dn["bits_per_res"]), 5) if _safe_float(mdm_dn.get("bits_per_res")) is not None else None,
            "mdm16_ood_jump_vs_mlm15": round(mdm_ood_jump, 5) if mdm_ood_jump is not None else None,
            "paired_3d_panel": struct_summary,
        }
        models_out.append(entry)
        if is_protocol_mismatch:
            protocol_flagged.append(
                {
                    "model_id": mid,
                    "reason": (
                        "Released Prot-T5-XL/XXL-UniRef50 checkpoints use BERT-style denoising rather than "
                        "standard T5 <extra_id_0> decoder-prefix infilling; raw LOO CE ~26.6 bits/res and "
                        "H_cal ~4.31 bits/res (near uniform H0=4.322) -> excluded from intra-family scaling fits."
                    ),
                    "Delta_cal_denovo": entry["eq3a_denovo"]["Delta_cal_denovo"] if entry["eq3a_denovo"] else None,
                    "H_cal_denovo": entry["eq3a_denovo"]["H_cal_denovo"] if entry["eq3a_denovo"] else None,
                }
            )
        else:
            by_family[fam_group].append(entry)

    scaling_laws = {
        fam: _fit_family_scaling_laws(rows)
        for fam, rows in sorted(by_family.items())
        if any(r.get("B_total_gbits_all") is not None for r in rows)
    }

    report: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "n_models_total": len(models_out),
            "n_models_in_protocol": len(models_out) - len(protocol_flagged),
            "n_protocol_mismatch_flagged": len(protocol_flagged),
            "families_present": sorted(by_family.keys()),
        },
        "intra_family_scaling_laws": scaling_laws,
        "protocol_mismatch_diagnostics": protocol_flagged,
        "models": sorted(models_out, key=lambda r: (str(r["family_group"]), int(r["n_params_non_emb"]))),
    }
    assert_no_forbidden_infra_strings(report, label="open_model_audit_summary")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-dir", type=Path, default=REPO_ROOT / "output/open_model_audit")
    p.add_argument(
        "--stage1-summary",
        type=Path,
        default=REPO_ROOT / "output/stage1_wsd/stage1_combined_wave3_summary.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results/extensions/supply_demand_phase2_20261128/open_model_audit_summary.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = summarize_open_model_audit(
        args.audit_dir,
        stage1_summary_path=args.stage1_summary if args.stage1_summary.is_file() else None,
    )
    write_json_with_sha256(args.out, report)
    size_kb = args.out.stat().st_size / 1024.0
    print(
        json.dumps(
            {
                "status": "ok",
                "out": str(args.out),
                "size_kb": round(size_kb, 2),
                "counts": report["counts"],
                "scaling_laws": report["intra_family_scaling_laws"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
