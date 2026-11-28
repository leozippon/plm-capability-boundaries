#!/usr/bin/env python3
"""Standalone CPU aggregation CLI for the 20-AA PLM parametric memory capacity study.

Reads completed per-run JSON receipts from ``--runs-dir``, verifies SHA-256 sidecars, computes
plateau saturation curves, ``c_max`` scaling laws, exposure-discount curves, AR-vs-MLM efficiency,
LOO pseudo-entropy inflation audits, and ablation contrasts, writes ``--out`` (+ ``.sha256``),
and prints a concise Markdown summary table to stdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.io import sha256_file  # noqa: E402
from src.capability.memory_capacity.analysis import summarize_memory_capacity_campaign  # noqa: E402


def _fmt_ci(stat: dict[str, Any] | None, digits: int = 3) -> str:
    if not stat or "point" not in stat:
        return "n/a"
    pt = float(stat["point"])
    ci = stat.get("ci95")
    if isinstance(ci, list) and len(ci) == 2:
        return f"{pt:.{digits}f} [{float(ci[0]):.{digits}f}, {float(ci[1]):.{digits}f}]"
    return f"{pt:.{digits}f}"


def render_markdown_report(summary: dict[str, Any], out_path: Path, digest: str) -> str:
    prov = summary["provenance"]
    lines: list[str] = [
        "# 20-AA PLM Native Parametric Memory Capacity Summary",
        "",
        f"- **Completed runs**: {prov['n_completed_runs']} "
        f"(SHA-256 sidecars verified: {prov['n_sidecars_verified']})",
        f"- **Tracks**: {prov['runs_by_track']}",
        f"- **Scales**: {prov['runs_by_scale']}",
        f"- **Artifact**: `{out_path.name}` (`sha256={digest[:16]}...`)",
        "",
        "## 1. Native Maximum Storage Capacity (`c_max` Calibration)",
        "",
        "| Paradigm | Regime | Anchor `e` | `c_max` (non-emb, ratio) | `c_max` (non-emb, OLS) | Log-Log `alpha` | `c_max` (total, ratio) | vs Morris (3.6) |",
        "| --- | --- | ---: | --- | --- | --- | --- | ---: |",
    ]

    cal_by_p = summary.get("c_max_calibration", {}).get("by_paradigm", {})
    for paradigm in sorted(cal_by_p.keys()):
        pblock = cal_by_p[paradigm]
        anchor_e = pblock.get("anchor_epoch")
        nlp_ref = (
            (pblock.get("nlp_reference_comparison") or {})
            .get("non_embedding_direct")
            or {}
        ).get("morris_et_al_2025_gpt")
        vs_morris = (
            f"{float(nlp_ref['ratio_plm_over_nlp']):.2f}x" if nlp_ref else "n/a"
        )

        for regime_key, label in (
            ("direct_anchor_unadjusted", "Direct unadjusted"),
            ("exposure_anchored_multi_scale", "Exposure-anchored"),
        ):
            rblock = pblock.get(regime_key) or {}
            ne = rblock.get("non_embedding_params") or {}
            tot = rblock.get("total_params") or {}
            zr = ne.get("zero_intercept_ratio")
            ols = ne.get("ols")
            ll = ne.get("log_log_scaling")
            zr_tot = tot.get("zero_intercept_ratio")

            zr_str = (
                f"{float(zr['c_max']):.3f} [{float(zr['ci95'][0]):.3f}, {float(zr['ci95'][1]):.3f}]"
                if zr
                else "n/a"
            )
            ols_str = f"{float(ols['c_max_slope']):.3f} (R²={float(ols['r2']):.4f})" if ols else "n/a"
            ll_str = f"{float(ll['alpha_exponent']):.3f}" if ll else "n/a"
            tot_str = f"{float(zr_tot['c_max']):.3f}" if zr_tot else "n/a"
            lines.append(
                f"| `{paradigm.upper()}` | {label} | {anchor_e} | {zr_str} | {ols_str} | {ll_str} | {tot_str} | {vs_morris} |"
            )

    lines.extend(
        [
            "",
            "## 2. Exposure-Discount Law `c_eff(e)` (Delayed-Hill & Nonparametric `e_crit`)",
            "",
            "| Paradigm | `c_inf` (bits/param) | `e_crit` (delayed-Hill) | `e_crit` (`dbpt=0.05`) | `e_crit` (`5% c_inf`) | `e_50` (epochs) | `e_90` (epochs) | Hill `alpha` | `R^2` |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    exp_curves = summary.get("exposure_discount_curves", {})
    for paradigm in sorted(exp_curves.keys()):
        pexp = exp_curves[paradigm]
        dhill = pexp.get("pooled_delayed_hill_fit_empirical") or pexp.get(
            "pooled_delayed_hill_fit_at_2x"
        )
        hill = pexp.get("pooled_hill_fit_empirical") or pexp.get("pooled_hill_fit_at_2x")
        ec_np = pexp.get("pooled_e_crit_nonparametric") or {}
        fit = dhill or hill
        ec_dh = (
            f"{float(dhill['e_crit_epochs']):.2f}"
            if dhill and dhill.get("e_crit_epochs") is not None
            else "n/a"
        )
        ec_dbpt = (
            f"{float(ec_np['e_crit_delta_bpt_0p05']):.2f}"
            if ec_np.get("e_crit_delta_bpt_0p05") is not None
            else "n/a"
        )
        ec_frac = (
            f"{float(ec_np['e_crit_frac_c_inf_0p05']):.2f}"
            if ec_np.get("e_crit_frac_c_inf_0p05") is not None
            else "n/a"
        )
        if fit:
            lines.append(
                f"| `{paradigm.upper()}` | {float(fit['c_inf_bits_per_param']):.3f} | "
                f"{ec_dh} | {ec_dbpt} | {ec_frac} | "
                f"{float(fit['e_50_epochs']):.2f} | {float(fit['e_90_epochs']):.1f} | "
                f"{float(fit['alpha_hill']):.3f} | {float(fit['r2']):.4f} |"
            )
        else:
            lines.append(
                f"| `{paradigm.upper()}` | n/a | n/a | {ec_dbpt} | {ec_frac} | n/a | n/a | n/a | n/a |"
            )

    ar_mlm = summary.get("ar_vs_mlm_comparison", {})
    e400_sat = ar_mlm.get("asymptotic_e400_saturated_ge2x") or ar_mlm.get("overall_saturated_ge2x") or {}
    loo = (ar_mlm.get("loo_inflation_audit") or {}).get("primary_mlm_observations") or {}

    lines.extend(
        [
            "",
            "## 3. Paired AR vs. MLM Efficiency & LOO Pseudo-Entropy Inflation",
            "",
            f"- **Paired cells (`n={ar_mlm.get('n_total_paired_cells', 0)}`)**:",
            f"  - `c_AR - c_MLM` (bits/non-emb param): {_fmt_ci(e400_sat.get('ar_minus_mlm_bits_per_param'))}",
            f"  - `M_MLM / M_AR` total-bits ratio: {_fmt_ci(e400_sat.get('mlm_over_ar_total_bits_ratio'))}",
            f"- **MLM Leave-One-Out Inflation (`LOO / Exact 16-pt`)**: "
            f"{_fmt_ci(loo.get('aggregate_inflation_ratio_sum_loo_over_sum_exact'))} "
            f"(excess `{_fmt_ci(loo.get('excess_loo_bits_per_non_emb_param'))}` bits/param)",
        ]
    )

    abl = summary.get("ablations", {})
    wd = abl.get("weight_decay_0_vs_0p01", {})
    fp = abl.get("precision_bf16_vs_fp32_strict", {})
    hz = abl.get("cosine_single_run_vs_dedicated_horizon", {})
    lines.extend(
        [
            "",
            "## 4. Methodological Ablations",
            "",
            f"- **Weight decay (`0.01 - 0.0`, `n={wd.get('n_paired_observations', 0)}`)**: "
            f"{_fmt_ci(wd.get('overall_delta_bits_per_param'), 4)} bits/param",
            f"- **Precision (`FP32 - BF16`, `n={fp.get('n_paired_observations', 0)}`)**: "
            f"{_fmt_ci(fp.get('overall_delta_bits_per_param'), 4)} bits/param",
            f"- **Schedule (`Dedicated horizon - Single trajectory`, `n={hz.get('n_paired_horizons', 0)}`)**: "
            f"{_fmt_ci(hz.get('overall_dedicated_minus_trajectory_bpp'), 4)} bits/param",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Directory containing completed per-run JSON receipts (<out_dir>/runs or <out_dir>)",
    )
    parser.add_argument(
        "--extra-runs-dir",
        type=Path,
        action="append",
        default=[],
        help="Optional additional runs directory to merge (e.g. combining Batch-1 and Stage-1 WSD; WSD/higher max_epochs wins per cell)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination path for parametric_capacity_summary.json",
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=2000,
        help="Number of bootstrap resamples for 95%% confidence intervals (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261127,
        help="Deterministic RNG seed for bootstrap resampling (default: 20261127)",
    )
    args = parser.parse_args(argv)

    summary = summarize_memory_capacity_campaign(
        args.runs_dir,
        extra_runs_dirs=args.extra_runs_dir,
        out_path=args.out,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    digest = sha256_file(args.out)
    print(render_markdown_report(summary, args.out, digest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
