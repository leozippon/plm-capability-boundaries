"""Unit and CLI tests for Stage-2 supply-demand campaign analysis."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.supply_demand import (
    ANALYSIS_SCHEMA,
    RUN_RECEIPT_SCHEMA,
    analyze_stage2_run,
    interpolate_stage1_c_eff,
    load_stage1_capacity_lookup,
    summarize_supply_demand_campaign,
)


def _make_receipt(
    run_id: str,
    arm_id: str,
    objective: str,
    milestones: list[tuple[int, float, float, float, float, float, float]],
    *,
    status: str = "complete",
    n_params: int = 35_000_000,
    n_tokens: int = 5_000_000,
) -> dict:
    # milestones tuple: (epoch, dh_univ, dh_spec, dh_res, ms_all, pg_assay, ct_auroc)
    ms_list = []
    for ep, du, ds, dr, ms_all, pg, ct in milestones:
        h_uf = 4.32 - du
        h_hom = h_uf - ds
        h_tr = h_hom - dr
        ms_list.append(
            {
                "epoch": ep,
                "global_step": ep * 100,
                "tokens_seen": ep * n_tokens,
                "lr": 1e-3,
                "three_tier_bits": {
                    "native": {
                        "H_0": 4.32,
                        "H_train": h_tr,
                        "H_unseen_homolog": h_hom,
                        "H_unseen_fold": h_uf,
                        "delta_h_univ": du,
                        "delta_h_spec": ds,
                        "delta_h_res": dr,
                        "delta_h_total": du + ds + dr,
                    },
                    "loo": {
                        "delta_h_univ": du,
                        "delta_h_spec": ds,
                        "delta_h_res": dr,
                    },
                },
                "megascale": {
                    "overall_spearman_mean": ms_all,
                    "by_stratum": {
                        "de_novo_design": {"spearman_mean": ms_all - 0.01},
                        "unseen_fold_natural": {"spearman_mean": ms_all + 0.01},
                        "seen_pool_natural": {"spearman_mean": ms_all},
                    },
                },
                "proteingym_milestone": {
                    "assay_mean_spearman": pg,
                    "cluster_mean_spearman": pg - 0.005,
                },
                "contacts": {
                    "long_range_sep_ge_24": {
                        "p_at_L": 0.05 + 0.1 * max(0.0, ct - 0.5),
                        "p_at_L_over_5": 0.08 + 0.1 * max(0.0, ct - 0.5),
                        "auroc": ct,
                    },
                    "sep_ge_6": {"auroc": ct + 0.04},
                },
            }
        )
    return {
        "schema": RUN_RECEIPT_SCHEMA,
        "status": status,
        "run_id": run_id,
        "arm_id": arm_id,
        "objective": objective,
        "scale_id": "35m",
        "epoch_done": milestones[-1][0],
        "parameter_counts": {"n_params_non_embedding": n_params},
        "corpus_stats": {
            "n_train_sequences": 20_000,
            "n_train_unique_tokens": n_tokens,
        },
        "run_spec": {"max_epochs": 300},
        "init_eval": {
            "epoch": 0,
            "three_tier_bits": {
                "native": {
                    "H_train": 4.32,
                    "H_unseen_homolog": 4.32,
                    "H_unseen_fold": 4.32,
                    "delta_h_univ": 0.0,
                    "delta_h_spec": 0.0,
                    "delta_h_res": 0.0,
                }
            },
        },
        "milestones": ms_list,
    }


def test_stage2_analysis_peaks_absorption_inversion_and_stage1_alignment(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    # AR: early K_res takeoff (~ep 5), sharp MegaScale/ProteinGym peak at ep 5 then inversion
    ar_doc = _make_receipt(
        "s2a_m64_ar_35m_e300",
        "M64",
        "ar",
        [
            (1, 0.18, 0.04, 0.00, -0.02, 0.01, 0.52),
            (2, 0.22, 0.08, 0.01, 0.04, 0.04, 0.55),
            (5, 0.20, 0.24, 0.06, 0.08, 0.07, 0.58),
            (10, 0.05, 0.40, 0.75, 0.00, -0.02, 0.54),
            (20, -2.50, 0.60, 5.80, -0.08, -0.06, 0.51),
            (40, -8.00, 1.40, 11.20, -0.03, -0.03, 0.50),
        ],
    )
    # MLM-15%: much later K_res takeoff (~ep 25), no negative inversion through ep 40
    mlm_doc = _make_receipt(
        "s2a_m64_mlm_15_35m_e300",
        "M64",
        "mlm_15",
        [
            (1, 0.16, 0.04, -0.01, -0.03, -0.01, 0.51),
            (5, 0.24, 0.05, -0.01, 0.01, 0.02, 0.54),
            (20, 0.25, 0.18, 0.02, 0.05, 0.04, 0.57),
            (40, 0.21, 0.40, 0.18, 0.04, 0.03, 0.56),
        ],
        status="in_progress",
    )
    (runs_dir / f"{ar_doc['run_id']}.json").write_text(json.dumps(ar_doc), encoding="utf-8")
    (runs_dir / f"{mlm_doc['run_id']}.json").write_text(json.dumps(mlm_doc), encoding="utf-8")

    # Stage-1 synthetic summary fixture
    s1_summary = tmp_path / "stage1_summary.json"
    s1_summary.write_text(
        json.dumps(
            {
                "exposure_discount_curves": {
                    "ar": {
                        "pooled_by_exposure": [
                            {"epoch": 2, "c_eff_empirical_mean": 0.002},
                            {"epoch": 5, "c_eff_empirical_mean": 0.025},
                            {"epoch": 10, "c_eff_empirical_mean": 0.15},
                            {"epoch": 40, "c_eff_empirical_mean": 2.40},
                        ],
                        "pooled_e_crit_nonparametric": {
                            "e_crit_c_eff_0p02": 4.7,
                            "c_inf_reference_empirical": 2.6,
                        },
                    },
                    "mlm_15": {
                        "pooled_by_exposure": [
                            {"epoch": 10, "c_eff_empirical_mean": 0.001},
                            {"epoch": 40, "c_eff_empirical_mean": 0.005},
                            {"epoch": 250, "c_eff_empirical_mean": 0.03},
                        ],
                        "pooled_e_crit_nonparametric": {
                            "e_crit_c_eff_0p02": 210.0,
                            "c_inf_reference_empirical": 0.65,
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    s1_lookup = load_stage1_capacity_lookup(s1_summary)
    assert interpolate_stage1_c_eff(s1_lookup, "ar", 5.0) == pytest.approx(0.025)

    ar_res = analyze_stage2_run(ar_doc, stage1_lookup=s1_lookup)
    mlm_res = analyze_stage2_run(mlm_doc, stage1_lookup=s1_lookup)
    assert ar_res["peaks_and_drops"]["megascale_overall"]["peak_epoch"] == 5
    assert ar_res["peaks_and_drops"]["megascale_overall"]["overfitting_drop"] == pytest.approx(-0.11)
    assert ar_res["peaks_and_drops"]["megascale_overall"]["max_post_peak_drop"] == pytest.approx(-0.16)
    assert ar_res["absorption_order_holds"] is True
    assert ar_res["res_takeoff_epoch"] is not None and mlm_res["res_takeoff_epoch"] is not None
    assert ar_res["res_takeoff_epoch"] < mlm_res["res_takeoff_epoch"]
    assert ar_res["demand_and_supply"]["predicted_e_star_from_stage1_supply"] is not None

    summary = summarize_supply_demand_campaign(runs_dir, stage1_path=s1_summary)
    assert summary["schema"] == ANALYSIS_SCHEMA
    assert summary["n_runs_analyzed"] == 2
    assert summary["n_runs_complete"] == 1
    assert summary["by_objective"]["ar"]["res_takeoff_epoch_median"] < summary["by_objective"]["mlm_15"]["res_takeoff_epoch_median"]
    assert summary["inversion_vs_delta_h_res_post_peak"]["megascale_overall"]["spearman_r"] < 0.0

    # Completed-only filtering
    comp_only = summarize_supply_demand_campaign(runs_dir, include_in_progress=False)
    assert comp_only["n_runs_analyzed"] == 1

    # End-to-end CLI script
    out_json = tmp_path / "stage2_summary.json"
    cli = REPO_ROOT / "scripts/capability/supply_demand/analyse_supply_demand.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(cli),
            "--runs-dir",
            str(runs_dir),
            "--stage1-path",
            str(s1_summary),
            "--out",
            str(out_json),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Analyzed 2 Stage-2 runs" in proc.stdout
    assert out_json.is_file()
    assert out_json.with_name(out_json.name + ".sha256").is_file()
