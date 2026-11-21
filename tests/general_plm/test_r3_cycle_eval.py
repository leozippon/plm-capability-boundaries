"""Unit and exact numerical-fidelity tests for R3 pairwise double-mutant cycle evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.general_plm.r3_cycle_eval import (
    INTERACTION_RUNGS,
    _fit_designs_cached_folds,
    evaluate_r3_pairwise_cycles_npz,
    load_pairwise_cycle_context,
)


def test_r3_cycle_control_rungs_match_archived_interaction_lane() -> None:
    """Control-only MSEs across all 3 rungs must match archived progen3-3b cells to <1e-9."""
    ctx = load_pairwise_cycle_context(ROOT)
    assert len(ctx.group_order) == 61
    assert len(ctx.panel["epsilon"]) == 7805
    assert ctx.raw_all_epsilon.shape == (8192,)

    # Run only the 3 control rungs and compare to archived interaction lane cells.
    designs = {f"ctrl::{k}": v for k, v in INTERACTION_RUNGS.items()}
    fit_out = _fit_designs_cached_folds(ctx, ctx.panel["blocks"], designs)

    cells_root = ROOT / "results/extensions/explicit_statistics_lanes_20261110/interaction/cells"
    if not cells_root.is_dir():
        pytest.skip("Archived interaction cells not present locally")

    archived_uc: dict[str, float] = {}
    for path in sorted(cells_root.rglob("*progen3-3b*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("support") == "complete_case" and doc.get("rung") in INTERACTION_RUNGS:
            archived_uc[doc["rung"]] = float(np.mean(doc["values"]["U_C"]))

    assert set(archived_uc) == set(INTERACTION_RUNGS)
    for rung, expected_mse in archived_uc.items():
        got_mse = float(np.mean(fit_out["avg_mse"][f"ctrl::{rung}"]))
        assert abs(got_mse - expected_mse) < 1e-9, (
            f"Rung {rung}: got {got_mse:.10f}, archived {expected_mse:.10f}"
        )


def test_r3_cycle_eval_on_real_dplm_150m_and_summary_cli(tmp_path: Path) -> None:
    """End-to-end CPU evaluation on real dplm-150m R3 npz and fit_general_plm_summary CLI."""
    npz_path = (
        ROOT
        / "results/shared/external_baseline/20261120213246_766da841c1a8/"
        "dplm_150m_r3_all/r3_pairwise_cycles_dplm-150m.npz"
    )
    if not npz_path.is_file():
        pytest.skip("dplm-150m R3 npz not present locally")

    ctx = load_pairwise_cycle_context(ROOT)
    res = evaluate_r3_pairwise_cycles_npz(
        npz_path, "dplm-150m", ctx=ctx, bootstrap_draws=200, include_r_cycle=False
    )
    assert res["support"]["n_groups"] == 61
    assert res["support"]["n_cycles"] == 7805
    assert np.isfinite(
        res["zero_shot"]["all_64_groups_raw_epsilon"]["within_background_spearman"]["point"]
    )
    assert "M_cycle" in res["nested_cv_channels"]
    for rung in INTERACTION_RUNGS:
        fwd = res["nested_cv_channels"]["M_cycle"][rung]["forward_model_given_controls"]
        assert np.isfinite(fwd["mse_reduction_kcal2"]["point"])
        assert np.isfinite(fwd["spearman_increment"]["point"])

    # Verify full production summary & R_cycle artifact if already generated.
    prod_eval = ROOT / "results/extensions/general_plm_20261120/r3_cycle_eval_dplm-150m.json"
    if prod_eval.is_file():
        pdoc = json.loads(prod_eval.read_text(encoding="utf-8"))
        assert "R_cycle" in pdoc["nested_cv_channels"]
        assert "MR_cycle_over_nuisance_profile_couplings" in pdoc["nested_cv_channels"]

    # Smoke-test CLI into tmp_path (skipping full refit by passing only V1 dir + custom out).
    v1_dir = ROOT / "results/shared/external_baseline/20261120133411_4956b7c48429"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/capability/general_plm/fit_general_plm_summary.py"),
            "--results-dir",
            str(v1_dir),
            "--out",
            str(tmp_path),
            "--skip-r3-cycle-fit",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "[summary]" in proc.stdout
    out_json = tmp_path / "general_plm_summary.json"
    assert out_json.is_file()
    assert out_json.with_name(out_json.name + ".sha256").is_file()
    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["frozen_33_arm_panel_mutated"] is False
    assert any(a["arm"] == "protgpt3-msa" for a in summary["arms"])
