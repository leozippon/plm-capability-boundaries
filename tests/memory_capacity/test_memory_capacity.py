"""Comprehensive unit and end-to-end integration tests for 20-AA PLM parametric memory capacity."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from scipy.special import expit
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.memory_capacity import (
    CAPACITY_MULTIPLIERS,
    EXPOSURE_CHECKPOINTS,
    LOG2_20,
    SCALE_ORDER,
    SCALE_SPECS,
    SEQ_LEN,
    SPECIAL_TOKEN_ID,
    VOCAB_AA,
    RunSpec,
    UniformVocabTransformer,
    build_campaign_runs,
    count_model_parameters,
    dataset_entropy_bits,
    evaluate_ar_joint_bits,
    evaluate_mlm_joint_bits_16pt,
    generate_iid_aa_dataset,
    gibbs_memorizable_bits,
    null_ordered_entropy_bits,
    solve_sequence_count_for_entropy,
    train_memory_capacity_run,
)
from src.capability.memory_capacity.analysis import summarize_memory_capacity_campaign
from src.capability.memory_theory.io_utils import write_json_with_sha256

CAMPAIGN_SCRIPT = REPO_ROOT / "scripts/capability/memory_capacity/run_memory_capacity_campaign.py"
ANALYSE_SCRIPT = REPO_ROOT / "scripts/capability/memory_capacity/analyse_memory_capacity.py"


def test_scale_parameter_counts_and_ar_mlm_parity() -> None:
    expected_non_emb = {
        "0.5m": 492_608,
        "2m": 1_992_640,
        "8m": 7_992_128,
        "35m": 35_337_088,
        "150m": 148_671_360,
    }
    assert tuple( SCALE_ORDER ) == ("0.5m", "2m", "8m", "35m", "150m")

    for scale_id, exp_p in expected_non_emb.items():
        spec = SCALE_SPECS[scale_id]
        assert spec.exact_non_embedding_params == exp_p

        dev_ctx = torch.device("cpu") if scale_id in ("0.5m", "2m", "8m") else torch.device("meta")
        with dev_ctx:
            ar_model = UniformVocabTransformer(spec, "ar")
            mlm_model = UniformVocabTransformer(spec, "mlm")
        ar_counts = count_model_parameters(ar_model, spec)
        mlm_counts = count_model_parameters(mlm_model, spec)

        assert ar_counts["n_params_non_embedding"] == exp_p
        assert ar_counts == mlm_counts

    span_ratio = expected_non_emb["150m"] / expected_non_emb["0.5m"]
    assert span_ratio > 300.0


def test_entropy_accounting_solver_and_prefix_consistent_data(tmp_path: Path) -> None:
    for n in (1, 2, 17, 500, 25_000):
        expected_h = n * SEQ_LEN * math.log2(VOCAB_AA) - math.lgamma(n + 1) / math.log(2.0)
        assert dataset_entropy_bits(n) == pytest.approx(expected_h, rel=1e-12)
        if n >= 2:
            assert gibbs_memorizable_bits(n) < dataset_entropy_bits(n)
        else:
            assert gibbs_memorizable_bits(1) == pytest.approx(dataset_entropy_bits(1), rel=1e-12)

    prev_h = 0.0
    for n in (1, 2, 5, 25, 250, 2500):
        cur_h = dataset_entropy_bits(n)
        assert cur_h > prev_h
        prev_h = cur_h

    for target_bits in (500.0, 2_500.0, 125_000.0, 5_000_000.0):
        n_opt = solve_sequence_count_for_entropy(target_bits)
        assert dataset_entropy_bits(n_opt) >= target_bits
        if n_opt > 1:
            assert dataset_entropy_bits(n_opt - 1) < target_bits

    cache_dir = tmp_path / "cache"
    seqs_40 = generate_iid_aa_dataset(
        40,
        seq_len=SEQ_LEN,
        data_seed=20261127,
        data_cache_dir=cache_dir,
        chunk_size=16,
    )
    seqs_95 = generate_iid_aa_dataset(
        95,
        seq_len=SEQ_LEN,
        data_seed=20261127,
        data_cache_dir=cache_dir,
        chunk_size=16,
    )
    seqs_40_cached = generate_iid_aa_dataset(
        40,
        seq_len=SEQ_LEN,
        data_seed=20261127,
        data_cache_dir=cache_dir,
        chunk_size=16,
    )
    seqs_other_seed = generate_iid_aa_dataset(
        40,
        seq_len=SEQ_LEN,
        data_seed=20261199,
        chunk_size=16,
    )

    assert seqs_40.shape == (40, SEQ_LEN)
    assert seqs_40.dtype == np.uint8
    assert int(seqs_40.min()) >= 0 and int(seqs_40.max()) < VOCAB_AA
    assert np.array_equal(seqs_95[:40], seqs_40)
    assert np.array_equal(seqs_40_cached, seqs_40)
    assert not np.array_equal(seqs_other_seed, seqs_40)


def test_random_init_calibration_and_zero_target_token_leakage() -> None:
    test_len = 32
    torch.manual_seed(12345)
    ar_model = UniformVocabTransformer("0.5m", "ar", seq_len=test_len).eval()
    torch.manual_seed(12345)
    mlm_model = UniformVocabTransformer("0.5m", "mlm", seq_len=test_len).eval()

    counts = count_model_parameters(ar_model)
    seqs_np = generate_iid_aa_dataset(24, seq_len=test_len, data_seed=20261127)
    data_cpu = torch.from_numpy(seqs_np)

    ar_init = evaluate_ar_joint_bits(
        ar_model,
        data_cpu,
        batch_size=12,
        n_params_non_embedding=counts["n_params_non_embedding"],
        n_params_total=counts["n_params_total"],
        precision="fp32_strict",
    )
    mlm_init = evaluate_mlm_joint_bits_16pt(
        mlm_model,
        data_cpu,
        batch_size=12,
        n_params_non_embedding=counts["n_params_non_embedding"],
        n_params_total=counts["n_params_total"],
        precision="fp32_strict",
        eval_max_sequences_mlm=24,
    )

    assert abs(ar_init["mean_tok_xent_bits"] - LOG2_20) < 0.08
    assert abs(mlm_init["mean_tok_xent_bits"] - LOG2_20) < 0.08

    # 1. Strict AR zero target / future-token leakage check
    x = data_cpu[:4].to(dtype=torch.long)
    target_pos = 11
    x_mut = x.clone()
    # Mutate both the target position t and all future tokens > t
    x_mut[:, target_pos:] = (x[:, target_pos:] + 7) % VOCAB_AA

    bos = torch.full((x.shape[0], 1), SPECIAL_TOKEN_ID, dtype=torch.long)
    in_orig = torch.cat((bos, x[:, :-1]), dim=1)
    in_mut = torch.cat((bos, x_mut[:, :-1]), dim=1)

    with torch.no_grad():
        lp_orig = F.log_softmax(ar_model(in_orig), dim=-1)
        lp_mut = F.log_softmax(ar_model(in_mut), dim=-1)

    # Predictions at positions 0..target_pos must be completely invariant to target & future tokens
    assert torch.allclose(lp_orig[:, : target_pos + 1, :], lp_mut[:, : target_pos + 1, :], atol=1e-6)
    # Predictions after target_pos must change since prefix changed at target_pos
    assert not torch.allclose(lp_orig[:, target_pos + 1 :, :], lp_mut[:, target_pos + 1 :, :], atol=1e-4)

    # 2. Strict MLM zero masked-target leakage check
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, [3, target_pos, 19, 27]] = True
    x_mlm_mut = x.clone()
    x_mlm_mut[mask] = (x[mask] + 11) % VOCAB_AA

    masked_orig = torch.where(mask, SPECIAL_TOKEN_ID, x)
    masked_mut = torch.where(mask, SPECIAL_TOKEN_ID, x_mlm_mut)
    assert torch.equal(masked_orig, masked_mut)

    with torch.no_grad():
        mlm_lp_orig = F.log_softmax(mlm_model(masked_orig), dim=-1)
        mlm_lp_mut = F.log_softmax(mlm_model(masked_mut), dim=-1)
    assert torch.equal(mlm_lp_orig, mlm_lp_mut)

    # Mutating an unmasked context token DOES alter bidirectional predictions at masked target_pos
    x_ctx_mut = x.clone()
    x_ctx_mut[:, 20] = (x[:, 20] + 5) % VOCAB_AA  # position 20 > target_pos=11 is unmasked
    masked_ctx_mut = torch.where(mask, SPECIAL_TOKEN_ID, x_ctx_mut)
    with torch.no_grad():
        mlm_lp_ctx_mut = F.log_softmax(mlm_model(masked_ctx_mut), dim=-1)
    assert not torch.allclose(
        mlm_lp_orig[:, target_pos, :], mlm_lp_ctx_mut[:, target_pos, :], atol=1e-5
    )


def test_end_to_end_tiny_training_and_analysis_roundtrip(tmp_path: Path) -> None:
    out_dir = tmp_path / "tiny_campaign"
    test_seq_len = 32
    for paradigm in ("ar", "mlm"):
        for mult, n_seq in ((0.5, 8), (1.0, 12), (2.0, 16)):
            spec = RunSpec(
                run_id=f"tiny_{paradigm}_0p5m_r{str(mult).replace('.', 'p')}",
                track="primary",
                paradigm=paradigm,
                scale_id="0.5m",
                multiplier=mult,
                n_sequences=n_seq,
                max_epochs=2,
                eval_epochs=(1, 2),
                weight_decay=0.0,
                precision="fp32_strict",
                seed=20261128,
                data_seed=20261127,
                save_checkpoint=False,
                ddp_preferred=False,
            )
            res = train_memory_capacity_run(
                spec,
                out_dir=out_dir,
                device="cpu",
                use_compile=False,
                seq_len=test_seq_len,
                batch_size_override=8,
                eval_max_sequences_mlm=16,
                loo_max_sequences_mlm=8,
                loo_probes_per_seq=4,
            )
            assert res["status"] == "complete"
            assert res["final_eval"]["mean_tok_xent_bits"] < res["init_eval"]["mean_tok_xent_bits"]

    summary_path = out_dir / "parametric_capacity_summary.json"
    summary = summarize_memory_capacity_campaign(
        out_dir / "runs",
        out_path=summary_path,
        bootstrap_draws=200,
    )
    assert summary_path.is_file()
    assert summary_path.with_name(summary_path.name + ".sha256").is_file()
    assert summary["provenance"]["n_completed_runs"] == 6
    assert "0.5m" in summary["saturation_curves_by_scale_exposure"]["ar"]
    assert "0.5m" in summary["saturation_curves_by_scale_exposure"]["mlm"]
    assert summary["ar_vs_mlm_comparison"]["n_total_paired_cells"] == 6


def _make_synthetic_receipt(
    *,
    run_id: str,
    track: str,
    paradigm: str,
    scale_id: str,
    multiplier: float,
    max_epochs: int,
    eval_epochs: tuple[int, ...],
    c_max_true: float,
    e_50_true: float,
    gamma_true: float = 4.0,
    weight_decay: float = 0.0,
    precision: str = "bf16",
    bpp_shift: float = 0.0,
) -> dict[str, Any]:
    spec = SCALE_SPECS[scale_id]
    p_ne = spec.exact_non_embedding_params
    p_tot = spec.exact_total_params
    p_emb = spec.exact_embedding_params

    n_seq = max(4, int(round((multiplier * 3.5 * p_ne) / (SEQ_LEN * LOG2_20))))
    h_null = null_ordered_entropy_bits(n_seq)
    h_data = dataset_entropy_bits(n_seq)
    h_gibbs = gibbs_memorizable_bits(n_seq)

    exposures: list[dict[str, Any]] = []
    for ep in eval_epochs:
        hill = float(expit(1.0 * (math.log(float(ep)) - math.log(e_50_true))))
        c_ep = max(0.05, (c_max_true + bpp_shift) * hill)
        cap_ep_bits = c_ep * p_ne
        # Smooth 2-regime CES saturation M(H) = C / (1 + (C / H)^gamma)^(1/gamma)
        mem_bits = cap_ep_bits / ((1.0 + (cap_ep_bits / h_data) ** gamma_true) ** (1.0 / gamma_true))
        total_xent = h_null - mem_bits
        mean_seq_xent = total_xent / n_seq
        mean_tok_xent = mean_seq_xent / SEQ_LEN

        quad_profile = None
        loo_tok_xent = None
        loo_mem_bits = None
        if paradigm == "mlm":
            rates = (np.arange(16, dtype=np.float64) + 0.5) / 16.0
            saved_mean = LOG2_20 - mean_tok_xent
            # More bits saved at low mask rate, approaching 0 saved as m -> 1
            weights = 2.0 * (1.0 - rates)
            quad_bpt = LOG2_20 - saved_mean * weights
            quad_profile = [float(x) for x in quad_bpt]
            loo_mem_bits = mem_bits * 1.32
            loo_tok_xent = LOG2_20 - (loo_mem_bits / (n_seq * SEQ_LEN))

        exposures.append(
            {
                "epoch": ep,
                "step": ep * 10,
                "tokens_seen": ep * n_seq * SEQ_LEN,
                "lr": 1e-3,
                "elapsed_sec": float(ep) * 0.1,
                "train_loss_last_epoch_bits_per_tok": mean_tok_xent,
                "eval_n_sequences": n_seq,
                "mean_seq_xent_bits": mean_seq_xent,
                "mean_tok_xent_bits": mean_tok_xent,
                "eval_mc_se_seq_bits": 0.01,
                "total_model_xent_bits": total_xent,
                "memorized_bits": mem_bits,
                "memorized_bits_se": 0.01 * n_seq,
                "bits_per_non_emb_param": mem_bits / p_ne,
                "bits_per_total_param": mem_bits / p_tot,
                "fraction_of_H_data": mem_bits / h_data,
                "fraction_of_H_gibbs": mem_bits / h_gibbs,
                "mlm_quadrature_16pt_bits_per_tok": quad_profile,
                "mlm_loo_pseudo_tok_xent_bits": loo_tok_xent,
                "mlm_loo_pseudo_memorized_bits": loo_mem_bits,
            }
        )

    return {
        "schema": "plm_parametric_capacity_run_v1",
        "status": "complete",
        "run_id": run_id,
        "track": track,
        "paradigm": paradigm,
        "scale_id": scale_id,
        "multiplier": multiplier,
        "n_sequences": n_seq,
        "seq_len": SEQ_LEN,
        "vocab_size": VOCAB_AA,
        "n_params_non_embedding": p_ne,
        "n_params_embedding": p_emb,
        "n_params_total": p_tot,
        "weight_decay": weight_decay,
        "precision": precision,
        "batch_size_global": 64,
        "peak_lr": 1e-3,
        "min_lr": 5e-5,
        "warmup_steps": 20,
        "total_steps": max_epochs * 10,
        "max_epochs": max_epochs,
        "seed": 20261128,
        "data_seed": 20261127,
        "world_size": 1,
        "H_null_ordered_bits": h_null,
        "H_data_bits": h_data,
        "H_gibbs_max_bits": h_gibbs,
        "exposures": exposures,
    }


def test_synthetic_5scale_campaign_curve_fitting_and_ablations(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)

    c_ar_true = 3.60
    c_mlm_true = 2.70
    e50_ar_true = 12.0
    e50_mlm_true = 18.0

    # Mirror spec.py's exact 5-scale campaign horizons (including 35m e=200@2x and 150m e=25@2x)
    campaign_specs = build_campaign_runs(include_ablations=True)
    for rspec in campaign_specs:
        c_true = c_ar_true if rspec.paradigm == "ar" else c_mlm_true
        e50_true = e50_ar_true if rspec.paradigm == "ar" else e50_mlm_true
        shift = 0.0
        if rspec.track == "ablation_wd":
            shift = -0.08
        elif rspec.track == "ablation_fp32":
            shift = 0.01
        elif rspec.track == "ablation_lr_horizon":
            shift = 0.05

        doc = _make_synthetic_receipt(
            run_id=rspec.run_id,
            track=rspec.track,
            paradigm=rspec.paradigm,
            scale_id=rspec.scale_id,
            multiplier=rspec.multiplier,
            max_epochs=rspec.max_epochs,
            eval_epochs=rspec.eval_epochs,
            c_max_true=c_true,
            e_50_true=e50_true,
            weight_decay=rspec.weight_decay,
            precision=rspec.precision,
            bpp_shift=shift,
        )
        write_json_with_sha256(runs_dir / f"{rspec.run_id}.json", doc)

    out_summary = tmp_path / "parametric_capacity_summary.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ANALYSE_SCRIPT),
            "--runs-dir",
            str(runs_dir),
            "--out",
            str(out_summary),
            "--bootstrap-draws",
            "300",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Native Parametric Memory Capacity Summary" in proc.stdout
    summary = json.loads(out_summary.read_text(encoding="utf-8"))

    # 1. Direct e=400 calibration strictly uses 0.5m, 2m, 8m (excludes 35m which only has <=0.5x at e=400)
    ar_cal = summary["c_max_calibration"]["by_paradigm"]["ar"]
    mlm_cal = summary["c_max_calibration"]["by_paradigm"]["mlm"]
    assert ar_cal["direct_anchor_unadjusted"]["non_embedding_params"]["scale_ids"] == [
        "0.5m",
        "2m",
        "8m",
    ]
    c_ar_est = ar_cal["direct_anchor_unadjusted"]["non_embedding_params"]["zero_intercept_ratio"][
        "c_max"
    ]
    c_mlm_est = mlm_cal["direct_anchor_unadjusted"]["non_embedding_params"]["zero_intercept_ratio"][
        "c_max"
    ]
    alpha_ar = ar_cal["direct_anchor_unadjusted"]["non_embedding_params"]["log_log_scaling"][
        "alpha_exponent"
    ]
    assert c_ar_est == pytest.approx(c_ar_true * (400.0 / 412.0), rel=0.03)
    assert c_mlm_est == pytest.approx(c_mlm_true * (400.0 / 418.0), rel=0.03)
    assert alpha_ar == pytest.approx(1.0, abs=0.01)

    # 2. Secondary 5-scale exposure-anchored calibration spans >300x across all 5 scales
    anch_ar = ar_cal["exposure_anchored_multi_scale"]["non_embedding_params"]
    assert anch_ar["scale_ids"] == ["0.5m", "2m", "8m", "35m", "150m"]
    assert anch_ar["scale_span_ratio"] > 300.0
    assert anch_ar["zero_intercept_ratio"]["c_max"] == pytest.approx(c_ar_est, rel=0.03)

    # 3. CES saturation curve & Hill exposure-discount law recovery
    ces_2m = summary["saturation_curves_by_scale_exposure"]["ar"]["2m"]["400"]["ces_saturation_fit"]
    assert ces_2m is not None
    assert ces_2m["gamma_sharpness"] == pytest.approx(4.0, rel=0.15)
    assert ces_2m["r2"] > 0.995

    hill_ar = summary["exposure_discount_curves"]["ar"]["pooled_hill_fit_at_2x"]
    assert hill_ar is not None
    assert len(summary["exposure_discount_curves"]["ar"]["pooled_by_exposure"]) == len(
        EXPOSURE_CHECKPOINTS
    )
    assert hill_ar["e_50_epochs"] == pytest.approx(e50_ar_true, rel=0.15)
    assert hill_ar["r2"] > 0.995

    # 4. Paired AR vs MLM, masking profile, and LOO pseudo-entropy inflation (>1.25x)
    comp = summary["ar_vs_mlm_comparison"]
    diff_400 = comp["asymptotic_e400_saturated_ge2x"]["ar_minus_mlm_bits_per_param"]["point"]
    assert diff_400 > 0.7
    loo_ratio = comp["loo_inflation_audit"]["primary_mlm_observations"][
        "aggregate_inflation_ratio_sum_loo_over_sum_exact"
    ]["point"]
    assert loo_ratio == pytest.approx(1.32, rel=0.02)
    reg = comp["masking_rate_profile"]["regime_summary"]
    assert reg["low_mask_lt_0p25_mean_saved_bpt"] > reg["high_mask_ge_0p75_mean_saved_bpt"]

    # 5. All 3 ablations populated with expected directional signs
    abl = summary["ablations"]
    assert abl["weight_decay_0_vs_0p01"]["overall_delta_bits_per_param"]["point"] < -0.02
    assert abl["precision_bf16_vs_fp32_strict"]["overall_delta_bits_per_param"]["point"] > 0.0
    assert (
        abl["cosine_single_run_vs_dedicated_horizon"]["overall_dedicated_minus_trajectory_bpp"][
            "point"
        ]
        > 0.0
    )


def test_cli_dry_run_and_forbidden_infra_hygiene(tmp_path: Path) -> None:
    dry_dir = tmp_path / "dry_out"
    res_6gpu = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN_SCRIPT),
            "--out-dir",
            str(dry_dir),
            "--gpus",
            "0,1,2,3,4,5",
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    all_runs = build_campaign_runs(include_ablations=True)
    exp_p1 = sum(1 for r in all_runs if not r.ddp_preferred)
    exp_p2 = sum(1 for r in all_runs if r.ddp_preferred)
    assert f"Matched jobs:         {len(all_runs)}" in res_6gpu.stdout
    assert f"Phase 1 (1-GPU LPT):  {exp_p1} jobs" in res_6gpu.stdout
    assert f"Phase 2 (DDP 6-GPU): {exp_p2} jobs" in res_6gpu.stdout
    assert not dry_dir.exists()

    # Single-GPU mode places all jobs into Phase 1 and 0 into Phase 2
    res_1gpu = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN_SCRIPT),
            "--out-dir",
            str(dry_dir),
            "--gpus",
            "0",
            "--scales",
            "0.5m,35m",
            "--tracks",
            "primary",
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"Phase 1 (1-GPU LPT):  {2 * len(CAPACITY_MULTIPLIERS) * 2} jobs" in res_1gpu.stdout
    assert "Phase 2 (DDP 1-GPU): 0 jobs" in res_1gpu.stdout

    # Infrastructure hygiene check: pod name or internal IP in receipt must raise ValueError
    bad_dir = tmp_path / "bad_runs"
    bad_dir.mkdir(parents=True)
    bad_doc = _make_synthetic_receipt(
        run_id="primary_ar_0p5m_r2p00",
        track="primary",
        paradigm="ar",
        scale_id="0.5m",
        multiplier=2.0,
        max_epochs=10,
        eval_epochs=(1, 10),
        c_max_true=3.5,
        e_50_true=10.0,
    )
    bad_doc["leaked_host"] = "plm-train-worker-0"
    write_json_with_sha256(bad_dir / "bad.json", bad_doc)
    bad_out = tmp_path / "should_not_exist.json"
    with pytest.raises(ValueError, match="forbidden"):
        summarize_memory_capacity_campaign(bad_dir, out_path=bad_out)
    assert not bad_out.exists()

    # SHA-256 sidecar tamper detection
    tamper_dir = tmp_path / "tamper_runs"
    tamper_dir.mkdir(parents=True)
    clean_doc = _make_synthetic_receipt(
        run_id="primary_ar_0p5m_r2p00",
        track="primary",
        paradigm="ar",
        scale_id="0.5m",
        multiplier=2.0,
        max_epochs=10,
        eval_epochs=(1, 10),
        c_max_true=3.5,
        e_50_true=10.0,
    )
    t_file = tamper_dir / "clean.json"
    write_json_with_sha256(t_file, clean_doc)
    tampered_doc = dict(clean_doc)
    tampered_doc["multiplier"] = 4.0
    t_file.write_text(json.dumps(tampered_doc, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 sidecar mismatch"):
        summarize_memory_capacity_campaign(tamper_dir)
