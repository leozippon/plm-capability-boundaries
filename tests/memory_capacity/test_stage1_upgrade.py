"""Unit and end-to-end tests for Stage-1 upgrades: MLM-15%, OA SE fix, WSD + early stop,
mid-epoch SIGTERM resume, legacy Batch-1 compatibility, e_crit analysis, and Spot leases.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from scipy.special import expit
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.memory_capacity import (
    EVAL_EPOCHS,
    LOG2_20,
    PARADIGMS,
    PREEMPTION_EXIT_CODE,
    SCALE_SPECS,
    SEQ_LEN,
    VOCAB_AA,
    AdaptiveEarlyStopConfig,
    AdaptiveEarlyStopTracker,
    RunSpec,
    SharedFSLeaseManager,
    UniformVocabTransformer,
    canonicalize_paradigm,
    choose_global_batch_size,
    compute_cooldown_epochs,
    compute_wsd_lr,
    count_model_parameters,
    dataset_entropy_bits,
    evaluate_mlm_fixed_rate_bits,
    evaluate_mlm_joint_bits_16pt,
    generate_iid_aa_dataset,
    generate_worker_token,
    gibbs_memorizable_bits,
    load_run_manifest,
    mlm_fixed_mask_count,
    null_ordered_entropy_bits,
    sample_mlm_fixed_rate_batch,
    train_memory_capacity_run,
    vram_safe_micro_batch_cap,
)
from src.capability.memory_capacity.analysis import (
    estimate_ecrit_nonparametric,
    fit_delayed_hill_exposure_curve,
    summarize_memory_capacity_campaign,
)
from src.capability.memory_theory.io_utils import write_json_with_sha256

SPOT_WORKER_SCRIPT = REPO_ROOT / "scripts/capability/memory_capacity/run_stage1_spot_worker.py"


def test_three_paradigms_legacy_alias_and_tiny_batch_capping() -> None:
    assert PARADIGMS == ("ar", "mlm_15", "oa_uniform")
    assert canonicalize_paradigm("mlm") == "oa_uniform"
    assert canonicalize_paradigm("OA_UNIFORM") == "oa_uniform"
    assert canonicalize_paradigm("mlm_15") == "mlm_15"
    assert mlm_fixed_mask_count(256, 0.15) == 38
    assert max(EVAL_EPOCHS) == 10_000

    # Batch-size capping guarantees >= min_steps_per_epoch on tiny datasets (e.g. 0.5m r=0.25)
    bs_tiny = choose_global_batch_size(393, 512, world_size=1, min_steps_per_epoch=16)
    assert math.ceil(393 / bs_tiny) >= 16
    bs_32_steps = choose_global_batch_size(393, 512, world_size=1, min_steps_per_epoch=32)
    assert math.ceil(393 / bs_32_steps) >= 32
    bs_ddp = choose_global_batch_size(120, 512, world_size=6, min_steps_per_epoch=16)
    assert bs_ddp % 6 == 0 and math.ceil(120 / bs_ddp) >= 16

    # Fresh independent masks across consecutive dynamic mlm_15 calls
    seqs = torch.from_numpy(generate_iid_aa_dataset(8, seq_len=64, data_seed=77))
    gen = torch.Generator().manual_seed(99)
    _, _, mask1, n1 = sample_mlm_fixed_rate_batch(seqs, mask_count=10, generator=gen)
    _, _, mask2, n2 = sample_mlm_fixed_rate_batch(seqs, mask_count=10, generator=gen)
    assert torch.all(n1 == 10) and torch.all(mask1.sum(dim=-1) == 10)
    assert torch.all(n2 == 10) and torch.all(mask2.sum(dim=-1) == 10)
    assert not torch.equal(mask1, mask2)


def test_mlm15_sw_pseudo_bounds_and_oa_se_nonzero_when_full_population() -> None:
    test_len = 64
    k = mlm_fixed_mask_count(test_len, 0.15)  # round(9.6) = 10
    assert k == 10

    torch.manual_seed(42)
    model = UniformVocabTransformer("0.5m", "mlm_15", seq_len=test_len).eval()
    counts = count_model_parameters(model)
    p_ne = counts["n_params_non_embedding"]
    n_seq = 16
    data = torch.from_numpy(generate_iid_aa_dataset(n_seq, seq_len=test_len, data_seed=101))

    # 1. MLM-15% fixed-rate evaluator: exact SW & pseudo formulas + nonzero within-seq SE at N == eval_max
    res_15 = evaluate_mlm_fixed_rate_bits(
        model,
        data,
        batch_size=8,
        n_params_non_embedding=p_ne,
        n_params_total=counts["n_params_total"],
        mask_count=k,
        num_draws=16,
        precision="fp32_strict",
        eval_max_sequences_mlm=n_seq,
    )
    h15 = res_15["mean_masked_token_xent_bits"]
    assert res_15["token_xent_bits"] == pytest.approx(h15, rel=1e-12)
    assert res_15["mean_tok_xent_bits"] == pytest.approx(h15, rel=1e-12)
    assert res_15["sw_memorized_bits"] == pytest.approx(n_seq * k * (LOG2_20 - h15), rel=1e-12)
    assert res_15["sw_bits_per_non_emb_param"] == pytest.approx(
        res_15["sw_memorized_bits"] / p_ne, rel=1e-12
    )
    assert res_15["pseudo_memorized_bits"] == pytest.approx(
        n_seq * test_len * (LOG2_20 - h15), rel=1e-12
    )
    assert res_15["pseudo_bits_per_non_emb_param"] == pytest.approx(
        res_15["pseudo_memorized_bits"] / p_ne, rel=1e-12
    )
    assert res_15["bits_per_non_emb_param"] == pytest.approx(
        res_15["pseudo_bits_per_non_emb_param"], rel=1e-12
    )
    # Full population (fpc == 0) must still have positive within-sequence mask-draw SE
    assert res_15["eval_between_seq_se_tok_bits"] == pytest.approx(0.0, abs=1e-15)
    assert res_15["eval_within_seq_se_tok_bits"] > 0.0
    assert res_15["mean_masked_token_xent_se_bits"] == pytest.approx(
        res_15["eval_within_seq_se_tok_bits"], rel=1e-12
    )
    assert res_15["sw_memorized_bits_se"] > 0.0

    # 2. OA-Uniform 16-pt evaluator: fpc=0 must NOT zero out within-sequence 16-stratum SE
    res_oa = evaluate_mlm_joint_bits_16pt(
        model,
        data,
        batch_size=8,
        n_params_non_embedding=p_ne,
        n_params_total=counts["n_params_total"],
        precision="fp32_strict",
        eval_max_sequences_mlm=n_seq,
    )
    assert res_oa["paradigm"] == "oa_uniform"
    assert res_oa["eval_between_seq_se_seq_bits"] == pytest.approx(0.0, abs=1e-15)
    assert res_oa["eval_within_seq_se_seq_bits"] > 0.0
    assert res_oa["eval_mc_se_seq_bits"] == pytest.approx(
        res_oa["eval_within_seq_se_seq_bits"], rel=1e-12
    )
    assert res_oa["memorized_bits_se"] > 0.0


def test_wsd_schedule_and_two_phase_adaptive_early_stopping() -> None:
    # WSD LR profile: warmup -> stable -> cosine cooldown
    assert compute_cooldown_epochs(50) == 20
    assert compute_cooldown_epochs(500) == 50
    assert compute_cooldown_epochs(5000) == 200

    lr_warm = compute_wsd_lr(5, peak_lr=1e-3, warmup_steps=10)
    lr_stable1 = compute_wsd_lr(50, peak_lr=1e-3, warmup_steps=10)
    lr_stable2 = compute_wsd_lr(500, peak_lr=1e-3, warmup_steps=10)
    lr_cd_mid = compute_wsd_lr(
        150, peak_lr=1e-3, warmup_steps=10, cooldown_start_step=100, cooldown_end_step=200
    )
    lr_cd_end = compute_wsd_lr(
        200, peak_lr=1e-3, warmup_steps=10, cooldown_start_step=100, cooldown_end_step=200
    )
    assert lr_warm == pytest.approx(5e-4)
    assert lr_stable1 == pytest.approx(1e-3)
    assert lr_stable2 == pytest.approx(1e-3)
    assert lr_cd_mid == pytest.approx(0.525e-3)
    assert lr_cd_end == pytest.approx(0.05e-3)

    # Phase A: flat pre-takeoff evals must NEVER plateau-stop
    cfg = AdaptiveEarlyStopConfig(enabled=True)
    tracker = AdaptiveEarlyStopTracker(config=cfg)
    for ep in (1, 2, 5, 10, 20, 50, 100, 200, 400):
        reason = tracker.update(
            epoch=ep,
            eval_rec={
                "mean_tok_xent_bits": LOG2_20 - 0.01,  # delta_bpt = 0.01 < 0.05
                "bits_per_non_emb_param": 0.002,
                "fraction_of_H_gibbs": 0.001,
            },
            paradigm="oa_uniform",
            global_step=ep * 10,
            steps_per_epoch=10,
            max_epochs=2000,
        )
        assert reason is None
        assert not tracker.takeoff_detected
        assert not tracker.cooldown_triggered

    # Phase A -> Phase B takeoff requires 2 consecutive evals with delta_bpt >= 0.05
    r1 = tracker.update(
        epoch=500,
        eval_rec={"mean_tok_xent_bits": LOG2_20 - 0.2, "bits_per_non_emb_param": 0.5, "fraction_of_H_gibbs": 0.2},
        paradigm="oa_uniform",
        global_step=5000,
        steps_per_epoch=10,
        max_epochs=4000,
    )
    assert r1 is None and not tracker.takeoff_detected
    r2 = tracker.update(
        epoch=600,
        eval_rec={"mean_tok_xent_bits": LOG2_20 - 0.8, "bits_per_non_emb_param": 1.8, "fraction_of_H_gibbs": 0.6},
        paradigm="oa_uniform",
        global_step=6000,
        steps_per_epoch=10,
        max_epochs=4000,
    )
    assert r2 is None and tracker.takeoff_detected and tracker.e_takeoff == 600

    # Capacity plateau before 2*e_takeoff (1200) must NOT trigger; triggers once >= 1200 & >= 1.5x span
    r3 = tracker.update(
        epoch=800,
        eval_rec={"mean_tok_xent_bits": LOG2_20 - 1.2, "bits_per_non_emb_param": 2.40, "fraction_of_H_gibbs": 0.70},
        paradigm="oa_uniform",
        global_step=8000,
        steps_per_epoch=10,
        max_epochs=4000,
    )
    assert r3 is None
    r4 = tracker.update(
        epoch=1000,
        eval_rec={"mean_tok_xent_bits": LOG2_20 - 1.2, "bits_per_non_emb_param": 2.401, "fraction_of_H_gibbs": 0.70},
        paradigm="oa_uniform",
        global_step=10000,
        steps_per_epoch=10,
        max_epochs=4000,
    )
    assert r4 is None  # < 2 * e_takeoff
    r5 = tracker.update(
        epoch=1300,
        eval_rec={"mean_tok_xent_bits": LOG2_20 - 1.2, "bits_per_non_emb_param": 2.402, "fraction_of_H_gibbs": 0.70},
        paradigm="oa_uniform",
        global_step=13000,
        steps_per_epoch=10,
        max_epochs=4000,
    )
    assert r5 == "capacity_plateau"
    assert tracker.cooldown_triggered
    assert tracker.cooldown_trigger_epoch == 1300
    assert tracker.stop_epoch == 1300 + 130  # round(0.1 * 1300) = 130

    # Data ceiling trigger check
    ceil_tracker = AdaptiveEarlyStopTracker(config=cfg)
    ceil_tracker.update(
        epoch=1,
        eval_rec={"mean_tok_xent_bits": 1.0, "bits_per_non_emb_param": 0.6, "fraction_of_H_gibbs": 0.7},
        paradigm="ar",
        global_step=10,
        steps_per_epoch=10,
        max_epochs=200,
    )
    rc = ceil_tracker.update(
        epoch=2,
        eval_rec={"mean_tok_xent_bits": 0.02, "bits_per_non_emb_param": 0.87, "fraction_of_H_gibbs": 0.996},
        paradigm="ar",
        global_step=20,
        steps_per_epoch=10,
        max_epochs=200,
    )
    assert rc == "data_ceiling"
    assert ceil_tracker.stop_epoch == 2 + 20

    # MLM-15% data ceiling MUST gate on masked-token xent <= 0.04 bpt, NOT pseudo/H_gibbs >= 0.995
    mlm15_tracker = AdaptiveEarlyStopTracker(config=cfg)
    mlm15_tracker.update(
        epoch=5,
        eval_rec={"mean_tok_xent_bits": 1.0, "bits_per_non_emb_param": 0.6, "fraction_of_H_gibbs": 0.7},
        paradigm="mlm_15",
        global_step=50,
        steps_per_epoch=10,
        max_epochs=200,
    )
    mlm15_tracker.update(
        epoch=10,
        eval_rec={"mean_tok_xent_bits": 0.2, "bits_per_non_emb_param": 0.85, "fraction_of_H_gibbs": 0.98},
        paradigm="mlm_15",
        global_step=100,
        steps_per_epoch=10,
        max_epochs=200,
    )
    assert mlm15_tracker.takeoff_detected and not mlm15_tracker.cooldown_triggered
    # pseudo/H_gibbs = 1.02 >= 0.995 but h_15 = 0.087 > 0.04 -> must NOT trigger data_ceiling
    r_no_ceil = mlm15_tracker.update(
        epoch=20,
        eval_rec={"mean_tok_xent_bits": 0.087, "bits_per_non_emb_param": 0.91, "fraction_of_H_gibbs": 1.02},
        paradigm="mlm_15",
        global_step=200,
        steps_per_epoch=10,
        max_epochs=200,
    )
    assert r_no_ceil is None and not mlm15_tracker.cooldown_triggered
    # h_15 = 0.035 <= 0.04 -> MUST trigger data_ceiling
    r_yes_ceil = mlm15_tracker.update(
        epoch=35,
        eval_rec={"mean_tok_xent_bits": 0.035, "bits_per_non_emb_param": 0.92, "fraction_of_H_gibbs": 1.03},
        paradigm="mlm_15",
        global_step=350,
        steps_per_epoch=10,
        max_epochs=200,
    )
    assert r_yes_ceil == "data_ceiling" and mlm15_tracker.cooldown_triggered


def test_end_to_end_wsd_early_stop_cooldown_and_mlm15_diagnostics(tmp_path: Path) -> None:
    out_dir = tmp_path / "wsd_early_stop"
    spec = RunSpec(
        run_id="test_mlm15_wsd_early",
        track="primary",
        paradigm="mlm_15",
        scale_id="0.5m",
        multiplier=0.25,
        n_sequences=8,
        max_epochs=40,
        eval_epochs=tuple(range(1, 41)),
        weight_decay=0.0,
        precision="fp32_strict",
        seed=20261128,
        data_seed=20261127,
        save_checkpoint=False,
        ddp_preferred=False,
        lr_schedule="wsd",
        adaptive_early_stop=True,
        delete_transient_ckpt_on_complete=True,
    )
    # Use a permissive data-ceiling threshold so tiny CPU run triggers cooldown quickly
    cfg = AdaptiveEarlyStopConfig(
        enabled=True,
        takeoff_delta_bpt=0.01,
        takeoff_consecutive_evals=1,
        mlm_xent_ceiling_bpt=LOG2_20,  # triggers data_ceiling right after takeoff
        cooldown_min_epochs=2,
        cooldown_max_epochs=5,
    )
    res = train_memory_capacity_run(
        spec,
        out_dir=out_dir,
        device="cpu",
        use_compile=False,
        seq_len=32,
        batch_size_override=4,
        eval_max_sequences_mlm=8,
        loo_max_sequences_mlm=8,
        loo_probes_per_seq=4,
        mlm_fixed_num_draws=4,
        checkpoint_interval_sec=0.01,  # force transient latest.pt creation during run
        adaptive_stop_config=cfg,
    )
    assert res["status"] == "complete"
    assert res["stopped_early"] is True
    assert res["stop_reason"] == "data_ceiling"
    assert res["epoch_done"] < spec.max_epochs
    assert res["final_eval"]["lr"] == pytest.approx(res["min_lr"], rel=1e-4)
    assert res["mlm_loo_diagnostic"] is not None
    assert res["mlm_oa_16pt_diagnostic"] is not None
    assert res["final_eval"]["sw_bits_per_non_emb_param"] is not None
    assert res["final_eval"]["pseudo_bits_per_non_emb_param"] is not None
    # Transient checkpoint must be deleted on clean completion
    assert not (out_dir / "checkpoints" / f"{spec.run_id}_latest.pt").exists()


def test_mid_epoch_sigterm_flush_exact_resume_and_spec_hash_guard(tmp_path: Path) -> None:
    spec = RunSpec(
        run_id="test_mid_epoch_resume",
        track="primary",
        paradigm="mlm_15",
        scale_id="0.5m",
        multiplier=0.25,
        n_sequences=12,
        max_epochs=2,
        eval_epochs=(1, 2),
        weight_decay=0.0,
        precision="fp32_strict",
        seed=20261128,
        data_seed=20261127,
        save_checkpoint=False,
        ddp_preferred=False,
        lr_schedule="wsd",
    )

    # 1. Uninterrupted reference run (3 steps/epoch x 2 epochs = 6 steps)
    ref_dir = tmp_path / "ref"
    ref_res = train_memory_capacity_run(
        spec,
        out_dir=ref_dir,
        device="cpu",
        use_compile=False,
        seq_len=32,
        batch_size_override=4,
        eval_max_sequences_mlm=12,
        loo_max_sequences_mlm=8,
        loo_probes_per_seq=4,
        mlm_fixed_num_draws=4,
        checkpoint_interval_sec=0.0,
    )

    # 2. Preempted run: send SIGTERM right after step 4 (mid-epoch 2: epoch=2, step_in_epoch=1)
    preempt_dir = tmp_path / "preempt"
    helper_code = f"""
import os, signal, sys
from pathlib import Path
import torch
sys.path.insert(0, {str(REPO_ROOT)!r})
from src.capability.memory_capacity import RunSpec, execute_run

spec = RunSpec.from_dict({spec.to_dict()!r})
orig_step = torch.optim.AdamW.step
calls = [0]
def wrapped_step(self, *args, **kwargs):
    out = orig_step(self, *args, **kwargs)
    calls[0] += 1
    if calls[0] == 4:
        os.kill(os.getpid(), signal.SIGTERM)
    return out
torch.optim.AdamW.step = wrapped_step
execute_run(
    spec,
    out_dir={str(preempt_dir)!r},
    device="cpu",
    use_compile=False,
    seq_len=32,
    batch_size_override=4,
    eval_max_sequences_mlm=12,
    loo_max_sequences_mlm=8,
    loo_probes_per_seq=4,
    mlm_fixed_num_draws=4,
    checkpoint_interval_sec=0.0,
)
"""
    proc = subprocess.run(
        [sys.executable, "-c", helper_code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == PREEMPTION_EXIT_CODE, proc.stderr

    latest_ckpt = preempt_dir / "checkpoints" / f"{spec.run_id}_latest.pt"
    assert latest_ckpt.is_file()
    ckpt_data = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
    assert ckpt_data["current_epoch"] == 2
    assert ckpt_data["step_in_epoch"] == 1
    assert ckpt_data["global_step"] == 4
    assert ckpt_data["run_spec_hash"] == spec.spec_hash()
    assert ckpt_data["rng_states"]["mask_gen_state"] is not None

    # 3. Verify incompatible RunSpec is rejected on resume
    bad_spec = RunSpec(**{**spec.to_dict(), "seed": 999999})
    with pytest.raises(ValueError, match="run_spec_hash mismatch"):
        train_memory_capacity_run(
            bad_spec,
            out_dir=preempt_dir,
            device="cpu",
            use_compile=False,
            seq_len=32,
            batch_size_override=4,
            eval_max_sequences_mlm=12,
            loo_max_sequences_mlm=8,
            loo_probes_per_seq=4,
            mlm_fixed_num_draws=4,
            checkpoint_interval_sec=0.0,
        )

    # 4. Resume cleanly from mid-epoch checkpoint and verify exact match with uninterrupted run
    resumed_res = train_memory_capacity_run(
        spec,
        out_dir=preempt_dir,
        device="cpu",
        use_compile=False,
        seq_len=32,
        batch_size_override=4,
        eval_max_sequences_mlm=12,
        loo_max_sequences_mlm=8,
        loo_probes_per_seq=4,
        mlm_fixed_num_draws=4,
        checkpoint_interval_sec=0.0,
    )
    assert resumed_res["status"] == "complete"
    assert resumed_res["final_eval"]["global_step"] == ref_res["final_eval"]["global_step"]
    assert resumed_res["final_eval"]["mean_masked_token_xent_bits"] == pytest.approx(
        ref_res["final_eval"]["mean_masked_token_xent_bits"], abs=1e-6
    )
    assert resumed_res["final_eval"]["train_loss_last_epoch_bits_per_tok"] == pytest.approx(
        ref_res["final_eval"]["train_loss_last_epoch_bits_per_tok"], abs=1e-6
    )


def test_legacy_batch1_json_compatibility_ecrit_and_mlm15_analysis(tmp_path: Path) -> None:
    # 1. Verify legacy Batch-1 RunSpec dict (paradigm="mlm", no lr_schedule) loads cleanly
    legacy_spec_dict = {
        "run_id": "primary_mlm_0p5m_r2p00",
        "track": "primary",
        "paradigm": "mlm",
        "scale_id": "0.5m",
        "multiplier": 2.0,
        "n_sequences": 3145,
        "max_epochs": 400,
        "eval_epochs": [1, 2, 5, 10, 25, 50, 100, 200, 400],
        "weight_decay": 0.0,
        "precision": "bf16",
        "seed": 20261128,
        "data_seed": 20261127,
        "save_checkpoint": False,
        "ddp_preferred": False,
    }
    loaded_spec = RunSpec.from_dict(legacy_spec_dict)
    assert loaded_spec.paradigm == "oa_uniform"
    assert loaded_spec.lr_schedule == "cosine"

    # 2. Direct e_crit nonparametric & delayed-Hill estimator recovery
    epochs = np.array([1, 2, 5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 800, 1500], dtype=np.float64)
    c_inf_true = 2.6
    e_crit_true = 40.0
    tau_true = 120.0
    alpha_true = 2.0
    shifted = np.maximum(epochs - e_crit_true, 0.0)
    y = np.where(
        epochs > e_crit_true,
        c_inf_true * expit(alpha_true * (np.log(np.maximum(shifted, 1e-12)) - math.log(tau_true))),
        0.0,
    )
    dh = fit_delayed_hill_exposure_curve(epochs, y)
    assert dh is not None
    assert dh["c_inf_bits_per_param"] == pytest.approx(c_inf_true, rel=0.05)
    assert dh["e_crit_epochs"] == pytest.approx(e_crit_true, abs=8.0)
    assert dh["e_50_epochs"] == pytest.approx(e_crit_true + tau_true, rel=0.05)
    assert dh["r2"] > 0.999

    ec_np = estimate_ecrit_nonparametric(epochs, y, 0.05 * c_inf_true)
    assert ec_np is not None and 50.0 < ec_np < 100.0

    # 3. Full campaign aggregation mixing legacy "mlm" receipts + "ar" + "mlm_15"
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    spec_05 = SCALE_SPECS["0.5m"]
    p_ne = spec_05.exact_non_embedding_params
    p_tot = spec_05.exact_total_params
    p_emb = spec_05.exact_embedding_params
    eval_grid = (1, 5, 20, 50, 100, 200, 400)

    for raw_paradigm in ("ar", "mlm", "mlm_15"):
        for mult in (0.5, 1.0, 2.0, 4.0):
            n_seq = max(8, int(round((mult * 3.5 * p_ne) / (SEQ_LEN * LOG2_20))))
            h_null = null_ordered_entropy_bits(n_seq)
            h_data = dataset_entropy_bits(n_seq)
            h_gibbs = gibbs_memorizable_bits(n_seq)
            exps: list[dict[str, Any]] = []
            for ep in eval_grid:
                e_delay = 0.0 if raw_paradigm == "ar" else (15.0 if raw_paradigm == "mlm_15" else 40.0)
                if ep <= e_delay:
                    c_ep = 0.001
                else:
                    c_ep = 2.5 * float(expit(1.5 * (math.log(ep - e_delay) - math.log(40.0))))
                mem_bits = min(h_data * 0.99, c_ep * p_ne)
                mean_seq_xent = (h_null - mem_bits) / n_seq
                mean_tok_xent = mean_seq_xent / SEQ_LEN
                rec: dict[str, Any] = {
                    "epoch": ep,
                    "step": ep * 10,
                    "tokens_seen": ep * n_seq * SEQ_LEN,
                    "lr": 1e-3,
                    "elapsed_sec": float(ep),
                    "train_loss_last_epoch_bits_per_tok": mean_tok_xent,
                    "eval_n_sequences": n_seq,
                    "mean_seq_xent_bits": mean_seq_xent,
                    "mean_tok_xent_bits": mean_tok_xent,
                    "eval_mc_se_seq_bits": 0.01,
                    "total_model_xent_bits": h_null - mem_bits,
                    "memorized_bits": mem_bits,
                    "memorized_bits_se": 0.01 * n_seq,
                    "bits_per_non_emb_param": mem_bits / p_ne,
                    "bits_per_total_param": mem_bits / p_tot,
                    "fraction_of_H_data": mem_bits / h_data,
                    "fraction_of_H_gibbs": mem_bits / h_gibbs,
                }
                if raw_paradigm == "mlm_15":
                    sw_bits = n_seq * 38 * (LOG2_20 - mean_tok_xent)
                    rec["sw_memorized_bits"] = sw_bits
                    rec["sw_bits_per_non_emb_param"] = sw_bits / p_ne
                    rec["pseudo_memorized_bits"] = mem_bits
                    rec["pseudo_bits_per_non_emb_param"] = mem_bits / p_ne
                    rec["mean_masked_token_xent_bits"] = mean_tok_xent
                exps.append(rec)

            doc = {
                "schema": "plm_parametric_capacity_run_v1",
                "status": "complete",
                "run_id": f"primary_{raw_paradigm}_0p5m_r{str(mult).replace('.', 'p')}",
                "track": "primary",
                "paradigm": raw_paradigm,
                "scale_id": "0.5m",
                "multiplier": mult,
                "n_sequences": n_seq,
                "seq_len": SEQ_LEN,
                "vocab_size": VOCAB_AA,
                "n_params_non_embedding": p_ne,
                "n_params_embedding": p_emb,
                "n_params_total": p_tot,
                "weight_decay": 0.0,
                "precision": "bf16",
                "max_epochs": 400,
                "seed": 20261128,
                "data_seed": 20261127,
                "H_null_ordered_bits": h_null,
                "H_data_bits": h_data,
                "H_gibbs_max_bits": h_gibbs,
                "exposures": exps,
            }
            write_json_with_sha256(runs_dir / f"{doc['run_id']}.json", doc)

    summary = summarize_memory_capacity_campaign(runs_dir, bootstrap_draws=100)
    assert set(summary["c_max_calibration"]["by_paradigm"].keys()) == {"ar", "mlm_15", "oa_uniform"}
    mlm15_dual = summary["c_max_calibration"]["by_paradigm"]["mlm_15"]["mlm_15_dual_capacity"]
    assert mlm15_dual is not None
    sw_c = mlm15_dual["sw_bits_per_non_emb_param"]["zero_intercept_ratio"]["c_max"]
    ps_c = mlm15_dual["pseudo_bits_per_non_emb_param"]["zero_intercept_ratio"]["c_max"]
    assert sw_c == pytest.approx(ps_c * (38.0 / 256.0), rel=1e-6)

    for p in ("ar", "mlm_15", "oa_uniform"):
        ec = summary["exposure_discount_curves"][p]["pooled_e_crit_nonparametric"]
        assert ec["e_crit_delta_bpt_0p05"] is not None
        assert ec["e_crit_frac_c_inf_0p05"] is not None
        assert summary["exposure_discount_curves"][p]["pooled_delayed_hill_fit_empirical"] is not None


def test_shared_fs_lease_claim_expire_steal_and_spot_worker_cli(tmp_path: Path) -> None:
    now = [1000.0]
    mgr = SharedFSLeaseManager(tmp_path / "shared", lease_ttl_sec=60.0, clock=lambda: now[0])

    spec_a = RunSpec(
        run_id="spot_ar_0p5m_r0p25",
        track="primary",
        paradigm="ar",
        scale_id="0.5m",
        multiplier=0.25,
        n_sequences=8,
        max_epochs=1,
        eval_epochs=(1,),
        weight_decay=0.0,
        precision="fp32_strict",
        seed=20261128,
        data_seed=20261127,
        save_checkpoint=False,
        ddp_preferred=False,
        lr_schedule="wsd",
    )
    spec_b = RunSpec(**{**spec_a.to_dict(), "run_id": "spot_mlm15_0p5m_r0p25", "paradigm": "mlm_15"})

    w1 = generate_worker_token("s1")
    w2 = generate_worker_token("s2")

    # Worker 1 claims spec_a; Worker 2 cannot double-claim spec_a and gets spec_b
    claim1 = mgr.claim_next([spec_a, spec_b], worker_id=w1)
    assert claim1 is not None and claim1[0].run_id == spec_a.run_id
    claim2 = mgr.claim_next([spec_a, spec_b], worker_id=w2)
    assert claim2 is not None and claim2[0].run_id == spec_b.run_id
    assert mgr.claim_next([spec_a, spec_b], worker_id=generate_worker_token("s3")) is None

    # Heartbeat renewal keeps lease alive past initial TTL
    now[0] += 40.0
    assert mgr.renew(claim1[1]) is True
    now[0] += 40.0  # t=1080: claim1 renewed at 1040 (age 40 <= 60), claim2 stale (age 80 > 60)!
    assert mgr.try_claim(spec_a.run_id, worker_id=w2) is None
    stolen = mgr.try_claim(spec_b.run_id, worker_id=w1)
    assert stolen is not None and stolen.worker_id == w1
    # Original owner w2 can no longer renew or release stolen lease
    assert mgr.renew(claim2[1]) is False
    assert mgr.release(claim2[1]) is False
    assert mgr.release(stolen) is True
    assert mgr.release(claim1[1]) is True

    # End-to-end Spot worker CLI on CPU over a 1-run manifest
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"runs": [spec_a.to_dict()]}, indent=2), encoding="utf-8")
    loaded = load_run_manifest(manifest_path)
    assert len(loaded) == 1 and loaded[0].run_id == spec_a.run_id

    spot_out = tmp_path / "spot_out"
    proc = subprocess.run(
        [
            sys.executable,
            str(SPOT_WORKER_SCRIPT),
            "--manifest",
            str(manifest_path),
            "--out-dir",
            str(spot_out),
            "--gpus",
            "cpu",
            "--mode",
            "single_gpu",
            "--no-compile",
            "--fail-fast",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"[COMPLETE single_gpu] run={spec_a.run_id}" in proc.stdout
    assert (spot_out / "runs" / f"{spec_a.run_id}.json").is_file()
    assert not (spot_out / "leases" / f"{spec_a.run_id}.lease.json").exists()


def test_pretakeoff_flat_curve_suppresses_spurious_ecrit_and_delayed_hill(tmp_path: Path) -> None:
    epochs = np.array([1, 2, 5, 10, 25, 50, 100, 200, 400], dtype=np.float64)
    flat_y = np.linspace(0.0, 0.0007, epochs.size)
    assert fit_delayed_hill_exposure_curve(epochs, flat_y) is None

    runs_dir = tmp_path / "flat_runs"
    runs_dir.mkdir(parents=True)
    spec_05 = SCALE_SPECS["0.5m"]
    p_ne = spec_05.exact_non_embedding_params
    p_tot = spec_05.exact_total_params
    p_emb = spec_05.exact_embedding_params
    for mult in (0.5, 1.0, 2.0, 4.0):
        n_seq = max(8, int(round((mult * 3.5 * p_ne) / (SEQ_LEN * LOG2_20))))
        h_null = null_ordered_entropy_bits(n_seq)
        h_data = dataset_entropy_bits(n_seq)
        h_gibbs = gibbs_memorizable_bits(n_seq)
        exps = []
        for ep, c_ep in zip(epochs, flat_y):
            mem_bits = float(c_ep) * p_ne
            mean_seq_xent = (h_null - mem_bits) / n_seq
            mean_tok_xent = mean_seq_xent / SEQ_LEN
            exps.append(
                {
                    "epoch": int(ep),
                    "step": int(ep) * 10,
                    "tokens_seen": int(ep) * n_seq * SEQ_LEN,
                    "lr": 1e-3,
                    "elapsed_sec": float(ep),
                    "train_loss_last_epoch_bits_per_tok": mean_tok_xent,
                    "eval_n_sequences": n_seq,
                    "mean_seq_xent_bits": mean_seq_xent,
                    "mean_tok_xent_bits": mean_tok_xent,
                    "eval_mc_se_seq_bits": 0.01,
                    "total_model_xent_bits": h_null - mem_bits,
                    "memorized_bits": mem_bits,
                    "memorized_bits_se": 0.01 * n_seq,
                    "bits_per_non_emb_param": mem_bits / p_ne,
                    "bits_per_total_param": mem_bits / p_tot,
                    "fraction_of_H_data": mem_bits / h_data,
                    "fraction_of_H_gibbs": mem_bits / h_gibbs,
                }
            )
        doc = {
            "schema": "plm_parametric_capacity_run_v1",
            "status": "complete",
            "run_id": f"primary_mlm_0p5m_r{str(mult).replace('.', 'p')}",
            "track": "primary",
            "paradigm": "mlm",
            "scale_id": "0.5m",
            "multiplier": mult,
            "n_sequences": n_seq,
            "seq_len": SEQ_LEN,
            "vocab_size": VOCAB_AA,
            "n_params_non_embedding": p_ne,
            "n_params_embedding": p_emb,
            "n_params_total": p_tot,
            "weight_decay": 0.0,
            "precision": "bf16",
            "max_epochs": 400,
            "seed": 20261128,
            "data_seed": 20261127,
            "H_null_ordered_bits": h_null,
            "H_data_bits": h_data,
            "H_gibbs_max_bits": h_gibbs,
            "exposures": exps,
        }
        write_json_with_sha256(runs_dir / f"{doc['run_id']}.json", doc)

    summary = summarize_memory_capacity_campaign(runs_dir, bootstrap_draws=50)
    oa_exp = summary["exposure_discount_curves"]["oa_uniform"]
    assert oa_exp["pooled_delayed_hill_fit_empirical"] is None
    assert oa_exp["pooled_e_crit_nonparametric"]["e_crit_delta_bpt_0p05"] is None
    assert oa_exp["pooled_e_crit_nonparametric"]["e_crit_frac_c_inf_0p05"] is None
    assert oa_exp["by_scale"]["0.5m"]["delayed_hill_fit_empirical"] is None
    assert oa_exp["by_scale"]["0.5m"]["e_crit_nonparametric"]["e_crit_frac_c_inf_0p05"] is None


def test_spot_worker_skips_deterministic_failure_and_completes_healthy_run(tmp_path: Path) -> None:
    # A run with corrupt pre-existing checkpoint (mismatched run_spec_hash) fails deterministically;
    # worker must skip it instead of looping forever and still finish the healthy second run.
    bad_spec = RunSpec(
        run_id="spot_bad_ckpt_run",
        track="primary",
        paradigm="ar",
        scale_id="0.5m",
        multiplier=0.25,
        n_sequences=8,
        max_epochs=1,
        eval_epochs=(1,),
        weight_decay=0.0,
        precision="fp32_strict",
        seed=20261128,
        data_seed=20261127,
        save_checkpoint=False,
        ddp_preferred=False,
        lr_schedule="wsd",
    )
    good_spec = RunSpec(**{**bad_spec.to_dict(), "run_id": "spot_good_run"})
    spot_out = tmp_path / "spot_fail_skip"
    ckpt_dir = spot_out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "run_spec_hash": "0" * 64,
            "run_spec": bad_spec.to_dict(),
            "epoch_done": 0,
            "current_epoch": 1,
            "step_in_epoch": 1,
        },
        ckpt_dir / f"{bad_spec.run_id}_latest.pt",
    )
    manifest_path = tmp_path / "manifest_fail_skip.json"
    manifest_path.write_text(
        json.dumps({"runs": [bad_spec.to_dict(), good_spec.to_dict()]}, indent=2),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(SPOT_WORKER_SCRIPT),
            "--manifest",
            str(manifest_path),
            "--out-dir",
            str(spot_out),
            "--gpus",
            "cpu",
            "--mode",
            "single_gpu",
            "--no-compile",
            "--max-run-retries",
            "0",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert f"[FAILED single_gpu] run={bad_spec.run_id}" in proc.stderr
    assert f"[COMPLETE single_gpu] run={good_spec.run_id}" in proc.stdout
    assert (spot_out / "runs" / f"{good_spec.run_id}.json").is_file()
    assert not (spot_out / "runs" / f"{bad_spec.run_id}.json").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_mid_epoch_sigterm_flush_and_exact_gpu_resume(tmp_path: Path) -> None:
    spec = RunSpec(
        run_id="test_cuda_mid_epoch_resume",
        track="primary",
        paradigm="mlm_15",
        scale_id="0.5m",
        multiplier=0.25,
        n_sequences=16,
        max_epochs=2,
        eval_epochs=(1, 2),
        weight_decay=0.0,
        precision="fp32_strict",
        seed=20261128,
        data_seed=20261127,
        save_checkpoint=False,
        ddp_preferred=False,
        lr_schedule="wsd",
    )

    ref_dir = tmp_path / "cuda_ref"
    ref_res = train_memory_capacity_run(
        spec,
        out_dir=ref_dir,
        device="cuda:0",
        use_compile=False,
        seq_len=64,
        batch_size_override=4,
        eval_max_sequences_mlm=16,
        loo_max_sequences_mlm=8,
        loo_probes_per_seq=4,
        mlm_fixed_num_draws=4,
        checkpoint_interval_sec=0.0,
    )

    preempt_dir = tmp_path / "cuda_preempt"
    helper_code = f"""
import os, signal, sys
import torch
sys.path.insert(0, {str(REPO_ROOT)!r})
from src.capability.memory_capacity import RunSpec, execute_run

spec = RunSpec.from_dict({spec.to_dict()!r})
orig_step = torch.optim.AdamW.step
calls = [0]
def wrapped_step(self, *args, **kwargs):
    out = orig_step(self, *args, **kwargs)
    calls[0] += 1
    if calls[0] == 5:
        os.kill(os.getpid(), signal.SIGTERM)
    return out
torch.optim.AdamW.step = wrapped_step
execute_run(
    spec,
    out_dir={str(preempt_dir)!r},
    device="cuda:0",
    use_compile=False,
    seq_len=64,
    batch_size_override=4,
    eval_max_sequences_mlm=16,
    loo_max_sequences_mlm=8,
    loo_probes_per_seq=4,
    mlm_fixed_num_draws=4,
    checkpoint_interval_sec=0.0,
)
"""
    proc = subprocess.run(
        [sys.executable, "-c", helper_code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == PREEMPTION_EXIT_CODE, proc.stderr

    latest_ckpt = preempt_dir / "checkpoints" / f"{spec.run_id}_latest.pt"
    assert latest_ckpt.is_file()
    ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
    assert ckpt["current_epoch"] == 2
    assert ckpt["step_in_epoch"] == 1
    assert ckpt["rng_states"]["mask_gen_state"].device.type == "cpu"

    resumed_res = train_memory_capacity_run(
        spec,
        out_dir=preempt_dir,
        device="cuda:0",
        use_compile=False,
        seq_len=64,
        batch_size_override=4,
        eval_max_sequences_mlm=16,
        loo_max_sequences_mlm=8,
        loo_probes_per_seq=4,
        mlm_fixed_num_draws=4,
        checkpoint_interval_sec=0.0,
    )
    assert resumed_res["status"] == "complete"
    assert resumed_res["final_eval"]["global_step"] == ref_res["final_eval"]["global_step"]
    assert resumed_res["final_eval"]["mean_masked_token_xent_bits"] == pytest.approx(
        ref_res["final_eval"]["mean_masked_token_xent_bits"], abs=1e-5
    )
    assert resumed_res["final_eval"]["train_loss_last_epoch_bits_per_tok"] == pytest.approx(
        ref_res["final_eval"]["train_loss_last_epoch_bits_per_tok"], abs=1e-5
    )


def test_stage1_wsd_calibration_manifest_grid_and_lpt_priority(tmp_path: Path) -> None:
    from src.capability.memory_capacity import build_stage1_wsd_calibration_runs

    runs = build_stage1_wsd_calibration_runs()
    assert len(runs) == 70
    assert len({r.run_id for r in runs}) == 70

    by_p: dict[str, int] = {}
    for r in runs:
        by_p[r.paradigm] = by_p.get(r.paradigm, 0) + 1
        assert r.track == "stage1_wsd"
        assert r.lr_schedule == "wsd"
        assert r.adaptive_early_stop is True
        assert r.delete_transient_ckpt_on_complete is True
        assert r.save_checkpoint is False
        assert r.min_steps_per_epoch == 16
        assert r.run_id.startswith(f"wsd_{r.paradigm}_")

    assert by_p == {"ar": 23, "mlm_15": 25, "oa_uniform": 22}

    # Verify single-GPU workers see 0.5m pilots (oa_uniform & mlm_15) first, then LPT
    single_runs = [r for r in runs if not r.ddp_preferred]
    pilot_count = sum(
        1 for r in single_runs if r.scale_id == "0.5m" and r.paradigm in ("oa_uniform", "mlm_15")
    )
    assert pilot_count == 13
    assert all(
        r.scale_id == "0.5m" and r.paradigm in ("oa_uniform", "mlm_15")
        for r in single_runs[:pilot_count]
    )
    non_pilot_costs = [
        int(r.estimated_tokens) * int(SCALE_SPECS[r.scale_id].exact_non_embedding_params)
        for r in single_runs[pilot_count:]
    ]
    assert non_pilot_costs == sorted(non_pilot_costs, reverse=True)

    # Verify DDP workers see 150m AR first, then remaining DDP runs in descending LPT
    ddp_runs = [r for r in runs if r.ddp_preferred]
    assert all(r.scale_id == "150m" and r.paradigm == "ar" for r in ddp_runs[:4])
    rem_ddp_costs = [
        int(r.estimated_tokens) * int(SCALE_SPECS[r.scale_id].exact_non_embedding_params)
        for r in ddp_runs[4:]
    ]
    assert rem_ddp_costs == sorted(rem_ddp_costs, reverse=True)

    # CLI manifest builder roundtrip via load_run_manifest
    manifest_script = REPO_ROOT / "scripts/capability/memory_capacity/build_stage1_wsd_manifest.py"
    out_manifest = tmp_path / "stage1_wsd_manifest.json"
    proc = subprocess.run(
        [sys.executable, str(manifest_script), "--out", str(out_manifest)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Wrote 70 Stage-1 WSD runs" in proc.stdout
    loaded = load_run_manifest(out_manifest)
    assert [r.run_id for r in loaded] == [r.run_id for r in runs]


def test_spot_worker_single_gpu_skips_ddp_preferred_unless_allowed(tmp_path: Path) -> None:
    single_spec = RunSpec(
        run_id="spot_single_only_0p5m",
        track="primary",
        paradigm="ar",
        scale_id="0.5m",
        multiplier=0.25,
        n_sequences=8,
        max_epochs=1,
        eval_epochs=(1,),
        weight_decay=0.0,
        precision="fp32_strict",
        seed=20261128,
        data_seed=20261127,
        save_checkpoint=False,
        ddp_preferred=False,
        lr_schedule="wsd",
    )
    ddp_spec = RunSpec(
        **{
            **single_spec.to_dict(),
            "run_id": "spot_ddp_pref_0p5m",
            "ddp_preferred": True,
        }
    )
    manifest_path = tmp_path / "manifest_ddp_guard.json"
    manifest_path.write_text(
        json.dumps({"runs": [ddp_spec.to_dict(), single_spec.to_dict()]}, indent=2),
        encoding="utf-8",
    )

    for mode in ("single_gpu", "auto"):
        out_dir = tmp_path / f"out_{mode}_guard"
        proc = subprocess.run(
            [
                sys.executable,
                str(SPOT_WORKER_SCRIPT),
                "--manifest",
                str(manifest_path),
                "--out-dir",
                str(out_dir),
                "--gpus",
                "cpu",
                "--mode",
                mode,
                "--no-compile",
                "--fail-fast",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        assert f"[COMPLETE single_gpu] run={single_spec.run_id}" in proc.stdout
        assert ddp_spec.run_id not in proc.stdout
        assert (out_dir / "runs" / f"{single_spec.run_id}.json").is_file()
        assert not (out_dir / "runs" / f"{ddp_spec.run_id}.json").exists()

    out_allowed = tmp_path / "out_allowed"
    proc_allowed = subprocess.run(
        [
            sys.executable,
            str(SPOT_WORKER_SCRIPT),
            "--manifest",
            str(manifest_path),
            "--out-dir",
            str(out_allowed),
            "--gpus",
            "cpu",
            "--mode",
            "single_gpu",
            "--allow-ddp-on-single-gpu",
            "--no-compile",
            "--fail-fast",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert f"[COMPLETE single_gpu] run={ddp_spec.run_id}" in proc_allowed.stdout
    assert f"[COMPLETE single_gpu] run={single_spec.run_id}" in proc_allowed.stdout
    assert (out_allowed / "runs" / f"{ddp_spec.run_id}.json").is_file()
    assert (out_allowed / "runs" / f"{single_spec.run_id}.json").is_file()


def test_vram_safe_micro_batch_cap_preserves_small_scales_and_caps_150m() -> None:
    gib = 1024 ** 3
    h100_bytes = int(79.18 * gib)
    l20_bytes = int(45.0 * gib)
    h200_bytes = int(143.0 * gib)
    gpu24_bytes = int(24.0 * gib)

    for scale_id in ("0.5m", "2m", "8m", "35m"):
        max_bs = SCALE_SPECS[scale_id].max_batch_size
        assert vram_safe_micro_batch_cap(scale_id, h100_bytes, max_bs) == max_bs
        assert vram_safe_micro_batch_cap(scale_id, l20_bytes, max_bs) == max_bs

    assert vram_safe_micro_batch_cap("150m", h100_bytes, 512) == 128
    assert vram_safe_micro_batch_cap("150m", l20_bytes, 512) == 128
    assert vram_safe_micro_batch_cap("150m", h200_bytes, 512) == 512
    assert vram_safe_micro_batch_cap("150m", gpu24_bytes, 512) == 64
    assert vram_safe_micro_batch_cap("35m", gpu24_bytes, 512) == 256
    assert vram_safe_micro_batch_cap("150m", None, 512) == 512
