"""Tests for Stage-2 upgrades: 650M scale, MDM alias, Muon, mask schedules, Eq. (3b) temp cal, large bundles."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.capability.supply_demand.build_large_scale_stage2_bundle import (
    build_bundle_command,
    parse_args as parse_large_bundle_args,
)
from src.capability.supply_demand import (
    AA20,
    LN2,
    LOG2_20,
    SCALE_SPECS,
    Stage2RunSpec,
    VariableLengthProteinTransformer,
    analyze_stage2_run,
    canonicalize_objective,
    collate_variable_length_sequences,
    compute_objective_loss,
    count_model_parameters,
    evaluate_three_tier_bits,
    execute_stage2_run,
    objective_display_name,
    prepare_objective_batch,
    should_enable_gradient_checkpointing,
)
from tests.supply_demand.test_evaluation_and_spot_training import _make_tiny_bundle


def test_stage2_650m_param_parity_and_mdm_alias() -> None:
    assert canonicalize_objective("mdm") == "oa_uniform"
    assert objective_display_name("mdm") == "MDM"
    assert objective_display_name("oa_uniform") == "MDM"
    assert "650m" in SCALE_SPECS
    spec650 = SCALE_SPECS["650m"]
    assert spec650.exact_non_embedding_params == 654_303_104
    with torch.device("meta"):
        m = VariableLengthProteinTransformer(spec650, "mdm", max_seq_len=384, gradient_checkpointing=True)
        counts = count_model_parameters(m, spec650)
        assert counts["n_params_non_embedding"] == 654_303_104
    assert should_enable_gradient_checkpointing("650m", 46 * 1024**3) is True


@pytest.mark.parametrize("objective", ["mlm_30", "mlm_50", "mdm_cosine", "mdm"])
def test_stage2_extended_mask_schedules_pad_exclusion_and_uniform_baseline(objective: str) -> None:
    seqs = ["".join(AA20[(i + j) % 20] for j in range(16 + 4 * i)) for i in range(8)]
    tokens, lengths = collate_variable_length_sequences(seqs, pad_to_multiple=8)
    gen = torch.Generator().manual_seed(77)
    prep = prepare_objective_batch(objective, tokens, lengths, generator=gen)
    for i in range(len(seqs)):
        L = int(lengths[i].item())
        assert not prep.loss_mask[i, L:].any()
        assert int(prep.loss_mask[i].sum().item()) == int(prep.n_loss_per_seq[i].item())
    zero_logits = torch.zeros((len(seqs), tokens.shape[1], 20), dtype=torch.float32)
    loss_bits = float(compute_objective_loss(zero_logits, prep).item()) / LN2
    assert math.isclose(loss_bits, LOG2_20, rel_tol=1e-5, abs_tol=1e-5)


def test_stage2_eq3b_temperature_calibration_muon_and_canonical_analysis(tmp_path: Path) -> None:
    """Verify exact Eq. (3b) H_unseen_fold_cal + Delta_cal in Stage-2 eval, Muon training, & analysis fields."""
    bundle = _make_tiny_bundle(tmp_path / "bundle", n_train=16, seed=31)
    splits = bundle.load_arm_splits("M4")

    # 1. Exact Eq. (3b) calibration invariance under 12x overconfident logit scaling
    torch.manual_seed(9)
    m_base = VariableLengthProteinTransformer("0.5m", "ar", max_seq_len=64)
    with torch.no_grad():
        m_base.lm_head.weight.mul_(5.0)
    m_scaled = VariableLengthProteinTransformer("0.5m", "ar", max_seq_len=64)
    m_scaled.load_state_dict(m_base.state_dict())
    with torch.no_grad():
        m_scaled.lm_head.weight.mul_(12.0)

    tt_base = evaluate_three_tier_bits(
        m_base,
        train_sub_store=splits["eval_train_sub"],
        unseen_homolog_store=splits["eval_unseen_homolog"],
        unseen_fold_store=splits["eval_unseen_fold"],
        objective="ar",
        batch_size=8,
        precision="fp32_strict",
        n_draws=4,
        loo_probes=4,
        eval_temperature_calibration=True,
    )
    tt_scaled = evaluate_three_tier_bits(
        m_scaled,
        train_sub_store=splits["eval_train_sub"],
        unseen_homolog_store=splits["eval_unseen_homolog"],
        unseen_fold_store=splits["eval_unseen_fold"],
        objective="ar",
        batch_size=8,
        precision="fp32_strict",
        n_draws=4,
        loo_probes=4,
        eval_temperature_calibration=True,
    )
    assert tt_scaled["native"]["H_unseen_fold"] > tt_base["native"]["H_unseen_fold"] + 0.5
    assert tt_scaled["native"]["Delta_cal"] > 0.5
    assert tt_scaled["native"]["H_unseen_fold_cal"] == pytest.approx(
        tt_base["native"]["H_unseen_fold_cal"], abs=0.05
    )

    # 2. End-to-end Stage-2 run with MDM alias + Muon + gradient_checkpointing + temperature calibration
    spec = Stage2RunSpec(
        run_id="e2e_mdm_muon",
        arm_id="M4",
        objective="mdm",
        scale_id="0.5m",
        max_epochs=2,
        eval_epochs=(1, 2),
        batch_size=8,
        precision="fp32_strict",
        optimizer="muon",
        muon_lr=0.02,
        gradient_checkpointing=True,
        eval_downstream_at_milestones=False,
        eval_contacts_at_milestones=False,
        eval_full_proteingym_at_final=False,
        eval_temperature_calibration=True,
    )
    receipt = execute_stage2_run(
        spec,
        bundle,
        tmp_path / "out",
        device="cpu",
        use_compile=False,
        n_eval_draws=4,
        loo_probes=4,
        install_signal_handlers=False,
    )
    assert receipt["status"] == "complete"
    assert receipt["paradigm_display"] == "MDM"
    assert receipt["training_config"]["optimizer"] == "muon"
    assert receipt["training_config"]["gradient_checkpointing"] is True
    final_tt = receipt["final_eval"]["three_tier_bits"]["native"]
    assert "H_unseen_fold_cal" in final_tt and "Delta_cal" in final_tt

    # 3. Canonical unified-notation fields in supply_demand/analysis.py
    analyzed = analyze_stage2_run(receipt)
    assert analyzed["paradigm_display"] == "MDM"
    assert "E_gen" in analyzed and "E_mem" in analyzed and "W_window" in analyzed and "Delta_cal" in analyzed
    assert analyzed["H_unseen_fold_cal"] is not None


def test_large_scale_stage2_bundle_cli_command_builder(tmp_path: Path) -> None:
    args = parse_large_bundle_args(
        [
            "--out-dir",
            str(tmp_path / "bundle_50m"),
            "--target-tokens",
            "50000000",
            "--arm-families",
            "256,1024,4096",
        ]
    )
    cmd = build_bundle_command(args)
    assert "--target-tokens" in cmd and "50000000" in cmd
    assert "--prejoined-parquet" in cmd
    assert any("uniref50_cath_l50_384.parquet" in tok for tok in cmd)
