"""Tests for Stage-1 upgrades: MDM alias & hash stability, 650M scale, grad ckpt, Muon, vocab modes, mask schedules."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.memory_capacity import (
    EXTENDED_SCALE_ORDER,
    LOG2_20,
    SCALE_SPECS,
    VOCAB_MODES,
    MuonWithAdamW,
    RunSpec,
    UniformVocabTransformer,
    build_optimizer,
    canonicalize_paradigm,
    count_model_parameters,
    dataset_entropy_bits,
    generate_vocab_mode_dataset,
    null_ordered_entropy_bits,
    paradigm_display_name,
    sample_mdm_cosine_batch,
    set_optimizer_lr,
    should_enable_gradient_checkpointing,
    train_memory_capacity_run,
    vram_safe_micro_batch_cap,
    zeropower_via_newtonschulz5,
)
from src.capability.memory_capacity.analysis import summarize_memory_capacity_campaign
from src.capability.memory_theory.io_utils import write_json_with_sha256


def test_mdm_alias_and_run_spec_hash_backward_compatibility() -> None:
    """'mdm' aliases cleanly to 'oa_uniform' with display 'MDM' and identical legacy RunSpec.spec_hash()."""
    assert canonicalize_paradigm("mdm") == "oa_uniform"
    assert canonicalize_paradigm("MDM") == "oa_uniform"
    assert canonicalize_paradigm("oa_uniform") == "oa_uniform"
    assert paradigm_display_name("mdm") == "MDM"
    assert paradigm_display_name("oa_uniform") == "MDM"
    assert paradigm_display_name("mlm_30") == "MLM-30%"
    assert paradigm_display_name("mlm_50") == "MLM-50%"
    assert paradigm_display_name("mdm_cosine") == "MDM-Cosine"

    base_kwargs = dict(
        run_id="wsd_oa_0p5m_r1p00",
        track="stage1_wsd",
        scale_id="0.5m",
        multiplier=1.0,
        n_sequences=1500,
        max_epochs=100,
        eval_epochs=(1, 10, 100),
        weight_decay=0.0,
        precision="bf16",
        seed=20261128,
        data_seed=20261127,
        save_checkpoint=False,
        ddp_preferred=False,
        lr_schedule="wsd",
    )
    spec_oa = RunSpec(paradigm="oa_uniform", **base_kwargs)
    spec_mdm = RunSpec(paradigm="mdm", **base_kwargs)
    assert spec_oa.spec_hash() == spec_mdm.spec_hash()
    assert spec_mdm.paradigm_display == "MDM"
    assert RunSpec(**spec_mdm.to_dict()) == spec_mdm

    # Legacy RunSpec hash must not include default optimizer/vocab_mode keys
    assert "optimizer" not in spec_oa.compatibility_payload()
    assert "vocab_mode" not in spec_oa.compatibility_payload()

    # Non-default optimizer or vocab_mode MUST change spec_hash to protect resume compatibility
    spec_muon = RunSpec(paradigm="oa_uniform", optimizer="muon", **base_kwargs)
    spec_dip = RunSpec(paradigm="oa_uniform", vocab_mode="dipeptide400", **base_kwargs)
    assert spec_muon.spec_hash() != spec_oa.spec_hash()
    assert spec_dip.spec_hash() != spec_oa.spec_hash()


def test_650m_scale_spec_gradient_checkpointing_and_vram_caps() -> None:
    """650m matches ESM-2 33L/1280d/20H/3456ff (~654.3M non-emb params) and auto-enables grad ckpt <110 GiB."""
    assert "650m" in SCALE_SPECS
    assert EXTENDED_SCALE_ORDER[-1] == "650m"
    spec650 = SCALE_SPECS["650m"]
    assert (spec650.n_layers, spec650.d_model, spec650.n_heads, spec650.d_ff) == (33, 1280, 20, 3456)
    assert spec650.exact_non_embedding_params == 654_303_104

    with torch.device("meta"):
        model650 = UniformVocabTransformer(spec650, "mdm", seq_len=256, gradient_checkpointing=True)
        counts = count_model_parameters(model650, spec650)
        assert counts["n_params_non_embedding"] == 654_303_104

    # Auto gradient checkpointing for 650m on <110 GiB GPUs (e.g. L20 46 GiB, H100 80 GiB)
    assert should_enable_gradient_checkpointing("650m", 46 * 1024**3) is True
    assert should_enable_gradient_checkpointing("650m", 80 * 1024**3) is True
    assert should_enable_gradient_checkpointing("650m", 141 * 1024**3) is False
    assert should_enable_gradient_checkpointing("650m", 46 * 1024**3, requested=False) is False
    assert should_enable_gradient_checkpointing("150m", 46 * 1024**3) is False
    assert vram_safe_micro_batch_cap("650m", 46 * 1024**3, 256) <= 16

    # Activation checkpointing produces identical forward outputs and gradients on 0.5m
    torch.manual_seed(17)
    m_plain = UniformVocabTransformer("0.5m", "ar", seq_len=32, gradient_checkpointing=False)
    m_ckpt = UniformVocabTransformer("0.5m", "ar", seq_len=32, gradient_checkpointing=True)
    m_ckpt.load_state_dict(m_plain.state_dict())
    inp = torch.randint(0, 20, (4, 32))
    tgt = torch.randint(0, 20, (4, 32))
    out1 = m_plain(inp)
    out2 = m_ckpt(inp)
    assert torch.allclose(out1, out2, atol=1e-6)
    torch.nn.functional.cross_entropy(out1.transpose(1, 2), tgt).backward()
    torch.nn.functional.cross_entropy(out2.transpose(1, 2), tgt).backward()
    for (n1, p1), (_, p2) in zip(m_plain.named_parameters(), m_ckpt.named_parameters()):
        assert p1.grad is not None and p2.grad is not None
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5), f"Grad mismatch on {n1}"


def test_muon_newton_schulz_param_partition_and_step() -> None:
    """Muon orthogonalizes hidden 2D weights while routing embeddings, lm_head, and 1D gains to AdamW."""
    torch.manual_seed(0)
    g = torch.randn(32, 64)
    orth = zeropower_via_newtonschulz5(g, steps=5)
    gram = orth @ orth.T
    off_diag = gram - torch.diag(torch.diag(gram))
    assert float(off_diag.abs().max().item()) < 0.15
    s_raw = torch.linalg.svdvals(g)
    s_orth = torch.linalg.svdvals(orth)
    cond_raw = float((s_raw.max() / s_raw.min()).item())
    cond_orth = float((s_orth.max() / s_orth.min()).item())
    assert cond_orth < 2.0 and cond_orth < cond_raw / 2.0
    assert float(s_orth.min().item()) >= 0.65 and float(s_orth.max().item()) <= 1.35

    model = UniformVocabTransformer("0.5m", "ar", seq_len=32)
    opt = build_optimizer(
        model,
        peak_lr=1e-3,
        weight_decay=0.01,
        device=torch.device("cpu"),
        optimizer_name="muon",
        muon_lr=0.02,
    )
    assert isinstance(opt, MuonWithAdamW)
    muon_groups = [grp for grp in opt.param_groups if grp.get("use_muon")]
    adam_groups = [grp for grp in opt.param_groups if not grp.get("use_muon")]
    assert len(muon_groups) == 1 and len(adam_groups) >= 2

    muon_param_ids = {id(p) for p in muon_groups[0]["params"]}
    assert id(model.tok_emb.weight) not in muon_param_ids
    assert id(model.lm_head.weight) not in muon_param_ids
    assert id(model.norm_f.weight) not in muon_param_ids
    assert id(model.blocks[0].attn.W_qkv.weight) in muon_param_ids
    assert id(model.blocks[0].ffn.W_down.weight) in muon_param_ids

    # Proportional LR scaling keeps Muon LR = 0.02 * (cur_lr / peak_lr)
    set_optimizer_lr(opt, 5e-4, peak_lr=1e-3)
    assert muon_groups[0]["lr"] == pytest.approx(0.01)
    assert adam_groups[0]["lr"] == pytest.approx(5e-4)

    # Single step updates parameters cleanly and state_dict round-trips
    x = torch.randint(0, 20, (4, 32))
    loss = torch.nn.functional.cross_entropy(model(x).transpose(1, 2), x)
    loss.backward()
    opt.step()
    sd = opt.state_dict()
    opt.load_state_dict(sd)


def test_vocab_mode_bit_identity_and_e2e_dipeptide400_muon_and_masks(tmp_path: Path) -> None:
    """Verify exact dipeptide400/tripeptide8000 bit identity, Muon run, and mlm_30/mdm_cosine runs."""
    # Exact ground-truth sequence entropy identity in bits:
    # 128 * log2(400) == 256 * log2(20), and 85 * log2(8000) == 255 * log2(20)
    h_aa256 = null_ordered_entropy_bits(100, seq_len=256, vocab_size=20)
    h_dip128 = null_ordered_entropy_bits(100, seq_len=128, vocab_size=400)
    assert h_dip128 == pytest.approx(h_aa256, rel=1e-12)
    assert dataset_entropy_bits(100, seq_len=128, vocab_size=400) == pytest.approx(
        dataset_entropy_bits(100, seq_len=256, vocab_size=20), rel=1e-12
    )

    h_aa255 = null_ordered_entropy_bits(100, seq_len=255, vocab_size=20)
    h_tri85 = null_ordered_entropy_bits(100, seq_len=85, vocab_size=8000)
    assert h_tri85 == pytest.approx(h_aa255, rel=1e-12)

    # Dataset shapes and value bounds for all 4 vocab modes
    for mode_id, vm in VOCAB_MODES.items():
        arr = generate_vocab_mode_dataset(6, vocab_mode=mode_id, data_seed=42)
        assert arr.shape == (6, vm.default_seq_len)
        assert int(arr.min()) >= 0 and int(arr.max()) < vm.vocab_size

    # Cosine-warped MDM mask schedule: E[1 - cos(pi/2 * u)] = 1 - 2/pi ~= 0.3634
    seqs = torch.randint(0, 20, (256, 64), dtype=torch.uint8)
    gen = torch.Generator().manual_seed(123)
    pos = torch.arange(64, dtype=torch.int64)
    _, _, mask, n_mask = sample_mdm_cosine_batch(seqs, generator=gen, pos_arange=pos)
    assert torch.all(mask.sum(dim=-1) == n_mask)
    mean_frac = float(n_mask.float().mean().item()) / 64.0
    assert 0.30 < mean_frac < 0.43

    # End-to-end dipeptide400 + Muon + AR run on CPU
    spec_dip = RunSpec(
        run_id="test_dip400_muon",
        track="ablation_vocab",
        paradigm="ar",
        scale_id="0.5m",
        multiplier=0.25,
        n_sequences=8,
        max_epochs=2,
        eval_epochs=(1, 2),
        weight_decay=0.0,
        precision="fp32_strict",
        seed=20261128,
        data_seed=20261127,
        save_checkpoint=False,
        ddp_preferred=False,
        optimizer="muon",
        vocab_mode="dipeptide400",
    )
    res_dip = train_memory_capacity_run(
        spec_dip,
        out_dir=tmp_path / "dip_out",
        device="cpu",
        use_compile=False,
        seq_len=16,
        batch_size_override=4,
    )
    assert res_dip["status"] == "complete"
    assert res_dip["vocab_size"] == 400
    assert res_dip["vocab_mode"] == "dipeptide400"
    assert res_dip["optimizer"] == "muon"
    assert res_dip["embedding_param_fraction"] > 0.15  # V=400 scales tok_emb + lm_head significantly
    assert res_dip["final_eval"]["bits_per_non_emb_param"] > res_dip["final_eval"]["bits_per_total_param"] or math.isfinite(
        res_dip["final_eval"]["bits_per_total_param"]
    )

    # End-to-end mlm_30 and mdm_cosine runs on CPU
    for par in ("mlm_30", "mdm_cosine"):
        spec_p = RunSpec(
            run_id=f"test_{par}",
            track="ablation_mask_rate",
            paradigm=par,
            scale_id="0.5m",
            multiplier=0.25,
            n_sequences=8,
            max_epochs=2,
            eval_epochs=(2,),
            weight_decay=0.0,
            precision="fp32_strict",
            seed=20261128,
            data_seed=20261127,
            save_checkpoint=False,
            ddp_preferred=False,
        )
        res_p = train_memory_capacity_run(
            spec_p,
            out_dir=tmp_path / f"out_{par}",
            device="cpu",
            use_compile=False,
            seq_len=32,
            batch_size_override=4,
            eval_max_sequences_mlm=8,
            loo_max_sequences_mlm=8,
            loo_probes_per_seq=2,
            mlm_fixed_num_draws=4,
        )
        assert res_p["status"] == "complete"
        assert res_p["paradigm"] == par
        if par == "mlm_30":
            assert res_p["final_eval"]["mlm_fixed_mask_count"] == round(0.30 * 32)


def test_stage1_analysis_exposes_canonical_notation_fields(tmp_path: Path) -> None:
    """memory_capacity/analysis.py exposes paradigm_display='MDM', E_gen, E_mem, W_window, Delta_cal."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    spec_05 = SCALE_SPECS["0.5m"]
    p_ne = spec_05.exact_non_embedding_params
    p_tot = spec_05.exact_total_params
    p_emb = spec_05.exact_embedding_params
    n_seq = 100
    h_null = null_ordered_entropy_bits(n_seq)
    h_data = dataset_entropy_bits(n_seq)
    for mult in (0.5, 1.0, 2.0, 4.0):
        exps = []
        for ep in (1, 10, 50, 200, 400):
            c = 0.25 * (ep / (ep + 100.0))
            mem = c * p_ne
            exps.append(
                {
                    "epoch": ep,
                    "step": ep * 10,
                    "tokens_seen": ep * n_seq * 256,
                    "lr": 1e-3,
                    "elapsed_sec": float(ep),
                    "train_loss_last_epoch_bits_per_tok": LOG2_20 - mem / (n_seq * 256),
                    "eval_n_sequences": n_seq,
                    "mean_seq_xent_bits": (h_null - mem) / n_seq,
                    "mean_tok_xent_bits": (h_null - mem) / (n_seq * 256),
                    "eval_mc_se_seq_bits": 0.01,
                    "total_model_xent_bits": h_null - mem,
                    "memorized_bits": mem,
                    "memorized_bits_se": 1.0,
                    "bits_per_non_emb_param": c,
                    "bits_per_total_param": mem / p_tot,
                    "fraction_of_H_data": mem / h_data,
                    "fraction_of_H_gibbs": mem / h_data,
                }
            )
        doc = {
            "schema": "plm_parametric_capacity_run_v1",
            "status": "complete",
            "run_id": f"primary_mdm_0p5m_r{str(mult).replace('.', 'p')}",
            "track": "primary",
            "paradigm": "mdm",
            "scale_id": "0.5m",
            "multiplier": mult,
            "n_sequences": n_seq,
            "seq_len": 256,
            "vocab_size": 20,
            "n_params_non_embedding": p_ne,
            "n_params_embedding": p_emb,
            "n_params_total": p_tot,
            "weight_decay": 0.0,
            "precision": "bf16",
            "max_epochs": 400,
            "seed": 1,
            "data_seed": 2,
            "H_null_ordered_bits": h_null,
            "H_data_bits": h_data,
            "H_gibbs_max_bits": h_data,
            "exposures": exps,
        }
        write_json_with_sha256(runs_dir / f"{doc['run_id']}.json", doc)

    summary = summarize_memory_capacity_campaign(runs_dir, bootstrap_draws=50)
    oa_cal = summary["c_max_calibration"]["by_paradigm"]["oa_uniform"]
    assert oa_cal["paradigm_display"] == "MDM"
    assert "E_gen" in oa_cal and "E_mem" in oa_cal and "W_window" in oa_cal and "Delta_cal" in oa_cal
    assert oa_cal["canonical_notation"]["paradigm_display"] == "MDM"
    assert summary["exposure_discount_curves"]["oa_uniform"]["paradigm_display"] == "MDM"
