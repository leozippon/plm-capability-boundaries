"""Comprehensive unit and live-checkpoint tests for the 4 official DiMA adapters and 84-arm roster."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.capability.unified.dima_adapter import (
    AA20_INDEX,
    AA20_SET,
    CHEAPESMFoldProjector,
    CHEAPLatentScaler,
    CHEAPSequenceDecoder,
    DIMA_SPECS,
    DiMAAdapter,
    DiMAEncNormalizer,
    DiMAScoreEstimator,
    DynamicSDE,
    OPENFOLD_ORDER_WITH_X,
    ScoreEstimatorConfig,
    TanhScheduler,
    load_dima_adapter,
    resolve_dima_root,
    resolve_encoder_dir,
)
from src.capability.unified.roster import (
    ARM_METADATA,
    DIMA_4_ARMS,
    EXPANDED_15_STRUCT_ARMS,
    EXPANDED_19_NEW_ARMS,
    PARADIGM_ORDER,
    PARADIGM_ORDER_80,
    PARADIGM_ORDER_84,
    UNIFIED_3_MSA_CONTEXT_ARMS,
    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
    UNIFIED_65_ARMS,
    UNIFIED_77_SINGLE_TARGET_ARMS,
    UNIFIED_80_ARMS,
    UNIFIED_81_SINGLE_TARGET_ARMS,
    UNIFIED_84_ARMS,
    arms_by_paradigm,
    get_arm_metadata,
)


def _test_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _has_dima_assets() -> bool:
    try:
        root = resolve_dima_root()
        return (root / DIMA_SPECS["dima-esm2-8m"].diffusion_relpath).is_file()
    except FileNotFoundError:
        return False


def _has_encoder(arm_id: str) -> bool:
    try:
        resolve_encoder_dir(DIMA_SPECS[arm_id].encoder_dir_names)
        return True
    except FileNotFoundError:
        return False


# --------------------------------------------------------------------------- #
# 1. Roster 84-Arm Invariants & Backwards Compatibility
# --------------------------------------------------------------------------- #


def test_roster_84_invariants_and_backwards_compat() -> None:
    assert len(UNIFIED_65_ARMS) == 65
    assert len(UNIFIED_62_SINGLE_SEQUENCE_ARMS) == 62
    assert len(UNIFIED_80_ARMS) == 80
    assert len(UNIFIED_77_SINGLE_TARGET_ARMS) == 77
    assert len(UNIFIED_84_ARMS) == 84
    assert len(UNIFIED_81_SINGLE_TARGET_ARMS) == 81
    assert len(DIMA_4_ARMS) == 4
    assert len(EXPANDED_19_NEW_ARMS) == 19
    assert set(EXPANDED_15_STRUCT_ARMS) | set(DIMA_4_ARMS) == set(EXPANDED_19_NEW_ARMS)
    assert set(UNIFIED_80_ARMS).isdisjoint(set(DIMA_4_ARMS))
    assert set(UNIFIED_81_SINGLE_TARGET_ARMS) | set(UNIFIED_3_MSA_CONTEXT_ARMS) == set(UNIFIED_84_ARMS)

    assert PARADIGM_ORDER_84 == (*PARADIGM_ORDER_80, "continuous_latent_diffusion")
    assert tuple(a for p in PARADIGM_ORDER for a in arms_by_paradigm(p)) == UNIFIED_65_ARMS
    assert tuple(a for p in PARADIGM_ORDER_80 for a in arms_by_paradigm(p, include_80=True)) == UNIFIED_80_ARMS
    assert tuple(a for p in PARADIGM_ORDER_84 for a in arms_by_paradigm(p, include_84=True)) == UNIFIED_84_ARMS
    assert set(arms_by_paradigm("continuous_latent_diffusion")) == set(DIMA_4_ARMS)

    for arm in DIMA_4_ARMS:
        meta = get_arm_metadata(arm)
        assert meta.paradigm == "continuous_latent_diffusion"
        assert meta.origin_cohort == "expanded_4_dima"
        assert meta.is_single_sequence
        assert meta.has_r1_raw_readout_1024
        assert arm in ARM_METADATA


# --------------------------------------------------------------------------- #
# 2. Tanh VP-SDE Schedule Math
# --------------------------------------------------------------------------- #


def test_tanh_scheduler_and_sde_math() -> None:
    sched = TanhScheduler(d=10.0, t_thr=0.95)
    sde = DynamicSDE(sched, T=1.0)
    ts = torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0], dtype=torch.float32)
    mu, std = sde.marginal_params(ts)
    assert torch.isclose(mu[0], torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(std[0], torch.tensor(0.0), atol=1e-6)
    assert torch.all(mu[:-1] >= mu[1:])
    assert torch.all(std[:-1] <= std[1:])
    assert torch.allclose(mu.pow(2) + std.pow(2), torch.ones_like(mu), atol=1e-5)

    # t_thr clamping on beta_t
    betas = sched.beta_t(torch.tensor([0.95, 0.99, 1.0]))
    assert torch.allclose(betas[0], betas[1]) and torch.allclose(betas[0], betas[2])

    x = torch.randn(2, 8, 16)
    sc = torch.randn(2, 8, 16)
    t_b = torch.tensor([0.5, 0.5])
    drift_ode, diff_ode = sde.reverse_params(x, t_b, sc, ode_sampling=True)
    drift_sde, diff_sde = sde.reverse_params(x, t_b, sc, ode_sampling=False)
    assert torch.all(diff_ode == 0.0)
    assert torch.all(diff_sde > 0.0)
    assert not torch.allclose(drift_ode, drift_sde)


# --------------------------------------------------------------------------- #
# 3. Real Staged DiMA Checkpoints Strict Load, EMA, Stats & CHEAP Assets
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _has_dima_assets(), reason="dima_all checkpoints not staged locally")
def test_all_4_real_dima_checkpoints_strict_load_and_ema() -> None:
    root = resolve_dima_root()
    for arm_id, spec in DIMA_SPECS.items():
        se = DiMAScoreEstimator(ScoreEstimatorConfig(embedding_size=spec.embedding_dim))
        meta = se.load_dima_checkpoint(root / spec.diffusion_relpath, use_ema=True)
        assert meta["ema_applied"] is True
        assert meta["step"] >= 500000

        norm = DiMAEncNormalizer.from_file(root / spec.stats_relpath)
        assert norm.enc_mean.shape == (1, 1, spec.embedding_dim)
        assert norm.enc_std.shape == (1, 1, spec.embedding_dim)
        x = torch.randn(2, 12, spec.embedding_dim)
        assert torch.allclose(norm.denormalize(norm.normalize(x)), x, atol=1e-5)

    # Verify CHEAP assets (esmfold_v1_s_proj.pt + LatentScaler + mlp.ckpt) strict load & roundtrip
    cheap_root = root / "cheap_assets"
    scaler = CHEAPLatentScaler(cheap_root / "statistics/cath/esmfold/subset_5000_nov28")
    proj = CHEAPESMFoldProjector(cheap_root / "esmfold_v1_s_proj.pt")
    dec = CHEAPSequenceDecoder.from_checkpoint(cheap_root / "sequence_decoder/mlp.ckpt")

    L = 16
    fake_hs = [torch.randn(1, L + 2, 2560) * 0.1 for _ in range(37)]
    aa = torch.tensor([[OPENFOLD_ORDER_WITH_X[c] for c in "MKTAYIAKQRQISFVK"]], dtype=torch.long)
    s_s_0 = proj.project(fake_hs, aa)
    assert s_s_0.shape == (1, L, 1024) and torch.isfinite(s_s_0).all()
    scaled = scaler.scale(s_s_0)
    assert torch.allclose(scaler.unscale(scaled), s_s_0, atol=1e-3)
    logits = dec(s_s_0)
    assert logits.shape == (1, L, 21) and torch.isfinite(logits).all()


# --------------------------------------------------------------------------- #
# 4. Live E2E Verification on dima-esm2-8m & dima-saprot-35m
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("arm_id", ["dima-esm2-8m", "dima-saprot-35m"])
def test_dima_real_encoder_full_e2e_scoring_and_generation(arm_id: str) -> None:
    if not (_has_dima_assets() and _has_encoder(arm_id)):
        pytest.skip(f"{arm_id} weights or frozen encoder not staged locally")

    dev = _test_device()
    ad = load_dima_adapter(arm_id, device=dev, dtype="float32")
    prov = ad.provenance()
    assert prov["arm"] == arm_id and prov["use_ema"] is True

    # Realistic 36-aa sequence
    wt = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPN"

    # 1. Normalized latent sanity check (verifies SaProt interleaved "M#K#..." avoids <unk> collapse!)
    z0, mask = ad.encode_latents([wt])
    assert z0.shape == (1, len(wt) + 2, DIMA_SPECS[arm_id].embedding_dim)
    z_res = z0[0, 1 : 1 + len(wt), :]
    assert abs(float(z_res.mean().item())) < 1.5
    assert 0.3 < float(z_res.std().item()) < 2.5

    # 2. STRICT ZERO-LEAKAGE TEST: mutating residue at pos p MUST NOT change site_conditional_logprobs[p]!
    pos_check = 5
    orig_aa = wt[pos_check]
    alt_aa = "W" if orig_aa != "W" else "G"
    wt_mut = wt[:pos_check] + alt_aa + wt[pos_check + 1 :]
    lp_wt = ad.site_conditional_logprobs(wt, positions=[pos_check], batch_size=2)
    lp_mut = ad.site_conditional_logprobs(wt_mut, positions=[pos_check], batch_size=2)
    assert lp_wt.shape == (1, 20) and np.isfinite(lp_wt).all()
    assert np.allclose(np.exp(lp_wt).sum(axis=-1), 1.0, atol=1e-5)
    assert np.allclose(lp_wt, lp_mut, atol=1e-5), f"{arm_id}: target-residue leakage detected!"

    # 3. exact_pll, pll_8fold, latent_denoising_mse, score_mutations (single + double)
    pv_ex = ad.exact_pll(wt, batch_size=12)
    pv_8f = ad.pll_8fold(wt)
    mse = ad.latent_denoising_mse(wt)
    assert pv_ex.terms.shape == (len(wt),) and pv_ex.nll_sum > 0.0
    assert pv_8f.terms.shape == (len(wt),) and pv_8f.nll_sum > 0.0
    assert np.isfinite(mse) and mse > 0.0

    m1 = f"{wt[0]}1A" if wt[0] != "A" else f"{wt[0]}1G"
    m2 = f"{wt[3]}4E" if wt[3] != "E" else f"{wt[3]}4K"
    scores = ad.score_mutations(wt, [m1, m2, f"{m1}:{m2}"], batch_size=4)
    assert scores.shape == (3,) and np.isfinite(scores).all()

    # 4. Readout 1024 (U-Net bottleneck R_mid + output R_last) & Contact Matrix & R4 Context
    h1024 = ad.extract_readout_1024([wt, wt_mut], batch_size=2)
    assert h1024.shape == (2, 1024) and h1024.dtype == np.float32 and np.isfinite(h1024).all()
    assert not np.allclose(h1024[:, :512], h1024[:, 512:])

    cm = ad.extract_contact_matrix(wt[:14], batch_size=7)
    assert cm["couplings"].shape == (14, 14)
    assert np.allclose(cm["couplings"], cm["couplings"].T, atol=1e-5)
    assert np.allclose(np.diag(cm["couplings"]), 0.0, atol=1e-6)

    r4 = ad.score_r4_context(wt[:16], [wt[16:]], budget_tokens=510)
    assert r4["n_target_residues"] == 16 and r4["n_context_residues"] == len(wt) - 16
    assert np.isfinite(r4["nll_per_residue"]) and r4["nll_per_residue"] > 0.0

    # 5. Long-sequence (>510 aa) centered/tiled windowing check
    long_seq = (wt * 15)[:530]
    lp_long = ad.site_conditional_logprobs(long_seq, positions=[0, 265, 529], batch_size=3)
    assert lp_long.shape == (3, 20) and np.isfinite(lp_long).all()
    h_long = ad.extract_readout_1024([long_seq], batch_size=1)
    assert h_long.shape == (1, 1024) and np.isfinite(h_long).all()

    # 6. Generation: unconditional SDE & ODE, R5a RePaint infill, R4 prefix continuation, Exp 4.2 guided
    g_sde = ad.generate_unconditional(24, n_samples=1, steps=10, ode_sampling=False, seed=42)[0]
    g_ode = ad.generate_unconditional(24, n_samples=1, steps=10, ode_sampling=True, seed=42)[0]
    for g in (g_sde, g_ode):
        assert len(g["sequence"]) == 24 and set(g["sequence"]) <= AA20_SET
        assert g["mean_applied_bias_norm"] == 0.0
        if arm_id == "dima-saprot-35m":
            assert g["threedi_sequence"] is not None and len(g["threedi_sequence"]) == 24

    inf = ad.infill(wt, start=10, extent=6, steps=10, ode_sampling=True, seed=7)
    assert len(inf["sequence"]) == len(wt)
    assert inf["sequence"][:10] == wt[:10] and inf["sequence"][16:] == wt[16:]
    assert set(inf["infilled_span"]) <= AA20_SET

    cont = ad.continue_from_prefix(wt[:12], target_length=24, steps=10, ode_sampling=True, seed=7)
    assert len(cont["sequence"]) == 24 and cont["sequence"][:12] == wt[:12]
    assert set(cont["sequence"]) <= AA20_SET

    bias = np.zeros((20, 20), dtype=np.float32)
    bias[:, AA20_INDEX["A"]] = 3.0
    g_ung = ad.sample_guided_family(20, profile_log_bias=None, steps=10, ode_sampling=True, seed=11)
    g_gdd = ad.sample_guided_family(20, profile_log_bias=bias, guidance_scale=1.5, steps=10, ode_sampling=True, seed=11)
    assert g_ung["mean_applied_bias_norm"] == 0.0
    assert g_gdd["mean_applied_bias_norm"] > 0.0
    assert len(g_gdd["sequence"]) == 20 and set(g_gdd["sequence"]) <= AA20_SET


# --------------------------------------------------------------------------- #
# 5. Live E2E Generation Check on dima-cheap-1024 (No ESM2-3B Encoder Needed)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _has_dima_assets(), reason="dima_all checkpoints not staged locally")
def test_dima_cheap_1024_generation_and_guidance_live() -> None:
    dev = _test_device()
    ad = DiMAAdapter("dima-cheap-1024", device=dev, dtype="float32", load_encoder=False)
    g_sde = ad.generate_unconditional(28, n_samples=1, steps=15, ode_sampling=False, seed=123)[0]
    g_ode = ad.generate_unconditional(28, n_samples=1, steps=15, ode_sampling=True, seed=123)[0]
    for g in (g_sde, g_ode):
        assert len(g["sequence"]) == 28 and set(g["sequence"]) <= AA20_SET
        assert g["threedi_sequence"] is None
        assert 0.0 < g["mean_max_prob"] <= 1.0

    bias = np.zeros((20, 20), dtype=np.float32)
    bias[:, AA20_INDEX["L"]] = 3.0
    g_gdd = ad.sample_guided_family(20, profile_log_bias=bias, guidance_scale=1.5, steps=12, ode_sampling=True, seed=99)
    assert g_gdd["mean_applied_bias_norm"] > 0.0
    assert len(g_gdd["sequence"]) == 20 and set(g_gdd["sequence"]) <= AA20_SET
