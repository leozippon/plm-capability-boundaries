"""Tests for 80-arm unified roster, ProteinMPNN (11 checkpoints), StructTokenizerEncoder, and DPLM-2."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import EsmConfig

from src.capability.core.amino_acids import AA20
from src.capability.unified.dplm2_adapter import (
    AA_MASK_ID,
    DPLM2_SPECS,
    STRUCT_CODEBOOK_SIZE,
    STRUCT_MASK_ID,
    STRUCT_VOCAB_OFFSET,
    DPLM2Adapter,
    EsmForDPLM2,
    EsmForDPLM2Bit,
    StructTokenizerEncoder,
    code_indices_to_lfq_bits,
    lfq_bits_to_code_indices,
    resolve_dplm2_dir,
)
from src.capability.unified.proteinmpnn_adapter import (
    ProteinMPNNAdapter,
    _order_mask_backward,
    resolve_proteinmpnn_root,
)
from src.capability.unified.roster import (
    ARM_METADATA,
    DPLM2_4_ARMS,
    EXPANDED_15_STRUCT_ARMS,
    PARADIGM_ORDER,
    PARADIGM_ORDER_80,
    PROTEINMPNN_11_ARMS,
    UNIFIED_3_MSA_CONTEXT_ARMS,
    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
    UNIFIED_65_ARMS,
    UNIFIED_77_SINGLE_TARGET_ARMS,
    UNIFIED_80_ARMS,
    arms_by_paradigm,
    get_arm_metadata,
)

AA20_SET = frozenset(AA20)


def _test_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _ideal_alpha_helix_coords(length: int = 24) -> np.ndarray:
    """Generate a realistic non-degenerate 3D alpha-helix (L, 4, 3) (N, CA, C, O) backbone."""
    coords = np.zeros((length, 4, 3), dtype=np.float32)
    radius = 2.3
    rise = 1.5
    twist = math.radians(100.0)
    for i in range(length):
        theta = i * twist
        z = i * rise
        ca = np.array([radius * math.cos(theta), radius * math.sin(theta), z], dtype=np.float32)
        t_hat = np.array([-math.sin(theta), math.cos(theta), 0.45], dtype=np.float32)
        t_hat /= np.linalg.norm(t_hat)
        r_hat = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=np.float32)
        b_hat = np.cross(t_hat, r_hat)
        b_hat /= np.linalg.norm(b_hat)
        n_atom = ca - 1.20 * t_hat - 0.80 * r_hat
        c_atom = ca + 1.25 * t_hat + 0.70 * b_hat
        o_atom = c_atom + 1.23 * r_hat + 0.30 * b_hat
        coords[i, 0] = n_atom
        coords[i, 1] = ca
        coords[i, 2] = c_atom
        coords[i, 3] = o_atom
    return coords


def _has_real_dplm2_weights(arm_id: str) -> bool:
    try:
        d = resolve_dplm2_dir(DPLM2_SPECS[arm_id].dir_name)
    except FileNotFoundError:
        return False
    return any(
        list(d.glob("*.bin")) + list(d.glob("*.safetensors"))
    )


# --------------------------------------------------------------------------- #
# 1. Roster 65 -> 77 -> 80 Invariants
# --------------------------------------------------------------------------- #


def test_roster_80_invariants_and_backwards_compat() -> None:
    assert len(DPLM2_4_ARMS) == 4
    assert len(PROTEINMPNN_11_ARMS) == 11
    assert len(EXPANDED_15_STRUCT_ARMS) == 15
    assert len(UNIFIED_65_ARMS) == 65
    assert len(UNIFIED_62_SINGLE_SEQUENCE_ARMS) == 62
    assert len(UNIFIED_77_SINGLE_TARGET_ARMS) == 77
    assert len(UNIFIED_80_ARMS) == 80
    assert len(ARM_METADATA) >= 80
    assert set(UNIFIED_80_ARMS) <= set(ARM_METADATA)

    assert set(UNIFIED_65_ARMS).isdisjoint(set(EXPANDED_15_STRUCT_ARMS))
    assert set(UNIFIED_77_SINGLE_TARGET_ARMS) == set(UNIFIED_62_SINGLE_SEQUENCE_ARMS) | set(EXPANDED_15_STRUCT_ARMS)
    assert set(UNIFIED_80_ARMS) == set(UNIFIED_77_SINGLE_TARGET_ARMS) | set(UNIFIED_3_MSA_CONTEXT_ARMS)

    # Backwards-compatible default arms_by_paradigm matches 65-arm universe exactly
    assert tuple(a for p in PARADIGM_ORDER for a in arms_by_paradigm(p)) == UNIFIED_65_ARMS
    assert tuple(a for p in PARADIGM_ORDER_80 for a in arms_by_paradigm(p, include_80=True)) == UNIFIED_80_ARMS

    for arm in DPLM2_4_ARMS:
        meta = get_arm_metadata(arm)
        assert meta.paradigm == "multimodal_diffusion"
        assert meta.modality == "protein_multimodal"
    for arm in PROTEINMPNN_11_ARMS:
        meta = get_arm_metadata(arm)
        assert meta.paradigm == "inverse_folding_gnn"
        assert meta.modality == "structure_to_sequence"


# --------------------------------------------------------------------------- #
# 2. ProteinMPNN Exact Math & Real Checkpoints (All 11)
# --------------------------------------------------------------------------- #


def test_order_mask_backward_matches_official_einsum() -> None:
    rng = np.random.default_rng(20261125)
    B, L = 6, 19
    orders = torch.as_tensor(np.stack([rng.permutation(L) for _ in range(B)], axis=0), dtype=torch.long)
    fast = _order_mask_backward(orders, dtype=torch.float32)

    perm_oh = torch.nn.functional.one_hot(orders, num_classes=L).float()
    tri = 1.0 - torch.triu(torch.ones(L, L))
    ref = torch.einsum("ij,biq,bjp->bqp", tri, perm_oh, perm_oh)
    assert torch.equal(fast, ref)


def test_proteinmpnn_all_11_checkpoints_load_strict_and_e2e() -> None:
    try:
        root = resolve_proteinmpnn_root()
    except FileNotFoundError:
        pytest.skip("proteinmpnn_all checkpoints not staged locally")

    dev = _test_device()
    # Verify all 11 checkpoints load with strict=True (weight parity)
    for arm_id in PROTEINMPNN_11_ARMS:
        adapter = ProteinMPNNAdapter(arm_id, model_root=root, device=dev, dtype="float32")
        prov = adapter.provenance()
        assert prov["arm"] == arm_id
        assert prov["k_neighbors"] == 48
        assert prov["parameters"] > 1_500_000

    coords = _ideal_alpha_helix_coords(24)
    wt = "AEAAAKEAAAKEAAAKEAAAKAAA"
    muts = ["A1V", "K6E", "A1V:K6E"]

    for arm_id in ("proteinmpnn-v-020", "proteinmpnn-s-020", "proteinmpnn-ca-020"):
        ad = ProteinMPNNAdapter(arm_id, model_root=root, device=dev, dtype="float32")

        # 1. Exact PLL & site conditional
        pv = ad.exact_pll(wt, coords, batch_size=8)
        assert pv.terms.shape == (24,)
        assert np.isfinite(pv.terms).all() and (pv.terms >= 0.0).all()
        lp0 = ad.native_site_conditional(wt, 0, coords=coords)
        assert lp0.shape == (20,)
        assert abs(float(np.exp(lp0).sum()) - 1.0) < 1e-5

        # 2. Single & double-mutant scoring + non-zero symmetric 3D epistasis
        scores = ad.score_variants(wt, muts, coords, batch_size=8, multi_permutations=4)
        assert scores.shape == (3,) and np.isfinite(scores).all()
        s_ar = ad.score_variants(wt, muts, coords, batch_size=8, multi_permutations=4, force_exact_autoregressive=True)
        epistasis = float(s_ar[2] - s_ar[0] - s_ar[1])
        assert abs(epistasis) > 1e-5, f"{arm_id}: expected non-zero 3D GNN epistasis, got {epistasis}"

        # 3. Frozen hidden states (R_mid + R_last -> 1024-d)
        var_seq = "VEAAAEAAAKEAAAKEAAAKAAAA"[:24]
        h_raw = ad.extract_hidden_states([wt, var_seq], coords, batch_size=4, return_projected_1024=False)
        h_proj = ad.extract_hidden_states([wt, var_seq], coords, batch_size=4, return_projected_1024=True)
        assert h_raw.shape == (2, 4, 128) and np.isfinite(h_raw).all()
        assert h_proj.shape == (2, 1024) and np.isfinite(h_proj).all()
        assert not np.allclose(h_raw[0], h_raw[1]), "Decoder hidden states must depend on sequence!"
        assert not np.allclose(h_proj[:, :512], h_proj[:, 512:]), "R_mid and R_last must differ!"

        # 4. E01 contact sensitivity matrix
        cm = ad.extract_contact_matrix(wt, coords, probe_aas=("A", "D", "W"), batch_size=12)
        assert cm["couplings"].shape == (24, 24)
        assert np.allclose(cm["couplings"], cm["couplings"].T, atol=1e-5)
        assert np.allclose(np.diag(cm["couplings"]), 0.0)

        # 5. Context scoring (no_context vs 2-chain visible context)
        no_ctx = ad.score_context(wt, coords, n_permutations=2)
        with_ctx = ad.score_context(wt, coords, context_seqs=[wt], context_coords=[coords], n_permutations=2)
        assert no_ctx["n_cross_chain_edges"] == 0
        assert with_ctx["n_cross_chain_edges"] > 0
        assert np.isfinite(with_ctx["nll_per_residue"])

        # 6. Generation: inverse folding, R5a span infilling, R4 suffix continuation
        gen_full = ad.generate(coords, n_samples=2, seed=42, temperature=0.1)
        assert len(gen_full) == 2
        for rec in gen_full:
            assert len(rec["sequence"]) == 24
            assert set(rec["sequence"]) <= AA20_SET
        infill = ad.infill_window(wt, coords, start=8, extent=5, seed=42, temperature=0.1)
        assert infill["sequence"][:8] == wt[:8] and infill["sequence"][13:] == wt[13:]
        assert set(infill["sequence"]) <= AA20_SET
        cont = ad.continue_suffix(wt, coords, prefix_len=10, n_samples=1, seed=42, temperature=0.1)[0]
        assert cont["sequence"][:10] == wt[:10]
        assert set(cont["sequence"]) <= AA20_SET


# --------------------------------------------------------------------------- #
# 3. StructTokenizerEncoder (Real Checkpoint + SE(3) Invariance + LFQ)
# --------------------------------------------------------------------------- #


def test_lfq_bit_roundtrip_all_8192_codes() -> None:
    codes = torch.arange(STRUCT_CODEBOOK_SIZE, dtype=torch.long)
    bits_pm1 = code_indices_to_lfq_bits(codes, codebook_dim=13)
    assert bits_pm1.shape == (STRUCT_CODEBOOK_SIZE, 13)
    assert set(bits_pm1.unique().tolist()) == {-1.0, 1.0}
    recovered = lfq_bits_to_code_indices((bits_pm1 > 0.0).long(), codebook_dim=13)
    assert torch.equal(codes, recovered)


def test_struct_tokenizer_real_ckpt_strict_and_se3_invariance() -> None:
    try:
        sdir = resolve_dplm2_dir("dplm2-struct-tokenizer")
    except FileNotFoundError:
        pytest.skip("dplm2-struct-tokenizer not staged locally")

    tok = StructTokenizerEncoder(sdir / "dplm2_struct_tokenizer.ckpt", device=_test_device())
    coords = _ideal_alpha_helix_coords(24)
    out1 = tok.encode_backbone(coords)
    assert out1["code_indices"].shape == (24,)
    assert int(out1["code_indices"].min()) >= 0 and int(out1["code_indices"].max()) < STRUCT_CODEBOOK_SIZE
    assert np.array_equal(out1["struct_token_ids"], out1["code_indices"] + STRUCT_VOCAB_OFFSET)
    assert out1["lfq_bits"].shape == (24, 13)

    # Apply random SO(3) rotation + translation -> must yield 100% identical discrete structure tokens!
    q, r = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))
    rot = (q * np.sign(np.diag(r))).astype(np.float32)
    if np.linalg.det(rot) < 0:
        rot[:, 0] *= -1.0
    shift = np.array([17.3, -42.1, 8.9], dtype=np.float32)
    coords_se3 = coords @ rot.T + shift
    out2 = tok.encode_backbone(coords_se3)
    assert np.array_equal(out1["code_indices"], out2["code_indices"]), "StructTokenizerEncoder must be SE(3)-invariant!"


# --------------------------------------------------------------------------- #
# 4. DPLM-2 & DPLM-2-Bit Synthetic Tiny-Model Parity & API Tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("is_bit", [False, True])
def test_dplm2_synthetic_both_architectures_and_modes(tmp_path: Path, is_bit: bool) -> None:
    """Verify EsmForDPLM2 and EsmForDPLM2Bit across seq_only and struct_conditioned modes."""
    arm_id = "dplm2-bit-650m" if is_bit else "dplm2-150m"
    ckpt_dir = tmp_path / arm_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg = EsmConfig(
        vocab_size=33 if is_bit else 8229,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=128,
        position_embedding_type="rotary",
        token_dropout=True,
        pad_token_id=1,
        mask_token_id=AA_MASK_ID,
        tie_word_embeddings=False,
    )
    cfg.dplm_type = "dplm2_bit_esm" if is_bit else "dplm2_esm"
    if is_bit:
        cfg.codebook_embed_dim = 13

    torch.manual_seed(1234)
    model = EsmForDPLM2Bit(cfg) if is_bit else EsmForDPLM2(cfg)
    model.save_pretrained(str(ckpt_dir), safe_serialization=False)
    # Ensure custom config fields are saved
    cfg_json = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    cfg_json["dplm_type"] = cfg.dplm_type
    if is_bit:
        cfg_json["codebook_embed_dim"] = 13
    (ckpt_dir / "config.json").write_text(json.dumps(cfg_json, indent=2), encoding="utf-8")

    wt = "MKTAYIAKQRQISFVK"
    st_codes = np.arange(len(wt), dtype=np.int64) * 17
    dev = _test_device()
    ad_seq = DPLM2Adapter(arm_id, mode="seq_only", model_root=tmp_path, device=dev, dtype="float32", load_struct_tokenizer=False)
    ad_str = DPLM2Adapter(arm_id, mode="struct_conditioned", model_root=tmp_path, device=dev, dtype="float32", load_struct_tokenizer=False)

    # 1. Exact PLL & 8-fold PLL
    pv_seq = ad_seq.exact_pll(wt, batch_size=4)
    pv_str = ad_str.exact_pll(wt, struct_tokens=st_codes, batch_size=4)
    pv_8f = ad_seq.pll_8fold(wt)
    assert pv_seq.terms.shape == (len(wt),) and np.isfinite(pv_seq.terms).all()
    assert pv_str.terms.shape == (len(wt),) and np.isfinite(pv_str.terms).all()
    assert pv_8f.terms.shape == (len(wt),) and np.isfinite(pv_8f.terms).all()
    assert not np.allclose(pv_seq.terms, pv_str.terms), "Struct-conditioned logits must differ from seq_only!"

    # 2. Variant scoring (single + multi)
    muts = ["M1A", "K2E", "M1A:K2E"]
    s_seq = ad_seq.score_variants(wt, muts, batch_size=4)
    s_str = ad_str.score_variants(wt, muts, struct_tokens=st_codes, batch_size=4, multi_exact_pll=True)
    assert s_seq.shape == (3,) and np.isfinite(s_seq).all()
    assert s_str.shape == (3,) and np.isfinite(s_str).all()

    # 3. Hidden states (4 blocks + 1024-d projection)
    h_raw = ad_str.extract_hidden_states([wt, "AKTAYIAKQRQISFVK"], struct_tokens_list=st_codes, batch_size=2)
    h_1024 = ad_str.extract_hidden_states([wt, "AKTAYIAKQRQISFVK"], struct_tokens_list=st_codes, batch_size=2, return_projected_1024=True)
    assert h_raw.shape == (2, 4, 64) and np.isfinite(h_raw).all()
    assert h_1024.shape == (2, 1024) and np.isfinite(h_1024).all()
    assert not np.allclose(h_1024[:, :512], h_1024[:, 512:])

    # 4. E01 Contact sensitivity matrix
    cm = ad_seq.extract_contact_matrix(wt, batch_size=4)
    assert cm["couplings"].shape == (len(wt), len(wt))
    assert np.allclose(cm["couplings"], cm["couplings"].T, atol=1e-5)

    # 5. R4 Context scoring
    r4 = ad_seq.score_r4_context(wt, ["MKTAYIAK"], budget_tokens=128)
    assert r4["n_context_residues"] == 8 and np.isfinite(r4["nll_per_residue"])

    # 6. Generation: infill_window, continue_suffix, co_generation, seq_only max_confidence/uniform_random, inverse_folding
    inf = ad_seq.infill_window(wt, start=4, extent=4, seed=7, steps=4)
    assert inf["sequence"][:4] == wt[:4] and inf["sequence"][8:] == wt[8:]
    suf = ad_seq.continue_suffix(wt[:6], len(wt), seed=7, steps=4)
    assert suf["sequence"][:6] == wt[:6] and len(suf["sequence"]) == len(wt)
    for smode in ("max_confidence", "uniform_random", "co_generation"):
        g = ad_seq.sample_diffusion_full(12, seed=7, steps=4, sampling_mode=smode)
        assert len(g["sequence"]) == 12 and set(g["sequence"]) <= AA20_SET
        if smode == "co_generation":
            assert all(STRUCT_VOCAB_OFFSET <= tid < STRUCT_VOCAB_OFFSET + STRUCT_CODEBOOK_SIZE for tid in g["struct_token_ids"])
        else:
            assert all(tid == STRUCT_MASK_ID for tid in g["struct_token_ids"])
    inv = ad_str.sample_diffusion_full(
        len(wt), seed=7, steps=4, struct_tokens=st_codes, profile_log_bias=np.zeros((len(wt), 20), dtype=np.float32)
    )
    assert len(inv["sequence"]) == len(wt) and set(inv["sequence"]) <= AA20_SET


# --------------------------------------------------------------------------- #
# 5. Real Staged DPLM-2 Checkpoint Smoke Test (Runs When Weights Are Present)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("arm_id", ["dplm2-150m", "dplm2-650m", "dplm2-bit-650m"])
def test_dplm2_real_checkpoint_smoke(arm_id: str) -> None:
    if not _has_real_dplm2_weights(arm_id):
        pytest.skip(f"{arm_id} weights not yet finished downloading locally")

    dev = _test_device()
    coords = _ideal_alpha_helix_coords(20)
    wt = "AEAAAKEAAAKEAAAKEAAA"

    ad = DPLM2Adapter(arm_id, mode="struct_conditioned", device=dev, dtype="float32")
    st_ids = ad.encode_backbone(coords)
    assert st_ids.shape == (20,)

    pv_seq = ad.exact_pll(wt, mode="seq_only", batch_size=10)
    pv_str = ad.exact_pll(wt, struct_tokens=st_ids, mode="struct_conditioned", batch_size=10)
    assert np.isfinite(pv_seq.nll_sum) and np.isfinite(pv_str.nll_sum)
    assert pv_seq.nll_sum > 0.0 and pv_str.nll_sum > 0.0
    assert not np.allclose(pv_seq.terms, pv_str.terms)

    h1024 = ad.extract_hidden_states([wt], struct_tokens_list=st_ids, return_projected_1024=True)
    assert h1024.shape == (1, 1024) and np.isfinite(h1024).all()
    gen = ad.sample_diffusion_full(12, seed=42, steps=4, sampling_mode="co_generation")
    assert len(gen["sequence"]) == 12 and set(gen["sequence"]) <= AA20_SET
