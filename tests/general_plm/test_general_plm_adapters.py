"""Unit and L20 GPU smoke tests for the general-PLM registry, adapters, and stage entrypoints.

Run with:
  CUDA_VISIBLE_DEVICES=4 /home/lzp/miniconda3/envs/ct/bin/pytest -q tests/general_plm/test_general_plm_adapters.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.amino_acids import AA20
from src.capability.general_plm import (
    apply_mutation,
    arms_by_family,
    arms_for_module,
    get_spec,
    list_arms,
    load_general_plm,
    parse_mutation,
    resolve_checkpoint_dir,
)
from src.capability.general_plm.adapters import normalize_dtype_name
from src.capability.general_plm.eval_utils import project_four_blocks
from src.capability.general_plm.msa_utils import build_query_msa, project_partner_to_query
from src.capability.interactions.pairwise_epistasis import ROSTER


# ------------------------------------------------------------ Registry & Pure Unit Tests


def test_registry_covers_all_required_families_without_touching_roster():
    """All requested general-PLM families are declared, and the frozen 33-arm ROSTER is untouched."""
    assert len(ROSTER) == 33

    expected_arms = {
        # ESM2 (6)
        "esm2-8m", "esm2-35m", "esm2-150m", "esm2-650m", "esm2-3b", "esm2-15b",
        # DPLM (3)
        "dplm-150m", "dplm-650m", "dplm-3b",
        # ESMC-hf (3)
        "esmc-300m", "esmc-600m", "esmc-6b",
        # AMPLIFY (2)
        "amplify-120m", "amplify-350m",
        # SaProt (3)
        "saprot-35m", "saprot-650m", "saprot-1.3b",
        # ProteinGLM-MLM (3)
        "proteinglm-1b-mlm", "proteinglm-3b-mlm", "proteinglm-10b-mlm",
        # Prime (1)
        "prime-690m",
        # Multi-seq R4 (3)
        "msa-transformer-100m", "protgpt3-msa", "dayhoff-3b-msa",
        # Newer causal PLMs (8 + 1.3b ref)
        "protgpt3-112m", "protgpt3-10b",
        "progen3-219m", "progen3-339m", "progen3-762m", "progen3-1b",
        "dayhoff-170m", "dayhoff-3b",
    }
    registered = set(list_arms())
    assert expected_arms <= registered
    assert len(arms_by_family("esm2")) == 6
    assert len(arms_by_family("dplm")) == 3
    assert set(arms_for_module("r5_diffusion")) == {"dplm-150m", "dplm-650m", "dplm-3b"}
    assert {"msa-transformer-100m", "protgpt3-msa", "dayhoff-3b-msa"} <= set(arms_for_module("r4"))


def test_precision_guardrails_fail_fast():
    """FP16 is refused on ESM2/DPLM; FP32 is refused on ProGen3; BF16 opt-in succeeds."""
    esm15 = get_spec("esm2-15b")
    assert normalize_dtype_name(esm15, None) == "float32"
    assert normalize_dtype_name(esm15, "bf16") == "bfloat16"
    with pytest.raises(ValueError, match="float16 is refused"):
        normalize_dtype_name(esm15, "fp16")

    pg3 = get_spec("progen3-219m")
    assert normalize_dtype_name(pg3, None) == "bfloat16"
    with pytest.raises(ValueError, match="no float32 kernel"):
        normalize_dtype_name(pg3, "float32")


def test_resolve_checkpoint_fails_fast_on_missing_or_empty_dir(tmp_path: Path):
    empty = tmp_path / "esm2-8m"
    empty.mkdir()
    (empty / "config.json").write_text('{"model_type": "esm"}', encoding="utf-8")
    # Missing weight files must NOT count as a valid checkpoint
    with pytest.raises(FileNotFoundError, match="no complete local checkpoint"):
        resolve_checkpoint_dir("esm2-8m", model_root=tmp_path)

    # Once a weight file exists inside tmp_path/esm2-8m, it resolves cleanly
    (empty / "model.safetensors").write_bytes(b"stub")
    resolved = resolve_checkpoint_dir("esm2-8m", model_root=tmp_path)
    assert resolved == empty.resolve()


def test_mutation_parser_and_applier_fail_fast():
    wt = "MKTAYIAKQRQISFVK"
    assert parse_mutation("A4V:K8R", wt) == [(3, "A", "V"), (7, "K", "R")]
    assert apply_mutation(wt, "A4V:K8R") == "MKTVYIARQRQISFVK"
    with pytest.raises(ValueError, match="wild-type residue"):
        parse_mutation("G4V", wt)
    with pytest.raises(ValueError, match="non-canonical"):
        parse_mutation("A4X", wt)
    with pytest.raises(ValueError, match="duplicate mutated position"):
        parse_mutation("A4V:A4L", wt)


def test_msa_projection_preserves_query_coordinates():
    q = "ACDEFGHIK"
    p_ins = "ACDEZZFGHIK"  # insertion ZZ relative to query
    p_del = "ACDFGHIK"  # deletion of E relative to query
    assert project_partner_to_query(q, p_ins) == "ACDEFGHIK"
    assert project_partner_to_query(q, p_del) == "ACD-FGHIK"
    msa = build_query_msa(q, [p_ins, p_del], max_depth=8, aligned=True)
    assert len(msa) == 3
    assert all(len(row) == len(q) for row in msa)


def test_four_block_projection_matches_frozen_1024_width():
    raw = np.ones((5, 4, 320), dtype=np.float32)
    proj = project_four_blocks(raw)
    assert proj.shape == (5, 1024)
    assert proj.dtype == np.float32
    assert np.isfinite(proj).all()


def test_stage_entrypoints_resolve_via_h200_resolver():
    """Verify every new script in scripts/capability/general_plm resolves via entrypoints.py."""
    from scripts.capability.entrypoints import stage_path

    expected_scripts = (
        "run_r1_ladder_and_conditionals.py",
        "run_r1_family_distribution_pll.py",
        "run_r2_phenotypes_and_readouts.py",
        "run_r3_pairwise_pll_and_contacts.py",
        "run_r4_multiseq_context.py",
        "run_r5_infill_and_diffusion_gen.py",
        "fit_general_plm_summary.py",
    )
    for name in expected_scripts:
        resolved = stage_path(ROOT, name)
        assert resolved.is_file()
        assert resolved.parent.name == "general_plm"


# ------------------------------------------------------------ L20 GPU Smoke Tests


ESM2_650M_DIR = Path("/Data/public/esm2_t33_650M_UR50D")
PROTGPT3_112M_DIR = Path("/Data/public/models_R2/ProtGPT3-112M")
AMPLIFY_120M_DIR = Path("/home/lzp/models/general_plm_20261120/amplify-120m")
DAYHOFF_170M_DIR = Path("/home/lzp/models/general_plm_20261120/dayhoff-170m")
PROTGPT3_MSA_DIR = Path("/home/lzp/models/general_plm_20261120/protgpt3-msa")


@pytest.mark.skipif(
    not (torch.cuda.is_available() and ESM2_650M_DIR.is_dir()),
    reason="Requires GPU and locally staged /Data/public/esm2_t33_650M_UR50D",
)
def test_esm2_650m_gpu_parity_with_masked_scorer_and_nonzero_epistasis(tmp_path: Path):
    """Verify <1e-6 nat parity against MaskedScorer AND eps_PLL != 0 on a double mutant."""
    from src.capability.position.position_likelihood import MaskedScorer

    device = "cuda:0"
    wt = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRV"
    mut_a = apply_mutation(wt, "A4V")
    mut_b = apply_mutation(wt, "I6L")
    mut_ab = apply_mutation(wt, "A4V:I6L")

    # Reference MaskedScorer requires symlink named esm2_650m under model_root
    link_root = tmp_path / "ref_root"
    link_root.mkdir()
    (link_root / "esm2_650m").symlink_to(ESM2_650M_DIR)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ref = MaskedScorer("esm2-650m", device=device, dtype="float32", model_root=link_root)
    ref_state = ref.pack(wt)
    ref_score = ref.score([ref_state], batch_size=16)[0]
    del ref
    torch.cuda.empty_cache()

    adapter = load_general_plm("esm2-650m", device=device, dtype="float32", model_root=ESM2_650M_DIR)
    our_wt = adapter.exact_pll(wt, batch_size=16)

    max_term_diff = float(np.max(np.abs(our_wt.terms.astype(np.float64) - ref_score.terms.astype(np.float64))))
    sum_diff = abs(our_wt.nll_sum - ref_score.nll_sum)
    assert max_term_diff < 1e-6, f"Per-position parity exceeded 1e-6 nats: {max_term_diff}"
    assert sum_diff < 1e-5, f"Total PLL parity exceeded 1e-5 nats: {sum_diff}"

    # Verify non-degenerate double-mutant epistasis eps_PLL = PLL(AB) - PLL(A) - PLL(B) + PLL(WT) != 0
    pll_a = adapter.exact_pll(mut_a, batch_size=16).pll
    pll_b = adapter.exact_pll(mut_b, batch_size=16).pll
    pll_ab = adapter.exact_pll(mut_ab, batch_size=16).pll
    eps_pll = pll_ab - pll_a - pll_b + our_wt.pll
    assert np.isfinite(eps_pll) and abs(eps_pll) > 1e-4, f"Expected non-zero eps_PLL, got {eps_pll}"

    # Verify 1-pass native site conditional normalizes to 1.0 over AA20
    log_q = adapter.native_site_conditional(wt, 3)
    assert log_q.shape == (20,)
    assert abs(float(np.exp(log_q).sum()) - 1.0) < 1e-6

    # Verify p=0 hidden-state extraction and 1024-d projection
    h = adapter.extract_hidden_states([wt, mut_ab], batch_size=2, pooling="mean_cls")
    assert h.shape == (2, 4, 1280)
    proj = project_four_blocks(h)
    assert proj.shape == (2, 1024)

    # Verify bidirectional infilling and 100%-masked cosine diffusion sampler API
    inf = adapter.infill_window(wt, start=10, extent=4, seed=42)
    assert inf["status"] == "filled" and len(inf["sequence"]) == len(wt)
    diff_gen = adapter.sample_diffusion_full(24, seed=42, steps=8)
    assert len(diff_gen["sequence"]) == 24 and not (set(diff_gen["sequence"]) - set(AA20))

    del adapter
    torch.cuda.empty_cache()


@pytest.mark.skipif(
    not (torch.cuda.is_available() and PROTGPT3_112M_DIR.is_dir()),
    reason="Requires GPU and locally staged /Data/public/models_R2/ProtGPT3-112M",
)
def test_protgpt3_112m_gpu_scoring_readouts_r4_r5():
    """Verify ProtGPT3-112M causal scoring, eps != 0, hidden states, R4 context, and free-length gen."""
    device = "cuda:0"
    wt = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRV"
    mut_a = apply_mutation(wt, "A4V")
    mut_b = apply_mutation(wt, "I6L")
    mut_ab = apply_mutation(wt, "A4V:I6L")

    adapter = load_general_plm(
        "protgpt3-112m", device=device, dtype="float32", model_root=PROTGPT3_112M_DIR
    )
    prov = adapter.provenance()
    assert prov["dtype_realised"] == "torch.float32"
    assert prov["packing"] == "bos_direction_seq"

    s_wt = adapter.exact_pll(wt)
    s_a = adapter.exact_pll(mut_a)
    s_b = adapter.exact_pll(mut_b)
    s_ab = adapter.exact_pll(mut_ab)
    eps = s_ab.pll - s_a.pll - s_b.pll + s_wt.pll
    assert np.isfinite(eps) and abs(eps) > 1e-6

    # Variant scoring matches exact_pll difference
    v = adapter.score_variants(wt, ["A4V", "A4V:I6L"])
    assert abs(v[0] - (s_a.pll - s_wt.pll)) < 1e-6
    assert abs(v[1] - (s_ab.pll - s_wt.pll)) < 1e-6

    # Hidden states
    h = adapter.extract_hidden_states([wt, mut_a])
    assert h.shape == (2, 4, 512)
    assert project_four_blocks(h).shape == (2, 1024)

    # R4 multi-item context
    r4 = adapter.score_r4_context(wt, [mut_a, mut_b], budget_tokens=256)
    assert r4["k_items"] == 2 and np.isfinite(r4["nll_sum"])

    # R5 free-length generation
    draws = adapter.generate_causal_free(2, seed=123, max_residues=24)
    assert len(draws) == 2 and all(d["length"] > 0 for d in draws)

    del adapter
    torch.cuda.empty_cache()


@pytest.mark.skipif(
    not (torch.cuda.is_available() and AMPLIFY_120M_DIR.is_dir()),
    reason="Requires GPU and staged amplify-120m",
)
def test_amplify_120m_xformers_shim_and_additive_mask_gpu():
    """Verify AMPLIFY-120M loads with xformers shim and passes additive float mask on GPU."""
    adapter = load_general_plm("amplify-120m", device="cuda:0", dtype="float32", model_root=AMPLIFY_120M_DIR)
    wt = "MKTAYIAKQRQISFVKSHFSRQLEERLG"
    vec = adapter.exact_pll(wt, batch_size=8)
    assert vec.terms.shape == (len(wt),) and np.isfinite(vec.nll_sum)
    # Test padded batch hidden states (two different lengths -> exercises additive float mask!)
    h = adapter.extract_hidden_states([wt, wt[:18]], batch_size=2, pooling="mean_cls")
    assert h.shape[0] == 2 and h.shape[1] == 4 and np.isfinite(h).all()
    del adapter
    torch.cuda.empty_cache()


@pytest.mark.skipif(
    not (torch.cuda.is_available() and DAYHOFF_170M_DIR.is_dir()),
    reason="Requires GPU and staged dayhoff-170m",
)
def test_dayhoff_170m_full_logits_and_explicit_bos_gpu():
    """Verify Dayhoff-170M overrides num_logits_to_keep=0 and scores full sequence cleanly."""
    adapter = load_general_plm("dayhoff-170m", device="cuda:0", dtype="float32", model_root=DAYHOFF_170M_DIR)
    wt = "MKTAYIAKQRQISFVKSHFSRQLEERLG"
    vec = adapter.exact_pll(wt)
    assert vec.terms.shape == (len(wt),) and np.isfinite(vec.nll_sum)
    r4 = adapter.score_r4_context(wt, [wt[:20]], budget_tokens=256)
    assert r4["k_items"] == 1 and np.isfinite(r4["nll_sum"])
    del adapter
    torch.cuda.empty_cache()


@pytest.mark.skipif(
    not (torch.cuda.is_available() and PROTGPT3_MSA_DIR.is_dir()),
    reason="Requires GPU and staged protgpt3-msa",
)
def test_protgpt3_msa_r4_context_gpu():
    """Verify ProtGPT3-MSA packs <|bos|> 1 <no_gap> + <s> separated homologs cleanly."""
    adapter = load_general_plm("protgpt3-msa", device="cuda:0", dtype="float32", model_root=PROTGPT3_MSA_DIR)
    wt = "MKTAYIAKQRQISFVKSHFSRQLEERLG"
    r4 = adapter.score_r4_context(wt, [wt, wt[:22]], budget_tokens=256)
    assert r4["k_items"] == 2 and np.isfinite(r4["nll_sum"])
    del adapter
    torch.cuda.empty_cache()
