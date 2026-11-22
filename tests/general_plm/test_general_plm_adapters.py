"""Unit and L20 GPU smoke tests for the general-PLM registry, adapters, and stage entrypoints.

Run with:
  CUDA_VISIBLE_DEVICES=4 /home/lzp/miniconda3/envs/ct/bin/pytest -q tests/general_plm/test_general_plm_adapters.py
"""

from __future__ import annotations

import json
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
from src.capability.general_plm.eval_utils import nested_group_ridge_eval, project_four_blocks
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


def test_nested_group_ridge_eval_on_full_r1_r2_r3_cohorts_cpu():
    """Verify row_weights, family_folds, and ridge_predict on FULL R1/R2/R3 cohort metadata."""
    rng = np.random.default_rng(20261120)

    # 1. FULL 201-assay R1 variant_table (25,728 rows across 163 families, including multi-assay families)
    vt_path = ROOT / "results/extensions/statistical_ladder_20261106/inputs/variant_table.json"
    if vt_path.is_file():
        samples = json.loads(vt_path.read_text(encoding="utf-8"))["samples"]
        groups = np.asarray([int(s["cluster"]) for s in samples])
        subunits = np.asarray([str(s["assay"]) for s in samples], dtype="U")
        n = len(samples)
        y = rng.normal(size=n)
        x = y * 0.5 + rng.normal(scale=0.5, size=n)
        res = nested_group_ridge_eval(
            y, groups, subunits, {"M": x, "P": rng.normal(size=n)}, {"P": ("P",), "P_M": ("P", "M")}, seeds=(20260923,)
        )
        assert res["n_rows"] == 25728
        assert res["n_groups"] == 163
        assert res["summary"]["P_M"]["group_mse"]["mean"] < res["summary"]["P"]["group_mse"]["mean"]

    # 2. FULL R2 Domainome abundance (428 domains, 109,568 rows)
    dom_path = ROOT / "results/R1/external_confirmation_20260924/cohort.json"
    if dom_path.is_file():
        domains = json.loads(dom_path.read_text(encoding="utf-8"))["domains"]
        groups = np.asarray([d["group"] for d in domains for _ in d["variants"]], dtype="U")
        subunits = np.asarray([f"{d['name']}:{v['position'] - 1}" for d in domains for v in d["variants"]], dtype="U")
        y = np.asarray([float(v["target"]) for d in domains for v in d["variants"]], dtype=np.float64)
        x = y * 0.4 + rng.normal(scale=0.5, size=y.size)
        res = nested_group_ridge_eval(
            y, groups, subunits, {"M": x}, {"M": ("M",)}, seeds=(20260923,), alphas=(0.1, 1.0, 10.0)
        )
        assert res["n_rows"] == 109568
        assert np.isfinite(res["summary"]["M"]["group_mse"]["mean"])

    # 3. FULL R2 MegaScale stability (101 backgrounds, 25,856 rows)
    ms_path = ROOT / "results/R4/gate_stability_20260924/cohort/cohort.json"
    if ms_path.is_file():
        bgs = json.loads(ms_path.read_text(encoding="utf-8"))["backgrounds"]
        groups = np.asarray([bg["group"] for bg in bgs for _ in bg["variants"]], dtype="U")
        subunits = np.asarray([f"{bg['name']}:{v['position'] - 1}" for bg in bgs for v in bg["variants"]], dtype="U")
        y = np.asarray([float(v["ddg"]) for bg in bgs for v in bg["variants"]], dtype=np.float64)
        x = y * 0.4 + rng.normal(scale=0.5, size=y.size)
        res = nested_group_ridge_eval(
            y, groups, subunits, {"M": x}, {"M": ("M",)}, seeds=(20260923,), alphas=(0.1, 1.0, 10.0)
        )
        assert res["n_rows"] == 25856
        assert np.isfinite(res["summary"]["M"]["group_mse"]["mean"])

    # 4. FULL R2 added phenotypes (qual-3 rows)
    pb_rows = ROOT / "results/extensions/phenotype_breadth_20261008/qual-3/rows"
    if pb_rows.is_dir():
        from scripts.capability.general_plm.run_r2_phenotypes_and_readouts import _load_phenotypes

        bgs, _potts, _gemme, _freqs = _load_phenotypes(ROOT, max_bgs=0)
        groups = np.asarray([bg["group"] for bg in bgs for _ in bg["mutations"]], dtype="U")
        subunits = np.asarray(
            [f"{bg['name']}:{int(m[1:-1]) - 1}" for bg in bgs for m in bg["mutations"]],
            dtype="U",
        )
        y = np.asarray([float(t) for bg in bgs for t in bg["targets"]], dtype=np.float64)
        x = y * 0.4 + rng.normal(scale=0.5, size=y.size)
        res = nested_group_ridge_eval(
            y, groups, subunits, {"M": x}, {"M": ("M",)}, seeds=(20260923,), alphas=(0.1, 1.0, 10.0)
        )
        assert res["n_rows"] > 10000
        assert np.isfinite(res["summary"]["M"]["group_mse"]["mean"])

    # 5. FULL R3 pairwise extraction plan (64 backgrounds, 8,192 cycles)
    ep_path = ROOT / "archive/logs/R3/pairwise_epistasis_20260924/extraction_plan.json"
    if ep_path.is_file():
        bgs = json.loads(ep_path.read_text(encoding="utf-8"))["backgrounds"]
        groups = np.asarray([bg["group"] for bg in bgs for _ in bg["cycles"]], dtype="U")
        subunits = np.asarray(
            [f"{bg['name']}:{c['positions'][0]}_{c['positions'][1]}" for bg in bgs for c in bg["cycles"]],
            dtype="U",
        )
        y = rng.normal(size=groups.size)
        x = y * 0.4 + rng.normal(scale=0.5, size=groups.size)
        res = nested_group_ridge_eval(
            y, groups, subunits, {"M": x}, {"M": ("M",)}, seeds=(20260923,), alphas=(0.1, 1.0, 10.0)
        )
        assert res["n_rows"] == 8192
        assert np.isfinite(res["summary"]["M"]["group_mse"]["mean"])


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
PROTEINGLM_1B_MLM_DIR = Path("/home/lzp/models/general_plm_20261120/proteinglm-1b-mlm")
SAPROT_35M_DIR = Path("/home/lzp/models/general_plm_20261120/saprot-35m")
PROGEN3_219M_DIR = Path("/home/lzp/models/general_plm_20261120/progen3-219m")
MSA_100M_DIR = Path("/home/lzp/models/general_plm_20261120/esm-msa1b-100m")
DAYHOFF_3B_MSA_DIR = Path("/home/lzp/models/general_plm_20261120/dayhoff-3b-GR-HM-c")
ESM2_8M_DIR = Path("/home/lzp/models/general_plm_20261120/esm2-8m")


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
    """Verify Dayhoff-170M overrides num_logits_to_keep=0 and output_router_logits=False."""
    adapter = load_general_plm("dayhoff-170m", device="cuda:0", dtype="float32", model_root=DAYHOFF_170M_DIR)
    assert adapter.model.config.output_router_logits is False
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


@pytest.mark.skipif(
    not (torch.cuda.is_available() and PROTEINGLM_1B_MLM_DIR.is_dir()),
    reason="Requires GPU and staged proteinglm-1b-mlm",
)
def test_proteinglm_1b_mlm_trailing_eos_encoding_and_hidden_states_gpu():
    """Verify ProteinGLM-1B-MLM (offset=0, trailing <eos>, time-first hidden states) on GPU."""
    adapter = load_general_plm(
        "proteinglm-1b-mlm", device="cuda:0", dtype="float32", model_root=PROTEINGLM_1B_MLM_DIR
    )
    wt = "MKTAYIAKQRQISFVKSHFSRQLEERLG"
    vec = adapter.exact_pll(wt, batch_size=8)
    assert vec.terms.shape == (len(wt),) and np.isfinite(vec.nll_sum)
    scores = adapter.score_variants(wt, ["A4V", "I6L"])
    assert scores.shape == (2,) and np.isfinite(scores).all()
    q = adapter.native_site_conditional(wt, 3)
    assert abs(float(np.exp(q).sum()) - 1.0) < 1e-6
    h = adapter.extract_hidden_states([wt, wt[:18]], batch_size=2, pooling="mean_cls")
    assert h.shape == (2, 4, 2048) and np.isfinite(h).all()
    del adapter
    torch.cuda.empty_cache()


@pytest.mark.skipif(
    not (torch.cuda.is_available() and SAPROT_35M_DIR.is_dir()),
    reason="Requires GPU and staged saprot-35m",
)
def test_saprot_35m_seq_only_and_foldseek_3di_modes_gpu():
    """Verify SaProt-35M in both seq_only (#-masked 20-3Di marginalization) and foldseek_3di modes."""
    wt = "MKTAYIAKQRQISFVKSHFSRQLEERLG"
    s3 = "v" * len(wt)

    ad_seq = load_general_plm(
        "saprot-35m", device="cuda:0", dtype="float32", model_root=SAPROT_35M_DIR, saprot_mode="seq_only"
    )
    v_seq = ad_seq.exact_pll(wt, batch_size=8)
    assert v_seq.terms.shape == (len(wt),) and np.isfinite(v_seq.nll_sum)
    h_seq = ad_seq.extract_hidden_states([wt, wt[:20]], batch_size=2, pooling="mean_cls")
    assert h_seq.shape == (2, 4, 480) and np.isfinite(h_seq).all()
    del ad_seq
    torch.cuda.empty_cache()

    ad_3di = load_general_plm(
        "saprot-35m", device="cuda:0", dtype="float32", model_root=SAPROT_35M_DIR, saprot_mode="foldseek_3di"
    )
    v_3di = ad_3di.exact_pll(wt, batch_size=8, structure_3di=s3)
    assert v_3di.terms.shape == (len(wt),) and np.isfinite(v_3di.nll_sum)
    assert abs(v_3di.nll_sum - v_seq.nll_sum) > 1.0
    h_3di = ad_3di.extract_hidden_states([wt], batch_size=1, pooling="mean_cls", structure_3di_list=[s3])
    assert h_3di.shape == (1, 4, 480) and np.isfinite(h_3di).all()
    del ad_3di
    torch.cuda.empty_cache()


@pytest.mark.skipif(
    not (torch.cuda.is_available() and PROGEN3_219M_DIR.is_dir()),
    reason="Requires GPU and staged progen3-219m",
)
def test_progen3_219m_megablocks_eager_strict_gpu():
    """Verify ProGen3-219M loads via megablocks->eager strict conversion in BF16."""
    adapter = load_general_plm(
        "progen3-219m", device="cuda:0", dtype="bfloat16", model_root=PROGEN3_219M_DIR
    )
    wt = "MKTAYIAKQRQISFVKSHFSRQLEERLG"
    vec = adapter.exact_pll(wt)
    assert vec.terms.shape == (len(wt),) and np.isfinite(vec.nll_sum)
    h = adapter.extract_hidden_states([wt])
    assert h.shape == (1, 4, 512) and np.isfinite(h).all()
    r4 = adapter.score_r4_context(wt, [wt[:20]], budget_tokens=256)
    assert r4["k_items"] == 1 and np.isfinite(r4["nll_sum"])
    del adapter
    torch.cuda.empty_cache()


@pytest.mark.skipif(
    not (torch.cuda.is_available() and ESM2_8M_DIR.is_dir()),
    reason="Requires GPU and staged esm2-8m",
)
def test_run_structural_contacts_on_real_assay_gpu(tmp_path: Path):
    """Verify run_structural_contacts against real sites.json.gz + coverage.json on 1 structural assay."""
    from scripts.capability.general_plm.run_r3_pairwise_pll_and_contacts import run_structural_contacts

    adapter = load_general_plm("esm2-8m", device="cuda:0", dtype="float32", model_root=ESM2_8M_DIR)
    res = run_structural_contacts(adapter, ROOT, batch_size=32, max_assays=1, out_dir=tmp_path)
    assert res["n_assays"] == 1
    row = res["assays"][0]
    assert row["n_pairs"] > 100 and row["n_contacts"] > 0
    assert row["auc_minus_half"] is not None and np.isfinite(row["auc_minus_half"])
    del adapter
    torch.cuda.empty_cache()


@pytest.mark.skipif(
    not (torch.cuda.is_available() and (MSA_100M_DIR / "esm_msa1b_t12_100M_UR50S.pt").is_file()),
    reason="Requires GPU and staged esm_msa1b_t12_100M_UR50S.pt",
)
def test_msa_transformer_100m_pure_pytorch_strict_gpu():
    """Verify pure-PyTorch MSA Transformer strict load and homolog preference on real R4 unit."""
    ch = json.loads((ROOT / "results/R5/context_homologue_20260826/cohort.json").read_text(encoding="utf-8"))["protein"]
    u = next(x for x in ch["units"] if x["band"] == "id_70_90" and x["stratum"] == "retained")
    target = ch["records"][u["target"]]
    homologs = [ch["records"][p] for p in u["partners"][:4]]
    unrelated = [ch["records"][(u["target"] + 500 + i) % len(ch["records"])] for i in range(4)]

    adapter = load_general_plm("msa-transformer-100m", device="cuda:0", dtype="float32")
    r_hom = adapter.score_r4_context(target, homologs, budget_tokens=32, batch_size=16)
    r_unr = adapter.score_r4_context(target, unrelated, budget_tokens=32, batch_size=16)
    assert r_hom["k_items"] == 4 and r_unr["k_items"] == 4
    assert np.isfinite(r_hom["nll_sum"]) and np.isfinite(r_unr["nll_sum"])
    assert r_hom["nll_per_residue"] < r_unr["nll_per_residue"]
    del adapter
    torch.cuda.empty_cache()


@pytest.mark.skipif(
    not (torch.cuda.is_available() and (DAYHOFF_3B_MSA_DIR / "model.safetensors.index.json").is_file()),
    reason="Requires GPU and staged dayhoff-3b-GR-HM-c",
)
def test_dayhoff_3b_msa_bf16_r4_context_gpu():
    """Verify dayhoff-3b-msa auto-resolves dayhoff-3b-GR-HM-c and scores R4 homolog context."""
    ch = json.loads((ROOT / "results/R5/context_homologue_20260826/cohort.json").read_text(encoding="utf-8"))["protein"]
    u = next(x for x in ch["units"] if x["band"] == "id_70_90" and x["stratum"] == "retained")
    target = ch["records"][u["target"]]
    homologs = [ch["records"][p] for p in u["partners"][:2]]
    unrelated = [ch["records"][(u["target"] + 500 + i) % len(ch["records"])] for i in range(2)]

    adapter = load_general_plm("dayhoff-3b-msa", device="cuda:0", dtype="bfloat16")
    assert adapter.model.config.output_router_logits is False
    r_hom = adapter.score_r4_context(target, homologs, budget_tokens=512)
    r_unr = adapter.score_r4_context(target, unrelated, budget_tokens=512)
    assert r_hom["k_items"] == 2 and np.isfinite(r_hom["nll_sum"]) and np.isfinite(r_unr["nll_sum"])
    assert r_hom["nll_per_residue"] < r_unr["nll_per_residue"]
    del adapter
    torch.cuda.empty_cache()


def test_dayhoff_jamba_config_disables_output_router_logits_cpu(monkeypatch, tmp_path: Path):
    """Fast CPU unit test ensuring _load_dayhoff_causal enforces output_router_logits=False."""
    import transformers

    from src.capability.general_plm.adapters import _load_dayhoff_causal
    from src.capability.general_plm.registry import GENERAL_PLM_REGISTRY

    class _FakeConfig:
        num_logits_to_keep = 1
        output_router_logits = True
        use_mamba_kernels = True
        bos_token_id = 29
        eos_token_id = 27

    class _FakeTokenizer:
        unk_token_id = 0

        def convert_tokens_to_ids(self, tok: str) -> int:
            if tok == "/":
                return 28
            return 1 + (AA20.index(tok) if tok in AA20 else 30)

    class _FakeModel(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg

        def to(self, _device):
            return self

    seen: dict[str, object] = {}

    def _fake_from_pretrained(_path, **kwargs):
        cfg = kwargs["config"]
        seen["cfg_before_load"] = cfg.output_router_logits
        seen["num_logits_before_load"] = cfg.num_logits_to_keep
        return _FakeModel(cfg), {"missing_keys": [], "mismatched_keys": [], "unexpected_keys": []}

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_k: _FakeTokenizer())
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *_a, **_k: _FakeConfig())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", _fake_from_pretrained)

    adapter = _load_dayhoff_causal(
        GENERAL_PLM_REGISTRY["dayhoff-3b-msa"], tmp_path, device="cpu", dtype="float32"
    )
    assert seen["cfg_before_load"] is False
    assert seen["num_logits_before_load"] == 0
    assert adapter.model.config.output_router_logits is False
    assert adapter.provenance()["output_router_logits_overridden_to"] is False


def test_esm_optional_contact_head_missing_keys_policy_cpu():
    """Only unused esm.contact_head.regression.{weight,bias} may be omitted; other keys fail fast."""
    from src.capability.general_plm.adapters import _check_strict_esm_loading_info

    ignored = _check_strict_esm_loading_info(
        "saprot-1.3b",
        "SaProt weight mismatch",
        {
            "missing_keys": [
                "esm.contact_head.regression.bias",
                "esm.contact_head.regression.weight",
            ],
            "mismatched_keys": [],
        },
    )
    assert set(ignored) == {
        "esm.contact_head.regression.bias",
        "esm.contact_head.regression.weight",
    }

    with pytest.raises(RuntimeError, match="lm_head.decoder.weight"):
        _check_strict_esm_loading_info(
            "saprot-1.3b",
            "SaProt weight mismatch",
            {
                "missing_keys": [
                    "esm.contact_head.regression.bias",
                    "lm_head.decoder.weight",
                ],
                "mismatched_keys": [],
            },
        )


SAPROT_1_3B_DIR = Path("/home/lzp/models/general_plm_20261120/saprot-1.3b")


@pytest.mark.skipif(
    not (torch.cuda.is_available() and (SAPROT_1_3B_DIR / "pytorch_model.bin").is_file()),
    reason="Requires GPU and staged saprot-1.3b",
)
def test_saprot_1_3b_loads_without_contact_head_gpu():
    """Verify real SaProt-1.3B loads cleanly (omitting unused contact_head) and scores on GPU."""
    adapter = load_general_plm(
        "saprot-1.3b", device="cuda:0", dtype="bfloat16", model_root=SAPROT_1_3B_DIR
    )
    prov = adapter.provenance()
    assert set(prov["ignored_optional_missing_keys"]) == {
        "esm.contact_head.regression.weight",
        "esm.contact_head.regression.bias",
    }
    vec = adapter.exact_pll("MKTAYIAKQRQISFVKSHFS", batch_size=10)
    assert vec.terms.shape == (20,) and np.isfinite(vec.nll_sum)
    del adapter
    torch.cuda.empty_cache()


def test_proteinglm_10b_synthesizes_missing_safetensors_index_cpu(tmp_path: Path):
    """Verify _maybe_derive_proteinglm_mlm_dir synthesizes model.safetensors.index.json from shards."""
    from safetensors.torch import save_file

    from src.capability.general_plm.adapters import _maybe_derive_proteinglm_mlm_dir

    src = tmp_path / "proteinglm-10b-mlm"
    src.mkdir()
    save_file({"layer.0.weight": torch.zeros(2, 2)}, str(src / "model-00001-of-00002.safetensors"))
    save_file({"layer.1.weight": torch.ones(2, 2)}, str(src / "model-00002-of-00002.safetensors"))
    (src / "config.json").write_text('{"model_type": "proteinglm"}\n', encoding="utf-8")

    derived = _maybe_derive_proteinglm_mlm_dir(src)
    assert derived != src
    assert not (src / "model.safetensors.index.json").exists()
    idx_path = derived / "model.safetensors.index.json"
    assert idx_path.is_file()
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    assert idx["weight_map"] == {
        "layer.0.weight": "model-00001-of-00002.safetensors",
        "layer.1.weight": "model-00002-of-00002.safetensors",
    }
    assert idx["metadata"]["total_size"] > 0


def test_prime_690m_strips_external_automap_prefix_and_disables_flash_attention_cpu(tmp_path: Path):
    """Verify _maybe_derive_local_automap_dir strips stale HF prefix, disables flash_attention,
    and relocates deprecated transformers.modeling_utils prune imports cleanly."""
    from src.capability.general_plm.adapters import _maybe_derive_local_automap_dir

    src = tmp_path / "Prime_690M"
    src.mkdir()
    orig_cfg = {
        "model_type": "proprime",
        "flash_attention": True,
        "auto_map": {
            "AutoModelForMaskedLM": "AI4Protein/ProPrime_650M--modeling_proprime.ProPrimeForMaskedLM"
        },
    }
    (src / "config.json").write_text(json.dumps(orig_cfg), encoding="utf-8")
    (src / "modeling_proprime.py").write_text(
        "from transformers.modeling_utils import (\n"
        "    PreTrainedModel,\n"
        "    find_pruneable_heads_and_indices,\n"
        "    prune_linear_layer,\n"
        ")\n",
        encoding="utf-8",
    )

    derived = _maybe_derive_local_automap_dir(src)
    assert derived != src
    # Source config remains untouched
    assert json.loads((src / "config.json").read_text(encoding="utf-8"))["flash_attention"] is True
    # Derived config strips external repo prefix and disables flash_attention
    new_cfg = json.loads((derived / "config.json").read_text(encoding="utf-8"))
    assert new_cfg["auto_map"]["AutoModelForMaskedLM"] == "modeling_proprime.ProPrimeForMaskedLM"
    assert new_cfg["flash_attention"] is False
    # Patched modeling_proprime.py executes cleanly under installed transformers 4.57+
    patched_src = (derived / "modeling_proprime.py").read_text(encoding="utf-8")
    assert "from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer" in patched_src
    ns: dict[str, object] = {}
    exec(compile(patched_src, str(derived / "modeling_proprime.py"), "exec"), ns)
    assert callable(ns["find_pruneable_heads_and_indices"])
    assert callable(ns["prune_linear_layer"])
    assert ns["PreTrainedModel"] is not None
