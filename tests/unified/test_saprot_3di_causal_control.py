"""Tests for fixed-20-AA-target SaProt 3Di structural correspondence causal control."""
from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
import pytest
import torch

from src.capability.core.io import sha256_file
from src.capability.memory_theory.gradient_probes import (
    compute_gradient_snr,
    compute_spectral_rank_metrics,
)
from src.capability.unified.roster import REPO_ROOT
from src.capability.unified.saprot_3di_causal_control import (
    CANONICAL_AA20,
    CONDITIONS,
    DEFAULT_COHORT_PATH,
    DEFAULT_OUTPUT_PATH,
    PAIRED_CONTRASTS,
    _bootstrap_counts_matrix,
    build_aa20_token_group_matrix,
    build_condition_sequences,
    compute_aa20_marginal_logits_and_loss,
    compute_snr_and_cosine_point_and_boot,
    compute_spectral_point_and_boot_exact_f64,
    load_cath_3d_domains,
)


def test_condition_sequence_construction_invariants() -> None:
    """Within-domain shuffle preserves per-domain 3Di counts; cross-domain shuffle preserves global pool."""
    domains = load_cath_3d_domains(REPO_ROOT / DEFAULT_COHORT_PATH, max_domains=12)
    conds = build_condition_sequences(domains, seed=20261120)

    assert set(conds.keys()) == set(CONDITIONS)
    global_orig = Counter()
    global_cross = Counter()

    for idx, dom in enumerate(domains):
        s_aa = conds["aa_only"][idx]
        s_cor = conds["correct_3di"][idx]
        s_w = conds["shuffled_3di"][idx]
        s_x = conds["cross_domain_shuffled_3di"][idx]

        # Identical character length and identical amino-acid sequence in all 4 conditions
        for s in (s_aa, s_cor, s_w, s_x):
            assert len(s) == 2 * dom.length
            assert s[0::2] == dom.aa_seq

        assert s_aa[1::2] == "#" * dom.length
        assert s_cor == dom.saprot_seq
        # Within-domain shuffle preserves exact 3Di character multiset while changing order
        assert Counter(s_w[1::2]) == Counter(dom.three_di_seq.lower())
        assert s_w[1::2] != dom.three_di_seq.lower()
        global_orig.update(dom.three_di_seq.lower())
        global_cross.update(s_x[1::2])

    # Cross-domain shuffle is an exact permutation of the global 3Di residue pool
    assert global_cross == global_orig


def test_aa20_marginal_logsumexp_simplex_and_vocab_grouping() -> None:
    """AA20 logsumexp marginal probabilities sum to 1.0 and match exact grouped probabilities."""
    # Fake SaProt-style tokenizer with exact 20 AA x 21 (20 3Di + '#') tokens
    vocab: dict[str, int] = {"<cls>": 0, "<pad>": 1, "<eos>": 2, "<unk>": 3, "<mask>": 4}
    tid = 5
    for aa in CANONICAL_AA20:
        for suf in "#acdefghiklmnpqrstvwy":
            vocab[f"{aa}{suf}"] = tid
            tid += 1

    class _DummyTok:
        def get_vocab(self) -> dict[str, int]:
            return vocab

    mat21 = build_aa20_token_group_matrix(_DummyTok(), mode="21_all_aa_tokens")
    mat20 = build_aa20_token_group_matrix(_DummyTok(), mode="20_3di_only")
    assert mat21.shape == (20, 21)
    assert mat20.shape == (20, 20)

    rng = torch.Generator().manual_seed(42)
    logits = torch.randn((7, len(vocab)), generator=rng, dtype=torch.float32)
    targets = torch.tensor([0, 5, 19, 3, 11, 2, 14], dtype=torch.long)

    aa_logits, loss, simp_err = compute_aa20_marginal_logits_and_loss(logits, targets, mat21)
    assert aa_logits.shape == (7, 20)
    assert loss.shape == (7,)
    assert simp_err < 1e-6

    # Verify exact algebraic identity: softmax(logsumexp_group) == sum_{v in G(a)} P(v | v in union G)
    full_p = torch.softmax(logits, dim=-1)
    grouped_mass = torch.stack([full_p[:, mat21[a]].sum(dim=-1) for a in range(20)], dim=-1)
    expected_p = grouped_mass / grouped_mass.sum(dim=-1, keepdim=True)
    actual_p = torch.softmax(aa_logits, dim=-1)
    assert float(torch.max(torch.abs(actual_p - expected_p)).item()) < 1e-6


def test_vectorized_snr_cosine_and_f64_spectral_match_reference() -> None:
    """Vectorized SNR/cosine and exact float64 spectral bootstrap match gradient_probes references."""
    rng = np.random.default_rng(1234)
    n_dom, d_model = 24, 32
    # Construct rank-19 token gradients (simulating final-layer 20-AA simplex rank cap)
    basis = rng.normal(size=(19, d_model))
    dom_tok_grads = [
        (rng.normal(size=(10 + (i % 5), 19)) @ basis).astype(np.float32) for i in range(n_dom)
    ]
    dom_tok_counts = np.array([g.shape[0] for g in dom_tok_grads], dtype=np.int64)
    mean_grads = np.stack([g.astype(np.float64).mean(axis=0) for g in dom_tok_grads], axis=0)

    # 1. Verify SNR & pairwise cosine match compute_gradient_snr to machine precision
    ref_snr = compute_gradient_snr(mean_grads, n_draws=20, seed=99)
    w_boot = _bootstrap_counts_matrix(n_dom, 64, seed=99)
    vec_p, vec_d = compute_snr_and_cosine_point_and_boot(mean_grads, w_boot)
    assert vec_p["mean_pairwise_cosine"] == pytest.approx(ref_snr["mean_pairwise_cosine"], abs=1e-12)
    assert vec_p["snr_unbiased"] == pytest.approx(ref_snr["snr_unbiased"], abs=1e-12)
    assert vec_p["snr_raw"] == pytest.approx(ref_snr["snr_raw"], abs=1e-12)
    assert vec_d["mean_pairwise_cosine"].shape == (64,)

    # 2. Verify float64 spectral point + bootstrap matches compute_spectral_rank_metrics & respects rank<=19
    gram_ref = sum(g.astype(np.float64).T @ g.astype(np.float64) for g in dom_tok_grads) / float(dom_tok_counts.sum())
    ref_spec = compute_spectral_rank_metrics(gram_ref, d_model)
    spec_p, spec_d = compute_spectral_point_and_boot_exact_f64(
        dom_tok_grads, dom_tok_counts, d_model, w_boot[:16], device="cpu", eig_batch_size=8
    )
    for k, v in ref_spec.items():
        assert spec_p[k] == pytest.approx(v, abs=1e-5)
    # Exact rank <= 19 -> top19_variance_share == 1.0 and rv_singular_effective_rank <= 19.0
    assert spec_p["top19_variance_share"] == pytest.approx(1.0, abs=1e-10)
    assert spec_p["rv_singular_effective_rank"] <= 19.0 + 1e-4
    assert float(np.max(spec_d["rv_singular_effective_rank"])) <= 19.0 + 1e-4


def test_sealed_saprot_3di_causal_control_artifact_integrity() -> None:
    """Validate sealed r4_saprot_3di_aa_target_causal_control.json, sidecar, and verification gate."""
    out_path = REPO_ROOT / DEFAULT_OUTPUT_PATH
    if not out_path.is_file():
        pytest.skip("r4_saprot_3di_aa_target_causal_control.json not yet built")

    sidecar = out_path.with_name(out_path.name + ".sha256")
    assert sidecar.is_file()
    expected_sha = sidecar.read_text(encoding="utf-8").strip().split()[0]
    assert sha256_file(out_path) == expected_sha

    doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert doc["completion_status"] == "complete"
    assert doc["verification_gate"]["passed"] is True
    assert doc["reproduction_gate"]["passed"] is True
    assert doc["protocol"]["output_simplex_dim"] == 20
    assert doc["protocol"]["output_simplex_dof"] == 19
    assert doc["protocol"]["allow_tf32"] is False

    for arm in ("saprot-35m", "saprot-650m", "saprot-1.3b"):
        assert arm in doc["models"]
        mrec = doc["models"][arm]
        assert mrec["verification_gate"]["passed"] is True
        assert mrec["n_domains"] >= 100
        for slot in ("final_layer", "middle_layer"):
            lrec = mrec["highlights"][slot]
            for cond in CONDITIONS:
                cm = lrec["conditions"][cond]
                assert math.isfinite(cm["mean_pairwise_cosine"]["point"])
                assert math.isfinite(cm["vn_effective_rank"]["point"])
                assert math.isfinite(cm["top19_variance_share"]["point"])
                assert math.isfinite(cm["snr_unbiased"]["point"])
            for cname, _, _ in PAIRED_CONTRASTS:
                assert cname in lrec["paired_contrasts"]
        # Predictive sanity check: true residue-aligned 3Di must beat aa_only & shuffled_3di on 20-AA NLL
        loss_c = mrec["aa20_loss"]["paired_contrasts"]
        assert loss_c["correct_3di_minus_aa_only"]["mean_aa20_nll_nats"]["estimate"] < 0.0
        assert loss_c["correct_3di_minus_shuffled_3di"]["mean_aa20_nll_nats"]["estimate"] < 0.0
