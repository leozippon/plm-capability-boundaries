"""Unit tests for open_model_audit metrics, Horvitz-Thompson estimation, and cohort construction."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.capability.open_model_audit.cohort import (
    AuditSequenceRecord,
    OpenAuditBundle,
    Paired3DRecord,
    atomic_write_json_with_sha256,
    atomic_write_jsonl,
    parse_pdb_ncaco_from_lines,
    scan_uniref50_depth_buckets,
    seq_sha256,
)
from src.capability.open_model_audit.metrics import (
    AA20,
    DEPTH_BUCKET_ORDER,
    H0_BITS,
    LN2,
    aa20_log_probs,
    aa20_restricted_logits,
    assign_depth_bucket,
    compute_horvitz_thompson_corpus_estimate,
    compute_multitrack_b_ext,
    compute_temperature_calibrated_bits,
    compute_three_tier_decomposition,
    count_model_parameters,
    deterministic_fixed_rate_mask,
    deterministic_loo_positions,
    deterministic_mdm_stratum_mask,
)


def test_aa20_renormalization_strips_non_aa_tokens():
    """Massive logits on <eos>/<pad>/X must not affect strictly renormalized AA20 entropy."""
    vocab_size = 33
    aa20_ids = list(range(4, 24))
    logits = torch.zeros((5, vocab_size), dtype=torch.float32)
    # Put huge mass on non-AA special tokens (<pad>=1, <eos>=2, X=24, <mask>=32)
    logits[:, 1] = 100.0
    logits[:, 2] = 50.0
    logits[:, 24] = 80.0
    logits[:, 32] = 90.0

    lp20 = aa20_log_probs(logits, aa20_ids)
    assert lp20.shape == (5, 20)
    # Uniform over the 20 canonical AAs -> exact log(1/20) and entropy H0 = log2(20)
    assert torch.allclose(lp20.exp().sum(dim=-1), torch.ones(5), atol=1e-6)
    entropy_bits = float((-lp20.exp() * lp20).sum(dim=-1).mean().item() / LN2)
    assert entropy_bits == pytest.approx(H0_BITS, abs=1e-5)

    # Also verify non-uniform AA logits stay invariant to non-AA perturbation
    logits[:, 4:24] = torch.randn((5, 20))
    lp_a = aa20_log_probs(logits, aa20_ids)
    logits_perturbed = logits.clone()
    logits_perturbed[:, 0:4] = -999.0
    logits_perturbed[:, 24:] = +999.0
    lp_b = aa20_log_probs(logits_perturbed, aa20_ids)
    assert torch.allclose(lp_a, lp_b, atol=1e-6)


def test_temperature_scaling_eq3b_scale_invariance():
    """Scaling logits z -> alpha * z inflates raw CE while calibrated H_cal stays invariant."""
    rng = np.random.default_rng(42)
    z = rng.normal(0.0, 1.0, size=(2000, 20)).astype(np.float32)
    # Generate targets correlated with z so optimal T is finite and nontrivial
    probs = np.exp(z - z.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    y = np.asarray([rng.choice(20, p=p) for p in probs], dtype=np.int64)

    base = compute_temperature_calibrated_bits(z, y)
    overconf = compute_temperature_calibrated_bits(z * 5.0, y)

    assert overconf["raw_bits_per_res"] > base["raw_bits_per_res"] + 1.0
    assert overconf["delta_cal_bits"] > 1.0
    assert overconf["calibrated_bits_per_res"] == pytest.approx(base["calibrated_bits_per_res"], abs=1e-3)
    assert overconf["optimal_temperature"] == pytest.approx(5.0 * base["optimal_temperature"], rel=0.05)


def test_three_tier_conservation_eq3a_exact():
    res = compute_three_tier_decomposition(
        h_seen=1.20,
        h_unseen_homolog=2.10,
        h_unseen_fold=3.45,
        h_seen_cal=1.18,
        h_unseen_homolog_cal=2.05,
        h_unseen_fold_cal=3.20,
    )
    assert res["conservation_abs_error"] < 1e-12
    assert res["delta_h_univ"] == pytest.approx(H0_BITS - 3.45, abs=1e-12)
    assert res["delta_h_spec"] == pytest.approx(3.45 - 2.10, abs=1e-12)
    assert res["delta_h_res"] == pytest.approx(2.10 - 1.20, abs=1e-12)
    assert res["calibrated"]["delta_cal_unseen_fold"] == pytest.approx(0.25, abs=1e-12)


def test_horvitz_thompson_corpus_estimator_and_apparent_bpp():
    pop = {
        "singleton": {"n_clusters": 20_000_000, "total_residues": 5_000_000_000},
        "small": {"n_clusters": 20_000_000, "total_residues": 5_000_000_000},
        "medium": {"n_clusters": 10_000_000, "total_residues": 3_000_000_000},
        "deep": {"n_clusters": 8_000_000, "total_residues": 1_500_000_000},
        "ultra_deep": {"n_clusters": 2_000_000, "total_residues": 500_000_000},
    }
    eval_stats = {
        "singleton": {"bits_per_res": 3.821928094887363, "se_bits_per_res": 0.02, "n_sequences": 256, "n_residues": 40000},
        "small": {"bits_per_res": 3.321928094887363, "se_bits_per_res": 0.02, "n_sequences": 256, "n_residues": 40000},
        "medium": {"bits_per_res": 2.821928094887363, "se_bits_per_res": 0.02, "n_sequences": 256, "n_residues": 40000},
        "deep": {"bits_per_res": 2.321928094887363, "se_bits_per_res": 0.02, "n_sequences": 256, "n_residues": 40000},
        "ultra_deep": {"bits_per_res": 1.821928094887363, "se_bits_per_res": 0.02, "n_sequences": 256, "n_residues": 40000},
    }
    # Expected B_total = 5B*0.5 + 5B*1.0 + 3B*1.5 + 1.5B*2.0 + 0.5B*2.5 = 16.25e9 bits
    p_non_emb = 1_000_000_000  # 1B params -> c_app = 16.25 bits/param
    ht = compute_horvitz_thompson_corpus_estimate(
        eval_stats, pop, n_params_non_emb=p_non_emb, n_params_total=1_050_000_000
    )
    assert ht["B_total_bits"] == pytest.approx(16.25e9, rel=1e-9)
    assert ht["c_app_bits_per_non_emb_param"] == pytest.approx(16.25, rel=1e-9)
    assert ht["B_total_se_bits"] > 0.0


def test_multitrack_b_ext_chain_rule():
    res = compute_multitrack_b_ext(h_conditioned=1.85, h_unconditioned=3.50)
    assert res["b_ext_bits_per_res"] == pytest.approx(1.65, abs=1e-12)
    assert res["chain_rule_abs_error"] < 1e-12


def test_depth_bucket_boundaries_and_deterministic_masks():
    assert assign_depth_bucket(1) == "singleton"
    assert assign_depth_bucket(2) == "small"
    assert assign_depth_bucket(16) == "small"
    assert assign_depth_bucket(17) == "medium"
    assert assign_depth_bucket(128) == "medium"
    assert assign_depth_bucket(129) == "deep"
    assert assign_depth_bucket(1024) == "deep"
    assert assign_depth_bucket(1025) == "ultra_deep"

    m1 = deterministic_fixed_rate_mask(7, 100, draw_idx=3, mask_rate=0.15)
    m2 = deterministic_fixed_rate_mask(7, 100, draw_idx=3, mask_rate=0.15)
    assert np.array_equal(m1, m2)
    assert int(m1.sum()) == 15

    # 16 MDM strata should span monotonically from low to high mask counts
    counts = [deterministic_mdm_stratum_mask(0, 160, stratum=s, n_strata=16)[1] for s in range(16)]
    assert counts[0] < counts[8] < counts[15]
    assert min(counts) >= 1 and max(counts) <= 160

    probes = deterministic_loo_positions(3, 80, n_probes=16)
    assert len(probes) == 16 and len(set(probes.tolist())) == 16


def test_count_model_parameters_handles_tied_embeddings():
    class TinyLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(32, 16)
            self.ff = nn.Linear(16, 16, bias=False)
            self.lm_head = nn.Linear(16, 32, bias=False)
            self.lm_head.weight = self.emb.weight  # weight tying

    m = TinyLM()
    pc = count_model_parameters(m)
    assert pc["n_params_emb"] == 32 * 16
    assert pc["n_params_non_emb"] == 16 * 16
    assert pc["n_params_total"] == 32 * 16 + 16 * 16


def test_scan_uniref50_depth_buckets_synthetic(tmp_path: Path):
    fasta = tmp_path / "mini_uniref50.fasta"
    seqs = [
        ("UniRef50_P00001", 1, "A" * 60),
        ("UniRef50_P00002", 5, "C" * 70),
        ("UniRef50_P00003", 50, "D" * 80),
        ("UniRef50_P00004", 300, "E" * 90),
        ("UniRef50_P00005", 2048, "F" * 100),
        ("UniRef50_P00006", 1, "A" * 20),  # too short for eligible, still in all!
    ]
    with fasta.open("w", encoding="utf-8") as fh:
        for cid, n, s in seqs:
            fh.write(f">{cid} test protein n={n} Tax=Ecoli RepID=X\n{s}\n")

    out = scan_uniref50_depth_buckets(
        fasta, n_per_bucket=4, min_len=50, max_len=384, swissprot_accs={"P00001"}
    )
    assert out["total_fasta_records_scanned"] == 6
    assert out["population_buckets_all"]["singleton"]["n_clusters"] == 2
    assert out["population_buckets_all"]["singleton"]["total_residues"] == 80
    assert out["population_buckets_eligible"]["singleton"]["n_clusters"] == 1
    assert out["population_buckets_eligible"]["singleton"]["total_residues"] == 60
    assert len(out["sampled_records"]) == 5
    by_b = {r.depth_bucket: r for r in out["sampled_records"]}
    assert set(by_b.keys()) == set(DEPTH_BUCKET_ORDER)
    assert by_b["singleton"].source_db == "swissprot"


def test_parse_pdb_ncaco_and_bundle_roundtrip(tmp_path: Path):
    pdb_lines = [
        "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 90.00           N  ",
        "ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00 91.00           C  ",
        "ATOM      3  C   ALA A   1       3.000   2.000   3.000  1.00 92.00           C  ",
        "ATOM      4  O   ALA A   1       3.000   3.000   3.000  1.00 93.00           O  ",
        "ATOM      5  N   GLY A   2       4.000   2.000   3.000  1.00 88.00           N  ",
        "ATOM      6  CA  GLY A   2       5.000   2.000   3.000  1.00 89.00           C  ",
        "ATOM      7  C   GLY A   2       6.000   2.000   3.000  1.00 87.00           C  ",
        "ATOM      8  O   GLY A   2       6.000   3.000   3.000  1.00 86.00           O  ",
    ]
    seq, coords, bf = parse_pdb_ncaco_from_lines(pdb_lines)
    assert seq == "AG"
    assert coords.shape == (2, 4, 3)
    assert np.isfinite(coords).all()
    assert bf[0] == pytest.approx(91.0)
    assert bf[1] == pytest.approx(89.0)

    # Verify OpenAuditBundle roundtrip loading
    splits = tmp_path / "splits"
    splits.mkdir(parents=True)
    rec = AuditSequenceRecord(
        record_id="r0",
        split="seen_pretrain_buckets",
        sub_stratum="singleton",
        sequence="ACDEFGHIKLMNPQRSTVWY" * 3,
        seq_len=60,
        seq_sha256=seq_sha256("ACDEFGHIKLMNPQRSTVWY" * 3),
        cluster_size=1,
        depth_bucket="singleton",
    )
    atomic_write_jsonl(splits / "seen_pretrain_buckets.jsonl", [rec.to_dict()])
    atomic_write_jsonl(splits / "seen_cath_matched.jsonl", [])
    atomic_write_jsonl(splits / "unseen_homolog.jsonl", [])
    atomic_write_jsonl(splits / "unseen_fold.jsonl", [])
    p3 = Paired3DRecord(
        record_id="p0",
        domain_group="natural_cath_s40",
        sequence="AG",
        seq_len=2,
        seq_sha256=seq_sha256("AG"),
        three_di_seq="DP",
        coords_key="nat_000",
        source_kind="test",
    )
    atomic_write_jsonl(splits / "paired_3d_panel.jsonl", [p3.to_dict()])
    np.savez_compressed(splits / "panel_3d_coords.npz", nat_000=coords)
    pop = {b: {"n_clusters": 10, "total_residues": 600} for b in DEPTH_BUCKET_ORDER}
    atomic_write_json_with_sha256(
        tmp_path / "bundle_manifest.json",
        {
            "schema_version": "open_model_audit_bundle_v1",
            "population_buckets_all": pop,
            "population_buckets_eligible": pop,
        },
    )
    bundle = OpenAuditBundle(tmp_path)
    assert len(bundle.seen_pretrain_buckets) == 1
    assert bundle.load_coords("nat_000").shape == (2, 4, 3)
