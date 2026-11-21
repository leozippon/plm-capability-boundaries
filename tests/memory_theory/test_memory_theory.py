"""Unit tests for Track B memory_theory modules (B1-B4)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from src.capability.memory_theory.capacity_scaling import (
    UNIFORM_AA20_BITS,
    classify_conservation_tier,
    compute_parametric_capacity_budget,
    shannon_entropy_bits,
)
from src.capability.memory_theory.cohort_builders import make_saprot_string
from src.capability.memory_theory.gradient_probes import (
    compute_gradient_snr,
    compute_spectral_rank_metrics,
)
from src.capability.memory_theory.horizon import compute_saturation_horizon
from src.capability.memory_theory.io_utils import (
    cluster_bootstrap_mean,
    cluster_bootstrap_scaling_slope,
    paired_cluster_bootstrap_diff,
    write_json_with_sha256,
)
from src.capability.memory_theory.kv_interference import (
    ISOTROPIC_ABS_DOT_BASELINE,
    compute_auroc,
    compute_pair_kv_geometry,
)
from src.capability.memory_theory.model_hooks import (
    ARM_DIR_CANDIDATES,
    format_sequence_for_arm,
    is_masked_arm,
    resolve_model_dir,
    select_depth_quantiles,
    split_packed_kv,
)


def test_shannon_entropy_and_tiers() -> None:
    unif = np.full(20, 1.0 / 20.0)
    assert math.isclose(shannon_entropy_bits(unif), UNIFORM_AA20_BITS, rel_tol=1e-9)
    delta = np.zeros(20)
    delta[0] = 1.0
    assert math.isclose(shannon_entropy_bits(delta), 0.0, abs_tol=1e-12)
    assert classify_conservation_tier(0.4) == "conserved"
    assert classify_conservation_tier(1.5) == "moderate"
    assert classify_conservation_tier(2.5) == "variable"


def test_cluster_bootstrap_and_scaling_slope() -> None:
    rng = np.random.default_rng(7)
    x = np.array([8.0, 9.0, 10.0])  # log10 N
    # True slope -dH/d(log10 N) = 0.75
    noise = rng.normal(0.0, 0.05, size=(3, 40))
    y = (3.5 - 0.75 * (x - 8.0))[:, None] + noise
    res = cluster_bootstrap_scaling_slope(x, y, n_draws=500, seed=42)
    assert abs(res["slope_bits_per_decade"] - 0.75) < 0.05
    assert res["ci95"][0] < 0.75 < res["ci95"][1]

    diff = paired_cluster_bootstrap_diff(y[0], y[2], n_draws=500, seed=42)
    assert abs(diff["point"] - 1.5) < 0.1
    assert diff["ci95"][0] > 1.3

    mean_w = cluster_bootstrap_mean([1.0, 3.0], weights=[3.0, 1.0], n_draws=200, seed=42)
    assert math.isclose(mean_w["point"], 1.5, rel_tol=1e-9)


def test_parametric_capacity_budget_exact_arithmetic() -> None:
    bud = compute_parametric_capacity_budget(profile_gain_bits_per_res=1.8787, mean_domain_length=250.0)
    # 60M * 250 * 1.8787 = 28.1805 Gbits
    assert math.isclose(bud["reference_scenarios_gbits"]["full_uniref50_60m_x_250aa"], 28.1805, rel_tol=1e-4)
    by_label = {r["scale_label"]: r for r in bud["model_scale_curve"]}
    assert by_label["7.0B"]["regime_vs_60m_uniref50"] == "capacity_starved"
    assert by_label["151M"]["regime_vs_10m_clusters"] == "capacity_starved"
    assert by_label["30B"]["regime_vs_60m_uniref50"] == "capacity_surplus"


def test_spectral_rank_and_rank_19_bottleneck() -> None:
    # Construct a rank-19 matrix in R^{1000 x 256} (exact AA20 unembedding rank limit!)
    rng = np.random.default_rng(123)
    basis, _ = np.linalg.qr(rng.normal(size=(256, 19)))
    coeffs = rng.normal(size=(1000, 19))
    g = coeffs @ basis.T
    gram = (g.T @ g) / 1000.0
    m = compute_spectral_rank_metrics(gram, d_model=256)
    assert m["vn_effective_rank"] <= 19.01
    assert m["numerical_rank_99"] <= 19
    assert math.isclose(m["top19_variance_share"], 1.0, abs_tol=1e-6)

    # Full-rank isotropic Gaussian gradients in R^{2000 x 64}
    g_full = rng.normal(size=(2000, 64))
    m_full = compute_spectral_rank_metrics((g_full.T @ g_full) / 2000.0, d_model=64)
    assert m_full["vn_effective_rank_ratio"] > 0.85


def test_gradient_snr_unbiased_signal_vs_noise() -> None:
    rng = np.random.default_rng(99)
    # Pure zero-mean noise -> unbiased SNR ~ 0, pairwise cosine ~ 0
    noise = rng.normal(size=(64, 128))
    res_noise = compute_gradient_snr(noise, n_draws=200, seed=1)
    assert res_noise["snr_unbiased"] < 0.05
    assert abs(res_noise["mean_pairwise_cosine"]) < 0.05

    # Strong shared signal + small noise -> high SNR & positive cosine
    signal = np.ones((1, 128))
    aligned = signal + 0.1 * rng.normal(size=(64, 128))
    res_sig = compute_gradient_snr(aligned, n_draws=200, seed=1)
    assert res_sig["snr_unbiased"] > 50.0
    assert res_sig["mean_pairwise_cosine"] > 0.95


def test_kv_geometry_isotropic_vs_constructive_vs_destructive() -> None:
    rng = np.random.default_rng(2026)
    t, h, dk = 40, 4, 64
    # Isotropic random K, V -> KCE near sqrt(2/pi) and KV alignment near 0
    ka = rng.normal(size=(t, h, dk))
    va = rng.normal(size=(t, h, dk))
    kb = rng.normal(size=(t, h, dk))
    vb = rng.normal(size=(t, h, dk))
    iso = compute_pair_kv_geometry(ka, va, kb, vb)
    assert abs(iso["key_collision_excess"] - ISOTROPIC_ABS_DOT_BASELINE) < 0.08
    assert abs(iso["kv_alignment_all"]) < 0.05

    # Constructive interference: identical keys AND identical values
    con = compute_pair_kv_geometry(ka, va, ka.copy(), va.copy())
    assert con["kv_alignment_top10"] > 0.15
    assert con["kv_similarity_corr"] > 0.5

    # Destructive interference: identical keys but OPPOSITE values
    des = compute_pair_kv_geometry(ka, va, ka.copy(), -va.copy())
    assert des["kv_alignment_top10"] < -0.15
    assert des["kv_similarity_corr"] < -0.5

    assert math.isclose(compute_auroc([0.8, 0.9, 1.0], [0.1, 0.2, 0.3]), 1.0)
    assert math.isclose(compute_auroc([0.1, 0.2], [0.8, 0.9]), 0.0)


def test_saturation_horizon_interpolation() -> None:
    grid = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    # Fast-saturating curve (drops 1.0 bit by L=16, flat thereafter)
    fast_h = [4.0, 3.6, 3.3, 3.1, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    l90_fast = compute_saturation_horizon(grid, fast_h, fraction=0.90)
    assert l90_fast is not None and 7.5 <= l90_fast <= 8.5

    # Slow-saturating linear-in-log2 curve (drops 0.1 bit per doubling)
    slow_h = [4.0 - 0.1 * i for i in range(len(grid))]
    l90_slow = compute_saturation_horizon(grid, slow_h, fraction=0.90)
    assert l90_slow is not None and l90_slow > 256.0


def test_helpers_and_atomic_write(tmp_path: Path) -> None:
    assert select_depth_quantiles(12) == [0, 3, 6, 8, 11]
    assert is_masked_arm("esm2-650m")
    assert is_masked_arm("saprot-650m")
    assert not is_masked_arm("progen2-small")
    assert make_saprot_string("ACD", "VDP") == "AvCdDp"
    assert format_sequence_for_arm("ACD", "saprot-650m", saprot_mode="aa_only") == "A#C#D#"
    assert format_sequence_for_arm("ACD", "progen2-small") == "1ACD2"
    assert format_sequence_for_arm("ACD", "proteinglm-7b-clm") == "<gmask><sop><eos>ACD"
    assert "esm2_650m" in ARM_DIR_CANDIDATES["esm2-650m"]

    # Verify underscore alias resolution via MODEL_ROOT
    fake_models = tmp_path / "models"
    (fake_models / "esm2_650m").mkdir(parents=True)
    import os
    old_root = os.environ.get("MODEL_ROOT")
    os.environ["MODEL_ROOT"] = str(fake_models)
    try:
        assert resolve_model_dir("esm2-650m") == fake_models / "esm2_650m"
    finally:
        if old_root is None:
            os.environ.pop("MODEL_ROOT", None)
        else:
            os.environ["MODEL_ROOT"] = old_root

    # Verify proteinglm_qkv head-interleaved split
    import torch
    x = torch.arange(2 * 3 * 2 * 3 * 4, dtype=torch.float32).view(2, 3, 2, 3, 4)
    packed = x.reshape(2, 3, 24)
    k, v = split_packed_kv(packed, "proteinglm_qkv", n_heads=2, head_dim=4)
    assert torch.equal(k, x[:, :, :, 1, :].reshape(2, 3, 8))
    assert torch.equal(v, x[:, :, :, 2, :].reshape(2, 3, 8))

    out = tmp_path / "sample.json"
    digest = write_json_with_sha256(out, {"ok": True, "val": 1.25})
    assert out.is_file()
    sidecar = tmp_path / "sample.json.sha256"
    assert sidecar.is_file()
    assert digest in sidecar.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        write_json_with_sha256(tmp_path / "bad.json", {"nan": float("nan")})


def test_proteinglm_no_deepspeed_derivation_regression() -> None:
    """Verify ProteinGLM checkpoints derive without requiring deepspeed to be installed."""
    pglm_dir = Path("/Data/public/models_R2/proteinglm-7b-clm")
    if not (pglm_dir / "modeling_proteinglm.py").is_file():
        pytest.skip("local proteinglm-7b-clm checkpoint absent")
    from src.capability.models.proteinglm import load_derived_classes

    cfg_cls, model_cls, pkg = load_derived_classes(pglm_dir)
    assert cfg_cls.__name__ == "ProteinGLMConfig"
    assert model_cls.__name__ == "ProteinGLMForCasualLM"
    derived_text = (pkg.path / "modeling_proteinglm.py").read_text(encoding="utf-8")
    assert "import torch, deepspeed" not in derived_text


def test_discover_general_plm_nested_v2_scores(tmp_path: Path) -> None:
    """Verify nested V2 r1_ladder_and_conditionals & r1_family_distribution ingestion."""
    import json
    from src.capability.memory_theory.capacity_scaling import (
        discover_general_plm_family_conditionals,
        discover_general_plm_family_distribution,
    )

    unit_order = ["u000", "u001"]
    site_records = [
        {"unit_id": "u000", "query_position": "1", "n_contexts": 1, "row_ids": ["r0"]},
        {"unit_id": "u001", "query_position": "2", "n_contexts": 1, "row_ids": ["r1"]},
    ]
    sub = tmp_path / "esm2_8m_r1"
    sub.mkdir()
    (sub / "r1_ladder_and_conditionals_esm2-8m.json").write_text(
        json.dumps(
            {
                "arm": "esm2-8m",
                "family_conditionals": {
                    "conditional_semantics": "native_1pass_masked_conditional_over_aa20",
                    "units": [
                        {"unit_id": "u000", "partition": "evaluation", "model_logp": -2.0 * math.log(2.0)},
                        {"unit_id": "u001", "partition": "evaluation", "model_logp": -3.0 * math.log(2.0)},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    # Multi-shard FD merge check (2 shards -> complete 2 eval units)
    for idx, uid in enumerate(unit_order):
        sd = tmp_path / f"esm2_15b_fd_s{idx}"
        sd.mkdir()
        (sd / f"r1_family_distribution_esm2-15b_s{idx}of2.json").write_text(
            json.dumps(
                {
                    "arm": "esm2-15b",
                    "units": [
                        {
                            "unit_id": uid,
                            "partition": "evaluation",
                            "column_contrast": {
                                "native_nats": 100.0 * math.log(2.0),
                                "total_length": 50.0,
                                "nats_per_residue": 0.5 * math.log(2.0),
                            },
                            "composition_contrast": {"nats_per_residue": 1.0 * math.log(2.0)},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    fc_map, _, sem_map = discover_general_plm_family_conditionals([tmp_path], unit_order, site_records)
    assert math.isclose(fc_map["esm2-8m"]["u000"], 2.0, rel_tol=1e-9)
    assert math.isclose(fc_map["esm2-8m"]["u001"], 3.0, rel_tol=1e-9)
    assert sem_map["esm2-8m"] == "native_1pass_masked_conditional_over_aa20"

    fd_map = discover_general_plm_family_distribution([tmp_path], unit_order)
    assert "esm2-15b" in fd_map
    assert math.isclose(fd_map["esm2-15b"]["native_bits"]["u000"], 2.0, rel_tol=1e-9)
    assert math.isclose(fd_map["esm2-15b"]["column_null_delta_bits"]["u001"], 0.5, rel_tol=1e-9)
