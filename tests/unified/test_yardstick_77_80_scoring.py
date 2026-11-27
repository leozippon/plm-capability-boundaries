"""Unit and integration tests for the expanded 77/80-model scoring yardstick
(DPLM-2 4 checkpoints seq-only/struct-conditioned + ProteinMPNN 11 checkpoints).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.capability.core.io import sha256_file
from src.capability.unified import yardstick_77_80_scoring as y77


def test_roster_cardinality_and_conditioning_regime_partition() -> None:
    assert len(y77.DPLM2_4_ARMS) == 4
    assert len(y77.DPLM2_4_STRUCT_ARMS) == 4
    assert len(y77.PROTEINMPNN_11_ARMS) == 11
    assert len(y77.DIMA_4_ARMS) == 4
    assert len(y77.NEW_15_PRIMARY_ARMS) == 15
    assert len(y77.NEW_19_PRIMARY_ARMS) == 19
    assert len(y77.NEW_23_SCORED_ENTRIES) == 23
    assert len(y77.UNIFIED_77_SINGLE_TARGET_ARMS) == 77
    assert len(y77.UNIFIED_80_ARMS) == 80
    assert len(y77.UNIFIED_81_SINGLE_TARGET_ARMS) == 81
    assert len(y77.UNIFIED_84_ARMS) == 84
    assert len(y77.SEQUENCE_ONLY_70_ARMS) == 70
    assert len(y77.ORACLE_3D_15_ENTRIES) == 15
    assert len(y77.SCALING_LADDERS_18) == 18

    # Every single arm in UNIFIED_84_ARMS + 4 struct companions must have valid metadata
    for arm_id in (*y77.UNIFIED_84_ARMS, *y77.DPLM2_4_STRUCT_ARMS):
        meta = y77.EXTENDED_ARM_METADATA[arm_id]
        if arm_id in y77.PROTEINMPNN_11_ARMS or arm_id in y77.DPLM2_4_STRUCT_ARMS:
            assert meta.conditioning_regime == "backbone_3d_conditioned_oracle"
        else:
            assert meta.conditioning_regime == "sequence_only"

    for arm_id in y77.DIMA_4_ARMS:
        meta = y77.EXTENDED_ARM_METADATA[arm_id]
        assert meta.paradigm == "continuous_latent_diffusion"
        assert meta.conditioning_regime == "sequence_only"
        assert meta.n_layers_L == 12

    # ProteinMPNN metadata checks
    for arm_id in y77.PROTEINMPNN_11_ARMS:
        meta = y77.EXTENDED_ARM_METADATA[arm_id]
        assert meta.paradigm == "inverse_folding_gnn"
        assert meta.mpnn_noise_angstrom in (0.02, 0.10, 0.20, 0.30)
        assert meta.mpnn_variant in ("vanilla", "soluble", "ca_only")


def test_arm_slug_roundtrip_and_uniqueness() -> None:
    slugs = [y77.arm_slug(a) for a in y77.NEW_19_SCORED_ENTRIES]
    assert len(set(slugs)) == len(y77.NEW_19_SCORED_ENTRIES)
    assert y77.arm_slug("dplm2-650m|struct") == "dplm2-650m__struct"
    assert y77.arm_slug("proteinmpnn-ca-020") == "proteinmpnn-ca-020"


def test_forbidden_infra_guard_and_verified_sidecar(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    digest = y77.write_verified_json_with_sha256(good, {"ok": True, "models": ["dplm2-650m"]})
    assert digest == sha256_file(good)
    sidecar = tmp_path / "good.json.sha256"
    assert sidecar.is_file()
    assert sidecar.read_text(encoding="utf-8").strip() == f"{digest}  good.json"

    for bad_token in ("tenant-job-" + "master-0", "worker-" + "7.k8s", "10.244." + "1.9"):
        with pytest.raises(ValueError, match="forbidden cluster/pod identifier"):
            y77.assert_no_forbidden_infra_strings({"leaked": bad_token}, label="test")


def test_dplm2_struct_minus_seq_paired_contrast() -> None:
    rng = np.random.default_rng(42)
    vecs = {}
    for arm_id in y77.DPLM2_4_ARMS:
        base = rng.normal(0.40, 0.05, size=32)
        vecs[arm_id] = base
        vecs[f"{arm_id}|struct"] = base + 0.15
    res = y77._dplm2_struct_minus_seq_paired(vecs, draws=200, seed=123)
    assert res["available"] is True
    assert res["n_pairs"] == 4
    assert res["n_groups"] == 32
    for arm_id in y77.DPLM2_4_ARMS:
        col = res["bands"]["columns"][f"{arm_id}__struct_minus_seq"]
        assert abs(col["point"] - 0.15) < 1e-12
        assert col["verdict_studentized"] == "positive"


def test_allow_pending_gate_strict_vs_permissive() -> None:
    with pytest.raises(FileNotFoundError, match="missing 15/15"):
        y77._check_pending_arms(y77.NEW_15_PRIMARY_ARMS, [], benchmark="test", allow_pending=False)
    missing = y77._check_pending_arms(
        y77.NEW_15_PRIMARY_ARMS, ["dplm2-150m"], benchmark="test", allow_pending=True
    )
    assert len(missing) == 14
    assert "dplm2-150m" not in missing


def test_load_r1_score_vector_npz_and_alignment_checks(tmp_path: Path) -> None:
    sids = np.asarray(["s0", "s1", "s2"])
    p = tmp_path / "r1_ladder_scores_dplm2-150m.npz"
    # Deliberately reversed order in .npz to verify join-by-id alignment
    np.savez_compressed(p, sample_ids=np.asarray(["s2", "s0", "s1"]), scores=np.asarray([3.0, 1.0, 2.0]))
    vec = y77._load_r1_score_vector_npz("dplm2-150m", p, sids)
    assert np.allclose(vec, [1.0, 2.0, 3.0])

    # Non-finite rejection
    p_bad = tmp_path / "bad.npz"
    np.savez_compressed(p_bad, sample_ids=sids, scores=np.asarray([1.0, np.nan, 3.0]))
    with pytest.raises(ValueError, match="missing/non-finite"):
        y77._load_r1_score_vector_npz("dplm2-150m", p_bad, sids)


def test_load_new_r4_main_arm_schema_and_validation(tmp_path: Path) -> None:
    units = []
    for idx in range(800):
        units.append(
            {
                "key": f"u{idx}",
                "band": "id_50_70",
                "group": idx % 710,
                "conditions": {
                    c: {"k_items": 1, "pll_8fold": {"nll_per_residue": 2.0 - (0.2 if c == "homologue" else 0.0)}}
                    for c in ("position_only", "mono_shuffled", "no_context", "unrelated", "homologue")
                },
                "pll_8fold": {
                    "hom_beats_unrelated": 1.0,
                    "hom_beats_mono_shuffled": 1.0,
                    "fractional_reduction_vs_unrelated": 0.1,
                    "fractional_reduction_vs_shuffled": 0.1,
                },
            }
        )
    p = tmp_path / "r4_unaligned_dplm2-150m.json"
    p.write_text(
        json.dumps(
            {
                "schema": "unified_r4_unaligned_scoring_arm_v1",
                "arm": "dplm2-150m",
                "budget_tokens": 1024,
                "provenance": {"dtype_realised": "torch.float32"},
                "units": units,
            }
        ),
        encoding="utf-8",
    )
    loaded = y77._load_new_r4_main_arm("dplm2-150m", p, tmp_path)
    assert loaded["arm_id"] == "dplm2-150m"
    assert loaded["conditioning_regime"] == "sequence_only"
    assert len(loaded["units"]) == 800
    assert loaded["units"][0]["nll_homologue"] == pytest.approx(1.8)


def _synthetic_helix_bb(seq: str) -> dict[str, object]:
    """Generate a smooth right-handed synthetic 3D backbone (L, 4, 3) [N, CA, C, O] for unit tests."""
    L = len(seq)
    t = np.arange(L, dtype=np.float32)
    angle = t * (2.0 * np.pi / 3.6)
    ca = np.stack([2.3 * np.cos(angle), 2.3 * np.sin(angle), 1.5 * t], axis=-1)
    n_atom = ca + np.array([-0.8, -0.5, -0.5], dtype=np.float32)
    c_atom = ca + np.array([0.8, 0.4, 0.5], dtype=np.float32)
    o_atom = c_atom + np.array([0.0, 0.9, 0.3], dtype=np.float32)
    coords = np.stack([n_atom, ca, c_atom, o_atom], axis=1)
    return {
        "sequence": seq,
        "coords": coords,
        "res_mask": np.ones(L, dtype=np.bool_),
        "ca_mask": np.ones(L, dtype=np.bool_),
    }


@pytest.mark.parametrize("arm_id", ["proteinmpnn-v-020", "proteinmpnn-ca-020"])
def test_scoring_bridge_live_proteinmpnn_all_methods(arm_id: str) -> None:
    """Live CPU smoke test of ScoringAdapterBridge against real staged ProteinMPNN weights."""
    from scripts.capability.unified.run_dplm2_mpnn_scoring_h200 import ScoringAdapterBridge

    mpnn_root = Path("/home/lzp/models/general_plm_20261120/proteinmpnn_all")
    if not mpnn_root.is_dir():
        pytest.skip("Staged proteinmpnn_all weights absent on this host")

    bridge = ScoringAdapterBridge(arm_id, device="cpu", dtype="float32", model_root=None)
    wt = "MKTAYIAKQRQISFVKSHFSRQLE"
    ctx = "MKTAYIAKQRQISFVKSHFSRQLD"
    bb_wt = _synthetic_helix_bb(wt)
    bb_ctx = _synthetic_helix_bb(ctx)

    # 1. Single & double mutant scoring (R1/R2)
    muts = ["K2A", "T3G", "K2A:T3G"]
    sc = bridge.score_variants(wt, muts, bb=bb_wt, batch_size=8)
    assert sc.shape == (3,)
    assert np.isfinite(sc).all()

    # 2. Hidden state 1024-d R_mid (0:512) & R_last (512:1024) responsiveness
    wt_r = bridge.extract_hidden_states_projected([wt], bb=bb_wt, batch_size=2)
    mut_r = bridge.extract_hidden_states_projected(["MATAYIAKQRQISFVKSHFSRQLE"], bb=bb_wt, batch_size=2)
    assert wt_r.shape == (1, 1024)
    delta_r = mut_r - wt_r
    assert float(np.linalg.norm(delta_r[:, :512])) > 1e-6  # R_mid responds to mutation!
    assert float(np.linalg.norm(delta_r[:, 512:])) > 1e-6  # R_last responds to mutation!

    # 3. R3 double-mutant epistasis cycle (non-degenerate)
    seqs_r3 = [
        wt,
        "MATAYIAKQRQISFVKSHFSRQLE",
        "MKGAYIAKQRQISFVKSHFSRQLE",
        "MAGAYIAKQRQISFVKSHFSRQLE",
    ]
    lls = bridge.score_r3_states_batch(seqs_r3, bb=bb_wt, batch_size=4)
    eps = float(lls[3] - lls[1] - lls[2] + lls[0])
    assert np.isfinite(eps) and abs(eps) > 1e-6

    # 4. E01 exact_pll_terms + gathered (L, 20)
    terms, nll_sum, gathered = bridge.exact_pll_terms(wt, bb=bb_wt, batch_size=8, return_gathered=True)
    assert terms.shape == (len(wt),)
    assert gathered is not None and gathered.shape == (len(wt), 20)
    assert abs(float(terms.sum()) - nll_sum) < 1e-4

    # 5. R4 Main Table 2-chain spatial-offset context scoring
    r4_noc = bridge.score_r4_condition("u0", wt, [], target_bb=bb_wt)
    r4_hom = bridge.score_r4_condition("u0", wt, [ctx], target_bb=bb_wt, context_bbs=[bb_ctx])
    assert r4_noc["k_items"] == 0 and r4_hom["k_items"] == 1
    assert r4_hom["n_cross_chain_edges"] > 0
    assert np.isfinite(r4_hom["pll_8fold"]["nll_per_residue"])

    # 6. R5 DMS with context
    sc_ctx = bridge.score_dms_with_context(
        wt, muts, [ctx], wt_bb=bb_wt, context_bbs=[bb_ctx], batch_size=8
    )
    assert sc_ctx.shape == (3,)
    assert np.isfinite(sc_ctx).all()


@pytest.mark.parametrize("arm_id", ["dima-esm2-8m", "dima-saprot-35m"])
def test_scoring_bridge_live_dima_all_methods(arm_id: str) -> None:
    """Live CPU/GPU smoke test of ScoringAdapterBridge against real staged DiMA checkpoints."""
    from scripts.capability.unified.run_dplm2_mpnn_scoring_h200 import ScoringAdapterBridge

    dima_root = Path("/home/lzp/models/general_plm_20261120/dima_all")
    if not dima_root.is_dir():
        pytest.skip("Staged dima_all weights absent on this host")

    bridge = ScoringAdapterBridge(arm_id, device="cpu", dtype="float32", model_root=None)
    wt = "MKTAYIAKQRQISFVKSHFSRQLE"
    ctx = "MKTAYIAKQRQISFVKSHFSRQLD"
    muts = ["K2A", "T3G", "K2A:T3G"]

    sc = bridge.score_variants(wt, muts, batch_size=8, multi_exact_pll=False)
    assert sc.shape == (3,) and np.isfinite(sc).all()

    wt_r = bridge.extract_hidden_states_projected([wt], batch_size=2)
    mut_r = bridge.extract_hidden_states_projected(["MATAYIAKQRQISFVKSHFSRQLE"], batch_size=2)
    assert wt_r.shape == (1, 1024)
    delta_r = mut_r - wt_r
    assert float(np.linalg.norm(delta_r[:, :512])) > 1e-6
    assert float(np.linalg.norm(delta_r[:, 512:])) > 1e-6

    terms, nll_sum, gathered = bridge.exact_pll_terms(wt, batch_size=8, return_gathered=True)
    assert terms.shape == (len(wt),) and gathered is not None and gathered.shape == (len(wt), 20)
    assert abs(float(terms.sum()) - nll_sum) < 1e-4

    nll8 = bridge.pll_8fold_nll_sum(wt, n_folds=8)
    assert np.isfinite(nll8) and nll8 > 0.0

    r4_hom = bridge.score_r4_condition("u0", wt, [ctx], budget_tokens=510)
    assert r4_hom["k_items"] == 1 and r4_hom["budget_tokens"] == 510 and np.isfinite(r4_hom["nll_per_residue"])

    sc_ctx = bridge.score_dms_with_context(wt, muts, [ctx], budget_tokens=510, batch_size=8)
    assert sc_ctx.shape == (3,) and np.isfinite(sc_ctx).all()

