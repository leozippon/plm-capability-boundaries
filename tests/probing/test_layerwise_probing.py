"""Unit and end-to-end tests for the frozen layerwise hidden-state linear probe suite."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_script_module(rel_path: str, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_layerwise_probes = _load_script_module(
    "scripts/capability/probing/run_layerwise_probes.py", "run_layerwise_probes"
)
train_probe_trajectory = _load_script_module(
    "scripts/capability/probing/train_probe_trajectory.py", "train_probe_trajectory"
)
from src.capability.probing.analysis import (
    PROBE_CHECKPOINT_SCHEMA,
    PROBE_SUMMARY_SCHEMA,
    aggregate_probe_results,
    compute_layer_retention_metrics,
    evaluate_checkpoint_probes,
)
from src.capability.probing.extractor import (
    canonical_objective,
    extract_layerwise_hidden_states,
    forward_layerwise_hidden_states,
    load_stage2_checkpoint,
    paradigm_display_name,
)
from src.capability.probing.probes import (
    DEFAULT_RIDGE_LAMBDAS,
    FoldSufficientStats,
    compute_residue_burial_targets,
    compute_weighted_temperature_calibrated_bits,
    evaluate_burial_layerwise_probe,
    evaluate_contact_layerwise_probe,
    evaluate_megascale_layerwise_probe,
    evaluate_unseen_fold_temperature_calibration,
    fit_ridge_from_stats_cv,
    fit_ridge_svd_cv,
)
from src.capability.supply_demand.corpus import (
    BUNDLE_SCHEMA,
    AnnotatedSequence,
    atomic_write_gzip_json,
    atomic_write_json_with_sha256,
    save_packed_split_npz,
)
from src.capability.supply_demand.evaluation import evaluate_split_native
from src.capability.supply_demand.model import VariableLengthProteinTransformer
from src.capability.supply_demand.objectives import (
    PackedSequenceStore,
    collate_variable_length_sequences,
    encode_aa_sequence,
    prepare_ar_batch,
)
from src.capability.supply_demand.spec import (
    AA20,
    SCALE_SPECS,
    Stage2RunSpec,
    assert_no_forbidden_infra_strings,
)
from src.capability.supply_demand.training import CHECKPOINT_SCHEMA, atomic_torch_save


def _rand_seq(rng: np.random.Generator, length: int) -> str:
    idx = rng.integers(0, len(AA20), size=length)
    return "".join(AA20[int(i)] for i in idx)


def _make_synthetic_contact_domains(
    rng: np.random.Generator,
    n_domains: int = 5,
) -> list[dict[str, Any]]:
    domains: list[dict[str, Any]] = []
    for i in range(n_domains):
        L = 34 + (i % 3) * 2
        seq = _rand_seq(rng, L)
        # Folded helix-hairpin 3D trace with both <8 A contacts and >8 A non-contacts for |i-j|>=24
        t = np.linspace(0.0, 2.0 * math.pi, L)
        coords = np.stack(
            [
                6.5 * np.cos(t),
                6.5 * np.sin(t),
                1.5 * np.sin(2.0 * t) + 0.15 * rng.normal(size=L),
            ],
            axis=-1,
        ).astype(np.float32)
        mask = np.ones(L, dtype=np.bool_)
        if i % 3 == 0:
            mask[-1] = False
        domains.append(
            {
                "domain_id": f"SYN{i:03d}",
                "sequence": seq,
                "length": L,
                "cath_superfamily": f"1.10.{100 + i}.10",
                "cath_topology": f"1.10.{100 + i}",
                "cb_coords": coords,
                "cb_mask": mask,
            }
        )
    return domains


def _make_synthetic_megascale(rng: np.random.Generator) -> dict[str, Any]:
    domains: list[dict[str, Any]] = []
    strata_counts = [
        ("seen_pool_natural", 6),
        ("unseen_fold_natural", 5),
        ("de_novo_design", 5),
    ]
    for stratum, n_wt in strata_counts:
        for w in range(n_wt):
            L = 20 + (w % 3) * 4
            wt = _rand_seq(rng, L)
            variants: list[list[Any]] = []
            for v in range(18):
                pos = int((v * 3 + w) % L)
                wt_aa = wt[pos]
                mut_aa = AA20[(AA20.index(wt_aa) + 1 + (v % 19)) % 20]
                ddg = float(rng.normal(loc=-0.5, scale=1.0))
                variants.append([pos, wt_aa, mut_aa, ddg])
            domains.append(
                {
                    "wt_name": f"{stratum}_wt{w}",
                    "wildtype": wt,
                    "length": L,
                    "stratum": stratum,
                    "variants": variants,
                }
            )
    return {
        "schema": "stage2_megascale_ddg",
        "n_domains": len(domains),
        "domains": domains,
    }


def _build_mini_stage2_bundle(tmp_path: Path) -> Path:
    rng = np.random.default_rng(12345)
    bundle_root = tmp_path / "mini_bundle"
    splits_dir = bundle_root / "splits"
    bench_dir = bundle_root / "benchmarks"
    splits_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)

    def _recs(prefix: str, n: int) -> list[AnnotatedSequence]:
        return [
            AnnotatedSequence(
                seq_id=f"{prefix}_{i}",
                sequence=_rand_seq(rng, 24 + (i % 5) * 4),
                cath_h="1.10.10.10",
                cath_t="1.10.10",
            )
            for i in range(n)
        ]

    arm_id = "M64"
    save_packed_split_npz(splits_dir / f"train_{arm_id}.npz", _recs("tr", 16))
    save_packed_split_npz(splits_dir / f"eval_train_sub_{arm_id}.npz", _recs("trs", 8))
    save_packed_split_npz(splits_dir / f"eval_unseen_homolog_{arm_id}.npz", _recs("hom", 8))
    save_packed_split_npz(splits_dir / "eval_unseen_fold.npz", _recs("unf", 10))

    ms = _make_synthetic_megascale(rng)
    atomic_write_gzip_json(bench_dir / "megascale_ddg.json.gz", ms)

    ct_domains = _make_synthetic_contact_domains(rng, n_domains=5)
    npz_arrays: dict[str, np.ndarray] = {}
    meta_entries: list[dict[str, Any]] = []
    for dom in ct_domains:
        did = dom["domain_id"]
        npz_arrays[f"cb_{did}"] = dom["cb_coords"]
        npz_arrays[f"mask_{did}"] = dom["cb_mask"]
        meta_entries.append({k: v for k, v in dom.items() if k not in ("cb_coords", "cb_mask")})
    np.savez_compressed(bench_dir / "unseen_fold_contacts.npz", **npz_arrays)
    atomic_write_json_with_sha256(
        bench_dir / "unseen_fold_contacts_meta.json",
        {
            "schema": "stage2_unseen_fold_contacts",
            "n_domains": len(meta_entries),
            "domains": meta_entries,
        },
    )

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "arms": {arm_id: {"n_train_sequences": 16}},
    }
    atomic_write_json_with_sha256(bundle_root / "bundle_manifest.json", manifest)
    return bundle_root


# --------------------------------------------------------------------------- #
# 1. Extractor Invariants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("objective", ["ar", "mlm_15", "oa_uniform"])
def test_extractor_variable_length_zero_pad_leakage(objective: str) -> None:
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    model = VariableLengthProteinTransformer("0.5m", objective, max_seq_len=64)
    model.eval()
    seqs = [_rand_seq(rng, 17), _rand_seq(rng, 31), _rand_seq(rng, 23)]

    batched = extract_layerwise_hidden_states(
        model, seqs, objective=objective, batch_size=3, precision="fp32_strict", pad_to_multiple=8
    )
    assert len(batched) == len(seqs)
    n_expected_taps = SCALE_SPECS["0.5m"].n_layers + 2  # emb + 3 blocks + norm_f

    for seq_i, seq in enumerate(seqs):
        single = extract_layerwise_hidden_states(
            model, [seq], objective=objective, batch_size=1, precision="fp32_strict", pad_to_multiple=1
        )[0]
        assert len(batched[seq_i]) == n_expected_taps
        for tap in range(n_expected_taps):
            assert batched[seq_i][tap].shape == (len(seq), SCALE_SPECS["0.5m"].d_model)
            np.testing.assert_allclose(batched[seq_i][tap], single[tap], atol=1e-5, rtol=1e-5)


def test_extractor_ar_bos_offset_causality_and_logits_match() -> None:
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    model = VariableLengthProteinTransformer("0.5m", "ar", max_seq_len=64)
    model.eval()

    wt_seq = _rand_seq(rng, 25)
    mut_pos = 11
    alt_aa = AA20[(AA20.index(wt_seq[mut_pos]) + 3) % 20]
    mut_seq = wt_seq[:mut_pos] + alt_aa + wt_seq[mut_pos + 1 :]

    tok_t, len_t = collate_variable_length_sequences([wt_seq, mut_seq], pad_to_multiple=1)
    layers, logits = forward_layerwise_hidden_states(
        model, tok_t, len_t, objective="ar", precision="fp32_strict", return_logits=True
    )
    assert logits is not None

    # Layer 0 must equal tok_emb(a_i) directly (accounting for <BOS> offset!)
    expected_emb_wt = model.tok_emb(torch.from_numpy(encode_aa_sequence(wt_seq)).long()).detach().numpy()
    np.testing.assert_allclose(layers[0][0].numpy(), expected_emb_wt, atol=1e-6)

    # Strict causality: positions j < mut_pos are identical across ALL layers; j >= mut_pos differ
    for tap_idx, h in enumerate(layers):
        h_wt = h[0].numpy()
        h_mut = h[1].numpy()
        np.testing.assert_allclose(h_wt[:mut_pos], h_mut[:mut_pos], atol=1e-6)
        assert float(np.linalg.norm(h_wt[mut_pos] - h_mut[mut_pos])) > 1e-4
        if tap_idx >= 1:
            # Downstream positions shift after attention layers
            assert float(np.linalg.norm(h_wt[mut_pos + 1 :] - h_mut[mut_pos + 1 :])) > 1e-4

    # Returned logits must match supply_demand prepare_ar_batch + model(...) bit-for-bit
    prep = prepare_ar_batch(tok_t, len_t)
    with torch.no_grad():
        ref_logits = model(prep.input_ids, prep.attention_mask, right_padded=True)
    np.testing.assert_allclose(logits.numpy(), ref_logits.numpy(), atol=1e-5, rtol=1e-5)


def test_checkpoint_loader_aliases_and_compiled_prefix(tmp_path: Path) -> None:
    assert canonical_objective("MDM") == "oa_uniform"
    assert canonical_objective("mdm") == "oa_uniform"
    assert paradigm_display_name("oa_uniform") == "MDM"
    assert paradigm_display_name("mlm_15") == "MLM-15%"
    assert paradigm_display_name("ar") == "AR"

    model = VariableLengthProteinTransformer("0.5m", "oa_uniform", max_seq_len=64)
    compiled_sd = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
    run_spec = Stage2RunSpec(
        run_id="stage2_M64_oa_uniform_0p5m_e10",
        arm_id="M64",
        objective="oa_uniform",
        scale_id="0.5m",
        max_epochs=10,
    )
    ckpt_path = tmp_path / "stage2_M64_oa_uniform_0p5m_e10_e0005.pt"
    atomic_torch_save(
        ckpt_path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "run_spec": run_spec.to_dict(),
            "epoch": 5,
            "global_step": 50,
            "tokens_seen": 10000,
            "model_state_dict": compiled_sd,
        },
    )
    loaded = load_stage2_checkpoint(ckpt_path, device="cpu")
    assert loaded.objective == "oa_uniform"
    assert loaded.paradigm_display == "MDM"
    assert loaded.epoch == 5
    assert loaded.layer_names == ("emb", "block_1", "block_2", "block_3", "norm_f")


# --------------------------------------------------------------------------- #
# 2. Closed-Form SVD / Sufficient-Stats Ridge Equivalence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_samples,n_features", [(120, 16), (30, 48)])
def test_closed_form_ridge_svd_and_stats_exact_match(n_samples: int, n_features: int) -> None:
    rng = np.random.default_rng(101 + n_samples)
    x = rng.normal(loc=2.5, scale=1.7, size=(n_samples, n_features))
    x[:, 3] = 5.0  # Constant zero-variance column to verify safe handling
    w_true = rng.normal(size=n_features)
    w_true[3] = 0.0
    y = x @ w_true + 1.2 + rng.normal(scale=0.3, size=n_samples)
    groups = np.arange(n_samples) % 5

    lam = 10.0
    res_svd = fit_ridge_svd_cv(
        x, y, groups, lambdas=[lam], n_cv_folds=5, return_oof_preds=True
    )

    # Reference explicit normal equation on z-standardized X
    mu_x = x.mean(axis=0)
    mu_y = float(y.mean())
    xc = x - mu_x
    std_x = np.sqrt(np.mean(xc * xc, axis=0))
    active = std_x > 1e-8
    inv_std = np.where(active, 1.0 / np.where(active, std_x, 1.0), 0.0)
    x_std = xc * inv_std
    w_std_ref = np.linalg.solve(x_std.T @ x_std + lam * np.eye(n_features), x_std.T @ (y - mu_y))
    w_ref = w_std_ref * inv_std
    b_ref = mu_y - float(np.dot(w_ref, mu_x))

    np.testing.assert_allclose(res_svd["weights"], w_ref, atol=1e-9, rtol=1e-9)
    assert math.isclose(res_svd["intercept"], b_ref, rel_tol=1e-9, abs_tol=1e-9)
    assert res_svd["weights"][3] == 0.0

    # Also verify FoldSufficientStats exact equivalence for multi-lambda CV
    folds = [FoldSufficientStats.empty(n_features) for _ in range(5)]
    shift = x.mean(axis=0)
    for f in range(5):
        m = groups == f
        folds[f].add_batch(x[m] - shift, y[m])
    res_stats = fit_ridge_from_stats_cv(folds, lambdas=DEFAULT_RIDGE_LAMBDAS)
    res_full = fit_ridge_svd_cv(x, y, groups, lambdas=DEFAULT_RIDGE_LAMBDAS, n_cv_folds=5)
    assert res_stats["best_lambda"] == res_full["best_lambda"]
    w_stats = res_stats["weights"]
    b_stats = float(res_stats["intercept"] - np.dot(w_stats, shift))
    np.testing.assert_allclose(w_stats, res_full["weights"], atol=1e-8, rtol=1e-8)
    assert math.isclose(b_stats, res_full["intercept"], rel_tol=1e-8, abs_tol=1e-8)


# --------------------------------------------------------------------------- #
# 3. Eq. (3b) Temperature Calibration Invariance & Native Agreement
# --------------------------------------------------------------------------- #


def test_weighted_temperature_calibration_scale_invariance() -> None:
    rng = np.random.default_rng(77)
    logits = torch.from_numpy(rng.normal(size=(500, 20)).astype(np.float32))
    targets = torch.from_numpy(rng.integers(0, 20, size=(500,), dtype=np.int64))
    weights = torch.from_numpy(rng.uniform(0.5, 2.0, size=(500,)).astype(np.float32))

    cal_1x = compute_weighted_temperature_calibrated_bits(logits, targets, weights)
    cal_5x = compute_weighted_temperature_calibrated_bits(logits * 5.0, targets, weights)

    assert cal_5x["H_raw"] > cal_1x["H_raw"]
    assert cal_5x["Delta_cal"] > 0.5
    assert math.isclose(cal_5x["H_cal"], cal_1x["H_cal"], rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(cal_5x["T_opt"] / cal_1x["T_opt"], 5.0, rel_tol=1e-2)


@pytest.mark.parametrize("objective", ["ar", "mlm_15", "oa_uniform"])
def test_unseen_fold_temperature_calibration_matches_native_bits(objective: str) -> None:
    torch.manual_seed(11)
    rng = np.random.default_rng(11)
    model = VariableLengthProteinTransformer("0.5m", objective, max_seq_len=64)
    store = PackedSequenceStore.from_sequences([_rand_seq(rng, 22 + i * 2) for i in range(8)])

    native = evaluate_split_native(
        model, store, objective=objective, batch_size=4, precision="fp32_strict", n_draws=4, eval_seed=999
    )
    cal = evaluate_unseen_fold_temperature_calibration(
        model, store, objective=objective, batch_size=4, precision="fp32_strict", n_draws=4, eval_seed=999
    )
    assert math.isclose(cal["H_unseen_fold"], native["bits_per_res"], rel_tol=1e-3, abs_tol=2e-3)
    assert cal["H_cal"] <= cal["H_unseen_fold"] + 1e-6
    assert cal["Delta_cal"] >= 0.0


# --------------------------------------------------------------------------- #
# 4. Probes 1, 2, 3 Unit Tests
# --------------------------------------------------------------------------- #


def test_burial_targets_exact_geometry() -> None:
    # 10 residues along a line with 1 A spacing except index 0 and 8 at distance 2 A (< 8 A, sep=8>=6)
    coords = np.zeros((10, 3), dtype=np.float64)
    for i in range(10):
        coords[i] = [i * 10.0, 0.0, 0.0]
    coords[8] = [2.0, 0.0, 0.0]  # Only (0, 8) has |i-j| >= 6 and dist < 8 A
    mask = np.ones(10, dtype=np.bool_)
    counts = compute_residue_burial_targets(coords, mask, min_sep=6, contact_cutoff=8.0)
    assert counts[0] == 1.0
    assert counts[8] == 1.0
    assert float(counts.sum()) == 2.0


@pytest.mark.parametrize("objective", ["ar", "mlm_15", "oa_uniform"])
def test_all_three_probes_run_cleanly(objective: str) -> None:
    torch.manual_seed(19)
    rng = np.random.default_rng(19)
    model = VariableLengthProteinTransformer("0.5m", objective, max_seq_len=64)
    ms = _make_synthetic_megascale(rng)
    cts = _make_synthetic_contact_domains(rng, n_domains=5)

    p1 = evaluate_megascale_layerwise_probe(
        model,
        ms,
        objective=objective,
        optimal_temperature=1.8,
        max_variants_per_wt=15,
        batch_size=32,
        precision="fp32_strict",
        n_cv_folds=3,
    )
    assert len(p1["layers"]) == SCALE_SPECS["0.5m"].n_layers + 2
    assert "unseen_fold_natural" in p1["layers"][0]["by_stratum"]
    assert "de_novo_design" in p1["layers"][0]["by_stratum"]
    assert "raw" in p1["zeroshot"] and "temperature_calibrated" in p1["zeroshot"]

    p2 = evaluate_contact_layerwise_probe(
        model,
        cts,
        objective=objective,
        batch_size=4,
        zeroshot_batch_size=64,
        precision="fp32_strict",
        n_folds=5,
        compute_zeroshot=True,
    )
    assert len(p2["layers"]) == SCALE_SPECS["0.5m"].n_layers + 2
    assert p2["zeroshot_categorical_jacobian"] is not None
    assert math.isfinite(p2["layers"][1]["sep_ge_6"]["auroc"])
    assert math.isfinite(p2["layers"][1]["long_range_sep_ge_24"]["auroc"])

    p3 = evaluate_burial_layerwise_probe(
        model,
        cts,
        objective=objective,
        batch_size=4,
        precision="fp32_strict",
        n_folds=5,
    )
    assert len(p3["layers"]) == SCALE_SPECS["0.5m"].n_layers + 2
    assert math.isfinite(p3["layers"][1]["pooled_r2"])
    assert math.isfinite(p3["layers"][1]["pooled_spearman"])


# --------------------------------------------------------------------------- #
# 5. H1 vs H2 Aggregation & End-to-End CLI Tests
# --------------------------------------------------------------------------- #


def test_h1_vs_h2_degradation_diagnostics() -> None:
    layer_names = ["emb", "block_1", "block_2", "block_3", "norm_f"]

    def _mock_doc(epoch: int, scores: list[float], zs: float) -> dict[str, Any]:
        return {
            "schema": PROBE_CHECKPOINT_SCHEMA,
            "checkpoint_id": f"ckpt_e{epoch:04d}",
            "run_id": "mock_run",
            "arm_id": "M1024",
            "objective": "ar",
            "paradigm_display": "AR",
            "scale_id": "8m",
            "E": epoch,
            "epoch": epoch,
            "layer_names": layer_names,
            "H_unseen_fold": 4.0,
            "H_cal": 3.8,
            "T_opt": 1.5,
            "Delta_cal": 0.2,
            "probe_megascale_ddg": {
                "zeroshot": {
                    "raw": {"unseen_fold_natural": {"spearman_mean": zs}},
                    "temperature_calibrated": {"unseen_fold_natural": {"spearman_mean": zs + 0.02}},
                },
                "layers": [
                    {
                        "layer_idx": i,
                        "layer_name": layer_names[i],
                        "by_stratum": {"unseen_fold_natural": {"spearman_mean": sc}},
                    }
                    for i, sc in enumerate(scores)
                ],
            },
        }

    # H1 case: zero-shot collapses from 0.30 -> 0.02 while middle block_2 retains 0.32 -> 0.30
    h1_summary = aggregate_probe_results(
        [
            _mock_doc(10, [0.10, 0.28, 0.32, 0.31, 0.30], zs=0.30),
            _mock_doc(300, [0.10, 0.27, 0.30, 0.18, 0.08], zs=0.02),
        ]
    )
    h1_prof = h1_summary["trajectories"]["M1024__ar__8m"]["layer_degradation_profile"][
        "megascale_unseen_fold_natural_spearman"
    ]
    assert h1_prof["E_peak"] == 10
    assert h1_prof["E_final"] == 300
    assert h1_prof["hypothesis_verdict"] == "H1_output_head_pollution"

    # H2 case: both zero-shot AND intermediate layers collapse
    h2_summary = aggregate_probe_results(
        [
            _mock_doc(10, [0.10, 0.28, 0.32, 0.31, 0.30], zs=0.30),
            _mock_doc(300, [0.05, 0.10, 0.12, 0.08, 0.04], zs=0.02),
        ]
    )
    h2_prof = h2_summary["trajectories"]["M1024__ar__8m"]["layer_degradation_profile"][
        "megascale_unseen_fold_natural_spearman"
    ]
    assert h2_prof["hypothesis_verdict"] == "H2_deep_representation_overwriting"

    ret = compute_layer_retention_metrics([0.1, 0.4, 0.2], ["emb", "block_1", "norm_f"], zeroshot_score=0.1)
    assert ret["best_layer_idx"] == 1
    assert math.isclose(ret["head_retention_ratio"], 0.25)
    assert math.isclose(ret["best_layer_vs_zeroshot_gain"], 0.30)


def test_end_to_end_trajectory_trainer_and_batch_runner(tmp_path: Path) -> None:
    bundle_dir = _build_mini_stage2_bundle(tmp_path)
    traj_out = tmp_path / "traj_out"
    probe_out = tmp_path / "probe_out"

    rc_train = train_probe_trajectory.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--out-dir",
            str(traj_out),
            "--arm-id",
            "M64",
            "--objective",
            "mdm",
            "--scale-id",
            "0.5m",
            "--max-epochs",
            "2",
            "--snapshot-epochs",
            "0,1,2",
            "--precision",
            "fp32_strict",
            "--batch-size",
            "8",
            "--device",
            "cpu",
        ]
    )
    assert rc_train == 0
    ckpt_dir = traj_out / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.pt"))
    assert len(ckpts) == 3  # E=0, 1, 2

    # Run batch probe runner sharded across 2 workers then aggregate
    for rank in (0, 1):
        rc_p = run_layerwise_probes.main(
            [
                "--checkpoints-dir",
                str(ckpt_dir),
                "--bundle-dir",
                str(bundle_dir),
                "--out-dir",
                str(probe_out),
                "--device",
                "cpu",
                "--precision",
                "fp32_strict",
                "--worker-rank",
                str(rank),
                "--num-workers",
                "2",
                "--cal-n-draws",
                "2",
                "--n-cv-folds",
                "3",
                "--max-variants-per-wt",
                "12",
                "--skip-zeroshot-contacts",
            ]
        )
        assert rc_p == 0

    summary_path = probe_out / "probe_summary.json"
    assert summary_path.is_file()
    assert summary_path.with_name(summary_path.name + ".sha256").is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema"] == PROBE_SUMMARY_SCHEMA
    assert summary["n_checkpoints"] == 3
    assert_no_forbidden_infra_strings(summary, label="probe_summary")

    # Verify per-checkpoint files and resume idempotency
    per_ckpts = sorted((probe_out / "checkpoints").glob("*.probe.json"))
    assert len(per_ckpts) == 3
    for jp in per_ckpts:
        doc = json.loads(jp.read_text(encoding="utf-8"))
        assert doc["paradigm_display"] == "MDM"
        assert doc["objective"] == "oa_uniform"
        assert "Delta_cal" in doc and "H_cal" in doc
        assert_no_forbidden_infra_strings(doc, label=jp.name)

    # Verify evaluate_checkpoint_probes direct call works with zeroshot contacts enabled too
    direct_doc = evaluate_checkpoint_probes(
        ckpts[-1],
        bundle_dir,
        device="cpu",
        precision="fp32_strict",
        max_variants_per_wt=12,
        cal_n_draws=2,
        n_cv_folds=3,
        compute_zeroshot_contacts=True,
    )
    assert direct_doc["probe_contacts"]["zeroshot_categorical_jacobian"] is not None
    assert_no_forbidden_infra_strings(direct_doc, label="direct_doc")
