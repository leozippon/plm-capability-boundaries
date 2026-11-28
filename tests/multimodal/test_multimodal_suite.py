"""Tests for the ESM-3-style controlled multimodal suite (pure_seq, struct_cond, joint_to_seq)."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.multimodal import (
    MULTIMODAL_BUNDLE_SCHEMA,
    MULTIMODAL_MODES,
    MultiTrackProteinTransformer,
    Multimodal3DBundle,
    MultimodalDomainRecord,
    MultimodalRunSpec,
    build_fold_disjoint_multimodal_splits,
    compute_backbone_geom_features,
    count_multimodal_parameters,
    discretize_local_3di_fallback,
    encode_3di_sequence,
    execute_multimodal_run,
    save_packed_multimodal_npz,
    summarize_multimodal_campaign,
)
from src.capability.supply_demand.corpus import (
    atomic_write_gzip_json,
    atomic_write_json_with_sha256,
)
from src.capability.supply_demand.spec import AA20, SCALE_SPECS


def _make_hairpin_coords(length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    half = length // 2
    cb = np.zeros((length, 3), dtype=np.float32)
    for i in range(length):
        if i < half:
            cb[i] = [float(i) * 3.8, 0.0, float(rng.normal(0.0, 0.3))]
        else:
            cb[i] = [float(length - 1 - i) * 3.8, 5.2, float(rng.normal(0.0, 0.3))]
    return cb, np.ones(length, dtype=np.bool_)


def _make_tiny_multimodal_bundle(bundle_dir: Path, *, seed: int = 19) -> Multimodal3DBundle:
    rng = np.random.default_rng(seed)
    records: list[MultimodalDomainRecord] = []
    # 6 distinct topologies, 6 domains per topology -> strict fold-disjoint split
    for t_i in range(6):
        topo = f"1.10.{100 + t_i}"
        sf = f"{topo}.10"
        for d_i in range(6):
            L = int(rng.integers(28, 40))
            seq = "".join(AA20[int(c)] for c in rng.integers(0, 20, size=L))
            cb, cbm = _make_hairpin_coords(L, rng)
            tdi = discretize_local_3di_fallback(cb, cbm)
            geom = compute_backbone_geom_features(cb, cbm)
            records.append(
                MultimodalDomainRecord(
                    domain_id=f"dom_t{t_i}_d{d_i}",
                    sequence=seq,
                    three_di=tdi,
                    cb_coords=cb,
                    cb_mask=cbm,
                    geom_feats=geom,
                    cath_h=sf,
                    cath_t=topo,
                )
            )

    splits = build_fold_disjoint_multimodal_splits(
        records, unseen_fold_frac=0.20, unseen_homolog_frac=0.20, seed=seed
    )
    splits_dir = bundle_dir / "splits"
    for name in ("train", "eval_train_sub", "eval_unseen_homolog", "eval_unseen_fold"):
        save_packed_multimodal_npz(splits_dir / f"{name}.npz", splits[name])

    bench_dir = bundle_dir / "benchmarks"
    # Synthetic MegaScale with attached WT 3D structure (three_di, cb_coords, cb_mask)
    ms_domains = []
    for i, stratum in enumerate(("de_novo_design", "unseen_fold_natural")):
        wt = "".join(AA20[int(c)] for c in rng.integers(0, 20, size=30))
        cb, cbm = _make_hairpin_coords(30, rng)
        tdi = discretize_local_3di_fallback(cb, cbm)
        variants = []
        for pos in range(0, 16, 2):
            w = wt[pos]
            m = AA20[(AA20.index(w) + 1) % 20]
            variants.append([pos, w, m, float(rng.normal())])
        ms_domains.append(
            {
                "wt_name": f"MS_{i}",
                "wildtype": wt,
                "length": 30,
                "stratum": stratum,
                "three_di": tdi,
                "cb_coords": cb.tolist(),
                "cb_mask": cbm.tolist(),
                "variants": variants,
            }
        )
    atomic_write_gzip_json(
        bench_dir / "megascale_ddg.json.gz",
        {"schema": "stage2_megascale_ddg", "domains": ms_domains},
    )

    # Synthetic ProteinGym
    wt_pg = "".join(AA20[int(c)] for c in rng.integers(0, 20, size=30))
    pg_vars = [[f"{wt_pg[p]}{p + 1}{AA20[(AA20.index(wt_pg[p]) + 2) % 20]}", float(rng.normal())] for p in range(8)]
    atomic_write_gzip_json(
        bench_dir / "proteingym_substitutions.json.gz",
        {
            "schema": "stage2_proteingym_substitutions",
            "assays": [
                {
                    "dms_id": "PG_0",
                    "cluster_id": "C0",
                    "wildtype": wt_pg,
                    "length": 30,
                    "fold_stratum": "unseen_fold",
                    "in_milestone_subset": True,
                    "variants": pg_vars,
                }
            ],
        },
    )

    # Unseen-fold 3D contacts
    ct_rec = splits["eval_unseen_fold"][0]
    np.savez_compressed(
        bench_dir / "unseen_fold_contacts.npz",
        **{
            f"cb_{ct_rec.domain_id}": ct_rec.cb_coords,
            f"mask_{ct_rec.domain_id}": ct_rec.cb_mask,
            f"three_di_{ct_rec.domain_id}": encode_3di_sequence(ct_rec.three_di),
        },
    )
    atomic_write_json_with_sha256(
        bench_dir / "unseen_fold_contacts_meta.json",
        {
            "schema": "multimodal_unseen_fold_contacts_v1",
            "n_domains": 1,
            "domains": [
                {
                    "domain_id": ct_rec.domain_id,
                    "sequence": ct_rec.sequence,
                    "length": ct_rec.length,
                    "cath_superfamily": ct_rec.cath_h,
                    "cath_topology": ct_rec.cath_t,
                }
            ],
        },
    )
    atomic_write_json_with_sha256(
        bundle_dir / "bundle_manifest.json",
        {"schema": MULTIMODAL_BUNDLE_SCHEMA, "fold_disjointness_verified": True},
    )
    return Multimodal3DBundle(bundle_dir)


def test_multimodal_trunk_param_parity_and_rigid_motion_invariance() -> None:
    """All 3 modes share bit-identical transformer trunk n_params_non_embedding across scales 0.5m..150m."""
    for scale_id in ("0.5m", "2m", "8m", "35m", "150m"):
        expected = SCALE_SPECS[scale_id].exact_non_embedding_params
        for mode in MULTIMODAL_MODES:
            with torch.device("meta"):
                model = MultiTrackProteinTransformer(scale_id, mode=mode, objective="mdm", max_seq_len=64)
                counts = count_multimodal_parameters(model)
                assert counts["n_params_non_embedding"] == expected
                assert counts["n_params_trunk"] == expected

    # Rotation + translation invariance of backbone geometric features
    rng = np.random.default_rng(11)
    cb, cbm = _make_hairpin_coords(32, rng)
    f1 = compute_backbone_geom_features(cb, cbm)
    # Random proper orthogonal matrix R (det=+1) + translation t
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    cb_rot = (cb @ q.T + np.array([15.0, -23.0, 42.0], dtype=np.float32)).astype(np.float32)
    f2 = compute_backbone_geom_features(cb_rot, cbm)
    assert np.allclose(f1, f2, atol=1e-4)


@pytest.mark.parametrize("mode", ["struct_cond", "joint_to_seq"])
def test_multimodal_strict_zero_leakage_when_structure_absent(mode: str) -> None:
    """When struct_present=False (Z = empty), adversarial garbage in struct/geom/coords MUST NOT affect AA logits."""
    torch.manual_seed(55)
    model = MultiTrackProteinTransformer("0.5m", mode=mode, objective="mdm", max_seq_len=64).eval()
    aa = torch.randint(0, 20, (2, 24))
    valid = torch.ones((2, 24), dtype=torch.bool)
    absent = torch.zeros((2,), dtype=torch.bool)

    logits_none = model(aa, valid, struct_present=absent)
    logits_garbage = model(
        aa,
        valid,
        struct_tokens=torch.randint(0, 20, (2, 24)),
        geom_feats=torch.randn(2, 24, 18) * 100.0,
        cb_coords=torch.randn(2, 24, 3) * 50.0,
        cb_mask=torch.ones((2, 24), dtype=torch.bool),
        struct_present=absent,
    )
    assert torch.allclose(logits_none, logits_garbage, atol=1e-6)


def test_multimodal_all_three_modes_e2e_analysis_and_spot_worker(tmp_path: Path) -> None:
    """End-to-end training & evaluation across pure_seq, struct_cond, and joint_to_seq + campaign analysis."""
    bundle_dir = tmp_path / "mm_bundle"
    bundle = _make_tiny_multimodal_bundle(bundle_dir)
    out_dir = tmp_path / "mm_runs"

    for mode in MULTIMODAL_MODES:
        spec = MultimodalRunSpec(
            run_id=f"mm3d_{mode}_mdm_0p5m_e2",
            mode=mode,
            objective="mdm",
            scale_id="0.5m",
            max_epochs=2,
            eval_epochs=(1, 2),
            batch_size=8,
            precision="fp32_strict",
            optimizer="muon" if mode == "joint_to_seq" else "adamw",
            eval_downstream_at_milestones=True,
            eval_contacts_at_milestones=True,
            eval_temperature_calibration=True,
        )
        res = execute_multimodal_run(
            spec,
            bundle,
            out_dir,
            device="cpu",
            n_eval_draws=4,
            loo_probes=2,
            install_signal_handlers=False,
        )
        assert res["status"] == "complete"
        assert res["paradigm_display"] == "MDM"
        final = res["final_eval"]
        tt = final["three_tier_bits"]
        assert tt["native"]["conservation_error"] < 1e-12
        assert "H_unseen_fold_cal" in tt["native"] and "Delta_cal" in tt["native"]
        assert final["megascale"]["n_domains"] == 2
        if mode == "struct_cond":
            assert tt["b_ext_diagnostics"]["B_ext_active_at_eval"] is True
            assert math.isfinite(tt["b_ext_diagnostics"]["b_ext_vs_shuffled_bits_per_res"])
            assert final["megascale"]["eval_condition"] == "Z_WT_conditioned"
            assert final["megascale"]["n_domains_with_wt_structure"] == 2
        elif mode == "joint_to_seq":
            assert tt["b_ext_diagnostics"]["B_ext_active_at_eval"] is False
            assert final["megascale"]["eval_condition"] == "Z_empty"
            s2s = final["seq_to_struct_readout"]
            assert s2s["available"] is True
            assert s2s["eval_condition"] == "Z_empty_pure_sequence"
            assert 0.0 <= s2s["three_di_accuracy"] <= 1.0
            assert 0.0 <= s2s["contact_head_long_range_sep_ge_24"]["auroc"] <= 1.0
            assert final["proteingym_milestone"]["n_assays"] == 1
            assert final["contacts"]["sep_ge_6"]["n_domains"] == 1

    # Unified-notation multimodal campaign analysis
    summary = summarize_multimodal_campaign(out_dir / "runs")
    assert summary["n_runs_analyzed"] == 3
    assert summary["n_runs_complete"] == 3
    assert len(summary["paired_vs_pure_seq_contrasts"]) == 2
    for row in summary["runs"]:
        assert row["paradigm_display"] == "MDM"
        assert "E_gen" in row and "E_mem" in row and "W_window" in row and "Delta_cal" in row

    # Spot worker CLI smoke test on 1 AR run
    worker_out = tmp_path / "worker_out"
    worker_cli = REPO_ROOT / "scripts/capability/multimodal/run_multimodal_spot_worker.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(worker_cli),
            "--bundle-dir",
            str(bundle_dir),
            "--out-dir",
            str(worker_out),
            "--modes",
            "pure_seq",
            "--objectives",
            "ar",
            "--scales",
            "0.5m",
            "--max-epochs",
            "1",
            "--batch-size",
            "8",
            "--precision",
            "fp32_strict",
            "--device",
            "cpu",
            "--no-downstream-milestones",
            "--no-contacts-milestones",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[mm3d-worker] Completed" in proc.stdout
    assert (worker_out / "runs" / "mm3d_pure_seq_ar_0p5m_e1.json").is_file()
