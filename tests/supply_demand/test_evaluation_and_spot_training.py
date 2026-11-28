"""Tests for Eq. (4) 3-tier bit readout, MegaScale/ProteinGym/Contacts evaluators, and Spot SIGTERM resume."""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from src.capability.supply_demand.corpus import (
    BUNDLE_SCHEMA,
    AnnotatedSequence,
    Stage2Bundle,
    atomic_write_gzip_json,
    atomic_write_json_with_sha256,
    save_packed_split_npz,
)
from src.capability.supply_demand.evaluation import (
    compute_categorical_jacobian_contact_map,
    compute_contact_metrics_for_domain,
    evaluate_megascale_ddg,
    evaluate_proteingym_substitutions,
    evaluate_three_tier_bits,
    evaluate_unseen_fold_contacts,
)
from src.capability.supply_demand.model import VariableLengthProteinTransformer
from src.capability.supply_demand.objectives import PackedSequenceStore
from src.capability.supply_demand.spec import (
    AA20,
    LOG2_20,
    OBJECTIVES,
    Stage2RunSpec,
)
from src.capability.supply_demand.training import (
    PREEMPT_EXIT_CODE,
    SharedLeaseManager,
    compute_lr,
    execute_stage2_run,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_tiny_bundle(bundle_dir: Path, *, n_train: int = 32, seed: int = 7) -> Stage2Bundle:
    rng = np.random.default_rng(seed)

    def _rand_recs(prefix: str, n: int, topo: str, sf: str) -> list[AnnotatedSequence]:
        out = []
        for i in range(n):
            L = int(rng.integers(24, 42))
            s = "".join(AA20[int(c)] for c in rng.integers(0, 20, size=L))
            out.append(AnnotatedSequence(f"{prefix}_{i}", s, sf, topo))
        return out

    train_recs = _rand_recs("tr", n_train, "1.10.1", "1.10.1.1")
    hom_recs = _rand_recs("hm", 12, "1.10.1", "1.10.1.1")
    unf_recs = _rand_recs("uf", 12, "2.60.40", "2.60.40.10")
    glob_recs = _rand_recs("gl", 12, "1.10.2", "1.10.2.1")

    splits_dir = bundle_dir / "splits"
    save_packed_split_npz(splits_dir / "train_M4.npz", train_recs)
    save_packed_split_npz(splits_dir / "eval_train_sub_M4.npz", train_recs[:12])
    save_packed_split_npz(splits_dir / "eval_unseen_homolog_M4.npz", hom_recs)
    save_packed_split_npz(splits_dir / "eval_unseen_fold.npz", unf_recs)
    save_packed_split_npz(splits_dir / "eval_global_natural.npz", glob_recs)

    # Synthetic MegaScale benchmark (3 strata)
    bench_dir = bundle_dir / "benchmarks"
    ms_domains = []
    for d_i, stratum in enumerate(("de_novo_design", "unseen_fold_natural", "seen_pool_natural")):
        wt = "".join(AA20[int(c)] for c in rng.integers(0, 20, size=28))
        variants = []
        for pos in range(0, 20, 2):
            w_aa = wt[pos]
            for alt_off in (1, 2):
                m_aa = AA20[(AA20.index(w_aa) + alt_off) % 20]
                variants.append([pos, w_aa, m_aa, float(rng.normal())])
        ms_domains.append(
            {
                "wt_name": f"MS_{d_i}",
                "wildtype": wt,
                "length": len(wt),
                "kind": "design" if stratum == "de_novo_design" else "natural",
                "stratum": stratum,
                "cath_topologies": ["2.60.40"] if stratum == "unseen_fold_natural" else [],
                "n_uniref50_hits": 0 if stratum == "de_novo_design" else 5,
                "variants": variants,
            }
        )
    atomic_write_gzip_json(
        bench_dir / "megascale_ddg.json.gz",
        {"schema": "stage2_megascale_ddg", "domains": ms_domains},
    )

    # Synthetic ProteinGym benchmark (single + double mutants, unseen_fold & seen_fold)
    pg_assays = []
    for a_i, (st, in_ms) in enumerate((("unseen_fold", True), ("seen_fold", True), ("seen_fold", False))):
        wt = "".join(AA20[int(c)] for c in rng.integers(0, 20, size=32))
        variants = []
        for pos in range(10):
            w_aa = wt[pos]
            m_aa = AA20[(AA20.index(w_aa) + 3) % 20]
            variants.append([f"{w_aa}{pos + 1}{m_aa}", float(rng.normal())])
        # Add a double mutant
        m2 = f"{wt[0]}1{AA20[(AA20.index(wt[0]) + 1) % 20]}:{wt[5]}6{AA20[(AA20.index(wt[5]) + 2) % 20]}"
        variants.append([m2, float(rng.normal())])
        pg_assays.append(
            {
                "dms_id": f"PG_{a_i}",
                "uniprot_id": f"PROT_{a_i % 2}",
                "cluster_id": f"PROT_{a_i % 2}",
                "coarse_selection_type": "Stability",
                "wildtype": wt,
                "length": len(wt),
                "fold_stratum": st,
                "in_milestone_subset": in_ms,
                "variants": variants,
            }
        )
    atomic_write_gzip_json(
        bench_dir / "proteingym_substitutions.json.gz",
        {"schema": "stage2_proteingym_substitutions", "assays": pg_assays},
    )

    # Synthetic 3D contacts (hairpin-like coordinates with both sep>=6 and sep>=24 contacts!)
    L_c = 40
    seq_c = "".join(AA20[int(c)] for c in rng.integers(0, 20, size=L_c))
    cb = np.zeros((L_c, 3), dtype=np.float32)
    for i in range(L_c):
        if i < 20:
            cb[i] = [float(i) * 3.8, 0.0, 0.0]
        else:
            # Fold back antiparallel at y=5.0 A so i and (39-i) are 5.0 A apart (including sep>=24!)
            cb[i] = [float(39 - i) * 3.8, 5.0, 0.0]
    cb_mask = np.ones(L_c, dtype=np.bool_)
    np.savez_compressed(bench_dir / "unseen_fold_contacts.npz", cb_DOM0=cb, mask_DOM0=cb_mask)
    atomic_write_json_with_sha256(
        bench_dir / "unseen_fold_contacts_meta.json",
        {
            "schema": "stage2_unseen_fold_contacts",
            "n_domains": 1,
            "n_distinct_topologies": 1,
            "npz_file": "unseen_fold_contacts.npz",
            "domains": [
                {
                    "domain_id": "DOM0",
                    "sequence": seq_c,
                    "length": L_c,
                    "cath_superfamily": "2.60.40.10",
                    "cath_topology": "2.60.40",
                }
            ],
        },
    )

    atomic_write_json_with_sha256(
        bundle_dir / "bundle_manifest.json",
        {
            "schema": BUNDLE_SCHEMA,
            "arms": {"M4": {"arm_id": "M4"}},
        },
    )
    return Stage2Bundle(bundle_dir)


@pytest.mark.parametrize("objective", list(OBJECTIVES))
def test_three_tier_exact_log2_20_and_conservation(tmp_path: Path, objective: str) -> None:
    """Zero-logit model gives exact H_* == log2(20) and delta_h_* == 0; trained/init obeys Eq. (4)."""
    bundle = _make_tiny_bundle(tmp_path / "bundle")
    splits = bundle.load_arm_splits("M4")

    model = VariableLengthProteinTransformer("0.5m", objective, max_seq_len=64)
    # 1) Exact uniform baseline via zero lm_head weights
    with torch.no_grad():
        model.lm_head.weight.zero_()

    res_zero = evaluate_three_tier_bits(
        model,
        train_sub_store=splits["eval_train_sub"],
        unseen_homolog_store=splits["eval_unseen_homolog"],
        unseen_fold_store=splits["eval_unseen_fold"],
        global_natural_store=splits["eval_global_natural"],
        objective=objective,
        batch_size=8,
        precision="fp32_strict",
        n_draws=4,
        loo_probes=4,
    )
    for key in ("native", "native_seq_mean", "loo"):
        eq4 = res_zero[key]
        assert math.isclose(eq4["H_train"], LOG2_20, abs_tol=1e-5)
        assert math.isclose(eq4["H_unseen_homolog"], LOG2_20, abs_tol=1e-5)
        assert math.isclose(eq4["H_unseen_fold"], LOG2_20, abs_tol=1e-5)
        assert abs(eq4["delta_h_univ"]) < 1e-5
        assert abs(eq4["delta_h_spec"]) < 1e-5
        assert abs(eq4["delta_h_res"]) < 1e-5
        assert eq4["conservation_error"] < 1e-12

    # 2) Non-zero weights still obey exact Eq. (4) conservation
    torch.manual_seed(11)
    model2 = VariableLengthProteinTransformer("0.5m", objective, max_seq_len=64)
    res_init = evaluate_three_tier_bits(
        model2,
        train_sub_store=splits["eval_train_sub"],
        unseen_homolog_store=splits["eval_unseen_homolog"],
        unseen_fold_store=splits["eval_unseen_fold"],
        objective=objective,
        batch_size=8,
        precision="fp32_strict",
        n_draws=4,
        loo_probes=4,
    )
    for key in ("native", "native_seq_mean", "loo"):
        eq4 = res_init[key]
        assert eq4["conservation_error"] < 1e-12
        assert abs(eq4["H_unseen_fold"] - LOG2_20) < 0.15


@pytest.mark.parametrize("objective", list(OBJECTIVES))
def test_downstream_evaluators_megascale_proteingym_contacts(tmp_path: Path, objective: str) -> None:
    """Verify MegaScale, ProteinGym, and Categorical Jacobian + APC contact evaluators."""
    bundle = _make_tiny_bundle(tmp_path / "bundle")
    torch.manual_seed(22)
    model = VariableLengthProteinTransformer("0.5m", objective, max_seq_len=64)

    ms_res = evaluate_megascale_ddg(
        model, bundle.load_megascale(), objective=objective, batch_size=32, precision="fp32_strict"
    )
    assert ms_res["n_domains"] == 3
    assert set(ms_res["by_stratum"].keys()) == {
        "de_novo_design",
        "unseen_fold_natural",
        "seen_pool_natural",
    }
    assert math.isfinite(ms_res["overall_spearman_mean"])
    if objective != "ar":
        assert "wt_marginal_overall_spearman_mean" in ms_res

    pg_ms = evaluate_proteingym_substitutions(
        model,
        bundle.load_proteingym(),
        objective=objective,
        milestone_only=True,
        batch_size=32,
        precision="fp32_strict",
    )
    pg_full = evaluate_proteingym_substitutions(
        model,
        bundle.load_proteingym(),
        objective=objective,
        milestone_only=False,
        batch_size=32,
        precision="fp32_strict",
    )
    assert pg_ms["n_assays"] == 2
    assert pg_full["n_assays"] == 3
    assert math.isfinite(pg_full["cluster_mean_spearman"])
    assert "unseen_fold" in pg_full["by_stratum"]

    ct_domains = bundle.load_contacts()
    ct_res = evaluate_unseen_fold_contacts(
        model, ct_domains, objective=objective, batch_size=64, precision="fp32_strict"
    )
    assert ct_res["sep_ge_6"]["n_domains"] == 1
    assert ct_res["long_range_sep_ge_24"]["n_domains"] == 1
    for stratum_key in ("sep_ge_6", "long_range_sep_ge_24"):
        for metric in ("p_at_L", "p_at_L_over_2", "p_at_L_over_5", "auroc"):
            val = ct_res[stratum_key][metric]
            assert 0.0 <= val <= 1.0

    # Verify Categorical Jacobian map is symmetric and zero-diagonal
    cj_map = compute_categorical_jacobian_contact_map(
        model, ct_domains[0]["sequence"], objective=objective, batch_size=64, precision="fp32_strict"
    )
    assert np.allclose(cj_map, cj_map.T, atol=1e-6)
    assert np.allclose(np.diag(cj_map), 0.0, atol=1e-6)


def test_contact_metrics_perfect_signal_oracle() -> None:
    """Perfect inverse-distance score matrix must yield AUROC == 1.0 and P@L/5 == 1.0."""
    L = 40
    cb = np.zeros((L, 3), dtype=np.float32)
    for i in range(L):
        cb[i] = [float(i) * 3.8, 0.0, 0.0] if i < 20 else [float(39 - i) * 3.8, 5.0, 0.0]
    dmat = np.linalg.norm(cb[:, None, :] - cb[None, :, :], axis=-1)
    oracle = -dmat
    np.fill_diagonal(oracle, 0.0)
    m6 = compute_contact_metrics_for_domain(oracle, cb, np.ones(L, dtype=np.bool_), min_sep=6)
    m24 = compute_contact_metrics_for_domain(oracle, cb, np.ones(L, dtype=np.bool_), min_sep=24)
    assert m6 is not None and math.isclose(m6["auroc"], 1.0, abs_tol=1e-9)
    assert m24 is not None and math.isclose(m24["auroc"], 1.0, abs_tol=1e-9)
    assert math.isclose(m6["p_at_L_over_5"], 1.0, abs_tol=1e-9)


def test_wsd_and_cosine_lr_schedules() -> None:
    # WSD: linear warmup -> constant peak -> cosine cooldown
    lr_mid = compute_lr(50, total_steps=100, peak_lr=1e-3, warmup_steps=10, schedule="wsd", cooldown_fraction=0.1)
    lr_end = compute_lr(100, total_steps=100, peak_lr=1e-3, warmup_steps=10, schedule="wsd", cooldown_fraction=0.1)
    assert math.isclose(lr_mid, 1e-3, rel_tol=1e-9)
    assert math.isclose(lr_end, 5e-5, rel_tol=1e-6)

    lr_cos_mid = compute_lr(55, total_steps=100, peak_lr=1e-3, warmup_steps=10, schedule="cosine")
    assert 4e-4 < lr_cos_mid < 6e-4


def test_shared_lease_manager_exclusivity_and_stale_reclaim(tmp_path: Path) -> None:
    mgr = SharedLeaseManager(tmp_path / "leases", stale_after_sec=0.05)
    lease1 = mgr.try_claim("run_a")
    assert lease1 is not None
    # Second concurrent claim must fail while fresh
    assert mgr.try_claim("run_a") is None

    # After stale_after_sec elapses without heartbeat, another worker can reclaim
    time.sleep(0.08)
    lease2 = mgr.try_claim("run_a")
    assert lease2 is not None
    assert lease2.owner_token != lease1.owner_token

    # Old owner heartbeat MUST NOT clobber stolen lease2
    assert mgr.heartbeat(lease1) is False
    cur_doc = json.loads(lease2.path.read_text(encoding="utf-8"))
    assert cur_doc["owner_token"] == lease2.owner_token

    # Valid owner heartbeat preserves claimed_epoch_sec
    assert mgr.heartbeat(lease2) is True
    cur_doc2 = json.loads(lease2.path.read_text(encoding="utf-8"))
    assert cur_doc2["owner_token"] == lease2.owner_token
    assert math.isclose(cur_doc2["claimed_epoch_sec"], lease2.claimed_epoch_sec, rel_tol=1e-9)

    # Releasing old lease1 must NOT delete lease2
    assert mgr.release(lease1) is False
    assert lease2.path.is_file()
    assert mgr.release(lease2) is True
    assert not lease2.path.is_file()


def test_resume_from_final_step_preemption_completes_eval_and_receipt(tmp_path: Path) -> None:
    """P0 regression test: preempting on the final step of max_epochs must still finalize eval & receipt on resume."""
    bundle = _make_tiny_bundle(tmp_path / "bundle", n_train=16, seed=13)
    out_dir = tmp_path / "out"
    spec = Stage2RunSpec(
        run_id="final_step_preempt_run",
        arm_id="M4",
        objective="ar",
        scale_id="0.5m",
        max_epochs=2,
        eval_epochs=(2,),
        batch_size=8,
        precision="fp32_strict",
        eval_downstream_at_milestones=False,
        eval_contacts_at_milestones=False,
        eval_full_proteingym_at_final=False,
    )
    # Create a synthetic checkpoint representing preemption right after step 2/2 of epoch 2 BEFORE eval
    torch.manual_seed(spec.seed)
    model = VariableLengthProteinTransformer("0.5m", "ar", max_seq_len=spec.max_seq_len)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "stage2_supply_demand_ckpt",
            "run_spec": spec.to_dict(),
            "epoch": 2,
            "step_in_epoch": 2,  # == steps_per_epoch (16 / 8)
            "eval_done": False,
            "global_step": 4,
            "tokens_seen": 1000,
            "elapsed_wall_sec": 1.0,
            "epoch_loss_sum": 6.0,
            "epoch_loss_steps": 2,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None,
            "init_eval": None,
            "milestones": [],
        },
        ckpt_dir / f"{spec.run_id}_latest.pt",
    )

    res = execute_stage2_run(
        spec,
        bundle,
        out_dir,
        device="cpu",
        use_compile=False,
        resume=True,
        n_eval_draws=4,
        loo_probes=4,
        install_signal_handlers=False,
    )
    assert res["status"] == "complete"
    assert res["global_step"] == 4  # Executed 0 extra training steps!
    assert len(res["milestones"]) == 1
    assert res["milestones"][0]["epoch"] == 2
    run_json = out_dir / "runs" / f"{spec.run_id}.json"
    assert run_json.is_file()
    disk_doc = json.loads(run_json.read_text(encoding="utf-8"))
    assert disk_doc["status"] == "complete"
    assert (ckpt_dir / f"{spec.run_id}_final.pt").is_file()
    assert not (ckpt_dir / f"{spec.run_id}_latest.pt").exists()


def test_sigterm_preemption_atomic_checkpoint_and_exact_resume(tmp_path: Path) -> None:
    """Real subprocess SIGTERM test: must flush atomic checkpoint, exit 143, and resume cleanly."""
    bundle_dir = tmp_path / "bundle"
    out_dir = tmp_path / "out"
    _make_tiny_bundle(bundle_dir, n_train=48, seed=99)

    spec = Stage2RunSpec(
        run_id="spot_preempt_test_run",
        arm_id="M4",
        objective="oa_uniform",
        scale_id="0.5m",
        max_epochs=6,
        eval_epochs=(3, 6),
        batch_size=4,
        precision="fp32_strict",
        checkpoint_interval_sec=300.0,
        eval_downstream_at_milestones=False,
        eval_contacts_at_milestones=False,
        eval_full_proteingym_at_final=False,
    )
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps([spec.to_dict()]), encoding="utf-8")

    worker_script = REPO_ROOT / "scripts/capability/supply_demand/run_stage2_spot_worker.py"
    cmd = [
        sys.executable,
        str(worker_script),
        "--bundle-dir",
        str(bundle_dir),
        "--out-dir",
        str(out_dir),
        "--queue-json",
        str(queue_path),
        "--device",
        "cpu",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Wait until lease appears (worker has entered training loop), then send SIGTERM
    lease_file = out_dir / "leases" / f"{spec.run_id}.lease.json"
    t_wait = time.monotonic()
    while not lease_file.is_file() and (time.monotonic() - t_wait) < 15.0:
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    # Give worker a moment to enter step loop
    time.sleep(0.35)
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
    stdout1, stderr1 = proc.communicate(timeout=30)

    # Either preempted cleanly with 143 (expected) or finished super fast on fast CPU
    latest_ckpt = out_dir / "checkpoints" / f"{spec.run_id}_latest.pt"
    run_json = out_dir / "runs" / f"{spec.run_id}.json"
    assert run_json.is_file(), f"Missing run_json. stdout={stdout1}\nstderr={stderr1}"
    doc1 = json.loads(run_json.read_text(encoding="utf-8"))

    if proc.returncode == PREEMPT_EXIT_CODE:
        assert doc1["status"] == "preempted"
        assert latest_ckpt.is_file()
        assert not lease_file.is_file(), "Lease must be released on SIGTERM preemption"
        preempt_step = int(doc1["global_step"])
        assert preempt_step > 0

        # Relaunch worker: must resume from exact saved step and finish to complete
        res2 = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
        assert res2.returncode == 0
        doc2 = json.loads(run_json.read_text(encoding="utf-8"))
        assert doc2["status"] == "complete"
        assert doc2["epoch_done"] == 6
        assert doc2["global_step"] == 6 * (48 // 4)
        assert not latest_ckpt.is_file()
        assert (out_dir / "checkpoints" / f"{spec.run_id}_final.pt").is_file()
    else:
        assert proc.returncode == 0, f"Unexpected exit {proc.returncode}: stderr={stderr1}"
        assert doc1["status"] == "complete"


def test_end_to_end_training_all_three_objectives_cpu(tmp_path: Path) -> None:
    """End-to-end training & milestone eval smoke test across ar, mlm_15, and oa_uniform."""
    bundle = _make_tiny_bundle(tmp_path / "bundle", n_train=16, seed=5)
    for obj in OBJECTIVES:
        spec = Stage2RunSpec(
            run_id=f"e2e_{obj}",
            arm_id="M4",
            objective=obj,
            scale_id="0.5m",
            max_epochs=2,
            eval_epochs=(1, 2),
            batch_size=8,
            precision="fp32_strict",
            eval_downstream_at_milestones=True,
            eval_contacts_at_milestones=True,
            eval_full_proteingym_at_final=True,
        )
        res = execute_stage2_run(
            spec,
            bundle,
            tmp_path / "runs_out",
            device="cpu",
            use_compile=False,
            n_eval_draws=4,
            loo_probes=4,
            install_signal_handlers=False,
        )
        assert res["status"] == "complete"
        assert res["epoch_done"] == 2
        assert len(res["milestones"]) == 2
        final_m = res["final_eval"]
        assert final_m["three_tier_bits"]["native"]["conservation_error"] < 1e-12
        assert final_m["megascale"]["n_domains"] == 3
        assert final_m["proteingym_milestone"]["n_assays"] == 2
        assert final_m["proteingym_full"]["n_assays"] == 3
        assert final_m["contacts"]["sep_ge_6"]["n_domains"] == 1
