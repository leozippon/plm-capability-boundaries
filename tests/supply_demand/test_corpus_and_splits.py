"""Tests for Stage-2 fold-disjoint splitting, token-matched (M, d) arms, and bundle round-trip."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.capability.supply_demand.corpus import (
    BUNDLE_SCHEMA,
    AnnotatedSequence,
    Stage2Bundle,
    atomic_write_json_with_sha256,
    build_controlled_natural_splits,
    save_packed_split_npz,
    seq_sha256,
    verify_fold_disjointness,
)
from src.capability.supply_demand.spec import AA20


def _make_synthetic_corpus(seed: int = 42) -> tuple[list[AnnotatedSequence], list[AnnotatedSequence]]:
    """Build a synthetic CATH-annotated corpus with 30 topologies, 60 H-families, + singletons."""
    rng = np.random.default_rng(seed)
    cath_recs: list[AnnotatedSequence] = []
    idx = 0
    for t_i in range(1, 31):
        topo = f"1.10.{t_i}"
        # 2 H-superfamilies per topology; first 16 topologies are deep, rest moderate
        n_h = 2
        depth = 80 if t_i <= 16 else 24
        for h_i in range(1, n_h + 1):
            sf = f"{topo}.{h_i}"
            for _ in range(depth):
                L = int(rng.integers(50, 90))
                seq = "".join(AA20[int(c)] for c in rng.integers(0, 20, size=L))
                cath_recs.append(
                    AnnotatedSequence(seq_id=f"U{idx:06d}", sequence=seq, cath_h=sf, cath_t=topo)
                )
                idx += 1

    singletons: list[AnnotatedSequence] = []
    for s_i in range(400):
        L = int(rng.integers(50, 90))
        seq = "".join(AA20[int(c)] for c in rng.integers(0, 20, size=L))
        singletons.append(
            AnnotatedSequence(seq_id=f"S{s_i:06d}", sequence=seq, cath_h="", cath_t="")
        )
    return cath_recs, singletons


def test_fold_disjoint_splits_and_token_budget_matching(tmp_path: Path) -> None:
    cath_recs, singletons = _make_synthetic_corpus(seed=101)
    # Exclude one specific sequence as a simulated MegaScale/ProteinGym WT
    excluded_seq = cath_recs[0].sequence
    excluded_hashes = {seq_sha256(excluded_seq)}

    res = build_controlled_natural_splits(
        cath_recs,
        singleton_uniref_pool=singletons,
        priority_unseen_topologies=["1.10.25", "1.10.26", "1.10.27"],
        excluded_eval_seq_hashes=excluded_hashes,
        target_unique_tokens=15_000,
        arm_family_targets=(4, 12, 24),
        include_singleton_arm=True,
        n_unseen_fold_eval=120,
        min_unseen_topologies=6,
        n_homolog_eval=40,
        n_train_subsample_eval=32,
        n_global_natural_eval=64,
        token_rel_tol=0.002,
        seed=2026,
    )

    unseen_topos = set(res["unseen_fold_topologies"])
    assert len(unseen_topos) >= 6
    assert {"1.10.25", "1.10.26", "1.10.27"}.issubset(unseen_topos)
    assert len(res["eval_unseen_fold"]) == 120

    assert set(res["arms"].keys()) == {"M4", "M12", "M24", "M_max_singleton"}
    for arm_id, arm in res["arms"].items():
        # Exact <0.2% token budget match across ALL arms
        assert arm["token_budget_rel_error"] <= 0.002
        train_topos = {r.cath_t for r in arm["train"] if r.cath_t}
        assert len(train_topos & unseen_topos) == 0
        # Excluded WT hash must never enter train
        assert seq_sha256(excluded_seq) not in {seq_sha256(r.sequence) for r in arm["train"]}
        # Homolog eval must come strictly from active training CATH H-superfamilies
        train_hs = {r.cath_h for r in arm["train"] if r.cath_h}
        for h_rec in arm["eval_unseen_homolog"]:
            assert h_rec.cath_h in train_hs
            assert h_rec.cath_t not in unseen_topos

    # Round-trip save & load via Stage2Bundle
    splits_dir = tmp_path / "splits"
    save_packed_split_npz(splits_dir / "eval_unseen_fold.npz", res["eval_unseen_fold"])
    save_packed_split_npz(splits_dir / "eval_global_natural.npz", res["eval_global_natural"])
    arms_manifest = {}
    for arm_id, arm in res["arms"].items():
        save_packed_split_npz(splits_dir / f"train_{arm_id}.npz", arm["train"])
        save_packed_split_npz(splits_dir / f"eval_train_sub_{arm_id}.npz", arm["eval_train_sub"])
        save_packed_split_npz(
            splits_dir / f"eval_unseen_homolog_{arm_id}.npz", arm["eval_unseen_homolog"]
        )
        arms_manifest[arm_id] = {"arm_id": arm_id}

    atomic_write_json_with_sha256(
        tmp_path / "bundle_manifest.json",
        {"schema": BUNDLE_SCHEMA, "arms": arms_manifest},
    )
    bundle = Stage2Bundle(tmp_path)
    loaded = bundle.load_arm_splits("M4")
    assert loaded["train"].total_tokens == res["arms"]["M4"]["realized_tokens"]
    assert loaded["eval_unseen_fold"].n_sequences == 120


def test_verify_fold_disjointness_catches_leakage() -> None:
    """Negative test: leaking a held-out C.A.T fold into training must raise AssertionError."""
    unseen = [AnnotatedSequence("u1", "A" * 55, "1.10.99.1", "1.10.99")]
    leaked_train = [AnnotatedSequence("t1", "C" * 55, "1.10.99.2", "1.10.99")]
    with pytest.raises(AssertionError, match="leaked"):
        verify_fold_disjointness(
            unseen_topologies={"1.10.99"},
            eval_unseen_fold=unseen,
            eval_global_natural=[],
            arms_data={"M_bad": {"train": leaked_train, "eval_unseen_homolog": []}},
        )
