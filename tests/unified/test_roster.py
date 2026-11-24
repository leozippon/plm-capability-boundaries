"""Tests for the unified 65-arm cross-paradigm roster and stage-file resolvers."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.capability.interactions.pairwise_epistasis import ROSTER as PREREGISTERED_ROSTER
from src.capability.unified import (
    ARM_METADATA,
    EXPANDED_32_ARMS,
    PARADIGM_ORDER,
    PREREGISTERED_33_ARMS,
    UNIFIED_3_MSA_CONTEXT_ARMS,
    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
    UNIFIED_62_SINGLE_SEQUENCE_ARMS_ALPHA,
    UNIFIED_65_ARMS,
    UNIFIED_65_ARMS_ALPHA,
    UNIFIED_OUTPUT_DIR,
    arms_by_paradigm,
    dtype_rank,
    get_arm_metadata,
    resolve_all_gp_r1_readouts,
    resolve_all_gp_r1_scores,
    resolve_gp_family_distribution_shards,
    resolve_gp_r1_json,
    resolve_gp_r1_readout,
    resolve_gp_r1_scores,
    resolve_gp_r2_arrays,
    resolve_gp_r3_cycle_eval,
    resolve_gp_r3_cycles,
    resolve_gp_r4_json,
    resolve_gp_r5_json,
    sort_arms,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HAS_LOCAL_ARTIFACTS = (
    REPO_ROOT / "results/extensions/general_plm_20261120/general_plm_summary.json"
).is_file()


def test_roster_cardinality_and_disjointness() -> None:
    assert len(PREREGISTERED_33_ARMS) == 33
    assert len(EXPANDED_32_ARMS) == 32
    assert set(PREREGISTERED_33_ARMS) == set(PREREGISTERED_ROSTER)
    assert set(PREREGISTERED_33_ARMS).isdisjoint(set(EXPANDED_32_ARMS))

    assert len(UNIFIED_65_ARMS) == 65
    assert len(set(UNIFIED_65_ARMS)) == 65
    assert len(UNIFIED_62_SINGLE_SEQUENCE_ARMS) == 62
    assert len(set(UNIFIED_62_SINGLE_SEQUENCE_ARMS)) == 62
    assert UNIFIED_3_MSA_CONTEXT_ARMS == (
        "msa-transformer-100m",
        "protgpt3-msa",
        "dayhoff-3b-msa",
    )
    assert set(UNIFIED_62_SINGLE_SEQUENCE_ARMS) | set(UNIFIED_3_MSA_CONTEXT_ARMS) == set(
        UNIFIED_65_ARMS
    )
    assert set(UNIFIED_62_SINGLE_SEQUENCE_ARMS).isdisjoint(set(UNIFIED_3_MSA_CONTEXT_ARMS))
    assert UNIFIED_65_ARMS_ALPHA == tuple(sorted(UNIFIED_65_ARMS))
    assert UNIFIED_62_SINGLE_SEQUENCE_ARMS_ALPHA == tuple(
        sorted(UNIFIED_62_SINGLE_SEQUENCE_ARMS)
    )


def test_metadata_completeness_and_sorting() -> None:
    valid_paradigms = set(PARADIGM_ORDER)
    n_r1_readout = 0
    for arm_id in UNIFIED_65_ARMS:
        meta = get_arm_metadata(arm_id)
        assert meta.arm_id == arm_id
        assert meta.display_name
        assert meta.family
        assert meta.paradigm in valid_paradigms
        assert math.isfinite(meta.params_m) and meta.params_m > 0.0
        assert meta.origin_cohort in ("preregistered_33", "expanded_32")
        if meta.has_r1_raw_readout_1024:
            n_r1_readout += 1
            assert meta.origin_cohort == "expanded_32"
            assert meta.is_single_sequence
    assert n_r1_readout == 28
    assert not ARM_METADATA["esm3-1.4b"].has_r1_raw_readout_1024

    # Paradigm partition covers all 65 arms exactly once.
    grouped = [a for p in PARADIGM_ORDER for a in arms_by_paradigm(p)]
    assert tuple(grouped) == UNIFIED_65_ARMS
    assert sort_arms(reversed(UNIFIED_65_ARMS), by="paradigm_scale") == UNIFIED_65_ARMS
    assert sort_arms(UNIFIED_65_ARMS, by="alphabetical") == UNIFIED_65_ARMS_ALPHA

    # Output dir constant is NOT created on import.
    assert UNIFIED_OUTPUT_DIR == Path("results/extensions/unified_yardstick_20261124")


def test_dtype_rank_and_refusals() -> None:
    assert dtype_rank("torch.float32") == 2
    assert dtype_rank("float64") == 2
    assert dtype_rank("torch.bfloat16") == 1
    assert dtype_rank("bf16") == 1
    assert dtype_rank(None) == 0

    with pytest.raises(KeyError):
        get_arm_metadata("nonexistent-arm")
    with pytest.raises(ValueError):
        resolve_gp_r1_scores("progen2-medium")
    with pytest.raises(ValueError):
        resolve_gp_r1_scores("msa-transformer-100m")
    with pytest.raises(ValueError):
        resolve_gp_r1_readout("esm3-1.4b")


@pytest.mark.skipif(not HAS_LOCAL_ARTIFACTS, reason="local general_plm campaign outputs absent")
def test_local_stage_resolvers_cover_all_expanded_arms() -> None:
    expected_gp29 = {
        a for a in EXPANDED_32_ARMS if ARM_METADATA[a].is_single_sequence
    }
    expected_gp28_readouts = {
        a for a in EXPANDED_32_ARMS if ARM_METADATA[a].has_r1_raw_readout_1024
    }
    assert len(expected_gp29) == 29
    assert len(expected_gp28_readouts) == 28

    r1_scores = resolve_all_gp_r1_scores(root=REPO_ROOT)
    assert set(r1_scores) == expected_gp29
    for arm_id, art in r1_scores.items():
        assert art.path.is_file(), f"{arm_id}: missing {art.path}"
        if arm_id == "esm3-1.4b":
            assert art.path.name == "r1_ladder_scores.jsonl"
        else:
            assert art.path.suffix == ".npz"

    # Dayhoff-170M must resolve to float32 across R1, R2, R3, R4, R5 (not newer bfloat16 runs).
    assert r1_scores["dayhoff-170m"].dtype_realised == "torch.float32"
    for lane in ("abundance", "stability", "phenotype"):
        art_r2 = resolve_gp_r2_arrays("dayhoff-170m", lane, root=REPO_ROOT)
        assert art_r2.path.is_file()
        assert art_r2.dtype_realised == "torch.float32"
    art_r3 = resolve_gp_r3_cycles("dayhoff-170m", root=REPO_ROOT)
    assert art_r3.path.is_file()
    assert art_r3.dtype_realised == "torch.float32"
    art_r4 = resolve_gp_r4_json("dayhoff-170m", budget_tokens=1024, root=REPO_ROOT)
    assert art_r4.path.is_file()
    assert art_r4.dtype_realised == "torch.float32"
    art_r5 = resolve_gp_r5_json("dayhoff-170m", root=REPO_ROOT)
    assert art_r5.path.is_file()
    assert art_r5.dtype_realised == "torch.float32"

    # All 28 GP R1 1024-d readouts resolve to existing .npz files.
    r1_readouts = resolve_all_gp_r1_readouts(root=REPO_ROOT)
    assert set(r1_readouts) == expected_gp28_readouts
    for arm_id, art in r1_readouts.items():
        assert art.path.is_file(), f"{arm_id}: missing {art.path}"
        assert art.path.suffix == ".npz"

    # ESM2-15B family_distribution resolves to exactly 4 deduplicated shards (0..3 of 4).
    shards_15b = resolve_gp_family_distribution_shards("esm2-15b", root=REPO_ROOT)
    assert len(shards_15b) == 4
    assert [s.shard_index for s in shards_15b] == [0, 1, 2, 3]
    assert all(s.num_shards == 4 and s.path.is_file() for s in shards_15b)
    assert "20261122081355_edeaa99e7b89" in str(shards_15b[1].path)

    # ESM3-1.4B resolves across R1, R2, R3, R4 Route A, and R5.
    assert resolve_gp_r1_json("esm3-1.4b", root=REPO_ROOT).path.is_file()
    for lane in ("abundance", "stability", "phenotype"):
        assert resolve_gp_r2_arrays("esm3-1.4b", lane, root=REPO_ROOT).path.is_file()
    assert resolve_gp_r3_cycles("esm3-1.4b", root=REPO_ROOT).path.is_file()
    assert resolve_gp_r3_cycle_eval("esm3-1.4b", root=REPO_ROOT).path.is_file()
    assert resolve_gp_r4_json("esm3-1.4b", root=REPO_ROOT).path.is_file()
    assert resolve_gp_r5_json("esm3-1.4b", root=REPO_ROOT).path.is_file()

    # Spot-check all 29 single-sequence GP arms have R3 cycle eval JSONs cached.
    for arm_id in sorted(expected_gp29):
        assert resolve_gp_r3_cycle_eval(arm_id, root=REPO_ROOT).path.is_file()
