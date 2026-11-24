"""Unit tests for Experiment 4.1 dual-axis unconditional generation & anchored controls."""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.unified.r5_experiment_4_1 import (
    POOL_SPECS_34,
    PoolEntrySpec,
    _aggregate_one_entry,
    _build_stage4_anchored_controls,
    bootstrap_mean_ci95,
    classify_identity_band,
    seq_sha256,
    sequence_distinct_5mer_fraction,
    sequence_shannon_entropy_bits,
)


def test_pool_specs_34_structure() -> None:
    assert len(POOL_SPECS_34) == 34
    keys = [s.entry_key for s in POOL_SPECS_34]
    assert len(set(keys)) == 34
    controls = [s for s in POOL_SPECS_34 if s.is_control]
    assert {c.entry_key for c in controls} == {"natural_fragments", "markov_order4"}
    non_controls = [s for s in POOL_SPECS_34 if not s.is_control]
    assert len(non_controls) == 32
    distinct_arms = {s.arm_id for s in non_controls}
    assert len(distinct_arms) == 29
    maxconf = [s for s in non_controls if not s.is_primary_mode]
    assert {m.entry_key for m in maxconf} == {
        "dplm-150m|max_confidence",
        "dplm-650m|max_confidence",
        "dplm-3b|max_confidence",
    }


def test_complexity_and_identity_helpers() -> None:
    # Homopolymer has 0 entropy and low distinct 5-mer rate
    poly_l = "L" * 100
    assert sequence_shannon_entropy_bits(poly_l) == 0.0
    assert math.isclose(sequence_distinct_5mer_fraction(poly_l), 1.0 / 96.0, rel_tol=1e-12)

    # Uniform 20-AA repeat has exact log2(20) entropy
    uni20 = "ACDEFGHIKLMNPQRSTVWY" * 5
    assert math.isclose(sequence_shannon_entropy_bits(uni20), math.log2(20.0), rel_tol=1e-12)

    assert classify_identity_band(0.0) == "lt30_novel"
    assert classify_identity_band(29.9) == "lt30_novel"
    assert classify_identity_band(30.0) == "30_to_70_homolog"
    assert classify_identity_band(70.0) == "30_to_70_homolog"
    assert classify_identity_band(70.1) == "gt70_near_copy"

    ci = bootstrap_mean_ci95([1.0, 2.0, 3.0, 4.0, 5.0], draws=500, seed=7)
    assert ci["n"] == 5
    assert math.isclose(ci["mean"], 3.0, rel_tol=1e-12)
    assert ci["ci95"][0] < ci["mean"] < ci["ci95"][1]


def test_strict_novel_gate_rejects_homopolymer_and_homolog_illusions() -> None:
    # Sequence 0: homopolymer collapse (pLDDT=85, UniRef50=0%, H=0.0b) -> MUST be rejected by H>=3.0 gate!
    # Sequence 1: natural homolog copy (pLDDT=90, UniRef50=85%, H=4.1b) -> MUST be rejected by <30% gate!
    # Sequence 2: true novel high-entropy foldable protein (pLDDT=80, UniRef50=12%, H=4.32b) -> PASSES!
    # Sequence 3: true novel high-entropy unfolded protein (pLDDT=45, UniRef50=5%, H=4.32b) -> in gated set, fails >=70!
    s0 = "L" * 80
    s1 = "ACDEFGHIKLMNPQRSTVWY" * 4
    s2 = ("ACDEFGHIKLMNPQRSTVWY" + "YVWTSRQPNMLKIHGFEDCA") * 2
    s3 = ("MNPQRSTVWYACDEFGHIKL" + "LKIHGFEDCAYVWTSRQPNM") * 2

    rows = [
        {
            "seq_id": f"s{i}",
            "sequence": seq,
            "sequence_sha256": seq_sha256(seq),
            "length": len(seq),
            "valid_aa20": True,
            "supported": True,
            "budget_censored": False,
            "cached_any_profile": None,
            "structure": {"mean_ca_plddt": plddt, "ptm": 0.5, "fold_status": "ok"},
        }
        for i, (seq, plddt) in enumerate([(s0, 85.0), (s1, 90.0), (s2, 80.0), (s3, 45.0)])
    ]
    dm_cache = {
        seq_sha256(s0): {"identity_over_query": 0.0, "band": "lt30_novel"},
        seq_sha256(s1): {"identity_over_query": 85.0, "band": "gt70_near_copy"},
        seq_sha256(s2): {"identity_over_query": 12.0, "band": "lt30_novel"},
        seq_sha256(s3): {"identity_over_query": 5.0, "band": "lt30_novel"},
    }
    pf_cache = {
        seq_sha256(s1): {"any_family": True, "complete_domain": True, "best_profile_coverage": 0.95, "families": ["PF00001"]},
    }
    spec = PoolEntrySpec("synth", "synth", "control", "synthetic", "synthetic", is_control=True)
    agg = _aggregate_one_entry(
        {"spec": spec, "rows": rows, "source_paths": []},
        dm_cache,
        pf_cache,
        {},
        draws=200,
        seed=42,
    )
    a2 = agg["axis2_structure_joined"]
    assert a2["n_folded"] == 4
    # Unfiltered: 3/4 have pLDDT >= 70 (s0, s1, s2)
    assert math.isclose(a2["all_folded"]["foldable_rate_plddt_ge_70"]["mean"], 0.75, rel_tol=1e-12)
    # Strictly gated (<30% AND H>=3.0): ONLY s2 and s3 pass (n=2), and 1/2 has pLDDT>=70 (cond=0.5, pool yield=0.25)
    assert a2["gated_lt30_h_ge_3_0"]["n"] == 2
    assert math.isclose(a2["gated_lt30_h_ge_3_0"]["foldable_rate_plddt_ge_70"]["mean"], 0.50, rel_tol=1e-12)
    assert math.isclose(a2["gated_lt30_h_ge_3_0"]["gated_yield_of_pool"]["mean"], 0.25, rel_tol=1e-12)
    assert a2["rejected_low_entropy_lt_3_0"]["n"] == 1
    assert a2["homolog_ge30_subset"]["n"] == 1


def test_stage4_anchored_controls_live() -> None:
    st4 = _build_stage4_anchored_controls(REPO_ROOT)
    r5a = st4["progressive_scaffold_anchoring_r5a"]
    assert r5a["n_gp_and_esm3_evals"] == 16
    assert r5a["n_original_ladder_entries"] == 7
    assert len(r5a["esm3_r5c_multimodal_conditioning"]) == 3
    lbl = st4["explicit_class_label_recall"]
    assert lbl["zymctrl"]["n_admitted_classes"] == 14
    assert lbl["zymctrl"]["max_p_mismatched"] == 0.0
    assert lbl["prollama"]["n_admitted_classes"] == 15
    assert lbl["prollama"]["max_p_mismatched"] == 0.0


def test_live_master_artifact_invariants() -> None:
    import json

    out_path = (
        REPO_ROOT
        / "results/extensions/unified_yardstick_20261124/r5_experiment_4_1_unified_generation.json"
    )
    if not out_path.is_file():
        return
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    entries = doc["entries_full_34"]
    assert len(entries) == 34
    for key, e in entries.items():
        a1 = e["axis1_full_pool"]
        assert a1["n_supported"] >= 1, f"{key} has zero supported sequences"
        assert a1["mean_entropy_bits"] > 0.0, f"{key} has zero mean entropy"
    assert entries["progen2-medium"]["axis1_full_pool"]["n_supported"] == 777
    assert entries["progen3-3b"]["axis1_full_pool"]["n_supported"] == 800
    assert entries["galactica-125m"]["axis1_full_pool"]["n_supported"] == 1
    assert entries["markov_order4"]["axis1_full_pool"]["pfam_complete_domain_rate_all"] < 0.01
    assert entries["natural_fragments"]["axis1_full_pool"]["pfam_complete_domain_rate_all"] > 0.80
    # Decoding artifact invariant: DPLM max_confidence collapses under strict gate while uniform_random retains yield
    assert entries["dplm-3b|max_confidence"]["axis2_structure_joined"]["gated_lt30_h_ge_3_0"]["n"] <= 5
    assert entries["dplm-3b|uniform_random"]["axis2_structure_joined"]["gated_lt30_h_ge_3_0"]["n"] > 500
