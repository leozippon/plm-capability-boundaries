"""Unit & reproduction-gate tests for Unified Result 5 Downstream-1 (DMS homolog-context ranking)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.general_plm.msa_utils import build_query_msa
from src.capability.unified.r5_dms_context_yardstick import (
    DMS_CONTEXT_CORE_KEYS,
    DMS_CONTEXT_SPECS_ALL,
    assign_balanced_shards,
    build_r5_dms_context_unified,
    fast_build_query_msa,
    longest_common_substring_len,
    plan_rescue_30_80_assay,
    verify_subtest_1a_reproduction,
    verify_subtest_1b_reproduction,
)


def test_fast_nw_projection_matches_reference_bit_for_bit() -> None:
    import numpy as np
    from src.capability.core.amino_acids import AA20

    rng = np.random.default_rng(99)
    for _ in range(10):
        q = "".join(rng.choice(list(AA20), size=int(rng.integers(25, 65))))
        partners = [
            "".join(rng.choice(list(AA20), size=int(rng.integers(20, 70))))
            for _ in range(3)
        ]
        assert fast_build_query_msa(q, partners, max_depth=4, aligned=True) == build_query_msa(
            q, partners, max_depth=4, aligned=True
        )
        assert fast_build_query_msa(q, partners, max_depth=4, aligned=False) == build_query_msa(
            q, partners, max_depth=4, aligned=False
        )


def test_input_alignment_format_contract_and_panel_coverage() -> None:
    assert len(DMS_CONTEXT_SPECS_ALL) == 68
    assert len(DMS_CONTEXT_CORE_KEYS) == 57
    aligned_entries = [
        s.entry_key for s in DMS_CONTEXT_SPECS_ALL if s.input_alignment_format == "explicit_column_aligned"
    ]
    assert aligned_entries == ["msa-transformer-100m|aligned=True"]
    # ProtGPT3-MSA, Dayhoff-3B-MSA, and MSA-Transformer aligned=False MUST be unaligned_concat
    by_k = {s.entry_key: s for s in DMS_CONTEXT_SPECS_ALL}
    for k in (
        "protgpt3-msa",
        "protgpt3-msa|b8192",
        "dayhoff-3b-msa",
        "dayhoff-3b-msa|b4096",
        "msa-transformer-100m|aligned=False",
        "esm2-15b",
        "esm3-1.4b",
        "progen3-3b",
    ):
        assert by_k[k].input_alignment_format == "unaligned_concat"
    # Verify balanced 6-way sharding covers every core entry exactly once
    core_specs = [by_k[k] for k in DMS_CONTEXT_CORE_KEYS]
    shards = assign_balanced_shards(core_specs, 6)
    flat = [s.entry_key for sh in shards for s in sh]
    assert len(flat) == len(core_specs) and set(flat) == set(DMS_CONTEXT_CORE_KEYS)


def test_lcs_and_rescue_planner() -> None:
    assert longest_common_substring_len("ACDEFGHIKLMN", "XXEFGHIKYY") == 6
    assert longest_common_substring_len("AAAA", "CCCC") == 0
    assay = {
        "wildtype": "ACDEFGHIKLMNPQRSTVWY",
        "sequences": ["VCDEFGHIKLMNPQRSTVWY"],
        "homolog_candidates": [
            {"subject": "h1", "sequence": "ACDEFGHIKLMNPQRSTVWA", "identity": 95.0},
        ],
        "unrelated_candidates": [
            {"subject": "u_bad", "sequence": "ACDEFGHIKLMNXXXXXXXX", "identity": 0.0},
            {"subject": "u_ok", "sequence": "WWWWWWWWWWYYYYYYYYYY", "identity": 0.0},
        ],
    }
    plan, reason = plan_rescue_30_80_assay(
        assay, token_length_fn=len, overhead_tokens=2, budget_tokens=100, multi_item_icl=False
    )
    assert reason is None and plan is not None
    assert plan["unrelated_items"] == ["WWWWWWWWWWYYYYYYYYYY"]
    assert plan["homolog_items"] == ["ACDEFGHIKLMNPQRSTVWA"]


def test_exact_reproduction_gates_1a_and_1b() -> None:
    r1a = verify_subtest_1a_reproduction(REPO_ROOT)
    assert r1a["passed"] is True
    assert r1a["n_common_assays"] == 164
    assert r1a["n_common_families"] == 134
    assert r1a["max_abs_diff"] < 1e-12

    r1b = verify_subtest_1b_reproduction(REPO_ROOT)
    assert r1b["passed"] is True
    assert r1b["max_abs_diff"] < 1e-12


def test_interim_partial_aggregation_live() -> None:
    doc = build_r5_dms_context_unified(
        REPO_ROOT, allow_pending=True, priority_scope="core", draws=200, seed=20261124
    )
    assert doc["reproduction_gates"]["subtest_1a_profile_increment_20260923"]["passed"] is True
    assert doc["reproduction_gates"]["subtest_1b_e09_curve_20261008"]["passed"] is True
    # All 4 cached 1A and 4 cached 1B arms are always present
    assert doc["protocol"]["n_completed_1a_rescue_30_80"] >= 4
    assert doc["protocol"]["n_completed_1b_e09_five_band"] >= 4
    # Verbatim-copy degradation verified on cached ProGen2-medium/xlarge
    p2x = doc["subtest_1b_e09_five_band"]["entries"]["progen2-xlarge"]
    assert p2x["verbatim_copy_diagnostics"]["exhibits_90_100_degradation_vs_no_context"] is True
    assert p2x["verbatim_copy_diagnostics"]["exhibits_self_copy_degradation_vs_no_context"] is True
