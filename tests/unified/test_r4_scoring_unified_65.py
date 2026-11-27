"""Unit & parity tests for Workstream R4-Score (65/65 unaligned homologous-prefix scoring)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.capability.unified.r4_scoring_yardstick import (
    CACHED_EARLY_PROTEIN_ARMS,
    CACHED_GP_SINGLE_ARMS,
    CACHED_MSA_ARMS,
    DEFAULT_COHORT_REL,
    FORBIDDEN_INFRA_RE,
    GROUP_A_CAUSAL_TEXT_ARMS,
    GROUP_B_BIDIRECTIONAL_ARMS,
    NEW_SCORE_ARMS,
    SCHEMA_MASTER,
    build_8fold_masks,
    build_r4_scoring_unified_65,
    build_unit_condition_inputs,
    deterministic_shuffle,
    pack_bidirectional_context,
    pack_causal_text_context,
    pick_unrelated_partners,
)
from src.capability.unified.roster import (
    REPO_ROOT,
    UNIFIED_3_MSA_CONTEXT_ARMS,
    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
    UNIFIED_65_ARMS,
)


def test_roster_partition_is_exact_65_arms() -> None:
    assert len(CACHED_EARLY_PROTEIN_ARMS) == 19
    assert len(CACHED_GP_SINGLE_ARMS) == 7
    assert len(GROUP_A_CAUSAL_TEXT_ARMS) == 14
    assert len(GROUP_B_BIDIRECTIONAL_ARMS) == 22
    assert len(CACHED_MSA_ARMS) == 3
    assert len(NEW_SCORE_ARMS) == 36
    single = set(CACHED_EARLY_PROTEIN_ARMS) | set(CACHED_GP_SINGLE_ARMS) | set(NEW_SCORE_ARMS)
    assert single == set(UNIFIED_62_SINGLE_SEQUENCE_ARMS)
    assert single | set(CACHED_MSA_ARMS) == set(UNIFIED_65_ARMS)
    assert not (single & set(UNIFIED_3_MSA_CONTEXT_ARMS))


def test_8fold_mask_partition_is_disjoint_complete_and_deterministic() -> None:
    for length in (5, 8, 17, 100, 257):
        f1 = build_8fold_masks("protein|id_70_90|retained|000252", length)
        f2 = build_8fold_masks("protein|id_70_90|retained|000252", length)
        assert f1 == f2
        assert len(f1) == min(8, length)
        flat = [idx for fold in f1 for idx in fold]
        assert len(flat) == length
        assert sorted(flat) == list(range(length))
        # Balanced fold sizes (differ by at most 1)
        sizes = [len(fold) for fold in f1]
        assert max(sizes) - min(sizes) <= 1


def test_bidirectional_and_causal_context_packing_respects_budget_and_invariance() -> None:
    # Bidirectional packing: target span exact, length <= 1024
    tgt = list(range(10, 110))  # 100 residues
    ctxs = [list(range(20, 220)) for _ in range(10)]  # 200 residues each
    full, s, e, k = pack_bidirectional_context(
        tgt, ctxs, bos_prefix=[0], eos_suffix=[2], sep_ids=[2], budget_tokens=1024
    )
    assert len(full) <= 1024
    assert full[s:e] == tgt
    assert k == 4  # 1 + 4*(200+1) + 100 + 1 = 906 <= 1024; 5th would be 1107 > 1024

    # Causal packing: scored token count strictly identical between no_context and homologue
    for mode, bos, eos, expected_scored in (
        ("sep_prefix", 50256, 50256, len(tgt)),
        ("bos_eos_blocks", 128000, 128001, len(tgt)),
        ("bare_eos_sep", None, 1, len(tgt) - 1),
    ):
        f0, s0, e0, k0 = pack_causal_text_context(
            tgt, [], mode=mode, bos_id=bos, eos_id=eos, budget_tokens=1024
        )
        f1, s1, e1, k1 = pack_causal_text_context(
            tgt, ctxs, mode=mode, bos_id=bos, eos_id=eos, budget_tokens=1024
        )
        assert k0 == 0 and k1 >= 4
        assert len(f0) <= 1024 and len(f1) <= 1024
        assert (e0 - s0) == (e1 - s1) == expected_scored
        assert f0[s0:e0] == f1[s1:e1]


def test_partner_shuffle_filler_parity_with_run_r4_multiseq_context() -> None:
    cohort_path = REPO_ROOT / DEFAULT_COHORT_REL
    if not cohort_path.is_file():
        pytest.skip("Frozen EXP-R2-228 cohort absent")

    # Dynamically load run_r4_multiseq_context.py to verify bit-for-bit identical helper outputs
    script_path = REPO_ROOT / "scripts/capability/general_plm/run_r4_multiseq_context.py"
    spec = importlib.util.spec_from_file_location("run_r4_multiseq_context_ref", script_path)
    assert spec is not None and spec.loader is not None
    ref_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_mod)

    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    prot = cohort["protein"]
    records = prot["records"]
    filler_idx = int(prot["filler"]) if isinstance(prot["filler"], int) else int(prot["filler"].get("index", 0))
    filler_seq = (
        records[filler_idx]
        if isinstance(prot["filler"], int)
        else str(prot["filler"].get("record") or records[filler_idx])
    )
    related = prot.get("related", {})
    retained = [u for u in prot["units"] if u.get("stratum") == "retained"][:16]

    for unit in retained:
        tgt, conds = build_unit_condition_inputs(
            unit, records, related, filler_idx=filler_idx, filler_seq=filler_seq
        )
        hom = [records[int(p)] for p in unit["partners"]]
        assert conds["homologue"] == hom
        assert conds["unrelated"] == ref_mod._pick_unrelated_partners(
            unit, records, related, len(hom), filler_idx=filler_idx
        )
        assert conds["unrelated"] == pick_unrelated_partners(
            unit, records, related, len(hom), filler_idx=filler_idx
        )
        assert conds["mono_shuffled"] == [
            ref_mod._deterministic_shuffle(s, f"{unit['key']}|mono|{i}") for i, s in enumerate(hom)
        ]
        assert conds["mono_shuffled"][0] == deterministic_shuffle(hom[0], f"{unit['key']}|mono|0")
        assert len(tgt) > 0


@pytest.mark.skipif(
    not (REPO_ROOT / DEFAULT_COHORT_REL).is_file(),
    reason="Frozen EXP-R2-228 cohort absent",
)
def test_cached_29_arm_parity_and_strict_refusal(tmp_path: Path) -> None:
    # Strict build against empty work_dir must refuse (36 arms missing)
    with pytest.raises(RuntimeError, match="Strict 65-arm R4 scoring build refused"):
        build_r4_scoring_unified_65(
            repo_root=REPO_ROOT,
            work_dir=tmp_path / "empty_work",
            out_path=tmp_path / "out.json",
            allow_pending=False,
        )

    # Interim build against empty work_dir must pass exact <1e-12 parity gate on all 29 cached protein arms!
    res = build_r4_scoring_unified_65(
        repo_root=REPO_ROOT,
        work_dir=tmp_path / "empty_work",
        out_path=tmp_path / "out_interim.json",
        allow_pending=True,
        draws=200,
    )
    p = res["payload"]
    assert p["schema"] == SCHEMA_MASTER
    assert p["completion_status"] == "interim_pending"
    assert p["coverage"]["n_arms_present"] == 29
    assert p["coverage"]["n_pending_arms"] == 36
    assert p["reproduction_gate"]["passed"] is True
    assert p["reproduction_gate"]["n_cached_arms_verified"] == 29
    assert p["reproduction_gate"]["max_abs_diff"] < 1e-12
    assert "gpt2-large|text_bm25_frozen" in p["companion_controls"]
    ht = p["horizontal_benchmark_table"]
    assert len(ht) == 29 + 3  # 29 cached arms + 3 sequence-context companion controls
    for row in ht:
        assert row["input_alignment_format"] in ("explicit_column_aligned", "unaligned_concat")
        assert row["context_training_regime"] in (
            "single_chain_causal",
            "single_chain_bidirectional",
            "multiseq_icl_unaligned",
            "msa_2d_axial",
        )
        if row["row_key"] == "msa-transformer-100m|b1024":
            assert row["input_alignment_format"] == "explicit_column_aligned"
        else:
            assert row["input_alignment_format"] == "unaligned_concat"
    assert not FORBIDDEN_INFRA_RE.search((tmp_path / "out_interim.json").read_text(encoding="utf-8"))
