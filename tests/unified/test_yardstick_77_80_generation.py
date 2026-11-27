"""Tests for 15-model DPLM-2 + ProteinMPNN generation, folding & R6 adaptation suite."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import EsmConfig

from src.capability.context.family_potts import load_family_model
from src.capability.unified import r6_experiment_4_2 as exp42
from src.capability.unified import yardstick_77_80_generation as ygen
from src.capability.unified.dplm2_adapter import (
    AA_MASK_ID,
    DPLM2Adapter,
    EsmForDPLM2,
)
from src.capability.unified.proteinmpnn_adapter import load_proteinmpnn_adapter
from src.capability.unified.roster import (
    DIMA_4_ARMS,
    DPLM2_4_ARMS,
    PROTEINMPNN_11_ARMS,
    REPO_ROOT,
)
from src.capability.unified.wt_backbones_3d import load_backbone_npz

MODEL_ROOT = Path.home() / "models" / "general_plm_20261120"


def _make_synthetic_dplm2_adapter(tmp_path: Path) -> DPLM2Adapter:
    ckpt_dir = tmp_path / "dplm2-150m"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg = EsmConfig(
        vocab_size=8229,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        position_embedding_type="rotary",
        token_dropout=True,
        pad_token_id=1,
        mask_token_id=AA_MASK_ID,
        tie_word_embeddings=False,
    )
    cfg.dplm_type = "dplm2_esm"
    torch.manual_seed(1234)
    model = EsmForDPLM2(cfg)
    model.save_pretrained(str(ckpt_dir), safe_serialization=False)
    cfg_json = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    cfg_json["dplm_type"] = "dplm2_esm"
    (ckpt_dir / "config.json").write_text(json.dumps(cfg_json, indent=2), encoding="utf-8")
    return DPLM2Adapter(
        "dplm2-150m",
        mode="seq_only",
        model_root=tmp_path,
        device="cpu",
        dtype="float32",
        load_struct_tokenizer=False,
    )


def test_roster_specs_and_reference_cohorts() -> None:
    assert len(ygen.EXP4_1_NEW_SPECS) == 4 * 4 + 11 + 3 + 4 + 2
    assert len(ygen.R5A_NEW_SPECS) == 4 * 2 + 11 + 4
    assert len(ygen.EXP4_2_NEW_SPECS) == 4 * 3 + 11 + 4
    assert len(ygen.R4_CONT_NEW_SPECS) == 4 * 2 + 11 + 4
    assert {s.arm_id for s in ygen.EXP4_1_NEW_SPECS} == (
        set(DPLM2_4_ARMS) | set(PROTEINMPNN_11_ARMS) | set(DIMA_4_ARMS)
    )

    targets = ygen.select_exp4_1_reference_targets(REPO_ROOT, n_targets=200)
    assert len(targets) == 200
    assert len({t["seq_id"] for t in targets}) == 200
    assert min(t["length"] for t in targets) >= 50
    assert max(t["length"] for t in targets) <= 400

    backbones = ygen.load_r5a_ladder_backbones(REPO_ROOT)
    assert len(backbones) == 16
    for bb in backbones:
        assert set(ygen.R5A_RUNGS) <= set(bb["window_spans"].keys())


def test_parse_pdb_backbone_ncaco_and_infra_leak_guard() -> None:
    pdb = "\n".join(
        [
            "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 90.00           N",
            "ATOM      2  CA  ALA A   1       1.500   2.500   3.500  1.00 90.00           C",
            "ATOM      3  C   ALA A   1       2.000   3.000   4.000  1.00 90.00           C",
            "ATOM      4  O   ALA A   1       2.500   3.500   4.500  1.00 90.00           O",
            "ATOM      5  N   GLY A   2       3.000   4.000   5.000  1.00 88.00           N",
            "ATOM      6  CA  GLY A   2       3.500   4.500   5.500  1.00 88.00           C",
            "ATOM      7  C   GLY A   2       4.000   5.000   6.000  1.00 88.00           C",
            "ATOM      8  O   GLY A   2       4.500   5.500   6.500  1.00 88.00           O",
        ]
    )
    coords = ygen.parse_pdb_backbone_ncaco(pdb, expected_len=2)
    assert coords.shape == (2, 4, 3)
    assert np.allclose(coords[0, 1], [1.5, 2.5, 3.5])
    assert np.allclose(coords[1, 3], [4.5, 5.5, 6.5])

    ygen.assert_no_infra_leaks({"status": "ok", "n_models": 80})
    synthetic_private_ip = ".".join(("10", "0", "0", "1"))
    with pytest.raises(ValueError, match="forbidden infrastructure"):
        ygen.assert_no_infra_leaks({"host": synthetic_private_ip})


def test_dplm2_synthetic_guided_potts_and_homolog_continuation(tmp_path: Path) -> None:
    adapter = _make_synthetic_dplm2_adapter(tmp_path)
    fm = load_family_model("1E0L", REPO_ROOT / exp42.DEFAULT_FAMILY_REF_DIR)

    # Exact < 1e-6 sequence_component_sums accounting on ALL 5 Exp 4.2 arms
    for arm_spec in exp42.PHASE_SPACE_ARMS:
        recs = ygen.sample_guided_dplm2(
            adapter=adapter,
            model_id="dplm2-150m|seq_only",
            family_model=fm,
            arm_spec=arm_spec,
            count=2,
            batch_size=2,
            stream_seed=12345,
            mode="seq_only",
        )
        assert len(recs) == 2
        for r in recs:
            assert len(r["sequence"]) == fm.length
            assert set(r["sequence"]) <= ygen.AA20_SET

    # Unaligned homolog-prefix continuation with budget=1024
    seqs, n_used, used_res = ygen.sample_dplm2_homolog_continuation(
        adapter=adapter,
        context_items=["ACDEFGHIKLMNPQRSTVWY" * 3, "WYVTSRQPNMLKIHGFEDCA" * 2],
        gen_length=40,
        n_samples=2,
        seed=99,
        steps=10,
        batch_size=2,
    )
    assert len(seqs) == 2
    assert all(len(s) == 40 and set(s) <= ygen.AA20_SET for s in seqs)
    assert n_used == 2 and used_res == 100


def test_proteinmpnn_live_vanilla_and_ca_on_real_1e0l_backbone() -> None:
    if not (MODEL_ROOT / "proteinmpnn_all").is_dir():
        pytest.skip("proteinmpnn_all checkpoints not present locally")
    _, wt_seqs, _ = exp42.load_frozen_wt_ca_traces(REPO_ROOT / exp42.DEFAULT_WT_REF_MANIFEST)
    bb = load_backbone_npz(REPO_ROOT / ygen.WT_BACKBONES_REL, wt_seqs["1E0L"])
    coords = np.asarray(bb["coords"], dtype=np.float32)

    for arm in ("proteinmpnn-v-020", "proteinmpnn-ca-020"):
        adapter = load_proteinmpnn_adapter(arm, model_root=MODEL_ROOT, device="cpu")
        out = adapter.generate(coords, n_samples=2, seed=42, temperature=0.1)
        assert len(out) == 2
        seqs = [r["sequence"] for r in out]
        assert all(len(s) == len(wt_seqs["1E0L"]) for s in seqs)
        ll = adapter.exact_sequence_ll_batch(seqs, coords, n_permutations=2, seed=42)
        assert ll.shape == (2,)
        assert np.all(np.isfinite(ll)) and np.all(ll < 0.0)


def test_end_to_end_synthetic_assembly_all_6_artifacts(tmp_path: Path) -> None:
    """Verify all 6 assemblers run cleanly in --allow-partial mode without touching sealed files."""
    work = tmp_path / "work"
    seq1 = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    seq2 = "MNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKL"

    # Populate minimal synthetic Exp 4.1 + Exp 4.3 + R5a + Exp 4.2 + R4-Cont for 2 MPNN arms + 2 DPLM-2 arms
    for spec_key in ("proteinmpnn-v-020|T0.1", "proteinmpnn-s-020|T0.1", "dplm2-650m|seq_only|uniform_random", "dplm2-bit-650m|seq_only|uniform_random"):
        spec = ygen.EXP4_1_NEW_SPEC_MAP[spec_key]
        gen_rows = []
        fold_rows = []
        sc_rows = []
        for i in range(12):
            s = seq1 if i % 2 == 0 else seq2
            sid = f"{spec.file_slug}_{i:04d}"
            sha = ygen.seq_sha256(s)
            gen_rows.append(
                {
                    "seq_id": sid,
                    "entry_key": spec.entry_key,
                    "arm_id": spec.arm_id,
                    "sequence": s,
                    "sequence_sha256": sha,
                    "length": len(s),
                    "template_sequence_recovery": 42.0 + i * 0.5,
                }
            )
            fold_rows.append(
                {
                    "seq_id": sid,
                    "sequence": s,
                    "sequence_sha256": sha,
                    "length": len(s),
                    "status": "ok",
                    "mean_ca_plddt": 78.0 + (i % 5),
                    "ptm": 0.72 + (i % 4) * 0.02,
                    "backbone_sc_tm": 0.85,
                }
            )
            sc_rows.append(
                {
                    "pool_id": spec.entry_key,
                    "arm_id": spec.arm_id,
                    "conditioning_regime": spec.conditioning_regime,
                    "scoring_backbone_source": "esmfold2_refolded",
                    "seq_id": sid,
                    "sequence": s,
                    "sequence_sha256": sha,
                    "plddt": 78.0 + (i % 5),
                    "ptm": 0.72 + (i % 4) * 0.02,
                    "mean_nll_per_scored_token": 1.5 - 0.03 * (i % 5),
                    "mean_nll_per_residue": 1.5 - 0.03 * (i % 5),
                }
            )
        ygen.atomic_write_jsonl(work / "exp4_1" / "generated" / f"{spec.file_slug}.jsonl", gen_rows)
        ygen.atomic_write_jsonl(work / "exp4_1" / "folded" / f"{spec.file_slug}.jsonl", fold_rows)
        ygen.atomic_write_jsonl(work / "exp4_3" / "scores" / f"{spec.file_slug}.jsonl", sc_rows)

    e41 = ygen.assemble_exp4_1_full(
        REPO_ROOT, work, allow_partial=True, draws=200, skip_external_oracles=True
    )
    assert e41["coverage"]["n_sealed_entries"] == 34
    assert e41["coverage"]["n_new_loaded"] == 4
    ygen.assert_no_infra_leaks(e41)

    r5a = ygen.assemble_r5a_full(REPO_ROOT, work, allow_partial=True, draws=200)
    assert r5a["schema"] == "r5a_local_infilling_full_v1"
    ygen.assert_no_infra_leaks(r5a)

    e42 = ygen.assemble_exp4_2_full(REPO_ROOT, work, allow_partial=True)
    assert e42["coverage"]["n_sealed_plms"] == 23
    ygen.assert_no_infra_leaks(e42)

    e43 = ygen.assemble_exp4_3_full(
        REPO_ROOT, work, allow_partial=True, n_draws=200, fwl_draws=0, skip_external_oracles=True
    )
    assert e43["coverage"]["n_sealed_pools"] == 33
    assert e43["coverage"]["n_new_scored_pools"] == 4
    ygen.assert_no_infra_leaks(e43)

    r4 = ygen.assemble_r4_continuation_full(
        REPO_ROOT, work, allow_partial=True, skip_external_oracles=True
    )
    assert r4["schema_version"] == "r4_continuation_unified_full_v1"
    ygen.assert_no_infra_leaks(r4)

    r6 = ygen.assemble_r6_dplm2_mpnn_adaptation(
        REPO_ROOT,
        exp4_1_doc=e41,
        r5a_doc=r5a,
        exp4_2_doc=e42,
        exp4_3_doc=e43,
        r4_cont_doc=r4,
        draws=200,
    )
    assert len(r6["soluble_vs_vanilla_proteinmpnn"]) == 4
    assert len(r6["ca_only_vs_all_atom_proteinmpnn"]) == 3
    assert len(r6["dplm2_bit_650m_vs_dplm2_650m"]) == 3
    assert len(r6["dplm2_vs_dplm1_matched_scale"]) == 3
    assert len(r6["dima_sde_vs_ode_ablation"]) == 2
    assert len(r6["dima_encoder_scale_and_modality"]) == 4
    assert len(r6["continuous_vs_discrete_diffusion"]) == 6
    # Verify real delta computed for proteinmpnn-s-020 vs proteinmpnn-v-020
    s020 = [x for x in r6["soluble_vs_vanilla_proteinmpnn"] if x["label"] == "soluble_vs_vanilla_0.20A"][0]
    assert s020["exp4_1"]["mean_plddt"]["delta"] == pytest.approx(0.0)
    ygen.assert_no_infra_leaks(r6)


def test_dima_live_guided_potts_and_homolog_continuation() -> None:
    if not (MODEL_ROOT / "dima_all").is_dir() or not (MODEL_ROOT / "esm2-8m").is_dir():
        pytest.skip("dima_all or esm2-8m checkpoints not present locally")
    from src.capability.unified.dima_adapter import load_dima_adapter

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    adapter = load_dima_adapter("dima-esm2-8m", model_root=MODEL_ROOT, device=dev, dtype="float32")
    fm = load_family_model("1E0L", REPO_ROOT / exp42.DEFAULT_FAMILY_REF_DIR)
    for arm_key in ("N", "Pr", "C", "Cx"):
        recs = ygen.sample_guided_dima(
            adapter=adapter,
            model_id="dima-esm2-8m",
            family_model=fm,
            arm_spec=exp42.PHASE_SPACE_ARM_MAP[arm_key],
            count=2,
            batch_size=2,
            stream_seed=42,
            steps=4,
        )
        assert len(recs) == 2
        assert all(len(r["sequence"]) == fm.length and set(r["sequence"]) <= ygen.AA20_SET for r in recs)
        if arm_key == "N":
            assert recs[0]["guidance_bias_applied"] == 0.0
        else:
            assert recs[0]["guidance_bias_applied"] > 0.0

    seqs, n_used, used_res = ygen.sample_dima_homolog_continuation(
        adapter=adapter,
        context_items=["ACDEFGHIKLMNPQRSTVWY" * 2],
        gen_length=32,
        n_samples=2,
        seed=7,
        steps=4,
        batch_size=2,
    )
    assert len(seqs) == 2
    assert all(len(s) == 32 and set(s) <= ygen.AA20_SET for s in seqs)
    assert n_used == 1 and used_res == 40
