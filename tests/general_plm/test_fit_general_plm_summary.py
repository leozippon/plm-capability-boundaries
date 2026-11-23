"""Unit tests for precision precedence in ``fit_general_plm_summary.py``."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_summary_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts/capability/general_plm/fit_general_plm_summary.py"
    )
    spec = importlib.util.spec_from_file_location("fit_general_plm_summary", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fit_general_plm_summary_prefers_float32_over_newer_bfloat16(tmp_path: Path) -> None:
    """Older float32 runs take precedence over newer bfloat16 duplicates across modules & R3 npz."""
    mod = _load_summary_module()

    older_fp32 = tmp_path / "20261121_run"
    newer_bf16 = tmp_path / "20261122_run"
    newest_fp32 = tmp_path / "20261123_run"
    for d in (older_fp32, newer_bf16, newest_fp32):
        (d / "dayhoff_170m").mkdir(parents=True)

    # Older float32 R2/R3/R4 + sibling R3 .npz
    (older_fp32 / "dayhoff_170m" / "r2_phenotypes_and_readouts_dayhoff-170m.json").write_text(
        json.dumps(
            {
                "schema": "general_plm_r2_phenotypes_and_readouts_v1",
                "arm": "dayhoff-170m",
                "provenance": {"dtype_realised": "torch.float32"},
                "lanes": {"abundance": {"n_variants": 100, "zero_shot_spearman": 0.36}},
            }
        ),
        encoding="utf-8",
    )
    (older_fp32 / "dayhoff_170m" / "r3_pairwise_and_contacts_dayhoff-170m.json").write_text(
        json.dumps(
            {
                "schema": "general_plm_r3_pairwise_pll_and_contacts_v1",
                "arm": "dayhoff-170m",
                "provenance": {"dtype_realised": "torch.float32"},
                "pairwise_cycles": {"n_cycles": 8192},
                "structural_contacts": {"summary_auc_minus_half": 0.01},
            }
        ),
        encoding="utf-8",
    )
    np.savez(older_fp32 / "dayhoff_170m" / "r3_pairwise_cycles_dayhoff-170m.npz", marker=np.array([32]))
    (older_fp32 / "dayhoff_170m" / "r4_multiseq_dayhoff-170m.json").write_text(
        json.dumps(
            {
                "schema": "general_plm_r4_multiseq_context_v1",
                "arm": "dayhoff-170m",
                "budget_tokens": 1024,
                "aligned_msa": True,
                "provenance": {"dtype_realised": "torch.float32"},
                "bands": {"id_70_90": {"mean": 0.99}},
            }
        ),
        encoding="utf-8",
    )

    # Newer bfloat16 duplicates MUST NOT overwrite the older float32 records
    (newer_bf16 / "dayhoff_170m" / "r2_phenotypes_and_readouts_dayhoff-170m.json").write_text(
        json.dumps(
            {
                "schema": "general_plm_r2_phenotypes_and_readouts_v1",
                "arm": "dayhoff-170m",
                "provenance": {"dtype_realised": "torch.bfloat16"},
                "lanes": {"abundance": {"n_variants": 100, "zero_shot_spearman": 0.11}},
            }
        ),
        encoding="utf-8",
    )
    (newer_bf16 / "dayhoff_170m" / "r3_pairwise_and_contacts_dayhoff-170m.json").write_text(
        json.dumps(
            {
                "schema": "general_plm_r3_pairwise_pll_and_contacts_v1",
                "arm": "dayhoff-170m",
                "provenance": {"dtype_realised": "torch.bfloat16"},
                "pairwise_cycles": {"n_cycles": 8192},
                "structural_contacts": {"summary_auc_minus_half": 0.02},
            }
        ),
        encoding="utf-8",
    )
    np.savez(newer_bf16 / "dayhoff_170m" / "r3_pairwise_cycles_dayhoff-170m.npz", marker=np.array([16]))
    (newer_bf16 / "dayhoff_170m" / "r4_multiseq_dayhoff-170m.json").write_text(
        json.dumps(
            {
                "schema": "general_plm_r4_multiseq_context_v1",
                "arm": "dayhoff-170m",
                "budget_tokens": 1024,
                "aligned_msa": True,
                "provenance": {"dtype_realised": "torch.bfloat16"},
                "bands": {"id_70_90": {"mean": 0.50}},
            }
        ),
        encoding="utf-8",
    )

    # Equal-precision tie break: newest float32 R4 SHOULD beat older float32 R4
    (newest_fp32 / "dayhoff_170m" / "r4_multiseq_dayhoff-170m.json").write_text(
        json.dumps(
            {
                "schema": "general_plm_r4_multiseq_context_v1",
                "arm": "dayhoff-170m",
                "budget_tokens": 1024,
                "aligned_msa": True,
                "provenance": {"dtype_realised": "torch.float32"},
                "bands": {"id_70_90": {"mean": 0.995}},
            }
        ),
        encoding="utf-8",
    )

    # Capture which R3 .npz path is passed to the evaluator (monkeypatch to keep test instant)
    seen_npz: list[Path] = []

    def _fake_eval(npz_path: Path, arm: str, **_kwargs):
        seen_npz.append(npz_path)
        return {
            "arm": arm,
            "support": {"n_groups": 61, "n_cycles": 7805},
            "npz_sha256": "deadbeef",
            "zero_shot": {"all_64_groups_raw_epsilon": {"within_background_spearman": {"point": 0.05}}},
            "nested_cv_channels": {},
        }

    mod._evaluate_or_load_r3_cycle_eval = _fake_eval

    out_dir = tmp_path / "out"
    payload = mod.aggregate_general_plm_summary(
        [older_fp32, newer_bf16, newest_fp32],
        out_dir,
        data_root=tmp_path,
    )
    row = payload["arms"][0]
    assert row["arm"] == "dayhoff-170m"
    assert row["modules"]["r2_phenotypes_and_readouts"]["dtype_realised"] == "torch.float32"
    assert row["modules"]["r2_phenotypes_and_readouts"]["source"].startswith("20261121_run/")
    assert row["modules"]["r3_pairwise_and_contacts"]["dtype_realised"] == "torch.float32"
    assert row["modules"]["r3_pairwise_and_contacts"]["source"].startswith("20261121_run/")
    assert row["modules"]["r4_b1024"]["dtype_realised"] == "torch.float32"
    assert row["modules"]["r4_b1024"]["source"].startswith("20261123_run/")
    assert row["modules"]["r4_b1024"]["bands"]["id_70_90"]["mean"] == 0.995
    assert seen_npz == [older_fp32 / "dayhoff_170m" / "r3_pairwise_cycles_dayhoff-170m.npz"]
