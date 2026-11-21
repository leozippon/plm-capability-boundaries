"""Unit and 2-sequence L20 GPU smoke tests for ``evaluate_r5_structures.py``.

Run GPU smoke test with:
  CUDA_VISIBLE_DEVICES=4 runtimes/esmfold2/bin/python -m pytest -v tests/general_plm/test_evaluate_r5_structures.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.capability.entrypoints import stage_path
from scripts.capability.general_plm import evaluate_r5_structures as ev

ESMFOLD2_DIR = Path("/Data/lzp/models/ESMFold2-hf")


def _has_esmfold2_runtime() -> bool:
    try:
        from transformers import EsmFold2Model  # noqa: F401
        return True
    except ImportError:
        return False


def test_evaluate_r5_structures_entrypoint_and_normalization():
    """Verify H200 stage resolution and schema normalization across ESM3 and general-PLM JSONLs."""
    resolved = stage_path(ROOT, "evaluate_r5_structures.py")
    assert resolved.is_file() and resolved.parent.name == "general_plm"

    esm3_ladder = ev.normalize_r5_record(
        {
            "arm": "esm3-sm-open-v1",
            "backbone_id": "lb_Q13EP1",
            "condition": "bidirectional_infill",
            "rung": "k5",
            "extent": 5,
            "pll_nll_sum_nats": 120.5,
            "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLG",
        },
        "ladder_products.jsonl",
        1,
    )
    assert esm3_ladder["category"] == "ladder"
    assert esm3_ladder["model_score"] == pytest.approx(-120.5)

    esm3_cond = ev.normalize_r5_record(
        {
            "arm": "esm3-sm-open-v1",
            "backbone_id": "lb_Q13EP1",
            "condition": "cond_3d_backbone",
            "seq_only_pll_mean_nll_nats": 0.42,
            "seq_identity_to_parent": 0.78,
            "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLG",
        },
        "conditioned_products.jsonl",
        1,
    )
    assert esm3_cond["category"] == "conditioned"
    assert esm3_cond["model_score"] == pytest.approx(-0.42)

    denovo = ev.normalize_r5_record(
        {
            "id": "esm3_denovo_0001",
            "arm": "esm3-sm-open-v1",
            "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLG",
        },
        "fold_cohort.jsonl",
        1,
    )
    assert denovo["category"] == "denovo"

    assert ev._identity_band(0.0) == "lt_30"
    assert ev._identity_band(29.9) == "lt_30"
    assert ev._identity_band(30.0) == "30_70"
    assert ev._identity_band(70.0) == "70_95"
    assert ev._identity_band(95.0) == "ge_95"


def test_kabsch_rmsd_and_summary_aggregation():
    """Verify exact Kabsch RMSD under rigid rotation/translation and Spearman/strata aggregation."""
    rng = np.random.default_rng(7)
    ref = rng.normal(size=(30, 3))
    # Rigid rotation (90 deg around z) + translation
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    mobile = ref @ rot.T + np.array([5.0, -3.0, 2.0])
    assert ev._kabsch_rmsd(mobile, ref) < 1e-12

    records = [
        {
            "record_id": f"r{i}",
            "arm": "dplm-150m",
            "category": "ladder",
            "backbone_id": "bb1",
            "condition": "bidirectional_infill",
            "rung": "k10",
            "length": 50,
            "model_score": float(i),
            "mean_ca_plddt": 60.0 + 5.0 * i,
            "ptm": 0.5 + 0.05 * i,
            "tm_score_to_parent": 0.6 + 0.05 * i,
            "scrmsd_angstrom": 3.0 - 0.3 * i,
        }
        for i in range(4)
    ] + [
        {
            "record_id": "dn1",
            "arm": "dplm-150m",
            "category": "denovo",
            "length": 60,
            "mean_ca_plddt": 78.0,
            "ptm": 0.72,
            "identity_band": "lt_30",
        },
        {
            "record_id": "dn2",
            "arm": "dplm-150m",
            "category": "denovo",
            "length": 65,
            "mean_ca_plddt": 84.0,
            "ptm": 0.80,
            "identity_band": "70_95",
        },
    ]
    summary = ev.summarize_evaluations(records)
    lad = summary["r5a_ladder"]["dplm-150m|bidirectional_infill|k10"]
    assert lad["within_backbone_spearman_score_plddt"]["mean"] == pytest.approx(1.0)
    assert lad["within_backbone_spearman_score_tm"]["mean"] == pytest.approx(1.0)
    dn = summary["r5b_denovo"]["dplm-150m"]
    assert dn["uniref50_identity_strata"]["lt_30"]["count"] == 1
    assert dn["uniref50_identity_strata"]["70_95"]["count"] == 1


@pytest.mark.skipif(
    not (torch.cuda.is_available() and ESMFOLD2_DIR.is_dir() and _has_esmfold2_runtime()),
    reason="Requires CUDA GPU, /Data/lzp/models/ESMFold2-hf, and runtimes/esmfold2/bin/python",
)
def test_evaluate_r5_structures_2seq_gpu_smoke(tmp_path: Path):
    """Real 2-sequence ESMFold2-hf GPU smoke test writing strictly into tmp_path (/tmp/)."""
    in_jsonl = tmp_path / "smoke_inputs.jsonl"
    bb_jsonl = tmp_path / "smoke_backbones.jsonl"
    seq1 = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRV"
    seq2 = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRK"

    bb_jsonl.write_text(
        json.dumps({"backbone_id": "bb_smoke", "sequence": seq1}) + "\n", encoding="utf-8"
    )
    in_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "arm": "smoke-arm",
                        "backbone_id": "bb_smoke",
                        "condition": "cond_3d_backbone",
                        "seq_only_pll_mean_nll_nats": 0.5,
                        "seq_identity_to_parent": 0.975,
                        "sequence": seq2,
                    }
                ),
                json.dumps(
                    {
                        "id": "smoke_dn_0",
                        "arm": "smoke-arm",
                        "sequence": seq1,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "eval_out"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/capability/general_plm/evaluate_r5_structures.py"),
        "--device",
        "cuda:0",
        "--out",
        str(out_dir),
        "--input-jsonl",
        str(in_jsonl),
        "--backbones",
        str(bb_jsonl),
        "--esmfold2-dir",
        str(ESMFOLD2_DIR),
        "--diffusion-samples",
        "1",
        "--skip-diamond",
        "--expect-file",
        "r5_structure_eval.json",
    ]
    subprocess.run(cmd, check=True)
    result = json.loads((out_dir / "r5_structure_eval.json").read_text(encoding="utf-8"))
    assert result["schema"] == ev.SCHEMA
    assert result["n_records_folded"] == 2
    assert result["n_unique_sequences_folded"] == 2
    cond = result["r5c_conditioned"]["smoke-arm|cond_3d_backbone"]
    assert 0.0 <= cond["mean_plddt"] <= 100.0
    assert 0.0 <= cond["mean_ptm"] <= 1.0
    assert cond["mean_scrmsd_angstrom"] is not None and np.isfinite(cond["mean_scrmsd_angstrom"])
    assert cond["mean_tm_score_to_parent"] is not None and 0.0 < cond["mean_tm_score_to_parent"] <= 1.0
