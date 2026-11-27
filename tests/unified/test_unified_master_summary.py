"""Tests for ``src.capability.unified`` lazy exports and 62/65 + 81/84 master summary builder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.capability.core.io import sha256_file
from src.capability.unified import (
    DIMA_4_ARMS,
    DPLM2_4_ARMS,
    LAZY_SUBMODULES,
    PARADIGM_ORDER_80,
    PARADIGM_ORDER_84,
    PROTEINMPNN_11_ARMS,
    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
    UNIFIED_65_ARMS,
    UNIFIED_77_SINGLE_TARGET_ARMS,
    UNIFIED_80_ARMS,
    UNIFIED_81_SINGLE_TARGET_ARMS,
    UNIFIED_84_ARMS,
    master_summary,
)
from src.capability.unified.master_summary import (
    CANONICAL_ARTIFACTS,
    CANONICAL_ARTIFACTS_65,
    CANONICAL_ARTIFACTS_FULL,
    FORBIDDEN_INFRA_RE,
    SCHEMA,
    build_unified_master_summary,
    verify_sidecar,
)
from src.capability.unified.roster import REPO_ROOT, UNIFIED_OUTPUT_DIR

OUT_DIR = REPO_ROOT / UNIFIED_OUTPUT_DIR
MASTER_JSON = OUT_DIR / "unified_master_summary.json"
PROVENANCE_MD = OUT_DIR / "PROVENANCE.md"


def test_unified_package_import_is_lazy_no_eager_torch() -> None:
    """Importing ``src.capability.unified`` must not eagerly load torch or heavy submodules."""
    code = (
        "import sys, src.capability.unified as u\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'transformers' not in sys.modules\n"
        "for mod in u.LAZY_SUBMODULES:\n"
        "    assert f'src.capability.unified.{mod}' not in sys.modules\n"
        "assert len(u.UNIFIED_65_ARMS) == 65\n"
        "assert len(u.UNIFIED_62_SINGLE_SEQUENCE_ARMS) == 62\n"
        "assert len(u.UNIFIED_80_ARMS) == 80\n"
        "assert len(u.UNIFIED_77_SINGLE_TARGET_ARMS) == 77\n"
        "assert len(u.UNIFIED_84_ARMS) == 84\n"
        "assert len(u.UNIFIED_81_SINGLE_TARGET_ARMS) == 81\n"
        "assert len(u.DPLM2_4_ARMS) == 4\n"
        "assert len(u.PROTEINMPNN_11_ARMS) == 11\n"
        "assert len(u.DIMA_4_ARMS) == 4\n"
        "ms = u.master_summary\n"
        "assert ms.SCHEMA == 'unified_master_summary_v1'\n"
        "assert 'src.capability.unified.dima_adapter' not in sys.modules\n"
        "assert 'src.capability.unified.r1_yardstick' not in sys.modules\n"
        "assert 'src.capability.unified.r2_experiment_2_1' not in sys.modules\n"
        "assert 'src.capability.unified.r5_experiment_4_1' not in sys.modules\n"
        "assert 'src.capability.unified.r6_experiment_4_2' not in sys.modules\n"
        "assert 'src.capability.unified.yardstick_77_80_scoring' not in sys.modules\n"
        "assert 'src.capability.unified.yardstick_77_80_generation' not in sys.modules\n"
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Lazy import check failed: {proc.stderr}"
    assert "master_summary" in LAZY_SUBMODULES
    assert "dima_adapter" in LAZY_SUBMODULES
    assert len(UNIFIED_65_ARMS) == 65
    assert len(UNIFIED_62_SINGLE_SEQUENCE_ARMS) == 62
    assert len(UNIFIED_80_ARMS) == 80
    assert len(UNIFIED_77_SINGLE_TARGET_ARMS) == 77
    assert len(UNIFIED_84_ARMS) == 84
    assert len(UNIFIED_81_SINGLE_TARGET_ARMS) == 81
    assert len(DPLM2_4_ARMS) == 4
    assert len(PROTEINMPNN_11_ARMS) == 11
    assert len(DIMA_4_ARMS) == 4
    assert len(PARADIGM_ORDER_80) == 10
    assert len(PARADIGM_ORDER_84) == 11
    assert master_summary.SCHEMA == SCHEMA


def test_verify_sidecar_and_strict_mode_failure_paths(tmp_path: Path) -> None:
    sample = tmp_path / "dummy.json"
    sample.write_text('{"ok": true}\n', encoding="utf-8")
    digest = sha256_file(sample)
    sample.with_name("dummy.json.sha256").write_text(f"{digest}  dummy.json\n", encoding="utf-8")

    ok = verify_sidecar(sample)
    assert ok["exists"] is True and ok["verified"] is True and ok["sha256"] == digest

    sample.with_name("dummy.json.sha256").write_text("0" * 64 + "  dummy.json\n", encoding="utf-8")
    bad = verify_sidecar(sample)
    assert bad["exists"] is True and bad["verified"] is False

    with pytest.raises(RuntimeError, match="Strict master summary build refused"):
        build_unified_master_summary(root=REPO_ROOT, out_dir=tmp_path, allow_pending=False)

    # allow_pending mode succeeds and accurately records all 41 (24 core_65 + 17 expanded_84 *_full.json) artifacts
    interim = build_unified_master_summary(root=REPO_ROOT, out_dir=tmp_path, allow_pending=True)
    p = interim["payload"]
    assert p["completion_status"] == "interim_pending"
    assert len(CANONICAL_ARTIFACTS_65) == 24
    assert len(CANONICAL_ARTIFACTS_FULL) == 17
    assert p["n_artifacts_total"] == len(CANONICAL_ARTIFACTS) == 41
    assert p["n_artifacts_present"] == 0
    assert p["tier_summary"]["core_65"]["n_total"] == 24
    assert p["tier_summary"]["expanded_84"]["n_total"] == 17
    assert p["failed_reproduction_gates"] == []

    # Verify roster_universe (65), expanded_roster_universe_80 (80), and expanded_roster_universe_84 (84)
    ru84 = p["expanded_roster_universe_84"]
    assert ru84["n_total_arms"] == 84
    assert ru84["n_single_target_arms"] == 81
    assert ru84["n_msa_context_arms"] == 3
    assert ru84["dplm2_4_arms"] == list(DPLM2_4_ARMS)
    assert ru84["proteinmpnn_11_arms"] == list(PROTEINMPNN_11_ARMS)
    assert ru84["dima_4_arms"] == list(DIMA_4_ARMS)
    assert ru84["paradigm_order_84"] == list(PARADIGM_ORDER_84)
    assert sum(ru84["counts_by_paradigm_84"].values()) == 84

    assert (tmp_path / "unified_master_summary.json.sha256").is_file()
    prov_text = (tmp_path / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "Expanded 84-Model Full Tier" in prov_text
    assert "r1_proteingym_unified_full.json" in prov_text
    assert "r5a_local_infilling_full.json" in prov_text
    assert "r6_dplm2_mpnn_adaptation_unified.json" in prov_text
    assert "r5_homologous_context_merged_full.json" in prov_text
    assert "r4_saprot_3di_aa_target_causal_control.json" in prov_text
    assert not FORBIDDEN_INFRA_RE.search(prov_text)


def test_failed_reproduction_gate_blocks_strict_and_marks_status(tmp_path: Path) -> None:
    """A failed reproduction/verification gate must fail strict mode and set failed_reproduction_gates."""
    sap_p = tmp_path / "r4_saprot_3di_aa_target_causal_control.json"
    sap_p.write_text(
        json.dumps(
            {
                "schema": "r4_saprot_3di_aa_target_causal_control_v1",
                "completion_status": "complete",
                "models": {"saprot-35m": {}},
                "verification_gate": {"passed": False, "max_abs_diff": 0.25},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sap_p.with_name(sap_p.name + ".sha256").write_text(f"{sha256_file(sap_p)}  {sap_p.name}\n", encoding="utf-8")

    # Strict mode must explicitly refuse with failed reproduction gate error
    with pytest.raises(RuntimeError, match="failed reproduction gates"):
        build_unified_master_summary(root=REPO_ROOT, out_dir=tmp_path, allow_pending=False)

    # allow_pending mode must never claim complete or interim_pending when a gate failed
    res = build_unified_master_summary(root=REPO_ROOT, out_dir=tmp_path, allow_pending=True)
    p = res["payload"]
    assert p["completion_status"] == "failed_reproduction_gates"
    assert p["all_reproduction_gates_passed"] is False
    assert p["failed_reproduction_gates"] == ["r4_saprot_3di_aa_target_causal_control"]
    assert "saprot_3di_aa_target_causal_control" in p["results_summary"]["result_4_generation_memory_and_alignment"]
    prov_txt = (tmp_path / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "FAILED" in prov_txt
    assert "r4_saprot_3di_aa_target_causal_control" in prov_txt


def test_expanded_84_completion_detection_in_tmp_dir(tmp_path: Path) -> None:
    """Verify expanded_84 *_full.json artifacts enforce both completion_status and coverage.is_complete."""
    p_full = tmp_path / "r5a_local_infilling_full.json"
    p_full.write_text(
        json.dumps({"schema": "r5a_local_infilling_full_v1", "coverage": {"is_complete": False}}) + "\n",
        encoding="utf-8",
    )
    p_full.with_name(p_full.name + ".sha256").write_text(f"{sha256_file(p_full)}  {p_full.name}\n", encoding="utf-8")

    p_ct = tmp_path / "contacts_unified_full.json"
    p_ct.write_text(
        json.dumps(
            {
                "schema": "contacts_unified_full_v1",
                "completion_status": "complete",
                "n_arms_primary": 81,
                "n_arms_with_struct_companions": 85,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    p_ct.with_name(p_ct.name + ".sha256").write_text(f"{sha256_file(p_ct)}  {p_ct.name}\n", encoding="utf-8")

    res = build_unified_master_summary(root=REPO_ROOT, out_dir=tmp_path, allow_pending=True)
    by_name = {m["filename"]: m for m in res["payload"]["artifact_manifest"]}
    assert by_name["r5a_local_infilling_full.json"]["status"] == "partial_pending_h200"
    assert by_name["contacts_unified_full.json"]["status"] == "complete"
    assert res["payload"]["tier_summary"]["expanded_84"]["n_present"] == 2
    assert res["payload"]["tier_summary"]["expanded_84"]["n_complete"] == 1
    assert "contacts_unified_full.json" in res["payload"]["results_summary"]["expanded_81_84_tier"]


@pytest.mark.skipif(not OUT_DIR.is_dir(), reason="unified_yardstick_20261124 directory absent")
def test_local_unified_directory_sidecars_and_master_build(tmp_path: Path) -> None:
    """Verify every existing canonical artifact sidecar on disk and exercise full extraction."""
    n_present = 0
    for spec in CANONICAL_ARTIFACTS:
        fpath = OUT_DIR / spec.filename
        if fpath.is_file():
            n_present += 1
            res = verify_sidecar(fpath)
            assert res["verified"] is True, f"Sidecar mismatch for {spec.filename}"
            (tmp_path / spec.filename).symlink_to(fpath)
            (tmp_path / f"{spec.filename}.sha256").symlink_to(fpath.with_name(f"{spec.filename}.sha256"))

    res_build = build_unified_master_summary(root=REPO_ROOT, out_dir=tmp_path, allow_pending=True)
    payload = res_build["payload"]
    assert payload["schema"] == SCHEMA
    assert payload["n_artifacts_present"] == n_present
    assert payload["n_sidecars_verified"] == n_present
    assert payload["all_reproduction_gates_passed"] is True
    assert payload["expanded_roster_universe_84"]["n_total_arms"] == 84
    assert not FORBIDDEN_INFRA_RE.search((tmp_path / "PROVENANCE.md").read_text(encoding="utf-8"))
    assert not FORBIDDEN_INFRA_RE.search(
        (tmp_path / "unified_master_summary.json").read_text(encoding="utf-8")
    )

    if MASTER_JSON.is_file():
        res = verify_sidecar(MASTER_JSON)
        assert res["verified"] is True
        doc = json.loads(MASTER_JSON.read_text(encoding="utf-8"))
        assert doc["schema"] == SCHEMA
        assert doc["all_reproduction_gates_passed"] is True
        assert PROVENANCE_MD.is_file()
        prov_txt = PROVENANCE_MD.read_text(encoding="utf-8")
        assert not FORBIDDEN_INFRA_RE.search(prov_txt)
        assert not FORBIDDEN_INFRA_RE.search(MASTER_JSON.read_text(encoding="utf-8"))
