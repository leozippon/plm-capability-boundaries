"""Tests for ``src.capability.unified`` lazy exports and master summary / provenance builder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.capability.core.io import sha256_file
from src.capability.unified import (
    LAZY_SUBMODULES,
    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
    UNIFIED_65_ARMS,
    master_summary,
)
from src.capability.unified.master_summary import (
    CANONICAL_ARTIFACTS,
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
        "ms = u.master_summary\n"
        "assert ms.SCHEMA == 'unified_master_summary_v1'\n"
        "assert 'src.capability.unified.r1_yardstick' not in sys.modules\n"
        "assert 'src.capability.unified.r2_experiment_2_1' not in sys.modules\n"
        "assert 'src.capability.unified.r5_experiment_4_1' not in sys.modules\n"
        "assert 'src.capability.unified.r6_experiment_4_2' not in sys.modules\n"
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
    assert len(UNIFIED_65_ARMS) == 65
    assert len(UNIFIED_62_SINGLE_SEQUENCE_ARMS) == 62
    assert master_summary.SCHEMA == SCHEMA


def test_verify_sidecar_and_strict_mode_failure_paths(tmp_path: Path) -> None:
    sample = tmp_path / "dummy.json"
    sample.write_text('{"ok": true}\n', encoding="utf-8")
    digest = sha256_file(sample)
    sample.with_name("dummy.json.sha256").write_text(f"{digest}  dummy.json\n", encoding="utf-8")

    ok = verify_sidecar(sample)
    assert ok["exists"] is True and ok["verified"] is True and ok["sha256"] == digest

    # Tampered sidecar must fail verification
    sample.with_name("dummy.json.sha256").write_text("0" * 64 + "  dummy.json\n", encoding="utf-8")
    bad = verify_sidecar(sample)
    assert bad["exists"] is True and bad["verified"] is False

    # Strict mode on empty directory must raise RuntimeError (never silently fake completion)
    with pytest.raises(RuntimeError, match="Strict master summary build refused"):
        build_unified_master_summary(root=REPO_ROOT, out_dir=tmp_path, allow_pending=False)

    # allow_pending mode succeeds and accurately records all 17 artifacts as missing
    interim = build_unified_master_summary(root=REPO_ROOT, out_dir=tmp_path, allow_pending=True)
    p = interim["payload"]
    assert p["completion_status"] == "interim_pending"
    assert p["n_artifacts_total"] == len(CANONICAL_ARTIFACTS) == 17
    assert p["n_artifacts_present"] == 0
    assert (tmp_path / "unified_master_summary.json.sha256").is_file()
    assert (tmp_path / "PROVENANCE.md").is_file()
    assert not FORBIDDEN_INFRA_RE.search((tmp_path / "PROVENANCE.md").read_text(encoding="utf-8"))


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

    # Exercise end-to-end extraction against real on-disk artifacts without mutating OUT_DIR.
    res_build = build_unified_master_summary(root=REPO_ROOT, out_dir=tmp_path, allow_pending=True)
    payload = res_build["payload"]
    assert payload["schema"] == SCHEMA
    assert payload["n_artifacts_present"] == n_present
    assert payload["n_sidecars_verified"] == n_present
    assert payload["all_reproduction_gates_passed"] is True
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
