"""Negative gates for first-wave v2 freeze identity and record admission."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
from scripts.capability.campaigns import ARCHIVE, records


def _load_admit():
    path = REPO / "scripts/capability/mutation/first_wave_v2_admit.py"
    spec = importlib.util.spec_from_file_location("first_wave_v2_admit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


admit = _load_admit()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_tracked_manifest_is_five_serial_gpu0_cells() -> None:
    if not ARCHIVE.is_file():
        pytest.skip("the historical recipe archive is host-local under h200/ and is not distributed")
    if not os.environ.get("TRANSFER_PROJECT_ROOT"):
        pytest.skip(
            "the archived recipe names the frozen project root in its interpreter column; "
            "set TRANSFER_PROJECT_ROOT, as h200/h200_env.sh does, to re-run this identity gate"
        )
    cells = admit.parse_campaign_cells(records()["campaign_first_wave_v2.tsv"]["content"])
    assert admit.campaign_identity_errors(cells) == []
    assert [cell["slot"] for cell in cells] == ["1", "2", "3", "4", "5"]
    assert {cell["gpu"] for cell in cells} == {"0"}
    assert "analyse" in cells[0]["args"]
    assert "rita-xl=" in cells[0]["args"]
    assert all("galactica-fp32-v2" in cell["args"] for cell in cells[1:])


def test_remote_manifest_string_must_stay_under_snapshot() -> None:
    snap = "/cluster/project/packages/run"
    admit.require_manifest_under_snapshot(
        f"{snap}/scripts/capability/campaign_first_wave_v2.tsv", snap
    )
    with pytest.raises(admit.AdmissionError, match="not inside frozen"):
        admit.require_manifest_under_snapshot(
            "/cluster/project/logs/campaign_first_wave_v2.tsv",
            snap,
        )


def test_manifest_outside_frozen_snapshot_is_refused(tmp_path: Path) -> None:
    snapshot = tmp_path / "packages" / "run"
    inside = snapshot / "scripts" / "capability"
    inside.mkdir(parents=True)
    frozen = inside / "campaign_first_wave_v2.tsv"
    frozen.write_text("ok\n", encoding="utf-8")
    outside = tmp_path / "logs" / "campaign_first_wave_v2.tsv"
    outside.parent.mkdir(parents=True)
    outside.write_text("mutable\n", encoding="utf-8")
    with pytest.raises(admit.AdmissionError, match="not inside frozen"):
        admit.require_manifest_in_snapshot(outside, snapshot)
    assert admit.require_manifest_in_snapshot(frozen, snapshot) == frozen.resolve()


def test_status_with_wrong_manifest_digest_is_refused(tmp_path: Path) -> None:
    snapshot = tmp_path / "snap"
    manifest = snapshot / 'scripts/capability/campaign_first_wave_v2.tsv'
    manifest.parent.mkdir(parents=True)
    manifest.write_text("cell\n", encoding="utf-8")
    digest = _sha(manifest.read_bytes())
    status = admit.parse_status(
        "\n".join(
            [
                f"# manifest\t{manifest}",
                f"# manifest_sha256\t{digest}",
                "slot\tlabel\tgpu\tstate\texit\tstarted_utc\tended_utc\tartifact\tlog",
                "1\trita_xl_independent_analyse\t0\texited-ok\t0\t-\t-\t/a.json\t/a.log",
            ]
        )
        + "\n"
    )
    admit.require_status_identity(
        status, snapshot=str(snapshot), manifest_sha256=digest
    )
    with pytest.raises(admit.AdmissionError, match="manifest_sha256 disagrees"):
        admit.require_status_identity(
            status, snapshot=str(snapshot), manifest_sha256="0" * 64
        )


def test_missing_receipt_and_nonzero_exit_are_refused() -> None:
    missing = admit.cell_receipt_errors(None, label="rita_xl_independent_analyse")
    assert missing == ["rita_xl_independent_analyse: missing status row"]
    running = admit.cell_receipt_errors(
        {
            "state": "running",
            "exit": "-",
            "artifact": "-",
        },
        label="native_fp32_probe_galactica_125m_b16",
    )
    assert any("exited-ok" in item for item in running)
    failed = admit.cell_receipt_errors(
        {
            "state": "exited-ok",
            "exit": "1",
            "artifact": "/tmp/out.json",
        },
        label="x",
    )
    assert any("exit" in item for item in failed)


def test_pull_log_without_admitted_is_refused() -> None:
    with pytest.raises(admit.AdmissionError, match="did not ADMITTED"):
        admit.require_records_admitted("digests verified; copied 1 file", where="rita")
    with pytest.raises(admit.AdmissionError, match="did not ADMITTED"):
        admit.require_records_admitted(
            "digest mismatch between pod and B; NOT ADMITTED: /tmp/out",
            where="rita",
        )
    admit.require_records_admitted(
        "[pull-records] digests verified; /tmp/out ADMITTED (1 file(s))",
        where="rita",
    )


def test_wrong_cohort_and_wrong_digest_refuse_rita() -> None:
    payload = {
        "phase": "analyse",
        "status": "completed analysis",
        "bootstrap": {
            "resamples": 2000,
            "seed": 20260913,
            "rita_group_offset": 1000,
        },
        "inputs": {
            "lookup_sha256": admit.LOOKUP_SHA,
            "probes": {"rita-xl": admit.RITA_PROBE_SHA},
            "scores": {"rita-xl": admit.RITA_SCORE_SHA},
        },
        "groups": {
            "rita-xl": {
                "n_assays": 201,
                "n_clusters": 163,
                "context_excluded_assays": [f"a{i}" for i in range(16)],
                "declared_cohort": {"n_assays": 217, "n_clusters": 174},
                "pairs": {"all_computed": []},
                "raw_spearman": {
                    "per_rung": {"rita-xl": {"mean": 0.3, "interval": [0.2, 0.4]}},
                    "adjacent_delta_rho": {},
                },
                "model_minus_lookup": {
                    "per_rung": {"rita-xl": {"mean": 0.0, "interval": [-0.1, 0.1]}},
                    "adjacent_delta_rho": {},
                },
                "model_minus_blosum62": {
                    "per_rung": {"rita-xl": {"mean": 0.1, "interval": [0.0, 0.2]}},
                    "adjacent_delta_rho": {},
                },
            }
        },
    }
    assert admit.rita_content_errors(payload) == []
    bad_hash = dict(payload)
    bad_hash["inputs"] = dict(payload["inputs"])
    bad_hash["inputs"]["scores"] = {"rita-xl": "ab" * 32}
    assert any("score hash" in item for item in admit.rita_content_errors(bad_hash))
    mixed = dict(payload)
    mixed["groups"] = dict(payload["groups"])
    mixed["groups"]["galactica"] = {"n_assays": 213}
    assert any("galactica" in item for item in admit.rita_content_errors(mixed))
    wrong_n = dict(payload)
    wrong_n["groups"] = {"rita-xl": dict(payload["groups"]["rita-xl"])}
    wrong_n["groups"]["rita-xl"]["n_assays"] = 217
    assert any("n_assays" in item for item in admit.rita_content_errors(wrong_n))


def test_old_single_batch_longest_shape_is_not_the_resource_gate() -> None:
    single = {
        "n_batches": 1,
        "n_sequences": 16,
        "finite_output_count": 16,
    }
    errors = admit.resource_batch_errors(single)
    assert any("n_batches" in item for item in errors)
    assert any("single-batch" in item for item in errors)
    triple = {
        "n_batches": 3,
        "n_sequences": 48,
        "finite_output_count": 48,
    }
    assert admit.resource_batch_errors(triple) == []
