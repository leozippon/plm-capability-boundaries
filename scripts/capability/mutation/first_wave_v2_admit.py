#!/usr/bin/env python3
"""Admission gates for first-wave v2 records.

Transport ADMITTED is digest evidence from pull_records_h200.sh. It is not a
scientific PASS. QUEUE_LAUNCHED is not completion. A file appearing on disk is
not completion. A single-batch longest-shape record is not the consecutive-batch
resource gate.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

LOOKUP_SHA = "1209b6c12d81c3d27020a3d87b00c253e4e70fc22f6bdaa150b8b0c9ef026ec3"
RITA_PROBE_SHA = "d574a969e188f8d7ad5bf1177ac388bac87d7ea8d3e1ed68c29caab5759c2ce6"
RITA_SCORE_SHA = "c932d3a3d9371ef5bb0a3bde4897b2804375583f06e79c2e9daa48277306e947"
WILDTYPES_JSON_SHA = "e080a37758e29c664d71e2a080f990f0bc0e8efdffbae7678b4a8b6271045faa"
WILDTYPES_FAA_SHA = "f4647ef0317278658f0a1ae9c8b8cdd9b3ce07aa19594a2cd57b4d3a621700a0"
#: Staged validated interpreter, relative to the in-pod project root. Both the
#: root and this suffix are a campaign's identity: the root is host-local, so it
#: comes from ``TRANSFER_PROJECT_ROOT`` (exported by ``h200/h200_env.sh``) and is
#: never pinned to one cluster's absolute path here. The suffix stays: the
#: admission gate is about which runtime the cells name, not about where the pod
#: mounted it.
CT_RUNTIME_SUFFIX = "runtimes/ct-20260905/bin/python"
EXPECTED_CT = {
    "python": "3.11.14",
    "torch": "2.9.1",
    "transformers": "4.57.3",
}
RESOURCE_BATCHES = 3
RESOURCE_SEQUENCES = 48
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260913
RITA_SEED_OFFSET = 1000
RITA_N_ASSAYS = 201
RITA_N_CLUSTERS = 163
RITA_EXCLUDED = 16
LOOKUP_DECLARED_ASSAYS = 217
LOOKUP_DECLARED_CLUSTERS = 174
CAMPAIGN_NAME = "campaign_first_wave_v2"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
TERMINAL_STATES = frozenset(
    {
        "exited-ok",
        "exited-ok-no-artifact",
        "exited-nonzero",
        "skipped-complete",
        "refused-busy-gpu",
    }
)


class AdmissionError(ValueError):
    """A required identity, receipt, or content gate failed."""


def validated_ct_python() -> str:
    """The staged validated interpreter, resolved against the in-pod root.

    The root is host-local, so it is read here rather than baked into this module;
    nothing is resolved at import time and there is no fallback path, because a
    wrong root would silently build a plausible-looking interpreter path.
    """

    root = os.environ.get("TRANSFER_PROJECT_ROOT")
    if not root:
        raise AdmissionError(
            "TRANSFER_PROJECT_ROOT is not set, so the validated ct interpreter cannot "
            f"resolve to <TRANSFER_PROJECT_ROOT>/{CT_RUNTIME_SUFFIX}; source "
            "h200/h200_env.sh, or set TRANSFER_PROJECT_ROOT to the in-pod project root"
        )
    return f"{root.rstrip('/')}/{CT_RUNTIME_SUFFIX}"


def require_manifest_under_snapshot(manifest: str, snapshot: str) -> None:
    """Refuse a campaign definition that is not the frozen snapshot copy."""

    snap = str(snapshot).rstrip("/")
    path = str(manifest)
    prefix = f"{snap}/scripts/capability/"
    if not path.startswith(prefix):
        raise AdmissionError(f"manifest {path} is not inside frozen {prefix}")
    relative = path[len(prefix) :]
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise AdmissionError(f"manifest {path} escapes frozen {prefix}")


def require_manifest_in_snapshot(manifest: Path, snapshot: Path) -> Path:
    """Local-path form of require_manifest_under_snapshot, used by tests."""

    manifest_r = Path(manifest).resolve()
    snapshot_r = Path(snapshot).resolve()
    require_manifest_under_snapshot(str(manifest_r), str(snapshot_r))
    if not manifest_r.is_file():
        raise AdmissionError(f"frozen manifest missing: {manifest_r}")
    return manifest_r


def parse_campaign_cells(text: str) -> list[dict[str, str]]:
    cells: list[dict[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip("\n")
        if not line or line.lstrip().startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 8:
            raise AdmissionError(f"manifest line is not 8 tab-separated fields: {line!r}")
        slot, key, gpu, stage, label, env, expect, args = fields
        cells.append(
            {
                "slot": slot,
                "key": key,
                "gpu": gpu,
                "stage": stage,
                "label": label,
                "env": env,
                "expect": expect,
                "args": args,
            }
        )
    return cells


def campaign_identity_errors(cells: Sequence[Mapping[str, str]]) -> list[str]:
    errors: list[str] = []
    if len(cells) != 5:
        errors.append(f"expected 5 cells, got {len(cells)}")
        return errors
    expected = (
        ("1", "rita_xl_independent_analyse", "native_dms_comparison.json"),
        ("2", "native_fp32_probe_galactica_125m_b16", "galactica_fp32_v2_outcome.json"),
        ("3", "native_fp32_probe_galactica_1p3b_b16", "galactica_fp32_v2_outcome.json"),
        ("4", "native_fp32_probe_galactica_6p7b_b16", "galactica_fp32_v2_outcome.json"),
        ("5", "native_fp32_probe_galactica_30b_b16", "galactica_fp32_v2_outcome.json"),
    )
    ct_python = validated_ct_python()
    gpus = []
    slots = []
    for cell, (slot, label, expect) in zip(cells, expected):
        slots.append(cell["slot"])
        gpus.append(cell["gpu"])
        if cell["slot"] != slot:
            errors.append(f"{label}: slot {cell['slot']!r} != {slot!r}")
        if cell["key"] != "v2":
            errors.append(f"{cell['label']}: key {cell['key']!r} != 'v2'")
        if cell["gpu"] != "0":
            errors.append(f"{cell['label']}: gpu {cell['gpu']!r} != '0'")
        if cell["stage"] != "native_dms_extension.py":
            errors.append(f"{cell['label']}: stage {cell['stage']!r}")
        if cell["label"] != label:
            errors.append(f"label {cell['label']!r} != {label!r}")
        if cell["expect"] != expect:
            errors.append(f"{label}: expect {cell['expect']!r} != {expect!r}")
        if f"TRANSFER_PYTHON={ct_python}" not in cell["env"]:
            errors.append(f"{label}: TRANSFER_PYTHON is not the validated ct interpreter")
        if "--device" in cell["args"] or "--out" in cell["args"]:
            errors.append(f"{label}: args must not name --device or --out")
    if slots != [row[0] for row in expected]:
        errors.append("slots must be 1-5 serial")
    if any(gpu != "0" for gpu in gpus):
        errors.append("every cell must use gpu 0")
    return errors


def parse_status(text: str) -> dict[str, Any]:
    meta: dict[str, str] = {}
    cells: dict[str, dict[str, str]] = {}
    header: list[str] | None = None
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("#"):
            body = line[1:].strip()
            if not body:
                continue
            if "\t" in body:
                key, value = body.split("\t", 1)
                meta[key.strip()] = value
            continue
        if header is None:
            header = line.split("\t")
            continue
        parts = line.split("\t")
        if len(parts) != len(header):
            raise AdmissionError("status row field count disagrees with header")
        row = dict(zip(header, parts))
        label = row.get("label")
        if not label:
            raise AdmissionError("status row missing label")
        cells[label] = row
    return {"meta": meta, "cells": cells}


def require_status_identity(
    status: Mapping[str, Any],
    *,
    snapshot: str,
    manifest_sha256: str,
) -> None:
    meta = status["meta"]
    manifest = meta.get("manifest") or ""
    if not manifest:
        raise AdmissionError("status missing # manifest")
    require_manifest_under_snapshot(manifest, snapshot)
    got = meta.get("manifest_sha256") or ""
    if not SHA_RE.fullmatch(got):
        raise AdmissionError("status missing manifest_sha256")
    if got != manifest_sha256:
        raise AdmissionError("status manifest_sha256 disagrees with the frozen blob")


def cell_receipt_errors(cell: Mapping[str, str] | None, *, label: str) -> list[str]:
    if not cell:
        return [f"{label}: missing status row"]
    errors: list[str] = []
    state = cell.get("state")
    if state != "exited-ok":
        errors.append(f"{label}: state {state!r} is not exited-ok")
    if cell.get("exit") != "0":
        errors.append(f"{label}: exit {cell.get('exit')!r} is not 0")
    artifact = cell.get("artifact") or ""
    if not artifact or artifact == "-":
        errors.append(f"{label}: missing artifact path")
    return errors


def records_admitted(pull_log: str) -> bool:
    text = pull_log or ""
    if "NOT ADMITTED" in text:
        return False
    return bool(re.search(r"\bADMITTED\b", text))


def require_records_admitted(pull_log: str, *, where: str) -> None:
    if not records_admitted(pull_log):
        raise AdmissionError(f"{where}: pull_records_h200.sh did not ADMITTED")


def ct_runtime_errors(runtime: Any) -> list[str]:
    if not isinstance(runtime, Mapping):
        return ["runtime missing"]
    errors: list[str] = []
    python = str(runtime.get("python") or "")
    torch = str(runtime.get("torch") or "")
    transformers = str(runtime.get("transformers") or "")
    if python.split()[0] != EXPECTED_CT["python"] and not python.startswith(
        EXPECTED_CT["python"]
    ):
        errors.append(f"runtime.python {python!r} is not ct {EXPECTED_CT['python']}")
    if EXPECTED_CT["torch"] not in torch:
        errors.append(f"runtime.torch {torch!r} is not ct {EXPECTED_CT['torch']}")
    if not transformers.startswith(EXPECTED_CT["transformers"]):
        errors.append(
            f"runtime.transformers {transformers!r} is not ct {EXPECTED_CT['transformers']}"
        )
    return errors


def _need(errors: list[str], cond: bool, message: str) -> None:
    if not cond:
        errors.append(message)


def rita_content_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    _need(errors, payload.get("phase") == "analyse", "phase is not analyse")
    _need(
        errors,
        payload.get("status") == "completed analysis",
        "status is not completed analysis",
    )
    groups = payload.get("groups")
    if not isinstance(groups, Mapping):
        return errors + ["groups missing"]
    _need(errors, "rita-xl" in groups, "missing rita-xl group")
    _need(errors, "galactica" not in groups, "unexpected galactica group")
    boot = payload.get("bootstrap")
    if not isinstance(boot, Mapping):
        errors.append("bootstrap missing")
    else:
        _need(errors, boot.get("resamples") == BOOTSTRAP_RESAMPLES, "bootstrap.resamples")
        _need(errors, boot.get("seed") == BOOTSTRAP_SEED, "bootstrap.seed")
        _need(
            errors,
            boot.get("rita_group_offset") == RITA_SEED_OFFSET,
            "bootstrap.rita_group_offset",
        )
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        errors.append("inputs missing")
    else:
        _need(errors, inputs.get("lookup_sha256") == LOOKUP_SHA, "lookup_sha256")
        probes = inputs.get("probes") or {}
        scores = inputs.get("scores") or {}
        _need(errors, probes.get("rita-xl") == RITA_PROBE_SHA, "probe hash")
        _need(errors, scores.get("rita-xl") == RITA_SCORE_SHA, "score hash")
        _need(errors, set(probes) == {"rita-xl"}, "analyse probes are not RITA-only")
        _need(errors, set(scores) == {"rita-xl"}, "analyse scores are not RITA-only")
    group = groups.get("rita-xl")
    if not isinstance(group, Mapping):
        return errors
    _need(errors, group.get("n_assays") == RITA_N_ASSAYS, "n_assays")
    _need(errors, group.get("n_clusters") == RITA_N_CLUSTERS, "n_clusters")
    excluded = group.get("context_excluded_assays")
    _need(
        errors,
        isinstance(excluded, list) and len(excluded) == RITA_EXCLUDED,
        "context_excluded_assays",
    )
    declared = group.get("declared_cohort") or {}
    _need(
        errors,
        declared.get("n_assays") == LOOKUP_DECLARED_ASSAYS,
        "declared_cohort.n_assays",
    )
    _need(
        errors,
        declared.get("n_clusters") == LOOKUP_DECLARED_CLUSTERS,
        "declared_cohort.n_clusters",
    )
    adjacent = (group.get("pairs") or {}).get("all_computed")
    _need(errors, adjacent == [], "unexpected adjacent pair")
    for channel in ("raw_spearman", "model_minus_lookup", "model_minus_blosum62"):
        record = ((group.get(channel) or {}).get("per_rung") or {}).get("rita-xl")
        if not isinstance(record, Mapping):
            errors.append(f"{channel} missing")
            continue
        interval = record.get("interval")
        _need(
            errors,
            isinstance(interval, list) and len(interval) == 2,
            f"{channel} interval",
        )
        adjacent_delta = (group.get(channel) or {}).get("adjacent_delta_rho")
        _need(errors, adjacent_delta == {}, f"{channel} adjacent_delta_rho")
    _need(
        errors,
        "selected_variants" not in payload and "selected_variants" not in group,
        "comparison must not invent selected_variants",
    )
    return errors


def resource_batch_errors(longest: Any) -> list[str]:
    if not isinstance(longest, Mapping):
        return ["longest_eligible_batch missing"]
    errors: list[str] = []
    n_batches = longest.get("n_batches")
    n_sequences = longest.get("n_sequences")
    finite = longest.get("finite_output_count")
    if n_batches != RESOURCE_BATCHES:
        errors.append(
            f"longest_eligible_batch.n_batches {n_batches!r} != {RESOURCE_BATCHES}"
        )
    if n_sequences != RESOURCE_SEQUENCES:
        errors.append(
            f"longest_eligible_batch.n_sequences {n_sequences!r} != {RESOURCE_SEQUENCES}"
        )
    if finite != RESOURCE_SEQUENCES:
        errors.append(
            f"longest_eligible_batch.finite_output_count {finite!r} != {RESOURCE_SEQUENCES}"
        )
    if n_batches == 1:
        errors.append("old single-batch longest-shape record is not this resource gate")
    return errors


def galactica_identity_errors(
    payload: Mapping[str, Any],
    *,
    arm: str,
    snapshot: str,
) -> list[str]:
    errors: list[str] = []
    _need(
        errors,
        payload.get("kind") == "galactica_fp32_v2_probe_not_v1_interface_gate",
        "kind",
    )
    _need(errors, payload.get("status") == "passed", "status")
    _need(errors, payload.get("probe_passed") is True, "probe_passed")
    _need(errors, payload.get("arm") == arm, "arm")
    _need(errors, payload.get("protocol_id") == "galactica-fp32-v2", "protocol_id")
    settings = payload.get("settings") or {}
    _need(errors, settings.get("dtype") == "float32", "dtype")
    _need(errors, settings.get("batch_size") == 16, "batch_size")
    _need(
        errors,
        settings.get("longest_resource_batches") == RESOURCE_BATCHES,
        "settings.longest_resource_batches",
    )
    _need(errors, settings.get("variant_seed") == 20260807, "variant_seed")
    _need(errors, settings.get("variant_cap") == 1000, "variant_cap")
    fingerprints = payload.get("fingerprints") or {}
    _need(errors, fingerprints.get("lookup_sha256") == LOOKUP_SHA, "lookup fingerprint")
    _need(
        errors,
        fingerprints.get("wildtypes_sha256") == WILDTYPES_JSON_SHA,
        "wildtypes fingerprint",
    )
    _need(
        errors,
        fingerprints.get("wildtypes_fasta_sha256") == WILDTYPES_FAA_SHA,
        "fasta fingerprint",
    )
    gate = payload.get("fp32_geometry_gate") or {}
    _need(errors, gate.get("passed") is True, "fp32_geometry_gate.passed")
    _need(errors, gate.get("n_comparisons") == 45, "fp32_geometry_gate.n_comparisons")
    errors.extend(resource_batch_errors(payload.get("longest_eligible_batch")))
    errors.extend(ct_runtime_errors(payload.get("runtime")))
    sources = payload.get("code_sources") or {}
    script = sources.get("script") or ""
    if snapshot and snapshot not in str(script):
        errors.append("code_sources.script is not the frozen snapshot")
    return errors


def rung_mean(payload: Mapping[str, Any], channel: str) -> Mapping[str, Any]:
    group = (payload.get("groups") or {}).get("rita-xl") or {}
    record = ((group.get(channel) or {}).get("per_rung") or {}).get("rita-xl")
    if not isinstance(record, Mapping):
        raise AdmissionError(f"missing rita-xl {channel}")
    return record
