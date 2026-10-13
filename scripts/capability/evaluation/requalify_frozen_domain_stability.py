#!/usr/bin/env python3
"""Requalify the unchanged frozen small-domain stability evaluator on the current table.

This is a **separate current-source qualification**, not a recovery of the
historical 678e8b... table (which is absent) and not a retraining. It answers one
question only: on the explicitly versioned current validation table, does the
byte-identical frozen E14 head still pass its own two declared conditions on the
eligible non-target transfer rows?

The boundary is the whole point, so it is enforced in two subcommands:

``prepare --protocol PATH --root INPUT_ROOT --out FRESH_OUT``
    hash every pinned input (including the full embedding array), validate the
    saved index against the current rows, resolve the twelve reviewed targets and
    their complete WT clusters from genotype/panel/WT metadata, exclude those
    clusters and any exact fit-sequence overlap, and freeze the ordered
    eligible-ID allowlist. **No measured value is read at all here.** The result
    is a *preparation*, not a passing qualification.

``evaluate --protocol PATH --root ROOT --prepared DIRECTORY --prepared-sha256 EXACT --device cpu --out FRESH_OUT``
    re-verify the protocol, the preparation checksum and every input, then read
    ``measured_delta_g`` **only** for IDs on the frozen allowlist. Apply the saved
    coefficients with no fitting or alpha selection, run the existing
    :func:`domain_stability.transfer_contrast` and
    :func:`domain_stability.evaluate_gate` with the protocol's own seed and draw
    count, and report pass/fail. A failed gate is a complete qualification whose
    pilot is blocked -- not a biological absence.

CPU only, and the stage runs with CUDA hidden so no backend can fall back to a
GPU. ``--device`` exists because the campaign queue injects it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.evaluation import domain_stability as ds  # noqa: E402
from src.capability.evaluation import generated_phenotype as gp  # noqa: E402

PROTOCOL_SCHEMA = "licensed_evaluator_requalification_protocol_v1"
PREPARATION_SCHEMA = "frozen_domain_stability_requalification_preparation_v1"
EVALUATION_SCHEMA = "frozen_domain_stability_requalification_v1"
PREPARATION_COMPLETION = "preparation.json"
EVALUATION_COMPLETION = "frozen_evaluator_requalification.json"
RUN_COMPLETION = "evaluator_requalification_run.json"
ELIGIBLE_ROWS = "eligible_rows.jsonl"
ELIGIBLE_SPLIT = "transfer"

#: The frozen evaluator's licensed band, declared by the protocol's eligible
#: population and cross-checked against the pinned fit report's own licensed band.
ELIGIBLE_BAND: tuple[int, int] = (60, 80)

#: The MGnify release folds the fit consumed. An ``mg_`` row outside them is an
#: unknown split and fails rather than being counted.
USED_SPLITS: tuple[str, ...] = ("train", "validation", "validation_online", "test")

FIT_DIR_REL = "results/external_baseline/20261008101801_33265fa60eac/stabdg_fit_and_validate_cpu"
EMBED_DIR_REL = "results/external_baseline/20261008101801_33265fa60eac/stabdg_embed_esm2_gpu2"
ROWS_REL = "results/R6/domain_stability_20261008/domain_stability_rows.jsonl"
COHORT_REL = "results/R4/gate_stability_20260924/cohort/cohort.json"
ROLES_REL = "data/r6_constraint_response_20261013/diagnostic_protocol.json"

#: input name -> (protocol mapping, protocol key). ``embedding_index`` is absent
#: on purpose: the protocol pins no producer-time index digest, so it is hashed
#: and frozen here for the first time.
PIN_KEYS: dict[str, tuple[str, str]] = {
    "rows": ("source_version", "current_rows_sha256"),
    "fit_report": ("pins", "fit_report_sha256"),
    "plm_head": ("pins", "plm_head_sha256"),
    "baseline_head": ("pins", "baseline_head_sha256"),
    "embedding_metadata": ("pins", "embedding_metadata_sha256"),
    "embeddings": ("pins", "embeddings_sha256"),
    "cohort": ("pins", "reviewed_cohort_sha256"),
    "roles": ("pins", "reviewed_roles_protocol_sha256"),
}

_ID_PEEK = re.compile(r'(?:^|[,{])\s*"id"\s*:\s*"((?:[^"\\]|\\.)*)"')


class Refusal(RuntimeError):
    """An explicit, non-negotiable refusal. Never a silent fallback."""


# ------------------------------------------------------------------ primitives


def _digest16(sequence: str) -> str:
    """The identifier suffix ``build_domain_stability_tables`` uses for mg_/ms_ ids."""

    return hashlib.blake2b(sequence.encode("utf-8"), digest_size=16).hexdigest()


def _text_digest(parts: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _npy_data_offset(path: Path) -> int:
    """Byte offset of the array data, read from the ``.npy`` header itself."""

    with Path(path).open("rb") as handle:
        magic = handle.read(8)
        if magic[:6] != b"\x93NUMPY":
            raise Refusal(f"{path} is not a .npy file")
        major = magic[6]
        if major == 1:
            header = int.from_bytes(handle.read(2), "little")
            base = 10
        elif major in (2, 3):
            header = int.from_bytes(handle.read(4), "little")
            base = 12
        else:
            raise Refusal(f"{path} uses an unknown .npy version {major}")
    return base + header


def input_paths(root: Path) -> dict[str, Path]:
    root = Path(root)
    return {
        "rows": root / ROWS_REL,
        "fit_report": root / FIT_DIR_REL / "domain_stability_fit.json",
        "plm_head": root / FIT_DIR_REL / "model_plm.npz",
        "baseline_head": root / FIT_DIR_REL / "model_baseline.npz",
        "embedding_metadata": root / EMBED_DIR_REL / "stability_embeddings.json",
        "embedding_index": root / EMBED_DIR_REL / "embedding_index.json",
        "embeddings": root / EMBED_DIR_REL / "embeddings.npy",
        "cohort": root / COHORT_REL,
        "roles": root / ROLES_REL,
    }


def load_protocol(path: Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != PROTOCOL_SCHEMA:
        raise Refusal(f"{path} is not a {PROTOCOL_SCHEMA} document")
    source = document.get("source_version") or {}
    if source.get("historical_rows_recovered") is not False:
        raise Refusal("the protocol must declare historical_rows_recovered: false")
    for section, key in PIN_KEYS.values():
        if not (document.get(section) or {}).get(key):
            raise Refusal(f"the protocol is missing {section}.{key}")
    reviewed = document.get("preparation", {}).get("reviewed_targets") or []
    if len(reviewed) != 12 or len(set(reviewed)) != 12:
        raise Refusal(f"the protocol must name twelve distinct reviewed targets, got {len(reviewed)}")
    validation = document.get("validation") or {}
    for key in ("seed", "bootstrap_draws", "G1", "G2", "joint_rule", "implementation"):
        if key not in validation:
            raise Refusal(f"the protocol is missing validation.{key}")
    if not document.get("limits"):
        raise Refusal("the protocol must carry its limits")
    return document


def _pin(protocol: dict[str, Any], name: str) -> str:
    section, key = PIN_KEYS[name]
    return str(protocol[section][key])


def hash_inputs(
    paths: dict[str, Path], protocol: dict[str, Any], recorded: dict[str, Any] | None
) -> dict[str, Any]:
    """Hash every input and check it against the protocol pin and any prior record."""

    seen: set[Path] = set()
    report: dict[str, Any] = {}
    for name, path in paths.items():
        path = Path(path)
        if not path.is_file():
            raise Refusal(f"{name} is missing at {path}")
        digest = sha256_file(path)
        entry = {"path": str(path), "bytes": int(path.stat().st_size), "sha256": digest}
        if name in PIN_KEYS:
            expected = _pin(protocol, name)
            if digest != expected:
                raise Refusal(f"{name} digest {digest} does not match the protocol pin {expected}")
            entry["protocol_pin"] = expected
            entry["matches_protocol_pin"] = True
        if recorded is not None:
            prior = recorded.get(name)
            if prior is None:
                raise Refusal(f"{name} is absent from the preparation input record")
            if prior.get("sha256") != digest:
                raise Refusal(
                    f"{name} digest changed since preparation: now {digest}, prepared {prior.get('sha256')}"
                )
        if path in seen:
            raise Refusal(f"input {name} resolves to the same file as another input: {path}")
        seen.add(path)
        report[name] = entry
    return report


def _resolved_paths(root: Path, protocol: dict[str, Any]) -> dict[str, Path]:
    paths = input_paths(root)
    roles = json.loads(paths["roles"].read_text(encoding="utf-8"))
    reviewed = list(protocol["preparation"]["reviewed_targets"])
    declared = list((roles.get("roles") or {}).get("development", [])) + list(
        (roles.get("roles") or {}).get("evaluation", [])
    )
    if sorted(declared) != sorted(reviewed):
        raise Refusal(
            "the protocol's reviewed targets and the reviewed roles protocol disagree: "
            f"{sorted(reviewed)} vs {sorted(declared)}"
        )
    return paths


# ---------------------------------------------------------------- table reading


def _without_outcome(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Decode a table object with the measured value dropped before it is stored.

    The outcome is removed by the object hook rather than popped afterwards, so no
    downstream metadata structure can carry a measured value even accidentally.
    """

    return {key: value for key, value in pairs if key != "measured_delta_g"}


def read_table_rows(rows_path: Path) -> list[dict[str, Any]]:
    """Every table row's metadata, with the measured value dropped at decode time."""

    rows: list[dict[str, Any]] = []
    with Path(rows_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line, object_pairs_hook=_without_outcome))
    if not rows:
        raise Refusal(f"{rows_path} carries no row")
    return rows


def _peek_id(line: str) -> str | None:
    match = _ID_PEEK.search(line)
    return match.group(1) if match else None


def _validate_row_identity(row: dict[str, Any]) -> None:
    sequence = str(row.get("sequence", ""))
    identifier = str(row.get("id", ""))
    if not identifier:
        raise Refusal("a table row carries no id")
    if not sequence or sequence != sequence.upper() or set(sequence) - set(ds.AA20):
        raise Refusal(f"{identifier}: non-canonical modelling sequence")
    if identifier[:3] in ("mg_", "ms_"):
        if identifier[3:] != _digest16(sequence):
            raise Refusal(
                f"{identifier}: the id suffix is not blake2b16(sequence); the row id does not "
                "identify this sequence"
            )
        expected = {"mg_": set(USED_SPLITS), "ms_": {ELIGIBLE_SPLIT}}[identifier[:3]]
        if str(row.get("split")) not in expected:
            raise Refusal(f"{identifier}: unknown split {row.get('split')!r} for this id namespace")
    elif identifier.startswith(("mg_", "ms_")):
        raise Refusal(f"{identifier}: malformed identifier namespace")
    elif str(row.get("split")) != "apply":
        raise Refusal(f"{identifier}: a non-hash id is only expected under the apply split")


def _validate_embedding_contract(
    index_path: Path,
    array_path: Path,
    metadata_path: Path,
    protocol: dict[str, Any],
    *,
    array_sha256: str,
) -> dict[str, Any]:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    historical = str(protocol["source_version"]["historical_rows_sha256"])
    if metadata.get("rows_sha256") != historical:
        raise Refusal(
            f"{metadata_path} declares rows_sha256 {metadata.get('rows_sha256')}, not the "
            f"historical {historical}; the saved metadata is not being relabelled"
        )
    if metadata.get("embeddings_sha256") != protocol["pins"]["embeddings_sha256"]:
        raise Refusal("the saved embedding metadata declares a different array digest than the protocol")
    if metadata.get("dtype_stored") != "float16":
        raise Refusal(f"embedding metadata dtype_stored is {metadata.get('dtype_stored')!r}, not float16")
    if metadata.get("dtype_compute") != "bfloat16":
        raise Refusal(
            f"embedding metadata dtype_compute is {metadata.get('dtype_compute')!r}, not the "
            "declared bfloat16"
        )
    if metadata.get("pooler_instantiated") is not False:
        raise Refusal("the embedding metadata does not declare pooler_instantiated: false")
    if "residue positions" not in str(metadata.get("pooling", "")):
        raise Refusal("the embedding pooling contract does not say it excludes special tokens")
    width = int(metadata.get("width", -1))
    if width < 1:
        raise Refusal("embedding metadata carries no usable width")

    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    ids = [str(value) for value in index.get("ids", [])]
    splits = [str(value) for value in index.get("splits", [])]
    lengths = [int(value) for value in index.get("lengths", [])]
    if not (len(ids) == len(splits) == len(lengths)) or not ids:
        raise Refusal("the embedding index arrays disagree on length")
    if len(set(ids)) != len(ids):
        raise Refusal("the embedding index carries duplicate identifiers")
    if int(metadata.get("n_rows", -1)) != len(ids):
        raise Refusal("embedding metadata n_rows and index length disagree")

    array = np.load(array_path, mmap_mode="r")
    if array.dtype != np.float16:
        raise Refusal(f"{array_path} stores {array.dtype}, not float16")
    if array.ndim != 2 or array.shape[1] != width:
        raise Refusal(f"{array_path} has shape {array.shape}, not (n, {width})")
    if array.shape[0] != len(ids):
        raise Refusal("the embedding array and index disagree on row count")
    data_bytes = Path(array_path).stat().st_size - _npy_data_offset(array_path)
    expected_bytes = len(ids) * width * 2
    if data_bytes != expected_bytes:
        raise Refusal(
            f"{array_path} holds {data_bytes} data bytes, not the {expected_bytes} implied by "
            f"{len(ids)} rows x {width} x float16"
        )
    return {
        "metadata": metadata,
        "ids": ids,
        "splits": splits,
        "lengths": lengths,
        "width": width,
        "n_rows": len(ids),
        "index_sha256": sha256_file(index_path),
        "index_bytes": int(Path(index_path).stat().st_size),
        "array_data_bytes": int(data_bytes),
        "array_sha256": array_sha256,
        "recorded_historical_rows_sha256": historical,
    }


def validate_rows_against_index(
    rows: list[dict[str, Any]], contract: dict[str, Any]
) -> dict[str, Any]:
    """Per-row identity coverage between the current rows and the saved index.

    Every saved-index row must appear exactly once in the current table with the
    same length and split. The only rows permitted to have no embedding are
    explicit ``split == "apply"`` application records from the ``apply`` table with
    a generic (non ``mg_/ms_``) identifier: the current application table grew
    after the frozen embedding run. Those rows are counted and listed, never enter
    the embedding mapping or the eligible population, and cannot excuse a missing
    MGnify, transfer or indexed-apply row.
    """

    ids = contract["ids"]
    position = {identifier: i for i, identifier in enumerate(ids)}
    by_split: dict[str, int] = {}
    indexed_by_split: dict[str, int] = {}
    unindexed_apply_ids: list[str] = []
    unindexed_apply_by_split: dict[str, int] = {}
    seen: set[str] = set()
    for row in rows:
        _validate_row_identity(row)
        identifier = str(row["id"])
        if identifier in seen:
            raise Refusal(f"duplicate table identifier {identifier}")
        seen.add(identifier)
        by_split[str(row["split"])] = by_split.get(str(row["split"]), 0) + 1
        index = position.get(identifier)
        if index is None:
            if (
                identifier[:3] in ("mg_", "ms_")
                or str(row["split"]) != "apply"
                or str(row.get("table")) != "apply"
            ):
                raise Refusal(
                    f"{identifier}: no embedding row in the saved index and it is not an explicit "
                    "apply-table application record; only unindexed apply rows are permitted"
                )
            unindexed_apply_ids.append(identifier)
            unindexed_apply_by_split["apply"] = unindexed_apply_by_split.get("apply", 0) + 1
            continue
        if contract["ids"][index] != identifier:
            raise Refusal(f"{identifier}: index position {index} does not map back to the id")
        if contract["lengths"][index] != int(row["length"]):
            raise Refusal(
                f"{identifier}: row length {row['length']} disagrees with the saved index "
                f"length {contract['lengths'][index]}"
            )
        if contract["splits"][index] != str(row["split"]):
            raise Refusal(
                f"{identifier}: row split {row['split']!r} disagrees with the saved index "
                f"split {contract['splits'][index]!r}"
            )
        indexed_by_split[str(row["split"])] = indexed_by_split.get(str(row["split"]), 0) + 1
    missing = [identifier for identifier in ids if identifier not in seen]
    if missing:
        raise Refusal(
            f"the saved index carries {len(missing)} identifiers with no current row "
            f"(first {missing[:3]}); an indexed row may not go missing"
        )
    contract["position"] = position
    contract["by_split"] = by_split
    contract["indexed_by_split"] = indexed_by_split
    contract["unindexed_apply_ids"] = unindexed_apply_ids
    contract["unindexed_apply_by_split"] = unindexed_apply_by_split
    contract["unindexed_apply_count"] = len(unindexed_apply_ids)
    return contract


# --------------------------------------------------------------------- targets


def bind_targets(
    transfer_rows: list[dict[str, Any]],
    backgrounds: list[dict[str, Any]],
    reviewed: list[str],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Resolve the twelve reviewed targets to complete transfer WT clusters, or refuse.

    Only genotype/panel/WT metadata is consulted: the cohort's panel sequence and
    cluster label, and the transfer row's ``wild_type_name`` and ``group``. A
    missing, ambiguous or inconsistent link is a blocker -- absence of a link is
    never read as absence of the target.
    """

    by_name = {str(background["name"]): background for background in backgrounds}
    resolution: list[dict[str, Any]] = []
    excluded: set[str] = set()
    for target in reviewed:
        background = by_name.get(target)
        if background is None:
            raise Refusal(
                f"reviewed target {target!r} has no cohort panel/WT metadata; an unlinked target "
                "is a blocker, not presumed absent"
            )
        sequences = [str(value).upper() for value in background.get("sequences") or []]
        if not sequences or set(sequences[0]) - set(ds.AA20):
            raise Refusal(f"reviewed target {target!r} carries no usable panel/WT sequence")
        panel = sequences[0]
        members = [
            row
            for row in transfer_rows
            if str(row.get("wild_type_name", "")).split("_")[0] == target
        ]
        if not members:
            raise Refusal(f"reviewed target {target!r} has no transfer genotype row")
        wild_types = [row for row in members if row.get("is_wild_type")]
        if len(wild_types) != 1:
            raise Refusal(
                f"reviewed target {target!r}: expected exactly one WT row, found {len(wild_types)}"
            )
        wild_type = wild_types[0]
        if wild_type["sequence"] != panel or int(wild_type["length"]) != len(panel):
            raise Refusal(
                f"reviewed target {target!r}: the transfer WT sequence does not match the cohort "
                "panel sequence; the raw WT-name link is not consistent"
            )
        member_groups = {str(row["group"]) for row in members}
        if len(member_groups) != 1:
            raise Refusal(
                f"reviewed target {target!r}: rows resolve to multiple WT clusters {sorted(member_groups)}"
            )
        group = member_groups.pop()
        cohort_cluster = str(background.get("cluster"))
        if cohort_cluster != group:
            raise Refusal(
                f"reviewed target {target!r}: cohort cluster {cohort_cluster!r} is inconsistent with "
                f"the transfer group {group!r}; alias groups are not guessed"
            )
        excluded.add(group)
        variants = sum(1 for row in members if not row.get("is_wild_type"))
        resolution.append(
            {
                "target": target,
                "cohort_group": str(background.get("group")),
                "cohort_cluster": cohort_cluster,
                "transfer_group": group,
                "wild_type_id": str(wild_type["id"]),
                "wild_type_sequence_digest": _digest16(panel),
                "n_cluster_rows": len(members),
                "n_variant_rows": variants,
                "linkage": "cohort panel sequence == transfer WT sequence and cohort cluster == transfer group",
            }
        )
    return resolution, excluded


def select_eligible(
    transfer_rows: list[dict[str, Any]],
    excluded_groups: set[str],
    fit_suffixes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The eligible population: in band, outside every target cluster and fit overlap."""

    low, high = ELIGIBLE_BAND
    in_band = [row for row in transfer_rows if low <= int(row["length"]) <= high]
    in_target = [row for row in transfer_rows if str(row["group"]) in excluded_groups]
    candidates = [
        row for row in in_band if str(row["group"]) not in excluded_groups
    ]
    overlap = [row for row in candidates if str(row["id"])[3:] in fit_suffixes]
    eligible = [row for row in candidates if str(row["id"])[3:] not in fit_suffixes]
    accounting = {
        "transfer_rows": len(transfer_rows),
        "transfer_out_of_band": len(transfer_rows) - len(in_band),
        "transfer_in_reviewed_target_clusters": len(in_target),
        "transfer_fit_sequence_overlap": len(overlap),
        "eligible_rows": len(eligible),
    }
    return eligible, accounting


# ------------------------------------------------------------------ preparation


def prepare(protocol_path: Path, root: Path, out: Path) -> dict[str, Any]:
    gp.require_fresh_out(out, PREPARATION_COMPLETION)
    protocol = load_protocol(protocol_path)
    paths = _resolved_paths(Path(root), protocol)
    inputs = hash_inputs(paths, protocol, recorded=None)

    fit = json.loads(paths["fit_report"].read_text(encoding="utf-8"))
    if list(fit.get("licensed_band", [])) != list(ELIGIBLE_BAND):
        raise Refusal(
            f"the pinned fit report licenses band {fit.get('licensed_band')}, not {list(ELIGIBLE_BAND)}"
        )

    contract = _validate_embedding_contract(
        paths["embedding_index"],
        paths["embeddings"],
        paths["embedding_metadata"],
        protocol,
        array_sha256=inputs["embeddings"]["sha256"],
    )
    rows = read_table_rows(paths["rows"])
    validate_rows_against_index(rows, contract)

    current = str(protocol["source_version"]["current_rows_sha256"])
    historical = str(protocol["source_version"]["historical_rows_sha256"])
    if current == historical:
        raise Refusal("the protocol pins the same digest as both current and historical")
    source_version = {
        "current_rows_sha256_observed": inputs["rows"]["sha256"],
        "current_rows_sha256_pinned": current,
        "historical_rows_sha256_pinned": historical,
        "historical_rows_recovered": False,
        "current_equals_historical": inputs["rows"]["sha256"] == historical,
        "note": (
            "this preparation validates the current table against the saved index. It does not "
            "recover, relabel or claim the absent historical table"
        ),
    }

    transfer_rows = [row for row in rows if str(row["split"]) == ELIGIBLE_SPLIT]
    fit_suffixes = {str(row["id"])[3:] for row in rows if str(row["id"]).startswith("mg_")}
    reviewed = list(protocol["preparation"]["reviewed_targets"])
    backgrounds = json.loads(paths["cohort"].read_text(encoding="utf-8"))["backgrounds"]
    resolution, excluded_groups = bind_targets(transfer_rows, backgrounds, reviewed)

    eligible, accounting = select_eligible(transfer_rows, excluded_groups, fit_suffixes)
    eligible.sort(key=lambda row: contract["position"][str(row["id"])])

    ordering = [str(row["id"]) for row in eligible]
    eligible_records = [
        {
            "id": str(row["id"]),
            "sequence": str(row["sequence"]),
            "length": int(row["length"]),
            "group": str(row["group"]),
            "index_position": int(contract["position"][str(row["id"])]),
            "sequence_digest": _digest16(str(row["sequence"])),
        }
        for row in eligible
    ]
    eligible_path = Path(out) / ELIGIBLE_ROWS
    _atomic_text(
        eligible_path, "".join(json.dumps(record, sort_keys=True) + "\n" for record in eligible_records)
    )

    record = {
        "schema": PREPARATION_SCHEMA,
        "status": "prepared",
        "status_note": (
            "prepared is not a validation result. Nothing here reads a measured value; only "
            "evaluate may qualify the gate"
        ),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frozen_date": date.today().isoformat(),
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "root": str(root),
        "source_version": source_version,
        "inputs": inputs,
        "embedding_contract": {
            key: contract[key]
            for key in (
                "n_rows",
                "width",
                "index_sha256",
                "index_bytes",
                "array_data_bytes",
                "array_sha256",
                "recorded_historical_rows_sha256",
                "by_split",
            )
        },
        "embedding_index_digest_note": (
            "the protocol records no producer-time index digest, so the index digest recorded here "
            "was computed today and is not claimed as a historical proof"
        ),
        "coverage": {
            "n_current_rows": len(rows),
            "n_index_rows": contract["n_rows"],
            "ids_unique": True,
            "indexed_identity_coverage_exact": True,
            "by_split": contract["by_split"],
            "indexed_by_split": contract["indexed_by_split"],
            "indexed_apply_rows": contract["indexed_by_split"].get("apply", 0),
            "unindexed_apply_count": contract["unindexed_apply_count"],
            "unindexed_apply_ids": contract["unindexed_apply_ids"],
            "unindexed_apply_by_split": contract["unindexed_apply_by_split"],
            "unindexed_apply_policy": (
                "explicit split=apply application records added to the current table after the frozen "
                "embedding run. They carry no embedding, are never eligible and cannot excuse any "
                "missing MGnify, transfer or indexed-apply row"
            ),
            "coverage_note": (
                "split and identity coverage only. A row-count or count match does not prove the "
                "original measurement labels are identical, and no label replay is claimed"
            ),
        },
        "targets": {
            "reviewed": reviewed,
            "resolution": resolution,
            "excluded_groups": sorted(excluded_groups),
            "excluded_rows": accounting["transfer_in_reviewed_target_clusters"],
            "policy": (
                "the complete union of the twelve reviewed targets' WT clusters, including every "
                "variant, is excluded from the eligible population before any measured value is read"
            ),
        },
        "eligible": {
            "band": list(ELIGIBLE_BAND),
            "band_source": "the frozen evaluator's declared licensed band, cross-checked against the pinned fit report",
            "n": len(eligible_records),
            "n_groups": len({record["group"] for record in eligible_records}),
            "ids_sha256": _text_digest(ordering),
            "rows_jsonl": ELIGIBLE_ROWS,
            "rows_sha256": sha256_file(eligible_path),
            "rows_bytes": int(eligible_path.stat().st_size),
        },
        "accounting": accounting,
        "outcomes_used": False,
        "no_measured_outcomes_read": True,
        "limits": [
            "this is a new current-source preparation, not reproduction of the missing historical table",
            "prepared does not qualify the gate; a passing preparation is a prerequisite, not a result",
        ],
    }
    preparation_path = Path(out) / PREPARATION_COMPLETION
    write_json(preparation_path, record)
    _atomic_text(Path(out) / f"{PREPARATION_COMPLETION}.sha256", sha256_file(preparation_path) + "\n")
    return record


# -------------------------------------------------------------------- evaluate


def _load_frozen_head(path: Path, expected_size: int) -> dict[str, Any]:
    with np.load(path) as data:
        model = {
            "coefficients": np.asarray(data["coefficients"], dtype=np.float64),
            "mean": np.asarray(data["mean"], dtype=np.float64),
            "scale": np.asarray(data["scale"], dtype=np.float64),
            "intercept": float(np.asarray(data["intercept"]).reshape(-1)[0]),
        }
    for key in ("coefficients", "mean", "scale"):
        if model[key].shape != (expected_size,):
            raise Refusal(f"{path} carries a {key} of shape {model[key].shape}, not ({expected_size},)")
        if not np.isfinite(model[key]).all():
            raise Refusal(f"{path} carries a non-finite {key}")
    if not np.isfinite(np.asarray(model["intercept"])):
        raise Refusal(f"{path} carries a non-finite intercept")
    return model


def feature_block(sequences: list[str], embedding_rows: np.ndarray) -> np.ndarray:
    """The original frozen feature block: 22 composition/length features, then the vector."""

    cheap = np.stack([ds.composition_features(sequence) for sequence in sequences])
    dense = np.asarray(embedding_rows, dtype=np.float64)
    return np.hstack([cheap, dense])


def qualify(
    truth: np.ndarray,
    plm: np.ndarray,
    baseline: np.ndarray,
    groups: list[str],
    *,
    seed: int,
    draws: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The existing gate, applied with the protocol's own seed and draw count."""

    contrast = ds.transfer_contrast(
        truth.tolist(), plm.tolist(), baseline.tolist(), groups, seed=seed, n_bootstrap=draws
    )
    return contrast, ds.evaluate_gate(contrast)


def _read_eligible_rows(prepared: Path) -> list[dict[str, Any]]:
    path = Path(prepared) / ELIGIBLE_ROWS
    if not path.is_file():
        raise Refusal(f"{path} is missing")
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not records:
        raise Refusal(f"{path} carries no eligible row")
    return records


def evaluate(
    protocol_path: Path,
    root: Path,
    prepared: Path,
    prepared_sha256: str,
    device: str,
    out: Path,
) -> dict[str, Any]:
    if device != "cpu":
        raise Refusal(f"the requalification stage is CPU only; got device {device!r}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gp.require_fresh_out(out, EVALUATION_COMPLETION)

    preparation_path = Path(prepared) / PREPARATION_COMPLETION
    if not preparation_path.is_file():
        raise Refusal(f"{preparation_path} is missing")
    observed_sha = sha256_file(preparation_path)
    if observed_sha != prepared_sha256:
        raise Refusal(
            f"preparation checksum mismatch: {preparation_path} is {observed_sha}, "
            f"not the requested {prepared_sha256}"
        )
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if preparation.get("schema") != PREPARATION_SCHEMA:
        raise Refusal("the preparation document is not a requalification preparation")

    protocol = load_protocol(protocol_path)
    protocol_sha = sha256_file(protocol_path)
    if preparation["protocol"]["sha256"] != protocol_sha:
        raise Refusal("the preparation was made against a different protocol document")
    paths = _resolved_paths(Path(root), protocol)
    inputs = hash_inputs(paths, protocol, recorded=preparation["inputs"])

    contract = _validate_embedding_contract(
        paths["embedding_index"],
        paths["embeddings"],
        paths["embedding_metadata"],
        protocol,
        array_sha256=inputs["embeddings"]["sha256"],
    )
    if contract["index_sha256"] != preparation["embedding_contract"]["index_sha256"]:
        raise Refusal("the embedding index digest changed since preparation")
    if contract["n_rows"] != preparation["embedding_contract"]["n_rows"]:
        raise Refusal("the embedding row count changed since preparation")

    rows = read_table_rows(paths["rows"])
    validate_rows_against_index(rows, contract)

    prepared_records = _read_eligible_rows(prepared)
    prepared_ids = [str(record["id"]) for record in prepared_records]
    if _text_digest(prepared_ids) != preparation["eligible"]["ids_sha256"]:
        raise Refusal("the frozen eligible-ID digest does not match the preparation record")
    if len(prepared_records) != int(preparation["eligible"]["n"]):
        raise Refusal("the frozen eligible count does not match the preparation record")
    if sha256_file(Path(prepared) / ELIGIBLE_ROWS) != preparation["eligible"]["rows_sha256"]:
        raise Refusal("the eligible rows file digest does not match the preparation record")
    allowlist = {str(record["id"]): record for record in prepared_records}
    if len(allowlist) != len(prepared_records):
        raise Refusal("the frozen eligible allowlist carries duplicate ids")

    # Read measured outcomes only inside the allowlist. Excluded lines are identified
    # from the raw bytes and never parsed, so no excluded measured value is inspected.
    truth: list[float] = []
    resolved: list[dict[str, Any]] = []
    found: set[str] = set()
    with paths["rows"].open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            identifier = _peek_id(line)
            if identifier is None or identifier not in allowlist:
                continue
            record = allowlist[identifier]
            row = json.loads(line)
            if identifier in found:
                raise Refusal(f"eligible id {identifier} occurs more than once in the current table")
            found.add(identifier)
            if str(row["sequence"]) != record["sequence"]:
                raise Refusal(f"{identifier}: the eligible sequence changed since preparation")
            if int(row["length"]) != int(record["length"]):
                raise Refusal(f"{identifier}: the eligible length changed since preparation")
            if str(row["group"]) != str(record["group"]):
                raise Refusal(f"{identifier}: the eligible group changed since preparation")
            position = int(record["index_position"])
            if contract["ids"][position] != identifier or contract["splits"][position] != ELIGIBLE_SPLIT:
                raise Refusal(f"{identifier}: the saved index cell no longer matches the allowlist")
            value = row["measured_delta_g"]
            if value is None or not np.isfinite(float(value)):
                raise Refusal(f"eligible id {identifier} carries a non-finite measured value")
            truth.append(float(value))
            resolved.append(record)

    missing = sorted(set(allowlist) - found)
    if missing:
        raise Refusal(f"{len(missing)} eligible ids have no current table row (first {missing[:3]})")
    if not resolved:
        raise Refusal("no eligible row carried a measured value; the qualification is unavailable")

    positions = np.asarray([int(record["index_position"]) for record in resolved], dtype=np.int64)
    sequences = [str(record["sequence"]) for record in resolved]
    groups = [str(record["group"]) for record in resolved]
    vectors = np.asarray(np.load(paths["embeddings"], mmap_mode="r")[positions], dtype=np.float64)
    if not np.isfinite(vectors).all():
        raise Refusal("a stored embedding vector is not finite")

    plm_model = _load_frozen_head(paths["plm_head"], len(ds.BASELINE_FEATURES) + contract["width"])
    baseline_model = _load_frozen_head(paths["baseline_head"], len(ds.BASELINE_FEATURES))
    plm_predictions = ds.predict_with(plm_model, feature_block(sequences, vectors))
    cheap_only = np.stack([ds.composition_features(sequence) for sequence in sequences])
    baseline_predictions = ds.predict_with(baseline_model, cheap_only)
    y = np.asarray(truth, dtype=np.float64)

    validation = protocol["validation"]
    contrast, gate = qualify(
        y,
        plm_predictions,
        baseline_predictions,
        groups,
        seed=int(validation["seed"]),
        draws=int(validation["bootstrap_draws"]),
    )

    gates_passed = bool(gate["passed"])
    record = {
        "schema": EVALUATION_SCHEMA,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gates_passed": gates_passed,
        "pilot_blocked": not gates_passed,
        "interpretation": (
            "both declared conditions hold; a passing gate licenses a predicted computational "
            "endpoint inside the licensed band only"
            if gates_passed
            else "the gate did not pass, so the gate-qualified pilot is blocked. This is a failed "
            "qualification of the frozen evaluator on this source, not a biological negative and "
            "not an equivalence or absence conclusion"
        ),
        "qualification_source": (
            "current validation table only; the absent historical 678e8b... table was not "
            "recovered and is not claimed"
        ),
        "backend": {
            "device": "cpu",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_used": False,
        },
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha},
        "preparation": {
            "path": str(preparation_path),
            "sha256": observed_sha,
            "eligible_rows_sha256": preparation["eligible"]["rows_sha256"],
            "n_eligible": len(resolved),
        },
        "inputs": inputs,
        "source_version": preparation["source_version"],
        "predictors": {
            "plm": {
                "path": str(paths["plm_head"]),
                "sha256": inputs["plm_head"]["sha256"],
                "n_coefficients": int(len(ds.BASELINE_FEATURES) + contract["width"]),
                "no_fit_performed": True,
                "no_alpha_selection": True,
            },
            "baseline": {
                "path": str(paths["baseline_head"]),
                "sha256": inputs["baseline_head"]["sha256"],
                "n_coefficients": int(len(ds.BASELINE_FEATURES)),
                "no_fit_performed": True,
                "no_alpha_selection": True,
            },
        },
        "features": {
            "composition": f"{len(ds.BASELINE_FEATURES)} ordered {ds.BASELINE_FEATURES}",
            "embedding": f"{contract['width']}-dimensional stored float16 cast to float64",
            "feature_block": "composition/length first, then the saved vector",
        },
        "eligibility": {
            "band": list(ELIGIBLE_BAND),
            "n": len(resolved),
            "n_groups": len(set(groups)),
            "frozen_allowlist_only": True,
            "excluded_outcomes_never_read": True,
            "excluded_groups": preparation["targets"]["excluded_groups"],
        },
        "contrast": contrast,
        "gates": gate,
        "gate_conditions": dict(ds.GATE_CONDITIONS),
        "quantities": dict(ds.QUANTITIES),
        "seed": int(validation["seed"]),
        "bootstrap_draws": int(validation["bootstrap_draws"]),
        "old_report_unchanged": True,
        "held_out_outcomes_read": False,
        "reading_guide": [
            "the contrast is the frozen evaluator's cross-population rank agreement on the current "
            "eligible transfer rows, resampled by the original WT cluster",
            "a failed gate blocks the pilot; it is not evidence that the endpoint is biologically absent",
            "this qualifies the frozen evaluator on this source only and does not reproduce the old fit",
        ],
        "limits": list(protocol["limits"]),
    }
    write_json(Path(out) / EVALUATION_COMPLETION, record)
    return record


# ------------------------------------------------------------------------- CLI


def run_requalification(
    protocol_path: Path, root: Path, device: str, out: Path
) -> dict[str, Any]:
    """One CPU queue cell: prepare, seal the preparation digest, then evaluate.

    Pure orchestration. It reads no table line and no measured value itself; it
    only hashes the preparation the existing `prepare` wrote and hands that exact
    digest to the existing `evaluate`.
    """

    if device != "cpu":
        raise Refusal(f"the requalification run is CPU only; got device {device!r}")
    gp.require_fresh_out(out, RUN_COMPLETION)
    preparation_dir = Path(out) / "preparation"
    qualification_dir = Path(out) / "qualification"
    prepare(protocol_path, root, preparation_dir)
    preparation_sha = sha256_file(preparation_dir / PREPARATION_COMPLETION)
    evaluate(protocol_path, root, preparation_dir, preparation_sha, device, qualification_dir)
    qualification_path = qualification_dir / EVALUATION_COMPLETION
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    record = {
        "schema": "frozen_domain_stability_requalification_run_v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": "cpu",
        "preparation_path": str(preparation_dir / PREPARATION_COMPLETION),
        "preparation_sha256": preparation_sha,
        "qualification_path": str(qualification_path),
        "qualification_sha256": sha256_file(qualification_path),
        "gates_passed": bool(qualification["gates_passed"]),
        "pilot_blocked": bool(qualification["pilot_blocked"]),
        "no_outcomes_read_by_run": True,
    }
    write_json(Path(out) / RUN_COMPLETION, record)
    return record


def run_prepare(args: argparse.Namespace) -> dict[str, Any]:
    return prepare(args.protocol, args.root, args.out)


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    return evaluate(
        args.protocol, args.root, args.prepared, args.prepared_sha256, args.device, args.out
    )


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    return run_requalification(args.protocol, args.root, args.device, args.out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # The campaign queue injects --device and --out before the sub-command. They are
    # accepted in both positions; parse_args() reconciles the two.
    parser.add_argument("--device", choices=("cpu",), default=None, dest="top_device",
                        help="resource contract, accepted before the sub-command")
    parser.add_argument("--out", type=Path, default=None, dest="top_out",
                        help="a fresh output directory, accepted before the sub-command")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_parser = sub.add_parser("prepare", help="freeze the eligible allowlist without reading outcomes")
    prepare_parser.add_argument("--protocol", type=Path, required=True)
    prepare_parser.add_argument("--root", type=Path, required=True, help="repository root holding the inputs")
    prepare_parser.add_argument("--device", choices=("cpu",), default=None,
                                help="explicit resource contract; the queue injects --device cpu")
    prepare_parser.add_argument("--out", type=Path, default=None, help="a fresh output directory")
    prepare_parser.set_defaults(handler=run_prepare)

    evaluate_parser = sub.add_parser("evaluate", help="qualify the gate on the frozen allowlist")
    evaluate_parser.add_argument("--protocol", type=Path, required=True)
    evaluate_parser.add_argument("--root", type=Path, required=True)
    evaluate_parser.add_argument("--prepared", type=Path, required=True, help="the prepare output directory")
    evaluate_parser.add_argument("--prepared-sha256", required=True, help="exact sha256 of preparation.json")
    evaluate_parser.add_argument("--device", choices=("cpu",), default=None,
                                 help="accepted because the campaign queue injects it")
    evaluate_parser.add_argument("--out", type=Path, default=None, help="a fresh output directory")
    evaluate_parser.set_defaults(handler=run_evaluate)

    run_parser = sub.add_parser("run", help="one CPU cell: prepare then evaluate with the sealed digest")
    run_parser.add_argument("--protocol", type=Path, required=True)
    run_parser.add_argument("--root", type=Path, required=True)
    run_parser.add_argument("--device", choices=("cpu",), default=None,
                            help="explicit resource contract; the queue injects this")
    run_parser.add_argument("--out", type=Path, default=None, help="a fresh output directory")
    run_parser.set_defaults(handler=run_all)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse argv; --device and --out may appear before or after the sub-command.

    A value given in both positions must agree, otherwise the parse fails. Device
    defaults to cpu, the only permitted resource contract; --out is required.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("device", "out"):
        before = getattr(args, f"top_{name}")
        after = getattr(args, name)
        if before is not None and after is not None and before != after:
            parser.error(f"conflicting --{name}: {before!r} before the sub-command, {after!r} after it")
        value = after if after is not None else before
        if value is None:
            if name == "out":
                parser.error("--out is required, before or after the sub-command")
            value = "cpu"
        setattr(args, name, value)
    return args


def main() -> None:
    args = parse_args()
    try:
        record = args.handler(args)
    except Refusal as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    if args.command == "prepare":
        print(
            json.dumps(
                {
                    "status": record["status"],
                    "n_eligible": record["eligible"]["n"],
                    "n_groups": record["eligible"]["n_groups"],
                    "outcomes_used": record["outcomes_used"],
                },
                sort_keys=True,
            )
        )
    elif args.command == "evaluate":
        print(
            json.dumps(
                {
                    "status": record["status"],
                    "gates_passed": record["gates_passed"],
                    "n_eligible": record["eligibility"]["n"],
                    "n_groups": record["eligibility"]["n_groups"],
                },
                sort_keys=True,
            )
        )
    else:
        print(
            json.dumps(
                {
                    "status": record["status"],
                    "gates_passed": record["gates_passed"],
                    "pilot_blocked": record["pilot_blocked"],
                    "preparation_sha256": record["preparation_sha256"],
                    "qualification_sha256": record["qualification_sha256"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()