"""Hash-bound inventory of the staged dataset directories under ``data/``.

Every external resource that enters a baseline has to be bound to a source
digest, and until this module existed exactly one of the fifteen staged dataset
directories carried a manifest. The registry closes that gap: one JSON record
holds the authored provenance of each directory -- source, release, retrieval
route, licence, redistribution constraint -- beside a measured file inventory
with a SHA-256 per file.

Two rules shape the schema. A provenance field that cannot be recovered is
recorded as ``{"recoverable": false, "reason": ...}`` rather than guessed, so an
admitted gap is visible and a fabricated release string cannot pass as one. And
the inventory is measured rather than authored, so a digest in the registry is
always a digest something computed over bytes on disk.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io import sha256_file

SCHEMA_VERSION = "dataset_registry_v1"

#: Registry location, relative to the repository root. It sits beside the data
#: it describes, as ``data/megascale_tsuboyama2023/manifest.json`` already does,
#: because ``data/`` is git-ignored and the on-disk record is the durable one.
REGISTRY_RELPATH = "data/dataset_registry.json"

#: Provenance fields every declared dataset carries. Each is either a non-empty
#: string or an explicit unrecoverable marker; nothing may be absent.
PROVENANCE_FIELDS = (
    "purpose",
    "serves",
    "source_name",
    "source_url",
    "release",
    "retrieved",
    "retrieval",
    "licence",
    "redistribution",
)


@dataclass(frozen=True)
class FileRecord:
    """One staged file: its path inside the dataset directory, size and digest."""

    path: str
    bytes: int
    sha256: str

    def as_json(self) -> dict[str, Any]:
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}


def dataset_files(directory: Path) -> list[str]:
    """Every regular file under ``directory``, as sorted relative POSIX paths."""

    found: list[str] = []
    for parent, _, names in os.walk(directory):
        for name in names:
            path = Path(parent) / name
            found.append(path.relative_to(directory).as_posix())
    return sorted(found)


def inventory(directory: Path, workers: int = 16) -> list[FileRecord]:
    """Measure size and SHA-256 for every file under ``directory``.

    Hashing is threaded because ``hashlib`` releases the interpreter lock for
    the block sizes used here, so the walk is bounded by disk throughput rather
    than by one core.
    """

    relpaths = dataset_files(directory)

    def measure(relpath: str) -> FileRecord:
        path = directory / relpath
        return FileRecord(relpath, path.stat().st_size, sha256_file(path))

    if not relpaths:
        return []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(measure, relpaths))


def unrecoverable(value: Any) -> bool:
    """True when a provenance field is an explicit unrecoverable marker."""

    return isinstance(value, dict) and value.get("recoverable") is False


def _field_problem(dataset: str, field: str, value: Any) -> str | None:
    if isinstance(value, str):
        return None if value.strip() else f"{dataset}.{field} is an empty string"
    if unrecoverable(value):
        reason = value.get("reason")
        if isinstance(reason, str) and reason.strip():
            return None
        return f"{dataset}.{field} is marked unrecoverable without a reason"
    return f"{dataset}.{field} is neither a value nor an unrecoverable marker"


def structural_problems(registry: dict[str, Any]) -> list[str]:
    """Every internal inconsistency in a registry, as human-readable lines.

    Checks the schema tag, the authored provenance fields, and the arithmetic of
    each measured inventory against its own file list. It reads no file, so it
    answers "is this record self-consistent" and never "does the data match".
    """

    problems: list[str] = []
    if registry.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version is {registry.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
        )
    datasets = registry.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        problems.append("datasets is missing or empty")
        return problems

    for name in sorted(datasets):
        entry = datasets[name]
        if not isinstance(entry, dict):
            problems.append(f"{name} is not an object")
            continue
        for field in PROVENANCE_FIELDS:
            if field not in entry:
                problems.append(f"{name}.{field} is missing")
                continue
            problem = _field_problem(name, field, entry[field])
            if problem is not None:
                problems.append(problem)

        flag = entry.get("volatile")
        if flag is not None:
            reason = flag.get("reason") if isinstance(flag, dict) else None
            if not (isinstance(reason, str) and reason.strip()):
                problems.append(f"{name}.volatile carries no reason")

        measured = entry.get("inventory")
        if measured is None:
            problems.append(f"{name} carries no measured inventory")
            continue
        files = measured.get("files")
        if not isinstance(files, list):
            problems.append(f"{name}.inventory.files is not a list")
            continue
        paths = [record.get("path") for record in files]
        if len(set(paths)) != len(paths):
            problems.append(f"{name}.inventory.files lists a path twice")
        if paths != sorted(paths):
            problems.append(f"{name}.inventory.files is not sorted by path")
        for record in files:
            digest = record.get("sha256")
            if not (isinstance(digest, str) and len(digest) == 64):
                problems.append(f"{name}:{record.get('path')} has no 64-character sha256")
            if not isinstance(record.get("bytes"), int):
                problems.append(f"{name}:{record.get('path')} has no integer size")
        if measured.get("file_count") != len(files):
            problems.append(
                f"{name}.inventory.file_count is {measured.get('file_count')}, "
                f"{len(files)} files listed"
            )
        total = sum(r["bytes"] for r in files if isinstance(r.get("bytes"), int))
        if measured.get("total_bytes") != total:
            problems.append(
                f"{name}.inventory.total_bytes is {measured.get('total_bytes')}, files sum to {total}"
            )
    return problems


def is_volatile(entry: dict[str, Any]) -> bool:
    """True when a dataset is declared as under concurrent write.

    A directory another campaign is still staging into cannot be bound to a
    digest: the bytes change while they are read. Such a dataset keeps its
    provenance and a dated inventory snapshot, and is excluded from digest
    verification rather than quietly failing it -- the exclusion is recorded in
    the registry with its reason, so it is a declared gap and not a silence.
    """

    return entry.get("volatile") is not None


def verify_dataset(
    directory: Path,
    files: Iterable[dict[str, Any]],
    *,
    max_bytes_per_file: int | None = None,
    budget_bytes: int | None = None,
    workers: int = 16,
) -> tuple[list[str], int, int]:
    """Re-read the staged bytes and compare them with the recorded inventory.

    Returns the failures, the number of digests actually recomputed and the
    number of bytes read. ``max_bytes_per_file`` and ``budget_bytes`` bound the
    hashing for callers that cannot afford a full pass; existence and size are
    checked for every listed file regardless, and a file present on disk but
    absent from the record is a failure in either mode.
    """

    records = list(files)
    failures: list[str] = []
    recorded = {record["path"] for record in records}
    for extra in sorted(set(dataset_files(directory)) - recorded):
        failures.append(f"{directory.name}:{extra} is on disk but not in the registry")

    to_hash: list[dict[str, Any]] = []
    read_bytes = 0
    for record in records:
        path = directory / record["path"]
        if not path.is_file():
            failures.append(f"{directory.name}:{record['path']} is listed but missing")
            continue
        size = path.stat().st_size
        if size != record["bytes"]:
            failures.append(
                f"{directory.name}:{record['path']} is {size} bytes, registry says {record['bytes']}"
            )
            continue
        if max_bytes_per_file is not None and size > max_bytes_per_file:
            continue
        if budget_bytes is not None and read_bytes + size > budget_bytes:
            continue
        read_bytes += size
        to_hash.append(record)

    def check(record: dict[str, Any]) -> str | None:
        digest = sha256_file(directory / record["path"])
        if digest == record["sha256"]:
            return None
        return (
            f"{directory.name}:{record['path']} hashes {digest}, "
            f"registry says {record['sha256']}"
        )

    if to_hash:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            failures.extend(problem for problem in pool.map(check, to_hash) if problem)
    return failures, len(to_hash), read_bytes
