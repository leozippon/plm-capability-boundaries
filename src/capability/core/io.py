"""The one way a transfer measurement reaches disk.

Every stage artefact this programme cites was written through :func:`write_json`,
and the two properties that matter are not stylistic:

**Atomic.** A campaign stage writes its report only after its whole per-arm loop
finishes. A partial file left by an interrupted write is indistinguishable from a
complete one to a reader that only checks the schema version, and this programme
has already lost a run to a stage that wrote and then raised.

**NaN-rejecting.** ``json.dump`` emits bare ``NaN`` and ``Infinity`` by default,
which are not JSON, and a downstream reader that accepts them turns a
non-finite intermediate into a plotted point. Every numeric guard in
``src/capability`` exists to make a non-finite value raise where it is produced;
serialising one silently would undo all of them at the last step.

Before EXP-R2-066 this module was ``src/revision/io.py``, shared with the
retired CLT/dictionary-qualification scope, and ``../d2/scripts/capability/mechanisms/gap_common.py``
carried a *third*, non-atomic ``write_json`` of its own. Appendix B rule 12 --
a single declaration, imported, never reimplemented -- applies to serialisation
as much as to rendering.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a file, for artefact and checkpoint identity."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    text = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
        # The rename is the atomic step, and it is a change to the *directory*.
        # fsync on the file commits its contents; nothing commits the entry that
        # points at them, so a host that loses power between the two can come
        # back with a directory that still names the temporary, or names
        # neither. The "Atomic" claim in this module's headline is about crash
        # behaviour and was only two thirds implemented. Costs one syscall per
        # artefact, against campaigns that run for days on a cluster this
        # programme does not administer.
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: Any) -> None:
    """Write sorted, indented JSON atomically, raising on NaN or infinity."""

    _atomic_write(Path(path), _json_bytes(value))
