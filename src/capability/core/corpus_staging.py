"""Provenance-bound staging of an external sequence corpus.

An external resource cannot enter a baseline on trust. Every file UniProt
distributes is covered by a digest in the release's own metadata document; this
module reads that digest, refuses a file the metadata does not cover, and
refuses bytes on disk that do not reproduce it. Nothing here transfers anything
-- the caller runs the download -- so both refusals are exercisable without a
network.

The free-space guard is here for the same reason the digest guard is: a partial
write of a 32 GB payload is indistinguishable from a complete one once the
transfer has exited, so the volume is measured before the transfer and again
before the index build, and a request that cannot complete raises rather than
being trimmed. The reserve is a caller's margin on top of that, not a property
of this module: pass zero when the whole volume may be used.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
from xml.etree import ElementTree
import hashlib
import os

#: Digest algorithms a release metadata document may declare, as the names
#: Metalink uses them, mapped onto :mod:`hashlib`. An algorithm outside this map
#: is reported as unsupported rather than silently skipped, because skipping it
#: would turn "no digest I can check" into "no digest is required".
SUPPORTED_ALGORITHMS: Mapping[str, str] = {
    "md5": "md5",
    "sha-1": "sha1",
    "sha1": "sha1",
    "sha-256": "sha256",
    "sha256": "sha256",
    "sha-512": "sha512",
    "sha512": "sha512",
}

#: Bytes per hashing read. Large enough that a 32 GB payload is bounded by the
#: volume's read rate rather than by call overhead.
CHUNK_BYTES = 1 << 22


@dataclass(frozen=True)
class PublishedDigest:
    """One distributed file's declared size and digest, with its own provenance.

    ``metadata_url`` is retained because the digest is only as good as the
    document it was read from, and a manifest that records the value without
    recording where it came from cannot be re-checked by a later reader.
    """

    filename: str
    size_bytes: int
    algorithm: str
    value: str
    metadata_url: str

    def record(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "declared_bytes": self.size_bytes,
            "declared_digest_algorithm": self.algorithm,
            "declared_digest": self.value,
            "declared_by": self.metadata_url,
        }


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def parse_release_metalink(text: str, filename: str, *, metadata_url: str) -> PublishedDigest:
    """The declared size and digest of ``filename`` in a Metalink document.

    Raises when the document does not name the file, declares no size, or
    declares no digest in a supported algorithm. A resource whose publisher
    states no digest is refused here: staging it would put an unverifiable
    corpus under a baseline, and a byte count is not a digest.
    """

    root = ElementTree.fromstring(text)
    for element in root.iter():
        if _localname(element.tag) != "file":
            continue
        if element.get("name") != filename:
            continue
        size: int | None = None
        digests: dict[str, str] = {}
        for child in element.iter():
            name = _localname(child.tag)
            if name == "size" and child.text and child.text.strip().isdigit():
                size = int(child.text.strip())
            elif name == "hash":
                algorithm = (child.get("type") or "").strip().lower()
                value = (child.text or "").strip().lower()
                if algorithm and value:
                    digests[algorithm] = value
        if size is None:
            raise ValueError(
                f"{metadata_url}: the release metadata declares no size for "
                f"{filename}; an external resource is not staged on an unstated length"
            )
        for algorithm, value in digests.items():
            if algorithm in SUPPORTED_ALGORITHMS:
                return PublishedDigest(filename=filename, size_bytes=size,
                                       algorithm=algorithm, value=value,
                                       metadata_url=metadata_url)
        if digests:
            raise ValueError(
                f"{metadata_url}: {filename} carries only digests this staging "
                f"cannot check ({sorted(digests)}); supported algorithms are "
                f"{sorted(SUPPORTED_ALGORITHMS)}"
            )
        raise ValueError(
            f"{metadata_url}: {filename} carries no digest in the release "
            "metadata; an unhashed external resource cannot enter a baseline"
        )
    raise KeyError(
        f"{metadata_url}: the release metadata does not name {filename}, so no "
        "digest is published for it"
    )


def file_digests(path: Path, algorithms: Iterable[str] = ("md5", "sha256")) -> dict[str, str]:
    """Hex digests of ``path`` under every named algorithm, in one pass."""

    wanted = list(algorithms)
    unsupported = [name for name in wanted if name not in SUPPORTED_ALGORITHMS]
    if unsupported:
        raise ValueError(f"unsupported digest algorithms: {unsupported}")
    hashers = {name: hashlib.new(SUPPORTED_ALGORITHMS[name]) for name in wanted}
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            for hasher in hashers.values():
                hasher.update(chunk)
    return {name: hasher.hexdigest() for name, hasher in hashers.items()}


def verify_staged_file(path: Path, digest: PublishedDigest,
                       *, record_also: Iterable[str] = ("sha256",)) -> dict[str, object]:
    """Check ``path`` against its publisher's declaration and record both digests.

    Raises on a missing file, on a length that differs from the declared one,
    and on a digest that differs from the declared one. The extra algorithms in
    ``record_also`` are computed in the same pass so the manifest can carry the
    repository's own durable digest beside the publisher's.
    """

    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != digest.size_bytes:
        raise ValueError(
            f"{path.name}: staged {size} bytes against a declared "
            f"{digest.size_bytes}; the transfer is incomplete or the release moved"
        )
    algorithms = [digest.algorithm, *(name for name in record_also
                                      if name != digest.algorithm)]
    digests = file_digests(path, algorithms)
    if digests[digest.algorithm] != digest.value:
        raise ValueError(
            f"{path.name}: {digest.algorithm} {digests[digest.algorithm]} does not "
            f"match the {digest.value} declared by {digest.metadata_url}"
        )
    return {"path": str(path), "bytes": size, "digests": digests,
            "verified_against": digest.record()}


def free_bytes(path: Path) -> int:
    """Bytes available to this user on the volume holding ``path``."""

    target = path
    while not target.exists():
        if target.parent == target:
            raise FileNotFoundError(path)
        target = target.parent
    usage = os.statvfs(target)
    return usage.f_bavail * usage.f_frsize


def require_free_space(path: Path, needed_bytes: int, *, reserve_bytes: int) -> int:
    """Refuse to start when the volume cannot hold the payload and its reserve.

    Returns the free space it measured, so a caller can record what the check
    saw rather than re-measuring a number that has already moved.
    """

    if needed_bytes < 0 or reserve_bytes < 0:
        raise ValueError("the needed and reserved byte counts must be non-negative")
    available = free_bytes(path)
    if available < needed_bytes + reserve_bytes:
        raise RuntimeError(
            f"{path}: {available / 1e9:.1f} GB free against {needed_bytes / 1e9:.1f} GB "
            f"needed plus a {reserve_bytes / 1e9:.1f} GB reserve; the staging "
            "stops instead of starting a transfer that cannot complete"
        )
    return available
