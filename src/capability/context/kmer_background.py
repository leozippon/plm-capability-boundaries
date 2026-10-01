"""The corpus fragment statistics a homologue-free referent is still exposed to.

Why this module exists
======================

Objective 3 needs a referent where retrieval is excluded *by construction*, and
EXP-R2-189 established that no natural protein supplies one: over 187 ProteinGym
wild types the least homologous is still 55.5% identical to a UniRef50 cluster.
The only class disjoint by construction is non-natural sequence, and the
certificate that decides disjointness is a DIAMOND search that returns no
alignment.

**A homologue-free referent is not an information-free one.** DIAMOND finding no
alignment says nothing about whether the referent's *fragments* are common in the
corpus. A decoder does not need to have seen a sequence to score it above chance;
it needs only to have seen its 3-mers and 4-mers, and every protein is built from
the same small alphabet. So the moment a design is certified zero-hit, the first
objection to any positive likelihood result is that the model is reproducing
corpus fragment statistics -- exactly the alternative Objective 3 exists to
exclude -- and the baseline that answers it has to be fragment-level, because the
homology-based LOOKUP baseline is empty by construction on a disjoint referent.

This module is that baseline's *input*: the background frequency of every 3-mer
and 4-mer in the staged corpus. It is deliberately a **count and not a model**.
Turning counts into a baseline needs a declared scoring rule -- what to do with
unseen contexts, how to combine positions, whether to condition -- and that is a
design decision that belongs to the stage that makes it, not to the counter.

What the count has to get right
===============================

One thing, and it is the reason the number is trustworthy at all: **a k-mer must
never span two records**. Concatenating a 24 GB FASTA and sliding a window over
it produces, at every one of 60 million record boundaries, a fragment that exists
in no protein. Those fragments are not rare noise -- they are systematically
*unusual*, because they join a C-terminus to an N-terminus, and a background that
contains them understates exactly the fragments a real sequence is made of.

The second thing is narrower and cost the previous artefact 3-4.5% of its
windows. FASTA wraps sequence lines, and the staged corpus wraps at 60 columns.
An implementation that treats the newline as an invalid symbol -- which is the
natural way to get the record-boundary property for free -- also drops every
window that spans a *line* break, which is one window in twenty for k = 3 and
one in fifteen for k = 4. The first pass over this corpus did exactly that: it
counted 16,640,807,917 3-mers against a record-local total of 17,156,475,069, a
3.01% shortfall, and 16,324,342,706 4-mers against 17,096,160,025, 4.51%. Line
wrapping is a property of the file format and not of the protein, so those
windows are real k-mers of real proteins and belong in the background.

The distinction matters less for the *distribution* than for the *count* -- line
breaks fall at fixed residue offsets and are uncorrelated with sequence content,
so dropping them is close to missing-at-random -- but a background whose stated
guarantee is "no k-mer crosses a record boundary" while it silently also drops
one window in twenty is a artefact whose documentation does not describe it.
This implementation joins a record's sequence lines and emits one invalid symbol
per header, so a window spans line breaks freely and can never span a record.

Chunking is the third thing, and it is tested rather than reasoned about. The
corpus does not fit in memory, so the pass is chunked; a chunk boundary is not a
record boundary and a window that straddles one must be counted exactly once.
:func:`count_kmers` therefore carries the trailing ``max(ks) - 1`` symbols across
chunks and counts, in each chunk, only the windows that reach new material. The
test suite runs the same file at several chunk sizes and requires identical
counts, because that is the only failure here that produces a plausible number.

Verification against an independent pass
=======================================

The residue total is checkable without trusting this module at all: DIAMOND's
``makedb`` reports the letters it indexed for the same FASTA. On the staged
UniRef50 that is 17,282,055,793 against this pass's 17,277,105,157, and the
0.03% difference is exactly the non-canonical residues (``X``, ``B``, ``Z``,
``U``, ``O``) that this module excludes and DIAMOND does not. Two independent
readers agreeing to 3 parts in 10,000 is what makes the count a measurement
rather than an assertion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core.amino_acids import AA20
from ..core.io import sha256_file, write_json

#: The canonical amino-acid alphabet, in the order that fixes each k-mer's index.
#: A k-mer's index is its base-20 value under this ordering, most significant
#: symbol first, so index and alphabet cannot drift apart without the tests
#: noticing.
ALPHABET: str = AA20

#: Every byte that is not a canonical residue maps here, including the header
#: marker, newlines, whitespace, lowercase and the non-canonical residue codes.
#: Any window containing one is dropped rather than substituted, which is what
#: keeps ``X`` from being silently read as an alanine.
INVALID: int = 255

#: Default read size. Large enough that the per-chunk Python work is negligible
#: against the NumPy pass, small enough that the k = 4 index array stays a few
#: gigabytes.
DEFAULT_CHUNK_BYTES: int = 256 * 1024 * 1024

_LOOKUP = np.full(256, INVALID, dtype=np.uint8)
for _index, _symbol in enumerate(ALPHABET):
    _LOOKUP[ord(_symbol)] = _index


def kmer_index(kmer: str) -> int:
    """The row this k-mer occupies in a count vector.

    Exposed because a consumer that wants one k-mer's frequency must not
    reimplement the base-20 encoding; that is Appendix B rule 12 applied to an
    indexing convention.
    """

    if not kmer:
        raise ValueError("a k-mer must have at least one symbol")
    index = 0
    for symbol in kmer:
        position = ALPHABET.find(symbol)
        if position < 0:
            raise ValueError(f"{symbol!r} is not in the canonical alphabet {ALPHABET}")
        index = index * len(ALPHABET) + position
    return index


@dataclass(frozen=True)
class KmerBackground:
    """Counts over one corpus, with what is needed to say which corpus."""

    counts: dict[int, np.ndarray]
    residues: int
    records: int
    source: Path
    source_bytes: int
    wall_seconds: float

    def __post_init__(self) -> None:
        if not self.counts:
            raise ValueError("a background with no k is not a background")
        for k, vector in self.counts.items():
            if k < 1:
                raise ValueError(f"k must be positive, got {k}")
            if vector.shape != (len(ALPHABET) ** k,):
                raise ValueError(
                    f"k = {k} needs a vector of {len(ALPHABET) ** k} counts, "
                    f"got shape {vector.shape}"
                )
            if vector.dtype != np.int64:
                raise ValueError(f"k = {k} counts must be int64, got {vector.dtype}")
            if (vector < 0).any():
                raise ValueError(f"k = {k} carries a negative count")
        if self.residues < 0 or self.records < 0:
            raise ValueError("residue and record counts cannot be negative")

    @property
    def ks(self) -> tuple[int, ...]:
        return tuple(sorted(self.counts))

    def frequencies(self, k: int) -> np.ndarray:
        """Counts normalised to a distribution over the observed k-mers.

        Raises when nothing was counted rather than returning zeros: an empty
        background is a configuration fault, and a consumer that divides by a
        zero total gets NaNs that :func:`.io.write_json` would refuse three
        stages later, where the cause is no longer visible.
        """

        vector = self.counts[k]
        total = int(vector.sum())
        if total == 0:
            raise ValueError(f"no k = {k} windows were counted; nothing to normalise")
        return vector / total

    def coverage(self, k: int) -> tuple[int, int]:
        """How many of the possible k-mers were seen, and how many there are."""

        return int((self.counts[k] > 0).sum()), len(ALPHABET) ** k

    def record(self) -> dict[str, Any]:
        source = Path(self.source)
        return {
            "schema_version": "kmer_background_v1",
            "alphabet": ALPHABET,
            "source": str(source),
            "source_bytes": self.source_bytes,
            "residues": self.residues,
            "records": self.records,
            "wall_seconds": round(float(self.wall_seconds), 1),
            "k": list(self.ks),
            "counts": {
                str(k): {
                    "observed": self.coverage(k)[0],
                    "possible": self.coverage(k)[1],
                    "total_kmers": int(self.counts[k].sum()),
                }
                for k in self.ks
            },
            "window_rule": (
                "A window is counted when all k of its symbols are canonical residues "
                "of one record. Sequence lines of a record are joined, so a window "
                "spans FASTA line wrapping; each header emits one invalid symbol, so no "
                "window spans a record boundary."
            ),
        }


def _counts_path(directory: Path, k: int) -> Path:
    return Path(directory) / f"kmer_counts_k{k}.npy"


def save(background: KmerBackground, directory: Path) -> dict[str, Any]:
    """Write the count vectors and a manifest that hashes them.

    The manifest is written last and through :func:`.io.write_json`, so a reader
    that finds a manifest finds count files whose digests it can check.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = background.record()
    digests = {}
    for k in background.ks:
        path = _counts_path(directory, k)
        np.save(path, background.counts[k])
        digests[str(k)] = sha256_file(path)
    manifest["sha256"] = digests
    write_json(directory / "manifest.json", manifest)
    return manifest


def load(directory: Path) -> KmerBackground:
    """Read back a saved background, refusing any file whose digest has moved."""

    import json

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "kmer_background_v1":
        raise ValueError(
            f"{directory} carries schema {manifest.get('schema_version')!r}, "
            "not kmer_background"
        )
    if manifest.get("alphabet") != ALPHABET:
        raise ValueError(
            f"{directory} was counted over alphabet {manifest.get('alphabet')!r}; "
            f"this module indexes {ALPHABET}"
        )
    counts = {}
    for k in manifest["k"]:
        path = _counts_path(directory, int(k))
        observed = sha256_file(path)
        expected = manifest["sha256"][str(k)]
        if observed != expected:
            raise RuntimeError(
                f"{path} hashes {observed}, manifest says {expected}"
            )
        counts[int(k)] = np.load(path)
    return KmerBackground(
        counts=counts,
        residues=int(manifest["residues"]),
        records=int(manifest["records"]),
        source=Path(manifest["source"]),
        source_bytes=int(manifest["source_bytes"]),
        wall_seconds=float(manifest["wall_seconds"]),
    )


def _emit(block: bytes) -> tuple[bytes, int]:
    """Turn a whole number of FASTA lines into a symbol stream, and count headers.

    Sequence lines are concatenated with nothing between them, so a window spans
    line wrapping. Each header becomes a single ``>`` byte, which the lookup
    table maps to :data:`INVALID`, so a window can never reach from one record
    into the next. Returning one byte per header rather than the header's own
    bytes keeps the stream short without weakening the separation.
    """

    lines = block.split(b"\n")
    headers = 0
    pieces = []
    for line in lines:
        if line[:1] == b">":
            headers += 1
            pieces.append(b">")
        else:
            pieces.append(line)
    return b"".join(pieces), headers


def count_kmers(
    fasta: Path,
    ks: Sequence[int] = (3, 4),
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> KmerBackground:
    """Count every k-mer of every record, in one chunked pass.

    ``chunk_bytes`` changes only how the file is read. The counts it produces are
    identical at every chunk size, which the test suite checks directly: the
    carry that makes that true is the one piece of state here, and a carry bug
    produces counts that are wrong by a few parts in a million and look fine.
    """

    fasta = Path(fasta)
    ks = tuple(sorted({int(k) for k in ks}))
    if not ks:
        raise ValueError("no k requested")
    if ks[0] < 1:
        raise ValueError(f"k must be positive, got {ks[0]}")
    if chunk_bytes < 1:
        raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")
    if not fasta.is_file():
        raise FileNotFoundError(f"{fasta} does not exist")

    width = len(ALPHABET)
    longest = ks[-1]
    counts = {k: np.zeros(width**k, dtype=np.int64) for k in ks}
    residues = 0
    records = 0
    line_carry = b""
    window_carry = np.empty(0, dtype=np.uint8)
    started = time.time()

    def consume(block: bytes) -> None:
        nonlocal residues, records, window_carry
        stream, headers = _emit(block)
        records += headers
        symbols = _LOOKUP[np.frombuffer(stream, dtype=np.uint8)]
        residues += int((symbols < width).sum())
        combined = np.concatenate((window_carry, symbols))
        carried = window_carry.size
        valid = combined < width
        safe = np.where(valid, combined, 0).astype(np.int64)
        total = combined.size
        for k in ks:
            if total < k:
                continue
            # A window is new when it reaches material this chunk brought in;
            # every earlier one was already counted against the previous chunk.
            first = max(0, carried - k + 1)
            last = total - k + 1
            if first >= last:
                continue
            index = np.zeros(last - first, dtype=np.int64)
            keep = np.ones(last - first, dtype=bool)
            for offset in range(k):
                index *= width
                index += safe[first + offset : last + offset]
                keep &= valid[first + offset : last + offset]
            counts[k] += np.bincount(index[keep], minlength=width**k)
        window_carry = combined[-(longest - 1) :] if longest > 1 else combined[:0]

    with fasta.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            buffer = line_carry + chunk
            cut = buffer.rfind(b"\n")
            if cut < 0:
                line_carry = buffer
                continue
            line_carry, block = buffer[cut + 1 :], buffer[: cut + 1]
            consume(block)
    if line_carry:
        # A file whose last line has no terminating newline. Dropping it would
        # lose a whole record silently.
        consume(line_carry)

    return KmerBackground(
        counts=counts,
        residues=residues,
        records=records,
        source=fasta,
        source_bytes=fasta.stat().st_size,
        wall_seconds=time.time() - started,
    )
