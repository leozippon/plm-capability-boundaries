"""Deterministic prefix-consistent i.i.d. uniform 20-AA sequence dataset generation."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from .spec import SEQ_LEN, VOCAB_AA

DEFAULT_DATA_CHUNK_SIZE: int = 65_536


def dataset_cache_filename(
    n_sequences: int,
    *,
    seq_len: int = SEQ_LEN,
    data_seed: int = 20261127,
    vocab_size: int = VOCAB_AA,
) -> str:
    """Canonical deterministic filename for optional on-disk .npy caching."""
    return f"iid_aa{vocab_size}_L{seq_len}_seed{data_seed}_N{n_sequences}.npy"


def compute_row_hashes_uint64(
    sequences: np.ndarray,
    *,
    row_chunk_size: int = DEFAULT_DATA_CHUNK_SIZE,
) -> np.ndarray:
    """Compute a fast SplitMix64 row digest over uint8 sequences of shape (N, L)."""
    if sequences.ndim != 2 or sequences.dtype != np.uint8:
        raise ValueError(
            f"compute_row_hashes_uint64 expects 2D uint8 array, got shape={sequences.shape}, "
            f"dtype={sequences.dtype}"
        )
    n_rows, seq_len = sequences.shape
    out = np.empty(n_rows, dtype=np.uint64)
    if n_rows == 0:
        return out

    pad_bytes = (-seq_len) % 8
    c0 = np.uint64(0x9E3779B97F4A7C15)
    m1 = np.uint64(0xBF58476D1CE4E5B9)
    m2 = np.uint64(0x94D049BB133111EB)
    m3 = np.uint64(0xD6E8FEB86659FD93)
    add_c = np.uint64(0xA5A5A5A5A5A5A5A5)
    s30 = np.uint64(30)
    s27 = np.uint64(27)
    s31 = np.uint64(31)
    s32 = np.uint64(32)
    init_val = np.uint64(0x9E3779B97F4A7C15 ^ seq_len)

    with np.errstate(over="ignore"):
        for start in range(0, n_rows, row_chunk_size):
            end = min(n_rows, start + row_chunk_size)
            chunk = np.ascontiguousarray(sequences[start:end])
            if pad_bytes > 0:
                padded = np.zeros((end - start, seq_len + pad_bytes), dtype=np.uint8)
                padded[:, :seq_len] = chunk
                words = padded.view(np.uint64)
            else:
                words = chunk.view(np.uint64)

            h = np.full(end - start, init_val, dtype=np.uint64)
            for col in range(words.shape[1]):
                k = words[:, col] + c0 + np.uint64(col)
                k = (k ^ (k >> s30)) * m1
                k = (k ^ (k >> s27)) * m2
                k = k ^ (k >> s31)
                h = (h ^ k) * m3 + add_c
                h = h ^ (h >> s32)
            out[start:end] = h

    return out


def verify_and_deduplicate_sequences(
    sequences: np.ndarray,
    *,
    data_seed: int,
    vocab_size: int = VOCAB_AA,
    max_attempts_per_row: int = 10_000,
) -> tuple[np.ndarray, int]:
    """Verify exact sequence uniqueness via 64-bit row hash + byte check.

    If any accidental collision occurs, deterministically resample only the later
    colliding row index so prefix invariance (N1 < N2) is strictly preserved.
    """
    if sequences.ndim != 2 or sequences.dtype != np.uint8:
        raise ValueError("sequences must be a 2D uint8 array")
    n_rows, seq_len = sequences.shape
    if n_rows <= 1:
        return sequences, 0
    if math.log2(float(n_rows)) > float(seq_len) * math.log2(float(vocab_size)):
        raise ValueError(
            f"Cannot generate {n_rows} unique sequences when vocab_size**seq_len = "
            f"{vocab_size}**{seq_len}"
        )

    hashes = compute_row_hashes_uint64(sequences)
    order = np.argsort(hashes, kind="mergesort")
    sorted_hashes = hashes[order]
    eq_adj = sorted_hashes[1:] == sorted_hashes[:-1]
    if not np.any(eq_adj):
        return sequences, 0

    duplicate_indices: list[int] = []
    idx = 0
    while idx < n_rows - 1:
        if sorted_hashes[idx] != sorted_hashes[idx + 1]:
            idx += 1
            continue
        run_end = idx + 2
        while run_end < n_rows and sorted_hashes[run_end] == sorted_hashes[idx]:
            run_end += 1
        group_rows = order[idx:run_end]
        kept: list[int] = []
        for row_idx_np in group_rows:
            r_i = int(row_idx_np)
            is_dup = any(
                bool(np.array_equal(sequences[r_i], sequences[prev_i]))
                for prev_i in kept
            )
            if is_dup:
                duplicate_indices.append(r_i)
            else:
                kept.append(r_i)
        idx = run_end

    if not duplicate_indices:
        return sequences, 0

    duplicate_indices.sort()
    hash_to_rows: dict[int, list[int]] = {}
    dup_set = set(duplicate_indices)
    for r_i in range(n_rows):
        if r_i in dup_set:
            continue
        h_int = int(hashes[r_i])
        hash_to_rows.setdefault(h_int, []).append(r_i)

    n_resampled = 0
    for dup_i in duplicate_indices:
        resolved = False
        for attempt in range(max_attempts_per_row):
            resample_rng = np.random.Generator(
                np.random.PCG64([int(data_seed), int(dup_i), int(attempt), 0x517CC1B7])
            )
            candidate = resample_rng.integers(
                0, vocab_size, size=(1, seq_len), dtype=np.uint8
            )
            cand_h = int(compute_row_hashes_uint64(candidate)[0])
            bucket = hash_to_rows.get(cand_h, [])
            if not any(bool(np.array_equal(candidate[0], sequences[p_i])) for p_i in bucket):
                sequences[dup_i] = candidate[0]
                hashes[dup_i] = np.uint64(cand_h)
                hash_to_rows.setdefault(cand_h, []).append(dup_i)
                n_resampled += 1
                resolved = True
                break
        if not resolved:
            raise RuntimeError(
                f"Failed to deduplicate row {dup_i} after {max_attempts_per_row} attempts"
            )

    return sequences, n_resampled


def generate_iid_aa_dataset(
    n_sequences: int,
    *,
    seq_len: int = SEQ_LEN,
    vocab_size: int = VOCAB_AA,
    data_seed: int = 20261127,
    data_cache_dir: Path | str | None = None,
    verify_unique: bool = True,
    chunk_size: int = DEFAULT_DATA_CHUNK_SIZE,
) -> np.ndarray:
    """Generate or load deterministic (n_sequences, seq_len) uint8 sequences in {0..vocab_size-1}.

    Fixed-chunk PCG64 generation guarantees:
      1. Paired AR and MLM runs with the same (n_sequences, data_seed) receive the exact
         identical (N, L) byte array.
      2. For a fixed data_seed, any prefix N1 < N2 is byte-for-byte identical.
    """
    if n_sequences < 1:
        raise ValueError(f"n_sequences must be >= 1, got {n_sequences}")
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if not (2 <= vocab_size <= 255):
        raise ValueError(f"vocab_size must be in [2, 255], got {vocab_size}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    cache_path: Path | None = None
    if data_cache_dir is not None:
        cache_dir = Path(data_cache_dir)
        cache_path = cache_dir / dataset_cache_filename(
            n_sequences,
            seq_len=seq_len,
            data_seed=data_seed,
            vocab_size=vocab_size,
        )
        if cache_path.is_file():
            cached = np.load(cache_path, allow_pickle=False)
            if (
                cached.ndim == 2
                and cached.shape == (n_sequences, seq_len)
                and cached.dtype == np.uint8
            ):
                return np.ascontiguousarray(cached)

    rng = np.random.Generator(np.random.PCG64(int(data_seed)))
    sequences = np.empty((n_sequences, seq_len), dtype=np.uint8)
    for start in range(0, n_sequences, chunk_size):
        block = rng.integers(0, vocab_size, size=(chunk_size, seq_len), dtype=np.uint8)
        take = min(chunk_size, n_sequences - start)
        sequences[start : start + take] = block[:take]

    if verify_unique:
        sequences, _ = verify_and_deduplicate_sequences(
            sequences,
            data_seed=data_seed,
            vocab_size=vocab_size,
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f".{cache_path.name}.tmp-{os.getpid()}.npy")
        try:
            with tmp_path.open("wb") as handle:
                np.save(handle, sequences, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            tmp_path.replace(cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return sequences


generate_iid_sequences = generate_iid_aa_dataset
