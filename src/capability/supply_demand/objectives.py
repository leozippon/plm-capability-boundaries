"""Variable-length sequence packing and exact AR / MLM-15% / OA-Uniform objectives."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .spec import (
    AA20,
    AA_TO_ID,
    BOS_TOKEN_ID,
    MASK_TOKEN_ID,
    OBJECTIVES,
    PAD_TOKEN_ID,
    VOCAB_AA,
)


def encode_aa_sequence(sequence: str) -> np.ndarray:
    """Encode a canonical 20-AA sequence into a uint8 array with values in [0, 19]."""
    seq = sequence.strip().upper()
    if not seq:
        raise ValueError("Cannot encode empty amino-acid sequence")
    out = np.empty(len(seq), dtype=np.uint8)
    for idx, ch in enumerate(seq):
        tok = AA_TO_ID.get(ch)
        if tok is None:
            raise ValueError(f"Non-canonical amino acid {ch!r} at position {idx}")
        out[idx] = tok
    return out


def decode_aa_tokens(tokens: np.ndarray | Sequence[int]) -> str:
    """Decode uint8/int token indices in [0, 19] back into a canonical AA string."""
    chars: list[str] = []
    for tok in tokens:
        t = int(tok)
        if t == PAD_TOKEN_ID:
            break
        if not (0 <= t < VOCAB_AA):
            raise ValueError(f"Cannot decode non-AA token id {t}")
        chars.append(AA20[t])
    return "".join(chars)


@dataclass(frozen=True)
class PackedSequenceStore:
    """Compact contiguous uint8 sequence store (~1 byte/residue) with O(1) indexing."""

    tokens_flat: np.ndarray  # shape (total_tokens,), uint8 in 0..19
    offsets: np.ndarray      # shape (n_sequences + 1,), int64
    lengths: np.ndarray      # shape (n_sequences,), int32

    def __post_init__(self) -> None:
        if self.tokens_flat.ndim != 1 or self.tokens_flat.dtype != np.uint8:
            raise ValueError("tokens_flat must be 1D uint8")
        if self.offsets.ndim != 1 or self.lengths.ndim != 1:
            raise ValueError("offsets and lengths must be 1D")
        if len(self.offsets) != len(self.lengths) + 1:
            raise ValueError("offsets length must equal len(lengths) + 1")
        if len(self.lengths) > 0 and int(self.offsets[-1]) != int(self.tokens_flat.size):
            raise ValueError("offsets[-1] must match tokens_flat.size")

    @property
    def n_sequences(self) -> int:
        return int(self.lengths.size)

    @property
    def total_tokens(self) -> int:
        return int(self.tokens_flat.size)

    def get_tokens(self, index: int) -> np.ndarray:
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return self.tokens_flat[start:end]

    def get_sequence(self, index: int) -> str:
        return decode_aa_tokens(self.get_tokens(index))

    def collate_indices(
        self,
        indices: Sequence[int] | np.ndarray,
        *,
        device: torch.device | str | None = None,
        pad_to_multiple: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx_arr = np.asarray(indices, dtype=np.int64)
        bsz = int(idx_arr.size)
        if bsz == 0:
            raise ValueError("Cannot collate empty index batch")
        lens = self.lengths[idx_arr]
        max_len = int(lens.max())
        if pad_to_multiple > 1:
            padded_len = ((max_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        else:
            padded_len = max_len
        buf = np.full((bsz, padded_len), PAD_TOKEN_ID, dtype=np.uint8)
        for row_i, seq_idx in enumerate(idx_arr):
            s = int(self.offsets[seq_idx])
            e = int(self.offsets[seq_idx + 1])
            buf[row_i, : e - s] = self.tokens_flat[s:e]
        tokens_t = torch.from_numpy(buf).to(device=device, dtype=torch.long)
        lengths_t = torch.from_numpy(lens.astype(np.int64, copy=False)).to(
            device=device, dtype=torch.long
        )
        return tokens_t, lengths_t

    @classmethod
    def from_sequences(cls, sequences: Sequence[str]) -> PackedSequenceStore:
        encoded = [encode_aa_sequence(seq) for seq in sequences]
        return cls.from_encoded(encoded)

    @classmethod
    def from_encoded(cls, encoded: Sequence[np.ndarray]) -> PackedSequenceStore:
        n_seq = len(encoded)
        lengths = np.asarray([len(arr) for arr in encoded], dtype=np.int32)
        offsets = np.zeros(n_seq + 1, dtype=np.int64)
        if n_seq > 0:
            np.cumsum(lengths, out=offsets[1:])
            tokens_flat = np.concatenate(encoded).astype(np.uint8, copy=False)
        else:
            tokens_flat = np.zeros(0, dtype=np.uint8)
        return cls(tokens_flat=tokens_flat, offsets=offsets, lengths=lengths)


def collate_variable_length_sequences(
    sequences: Sequence[str | np.ndarray],
    *,
    device: torch.device | str | None = None,
    pad_to_multiple: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate variable-length AA sequences into padded (tokens, lengths) tensors."""
    if not sequences:
        raise ValueError("Cannot collate empty sequence list")
    encoded = [
        encode_aa_sequence(s) if isinstance(s, str) else np.asarray(s, dtype=np.uint8)
        for s in sequences
    ]
    lens = np.asarray([len(a) for a in encoded], dtype=np.int64)
    max_len = int(lens.max())
    if pad_to_multiple > 1:
        padded_len = ((max_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    else:
        padded_len = max_len
    buf = np.full((len(encoded), padded_len), PAD_TOKEN_ID, dtype=np.uint8)
    for idx, arr in enumerate(encoded):
        buf[idx, : len(arr)] = arr
    tokens_t = torch.from_numpy(buf).to(device=device, dtype=torch.long)
    lengths_t = torch.from_numpy(lens).to(device=device, dtype=torch.long)
    return tokens_t, lengths_t


@dataclass(frozen=True)
class PreparedBatch:
    """Model inputs, targets, valid/loss masks, and per-sequence loss counts."""

    objective: str
    input_ids: torch.Tensor       # (B, T) long in 0..22
    attention_mask: torch.Tensor  # (B, T) bool: True for valid real tokens, False for <PAD>
    targets: torch.Tensor         # (B, T) long in 0..19 at loss_mask positions (PAD elsewhere)
    loss_mask: torch.Tensor       # (B, T) bool: True strictly at supervised real AA positions
    n_loss_per_seq: torch.Tensor  # (B,) long: exact supervised count per sequence (>= 1)
    lengths: torch.Tensor         # (B,) long: real sequence lengths L_i


def prepare_ar_batch(
    batch_tokens: torch.Tensor,
    lengths: torch.Tensor,
) -> PreparedBatch:
    """Construct right-padded causal AR batch: input=[<BOS>, a_0..a_{L-2}, <PAD>..], target=[a_0..a_{L-1}]."""
    bsz, max_len = batch_tokens.shape
    device = batch_tokens.device
    pos = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)
    valid_mask = pos < lengths.unsqueeze(1)

    targets = torch.where(valid_mask, batch_tokens.to(dtype=torch.long), PAD_TOKEN_ID)
    input_ids = torch.full((bsz, max_len), PAD_TOKEN_ID, dtype=torch.long, device=device)
    input_ids[:, 0] = BOS_TOKEN_ID
    if max_len > 1:
        shifted_valid = pos[:, 1:] < lengths.unsqueeze(1)
        input_ids[:, 1:] = torch.where(
            shifted_valid,
            batch_tokens[:, :-1].to(dtype=torch.long),
            PAD_TOKEN_ID,
        )

    return PreparedBatch(
        objective="ar",
        input_ids=input_ids,
        attention_mask=valid_mask,
        targets=targets,
        loss_mask=valid_mask,
        n_loss_per_seq=lengths.to(dtype=torch.long),
        lengths=lengths.to(dtype=torch.long),
    )


def _sample_uniform_subset_mask(
    lengths: torch.Tensor,
    n_mask: torch.Tensor,
    max_len: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Uniformly sample exactly n_mask[i] distinct positions strictly inside [0, lengths[i])."""
    bsz = int(lengths.shape[0])
    device = lengths.device
    pos = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)
    valid = pos < lengths.unsqueeze(1)

    # Assign random priorities in [0, 1) to valid positions and +2.0 to <PAD> positions
    # so argsort NEVER places a <PAD> position before any valid position.
    noise = torch.rand((bsz, max_len), device=device, generator=generator, dtype=torch.float32)
    priorities = torch.where(valid, noise, torch.full_like(noise, 2.0))
    order = torch.argsort(priorities, dim=-1)

    mask = torch.zeros((bsz, max_len), dtype=torch.bool, device=device)
    mask.scatter_(1, order, pos < n_mask.unsqueeze(1))
    # Explicit safety intersection with valid positions
    return mask & valid


def prepare_mlm15_batch(
    batch_tokens: torch.Tensor,
    lengths: torch.Tensor,
    *,
    mask_fraction: float = 0.15,
    generator: torch.Generator | None = None,
) -> PreparedBatch:
    """Dynamic per-sequence 15% MLM batch with k_i = max(1, round(0.15 * L_i)) masked real AAs."""
    bsz, max_len = batch_tokens.shape
    device = batch_tokens.device
    pos = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)
    valid_mask = pos < lengths.unsqueeze(1)

    k_mask = torch.clamp(
        torch.round(lengths.to(dtype=torch.float32) * float(mask_fraction)).to(dtype=torch.long),
        min=1,
    )
    k_mask = torch.minimum(k_mask, lengths.to(dtype=torch.long))
    loss_mask = _sample_uniform_subset_mask(
        lengths, k_mask, max_len, generator=generator
    )

    targets = torch.where(valid_mask, batch_tokens.to(dtype=torch.long), PAD_TOKEN_ID)
    input_ids = torch.where(loss_mask, MASK_TOKEN_ID, targets)

    return PreparedBatch(
        objective="mlm_15",
        input_ids=input_ids,
        attention_mask=valid_mask,
        targets=targets,
        loss_mask=loss_mask,
        n_loss_per_seq=k_mask,
        lengths=lengths.to(dtype=torch.long),
    )


def prepare_oa_uniform_batch(
    batch_tokens: torch.Tensor,
    lengths: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> PreparedBatch:
    """Stratified per-sequence OA-Uniform batch with m_i uniform over 1..L_i real AA positions."""
    bsz, max_len = batch_tokens.shape
    device = batch_tokens.device
    pos = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)
    valid_mask = pos < lengths.unsqueeze(1)

    perm = torch.randperm(bsz, device=device, generator=generator, dtype=torch.long).float()
    jitter = torch.rand(bsz, device=device, generator=generator, dtype=torch.float32)
    u = (perm + jitter) / float(bsz)
    m_mask = torch.clamp(
        torch.floor(u * lengths.to(dtype=torch.float32)).to(dtype=torch.long) + 1,
        min=1,
    )
    m_mask = torch.minimum(m_mask, lengths.to(dtype=torch.long))

    loss_mask = _sample_uniform_subset_mask(
        lengths, m_mask, max_len, generator=generator
    )
    targets = torch.where(valid_mask, batch_tokens.to(dtype=torch.long), PAD_TOKEN_ID)
    input_ids = torch.where(loss_mask, MASK_TOKEN_ID, targets)

    return PreparedBatch(
        objective="oa_uniform",
        input_ids=input_ids,
        attention_mask=valid_mask,
        targets=targets,
        loss_mask=loss_mask,
        n_loss_per_seq=m_mask,
        lengths=lengths.to(dtype=torch.long),
    )


def prepare_objective_batch(
    objective: str,
    batch_tokens: torch.Tensor,
    lengths: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> PreparedBatch:
    """Dispatch variable-length batch preparation to ar, mlm_15, or oa_uniform."""
    if objective == "ar":
        return prepare_ar_batch(batch_tokens, lengths)
    if objective == "mlm_15":
        return prepare_mlm15_batch(batch_tokens, lengths, generator=generator)
    if objective == "oa_uniform":
        return prepare_oa_uniform_batch(batch_tokens, lengths, generator=generator)
    raise ValueError(f"Unsupported objective {objective!r}; expected {OBJECTIVES}")


def compute_objective_loss(
    logits: torch.Tensor,
    prepared: PreparedBatch,
) -> torch.Tensor:
    """Compute exact fp32 objective loss in nats/residue with strict <PAD> exclusion.

    - ``ar``: token-averaged CE strictly over the L_i real AA positions.
    - ``mlm_15``: token-averaged CE strictly over the k_i masked real AA positions.
    - ``oa_uniform``: OA-ARDM per-sequence weighted CE:
      ``mean_i( (1 / m_i) * sum_{j in M_i} CE_j )``, exact unbiased per-residue NELBO.
    """
    if logits.shape[-1] != VOCAB_AA:
        raise ValueError(f"Expected output vocab {VOCAB_AA}, got {logits.shape[-1]}")

    safe_targets = torch.where(
        prepared.loss_mask,
        prepared.targets,
        torch.zeros_like(prepared.targets),
    )
    tok_ce = F.cross_entropy(
        logits.float().transpose(1, 2),
        safe_targets,
        reduction="none",
    )
    masked_ce = torch.where(prepared.loss_mask, tok_ce, torch.zeros_like(tok_ce))

    if prepared.objective == "oa_uniform":
        per_seq = masked_ce.sum(dim=-1) / prepared.n_loss_per_seq.to(dtype=torch.float32)
        weights = prepared.lengths.to(dtype=torch.float32)
        return (per_seq * weights).sum() / weights.sum().clamp(min=1.0)

    if prepared.objective in ("ar", "mlm_15"):
        denom = prepared.loss_mask.sum().to(dtype=torch.float32).clamp(min=1.0)
        return masked_ce.sum() / denom

    raise ValueError(f"Unsupported objective {prepared.objective!r}")
