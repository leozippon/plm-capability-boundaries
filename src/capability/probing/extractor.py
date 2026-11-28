"""Exact layerwise hidden-state extractor for Stage-2 VariableLengthProteinTransformer checkpoints.

Supports:
  - Frozen extraction of post-embedding (``l = 0``, ``"emb"``), post-block
    (``l = 1 .. n_layers``, ``"block_1" .. "block_N"``), and final RMSNorm
    (``l = n_layers + 1``, ``"norm_f"``) hidden states ``h^(l)(x)`` of shape ``(L_seq, d)``.
  - Strict variable-length causal alignment for ``ar`` (prepending ``<BOS>`` and slicing
    positions ``1 .. L_seq`` so ``h^(l)_i`` represents residue ``i`` after observing
    ``<BOS>, a_0, ..., a_i``, while ``logits[:, :L_seq, :]`` gives exact causal next-token
    predictions for ``a_0 .. a_{L-1}`` with zero ``<PAD>`` leakage).
  - Strict variable-length bidirectional alignment for ``mlm_15`` and ``oa_uniform`` (``MDM``),
    blocking all ``<PAD>`` keys in SDPA so batched padded extraction matches single-sequence
    extraction identically.
  - Flexible loading of ``_final.pt``, ``_latest.pt``, and milestone ``.pt`` checkpoints.
"""
from __future__ import annotations

import contextlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.capability.supply_demand.model import VariableLengthProteinTransformer
from src.capability.supply_demand.objectives import encode_aa_sequence
from src.capability.supply_demand.spec import (
    BOS_TOKEN_ID,
    DEFAULT_MAX_SEQ_LEN,
    OBJECTIVES,
    PAD_TOKEN_ID,
    SCALE_SPECS,
)

PARADIGM_DISPLAY: dict[str, str] = {
    "ar": "AR",
    "mlm_15": "MLM-15%",
    "oa_uniform": "MDM",
}

_CKPT_EPOCH_RE = re.compile(r"_e(\d+)(?:\.pt)?$", re.IGNORECASE)


def canonical_objective(raw: str) -> str:
    """Normalize objective aliases (e.g. ``mdm`` / ``MDM`` -> ``oa_uniform``)."""
    s = str(raw).strip().lower().replace("-", "_")
    if s in ("mdm", "oa_ardm", "oa_uniform"):
        return "oa_uniform"
    if s in ("mlm", "mlm15", "mlm_15"):
        return "mlm_15"
    if s == "ar":
        return "ar"
    if s in OBJECTIVES:
        return s
    raise ValueError(
        f"Unsupported objective {raw!r}; expected one of ('ar', 'mlm_15', 'oa_uniform', 'mdm')"
    )


def paradigm_display_name(objective: str) -> str:
    """Return canonical manuscript display symbol (``AR``, ``MLM-15%``, ``MDM``)."""
    return PARADIGM_DISPLAY[canonical_objective(objective)]


def get_layer_names(n_layers: int) -> list[str]:
    """Return ordered names for all ``n_layers + 2`` extracted hidden-state taps."""
    names = ["emb"]
    names.extend(f"block_{i}" for i in range(1, int(n_layers) + 1))
    names.append("norm_f")
    return names


def _autocast_ctx(device: torch.device, precision: str) -> Any:
    if precision == "bf16" and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _strip_compiled_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            out[k[len("_orig_mod.") :]] = v
        elif k.startswith("module."):
            out[k[len("module.") :]] = v
        else:
            out[k] = v
    return out


def _infer_scale_id_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> str:
    emb_w = state_dict.get("tok_emb.weight")
    if emb_w is None:
        raise ValueError("Checkpoint state_dict missing tok_emb.weight; cannot infer scale_id")
    d_model = int(emb_w.shape[1])
    block_indices = set()
    for key in state_dict:
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.add(int(parts[1]))
    n_layers = len(block_indices)
    for scale_id, spec in SCALE_SPECS.items():
        if spec.d_model == d_model and spec.n_layers == n_layers:
            return scale_id
    raise ValueError(
        f"Could not match (n_layers={n_layers}, d_model={d_model}) to any known SCALE_SPECS"
    )


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Loaded frozen Stage-2 model plus normalized checkpoint metadata."""

    model: VariableLengthProteinTransformer
    checkpoint_path: Path
    checkpoint_id: str
    run_id: str
    arm_id: str
    objective: str
    paradigm_display: str
    scale_id: str
    epoch: int
    global_step: int
    tokens_seen: int
    n_layers: int
    d_model: int
    layer_names: tuple[str, ...]
    run_spec: dict[str, Any]


def load_stage2_checkpoint(
    checkpoint_path: Path | str,
    *,
    device: torch.device | str = "cpu",
    objective_override: str | None = None,
    scale_id_override: str | None = None,
) -> LoadedCheckpoint:
    """Load a Stage-2 ``_final.pt``, ``_latest.pt``, or milestone ``.pt`` into eval mode."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    dev = torch.device(device)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected dict checkpoint payload at {path}, got {type(payload).__name__}")

    raw_sd = payload.get("model_state_dict")
    if raw_sd is None:
        if "tok_emb.weight" in payload:
            raw_sd = payload
        else:
            raise ValueError(f"Checkpoint {path} missing 'model_state_dict'")
    state_dict = _strip_compiled_prefix(raw_sd)

    run_spec = dict(payload.get("run_spec") or {})
    raw_obj = objective_override or run_spec.get("objective") or payload.get("objective")
    if raw_obj is None:
        stem_low = path.stem.lower()
        if "_ar_" in stem_low:
            raw_obj = "ar"
        elif "_mlm_15_" in stem_low or "_mlm15_" in stem_low:
            raw_obj = "mlm_15"
        elif "_oa_uniform_" in stem_low or "_mdm_" in stem_low:
            raw_obj = "oa_uniform"
        else:
            raise ValueError(f"Cannot infer objective for checkpoint {path}; pass objective_override")
    objective = canonical_objective(str(raw_obj))

    scale_id = str(
        scale_id_override
        or run_spec.get("scale_id")
        or payload.get("scale_id")
        or _infer_scale_id_from_state_dict(state_dict)
    )
    if scale_id not in SCALE_SPECS:
        raise ValueError(f"Unknown scale_id {scale_id!r} in checkpoint {path}")

    max_seq_len = int(run_spec.get("max_seq_len", DEFAULT_MAX_SEQ_LEN))
    model = VariableLengthProteinTransformer(
        SCALE_SPECS[scale_id],
        objective,
        max_seq_len=max_seq_len,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    run_id = str(run_spec.get("run_id") or payload.get("run_id") or path.stem)
    arm_id = str(run_spec.get("arm_id") or payload.get("arm_id") or "unknown")

    epoch_val = payload.get("epoch")
    if epoch_val is None:
        m = _CKPT_EPOCH_RE.search(path.stem)
        epoch_val = int(m.group(1)) if m else int(run_spec.get("max_epochs", 0))
    epoch = int(epoch_val)
    global_step = int(payload.get("global_step", 0))
    tokens_seen = int(payload.get("tokens_seen", 0))

    spec = SCALE_SPECS[scale_id]
    return LoadedCheckpoint(
        model=model,
        checkpoint_path=path,
        checkpoint_id=path.stem,
        run_id=run_id,
        arm_id=arm_id,
        objective=objective,
        paradigm_display=paradigm_display_name(objective),
        scale_id=scale_id,
        epoch=epoch,
        global_step=global_step,
        tokens_seen=tokens_seen,
        n_layers=spec.n_layers,
        d_model=spec.d_model,
        layer_names=tuple(get_layer_names(spec.n_layers)),
        run_spec=run_spec,
    )


def prepare_probe_input_batch(
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    *,
    objective: str,
) -> tuple[torch.Tensor, torch.Tensor | None, bool, int]:
    """Prepare exact model inputs for hidden-state extraction.

    Returns ``(input_ids, attention_mask, right_padded, residue_offset)``:
      - ``ar``: ``input_ids = [<BOS>, a_0, ..., a_{L-1}, <PAD>, ...]`` of width ``max_len + 1``.
        Position ``i + 1`` (``residue_offset = 1``) holds residue ``a_i``'s causal state
        having observed ``<BOS>, a_0, ..., a_i``, while ``logits[:, :L, :]`` holds the exact
        causal next-token predictions for ``a_0 .. a_{L-1}``.
      - ``mlm_15`` / ``oa_uniform`` (``MDM``): ``input_ids = [a_0, ..., a_{L-1}, <PAD>, ...]``
        with ``attention_mask = (pos < L)`` blocking ``<PAD>`` keys in SDPA (``residue_offset = 0``).
    """
    obj = canonical_objective(objective)
    bsz, max_len = tokens.shape
    device = tokens.device
    pos = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)
    valid = pos < lengths.to(dtype=torch.long).unsqueeze(1)

    if obj == "ar":
        out_len = max_len + 1
        input_ids = torch.full((bsz, out_len), PAD_TOKEN_ID, dtype=torch.long, device=device)
        input_ids[:, 0] = BOS_TOKEN_ID
        input_ids[:, 1:] = torch.where(valid, tokens.to(dtype=torch.long), PAD_TOKEN_ID)
        # Strictly right-padded causal batch: valid tokens occupy [0, L_i], so native causal SDPA
        # has zero <PAD> leakage at every position t <= L_i.
        pos_ext = torch.arange(out_len, device=device, dtype=torch.long).unsqueeze(0)
        attn_mask = pos_ext < (lengths.to(dtype=torch.long) + 1).unsqueeze(1)
        return input_ids, attn_mask, True, 1

    input_ids = torch.where(valid, tokens.to(dtype=torch.long), PAD_TOKEN_ID)
    return input_ids, valid, True, 0


@torch.no_grad()
def forward_layerwise_hidden_states(
    model: VariableLengthProteinTransformer,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    *,
    objective: str | None = None,
    precision: str = "bf16",
    layer_indices: Sequence[int] | None = None,
    return_logits: bool = False,
    output_dtype: torch.dtype = torch.float32,
) -> tuple[list[torch.Tensor], torch.Tensor | None]:
    """Run one frozen forward pass and return residue-aligned hidden states per requested layer.

    Each returned hidden tensor has shape ``(B, max_len, d_model)`` where ``max_len = tokens.shape[1]``
    and positions ``0 .. lengths[b] - 1`` align strictly to residues ``a_0 .. a_{L_b - 1}`` (already
    stripped of the ``<BOS>`` offset for ``ar``). Positions ``>= lengths[b]`` are zeroed out.

    If ``return_logits=True``, returns ``logits`` of shape ``(B, max_len, 20)`` in fp32:
      - For ``ar``: ``logits[:, t, :]`` predicts residue ``a_t`` from ``[<BOS>, a_0, ..., a_{t-1}]``.
      - For ``mlm_15`` / ``oa_uniform``: ``logits[:, t, :]`` is the unmasked logit at position ``t``.
    """
    obj = canonical_objective(objective if objective is not None else model.objective)
    n_total_taps = len(model.blocks) + 2  # 0 (emb), 1..n_layers (blocks), n_layers+1 (norm_f)
    if layer_indices is None:
        sel_set = set(range(n_total_taps))
        sel_order = list(range(n_total_taps))
    else:
        sel_order = [int(i) for i in layer_indices]
        for idx in sel_order:
            if not (0 <= idx < n_total_taps):
                raise ValueError(f"Layer index {idx} out of range [0, {n_total_taps - 1}]")
        sel_set = set(sel_order)

    bsz, max_len = tokens.shape
    device = next(model.parameters()).device
    tokens = tokens.to(device=device, dtype=torch.long)
    lengths = lengths.to(device=device, dtype=torch.long)

    input_ids, attn_mask, right_padded, offset = prepare_probe_input_batch(
        tokens, lengths, objective=obj
    )
    cur_seq_len = input_ids.shape[1]
    pos_res = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)
    res_valid_f = (pos_res < lengths.unsqueeze(1)).unsqueeze(-1)

    collected: dict[int, torch.Tensor] = {}
    was_training = model.training
    model.eval()

    def _slice_residue(tensor: torch.Tensor) -> torch.Tensor:
        sliced = tensor[:, offset : offset + max_len, :].to(dtype=output_dtype)
        return torch.where(res_valid_f, sliced, torch.zeros_like(sliced))

    try:
        with _autocast_ctx(device, precision):
            cos, sin = model._get_rope_tables(cur_seq_len, device)
            sdpa_mask = model._build_sdpa_mask(
                input_ids, attn_mask, right_padded=right_padded
            )
            x = model.tok_emb(input_ids)
            if 0 in sel_set:
                collected[0] = _slice_residue(x)

            for blk_idx, block in enumerate(model.blocks, start=1):
                x = block(x, cos, sin, sdpa_mask=sdpa_mask)
                if blk_idx in sel_set:
                    collected[blk_idx] = _slice_residue(x)

            h = model.norm_f(x)
            norm_idx = len(model.blocks) + 1
            if norm_idx in sel_set:
                collected[norm_idx] = _slice_residue(h)

        logits_out: torch.Tensor | None = None
        if return_logits:
            # For AR: positions 0..max_len-1 predict a_0..a_{max_len-1}.
            # For bidirectional: positions 0..max_len-1 correspond directly to a_0..a_{max_len-1}.
            h_pred = h[:, :max_len, :]
            amp_device = "cuda" if h_pred.is_cuda else "cpu"
            with torch.amp.autocast(amp_device, enabled=False):
                logits_out = F.linear(h_pred.float(), model.lm_head.weight.float())
    finally:
        if was_training:
            model.train()

    return [collected[i] for i in sel_order], logits_out


@torch.no_grad()
def extract_layerwise_hidden_states(
    model: VariableLengthProteinTransformer,
    sequences: Sequence[str | np.ndarray],
    *,
    objective: str | None = None,
    batch_size: int = 64,
    precision: str = "bf16",
    layer_indices: Sequence[int] | None = None,
    pad_to_multiple: int = 8,
    output_dtype: np.dtype[Any] | type[np.floating[Any]] = np.float32,
) -> list[list[np.ndarray]]:
    """Extract variable-length residue-aligned hidden states for a sequence list.

    Returns a nested list ``result[seq_idx][tap_pos]`` where each array has exact shape
    ``(L_seq, d_model)`` (no padding) in ``output_dtype``.
    """
    if not sequences:
        return []
    obj = canonical_objective(objective if objective is not None else model.objective)
    n_taps = len(model.blocks) + 2 if layer_indices is None else len(layer_indices)
    encoded = [
        encode_aa_sequence(s) if isinstance(s, str) else np.asarray(s, dtype=np.uint8)
        for s in sequences
    ]
    n_seq = len(encoded)
    lengths_all = np.asarray([len(a) for a in encoded], dtype=np.int64)
    # Sort by length to minimize padding overhead, then restore original sequence order
    order = np.argsort(lengths_all, kind="mergesort")
    out: list[list[np.ndarray]] = [[] for _ in range(n_seq)]
    device = next(model.parameters()).device

    for start in range(0, n_seq, max(1, int(batch_size))):
        chunk_idx = order[start : start + int(batch_size)]
        bsz = int(chunk_idx.size)
        chunk_lens = lengths_all[chunk_idx]
        max_len = int(chunk_lens.max())
        if pad_to_multiple > 1:
            padded_len = ((max_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        else:
            padded_len = max_len
        buf = np.full((bsz, padded_len), PAD_TOKEN_ID, dtype=np.uint8)
        for row_i, orig_i in enumerate(chunk_idx):
            arr = encoded[int(orig_i)]
            buf[row_i, : len(arr)] = arr

        tok_t = torch.from_numpy(buf).to(device=device, dtype=torch.long)
        len_t = torch.from_numpy(chunk_lens).to(device=device, dtype=torch.long)
        layer_tensors, _ = forward_layerwise_hidden_states(
            model,
            tok_t,
            len_t,
            objective=obj,
            precision=precision,
            layer_indices=layer_indices,
            return_logits=False,
            output_dtype=torch.float32,
        )
        layer_np = [t.cpu().numpy() for t in layer_tensors]
        for row_i, orig_i in enumerate(chunk_idx):
            L_i = int(chunk_lens[row_i])
            out[int(orig_i)] = [
                np.asarray(layer_np[tap][row_i, :L_i, :], dtype=output_dtype)
                for tap in range(n_taps)
            ]
    return out
