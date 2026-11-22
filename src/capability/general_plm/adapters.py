"""Unified, fail-fast model adapters for all general-PLM families.

Design invariants:
1. Offline-only (``local_files_only=True``) — zero network access at load time.
2. FP32 by default with ``allow_tf32=False`` via ``fp32_matmul_context``;
   explicit ``--dtype bf16`` opt-in is permitted and recorded in provenance.
   FP16 is refused on ESM2/DPLM and any arm that does not explicitly allow it.
3. Exact numerical parity (``<1e-6`` nats) with
   ``src.capability.position.position_likelihood.MaskedScorer`` for ESM2/DPLM.
4. Explicit support for SaProt both in sequence-only ``#``-masked mode
   (marginalizing over 20 3Di tokens per AA) and Foldseek-3Di-supplied mode.
5. Reuses ``src.capability.models.progen3`` for ProGen3 checkpoints (never
   calling raw ``from_pretrained`` which leaves megablocks experts random!).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..core.amino_acids import AA20
from ..core.precision_policy import fp32_matmul_context, requested_fp32_matmul_policy
from .msa_utils import build_query_msa
from .registry import GeneralPLMSpec, get_spec, resolve_checkpoint_dir

AA20_SET = frozenset(AA20)
AA20_INDEX = {aa: idx for idx, aa in enumerate(AA20)}


@dataclass(frozen=True)
class PositionVector:
    """Per-position NLL vector and exact scalar summary for one sequence."""

    terms: np.ndarray  # float32, shape (L,) or (T_scored,)
    nll_sum: float
    pll: float  # -nll_sum (higher is more likely)
    paradigm: str


def parse_mutation(mutation: str, wildtype: str) -> list[tuple[int, str, str]]:
    """Parse ``A30V`` or ``A30V:K45R`` into 0-based ``(pos, wt, mut)`` triples."""
    triples: list[tuple[int, str, str]] = []
    seen_pos: set[int] = set()
    for token in mutation.strip().split(":"):
        token = token.strip()
        if len(token) < 3:
            raise ValueError(f"Invalid mutation token {token!r} in {mutation!r}")
        wt_aa, mut_aa = token[0], token[-1]
        pos = int(token[1:-1]) - 1
        if not 0 <= pos < len(wildtype):
            raise ValueError(f"{mutation}: position {pos + 1} outside length {len(wildtype)}")
        if wildtype[pos] != wt_aa:
            raise ValueError(
                f"{mutation}: wild-type residue at {pos + 1} is {wildtype[pos]!r}, expected {wt_aa!r}"
            )
        if mut_aa not in AA20_SET:
            raise ValueError(f"{mutation}: non-canonical mutant residue {mut_aa!r}")
        if pos in seen_pos:
            raise ValueError(f"{mutation}: duplicate mutated position {pos + 1}")
        seen_pos.add(pos)
        triples.append((pos, wt_aa, mut_aa))
    return triples


def apply_mutation(wildtype: str, mutation: str) -> str:
    """Apply a verified colon-separated substitution string to ``wildtype``."""
    chars = list(wildtype)
    for pos, _wt, mut in parse_mutation(mutation, wildtype):
        chars[pos] = mut
    return "".join(chars)


def normalize_dtype_name(spec: GeneralPLMSpec, dtype: str | None) -> str:
    """Resolve requested precision and enforce fail-fast precision guardrails."""
    raw = (dtype or "").strip().lower()
    if raw in ("", "auto", "default"):
        resolved = spec.default_dtype
    elif raw in ("fp32", "float32"):
        resolved = "float32"
    elif raw in ("bf16", "bfloat16"):
        resolved = "bfloat16"
    elif raw in ("fp16", "float16"):
        resolved = "float16"
    else:
        raise ValueError(f"{spec.arm}: unsupported dtype {dtype!r}; use float32 or bfloat16")

    if resolved == "float16" and not spec.allow_fp16:
        raise ValueError(
            f"{spec.arm}: float16 is refused (overflow risk); use float32 or explicit bfloat16"
        )
    if spec.family == "progen3" and resolved == "float32":
        raise ValueError(
            f"{spec.arm}: ProGen3 flash SDPA has no float32 kernel; use bfloat16"
        )
    return resolved


def torch_dtype_of(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype name: {name!r}")
    return mapping[name]


def _from_pretrained_dtype_kwarg(loader_cls: Any, torch_dt: torch.dtype) -> dict[str, Any]:
    """Use ``torch_dtype`` for transformers 4.x compatibility; fall back to ``dtype`` only if removed."""
    try:
        sig = inspect.signature(loader_cls.from_pretrained)
        if "torch_dtype" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ):
            return {"torch_dtype": torch_dt}
    except (TypeError, ValueError):
        pass
    return {"dtype": torch_dt}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _checkpoint_inventory(checkpoint_dir: Path) -> dict[str, Any]:
    config_path = checkpoint_dir / "config.json"
    tensors = [
        {"name": item.name, "bytes": int(item.stat().st_size)}
        for item in sorted(checkpoint_dir.iterdir())
        if item.is_file() and item.suffix in (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
    ]
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "config_sha256": _sha256_file(config_path) if config_path.is_file() else None,
        "tensor_files": tensors,
    }


def _discover_transformer_blocks(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    """Locate the ordered transformer layer stack for PyTorch forward hooks."""
    candidates = [
        ("esm", "encoder", "layer"),
        ("pro_prime", "encoder", "layer"),
        ("proprime", "encoder", "layer"),
        ("esmc", "layers"),
        ("encoder", "layer"),
        ("transformer_encoder",),
        ("model", "layers"),
        ("transformer", "encoder", "layers"),
        ("transformer", "layers"),
        ("transformer", "blocks"),
        ("transformer", "h"),
        ("layers",),
        ("blocks",),
        ("backbone", "layers"),
    ]
    for chain in candidates:
        obj: Any = model
        ok = True
        for attr in chain:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, (torch.nn.ModuleList, list, tuple)) and len(obj) > 0:
            return obj
    raise AttributeError(
        f"Cannot locate transformer block stack on {type(model).__name__}; "
        f"top-level children: {[name for name, _ in model.named_children()]}"
    )


def _nucleus_logprobs(logprobs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus truncation in log-probability space, matching ``ladder/runtime.py``."""
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must lie in (0, 1], got {top_p}")
    if top_p >= 1.0:
        return logprobs
    ordered, order = torch.sort(logprobs, descending=True)
    probs = ordered.exp()
    cumulative = torch.cumsum(probs, dim=-1)
    keep = (cumulative - probs) < top_p
    keep[..., 0] = True
    neg_inf = torch.full_like(ordered, float("-inf"))
    masked = torch.full_like(logprobs, float("-inf"))
    masked.scatter_(-1, order, torch.where(keep, ordered, neg_inf))
    return F.log_softmax(masked, dim=-1)


def _realised_dtype(model: torch.nn.Module) -> str:
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    return "/".join(dtypes) if dtypes else "none"


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# ----------------------------------------------------------------- Base class


class GeneralPLMAdapter:
    """Common interface for scoring, representation extraction, and sampling."""

    def __init__(
        self,
        spec: GeneralPLMSpec,
        checkpoint_dir: Path,
        *,
        device: str,
        dtype: str,
        model: torch.nn.Module,
        tokenizer: Any,
        extra_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.spec = spec
        self.arm = spec.arm
        self.checkpoint_dir = checkpoint_dir
        self.device = str(device)
        self.dtype = dtype
        self.torch_dtype = torch_dtype_of(dtype)
        self.model = model
        self.tokenizer = tokenizer
        self._extra_provenance = dict(extra_provenance or {})

    def _precision_ctx(self):
        if self.dtype == "float32" and self.device.startswith("cuda"):
            return fp32_matmul_context(torch)
        return nullcontext()

    def validate_sequence(self, sequence: str) -> None:
        if not sequence:
            raise ValueError(f"{self.arm}: empty sequence")
        bad = sorted(set(sequence) - AA20_SET)
        if bad:
            raise ValueError(f"{self.arm}: non-AA20 symbols {bad} in sequence")
        if len(sequence) > self.spec.max_residues:
            raise ValueError(
                f"{self.arm}: length {len(sequence)} exceeds max_residues={self.spec.max_residues}"
            )

    def provenance(self) -> dict[str, Any]:
        import transformers

        return {
            "arm": self.arm,
            "family": self.spec.family,
            "paradigm": self.spec.paradigm,
            "loader_kind": self.spec.loader_kind,
            "hf_repo": self.spec.hf_repo,
            "dtype_requested": self.dtype,
            "dtype_realised": _realised_dtype(self.model),
            "fp32_matmul_policy": (
                requested_fp32_matmul_policy()
                if self.dtype == "float32"
                else {"declared_non_fp32_opt_in": self.dtype}
            ),
            "parameters": _parameter_count(self.model),
            "max_residues": self.spec.max_residues,
            "device": self.device,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            **_checkpoint_inventory(self.checkpoint_dir),
            **self._extra_provenance,
        }

    # --- Abstract scoring/representation/generation API ---

    def exact_pll(
        self,
        sequence: str,
        *,
        batch_size: int = 16,
        structure_3di: str | None = None,
    ) -> PositionVector:
        raise NotImplementedError

    def score_variants(
        self,
        wildtype: str,
        mutations: Sequence[str],
        *,
        batch_size: int = 16,
        multi_exact_pll: bool = False,
        structure_3di: str | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def native_site_conditional(
        self,
        sequence: str,
        index: int,
        *,
        structure_3di: str | None = None,
    ) -> np.ndarray:
        """Return length-20 log-probabilities over AA20 at 0-based ``index``."""
        raise NotImplementedError

    def extract_hidden_states(
        self,
        sequences: Sequence[str],
        *,
        batch_size: int = 8,
        pooling: str = "mean_cls",
        structure_3di_list: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Return ``(N, 4, hidden_dim)`` float32: middle/final x (mean, cls/last)."""
        raise NotImplementedError

    def infill_window(
        self,
        parent: str,
        *,
        start: int,
        extent: int,
        seed: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        structure_3di: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def sample_diffusion_full(
        self,
        length: int,
        *,
        seed: int,
        steps: int | None = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        sampling_mode: str = "max_confidence",
    ) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.arm}: full-length 100%-masked diffusion sampling is not supported"
        )

    def generate_causal_free(
        self,
        n_draws: int,
        *,
        seed: int,
        max_residues: int = 400,
        temperature: float = 1.0,
        top_p: float = 0.95,
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError(
            f"{self.arm}: unconditional free-length causal generation is not supported"
        )

    def score_r4_context(
        self,
        target: str,
        context_items: Sequence[str],
        *,
        budget_tokens: int = 1024,
        batch_size: int = 16,
        aligned_msa: bool = True,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.arm}: R4 multi-sequence context scoring is not supported"
        )


# ------------------------------------------------------- Masked LM base class


class _BaseMaskedAdapter(GeneralPLMAdapter):
    """Shared exact leave-one-out PLL, single-site marginal, hidden-state hook, and infilling."""

    def __init__(
        self,
        spec: GeneralPLMSpec,
        checkpoint_dir: Path,
        *,
        device: str,
        dtype: str,
        model: torch.nn.Module,
        tokenizer: Any,
        mask_id: int,
        aa20_ids: list[int],
        extra_provenance: Mapping[str, Any] | None = None,
        time_first_hidden: bool = False,
        additive_attention_mask: bool = False,
    ) -> None:
        super().__init__(
            spec,
            checkpoint_dir,
            device=device,
            dtype=dtype,
            model=model,
            tokenizer=tokenizer,
            extra_provenance=extra_provenance,
        )
        self.mask_id = int(mask_id)
        self.aa20_ids = [int(x) for x in aa20_ids]
        self.aa_to_id = {aa: tid for aa, tid in zip(AA20, self.aa20_ids)}
        self.time_first_hidden = bool(time_first_hidden)
        self.additive_attention_mask = bool(additive_attention_mask)

    def _encode_sequence(self, sequence: str, structure_3di: str | None = None) -> tuple[list[int], int]:
        """Return ``(token_ids, residue_offset)`` where residue ``j`` is at ``residue_offset + j``."""
        self.validate_sequence(sequence)
        expected = [self.aa_to_id[aa] for aa in sequence]
        length = len(sequence)

        candidates = [
            [int(v) for v in self.tokenizer(sequence)["input_ids"]],
        ]
        if len(candidates[0]) < length:
            candidates.append([int(v) for v in self.tokenizer(" ".join(sequence))["input_ids"]])

        for ids in candidates:
            # Handle [CLS]+seq+[EOS] (offset=1), [CLS]+seq (offset=1), seq+[EOS] (offset=0, e.g. ProteinGLM-MLM), or bare seq (offset=0).
            if len(ids) == length + 2 and ids[1 : 1 + length] == expected:
                return ids, 1
            if len(ids) == length + 1:
                if ids[0:length] == expected:
                    return ids, 0
                if ids[1 : 1 + length] == expected:
                    return ids, 1
            if len(ids) == length and ids == expected:
                return ids, 0

        ids = candidates[0]
        raise ValueError(
            f"{self.arm}: encoded sequence length {len(ids)} (first tokens={ids[:6]}) "
            f"does not match 1-token-per-residue AA20 packing for length {length} "
            f"(expected first tokens={expected[:6]})"
        )

    def _forward_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            if self.additive_attention_mask:
                # AMPLIFY asserts additive float attention mask (0.0 valid, -inf pad), or None when unpadded.
                if bool(torch.all(attention_mask == 1)):
                    kwargs["attention_mask"] = None
                else:
                    kwargs["attention_mask"] = torch.where(
                        attention_mask.bool(),
                        torch.zeros((), dtype=self.torch_dtype, device=input_ids.device),
                        torch.tensor(float("-inf"), dtype=self.torch_dtype, device=input_ids.device),
                    )
            else:
                kwargs["attention_mask"] = attention_mask
        out = self.model(**kwargs)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return logits.float()

    @torch.no_grad()
    def masked_position_logprobs(
        self,
        sequence: str,
        positions: Sequence[int],
        *,
        batch_size: int = 16,
        structure_3di: str | None = None,
    ) -> dict[int, np.ndarray]:
        """Full-vocab log-probabilities at each 0-based position, that position alone masked."""
        ids, offset = self._encode_sequence(sequence, structure_3di=structure_3di)
        ordered = sorted({int(p) for p in positions})
        row = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        out: dict[int, np.ndarray] = {}
        with self._precision_ctx():
            for start in range(0, len(ordered), max(1, int(batch_size))):
                chunk = ordered[start : start + max(1, int(batch_size))]
                batch = row.unsqueeze(0).repeat(len(chunk), 1)
                for idx, pos in enumerate(chunk):
                    batch[idx, offset + pos] = self.mask_id
                logits = self._forward_logits(batch)
                for idx, pos in enumerate(chunk):
                    lp = F.log_softmax(logits[idx, offset + pos], dim=-1)
                    out[pos] = lp.detach().cpu().numpy().astype(np.float32, copy=False)
                del logits
        return out

    @torch.no_grad()
    def exact_pll(
        self,
        sequence: str,
        *,
        batch_size: int = 16,
        structure_3di: str | None = None,
    ) -> PositionVector:
        """Exact deterministic O(L) leave-one-out pseudo-log-likelihood.

        Matches ``src.capability.position.position_likelihood.MaskedScorer.score``
        to <1e-6 nats in FP32 with TF32 disabled.
        """
        ids, offset = self._encode_sequence(sequence, structure_3di=structure_3di)
        length = len(sequence)
        row = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        terms = torch.empty(length, dtype=torch.float32, device=self.device)
        bs = max(1, int(batch_size))
        with self._precision_ctx():
            for start in range(0, length, bs):
                stop = min(start + bs, length)
                batch = row.unsqueeze(0).repeat(stop - start, 1)
                for idx, pos in enumerate(range(start, stop)):
                    batch[idx, offset + pos] = self.mask_id
                logits = self._forward_logits(batch)
                for idx, pos in enumerate(range(start, stop)):
                    lp = F.log_softmax(logits[idx, offset + pos], dim=-1)
                    terms[pos] = -lp[row[offset + pos]]
                del logits
        if not torch.isfinite(terms).all():
            raise FloatingPointError(f"{self.arm}: non-finite PLL term encountered")
        nll_sum = float(terms.sum())
        arr = terms.detach().cpu().numpy().astype(np.float32, copy=False)
        return PositionVector(
            terms=arr,
            nll_sum=nll_sum,
            pll=-nll_sum,
            paradigm=self.spec.paradigm,
        )

    def score_variants(
        self,
        wildtype: str,
        mutations: Sequence[str],
        *,
        batch_size: int = 16,
        multi_exact_pll: bool = False,
        structure_3di: str | None = None,
    ) -> np.ndarray:
        """Score single/multi substitutions: log P(mut) - log P(wt) (higher = better)."""
        parsed = [parse_mutation(m, wildtype) for m in mutations]
        single_indices = [i for i, entry in enumerate(parsed) if len(entry) == 1]
        multi_indices = [i for i, entry in enumerate(parsed) if len(entry) > 1]

        values = np.empty(len(mutations), dtype=np.float64)

        # Single substitutions (and multi-substitutions when additive) share 1 forward per unique site.
        additive_indices = single_indices if multi_exact_pll else list(range(len(mutations)))
        if additive_indices:
            needed_positions = sorted(
                {pos for idx in additive_indices for pos, _, _ in parsed[idx]}
            )
            table = self.masked_position_logprobs(
                wildtype, needed_positions, batch_size=batch_size, structure_3di=structure_3di
            )
            for idx in additive_indices:
                total = 0.0
                for pos, wt_aa, mut_aa in parsed[idx]:
                    row_lp = table[pos]
                    total += float(row_lp[self.aa_to_id[mut_aa]] - row_lp[self.aa_to_id[wt_aa]])
                values[idx] = total

        if multi_exact_pll and multi_indices:
            wt_pll = self.exact_pll(
                wildtype, batch_size=batch_size, structure_3di=structure_3di
            ).pll
            for idx in multi_indices:
                mut_seq = apply_mutation(wildtype, mutations[idx])
                mut_pll = self.exact_pll(
                    mut_seq, batch_size=batch_size, structure_3di=structure_3di
                ).pll
                values[idx] = float(mut_pll - wt_pll)

        if not np.isfinite(values).all():
            raise FloatingPointError(f"{self.arm}: non-finite variant score produced")
        return values

    @torch.no_grad()
    def native_site_conditional(
        self,
        sequence: str,
        index: int,
        *,
        structure_3di: str | None = None,
    ) -> np.ndarray:
        """1-pass masked conditional log p(a | x_{-i}) normalized strictly over AA20."""
        if not 0 <= index < len(sequence):
            raise ValueError(f"index {index} out of bounds for length {len(sequence)}")
        table = self.masked_position_logprobs(
            sequence, [index], batch_size=1, structure_3di=structure_3di
        )
        raw_aa20 = np.asarray([table[index][tid] for tid in self.aa20_ids], dtype=np.float64)
        peak = float(raw_aa20.max())
        lse = peak + math.log(float(np.exp(raw_aa20 - peak).sum()))
        return (raw_aa20 - lse).astype(np.float64)

    @torch.no_grad()
    def extract_hidden_states(
        self,
        sequences: Sequence[str],
        *,
        batch_size: int = 8,
        pooling: str = "mean_cls",
        structure_3di_list: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Extract p=0 (unmasked) middle & final block representations via PyTorch forward hooks."""
        if pooling not in ("mean_cls", "mean_last"):
            raise ValueError(f"Unsupported pooling {pooling!r}; use 'mean_cls' or 'mean_last'")
        if not sequences:
            return np.zeros((0, 4, 0), dtype=np.float32)
        if structure_3di_list is not None and len(structure_3di_list) != len(sequences):
            raise ValueError("structure_3di_list length must match sequences length")

        blocks = _discover_transformer_blocks(self.model)
        mid_idx = (len(blocks) - 1) // 2
        fin_idx = len(blocks) - 1
        pad_id = (
            self.tokenizer.pad_token_id
            if getattr(self.tokenizer, "pad_token_id", None) is not None
            else 0
        )

        captured: dict[str, torch.Tensor] = {}

        def _make_hook(label: str):
            def _hook(_mod, _inp, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[label] = h.detach()
            return _hook

        handles = [
            blocks[mid_idx].register_forward_hook(_make_hook("mid")),
            blocks[fin_idx].register_forward_hook(_make_hook("fin")),
        ]
        all_rows: list[np.ndarray] = []
        bs = max(1, int(batch_size))
        try:
            with self._precision_ctx():
                for start in range(0, len(sequences), bs):
                    chunk_seqs = sequences[start : start + bs]
                    chunk_3di = (
                        structure_3di_list[start : start + bs]
                        if structure_3di_list is not None
                        else [None] * len(chunk_seqs)
                    )
                    encoded = [
                        self._encode_sequence(s, structure_3di=s3)
                        for s, s3 in zip(chunk_seqs, chunk_3di)
                    ]
                    max_len = max(len(ids) for ids, _ in encoded)
                    input_ids = torch.full(
                        (len(encoded), max_len), int(pad_id), dtype=torch.long, device=self.device
                    )
                    attn_mask = torch.zeros_like(input_ids)
                    spans: list[tuple[int, int]] = []
                    for i, ((ids, offset), seq) in enumerate(zip(encoded, chunk_seqs)):
                        input_ids[i, : len(ids)] = torch.as_tensor(ids, dtype=torch.long, device=self.device)
                        attn_mask[i, : len(ids)] = 1
                        spans.append((offset, offset + len(seq)))

                    captured.clear()
                    _ = self._forward_logits(input_ids, attention_mask=attn_mask)
                    h_mid = captured["mid"]
                    h_fin = captured["fin"]
                    # Handle time-first (T, B, H) layouts transparently (e.g. ProteinGLM).
                    if self.time_first_hidden or (
                        h_mid.shape[0] == max_len and h_mid.shape[1] == len(encoded) and max_len != len(encoded)
                    ):
                        h_mid = h_mid.transpose(0, 1)
                        h_fin = h_fin.transpose(0, 1)

                    batch_out = []
                    for i, (left, right) in enumerate(spans):
                        m_res = h_mid[i, left:right].float()
                        f_res = h_fin[i, left:right].float()
                        if pooling == "mean_cls":
                            m_anchor = h_mid[i, 0].float()
                            f_anchor = h_fin[i, 0].float()
                        else:
                            m_anchor = m_res[-1]
                            f_anchor = f_res[-1]
                        batch_out.append(
                            torch.stack([m_res.mean(0), m_anchor, f_res.mean(0), f_anchor], dim=0)
                        )
                    chunk_np = torch.stack(batch_out, dim=0).cpu().numpy().astype(np.float32, copy=False)
                    if not np.isfinite(chunk_np).all():
                        raise FloatingPointError(f"{self.arm}: non-finite hidden representation")
                    all_rows.append(chunk_np)
        finally:
            for handle in handles:
                handle.remove()

        return np.concatenate(all_rows, axis=0)

    @torch.no_grad()
    def infill_window(
        self,
        parent: str,
        *,
        start: int,
        extent: int,
        seed: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        structure_3di: str | None = None,
    ) -> dict[str, Any]:
        """Iterative max-confidence unmasking over AA20 (identical protocol to ``ladder/runtime.py``)."""
        if extent < 1:
            raise ValueError("extent must be >= 1")
        stop = start + extent
        if not 1 <= start < stop <= len(parent) - 1:
            raise ValueError(f"window [{start}, {stop}) must be strictly interior to len={len(parent)}")
        ids_list, offset = self._encode_sequence(parent, structure_3di=structure_3di)
        ids = torch.as_tensor(ids_list, dtype=torch.long, device=self.device)
        pending = list(range(start, stop))
        for pos in pending:
            ids[offset + pos] = self.mask_id
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        support = torch.as_tensor(self.aa20_ids, dtype=torch.long, device=self.device)
        order: list[int] = []
        chosen_aas: dict[int, str] = {}
        with self._precision_ctx():
            while pending:
                logits = self._forward_logits(ids.unsqueeze(0))[0]
                rows = torch.as_tensor([offset + p for p in pending], dtype=torch.long, device=self.device)
                aa_logits = logits.index_select(0, rows).index_select(1, support) / float(temperature)
                restricted = F.log_softmax(aa_logits, dim=-1)
                truncated = _nucleus_logprobs(restricted, float(top_p))
                probs = truncated.exp().cpu()
                sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
                conf = probs.gather(1, sampled.unsqueeze(-1)).squeeze(-1)
                best = int(torch.argmax(conf))
                pos = pending.pop(best)
                aa_idx = int(sampled[best])
                ids[offset + pos] = int(support[aa_idx])
                chosen_aas[pos] = AA20[aa_idx]
                order.append(pos)
        window = "".join(chosen_aas[p] for p in range(start, stop))
        sequence = parent[:start] + window + parent[stop:]
        return {
            "status": "filled",
            "window": window,
            "sequence": sequence,
            "commit_order": order,
            "sampler": "one_position_per_step_max_confidence_iterative_unmasking",
            "support": "aa20_restricted_softmax",
        }

    @torch.no_grad()
    def sample_diffusion_full(
        self,
        length: int,
        *,
        seed: int,
        steps: int | None = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        sampling_mode: str = "max_confidence",
    ) -> dict[str, Any]:
        """100%-masked iterative unmasking strictly restricted to AA20.

        ``sampling_mode`` selects the unmasking-order rule:
        - ``"max_confidence"``: cosine schedule with deterministic ``topk(p(sampled))``
          (legacy default; note that selecting positions by highest sampled-token
          probability without order noise causes severe homopolymer mode amplification
          on 100%-masked unconditional generation).
        - ``"uniform_random"``: exact absorbing-state discrete diffusion reverse process
          with a linear unmasking schedule and uniform random position order independent
          of token probability (DPLM canonical ``rdm``/``uniform`` reverse transition).
        - ``"gumbel_annealed"``: cosine schedule with MaskGIT-style annealed Gumbel noise
          ``log(p(sampled)) + (1 - (step - 1) / total_steps) * Gumbel(0, 1)``.
        """
        valid_modes = ("max_confidence", "uniform_random", "gumbel_annealed")
        if sampling_mode not in valid_modes:
            raise ValueError(
                f"{self.arm}: invalid sampling_mode {sampling_mode!r}; expected one of {valid_modes}"
            )
        if not 1 <= length <= self.spec.max_residues:
            raise ValueError(f"{self.arm}: invalid length {length}")
        # Build 100%-masked sequence with proper CLS/EOS framing from a dummy template
        template_ids, offset = self._encode_sequence("A" * length)
        ids = torch.as_tensor(template_ids, dtype=torch.long, device=self.device)
        for pos in range(length):
            ids[offset + pos] = self.mask_id

        total_steps = int(steps) if steps is not None else min(length, 64)
        total_steps = max(1, min(total_steps, length))
        pending = list(range(length))
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        support = torch.as_tensor(self.aa20_ids, dtype=torch.long, device=self.device)
        chosen_aas: list[str] = [""] * length

        with self._precision_ctx():
            for step in range(1, total_steps + 1):
                if not pending:
                    break
                if sampling_mode == "uniform_random":
                    frac_remaining = 1.0 - (step / total_steps)
                else:
                    frac_remaining = math.cos(0.5 * math.pi * (step / total_steps))
                target_remaining = 0 if step == total_steps else int(math.floor(length * frac_remaining))
                n_commit = max(1, len(pending) - target_remaining)
                n_commit = min(n_commit, len(pending))

                logits = self._forward_logits(ids.unsqueeze(0))[0]
                rows = torch.as_tensor([offset + p for p in pending], dtype=torch.long, device=self.device)
                # Strictly restrict to AA20 (masks out <mask>, X, specials).
                aa_logits = logits.index_select(0, rows).index_select(1, support) / float(temperature)
                restricted = F.log_softmax(aa_logits, dim=-1)
                truncated = _nucleus_logprobs(restricted, float(top_p))
                probs = truncated.exp().cpu()
                sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
                conf = probs.gather(1, sampled.unsqueeze(-1)).squeeze(-1)

                if sampling_mode == "uniform_random":
                    selection_score = torch.rand(len(pending), generator=generator)
                elif sampling_mode == "gumbel_annealed":
                    u = torch.rand(len(pending), generator=generator).clamp(1e-6, 1.0 - 1e-6)
                    gumbel = -torch.log(-torch.log(u))
                    t_noise = 1.0 - float(step - 1) / float(total_steps)
                    selection_score = torch.log(conf.clamp_min(1e-12)) + t_noise * gumbel
                else:
                    selection_score = conf

                top_local = torch.topk(selection_score, k=n_commit, largest=True).indices.tolist()
                for local_idx in sorted(top_local, reverse=True):
                    pos = pending.pop(local_idx)
                    aa_idx = int(sampled[local_idx])
                    ids[offset + pos] = int(support[aa_idx])
                    chosen_aas[pos] = AA20[aa_idx]

        sequence = "".join(chosen_aas)
        if len(sequence) != length or set(sequence) - AA20_SET:
            raise RuntimeError(f"{self.arm}: diffusion sampling produced invalid sequence")
        sampler_labels = {
            "max_confidence": "cosine_schedule_max_confidence_unmasking_aa20_only",
            "uniform_random": "linear_schedule_uniform_random_unmasking_aa20_only",
            "gumbel_annealed": "cosine_schedule_gumbel_annealed_unmasking_aa20_only",
        }
        return {
            "status": "filled",
            "sequence": sequence,
            "length": length,
            "steps": total_steps,
            "sampling_mode": sampling_mode,
            "sampler": sampler_labels[sampling_mode],
            "temperature": float(temperature),
            "top_p": float(top_p),
        }


# ---------------------------------------------- Concrete Masked Family Loaders


def _resolve_aa20_ids(tokenizer: Any, arm: str) -> list[int]:
    unk = getattr(tokenizer, "unk_token_id", None)
    aa_ids: list[int] = []
    for aa in AA20:
        tid = tokenizer.convert_tokens_to_ids(aa)
        if tid is None or (unk is not None and int(tid) == int(unk)):
            raise ValueError(f"{arm}: tokenizer lacks dedicated token id for residue {aa!r}")
        aa_ids.append(int(tid))
    if len(set(aa_ids)) != len(AA20):
        raise ValueError(f"{arm}: duplicate token ids among AA20 residues")
    return aa_ids


def _resolve_mask_id(tokenizer: Any, arm: str) -> int:
    mid = getattr(tokenizer, "mask_token_id", None)
    if mid is None:
        for cand in ("<mask>", "[MASK]", "<gmask>", "<MASK>"):
            tid = tokenizer.convert_tokens_to_ids(cand)
            if tid is not None and tid != getattr(tokenizer, "unk_token_id", None):
                mid = int(tid)
                break
    if mid is None:
        raise ValueError(f"{arm}: tokenizer declares no mask token id")
    return int(mid)


_OPTIONAL_ESM_AUX_MISSING_KEYS: frozenset[str] = frozenset(
    {
        "esm.contact_head.regression.weight",
        "esm.contact_head.regression.bias",
    }
)


def _check_strict_esm_loading_info(
    arm: str, label: str, info: Mapping[str, Any]
) -> list[str]:
    """Fail fast on any missing/mismatched weight except unused ESM contact-head regression keys."""
    raw_missing = list(info.get("missing_keys", ()))
    ignored = [k for k in raw_missing if k in _OPTIONAL_ESM_AUX_MISSING_KEYS]
    hard_missing = [k for k in raw_missing if k not in _OPTIONAL_ESM_AUX_MISSING_KEYS]
    mismatched = list(info.get("mismatched_keys", ()))
    unmatched: dict[str, list[str]] = {}
    if hard_missing:
        unmatched["missing_keys"] = hard_missing
    if mismatched:
        unmatched["mismatched_keys"] = mismatched
    if unmatched:
        raise RuntimeError(f"{arm}: {label}: {unmatched}")
    return ignored


def _patch_deprecated_transformers_imports(source: str) -> str:
    """Relocate ``find_pruneable_heads_and_indices`` and ``prune_linear_layer`` from
    ``transformers.modeling_utils`` to ``transformers.pytorch_utils`` for transformers>=4.50.
    """
    if (
        "transformers.modeling_utils" not in source
        or ("find_pruneable_heads_and_indices" not in source and "prune_linear_layer" not in source)
    ):
        return source
    pattern = re.compile(
        r"from\s+transformers\.modeling_utils\s+import\s*\(([^)]*)\)",
        flags=re.MULTILINE,
    )
    moved = {"find_pruneable_heads_and_indices", "prune_linear_layer"}

    def _repl(match: re.Match[str]) -> str:
        items = [tok.strip() for tok in match.group(1).split(",") if tok.strip()]
        keep = [tok for tok in items if tok not in moved]
        reloc = [tok for tok in items if tok in moved]
        if not reloc:
            return match.group(0)
        lines: list[str] = []
        if keep:
            lines.append(f"from transformers.modeling_utils import {', '.join(keep)}")
        lines.append(f"from transformers.pytorch_utils import {', '.join(reloc)}")
        return "\n".join(lines)

    patched = pattern.sub(_repl, source)
    # Also handle single-line non-parenthesized form.
    single_pat = re.compile(r"from\s+transformers\.modeling_utils\s+import\s+([^\n(]+)")

    def _repl_single(match: re.Match[str]) -> str:
        items = [tok.strip() for tok in match.group(1).split(",") if tok.strip()]
        keep = [tok for tok in items if tok not in moved]
        reloc = [tok for tok in items if tok in moved]
        if not reloc:
            return match.group(0)
        lines: list[str] = []
        if keep:
            lines.append(f"from transformers.modeling_utils import {', '.join(keep)}")
        lines.append(f"from transformers.pytorch_utils import {', '.join(reloc)}")
        return "\n".join(lines)

    return single_pat.sub(_repl_single, patched)


def _maybe_derive_local_automap_dir(checkpoint_dir: Path) -> Path:
    """Derive a temp checkpoint dir stripping stale external HF repo prefixes from ``auto_map``
    when the target ``.py`` module exists locally beside ``config.json`` (e.g., ``Prime_690M``),
    disabling ``flash_attention`` when set to ``True`` so ``output_attentions=True`` is safe,
    and patching deprecated ``transformers.modeling_utils`` prune-helper imports.
    """
    cfg_path = checkpoint_dir / "config.json"
    if not cfg_path.is_file():
        return checkpoint_dir
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    auto_map = cfg.get("auto_map")
    needs_rewrite = False
    new_auto_map: dict[str, Any] = {}
    if isinstance(auto_map, dict):
        for k, v in auto_map.items():
            if isinstance(v, str) and "--" in v:
                local_ref = v.split("--", 1)[1]
                mod_stem = local_ref.split(".", 1)[0]
                if (checkpoint_dir / f"{mod_stem}.py").is_file():
                    new_auto_map[k] = local_ref
                    needs_rewrite = True
                    continue
            new_auto_map[k] = v
    disable_flash = bool(cfg.get("flash_attention") is True)
    if disable_flash:
        needs_rewrite = True

    patched_py: dict[str, str] = {}
    for py_file in checkpoint_dir.glob("*.py"):
        raw = py_file.read_text(encoding="utf-8")
        fixed = _patch_deprecated_transformers_imports(raw)
        if fixed != raw:
            patched_py[py_file.name] = fixed
            needs_rewrite = True

    if not needs_rewrite:
        return checkpoint_dir

    if isinstance(auto_map, dict):
        cfg["auto_map"] = new_auto_map
    if disable_flash:
        cfg["flash_attention"] = False

    dest = Path(tempfile.mkdtemp(prefix="esm_mlm_derived_"))
    for item in checkpoint_dir.iterdir():
        target = dest / item.name
        if item.name == "config.json":
            target.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        elif item.name in patched_py:
            target.write_text(patched_py[item.name], encoding="utf-8")
        else:
            target.symlink_to(item.resolve())
    return dest


def _load_esm_mlm(
    spec: GeneralPLMSpec, checkpoint_dir: Path, *, device: str, dtype: str
) -> _BaseMaskedAdapter:
    from transformers import AutoModelForMaskedLM, AutoTokenizer, EsmForMaskedLM

    load_dir = _maybe_derive_local_automap_dir(checkpoint_dir)
    cfg = json.loads((load_dir / "config.json").read_text(encoding="utf-8"))
    torch_dt = torch_dtype_of(dtype)
    use_auto = bool(cfg.get("auto_map")) or cfg.get("model_type") != "esm"
    tokenizer = AutoTokenizer.from_pretrained(
        str(load_dir), local_files_only=True, trust_remote_code=spec.trust_remote_code
    )
    cls = AutoModelForMaskedLM if use_auto else EsmForMaskedLM
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "output_loading_info": True,
        **_from_pretrained_dtype_kwarg(cls, torch_dt),
    }
    if use_auto and spec.trust_remote_code:
        kwargs["trust_remote_code"] = True
    model, info = cls.from_pretrained(str(load_dir), **kwargs)
    ignored_missing = _check_strict_esm_loading_info(
        spec.arm, "checkpoint failed strict weight check", info
    )
    model = model.eval().to(device)
    return _BaseMaskedAdapter(
        spec,
        checkpoint_dir,
        device=device,
        dtype=dtype,
        model=model,
        tokenizer=tokenizer,
        mask_id=_resolve_mask_id(tokenizer, spec.arm),
        aa20_ids=_resolve_aa20_ids(tokenizer, spec.arm),
        extra_provenance={
            "ignored_optional_missing_keys": ignored_missing,
            "unexpected_keys": list(info.get("unexpected_keys", ())),
            "model_class": type(model).__name__,
        },
    )


def _load_esmc_hf(
    spec: GeneralPLMSpec, checkpoint_dir: Path, *, device: str, dtype: str
) -> _BaseMaskedAdapter:
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    cfg = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    torch_dt = torch_dtype_of(dtype)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_dir), local_files_only=True, trust_remote_code=True
        )
        try:
            model, info = AutoModelForMaskedLM.from_pretrained(
                str(checkpoint_dir),
                local_files_only=True,
                trust_remote_code=True,
                output_loading_info=True,
                **_from_pretrained_dtype_kwarg(AutoModelForMaskedLM, torch_dt),
            )
        except Exception:
            model, info = AutoModel.from_pretrained(
                str(checkpoint_dir),
                local_files_only=True,
                trust_remote_code=True,
                output_loading_info=True,
                **_from_pretrained_dtype_kwarg(AutoModel, torch_dt),
            )
    except Exception as exc:
        raise RuntimeError(
            f"{spec.arm}: failed to load ESMC-hf ({cfg.get('model_type')}) under current "
            f"transformers version; if checkpoint requires transformers>=5, invoke with "
            f"runtimes/esmfold2/bin/python. Original error: {exc}"
        ) from exc

    unmatched = {k: info[k] for k in ("missing_keys", "mismatched_keys") if info.get(k)}
    if unmatched:
        raise RuntimeError(f"{spec.arm}: ESMC-hf weight mismatch: {unmatched}")
    model = model.eval().to(device)
    return _BaseMaskedAdapter(
        spec,
        checkpoint_dir,
        device=device,
        dtype=dtype,
        model=model,
        tokenizer=tokenizer,
        mask_id=_resolve_mask_id(tokenizer, spec.arm),
        aa20_ids=_resolve_aa20_ids(tokenizer, spec.arm),
        extra_provenance={
            "unexpected_keys": list(info.get("unexpected_keys", ())),
            "model_class": type(model).__name__,
        },
    )


def _install_xformers_ops_shim_if_needed() -> bool:
    """Install a pure-PyTorch ``xformers.ops`` shim when ``xformers`` is absent (needed by AMPLIFY)."""
    try:
        import xformers.ops  # noqa: F401
        return False
    except ImportError:
        pass

    import types

    class _SwiGLU(torch.nn.Module):
        """Matches xformers.ops.SwiGLU parameter names (w12, w3) and forward math."""

        def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: int | None = None,
            bias: bool = True,
            **_kwargs: Any,
        ) -> None:
            super().__init__()
            out_f = in_features if out_features is None else out_features
            self.w12 = torch.nn.Linear(in_features, 2 * hidden_features, bias=bias)
            self.w3 = torch.nn.Linear(hidden_features, out_f, bias=bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x12 = self.w12(x)
            x1, x2 = x12.chunk(2, dim=-1)
            return self.w3(F.silu(x1) * x2)

    def _memory_efficient_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        p: float = 0.0,
        scale: float | None = None,
    ) -> torch.Tensor:
        # xformers layout: (B, M, H, K) -> PyTorch SDPA layout: (B, H, M, K)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=float(p), scale=scale
        )
        return out.transpose(1, 2).contiguous()

    xf_mod = types.ModuleType("xformers")
    ops_mod = types.ModuleType("xformers.ops")
    ops_mod.SwiGLU = _SwiGLU  # type: ignore[attr-defined]
    ops_mod.memory_efficient_attention = _memory_efficient_attention  # type: ignore[attr-defined]
    xf_mod.ops = ops_mod  # type: ignore[attr-defined]
    sys.modules["xformers"] = xf_mod
    sys.modules["xformers.ops"] = ops_mod
    return True


def _load_amplify_mlm(
    spec: GeneralPLMSpec, checkpoint_dir: Path, *, device: str, dtype: str
) -> _BaseMaskedAdapter:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    shimmed = _install_xformers_ops_shim_if_needed()
    torch_dt = torch_dtype_of(dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_dir), local_files_only=True, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(
        str(checkpoint_dir), local_files_only=True, trust_remote_code=True
    )
    model, info = AutoModel.from_pretrained(
        str(checkpoint_dir),
        config=config,
        local_files_only=True,
        trust_remote_code=True,
        output_loading_info=True,
        **_from_pretrained_dtype_kwarg(AutoModel, torch_dt),
    )
    unmatched = {k: info[k] for k in ("missing_keys", "mismatched_keys") if info.get(k)}
    if unmatched:
        raise RuntimeError(f"{spec.arm}: AMPLIFY weight mismatch: {unmatched}")
    model = model.eval().to(device)
    return _BaseMaskedAdapter(
        spec,
        checkpoint_dir,
        device=device,
        dtype=dtype,
        model=model,
        tokenizer=tokenizer,
        mask_id=_resolve_mask_id(tokenizer, spec.arm),
        aa20_ids=_resolve_aa20_ids(tokenizer, spec.arm),
        additive_attention_mask=True,
        extra_provenance={
            "unexpected_keys": list(info.get("unexpected_keys", ())),
            "model_class": type(model).__name__,
            "xformers_ops_shimmed": shimmed,
            "additive_attention_mask": True,
        },
    )


def _maybe_derive_proteinglm_mlm_dir(checkpoint_dir: Path) -> Path:
    """Derive a temp ProteinGLM-MLM dir when ``deepspeed`` patching or a missing
    ``model.safetensors.index.json`` (e.g. ``proteinglm-10b-mlm``) is required.
    """
    modeling_py = checkpoint_dir / "modeling_proteinglm.py"
    source = modeling_py.read_text(encoding="utf-8") if modeling_py.is_file() else ""
    needs_deepspeed_patch = False
    if "deepspeed" in source:
        try:
            import deepspeed  # noqa: F401
        except ImportError:
            needs_deepspeed_patch = True

    shards = sorted(checkpoint_dir.glob("model-*-of-*.safetensors"))
    needs_safetensors_index = (
        bool(shards)
        and not (checkpoint_dir / "model.safetensors.index.json").is_file()
        and not (checkpoint_dir / "model.safetensors").is_file()
    )
    if not needs_deepspeed_patch and not needs_safetensors_index:
        return checkpoint_dir

    dest = Path(tempfile.mkdtemp(prefix="proteinglm_mlm_derived_"))
    for item in checkpoint_dir.iterdir():
        target = dest / item.name
        if needs_deepspeed_patch and item.name == "modeling_proteinglm.py":
            patched = source.replace("import torch, deepspeed", "import torch")
            patched = patched.replace("import deepspeed", "deepspeed = None")
            target.write_text(patched, encoding="utf-8")
        else:
            target.symlink_to(item.resolve())

    if needs_safetensors_index:
        from safetensors import safe_open

        weight_map: dict[str, str] = {}
        total_size = 0
        for shard in shards:
            total_size += int(shard.stat().st_size)
            with safe_open(str(shard), framework="pt", device="cpu") as fh:
                for key in fh.keys():
                    weight_map[str(key)] = shard.name
        index_payload = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        (dest / "model.safetensors.index.json").write_text(
            json.dumps(index_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return dest


def _load_proteinglm_mlm(
    spec: GeneralPLMSpec, checkpoint_dir: Path, *, device: str, dtype: str
) -> _BaseMaskedAdapter:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    load_dir = _maybe_derive_proteinglm_mlm_dir(checkpoint_dir)
    torch_dt = torch_dtype_of(dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        str(load_dir), local_files_only=True, trust_remote_code=True
    )
    model, info = AutoModelForMaskedLM.from_pretrained(
        str(load_dir),
        local_files_only=True,
        trust_remote_code=True,
        output_loading_info=True,
        **_from_pretrained_dtype_kwarg(AutoModelForMaskedLM, torch_dt),
    )
    unmatched = {k: info[k] for k in ("missing_keys", "mismatched_keys") if info.get(k)}
    if unmatched:
        raise RuntimeError(f"{spec.arm}: ProteinGLM-MLM weight mismatch: {unmatched}")
    model = model.eval().to(device)
    return _BaseMaskedAdapter(
        spec,
        checkpoint_dir,
        device=device,
        dtype=dtype,
        model=model,
        tokenizer=tokenizer,
        mask_id=_resolve_mask_id(tokenizer, spec.arm),
        aa20_ids=_resolve_aa20_ids(tokenizer, spec.arm),
        time_first_hidden=True,
        extra_provenance={
            "unexpected_keys": list(info.get("unexpected_keys", ())),
            "model_class": type(model).__name__,
            "derived_no_deepspeed": load_dir != checkpoint_dir,
        },
    )


# ----------------------------------------------------------- SaProt 3Di+AA MLM


class SaProtAdapter(_BaseMaskedAdapter):
    """SaProt 441-token AA+3Di adapter supporting both seq_only (#-masked) and foldseek_3di modes."""

    def __init__(
        self,
        spec: GeneralPLMSpec,
        checkpoint_dir: Path,
        *,
        device: str,
        dtype: str,
        model: torch.nn.Module,
        tokenizer: Any,
        saprot_mode: str = "seq_only",
        extra_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        if saprot_mode not in ("seq_only", "foldseek_3di"):
            raise ValueError(f"saprot_mode must be 'seq_only' or 'foldseek_3di', got {saprot_mode!r}")
        self.saprot_mode = saprot_mode
        vocab = tokenizer.get_vocab()
        mask_id = _resolve_mask_id(tokenizer, spec.arm)

        # Dynamically discover all 20 lowercase 3Di tokens per AA20 residue and AA# tokens.
        aa_struct_ids: dict[str, list[int]] = {}
        aa_hash_id: dict[str, int] = {}
        hash_struct_id: dict[str, int] = {}
        struct_chars_seen: set[str] = set()
        for tok, tid in vocab.items():
            if len(tok) == 2 and tok[0] in AA20_SET:
                if tok[1] == "#":
                    aa_hash_id[tok[0]] = int(tid)
                elif tok[1].islower():
                    aa_struct_ids.setdefault(tok[0], []).append((tok[1], int(tid)))
                    struct_chars_seen.add(tok[1])
            elif len(tok) == 2 and tok[0] == "#" and tok[1].islower():
                hash_struct_id[tok[1]] = int(tid)

        for aa in AA20:
            if aa not in aa_hash_id:
                raise ValueError(f"{spec.arm}: SaProt vocab missing sequence-only token {aa + '#' !r}")
            pairs = sorted(aa_struct_ids.get(aa, ()))
            if len(pairs) != 20:
                raise ValueError(
                    f"{spec.arm}: expected 20 3Di structure tokens for {aa}, found {len(pairs)}"
                )
            aa_struct_ids[aa] = [tid for _, tid in pairs]

        self.aa_hash_id = aa_hash_id
        self.aa_struct_ids = aa_struct_ids  # {aa: [20 token ids]}
        self.hash_struct_id = hash_struct_id  # {3di_char: token_id}
        self.vocab = vocab
        self.struct_chars = "".join(sorted(struct_chars_seen))

        super().__init__(
            spec,
            checkpoint_dir,
            device=device,
            dtype=dtype,
            model=model,
            tokenizer=tokenizer,
            mask_id=mask_id,
            aa20_ids=[aa_hash_id[aa] for aa in AA20],
            extra_provenance={
                "saprot_mode": saprot_mode,
                "n_3di_alphabet": len(self.struct_chars),
                "has_hash_3di_mask_tokens": len(hash_struct_id) == 20,
                **(dict(extra_provenance) if extra_provenance else {}),
            },
        )
        # Tensor of shape (20, 20): row i = the 20 3Di token ids for AA20[i].
        self._struct_id_matrix = torch.as_tensor(
            [self.aa_struct_ids[aa] for aa in AA20], dtype=torch.long, device=self.device
        )

    def _encode_sequence(self, sequence: str, structure_3di: str | None = None) -> tuple[list[int], int]:
        self.validate_sequence(sequence)
        use_3di = structure_3di is not None and self.saprot_mode == "foldseek_3di"
        if use_3di:
            assert structure_3di is not None
            if len(structure_3di) != len(sequence):
                raise ValueError(
                    f"{self.arm}: structure_3di length {len(structure_3di)} != sequence length {len(sequence)}"
                )
            combined = "".join(f"{aa}{s.lower()}" for aa, s in zip(sequence, structure_3di))
        else:
            combined = "".join(f"{aa}#" for aa in sequence)
        ids = [int(v) for v in self.tokenizer(combined)["input_ids"]]
        if len(ids) != len(sequence) + 2:
            raise ValueError(
                f"{self.arm}: expected {len(sequence) + 2} SaProt tokens for {len(sequence)} residues, got {len(ids)}"
            )
        return ids, 1

    def _mask_token_for_pos(self, pos: int, structure_3di: str | None) -> tuple[int, str | None]:
        if self.saprot_mode == "foldseek_3di" and structure_3di is not None:
            s = structure_3di[pos].lower()
            if s != "#" and s in self.hash_struct_id:
                return self.hash_struct_id[s], s
        return self.mask_id, None

    def _aa20_marginal_logprobs_from_full(
        self, full_logprobs: torch.Tensor, known_3di: str | None
    ) -> torch.Tensor:
        """Compute length-20 residue log-probabilities from full-vocab log_softmax.

        In ``seq_only`` mode (``known_3di is None``), marginalizes over all 20
        structure tokens per AA via ``logsumexp`` (preserving full-vocab normalization).
        In ``foldseek_3di`` mode with known 3Di char ``s``, reads the exact ``AA+s``
        token log-probability.
        """
        if known_3di is not None:
            ids_20 = torch.as_tensor(
                [int(self.vocab[f"{aa}{known_3di}"]) for aa in AA20],
                dtype=torch.long,
                device=full_logprobs.device,
            )
            return full_logprobs.index_select(0, ids_20)
        # shape (20, 20) -> logsumexp over 20 structure tokens -> shape (20,)
        gathered = full_logprobs[self._struct_id_matrix]
        return torch.logsumexp(gathered, dim=-1)

    @torch.no_grad()
    def masked_aa20_logprobs(
        self,
        sequence: str,
        positions: Sequence[int],
        *,
        batch_size: int = 16,
        structure_3di: str | None = None,
    ) -> dict[int, np.ndarray]:
        """Return ``{pos: logprobs_20}`` (full-vocab marginalized over 20 3Di tokens or exact 3Di)."""
        ids, offset = self._encode_sequence(sequence, structure_3di=structure_3di)
        ordered = sorted({int(p) for p in positions})
        row = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        out: dict[int, np.ndarray] = {}
        bs = max(1, int(batch_size))
        with self._precision_ctx():
            for start in range(0, len(ordered), bs):
                chunk = ordered[start : start + bs]
                batch = row.unsqueeze(0).repeat(len(chunk), 1)
                known_chars: list[str | None] = []
                for idx, pos in enumerate(chunk):
                    mask_tok, known_3di = self._mask_token_for_pos(pos, structure_3di)
                    batch[idx, offset + pos] = mask_tok
                    known_chars.append(known_3di)
                logits = self._forward_logits(batch)
                for idx, (pos, known_3di) in enumerate(zip(chunk, known_chars)):
                    full_lp = F.log_softmax(logits[idx, offset + pos], dim=-1)
                    aa_lp = self._aa20_marginal_logprobs_from_full(full_lp, known_3di)
                    out[pos] = aa_lp.detach().cpu().numpy().astype(np.float32, copy=False)
                del logits
        return out

    @torch.no_grad()
    def exact_pll(
        self,
        sequence: str,
        *,
        batch_size: int = 16,
        structure_3di: str | None = None,
    ) -> PositionVector:
        table = self.masked_aa20_logprobs(
            sequence, range(len(sequence)), batch_size=batch_size, structure_3di=structure_3di
        )
        terms = np.asarray(
            [-float(table[pos][AA20_INDEX[aa]]) for pos, aa in enumerate(sequence)],
            dtype=np.float32,
        )
        if not np.isfinite(terms).all():
            raise FloatingPointError(f"{self.arm}: non-finite SaProt PLL term")
        nll_sum = float(terms.sum(dtype=np.float64))
        return PositionVector(
            terms=terms, nll_sum=nll_sum, pll=-nll_sum, paradigm=self.spec.paradigm
        )

    def score_variants(
        self,
        wildtype: str,
        mutations: Sequence[str],
        *,
        batch_size: int = 16,
        multi_exact_pll: bool = False,
        structure_3di: str | None = None,
    ) -> np.ndarray:
        parsed = [parse_mutation(m, wildtype) for m in mutations]
        single_indices = [i for i, entry in enumerate(parsed) if len(entry) == 1]
        multi_indices = [i for i, entry in enumerate(parsed) if len(entry) > 1]
        values = np.empty(len(mutations), dtype=np.float64)

        additive_indices = single_indices if multi_exact_pll else list(range(len(mutations)))
        if additive_indices:
            needed = sorted({pos for idx in additive_indices for pos, _, _ in parsed[idx]})
            table = self.masked_aa20_logprobs(
                wildtype, needed, batch_size=batch_size, structure_3di=structure_3di
            )
            for idx in additive_indices:
                total = 0.0
                for pos, wt_aa, mut_aa in parsed[idx]:
                    row_lp = table[pos]
                    total += float(row_lp[AA20_INDEX[mut_aa]] - row_lp[AA20_INDEX[wt_aa]])
                values[idx] = total

        if multi_exact_pll and multi_indices:
            wt_pll = self.exact_pll(
                wildtype, batch_size=batch_size, structure_3di=structure_3di
            ).pll
            for idx in multi_indices:
                mut_seq = apply_mutation(wildtype, mutations[idx])
                mut_pll = self.exact_pll(
                    mut_seq, batch_size=batch_size, structure_3di=structure_3di
                ).pll
                values[idx] = float(mut_pll - wt_pll)

        return values

    @torch.no_grad()
    def native_site_conditional(
        self,
        sequence: str,
        index: int,
        *,
        structure_3di: str | None = None,
    ) -> np.ndarray:
        table = self.masked_aa20_logprobs(
            sequence, [index], batch_size=1, structure_3di=structure_3di
        )
        raw = np.asarray(table[index], dtype=np.float64)
        peak = float(raw.max())
        lse = peak + math.log(float(np.exp(raw - peak).sum()))
        return raw - lse

    @torch.no_grad()
    def infill_window(
        self,
        parent: str,
        *,
        start: int,
        extent: int,
        seed: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        structure_3di: str | None = None,
    ) -> dict[str, Any]:
        if extent < 1:
            raise ValueError("extent must be >= 1")
        stop = start + extent
        if not 1 <= start < stop <= len(parent) - 1:
            raise ValueError(f"window [{start}, {stop}) must be strictly interior")
        ids_list, offset = self._encode_sequence(parent, structure_3di=structure_3di)
        ids = torch.as_tensor(ids_list, dtype=torch.long, device=self.device)
        pending = list(range(start, stop))
        known_map: dict[int, str | None] = {}
        for pos in pending:
            mask_tok, known_3di = self._mask_token_for_pos(pos, structure_3di)
            ids[offset + pos] = mask_tok
            known_map[pos] = known_3di
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        order: list[int] = []
        chosen_aas: dict[int, str] = {}
        with self._precision_ctx():
            while pending:
                logits = self._forward_logits(ids.unsqueeze(0))[0]
                aa_lps = []
                for pos in pending:
                    full_lp = F.log_softmax(logits[offset + pos] / float(temperature), dim=-1)
                    raw_20 = self._aa20_marginal_logprobs_from_full(full_lp, known_map[pos])
                    aa_lps.append(F.log_softmax(raw_20, dim=-1))
                stacked = torch.stack(aa_lps, dim=0)
                truncated = _nucleus_logprobs(stacked, float(top_p))
                probs = truncated.exp().cpu()
                sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
                conf = probs.gather(1, sampled.unsqueeze(-1)).squeeze(-1)
                best = int(torch.argmax(conf))
                pos = pending.pop(best)
                aa = AA20[int(sampled[best])]
                chosen_aas[pos] = aa
                s = known_map[pos]
                ids[offset + pos] = int(self.vocab[f"{aa}{s}"]) if s is not None else self.aa_hash_id[aa]
                order.append(pos)
        window = "".join(chosen_aas[p] for p in range(start, stop))
        return {
            "status": "filled",
            "window": window,
            "sequence": parent[:start] + window + parent[stop:],
            "commit_order": order,
            "sampler": f"saprot_{self.saprot_mode}_max_confidence_unmasking",
            "support": "aa20_marginalized_softmax",
        }


def _load_saprot_mlm(
    spec: GeneralPLMSpec,
    checkpoint_dir: Path,
    *,
    device: str,
    dtype: str,
    saprot_mode: str = "seq_only",
) -> SaProtAdapter:
    from transformers import AutoTokenizer, EsmForMaskedLM

    torch_dt = torch_dtype_of(dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), local_files_only=True)
    model, info = EsmForMaskedLM.from_pretrained(
        str(checkpoint_dir),
        local_files_only=True,
        output_loading_info=True,
        **_from_pretrained_dtype_kwarg(EsmForMaskedLM, torch_dt),
    )
    ignored_missing = _check_strict_esm_loading_info(
        spec.arm, "SaProt weight mismatch", info
    )
    model = model.eval().to(device)
    return SaProtAdapter(
        spec,
        checkpoint_dir,
        device=device,
        dtype=dtype,
        model=model,
        tokenizer=tokenizer,
        saprot_mode=saprot_mode,
        extra_provenance={
            "ignored_optional_missing_keys": ignored_missing,
            "unexpected_keys": list(info.get("unexpected_keys", ())),
            "model_class": type(model).__name__,
        },
    )


# ------------------------------------------------------------ Causal Adapters


class _BaseCausalAdapter(GeneralPLMAdapter):
    """Exact causal next-token NLL, singleton scoring, hidden states, R4 context, and R5 generation."""

    def __init__(
        self,
        spec: GeneralPLMSpec,
        checkpoint_dir: Path,
        *,
        device: str,
        dtype: str,
        model: torch.nn.Module,
        tokenizer: Any,
        bos_prefix_ids: list[int],
        eos_suffix_ids: list[int],
        separator_ids: list[int],
        aa20_ids: list[int],
        extra_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            spec,
            checkpoint_dir,
            device=device,
            dtype=dtype,
            model=model,
            tokenizer=tokenizer,
            extra_provenance=extra_provenance,
        )
        self.bos_prefix_ids = [int(x) for x in bos_prefix_ids]
        self.eos_suffix_ids = [int(x) for x in eos_suffix_ids]
        self.separator_ids = [int(x) for x in separator_ids]
        self.aa20_ids = [int(x) for x in aa20_ids]
        self.aa_to_id = {aa: tid for aa, tid in zip(AA20, self.aa20_ids)}
        self.id_to_aa = {tid: aa for aa, tid in self.aa_to_id.items()}

    def _pack_item(self, sequence: str) -> tuple[list[int], tuple[int, int]]:
        """Pack one sequence into ``(ids, (scored_start, scored_end))``."""
        self.validate_sequence(sequence)
        body = [self.aa_to_id[aa] for aa in sequence]
        ids = list(self.bos_prefix_ids) + body + list(self.eos_suffix_ids)
        start = len(self.bos_prefix_ids)
        end = start + len(body)
        if start < 1:
            # Need at least 1 context token before the first scored token; score residues 2..L.
            start = 1
        return ids, (start, end)

    def _forward_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": False}
        # Dayhoff JambaForCausalLM defaults num_logits_to_keep=1 in config.json; force full logits.
        sig = inspect.signature(self.model.forward)
        if "logits_to_keep" in sig.parameters:
            kwargs["logits_to_keep"] = 0
        elif "num_logits_to_keep" in sig.parameters:
            kwargs["num_logits_to_keep"] = 0
        out = self.model(**kwargs)
        return out.logits.float()

    @torch.no_grad()
    def _score_span(self, ids: Sequence[int], start: int, end: int) -> PositionVector:
        if not 1 <= start < end <= len(ids):
            raise ValueError(f"{self.arm}: invalid scored span ({start}, {end}) for len={len(ids)}")
        tensor = torch.as_tensor([list(ids)], dtype=torch.long, device=self.device)
        with self._precision_ctx():
            logits = self._forward_ids(tensor)
            shift_logits = logits[0, start - 1 : end - 1]
            targets = tensor[0, start:end]
            terms = F.cross_entropy(shift_logits, targets, reduction="none")
        if not torch.isfinite(terms).all():
            raise FloatingPointError(f"{self.arm}: non-finite causal NLL term")
        nll_sum = float(terms.sum())
        arr = terms.detach().cpu().numpy().astype(np.float32, copy=False)
        return PositionVector(
            terms=arr, nll_sum=nll_sum, pll=-nll_sum, paradigm=self.spec.paradigm
        )

    def exact_pll(
        self,
        sequence: str,
        *,
        batch_size: int = 1,
        structure_3di: str | None = None,
    ) -> PositionVector:
        ids, (start, end) = self._pack_item(sequence)
        return self._score_span(ids, start, end)

    def score_variants(
        self,
        wildtype: str,
        mutations: Sequence[str],
        *,
        batch_size: int = 1,
        multi_exact_pll: bool = False,
        structure_3di: str | None = None,
    ) -> np.ndarray:
        """Exact causal sequence log-likelihood difference log P(mut) - log P(wt) (singleton)."""
        wt_pll = self.exact_pll(wildtype).pll
        values = np.empty(len(mutations), dtype=np.float64)
        for i, mut in enumerate(mutations):
            mut_seq = apply_mutation(wildtype, mut)
            values[i] = float(self.exact_pll(mut_seq).pll - wt_pll)
        return values

    def native_site_conditional(
        self,
        sequence: str,
        index: int,
        *,
        structure_3di: str | None = None,
    ) -> np.ndarray:
        """Exact causal 20-way induced conditional softmax(-NLL(x^{i->a})) matching family_conditionals."""
        if not 0 <= index < len(sequence):
            raise ValueError(f"index {index} out of bounds for length {len(sequence)}")
        scores = np.empty(len(AA20), dtype=np.float64)
        for idx, aa in enumerate(AA20):
            cand = sequence[:index] + aa + sequence[index + 1 :]
            scores[idx] = self.exact_pll(cand).pll
        peak = float(scores.max())
        lse = peak + math.log(float(np.exp(scores - peak).sum()))
        return scores - lse

    @torch.no_grad()
    def extract_hidden_states(
        self,
        sequences: Sequence[str],
        *,
        batch_size: int = 1,
        pooling: str = "mean_last",
        structure_3di_list: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Extract singleton middle/final hidden states (mean & last residue token)."""
        if not sequences:
            return np.zeros((0, 4, 0), dtype=np.float32)
        blocks = _discover_transformer_blocks(self.model)
        mid_idx = (len(blocks) - 1) // 2
        fin_idx = len(blocks) - 1
        captured: dict[str, torch.Tensor] = {}

        def _make_hook(label: str):
            def _hook(_mod, _inp, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[label] = h.detach()
            return _hook

        handles = [
            blocks[mid_idx].register_forward_hook(_make_hook("mid")),
            blocks[fin_idx].register_forward_hook(_make_hook("fin")),
        ]
        rows: list[np.ndarray] = []
        try:
            with self._precision_ctx():
                for seq in sequences:
                    ids, (start, end) = self._pack_item(seq)
                    left = len(self.bos_prefix_ids)
                    right = left + len(seq)
                    tensor = torch.as_tensor([ids], dtype=torch.long, device=self.device)
                    captured.clear()
                    _ = self._forward_ids(tensor)
                    h_mid = captured["mid"][0, left:right].float()
                    h_fin = captured["fin"][0, left:right].float()
                    vec = torch.stack(
                        [h_mid.mean(0), h_mid[-1], h_fin.mean(0), h_fin[-1]], dim=0
                    )
                    arr = vec.cpu().numpy().astype(np.float32, copy=False)
                    if not np.isfinite(arr).all():
                        raise FloatingPointError(f"{self.arm}: non-finite causal hidden state")
                    rows.append(arr)
        finally:
            for handle in handles:
                handle.remove()
        return np.stack(rows, axis=0)

    def score_r4_context(
        self,
        target: str,
        context_items: Sequence[str],
        *,
        budget_tokens: int = 1024,
        batch_size: int = 1,
        aligned_msa: bool = True,
    ) -> dict[str, Any]:
        """Concatenate as many context items as fit inside ``budget_tokens`` before ``target``.

        For multi-sequence-pretrained arms (``protgpt3_msa``, ``dayhoff_msa``),
        ``bos_prefix_ids`` is placed once at the sequence start and items are
        joined by ``separator_ids`` (``<s>`` or ``/``). For single-sequence
        causal arms, each item carries its own native framing (matching EXP-R2-228).
        """
        self.validate_sequence(target)
        if self.spec.family in ("protgpt3_msa", "dayhoff_msa"):
            target_body = [self.aa_to_id[aa] for aa in target]
            prefix = list(self.bos_prefix_ids)
            remaining = int(budget_tokens) - len(prefix) - len(target_body)
            if remaining < 0:
                raise ValueError(
                    f"{self.arm}: target length {len(target_body)} exceeds budget_tokens={budget_tokens}"
                )
            used_k = 0
            for item in context_items:
                self.validate_sequence(item)
                chunk = [self.aa_to_id[aa] for aa in item] + list(self.separator_ids)
                if len(prefix) - len(self.bos_prefix_ids) + len(chunk) > remaining:
                    break
                prefix.extend(chunk)
                used_k += 1
            full_ids = prefix + target_body
            span_start = max(1, len(prefix))
            span_end = len(prefix) + len(target_body)
        else:
            target_ids, (rel_start, rel_end) = self._pack_item(target)
            remaining = int(budget_tokens) - len(target_ids)
            if remaining < 0:
                raise ValueError(
                    f"{self.arm}: target length {len(target_ids)} exceeds budget_tokens={budget_tokens}"
                )
            prefix = []
            used_k = 0
            for item in context_items:
                item_ids, _ = self._pack_item(item)
                chunk = item_ids + list(self.separator_ids)
                if len(prefix) + len(chunk) > remaining:
                    break
                prefix.extend(chunk)
                used_k += 1
            full_ids = prefix + target_ids
            span_start = len(prefix) + rel_start
            span_end = len(prefix) + rel_end
        score = self._score_span(full_ids, span_start, span_end)
        return {
            "nll_sum": float(score.nll_sum),
            "nll_per_residue": float(score.nll_sum) / len(target),
            "scored_tokens": int(span_end - span_start),
            "total_tokens": len(full_ids),
            "k_items": used_k,
        }

    @torch.no_grad()
    def _sample_tokens(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        seed: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        stop_ids: set[int] | None = None,
    ) -> list[int]:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        cur = torch.as_tensor([prompt_ids], dtype=torch.long, device=self.device)
        produced: list[int] = []
        eos_set = set(stop_ids or ())
        with self._precision_ctx():
            past = None
            for step in range(max_new_tokens):
                inp = cur if past is None else cur[:, -1:]
                try:
                    out = self.model(input_ids=inp, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                except Exception:
                    out = self.model(input_ids=cur, use_cache=False)
                    past = None
                logits = out.logits[0, -1].float() / float(temperature)
                lp = F.log_softmax(logits, dim=-1)
                trunc = _nucleus_logprobs(lp, float(top_p))
                probs = trunc.exp().cpu()
                nxt = int(torch.multinomial(probs, num_samples=1, generator=gen).item())
                produced.append(nxt)
                cur = torch.cat(
                    [cur, torch.tensor([[nxt]], dtype=torch.long, device=self.device)], dim=1
                )
                if nxt in eos_set:
                    break
        return produced

    def infill_window(
        self,
        parent: str,
        *,
        start: int,
        extent: int,
        seed: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        structure_3di: str | None = None,
    ) -> dict[str, Any]:
        """Causal prefix regeneration of ``extent`` residues from ``parent[:start]``, restoring suffix."""
        if extent < 1:
            raise ValueError("extent must be >= 1")
        stop = start + extent
        if not 1 <= start < stop <= len(parent) - 1:
            raise ValueError(f"window [{start}, {stop}) must be strictly interior")
        prefix_ids = list(self.bos_prefix_ids) + [self.aa_to_id[aa] for aa in parent[:start]]
        eos_set = set(self.eos_suffix_ids)
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            eos_set.add(int(self.tokenizer.eos_token_id))
        tokens = self._sample_tokens(
            prefix_ids,
            max_new_tokens=extent + 16,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            stop_ids=eos_set,
        )
        residues: list[str] = []
        status = "filled"
        for tid in tokens:
            if tid in eos_set:
                if len(residues) < extent:
                    status = "terminated_short"
                break
            if tid in self.id_to_aa:
                residues.append(self.id_to_aa[tid])
                if len(residues) == extent:
                    break
            else:
                status = "off_alphabet"
                break
        if len(residues) < extent and status == "filled":
            status = "max_new_tokens"
        window = "".join(residues[:extent])
        sequence = parent[:start] + window + parent[stop:] if status == "filled" else None
        return {
            "status": status,
            "window": window,
            "sequence": sequence,
            "sampler": "causal_prefix_regeneration",
            "temperature": float(temperature),
            "top_p": float(top_p),
        }

    def generate_causal_free(
        self,
        n_draws: int,
        *,
        seed: int,
        max_residues: int = 400,
        temperature: float = 1.0,
        top_p: float = 0.95,
        batch_size: int = 1,
    ) -> list[dict[str, Any]]:
        """Unconditional free-length causal sequence generation up to ``max_residues``."""
        eos_set = set(self.eos_suffix_ids)
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            eos_set.add(int(self.tokenizer.eos_token_id))
        draws: list[dict[str, Any]] = []
        for idx in range(int(n_draws)):
            draw_seed = int(seed) + idx
            tokens = self._sample_tokens(
                list(self.bos_prefix_ids),
                max_new_tokens=int(max_residues) + 16,
                seed=draw_seed,
                temperature=temperature,
                top_p=top_p,
                stop_ids=eos_set,
            )
            residues: list[str] = []
            terminated = False
            off_alphabet = False
            for tid in tokens:
                if tid in eos_set:
                    terminated = True
                    break
                if tid in self.id_to_aa:
                    residues.append(self.id_to_aa[tid])
                    if len(residues) >= max_residues:
                        break
                else:
                    off_alphabet = True
                    break
            seq = "".join(residues)
            draws.append(
                {
                    "draw_index": idx,
                    "seed": draw_seed,
                    "sequence": seq,
                    "length": len(seq),
                    "terminated_eos": terminated,
                    "off_alphabet": off_alphabet,
                }
            )
        return draws


def _load_protgpt3_causal(
    spec: GeneralPLMSpec, checkpoint_dir: Path, *, device: str, dtype: str
) -> _BaseCausalAdapter:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dt = torch_dtype_of(dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), local_files_only=True)
    model, info = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        local_files_only=True,
        output_loading_info=True,
        **_from_pretrained_dtype_kwarg(AutoModelForCausalLM, torch_dt),
    )
    unmatched = {k: info[k] for k in ("missing_keys", "mismatched_keys") if info.get(k)}
    if unmatched:
        raise RuntimeError(f"{spec.arm}: ProtGPT3 weight mismatch: {unmatched}")
    model = model.eval().to(device)

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.convert_tokens_to_ids("<|bos|>")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.convert_tokens_to_ids("<|eos|>")
    dir_id = tokenizer.convert_tokens_to_ids("1")
    if bos_id is None or dir_id is None:
        raise ValueError(
            f"{spec.arm}: ProtGPT3 tokenizer missing <|bos|> ({bos_id}) or direction '1' ({dir_id})"
        )
    aa_ids = _resolve_aa20_ids(tokenizer, spec.arm)

    # ProtGPT3-MSA uses `<|bos|> 1 <no_gap>` prefix and `<s>` between concatenated homologs.
    bos_prefix = [int(bos_id), int(dir_id)]
    sep_ids: list[int] = []
    if spec.family == "protgpt3_msa":
        no_gap_id = tokenizer.convert_tokens_to_ids("<no_gap>")
        s_sep_id = tokenizer.convert_tokens_to_ids("<s>")
        if no_gap_id is not None and no_gap_id != getattr(tokenizer, "unk_token_id", None):
            bos_prefix.append(int(no_gap_id))
        if s_sep_id is not None and s_sep_id != getattr(tokenizer, "unk_token_id", None):
            sep_ids = [int(s_sep_id)]
        elif eos_id is not None:
            sep_ids = [int(eos_id)]
    return _BaseCausalAdapter(
        spec,
        checkpoint_dir,
        device=device,
        dtype=dtype,
        model=model,
        tokenizer=tokenizer,
        bos_prefix_ids=bos_prefix,
        eos_suffix_ids=[],
        separator_ids=sep_ids,
        aa20_ids=aa_ids,
        extra_provenance={
            "packing": "protgpt3_msa_whitespace_wordlevel" if spec.family == "protgpt3_msa" else "bos_direction_seq",
            "bos_prefix_ids": bos_prefix,
            "separator_ids": sep_ids,
            "unexpected_keys": list(info.get("unexpected_keys", ())),
            "model_class": type(model).__name__,
        },
    )


def _load_dayhoff_causal(
    spec: GeneralPLMSpec, checkpoint_dir: Path, *, device: str, dtype: str
) -> _BaseCausalAdapter:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch_dt = torch_dtype_of(dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_dir), local_files_only=True, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(
        str(checkpoint_dir), local_files_only=True, trust_remote_code=True
    )
    # Dayhoff JambaForCausalLM sets num_logits_to_keep=1 in config.json; override to 0 for full-sequence NLL.
    if hasattr(config, "num_logits_to_keep"):
        config.num_logits_to_keep = 0
    # Disable MoE router-logit aux loss during inference (config.json sets output_router_logits=true,
    # which can crash inside Jamba load_balancing_loss_func during label-free forward passes).
    if hasattr(config, "output_router_logits"):
        config.output_router_logits = False
    # Fall back to PyTorch naive Mamba implementation when fast CUDA mamba_ssm/causal_conv1d kernels are absent.
    mamba_fast_available = False
    try:
        import causal_conv1d  # noqa: F401
        import mamba_ssm  # noqa: F401

        mamba_fast_available = True
    except ImportError:
        mamba_fast_available = False
    if hasattr(config, "use_mamba_kernels") and not mamba_fast_available:
        config.use_mamba_kernels = False
    model, info = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        config=config,
        local_files_only=True,
        trust_remote_code=True,
        output_loading_info=True,
        **_from_pretrained_dtype_kwarg(AutoModelForCausalLM, torch_dt),
    )
    unmatched = {k: info[k] for k in ("missing_keys", "mismatched_keys") if info.get(k)}
    if unmatched:
        raise RuntimeError(f"{spec.arm}: Dayhoff weight mismatch: {unmatched}")
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = False
    model = model.eval().to(device)

    bos_id = getattr(config, "bos_token_id", None)
    if bos_id is None:
        bos_id = tokenizer.convert_tokens_to_ids("@")
    eos_id = getattr(config, "eos_token_id", None)
    if eos_id is None:
        eos_id = tokenizer.convert_tokens_to_ids("*")
    sep_id = tokenizer.convert_tokens_to_ids("/")
    if sep_id is None or sep_id == getattr(tokenizer, "unk_token_id", None):
        sep_id = eos_id
    bos_prefix = [int(bos_id)] if bos_id is not None else []
    sep_ids = [int(sep_id)] if sep_id is not None else []
    aa_ids = _resolve_aa20_ids(tokenizer, spec.arm)
    return _BaseCausalAdapter(
        spec,
        checkpoint_dir,
        device=device,
        dtype=dtype,
        model=model,
        tokenizer=tokenizer,
        bos_prefix_ids=bos_prefix,
        eos_suffix_ids=[],
        separator_ids=sep_ids,
        aa20_ids=aa_ids,
        extra_provenance={
            "packing": "dayhoff_jamba_explicit_bos_sep",
            "bos_prefix_ids": bos_prefix,
            "eos_token_id": int(eos_id) if eos_id is not None else None,
            "separator_ids": sep_ids,
            "num_logits_to_keep_overridden_to": 0,
            "output_router_logits_overridden_to": False,
            "unexpected_keys": list(info.get("unexpected_keys", ())),
            "model_class": type(model).__name__,
        },
    )


# --------------------------------------------------- ProGen3 Reuse Adapter


class ProGen3CausalAdapter(GeneralPLMAdapter):
    """Wraps ``src.capability.models.progen3`` for all ProGen3 ladder rungs (219M/339M/762M/1B)."""

    def __init__(
        self,
        spec: GeneralPLMSpec,
        checkpoint_dir: Path,
        *,
        device: str,
        dtype: str,
    ) -> None:
        from ..models import progen3 as pg3

        torch_dt = torch_dtype_of(dtype)
        self._pg3 = pg3
        self.handle = pg3.load_progen3(checkpoint_dir, device=device, dtype=torch_dt)
        super().__init__(
            spec,
            checkpoint_dir,
            device=device,
            dtype=dtype,
            model=self.handle.model,
            tokenizer=self.handle.tokenizer,
            extra_provenance={
                "packing": "progen3_clm_megablocks_eager_strict",
                "n_layers": self.handle.n_layers,
                "n_heads": self.handle.n_heads,
                "singleton_only": True,
            },
        )
        self.aa_to_id = {
            aa: int(self.tokenizer.encode(aa).ids[0])
            if hasattr(self.tokenizer.encode(aa), "ids")
            else int(self.tokenizer.convert_tokens_to_ids(aa))
            for aa in AA20
        }
        self.id_to_aa = {tid: aa for aa, tid in self.aa_to_id.items()}

    @torch.no_grad()
    def exact_pll(
        self,
        sequence: str,
        *,
        batch_size: int = 1,
        structure_3di: str | None = None,
    ) -> PositionVector:
        self.validate_sequence(sequence)
        batch = self.handle.batch([sequence], reverse=False)
        out = self._pg3.forward(self.handle, batch)
        logits = out.logits[0].float()  # (T, V)
        ids = batch["input_ids"][0]
        # ProGen3 brackets residues between <bos> (1) and '1' (prefix len 2) and trailing '2', <eos>.
        start = 2
        end = start + len(sequence)
        terms = F.cross_entropy(logits[start - 1 : end - 1], ids[start:end], reduction="none")
        if not torch.isfinite(terms).all():
            raise FloatingPointError(f"{self.arm}: non-finite ProGen3 NLL term")
        nll_sum = float(terms.sum())
        return PositionVector(
            terms=terms.detach().cpu().numpy().astype(np.float32, copy=False),
            nll_sum=nll_sum,
            pll=-nll_sum,
            paradigm=self.spec.paradigm,
        )

    def score_variants(
        self,
        wildtype: str,
        mutations: Sequence[str],
        *,
        batch_size: int = 1,
        multi_exact_pll: bool = False,
        structure_3di: str | None = None,
    ) -> np.ndarray:
        wt_pll = self.exact_pll(wildtype).pll
        values = np.empty(len(mutations), dtype=np.float64)
        for i, mut in enumerate(mutations):
            values[i] = float(self.exact_pll(apply_mutation(wildtype, mut)).pll - wt_pll)
        return values

    def native_site_conditional(
        self,
        sequence: str,
        index: int,
        *,
        structure_3di: str | None = None,
    ) -> np.ndarray:
        scores = np.empty(len(AA20), dtype=np.float64)
        for idx, aa in enumerate(AA20):
            cand = sequence[:index] + aa + sequence[index + 1 :]
            scores[idx] = self.exact_pll(cand).pll
        peak = float(scores.max())
        return scores - (peak + math.log(float(np.exp(scores - peak).sum())))

    @torch.no_grad()
    def extract_hidden_states(
        self,
        sequences: Sequence[str],
        *,
        batch_size: int = 1,
        pooling: str = "mean_last",
        structure_3di_list: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        if not sequences:
            return np.zeros((0, 4, 0), dtype=np.float32)
        blocks = self.handle.model.model.layers
        mid_idx = (len(blocks) - 1) // 2
        fin_idx = len(blocks) - 1
        captured: dict[str, torch.Tensor] = {}

        def _make_hook(label: str):
            def _hook(_mod, _inp, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[label] = h.detach()
            return _hook

        handles = [
            blocks[mid_idx].register_forward_hook(_make_hook("mid")),
            blocks[fin_idx].register_forward_hook(_make_hook("fin")),
        ]
        rows: list[np.ndarray] = []
        try:
            for seq in sequences:
                self.validate_sequence(seq)
                batch = self.handle.batch([seq], reverse=False)
                captured.clear()
                _ = self._pg3.forward(self.handle, batch)
                start = 2
                end = start + len(seq)
                h_mid = captured["mid"][0, start:end].float()
                h_fin = captured["fin"][0, start:end].float()
                vec = torch.stack([h_mid.mean(0), h_mid[-1], h_fin.mean(0), h_fin[-1]], dim=0)
                rows.append(vec.cpu().numpy().astype(np.float32, copy=False))
        finally:
            for handle in handles:
                handle.remove()
        return np.stack(rows, axis=0)

    @torch.no_grad()
    def score_r4_context(
        self,
        target: str,
        context_items: Sequence[str],
        *,
        budget_tokens: int = 1024,
        batch_size: int = 1,
        aligned_msa: bool = True,
    ) -> dict[str, Any]:
        target_batch = self.handle.batch([target], reverse=False)
        target_ids = target_batch["input_ids"][0].tolist()
        remaining = int(budget_tokens) - len(target_ids)
        prefix: list[int] = []
        used_k = 0
        for item in context_items:
            item_ids = self.handle.batch([item], reverse=False)["input_ids"][0].tolist()
            if len(prefix) + len(item_ids) > remaining:
                break
            prefix.extend(item_ids)
            used_k += 1
        full_ids = prefix + target_ids
        input_ids = torch.as_tensor([full_ids], dtype=torch.long, device=self.device)
        pos_ids = torch.arange(len(full_ids), dtype=torch.long, device=self.device).unsqueeze(0)
        seq_ids = torch.zeros_like(input_ids)
        out = self.handle.model(
            input_ids=input_ids, position_ids=pos_ids, sequence_ids=seq_ids, use_cache=False
        )
        logits = out.logits[0].float()
        span_start = len(prefix) + 2
        span_end = span_start + len(target)
        terms = F.cross_entropy(logits[span_start - 1 : span_end - 1], input_ids[0, span_start:span_end], reduction="none")
        nll_sum = float(terms.sum())
        return {
            "nll_sum": nll_sum,
            "nll_per_residue": nll_sum / len(target),
            "scored_tokens": len(target),
            "total_tokens": len(full_ids),
            "k_items": used_k,
        }

    @torch.no_grad()
    def _sample_progen3(
        self, prefix_ids: list[int], *, max_new_tokens: int, seed: int, temperature: float, top_p: float
    ) -> list[int]:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        cur = list(prefix_ids)
        produced: list[int] = []
        stop_ids = {2, 0}  # ProGen3 EOS/end markers
        for _ in range(max_new_tokens):
            inp = torch.as_tensor([cur], dtype=torch.long, device=self.device)
            pos = torch.arange(len(cur), dtype=torch.long, device=self.device).unsqueeze(0)
            seq_ids = torch.zeros_like(inp)
            out = self.handle.model(
                input_ids=inp, position_ids=pos, sequence_ids=seq_ids, use_cache=False
            )
            logits = out.logits[0, -1].float() / float(temperature)
            lp = _nucleus_logprobs(F.log_softmax(logits, dim=-1), float(top_p))
            nxt = int(torch.multinomial(lp.exp().cpu(), num_samples=1, generator=gen).item())
            produced.append(nxt)
            cur.append(nxt)
            if nxt in stop_ids:
                break
        return produced

    def infill_window(
        self,
        parent: str,
        *,
        start: int,
        extent: int,
        seed: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        structure_3di: str | None = None,
    ) -> dict[str, Any]:
        stop = start + extent
        full_prefix = self.handle.batch([parent[:start]], reverse=False)["input_ids"][0].tolist()
        # Drop trailing end/eos tokens (last 2 tokens of ProGen3BatchPreparer)
        prompt = full_prefix[:-2]
        tokens = self._sample_progen3(
            prompt, max_new_tokens=extent + 8, seed=seed, temperature=temperature, top_p=top_p
        )
        residues = [self.id_to_aa[t] for t in tokens if t in self.id_to_aa][:extent]
        status = "filled" if len(residues) == extent else "terminated_short"
        window = "".join(residues)
        return {
            "status": status,
            "window": window,
            "sequence": parent[:start] + window + parent[stop:] if status == "filled" else None,
            "sampler": "progen3_causal_prefix_regeneration",
            "temperature": float(temperature),
            "top_p": float(top_p),
        }

    def generate_causal_free(
        self,
        n_draws: int,
        *,
        seed: int,
        max_residues: int = 400,
        temperature: float = 1.0,
        top_p: float = 0.95,
        batch_size: int = 1,
    ) -> list[dict[str, Any]]:
        prompt = self.handle.batch(["A"], reverse=False)["input_ids"][0].tolist()[:2]
        draws: list[dict[str, Any]] = []
        for idx in range(int(n_draws)):
            tokens = self._sample_progen3(
                prompt,
                max_new_tokens=int(max_residues) + 8,
                seed=int(seed) + idx,
                temperature=temperature,
                top_p=top_p,
            )
            residues = [self.id_to_aa[t] for t in tokens if t in self.id_to_aa][:max_residues]
            seq = "".join(residues)
            draws.append(
                {
                    "draw_index": idx,
                    "seed": int(seed) + idx,
                    "sequence": seq,
                    "length": len(seq),
                    "terminated_eos": len(residues) < max_residues,
                    "off_alphabet": False,
                }
            )
        return draws


# --------------------------------------------------- MSA Transformer Adapter


class MsaTransformerAdapter(GeneralPLMAdapter):
    """2D axial-attention MSA Transformer adapter for R4 homolog-aligned context scoring."""

    def __init__(
        self,
        spec: GeneralPLMSpec,
        checkpoint_dir: Path,
        *,
        device: str,
        dtype: str,
        model: torch.nn.Module,
        tokenizer: Any,
        backend: str,
        mask_id: int,
        aa20_ids: list[int],
        cls_id: int,
        gap_id: int,
        extra_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            spec,
            checkpoint_dir,
            device=device,
            dtype=dtype,
            model=model,
            tokenizer=tokenizer,
            extra_provenance={"backend": backend, **(dict(extra_provenance) if extra_provenance else {})},
        )
        self.backend = backend
        self.mask_id = int(mask_id)
        self.cls_id = int(cls_id)
        self.gap_id = int(gap_id)
        self.aa_to_id = {aa: int(tid) for aa, tid in zip(AA20, aa20_ids)}

    def _encode_msa(self, aligned_rows: Sequence[str]) -> torch.Tensor:
        """Encode ``(num_seqs, L)`` aligned strings into ``(1, num_seqs, L + 1)`` token ids."""
        length = len(aligned_rows[0])
        mat = np.full((len(aligned_rows), length + 1), self.gap_id, dtype=np.int64)
        mat[:, 0] = self.cls_id
        for r_idx, row in enumerate(aligned_rows):
            if len(row) != length:
                raise ValueError("All MSA rows must share exact aligned length")
            for c_idx, ch in enumerate(row):
                if ch == "-":
                    mat[r_idx, c_idx + 1] = self.gap_id
                elif ch in self.aa_to_id:
                    mat[r_idx, c_idx + 1] = self.aa_to_id[ch]
                else:
                    mat[r_idx, c_idx + 1] = self.gap_id
        return torch.as_tensor(mat, dtype=torch.long, device=self.device).unsqueeze(0)

    def _forward_msa_logits(self, tokens_3d: torch.Tensor) -> torch.Tensor:
        if self.backend == "fair_esm":
            out = self.model(tokens_3d, repr_layers=[], need_head_weights=False)
            return out["logits"].float()
        if self.backend == "pure_pytorch_esm_msa":
            out = self.model(tokens_3d)
            return out["logits"].float()
        out = self.model(input_ids=tokens_3d)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return logits.float()

    @torch.no_grad()
    def score_r4_context(
        self,
        target: str,
        context_items: Sequence[str],
        *,
        budget_tokens: int = 32,
        batch_size: int = 8,
        aligned_msa: bool = True,
    ) -> dict[str, Any]:
        """Exact leave-one-out masked-marginal PLL on row 0 (target) of the query-coordinate MSA."""
        self.validate_sequence(target)
        # For MSA Transformer, budget_tokens >= 64 is treated as max MSA depth 32; explicit small int is depth.
        max_depth = 32 if int(budget_tokens) > 64 else max(1, int(budget_tokens))
        msa_rows = build_query_msa(target, context_items, max_depth=max_depth, aligned=aligned_msa)
        base_tokens = self._encode_msa(msa_rows)  # (1, D, L + 1)
        length = len(target)
        terms = torch.empty(length, dtype=torch.float32, device=self.device)
        bs = max(1, int(batch_size))

        with self._precision_ctx():
            for start in range(0, length, bs):
                stop = min(start + bs, length)
                batch = base_tokens.repeat(stop - start, 1, 1)  # (B, D, L + 1)
                for offset, pos in enumerate(range(start, stop)):
                    batch[offset, 0, pos + 1] = self.mask_id
                logits = self._forward_msa_logits(batch)  # (B, D, L + 1, V)
                for offset, pos in enumerate(range(start, stop)):
                    lp = F.log_softmax(logits[offset, 0, pos + 1], dim=-1)
                    terms[pos] = -lp[base_tokens[0, 0, pos + 1]]
                del logits

        nll_sum = float(terms.sum())
        return {
            "nll_sum": nll_sum,
            "nll_per_residue": nll_sum / length,
            "scored_tokens": length,
            "msa_depth": len(msa_rows),
            "k_items": len(msa_rows) - 1,
            "aligned_msa": bool(aligned_msa),
            "likelihood_kind": "msa_masked_pseudo_log_likelihood",
        }


def _load_msa_transformer(
    spec: GeneralPLMSpec, checkpoint_dir: Path, *, device: str, dtype: str
) -> MsaTransformerAdapter:
    torch_dt = torch_dtype_of(dtype)
    # Try HF AutoModelForMaskedLM first (katielink/esm_msa1b_t12_100M_UR50S).
    if (checkpoint_dir / "config.json").is_file():
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_dir), local_files_only=True, trust_remote_code=True
        )
        model, info = AutoModelForMaskedLM.from_pretrained(
            str(checkpoint_dir),
            local_files_only=True,
            trust_remote_code=True,
            output_loading_info=True,
            **_from_pretrained_dtype_kwarg(AutoModelForMaskedLM, torch_dt),
        )
        unmatched = {k: info[k] for k in ("missing_keys", "mismatched_keys") if info.get(k)}
        if unmatched:
            raise RuntimeError(f"{spec.arm}: MSA Transformer weight mismatch: {unmatched}")
        model = model.eval().to(device)
        cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0
        gap_id = tokenizer.convert_tokens_to_ids("-")
        if gap_id is None:
            gap_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
        return MsaTransformerAdapter(
            spec,
            checkpoint_dir,
            device=device,
            dtype=dtype,
            model=model,
            tokenizer=tokenizer,
            backend="hf_auto",
            mask_id=_resolve_mask_id(tokenizer, spec.arm),
            aa20_ids=_resolve_aa20_ids(tokenizer, spec.arm),
            cls_id=int(cls_id),
            gap_id=int(gap_id),
            extra_provenance={"model_class": type(model).__name__},
        )

    # Standalone fair-esm .pt checkpoint (e.g. katielink/esm_msa1b_t12_100M_UR50S/esm_msa1b_t12_100M_UR50S.pt).
    pt_files = sorted(checkpoint_dir.glob("*.pt"))
    if pt_files:
        try:
            import esm  # type: ignore

            model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(pt_files[0]))
            model = model.eval().to(device=device, dtype=torch_dt)
            aa_ids = [int(alphabet.get_idx(aa)) for aa in AA20]
            return MsaTransformerAdapter(
                spec,
                checkpoint_dir,
                device=device,
                dtype=dtype,
                model=model,
                tokenizer=alphabet,
                backend="fair_esm",
                mask_id=int(alphabet.mask_idx),
                aa20_ids=aa_ids,
                cls_id=int(alphabet.cls_idx),
                gap_id=int(alphabet.get_idx("-")),
                extra_provenance={"pt_file": pt_files[0].name},
            )
        except ImportError:
            from .msa_transformer_model import (
                MSA_CLS_IDX,
                MSA_GAP_IDX,
                MSA_MASK_IDX,
                MSA_TOKEN_TO_ID,
                load_pure_pytorch_msa_transformer,
            )

            model, meta = load_pure_pytorch_msa_transformer(
                pt_files[0], device=device, dtype=torch_dt
            )
            aa_ids = [int(MSA_TOKEN_TO_ID[aa]) for aa in AA20]
            return MsaTransformerAdapter(
                spec,
                checkpoint_dir,
                device=device,
                dtype=dtype,
                model=model,
                tokenizer=MSA_TOKEN_TO_ID,
                backend="pure_pytorch_esm_msa",
                mask_id=MSA_MASK_IDX,
                aa20_ids=aa_ids,
                cls_id=MSA_CLS_IDX,
                gap_id=MSA_GAP_IDX,
                extra_provenance={**meta, "model_class": type(model).__name__},
            )

    raise FileNotFoundError(f"{spec.arm}: no config.json or .pt checkpoint under {checkpoint_dir}")


# -------------------------------------------------------------- Public Factory


def load_general_plm(
    arm: str,
    *,
    device: str = "cuda:0",
    dtype: str | None = None,
    model_root: Path | str | None = None,
    saprot_mode: str = "seq_only",
) -> GeneralPLMAdapter:
    """Resolve and load any declared general-PLM arm with strict offline validation."""
    spec = get_spec(arm)
    resolved_dtype = normalize_dtype_name(spec, dtype)
    checkpoint_dir = resolve_checkpoint_dir(arm, model_root=model_root)

    kind = spec.loader_kind
    if kind in ("esm_mlm", "prime_mlm"):
        return _load_esm_mlm(spec, checkpoint_dir, device=device, dtype=resolved_dtype)
    if kind == "esmc_hf":
        return _load_esmc_hf(spec, checkpoint_dir, device=device, dtype=resolved_dtype)
    if kind == "amplify_mlm":
        return _load_amplify_mlm(spec, checkpoint_dir, device=device, dtype=resolved_dtype)
    if kind == "saprot_mlm":
        return _load_saprot_mlm(
            spec, checkpoint_dir, device=device, dtype=resolved_dtype, saprot_mode=saprot_mode
        )
    if kind == "proteinglm_mlm":
        return _load_proteinglm_mlm(spec, checkpoint_dir, device=device, dtype=resolved_dtype)
    if kind == "protgpt3_causal":
        return _load_protgpt3_causal(spec, checkpoint_dir, device=device, dtype=resolved_dtype)
    if kind == "progen3_causal":
        return ProGen3CausalAdapter(spec, checkpoint_dir, device=device, dtype=resolved_dtype)
    if kind == "dayhoff_causal":
        return _load_dayhoff_causal(spec, checkpoint_dir, device=device, dtype=resolved_dtype)
    if kind == "msa_transformer":
        return _load_msa_transformer(spec, checkpoint_dir, device=device, dtype=resolved_dtype)
    raise ValueError(f"{arm}: unhandled loader_kind={kind!r}")
