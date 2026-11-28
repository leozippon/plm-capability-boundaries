"""Unified 4-family open-source PLM information evaluator (`open_model_audit`).

Supports four open-source PLM families with strictly uniform 20-canonical-AA
logit renormalization ($H_0 = \\log_2 20 = 4.321928$ bits/res):

1. **Causal AR PLMs** (``CausalAROpenEvaluator``):
   ProGen2 (small/base/medium/large/xlarge with native ``'1'``/``'2'`` direction
   tags), RITA, ProtGPT3, ProGen3, Dayhoff, Tranception, and generic HF CausalLM.
2. **Masked PLMs** (``MaskedLMOpenEvaluator``):
   ESM-2 (8M-15B), ESM-1b, AMPLIFY, ESM-C, ProteinGLM-MLM, Prime-690M,
   ANKH/ProtBert/CARP, and generic HF MaskedLM — measuring ALL THREE readouts:
   in-distribution 15%-mask CE (8 draws/seq), single-site LOO (16 probes/seq),
   AND 16-stratum MDM/OA joint NELBO (plus per-stratum curve testing high-mask
   OOD breakdown!).
3. **Discrete Diffusion / Order-Agnostic PLMs** (``MDMOpenEvaluator``):
   DPLM (150M-3B), EvoDiff-OA, and Stage-1/Stage-2 MDM checkpoints — primary
   16-stratum joint NELBO + 15%-mask + 16-site LOO + Eq. (3b) $\\Delta_{\\mathrm{cal}}$.
4. **Structure-Conditioned & Multi-Track PLMs**:
   - ``ProteinMPNNOpenEvaluator`` (vanilla/soluble/CA-only): permutation-averaged
     AR joint entropy + LOO strictly over AA20 (stripping ``X``) on 3D backbones.
   - ``SaProtOpenEvaluator`` (35M/650M/1.3B): paired $H(X \\mid Z=\\varnothing)$
     (``#``-ablated) vs $H(X \\mid Z_{\\mathrm{3D}})$ (Foldseek 3Di) -> exact
     $b_{\\mathrm{ext}}(Z)$.
   - ``DPLM2OpenEvaluator`` (150M/650M/bit-650M/3B): paired ``seq_only`` vs
     ``struct_conditioned`` (StructTokenizerEncoder) -> exact $b_{\\mathrm{ext}}(Z)$.
   - ``ESM3OpenEvaluator`` (esm3-sm-open-v1): unconditioned vs 3D-conditioned
     sequence track -> exact $b_{\\mathrm{ext}}(Z)$ when the ``esm`` runtime is
     installed.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.capability.core.precision_policy import fp32_matmul_context

from .cohort import AuditSequenceRecord, OpenAuditBundle, Paired3DRecord, _now_utc
from .metrics import (
    AA20,
    AA20_INDEX,
    AA20_SET,
    DEFAULT_LOO_PROBES,
    DEFAULT_MDM_STRATA,
    DEFAULT_MLM15_DRAWS,
    DEPTH_BUCKET_ORDER,
    H0_BITS,
    LN2,
    aa20_restricted_logits,
    compute_horvitz_thompson_corpus_estimate,
    compute_multitrack_b_ext,
    compute_temperature_calibrated_bits,
    compute_three_tier_decomposition,
    count_model_parameters,
    deterministic_fixed_rate_mask,
    deterministic_loo_positions,
    deterministic_mdm_stratum_mask,
)

AUDIT_REPORT_SCHEMA = "open_model_audit_report_v1"

DEFAULT_CAUSAL_MODEL_ROOTS: tuple[Path, ...] = (
    Path("/Data/public/models_R2"),
    Path("/home/lzp/models/general_plm_20261120"),
    Path("/Data/lzp/models/general_plm_20261120"),
    Path("/Data/public"),
    Path("/mnt/c20250608_hs/liuzipeng/models"),
    Path("/mnt/c20250608_hs/Ming/model"),
    Path("/mnt/c20250608_hs/Ming/model/ESM-2"),
    Path("/mnt/c20250608_hs/Ming/model/DPLM"),
    Path("/mnt/c20250608_hs/Ming/model/Prot-T5"),
    Path("/mnt/c20250608_hs/Ming/model/ProtGPT3"),
    Path("/mnt/c20250608/heliang/checkpoint"),
    Path("/mnt/c20250608/heliang/checkpoint/esm"),
    Path("/mnt/c20250608/shared/checkpoint"),
    Path("/mnt/c20250608/shared/checkpoint/esm"),
)

_CUIHU_MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "esm2-8m": ("esm2-8m", "esm2_t6_8M_UR50D"),
    "esm2-35m": ("esm2-35m", "esm2_t12_35M_UR50D"),
    "esm2-150m": ("esm2-150m", "esm2_t30_150M_UR50D"),
    "esm2-650m": ("esm2-650m", "esm2_t33_650M_UR50D"),
    "esm2-3b": ("esm2-3b", "esm2_t36_3B_UR50D"),
    "esm2-15b": ("esm2-15b", "esm2_t48_15B_UR50D"),
    "dplm-150m": ("dplm-150m", "dplm_150m"),
    "dplm-650m": ("dplm-650m", "dplm_650m"),
    "dplm-3b": ("dplm-3b", "dplm_3b"),
    "esmc-300m": ("esmc-300m-hf", "ESMC-300M", "ESMC-300M-hf", "esmc-300m"),
    "esmc-600m": ("esmc-600m-hf", "ESMC-600M", "ESMC-600M-hf", "esmc-600m"),
    "esmc-6b": ("esmc-6b-hf", "ESMC-6B", "ESMC-6B-hf", "esmc-6b"),
    "saprot-35m": ("saprot-35m", "SaProt_35M_AF2"),
    "saprot-650m": ("saprot-650m", "SaProt_650M_AF2"),
    "saprot-650m-pdb": ("SaProt_650M_PDB", "saprot-650m-pdb"),
    "saprot-1.3b": ("saprot-1.3b", "SaProt_1.3B_AF2", "SaProt_1.3B_AFDB_OMG_NCBI"),
    "saprot-1.3b-afdb-omg-ncbi": ("SaProt_1.3B_AFDB_OMG_NCBI",),
    "protgpt3-112m": ("ProtGPT3-112M", "protgpt3-112m"),
    "protgpt3-112m-new": ("ProtGPT3-112M_new",),
    "protgpt3-1.3b": ("ProtGPT3-1.3B", "protgpt3-1.3b"),
    "protgpt3-10b": ("ProtGPT3-10B", "protgpt3-10b"),
    "progen3-112m": ("progen3-112m", "ProGen3-112M"),
    "prot-t5-xl": ("prot_t5_xl_uniref50",),
    "prot-t5-xxl": ("prot_t5_xxl_uniref50",),
    "galactica-125m": ("galactica-125m",),
    "galactica-1.3b": ("galactica-1.3b",),
    "galactica-6.7b": ("galactica-6.7b",),
    "galactica-30b": ("galactica-30b",),
}

_PROGEN2_NAMES: frozenset[str] = frozenset(
    {
        "progen2-small",
        "progen2-base",
        "progen2-medium",
        "progen2-large",
        "progen2-xlarge",
        "progen2-small-hugohrban",
        "progen2-small-mix7",
    }
)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return str(device)


def _resolve_torch_dtype(dtype: str) -> tuple[str, torch.dtype]:
    d = (dtype or "float32").strip().lower()
    if d in ("fp32", "float32", "auto", "default"):
        return "float32", torch.float32
    if d in ("bf16", "bfloat16"):
        return "bfloat16", torch.bfloat16
    if d in ("fp16", "float16"):
        return "float16", torch.float16
    raise ValueError(f"Unsupported dtype {dtype!r}; use float32, bfloat16, or float16")


def _precision_ctx(device: str, dtype_name: str):
    if dtype_name == "float32" and device.startswith("cuda"):
        return fp32_matmul_context(torch)
    return nullcontext()


def _summarize_seq_bits(
    seq_bits: Sequence[float],
    seq_lengths: Sequence[int],
    *,
    logits_20: np.ndarray | None = None,
    targets_20: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Aggregate residue-weighted mean bits/res, SE, and Eq. (3b) temperature calibration."""
    if not seq_bits:
        return {
            "n_sequences": 0,
            "n_residues": 0,
            "bits_per_res": float("nan"),
            "seq_mean_bits_per_res": float("nan"),
            "se_bits_per_res": float("nan"),
            "delta_h_bits_per_res": float("nan"),
            "calibrated_bits_per_res": float("nan"),
            "optimal_temperature": float("nan"),
            "delta_cal_bits": float("nan"),
        }
    b_arr = np.asarray(seq_bits, dtype=np.float64)
    l_arr = np.asarray(seq_lengths, dtype=np.float64)
    tot_res = float(l_arr.sum())
    w_mean = float(np.sum(b_arr * l_arr) / max(tot_res, 1.0))
    seq_mean = float(np.mean(b_arr))
    if len(b_arr) > 1:
        # Delta-method / ratio-estimator variance for residue-weighted mean sum(L_i * h_i) / sum(L_i)
        resid = l_arr * (b_arr - w_mean)
        n = float(len(b_arr))
        var_w = float((n / (n - 1.0)) * np.sum(resid**2) / (max(tot_res, 1.0) ** 2))
        se_w = math.sqrt(max(0.0, var_w))
    else:
        se_w = 0.0

    out: dict[str, Any] = {
        "n_sequences": int(len(b_arr)),
        "n_residues": int(tot_res),
        "bits_per_res": w_mean,
        "seq_mean_bits_per_res": seq_mean,
        "se_bits_per_res": se_w,
        "delta_h_bits_per_res": float(H0_BITS - w_mean),
    }
    if logits_20 is not None and targets_20 is not None and len(targets_20) > 0:
        cal = compute_temperature_calibrated_bits(logits_20, targets_20, weights=weights)
        out.update(
            {
                "raw_calibration_check_bits_per_res": cal["raw_bits_per_res"],
                "calibrated_bits_per_res": cal["calibrated_bits_per_res"],
                "optimal_temperature": cal["optimal_temperature"],
                "delta_cal_bits": cal["delta_cal_bits"],
            }
        )
    else:
        out.update(
            {
                "calibrated_bits_per_res": float("nan"),
                "optimal_temperature": float("nan"),
                "delta_cal_bits": float("nan"),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Base Evaluator Interface
# --------------------------------------------------------------------------- #


class OpenModelAuditEvaluator:
    """Common interface for all 4 open-model audit families."""

    def __init__(
        self,
        *,
        model_id: str,
        family_kind: str,  # "causal_ar" | "mlm" | "mdm" | "structure_conditioned"
        primary_readout: str,
        n_params_total: int,
        n_params_emb: int,
        n_params_non_emb: int,
        device: str = "cpu",
        dtype: str = "float32",
        supports_sequence_splits: bool = True,
        supports_3d_panel: bool = False,
        supports_3d_ablation: bool = False,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_id = str(model_id)
        self.family_kind = str(family_kind)
        self.primary_readout = str(primary_readout)
        self.n_params_total = int(n_params_total)
        self.n_params_emb = int(n_params_emb)
        self.n_params_non_emb = int(n_params_non_emb)
        self.device = str(device)
        self.dtype = str(dtype)
        self.supports_sequence_splits = bool(supports_sequence_splits)
        self.supports_3d_panel = bool(supports_3d_panel)
        self.supports_3d_ablation = bool(supports_3d_ablation)
        self.extra_metadata = dict(extra_metadata or {})

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family_kind": self.family_kind,
            "primary_readout": self.primary_readout,
            "n_params_total": self.n_params_total,
            "n_params_emb": self.n_params_emb,
            "n_params_non_emb": self.n_params_non_emb,
            "device": self.device,
            "dtype": self.dtype,
            "supports_sequence_splits": self.supports_sequence_splits,
            "supports_3d_panel": self.supports_3d_panel,
            "supports_3d_ablation": self.supports_3d_ablation,
            **self.extra_metadata,
        }

    def evaluate_sequence_split(
        self,
        records: Sequence[AuditSequenceRecord],
        *,
        batch_size: int = 16,
        n_mlm15_draws: int = DEFAULT_MLM15_DRAWS,
        n_loo_probes: int = DEFAULT_LOO_PROBES,
        n_mdm_strata: int = DEFAULT_MDM_STRATA,
        eval_seed: int = 20261129,
        max_calibration_positions: int = 65_536,
    ) -> dict[str, dict[str, Any]]:
        """Evaluate a sequence-only split; returns ``{readout_name: summary_dict}``."""
        raise NotImplementedError

    def evaluate_3d_panel(
        self,
        records: Sequence[Paired3DRecord],
        coords_by_key: Mapping[str, np.ndarray],
        *,
        batch_size: int = 16,
        n_mlm15_draws: int = DEFAULT_MLM15_DRAWS,
        n_loo_probes: int = DEFAULT_LOO_PROBES,
        n_mdm_strata: int = DEFAULT_MDM_STRATA,
        eval_seed: int = 20261129,
        seq_reference_bits: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate the 3D paired panel (natural vs de novo) and $b_{\\mathrm{ext}}(Z)$."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Helper: Subsampling Cap for Calibration Buffers (Memory-Safe)
# --------------------------------------------------------------------------- #


class _CalibrationBuffer:
    """Reservoir-capped buffer storing ``(logits_20, targets_20, weights)`` for Eq. (3b)."""

    def __init__(self, max_positions: int = 65_536, seed: int = 20261199) -> None:
        self.max_positions = max(1024, int(max_positions))
        self._rng = np.random.default_rng(int(seed))
        self._logits: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []
        self._weights: list[np.ndarray] = []
        self._count = 0

    def add(
        self,
        logits_20: np.ndarray,
        targets_20: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        z = np.asarray(logits_20, dtype=np.float32).reshape(-1, 20)
        y = np.asarray(targets_20, dtype=np.int64).reshape(-1)
        w = (
            np.ones(len(y), dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64).reshape(-1)
        )
        if len(y) == 0:
            return
        self._logits.append(z)
        self._targets.append(y)
        self._weights.append(w)
        self._count += len(y)

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._logits:
            return (
                np.zeros((0, 20), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float64),
            )
        z = np.concatenate(self._logits, axis=0)
        y = np.concatenate(self._targets, axis=0)
        w = np.concatenate(self._weights, axis=0)
        if len(y) > self.max_positions:
            idx = self._rng.choice(len(y), size=self.max_positions, replace=False)
            z, y, w = z[idx], y[idx], w[idx]
        return z, y, w


# --------------------------------------------------------------------------- #
# 1. Causal AR PLM Evaluator (ProGen2, RITA, ProtGPT3, ProGen3, Dayhoff, HF)
# --------------------------------------------------------------------------- #


class CausalAROpenEvaluator(OpenModelAuditEvaluator):
    """Exact AA20-restricted causal NLL + Eq. (3b) temperature scaling."""

    def __init__(
        self,
        *,
        model_id: str,
        model: nn.Module,
        aa20_ids: Sequence[int],
        bos_prefix_ids: Sequence[int],
        eos_suffix_ids: Sequence[int] = (),
        reverse_bos_prefix_ids: Sequence[int] | None = None,
        reverse_eos_suffix_ids: Sequence[int] = (),
        score_bidirectional: bool = False,
        max_positions: int = 1024,
        device: str = "cpu",
        dtype: str = "float32",
        forward_fn: Any | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        pcounts = count_model_parameters(model)
        super().__init__(
            model_id=model_id,
            family_kind="causal_ar",
            primary_readout="ar_exact",
            n_params_total=pcounts["n_params_total"],
            n_params_emb=pcounts["n_params_emb"],
            n_params_non_emb=pcounts["n_params_non_emb"],
            device=device,
            dtype=dtype,
            supports_sequence_splits=True,
            supports_3d_panel=False,
            supports_3d_ablation=False,
            extra_metadata={
                "bos_prefix_ids": [int(x) for x in bos_prefix_ids],
                "score_bidirectional": bool(score_bidirectional and reverse_bos_prefix_ids is not None),
                **(dict(extra_metadata) if extra_metadata else {}),
            },
        )
        self.model = model
        self.aa20_ids = [int(x) for x in aa20_ids]
        self.aa20_ids_t = torch.as_tensor(self.aa20_ids, dtype=torch.long, device=self.device)
        self.aa_to_id = {aa: tid for aa, tid in zip(AA20, self.aa20_ids)}
        self.bos_prefix_ids = [int(x) for x in bos_prefix_ids]
        self.eos_suffix_ids = [int(x) for x in eos_suffix_ids]
        self.reverse_bos_prefix_ids = (
            [int(x) for x in reverse_bos_prefix_ids] if reverse_bos_prefix_ids is not None else None
        )
        self.reverse_eos_suffix_ids = [int(x) for x in reverse_eos_suffix_ids]
        self.score_bidirectional = bool(score_bidirectional and self.reverse_bos_prefix_ids is not None)
        self.max_positions = int(max_positions)
        self._forward_fn = forward_fn

    def _forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._forward_fn is not None:
            return self._forward_fn(input_ids).float()
        kwargs: dict[str, Any] = {"input_ids": input_ids}
        try:
            out = self.model(**kwargs, use_cache=False)
        except TypeError:
            out = self.model(**kwargs)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return logits.float()

    @torch.no_grad()
    def _score_one_direction(
        self,
        records: Sequence[AuditSequenceRecord],
        *,
        reverse: bool = False,
        max_calibration_positions: int = 65_536,
    ) -> tuple[list[float], list[int], _CalibrationBuffer]:
        bos = self.reverse_bos_prefix_ids if (reverse and self.reverse_bos_prefix_ids is not None) else self.bos_prefix_ids
        eos = self.reverse_eos_suffix_ids if reverse else self.eos_suffix_ids
        seq_bits: list[float] = []
        seq_lens: list[int] = []
        cal_buf = _CalibrationBuffer(max_positions=max_calibration_positions)

        with _precision_ctx(self.device, self.dtype):
            for rec in records:
                seq = rec.sequence[::-1] if reverse else rec.sequence
                max_aa = max(2, self.max_positions - len(bos) - len(eos))
                if len(seq) > max_aa:
                    seq = seq[:max_aa]
                body_ids = [self.aa_to_id[aa] for aa in seq]
                target_20 = np.asarray([AA20_INDEX[aa] for aa in seq], dtype=np.int64)
                ids = list(bos) + body_ids + list(eos)
                start = len(bos)
                if start < 1:
                    # No BOS prefix: score residues 1..L-1 conditioned on residue 0
                    start = 1
                    target_20 = target_20[1:]
                end = len(bos) + len(body_ids)
                if end <= start:
                    continue

                inp = torch.as_tensor([ids], dtype=torch.long, device=self.device)
                logits = self._forward_logits(inp)[0, start - 1 : end - 1, :]
                z20 = aa20_restricted_logits(logits, self.aa20_ids_t)
                y_t = torch.as_tensor(target_20, dtype=torch.long, device=self.device)
                nll_bits = F.cross_entropy(z20, y_t, reduction="none").to(torch.float64) / LN2
                seq_bits.append(float(nll_bits.mean().item()))
                seq_lens.append(int(len(target_20)))
                cal_buf.add(z20.detach().cpu().numpy(), target_20)
        return seq_bits, seq_lens, cal_buf

    def evaluate_sequence_split(
        self,
        records: Sequence[AuditSequenceRecord],
        *,
        batch_size: int = 16,
        n_mlm15_draws: int = DEFAULT_MLM15_DRAWS,
        n_loo_probes: int = DEFAULT_LOO_PROBES,
        n_mdm_strata: int = DEFAULT_MDM_STRATA,
        eval_seed: int = 20261129,
        max_calibration_positions: int = 65_536,
    ) -> dict[str, dict[str, Any]]:
        del batch_size, n_mlm15_draws, n_loo_probes, n_mdm_strata, eval_seed
        fwd_bits, fwd_lens, fwd_cal = self._score_one_direction(
            records, reverse=False, max_calibration_positions=max_calibration_positions
        )
        zf, yf, wf = fwd_cal.finalize()
        out: dict[str, dict[str, Any]] = {
            "ar_exact": _summarize_seq_bits(fwd_bits, fwd_lens, logits_20=zf, targets_20=yf, weights=wf)
        }
        if self.score_bidirectional:
            rev_bits, rev_lens, rev_cal = self._score_one_direction(
                records, reverse=True, max_calibration_positions=max_calibration_positions
            )
            zr, yr, wr = rev_cal.finalize()
            out["ar_c_to_n"] = _summarize_seq_bits(
                rev_bits, rev_lens, logits_20=zr, targets_20=yr, weights=wr
            )
            bidir_bits = [0.5 * (a + b) for a, b in zip(fwd_bits, rev_bits)]
            out["ar_bidirectional_avg"] = _summarize_seq_bits(bidir_bits, fwd_lens)
        return out


# --------------------------------------------------------------------------- #
# 2 & 3. Masked LM & MDM / Discrete Diffusion Evaluator
# --------------------------------------------------------------------------- #


class MaskedLMOpenEvaluator(OpenModelAuditEvaluator):
    """Unified Masked LM & MDM evaluator computing MLM-15%, 16-site LOO, and 16-stratum MDM NELBO."""

    def __init__(
        self,
        *,
        model_id: str,
        model: nn.Module,
        aa20_ids: Sequence[int],
        mask_id: int,
        pad_id: int,
        cls_id: int | None,
        eos_id: int | None,
        family_kind: str = "mlm",
        primary_readout: str = "mlm_15",
        max_residues: int = 1022,
        device: str = "cpu",
        dtype: str = "float32",
        additive_attention_mask: bool = False,
        forward_fn: Any | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        pcounts = count_model_parameters(model)
        super().__init__(
            model_id=model_id,
            family_kind=family_kind,
            primary_readout=primary_readout,
            n_params_total=pcounts["n_params_total"],
            n_params_emb=pcounts["n_params_emb"],
            n_params_non_emb=pcounts["n_params_non_emb"],
            device=device,
            dtype=dtype,
            supports_sequence_splits=True,
            supports_3d_panel=False,
            supports_3d_ablation=False,
            extra_metadata=extra_metadata,
        )
        self.model = model
        self.aa20_ids = [int(x) for x in aa20_ids]
        self.aa20_ids_t = torch.as_tensor(self.aa20_ids, dtype=torch.long, device=self.device)
        self.aa_to_id = {aa: tid for aa, tid in zip(AA20, self.aa20_ids)}
        self.mask_id = int(mask_id)
        self.pad_id = int(pad_id)
        self.cls_id = int(cls_id) if cls_id is not None else None
        self.eos_id = int(eos_id) if eos_id is not None else None
        self.max_residues = int(max_residues)
        self.additive_attention_mask = bool(additive_attention_mask)
        self._forward_fn = forward_fn
        _, self.torch_dtype = _resolve_torch_dtype(dtype)

    def _encode_aa_tokens(self, seq: str) -> tuple[list[int], int]:
        s = seq[: self.max_residues]
        body = [self.aa_to_id[aa] for aa in s]
        ids: list[int] = []
        offset = 0
        if self.cls_id is not None:
            ids.append(self.cls_id)
            offset = 1
        ids.extend(body)
        if self.eos_id is not None:
            ids.append(self.eos_id)
        return ids, offset

    def _forward_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self._forward_fn is not None:
            return self._forward_fn(input_ids, attention_mask).float()
        kwargs: dict[str, Any] = {"input_ids": input_ids}
        if self.additive_attention_mask:
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

    def _collate_masked_batch(
        self,
        items: Sequence[tuple[list[int], int, np.ndarray]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_t = max(len(ids) for ids, _, _ in items)
        bsz = len(items)
        inp = torch.full((bsz, max_t), self.pad_id, dtype=torch.long, device=self.device)
        attn = torch.zeros((bsz, max_t), dtype=torch.long, device=self.device)
        for row_i, (ids, offset, mask_bool) in enumerate(items):
            t_ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
            masked_positions = np.flatnonzero(mask_bool)
            if len(masked_positions) > 0:
                t_ids[offset + torch.as_tensor(masked_positions, dtype=torch.long, device=self.device)] = self.mask_id
            inp[row_i, : len(ids)] = t_ids
            attn[row_i, : len(ids)] = 1
        return inp, attn

    @torch.no_grad()
    def _eval_mlm15(
        self,
        records: Sequence[AuditSequenceRecord],
        *,
        batch_size: int,
        n_draws: int,
        eval_seed: int,
        max_calibration_positions: int,
    ) -> dict[str, Any]:
        # Sort by length for compact padding while preserving original seq_idx for SplitMix64 invariance
        indexed = sorted(enumerate(records), key=lambda kv: len(kv[1].sequence))
        encoded_map = {orig_i: self._encode_aa_tokens(rec.sequence) for orig_i, rec in indexed}
        seq_draw_bits: dict[int, list[float]] = {orig_i: [] for orig_i, _ in indexed}
        seq_lens = {orig_i: min(len(rec.sequence), self.max_residues) for orig_i, rec in indexed}
        cal_buf = _CalibrationBuffer(max_positions=max_calibration_positions, seed=eval_seed + 11)

        with _precision_ctx(self.device, self.dtype):
            for draw in range(int(n_draws)):
                for start in range(0, len(indexed), max(1, int(batch_size))):
                    chunk = indexed[start : start + max(1, int(batch_size))]
                    batch_items = []
                    masks = []
                    for orig_i, rec in chunk:
                        ids, offset = encoded_map[orig_i]
                        L = seq_lens[orig_i]
                        m = deterministic_fixed_rate_mask(
                            orig_i, L, draw_idx=draw, mask_rate=0.15, eval_seed=eval_seed + 100
                        )
                        batch_items.append((ids, offset, m))
                        masks.append(m)
                    inp, attn = self._collate_masked_batch(batch_items)
                    logits = self._forward_batch(inp, attn)
                    for row_i, ((orig_i, rec), (_, offset, _), m) in enumerate(zip(chunk, batch_items, masks)):
                        pos = np.flatnonzero(m)
                        tok_idx = torch.as_tensor(offset + pos, dtype=torch.long, device=self.device)
                        z20 = aa20_restricted_logits(logits[row_i].index_select(0, tok_idx), self.aa20_ids_t)
                        y_np = np.asarray([AA20_INDEX[rec.sequence[int(p)]] for p in pos], dtype=np.int64)
                        y_t = torch.as_tensor(y_np, dtype=torch.long, device=self.device)
                        ce_bits = F.cross_entropy(z20, y_t, reduction="none").to(torch.float64) / LN2
                        seq_draw_bits[orig_i].append(float(ce_bits.mean().item()))
                        L_seq = float(seq_lens[orig_i])
                        w_site = np.full(len(pos), L_seq / float(int(n_draws) * len(pos)), dtype=np.float64)
                        cal_buf.add(z20.detach().cpu().numpy(), y_np, weights=w_site)

        final_bits = [float(np.mean(seq_draw_bits[i])) for i in range(len(records))]
        final_lens = [seq_lens[i] for i in range(len(records))]
        z, y, w = cal_buf.finalize()
        summary = _summarize_seq_bits(final_bits, final_lens, logits_20=z, targets_20=y, weights=w)
        summary["n_draws"] = int(n_draws)
        return summary

    @torch.no_grad()
    def _eval_mdm16(
        self,
        records: Sequence[AuditSequenceRecord],
        *,
        batch_size: int,
        n_strata: int,
        eval_seed: int,
        max_calibration_positions: int,
    ) -> dict[str, Any]:
        indexed = sorted(enumerate(records), key=lambda kv: len(kv[1].sequence))
        encoded_map = {orig_i: self._encode_aa_tokens(rec.sequence) for orig_i, rec in indexed}
        seq_lens = {orig_i: min(len(rec.sequence), self.max_residues) for orig_i, rec in indexed}
        seq_stratum_bits: dict[int, list[float]] = {orig_i: [] for orig_i, _ in indexed}
        stratum_weighted_sum = np.zeros(int(n_strata), dtype=np.float64)
        stratum_res_sum = np.zeros(int(n_strata), dtype=np.float64)
        stratum_mask_frac_sum = np.zeros(int(n_strata), dtype=np.float64)
        cal_buf = _CalibrationBuffer(max_positions=max_calibration_positions, seed=eval_seed + 29)

        with _precision_ctx(self.device, self.dtype):
            for s in range(int(n_strata)):
                for start in range(0, len(indexed), max(1, int(batch_size))):
                    chunk = indexed[start : start + max(1, int(batch_size))]
                    batch_items = []
                    masks = []
                    m_counts = []
                    for orig_i, rec in chunk:
                        ids, offset = encoded_map[orig_i]
                        L = seq_lens[orig_i]
                        m, m_mask, _u = deterministic_mdm_stratum_mask(
                            orig_i, L, stratum=s, n_strata=n_strata, eval_seed=eval_seed
                        )
                        batch_items.append((ids, offset, m))
                        masks.append(m)
                        m_counts.append(m_mask)
                    inp, attn = self._collate_masked_batch(batch_items)
                    logits = self._forward_batch(inp, attn)
                    for row_i, ((orig_i, rec), (_, offset, _), m, m_mask) in enumerate(
                        zip(chunk, batch_items, masks, m_counts)
                    ):
                        L = seq_lens[orig_i]
                        pos = np.flatnonzero(m)
                        tok_idx = torch.as_tensor(offset + pos, dtype=torch.long, device=self.device)
                        z20 = aa20_restricted_logits(logits[row_i].index_select(0, tok_idx), self.aa20_ids_t)
                        y_np = np.asarray([AA20_INDEX[rec.sequence[int(p)]] for p in pos], dtype=np.int64)
                        y_t = torch.as_tensor(y_np, dtype=torch.long, device=self.device)
                        ce_bits = F.cross_entropy(z20, y_t, reduction="none").to(torch.float64) / LN2
                        h_s = float(ce_bits.mean().item())
                        seq_stratum_bits[orig_i].append(h_s)
                        stratum_weighted_sum[s] += h_s * float(L)
                        stratum_res_sum[s] += float(L)
                        stratum_mask_frac_sum[s] += (float(m_mask) / float(L)) * float(L)
                        # Exact OA-ARDM NELBO weight per masked site: L / (n_strata * m_mask)
                        w_site = np.full(len(pos), float(L) / float(n_strata * m_mask), dtype=np.float64)
                        cal_buf.add(z20.detach().cpu().numpy(), y_np, weights=w_site)

        final_bits = [float(np.mean(seq_stratum_bits[i])) for i in range(len(records))]
        final_lens = [seq_lens[i] for i in range(len(records))]
        z, y, w = cal_buf.finalize()
        summary = _summarize_seq_bits(final_bits, final_lens, logits_20=z, targets_20=y, weights=w)
        stratum_curve = [
            {
                "stratum": int(s),
                "mean_mask_frac": float(stratum_mask_frac_sum[s] / max(stratum_res_sum[s], 1.0)),
                "bits_per_res": float(stratum_weighted_sum[s] / max(stratum_res_sum[s], 1.0)),
            }
            for s in range(int(n_strata))
        ]
        summary["n_strata"] = int(n_strata)
        summary["stratum_curve"] = stratum_curve
        if int(n_strata) >= 4:
            q = max(1, int(n_strata) // 4)
            low_m = float(np.mean([c["bits_per_res"] for c in stratum_curve[:q]]))
            high_m = float(np.mean([c["bits_per_res"] for c in stratum_curve[-q:]]))
            summary["low_mask_quartile_bits"] = low_m
            summary["high_mask_quartile_bits"] = high_m
            summary["high_mask_ood_jump_bits"] = float(high_m - low_m)
        return summary

    @torch.no_grad()
    def _eval_loo(
        self,
        records: Sequence[AuditSequenceRecord],
        *,
        batch_size: int,
        n_probes: int,
        eval_seed: int,
        max_calibration_positions: int,
    ) -> dict[str, Any]:
        seq_bits: list[float] = []
        seq_lens: list[int] = []
        cal_buf = _CalibrationBuffer(max_positions=max_calibration_positions, seed=eval_seed + 53)
        bs = max(1, int(batch_size))

        with _precision_ctx(self.device, self.dtype):
            for orig_i, rec in enumerate(records):
                ids, offset = self._encode_aa_tokens(rec.sequence)
                L = min(len(rec.sequence), self.max_residues)
                probes = deterministic_loo_positions(orig_i, L, n_probes=n_probes, eval_seed=eval_seed + 200)
                row = torch.as_tensor(ids, dtype=torch.long, device=self.device)
                site_bits: list[float] = []
                for start in range(0, len(probes), bs):
                    chunk = probes[start : start + bs]
                    b_inp = row.unsqueeze(0).repeat(len(chunk), 1)
                    b_attn = torch.ones_like(b_inp)
                    for r_i, p in enumerate(chunk):
                        b_inp[r_i, offset + int(p)] = self.mask_id
                    logits = self._forward_batch(b_inp, b_attn)
                    idx_rows = torch.arange(len(chunk), device=self.device)
                    idx_cols = torch.as_tensor(offset + chunk, dtype=torch.long, device=self.device)
                    z20 = aa20_restricted_logits(logits[idx_rows, idx_cols, :], self.aa20_ids_t)
                    y_np = np.asarray([AA20_INDEX[rec.sequence[int(p)]] for p in chunk], dtype=np.int64)
                    y_t = torch.as_tensor(y_np, dtype=torch.long, device=self.device)
                    ce_bits = F.cross_entropy(z20, y_t, reduction="none").to(torch.float64) / LN2
                    site_bits.extend(ce_bits.detach().cpu().tolist())
                    w_chunk = np.full(len(chunk), float(L) / float(len(probes)), dtype=np.float64)
                    cal_buf.add(z20.detach().cpu().numpy(), y_np, weights=w_chunk)
                seq_bits.append(float(np.mean(site_bits)))
                seq_lens.append(int(L))

        z, y, w = cal_buf.finalize()
        summary = _summarize_seq_bits(seq_bits, seq_lens, logits_20=z, targets_20=y, weights=w)
        summary["n_probes_per_seq"] = int(n_probes)
        return summary

    def evaluate_sequence_split(
        self,
        records: Sequence[AuditSequenceRecord],
        *,
        batch_size: int = 16,
        n_mlm15_draws: int = DEFAULT_MLM15_DRAWS,
        n_loo_probes: int = DEFAULT_LOO_PROBES,
        n_mdm_strata: int = DEFAULT_MDM_STRATA,
        eval_seed: int = 20261129,
        max_calibration_positions: int = 65_536,
    ) -> dict[str, dict[str, Any]]:
        return {
            "mlm_15": self._eval_mlm15(
                records,
                batch_size=batch_size,
                n_draws=n_mlm15_draws,
                eval_seed=eval_seed,
                max_calibration_positions=max_calibration_positions,
            ),
            "loo": self._eval_loo(
                records,
                batch_size=batch_size,
                n_probes=n_loo_probes,
                eval_seed=eval_seed,
                max_calibration_positions=max_calibration_positions,
            ),
            "mdm_16": self._eval_mdm16(
                records,
                batch_size=batch_size,
                n_strata=n_mdm_strata,
                eval_seed=eval_seed,
                max_calibration_positions=max_calibration_positions,
            ),
        }


class MDMOpenEvaluator(MaskedLMOpenEvaluator):
    """Discrete Diffusion / Order-Agnostic PLM evaluator (DPLM, EvoDiff-OA, MDM)."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("family_kind", "mdm")
        kwargs.setdefault("primary_readout", "mdm_16")
        super().__init__(**kwargs)


# --------------------------------------------------------------------------- #
# 4A. ProteinMPNN Structure-Conditioned Evaluator
# --------------------------------------------------------------------------- #


class ProteinMPNNOpenEvaluator(OpenModelAuditEvaluator):
    """Evaluates ProteinMPNN strictly over canonical AA20 (stripping index 20 ``X``)."""

    def __init__(
        self,
        adapter: Any,
        *,
        n_permutations: int = 4,
    ) -> None:
        pcounts = count_model_parameters(adapter.model)
        super().__init__(
            model_id=str(adapter.arm),
            family_kind="structure_conditioned",
            primary_readout="ar_3d_perm_avg",
            n_params_total=pcounts["n_params_total"],
            n_params_emb=pcounts["n_params_emb"],
            n_params_non_emb=pcounts["n_params_non_emb"],
            device=str(adapter.device),
            dtype=str(adapter.dtype),
            supports_sequence_splits=False,
            supports_3d_panel=True,
            supports_3d_ablation=False,
            extra_metadata={
                "variant": adapter.spec.variant,
                "noise_angstrom": adapter.ckpt_noise_level,
                "ca_only": adapter.spec.ca_only,
                "n_permutations": int(n_permutations),
            },
        )
        self.adapter = adapter
        self.n_permutations = int(n_permutations)

    @torch.no_grad()
    def _eval_3d_subset(
        self,
        records: Sequence[Paired3DRecord],
        coords_by_key: Mapping[str, np.ndarray],
        *,
        n_loo_probes: int,
        eval_seed: int,
        seq_reference_bits: float | None,
    ) -> dict[str, Any]:
        from src.capability.unified.proteinmpnn_adapter import _canonical_permutations, _encode_mpnn_seq

        ar_seq_bits: list[float] = []
        loo_seq_bits: list[float] = []
        seq_lens: list[int] = []
        ar_cal = _CalibrationBuffer(seed=eval_seed + 1)
        loo_cal = _CalibrationBuffer(seed=eval_seed + 2)
        aa20_t = torch.arange(20, dtype=torch.long, device=self.device)

        with _precision_ctx(self.device, self.dtype):
            for idx, rec in enumerate(records):
                seq = rec.sequence
                L = len(seq)
                coords = coords_by_key[rec.coords_key]
                h_V_enc, h_E, E_idx, h_EXV, mask_t = self.adapter._encode_single_backbone(coords)
                S_base = torch.from_numpy(_encode_mpnn_seq(seq)).to(device=self.device, dtype=torch.long).unsqueeze(0)
                y_np = np.asarray([AA20_INDEX[aa] for aa in seq], dtype=np.int64)
                y_t = torch.as_tensor(y_np, dtype=torch.long, device=self.device)

                # 1. Permutation-averaged exact AR joint entropy strictly over AA20 (logits[..., :20])
                orders_np = _canonical_permutations(L, n_permutations=self.n_permutations, seed=eval_seed + idx)
                n_p = orders_np.shape[0]
                orders_t = torch.from_numpy(orders_np).to(device=self.device, dtype=torch.long)
                logits_ar, _ = self.adapter.model.decode_from_encoded(
                    h_V_enc.expand(n_p, -1, -1),
                    h_E.expand(n_p, -1, -1, -1),
                    E_idx.expand(n_p, -1, -1),
                    h_EXV.expand(n_p, -1, -1, -1),
                    S_base.expand(n_p, -1),
                    mask_t.expand(n_p, -1),
                    orders_t,
                )
                z20_ar = aa20_restricted_logits(logits_ar.float(), aa20_t)  # (n_p, L, 20)
                ce_ar_bits = (
                    F.cross_entropy(
                        z20_ar.reshape(-1, 20),
                        y_t.unsqueeze(0).expand(n_p, -1).reshape(-1),
                        reduction="none",
                    ).to(torch.float64)
                    / LN2
                )
                ar_seq_bits.append(float(ce_ar_bits.mean().item()))
                ar_cal.add(
                    z20_ar.reshape(-1, 20).detach().cpu().numpy(),
                    np.tile(y_np, n_p),
                )

                # 2. Single-site LOO (place each probed position LAST in decoding order)
                probes = deterministic_loo_positions(idx, L, n_probes=n_loo_probes, eval_seed=eval_seed + 300)
                base_order = np.arange(L, dtype=np.int64)
                loo_orders = np.empty((len(probes), L), dtype=np.int64)
                for r_i, p in enumerate(probes):
                    loo_orders[r_i, :-1] = base_order[base_order != int(p)]
                    loo_orders[r_i, -1] = int(p)
                loo_orders_t = torch.from_numpy(loo_orders).to(device=self.device, dtype=torch.long)
                logits_loo, _ = self.adapter.model.decode_from_encoded(
                    h_V_enc.expand(len(probes), -1, -1),
                    h_E.expand(len(probes), -1, -1, -1),
                    E_idx.expand(len(probes), -1, -1),
                    h_EXV.expand(len(probes), -1, -1, -1),
                    S_base.expand(len(probes), -1),
                    mask_t.expand(len(probes), -1),
                    loo_orders_t,
                )
                row_idx = torch.arange(len(probes), device=self.device)
                col_idx = torch.as_tensor(probes, dtype=torch.long, device=self.device)
                z20_loo = aa20_restricted_logits(logits_loo[row_idx, col_idx, :].float(), aa20_t)
                y_loo_np = np.asarray([AA20_INDEX[seq[int(p)]] for p in probes], dtype=np.int64)
                y_loo_t = torch.as_tensor(y_loo_np, dtype=torch.long, device=self.device)
                ce_loo_bits = F.cross_entropy(z20_loo, y_loo_t, reduction="none").to(torch.float64) / LN2
                loo_seq_bits.append(float(ce_loo_bits.mean().item()))
                w_loo = np.full(len(probes), float(L) / float(len(probes)), dtype=np.float64)
                loo_cal.add(z20_loo.detach().cpu().numpy(), y_loo_np, weights=w_loo)
                seq_lens.append(L)

        za, ya, wa = ar_cal.finalize()
        zl, yl, wl = loo_cal.finalize()
        ar_summary = _summarize_seq_bits(ar_seq_bits, seq_lens, logits_20=za, targets_20=ya, weights=wa)
        loo_summary = _summarize_seq_bits(loo_seq_bits, seq_lens, logits_20=zl, targets_20=yl, weights=wl)
        return {
            "ar_3d_perm_avg": ar_summary,
            "loo_3d": loo_summary,
            "b_ext_ar": compute_multitrack_b_ext(
                ar_summary["bits_per_res"],
                h_unconditioned=None,
                h_seq_reference=seq_reference_bits,
            ),
            "b_ext_loo": compute_multitrack_b_ext(
                loo_summary["bits_per_res"],
                h_unconditioned=None,
                h_seq_reference=seq_reference_bits,
            ),
        }

    def evaluate_3d_panel(
        self,
        records: Sequence[Paired3DRecord],
        coords_by_key: Mapping[str, np.ndarray],
        *,
        batch_size: int = 16,
        n_mlm15_draws: int = DEFAULT_MLM15_DRAWS,
        n_loo_probes: int = DEFAULT_LOO_PROBES,
        n_mdm_strata: int = DEFAULT_MDM_STRATA,
        eval_seed: int = 20261129,
        seq_reference_bits: float | None = None,
    ) -> dict[str, Any]:
        del batch_size, n_mlm15_draws, n_mdm_strata
        by_group: dict[str, list[Paired3DRecord]] = defaultdict(list)
        for r in records:
            by_group[r.domain_group].append(r)

        out: dict[str, Any] = {
            "overall": self._eval_3d_subset(
                records,
                coords_by_key,
                n_loo_probes=n_loo_probes,
                eval_seed=eval_seed,
                seq_reference_bits=seq_reference_bits,
            ),
            "by_domain_group": {},
        }
        for grp in sorted(by_group.keys()):
            out["by_domain_group"][grp] = self._eval_3d_subset(
                by_group[grp],
                coords_by_key,
                n_loo_probes=n_loo_probes,
                eval_seed=eval_seed,
                seq_reference_bits=seq_reference_bits,
            )
        return out


# --------------------------------------------------------------------------- #
# 4B. SaProt Multi-Track (AA + Foldseek 3Di) Evaluator
# --------------------------------------------------------------------------- #


class SaProtOpenEvaluator(OpenModelAuditEvaluator):
    """Evaluates SaProt with BOTH 3Di structure context $Z_{\\mathrm{3D}}$ and `#`-ablated $Z=\\varnothing$."""

    def __init__(self, adapter: Any) -> None:
        pcounts = count_model_parameters(adapter.model)
        super().__init__(
            model_id=str(adapter.arm),
            family_kind="structure_conditioned",
            primary_readout="mlm_15",
            n_params_total=pcounts["n_params_total"],
            n_params_emb=pcounts["n_params_emb"],
            n_params_non_emb=pcounts["n_params_non_emb"],
            device=str(adapter.device),
            dtype=str(adapter.dtype),
            supports_sequence_splits=True,
            supports_3d_panel=True,
            supports_3d_ablation=True,
            extra_metadata={"vocab_kind": "saprot_441_aa_3di"},
        )
        self.adapter = adapter
        self.pad_id = (
            int(adapter.tokenizer.pad_token_id)
            if getattr(adapter.tokenizer, "pad_token_id", None) is not None
            else 1
        )
        # (20, 21) matrix: for each AA, all 20 lowercase 3Di tokens + the AA# token
        mat_21 = [list(adapter.aa_struct_ids[aa]) + [int(adapter.aa_hash_id[aa])] for aa in AA20]
        self._aa20_21_ids_t = torch.as_tensor(mat_21, dtype=torch.long, device=self.device)
        # Lookup {3di_char: tensor(20,)} for exact AA+s conditioning
        self._ids_given_3di: dict[str, torch.Tensor] = {}
        for s_char in adapter.struct_chars:
            self._ids_given_3di[s_char] = torch.as_tensor(
                [int(adapter.vocab[f"{aa}{s_char}"]) for aa in AA20],
                dtype=torch.long,
                device=self.device,
            )

    def _encode(self, sequence: str, three_di: str | None) -> tuple[list[int], int]:
        if three_di is None:
            combined = "".join(f"{aa}#" for aa in sequence)
        else:
            combined = "".join(f"{aa}{s.lower()}" for aa, s in zip(sequence, three_di))
        ids = [int(v) for v in self.adapter.tokenizer(combined)["input_ids"]]
        if len(ids) != len(sequence) + 2:
            raise ValueError(f"SaProt token length mismatch: got {len(ids)}, expected {len(sequence) + 2}")
        return ids, 1

    def _aa20_logits_from_full(self, row_logits: torch.Tensor, known_3di: str | None) -> torch.Tensor:
        """Convert full 446-vocab logits ``(V,)`` to strict 20-AA logits ``(20,)``."""
        if known_3di is not None and known_3di in self._ids_given_3di:
            return row_logits.index_select(0, self._ids_given_3di[known_3di]).float()
        # Marginalize over all 21 tokens per AA in log-space (strict AA20 simplex, matching causal control)
        full_lp = F.log_softmax(row_logits.float(), dim=-1)
        return torch.logsumexp(full_lp[self._aa20_21_ids_t], dim=-1)

    @torch.no_grad()
    def _eval_readouts_generic(
        self,
        seqs: Sequence[str],
        three_di_list: Sequence[str | None],
        *,
        use_target_3di_mask: bool,
        batch_size: int,
        n_mlm15_draws: int,
        n_loo_probes: int,
        n_mdm_strata: int,
        eval_seed: int,
        max_calibration_positions: int = 65_536,
    ) -> dict[str, dict[str, Any]]:
        encoded = [self._encode(s, s3) for s, s3 in zip(seqs, three_di_list)]
        bs = max(1, int(batch_size))

        def _run_mask_mode(mode: str, n_steps: int) -> dict[str, Any]:
            seq_step_bits: list[list[float]] = [[] for _ in range(len(seqs))]
            seq_lens = [len(s) for s in seqs]
            cal_buf = _CalibrationBuffer(max_positions=max_calibration_positions, seed=eval_seed + 7)
            stratum_weighted_sum = np.zeros(n_steps, dtype=np.float64)
            stratum_res_sum = np.zeros(n_steps, dtype=np.float64)
            stratum_mask_frac_sum = np.zeros(n_steps, dtype=np.float64)

            with _precision_ctx(self.device, self.dtype):
                for step in range(n_steps):
                    for start in range(0, len(seqs), bs):
                        stop = min(start + bs, len(seqs))
                        chunk_ids = [encoded[i][0] for i in range(start, stop)]
                        max_t = max(len(x) for x in chunk_ids)
                        inp = torch.full((stop - start, max_t), self.pad_id, dtype=torch.long, device=self.device)
                        attn = torch.zeros_like(inp)
                        chunk_pos: list[np.ndarray] = []
                        chunk_known: list[list[str | None]] = []

                        for row_i, seq_i in enumerate(range(start, stop)):
                            ids, offset = encoded[seq_i]
                            L = len(seqs[seq_i])
                            s3 = three_di_list[seq_i]
                            if mode == "mlm_15":
                                m = deterministic_fixed_rate_mask(
                                    seq_i, L, draw_idx=step, mask_rate=0.15, eval_seed=eval_seed + 100
                                )
                                pos = np.flatnonzero(m)
                            else:
                                m, _m_cnt, _u = deterministic_mdm_stratum_mask(
                                    seq_i, L, stratum=step, n_strata=n_steps, eval_seed=eval_seed
                                )
                                pos = np.flatnonzero(m)
                            t_ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
                            known_list: list[str | None] = []
                            for p in pos:
                                if use_target_3di_mask and s3 is not None:
                                    sc = s3[int(p)].lower()
                                    if sc in self.adapter.hash_struct_id:
                                        t_ids[offset + int(p)] = self.adapter.hash_struct_id[sc]
                                        known_list.append(sc)
                                    else:
                                        t_ids[offset + int(p)] = self.adapter.mask_id
                                        known_list.append(None)
                                else:
                                    t_ids[offset + int(p)] = self.adapter.mask_id
                                    known_list.append(None)
                            inp[row_i, : len(ids)] = t_ids
                            attn[row_i, : len(ids)] = 1
                            chunk_pos.append(pos)
                            chunk_known.append(known_list)

                        logits = self.adapter._forward_logits(inp, attention_mask=attn)
                        for row_i, seq_i in enumerate(range(start, stop)):
                            _, offset = encoded[seq_i]
                            pos = chunk_pos[row_i]
                            known_list = chunk_known[row_i]
                            L = len(seqs[seq_i])
                            z_rows = torch.stack(
                                [
                                    self._aa20_logits_from_full(logits[row_i, offset + int(p)], kn)
                                    for p, kn in zip(pos, known_list)
                                ],
                                dim=0,
                            )
                            y_np = np.asarray([AA20_INDEX[seqs[seq_i][int(p)]] for p in pos], dtype=np.int64)
                            y_t = torch.as_tensor(y_np, dtype=torch.long, device=self.device)
                            ce_bits = F.cross_entropy(z_rows, y_t, reduction="none").to(torch.float64) / LN2
                            h_val = float(ce_bits.mean().item())
                            seq_step_bits[seq_i].append(h_val)
                            if mode == "mdm_16":
                                stratum_weighted_sum[step] += h_val * float(L)
                                stratum_res_sum[step] += float(L)
                                stratum_mask_frac_sum[step] += (float(len(pos)) / float(L)) * float(L)
                            w_site = np.full(len(pos), float(L) / float(n_steps * len(pos)), dtype=np.float64)
                            cal_buf.add(z_rows.detach().cpu().numpy(), y_np, weights=w_site)

            final_bits = [float(np.mean(v)) for v in seq_step_bits]
            z, y, w = cal_buf.finalize()
            res = _summarize_seq_bits(final_bits, seq_lens, logits_20=z, targets_20=y, weights=w)
            if mode == "mdm_16":
                res["stratum_curve"] = [
                    {
                        "stratum": int(s),
                        "mean_mask_frac": float(stratum_mask_frac_sum[s] / max(stratum_res_sum[s], 1.0)),
                        "bits_per_res": float(stratum_weighted_sum[s] / max(stratum_res_sum[s], 1.0)),
                    }
                    for s in range(n_steps)
                ]
            return res

        # Single-site LOO
        loo_bits: list[float] = []
        loo_lens = [len(s) for s in seqs]
        loo_cal = _CalibrationBuffer(max_positions=max_calibration_positions, seed=eval_seed + 19)
        with _precision_ctx(self.device, self.dtype):
            for seq_i, (seq, s3) in enumerate(zip(seqs, three_di_list)):
                ids, offset = encoded[seq_i]
                L = len(seq)
                probes = deterministic_loo_positions(seq_i, L, n_probes=n_loo_probes, eval_seed=eval_seed + 200)
                row = torch.as_tensor(ids, dtype=torch.long, device=self.device)
                site_bits: list[float] = []
                for start in range(0, len(probes), bs):
                    chunk = probes[start : start + bs]
                    b_inp = row.unsqueeze(0).repeat(len(chunk), 1)
                    b_attn = torch.ones_like(b_inp)
                    known_list = []
                    for r_i, p in enumerate(chunk):
                        if use_target_3di_mask and s3 is not None and s3[int(p)].lower() in self.adapter.hash_struct_id:
                            sc = s3[int(p)].lower()
                            b_inp[r_i, offset + int(p)] = self.adapter.hash_struct_id[sc]
                            known_list.append(sc)
                        else:
                            b_inp[r_i, offset + int(p)] = self.adapter.mask_id
                            known_list.append(None)
                    logits = self.adapter._forward_logits(b_inp, attention_mask=b_attn)
                    z_rows = torch.stack(
                        [
                            self._aa20_logits_from_full(logits[r_i, offset + int(p)], kn)
                            for r_i, (p, kn) in enumerate(zip(chunk, known_list))
                        ],
                        dim=0,
                    )
                    y_np = np.asarray([AA20_INDEX[seq[int(p)]] for p in chunk], dtype=np.int64)
                    y_t = torch.as_tensor(y_np, dtype=torch.long, device=self.device)
                    ce_bits = F.cross_entropy(z_rows, y_t, reduction="none").to(torch.float64) / LN2
                    site_bits.extend(ce_bits.detach().cpu().tolist())
                    w_chunk = np.full(len(chunk), float(L) / float(len(probes)), dtype=np.float64)
                    loo_cal.add(z_rows.detach().cpu().numpy(), y_np, weights=w_chunk)
                loo_bits.append(float(np.mean(site_bits)))

        zl, yl, wl = loo_cal.finalize()
        return {
            "mlm_15": _run_mask_mode("mlm_15", int(n_mlm15_draws)),
            "loo": _summarize_seq_bits(loo_bits, loo_lens, logits_20=zl, targets_20=yl, weights=wl),
            "mdm_16": _run_mask_mode("mdm_16", int(n_mdm_strata)),
        }

    def evaluate_sequence_split(
        self,
        records: Sequence[AuditSequenceRecord],
        *,
        batch_size: int = 16,
        n_mlm15_draws: int = DEFAULT_MLM15_DRAWS,
        n_loo_probes: int = DEFAULT_LOO_PROBES,
        n_mdm_strata: int = DEFAULT_MDM_STRATA,
        eval_seed: int = 20261129,
        max_calibration_positions: int = 65_536,
    ) -> dict[str, dict[str, Any]]:
        seqs = [r.sequence for r in records]
        return self._eval_readouts_generic(
            seqs,
            [None] * len(seqs),
            use_target_3di_mask=False,
            batch_size=batch_size,
            n_mlm15_draws=n_mlm15_draws,
            n_loo_probes=n_loo_probes,
            n_mdm_strata=n_mdm_strata,
            eval_seed=eval_seed,
            max_calibration_positions=max_calibration_positions,
        )

    def _eval_3d_subset(
        self,
        records: Sequence[Paired3DRecord],
        *,
        batch_size: int,
        n_mlm15_draws: int,
        n_loo_probes: int,
        n_mdm_strata: int,
        eval_seed: int,
    ) -> dict[str, Any]:
        seqs = [r.sequence for r in records]
        tdis = [r.three_di_seq for r in records]
        uncond = self._eval_readouts_generic(
            seqs,
            [None] * len(seqs),
            use_target_3di_mask=False,
            batch_size=batch_size,
            n_mlm15_draws=n_mlm15_draws,
            n_loo_probes=n_loo_probes,
            n_mdm_strata=n_mdm_strata,
            eval_seed=eval_seed,
        )
        cond_full = self._eval_readouts_generic(
            seqs,
            tdis,
            use_target_3di_mask=True,
            batch_size=batch_size,
            n_mlm15_draws=n_mlm15_draws,
            n_loo_probes=n_loo_probes,
            n_mdm_strata=n_mdm_strata,
            eval_seed=eval_seed,
        )
        cond_ctx_only = self._eval_readouts_generic(
            seqs,
            tdis,
            use_target_3di_mask=False,
            batch_size=batch_size,
            n_mlm15_draws=n_mlm15_draws,
            n_loo_probes=n_loo_probes,
            n_mdm_strata=n_mdm_strata,
            eval_seed=eval_seed,
        )
        b_ext_full = {
            k: compute_multitrack_b_ext(
                cond_full[k]["bits_per_res"], h_unconditioned=uncond[k]["bits_per_res"]
            )
            for k in ("mlm_15", "loo", "mdm_16")
        }
        b_ext_ctx_only = {
            k: compute_multitrack_b_ext(
                cond_ctx_only[k]["bits_per_res"], h_unconditioned=uncond[k]["bits_per_res"]
            )
            for k in ("mlm_15", "loo", "mdm_16")
        }
        return {
            "unconditioned_null_Z": uncond,
            "conditioned_full_3di": cond_full,
            "conditioned_context_only_3di": cond_ctx_only,
            "b_ext_full_3di": b_ext_full,
            "b_ext_context_only_3di": b_ext_ctx_only,
        }

    def evaluate_3d_panel(
        self,
        records: Sequence[Paired3DRecord],
        coords_by_key: Mapping[str, np.ndarray],
        *,
        batch_size: int = 16,
        n_mlm15_draws: int = DEFAULT_MLM15_DRAWS,
        n_loo_probes: int = DEFAULT_LOO_PROBES,
        n_mdm_strata: int = DEFAULT_MDM_STRATA,
        eval_seed: int = 20261129,
        seq_reference_bits: float | None = None,
    ) -> dict[str, Any]:
        del coords_by_key, seq_reference_bits
        by_group: dict[str, list[Paired3DRecord]] = defaultdict(list)
        for r in records:
            by_group[r.domain_group].append(r)
        out: dict[str, Any] = {
            "overall": self._eval_3d_subset(
                records,
                batch_size=batch_size,
                n_mlm15_draws=n_mlm15_draws,
                n_loo_probes=n_loo_probes,
                n_mdm_strata=n_mdm_strata,
                eval_seed=eval_seed,
            ),
            "by_domain_group": {},
        }
        for grp in sorted(by_group.keys()):
            out["by_domain_group"][grp] = self._eval_3d_subset(
                by_group[grp],
                batch_size=batch_size,
                n_mlm15_draws=n_mlm15_draws,
                n_loo_probes=n_loo_probes,
                n_mdm_strata=n_mdm_strata,
                eval_seed=eval_seed,
            )
        return out


# --------------------------------------------------------------------------- #
# 4C. DPLM-2 Multi-Track (MDM Sequence + 3D Structure Tokenizer) Evaluator
# --------------------------------------------------------------------------- #


class DPLM2OpenEvaluator(OpenModelAuditEvaluator):
    """Evaluates DPLM-2 in BOTH ``seq_only`` ($Z=\\varnothing$) and ``struct_conditioned`` ($Z_{\\mathrm{3D}}$)."""

    def __init__(self, adapter_seq: Any, adapter_struct: Any | None = None) -> None:
        pcounts = count_model_parameters(adapter_seq.model)
        super().__init__(
            model_id=str(adapter_seq.arm),
            family_kind="mdm",
            primary_readout="mdm_16",
            n_params_total=pcounts["n_params_total"],
            n_params_emb=pcounts["n_params_emb"],
            n_params_non_emb=pcounts["n_params_non_emb"],
            device=str(adapter_seq.device),
            dtype=str(adapter_seq.dtype),
            supports_sequence_splits=True,
            supports_3d_panel=adapter_struct is not None,
            supports_3d_ablation=adapter_struct is not None,
            extra_metadata={"multitrack_backend": "dplm2_struct_tokenizer"},
        )
        self.adapter_seq = adapter_seq
        self.adapter_struct = adapter_struct

    @torch.no_grad()
    def _eval_dplm2_records(
        self,
        adapter: Any,
        seqs: Sequence[str],
        coords_list: Sequence[np.ndarray | None],
        *,
        n_mlm15_draws: int,
        n_loo_probes: int,
        n_mdm_strata: int,
        eval_seed: int,
        max_calibration_positions: int = 65_536,
    ) -> dict[str, dict[str, Any]]:
        from src.capability.unified.dplm2_adapter import (
            AA_MASK_ID,
            DPLM2_AA20_IDS,
            _encode_aa_to_ids,
        )

        aa20_t = torch.as_tensor(DPLM2_AA20_IDS, dtype=torch.long, device=self.device)
        prepared: list[tuple[torch.Tensor, int, bool]] = []
        for s, c in zip(seqs, coords_list):
            aa_ids = _encode_aa_to_ids(s)
            s_ids, single_mod = adapter._resolve_struct_ids(len(s), coords=c)
            base_paired, half = adapter._build_paired_ids([aa_ids], [s_ids])
            prepared.append((base_paired, half, bool(single_mod)))
        seq_lens = [len(s) for s in seqs]

        def _run_mask_mode(mode: str, n_steps: int) -> dict[str, Any]:
            seq_bits: list[list[float]] = [[] for _ in range(len(seqs))]
            cal = _CalibrationBuffer(max_positions=max_calibration_positions, seed=eval_seed + 3)
            stratum_weighted_sum = np.zeros(n_steps, dtype=np.float64)
            stratum_res_sum = np.zeros(n_steps, dtype=np.float64)
            stratum_mask_frac_sum = np.zeros(n_steps, dtype=np.float64)

            with _precision_ctx(self.device, self.dtype):
                for seq_i, (seq, (base_paired, half, single_mod)) in enumerate(zip(seqs, prepared)):
                    L = len(seq)
                    for step in range(n_steps):
                        if mode == "mlm_15":
                            m = deterministic_fixed_rate_mask(
                                seq_i, L, draw_idx=step, mask_rate=0.15, eval_seed=eval_seed + 100
                            )
                        else:
                            m, _mc, _u = deterministic_mdm_stratum_mask(
                                seq_i, L, stratum=step, n_strata=n_steps, eval_seed=eval_seed
                            )
                        pos = np.flatnonzero(m)
                        pos_t = torch.as_tensor(pos, dtype=torch.long, device=self.device)
                        inp = base_paired.clone()
                        inp[0, half + 1 + pos_t] = AA_MASK_ID
                        net_out = adapter._forward_paired(inp, single_modality=single_mod)
                        aa_logits = adapter._extract_aa_logits(net_out, half)[0].float()
                        z20 = aa20_restricted_logits(aa_logits.index_select(0, 1 + pos_t), aa20_t)
                        y_np = np.asarray([AA20_INDEX[seq[int(p)]] for p in pos], dtype=np.int64)
                        y_t = torch.as_tensor(y_np, dtype=torch.long, device=self.device)
                        ce_bits = F.cross_entropy(z20, y_t, reduction="none").to(torch.float64) / LN2
                        h_val = float(ce_bits.mean().item())
                        seq_bits[seq_i].append(h_val)
                        if mode == "mdm_16":
                            stratum_weighted_sum[step] += h_val * float(L)
                            stratum_res_sum[step] += float(L)
                            stratum_mask_frac_sum[step] += (float(len(pos)) / float(L)) * float(L)
                        w_site = np.full(len(pos), float(L) / float(n_steps * len(pos)), dtype=np.float64)
                        cal.add(z20.detach().cpu().numpy(), y_np, weights=w_site)
            z, y, w = cal.finalize()
            res = _summarize_seq_bits(
                [float(np.mean(v)) for v in seq_bits], seq_lens, logits_20=z, targets_20=y, weights=w
            )
            if mode == "mdm_16":
                res["stratum_curve"] = [
                    {
                        "stratum": int(s),
                        "mean_mask_frac": float(stratum_mask_frac_sum[s] / max(stratum_res_sum[s], 1.0)),
                        "bits_per_res": float(stratum_weighted_sum[s] / max(stratum_res_sum[s], 1.0)),
                    }
                    for s in range(n_steps)
                ]
            return res

        loo_bits: list[float] = []
        loo_cal = _CalibrationBuffer(max_positions=max_calibration_positions, seed=eval_seed + 9)
        with _precision_ctx(self.device, self.dtype):
            for seq_i, (seq, c) in enumerate(zip(seqs, coords_list)):
                L = len(seq)
                probes = deterministic_loo_positions(seq_i, L, n_probes=n_loo_probes, eval_seed=eval_seed + 200)
                lp = adapter.site_conditional_logprobs(seq, positions=probes.tolist(), coords=c)
                y_np = np.asarray([AA20_INDEX[seq[int(p)]] for p in probes], dtype=np.int64)
                ce_bits = -lp[np.arange(len(probes)), y_np] / LN2
                loo_bits.append(float(np.mean(ce_bits)))
                w_loo = np.full(len(probes), float(L) / float(len(probes)), dtype=np.float64)
                loo_cal.add(lp, y_np, weights=w_loo)
        zl, yl, wl = loo_cal.finalize()
        return {
            "mlm_15": _run_mask_mode("mlm_15", int(n_mlm15_draws)),
            "loo": _summarize_seq_bits(loo_bits, seq_lens, logits_20=zl, targets_20=yl, weights=wl),
            "mdm_16": _run_mask_mode("mdm_16", int(n_mdm_strata)),
        }

    def evaluate_sequence_split(
        self,
        records: Sequence[AuditSequenceRecord],
        *,
        batch_size: int = 16,
        n_mlm15_draws: int = DEFAULT_MLM15_DRAWS,
        n_loo_probes: int = DEFAULT_LOO_PROBES,
        n_mdm_strata: int = DEFAULT_MDM_STRATA,
        eval_seed: int = 20261129,
        max_calibration_positions: int = 65_536,
    ) -> dict[str, dict[str, Any]]:
        del batch_size
        seqs = [r.sequence for r in records]
        return self._eval_dplm2_records(
            self.adapter_seq,
            seqs,
            [None] * len(seqs),
            n_mlm15_draws=n_mlm15_draws,
            n_loo_probes=n_loo_probes,
            n_mdm_strata=n_mdm_strata,
            eval_seed=eval_seed,
            max_calibration_positions=max_calibration_positions,
        )

    def evaluate_3d_panel(
        self,
        records: Sequence[Paired3DRecord],
        coords_by_key: Mapping[str, np.ndarray],
        *,
        batch_size: int = 16,
        n_mlm15_draws: int = DEFAULT_MLM15_DRAWS,
        n_loo_probes: int = DEFAULT_LOO_PROBES,
        n_mdm_strata: int = DEFAULT_MDM_STRATA,
        eval_seed: int = 20261129,
        seq_reference_bits: float | None = None,
    ) -> dict[str, Any]:
        del batch_size, seq_reference_bits
        if self.adapter_struct is None:
            raise RuntimeError(f"{self.model_id}: struct_conditioned adapter not loaded")

        def _subset(sub: Sequence[Paired3DRecord]) -> dict[str, Any]:
            seqs = [r.sequence for r in sub]
            coords = [coords_by_key[r.coords_key] for r in sub]
            uncond = self._eval_dplm2_records(
                self.adapter_seq,
                seqs,
                [None] * len(seqs),
                n_mlm15_draws=n_mlm15_draws,
                n_loo_probes=n_loo_probes,
                n_mdm_strata=n_mdm_strata,
                eval_seed=eval_seed,
            )
            cond = self._eval_dplm2_records(
                self.adapter_struct,
                seqs,
                coords,
                n_mlm15_draws=n_mlm15_draws,
                n_loo_probes=n_loo_probes,
                n_mdm_strata=n_mdm_strata,
                eval_seed=eval_seed,
            )
            return {
                "unconditioned_null_Z": uncond,
                "conditioned_3d": cond,
                "b_ext": {
                    k: compute_multitrack_b_ext(cond[k]["bits_per_res"], h_unconditioned=uncond[k]["bits_per_res"])
                    for k in ("mlm_15", "loo", "mdm_16")
                },
            }

        by_group: dict[str, list[Paired3DRecord]] = defaultdict(list)
        for r in records:
            by_group[r.domain_group].append(r)
        return {
            "overall": _subset(records),
            "by_domain_group": {g: _subset(by_group[g]) for g in sorted(by_group.keys())},
        }


# --------------------------------------------------------------------------- #
# 4D. ESM-3 Multi-Track (Optional — requires ``esm`` SDK runtime)
# --------------------------------------------------------------------------- #


class ESM3OpenEvaluator(OpenModelAuditEvaluator):
    """Evaluates ESM-3 sequence track unconditioned ($Z=\\varnothing$) vs 3D-conditioned ($Z_{\\mathrm{3D}}$).

    Requires the ``esm`` Python package (e.g., ``runtimes/esm3/bin/python``).
    """

    def __init__(self, model: Any, *, model_id: str = "esm3-sm-open-v1", device: str = "cuda:0") -> None:
        pcounts = count_model_parameters(model)
        first_param = next(model.parameters(), None)
        realised_dt = str(first_param.dtype).removeprefix("torch.") if first_param is not None else "float32"
        super().__init__(
            model_id=model_id,
            family_kind="structure_conditioned",
            primary_readout="mlm_15",
            n_params_total=pcounts["n_params_total"],
            n_params_emb=pcounts["n_params_emb"],
            n_params_non_emb=pcounts["n_params_non_emb"],
            device=str(device),
            dtype=realised_dt,
            supports_sequence_splits=True,
            supports_3d_panel=True,
            supports_3d_ablation=True,
            extra_metadata={"multitrack_backend": "esm3_sdk"},
        )
        self.model = model
        seq_tok = model.tokenizers.sequence
        self._mask_id = int(seq_tok.mask_token_id)
        self._aa20_ids_t = torch.as_tensor(
            _validate_aa20_ids(
                [seq_tok.convert_tokens_to_ids(aa) for aa in AA20],
                getattr(seq_tok, "unk_token_id", None),
                model_id,
            ),
            dtype=torch.long,
            device=self.device,
        )

    def _esm3_autocast_ctx(self):
        """Match ESM3 weight dtype during forward/encode (prevents Float vs BFloat16 in plddt_projection)."""
        first_param = next(self.model.parameters(), None)
        if first_param is None:
            return nullcontext()
        pdt = first_param.dtype
        dev_type = torch.device(self.device).type
        if pdt in (torch.bfloat16, torch.float16) and dev_type in ("cuda", "cpu"):
            if dev_type == "cpu" and pdt == torch.float16:
                return nullcontext()
            return torch.autocast(device_type=dev_type, dtype=pdt)
        return _precision_ctx(self.device, self.dtype)

    @torch.no_grad()
    def _eval_esm3_subset(
        self,
        seqs: Sequence[str],
        coords_list: Sequence[np.ndarray | None],
        *,
        n_mlm15_draws: int,
        n_loo_probes: int,
        n_mdm_strata: int,
        eval_seed: int,
    ) -> dict[str, dict[str, Any]]:
        try:
            from esm.sdk.api import ESMProtein  # type: ignore
        except ImportError:
            ESMProtein = None  # type: ignore[assignment]

        def _encode_protein(seq: str, coords_ncaco: np.ndarray | None):
            if ESMProtein is None:
                return self.model.encode((seq, coords_ncaco))
            if coords_ncaco is None:
                prot = ESMProtein(sequence=seq)
            else:
                # Expand (L, 4, 3) [N, CA, C, O] -> atom37 (L, 37, 3) with N=0, CA=1, C=2, O=4
                atom37 = np.full((len(seq), 37, 3), np.nan, dtype=np.float32)
                atom37[:, 0, :] = coords_ncaco[:, 0, :]
                atom37[:, 1, :] = coords_ncaco[:, 1, :]
                atom37[:, 2, :] = coords_ncaco[:, 2, :]
                atom37[:, 4, :] = coords_ncaco[:, 3, :]
                prot = ESMProtein(sequence=seq, coordinates=torch.from_numpy(atom37))
            with self._esm3_autocast_ctx():
                return self.model.encode(prot)

        encoded = [_encode_protein(s, c) for s, c in zip(seqs, coords_list)]
        seq_lens = [len(s) for s in seqs]

        def _run_mode(mode: str, n_steps: int) -> dict[str, Any]:
            seq_bits: list[list[float]] = [[] for _ in range(len(seqs))]
            cal = _CalibrationBuffer(seed=eval_seed + 17)
            with self._esm3_autocast_ctx():
                for seq_i, (seq, pt) in enumerate(zip(seqs, encoded)):
                    L = len(seq)
                    base_seq = pt.sequence.to(self.device)
                    struct_toks = pt.structure.to(self.device) if getattr(pt, "structure", None) is not None else None
                    for step in range(n_steps):
                        if mode == "mlm_15":
                            m = deterministic_fixed_rate_mask(
                                seq_i, L, draw_idx=step, mask_rate=0.15, eval_seed=eval_seed + 100
                            )
                        else:
                            m, _mc, _u = deterministic_mdm_stratum_mask(
                                seq_i, L, stratum=step, n_strata=n_steps, eval_seed=eval_seed
                            )
                        pos = np.flatnonzero(m)
                        masked_seq = base_seq.clone()
                        masked_seq[1 + torch.as_tensor(pos, dtype=torch.long, device=self.device)] = self._mask_id
                        out = self.model.forward(
                            sequence_tokens=masked_seq.unsqueeze(0),
                            structure_tokens=struct_toks.unsqueeze(0) if struct_toks is not None else None,
                        )
                        logits = out.sequence_logits[0].float()
                        tok_idx = torch.as_tensor(1 + pos, dtype=torch.long, device=self.device)
                        z20 = aa20_restricted_logits(logits.index_select(0, tok_idx), self._aa20_ids_t)
                        y_np = np.asarray([AA20_INDEX[seq[int(p)]] for p in pos], dtype=np.int64)
                        y_t = torch.as_tensor(y_np, dtype=torch.long, device=self.device)
                        ce_bits = F.cross_entropy(z20, y_t, reduction="none").to(torch.float64) / LN2
                        seq_bits[seq_i].append(float(ce_bits.mean().item()))
                        w_site = np.full(len(pos), float(L) / float(n_steps * len(pos)), dtype=np.float64)
                        cal.add(z20.detach().cpu().numpy(), y_np, weights=w_site)
            z, y, w = cal.finalize()
            return _summarize_seq_bits([float(np.mean(v)) for v in seq_bits], seq_lens, logits_20=z, targets_20=y, weights=w)

        del n_loo_probes
        return {
            "mlm_15": _run_mode("mlm_15", int(n_mlm15_draws)),
            "mdm_16": _run_mode("mdm_16", int(n_mdm_strata)),
        }

    def evaluate_sequence_split(
        self,
        records: Sequence[AuditSequenceRecord],
        *,
        batch_size: int = 16,
        n_mlm15_draws: int = DEFAULT_MLM15_DRAWS,
        n_loo_probes: int = DEFAULT_LOO_PROBES,
        n_mdm_strata: int = DEFAULT_MDM_STRATA,
        eval_seed: int = 20261129,
        max_calibration_positions: int = 65_536,
    ) -> dict[str, dict[str, Any]]:
        del batch_size, max_calibration_positions
        seqs = [r.sequence for r in records]
        return self._eval_esm3_subset(
            seqs,
            [None] * len(seqs),
            n_mlm15_draws=n_mlm15_draws,
            n_loo_probes=n_loo_probes,
            n_mdm_strata=n_mdm_strata,
            eval_seed=eval_seed,
        )

    def evaluate_3d_panel(
        self,
        records: Sequence[Paired3DRecord],
        coords_by_key: Mapping[str, np.ndarray],
        *,
        batch_size: int = 16,
        n_mlm15_draws: int = DEFAULT_MLM15_DRAWS,
        n_loo_probes: int = DEFAULT_LOO_PROBES,
        n_mdm_strata: int = DEFAULT_MDM_STRATA,
        eval_seed: int = 20261129,
        seq_reference_bits: float | None = None,
    ) -> dict[str, Any]:
        del batch_size, seq_reference_bits

        def _sub(sub: Sequence[Paired3DRecord]) -> dict[str, Any]:
            seqs = [r.sequence for r in sub]
            coords = [coords_by_key[r.coords_key] for r in sub]
            uncond = self._eval_esm3_subset(
                seqs, [None] * len(seqs), n_mlm15_draws=n_mlm15_draws, n_loo_probes=n_loo_probes, n_mdm_strata=n_mdm_strata, eval_seed=eval_seed
            )
            cond = self._eval_esm3_subset(
                seqs, coords, n_mlm15_draws=n_mlm15_draws, n_loo_probes=n_loo_probes, n_mdm_strata=n_mdm_strata, eval_seed=eval_seed
            )
            return {
                "unconditioned_null_Z": uncond,
                "conditioned_3d": cond,
                "b_ext": {
                    k: compute_multitrack_b_ext(cond[k]["bits_per_res"], h_unconditioned=uncond[k]["bits_per_res"])
                    for k in ("mlm_15", "mdm_16")
                },
            }

        by_group: dict[str, list[Paired3DRecord]] = defaultdict(list)
        for r in records:
            by_group[r.domain_group].append(r)
        return {
            "overall": _sub(records),
            "by_domain_group": {g: _sub(by_group[g]) for g in sorted(by_group.keys())},
        }


# --------------------------------------------------------------------------- #
# 4E. Prot-T5 Encoder-Decoder Sentinel LOO Evaluator
# --------------------------------------------------------------------------- #


def _validate_aa20_ids(aa20_ids: Sequence[int | None], unk_id: int | None, label: str) -> list[int]:
    """Fail fast if any canonical AA20 residue lacks a distinct valid token ID."""
    resolved: list[int] = []
    for aa, tid in zip(AA20, aa20_ids):
        if tid is None or int(tid) < 0 or (unk_id is not None and int(tid) == int(unk_id)):
            raise ValueError(f"{label}: tokenizer lacks valid dedicated token ID for AA {aa!r} (got {tid!r}, unk={unk_id!r})")
        resolved.append(int(tid))
    if len(set(resolved)) != 20:
        raise ValueError(f"{label}: duplicate token IDs across AA20 residues: {resolved}")
    return resolved


class ProtT5OpenEvaluator(OpenModelAuditEvaluator):
    """Single-site sentinel ``<extra_id_0>`` LOO evaluator for ``T5ForConditionalGeneration`` (Prot-T5)."""

    def __init__(
        self,
        *,
        model_id: str,
        model: nn.Module,
        tokenizer: Any,
        aa20_ids: Sequence[int],
        sentinel_id: int,
        decoder_start_id: int,
        max_residues: int = 1022,
        device: str = "cpu",
        dtype: str = "float32",
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        pcounts = count_model_parameters(model)
        super().__init__(
            model_id=model_id,
            family_kind="mlm",
            primary_readout="loo",
            n_params_total=pcounts["n_params_total"],
            n_params_emb=pcounts["n_params_emb"],
            n_params_non_emb=pcounts["n_params_non_emb"],
            device=device,
            dtype=dtype,
            supports_sequence_splits=True,
            supports_3d_panel=False,
            supports_3d_ablation=False,
            extra_metadata={"architecture": "t5_seq2seq_sentinel_loo", **(dict(extra_metadata) if extra_metadata else {})},
        )
        self.model = model
        self.tokenizer = tokenizer
        self.aa20_ids = [int(x) for x in aa20_ids]
        self.aa20_ids_t = torch.as_tensor(self.aa20_ids, dtype=torch.long, device=self.device)
        self.aa_to_id = {aa: tid for aa, tid in zip(AA20, self.aa20_ids)}
        self.sentinel_id = int(sentinel_id)
        self.decoder_start_id = int(decoder_start_id)
        self.eos_id = getattr(tokenizer, "eos_token_id", None)
        self.max_residues = int(max_residues)

    @torch.no_grad()
    def evaluate_sequence_split(
        self,
        records: Sequence[AuditSequenceRecord],
        *,
        batch_size: int = 16,
        n_mlm15_draws: int = DEFAULT_MLM15_DRAWS,
        n_loo_probes: int = DEFAULT_LOO_PROBES,
        n_mdm_strata: int = DEFAULT_MDM_STRATA,
        eval_seed: int = 20261129,
        max_calibration_positions: int = 65_536,
    ) -> dict[str, dict[str, Any]]:
        del n_mlm15_draws, n_mdm_strata
        seq_bits: list[float] = []
        seq_lens: list[int] = []
        cal_buf = _CalibrationBuffer(max_positions=max_calibration_positions, seed=eval_seed + 61)
        bs = max(1, int(batch_size))

        with _precision_ctx(self.device, self.dtype):
            for orig_i, rec in enumerate(records):
                seq = rec.sequence[: self.max_residues]
                L = len(seq)
                base_ids = [self.aa_to_id[aa] for aa in seq]
                if self.eos_id is not None:
                    base_ids.append(int(self.eos_id))
                row = torch.as_tensor(base_ids, dtype=torch.long, device=self.device)
                probes = deterministic_loo_positions(orig_i, L, n_probes=n_loo_probes, eval_seed=eval_seed + 200)
                site_bits: list[float] = []
                for start in range(0, len(probes), bs):
                    chunk = probes[start : start + bs]
                    b_inp = row.unsqueeze(0).repeat(len(chunk), 1)
                    for r_i, p in enumerate(chunk):
                        b_inp[r_i, int(p)] = self.sentinel_id
                    b_attn = torch.ones_like(b_inp)
                    dec_inp = torch.tensor(
                        [[self.decoder_start_id, self.sentinel_id]] * len(chunk),
                        dtype=torch.long,
                        device=self.device,
                    )
                    out = self.model(input_ids=b_inp, attention_mask=b_attn, decoder_input_ids=dec_inp)
                    # Decoder step 1 (conditioned on [decoder_start_id, <extra_id_0>]) predicts the masked AA
                    step_logits = out.logits[:, 1, :].float()
                    z20 = aa20_restricted_logits(step_logits, self.aa20_ids_t)
                    y_np = np.asarray([AA20_INDEX[seq[int(p)]] for p in chunk], dtype=np.int64)
                    y_t = torch.as_tensor(y_np, dtype=torch.long, device=self.device)
                    ce_bits = F.cross_entropy(z20, y_t, reduction="none").to(torch.float64) / LN2
                    site_bits.extend(ce_bits.detach().cpu().tolist())
                    w_chunk = np.full(len(chunk), float(L) / float(len(probes)), dtype=np.float64)
                    cal_buf.add(z20.detach().cpu().numpy(), y_np, weights=w_chunk)
                seq_bits.append(float(np.mean(site_bits)))
                seq_lens.append(L)

        z, y, w = cal_buf.finalize()
        summary = _summarize_seq_bits(seq_bits, seq_lens, logits_20=z, targets_20=y, weights=w)
        summary["n_probes_per_seq"] = int(n_loo_probes)
        return {"loo": summary}


# --------------------------------------------------------------------------- #
# 5. Unified Model Loader Factory (Handles Short Names & Direct Cuihu Paths)
# --------------------------------------------------------------------------- #


def _resolve_local_dir(name_or_path: str, model_root: Path | str | None = None) -> Path:
    p = Path(name_or_path).expanduser()
    if p.is_dir():
        return p.resolve()
    roots: list[Path] = []
    if model_root is not None:
        roots.append(Path(model_root).expanduser())
    for var in ("GENERAL_PLM_MODEL_ROOT", "MODEL_ROOT", "MODEL_BASE_DIR"):
        v = os.environ.get(var, "").strip()
        if v:
            roots.append(Path(v).expanduser())
    roots.extend(DEFAULT_CAUSAL_MODEL_ROOTS)

    candidates = [name_or_path]
    alias_hits = _CUIHU_MODEL_ALIASES.get(name_or_path.lower())
    if alias_hits:
        candidates.extend(alias_hits)

    for root in roots:
        for cand_name in candidates:
            cand = root / cand_name
            if cand.is_dir():
                return cand.resolve()
    raise FileNotFoundError(
        f"Could not locate local checkpoint directory for {name_or_path!r} "
        f"(checked aliases={candidates} across {len(roots)} roots)"
    )


def _read_config_json(ckpt_dir: Path) -> dict[str, Any]:
    cfg_p = ckpt_dir / "config.json"
    if cfg_p.is_file():
        try:
            return json.loads(cfg_p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _detect_family_from_dir(ckpt_dir: Path, model_spec: str) -> str:
    """Infer exact loader route from directory name, files, and ``config.json``."""
    low = f"{model_spec} {ckpt_dir.name}".lower()
    cfg = _read_config_json(ckpt_dir)
    mtype = str(cfg.get("model_type", "")).lower()
    archs = " ".join(str(a).lower() for a in (cfg.get("architectures") or ()))

    if (ckpt_dir / "weight/vanilla_model_weights").is_dir() or "proteinmpnn" in low:
        return "proteinmpnn"
    if "dplm2" in low or "dplm_2" in low or int(cfg.get("vocab_size", 0)) >= 8000 and "esm" in mtype:
        return "dplm2"
    if "saprot" in low or (mtype == "esm" and int(cfg.get("vocab_size", 0)) in (441, 446)):
        return "saprot"
    if "esm3" in low:
        return "esm3"
    if "esmc" in low or mtype == "esmc":
        return "esmc_hf"
    if "amplify" in low or mtype == "amplify":
        return "amplify_mlm"
    if "proteinglm" in low and ("mlm" in low or "maskedlm" in archs):
        return "proteinglm_mlm"
    if "dplm" in low:
        return "dplm_mdm"
    if "progen3" in low or mtype == "progen3":
        return "progen3_causal"
    if "protgpt3" in low:
        return "protgpt3_causal"
    if "dayhoff" in low or mtype == "jamba":
        return "dayhoff_causal"
    if "galactica" in low:
        return "galactica_causal"
    if "prot_t5" in low or "prot-t5" in low or mtype == "t5" or "t5for" in archs or "t5encoder" in archs:
        return "prot_t5"
    if (
        "progen2" in low
        or mtype == "progen"
        or "rita" in low
        or "causallm" in archs
        or mtype in ("opt", "gpt2", "llama", "mixtral", "mistral", "qwen2")
    ):
        return "causal_ar"
    if "esm" in low or mtype == "esm" or "maskedlm" in archs:
        return "esm_mlm"
    return "generic_mlm"


def _wrap_general_plm_adapter(
    ad: Any,
    *,
    model_id: str,
    family_override: str,
    dtype_name: str,
    device: str,
) -> OpenModelAuditEvaluator:
    spec = ad.spec
    if spec.family == "saprot" or family_override == "saprot":
        return SaProtOpenEvaluator(ad)
    if spec.is_causal:
        if spec.family == "progen3":
            def _pg3_fwd(inp: torch.Tensor) -> torch.Tensor:
                pos_ids = torch.arange(inp.shape[1], dtype=torch.long, device=inp.device).unsqueeze(0)
                seq_ids = torch.zeros_like(inp)
                return ad.handle.model(input_ids=inp, position_ids=pos_ids, sequence_ids=seq_ids, use_cache=False).logits

            sample_prefix = ad.handle.batch(["A"], reverse=False)["input_ids"][0].tolist()[:2]
            return CausalAROpenEvaluator(
                model_id=model_id,
                model=ad.model,
                aa20_ids=[ad.aa_to_id[aa] for aa in AA20],
                bos_prefix_ids=sample_prefix,
                max_positions=spec.max_residues,
                device=device,
                dtype=ad.dtype,
                forward_fn=_pg3_fwd,
                extra_metadata={"checkpoint_dir": str(ad.checkpoint_dir)},
            )
        return CausalAROpenEvaluator(
            model_id=model_id,
            model=ad.model,
            aa20_ids=ad.aa20_ids,
            bos_prefix_ids=ad.bos_prefix_ids,
            eos_suffix_ids=ad.eos_suffix_ids,
            max_positions=spec.max_residues,
            device=device,
            dtype=ad.dtype,
            forward_fn=ad._forward_ids,
            extra_metadata={"checkpoint_dir": str(ad.checkpoint_dir)},
        )

    pad_id = getattr(ad.tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = 1
    probe_ids, probe_off = ad._encode_sequence("ACDE")
    eff_cls = probe_ids[0] if probe_off == 1 else None
    eff_eos = probe_ids[-1] if len(probe_ids) > probe_off + 4 else None
    cls_cls = MDMOpenEvaluator if (spec.paradigm == "discrete_diffusion" or family_override == "mdm") else MaskedLMOpenEvaluator
    return cls_cls(
        model_id=model_id,
        model=ad.model,
        aa20_ids=ad.aa20_ids,
        mask_id=ad.mask_id,
        pad_id=int(pad_id),
        cls_id=eff_cls,
        eos_id=eff_eos,
        max_residues=spec.max_residues,
        device=device,
        dtype=dtype_name,
        additive_attention_mask=ad.additive_attention_mask,
        forward_fn=ad._forward_logits,
        extra_metadata={"checkpoint_dir": str(ad.checkpoint_dir), "paradigm": spec.paradigm},
    )


def _resolve_progen2_or_hf_causal(
    model_spec: str,
    *,
    ckpt_dir: Path | None = None,
    device: str,
    dtype: str,
    model_root: Path | str | None = None,
    progen2_bidirectional: bool = False,
) -> CausalAROpenEvaluator:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_dir = ckpt_dir if ckpt_dir is not None else _resolve_local_dir(model_spec, model_root=model_root)
    dtype_name, torch_dt = _resolve_torch_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        str(resolved_dir),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dt,
    ).eval().to(device)
    model.requires_grad_(False)

    # Support ProGen2 raw tokenizers.Tokenizer JSON without tokenizer_config.json
    tok_json = resolved_dir / "tokenizer.json"
    is_progen2 = (
        resolved_dir.name.lower() in _PROGEN2_NAMES
        or "progen2" in model_spec.lower()
        or getattr(getattr(model, "config", None), "model_type", "") == "progen"
    )
    if is_progen2 and tok_json.is_file() and not (resolved_dir / "tokenizer_config.json").is_file():
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tok_json))
        aa20_ids = _validate_aa20_ids([tok.token_to_id(aa) for aa in AA20], None, model_spec)
        id_1 = int(tok.token_to_id("1"))
        id_2 = int(tok.token_to_id("2"))
        return CausalAROpenEvaluator(
            model_id=model_spec,
            model=model,
            aa20_ids=aa20_ids,
            bos_prefix_ids=[id_1],
            eos_suffix_ids=[id_2],
            reverse_bos_prefix_ids=[id_2],
            reverse_eos_suffix_ids=[id_1],
            score_bidirectional=progen2_bidirectional,
            max_positions=int(getattr(model.config, "n_positions", 1024)),
            device=device,
            dtype=dtype_name,
            extra_metadata={"checkpoint_dir": str(resolved_dir), "tokenizer_kind": "progen2_tokenizers_json"},
        )

    tokenizer = AutoTokenizer.from_pretrained(str(resolved_dir), local_files_only=True, trust_remote_code=True)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    raw_aa_ids: list[int | None] = []
    for aa in AA20:
        tid = tokenizer.convert_tokens_to_ids(aa)
        if tid is None or (unk_id is not None and int(tid) == int(unk_id)):
            enc = tokenizer.encode(aa, add_special_tokens=False)
            tid = int(enc[-1]) if len(enc) == 1 else None
        raw_aa_ids.append(tid)
    aa20_ids = _validate_aa20_ids(raw_aa_ids, unk_id, model_spec)

    # Detect Galactica protein mode ([START_AMINO]...[END_AMINO])
    start_amino_id = tokenizer.convert_tokens_to_ids("[START_AMINO]")
    end_amino_id = tokenizer.convert_tokens_to_ids("[END_AMINO]")
    has_amino_tags = (
        start_amino_id is not None
        and end_amino_id is not None
        and (unk_id is None or (int(start_amino_id) != int(unk_id) and int(end_amino_id) != int(unk_id)))
    )
    is_galactica = "galactica" in f"{model_spec} {resolved_dir.name}".lower() or has_amino_tags

    bos_prefix: list[int] = []
    eos_suffix: list[int] = []
    rev_bos: list[int] | None = None
    rev_eos: list[int] = []
    extra_meta: dict[str, Any] = {"checkpoint_dir": str(resolved_dir), "model_class": type(model).__name__}

    if is_galactica:
        if not has_amino_tags:
            raise ValueError(
                f"{model_spec}: Galactica checkpoint missing required [START_AMINO] / [END_AMINO] tokens "
                f"(got start={start_amino_id!r}, end={end_amino_id!r})"
            )
        bos_prefix = [int(start_amino_id)]
        eos_suffix = [int(end_amino_id)]
        extra_meta["packing"] = "galactica_start_end_amino"
    elif is_progen2:
        t1 = tokenizer.convert_tokens_to_ids("1")
        t2 = tokenizer.convert_tokens_to_ids("2")
        if t1 is not None:
            bos_prefix = [int(t1)]
        if t2 is not None:
            eos_suffix = [int(t2)]
            rev_bos = [int(t2)]
            rev_eos = [int(t1)] if t1 is not None else []
    elif tokenizer.bos_token_id is not None:
        bos_prefix = [int(tokenizer.bos_token_id)]
    elif tokenizer.eos_token_id is not None:
        bos_prefix = [int(tokenizer.eos_token_id)]

    max_pos = int(
        getattr(model.config, "n_positions", None)
        or getattr(model.config, "max_position_embeddings", 1024)
    )
    return CausalAROpenEvaluator(
        model_id=model_spec,
        model=model,
        aa20_ids=aa20_ids,
        bos_prefix_ids=bos_prefix,
        eos_suffix_ids=eos_suffix,
        reverse_bos_prefix_ids=rev_bos,
        reverse_eos_suffix_ids=rev_eos,
        score_bidirectional=progen2_bidirectional and is_progen2,
        max_positions=max_pos,
        device=device,
        dtype=dtype_name,
        extra_metadata=extra_meta,
    )


def _load_prot_t5(
    model_spec: str,
    ckpt_dir: Path,
    *,
    device: str,
    dtype: str,
) -> ProtT5OpenEvaluator:
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    cfg = _read_config_json(ckpt_dir)
    archs = [str(a) for a in (cfg.get("architectures") or ())]
    if any("EncoderModel" in a for a in archs) and not any("ConditionalGeneration" in a for a in archs):
        raise RuntimeError(
            f"{model_spec}: checkpoint at {ckpt_dir} is encoder-only ({archs}) with no T5 LM head/decoder; "
            "ProtT5OpenEvaluator requires T5ForConditionalGeneration weights."
        )
    dtype_name, torch_dt = _resolve_torch_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True, legacy=False)
    model, info = T5ForConditionalGeneration.from_pretrained(
        str(ckpt_dir),
        local_files_only=True,
        output_loading_info=True,
        torch_dtype=torch_dt,
    )
    missing_dec = [k for k in info.get("missing_keys", ()) if "decoder" in k or "lm_head" in k]
    if missing_dec:
        raise RuntimeError(
            f"{model_spec}: Prot-T5 checkpoint at {ckpt_dir} is missing decoder/lm_head weights "
            f"(e.g. {missing_dec[:5]}); encoder-only checkpoints cannot be evaluated for AA20 cross-entropy."
        )
    model = model.eval().to(device)
    model.requires_grad_(False)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    raw_ids: list[int | None] = []
    for aa in AA20:
        tid = tokenizer.convert_tokens_to_ids(f"\u2581{aa}")
        if tid is None or (unk_id is not None and int(tid) == int(unk_id)):
            tid = tokenizer.convert_tokens_to_ids(aa)
        raw_ids.append(tid)
    aa20_ids = _validate_aa20_ids(raw_ids, unk_id, model_spec)
    sentinel_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    if sentinel_id is None or (unk_id is not None and int(sentinel_id) == int(unk_id)):
        raise ValueError(f"{model_spec}: Prot-T5 tokenizer missing <extra_id_0> sentinel token")
    dec_start = getattr(model.config, "decoder_start_token_id", None)
    if dec_start is None:
        dec_start = getattr(tokenizer, "pad_token_id", 0) or 0
    return ProtT5OpenEvaluator(
        model_id=model_spec,
        model=model,
        tokenizer=tokenizer,
        aa20_ids=aa20_ids,
        sentinel_id=int(sentinel_id),
        decoder_start_id=int(dec_start),
        device=device,
        dtype=dtype_name,
        extra_metadata={"checkpoint_dir": str(ckpt_dir)},
    )


def _strip_state_dict_prefix(sd: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Unwrap nested state_dict containers and strip common ``model.``/``module.`` prefixes."""
    if "state_dict" in sd and isinstance(sd["state_dict"], Mapping):
        sd = sd["state_dict"]  # type: ignore[assignment]
    elif "model" in sd and isinstance(sd["model"], Mapping):
        sd = sd["model"]  # type: ignore[assignment]
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        nk = str(k)
        for pref in ("module.", "model."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        out[nk] = v
    return out


def convert_esmc_sdk_state_dict_to_hf(sdk_sd: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert raw EvolutionaryScale SDK ESMC state_dict keys into HuggingFace ``EsmcForMaskedLM`` keys."""
    sd = _strip_state_dict_prefix(sdk_sd)
    if "esmc.embed_tokens.weight" in sd:
        return dict(sd)
    hf: dict[str, torch.Tensor] = {}
    if "embed.weight" in sd:
        hf["esmc.embed_tokens.weight"] = sd["embed.weight"]
    if "transformer.norm.weight" in sd:
        hf["esmc.norm.weight"] = sd["transformer.norm.weight"]
    if "transformer.norm.bias" in sd:
        hf["esmc.norm.bias"] = sd["transformer.norm.bias"]
    seq_head_map = {
        "sequence_head.0.weight": "lm_head.dense.weight",
        "sequence_head.0.bias": "lm_head.dense.bias",
        "sequence_head.2.weight": "lm_head.layer_norm.weight",
        "sequence_head.2.bias": "lm_head.layer_norm.bias",
        "sequence_head.3.weight": "lm_head.decoder.weight",
        "sequence_head.3.bias": "lm_head.decoder.bias",
    }
    for sk, hk in seq_head_map.items():
        if sk in sd:
            hf[hk] = sd[sk]

    block_indices = sorted(
        {
            int(k.split(".")[2])
            for k in sd
            if k.startswith("transformer.blocks.") and len(k.split(".")) > 3 and k.split(".")[2].isdigit()
        }
    )
    for i in block_indices:
        sp = f"transformer.blocks.{i}"
        hp = f"esmc.layers.{i}"
        for s_suf, h_suf in (
            ("attn.layernorm_qkv.0.weight", "input_layernorm.weight"),
            ("attn.layernorm_qkv.0.bias", "input_layernorm.bias"),
            ("attn.out_proj.weight", "self_attn.o_proj.weight"),
            ("attn.q_ln.weight", "self_attn.q_norm.weight"),
            ("attn.k_ln.weight", "self_attn.k_norm.weight"),
            ("ffn.0.weight", "post_attention_layernorm.weight"),
            ("ffn.0.bias", "post_attention_layernorm.bias"),
            ("ffn.3.weight", "mlp.down_proj.weight"),
        ):
            if f"{sp}.{s_suf}" in sd:
                hf[f"{hp}.{h_suf}"] = sd[f"{sp}.{s_suf}"]
        qkv_k = f"{sp}.attn.layernorm_qkv.1.weight"
        if qkv_k in sd:
            q, k_w, v = sd[qkv_k].chunk(3, dim=0)
            hf[f"{hp}.self_attn.q_proj.weight"] = q
            hf[f"{hp}.self_attn.k_proj.weight"] = k_w
            hf[f"{hp}.self_attn.v_proj.weight"] = v
        ffn1_k = f"{sp}.ffn.1.weight"
        if ffn1_k in sd:
            # EvolutionaryScale SwiGLU: x1, x2 = x.chunk(2, dim=-1); F.silu(x1) * x2 -> [gate, up]
            gate, up = sd[ffn1_k].chunk(2, dim=0)
            hf[f"{hp}.mlp.gate_proj.weight"] = gate
            hf[f"{hp}.mlp.up_proj.weight"] = up
    return hf


def convert_esmc_hf_state_dict_to_sdk(hf_sd: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert HuggingFace ``EsmcForMaskedLM`` state_dict keys into raw EvolutionaryScale SDK ESMC keys."""
    sd = _strip_state_dict_prefix(hf_sd)
    if "embed.weight" in sd:
        return dict(sd)
    sdk: dict[str, torch.Tensor] = {}
    if "esmc.embed_tokens.weight" in sd:
        sdk["embed.weight"] = sd["esmc.embed_tokens.weight"]
    if "esmc.norm.weight" in sd:
        sdk["transformer.norm.weight"] = sd["esmc.norm.weight"]
    if "esmc.norm.bias" in sd:
        sdk["transformer.norm.bias"] = sd["esmc.norm.bias"]
    head_map = {
        "lm_head.dense.weight": "sequence_head.0.weight",
        "lm_head.dense.bias": "sequence_head.0.bias",
        "lm_head.layer_norm.weight": "sequence_head.2.weight",
        "lm_head.layer_norm.bias": "sequence_head.2.bias",
        "lm_head.decoder.weight": "sequence_head.3.weight",
        "lm_head.decoder.bias": "sequence_head.3.bias",
    }
    for hk, sk in head_map.items():
        if hk in sd:
            sdk[sk] = sd[hk]

    layer_indices = sorted(
        {
            int(k.split(".")[2])
            for k in sd
            if k.startswith("esmc.layers.") and len(k.split(".")) > 3 and k.split(".")[2].isdigit()
        }
    )
    for i in layer_indices:
        hp = f"esmc.layers.{i}"
        sp = f"transformer.blocks.{i}"
        for h_suf, s_suf in (
            ("input_layernorm.weight", "attn.layernorm_qkv.0.weight"),
            ("input_layernorm.bias", "attn.layernorm_qkv.0.bias"),
            ("self_attn.o_proj.weight", "attn.out_proj.weight"),
            ("self_attn.q_norm.weight", "attn.q_ln.weight"),
            ("self_attn.k_norm.weight", "attn.k_ln.weight"),
            ("post_attention_layernorm.weight", "ffn.0.weight"),
            ("post_attention_layernorm.bias", "ffn.0.bias"),
            ("mlp.down_proj.weight", "ffn.3.weight"),
        ):
            if f"{hp}.{h_suf}" in sd:
                sdk[f"{sp}.{s_suf}"] = sd[f"{hp}.{h_suf}"]
        q_k, k_k, v_k = (
            f"{hp}.self_attn.q_proj.weight",
            f"{hp}.self_attn.k_proj.weight",
            f"{hp}.self_attn.v_proj.weight",
        )
        if q_k in sd and k_k in sd and v_k in sd:
            sdk[f"{sp}.attn.layernorm_qkv.1.weight"] = torch.cat([sd[q_k], sd[k_k], sd[v_k]], dim=0)
        g_k, u_k = f"{hp}.mlp.gate_proj.weight", f"{hp}.mlp.up_proj.weight"
        if g_k in sd and u_k in sd:
            sdk[f"{sp}.ffn.1.weight"] = torch.cat([sd[g_k], sd[u_k]], dim=0)
    return sdk


def _load_raw_state_dict_from_dir(ckpt_dir: Path) -> dict[str, torch.Tensor]:
    """Load tensors from safetensors or PyTorch .pth/.pt/.bin shards inside ``ckpt_dir``."""
    search_dirs = [ckpt_dir]
    if (ckpt_dir / "data/weights").is_dir():
        search_dirs.insert(0, ckpt_dir / "data/weights")
    for sdir in search_dirs:
        st_files = sorted(sdir.glob("*.safetensors"))
        if st_files:
            from safetensors.torch import load_file

            merged: dict[str, torch.Tensor] = {}
            for sf in st_files:
                merged.update(load_file(str(sf), device="cpu"))
            return _strip_state_dict_prefix(merged)
        pt_files = sorted(
            [p for p in sdir.iterdir() if p.is_file() and p.suffix in (".pth", ".pt", ".bin") and "optimizer" not in p.name.lower()]
        )
        if pt_files:
            merged_pt: dict[str, torch.Tensor] = {}
            for pf in pt_files:
                loaded = torch.load(str(pf), map_location="cpu")
                if isinstance(loaded, Mapping):
                    merged_pt.update(_strip_state_dict_prefix(loaded))
            if merged_pt:
                return merged_pt
    raise FileNotFoundError(f"No ESMC weight files (.safetensors/.pth/.pt/.bin) found under {ckpt_dir}")


def _infer_esmc_dims(sdk_sd: Mapping[str, torch.Tensor]) -> tuple[int, int, int, int]:
    """Infer ``(d_model, n_heads, n_layers, intermediate_size)`` from SDK ESMC weights."""
    d_model = int(sdk_sd["embed.weight"].shape[1])
    n_layers = (
        max(
            int(k.split(".")[2])
            for k in sdk_sd
            if k.startswith("transformer.blocks.") and len(k.split(".")) > 3 and k.split(".")[2].isdigit()
        )
        + 1
    )
    known_heads = {960: 15, 1152: 18, 2560: 40}
    n_heads = known_heads.get(d_model, max(1, d_model // 64))
    ffn1_k = "transformer.blocks.0.ffn.1.weight"
    intermediate = int(sdk_sd[ffn1_k].shape[0] // 2) if ffn1_k in sdk_sd else int(round(d_model * 8 / 3))
    return d_model, n_heads, n_layers, intermediate


def _load_esmc_unified(
    model_spec: str,
    ckpt_dir: Path,
    *,
    device: str,
    dtype: str,
    family_override: str = "auto",
) -> OpenModelAuditEvaluator:
    """Load ESMC cleanly from either HF ``esmc-*-hf`` OR raw EvolutionaryScale SDK ``ESMC-*`` checkpoints."""
    from src.capability.general_plm.adapters import _load_esmc_hf
    from src.capability.general_plm.registry import GeneralPLMSpec

    dtype_name, torch_dt = _resolve_torch_dtype(dtype)
    cfg = _read_config_json(ckpt_dir)
    is_hf_dir = cfg.get("model_type") == "esmc" and (
        any(ckpt_dir.glob("*.safetensors")) or any(ckpt_dir.glob("*.bin"))
    )

    # 1. Fast path: native HF EsmcForMaskedLM directory under transformers>=5
    hf_err: Exception | None = None
    if is_hf_dir:
        try:
            synth_spec = GeneralPLMSpec(
                arm=model_spec,
                family="esmc",
                paradigm="masked_lm",
                loader_kind="esmc_hf",
                candidate_dirs=(ckpt_dir.name,),
                hf_repo=ckpt_dir.name,
                max_residues=2048,
                trust_remote_code=True,
            )
            ad = _load_esmc_hf(synth_spec, ckpt_dir, device=device, dtype=dtype_name)
            return _wrap_general_plm_adapter(
                ad, model_id=model_spec, family_override=family_override, dtype_name=dtype_name, device=device
            )
        except Exception as exc:
            hf_err = exc

    # 2. Load raw state dict (accepts both SDK `ESMC-*` and HF `esmc-*-hf`!)
    raw_sd = _load_raw_state_dict_from_dir(ckpt_dir)
    sdk_sd = convert_esmc_hf_state_dict_to_sdk(raw_sd) if "esmc.embed_tokens.weight" in raw_sd else raw_sd
    hf_sd = convert_esmc_sdk_state_dict_to_hf(raw_sd) if "embed.weight" in raw_sd else raw_sd
    d_model, n_heads, n_layers, intermediate = _infer_esmc_dims(sdk_sd)

    # 2a. Try native transformers>=5 EsmcForMaskedLM with converted HF state_dict
    try:
        from transformers import EsmcConfig, EsmcForMaskedLM  # type: ignore

        e_cfg = EsmcConfig(
            vocab_size=int(hf_sd["esmc.embed_tokens.weight"].shape[0]),
            hidden_size=d_model,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            intermediate_size=intermediate,
            max_position_embeddings=2048,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            mask_token_id=32,
            tie_word_embeddings=False,
        )
        model = EsmcForMaskedLM(e_cfg)
        missing, unexpected = model.load_state_dict(hf_sd, strict=False)
        if missing:
            raise RuntimeError(f"EsmcForMaskedLM missing keys after SDK->HF conversion: {missing[:6]}")
        model = model.to(device=device, dtype=torch_dt).eval()
        model.requires_grad_(False)
        # Canonical ESMC token IDs (identical to ESM-2 / EsmSequenceTokenizer)
        esmc_aa_order = "LAGVSERTIDPKQNFYMHWC"
        aa_map = {aa: idx + 4 for idx, aa in enumerate(esmc_aa_order)}
        aa20_ids = _validate_aa20_ids([aa_map[aa] for aa in AA20], 3, model_spec)
        cls_cls = MDMOpenEvaluator if family_override == "mdm" else MaskedLMOpenEvaluator
        return cls_cls(
            model_id=model_spec,
            model=model,
            aa20_ids=aa20_ids,
            mask_id=32,
            pad_id=1,
            cls_id=0,
            eos_id=2,
            max_residues=2048,
            device=device,
            dtype=dtype_name,
            extra_metadata={
                "checkpoint_dir": str(ckpt_dir),
                "esmc_backend": "hf_converted_from_sdk",
                "unexpected_keys": list(unexpected),
            },
        )
    except Exception as conv_hf_err:
        # 2b. Fallback: EvolutionaryScale `esm` SDK ESMC (accepts BOTH SDK and HF->SDK converted weights!)
        try:
            from esm.models.esmc import ESMC  # type: ignore
            from esm.tokenization import get_esmc_model_tokenizers  # type: ignore

            tok = get_esmc_model_tokenizers()
            sdk_model = ESMC(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                tokenizer=tok,
                use_flash_attn=False,
            )
            missing, unexpected = sdk_model.load_state_dict(sdk_sd, strict=False)
            if missing:
                raise RuntimeError(f"SDK ESMC missing keys: {missing[:6]}")
            sdk_model = sdk_model.to(device=device, dtype=torch_dt).eval()
            sdk_model.requires_grad_(False)
            aa20_ids = _validate_aa20_ids(
                [tok.convert_tokens_to_ids(aa) for aa in AA20],
                getattr(tok, "unk_token_id", None),
                model_spec,
            )

            def _sdk_fwd(input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
                del attention_mask
                dev_type = torch.device(device).type
                ctx = (
                    torch.autocast(device_type=dev_type, dtype=torch_dt)
                    if torch_dt in (torch.bfloat16, torch.float16) and dev_type in ("cuda", "cpu")
                    else nullcontext()
                )
                with ctx:
                    out = sdk_model(sequence_tokens=input_ids)
                return out.sequence_logits.float()

            cls_cls = MDMOpenEvaluator if family_override == "mdm" else MaskedLMOpenEvaluator
            return cls_cls(
                model_id=model_spec,
                model=sdk_model,
                aa20_ids=aa20_ids,
                mask_id=int(tok.mask_token_id),
                pad_id=int(tok.pad_token_id),
                cls_id=int(tok.cls_token_id),
                eos_id=int(tok.eos_token_id),
                max_residues=2048,
                device=device,
                dtype=dtype_name,
                forward_fn=_sdk_fwd,
                extra_metadata={
                    "checkpoint_dir": str(ckpt_dir),
                    "esmc_backend": "esm_sdk",
                    "unexpected_keys": list(unexpected),
                },
            )
        except Exception as sdk_err:
            raise RuntimeError(
                f"{model_spec}: failed to load ESMC from {ckpt_dir} via HF ({hf_err or conv_hf_err}) "
                f"and esm SDK ({sdk_err}). Use a runtime with transformers>=5 (EsmcForMaskedLM) or the 'esm' package."
            ) from sdk_err


def _load_esm3_evaluator(
    model_spec: str,
    *,
    device: str,
    dtype: str,
    model_root: Path | str | None = None,
) -> ESM3OpenEvaluator:
    """Load ESM-3 offline from a local checkpoint dir or via ``ESM3.from_pretrained`` with matched dtype."""
    try:
        from esm.models.esm3 import ESM3  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"{model_spec}: ESM-3 requires the 'esm' package (e.g. runtimes/esm3/bin/python)"
        ) from exc

    dtype_name, torch_dt = _resolve_torch_dtype(dtype)
    dev = torch.device(device)

    local_dir: Path | None = None
    try:
        cand = _resolve_local_dir(model_spec, model_root=model_root)
        if (cand / "data/weights/esm3_sm_open_v1.pth").is_file():
            local_dir = cand
    except FileNotFoundError:
        local_dir = None

    if local_dir is not None:
        import esm.utils.constants.esm3 as esm3_const  # type: ignore
        import esm.tokenization.function_tokenizer as esm3_ftok  # type: ignore
        import esm.tokenization.residue_tokenizer as esm3_rtok  # type: ignore
        from esm.models.function_decoder import FunctionTokenDecoder  # type: ignore
        from esm.models.vqvae import StructureTokenDecoder, StructureTokenEncoder  # type: ignore
        from esm.tokenization import get_esm3_model_tokenizers  # type: ignore
        from esm.utils.constants.models import ESM3_OPEN_SMALL  # type: ignore

        wdir = local_dir / "data/weights"
        # Redirect esm's internal data_root() to local_dir so tokenizers load 100% offline (zero HF download!)
        _local_data_root = lambda _model="esm3": local_dir  # noqa: E731
        esm3_const.data_root = _local_data_root
        if hasattr(esm3_ftok, "data_root"):
            esm3_ftok.data_root = _local_data_root
        if hasattr(esm3_rtok, "data_root"):
            esm3_rtok.data_root = _local_data_root
        # Also clear lru_cache / cache on any data_root reference if present
        import esm.pretrained as esm_pre  # type: ignore
        if hasattr(esm_pre, "data_root"):
            esm_pre.data_root = _local_data_root

        def _make_struct_enc(d: torch.device | str = "cpu"):
            enc = StructureTokenEncoder(d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096).eval()
            enc.load_state_dict(torch.load(str(wdir / "esm3_structure_encoder_v0.pth"), map_location=d))
            return enc.to(d)

        def _make_struct_dec(d: torch.device | str = "cpu"):
            dec = StructureTokenDecoder(d_model=1280, n_heads=20, n_layers=30).eval()
            dec.load_state_dict(torch.load(str(wdir / "esm3_structure_decoder_v0.pth"), map_location=d))
            return dec.to(d)

        def _make_func_dec(d: torch.device | str = "cpu"):
            fdec = FunctionTokenDecoder().eval()
            fdec.load_state_dict(torch.load(str(wdir / "esm3_function_decoder_v0.pth"), map_location=d))
            return fdec.to(d)

        with torch.device(dev):
            esm3_model = ESM3(
                d_model=1536,
                n_heads=24,
                v_heads=256,
                n_layers=48,
                structure_encoder_fn=_make_struct_enc,
                structure_decoder_fn=_make_struct_dec,
                function_decoder_fn=_make_func_dec,
                tokenizers=get_esm3_model_tokenizers(ESM3_OPEN_SMALL),
            ).eval()
        state_dict = torch.load(str(wdir / "esm3_sm_open_v1.pth"), map_location=dev)
        esm3_model.load_state_dict(state_dict)
    else:
        esm3_model = ESM3.from_pretrained(model_spec, device=dev).eval()

    # Respect requested dtype (bfloat16 on CUDA/CPU or float32)
    if dev.type == "cpu" and torch_dt == torch.float16:
        torch_dt = torch.float32
    esm3_model = esm3_model.to(device=dev, dtype=torch_dt).eval()
    esm3_model.requires_grad_(False)
    return ESM3OpenEvaluator(esm3_model, model_id=model_spec, device=device)


def load_open_audit_evaluator(
    model_spec: str,
    *,
    family: str = "auto",
    device: str = "auto",
    dtype: str = "float32",
    model_root: Path | str | None = None,
    progen2_bidirectional: bool = False,
) -> OpenModelAuditEvaluator:
    """Load any supported open-source PLM checkpoint (short name or direct Cuihu/local path)."""
    eff_dev = _resolve_device(device)
    dtype_name, torch_dt = _resolve_torch_dtype(dtype)
    fam = (family or "auto").strip().lower()

    # 1. Exact registered ProteinMPNN arms
    from src.capability.unified.proteinmpnn_adapter import PROTEINMPNN_SPECS, load_proteinmpnn_adapter

    if fam == "proteinmpnn" or model_spec in PROTEINMPNN_SPECS:
        arm_name = model_spec if model_spec in PROTEINMPNN_SPECS else "proteinmpnn-v-020"
        eff_root = model_spec if Path(model_spec).is_dir() else model_root
        adapter = load_proteinmpnn_adapter(arm_name, model_root=eff_root, device=eff_dev, dtype=dtype_name)
        return ProteinMPNNOpenEvaluator(adapter)

    # 2. Exact registered DPLM-2 arms
    from src.capability.unified.dplm2_adapter import DPLM2_SPECS, load_dplm2_adapter

    if fam == "dplm2" or model_spec in DPLM2_SPECS:
        if model_spec in DPLM2_SPECS:
            dplm2_arm = model_spec
            dplm2_root = model_root
        else:
            p_dir = _resolve_local_dir(model_spec, model_root=model_root)
            low_n = p_dir.name.lower().replace("_", "-")
            dplm2_arm = next((k for k in DPLM2_SPECS if k in low_n), "dplm2-150m")
            dplm2_root = p_dir.parent
        ad_seq = load_dplm2_adapter(dplm2_arm, mode="seq_only", model_root=dplm2_root, device=eff_dev, dtype=dtype_name)
        try:
            ad_struct = load_dplm2_adapter(
                dplm2_arm, mode="struct_conditioned", model_root=dplm2_root, device=eff_dev, dtype=dtype_name
            )
        except Exception:
            ad_struct = None
        return DPLM2OpenEvaluator(ad_seq, ad_struct)

    # 2B. ESM-3 multi-track (requires esm SDK runtime)
    if fam == "esm3" or "esm3" in model_spec.lower():
        return _load_esm3_evaluator(
            model_spec, device=eff_dev, dtype=dtype_name, model_root=model_root
        )

    # 3. Exact registered General-PLM registry arms (with Cuihu search roots included)
    from src.capability.general_plm.adapters import (
        ProGen3CausalAdapter,
        _load_amplify_mlm,
        _load_dayhoff_causal,
        _load_esm_mlm,
        _load_esmc_hf,
        _load_proteinglm_mlm,
        _load_protgpt3_causal,
        _load_saprot_mlm,
        load_general_plm,
    )
    from src.capability.general_plm.registry import GENERAL_PLM_REGISTRY, GeneralPLMSpec

    if model_spec in GENERAL_PLM_REGISTRY:
        spec = GENERAL_PLM_REGISTRY[model_spec]
        eff_dt = "bfloat16" if spec.family == "progen3" else dtype_name
        try:
            ad = load_general_plm(
                model_spec,
                device=eff_dev,
                dtype=eff_dt,
                model_root=model_root,
                saprot_mode="foldseek_3di",
            )
            return _wrap_general_plm_adapter(ad, model_id=model_spec, family_override=fam, dtype_name=eff_dt, device=eff_dev)
        except FileNotFoundError:
            # Fall through to Cuihu alias & subdir resolution below
            pass
        except RuntimeError:
            if spec.family != "esmc":
                raise
            # Fall through to _load_esmc_unified (SDK <-> HF converter) when transformers<5 or raw SDK weights are staged

    # 4. Resolve local / Cuihu directory & dispatch by architecture/family (handles direct paths & unlisted variants!)
    ckpt_dir = _resolve_local_dir(model_spec, model_root=model_root)
    detected = _detect_family_from_dir(ckpt_dir, model_spec) if fam == "auto" else {
        "ar": "causal_ar",
        "mlm": "esm_mlm",
        "mdm": "dplm_mdm",
        "saprot": "saprot",
        "prot_t5": "prot_t5",
    }.get(fam, _detect_family_from_dir(ckpt_dir, model_spec))

    if detected == "proteinmpnn":
        adapter = load_proteinmpnn_adapter("proteinmpnn-v-020", model_root=ckpt_dir, device=eff_dev, dtype=dtype_name)
        return ProteinMPNNOpenEvaluator(adapter)

    if detected == "dplm2":
        low_n = ckpt_dir.name.lower().replace("_", "-")
        dplm2_arm = next((k for k in DPLM2_SPECS if k in low_n), "dplm2-150m")
        ad_seq = load_dplm2_adapter(dplm2_arm, mode="seq_only", model_root=ckpt_dir.parent, device=eff_dev, dtype=dtype_name)
        try:
            ad_struct = load_dplm2_adapter(
                dplm2_arm, mode="struct_conditioned", model_root=ckpt_dir.parent, device=eff_dev, dtype=dtype_name
            )
        except Exception:
            ad_struct = None
        return DPLM2OpenEvaluator(ad_seq, ad_struct)

    if detected == "saprot":
        synth_spec = GeneralPLMSpec(
            arm=model_spec, family="saprot", paradigm="structure_aware_mlm", loader_kind="saprot_mlm",
            candidate_dirs=(ckpt_dir.name,), hf_repo=ckpt_dir.name, max_residues=1024,
        )
        ad = _load_saprot_mlm(synth_spec, ckpt_dir, device=eff_dev, dtype=dtype_name, saprot_mode="foldseek_3di")
        return SaProtOpenEvaluator(ad)

    if detected == "dplm_mdm":
        synth_spec = GeneralPLMSpec(
            arm=model_spec, family="dplm", paradigm="discrete_diffusion", loader_kind="esm_mlm",
            candidate_dirs=(ckpt_dir.name,), hf_repo=ckpt_dir.name, max_residues=1024,
        )
        ad = _load_esm_mlm(synth_spec, ckpt_dir, device=eff_dev, dtype=dtype_name)
        return _wrap_general_plm_adapter(ad, model_id=model_spec, family_override="mdm", dtype_name=dtype_name, device=eff_dev)

    if detected == "amplify_mlm":
        synth_spec = GeneralPLMSpec(
            arm=model_spec, family="amplify", paradigm="masked_lm", loader_kind="amplify_mlm",
            candidate_dirs=(ckpt_dir.name,), hf_repo=ckpt_dir.name, max_residues=2048, trust_remote_code=True,
        )
        ad = _load_amplify_mlm(synth_spec, ckpt_dir, device=eff_dev, dtype=dtype_name)
        return _wrap_general_plm_adapter(ad, model_id=model_spec, family_override=fam, dtype_name=dtype_name, device=eff_dev)

    if detected == "esmc_hf":
        return _load_esmc_unified(
            model_spec, ckpt_dir, device=eff_dev, dtype=dtype_name, family_override=fam
        )

    if detected == "proteinglm_mlm":
        synth_spec = GeneralPLMSpec(
            arm=model_spec, family="proteinglm_mlm", paradigm="masked_lm", loader_kind="proteinglm_mlm",
            candidate_dirs=(ckpt_dir.name,), hf_repo=ckpt_dir.name, max_residues=1024, trust_remote_code=True,
        )
        ad = _load_proteinglm_mlm(synth_spec, ckpt_dir, device=eff_dev, dtype=dtype_name)
        return _wrap_general_plm_adapter(ad, model_id=model_spec, family_override=fam, dtype_name=dtype_name, device=eff_dev)

    if detected == "progen3_causal":
        synth_spec = GeneralPLMSpec(
            arm=model_spec, family="progen3", paradigm="causal_lm", loader_kind="progen3_causal",
            candidate_dirs=(ckpt_dir.name,), hf_repo=ckpt_dir.name, default_dtype="bfloat16", max_residues=2048, allow_fp16=True,
        )
        ad = ProGen3CausalAdapter(synth_spec, ckpt_dir, device=eff_dev, dtype="bfloat16")
        return _wrap_general_plm_adapter(ad, model_id=model_spec, family_override="ar", dtype_name="bfloat16", device=eff_dev)

    if detected == "protgpt3_causal":
        fam_pg3 = "protgpt3_msa" if "msa" in ckpt_dir.name.lower() else "protgpt3"
        synth_spec = GeneralPLMSpec(
            arm=model_spec, family=fam_pg3, paradigm="causal_lm", loader_kind="protgpt3_causal",
            candidate_dirs=(ckpt_dir.name,), hf_repo=ckpt_dir.name, max_residues=1022,
        )
        ad = _load_protgpt3_causal(synth_spec, ckpt_dir, device=eff_dev, dtype=dtype_name)
        return _wrap_general_plm_adapter(ad, model_id=model_spec, family_override="ar", dtype_name=dtype_name, device=eff_dev)

    if detected == "dayhoff_causal":
        fam_dh = "dayhoff_msa" if ("msa" in ckpt_dir.name.lower() or "hm" in ckpt_dir.name.lower()) else "dayhoff"
        synth_spec = GeneralPLMSpec(
            arm=model_spec, family=fam_dh, paradigm="causal_lm", loader_kind="dayhoff_causal",
            candidate_dirs=(ckpt_dir.name,), hf_repo=ckpt_dir.name, max_residues=8192, trust_remote_code=True,
        )
        ad = _load_dayhoff_causal(synth_spec, ckpt_dir, device=eff_dev, dtype=dtype_name)
        return _wrap_general_plm_adapter(ad, model_id=model_spec, family_override="ar", dtype_name=dtype_name, device=eff_dev)

    if detected == "prot_t5":
        return _load_prot_t5(model_spec, ckpt_dir, device=eff_dev, dtype=dtype_name)

    if detected in ("galactica_causal", "causal_ar"):
        return _resolve_progen2_or_hf_causal(
            model_spec,
            ckpt_dir=ckpt_dir,
            device=eff_dev,
            dtype=dtype_name,
            model_root=model_root,
            progen2_bidirectional=progen2_bidirectional,
        )

    if detected == "esm_mlm":
        synth_spec = GeneralPLMSpec(
            arm=model_spec, family="esm2", paradigm="masked_lm", loader_kind="esm_mlm",
            candidate_dirs=(ckpt_dir.name,), hf_repo=ckpt_dir.name, max_residues=1024, trust_remote_code=True,
        )
        ad = _load_esm_mlm(synth_spec, ckpt_dir, device=eff_dev, dtype=dtype_name)
        return _wrap_general_plm_adapter(ad, model_id=model_spec, family_override=fam, dtype_name=dtype_name, device=eff_dev)

    # Generic local HF MaskedLM / MDM fallback with strict fail-fast AA20 validation
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        str(ckpt_dir), local_files_only=True, trust_remote_code=True, torch_dtype=torch_dt
    ).eval().to(eff_dev)
    model.requires_grad_(False)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    aa20_ids = _validate_aa20_ids([tokenizer.convert_tokens_to_ids(aa) for aa in AA20], unk_id, model_spec)
    if tokenizer.mask_token_id is None:
        raise ValueError(f"{model_spec}: generic MaskedLM tokenizer declares no mask_token_id")
    mask_id = int(tokenizer.mask_token_id)
    pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0
    cls_cls = MDMOpenEvaluator if fam == "mdm" else MaskedLMOpenEvaluator
    return cls_cls(
        model_id=model_spec,
        model=model,
        aa20_ids=aa20_ids,
        mask_id=mask_id,
        pad_id=pad_id,
        cls_id=tokenizer.cls_token_id,
        eos_id=tokenizer.eos_token_id,
        device=eff_dev,
        dtype=dtype_name,
        extra_metadata={"checkpoint_dir": str(ckpt_dir)},
    )


# --------------------------------------------------------------------------- #
# 6. End-to-End Audit Orchestrator
# --------------------------------------------------------------------------- #


def _slice_records(records: Sequence[Any], limit: int) -> list[Any]:
    if limit <= 0 or len(records) <= limit:
        return list(records)
    return list(records[:limit])


def run_audit_for_model(
    evaluator: OpenModelAuditEvaluator,
    bundle: OpenAuditBundle,
    *,
    batch_size: int = 16,
    n_mlm15_draws: int = DEFAULT_MLM15_DRAWS,
    n_loo_probes: int = DEFAULT_LOO_PROBES,
    n_mdm_strata: int = DEFAULT_MDM_STRATA,
    max_seqs_per_bucket: int = 0,
    max_seqs_per_split: int = 0,
    max_panel_per_group: int = 0,
    eval_seed: int = 20261129,
    seq_reference_bits: float | None = None,
) -> dict[str, Any]:
    """Execute the full quantitative physical exam for one model on ``bundle``."""
    t0 = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": AUDIT_REPORT_SCHEMA,
        "created_utc": _now_utc(),
        "bundle_dir": str(bundle.bundle_dir),
        "bundle_schema_version": bundle.manifest.get("schema_version"),
        "model": evaluator.metadata(),
        "eval_config": {
            "batch_size": int(batch_size),
            "n_mlm15_draws": int(n_mlm15_draws),
            "n_loo_probes": int(n_loo_probes),
            "n_mdm_strata": int(n_mdm_strata),
            "max_seqs_per_bucket": int(max_seqs_per_bucket),
            "max_seqs_per_split": int(max_seqs_per_split),
            "max_panel_per_group": int(max_panel_per_group),
            "eval_seed": int(eval_seed),
            "H0_bits_aa20": H0_BITS,
        },
    }

    if evaluator.supports_sequence_splits:
        # 1. Stratified depth buckets for Horvitz-Thompson corpus bits explained & apparent bits/param
        buckets_grouped = bundle.seen_by_bucket()
        bucket_eval_by_readout: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for b_name in DEPTH_BUCKET_ORDER:
            recs = _slice_records(buckets_grouped.get(b_name, []), max_seqs_per_bucket)
            if not recs:
                continue
            res_b = evaluator.evaluate_sequence_split(
                recs,
                batch_size=batch_size,
                n_mlm15_draws=n_mlm15_draws,
                n_loo_probes=n_loo_probes,
                n_mdm_strata=n_mdm_strata,
                eval_seed=eval_seed,
            )
            for readout_name, stats in res_b.items():
                bucket_eval_by_readout[readout_name][b_name] = stats

        # 2. Seen CATH-matched, unseen_homolog, unseen_fold (denovo + orphan)
        seen_cath_recs = _slice_records(bundle.seen_cath_matched, max_seqs_per_split)
        hom_recs = _slice_records(bundle.unseen_homolog, max_seqs_per_split)
        denovo_recs = _slice_records(
            [r for r in bundle.unseen_fold if r.sub_stratum == "denovo_zero_hit"], max_seqs_per_split
        )
        orphan_recs = _slice_records(
            [r for r in bundle.unseen_fold if r.sub_stratum == "orphan_cath_fold"], max_seqs_per_split
        )

        seen_cath_res = (
            evaluator.evaluate_sequence_split(
                seen_cath_recs,
                batch_size=batch_size,
                n_mlm15_draws=n_mlm15_draws,
                n_loo_probes=n_loo_probes,
                n_mdm_strata=n_mdm_strata,
                eval_seed=eval_seed + 1,
            )
            if seen_cath_recs
            else {}
        )
        hom_res = (
            evaluator.evaluate_sequence_split(
                hom_recs,
                batch_size=batch_size,
                n_mlm15_draws=n_mlm15_draws,
                n_loo_probes=n_loo_probes,
                n_mdm_strata=n_mdm_strata,
                eval_seed=eval_seed + 2,
            )
            if hom_recs
            else {}
        )
        denovo_res = (
            evaluator.evaluate_sequence_split(
                denovo_recs,
                batch_size=batch_size,
                n_mlm15_draws=n_mlm15_draws,
                n_loo_probes=n_loo_probes,
                n_mdm_strata=n_mdm_strata,
                eval_seed=eval_seed + 3,
            )
            if denovo_recs
            else {}
        )
        orphan_res = (
            evaluator.evaluate_sequence_split(
                orphan_recs,
                batch_size=batch_size,
                n_mlm15_draws=n_mlm15_draws,
                n_loo_probes=n_loo_probes,
                n_mdm_strata=n_mdm_strata,
                eval_seed=eval_seed + 4,
            )
            if orphan_recs
            else {}
        )

        # 3. Horvitz-Thompson corpus estimator & Eq. (3a)-(3b) three-tier decomposition per readout
        ht_all: dict[str, Any] = {}
        ht_elig: dict[str, Any] = {}
        three_tier_denovo: dict[str, Any] = {}
        three_tier_orphan: dict[str, Any] = {}

        for readout_name, b_map in bucket_eval_by_readout.items():
            ht_all[readout_name] = compute_horvitz_thompson_corpus_estimate(
                b_map,
                bundle.population_buckets_all,
                n_params_non_emb=evaluator.n_params_non_emb,
                n_params_total=evaluator.n_params_total,
            )
            ht_elig[readout_name] = compute_horvitz_thompson_corpus_estimate(
                b_map,
                bundle.population_buckets_eligible,
                n_params_non_emb=evaluator.n_params_non_emb,
                n_params_total=evaluator.n_params_total,
            )
            h_seen = (
                seen_cath_res[readout_name]["bits_per_res"]
                if readout_name in seen_cath_res
                else ht_all[readout_name]["corpus_weighted_H_bits_per_res"]
            )
            h_seen_cal = (
                seen_cath_res[readout_name].get("calibrated_bits_per_res")
                if readout_name in seen_cath_res
                else None
            )
            if readout_name in hom_res and readout_name in denovo_res:
                three_tier_denovo[readout_name] = compute_three_tier_decomposition(
                    h_seen,
                    hom_res[readout_name]["bits_per_res"],
                    denovo_res[readout_name]["bits_per_res"],
                    h_seen_cal=h_seen_cal,
                    h_unseen_homolog_cal=hom_res[readout_name].get("calibrated_bits_per_res"),
                    h_unseen_fold_cal=denovo_res[readout_name].get("calibrated_bits_per_res"),
                )
            if readout_name in hom_res and readout_name in orphan_res:
                three_tier_orphan[readout_name] = compute_three_tier_decomposition(
                    h_seen,
                    hom_res[readout_name]["bits_per_res"],
                    orphan_res[readout_name]["bits_per_res"],
                    h_seen_cal=h_seen_cal,
                    h_unseen_homolog_cal=hom_res[readout_name].get("calibrated_bits_per_res"),
                    h_unseen_fold_cal=orphan_res[readout_name].get("calibrated_bits_per_res"),
                )

        report["sequence_splits"] = {
            "seen_pretrain_buckets": dict(bucket_eval_by_readout),
            "seen_cath_matched": seen_cath_res,
            "unseen_homolog": hom_res,
            "unseen_fold_denovo_zero_hit": denovo_res,
            "unseen_fold_orphan_cath": orphan_res,
        }
        report["horvitz_thompson_all_uniref50"] = ht_all
        report["horvitz_thompson_eligible_uniref50"] = ht_elig
        report["three_tier_eq3a_denovo_unseen_fold"] = three_tier_denovo
        report["three_tier_eq3a_orphan_unseen_fold"] = three_tier_orphan

    if evaluator.supports_3d_panel and bundle.paired_3d_panel:
        by_grp: dict[str, list[Paired3DRecord]] = defaultdict(list)
        for rec in bundle.paired_3d_panel:
            by_grp[rec.domain_group].append(rec)
        panel_selected: list[Paired3DRecord] = []
        for g in sorted(by_grp.keys()):
            panel_selected.extend(_slice_records(by_grp[g], max_panel_per_group))
        coords_map = {r.coords_key: bundle.load_coords(r.coords_key) for r in panel_selected}
        report["paired_3d_panel"] = evaluator.evaluate_3d_panel(
            panel_selected,
            coords_map,
            batch_size=batch_size,
            n_mlm15_draws=n_mlm15_draws,
            n_loo_probes=n_loo_probes,
            n_mdm_strata=n_mdm_strata,
            eval_seed=eval_seed + 5,
            seq_reference_bits=seq_reference_bits,
        )

    report["wall_seconds"] = round(time.perf_counter() - t0, 3)
    return report
