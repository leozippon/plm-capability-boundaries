#!/usr/bin/env python3
"""GPU runner for Unified Result 5 Downstream-1: Homologous-Prefix DMS Mutation Ranking.

Scores BOTH Subtest 1A (``rescue_30_80``, 217 assays x 128 variants) AND Subtest 1B
(``e09_five_band``, 201 assays x 128 variants across 5 identity bands + self_copy +
unrelated + no_context) across ALL cross-paradigm model families:

  - Bidirectional / Masked / 3Di / Diffusion / ESM3 (22 arms) & MSA-Transformer-100M
    (aligned=True & aligned=False): exact site-batched Masked-Marginal
    (``sum [log P(mut | ctx, WT_{\\p}) - log P(wt | ctx, WT_{\\p})]``).
  - Causal Protein / Joint / Text & Multi-Seq ICL (ProtGPT3-MSA, Dayhoff-3B-MSA):
    exact zero-padding same-length batched NLL with automatic assay-0 batch-vs-singleton
    parity check (``<1e-3`` nats on mutant deltas, auto-falling back to singleton if needed).
  - ProGen3 MoE (112M..3B): probes KV-prefix caching vs full singleton on assay 0
    (``<1e-4`` nats), using fast KV-prefix when exact and singleton otherwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("TRANSFER_DATA_ROOT") or ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.core.precision_policy import fp32_matmul_context  # noqa: E402
from src.capability.general_plm.adapters import parse_mutation  # noqa: E402
from src.capability.general_plm.eval_utils import atomic_write_json, sha256_file  # noqa: E402
from src.capability.models.fitness import parse_mutant  # noqa: E402
from src.capability.unified.r4_scoring_yardstick import (  # noqa: E402
    pack_bidirectional_context,
    pack_causal_text_context,
)
from src.capability.unified.r5_dms_context_yardstick import (  # noqa: E402
    CACHED_1A_PATHS,
    CACHED_1B_PATHS,
    DMS_CONTEXT_CORE_KEYS,
    DMS_CONTEXT_SPEC_BY_KEY,
    DMS_CONTEXT_SPECS_ALL,
    DMS_SCORES_DIR_REL,
    E09_BINS_ORDERED,
    E09_COHORT_REL,
    FORBIDDEN_INFRA_RE,
    RESCUE_COHORT_REL,
    RESCUE_CONDITIONS,
    SCHEMA_DMS_ARM_SCORE,
    DmsContextSpec,
    assign_balanced_shards,
    fast_build_query_msa,
    plan_e09_target,
    plan_rescue_30_80_assay,
)

AA20_INDEX = {aa: i for i, aa in enumerate(AA20)}


def _rho(x: Sequence[float], y: Sequence[float]) -> float | None:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size < 3 or np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
        return None
    v = float(spearmanr(xa, ya).statistic)
    return v if np.isfinite(v) else None


def _rebuild_variants(wildtype: str, mutants: Sequence[str]) -> list[str]:
    out: list[str] = []
    for mut in mutants:
        chars = list(wildtype)
        for wt_aa, pos, mut_aa in parse_mutant(mut):
            if chars[pos - 1] != wt_aa:
                raise ValueError(f"{mut}: expected {wt_aa} at {pos}, found {chars[pos - 1]}")
            chars[pos - 1] = mut_aa
        out.append("".join(chars))
    return out


def _precision_ctx(device: str, dtype_name: str):
    if dtype_name == "float32" and str(device).startswith("cuda"):
        return fp32_matmul_context(torch)
    return nullcontext()


def _resolve_dtype(spec: DmsContextSpec, dtype_arg: str) -> tuple[str, torch.dtype]:
    from src.capability.unified.roster import ARM_METADATA

    raw = dtype_arg.strip().lower()
    if raw in ("", "auto", "default"):
        # Per #184: >= 1B arms, ProGen3 MoE, ESM3-1.4B, and 2D MSA-Transformer default to bfloat16
        # (with float32 log_softmax & NLL accumulation; rita-xl kept in float32 for custom conv stability).
        params_m = ARM_METADATA[spec.arm_id].params_m if spec.arm_id in ARM_METADATA else 0.0
        if spec.arm_id == "rita-xl":
            return "float32", torch.float32
        if (
            spec.scorer_kind in ("progen3_causal", "esm3_mlm", "msa_transformer")
            or params_m >= 900.0
        ):
            return "bfloat16", torch.bfloat16
        return "float32", torch.float32
    if raw in ("fp32", "float32"):
        if spec.scorer_kind == "progen3_causal":
            return "bfloat16", torch.bfloat16
        return "float32", torch.float32
    if raw in ("bf16", "bfloat16"):
        return "bfloat16", torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_arg!r}")


def _default_batch_size(spec: DmsContextSpec) -> int:
    if spec.scorer_kind in ("msa_transformer", "progen3_causal"):
        return 16
    if spec.arm_id in ("galactica-30b", "qwen2.5-32b", "esm2-15b", "proteinglm-10b-mlm"):
        return 16
    if spec.arm_id in (
        "protgpt3-10b",
        "proteinglm-7b-clm",
        "llama-2-7b",
        "qwen2.5-7b",
        "qwen3-8b-base",
        "galactica-6.7b",
        "prollama",
        "prollama-stage-1",
        "esmc-6b",
        "dayhoff-3b-msa",
    ):
        return 32
    return 64


# ============================================================================
# Unified Model Scorer Abstraction
# ============================================================================


class _BaseDmsScorer:
    """Common interface for scoring one (wildtype, mutants, context_items) DMS condition."""

    def __init__(self, spec: DmsContextSpec, *, device: str, dtype_name: str, batch_size: int) -> None:
        self.spec = spec
        self.device = device
        self.dtype_name = dtype_name
        self.batch_size = max(1, int(batch_size))
        self.parity_checks: dict[str, Any] = {}

    def token_length(self, seq: str) -> int:
        raise NotImplementedError

    def overhead_tokens(self) -> int:
        return 0

    def score_condition(
        self,
        wildtype: str,
        mutants: Sequence[str],
        variant_seqs: Sequence[str],
        context_items: Sequence[str],
        *,
        budget_tokens: int,
        check_parity: bool = False,
    ) -> tuple[list[float], float, int]:
        """Return ``(variant_scores_higher_better, wt_log_likelihood_proxy, wt_scored_tokens)``."""
        raise NotImplementedError

    def provenance(self) -> dict[str, Any]:
        return {
            "entry_key": self.spec.entry_key,
            "arm_id": self.spec.arm_id,
            "scorer_kind": self.spec.scorer_kind,
            "input_alignment_format": self.spec.input_alignment_format,
            "context_training_regime": self.spec.context_training_regime,
            "budget_tokens": self.spec.budget_tokens,
            "aligned_msa": self.spec.aligned_msa,
            "dtype_requested": self.dtype_name,
            "batch_size": self.batch_size,
            "parity_checks": self.parity_checks,
        }


class _BidirectionalDmsScorer(_BaseDmsScorer):
    """Exact site-batched Masked-Marginal scorer for all 22 bidirectional/masked/3Di/diffusion/ESM3 arms."""

    def __init__(
        self,
        spec: DmsContextSpec,
        *,
        device: str,
        dtype_name: str,
        batch_size: int,
        model_root: Path | None,
    ) -> None:
        super().__init__(spec, device=device, dtype_name=dtype_name, batch_size=batch_size)
        self.is_esm3 = spec.scorer_kind == "esm3_mlm"
        self.is_saprot = False
        if self.is_esm3:
            from src.capability.esm3_ext.esm3_adapter import load_esm3_offline

            esm3_dir = (
                (model_root / "esm3-sm-open-v1")
                if model_root and (model_root / "esm3-sm-open-v1").is_dir()
                else None
            )
            self.esm3 = load_esm3_offline(
                model_dir=esm3_dir, device=device, dtype=dtype_name, eager_load_aux=False
            )
            self.bos_prefix = [int(self.esm3.bos_id)]
            self.eos_suffix = [int(self.esm3.eos_id)]
            self.sep_ids = [int(self.esm3.eos_id)]
            self.mask_id = int(self.esm3.mask_id)
            self.aa_to_id = dict(self.esm3.aa_to_id)
            self.adapter = None
            self.realised_dtype = str(self.esm3.dtype)
        else:
            from src.capability.general_plm import load_general_plm
            from src.capability.general_plm.adapters import SaProtAdapter

            self.esm3 = None
            self.adapter = load_general_plm(
                spec.arm_id,
                device=device,
                dtype=dtype_name,
                model_root=model_root,
                saprot_mode="seq_only",
            )
            self.is_saprot = isinstance(self.adapter, SaProtAdapter)
            probe_ids, offset = self.adapter._encode_sequence("ACDE")
            self.bos_prefix = list(probe_ids[:offset])
            self.eos_suffix = list(probe_ids[offset + 4 :])
            if self.eos_suffix:
                self.sep_ids = list(self.eos_suffix)
            elif getattr(self.adapter.tokenizer, "eos_token_id", None) is not None:
                self.sep_ids = [int(self.adapter.tokenizer.eos_token_id)]
            else:
                self.sep_ids = list(self.bos_prefix)
            self.mask_id = int(self.adapter.mask_id)
            self.aa_to_id = dict(self.adapter.aa_to_id)
            self.realised_dtype = self.adapter.provenance().get("dtype_realised", dtype_name)

    def token_length(self, seq: str) -> int:
        return len(seq) + len(self.sep_ids)

    def overhead_tokens(self) -> int:
        return len(self.bos_prefix) + len(self.eos_suffix)

    @torch.no_grad()
    def _forward_logits(self, batch_ids: torch.Tensor) -> torch.Tensor:
        if self.is_esm3:
            from src.capability.esm3_ext.esm3_adapter import forward_tracks

            return forward_tracks(self.esm3, sequence_tokens=batch_ids).sequence_logits.float()
        assert self.adapter is not None
        with self.adapter._precision_ctx():
            return self.adapter._forward_logits(batch_ids)

    def _aa20_logprobs_at(self, logits_vec: torch.Tensor) -> np.ndarray:
        """Return length-20 AA20 log-probabilities (marginalizing 3Di for SaProt)."""
        full_lp = F.log_softmax(logits_vec, dim=-1)
        if self.is_saprot:
            assert self.adapter is not None
            struct_mat = self.adapter._struct_id_matrix  # (20, 20)
            gathered = full_lp.index_select(0, struct_mat.reshape(-1)).reshape(20, 20)
            return torch.logsumexp(gathered, dim=-1).detach().cpu().numpy().astype(np.float64)
        ids_t = torch.as_tensor([self.aa_to_id[aa] for aa in AA20], dtype=torch.long, device=logits_vec.device)
        return full_lp.index_select(0, ids_t).detach().cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def score_condition(
        self,
        wildtype: str,
        mutants: Sequence[str],
        variant_seqs: Sequence[str],
        context_items: Sequence[str],
        *,
        budget_tokens: int,
        check_parity: bool = False,
    ) -> tuple[list[float], float, int]:
        tgt_ids = [self.aa_to_id[aa] for aa in wildtype]
        ctx_ids = [[self.aa_to_id[aa] for aa in s] for s in context_items]
        full_ids, target_start, _target_end, _k = pack_bidirectional_context(
            tgt_ids,
            ctx_ids,
            bos_prefix=self.bos_prefix,
            eos_suffix=self.eos_suffix,
            sep_ids=self.sep_ids,
            budget_tokens=budget_tokens,
        )
        parsed = [parse_mutation(m, wildtype) for m in mutants]
        unique_pos = sorted({pos for entry in parsed for pos, _, _ in entry})

        base = torch.as_tensor(full_ids, dtype=torch.long, device=self.device)
        site_lp: dict[int, np.ndarray] = {}
        for start in range(0, len(unique_pos), self.batch_size):
            chunk = unique_pos[start : start + self.batch_size]
            batch = base.unsqueeze(0).repeat(len(chunk), 1)
            for row_i, pos in enumerate(chunk):
                batch[row_i, target_start + pos] = self.mask_id
            logits = self._forward_logits(batch)
            for row_i, pos in enumerate(chunk):
                site_lp[pos] = self._aa20_logprobs_at(logits[row_i, target_start + pos])
            del logits

        scores: list[float] = []
        for entry in parsed:
            tot = 0.0
            for pos, wt_aa, mut_aa in entry:
                lp = site_lp[pos]
                tot += float(lp[AA20_INDEX[mut_aa]] - lp[AA20_INDEX[wt_aa]])
            scores.append(tot)
        wt_ll_proxy = float(sum(site_lp[p][AA20_INDEX[wildtype[p]]] for p in unique_pos))
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"{self.spec.entry_key}: non-finite masked-marginal score")
        return scores, wt_ll_proxy, max(1, len(unique_pos))


class _MsaTransformerDmsScorer(_BaseDmsScorer):
    """Exact site-batched Masked-Marginal scorer for MSA-Transformer-100M (aligned=True / False)."""

    def __init__(
        self,
        spec: DmsContextSpec,
        *,
        device: str,
        dtype_name: str,
        batch_size: int,
        model_root: Path | None,
    ) -> None:
        super().__init__(spec, device=device, dtype_name=dtype_name, batch_size=batch_size)
        from src.capability.general_plm import load_general_plm

        self.adapter = load_general_plm(
            "msa-transformer-100m", device=device, dtype=dtype_name, model_root=model_root
        )
        self.aligned_msa = bool(spec.aligned_msa)
        self.realised_dtype = self.adapter.provenance().get("dtype_realised", dtype_name)

    def token_length(self, seq: str) -> int:
        return len(seq)

    @torch.no_grad()
    def score_condition(
        self,
        wildtype: str,
        mutants: Sequence[str],
        variant_seqs: Sequence[str],
        context_items: Sequence[str],
        *,
        budget_tokens: int,
        check_parity: bool = False,
    ) -> tuple[list[float], float, int]:
        msa_rows = fast_build_query_msa(wildtype, context_items, max_depth=16, aligned=self.aligned_msa)
        base_tokens = self.adapter._encode_msa(msa_rows)  # (1, D, L+1)
        parsed = [parse_mutation(m, wildtype) for m in mutants]
        unique_pos = sorted({pos for entry in parsed for pos, _, _ in entry})

        aa20_ids = torch.as_tensor(
            [self.adapter.aa_to_id[aa] for aa in AA20], dtype=torch.long, device=self.device
        )
        # Bound total 2D tokens (B * D * L) per forward pass to prevent FFN OOM on long/deep MSAs
        msa_depth = int(base_tokens.shape[1])
        msa_cols = int(base_tokens.shape[2])
        token_cap = 65536 if self.dtype_name == "bfloat16" else 32768
        eff_bs = max(1, min(self.batch_size, token_cap // max(1, msa_depth * msa_cols)))
        site_lp: dict[int, np.ndarray] = {}
        with self.adapter._precision_ctx():
            for start in range(0, len(unique_pos), eff_bs):
                chunk = unique_pos[start : start + eff_bs]
                batch = base_tokens.repeat(len(chunk), 1, 1)
                for row_i, pos in enumerate(chunk):
                    batch[row_i, 0, pos + 1] = self.adapter.mask_id
                logits = self.adapter._forward_msa_logits(batch)
                for row_i, pos in enumerate(chunk):
                    full_lp = F.log_softmax(logits[row_i, 0, pos + 1], dim=-1)
                    site_lp[pos] = full_lp.index_select(0, aa20_ids).detach().cpu().numpy().astype(np.float64)
                del logits

        scores: list[float] = []
        for entry in parsed:
            tot = 0.0
            for pos, wt_aa, mut_aa in entry:
                lp = site_lp[pos]
                tot += float(lp[AA20_INDEX[mut_aa]] - lp[AA20_INDEX[wt_aa]])
            scores.append(tot)
        wt_ll_proxy = float(sum(site_lp[p][AA20_INDEX[wildtype[p]]] for p in unique_pos))
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"{self.spec.entry_key}: non-finite MSA masked-marginal score")
        return scores, wt_ll_proxy, max(1, len(unique_pos))


class _GpCausalDmsScorer(_BaseDmsScorer):
    """Exact same-length zero-padding batched scorer for ProtGPT3, ProtGPT3-MSA, Dayhoff-170M, Dayhoff-3B-MSA."""

    def __init__(
        self,
        spec: DmsContextSpec,
        *,
        device: str,
        dtype_name: str,
        batch_size: int,
        model_root: Path | None,
    ) -> None:
        super().__init__(spec, device=device, dtype_name=dtype_name, batch_size=batch_size)
        from src.capability.general_plm import load_general_plm

        self.adapter = load_general_plm(
            spec.arm_id, device=device, dtype=dtype_name, model_root=model_root
        )
        self.bos_prefix = list(self.adapter.bos_prefix_ids)
        self.sep_ids = list(self.adapter.separator_ids)
        self.aa_to_id = dict(self.adapter.aa_to_id)
        self.force_singleton = False
        self.is_sparse_moe = int(getattr(getattr(self.adapter.model, "config", None), "num_experts", 1) or 1) > 1
        self.realised_dtype = self.adapter.provenance().get("dtype_realised", dtype_name)

    def token_length(self, seq: str) -> int:
        return len(seq) + len(self.sep_ids)

    def overhead_tokens(self) -> int:
        return len(self.bos_prefix)

    @torch.no_grad()
    def _nll_batch_same_len(self, rows: Sequence[list[int]], span_start: int, span_end: int) -> list[float]:
        inp = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        with self.adapter._precision_ctx():
            out = self.adapter.model(input_ids=inp, use_cache=False)
            logits = out.logits[:, span_start - 1 : span_end - 1, :].float()
            targets = inp[:, span_start:span_end]
            nlls = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
            ).reshape(len(rows), -1).sum(dim=1)
        return (-nlls).detach().cpu().tolist()

    @torch.no_grad()
    def score_condition(
        self,
        wildtype: str,
        mutants: Sequence[str],
        variant_seqs: Sequence[str],
        context_items: Sequence[str],
        *,
        budget_tokens: int,
        check_parity: bool = False,
    ) -> tuple[list[float], float, int]:
        prefix = list(self.bos_prefix)
        for item in context_items:
            prefix.extend(self.aa_to_id[aa] for aa in item)
            prefix.extend(self.sep_ids)
        span_start = len(prefix)
        span_end = span_start + len(wildtype)
        all_seqs = [wildtype, *variant_seqs]
        packed = [prefix + [self.aa_to_id[aa] for aa in s] for s in all_seqs]

        seq_len = len(packed[0]) if packed else 1
        eff_bs = 1 if self.force_singleton else max(1, min(self.batch_size, 32768 // max(1, seq_len)))
        lls: list[float] = []
        for start in range(0, len(packed), eff_bs):
            lls.extend(self._nll_batch_same_len(packed[start : start + eff_bs], span_start, span_end))

        if check_parity and eff_bs > 1 and len(packed) >= 3:
            singles = [self._nll_batch_same_len([packed[j]], span_start, span_end)[0] for j in range(3)]
            deltas = [abs((lls[j] - lls[0]) - (singles[j] - singles[0])) for j in (1, 2)]
            max_d = max(deltas)
            max_d_per_tok = max_d / max(1, len(wildtype))
            chk: dict[str, Any] = {
                "max_mutant_delta_diff": max_d,
                "max_mutant_delta_diff_per_tok": max_d_per_tok,
                "batch_size": eff_bs,
            }
            if self.is_sparse_moe and self.dtype_name == "bfloat16":
                # BF16 SparseMoE (e.g. Dayhoff/Jamba 16-expert top-2) flips discrete router ties across
                # GEMM sizes (even on identical-row copies) despite zero cross-row leakage (FP32 diff ~3e-6).
                # Verify row symmetry on [row0, row0] and guard only against gross in-distribution divergence.
                dup = self._nll_batch_same_len([packed[0], packed[0]], span_start, span_end)
                sym_diff = abs(dup[0] - dup[1])
                chk["sparse_moe_row_symmetry_diff"] = sym_diff
                self.parity_checks[f"ctx_{len(context_items)}"] = chk
                if sym_diff > 1e-3 or (len(context_items) == 0 and max_d_per_tok > 5e-2):
                    self.force_singleton = True
                    lls = [self._nll_batch_same_len([row], span_start, span_end)[0] for row in packed]
            else:
                self.parity_checks[f"ctx_{len(context_items)}"] = chk
                if max_d_per_tok > 5e-3:
                    self.force_singleton = True
                    lls = [self._nll_batch_same_len([row], span_start, span_end)[0] for row in packed]

        wt_ll = float(lls[0])
        scores = [float(v - wt_ll) for v in lls[1:]]
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"{self.spec.entry_key}: non-finite GP causal score")
        return scores, wt_ll, len(wildtype)


class _ProGen3DmsScorer(_BaseDmsScorer):
    """Exact zero-padding same-length batched scorer for ProGen3 MoE rungs (progen3-112m..3b)."""

    def __init__(
        self,
        spec: DmsContextSpec,
        *,
        device: str,
        dtype_name: str,
        batch_size: int,
        model_root: Path | None,
    ) -> None:
        super().__init__(spec, device=device, dtype_name=dtype_name, batch_size=batch_size)
        self.force_singleton = False
        from src.capability.models import progen3 as pg3

        if spec.arm_id in ("progen3-112m", "progen3-3b"):
            # Pre-registered ProGen3 rungs live in core.arms / model_root rather than general_plm.registry
            from src.capability.core import arms as core_arms

            ckpt_dir: Path | None = None
            candidates: list[Path] = []
            if model_root is not None:
                candidates.extend(
                    [model_root / spec.arm_id, model_root / spec.arm_id.replace("-", "_")]
                )
            for env_var in ("MODEL_ROOT", "MODEL_BASE_DIR", "GENERAL_PLM_MODEL_ROOT"):
                ev = os.environ.get(env_var)
                if ev:
                    candidates.extend(
                        [Path(ev) / spec.arm_id, Path(ev) / spec.arm_id.replace("-", "_")]
                    )
            if spec.arm_id == "progen3-112m" and os.environ.get("PROGEN3_DIR"):
                candidates.append(Path(os.environ["PROGEN3_DIR"]))
            all_specs = {**core_arms.STAGED_ARMS, **core_arms.PANEL}
            if spec.arm_id in all_specs:
                candidates.append(Path(all_specs[spec.arm_id].path))
            for cand in candidates:
                if cand.is_dir():
                    ckpt_dir = cand
                    break
            if ckpt_dir is None:
                raise FileNotFoundError(f"{spec.arm_id}: checkpoint dir not found in {candidates}")
            self.adapter = None
            self.handle = pg3.load_progen3(ckpt_dir, device=device, dtype=torch.bfloat16)
            self.realised_dtype = "torch.bfloat16"
        else:
            from src.capability.general_plm import load_general_plm

            self.adapter = load_general_plm(
                spec.arm_id, device=device, dtype=dtype_name, model_root=model_root
            )
            self.handle = self.adapter.handle
            self.realised_dtype = self.adapter.provenance().get("dtype_realised", dtype_name)

    def token_length(self, seq: str) -> int:
        # ProGen3 brackets each sequence with <bos>, '1', seq, '2', <eos> -> len(seq) + 4
        return len(seq) + 4

    def _encode_item(self, seq: str) -> list[int]:
        return self.handle.batch([seq], reverse=False)["input_ids"][0].tolist()

    @torch.no_grad()
    def _score_batch_same_len(
        self, rows: Sequence[list[int]], span_start: int, span_end: int
    ) -> list[float]:
        inp = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        bsz, seq_len = inp.shape
        pos = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(bsz, -1)
        seq_ids = torch.zeros_like(inp)
        out = self.handle.model(
            input_ids=inp, position_ids=pos, sequence_ids=seq_ids, use_cache=False
        )
        logits = out.logits[:, span_start - 1 : span_end - 1, :].float()
        targets = inp[:, span_start:span_end]
        nlls = (
            F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
            )
            .reshape(bsz, -1)
            .sum(dim=1)
        )
        return (-nlls).detach().cpu().tolist()

    @torch.no_grad()
    def score_condition(
        self,
        wildtype: str,
        mutants: Sequence[str],
        variant_seqs: Sequence[str],
        context_items: Sequence[str],
        *,
        budget_tokens: int,
        check_parity: bool = False,
    ) -> tuple[list[float], float, int]:
        prefix: list[int] = []
        for item in context_items:
            prefix.extend(self._encode_item(item))
        span_start = len(prefix) + 2
        span_end = span_start + len(wildtype)
        all_seqs = [wildtype, *variant_seqs]
        packed = [prefix + self._encode_item(s) for s in all_seqs]

        by_len: dict[int, list[int]] = {}
        for idx, row in enumerate(packed):
            by_len.setdefault(len(row), []).append(idx)

        lls = [0.0] * len(packed)
        for length, idxs in by_len.items():
            eff_bs = 1 if self.force_singleton else max(1, min(self.batch_size, 16384 // max(1, length)))
            for start in range(0, len(idxs), eff_bs):
                chunk_idx = idxs[start : start + eff_bs]
                vals = self._score_batch_same_len(
                    [packed[i] for i in chunk_idx], span_start, span_end
                )
                for i, v in zip(chunk_idx, vals):
                    lls[i] = v

        eff_bs0 = (
            1
            if self.force_singleton
            else max(1, min(self.batch_size, 16384 // max(1, len(packed[0]) if packed else 1)))
        )
        if check_parity and eff_bs0 > 1 and len(packed) >= 3:
            n_chk = min(6, len(packed))
            singles = [self._score_batch_same_len([packed[j]], span_start, span_end)[0] for j in range(n_chk)]
            deltas = [abs((lls[j] - lls[0]) - (singles[j] - singles[0])) for j in range(1, n_chk)]
            max_d = max(deltas)
            med_d = float(np.median(deltas))
            max_d_per_tok = max_d / max(1, len(wildtype))
            med_d_per_tok = med_d / max(1, len(wildtype))
            dup = self._score_batch_same_len([packed[0], packed[0]], span_start, span_end)
            sym_diff = abs(dup[0] - dup[1])
            self.parity_checks[f"ctx_{len(context_items)}"] = {
                "max_mutant_delta_diff": max_d,
                "max_mutant_delta_diff_per_tok": max_d_per_tok,
                "median_mutant_delta_diff_per_tok": med_d_per_tok,
                "sparse_moe_row_symmetry_diff": sym_diff,
                "batch_size": eff_bs0,
            }
            # SparseMoE in bfloat16 can flip top-2 router ties across GEMM sizes (even KV-cache bs=1 vs
            # full bs=1); gate fallback on row symmetry + in-distribution (no_context) median parity.
            if sym_diff > 1e-3 or (len(context_items) == 0 and med_d_per_tok > 5e-2):
                self.force_singleton = True
                lls = [self._score_batch_same_len([row], span_start, span_end)[0] for row in packed]

        wt_ll = float(lls[0])
        scores = [float(v - wt_ll) for v in lls[1:]]
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"{self.spec.entry_key}: non-finite ProGen3 score")
        return scores, wt_ll, len(wildtype)


class _PreregisteredCausalDmsScorer(_BaseDmsScorer):
    """Exact native-packed scorer for PREREGISTERED_33_ARMS causal protein/joint models via context_homologue."""

    def __init__(
        self,
        spec: DmsContextSpec,
        *,
        device: str,
        dtype_name: str,
        batch_size: int,
    ) -> None:
        super().__init__(spec, device=device, dtype_name=dtype_name, batch_size=batch_size)
        from scripts.capability.entrypoints import stage_path
        from src.capability.context import context_homologue as ch

        mod_spec = importlib.util.spec_from_file_location(
            "context_homologue_stage", stage_path(ROOT, "context_homologue.py")
        )
        assert mod_spec is not None and mod_spec.loader is not None
        stage = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(stage)
        self.ch = ch
        self.stage = stage
        self.arm = stage.load_scorable_arm(spec.arm_id, device=device, dtype=dtype_name)
        self.row_prefix = list(ch.row_prefix_ids(self.arm))
        self.force_singleton = spec.arm_id in ("rita-xl",)
        self.realised_dtype = dtype_name

    def token_length(self, seq: str) -> int:
        return len(self.ch.item_ids(self.arm, seq, modality="protein"))

    def overhead_tokens(self) -> int:
        return len(self.row_prefix)

    @torch.no_grad()
    def _score_same_len_batch(
        self, batch_rows: Sequence[list[int]], spans: Sequence[tuple[int, int]], prefix_len: int
    ) -> list[float]:
        with _precision_ctx(self.device, self.dtype_name):
            logits, ids = self.stage._forward_rows(self.arm, list(batch_rows))
            out: list[float] = []
            for j in range(len(batch_rows)):
                left, right = spans[j]
                res = self.stage._target_nll(
                    logits[j : j + 1], ids[j : j + 1], prefix_len + left, prefix_len + right
                )
                out.append(float(-res["nll_sum"]))
            del logits, ids
        return out

    @torch.no_grad()
    def score_condition(
        self,
        wildtype: str,
        mutants: Sequence[str],
        variant_seqs: Sequence[str],
        context_items: Sequence[str],
        *,
        budget_tokens: int,
        check_parity: bool = False,
    ) -> tuple[list[float], float, int]:
        ctx_ids: list[int] = []
        for item in context_items:
            ctx_ids.extend(self.ch.item_ids(self.arm, item, modality="protein"))
        prefix = self.row_prefix + ctx_ids
        prefix_len = len(prefix)

        all_seqs = [wildtype, *variant_seqs]
        t_ids = [self.ch.item_ids(self.arm, s, modality="protein") for s in all_seqs]
        spans = [self.ch.target_span(self.arm, ids, record=s) for ids, s in zip(t_ids, all_seqs)]
        packed = [prefix + ids for ids in t_ids]

        # Group indices by exact token length -> ZERO padding in every forward pass!
        by_len: dict[int, list[int]] = {}
        for idx, row in enumerate(packed):
            by_len.setdefault(len(row), []).append(idx)

        lls = [0.0] * len(packed)
        for length, idxs in by_len.items():
            eff_bs = 1 if self.force_singleton else max(1, min(self.batch_size, 32768 // max(1, length)))
            for start in range(0, len(idxs), eff_bs):
                chunk_idx = idxs[start : start + eff_bs]
                vals = self._score_same_len_batch(
                    [packed[i] for i in chunk_idx], [spans[i] for i in chunk_idx], prefix_len
                )
                for i, v in zip(chunk_idx, vals):
                    lls[i] = v

        eff_bs0 = (
            1
            if self.force_singleton
            else max(1, min(self.batch_size, 32768 // max(1, len(packed[0]) if packed else 1)))
        )
        if check_parity and eff_bs0 > 1 and len(packed) >= 3:
            singles = [
                self._score_same_len_batch([packed[j]], [spans[j]], prefix_len)[0] for j in range(3)
            ]
            deltas = [abs((lls[j] - lls[0]) - (singles[j] - singles[0])) for j in (1, 2)]
            max_d = max(deltas)
            target_len = max(1, int(spans[0][1] - spans[0][0]))
            max_d_per_tok = max_d / target_len
            self.parity_checks[f"ctx_{len(context_items)}"] = {
                "max_mutant_delta_diff": max_d,
                "max_mutant_delta_diff_per_tok": max_d_per_tok,
                "batch_size": eff_bs0,
            }
            if max_d_per_tok > 5e-3:
                self.force_singleton = True
                lls = [
                    self._score_same_len_batch([packed[i]], [spans[i]], prefix_len)[0]
                    for i in range(len(packed))
                ]

        wt_ll = float(lls[0])
        scores = [float(v - wt_ll) for v in lls[1:]]
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"{self.spec.entry_key}: non-finite preregistered causal score")
        return scores, wt_ll, int(spans[0][1] - spans[0][0])


class _CausalTextDmsScorer(_BaseDmsScorer):
    """Exact same-length zero-padding batched scorer for Group A causal text models."""

    def __init__(
        self,
        spec: DmsContextSpec,
        *,
        device: str,
        dtype_name: str,
        torch_dt: torch.dtype,
        batch_size: int,
        model_root: Path | None,
    ) -> None:
        super().__init__(spec, device=device, dtype_name=dtype_name, batch_size=batch_size)
        from scripts.capability.unified.run_r4_unaligned_scoring_h200 import (
            CAUSAL_TEXT_PACKING,
            _resolve_text_checkpoint_dir,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ckpt = _resolve_text_checkpoint_dir(spec.arm_id, model_root)
        self.mode, self.bos_id, self.eos_id = CAUSAL_TEXT_PACKING[spec.arm_id]
        trust_remote = spec.arm_id.startswith("bygpt5-")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(ckpt), local_files_only=True, trust_remote_code=trust_remote
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(ckpt), local_files_only=True, trust_remote_code=trust_remote, torch_dtype=torch_dt
        ).eval().to(device)
        self.force_singleton = False
        self.realised_dtype = "/".join(sorted({str(p.dtype) for p in self.model.parameters()}))

    def _encode_bare(self, seq: str) -> list[int]:
        return [int(v) for v in self.tokenizer.encode(seq, add_special_tokens=False)]

    def token_length(self, seq: str) -> int:
        # Bare length + 1 boundary token (conservative for packing)
        return len(self._encode_bare(seq)) + 1

    def overhead_tokens(self) -> int:
        return 4

    @torch.no_grad()
    def _nll_batch_same_len(
        self, rows: Sequence[list[int]], spans: Sequence[tuple[int, int]]
    ) -> list[float]:
        inp = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        with _precision_ctx(self.device, self.dtype_name):
            logits = self.model(input_ids=inp).logits.float()
            out: list[float] = []
            for j, (s0, s1) in enumerate(spans):
                nll = float(F.cross_entropy(logits[j, s0 - 1 : s1 - 1], inp[j, s0:s1], reduction="sum").item())
                out.append(-nll)
            del logits
        return out

    @torch.no_grad()
    def score_condition(
        self,
        wildtype: str,
        mutants: Sequence[str],
        variant_seqs: Sequence[str],
        context_items: Sequence[str],
        *,
        budget_tokens: int,
        check_parity: bool = False,
    ) -> tuple[list[float], float, int]:
        ctx_ids = [self._encode_bare(s) for s in context_items]
        all_seqs = [wildtype, *variant_seqs]
        packed: list[list[int]] = []
        spans: list[tuple[int, int]] = []
        for s in all_seqs:
            full_ids, s0, s1, _k = pack_causal_text_context(
                self._encode_bare(s),
                ctx_ids,
                mode=self.mode,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                budget_tokens=budget_tokens,
            )
            packed.append(full_ids)
            spans.append((s0, s1))

        by_len: dict[int, list[int]] = {}
        for idx, row in enumerate(packed):
            by_len.setdefault(len(row), []).append(idx)

        lls = [0.0] * len(packed)
        for length, idxs in by_len.items():
            eff_bs = 1 if self.force_singleton else max(1, min(self.batch_size, 32768 // max(1, length)))
            for start in range(0, len(idxs), eff_bs):
                chunk_idx = idxs[start : start + eff_bs]
                vals = self._nll_batch_same_len([packed[i] for i in chunk_idx], [spans[i] for i in chunk_idx])
                for i, v in zip(chunk_idx, vals):
                    lls[i] = v

        eff_bs0 = (
            1
            if self.force_singleton
            else max(1, min(self.batch_size, 32768 // max(1, len(packed[0]) if packed else 1)))
        )
        if check_parity and eff_bs0 > 1 and len(packed) >= 3:
            singles = [self._nll_batch_same_len([packed[j]], [spans[j]])[0] for j in range(3)]
            deltas = [abs((lls[j] - lls[0]) - (singles[j] - singles[0])) for j in (1, 2)]
            max_d = max(deltas)
            target_len = max(1, int(spans[0][1] - spans[0][0]))
            max_d_per_tok = max_d / target_len
            self.parity_checks[f"ctx_{len(context_items)}"] = {
                "max_mutant_delta_diff": max_d,
                "max_mutant_delta_diff_per_tok": max_d_per_tok,
                "batch_size": eff_bs0,
            }
            if max_d_per_tok > 5e-3:
                self.force_singleton = True
                lls = [self._nll_batch_same_len([packed[i]], [spans[i]])[0] for i in range(len(packed))]

        wt_ll = float(lls[0])
        scores = [float(v - wt_ll) for v in lls[1:]]
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"{self.spec.entry_key}: non-finite causal text score")
        return scores, wt_ll, int(spans[0][1] - spans[0][0])


def _load_scorer(
    spec: DmsContextSpec,
    *,
    device: str,
    dtype_arg: str,
    model_root: Path | None,
    batch_size_override: int,
) -> _BaseDmsScorer:
    dtype_name, torch_dt = _resolve_dtype(spec, dtype_arg)
    bs = batch_size_override if batch_size_override > 0 else _default_batch_size(spec)
    if spec.scorer_kind in ("bidirectional_mlm", "esm3_mlm"):
        return _BidirectionalDmsScorer(
            spec, device=device, dtype_name=dtype_name, batch_size=bs, model_root=model_root
        )
    if spec.scorer_kind == "msa_transformer":
        return _MsaTransformerDmsScorer(
            spec, device=device, dtype_name=dtype_name, batch_size=bs, model_root=model_root
        )
    if spec.scorer_kind == "gp_causal":
        return _GpCausalDmsScorer(
            spec, device=device, dtype_name=dtype_name, batch_size=bs, model_root=model_root
        )
    if spec.scorer_kind == "progen3_causal":
        return _ProGen3DmsScorer(
            spec, device=device, dtype_name=dtype_name, batch_size=bs, model_root=model_root
        )
    if spec.scorer_kind == "preregistered_causal":
        return _PreregisteredCausalDmsScorer(
            spec, device=device, dtype_name=dtype_name, batch_size=bs
        )
    if spec.scorer_kind == "causal_text":
        return _CausalTextDmsScorer(
            spec,
            device=device,
            dtype_name=dtype_name,
            torch_dt=torch_dt,
            batch_size=bs,
            model_root=model_root,
        )
    raise ValueError(f"Unknown scorer_kind: {spec.scorer_kind}")


# ============================================================================
# Subtest 1A & Subtest 1B Execution Loops (+ Assay-Level Sharding & Checkpointing)
# ============================================================================


def _shard_path(
    out_dir: Path, subtest: str, file_slug: str, shard_index: int, num_shards: int
) -> Path:
    return out_dir / "shards" / subtest / f"{file_slug}.shard{shard_index}of{num_shards}.json"


def _write_sealed_json(out_file: Path, payload: Mapping[str, Any]) -> None:
    raw_txt = json.dumps(payload, sort_keys=True)
    if FORBIDDEN_INFRA_RE.search(raw_txt):
        raise ValueError("Refusing to write output containing forbidden pod/node identifier")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_file, payload)
    digest = sha256_file(out_file)
    out_file.with_name(out_file.name + ".sha256").write_text(
        f"{digest}  {out_file.name}\n", encoding="utf-8"
    )


def _verify_sha256_sidecar(path: Path) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"Missing shard or sha256 sidecar: {path}")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def _load_ckpt_jsonl(
    ckpt_path: Path,
    *,
    entry_key: str,
    subtest: str,
    assay_shard_index: int,
    num_assay_shards: int,
    assay_limit: int,
) -> dict[int, tuple[str, dict[str, Any]]]:
    loaded: dict[int, tuple[str, dict[str, Any]]] = {}
    if not ckpt_path.is_file():
        return loaded
    with ckpt_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                obj.get("entry_key") == entry_key
                and obj.get("subtest") == subtest
                and obj.get("assay_shard_index") == assay_shard_index
                and obj.get("num_assay_shards") == num_assay_shards
                and obj.get("assay_limit") == assay_limit
                and "canon_idx" in obj
                and "kind" in obj
                and "record" in obj
            ):
                loaded[int(obj["canon_idx"])] = (str(obj["kind"]), dict(obj["record"]))
    return loaded


def _append_ckpt_jsonl(
    ckpt_path: Path,
    *,
    entry_key: str,
    subtest: str,
    assay_shard_index: int,
    num_assay_shards: int,
    assay_limit: int,
    canon_idx: int,
    kind: str,
    record: Mapping[str, Any],
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {
            "entry_key": entry_key,
            "subtest": subtest,
            "assay_shard_index": assay_shard_index,
            "num_assay_shards": num_assay_shards,
            "assay_limit": assay_limit,
            "canon_idx": canon_idx,
            "kind": kind,
            "record": record,
        },
        separators=(",", ":"),
    )
    with ckpt_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


def _all_shards_ready(out_dir: Path, subtest: str, file_slug: str, num_assay_shards: int) -> bool:
    for k in range(num_assay_shards):
        sp = _shard_path(out_dir, subtest, file_slug, k, num_assay_shards)
        if not (sp.is_file() and sp.with_name(sp.name + ".sha256").is_file()):
            return False
    return True


def _detect_num_assay_shards(out_dir: Path, subtest: str, file_slug: str) -> int:
    shard_dir = out_dir / "shards" / subtest
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"Shard directory not found: {shard_dir}")
    prefix = f"{file_slug}.shard0of"
    cands: list[int] = []
    for p in shard_dir.glob(f"{prefix}*.json"):
        stem = p.name[len(prefix) : -len(".json")]
        if stem.isdigit():
            cands.append(int(stem))
    if len(cands) != 1:
        raise ValueError(
            f"Cannot unambiguously detect num_assay_shards for {subtest}/{file_slug}: found {cands}"
        )
    return cands[0]


def merge_assay_shards(
    spec: DmsContextSpec,
    subtest: str,
    out_dir: Path,
    *,
    num_assay_shards: int,
) -> dict[str, Any]:
    if num_assay_shards < 2:
        raise ValueError(f"merge_assay_shards requires num_assay_shards >= 2, got {num_assay_shards}")

    shards: list[dict[str, Any]] = []
    for k in range(num_assay_shards):
        sp = _shard_path(out_dir, subtest, spec.file_slug, k, num_assay_shards)
        _verify_sha256_sidecar(sp)
        sh = json.loads(sp.read_text(encoding="utf-8"))
        if sh.get("entry_key") != spec.entry_key or sh.get("subtest") != subtest:
            raise ValueError(f"Shard metadata mismatch in {sp}")
        meta = sh.get("assay_shard") or {}
        if meta.get("index") != k or meta.get("num_shards") != num_assay_shards:
            raise ValueError(f"Shard index/count mismatch in {sp}: {meta}")
        shards.append(sh)

    assay_limits = {int(sh.get("assay_limit", 0)) for sh in shards}
    if len(assay_limits) != 1:
        raise ValueError(f"Inconsistent assay_limit across shards: {assay_limits}")
    assay_limit = next(iter(assay_limits))

    base_prov = dict(shards[0].get("provenance") or {})
    base_prov["merged_from_assay_shards"] = int(num_assay_shards)
    base_prov["shard_parity_checks"] = {
        f"shard{k}": (shards[k].get("provenance") or {}).get("parity_checks", {})
        for k in range(num_assay_shards)
    }
    total_elapsed = float(sum(float(sh.get("elapsed_seconds", 0.0)) for sh in shards))

    if subtest == "rescue_30_80":
        totals = {int((sh.get("assay_shard") or {}).get("total_selected", -1)) for sh in shards}
        if len(totals) != 1 or next(iter(totals)) < 0:
            raise ValueError(f"Inconsistent total_selected across 1A shards: {totals}")
        total_selected = next(iter(totals))

        ref_skipped = shards[0].get("skipped", [])
        for k, sh in enumerate(shards[1:], 1):
            if sh.get("skipped", []) != ref_skipped:
                raise ValueError(f"1A skipped list mismatch between shard0 and shard{k}")

        by_idx: dict[int, dict[str, Any]] = {}
        for k, sh in enumerate(shards):
            cidxs = list((sh.get("assay_shard") or {}).get("canonical_indices", []))
            recs = list(sh.get("assays", []))
            if len(cidxs) != len(recs):
                raise ValueError(f"1A shard{k}: canonical_indices ({len(cidxs)}) != assays ({len(recs)})")
            for ci, rec in zip(cidxs, recs):
                if ci in by_idx:
                    raise ValueError(f"Duplicate canonical index {ci} across 1A shards")
                by_idx[int(ci)] = rec

        expected_idxs = set(range(total_selected))
        if set(by_idx.keys()) != expected_idxs:
            missing = sorted(expected_idxs - set(by_idx.keys()))
            extra = sorted(set(by_idx.keys()) - expected_idxs)
            raise ValueError(f"1A shard coverage incomplete: missing={missing[:10]}, extra={extra[:10]}")

        merged_assays = [by_idx[i] for i in range(total_selected)]
        payload: dict[str, Any] = {
            "schema": SCHEMA_DMS_ARM_SCORE,
            "subtest": "rescue_30_80",
            "status": "complete",
            "entry_key": spec.entry_key,
            "arm": spec.arm_id,
            "input_alignment_format": spec.input_alignment_format,
            "context_training_regime": spec.context_training_regime,
            "budget": spec.budget_tokens,
            "aligned_msa": spec.aligned_msa,
            "assay_limit": int(assay_limit),
            "elapsed_seconds": total_elapsed,
            "provenance": base_prov,
            "assays_scored": len(merged_assays),
            "assays_skipped": len(ref_skipped),
            "skipped": ref_skipped,
            "assays": merged_assays,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
    elif subtest == "e09_five_band":
        totals = {int((sh.get("assay_shard") or {}).get("total_declared", -1)) for sh in shards}
        if len(totals) != 1 or next(iter(totals)) < 0:
            raise ValueError(f"Inconsistent total_declared across 1B shards: {totals}")
        total_declared = next(iter(totals))

        scored_by_idx: dict[int, dict[str, Any]] = {}
        skipped_by_idx: dict[int, dict[str, Any]] = {}
        for k, sh in enumerate(shards):
            meta = sh.get("assay_shard") or {}
            sc_idxs = list(meta.get("scored_canonical_indices", []))
            sk_idxs = list(meta.get("skipped_canonical_indices", []))
            recs = list(sh.get("assays", []))
            skips = list(sh.get("skipped", []))
            if len(sc_idxs) != len(recs) or len(sk_idxs) != len(skips):
                raise ValueError(f"1B shard{k}: index/record length mismatch")
            for ci, rec in zip(sc_idxs, recs):
                ci = int(ci)
                if ci in scored_by_idx or ci in skipped_by_idx:
                    raise ValueError(f"Duplicate canonical index {ci} across 1B shards")
                scored_by_idx[ci] = rec
            for ci, sk in zip(sk_idxs, skips):
                ci = int(ci)
                if ci in scored_by_idx or ci in skipped_by_idx:
                    raise ValueError(f"Duplicate canonical index {ci} across 1B shards")
                skipped_by_idx[ci] = sk

        covered = set(scored_by_idx.keys()) | set(skipped_by_idx.keys())
        expected_idxs = set(range(total_declared))
        if covered != expected_idxs:
            missing = sorted(expected_idxs - covered)
            extra = sorted(covered - expected_idxs)
            raise ValueError(f"1B shard coverage incomplete: missing={missing[:10]}, extra={extra[:10]}")

        merged_assays = [scored_by_idx[i] for i in sorted(scored_by_idx.keys())]
        merged_skipped = [skipped_by_idx[i] for i in sorted(skipped_by_idx.keys())]
        payload = {
            "schema": SCHEMA_DMS_ARM_SCORE,
            "subtest": "e09_five_band",
            "status": "complete",
            "entry_key": spec.entry_key,
            "arm": spec.arm_id,
            "input_alignment_format": spec.input_alignment_format,
            "context_training_regime": spec.context_training_regime,
            "budget": spec.budget_tokens,
            "aligned_msa": spec.aligned_msa,
            "assay_limit": int(assay_limit),
            "elapsed_seconds": total_elapsed,
            "provenance": base_prov,
            "assays_declared": total_declared,
            "assays_scored": len(merged_assays),
            "assays_skipped": len(merged_skipped),
            "skipped": merged_skipped,
            "assays": merged_assays,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
    else:
        raise ValueError(f"Unknown subtest: {subtest}")

    out_file = out_dir / subtest / f"{spec.file_slug}.json"
    _write_sealed_json(out_file, payload)
    print(
        f"[R5-DMS:merge] {spec.entry_key} ({subtest}): merged {num_assay_shards} shards -> "
        f"{out_file.name} (scored={payload['assays_scored']}, skipped={payload['assays_skipped']})",
        flush=True,
    )
    return payload


def _run_subtest_1a(
    spec: DmsContextSpec,
    scorer: _BaseDmsScorer,
    cohort: Mapping[str, Any],
    out_file: Path,
    *,
    assay_limit: int = 0,
    assay_shard_index: int = 0,
    num_assay_shards: int = 1,
) -> dict[str, Any]:
    t0 = time.monotonic()
    assays_in = list(cohort["assays"])
    multi_icl = spec.context_training_regime == "multiseq_icl_unaligned"
    msa_mode = spec.scorer_kind == "msa_transformer"

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    skipped: list[dict[str, Any]] = []
    for row in assays_in:
        plan, reason = plan_rescue_30_80_assay(
            row,
            token_length_fn=scorer.token_length,
            overhead_tokens=scorer.overhead_tokens(),
            budget_tokens=spec.budget_tokens,
            multi_item_icl=multi_icl,
            msa_mode=msa_mode,
        )
        if plan is None:
            skipped.append({"assay": row["assay"], "reason": reason})
        else:
            selected.append((row, plan))
            if assay_limit > 0 and len(selected) >= assay_limit:
                break

    if num_assay_shards > 1:
        shard_items = [
            (canon_idx, row, plan)
            for canon_idx, (row, plan) in enumerate(selected)
            if canon_idx % num_assay_shards == assay_shard_index
        ]
    else:
        shard_items = [(canon_idx, row, plan) for canon_idx, (row, plan) in enumerate(selected)]

    ckpt_path = out_file.with_suffix(".ckpt.jsonl")
    cached_ckpt = _load_ckpt_jsonl(
        ckpt_path,
        entry_key=spec.entry_key,
        subtest="rescue_30_80",
        assay_shard_index=assay_shard_index,
        num_assay_shards=num_assay_shards,
        assay_limit=assay_limit,
    )
    if cached_ckpt:
        print(
            f"[1A:{spec.entry_key}] resumed {len(cached_ckpt)}/{len(shard_items)} assays from {ckpt_path.name}",
            flush=True,
        )

    results: list[dict[str, Any]] = []
    checked_first = False
    tag = (
        f"1A:{spec.entry_key}:s{assay_shard_index}/{num_assay_shards}"
        if num_assay_shards > 1
        else f"1A:{spec.entry_key}"
    )
    for local_idx, (canon_idx, row, plan) in enumerate(shard_items, 1):
        if canon_idx in cached_ckpt and cached_ckpt[canon_idx][0] == "scored":
            results.append(cached_ckpt[canon_idx][1])
            continue
        wt = str(row["wildtype"])
        muts = list(row["mutants"])
        vseqs = list(row["sequences"])
        cond_items = {
            "no_context": [],
            "unrelated": plan["unrelated_items"],
            "homolog": plan["homolog_items"],
        }
        scores: dict[str, list[float]] = {}
        wt_lls: dict[str, float] = {}
        wt_toks = len(wt)
        for cond in RESCUE_CONDITIONS:
            sc, wll, wtok = scorer.score_condition(
                wt,
                muts,
                vseqs,
                cond_items[cond],
                budget_tokens=spec.budget_tokens,
                check_parity=(not checked_first),
            )
            scores[cond] = sc
            wt_lls[cond] = wll
            wt_toks = wtok
        checked_first = True
        rec = {
            "assay": row["assay"],
            "wildtype_id": row["wildtype_id"],
            "cluster": int(row["cluster"]),
            "mutants": muts,
            "measured": row["measured"],
            "profile_scores": row["profile_scores"],
            "mutant_digest": row["mutant_digest"],
            "original_mutant_digest": row["original_mutant_digest"],
            "scores": scores,
            "wt_log_likelihood": wt_lls,
            "wt_scored_tokens": wt_toks,
            "context_tokens": int(plan["context_tokens"]),
            "k_items": int(plan["k_items"]),
            "homolog_identity": float(plan["homolog_identity"]),
            "max_lcs": int(plan["max_lcs"]),
            "spearman": {c: _rho(scores[c], row["measured"]) for c in RESCUE_CONDITIONS},
            "lookup_spearman": _rho(row["profile_scores"], row["measured"]),
        }
        results.append(rec)
        _append_ckpt_jsonl(
            ckpt_path,
            entry_key=spec.entry_key,
            subtest="rescue_30_80",
            assay_shard_index=assay_shard_index,
            num_assay_shards=num_assay_shards,
            assay_limit=assay_limit,
            canon_idx=canon_idx,
            kind="scored",
            record=rec,
        )
        if local_idx % 20 == 0 or local_idx == len(shard_items):
            print(
                f"[{tag}] {local_idx}/{len(shard_items)} ({time.monotonic() - t0:.1f}s) "
                f"last={rec['assay']} sp={rec['spearman']}",
                flush=True,
            )

    payload: dict[str, Any] = {
        "schema": SCHEMA_DMS_ARM_SCORE,
        "subtest": "rescue_30_80",
        "status": "complete",
        "entry_key": spec.entry_key,
        "arm": spec.arm_id,
        "input_alignment_format": spec.input_alignment_format,
        "context_training_regime": spec.context_training_regime,
        "budget": spec.budget_tokens,
        "aligned_msa": spec.aligned_msa,
        "assay_limit": int(assay_limit),
        "elapsed_seconds": time.monotonic() - t0,
        "provenance": scorer.provenance(),
        "assays_scored": len(results),
        "assays_skipped": len(skipped),
        "skipped": skipped,
        "assays": results,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    if num_assay_shards > 1:
        payload["assay_shard"] = {
            "index": int(assay_shard_index),
            "num_shards": int(num_assay_shards),
            "total_selected": len(selected),
            "canonical_indices": [int(c) for c, _, _ in shard_items],
        }
    _write_sealed_json(out_file, payload)
    if ckpt_path.is_file():
        ckpt_path.unlink()
    return payload


def _run_subtest_1b(
    spec: DmsContextSpec,
    scorer: _BaseDmsScorer,
    cohort: Mapping[str, Any],
    out_file: Path,
    *,
    assay_limit: int = 0,
    assay_shard_index: int = 0,
    num_assay_shards: int = 1,
) -> dict[str, Any]:
    t0 = time.monotonic()
    msa_mode = spec.scorer_kind == "msa_transformer"
    long_icl = spec.budget_tokens > 1024

    targets_by_id = {t["target_id"]: t for t in cohort["targets"]}
    assays_by_target: dict[str, list[dict[str, Any]]] = {}
    for a in cohort["assays"]:
        assays_by_target.setdefault(a["target_id"], []).append(a)

    all_conds = ("no_context", "unrelated", *E09_BINS_ORDERED)
    assays_list = list(cohort["assays"])
    if assay_limit > 0:
        assays_list = assays_list[:assay_limit]

    if num_assay_shards > 1:
        shard_items = [
            (canon_idx, arow)
            for canon_idx, arow in enumerate(assays_list)
            if canon_idx % num_assay_shards == assay_shard_index
        ]
    else:
        shard_items = [(canon_idx, arow) for canon_idx, arow in enumerate(assays_list)]

    # Plan only targets actually present in this shard's slice
    needed_tids = {arow["target_id"] for _, arow in shard_items}
    plans_by_target: dict[str, dict[str, Any]] = {}
    for tid in sorted(needed_tids):
        plans_by_target[tid] = plan_e09_target(
            targets_by_id[tid],
            assays_by_target.get(tid, ()),
            token_length_fn=scorer.token_length,
            overhead_tokens=scorer.overhead_tokens(),
            budget_tokens=spec.budget_tokens,
            msa_mode=msa_mode,
            long_icl_mode=long_icl,
        )

    ckpt_path = out_file.with_suffix(".ckpt.jsonl")
    cached_ckpt = _load_ckpt_jsonl(
        ckpt_path,
        entry_key=spec.entry_key,
        subtest="e09_five_band",
        assay_shard_index=assay_shard_index,
        num_assay_shards=num_assay_shards,
        assay_limit=assay_limit,
    )
    if cached_ckpt:
        print(
            f"[1B:{spec.entry_key}] resumed {len(cached_ckpt)}/{len(shard_items)} assays from {ckpt_path.name}",
            flush=True,
        )

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    scored_canon_indices: list[int] = []
    skipped_canon_indices: list[int] = []
    checked_first = False
    tag = (
        f"1B:{spec.entry_key}:s{assay_shard_index}/{num_assay_shards}"
        if num_assay_shards > 1
        else f"1B:{spec.entry_key}"
    )
    for local_idx, (canon_idx, arow) in enumerate(shard_items, 1):
        if canon_idx in cached_ckpt:
            kind, rec = cached_ckpt[canon_idx]
            if kind == "scored":
                results.append(rec)
                scored_canon_indices.append(int(canon_idx))
            else:
                skipped.append(rec)
                skipped_canon_indices.append(int(canon_idx))
            continue

        tid = arow["target_id"]
        trow = targets_by_id[tid]
        tplan = plans_by_target[tid]
        present_conds = [
            c for c in all_conds if (tplan["conditions"].get(c) or {}).get("status") == "present"
        ]
        if not present_conds:
            sk_rec = {"assay": arow["assay"], "reason": tplan.get("skipped_reason", "no_present_conds")}
            skipped.append(sk_rec)
            skipped_canon_indices.append(int(canon_idx))
            _append_ckpt_jsonl(
                ckpt_path,
                entry_key=spec.entry_key,
                subtest="e09_five_band",
                assay_shard_index=assay_shard_index,
                num_assay_shards=num_assay_shards,
                assay_limit=assay_limit,
                canon_idx=canon_idx,
                kind="skipped",
                record=sk_rec,
            )
            continue
        wt = str(trow["wildtype"])
        muts = list(arow["mutants"])
        vseqs = _rebuild_variants(wt, muts)

        sp_map: dict[str, float | None] = {}
        wt_lls: dict[str, float] = {}
        ctx_id_map: dict[str, list[float]] = {}
        ctx_lcs_map: dict[str, list[int]] = {}
        ctx_tok_map: dict[str, int] = {}
        for cond in present_conds:
            items = tplan["items"].get(cond) or []
            seqs = [it["sequence"] for it in items]
            sc, wll, _wtok = scorer.score_condition(
                wt,
                muts,
                vseqs,
                seqs,
                budget_tokens=spec.budget_tokens,
                check_parity=(not checked_first),
            )
            sp_map[cond] = _rho(sc, arow["measured"])
            wt_lls[cond] = wll
            ctx_id_map[cond] = [float(it.get("identity", 0.0)) for it in items]
            ctx_lcs_map[cond] = [int(it.get("max_lcs", 0)) for it in items]
            ctx_tok_map[cond] = int(sum(int(it.get("tokens", len(it["sequence"]))) for it in items))
        checked_first = True

        rec = {
            "assay": arow["assay"],
            "target_id": tid,
            "cluster": int(arow["cluster"]),
            "mutant_digest": arow["mutant_digest"],
            "item_count": int(tplan.get("item_count", 0)),
            "absent_conditions": [c for c in all_conds if c not in present_conds],
            "spearman": sp_map,
            "lookup_spearman": _rho(arow["profile_scores"], arow["measured"]),
            "wt_log_likelihood": wt_lls,
            "context_identities": ctx_id_map,
            "context_max_lcs": ctx_lcs_map,
            "context_tokens": ctx_tok_map,
        }
        results.append(rec)
        scored_canon_indices.append(int(canon_idx))
        _append_ckpt_jsonl(
            ckpt_path,
            entry_key=spec.entry_key,
            subtest="e09_five_band",
            assay_shard_index=assay_shard_index,
            num_assay_shards=num_assay_shards,
            assay_limit=assay_limit,
            canon_idx=canon_idx,
            kind="scored",
            record=rec,
        )
        if local_idx % 20 == 0 or local_idx == len(shard_items):
            print(
                f"[{tag}] {local_idx}/{len(shard_items)} ({time.monotonic() - t0:.1f}s) "
                f"last={arow['assay']} n_conds={len(present_conds)}",
                flush=True,
            )

    payload: dict[str, Any] = {
        "schema": SCHEMA_DMS_ARM_SCORE,
        "subtest": "e09_five_band",
        "status": "complete",
        "entry_key": spec.entry_key,
        "arm": spec.arm_id,
        "input_alignment_format": spec.input_alignment_format,
        "context_training_regime": spec.context_training_regime,
        "budget": spec.budget_tokens,
        "aligned_msa": spec.aligned_msa,
        "assay_limit": int(assay_limit),
        "elapsed_seconds": time.monotonic() - t0,
        "provenance": scorer.provenance(),
        "assays_declared": len(shard_items),
        "assays_scored": len(results),
        "assays_skipped": len(skipped),
        "skipped": skipped,
        "assays": results,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    if num_assay_shards > 1:
        payload["assay_shard"] = {
            "index": int(assay_shard_index),
            "num_shards": int(num_assay_shards),
            "total_declared": len(assays_list),
            "canonical_indices": [int(c) for c, _ in shard_items],
            "scored_canonical_indices": scored_canon_indices,
            "skipped_canonical_indices": skipped_canon_indices,
        }
    _write_sealed_json(out_file, payload)
    if ckpt_path.is_file():
        ckpt_path.unlink()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subtest", default="both", choices=("rescue_30_80", "e09_five_band", "both"))
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated entry_keys (e.g. 'esm2-8m,msa-transformer-100m|aligned=True,progen2-small').",
    )
    parser.add_argument("--priority", default="core", choices=("core", "all"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--assay-shard-index",
        type=int,
        default=0,
        help="0-based assay-level shard index within each model/subtest.",
    )
    parser.add_argument(
        "--num-assay-shards",
        type=int,
        default=1,
        help=">1 splits assays modulo N into shards/<subtest>/<slug>.shard<k>of<n>.json.",
    )
    parser.add_argument(
        "--merge-assay-shards",
        action="store_true",
        help="Pure-CPU merge of completed assay shard JSONs into final <subtest>/<slug>.json (+ .sha256).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16"))
    parser.add_argument("--batch-size", type=int, default=0, help="0 = auto per-model batch size")
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DATA_ROOT / DMS_SCORES_DIR_REL)
    parser.add_argument("--assay-limit", type=int, default=0, help=">0 for smoke tests only")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force-rescore-cached", action="store_true")
    args = parser.parse_args()

    if args.num_assay_shards < 1:
        raise ValueError(f"--num-assay-shards must be >= 1, got {args.num_assay_shards}")
    if not (0 <= args.assay_shard_index < args.num_assay_shards):
        raise ValueError(
            f"--assay-shard-index {args.assay_shard_index} out of range [0, {args.num_assay_shards})"
        )

    if args.model_root is not None and not os.environ.get("MODEL_ROOT"):
        os.environ["MODEL_ROOT"] = str(args.model_root)

    if args.models.strip():
        req_keys = [k.strip() for k in args.models.split(",") if k.strip()]
        specs = [DMS_CONTEXT_SPEC_BY_KEY[k] for k in req_keys]
    else:
        pool = (
            [DMS_CONTEXT_SPEC_BY_KEY[k] for k in DMS_CONTEXT_CORE_KEYS]
            if args.priority == "core"
            else list(DMS_CONTEXT_SPECS_ALL)
        )
        shards = assign_balanced_shards(pool, args.num_shards)
        specs = shards[args.shard_index]

    subtests = ("rescue_30_80", "e09_five_band") if args.subtest == "both" else (args.subtest,)

    if args.merge_assay_shards:
        for spec in specs:
            for st in subtests:
                final_f = args.out_dir / st / f"{spec.file_slug}.json"
                if (
                    args.skip_existing
                    and final_f.is_file()
                    and final_f.with_name(final_f.name + ".sha256").is_file()
                ):
                    print(f"[R5-DMS:merge] {spec.entry_key} ({st}): skipping existing {final_f.name}", flush=True)
                    continue
                n_sh = (
                    args.num_assay_shards
                    if args.num_assay_shards > 1
                    else _detect_num_assay_shards(args.out_dir, st, spec.file_slug)
                )
                merge_assay_shards(spec, st, args.out_dir, num_assay_shards=n_sh)
        return

    cohort_1a = (
        json.loads((DATA_ROOT / RESCUE_COHORT_REL).read_text(encoding="utf-8"))
        if "rescue_30_80" in subtests
        else None
    )
    cohort_1b = (
        json.loads((DATA_ROOT / E09_COHORT_REL).read_text(encoding="utf-8"))
        if "e09_five_band" in subtests
        else None
    )

    print(
        f"[R5-DMS] shard {args.shard_index}/{args.num_shards} | "
        f"assay_shard {args.assay_shard_index}/{args.num_assay_shards} | "
        f"subtests={subtests} | {len(specs)} entries: {[s.entry_key for s in specs]}",
        flush=True,
    )

    for spec in specs:
        todo_subtests: list[str] = []
        for st in subtests:
            if not args.force_rescore_cached:
                if st == "rescue_30_80" and spec.entry_key in CACHED_1A_PATHS and (DATA_ROOT / CACHED_1A_PATHS[spec.entry_key]).is_file():
                    print(f"[R5-DMS] {spec.entry_key} ({st}): reusing frozen cached legacy artifact", flush=True)
                    continue
                if st == "e09_five_band" and spec.entry_key in CACHED_1B_PATHS and (DATA_ROOT / CACHED_1B_PATHS[spec.entry_key]).is_file():
                    print(f"[R5-DMS] {spec.entry_key} ({st}): reusing frozen cached legacy artifact", flush=True)
                    continue
            final_f = args.out_dir / st / f"{spec.file_slug}.json"
            if args.skip_existing and final_f.is_file() and final_f.with_name(final_f.name + ".sha256").is_file():
                print(f"[R5-DMS] {spec.entry_key} ({st}): skipping existing {final_f.name}", flush=True)
                continue
            if args.num_assay_shards > 1:
                sp = _shard_path(
                    args.out_dir, st, spec.file_slug, args.assay_shard_index, args.num_assay_shards
                )
                if args.skip_existing and sp.is_file() and sp.with_name(sp.name + ".sha256").is_file():
                    print(f"[R5-DMS] {spec.entry_key} ({st}): skipping existing shard {sp.name}", flush=True)
                    if _all_shards_ready(args.out_dir, st, spec.file_slug, args.num_assay_shards):
                        merge_assay_shards(spec, st, args.out_dir, num_assay_shards=args.num_assay_shards)
                    continue
            todo_subtests.append(st)

        if not todo_subtests:
            continue

        scorer = _load_scorer(
            spec,
            device=args.device,
            dtype_arg=args.dtype,
            model_root=args.model_root,
            batch_size_override=args.batch_size,
        )
        try:
            for st in todo_subtests:
                if args.num_assay_shards > 1:
                    out_f = _shard_path(
                        args.out_dir, st, spec.file_slug, args.assay_shard_index, args.num_assay_shards
                    )
                else:
                    out_f = args.out_dir / st / f"{spec.file_slug}.json"
                out_f.parent.mkdir(parents=True, exist_ok=True)
                if st == "rescue_30_80":
                    assert cohort_1a is not None
                    _run_subtest_1a(
                        spec,
                        scorer,
                        cohort_1a,
                        out_f,
                        assay_limit=args.assay_limit,
                        assay_shard_index=args.assay_shard_index,
                        num_assay_shards=args.num_assay_shards,
                    )
                else:
                    assert cohort_1b is not None
                    _run_subtest_1b(
                        spec,
                        scorer,
                        cohort_1b,
                        out_f,
                        assay_limit=args.assay_limit,
                        assay_shard_index=args.assay_shard_index,
                        num_assay_shards=args.num_assay_shards,
                    )
                if args.num_assay_shards > 1 and _all_shards_ready(
                    args.out_dir, st, spec.file_slug, args.num_assay_shards
                ):
                    merge_assay_shards(spec, st, args.out_dir, num_assay_shards=args.num_assay_shards)
        finally:
            del scorer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
