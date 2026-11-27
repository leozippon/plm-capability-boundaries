#!/usr/bin/env python3
"""GPU runner for Workstream R4-Score: unaligned homologous-prefix scoring across
the 36 missing single-chain protein arms on the frozen 800-unit EXP-R2-228 cohort.

Covers:
  - Group A (14 causal text arms): bygpt5-{small,base,medium}-en, dialogpt-small,
    gpt2, gpt2-medium, gpt2-large, gpt2-xl, llama-3.2-3b, qwen2.5-0.5b,
    qwen2.5-0.5b-instruct, qwen2.5-7b, qwen2.5-32b, qwen3-8b-base.
  - Group B (22 bidirectional / masked / 3Di / diffusion / multimodal single-chain arms):
    esm2-{8m,35m,150m,650m,3b,15b}, esmc-{300m,600m,6b}, proteinglm-{1b,3b,10b}-mlm,
    amplify-{120m,350m}, prime-690m, saprot-{35m,650m,1.3b} (#-masked seq-only),
    dplm-{150m,650m,3b}, esm3-1.4b (sequence-only unaligned prefix; requires ESM3_SITE).
  - Optional parity verification on cached causal arms (e.g. ``--arm progen2-small --max-units 16``).
  - Optional 32-unit ``exact_pll`` vs ``pll_8fold`` diagnostic (``--diagnostic-exact-pll-units 32``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("TRANSFER_DATA_ROOT") or ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.amino_acids import AA20
from src.capability.core.precision_policy import fp32_matmul_context
from src.capability.general_plm.eval_utils import atomic_write_json, sha256_file
from src.capability.unified.r4_scoring_yardstick import (
    CONDITIONS,
    FORBIDDEN_INFRA_RE,
    GROUP_A_CAUSAL_TEXT_ARMS,
    GROUP_B_BIDIRECTIONAL_ARMS,
    IDENTITY_BANDS,
    SCHEMA_ARM_RUNNER,
    build_8fold_masks,
    build_unit_condition_inputs,
    pack_bidirectional_context,
    pack_causal_text_context,
    summarize_unit_rows,
)

AA20_INDEX = {aa: i for i, aa in enumerate(AA20)}

# Candidate local directory names under TEXT_MODEL_BASE_DIR / MODEL_ROOT / MODEL_BASE_DIR
TEXT_DIR_CANDIDATES: dict[str, tuple[str, ...]] = {
    "bygpt5-small-en": ("bygpt5-small-en",),
    "bygpt5-base-en": ("bygpt5-base-en",),
    "bygpt5-medium-en": ("bygpt5-medium-en",),
    "dialogpt-small": ("DialoGPT-small", "dialogpt-small"),
    "gpt2": ("gpt2",),
    "gpt2-medium": ("gpt2-medium",),
    "gpt2-large": ("gpt2-large",),
    "gpt2-xl": ("gpt2-xl",),
    "llama-3.2-3b": ("Llama-3.2-3B", "llama-3.2-3b"),
    "qwen2.5-0.5b": ("Qwen2.5-0.5B", "qwen2.5-0.5b"),
    "qwen2.5-0.5b-instruct": ("Qwen2.5-0.5B-Instruct", "qwen2.5-0.5b-instruct"),
    "qwen2.5-7b": ("Qwen2.5-7B", "qwen2.5-7b"),
    "qwen2.5-32b": ("Qwen2.5-32B", "qwen2.5-32b"),
    "qwen3-8b-base": ("Qwen3-8B-Base", "qwen3-8b-base"),
}

# Exact boundary / packing rules aligned with src/capability/text_aa_boundaries.json
CAUSAL_TEXT_PACKING: dict[str, tuple[str, int | None, int | None]] = {
    "gpt2": ("sep_prefix", 50256, 50256),
    "gpt2-medium": ("sep_prefix", 50256, 50256),
    "gpt2-large": ("sep_prefix", 50256, 50256),
    "gpt2-xl": ("sep_prefix", 50256, 50256),
    "dialogpt-small": ("sep_prefix", 50256, 50256),
    "qwen2.5-0.5b": ("sep_prefix", 151643, 151643),
    "qwen2.5-0.5b-instruct": ("sep_prefix", 151643, 151643),
    "qwen2.5-7b": ("sep_prefix", 151643, 151643),
    "qwen2.5-32b": ("sep_prefix", 151643, 151643),
    "qwen3-8b-base": ("sep_prefix", 151643, 151643),
    "llama-3.2-3b": ("bos_eos_blocks", 128000, 128001),
    "bygpt5-small-en": ("bare_eos_sep", None, 1),
    "bygpt5-base-en": ("bare_eos_sep", None, 1),
    "bygpt5-medium-en": ("bare_eos_sep", None, 1),
}

ALL_SUPPORTED_ARMS: tuple[str, ...] = (
    *GROUP_A_CAUSAL_TEXT_ARMS,
    *GROUP_B_BIDIRECTIONAL_ARMS,
    "progen2-small",
)


def _resolve_text_checkpoint_dir(arm: str, explicit_root: Path | None) -> Path:
    """Resolve local checkpoint directory for a Group A causal text arm without network access."""
    from src.capability.core import arms as core_arms

    roots: list[Path] = []
    if explicit_root is not None:
        roots.append(explicit_root)
    for var in ("TEXT_MODEL_BASE_DIR", "MODEL_ROOT", "MODEL_BASE_DIR"):
        val = os.environ.get(var)
        if val:
            roots.append(Path(val))
    roots.extend([core_arms.TEXT_MODEL_BASE, core_arms.MODEL_ROOT, core_arms.MODEL_BASE_DIR])

    # Check declared ArmSpec path if present
    all_specs = {**core_arms.STAGED_ARMS, **core_arms.PANEL}
    if arm in all_specs and all_specs[arm].path.is_dir():
        return all_specs[arm].path

    for root in roots:
        for sub in TEXT_DIR_CANDIDATES.get(arm, (arm,)):
            cand = root / sub
            if cand.is_dir():
                return cand
    raise FileNotFoundError(
        f"{arm}: checkpoint directory not found; searched roots={[str(r) for r in roots]}"
    )


def _precision_ctx(device: str, dtype_name: str):
    if dtype_name == "float32" and device.startswith("cuda"):
        return fp32_matmul_context(torch)
    return nullcontext()


def _resolve_torch_dtype(arm: str, dtype_arg: str) -> tuple[str, torch.dtype]:
    raw = dtype_arg.strip().lower()
    if raw in ("", "auto", "default"):
        # FP32 by default for <=15B arms; BF16 default for 32B text arm & ESM3-1.4B unless overridden
        if arm in ("qwen2.5-32b", "esm3-1.4b"):
            return "bfloat16", torch.bfloat16
        return "float32", torch.float32
    if raw in ("fp32", "float32"):
        return "float32", torch.float32
    if raw in ("bf16", "bfloat16"):
        return "bfloat16", torch.bfloat16
    raise ValueError(f"Unsupported dtype {dtype_arg!r}; use auto, float32, or bfloat16")


# ============================================================================
# Group A: Causal Text Scorer
# ============================================================================


class _CausalTextR4Scorer:
    def __init__(self, arm: str, *, device: str, dtype_name: str, torch_dt: torch.dtype, model_root: Path | None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.arm = arm
        self.device = device
        self.dtype_name = dtype_name
        self.checkpoint_dir = _resolve_text_checkpoint_dir(arm, model_root)
        self.mode, self.bos_id, self.eos_id = CAUSAL_TEXT_PACKING[arm]

        trust_remote = arm.startswith("bygpt5-")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.checkpoint_dir), local_files_only=True, trust_remote_code=trust_remote
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.checkpoint_dir),
            local_files_only=True,
            trust_remote_code=trust_remote,
            torch_dtype=torch_dt,
        )
        self.model.eval().to(self.device)
        self._realised_dtype = "/".join(sorted({str(p.dtype) for p in self.model.parameters()}))
        self._n_params = int(sum(p.numel() for p in self.model.parameters()))

    def _encode_bare(self, seq: str) -> list[int]:
        ids = self.tokenizer.encode(seq, add_special_tokens=False)
        return [int(v) for v in ids]

    @torch.no_grad()
    def score_condition(
        self,
        target_seq: str,
        context_items: Sequence[str],
        *,
        budget_tokens: int,
    ) -> dict[str, Any]:
        target_ids = self._encode_bare(target_seq)
        ctx_ids = [self._encode_bare(s) for s in context_items]
        full_ids, span_start, span_end, k_items = pack_causal_text_context(
            target_ids,
            ctx_ids,
            mode=self.mode,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            budget_tokens=budget_tokens,
        )
        inp = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        with _precision_ctx(self.device, self.dtype_name):
            out = self.model(input_ids=inp)
            logits = out.logits[0, span_start - 1 : span_end - 1, :].float()
            targets = inp[0, span_start:span_end]
            nll_sum = float(F.cross_entropy(logits, targets, reduction="sum").item())
        if not np.isfinite(nll_sum):
            raise FloatingPointError(f"{self.arm}: non-finite causal NLL sum")
        n_scored = int(span_end - span_start)
        return {
            "nll_sum": nll_sum,
            "nll_per_residue": nll_sum / len(target_seq),
            "nats_per_token": nll_sum / n_scored,
            "scored_tokens": n_scored,
            "total_tokens": len(full_ids),
            "k_items": k_items,
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "group": "group_a_causal_text",
            "packing_mode": self.mode,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "dtype_requested": self.dtype_name,
            "dtype_realised": self._realised_dtype,
            "parameters": self._n_params,
            "checkpoint_dir": str(self.checkpoint_dir),
            "device": self.device,
        }


# ============================================================================
# Group B: Bidirectional / Masked / SaProt / DPLM / ESM3-1.4B Scorer
# ============================================================================


class _BidirectionalR4Scorer:
    def __init__(
        self,
        arm: str,
        *,
        device: str,
        dtype_name: str,
        model_root: Path | None,
        fold_batch_size: int = 9,
    ):
        self.arm = arm
        self.device = device
        self.dtype_name = dtype_name
        self.fold_batch_size = max(1, int(fold_batch_size))
        self.is_esm3 = arm == "esm3-1.4b"
        self.is_saprot = False

        if self.is_esm3:
            from src.capability.esm3_ext.esm3_adapter import load_esm3_offline

            self.esm3 = load_esm3_offline(
                model_dir=(model_root / "esm3-sm-open-v1") if model_root and (model_root / "esm3-sm-open-v1").is_dir() else None,
                device=device,
                dtype=dtype_name,
                eager_load_aux=False,
            )
            self.bos_prefix = [int(self.esm3.bos_id)]
            self.eos_suffix = [int(self.esm3.eos_id)]
            self.sep_ids = [int(self.esm3.eos_id)]
            self.mask_id = int(self.esm3.mask_id)
            self.aa_to_id = dict(self.esm3.aa_to_id)
            self.adapter = None
        else:
            from src.capability.general_plm import load_general_plm
            from src.capability.general_plm.adapters import SaProtAdapter

            self.esm3 = None
            self.adapter = load_general_plm(
                arm,
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

    def _encode_residues(self, seq: str) -> list[int]:
        return [self.aa_to_id[aa] for aa in seq]

    @torch.no_grad()
    def _forward_logits_batch(self, batch_ids: torch.Tensor) -> torch.Tensor:
        if self.is_esm3:
            from src.capability.esm3_ext.esm3_adapter import forward_tracks

            out = forward_tracks(self.esm3, sequence_tokens=batch_ids)
            return out.sequence_logits.float()
        assert self.adapter is not None
        with self.adapter._precision_ctx():
            return self.adapter._forward_logits(batch_ids)

    def _score_masked_positions_from_logits(
        self,
        logits_row: torch.Tensor,
        positions: Sequence[int],
        target_seq: str,
        target_start: int,
    ) -> float:
        """Sum negative log-probabilities at ``target_start + pos`` for true AA20 residues."""
        pos_idx = torch.as_tensor([target_start + p for p in positions], dtype=torch.long, device=logits_row.device)
        sel_logits = logits_row.index_select(0, pos_idx)  # (M, V)
        full_lp = F.log_softmax(sel_logits, dim=-1)  # (M, V)
        if self.is_saprot:
            assert self.adapter is not None
            # Marginalize over all 20 3Di tokens per AA20 residue (preserving full-vocab normalization)
            struct_mat = self.adapter._struct_id_matrix  # (20, 20)
            aa_indices = torch.as_tensor(
                [AA20_INDEX[target_seq[p]] for p in positions], dtype=torch.long, device=logits_row.device
            )
            tok_20 = struct_mat.index_select(0, aa_indices)  # (M, 20)
            gathered = torch.gather(full_lp, dim=-1, index=tok_20)  # (M, 20)
            aa_lp = torch.logsumexp(gathered, dim=-1)  # (M,)
            return float((-aa_lp).sum(dtype=torch.float64).item())
        true_ids = torch.as_tensor(
            [self.aa_to_id[target_seq[p]] for p in positions], dtype=torch.long, device=logits_row.device
        )
        chosen_lp = full_lp.gather(1, true_ids.unsqueeze(1)).squeeze(1)
        return float((-chosen_lp).sum(dtype=torch.float64).item())

    @torch.no_grad()
    def score_condition(
        self,
        unit_key: str,
        target_seq: str,
        context_items: Sequence[str],
        *,
        budget_tokens: int,
    ) -> dict[str, Any]:
        """Score both ``pll_8fold`` (primary) and ``full_target_masked`` (companion) in one unpadded batch."""
        tgt_ids = self._encode_residues(target_seq)
        ctx_ids = [self._encode_residues(s) for s in context_items]
        full_ids, target_start, target_end, k_items = pack_bidirectional_context(
            tgt_ids,
            ctx_ids,
            bos_prefix=self.bos_prefix,
            eos_suffix=self.eos_suffix,
            sep_ids=self.sep_ids,
            budget_tokens=budget_tokens,
        )
        L = len(target_seq)
        folds = build_8fold_masks(unit_key, L, n_folds=8)
        n_folds = len(folds)

        # Build (n_folds + 1, len(full_ids)) batch: rows 0..n_folds-1 = 8-fold masks; last row = full target mask
        base = torch.tensor(full_ids, dtype=torch.long, device=self.device)
        batch = base.unsqueeze(0).repeat(n_folds + 1, 1)
        for fi, pos_list in enumerate(folds):
            for p in pos_list:
                batch[fi, target_start + p] = self.mask_id
        batch[n_folds, target_start:target_end] = self.mask_id

        pll8_nll_sum = 0.0
        full_nll_sum = 0.0
        bs = self.fold_batch_size
        all_positions = list(range(L))
        for start in range(0, n_folds + 1, bs):
            stop = min(start + bs, n_folds + 1)
            logits = self._forward_logits_batch(batch[start:stop])
            for local_i, row_idx in enumerate(range(start, stop)):
                if row_idx < n_folds:
                    pll8_nll_sum += self._score_masked_positions_from_logits(
                        logits[local_i], folds[row_idx], target_seq, target_start
                    )
                else:
                    full_nll_sum = self._score_masked_positions_from_logits(
                        logits[local_i], all_positions, target_seq, target_start
                    )
            del logits

        if not (np.isfinite(pll8_nll_sum) and np.isfinite(full_nll_sum)):
            raise FloatingPointError(f"{self.arm}: non-finite bidirectional NLL sum on {unit_key}")
        return {
            "total_tokens": len(full_ids),
            "target_length": L,
            "k_items": k_items,
            "n_folds": n_folds,
            "pll_8fold": {
                "nll_sum": pll8_nll_sum,
                "nll_per_residue": pll8_nll_sum / L,
            },
            "full_target_masked": {
                "nll_sum": full_nll_sum,
                "nll_per_residue": full_nll_sum / L,
            },
        }

    @torch.no_grad()
    def score_condition_exact_loo(
        self,
        target_seq: str,
        context_items: Sequence[str],
        *,
        budget_tokens: int,
        batch_size: int = 16,
    ) -> float:
        """Exact 1-by-1 leave-one-out target PLL inside the same unaligned context (for diagnostic check)."""
        tgt_ids = self._encode_residues(target_seq)
        ctx_ids = [self._encode_residues(s) for s in context_items]
        full_ids, target_start, _target_end, _k_items = pack_bidirectional_context(
            tgt_ids,
            ctx_ids,
            bos_prefix=self.bos_prefix,
            eos_suffix=self.eos_suffix,
            sep_ids=self.sep_ids,
            budget_tokens=budget_tokens,
        )
        L = len(target_seq)
        base = torch.tensor(full_ids, dtype=torch.long, device=self.device)
        nll_sum = 0.0
        bs = max(1, int(batch_size))
        for start in range(0, L, bs):
            stop = min(start + bs, L)
            batch = base.unsqueeze(0).repeat(stop - start, 1)
            for idx, p in enumerate(range(start, stop)):
                batch[idx, target_start + p] = self.mask_id
            logits = self._forward_logits_batch(batch)
            for idx, p in enumerate(range(start, stop)):
                nll_sum += self._score_masked_positions_from_logits(
                    logits[idx], [p], target_seq, target_start
                )
            del logits
        return float(nll_sum)

    def provenance(self) -> dict[str, Any]:
        if self.is_esm3:
            return {
                "arm": self.arm,
                "group": "group_b_bidirectional",
                "loader": "esm3_offline_seqonly",
                "dtype_requested": self.dtype_name,
                "dtype_realised": str(self.esm3.dtype),
                "bos_prefix": self.bos_prefix,
                "eos_suffix": self.eos_suffix,
                "sep_ids": self.sep_ids,
                "mask_id": self.mask_id,
                "device": self.device,
            }
        assert self.adapter is not None
        return {
            **self.adapter.provenance(),
            "group": "group_b_bidirectional",
            "bos_prefix": self.bos_prefix,
            "eos_suffix": self.eos_suffix,
            "sep_ids": self.sep_ids,
            "mask_id": self.mask_id,
        }


# ============================================================================
# Parity Helper for Cached Causal Arm (progen2-small)
# ============================================================================


def _run_cached_causal_parity(
    arm: str,
    units: Sequence[dict[str, Any]],
    records: Sequence[str],
    related_map: Any,
    *,
    filler_idx: int,
    filler_seq: str,
    device: str,
    dtype_name: str,
    budget_tokens: int,
) -> dict[str, Any]:
    """Score a cached protein causal arm (``progen2-small``) via core ``arms.load_arm`` for parity verification."""
    from src.capability.core import arms as core_arms

    loaded = core_arms.load_arm(arm, device=device, dtype=dtype_name)
    model = loaded.model
    tok = loaded.tokenizer

    def _encode_pg2(seq: str) -> list[int]:
        return [int(v) for v in tok.encode(f"1{seq}2", add_special_tokens=False)]

    rows: list[dict[str, Any]] = []
    for unit in units:
        target_seq, cond_inputs = build_unit_condition_inputs(
            unit, records, related_map, filler_idx=filler_idx, filler_seq=filler_seq
        )
        t_ids = _encode_pg2(target_seq)
        cond_scores: dict[str, Any] = {}
        for cond in CONDITIONS:
            prefix: list[int] = []
            k_used = 0
            rem = budget_tokens - len(t_ids)
            for item in cond_inputs[cond]:
                c_ids = _encode_pg2(item)
                if len(prefix) + len(c_ids) > rem:
                    break
                prefix.extend(c_ids)
                k_used += 1
            full_ids = prefix + t_ids
            span_start = len(prefix) + 1  # content after '1'
            span_end = len(prefix) + len(t_ids) - 1  # exclude trailing '2'
            inp = torch.tensor([full_ids], dtype=torch.long, device=device)
            with torch.no_grad(), _precision_ctx(device, dtype_name):
                logits = model(input_ids=inp).logits[0, span_start - 1 : span_end - 1, :].float()
                targets = inp[0, span_start:span_end]
                nll_sum = float(F.cross_entropy(logits, targets, reduction="sum").item())
            cond_scores[cond] = {
                "nll_sum": nll_sum,
                "nll_per_residue": nll_sum / len(target_seq),
                "scored_tokens": span_end - span_start,
                "total_tokens": len(full_ids),
                "k_items": k_used,
            }
        nll_hom = cond_scores["homologue"]["nll_sum"]
        nll_unr = cond_scores["unrelated"]["nll_sum"]
        nll_shuf = cond_scores["mono_shuffled"]["nll_sum"]
        nll_pos = cond_scores["position_only"]["nll_sum"]
        rows.append(
            {
                "key": unit["key"],
                "band": unit["band"],
                "stratum": unit["stratum"],
                "group": int(unit["group"]),
                "target_length": len(target_seq),
                "k_homologue": cond_scores["homologue"]["k_items"],
                "conditions": cond_scores,
                "hom_beats_unrelated": 1.0 if nll_hom < nll_unr else (0.5 if nll_hom == nll_unr else 0.0),
                "hom_beats_mono_shuffled": 1.0 if nll_hom < nll_shuf else (0.5 if nll_hom == nll_shuf else 0.0),
                "fractional_reduction_vs_unrelated": (nll_unr - nll_hom) / nll_pos if nll_pos > 0 else None,
                "fractional_reduction_vs_shuffled": (nll_shuf - nll_hom) / nll_pos if nll_pos > 0 else None,
            }
        )
    return {
        "provenance": {"arm": arm, "group": "parity_cached_causal", "dtype_requested": dtype_name, "device": device},
        "units": rows,
        "summary": summarize_unit_rows(rows),
    }


def _compute_contrasts_from_nlls(
    nll_hom: float, nll_unr: float, nll_shuf: float, nll_pos: float
) -> dict[str, Any]:
    return {
        "hom_beats_unrelated": 1.0 if nll_hom < nll_unr else (0.5 if nll_hom == nll_unr else 0.0),
        "hom_beats_mono_shuffled": 1.0 if nll_hom < nll_shuf else (0.5 if nll_hom == nll_shuf else 0.0),
        "fractional_reduction_vs_unrelated": (nll_unr - nll_hom) / nll_pos if nll_pos > 0 else None,
        "fractional_reduction_vs_shuffled": (nll_shuf - nll_hom) / nll_pos if nll_pos > 0 else None,
    }


def _select_stratified_diagnostic_indices(units: Sequence[dict[str, Any]], n_total: int) -> set[int]:
    """Select ``min(n_total, len(units))`` stratified unit indices balanced across present bands."""
    if n_total <= 0 or not units:
        return set()
    target = min(int(n_total), len(units))
    present_bands = [b for b in IDENTITY_BANDS if any(u["band"] == b for u in units)]
    per_band = max(1, target // max(1, len(present_bands)))
    picked: set[int] = set()
    for band in present_bands:
        count = 0
        for idx, u in enumerate(units):
            if u["band"] == band and idx not in picked:
                picked.add(idx)
                count += 1
                if count >= per_band or len(picked) >= target:
                    break
    if len(picked) < target:
        for idx in range(len(units)):
            if idx not in picked:
                picked.add(idx)
                if len(picked) >= target:
                    break
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ALL_SUPPORTED_ARMS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16"))
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--budget-tokens", type=int, default=1024)
    parser.add_argument("--fold-batch-size", type=int, default=9)
    parser.add_argument("--stratum", default="retained", choices=("retained", "all"))
    parser.add_argument("--max-units", type=int, default=0, help="0 = all units in selected stratum")
    parser.add_argument(
        "--diagnostic-exact-pll-units",
        type=int,
        default=0,
        help="Optional stratified N units (e.g. 32 on esm2-150m) to compare pll_8fold vs 1-by-1 exact_pll",
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=DATA_ROOT / "results/R5/context_homologue_20260826/cohort.json",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"r4_unaligned_{args.arm}.json"
    if args.skip_existing and out_file.is_file() and out_file.with_name(out_file.name + ".sha256").is_file():
        print(f"[R4-Score:{args.arm}] skipping existing {out_file}", flush=True)
        return

    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    prot = cohort["protein"]
    records: list[str] = prot["records"]
    filler_idx = int(prot["filler"]) if isinstance(prot["filler"], int) else int(prot["filler"].get("index", 0))
    filler_seq = (
        records[filler_idx]
        if isinstance(prot["filler"], int)
        else str(prot["filler"].get("record") or prot["filler"].get("sequence") or records[filler_idx])
    )
    related_map = prot.get("related", {})

    units = prot["units"]
    if args.stratum != "all":
        units = [u for u in units if u.get("stratum") == args.stratum]
    if args.max_units > 0:
        units = units[: args.max_units]
    if not units:
        raise SystemExit("No R4 units selected")

    dtype_name, torch_dt = _resolve_torch_dtype(args.arm, args.dtype)
    t0 = time.monotonic()

    if args.arm == "progen2-small":
        res = _run_cached_causal_parity(
            args.arm,
            units,
            records,
            related_map,
            filler_idx=filler_idx,
            filler_seq=filler_seq,
            device=args.device,
            dtype_name=dtype_name,
            budget_tokens=args.budget_tokens,
        )
        payload = {
            "schema": SCHEMA_ARM_RUNNER,
            "arm": args.arm,
            "cohort_sha256": sha256_file(args.cohort),
            "budget_tokens": args.budget_tokens,
            "stratum": args.stratum,
            "n_units": len(res["units"]),
            "elapsed_seconds": time.monotonic() - t0,
            "provenance": res["provenance"],
            "bands": res["summary"]["bands"],
            "pooled": res["summary"]["pooled"],
            "units": res["units"],
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
    elif args.arm in GROUP_A_CAUSAL_TEXT_ARMS:
        scorer = _CausalTextR4Scorer(
            args.arm,
            device=args.device,
            dtype_name=dtype_name,
            torch_dt=torch_dt,
            model_root=args.model_root,
        )
        unit_rows: list[dict[str, Any]] = []
        for u_idx, unit in enumerate(units, 1):
            target_seq, cond_inputs = build_unit_condition_inputs(
                unit, records, related_map, filler_idx=filler_idx, filler_seq=filler_seq
            )
            cond_scores: dict[str, Any] = {}
            scored_counts: set[int] = set()
            for cond in CONDITIONS:
                cs = scorer.score_condition(target_seq, cond_inputs[cond], budget_tokens=args.budget_tokens)
                cond_scores[cond] = cs
                scored_counts.add(int(cs["scored_tokens"]))
            if len(scored_counts) != 1:
                raise AssertionError(
                    f"{args.arm} {unit['key']}: target scored_tokens varied across conditions: {scored_counts}"
                )
            contrasts = _compute_contrasts_from_nlls(
                cond_scores["homologue"]["nll_sum"],
                cond_scores["unrelated"]["nll_sum"],
                cond_scores["mono_shuffled"]["nll_sum"],
                cond_scores["position_only"]["nll_sum"],
            )
            unit_rows.append(
                {
                    "key": unit["key"],
                    "band": unit["band"],
                    "stratum": unit["stratum"],
                    "group": int(unit["group"]),
                    "target_length": len(target_seq),
                    "k_homologue": int(cond_scores["homologue"]["k_items"]),
                    "conditions": cond_scores,
                    **contrasts,
                }
            )
            if u_idx % 50 == 0 or u_idx == len(units):
                print(
                    f"[R4-Score:{args.arm}] {u_idx}/{len(units)} units ({time.monotonic() - t0:.1f}s)",
                    flush=True,
                )
        summary = summarize_unit_rows(unit_rows)
        payload = {
            "schema": SCHEMA_ARM_RUNNER,
            "arm": args.arm,
            "cohort_sha256": sha256_file(args.cohort),
            "budget_tokens": args.budget_tokens,
            "stratum": args.stratum,
            "n_units": len(unit_rows),
            "elapsed_seconds": time.monotonic() - t0,
            "provenance": scorer.provenance(),
            "bands": summary["bands"],
            "pooled": summary["pooled"],
            "units": unit_rows,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
    else:
        scorer_b = _BidirectionalR4Scorer(
            args.arm,
            device=args.device,
            dtype_name=dtype_name,
            model_root=args.model_root,
            fold_batch_size=args.fold_batch_size,
        )
        diag_indices = _select_stratified_diagnostic_indices(units, args.diagnostic_exact_pll_units)
        diag_rows: list[dict[str, Any]] = []
        unit_rows_b: list[dict[str, Any]] = []
        p8_flat: list[dict[str, Any]] = []
        fm_flat: list[dict[str, Any]] = []

        for u_idx, unit in enumerate(units, 1):
            target_seq, cond_inputs = build_unit_condition_inputs(
                unit, records, related_map, filler_idx=filler_idx, filler_seq=filler_seq
            )
            cond_scores = {
                cond: scorer_b.score_condition(
                    unit["key"], target_seq, cond_inputs[cond], budget_tokens=args.budget_tokens
                )
                for cond in CONDITIONS
            }
            p8_c = _compute_contrasts_from_nlls(
                cond_scores["homologue"]["pll_8fold"]["nll_sum"],
                cond_scores["unrelated"]["pll_8fold"]["nll_sum"],
                cond_scores["mono_shuffled"]["pll_8fold"]["nll_sum"],
                cond_scores["position_only"]["pll_8fold"]["nll_sum"],
            )
            fm_c = _compute_contrasts_from_nlls(
                cond_scores["homologue"]["full_target_masked"]["nll_sum"],
                cond_scores["unrelated"]["full_target_masked"]["nll_sum"],
                cond_scores["mono_shuffled"]["full_target_masked"]["nll_sum"],
                cond_scores["position_only"]["full_target_masked"]["nll_sum"],
            )
            row = {
                "key": unit["key"],
                "band": unit["band"],
                "stratum": unit["stratum"],
                "group": int(unit["group"]),
                "target_length": len(target_seq),
                "k_homologue": int(cond_scores["homologue"]["k_items"]),
                "conditions": cond_scores,
                "pll_8fold": p8_c,
                "full_target_masked": fm_c,
            }
            unit_rows_b.append(row)
            p8_flat.append(
                {
                    "key": unit["key"],
                    "band": unit["band"],
                    "group": int(unit["group"]),
                    "k_homologue": int(cond_scores["homologue"]["k_items"]),
                    **p8_c,
                }
            )
            fm_flat.append(
                {
                    "key": unit["key"],
                    "band": unit["band"],
                    "group": int(unit["group"]),
                    "k_homologue": int(cond_scores["homologue"]["k_items"]),
                    **fm_c,
                }
            )

            if (u_idx - 1) in diag_indices:
                exact_nlls = {
                    cond: scorer_b.score_condition_exact_loo(
                        target_seq, cond_inputs[cond], budget_tokens=args.budget_tokens, batch_size=16
                    )
                    for cond in CONDITIONS
                }
                exact_c = _compute_contrasts_from_nlls(
                    exact_nlls["homologue"],
                    exact_nlls["unrelated"],
                    exact_nlls["mono_shuffled"],
                    exact_nlls["position_only"],
                )
                diag_rows.append(
                    {
                        "key": unit["key"],
                        "band": unit["band"],
                        "target_length": len(target_seq),
                        "pll_8fold_nll_per_res": {
                            c: cond_scores[c]["pll_8fold"]["nll_per_residue"] for c in CONDITIONS
                        },
                        "exact_loo_nll_per_res": {c: exact_nlls[c] / len(target_seq) for c in CONDITIONS},
                        "pll_8fold_frac_vs_unrelated": p8_c["fractional_reduction_vs_unrelated"],
                        "exact_loo_frac_vs_unrelated": exact_c["fractional_reduction_vs_unrelated"],
                        "pll_8fold_hom_beats_unrelated": p8_c["hom_beats_unrelated"],
                        "exact_loo_hom_beats_unrelated": exact_c["hom_beats_unrelated"],
                    }
                )

            if u_idx % 50 == 0 or u_idx == len(units):
                print(
                    f"[R4-Score:{args.arm}] {u_idx}/{len(units)} units ({time.monotonic() - t0:.1f}s)",
                    flush=True,
                )

        diag_summary: dict[str, Any] | None = None
        if diag_rows:
            from scipy.stats import pearsonr, spearmanr

            f8 = np.asarray([r["pll_8fold_frac_vs_unrelated"] for r in diag_rows], dtype=np.float64)
            fe = np.asarray([r["exact_loo_frac_vs_unrelated"] for r in diag_rows], dtype=np.float64)
            a8 = np.asarray([r["pll_8fold_hom_beats_unrelated"] for r in diag_rows], dtype=np.float64)
            ae = np.asarray([r["exact_loo_hom_beats_unrelated"] for r in diag_rows], dtype=np.float64)
            nll8 = np.asarray(
                [r["pll_8fold_nll_per_res"][c] for r in diag_rows for c in CONDITIONS], dtype=np.float64
            )
            nlle = np.asarray(
                [r["exact_loo_nll_per_res"][c] for r in diag_rows for c in CONDITIONS], dtype=np.float64
            )
            diag_summary = {
                "n_units": len(diag_rows),
                "nll_per_res_pearson_r": float(pearsonr(nll8, nlle).statistic) if len(nll8) >= 2 else None,
                "nll_per_res_spearman_rho": float(spearmanr(nll8, nlle).statistic) if len(nll8) >= 2 else None,
                "nll_per_res_mean_abs_diff": float(np.mean(np.abs(nll8 - nlle))),
                "frac_reduction_pearson_r": float(pearsonr(f8, fe).statistic) if len(f8) >= 2 else None,
                "frac_reduction_spearman_rho": float(spearmanr(f8, fe).statistic) if len(f8) >= 2 else None,
                "frac_reduction_mean_abs_diff": float(np.mean(np.abs(f8 - fe))),
                "auroc_pll_8fold_mean": float(np.mean(a8)),
                "auroc_exact_loo_mean": float(np.mean(ae)),
                "auroc_unit_agreement_rate": float(np.mean(a8 == ae)),
                "rows": diag_rows,
            }

        p8_sum = summarize_unit_rows(p8_flat)
        fm_sum = summarize_unit_rows(fm_flat)
        payload = {
            "schema": SCHEMA_ARM_RUNNER,
            "arm": args.arm,
            "cohort_sha256": sha256_file(args.cohort),
            "budget_tokens": args.budget_tokens,
            "stratum": args.stratum,
            "n_units": len(unit_rows_b),
            "elapsed_seconds": time.monotonic() - t0,
            "provenance": scorer_b.provenance(),
            "bands": p8_sum["bands"],
            "pooled": p8_sum["pooled"],
            "companion_full_target_masked_bands": fm_sum["bands"],
            "companion_full_target_masked_pooled": fm_sum["pooled"],
            "exact_pll_diagnostic": diag_summary,
            "units": unit_rows_b,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }

    # Guard against forbidden infra identifiers
    raw_txt = json.dumps(payload, sort_keys=True)
    if FORBIDDEN_INFRA_RE.search(raw_txt):
        raise ValueError("Refusing to write R4 output containing forbidden pod/node identifier")

    atomic_write_json(out_file, payload)
    digest = sha256_file(out_file)
    out_file.with_name(out_file.name + ".sha256").write_text(
        f"{digest}  {out_file.name}\n", encoding="utf-8"
    )
    print(
        f"[R4-Score:{args.arm}] DONE in {payload['elapsed_seconds']:.1f}s -> {out_file} "
        f"(sha256={digest[:16]}..., pooled AUROC={payload['pooled']['auroc_hom_vs_unrelated']['mean']:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
