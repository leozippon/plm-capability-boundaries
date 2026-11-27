#!/usr/bin/env python3
"""Unified shardable H200 GPU scoring runner for ALL 15 new models
(4 DPLM-2 checkpoints in BOTH ``seq_only`` and ``struct_conditioned`` modes +
11 ProteinMPNN checkpoints: vanilla, soluble, CA-only = 19 scored entries)
across ALL scoring benchmarks in Results 1-5:

  - ``r1``      : 201-assay ProteinGym DMS ladder (M + 1024-d R)
  - ``r1_fc``   : 68-family conditionals (4,075 contexts; DPLM-2 seq-only native)
  - ``r1_fd``   : 68-family distribution exact O(L) PLL (native + column/comp nulls)
  - ``r2``      : MegaScale (101 bgs, 25,856 vars) + Domainome (428 doms, 109,568 vars) (M + 1024-d R)
  - ``r3``      : MegaScale double-mutant epistasis (64 bgs, 12,977 states, M_cycle + 1024-d R_cycle)
  - ``e01``     : 30 admitted structural E01 contact-propagation archives (position_likelihood_v1)
  - ``r4_main`` : Result 5 Main Table (800 retained units, 5 conditions)
  - ``r5_dms``  : Result 5 Downstream 1 DMS homolog-context ranking (1A rescue_30_80 + 1B e09_five_band)

Strictly uses leakage-free wild-type 3D backbones from ``wt_backbones_3d`` and NEVER
modifies any already-sealed ``*_62`` / ``*_65`` artifact.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("TRANSFER_DATA_ROOT") or ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.amino_acids import AA20
from src.capability.core.io import sha256_file, write_json
from src.capability.extensions import family_conditionals as fc_mod
from src.capability.extensions import family_distribution as fd_mod
from src.capability.general_plm.adapters import apply_mutation
from src.capability.general_plm.eval_utils import (
    atomic_write_json,
    cluster_bootstrap_mean,
    safe_spearman,
)
from src.capability.position.contact_response import rebuild_states
from src.capability.position.position_likelihood import (
    MASKED,
    PREFIX_INVARIANT_RULE,
    SCHEMA as POS_SCHEMA,
    TERM_SEMANTICS,
    TIER2_NATS,
    TIER2_REPEAT_MULTIPLE,
    PackedState,
    PositionArchive,
    PositionScore,
    repeat_residual,
    residual_tier,
)
from src.capability.position.position_terms import ALIGNMENT, alignment, blas_pinning, retention_residual
from src.capability.unified import r4_scoring_yardstick as r4_mod
from src.capability.unified import r5_dms_context_yardstick as r5_dms_mod
from src.capability.unified import wt_backbones_3d as bb_mod
from src.capability.unified.yardstick_77_80_scoring import (
    DEFAULT_NEW_SCORING_DIR,
    DIMA_4_ARMS,
    DPLM2_4_ARMS,
    DPLM2_4_STRUCT_ARMS,
    EXTENDED_ARM_METADATA,
    NEW_23_SCORED_ENTRIES,
    PROTEINMPNN_11_ARMS,
    arm_slug,
    assert_no_forbidden_infra_strings,
)

AA20_INDEX = {aa: idx for idx, aa in enumerate(AA20)}
ALL_STAGES: tuple[str, ...] = ("r1", "r1_fc", "r1_fd", "r2", "r3", "e01", "r4_main", "r5_dms")

MODEL_GROUPS: Mapping[str, tuple[str, ...]] = {
    "all": NEW_23_SCORED_ENTRIES,
    "dplm2_seq": DPLM2_4_ARMS,
    "dplm2_struct": DPLM2_4_STRUCT_ARMS,
    "dplm2_all": DPLM2_4_ARMS + DPLM2_4_STRUCT_ARMS,
    "proteinmpnn": PROTEINMPNN_11_ARMS,
    "proteinmpnn_v": tuple(a for a in PROTEINMPNN_11_ARMS if a.startswith("proteinmpnn-v-")),
    "proteinmpnn_s": tuple(a for a in PROTEINMPNN_11_ARMS if a.startswith("proteinmpnn-s-")),
    "proteinmpnn_ca": tuple(a for a in PROTEINMPNN_11_ARMS if a.startswith("proteinmpnn-ca-")),
    "dima": DIMA_4_ARMS,
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_model_list(spec: str) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        items = MODEL_GROUPS.get(token)
        if items is None:
            if token not in NEW_23_SCORED_ENTRIES:
                raise ValueError(
                    f"Unknown model/group {token!r}; expected one of {list(MODEL_GROUPS)} "
                    f"or {list(NEW_23_SCORED_ENTRIES)}"
                )
            items = (token,)
        for item in items:
            if item not in seen:
                seen.add(item)
                selected.append(item)
    return selected


def _resolve_stage_dtype(arm_id: str, stage: str, dtype_arg: str) -> str:
    raw = dtype_arg.strip().lower()
    if raw not in ("", "auto", "default"):
        return raw
    base = arm_id.split("|")[0]
    # ProteinMPNN (1.66M), DPLM-2 <= 650M, and small DiMA (8M/35M) always run FP32.
    # 3B-backbone models (dplm2-3b, dima-esm2-3b, dima-cheap-1024) use FP32 on precision-sensitive
    # E01/R3 and BF16 on high-volume DMS/R2/R4/R5.
    if base in ("dplm2-3b", "dima-esm2-3b", "dima-cheap-1024") and stage in (
        "r1",
        "r1_fc",
        "r1_fd",
        "r2",
        "r4_main",
        "r5_dms",
    ):
        return "bfloat16"
    return "float32"


# ============================================================================
# Backbone Cache Wrapper (strictly leakage-free wild-type / template lookup)
# ============================================================================


class BackboneStore:
    """In-memory LRU wrapper over ``wt_backbones_3d.load_backbone_npz``."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._cache: dict[str, dict[str, Any]] = {}

    def has(self, sequence: str) -> bool:
        digest = bb_mod.seq_sha256(sequence)
        if digest in self._cache:
            return True
        return (self.cache_dir / bb_mod.backbone_npz_relpath(digest)).is_file()

    def get(self, sequence: str) -> dict[str, Any]:
        digest = bb_mod.seq_sha256(sequence)
        hit = self._cache.get(digest)
        if hit is not None:
            return hit
        rec = bb_mod.load_backbone_npz(self.cache_dir, digest)
        if rec["sequence"] != sequence.strip().upper():
            raise ValueError(
                f"Backbone sequence mismatch for {digest}: cached len={len(rec['sequence'])} "
                f"vs query len={len(sequence)}"
            )
        self._cache[digest] = rec
        return rec

    def get_optional(self, sequence: str) -> dict[str, Any] | None:
        return self.get(sequence) if self.has(sequence) else None


# ============================================================================
# Unified Adapter Bridge (binds cleanly to dplm2_adapter & proteinmpnn_adapter)
# ============================================================================


class ScoringAdapterBridge:
    """Thin uniform scoring interface wrapping ``DPLM2Adapter`` and ``ProteinMPNNAdapter``.

    Binds directly to ``src.capability.unified.dplm2_adapter.load_dplm2_adapter`` and
    ``src.capability.unified.proteinmpnn_adapter.load_proteinmpnn_adapter`` so every
    benchmark stage consumes one clean contract across DPLM-2 (seq_only / struct_conditioned)
    and ProteinMPNN (vanilla / soluble / CA-only).
    """

    def __init__(
        self,
        arm_id: str,
        *,
        device: str,
        dtype: str,
        model_root: Path | None,
        dima_root: Path | None = None,
    ) -> None:
        self.arm_id = arm_id
        self.base_arm_id = arm_id.split("|")[0]
        self.meta = EXTENDED_ARM_METADATA[arm_id]
        self.is_mpnn = self.meta.paradigm == "inverse_folding_gnn"
        self.is_dima = self.meta.paradigm == "continuous_latent_diffusion"
        self.is_dplm2_struct = arm_id.endswith("|struct")
        self.requires_backbone = self.is_mpnn or self.is_dplm2_struct
        self.device = device
        self.dtype = dtype
        self.model_root = model_root
        self.dima_root = dima_root

        if self.is_mpnn:
            from src.capability.unified.proteinmpnn_adapter import load_proteinmpnn_adapter

            self.impl = load_proteinmpnn_adapter(
                self.base_arm_id,
                model_root=model_root,
                device=device,
                dtype=dtype,
            )
        elif self.is_dima:
            from src.capability.unified.dima_adapter import load_dima_adapter

            self.impl = load_dima_adapter(
                self.base_arm_id,
                dima_root=dima_root,
                model_root=model_root,
                device=device,
                dtype=dtype,
            )
        else:
            from src.capability.unified.dplm2_adapter import load_dplm2_adapter

            mode = "struct_conditioned" if self.is_dplm2_struct else "seq_only"
            self.impl = load_dplm2_adapter(
                self.base_arm_id,
                mode=mode,
                model_root=model_root,
                device=device,
                dtype=dtype,
                load_struct_tokenizer=self.is_dplm2_struct,
            )

    def _dplm2_struct_kwargs(self, bb: dict[str, Any] | None) -> dict[str, Any]:
        if not self.is_dplm2_struct:
            return {}
        if bb is None:
            raise ValueError(f"{self.arm_id}: requires 3D backbone coordinates/tokens, got None")
        if "dplm2_struct_tokens" in bb:
            return {"struct_tokens": bb["dplm2_struct_tokens"]}
        return {"coords": bb["coords"]}

    def _safe_bs(self, seq_len: int, requested: int) -> int:
        """Bound batch size by token count to prevent OOM on long (>500 aa) chains."""
        req = max(1, int(requested))
        if self.is_mpnn:
            return req
        if self.is_dima:
            eff_len = max(1, min(int(seq_len), 510) + 2)
            cap = 2560 if self.base_arm_id in ("dima-esm2-3b", "dima-cheap-1024") else 8192
            return max(1, min(req, cap // eff_len))
        # Paired sequence+structure length is 2 * (L + 2); keep per-step paired tokens well inside VRAM
        paired_len = max(1, 2 * (int(seq_len) + 2))
        if self.base_arm_id == "dplm2-3b":
            cap_tokens = 2560
        elif "650m" in self.base_arm_id:
            cap_tokens = 4096
        else:
            cap_tokens = 6144
        return max(1, min(req, cap_tokens // paired_len))

    def provenance(self) -> dict[str, Any]:
        prov = dict(self.impl.provenance())
        prov.update(
            {
                "arm_id": self.arm_id,
                "base_arm_id": self.base_arm_id,
                "paradigm": self.meta.paradigm,
                "conditioning_regime": self.meta.conditioning_regime,
                "dtype_requested": self.dtype,
            }
        )
        return prov

    def score_variants(
        self,
        wt: str,
        mutations: Sequence[str],
        *,
        bb: dict[str, Any] | None = None,
        batch_size: int = 32,
        multi_exact_pll: bool = False,
    ) -> np.ndarray:
        if self.is_mpnn:
            if bb is None:
                raise ValueError(f"{self.arm_id}: requires coords for score_variants")
            vals = self.impl.score_variants(
                wt, list(mutations), bb["coords"], batch_size=batch_size, multi_permutations=4
            )
        elif self.is_dima:
            eff_bs = self._safe_bs(len(wt), batch_size)
            vals = self.impl.score_mutations(
                wt, list(mutations), batch_size=eff_bs, multi_exact_pll=multi_exact_pll
            )
        else:
            eff_bs = self._safe_bs(len(wt), batch_size)
            vals = self.impl.score_variants(
                wt,
                list(mutations),
                batch_size=eff_bs,
                multi_exact_pll=multi_exact_pll,
                **self._dplm2_struct_kwargs(bb),
            )
        arr = np.asarray(vals, dtype=np.float64)
        if arr.shape != (len(mutations),) or not np.isfinite(arr).all():
            raise FloatingPointError(f"{self.arm_id}: invalid variant scores shape={arr.shape}")
        return arr

    def extract_hidden_states_projected(
        self,
        sequences: Sequence[str],
        *,
        bb: dict[str, Any] | None = None,
        batch_size: int = 16,
    ) -> np.ndarray:
        """Return ``(N, 1024)`` float32 projected hidden states (``0:512 = R_mid``, ``512:1024 = R_last``)."""
        if self.is_mpnn:
            if bb is None:
                raise ValueError(f"{self.arm_id}: requires coords for extract_hidden_states")
            proj = self.impl.extract_hidden_states(
                list(sequences), bb["coords"], batch_size=batch_size, return_projected_1024=True
            )
            return np.asarray(proj, dtype=np.float32)

        if self.is_dima:
            max_len = max((len(s) for s in sequences), default=1)
            eff_bs = self._safe_bs(max_len, batch_size)
            return np.asarray(
                self.impl.extract_readout_1024(list(sequences), batch_size=eff_bs), dtype=np.float32
            )

        kw: dict[str, Any] = {}
        if self.is_dplm2_struct:
            if bb is None:
                raise ValueError(f"{self.arm_id}: requires backbone for struct_conditioned hidden states")
            if "dplm2_struct_tokens" in bb:
                kw["struct_tokens_list"] = bb["dplm2_struct_tokens"]
            else:
                kw["coords_list"] = bb["coords"]
        max_len = max((len(s) for s in sequences), default=1)
        eff_bs = self._safe_bs(max_len, batch_size)
        proj = self.impl.extract_hidden_states(
            list(sequences), batch_size=eff_bs, return_projected_1024=True, **kw
        )
        return np.asarray(proj, dtype=np.float32)

    def score_r3_states_batch(
        self,
        sequences: Sequence[str],
        *,
        bb: dict[str, Any] | None = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return per-state log-likelihood / PLL (higher = more probable) for R3 double-mutant cycles."""
        if self.is_mpnn:
            if bb is None:
                raise ValueError(f"{self.arm_id}: requires coords for R3 states")
            return np.asarray(
                self.impl.exact_sequence_ll_batch(
                    list(sequences), bb["coords"], n_permutations=4, batch_size=batch_size
                ),
                dtype=np.float64,
            )
        if self.is_dima:
            out = np.empty(len(sequences), dtype=np.float64)
            for idx, seq in enumerate(sequences):
                eff_bs = self._safe_bs(len(seq), batch_size)
                out[idx] = float(self.impl.exact_pll(seq, batch_size=eff_bs).pll)
            return out

        kw = self._dplm2_struct_kwargs(bb)
        out = np.empty(len(sequences), dtype=np.float64)
        for idx, seq in enumerate(sequences):
            eff_bs = self._safe_bs(len(seq), batch_size)
            out[idx] = float(self.impl.exact_pll(seq, batch_size=eff_bs, **kw).pll)
        return out

    def exact_pll_terms(
        self,
        sequence: str,
        *,
        bb: dict[str, Any] | None = None,
        batch_size: int = 32,
        return_gathered: bool = False,
    ) -> tuple[np.ndarray, float, np.ndarray | None]:
        """Return ``(per_pos_nll_terms (L,), nll_sum, optional_aa20_logp (L, 20))``."""
        if self.is_mpnn:
            if bb is None:
                raise ValueError(f"{self.arm_id}: requires coords for exact_pll")
            if return_gathered:
                from src.capability.unified.proteinmpnn_adapter import _encode_mpnn_seq

                lp = self.impl.site_conditional_logprobs(
                    sequence, bb["coords"], positions=None, batch_size=batch_size
                )
                seq_idx = _encode_mpnn_seq(sequence)
                terms = np.maximum((-lp[np.arange(len(sequence)), seq_idx]).astype(np.float32), 0.0)
                return terms.astype(np.float64), float(terms.sum(dtype=np.float64)), lp.astype(np.float32)
            res = self.impl.exact_pll(sequence, bb["coords"], batch_size=batch_size)
            return np.asarray(res.terms, dtype=np.float64), float(res.nll_sum), None

        if self.is_dima:
            eff_bs = self._safe_bs(len(sequence), batch_size)
            if return_gathered:
                lp = self.impl.site_conditional_logprobs(sequence, positions=None, batch_size=eff_bs)
                aa_idx = np.asarray([AA20_INDEX[ch] for ch in sequence], dtype=np.int64)
                terms = np.maximum((-lp[np.arange(len(sequence)), aa_idx]).astype(np.float32), 0.0)
                return terms.astype(np.float64), float(terms.sum(dtype=np.float64)), lp.astype(np.float32)
            res = self.impl.exact_pll(sequence, batch_size=eff_bs)
            return np.asarray(res.terms, dtype=np.float64), float(res.nll_sum), None

        kw = self._dplm2_struct_kwargs(bb)
        eff_bs = self._safe_bs(len(sequence), batch_size)
        if return_gathered:
            lp = self.impl.site_conditional_logprobs(sequence, positions=None, batch_size=eff_bs, **kw)
            aa_idx = np.asarray([AA20_INDEX[ch] for ch in sequence], dtype=np.int64)
            terms = np.maximum((-lp[np.arange(len(sequence)), aa_idx]).astype(np.float32), 0.0)
            return terms.astype(np.float64), float(terms.sum(dtype=np.float64)), lp.astype(np.float32)
        res = self.impl.exact_pll(sequence, batch_size=eff_bs, **kw)
        return np.asarray(res.terms, dtype=np.float64), float(res.nll_sum), None

    @torch.no_grad()
    def pll_8fold_nll_sum(
        self,
        sequence: str,
        *,
        bb: dict[str, Any] | None = None,
        n_folds: int = 8,
    ) -> float:
        """Fast O(1)-pass deterministic 8-fold stride-masked PLL NLL sum (chunked via ``_safe_bs``)."""
        if self.is_mpnn:
            if bb is None:
                raise ValueError(f"{self.arm_id}: requires coords for pll_8fold_nll_sum")
            return float(-self.impl.exact_sequence_ll(sequence, bb["coords"], n_permutations=4))
        if self.is_dima:
            return float(self.impl.pll_8fold(sequence, n_folds=n_folds).nll_sum)

        import torch.nn.functional as F
        from src.capability.unified.dplm2_adapter import AA_MASK_ID, _encode_aa_to_ids

        length = len(sequence)
        aa_ids = _encode_aa_to_ids(sequence)
        kw = self._dplm2_struct_kwargs(bb)
        s_ids, single_mod = self.impl._resolve_struct_ids(length, **kw)
        base_paired, half = self.impl._build_paired_ids([aa_ids], [s_ids])

        k = min(int(n_folds), length)
        fold_positions = [list(range(f, length, k)) for f in range(k)]
        chunk_bs = self._safe_bs(length, k)

        nll_total = 0.0
        with self.impl._precision_ctx():
            for start in range(0, k, chunk_bs):
                stop = min(start + chunk_bs, k)
                cur = stop - start
                batch = base_paired.expand(cur, -1).clone()
                for b_i, f_idx in enumerate(range(start, stop)):
                    pos_f = fold_positions[f_idx]
                    batch[b_i, [half + 1 + p for p in pos_f]] = AA_MASK_ID
                net_out = self.impl._forward_paired(batch, single_modality=single_mod)
                aa_logits = self.impl._extract_aa_logits(net_out, half)
                lp20 = F.log_softmax(
                    aa_logits.index_select(-1, self.impl.aa20_ids_t).to(dtype=torch.float32), dim=-1
                ).cpu().numpy()
                for b_i, f_idx in enumerate(range(start, stop)):
                    for p in fold_positions[f_idx]:
                        nll_total += max(0.0, float(-lp20[b_i, 1 + p, AA20_INDEX[sequence[p]]]))
        if not np.isfinite(nll_total):
            raise FloatingPointError(f"{self.arm_id}: non-finite pll_8fold_nll_sum")
        return float(nll_total)

    def native_site_conditional(
        self,
        sequence: str,
        pos_0based: int,
        *,
        bb: dict[str, Any] | None = None,
    ) -> np.ndarray:
        if self.is_mpnn:
            if bb is None:
                raise ValueError(f"{self.arm_id}: requires coords for native_site_conditional")
            return np.asarray(
                self.impl.native_site_conditional(sequence, pos_0based, coords=bb["coords"]),
                dtype=np.float64,
            )
        if self.is_dima:
            return np.asarray(
                self.impl.site_conditional_logprobs(sequence, positions=[pos_0based], batch_size=1)[0],
                dtype=np.float64,
            )
        return np.asarray(
            self.impl.native_site_conditional(sequence, pos_0based, **self._dplm2_struct_kwargs(bb)),
            dtype=np.float64,
        )

    @torch.no_grad()
    def score_r4_condition(
        self,
        unit_key: str,
        target_seq: str,
        context_items: Sequence[str],
        *,
        target_bb: dict[str, Any] | None = None,
        context_bbs: Sequence[dict[str, Any] | None] | None = None,
        budget_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Score one R4/Result-5 Main Table condition (8-fold PLL + full-mask, or MPNN 2-chain)."""
        if self.is_mpnn:
            if target_bb is None:
                raise ValueError(f"{self.arm_id}: requires target_bb for R4")
            ctx_seqs_v: list[str] = []
            ctx_coords_v: list[np.ndarray] = []
            if context_items and context_bbs:
                for c_seq, c_bb in zip(context_items, context_bbs):
                    if c_bb is not None and len(c_seq) == int(c_bb["coords"].shape[0]):
                        ctx_seqs_v.append(c_seq)
                        ctx_coords_v.append(c_bb["coords"])
                        break  # 1-chain canonical 2-chain spatial-offset pairing
            res = self.impl.score_context(
                target_seq,
                target_bb["coords"],
                context_seqs=ctx_seqs_v,
                context_coords=ctx_coords_v,
                n_permutations=4,
            )
            nll_res = float(res["nll_per_residue"])
            nll_sum = float(-res["ll_sum"])
            return {
                "k_items": int(res["n_context_chains"]),
                "n_cross_chain_edges": int(res["n_cross_chain_edges"]),
                "nll_sum": nll_sum,
                "nll_per_residue": nll_res,
                "pll_8fold": {"nll_sum": nll_sum, "nll_per_residue": nll_res},
                "full_target_masked": {"nll_sum": nll_sum, "nll_per_residue": nll_res},
            }

        if self.is_dima:
            eff_b = min(int(budget_tokens), int(self.impl.max_residues))
            res8 = self.impl.score_r4_context(
                target_seq, list(context_items), budget_tokens=eff_b, n_folds=8
            )
            res1 = self.impl.score_r4_context(
                target_seq, list(context_items), budget_tokens=eff_b, n_folds=1
            )
            nll_8 = float(res8["nll_sum"])
            nll_res8 = float(res8["nll_per_residue"])
            return {
                "k_items": int(res8["n_context_items_used"]),
                "n_context_residues": int(res8["n_context_residues"]),
                "budget_tokens": eff_b,
                "nll_sum": nll_8,
                "nll_per_residue": nll_res8,
                "pll_8fold": {"nll_sum": nll_8, "nll_per_residue": nll_res8},
                "full_target_masked": {
                    "nll_sum": float(res1["nll_sum"]),
                    "nll_per_residue": float(res1["nll_per_residue"]),
                },
            }

        import torch.nn.functional as F
        from src.capability.unified.dplm2_adapter import (
            AA_MASK_ID,
            STRUCT_MASK_ID,
            _encode_aa_to_ids,
        )

        single_mod = not self.is_dplm2_struct
        max_ctx = max(0, int(budget_tokens) - len(target_seq) - 4)

        ctx_seqs_used: list[str] = []
        ctx_st_used: list[list[int]] = []
        n_ctx_with_struct = 0
        used_len = 0
        for idx, c_seq in enumerate(context_items):
            rem = max_ctx - used_len
            if rem <= 0:
                break
            take = c_seq[:rem] if len(c_seq) > rem else c_seq
            ctx_seqs_used.append(take)
            c_bb = context_bbs[idx] if (context_bbs is not None and idx < len(context_bbs)) else None
            if (
                not single_mod
                and c_bb is not None
                and int(c_bb["coords"].shape[0]) >= len(take)
            ):
                if "dplm2_struct_tokens" in c_bb and len(c_bb["dplm2_struct_tokens"]) >= len(take):
                    st_ids = [int(x) for x in c_bb["dplm2_struct_tokens"][: len(take)]]
                else:
                    st_ids = [int(x) for x in self.impl.encode_backbone(c_bb["coords"][: len(take)])]
                ctx_st_used.append(st_ids)
                n_ctx_with_struct += 1
            else:
                ctx_st_used.append([STRUCT_MASK_ID] * len(take))
            used_len += len(take)
            if len(take) < len(c_seq):
                break

        full_seq = "".join(ctx_seqs_used) + target_seq
        tgt_start = len(full_seq) - len(target_seq)
        aa_ids = _encode_aa_to_ids(full_seq)
        if single_mod:
            s_ids = [STRUCT_MASK_ID] * len(full_seq)
        else:
            if target_bb is None or int(target_bb["coords"].shape[0]) != len(target_seq):
                raise ValueError(
                    f"{self.arm_id}/{unit_key}: target_bb length mismatch "
                    f"(seq={len(target_seq)}, bb={None if target_bb is None else int(target_bb['coords'].shape[0])})"
                )
            if "dplm2_struct_tokens" in target_bb and len(target_bb["dplm2_struct_tokens"]) == len(target_seq):
                tgt_st = [int(x) for x in target_bb["dplm2_struct_tokens"]]
            else:
                tgt_st = [int(x) for x in self.impl.encode_backbone(target_bb["coords"])]
            s_ids = [tid for part in ctx_st_used for tid in part] + tgt_st

        if len(aa_ids) != len(s_ids):
            raise AssertionError(
                f"{self.arm_id}/{unit_key}: AA length {len(aa_ids)} != struct length {len(s_ids)}"
            )

        base_paired, half = self.impl._build_paired_ids([aa_ids], [s_ids])
        L_tgt = len(target_seq)
        k = min(8, L_tgt)
        fold_pos = [list(range(f, L_tgt, k)) for f in range(k)]
        chunk_bs = self._safe_bs(len(full_seq), k)

        nll_8fold = 0.0
        with self.impl._precision_ctx():
            for start in range(0, k, chunk_bs):
                stop = min(start + chunk_bs, k)
                cur = stop - start
                batch = base_paired.expand(cur, -1).clone()
                for b_i, f_idx in enumerate(range(start, stop)):
                    batch[b_i, [half + 1 + tgt_start + p for p in fold_pos[f_idx]]] = AA_MASK_ID
                net_out = self.impl._forward_paired(batch, single_modality=single_mod)
                aa_logits = self.impl._extract_aa_logits(net_out, half)
                lp20 = F.log_softmax(
                    aa_logits.index_select(-1, self.impl.aa20_ids_t).to(dtype=torch.float32), dim=-1
                ).cpu().numpy()
                for b_i, f_idx in enumerate(range(start, stop)):
                    for p in fold_pos[f_idx]:
                        nll_8fold += max(0.0, float(-lp20[b_i, 1 + tgt_start + p, AA20_INDEX[target_seq[p]]]))

            # Companion full-target-masked 1-pass CE reusing exact same base_paired tensor
            batch_fm = base_paired.clone()
            batch_fm[0, half + 1 + tgt_start : half + 1 + tgt_start + L_tgt] = AA_MASK_ID
            out_fm = self.impl._forward_paired(batch_fm, single_modality=single_mod)
            lp20_fm = F.log_softmax(
                self.impl._extract_aa_logits(out_fm, half)[0, 1 + tgt_start : 1 + tgt_start + L_tgt]
                .index_select(-1, self.impl.aa20_ids_t)
                .to(dtype=torch.float32),
                dim=-1,
            ).cpu().numpy()
            nll_fm = float(
                sum(max(0.0, float(-lp20_fm[p, AA20_INDEX[target_seq[p]]])) for p in range(L_tgt))
            )

        return {
            "k_items": len(ctx_seqs_used),
            "n_context_items_with_struct": int(n_ctx_with_struct),
            "n_context_residues": int(tgt_start),
            "nll_sum": float(nll_8fold),
            "nll_per_residue": float(nll_8fold / max(L_tgt, 1)),
            "pll_8fold": {
                "nll_sum": float(nll_8fold),
                "nll_per_residue": float(nll_8fold / max(L_tgt, 1)),
            },
            "full_target_masked": {
                "nll_sum": float(nll_fm),
                "nll_per_residue": float(nll_fm / max(L_tgt, 1)),
            },
        }

    @torch.no_grad()
    def score_dms_with_context(
        self,
        wt: str,
        mutations: Sequence[str],
        context_items: Sequence[str],
        *,
        wt_bb: dict[str, Any] | None = None,
        context_bbs: Sequence[dict[str, Any] | None] | None = None,
        budget_tokens: int = 1024,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Score DMS variants conditioned on unaligned homolog/unrelated context (or 2-chain MPNN)."""
        if not context_items:
            return self.score_variants(wt, mutations, bb=wt_bb, batch_size=batch_size, multi_exact_pll=False)

        from src.capability.general_plm.adapters import parse_mutation

        parsed = [parse_mutation(m, wt) for m in mutations]
        scores = np.empty(len(mutations), dtype=np.float64)

        if self.is_mpnn:
            import torch.nn.functional as F
            from src.capability.unified.proteinmpnn_adapter import (
                MPNN_AA_TO_IDX,
                _canonical_permutations,
                _encode_mpnn_seq,
                _prepare_coords_array,
            )

            if wt_bb is None:
                raise ValueError(f"{self.arm_id}: requires wt_bb for DMS context")
            ctx_seq = None
            ctx_coords = None
            if context_bbs:
                for cs, cb in zip(context_items, context_bbs):
                    if cb is not None and len(cs) == int(cb["coords"].shape[0]):
                        ctx_seq = cs
                        ctx_coords = cb["coords"]
                        break
            if ctx_seq is None or ctx_coords is None:
                return self.score_variants(wt, mutations, bb=wt_bb, batch_size=batch_size, multi_exact_pll=False)

            # Build and encode the 2-chain spatial-offset complex ONCE per (assay, condition)!
            ca_only = self.impl.spec.ca_only
            tgt_clean, tgt_valid = _prepare_coords_array(wt_bb["coords"], ca_only=ca_only)
            c_clean, c_valid = _prepare_coords_array(ctx_coords, ca_only=ca_only)
            tgt_ca = tgt_clean if ca_only else tgt_clean[:, 1, :]
            tgt_vca = tgt_ca[tgt_valid > 0.5]
            c_ca = c_clean if ca_only else c_clean[:, 1, :]
            c_vca = c_ca[c_valid > 0.5]
            if len(tgt_vca) and len(c_vca):
                shift = np.zeros(3, dtype=np.float32)
                shift[0] = (float(tgt_vca[:, 0].max()) + 5.5) - float(c_vca[:, 0].min())
                shift[1:] = tgt_vca[:, 1:].mean(axis=0) - c_vca[:, 1:].mean(axis=0)
                c_clean = c_clean + shift

            L_c, L_t = len(ctx_seq), len(wt)
            all_coords = np.concatenate([c_clean, tgt_clean], axis=0)
            all_valid = np.concatenate([c_valid, tgt_valid], axis=0)
            all_res = np.concatenate(
                [np.arange(L_c, dtype=np.int64), L_c + 100 + np.arange(L_t, dtype=np.int64)]
            )
            all_chain = np.concatenate([np.ones(L_c, dtype=np.int64), np.full(L_t, 2, dtype=np.int64)])
            X_t = torch.from_numpy(all_coords).unsqueeze(0).to(device=self.impl.device, dtype=self.impl.torch_dtype)
            mask_t = torch.from_numpy(all_valid).unsqueeze(0).to(device=self.impl.device, dtype=self.impl.torch_dtype)
            res_t = torch.from_numpy(all_res).unsqueeze(0).to(device=self.impl.device, dtype=torch.long)
            chain_t = torch.from_numpy(all_chain).unsqueeze(0).to(device=self.impl.device, dtype=torch.long)
            ctx_S = _encode_mpnn_seq(ctx_seq)
            wt_S = _encode_mpnn_seq(wt)
            S_wt = torch.from_numpy(np.concatenate([ctx_S, wt_S])).unsqueeze(0).to(device=self.impl.device, dtype=torch.long)

            with self.impl._precision_ctx():
                h_V_enc, h_E, E_idx, h_EXV = self.impl.model.encode_structure(X_t, mask_t, res_t, chain_t)
                single_idx = [i for i, t in enumerate(parsed) if len(t) == 1]
                multi_idx = [i for i, t in enumerate(parsed) if len(t) > 1]

                if single_idx:
                    uniq_pos = sorted({parsed[i][0][0] for i in single_idx})
                    pos_map = {p: r for r, p in enumerate(uniq_pos)}
                    lp_mat = np.empty((len(uniq_pos), 20), dtype=np.float32)
                    base_all = np.arange(L_c + L_t, dtype=np.int64)
                    bs = max(1, int(batch_size))
                    for start in range(0, len(uniq_pos), bs):
                        chunk_p = uniq_pos[start : start + bs]
                        cur = len(chunk_p)
                        orders = np.empty((cur, L_c + L_t), dtype=np.int64)
                        for r_i, p in enumerate(chunk_p):
                            gp = L_c + p
                            if gp == L_c + L_t - 1:
                                orders[r_i] = base_all
                            else:
                                orders[r_i, :gp] = base_all[:gp]
                                orders[r_i, gp : L_c + L_t - 1] = base_all[gp + 1 :]
                                orders[r_i, L_c + L_t - 1] = gp
                        ord_t = torch.from_numpy(orders).to(device=self.impl.device, dtype=torch.long)
                        logits, _ = self.impl.model.decode_from_encoded(
                            h_V_enc.expand(cur, -1, -1),
                            h_E.expand(cur, -1, -1, -1),
                            E_idx.expand(cur, -1, -1),
                            h_EXV.expand(cur, -1, -1, -1),
                            S_wt.expand(cur, -1),
                            mask_t.expand(cur, -1),
                            ord_t,
                        )
                        row_ar = torch.arange(cur, device=self.impl.device, dtype=torch.long)
                        pos_t = torch.as_tensor([L_c + p for p in chunk_p], device=self.impl.device, dtype=torch.long)
                        lp_mat[start : start + cur] = F.log_softmax(
                            logits[row_ar, pos_t, :20].to(dtype=torch.float32), dim=-1
                        ).cpu().numpy()
                    for idx in single_idx:
                        pos, wt_aa, mut_aa = parsed[idx][0]
                        row = pos_map[pos]
                        scores[idx] = float(lp_mat[row, MPNN_AA_TO_IDX[mut_aa]] - lp_mat[row, MPNN_AA_TO_IDX[wt_aa]])

                if multi_idx:
                    perms = _canonical_permutations(L_t, n_permutations=4, seed=20261125)
                    orders = np.empty((4, L_c + L_t), dtype=np.int64)
                    orders[:, :L_c] = np.arange(L_c, dtype=np.int64)
                    orders[:, L_c:] = L_c + perms
                    ord_t = torch.from_numpy(orders).to(device=self.impl.device, dtype=torch.long)
                    var_seqs = [wt, *[apply_mutation(wt, mutations[i]) for i in multi_idx]]
                    S_mat = torch.as_tensor(
                        np.stack([np.concatenate([ctx_S, _encode_mpnn_seq(s)]) for s in var_seqs], axis=0),
                        device=self.impl.device,
                        dtype=torch.long,
                    )
                    tgt_mask = mask_t[:, L_c:].to(dtype=torch.float32)
                    ll_acc = np.zeros(len(var_seqs), dtype=np.float64)
                    bs = max(1, int(batch_size))
                    for p_i in range(4):
                        o1 = ord_t[p_i : p_i + 1]
                        for start in range(0, len(var_seqs), bs):
                            stop = min(start + bs, len(var_seqs))
                            cur = stop - start
                            Sc = S_mat[start:stop]
                            logits, _ = self.impl.model.decode_from_encoded(
                                h_V_enc.expand(cur, -1, -1),
                                h_E.expand(cur, -1, -1, -1),
                                E_idx.expand(cur, -1, -1),
                                h_EXV.expand(cur, -1, -1, -1),
                                Sc,
                                mask_t.expand(cur, -1),
                                o1.expand(cur, -1),
                            )
                            lp20 = F.log_softmax(logits[:, L_c:, :20].to(dtype=torch.float32), dim=-1)
                            tS = Sc[:, L_c:]
                            gath = torch.gather(lp20, 2, tS.clamp(max=19).unsqueeze(-1)).squeeze(-1)
                            ll_acc[start:stop] += (gath * (tS < 20).to(dtype=torch.float32) * tgt_mask).sum(dim=1).cpu().numpy()
                    ll_mean = ll_acc / 4.0
                    for offset, idx in enumerate(multi_idx):
                        scores[idx] = float(ll_mean[offset + 1] - ll_mean[0])
            return scores

        if self.is_dima:
            eff_budget = min(int(budget_tokens), int(self.impl.max_residues))
            if len(wt) > eff_budget:
                return self.score_variants(wt, mutations, batch_size=batch_size, multi_exact_pll=False)
            max_ctx = max(0, eff_budget - len(wt))
            ctx_parts: list[str] = []
            used_c = 0
            for c_seq in context_items:
                if used_c >= max_ctx:
                    break
                take = c_seq[: max_ctx - used_c]
                if take:
                    ctx_parts.append(take)
                    used_c += len(take)
            full_wt = "".join(ctx_parts) + wt
            tgt_start = len(full_wt) - len(wt)
            single_idx = [i for i, t in enumerate(parsed) if len(t) == 1]
            multi_idx = [i for i, t in enumerate(parsed) if len(t) > 1]
            eff_bs = self._safe_bs(len(full_wt), batch_size)
            if single_idx:
                uniq_pos = sorted({parsed[i][0][0] for i in single_idx})
                pos_map = {p: r for r, p in enumerate(uniq_pos)}
                lp_all = self.impl._masked_denoising_logprobs_window(
                    full_wt, [{tgt_start + p} for p in uniq_pos], batch_size=eff_bs
                )
                for idx in single_idx:
                    pos, wt_aa, mut_aa = parsed[idx][0]
                    row = pos_map[pos]
                    scores[idx] = float(
                        lp_all[row, tgt_start + pos, AA20_INDEX[mut_aa]]
                        - lp_all[row, tgt_start + pos, AA20_INDEX[wt_aa]]
                    )
            if multi_idx:
                mask_sets = [{tgt_start + pos for pos, _, _ in parsed[i]} for i in multi_idx]
                lp_m = self.impl._masked_denoising_logprobs_window(full_wt, mask_sets, batch_size=eff_bs)
                for row_i, idx in enumerate(multi_idx):
                    tot = 0.0
                    for pos, wt_aa, mut_aa in parsed[idx]:
                        tot += float(
                            lp_m[row_i, tgt_start + pos, AA20_INDEX[mut_aa]]
                            - lp_m[row_i, tgt_start + pos, AA20_INDEX[wt_aa]]
                        )
                    scores[idx] = tot
            return scores

        # DPLM-2 (seq_only or struct_conditioned) masked-marginal with concatenated context prefix
        import torch.nn.functional as F
        from src.capability.unified.dplm2_adapter import AA_MASK_ID, STRUCT_MASK_ID, _encode_aa_to_ids

        single_mod = not self.is_dplm2_struct
        max_ctx = max(0, int(budget_tokens) - len(wt) - 4)
        ctx_parts: list[str] = []
        ctx_st_parts: list[list[int]] = []
        used = 0
        for idx_c, c_seq in enumerate(context_items):
            if used >= max_ctx:
                break
            take = c_seq[: max_ctx - used]
            ctx_parts.append(take)
            if not single_mod and context_bbs and idx_c < len(context_bbs) and context_bbs[idx_c] is not None:
                c_bb = context_bbs[idx_c]
                if "dplm2_struct_tokens" in c_bb and len(c_bb["dplm2_struct_tokens"]) >= len(take):
                    ctx_st_parts.append([int(x) for x in c_bb["dplm2_struct_tokens"][: len(take)]])
                else:
                    ctx_st_parts.append(list(self.impl.encode_backbone(c_bb["coords"][: len(take)])))
            else:
                ctx_st_parts.append([STRUCT_MASK_ID] * len(take))
            used += len(take)

        full_wt = "".join(ctx_parts) + wt
        tgt_start = len(full_wt) - len(wt)
        aa_ids = _encode_aa_to_ids(full_wt)
        if single_mod:
            s_ids = [STRUCT_MASK_ID] * len(full_wt)
        else:
            if wt_bb is not None and "dplm2_struct_tokens" in wt_bb:
                tgt_st = [int(x) for x in wt_bb["dplm2_struct_tokens"]]
            elif wt_bb is not None:
                tgt_st = list(self.impl.encode_backbone(wt_bb["coords"]))
            else:
                tgt_st = [STRUCT_MASK_ID] * len(wt)
            s_ids = [tid for part in ctx_st_parts for tid in part] + tgt_st

        base_paired, half = self.impl._build_paired_ids([aa_ids], [s_ids])
        single_idx = [i for i, t in enumerate(parsed) if len(t) == 1]
        multi_idx = [i for i, t in enumerate(parsed) if len(t) > 1]
        bs = self._safe_bs(len(full_wt), batch_size)

        with self.impl._precision_ctx():
            if single_idx:
                uniq_pos = sorted({parsed[i][0][0] for i in single_idx})
                pos_map = {p: r for r, p in enumerate(uniq_pos)}
                lp_mat = np.empty((len(uniq_pos), 20), dtype=np.float32)
                for start in range(0, len(uniq_pos), bs):
                    chunk_p = uniq_pos[start : start + bs]
                    cur = len(chunk_p)
                    batch = base_paired.expand(cur, -1).clone()
                    row_ar = torch.arange(cur, device=self.impl.device, dtype=torch.long)
                    tok_pos = torch.as_tensor(
                        [half + 1 + tgt_start + p for p in chunk_p], device=self.impl.device, dtype=torch.long
                    )
                    batch[row_ar, tok_pos] = AA_MASK_ID
                    net_out = self.impl._forward_paired(batch, single_modality=single_mod)
                    aa_logits = self.impl._extract_aa_logits(net_out, half)
                    pos_logits = aa_logits[row_ar, tok_pos - half].index_select(-1, self.impl.aa20_ids_t).to(dtype=torch.float32)
                    lp_mat[start : start + cur] = F.log_softmax(pos_logits, dim=-1).cpu().numpy()
                for idx in single_idx:
                    pos, wt_aa, mut_aa = parsed[idx][0]
                    row = pos_map[pos]
                    scores[idx] = float(lp_mat[row, AA20_INDEX[mut_aa]] - lp_mat[row, AA20_INDEX[wt_aa]])

            if multi_idx:
                for start in range(0, len(multi_idx), bs):
                    chunk_i = multi_idx[start : start + bs]
                    cur = len(chunk_i)
                    batch = base_paired.expand(cur, -1).clone()
                    for b_i, m_idx in enumerate(chunk_i):
                        for pos, _, _ in parsed[m_idx]:
                            batch[b_i, half + 1 + tgt_start + pos] = AA_MASK_ID
                    net_out = self.impl._forward_paired(batch, single_modality=single_mod)
                    aa_logits = self.impl._extract_aa_logits(net_out, half)
                    lp20 = F.log_softmax(
                        aa_logits.index_select(-1, self.impl.aa20_ids_t).to(dtype=torch.float32), dim=-1
                    ).cpu().numpy()
                    for b_i, m_idx in enumerate(chunk_i):
                        tot = 0.0
                        for pos, wt_aa, mut_aa in parsed[m_idx]:
                            tot += float(
                                lp20[b_i, 1 + tgt_start + pos, AA20_INDEX[mut_aa]]
                                - lp20[b_i, 1 + tgt_start + pos, AA20_INDEX[wt_aa]]
                            )
                        scores[m_idx] = tot
        return scores


# ============================================================================
# Stage Runners (R1, R1-FC, R1-FD, R2, R3, E01, R4 Main, R5 DMS)
# ============================================================================


def run_stage_r1(
    bridge: ScoringAdapterBridge,
    bb_store: BackboneStore,
    root: Path,
    out_root: Path,
    *,
    batch_size: int,
    readout_batch_size: int,
    max_assays: int,
    skip_existing: bool,
) -> dict[str, Any]:
    r1_dir = out_root / "r1"
    r1_dir.mkdir(parents=True, exist_ok=True)
    slug = arm_slug(bridge.arm_id)
    scores_path = r1_dir / f"r1_ladder_scores_{slug}.npz"
    readout_path = r1_dir / f"r1_readout_blocks_{slug}.npz"
    summary_path = r1_dir / f"r1_summary_{slug}.json"
    if skip_existing and scores_path.is_file() and readout_path.is_file() and summary_path.is_file():
        return {"stage": "r1", "arm_id": bridge.arm_id, "status": "skipped_existing"}

    vt = json.loads((root / "results/extensions/statistical_ladder_20261106/inputs/variant_table.json").read_text())
    cohort = json.loads((root / "results/R5/context_mutation_rescue_20260923/cohort.json").read_text())
    cohort_by_assay = {r["assay"]: r for r in cohort["assays"]}

    assays: dict[str, dict[str, Any]] = {}
    for sample in vt["samples"]:
        aname = sample["assay"]
        entry = assays.get(aname)
        if entry is None:
            crow = cohort_by_assay[aname]
            entry = {
                "assay": aname,
                "cluster": int(sample["cluster"]),
                "wildtype": crow["wildtype"],
                "mut_to_measured": {m: float(y) for m, y in zip(crow["mutants"], crow["measured"])},
                "samples": [],
            }
            assays[aname] = entry
        entry["samples"].append(sample)

    assay_names = sorted(assays.keys())
    if max_assays > 0:
        assay_names = assay_names[:max_assays]

    t0 = time.monotonic()
    sample_scores: dict[str, float] = {}
    r_chunks: list[np.ndarray] = []
    r_sample_ids: list[str] = []
    rhos: list[float] = []
    clusters: list[int] = []
    per_assay: list[dict[str, Any]] = []

    for idx, aname in enumerate(assay_names, 1):
        entry = assays[aname]
        wt = entry["wildtype"]
        bb = bb_store.get(wt) if bridge.requires_backbone else None
        samples = entry["samples"]
        muts = [s["mutation"] for s in samples]
        measured = np.asarray([entry["mut_to_measured"][m] for m in muts], dtype=np.float64)

        scores = bridge.score_variants(wt, muts, bb=bb, batch_size=batch_size, multi_exact_pll=False)
        for s, sc in zip(samples, scores):
            sample_scores[str(s["sample_id"])] = float(sc)

        mut_seqs = [apply_mutation(wt, m) for m in muts]
        wt_r = bridge.extract_hidden_states_projected([wt], bb=bb, batch_size=1)
        mut_r = bridge.extract_hidden_states_projected(mut_seqs, bb=bb, batch_size=readout_batch_size)
        r_chunks.append((mut_r - wt_r).astype(np.float32, copy=False))
        r_sample_ids.extend(str(s["sample_id"]) for s in samples)

        rho = safe_spearman(scores, measured)
        if rho is not None:
            rhos.append(rho)
            clusters.append(entry["cluster"])
        per_assay.append(
            {
                "assay": aname,
                "cluster": entry["cluster"],
                "n_variants": len(samples),
                "model_spearman": rho,
            }
        )
        if idx % 25 == 0 or idx == len(assay_names):
            print(f"[R1:{bridge.arm_id}] {idx}/{len(assay_names)} assays ({time.monotonic() - t0:.1f}s)", flush=True)

    np.savez_compressed(
        scores_path,
        sample_ids=np.asarray(list(sample_scores.keys()), dtype="U"),
        scores=np.asarray(list(sample_scores.values()), dtype=np.float64),
    )
    np.savez_compressed(
        readout_path,
        sample_ids=np.asarray(r_sample_ids, dtype="U"),
        R=np.concatenate(r_chunks, axis=0),
    )
    summary = {
        "schema": "dplm2_mpnn_r1_summary_v1",
        "arm_id": bridge.arm_id,
        "provenance": bridge.provenance(),
        "n_assays": len(per_assay),
        "n_variants": len(sample_scores),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "zero_shot_spearman": cluster_bootstrap_mean(rhos, clusters, seed=20261120),
        "assays": per_assay,
        "generated_utc": _now_utc(),
    }
    assert_no_forbidden_infra_strings(summary, label=summary_path.name)
    atomic_write_json(summary_path, summary)
    return {"stage": "r1", "arm_id": bridge.arm_id, "status": "complete", "n_assays": len(per_assay)}


def run_stage_r1_fc(
    bridge: ScoringAdapterBridge,
    bb_store: BackboneStore,
    root: Path,
    out_root: Path,
    *,
    max_units: int,
    skip_existing: bool,
) -> dict[str, Any]:
    r1_dir = out_root / "r1"
    r1_dir.mkdir(parents=True, exist_ok=True)
    slug = arm_slug(bridge.arm_id)
    out_path = r1_dir / f"r1_family_conditionals_{slug}.json"
    if skip_existing and out_path.is_file():
        return {"stage": "r1_fc", "arm_id": bridge.arm_id, "status": "skipped_existing"}

    if bridge.requires_backbone:
        # Honest restraint: unaligned homolog contexts in family_conditionals do not carry per-homolog PDBs.
        return {
            "stage": "r1_fc",
            "arm_id": bridge.arm_id,
            "status": "not_applicable_unaligned_homolog_backbones_absent",
        }

    doc = json.loads((root / "results/extensions/family_conditionals_20261107/conditional-inputs.json").read_text())
    units = doc["units"][:max_units] if max_units > 0 else doc["units"]
    t0 = time.monotonic()
    unit_records: list[dict[str, Any]] = []
    for u_idx, unit in enumerate(units, 1):
        fc_mod._attach_baselines(unit)
        site_obs, site_prof, site_mkv = [], [], []
        n_ctx = 0
        for site in unit["sites"]:
            prof_lp = site["baseline"]["profile"]
            mkv_lps = site["baseline"]["markov"]
            c_obs, c_prof, c_mkv = [], [], []
            for ctx, mlp in zip(site["contexts"], mkv_lps):
                native = ctx["native"]
                nat_idx = int(ctx["native_index"])
                obs_idx = AA20_INDEX[ctx["observed"]]
                log_q = bridge.native_site_conditional(native, nat_idx)
                c_obs.append(float(log_q[obs_idx]))
                c_prof.append(float(prof_lp[obs_idx]))
                c_mkv.append(float(mlp[obs_idx]))
                n_ctx += 1
            if c_obs:
                site_obs.append(float(np.mean(c_obs)))
                site_prof.append(float(np.mean(c_prof)))
                site_mkv.append(float(np.mean(c_mkv)))
        u_obs = float(np.mean(site_obs))
        u_prof = float(np.mean(site_prof))
        u_mkv = float(np.mean(site_mkv))
        unit_records.append(
            {
                "unit_id": unit["unit_id"],
                "partition": unit["partition"],
                "n_sites": len(site_obs),
                "n_contexts": n_ctx,
                "model_logp": u_obs,
                "profile_logp": u_prof,
                "markov_logp": u_mkv,
                "minus_profile": u_obs - u_prof,
                "minus_markov": u_obs - u_mkv,
            }
        )
        if u_idx % 16 == 0 or u_idx == len(units):
            print(f"[R1-FC:{bridge.arm_id}] {u_idx}/{len(units)} units ({time.monotonic() - t0:.1f}s)", flush=True)

    payload = {
        "schema": "dplm2_mpnn_family_conditionals_v1",
        "arm_id": bridge.arm_id,
        "conditional_semantics": "masked_1pass_dplm2_seq_only",
        "provenance": bridge.provenance(),
        "n_units": len(unit_records),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "units": unit_records,
        "generated_utc": _now_utc(),
    }
    assert_no_forbidden_infra_strings(payload, label=out_path.name)
    atomic_write_json(out_path, payload)
    return {"stage": "r1_fc", "arm_id": bridge.arm_id, "status": "complete", "n_units": len(unit_records)}


def run_stage_r1_fd(
    bridge: ScoringAdapterBridge,
    root: Path,
    out_root: Path,
    *,
    batch_size: int,
    max_units: int,
    skip_existing: bool,
) -> dict[str, Any]:
    fd_dir = out_root / "r1_family_distribution"
    fd_dir.mkdir(parents=True, exist_ok=True)
    slug = arm_slug(bridge.arm_id)
    out_path = fd_dir / f"r1_family_distribution_{slug}.json"
    if skip_existing and out_path.is_file():
        return {"stage": "r1_fd", "arm_id": bridge.arm_id, "status": "skipped_existing"}

    if bridge.requires_backbone:
        return {
            "stage": "r1_fd",
            "arm_id": bridge.arm_id,
            "status": "not_applicable_unaligned_homolog_backbones_absent",
        }

    q_path = root / "results/extensions/family_distribution_20261106/queue.json"
    queue = json.loads(q_path.read_text(encoding="utf-8"))
    col_seeds = [int(s) for s in queue["protocol"]["null"]["column_shuffle_seeds"]]
    queue_seed = int(queue["protocol"]["queue"]["seed"])
    units = sorted(queue["units"], key=lambda u: u["unit_id"])
    if max_units > 0:
        units = units[:max_units]

    t0 = time.monotonic()
    unit_results: list[dict[str, Any]] = []
    diagnostic_exact_vs_8fold: dict[str, Any] | None = None
    for u_idx, unit in enumerate(units, 1):
        wt_entry = unit["wildtypes"][0]
        rows = fd_mod.evaluation_rows(wt_entry)
        natives = [r.native for r in rows]
        lengths = np.asarray([len(s) for s in natives], dtype=np.float64)
        col_null_sets = []
        for s_seed in col_seeds:
            shuf = fd_mod.column_null(rows, s_seed)
            fd_mod.verify_column_null(rows, shuf)
            col_null_sets.append(shuf)
        comp_nulls = [fd_mod.composition_null(row, queue_seed + i) for i, row in enumerate(rows)]

        native_nll = np.asarray([bridge.pll_8fold_nll_sum(s, n_folds=8) for s in natives], dtype=np.float64)
        col_nlls = np.stack(
            [
                np.asarray([bridge.pll_8fold_nll_sum(s, n_folds=8) for s in sset], dtype=np.float64)
                for sset in col_null_sets
            ],
            axis=0,
        )
        comp_nll = np.asarray([bridge.pll_8fold_nll_sum(s, n_folds=8) for s in comp_nulls], dtype=np.float64)

        if u_idx == 1:
            # 16-sequence exact O(L) PLL vs 8-fold stride PLL parity diagnostic on units[0]
            k_diag = min(8, len(natives), len(comp_nulls))
            diag_seqs = list(natives[:k_diag]) + list(comp_nulls[:k_diag])
            diag_lens = np.asarray([len(s) for s in diag_seqs], dtype=np.float64)
            diag_8f = np.concatenate([native_nll[:k_diag], comp_nll[:k_diag]]) / diag_lens
            diag_ex = np.asarray(
                [bridge.exact_pll_terms(s, batch_size=batch_size)[1] for s in diag_seqs], dtype=np.float64
            ) / diag_lens
            pearson_r = (
                float(np.corrcoef(diag_ex, diag_8f)[0, 1])
                if len(diag_seqs) >= 3 and np.std(diag_ex) > 1e-12 and np.std(diag_8f) > 1e-12
                else None
            )
            diagnostic_exact_vs_8fold = {
                "unit_id": unit["unit_id"],
                "n_sequences": int(len(diag_seqs)),
                "n_native": int(k_diag),
                "n_comp_null": int(k_diag),
                "pearson_r": pearson_r,
                "spearman_rho": safe_spearman(diag_ex, diag_8f),
                "mean_abs_diff_nats_per_res": float(np.mean(np.abs(diag_ex - diag_8f))),
                "exact_comp_minus_native_nats_per_res": float(np.mean(diag_ex[k_diag:] - diag_ex[:k_diag])),
                "pll8fold_comp_minus_native_nats_per_res": float(np.mean(diag_8f[k_diag:] - diag_8f[:k_diag])),
            }

        unit_results.append(
            {
                "unit_id": unit["unit_id"],
                "wildtype_id": wt_entry["wildtype_id"],
                "partition": unit["partition"],
                "n_native": len(natives),
                "total_residues": int(lengths.sum()),
                "column_contrast": fd_mod.contrast(native_nll, col_nlls, lengths),
                "composition_contrast": fd_mod.contrast(native_nll, comp_nll[None, :], lengths),
            }
        )
        if u_idx % 8 == 0 or u_idx == len(units):
            print(f"[R1-FD:{bridge.arm_id}] {u_idx}/{len(units)} units ({time.monotonic() - t0:.1f}s)", flush=True)

    payload = {
        "schema": "general_plm_r1_family_distribution_pll_v1",
        "arm": bridge.arm_id,
        "score_semantics": "pll_8fold_stride_within_model_delta",
        "n_folds": 8,
        "exact_pll_vs_pll_8fold_diagnostic": diagnostic_exact_vs_8fold,
        "queue_sha256": sha256_file(q_path),
        "n_units": len(unit_results),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "provenance": bridge.provenance(),
        "units": unit_results,
        "generated_utc": _now_utc(),
    }
    assert_no_forbidden_infra_strings(payload, label=out_path.name)
    atomic_write_json(out_path, payload)
    return {"stage": "r1_fd", "arm_id": bridge.arm_id, "status": "complete", "n_units": len(unit_results)}


def run_stage_r2(
    bridge: ScoringAdapterBridge,
    bb_store: BackboneStore,
    root: Path,
    out_root: Path,
    *,
    batch_size: int,
    readout_batch_size: int,
    max_bgs: int,
    skip_existing: bool,
) -> dict[str, Any]:
    r2_dir = out_root / "r2"
    r2_dir.mkdir(parents=True, exist_ok=True)
    slug = arm_slug(bridge.arm_id)
    ms_path = r2_dir / f"r2_stability_arrays_{slug}.npz"
    dm_path = r2_dir / f"r2_abundance_arrays_{slug}.npz"
    sum_path = r2_dir / f"r2_summary_{slug}.json"
    if skip_existing and ms_path.is_file() and dm_path.is_file() and sum_path.is_file():
        return {"stage": "r2", "arm_id": bridge.arm_id, "status": "skipped_existing"}

    def _run_lane(lane_name: str, items: list[dict[str, Any]], dest: Path) -> dict[str, Any]:
        t0 = time.monotonic()
        all_scores, all_targets, all_groups, all_subs = [], [], [], []
        r_chunks: list[np.ndarray] = []
        rhos, grp_ids = [], []
        for idx, bg in enumerate(items, 1):
            wt = bg["wildtype"]
            bb = bb_store.get(wt) if bridge.requires_backbone else None
            muts = bg["mutations"]
            targets = np.asarray(bg["targets"], dtype=np.float64)
            scores = bridge.score_variants(wt, muts, bb=bb, batch_size=batch_size, multi_exact_pll=False)
            rho = safe_spearman(scores, targets)
            if rho is not None:
                rhos.append(rho)
                grp_ids.append(str(bg["group"]))
            mut_seqs = [apply_mutation(wt, m) for m in muts]
            wt_r = bridge.extract_hidden_states_projected([wt], bb=bb, batch_size=1)
            mut_r = bridge.extract_hidden_states_projected(mut_seqs, bb=bb, batch_size=readout_batch_size)
            r_chunks.append((mut_r - wt_r).astype(np.float32, copy=False))
            for m, sc, yv in zip(muts, scores, targets):
                pos0 = int(m[1:-1]) - 1
                all_scores.append(float(sc))
                all_targets.append(float(yv))
                all_groups.append(str(bg["group"]))
                all_subs.append(f"{bg['name']}:{pos0}")
            if idx % 50 == 0 or idx == len(items):
                print(f"[R2-{lane_name}:{bridge.arm_id}] {idx}/{len(items)} ({time.monotonic() - t0:.1f}s)", flush=True)

        np.savez_compressed(
            dest,
            score=np.asarray(all_scores, dtype=np.float64),
            target=np.asarray(all_targets, dtype=np.float64),
            group=np.asarray(all_groups, dtype="U"),
            subunit=np.asarray(all_subs, dtype="U"),
            R=np.concatenate(r_chunks, axis=0),
        )
        return {
            "lane": lane_name,
            "n_backgrounds": len(items),
            "n_variants": len(all_scores),
            "elapsed_seconds": round(time.monotonic() - t0, 2),
            "zero_shot_spearman": cluster_bootstrap_mean(rhos, grp_ids, seed=20261120),
        }

    # MegaScale
    ms_cohort = json.loads((root / "results/R4/gate_stability_20260924/cohort/cohort.json").read_text())
    ms_rows = ms_cohort["backgrounds"][:max_bgs] if max_bgs > 0 else ms_cohort["backgrounds"]
    ms_items = [
        {
            "name": bg["name"],
            "group": bg["group"],
            "wildtype": bg["wildtype"],
            "mutations": [f"{bg['wildtype'][v['position'] - 1]}{v['position']}{v['mutant']}" for v in bg["variants"]],
            "targets": [float(v["ddg"]) for v in bg["variants"]],
        }
        for bg in ms_rows
    ]
    # Domainome
    dm_cohort = json.loads((root / "results/R1/external_confirmation_20260924/cohort.json").read_text())
    dm_rows = dm_cohort["domains"][:max_bgs] if max_bgs > 0 else dm_cohort["domains"]
    dm_items = [
        {
            "name": dom["name"],
            "group": dom["group"],
            "wildtype": dom["wildtype"],
            "mutations": [f"{dom['wildtype'][v['position'] - 1]}{v['position']}{v['mutant']}" for v in dom["variants"]],
            "targets": [float(v["target"]) for v in dom["variants"]],
        }
        for dom in dm_rows
    ]

    ms_res = _run_lane("stability", ms_items, ms_path)
    dm_res = _run_lane("abundance", dm_items, dm_path)
    summary = {
        "schema": "dplm2_mpnn_r2_summary_v1",
        "arm_id": bridge.arm_id,
        "provenance": bridge.provenance(),
        "stability": ms_res,
        "abundance": dm_res,
        "generated_utc": _now_utc(),
    }
    assert_no_forbidden_infra_strings(summary, label=sum_path.name)
    atomic_write_json(sum_path, summary)
    return {"stage": "r2", "arm_id": bridge.arm_id, "status": "complete"}


def run_stage_r3(
    bridge: ScoringAdapterBridge,
    bb_store: BackboneStore,
    root: Path,
    out_root: Path,
    *,
    batch_size: int,
    readout_batch_size: int,
    max_bgs: int,
    skip_existing: bool,
) -> dict[str, Any]:
    r3_dir = out_root / "r3"
    r3_dir.mkdir(parents=True, exist_ok=True)
    slug = arm_slug(bridge.arm_id)
    npz_path = r3_dir / f"r3_pairwise_cycles_{slug}.npz"
    sum_path = r3_dir / f"r3_summary_{slug}.json"
    if skip_existing and npz_path.is_file() and sum_path.is_file():
        return {"stage": "r3", "arm_id": bridge.arm_id, "status": "skipped_existing"}

    plan_path = root / "archive/logs/R3/pairwise_epistasis_20260924/extraction_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    bgs = plan["backgrounds"][:max_bgs] if max_bgs > 0 else plan["backgrounds"]

    t0 = time.monotonic()
    eps_list: list[float] = []
    grp_list: list[str] = []
    sep_list: list[str] = []
    r_cycle_chunks: list[np.ndarray] = []
    state_scores_by_bg: dict[str, np.ndarray] = {}
    total_states = 0

    for b_idx, bg in enumerate(bgs, 1):
        seqs: list[str] = bg["sequences"]
        wt = seqs[0]  # State 0 is always the WT background in extraction_plan.json
        bb = bb_store.get(wt) if bridge.requires_backbone else None
        total_states += len(seqs)
        plls = bridge.score_r3_states_batch(seqs, bb=bb, batch_size=batch_size)
        state_scores_by_bg[bg["name"]] = plls

        proj_states = bridge.extract_hidden_states_projected(seqs, bb=bb, batch_size=readout_batch_size)
        bg_r = []
        for cyc in bg["cycles"]:
            i_wt, i_a, i_b, i_ab = [int(x) for x in cyc["states"]]
            eps_list.append(float(plls[i_ab] - plls[i_a] - plls[i_b] + plls[i_wt]))
            grp_list.append(str(bg["group"]))
            sep_list.append(str(cyc["separation"]))
            bg_r.append(proj_states[i_ab] - proj_states[i_a] - proj_states[i_b] + proj_states[i_wt])
        r_cycle_chunks.append(np.stack(bg_r, axis=0))
        if b_idx % 8 == 0 or b_idx == len(bgs):
            print(f"[R3:{bridge.arm_id}] {b_idx}/{len(bgs)} bgs ({total_states} states, {time.monotonic() - t0:.1f}s)", flush=True)

    eps_arr = np.asarray(eps_list, dtype=np.float64)
    np.savez_compressed(
        npz_path,
        cycle_eps_pll=eps_arr,
        cycle_group=np.asarray(grp_list, dtype="U"),
        cycle_separation=np.asarray(sep_list, dtype="U"),
        R_cycle=np.concatenate(r_cycle_chunks, axis=0),
        **{f"state_pll__{k}": v for k, v in state_scores_by_bg.items()},
    )
    summary = {
        "schema": "dplm2_mpnn_r3_summary_v1",
        "arm_id": bridge.arm_id,
        "provenance": bridge.provenance(),
        "n_backgrounds": len(bgs),
        "n_states": total_states,
        "n_cycles": int(eps_arr.size),
        "nonzero_cycle_fraction": float(np.mean(np.abs(eps_arr) > 1e-8)) if eps_arr.size else 0.0,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "generated_utc": _now_utc(),
    }
    assert_no_forbidden_infra_strings(summary, label=sum_path.name)
    atomic_write_json(sum_path, summary)
    return {"stage": "r3", "arm_id": bridge.arm_id, "status": "complete", "n_cycles": int(eps_arr.size)}


def run_stage_e01(
    bridge: ScoringAdapterBridge,
    bb_store: BackboneStore,
    root: Path,
    out_root: Path,
    *,
    batch_size: int,
    max_assays: int,
    skip_existing: bool,
) -> dict[str, Any]:
    """Emit canonical ``position_likelihood_v1`` E01 contact archives for all 30 structural assays."""
    slug = arm_slug(bridge.arm_id)
    arm_dir = out_root / "e01_contacts" / f"posresp_extract_{slug}"
    completion_path = arm_dir / "position_likelihood.json"
    if skip_existing and completion_path.is_file():
        return {"stage": "e01", "arm_id": bridge.arm_id, "status": "skipped_existing"}

    arm_dir.mkdir(parents=True, exist_ok=True)
    archives_dir = arm_dir / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)

    cohort_path = root / "results/R5/context_mutation_rescue_20260923/cohort.json"
    sites_path = root / "results/extensions/mutation_structure_20261006/structure/annotation/sites.json.gz"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    with gzip.open(sites_path, "rt", encoding="utf-8") as fh:
        sites_doc = json.load(fh)
    site_rows = sites_doc["sites"] if isinstance(sites_doc, dict) else sites_doc
    admitted_site_assays = {r["assay_id"] for r in site_rows if r["status"] == "admitted"}
    rows_by_assay = {r["assay"]: r for r in cohort["assays"]}

    selected = [
        rows_by_assay[name]
        for name in sorted(admitted_site_assays & set(rows_by_assay))
        if 20 <= len(rows_by_assay[name]["wildtype"]) <= 1000
        and not (set(rows_by_assay[name]["wildtype"]) - set(AA20))
    ]
    selected.sort(key=lambda r: -len(r["wildtype"]) * (1 + len(r["mutants"])))
    if max_assays > 0:
        selected = list(reversed(selected))[:max_assays]

    identity = {
        "arm": bridge.arm_id,
        "paradigm": MASKED,
        "dtype": bridge.dtype,
        "conditioning_regime": bridge.meta.conditioning_regime,
        "term_semantics": TERM_SEMANTICS[MASKED],
        "alignment_rule": ALIGNMENT,
        "provenance": bridge.provenance(),
    }

    t0 = time.monotonic()
    receipts: list[dict[str, Any]] = []
    for idx_a, row in enumerate(selected, 1):
        assay = row["assay"]
        wt = row["wildtype"]
        bb = bb_store.get(wt) if bridge.requires_backbone else None
        sequences = [wt, *row["sequences"]]
        fname = "position_" + hashlib.sha256(assay.encode("utf-8")).hexdigest()[:20] + ".npz"
        arc_path = archives_dir / fname

        packed_states: list[PackedState] = []
        pos_scores: list[PositionScore] = []
        for s_idx, seq in enumerate(sequences):
            terms, nll_sum, gathered = bridge.exact_pll_terms(
                seq, bb=bb, batch_size=batch_size, return_gathered=(s_idx == 0)
            )
            terms_f32 = np.clip(terms.astype(np.float32), 0.0, None)
            nll_f32 = float(torch.as_tensor(terms_f32, dtype=torch.float32, device="cpu").sum())
            res = float(retention_residual(terms_f32, nll_f32, "cpu"))
            if res != 0.0:
                raise FloatingPointError(f"{assay}: non-zero retention residual {res}")
            ids = (0, *(AA20_INDEX[ch] + 1 for ch in seq))
            packed_states.append(
                PackedState(
                    sequence=seq,
                    ids=ids,
                    span=(1, 1 + len(seq)),
                    counts=np.ones(len(seq), dtype=np.int64),
                    offset=0,
                )
            )
            pos_scores.append(
                PositionScore(
                    terms=terms_f32,
                    nll_sum=nll_f32,
                    retention_residual_nats=res,
                    gathered=gathered if s_idx == 0 else None,
                )
            )

        # Repeat check on WT
        rep_terms, _, _ = bridge.exact_pll_terms(wt, bb=bb, batch_size=batch_size)
        rep_f32 = np.clip(rep_terms.astype(np.float32), 0.0, None)
        rep_score = PositionScore(
            terms=rep_f32,
            nll_sum=float(np.sum(rep_f32, dtype=np.float64)),
            retention_residual_nats=0.0,
            gathered=None,
        )
        rep_val = float(repeat_residual(pos_scores[0], rep_score))

        # Alignment check using canonical PackedState.sidecar()
        wild_sidecar = packed_states[0].sidecar()
        aligned = 0
        misaligned: list[dict[str, Any]] = []
        for mut_str, st in zip(row["mutants"], packed_states[1:]):
            if ":" in mut_str:
                continue
            pos0 = int(mut_str[1:-1]) - 1
            verdict = alignment(wild_sidecar, st.sidecar(), pos0)
            if verdict["aligned"]:
                aligned += 1
            else:
                misaligned.append({"mutation": mut_str, "reason": verdict.get("reason", "misaligned")})

        archive = PositionArchive(assay=assay, paradigm=MASKED)
        archive.add_wildtype(packed_states[0], pos_scores[0])
        for mut_str, st, sc in zip(row["mutants"], packed_states[1:], pos_scores[1:]):
            archive.add_mutant(mut_str, st, sc)

        extras: dict[str, Any] = {}
        cond_meta: dict[str, Any] = {"available": False, "reason": "not computed"}
        if pos_scores[0].gathered is not None:
            extras = {
                "wt_conditional_logprobs": np.asarray(pos_scores[0].gathered, dtype=np.float32),
                "wt_conditional_positions": np.arange(
                    packed_states[0].offset,
                    packed_states[0].offset + packed_states[0].scored_tokens,
                    dtype=np.int64,
                ),
                "wt_conditional_residues": np.asarray(list(AA20), dtype=object).astype("U"),
            }
            cond_meta = {"available": True, "residues": list(AA20), "resolved_residues": len(AA20)}

        metadata = {
            "schema": POS_SCHEMA,
            "identity": identity,
            "assay": assay,
            "cluster": row["cluster"],
            "mutant_digest": row.get("mutant_digest", ""),
            "states": len(sequences),
            "blas": blas_pinning(),
            "repeat_max_abs_nats": rep_val,
        }
        archive.write(arc_path, metadata=metadata, extras=extras)
        # Validate canonical RetainedResponses roundtrip immediately
        from src.capability.extensions.responses import RetainedResponses

        with np.load(arc_path, allow_pickle=False) as arc_data:
            arc_payload = {k: arc_data[k] for k in arc_data.files if k != "metadata"}
            states_rt = rebuild_states(arc_payload, sequences)
            retained = RetainedResponses(
                arc_data,
                states_rt,
                {"wildtype": wt, "mutants": row["mutants"], "sequences": row["sequences"]},
            )
            if not retained.selected_indices:
                raise RuntimeError(f"{assay}: RetainedResponses selected zero variants")

        receipts.append(
            {
                "assay": assay,
                "file": fname,
                "sha256": sha256_file(arc_path),
                "resumed": False,
                "cluster": row["cluster"],
                "residues": len(wt),
                "states": len(sequences),
                "packed_tokens": len(wt) + 1,
                "scored_tokens": len(wt),
                "retention_max_abs_nats": float(archive.retention_max_abs_nats),
                "upstream_max_abs_nats": 0.0,
                "repeat_max_abs_nats": rep_val,
                "aligned_single_substitutions": aligned,
                "misaligned_single_substitutions": misaligned,
                "conditional": cond_meta,
            }
        )
        if idx_a % 5 == 0 or idx_a == len(selected):
            print(f"[E01:{bridge.arm_id}] {idx_a}/{len(selected)} assays ({time.monotonic() - t0:.1f}s)", flush=True)

    repeat_max = max((float(r["repeat_max_abs_nats"]) for r in receipts), default=0.0)
    by_assay = {r["assay"]: r for r in selected}
    write_json(
        arm_dir / "archive-input.json",
        {
            "schema": "position_response_archive_input_v1",
            "arm": bridge.arm_id,
            "assays": [
                {
                    "assay": item["assay"],
                    "cluster": by_assay[item["assay"]]["cluster"],
                    "protein": item["assay"],
                    "wildtype": by_assay[item["assay"]]["wildtype"],
                    "mutants": by_assay[item["assay"]]["mutants"],
                    "sequences": by_assay[item["assay"]]["sequences"],
                    "archive": f"archives/{item['file']}",
                }
                for item in receipts
            ],
        },
    )
    completion_doc = {
        "status": "complete",
        "identity": identity,
        "prefix_invariant": {
            "rule": PREFIX_INVARIANT_RULE,
            "tier2_nats": TIER2_NATS,
            "tier2_repeat_multiple": TIER2_REPEAT_MULTIPLE,
            "repeat_max_abs_nats": repeat_max,
            "admitted_tolerance_nats": 0.0,
            "upstream_max_abs_nats": 0.0,
            "tier": residual_tier(0.0, repeat_max),
            "exactly_zero": True,
        },
        "created_utc": _now_utc(),
        "assays": receipts,
        "totals": {
            "assays_present": len(receipts),
            "states": sum(int(r["states"]) for r in receipts),
            "repeat_max_abs_nats": repeat_max,
        },
    }
    assert_no_forbidden_infra_strings(completion_doc, label=completion_path.name)
    write_json(completion_path, completion_doc)
    return {"stage": "e01", "arm_id": bridge.arm_id, "status": "complete", "n_assays": len(receipts)}


def run_stage_r4_main(
    bridge: ScoringAdapterBridge,
    bb_store: BackboneStore,
    root: Path,
    out_root: Path,
    *,
    max_units: int,
    skip_existing: bool,
) -> dict[str, Any]:
    r4_dir = out_root / "r4_main"
    r4_dir.mkdir(parents=True, exist_ok=True)
    slug = arm_slug(bridge.arm_id)
    out_path = r4_dir / f"r4_unaligned_{slug}.json"
    if skip_existing and out_path.is_file():
        return {"stage": "r4_main", "arm_id": bridge.arm_id, "status": "skipped_existing"}

    cohort = json.loads((root / r4_mod.DEFAULT_COHORT_REL).read_text(encoding="utf-8"))
    prot = cohort["protein"]
    records: list[str] = prot["records"]
    related_map = prot.get("related", {})
    filler_raw = prot["filler"]
    filler_idx = int(filler_raw) if isinstance(filler_raw, int) else int(filler_raw.get("index", 0))
    filler_seq = (
        records[filler_idx]
        if isinstance(filler_raw, int)
        else str(filler_raw.get("record") or filler_raw.get("sequence") or records[filler_idx])
    )
    retained = [u for u in prot["units"] if u.get("stratum") == "retained"]
    if max_units > 0:
        retained = retained[:max_units]

    t0 = time.monotonic()
    unit_rows: list[dict[str, Any]] = []
    for idx, unit in enumerate(retained, 1):
        target_seq, cond_inputs = r4_mod.build_unit_condition_inputs(
            unit, records, related_map, filler_idx=filler_idx, filler_seq=filler_seq
        )
        target_bb = bb_store.get(target_seq) if bridge.requires_backbone else None
        hom_bbs = (
            [bb_store.get_optional(s) for s in cond_inputs["homologue"]]
            if bridge.requires_backbone
            else None
        )
        unr_bbs = (
            [bb_store.get_optional(s) for s in cond_inputs["unrelated"]]
            if bridge.requires_backbone
            else None
        )
        filler_bb = bb_store.get_optional(filler_seq) if bridge.requires_backbone else None

        cond_out: dict[str, Any] = {}
        for cname in r4_mod.CONDITIONS:
            ctx_items = cond_inputs[cname]
            if not bridge.requires_backbone:
                ctx_bbs = None
            elif cname == "homologue":
                ctx_bbs = hom_bbs
            elif cname == "unrelated":
                ctx_bbs = unr_bbs
            elif cname == "mono_shuffled":
                # Same backbone geometry as homolog, perturbed monoresidue sequence
                ctx_bbs = hom_bbs
            elif cname == "position_only":
                # Same-length filler truncated/tiled to homolog backbone geometry
                ctx_bbs = hom_bbs if hom_bbs else ([filler_bb] * len(ctx_items) if filler_bb else None)
            else:
                ctx_bbs = []

            r4_budget = 510 if bridge.is_dima else 1024
            cond_out[cname] = bridge.score_r4_condition(
                str(unit["key"]),
                target_seq,
                ctx_items,
                target_bb=target_bb,
                context_bbs=ctx_bbs,
                budget_tokens=r4_budget,
            )

        def _contrast(key: str) -> dict[str, Any]:
            n_hom = float(cond_out["homologue"][key]["nll_per_residue"])
            n_unr = float(cond_out["unrelated"][key]["nll_per_residue"])
            n_shf = float(cond_out["mono_shuffled"][key]["nll_per_residue"])
            return {
                "hom_beats_unrelated": 1.0 if n_hom < n_unr else (0.5 if n_hom == n_unr else 0.0),
                "hom_beats_mono_shuffled": 1.0 if n_hom < n_shf else (0.5 if n_hom == n_shf else 0.0),
                "fractional_reduction_vs_unrelated": (n_unr - n_hom) / n_unr if n_unr > 0 else None,
                "fractional_reduction_vs_shuffled": (n_shf - n_hom) / n_shf if n_shf > 0 else None,
            }

        unit_rows.append(
            {
                "key": str(unit["key"]),
                "band": str(unit["band"]),
                "group": int(unit["group"]),
                "conditions": cond_out,
                "pll_8fold": _contrast("pll_8fold"),
                "full_target_masked": _contrast("full_target_masked"),
            }
        )
        if idx % 100 == 0 or idx == len(retained):
            print(f"[R4-main:{bridge.arm_id}] {idx}/{len(retained)} units ({time.monotonic() - t0:.1f}s)", flush=True)

    payload = {
        "schema": r4_mod.SCHEMA_ARM_RUNNER,
        "arm": bridge.arm_id,
        "budget_tokens": 510 if bridge.is_dima else 1024,
        "primary_estimand": (
            "inverse_folding_2chain_nll" if bridge.is_mpnn else "pll_8fold"
        ),
        "provenance": bridge.provenance(),
        "n_units": len(unit_rows),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "units": unit_rows,
        "generated_utc": _now_utc(),
    }
    assert_no_forbidden_infra_strings(payload, label=out_path.name)
    atomic_write_json(out_path, payload)
    return {"stage": "r4_main", "arm_id": bridge.arm_id, "status": "complete", "n_units": len(unit_rows)}


def run_stage_r5_dms(
    bridge: ScoringAdapterBridge,
    bb_store: BackboneStore,
    root: Path,
    out_root: Path,
    *,
    batch_size: int,
    max_assays: int,
    skip_existing: bool,
) -> dict[str, Any]:
    r5_dir = out_root / "r5_dms"
    (r5_dir / "rescue_30_80").mkdir(parents=True, exist_ok=True)
    (r5_dir / "e09_five_band").mkdir(parents=True, exist_ok=True)
    slug = arm_slug(bridge.arm_id)
    p1a = r5_dir / "rescue_30_80" / f"{slug}.json"
    p1b = r5_dir / "e09_five_band" / f"{slug}.json"

    token_len_fn = lambda s: len(s) + 1  # noqa: E731
    dms_budget = 510 if bridge.is_dima else 1024
    results = {}
    if not (skip_existing and p1a.is_file()):
        t0 = time.monotonic()
        cohort_1a = json.loads((root / r5_dms_mod.RESCUE_COHORT_REL).read_text(encoding="utf-8"))
        selected_1a: list[tuple[dict[str, Any], dict[str, Any]]] = []
        skipped_1a: list[dict[str, Any]] = []
        for row in cohort_1a["assays"]:
            plan, reason = r5_dms_mod.plan_rescue_30_80_assay(
                row,
                token_length_fn=token_len_fn,
                overhead_tokens=4,
                budget_tokens=dms_budget,
                multi_item_icl=False,
                msa_mode=False,
            )
            if plan is None:
                skipped_1a.append({"assay": row["assay"], "reason": reason})
            else:
                selected_1a.append((row, plan))
                if max_assays > 0 and len(selected_1a) >= max_assays:
                    break

        scored_1a: list[dict[str, Any]] = []
        for idx, (row, plan) in enumerate(selected_1a, 1):
            wt = str(row["wildtype"])
            muts = list(row["mutants"])
            wt_bb = bb_store.get(wt) if bridge.requires_backbone else None
            cond_items = {
                "no_context": [],
                "unrelated": list(plan["unrelated_items"]),
                "homolog": list(plan["homolog_items"]),
            }
            scores_map: dict[str, list[float]] = {}
            for cond in r5_dms_mod.RESCUE_CONDITIONS:
                c_items = cond_items[cond]
                c_bbs = [bb_store.get_optional(s) for s in c_items] if bridge.requires_backbone else None
                scores_map[cond] = bridge.score_dms_with_context(
                    wt,
                    muts,
                    c_items,
                    wt_bb=wt_bb,
                    context_bbs=c_bbs,
                    budget_tokens=dms_budget,
                    batch_size=batch_size,
                ).tolist()
            scored_1a.append(
                {
                    "assay": row["assay"],
                    "wildtype_id": row["wildtype_id"],
                    "cluster": int(row["cluster"]),
                    "mutants": muts,
                    "measured": row["measured"],
                    "profile_scores": row["profile_scores"],
                    "mutant_digest": row["mutant_digest"],
                    "original_mutant_digest": row["original_mutant_digest"],
                    "scores": scores_map,
                    "context_tokens": int(plan["context_tokens"]),
                    "k_items": int(plan["k_items"]),
                    "homolog_identity": float(plan["homolog_identity"]),
                    "max_lcs": int(plan["max_lcs"]),
                    "spearman": {
                        c: safe_spearman(scores_map[c], row["measured"]) for c in r5_dms_mod.RESCUE_CONDITIONS
                    },
                }
            )
            if idx % 40 == 0 or idx == len(selected_1a):
                print(f"[R5-1A:{bridge.arm_id}] {idx}/{len(selected_1a)} ({time.monotonic() - t0:.1f}s)", flush=True)
        doc_1a = {
            "schema": r5_dms_mod.SCHEMA_DMS_ARM_SCORE,
            "subtest": "rescue_30_80",
            "status": "complete",
            "entry_key": bridge.arm_id,
            "arm": bridge.base_arm_id,
            "input_alignment_format": (
                "two_chain_spatial_offset_3d" if bridge.is_mpnn else "unaligned_concat"
            ),
            "conditioning_regime": bridge.meta.conditioning_regime,
            "budget": dms_budget,
            "provenance": bridge.provenance(),
            "assays_scored": len(scored_1a),
            "assays_skipped": len(skipped_1a),
            "skipped": skipped_1a,
            "elapsed_seconds": round(time.monotonic() - t0, 2),
            "assays": scored_1a,
            "generated_utc": _now_utc(),
        }
        assert_no_forbidden_infra_strings(doc_1a, label=p1a.name)
        atomic_write_json(p1a, doc_1a)
        results["rescue_30_80"] = len(scored_1a)
    else:
        results["rescue_30_80"] = "skipped_existing"

    if not (skip_existing and p1b.is_file()):
        t0 = time.monotonic()
        cohort_1b = json.loads((root / r5_dms_mod.E09_COHORT_REL).read_text(encoding="utf-8"))
        targets_by_id = {t["target_id"]: t for t in cohort_1b["targets"]}
        assays_by_target: dict[str, list[dict[str, Any]]] = {}
        for a in cohort_1b["assays"]:
            assays_by_target.setdefault(a["target_id"], []).append(a)
        all_conds = ("no_context", "unrelated", *r5_dms_mod.E09_BINS_ORDERED)
        assays_list = list(cohort_1b["assays"])
        if max_assays > 0:
            assays_list = assays_list[:max_assays]

        plans_by_target: dict[str, dict[str, Any]] = {}
        for tid in sorted({a["target_id"] for a in assays_list}):
            plans_by_target[tid] = r5_dms_mod.plan_e09_target(
                targets_by_id[tid],
                assays_by_target.get(tid, ()),
                token_length_fn=token_len_fn,
                overhead_tokens=4,
                budget_tokens=dms_budget,
                msa_mode=False,
                long_icl_mode=False,
            )

        scored_1b: list[dict[str, Any]] = []
        skipped_1b: list[dict[str, Any]] = []
        for idx, arow in enumerate(assays_list, 1):
            tid = arow["target_id"]
            trow = targets_by_id[tid]
            tplan = plans_by_target[tid]
            present_conds = [
                c for c in all_conds if (tplan["conditions"].get(c) or {}).get("status") == "present"
            ]
            if not present_conds:
                skipped_1b.append({"assay": arow["assay"], "reason": "no_present_conditions"})
                continue
            wt = str(trow["wildtype"])
            muts = list(arow["mutants"])
            wt_bb = bb_store.get(wt) if bridge.requires_backbone else None
            cond_scores: dict[str, list[float]] = {}
            for cond in present_conds:
                items = tplan["items"].get(cond) or []
                c_items = [str(it["sequence"]) for it in items]
                c_bbs = [bb_store.get_optional(s) for s in c_items] if bridge.requires_backbone else None
                cond_scores[cond] = bridge.score_dms_with_context(
                    wt,
                    muts,
                    c_items,
                    wt_bb=wt_bb,
                    context_bbs=c_bbs,
                    budget_tokens=dms_budget,
                    batch_size=batch_size,
                ).tolist()
            scored_1b.append(
                {
                    "assay": arow["assay"],
                    "target_id": tid,
                    "cluster": int(arow["cluster"]),
                    "mutants": muts,
                    "measured": arow["measured"],
                    "scores": cond_scores,
                    "spearman": {
                        c: safe_spearman(cond_scores[c], arow["measured"]) for c in present_conds
                    },
                }
            )
            if idx % 40 == 0 or idx == len(assays_list):
                print(f"[R5-1B:{bridge.arm_id}] {idx}/{len(assays_list)} ({time.monotonic() - t0:.1f}s)", flush=True)

        doc_1b = {
            "schema": r5_dms_mod.SCHEMA_DMS_ARM_SCORE,
            "subtest": "e09_five_band",
            "status": "complete",
            "entry_key": bridge.arm_id,
            "arm": bridge.base_arm_id,
            "input_alignment_format": (
                "two_chain_spatial_offset_3d" if bridge.is_mpnn else "unaligned_concat"
            ),
            "conditioning_regime": bridge.meta.conditioning_regime,
            "budget": dms_budget,
            "provenance": bridge.provenance(),
            "assays_scored": len(scored_1b),
            "assays_skipped": len(skipped_1b),
            "skipped": skipped_1b,
            "elapsed_seconds": round(time.monotonic() - t0, 2),
            "assays": scored_1b,
            "generated_utc": _now_utc(),
        }
        assert_no_forbidden_infra_strings(doc_1b, label=p1b.name)
        atomic_write_json(p1b, doc_1b)
        results["e09_five_band"] = len(scored_1b)
    else:
        results["e09_five_band"] = "skipped_existing"

    return {"stage": "r5_dms", "arm_id": bridge.arm_id, "status": "complete", "subtests": results}


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", default="all", help=f"Comma-separated model IDs or groups {list(MODEL_GROUPS)}")
    parser.add_argument("--stages", default="all", help=f"Comma-separated stages from {list(ALL_STAGES)} or 'all'")
    parser.add_argument("--dtype", default="auto", help="auto (default), float32, or bfloat16")
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--dima-root", type=Path, default=None)
    parser.add_argument(
        "--backbone-dir",
        type=Path,
        default=DATA_ROOT / bb_mod.DEFAULT_CACHE_DIR,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_ROOT / DEFAULT_NEW_SCORING_DIR,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--readout-batch-size", type=int, default=16)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-assays", type=int, default=0, help="Limit assays/bgs for smoke tests (0 = all)")
    parser.add_argument("--max-units", type=int, default=0, help="Limit units for smoke tests (0 = all)")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--list-models", action="store_true", help="Print all 23 scored entries & exit")
    args = parser.parse_args()

    if args.list_models:
        print(
            json.dumps(
                {
                    "n_entries": len(NEW_23_SCORED_ENTRIES),
                    "entries": [
                        {
                            "arm_id": a,
                            "slug": arm_slug(a),
                            "paradigm": EXTENDED_ARM_METADATA[a].paradigm,
                            "conditioning_regime": EXTENDED_ARM_METADATA[a].conditioning_regime,
                        }
                        for a in NEW_23_SCORED_ENTRIES
                    ],
                },
                indent=2,
            )
        )
        return

    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit(f"Invalid shard {args.shard_index}/{args.num_shards}")

    models = _resolve_model_list(args.models)
    stages = (
        list(ALL_STAGES)
        if args.stages.strip().lower() == "all"
        else [s.strip() for s in args.stages.split(",") if s.strip()]
    )
    unknown_stages = sorted(set(stages) - set(ALL_STAGES))
    if unknown_stages:
        raise SystemExit(f"Unknown stage(s): {unknown_stages}; expected subset of {list(ALL_STAGES)}")

    shard_models = [m for idx, m in enumerate(models) if idx % args.num_shards == args.shard_index]
    if not shard_models:
        print(f"[dplm2-mpnn] Shard {args.shard_index}/{args.num_shards} has 0 models; exiting cleanly.", flush=True)
        return

    bb_store = BackboneStore(args.backbone_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []

    for arm_id in shard_models:
        # Group consecutive stages that share the same resolved dtype so each checkpoint loads once
        dtype_by_stage = {st: _resolve_stage_dtype(arm_id, st, args.dtype) for st in stages}
        by_dtype: dict[str, list[str]] = {}
        for st in stages:
            by_dtype.setdefault(dtype_by_stage[st], []).append(st)

        for dt_name, dt_stages in by_dtype.items():
            print(
                f"[dplm2-mpnn] Loading arm={arm_id} dtype={dt_name} device={args.device} stages={dt_stages}",
                flush=True,
            )
            bridge = ScoringAdapterBridge(
                arm_id,
                device=args.device,
                dtype=dt_name,
                model_root=args.model_root,
                dima_root=args.dima_root,
            )
            for st in dt_stages:
                if st == "r1":
                    rec = run_stage_r1(
                        bridge, bb_store, DATA_ROOT, args.out_dir,
                        batch_size=args.batch_size, readout_batch_size=args.readout_batch_size,
                        max_assays=args.max_assays, skip_existing=args.skip_existing,
                    )
                elif st == "r1_fc":
                    rec = run_stage_r1_fc(
                        bridge, bb_store, DATA_ROOT, args.out_dir,
                        max_units=args.max_units, skip_existing=args.skip_existing,
                    )
                elif st == "r1_fd":
                    rec = run_stage_r1_fd(
                        bridge, DATA_ROOT, args.out_dir,
                        batch_size=args.batch_size, max_units=args.max_units,
                        skip_existing=args.skip_existing,
                    )
                elif st == "r2":
                    rec = run_stage_r2(
                        bridge, bb_store, DATA_ROOT, args.out_dir,
                        batch_size=args.batch_size, readout_batch_size=args.readout_batch_size,
                        max_bgs=args.max_assays, skip_existing=args.skip_existing,
                    )
                elif st == "r3":
                    rec = run_stage_r3(
                        bridge, bb_store, DATA_ROOT, args.out_dir,
                        batch_size=args.batch_size, readout_batch_size=args.readout_batch_size,
                        max_bgs=args.max_assays, skip_existing=args.skip_existing,
                    )
                elif st == "e01":
                    rec = run_stage_e01(
                        bridge, bb_store, DATA_ROOT, args.out_dir,
                        batch_size=args.batch_size, max_assays=args.max_assays,
                        skip_existing=args.skip_existing,
                    )
                elif st == "r4_main":
                    rec = run_stage_r4_main(
                        bridge, bb_store, DATA_ROOT, args.out_dir,
                        max_units=args.max_units, skip_existing=args.skip_existing,
                    )
                elif st == "r5_dms":
                    rec = run_stage_r5_dms(
                        bridge, bb_store, DATA_ROOT, args.out_dir,
                        batch_size=args.batch_size, max_assays=args.max_assays,
                        skip_existing=args.skip_existing,
                    )
                else:
                    raise AssertionError(f"Unhandled stage {st}")
                receipts.append(rec)
            del bridge
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(json.dumps({"shard": args.shard_index, "num_shards": args.num_shards, "receipts": receipts}, indent=2), flush=True)


if __name__ == "__main__":
    main()
