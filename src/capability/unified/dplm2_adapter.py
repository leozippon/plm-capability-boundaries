"""Self-contained PyTorch adapter for all 4 official DPLM-2 checkpoints + StructTokenizer.

Implements the Apache-2.0 ByteDance DPLM-2 architecture (``EsmForDPLM2`` for
``dplm2-{150m,650m,3b}`` and ``EsmForDPLM2Bit`` for ``dplm2-bit-650m``) and a
zero-dependency pure-PyTorch ``StructTokenizerEncoder`` (adapting the MIT-licensed
ESM-IF1 ``GVPTransformerEncoder`` + ByteDance ``pre_quant`` MLP + 13-bit ``LFQ``
without requiring ``byprot``, ``fair-esm``, ``torch_geometric``, ``torch_scatter``,
or ``openfold``).

Supports **both** ``mode="seq_only"`` (``<mask_struct>`` + official
``single_modality`` cross-modal attention masking) and
``mode="struct_conditioned"`` (real 3D backbone coordinates encoded via
``StructTokenizerEncoder`` or pre-cached ``dplm2_struct_tokens``) across:

1. Exact ``O(L)`` leave-one-out masked marginal AA20 log-probabilities
   (``site_conditional_logprobs``), ``score_variants``, ``exact_pll``,
   ``pll_8fold``, and ``score_r4_context``.
2. Unmasked (``p=0``) frozen hidden-state extraction over the AA-half residue
   span (``blocks[(L-1)//2]`` & ``blocks[L-1]``, ``[mean, cls]`` -> ``(N, 4, H)``
   or ``project_four_blocks`` -> ``(N, 1024)`` where ``0:512`` = ``R_mid`` and
   ``512:1024`` = ``R_last``).
3. E01 perturbation-propagation contact sensitivity matrix ``J_{ij}``.
4. Generation: ``infill_window`` (``R5a``), ``continue_suffix`` (``R4``
   Downstream 2), and ``sample_diffusion_full`` (``co_generation`` annealing,
   ``seq_only`` ``max_confidence`` / ``uniform_random``, ``inverse_folding``,
   and profile-guided ``Exp 4.2``).
"""

from __future__ import annotations

import hashlib
import math
import os
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.esm.modeling_esm import (
    EsmAttention,
    EsmContactPredictionHead,
    EsmEmbeddings,
    EsmEncoder,
    EsmForMaskedLM,
    EsmIntermediate,
    EsmLayer,
    EsmLMHead,
    EsmModel,
    EsmOutput,
    EsmPooler,
    EsmPreTrainedModel,
    EsmSelfAttention,
    EsmSelfOutput,
    RotaryEmbedding,
    apply_rotary_pos_emb,
)

from src.capability.core.amino_acids import AA20
from src.capability.core.precision_policy import fp32_matmul_context, requested_fp32_matmul_policy
from src.capability.general_plm.adapters import PositionVector, apply_mutation, parse_mutation
from src.capability.general_plm.eval_utils import project_four_blocks

# Exact DPLM-2 vocab constants (matches vocab.txt of all 4 official checkpoints).
AA_CLS_ID = 0
PAD_ID = 1
AA_EOS_ID = 2
AA_UNK_ID = 3
AA_MASK_ID = 32
STRUCT_CLS_ID = 33
STRUCT_EOS_ID = 34
STRUCT_UNK_ID = 35
STRUCT_VOCAB_OFFSET = 36
STRUCT_CODEBOOK_SIZE = 8192
STRUCT_MASK_ID = 8228  # line 8229 (0-based index 8228) in vocab.txt

AA_TYPE = 1
STRUCT_TYPE = 0
PAD_TYPE = 2

# Exact ESM2/DPLM-2 token IDs 4..23: L A G V S E R T I D P K Q N F Y M H W C
_DPLM2_AA_ORDER = "LAGVSERTIDPKQNFYMHWC"
DPLM2_AA_TO_ID: dict[str, int] = {aa: idx + 4 for idx, aa in enumerate(_DPLM2_AA_ORDER)}
DPLM2_ID_TO_AA: dict[int, str] = {idx + 4: aa for idx, aa in enumerate(_DPLM2_AA_ORDER)}
DPLM2_AA20_IDS: list[int] = [DPLM2_AA_TO_ID[aa] for aa in AA20]  # ordered by canonical AA20
AA20_INDEX: dict[str, int] = {aa: idx for idx, aa in enumerate(AA20)}

DEFAULT_DPLM2_MODEL_ROOT = Path("/home/lzp/models/general_plm_20261120")


@dataclass(frozen=True)
class DPLM2Spec:
    arm_id: str
    dir_name: str
    is_bit: bool
    num_layers: int
    hidden_size: int
    default_dtype: str = "float32"


DPLM2_SPECS: Mapping[str, DPLM2Spec] = {
    "dplm2-150m": DPLM2Spec("dplm2-150m", "dplm2-150m", False, 30, 640, "float32"),
    "dplm2-650m": DPLM2Spec("dplm2-650m", "dplm2-650m", False, 33, 1280, "float32"),
    "dplm2-bit-650m": DPLM2Spec("dplm2-bit-650m", "dplm2-bit-650m", True, 33, 1280, "float32"),
    "dplm2-3b": DPLM2Spec("dplm2-3b", "dplm2-3b", False, 36, 2560, "bfloat16"),
}


def resolve_dplm2_dir(name: str, model_root: Path | str | None = None) -> Path:
    """Resolve a DPLM-2 checkpoint or ``dplm2-struct-tokenizer`` directory."""
    candidates: list[Path] = []
    if model_root is not None:
        p = Path(model_root)
        candidates.extend([p / name, p])
    for env_key in ("DPLM2_MODEL_ROOT", "GENERAL_PLM_MODEL_ROOT", "MODEL_ROOT", "MODEL_BASE_DIR"):
        raw = os.environ.get(env_key, "").strip()
        if raw:
            ep = Path(raw)
            candidates.extend([ep / name, ep])
    candidates.append(DEFAULT_DPLM2_MODEL_ROOT / name)
    for cand in candidates:
        if cand.is_dir() and (
            (cand / "config.json").is_file() or (cand / "dplm2_struct_tokenizer.ckpt").is_file()
        ):
            return cand
    raise FileNotFoundError(
        f"Could not locate {name!r}; checked: " + ", ".join(str(c) for c in candidates)
    )


def code_indices_to_lfq_bits(code_indices: torch.Tensor, codebook_dim: int = 13) -> torch.Tensor:
    """Map integer code indices ``0..8191`` -> big-endian ``{-1.0, +1.0}`` (``LFQ.get_codebook_entry``)."""
    safe = code_indices.clamp(min=0, max=(1 << codebook_dim) - 1).to(dtype=torch.long)
    mask = (2 ** torch.arange(codebook_dim - 1, -1, -1, device=safe.device, dtype=torch.long))
    bits = ((safe.unsqueeze(-1) & mask) != 0).to(dtype=torch.float32)
    return bits * 2.0 - 1.0


def lfq_bits_to_code_indices(bits_01: torch.Tensor, codebook_dim: int = 13) -> torch.Tensor:
    """Map ``{0, 1}`` big-endian bits ``(..., 13)`` -> integer code indices ``0..8191``."""
    mask = (2 ** torch.arange(codebook_dim - 1, -1, -1, device=bits_01.device, dtype=torch.long))
    return (bits_01.to(dtype=torch.long) * mask).sum(dim=-1)


# --------------------------------------------------------------------------- #
# Pure-PyTorch StructTokenizerEncoder (ESM-IF1 GVPTransformerEncoder + LFQ)
# Zero fair-esm / torch_geometric / torch_scatter / openfold dependencies!
# --------------------------------------------------------------------------- #


def _nan_to_num(ts: torch.Tensor, val: float = 0.0) -> torch.Tensor:
    return torch.where(~torch.isfinite(ts), torch.tensor(val, dtype=ts.dtype, device=ts.device), ts)


def _norm(tensor: torch.Tensor, dim: int, eps: float = 1e-8, keepdim: bool = False) -> torch.Tensor:
    return torch.sqrt(torch.sum(torch.square(tensor), dim=dim, keepdim=keepdim) + eps)


def _normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return _nan_to_num(torch.div(tensor, _norm(tensor, dim=dim, keepdim=True)))


def _rbf(values: torch.Tensor, v_min: float, v_max: float, n_bins: int = 16) -> torch.Tensor:
    centers = torch.linspace(v_min, v_max, n_bins, device=values.device, dtype=values.dtype)
    centers = centers.view([1] * len(values.shape) + [-1])
    std = (v_max - v_min) / n_bins
    return torch.exp(-(((values.unsqueeze(-1) - centers) / std) ** 2))


def _get_rotation_frames(coords: torch.Tensor) -> torch.Tensor:
    v1 = coords[:, :, 2] - coords[:, :, 1]
    v2 = coords[:, :, 0] - coords[:, :, 1]
    e1 = _normalize(v1, dim=-1)
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = _normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-2)


def _rotate(v: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    return torch.sum(v.unsqueeze(-1) * R.unsqueeze(-3), dim=-2)


def _norm_no_nan(x: torch.Tensor, axis: int = -1, keepdims: bool = False, eps: float = 1e-8, sqrt: bool = True) -> torch.Tensor:
    out = torch.sum(torch.square(x), dim=axis, keepdim=keepdims) + eps
    return torch.sqrt(out) if sqrt else out


def _tuple_sum(tp1: tuple[torch.Tensor, torch.Tensor | None], tp2: tuple[torch.Tensor, torch.Tensor | None]):
    s1, v1 = tp1
    s2, v2 = tp2
    if v1 is None and v2 is None:
        return (s1 + s2, None)
    return (s1 + s2, v1 + v2)  # type: ignore[operator]


def _tuple_cat(*args: tuple[torch.Tensor, torch.Tensor], dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    d = dim % len(args[0][0].shape)
    s_args, v_args = zip(*args)
    return torch.cat(s_args, dim=d), torch.cat(v_args, dim=d)


class _GVP(nn.Module):
    def __init__(
        self,
        in_dims: tuple[int, int],
        out_dims: tuple[int, int],
        h_dim: int | None = None,
        vector_gate: bool = False,
        activations: tuple[Any, Any] = (F.relu, torch.sigmoid),
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if vector_gate:
                    self.wg = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)
        self.vector_gate = vector_gate
        self.scalar_act, self.vector_act = activations
        self.eps = eps

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.vi:
            s, v = x
            vh = self.wh(torch.transpose(v, -1, -2))
            vn = _norm_no_nan(vh, axis=-2, eps=self.eps)
            s = self.ws(torch.cat([s, vn], -1))
            if self.scalar_act is not None:
                s = self.scalar_act(s)
            if self.vo:
                v_out = torch.transpose(self.wv(vh), -1, -2)
                g = self.wg(s).unsqueeze(-1) if self.vector_gate else _norm_no_nan(v_out, axis=-1, keepdims=True, eps=self.eps)
                if self.vector_act is not None:
                    v_out = v_out * self.vector_act(g)
                return (s, v_out)
            return (s, None)
        s = self.ws(x[0])
        if self.scalar_act is not None:
            s = self.scalar_act(s)
        if self.vo:
            return (s, torch.zeros(list(s.shape)[:-1] + [self.vo, 3], device=s.device, dtype=s.dtype))
        return (s, None)


class _GVPLayerNorm(nn.Module):
    def __init__(self, dims: tuple[int, int], eps: float = 1e-8) -> None:
        super().__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
        self.eps = eps

    def forward(self, x: tuple[torch.Tensor, torch.Tensor | None]) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.v:
            return self.scalar_norm(x[0]), None
        s, v = x
        assert v is not None
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False, eps=self.eps)
        nonzero_mask = vn > 2.0 * self.eps
        vn = torch.sum(vn * nonzero_mask, dim=-2, keepdim=True) / (
            self.eps + torch.sum(nonzero_mask, dim=-2, keepdim=True)
        )
        v = nonzero_mask * (v / torch.sqrt(vn + self.eps))
        return self.scalar_norm(s), v


class _PureGVPConv(nn.Module):
    """Exact pure-PyTorch drop-in replacement for PyG ``GVPConv(aggr='mean')``."""

    def __init__(
        self,
        in_dims: tuple[int, int],
        out_dims: tuple[int, int],
        edge_dims: tuple[int, int],
        n_layers: int = 3,
        vector_gate: bool = True,
        eps: float = 1e-4,
        activations: tuple[Any, Any] = (F.relu, torch.sigmoid),
    ) -> None:
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        mods: list[nn.Module] = [
            _GVP(
                (2 * self.si + self.se, 2 * self.vi + self.ve),
                out_dims,
                vector_gate=vector_gate,
                activations=activations,
                eps=1e-8,
            )
        ]
        for _ in range(n_layers - 2):
            mods.append(_GVP(out_dims, out_dims, vector_gate=vector_gate, eps=1e-8))
        mods.append(_GVP(out_dims, out_dims, activations=(None, None), eps=1e-8))
        self.message_func = nn.Sequential(*mods)

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_s, x_v = x
        n_nodes = x_s.shape[0]
        src, dst = edge_index[0], edge_index[1]
        # PyG MessagePassing flow="source_to_target": j=src, i=dst
        msg_in = _tuple_cat((x_s[src], x_v[src]), edge_attr, (x_s[dst], x_v[dst]))
        msg_s, msg_v = self.message_func(msg_in)

        out_s = torch.zeros((n_nodes, self.so), device=x_s.device, dtype=msg_s.dtype)
        out_v = torch.zeros((n_nodes, self.vo, 3), device=x_v.device, dtype=msg_v.dtype)
        cnt = torch.zeros((n_nodes,), device=x_s.device, dtype=msg_s.dtype)
        out_s.index_add_(0, dst, msg_s)
        out_v.index_add_(0, dst, msg_v)
        cnt.index_add_(0, dst, torch.ones_like(dst, dtype=msg_s.dtype))
        denom = cnt.clamp(min=1.0)
        return out_s / denom.unsqueeze(-1), out_v / denom.unsqueeze(-1).unsqueeze(-1)


class _GVPConvLayer(nn.Module):
    def __init__(
        self,
        node_dims: tuple[int, int],
        edge_dims: tuple[int, int],
        vector_gate: bool = True,
        n_message: int = 3,
        n_feedforward: int = 2,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.conv = _PureGVPConv(
            node_dims,
            node_dims,
            edge_dims,
            n_layers=n_message,
            vector_gate=vector_gate,
            eps=eps,
            activations=(F.relu, torch.sigmoid),
        )
        self.norm = nn.ModuleList([_GVPLayerNorm(node_dims, eps=eps) for _ in range(2)])
        hid_dims = (4 * node_dims[0], 2 * node_dims[1])
        self.ff_func = nn.Sequential(
            _GVP(node_dims, hid_dims, vector_gate=vector_gate, eps=1e-8),
            _GVP(hid_dims, node_dims, activations=(None, None), eps=1e-8),
        )

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        dh = self.conv(x, edge_index, edge_attr)
        x = self.norm[0](_tuple_sum(x, dh))
        dh = self.ff_func(x)
        x = self.norm[1](_tuple_sum(x, dh))
        return x, edge_attr


class _Normalize(nn.Module):
    def __init__(self, features: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = torch.sqrt(x.var(-1, keepdim=True) + self.epsilon)
        return self.gain * (x - mu) / (sigma + self.epsilon) + self.bias


class _DihedralFeatures(nn.Module):
    def __init__(self, node_embed_dim: int) -> None:
        super().__init__()
        self.node_embedding = nn.Linear(6, node_embed_dim, bias=True)
        self.norm_nodes = _Normalize(node_embed_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        Xf = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)
        dX = Xf[:, 1:, :] - Xf[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2, u_1, u_0 = U[:, :-2, :], U[:, 1:-1, :], U[:, 2:, :]
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)
        cosD = torch.clamp((n_2 * n_1).sum(-1), -1.0 + eps, 1.0 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        D = F.pad(D, (1, 2), "constant", 0)
        D = D.view(D.size(0), D.size(1) // 3, 3)
        V = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return self.norm_nodes(self.node_embedding(V))


def _gvp_node_features(coords: torch.Tensor, coord_mask: torch.Tensor, with_coord_mask: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-7
    Xf = torch.flatten(coords[:, :, :3], 1, 2)
    bsz = Xf.shape[0]
    U = _normalize(Xf[:, 1:] - Xf[:, :-1], dim=-1)
    u_2, u_1, u_0 = U[:, :-2], U[:, 1:-1], U[:, 2:]
    n_2 = _normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)
    cosD = torch.clamp(torch.sum(n_2 * n_1, -1), -1.0 + eps, 1.0 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    D = F.pad(D, [1, 2]).reshape(bsz, -1, 3)
    node_s = torch.cat([torch.cos(D), torch.sin(D)], -1)
    if with_coord_mask:
        node_s = torch.cat([node_s, coord_mask.to(dtype=node_s.dtype).unsqueeze(-1)], dim=-1)

    X_ca = coords[:, :, 1]
    fwd = F.pad(_normalize(X_ca[:, 1:] - X_ca[:, :-1]), [0, 0, 0, 1])
    bwd = F.pad(_normalize(X_ca[:, :-1] - X_ca[:, 1:]), [0, 0, 1, 0])
    orientations = torch.cat([fwd.unsqueeze(-2), bwd.unsqueeze(-2)], -2)
    n, origin, c = coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]
    c_u, n_u = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c_u + n_u)
    perp = _normalize(torch.cross(c_u, n_u, dim=-1))
    sidechain = -bisector * math.sqrt(1.0 / 3.0) - perp * math.sqrt(2.0 / 3.0)
    node_v = torch.cat([orientations, sidechain.unsqueeze(-2)], dim=-2)
    return node_s, node_v


class _GVPGraphEmbedding(nn.Module):
    def __init__(self, top_k_neighbors: int = 30) -> None:
        super().__init__()
        self.top_k_neighbors = top_k_neighbors
        self.num_positional_embeddings = 16
        node_hidden_dim = (1024, 256)
        edge_hidden_dim = (32, 1)
        self.embed_node = nn.Sequential(
            _GVP((7, 3), node_hidden_dim, activations=(None, None)),
            _GVPLayerNorm(node_hidden_dim, eps=1e-4),
        )
        self.embed_edge = nn.Sequential(
            _GVP((34, 1), edge_hidden_dim, activations=(None, None)),
            _GVPLayerNorm(edge_hidden_dim, eps=1e-4),
        )
        self.embed_confidence = nn.Linear(16, 1024)

    def _get_edge_features(
        self,
        coords: torch.Tensor,
        coord_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        X_ca = coords[:, :, 1]
        bsz, maxlen = X_ca.size(0), X_ca.size(1)
        coord_mask_2D = coord_mask.unsqueeze(1) * coord_mask.unsqueeze(2)
        residue_mask = ~padding_mask
        residue_mask_2D = residue_mask.unsqueeze(1) * residue_mask.unsqueeze(2)
        D = coord_mask_2D * _norm(X_ca.unsqueeze(1) - X_ca.unsqueeze(2), dim=-1)
        seqpos = torch.arange(maxlen, device=X_ca.device)
        Dseq = torch.abs(seqpos.unsqueeze(1) - seqpos.unsqueeze(0)).repeat(bsz, 1, 1)
        D_adjust = _nan_to_num(D) + (~coord_mask_2D) * (1e8 + Dseq * 1e6) + (~residue_mask_2D) * 1e10
        k = min(self.top_k_neighbors, maxlen)
        E_dist, E_idx = torch.topk(D_adjust, k, dim=-1, largest=False)
        E_coord_mask = (E_dist < 5e7).flatten(1, 2).unsqueeze(-1)
        E_residue_mask = (E_dist < 5e9).flatten(1, 2)

        src = torch.arange(maxlen, device=E_idx.device).view(1, maxlen, 1).expand(bsz, maxlen, k)
        edge_index = torch.stack([src, E_idx], dim=0).flatten(2, 3)  # (2, B, E)
        E_dist_flat = E_dist.flatten(1, 2)

        d = edge_index[0] - edge_index[1]
        freq = torch.exp(
            torch.arange(0, self.num_positional_embeddings, 2, dtype=torch.float32, device=E_idx.device)
            * -(math.log(10000.0) / self.num_positional_embeddings)
        )
        angles = d.unsqueeze(-1) * freq
        pos_emb = torch.cat((torch.cos(angles), torch.sin(angles)), -1).to(dtype=coords.dtype)
        D_rbf = _rbf(E_dist_flat, 0.0, 20.0)

        X_src = X_ca.unsqueeze(2).expand(-1, -1, k, -1).flatten(1, 2)
        X_dest = torch.gather(X_ca, 1, edge_index[1].unsqueeze(-1).expand(bsz, maxlen * k, 3))
        cm_src = coord_mask.unsqueeze(2).expand(-1, -1, k).flatten(1, 2)
        cm_dest = torch.gather(coord_mask, 1, edge_index[1].expand(bsz, maxlen * k))
        E_vec = X_src - X_dest
        E_vec_mean = torch.sum(E_vec * E_coord_mask, dim=1, keepdim=True) / torch.sum(
            E_coord_mask, dim=1, keepdim=True
        ).clamp(min=1)
        E_vec = E_vec * E_coord_mask + E_vec_mean * (~E_coord_mask)
        edge_s = _nan_to_num(torch.cat([D_rbf, pos_emb], dim=-1))
        edge_v = _nan_to_num(_normalize(E_vec).unsqueeze(-2))
        edge_s = torch.cat(
            [
                edge_s,
                (~cm_src).to(dtype=edge_s.dtype).unsqueeze(-1),
                (~cm_dest).to(dtype=edge_s.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        edge_index[:, ~E_residue_mask] = -1
        edge_index[:, ~E_coord_mask.squeeze(-1)] = -1
        return (edge_s, edge_v), edge_index.transpose(0, 1)

    def forward(
        self,
        coords: torch.Tensor,
        coord_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        with torch.no_grad():
            node_feats = _gvp_node_features(coords, coord_mask, with_coord_mask=True)
            edge_feats, edge_index = self._get_edge_features(coords, coord_mask, padding_mask)
        ns, nv = self.embed_node(node_feats)
        es, ev = self.embed_edge(edge_feats)
        ns = ns + self.embed_confidence(_rbf(confidence, 0.0, 1.0))

        # Flatten graph across batch
        bsz, n_res = ns.shape[0], ns.shape[1]
        node_emb = (torch.flatten(ns, 0, 1), torch.flatten(nv, 0, 1))
        edge_emb = (torch.flatten(es, 0, 1), torch.flatten(ev, 0, 1))
        edge_mask = torch.any(edge_index != -1, dim=1)
        edge_index = edge_index + (torch.arange(bsz, device=edge_index.device) * n_res).unsqueeze(-1).unsqueeze(-1)
        edge_index = edge_index.permute(1, 0, 2).flatten(1, 2)
        edge_mask = edge_mask.flatten()
        edge_index = edge_index[:, edge_mask]
        edge_emb = (edge_emb[0][edge_mask], edge_emb[1][edge_mask])
        return node_emb, edge_emb, edge_index


class _GVPEncoder(nn.Module):
    def __init__(self, num_encoder_layers: int = 4, top_k_neighbors: int = 30) -> None:
        super().__init__()
        self.embed_graph = _GVPGraphEmbedding(top_k_neighbors=top_k_neighbors)
        self.encoder_layers = nn.ModuleList(
            [_GVPConvLayer((1024, 256), (32, 1), vector_gate=True, n_message=3, eps=1e-4) for _ in range(num_encoder_layers)]
        )

    def forward(
        self,
        coords: torch.Tensor,
        coord_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_emb, edge_emb, edge_index = self.embed_graph(coords, coord_mask, padding_mask, confidence)
        for layer in self.encoder_layers:
            node_emb, edge_emb = layer(node_emb, edge_index, edge_emb)
        bsz = coords.shape[0]
        return (
            node_emb[0].reshape(bsz, -1, node_emb[0].shape[1]),
            node_emb[1].reshape(bsz, -1, node_emb[1].shape[1], node_emb[1].shape[2]),
        )


class _SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, padding_idx: int = 1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.weights: torch.Tensor | None = None

    def _get_embedding(self, num_embeddings: int) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float32).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if self.padding_idx is not None:
            emb[self.padding_idx, :] = 0.0
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = x.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = self._get_embedding(max_pos)
        self.weights = self.weights.to(device=x.device, dtype=self._float_tensor.dtype)
        mask = x.ne(self.padding_idx)
        range_buf = torch.arange(seq_len, device=x.device).expand_as(x) + self.padding_idx + 1
        positions = range_buf * mask.long() + self.padding_idx * (1 - mask.long())
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()


class _FairseqMultiheadAttention(nn.Module):
    """Exact state-dict match for fair-esm ``MultiheadAttention`` (q/k/v/out_proj)."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (T, B, C)
        tgt_len, bsz, embed_dim = x.shape
        q = self.q_proj(x).view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = self.k_proj(x).view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = self.v_proj(x).view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        attn_mask: torch.Tensor | None = None
        if key_padding_mask is not None and bool(key_padding_mask.any()):
            attn_mask = torch.zeros((bsz, 1, 1, tgt_len), device=x.device, dtype=q.dtype)
            attn_mask.masked_fill_(key_padding_mask[:, None, None, :].bool(), float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, embed_dim)
        return self.out_proj(out)


class _IF1TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int = 512, ffn_dim: int = 2048, num_heads: int = 8) -> None:
        super().__init__()
        self.self_attn = _FairseqMultiheadAttention(embed_dim, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, encoder_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        res = x
        x = self.self_attn(self.self_attn_layer_norm(x), key_padding_mask=encoder_padding_mask)
        x = res + x
        res = x
        x = self.fc2(F.relu(self.fc1(self.final_layer_norm(x))))
        return res + x


class _PureGVPTransformerEncoder(nn.Module):
    """Exact state-dict match for ``encoder.encoder.*`` inside ``dplm2_struct_tokenizer.ckpt``."""

    def __init__(self) -> None:
        super().__init__()
        embed_dim = 512
        self.padding_idx = 1
        self.mask_idx = 32
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_tokens = nn.Embedding(35, embed_dim, padding_idx=self.padding_idx)
        self.embed_positions = _SinusoidalPositionalEmbedding(embed_dim, padding_idx=self.padding_idx)
        self.embed_gvp_input_features = nn.Linear(15, embed_dim)
        self.embed_confidence = nn.Linear(16, embed_dim)
        self.embed_dihedrals = _DihedralFeatures(embed_dim)
        self.gvp_encoder = _GVPEncoder(num_encoder_layers=4, top_k_neighbors=30)
        self.embed_gvp_output = nn.Linear(1024 + 3 * 256, embed_dim)
        self.layers = nn.ModuleList([_IF1TransformerEncoderLayer(embed_dim, 2048, 8) for _ in range(8)])
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        coords: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        coord_mask = torch.all(torch.all(torch.isfinite(coords), dim=-1), dim=-1)
        coords_clean = _nan_to_num(coords)
        mask_tokens = (
            encoder_padding_mask.long() * self.padding_idx + (~encoder_padding_mask).long() * self.mask_idx
        )
        tok_emb = self.embed_tokens(mask_tokens) * self.embed_scale
        dih_emb = self.embed_dihedrals(coords_clean)
        gvp_s, gvp_v = self.gvp_encoder(coords_clean, coord_mask, encoder_padding_mask, confidence)
        R = _get_rotation_frames(coords_clean)
        gvp_out_feats = torch.cat([gvp_s, _rotate(gvp_v, R.transpose(-2, -1)).flatten(-2, -1)], dim=-1)
        gvp_out_emb = self.embed_gvp_output(gvp_out_feats)
        conf_emb = self.embed_confidence(_rbf(confidence, 0.0, 1.0))

        in_s, in_v = _gvp_node_features(coords_clean, coord_mask, with_coord_mask=False)
        in_feats = torch.cat([in_s, _rotate(in_v, R.transpose(-2, -1)).flatten(-2, -1)], dim=-1)
        gvp_in_emb = self.embed_gvp_input_features(in_feats)

        x = tok_emb + dih_emb + gvp_out_emb + conf_emb + gvp_in_emb + self.embed_positions(mask_tokens)
        x = x * (1.0 - encoder_padding_mask.unsqueeze(-1).to(dtype=x.dtype))
        x = x.transpose(0, 1)  # (L, B, 512)
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=encoder_padding_mask)
        x = self.layer_norm(x)
        return x.transpose(0, 1)  # (B, L, 512)


class StructTokenizerEncoder(nn.Module):
    """Self-contained DPLM-2 3D backbone -> discrete structure token encoder (zero OpenFold/fair-esm)."""

    def __init__(self, ckpt_path: Path | str, *, device: str = "cpu") -> None:
        super().__init__()
        self.ckpt_path = Path(ckpt_path)
        self.device_str = str(device)
        self.codebook_dim = 13
        self.codebook_size = STRUCT_CODEBOOK_SIZE
        self.encoder = _PureGVPTransformerEncoder()
        self.pre_quant = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 13),
            nn.ReLU(),
            nn.Linear(13, 13),
        )
        self._load_weights(self.ckpt_path)
        self.to(device=self.device_str, dtype=torch.float32)
        self.eval()
        self.requires_grad_(False)

    def _load_weights(self, ckpt_path: Path) -> None:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        enc_sd: dict[str, torch.Tensor] = {}
        pq_sd: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if k.startswith("encoder.encoder."):
                enc_sd[k[len("encoder.encoder.") :]] = v
            elif k.startswith("pre_quant."):
                pq_sd[k[len("pre_quant.") :]] = v
        # Strict loading guarantees 100% parameter correspondence on all 282 encoder + pre_quant tensors!
        self.encoder.load_state_dict(enc_sd, strict=True)
        self.pre_quant.load_state_dict(pq_sd, strict=True)

    @classmethod
    def from_pretrained(cls, model_root: Path | str | None = None, *, device: str = "cpu") -> "StructTokenizerEncoder":
        sdir = resolve_dplm2_dir("dplm2-struct-tokenizer", model_root)
        return cls(sdir / "dplm2_struct_tokenizer.ckpt", device=device)

    @torch.no_grad()
    def encode_backbone(
        self,
        coords: np.ndarray | torch.Tensor,
        res_mask: np.ndarray | torch.Tensor | None = None,
    ) -> dict[str, np.ndarray]:
        """Encode ``(L, 3+, 3)`` ``(N, CA, C[, O])`` backbone into 0..8191 codes & DPLM-2 token IDs.

        Returns dict with:
        * ``code_indices``: ``(L,)`` int64 in ``0..8191``
        * ``struct_token_ids``: ``(L,)`` int64 in ``36..8227`` (``36 + code_indices``)
        * ``lfq_bits``: ``(L, 13)`` float32 in ``{-1.0, +1.0}``
        * ``res_mask``: ``(L,)`` bool
        """
        arr = np.asarray(coords, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[1] >= 3 and arr.shape[2] == 3:
            c3 = arr[:, :3, :]
        else:
            raise ValueError(f"Expected coords shape (L, 3+, 3), got {arr.shape}")
        length = c3.shape[0]
        finite_mask = np.isfinite(c3).all(axis=(-1, -2))
        if res_mask is not None:
            rm = np.asarray(res_mask, dtype=bool) & finite_mask
        else:
            rm = finite_mask

        coords_t = torch.from_numpy(c3).unsqueeze(0).to(device=self.device_str, dtype=torch.float32)
        mask_t = torch.from_numpy(rm).unsqueeze(0).to(device=self.device_str, dtype=torch.bool)
        padding_mask = torch.zeros((1, length), device=self.device_str, dtype=torch.bool)
        # Match GVPTransformerEncoderWrapper2.forward verbatim: mask missing coords to NaN, confidence = mask.float()
        coords_masked = coords_t.masked_fill(~mask_t[:, :, None, None], float("nan"))
        confidence = mask_t.float()

        enc_feats = self.encoder(coords_masked, padding_mask, confidence)
        pre_q = self.pre_quant(enc_feats) * mask_t.float().unsqueeze(-1)
        quant = torch.where(pre_q > 0.0, torch.ones_like(pre_q), -torch.ones_like(pre_q))
        bits_01 = (quant > 0.0).to(dtype=torch.long)
        codes = lfq_bits_to_code_indices(bits_01, codebook_dim=self.codebook_dim)[0]
        # Missing positions map to code 0 (matching structok_lfq.py:156)
        codes = torch.where(mask_t[0], codes, torch.zeros_like(codes))
        codes_np = codes.cpu().numpy().astype(np.int64)
        token_ids_np = np.where(rm, codes_np + STRUCT_VOCAB_OFFSET, STRUCT_UNK_ID).astype(np.int64)
        return {
            "code_indices": codes_np,
            "struct_token_ids": token_ids_np,
            "lfq_bits": quant[0].cpu().numpy().astype(np.float32),
            "res_mask": rm.astype(bool),
        }


# --------------------------------------------------------------------------- #
# Pure-PyTorch DPLM-2 & DPLM-2-Bit ESM Backbones (Apache-2.0 — ByteDance 2025)
# Compatible with transformers 4.57.3 (zero byprot dependency!)
# --------------------------------------------------------------------------- #


class _ModifiedRotaryEmbedding(RotaryEmbedding):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.aa_type = AA_TYPE
        self.struct_type = STRUCT_TYPE

    def _update_cos_sin_tables(self, x: torch.Tensor, type_ids: torch.Tensor | None = None, seq_dimension: int = 2):
        seq_len = int(x.shape[seq_dimension])
        both_mods = type_ids is not None and bool((type_ids == self.aa_type).any() and (type_ids == self.struct_type).any())
        if both_mods:
            seq_len = seq_len // 2
        if seq_len != self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
        return self._cos_cached, self._sin_cached, both_mods

    def forward(self, q: torch.Tensor, k: torch.Tensor, type_ids: torch.Tensor | None = None):
        cos, sin, both_mods = self._update_cos_sin_tables(k, type_ids=type_ids, seq_dimension=-2)
        if both_mods:
            q1, q2 = q.chunk(2, dim=-2)
            k1, k2 = k.chunk(2, dim=-2)
            q = torch.cat((apply_rotary_pos_emb(q1, cos, sin), apply_rotary_pos_emb(q2, cos, sin)), dim=-2)
            k = torch.cat((apply_rotary_pos_emb(k1, cos, sin), apply_rotary_pos_emb(k2, cos, sin)), dim=-2)
        else:
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)
        return q.to(dtype=q.dtype), k.to(dtype=k.dtype)


class _ModifiedEsmSelfAttention(EsmSelfAttention):
    def __init__(self, config, position_embedding_type=None) -> None:
        super().__init__(config, position_embedding_type)
        self.rotary_embeddings = _ModifiedRotaryEmbedding(dim=self.attention_head_size)

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(new_shape).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions: bool = False,
        type_ids: torch.Tensor | None = None,
    ):
        query_layer = self._transpose_for_scores(self.query(hidden_states))
        key_layer = self._transpose_for_scores(self.key(hidden_states))
        value_layer = self._transpose_for_scores(self.value(hidden_states))
        query_layer = query_layer * (self.attention_head_size**-0.5)

        if self.position_embedding_type == "rotary":
            q_dt = query_layer.dtype
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer, type_ids)
            query_layer = query_layer.to(dtype=q_dt)
            key_layer = key_layer.to(dtype=q_dt)

        attn_mask = attention_mask.to(dtype=query_layer.dtype) if attention_mask is not None else None
        context_layer = F.scaled_dot_product_attention(
            query_layer.contiguous(),
            key_layer.contiguous(),
            value_layer.contiguous(),
            attn_mask=attn_mask,
            scale=1.0,
        )
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size()[:-2] + (self.all_head_size,))
        return (context_layer,)


class _ModifiedEsmAttention(EsmAttention):
    def __init__(self, config) -> None:
        nn.Module.__init__(self)
        self.self = _ModifiedEsmSelfAttention(config)
        self.output = EsmSelfOutput(config)
        self.pruned_heads = set()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, type_ids=None):
        self_outputs = self.self(
            self.LayerNorm(hidden_states),
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            type_ids=type_ids,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        return (attention_output,) + self_outputs[1:]


class _ModifiedEsmLayer(EsmLayer):
    def __init__(self, config) -> None:
        nn.Module.__init__(self)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = _ModifiedEsmAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, type_ids=None):
        attn_out = self.attention(hidden_states, attention_mask=attention_mask, type_ids=type_ids)[0]
        layer_output = self.feed_forward_chunk(attn_out)
        return (layer_output,)


class _ModifiedEsmEncoder(EsmEncoder):
    def __init__(self, config) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.layer = nn.ModuleList([_ModifiedEsmLayer(config) for _ in range(config.num_hidden_layers)])
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        type_ids: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        all_hidden_states = () if output_hidden_states else None
        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore[operator]
            hidden_states = layer_module(hidden_states, attention_mask=attention_mask, type_ids=type_ids)[0]
        if self.emb_layer_norm_after is not None:
            hidden_states = self.emb_layer_norm_after(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore[operator]
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class _ModifiedEsmModel(EsmModel):
    def __init__(self, config, add_pooling_layer: bool = False) -> None:
        EsmPreTrainedModel.__init__(self, config)
        self.config = config
        self.embeddings = EsmEmbeddings(config)
        self.encoder = _ModifiedEsmEncoder(config)
        self.pooler = EsmPooler(config) if add_pooling_layer else None
        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads,
            bias=True,
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        type_ids: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        input_shape = input_ids.size() if input_ids is not None else inputs_embeds.size()[:-1]  # type: ignore[union-attr]
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=(input_ids if input_ids is not None else inputs_embeds).device)  # type: ignore[union-attr]
        if attention_mask.dim() == 4:
            extended_attention_mask = attention_mask
        else:
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # Match official dplm2_modeling_esm.py line 597 verbatim (compatible with transformers 4.57.3).
        emb_mask = input_ids.ne(self.config.pad_token_id) if input_ids is not None else None
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=emb_mask,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_hidden_states=output_hidden_states,
            type_ids=type_ids,
        )
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
        )


class EsmForDPLM2(EsmForMaskedLM):
    """Exact state-dict-compatible DPLM-2 model (``dplm2-150m``, ``dplm2-650m``, ``dplm2-3b``)."""

    def __init__(self, config) -> None:
        config.tie_word_embeddings = False
        EsmPreTrainedModel.__init__(self, config)
        self.esm = _ModifiedEsmModel(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)
        self.post_init()
        self.pad_id = config.pad_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        type_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> dict[str, Any]:
        outputs = self.esm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            type_ids=type_ids,
        )
        seq_out = outputs.last_hidden_state
        logits = self.lm_head(seq_out)
        res: dict[str, Any] = {"logits": logits, "last_hidden_state": seq_out}
        if output_hidden_states:
            res["all_hidden_states"] = outputs.hidden_states
        return res


class EsmForDPLM2Bit(EsmForMaskedLM):
    """Exact state-dict-compatible DPLM-2-Bit model (``dplm2-bit-650m``)."""

    def __init__(self, config) -> None:
        config.tie_word_embeddings = False
        EsmPreTrainedModel.__init__(self, config)
        self.esm = _ModifiedEsmModel(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)
        codebook_dim = int(getattr(config, "codebook_embed_dim", 13))
        struct_config = deepcopy(config)
        struct_config.vocab_size = codebook_dim * 2
        self.lm_head_struct = EsmLMHead(struct_config)
        self.struct_bos_emb = nn.Parameter(torch.randn(config.hidden_size))
        self.struct_eos_emb = nn.Parameter(torch.randn(config.hidden_size))
        self.struct_mask_emb = nn.Parameter(torch.randn(config.hidden_size))
        self.quant2emb = nn.Linear(codebook_dim, config.hidden_size)
        self.post_init()
        self.pad_id = config.pad_token_id
        self.codebook_embed_dim = codebook_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        type_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> dict[str, Any]:
        outputs = self.esm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            type_ids=type_ids,
        )
        seq_out = outputs.last_hidden_state
        struct_h, aa_h = seq_out.chunk(2, dim=1)
        res: dict[str, Any] = {
            "aatype_logits": self.lm_head(aa_h),
            "struct_logits": self.lm_head_struct(struct_h),
            "last_hidden_state": seq_out,
        }
        if output_hidden_states:
            res["all_hidden_states"] = outputs.hidden_states
        return res


# --------------------------------------------------------------------------- #
# Helper Utilities
# --------------------------------------------------------------------------- #


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _encode_aa_to_ids(sequence: str) -> list[int]:
    ids = []
    for i, ch in enumerate(sequence):
        tid = DPLM2_AA_TO_ID.get(ch)
        if tid is None:
            raise ValueError(f"Non-AA20 symbol {ch!r} at pos {i} in sequence")
        ids.append(tid)
    return ids


def _normalize_struct_token_ids(struct_tokens: Sequence[int] | np.ndarray, expected_len: int) -> list[int]:
    """Accept either 0..8191 LFQ code indices OR 36..8227 DPLM-2 vocab token IDs."""
    arr = np.asarray(struct_tokens, dtype=np.int64).reshape(-1)
    if arr.shape[0] != expected_len:
        raise ValueError(f"struct_tokens length {arr.shape[0]} != sequence length {expected_len}")
    out: list[int] = []
    for val in arr.tolist():
        if val == STRUCT_MASK_ID or val == STRUCT_UNK_ID:
            out.append(int(val))
        elif 0 <= val < STRUCT_CODEBOOK_SIZE:
            out.append(int(val) + STRUCT_VOCAB_OFFSET)
        elif STRUCT_VOCAB_OFFSET <= val < STRUCT_VOCAB_OFFSET + STRUCT_CODEBOOK_SIZE:
            out.append(int(val))
        else:
            raise ValueError(f"Out-of-range structure token value {val}")
    return out


def _top_p_filter(logprobs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logprobs
    ordered, order = torch.sort(logprobs, descending=True)
    probs = ordered.exp()
    cumulative = torch.cumsum(probs, dim=-1)
    keep = (cumulative - probs) < top_p
    keep[..., 0] = True
    masked = torch.full_like(logprobs, float("-inf"))
    masked.scatter_(-1, order, torch.where(keep, ordered, torch.full_like(ordered, float("-inf"))))
    return F.log_softmax(masked, dim=-1)


# --------------------------------------------------------------------------- #
# High-Level DPLM-2 Adapter
# --------------------------------------------------------------------------- #


class DPLM2Adapter:
    """Unified adapter over ``dplm2-150m``, ``dplm2-650m``, ``dplm2-bit-650m``, and ``dplm2-3b``.

    Supports ``mode="seq_only"`` (no 3D coordinates; ``<mask_struct>`` + official
    ``single_modality`` cross-modal attention isolation) and
    ``mode="struct_conditioned"`` (3D backbone coordinates or pre-encoded
    ``struct_tokens``).
    """

    def __init__(
        self,
        arm_id: str,
        *,
        mode: str = "seq_only",
        model_root: Path | str | None = None,
        device: str = "cpu",
        dtype: str | None = None,
        struct_tokenizer: StructTokenizerEncoder | None = None,
        load_struct_tokenizer: bool | None = None,
    ) -> None:
        if arm_id not in DPLM2_SPECS:
            raise KeyError(f"Unknown DPLM-2 arm_id {arm_id!r}; expected one of {sorted(DPLM2_SPECS)}")
        if mode not in ("seq_only", "struct_conditioned"):
            raise ValueError(f"Unsupported DPLM-2 mode {mode!r}; expected 'seq_only' or 'struct_conditioned'")

        self.spec = DPLM2_SPECS[arm_id]
        self.arm = arm_id
        self.mode = mode
        self.device = str(device)
        self.model_root = model_root

        dt_raw = (dtype or self.spec.default_dtype).strip().lower()
        if dt_raw in ("fp32", "float32", "auto", "default"):
            self.dtype = "float32"
            self.torch_dtype = torch.float32
        elif dt_raw in ("bf16", "bfloat16"):
            self.dtype = "bfloat16"
            self.torch_dtype = torch.bfloat16
        else:
            raise ValueError(f"{arm_id}: unsupported dtype {dtype!r}; use float32 or bfloat16")

        self.checkpoint_dir = resolve_dplm2_dir(self.spec.dir_name, model_root)
        config = AutoConfig.from_pretrained(str(self.checkpoint_dir), local_files_only=True)
        model_cls = EsmForDPLM2Bit if self.spec.is_bit else EsmForDPLM2
        self.model = model_cls.from_pretrained(
            str(self.checkpoint_dir),
            config=config,
            torch_dtype=self.torch_dtype,
            local_files_only=True,
        )
        self.model.to(device=self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        self._struct_tokenizer: StructTokenizerEncoder | None = struct_tokenizer
        should_load_st = (mode == "struct_conditioned") if load_struct_tokenizer is None else bool(load_struct_tokenizer)
        if should_load_st and self._struct_tokenizer is None:
            self._struct_tokenizer = StructTokenizerEncoder.from_pretrained(model_root, device=self.device)

        self.aa20_ids = list(DPLM2_AA20_IDS)
        self.aa20_ids_t = torch.as_tensor(self.aa20_ids, device=self.device, dtype=torch.long)
        self.mask_id = AA_MASK_ID
        self.max_residues = 1022

    @property
    def struct_tokenizer(self) -> StructTokenizerEncoder:
        if self._struct_tokenizer is None:
            self._struct_tokenizer = StructTokenizerEncoder.from_pretrained(self.model_root, device=self.device)
        return self._struct_tokenizer

    def _precision_ctx(self):
        if self.dtype == "float32" and self.device.startswith("cuda"):
            return fp32_matmul_context(torch)
        return nullcontext()

    def provenance(self) -> dict[str, Any]:
        cfg = self.checkpoint_dir / "config.json"
        return {
            "arm": self.arm,
            "family": "DPLM-2-Bit" if self.spec.is_bit else "DPLM-2",
            "paradigm": "multimodal_diffusion",
            "mode": self.mode,
            "is_bit": self.spec.is_bit,
            "num_layers": int(self.model.config.num_hidden_layers),
            "hidden_size": int(self.model.config.hidden_size),
            "dtype_requested": self.dtype,
            "dtype_realised": "/".join(sorted({str(p.dtype) for p in self.model.parameters()})),
            "fp32_matmul_policy": (
                requested_fp32_matmul_policy()
                if self.dtype == "float32"
                else {"declared_non_fp32_opt_in": self.dtype}
            ),
            "parameters": int(sum(p.numel() for p in self.model.parameters())),
            "device": self.device,
            "checkpoint_dir": str(self.checkpoint_dir),
            "config_sha256": _sha256_file(cfg) if cfg.is_file() else None,
        }

    def encode_backbone(self, coords: np.ndarray, res_mask: np.ndarray | None = None) -> np.ndarray:
        """Encode ``(L, 3+, 3)`` coords -> ``(L,)`` DPLM-2 struct token IDs in ``36..8227``."""
        return self.struct_tokenizer.encode_backbone(coords, res_mask=res_mask)["struct_token_ids"]

    @staticmethod
    def _modality_type(input_ids: torch.Tensor) -> torch.Tensor:
        input_mask = input_ids.ne(PAD_ID)
        mtype = ((input_ids < 33) & input_mask).int()
        mtype[~input_mask] = PAD_TYPE
        return mtype

    def _resolve_struct_ids(
        self,
        length: int,
        *,
        coords: np.ndarray | None = None,
        struct_tokens: Sequence[int] | np.ndarray | None = None,
        mode: str | None = None,
    ) -> tuple[list[int], bool]:
        eff_mode = mode or self.mode
        if eff_mode == "seq_only":
            return [STRUCT_MASK_ID] * length, True
        if struct_tokens is not None:
            return _normalize_struct_token_ids(struct_tokens, length), False
        if coords is not None:
            toks = self.encode_backbone(coords)
            return _normalize_struct_token_ids(toks, length), False
        raise ValueError(
            f"{self.arm}: mode='struct_conditioned' requires either coords or struct_tokens"
        )

    def _build_paired_ids(
        self,
        aa_ids_list: Sequence[Sequence[int]],
        struct_ids_list: Sequence[Sequence[int]],
    ) -> tuple[torch.Tensor, int]:
        """Build ``(B, 2 * L_half)`` tensor ``[struct_cls, struct..., struct_eos, pad..., aa_cls, aa..., aa_eos, pad...]``."""
        max_len = max(len(a) for a in aa_ids_list)
        half = max_len + 2
        bsz = len(aa_ids_list)
        out = torch.full((bsz, 2 * half), PAD_ID, device=self.device, dtype=torch.long)
        for i, (a_ids, s_ids) in enumerate(zip(aa_ids_list, struct_ids_list)):
            L = len(a_ids)
            if len(s_ids) != L:
                raise ValueError(f"AA length {L} != struct length {len(s_ids)}")
            out[i, 0] = STRUCT_CLS_ID
            out[i, 1 : 1 + L] = torch.as_tensor(s_ids, device=self.device, dtype=torch.long)
            out[i, 1 + L] = STRUCT_EOS_ID
            out[i, half] = AA_CLS_ID
            out[i, half + 1 : half + 1 + L] = torch.as_tensor(a_ids, device=self.device, dtype=torch.long)
            out[i, half + 1 + L] = AA_EOS_ID
        return out, half

    def _forward_paired(
        self,
        input_ids: torch.Tensor,
        *,
        single_modality: bool = False,
        output_hidden_states: bool = False,
    ) -> dict[str, Any]:
        """Exact official DPLM-2 / DPLM-2-Bit forward pass on paired ``[struct_half, aa_half]`` tensor."""
        bsz, total_len = input_ids.shape
        half = total_len // 2
        input_mask = input_ids.ne(PAD_ID)
        type_ids = self._modality_type(input_ids)
        num_heads = self.model.config.num_attention_heads

        # [B, num_heads, 2*half, 2*half] matching dplm2.py:237-256 verbatim
        attn_bias = self.model.esm.get_extended_attention_mask(input_mask, input_ids.shape).repeat(
            1, num_heads, total_len, 1
        )
        if single_modality:
            struct_bias, aa_bias = attn_bias.chunk(2, dim=-2)
            struct_bias = struct_bias.clone()
            aa_bias = aa_bias.clone()
            struct_bias[:, :, :, half:] = -math.inf
            aa_bias[:, :, :, :half] = -math.inf
            attn_bias = torch.cat([struct_bias, aa_bias], dim=-2)

        if self.spec.is_bit:
            s_ids, a_ids = input_ids.chunk(2, dim=1)
            _, a_mask = input_mask.chunk(2, dim=1)
            aa_embeds = self.model.esm.embeddings(a_ids, attention_mask=a_mask)
            quant = code_indices_to_lfq_bits(s_ids - STRUCT_VOCAB_OFFSET, self.model.codebook_embed_dim).to(
                dtype=self.model.quant2emb.weight.dtype
            )
            s_embeds = self.model.quant2emb(quant).to(dtype=aa_embeds.dtype)
            s_embeds[:, 0] = self.model.struct_bos_emb.to(dtype=aa_embeds.dtype)
            s_embeds[s_ids == STRUCT_EOS_ID] = self.model.struct_eos_emb.to(dtype=aa_embeds.dtype)
            s_embeds[s_ids == STRUCT_MASK_ID] = self.model.struct_mask_emb.to(dtype=aa_embeds.dtype)
            s_embeds[s_ids == PAD_ID] = 0.0
            inputs_embeds = torch.cat([s_embeds, aa_embeds], dim=1)
        else:
            inputs_embeds = self.model.esm.embeddings(input_ids, attention_mask=input_mask)

        return self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attn_bias,
            type_ids=type_ids,
            output_hidden_states=output_hidden_states,
        )

    def _extract_aa_logits(self, out: dict[str, Any], half: int) -> torch.Tensor:
        """Return AA-half logits ``(B, half, vocab_aa)`` for both ``EsmForDPLM2`` and ``EsmForDPLM2Bit``."""
        if self.spec.is_bit:
            return out["aatype_logits"]
        return out["logits"][:, half:, :]

    @torch.no_grad()
    def site_conditional_logprobs(
        self,
        sequence: str,
        *,
        positions: Sequence[int] | None = None,
        coords: np.ndarray | None = None,
        struct_tokens: Sequence[int] | np.ndarray | None = None,
        mode: str | None = None,
        batch_size: int = 16,
    ) -> np.ndarray:
        """Exact leave-one-out masked marginal AA20 log-probabilities ``(len(positions), 20)``."""
        length = len(sequence)
        pos_list = list(range(length)) if positions is None else [int(p) for p in positions]
        if any(not (0 <= p < length) for p in pos_list):
            raise ValueError(f"Position out of range for length {length}")
        if not pos_list:
            return np.zeros((0, 20), dtype=np.float32)

        aa_ids = _encode_aa_to_ids(sequence)
        s_ids, single_mod = self._resolve_struct_ids(length, coords=coords, struct_tokens=struct_tokens, mode=mode)
        base_paired, half = self._build_paired_ids([aa_ids], [s_ids])

        out = np.empty((len(pos_list), 20), dtype=np.float32)
        bs = max(1, int(batch_size))
        with self._precision_ctx():
            for start in range(0, len(pos_list), bs):
                chunk = pos_list[start : start + bs]
                cur = len(chunk)
                batch = base_paired.expand(cur, -1).clone()
                row_ar = torch.arange(cur, device=self.device, dtype=torch.long)
                pos_t = torch.as_tensor(chunk, device=self.device, dtype=torch.long)
                batch[row_ar, half + 1 + pos_t] = AA_MASK_ID
                net_out = self._forward_paired(batch, single_modality=single_mod)
                aa_logits = self._extract_aa_logits(net_out, half)
                site_logits = aa_logits[row_ar, 1 + pos_t, :].index_select(-1, self.aa20_ids_t).to(dtype=torch.float32)
                out[start : start + cur] = F.log_softmax(site_logits, dim=-1).cpu().numpy()
        return out

    def native_site_conditional(
        self,
        sequence: str,
        index: int,
        *,
        coords: np.ndarray | None = None,
        struct_tokens: Sequence[int] | np.ndarray | None = None,
        mode: str | None = None,
    ) -> np.ndarray:
        return self.site_conditional_logprobs(
            sequence, positions=[index], coords=coords, struct_tokens=struct_tokens, mode=mode
        )[0]

    def exact_pll(
        self,
        sequence: str,
        *,
        coords: np.ndarray | None = None,
        struct_tokens: Sequence[int] | np.ndarray | None = None,
        mode: str | None = None,
        batch_size: int = 16,
    ) -> PositionVector:
        lp = self.site_conditional_logprobs(
            sequence, positions=None, coords=coords, struct_tokens=struct_tokens, mode=mode, batch_size=batch_size
        )
        aa_idx = np.asarray([AA20_INDEX[ch] for ch in sequence], dtype=np.int64)
        terms = np.maximum((-lp[np.arange(len(sequence)), aa_idx]).astype(np.float32), 0.0)
        nll_sum = float(terms.sum(dtype=np.float64))
        return PositionVector(terms=terms, nll_sum=nll_sum, pll=-nll_sum, paradigm="multimodal_diffusion")

    @torch.no_grad()
    def pll_8fold(
        self,
        sequence: str,
        *,
        coords: np.ndarray | None = None,
        struct_tokens: Sequence[int] | np.ndarray | None = None,
        mode: str | None = None,
        n_folds: int = 8,
    ) -> PositionVector:
        """Fast 8-fold stride-masked PLL matching R4/R5 context protocol."""
        length = len(sequence)
        aa_ids = _encode_aa_to_ids(sequence)
        s_ids, single_mod = self._resolve_struct_ids(length, coords=coords, struct_tokens=struct_tokens, mode=mode)
        base_paired, half = self._build_paired_ids([aa_ids], [s_ids])

        k = min(int(n_folds), length)
        batch = base_paired.expand(k, -1).clone()
        fold_positions = [list(range(f, length, k)) for f in range(k)]
        for f, pos_f in enumerate(fold_positions):
            batch[f, [half + 1 + p for p in pos_f]] = AA_MASK_ID

        terms = np.empty(length, dtype=np.float32)
        with self._precision_ctx():
            net_out = self._forward_paired(batch, single_modality=single_mod)
            aa_logits = self._extract_aa_logits(net_out, half)
            lp20 = F.log_softmax(aa_logits.index_select(-1, self.aa20_ids_t).to(dtype=torch.float32), dim=-1).cpu().numpy()
        for f, pos_f in enumerate(fold_positions):
            for p in pos_f:
                terms[p] = max(0.0, float(-lp20[f, 1 + p, AA20_INDEX[sequence[p]]]))
        nll_sum = float(terms.sum(dtype=np.float64))
        return PositionVector(terms=terms, nll_sum=nll_sum, pll=-nll_sum, paradigm="multimodal_diffusion")

    @torch.no_grad()
    def score_variants(
        self,
        wildtype: str,
        mutations: Sequence[str],
        *,
        coords: np.ndarray | None = None,
        struct_tokens: Sequence[int] | np.ndarray | None = None,
        mode: str | None = None,
        batch_size: int = 16,
        multi_exact_pll: bool = False,
    ) -> np.ndarray:
        """Score single and multi-mutant variants in ``seq_only`` or ``struct_conditioned`` mode.

        * Single mutants: exact leave-one-out masked marginal log-odds over unique mutated sites.
        * Multi-mutants: when ``multi_exact_pll=True`` (e.g., R3 cycles), exact
          ``PLL(mut_seq) - PLL(wt_seq)`` (or sum of joint-masked marginals when ``False``).
        """
        if not mutations:
            return np.zeros((0,), dtype=np.float64)
        length = len(wildtype)
        s_ids, _ = self._resolve_struct_ids(length, coords=coords, struct_tokens=struct_tokens, mode=mode)
        parsed = [parse_mutation(m, wildtype) for m in mutations]
        scores = np.empty(len(mutations), dtype=np.float64)

        single_indices = [i for i, t in enumerate(parsed) if len(t) == 1]
        multi_indices = [i for i, t in enumerate(parsed) if len(t) > 1]

        if single_indices:
            uniq_pos = sorted({parsed[i][0][0] for i in single_indices})
            pos_map = {p: r for r, p in enumerate(uniq_pos)}
            lp_mat = self.site_conditional_logprobs(
                wildtype, positions=uniq_pos, struct_tokens=s_ids, mode=mode, batch_size=batch_size
            )
            for idx in single_indices:
                pos, wt_aa, mut_aa = parsed[idx][0]
                row = pos_map[pos]
                scores[idx] = float(lp_mat[row, AA20_INDEX[mut_aa]] - lp_mat[row, AA20_INDEX[wt_aa]])

        if multi_indices:
            if multi_exact_pll:
                wt_pll = self.exact_pll(wildtype, struct_tokens=s_ids, mode=mode, batch_size=batch_size).pll
                for idx in multi_indices:
                    var_seq = apply_mutation(wildtype, mutations[idx])
                    scores[idx] = self.exact_pll(var_seq, struct_tokens=s_ids, mode=mode, batch_size=batch_size).pll - wt_pll
            else:
                # Jointly mask all mutated sites of each multi-mutant (exact masked-marginal multi-site score)
                aa_wt = _encode_aa_to_ids(wildtype)
                eff_mode = mode or self.mode
                single_mod = eff_mode == "seq_only"
                base_paired, half = self._build_paired_ids([aa_wt], [s_ids])
                bs = max(1, int(batch_size))
                with self._precision_ctx():
                    for start in range(0, len(multi_indices), bs):
                        chunk_idx = multi_indices[start : start + bs]
                        cur = len(chunk_idx)
                        batch = base_paired.expand(cur, -1).clone()
                        for b_i, m_idx in enumerate(chunk_idx):
                            for pos, _, _ in parsed[m_idx]:
                                batch[b_i, half + 1 + pos] = AA_MASK_ID
                        net_out = self._forward_paired(batch, single_modality=single_mod)
                        aa_logits = self._extract_aa_logits(net_out, half)
                        lp20 = F.log_softmax(
                            aa_logits.index_select(-1, self.aa20_ids_t).to(dtype=torch.float32), dim=-1
                        ).cpu().numpy()
                        for b_i, m_idx in enumerate(chunk_idx):
                            total = 0.0
                            for pos, wt_aa, mut_aa in parsed[m_idx]:
                                total += float(lp20[b_i, 1 + pos, AA20_INDEX[mut_aa]] - lp20[b_i, 1 + pos, AA20_INDEX[wt_aa]])
                            scores[m_idx] = total
        return scores

    @torch.no_grad()
    def extract_hidden_states(
        self,
        sequences: Sequence[str],
        *,
        coords_list: np.ndarray | Sequence[np.ndarray | None] | None = None,
        struct_tokens_list: Sequence[Sequence[int] | np.ndarray | None] | None = None,
        mode: str | None = None,
        batch_size: int = 8,
        return_projected_1024: bool = False,
    ) -> np.ndarray:
        """Extract unmasked (``p=0``) AA-half hidden states at ``blocks[(L-1)//2]`` & ``blocks[L-1]``.

        Pools ``[mid_mean, mid_cls, final_mean, final_cls]`` over the AA half
        (``cls`` = ``<cls_aa>`` at index ``half``, ``mean`` = residues ``half+1 : half+1+seq_len``),
        returning ``(N, 4, hidden_size)`` float32 (or ``(N, 1024)`` via ``project_four_blocks``
        where ``0:512`` = ``R_mid`` and ``512:1024`` = ``R_last``).
        """
        if not sequences:
            return np.zeros((0, 1024), dtype=np.float32) if return_projected_1024 else np.zeros((0, 4, self.spec.hidden_size), dtype=np.float32)

        eff_mode = mode or self.mode
        n_seq = len(sequences)
        n_layers = int(self.model.config.num_hidden_layers)
        # hidden_states tuple in ModifiedEsmEncoder: [emb, after_layer_0, ..., after_layer_{L-1}_norm]
        # blocks[(L-1)//2] output is hidden_states[(L-1)//2 + 1]; final block L-1 is hidden_states[L] (last).
        mid_hs_idx = ((n_layers - 1) // 2) + 1
        fin_hs_idx = n_layers

        # Resolve per-sequence structure tokens (supporting shared WT coords/tokens cleanly!)
        s_ids_all: list[list[int]] = []
        single_mod = eff_mode == "seq_only"
        if single_mod:
            s_ids_all = [[STRUCT_MASK_ID] * len(seq) for seq in sequences]
        else:
            if struct_tokens_list is not None:
                if isinstance(struct_tokens_list, np.ndarray) and struct_tokens_list.ndim == 1:
                    shared_toks = _normalize_struct_token_ids(struct_tokens_list, len(sequences[0]))
                    s_ids_all = [shared_toks for _ in sequences]
                else:
                    s_ids_all = [
                        _normalize_struct_token_ids(st, len(seq))  # type: ignore[arg-type]
                        for seq, st in zip(sequences, struct_tokens_list)
                    ]
            elif coords_list is not None:
                if isinstance(coords_list, np.ndarray) and coords_list.ndim == 3:
                    shared_toks = list(self.encode_backbone(coords_list))
                    s_ids_all = [shared_toks for _ in sequences]
                else:
                    s_ids_all = [
                        list(self.encode_backbone(c))  # type: ignore[arg-type]
                        for c in coords_list
                    ]
            else:
                raise ValueError(f"{self.arm}: struct_conditioned hidden extraction needs coords_list or struct_tokens_list")

        out = np.empty((n_seq, 4, int(self.model.config.hidden_size)), dtype=np.float32)
        bs = max(1, int(batch_size))
        with self._precision_ctx():
            for start in range(0, n_seq, bs):
                stop = min(start + bs, n_seq)
                chunk_seqs = sequences[start:stop]
                aa_chunk = [_encode_aa_to_ids(s) for s in chunk_seqs]
                st_chunk = s_ids_all[start:stop]
                paired, half = self._build_paired_ids(aa_chunk, st_chunk)
                net_out = self._forward_paired(paired, single_modality=single_mod, output_hidden_states=True)
                hs = net_out["all_hidden_states"]
                h_mid = hs[mid_hs_idx][:, half:, :].to(dtype=torch.float32)
                h_fin = hs[fin_hs_idx][:, half:, :].to(dtype=torch.float32)
                for b_i, seq in enumerate(chunk_seqs):
                    L = len(seq)
                    mid_mean = h_mid[b_i, 1 : 1 + L, :].mean(dim=0)
                    mid_cls = h_mid[b_i, 0, :]
                    fin_mean = h_fin[b_i, 1 : 1 + L, :].mean(dim=0)
                    fin_cls = h_fin[b_i, 0, :]
                    out[start + b_i] = torch.stack([mid_mean, mid_cls, fin_mean, fin_cls], dim=0).cpu().numpy()

        if return_projected_1024:
            return project_four_blocks(out)
        return out

    @torch.no_grad()
    def extract_contact_matrix(
        self,
        sequence: str,
        *,
        coords: np.ndarray | None = None,
        struct_tokens: Sequence[int] | np.ndarray | None = None,
        mode: str | None = None,
        batch_size: int = 16,
        apc: bool = True,
    ) -> dict[str, np.ndarray]:
        """E01 perturbation-propagation contact sensitivity ``J_{ij}`` (mask-i -> shift at j)."""
        length = len(sequence)
        aa_ids = _encode_aa_to_ids(sequence)
        s_ids, single_mod = self._resolve_struct_ids(length, coords=coords, struct_tokens=struct_tokens, mode=mode)
        base_paired, half = self._build_paired_ids([aa_ids], [s_ids])

        with self._precision_ctx():
            # Unmasked baseline AA20 log-probs across all L positions
            base_out = self._forward_paired(base_paired, single_modality=single_mod)
            base_lp = F.log_softmax(
                self._extract_aa_logits(base_out, half)[0, 1 : 1 + length, :]
                .index_select(-1, self.aa20_ids_t)
                .to(dtype=torch.float32),
                dim=-1,
            )  # (L, 20)

            raw = np.zeros((length, length), dtype=np.float64)
            bs = max(1, int(batch_size))
            for start in range(0, length, bs):
                stop = min(start + bs, length)
                cur = stop - start
                batch = base_paired.expand(cur, -1).clone()
                row_ar = torch.arange(cur, device=self.device, dtype=torch.long)
                pos_t = torch.arange(start, stop, device=self.device, dtype=torch.long)
                batch[row_ar, half + 1 + pos_t] = AA_MASK_ID
                net_out = self._forward_paired(batch, single_modality=single_mod)
                aa_logits = self._extract_aa_logits(net_out, half)[:, 1 : 1 + length, :]
                pert_lp = F.log_softmax(aa_logits.index_select(-1, self.aa20_ids_t).to(dtype=torch.float32), dim=-1)
                diff = torch.linalg.norm(pert_lp - base_lp.unsqueeze(0), dim=-1)  # (cur, L)
                raw[start:stop, :] = diff.cpu().numpy()

        np.fill_diagonal(raw, 0.0)
        sym = 0.5 * (raw + raw.T)
        if apc and length > 2:
            row_mean = sym.sum(axis=1) / float(length - 1)
            grand_mean = float(sym.sum()) / float(length * (length - 1))
            mat = sym - np.outer(row_mean, row_mean) / max(grand_mean, 1e-12) if grand_mean > 0 else sym.copy()
            np.fill_diagonal(mat, 0.0)
        else:
            mat = sym.copy()
        return {"sensitivity": sym.astype(np.float32), "couplings": mat.astype(np.float32)}

    @torch.no_grad()
    def score_r4_context(
        self,
        target: str,
        context_items: Sequence[str] = (),
        *,
        target_coords: np.ndarray | None = None,
        context_coords: Sequence[np.ndarray] = (),
        mode: str | None = None,
        budget_tokens: int = 1024,
        n_folds: int = 8,
    ) -> dict[str, Any]:
        """Score ``target`` via 8-fold stride-masked PLL with concatenated ``context_items`` prefix."""
        eff_mode = mode or self.mode
        single_mod = eff_mode == "seq_only"
        max_ctx = max(0, int(budget_tokens) - len(target) - 4)

        ctx_seqs_used: list[str] = []
        ctx_st_used: list[list[int]] = []
        used_len = 0
        for idx, c_seq in enumerate(context_items):
            if used_len + len(c_seq) > max_ctx:
                rem = max_ctx - used_len
                if rem <= 0:
                    break
                c_slice = c_seq[:rem]
                ctx_seqs_used.append(c_slice)
                if not single_mod and idx < len(context_coords):
                    ctx_st_used.append(list(self.encode_backbone(context_coords[idx][:rem])))
                else:
                    ctx_st_used.append([STRUCT_MASK_ID] * len(c_slice))
                used_len += len(c_slice)
                break
            ctx_seqs_used.append(c_seq)
            if not single_mod and idx < len(context_coords):
                ctx_st_used.append(list(self.encode_backbone(context_coords[idx])))
            else:
                ctx_st_used.append([STRUCT_MASK_ID] * len(c_seq))
            used_len += len(c_seq)

        full_seq = "".join(ctx_seqs_used) + target
        tgt_start = len(full_seq) - len(target)
        aa_ids = _encode_aa_to_ids(full_seq)
        if single_mod:
            s_ids = [STRUCT_MASK_ID] * len(full_seq)
        else:
            tgt_st = (
                list(self.encode_backbone(target_coords))
                if target_coords is not None
                else [STRUCT_MASK_ID] * len(target)
            )
            s_ids = [tid for part in ctx_st_used for tid in part] + tgt_st

        base_paired, half = self._build_paired_ids([aa_ids], [s_ids])
        k = min(int(n_folds), len(target))
        batch = base_paired.expand(k, -1).clone()
        fold_pos = [list(range(f, len(target), k)) for f in range(k)]
        for f, pos_f in enumerate(fold_pos):
            batch[f, [half + 1 + tgt_start + p for p in pos_f]] = AA_MASK_ID

        terms = np.empty(len(target), dtype=np.float32)
        with self._precision_ctx():
            net_out = self._forward_paired(batch, single_modality=single_mod)
            aa_logits = self._extract_aa_logits(net_out, half)
            lp20 = F.log_softmax(aa_logits.index_select(-1, self.aa20_ids_t).to(dtype=torch.float32), dim=-1).cpu().numpy()
        for f, pos_f in enumerate(fold_pos):
            for p in pos_f:
                terms[p] = max(0.0, float(-lp20[f, 1 + tgt_start + p, AA20_INDEX[target[p]]]))
        nll_sum = float(terms.sum(dtype=np.float64))
        return {
            "arm": self.arm,
            "mode": eff_mode,
            "n_target_residues": len(target),
            "n_context_residues": tgt_start,
            "n_context_items_used": len(ctx_seqs_used),
            "nll_sum": nll_sum,
            "pll": -nll_sum,
            "nll_per_residue": nll_sum / max(len(target), 1),
            "terms": terms,
        }

    # --- Diffusion Generation Suite (R5a Infilling, R4 Suffix, Exp 4.1/4.2 Full) ---

    @torch.no_grad()
    def _iterative_unmask(
        self,
        init_aa_ids: list[int],
        init_struct_ids: list[int],
        *,
        aa_editable_mask: np.ndarray,
        struct_editable_mask: np.ndarray,
        single_modality: bool,
        steps: int,
        seed: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        sampling_mode: str = "max_confidence",
        anneal_range: tuple[float, float] | None = None,
        profile_log_bias: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Iterative masked discrete diffusion sampler supporting AA-only, inverse folding & co-generation."""
        length = len(init_aa_ids)
        paired, half = self._build_paired_ids([init_aa_ids], [init_struct_ids])
        aa_edit = torch.as_tensor(aa_editable_mask, device=self.device, dtype=torch.bool)
        st_edit = torch.as_tensor(struct_editable_mask, device=self.device, dtype=torch.bool)

        # Mask editable positions initially
        paired[0, half + 1 : half + 1 + length][aa_edit] = AA_MASK_ID
        paired[0, 1 : 1 + length][st_edit] = STRUCT_MASK_ID

        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        bias_t = (
            torch.as_tensor(profile_log_bias, device=self.device, dtype=torch.float32)
            if profile_log_bias is not None
            else None
        )
        n_steps = max(1, int(steps))
        n_aa_total = int(aa_edit.sum().item())
        n_st_total = int(st_edit.sum().item())

        with self._precision_ctx():
            for step in range(n_steps):
                aa_rem_mask = (paired[0, half + 1 : half + 1 + length] == AA_MASK_ID) & aa_edit
                st_rem_mask = (paired[0, 1 : 1 + length] == STRUCT_MASK_ID) & st_edit
                if not (aa_rem_mask.any() or st_rem_mask.any()):
                    break

                frac_left = max(0.0, 1.0 - float(step + 1) / float(n_steps))
                if anneal_range is not None:
                    t_max, t_min = anneal_range
                    cur_temp = float(t_min + (t_max - t_min) * (1.0 - float(step) / float(n_steps)))
                else:
                    cur_temp = float(temperature)

                net_out = self._forward_paired(paired, single_modality=single_modality)

                # 1. Sample AA proposals on remaining masked AA positions
                if aa_rem_mask.any():
                    aa_logits = self._extract_aa_logits(net_out, half)[0, 1 : 1 + length, :]
                    aa20_logits = aa_logits.index_select(-1, self.aa20_ids_t).to(dtype=torch.float32)
                    if bias_t is not None:
                        aa20_logits = aa20_logits + bias_t
                    aa_lp = F.log_softmax(aa20_logits, dim=-1)
                    if top_p < 1.0:
                        aa_lp_samp = _top_p_filter(aa_lp, top_p)
                    else:
                        aa_lp_samp = aa_lp
                    if cur_temp <= 1e-6:
                        aa_choice_idx = aa_lp_samp.argmax(dim=-1)
                    else:
                        probs = F.softmax(aa_lp_samp / cur_temp, dim=-1)
                        aa_choice_idx = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
                    aa_scores = aa_lp.gather(-1, aa_choice_idx.unsqueeze(-1)).squeeze(-1)
                    aa_token_ids = self.aa20_ids_t[aa_choice_idx]

                    rem_idx = torch.where(aa_rem_mask)[0]
                    target_remaining = int(round(n_aa_total * frac_left))
                    n_unmask = max(1, int(rem_idx.numel()) - target_remaining)
                    if step == n_steps - 1:
                        n_unmask = int(rem_idx.numel())
                    if sampling_mode == "uniform_random":
                        perm = torch.randperm(rem_idx.numel(), generator=gen, device=self.device)
                        chosen_pos = rem_idx[perm[:n_unmask]]
                    else:
                        # max_confidence / co_generation: highest log-prob (with tiny stochastic tie-break if co_generation)
                        sc = aa_scores[rem_idx]
                        if sampling_mode == "co_generation" and step < n_steps - 1:
                            u = torch.rand(sc.shape, generator=gen, device=self.device).clamp(1e-7, 1.0 - 1e-7)
                            sc = sc + (1.0 - float(step) / float(n_steps)) * (-torch.log(-torch.log(u)))
                        topk = torch.topk(sc, k=n_unmask, largest=True).indices
                        chosen_pos = rem_idx[topk]
                    paired[0, half + 1 + chosen_pos] = aa_token_ids[chosen_pos]

                # 2. Sample structure proposals if co-generating structure tokens
                if st_rem_mask.any():
                    rem_st = torch.where(st_rem_mask)[0]
                    target_st_rem = int(round(n_st_total * frac_left))
                    n_st_unmask = max(1, int(rem_st.numel()) - target_st_rem)
                    if step == n_steps - 1:
                        n_st_unmask = int(rem_st.numel())

                    if self.spec.is_bit:
                        st_logits = net_out["struct_logits"][0, 1 : 1 + length, :].reshape(length, -1, 2).to(dtype=torch.float32)
                        st_lp = F.log_softmax(st_logits, dim=-1)
                        if cur_temp <= 1e-6:
                            bits_choice = st_lp.argmax(dim=-1)
                        else:
                            probs_b = F.softmax(st_lp / cur_temp, dim=-1).reshape(-1, 2)
                            bits_choice = torch.multinomial(probs_b, 1, generator=gen).reshape(length, -1)
                        st_scores = st_lp.gather(-1, bits_choice.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
                        st_token_ids = lfq_bits_to_code_indices(bits_choice, self.model.codebook_embed_dim) + STRUCT_VOCAB_OFFSET
                    else:
                        st_logits = net_out["logits"][0, 1 : 1 + length, STRUCT_VOCAB_OFFSET : STRUCT_VOCAB_OFFSET + STRUCT_CODEBOOK_SIZE].to(dtype=torch.float32)
                        st_lp = F.log_softmax(st_logits, dim=-1)
                        if top_p < 1.0:
                            st_lp_samp = _top_p_filter(st_lp, top_p)
                        else:
                            st_lp_samp = st_lp
                        if cur_temp <= 1e-6:
                            st_choice = st_lp_samp.argmax(dim=-1)
                        else:
                            probs_s = F.softmax(st_lp_samp / cur_temp, dim=-1)
                            st_choice = torch.multinomial(probs_s, 1, generator=gen).squeeze(-1)
                        st_scores = st_lp.gather(-1, st_choice.unsqueeze(-1)).squeeze(-1)
                        st_token_ids = st_choice + STRUCT_VOCAB_OFFSET

                    if sampling_mode == "uniform_random":
                        perm = torch.randperm(rem_st.numel(), generator=gen, device=self.device)
                        chosen_st = rem_st[perm[:n_st_unmask]]
                    else:
                        sc = st_scores[rem_st]
                        if sampling_mode == "co_generation" and step < n_steps - 1:
                            u = torch.rand(sc.shape, generator=gen, device=self.device).clamp(1e-7, 1.0 - 1e-7)
                            sc = sc + (1.0 - float(step) / float(n_steps)) * (-torch.log(-torch.log(u)))
                        topk = torch.topk(sc, k=n_st_unmask, largest=True).indices
                        chosen_st = rem_st[topk]
                    paired[0, 1 + chosen_st] = st_token_ids[chosen_st]

        final_aa_ids = paired[0, half + 1 : half + 1 + length].cpu().tolist()
        final_st_ids = paired[0, 1 : 1 + length].cpu().tolist()
        seq_str = "".join(DPLM2_ID_TO_AA[int(tid)] for tid in final_aa_ids)
        return {
            "sequence": seq_str,
            "length": length,
            "struct_token_ids": final_st_ids,
            "steps": n_steps,
            "seed": int(seed),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "sampling_mode": sampling_mode,
        }

    def infill_window(
        self,
        parent: str,
        *,
        start: int,
        extent: int,
        seed: int = 20261125,
        steps: int | None = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        sampling_mode: str = "max_confidence",
        coords: np.ndarray | None = None,
        struct_tokens: Sequence[int] | np.ndarray | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """``R5a`` local span infilling over ``[start, start + extent)``."""
        length = len(parent)
        if not (0 <= start < length and extent >= 1 and start + extent <= length):
            raise ValueError(f"Invalid span [{start}, {start + extent}) for length {length}")
        aa_ids = _encode_aa_to_ids(parent)
        s_ids, single_mod = self._resolve_struct_ids(length, coords=coords, struct_tokens=struct_tokens, mode=mode)
        aa_edit = np.zeros(length, dtype=bool)
        aa_edit[start : start + extent] = True
        st_edit = np.zeros(length, dtype=bool)
        rec = self._iterative_unmask(
            aa_ids,
            s_ids,
            aa_editable_mask=aa_edit,
            struct_editable_mask=st_edit,
            single_modality=single_mod,
            steps=extent if steps is None else int(steps),
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            sampling_mode=sampling_mode,
        )
        rec["start"] = int(start)
        rec["extent"] = int(extent)
        rec["infilled_span"] = rec["sequence"][start : start + extent]
        return rec

    def continue_suffix(
        self,
        prefix: str,
        target_length: int,
        *,
        seed: int = 20261125,
        steps: int | None = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        sampling_mode: str = "max_confidence",
        coords: np.ndarray | None = None,
        struct_tokens: Sequence[int] | np.ndarray | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """``R4`` Downstream 2 prefix-fixed suffix diffusion continuation."""
        pref_len = len(prefix)
        if target_length <= pref_len:
            raise ValueError(f"target_length {target_length} must exceed len(prefix)={pref_len}")
        aa_ids = _encode_aa_to_ids(prefix) + [AA_MASK_ID] * (target_length - pref_len)
        s_ids, single_mod = self._resolve_struct_ids(
            target_length, coords=coords, struct_tokens=struct_tokens, mode=mode
        )
        aa_edit = np.zeros(target_length, dtype=bool)
        aa_edit[pref_len:] = True
        st_edit = np.zeros(target_length, dtype=bool)
        n_steps = min(target_length - pref_len, 100) if steps is None else int(steps)
        rec = self._iterative_unmask(
            aa_ids,
            s_ids,
            aa_editable_mask=aa_edit,
            struct_editable_mask=st_edit,
            single_modality=single_mod,
            steps=n_steps,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            sampling_mode=sampling_mode,
        )
        rec["prefix_len"] = pref_len
        return rec

    def sample_diffusion_full(
        self,
        length: int,
        *,
        seed: int = 20261125,
        steps: int | None = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        sampling_mode: str = "max_confidence",
        coords: np.ndarray | None = None,
        struct_tokens: Sequence[int] | np.ndarray | None = None,
        profile_log_bias: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Full-length diffusion generation across 4 modes:

        * ``sampling_mode="co_generation"``: joint sequence + structure token annealing
          (``2.0 -> 0.1`` for standard DPLM-2, ``1.1 -> 0.1`` for DPLM-2-Bit).
        * ``sampling_mode="max_confidence"`` or ``"uniform_random"`` with ``coords=None``
          and ``struct_tokens=None``: ``seq_only`` single-modality diffusion.
        * ``coords`` or ``struct_tokens`` supplied: backbone-conditioned inverse folding
          (optionally combined with ``profile_log_bias`` of shape ``(L, 20)`` for ``Exp 4.2``).
        """
        n_steps = min(int(length), 100) if steps is None else int(steps)
        aa_ids = [AA_MASK_ID] * int(length)
        aa_edit = np.ones(int(length), dtype=bool)

        if sampling_mode == "co_generation":
            s_ids = [STRUCT_MASK_ID] * int(length)
            st_edit = np.ones(int(length), dtype=bool)
            single_mod = False
            anneal = (1.1, 0.1) if self.spec.is_bit else (2.0, 0.1)
        elif coords is not None or struct_tokens is not None:
            s_ids, _ = self._resolve_struct_ids(
                int(length), coords=coords, struct_tokens=struct_tokens, mode="struct_conditioned"
            )
            st_edit = np.zeros(int(length), dtype=bool)
            single_mod = False
            anneal = None
        else:
            s_ids = [STRUCT_MASK_ID] * int(length)
            st_edit = np.zeros(int(length), dtype=bool)
            single_mod = True
            anneal = None

        return self._iterative_unmask(
            aa_ids,
            s_ids,
            aa_editable_mask=aa_edit,
            struct_editable_mask=st_edit,
            single_modality=single_mod,
            steps=n_steps,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            sampling_mode=sampling_mode,
            anneal_range=anneal,
            profile_log_bias=profile_log_bias,
        )


def load_dplm2_adapter(
    arm_id: str,
    *,
    mode: str = "seq_only",
    model_root: Path | str | None = None,
    device: str = "cpu",
    dtype: str | None = None,
    struct_tokenizer: StructTokenizerEncoder | None = None,
    load_struct_tokenizer: bool | None = None,
) -> DPLM2Adapter:
    """Convenience factory returning a loaded ``DPLM2Adapter``."""
    return DPLM2Adapter(
        arm_id,
        mode=mode,
        model_root=model_root,
        device=device,
        dtype=dtype,
        struct_tokenizer=struct_tokenizer,
        load_struct_tokenizer=load_struct_tokenizer,
    )
