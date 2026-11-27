"""Self-contained PyTorch adapter for all 11 official ProteinMPNN checkpoints.

Wraps the MIT-licensed ProteinMPNN message-passing neural network architecture
(Dauparas et al., Science 2022; ``v_48_{002,010,020,030}.pt`` vanilla,
``soluble_model_weights/v_48_{002,010,020,030}.pt``, and
``ca_model_weights/v_48_{002,010,020}.pt``) with zero external import-path
dependencies, strict checkpoint loading, and a unified interface for:

1. Exact site-conditional AA20 log-probabilities & leave-one-out PLL
   (placing each target position LAST in ``decoding_order`` so it conditions
   on ``(X_WT, s_{-i})``; batched across unique mutated sites while caching
   the structure encoder output ONCE per backbone).
2. Permutation-averaged exact autoregressive sequence log-likelihood (4
   deterministic canonical orders: N->C, C->N, + 2 seeded permutations) for
   multi-mutant scoring & R3 double-mutant thermodynamic cycles (exact
   non-zero symmetric 3D GNN epistasis).
3. Frozen hidden-state extraction hooking ``decoder_layers[1]`` (``R_mid``)
   and ``decoder_layers[2]`` (``R_last``) under deterministic N->C teacher
   forcing, pooled ``[mean, last]`` -> ``(N, 4, 128)`` and optionally
   projected via ``project_four_blocks`` -> ``(N, 1024)`` (``0:512`` = ``R_mid``,
   ``512:1024`` = ``R_last``).
4. E01 perturbation-propagation contact sensitivity matrix ``J_{ij}``.
5. Multi-chain / spatial-offset context scoring (``R4`` 800-unit & ``R5`` DMS).
6. Temperature-controlled autoregressive sequence design, local span infilling
   (``R5a``), prefix-fixed suffix continuation (``R4`` Downstream 2), and
   anchored inverse folding (``Exp 4.1`` & ``Exp 4.2``).
"""

from __future__ import annotations

import hashlib
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.capability.core.amino_acids import AA20
from src.capability.core.precision_policy import fp32_matmul_context, requested_fp32_matmul_policy
from src.capability.general_plm.adapters import PositionVector, apply_mutation, parse_mutation
from src.capability.general_plm.eval_utils import project_four_blocks

MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
assert MPNN_ALPHABET[:20] == AA20, "MPNN first 20 tokens must match canonical AA20"
MPNN_AA_TO_IDX = {aa: idx for idx, aa in enumerate(MPNN_ALPHABET)}
MPNN_X_IDX = 20

DEFAULT_PROTEINMPNN_ROOT = Path("/home/lzp/models/general_plm_20261120/proteinmpnn_all")


@dataclass(frozen=True)
class ProteinMPNNSpec:
    """Authoritative checkpoint specification for one ProteinMPNN arm."""

    arm_id: str
    variant: str  # "vanilla" | "soluble" | "ca_only"
    noise_angstrom: float
    rel_weight_path: str
    ca_only: bool


PROTEINMPNN_SPECS: Mapping[str, ProteinMPNNSpec] = {
    "proteinmpnn-v-002": ProteinMPNNSpec("proteinmpnn-v-002", "vanilla", 0.02, "weight/vanilla_model_weights/v_48_002.pt", False),
    "proteinmpnn-v-010": ProteinMPNNSpec("proteinmpnn-v-010", "vanilla", 0.10, "weight/vanilla_model_weights/v_48_010.pt", False),
    "proteinmpnn-v-020": ProteinMPNNSpec("proteinmpnn-v-020", "vanilla", 0.20, "weight/vanilla_model_weights/v_48_020.pt", False),
    "proteinmpnn-v-030": ProteinMPNNSpec("proteinmpnn-v-030", "vanilla", 0.30, "weight/vanilla_model_weights/v_48_030.pt", False),
    "proteinmpnn-s-002": ProteinMPNNSpec("proteinmpnn-s-002", "soluble", 0.02, "weight/soluble_model_weights/v_48_002.pt", False),
    "proteinmpnn-s-010": ProteinMPNNSpec("proteinmpnn-s-010", "soluble", 0.10, "weight/soluble_model_weights/v_48_010.pt", False),
    "proteinmpnn-s-020": ProteinMPNNSpec("proteinmpnn-s-020", "soluble", 0.20, "weight/soluble_model_weights/v_48_020.pt", False),
    "proteinmpnn-s-030": ProteinMPNNSpec("proteinmpnn-s-030", "soluble", 0.30, "weight/soluble_model_weights/v_48_030.pt", False),
    "proteinmpnn-ca-002": ProteinMPNNSpec("proteinmpnn-ca-002", "ca_only", 0.02, "weight/ca_model_weights/v_48_002.pt", True),
    "proteinmpnn-ca-010": ProteinMPNNSpec("proteinmpnn-ca-010", "ca_only", 0.10, "weight/ca_model_weights/v_48_010.pt", True),
    "proteinmpnn-ca-020": ProteinMPNNSpec("proteinmpnn-ca-020", "ca_only", 0.20, "weight/ca_model_weights/v_48_020.pt", True),
}


def resolve_proteinmpnn_root(model_root: Path | str | None = None) -> Path:
    """Resolve the staged ``proteinmpnn_all`` directory on Compute or H200 GPFS."""
    candidates: list[Path] = []
    if model_root is not None:
        p = Path(model_root)
        candidates.extend([p, p / "proteinmpnn_all"])
    for env_key in ("PROTEINMPNN_ROOT", "GENERAL_PLM_MODEL_ROOT", "MODEL_ROOT", "MODEL_BASE_DIR"):
        raw = os.environ.get(env_key, "").strip()
        if raw:
            ep = Path(raw)
            candidates.extend([ep, ep / "proteinmpnn_all"])
    candidates.append(DEFAULT_PROTEINMPNN_ROOT)
    for cand in candidates:
        if (cand / "weight/vanilla_model_weights/v_48_020.pt").is_file():
            return cand
    raise FileNotFoundError(
        "Could not locate proteinmpnn_all checkpoint directory; checked: "
        + ", ".join(str(c) for c in candidates)
    )


# --------------------------------------------------------------------------- #
# Pure-PyTorch ProteinMPNN Architecture (MIT License — Dauparas et al. 2022)
# --------------------------------------------------------------------------- #


def _gather_edges(edges: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)


def _gather_nodes(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    return neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])


def _cat_neighbors_nodes(
    h_nodes: torch.Tensor,
    h_neighbors: torch.Tensor,
    E_idx: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([h_neighbors, _gather_nodes(h_nodes, E_idx)], -1)


def _order_mask_backward(decoding_order: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Exact O(B * L^2) equivalent of official ProteinMPNN einsum backward mask.

    ``mask[b, q, p] = 1.0`` iff residue ``p`` is decoded strictly before ``q``
    under ``decoding_order[b]``.
    """
    bsz, length = decoding_order.shape
    ranks = torch.empty_like(decoding_order)
    arange = torch.arange(length, device=decoding_order.device, dtype=decoding_order.dtype)
    ranks.scatter_(1, decoding_order, arange.unsqueeze(0).expand(bsz, -1))
    return (ranks.unsqueeze(2) > ranks.unsqueeze(1)).to(dtype=dtype)


class _PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden: int, num_ff: int) -> None:
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = nn.GELU()

    def forward(self, h_V: torch.Tensor) -> torch.Tensor:
        return self.W_out(self.act(self.W_in(h_V)))


class _PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings: int, max_relative_feature: int = 32) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 2, num_embeddings)

    def forward(self, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        d = (
            torch.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature) * mask
            + (1 - mask) * (2 * self.max_relative_feature + 1)
        )
        d_onehot = F.one_hot(d, 2 * self.max_relative_feature + 2)
        return self.linear(d_onehot.to(self.linear.weight.dtype))


class _EncLayer(nn.Module):
    def __init__(self, num_hidden: int, num_in: int, dropout: float = 0.1, scale: float = 30.0) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()
        self.dense = _PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(
        self,
        h_V: torch.Tensor,
        h_E: torch.Tensor,
        E_idx: torch.Tensor,
        mask_V: torch.Tensor | None = None,
        mask_attend: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_EV = _cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1) * h_V

        h_EV = _cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class _DecLayer(nn.Module):
    def __init__(self, num_hidden: int, num_in: int, dropout: float = 0.1, scale: float = 30.0) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()
        self.dense = _PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(
        self,
        h_V: torch.Tensor,
        h_E: torch.Tensor,
        mask_V: torch.Tensor | None = None,
        mask_attend: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1) * h_V
        return h_V


class _CAProteinFeatures(nn.Module):
    def __init__(
        self,
        edge_features: int,
        node_features: int,
        num_positional_embeddings: int = 16,
        num_rbf: int = 16,
        top_k: int = 30,
        augment_eps: float = 0.0,
    ) -> None:
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.embeddings = _PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 3, num_positional_embeddings + num_rbf * 9 + 7
        self.node_embedding = nn.Linear(node_in, node_features, bias=False)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_nodes = nn.LayerNorm(node_features)
        self.norm_edges = nn.LayerNorm(edge_features)

    @staticmethod
    def _quaternions(R: torch.Tensor) -> torch.Tensor:
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(
            torch.abs(
                1.0
                + torch.stack(
                    [Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz],
                    dim=-1,
                )
            )
        )
        signs = torch.sign(
            torch.stack(
                [
                    R[:, :, :, 2, 1] - R[:, :, :, 1, 2],
                    R[:, :, :, 0, 2] - R[:, :, :, 2, 0],
                    R[:, :, :, 1, 0] - R[:, :, :, 0, 1],
                ],
                dim=-1,
            )
        )
        xyz = signs * magnitudes
        w = torch.sqrt(F.relu(1.0 + diag.sum(-1, keepdim=True))) / 2.0
        return F.normalize(torch.cat((xyz, w), dim=-1), dim=-1)

    def _orientations_coarse(self, X: torch.Tensor, E_idx: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        dX = X[:, 1:, :] - X[:, :-1, :]
        dX_norm = torch.norm(dX, dim=-1)
        dX_mask = (3.6 < dX_norm) & (dX_norm < 4.0)
        dX = dX * dX_mask[:, :, None]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

        cosA = torch.clamp(-(u_1 * u_0).sum(-1), -1.0 + eps, 1.0 - eps)
        A = torch.acos(cosA)
        cosD = torch.clamp((n_2 * n_1).sum(-1), -1.0 + eps, 1.0 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        AD_features = torch.stack((torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), 2)
        AD_features = F.pad(AD_features, (0, 0, 1, 2), "constant", 0)

        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O_mat = torch.stack((o_1, n_2, torch.cross(o_1, n_2, dim=-1)), 2)
        O_mat = O_mat.view(list(O_mat.shape[:2]) + [9])
        O_mat = F.pad(O_mat, (0, 0, 1, 2), "constant", 0)
        O_neighbors = _gather_nodes(O_mat, E_idx)
        X_neighbors = _gather_nodes(X, E_idx)

        O_mat = O_mat.view(list(O_mat.shape[:2]) + [3, 3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])
        dX_nb = X_neighbors - X.unsqueeze(-2)
        dU = F.normalize(torch.matmul(O_mat.unsqueeze(2), dX_nb.unsqueeze(-1)).squeeze(-1), dim=-1)
        R = torch.matmul(O_mat.unsqueeze(2).transpose(-1, -2), O_neighbors)
        Q = self._quaternions(R)
        return AD_features, torch.cat((dU, Q), dim=-1)

    def _dist(self, X: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
        dX = X.unsqueeze(1) - X.unsqueeze(2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        k = min(self.top_k, X.shape[1])
        D_neighbors, E_idx = torch.topk(D_adjust, k, dim=-1, largest=False)
        mask_neighbors = _gather_edges(mask_2D.unsqueeze(-1), E_idx)
        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D: torch.Tensor) -> torch.Tensor:
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device, dtype=D.dtype).view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        return torch.exp(-(((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2))

    def _get_rbf(self, A: torch.Tensor, B: torch.Tensor, E_idx: torch.Tensor) -> torch.Tensor:
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)
        D_A_B_neighbors = _gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]
        return self._rbf(D_A_B_neighbors)

    def forward(
        self,
        Ca: torch.Tensor,
        mask: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        D_neighbors, E_idx, _ = self._dist(Ca, mask)
        Ca_0 = torch.zeros_like(Ca)
        Ca_2 = torch.zeros_like(Ca)
        Ca_0[:, 1:, :] = Ca[:, :-1, :]
        Ca_1 = Ca
        Ca_2[:, :-1, :] = Ca[:, 1:, :]
        _, O_features = self._orientations_coarse(Ca, E_idx)

        RBF_all = torch.cat(
            (
                self._rbf(D_neighbors),
                self._get_rbf(Ca_0, Ca_0, E_idx),
                self._get_rbf(Ca_2, Ca_2, E_idx),
                self._get_rbf(Ca_0, Ca_1, E_idx),
                self._get_rbf(Ca_0, Ca_2, E_idx),
                self._get_rbf(Ca_1, Ca_0, E_idx),
                self._get_rbf(Ca_1, Ca_2, E_idx),
                self._get_rbf(Ca_2, Ca_0, E_idx),
                self._get_rbf(Ca_2, Ca_1, E_idx),
            ),
            dim=-1,
        )
        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = _gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]
        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = _gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all, O_features), -1)
        return self.norm_edges(self.edge_embedding(E)), E_idx


class _ProteinFeatures(nn.Module):
    def __init__(
        self,
        edge_features: int,
        node_features: int,
        num_positional_embeddings: int = 16,
        num_rbf: int = 16,
        top_k: int = 30,
        augment_eps: float = 0.0,
    ) -> None:
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.embeddings = _PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
        dX = X.unsqueeze(1) - X.unsqueeze(2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        k = min(self.top_k, X.shape[1])
        return torch.topk(D_adjust, k, dim=-1, largest=False)

    def _rbf(self, D: torch.Tensor) -> torch.Tensor:
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device, dtype=D.dtype).view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        return torch.exp(-(((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2))

    def _get_rbf(self, A: torch.Tensor, B: torch.Tensor, E_idx: torch.Tensor) -> torch.Tensor:
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)
        D_A_B_neighbors = _gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]
        return self._rbf(D_A_B_neighbors)

    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        N, Ca, C, O_atom = X[:, :, 0, :], X[:, :, 1, :], X[:, :, 2, :], X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)
        RBF_all = torch.cat(
            (
                self._rbf(D_neighbors),
                self._get_rbf(N, N, E_idx),
                self._get_rbf(C, C, E_idx),
                self._get_rbf(O_atom, O_atom, E_idx),
                self._get_rbf(Cb, Cb, E_idx),
                self._get_rbf(Ca, N, E_idx),
                self._get_rbf(Ca, C, E_idx),
                self._get_rbf(Ca, O_atom, E_idx),
                self._get_rbf(Ca, Cb, E_idx),
                self._get_rbf(N, C, E_idx),
                self._get_rbf(N, O_atom, E_idx),
                self._get_rbf(N, Cb, E_idx),
                self._get_rbf(Cb, C, E_idx),
                self._get_rbf(Cb, O_atom, E_idx),
                self._get_rbf(O_atom, C, E_idx),
                self._get_rbf(N, Ca, E_idx),
                self._get_rbf(C, Ca, E_idx),
                self._get_rbf(O_atom, Ca, E_idx),
                self._get_rbf(Cb, Ca, E_idx),
                self._get_rbf(C, N, E_idx),
                self._get_rbf(O_atom, N, E_idx),
                self._get_rbf(Cb, N, E_idx),
                self._get_rbf(C, Cb, E_idx),
                self._get_rbf(O_atom, Cb, E_idx),
                self._get_rbf(C, O_atom, E_idx),
            ),
            dim=-1,
        )
        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = _gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]
        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = _gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        return self.norm_edges(self.edge_embedding(E)), E_idx


class ProteinMPNNModel(nn.Module):
    """Exact state-dict-compatible ProteinMPNN module with cached-encoder helpers."""

    def __init__(
        self,
        num_letters: int = 21,
        node_features: int = 128,
        edge_features: int = 128,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        vocab: int = 21,
        k_neighbors: int = 48,
        augment_eps: float = 0.0,
        dropout: float = 0.1,
        ca_only: bool = False,
    ) -> None:
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.ca_only = ca_only

        if ca_only:
            self.features: nn.Module = _CAProteinFeatures(
                node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps
            )
            self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        else:
            self.features = _ProteinFeatures(
                node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps
            )
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        self.encoder_layers = nn.ModuleList(
            [_EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(num_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [_DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout) for _ in range(num_decoder_layers)]
        )
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

    def encode_structure(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run structure featurization + 3 encoder layers once per backbone."""
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device, dtype=E.dtype)
        h_E = self.W_e(E)
        mask_attend = _gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        h_EX_encoder = _cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
        h_EXV_encoder = _cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        return h_V, h_E, E_idx, h_EXV_encoder

    def decode_from_encoded(
        self,
        h_V_enc: torch.Tensor,
        h_E: torch.Tensor,
        E_idx: torch.Tensor,
        h_EXV_encoder: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        decoding_order: torch.Tensor,
        *,
        return_layer_hiddens: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run 3 autoregressive decoder layers for a batch sharing an encoded backbone."""
        h_S = self.W_s(S)
        h_ES = _cat_neighbors_nodes(h_S, h_E, E_idx)
        order_mask_bw = _order_mask_backward(decoding_order, dtype=h_V_enc.dtype)
        mask_attend = torch.gather(order_mask_bw, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view(mask.size(0), mask.size(1), 1, 1)
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        h_V = h_V_enc
        layer_hiddens: list[torch.Tensor] = []
        for layer in self.decoder_layers:
            h_ESV = _cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)
            if return_layer_hiddens:
                layer_hiddens.append(h_V)
        logits = self.W_out(h_V)
        return logits, layer_hiddens


# --------------------------------------------------------------------------- #
# Coordinate / Sequence Helpers
# --------------------------------------------------------------------------- #


def _prepare_coords_array(coords: np.ndarray, *, ca_only: bool) -> tuple[np.ndarray, np.ndarray]:
    """Normalize input backbone coordinates to ``(L, 3)`` (CA-only) or ``(L, 4, 3)`` (N,CA,C,O).

    Accepts ``(L, 4, 3)``, ``(L, 37, 3)`` (OpenFold atom37 where N=0,CA=1,C=2,O=4),
    or ``(L, 1, 3)`` / ``(L, 3)`` for ``ca_only=True``.
    """
    arr = np.asarray(coords, dtype=np.float32)
    if ca_only:
        if arr.ndim == 2 and arr.shape[1] == 3:
            ca = arr
        elif arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 3:
            ca = arr[:, 0, :]
        elif arr.ndim == 3 and arr.shape[1] >= 3 and arr.shape[2] == 3:
            ca = arr[:, 1, :]  # CA is index 1 in both (N,CA,C,O) and atom37
        else:
            raise ValueError(f"Unsupported CA coords shape {arr.shape}")
        valid = np.isfinite(ca).all(axis=-1).astype(np.float32)
        clean = np.nan_to_num(ca, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return clean, valid

    if arr.ndim == 3 and arr.shape[1] == 4 and arr.shape[2] == 3:
        bb = arr
    elif arr.ndim == 3 and arr.shape[1] == 37 and arr.shape[2] == 3:
        bb = arr[:, [0, 1, 2, 4], :]
    else:
        raise ValueError(f"Expected backbone coords (L, 4, 3) or atom37 (L, 37, 3), got {arr.shape}")
    valid = np.isfinite(bb).all(axis=(-1, -2)).astype(np.float32)
    clean = np.nan_to_num(bb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return clean, valid


def _encode_mpnn_seq(sequence: str) -> np.ndarray:
    out = np.empty(len(sequence), dtype=np.int64)
    for i, ch in enumerate(sequence):
        idx = MPNN_AA_TO_IDX.get(ch)
        if idx is None:
            raise ValueError(f"Unsupported residue {ch!r} at pos {i} for ProteinMPNN")
        out[i] = idx
    return out


def _canonical_permutations(length: int, n_permutations: int = 4, seed: int = 20261125) -> np.ndarray:
    """Return ``(n_permutations, L)`` deterministic decoding orders (N->C, C->N, + seeded)."""
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    orders = [np.arange(length, dtype=np.int64)]
    if n_permutations >= 2:
        orders.append(np.arange(length - 1, -1, -1, dtype=np.int64))
    rng = np.random.default_rng(seed)
    while len(orders) < n_permutations:
        orders.append(rng.permutation(length).astype(np.int64))
    return np.stack(orders, axis=0)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# High-Level ProteinMPNN Adapter
# --------------------------------------------------------------------------- #


class ProteinMPNNAdapter:
    """Unified adapter over any of the 11 official ProteinMPNN checkpoints."""

    def __init__(
        self,
        arm_id: str,
        *,
        model_root: Path | str | None = None,
        device: str = "cpu",
        dtype: str = "float32",
    ) -> None:
        if arm_id not in PROTEINMPNN_SPECS:
            raise KeyError(f"Unknown ProteinMPNN arm_id {arm_id!r}; expected one of {sorted(PROTEINMPNN_SPECS)}")
        self.spec = PROTEINMPNN_SPECS[arm_id]
        self.arm = arm_id
        self.device = str(device)
        dt_norm = dtype.strip().lower()
        if dt_norm in ("fp32", "float32", "auto", "default"):
            self.dtype = "float32"
            self.torch_dtype = torch.float32
        elif dt_norm in ("bf16", "bfloat16"):
            self.dtype = "bfloat16"
            self.torch_dtype = torch.bfloat16
        else:
            raise ValueError(f"{arm_id}: unsupported dtype {dtype!r}; use float32 or bfloat16")

        self.root_dir = resolve_proteinmpnn_root(model_root)
        self.checkpoint_path = self.root_dir / self.spec.rel_weight_path
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"{arm_id}: missing checkpoint {self.checkpoint_path}")

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.k_neighbors = int(ckpt.get("num_edges", 48))
        self.ckpt_noise_level = float(ckpt.get("noise_level", self.spec.noise_angstrom))
        self.model = ProteinMPNNModel(
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            vocab=21,
            k_neighbors=self.k_neighbors,
            augment_eps=0.0,
            dropout=0.1,
            ca_only=self.spec.ca_only,
        )
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.to(device=self.device, dtype=self.torch_dtype)
        self.model.eval()
        self.model.requires_grad_(False)

    def _precision_ctx(self):
        if self.dtype == "float32" and self.device.startswith("cuda"):
            return fp32_matmul_context(torch)
        return nullcontext()

    def provenance(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "family": f"ProteinMPNN-{self.spec.variant}",
            "paradigm": "inverse_folding_gnn",
            "variant": self.spec.variant,
            "ca_only": self.spec.ca_only,
            "noise_angstrom": self.ckpt_noise_level,
            "k_neighbors": self.k_neighbors,
            "dtype_requested": self.dtype,
            "dtype_realised": "/".join(sorted({str(p.dtype) for p in self.model.parameters()})),
            "fp32_matmul_policy": (
                requested_fp32_matmul_policy()
                if self.dtype == "float32"
                else {"declared_non_fp32_opt_in": self.dtype}
            ),
            "parameters": int(sum(p.numel() for p in self.model.parameters())),
            "device": self.device,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": _sha256_file(self.checkpoint_path),
        }

    def _encode_single_backbone(
        self,
        coords: np.ndarray,
        *,
        residue_idx: np.ndarray | None = None,
        chain_labels: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        clean, valid = _prepare_coords_array(coords, ca_only=self.spec.ca_only)
        length = clean.shape[0]
        X_t = torch.from_numpy(clean).unsqueeze(0).to(device=self.device, dtype=self.torch_dtype)
        mask_t = torch.from_numpy(valid).unsqueeze(0).to(device=self.device, dtype=self.torch_dtype)
        res_t = (
            torch.arange(length, device=self.device, dtype=torch.long).unsqueeze(0)
            if residue_idx is None
            else torch.as_tensor(residue_idx, device=self.device, dtype=torch.long).unsqueeze(0)
        )
        chain_t = (
            torch.ones((1, length), device=self.device, dtype=torch.long)
            if chain_labels is None
            else torch.as_tensor(chain_labels, device=self.device, dtype=torch.long).unsqueeze(0)
        )
        with torch.no_grad(), self._precision_ctx():
            h_V, h_E, E_idx, h_EXV = self.model.encode_structure(X_t, mask_t, res_t, chain_t)
        return h_V, h_E, E_idx, h_EXV, mask_t

    @torch.no_grad()
    def site_conditional_logprobs(
        self,
        sequence: str,
        coords: np.ndarray,
        *,
        positions: Sequence[int] | None = None,
        batch_size: int = 32,
        encoded_backbone: tuple[torch.Tensor, ...] | None = None,
    ) -> np.ndarray:
        """Return AA20 log-probs ``(len(positions), 20)`` placing each site LAST in decoding order.

        For each requested 0-based position ``i``, every residue ``j != i`` is decoded
        before ``i`` (deterministic canonical N->C order over ``j != i``), so site ``i``
        conditions on ``(X_WT, s_{-i})``.
        """
        length = len(sequence)
        pos_list = list(range(length)) if positions is None else [int(p) for p in positions]
        if any(not (0 <= p < length) for p in pos_list):
            raise ValueError(f"Position out of range for sequence length {length}")
        if not pos_list:
            return np.zeros((0, 20), dtype=np.float32)

        h_V_enc, h_E, E_idx, h_EXV, mask_t = (
            self._encode_single_backbone(coords) if encoded_backbone is None else encoded_backbone
        )
        if mask_t.shape[1] != length:
            raise ValueError(f"Sequence length {length} != coords length {mask_t.shape[1]}")

        S_base = torch.from_numpy(_encode_mpnn_seq(sequence)).to(device=self.device, dtype=torch.long).unsqueeze(0)
        out = np.empty((len(pos_list), 20), dtype=np.float32)
        base_order = np.arange(length, dtype=np.int64)
        bs = max(1, int(batch_size))

        with self._precision_ctx():
            for start in range(0, len(pos_list), bs):
                chunk = pos_list[start : start + bs]
                cur = len(chunk)
                orders = np.empty((cur, length), dtype=np.int64)
                for b_idx, p in enumerate(chunk):
                    if p == length - 1:
                        orders[b_idx] = base_order
                    else:
                        orders[b_idx, :p] = base_order[:p]
                        orders[b_idx, p : length - 1] = base_order[p + 1 :]
                        orders[b_idx, length - 1] = p
                order_t = torch.from_numpy(orders).to(device=self.device, dtype=torch.long)
                logits, _ = self.model.decode_from_encoded(
                    h_V_enc.expand(cur, -1, -1),
                    h_E.expand(cur, -1, -1, -1),
                    E_idx.expand(cur, -1, -1),
                    h_EXV.expand(cur, -1, -1, -1),
                    S_base.expand(cur, -1),
                    mask_t.expand(cur, -1),
                    order_t,
                )
                row_ar = torch.arange(cur, device=self.device, dtype=torch.long)
                pos_t = torch.as_tensor(chunk, device=self.device, dtype=torch.long)
                site_logits = logits[row_ar, pos_t, :20].to(dtype=torch.float32)
                site_lp = F.log_softmax(site_logits, dim=-1)
                out[start : start + cur] = site_lp.cpu().numpy()
        return out

    def native_site_conditional(self, sequence: str, index: int, *, coords: np.ndarray) -> np.ndarray:
        """Return length-20 AA20 log-probabilities at 0-based ``index``."""
        return self.site_conditional_logprobs(sequence, coords, positions=[index])[0]

    def exact_pll(self, sequence: str, coords: np.ndarray, *, batch_size: int = 32) -> PositionVector:
        """Exact leave-one-out site-conditional pseudo-log-likelihood over all ``L`` residues."""
        lp = self.site_conditional_logprobs(sequence, coords, positions=None, batch_size=batch_size)
        seq_idx = _encode_mpnn_seq(sequence)
        terms = (-lp[np.arange(len(sequence)), seq_idx]).astype(np.float32)
        terms = np.maximum(terms, 0.0)
        nll_sum = float(terms.sum(dtype=np.float64))
        return PositionVector(terms=terms, nll_sum=nll_sum, pll=-nll_sum, paradigm="inverse_folding_gnn")

    @torch.no_grad()
    def exact_sequence_ll_batch(
        self,
        sequences: Sequence[str],
        coords: np.ndarray,
        *,
        n_permutations: int = 4,
        batch_size: int = 32,
        seed: int = 20261125,
        encoded_backbone: tuple[torch.Tensor, ...] | None = None,
    ) -> np.ndarray:
        """Exact autoregressive ``log p(s | X_WT)`` averaged over ``n_permutations`` canonical orders.

        Returns summed sequence log-likelihoods ``(len(sequences),)`` float64 over valid residues.
        """
        if not sequences:
            return np.zeros((0,), dtype=np.float64)
        length = len(sequences[0])
        if any(len(s) != length for s in sequences):
            raise ValueError("All sequences in exact_sequence_ll_batch must match backbone length")

        h_V_enc, h_E, E_idx, h_EXV, mask_t = (
            self._encode_single_backbone(coords) if encoded_backbone is None else encoded_backbone
        )
        if mask_t.shape[1] != length:
            raise ValueError(f"Sequence length {length} != coords length {mask_t.shape[1]}")

        perms = _canonical_permutations(length, n_permutations=n_permutations, seed=seed)
        perms_t = torch.from_numpy(perms).to(device=self.device, dtype=torch.long)
        S_all = torch.as_tensor(
            np.stack([_encode_mpnn_seq(s) for s in sequences], axis=0),
            device=self.device,
            dtype=torch.long,
        )
        mask_f32 = mask_t.to(dtype=torch.float32)
        n_seq = len(sequences)
        ll_acc = np.zeros(n_seq, dtype=np.float64)
        bs = max(1, int(batch_size))

        with self._precision_ctx():
            for p_idx in range(n_permutations):
                order_1 = perms_t[p_idx : p_idx + 1]
                for start in range(0, n_seq, bs):
                    stop = min(start + bs, n_seq)
                    cur = stop - start
                    S_chunk = S_all[start:stop]
                    logits, _ = self.model.decode_from_encoded(
                        h_V_enc.expand(cur, -1, -1),
                        h_E.expand(cur, -1, -1, -1),
                        E_idx.expand(cur, -1, -1),
                        h_EXV.expand(cur, -1, -1, -1),
                        S_chunk,
                        mask_t.expand(cur, -1),
                        order_1.expand(cur, -1),
                    )
                    lp20 = F.log_softmax(logits[:, :, :20].to(dtype=torch.float32), dim=-1)
                    gathered = torch.gather(lp20, 2, S_chunk.clamp(max=19).unsqueeze(-1)).squeeze(-1)
                    valid_aa = (S_chunk < 20).to(dtype=torch.float32) * mask_f32
                    ll_sum = (gathered * valid_aa).sum(dim=1).cpu().numpy().astype(np.float64)
                    ll_acc[start:stop] += ll_sum
        return ll_acc / float(n_permutations)

    def exact_sequence_ll(
        self,
        sequence: str,
        coords: np.ndarray,
        *,
        n_permutations: int = 4,
        seed: int = 20261125,
    ) -> float:
        return float(
            self.exact_sequence_ll_batch([sequence], coords, n_permutations=n_permutations, seed=seed)[0]
        )

    def score_variants(
        self,
        wildtype: str,
        mutations: Sequence[str],
        coords: np.ndarray,
        *,
        batch_size: int = 32,
        multi_permutations: int = 4,
        force_exact_autoregressive: bool = False,
    ) -> np.ndarray:
        """Score single and multi-mutant substitutions on ``(X_WT, wildtype)``.

        * Single mutants (unless ``force_exact_autoregressive=True``): exact site
          conditional log-odds ``log p(mut | X_WT, s_{-i}) - log p(wt | X_WT, s_{-i})``
          evaluated in a single batched pass over unique mutated sites.
        * Multi-mutants (e.g., R3 double-mutant cycles): exact permutation-averaged
          autoregressive ``log p(s_mut | X_WT) - log p(s_wt | X_WT)`` across
          ``multi_permutations`` canonical orders (N->C, C->N, + 2 seeded),
          yielding exact symmetric non-zero 3D GNN epistasis.
        """
        if not mutations:
            return np.zeros((0,), dtype=np.float64)
        encoded = self._encode_single_backbone(coords)
        parsed = [parse_mutation(m, wildtype) for m in mutations]
        scores = np.empty(len(mutations), dtype=np.float64)

        single_indices: list[int] = []
        multi_indices: list[int] = []
        for idx, triples in enumerate(parsed):
            if len(triples) == 1 and not force_exact_autoregressive:
                single_indices.append(idx)
            else:
                multi_indices.append(idx)

        if single_indices:
            uniq_pos = sorted({parsed[idx][0][0] for idx in single_indices})
            pos_to_row = {p: r for r, p in enumerate(uniq_pos)}
            lp_mat = self.site_conditional_logprobs(
                wildtype, coords, positions=uniq_pos, batch_size=batch_size, encoded_backbone=encoded
            )
            for idx in single_indices:
                pos, wt_aa, mut_aa = parsed[idx][0]
                row = pos_to_row[pos]
                scores[idx] = float(lp_mat[row, MPNN_AA_TO_IDX[mut_aa]] - lp_mat[row, MPNN_AA_TO_IDX[wt_aa]])

        if multi_indices:
            var_seqs = [apply_mutation(wildtype, mutations[idx]) for idx in multi_indices]
            lls = self.exact_sequence_ll_batch(
                [wildtype, *var_seqs],
                coords,
                n_permutations=multi_permutations,
                batch_size=batch_size,
                encoded_backbone=encoded,
            )
            wt_ll = float(lls[0])
            for offset, idx in enumerate(multi_indices):
                scores[idx] = float(lls[offset + 1] - wt_ll)

        return scores

    @torch.no_grad()
    def extract_hidden_states(
        self,
        sequences: Sequence[str],
        coords: np.ndarray | Sequence[np.ndarray],
        *,
        batch_size: int = 64,
        return_projected_1024: bool = False,
    ) -> np.ndarray:
        """Extract frozen decoder hidden states ``R_mid`` (layer 1) & ``R_last`` (layer 2).

        Uses deterministic N->C teacher forcing on ``(X_WT, s_variant)`` and pools
        ``[mid_mean, mid_last, final_mean, final_last]`` -> ``(N, 4, 128)`` float32
        (or ``(N, 1024)`` via ``project_four_blocks`` when ``return_projected_1024=True``,
        where ``0:512`` is ``R_mid`` and ``512:1024`` is ``R_last``).
        """
        if not sequences:
            return np.zeros((0, 1024 if return_projected_1024 else 4, 128), dtype=np.float32) if not return_projected_1024 else np.zeros((0, 1024), dtype=np.float32)

        shared_backbone = isinstance(coords, np.ndarray) and (
            (coords.ndim == 3 and coords.shape[-1] == 3)
            or (self.spec.ca_only and coords.ndim == 2 and coords.shape[-1] == 3)
        )
        n_seq = len(sequences)
        blocks = np.empty((n_seq, 4, self.model.hidden_dim), dtype=np.float32)

        def _pool_layers(h_mid: torch.Tensor, h_fin: torch.Tensor, mask_1d: torch.Tensor) -> np.ndarray:
            # h_mid, h_fin: (B, L, H), mask_1d: (B, L)
            m = mask_1d.to(dtype=torch.float32)
            denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
            hm = h_mid.to(dtype=torch.float32)
            hf = h_fin.to(dtype=torch.float32)
            mid_mean = (hm * m.unsqueeze(-1)).sum(dim=1) / denom
            fin_mean = (hf * m.unsqueeze(-1)).sum(dim=1) / denom
            # Last valid residue index per sequence
            L = m.shape[1]
            idx_grid = torch.arange(L, device=m.device).unsqueeze(0).expand_as(m)
            last_idx = torch.where(m > 0, idx_grid, torch.zeros_like(idx_grid)).amax(dim=1)
            row_ar = torch.arange(hm.shape[0], device=hm.device)
            mid_last = hm[row_ar, last_idx]
            fin_last = hf[row_ar, last_idx]
            return torch.stack([mid_mean, mid_last, fin_mean, fin_last], dim=1).cpu().numpy()

        bs = max(1, int(batch_size))
        with self._precision_ctx():
            if shared_backbone:
                length = len(sequences[0])
                if any(len(s) != length for s in sequences):
                    raise ValueError("Shared backbone requires equal-length sequences")
                h_V_enc, h_E, E_idx, h_EXV, mask_t = self._encode_single_backbone(coords)  # type: ignore[arg-type]
                order_1 = torch.arange(length, device=self.device, dtype=torch.long).unsqueeze(0)
                S_all = torch.as_tensor(
                    np.stack([_encode_mpnn_seq(s) for s in sequences], axis=0),
                    device=self.device,
                    dtype=torch.long,
                )
                for start in range(0, n_seq, bs):
                    stop = min(start + bs, n_seq)
                    cur = stop - start
                    _, hiddens = self.model.decode_from_encoded(
                        h_V_enc.expand(cur, -1, -1),
                        h_E.expand(cur, -1, -1, -1),
                        E_idx.expand(cur, -1, -1),
                        h_EXV.expand(cur, -1, -1, -1),
                        S_all[start:stop],
                        mask_t.expand(cur, -1),
                        order_1.expand(cur, -1),
                        return_layer_hiddens=True,
                    )
                    # 3 decoder layers: middle=(3-1)//2=1, final=2
                    blocks[start:stop] = _pool_layers(hiddens[1], hiddens[2], mask_t.expand(cur, -1))
            else:
                coords_list = list(coords)
                if len(coords_list) != n_seq:
                    raise ValueError(f"Expected {n_seq} coordinate arrays, got {len(coords_list)}")
                for i, (seq, c_arr) in enumerate(zip(sequences, coords_list)):
                    h_V_enc, h_E, E_idx, h_EXV, mask_t = self._encode_single_backbone(c_arr)
                    S_t = torch.from_numpy(_encode_mpnn_seq(seq)).to(device=self.device, dtype=torch.long).unsqueeze(0)
                    order_1 = torch.arange(len(seq), device=self.device, dtype=torch.long).unsqueeze(0)
                    _, hiddens = self.model.decode_from_encoded(
                        h_V_enc, h_E, E_idx, h_EXV, S_t, mask_t, order_1, return_layer_hiddens=True
                    )
                    blocks[i : i + 1] = _pool_layers(hiddens[1], hiddens[2], mask_t)

        if return_projected_1024:
            return project_four_blocks(blocks)
        return blocks

    @torch.no_grad()
    def extract_contact_matrix(
        self,
        sequence: str,
        coords: np.ndarray,
        *,
        probe_aas: Sequence[str] | None = None,
        batch_size: int = 32,
        apc: bool = True,
    ) -> dict[str, np.ndarray]:
        """Measure 3D GNN perturbation-propagation contact sensitivity ``J_{ij}``.

        Perturbs each residue ``i`` across ``probe_aas`` (default: all 19 non-WT
        AA20 residues when ``L <= 160`` or 4 biochemically diverse probes
        ``('A', 'G', 'D', 'W')`` for longer chains unless explicitly provided) and
        measures the L2 shift in conditional AA20 log-probabilities at every site
        ``j`` decoded LAST on ``X_WT``. Returns both raw symmetric ``sensitivity``
        and APC-corrected ``couplings`` of shape ``(L, L)`` float32.
        """
        length = len(sequence)
        encoded = self._encode_single_backbone(coords)
        h_V_enc, h_E, E_idx, h_EXV, mask_t = encoded
        wt_idx = _encode_mpnn_seq(sequence)
        base_lp = self.site_conditional_logprobs(
            sequence, coords, positions=None, batch_size=batch_size, encoded_backbone=encoded
        )  # (L, 20)

        # Build (L, L) decoding orders where row j places site j LAST.
        base_order = np.arange(length, dtype=np.int64)
        orders_all = np.empty((length, length), dtype=np.int64)
        for j in range(length):
            if j == length - 1:
                orders_all[j] = base_order
            else:
                orders_all[j, :j] = base_order[:j]
                orders_all[j, j : length - 1] = base_order[j + 1 :]
                orders_all[j, length - 1] = j

        # Only neighbors in E_idx (or 3-hop GNN neighborhood) can respond; exact full pass over j is fast!
        probes = list(probe_aas) if probe_aas is not None else (list(AA20) if length <= 128 else ["A", "G", "D", "K", "W"])
        raw = np.zeros((length, length), dtype=np.float64)
        S_wt_t = torch.from_numpy(wt_idx).to(device=self.device, dtype=torch.long)
        base_lp_t = torch.from_numpy(base_lp).to(device=self.device, dtype=torch.float32)
        orders_t = torch.from_numpy(orders_all).to(device=self.device, dtype=torch.long)
        bs = max(1, int(batch_size))

        with self._precision_ctx():
            for i in range(length):
                alt_ids = [MPNN_AA_TO_IDX[aa] for aa in probes if MPNN_AA_TO_IDX[aa] != wt_idx[i]]
                if not alt_ids:
                    continue
                acc_j = torch.zeros(length, device=self.device, dtype=torch.float32)
                for alt in alt_ids:
                    S_pert = S_wt_t.clone()
                    S_pert[i] = alt
                    S_pert_b = S_pert.unsqueeze(0)
                    for start in range(0, length, bs):
                        stop = min(start + bs, length)
                        cur = stop - start
                        logits, _ = self.model.decode_from_encoded(
                            h_V_enc.expand(cur, -1, -1),
                            h_E.expand(cur, -1, -1, -1),
                            E_idx.expand(cur, -1, -1),
                            h_EXV.expand(cur, -1, -1, -1),
                            S_pert_b.expand(cur, -1),
                            mask_t.expand(cur, -1),
                            orders_t[start:stop],
                        )
                        row_ar = torch.arange(cur, device=self.device, dtype=torch.long)
                        j_idx = torch.arange(start, stop, device=self.device, dtype=torch.long)
                        lp_pert = F.log_softmax(logits[row_ar, j_idx, :20].to(dtype=torch.float32), dim=-1)
                        diff_l2 = torch.linalg.norm(lp_pert - base_lp_t[start:stop], dim=-1)
                        acc_j[start:stop] += diff_l2
                raw[i, :] = (acc_j / float(len(alt_ids))).cpu().numpy()

        np.fill_diagonal(raw, 0.0)
        sym = 0.5 * (raw + raw.T)
        if apc and length > 2:
            row_mean = sym.sum(axis=1) / float(length - 1)
            grand_mean = float(sym.sum()) / float(length * (length - 1))
            mat = sym - np.outer(row_mean, row_mean) / max(grand_mean, 1e-12) if grand_mean > 0 else sym.copy()
            np.fill_diagonal(mat, 0.0)
        else:
            mat = sym.copy()
        return {
            "sensitivity": sym.astype(np.float32),
            "couplings": mat.astype(np.float32),
        }

    @torch.no_grad()
    def score_context(
        self,
        target_seq: str,
        target_coords: np.ndarray,
        context_seqs: Sequence[str] = (),
        context_coords: Sequence[np.ndarray] = (),
        *,
        n_permutations: int = 4,
        seed: int = 20261125,
        spatial_gap_angstrom: float | None = 5.5,
    ) -> dict[str, Any]:
        """Score ``target_seq`` alone (``no_context``) or with visible context chains (``chain_M=0``).

        When ``context_seqs`` and ``context_coords`` are provided, each context chain is
        placed as a distinct visible chain (``chain_M=0``, residue index offset ``+100`` per
        chain) decoded strictly BEFORE ``target_seq`` (``chain_M=1``). If
        ``spatial_gap_angstrom`` is not ``None``, each context backbone is rigidly translated
        along the x-axis to sit adjacent to the target backbone surface at ``spatial_gap_angstrom``
        (preventing steric overlap while allowing ``top_k=48`` cross-chain GNN message passing).
        """
        if len(context_seqs) != len(context_coords):
            raise ValueError("context_seqs and context_coords must have equal length")

        tgt_clean, tgt_valid = _prepare_coords_array(target_coords, ca_only=self.spec.ca_only)
        L_tgt = len(target_seq)
        if tgt_clean.shape[0] != L_tgt:
            raise ValueError(f"target_seq length {L_tgt} != target_coords length {tgt_clean.shape[0]}")

        coords_parts = []
        valid_parts = []
        seq_parts = []
        res_idx_parts = []
        chain_parts = []
        chain_id = 1
        offset_base = 0

        # Track target bounding box along x-axis for collision-free placement
        tgt_ca = tgt_clean if self.spec.ca_only else tgt_clean[:, 1, :]
        tgt_valid_ca = tgt_ca[tgt_valid > 0.5]
        cur_max_x = float(tgt_valid_ca[:, 0].max()) if len(tgt_valid_ca) else 0.0
        tgt_center_yz = tgt_valid_ca[:, 1:].mean(axis=0) if len(tgt_valid_ca) else np.zeros(2, dtype=np.float32)

        n_ctx_res = 0
        for c_seq, c_arr in zip(context_seqs, context_coords):
            c_clean, c_valid = _prepare_coords_array(c_arr, ca_only=self.spec.ca_only)
            if c_clean.shape[0] != len(c_seq):
                raise ValueError(f"Context seq len {len(c_seq)} != coords len {c_clean.shape[0]}")
            if spatial_gap_angstrom is not None:
                c_ca = c_clean if self.spec.ca_only else c_clean[:, 1, :]
                cv = c_ca[c_valid > 0.5]
                if len(cv):
                    shift = np.zeros(3, dtype=np.float32)
                    shift[0] = (cur_max_x + float(spatial_gap_angstrom)) - float(cv[:, 0].min())
                    shift[1:] = tgt_center_yz - cv[:, 1:].mean(axis=0)
                    c_clean = c_clean + shift
                    cur_max_x = float((cv[:, 0] + shift[0]).max())
            L_c = len(c_seq)
            coords_parts.append(c_clean)
            valid_parts.append(c_valid)
            seq_parts.append(_encode_mpnn_seq(c_seq))
            res_idx_parts.append(offset_base + np.arange(L_c, dtype=np.int64))
            chain_parts.append(np.full(L_c, chain_id, dtype=np.int64))
            offset_base += L_c + 100
            chain_id += 1
            n_ctx_res += L_c

        # Append target chain last
        coords_parts.append(tgt_clean)
        valid_parts.append(tgt_valid)
        seq_parts.append(_encode_mpnn_seq(target_seq))
        res_idx_parts.append(offset_base + np.arange(L_tgt, dtype=np.int64))
        chain_parts.append(np.full(L_tgt, chain_id, dtype=np.int64))

        all_coords = np.concatenate(coords_parts, axis=0)
        all_valid = np.concatenate(valid_parts, axis=0)
        all_S = np.concatenate(seq_parts, axis=0)
        all_res = np.concatenate(res_idx_parts, axis=0)
        all_chain = np.concatenate(chain_parts, axis=0)

        X_t = torch.from_numpy(all_coords).unsqueeze(0).to(device=self.device, dtype=self.torch_dtype)
        mask_t = torch.from_numpy(all_valid).unsqueeze(0).to(device=self.device, dtype=self.torch_dtype)
        res_t = torch.from_numpy(all_res).unsqueeze(0).to(device=self.device, dtype=torch.long)
        chain_t = torch.from_numpy(all_chain).unsqueeze(0).to(device=self.device, dtype=torch.long)
        S_t = torch.from_numpy(all_S).unsqueeze(0).to(device=self.device, dtype=torch.long)

        with self._precision_ctx():
            h_V_enc, h_E, E_idx, h_EXV = self.model.encode_structure(X_t, mask_t, res_t, chain_t)
            # Count cross-chain edges entering target chain
            tgt_edges = E_idx[0, n_ctx_res:, :]
            n_cross_edges = int((tgt_edges < n_ctx_res).sum().item()) if n_ctx_res > 0 else 0

            tgt_perms = _canonical_permutations(L_tgt, n_permutations=n_permutations, seed=seed)
            ctx_prefix = np.arange(n_ctx_res, dtype=np.int64)
            orders = np.empty((n_permutations, n_ctx_res + L_tgt), dtype=np.int64)
            for p_idx in range(n_permutations):
                if n_ctx_res > 0:
                    orders[p_idx, :n_ctx_res] = ctx_prefix
                orders[p_idx, n_ctx_res:] = n_ctx_res + tgt_perms[p_idx]
            orders_t = torch.from_numpy(orders).to(device=self.device, dtype=torch.long)
            logits, _ = self.model.decode_from_encoded(
                h_V_enc.expand(n_permutations, -1, -1),
                h_E.expand(n_permutations, -1, -1, -1),
                E_idx.expand(n_permutations, -1, -1),
                h_EXV.expand(n_permutations, -1, -1, -1),
                S_t.expand(n_permutations, -1),
                mask_t.expand(n_permutations, -1),
                orders_t,
            )
            tgt_logits = logits[:, n_ctx_res:, :20].to(dtype=torch.float32)
            tgt_lp = F.log_softmax(tgt_logits, dim=-1)
            tgt_S = S_t[:, n_ctx_res:].expand(n_permutations, -1)
            gathered = torch.gather(tgt_lp, 2, tgt_S.clamp(max=19).unsqueeze(-1)).squeeze(-1)
            tgt_mask = mask_t[:, n_ctx_res:].to(dtype=torch.float32) * (tgt_S < 20).to(dtype=torch.float32)
            ll_per_perm = (gathered * tgt_mask).sum(dim=1).cpu().numpy().astype(np.float64)
            n_valid = float(tgt_mask[0].sum().item())
            mean_ll_sum = float(ll_per_perm.mean())
            nll_per_res = -mean_ll_sum / max(n_valid, 1.0)
            per_pos_nll = (-gathered.mean(dim=0)).cpu().numpy().astype(np.float32)

        return {
            "arm": self.arm,
            "n_target_residues": L_tgt,
            "n_context_chains": len(context_seqs),
            "n_context_residues": n_ctx_res,
            "n_cross_chain_edges": n_cross_edges,
            "ll_sum": mean_ll_sum,
            "pll": mean_ll_sum,
            "nll_per_residue": float(nll_per_res),
            "per_pos_nll": per_pos_nll,
            "n_permutations": int(n_permutations),
        }

    @torch.no_grad()
    def generate(
        self,
        coords: np.ndarray,
        *,
        n_samples: int = 1,
        seed: int = 20261125,
        temperature: float = 0.1,
        fixed_seq: str | None = None,
        fixed_mask: Sequence[bool] | np.ndarray | None = None,
        decoding_order: Sequence[int] | np.ndarray | None = None,
        omit_X: bool = True,
    ) -> list[dict[str, Any]]:
        """Autoregressive inverse-folding design, span infilling, or suffix continuation.

        * ``fixed_mask``: boolean length-``L`` array where ``True`` = fixed (visible from
          ``fixed_seq``) and ``False`` = designed by ProteinMPNN.
        * If ``fixed_mask is None``: all ``L`` positions are designed from backbone alone.
        * ``temperature``: ``0.1`` (canonical ProteinMPNN) or ``1.0`` (unannealed).
        """
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        clean, valid = _prepare_coords_array(coords, ca_only=self.spec.ca_only)
        length = clean.shape[0]
        if fixed_mask is not None:
            fmask = np.asarray(fixed_mask, dtype=bool)
            if fmask.shape != (length,):
                raise ValueError(f"fixed_mask shape {fmask.shape} != ({length},)")
            if fixed_seq is None or len(fixed_seq) != length:
                raise ValueError("fixed_seq of length L is required when fixed_mask is provided")
            S_true_np = _encode_mpnn_seq(fixed_seq)
            design_mask_np = (~fmask).astype(np.float32) * valid
        else:
            S_true_np = np.full(length, MPNN_X_IDX, dtype=np.int64)
            design_mask_np = valid.copy()

        h_V_enc, h_E, E_idx, h_EXV, mask_t = self._encode_single_backbone(coords)
        B = int(n_samples)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))

        if decoding_order is not None:
            ord_np = np.asarray(decoding_order, dtype=np.int64)
            if ord_np.shape != (length,):
                raise ValueError(f"decoding_order shape {ord_np.shape} != ({length},)")
            order_t = torch.from_numpy(ord_np).unsqueeze(0).expand(B, -1).to(device=self.device, dtype=torch.long)
        else:
            # Official ProteinMPNN ordering: fixed positions (design_mask=0) decoded strictly before designed (=1)
            randn = torch.randn((B, length), generator=gen).to(device=self.device, dtype=torch.float32)
            dm_t = torch.from_numpy(design_mask_np).unsqueeze(0).to(device=self.device, dtype=torch.float32)
            order_t = torch.argsort((dm_t + 0.0001) * torch.abs(randn), dim=-1)

        h_V_b = h_V_enc.expand(B, -1, -1)
        h_E_b = h_E.expand(B, -1, -1, -1)
        E_idx_b = E_idx.expand(B, -1, -1)
        mask_b = mask_t.expand(B, -1)
        h_EXV_b = h_EXV.expand(B, -1, -1, -1)
        S_true_b = torch.from_numpy(S_true_np).unsqueeze(0).expand(B, -1).to(device=self.device, dtype=torch.long)
        dm_b = torch.from_numpy(design_mask_np).unsqueeze(0).expand(B, -1).to(device=self.device, dtype=torch.float32)

        order_mask_bw = _order_mask_backward(order_t, dtype=h_V_b.dtype)
        mask_attend = torch.gather(order_mask_bw, 2, E_idx_b).unsqueeze(-1)
        mask_1D = mask_b.view(B, length, 1, 1)
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)
        h_EXV_fw = mask_fw * h_EXV_b

        h_S = torch.zeros_like(h_V_b)
        S = torch.zeros((B, length), dtype=torch.long, device=self.device)
        chosen_lp = torch.zeros((B, length), dtype=torch.float32, device=self.device)
        h_V_stack = [h_V_b.clone()] + [torch.zeros_like(h_V_b) for _ in range(len(self.model.decoder_layers))]

        # Deterministic sampling generator on device
        sample_gen = torch.Generator(device=self.device).manual_seed(int(seed) + 17)

        with self._precision_ctx():
            for t_step in range(length):
                t = order_t[:, t_step]  # (B,)
                dm_t_step = torch.gather(dm_b, 1, t[:, None])  # (B, 1)
                mask_t_step = torch.gather(mask_b, 1, t[:, None])  # (B, 1)
                if (mask_t_step == 0).all():
                    S_t = torch.gather(S_true_b, 1, t[:, None])
                    lp_t = torch.zeros((B, 1), device=self.device, dtype=torch.float32)
                else:
                    E_idx_t = torch.gather(E_idx_b, 1, t[:, None, None].expand(-1, 1, E_idx_b.shape[-1]))
                    h_E_t = torch.gather(h_E_b, 1, t[:, None, None, None].expand(-1, 1, h_E_b.shape[-2], h_E_b.shape[-1]))
                    h_ES_t = _cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                    h_EXV_t = torch.gather(
                        h_EXV_fw, 1, t[:, None, None, None].expand(-1, 1, h_EXV_fw.shape[-2], h_EXV_fw.shape[-1])
                    )
                    mask_bw_t = torch.gather(
                        mask_bw, 1, t[:, None, None, None].expand(-1, 1, mask_bw.shape[-2], mask_bw.shape[-1])
                    )
                    for l_idx, layer in enumerate(self.model.decoder_layers):
                        h_ESV_dec_t = _cat_neighbors_nodes(h_V_stack[l_idx], h_ES_t, E_idx_t)
                        h_V_t = torch.gather(
                            h_V_stack[l_idx], 1, t[:, None, None].expand(-1, 1, h_V_stack[l_idx].shape[-1])
                        )
                        h_ESV_t = mask_bw_t * h_ESV_dec_t + h_EXV_t
                        h_V_stack[l_idx + 1].scatter_(
                            1,
                            t[:, None, None].expand(-1, 1, h_V_b.shape[-1]),
                            layer(h_V_t, h_ESV_t, mask_V=mask_t_step),
                        )
                    h_V_last = torch.gather(
                        h_V_stack[-1], 1, t[:, None, None].expand(-1, 1, h_V_stack[-1].shape[-1])
                    )[:, 0]
                    raw_logits = self.model.W_out(h_V_last).to(dtype=torch.float32)
                    if omit_X:
                        raw_logits[:, MPNN_X_IDX] = -1e9
                    logp_true = F.log_softmax(raw_logits, dim=-1)
                    probs = F.softmax(raw_logits / float(temperature), dim=-1)
                    S_t = torch.multinomial(probs, 1, generator=sample_gen)
                    S_true_t = torch.gather(S_true_b, 1, t[:, None])
                    S_t = torch.where(dm_t_step > 0.5, S_t, S_true_t).long()
                    lp_t = torch.gather(logp_true, 1, S_t)

                emb_t = self.model.W_s(S_t)
                h_S.scatter_(1, t[:, None, None].expand(-1, 1, emb_t.shape[-1]), emb_t)
                S.scatter_(1, t[:, None], S_t)
                chosen_lp.scatter_(1, t[:, None], lp_t)

        S_cpu = S.cpu().numpy()
        lp_cpu = chosen_lp.cpu().numpy()
        dm_cpu = design_mask_np > 0.5
        results: list[dict[str, Any]] = []
        for b in range(B):
            seq_str = "".join(MPNN_ALPHABET[int(c)] for c in S_cpu[b])
            designed_lp = lp_cpu[b, dm_cpu]
            results.append(
                {
                    "sequence": seq_str,
                    "length": length,
                    "temperature": float(temperature),
                    "seed": int(seed),
                    "sample_index": b,
                    "n_designed": int(dm_cpu.sum()),
                    "mean_logprob_designed": float(designed_lp.mean()) if designed_lp.size else math.nan,
                    "sum_logprob_designed": float(designed_lp.sum()) if designed_lp.size else 0.0,
                }
            )
        return results

    def infill_window(
        self,
        parent: str,
        coords: np.ndarray,
        *,
        start: int,
        extent: int,
        seed: int = 20261125,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """``R5a`` local span infilling keeping residues outside ``[start, start + extent)`` fixed."""
        length = len(parent)
        if not (0 <= start < length and extent >= 1 and start + extent <= length):
            raise ValueError(f"Invalid span [{start}, {start + extent}) for length {length}")
        fixed = np.ones(length, dtype=bool)
        fixed[start : start + extent] = False
        rec = self.generate(
            coords,
            n_samples=1,
            seed=seed,
            temperature=temperature,
            fixed_seq=parent,
            fixed_mask=fixed,
        )[0]
        rec["start"] = int(start)
        rec["extent"] = int(extent)
        rec["infilled_span"] = rec["sequence"][start : start + extent]
        return rec

    def continue_suffix(
        self,
        wildtype: str,
        coords: np.ndarray,
        *,
        prefix_len: int,
        n_samples: int = 1,
        seed: int = 20261125,
        temperature: float = 0.1,
        n_to_c_order: bool = True,
    ) -> list[dict[str, Any]]:
        """``R4`` Downstream 2 prefix-fixed suffix continuation conditioned on full WT backbone."""
        length = len(wildtype)
        if not (0 <= prefix_len < length):
            raise ValueError(f"Invalid prefix_len={prefix_len} for length {length}")
        fixed = np.zeros(length, dtype=bool)
        fixed[:prefix_len] = True
        order = np.arange(length, dtype=np.int64) if n_to_c_order else None
        out = self.generate(
            coords,
            n_samples=n_samples,
            seed=seed,
            temperature=temperature,
            fixed_seq=wildtype,
            fixed_mask=fixed,
            decoding_order=order,
        )
        for rec in out:
            rec["prefix_len"] = int(prefix_len)
        return out


def load_proteinmpnn_adapter(
    arm_id: str,
    *,
    model_root: Path | str | None = None,
    device: str = "cpu",
    dtype: str = "float32",
) -> ProteinMPNNAdapter:
    """Convenience factory returning a loaded ``ProteinMPNNAdapter``."""
    return ProteinMPNNAdapter(arm_id, model_root=model_root, device=device, dtype=dtype)
