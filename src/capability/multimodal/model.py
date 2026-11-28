"""Matched Multi-Track Protein Transformer (AA sequence + 3Di + C_beta geometry).

Crucial controlled-experiment invariant:
  - The Transformer trunk (``blocks`` + ``norm_f``) has the **exact same**
    ``n_params_non_embedding`` as Stage-1 ``UniformVocabTransformer`` and Stage-2
    ``VariableLengthProteinTransformer`` across all scales (``0.5m, 2m, 8m, 35m, 150m, 650m``).
  - Structural modality encoders (``struct_3di_emb``, ``geom_proj``, ``pair_dist_emb``) and
    auxiliary structural prediction heads (``struct_3di_head``, ``contact_head``) live outside
    the trunk and are strictly zero-gated whenever ``struct_present == False`` (Z = empty),
    guaranteeing zero structural leakage in ``pure_seq`` and ``joint_to_seq`` pure-sequence eval.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from ..supply_demand.model import (
    RMSNorm,
    SwiGLUFFN,
    apply_rotary_pos_emb,
    precompute_rope_cos_sin,
)
from .spec import (
    AA_INPUT_VOCAB_SIZE,
    DEFAULT_MAX_SEQ_LEN,
    GEOM_FEAT_DIM,
    PAD_TOKEN_ID,
    PAIR_DIST_BIN_EDGES,
    PAIR_DIST_NUM_BINS,
    SCALE_SPECS,
    STRUCT_ABSENT_TOKEN_ID,
    STRUCT_INPUT_VOCAB_SIZE,
    STRUCT_PAD_TOKEN_ID,
    VOCAB_3DI,
    VOCAB_AA,
    ScaleSpec,
    canonicalize_mode,
    canonicalize_objective,
)


@dataclass
class MultimodalOutput:
    """Structured multi-track outputs from MultiTrackProteinTransformer."""

    aa_logits: torch.Tensor                  # (B, T, 20) fp32
    struct_3di_logits: torch.Tensor | None   # (B, T, 20) fp32 when structure head active
    contact_logits: torch.Tensor | None      # (B, T, T) fp32 symmetric contact logits when active
    hidden_states: torch.Tensor | None = None


class MultiTrackSelfAttention(nn.Module):
    """Exact Stage-1/Stage-2 QK-Norm + RoPE self-attention accepting optional additive pair bias."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        is_causal: bool,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.is_causal = bool(is_causal)

        self.W_qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        sdpa_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q, k, v = self.W_qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if sdpa_mask is not None:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=sdpa_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=self.is_causal,
            )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.W_o(attn_out)


class MultiTrackTransformerBlock(nn.Module):
    """Pre-LN Transformer block with exact parameter parity with Stage-1/Stage-2 TransformerBlock."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        *,
        is_causal: bool,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = MultiTrackSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            is_causal=is_causal,
            norm_eps=norm_eps,
        )
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        sdpa_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, sdpa_mask=sdpa_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SymmetricContactHead(nn.Module):
    """Lightweight symmetric low-rank pairwise contact head (B, L, d) -> (B, L, L) contact logits."""

    def __init__(self, d_model: int, proj_dim: int = 32) -> None:
        super().__init__()
        self.proj = nn.Linear(int(d_model), int(proj_dim), bias=False)
        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(float(proj_dim))))
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.proj(h.float())
        logits = torch.matmul(z, z.transpose(-1, -2)) * self.scale.float() + self.bias.float()
        return 0.5 * (logits + logits.transpose(-1, -2))


class MultiTrackProteinTransformer(nn.Module):
    """Matched Multi-Track Protein Transformer supporting pure_seq, struct_cond, and joint_to_seq."""

    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    dist_edges: torch.Tensor

    def __init__(
        self,
        spec: ScaleSpec | str,
        *,
        mode: str = "pure_seq",
        objective: str = "oa_uniform",
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        use_pair_geom: bool = True,
        contact_proj_dim: int = 32,
        rope_base: float = 10000.0,
        norm_eps: float = 1e-6,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(spec, str):
            if spec not in SCALE_SPECS:
                raise ValueError(f"Unknown scale_id {spec!r}; expected {tuple(SCALE_SPECS)}")
            spec = SCALE_SPECS[spec]
        self.spec = spec
        self.mode = canonicalize_mode(mode)
        self.objective = canonicalize_objective(objective)
        self.max_seq_len = int(max_seq_len)
        self.use_pair_geom = bool(use_pair_geom) and (self.mode != "pure_seq")
        self.rope_base = float(rope_base)
        self.is_causal = self.objective == "ar"
        self.gradient_checkpointing = bool(gradient_checkpointing)

        # AA sequence I/O
        self.tok_emb = nn.Embedding(AA_INPUT_VOCAB_SIZE, spec.d_model)
        self.lm_head = nn.Linear(spec.d_model, VOCAB_AA, bias=False)

        # Matched Transformer trunk (exact Stage-1/Stage-2 non-embedding parameter count!)
        self.blocks = nn.ModuleList(
            [
                MultiTrackTransformerBlock(
                    d_model=spec.d_model,
                    n_heads=spec.n_heads,
                    d_ff=spec.d_ff,
                    is_causal=self.is_causal,
                    norm_eps=norm_eps,
                )
                for _ in range(spec.n_layers)
            ]
        )
        self.norm_f = RMSNorm(spec.d_model, eps=norm_eps)

        # Optional structural modality adapters & heads (outside trunk!)
        self.has_struct_inputs = self.mode in ("struct_cond", "joint_to_seq")
        self.has_struct_heads = self.mode == "joint_to_seq"

        if self.has_struct_inputs:
            self.struct_3di_emb = nn.Embedding(STRUCT_INPUT_VOCAB_SIZE, spec.d_model)
            self.geom_proj = nn.Linear(GEOM_FEAT_DIM, spec.d_model, bias=False)
            self.pair_dist_emb = (
                nn.Embedding(PAIR_DIST_NUM_BINS, spec.n_heads) if self.use_pair_geom else None
            )
        else:
            self.struct_3di_emb = None
            self.geom_proj = None
            self.pair_dist_emb = None

        if self.has_struct_heads:
            self.struct_3di_head = nn.Linear(spec.d_model, VOCAB_3DI, bias=False)
            self.contact_head = SymmetricContactHead(spec.d_model, proj_dim=contact_proj_dim)
        else:
            self.struct_3di_head = None
            self.contact_head = None

        cos, sin = precompute_rope_cos_sin(self.max_seq_len, spec.head_dim, base=self.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer(
            "dist_edges",
            torch.tensor(PAIR_DIST_BIN_EDGES, dtype=torch.float32),
            persistent=False,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        base_std = 0.02
        residual_std = base_std / math.sqrt(2.0 * float(self.spec.n_layers))
        readout_std = base_std / math.sqrt(float(self.spec.d_model))

        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=base_std)
        for block in self.blocks:
            assert isinstance(block, MultiTrackTransformerBlock)
            nn.init.normal_(block.attn.W_qkv.weight, mean=0.0, std=base_std)
            nn.init.normal_(block.attn.W_o.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.W_gate_up.weight, mean=0.0, std=base_std)
            nn.init.normal_(block.ffn.W_down.weight, mean=0.0, std=residual_std)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=readout_std)

        if self.struct_3di_emb is not None:
            nn.init.normal_(self.struct_3di_emb.weight, mean=0.0, std=base_std)
            with torch.no_grad():
                self.struct_3di_emb.weight[STRUCT_PAD_TOKEN_ID].zero_()
                self.struct_3di_emb.weight[STRUCT_ABSENT_TOKEN_ID].zero_()
        if self.geom_proj is not None:
            nn.init.normal_(self.geom_proj.weight, mean=0.0, std=base_std)
        if self.pair_dist_emb is not None:
            nn.init.zeros_(self.pair_dist_emb.weight)
        if self.struct_3di_head is not None:
            nn.init.normal_(self.struct_3di_head.weight, mean=0.0, std=readout_std)
        if self.contact_head is not None:
            nn.init.normal_(self.contact_head.proj.weight, mean=0.0, std=readout_std)

    def _get_rope_tables(
        self,
        cur_seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cur_seq_len <= self.rope_cos.shape[2]:
            return (
                self.rope_cos[:, :, :cur_seq_len, :],
                self.rope_sin[:, :, :cur_seq_len, :],
            )
        return precompute_rope_cos_sin(
            cur_seq_len, self.spec.head_dim, base=self.rope_base, device=device
        )

    def _compute_pair_dist_bias(
        self,
        cb_coords: torch.Tensor,
        cb_mask: torch.Tensor,
        struct_present: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute additive pairwise C_beta distance attention bias (B, H, T, T) strictly zeroed when absent."""
        assert self.pair_dist_emb is not None
        # Compute pairwise Euclidean distance in fp32
        coords_f = cb_coords.float()
        diff = coords_f.unsqueeze(2) - coords_f.unsqueeze(1)
        dists = torch.sqrt((diff * diff).sum(dim=-1).clamp(min=0.0))  # (B, T, T)
        # Discretize into bins 1..PAIR_DIST_NUM_BINS-1; bin 0 reserved strictly for missing/absent
        raw_bins = torch.bucketize(dists, self.dist_edges) + 1
        pair_valid = (
            cb_mask.unsqueeze(2)
            & cb_mask.unsqueeze(1)
            & struct_present.view(-1, 1, 1)
        )
        bins = torch.where(pair_valid, raw_bins, torch.zeros_like(raw_bins))
        # (B, T, T, H) -> (B, H, T, T), explicitly zeroing non-valid entries
        bias = self.pair_dist_emb(bins).permute(0, 3, 1, 2)
        bias = torch.where(pair_valid.unsqueeze(1), bias, torch.zeros_like(bias))
        return bias.to(dtype=target_dtype)

    def _build_sdpa_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        right_padded: bool,
        pair_bias: torch.Tensor | None,
        target_dtype: torch.dtype,
    ) -> torch.Tensor | None:
        valid = (
            (input_ids != PAD_TOKEN_ID)
            if attention_mask is None
            else attention_mask.to(dtype=torch.bool)
        )
        cur_len = input_ids.shape[1]

        if pair_bias is None:
            if self.is_causal:
                if right_padded:
                    return None
                tril = torch.ones((cur_len, cur_len), dtype=torch.bool, device=input_ids.device).tril()
                mask = tril.unsqueeze(0).unsqueeze(1) & valid[:, None, None, :]
                diag = torch.eye(cur_len, dtype=torch.bool, device=input_ids.device).unsqueeze(0).unsqueeze(1)
                return mask | diag
            return valid[:, None, None, :]

        # Combine boolean attention admissibility mask with additive float pair_bias
        if self.is_causal:
            tril = torch.ones((cur_len, cur_len), dtype=torch.bool, device=input_ids.device).tril()
            allow = tril.unsqueeze(0).unsqueeze(1) & valid[:, None, None, :]
            diag = torch.eye(cur_len, dtype=torch.bool, device=input_ids.device).unsqueeze(0).unsqueeze(1)
            allow = allow | diag
        else:
            allow = valid[:, None, None, :].expand(-1, 1, cur_len, -1)

        neg_inf = torch.finfo(target_dtype).min
        return torch.where(allow, pair_bias, torch.full_like(pair_bias, neg_inf))

    def forward_multimodal(
        self,
        aa_input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        struct_tokens: torch.Tensor | None = None,
        geom_feats: torch.Tensor | None = None,
        cb_coords: torch.Tensor | None = None,
        cb_mask: torch.Tensor | None = None,
        struct_present: torch.Tensor | None = None,
        right_padded: bool = True,
        return_struct_heads: bool = True,
    ) -> MultimodalOutput:
        """Forward pass supporting both pure-sequence (Z = empty) and structure-conditioned tracks."""
        bsz, cur_seq_len = aa_input_ids.shape
        device = aa_input_ids.device
        cos, sin = self._get_rope_tables(cur_seq_len, device)

        x = self.tok_emb(aa_input_ids)
        pair_bias: torch.Tensor | None = None

        if self.has_struct_inputs and self.struct_3di_emb is not None:
            if struct_present is None:
                has_any_struct = struct_tokens is not None or geom_feats is not None or cb_coords is not None
                present_b = torch.full((bsz,), bool(has_any_struct), dtype=torch.bool, device=device)
            else:
                present_b = struct_present.to(device=device, dtype=torch.bool).view(bsz)

            if struct_tokens is not None:
                # Strictly replace struct_tokens with STRUCT_ABSENT_TOKEN_ID whenever present_b is False
                eff_s3 = torch.where(
                    present_b.view(bsz, 1),
                    struct_tokens.to(device=device, dtype=torch.long),
                    torch.full((bsz, cur_seq_len), STRUCT_ABSENT_TOKEN_ID, dtype=torch.long, device=device),
                )
            else:
                eff_s3 = torch.full(
                    (bsz, cur_seq_len), STRUCT_ABSENT_TOKEN_ID, dtype=torch.long, device=device
                )
            x = x + self.struct_3di_emb(eff_s3)

            if geom_feats is not None and self.geom_proj is not None and bool(present_b.any().item()):
                # Zero out geometric features strictly for sequences with struct_present == False
                gated_geom = geom_feats.to(device=device, dtype=x.dtype) * present_b.view(bsz, 1, 1).to(dtype=x.dtype)
                x = x + self.geom_proj(gated_geom)

            if (
                self.pair_dist_emb is not None
                and cb_coords is not None
                and cb_mask is not None
                and bool(present_b.any().item())
            ):
                pair_bias = self._compute_pair_dist_bias(
                    cb_coords.to(device=device),
                    cb_mask.to(device=device, dtype=torch.bool),
                    present_b,
                    target_dtype=x.dtype,
                )

        sdpa_mask = self._build_sdpa_mask(
            aa_input_ids,
            attention_mask,
            right_padded=right_padded,
            pair_bias=pair_bias,
            target_dtype=x.dtype,
        )

        use_ckpt = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if use_ckpt:
                x = activation_checkpoint(block, x, cos, sin, sdpa_mask, use_reentrant=False)
            else:
                x = block(x, cos, sin, sdpa_mask=sdpa_mask)
        h = self.norm_f(x)

        amp_device = "cuda" if h.is_cuda else "cpu"
        with torch.amp.autocast(amp_device, enabled=False):
            h_fp32 = h.float()
            aa_logits = F.linear(h_fp32, self.lm_head.weight.float())
            struct_3di_logits = (
                F.linear(h_fp32, self.struct_3di_head.weight.float())
                if (return_struct_heads and self.struct_3di_head is not None)
                else None
            )
            contact_logits = (
                self.contact_head(h_fp32)
                if (return_struct_heads and self.contact_head is not None)
                else None
            )

        return MultimodalOutput(
            aa_logits=aa_logits,
            struct_3di_logits=struct_3di_logits,
            contact_logits=contact_logits,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        struct_tokens: torch.Tensor | None = None,
        geom_feats: torch.Tensor | None = None,
        cb_coords: torch.Tensor | None = None,
        cb_mask: torch.Tensor | None = None,
        struct_present: torch.Tensor | None = None,
        right_padded: bool = True,
    ) -> torch.Tensor:
        """Drop-in Stage-2 compatible forward returning AA logits (B, T, 20)."""
        return self.forward_multimodal(
            input_ids,
            attention_mask,
            struct_tokens=struct_tokens,
            geom_feats=geom_feats,
            cb_coords=cb_coords,
            cb_mask=cb_mask,
            struct_present=struct_present,
            right_padded=right_padded,
            return_struct_heads=False,
        ).aa_logits


def count_multimodal_parameters(
    model: MultiTrackProteinTransformer,
    spec: ScaleSpec | None = None,
) -> dict[str, int]:
    """Count exact trunk (non-embedding), AA I/O, structure I/O, and total parameters.

    Strictly asserts that ``n_params_non_embedding`` (transformer trunk) matches
    ``ScaleSpec.exact_non_embedding_params`` identically across pure_seq, struct_cond, and joint_to_seq!
    """
    target_spec = spec if spec is not None else model.spec
    trunk_params = sum(p.numel() for p in model.blocks.parameters()) + sum(
        p.numel() for p in model.norm_f.parameters()
    )
    aa_io_params = int(model.tok_emb.weight.numel() + model.lm_head.weight.numel())
    total_params = int(sum(p.numel() for p in model.parameters()))
    struct_io_params = total_params - trunk_params - aa_io_params

    if trunk_params != target_spec.exact_non_embedding_params:
        raise AssertionError(
            f"Multimodal trunk non-embedding param mismatch for {target_spec.scale_id}: "
            f"got {trunk_params}, expected {target_spec.exact_non_embedding_params}"
        )
    return {
        "n_params_non_embedding": int(trunk_params),
        "n_params_trunk": int(trunk_params),
        "n_params_aa_io": int(aa_io_params),
        "n_params_struct_io": int(struct_io_params),
        "n_params_total": int(total_params),
    }
