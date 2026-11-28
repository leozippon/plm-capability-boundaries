"""Modern Pre-LN Transformer (RMSNorm + QK-Norm + RoPE + SwiGLU) for AR and MLM."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .spec import (
    INPUT_VOCAB_SIZE,
    SCALE_SPECS,
    SEQ_LEN,
    VOCAB_AA,
    ScaleSpec,
    canonicalize_paradigm,
)


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization with fp32 variance calculation."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        inv_rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_fp32 * inv_rms * self.weight.float()).to(dtype=x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def precompute_rope_cos_sin(
    seq_len: int,
    head_dim: int,
    *,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute fp32 rotary position embedding tables of shape (1, 1, seq_len, head_dim)."""
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
    inv_freq = 1.0 / (
        float(base)
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / float(head_dim))
    )
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(0)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)
    return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE in fp32 and cast back to q/k dtype so SDPA dtypes match v."""
    target_dtype = q.dtype
    q_fp32 = q.float()
    k_fp32 = k.float()
    q_rot = (q_fp32 * cos + _rotate_half(q_fp32) * sin).to(dtype=target_dtype)
    k_rot = (k_fp32 * cos + _rotate_half(k_fp32) * sin).to(dtype=target_dtype)
    return q_rot, k_rot


class SelfAttention(nn.Module):
    """Multi-head self-attention with per-head QK-Norm, RoPE, and Flash/SDPA kernel."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        is_causal: bool,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
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
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.W_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=self.is_causal,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.W_o(attn_out)


class SwiGLUFFN(nn.Module):
    """Bias-free SwiGLU feed-forward network with zero dropout."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.W_gate_up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.W_gate_up(x).chunk(2, dim=-1)
        return self.W_down(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block shared identically by AR and MLM paradigms."""

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
        self.attn = SelfAttention(
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
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class UniformVocabTransformer(nn.Module):
    """Matched-architecture 20-AA Pre-LN Transformer supporting AR and MLM paradigms."""

    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(
        self,
        spec: ScaleSpec | str,
        paradigm: str,
        *,
        seq_len: int = SEQ_LEN,
        input_vocab_size: int = INPUT_VOCAB_SIZE,
        output_vocab_size: int = VOCAB_AA,
        rope_base: float = 10000.0,
        norm_eps: float = 1e-6,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(spec, str):
            if spec not in SCALE_SPECS:
                raise ValueError(f"Unknown scale_id {spec!r}; expected {tuple(SCALE_SPECS)}")
            spec = SCALE_SPECS[spec]
        canon_paradigm = canonicalize_paradigm(paradigm)

        self.spec = spec
        self.paradigm = canon_paradigm
        self.seq_len = int(seq_len)
        self.input_vocab_size = int(input_vocab_size)
        self.output_vocab_size = int(output_vocab_size)
        self.rope_base = float(rope_base)
        self.is_causal = canon_paradigm == "ar"
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.tok_emb = nn.Embedding(self.input_vocab_size, spec.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
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
        self.lm_head = nn.Linear(spec.d_model, self.output_vocab_size, bias=False)

        cos, sin = precompute_rope_cos_sin(
            self.seq_len,
            spec.head_dim,
            base=self.rope_base,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        base_std = 0.02
        residual_std = base_std / math.sqrt(2.0 * float(self.spec.n_layers))
        readout_std = base_std / math.sqrt(float(self.spec.d_model))

        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=base_std)
        for block in self.blocks:
            assert isinstance(block, TransformerBlock)
            nn.init.normal_(block.attn.W_qkv.weight, mean=0.0, std=base_std)
            nn.init.normal_(block.attn.W_o.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.W_gate_up.weight, mean=0.0, std=base_std)
            nn.init.normal_(block.ffn.W_down.weight, mean=0.0, std=residual_std)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=readout_std)

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
            cur_seq_len,
            self.spec.head_dim,
            base=self.rope_base,
            device=device,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        cur_seq_len = input_ids.shape[1]
        cos, sin = self._get_rope_tables(cur_seq_len, input_ids.device)

        x = self.tok_emb(input_ids)
        use_ckpt = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if use_ckpt:
                x = activation_checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        h = self.norm_f(x)

        amp_device = "cuda" if h.is_cuda else "cpu"
        with torch.amp.autocast(amp_device, enabled=False):
            logits = F.linear(h.float(), self.lm_head.weight.float())
        return logits


def count_model_parameters(
    model: UniformVocabTransformer,
    spec: ScaleSpec | None = None,
) -> dict[str, int]:
    """Count exact embedding, non-embedding, and total parameters, asserting ScaleSpec match."""
    target_spec = spec if spec is not None else model.spec
    n_params_embedding = int(model.tok_emb.weight.numel() + model.lm_head.weight.numel())
    n_params_total = int(sum(p.numel() for p in model.parameters()))
    n_params_non_embedding = n_params_total - n_params_embedding

    if n_params_non_embedding != target_spec.exact_non_embedding_params:
        raise AssertionError(
            f"Non-embedding parameter mismatch for {target_spec.scale_id}: "
            f"got {n_params_non_embedding}, expected {target_spec.exact_non_embedding_params}"
        )
    if (
        model.input_vocab_size == INPUT_VOCAB_SIZE
        and model.output_vocab_size == VOCAB_AA
        and n_params_total != target_spec.exact_total_params
    ):
        raise AssertionError(
            f"Total parameter mismatch for {target_spec.scale_id}: "
            f"got {n_params_total}, expected {target_spec.exact_total_params}"
        )

    return {
        "n_params_non_embedding": n_params_non_embedding,
        "n_params_embedding": n_params_embedding,
        "n_params_total": n_params_total,
    }
