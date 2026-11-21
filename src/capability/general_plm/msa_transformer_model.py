"""Pure-PyTorch, zero-dependency loader for fair-esm MSA Transformer (``esm_msa1b_t12_100M_UR50S.pt``).

Matches fair-esm's ``MSATransformer`` state dict key-for-key with ``strict=True``
so ``msa-transformer-100m`` runs identically in ``ct`` on Compute and inside
air-gapped H200 pods without requiring the ``fair-esm`` package.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Standard 33-token ESM-1b / MSA-1b alphabet (matches ``esm.data.Alphabet.from_architecture('MSA Transformer')``).
MSA_ALPHABET_TOKENS: tuple[str, ...] = (
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
    "<null_1>",
    "<mask>",
)
MSA_TOKEN_TO_ID: dict[str, int] = {tok: idx for idx, tok in enumerate(MSA_ALPHABET_TOKENS)}
MSA_CLS_IDX = MSA_TOKEN_TO_ID["<cls>"]
MSA_PAD_IDX = MSA_TOKEN_TO_ID["<pad>"]
MSA_EOS_IDX = MSA_TOKEN_TO_ID["<eos>"]
MSA_GAP_IDX = MSA_TOKEN_TO_ID["-"]
MSA_MASK_IDX = MSA_TOKEN_TO_ID["<mask>"]


class _RowSelfAttention(nn.Module):
    """Tied multi-head row self-attention across MSA rows (``attn_average='sqrt'``)."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (R, C, B, D)
        num_rows, num_cols, batch_size, embed_dim = x.shape
        q = self.q_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)

        scaling = 1.0 / math.sqrt(self.head_dim * num_rows)
        q = q * scaling
        # Tied row attention weights: sum over rows -> (B, H, C_q, C_k)
        attn_weights = torch.einsum("ribhd,rjbhd->bhij", q, k)
        attn_probs = F.softmax(attn_weights, dim=-1)
        out = torch.einsum("bhij,rjbhd->ribhd", attn_probs, v)
        out = out.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
        return self.out_proj(out)


class _ColumnSelfAttention(nn.Module):
    """Multi-head column self-attention across MSA rows at each sequence column."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (R, C, B, D)
        num_rows, num_cols, batch_size, embed_dim = x.shape
        if num_rows == 1:
            # Exact fair-esm behavior when R=1: softmax over 1 element is 1.0 -> out_proj(v_proj(x))
            v = self.v_proj(x)
            return self.out_proj(v)

        q = self.q_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)

        scaling = 1.0 / math.sqrt(self.head_dim)
        # (B, C, H, R, d)
        q = q.permute(2, 1, 3, 0, 4) * scaling
        k = k.permute(2, 1, 3, 0, 4)
        v = v.permute(2, 1, 3, 0, 4)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_probs = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_probs, v)  # (B, C, H, R, d)
        out = out.permute(3, 1, 0, 2, 4).contiguous().view(num_rows, num_cols, batch_size, embed_dim)
        return self.out_proj(out)


class _NormalizedResidualBlock(nn.Module):
    """Pre-LN residual wrapper matching fair-esm ``NormalizedResidualBlock`` parameter names."""

    def __init__(self, layer: nn.Module, embed_dim: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer(self.layer_norm(x))


class _FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _AxialTransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, num_heads: int, *, attn_order: str = "rowcol") -> None:
        super().__init__()
        self.attn_order = attn_order
        self.row_self_attention = _NormalizedResidualBlock(
            _RowSelfAttention(embed_dim, num_heads), embed_dim
        )
        self.column_self_attention = _NormalizedResidualBlock(
            _ColumnSelfAttention(embed_dim, num_heads), embed_dim
        )
        self.feed_forward_layer = _NormalizedResidualBlock(
            _FeedForward(embed_dim, ffn_dim), embed_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn_order == "colrow":
            x = self.column_self_attention(x)
            x = self.row_self_attention(x)
        else:
            x = self.row_self_attention(x)
            x = self.column_self_attention(x)
        x = self.feed_forward_layer(x)
        return x


class _SentenceEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 33,
        max_positions: int = 1024,
        num_layers: int = 12,
        embed_dim: int = 768,
        ffn_dim: int = 3072,
        num_heads: int = 12,
        attn_order: str = "rowcol",
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=MSA_PAD_IDX)
        self.embed_positions = nn.Embedding(max_positions + MSA_PAD_IDX + 1, embed_dim, padding_idx=MSA_PAD_IDX)
        self.msa_position_embedding = nn.Parameter(torch.zeros(1, max_positions, 1, embed_dim))
        self.emb_layer_norm_before = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList(
            [_AxialTransformerLayer(embed_dim, ffn_dim, num_heads, attn_order=attn_order) for _ in range(num_layers)]
        )
        self.emb_layer_norm_after = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, R, C)
        batch_size, num_alignments, seqlen = tokens.shape
        x = self.embed_tokens(tokens)
        # Fairseq LearnedPositionalEmbedding: non-pad indices start at padding_idx + 1 (= 2)
        mask = tokens.ne(MSA_PAD_IDX).long()
        positions = torch.cumsum(mask, dim=-1) * mask + MSA_PAD_IDX
        x = x + self.embed_positions(positions)
        x = x + self.msa_position_embedding[:, :num_alignments]
        x = self.emb_layer_norm_before(x)
        # Zero out padding tokens (matches fair-esm)
        x = x * mask.unsqueeze(-1).to(x.dtype)
        # Transpose (B, R, C, D) -> (R, C, B, D)
        x = x.permute(1, 2, 0, 3)
        for layer in self.layers:
            x = layer(x)
        x = self.emb_layer_norm_after(x)
        # Back to (B, R, C, D)
        return x.permute(2, 0, 1, 3)


class _RobertaLMHead(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.weight = nn.Parameter(torch.empty(vocab_size, embed_dim))
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.dense(features))
        x = self.layer_norm(x)
        return F.linear(x, self.weight, self.bias)


class PurePyTorchMSATransformer(nn.Module):
    """Exact pure-PyTorch ``MSATransformer`` matching fairseq checkpoint tensor keys 1:1."""

    def __init__(
        self,
        *,
        vocab_size: int = 33,
        max_positions: int = 1024,
        num_layers: int = 12,
        embed_dim: int = 768,
        ffn_dim: int = 3072,
        num_heads: int = 12,
        attn_order: str = "rowcol",
    ) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.sentence_encoder = _SentenceEncoder(
            vocab_size=vocab_size,
            max_positions=max_positions,
            num_layers=num_layers,
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            attn_order=attn_order,
        )
        self.encoder.lm_head = _RobertaLMHead(embed_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder.sentence_encoder(tokens)
        logits = self.encoder.lm_head(hidden)
        return {"logits": logits}


def load_pure_pytorch_msa_transformer(
    pt_path: Path, *, device: str, dtype: torch.dtype
) -> tuple[PurePyTorchMSATransformer, dict[str, Any]]:
    """Load ``esm_msa1b_t12_100M_UR50S.pt`` with ``strict=True`` key verification."""
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    args = ckpt.get("args")
    sd = ckpt["model"]

    num_layers = int(getattr(args, "encoder_layers", 12))
    embed_dim = int(getattr(args, "encoder_embed_dim", 768))
    ffn_dim = int(getattr(args, "encoder_ffn_embed_dim", 3072))
    num_heads = int(getattr(args, "encoder_attention_heads", 12))
    max_positions = int(getattr(args, "max_positions", 1024))
    # Note: fair-esm's AxialTransformerLayer executes row_self_attention then column_self_attention
    # regardless of the legacy fairseq `attn_order` string; we record the checkpoint arg verbatim.
    model = PurePyTorchMSATransformer(
        vocab_size=len(MSA_ALPHABET_TOKENS),
        max_positions=max_positions,
        num_layers=num_layers,
        embed_dim=embed_dim,
        ffn_dim=ffn_dim,
        num_heads=num_heads,
        attn_order="rowcol",
    )
    expected = set(model.state_dict().keys())
    provided = set(sd.keys())
    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"MSA Transformer checkpoint {pt_path.name} key mismatch: "
            f"missing={missing[:5]} ({len(missing)}), unexpected={unexpected[:5]} ({len(unexpected)})"
        )
    model.load_state_dict(sd, strict=True)
    model = model.eval().to(device=device, dtype=dtype)
    meta = {
        "pt_file": pt_path.name,
        "num_layers": num_layers,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "max_positions": max_positions,
        "checkpoint_attn_order_arg": str(getattr(args, "attn_order", "rowcol")),
        "executed_attn_order": "rowcol",
    }
    return model, meta
