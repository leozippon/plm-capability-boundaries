"""Self-contained pure-PyTorch adapter for all 4 official DiMA continuous latent diffusion models.

Covers:
* ``dima-esm2-8m``: 12-layer U-Net ``ScoreEstimator`` (320-d) over frozen ``ESM2-8M`` + 3L Transformer decoder.
* ``dima-esm2-3b``: 12-layer U-Net ``ScoreEstimator`` (2560->320->2560) over frozen ``ESM2-3B`` + 3L Transformer decoder.
* ``dima-saprot-35m``: 12-layer U-Net ``ScoreEstimator`` (480->320->480) over frozen ``SaProt-35M`` (interleaved
  ``"M#K#..."`` sequence-only encoding + exact marginalization over 21 3Di variants per AA on decode) + 3L decoder.
* ``dima-cheap-1024``: 12-layer U-Net ``ScoreEstimator`` (1024->320->1024) over frozen ``ESM2-3B`` +
  ``esmfold_v1_s_proj.pt`` (``s_s_0``, no BOS/EOS) + ``LatentScaler`` (``channel_minmaxnorm``) +
  ``FullyConnectedNetwork`` (1024->512->256->21 OpenFold order, dropping ``X``).

Zero external dependencies on ``omegaconf``, ``hydra``, ``openfold``, ``lightning``, or ``fair-esm``.
Implements the official ``Tanh(d=10.0)`` VP-SDE (``T=1.0, t_min=0.05``), EMA shadow parameter loading,
2-pass self-conditioned denoising, ``EulerDiffEqSolver`` (SDE & deterministic ODE), centered/tiled
``<=510``-aa windowing, leakage-free site-masked denoising conditionals, 8-fold stride-masked denoising PLL,
U-Net bottleneck/output 1024-d JL readouts, RePaint infilling/continuation, and Potts/profile-guided sampling.
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.capability.core.amino_acids import AA20
from src.capability.core.precision_policy import fp32_matmul_context, requested_fp32_matmul_policy
from src.capability.general_plm.adapters import PositionVector
from src.capability.general_plm.eval_utils import project_four_blocks

AA20_INDEX: Mapping[str, int] = MappingProxyType({ch: i for i, ch in enumerate(AA20)})
AA20_SET = frozenset(AA20)

# --------------------------------------------------------------------------- #
# Canonical Constants & Model Specifications
# --------------------------------------------------------------------------- #

OPENFOLD_RESTYPES_20 = "ARNDCQEGHILKMFPSTWYV"
OPENFOLD_RESTYPES_WITH_X = OPENFOLD_RESTYPES_20 + "X"
OPENFOLD_ORDER_WITH_X: Mapping[str, int] = MappingProxyType(
    {ch: idx for idx, ch in enumerate(OPENFOLD_RESTYPES_WITH_X)}
)
OPENFOLD_UNK_IDX = 20
#: Index permutation mapping OpenFold 20-AA order -> canonical alphabetical AA20 order.
OPENFOLD_TO_AA20_INDICES: tuple[int, ...] = tuple(
    OPENFOLD_RESTYPES_20.index(ch) for ch in AA20
)

DIMA_MAX_POS_EMBEDDINGS = 512
DIMA_MAX_RESIDUES = 510  # 510 + <cls> + <eos> == 512
DIMA_DEFAULT_T_EVAL = 0.5
DIMA_DEFAULT_T_MIN = 0.05
DIMA_DEFAULT_T_MAX = 1.0
DIMA_TANH_D = 10.0
DIMA_TANH_T_THR = 0.95


@dataclass(frozen=True)
class DiMASpec:
    """Authoritative file-layout and architecture spec for one DiMA checkpoint."""

    arm_id: str
    encoder_kind: str  # "esm2" | "saprot" | "cheap"
    encoder_dir_names: tuple[str, ...]
    embedding_dim: int
    diffusion_relpath: str
    stats_relpath: str
    decoder_relpath: str | None  # None for CHEAP (uses cheap_assets/sequence_decoder/mlp.ckpt)
    has_bos_eos: bool
    default_dtype: str = "float32"


DIMA_SPECS: Mapping[str, DiMASpec] = MappingProxyType(
    {
        "dima-esm2-8m": DiMASpec(
            arm_id="dima-esm2-8m",
            encoder_kind="esm2",
            encoder_dir_names=("esm2-8m", "esm2_t6_8M_UR50D"),
            embedding_dim=320,
            diffusion_relpath="checkpoints/diffusion_checkpoints/DiMA-bert_35M-ESM2_8M-AFDB/1000000.pth",
            stats_relpath="checkpoints/statistics/encodings-ESM2-8M.pth",
            decoder_relpath="checkpoints/decoder_checkpoints/transformer-decoder-ESM2-8M.pth",
            has_bos_eos=True,
            default_dtype="float32",
        ),
        "dima-esm2-3b": DiMASpec(
            arm_id="dima-esm2-3b",
            encoder_kind="esm2",
            encoder_dir_names=("esm2-3b", "esm2_t36_3B_UR50D"),
            embedding_dim=2560,
            diffusion_relpath="checkpoints/diffusion_checkpoints/DiMA-bert_35M-ESM2_3B-AFDB/1000000.pth",
            stats_relpath="checkpoints/statistics/encodings-ESM2-3B.pth",
            decoder_relpath="checkpoints/decoder_checkpoints/transformer-decoder-ESM2-3B.pth",
            has_bos_eos=True,
            default_dtype="float32",
        ),
        "dima-saprot-35m": DiMASpec(
            arm_id="dima-saprot-35m",
            encoder_kind="saprot",
            encoder_dir_names=("saprot-35m", "SaProt_35M_AF2"),
            embedding_dim=480,
            diffusion_relpath="checkpoints/diffusion_checkpoints/DiMA-bert_35M-SaProt-35M-AFDB/1000000.pth",
            stats_relpath="checkpoints/statistics/encodings-SaProt-35M.pth",
            decoder_relpath="checkpoints/decoder_checkpoints/transformer-decoder-SaProt-35M.pth",
            has_bos_eos=True,
            default_dtype="float32",
        ),
        "dima-cheap-1024": DiMASpec(
            arm_id="dima-cheap-1024",
            encoder_kind="cheap",
            encoder_dir_names=("esm2-3b", "esm2_t36_3B_UR50D"),
            embedding_dim=1024,
            diffusion_relpath="checkpoints/diffusion_checkpoints/DiMA-bert_35M-CHEAP_shorten_1_dim_1024-AFDB/500000.pth",
            stats_relpath="checkpoints/statistics/encodings-CHEAP_shorten_1_dim_1024.pth",
            decoder_relpath=None,
            has_bos_eos=False,
            default_dtype="float32",
        ),
    }
)


# --------------------------------------------------------------------------- #
# Zero-Dependency OmegaConf/Lightning-Safe Checkpoint Unpickler
# --------------------------------------------------------------------------- #


class _ForeignObjectStub:
    """Harmless placeholder for omegaconf/lightning metadata inside .pth/.ckpt files."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._args = args
        self._kwargs = kwargs

    def __setstate__(self, state: Any) -> None:
        self._state = state


class _SafeForeignUnpickler(pickle.Unpickler):
    """Unpickler that stubs out unavailable ``omegaconf``/``hydra``/``lightning`` types."""

    _STUB_PREFIXES = (
        "omegaconf",
        "hydra",
        "pytorch_lightning",
        "lightning",
    )

    def find_class(self, module: str, name: str) -> Any:
        for pref in self._STUB_PREFIXES:
            if module == pref or module.startswith(pref + "."):
                return _ForeignObjectStub
        return super().find_class(module, name)


class _SafePickleNamespace:
    """Minimal pickle-module interface passed to ``torch.load(..., pickle_module=...)``."""

    Unpickler = _SafeForeignUnpickler
    load = staticmethod(lambda file, **kw: _SafeForeignUnpickler(file, **kw).load())
    loads = staticmethod(lambda data, **kw: pickle.loads(data, **kw))
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
    DEFAULT_PROTOCOL = pickle.DEFAULT_PROTOCOL
    PicklingError = pickle.PicklingError
    UnpicklingError = pickle.UnpicklingError


def _omegaconf_safe_torch_load(path: Path | str, map_location: str = "cpu") -> Any:
    """Load a DiMA/CHEAP checkpoint without requiring ``omegaconf`` or ``lightning``."""
    return torch.load(
        str(path),
        map_location=map_location,
        weights_only=False,
        pickle_module=_SafePickleNamespace,
    )


# --------------------------------------------------------------------------- #
# Path Resolution
# --------------------------------------------------------------------------- #


def _candidate_model_roots(model_root: Path | str | None = None) -> list[Path]:
    roots: list[Path] = []
    if model_root is not None:
        roots.append(Path(model_root))
    for env_var in ("DIMA_MODEL_ROOT", "GENERAL_PLM_MODEL_ROOT", "TRANSFER_MODEL_ROOT"):
        val = os.environ.get(env_var, "").strip()
        if val:
            roots.append(Path(val))
    roots.extend(
        [
            Path("/home/lzp/models/general_plm_20261120"),
            Path("/home/lzp/models"),
            Path("/gpfs/jiaotongdamoxing/zhk_zip/models"),
            Path("/gpfs/jiaotongdamoxing/zhk_zip/InterpretabilityTransfer/models"),
        ]
    )
    out: list[Path] = []
    seen: set[Path] = set()
    for r in roots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def resolve_dima_root(dima_root: Path | str | None = None, model_root: Path | str | None = None) -> Path:
    """Locate the staged ``dima_all`` root directory containing ``checkpoints/`` and ``cheap_assets/``."""
    if dima_root is not None:
        p = Path(dima_root)
        if (p / "checkpoints").is_dir():
            return p
        if (p / "dima_all" / "checkpoints").is_dir():
            return p / "dima_all"
        raise FileNotFoundError(f"Invalid dima_root {p}: expected checkpoints/ directory")
    for base in _candidate_model_roots(model_root):
        for cand in (base, base / "dima_all"):
            if (cand / "checkpoints").is_dir():
                return cand
    raise FileNotFoundError(
        "Could not locate DiMA assets (dima_all/checkpoints/). "
        "Pass dima_root=... or set DIMA_MODEL_ROOT / GENERAL_PLM_MODEL_ROOT."
    )


def resolve_encoder_dir(
    dir_names: Sequence[str],
    *,
    encoder_dir: Path | str | None = None,
    model_root: Path | str | None = None,
) -> Path:
    """Locate a staged HuggingFace frozen encoder directory (e.g. ``esm2-8m``, ``esm2-3b``, ``saprot-35m``)."""
    if encoder_dir is not None:
        p = Path(encoder_dir)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Specified encoder_dir does not exist: {p}")
    for base in _candidate_model_roots(model_root):
        for name in dir_names:
            cand = base / name
            if (cand / "config.json").is_file():
                return cand
    raise FileNotFoundError(
        f"Could not locate staged frozen encoder matching {list(dir_names)} under "
        f"{[str(r) for r in _candidate_model_roots(model_root)]}"
    )


# --------------------------------------------------------------------------- #
# Official DiMA ScoreEstimator & TransformerDecoder (100% Verbatim Parity)
# --------------------------------------------------------------------------- #


@dataclass
class ScoreEstimatorConfig:
    """Exact configuration matching ``dima_src/src/configs/model/bert_base.yaml``."""

    embedding_size: int = 320
    hidden_size: int = 320
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    attention_head_size: int = 20
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    max_position_embeddings: int = DIMA_MAX_POS_EMBEDDINGS
    add_cross_attention: bool = False
    qk_norm: bool = False
    use_self_cond: bool = True


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hs = hidden_states.to(torch.float32)
        variance = hs.pow(2).mean(-1, keepdim=True)
        hs = hs * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hs.to(input_dtype)


def _timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class _FlashAttention(nn.Module):
    def __init__(self, config: ScoreEstimatorConfig) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qk_norm = config.qk_norm

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.query_norm: nn.Module = _RMSNorm(self.attention_head_size) if self.qk_norm else nn.Identity()
        self.key_norm: nn.Module = _RMSNorm(self.attention_head_size) if self.qk_norm else nn.Identity()
        self.dropout_p = config.attention_probs_dropout_prob
        self.o_proj = nn.Linear(self.all_head_size, config.hidden_size)

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(new_shape).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.query_norm(self._transpose_for_scores(self.query(hidden_states)))
        k = self.key_norm(self._transpose_for_scores(self.key(hidden_states)))
        v = self._transpose_for_scores(self.value(hidden_states))
        attn_mask = attention_mask.to(dtype=q.dtype) if attention_mask is not None else None
        ctx = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(ctx.size()[:-2] + (self.all_head_size,))
        return self.o_proj(ctx)


class _BertAttention(nn.Module):
    def __init__(self, config: ScoreEstimatorConfig) -> None:
        super().__init__()
        self.self = _FlashAttention(config)
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.self(hidden_states, attention_mask)
        out = self.dropout(out)
        return self.output_norm(out + hidden_states)


class _BertMLP(nn.Module):
    def __init__(self, config: ScoreEstimatorConfig) -> None:
        super().__init__()
        self.input_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.output_dense(self.intermediate_act_fn(self.input_dense(hidden_states)))
        return self.dropout(out)


class _BertBlock(nn.Module):
    def __init__(self, config: ScoreEstimatorConfig) -> None:
        super().__init__()
        self.attention = _BertAttention(config)
        self.mlp = _BertMLP(config)
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attention(hidden_states, attention_mask)
        return self.output_norm(self.mlp(attn_out) + attn_out)


class _DiMAUNetEncoder(nn.Module):
    """Exact 12-layer U-Net transformer matching ``dima_src/src/models/score_estimator.py``."""

    def __init__(self, config: ScoreEstimatorConfig) -> None:
        super().__init__()
        self.config = config
        self.use_self_cond = config.use_self_cond
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        half = self.num_hidden_layers // 2
        self.input_blocks = nn.ModuleList([_BertBlock(config) for _ in range(half)])
        self.output_blocks = nn.ModuleList([_BertBlock(config) for _ in range(half)])
        self.time_layers = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_hidden_layers)]
        )
        if self.use_self_cond:
            self.self_cond_layers = nn.ModuleList(
                [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_hidden_layers)]
            )

    def forward(
        self,
        x: torch.Tensor,
        emb_t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        x_0_self_cond: torch.Tensor | None = None,
        return_mid_and_last: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_input_list: list[torch.Tensor] = []
        half = self.num_hidden_layers // 2

        for i, block in enumerate(self.input_blocks):
            x_input_list.append(x)
            x = x + self.time_layers[i](emb_t)
            if self.use_self_cond and x_0_self_cond is not None:
                x = x + self.self_cond_layers[i](x_0_self_cond)
            x = block(hidden_states=x, attention_mask=attention_mask)

        mid_hidden = x  # output of input_blocks[half - 1] (layer 6/12 U-Net bottleneck)

        for i, block in enumerate(self.output_blocks):
            ind = i + half
            x = x + x_input_list.pop()
            x = x + self.time_layers[ind](emb_t)
            if self.use_self_cond and x_0_self_cond is not None:
                x = x + self.self_cond_layers[ind](x_0_self_cond)
            x = block(hidden_states=x, attention_mask=attention_mask)

        if return_mid_and_last:
            return x, mid_hidden, x
        return x


class DiMAScoreEstimator(nn.Module):
    """Exact state-dict-compatible DiMA ``ScoreEstimator`` (strict EMA weight loading)."""

    def __init__(self, config: ScoreEstimatorConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self._max_position_embeddings = config.max_position_embeddings

        if self.embedding_size != self.hidden_size:
            self.input_projector_x_t = nn.Linear(self.embedding_size, self.hidden_size)
            self.output_projector_x_t = nn.Linear(self.hidden_size, self.embedding_size)
            self.projector_self_cond = nn.Linear(self.embedding_size, self.hidden_size)

        self.time_emb = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )
        self.encoder = _DiMAUNetEncoder(config)
        self.register_buffer("position_ids", torch.arange(self._max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(self._max_position_embeddings, self.hidden_size)

    def get_extended_attention_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        ext = attention_mask[:, None, None, :].to(dtype=dtype)
        return (1.0 - ext) * torch.finfo(dtype).min

    def forward(
        self,
        x_t: torch.Tensor,
        time_t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        x_0_self_cond: torch.Tensor | None = None,
        return_mid_and_last: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_length = int(x_t.size(1))
        if seq_length > self._max_position_embeddings:
            raise ValueError(
                f"DiMAScoreEstimator received length {seq_length} > max_position_embeddings={self._max_position_embeddings}"
            )
        t_emb = _timestep_embedding(time_t, self.hidden_size).to(dtype=x_t.dtype)
        hidden_t = self.time_emb(t_emb)[:, None, :]

        emb_x = x_t
        sc_h = x_0_self_cond
        if self.embedding_size != self.hidden_size:
            emb_x = self.input_projector_x_t(x_t)
            if sc_h is not None:
                sc_h = self.projector_self_cond(sc_h)

        pos_ids = self.position_ids[:, :seq_length]
        hidden_state = emb_x + self.position_embeddings(pos_ids).to(dtype=emb_x.dtype)

        ext_mask = (
            self.get_extended_attention_mask(attention_mask, hidden_state.dtype)
            if attention_mask is not None
            else None
        )
        enc_out = self.encoder(
            x=hidden_state,
            emb_t=hidden_t,
            attention_mask=ext_mask,
            x_0_self_cond=sc_h,
            return_mid_and_last=return_mid_and_last,
        )
        if return_mid_and_last:
            out_h, mid_h, last_h = enc_out  # type: ignore[misc]
            out = self.output_projector_x_t(out_h) if self.embedding_size != self.hidden_size else out_h
            return out, mid_h, last_h
        out = self.output_projector_x_t(enc_out) if self.embedding_size != self.hidden_size else enc_out  # type: ignore[arg-type]
        return out

    def load_dima_checkpoint(self, ckpt_path: Path | str, *, use_ema: bool = True) -> dict[str, Any]:
        """Strictly load ``model`` state_dict and copy ``ema['shadow_params']`` onto parameters."""
        ckpt = _omegaconf_safe_torch_load(ckpt_path, map_location="cpu")
        self.load_state_dict(ckpt["model"], strict=True)
        ema_applied = False
        if use_ema and "ema" in ckpt and "shadow_params" in ckpt["ema"]:
            shadow = ckpt["ema"]["shadow_params"]
            params = [p for p in self.parameters() if p.requires_grad]
            if len(shadow) != len(params):
                raise RuntimeError(
                    f"EMA shadow_params count ({len(shadow)}) != trainable params ({len(params)}) in {ckpt_path}"
                )
            with torch.no_grad():
                for sp, p in zip(shadow, params):
                    if tuple(sp.shape) != tuple(p.shape):
                        raise RuntimeError(f"EMA shape mismatch {tuple(sp.shape)} vs {tuple(p.shape)}")
                    p.data.copy_(sp.to(device=p.device, dtype=p.dtype))
            ema_applied = True
        return {"step": int(ckpt.get("step", -1)), "ema_applied": ema_applied}


class DiMATransformerDecoder(nn.Module):
    """Exact 3-layer Transformer latent-to-token decoder matching ``dima_src/src/encoders/transformer_decoder.py``."""

    def __init__(self, encoder_config: Any, num_hidden_layers: int = 3) -> None:
        super().__init__()
        base_cfg = AutoConfig.from_pretrained(encoder_config._name_or_path, local_files_only=True) if isinstance(encoder_config, str) else encoder_config
        base_cfg.num_hidden_layers = int(num_hidden_layers)
        self._max_position_embeddings = int(base_cfg.max_position_embeddings)
        self.register_buffer("position_ids", torch.arange(self._max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(self._max_position_embeddings, int(base_cfg.hidden_size))
        self.transformer = AutoModel.from_config(base_cfg).encoder
        self.fc = nn.Linear(int(base_cfg.hidden_size), int(base_cfg.vocab_size))

    def get_extended_attention_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        ext = attention_mask[:, None, None, :].to(dtype=dtype)
        return (1.0 - ext) * torch.finfo(dtype).min

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        ext_mask = self.get_extended_attention_mask(mask, x.dtype) if mask is not None else None
        seq_length = int(x.size(1))
        pos_ids = self.position_ids[:, :seq_length]
        x = x + self.position_embeddings(pos_ids).to(dtype=x.dtype)
        return self.fc(self.transformer(x, attention_mask=ext_mask).last_hidden_state)


class DiMAEncNormalizer(nn.Module):
    """Exact latent channel normalizer matching ``dima_src/src/encoders/enc_normalizer.py``."""

    def __init__(self, enc_mean: torch.Tensor, enc_std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("enc_mean", enc_mean.float().reshape(1, 1, -1))
        self.register_buffer("enc_std", enc_std.float().reshape(1, 1, -1))

    @classmethod
    def from_file(cls, stats_path: Path | str) -> "DiMAEncNormalizer":
        sd = _omegaconf_safe_torch_load(stats_path, map_location="cpu")
        return cls(sd["enc_mean"], sd["enc_std"])

    def normalize(self, encoding: torch.Tensor) -> torch.Tensor:
        mean = self.enc_mean.to(device=encoding.device, dtype=encoding.dtype)
        std = self.enc_std.to(device=encoding.device, dtype=encoding.dtype)
        safe_std = torch.where(std < 1e-5, torch.ones_like(std), std)
        return (encoding - mean) / safe_std

    def denormalize(self, pred_x_0: torch.Tensor) -> torch.Tensor:
        mean = self.enc_mean.to(device=pred_x_0.device, dtype=pred_x_0.dtype)
        std = self.enc_std.to(device=pred_x_0.device, dtype=pred_x_0.dtype)
        return pred_x_0 * std + mean


# --------------------------------------------------------------------------- #
# CHEAP_shorten_1_dim_1024 Pure-PyTorch Components (Zero openfold/lightning)
# --------------------------------------------------------------------------- #


class CHEAPLatentScaler(nn.Module):
    """Exact ``channel_minmaxnorm`` scaler into ``[-1, 1]`` from ``cheap/utils/_latent_scaler.py``."""

    def __init__(self, stats_dir: Path | str) -> None:
        super().__init__()
        sdir = Path(stats_dir)
        minv = np.load(sdir / "channel_min.pkl.npy").astype(np.float32)
        maxv = np.load(sdir / "channel_max.pkl.npy").astype(np.float32)
        self.register_buffer("minv", torch.from_numpy(minv).reshape(1, 1, -1))
        self.register_buffer("maxv", torch.from_numpy(maxv).reshape(1, 1, -1))

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        lo = self.minv.to(device=x.device, dtype=x.dtype)
        hi = self.maxv.to(device=x.device, dtype=x.dtype)
        return ((x - lo) / (hi - lo)) * 2.0 - 1.0

    def unscale(self, x_scaled: torch.Tensor) -> torch.Tensor:
        lo = self.minv.to(device=x_scaled.device, dtype=x_scaled.dtype)
        hi = self.maxv.to(device=x_scaled.device, dtype=x_scaled.dtype)
        return ((x_scaled + 1.0) * 0.5) * (hi - lo) + lo


class CHEAPSequenceDecoder(nn.Module):
    """Exact 3-layer MLP sequence decoder matching ``cheap/decoder.py:FullyConnectedNetwork``."""

    def __init__(self, mlp_hidden_dim: int = 1024, n_classes: int = 21, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(mlp_hidden_dim // 2, mlp_hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(mlp_hidden_dim // 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @classmethod
    def from_checkpoint(cls, ckpt_path: Path | str) -> "CHEAPSequenceDecoder":
        ckpt = _omegaconf_safe_torch_load(ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model = cls()
        model.load_state_dict(sd, strict=True)
        model.eval()
        model.requires_grad_(False)
        return model


class CHEAPESMFoldProjector(nn.Module):
    """Exact ``ESMFoldEmbed`` projection from ``ESM2-3B`` 37-layer states -> ``s_s_0`` ``(B, L, 1024)``."""

    def __init__(self, proj_ckpt_path: Path | str, esm_feats: int = 2560, num_layers: int = 36, c_s: int = 1024) -> None:
        super().__init__()
        self.c_s = c_s
        self.register_buffer("af2_to_esm", torch.zeros(22, dtype=torch.long))
        self.esm_s_combine = nn.Parameter(torch.zeros(num_layers + 1))
        self.esm_s_mlp = nn.Sequential(
            nn.LayerNorm(esm_feats),
            nn.Linear(esm_feats, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )
        self.embedding = nn.Embedding(23, c_s, padding_idx=0)
        sd = _omegaconf_safe_torch_load(proj_ckpt_path, map_location="cpu")
        keep = {
            k: v
            for k, v in sd.items()
            if k in ("af2_to_esm", "esm_s_combine", "embedding.weight") or k.startswith("esm_s_mlp.")
        }
        self.load_state_dict(keep, strict=True)
        self.eval()
        self.requires_grad_(False)

    def af2_idx_to_esm_idx(self, aa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        shifted = (aa + 1).masked_fill(mask != 1, 0)
        return self.af2_to_esm[shifted]

    def project(self, esm_hidden_states: Sequence[torch.Tensor], aa: torch.Tensor) -> torch.Tensor:
        """Combine 37-layer ``ESM2-3B`` states ``(B, L+2, 2560)`` -> ``s_s_0`` ``(B, L, 1024)``."""
        # Incremental weighted sum over all 37 layers (avoids 37x peak VRAM stack allocation).
        weights = self.esm_s_combine.softmax(0)
        target_dtype = self.esm_s_combine.dtype
        combined = esm_hidden_states[0][:, 1:-1, :].to(dtype=target_dtype) * weights[0]
        for layer_idx in range(1, len(esm_hidden_states)):
            combined = combined + esm_hidden_states[layer_idx][:, 1:-1, :].to(dtype=target_dtype) * weights[layer_idx]
        s_s_0 = self.esm_s_mlp(combined) + self.embedding(aa)
        return s_s_0


# --------------------------------------------------------------------------- #
# Official Tanh(d=10.0) VP-SDE & Euler Reverse Solver
# --------------------------------------------------------------------------- #


class TanhScheduler:
    """Exact ``Tanh(d=10.0)`` schedule matching ``dima_src/src/diffusion/schedulers.py``."""

    def __init__(self, d: float = DIMA_TANH_D, t_thr: float = DIMA_TANH_T_THR) -> None:
        self.d = float(d)
        self.t_thr = float(t_thr)

    def beta_t(self, t: torch.Tensor) -> torch.Tensor:
        tc = torch.clamp(t.float(), 0.0, self.t_thr)
        tan = torch.tan(math.pi * tc * 0.5)
        d2 = self.d * self.d
        return math.pi * d2 * tan * (1.0 + tan * tan) / (1.0 + d2 * tan * tan)

    def params(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t3 = t.float()[:, None, None]
        tan = torch.tan(math.pi * t3 * 0.5)
        alpha_t = torch.rsqrt(1.0 + (tan * tan) * (self.d * self.d))
        std_t = torch.sqrt(torch.clamp(1.0 - alpha_t * alpha_t, min=0.0))
        return torch.clamp(alpha_t, 0.0, 1.0), torch.clamp(std_t, 0.0, 1.0)


class DynamicSDE:
    """Exact VP-SDE matching ``dima_src/src/diffusion/dynamic.py``."""

    def __init__(self, scheduler: TanhScheduler, T: float = DIMA_DEFAULT_T_MAX) -> None:
        self.scheduler = scheduler
        self.T = float(T)

    def marginal_params(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scheduler.params(t)

    def reverse_params(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
        *,
        ode_sampling: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        beta_t = self.scheduler.beta_t(t).to(dtype=x_t.dtype)
        drift_sde = -0.5 * beta_t[:, None, None] * x_t
        if ode_sampling:
            drift = drift_sde - 0.5 * beta_t[:, None, None] * score
            diffusion = torch.zeros_like(beta_t)
        else:
            drift = drift_sde - beta_t[:, None, None] * score
            diffusion = torch.sqrt(beta_t)
        return drift, diffusion


# --------------------------------------------------------------------------- #
# Helper Utilities
# --------------------------------------------------------------------------- #


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_aa20_sequence(sequence: str) -> str:
    seq = sequence.strip().upper()
    bad = [ch for ch in seq if ch not in AA20_SET]
    if bad:
        raise ValueError(f"Non-AA20 characters in sequence: {sorted(set(bad))}")
    if not seq:
        raise ValueError("Empty sequence")
    return seq


def _centered_window(length: int, pos: int, max_len: int = DIMA_MAX_RESIDUES) -> tuple[int, int]:
    """Return ``(start, end)`` slice of length ``<= max_len`` centered on ``pos``."""
    if length <= max_len:
        return 0, length
    half = max_len // 2
    start = max(0, min(pos - half, length - max_len))
    return start, start + max_len


def _parse_mutation_token(token: str) -> tuple[str, int, str]:
    tok = token.strip().upper()
    if len(tok) < 3:
        raise ValueError(f"Invalid mutation token {token!r}")
    wt_aa, mut_aa = tok[0], tok[-1]
    pos_1based = int(tok[1:-1])
    if wt_aa not in AA20_SET or mut_aa not in AA20_SET or pos_1based < 1:
        raise ValueError(f"Invalid AA20 mutation token {token!r}")
    return wt_aa, pos_1based - 1, mut_aa


# --------------------------------------------------------------------------- #
# High-Level DiMA Adapter
# --------------------------------------------------------------------------- #


class DiMAAdapter:
    """Unified adapter over all 4 official DiMA continuous latent diffusion models.

    Supports:
    * ``dima-esm2-8m``
    * ``dima-esm2-3b``
    * ``dima-saprot-35m``
    * ``dima-cheap-1024``
    """

    def __init__(
        self,
        arm_id: str,
        *,
        dima_root: Path | str | None = None,
        model_root: Path | str | None = None,
        encoder_dir: Path | str | None = None,
        device: str = "cpu",
        dtype: str | None = None,
        use_ema: bool = True,
        load_encoder: bool = True,
        _skip_weights_init: bool = False,
    ) -> None:
        if arm_id not in DIMA_SPECS:
            raise KeyError(f"Unknown DiMA arm_id {arm_id!r}; expected one of {sorted(DIMA_SPECS)}")
        self.spec = DIMA_SPECS[arm_id]
        self.arm = arm_id
        self.device = str(device)
        self.use_ema = bool(use_ema)
        self.max_residues = DIMA_MAX_RESIDUES
        self.res_offset = 1 if self.spec.has_bos_eos else 0

        dt_raw = (dtype or self.spec.default_dtype).strip().lower()
        if dt_raw in ("fp32", "float32", "auto", "default"):
            self.dtype = "float32"
            self.torch_dtype = torch.float32
        elif dt_raw in ("bf16", "bfloat16"):
            self.dtype = "bfloat16"
            self.torch_dtype = torch.bfloat16
        else:
            raise ValueError(f"{arm_id}: unsupported dtype {dtype!r}; use float32 or bfloat16")

        self.scheduler = TanhScheduler(d=DIMA_TANH_D, t_thr=DIMA_TANH_T_THR)
        self.dynamic = DynamicSDE(self.scheduler, T=DIMA_DEFAULT_T_MAX)

        # Populated below (or by synthetic test fixtures when _skip_weights_init=True)
        self.dima_root: Path | None = None
        self.encoder_dir: Path | None = None
        self.diffusion_ckpt_path: Path | None = None
        self.ckpt_meta: dict[str, Any] = {}
        self.score_estimator: DiMAScoreEstimator
        self.enc_normalizer: DiMAEncNormalizer
        self.encoder: nn.Module | None = None
        self.tokenizer: Any = None
        self.transformer_decoder: DiMATransformerDecoder | None = None
        self.cheap_projector: CHEAPESMFoldProjector | None = None
        self.cheap_scaler: CHEAPLatentScaler | None = None
        self.cheap_decoder: CHEAPSequenceDecoder | None = None
        self._aa20_token_ids_t: torch.Tensor | None = None
        self._saprot_aa_group_ids: list[torch.Tensor] | None = None
        self._saprot_valid_420_ids_t: torch.Tensor | None = None
        self._saprot_valid_420_aa: list[str] | None = None
        self._saprot_valid_420_3di: list[str] | None = None
        self._openfold_to_aa20_t = torch.as_tensor(OPENFOLD_TO_AA20_INDICES, dtype=torch.long, device=self.device)

        if _skip_weights_init:
            return

        self.dima_root = resolve_dima_root(dima_root=dima_root, model_root=model_root)
        self.diffusion_ckpt_path = self.dima_root / self.spec.diffusion_relpath
        stats_path = self.dima_root / self.spec.stats_relpath

        se_cfg = ScoreEstimatorConfig(embedding_size=self.spec.embedding_dim)
        self.score_estimator = DiMAScoreEstimator(se_cfg)
        self.ckpt_meta = self.score_estimator.load_dima_checkpoint(self.diffusion_ckpt_path, use_ema=self.use_ema)
        self.score_estimator.to(device=self.device, dtype=self.torch_dtype).eval().requires_grad_(False)

        self.enc_normalizer = DiMAEncNormalizer.from_file(stats_path).to(device=self.device)

        if self.spec.encoder_kind == "cheap":
            cheap_root = self.dima_root / "cheap_assets"
            self.cheap_scaler = CHEAPLatentScaler(
                cheap_root / "statistics/cath/esmfold/subset_5000_nov28"
            ).to(device=self.device)
            self.cheap_decoder = CHEAPSequenceDecoder.from_checkpoint(
                cheap_root / "sequence_decoder/mlp.ckpt"
            ).to(device=self.device, dtype=self.torch_dtype)
            self.cheap_projector = CHEAPESMFoldProjector(
                cheap_root / "esmfold_v1_s_proj.pt"
            ).to(device=self.device, dtype=torch.float32)
            if load_encoder:
                self.encoder_dir = resolve_encoder_dir(
                    self.spec.encoder_dir_names, encoder_dir=encoder_dir, model_root=model_root
                )
                self.encoder = (
                    AutoModel.from_pretrained(
                        str(self.encoder_dir),
                        add_pooling_layer=False,
                        local_files_only=True,
                        dtype=self.torch_dtype,
                    )
                    .to(device=self.device)
                    .eval()
                    .requires_grad_(False)
                )
        else:
            self.encoder_dir = resolve_encoder_dir(
                self.spec.encoder_dir_names, encoder_dir=encoder_dir, model_root=model_root
            )
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.encoder_dir), local_files_only=True)
            enc_cfg = AutoConfig.from_pretrained(str(self.encoder_dir), local_files_only=True)
            dec_path = self.dima_root / str(self.spec.decoder_relpath)
            self.transformer_decoder = DiMATransformerDecoder(enc_cfg, num_hidden_layers=3)
            dec_ckpt = _omegaconf_safe_torch_load(dec_path, map_location="cpu")
            self.transformer_decoder.load_state_dict(dec_ckpt["decoder"], strict=True)
            self.transformer_decoder.to(device=self.device, dtype=self.torch_dtype).eval().requires_grad_(False)

            if load_encoder:
                self.encoder = (
                    AutoModel.from_pretrained(
                        str(self.encoder_dir),
                        add_pooling_layer=False,
                        local_files_only=True,
                        dtype=self.torch_dtype,
                    )
                    .to(device=self.device)
                    .eval()
                    .requires_grad_(False)
                )
            self._init_vocab_tables()

    def _init_vocab_tables(self) -> None:
        if self.spec.encoder_kind == "esm2":
            ids = [int(self.tokenizer.convert_tokens_to_ids(aa)) for aa in AA20]
            self._aa20_token_ids_t = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        elif self.spec.encoder_kind == "saprot":
            vocab = self.tokenizer.get_vocab()
            inv = {int(idx): tok for tok, idx in vocab.items()}
            groups: dict[str, list[int]] = {aa: [] for aa in AA20}
            valid_ids: list[int] = []
            valid_aa: list[str] = []
            valid_3di: list[str] = []
            for idx in sorted(inv):
                tok = inv[idx]
                if len(tok) == 2 and tok[0] in AA20_SET:
                    groups[tok[0]].append(idx)
                    valid_ids.append(idx)
                    valid_aa.append(tok[0])
                    valid_3di.append(tok[1])
            for aa in AA20:
                if not groups[aa]:
                    raise RuntimeError(f"SaProt vocab missing tokens for amino acid {aa!r}")
            self._saprot_aa_group_ids = [
                torch.as_tensor(groups[aa], dtype=torch.long, device=self.device) for aa in AA20
            ]
            self._saprot_valid_420_ids_t = torch.as_tensor(valid_ids, dtype=torch.long, device=self.device)
            self._saprot_valid_420_aa = valid_aa
            self._saprot_valid_420_3di = valid_3di

    def _precision_ctx(self):
        if self.dtype == "float32" and self.device.startswith("cuda"):
            return fp32_matmul_context(torch)
        return nullcontext()

    def provenance(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "family": "DiMA",
            "paradigm": "continuous_latent_diffusion",
            "encoder_kind": self.spec.encoder_kind,
            "embedding_dim": self.spec.embedding_dim,
            "hidden_size": int(self.score_estimator.hidden_size),
            "num_layers": int(self.score_estimator.config.num_hidden_layers),
            "max_position_embeddings": DIMA_MAX_POS_EMBEDDINGS,
            "max_residues": self.max_residues,
            "scheduler": f"Tanh(d={DIMA_TANH_D}, t_thr={DIMA_TANH_T_THR})",
            "t_max": DIMA_DEFAULT_T_MAX,
            "t_min": DIMA_DEFAULT_T_MIN,
            "use_ema": self.use_ema,
            "ckpt_step": self.ckpt_meta.get("step"),
            "dtype_requested": self.dtype,
            "dtype_realised": "/".join(sorted({str(p.dtype) for p in self.score_estimator.parameters()})),
            "fp32_matmul_policy": (
                requested_fp32_matmul_policy()
                if self.dtype == "float32"
                else {"declared_non_fp32_opt_in": self.dtype}
            ),
            "score_estimator_parameters": int(sum(p.numel() for p in self.score_estimator.parameters())),
            "device": self.device,
            "dima_root": str(self.dima_root) if self.dima_root is not None else None,
            "encoder_dir": str(self.encoder_dir) if self.encoder_dir is not None else None,
            "diffusion_ckpt_sha256": (
                _sha256_file(self.diffusion_ckpt_path)
                if self.diffusion_ckpt_path is not None and self.diffusion_ckpt_path.is_file()
                else None
            ),
        }

    # ----------------------------------------------------------------------- #
    # Frozen Encoder & Decoder Primitives
    # ----------------------------------------------------------------------- #

    def _format_saprot_seq(self, seq: str, mask_positions: set[int] | None = None) -> str:
        """Format AA sequence into SaProt sequence-only 2-char tokens (``"M#K#..."``) with ``<mask>``."""
        parts: list[str] = []
        for idx, ch in enumerate(seq):
            if mask_positions and idx in mask_positions:
                parts.append("<mask>")
            else:
                parts.append(f"{ch}#")
        return "".join(parts)

    @torch.no_grad()
    def encode_latents(
        self,
        sequences: Sequence[str],
        *,
        mask_positions_list: Sequence[Sequence[int] | set[int] | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode sequences (each ``len <= 510``) into normalized DiMA latents ``(B, T, D)`` & ``mask (B, T)``.

        For ``esm2`` and ``saprot``: ``T = max_L + 2`` (includes ``<cls>`` and ``<eos>``; residues at ``1..L``).
        For ``cheap``: ``T = max_L`` (no BOS/EOS; residues at ``0..L-1``).
        """
        if self.encoder is None:
            raise RuntimeError(f"{self.arm}: frozen encoder was not loaded (load_encoder=False)")
        seqs = [_validate_aa20_sequence(s) for s in sequences]
        if any(len(s) > self.max_residues for s in seqs):
            raise ValueError(f"encode_latents expects each sequence len <= {self.max_residues}")
        bsz = len(seqs)
        masks_list: list[set[int]] = [
            set(int(p) for p in mp) if (mask_positions_list is not None and mp is not None) else set()
            for mp in (mask_positions_list if mask_positions_list is not None else [None] * bsz)
        ]

        if self.spec.encoder_kind == "cheap":
            assert self.cheap_projector is not None and self.cheap_scaler is not None
            max_l = max(len(s) for s in seqs)
            aa_rows: list[list[int]] = []
            mask_rows: list[list[int]] = []
            for s, mp in zip(seqs, masks_list):
                pad_n = max_l - len(s)
                if mp:
                    row_aa = [OPENFOLD_UNK_IDX if j in mp else OPENFOLD_ORDER_WITH_X[ch] for j, ch in enumerate(s)]
                else:
                    row_aa = [OPENFOLD_ORDER_WITH_X[ch] for ch in s]
                if pad_n:
                    row_aa.extend([0] * pad_n)
                aa_rows.append(row_aa)
                mask_rows.append([1] * len(s) + [0] * pad_n)
            aa = torch.tensor(aa_rows, dtype=torch.long).to(device=self.device, non_blocking=True)
            res_mask = torch.tensor(mask_rows, dtype=torch.long).to(device=self.device, non_blocking=True)
            # Map OpenFold indices -> ESM2 token IDs with BOS=0, EOS=2, PAD=1
            esmaa = self.cheap_projector.af2_idx_to_esm_idx(aa, res_mask)
            bos = torch.zeros((bsz, 1), dtype=torch.long, device=self.device)
            pad_tail = torch.ones((bsz, 1), dtype=torch.long, device=self.device)
            esm_in = torch.cat([bos, esmaa, pad_tail], dim=1)
            lengths = res_mask.sum(dim=1)
            esm_in[torch.arange(bsz, device=self.device), lengths + 1] = 2  # <eos>
            esm_attn = esm_in.ne(1).long()
            out = self.encoder(input_ids=esm_in, attention_mask=esm_attn, output_hidden_states=True)
            s_s_0 = self.cheap_projector.project(out.hidden_states, aa)
            scaled = self.cheap_scaler.scale(s_s_0).to(dtype=self.torch_dtype)
            normed = self.enc_normalizer.normalize(scaled)
            normed = normed * res_mask[:, :, None].to(dtype=normed.dtype)
            return normed, res_mask

        if self.spec.encoder_kind == "saprot":
            formatted = [self._format_saprot_seq(s, mp) for s, mp in zip(seqs, masks_list)]
            tok_batch = self.tokenizer(
                formatted,
                return_attention_mask=True,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            input_ids = tok_batch["input_ids"].to(device=self.device, non_blocking=True)
            attn_mask = tok_batch["attention_mask"].to(device=self.device, non_blocking=True)
        else:
            tok_batch = self.tokenizer(
                seqs,
                return_attention_mask=True,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            input_ids_cpu = tok_batch["input_ids"]
            mask_id = int(self.tokenizer.mask_token_id)
            for i, mp in enumerate(masks_list):
                if mp:
                    input_ids_cpu[i, [1 + p for p in mp]] = mask_id
            input_ids = input_ids_cpu.to(device=self.device, non_blocking=True)
            attn_mask = tok_batch["attention_mask"].to(device=self.device, non_blocking=True)

        enc_h = self.encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
        normed = self.enc_normalizer.normalize(enc_h.to(dtype=self.torch_dtype))
        return normed, attn_mask

    def decode_raw_logits(
        self,
        latents: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Denormalize DiMA latents ``(B, T, D)`` and return raw decoder logits ``(B, T, V)``."""
        denorm = self.enc_normalizer.denormalize(latents)
        if self.spec.encoder_kind == "cheap":
            assert self.cheap_scaler is not None and self.cheap_decoder is not None
            s_s_0 = self.cheap_scaler.unscale(denorm).to(dtype=self.torch_dtype)
            return self.cheap_decoder(s_s_0)
        assert self.transformer_decoder is not None
        return self.transformer_decoder(denorm.to(dtype=self.torch_dtype), mask=attention_mask)

    def decode_aa20_logprobs(
        self,
        latents: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return canonical alphabetical ``AA20`` log-probabilities ``(B, T, 20)`` in float32."""
        raw = self.decode_raw_logits(latents, attention_mask=attention_mask).float()
        if self.spec.encoder_kind == "esm2":
            assert self._aa20_token_ids_t is not None
            return F.log_softmax(raw.index_select(-1, self._aa20_token_ids_t), dim=-1)
        if self.spec.encoder_kind == "cheap":
            # Drop index 20 ('X') and permute OpenFold 20 -> canonical AA20
            return F.log_softmax(raw.index_select(-1, self._openfold_to_aa20_t), dim=-1)
        # SaProt: exact marginalization over 21 3Di variants per AA over the 420 valid AA+3Di tokens
        assert self._saprot_valid_420_ids_t is not None and self._saprot_aa_group_ids is not None
        # Compute log-normalizer across all 420 valid AA tokens so sum_aa exp(lp20) == 1.0 exactly
        lse_420 = torch.logsumexp(raw.index_select(-1, self._saprot_valid_420_ids_t), dim=-1, keepdim=True)
        aa_lse = torch.stack(
            [torch.logsumexp(raw.index_select(-1, g_ids), dim=-1) for g_ids in self._saprot_aa_group_ids],
            dim=-1,
        )
        return aa_lse - lse_420

    def decode_sequences_from_latents(
        self,
        latents: torch.Tensor,
        lengths: Sequence[int],
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[list[str], list[str | None], list[float], list[float]]:
        """Decode ``(B, T, D)`` latents into guaranteed valid AA20 sequences (and SaProt 3Di if applicable)."""
        raw = self.decode_raw_logits(latents, attention_mask=attention_mask).float()
        lp20 = self.decode_aa20_logprobs(latents, attention_mask=attention_mask)
        probs20 = lp20.exp()
        ent20 = -(probs20 * lp20).sum(dim=-1)
        maxp20 = probs20.max(dim=-1).values

        seqs: list[str] = []
        threedis: list[str | None] = []
        mean_maxps: list[float] = []
        mean_ents: list[float] = []

        off = self.res_offset
        for i, L in enumerate(lengths):
            sl = slice(off, off + L)
            if self.spec.encoder_kind == "saprot":
                assert (
                    self._saprot_valid_420_ids_t is not None
                    and self._saprot_valid_420_aa is not None
                    and self._saprot_valid_420_3di is not None
                )
                sub = raw[i, sl, :].index_select(-1, self._saprot_valid_420_ids_t)
                top_idx = sub.argmax(dim=-1).cpu().tolist()
                seqs.append("".join(self._saprot_valid_420_aa[k] for k in top_idx))
                threedis.append("".join(self._saprot_valid_420_3di[k] for k in top_idx))
            else:
                aa_idx = lp20[i, sl, :].argmax(dim=-1).cpu().tolist()
                seqs.append("".join(AA20[k] for k in aa_idx))
                threedis.append(None)
            mean_maxps.append(float(maxp20[i, sl].mean().item()))
            mean_ents.append(float(ent20[i, sl].mean().item()))
        return seqs, threedis, mean_maxps, mean_ents

    def denoise_self_cond(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_mid_and_last: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """2-pass self-conditioned ``ScoreEstimator`` denoising matching ``base_trainer.py:217-237``."""
        zeros = torch.zeros_like(x_t)
        x_0_sc = self.score_estimator(
            x_t=x_t, time_t=t, attention_mask=attention_mask, x_0_self_cond=zeros
        )
        return self.score_estimator(
            x_t=x_t,
            time_t=t,
            attention_mask=attention_mask,
            x_0_self_cond=x_0_sc.detach(),
            return_mid_and_last=return_mid_and_last,
        )

    def _calc_score(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None,
        x_0_self_cond: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mu, std = self.dynamic.marginal_params(t)
        mu = mu.to(device=x_t.device, dtype=x_t.dtype)
        std = std.to(device=x_t.device, dtype=x_t.dtype).clamp(min=1e-5)
        x_0 = self.score_estimator(
            x_t=x_t, time_t=t, attention_mask=attention_mask, x_0_self_cond=x_0_self_cond
        )
        eps_theta = (x_t - mu * x_0) / std
        score = -eps_theta / std
        return {"score": score, "x_0": x_0, "eps_theta": eps_theta}

    # ----------------------------------------------------------------------- #
    # Leakage-Free Denoising Scoring Suite (R1, R2, R3, E01, R4, R5-DMS, 4.3)
    # ----------------------------------------------------------------------- #

    def _masked_denoising_logprobs_window(
        self,
        window_seq: str,
        mask_sets: Sequence[set[int]],
        *,
        t_eval: float = DIMA_DEFAULT_T_EVAL,
        batch_size: int = 16,
    ) -> torch.Tensor:
        """Run leakage-free masked denoising on one ``<=510``-aa window for multiple mask sets.

        At every masked position ``p`` in each batch item:
        1. Input residue ``p`` is replaced with ``<mask>`` (ESM2/SaProt) or ``'X'`` (CHEAP) BEFORE encoding.
        2. Latent ``x_t`` is set to ``mu(t_eval) * z_masked`` on unmasked context and STRICTLY ``0.0`` on masked sites.
        3. 2-pass self-conditioned ``ScoreEstimator`` predicts ``x_0`` -> ``Decoder`` -> canonical AA20 log-probs.
        Returns ``(len(mask_sets), len(window_seq), 20)`` float32 tensor on CPU.
        """
        n_items = len(mask_sets)
        L = len(window_seq)
        out = torch.empty((n_items, L, 20), dtype=torch.float32)
        bs = max(1, int(batch_size))
        off = self.res_offset

        with self._precision_ctx():
            for start in range(0, n_items, bs):
                stop = min(start + bs, n_items)
                chunk_masks = mask_sets[start:stop]
                cur = len(chunk_masks)
                z_masked, attn_mask = self.encode_latents(
                    [window_seq] * cur, mask_positions_list=chunk_masks
                )
                t_vec = torch.full((cur,), float(t_eval), device=self.device, dtype=torch.float32)
                mu, _ = self.dynamic.marginal_params(t_vec)
                x_t = z_masked * mu.to(device=self.device, dtype=z_masked.dtype)
                zero_rows: list[int] = []
                zero_cols: list[int] = []
                for b_i, mp in enumerate(chunk_masks):
                    for p in mp:
                        zero_rows.append(b_i)
                        zero_cols.append(off + p)
                if zero_rows:
                    row_t = torch.tensor(zero_rows, dtype=torch.long).to(device=self.device, non_blocking=True)
                    col_t = torch.tensor(zero_cols, dtype=torch.long).to(device=self.device, non_blocking=True)
                    x_t[row_t, col_t, :] = 0.0
                x_0_hat = self.denoise_self_cond(x_t, t_vec, attn_mask)
                lp20 = self.decode_aa20_logprobs(x_0_hat, attention_mask=attn_mask)  # type: ignore[arg-type]
                out[start:stop] = lp20[:, off : off + L, :].cpu()
        return out

    @torch.no_grad()
    def site_conditional_logprobs(
        self,
        sequence: str,
        *,
        positions: Sequence[int] | None = None,
        t_eval: float = DIMA_DEFAULT_T_EVAL,
        batch_size: int = 16,
    ) -> np.ndarray:
        """Exact leakage-free leave-one-out denoising AA20 log-probabilities ``(len(positions), 20)``."""
        seq = _validate_aa20_sequence(sequence)
        length = len(seq)
        pos_list = list(range(length)) if positions is None else [int(p) for p in positions]
        if any(not (0 <= p < length) for p in pos_list):
            raise ValueError(f"Position out of range for sequence length {length}")
        if not pos_list:
            return np.zeros((0, 20), dtype=np.float32)

        out = np.empty((len(pos_list), 20), dtype=np.float32)
        if length <= self.max_residues:
            lp = self._masked_denoising_logprobs_window(
                seq, [{p} for p in pos_list], t_eval=t_eval, batch_size=batch_size
            )
            for idx, p in enumerate(pos_list):
                out[idx] = lp[idx, p].numpy()
            return out

        # Group positions sharing the exact same centered 510-aa window to batch efficiently
        win_groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for out_idx, p in enumerate(pos_list):
            win = _centered_window(length, p, self.max_residues)
            win_groups.setdefault(win, []).append((out_idx, p - win[0]))
        for (ws, we), entries in win_groups.items():
            sub_seq = seq[ws:we]
            lp = self._masked_denoising_logprobs_window(
                sub_seq, [{local_p} for _, local_p in entries], t_eval=t_eval, batch_size=batch_size
            )
            for row_i, (out_idx, local_p) in enumerate(entries):
                out[out_idx] = lp[row_i, local_p].numpy()
        return out

    def native_site_conditional(
        self,
        sequence: str,
        *,
        positions: Sequence[int] | None = None,
        t_eval: float = DIMA_DEFAULT_T_EVAL,
        batch_size: int = 16,
    ) -> dict[str, Any]:
        seq = _validate_aa20_sequence(sequence)
        pos_list = list(range(len(seq))) if positions is None else [int(p) for p in positions]
        lp = self.site_conditional_logprobs(seq, positions=pos_list, t_eval=t_eval, batch_size=batch_size)
        return {
            "logprobs": lp,
            "positions": np.asarray(pos_list, dtype=np.int64),
            "alphabet": AA20,
            "arm": self.arm,
            "t_eval": float(t_eval),
        }

    def exact_pll(
        self,
        sequence: str,
        *,
        t_eval: float = DIMA_DEFAULT_T_EVAL,
        batch_size: int = 16,
    ) -> PositionVector:
        """Exact leave-one-out site-masked denoising pseudo-log-likelihood."""
        seq = _validate_aa20_sequence(sequence)
        lp = self.site_conditional_logprobs(seq, t_eval=t_eval, batch_size=batch_size)
        terms = np.empty(len(seq), dtype=np.float32)
        for p, ch in enumerate(seq):
            terms[p] = max(0.0, float(-lp[p, AA20_INDEX[ch]]))
        nll_sum = float(terms.sum(dtype=np.float64))
        return PositionVector(
            terms=terms,
            nll_sum=nll_sum,
            pll=-nll_sum,
            paradigm="continuous_latent_diffusion",
        )

    @torch.no_grad()
    def pll_8fold(
        self,
        sequence: str,
        *,
        n_folds: int = 8,
        t_eval: float = DIMA_DEFAULT_T_EVAL,
    ) -> PositionVector:
        """8-fold stride-masked denoising PLL (1 batched pass per <=510-aa tile)."""
        seq = _validate_aa20_sequence(sequence)
        length = len(seq)
        terms = np.empty(length, dtype=np.float32)

        for tile_start in range(0, length, self.max_residues):
            tile_end = min(tile_start + self.max_residues, length)
            sub = seq[tile_start:tile_end]
            sub_len = len(sub)
            k = min(int(n_folds), sub_len)
            fold_pos = [list(range(f, sub_len, k)) for f in range(k)]
            lp = self._masked_denoising_logprobs_window(
                sub, [set(pf) for pf in fold_pos], t_eval=t_eval, batch_size=k
            )
            for f, pf in enumerate(fold_pos):
                for p in pf:
                    terms[tile_start + p] = max(0.0, float(-lp[f, p, AA20_INDEX[sub[p]]]))

        nll_sum = float(terms.sum(dtype=np.float64))
        return PositionVector(
            terms=terms,
            nll_sum=nll_sum,
            pll=-nll_sum,
            paradigm="continuous_latent_diffusion",
        )

    @torch.no_grad()
    def latent_denoising_mse(
        self,
        sequence: str,
        *,
        t_eval: float = DIMA_DEFAULT_T_EVAL,
        seed: int = 20261125,
    ) -> float:
        """Deterministic latent denoising MSE ``||x_0_hat - z_0||^2`` diagnostic for ``Exp 4.3``."""
        seq = _validate_aa20_sequence(sequence)
        length = len(seq)
        off = self.res_offset
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        sq_err_sum = 0.0
        n_elem = 0

        with self._precision_ctx():
            for tile_start in range(0, length, self.max_residues):
                sub = seq[tile_start : min(tile_start + self.max_residues, length)]
                z_0, attn_mask = self.encode_latents([sub])
                t_vec = torch.full((1,), float(t_eval), device=self.device, dtype=torch.float32)
                mu, std = self.dynamic.marginal_params(t_vec)
                noise = torch.randn(z_0.shape, generator=gen, device=self.device, dtype=z_0.dtype)
                x_t = z_0 * mu.to(device=self.device, dtype=z_0.dtype) + noise * std.to(device=self.device, dtype=z_0.dtype)
                x_0_hat = self.denoise_self_cond(x_t, t_vec, attn_mask)
                diff = (x_0_hat[:, off : off + len(sub), :].float() - z_0[:, off : off + len(sub), :].float())  # type: ignore[index]
                sq_err_sum += float(diff.pow(2).sum().item())
                n_elem += int(diff.numel())
        return sq_err_sum / max(n_elem, 1)

    @torch.no_grad()
    def score_mutations(
        self,
        wt_sequence: str,
        mutations: Sequence[str],
        *,
        t_eval: float = DIMA_DEFAULT_T_EVAL,
        batch_size: int = 16,
        multi_exact_pll: bool = False,
    ) -> np.ndarray:
        """Score single- and multi-site AA20 mutations via leakage-free masked denoising.

        * Single mutations: exact masked-marginal LLR ``lp(mut) - lp(wt)`` (batched across unique sites).
        * Multi-site mutations: joint simultaneous site-masking LLR (or full ``exact_pll(mut) - exact_pll(wt)``
          when ``multi_exact_pll=True``).
        """
        wt = _validate_aa20_sequence(wt_sequence)
        length = len(wt)
        parsed: list[list[tuple[str, int, str]]] = []
        single_sites: set[int] = set()
        for mut_str in mutations:
            toks = [t for t in str(mut_str).strip().split(":") if t]
            items = [_parse_mutation_token(t) for t in toks]
            for wt_aa, pos, _ in items:
                if not (0 <= pos < length):
                    raise ValueError(f"Mutation {mut_str!r} position out of range for length {length}")
                if wt[pos] != wt_aa:
                    raise ValueError(f"Mutation {mut_str!r} WT mismatch: expected {wt[pos]} at 1-based pos {pos + 1}")
            parsed.append(items)
            if len(items) == 1:
                single_sites.add(items[0][1])

        single_lp_map: dict[int, np.ndarray] = {}
        if single_sites:
            ordered = sorted(single_sites)
            lp_mat = self.site_conditional_logprobs(wt, positions=ordered, t_eval=t_eval, batch_size=batch_size)
            for idx, pos in enumerate(ordered):
                single_lp_map[pos] = lp_mat[idx]

        wt_pll_val: float | None = None
        if multi_exact_pll and any(len(items) > 1 for items in parsed):
            wt_pll_val = self.exact_pll(wt, t_eval=t_eval, batch_size=batch_size).pll

        scores = np.empty(len(parsed), dtype=np.float64)
        multi_requests: list[tuple[int, list[tuple[str, int, str]]]] = []
        for idx, items in enumerate(parsed):
            if not items:
                scores[idx] = 0.0
            elif len(items) == 1:
                wt_aa, pos, mut_aa = items[0]
                lp = single_lp_map[pos]
                scores[idx] = float(lp[AA20_INDEX[mut_aa]] - lp[AA20_INDEX[wt_aa]])
            elif multi_exact_pll:
                mut_chars = list(wt)
                for _, pos, mut_aa in items:
                    mut_chars[pos] = mut_aa
                assert wt_pll_val is not None
                scores[idx] = self.exact_pll("".join(mut_chars), t_eval=t_eval, batch_size=batch_size).pll - wt_pll_val
            else:
                multi_requests.append((idx, items))

        # Batch multi-site joint-masked scoring cleanly (grouping identical windows for L > 510)
        if multi_requests:
            if length <= self.max_residues:
                mask_sets = [{pos for _, pos, _ in items} for _, items in multi_requests]
                lp_all = self._masked_denoising_logprobs_window(
                    wt, mask_sets, t_eval=t_eval, batch_size=batch_size
                )
                for row_i, (out_idx, items) in enumerate(multi_requests):
                    total = 0.0
                    for wt_aa, pos, mut_aa in items:
                        total += float(lp_all[row_i, pos, AA20_INDEX[mut_aa]] - lp_all[row_i, pos, AA20_INDEX[wt_aa]])
                    scores[out_idx] = total
            else:
                for out_idx, items in multi_requests:
                    # Center window on midpoint of mutated sites; fallback to per-site windows if span > 510
                    min_p = min(pos for _, pos, _ in items)
                    max_p = max(pos for _, pos, _ in items)
                    if max_p - min_p < self.max_residues:
                        ws, we = _centered_window(length, (min_p + max_p) // 2, self.max_residues)
                        ws = min(ws, min_p)
                        we = ws + self.max_residues
                        local_set = {pos - ws for _, pos, _ in items}
                        lp_win = self._masked_denoising_logprobs_window(
                            wt[ws:we], [local_set], t_eval=t_eval, batch_size=1
                        )[0]
                        scores[out_idx] = sum(
                            float(lp_win[pos - ws, AA20_INDEX[mut_aa]] - lp_win[pos - ws, AA20_INDEX[wt_aa]])
                            for wt_aa, pos, mut_aa in items
                        )
                    else:
                        total = 0.0
                        for wt_aa, pos, mut_aa in items:
                            lp_p = self.site_conditional_logprobs(wt, positions=[pos], t_eval=t_eval, batch_size=1)[0]
                            total += float(lp_p[AA20_INDEX[mut_aa]] - lp_p[AA20_INDEX[wt_aa]])
                        scores[out_idx] = total
        return scores

    # Alias matching DPLM2Adapter / ProteinMPNNAdapter
    score_variants = score_mutations

    @torch.no_grad()
    def extract_hidden_states(
        self,
        sequences: Sequence[str],
        *,
        t_readout: float = DIMA_DEFAULT_T_MIN,
        batch_size: int = 8,
        return_projected_1024: bool = False,
    ) -> np.ndarray:
        """Extract U-Net bottleneck (``input_blocks[5]``, layer 6/12) and output (``output_blocks[5]``, layer 12/12).

        At deterministic ``t = t_min = 0.05`` (``eps = 0``, ``x_t = mu(t) * z_0``, 2-pass self-cond),
        pools ``[middle_mean, middle_last, final_mean, final_last]`` over residue tokens ->
        ``(N, 4, 320)`` float32 (or ``(N, 1024)`` float32 via deterministic Gaussian JL projection
        ``seed=20260923``, ``320 -> 256`` per block, matching ``project_four_blocks``).
        """
        if not sequences:
            return (
                np.zeros((0, 1024), dtype=np.float32)
                if return_projected_1024
                else np.zeros((0, 4, int(self.score_estimator.hidden_size)), dtype=np.float32)
            )
        seqs = [_validate_aa20_sequence(s) for s in sequences]
        n_seq = len(seqs)
        h_dim = int(self.score_estimator.hidden_size)
        out = np.empty((n_seq, 4, h_dim), dtype=np.float32)
        off = self.res_offset

        short_indices = [i for i, s in enumerate(seqs) if len(s) <= self.max_residues]
        long_indices = [i for i, s in enumerate(seqs) if len(s) > self.max_residues]
        bs = max(1, int(batch_size))

        with self._precision_ctx():
            for start in range(0, len(short_indices), bs):
                idx_chunk = short_indices[start : start + bs]
                chunk_seqs = [seqs[i] for i in idx_chunk]
                z_0, attn_mask = self.encode_latents(chunk_seqs)
                t_vec = torch.full((len(chunk_seqs),), float(t_readout), device=self.device, dtype=torch.float32)
                mu, _ = self.dynamic.marginal_params(t_vec)
                x_t = z_0 * mu.to(device=self.device, dtype=z_0.dtype)
                _, h_mid, h_fin = self.denoise_self_cond(x_t, t_vec, attn_mask, return_mid_and_last=True)  # type: ignore[misc]
                h_mid_f = h_mid.float()
                h_fin_f = h_fin.float()
                for b_i, orig_i in enumerate(idx_chunk):
                    L = len(chunk_seqs[b_i])
                    mid_res = h_mid_f[b_i, off : off + L, :]
                    fin_res = h_fin_f[b_i, off : off + L, :]
                    out[orig_i] = torch.stack(
                        [mid_res.mean(dim=0), mid_res[-1], fin_res.mean(dim=0), fin_res[-1]], dim=0
                    ).cpu().numpy()

            # Tiled pooling for sequences > 510 aa
            for orig_i in long_indices:
                seq = seqs[orig_i]
                tiles = [seq[s : s + self.max_residues] for s in range(0, len(seq), self.max_residues)]
                mid_sum = torch.zeros(h_dim, device=self.device, dtype=torch.float32)
                fin_sum = torch.zeros(h_dim, device=self.device, dtype=torch.float32)
                mid_last = torch.zeros(h_dim, device=self.device, dtype=torch.float32)
                fin_last = torch.zeros(h_dim, device=self.device, dtype=torch.float32)
                total_res = 0
                for tile in tiles:
                    z_0, attn_mask = self.encode_latents([tile])
                    t_vec = torch.full((1,), float(t_readout), device=self.device, dtype=torch.float32)
                    mu, _ = self.dynamic.marginal_params(t_vec)
                    x_t = z_0 * mu.to(device=self.device, dtype=z_0.dtype)
                    _, h_mid, h_fin = self.denoise_self_cond(x_t, t_vec, attn_mask, return_mid_and_last=True)  # type: ignore[misc]
                    L_t = len(tile)
                    mid_res = h_mid[0, off : off + L_t, :].float()
                    fin_res = h_fin[0, off : off + L_t, :].float()
                    mid_sum += mid_res.sum(dim=0)
                    fin_sum += fin_res.sum(dim=0)
                    mid_last = mid_res[-1]
                    fin_last = fin_res[-1]
                    total_res += L_t
                out[orig_i] = torch.stack(
                    [mid_sum / total_res, mid_last, fin_sum / total_res, fin_last], dim=0
                ).cpu().numpy()

        if return_projected_1024:
            return project_four_blocks(out)
        return out

    def extract_readout_1024(
        self,
        sequences: Sequence[str],
        *,
        t_readout: float = DIMA_DEFAULT_T_MIN,
        batch_size: int = 8,
    ) -> np.ndarray:
        """Return deterministic ``(N, 1024)`` float32 readout (``0:512`` = ``R_mid``, ``512:1024`` = ``R_last``)."""
        return self.extract_hidden_states(
            sequences, t_readout=t_readout, batch_size=batch_size, return_projected_1024=True
        )

    @torch.no_grad()
    def extract_contact_matrix(
        self,
        sequence: str,
        *,
        t_eval: float = DIMA_DEFAULT_T_EVAL,
        batch_size: int = 16,
        apc: bool = True,
    ) -> dict[str, np.ndarray]:
        """E01 perturbation-propagation contact sensitivity ``J_{ij}`` (mask-i -> shift at j)."""
        seq = _validate_aa20_sequence(sequence)
        length = len(seq)
        raw = np.zeros((length, length), dtype=np.float64)

        if length <= self.max_residues:
            base_lp = self._masked_denoising_logprobs_window(seq, [set()], t_eval=t_eval, batch_size=1)[0]  # (L, 20)
            pert_lp = self._masked_denoising_logprobs_window(
                seq, [{i} for i in range(length)], t_eval=t_eval, batch_size=batch_size
            )  # (L, L, 20)
            diff = torch.linalg.norm(pert_lp - base_lp.unsqueeze(0), dim=-1)  # (L, L)
            raw[:, :] = diff.numpy()
        else:
            win_groups: dict[tuple[int, int], list[int]] = {}
            for i in range(length):
                win = _centered_window(length, i, self.max_residues)
                win_groups.setdefault(win, []).append(i)
            for (ws, we), sites in win_groups.items():
                sub = seq[ws:we]
                base_lp = self._masked_denoising_logprobs_window(sub, [set()], t_eval=t_eval, batch_size=1)[0]
                pert_lp = self._masked_denoising_logprobs_window(
                    sub, [{i - ws} for i in sites], t_eval=t_eval, batch_size=batch_size
                )
                diff = torch.linalg.norm(pert_lp - base_lp.unsqueeze(0), dim=-1).numpy()
                for row_k, orig_i in enumerate(sites):
                    raw[orig_i, ws:we] = diff[row_k]

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
        budget_tokens: int = DIMA_MAX_RESIDUES,
        n_folds: int = 8,
        t_eval: float = DIMA_DEFAULT_T_EVAL,
    ) -> dict[str, Any]:
        """Score ``target`` via 8-fold stride-masked denoising PLL conditioned on ``context_items`` (``<=510`` aa)."""
        tgt = _validate_aa20_sequence(target)
        eff_budget = min(int(budget_tokens), self.max_residues)

        if len(tgt) > eff_budget:
            # Target alone exceeds 510-aa positional limit; score target via tiled pll_8fold with 0 context
            res = self.pll_8fold(tgt, n_folds=n_folds, t_eval=t_eval)
            return {
                "arm": self.arm,
                "n_target_residues": len(tgt),
                "n_context_residues": 0,
                "n_context_items_used": 0,
                "budget_tokens": eff_budget,
                "nll_sum": res.nll_sum,
                "pll": res.pll,
                "nll_per_residue": res.nll_sum / max(len(tgt), 1),
                "terms": res.terms,
            }

        max_ctx = max(0, eff_budget - len(tgt))
        ctx_used: list[str] = []
        used_len = 0
        for raw_c in context_items:
            c_seq = _validate_aa20_sequence(raw_c)
            if used_len + len(c_seq) > max_ctx:
                rem = max_ctx - used_len
                if rem > 0:
                    ctx_used.append(c_seq[:rem])
                    used_len += rem
                break
            ctx_used.append(c_seq)
            used_len += len(c_seq)

        full_seq = "".join(ctx_used) + tgt
        tgt_start = len(full_seq) - len(tgt)
        k = min(int(n_folds), len(tgt))
        fold_pos = [list(range(f, len(tgt), k)) for f in range(k)]
        mask_sets = [{tgt_start + p for p in pf} for pf in fold_pos]
        lp = self._masked_denoising_logprobs_window(full_seq, mask_sets, t_eval=t_eval, batch_size=k)
        terms = np.empty(len(tgt), dtype=np.float32)
        for f, pf in enumerate(fold_pos):
            for p in pf:
                terms[p] = max(0.0, float(-lp[f, tgt_start + p, AA20_INDEX[tgt[p]]]))
        nll_sum = float(terms.sum(dtype=np.float64))
        return {
            "arm": self.arm,
            "n_target_residues": len(tgt),
            "n_context_residues": tgt_start,
            "n_context_items_used": len(ctx_used),
            "budget_tokens": eff_budget,
            "nll_sum": nll_sum,
            "pll": -nll_sum,
            "nll_per_residue": nll_sum / max(len(tgt), 1),
            "terms": terms,
        }

    # ----------------------------------------------------------------------- #
    # Continuous Latent Diffusion Generation Suite (Exp 4.1, 4.2, R5a, R4-Cont)
    # ----------------------------------------------------------------------- #

    def _run_reverse_diffusion(
        self,
        lengths: Sequence[int],
        *,
        steps: int = 250,
        ode_sampling: bool = False,
        seed: int = 20261125,
        t_min: float = DIMA_DEFAULT_T_MIN,
        z_ctx: torch.Tensor | None = None,
        ctx_mask: torch.Tensor | None = None,
        profile_log_bias: np.ndarray | torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Run official Euler reverse SDE/ODE solver with optional RePaint clamping and decoder guidance.

        * ``z_ctx`` ``(B, T, D)`` & ``ctx_mask`` ``(B, T)`` bool: positions with ``ctx_mask=True`` are clamped
          at every reverse step to ``mu(next_t) * z_ctx + std(next_t) * eps`` (RePaint continuous conditioning).
        * ``profile_log_bias`` ``(L, 20)`` or ``(B, L, 20)``: differentiable Potts/profile log-odds guidance on
          ``x_0_hat`` via the latent decoder (exact ``mean_applied_bias_norm`` returned; ``0.0`` when unguided).
        """
        bsz = len(lengths)
        max_l = max(int(L) for L in lengths)
        if max_l > self.max_residues:
            raise ValueError(f"Generation length {max_l} exceeds DiMA max_residues={self.max_residues}")
        total_t = max_l + (2 if self.spec.has_bos_eos else 0)
        dim = self.spec.embedding_dim

        attn_mask = torch.zeros((bsz, total_t), device=self.device, dtype=torch.long)
        for i, L in enumerate(lengths):
            attn_mask[i, : int(L) + (2 if self.spec.has_bos_eos else 0)] = 1

        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        x = torch.randn((bsz, total_t, dim), generator=gen, device=self.device, dtype=self.torch_dtype)
        x_0_sc = torch.zeros_like(x)

        bias_t: torch.Tensor | None = None
        if profile_log_bias is not None and float(guidance_scale) != 0.0:
            bt = torch.as_tensor(profile_log_bias, device=self.device, dtype=torch.float32)
            if bt.ndim == 2:
                bt = bt.unsqueeze(0).expand(bsz, -1, -1)
            if bt.shape != (bsz, max_l, 20):
                raise ValueError(f"Expected profile_log_bias shape ({bsz}, {max_l}, 20), got {tuple(bt.shape)}")
            bias_t = bt

        if z_ctx is not None and ctx_mask is not None:
            t0 = torch.full((bsz,), self.dynamic.T, device=self.device, dtype=torch.float32)
            mu0, std0 = self.dynamic.marginal_params(t0)
            eps0 = torch.randn(x.shape, generator=gen, device=self.device, dtype=x.dtype)
            x_ctx0 = z_ctx * mu0.to(dtype=x.dtype) + eps0 * std0.to(dtype=x.dtype)
            x = torch.where(ctx_mask.unsqueeze(-1), x_ctx0, x)

        n_steps = max(1, int(steps))
        timesteps = torch.linspace(self.dynamic.T, float(t_min), n_steps + 1, device=self.device)
        bias_norm_acc = 0.0
        bias_norm_steps = 0
        off = self.res_offset
        x_mean = x

        with self._precision_ctx():
            for idx in range(n_steps):
                t_cur = torch.full((bsz,), float(timesteps[idx].item()), device=self.device, dtype=torch.float32)
                t_nxt = torch.full((bsz,), float(timesteps[idx + 1].item()), device=self.device, dtype=torch.float32)
                dt = (t_nxt - t_cur).view(-1, 1, 1).to(dtype=x.dtype)

                with torch.no_grad():
                    score_out = self._calc_score(x, t_cur, attn_mask, x_0_sc)
                    score = score_out["score"]
                    x_0_hat = score_out["x_0"]

                # Optional differentiable Potts/profile log-odds guidance on x_0_hat via decoder
                if bias_t is not None:
                    with torch.enable_grad():
                        x0_g = x_0_hat.detach().float().requires_grad_(True)
                        lp20 = self.decode_aa20_logprobs(x0_g.to(dtype=self.torch_dtype), attention_mask=attn_mask)
                        lp_res = lp20[:, off : off + max_l, :]
                        # Exact log partition-function increment under AA20 log-odds tilt
                        pot = torch.logsumexp(lp_res + bias_t, dim=-1).sum()
                        grad_x0 = torch.autograd.grad(pot, x0_g)[0]
                    # Zero gradient on padded / clamped context slots
                    valid_res = torch.zeros((bsz, total_t), device=self.device, dtype=torch.bool)
                    for b_i, L in enumerate(lengths):
                        valid_res[b_i, off : off + int(L)] = True
                    if ctx_mask is not None:
                        valid_res = valid_res & (~ctx_mask)
                    grad_x0 = grad_x0 * valid_res.unsqueeze(-1).to(dtype=grad_x0.dtype)
                    # Clip per-token gradient norm for numerical stability across scales
                    gnorm = grad_x0.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    grad_clipped = grad_x0 * torch.clamp(1.0 / gnorm, max=1.0)
                    _, std_cur = self.dynamic.marginal_params(t_cur)
                    delta_score = (float(guidance_scale) * grad_clipped / std_cur.to(device=self.device).clamp(min=0.05)).to(dtype=score.dtype)
                    score = score + delta_score
                    bias_norm_acc += float(delta_score.float().norm(dim=-1)[valid_res].mean().item())
                    bias_norm_steps += 1

                with torch.no_grad():
                    drift, diffusion = self.dynamic.reverse_params(x, t_cur, score, ode_sampling=ode_sampling)
                    x_mean = x + drift * dt
                    if ode_sampling:
                        x = x_mean
                    else:
                        noise = torch.randn(x.shape, generator=gen, device=self.device, dtype=x.dtype)
                        x = x_mean + diffusion.view(-1, 1, 1) * torch.sqrt(-dt) * noise

                    if z_ctx is not None and ctx_mask is not None:
                        if idx == n_steps - 1 or ode_sampling:
                            mu_n, _ = self.dynamic.marginal_params(t_nxt)
                            x_ctx_n = z_ctx * mu_n.to(dtype=x.dtype)
                        else:
                            mu_n, std_n = self.dynamic.marginal_params(t_nxt)
                            eps_n = torch.randn(x.shape, generator=gen, device=self.device, dtype=x.dtype)
                            x_ctx_n = z_ctx * mu_n.to(dtype=x.dtype) + eps_n * std_n.to(dtype=x.dtype)
                        x = torch.where(ctx_mask.unsqueeze(-1), x_ctx_n, x)
                        x_mean = torch.where(ctx_mask.unsqueeze(-1), z_ctx * self.dynamic.marginal_params(t_nxt)[0].to(dtype=x.dtype), x_mean)

                    x_0_sc = x_0_hat.detach()

        mean_bias_norm = bias_norm_acc / bias_norm_steps if bias_norm_steps > 0 else 0.0
        return x_mean, attn_mask, mean_bias_norm

    @torch.no_grad()
    def generate_unconditional(
        self,
        length: int,
        *,
        n_samples: int = 1,
        steps: int = 250,
        ode_sampling: bool = False,
        seed: int = 20261125,
        t_min: float = DIMA_DEFAULT_T_MIN,
    ) -> list[dict[str, Any]]:
        """Unconditional continuous latent diffusion generation (``Exp 4.1``, SDE or ODE)."""
        L = int(length)
        if not (1 <= L <= self.max_residues):
            raise ValueError(f"Unconditional length {L} must be in [1, {self.max_residues}]")
        n = max(1, int(n_samples))
        x_mean, attn_mask, _ = self._run_reverse_diffusion(
            [L] * n, steps=steps, ode_sampling=ode_sampling, seed=seed, t_min=t_min
        )
        seqs, threedis, maxps, ents = self.decode_sequences_from_latents(
            x_mean, [L] * n, attention_mask=attn_mask
        )
        return [
            {
                "arm": self.arm,
                "sequence": seqs[i],
                "threedi_sequence": threedis[i],
                "length": L,
                "steps": int(steps),
                "ode_sampling": bool(ode_sampling),
                "seed": int(seed),
                "sample_index": i,
                "mean_max_prob": maxps[i],
                "mean_entropy": ents[i],
                "mean_applied_bias_norm": 0.0,
            }
            for i in range(n)
        ]

    @torch.no_grad()
    def infill(
        self,
        parent: str,
        *,
        start: int,
        extent: int,
        steps: int = 250,
        ode_sampling: bool = False,
        seed: int = 20261125,
        t_min: float = DIMA_DEFAULT_T_MIN,
    ) -> dict[str, Any]:
        """``R5a`` RePaint continuous-latent span infilling over ``[start, start + extent)``."""
        seq = _validate_aa20_sequence(parent)
        length = len(seq)
        if not (0 <= start < length and extent >= 1 and start + extent <= length):
            raise ValueError(f"Invalid infilling span [{start}, {start + extent}) for length {length}")

        # Handle long parents (> 510 aa) via centered 510-aa window around the target span
        if length <= self.max_residues:
            ws, we = 0, length
        else:
            if extent > self.max_residues:
                raise ValueError(f"Infill extent {extent} exceeds DiMA max_residues={self.max_residues}")
            mid = start + extent // 2
            ws, we = _centered_window(length, mid, self.max_residues)
            ws = min(ws, start)
            if ws + self.max_residues < start + extent:
                ws = max(0, start + extent - self.max_residues)
            we = ws + self.max_residues

        sub_parent = seq[ws:we]
        local_start = start - ws
        local_end = local_start + extent
        sub_len = len(sub_parent)
        off = self.res_offset

        # Mask target span BEFORE encoding so scaffold latents contain zero target-span leakage!
        span_set = set(range(local_start, local_end))
        z_ctx, attn_mask = self.encode_latents([sub_parent], mask_positions_list=[span_set])
        ctx_mask = attn_mask.bool().clone()
        ctx_mask[0, off + local_start : off + local_end] = False

        x_mean, _, _ = self._run_reverse_diffusion(
            [sub_len],
            steps=steps,
            ode_sampling=ode_sampling,
            seed=seed,
            t_min=t_min,
            z_ctx=z_ctx,
            ctx_mask=ctx_mask,
        )
        decoded_sub, threedis, maxps, ents = self.decode_sequences_from_latents(
            x_mean, [sub_len], attention_mask=attn_mask
        )
        infilled_span = decoded_sub[0][local_start:local_end]
        full_out = seq[:start] + infilled_span + seq[start + extent :]
        return {
            "arm": self.arm,
            "sequence": full_out,
            "infilled_span": infilled_span,
            "threedi_sequence": threedis[0],
            "start": int(start),
            "extent": int(extent),
            "length": length,
            "window": [ws, we],
            "steps": int(steps),
            "ode_sampling": bool(ode_sampling),
            "seed": int(seed),
            "mean_max_prob": maxps[0],
            "mean_entropy": ents[0],
        }

    # Alias matching DPLM2Adapter
    infill_window = infill

    @torch.no_grad()
    def continue_from_prefix(
        self,
        prefix: str,
        target_length: int,
        *,
        steps: int = 250,
        ode_sampling: bool = False,
        seed: int = 20261125,
        t_min: float = DIMA_DEFAULT_T_MIN,
    ) -> dict[str, Any]:
        """``r4_cont`` RePaint continuous-latent prefix continuation within ``510``-aa budget."""
        pref = _validate_aa20_sequence(prefix)
        pref_len = len(pref)
        if target_length <= pref_len:
            raise ValueError(f"target_length {target_length} must exceed len(prefix)={pref_len}")

        eff_target = min(int(target_length), self.max_residues)
        if pref_len >= eff_target:
            raise ValueError(
                f"Prefix length {pref_len} leaves no suffix budget under DiMA max_residues={self.max_residues}"
            )
        suffix_len = eff_target - pref_len
        off = self.res_offset

        # Build dummy placeholder of length eff_target with all suffix sites masked before encoding
        placeholder = pref + ("A" * suffix_len)
        suffix_set = set(range(pref_len, eff_target))
        z_ctx, attn_mask = self.encode_latents([placeholder], mask_positions_list=[suffix_set])
        ctx_mask = torch.zeros_like(attn_mask, dtype=torch.bool)
        if self.spec.has_bos_eos:
            ctx_mask[0, 0] = True  # clamp <cls>
            ctx_mask[0, off + eff_target] = True  # clamp <eos>
        ctx_mask[0, off : off + pref_len] = True

        x_mean, _, _ = self._run_reverse_diffusion(
            [eff_target],
            steps=steps,
            ode_sampling=ode_sampling,
            seed=seed,
            t_min=t_min,
            z_ctx=z_ctx,
            ctx_mask=ctx_mask,
        )
        decoded, threedis, maxps, ents = self.decode_sequences_from_latents(
            x_mean, [eff_target], attention_mask=attn_mask
        )
        gen_seq = pref + decoded[0][pref_len:eff_target]
        return {
            "arm": self.arm,
            "sequence": gen_seq,
            "threedi_sequence": threedis[0],
            "prefix_len": pref_len,
            "length": len(gen_seq),
            "target_length_requested": int(target_length),
            "steps": int(steps),
            "ode_sampling": bool(ode_sampling),
            "seed": int(seed),
            "mean_max_prob": maxps[0],
            "mean_entropy": ents[0],
        }

    # Alias matching DPLM2Adapter
    continue_suffix = continue_from_prefix

    def sample_guided_family(
        self,
        length: int,
        *,
        profile_log_bias: np.ndarray | None = None,
        guidance_scale: float = 1.0,
        steps: int = 250,
        ode_sampling: bool = False,
        seed: int = 20261125,
        t_min: float = DIMA_DEFAULT_T_MIN,
    ) -> dict[str, Any]:
        """``Exp 4.2`` unguided (``N``) and Potts/profile-guided (``Pr/Pr_strong/C/Cx``) sampling.

        Reports exact ``mean_applied_bias_norm`` (``0.0`` when ``profile_log_bias is None``).
        """
        L = int(length)
        if not (1 <= L <= self.max_residues):
            raise ValueError(f"Guided sampling length {L} must be in [1, {self.max_residues}]")
        x_mean, attn_mask, mean_bias_norm = self._run_reverse_diffusion(
            [L],
            steps=steps,
            ode_sampling=ode_sampling,
            seed=seed,
            t_min=t_min,
            profile_log_bias=profile_log_bias,
            guidance_scale=guidance_scale,
        )
        with torch.no_grad():
            seqs, threedis, maxps, ents = self.decode_sequences_from_latents(
                x_mean, [L], attention_mask=attn_mask
            )
        return {
            "arm": self.arm,
            "sequence": seqs[0],
            "threedi_sequence": threedis[0],
            "length": L,
            "steps": int(steps),
            "ode_sampling": bool(ode_sampling),
            "seed": int(seed),
            "guidance_scale": float(guidance_scale) if profile_log_bias is not None else 0.0,
            "mean_applied_bias_norm": float(mean_bias_norm),
            "mean_max_prob": maxps[0],
            "mean_entropy": ents[0],
        }


def load_dima_adapter(
    arm_id: str,
    *,
    dima_root: Path | str | None = None,
    model_root: Path | str | None = None,
    encoder_dir: Path | str | None = None,
    device: str = "cpu",
    dtype: str | None = None,
    use_ema: bool = True,
    load_encoder: bool = True,
) -> DiMAAdapter:
    """Convenience factory returning a loaded ``DiMAAdapter``."""
    return DiMAAdapter(
        arm_id,
        dima_root=dima_root,
        model_root=model_root,
        encoder_dir=encoder_dir,
        device=device,
        dtype=dtype,
        use_ema=use_ema,
        load_encoder=load_encoder,
    )
