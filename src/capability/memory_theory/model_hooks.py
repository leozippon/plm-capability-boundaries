"""Architecture-agnostic model loader and PyTorch hook utilities for Track B (B2-B4).

Supports causal decoders (GPT-2, ProtGPT2, ProGen2, ProGen3, ProtGPT3, ProteinGLM-CLM,
Llama, Qwen2.5, Qwen3, Galactica, RITA) and masked / diffusion / structure-aware PLMs
(ESM2, DPLM, ESMC, AMPLIFY, SaProt in AA-only '#' or full Foldseek-3Di+AA mode).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

# Canonical local/cluster directory name aliases for arms
ARM_DIR_CANDIDATES: dict[str, list[str]] = {
    "gpt2": ["gpt2"],
    "gpt2-medium": ["gpt2-medium"],
    "gpt2-large": ["gpt2-large"],
    "gpt2-xl": ["gpt2-xl"],
    "llama-3.2-3b": ["Llama-3.2-3B", "llama-3.2-3b"],
    "llama-2-7b": ["Llama-2-7b-hf", "llama-2-7b"],
    "qwen2.5-0.5b": ["Qwen2.5-0.5B", "qwen2.5-0.5b"],
    "qwen2.5-3b": ["Qwen2.5-3B", "qwen2.5-3b"],
    "qwen2.5-7b": ["Qwen2.5-7B", "qwen2.5-7b"],
    "qwen2.5-32b": ["Qwen2.5-32B", "qwen2.5-32b"],
    "qwen3-8b-base": ["Qwen3-8B-Base", "qwen3-8b-base"],
    "progen2-small": ["progen2-small"],
    "progen2-medium": ["progen2-medium"],
    "progen2-base": ["progen2-base"],
    "progen2-large": ["progen2-large"],
    "progen2-xlarge": ["progen2-xlarge"],
    "progen3-112m": ["progen3-112m"],
    "progen3-3b": ["progen3-3b"],
    "protgpt2": ["ProtGPT2", "protgpt2"],
    "protgpt3-112m": ["ProtGPT3-112M", "protgpt3-112m"],
    "protgpt3-1.3b": ["ProtGPT3-1.3B", "protgpt3-1.3b"],
    "protgpt3-10b": ["ProtGPT3-10B", "protgpt3-10b"],
    "proteinglm-7b-clm": ["proteinglm-7b-clm"],
    "rita-xl": ["RITA_xl", "rita-xl"],
    "galactica-125m": ["galactica-125m"],
    "galactica-1.3b": ["galactica-1.3b"],
    "galactica-6.7b": ["galactica-6.7b"],
    "galactica-30b": ["galactica-30b"],
    "esm2-8m": ["esm2_t6_8M_UR50D", "esm2-8m", "esm2_8m"],
    "esm2-35m": ["esm2_t12_35M_UR50D", "esm2-35m", "esm2_35m"],
    "esm2-150m": ["esm2_t30_150M_UR50D", "esm2-150m", "esm2_150m"],
    "esm2-650m": ["esm2_t33_650M_UR50D", "esm2-650m", "esm2_650m"],
    "esm2-3b": ["esm2_t36_3B_UR50D", "esm2-3b", "esm2_3b"],
    "esm2-15b": ["esm2_t48_15B_UR50D", "esm2-15b", "esm2_15b"],
    "dplm-150m": ["dplm_150m", "dplm-150m"],
    "dplm-650m": ["dplm_650m", "dplm-650m"],
    "dplm-3b": ["dplm_3b", "dplm-3b"],
    "saprot-35m": ["SaProt_35M_AF2", "saprot-35m", "saprot_35m"],
    "saprot-650m": ["SaProt_650M_AF2", "saprot-650m", "saprot_650m"],
    "saprot-1.3b": ["SaProt_1.3B_AF2", "saprot-1.3b", "saprot_1.3b"],
    "esmc-300m": ["esmc-300m-hf", "esmc-300m", "esmc_300m"],
    "esmc-600m": ["esmc-600m-hf", "esmc-600m", "esmc_600m"],
    "amplify-120m": ["AMPLIFY_120M", "amplify-120m", "amplify_120m"],
    "amplify-350m": ["AMPLIFY_350M", "amplify-350m", "amplify_350m"],
    "proteinglm-1b-mlm": ["proteinglm-1b-mlm", "proteinglm_1b_mlm"],
    "proteinglm-3b-mlm": ["proteinglm-3b-mlm", "proteinglm_3b_mlm"],
    "proteinglm-10b-mlm": ["proteinglm-10b-mlm", "proteinglm_10b_mlm"],
}

MASKED_ARMS_PREFIXES = ("esm2", "dplm", "saprot", "esmc", "amplify", "pepmlm", "proteinglm-1b-mlm", "proteinglm-3b-mlm", "proteinglm-10b-mlm")
TEXT_ARM_PREFIXES = ("gpt2", "llama", "qwen", "bygpt5", "dialogpt")


def is_masked_arm(arm: str) -> bool:
    low = arm.lower()
    return low.startswith(MASKED_ARMS_PREFIXES) or low.endswith("-mlm")


def infer_arm_modality(arm: str) -> str:
    low = arm.lower()
    if low.startswith(TEXT_ARM_PREFIXES):
        return "text"
    return "protein"


def resolve_model_dir(arm: str, explicit_dir: Path | None = None) -> Path:
    """Resolve checkpoint directory cleanly across local L20 and remote H200 hosts."""
    if explicit_dir is not None:
        if not explicit_dir.exists():
            raise FileNotFoundError(f"Explicit --model-dir does not exist: {explicit_dir}")
        return explicit_dir

    roots: list[Path] = []
    for env_var in ("MODEL_ROOT", "TEXT_MODEL_BASE_DIR", "MODEL_BASE_DIR"):
        val = os.environ.get(env_var)
        if val:
            roots.append(Path(val))
    roots.extend(
        [
            Path("/home/lzp/models/general_plm_20261120"),
            Path("/Data/public/models_R2"),
            Path("/Data/public"),
            Path.home() / "models",
            Path("/Data/lzp/models"),
        ]
    )

    candidates = list(ARM_DIR_CANDIDATES.get(arm, [arm]))
    for variant in (arm.replace("-", "_"), arm.replace("_", "-")):
        if variant not in candidates:
            candidates.append(variant)
    for root in roots:
        if not root.is_dir():
            continue
        for cname in candidates:
            p = root / cname
            if p.is_dir():
                return p
    raise FileNotFoundError(
        f"Could not locate checkpoint for arm={arm!r}; tried names={candidates} under {[str(r) for r in roots]}"
    )


@dataclass
class LoadedModelBundle:
    arm: str
    model_dir: Path
    model: nn.Module
    tokenizer: Any
    is_masked: bool
    modality: str
    d_model: int
    n_layers: int
    n_heads: int
    head_dim: int
    vocab_size: int
    dtype: torch.dtype
    device: torch.device
    seq_first_hidden: bool = False


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_name!r}; expected one of {sorted(mapping)}")
    return mapping[key]


def _extract_model_dims(model: nn.Module, tokenizer: Any) -> tuple[int, int, int, int, int]:
    cfg = model.config
    d_model = int(
        getattr(cfg, "hidden_size", None)
        or getattr(cfg, "n_embd", None)
        or getattr(cfg, "d_model", None)
        or getattr(cfg, "embed_dim", 0)
    )
    n_layers = int(
        getattr(cfg, "num_hidden_layers", None)
        or getattr(cfg, "n_layer", None)
        or getattr(cfg, "num_layers", 0)
    )
    n_heads = int(
        getattr(cfg, "num_attention_heads", None)
        or getattr(cfg, "n_head", None)
        or getattr(cfg, "num_heads", 1)
    )
    head_dim = int(getattr(cfg, "head_dim", d_model // max(n_heads, 1)))
    vocab_size = int(getattr(cfg, "vocab_size", len(tokenizer)))
    return d_model, n_layers, n_heads, head_dim, vocab_size


def load_model_bundle(
    arm: str,
    *,
    device: str = "cuda:0",
    dtype: str = "float32",
    model_dir: Path | None = None,
    modality: str | None = None,
) -> LoadedModelBundle:
    """Load tokenizer and model offline with trust_remote_code when required."""
    ckpt_dir = resolve_model_dir(arm, explicit_dir=model_dir)
    torch_dtype = _resolve_dtype(dtype)
    dev = torch.device(device)
    masked = is_masked_arm(arm)
    mod = modality or infer_arm_modality(arm)

    # ProteinGLM checkpoints ship modeling_proteinglm.py with a top-level `import torch, deepspeed`;
    # reuse src/capability/models/proteinglm.py's derived inference-only package (no deepspeed needed).
    seq_first_hidden = False
    if (ckpt_dir / "modeling_proteinglm.py").is_file():
        from ..models.proteinglm import _import_derived_module, derive_serving_package

        seq_first_hidden = True
        pkg = derive_serving_package(ckpt_dir)
        _import_derived_module(pkg, "configuration_proteinglm")
        _import_derived_module(pkg, "quantization")
        pglm_mod = _import_derived_module(pkg, "modeling_proteinglm")
        cfg = pglm_mod.ProteinGLMConfig.from_pretrained(str(ckpt_dir), local_files_only=True)
        tok = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True, trust_remote_code=True)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        model_cls = pglm_mod.ProteinGLMForMaskedLM if masked else pglm_mod.ProteinGLMForCasualLM
        model = model_cls.from_pretrained(
            str(ckpt_dir),
            config=cfg,
            local_files_only=True,
            torch_dtype=torch_dtype,
        )
    else:
        # Check if custom code is needed (e.g., ProGen2, RITA, AMPLIFY)
        cfg = AutoConfig.from_pretrained(str(ckpt_dir), local_files_only=True, trust_remote_code=True)
        model_type = str(getattr(cfg, "model_type", "")).lower()
        if model_type in ("esm", "roberta", "bert", "amplify", "dplm"):
            masked = True

        tok = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True, trust_remote_code=True)
        if tok.pad_token is None:
            if tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            elif getattr(tok, "mask_token", None) is not None:
                tok.pad_token = tok.mask_token

        if masked:
            model = AutoModelForMaskedLM.from_pretrained(
                str(ckpt_dir),
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(ckpt_dir),
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )

    model.to(dev)
    model.eval()
    d_model, n_layers, n_heads, head_dim, vocab_size = _extract_model_dims(model, tok)
    return LoadedModelBundle(
        arm=arm,
        model_dir=ckpt_dir,
        model=model,
        tokenizer=tok,
        is_masked=masked,
        modality=mod,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        dtype=torch_dtype,
        device=dev,
        seq_first_hidden=seq_first_hidden,
    )


def get_input_embedding_module(model: nn.Module) -> nn.Module:
    """Return input token embedding module, with fallbacks for custom HF models like ProGen2."""
    try:
        emb = model.get_input_embeddings()
        if emb is not None:
            return emb
    except NotImplementedError:
        pass

    candidate_paths = (
        ("transformer", "wte"),
        ("model", "embed_tokens"),
        ("model", "decoder", "embed_tokens"),
        ("esm", "embeddings", "word_embeddings"),
        ("roberta", "embeddings", "word_embeddings"),
        ("transformer", "embedding", "word_embeddings"),
        ("rita", "wte"),
    )
    for path in candidate_paths:
        obj: Any = model
        ok = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, nn.Module):
            return obj

    for mod in model.modules():
        if isinstance(mod, nn.Embedding):
            return mod
    raise RuntimeError(f"Cannot locate input embedding module on {type(model).__name__}")


def get_transformer_blocks(model: nn.Module) -> list[nn.Module]:
    """Return ordered list of transformer blocks across standard HF/PLM architectures."""
    candidate_paths = (
        ("transformer", "h"),          # GPT-2, ProtGPT2, ProGen2, GPT-J
        ("model", "layers"),           # Llama, Qwen2, Qwen3, Mistral, ProGen3, ProtGPT3
        ("model", "decoder", "layers"),  # OPT / Galactica
        ("esm", "encoder", "layer"),   # ESM2, SaProt, DPLM
        ("roberta", "encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("encoder", "layer"),
        ("transformer", "encoder", "layers"),  # ChatGLM / ProteinGLM
        ("rita", "layers"),
        ("layers",),
    )
    for path in candidate_paths:
        obj: Any = model
        ok = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, list, tuple)) and len(obj) > 0:
            return list(obj)
    raise RuntimeError(
        f"Cannot locate transformer block list on {type(model).__name__}; "
        f"top children={list(dict(model.named_children()).keys())}"
    )


def select_depth_quantiles(n_layers: int) -> list[int]:
    """Select 5 canonical layer indices {0, L/4, L/2, 3L/4, L-1}, deduplicated."""
    if n_layers <= 0:
        raise ValueError(f"Invalid n_layers={n_layers}")
    raw = [
        0,
        int(round(0.25 * (n_layers - 1))),
        int(round(0.50 * (n_layers - 1))),
        int(round(0.75 * (n_layers - 1))),
        n_layers - 1,
    ]
    seen: list[int] = []
    for idx in raw:
        idx = max(0, min(n_layers - 1, idx))
        if idx not in seen:
            seen.append(idx)
    return seen


def _find_first_attr(obj: nn.Module, names: Sequence[str]) -> nn.Module | None:
    for name in names:
        if hasattr(obj, name):
            sub = getattr(obj, name)
            if isinstance(sub, nn.Module):
                return sub
    return None


@dataclass
class BlockKVHookSpec:
    """Describes how to extract attention (K, V) and MLP hidden activations from one block."""
    layer_idx: int
    k_module: nn.Module
    v_module: nn.Module
    kv_mode: str  # "separate", "gpt2_c_attn", "progen2_qkv", "qkv_packed"
    mlp_act_module: nn.Module | None
    mlp_kind: str  # "up_proj", "fc_in", "c_fc", "intermediate"


def inspect_block_kv_modules(block: nn.Module, layer_idx: int) -> BlockKVHookSpec:
    """Locate attention K/V projection and MLP up/intermediate projection inside a block."""
    attn = _find_first_attr(block, ("attn", "self_attn", "self_attention", "attention"))
    if attn is None:
        raise RuntimeError(f"Layer {layer_idx}: cannot find attention submodule on {type(block).__name__}")

    # Nested ESM/RoBERTa attention.self
    attn_inner = _find_first_attr(attn, ("self",)) or attn

    k_mod: nn.Module | None = None
    v_mod: nn.Module | None = None
    kv_mode = "separate"

    if hasattr(attn_inner, "k_proj") and hasattr(attn_inner, "v_proj"):
        k_mod = getattr(attn_inner, "k_proj")
        v_mod = getattr(attn_inner, "v_proj")
        kv_mode = "separate"
    elif hasattr(attn_inner, "key") and hasattr(attn_inner, "value"):
        k_mod = getattr(attn_inner, "key")
        v_mod = getattr(attn_inner, "value")
        kv_mode = "separate"
    elif hasattr(attn_inner, "c_attn"):
        k_mod = getattr(attn_inner, "c_attn")
        v_mod = k_mod
        kv_mode = "gpt2_c_attn"
    elif hasattr(attn_inner, "qkv_proj"):
        k_mod = getattr(attn_inner, "qkv_proj")
        v_mod = k_mod
        # ProGen2 packs QKV as interleaved mp_num blocks or standard 3-way split
        kv_mode = "progen2_qkv" if "progen" in type(attn_inner).__name__.lower() else "qkv_packed"
    elif hasattr(attn_inner, "query_key_value"):
        k_mod = getattr(attn_inner, "query_key_value")
        v_mod = k_mod
        kv_mode = "proteinglm_qkv" if hasattr(attn_inner, "num_attention_heads_per_partition") else "qkv_packed"
    else:
        raise RuntimeError(
            f"Layer {layer_idx}: unrecognized attention projection structure on {type(attn_inner).__name__}: "
            f"{list(dict(attn_inner.named_children()).keys())}"
        )

    # MLP hidden activation source
    mlp = _find_first_attr(block, ("mlp", "feed_forward", "ffn", "intermediate"))
    mlp_act_mod: nn.Module | None = None
    mlp_kind = "none"
    if mlp is not None:
        if hasattr(mlp, "up_proj"):
            mlp_act_mod = getattr(mlp, "up_proj")
            mlp_kind = "up_proj"
        elif hasattr(mlp, "fc_in"):
            mlp_act_mod = getattr(mlp, "fc_in")
            mlp_kind = "fc_in"
        elif hasattr(mlp, "c_fc"):
            mlp_act_mod = getattr(mlp, "c_fc")
            mlp_kind = "c_fc"
        elif hasattr(mlp, "dense"):
            mlp_act_mod = getattr(mlp, "dense")
            mlp_kind = "intermediate"
        elif hasattr(mlp, "dense_h_to_4h"):
            mlp_act_mod = getattr(mlp, "dense_h_to_4h")
            mlp_kind = "up_proj"
    elif hasattr(block, "fc1"):
        mlp_act_mod = getattr(block, "fc1")
        mlp_kind = "fc1"

    return BlockKVHookSpec(
        layer_idx=layer_idx,
        k_module=k_mod,
        v_module=v_mod,
        kv_mode=kv_mode,
        mlp_act_module=mlp_act_mod,
        mlp_kind=mlp_kind,
    )


def split_packed_kv(
    tensor: torch.Tensor,
    mode: str,
    *,
    n_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split packed QKV projection output (B, T, 3*D) into (K, V) of shape (B, T, D)."""
    if mode == "gpt2_c_attn" or mode == "qkv_packed":
        q, k, v = torch.chunk(tensor, 3, dim=-1)
        return k, v
    if mode == "progen2_qkv":
        # ProGen2 splits along last dim into mp_num=4 chunks, each split into (Q, V, K) or (Q, K, V)
        # In HF ProGen2 modeling_progen.py: qkv = qkv_proj(x); reshaped (B, T, mp_num, -1); split 3 -> q, v, k
        b, t, total = tensor.shape
        if total % 3 != 0:
            raise ValueError(f"Unexpected ProGen2 qkv dim {total}")
        # Standard ProGen2 uses mp_num=4 when total == 3 * n_heads * head_dim
        mp_num = 4 if (total // 3) % 4 == 0 else 1
        chunk = tensor.view(b, t, mp_num, total // mp_num)
        q, v, k = torch.chunk(chunk, 3, dim=-1)
        return k.reshape(b, t, -1), v.reshape(b, t, -1)
    if mode == "proteinglm_qkv":
        # ProteinGLM reshapes (..., n_heads, 3 * head_dim) and splits along the last dim
        b, t, total = tensor.shape
        if total != 3 * n_heads * head_dim:
            q, k, v = torch.chunk(tensor, 3, dim=-1)
            return k, v
        chunk = tensor.view(b, t, n_heads, 3 * head_dim)
        q, k, v = torch.chunk(chunk, 3, dim=-1)
        return k.reshape(b, t, -1), v.reshape(b, t, -1)
    raise ValueError(f"Unsupported packed kv mode: {mode}")


def format_sequence_for_arm(
    seq: str,
    arm: str,
    *,
    saprot_seq: str | None = None,
    saprot_mode: str = "3di",
) -> str:
    """Format a protein sequence according to the arm's expected tokenization."""
    low = arm.lower()
    if low.startswith("saprot"):
        if saprot_mode == "3di":
            if saprot_seq is None:
                raise ValueError("SaProt 3di mode requires saprot_seq (interleaved AA+3Di)")
            return saprot_seq
        if saprot_mode == "aa_only":
            return "".join(f"{aa}#" for aa in seq)
        raise ValueError(f"Unknown saprot_mode={saprot_mode!r}")
    if low.startswith("progen2"):
        return "1" + seq + "2"
    if low.startswith("proteinglm") and low.endswith("clm"):
        return "<gmask><sop><eos>" + seq
    if low.startswith("protgpt2"):
        return seq
    if low.startswith("galactica"):
        return f"[START_AMINO]{seq}[END_AMINO]"
    return seq
