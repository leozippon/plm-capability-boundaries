"""Inference-only ProteinGLM budget serving.

This module is the single source for the ProteinGLM continuation estimand
served here: native ``<gmask><sop><eos>`` prefix, AA20 residues, no trailing
``<eos>``, scored over residues 2..L. That is continuation NLL, not a full
protein CLM (1..L plus EOS) and not ProteinGym admission.

The released checkpoint is not vendored. Modeling, configuration, and
quantization files are copied from the checkpoint directory into a derived
package; the only patches are dropping the top-level ``deepspeed`` import and
making ``get_checkpoint_fn`` raise. Gradient-checkpointing is not rewritten to
``torch.utils.checkpoint``. Importing the derived modeling file still runs the
upstream JIT fusion flags (``torch._C._jit_set_profiling_mode`` and related
overrides) as a process-wide side effect; this serving path does not stub them.

``config.max_length`` is supplied by ``PretrainedConfig`` as 20 even though
``config.json`` and ``ProteinGLMConfig.__init__`` do not declare it. Because the
attribute exists, this loader does not overwrite it. Budget forward does not
read ``max_sequence_length``. Generation, cache extraction, and 7B numerical
qualification are out of scope.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.amino_acids import AA20

#: ArmSpec.input_format for the served continuation rendering.
INPUT_FORMAT = "gmask_sop_eos"

#: Chat-template generation prefix, encoded with ``add_special_tokens=False``.
NATIVE_PREFIX = "<gmask><sop><eos>"

#: Tokenizer ids of :data:`NATIVE_PREFIX`: ``<gmask>``, ``<sop>``, ``<eos>``.
PREFIX_IDS: tuple[int, ...] = (29, 32, 34)

PREFIX_LENGTH = len(PREFIX_IDS)

#: Checkpoint ``seq_length``. Tokenizer ``model_max_length`` is 2048 and is not
#: the serving window.
CONTEXT_LENGTH = 1024

#: ``sequence_target_mask`` rule: keep prediction columns ``q >= PREFIX_LENGTH``.
TARGET_RULE = "residues_2_to_L"

SOURCE_FILENAMES: tuple[str, ...] = (
    "configuration_proteinglm.py",
    "quantization.py",
    "modeling_proteinglm.py",
)

_MODELING = "modeling_proteinglm.py"
_DEEPSPEED_IMPORT = "import torch, deepspeed\n"
_TORCH_IMPORT = "import torch\n"
_CHECKPOINT_FN_SOURCE = (
    "def get_checkpoint_fn():\n"
    "    if deepspeed.checkpointing.is_configured():\n"
    "        checkpoint = deepspeed.checkpointing.checkpoint\n"
    "    else:\n"
    "        checkpoint = torch.utils.checkpoint.checkpoint\n"
    "    return checkpoint\n"
)
_CHECKPOINT_FN_DERIVED = (
    "def get_checkpoint_fn():\n"
    "    raise RuntimeError(\n"
    '        "ProteinGLM serving is inference-only; '
    'gradient checkpointing is unsupported"\n'
    "    )\n"
)
PATCH_DROP_DEEPSPEED = "drop_toplevel_deepspeed_import"
PATCH_INFERENCE_CHECKPOINT = "inference_only_checkpoint_fn"

_DERIVED_PACKAGES: dict[str, "DerivedPackage"] = {}


@dataclass(frozen=True)
class DerivedFile:
    name: str
    patch_kinds: tuple[str, ...]


@dataclass(frozen=True)
class DerivedPackage:
    """A derived inference-only copy of the three ProteinGLM Python files."""

    source_dir: Path
    path: Path
    package_name: str
    files: tuple[DerivedFile, ...]

    def record(self) -> dict[str, Any]:
        return {
            "source_dir": str(self.source_dir),
            "derived_dir": str(self.path),
            "package_name": self.package_name,
            "files": [
                {
                    "name": item.name,
                    "patch_kinds": list(item.patch_kinds),
                }
                for item in self.files
            ],
            "jit_fusion_flags": (
                "importing derived modeling_proteinglm.py still executes the "
                "upstream torch._C JIT fusion overrides on non-Darwin platforms"
            ),
        }


def patch_modeling_source(source: str) -> str:
    """Apply the two allowed modeling patches, and no others."""

    if source.count(_DEEPSPEED_IMPORT) != 1:
        raise ValueError("expected exactly one top-level 'import torch, deepspeed'")
    if source.count(_CHECKPOINT_FN_SOURCE) != 1:
        raise ValueError("expected exactly one get_checkpoint_fn definition to replace")
    patched = source.replace(_DEEPSPEED_IMPORT, _TORCH_IMPORT, 1)
    patched = patched.replace(_CHECKPOINT_FN_SOURCE, _CHECKPOINT_FN_DERIVED, 1)
    if "import torch, deepspeed" in patched:
        raise ValueError("derived modeling still aliases deepspeed at import")
    if "deepspeed.checkpointing" in patched:
        raise ValueError("derived modeling still calls deepspeed checkpointing")
    if "torch.utils.checkpoint.checkpoint" in patched.split("def get_checkpoint_fn()")[1].split("\n\n", 1)[0]:
        raise ValueError("get_checkpoint_fn must not fall back to torch checkpoint")
    if "gradient checkpointing is unsupported" not in patched:
        raise ValueError("get_checkpoint_fn was not rewritten to the inference-only error")
    return patched


def derive_serving_package(source_dir: Path) -> DerivedPackage:
    """Copy configuration/quantization/modeling and apply the allowed patches."""

    source_dir = source_dir.resolve()
    payloads: dict[str, bytes] = {}
    for name in SOURCE_FILENAMES:
        path = source_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"ProteinGLM source file missing: {path}")
        payloads[name] = path.read_bytes()
    modeling_text = payloads[_MODELING].decode("utf-8")
    derived_modeling = patch_modeling_source(modeling_text).encode("utf-8")
    key = str(source_dir)
    cached = _DERIVED_PACKAGES.get(key)
    if cached is not None and cached.path.is_dir():
        return cached
    dest = Path(tempfile.mkdtemp(prefix="proteinglm_derived_"))
    (dest / "__init__.py").write_text("# derived ProteinGLM inference package\n", encoding="utf-8")
    files: list[DerivedFile] = []
    for name in SOURCE_FILENAMES:
        if name == _MODELING:
            data = derived_modeling
            kinds = (PATCH_DROP_DEEPSPEED, PATCH_INFERENCE_CHECKPOINT)
        else:
            data = payloads[name]
            kinds = ()
        (dest / name).write_bytes(data)
        files.append(DerivedFile(name=name, patch_kinds=kinds))
    package = DerivedPackage(
        source_dir=source_dir,
        path=dest,
        package_name=dest.name.replace("-", "_"),
        files=tuple(files),
    )
    _DERIVED_PACKAGES[key] = package
    return package


def _import_derived_module(package: DerivedPackage, module: str) -> Any:
    pkg_name = package.package_name
    if pkg_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            pkg_name,
            package.path / "__init__.py",
            submodule_search_locations=[str(package.path)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import derived ProteinGLM package {pkg_name}")
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[pkg_name] = pkg
        spec.loader.exec_module(pkg)
    full = f"{pkg_name}.{module}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, package.path / f"{module}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import derived ProteinGLM module {full}")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[full] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def load_derived_classes(source_dir: Path) -> tuple[Any, Any, DerivedPackage]:
    """Return ``(ProteinGLMConfig, ProteinGLMForCasualLM, package)``."""

    package = derive_serving_package(source_dir)
    _import_derived_module(package, "configuration_proteinglm")
    _import_derived_module(package, "quantization")
    modeling = _import_derived_module(package, "modeling_proteinglm")
    config_cls = modeling.ProteinGLMConfig
    model_cls = modeling.ProteinGLMForCasualLM
    if config_cls.__name__ != "ProteinGLMConfig":
        raise TypeError(f"unexpected config class {config_cls!r}")
    if model_cls.__name__ != "ProteinGLMForCasualLM":
        raise TypeError(f"unexpected causal LM class {model_cls!r}")
    return config_cls, model_cls, package


def load_serving_config(source_dir: Path) -> Any:
    """Read ProteinGLMConfig from a checkpoint directory without loading weights."""

    source_dir = source_dir.resolve()
    config_cls, _, _ = load_derived_classes(source_dir)
    return config_cls.from_pretrained(str(source_dir), local_files_only=True)


def require_serving_config(config: Any) -> None:
    """Refuse a config that is not the served causal 1024-window ProteinGLM contract.

    ``tiny_config`` may use a shorter window for CPU tests. Production
    :func:`load_pretrained` always goes through this check, so a metadata file
    that still claims causal 1024 cannot run some other layout.
    """

    def _as_int(value: Any) -> Any:
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    observed = {
        "is_causal": getattr(config, "is_causal", None),
        "rotary_embedding_2d": getattr(config, "rotary_embedding_2d", None),
        "seq_length": _as_int(getattr(config, "seq_length", None)),
        "quantization_bit": _as_int(getattr(config, "quantization_bit", None)),
        "moe": getattr(config, "moe", None),
    }
    expected = {
        "is_causal": True,
        "rotary_embedding_2d": False,
        "seq_length": CONTEXT_LENGTH,
        "quantization_bit": 0,
        "moe": False,
    }
    mismatches = [name for name, want in expected.items() if observed[name] != want]
    if mismatches:
        detail = ", ".join(
            f"{name}={observed[name]!r} (required {expected[name]!r})"
            for name in mismatches
        )
        raise ValueError(
            "ProteinGLM serving requires is_causal=True, rotary_embedding_2d=False, "
            f"seq_length={CONTEXT_LENGTH}, quantization_bit=0, moe=False; got {detail}"
        )


def max_length_compatibility(config: Any) -> dict[str, Any]:
    """Record whether ``max_length`` already exists; never overwrite it.

    Hugging Face ``PretrainedConfig`` supplies ``max_length=20`` even when the
    checkpoint JSON omits it. The parent decision is: fill from ``seq_length``
    only when the attribute is missing. Because it is present, this function
    leaves it unchanged.
    """

    present = hasattr(config, "max_length")
    value = getattr(config, "max_length", None) if present else None
    if (not present) or value is None:
        config.max_length = int(config.seq_length)
        return {
            "filled_max_length_from_seq_length": True,
            "max_length": int(config.max_length),
            "seq_length": int(config.seq_length),
        }
    return {
        "filled_max_length_from_seq_length": False,
        "max_length": int(value),
        "seq_length": int(config.seq_length),
        "note": (
            "PretrainedConfig already provides max_length; serving does not "
            "rewrite it. Budget forward does not consult max_sequence_length."
        ),
    }


def residue_token_ids(tokenizer: Any) -> dict[str, int]:
    unknown = tokenizer.unk_token_id
    resolved: dict[str, int] = {}
    for residue in AA20:
        token_id = tokenizer.convert_tokens_to_ids(residue)
        if token_id is None or token_id == unknown:
            raise ValueError(f"ProteinGLM tokenizer has no AA20 id for {residue!r}")
        resolved[residue] = int(token_id)
    return resolved


def encode_budget_text(tokenizer: Any, text: str, *, max_len: int) -> list[int]:
    """Encode one already-prefixed continuation string; no trailing EOS."""

    if max_len < PREFIX_LENGTH + 2:
        raise ValueError(
            "ProteinGLM budget max_len must admit the prefix and two residues"
        )
    if not text.startswith(NATIVE_PREFIX):
        raise ValueError("ProteinGLM budget text must start with <gmask><sop><eos>")
    residues = text[len(NATIVE_PREFIX) :]
    if any(character not in AA20 for character in residues):
        raise ValueError("ProteinGLM budget residues must be AA20 with no extra symbols")
    length = len(residues)
    if length < 2:
        raise ValueError("ProteinGLM budget scoring needs at least two residues")
    if PREFIX_LENGTH + length > CONTEXT_LENGTH:
        raise ValueError(
            f"ProteinGLM budget prefix+L exceeds seq_length {CONTEXT_LENGTH}"
        )
    if PREFIX_LENGTH + length > max_len:
        raise ValueError(
            f"ProteinGLM budget prefix+L exceeds max_len {max_len}; "
            "the continuation estimand is not silently truncated"
        )
    encoded = tokenizer(text, add_special_tokens=False, return_tensors=None)["input_ids"]
    if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    ids = [int(value) for value in encoded]
    if ids[:PREFIX_LENGTH] != list(PREFIX_IDS):
        raise ValueError("ProteinGLM budget encoding does not match prefix ids [29, 32, 34]")
    allowed = set(residue_token_ids(tokenizer).values())
    content = ids[PREFIX_LENGTH:]
    if len(content) != length or any(token_id not in allowed for token_id in content):
        raise ValueError(
            "ProteinGLM budget encoding must be prefix plus AA20 ids and no extra specials"
        )
    return ids


def _budget_residue_length(sequence: str) -> int:
    """The AA20 and two-residue gates, declared once for both budget paths.

    These two checks are about what the sequence *is*, not about whether it fits,
    so they belong to the render and the length probe alike and are stated once
    rather than twice (rule 12). The serving-window check deliberately is not
    here: it is a scoring-path refusal and stays in
    :func:`render_budget_sequence` and :func:`encode_budget_text`.
    """

    if any(character not in AA20 for character in sequence):
        raise ValueError("ProteinGLM budget residues must be AA20 with no extra symbols")
    if len(sequence) < 2:
        raise ValueError("ProteinGLM budget scoring needs at least two residues")
    return len(sequence)


def budget_token_length(sequence: str) -> int:
    """Rendered token length of one budget continuation. **A length probe.**

    This is not a scoring path and it does not refuse an over-context sequence.
    Stage 20 asks every scorer for ``token_lengths`` before it scores an assay and
    turns a maximum above that arm's context into a recorded skip --
    "the rendered variant exceeds this arm's context; truncating would score a
    sequence that may not contain the mutated position"
    (``retrieval_bound.py``'s ``_score_one_arm``). A probe that
    raises instead of returning converts an assay-level exclusion into a failure
    of the whole arm, which is exactly what EXP-R2-240's `pgym_proteinglm-7b-clm`
    cell recorded: it died in ``token_lengths``, through
    :func:`render_budget_sequence`, without ever reaching that branch.

    **The refusal is not moved, only the measurement is made total.** Every path
    that actually scores a sequence still goes through
    :func:`render_budget_sequence` or :func:`encode_budget_text`, both of which
    raise when ``PREFIX_LENGTH + L > CONTEXT_LENGTH``; and
    :func:`encode_budget_text` additionally requires the tokenizer to return
    exactly ``PREFIX_LENGTH + L`` ids, so this arithmetic cannot drift from the
    encoding it measures without that check firing at score time. The boundary is
    the same on both paths: 1021 residues render to 1024 tokens and score, 1022
    render to 1025 and are excluded.

    Illegal or too-short input still raises here, because it is a defect in the
    sequence rather than a statement about the window (Failure Principle).
    """

    return PREFIX_LENGTH + _budget_residue_length(sequence)


def render_budget_sequence(sequence: str) -> str:
    """Prefix one continuation; refuses anything outside the serving window."""

    length = _budget_residue_length(sequence)
    if PREFIX_LENGTH + length > CONTEXT_LENGTH:
        raise ValueError(
            f"ProteinGLM budget prefix+L exceeds seq_length {CONTEXT_LENGTH}"
        )
    return NATIVE_PREFIX + sequence


def load_pretrained(
    source_dir: Path,
    *,
    device: str,
    dtype: str,
    strict: bool = False,
) -> dict[str, Any]:
    """Load ProteinGLMForCasualLM from derived classes over the original weights.

    Does not register AutoConfig/AutoModel and does not execute the original
    modeling file. Tokenizer still comes from the checkpoint's own custom class.
    """

    if dtype != "float32":
        raise ValueError(
            f"ProteinGLM budget serving is FP32-only; refused dtype {dtype!r}"
        )
    source_dir = source_dir.resolve()
    config_cls, model_cls, package = load_derived_classes(source_dir)
    config = config_cls.from_pretrained(str(source_dir), local_files_only=True)
    require_serving_config(config)
    max_length_note = max_length_compatibility(config)
    load_kwargs: dict[str, Any] = {
        "config": config,
        "torch_dtype": __import__("torch").float32,
        "local_files_only": True,
        "device_map": {"": device},
    }
    strict_load: dict[str, int] | None = None
    if strict:
        from ..core.arms import require_clean_loading_info, unpack_pretrained_loading_info

        loaded = model_cls.from_pretrained(
            str(source_dir), output_loading_info=True, **load_kwargs
        )
        model, info = unpack_pretrained_loading_info(loaded)
        strict_load = require_clean_loading_info(info, arm="proteinglm-7b-clm")
    else:
        model = model_cls.from_pretrained(str(source_dir), **load_kwargs)
    model.eval()
    encoder = getattr(getattr(model, "transformer", None), "encoder", None)
    if encoder is not None and getattr(encoder, "gradient_checkpointing", False):
        raise RuntimeError("ProteinGLM serving refuses a training checkpoint path")
    if model.training:
        raise RuntimeError("ProteinGLM serving requires eval(); training is unsupported")
    import torch
    from transformers import AutoTokenizer

    observed = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in model.parameters()
            if parameter.is_floating_point()
        }
    )
    if observed != ["float32"]:
        raise ValueError(f"proteinglm-7b-clm: declared dtype float32, observed {observed}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(source_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    return {
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
        "strict_load": strict_load,
        "derived": package.record(),
        "max_length": max_length_note,
        "attn_implementation": None,
        "torch": torch,
    }


def tiny_config(source_dir: Path, *, num_layers: int = 2, hidden_size: int = 32, seq_length: int = 64) -> Any:
    """A derived real ProteinGLMConfig small enough for CPU tests."""

    import torch

    config_cls, _, _ = load_derived_classes(source_dir)
    if hidden_size % 4 != 0:
        raise ValueError("tiny hidden_size must divide the test head count")
    config = config_cls(
        num_layers=num_layers,
        padded_vocab_size=128,
        hidden_size=hidden_size,
        ffn_hidden_size=hidden_size * 2,
        kv_channels=hidden_size // 4,
        num_attention_heads=4,
        seq_length=seq_length,
        quantization_bit=0,
        rotary_embedding_2d=False,
        use_pytorch_sdpa=True,
        is_causal=True,
        moe=False,
        num_experts=0,
        experts_per_token=0,
        torch_dtype=torch.float32,
    )
    max_length_compatibility(config)
    return config


def tiny_causal_lm(source_dir: Path, *, use_pytorch_sdpa: bool = True, **config_kwargs: Any) -> Any:
    """Random-init derived ProteinGLMForCasualLM; does not load 7B weights."""

    config = tiny_config(source_dir, **config_kwargs)
    config.use_pytorch_sdpa = bool(use_pytorch_sdpa)
    _, model_cls, _ = load_derived_classes(source_dir)
    model = model_cls(config, empty_init=False, device="cpu")
    model.eval()
    return model
