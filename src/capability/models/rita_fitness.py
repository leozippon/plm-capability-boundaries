"""Native ProteinGym scoring door for a RITA checkpoint.

This is a dedicated scoring route, not an :class:`~src.capability.core.arms.ArmSpec`,
not a panel member, and not a change to generic ``load_arm`` / ``tokenize_batch``.
The factory supplies the staged path; this module does not copy a checkpoint
table. Dual-mode or context-information qualification is a separate record and
is not restated as a fitness verdict. LOOKUP, if a later stage compares against
it, remains an external profile baseline, not a retrieval exclusion.

Inference is float32 only. The checkpoint JSON may store ``torch_dtype``
float16 and an out-of-vocabulary ``eos_token_id``; those are recorded and are
not the scoring dtype or the EOS id. Native encoding is the arm's own rendering:
the document-boundary ``<EOS>`` that RITA's training stream separates documents
with, then the residues, then the tokenizer postprocessor's terminal ``<EOS>``
(id 2). The boundary and the terminator are one id, so neither is content and
the scored targets are the sequence's residues. Scoring does not invent a pad
token, does not pass an attention mask, and does not truncate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core.amino_acids import AA20
from ..core.arms import (
    EOS_BOUNDED_BOUNDARY,
    config_context_length,
    config_shape,
    eos_bounded_rendering,
    require_clean_loading_info,
    require_input_path,
    unpack_pretrained_loading_info,
)
from .scale_comparison import STRATUM_N_TO_C

__all__ = [
    "CANONICAL_RESIDUES",
    "CHECKPOINT_VARIABLE",
    "INFERENCE_DTYPE",
    "NATIVE_EOS_ID",
    "NATIVE_PAD_ID",
    "NATIVE_RESIDUE_IDS",
    "RITA_ARM",
    "RITA_CORPUS",
    "VOCAB_SIZE",
    "RitaFitnessScorer",
]

CHECKPOINT_VARIABLE = "MODEL_ROOT"
INFERENCE_DTYPE = "float32"
NATIVE_EOS_ID = 2
NATIVE_PAD_ID = 1
VOCAB_SIZE = 26
NATIVE_RESIDUE_IDS = frozenset(range(3, 23))
CANONICAL_RESIDUES = frozenset(AA20)
RITA_ARM = "rita-xl"
RITA_CORPUS: dict[str, dict[str, str]] = {
    RITA_ARM: {
        "declared": "uniref100",
        "identification": (
            "external UniRef50 profile baseline, not a retrieval bound"
        ),
        "note": (
            "RITA's released documentation names UniRef-100 as the pretraining "
            "corpus family. The staged UniRef50 snapshot is NOT identified as "
            "that set or as contained in it, so LOOKUP is an external UniRef50 "
            "profile channel and MODEL - LOOKUP is a capability comparison "
            "against it, not a retrieval exclusion and not a lower bound on "
            "retrieval. Neither containment direction is evidenced, so the "
            "residual bias is not signed. This door does not mint a fitness "
            "PASS from a checkpoint name."
        ),
    }
}


def _as_int(value: Any, *, what: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} is not an integer: {value!r}") from exc


def _as_int_list(ids: Any) -> list[int]:
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    try:
        return [_as_int(value, what="token id") for value in ids]
    except TypeError as exc:
        raise ValueError(f"encoded ids are not a sequence: {ids!r}") from exc


def _floating_dtypes(module: Any) -> tuple[list[str], list[str]]:
    parameters = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in module.parameters()
            if parameter.is_floating_point()
        }
    )
    buffers = sorted(
        {
            str(buffer.dtype).removeprefix("torch.")
            for buffer in module.buffers()
            if buffer.is_floating_point()
        }
    )
    return parameters, buffers


def _sequence_of(rendered: str) -> str:
    """The AA20 sequence of an ``eos_bounded_seq`` rendering, boundary removed.

    The rendering is refused rather than interpreted when the boundary is
    missing: a bare residue string is the input this door used to score, and
    scoring it silently under the corrected rendering would put a number against
    a format the checkpoint was not given.
    """

    text = str(rendered)
    if not text.startswith(EOS_BOUNDED_BOUNDARY):
        raise ValueError(
            f"native RITA scoring takes the arm's own rendering -- "
            f"{EOS_BOUNDED_BOUNDARY!r} then the sequence -- and {text!r} does not "
            "start with the document boundary"
        )
    sequence = text[len(EOS_BOUNDED_BOUNDARY) :]
    if not sequence or any(symbol not in CANONICAL_RESIDUES for symbol in sequence):
        raise ValueError(
            f"non-canonical amino-acid sequence in rendering {text!r}; RITA scoring "
            "accepts non-empty AA20 only and does not apply O/U/X/B/Z/J "
            "normalisation"
        )
    return sequence


def _native_special_ids(tokenizer: Any) -> tuple[int, int]:
    eos_id = _as_int(
        tokenizer.convert_tokens_to_ids(EOS_BOUNDED_BOUNDARY),
        what="tokenizer <EOS> id",
    )
    pad_id = _as_int(
        tokenizer.convert_tokens_to_ids("<PAD>"), what="tokenizer <PAD> id"
    )
    if eos_id != NATIVE_EOS_ID:
        raise ValueError(
            f"native <EOS> must be id {NATIVE_EOS_ID}; tokenizer maps it to {eos_id}"
        )
    if pad_id != NATIVE_PAD_ID:
        raise ValueError(
            f"native <PAD> must be id {NATIVE_PAD_ID}; tokenizer maps it to {pad_id}"
        )
    mapped_eos = tokenizer.eos_token_id
    if mapped_eos is not None and _as_int(mapped_eos, what="tokenizer.eos_token_id") != NATIVE_EOS_ID:
        raise ValueError(
            f"tokenizer eos_token_id {mapped_eos} is not native <EOS>="
            f"{NATIVE_EOS_ID}; the config's out-of-vocabulary eos_token_id "
            "must not be adopted"
        )
    mapped_pad = tokenizer.pad_token_id
    if mapped_pad is not None and _as_int(mapped_pad, what="tokenizer.pad_token_id") != NATIVE_PAD_ID:
        raise ValueError(
            f"tokenizer pad_token_id {mapped_pad} is not native <PAD>="
            f"{NATIVE_PAD_ID}; this door does not invent or rewrite pad"
        )
    return NATIVE_EOS_ID, NATIVE_PAD_ID


def native_encode_for_budget(tokenizer: Any, text: str) -> list[int]:
    """Document boundary, residues, tokenizer-native EOS, with no pad token.

    ``text`` is the arm's own rendering, :func:`src.capability.core.arms.Cohort.input_strings`
    output; a bare residue string is refused rather than wrapped.
    """

    cache = getattr(tokenizer, "_rita_budget_native", None)
    if cache is None:
        eos_id, pad_id = _native_special_ids(tokenizer)
        residue_ids = _verify_residue_tokenisation(tokenizer, eos_id=eos_id)
        cache = (eos_id, pad_id, residue_ids)
        tokenizer._rita_budget_native = cache
    eos_id, _, residue_ids = cache
    return _encode_native(
        tokenizer, text, eos_id=eos_id, residue_ids=residue_ids
    )


def _encode_native(
    tokenizer: Any,
    text: str,
    *,
    eos_id: int,
    residue_ids: dict[str, int],
) -> list[int]:
    sequence = _sequence_of(text)
    expected = [residue_ids[symbol] for symbol in sequence]
    boundary_id = _as_int(
        tokenizer.convert_tokens_to_ids(EOS_BOUNDED_BOUNDARY),
        what=f"tokenizer {EOS_BOUNDED_BOUNDARY} id",
    )
    if boundary_id != eos_id:
        raise ValueError(
            f"the rendering prefixes {EOS_BOUNDED_BOUNDARY!r} at id {boundary_id} "
            f"while this checkpoint's terminal <EOS> is {eos_id}; the document "
            "boundary and the terminator are the same token, so the two must agree"
        )
    residues = _as_int_list(tokenizer.encode(text, add_special_tokens=False))
    native = _as_int_list(tokenizer.encode(text))
    if residues != [boundary_id] + expected:
        raise ValueError(
            f"encoding {text!r} without special tokens must be the document "
            f"boundary {boundary_id} plus the per-letter id list {expected}; got "
            f"{residues}"
        )
    if native != [boundary_id] + expected + [eos_id]:
        raise ValueError(
            "native encoding must be the document boundary, the per-letter residue "
            f"ids, and the tokenizer-native terminal EOS {eos_id}, with no other "
            f"special token; got {native} from expected {expected}"
        )
    if any(token_id < 0 or token_id >= VOCAB_SIZE for token_id in native):
        raise ValueError(
            f"encoded ids {native} are not inside the {VOCAB_SIZE}-wide vocabulary"
        )
    return native


def _verify_residue_tokenisation(
    tokenizer: Any, *, eos_id: int
) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for residue in sorted(CANONICAL_RESIDUES):
        token_id = _as_int(
            tokenizer.convert_tokens_to_ids(residue), what=f"residue {residue!r} id"
        )
        if token_id not in NATIVE_RESIDUE_IDS:
            raise ValueError(
                f"residue {residue!r} must map to a native AA20 id in 3..22; "
                f"got {token_id}"
            )
        try:
            label = tokenizer.convert_ids_to_tokens(token_id)
        except (AttributeError, TypeError, ValueError, KeyError) as exc:
            raise ValueError(
                f"cannot decode residue id {token_id} for {residue!r}"
            ) from exc
        if str(label) != residue:
            raise ValueError(
                f"id {token_id} decodes to {label!r}, not residue {residue!r}"
            )
        mapping[residue] = token_id
    if len(set(mapping.values())) != len(CANONICAL_RESIDUES):
        raise ValueError(
            "AA20 must occupy 20 distinct non-special ids in 3..22; got "
            f"{sorted(mapping.values())}"
        )
    probe = eos_bounded_rendering("".join(sorted(CANONICAL_RESIDUES)))
    _encode_native(tokenizer, probe, eos_id=eos_id, residue_ids=mapping)
    return mapping


def _require_rita_model(model: Any, *, dtype: str) -> dict[str, Any]:
    config = model.config
    model_type = str(getattr(config, "model_type", "undeclared"))
    if model_type != "rita":
        raise ValueError(f"RITA scoring requires model_type 'rita'; got {model_type!r}")
    vocab = _as_int(getattr(config, "vocab_size", -1), what="config.vocab_size")
    if vocab != VOCAB_SIZE:
        raise ValueError(
            f"RITA scoring requires config.vocab_size {VOCAB_SIZE}; got {vocab}"
        )
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if embedding is None or head is None:
        raise ValueError("RITA scoring requires input embeddings and an lm_head")
    embed_rows = _as_int(embedding.weight.shape[0], what="input embedding rows")
    head_rows = _as_int(head.weight.shape[0], what="lm_head rows")
    if embed_rows != VOCAB_SIZE or head_rows != VOCAB_SIZE:
        raise ValueError(
            "RITA scoring requires input embedding and lm_head rows "
            f"{VOCAB_SIZE}; got embedding {embed_rows} and lm_head {head_rows}"
        )
    parameter_dtypes, buffer_dtypes = _floating_dtypes(model)
    if parameter_dtypes != [dtype]:
        raise ValueError(
            f"requested inference dtype {dtype}, observed parameter dtypes "
            f"{parameter_dtypes}"
        )
    if buffer_dtypes and buffer_dtypes != [dtype]:
        raise ValueError(
            f"requested inference dtype {dtype}, observed buffer dtypes "
            f"{buffer_dtypes}; mixed rotary/buffer dtypes are a failure, not a "
            "reason to patch the author checkpoint"
        )
    n_layers, d_model = config_shape(config)
    context = _as_int(config_context_length(config), what="max_seq_len")
    return {
        "model_type": model_type,
        "architectures": list(getattr(config, "architectures", []) or []),
        "n_layers": _as_int(n_layers, what="num_layers"),
        "d_model": _as_int(d_model, what="d_model"),
        "vocab_size": vocab,
        "context": context,
        "embedding_rows": embed_rows,
        "lm_head_rows": head_rows,
        "dtype_observed_parameters": parameter_dtypes,
        "dtype_observed_buffers": buffer_dtypes,
        "storage_torch_dtype": str(getattr(config, "torch_dtype", "undeclared")),
        "config_eos_token_id": getattr(config, "eos_token_id", None),
    }


class RitaFitnessScorer:
    """Summed N-to-C log-likelihood of a RITA document under its native rendering.

    The rendering is the document boundary followed by the sequence, with the
    tokenizer's terminal ``<EOS>`` still appended. The boundary and that terminal
    token carry one id and a position whose target is a marker is not content, so
    the scored targets are the sequence's residues: the first residue is a scored
    target, conditioned on the boundary, and the terminator is not. The softmax is
    the full 26-wide vocabulary. Equal-length substitution batches are stacked
    with no padding and no attention mask.
    """

    score_description = (
        "document-boundary rendering; summed N-to-C next-token log likelihood "
        "over the sequence's residues"
    )
    scoring_stratum = STRATUM_N_TO_C

    def __init__(
        self,
        *,
        checkpoint: Path,
        device: str,
        dtype: str,
        batch_size: int,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if dtype != INFERENCE_DTYPE:
            raise ValueError(
                f"RITA scoring accepts only {INFERENCE_DTYPE} inference; got {dtype!r}. "
                "Checkpoint storage float16 is not the scoring dtype"
            )
        self.batch_size = _as_int(batch_size, what="batch_size")
        if self.batch_size < 1:
            raise ValueError("batch size must be positive")
        self.torch = torch
        resolved = require_input_path(Path(checkpoint).resolve(), CHECKPOINT_VARIABLE)
        tokenizer = AutoTokenizer.from_pretrained(
            str(resolved),
            local_files_only=True,
            trust_remote_code=True,
        )
        eos_id, pad_id = _native_special_ids(tokenizer)
        residue_ids = _verify_residue_tokenisation(tokenizer, eos_id=eos_id)
        loaded = AutoModelForCausalLM.from_pretrained(
            str(resolved),
            torch_dtype=torch.float32,
            device_map={"": device},
            output_loading_info=True,
            local_files_only=True,
            trust_remote_code=True,
        )
        model, loading_info = unpack_pretrained_loading_info(loaded)
        model.eval()
        diagnostics = require_clean_loading_info(loading_info, arm=resolved.name)
        shapes = _require_rita_model(model, dtype=dtype)
        config_eos = shapes["config_eos_token_id"]
        config_eos_status = "matches native <EOS>"
        if config_eos is None or _as_int(config_eos, what="config.eos_token_id") != eos_id:
            config_eos_status = (
                "ignored; scoring uses tokenizer token <EOS>="
                f"{eos_id}, not config.eos_token_id"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.eos_id = _as_int(eos_id, what="native EOS id")
        self.pad_id = _as_int(pad_id, what="native PAD id")
        self.residue_ids = residue_ids
        self.context = _as_int(shapes["context"], what="context")
        self.facts = {
            "checkpoint": str(resolved),
            "scientific_role": (
                "staged RITA ProteinGym scoring door; not a qualified "
                "information arm, not a generation-capability claim, and not a "
                "retrieval exclusion"
            ),
            "device": device,
            "dtype_requested": dtype,
            "inference_dtype": INFERENCE_DTYPE,
            "tokenizer_class": type(tokenizer).__name__,
            "native_eos_token_id": _as_int(eos_id, what="native EOS id"),
            "native_document_boundary": EOS_BOUNDED_BOUNDARY,
            "scoring_span": (
                "the sequence's residues; the document boundary and the "
                "tokenizer-native terminal EOS carry marker id "
                f"{_as_int(eos_id, what='native EOS id')} and are not scored"
            ),
            "native_pad_token_id": _as_int(pad_id, what="native PAD id"),
            "pad_used_for_scoring": False,
            "config_eos_token_id": (
                None
                if config_eos is None
                else _as_int(config_eos, what="config.eos_token_id")
            ),
            "config_eos_token_id_status": config_eos_status,
            "strict_loading_diagnostics": diagnostics,
            "facts_source": (
                "read back from the loaded tokenizer, model config, and "
                "parameters; storage torch_dtype is not inference dtype"
            ),
            **shapes,
        }

    def token_lengths(self, sequences: Sequence[str]) -> list[int]:
        """Encoded length including both boundaries; used for a context pre-check."""

        return [
            len(
                _encode_native(
                    self.tokenizer,
                    eos_bounded_rendering(sequence),
                    eos_id=self.eos_id,
                    residue_ids=self.residue_ids,
                )
            )
            for sequence in sequences
        ]

    def log_likelihood(self, sequences: Sequence[str]) -> np.ndarray:
        torch = self.torch
        encoded = [
            _encode_native(
                self.tokenizer,
                eos_bounded_rendering(sequence),
                eos_id=self.eos_id,
                residue_ids=self.residue_ids,
            )
            for sequence in sequences
        ]
        if not encoded:
            raise ValueError("nothing to score")
        widths = {len(ids) for ids in encoded}
        if len(widths) != 1:
            raise ValueError(
                "variable-length batch is refused; substitution scoring stacks "
                "equal encodings and does not pad, group, or invent a pad token"
            )
        width = widths.pop()
        if width > self.context:
            raise ValueError(
                f"a native encoding of {width} tokens exceeds this checkpoint's "
                f"{self.context}-position context; truncating would score a "
                "sequence that may not contain the mutated position"
            )
        totals = np.empty(len(encoded), dtype=np.float64)
        device = self.model.device
        with torch.no_grad():
            for start in range(0, len(encoded), self.batch_size):
                chunk = encoded[start : start + self.batch_size]
                ids = torch.tensor(chunk, dtype=torch.long, device=device)
                outputs = self.model(input_ids=ids)
                logits = outputs.logits
                expected = (len(chunk), width, VOCAB_SIZE)
                if tuple(logits.shape) != expected:
                    raise ValueError(
                        f"RITA logits must have shape {expected}; got "
                        f"{tuple(logits.shape)}"
                    )
                logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
                target = ids[:, 1:]
                token = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                # The document boundary is the first token and is context by
                # construction; the tokenizer's terminal EOS carries the same id
                # and is the marker the rendering declares, so it is not content
                # either. What is scored is the sequence's residues.
                content = target != self.eos_id
                totals[start : start + len(chunk)] = (
                    token.masked_fill(~content, 0.0).sum(1).double().cpu().numpy()
                )
        if not np.all(np.isfinite(totals)):
            raise RuntimeError("a scored sequence returned a non-finite total")
        return totals

    def release(self) -> None:
        del self.model
        del self.tokenizer
        self.torch.cuda.empty_cache()
