"""Native ProteinGym scoring for the four Galactica checkpoints.

This is a new scoring door, not an :class:`~src.capability.core.arms.ArmSpec` and not a
change to the frozen EXP-R2-225 endpoints. Dual-mode interface qualification is
a separate, already-recorded check; nothing here mints a fitness PASS from a
checkpoint name. ``galactica-125m`` is included as the small-scale reference
whose protein mode was not identified under that interface check.

LOOKUP against the staged UniRef50 snapshot is an external profile baseline, not
a retrieval bound: Galactica's released scientific corpus is not identified as
that snapshot, and neither containment direction is evidenced, so residual bias
is not signed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core.arms import (
    MODEL_ROOT,
    adopt_config_declared_pad_token,
    config_context_length,
    config_shape,
    require_clean_loading_info,
    require_input_path,
    unpack_pretrained_loading_info,
)
from .joint_modes import (
    JointTokenisation,
    RESIDUE_UNIT,
    RenderedProtein,
    resolve,
)
from .scale_comparison import STRATUM_N_TO_C

__all__ = [
    "CHECKPOINT_VARIABLE",
    "CHECKPOINTS",
    "DECLARED_PAD_TOKEN_ID",
    "GALACTICA_CORPUS",
    "GALACTICA_RUNGS",
    "GalacticaFitnessScorer",
    "LoadedGalactica",
    "RENDERING_FAMILY",
    "load_galactica",
    "rung",
]

RENDERING_FAMILY = "galactica"
CHECKPOINT_VARIABLE = "MODEL_ROOT"
DECLARED_PAD_TOKEN_ID = 1

GALACTICA_RUNGS: tuple[str, ...] = (
    "galactica-125m",
    "galactica-1.3b",
    "galactica-6.7b",
    "galactica-30b",
)

CHECKPOINTS: dict[str, Path] = {
    name: MODEL_ROOT / name for name in GALACTICA_RUNGS
}

_CORPUS_IDENTIFICATION = "external UniRef50 profile baseline, not a retrieval bound"

_CORPUS_DECLARED = "galactica_scientific_corpus"

_CORPUS_NOTE = (
    "Galactica's released documentation describes a scientific corpus of papers, "
    "code, and knowledge bases. The staged UniRef50 snapshot is NOT identified as "
    "that set or as contained in it, so LOOKUP is an external UniRef50 profile "
    "channel and MODEL - LOOKUP is a capability comparison against it, not a "
    "retrieval exclusion and not a lower bound on retrieval. Neither containment "
    "direction is evidenced, so the residual bias is not signed. Dual-mode "
    "interface status is a separate existing record and is not restated as a "
    "fitness verdict here."
)

_CORPUS_NOTE_125M = (
    _CORPUS_NOTE
    + " This rung is the small-scale reference whose protein mode was not "
    "identified under that interface check; that is a recorded interface status, "
    "not a ProteinGym PASS or FAIL."
)


def _corpus_record(name: str) -> dict[str, str]:
    return {
        "declared": _CORPUS_DECLARED,
        "identification": _CORPUS_IDENTIFICATION,
        "note": _CORPUS_NOTE_125M if name == "galactica-125m" else _CORPUS_NOTE,
    }


GALACTICA_CORPUS: dict[str, dict[str, str]] = {
    name: _corpus_record(name) for name in GALACTICA_RUNGS
}

if sorted(GALACTICA_CORPUS) != sorted(GALACTICA_RUNGS):
    raise AssertionError(
        "every Galactica rung this module can load must declare what its LOOKUP "
        "channel is entitled to"
    )


def rung(name: str) -> str:
    """Refuse a name this scoring door does not carry."""

    if name not in CHECKPOINTS:
        raise KeyError(
            f"unknown Galactica rung {name!r}; declared: {list(GALACTICA_RUNGS)}"
        )
    return name


def _scientific_role(name: str) -> str:
    if name == "galactica-125m":
        return (
            "small-scale dual-mode-unidentified reference of this ProteinGym "
            "supplement; not a fitness verdict"
        )
    return (
        "Galactica checkpoint on this ProteinGym supplement; dual-mode interface "
        "status is a separate existing record and is not restated here"
    )


@dataclass
class LoadedGalactica:
    """One Galactica checkpoint, its tokenizer, and the resolved protein rendering."""

    name: str
    model: Any
    tokenizer: Any
    tokenisation: JointTokenisation
    context: int
    facts: dict[str, Any]

    @property
    def device(self) -> Any:
        return self.model.device


def load_galactica(name: str, *, device: str, dtype: str) -> LoadedGalactica:
    """Load one Galactica checkpoint strictly, then resolve the declared rendering.

    Tokenizer and rendering first, then weights: a vocabulary that cannot carry
    the per-residue ``[START_AMINO]...[END_AMINO]`` alphabet is a configuration
    error, and paying a 30B load to discover it is the shape the joint loader
    already moved ahead of. The load is local and offline. Missing, unexpected,
    mismatched, or error-reported keys are refused, as is a floating-point dtype
    that does not match the request. The pad id is the checkpoint config's own
    declaration, read back through the tokenizer; it is not invented and the
    vocabulary is not resized.
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rung(name)
    resolved = require_input_path(CHECKPOINTS[name].resolve(), CHECKPOINT_VARIABLE)
    tokenizer = AutoTokenizer.from_pretrained(str(resolved), local_files_only=True)
    tokenisation = resolve(tokenizer, RENDERING_FAMILY)
    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError(f"unsupported inference dtype {dtype!r}")
    torch_dtype = getattr(torch, dtype)
    loaded = AutoModelForCausalLM.from_pretrained(
        str(resolved),
        torch_dtype=torch_dtype,
        device_map={"": device},
        output_loading_info=True,
        local_files_only=True,
    )
    model, loading_info = unpack_pretrained_loading_info(loaded)
    model.eval()
    diagnostics = require_clean_loading_info(loading_info, arm=name)
    observed = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in model.parameters()
            if parameter.is_floating_point()
        }
    )
    if observed != [dtype]:
        raise ValueError(f"{name}: requested dtype {dtype}, observed {observed}")
    config = model.config
    model_type = getattr(config, "model_type", None)
    if model_type != "opt":
        raise ValueError(f"{name}: expected an OPT checkpoint, got {model_type!r}")
    input_vocab = int(model.get_input_embeddings().num_embeddings)
    output_vocab = int(model.get_output_embeddings().out_features)
    if not input_vocab == output_vocab == int(config.vocab_size) == len(tokenizer):
        raise ValueError(
            f"{name}: tokenizer, config, input and output vocabulary sizes disagree: "
            f"{len(tokenizer)}, {config.vocab_size}, {input_vocab}, {output_vocab}"
        )
    adopt_config_declared_pad_token(tokenizer, config, arm=name)
    pad = tokenizer.pad_token_id
    if pad is None or int(pad) != DECLARED_PAD_TOKEN_ID:
        raise ValueError(
            f"{name}: expected the config-declared pad token id "
            f"{DECLARED_PAD_TOKEN_ID}, read back {pad!r}. The id is adopted from "
            "the checkpoint config, not invented"
        )
    context = config_context_length(config)
    n_layers, d_model = config_shape(config)
    facts = {
        "rung": name,
        "scientific_role": _scientific_role(name),
        "checkpoint": str(resolved),
        "model_type": str(getattr(config, "model_type", "undeclared")),
        "architectures": list(getattr(config, "architectures", []) or []),
        "n_layers": int(n_layers),
        "d_model": int(d_model),
        "vocab_size": int(config.vocab_size),
        "input_vocab_size": input_vocab,
        "output_vocab_size": output_vocab,
        "context": int(context),
        "pad_token_id": int(pad),
        "dtype_requested": dtype,
        "dtype_observed": observed,
        "device": device,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": int(len(tokenizer)),
        "rendering": RENDERING_FAMILY,
        "symbol_unit": RESIDUE_UNIT,
        "strict_loading_diagnostics": diagnostics,
        "facts_source": (
            "read back from the loaded model's config and parameters, not echoed "
            "from the request"
        ),
    }
    return LoadedGalactica(
        name=name,
        model=model,
        tokenizer=tokenizer,
        tokenisation=tokenisation,
        context=int(context),
        facts=facts,
    )


class GalacticaFitnessScorer:
    """Summed N-to-C log-likelihood of the residue tokens under the declared rendering.

    The sum is over :meth:`JointTokenisation.render` scored positions only -- the
    protein-content run between the delimiter tokens -- under a full-vocabulary
    softmax. Boundary tokens, padding, and any document context are not summed.
    One scored symbol is one residue.
    """

    score_description = (
        "summed log-likelihood of the residue tokens of the declared "
        "[START_AMINO]...[END_AMINO] rendering, over the protein-content "
        "positions only, full-vocabulary softmax, N-to-C, one residue per scored token"
    )
    scoring_stratum = STRATUM_N_TO_C
    symbol_unit = RESIDUE_UNIT

    def __init__(self, loaded: LoadedGalactica, *, batch_size: int) -> None:
        import torch

        self.torch = torch
        self.loaded = loaded
        self.name = loaded.name
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch size must be positive")
        self.context = loaded.context
        self.vocab_size = int(loaded.facts["vocab_size"])
        self.tokenisation = loaded.tokenisation
        pad = loaded.tokenizer.pad_token_id
        if pad is None or int(pad) != DECLARED_PAD_TOKEN_ID:
            raise ValueError(
                f"{self.name}: Galactica scoring right-pads with config-declared "
                f"id {DECLARED_PAD_TOKEN_ID}; read back {pad!r}"
            )
        self.pad_id = int(pad)
        self.residues = 0
        self.scored_tokens = 0
        self._cached_key: tuple[str, ...] | None = None
        self._cached_records: list[RenderedProtein] | None = None

    @property
    def facts(self) -> dict[str, Any]:
        """Observed loader facts; not re-derived at scoring time."""

        return self.loaded.facts

    def render(self, sequences: Sequence[str]) -> list[RenderedProtein]:
        """Every sequence rendered with the declared protein format and no context."""

        return [
            self.tokenisation.render(str(sequence), context=None)
            for sequence in sequences
        ]

    def _records(self, sequences: Sequence[str]) -> list[RenderedProtein]:
        key = tuple(str(sequence) for sequence in sequences)
        if self._cached_key != key:
            records = self.render(key)
            self._cached_key, self._cached_records = key, records
        assert self._cached_records is not None
        return self._cached_records

    def token_lengths(self, sequences: Sequence[str]) -> list[int]:
        """The whole rendered length, which is what a context budget is spent on."""

        return [len(record.token_ids) for record in self._records(sequences)]

    def log_likelihood(self, sequences: Sequence[str]) -> np.ndarray:
        torch = self.torch
        records = self._records(sequences)
        if not records:
            raise ValueError("nothing to score")
        width = max(len(record.token_ids) for record in records)
        if width > self.context:
            raise ValueError(
                f"{self.name}: a rendering of {width} tokens exceeds this "
                f"checkpoint's {self.context}-position context; truncating "
                "would score a sequence that may not contain the mutated "
                "position"
            )
        totals = np.empty(len(records), dtype=np.float64)
        model = self.loaded.model
        device = model.device
        with torch.no_grad():
            for start in range(0, len(records), self.batch_size):
                chunk = records[start : start + self.batch_size]
                chunk_width = max(len(record.token_ids) for record in chunk)
                ids = torch.full(
                    (len(chunk), chunk_width), self.pad_id, dtype=torch.long
                )
                mask = torch.zeros((len(chunk), chunk_width), dtype=torch.long)
                scored = torch.zeros((len(chunk), chunk_width), dtype=torch.bool)
                for row, record in enumerate(chunk):
                    length = len(record.token_ids)
                    ids[row, :length] = torch.tensor(record.token_ids, dtype=torch.long)
                    mask[row, :length] = 1
                    scored[row, list(record.scored_positions)] = True
                    self.residues += record.n_residues
                    self.scored_tokens += record.n_scored_tokens
                ids = ids.to(device)
                logits = model(
                    input_ids=ids, attention_mask=mask.to(device), use_cache=False
                ).logits
                expected_shape = (*ids.shape, self.vocab_size)
                if tuple(logits.shape) != expected_shape:
                    raise RuntimeError(
                        f"{self.name}: expected logits shape {expected_shape}, "
                        f"got {tuple(logits.shape)}"
                    )
                logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
                token = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                keep = scored[:, 1:].to(device)
                totals[start : start + len(chunk)] = (
                    (token * keep).sum(1).double().cpu().numpy()
                )
                # Drop this batch's tensors so they cannot survive into the next
                # forward (the next logits/logp RHS is evaluated before rebinding).
                del logits, logp, token, keep
        if not np.all(np.isfinite(totals)):
            raise RuntimeError(
                f"{self.name}: a scored sequence returned a non-finite total"
            )
        return totals

    def residue_accounting(self) -> dict[str, Any]:
        if not self.scored_tokens:
            raise RuntimeError("no scored token has been seen, so no rate was measured")
        return {
            "symbol_unit": self.symbol_unit,
            "residues": int(self.residues),
            "scored_tokens": int(self.scored_tokens),
            "residues_per_scored_token": self.residues / self.scored_tokens,
            "note": (
                "measured on this run's own scored queue. One scored symbol here is "
                "one residue, so a magnitude from this family is in nats per residue"
            ),
        }

    def release(self) -> None:
        del self.loaded
        self._cached_key = None
        self._cached_records = None
        self.torch.cuda.empty_cache()
