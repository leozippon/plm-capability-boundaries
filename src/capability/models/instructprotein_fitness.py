"""Native ProteinGym scoring for InstructProtein protein mode.

This is a joint-rendering door, not an :class:`~src.capability.core.arms.ArmSpec` and
not a widening of the frozen three-arm ProteinGym run. Dual-mode qualification
is a separate existing record: protein mode is measurable and text mode is not.
This module scores protein mode only. LOOKUP against staged UniRef50 is an
external profile baseline, not a retrieval bound.

Padding uses the checkpoint config's own pad id, read back through the
tokenizer. No pad token is invented.

Embedding padding is a different quantity from a padding token. The released
checkpoint's embedding and output head carry more rows than its tokenizer has
tokens (50287 real tokens padded to 50304, a multiple of 128), which is the
checkpoint's own shape and is recorded as a measured fact rather than refused.
What is required is a row for every id the tokenizer can emit; the rendering
refuses an id above ``len(tokenizer)`` in
:meth:`src.capability.models.joint_modes.JointTokenisation.render`, and that is what makes
the tail safe.
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
    "INSTRUCTPROTEIN_ARM",
    "INSTRUCTPROTEIN_CORPUS",
    "InstructProteinFitnessScorer",
    "LoadedInstructProtein",
    "RENDERING_FAMILY",
    "load_instructprotein",
]

RENDERING_FAMILY = "instructprotein"
CHECKPOINT_VARIABLE = "MODEL_ROOT"
INSTRUCTPROTEIN_ARM = "instructprotein"

CHECKPOINTS: dict[str, Path] = {
    INSTRUCTPROTEIN_ARM: MODEL_ROOT / "InstructProtein",
}

INSTRUCTPROTEIN_CORPUS: dict[str, dict[str, str]] = {
    INSTRUCTPROTEIN_ARM: {
        "declared": "uniref100_continued_pretraining_then_instruction_tuning",
        "identification": (
            "external UniRef50 profile baseline, not a retrieval bound"
        ),
        "note": (
            "InstructProtein's released documentation names UniRef100 continued "
            "pretraining and instruction tuning. The staged UniRef50 snapshot is "
            "NOT identified as that set or as contained in it, so LOOKUP is an "
            "external UniRef50 profile channel and MODEL−LOOKUP is a capability "
            "comparison against it, not a retrieval exclusion and not a lower "
            "bound on retrieval. Neither containment direction is evidenced, so "
            "the residual bias is not signed. Dual-mode interface status is a "
            "separate existing record: this door scores protein mode only and "
            "does not mint a fitness PASS from a checkpoint name"
        ),
    }
}


def _as_int(value: Any, *, what: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} is not an integer: {value!r}") from exc


@dataclass
class LoadedInstructProtein:
    """InstructProtein weights, tokenizer, and the resolved protein rendering."""

    name: str
    model: Any
    tokenizer: Any
    tokenisation: JointTokenisation
    context: int
    facts: dict[str, Any]

    @property
    def device(self) -> Any:
        return self.model.device


def load_instructprotein(name: str, *, device: str, dtype: str) -> LoadedInstructProtein:
    """Load InstructProtein strictly and resolve the declared protein rendering.

    Tokenizer and rendering first, then weights. Text mode is not a scoring
    door. The pad id is the checkpoint config's own declaration, not invented.
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if name != INSTRUCTPROTEIN_ARM:
        raise KeyError(
            f"unknown InstructProtein arm {name!r}; declared: [{INSTRUCTPROTEIN_ARM}]"
        )
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
    if getattr(config, "pad_token_id", None) is None:
        raise ValueError(
            f"{name}: config declares no pad_token_id; scoring will not invent one"
        )
    adopt_config_declared_pad_token(tokenizer, config, arm=name)
    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError(
            f"{name}: config-declared pad did not read back; scoring will not invent one"
        )
    input_vocab = int(model.get_input_embeddings().num_embeddings)
    output_vocab = int(model.get_output_embeddings().out_features)
    config_vocab = int(config.vocab_size)
    tokenizer_vocab = int(len(tokenizer))
    if not input_vocab == output_vocab == config_vocab:
        raise ValueError(
            f"{name}: config, input and output vocabulary sizes disagree: "
            f"{config_vocab}, {input_vocab}, {output_vocab}"
        )
    if tokenizer_vocab > input_vocab:
        raise ValueError(
            f"{name}: the tokenizer carries {tokenizer_vocab} tokens but the config, "
            f"input and output vocabularies are {input_vocab} rows, so ids "
            f"{input_vocab}..{tokenizer_vocab - 1} have no embedding row to be fed "
            "through or scored against"
        )
    padded_rows = input_vocab - tokenizer_vocab
    context = config_context_length(config)
    n_layers, d_model = config_shape(config)
    facts = {
        "rung": name,
        "scientific_role": (
            "InstructProtein protein-mode ProteinGym door; text mode is not "
            "scored here. Dual-mode interface status is a separate existing "
            "record and is not restated as a fitness verdict"
        ),
        "checkpoint": str(resolved),
        "model_type": str(model_type),
        "architectures": list(getattr(config, "architectures", []) or []),
        "n_layers": int(n_layers),
        "d_model": int(d_model),
        "vocab_size": config_vocab,
        "input_vocab_size": input_vocab,
        "output_vocab_size": output_vocab,
        "context": int(context),
        "pad_token_id": _as_int(pad, what="pad_token_id"),
        "dtype_requested": dtype,
        "dtype_observed": observed,
        "device": device,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": tokenizer_vocab,
        "padded_embedding_rows": padded_rows,
        "padded_embedding_note": (
            f"the {padded_rows} embedding rows above this tokenizer's "
            f"{tokenizer_vocab} tokens are padding the tokenizer can never emit, "
            "so those ids are unreachable and no scored position can land on one. "
            f"config.vocab_size {config_vocab} declares that padded width and the "
            "input and output vocabularies agree with it, so the checkpoint is "
            "internally consistent while its tokenizer is smaller than its "
            "embedding"
        ),
        "rendering": RENDERING_FAMILY,
        "joint_mode": "protein",
        "symbol_unit": RESIDUE_UNIT,
        "strict_loading_diagnostics": diagnostics,
        "facts_source": (
            "read back from the loaded model's config and parameters, not echoed "
            "from the request"
        ),
    }
    return LoadedInstructProtein(
        name=name,
        model=model,
        tokenizer=tokenizer,
        tokenisation=tokenisation,
        context=int(context),
        facts=facts,
    )


class InstructProteinFitnessScorer:
    """Summed N-to-C log-likelihood of the dedicated residue tokens.

    The sum is over :meth:`JointTokenisation.render` scored positions only —
    the ``ƤA``..``ƤY`` run inside ``<protein>...</protein>`` — under a
    full-vocabulary softmax. Delimiters, padding, and any instruction context
    are not summed. One scored symbol is one residue.
    """

    score_description = (
        "summed log-likelihood of the dedicated residue tokens of the declared "
        "<protein>ƤA..ƤY</protein> rendering, over the protein-content "
        "positions only, full-vocabulary softmax, N-to-C, one residue per scored token"
    )
    scoring_stratum = STRATUM_N_TO_C
    symbol_unit = RESIDUE_UNIT

    def __init__(self, loaded: LoadedInstructProtein, *, batch_size: int) -> None:
        import torch

        self.torch = torch
        self.loaded = loaded
        self.name = loaded.name
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch size must be positive")
        if loaded.tokenisation.declaration.name != RENDERING_FAMILY:
            raise ValueError(
                f"{self.name}: expected rendering {RENDERING_FAMILY!r}, got "
                f"{loaded.tokenisation.declaration.name!r}"
            )
        self.context = loaded.context
        self.vocab_size = int(loaded.facts["vocab_size"])
        self.tokenisation = loaded.tokenisation
        pad = loaded.tokenizer.pad_token_id
        if pad is None:
            raise ValueError(
                f"{self.name}: tokenizer has no pad token; scoring will not invent one"
            )
        self.pad_id = int(pad)
        self.residues = 0
        self.scored_tokens = 0
        self._cached_key: tuple[str, ...] | None = None
        self._cached_records: list[RenderedProtein] | None = None

    @property
    def facts(self) -> dict[str, Any]:
        return self.loaded.facts

    def render(self, sequences: Sequence[str]) -> list[RenderedProtein]:
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
