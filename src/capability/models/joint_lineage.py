"""The ProLLaMA training-stage lineage, and the one rendering all of it is read under.

**What this module is for.** ``retrieval_bound.py`` and ``designed_referent.py``
resolve an arm through :mod:`src.capability.core.arms`, and a joint language-protein
checkpoint is not an arm: ``joint_mode_qualification.py``'s standing rule is
that a checkpoint which has not passed that stage "must not be in ``arms.py`` at
all", so there is nothing there for either stage to name. EXP-R2-226 needs those
two frozen protein-capability queues read on three checkpoints of one lineage,
so the checkpoints are declared here -- by directory, beside the rendering family
they take and the training stage each one is -- and the two stages reach them
through this one declaration rather than through two of their own.

**The lineage.** ``Llama-2-7b-hf`` -> ``ProLLaMA_Stage_1`` -> ``ProLLaMA``: an
unadapted text decoder, the same weights after LoRA continued pretraining on a
protein corpus with ``embed_tokens`` and ``lm_head`` retrained outright, and that
checkpoint after instruction tuning. Adjacent pairs are base-Stage 1 and
Stage 1-Stage 2. All three are ``LlamaForCausalLM``, 32 layers, width 4096,
vocabulary 32000, ``max_position_embeddings`` 4096, and their ``tokenizer.model``
files are byte-identical, so a rendered variant segments identically on all three
and a paired difference is formed on identical scored units.

**One rendering, and it is the unconditioned one.** All three rungs are scored
under the identical bare ``Seq=<...>`` block --
:meth:`src.capability.models.joint_modes.JointRendering.render_protein` with no context,
which is a supported path and not a workaround. Stage 2's declared instruction
form ``[Generate by superfamily] Superfamily=<...>`` is deliberately not used:
supplying a true superfamily would put the assay's own family label into the
prompt, which is a conditioning leak on the very quantity being
measured, and fabricating one would put a false fact inside the measurement.
The asymmetry that creates is declared rather than hidden -- Stage 2 is measured
outside the format it was tuned for, so a Stage 2 reading at or below Stage 1's
is **not** evidence that instruction tuning removed a capability.

**What one scored symbol is.** This family declares the TOKEN as its symbol unit
(:mod:`src.capability.models.joint_modes`): the unmodified LLaMA-2 SentencePiece
vocabulary merges residue runs and there is no escape that reaches a per-residue
alphabet, because the merged rendering *is* the trained format. So the summed
log-likelihood here is over merged multi-residue pieces, which is a different
scoring functional from the single-residue sum every panel protein arm is read
under. A magnitude from this lineage is not commensurable with a residue-unit
family's (Appendix B rule 26, limitation L23); the measured residues per scored
token is accumulated by the scorer and travels beside every magnitude.

**Why the loader is here and not borrowed.** ``joint_mode_qualification.py``
loads a joint checkpoint and reads its shape back off the built object, but it
does not expose Transformers' loading diagnostics, and EXP-R2-226's first
qualification clause is exactly that ``missing_keys``, ``unexpected_keys``,
``mismatched_keys`` and ``error_msgs`` are all empty -- a newly initialised head
is unavailable, never a pass (L24). :func:`load_rung` is therefore this
campaign's one declaration of how a rung reaches a card, used by its
qualification gate and by both fitness stages, so a scored rung and a qualified
rung cannot have been built differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..core.arms import (
    LOADING_INFO_KEYS,
    MODEL_ROOT,
    config_context_length,
    require_clean_loading_info,
    require_input_path,
    unpack_pretrained_loading_info,
)
from .joint_modes import (
    JointTokenisation,
    RenderedProtein,
    TOKEN_UNIT,
    rendering,
    resolve,
)
from .scale_comparison import STRATUM_N_TO_C

__all__ = [
    "ADJACENT_PAIRS",
    "LINEAGE_RUNGS",
    "LoadedRung",
    "LineageRung",
    "PROTEIN_MODE",
    "RENDERING_FAMILY",
    "RUNGS",
    "BareBlockScorer",
    "load_rung",
    "rung",
]

#: The declared family whose input format every rung of this lineage takes. One
#: name and not a per-rung field: three checkpoints that did not share a
#: rendering would not be a lineage this comparison could pair.
RENDERING_FAMILY = "prollama"

#: The mode being read. The lineage has a text mode too and F14 has already
#: measured it; this module exists for the protein side.
PROTEIN_MODE = "protein"


@dataclass(frozen=True)
class LineageRung:
    """One checkpoint of the lineage, and what may be asked of it.

    ``directory_name`` rather than a path so that relocating the staged model
    root -- which is what ``MODEL_ROOT`` does between the
    workstation and the pod -- moves all three rungs together.

    ``reversal_control_applies`` is the one field that differs between the base
    rung and the adapted ones, and it is a declaration rather than a
    convenience. The base checkpoint's measured directional-reversal cost is
    -0.0013 nats/token: it reads protein with no directional structure at all,
    which is what an unadapted text model on protein should look like. It enters
    the ladder as a **declared floor** so the comparison has a pre-adaptation
    reference, and the reversal clause is deliberately not applied to it. A
    base-rung correlation prices what these queues yield from a checkpoint with
    no directional reading of sequence; it is not a protein capability and must
    never be reported as one.
    """

    name: str
    directory_name: str
    training_stage: str
    reversal_control_applies: bool
    note: str

    @property
    def checkpoint(self) -> Path:
        return MODEL_ROOT / self.directory_name


#: The rungs, in ladder order. The order is the declaration: the adjacent pairs
#: below are derived from it rather than spelled a second time.
RUNGS: dict[str, LineageRung] = {
    "llama-2-7b": LineageRung(
        name="llama-2-7b",
        directory_name="Llama-2-7b-hf",
        training_stage="base_text_decoder_no_protein_adaptation",
        reversal_control_applies=False,
        note=(
            "the unadapted text decoder this lineage starts from. It is scored so "
            "the ladder has a pre-adaptation reference and enters as a DECLARED "
            "FLOOR rather than as a qualified protein arm: its measured "
            "directional-reversal cost is -0.0013 nats per scored token, so it "
            "reads protein with no directional structure, and whatever "
            "correlation it returns is what this queue yields without one"
        ),
    ),
    "prollama-stage-1": LineageRung(
        name="prollama-stage-1",
        directory_name="ProLLaMA_Stage_1",
        training_stage="lora_continued_pretraining_with_embed_tokens_and_lm_head_retrained",
        reversal_control_applies=True,
        note=(
            "LoRA continued pretraining on a protein corpus with embed_tokens and "
            "lm_head in modules_to_save -- 262.1 M of a 582.0 M trainable budget, "
            "45.0% of it, so this stage is not low-rank where it matters "
            "(EXP-R2-152's correction). It also fixes a corpus, a schedule and a "
            "data order at once, so what a difference across this step prices is "
            "the released adaptation as a whole and not any one of those factors"
        ),
    ),
    "prollama": LineageRung(
        name="prollama",
        directory_name="ProLLaMA",
        training_stage="instruction_tuned_on_top_of_stage_1",
        reversal_control_applies=True,
        note=(
            "instruction tuning on top of Stage 1, LoRA only and touching neither "
            "the embedding nor the head. It is scored under the same bare "
            "Seq=<...> block as the other two rungs and NOT under its own declared "
            "[Generate by superfamily] instruction form, so a reading at or below "
            "Stage 1's is not evidence that instruction tuning removed a capability"
        ),
    ),
}

LINEAGE_RUNGS: tuple[str, ...] = ("llama-2-7b", "prollama-stage-1", "prollama")

ADJACENT_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (LINEAGE_RUNGS[index], LINEAGE_RUNGS[index + 1])
    for index in range(len(LINEAGE_RUNGS) - 1)
)

if set(LINEAGE_RUNGS) != set(RUNGS) or len(LINEAGE_RUNGS) != len(RUNGS):
    raise AssertionError(
        "the ladder order and the rung table disagree; one of them is not the "
        "declaration it claims to be"
    )


def rung(name: str) -> LineageRung:
    """One rung's declaration, refusing a name this lineage does not carry."""

    if name not in RUNGS:
        raise KeyError(
            f"unknown lineage rung {name!r}; this lineage is {list(LINEAGE_RUNGS)}. "
            "A joint checkpoint that has not passed joint_mode_qualification.py "
            "is not in the panel and is not reachable here either"
        )
    return RUNGS[name]


# ------------------------------------------------------------------ the load


@dataclass(frozen=True)
class LoadedRung:
    """One rung on a card: its weights, its tokenizer, and its resolved rendering."""

    rung: LineageRung
    model: Any
    tokenizer: Any
    tokenisation: JointTokenisation
    context: int
    facts: dict[str, Any]

    @property
    def device(self) -> Any:
        return self.model.device


def load_rung(name: str, *, device: str, dtype: str) -> LoadedRung:
    """Load one rung strictly, and resolve the lineage's rendering against it.

    The tokenizer and the rendering come first, then the weights: a checkpoint
    whose vocabulary cannot carry the declared ``Seq=<...>`` block is a
    configuration error, and paying a 13 GB load to discover it is the shape
    stage 01 already moved its own refusals ahead of.

    Every fact recorded is read back off what was built rather than echoed from
    the request, the observed floating-point dtype included: a build that
    ignored the requested precision would otherwise be recorded as honouring it,
    and a paired difference across rungs at two precisions is partly a
    difference of arithmetic.
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    declaration = rung(name)
    resolved = require_input_path(declaration.checkpoint.resolve(), f"{name} checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(str(resolved))
    tokenisation = resolve(tokenizer, rendering(RENDERING_FAMILY))
    torch_dtype = getattr(torch, dtype, None)
    if not isinstance(torch_dtype, torch.dtype):
        raise ValueError(f"unsupported inference dtype {dtype!r}")
    loaded = AutoModelForCausalLM.from_pretrained(
        str(resolved),
        # ``torch_dtype`` rather than ``dtype`` for the reason
        # ``src.capability.core.arms.load_arm`` records: it is the spelling both the
        # workstation's transformers and the pod's honour. The observed-dtype
        # check below is what actually enforces the outcome.
        torch_dtype=torch_dtype,
        device_map={"": device},
        output_loading_info=True,
    )
    model, loading_info = unpack_pretrained_loading_info(loaded)
    model.eval()
    require_clean_loading_info(loading_info, arm=name)
    diagnostics = {key: list(loading_info[key]) for key in LOADING_INFO_KEYS}
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
    context = config_context_length(config)
    facts = {
        "rung": name,
        "training_stage": declaration.training_stage,
        "checkpoint": str(resolved),
        "model_type": str(getattr(config, "model_type", "undeclared")),
        "architectures": list(getattr(config, "architectures", []) or []),
        "n_layers": int(config.num_hidden_layers),
        "d_model": int(config.hidden_size),
        "vocab_size": int(config.vocab_size),
        "context": int(context),
        "dtype_requested": dtype,
        "dtype_observed": observed,
        "device": device,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": int(len(tokenizer)),
        "strict_loading_diagnostics": diagnostics,
        "facts_source": (
            "read back from the loaded model's config and parameters, not echoed "
            "from the request"
        ),
    }
    return LoadedRung(
        rung=declaration,
        model=model,
        tokenizer=tokenizer,
        tokenisation=tokenisation,
        context=int(context),
        facts=facts,
    )


# --------------------------------------------------------------- the scorer


class BareBlockScorer:
    """Summed log-likelihood of a sequence's own token run under the bare block.

    The sum is taken over the **scored positions** the rendering declares --
    :func:`src.capability.models.joint_modes.scored_target_positions`' one contiguous run
    whose spellings are exactly the sequence -- and over nothing else. The four
    wrapper tokens are not summed: three of them precede the sequence and are
    constant within an assay, but the closing ``>`` is predicted from the
    sequence and folding it in would put "how confident the model is that the
    protein ended here" inside a fitness score. Restricting to the declared span
    also makes the family's own guard load-bearing rather than decorative: a
    rendering in which a residue merged into a delimiter raises here rather than
    producing a plausible number over the wrong positions.

    ``residues`` and ``scored_tokens`` accumulate across everything this scorer
    is asked for, so the measured residues per scored token that must travel
    beside every magnitude is a property of the run rather than a constant
    quoted from another cohort.
    """

    #: What this scorer computes, and the stratum that convention belongs to.
    #: The stratum is the N-to-C left-to-right sum -- the axis that separates it
    #: from a bidirectional or pseudo-likelihood reading. The *tokenisation* axis
    #: is a separate one and is carried by ``symbol_unit`` below, because a
    #: merged-piece sum and a single-residue sum are both N-to-C and are still
    #: not one estimand.
    score_description = (
        "summed log-likelihood of the residue-spelling token run of the bare "
        "Seq=<...> block, over merged multi-residue SentencePiece pieces"
    )
    scoring_stratum = STRATUM_N_TO_C
    symbol_unit = TOKEN_UNIT

    def __init__(self, loaded: LoadedRung, *, batch_size: int) -> None:
        import torch

        self.torch = torch
        self.loaded = loaded
        self.name = loaded.rung.name
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch size must be positive")
        self.context = loaded.context
        # Padded positions are masked out of attention and are never scored, so
        # the id itself is arbitrary; it is still declared rather than left to a
        # tokenizer that may carry no pad token at all.
        pad = loaded.tokenizer.pad_token_id
        if pad is None:
            pad = loaded.tokenizer.eos_token_id
        self.pad_id = int(pad if pad is not None else 0)
        self.residues = 0
        self.scored_tokens = 0

    # -- rendering -------------------------------------------------------

    def render(self, sequences: Sequence[str]) -> list[RenderedProtein]:
        """Every sequence rendered, tokenised, located and verified."""

        return [self.loaded.tokenisation.render(str(sequence)) for sequence in sequences]

    def token_lengths(self, rendered: Sequence[RenderedProtein]) -> list[int]:
        """The whole rendered length, which is what a context budget is spent on."""

        return [len(record.token_ids) for record in rendered]

    # -- scoring ---------------------------------------------------------

    def log_likelihood(self, rendered: Sequence[RenderedProtein]) -> np.ndarray:
        torch = self.torch
        records = list(rendered)
        if not records:
            raise ValueError("nothing to score")
        totals = np.empty(len(records), dtype=np.float64)
        model = self.loaded.model
        device = model.device
        with torch.no_grad():
            for start in range(0, len(records), self.batch_size):
                chunk = records[start : start + self.batch_size]
                width = max(len(record.token_ids) for record in chunk)
                if width > self.context:
                    raise ValueError(
                        f"{self.name}: a rendering of {width} tokens exceeds this "
                        f"checkpoint's {self.context}-position context; truncating "
                        "would score a sequence that may not contain the mutated "
                        "position"
                    )
                ids = torch.full((len(chunk), width), self.pad_id, dtype=torch.long)
                mask = torch.zeros((len(chunk), width), dtype=torch.long)
                scored = torch.zeros((len(chunk), width), dtype=torch.bool)
                for row, record in enumerate(chunk):
                    length = len(record.token_ids)
                    ids[row, :length] = torch.tensor(record.token_ids, dtype=torch.long)
                    mask[row, :length] = 1
                    scored[row, list(record.scored_positions)] = True
                    self.residues += record.n_residues
                    self.scored_tokens += record.n_scored_tokens
                ids = ids.to(device)
                logits = model(input_ids=ids, attention_mask=mask.to(device)).logits
                logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
                token = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                keep = scored[:, 1:].to(device)
                totals[start : start + len(chunk)] = (
                    (token * keep).sum(1).double().cpu().numpy()
                )
        if not np.all(np.isfinite(totals)):
            raise RuntimeError(f"{self.name}: a scored sequence returned a non-finite total")
        return totals

    # -- accounting ------------------------------------------------------

    def residue_accounting(self) -> dict[str, Any]:
        """The measured residues per scored token, over everything scored so far."""

        if not self.scored_tokens:
            raise RuntimeError("no scored token has been seen, so no rate was measured")
        return {
            "symbol_unit": self.symbol_unit,
            "residues": int(self.residues),
            "scored_tokens": int(self.scored_tokens),
            "residues_per_scored_token": self.residues / self.scored_tokens,
            "note": (
                "measured on this run's own scored queue. One scored symbol here is "
                "one token carrying one or more residues, so a magnitude from this "
                "lineage is in nats per token and is NOT commensurable with a "
                "residue-unit family's (Appendix B rule 26, limitation L23). It is "
                "reported beside every magnitude this campaign publishes"
            ),
        }

    def release(self) -> None:
        del self.loaded
        self.torch.cuda.empty_cache()


def bulk_residues_per_scored_token(records: Iterable[RenderedProtein]) -> float:
    """The same rate over an arbitrary set of renderings, for a CPU-side census."""

    residues = tokens = 0
    for record in records:
        residues += record.n_residues
        tokens += record.n_scored_tokens
    if not tokens:
        raise ValueError("no scored token, so no rate is defined")
    return residues / tokens
