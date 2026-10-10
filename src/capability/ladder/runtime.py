"""Everything on the ladder that touches a checkpoint: loading, sampling, scoring.

One module, because the three operations share the thing that decides whether
any of them is correct -- the arm's **native rendering**. This project has
already paid for getting that wrong: a ProtGPT2 sequence scored as one unwrapped
line costs 1.42 nats/token more than the same sequence in the FASTA layout its
BPE merges were learned over, and ZymCTRL's EC tag leaks 1.73 nats if it is
scored as content instead of as a prompt. Both are larger than any divergence
this experiment could find. So every rendering here is resolved from the audited
declarations -- :meth:`src.capability.core.arms.Cohort.input_strings` for a panel
arm and :mod:`src.capability.models.joint_modes` for the lineage -- and none of
them is spelled a second time in this file.

What a causal window draw is
============================

The prompt is the arm's own rendering of the parent, cut at the first residue of
the window and proved to be a character-exact prefix of it
(:func:`src.capability.ladder.design.causal_prefix_prompt`). The model then
writes forward with no knowledge of what follows, the first ``k`` residues of
what it wrote become the window, and the parent's original suffix is restored.
That restoration is the point: the model's choice was made without the
downstream context, and the asymmetry between that and true infilling is part of
what the ladder measures.

A draw that ends before it has written ``k`` residues has not expressed the
rung. It is recorded with its reason and counted, never quietly replaced by a
shorter window -- a shorter window would be a different rung.

What an infilling draw is
=========================

A bidirectional or absorbing-state arm sees both flanks throughout. The ``k``
window positions are masked and filled by iterative unmasking, one position per
step, committing whichever position's sampled residue the model gave the highest
probability. This is a **declared** member of the family and not a reference
release sampler: the reference ``byprot`` loader has never been executed in this
project, so no numerical A/B supports equivalence to it, and that limitation
travels with every number from these arms.

What the scalars are
====================

A causal arm yields a sequence negative log-likelihood under its own rendering,
summed over the positions that rendering declares scorable. A masked or
diffusion arm has **no** sequence likelihood; its scalar is a sum of masked
marginals, one position masked at a time, which is a pseudo-log-likelihood. The
two are comparable within an arm, which is all the ladder's within-backbone
statistic needs, and they are not comparable across arms, which is why no
cross-arm magnitude is emitted anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import design

SCHEMA_VERSION = "ladder_runtime_v1"

#: How a causal window draw ended.
DRAW_FILLED = "filled"
DRAW_TERMINATED_SHORT = "terminated_short"
DRAW_OFF_ALPHABET = "off_alphabet"
DRAW_BUDGET_EXHAUSTED = "max_new_tokens"

DRAW_STATUSES: tuple[str, ...] = (
    DRAW_FILLED,
    DRAW_TERMINATED_SHORT,
    DRAW_OFF_ALPHABET,
    DRAW_BUDGET_EXHAUSTED,
)

#: Normalisation of the infilling sampler's categorical. Restricted to the twenty
#: canonical residue tokens so that a mask, unknown or special token can never be
#: emitted into a protein. Declared, because the pseudo-log-likelihood below uses
#: the *full* vocabulary and the two are therefore different normalisations used
#: for different purposes.
INFILL_SUPPORT = "aa20_restricted_softmax"
PLL_SUPPORT = "full_vocabulary_softmax"


# ------------------------------------------------------------ causal arms


@dataclass
class CausalArmHandle:
    """A loaded causal arm, with its rendering resolved."""

    name: str
    loader: str
    model: Any
    tokenizer: Any
    panel_arm: Any | None
    loaded_rung: Any | None
    end_delimiter: str
    terminal_marker: str | None
    input_format: str
    facts: dict[str, Any]

    def render(self, sequence: str, *, ec_label: str | None = None) -> str:
        """The arm's native rendering of one complete sequence."""

        if self.loader == "lineage":
            from ..models.joint_modes import rendering

            return rendering(_lineage_rendering_name()).render_protein(sequence)
        from ..core.arms import Cohort

        metadata: dict[str, Any] = {}
        if self.input_format == "ec_conditioned":
            if not ec_label:
                raise ValueError(f"{self.name} renders an EC-conditioned prompt and needs a label")
            metadata["ec_labels"] = [ec_label]
        cohort = Cohort(
            name=f"ladder_{self.name}",
            kind="protein",
            records=[sequence],
            min_symbols=len(sequence),
            max_symbols=len(sequence),
            metadata=metadata,
        )
        return cohort.input_strings(self.panel_arm)[0]


def _lineage_rendering_name() -> str:
    from ..models.joint_lineage import RENDERING_FAMILY

    return RENDERING_FAMILY


def load_causal_arm(name: str, *, device: str, dtype: str = "bfloat16") -> CausalArmHandle:
    """Load one causal ladder arm through the door its declaration names."""

    spec = design.arm(name)
    if spec.likelihood_kind != design.LIKELIHOOD_CAUSAL_NLL:
        raise ValueError(f"{name} is not a causal ladder arm")
    if spec.loader == "lineage":
        from ..models.joint_lineage import load_rung
        from ..models.joint_modes import rendering

        loaded = load_rung(spec.checkpoint, device=device, dtype=dtype)
        family = rendering(_lineage_rendering_name())
        return CausalArmHandle(
            name=name,
            loader="lineage",
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            panel_arm=None,
            loaded_rung=loaded,
            end_delimiter=family.protein_end,
            terminal_marker=family.protein_end,
            input_format="bare_seq_block",
            facts=dict(loaded.facts),
        )
    if spec.loader != "panel":
        raise ValueError(f"{name}: unsupported causal loader {spec.loader!r}")
    from ..core.arms import CONDITIONING_END, load_arm

    panel = load_arm(spec.checkpoint, device=device, dtype=dtype)
    fmt = panel.spec.input_format
    if fmt == "ec_conditioned":
        terminal = CONDITIONING_END
        delimiter = CONDITIONING_END
    else:
        terminal = None
        delimiter = panel.tokenizer.eos_token
        if delimiter is None:
            raise ValueError(f"{name}: no terminal marker is declared and the tokenizer has no eos")
    return CausalArmHandle(
        name=name,
        loader="panel",
        model=panel.model,
        tokenizer=panel.tokenizer,
        panel_arm=panel,
        loaded_rung=None,
        end_delimiter=delimiter,
        terminal_marker=terminal,
        input_format=fmt,
        facts={
            "arm": name,
            "checkpoint": str(panel.spec.path),
            "input_format": fmt,
            "tokenisation": panel.spec.tokenisation,
            "architecture": panel.spec.architecture,
            "pretraining_corpus": panel.spec.pretraining_corpus,
            "dtype_requested": dtype,
        },
    )


def prefix_prompt(handle: CausalArmHandle, parent: str, start: int, *, ec_label: str | None) -> str:
    """The prompt that leaves a causal arm at residue ``start`` of ``parent``."""

    if not 1 <= start < len(parent):
        raise ValueError(f"a prefix prompt needs a strictly interior start; got {start}")
    return design.causal_prefix_prompt(
        handle.render(parent, ec_label=ec_label),
        handle.render(parent[:start], ec_label=ec_label),
        terminal_marker=handle.terminal_marker,
    )


def _outward(radius: int):
    """0, -1, +1, -2, +2, ... up to ``radius``: the nearest boundary first."""

    yield 0
    for step in range(1, int(radius) + 1):
        yield -step
        yield step


def aligned_prefix(
    handle: CausalArmHandle,
    parent: str,
    anchor: int,
    *,
    ec_label: str | None,
    max_extent: int,
    radius: int = design.TOKEN_ALIGNMENT_RADIUS,
) -> dict[str, Any] | None:
    """The nearest window start at which this arm's own tokenisation can be cut.

    A prompt is admissible only when its own tokenisation **is** a prefix of the
    parent's tokenisation. That is the property a mid-token cut breaks, and the
    consequence was measured rather than assumed: ProtGPT2, prompted with a
    prefix whose last token was the dangling single residue ``V``, emitted
    end-of-text on 8 of 8 draws, while the same backbone cut two residues earlier
    filled every draw. A rung built on unaligned cuts measures where the merges
    fell, not modification extent.

    The search walks outward from the declared anchor and takes the first start
    that satisfies the property, so the realised start is the nearest admissible
    one. ``None`` means no admissible start exists inside ``radius``: the arm
    cannot be stopped near this backbone's window, which is recorded as an empty
    cell rather than worked around.
    """

    text = handle.render(parent, ec_label=ec_label)
    parent_ids = list(handle.tokenizer(text)["input_ids"])
    for offset in _outward(radius):
        start = int(anchor) + offset
        if start < 1 or start + int(max_extent) > len(parent) - 1:
            continue
        prompt = design.causal_prefix_prompt(
            text,
            handle.render(parent[:start], ec_label=ec_label),
            terminal_marker=handle.terminal_marker,
        )
        ids = list(handle.tokenizer(prompt)["input_ids"])
        if ids and ids == parent_ids[: len(ids)]:
            return {
                "start": start,
                "prompt": prompt,
                "n_prompt_tokens": len(ids),
                "residues_from_anchor": offset,
                "alignment": "prompt tokenisation is a prefix of the parent tokenisation",
            }
    return None


def classify_draw(continuation: str, residues: str, extent: int, *, end_delimiter: str) -> str:
    """Why a causal window draw ended, from the continuation it produced.

    The order is the information order: a delimiter before the window was filled
    is the model deciding the protein ends there, which is a different outcome
    from wandering off the residue alphabet, which is different again from simply
    running out of budget. Collapsing them would hide which one an arm does at
    which extent, and that is itself a finding about the arm.
    """

    if len(residues) >= extent:
        return DRAW_FILLED
    head = continuation.split(end_delimiter, 1)[0]
    if end_delimiter in continuation:
        return DRAW_TERMINATED_SHORT
    if any(not (character in design.AA20 or character.isspace()) for character in head):
        return DRAW_OFF_ALPHABET
    return DRAW_BUDGET_EXHAUSTED


def sample_window(
    handle: CausalArmHandle,
    *,
    prompt: str,
    extent: int,
    draws: int,
    seed: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    """``draws`` independent causal window draws from one aligned prefix prompt."""

    from ..core.protein_annotations import extract_generated_sequence
    from ..generation.conditioned_generation import sample_continuations

    continuations = sample_continuations(
        handle.model,
        handle.tokenizer,
        prompt,
        n=draws,
        seed=seed,
        batch_size=batch_size,
        max_new_tokens=design.max_new_tokens(extent),
        temperature=design.TEMPERATURE,
        top_p=design.TOP_P,
        top_k=design.TOP_K,
        use_cache=True,
    )
    rows: list[dict[str, Any]] = []
    for index, continuation in enumerate(continuations):
        residues = extract_generated_sequence(continuation, end_delimiter=handle.end_delimiter)
        status = classify_draw(continuation, residues, extent, end_delimiter=handle.end_delimiter)
        rows.append(
            {
                "draw": index,
                "status": status,
                "window": residues[:extent] if status == DRAW_FILLED else "",
                "n_residues_written": len(residues),
                "prompt_sha256": _digest(prompt),
                "prompt_characters": len(prompt),
            }
        )
    return rows


def sample_full(
    handle: CausalArmHandle,
    *,
    draws: int,
    seed: int,
    ec_label: str | None,
    batch_size: int,
) -> list[dict[str, Any]]:
    """The unmatched anchor: whole sequences from the arm's own native prompt.

    The prompt is the arm's rendering with no sequence content at all, which for
    a conditioned arm still carries its class tag -- an EC-conditioned decoder
    has no unconditioned mode that is on its training distribution, and inventing
    one would measure a different model. Length is whatever the arm produces and
    is reported, never matched.
    """

    from ..core.protein_annotations import extract_generated_sequence
    from ..generation.conditioned_generation import sample_continuations

    if handle.input_format == "ec_conditioned":
        if not ec_label:
            raise ValueError(f"{handle.name} needs an EC label even for the anchor")
        from ..core.arms import CONDITIONING_START

        prompt = f"{ec_label}<sep>{CONDITIONING_START}"
    elif handle.loader == "lineage":
        from ..models.joint_modes import rendering

        prompt = rendering(_lineage_rendering_name()).protein_start
    elif handle.input_format == "fasta_wrapped":
        prompt = handle.tokenizer.eos_token + "\n"
    elif handle.input_format == "n_to_c_control":
        from ..core.arms import N_TO_C_MARKER

        prompt = N_TO_C_MARKER
    else:
        raise ValueError(f"{handle.name}: no anchor prompt is declared for {handle.input_format!r}")
    continuations = sample_continuations(
        handle.model,
        handle.tokenizer,
        prompt,
        n=draws,
        seed=seed,
        batch_size=batch_size,
        max_new_tokens=design.FULL_GENERATION_MAX_RESIDUES + 64,
        temperature=design.TEMPERATURE,
        top_p=design.TOP_P,
        top_k=design.TOP_K,
        use_cache=True,
    )
    rows: list[dict[str, Any]] = []
    for index, continuation in enumerate(continuations):
        residues = extract_generated_sequence(continuation, end_delimiter=handle.end_delimiter)
        rows.append(
            {
                "draw": index,
                "status": DRAW_FILLED if residues else DRAW_OFF_ALPHABET,
                "sequence": residues[: design.FULL_GENERATION_MAX_RESIDUES],
                "n_residues_written": len(residues),
                "terminated": handle.end_delimiter in continuation,
                # A product the decoder had not finished when the cap was reached
                # is a censored observation. The cap is recorded per row rather
                # than left to be inferred from a length that equals it.
                "truncated_at_cap": len(residues) > design.FULL_GENERATION_MAX_RESIDUES,
                "length_cap_residues": design.FULL_GENERATION_MAX_RESIDUES,
                "prompt_sha256": _digest(prompt),
            }
        )
    return rows


def causal_sequence_nll(
    handle: CausalArmHandle,
    sequences: Sequence[str],
    *,
    ec_labels: Sequence[str | None] | None = None,
    batch_size: int = 8,
    max_len: int = 1024,
) -> list[dict[str, Any]]:
    """Sequence negative log-likelihood of each sequence under the arm's rendering.

    Panel arms go through the audited target-mask machinery, so the excluded
    marker and boundary spans are the ones the project already verified. The
    lineage goes through its own bare-block scorer, whose scored span is the
    token run that spells the sequence and which refuses a rendering where a
    residue merged into a delimiter.

    A rendering that hits ``max_len`` lost residues from its own tail and its mean
    would be a mean over a prefix, so it raises rather than being recorded.
    """

    import torch

    rows: list[dict[str, Any]] = []
    if handle.loader == "lineage":
        from ..models.joint_lineage import BareBlockScorer

        scorer = BareBlockScorer(handle.loaded_rung, batch_size=batch_size)
        rendered = scorer.render(list(sequences))
        totals = scorer.log_likelihood(rendered)
        for sequence, record, total in zip(sequences, rendered, totals):
            rows.append(
                {
                    "length": len(sequence),
                    "scored_tokens": int(record.n_scored_tokens),
                    "nll_sum_nats": float(-total),
                    "mean_nll_per_token_nats": float(-total) / int(record.n_scored_tokens),
                    "mean_nll_per_residue_nats": float(-total) / len(sequence),
                    "symbol_unit": scorer.symbol_unit,
                }
            )
        return rows

    from ..core import scoring
    from ..core.arms import (
        Cohort,
        conditioning_boundary_ids,
        rendering_marker_ids,
        tokenize_batch,
    )

    labels = list(ec_labels) if ec_labels is not None else [None] * len(sequences)
    if len(labels) != len(sequences):
        raise ValueError("ec_labels must align with sequences")
    metadata: dict[str, Any] = {}
    if handle.input_format == "ec_conditioned":
        if any(not label for label in labels):
            raise ValueError(f"{handle.name} needs an EC label for every scored sequence")
        metadata["ec_labels"] = [str(label) for label in labels]
    cohort = Cohort(
        name=f"ladder_{handle.name}",
        kind="protein",
        records=[str(sequence) for sequence in sequences],
        min_symbols=min(len(sequence) for sequence in sequences),
        max_symbols=max(len(sequence) for sequence in sequences),
        metadata=metadata,
    )
    rendered = cohort.input_strings(handle.panel_arm)
    rule = scoring.target_rule(handle.input_format, ec_conditioning="native")
    start_id, end_id = conditioning_boundary_ids(handle.panel_arm, ec_conditioning="native")
    markers = () if handle.input_format == "ec_conditioned" else rendering_marker_ids(handle.panel_arm)
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            chunk = list(range(start, min(start + batch_size, len(sequences))))
            ids, mask = tokenize_batch(
                handle.panel_arm, [rendered[index] for index in chunk], max_len
            )
            target_mask = scoring.sequence_target_mask(
                ids,
                mask,
                rule=rule,
                start_token_id=start_id,
                end_token_id=end_id,
                marker_token_ids=markers,
            )
            logits = (
                handle.model(
                    input_ids=ids.to(handle.panel_arm.device),
                    attention_mask=mask.to(handle.panel_arm.device),
                )
                .logits.float()
                .cpu()
            )
            per_sequence = scoring.per_sequence_scores(logits, logits, ids, target_mask)
            for offset, index in enumerate(chunk):
                if int(mask[offset].sum()) >= max_len:
                    raise SystemExit(
                        f"{handle.name}: the rendering of sequence {index} filled the "
                        f"{max_len}-token window, so its tail was truncated and its mean "
                        "would be a mean over a prefix"
                    )
                entry = per_sequence[offset]
                tokens = int(entry["token_count"])
                rows.append(
                    {
                        "length": len(sequences[index]),
                        "scored_tokens": tokens,
                        "nll_sum_nats": float(entry["clean_nll_sum"]),
                        "mean_nll_per_token_nats": float(entry["clean_nll_sum"]) / tokens,
                        "mean_nll_per_residue_nats": float(entry["clean_nll_sum"])
                        / len(sequences[index]),
                        "symbol_unit": handle.panel_arm.spec.tokenisation,
                    }
                )
    return rows


# ------------------------------------------------ masked and diffusion arms


@dataclass
class MaskedArmHandle:
    """A loaded bidirectional or absorbing-state arm and its residue token ids."""

    name: str
    model: Any
    tokenizer: Any
    device: str
    mask_id: int
    residue_ids: tuple[int, ...]
    facts: dict[str, Any]


def load_masked_arm(name: str, *, model_root: Path, device: str) -> MaskedArmHandle:
    """Load one masked-LM ladder arm, float32, refusing an imperfect fit.

    Opened the way this project's denoising roster already opens these
    checkpoints -- stock ``EsmForMaskedLM``, float32, zero missing or mismatched
    keys -- so that a scalar read here is the scalar that roster would read.
    """

    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM

    spec = design.arm(name)
    if spec.loader != "masked_lm":
        raise ValueError(f"{name} is not a masked ladder arm")
    directory = Path(model_root) / spec.checkpoint
    if not directory.is_dir():
        raise FileNotFoundError(f"{name}: no checkpoint directory at {directory}")
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    model, info = EsmForMaskedLM.from_pretrained(
        str(directory),
        torch_dtype=torch.float32,
        output_loading_info=True,
        local_files_only=True,
    )
    unmatched = {key: info[key] for key in ("missing_keys", "mismatched_keys") if info[key]}
    if unmatched:
        raise RuntimeError(f"{name}: checkpoint does not fit the graph: {unmatched}")
    model = model.eval().to(device)
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError(f"{name}: the tokenizer declares no mask token")
    residue_ids = []
    for residue in design.AA20:
        token = tokenizer.convert_tokens_to_ids(residue)
        if token is None or int(token) == tokenizer.unk_token_id:
            raise ValueError(f"{name}: residue {residue} has no token of its own")
        residue_ids.append(int(token))
    if len(set(residue_ids)) != len(residue_ids):
        raise ValueError(f"{name}: two residues share a token id")
    return MaskedArmHandle(
        name=name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        mask_id=int(mask_id),
        residue_ids=tuple(residue_ids),
        facts={
            "arm": name,
            "checkpoint": str(directory),
            "dtype": "float32",
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "unexpected_keys": list(info["unexpected_keys"]),
            "reference_loader_executed": False,
            "infill_support": INFILL_SUPPORT,
            "pll_support": PLL_SUPPORT,
            "packing": "residue_tokens_between_cls_and_eos",
        },
    )


def _pack(handle: MaskedArmHandle, sequence: str) -> Any:
    """Token ids of one sequence, proved to be one token per residue."""

    import torch

    ids = [int(value) for value in handle.tokenizer(sequence)["input_ids"]]
    pieces = handle.tokenizer.convert_ids_to_tokens(ids[1 : 1 + len(sequence)])
    if list(pieces) != list(sequence):
        raise ValueError(
            f"{handle.name}: residue tokens do not reconstruct the sequence; this "
            "member requires a one-token-per-residue packing"
        )
    if len(ids) != len(sequence) + 2:
        raise ValueError(f"{handle.name}: unexpected packing width {len(ids)}")
    return torch.as_tensor(ids, dtype=torch.long, device=handle.device)


def _nucleus(logprobs: Any, top_p: float) -> Any:
    """Nucleus truncation of one categorical, in log space."""

    import torch

    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p lies in (0, 1]")
    if top_p == 1.0:
        return logprobs
    ordered, order = torch.sort(logprobs, descending=True)
    cumulative = torch.cumsum(ordered.exp(), dim=-1)
    keep = cumulative - ordered.exp() < top_p
    keep[..., 0] = True
    masked = torch.full_like(logprobs, float("-inf"))
    masked.scatter_(-1, order, torch.where(keep, ordered, torch.full_like(ordered, float("-inf"))))
    return torch.log_softmax(masked, dim=-1)


def infill_window(
    handle: MaskedArmHandle,
    parent: str,
    *,
    start: int,
    extent: int,
    seed: int,
) -> dict[str, Any]:
    """One true-infilling draw: ``extent`` residues written with both flanks visible.

    Iterative unmasking, one position per step. At every step the whole
    partially-filled sequence is re-read, a residue is sampled at each still
    masked position from the AA20-restricted nucleus, and the position whose
    sampled residue carried the highest probability is committed. The remaining
    positions go back to the mask, so each commitment is conditioned on every
    earlier one and on both flanks throughout.

    This is a declared member, not a reference release sampler. It is used
    identically for the masked and the diffusion arm so the pair differs by
    training and not by decoder.
    """

    import torch

    if extent < 1:
        raise ValueError("an extent is at least one residue")
    stop = start + extent
    if not 1 <= start < stop <= len(parent) - 1:
        raise ValueError(f"window [{start}, {stop}) is not strictly interior to the parent")
    ids = _pack(handle, parent)
    pending = list(range(start, stop))
    for position in pending:
        ids[position + 1] = handle.mask_id
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    support = torch.as_tensor(handle.residue_ids, dtype=torch.long, device=handle.device)
    order: list[int] = []
    with torch.no_grad():
        while pending:
            logits = handle.model(input_ids=ids.unsqueeze(0)).logits.float()[0]
            rows = torch.as_tensor([p + 1 for p in pending], dtype=torch.long, device=handle.device)
            restricted = torch.log_softmax(
                logits.index_select(0, rows).index_select(1, support) / design.TEMPERATURE, dim=-1
            )
            truncated = _nucleus(restricted, design.TOP_P)
            probabilities = truncated.exp().cpu()
            sampled = torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)
            chosen = probabilities.gather(1, sampled.unsqueeze(-1)).squeeze(-1)
            best = int(torch.argmax(chosen))
            position = pending.pop(best)
            ids[position + 1] = int(support[int(sampled[best])])
            order.append(position)
    window = "".join(
        handle.tokenizer.convert_ids_to_tokens([int(ids[position + 1])])[0]
        for position in range(start, stop)
    )
    if set(window) - set(design.AA20) or len(window) != extent:
        raise RuntimeError(f"{handle.name}: the infilled window is not {extent} canonical residues")
    return {
        "status": DRAW_FILLED,
        "window": window,
        "commit_order": order,
        "sampler": "one_position_per_step_max_confidence_iterative_unmasking",
        "support": INFILL_SUPPORT,
    }


def masked_pseudo_log_likelihood(
    handle: MaskedArmHandle, sequence: str, *, batch_size: int = 32
) -> dict[str, Any]:
    """Sum of masked marginals over every residue of one sequence.

    One forward per position, that position alone masked, full-vocabulary
    softmax. A masked language model has no sequence log likelihood; this is a
    pseudo-log-likelihood and is labelled as one wherever it appears.
    """

    import torch

    ids = _pack(handle, sequence)
    length = len(sequence)
    terms = np.empty(length, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, length, batch_size):
            stop = min(start + batch_size, length)
            batch = ids.unsqueeze(0).repeat(stop - start, 1)
            for offset, position in enumerate(range(start, stop)):
                batch[offset, position + 1] = handle.mask_id
            logits = handle.model(input_ids=batch).logits.float()
            logprobs = torch.log_softmax(logits, dim=-1)
            for offset, position in enumerate(range(start, stop)):
                terms[position] = float(logprobs[offset, position + 1, int(ids[position + 1])])
    if not np.isfinite(terms).all():
        raise RuntimeError(f"{handle.name}: a non-finite masked marginal was produced")
    total = float(terms.sum())
    return {
        "length": length,
        "scored_tokens": length,
        "nll_sum_nats": -total,
        "mean_nll_per_token_nats": -total / length,
        "mean_nll_per_residue_nats": -total / length,
        "symbol_unit": "residue",
        "likelihood_kind": design.LIKELIHOOD_MASKED_PLL,
        "support": PLL_SUPPORT,
    }


def _digest(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()
