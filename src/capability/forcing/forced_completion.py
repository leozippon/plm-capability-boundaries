"""Forcing one residue of a natural prefix and reading what the model emits after it.

The intervention is at the token grid, so the token grid is checked before any
measurement is taken. Three properties have to hold for an arm to carry a
position-level forcing intervention at all:

1. every canonical residue is a single token, so "the residue at position ``p``"
   names one logit column;
2. the arm's own rendering of a prefix is the prefix of its rendering of the
   whole, so forcing position ``i`` changes position ``i`` and nothing before it;
3. the rendering places a fixed, known number of marker tokens in front of the
   content, so a residue position maps to a token index arithmetically.

:func:`residue_grid` measures all three on the loaded checkpoint and refuses an
arm that fails any of them. ProtGPT2 fails (1) and (2) -- measured on the staged
checkpoint, 29 ids for 78 residues and a prefix that retokenises -- which is why
a multi-residue BPE arm is outside this design rather than inside it with a
caveat.

What a cell produces. For one unit and one condition the sampled mode emits
``DRAWS_PER_CELL`` continuations of ``i+1 .. span_end``, where ``span_end`` is the
furthest read position the unit needs. Generation stops there: the gate never
needs a whole protein, which is what makes it cheap. Each draw carries the
residue it realised at every read position, the warped sampling distribution the
model used at those positions, and its own composition and repeat flags. The
teacher-forced mode fixes the intervening residues to wild type and reads the
same conditionals exactly from one forward pass, which costs one pass per cell
and is the design's free secondary decomposition.

Censoring is defined once and arm-agnostically: the first generated token that is
not one of the twenty canonical residues **terminates** the completion, and every
read position at or after it is censored. That covers an end delimiter, an
end-of-text token, a pad token and a non-canonical residue under one rule, and it
does not require each arm to declare which of its tokens means "done". The
terminating token is recorded, so a reader can see what the model actually
emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.amino_acids import AA20
from ..core.arms import (
    BOS_DIRECTION_N_TO_C,
    INPUT_FORMAT_BOS_DIRECTION_SEQ,
    INPUT_FORMAT_EOS_BOUNDED_SEQ,
    N_TO_C_MARKER,
    eos_bounded_rendering,
)
from ..interactions.contact_enrichment import HYDROPHOBIC
from . import forcing_design as D
from .backbone_cohort import distinct_shingle_fraction

SCHEMA = "forcing_forced_completion_v1"

#: The renderings a prefix is defined for, with the rendering function. Each one
#: resolves the marker string the panel already declares rather than spelling it
#: again; an arm whose format is absent here is refused by name.
PREFIX_RENDERINGS: tuple[str, ...] = (
    "n_to_c_control",
    INPUT_FORMAT_EOS_BOUNDED_SEQ,
    INPUT_FORMAT_BOS_DIRECTION_SEQ,
)

PREFIX_RENDERING_REFUSAL = (
    "a prefix is only defined for a rendering that prefixes markers to the bare "
    "residue run. A rendering that wraps the sequence -- protgpt2's 60-column "
    "FASTA, zymctrl's <start>/<end> pair, proteinglm's gmask/sop/eos -- has no "
    "prefix in this sense, because the wrapper's tail is part of the trained "
    "input and removing it puts the model off distribution"
)

#: Length of the probe the token-grid gate is measured on, and the prefix cuts it
#: is checked at. Long enough that a BPE tokeniser would have merged, short
#: enough to cost nothing.
GRID_PROBE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
GRID_PROBE_CUTS = (10, 30, 50)

#: What produced the draws, recorded with every cell. One sampler serves both
#: modes, which is what makes the sampled and teacher-forced endpoints a
#: comparison under one decoding policy rather than under two implementations of
#: it.
SAMPLER_NOTE = (
    "explicit per-step sampler: a full forward over the sequence so far, the "
    "declared policy applied by forcing.forced_completion.warp, and a multinomial "
    "draw over the full vocabulary from a NumPy generator seeded by the campaign "
    "seed. The full vocabulary rather than the residues alone, so that early "
    "termination stays a measurable outcome instead of being suppressed"
)

def render_prefix(arm: Any, residues: str) -> str:
    """One arm's native rendering of a residue prefix, through the panel's declaration."""

    fmt = arm.spec.input_format
    if fmt == "n_to_c_control":
        return N_TO_C_MARKER + str(residues)
    if fmt == INPUT_FORMAT_EOS_BOUNDED_SEQ:
        return eos_bounded_rendering(str(residues))
    if fmt == INPUT_FORMAT_BOS_DIRECTION_SEQ:
        bos = arm.tokenizer.bos_token
        if bos is None:
            raise ValueError(
                f"{arm.name}: its rendering places a BOS before the direction token, but "
                "the tokenizer declares none"
            )
        return f"{bos}{BOS_DIRECTION_N_TO_C}{residues}"
    raise ValueError(
        f"{arm.name}: input format {fmt!r} has no prefix rendering. "
        f"{PREFIX_RENDERING_REFUSAL}. Declared: {PREFIX_RENDERINGS}"
    )


@dataclass(frozen=True)
class ResidueGrid:
    """The measured token grid of one admissible arm.

    ``markers`` is how many token ids the rendering puts in front of the content,
    so the id that carries residue position ``p`` is at index ``markers + p`` and
    the logit row that predicts it is at index ``markers + p - 1``.
    """

    arm: str
    markers: int
    residue_ids: tuple[int, ...]
    evidence: Mapping[str, Any]

    @property
    def residue_of_id(self) -> dict[int, str]:
        return {token: residue for residue, token in zip(AA20, self.residue_ids)}


def residue_grid(arm: Any) -> ResidueGrid:
    """Measure the three properties a forcing intervention needs, or refuse the arm.

    Nothing here is assumed from the arm's declared ``tokenisation`` field: a
    declaration says what the authors called it, and this design needs the grid
    the loaded tokenizer actually produces.
    """

    tokenizer = arm.tokenizer
    multi = {}
    for residue in AA20:
        ids = tokenizer(residue, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            multi[residue] = list(ids)
    if multi:
        raise ValueError(
            f"{arm.name}: {len(multi)} canonical residue(s) are not a single token "
            f"({multi}); a position-level forcing intervention is undefined on a "
            "multi-residue tokenisation and this arm is refused rather than approximated"
        )
    residue_ids = tuple(
        int(tokenizer(residue, add_special_tokens=False)["input_ids"][0]) for residue in AA20
    )
    if len(set(residue_ids)) != len(AA20):
        raise ValueError(f"{arm.name}: two canonical residues share one token id")
    round_trip = {
        residue: tokenizer.decode([token])
        for residue, token in zip(AA20, residue_ids)
        if tokenizer.decode([token]) != residue
    }
    if round_trip:
        raise ValueError(f"{arm.name}: residue ids do not decode back to their residue: {round_trip}")

    full = tokenizer(render_prefix(arm, GRID_PROBE), add_special_tokens=False)["input_ids"]
    markers = len(full) - len(GRID_PROBE)
    if markers < 0:
        raise ValueError(
            f"{arm.name}: its rendering of {len(GRID_PROBE)} residues is {len(full)} tokens, "
            "which is fewer than one token per residue"
        )
    if list(full[markers:]) != list(residue_ids_of(residue_ids, GRID_PROBE)):
        raise ValueError(
            f"{arm.name}: the content ids of a rendered prefix are not the per-residue ids; "
            "the rendering is not one token per residue after a fixed marker run"
        )
    for cut in GRID_PROBE_CUTS:
        cut_ids = tokenizer(render_prefix(arm, GRID_PROBE[:cut]), add_special_tokens=False)["input_ids"]
        if list(cut_ids) != list(full[: markers + cut]):
            raise ValueError(
                f"{arm.name}: its tokenisation of a {cut}-residue prefix is not the prefix of "
                "its tokenisation of the whole, so forcing a position would also change the "
                "positions before it"
            )
    return ResidueGrid(
        arm=arm.name,
        markers=int(markers),
        residue_ids=residue_ids,
        evidence={
            "schema": SCHEMA,
            "input_format": arm.spec.input_format,
            "marker_tokens": int(markers),
            "marker_ids": [int(value) for value in full[:markers]],
            "probe_residues": len(GRID_PROBE),
            "probe_ids": len(full),
            "prefix_cuts_checked": list(GRID_PROBE_CUTS),
            "declared_tokenisation": arm.spec.tokenisation,
            "measured": (
                "every canonical residue is one token and decodes back to itself; the "
                "rendered content ids are exactly the per-residue ids after a fixed marker "
                "run; and the rendering of a prefix is the prefix of the rendering of the "
                "whole at every checked cut"
            ),
        },
    )


def residue_ids_of(residue_ids: Sequence[int], residues: str) -> list[int]:
    table = dict(zip(AA20, residue_ids))
    missing = sorted(set(residues) - set(table))
    if missing:
        raise ValueError(f"non-canonical residue(s) {missing} have no token id")
    return [table[residue] for residue in residues]


# --------------------------------------------------------------- the warping


def warp(logits: np.ndarray, *, top_p: float = D.TOP_P, temperature: float = D.TEMPERATURE,
         top_k: int = D.TOP_K) -> np.ndarray:
    """The declared decoding policy's sampling distribution, over the full vocabulary.

    The policy is temperature 1.0, top-p 0.95, top-k off, so the only operative
    step is nucleus truncation; temperature and top-k are applied anyway rather
    than assumed away, so a future change to the declared policy cannot leave
    this path silently implementing the old one. The result is the distribution
    the sampler draws from, which is what "the realised residue distribution"
    means.
    """

    row = np.asarray(logits, dtype=np.float64)
    if row.ndim != 1 or not np.isfinite(row).all():
        raise ValueError("a logit row is one-dimensional and finite")
    if temperature <= 0:
        raise ValueError("temperature is positive")
    row = row / float(temperature)
    if int(top_k) > 0:
        threshold = np.sort(row)[-int(top_k)]
        row = np.where(row < threshold, -np.inf, row)
    shifted = row - row.max()
    probability = np.exp(shifted)
    probability /= probability.sum()
    if 0.0 < float(top_p) < 1.0:
        order = np.argsort(-probability, kind="stable")
        cumulative = np.cumsum(probability[order])
        # Transformers' nucleus rule keeps every token up to and including the one
        # that crosses the threshold, so the retained mass is never below it. The
        # cutoff is then applied by PROBABILITY rather than by rank, which is the
        # one place this implementation deliberately differs from the library's.
        #
        # Measured on a real cell: a bfloat16 checkpoint quantises many logits to
        # exactly equal values, so several tokens routinely tie at the nucleus
        # boundary, and `torch.sort` and `np.argsort` break those ties differently.
        # On one progen2-medium row both rules kept eighteen tokens of identical
        # total mass but disagreed on which two of several exactly-equal-probability
        # tokens were in. A truncation whose output depends on sort order among
        # equal probabilities is not a well-defined rule, so this one keeps every
        # token tied with the last member of the nucleus. It is deterministic,
        # permutation-invariant, and a superset of the library's nucleus.
        cutoff = float(probability[order[int(np.searchsorted(cumulative, float(top_p)))]])
        probability = np.where(probability >= cutoff, probability, 0.0)
        probability /= probability.sum()
    return probability


def residue_distribution(probability: np.ndarray, grid: ResidueGrid) -> np.ndarray:
    """A full-vocabulary distribution restricted and renormalised to the twenty residues.

    Restriction is what makes two arms with different vocabularies comparable at
    all, and renormalisation is what keeps the statistic inside the residue
    alphabet: an arm that places mass on tokens that are not residues does not
    move this quantity.
    """

    mass = np.asarray(probability, dtype=np.float64)[list(grid.residue_ids)]
    total = mass.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("the decoding policy left no mass on any canonical residue")
    return mass / total


# --------------------------------------------------------------- the sampled cell


def completion_flags(completion: str) -> dict[str, Any]:
    """Composition and repeat descriptors of one completion, used for binning."""

    length = len(completion)
    hydrophobic = sum(1 for residue in completion if residue in HYDROPHOBIC)
    fraction = distinct_shingle_fraction(completion)
    return {
        "length": length,
        "hydrophobic_fraction": (hydrophobic / length) if length else None,
        "distinct_shingle_fraction": fraction,
        "repeat_flagged": bool(length >= 5 and fraction < D.MIN_DISTINCT_SHINGLE_FRACTION),
    }


def sample_cell(
    arm: Any, grid: ResidueGrid, *, wildtype: str, anchor: int, forced_residue: str,
    span_end: int, read_positions: Sequence[int], draws: int, seed: int,
    batch_size: int = 24,
) -> dict[str, Any]:
    """``draws`` sampled completions of ``anchor+1 .. span_end`` under one forced anchor.

    The prompt is the natural prefix ``w_{<anchor}`` with the forced residue
    appended, rendered the arm's own way. The forced residue is part of the prompt
    and is never scored.

    The sampling loop is explicit rather than delegated to ``generate``, for three
    measured reasons and one scientific one. Measured, on the staged checkpoints:
    ProGen2's released modelling code carries the legacy tuple cache that current
    Transformers no longer builds, so ``generate`` raises on
    ``config.num_hidden_layers`` before a token is produced; RITA's does not
    inherit ``generate`` at all; and RITA's ``forward`` rejects the
    ``cache_position`` the current generation loop always passes. Three
    incompatibilities in three arms is a reason to stop adapting to the library
    rather than to adapt three times. Scientific: the teacher-forced mode reads a
    conditional ``generate`` never produced, so with ``generate`` in the loop the
    two modes decode under two implementations of one policy -- and they were
    measured to disagree, by up to 0.026 in probability, wherever a bfloat16
    checkpoint quantises logits to equal values at the nucleus boundary. One
    sampler, applied by :func:`warp` in both modes, removes that question instead
    of bounding it.

    Each step is a full forward over the sequence so far, which is exactly what
    ``generate`` with the cache off already did for these arms; nothing is slower
    for being explicit.

    Sampling is a NumPy generator seeded from the campaign seed, so a draw depends
    on the seed and the order of consumption alone rather than on device-side
    random state.
    """

    import torch

    if forced_residue not in AA20:
        raise ValueError(f"the forced residue must be canonical, got {forced_residue!r}")
    if not 0 <= anchor < len(wildtype):
        raise ValueError("the anchor lies inside the wild type")
    if span_end <= anchor or span_end >= len(wildtype):
        raise ValueError("the span ends after the anchor and inside the wild type")
    positions = [int(value) for value in read_positions]
    if any(not anchor < position <= span_end for position in positions):
        raise ValueError("every read position lies after the anchor and at or before the span end")
    if draws < 1 or batch_size < 1:
        raise ValueError("a cell needs a positive draw count and batch size")

    prompt = render_prefix(arm, wildtype[:anchor] + forced_residue)
    encoded = arm.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    prompt_ids = encoded["input_ids"]
    expected = grid.markers + anchor + 1
    if int(prompt_ids.shape[1]) != expected:
        raise ValueError(
            f"{arm.name}: the forced prompt is {int(prompt_ids.shape[1])} tokens where the "
            f"grid predicts {expected}; the token grid and the prompt disagree"
        )
    prompt_ids = prompt_ids.to(arm.model.device)
    steps = span_end - anchor
    residue_of_id = grid.residue_of_id
    generator = np.random.default_rng(int(seed))
    slot_of_position = {position: slot for slot, position in enumerate(positions)}

    rows: list[dict[str, Any]] = []
    distributions = np.zeros((draws, len(positions), len(AA20)), dtype=np.float32)
    with torch.no_grad():
        while len(rows) < draws:
            size = min(int(batch_size), draws - len(rows))
            base = len(rows)
            ids = prompt_ids.repeat(size, 1)
            emitted: list[list[str]] = [[] for _ in range(size)]
            terminator: list[dict[str, Any] | None] = [None] * size
            for step in range(steps):
                if all(entry is not None for entry in terminator):
                    break
                logits = arm.model(input_ids=ids).logits[:, -1, :].float().to("cpu").numpy()
                position = anchor + 1 + step
                chosen = np.empty(size, dtype=np.int64)
                for member in range(size):
                    probability = warp(logits[member])
                    chosen[member] = int(
                        generator.choice(probability.size, p=probability)
                    )
                    if terminator[member] is not None:
                        continue
                    residue = residue_of_id.get(int(chosen[member]))
                    if residue is None:
                        terminator[member] = {
                            "step": step,
                            "position": position,
                            "token_id": int(chosen[member]),
                            "token": arm.tokenizer.decode([int(chosen[member])]),
                        }
                        continue
                    emitted[member].append(residue)
                    if position in slot_of_position:
                        distributions[base + member, slot_of_position[position]] = (
                            residue_distribution(probability, grid).astype(np.float32)
                        )
                ids = torch.cat(
                    [ids, torch.tensor(chosen, dtype=ids.dtype, device=ids.device)[:, None]],
                    dim=1,
                )
            for member in range(size):
                completion = "".join(emitted[member])
                realised = {
                    str(position): (
                        completion[position - anchor - 1]
                        if position - anchor - 1 < len(completion) else None
                    )
                    for position in positions
                }
                rows.append(
                    {
                        "draw": base + member,
                        "completion": completion,
                        "emitted_residues": len(completion),
                        "requested_residues": steps,
                        "censored": terminator[member] is not None,
                        "terminator": terminator[member],
                        "realised": realised,
                        **completion_flags(completion),
                    }
                )
    return {
        "schema": SCHEMA,
        "mode": D.MODE_SAMPLED,
        "arm": arm.name,
        "anchor": int(anchor),
        "forced_residue": forced_residue,
        "span_end": int(span_end),
        "read_positions": positions,
        "prompt_tokens": int(prompt_ids.shape[1]),
        "new_tokens": int(steps),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "sampler": SAMPLER_NOTE,
        "draws": rows,
        "distributions": distributions,
        "censored_draws": sum(1 for row in rows if row["censored"]),
    }


def teacher_forced_cell(
    arm: Any, grid: ResidueGrid, *, wildtype: str, anchor: int, forced_residue: str,
    span_end: int, read_positions: Sequence[int],
) -> dict[str, Any]:
    """The exact conditional at every read position with the intervening residues fixed.

    One forward pass over the wild type with position ``anchor`` replaced by the
    forced residue gives the conditional at every later position simultaneously,
    because every intervening residue is known. This is the design's free
    secondary decomposition: it costs one pass per cell and it is exact, so the
    sampled-minus-teacher-forced difference is what the model's own intervening
    commitments contribute.
    """

    import torch

    if forced_residue not in AA20:
        raise ValueError(f"the forced residue must be canonical, got {forced_residue!r}")
    positions = [int(value) for value in read_positions]
    if any(not anchor < position <= span_end for position in positions):
        raise ValueError("every read position lies after the anchor and at or before the span end")
    forced = wildtype[:anchor] + forced_residue + wildtype[anchor + 1: span_end + 1]
    prompt = render_prefix(arm, forced)
    encoded = arm.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    ids = encoded["input_ids"]
    if int(ids.shape[1]) != grid.markers + len(forced):
        raise ValueError(
            f"{arm.name}: the teacher-forced prompt is {int(ids.shape[1])} tokens where the "
            f"grid predicts {grid.markers + len(forced)}"
        )
    with torch.no_grad():
        logits = arm.model(input_ids=ids.to(arm.model.device)).logits[0].float().to("cpu").numpy()
    distributions = np.zeros((len(positions), len(AA20)), dtype=np.float32)
    for slot, position in enumerate(positions):
        row = grid.markers + position - 1
        if not 0 <= row < logits.shape[0]:
            raise ValueError(f"position {position} has no predicting logit row")
        distributions[slot] = residue_distribution(warp(logits[row]), grid).astype(np.float32)
    return {
        "schema": SCHEMA,
        "mode": D.MODE_TEACHER_FORCED,
        "arm": arm.name,
        "anchor": int(anchor),
        "forced_residue": forced_residue,
        "span_end": int(span_end),
        "read_positions": positions,
        "prompt_tokens": int(ids.shape[1]),
        "distributions": distributions,
        "wild_type_residues": {
            str(position): wildtype[position] for position in positions
        },
    }
