"""What a transfer measurement scores: which depth, which window, which targets.

Four decisions sit upstream of every number this programme reports, and each is
declared exactly once here:

``analysis_layer`` / ``analysis_layers``
    the absolute layer a *relative depth* names on an arm of a given size;
``source_layers_for_target``
    the layers a window of a given width spans, ending at a target layer;
``target_rule`` / ``sequence_target_mask``
    which next-token positions belong to the cohort rather than to the prompt;
``per_sequence_scores`` / ``aggregate_variant``
    how those positions turn into a cross-entropy, a KL and an agreement rate.

Two of these were duplicated before EXP-R2-066, and both duplicates disagreed
with their originals.

**Depth.** ``../d2/src/capability/relational.py`` carried a second depth conversion,
``int(round(fraction * (n_layer - 1)))``. Python's ``round`` is banker's
rounding, so at an exact half it goes to the *even* neighbour while
:func:`analysis_layer`'s ``floor(x + 0.5)`` goes up. On this panel they disagree
for exactly one cell -- the 27-layer ProGen2 arms at relative depth 0.25, where
``0.25 * 26 = 6.5`` is exact in binary: ``analysis_layer`` says layer 7 and the
duplicate said layer 6. ``probe_and_erasure.py`` reached the duplicate and
``02``, ``03`` and ``08`` reached the original, so two stages of one campaign
disagreed about what depth 0.25 meant on the two protein arms carrying the
protein-side corpus contrast. See the EXP-R2-066 log entry for the blast radius.

**Targets.** :func:`sequence_target_mask` used to dispatch on the arm's *name*:
it stripped a conditioning prompt for the literal string ``"zymctrl"`` and, for
every other name, discarded the boundary token ids it was handed without a word.
An EC-conditioned arm named anything else would have had its EC tag, separator
and terminator scored as cohort content -- the positions the model predicts most
confidently -- which lowers clean cross-entropy and raises context information,
with no exception anywhere. ``src/capability/core/budget.py`` carried a guard against
that (``require_boundary_masking_support``), and ``../d2/src/capability/probes.py``
worked around it by synthesising fake names (``f"{arm.name}_plain"``) to select a
rule. The rule is now a parameter, derived from ``ArmSpec.input_format``, so the
guard and the fake names are both gone and a second EC-conditioned arm is handled
correctly by construction rather than refused by an allow-list.

One negative control lives here too, because it is defined entirely in terms of
those targets. :class:`TargetTokenShuffle` permutes exactly the positions
:func:`sequence_target_mask` selects and nothing else, which leaves each record's
target multiset -- and so any unigram baseline taken over it -- exactly
unchanged. It is never applied unless a caller asks for it by name.

Vendored from ``src/revision/dictionary_fidelity.py``, whose remaining contents
(windowed-transcoder encode/decode, checkpoint identity, the frozen fidelity
spec) served the retired CLT / dictionary-qualification scope.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..models.proteinglm import INPUT_FORMAT as PROTEINGLM_INPUT_FORMAT
from ..models.proteinglm import PREFIX_IDS as PROTEINGLM_PREFIX_IDS
from ..models.proteinglm import PREFIX_LENGTH as PROTEINGLM_PREFIX_LENGTH
from ..models.proteinglm import TARGET_RULE as PROTEINGLM_TARGET_RULE


# ------------------------------------------------------------------ depth grid


def analysis_layer(n_layers: int, fraction: float) -> int:
    """The absolute layer index that relative depth ``fraction`` names.

    Round half up, deliberately: ``round()`` is round-half-to-even and turns an
    exact half into a value that depends on the parity of the neighbour, which is
    not a property anyone intends to compare arms on.
    """

    if n_layers < 1 or not 0 <= fraction <= 1:
        raise ValueError("invalid layer count or analysis-layer fraction")
    return int(math.floor(fraction * (n_layers - 1) + 0.5))


def analysis_layers(n_layers: int, fractions: Sequence[float]) -> list[int]:
    """The sorted, deduplicated absolute layers a depth grid names on this arm."""

    if not fractions:
        raise ValueError("at least one layer fraction is required")
    return sorted({analysis_layer(n_layers, float(value)) for value in fractions})


def source_layers_for_target(
    target_layer: int, *, n_layers: int, window: int
) -> tuple[int, ...]:
    """The layers a width-``window`` source window spans, ending at ``target_layer``.

    Clipped at layer 0 rather than shifted, so a window anchored near the bottom
    of a model is narrower than requested instead of covering layers above its
    anchor. Callers that care record both endpoints in the scope name.
    """

    if not 0 <= target_layer < n_layers or window < 1:
        raise ValueError("invalid target layer or decoder window")
    return tuple(range(max(0, target_layer - window + 1), target_layer + 1))


# --------------------------------------------------------------- scored targets

#: ``all_valid``
#:     every next-token position where both the source and the target token are
#:     real rather than padding.
#: ``between_boundaries``
#:     ``all_valid`` intersected with the span strictly between the conditioning
#:     prompt's ``<start>`` and its ``<end>``, so neither the prompt nor the
#:     terminator is scored as cohort content.
#: ``residues_2_to_L``
#:     ProteinGLM continuation: drop prediction columns ``q < 3`` (the three
#:     prefix tokens predicting through AA1) and keep AA2..AAL. That is three
#:     dropped columns, not four.
TARGET_RULES = ("all_valid", "between_boundaries", PROTEINGLM_TARGET_RULE)

#: Input formats whose rendering prepends a conditioning prompt that is *not*
#: cohort content. Derived from ``ArmSpec.input_format`` rather than from an arm
#: name, which is what makes a future EC-conditioned arm correct by default.
CONDITIONED_INPUT_FORMATS = frozenset({"ec_conditioned"})


def target_rule(input_format: str, *, ec_conditioning: str = "native") -> str:
    """The masking rule an arm rendered in ``input_format`` is scored under.

    ``ec_conditioning="unconditioned"`` means the rendering carries no tag and so
    no boundaries; it takes the plain rule even on a conditioned arm. That mode
    is off the arm's training distribution and is measured as such (EXP-R2-034
    prices the tag at 1.73 nats), which is a separate fact from how it is scored.
    """

    if input_format == PROTEINGLM_INPUT_FORMAT:
        return PROTEINGLM_TARGET_RULE
    if input_format in CONDITIONED_INPUT_FORMATS and ec_conditioning != "unconditioned":
        return "between_boundaries"
    return "all_valid"


def sequence_target_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    rule: str,
    start_token_id: int | None = None,
    end_token_id: int | None = None,
    marker_token_ids: Sequence[int] = (),
) -> torch.Tensor:
    """Return a mask over next-token targets belonging to the scored cohort.

    Column ``q`` of the returned mask governs the prediction of input token
    ``q + 1``, so it has one fewer column than ``input_ids``.

    ``marker_token_ids`` are the ids the arm's own rendering declares as markers
    rather than cohort content -- :func:`src.capability.core.arms.rendering_marker_ids`,
    resolved from the same declaration the rendering is built from. A position
    whose *target* is one of them is not content under **any** rule, so the
    exclusion is applied once here rather than per rule or per caller: a
    rendering's prefix is the arm's statement about which positions it added,
    and no rule can revise it. That matters as soon as a rendering prefixes more
    than one token, because only the first token of a batch is context and never
    a target: a one-token marker is excluded by construction, a second one is
    not, and the model predicts it with more confidence than it predicts any
    residue while a pooled unigram baseline prices it as rare.
    """

    if rule not in TARGET_RULES:
        raise ValueError(f"unknown target rule {rule!r}; rules are {list(TARGET_RULES)}")
    if (
        input_ids.ndim != 2
        or attention_mask.shape != input_ids.shape
        or input_ids.shape[1] < 2
    ):
        raise ValueError("invalid token or attention-mask shape")
    valid = attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
    if len(marker_token_ids):
        markers = input_ids.new_tensor(list(marker_token_ids))
        valid = valid & ~torch.isin(input_ids[:, 1:], markers)
    if rule == PROTEINGLM_TARGET_RULE:
        if start_token_id is not None or end_token_id is not None:
            raise ValueError(
                "boundary token ids were supplied under the ProteinGLM residue "
                "rule, which locates content from a fixed prefix rather than EC "
                "boundaries"
            )
        if input_ids.shape[1] < PROTEINGLM_PREFIX_LENGTH + 2:
            raise ValueError(
                "ProteinGLM budget scoring needs the native prefix and two residues"
            )
        prefix = input_ids[:, :PROTEINGLM_PREFIX_LENGTH]
        expected = input_ids.new_tensor(PROTEINGLM_PREFIX_IDS)
        if not bool((prefix == expected).all()):
            raise ValueError(
                "ProteinGLM budget batch does not start with <gmask><sop><eos> ids"
            )
        keep = valid.clone()
        keep[:, :PROTEINGLM_PREFIX_LENGTH] = False
        return keep
    if rule == "all_valid":
        if start_token_id is not None or end_token_id is not None:
            raise ValueError(
                "boundary token ids were supplied under the 'all_valid' rule, which "
                "ignores them; pass rule='between_boundaries' or drop the ids"
            )
        return valid
    if start_token_id is None or end_token_id is None:
        raise ValueError("the 'between_boundaries' rule requires start and end token IDs")
    result = torch.zeros_like(valid)
    for row in range(input_ids.shape[0]):
        ids = input_ids[row]
        starts = torch.nonzero(ids == start_token_id, as_tuple=False).flatten()
        ends = torch.nonzero(ids == end_token_id, as_tuple=False).flatten()
        if starts.numel() != 1 or ends.numel() != 1 or ends[0] <= starts[0] + 1:
            raise ValueError("conditioned prompt lacks one valid start/end boundary")
        # Target column q predicts input token q+1. Keep content targets strictly
        # after <start> and strictly before <end>.
        result[row, int(starts[0]) : int(ends[0]) - 1] = True
    return result & valid


# ------------------------------------------- negative control: shuffled targets

#: Name of the E3 negative control, spelled once. It travels in every artefact a
#: control run writes and is what a reader greps for to be sure a report is not a
#: measurement.
TOKEN_SHUFFLE_CONTROL = "within_record_token_shuffle"


@dataclass(frozen=True)
class TargetTokenShuffle:
    """A seeded within-record permutation of exactly the scored target positions.

    E3 of ``archive/docs/R5/context-measurement-protocols.md`` asks for an
    arm whose true context information is known to be small, because neither the
    0.30 nats/token floor nor the sign criterion has ever been exercised near its
    boundary: the panel jumps from -4.08 nats on ``dialogpt-small`` to +1.06 on
    ``progen2-base`` with nothing in between. Permuting a record's target tokens
    destroys the sequential structure a model could read while leaving that
    record's target multiset untouched, so ``H_baseline`` -- a unigram
    cross-entropy over exactly those targets -- is unchanged to the last bit and
    only ``H_model`` moves.

    **Which positions move.** Exactly the positions :func:`sequence_target_mask`
    scores, and nothing else. Column ``q`` of that mask governs the prediction of
    input token ``q + 1``, so the permuted set is ``{q + 1 : mask[q]}``: under
    ``all_valid`` every real token except the first, and under
    ``between_boundaries`` the span strictly between ``<start>`` and ``<end>``.
    The first token of a plain rendering, an EC-conditioned rendering's tag,
    ``<sep>`` and both boundary markers, and every padding position therefore
    stay where they are. Two consequences are the whole point: the per-record
    target multiset is invariant, and the mask recomputed after the permutation
    selects the identical positions, because nothing it reads has moved. No token
    crosses a boundary in either direction.

    **One quantity downstream of the multiset is nonetheless not bit-invariant.**
    A per-symbol conversion counts the *characters* the scored ids decode to, and
    a byte-level BPE can split one multi-byte character across two tokens: reorder
    them and those bytes no longer form that character, so the decoder emits
    replacement characters instead. Measured on gpt2 over 24 OpenWebText records,
    ``symbols_per_token`` moves from 4.4699 to 4.4831 -- three parts in a
    thousand -- and every ``*_bits_per_symbol`` field moves with it. Residue-level
    arms and ProtGPT2 are unaffected, since their pieces decode to whole ASCII
    characters, and no nats-per-token figure is touched on any arm.

    **Where it is applied.** After truncation to ``max_len`` and after padding,
    in :func:`src.capability.core.arms.tokenize_batch`. Permuting before truncation
    would change *which* tokens survive the window and therefore the multiset,
    which is the one thing this control may not do.

    **Reproducibility.** ``seed`` is the run's declared seed; the permutation of
    one record is drawn from it combined with a digest of that record's own
    unpadded token ids. It therefore does not depend on the batch size, on the
    record's position in the cohort, or on the padding width it was batched with,
    and re-running at the same seed reproduces every permutation. Two records
    with identical token ids receive identical permutations, which is what
    keying by content means.

    **What this control establishes, and what it does not.** It bounds what a
    predictor with no usable sequential context achieves *on shuffled input*.
    That is not the same thing as a model whose true context information is zero.
    Shuffled text is off the training distribution in ways beyond the loss of
    context, so what the model produces at a scored position is not its marginal
    token distribution: it can be worse than a held-out unigram, which reads as a
    negative context information of no fixed size, and it is not guaranteed to
    land near the floor the criteria are being tested against. The control's
    value is that the criteria and the bootstrap are then exercised at whatever
    value it does land on, instead of four nats from any boundary; the value
    itself must be read as a measurement of this control and not as a zero. It is
    also silent about the smoothing bias recorded separately in the
    pre-registration: a genuinely context-free predictor still reads a positive
    ``I`` of order ``log(1 + alpha V / R)``.
    """

    seed: int
    rule: str
    start_token_id: int | None = None
    end_token_id: int | None = None
    marker_token_ids: tuple[int, ...] = ()

    def apply(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """``input_ids`` with each record's scored target positions permuted.

        The mask is recomputed from the arguments rather than taken from the
        caller, so the permuted set is by construction the set this module
        scores; a caller cannot hand in a different one.
        """

        mask = sequence_target_mask(
            input_ids,
            attention_mask,
            rule=self.rule,
            start_token_id=self.start_token_id,
            end_token_id=self.end_token_id,
            marker_token_ids=self.marker_token_ids,
        )
        shuffled = input_ids.clone()
        for row in range(input_ids.shape[0]):
            positions = torch.nonzero(mask[row], as_tuple=False).flatten() + 1
            if positions.numel() < 2:
                continue
            content = input_ids[row][attention_mask[row].bool()]
            order = self._permutation(content, int(positions.numel()))
            shuffled[row, positions] = input_ids[row, positions[order]]
        return shuffled

    def record(self) -> dict[str, object]:
        """The block an artefact carries so a control run cannot read as a measurement.

        Written into every artefact the control produces, beside the seed that
        reproduces it.
        """

        return {
            "control": TOKEN_SHUFFLE_CONTROL,
            "seed": int(self.seed),
            "target_rule": self.rule,
            "start_token_id": self.start_token_id,
            "end_token_id": self.end_token_id,
            "marker_token_ids": list(self.marker_token_ids),
            "permuted_positions": (
                "exactly the scored next-token target positions of each record, "
                "after truncation and padding; the first token of a plain "
                "rendering, a conditioning prompt and its <start>/<end> markers, "
                "every declared rendering marker, and every padding position "
                "are held in place"
            ),
            "invariant": (
                "the per-record target-token multiset, and therefore every "
                "unigram baseline taken over it"
            ),
            "permutation_key": (
                "seed combined with a blake2b-64 digest of the record's own "
                "unpadded token ids, so the permutation is independent of batch "
                "size and cohort order"
            ),
            "reads_as": (
                "a bound on what a predictor with no usable sequential context "
                "achieves on shuffled input, which is not the same as a model "
                "whose true context information is zero: shuffled input is off "
                "the training distribution and the measured value may sit either "
                "side of zero"
            ),
        }

    def _permutation(self, record_ids: torch.Tensor, count: int) -> torch.Tensor:
        digest = hashlib.blake2b(
            record_ids.detach().to("cpu").numpy().astype("<i8").tobytes(), digest_size=8
        ).digest()
        generator = np.random.default_rng([int(self.seed), int.from_bytes(digest, "big")])
        return torch.as_tensor(
            generator.permutation(count).astype(np.int64), device=record_ids.device
        )


# ------------------------------------------------------------------- scoring


def per_sequence_scores(
    clean_logits: torch.Tensor,
    variant_logits: torch.Tensor,
    input_ids: torch.Tensor,
    target_mask: torch.Tensor,
) -> list[dict[str, float | int]]:
    """Per-sequence sums, so the aggregate can be token-weighted downstream.

    Sums rather than means: a per-sequence mean averaged over sequences weights a
    40-residue protein like a 2000-residue one, and the panel's cohorts are not
    length-matched across arms.
    """

    if (
        clean_logits.shape != variant_logits.shape
        or clean_logits.ndim != 3
        or input_ids.shape != clean_logits.shape[:2]
        or target_mask.shape != (input_ids.shape[0], input_ids.shape[1] - 1)
    ):
        raise ValueError("logit, token and target-mask shapes disagree")
    clean_logp = F.log_softmax(clean_logits[:, :-1].float(), dim=-1)
    variant_logp = F.log_softmax(variant_logits[:, :-1].float(), dim=-1)
    clean_p = clean_logp.exp()
    targets = input_ids[:, 1:]
    clean_nll = -clean_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    variant_nll = -variant_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    kl = (clean_p * (clean_logp - variant_logp)).sum(-1)
    agreement = clean_logp.argmax(-1) == variant_logp.argmax(-1)
    rows: list[dict[str, float | int]] = []
    for index in range(input_ids.shape[0]):
        mask = target_mask[index]
        count = int(mask.sum())
        if count < 1:
            raise ValueError("sequence has no scored next-token targets")
        rows.append(
            {
                "token_count": count,
                "clean_nll_sum": float(clean_nll[index][mask].sum()),
                "variant_nll_sum": float(variant_nll[index][mask].sum()),
                "kl_sum": float(kl[index][mask].sum()),
                "argmax_agreement_count": int(agreement[index][mask].sum()),
            }
        )
    return rows


def aggregate_variant(rows: Sequence[Mapping[str, float | int]]) -> dict[str, float]:
    """Token-weighted aggregate over :func:`per_sequence_scores` rows."""

    if not rows:
        raise ValueError("cannot aggregate an empty sequence set")
    tokens = sum(int(row["token_count"]) for row in rows)
    if tokens < 1:
        raise ValueError("aggregate contains no scored targets")
    return {
        "clean_ce_nats": sum(float(row["clean_nll_sum"]) for row in rows) / tokens,
        "variant_ce_nats": sum(float(row["variant_nll_sum"]) for row in rows) / tokens,
        "clean_to_variant_kl_nats": sum(float(row["kl_sum"]) for row in rows) / tokens,
        "argmax_agreement": sum(int(row["argmax_agreement_count"]) for row in rows)
        / tokens,
        "scored_tokens": tokens,
        "sequences": len(rows),
    }
