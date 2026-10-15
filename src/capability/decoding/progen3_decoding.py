"""ProGen3 behind E20's guided decoder: the interface adapter, and nothing else.

Why this module exists
======================

:mod:`src.capability.decoding.guided_decoding` is deliberately model-agnostic. It
derives the character layout from the arm's own rendering, calls
``arm.model(input_ids=..., past_key_values=..., use_cache=True)`` and reads the
arm's tokenizer through the Hugging Face surface. ProGen3 satisfies none of those
three by itself, for reasons that are properties of the checkpoint rather than
oversights:

``position_ids`` and ``sequence_ids`` are required arguments.
    ``ProGen3ForCausalLM.forward`` declares both positionally with no default, and
    ``sequence_ids`` indexes a learned embedding that is added to every token
    embedding. A generic decoder call raises ``TypeError``; a call that guessed
    them would be a silent wrong answer (see ``_forward_progen3`` in
    ``scripts/capability/stages/context_homologue.py``, whose withdrawn Stage-46
    measurement is exactly that defect). :class:`ProGen3DecodingModel` supplies
    both from the native convention and tracks the cache offset.

The tokenizer is a ``tokenizers.Tokenizer``, not a ``PreTrainedTokenizer``.
    It is not callable, has ``decode_batch`` rather than ``batch_decode``, and its
    ``decode`` takes no ``clean_up_tokenization_spaces``. It also adds no
    ``<bos>``: ProGen3's ``tokenizer.json`` carries a ByteLevel post-processor, not
    a template, so the opening special comes from the batch preparer rather than
    from ``add_special_tokens``. :class:`ProGen3DecodingTokenizer` presents the
    four calls the decoder makes.

Its ``ArmSpec.input_format`` is ``raw``, which does not describe it.
    ``context_homologue.packing_of`` already records that: "ProGen3 declares
    ``raw``, which is GPT-2's EOT-joined stream and is the wrong packing", and
    every consumer that cares dispatches on :data:`PACKING_PROGEN3` instead. The
    decoder, however, reads the rendering through ``Cohort.input_strings``, so an
    unadapted ProGen3 arm would have its layout derived from the one declaration
    the repository has already marked as not describing it. The adapter gives the
    decoding arm a spec view whose ``input_format`` is ``n_to_c_control`` and
    *checks* that view against :func:`generation_rendering`, which is derived from
    the checkpoint's own ``ProGen3BatchPreparer``. The two lineages share the
    ``"1"`` marker, so the strings coincide -- but they coincide as a verified
    identity rather than as a borrowing, and the check fails loudly if either side
    moves.

The native layout, and where a Potts column lands
=================================================

``ProGen3BatchPreparer.prepare_clm`` renders one CLM record as

    ``<bos>`` ``1`` r(0) r(1) ... r(L-1) ``2`` ``<eos>``

one token per residue, and ``get_generation_kwargs`` conditions generation on the
first ``len - 2`` tokens -- that is, on ``<bos> 1`` alone for an empty prompt.
Both directions are rendered as this same **forward** stream: C-to-N reverses the
residue string and swaps the marker pair to ``2 ... 1``, so there is no backward
pass and no second layout. This programme samples the N-to-C stream only, which
is also the single direction the project scores ProGen3 on.

Residue ``t`` therefore sits at absolute token index ``t + 2``, the logits column
that governs it is ``t + 1``, and after the two-token prompt the decoder's step
``t`` draws residue ``t``. Potts column ``t`` maps to decode step ``t`` with no
shift, and no terminal or direction token sits between them: the layout stops at
``L`` residues and the ``2 <eos>`` tail is never emitted, exactly as ProGen2's
``<|eos|>`` is never emitted.

What this module does not do
============================

It does not load the checkpoint itself -- :func:`load_progen3` is the only
supported way in, and this wraps what that returns -- and it declares no dose,
domain or protocol. The dtype is the loader's: ProGen3's attention is pinned to
the flash SDPA backend, which has no float32 kernel, so a ProGen3 decoding arm is
bfloat16 or float16 and cannot meet a float32 sampling contract.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from typing import Any

import torch

from ..core.arms import N_TO_C_MARKER, Arm, arm_spec
from ..generation.unconditional_generation import UnconditionalArm
from ..models.progen3 import (
    PROGEN3_DTYPES, PROGEN3_SOURCE, ProGen3, load_progen3, self_check)

__all__ = [
    "PROGEN3_GENERATION_PROMPT",
    "RENDERING_PROBE",
    "ProGen3DecodingArm",
    "ProGen3DecodingModel",
    "ProGen3DecodingTokenizer",
    "decoding_arm_from_handle",
    "generation_point_for",
    "generation_rendering",
    "progen3_decoding_arm",
    "require_generation_prompt_ids",
    "require_rendering_agreement",
]

#: The prompt string the native generator is driven with, and the one
#: ``EXP-R2-233`` recorded. ``parse_directed_prompt`` reads the leading ``"1"`` as
#: the N-to-C direction and leaves an empty residue prompt, so the conditioning
#: row is ``<bos> 1``. The string form is what the decoder's layout derivation
#: needs; :func:`require_generation_prompt_ids` binds it to the token form.
PROGEN3_GENERATION_PROMPT = N_TO_C_MARKER


def generation_rendering(pg: ProGen3, sequence: str) -> str:
    """ProGen3's own generation character stream for one sequence, from its preparer.

    The CLM token texts of ``prepare_clm``, with the two specials and the trailing
    terminator dropped -- that is, what the official generator conditions on
    followed by what it emits, and nothing else. Derived rather than written down,
    so the decoder's layout cannot drift from the checkpoint's own rendering.
    """

    prepared = pg.preparer.prepare_clm(sequence, False)
    ids = [int(value) for value in prepared["input_ids"].tolist()]
    tokens = [pg.tokenizer.id_to_token(value) for value in ids]
    affixes = (tokens[0], tokens[1], tokens[-2], tokens[-1])
    if affixes != ("<bos>", N_TO_C_MARKER, "2", "<eos>"):
        raise RuntimeError(
            f"ProGen3's CLM rendering opens and closes with {affixes}, not "
            "('<bos>', '1', '2', '<eos>'); the generation layout this module declares "
            "no longer describes the checkpoint's preparer"
        )
    return "".join(tokens[1:-2])


def require_generation_prompt_ids(pg: ProGen3) -> list[int]:
    """The conditioning row the official generator uses, taken through its own chain.

    ``ProGen3Generator.generate`` reads the prompt in two steps, and both are
    load-bearing: ``parse_directed_prompt`` consumes the leading ``"1"`` as the
    *direction* and hands the preparer the residue prompt alone, and only then does
    ``get_generation_kwargs`` render ``<bos> 1 ... 2 <eos>`` and keep the prefix.
    Handing ``"1"`` straight to the preparer instead renders ``<bos> 1 1 2 <eos>``
    and conditions generation on ``[bos, 1, 1]`` -- a doubled direction marker,
    three tokens where the native interface has two. That is the shape of mistake
    this function exists to make impossible, so the chain is reproduced here rather
    than short-circuited, and a prompt that is not two tokens of ``<bos>`` plus the
    N-to-C marker is refused.
    """

    if str(PROGEN3_SOURCE) not in sys.path:
        sys.path.insert(0, str(PROGEN3_SOURCE))
    from progen3.generator import parse_directed_prompt  # noqa: PLC0415

    residues, direction = parse_directed_prompt(PROGEN3_GENERATION_PROMPT)
    if residues or direction != "fwd":
        raise RuntimeError(
            f"the prompt {PROGEN3_GENERATION_PROMPT!r} parses to residues {residues!r} in "
            f"direction {direction!r}; this programme samples the empty N-to-C prompt only"
        )
    kwargs = pg.preparer.get_generation_kwargs(residues, direction == "rev")
    ids = [int(value) for value in kwargs["input_ids"][0].tolist()]
    expected = [int(pg.config.bos_token_id), pg.tokenizer.token_to_id(N_TO_C_MARKER)]
    if ids != expected:
        raise RuntimeError(
            f"ProGen3's native generation conditions on {ids}, not the two-token "
            f"{expected} this module's column arithmetic is derived from"
        )
    positions = [int(value) for value in kwargs["position_ids"][0].tolist()]
    sequences = {int(value) for value in kwargs["sequence_ids"][0].tolist()}
    if positions != list(range(len(ids))):
        raise RuntimeError(
            f"ProGen3's generation prompt carries position ids {positions}, not a "
            "contiguous run from zero; the decoder's position arithmetic would shift"
        )
    if sequences != {0}:
        raise RuntimeError(
            f"ProGen3's generation prompt carries sequence ids {sorted(sequences)}, "
            "not the native independent-sequence zero"
        )
    return ids


class ProGen3DecodingTokenizer:
    """The Hugging Face tokenizer surface the guided decoder calls, over ProGen3's.

    Exactly the four calls the decoder makes, and no more: a wider shim would
    invite a caller to treat a ``tokenizers.Tokenizer`` as a
    ``PreTrainedTokenizer`` everywhere. ``add_special_tokens`` prepends the opening
    special the batch preparer places, because ProGen3's post-processor does not;
    :func:`progen3_decoding_arm` checks the result against the released generator's
    own conditioning row.
    """

    def __init__(self, tokenizer: Any, *, bos_token_id: int):
        self._tokenizer = tokenizer
        self.bos_token_id = int(bos_token_id)

    def get_vocab(self) -> dict[str, int]:
        return self._tokenizer.get_vocab()

    def batch_decode(
        self,
        sequences: Any,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> list[str]:
        rows = [[int(value) for value in row] for row in sequences]
        return self._tokenizer.decode_batch(rows, skip_special_tokens=skip_special_tokens)

    def decode(
        self,
        ids: Any,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        return self._tokenizer.decode(
            [int(value) for value in ids], skip_special_tokens=skip_special_tokens
        )

    def __call__(
        self, text: str, add_special_tokens: bool = True, return_tensors: Any = None
    ) -> dict[str, list[int]]:
        if return_tensors is not None:
            raise ValueError(
                "the ProGen3 decoding tokenizer returns plain id lists; "
                f"return_tensors={return_tensors!r} is not supported"
            )
        ids = [int(value) for value in self._tokenizer.encode(text, add_special_tokens=False).ids]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids}


class ProGen3DecodingModel:
    """``model(input_ids=..., past_key_values=..., use_cache=...)`` on ProGen3.

    Supplies the two arguments ProGen3's forward requires and the decoder does not
    pass: contiguous ``position_ids`` continued from the cache's own length, and
    the native independent-sequence ``sequence_ids`` of zero for every token at
    every batch position. Both come from
    :class:`~progen3.batch_preparer.ProGen3BatchPreparer`'s CLM convention, which
    :func:`require_generation_prompt_ids` checks at construction.

    **One row at a time, and a wider batch is refused.** ProGen3's expert mixture
    reduces over the whole flattened batch, so a row's logits move with the
    batch's *extent* -- not with its content: four copies of one row disagree with
    that row alone by the same amount as four different rows do.
    ``SINGLETON_ONLY_PACKINGS`` in
    ``scripts/capability/stages/context_homologue.py`` already records the
    magnitude for scoring, 0.0246 (3B, batch two) to 0.0280 (112M, batch eight)
    nats per token; measured on the 3B here it is up to 0.036 nats on the
    twenty-residue log-probabilities between batch one and batch four, and that is
    enough to move a draw. Sampling the identical cell at batch four rather than
    batch one changed **18 of 48** sequences (4/16 at L=37, 6/16 at L=52, 8/16 at
    L=72), diverging from step 7 onwards.

    :mod:`src.capability.decoding.guided_decoding` declares that a sequence is a
    function of its seed and the step distributions alone and does not depend on
    which batch it was drawn in. ProGen3 cannot honour that at any batch extent
    above one, so a wider batch is refused here rather than recorded as a cell it
    is not: the alternative is a pool whose sequences are reproducible only
    together with the batch size that happened to be scheduled.
    """

    def __init__(self, pg: ProGen3):
        self._pg = pg
        self.config = pg.config

    @property
    def device(self) -> torch.device:
        return self._pg.device

    def __call__(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        use_cache: bool = True,
    ) -> Any:
        if input_ids.dim() != 2:
            raise ValueError(f"expected (batch, time) input ids, got shape {tuple(input_ids.shape)}")
        rows, width = input_ids.shape
        if rows != 1:
            raise ValueError(
                f"ProGen3 refuses a {rows}-row forward: its expert mixture reduces over "
                "the whole flattened batch, so each row's logits move with the batch's "
                "extent. Sampling one cell at batch four rather than batch one changed 18 "
                "of 48 sequences from the same seeds, so a batched pool is not the cell it "
                "is labelled as. Sample this arm with batch_size=1"
            )
        offset = 0 if past_key_values is None else int(past_key_values.get_seq_length())
        device = input_ids.device
        position_ids = (
            torch.arange(offset, offset + width, device=device).unsqueeze(0).expand(rows, width)
        )
        sequence_ids = torch.zeros((rows, width), dtype=torch.long, device=device)
        return self._pg.model(
            input_ids=input_ids,
            position_ids=position_ids,
            sequence_ids=sequence_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )


def generation_point_for(name: str) -> UnconditionalArm:
    """ProGen3's own generation point, declared here rather than drawn from ADMITTED.

    ``unconditional_generation.ADMITTED`` is the generation point of the
    *unlabelled-generation* campaign. It refuses ``progen3-3b`` because that
    campaign is complete -- a statement about EXP-R2-233, not about whether any
    later experiment may sample the checkpoint -- and it admits ``progen3-112m``
    through the released generator rather than through a forward loop. Either way
    it is not this experiment's declaration, so the adapter carries its own,
    spelled in the same type so a generation point has one shape.
    """

    return UnconditionalArm(
        name=name,
        prompt=PROGEN3_GENERATION_PROMPT,
        close_token="<eos>",
        batch_size=1,
        add_special_tokens=True,
        official_progen3=True,
        dtype="bfloat16",
    )


@dataclass
class ProGen3DecodingArm(Arm):
    """A ProGen3 arm the guided decoder can drive, with its own generation point."""

    generation_point: UnconditionalArm | None = None


def decoding_arm_from_handle(
    name: str, pg: ProGen3, *, dtype: str, self_check_record: Any = None
) -> ProGen3DecodingArm:
    """Wrap a loaded handle as a decoding arm, binding the layout to the preparer.

    Separated from :func:`progen3_decoding_arm` so that every check the adapter
    makes -- the prompt against the released generator's conditioning row, and the
    spec view against the checkpoint's own rendering -- is exercised against the
    real tokenizer and the real preparer without a GPU or six gigabytes of
    weights. Both are properties of the tokenizer and the preparer, which ship
    with the third-party package rather than with the checkpoint.
    """

    spec = arm_spec(name)
    if spec.architecture != "progen3":
        raise ValueError(f"{name}: a ProGen3 decoding arm is for architecture progen3")
    tokenizer = ProGen3DecodingTokenizer(pg.tokenizer, bos_token_id=int(pg.config.bos_token_id))
    prompt_ids = require_generation_prompt_ids(pg)
    tokenised = tokenizer(PROGEN3_GENERATION_PROMPT, add_special_tokens=True)["input_ids"]
    if tokenised != prompt_ids:
        raise RuntimeError(
            f"{name}: the decoding tokenizer renders the generation prompt as "
            f"{tokenised}, while the released generator conditions on {prompt_ids}; "
            "the first sampled step would not be the first residue"
        )
    arm = ProGen3DecodingArm(
        spec=replace(spec, input_format="n_to_c_control"),
        model=ProGen3DecodingModel(pg),
        tokenizer=tokenizer,
        device=str(pg.device),
        dtype=dtype,
        generation_point=generation_point_for(name),
        serving_provenance={
            "progen3": pg,
            "self_check": self_check_record,
            "generation_prompt_ids": prompt_ids,
            "sampling_directions": ("n_to_c",),
            "input_format_view": (
                "n_to_c_control, checked against the checkpoint's own preparer; the "
                "declared ArmSpec.input_format 'raw' describes neither ProGen3's "
                "scoring nor its generation rendering"
            ),
            "singleton_only": (
                "ProGen3's expert mixture reduces over the whole flattened batch, so a "
                "cell sampled at batch size > 1 is a different draw from the same cell "
                "sampled singly"
            ),
        },
    )
    require_rendering_agreement(arm, pg)
    return arm


#: The probe the rendering check is taken on: nine distinct canonical residues, so
#: a rendering that dropped, duplicated or reordered one is visible.
RENDERING_PROBE = "ACDEFGHIK"


def require_rendering_agreement(arm: ProGen3DecodingArm, pg: ProGen3) -> str:
    """Bind the rendering the decoder derives its layout from to ProGen3's own.

    The decoder reads :meth:`~src.capability.core.arms.Cohort.input_strings`
    through ``render_records``, which the adapter steers to ``n_to_c_control`` --
    ProGen2's declaration, which happens to spell the same ``"1"`` marker. That
    coincidence is what the layout rests on, so it is checked against
    :func:`generation_rendering` rather than assumed. Returns the agreed rendering,
    so a caller that records provenance records the string that was checked.
    """

    from ..context.context_homologue import render_records  # noqa: PLC0415

    derived = render_records(arm, [RENDERING_PROBE], modality="protein")[0]
    native = generation_rendering(pg, RENDERING_PROBE)
    if derived != native:
        raise RuntimeError(
            f"{arm.name}: the rendering the decoder derives its layout from is "
            f"{derived!r}, while ProGen3's own batch preparer renders {native!r}; the "
            "layout would not be this checkpoint's and the Potts columns would shift"
        )
    if not native.startswith(arm.generation_point.prompt):
        raise RuntimeError(
            f"{arm.name}: the native rendering {native!r} does not open with the "
            f"declared generation prompt {arm.generation_point.prompt!r}"
        )
    return native


def progen3_decoding_arm(
    name: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    check: bool = True,
) -> ProGen3DecodingArm:
    """Load one ProGen3 rung as a guided-decoding arm.

    Not :func:`src.capability.core.arms.load_arm_spec`: that path is AutoModel and
    would sample random experts. The loader's numerical self-check runs before the
    arm is returned, so a checkpoint whose experts did not load cannot be sampled.
    """

    spec = arm_spec(name)
    if spec.architecture != "progen3":
        raise ValueError(f"{name}: progen3_decoding_arm is for architecture progen3")
    torch_dtype = PROGEN3_DTYPES.get(dtype)
    if torch_dtype is None:
        raise ValueError(
            f"ProGen3 has no kernel for {dtype!r}: its attention is pinned to the "
            f"flash SDPA backend. Supported: {sorted(PROGEN3_DTYPES)}"
        )
    pg = load_progen3(spec.path, device=device, dtype=torch_dtype)
    return decoding_arm_from_handle(
        name, pg, dtype=dtype, self_check_record=self_check(pg) if check else None
    )
