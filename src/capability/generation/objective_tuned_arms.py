"""Objective-tuned checkpoints and their matched bases, read beside the panel.

The question these arms exist to answer is declared in ``docs/objective-alignment.md``
and amended into the programme in ``docs/logbook/PROJECT_LOG_2026-11-09``: when a
protein generative model **is** explicitly optimised for an objective, does it
acquire a design-relevant distributional bias, or does it concentrate its
statistical memorisation on a narrower slice of the evolutionary record? Every
checkpoint here is a public release that somebody else tuned, paired with the
released base sibling it was tuned from, so the effect of the tuning is a paired
contrast and the base arm is the control for its own sibling.

**None of these arms is a panel arm.** :data:`ARMS` is separate from
:data:`src.capability.core.arms.PANEL`, from its staged doors, and from
``src.capability.interactions.pairwise_epistasis.ROSTER``; nothing in this file
touches any of them, so the admitted panel stays at 33 arms and every panel
count, display category, release family and verdict is unchanged by anything
measured here. An objective-tuned row is read beside the panel, never inside it.
That is the precedent ``scripts/capability/interactions/exploratory_denoising_arms.py``
set for its ``EXPLORATORY_ROSTER`` and the one the ProGen3 amendment set for
:data:`src.capability.core.arms.PROGEN3_ARMS`.

Why the declarations sit here rather than in ``src.capability.core.arms``
------------------------------------------------------------------------
Both staged doors in that module carry obligations these arms cannot honestly
meet. ``STAGED_ARMS`` requires every member to declare a
``scoring_target_alphabet_size`` *as an admission to budget scoring*, and its
keys must be mirrored by ``scripts.capability.panel_contract.STAGED_BUT_NOT_ADMITTED``
with a reason per checkpoint. These arms enter one lane -- free-length
generation -- and no scoring lane at all, which is what ``capabilities``
records below as an empty set. Declaring them there would buy a scoring
admission nobody measured. They are declared instead where the experiment that
reads them lives, and :func:`spec_for` is the one resolver that sees both sides.

The pairing is the measurement
------------------------------
Downstream analysis is paired, so :data:`PAIRS` is the single authoritative
statement of which tuned arm has which base, what its tuning optimised, which
generation lane honours its native interface, and whether its weights are
staged. A call site that needs any of those four facts resolves them here and
does not restate them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from ..core.amino_acids import AA20
from ..core.arms import (
    INPUT_FORMAT_BOS_DIRECTION_SEQ,
    MODEL_ROOT,
    PRETRAINING_UNDECLARED,
    ArmSpec,
    arm_spec,
    config_shape,
    rendering_marker_ids,
)

#: The run identity every artefact of this experiment carries. It is the date of
#: the amendment that reopened the fine-tuned-checkpoint half of R6, not a new
#: E-number: ``docs/objective-alignment.md`` is the registration.
CAMPAIGN = "objective_tuned_20261109"

#: The renderings this experiment's arms declare, spelled as the panel spells
#: them so that the two cannot drift.
INPUT_FORMAT_N_TO_C_CONTROL = "n_to_c_control"
INPUT_FORMAT_EC_CONDITIONED = "ec_conditioned"

# ------------------------------------------------------------------ the arms
#
# Depth and width below are read off each checkpoint's own ``config.json`` and are
# verified against the live config by ``load_arm_spec`` on every load, so a
# declaration that drifts from the weights refuses the run instead of producing a
# number. ``evaluation_cohort_source`` is the protein default these arms inherit
# from their panel-declared siblings and is inert: ``capabilities`` is empty, so
# no scoring lane will serve them and no cohort is ever drawn for them.
# ``scoring_target_alphabet_size`` is the measured width of each head -- 31 for
# the ProtGPT3 MoE, 32 for the ProGen2 lineage -- recorded because
# ``progen2-small-mix7`` declares no ``config.vocab_size`` at all (only
# ``vocab_size_emb: 37`` and ``vocab_size_lm_head: 32``), and a stage that read
# the config key directly would raise at an arbitrary depth of a run.


def _protgpt3_spec(name: str, directory: str, n_layer: int, d_model: int) -> ArmSpec:
    """One ProtGPT3 rung: a Mixtral-style protein decoder MoE, per-residue tokenizer.

    The whole family -- base and DPO sibling, 112M to 10B -- ships one tokenizer
    file; it is byte-identical across the five staged directories and
    :func:`require_matched_pair` proves that per pair rather than assuming it.
    The rendering is :data:`INPUT_FORMAT_BOS_DIRECTION_SEQ`, which
    ``src.capability.core.arms`` already declares and priced for
    ``protgpt3-1.3b``.
    """

    return ArmSpec(
        name=name,
        path=MODEL_ROOT / directory,
        path_variable="MODEL_ROOT",
        modality="protein",
        n_layer=n_layer,
        d_model=d_model,
        tokenisation="residue",
        input_format=INPUT_FORMAT_BOS_DIRECTION_SEQ,
        evaluation_cohort_source="swissprot",
        architecture="mixtral",
        pretraining_corpus=PRETRAINING_UNDECLARED,
        capabilities=frozenset(),
        scoring_target_alphabet_size=31,
    )


def _progen2_community_spec(name: str, directory: str) -> ArmSpec:
    """One rung of the ProGen2-finetuning author's own ProGen2-small lineage.

    Both carry their own ``configuration_progen.py``/``modeling_progen.py`` through
    ``auto_map``, so the architecture is remote code and the matched base has to be
    the same author's mirror rather than an officially-sourced conversion. The two
    modeling files are **not** identical: the base mirror's ``ProGenForCausalLM``
    inherits ``GenerationMixin`` and the fine-tune's does not, which Transformers
    4.50+ makes the difference between a model that has ``generate`` and one that
    does not. ``conditioned_generation.ensure_generate`` closes that gap for the
    fine-tune, and the difference is recorded here because it is a property of the
    pair rather than of the campaign.
    """

    return ArmSpec(
        name=name,
        path=MODEL_ROOT / directory,
        path_variable="MODEL_ROOT",
        modality="protein",
        n_layer=12,
        d_model=1024,
        tokenisation="residue",
        input_format=INPUT_FORMAT_N_TO_C_CONTROL,
        evaluation_cohort_source="swissprot",
        architecture="progen",
        pretraining_corpus="uniref90_bfd30",
        capabilities=frozenset(),
        scoring_target_alphabet_size=32,
    )


ARMS: dict[str, ArmSpec] = {
    "protgpt3-112m": _protgpt3_spec("protgpt3-112m", "ProtGPT3-112M", 8, 512),
    "protgpt3-112m-dpo": _protgpt3_spec("protgpt3-112m-dpo", "ProtGPT3-112M-dpo", 8, 512),
    # The base 1.3B rung is already declared, as
    # ``src.capability.core.arms.STAGED_ARMS["protgpt3-1.3b"]``; only its DPO
    # sibling is new, and :func:`spec_for` resolves the two through one door.
    "protgpt3-1.3b-dpo": _protgpt3_spec("protgpt3-1.3b-dpo", "ProtGPT3-1.3B-dpo", 17, 1024),
    "protgpt3-10b": _protgpt3_spec("protgpt3-10b", "ProtGPT3-10B", 32, 2048),
    # This rung's weights were still arriving while the declarations were written,
    # which is why staging is a filesystem question answered at call time by
    # :func:`weights_complete` rather than a string frozen here. The shape is read
    # off its own config, like every other rung.
    "protgpt3-10b-dpo": _protgpt3_spec("protgpt3-10b-dpo", "ProtGPT3-10B-dpo", 32, 2048),
    "progen2-small-hugohrban": _progen2_community_spec(
        "progen2-small-hugohrban", "progen2-small-hugohrban"
    ),
    "progen2-small-mix7": _progen2_community_spec(
        "progen2-small-mix7", "progen2-small-mix7"
    ),
}

#: The arms the free-length generation lane must learn about: everything declared
#: here, because an arm's prompt contract is a declaration rather than a run and a
#: checkpoint that finishes downloading needs no further edit. Arms already
#: declared in that lane -- ``protgpt3-1.3b``, ``protgpt2`` -- are deliberately
#: absent, so the lane keeps one generation point per checkpoint.
OFF_PANEL_GENERATION_ARMS: tuple[str, ...] = tuple(ARMS)

# -------------------------------------------------------------- the objectives

OBJECTIVE_DPO_PLDDT_COMPLEXITY = (
    "DPO over the model's own 80,000 samples, positives being those an ESMFold "
    "proxy scored above 0.7 pLDDT with fewer than 25% low-complexity residues; "
    "no experimental feedback and no external environment"
)
OBJECTIVE_EC_ENZYME = (
    "fine-tuned on EC-labelled enzyme sequences and conditioned on the EC class, "
    "so the objective is a function label rather than a structural proxy"
)
OBJECTIVE_PFAM_SEVEN_FAMILIES = (
    "fine-tuned on seven Pfam families -- PF00002, PF00042, PF00125, PF00127, "
    "PF00257, PF00262, PF03668 -- which is an explicit domain narrowing and is "
    "here as the positive control for narrowing rather than as a design claim"
)

# ------------------------------------------------------------------ the lanes

#: Free-length unlabelled generation: ``src.capability.generation.unconditional_generation``
#: at its frozen operating point. Both sides of a pair in this lane accept the
#: same prompt, which is what makes the contrast paired.
LANE_FREE_LENGTH = "free_length_unconditional"

#: Native class-conditioned generation: ``src.capability.generation.conditioned_generation``
#: (EXP-R2-227), whose design already carries the within-arm mismatched-label
#: negative that makes a conditioned rate readable. A pair lands here when the
#: tuned arm's native interface is a class request its base cannot receive.
LANE_EC_CONDITIONED = "ec_conditioned"

LANES: frozenset[str] = frozenset({LANE_FREE_LENGTH, LANE_EC_CONDITIONED})


@dataclass(frozen=True)
class ObjectiveTunedPair:
    """One tuned checkpoint, the base it was tuned from, and what was optimised.

    ``lane`` is a declaration about the arms, not a preference about the campaign:
    it names the generation door whose prompt contract both sides honour. A pair
    cannot be moved to the other lane without changing one side's interface, and
    changing an interface is what puts a checkpoint out of distribution.

    Whether the pair's weights are *present* is deliberately not a field here.
    One of these pairs was still downloading while the declarations were written,
    so a frozen status string would have been stale within the hour;
    :func:`staged` reads the filesystem at call time instead and
    :func:`staging_report` writes the answer into the campaign record.

    ``secondary`` marks a pair whose checkpoints carry thin training
    documentation. It is reported, never gated on, and never merged into the
    primary family.
    """

    base: str
    tuned: str
    objective: str
    lane: str
    note: str
    secondary: bool = False

    @property
    def arms(self) -> tuple[str, str]:
        return (self.base, self.tuned)


PAIRS: tuple[ObjectiveTunedPair, ...] = (
    ObjectiveTunedPair(
        base="protgpt3-112m",
        tuned="protgpt3-112m-dpo",
        objective=OBJECTIVE_DPO_PLDDT_COMPLEXITY,
        lane=LANE_FREE_LENGTH,
        note=(
            "the smallest ProtGPT3 rung, 8 blocks of width 512. Their model "
            "configs differ only by max_position_embeddings (1025 against 16384) "
            "and router_aux_loss_coef, neither of which a two-token prompt and "
            "400 new tokens can reach, and their generation configs only by the "
            "base's scalar against the sibling's single-element eos_token_id, "
            "which Transformers treats alike"
        ),
    ),
    ObjectiveTunedPair(
        base="protgpt3-1.3b",
        tuned="protgpt3-1.3b-dpo",
        objective=OBJECTIVE_DPO_PLDDT_COMPLEXITY,
        lane=LANE_FREE_LENGTH,
        note=(
            "the centre of the experiment: its base is already a staged arm with a "
            "priced native rendering, so the pair inherits a rendering this "
            "repository measured rather than one it adopted here"
        ),
    ),
    ObjectiveTunedPair(
        base="protgpt3-10b",
        tuned="protgpt3-10b-dpo",
        objective=OBJECTIVE_DPO_PLDDT_COMPLEXITY,
        lane=LANE_FREE_LENGTH,
        note=(
            "the third alignment rung, and the one whose staging lagged the "
            "declarations: on 2026-11-09 the base's five shards completed while the "
            "sibling's had not begun. A schedule reads :func:`staged` rather than "
            "this note, and a campaign that runs without this pair loses the scale "
            "axis of the alignment contrast and nothing else -- the 112M and 1.3B "
            "pairs carry the primary comparison on their own"
        ),
    ),
    ObjectiveTunedPair(
        base="protgpt2",
        tuned="zymctrl",
        objective=OBJECTIVE_EC_ENZYME,
        lane=LANE_EC_CONDITIONED,
        note=(
            "both sides are panel arms and neither is declared here. The lane is "
            "not free-length because ZymCTRL's native input is an EC class tag and "
            "ProtGPT2 has no channel to receive one: an unconditioned ZymCTRL "
            "prompt would be a '<start>' that never began a training record, and a "
            "conditioned ProtGPT2 prompt does not exist. EXP-R2-227 already "
            "generated this pair through its own door, conditioned arm against "
            "unconditioned floor, with the within-arm mismatched-label negative "
            "that prices the tag; re-running it free-length would duplicate that "
            "design with one side out of distribution"
        ),
    ),
    ObjectiveTunedPair(
        base="progen2-small-hugohrban",
        tuned="progen2-small-mix7",
        objective=OBJECTIVE_PFAM_SEVEN_FAMILIES,
        lane=LANE_FREE_LENGTH,
        note=(
            "the narrowing control, at secondary status because both checkpoints "
            "are community releases with thin training documentation. The pair is "
            "prompted with the shared n-to-c direction marker, which is the only "
            "prompt both sides accept: the fine-tune's native conditional form "
            "prefixes a '<|pfXXXXX|>' family token that exists in its tokenizer "
            "and not in its base's, so a family-conditioned cell would have no "
            "control. Those seven tokens are embedding-only -- the head stays 32 "
            "wide -- so they cannot be emitted and an unconditional pool is not "
            "quietly conditioned. One further asymmetry is declared rather than "
            "corrected: the fine-tune's tokenizer file carries a fixed-1024 "
            "padding and truncation configuration its base's does not, which the "
            "Transformers fast wrapper neutralises on load, and which this lane "
            "never reaches because a cell encodes one prompt and repeats it"
        ),
        secondary=True,
    ),
)


# ------------------------------------------------------------------- resolvers


def spec_for(name: str) -> ArmSpec:
    """The declaration for an objective-tuned arm or any already-declared arm.

    One resolver, so that a paired contrast reads both sides' rendering, shape and
    path off the same field whichever door declared them. It does **not** widen
    :func:`src.capability.core.arms.load_arm`, which stays panel-only, and it does
    not make an arm declared here reachable by name from any scoring stage: that is
    what the empty ``capabilities`` on every :data:`ARMS` member refuses.
    """

    if name in ARMS:
        return ARMS[name]
    return arm_spec(name)


def pair_for(arm: str) -> ObjectiveTunedPair:
    """The pair one arm belongs to, by either name."""

    for pair in PAIRS:
        if arm in pair.arms:
            return pair
    raise KeyError(
        f"{arm!r} is not an objective-tuned arm; the declared pairs are "
        f"{[pair.arms for pair in PAIRS]}"
    )


def base_of(tuned: str) -> str:
    """The matched base of one tuned arm, refusing a base name."""

    pair = pair_for(tuned)
    if pair.tuned != tuned:
        raise KeyError(f"{tuned!r} is the base of {pair.tuned!r}, not a tuned arm")
    return pair.base


def pairs_in_lane(lane: str) -> tuple[ObjectiveTunedPair, ...]:
    """Every declared pair of one lane, staged or not."""

    if lane not in LANES:
        raise KeyError(f"unknown generation lane {lane!r}; declared: {sorted(LANES)}")
    return tuple(pair for pair in PAIRS if pair.lane == lane)


#: Weight-file extensions a checkpoint directory may hold. ProtGPT2 and ZymCTRL
#: ship ``pytorch_model.bin``; everything newer here ships safetensors, sharded
#: above a few gigabytes.
_WEIGHT_SUFFIXES = (".safetensors", ".bin")


def weights_complete(name: str) -> bool:
    """Whether this arm's weights are wholly present, shards included.

    A sharded checkpoint mid-download has weight files and is not loadable, so the
    presence of *a* file is not the question. When an index is present every shard
    it names must be present; otherwise a single-file weight must be.
    """

    path = spec_for(name).path
    if not path.is_dir():
        return False
    index = path / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return all((path / shard).is_file() for shard in set(weight_map.values()))
    return any(
        item.is_file()
        for suffix in _WEIGHT_SUFFIXES
        for item in path.glob(f"*{suffix}")
    )


def staged(pair: ObjectiveTunedPair) -> bool:
    """Whether both sides of a pair are loadable on this host, read now."""

    return all(weights_complete(name) for name in pair.arms)


def staged_pairs(lane: str) -> tuple[ObjectiveTunedPair, ...]:
    """The pairs of one lane a schedule may reach, by the live staging check."""

    return tuple(pair for pair in pairs_in_lane(lane) if staged(pair))


def scheduled_free_length_arms() -> tuple[str, ...]:
    """Every arm a free-length schedule may name, base before tuned, in pair order."""

    out: list[str] = []
    for pair in staged_pairs(LANE_FREE_LENGTH):
        out.extend(pair.arms)
    return tuple(out)


# ---------------------------------------------------------- interface contract


def _shim(name: str, spec: ArmSpec, tokenizer: Any) -> Any:
    """The three attributes :func:`rendering_marker_ids` reads, without a load.

    The ids a rendering prefixes are declared once, in ``src.capability.core.arms``,
    against a loaded :class:`~src.capability.core.arms.Arm`. A tokenizer-only check
    must resolve *that* declaration rather than a second copy of it, so it passes
    the declaration what it reads and nothing else.
    """

    return SimpleNamespace(name=name, spec=spec, tokenizer=tokenizer)


def prompt_contract(arm: str, tokenizer: Any, prompt: str) -> dict[str, Any]:
    """Verify that ``prompt`` encodes to exactly the marker ids its rendering declares.

    This is the check the ProtGPT3 arms need and the one a plausible-looking
    tokenizer keyword silently breaks. Their published usage constructs the
    tokenizer with ``add_bos_token=True`` and prompts with the bare direction
    token ``"1"``; the free-length lane spells ``"<|bos|>1"`` and leaves the
    published ``add_bos_token: false`` default in place. Both encode to
    ``(bos_id, direction_id)`` and are the same contract. Combining them does
    not: ``add_bos_token=True`` over ``"<|bos|>1"`` encodes ``(bos, bos,
    direction)``, a prompt the checkpoint was never trained behind, and dropping
    the BOS from both encodes ``(direction,)``, the rendering this repository
    already priced at 3.0495 nats/residue against 1.7957. Neither failure is
    visible in a decoded string, so the id sequence is what is checked.

    Returns the verified facts for the interface record.
    """

    spec = spec_for(arm)
    expected = rendering_marker_ids(_shim(arm, spec, tokenizer))
    encoded = tuple(int(value) for value in tokenizer(prompt)["input_ids"])
    if encoded != expected:
        raise ValueError(
            f"{arm}: the prompt {prompt!r} encodes to {encoded}, but its declared "
            f"{spec.input_format} rendering prefixes {expected}. A prompt that is "
            "not its rendering puts the checkpoint out of distribution; it is "
            "refused rather than sampled"
        )
    return {
        "arm": arm,
        "input_format": spec.input_format,
        "prompt": prompt,
        "prompt_ids": list(encoded),
        "declared_marker_ids": list(expected),
    }


def residue_alphabet_contract(arm: str, tokenizer: Any) -> dict[str, Any]:
    """Verify that every canonical residue is one id that decodes back to itself.

    A per-residue tokenizer is the premise of this lane: the extractor reads a
    decoded string character by character, so a residue that encodes to two pieces
    or decodes to something other than itself would silently shorten every
    generated sequence instead of failing.
    """

    ids: dict[str, int] = {}
    for residue in AA20:
        encoded = [
            int(value)
            for value in tokenizer(residue, add_special_tokens=False)["input_ids"]
        ]
        if len(encoded) != 1:
            raise ValueError(
                f"{arm}: residue {residue!r} encodes to {encoded}, not one id; this "
                "lane's extractor reads decoded characters and cannot carry a "
                "multi-piece residue"
            )
        decoded = tokenizer.decode(encoded, skip_special_tokens=False).strip()
        if decoded != residue:
            raise ValueError(
                f"{arm}: id {encoded[0]} round-trips to {decoded!r} rather than "
                f"{residue!r}"
            )
        ids[residue] = encoded[0]
    return {"arm": arm, "residue_ids": ids, "alphabet": "".join(AA20)}


def require_residue_sequences(arm: str, sequences: Iterable[str]) -> dict[str, Any]:
    """Refuse a sample whose continuations are not residue strings.

    Empty and budget-censored continuations stay in the denominator -- that is the
    lane's own requirement and this check does not weaken it -- but a sample in
    which *nothing* extracted is an interface failure rather than a measurement,
    and a non-residue character in an extracted string is a broken extractor.
    """

    alphabet = set(AA20)
    retained = list(sequences)
    if not retained:
        raise ValueError(f"{arm}: interface verification needs at least one continuation")
    for sequence in retained:
        stray = sorted(set(sequence) - alphabet)
        if stray:
            raise ValueError(
                f"{arm}: extracted sequence carries non-residue characters {stray}"
            )
    non_empty = [sequence for sequence in retained if sequence]
    if not non_empty:
        raise ValueError(
            f"{arm}: every one of {len(retained)} continuations extracted to an "
            "empty residue run; the prompt contract holds but the checkpoint "
            "generated nothing this lane can read"
        )
    return {
        "arm": arm,
        "attempts": len(retained),
        "non_empty": len(non_empty),
        "residues_min": min(len(sequence) for sequence in non_empty),
        "residues_max": max(len(sequence) for sequence in non_empty),
    }


def declared_dtype(config: Any) -> str | None:
    """The precision a loaded config declares, as a bare torch dtype name.

    Both spellings are read because Transformers renamed the field: 4.x writes
    ``torch_dtype`` and 5.x writes ``dtype``. ``None`` means the config declares
    no precision at all, which is not the same as declaring float32.

    This is what the *config* says, which on one checkpoint here is not what its
    tensors are: ``ProtGPT3-10B-dpo``'s ``config.json`` carries both
    ``dtype: bfloat16`` and a stale ``torch_dtype: float32``, Transformers 4.57
    resolves the pair to bfloat16, and the tensors on disk are float32 -- twice
    the base's size for the same 32x2048 architecture. :func:`stored_dtype` is
    therefore the authoritative answer and this one is kept beside it, because a
    config that disagrees with its own weights is worth recording rather than
    silently preferring one of the two.
    """

    for field in ("torch_dtype", "dtype"):
        value = getattr(config, field, None)
        if value is not None:
            return str(value).removeprefix("torch.")
    return None


#: safetensors spells dtypes in its own header vocabulary.
_SAFETENSORS_DTYPES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}


def _safetensors_header(path: Path) -> dict[str, Any]:
    """One safetensors file's header, read without mapping its tensors."""

    with path.open("rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        return json.loads(handle.read(length).decode("utf-8"))


def stored_dtype(path: Path) -> str | None:
    """The floating precision a checkpoint's tensors are actually serialised in.

    Read from the safetensors headers rather than from the config, because the
    config can be wrong about its own weights and the loaded config object hides
    which of two spellings won. Every shard's header is read; they are a few
    kilobytes each and a checkpoint whose shards disagree must not be reported as
    though it had one precision. Returns ``None`` for a checkpoint this cannot
    read -- a ``.bin`` pickle, or a directory with no weights yet -- rather than
    guessing, and a comma-joined list when the tensors genuinely are mixed.
    """

    path = Path(path)
    index = path / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted({path / shard for shard in weight_map.values()})
    else:
        shards = sorted(path.glob("*.safetensors"))
    found: set[str] = set()
    for shard in shards:
        if not shard.is_file():
            return None
        for name, entry in _safetensors_header(shard).items():
            if name == "__metadata__":
                continue
            spelled = _SAFETENSORS_DTYPES.get(str(entry.get("dtype")))
            if spelled is not None:
                found.add(spelled)
    if not found:
        return None
    return ",".join(sorted(found))


def require_matched_pair(
    pair: ObjectiveTunedPair, loaded: Mapping[str, Any]
) -> dict[str, Any]:
    """Refuse a pair whose two sides are not the same architecture and tokenizer.

    ``loaded`` maps each arm name to an object carrying ``config`` and
    ``tokenizer``. The pair is the experiment's only control, so the two
    properties that make it one are asserted rather than assumed: a differently
    shaped or differently tokenised sibling would turn a tuned-minus-base
    difference into a cross-model comparison, which is the design this experiment
    exists to avoid.

    Tokenizer equality is checked over the **content alphabet**: every token that
    neither side declares special must carry the same id on both, which is what
    makes a residue, a direction marker and an extracted sequence mean the same
    thing in the two pools. Three further facts are checked because generation
    depends on them directly -- the checkpoints' own ``bos``, ``eos`` and ``pad``
    token ids, which are what ``generate`` stops and fills on.

    Special-token id differences are **reported rather than refused**, and the
    ProGen2 pair has one that matters to read correctly. Neither community
    checkpoint ships a tokenizer configuration, so ``AutoTokenizer`` falls back to
    GPT-2's defaults and injects ``<|endoftext|>`` above the declared vocabulary:
    at id 30 on the base and at 37 on the fine-tune, because the fine-tune
    declares seven ``<|pfXXXXX|>`` family tokens first. That shift is an artefact
    of tokenizer construction over a 30-token protein vocabulary, not a property
    of either checkpoint, and it sits above every id the 32-wide head can reach on
    the fine-tune. The seven family tokens themselves are embedding-only -- the
    head stays 32 wide -- so they cannot be emitted; they are returned as
    ``tokens_added_by_tuning`` so that a declared extension and an undetected
    mismatch cannot look alike.

    The **precision** is reported rather than refused, for the same reason.
    ``ProtGPT3-10B`` is serialised bfloat16 and ``ProtGPT3-10B-dpo`` float32 --
    twice the size for the same 32x2048 architecture -- which is how each file was
    written and not a difference in what either model is. The lane loads both
    sides of a pair at one compute dtype: ``load_arm_spec`` is given it explicitly
    and then refuses any parameter that did not arrive in it, so the asymmetry is
    neutralised at load and would otherwise vanish from the record. Letting each
    arm sample at its own stored precision is the thing that must not happen, as a
    float32 tuned arm against a bfloat16 base would read any precision artefact as
    an effect of the preference optimisation. Refusing the pair instead would
    reject two matched checkpoints over how their files were written.

    Both answers are carried, per arm, in ``checkpoint_dtypes``, because they
    disagree on exactly the checkpoint that matters: ``ProtGPT3-10B-dpo``'s
    ``config.json`` holds ``dtype: bfloat16`` beside a stale
    ``torch_dtype: float32``, Transformers resolves that to bfloat16, and its
    tensors are float32 regardless. ``stored`` comes from the caller, which reads
    the weight headers through :func:`stored_dtype`; a caller that supplies none
    leaves it ``None`` rather than letting the config stand in for it.
    """

    missing = [name for name in pair.arms if name not in loaded]
    if missing:
        raise ValueError(f"{pair.arms}: nothing loaded for {missing}")
    base, tuned = (loaded[name] for name in pair.arms)
    base_spec, tuned_spec = (spec_for(name) for name in pair.arms)
    if base_spec.architecture != tuned_spec.architecture:
        raise ValueError(
            f"{pair.arms}: declared architectures {base_spec.architecture!r} and "
            f"{tuned_spec.architecture!r} differ, so the pair is not matched"
        )
    live = [str(getattr(side.config, "model_type", "")) for side in (base, tuned)]
    if live[0] != live[1]:
        raise ValueError(
            f"{pair.arms}: live config model_type {live[0]!r} and {live[1]!r} differ"
        )
    shapes = [config_shape(side.config) for side in (base, tuned)]
    if shapes[0] != shapes[1]:
        raise ValueError(
            f"{pair.arms}: live (depth, width) {shapes[0]} and {shapes[1]} differ"
        )
    boundary = {}
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        values = [getattr(side.config, field, None) for side in (base, tuned)]
        normalised = [
            tuple(value) if isinstance(value, (list, tuple)) else value for value in values
        ]
        if normalised[0] != normalised[1]:
            raise ValueError(
                f"{pair.arms}: the checkpoints declare different {field}, "
                f"{values[0]!r} against {values[1]!r}; generation stops and fills on "
                "these ids, so the pair is not matched"
            )
        boundary[field] = normalised[0]
    base_vocab = base.tokenizer.get_vocab()
    tuned_vocab = tuned.tokenizer.get_vocab()
    special = set(base.tokenizer.all_special_tokens) | set(tuned.tokenizer.all_special_tokens)
    disagreeing = sorted(
        token for token, index in base_vocab.items() if tuned_vocab.get(token) != index
    )
    content = [token for token in disagreeing if token not in special]
    if content:
        raise ValueError(
            f"{pair.arms}: the tuned tokenizer gives different ids to "
            f"{len(content)} content tokens, the first being {content[:5]}; the pair "
            "is not tokenizer-matched"
        )
    dtypes = {
        name: {
            "config": declared_dtype(side.config),
            "stored": getattr(side, "stored_dtype", None),
        }
        for name, side in zip(pair.arms, (base, tuned), strict=True)
    }
    return {
        "base": pair.base,
        "tuned": pair.tuned,
        "architecture": base_spec.architecture,
        "live_model_type": live[0],
        "depth_width": list(shapes[0]),
        "declared_boundary_ids": boundary,
        "checkpoint_dtypes": dtypes,
        "config_dtype_agrees": len({entry["config"] for entry in dtypes.values()}) == 1,
        "stored_dtype_agrees": len({entry["stored"] for entry in dtypes.values()}) == 1,
        "content_alphabet_tokens": len(set(base_vocab) - special),
        "tokens_added_by_tuning": sorted(set(tuned_vocab) - set(base_vocab)),
        "special_token_id_differences": {
            token: [base_vocab.get(token), tuned_vocab.get(token)] for token in disagreeing
        },
    }


# ---------------------------------------------------------- declaration checks


def _check_declarations() -> None:
    """One name means one thing, and every pair resolves both of its sides."""

    for name, spec in ARMS.items():
        if spec.name != name:
            raise AssertionError(f"arm {name!r} declares the name {spec.name!r}")
        if spec.capabilities:
            raise AssertionError(
                f"{name}: an objective-tuned arm declares no measurement capability, "
                f"got {sorted(spec.capabilities)}"
            )
        try:
            arm_spec(name)
        except KeyError:
            continue
        raise AssertionError(
            f"{name} is declared both here and in src.capability.core.arms; one name "
            "must mean one thing"
        )
    seen: dict[str, tuple[str, str]] = {}
    for pair in PAIRS:
        if pair.lane not in LANES:
            raise AssertionError(f"{pair.arms}: unknown lane {pair.lane!r}")
        if pair.base == pair.tuned:
            raise AssertionError(f"{pair.base}: a pair cannot be its own control")
        if not pair.objective or not pair.note:
            raise AssertionError(
                f"{pair.arms}: a pair declares what its tuning optimised and why it "
                "is in the lane it is in"
            )
        for name in pair.arms:
            if name in seen:
                raise AssertionError(
                    f"{name} appears in two pairs, {seen[name]} and {pair.arms}; the "
                    "pairing must be one base per tuned arm"
                )
            seen[name] = pair.arms
            spec_for(name)
    orphans = sorted(set(ARMS) - set(seen))
    if orphans:
        raise AssertionError(
            f"{orphans} are declared as arms but belong to no pair; an arm with no "
            "control cannot enter a paired contrast"
        )


_check_declarations()


def staging_report() -> dict[str, Any]:
    """Which declared arms are loadable on this host, read at call time.

    Existence is deliberately not checked at import: a declaration must not depend
    on a download finishing, and this experiment had a pair whose had not. Every
    campaign record carries this report, so a pair that was skipped is skipped
    with the reason attached rather than silently absent.
    """

    out: dict[str, Any] = {"model_root": str(MODEL_ROOT), "arms": {}, "pairs": {}}
    for name in sorted({name for pair in PAIRS for name in pair.arms} | set(ARMS)):
        path = spec_for(name).path
        out["arms"][name] = {
            "path": str(path),
            "directory_exists": path.is_dir(),
            "weights_complete": weights_complete(name),
            "declared_here": name in ARMS,
        }
    for pair in PAIRS:
        out["pairs"][pair.tuned] = {
            "base": pair.base,
            "lane": pair.lane,
            "objective": pair.objective,
            "secondary": pair.secondary,
            "staged": staged(pair),
            "incomplete_sides": [
                name for name in pair.arms if not out["arms"][name]["weights_complete"]
            ],
        }
    return out
