#!/usr/bin/env python3
"""EXP-R2-225 interface qualification. Not a panel stage and not a capability claim.

The sibling of ``scale_interface_qualification.py``, and a sibling rather than a
parameterisation of it. That stage's constants -- the 74-residue probe and its
SHA-256, the ``"1"``/``"2"`` direction-marker pair, the scoring-target support of
32 and the three frozen live widths -- are ProGen2-lineage facts. None of them is
true of a Qwen2.5 rung or of RITA, and a stage that took them as arguments would
be a stage whose frozen numbers live at its call site, which is the one property
an all-or-stop gate must not have. So this file freezes its own.

**What EXP-R2-225 asks of a checkpoint before it may be scored**, and what is
implemented here, clause by clause:

1. *Strict load.* ``load_arm_spec(..., strict=True)`` refuses any missing,
   unexpected, mismatched or error-reported key, and the counts it verified are
   recorded on the arm. An arm that arrives without them is refused rather than
   recorded as a clean load on no evidence -- the L24 silent-random-head failure
   is what that clause exists for.
2. *Native rendering as trained, scored positions identified, output semantics
   verified.* The probe is rendered through ``Cohort.input_strings``, so the
   rendering is the panel's declaration and not this stage's (Appendix B rule 12).
   Every scored target id is required to lie inside the arm's declared
   scoring-target alphabet, and the live output width is required to equal this
   campaign's frozen declaration for that checkpoint.
3. *A fixed NLL self-check and a negative control that makes NLL substantially
   worse.* The native probe is scored twice and must repeat to
   :data:`REPEAT_MAX_ABS` nats; the control is a frozen **anagram** of the probe
   -- the same symbols in a different order -- and its mean NLL must exceed the
   native mean by more than :data:`SHUFFLE_COST_MIN` nats per scored target.
4. *An unavailable checkpoint is reported unavailable, never substituted.*
   :data:`DECLARED_UNAVAILABLE` names each one with the measured reason, resolved
   before anything is loaded.

**One arm per invocation.** EXP-R2-224's qualification is all-or-stop across
three rungs in one process because that campaign's ladder is one lineage and a
rung that cannot be read invalidates the pair it belongs to. EXP-R2-225 says the
opposite in its own words -- "Failure of any clause stops **that arm**... no
other arm is moved into the vacant slot" -- so the unit here is the arm, and each
one is a separate cell that can fail without touching another's card.

**What is qualified elsewhere, named rather than omitted.** Two of this
campaign's waves do not come through this door and must not be given a second,
weaker one:

* ProGen3 (112M and 3B) qualifies through
  :func:`src.capability.models.progen3.self_check`, which is a stronger check of exactly
  this shape -- a frozen sequence set, a per-checkpoint band measured on that
  checkpoint, and three recorded corruptions including the strict-clean wrong
  expert mapping that no ``load_state_dict`` can see. ``retrieval_bound.py``
  runs it before it scores anything.
* The joint wave (``facebook/galactica-{1.3b,6.7b,30b}``) qualifies through
  ``joint_mode_qualification.py``, which is the stage EXP-R2-225 means by
  "joint checkpoints qualify each mode separately" and which carries the
  directional reversal control the prereg requires of a protein mode. Galactica
  is reached by path and is declared as an arm nowhere, so it cannot reach
  :func:`load_arm_spec` and this stage's route does not exist for it.

A PASS says the frozen loader, the output interface and a negative control are
usable at the precision it was run at. It is not panel admission, not protein or
text capability, not a scale transition, not mechanism evidence, and not
biological knowledge.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.arms import (  # noqa: E402
    PANEL,
    REPO,
    STAGED_SECOND_STAGE_ARMS,
    Arm,
    Cohort,
    arm_spec,
    load_arm_spec,
    output_logit_width,
    rendering_marker_ids,
    require_scoring_target_ids,
    scoring_target_alphabet,
)
from src.capability.core.io import write_json  # noqa: E402
from src.capability.core.scoring import sequence_target_mask, target_rule  # noqa: E402

SCHEMA_VERSION = "r2_transfer_second_stage_interface_qualification_v1"
DEFAULT_OUT = REPO / "results/R1/second_stage_interface_qualification"


def artefact_name(arm: str) -> str:
    """One artefact per arm, named for it, because the unit here is the arm."""

    return f"second_stage_interface_{arm}.json"


# ------------------------------------------------------------------ the probes


@dataclass(frozen=True)
class Probe:
    """A frozen probe and the frozen anagram that is its negative control.

    ``control`` is the *same symbols in a different order*. That is what makes it
    a control rather than a second measurement: content, length and composition
    are held fixed, so a cost paid on it is a cost of order alone. Both strings
    and both digests are literals here, so the control cannot be regenerated at
    run time by a permutation that drifts with a library's RNG.
    """

    label: str
    kind: str
    native: str
    native_sha256: str
    control: str
    control_sha256: str
    control_note: str


def sequence_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#: The residue probe: the first 80 residues of avGFP, a natural sequence rather
#: than an alphabet run, so the native reading is a reading on protein and not on
#: a pattern. Its control is a frozen permutation of those same 80 residues.
PROTEIN_PROBE = Probe(
    label="avgfp_n80",
    kind="protein",
    native=(
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQ"
    ),
    native_sha256="3a6033d2eb88fa724c2aab48cb19d903201b1c2efefda1b82b8edb83a03ebbcf",
    control=(
        "FVVFYRPVFPCCKEKFTYIGGMTLSEYDVLTESTNPWDLPTKSAKHVSGQGIESGQGEFLTTVGLDVKLPKDHMGVLGGT"
    ),
    control_sha256="0e9832822641d7d4c4f71a8ce1b9e93a6b7d21f2fd94d443aaa87c7ee32513bb",
    control_note="the same 80 residues in a frozen shuffled order",
)

#: The text probe: one fixed English passage. Its control is a frozen permutation
#: of its words, which is a character-level anagram of the passage and destroys
#: word order while leaving vocabulary, length and punctuation intact.
TEXT_PROBE = Probe(
    label="english_passage_47w",
    kind="text",
    native=(
        "The kestrel hovered above the motorway embankment, holding its position "
        "against a steady crosswind while the traffic below thickened into the "
        "long amber crawl of a winter evening. Nothing in the field moved for a "
        "while, and then a single vole broke cover near the fence line."
    ),
    native_sha256="5018f8415b8869e0672b7bd064adc294c3bc7fcb66eed6b417dc25811314b291",
    control=(
        "then traffic in fence line. long motorway the amber The and of single "
        "crawl against the while below winter while, the steady thickened "
        "Nothing for near above field kestrel evening. broke embankment, a cover "
        "a holding its crosswind moved position into vole the hovered a a the"
    ),
    control_sha256="44e969354554a9ed1af738dd143a10470296862ea89eb1f695f9a166ce732e1e",
    control_note="the same 47 words in a frozen shuffled order",
)

PROBES: dict[str, Probe] = {probe.kind: probe for probe in (PROTEIN_PROBE, TEXT_PROBE)}

#: Numerical determinism of one forward pass, in nats per target. Two runs of the
#: same tokens through the same weights on the same device must agree to this;
#: anything larger is non-determinism the scoring cannot be read through.
REPEAT_MAX_ABS = 1e-6

#: How much worse the anagram control must read, in nats per scored target.
#:
#: **EXP-R2-225's own number, frozen before any checkpoint on this track was
#: scored, and not ProGen2's 0.05.** That one bounds a *direction-marker* swap,
#: which moves one token of a rendering; this one bounds an order shuffle of the
#: whole probe, which is a far larger perturbation, and a floor carried over from
#: the smaller perturbation would be a bar nothing could fail. The two shuffle
#: costs this repository has already recorded on protein checkpoints are
#: ProteinGLM-7B-CLM at 1.1277 nats/residue native against 2.8974 shuffled
#: (+1.77) and ProGen3-112M at 1.983 on Swiss-Prot against 2.940 residue-shuffled
#: (+0.96). 0.25 sits an order of magnitude below the first and a factor of four
#: below the second, so it is a floor rather than a threshold tuned to a reading;
#: no Qwen or RITA shuffle cost was known when it was written.
SHUFFLE_COST_MIN = 0.25


# ------------------------------------------------------------------- the arms


@dataclass(frozen=True)
class LiveWidth:
    """This campaign's frozen live output width for one checkpoint.

    ``measured`` separates a width read from a live forward pass on this
    repository's host from one read off the checkpoint's ``config.json`` and not
    yet confirmed against a head. Both are checked identically -- a live width
    that disagrees stops the arm either way -- and the artefact says which kind
    of declaration it cleared, so an unmeasured rung cannot be reported as though
    a card had confirmed it.
    """

    size: int
    measured: bool
    source: str


#: Frozen live output widths, per checkpoint.
#:
#: Two were read from a live forward pass on this repository's hosts; two are
#: declared from the checkpoint's own ``config.json`` and have **not** been
#: confirmed against a loaded head, because every card on this host was occupied
#: when this stage was written and EXP-R2-225 forbids claiming that
#: ``Qwen2.5-32B`` runs until device memory has been measured on it. An
#: unmeasured declaration is still a check -- a head that loads at another width
#: stops the arm -- and the artefact records which kind it cleared.
REQUIRED_LIVE_WIDTH: dict[str, LiveWidth] = {
    "qwen2.5-0.5b": LiveWidth(
        151936, False, "config.vocab_size; declared, no live head read on this track"
    ),
    "qwen2.5-7b": LiveWidth(152064, True, "lm_head.out_features, measured"),
    "qwen2.5-32b": LiveWidth(
        152064, False, "config.vocab_size; declared, no live head read yet"
    ),
    "rita-xl": LiveWidth(26, True, "lm_head.out_features, measured"),
}

#: Checkpoints EXP-R2-225 stages that this repository declares **unavailable**,
#: with the measured reason. Resolved before anything is loaded, so the refusal
#: costs nothing and cannot be confused with a check that ran and failed.
DECLARED_UNAVAILABLE: dict[str, str] = {
    "proteinglm-7b-clm": (
        "7B numerical qualification has not run. The original AutoModelForCausalLM "
        "path remains unloadable: modeling_proteinglm.py line 15 reads "
        "'import torch, deepspeed' and Transformers' AST-based import check "
        "raises ImportError before the module body runs. A derived inference-only "
        "copy now serves budget continuation (gmask_sop_eos, residues 2..L); that "
        "is an implementation door, not this stage's 7B interface qualification. "
        "Reported unavailable for 7B qualification; no other model is moved into "
        "its slot"
    ),
}

#: The arms this stage can qualify, in the prereg's own order: the pure-text
#: ladder bottom to top, then the Wave B single point.
#:
#: ``qwen2.5-0.5b`` is a :data:`PANEL` member and is here for one reason: it is
#: the bottom rung of EXP-R2-225's 0.5->7->32B trajectory, and a ladder whose
#: lowest rung never met the interface check its upper rungs met is read partly
#: on an unchecked interface. Qualifying it re-admits nothing and re-measures no
#: panel estimand.
SECOND_STAGE_INTERFACE_ARMS = (
    "qwen2.5-0.5b",
    "qwen2.5-7b",
    "qwen2.5-32b",
    "rita-xl",
)

QUALIFIABLE_ARMS = tuple(SECOND_STAGE_INTERFACE_ARMS) + tuple(
    sorted(DECLARED_UNAVAILABLE)
)

LOGITS_NOT_CROPPED_NOTE = (
    "full-width log_softmax/gather; no crop, slice, or renormalisation onto the "
    "scoring-target alphabet"
)
NOT_PANEL_ADMISSION = (
    "this run does not admit a checkpoint to PANEL. qwen2.5-0.5b is already a "
    "member and gains nothing here; every other arm remains a staged non-member"
)
DESCRIPTIVE_NOT_CAUSAL = (
    "a pass here is not a causal scale claim and is not a capability result"
)
NO_KNOWLEDGE = (
    "a pass here is not evidence that a model has learned biological knowledge "
    "or language"
)
PASS_MEANS = (
    "the loader, the output interface and the anagram negative control are "
    "usable at this precision; not a capability or knowledge result"
)
UNAVAILABLE_MEANS = (
    "this repository cannot read this checkpoint's interface, for the recorded "
    "reason. It is not a capability verdict and no other model takes its slot"
)


# -------------------------------------------------------------- frozen checks


def require_probe(probe: Probe) -> Probe:
    """Refuse a probe whose literals have drifted from their frozen digests.

    The anagram property is checked here rather than asserted in prose: a control
    that is not a permutation of the probe would be measuring content as well as
    order, and the recorded cost would no longer bound what it claims to.
    """

    if sequence_digest(probe.native) != probe.native_sha256:
        raise ValueError(
            f"{probe.label}: native probe digest is "
            f"{sequence_digest(probe.native)}, not the frozen {probe.native_sha256}"
        )
    if sequence_digest(probe.control) != probe.control_sha256:
        raise ValueError(
            f"{probe.label}: control digest is "
            f"{sequence_digest(probe.control)}, not the frozen {probe.control_sha256}"
        )
    if probe.control == probe.native:
        raise ValueError(f"{probe.label}: the control is the probe itself")
    if sorted(probe.control) != sorted(probe.native):
        raise ValueError(
            f"{probe.label}: the control is not an anagram of the probe, so a "
            "cost measured on it is not a cost of order alone"
        )
    return probe


def require_qualifiable_arm(name: str) -> str:
    """Refuse a name this campaign does not stage, and say what it does."""

    if name in QUALIFIABLE_ARMS:
        return name
    raise ValueError(
        f"{name!r} is not an EXP-R2-225 checkpoint this stage qualifies; it "
        f"qualifies {list(QUALIFIABLE_ARMS)}. ProGen3 qualifies through "
        "src.capability.models.progen3.self_check and the joint wave through "
        "joint_mode_qualification.py; neither is given a second door here"
    )


def probe_for(name: str) -> Probe:
    """The probe for an arm's declared modality. One probe per modality, frozen."""

    modality = arm_spec(name).modality
    probe = PROBES.get(modality)
    if probe is None:
        raise ValueError(
            f"{name}: no frozen probe for modality {modality!r}; this stage "
            f"declares probes for {sorted(PROBES)}"
        )
    return require_probe(probe)


def require_live_width(name: str, live_width: int) -> LiveWidth:
    """Refuse a live output width other than this campaign's frozen declaration."""

    declared = REQUIRED_LIVE_WIDTH.get(name)
    if declared is None:
        raise ValueError(
            f"no frozen live output width for {name!r}; a width is declared "
            "before a checkpoint is loaded, never inferred from the head that "
            "turns up"
        )
    if int(live_width) != declared.size:
        raise ValueError(
            f"{name}: live output width must be {declared.size} "
            f"({declared.source}), got {live_width}"
        )
    return declared


def require_uncropped_logits(logits: torch.Tensor, live_width: int, *, arm: str) -> None:
    if logits.ndim != 3:
        raise ValueError(
            f"{arm}: logits must be [batch, time, width], got {tuple(logits.shape)}"
        )
    if int(logits.shape[-1]) != int(live_width):
        raise ValueError(
            f"{arm}: logits width {int(logits.shape[-1])} != live width "
            f"{live_width}; cropping is forbidden"
        )


def full_width_target_nll(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-target NLL via full-dim log_softmax and gather. No column crop."""

    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("logits must be [batch, time, width] and input_ids [batch, time]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids time dimensions disagree")
    if logits.shape[1] < 2:
        raise ValueError("need at least one scored target")
    width = int(logits.shape[-1])
    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    if int(logp.shape[-1]) != width:
        raise RuntimeError("log_softmax changed the output width")
    targets = input_ids[:, 1:]
    return -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def require_finite(values: Sequence[float], *, label: str) -> None:
    for index, value in enumerate(values):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label}: non-finite NLL at target {index}")


def mean_nll(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("no scored targets")
    return float(sum(float(value) for value in values) / len(values))


def repeat_max_abs_diff(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        raise ValueError("native NLL repeats have different lengths")
    if not first:
        raise ValueError("native NLL is empty")
    return max(abs(float(left) - float(right)) for left, right in zip(first, second))


def require_repeat(first: Sequence[float], second: Sequence[float]) -> float:
    require_finite(first, label="native run 1")
    require_finite(second, label="native run 2")
    delta = repeat_max_abs_diff(first, second)
    if delta > REPEAT_MAX_ABS:
        raise ValueError(
            f"native NLL repeat max abs diff {delta} exceeds {REPEAT_MAX_ABS}"
        )
    return delta


def shuffle_cost(mean_control: float, mean_native: float) -> float:
    return float(mean_control) - float(mean_native)


def require_shuffle_cost(cost: float) -> float:
    if not (cost > SHUFFLE_COST_MIN):
        raise ValueError(
            f"anagram control cost {cost} nats/target is not strictly > "
            f"{SHUFFLE_COST_MIN}; a self-check that cannot fail is not a "
            "self-check, and one the control passes is not one either"
        )
    return cost


def target_ids_digest(target_ids: Sequence[int]) -> str:
    payload = ",".join(str(int(value)) for value in target_ids)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


# ------------------------------------------------------------------- scoring


def render(arm: Arm, text: str) -> str:
    """The arm's own native rendering of one probe, via the panel's renderer."""

    cohort = Cohort(
        name="second_stage_interface_probe",
        kind=arm.spec.modality,
        records=[text],
        min_symbols=len(text),
        max_symbols=len(text),
        metadata={},
    )
    rendered = cohort.input_strings(arm)
    if len(rendered) != 1:
        raise ValueError(f"{arm.name}: rendering one probe returned {len(rendered)}")
    return rendered[0]


def encode_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, return_tensors=None)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("tokenizer returned a batch, not a single sequence")
        ids = ids[0]
    return [int(value) for value in ids]


def scored_target_ids(arm: Arm, token_ids: Sequence[int]) -> list[int]:
    """The targets this arm's rendering declares as content.

    Not ``token_ids[1:]``: a rendering may prefix a marker, and a position whose
    target is a marker the rendering itself added is not content under any rule.
    Resolved through the same two declarations
    :func:`src.capability.core.budget.scored_tokens` resolves -- the format's rule and
    the ids the rendering declares -- so this check cannot certify a span the
    gate does not score.
    """

    ids = torch.tensor([list(token_ids)], dtype=torch.long)
    keep = sequence_target_mask(
        ids,
        torch.ones_like(ids),
        rule=target_rule(arm.spec.input_format),
        marker_token_ids=rendering_marker_ids(arm),
    )[0]
    return [int(value) for value in ids[0][1:][keep].tolist()]


def score_token_ids(arm: Arm, token_ids: Sequence[int], *, live_width: int) -> list[float]:
    ids = torch.tensor([list(token_ids)], dtype=torch.long, device=arm.device)
    output = arm.model(input_ids=ids)
    if not hasattr(output, "logits"):
        raise TypeError(f"{arm.name}: model output has no logits")
    require_uncropped_logits(output.logits, live_width, arm=arm.name)
    nll = full_width_target_nll(output.logits, ids)
    values = [float(value) for value in nll[0].detach().cpu().tolist()]
    if len(values) != len(token_ids) - 1:
        raise ValueError(
            f"{arm.name}: scored {len(values)} targets from {len(token_ids)} tokens"
        )
    return values


def qualify_loaded_arm(arm: Arm) -> dict[str, Any]:
    """Run this campaign's frozen interface checks on one already-loaded arm.

    The strict-load counts are read off the arm rather than reconstructed here:
    this function receives an already-built arm and cannot observe how it was
    loaded, so an arm carrying none is refused instead of being recorded as a
    clean strict load on no evidence.
    """

    name = arm.name
    if arm.strict_load is None:
        raise ValueError(
            f"{name}: this arm was not loaded strictly; refusing to record a "
            "strict-load block for a check that was never performed"
        )
    probe = probe_for(name)
    alphabet = scoring_target_alphabet(arm.spec, getattr(arm.model, "config", None))
    live = output_logit_width(arm)
    live_width = int(live["size"])
    declared_width = require_live_width(name, live_width)

    native_text = render(arm, probe.native)
    control_text = render(arm, probe.control)
    native_ids = encode_ids(arm.tokenizer, native_text)
    control_ids = encode_ids(arm.tokenizer, control_text)
    if len(native_ids) < 2 or len(control_ids) < 2:
        raise ValueError(f"{name}: a rendered probe scored fewer than one target")
    native_targets = scored_target_ids(arm, native_ids)
    control_targets = scored_target_ids(arm, control_ids)
    require_scoring_target_ids(native_targets, alphabet, arm=name)
    require_scoring_target_ids(control_targets, alphabet, arm=name)

    native_1 = score_token_ids(arm, native_ids, live_width=live_width)
    native_2 = score_token_ids(arm, native_ids, live_width=live_width)
    repeat_delta = require_repeat(native_1, native_2)
    control_nll = score_token_ids(arm, control_ids, live_width=live_width)
    require_finite(control_nll, label="anagram control")
    mean_native = mean_nll(native_1)
    mean_control = mean_nll(control_nll)
    cost = require_shuffle_cost(shuffle_cost(mean_control, mean_native))
    return {
        "name": name,
        "verdict": "PASS",
        "modality": arm.spec.modality,
        "input_format": arm.spec.input_format,
        "shape": {"n_layer": int(arm.spec.n_layer), "d_model": int(arm.spec.d_model)},
        "dtype": arm.dtype,
        "strict_load": dict(arm.strict_load),
        "scoring_target_support": {
            "size": int(alphabet["size"]),
            "source": alphabet["source"],
        },
        "live_output_width": {
            "size": live_width,
            "source": live["source"],
            "declared": declared_width.size,
            "declaration_measured_live": declared_width.measured,
            "declaration_source": declared_width.source,
        },
        "probe": {
            "label": probe.label,
            "kind": probe.kind,
            "native_sha256": probe.native_sha256,
            "control_sha256": probe.control_sha256,
            "control_note": probe.control_note,
        },
        "native_target_count": len(native_targets),
        "control_target_count": len(control_targets),
        "native_target_ids_sha256": target_ids_digest(native_targets),
        "control_target_ids_sha256": target_ids_digest(control_targets),
        # Over the whole rendered id sequence, not the scored targets: the first
        # token is context rather than a target, so a residue anagram differs
        # from its probe by exactly that one element even when the renderings
        # are permutations of each other. A residue tokenizer does render an
        # anagram to the same multiset of ids; a BPE tokenizer does not, because
        # merges cross word boundaries. Measured and recorded rather than
        # required, so the artefact says which of the two happened.
        "control_is_a_token_multiset_permutation": sorted(control_ids)
        == sorted(native_ids),
        "native_nll_run_1": native_1,
        "native_nll_run_2": native_2,
        "native_mean_nll_run_1": mean_native,
        "native_mean_nll_run_2": mean_nll(native_2),
        "native_repeat_max_abs_diff": float(repeat_delta),
        "control_nll": control_nll,
        "control_mean_nll": mean_control,
        "shuffle_cost_nats_per_target": float(cost),
        "shuffle_cost_floor_nats_per_target": SHUFFLE_COST_MIN,
    }


def release_arm(arm: Arm | None) -> None:
    if arm is None:
        return
    if getattr(arm, "model", None) is not None:
        arm.model = None  # type: ignore[assignment]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_qualification_arm(name: str, *, device: str, dtype: str) -> Arm:
    return load_arm_spec(arm_spec(name), device=device, dtype=dtype, strict=True)


# ------------------------------------------------------------------ artefacts


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _envelope(name: str, *, device: str, dtype: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _timestamp(),
        "campaign": "EXP-R2-225",
        "arm": name,
        "device": device,
        "dtype": dtype,
        "not_panel_admission": True,
        "not_panel_admission_note": NOT_PANEL_ADMISSION,
        "descriptive_not_causal": True,
        "descriptive_not_causal_note": DESCRIPTIVE_NOT_CAUSAL,
        "no_knowledge": True,
        "no_knowledge_note": NO_KNOWLEDGE,
        "is_panel_member": name in PANEL,
        "is_staged_second_stage_arm": name in STAGED_SECOND_STAGE_ARMS,
    }


def unavailable_payload(name: str, *, device: str, dtype: str) -> dict[str, Any]:
    """The artefact for a checkpoint this repository declares unavailable."""

    reason = DECLARED_UNAVAILABLE[name]
    payload = _envelope(name, device=device, dtype=dtype)
    payload.update(
        {
            "verdict": "UNAVAILABLE",
            "unavailable_means": UNAVAILABLE_MEANS,
            "reason": reason,
            "loaded": False,
            "substituted": False,
        }
    )
    return payload


def build_payload(
    name: str, *, device: str, dtype: str, report: Mapping[str, Any]
) -> dict[str, Any]:
    if report.get("verdict") != "PASS":
        raise ValueError("refusing to build a payload from a non-PASS report")
    if str(report.get("name")) != name:
        raise ValueError(
            f"report is for {report.get('name')!r}, not the requested {name!r}"
        )
    payload = _envelope(name, device=device, dtype=dtype)
    payload.update(
        {
            "verdict": "PASS",
            "pass_means": PASS_MEANS,
            "logits_not_cropped": True,
            "logits_not_cropped_note": LOGITS_NOT_CROPPED_NOTE,
            "repeat_max_abs_nats": REPEAT_MAX_ABS,
            "loaded": True,
            "result": dict(report),
        }
    )
    return payload


def write_artefact(directory: Path, payload: Mapping[str, Any]) -> Path:
    verdict = payload.get("verdict")
    if verdict not in ("PASS", "UNAVAILABLE"):
        raise ValueError(f"refusing to write an artefact with verdict {verdict!r}")
    destination = Path(directory) / artefact_name(str(payload["arm"]))
    write_json(destination, dict(payload))
    return destination


def qualify_arm(
    name: str,
    *,
    device: str,
    dtype: str,
    load_fn: Callable[..., Arm] | None = None,
    score_fn: Callable[[Arm], dict[str, Any]] | None = None,
    release_fn: Callable[[Arm | None], None] | None = None,
) -> dict[str, Any]:
    """Qualify one arm, or report it unavailable without loading anything."""

    require_qualifiable_arm(name)
    if name in DECLARED_UNAVAILABLE:
        return unavailable_payload(name, device=device, dtype=dtype)
    load = load_fn or load_qualification_arm
    score = score_fn or qualify_loaded_arm
    release = release_fn or release_arm
    arm = None
    try:
        arm = load(name, device=device, dtype=dtype)
        report = score(arm)
    finally:
        release(arm)
    return build_payload(name, device=device, dtype=dtype, report=report)


def run_qualification(
    *,
    arm: str,
    device: str,
    dtype: str,
    out: Path,
    load_fn: Callable[..., Arm] | None = None,
    score_fn: Callable[[Arm], dict[str, Any]] | None = None,
    release_fn: Callable[[Arm | None], None] | None = None,
) -> Path:
    destination_dir = Path(out)
    destination = destination_dir / artefact_name(arm)
    if destination.exists():
        destination.unlink()
    destination_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload = qualify_arm(
            arm,
            device=device,
            dtype=dtype,
            load_fn=load_fn,
            score_fn=score_fn,
            release_fn=release_fn,
        )
        return write_artefact(destination_dir, payload)
    except Exception:
        if destination.exists():
            destination.unlink()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        required=True,
        choices=list(QUALIFIABLE_ARMS),
        help="the one checkpoint this invocation qualifies",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    path = run_qualification(
        arm=args.arm, device=args.device, dtype=args.dtype, out=args.out
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
