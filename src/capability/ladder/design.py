"""The ladder of modification extent, declared before anything is measured.

What this experiment asks
=========================

Across this programme a model that ranks single substitutions well does not
reliably generate better proteins. One candidate explanation is a
**local-to-global generalization failure**: a likelihood may carry real
information about one residue swapped into a fixed context and little or none
about a sequence the model had to construct itself. If that is what is
happening, then along a ladder of increasing modification extent -- built on the
*same* natural backbones throughout -- model likelihood and an independent
structural evaluation should start out agreeing and, at some extent, stop.

The ladder is therefore one axis and one axis only: the number of contiguous
residues ``k`` the model is asked to write. :data:`WINDOW_EXTENTS` is that axis.
``k = 1`` is a single substitution and ``k = 2`` a double substitution, proposed
by the model rather than drawn by us, so the bottom of the ladder is the same
construction as the top and not a different experiment with the same label.

Three design points that decide what may be claimed
===================================================

**The window rungs preserve length exactly; the full-generation rung does not.**
Both likelihood and predicted structural confidence are strongly
length-dependent in this project's own measurements, and they are dependent in
*opposite* directions, so a comparison across rungs of different length is a
comparison of lengths. Every window rung substitutes exactly ``k`` residues into
a fixed backbone, so the variant has the parent's length to the residue and the
parent's own fold is a legitimate referent. The full-generation rung has neither
property: it is an **anchor**, read as a marginal distribution, and its
parent-fold columns are empty rather than filled with a number that would not
mean the same thing. Length travels beside every confidence number regardless
(:data:`REPORT_LENGTH_WITH_CONFIDENCE`).

**"Local regeneration" is not one condition.** A left-to-right decoder cannot
condition on the residues after the window. For a causal arm the rung is
therefore *prefix* regeneration: the model writes ``k`` residues given the
prefix alone and the original suffix is restored afterwards, so its choice is
made without the downstream context. For a bidirectional or absorbing-state
arm the rung is genuine fixed-length infilling that sees both flanks. These are
:data:`CONDITION_CAUSAL_PREFIX` and :data:`CONDITION_BIDIRECTIONAL_INFILL`; they
are reported separately and **never averaged**, because the asymmetry between
them is part of the question. An arm that cannot express a rung leaves the cell
empty.

**The window position is fixed and nested.** One window per backbone per ``k``,
centred, so the ``k = 5`` window sits inside the ``k = 10`` window and so on
(:func:`window_span`). Extent is then the only thing that varies across rungs.
The price is that position dependence is not estimated at all, and no claim
about it may be read out of this design.

What the two measured quantities are, and that they are independent
===================================================================

The model likelihood of a sequence and the ESMFold2 evaluation of that sequence
are computed by different checkpoints in different processes from the same
frozen string. The structure predictor never sees a likelihood and the scorer
never sees a structure. That independence is what makes their rank correlation
interpretable at all.

Structural confidence is **not** stability and **not** function. No stability
predictor is applied anywhere in this experiment and nothing here is
experimentally verified; :data:`CEILING` says so in the artefact.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "ladder_design_v1"

#: The campaign this declaration belongs to. Names the manifest under
#: ``h200/campaigns/`` and the results directory, so an artefact can be traced
#: to the dispatch that produced it.
CAMPAIGN = "ladder_divergence_20261010"

AA20 = "ACDEFGHIKLMNPQRSTVWY"

# --------------------------------------------------------------- the ladder

#: The modification-extent axis, in residues. ``1`` is a single substitution and
#: ``2`` a double substitution; both are the model's own proposal under exactly
#: the construction the larger rungs use, which is why they are rungs of this
#: ladder rather than a separately drawn mutant set.
#:
#: A contiguous pair rather than two dispersed sites: dispersion is a second
#: factor and would be confounded with extent on a one-dimensional ladder. The
#: dispersed-double condition is not measured here and is not claimed.
WINDOW_EXTENTS: tuple[int, ...] = (1, 2, 5, 10, 20, 40)

#: The rung name for the unmatched anchor. Not a window rung: it has no parent
#: fold, no preserved length and no matched comparison.
FULL_GENERATION = "full"

#: Every rung, in ladder order.
RUNGS: tuple[str, ...] = tuple(f"k{extent}" for extent in WINDOW_EXTENTS) + (FULL_GENERATION,)


def rung_extent(rung: str) -> int | None:
    """The nominal extent of a rung, or ``None`` for the unmatched anchor."""

    if rung == FULL_GENERATION:
        return None
    if rung not in RUNGS:
        raise KeyError(f"unknown rung {rung!r}; the ladder is {list(RUNGS)}")
    return int(rung[1:])


# ------------------------------------------------------------- the conditions

CONDITION_CAUSAL_PREFIX = "causal_prefix_regeneration"
CONDITION_BIDIRECTIONAL_INFILL = "bidirectional_infill"
CONDITION_FULL_GENERATION = "full_generation"
CONDITION_COMPOSITION_SHUFFLE = "composition_matched_shuffle"

CONDITIONS: tuple[str, ...] = (
    CONDITION_CAUSAL_PREFIX,
    CONDITION_BIDIRECTIONAL_INFILL,
    CONDITION_FULL_GENERATION,
    CONDITION_COMPOSITION_SHUFFLE,
)

#: What each condition's window regeneration actually conditions on. Printed into
#: every artefact: the whole point of separating the two local conditions is lost
#: if a reader has to infer which one a number came from.
CONDITION_NOTE: dict[str, str] = {
    CONDITION_CAUSAL_PREFIX: (
        "the model writes k residues given the prefix alone; the original suffix is "
        "restored afterwards, so the choice was made without the downstream context"
    ),
    CONDITION_BIDIRECTIONAL_INFILL: (
        "fixed-length infilling: the k window positions are masked and filled by "
        "iterative unmasking with both flanks visible throughout"
    ),
    CONDITION_FULL_GENERATION: (
        "the whole sequence, conditioned on nothing beyond the arm's own native "
        "prompt. Length is not matched and no parent fold exists; this is an anchor"
    ),
    CONDITION_COMPOSITION_SHUFFLE: (
        "the k window is replaced by residues drawn from the parent's own "
        "composition. No model proposes it; it is the extent reference that says "
        "how much of a structural change comes from extent alone"
    ),
}


# ------------------------------------------------------------------- the arms


@dataclass(frozen=True)
class LadderArm:
    """One arm of the ladder, and which rungs it can express.

    ``model_class`` is the grouping the central question is asked of -- whether
    the divergence point differs by model class -- and it is declared here rather
    than derived from a name at a call site.

    ``loader`` names the declared door the weights come through:
    ``panel`` is :func:`src.capability.core.arms.load_arm`, ``lineage`` is
    :func:`src.capability.models.joint_lineage.load_rung`, ``masked_lm`` is the
    stock ``transformers.EsmForMaskedLM`` load the project's denoising roster
    already uses, and ``none`` needs no weights at all.
    """

    name: str
    model_class: str
    condition: str
    loader: str
    #: Panel arm name, lineage rung name, or staged checkpoint directory name.
    checkpoint: str
    #: Whether the arm's native rendering requires an EC class label.
    needs_ec_label: bool
    #: How a likelihood-like scalar is obtained for this arm's own products.
    likelihood_kind: str
    note: str


#: Native left-to-right sequence likelihood under the arm's own rendering.
LIKELIHOOD_CAUSAL_NLL = "causal_sequence_nll"
#: Sum of masked marginals, one position masked at a time. This is a
#: pseudo-log-likelihood: a masked language model has no sequence likelihood and
#: none is claimed. Never compared across arms or against a causal NLL.
LIKELIHOOD_MASKED_PLL = "masked_pseudo_log_likelihood"

CLASS_CAUSAL_PROTEIN = "causal_protein_decoder"
CLASS_CAUSAL_JOINT = "causal_joint_text_protein"
CLASS_BIDIRECTIONAL = "bidirectional_masked_lm"
CLASS_DIFFUSION = "absorbing_state_discrete_diffusion"
CLASS_NO_MODEL = "no_model_extent_reference"

ARMS: dict[str, LadderArm] = {
    "protgpt2": LadderArm(
        name="protgpt2",
        model_class=CLASS_CAUSAL_PROTEIN,
        condition=CONDITION_CAUSAL_PREFIX,
        loader="panel",
        checkpoint="protgpt2",
        needs_ec_label=False,
        likelihood_kind=LIKELIHOOD_CAUSAL_NLL,
        note=(
            "the unconditioned protein decoder the generation experiments use. Its "
            "rendering is FASTA hard-wrapped at 60 residues behind the end-of-text "
            "token, and a prefix prompt is a character-exact prefix of that rendering"
        ),
    ),
    "zymctrl": LadderArm(
        name="zymctrl",
        model_class=CLASS_CAUSAL_PROTEIN,
        condition=CONDITION_CAUSAL_PREFIX,
        loader="panel",
        checkpoint="zymctrl",
        needs_ec_label=True,
        likelihood_kind=LIKELIHOOD_CAUSAL_NLL,
        note=(
            "the conditioned protein decoder the generation experiments use. The EC "
            "tag is the backbone's own, so the conditioning is true rather than a "
            "mismatched label, and it is scored as a prompt and never as content"
        ),
    ),
    "prollama": LadderArm(
        name="prollama",
        model_class=CLASS_CAUSAL_JOINT,
        condition=CONDITION_CAUSAL_PREFIX,
        loader="lineage",
        checkpoint="prollama",
        needs_ec_label=False,
        likelihood_kind=LIKELIHOOD_CAUSAL_NLL,
        note=(
            "the joint text-and-protein arm, stage 2 of the ProLLaMA lineage, read "
            "under the bare Seq=<...> block the whole lineage shares rather than its "
            "own superfamily instruction form, so the prefix carries no class request"
        ),
    ),
    "prollama-stage-1": LadderArm(
        name="prollama-stage-1",
        model_class=CLASS_CAUSAL_JOINT,
        condition=CONDITION_CAUSAL_PREFIX,
        loader="lineage",
        checkpoint="prollama-stage-1",
        needs_ec_label=False,
        likelihood_kind=LIKELIHOOD_CAUSAL_NLL,
        note=(
            "the first adaptation stage of the same lineage. Present so the one "
            "within-lineage scoring/generation separation this programme already "
            "documented can be read on the ladder as well"
        ),
    ),
    "esm2-650m": LadderArm(
        name="esm2-650m",
        model_class=CLASS_BIDIRECTIONAL,
        condition=CONDITION_BIDIRECTIONAL_INFILL,
        loader="masked_lm",
        checkpoint="esm2_650m",
        needs_ec_label=False,
        likelihood_kind=LIKELIHOOD_MASKED_PLL,
        note=(
            "the bidirectional arm that can express true infilling. It needs a "
            "masked-LM head, not the generic causal interface, and it carries no "
            "sequence likelihood: its scalar is a pseudo-log-likelihood"
        ),
    ),
    "dplm-650m": LadderArm(
        name="dplm-650m",
        model_class=CLASS_DIFFUSION,
        condition=CONDITION_BIDIRECTIONAL_INFILL,
        loader="masked_lm",
        checkpoint="dplm_650m",
        needs_ec_label=False,
        likelihood_kind=LIKELIHOOD_MASKED_PLL,
        note=(
            "the absorbing-state discrete-diffusion arm, initialised from ESM2-650M, "
            "so the pair separates diffusion training from bidirectional masked "
            "scoring. The reference byprot loader has never been executed in this "
            "project; that limitation travels with every number from this arm"
        ),
    ),
    "composition-shuffle": LadderArm(
        name="composition-shuffle",
        model_class=CLASS_NO_MODEL,
        condition=CONDITION_COMPOSITION_SHUFFLE,
        loader="none",
        checkpoint="",
        needs_ec_label=False,
        likelihood_kind="scored_under_every_causal_arm",
        note=(
            "no model writes this window. It fixes how much of the structural change "
            "at extent k is extent alone, and because every causal arm scores it, it "
            "also asks whether an arm's likelihood tracks structure on sequences it "
            "did not produce"
        ),
    ),
}

#: Arms whose own products the causal likelihood scorer may be pointed at.
CAUSAL_ARMS: tuple[str, ...] = tuple(
    name for name, arm in ARMS.items() if arm.likelihood_kind == LIKELIHOOD_CAUSAL_NLL
)

#: Arms that express the true-infilling condition.
INFILL_ARMS: tuple[str, ...] = tuple(
    name for name, arm in ARMS.items() if arm.condition == CONDITION_BIDIRECTIONAL_INFILL
)


def arm(name: str) -> LadderArm:
    if name not in ARMS:
        raise KeyError(f"unknown ladder arm {name!r}; declared arms are {sorted(ARMS)}")
    return ARMS[name]


def model_classes() -> dict[str, tuple[str, ...]]:
    """Arms grouped by model class, which is the unit of the central question."""

    grouped: dict[str, list[str]] = {}
    for name, spec in ARMS.items():
        grouped.setdefault(spec.model_class, []).append(name)
    return {key: tuple(sorted(value)) for key, value in sorted(grouped.items())}


# ------------------------------------------------------------ the backbone set

#: The declared length band of the backbone set, in residues. Lower bound: this
#: project measured a 64-residue fragment of a real protein folding to mean CA
#: pLDDT 0.477, so a band that reached down there would make the parent fold
#: itself unreliable. Upper bound: k = 40 is 27% of a 150-residue chain and 13%
#: of a 300-residue one, and widening the band widens that spread, which is a
#: confound between absolute and fractional extent rather than a free choice.
BACKBONE_LENGTH_BAND: tuple[int, int] = (150, 300)

#: Admission thresholds for a backbone, on the *parent* fold and before any
#: variant exists. High confidence at full length is a precondition of the whole
#: design: a divergence can only be read against a parent the instrument is sure
#: about.
BACKBONE_MIN_MEAN_CA_PLDDT = 90.0
BACKBONE_MIN_FRACTION_CA_PLDDT_GE70 = 0.95
BACKBONE_MIN_PTM = 0.80

#: How many backbones, and how they are spread over the band. Four strata of
#: four: the strata exist so the set spans the band rather than clustering, and
#: the count is the bootstrap unit count -- 16 clusters against this package's
#: eight-unit floor.
BACKBONE_STRATA = 4
BACKBONES_PER_STRATUM = 4
N_BACKBONES = BACKBONE_STRATA * BACKBONES_PER_STRATUM

#: Independent draws per backbone per rung per arm. Declared so that a divergence
#: can be told apart from sampling noise: with 16 backbones this is 128
#: observations per ladder cell and 16 resampling units.
DRAWS_PER_CELL = 8

#: One campaign seed. Every per-cell and per-draw seed is derived from it, so the
#: whole ladder is reproducible from this integer and the declarations here.
CAMPAIGN_SEED = 20261010

#: Sampling temperature and nucleus for every arm that samples. One setting for
#: the whole ladder: a per-arm tuned decoder would make the rungs incomparable,
#: and these are the settings the project's generation experiments already use.
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 0

#: The ceiling on a causal window draw. ProtGPT2 merges about four residues into
#: a piece and ProLLaMA about 1.5, while ZymCTRL is one residue per token, so the
#: worst case is residue-level: k tokens plus slack for the rendering's own
#: newlines and for a decoder that spends a step on something else.
def max_new_tokens(extent: int) -> int:
    if extent < 1:
        raise ValueError("an extent is at least one residue")
    return int(extent) + 48


#: Length ceiling for the unmatched anchor, in residues. The band's upper bound
#: plus a margin, so the anchor is read over a comparable scale without being
#: length-matched -- which it is not, and which is said rather than implied.
FULL_GENERATION_MAX_RESIDUES = 400

#: Draws per arm for the unmatched anchor: the same budget as one arm's window
#: rung, so the anchor's own spread is measured at the same precision.
FULL_GENERATION_DRAWS = N_BACKBONES * DRAWS_PER_CELL


# ------------------------------------------------------- the structural readout

#: The independent structural evaluation that carries the main claim. A
#: TM-score-like global similarity to the parent fold, computed under the
#: residue correspondence the construction *knows* -- the window rungs preserve
#: length, so residue i of the variant is residue i of the parent and no
#: structural alignment is being inferred.
PRIMARY_STRUCTURE_READOUT = "tm_score_to_parent"

#: The local measure over the modified window, superposition-free.
LOCAL_STRUCTURE_READOUT = "window_lddt_to_parent"

#: Read beside the two above and never instead of them. Confidence is strongly
#: length-dependent, so it is a within-backbone readout here or nothing.
SECONDARY_STRUCTURE_READOUTS: tuple[str, ...] = (
    "mean_ca_plddt",
    "fraction_ca_plddt_ge70",
    "ptm",
    "mean_pae_angstrom",
    "window_mean_ca_plddt",
    "window_flank_mean_pae_angstrom",
    "window_rmsd_flank_superposed_angstrom",
)

REPORT_LENGTH_WITH_CONFIDENCE = (
    "every confidence number in this experiment is reported beside the length of "
    "the sequence it was read on, and no confidence is differenced across "
    "conditions of different length without a length control"
)

CEILING: dict[str, str] = {
    "confidence_is_not_stability": (
        "pLDDT, pTM and PAE are predicted structural confidence. They are not "
        "thermodynamic stability and not function. No stability predictor is applied "
        "anywhere in this experiment"
    ),
    "nothing_is_measured": (
        "no sequence here has been expressed, folded or assayed. Every structural "
        "number is a prediction about a string"
    ),
    "length_is_the_first_confound": REPORT_LENGTH_WITH_CONFIDENCE,
    "the_anchor_is_not_matched": (
        "the full-generation rung differs from the window rungs in length and has no "
        "parent fold. It is an anchor, not a matched comparison, and its parent-fold "
        "columns are empty"
    ),
    "two_local_conditions_are_not_one": (
        "causal prefix regeneration and bidirectional infilling condition on "
        "different information. They are reported separately and never averaged"
    ),
    "position_is_not_varied": (
        "one window start per backbone, shared by every extent, so the windows are "
        "nested. Position dependence is not estimated and no claim about it follows "
        "from this design"
    ),
    "the_start_is_snapped_per_arm": (
        "a causal arm's window start is moved to the nearest boundary of its own "
        "tokenisation, within the declared radius, because a prompt cut inside a "
        "byte-pair token is off-distribution and terminates the sequence. The "
        "realised start is recorded on every row, so the comparison of two arms' "
        "correlation *levels* is confounded by a few residues of position while each "
        "arm's own ladder shape is not"
    ),
    "composition_drift_is_not_structure": (
        "amino-acid composition distance from the parent is reported as a covariate "
        "and a composition-adjusted correlation is reported beside the raw one. A "
        "composition change is not read as a structural effect"
    ),
    "pll_is_not_a_likelihood": (
        "the bidirectional and diffusion arms have no sequence likelihood. Their "
        "scalar is a sum of masked marginals, comparable within an arm only"
    ),
    "sampler_is_declared_not_reference": (
        "the infilling sampler is the declared iterative-unmasking member in "
        "src.capability.ladder.design, not a reference release sampler. No numerical "
        "A/B against a reference loader supports it"
    ),
    "backbones_are_enzymes": (
        "every backbone carries a single EC number, because the EC-conditioned arm "
        "must be expressible on all of them. The set is therefore EC-annotated "
        "enzymes and nothing here generalises to non-enzymes"
    ),
}


# ------------------------------------------------------- windows and splicing


#: How far from the declared anchor an arm may move its window start so that the
#: prompt ends on a boundary of the arm's *own* tokenisation. It has to be
#: allowed to move at all: a prompt cut in the middle of a byte-pair token is a
#: string the arm's renderer never produces, and it was measured here --
#: ProtGPT2, prompted with a prefix whose last token was the dangling single
#: residue ``V``, emitted the end-of-text token on 8 of 8 draws, while the same
#: backbone cut two residues earlier filled every draw. An unaligned cut does not
#: measure modification extent; it measures where the merges happened to fall.
#: Twenty residues is about five tokens at ProtGPT2's measured merge rate, so an
#: aligned boundary is found well inside it or the arm genuinely cannot be
#: stopped there, which is recorded rather than worked around.
TOKEN_ALIGNMENT_RADIUS = 20


def window_anchor(length: int) -> int:
    """The declared window start for a chain of ``length`` residues.

    One start per backbone, shared by every extent, so the windows are nested --
    they share their left edge -- and the position is held fixed: extent is then
    the only thing the ladder varies. It is placed so that the *largest* extent
    sits centrally, which is what keeps every rung strictly interior.
    """

    length = int(length)
    widest = max(WINDOW_EXTENTS)
    if length < widest + 2:
        raise ValueError(
            f"a chain of {length} residues cannot carry a strictly interior window of "
            f"{widest}: it would leave no prefix or no suffix"
        )
    return max(1, (length - widest) // 2)


def window_span(length: int, extent: int, *, start: int | None = None) -> tuple[int, int]:
    """The window of ``extent`` residues, from the declared or a realised start.

    ``start`` is supplied when an arm has moved the boundary to its own
    tokenisation (:data:`TOKEN_ALIGNMENT_RADIUS`); the realised start travels on
    every row rather than being recomputed. The window is required to be strictly
    interior, so a causal prefix is never empty and a restored suffix never
    vanishes -- the downstream context the causal arm was denied has to exist for
    its absence to mean anything.
    """

    length = int(length)
    extent = int(extent)
    if extent < 1:
        raise ValueError("an extent is at least one residue")
    begin = window_anchor(length) if start is None else int(start)
    stop = begin + extent
    if begin < 1 or stop > length - 1:
        raise ValueError(
            f"window [{begin}, {stop}) of extent {extent} is not strictly interior to a "
            f"{length}-residue chain"
        )
    return begin, stop


def splice(parent: str, start: int, window: str) -> str:
    """The parent with ``window`` substituted at ``start``, length preserved.

    Refuses a window of the wrong length rather than returning a sequence of a
    different length than the parent: the length-preserving property is what the
    main claim rests on, so it is enforced here and not checked downstream.
    """

    if not parent:
        raise ValueError("a parent sequence is required")
    start = int(start)
    stop = start + len(window)
    if start < 1 or stop > len(parent) - 1:
        raise ValueError(f"window [{start}, {stop}) is not strictly interior to the parent")
    if set(window) - set(AA20):
        raise ValueError("a spliced window carries non-canonical residues")
    spliced = parent[:start] + window + parent[stop:]
    if len(spliced) != len(parent):
        raise AssertionError("splicing changed the length")
    return spliced


def realised_substitutions(parent: str, variant: str) -> int:
    """How many residues actually differ. Not the same number as the extent.

    A model asked to rewrite one residue may re-emit the original, so the nominal
    extent is an upper bound on the realised one. Both are reported, and the
    realised count is the continuous axis the finer localisation uses.
    """

    if len(parent) != len(variant):
        raise ValueError("a realised substitution count needs two sequences of one length")
    return sum(1 for left, right in zip(parent, variant) if left != right)


def causal_prefix_prompt(
    parent_rendering: str, prefix_rendering: str, *, terminal_marker: str | None = None
) -> str:
    """The prompt that puts a causal arm exactly where residue ``start`` begins.

    Both arguments come from the one declaration that renders this project's
    protein inputs (``Cohort.input_strings`` for a panel arm, the joint
    rendering for the lineage), so no rendering is spelled a second time here.
    What this function adds is the one thing a prompt built that way still needs:
    the closing marker a conditioned or delimited rendering puts after the
    sequence has to come off, and the result has to be a **character-exact
    prefix** of the parent's own rendering. That is checked rather than assumed,
    and a mismatch raises instead of being sampled from a string the arm's own
    renderer does not produce.
    """

    candidate = prefix_rendering
    if terminal_marker and candidate.endswith(terminal_marker):
        candidate = candidate[: -len(terminal_marker)]
    if not parent_rendering.startswith(candidate):
        raise ValueError(
            "the prefix rendering is not a prefix of the parent rendering; prompting "
            "from it would feed the model a string its own renderer does not produce"
        )
    return candidate


# --------------------------------------------------------------- the seeds


def cell_seed(*, arm_name: str, backbone_id: str, rung: str, draw: int) -> int:
    """A per-draw seed derived from :data:`CAMPAIGN_SEED` alone.

    Derived rather than drawn so the ladder is reproducible, and derived *per
    draw* so two cells that happen to share a prompt -- which k = 1 and k = 2 on
    one backbone very nearly do -- do not share a sample and become dependent in
    a way the backbone-clustered bootstrap does not model.
    """

    material = f"{CAMPAIGN}|{arm_name}|{backbone_id}|{rung}|{int(draw)}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
    return (CAMPAIGN_SEED + offset) % (2**31 - 1)


def variant_id(*, arm_name: str, backbone_id: str, rung: str, draw: int) -> str:
    material = f"{CAMPAIGN}|{arm_name}|{backbone_id}|{rung}|{int(draw)}".encode("utf-8")
    return "lv_" + hashlib.sha256(material).hexdigest()[:20]


def sequence_digest(sequence: str) -> str:
    """The digest the structure instrument keys its fold objects by."""

    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def variant_record(
    *,
    arm_name: str,
    condition: str,
    backbone: Mapping[str, Any],
    rung: str,
    draw: int,
    sequence: str,
    status: str,
    window_span: Sequence[int] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One ladder variant, in the shape every later stage reads.

    Three arms write these rows -- causal prefix regeneration, true infilling and
    the composition-matched extent reference -- so the shape is declared once
    here. The fields a reader must never have to reconstruct are present on every
    row whatever the outcome: the draw's own seed, the window span, the parent's
    length, and ``status``, so a draw that did not express its rung stays in the
    denominator instead of disappearing.

    ``sequence`` is empty exactly when ``status`` is not
    :data:`src.capability.ladder.runtime.DRAW_FILLED`. An empty sequence is a
    recorded outcome and the structure instrument keeps it as an ineligible row.
    """

    extent = rung_extent(rung)
    if rung == FULL_GENERATION:
        span = None
    elif window_span is not None:
        span = [int(value) for value in window_span]
    else:
        span = list(backbone["window_spans"][rung])
    row: dict[str, Any] = {
        "id": variant_id(arm_name=arm_name, backbone_id=backbone["backbone_id"], rung=rung, draw=draw),
        "arm": arm_name,
        "condition": condition,
        "rung": rung,
        "nominal_extent": extent,
        "draw": int(draw),
        "backbone_id": backbone["backbone_id"],
        "accession": backbone["accession"],
        "ec_label": backbone["ec_label"],
        "stratum": backbone["stratum"],
        "parent_length": int(backbone["length"]),
        "parent_sequence_sha256": backbone["parent_sequence_sha256"],
        "window_span": span,
        "status": status,
        "sequence": sequence,
        "sequence_sha256": sequence_digest(sequence) if sequence else "",
        "length_preserved": bool(sequence) and len(sequence) == int(backbone["length"]),
        "seed": cell_seed(arm_name=arm_name, backbone_id=backbone["backbone_id"], rung=rung, draw=draw),
    }
    if sequence:
        row.update(
            sequence_descriptors(
                sequence,
                parent=str(backbone["sequence"]) if rung != FULL_GENERATION else None,
            )
        )
        if span is not None:
            row["window"] = sequence[span[0] : span[1]]
            row["parent_window"] = str(backbone["sequence"])[span[0] : span[1]]
    row.update(dict(extra or {}))
    return row


# ------------------------------------------------------------- the controls


def composition(sequence: str) -> dict[str, float]:
    """The AA20 frequency vector, zero-filled, of one sequence."""

    if not sequence:
        raise ValueError("composition needs a sequence")
    counts = Counter(sequence)
    total = float(len(sequence))
    return {residue: counts.get(residue, 0) / total for residue in AA20}


def composition_distance(left: str, right: str) -> float:
    """Total-variation distance between two composition vectors.

    The covariate the composition adjustment conditions on. Total variation
    rather than a cosine because it is in the units the comparison is about --
    the fraction of residues that would have to move -- and because this
    repository's own matching work records that a tripeptide cosine at these
    lengths is a repeat detector and not a frequency comparison.
    """

    a, b = composition(left), composition(right)
    return 0.5 * sum(abs(a[residue] - b[residue]) for residue in AA20)


def longest_homopolymer(sequence: str) -> int:
    """The longest run of one residue. A repeat and low-complexity readout."""

    if not sequence:
        return 0
    best = run = 1
    for index in range(1, len(sequence)):
        run = run + 1 if sequence[index] == sequence[index - 1] else 1
        best = max(best, run)
    return best


def composition_entropy_bits(sequence: str) -> float:
    """Shannon entropy of the composition, in bits. Low-complexity readout."""

    frequencies = [value for value in composition(sequence).values() if value > 0.0]
    return -sum(value * math.log2(value) for value in frequencies)


def sequence_descriptors(sequence: str, *, parent: str | None = None) -> dict[str, Any]:
    """Every control readout one sequence carries into the analysis.

    Reported, never gated. A window that came out as a polyalanine run is a
    finding about the arm at that extent; dropping it would hide the finding and
    would also bias the structural distribution of whatever remained.
    """

    descriptors: dict[str, Any] = {
        "length": len(sequence),
        "longest_homopolymer": longest_homopolymer(sequence),
        "composition_entropy_bits": composition_entropy_bits(sequence),
    }
    if parent is not None:
        descriptors["composition_distance_to_parent"] = composition_distance(sequence, parent)
        if len(parent) == len(sequence):
            descriptors["realised_substitutions"] = realised_substitutions(parent, sequence)
    return descriptors


# ------------------------------------------------------- backbone admission

#: Why a candidate was not admitted. A refusal reason per candidate, so the
#: backbone set can be replayed from the pool it was drawn from.
BACKBONE_REJECTIONS: tuple[str, ...] = (
    "fold_not_ok",
    "outside_length_band",
    "noncanonical_residues",
    "below_confidence_floor",
    "no_single_ec_label",
    "near_duplicate_of_admitted",
    "stratum_full",
)


def backbone_rejection(
    row: Mapping[str, Any], *, ec_labels: Mapping[str, Sequence[str]]
) -> str | None:
    """The first admission condition a candidate fails, or ``None`` if it passes.

    Order matters only for which reason is reported; every condition is declared
    here rather than spread over the selection loop, so the artefact's attrition
    table and this function cannot drift apart.
    """

    structure = row.get("structure") or {}
    if structure.get("status") != "ok":
        return "fold_not_ok"
    low, high = BACKBONE_LENGTH_BAND
    length = int(row.get("length") or 0)
    if not low <= length <= high:
        return "outside_length_band"
    sequence = str(row.get("sequence") or "")
    if not sequence or set(sequence) - set(AA20) or len(sequence) != length:
        return "noncanonical_residues"
    if (
        float(structure.get("mean_ca_plddt", -1.0)) < BACKBONE_MIN_MEAN_CA_PLDDT
        or float(structure.get("fraction_ca_plddt_ge70", -1.0))
        < BACKBONE_MIN_FRACTION_CA_PLDDT_GE70
        or float(structure.get("ptm", -1.0)) < BACKBONE_MIN_PTM
    ):
        return "below_confidence_floor"
    accession = str(row.get("accession") or "")
    labels = list(ec_labels.get(accession) or ())
    if len(labels) != 1:
        return "no_single_ec_label"
    return None


def length_strata() -> tuple[tuple[int, int], ...]:
    """The band cut into :data:`BACKBONE_STRATA` equal-width closed strata."""

    low, high = BACKBONE_LENGTH_BAND
    edges = [low + round(index * (high - low) / BACKBONE_STRATA) for index in range(BACKBONE_STRATA + 1)]
    return tuple(
        (edges[index], edges[index + 1] if index == BACKBONE_STRATA - 1 else edges[index + 1] - 1)
        for index in range(BACKBONE_STRATA)
    )


def stratum_of(length: int) -> int:
    for index, (low, high) in enumerate(length_strata()):
        if low <= length <= high:
            return index
    raise ValueError(f"length {length} lies outside {BACKBONE_LENGTH_BAND}")


def declaration() -> dict[str, Any]:
    """The whole declaration, for the artefact that opens the campaign."""

    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "rungs": list(RUNGS),
        "window_extents": list(WINDOW_EXTENTS),
        "conditions": {name: CONDITION_NOTE[name] for name in CONDITIONS},
        "arms": {
            name: {
                "model_class": spec.model_class,
                "condition": spec.condition,
                "loader": spec.loader,
                "checkpoint": spec.checkpoint,
                "needs_ec_label": spec.needs_ec_label,
                "likelihood_kind": spec.likelihood_kind,
                "note": spec.note,
            }
            for name, spec in ARMS.items()
        },
        "model_classes": {key: list(value) for key, value in model_classes().items()},
        "backbones": {
            "length_band": list(BACKBONE_LENGTH_BAND),
            "strata": [list(bounds) for bounds in length_strata()],
            "per_stratum": BACKBONES_PER_STRATUM,
            "n_backbones": N_BACKBONES,
            "min_mean_ca_plddt": BACKBONE_MIN_MEAN_CA_PLDDT,
            "min_fraction_ca_plddt_ge70": BACKBONE_MIN_FRACTION_CA_PLDDT_GE70,
            "min_ptm": BACKBONE_MIN_PTM,
            "rejection_reasons": list(BACKBONE_REJECTIONS),
        },
        "sampling": {
            "draws_per_cell": DRAWS_PER_CELL,
            "campaign_seed": CAMPAIGN_SEED,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "full_generation_draws": FULL_GENERATION_DRAWS,
            "full_generation_max_residues": FULL_GENERATION_MAX_RESIDUES,
        },
        "structure_readouts": {
            "primary": PRIMARY_STRUCTURE_READOUT,
            "local": LOCAL_STRUCTURE_READOUT,
            "secondary": list(SECONDARY_STRUCTURE_READOUTS),
        },
        "ceiling": dict(CEILING),
    }


def require_fresh_out(out: Any, completion: str) -> None:
    """Accept an empty output directory, refuse one that already holds work.

    The campaign queue creates a cell's output directory before launching it, so
    refusing on mere existence would lose every cell of the slot. What is refused
    is evidence of prior work. A resumable instrument manages its own tree and
    does not come through here.
    """

    from pathlib import Path

    directory = Path(out)
    marker = directory / completion
    if marker.exists():
        raise SystemExit(
            f"{marker} already exists: this output directory holds a completed run. "
            "Refusing to overwrite it; point --out at a new directory"
        )
    if directory.exists():
        contents = sorted(item.name for item in directory.iterdir())
        if contents:
            raise SystemExit(
                f"{directory} is not empty (holds {contents[:8]}): refusing to write "
                "into a directory that already carries work"
            )


def read_jsonl(path: Any) -> list[dict[str, Any]]:
    """Every record of a JSONL file, refusing an empty one."""

    import json
    from pathlib import Path

    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"{path} carries no record")
    return rows


def write_jsonl(path: Any, rows: Iterable[Mapping[str, Any]]) -> str:
    """Write records in one atomic step and return the payload digest."""

    import json
    from pathlib import Path

    from ..core.io import _atomic_write

    payload = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")
    _atomic_write(Path(path), payload)
    return hashlib.sha256(payload).hexdigest()
