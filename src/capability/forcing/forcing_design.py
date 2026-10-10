"""E19: forcing an early residue and reading the consequence at its prescribed partner.

E02 asked whether a generative protein model's conditional at position ``i``
already favours the residue it will later contact. That question is **not
identifiable as an intervention** for a causal model -- the conditional at ``i``
is a function of the prefix ``w_{<i}``, so a later partner ``w_j`` is outside the
conditioning set and no manipulation of it can move the conditional. E02 therefore
measured the identifiable predictive quantity instead, and the audit closed it as
a bounded null for a reason that is structural rather than statistical: given the
anchor, the E02 estimand depends on the partner only through its residue
*identity*, so a partner-identity-matched control is identically zero and the
measured excess is alignment between the anchor's conditional and the
**composition** of its contacting partners. 56 % of the measured +0.168 nats was
reproduced by permuting the anchor within the same protein; the remainder did not
survive matching on burial band, was unresolved within either band, and rested on
three of thirty families. More compute would not have helped.

**Generation makes the same scientific question identifiable.** In generation the
prefix *is* the conditioning set, so forcing the residue at ``i`` is a genuine
intervention, and everything the model emits afterwards is a consequence of it.
This module declares that experiment before it runs.

The intervention
----------------
Take a natural backbone of length ``L`` with a trusted reference structure. Choose
an anchor ``i`` and a prescribed partner ``j`` that contact in the reference
(C-beta below :data:`CONTACT_ANGSTROM`) at sequence separation at least
:data:`MIN_SEPARATION`, with ``ANCHOR_WINDOW_FRACTION * L <= i`` and ``i`` far
enough from the C-terminus that ``j`` exists. Feed the natural prefix before
``i``, **force** position ``i``, and let the model generate the rest. The anchor
residue itself is never scored; only what the model emits after it.

The counterfactual, by reciprocal transplant
--------------------------------------------
A composition-matched counterfactual is the whole reason this cannot become a
second composition result. Each unit -- one backbone and one prescribed pair --
is matched to a unit of a *different* backbone on length band, anchor burial
band, anchor secondary-structure class and separation stratum, whose own anchor
residue differs. The two treatments are the reciprocal swap: under
:data:`CONDITION_NATIVE` each unit is forced to its own anchor residue, under
:data:`CONDITION_TRANSPLANT` each is forced to its partner's. The multiset of
forced residues is therefore identical between the two conditions **by
construction**, not by adjustment, which :func:`forced_residue_multisets_match`
checks on the realised cohort rather than asserting.

The endpoint, a double difference
---------------------------------
*Sequence channel -- the gate, and the only channel this module sizes.* For one
unit, ``D_j`` is the Jensen-Shannon divergence between the realised residue
distributions at the prescribed partner ``j`` under the two forced anchors, and
``D_j'`` is the same quantity at :data:`N_REFERENCE_PARTNERS` **non-contacting**
reference partners matched to ``j`` on sequence separation and on
relative-accessibility band in the reference structure. The burial match is the
specific lesson from E02's collapse: burial band carried the whole of that
effect. The endpoint is ``D_j - mean(D_j')`` and its null is exactly zero. A
global shift in composition or quality caused by forcing a foreign residue moves
both terms and cancels; only a change tied to the **prescribed partner position**
survives.

*Structure channel -- the claim, and only if the gate resolves.* Fold every
completion with ESMFold2 through the contract in
:mod:`src.capability.generation.structure_evidence`, on a shuffled label-stripped
manifest, and read predicted C-beta distance, ``PAE(i, j)`` and distogram mass
below the contact cutoff at the pair. The endpoint is again the double difference.
This module declares that stage and its smallest claimable effect; it does not
implement it, because a structure channel read on a gate that did not resolve
would be an answer to a question the gate said was not there.

*Free secondary decomposition.* The same intervention with the residues between
``i`` and ``j`` teacher-forced to wild type costs one forward pass per unit and
condition, because every downstream conditional is then available from a single
pass over a known sequence. The difference between the sampled and teacher-forced
endpoints isolates how much of the effect travels through the model's own
intervening commitments rather than through direct conditioning on ``i``.

What this design cannot establish, named rather than estimated
--------------------------------------------------------------
See :data:`NON_IDENTIFIABLE`. The list is part of the output.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..core.amino_acids import AA20
from ..interactions.contact_enrichment import CB_CONTACT_ANGSTROM

SCHEMA = "forcing_gate_v1"
EXPERIMENT = "E19"
PRE_REGISTRATION = "EXP-R2-229"

#: The two treatments of the reciprocal swap. ``native`` forces a unit's own
#: anchor residue, ``transplant`` forces its matched partner's. Spelled here
#: because the cohort, the generation stage and the analysis must agree on them.
CONDITION_NATIVE = "native"
CONDITION_TRANSPLANT = "transplant"
CONDITIONS = (CONDITION_NATIVE, CONDITION_TRANSPLANT)

#: The two completion modes. ``sampled`` lets the model emit ``i+1 .. j``
#: itself; ``teacher_forced`` fixes ``i+1 .. j-1`` to wild type and reads the
#: conditional at ``j`` exactly.
MODE_SAMPLED = "sampled"
MODE_TEACHER_FORCED = "teacher_forced"
MODES = (MODE_SAMPLED, MODE_TEACHER_FORCED)


# --------------------------------------------------------------- the geometry

#: Contact cutoff on the C-beta (glycine C-alpha) distance, in angstrom. Taken
#: from :mod:`src.capability.interactions.contact_enrichment`, where this project
#: declared it, so a contact here is the same relation a contact is everywhere
#: else in the repository.
CONTACT_ANGSTROM = float(CB_CONTACT_ANGSTROM)

#: Distance at or above which a reference partner counts as non-contacting.
#: Deliberately **not** the contact cutoff itself: a pair at 8.1 angstrom is in
#: van der Waals reach of its neighbour's side chain, and admitting it into the
#: control would dilute the very contrast the control calibrates, biasing the
#: double difference toward zero. Pairs between the two thresholds are in neither
#: class and are counted, not reassigned.
NON_CONTACT_ANGSTROM = 12.0

#: Sequence-separation floor for a prescribed pair. At this separation no
#: secondary-structure element guarantees proximity, so the contact is tertiary.
MIN_SEPARATION = 24

#: The anchor must lie at or beyond this fraction of the length, so the model is
#: given a real prefix to form a conditional from rather than a few residues.
ANCHOR_WINDOW_FRACTION = 0.25

#: Non-contacting reference partners per prescribed pair.
N_REFERENCE_PARTNERS = 3

#: A reference partner's sequence separation may differ from the prescribed
#: pair's by at most this many residues.
#:
#: Zero would be ideal and is unattainable: three non-contacting controls have to
#: be found among the positions within the tolerance, so a tight window leaves
#: too few candidates once the non-contact floor and the burial-band match are
#: applied. Measured on the staged release over the 127 ranked candidate
#: backbones, the number carrying at least one admissible prescribed pair is 46
#: at +-2, 79 at +-3, 97 at +-4, 112 at +-6, 114 at +-8 and 115 at +-12: the
#: yield has its knee at six and buys almost nothing beyond it.
#:
#: Six is also small in the terms this design works in. It is a quarter of the
#: separation floor, and the separation strata the transplant matches on have
#: boundaries at 32, 64 and 128, so a control sits in the prescribed pair's own
#: stratum unless the pair is within six residues of a boundary. The relaxation
#: is not assumed harmless: the realised separation imbalance between the
#: prescribed partner and its controls is reported beside every estimate, which
#: is the guard this project already applies to a contact contrast formed inside
#: a separation stratum.
SEPARATION_TOLERANCE = 6

#: Separation-stratum boundaries used as a transplant matching key. These are the
#: upper boundaries :mod:`src.capability.extensions.responses` declares for E01's
#: distance bands (8, 16, 32, 64, 128) restricted to this design's support: every
#: separation here is at least :data:`MIN_SEPARATION`, so the 8 and 16 boundaries
#: cannot be crossed and the partition on this support is the same one. Restated
#: rather than imported because that module pulls torch into what is otherwise a
#: CPU cohort and analysis path.
SEPARATION_EDGES: tuple[int, ...] = (32, 64, 128)
SEPARATION_NAMES: tuple[str, ...] = ("24-31", "32-63", "64-127", "128+")


def separation_stratum(separation: int) -> str:
    """The declared separation stratum of one pair, refusing a separation below the floor."""

    value = int(separation)
    if value < MIN_SEPARATION:
        raise ValueError(
            f"separation {value} is below the {MIN_SEPARATION}-residue floor; a pair "
            "this close along the chain is not in this design's population"
        )
    for index, edge in enumerate(SEPARATION_EDGES):
        if value < edge:
            return SEPARATION_NAMES[index]
    return SEPARATION_NAMES[-1]


# ------------------------------------------------------------ the length bands

#: The two fixed length bands, named and half-width. Confidence and likelihood
#: both correlate with length, so nothing is ever compared across these two.
LENGTH_BANDS: tuple[tuple[str, int, int], ...] = (
    ("short", 120, 10),
    ("long", 240, 10),
)

#: Backbones per length band.
BACKBONES_PER_BAND = 60

#: Prescribed pairs per backbone.
PAIRS_PER_BACKBONE = 3

#: Sampled completions per unit and condition.
DRAWS_PER_CELL = 24

BAND_COMPARISON_REFUSAL = (
    "predicted structural confidence and sequence likelihood both vary strongly "
    "with length, so no quantity is pooled or compared across the short and long "
    "bands; each band carries its own estimate and the pooled estimate is a "
    "band-equal average of unit values, never a mixture of the two length regimes"
)


def length_band(length: int) -> str | None:
    """The declared band a length falls in, or ``None`` if it falls in neither."""

    for name, centre, half_width in LENGTH_BANDS:
        if abs(int(length) - centre) <= half_width:
            return name
    return None


def band_bounds(name: str) -> tuple[int, int]:
    for band, centre, half_width in LENGTH_BANDS:
        if band == name:
            return centre - half_width, centre + half_width
    raise KeyError(f"unknown length band {name!r}; declared: {[b for b, _, _ in LENGTH_BANDS]}")


# ------------------------------------------------------- the admission filters

#: Three confidence floors, because "a trusted reference structure" is a claim
#: about different regions for different purposes.
#:
#: :data:`WINDOW_PLDDT_FLOOR` is the design's own requirement -- mean reference
#: pLDDT over the **window the intervention reads**, which is the anchor through
#: the furthest read position. It is the region the prescribed contact and its
#: controls are taken from, so it is the region that has to be trustworthy.
#:
#: :data:`RESIDUE_PLDDT_FLOOR` applies to every position actually used, one at a
#: time: a confident window does not license an anchor the predictor was unsure
#: of.
#:
#: :data:`MODEL_PLDDT_FLOOR` is the weaker whole-model screen, reused from this
#: project's existing declaration of a confident AlphaFold model
#: (``scripts/ops/build_composition_matched_fold_set.MODEL_MEAN_PLDDT_FLOOR``). It
#: exists because a globally unreliable model is not a reference structure even
#: when one window of it is confident, and because the structure channel -- if the
#: gate opens it -- reads the whole fold rather than the window. Applying the
#: 90 floor to the whole model instead was measured on the staged release and
#: removed 970 of 1,317 banded candidates, which is a much stronger requirement
#: than the design states.
WINDOW_PLDDT_FLOOR = 90.0
RESIDUE_PLDDT_FLOOR = 90.0
MODEL_PLDDT_FLOOR = 70.0

#: Pairwise identity ceiling within the admitted cohort, in percent over the
#: shorter sequence. Above this two backbones are one unit, not two.
PAIRWISE_IDENTITY_MAX = 30.0

#: Distinct-5-mer fraction a backbone must reach: ``|shingles| / (L - 4)``, using
#: the residue shingle length :mod:`src.capability.generation.near_duplicates`
#: already declares. A repeat or low-complexity sequence falls below it. The
#: threshold is a declared screen, not a calibrated one; a backbone it excludes
#: is excluded, and the per-draw version of the same quantity flags completions.
MIN_DISTINCT_SHINGLE_FRACTION = 0.90

#: A backbone is single-domain when the CATH table assigns it exactly one
#: superfamily. The superfamily is also the deduplication key and the bootstrap
#: unit, so one backbone per superfamily is admitted.
SINGLE_DOMAIN_RULE = "exactly one CATH superfamily in the extracted protein2ipr table"

#: The ordered admission filters, as the census reports them. The order is the
#: order they are applied in, cheapest first, so a reader can see the funnel.
FILTER_ORDER: tuple[str, ...] = (
    "alphafold_models",
    "single_fragment",
    "length_band",
    "swissprot_reviewed",
    "sequence_matches_swissprot",
    "canonical_residues",
    "model_plddt",
    "distinct_shingle_fraction",
    "single_cath_superfamily",
    "one_per_superfamily",
    "pairwise_identity",
    "prescribed_pairs_available",
    "band_quota",
    "transplant_partner_available",
)

#: Backbones are ranked for the band quota by mean reference pLDDT, descending,
#: with the accession as tie-break. Model-blind and reference-only: no likelihood,
#: no generated sequence and no endpoint enters the selection.
BAND_QUOTA_PRIORITY = "descending mean reference pLDDT, then accession"

TSUBOYAMA_NOT_REUSABLE = (
    "the existing 30-domain Tsuboyama set is not reusable here: at 37-72 residues "
    "it cannot supply separation->=24 pairs in quantity, and it sits in the steep "
    "part of the length-confidence curve where a structure read-out is dominated "
    "by length"
)


# ----------------------------------------------------------- the burial bands

#: Relative-accessibility boundary separating the buried and exposed bands.
#: A fixed boundary rather than a median split of whatever the cohort happens to
#: contain, because the anchor band is a **matching key** shared between two
#: backbones and a cohort-dependent boundary would move when the cohort moved.
#: 0.25 is the conventional buried/exposed boundary on Tien-2013-normalised
#: relative accessibility.
RSA_BOUNDARY = 0.25

RSA_METHOD = (
    "un-clipped Tien-2013-normalised Shrake-Rupley accessibility at 256 sphere "
    "points with a 1.4 angstrom probe, over all heavy atoms of the isolated "
    "AlphaFold chain, through the implementation "
    "src.capability.interactions.contact_enrichment already declares"
)

#: The three secondary-structure classes of the CA-trace assignment, in the
#: integer order :func:`src.capability.generation.structure_inputs.ca_secondary_structure`
#: returns. Coordinate-only and not DSSP: it is used here as a matching key, and
#: its absolute fractions are never quoted as secondary-structure content.
SS_CLASSES = ("helix", "strand", "coil")

#: The keys a reciprocal transplant must agree on.
TRANSPLANT_MATCH_KEYS = (
    "length_band",
    "anchor_rsa_band",
    "anchor_ss_class",
    "separation_stratum",
)


def rsa_band(rsa: float | None) -> str | None:
    """``buried`` / ``exposed`` on the declared boundary, or ``None`` if absent."""

    if rsa is None:
        return None
    value = float(rsa)
    if not math.isfinite(value):
        raise ValueError("a relative accessibility is finite or absent, never infinite")
    return "buried" if value < RSA_BOUNDARY else "exposed"


def ss_class(assignment: int) -> str:
    index = int(assignment)
    if not 0 <= index < len(SS_CLASSES):
        raise ValueError(f"secondary-structure assignment {assignment!r} is outside {SS_CLASSES}")
    return SS_CLASSES[index]


# --------------------------------------------------------------- the decoding

#: The decoding policy. The gate measures the realised residue distribution, and
#: "realised" is only defined relative to a policy, so these three numbers are
#: part of the estimand rather than runner settings. They are the same values
#: :mod:`src.capability.generation.conditioned_generation` froze for E11 --
#: deliberately, so the two campaigns sample comparably -- but they are restated
#: here instead of imported: coupling this estimand to another campaign's frozen
#: parameters would let a revision there silently redefine what this experiment
#: measured.
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 0

DECODING_POLICY_NOTE = (
    "temperature 1.0, top-p 0.95, top-k off: the realised residue distribution is "
    "the distribution under this policy, and the policy is therefore part of the "
    "estimand. The values match E11's frozen generation parameters so the two "
    "campaigns sample comparably, and are restated rather than imported so a "
    "revision there cannot redefine this measurement"
)

#: Inference precision. float32, for two independent reasons, both measured.
#:
#: RITA's released attention implementation multiplies a float32 mask against the
#: value tensor and raises ``expected scalar type Float but found BFloat16``, so
#: bfloat16 is not an option for that arm at all. And a reduced-precision logit
#: row quantises many logits to exactly equal values, which on a real
#: progen2-medium cell moved the nucleus boundary and changed the sampling
#: distribution by up to 0.026 in probability -- against a claimable effect of
#: 0.054 nats, that is not round-off worth accepting for speed. This is also the
#: precision :func:`src.capability.position.position_likelihood.default_dtype`
#: declares for every arm here; the floor is restated rather than imported
#: because that declaration is about a position-likelihood door and this is a
#: generation door, and because importing it would pull a torch dependency into
#: a design module that has none.
DTYPE = "float32"

#: Campaign sampling seed. Per-cell seeds are derived from it so that two cells
#: sharing a prompt do not share a sample.
SAMPLING_SEED = 20261010

#: Draws of the forced-anchor label permutation that calibrates the endpoint, and
#: its seed.
PERMUTATION_DRAWS = 2000
PERMUTATION_SEED = 20261010

#: Bootstrap draws and seed for the unit-level interval.
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261010


def cell_seed(*, arm: str, unit: str, condition: str, mode: str, seed: int = SAMPLING_SEED) -> int:
    """A per-cell sampling seed derived from the campaign seed alone.

    One seed for the whole campaign would give the two conditions of a unit the
    same sample whenever their prompts coincide, making them dependent in a way
    the unit bootstrap does not model. Derived rather than drawn, so the campaign
    is reproducible from :data:`SAMPLING_SEED`.
    """

    import hashlib

    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; declared: {CONDITIONS}")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; declared: {MODES}")
    material = f"{PRE_REGISTRATION}|{arm}|{unit}|{condition}|{mode}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
    return (int(seed) + offset) % (2**31 - 1)


# ------------------------------------------------------------------- the arms

#: The four causal protein arms. Three rungs of one declared scale ladder, whose
#: rendering and pretraining corpus are identical so that the endpoint can be
#: read against scale, plus one independent architecture and corpus.
#:
#: Admissibility is measured, not assumed: :func:`forced_completion.residue_grid`
#: refuses an arm whose tokenisation is not one token per residue or whose
#: tokenisation of a prefix is not the prefix of its tokenisation of the whole.
#: ProtGPT2 fails both (measured on the staged checkpoint: 29 ids for 78
#: residues, and a prefix retokenises), which is why a multi-residue BPE arm
#: cannot carry a position-level forcing intervention at all.
ARMS: tuple[str, ...] = ("progen2-medium", "progen2-large", "progen2-xlarge", "rita-xl")

ARM_PANEL_NOTE = (
    "progen2-medium/large/xlarge are the three staged rungs of the declared "
    "protein scale ladder: one rendering, one pretraining corpus (uniref90_bfd30), "
    "so their three endpoints differ in scale alone. rita-xl supplies an "
    "independent architecture and an independent corpus (uniref100). "
    "protgpt3-1.3b is admissible on the tokenisation gate but declares no "
    "pretraining corpus, so its reference-database-coverage stratum would not be "
    "interpretable, and it is recorded as admissible-but-unselected rather than "
    "silently dropped. protgpt2 is refused by the tokenisation gate. Masked and "
    "bidirectional arms cannot enter: their conditional at the anchor has already "
    "seen the partner, so they are an upper reference at best, never competitors"
)

UNSELECTED_ARMS: Mapping[str, str] = {
    "protgpt3-1.3b": "admissible on the tokenisation gate; pretraining corpus undeclared",
    "protgpt2": "refused: multi-residue BPE, so position-level forcing is undefined",
    "zymctrl": "refused: its native rendering requires an EC class request this design does not make",
    "proteinglm-7b-clm": "refused: its rendering wraps the sequence in gmask/sop markers, so a prefix is not a prefix",
}


# ------------------------------------------------------------ the pre-registration

#: Per-unit standard deviations the sizing rests on, declared before the run.
#: Both are assumptions about dispersion, not measurements; the realised
#: dispersion is reported beside the estimate and the pre-registered threshold is
#: not revised to meet it.
ASSUMED_UNIT_SD = {
    "sequence_double_difference": 0.21,
    "structure_contact_realisation": 0.15,
}

#: Pre-registered smallest claimable effects. The structural threshold is the
#: design's rounded-up value and is therefore *more* conservative than the
#: computed minimum detectable effect; the computed value is reported beside it
#: so the gap is visible.
CLAIMABLE = {
    "sequence_double_difference": 0.054,
    "structure_contact_realisation": 0.05,
}

POWER = 0.80
ALPHA = 0.05


def minimum_detectable_effect(
    *, unit_sd: float, units: int, power: float = POWER, alpha: float = ALPHA
) -> dict[str, float]:
    """Two-sided normal-approximation MDE for a mean over independent units.

    Derived rather than quoted, so a cohort that lands at a different unit count
    reports the effect size it actually has power for instead of the one the
    design hoped for.
    """

    if unit_sd <= 0 or not math.isfinite(unit_sd):
        raise ValueError("a per-unit standard deviation is positive and finite")
    if units < 2:
        raise ValueError("a minimum detectable effect needs at least two units")
    if not 0.5 <= power < 1.0 or not 0.0 < alpha < 1.0:
        raise ValueError("power lies in [0.5, 1) and alpha in (0, 1)")
    standard_error = float(unit_sd) / math.sqrt(int(units))
    z_alpha = _normal_quantile(1.0 - alpha / 2.0)
    z_power = _normal_quantile(power)
    return {
        "unit_sd": float(unit_sd),
        "units": int(units),
        "standard_error": standard_error,
        "minimum_detectable_effect": (z_alpha + z_power) * standard_error,
        "power": float(power),
        "alpha": float(alpha),
    }


def _normal_quantile(probability: float) -> float:
    """Standard-normal quantile, by bisection on ``erf``.

    Ten lines rather than a SciPy dependency this package does not otherwise
    take, and exact to machine precision on the two quantiles the sizing needs.
    """

    if not 0.0 < probability < 1.0:
        raise ValueError("a quantile probability lies strictly inside (0, 1)")
    low, high = -40.0, 40.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if 0.5 * (1.0 + math.erf(middle / math.sqrt(2.0))) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def pre_registration(
    *, units: int = BACKBONES_PER_BAND * len(LENGTH_BANDS), pairs: int | None = None,
) -> dict[str, Any]:
    """The whole pre-registration as one record, for the artefact's header.

    ``units`` is the realised backbone count and ``pairs`` the realised
    prescribed-pair count, so an artefact states the effect size and the sampling
    cost it actually has rather than the ones the design hoped for. Both default
    to the designed figures; the designed and realised completion counts are
    reported side by side so the gap is a number rather than an omission.
    """

    designed_pairs = BACKBONES_PER_BAND * len(LENGTH_BANDS) * PAIRS_PER_BACKBONE
    realised_pairs = designed_pairs if pairs is None else int(pairs)
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "pre_registration": PRE_REGISTRATION,
        "sampling_unit": SAMPLING_UNIT,
        "units": int(units),
        "sizing": {
            channel: minimum_detectable_effect(unit_sd=sd, units=units)
            | {"pre_registered_claimable": CLAIMABLE[channel]}
            for channel, sd in ASSUMED_UNIT_SD.items()
        },
        "prescribed_pairs": realised_pairs,
        "designed_prescribed_pairs": designed_pairs,
        "completions_per_arm": realised_pairs * len(CONDITIONS) * DRAWS_PER_CELL,
        "designed_completions_per_arm": designed_pairs * len(CONDITIONS) * DRAWS_PER_CELL,
        "teacher_forced_passes_per_arm": realised_pairs * len(CONDITIONS),
        "arms": list(ARMS),
        "arm_panel_note": ARM_PANEL_NOTE,
        "unselected_arms": dict(UNSELECTED_ARMS),
        "decoding": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "dtype": DTYPE,
            "note": DECODING_POLICY_NOTE,
        },
        "structure_channel": STRUCTURE_CHANNEL_CONTRACT,
        "band_comparison_refusal": BAND_COMPARISON_REFUSAL,
        "tsuboyama_not_reusable": TSUBOYAMA_NOT_REUSABLE,
        "non_identifiable": dict(NON_IDENTIFIABLE),
        "confounds": dict(CONFOUNDS),
        "interpretation": INTERPRETATION,
        "residue_alphabet": AA20,
    }


SAMPLING_UNIT = (
    "the backbone, which is also its CATH superfamily because one backbone per "
    "superfamily is admitted. A unit's value is the equal-weighted mean over its "
    "prescribed pairs; draws are never the resampled unit, and families are never "
    "pooled across metrics"
)


# ------------------------------------------------- the confounds and the limits

CONFOUNDS: Mapping[str, str] = {
    "length_and_completeness": (
        "every draw in a cell targets the same L and the same span i+1..j, so the "
        "two conditions of a unit are length-identical by construction. A draw "
        "that emits an end token before reaching j is recorded as a CENSORED "
        "outcome and the endpoint is reported both all-draws and completed-only; "
        "confidence and likelihood are never compared across the two length bands"
    ),
    "fold_level_composition": (
        "handled by the reciprocal swap, which equalises the forced-residue "
        "multiset by construction rather than by adjustment; completion "
        "composition is reported per cell and the endpoint is additionally "
        "reported within bins of the completion's own hydrophobic fraction"
    ),
    "repeat_and_low_complexity": (
        "backbones below the distinct-shingle floor are excluded, every draw "
        "carries the same quantity as a flag, and the endpoint is reported with "
        "and without flagged draws"
    ),
    "reference_database_coverage": (
        "a UniRef90 search on the host that holds the index bands every backbone "
        "by best-hit identity; the endpoint is stratified by band and the remote "
        "band is never pooled with the close band. Coverage is not pretraining "
        "exposure and is not reported as one"
    ),
    "duplication": (
        "near-identical completions inside a cell are grouped by the existing "
        "union-find shingle grouping and a group contributes once to the realised "
        "distribution, so a cell cannot be carried by one repeated completion"
    ),
    "simultaneous_inference": (
        "a panel-wide claim across arms is made only through a simultaneous band "
        "over shared unit draws; marginal per-arm intervals are not simultaneous "
        "statements"
    ),
}

NON_IDENTIFIABLE: Mapping[str, str] = {
    "partner_identity_matched_scoring_contrast": (
        "identically zero by construction -- given the anchor, E02's statistic "
        "depends on the partner only through its residue identity. E02 settles it "
        "and it is not attempted here"
    ),
    "structural_chemistry_versus_learned_coevolution": (
        "there is no coevolution-free holdout, so a model-free pair conditional "
        "from the family's own alignment is reported as a REFERENCE LEVEL and "
        "never subtracted as a baseline. An effect that does not exceed it is "
        "reported as reproducing coevolution, not as structural anticipation"
    ),
    "mechanism_attribution_to_the_anchor": (
        "forcing i perturbs the prefix of every later position, so the "
        "partner-versus-matched-non-partner double difference is the strongest "
        "available guard and still cannot establish knowledge of the contact"
    ),
    "absolute_contact_realisation_across_arms": (
        "not comparable; only within-arm double differences are"
    ),
    "bidirectional_and_masked_arms": (
        "cannot enter as competitors -- their conditional at the anchor has "
        "already seen the partner -- and are carried, if at all, as an upper "
        "reference"
    ),
    "stability_and_function": "out of scope entirely; nothing here measures either",
}

INTERPRETATION = (
    "The gate measures whether forcing an early residue changes what the model "
    "emits at a prescribed contacting position MORE than at matched "
    "non-contacting positions. A resolved positive says the prefix carries "
    "position-specific consequence, not that the model knows the contact. "
    "ESMFold2 confidence, where the structure channel is later read, is predicted "
    "structural compatibility only -- never thermodynamic stability, never "
    "function, and never experimentally verified folding. ESMFold v1's contact "
    "head has randomly initialised weights and is refused."
)

STRUCTURE_CHANNEL_CONTRACT = (
    "ESMFold2 (biohub/ESMFold2-hf) through "
    "src.capability.generation.structure_evidence, run on a shuffled "
    "label-stripped manifest so the folder cannot see the treatment. It is "
    "independent of the generative arms -- different weights, different "
    "objective, no shared head -- but it is NOT independent of sequence "
    "databases. Pairwise confidence (PAE, distogram) is kept and never reduced to "
    "a scalar. This channel is declared here and is read only if the gate "
    "resolves"
)


def forced_residue_multisets_match(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Whether the reciprocal swap really equalises the forced-residue multiset.

    The design's central guarantee, checked on the realised cohort rather than
    asserted. Returns the two multisets and whether they agree; a caller that
    needs a refusal raises on ``matches`` being false.
    """

    from collections import Counter

    native = Counter(str(unit["native_residue"]) for unit in units)
    transplant = Counter(str(unit["transplant_residue"]) for unit in units)
    return {
        "native": dict(sorted(native.items())),
        "transplant": dict(sorted(transplant.items())),
        "matches": native == transplant,
        "rule": (
            "every matched pair contributes its two anchor residues to both "
            "conditions, so the two multisets are equal over any set of units "
            "closed under the matching"
        ),
    }
