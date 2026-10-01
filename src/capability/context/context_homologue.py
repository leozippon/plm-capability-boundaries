"""In-context homologue conditioning in frozen, single-sequence-pretrained decoders.

EXP-R2-228. **The question is whether an in-context conditioning gain
exists at all in a decoder that never saw a homologue set during pretraining**,
measured against controls that separate evolutionary relatedness from in-context
copying, with the identical estimand run on a text decoder.

The hypothesis this module is built to defeat
=============================================

Kantroo, Wagner & Machta (arXiv:2504.17068, 2025) measure **ProGen2-M**, which is
this repository's ``progen2-medium``. Appending a copy of a sequence to itself
collapses its likelihood; the collapse also occurs for **random, non-natural**
sequences; it persists to **50% sequence divergence**; and it fires on needles as
small as **10 residues**. That is an induction/copying mechanism producing exactly
the signal an in-context homology experiment is designed to detect, for reasons
that have nothing to do with homology.

Copying is therefore the leading hypothesis, not one control among several, and
the whole control structure follows from it:

* **Identity band is the primary design factor.** Global identity demonstrably
  does not screen the mechanism, so it cannot be a covariate to adjust for; it is
  the axis. ``>= 90`` is excluded at construction, and the four remaining bands
  are measured separately.
* **A local-overlap screen sits underneath the identity band.** Every (target,
  context item) pair carries its longest common substring and its shared 7-mer
  count. A pair at or above :data:`HIGH_LOCAL_OVERLAP_LCS` never enters a primary
  context; those pairs are collected into their own stratum and **scored**, so the
  copying effect is measured rather than only excluded.
* **The decisive stratum is the bottom local-overlap tercile of the ``< 30``
  band**, where no long shared substring exists and global identity is minimal.
  Measured on the frozen pool, that stratum's longest common substrings sit at 4-6
  residues -- *below* Kantroo et al.'s 10-residue needle, which is the property
  clause 3 of the gate depends on.

What is fixed and what is an outcome
====================================

**The budget is tokens, not k.** For each target the context budget is the arm's
:data:`POSITION_BUDGET` positions minus that target's own rendered length. Items
are added under a frozen order until the budget is exhausted, so **k is an
outcome, reported per arm as a distribution**, and it is the *fraction of the
window one item consumes* that is matched across arms rather than the item count.

Within one target every condition is built to the **same k** and to the **same
per-item token lengths**, so the paired contrast holds position, length and item
count fixed and varies only content.

Endpoints
=========

Both primary endpoints are dimensionless and paired:

``auroc``
    Over units, the probability that a true-homologue context yields a lower
    target NLL than its own matched-unrelated context, ties at one half. A
    discrimination statistic with no unit and no denominator -- the one quantity
    here that is legitimately comparable across arms and across modalities, and,
    being task-shaped rather than loss-shaped, the endpoint Bertsch et al.'s
    "next-token loss keeps falling while capability plateaus" warning does not
    undermine.
``fractional_reduction``
    The paired homologue-minus-matched-unrelated difference divided by that
    arm's own position-only-context NLL. Dimensionless, within-arm, and the
    denominator travels with it in every record (Appendix B rule 27).

**ΔNLL in nats per scored token is a descriptive curve and nothing more.** It is
reported per arm beside that arm's measured symbols per token and is never
differenced across arms (L23, Appendix B rule 26).

**Never (k minus k = 0).** :data:`NO_CONTEXT` is scored because the
registration's own failure branch -- "the position-only control moves the target's
NLL as much as the homologue condition" -- cannot be checked without it, and for
no other reason. :func:`gate` refuses to read it, and every record carrying it is
labelled :data:`DIAGNOSTIC_NEVER_THE_EFFECT`.

The mono-shuffle, measured rather than assumed
=============================================

The registration asks for "mono-residue-shuffled homologues" whose composition is
preserved *exactly*, and separately for every control to be **token-length-matched
per arm**. On a residue-level tokeniser those are one operation. On ProtGPT2 they
are two, and three things were measured before the shuffle was written.

*A token-level shuffle is not an acceptable substitute on this arm.* ProtGPT2's
BPE pieces reach **32 residues** on natural Swiss-Prot sequences in the frozen
pool (median 3, 95th percentile 4). Permuting tokens would leave runs three times
Kantroo et al.'s 10-residue needle intact inside a single piece, which is exactly
the copying signal this control exists to destroy. So the protein shuffle is
residues, as the registration froze it, and the text shuffle is tokens, as the
registration froze it (:data:`SHUFFLE_UNITS`).

*A residue permutation costs a few per cent more tokens, with a heavy tail.* Over
300 pool records the rendered token count of a permutation is a median **1.027x**
the natural sequence's, 1.111x at the 90th percentile and up to 2x. Summed over a
k of up to 37 items that is a context tens to hundreds of tokens longer than the
homologue context it must match -- a control that varies length as well as
content.

*Most items admit a permutation of exactly the right length.* Within 24 seeds,
**77%** of items have a residue permutation rendering to exactly the natural token
count. :func:`shuffled_item_ids` therefore searches :data:`SHUFFLE_SEED_BUDGET`
seeds for an exact match and takes the first, which preserves composition exactly,
destroys order exactly and matches the token length exactly. Where no seed inside
that budget matches, the closest longer candidate is truncated to the required
token count -- a uniform random subsample of the item's residues, because a prefix
of a permutation is one -- and every such item is counted in the plan as
``shuffle_trimmed_items``. Conditioning on token count does mildly select
permutations whose local statistics are more BPE-compressible, which makes the
control look slightly *more* natural and the design correspondingly more
conservative; it is declared here rather than left implicit.

**The corpus draw seed.** The registration freezes ``20260826`` for this
campaign's own draws. The repository's cohort-draw contract requires every stage
that calls a corpus constructor to expose ``--cohort-draw-seed`` defaulting to the
one imported :data:`~src.capability.core.arms.DEFAULT_CORPUS_DRAW_SEED`, and forbids a
stage from restating a seed as a literal. Both are honoured: the flag carries the
panel default and the campaign manifest pins it to :data:`DRAW_SEED`, which the
cohort artefact records and its digest covers.

The panel, and the three exclusions declared rather than dropped
================================================================

See :data:`PROTEIN_ARMS`, :data:`TEXT_ARMS` and :data:`EXCLUDED_ARMS`.
:data:`CEILING` carries the registration's binding ceiling and travels in every
artefact this module writes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..core.amino_acids import AA20
from ..core.arms import EOS_BOUNDED_BOUNDARY, Arm, Cohort, PANEL, config_context_length
from ..core.statistics import bootstrap_unit_floor

SCHEMA_VERSION = "r2_context_homologue_v1"
PRE_REGISTRATION = "EXP-R2-228"

#: The registration's binding ceiling, written before any score existed. It
#: travels in every artefact so that a reader of the artefact alone cannot take
#: the reading further than the registration allows.
CEILING: dict[str, str] = {
    "not_a_knowledge_or_mechanism_claim": (
        "a positive says a behaviour exists; it does not say which heads, layers "
        "or circuits produce it, and this campaign runs no intervention. It does "
        "not reopen the induction line and does not contribute to it"
    ),
    "a_positive_is_not_homology_understanding": (
        "it is a statement that context relatedness lowers a target's NLL beyond "
        "four matched controls on one cohort. F15 stands over it: an "
        "alignment-level screen does not exclude profile-level homology, so even "
        "the < 30 band is not a homology-free stratum"
    ),
    "a_negative_is_bounded_to_frozen_decoders_at_this_budget": (
        "it is not evidence against in-context homologue conditioning, which "
        "PoET, ProtMamba, E1, Protriever and ProFam already establish for models "
        "trained for it"
    ),
    "cross_modality_comparison_is_confined_to_the_dimensionless_endpoints": (
        "no nats-per-token magnitude, no k, and no band label crosses the "
        "modality boundary"
    ),
    "the_text_and_protein_band_definitions_are_different_quantities": (
        "a BM25 rank band and a DIAMOND identity band are never differenced, "
        "ranked together, or plotted on a shared axis"
    ),
    "protgpt2_carries_a_prior_multi_sequence_exposure": (
        "ProtGPT2 was pretrained on FASTA-formatted UniRef50 in which sequences "
        "are hard-wrapped at 60 residues and separated by the end-of-text token, "
        "with its BPE merges learned over exactly that byte stream -- a "
        "multi-sequence window the ProGen2 arms never saw. A pattern in which "
        "ProtGPT2 shows a gain and the ProGen2 arms do not is consistent with "
        "that prior exposure and licenses no tokenisation, modality or scale "
        "reading. It is a confound of the arm, not of the modality"
    ),
    "no_claim_about_scale": (
        "progen2-small and progen2-medium are two rungs on one lineage at one "
        "budget; this campaign forms no scale gate and may not receive the "
        "descriptive_gate_transition label"
    ),
    "not_a_claim_about_in_context_learning_as_a_general_capacity": (
        "nothing here is about few-shot task learning or a scaling law in either "
        "modality, which is the reading the Breslow et al. headline would invite"
    ),
}

#: Reported beside any quantity derived from the k = 0 condition.
DIAGNOSTIC_NEVER_THE_EFFECT = (
    "k versus k = 0 confounds content with position and with the mere presence of "
    "a filled context. This quantity exists only to price the position-only "
    "control against the registration's own failure branch and is never the effect"
)

# ------------------------------------------------------------------- the arms

#: F16 names keep working exactly as they do today. The expansion wave adds
#: protein arms under the same imposed 1024-token budget; it does not rescore
#: the F16 four and does not admit ZymCTRL or further text arms.
F16_PROTEIN_ARMS: tuple[str, ...] = ("protgpt2", "progen2-small", "progen2-medium")
F16_TEXT_ARMS: tuple[str, ...] = ("gpt2-large",)
EXPANSION_PROTEIN_ARMS: tuple[str, ...] = (
    "progen2-base",
    "progen2-large",
    "progen2-xlarge",
    "progen3-112m",
    "progen3-3b",
    "proteinglm-7b-clm",
    "protgpt3-1.3b",
    "rita-xl",
    "galactica-125m",
    "galactica-1.3b",
    "galactica-6.7b",
    "galactica-30b",
    "instructprotein",
    "llama-2-7b",
    "prollama-stage-1",
    "prollama",
)
PROTEIN_ARMS: tuple[str, ...] = F16_PROTEIN_ARMS + EXPANSION_PROTEIN_ARMS
TEXT_ARMS: tuple[str, ...] = F16_TEXT_ARMS
ARMS: tuple[str, ...] = TEXT_ARMS + PROTEIN_ARMS

#: Excluded by name, with the reason, rather than quietly absent.
#: progen2-base / large / xlarge are expansion arms under the imposed 1024
#: budget and must not be differenced as a scale or window contrast.
EXCLUDED_ARMS: dict[str, str] = {
    "zymctrl": (
        "its declared rendering wraps content in an EC conditioning tag, so every "
        "in-context homologue would arrive carrying an enzyme-class label and the "
        "estimand would no longer be about sequence context. Scoring without the "
        "tag is 1.73 nats off distribution (EXP-R2-034) and scoring with a "
        "fabricated one is worse"
    ),
}

#: Packing family names. Dispatch is by arm name first: ProGen3's ArmSpec
#: input_format is ``raw``, which would otherwise inherit GPT-2's EOT prefix.
PACKING_F16 = "f16"
PACKING_BOS_DIRECTION = "bos_direction_seq"
PACKING_GMASK = "gmask_sop_eos"
PACKING_PROGEN3 = "progen3_clm"
PACKING_RITA = "rita_document_stream"
PACKING_GALACTICA = "galactica_amino_blocks"
PACKING_INSTRUCT = "instructprotein_protein_blocks"
PACKING_LLAMA = "llama_seq_blocks"

#: Recorded on every expansion-arm artefact. A certificate or gate gap is a
#: caveat, not a reason to skip the arm.
CAVEATS: dict[str, str] = {
    "progen2-base": (
        "unidentified mixture; 2048 ceiling scored at imposed 1024; no scale/window contrast"
    ),
    "progen2-large": (
        "staged non-member; 1024 budget imposed; no scale/window contrast"
    ),
    "progen2-xlarge": (
        "staged non-member; 1024 budget imposed; no scale/window contrast"
    ),
    "progen3-112m": (
        "N→C only, not bidirectional ProteinGym mean; MoE requires sequence_ids"
    ),
    "progen3-3b": (
        "N→C only, not bidirectional ProteinGym mean; MoE requires sequence_ids"
    ),
    "proteinglm-7b-clm": (
        "UniRef mixture unverified; continuation scores residues 2..L"
    ),
    "protgpt3-1.3b": "candidate arm; native bos_direction_seq",
    "rita-xl": "UniRef100 family; document-stream packing",
    "galactica-125m": (
        "protein-mode context-information unidentified; UniRef50 certificate unsigned"
    ),
    "galactica-1.3b": "UniRef50 certificate unsigned",
    "galactica-6.7b": "UniRef50 certificate unsigned",
    "galactica-30b": "UniRef50 certificate unsigned",
    "instructprotein": (
        "UniRef100 family relation unestablished; do not use instruction slot"
    ),
    "llama-2-7b": "text-initialized protein-mode floor, not protein-pretrained",
    "prollama-stage-1": (
        "Stage 1 protein mode under bare Seq=<...>; Superfamily= is a different experiment"
    ),
    "prollama": (
        "Stage 2 protein mode is off its tuned template; Superfamily= is a different experiment"
    ),
}

#: Every arm is scored inside this many positions, on every condition. It is a
#: budget this campaign imposes, not a checkpoint ceiling: the ceiling is read
#: from each arm's own config through ``arms.config_context_length`` and refused
#: if it is smaller.
POSITION_BUDGET = 1024

# ------------------------------------------------------------------ the cohort

#: The qualifying protein band (Appendix B rule 13): no arm is scored on a band
#: it was not qualified on.
PROTEIN_BAND: tuple[int, int] = (64, 246)

#: Text items are contiguous passages of this many GPT-2 BPE tokens, carved from
#: the frozen OpenWebText screening subset and never a prefix. The band matches
#: the FRACTION of a 1024-position window one item consumes, not the item count:
#: over that subset at the declared 800-character floor a median document is 711
#: BPE tokens, so gpt2-large cannot reach k = 1 for it, while protgpt2 spends a
#: median 50-58 tokens on a Swiss-Prot protein in band 64-246 and reaches k in the
#: tens inside the same window. The asymmetry is a property of the cohorts, not of
#: the models.
TEXT_TOKEN_BAND: tuple[int, int] = (80, 130)

#: Document floor for the passage pool. Unchanged from the panel's declared text
#: eligibility so that the passage cohort is carved from the same population every
#: other text measurement is drawn from.
TEXT_DOCUMENT_MIN_CHARS = 800

#: DIAMOND identity at or above which a pair is a near duplicate. Excluded from
#: every context at construction, and the relation whose connected components are
#: the protein bootstrap unit.
NEAR_DUPLICATE_IDENTITY = 90.0

#: The primary design factor. Identity is ``identity_over_query`` -- percent of
#: the *target* identically matched -- so a short high-identity fragment cannot
#: enter a high band on the strength of ``pident`` alone.
IDENTITY_BANDS: tuple[tuple[str, float, float], ...] = (
    ("id_70_90", 70.0, 90.0),
    ("id_50_70", 50.0, 70.0),
    ("id_30_50", 30.0, 50.0),
    ("id_lt_30", 0.0, 30.0),
)

#: The band clause 3 of the gate is decided in.
DECISIVE_BAND = "id_lt_30"

#: Text relatedness bands. A non-neural BM25 rank over the passage pool. **A text
#: band and a protein identity band are two different measurements** and are never
#: differenced, ranked together or plotted on a shared axis.
TEXT_BANDS: tuple[tuple[str, int, int], ...] = (
    ("bm25_top10", 0, 10),
    ("bm25_10_100", 10, 100),
    ("bm25_100_1000", 100, 1000),
    ("bm25_random", 1000, 0),  # 0 means "to the end of the ranking"
)

#: Rank beyond which a passage counts as unrelated to a target for the
#: matched-unrelated control.
TEXT_UNRELATED_RANK = 1000

#: Pre-declared per-band target floor. A band that cannot reach it is reported as
#: unpopulated and dropped **before scoring**; it is never merged into a
#: neighbouring band after a result exists.
BAND_TARGET_FLOOR = 200

#: A target enters a band only if that band supplies at least this many context
#: items, so k is decided by the token budget rather than by supply on the
#: residue-level arms. Measured on the frozen 20,000-record pool, every retained
#: identity band clears the 200-target floor at this value.
MIN_CONTEXT_ITEMS = 8

#: Recording cap on the frozen per-unit item order. It exists only to bound the
#: artefact and must never bind: :func:`plan_units` raises if an arm's k reaches
#: it, because a cap that binds is the design and not a bound.
MAX_CONTEXT_ITEMS = 48

#: How many of a band's partners are screened for local overlap before the strata
#: are formed. A seeded uniform subsample rather than the whole band, because the
#: 30-50 identity band and the BM25 100-1000 band each carry hundreds of partners
#: per target while no arm's budget admits more than a few dozen, and the screen
#: is an O(n*m) alignment per pair. Four times the recording cap, so a stratum
#: still has ample supply after the screen removes what it removes.
PRE_SCREEN_CANDIDATES = 4 * MAX_CONTEXT_ITEMS

# ---------------------------------------------------------- the overlap screen

#: Local-overlap unit on the protein side: k of the shared-k-mer count.
LOCAL_OVERLAP_KMER = 7

#: A (target, context item) pair whose longest common substring reaches this many
#: residues never enters a primary context.
#:
#: **Twenty rather than ten, and the terciles are why.** Kantroo et al. fire on a
#: 10-residue needle, so a threshold that admitted 10-residue matches would leave
#: copying-explainable pairs in the primary analysis. A threshold *at* ten,
#: however, empties the 70-90 and 50-70 bands outright (measured: the median
#: max-LCS over eight items is 30 residues at 70-90 and 15 at 50-70). The
#: exclusion here removes verbatim copies; what defends against a 10-residue
#: needle is clause 3 of the gate, which is decided inside the **bottom
#: local-overlap tercile of the < 30 band**, whose measured max-LCS is 4-6
#: residues -- below the needle. Both facts are published per band.
HIGH_LOCAL_OVERLAP_LCS = 20

#: Word-shingle length for the text near-duplicate screen.
TEXT_SHINGLE_WORDS = 13

#: Character longest-common-substring at or above which a text pair is a near
#: duplicate. A pair sharing any 13-word shingle is one regardless of length.
TEXT_HIGH_OVERLAP_LCS = 50

#: The two strata every band is split into. ``retained`` is the primary analysis;
#: ``high_local_overlap`` is scored and reported separately so that the copying
#: effect is measured rather than only excluded.
STRATA: tuple[str, ...] = ("retained", "high_local_overlap")

#: Terciles of local overlap within a band, computed over retained units only.
TERCILES: tuple[str, ...] = ("bottom", "middle", "top")

# ------------------------------------------------------------- the conditions

NO_CONTEXT = "no_context"
POSITION_ONLY = "position_only"
MONO_SHUFFLED = "mono_shuffled"
UNRELATED = "unrelated"
HOMOLOGUE = "homologue"

#: Scored in this order, and the order is the registration's: the two conditions
#: most likely to close the line are measured before the condition the campaign
#: hopes for. ``no_context`` sits third because it is a diagnostic and not a
#: condition anything is gated on.
CONDITIONS: tuple[str, ...] = (
    POSITION_ONLY,
    MONO_SHUFFLED,
    NO_CONTEXT,
    UNRELATED,
    HOMOLOGUE,
)

#: What each condition is for, carried into every artefact.
CONDITION_PURPOSE: dict[str, str] = {
    POSITION_ONLY: (
        "a single fixed filler item, identical across every target, repeated to "
        "the same k and the same per-item token lengths. Prices what merely "
        "having a filled context does to a target's likelihood -- the quantity a "
        "naive (k versus k = 0) comparison silently folds into the effect -- and "
        "is the denominator of the fractional reduction"
    ),
    MONO_SHUFFLED: (
        "the homologue items with their symbols permuted within each item, so "
        "the arm's own symbol multiset is preserved exactly and evolutionary "
        "order is destroyed. This is the control that answers Kantroo et al. on "
        "the composition axis"
    ),
    NO_CONTEXT: DIAGNOSTIC_NEVER_THE_EFFECT,
    UNRELATED: (
        "composition-matched unrelated naturals (protein) / unigram-matched "
        "unrelated passages (text), token-length matched item by item. Isolates "
        "bulk composition, and is the referent of both primary endpoints"
    ),
    HOMOLOGUE: (
        "true relatives of the target drawn from one identity or BM25 band, in a "
        "frozen seeded order, with every pair carrying its longest common "
        "substring and its shared-k-mer count"
    ),
}

#: The symbol the mono-shuffle permutes, as the registration froze it: residues
#: on the protein side, tokens on the text side. See the module docstring for the
#: measurement that established a residue shuffle is affordable on ProtGPT2's BPE
#: as well, which is the reason this table is keyed by modality and not by
#: tokenisation.
SHUFFLE_UNITS: dict[str, str] = {"protein": "residues", "text": "tokens"}

#: How many permutation seeds are tried for one whose rendered length matches the
#: item it replaces. A residue-level tokeniser matches on the first, and 77% of
#: ProtGPT2's items match inside 24; the budget is set above that so the trimming
#: fallback stays the exception it is reported as.
SHUFFLE_SEED_BUDGET = 64

# ------------------------------------------------------------------ the freeze

#: This campaign's own draws: target selection, context item order, the shuffles,
#: the passage carving and the bootstrap.
DRAW_SEED = 20260826
BOOTSTRAP_SEED = 20260826
BOOTSTRAP_RESAMPLES = 2000

#: Pool sizes. The protein pool is searched all-against-all with DIAMOND; the
#: text pool is ranked with BM25.
PROTEIN_POOL = 20_000
TEXT_POOL = 12_000

#: DIAMOND search parameters. ``very-sensitive`` because the claim that matters
#: most on the control side is a *negative* one -- that a record has no detectable
#: relative -- and a fast search cannot support a negative.
DIAMOND_SENSITIVITY = "very-sensitive"
DIAMOND_EVALUE = 1e-3
DIAMOND_MAX_TARGET_SEQS = 500

#: Per-item token-length tolerance for a matched control item.
TOKEN_MATCH_TOLERANCE = 0.05
TOKEN_MATCH_FLOOR = 2

#: How many unrelated candidates the text unigram match scores per item. The
#: protein composition match is exact over the whole eligible set (20 dimensions
#: vectorise); a BPE unigram distance does not, so the candidate set is
#: subsampled under the campaign seed and the subsample size is declared.
TEXT_MATCH_CANDIDATES = 200


def require_frozen_parameters(
    *,
    resamples: int,
    bootstrap_seed: int,
    draw_seed: int,
    position_budget: int,
) -> None:
    """Refuse a scored run at parameters the registration did not freeze."""

    frozen = {
        "resamples": (resamples, BOOTSTRAP_RESAMPLES),
        "bootstrap_seed": (bootstrap_seed, BOOTSTRAP_SEED),
        "draw_seed": (draw_seed, DRAW_SEED),
        "position_budget": (position_budget, POSITION_BUDGET),
    }
    moved = [
        f"{name}={observed!r} (frozen {expected!r})"
        for name, (observed, expected) in frozen.items()
        if observed != expected
    ]
    if moved:
        raise ValueError(
            f"{PRE_REGISTRATION} froze these before any score existed and they have "
            "moved: " + "; ".join(moved)
        )


def arm_names(modality: str) -> tuple[str, ...]:
    if modality == "protein":
        return PROTEIN_ARMS
    if modality == "text":
        return TEXT_ARMS
    raise ValueError(f"unknown modality {modality!r}")


def modality_of(arm: str) -> str:
    if arm in PROTEIN_ARMS:
        return "protein"
    if arm in TEXT_ARMS:
        return "text"
    if arm in EXCLUDED_ARMS:
        raise ValueError(f"{arm} is excluded from {PRE_REGISTRATION}: {EXCLUDED_ARMS[arm]}")
    raise ValueError(f"{arm!r} is not an arm of {PRE_REGISTRATION}; arms are {list(ARMS)}")


def assign_identity_band(identity: float) -> str:
    """Which declared band one pair's ``identity_over_query`` belongs to.

    Raises rather than returning a default. An identity at or above
    :data:`NEAR_DUPLICATE_IDENTITY` belongs to the excluded near-duplicate
    relation and is not a scored band; anything outside ``[0, 100]`` is not an
    identity at all. Both used to be silently dropped by a ``for``/``break``
    search, which is the shape that lets a pair the design never considered leave
    no trace in the census.
    """

    if not 0.0 <= float(identity) <= 100.0:
        raise ValueError(
            f"{identity!r} is not a percent identity; it belongs to no band of "
            f"{PRE_REGISTRATION} and is refused rather than dropped"
        )
    if identity >= NEAR_DUPLICATE_IDENTITY:
        raise ValueError(
            f"identity {identity} is a near duplicate at or above "
            f"{NEAR_DUPLICATE_IDENTITY} and is excluded at construction; it is not "
            "a scored band"
        )
    for band, low, high in IDENTITY_BANDS:
        if low <= identity < high:
            return band
    raise ValueError(
        f"identity {identity} falls in no declared band; bands are "
        f"{[name for name, _, _ in IDENTITY_BANDS]}"
    )


def require_position_budget(config: Any, *, arm: str) -> int:
    """Refuse an arm whose declared window is smaller than the fixed budget.

    Read through :func:`~src.capability.core.arms.config_context_length` rather than off
    ``n_positions``, because that resolver declares the fallback order -- four
    spellings among the checkpoints this repository reaches -- and raises on one
    it does not know instead of resolving to whichever attribute happened to
    exist. Takes a config rather than a loaded arm so that the planning stage,
    which loads tokenizers and no weights, checks exactly what the scoring stage
    checks.
    """

    declared = config_context_length(config)
    if declared < POSITION_BUDGET:
        raise ValueError(
            f"{arm} declares {declared} positions, below this campaign's fixed "
            f"{POSITION_BUDGET}-position budget; it cannot be scored here"
        )
    return int(declared)


def packing_of(name: str) -> str:
    """Which id-concatenation family one arm uses.

    Name first, not ``ArmSpec.input_format``: ProGen3 declares ``raw``, which is
    GPT-2's EOT-joined stream and is the wrong packing.
    """

    if name.startswith("progen3-"):
        return PACKING_PROGEN3
    if name == "protgpt3-1.3b":
        return PACKING_BOS_DIRECTION
    if name == "proteinglm-7b-clm":
        return PACKING_GMASK
    if name == "rita-xl":
        return PACKING_RITA
    if name.startswith("galactica-"):
        return PACKING_GALACTICA
    if name == "instructprotein":
        return PACKING_INSTRUCT
    if name in {"llama-2-7b", "prollama-stage-1", "prollama"}:
        return PACKING_LLAMA
    if name in ARMS:
        return PACKING_F16
    raise ValueError(f"{name!r} is not an arm of {PRE_REGISTRATION}")


def _joint_family(name: str) -> str:
    packing = packing_of(name)
    if packing == PACKING_GALACTICA:
        return "galactica"
    if packing == PACKING_INSTRUCT:
        return "instructprotein"
    if packing == PACKING_LLAMA:
        return "prollama"
    raise ValueError(f"{name}: not a joint-rendering packing")


def _joint_tokenisation(arm: Arm) -> Any:
    provenance = arm.serving_provenance or {}
    cached = provenance.get("joint_tokenisation")
    if cached is not None:
        return cached
    from ..models.joint_modes import rendering, resolve

    return resolve(arm.tokenizer, rendering(_joint_family(arm.name)))


def _leading_prefix_len(arm: Arm, ids: Sequence[int], tokenisation: Any) -> int:
    markers = tuple(int(value) for value in tokenisation.prefix_marker_ids)
    if markers:
        if tuple(ids[: len(markers)]) != markers:
            raise ValueError(
                f"{arm.name}: joint block does not open with the declared prefix "
                f"{list(markers)}; got {list(ids[: len(markers)])}"
            )
        return len(markers)
    bos = _bos_id(arm.tokenizer)
    if bos is not None and ids and int(ids[0]) == bos:
        return 1
    raise ValueError(
        f"{arm.name}: subsequent packed blocks must drop a leading BOS, but none was found"
    )


def _bos_id(tokenizer: Any) -> int | None:
    for attribute in ("bos_token_id", "bos_id"):
        value = getattr(tokenizer, attribute, None)
        if value is not None:
            return int(value)
    return None


def _joint_item_ids(arm: Arm, record: str, *, drop_prefix: bool) -> list[int]:
    tokenisation = _joint_tokenisation(arm)
    rendered = tokenisation.render(record, context=None)
    ids = list(rendered.token_ids)
    if not drop_prefix:
        return ids
    return ids[_leading_prefix_len(arm, ids, tokenisation) :]


def _progen3_special_ids(tokenizer: Any) -> dict[str, int]:
    vocab = tokenizer.get_vocab()
    names = ("<bos>", "1", "2", "<eos>", "<pad>")
    missing = [name for name in names if name not in vocab]
    if missing:
        raise ValueError(
            f"ProGen3 tokenizer is missing {missing}; ids are resolved from the "
            "checkpoint vocab and are not hardcoded"
        )
    return {name: int(vocab[name]) for name in names}


def _progen3_item_ids(arm: Arm, record: str) -> list[int]:
    specials = _progen3_special_ids(arm.tokenizer)
    residues = [_progen3_residue_id(arm.tokenizer, residue) for residue in record]
    return [specials["<bos>"], specials["1"], *residues, specials["2"], specials["<eos>"]]


def _progen3_residue_id(tokenizer: Any, residue: str) -> int:
    vocab = tokenizer.get_vocab()
    if residue not in vocab:
        resolved = tokenizer.convert_tokens_to_ids(residue)
        if resolved is None or resolved == tokenizer.unk_token_id:
            raise ValueError(f"ProGen3 vocab has no residue {residue!r}")
        return int(resolved)
    return int(vocab[residue])


def _rita_eos(arm: Arm) -> int:
    """Native document-boundary id. tokenizer.eos_token_id may be unset."""

    from ..models.rita_fitness import NATIVE_EOS_ID

    token_id = arm.tokenizer.convert_tokens_to_ids(EOS_BOUNDED_BOUNDARY)
    if token_id is None or int(token_id) < 0:
        raise ValueError(f"{arm.name}: RITA document stream needs a native EOS id")
    resolved = int(token_id)
    if resolved != NATIVE_EOS_ID:
        raise ValueError(
            f"{arm.name}: native <EOS> must be id {NATIVE_EOS_ID}, got {resolved}"
        )
    return NATIVE_EOS_ID


def _residue_token_ids(arm: Arm, record: str) -> list[int]:
    ids = arm.tokenizer(record, add_special_tokens=False, return_tensors=None)["input_ids"]
    if not ids:
        raise ValueError(f"{arm.name}: residue string tokenised to nothing")
    return list(ids)


def row_prefix_ids(arm: Arm) -> list[int]:
    """Ids written once at the start of a packed row, never repeated per item."""

    packing = packing_of(arm.name)
    if packing == PACKING_RITA:
        return [_rita_eos(arm)]
    if packing == PACKING_INSTRUCT:
        return list(_joint_tokenisation(arm).prefix_marker_ids)
    if packing == PACKING_LLAMA:
        bos = _bos_id(arm.tokenizer)
        if bos is None:
            raise ValueError(f"{arm.name}: Llama-lineage packing needs a BOS id")
        return [bos]
    return []


def _item_affixes(arm: Arm, ids: Sequence[int], *, record: str | None = None) -> tuple[int, int]:
    """How many leading and trailing non-content tokens one packed item carries."""

    packing = packing_of(arm.name)
    if packing == PACKING_F16:
        return content_offset(arm), 0
    if packing == PACKING_BOS_DIRECTION:
        return 2, 0
    if packing == PACKING_GMASK:
        return 3, 0
    if packing == PACKING_PROGEN3:
        return 2, 2
    if packing == PACKING_RITA:
        return 0, 1
    if packing in {PACKING_GALACTICA, PACKING_INSTRUCT, PACKING_LLAMA}:
        start, end = target_span(arm, ids, record=record)
        return start, len(ids) - end
    raise ValueError(f"{arm.name}: no affix rule for packing {packing!r}")


# --------------------------------------------------------- the overlap screen


def _codes(record: str) -> np.ndarray:
    return np.frombuffer(record.encode("utf-8", "surrogatepass"), dtype=np.uint8)


def longest_common_substrings(target: str, candidates: Sequence[str]) -> np.ndarray:
    """Longest common substring of ``target`` against each candidate.

    The classic O(n*m) dynamic program, vectorised across the candidate axis so
    that one target's whole partner list costs ``len(target)`` array operations
    rather than one Python loop per pair. Measured on the frozen pool: 51,033
    pairs in 13.3 s, 640,000 in about 160 s, which is what makes a screen over
    every banded pair affordable at cohort-construction time.
    """

    if not candidates:
        return np.zeros(0, dtype=np.int32)
    encoded = [_codes(candidate) for candidate in candidates]
    width = max(len(item) for item in encoded)
    if width == 0 or len(target) == 0:
        return np.zeros(len(candidates), dtype=np.int32)
    block = np.zeros((len(encoded), width), dtype=np.uint8)
    valid = np.zeros((len(encoded), width), dtype=bool)
    for row, item in enumerate(encoded):
        block[row, : len(item)] = item
        valid[row, : len(item)] = True
    previous = np.zeros((len(encoded), width), dtype=np.int32)
    best = np.zeros(len(encoded), dtype=np.int32)
    for symbol in _codes(target):
        equal = (block == symbol) & valid
        current = np.zeros_like(previous)
        current[:, 0] = equal[:, 0]
        current[:, 1:] = np.where(equal[:, 1:], previous[:, :-1] + 1, 0)
        best = np.maximum(best, current.max(axis=1))
        previous = current
    return best


def _kmer_codes(record: str, k: int) -> np.ndarray:
    if len(record) < k:
        return np.zeros(0, dtype=np.int64)
    raw = _codes(record).astype(np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(raw, k)
    weights = (256 ** np.arange(k, dtype=np.int64)).reshape(1, k)
    return np.unique((windows * weights).sum(axis=1))


def shared_kmer_counts(
    target: str, candidates: Sequence[str], *, k: int = LOCAL_OVERLAP_KMER
) -> np.ndarray:
    """Number of distinct ``k``-mers each candidate shares with ``target``."""

    if not candidates:
        return np.zeros(0, dtype=np.int32)
    reference = _kmer_codes(target, k)
    if reference.size == 0:
        return np.zeros(len(candidates), dtype=np.int32)
    counts = np.zeros(len(candidates), dtype=np.int32)
    for index, candidate in enumerate(candidates):
        other = _kmer_codes(candidate, k)
        counts[index] = np.intersect1d(reference, other, assume_unique=True).size
    return counts


_WORD = re.compile(r"\w+")


def word_shingles(record: str, *, length: int = TEXT_SHINGLE_WORDS) -> frozenset[str]:
    """Word shingles of one passage, for the text near-duplicate screen."""

    words = _WORD.findall(record.lower())
    if len(words) < length:
        return frozenset()
    return frozenset(
        " ".join(words[index : index + length]) for index in range(len(words) - length + 1)
    )


def pair_overlap(
    target: str,
    candidates: Sequence[str],
    *,
    modality: str,
    target_shingles: frozenset[str] | None = None,
    candidate_shingles: Sequence[frozenset[str]] | None = None,
) -> dict[str, np.ndarray]:
    """Both local-overlap coordinates of one target against a candidate list.

    ``lcs`` is the copying-relevant coordinate and is what the strata and the
    terciles are built on; the shingle count travels beside it because a pair can
    share a great deal of short-range material without any single long run. The
    text path takes its shingles precomputed because the same passage is a
    candidate for many targets and reshingling it each time is the expensive half.
    """

    lcs = longest_common_substrings(target, candidates)
    if modality == "protein":
        shared = shared_kmer_counts(target, candidates)
    elif modality == "text":
        reference = word_shingles(target) if target_shingles is None else target_shingles
        others = (
            [word_shingles(candidate) for candidate in candidates]
            if candidate_shingles is None
            else list(candidate_shingles)
        )
        if len(others) != len(candidates):
            raise ValueError("precomputed shingles do not align with the candidates")
        shared = np.array([len(reference & entry) for entry in others], dtype=np.int32)
    else:
        raise ValueError(f"unknown modality {modality!r}")
    return {"lcs": lcs, "shared": shared}


def high_local_overlap(overlap: Mapping[str, np.ndarray], *, modality: str) -> np.ndarray:
    """Which pairs are excluded from a primary context."""

    if modality == "protein":
        return overlap["lcs"] >= HIGH_LOCAL_OVERLAP_LCS
    if modality == "text":
        return (overlap["lcs"] >= TEXT_HIGH_OVERLAP_LCS) | (overlap["shared"] > 0)
    raise ValueError(f"unknown modality {modality!r}")


# ----------------------------------------------------------------------- BM25

BM25_K1 = 1.2
BM25_B = 0.75


class Bm25Index:
    """A plain BM25 ranking over the passage pool.

    Non-neural by design: the text relatedness band has to be a *retrieval* band
    rather than a model's own similarity, or the text arm would be scored against
    a partition one of its own representations produced.
    """

    def __init__(self, documents: Sequence[str], *, k1: float = BM25_K1, b: float = BM25_B):
        from scipy import sparse

        tokenised = [_WORD.findall(document.lower()) for document in documents]
        vocabulary: dict[str, int] = {}
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        lengths = np.array([len(item) for item in tokenised], dtype=np.float64)
        if not lengths.size or lengths.max() == 0:
            raise ValueError("BM25 needs non-empty documents")
        average = float(lengths[lengths > 0].mean())
        for index, item in enumerate(tokenised):
            for term, count in Counter(item).items():
                column = vocabulary.setdefault(term, len(vocabulary))
                rows.append(index)
                columns.append(column)
                values.append(float(count))
        counts = sparse.csr_matrix(
            (values, (rows, columns)), shape=(len(documents), len(vocabulary))
        )
        document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log(
            1.0
            + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        normaliser = k1 * (1.0 - b + b * lengths / average)
        weighted = counts.tocoo()
        numerator = weighted.data * (k1 + 1.0)
        denominator = weighted.data + normaliser[weighted.row]
        scored = numerator / denominator * idf[weighted.col]
        self.matrix = sparse.csr_matrix(
            (scored, (weighted.row, weighted.col)), shape=counts.shape
        )
        self.presence = sparse.csr_matrix(
            (np.ones_like(weighted.data), (weighted.row, weighted.col)), shape=counts.shape
        )
        self.vocabulary = vocabulary
        self.n_documents = len(documents)

    def ranking(self, index: int) -> np.ndarray:
        """Every other document, best first, as document indices."""

        query = self.presence[index]
        scores = np.asarray((self.matrix @ query.T).todense()).ravel()
        scores[index] = -np.inf
        order = np.argsort(-scores, kind="stable")
        return order

    def max_score_against(self, index: int, others: Sequence[int]) -> float:
        query = self.presence[index]
        scores = np.asarray((self.matrix @ query.T).todense()).ravel()
        return float(max(scores[other] for other in others if other != index))


# ------------------------------------------------------------- the renderings


def render_records(arm: Arm, records: Sequence[str], *, modality: str) -> list[str]:
    """One rendering declaration, the panel's own.

    Delegates to :meth:`~src.capability.core.arms.Cohort.input_strings` rather than
    spelling a second copy of the FASTA wrap, the ``1`` direction marker or the
    raw passthrough, so an arm whose rendering changes changes here too.
    """

    cohort = Cohort(
        name="context_item",
        kind=modality,
        records=list(records),
        min_symbols=0,
        max_symbols=0,
    )
    return cohort.input_strings(arm)


def item_prefix(arm: Arm) -> str:
    """The non-content prefix one rendered item carries, as a string.

    Three of the four arms are self-delimiting: ``fasta_wrapped`` begins each item
    with the end-of-text token and a newline, which is exactly ProtGPT2's
    pretraining stream, and ``n_to_c_control`` begins each item with ProGen2's
    ``1`` direction marker. ``raw`` carries nothing, so a text item is prefixed
    with the end-of-text token -- the separator GPT-2's own pretraining documents
    were joined by. Declared once here for every consumer: the concatenation, the
    scored-span offset and the self-check all read it.
    """

    if packing_of(arm.name) != PACKING_F16:
        return ""
    fmt = arm.spec.input_format
    if fmt in ("fasta_wrapped", "n_to_c_control"):
        return ""
    if fmt == "raw":
        eot = arm.tokenizer.eos_token
        if eot is None:
            raise ValueError(f"{arm.name}: tokenizer has no end-of-text token")
        return eot
    raise ValueError(
        f"{arm.name}: {PRE_REGISTRATION} declares no item delimiter for input "
        f"format {fmt!r}"
    )


def content_offset(arm: Arm) -> int:
    """How many leading tokens of a rendered item are marker rather than content.

    Resolved by tokenising the marker alone and requiring it to be a clean prefix
    of the rendered item, which is checked per record by :func:`item_ids`: a BPE
    merge that straddles the marker boundary would silently move the scored span,
    and this campaign's whole contrast lives on that span.
    """

    packing = packing_of(arm.name)
    if packing == PACKING_BOS_DIRECTION:
        return 2
    if packing == PACKING_GMASK:
        return 3
    if packing == PACKING_PROGEN3:
        return 2
    if packing in {
        PACKING_RITA,
        PACKING_GALACTICA,
        PACKING_INSTRUCT,
        PACKING_LLAMA,
    }:
        return 0
    fmt = arm.spec.input_format
    if fmt == "raw":
        return len(arm.tokenizer(item_prefix(arm), return_tensors=None)["input_ids"])
    if fmt == "n_to_c_control":
        return 1
    if fmt == "fasta_wrapped":
        return len(
            arm.tokenizer(
                arm.tokenizer.eos_token + "\n", return_tensors=None
            )["input_ids"]
        )
    raise ValueError(f"{arm.name}: no declared content offset for {fmt!r}")


def item_ids(arm: Arm, record: str, *, modality: str) -> list[int]:
    """Token ids of one rendered context or target item.

    Items are tokenised **separately and concatenated as ids**, never by joining
    the rendered strings and tokenising once. That is what makes the target's
    token grid identical under every condition, which is the precondition for a
    paired NLL contrast: a BPE merge reaching across a context boundary would
    otherwise change which tokens are scored when the context content changes.
    """

    packing = packing_of(arm.name)
    if packing == PACKING_F16:
        rendered = item_prefix(arm) + render_records(arm, [record], modality=modality)[0]
        ids = arm.tokenizer(rendered, return_tensors=None)["input_ids"]
        if not ids:
            raise ValueError(f"{arm.name}: a rendered item tokenised to nothing")
        return list(ids)
    if packing == PACKING_BOS_DIRECTION:
        from ..core.arms import BOS_DIRECTION_N_TO_C

        bos = arm.tokenizer.bos_token_id
        direction = arm.tokenizer.convert_tokens_to_ids(BOS_DIRECTION_N_TO_C)
        if bos is None or direction is None or direction == arm.tokenizer.unk_token_id:
            raise ValueError(
                f"{arm.name}: native bos_direction_seq needs BOS and "
                f"{BOS_DIRECTION_N_TO_C!r} ids from the tokenizer"
            )
        return [int(bos), int(direction)] + _residue_token_ids(arm, record)
    if packing == PACKING_GMASK:
        from ..models.proteinglm import PREFIX_IDS

        return list(PREFIX_IDS) + _residue_token_ids(arm, record)
    if packing == PACKING_PROGEN3:
        return _progen3_item_ids(arm, record)
    if packing == PACKING_RITA:
        return _residue_token_ids(arm, record) + [_rita_eos(arm)]
    if packing == PACKING_GALACTICA:
        return _joint_item_ids(arm, record, drop_prefix=False)
    if packing in {PACKING_INSTRUCT, PACKING_LLAMA}:
        return _joint_item_ids(arm, record, drop_prefix=True)
    raise ValueError(f"{arm.name}: no item encoding for packing {packing!r}")


def target_span(
    arm: Arm, ids: Sequence[int], record: str | None = None
) -> tuple[int, int]:
    """The half-open scored span of a rendered target, marker tokens removed."""

    packing = packing_of(arm.name)
    if packing == PACKING_F16:
        offset = content_offset(arm)
        if len(ids) <= offset:
            raise ValueError(f"{arm.name}: a rendered target carries no content tokens")
        return offset, len(ids)
    if packing == PACKING_BOS_DIRECTION:
        offset = 2
        if len(ids) <= offset:
            raise ValueError(f"{arm.name}: a rendered target carries no content tokens")
        return offset, len(ids)
    if packing == PACKING_GMASK:
        if len(ids) < 5:
            raise ValueError(
                f"{arm.name}: continuation scores residues 2..L and needs at least "
                "two residues after the three-token prefix"
            )
        return 4, len(ids)
    if packing == PACKING_PROGEN3:
        if len(ids) < 5:
            raise ValueError(
                f"{arm.name}: prepare_clm scores residue tokens only and needs at "
                "least one residue between the direction markers"
            )
        return 2, len(ids) - 2
    if packing == PACKING_RITA:
        if len(ids) < 2:
            raise ValueError(f"{arm.name}: a RITA item needs residues and a trailing EOS")
        return 0, len(ids) - 1
    if packing in {PACKING_GALACTICA, PACKING_INSTRUCT, PACKING_LLAMA}:
        return _joint_target_span(arm, ids, record=record)
    raise ValueError(f"{arm.name}: no target span for packing {packing!r}")


def _joint_target_span(
    arm: Arm, ids: Sequence[int], *, record: str | None
) -> tuple[int, int]:
    tokenisation = _joint_tokenisation(arm)
    packing = packing_of(arm.name)
    if record is None:
        raise ValueError(
            f"{arm.name}: joint packing locates the scored span from the declared "
            "rendering and needs the residue string"
        )
    rendered = tokenisation.render(record, context=None)
    full = list(rendered.token_ids)
    drop = 0 if packing == PACKING_GALACTICA else _leading_prefix_len(arm, full, tokenisation)
    positions = [int(position) - drop for position in rendered.scored_positions]
    if not positions:
        raise ValueError(f"{arm.name}: the declared rendering selected no scored tokens")
    if positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError(
            f"{arm.name}: scored positions {positions} are not a contiguous span"
        )
    if list(ids) != full[drop:]:
        raise ValueError(
            f"{arm.name}: target ids drifted from the declared rendering of this record"
        )
    return positions[0], positions[-1] + 1


# ---------------------------------------------------- context reconstruction

#: The three recipe verbs a plan may carry. A plan stores recipes rather than
#: token ids so that the frozen cohort remains the one source of content and a
#: scoring run rebuilds every condition deterministically from it -- and then
#: checks the rebuilt token counts against the ones the plan recorded, so a drift
#: between planning and scoring is refused instead of averaged.
RECIPE_VERBS: tuple[str, ...] = ("pool", "shuffle", "filler")


def _one_shuffle(arm: Arm, record: str, *, modality: str, unit: str, seed: int) -> list[int]:
    generator = np.random.default_rng(seed)
    if unit == "residues":
        symbols = np.array(list(record))
        permuted = "".join(symbols[generator.permutation(symbols.size)])
        return item_ids(arm, permuted, modality=modality)
    ids = item_ids(arm, record, modality=modality)
    offset = content_offset(arm)
    content = np.array(ids[offset:], dtype=np.int64)
    permuted_ids = content[generator.permutation(content.size)]
    return list(ids[:offset]) + [int(value) for value in permuted_ids]


def shuffled_item_ids(
    arm: Arm,
    record: str,
    *,
    modality: str,
    seed: int,
    target_tokens: int | None = None,
) -> list[int]:
    """One context item with its symbols permuted within itself.

    The permuted symbol is the registration's: residues on the protein side,
    tokens on the text side. The item's non-content marker prefix is never
    permuted.

    ``target_tokens`` asks for a permutation that renders to exactly that many
    tokens, which is the token-length clause the same registration imposes on
    every control. See the module docstring for the measurement behind the seed
    search and for what the trimming fallback costs when the search fails.
    """

    unit = SHUFFLE_UNITS.get(modality)
    if unit is None:
        raise ValueError(
            f"{PRE_REGISTRATION} declares no mono-shuffle unit for modality {modality!r}"
        )
    return shuffled_item(
        arm, record, modality=modality, seed=seed, target_tokens=target_tokens
    )[0]


#: What the length-matched shuffle had to do to reach the item's token count.
SHUFFLE_OUTCOMES: tuple[str, ...] = ("exact", "trimmed", "short")


def shuffled_item(
    arm: Arm,
    record: str,
    *,
    modality: str,
    seed: int,
    target_tokens: int | None = None,
) -> tuple[list[int], str, int]:
    """:func:`shuffled_item_ids` and what it had to do, for the plan's census.

    Returns the ids, the outcome, and the token count the chosen permutation
    produced *before* any truncation, so the plan can report what a trim cost
    rather than only how often one happened.
    """

    unit = SHUFFLE_UNITS.get(modality)
    if unit is None:
        raise ValueError(
            f"{PRE_REGISTRATION} declares no mono-shuffle unit for modality {modality!r}"
        )
    if target_tokens is None:
        ids = _one_shuffle(arm, record, modality=modality, unit=unit, seed=seed)
        return ids, "exact", len(ids)
    if target_tokens <= content_offset(arm):
        raise ValueError("a shuffled item must carry at least one content token")
    best: list[int] | None = None
    for attempt in range(SHUFFLE_SEED_BUDGET):
        candidate = _one_shuffle(
            arm, record, modality=modality, unit=unit, seed=_unit_seed(seed, "attempt", attempt)
        )
        if len(candidate) == target_tokens:
            return candidate, "exact", len(candidate)
        if len(candidate) > target_tokens and (best is None or len(candidate) < len(best)):
            best = candidate
    if best is None:
        # Every permutation of this item rendered shorter than the item itself.
        # Nothing can be added without inventing residues, so the first candidate
        # is taken and the plan records the gap rather than padding it away.
        ids = _one_shuffle(
            arm, record, modality=modality, unit=unit, seed=_unit_seed(seed, "attempt", 0)
        )
        return ids, "short", len(ids)
    return best[:target_tokens], "trimmed", len(best)


def filler_item_ids(arm: Arm, filler: str, *, modality: str, n_tokens: int) -> list[int]:
    """The fixed filler, cut to exactly ``n_tokens`` tokens.

    Cut at the token level rather than at the symbol level so the match is exact
    on every tokeniser. The filler is drawn at the top of the item band precisely
    so that tiling is rare; when it is still too short the content is repeated,
    and :func:`plan_units` counts every time that happens.
    """

    if n_tokens < 1:
        raise ValueError("a context item needs at least one token")
    ids = item_ids(arm, filler, modality=modality)
    prefix_len, suffix_len = _item_affixes(arm, ids, record=filler)
    interior = ids[prefix_len : len(ids) - suffix_len if suffix_len else None]
    if not interior:
        raise ValueError(f"{arm.name}: the filler renders to no content tokens")
    wanted = n_tokens - prefix_len - suffix_len
    if wanted < 1:
        raise ValueError(
            f"{arm.name}: a {n_tokens}-token filler item cannot carry this arm's "
            f"{prefix_len}+{suffix_len} affix tokens"
        )
    repeats = math.ceil(wanted / len(interior))
    suffix = list(ids[len(ids) - suffix_len :]) if suffix_len else []
    return list(ids[:prefix_len]) + list((interior * repeats)[:wanted]) + suffix


def build_item(
    arm: Arm,
    recipe: Sequence[Any],
    *,
    records: Sequence[str],
    filler: str,
    modality: str,
) -> list[int]:
    """Rebuild one context item from its recipe."""

    verb = recipe[0]
    if verb == "pool":
        return item_ids(arm, records[int(recipe[1])], modality=modality)
    if verb == "shuffle":
        return shuffled_item_ids(
            arm,
            records[int(recipe[1])],
            modality=modality,
            seed=int(recipe[2]),
            target_tokens=int(recipe[3]),
        )
    if verb == "filler":
        return filler_item_ids(arm, filler, modality=modality, n_tokens=int(recipe[1]))
    raise ValueError(f"unknown recipe verb {verb!r}; verbs are {list(RECIPE_VERBS)}")


def build_row(
    arm: Arm,
    unit: Mapping[str, Any],
    condition: str,
    *,
    records: Sequence[str],
    filler: str,
    modality: str,
) -> tuple[list[int], int]:
    """The full token row for one (unit, condition), and the scored span's start.

    The target's rendered ids are appended last and are byte-identical under every
    condition, which is what makes the contrast paired at the token level.
    """

    recipes = unit["conditions"][condition]
    context: list[int] = []
    for recipe in recipes:
        context.extend(build_item(arm, recipe, records=records, filler=filler, modality=modality))
    target_record = records[int(unit["target"])]
    target = item_ids(arm, target_record, modality=modality)
    offset, _end = target_span(arm, target, record=target_record)
    prefix = row_prefix_ids(arm)
    row = prefix + context + target
    if len(row) > POSITION_BUDGET:
        raise ValueError(
            f"{arm.name}: unit {unit['key']} condition {condition!r} builds "
            f"{len(row)} tokens, past the fixed {POSITION_BUDGET}-position budget"
        )
    return row, len(prefix) + len(context) + offset


# --------------------------------------------------------- frozen artefacts


def _digest(material: Any) -> str:
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def cohort_digest(payload: Mapping[str, Any]) -> str:
    """Content digest of a frozen cohort: what was drawn, and nothing else.

    Excludes timestamps and tool provenance so that rebuilding the cohort from the
    same corpus under the same seeds reproduces the digest a scoring run was
    pinned to, and includes every record and every unit so that a cohort whose
    content moved cannot wear the digest of the one that was scored.
    """

    return _digest(
        {
            "pre_registration": payload["pre_registration"],
            "draw": payload["draw"],
            "protein": {
                "records": payload["protein"]["records"],
                "units": payload["protein"]["units"],
                "filler": payload["protein"]["filler"],
                "groups": payload["protein"]["groups"],
            },
            "text": {
                "records": payload["text"]["records"],
                "units": payload["text"]["units"],
                "filler": payload["text"]["filler"],
                "groups": payload["text"]["groups"],
            },
        }
    )


def load_cohort(path: Path) -> dict[str, Any]:
    """Read a frozen cohort and refuse one that has drifted from its own digest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded = payload.get("digest")
    if not recorded:
        raise ValueError(f"{path} carries no digest; it is not a frozen cohort")
    observed = cohort_digest(payload)
    if observed != recorded:
        raise ValueError(
            f"{path} hashes to {observed} and declares {recorded}: the frozen cohort "
            "has drifted from the digest a scoring run was pinned to. It is a "
            "different cohort under the same name and is refused"
        )
    if payload.get("pre_registration") != PRE_REGISTRATION:
        raise ValueError(
            f"{path} was frozen for {payload.get('pre_registration')!r}, "
            f"not {PRE_REGISTRATION!r}"
        )
    return payload


def plan_digest(payload: Mapping[str, Any]) -> str:
    """Content digest of one arm's plan, including the cohort it was built on."""

    return _digest(
        {
            "pre_registration": payload["pre_registration"],
            "arm": payload["arm"],
            "cohort_digest": payload["cohort_digest"],
            "units": payload["units"],
        }
    )


def resolve_plan_path(path: Path, *, arm: str | None = None) -> Path:
    """Find a plan at the given path or in the queue's per-cell directory."""

    path = Path(path)
    candidates = [path]
    if arm:
        candidates.append(path.parent / f"s46x_plan_{arm}" / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"plan not found at {path}"
        + (f" or {candidates[-1]}" if len(candidates) > 1 else "")
    )


def load_plan(path: Path, *, cohort: Mapping[str, Any], arm: str | None = None) -> dict[str, Any]:
    """Read a frozen plan, refusing drift in the plan or in the cohort under it."""

    payload = json.loads(resolve_plan_path(path, arm=arm).read_text(encoding="utf-8"))
    recorded = payload.get("digest")
    if not recorded:
        raise ValueError(f"{path} carries no digest; it is not a frozen plan")
    observed = plan_digest(payload)
    if observed != recorded:
        raise ValueError(
            f"{path} hashes to {observed} and declares {recorded}: the frozen plan "
            "has drifted from the digest a scoring run was pinned to and is refused"
        )
    if payload.get("pre_registration") != PRE_REGISTRATION:
        raise ValueError(
            f"{path} was frozen for {payload.get('pre_registration')!r}, "
            f"not {PRE_REGISTRATION!r}"
        )
    if payload["cohort_digest"] != cohort["digest"]:
        raise ValueError(
            f"{path} was planned against cohort {payload['cohort_digest']} and the "
            f"cohort supplied is {cohort['digest']}: the plan's item indices name "
            "records this cohort does not hold, so it is refused"
        )
    if arm is not None and payload["arm"] != arm:
        raise ValueError(f"{path} is the plan for {payload['arm']!r}, not {arm!r}")
    return payload


# ------------------------------------------------------------ cohort assembly


def _unit_seed(*parts: Any) -> int:
    """A stable derived seed, so every draw inside the cohort is reproducible."""

    material = "|".join(str(part) for part in (DRAW_SEED, *parts))
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16)


def connected_groups(n_records: int, pairs: Iterable[tuple[int, int]]) -> list[int]:
    """Connected components of a symmetric relation, in record order.

    The protein bootstrap unit: a target's near-duplicate group under the same
    DIAMOND alignment that defines the bands. Group ids are allocated in record
    order so the grouping is a function of the records and the relation alone.
    """

    parent = list(range(n_records))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in pairs:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)
    labels: dict[int, int] = {}
    groups: list[int] = []
    for index in range(n_records):
        root = find(index)
        if root not in labels:
            labels[root] = len(labels)
        groups.append(labels[root])
    return groups


def _draw_units(
    *,
    modality: str,
    groups: Sequence[int],
    candidates: Mapping[str, dict[int, dict[str, Any]]],
    band_floor: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the frozen targets of every (band, stratum), and census the rest.

    ``candidates`` maps band -> target -> ``{"retained": ..., "high_local_overlap":
    ...}``, each holding the partner order and its two overlap coordinates. A
    stratum that cannot reach ``band_floor`` targets is reported unpopulated and
    dropped **here**, before any model is loaded, and is never merged into a
    neighbour afterwards.
    """

    units: list[dict[str, Any]] = []
    census: dict[str, Any] = {}
    for band, per_target in candidates.items():
        census[band] = {}
        for stratum in STRATA:
            eligible = sorted(
                target
                for target, blocks in per_target.items()
                if len(blocks[stratum]["partners"]) >= MIN_CONTEXT_ITEMS
            )
            populated = len(eligible) >= band_floor
            census[band][stratum] = {
                "eligible_targets": len(eligible),
                "floor": band_floor,
                "populated": populated,
                "drawn": band_floor if populated else 0,
            }
            if not populated:
                census[band][stratum]["reason"] = (
                    f"{len(eligible)} targets carry at least {MIN_CONTEXT_ITEMS} "
                    f"{stratum} context items, below the pre-declared floor of "
                    f"{band_floor}; this stratum is dropped before any score exists "
                    "and is not merged into a neighbouring band"
                )
                continue
            order = np.random.default_rng(_unit_seed(modality, band, stratum, "targets"))
            chosen = sorted(
                int(eligible[position])
                for position in order.permutation(len(eligible))[:band_floor]
            )
            for target in chosen:
                block = per_target[target][stratum]
                units.append(
                    {
                        "key": f"{modality}|{band}|{stratum}|{target:06d}",
                        "modality": modality,
                        "band": band,
                        "stratum": stratum,
                        "target": int(target),
                        "group": int(groups[target]),
                        "partners": [int(value) for value in block["partners"]],
                        "partner_lcs": [int(value) for value in block["lcs"]],
                        "partner_shared": [int(value) for value in block["shared"]],
                    }
                )
    return units, census


def _pre_screen_sample(
    partners: Sequence[int], modality: str, band: str, target: int
) -> list[int]:
    """A seeded uniform subsample of one band's partners, before the screen."""

    if len(partners) <= PRE_SCREEN_CANDIDATES:
        return list(partners)
    generator = np.random.default_rng(_unit_seed(modality, band, target, "prescreen"))
    order = generator.permutation(len(partners))[:PRE_SCREEN_CANDIDATES]
    return sorted(int(partners[position]) for position in order)


def _order_partners(
    modality: str, band: str, stratum: str, target: int, partners: Sequence[int]
) -> np.ndarray:
    """The frozen context-item order for one unit: a seeded permutation.

    Seeded rather than ranked, because ordering a band's partners by identity and
    taking a prefix would make the arm with the smaller budget see the *closest*
    relatives and the arm with the larger budget see a mixture -- a confound
    between k and relatedness that the band structure exists to prevent.
    """

    generator = np.random.default_rng(_unit_seed(modality, band, stratum, target, "order"))
    order = generator.permutation(len(partners))[:MAX_CONTEXT_ITEMS]
    return np.asarray(partners, dtype=np.int64)[order]


def protein_cohort_units(
    records: Sequence[str],
    hits: Mapping[tuple[int, int], float],
) -> dict[str, Any]:
    """Bands, strata, groups, frozen units and the filler, from one DIAMOND search.

    ``hits`` is best ``identity_over_query`` per ordered ``(target, partner)``
    pair. Direction matters: identity over the *query* is the fraction of the
    target that a partner covers, which is the quantity the bands are declared in.
    """

    n_records = len(records)
    duplicates = [(q, s) for (q, s), value in hits.items() if value >= NEAR_DUPLICATE_IDENTITY]
    groups = connected_groups(n_records, duplicates)
    related: dict[int, set[int]] = {}
    for (q, s) in hits:
        related.setdefault(q, set()).add(s)
        related.setdefault(s, set()).add(q)

    banded: dict[str, dict[int, list[int]]] = {band: {} for band, _, _ in IDENTITY_BANDS}
    for (target, partner), value in hits.items():
        if value >= NEAR_DUPLICATE_IDENTITY and 0.0 <= value <= 100.0:
            continue
        if groups[target] == groups[partner]:
            continue
        banded[assign_identity_band(value)].setdefault(target, []).append(partner)

    candidates: dict[str, dict[int, dict[str, Any]]] = {}
    for band, per_target in banded.items():
        candidates[band] = {}
        for target, partners in per_target.items():
            if len(partners) < MIN_CONTEXT_ITEMS:
                continue
            partners = _pre_screen_sample(sorted(partners), "protein", band, target)
            overlap = pair_overlap(
                records[target], [records[p] for p in partners], modality="protein"
            )
            high = high_local_overlap(overlap, modality="protein")
            block: dict[str, Any] = {}
            for stratum, mask in (("retained", ~high), ("high_local_overlap", high)):
                selected = [partners[i] for i in np.flatnonzero(mask)]
                ordered = _order_partners("protein", band, stratum, target, selected) if selected else np.zeros(0, dtype=np.int64)
                index = {partner: position for position, partner in enumerate(partners)}
                block[stratum] = {
                    "partners": [int(value) for value in ordered],
                    "lcs": [int(overlap["lcs"][index[int(value)]]) for value in ordered],
                    "shared": [int(overlap["shared"][index[int(value)]]) for value in ordered],
                }
            candidates[band][target] = block

    units, census = _draw_units(
        modality="protein",
        groups=groups,
        candidates=candidates,
        band_floor=BAND_TARGET_FLOOR,
    )
    needed = sorted({unit["target"] for unit in units})
    filler = _choose_filler(records, related, needed)
    return {
        "records": list(records),
        "groups": groups,
        "units": units,
        "census": census,
        "related": {str(target): sorted(related.get(target, set())) for target in needed},
        "filler": filler,
        "near_duplicate_pairs": len(duplicates),
        "bootstrap_unit": (
            "the target's near-duplicate group, the connected components of the "
            f"identity >= {NEAR_DUPLICATE_IDENTITY} relation from the same DIAMOND "
            "alignment that defines the bands"
        ),
    }


def _choose_filler(
    records: Sequence[str], related: Mapping[int, set[int]], targets: Sequence[int]
) -> dict[str, Any]:
    """The single fixed filler item, identical across every target.

    Longest among the records with no detected relative anywhere in the pool, so
    that cutting it to a context item's token length almost never has to tile it,
    and so that the one item repeated across the whole campaign is not a relative
    of anything it is a control for. Ties by lowest index, which makes the choice
    a function of the pool.
    """

    free = [index for index in range(len(records)) if not related.get(index)]
    if not free:
        raise RuntimeError(
            "every pool record has a detected relative, so no unrelated filler "
            "exists; the position-only control cannot be built on this pool"
        )
    chosen = max(free, key=lambda index: (len(records[index]), -index))
    return {
        "index": int(chosen),
        "record": records[chosen],
        "symbols": len(records[chosen]),
        "candidates_with_no_detected_relative": len(free),
        "overlaps_a_target": bool(chosen in set(targets)),
    }


def text_cohort_units(
    records: Sequence[str],
    documents: Sequence[int],
    index: Bm25Index,
    *,
    max_examined: int | None = None,
) -> dict[str, Any]:
    """BM25 bands, strata and frozen units over the passage pool.

    The draw walks **one seeded permutation of the pool** and admits a passage to
    every band that supplies it at least :data:`MIN_CONTEXT_ITEMS` retained items,
    stopping once every band holds enough. That is the same uniform sample of each
    band's eligible set the protein path takes as a prefix of a permutation of a
    precomputed eligible list; it is written this way only because a BM25 ranking
    costs a sparse product and computing one for every passage in the pool to
    discover an eligibility that almost never fails would be the expensive half of
    the campaign.

    The bootstrap unit is the **source document**. One passage is carved per
    document, so each unit is its own group and no two units share a source.
    """

    n_records = len(records)
    if len(documents) != n_records:
        raise ValueError("every passage must name the document it was carved from")
    if len(set(documents)) != len(documents):
        raise ValueError(
            "two passages share a source document; the text bootstrap unit is the "
            "document, so the pool must carry one passage per document"
        )
    shingles = [word_shingles(record) for record in records]
    walk = np.random.default_rng(_unit_seed("text", "targets", "walk")).permutation(n_records)
    limit = n_records if max_examined is None else min(n_records, int(max_examined))
    candidates: dict[str, dict[int, dict[str, Any]]] = {band: {} for band, _, _ in TEXT_BANDS}
    related: dict[int, set[int]] = {}
    examined = 0
    satisfied: set[str] = set()
    for position in walk[:limit]:
        target = int(position)
        examined += 1
        ranking = index.ranking(target)
        related[target] = {int(value) for value in ranking[:TEXT_UNRELATED_RANK]}
        admitted = False
        for band, low, high in TEXT_BANDS:
            if band == "bm25_random":
                candidates_in_band = [int(value) for value in ranking[TEXT_UNRELATED_RANK:]]
            else:
                candidates_in_band = [int(value) for value in ranking[low:high]]
            if not candidates_in_band:
                continue
            partners = _pre_screen_sample(candidates_in_band, "text", band, target)
            overlap = pair_overlap(
                records[target],
                [records[p] for p in partners],
                modality="text",
                target_shingles=shingles[target],
                candidate_shingles=[shingles[p] for p in partners],
            )
            high_mask = high_local_overlap(overlap, modality="text")
            block: dict[str, Any] = {}
            for stratum, mask in (("retained", ~high_mask), ("high_local_overlap", high_mask)):
                selected = [partners[i] for i in np.flatnonzero(mask)]
                ordered = (
                    _order_partners("text", band, stratum, target, selected)
                    if selected
                    else np.zeros(0, dtype=np.int64)
                )
                lookup = {partner: place for place, partner in enumerate(partners)}
                block[stratum] = {
                    "partners": [int(value) for value in ordered],
                    "lcs": [int(overlap["lcs"][lookup[int(v)]]) for v in ordered],
                    "shared": [int(overlap["shared"][lookup[int(v)]]) for v in ordered],
                }
            candidates[band][target] = block
            if len(block["retained"]["partners"]) >= MIN_CONTEXT_ITEMS:
                admitted = True
        if admitted:
            for band in candidates:
                enough = sum(
                    1
                    for blocks in candidates[band].values()
                    if len(blocks["retained"]["partners"]) >= MIN_CONTEXT_ITEMS
                )
                if enough >= BAND_TARGET_FLOOR:
                    satisfied.add(band)
        if len(satisfied) == len(candidates):
            break

    groups = list(range(n_records))
    units, census = _draw_units(
        modality="text",
        groups=groups,
        candidates=candidates,
        band_floor=BAND_TARGET_FLOOR,
    )
    census["examined_passages"] = examined
    needed = sorted({unit["target"] for unit in units})
    filler = _choose_text_filler(records, shingles, related, needed)
    return {
        "records": list(records),
        "documents": [int(value) for value in documents],
        "groups": groups,
        "units": units,
        "census": census,
        "related": {str(target): sorted(related.get(target, set())) for target in needed},
        "filler": filler,
        "bootstrap_unit": (
            "the source document; one passage is carved per document, so each unit "
            "is its own group"
        ),
    }


def _choose_text_filler(
    records: Sequence[str],
    shingles: Sequence[frozenset[str]],
    related: Mapping[int, set[int]],
    targets: Sequence[int],
) -> dict[str, Any]:
    """The fixed text filler: outside every target's top-1000 and sharing no shingle."""

    excluded: set[int] = set(targets)
    for target in targets:
        excluded |= related.get(target, set())
    target_shingles: set[str] = set()
    for target in targets:
        target_shingles |= shingles[target]
    free = [
        index
        for index in range(len(records))
        if index not in excluded and not (shingles[index] & target_shingles)
    ]
    if not free:
        raise RuntimeError(
            "no passage sits outside every target's top-1000 while sharing no "
            f"{TEXT_SHINGLE_WORDS}-word shingle with any of them; the position-only "
            "control cannot be built on this pool"
        )
    chosen = max(free, key=lambda index: (len(records[index]), -index))
    return {
        "index": int(chosen),
        "record": records[chosen],
        "symbols": len(records[chosen]),
        "candidates_unrelated_to_every_target": len(free),
        "overlaps_a_target": False,
    }


# ------------------------------------------------------------- the arm's plan

#: Residue alphabet the protein composition match is computed over.
COMPOSITION_ALPHABET = AA20


def pool_token_lengths(
    arm: Arm, records: Sequence[str], *, modality: str, batch: int = 512
) -> np.ndarray:
    """Rendered token length of every pool record under one arm's own rendering."""

    if packing_of(arm.name) != PACKING_F16:
        return np.array(
            [len(item_ids(arm, record, modality=modality)) for record in records],
            dtype=np.int64,
        )
    lengths = np.zeros(len(records), dtype=np.int64)
    prefix = item_prefix(arm)
    for start in range(0, len(records), batch):
        chunk = list(records[start : start + batch])
        rendered = [prefix + item for item in render_records(arm, chunk, modality=modality)]
        encoded = arm.tokenizer(rendered, return_tensors=None)["input_ids"]
        for offset, ids in enumerate(encoded):
            if not ids:
                raise ValueError(f"{arm.name}: pool record {start + offset} tokenised to nothing")
            lengths[start + offset] = len(ids)
    return lengths


def composition_matrix(records: Sequence[str]) -> np.ndarray:
    """Row-normalised residue composition, for the composition-matched control."""

    lookup = {residue: position for position, residue in enumerate(COMPOSITION_ALPHABET)}
    matrix = np.zeros((len(records), len(COMPOSITION_ALPHABET)), dtype=np.float64)
    for row, record in enumerate(records):
        for residue in record:
            column = lookup.get(residue)
            if column is not None:
                matrix[row, column] += 1.0
    totals = matrix.sum(axis=1, keepdims=True)
    totals[totals == 0.0] = 1.0
    return matrix / totals


def _token_window(length: int) -> int:
    return max(TOKEN_MATCH_FLOOR, int(math.ceil(TOKEN_MATCH_TOLERANCE * length)))


def _eligible_lengths(
    lengths: np.ndarray, forbidden: np.ndarray, *, lower: int, upper: int
) -> tuple[np.ndarray, bool]:
    """Candidates inside the length window, widened downward if it is empty.

    Widening is downward only. The upper bound is not a preference but a budget:
    the matched context must fit the same 1024 positions the homologue context
    does, and a longer item would spend positions the paired contrast is supposed
    to hold fixed.
    """

    inside = (lengths >= lower) & (lengths <= upper) & ~forbidden
    if inside.any():
        return inside, False
    relaxed = (lengths <= upper) & ~forbidden
    if not relaxed.any():
        raise RuntimeError(
            f"no unrelated record of at most {upper} tokens remains for this context item"
        )
    return relaxed, True


def _match_protein(
    *,
    composition: np.ndarray,
    lengths: np.ndarray,
    forbidden: np.ndarray,
    reference: int,
    lower: int,
    upper: int,
) -> tuple[int, bool]:
    """The composition-matched unrelated natural for one homologue item."""

    eligible, widened = _eligible_lengths(lengths, forbidden, lower=lower, upper=upper)
    distance = np.abs(composition - composition[reference]).sum(axis=1)
    distance = np.where(eligible, distance, np.inf)
    return int(np.argmin(distance)), widened


def _match_text(
    *,
    unigrams: Sequence[Counter],
    lengths: np.ndarray,
    forbidden: np.ndarray,
    reference: int,
    lower: int,
    upper: int,
    seed: int,
) -> tuple[int, bool]:
    """The unigram-matched unrelated passage for one homologue item.

    A BPE unigram distance does not vectorise the way a 20-dimensional residue
    composition does, so the eligible set is subsampled to
    :data:`TEXT_MATCH_CANDIDATES` under the campaign seed and the best histogram
    intersection among them is taken. The subsample size is declared rather than
    tuned, and it travels in the plan.
    """

    mask, widened = _eligible_lengths(lengths, forbidden, lower=lower, upper=upper)
    eligible = np.flatnonzero(mask)
    order = np.random.default_rng(seed).permutation(eligible.size)[:TEXT_MATCH_CANDIDATES]
    reference_counts = unigrams[reference]
    best_index = -1
    best_score = -1.0
    for position in sorted(int(value) for value in order):
        candidate = int(eligible[position])
        counts = unigrams[candidate]
        smaller, larger = (
            (counts, reference_counts)
            if len(counts) < len(reference_counts)
            else (reference_counts, counts)
        )
        intersection = sum(min(value, larger.get(key, 0)) for key, value in smaller.items())
        total = sum(reference_counts.values()) + sum(counts.values())
        score = 2.0 * intersection / total if total else 0.0
        if score > best_score:
            best_score, best_index = score, candidate
    if best_index < 0:
        raise RuntimeError("the unigram match produced no candidate")
    return best_index, widened


def plan_units(arm: Arm, cohort: Mapping[str, Any], *, modality: str) -> dict[str, Any]:
    """Resolve k and build every condition, for one arm, before any weights load.

    Everything here is a tokenizer fact. The realised k distribution, the per-unit
    token budgets and the control constructions are published from this stage so
    that the registration's operational sequence -- publish the censuses before any
    model is loaded -- is executable rather than a sentence.
    """

    block = cohort[modality]
    records: list[str] = block["records"]
    filler: str = block["filler"]["record"]
    lengths = pool_token_lengths(arm, records, modality=modality)
    row_overhead = len(row_prefix_ids(arm))
    composition = composition_matrix(records) if modality == "protein" else None
    unigrams: list[Counter] | None = None
    shingles: list[frozenset[str]] | None = None
    if modality == "text":
        prefix = item_prefix(arm)
        encoded = arm.tokenizer(
            [prefix + item for item in render_records(arm, records, modality=modality)],
            return_tensors=None,
        )["input_ids"]
        offset = content_offset(arm)
        unigrams = [Counter(ids[offset:]) for ids in encoded]
        shingles = [word_shingles(record) for record in records]

    planned: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    tiled_filler_items = 0
    widened_matches = 0
    shuffle_exact_items = 0
    shuffle_trimmed_items = 0
    shuffle_short_items = 0
    shuffle_trim_excess: list[int] = []
    for unit in block["units"]:
        target = int(unit["target"])
        target_tokens = int(lengths[target])
        budget = POSITION_BUDGET - target_tokens - row_overhead
        partners = [int(value) for value in unit["partners"]]
        used = 0
        chosen: list[int] = []
        shuffle_seeds: list[int] = []
        for position, partner in enumerate(partners):
            cost = int(lengths[partner])
            if used + cost > budget:
                break
            used += cost
            chosen.append(partner)
            shuffle_seeds.append(_unit_seed(arm.name, unit["key"], "shuffle", position))
        if not chosen:
            refusals.append(
                {
                    "key": unit["key"],
                    "reason": (
                        f"the target renders to {target_tokens} tokens, leaving "
                        f"{budget} of the fixed {POSITION_BUDGET}-position budget, "
                        f"which does not admit even one {int(lengths[partners[0]])}"
                        "-token context item. This is a cohort fact about this arm's "
                        "rendering, not a null about the model, and the target is "
                        "never shortened to make room"
                    ),
                }
            )
            continue
        k = len(chosen)
        if k >= MAX_CONTEXT_ITEMS:
            raise RuntimeError(
                f"{arm.name}: unit {unit['key']} reached the {MAX_CONTEXT_ITEMS}-item "
                "recording cap, so k is set by the cap rather than by the token "
                "budget. The cap must never bind; raise it and rebuild the cohort"
            )
        forbidden = np.zeros(len(records), dtype=bool)
        forbidden[target] = True
        for index in block["related"].get(str(target), []):
            forbidden[int(index)] = True
        if modality == "text":
            reference_shingles = shingles[target]
            for index, entry in enumerate(shingles):
                if entry & reference_shingles:
                    forbidden[index] = True
        for partner in partners:
            forbidden[partner] = True

        unrelated: list[int] = []
        # The matched context must fit the same window the homologue context does,
        # so the per-item tolerance is spent out of the slack the homologue context
        # left rather than added on top of the budget.
        slack = budget - used
        for position, partner in enumerate(chosen):
            wanted = int(lengths[partner])
            window = _token_window(wanted)
            lower = max(1, wanted - window)
            upper = wanted + min(window, slack)
            if modality == "protein":
                match, widened = _match_protein(
                    composition=composition,
                    lengths=lengths,
                    forbidden=forbidden,
                    reference=partner,
                    lower=lower,
                    upper=upper,
                )
            else:
                match, widened = _match_text(
                    unigrams=unigrams,
                    lengths=lengths,
                    forbidden=forbidden,
                    reference=partner,
                    lower=lower,
                    upper=upper,
                    seed=_unit_seed(arm.name, unit["key"], "unrelated", position),
                )
            widened_matches += int(widened)
            slack -= int(lengths[match]) - wanted
            forbidden[match] = True
            unrelated.append(match)

        conditions = {
            HOMOLOGUE: [["pool", partner] for partner in chosen],
            UNRELATED: [["pool", partner] for partner in unrelated],
            MONO_SHUFFLED: [
                ["shuffle", partner, seed, int(lengths[partner])]
                for partner, seed in zip(chosen, shuffle_seeds)
            ],
            POSITION_ONLY: [["filler", int(lengths[partner])] for partner in chosen],
            NO_CONTEXT: [],
        }
        context_tokens: dict[str, int] = {}
        for condition, recipes in conditions.items():
            total = 0
            for recipe in recipes:
                built = build_item(
                    arm, recipe, records=records, filler=filler, modality=modality
                )
                if condition == MONO_SHUFFLED:
                    _, outcome, produced = shuffled_item(
                        arm,
                        records[int(recipe[1])],
                        modality=modality,
                        seed=int(recipe[2]),
                        target_tokens=int(recipe[3]),
                    )
                    shuffle_exact_items += int(outcome == "exact")
                    shuffle_short_items += int(outcome == "short")
                    if outcome == "trimmed":
                        shuffle_trimmed_items += 1
                        shuffle_trim_excess.append(produced - int(recipe[3]))
                total += len(built)
            context_tokens[condition] = total
            if total + target_tokens + row_overhead > POSITION_BUDGET:
                raise RuntimeError(
                    f"{arm.name}: unit {unit['key']} condition {condition!r} needs "
                    f"{total + target_tokens + row_overhead} positions, past the fixed budget"
                )
        filler_ids = item_ids(arm, filler, modality=modality)
        tiled_filler_items += sum(
            1 for partner in chosen if int(lengths[partner]) > len(filler_ids)
        )
        overlaps = [int(value) for value in unit["partner_lcs"][:k]]
        shared = [int(value) for value in unit["partner_shared"][:k]]
        planned.append(
            {
                "key": unit["key"],
                "modality": modality,
                "band": unit["band"],
                "stratum": unit["stratum"],
                "target": target,
                "group": int(unit["group"]),
                "k": k,
                "target_tokens": target_tokens,
                "budget_tokens": budget,
                "context_tokens": context_tokens,
                "context_fraction": context_tokens[HOMOLOGUE] / POSITION_BUDGET,
                "max_lcs": max(overlaps) if overlaps else 0,
                "shared_kmers": int(sum(shared)),
                "conditions": conditions,
            }
        )

    realised = np.array([unit["k"] for unit in planned], dtype=np.int64)
    fractions = np.array([unit["context_fraction"] for unit in planned], dtype=np.float64)
    matched = np.array(
        [
            unit["context_tokens"][UNRELATED] - unit["context_tokens"][HOMOLOGUE]
            for unit in planned
        ],
        dtype=np.int64,
    )
    return {
        "units": planned,
        "refusals": refusals,
        "k_distribution": summarise(realised),
        "context_fraction": summarise(fractions),
        "unrelated_token_gap": summarise(matched),
        "tiled_filler_items": tiled_filler_items,
        "widened_length_matches": widened_matches,
        "shuffle_exact_items": shuffle_exact_items,
        "shuffle_trimmed_items": shuffle_trimmed_items,
        "shuffle_short_items": shuffle_short_items,
        "shuffle_trim_excess_tokens": summarise(np.array(shuffle_trim_excess, dtype=np.float64)),
        "shuffle_seed_budget": SHUFFLE_SEED_BUDGET,
        "text_match_candidates": TEXT_MATCH_CANDIDATES if modality == "text" else None,
    }


def summarise(values: np.ndarray) -> dict[str, Any]:
    """Five-number summary of one distribution, or an explicit empty record."""

    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "q1": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "q3": float(np.percentile(values, 75)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


# ------------------------------------------------------------ the statistics


def group_bootstrap_mean(
    values: Sequence[float],
    groups: Sequence[int],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Mean of a paired per-unit statistic, with a percentile interval over groups.

    The resampling unit is the group -- a target's near-duplicate group on the
    protein side, its source document on the text side -- because two units drawn
    from one near-duplicate group are not independent evidence about an arm.
    Refused below the package's declared unit floor and reported as degenerate
    rather than published beside a unit count the reader has to notice.
    """

    array = np.asarray(values, dtype=np.float64)
    identifiers = np.asarray(groups)
    if array.ndim != 1 or identifiers.shape != array.shape:
        raise ValueError("values and groups must be one-dimensional and aligned")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not np.isfinite(array).all():
        raise ValueError("a non-finite per-unit statistic reached the bootstrap")
    unique = np.unique(identifiers)
    floor = bootstrap_unit_floor(int(unique.size))
    # ``n_rows`` counts scored units and ``n_units`` -- which the floor record
    # spells -- counts the RESAMPLING unit, the group. The two were one key
    # named ``n_units`` until the floor's copy silently overwrote it, so a
    # 60-unit stratum reported 56 and the number a reader takes for the sample
    # size was the group count.
    record: dict[str, Any] = {
        "mean": float(array.mean()) if array.size else None,
        "n_rows": int(array.size),
        "n_groups": int(unique.size),
        "resamples": int(resamples),
        "seed": int(seed),
        **floor,
    }
    if floor["degenerate"] or array.size == 0:
        record["ci95"] = None
        return record
    membership = {group: np.flatnonzero(identifiers == group) for group in unique}
    generator = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = generator.choice(unique, size=unique.size, replace=True)
        rows = np.concatenate([membership[group] for group in sampled])
        draws[index] = array[rows].mean()
    record["ci95"] = [
        float(np.percentile(draws, 100.0 * alpha / 2.0)),
        float(np.percentile(draws, 100.0 * (1.0 - alpha / 2.0))),
    ]
    return record


def paired_statistics(unit: Mapping[str, float]) -> dict[str, float]:
    """The per-unit endpoints, from one unit's per-condition per-token NLLs."""

    homologue = float(unit[HOMOLOGUE])
    unrelated = float(unit[UNRELATED])
    position_only = float(unit[POSITION_ONLY])
    if not position_only > 0.0:
        raise ValueError(
            "the position-only NLL is the fractional reduction's denominator and "
            "must be positive"
        )
    if homologue < unrelated:
        concordance = 1.0
    elif homologue > unrelated:
        concordance = 0.0
    else:
        concordance = 0.5
    return {
        "fractional_reduction": (unrelated - homologue) / position_only,
        "auroc": concordance,
        "delta_nll_homologue_minus_unrelated": homologue - unrelated,
        "delta_nll_mono_minus_homologue": float(unit[MONO_SHUFFLED]) - homologue,
        "delta_nll_position_only_minus_homologue": position_only - homologue,
    }


def terciles(units: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    """Rank-based local-overlap terciles of one band's retained units.

    Ranked and split into equal thirds rather than cut at percentile *values*,
    because the ``< 30`` band's longest common substrings take a handful of
    distinct integer values and a value cut there produces one tercile holding
    everything. Ties are ordered by unit key, so the split is a function of the
    cohort.
    """

    order = sorted(range(len(units)), key=lambda i: (units[i]["max_lcs"], units[i]["key"]))
    edges = [round(len(order) * fraction / 3.0) for fraction in range(4)]
    return {
        name: [order[position] for position in range(edges[index], edges[index + 1])]
        for index, name in enumerate(TERCILES)
    }


def endpoint_block(
    rows: Sequence[Mapping[str, Any]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Both primary endpoints and the descriptive contrasts, over one stratum."""

    if not rows:
        return {"n_rows": 0, "n_groups": 0, "fractional_reduction": None, "auroc": None}
    groups = [int(row["group"]) for row in rows]
    block: dict[str, Any] = {"n_rows": len(rows), "n_groups": len(set(groups))}
    for name in (
        "fractional_reduction",
        "auroc",
        "delta_nll_homologue_minus_unrelated",
        "delta_nll_mono_minus_homologue",
        "delta_nll_position_only_minus_homologue",
    ):
        block[name] = group_bootstrap_mean(
            [row[name] for row in rows], groups, resamples=resamples, seed=seed
        )
    block["position_only_nll_nats_per_token"] = float(
        np.mean([row[POSITION_ONLY] for row in rows])
    )
    block["max_lcs"] = summarise(np.array([row["max_lcs"] for row in rows], dtype=np.float64))
    block["k"] = summarise(np.array([row["k"] for row in rows], dtype=np.float64))
    block["denominator_note"] = (
        "fractional_reduction is (unrelated - homologue) / position_only, and its "
        "denominator travels with it: position_only_nll_nats_per_token"
    )
    return block


def gate(arm_block: Mapping[str, Any]) -> dict[str, Any]:
    """The registration's three-clause compound, per arm.

    Clause 3 is the one the design exists for. Nothing derived from the k = 0
    diagnostic may reach this function, and it refuses a block that offers one.
    """

    if NO_CONTEXT in arm_block:
        raise ValueError(
            "the k = 0 diagnostic was passed to the gate. " + DIAGNOSTIC_NEVER_THE_EFFECT
        )
    pooled = arm_block["pooled"]
    decisive = arm_block.get("decisive_stratum")
    clauses: dict[str, Any] = {}

    def lower(block: Mapping[str, Any] | None, name: str) -> float | None:
        if not block or not block.get(name) or block[name].get("ci95") is None:
            return None
        return float(block[name]["ci95"][0])

    fractional = lower(pooled, "fractional_reduction")
    auroc = lower(pooled, "auroc")
    clauses["fractional_reduction_lower_bound_above_zero"] = {
        "lower_bound": fractional,
        "holds": bool(fractional is not None and fractional > 0.0),
    }
    clauses["auroc_lower_bound_above_half"] = {
        "lower_bound": auroc,
        "holds": bool(auroc is not None and auroc > 0.5),
    }
    decisive_fractional = lower(decisive, "fractional_reduction")
    decisive_auroc = lower(decisive, "auroc")
    clauses["holds_in_the_bottom_overlap_tercile_of_the_low_identity_band"] = {
        "band": DECISIVE_BAND,
        "tercile": TERCILES[0],
        "fractional_reduction_lower_bound": decisive_fractional,
        "auroc_lower_bound": decisive_auroc,
        "holds": bool(
            decisive_fractional is not None
            and decisive_auroc is not None
            and decisive_fractional > 0.0
            and decisive_auroc > 0.5
        ),
    }
    first_two = (
        clauses["fractional_reduction_lower_bound_above_zero"]["holds"]
        and clauses["auroc_lower_bound_above_half"]["holds"]
    )
    third = clauses["holds_in_the_bottom_overlap_tercile_of_the_low_identity_band"]["holds"]
    if first_two and third:
        outcome = "in_context_relatedness_beyond_composition_and_local_copying"
        licence = (
            "this frozen, single-sequence-pretrained decoder uses in-context "
            "sequence relatedness beyond composition and beyond local copying, on "
            "this cohort. An existence result about this arm and nothing more"
        )
    elif first_two:
        outcome = "in_context_copying_and_local_overlap"
        licence = (
            "the gain is in-context copying and local overlap, which is Kantroo et "
            "al.'s mechanism. The line closes on that reading and is not narrowed "
            "to a favourable band"
        )
    else:
        outcome = "no_gain_at_this_budget"
        licence = (
            "no gain exists on this arm at this budget -- a bounded negative about "
            "frozen decoders at k in the measured range, not a claim about "
            "in-context homologue conditioning in general, since every established "
            "positive in the literature comes from a model trained for it"
        )
    return {"clauses": clauses, "outcome": outcome, "licence": licence}


# ------------------------------------------------------- loading and checking

#: The fixed records every arm's self-check is anchored on. They are constants
#: rather than cohort draws so that the number an arm reports is comparable
#: between runs, between snapshots and between pods, and so that a checkpoint that
#: loaded differently is visible before any unit is scored.
SELF_CHECK_PROTEIN = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKAL"
    "PDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMG"
)
SELF_CHECK_TEXT = (
    "The measurement is only as good as the control it is read against, and a "
    "control drawn under a different sampling rule from the arm it controls is "
    "not a control at all."
)


def self_check_record(modality: str) -> str:
    return SELF_CHECK_PROTEIN if modality == "protein" else SELF_CHECK_TEXT


def tokenizer_arm(name: str) -> Arm:
    """One arm's declaration and tokenizer, with **no weights loaded**.

    The planning stage decides k, the control constructions and the published
    censuses, and every one of those is a tokenizer fact. Loading a checkpoint to
    read a tokenizer would put the registration's "publish the censuses before any
    model is loaded" step on a GPU for no reason. Only the rendering, tokenisation
    and planning functions here accept such an arm; anything that reaches
    ``arm.model`` raises, which is the intended failure.

    Expansion names are not registered into PANEL or STAGED_ARMS. A tokenizer-only
    handle is built from the family's own spec or checkpoint path.
    """

    from transformers import AutoConfig, AutoTokenizer

    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; arms are {list(ARMS)}")
    if name in EXCLUDED_ARMS:
        raise ValueError(f"{name} is excluded from {PRE_REGISTRATION}: {EXCLUDED_ARMS[name]}")
    if name in PANEL:
        spec = PANEL[name]
        config = AutoConfig.from_pretrained(spec.path, trust_remote_code=True)
        require_position_budget(config, arm=name)
        tokenizer = AutoTokenizer.from_pretrained(spec.path, trust_remote_code=True)
        return Arm(spec=spec, model=None, tokenizer=tokenizer, device="cpu", dtype="none")
    return _tokenizer_only_expansion(name)


def _tokenizer_only_expansion(name: str) -> Arm:
    from types import SimpleNamespace
    from transformers import AutoConfig, AutoTokenizer

    from ..core.arms import STAGED_ARMS, arm_spec, require_input_path

    packing = packing_of(name)
    if name in STAGED_ARMS:
        spec = arm_spec(name)
        config = AutoConfig.from_pretrained(spec.path, trust_remote_code=True)
        require_position_budget(config, arm=name)
        tokenizer = AutoTokenizer.from_pretrained(spec.path, trust_remote_code=True)
        return Arm(spec=spec, model=None, tokenizer=tokenizer, device="cpu", dtype="none")
    if packing == PACKING_PROGEN3:
        spec = arm_spec(name)
        tokenizer, preparer = _progen3_tokenizer_only()
        raw = json.loads(Path(spec.path, "config.json").read_text(encoding="utf-8"))
        require_position_budget(SimpleNamespace(**raw), arm=name)
        return Arm(
            spec=spec,
            model=None,
            tokenizer=tokenizer,
            device="cpu",
            dtype="none",
            serving_provenance={"progen3_preparer": preparer},
        )
    if packing == PACKING_GALACTICA:
        from ..models.galactica_fitness import CHECKPOINTS

        path = require_input_path(CHECKPOINTS[name].resolve(), name)
        return _tokenizer_only_joint(name, path, architecture="opt", tokenisation="bpe")
    if packing == PACKING_INSTRUCT:
        from ..models.instructprotein_fitness import CHECKPOINTS

        path = require_input_path(CHECKPOINTS[name].resolve(), name)
        return _tokenizer_only_joint(name, path, architecture="opt", tokenisation="residue")
    if packing == PACKING_LLAMA:
        from ..models.joint_lineage import rung

        path = require_input_path(rung(name).checkpoint.resolve(), name)
        return _tokenizer_only_joint(name, path, architecture="llama", tokenisation="sentencepiece")
    raise KeyError(f"{name}: no tokenizer-only route; packing is {packing!r}")


def _progen3_tokenizer_only() -> tuple[Any, Any]:
    """Checkpoint tokenizer via progen3. Does not load experts."""

    import os
    import sys

    from ..models.progen3 import PROGEN3_SOURCE

    source = Path(os.environ.get("TRANSFER_PROGEN3_SRC", PROGEN3_SOURCE))
    if not (source / "progen3").is_dir():
        raise FileNotFoundError(
            f"no progen3 package under {source}; set TRANSFER_PROGEN3_SRC to the "
            "src/ directory of the patched third-party copy"
        )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from progen3.batch_preparer import ProGen3BatchPreparer

    preparer = ProGen3BatchPreparer()
    return preparer.tokenizer, preparer


def _tokenizer_only_joint(
    name: str, path: Path, *, architecture: str, tokenisation: str
) -> Arm:
    from transformers import AutoConfig, AutoTokenizer

    from ..core.arms import ArmSpec
    from ..models.joint_modes import rendering, resolve

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    require_position_budget(config, arm=name)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenisation_obj = resolve(tokenizer, rendering(_joint_family(name)))
    spec = ArmSpec(
        name=name,
        path=path,
        path_variable="MODEL_ROOT",
        modality="protein",
        n_layer=1,
        d_model=1,
        tokenisation=tokenisation,
        input_format=packing_of(name),
        evaluation_cohort_source="swissprot",
        architecture=architecture,
    )
    return Arm(
        spec=spec,
        model=None,
        tokenizer=tokenizer,
        device="cpu",
        dtype="none",
        serving_provenance={"joint_tokenisation": tokenisation_obj},
    )


def rendering_check(arm: Arm, *, modality: str) -> dict[str, Any]:
    """That this arm's rendering, marker prefix and scored span resolve as declared.

    The scored span is where this campaign's whole contrast lives, so a BPE merge
    straddling the marker boundary would move it silently. Checked per arm rather
    than assumed, and reported with the two facts a reader needs: how many leading
    tokens are marker, and whether id-level concatenation agrees with tokenising
    the joined string.
    """

    record = self_check_record(modality)
    ids = item_ids(arm, record, modality=modality)
    offset, end = target_span(arm, ids, record=record)
    packing = packing_of(arm.name)
    if packing != PACKING_F16:
        packed = _self_check_packed_row(arm, record, modality=modality)
        assertions = packing_assertions(arm, packed)
        return {
            "input_format": arm.spec.input_format,
            "packing": packing,
            "tokenisation": arm.spec.tokenisation,
            "content_offset_tokens": int(offset),
            "scored_span": [int(offset), int(end)],
            "rendered_tokens": int(end),
            "shuffle_unit": SHUFFLE_UNITS[modality],
            "packing_assertions": assertions,
            "id_concatenation_note": (
                "items are tokenised separately and concatenated as ids so the target's "
                "token grid is identical under every condition"
            ),
        }
    marker_text = arm.tokenizer.decode(ids[:offset])
    rendered = item_prefix(arm) + render_records(arm, [record], modality=modality)[0]
    if not rendered.startswith(marker_text):
        raise ValueError(
            f"{arm.name}: the first {offset} tokens decode to {marker_text!r}, which "
            f"is not a prefix of the rendered item. A BPE merge crosses the marker "
            "boundary, so the scored span cannot be located and this arm stops"
        )
    joined = item_ids(arm, record, modality=modality) * 2
    together = arm.tokenizer(rendered + rendered, return_tensors=None)["input_ids"]
    return {
        "input_format": arm.spec.input_format,
        "packing": packing,
        "tokenisation": arm.spec.tokenisation,
        "item_prefix_repr": repr(item_prefix(arm)),
        "content_offset_tokens": int(offset),
        "marker_decodes_to": marker_text,
        "rendered_tokens": int(end),
        "shuffle_unit": SHUFFLE_UNITS[modality],
        "id_concatenation_matches_joint_tokenisation": bool(joined == list(together)),
        "id_concatenation_note": (
            "items are tokenised separately and concatenated as ids so the target's "
            "token grid is identical under every condition. Where this flag is false "
            "the arm's BPE would merge across an item boundary, which is exactly the "
            "drift the id-level concatenation removes"
        ),
    }


def _self_check_packed_row(arm: Arm, record: str, *, modality: str) -> list[int]:
    unit = {
        "key": "self-check",
        "target": 1,
        "conditions": {HOMOLOGUE: [["pool", 0]]},
    }
    row, _start = build_row(
        arm,
        unit,
        HOMOLOGUE,
        records=[record, record],
        filler=record,
        modality=modality,
    )
    return row


def _decode_packed_row(arm: Arm, row: Sequence[int]) -> str:
    decode = getattr(arm.tokenizer, "decode", None)
    if callable(decode):
        return decode(list(row))
    convert = getattr(arm.tokenizer, "convert_ids_to_tokens", None)
    if callable(convert):
        return "".join(str(convert(int(value)) or "") for value in row)
    return ""


def packing_assertions(arm: Arm, row: Sequence[int]) -> dict[str, bool]:
    """Refuse a packed expansion row that used a forbidden template."""

    packing = packing_of(arm.name)
    decoded = _decode_packed_row(arm, row)
    checks: dict[str, bool] = {}
    if packing == PACKING_GALACTICA:
        if "# " in decoded:
            raise ValueError(
                f"{arm.name}: packed ids contain a '#' heading; never use # {{context}}"
            )
        checks["no_hash_heading"] = True
    if packing == PACKING_INSTRUCT:
        if "Instruction:" in decoded:
            raise ValueError(
                f"{arm.name}: packed string contains Instruction:; do not use the instruction slot"
            )
        checks["no_instruction_slot"] = True
    if packing == PACKING_LLAMA:
        if "Superfamily=" in decoded:
            raise ValueError(
                f"{arm.name}: packed string contains Superfamily=; that template is a different experiment"
            )
        checks["no_superfamily_template"] = True
    if packing == PACKING_RITA:
        eos = _rita_eos(arm)
        for left, right in zip(row, row[1:]):
            if int(left) == eos and int(right) == eos:
                raise ValueError(f"{arm.name}: packed ids have adjacent {eos},{eos}")
        checks["no_adjacent_eos"] = True
    if packing == PACKING_GMASK:
        from ..models.proteinglm import NATIVE_PREFIX

        if NATIVE_PREFIX not in decoded and "<gmask>" not in decoded:
            # id-concatenated rows decode the prefix tokens; absence is a drift
            raise ValueError(
                f"{arm.name}: packed proteinglm row does not decode its native prefix"
            )
        checks["native_prefix_present"] = True
    return checks
