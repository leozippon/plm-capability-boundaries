"""Identity-binned homologous context: one predeclared design for E09, E10 and E11.

The question E09 asks
=====================

**How distant may a homologous sequence be and still carry usable information
about a protein's mutation effects?** The answer this module is built to produce
is a *curve*: the held-out ranking gain of a frozen decoder, as a function of the
sequence identity between the target and the homologue placed in its context,
together with the identity at which that gain vanishes.

What this replaces, stated accurately
=====================================

The experiment this one succeeds (``results/R5/remote_homology_20260924``) did
**not** choose its identity cut after seeing its fits: its strata declaration is
digest-bound to the cohort, label-blind and timestamped before the panel. That
charge does not apply to it and is not made here. Its three real limits are the
ones this design answers:

* **The corpus searched was not the corpus the arms were trained on.** The
  repository's own registry says so: ``uniref90_bfd30`` arms were banded through a
  staged UniRef50 snapshot, which under-counts retrievable support
  (:data:`~..core.arms.UNIREF90_BFD30_INCOMPLETE_SEARCH`), so a group called
  remote can lie inside those arms' training distribution. The two arms that read
  positive on remote groups only are exactly two of those arms. This design
  therefore searches a corpus the registry names for the arm being read, and
  :mod:`~.corpus_distance` refuses an arm whose corpus it cannot stand for.
* **The snapshot had no release identity.** A band against an unidentifiable
  corpus cannot be reproduced or dated. Every corpus used here is admitted only
  with a published release record or, failing that, pinned as an object and
  reported with ``release: null`` and never called by a corpus name.
* **The close-identity positive control was underpowered by one to two orders of
  magnitude**, so "no arm passes the close control" was never evidence that no
  close effect exists. Power is therefore part of this design rather than a
  post-hoc excuse: see "Equal power, not equal width" below.

Every edge, screen, floor, seed and referent is a module constant here, in the
versioned tree, and :func:`declaration_digest` is written into every artefact any
stage of this experiment produces. A run whose digest differs from the digest in
its own inputs is refused rather than reconciled. Nothing in this file is a
function of a measured label, a model score or a fitted prediction: bin
membership is a function of an alignment, context length of the target, and the
unrelated control of the target. None of them can be a function of the gain they
are used to read.

Equal power, not equal width
============================

Identity bins of equal width do not carry equal power, and a bin whose interval
spans zero on thin support says nothing about homology. Two declarations follow:

* A bin is **admitted** only if its own retrieval support reaches
  :data:`GROUP_FLOOR` independent family groups (:func:`admitted_bins`).
  Admission is decided by retrieval alone, before any model is loaded.
* The **primary** panel reads each bin against the shared referent on that bin's
  own paired support: every point on the curve is a paired, within-target
  contrast, and every point carries its own group count, standard error and the
  group count it would need to resolve its own estimate
  (:func:`required_groups`). Measured on the real retrieval, that gives 13 to 114
  independent clusters per bin -- all above the floor and within a factor of
  nine, which is the most power this cohort can carry.
* The **complete-case** panel over the admitted bins (:func:`balanced_targets`)
  would hold group support identical across bins by construction, which is
  stronger still. It is therefore computed and reported -- but on the real
  retrieval it reaches only 7 clusters, below the floor, because a target has to
  supply five bins *and* both controls at a matched token total. It is reported
  as unsupported rather than quietly promoted, and the per-bin target overlap is
  published so that a difference between two bins can be read against how much
  of their support they share.

An unresolved bin is reported as unresolved. It is never reported as a bin where
the gain vanished.

What is fixed, and what is an outcome
=====================================

Fixed: the identity bins, the identity definition, the coverage floor, the search
parameters, the per-item token-length referent, the item-count ceiling, the
overlap screens, the independence floor and the referents of both contrasts.

An outcome, reported and never tuned: how many targets each bin reaches, how many
context items fit a given target's budget, the realised identity inside each bin,
and whether a bin is *absent* for a target. **A target whose bin is empty is
recorded as absent with its reason and stays in the target inventory.** It is
never dropped silently, because a denominator that quietly follows supply makes a
thin bin look like a strong one.

Holding length constant
=======================

A context that grows with identity measures context length, not homology. So the
quantity held fixed across the conditions of one target is the **total** context
token count, matched to ``k`` times that target's own rendered item length -- a
referent that is a property of the target and not of any bin -- with ``k`` itself
identical across every condition of that target. Per-item length is bounded, not
matched: what a paired contrast has to hold fixed is how much of the window the
context occupies and how many item boundaries it carries. The realised
per-condition total is published for every target, so the balance is auditable
rather than asserted.

The copying hypothesis
======================

Kantroo, Wagner & Machta (arXiv:2504.17068) show that appending a near-copy of a
sequence to itself collapses its likelihood in ProGen2-M, that the collapse
survives to 50% divergence, and that it fires on ten-residue needles. Copying is
therefore the leading alternative explanation for any gain measured here, not one
control among many. Three consequences are declared rather than discovered:

* the matched-unrelated condition, not the empty context, is the **primary**
  referent (:data:`PRIMARY_REFERENT`);
* every (target, item) pair carries its longest common substring, and
  :data:`SELF_COPY` -- the target's own verbatim corpus record, where one exists
  -- is scored as a declared *ceiling* that is reported beside the curve and
  never pooled into it;
* for generated sequences (E11) copying is an endpoint in its own right
  (:func:`copy_verdict`), and yield is reported both before and after copy
  exclusion, on the same all-attempts denominator.

Corpus distance (E10)
=====================

E10 does not re-band anything. A target's distance to a *named* corpus is banded
with the repository's already-frozen :data:`~.homology.STRATUM_EDGES`, imported
rather than restated, and a checkpoint enters E10 only if the registry declares a
pretraining corpus that the searched corpus can stand for
(:mod:`~.corpus_distance`). A present-day UniRef release is a proxy for the
historical training snapshot, and the direction of that error is stated there.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .homology import Hit, STRATUM_EDGES, STRATUM_NAMES
from ..core.amino_acids import AA20
from ..core.statistics import MINIMUM_BOOTSTRAP_UNITS, bootstrap_unit_floor, mean_interval

SCHEMA_VERSION = "homology_context_v1"

#: Experiment identifiers this declaration serves.
EXPERIMENTS: tuple[str, ...] = ("E09", "E10", "E11")

#: When the thresholds below were frozen, in the repository, before the first
#: retrieval run of this experiment.
PREDECLARED_UTC = "2026-10-08"

# ------------------------------------------------------------------- retrieval

#: DIAMOND search parameters. ``very-sensitive`` because the load-bearing claims
#: in every bin but the top one are *negative* ones -- that nothing closer
#: exists, or that a bin is empty -- and a fast search cannot support a negative.
#: ``--masking 0`` because repeat masking truncates alignments of exactly the
#: repeat-rich records this panel contains and biases identity downward (see
#: :func:`~.homology.run_diamond_blastp`).
SENSITIVITY = "very-sensitive"
EVALUE = 1e-3
MAX_TARGET_SEQS = 5000
MASKING = 0

#: Identity is percent of the **target** identically matched,
#: ``100 * nident / qlen``, never DIAMOND's ``pident``: a corpus record that is a
#: 60%-length fragment of the target scores 100 on ``pident`` while carrying
#: nothing like the target's sequence.
IDENTITY_DEFINITION = "100 * nident / qlen (percent of the target identically matched)"

#: A hit enters a bin only if its alignment spans at least this fraction of the
#: target. Without it the lower bins fill with short local matches whose
#: relationship to the target is a domain hit rather than a homologous sequence,
#: and the context item would not be comparable in length to the items of the
#: higher bins.
COVERAGE_FLOOR = 0.80

#: A context item is the aligned subject segment with its gap characters removed,
#: not the whole corpus record. The segment is what the alignment evidences; the
#: rest of a corpus record may be unrelated domains of arbitrary length, which
#: would break the length match without adding homologous information.
ITEM_SOURCE = "aligned subject segment of the HSP, gaps removed"

# ------------------------------------------------------------- identity bins

#: The primary design axis, as half-open ``[low, high)`` percent-identity bins.
#: The top edge admits 100.0 exactly, in the convention
#: :data:`~.homology.STRATUM_EDGES` already uses.
IDENTITY_BINS: tuple[tuple[str, float, float], ...] = (
    ("id_90_100", 90.0, 100.000001),
    ("id_70_90", 70.0, 90.0),
    ("id_50_70", 50.0, 70.0),
    ("id_30_50", 30.0, 50.0),
    ("id_lt_30", 0.0, 30.0),
)

#: Bin names, highest identity first.
BIN_NAMES: tuple[str, ...] = tuple(name for name, _, _ in IDENTITY_BINS)

#: Bin names in curve order: increasing identity along the x axis.
CURVE_ORDER: tuple[str, ...] = tuple(reversed(BIN_NAMES))

#: Candidates retained per target per bin. Four times the item ceiling, so the
#: token-length match has supply to choose from, and small enough that the
#: artefact stays pushable. The retention order is frozen in
#: :func:`bin_candidates` and reads no outcome.
BIN_CANDIDATE_CAP = 16

# -------------------------------------------------------------- the conditions

NO_CONTEXT = "no_context"
UNRELATED = "unrelated"
SELF_COPY = "self_copy"

#: Scored in this order: both controls before any homologue bin, and the
#: homologue bins from the most distant to the closest, so the condition the
#: experiment hopes for is measured last.
CONTROL_CONDITIONS: tuple[str, ...] = (NO_CONTEXT, UNRELATED)
CONDITIONS: tuple[str, ...] = CONTROL_CONDITIONS + CURVE_ORDER

#: The matched-unrelated condition. Isolates bulk composition and the mere
#: presence of a filled context from homology, and is the referent of the
#: primary contrast.
PRIMARY_REFERENT = UNRELATED

#: The empty context. A diagnostic referent: it differs from a homologue
#: condition in position occupancy as well as in content, so a gain measured
#: against it is not evidence about homology on its own.
SECONDARY_REFERENT = NO_CONTEXT

#: The evolutionary-profile referent: the frozen position-specific profile score
#: of the same variants, carried with the cohort and built by this repository's own
#: retrieval-bound stage from a UniRef50 search of the same wild type.
#:
#: It is a declared referent rather than a footnote because of what happened in
#: the phenotype programme on this very cohort: apparent ranking gains of about
#: +0.10 collapsed to +0.008 and went unresolved once an evolutionary-profile
#: control was added. A context gain that does not exceed what a count model of
#: homologous sequence already supplies is not evidence that the decoder uses
#: homology; it is evidence that homology is informative, which was never in
#: question. Every bin is therefore read against this referent as well.
PROFILE_REFERENT = "profile_lookup"

#: The three referents, in the order they are reported.
REFERENTS: tuple[str, ...] = (PRIMARY_REFERENT, SECONDARY_REFERENT, PROFILE_REFERENT)

#: The target's own verbatim corpus record, scored where one exists. A declared
#: ceiling on the curve and the price of outright copying; reported beside the
#: curve and never pooled into any bin.
CEILING_CONDITION = SELF_COPY

CONDITION_PURPOSE: dict[str, str] = {
    NO_CONTEXT: (
        "the target alone, in its native rendering. Diagnostic referent only: it "
        "differs from every homologue condition in how much of the window is "
        "occupied as well as in what occupies it"
    ),
    UNRELATED: (
        "k corpus segments from other family groups, absent from the target's own "
        "hit list, screened for local overlap with the target, token-length "
        "matched to the target's own rendered length and chosen to minimise "
        "amino-acid composition distance to the target. The primary referent: it "
        "holds position occupancy, item count, item length and bulk composition "
        "fixed and varies only homology"
    ),
    SELF_COPY: (
        "the target's own verbatim corpus record. The declared ceiling, and the "
        "price of outright copying"
    ),
    PROFILE_REFERENT: (
        "the frozen position-specific profile score of the same variants, from this "
        "repository's own retrieval-bound stage. A referent, not a condition: it is "
        "a count model of homologous sequence and no decoder is involved"
    ),
    **{
        name: (
            f"k retrieved relatives whose identity over the target lies in "
            f"[{low}, {high}) percent, each covering at least "
            f"{COVERAGE_FLOOR:.0%} of the target"
        )
        for name, low, high in IDENTITY_BINS
    },
}

# ------------------------------------------------------------- the token budget

#: Positions every condition of every arm is built inside. A budget this
#: experiment imposes, not a checkpoint ceiling, so that one target's conditions
#: are comparable; the ceiling itself is read from each arm's own configuration
#: by :func:`~.context_homologue.require_position_budget`.
POSITION_BUDGET = 1024

#: Ceiling and floor on the number of context items per condition. ``k`` itself
#: is an outcome of the target's own length against the budget, identical across
#: every condition of that target, and reported as a distribution.
CONTEXT_ITEMS_MAX = 4
CONTEXT_ITEMS_MIN = 1

#: The matched quantity is the **total** context token count of a condition, and
#: the referent it is matched to is ``k`` times the target's own rendered item
#: length -- a property of the target that no bin can influence. Per-item length
#: is bounded rather than matched: what the paired contrast has to hold fixed is
#: how much of the window the context occupies and how many item boundaries it
#: contains, not which item carries which share of it.
#:
#: **Measured, and the reason these are two numbers rather than one.** A single
#: per-item window of +-5% was tried first and starved the controls: on a
#: six-target interface run, a 119-token target had 113 screened donors and not
#: one subset of four inside a 114-124 token window, so the matched-unrelated
#: condition -- the primary referent -- went absent on targets whose homologue
#: bins were populated. Bounding each item at +-25% and matching the total to
#: within 5% keeps the occupancy identical to the same tolerance while leaving
#: the supply a condition needs. The replacement was made before any gain was
#: estimated, on support counts alone.
ITEM_TOKEN_TOLERANCE = 0.25
TOTAL_TOKEN_TOLERANCE = 0.05
ITEM_TOKEN_FLOOR = 2

#: Candidates considered per condition when matching the total. Four times the
#: item ceiling: enough that a subset can land on the referent, small enough that
#: the exhaustive search over k-subsets stays exact and quick.
MATCH_POOL = 4 * CONTEXT_ITEMS_MAX

# ----------------------------------------------------------------- the screens

#: Longest common substring, in residues, at or above which a candidate for the
#: *unrelated* control is refused. An unrelated item that shares a long verbatim
#: run with the target is not an unrelated item.
UNRELATED_MAX_LCS = 10

#: Donor segments retained per target for the unrelated control, ordered by
#: composition distance to the target and then by subject accession.
DONOR_POOL_CAP = 256

#: Residue-length band a donor must fall in before it can be retained.
#:
#: **Measured, and the reason the filter comes first.** The pool is ordered by
#: composition distance and then capped, so without a length filter the cap can
#: fill with segments of the wrong length and the matched-unrelated condition --
#: the primary referent -- goes absent while the homologue bins are populated. On
#: a six-target interface run that is exactly what happened: 113 retained donors
#: and not one inside a 119-token target's window. The filter is in residues
#: rather than tokens because the retrieval stage is arm-agnostic, and it is set
#: to the per-item token tolerance, which residue length tracks monotonically.
DONOR_LENGTH_TOLERANCE = ITEM_TOKEN_TOLERANCE

#: k of the k-mer used by the local-overlap and copying statistics.
OVERLAP_KMER = 7

# ------------------------------------------------- copying endpoints for E11

#: A generated sequence is a copy if **any** of these fires against **any** item
#: of the context it was generated under. All three are declared because they
#: fail differently: a long verbatim run, a reshuffled near-duplicate, and an
#: alignable relative of the prompt that shares no long run.
COPY_LCS_RESIDUES = 30
COPY_KMER_CONTAINMENT = 0.50
COPY_IDENTITY_PERCENT = 50.0

#: Minimum residues for a generated attempt to count as a product at all. Below
#: this neither a copy test nor a family test is meaningful; such attempts are
#: counted in the denominator as incomplete, never discarded.
MIN_PRODUCT_RESIDUES = 32

#: Coverage of the generated sequence an annotation hit must reach before the
#: generated sequence is called a member of the prompt's family.
FAMILY_COVERAGE_FLOOR = 0.50

# -------------------------------------- predicted structure, and its confounder

#: Length bands for the structural comparison. These are the generation lane's own
#: strata (``generation_evidence.POLICY["length_strata"]``), restated here only so
#: that this module stays importable without the generation package; a test
#: asserts the two agree, so a drift fails rather than passes quietly.
STRUCTURE_LENGTH_BANDS: tuple[tuple[int, int], ...] = (
    (16, 128),
    (129, 256),
    (257, 512),
    (513, 1024),
)

#: Products folded per condition per length band. The structural comparison is
#: **within** a band and with an equal count per condition, because predicted
#: confidence is strongly length- and completeness-dependent: a truncated
#: 64-residue fragment of a real protein folds to pLDDT 0.477 while its full
#: 320-residue form reaches 0.970. A condition that generates longer products
#: would otherwise win on pLDDT for reasons that have nothing to do with the
#: homology in its context. Every confidence number is reported beside the length
#: it was measured at.
STRUCTURE_SAMPLES_PER_BAND = 16

#: The rule, declared before any structure is predicted.
STRUCTURE_COMPARISON_RULE = (
    "within each length band, draw the same number of products from every "
    "condition -- the per-band minimum across conditions, capped at "
    "STRUCTURE_SAMPLES_PER_BAND -- under a seeded permutation; compare predicted "
    "confidence within a band, and report the band-weighted comparison, the "
    "per-band length distribution and the same comparison after excluding copies. "
    "A band that any condition cannot populate is reported unused, not pooled"
)

# ------------------------- E11's conditions, and the 2026-10-10 reading of them

#: The four generation conditions of E11, each naming a declared E09 condition so
#: the two experiments are built from one retrieval artefact and one set of edges.
#: Declared here rather than in the generation stage because the structural
#: comparison, the copying diagnostic and the generation stage all name them.
CLOSE_HOMOLOG = "close_homolog"
REMOTE_HOMOLOG = "remote_homolog"
GENERATION_CONDITIONS: dict[str, str] = {
    NO_CONTEXT: NO_CONTEXT,
    UNRELATED: UNRELATED,
    CLOSE_HOMOLOG: "id_70_90",
    REMOTE_HOMOLOG: "id_30_50",
}

#: The contrast family the structural panel is read on, and the only one. Five
#: contrasts, each with its own estimand:
#:
#: * against :data:`UNRELATED` -- the primary referent, which holds position
#:   occupancy, item count, item length and bulk composition fixed and varies
#:   only homology, so this is the homology-specific contrast;
#: * against :data:`NO_CONTEXT` -- the reading referent, which differs in how
#:   much of the window is occupied as well as in what occupies it, so a gain
#:   here mixes homology with the mere presence of a prefix;
#: * ``unrelated`` against ``no_context`` -- the price of having a prefix at all,
#:   estimated rather than argued about, and the column that separates the first
#:   two readings.
#:
#: Simultaneous inference is taken over this family and the strata it is read in,
#: jointly, because the claim made from it is panel-wide.
STRUCTURE_CONTRASTS: tuple[tuple[str, str], ...] = (
    (CLOSE_HOMOLOG, UNRELATED),
    (REMOTE_HOMOLOG, UNRELATED),
    (CLOSE_HOMOLOG, NO_CONTEXT),
    (REMOTE_HOMOLOG, NO_CONTEXT),
    (UNRELATED, NO_CONTEXT),
)

#: Predicted-confidence endpoints, and which way each one reads. Each is a panel
#: of its own: a simultaneous statement over pLDDT columns says nothing about the
#: pTM columns, and the three are never pooled into one family.
STRUCTURE_CONFIDENCE_FIELDS: tuple[str, ...] = ("plddt", "ptm", "pae")
STRUCTURE_PRIMARY_CONFIDENCE = "plddt"
STRUCTURE_CONFIDENCE_HIGHER_IS_BETTER: dict[str, bool] = {
    "plddt": True,
    "ptm": True,
    "pae": False,
}

#: The pairwise confidence fields kept per folded product. They are kept as
#: arrays, not reduced to a scalar: a mean PAE of 21 A describes neither which
#: pairs the model is confident about nor whether the uncertainty is diffuse or
#: confined to one terminus, and a generated product's confidence is usually the
#: second. The scalar means stay in the record beside them for the panel.
STRUCTURE_PAIRWISE_FIELDS: tuple[str, ...] = ("pae", "pde", "distogram_logits")

#: Products drawn per condition for the unmatched comparison.
STRUCTURE_UNMATCHED_SAMPLES = 32

#: The second structural draw, and what it is for.
#:
#: The length-matched draw answers "at equal length, does homologous context
#: produce a product the folding model is more confident about". It cannot answer
#: "are the products of homologous context better", because matching on length
#: discards the way the conditions differ in length -- and they do differ: under
#: an empty context this arm reaches a native terminator in 4% of attempts
#: against 35% under a close homologue. The unmatched draw therefore takes the
#: same number of products from each condition's *own* generation distribution,
#: so the count confound is removed and the length confound is deliberately left
#: in, named, and reported beside the matched estimate rather than instead of it.
STRUCTURE_UNMATCHED_RULE = (
    "draw the same number of products from every condition -- the minimum "
    "availability across conditions, capped at STRUCTURE_UNMATCHED_SAMPLES -- "
    "under a seeded permutation of each condition's own products, with no length "
    "stratification; this estimates the difference in predicted confidence "
    "between the conditions as they generate, length included, and is reported "
    "beside the length-matched estimate, never in place of it"
)

#: The structural extension's own declaration date. It is published under its own
#: digest and adds nothing to :func:`declaration`, so every artefact frozen under
#: the 2026-10-08 declaration still validates byte for byte; what was added after
#: the fact is visible as having been added after the fact.
STRUCTURE_EXTENSION_PREDECLARED_UTC = "2026-10-10"


def structure_extension() -> dict[str, Any]:
    """The structural reading declared on 2026-10-10, as one serialisable record.

    Separate from :func:`declaration` on purpose. The frozen declaration governs
    what was generated and scored; this governs how the predicted structures are
    compared, which was fixed only once the products existed. Mixing the two
    would silently invalidate every artefact that carries the frozen digest.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "predeclared_utc": STRUCTURE_EXTENSION_PREDECLARED_UTC,
        "extends": "the frozen homology_context declaration; it alters nothing in it",
        "generation_conditions": dict(GENERATION_CONDITIONS),
        "contrasts": [list(pair) for pair in STRUCTURE_CONTRASTS],
        "confidence_fields": list(STRUCTURE_CONFIDENCE_FIELDS),
        "primary_confidence": STRUCTURE_PRIMARY_CONFIDENCE,
        "confidence_higher_is_better": dict(STRUCTURE_CONFIDENCE_HIGHER_IS_BETTER),
        "pairwise_fields": list(STRUCTURE_PAIRWISE_FIELDS),
        "matched_rule": STRUCTURE_COMPARISON_RULE,
        "unmatched_rule": STRUCTURE_UNMATCHED_RULE,
        "unmatched_samples": STRUCTURE_UNMATCHED_SAMPLES,
        "samples_per_band": STRUCTURE_SAMPLES_PER_BAND,
        "length_bands": [list(band) for band in STRUCTURE_LENGTH_BANDS],
        "unit": "wild-type family group, paired within the group",
        "group_floor": GROUP_FLOOR,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "draw_seed": DRAW_SEED,
        "copies_excluded": (
            "a product that is a copy of its context is excluded from every "
            "structural comparison, because a copy of a real relative folds like a "
            "real relative; the excluded members are counted and the rule that "
            "fired is named for each"
        ),
        "estimands": {
            "length_matched": (
                "the within-band, equal-count difference in predicted confidence "
                "between two conditions, paired inside the wild-type family group. "
                "It estimates what homologous context does at a fixed product "
                "length, and it is silent about the conditions' different length "
                "and termination behaviour, which the matching removes"
            ),
            "unmatched": (
                "the equal-count difference in predicted confidence between two "
                "conditions over each condition's own generation distribution. It "
                "estimates what the conditions deliver as they generate, and it "
                "confounds homology with product length and completeness, which "
                "differ between the conditions by construction"
            ),
        },
        "not_stability": (
            "every field here is the folding model's own confidence. None of them "
            "is thermodynamic stability, none is function, and none is measured: "
            "no wet-lab result supports any statement in this panel"
        ),
    }


def structure_extension_digest() -> str:
    """Digest of the structural extension, written into every artefact using it."""

    return _digest(structure_extension(), None)


# ------------------------------------------------------------------- inference

#: The independent unit is the wild-type family group, never the assay and never
#: the variant. The floor is the package's own.
GROUP_FLOOR = MINIMUM_BOOTSTRAP_UNITS

# ------------------------------------------------------------------- numerics

#: Rows per forward pass when scoring. **One**, and not for throughput.
#:
#: The endpoint is a difference of two scored states -- a variant's summed log
#: likelihood minus its own wild type's -- so anything that shifts a row's score
#: by an amount depending on *which batch it was scored in* lands on the endpoint
#: directly. That dependence is real and is arithmetic, not modelling: a
#: linear-algebra library reduces over the hidden dimension differently at
#: different batch extents. The position lane measured it at up to 3.24e-5 nats on
#: shared-prefix terms and answered it by scoring one row per forward, which is
#: also the protocol the pairwise and stability panels run under.
#:
#: **Measured here, on this experiment's own rows.** At eight rows per forward the
#: mutant-minus-wild differences moved by 1.495e-3 nats on ProGen2-medium and
#: 4.578e-3 on ProGen2-xlarge -- both exact binary fractions, the signature of
#: half-precision rounding -- against 2.4e-4 on the same quantity locally. The two
#: larger ProGen2 rungs are the ones that failed; the per-token figure stayed near
#: 1e-6 throughout, which is why a per-token tolerance cannot catch this.
#:
#: Widening the tolerance would have admitted an artifact and, with it, a real
#: defect of the same size. Scoring one row per forward removes the dependence at
#: source and lets the tolerance stay a genuine check.
SCORING_ROWS_PER_FORWARD = 1

#: With one row per forward the batched-versus-single comparison is degenerate, so
#: the gate becomes the one the extraction lane already declares for batch-size-one
#: scoring: the same row scored twice must give the identical number. A nonzero
#: difference there is non-determinism, not rounding, and nothing is scored under
#: it.
REPEAT_TOLERANCE_NATS = 0.0

#: The batch-extent spread is still *measured* at startup and published, because it
#: is the evidence for the protocol above rather than a number anything is gated
#: on.
BATCH_EXTENT_PROBE_ROWS = 8

#: This experiment's own draws: candidate ordering, donor ordering, generation
#: and the bootstrap.
DRAW_SEED = 20261008
BOOTSTRAP_SEED = 20261008
BOOTSTRAP_DRAWS = 10000

#: The declared reading of the curve, fixed before any number exists. Scanning
#: the curve from the highest identity bin downward, the vanishing point is the
#: lower edge of the last bin whose primary contrast resolves; the first bin that
#: does not resolve is where the gain is no longer established. "Resolves" means
#: the simultaneous 95% interval over the whole arm-by-bin family excludes zero;
#: the pointwise reading is reported beside it and is explicitly weaker.
VANISHING_POINT_RULE = (
    "read on the primary per-bin paired panel against the READING_REFERENT (the "
    "empty context): scan CURVE_ORDER from the highest identity bin downward; the "
    "vanishing point is the lower identity edge of the last bin whose contrast has "
    "a simultaneous 95% interval above zero, and the first bin below it is where "
    "the gain is no longer established. A bin that fails admission, or whose "
    "interval spans zero, is reported as unresolved -- which is not the same as a "
    "bin where the gain vanished, and a bin resolved below zero is reported as "
    "that rather than as unresolved"
)

#: Which panel decides the vanishing point, and which is reported beside it.
PRIMARY_PANEL = "per_bin_paired_support"
SECONDARY_PANEL = "complete_case_over_admitted_bins"

#: The referent the curve is **read** on, as distinct from the referent that
#: controls for content (:data:`PRIMARY_REFERENT`).
#:
#: A gain measured only against the matched-unrelated context cannot tell "a
#: homologous prefix helps less than an unrelated one" from "any prefix at all
#: hurts, and an unrelated one hurts more". Those are different claims, and the
#: second is answered by the matched-unrelated condition's own contrast against
#: the empty context -- the general prefix cost -- which is estimated as a column
#: of the no-context panel rather than argued about. So each bin is read against
#: the empty context, and the matched-unrelated comparison is reported beside it.
#:
#: This is a reading rule and lives in the inference section, so it changes no
#: digest and invalidates no scored record: all of these conditions were already
#: scored.
READING_REFERENT = NO_CONTEXT

#: What this design cannot establish, recorded here so no artefact has to
#: rediscover it.
LIMITATIONS: tuple[str, ...] = (
    "A retrieved corpus segment placed in a prefix is not a multiple sequence "
    "alignment: the curve bounds what one to four homologues carry in context, "
    "not what an alignment-based profile carries.",
    "The lowest bin exists only where sub-30% homology is still detectable by "
    "DIAMOND at e <= 1e-3; a target with no detectable distant relative is "
    "recorded absent in that bin, and the matched-unrelated condition, not the "
    "empty bin, is this design's reference for 'no homology'.",
    "A bin is a property of one search against one corpus snapshot. No detected "
    "relative is not absence from any model's pretraining corpus.",
    "Identity is a scalar summary of one alignment; two items in the same bin "
    "can carry very different information.",
    "Scores produced at different rows-per-forward are not the same arithmetic "
    "and are never pooled or compared across arms.",
    "The evolutionary-profile referent is a difference of correlations, not a "
    "nested increment over a joint baseline: a bin that exceeds the profile here "
    "has not been shown to add information to a fit that already contains it.",
    "Predicted structural confidence is not measured folding, and it rises with "
    "product length and completeness, which is why it is only ever compared "
    "within a length band at an equal count per condition.",
)


def declaration() -> dict[str, Any]:
    """Every frozen parameter of this experiment, as one serialisable record."""

    return {
        "schema_version": SCHEMA_VERSION,
        "experiments": list(EXPERIMENTS),
        "predeclared_utc": PREDECLARED_UTC,
        "search": {
            "sensitivity": SENSITIVITY,
            "evalue": EVALUE,
            "max_target_seqs": MAX_TARGET_SEQS,
            "masking": MASKING,
            "identity_definition": IDENTITY_DEFINITION,
            "coverage_floor": COVERAGE_FLOOR,
            "item_source": ITEM_SOURCE,
        },
        "identity_bins": [list(entry) for entry in IDENTITY_BINS],
        "curve_order": list(CURVE_ORDER),
        "bin_candidate_cap": BIN_CANDIDATE_CAP,
        "conditions": list(CONDITIONS),
        "ceiling_condition": CEILING_CONDITION,
        "primary_referent": PRIMARY_REFERENT,
        "secondary_referent": SECONDARY_REFERENT,
        "profile_referent": PROFILE_REFERENT,
        "referents": list(REFERENTS),
        "condition_purpose": dict(CONDITION_PURPOSE),
        "budget": {
            "position_budget": POSITION_BUDGET,
            "context_items_max": CONTEXT_ITEMS_MAX,
            "context_items_min": CONTEXT_ITEMS_MIN,
            "item_token_tolerance": ITEM_TOKEN_TOLERANCE,
            "total_token_tolerance": TOTAL_TOKEN_TOLERANCE,
            "item_token_floor": ITEM_TOKEN_FLOOR,
            "match_pool": MATCH_POOL,
            "item_token_referent": "the target's own rendered item token length",
            "matched_quantity": (
                "the total context token count of a condition, matched to the item "
                "count times the target's own rendered item length"
            ),
        },
        "screens": {
            "unrelated_max_lcs": UNRELATED_MAX_LCS,
            "donor_pool_cap": DONOR_POOL_CAP,
            "donor_length_tolerance": DONOR_LENGTH_TOLERANCE,
            "overlap_kmer": OVERLAP_KMER,
        },
        "copying": {
            "lcs_residues": COPY_LCS_RESIDUES,
            "kmer_containment": COPY_KMER_CONTAINMENT,
            "identity_percent": COPY_IDENTITY_PERCENT,
            "min_product_residues": MIN_PRODUCT_RESIDUES,
            "family_coverage_floor": FAMILY_COVERAGE_FLOOR,
        },
        "structure": {
            "length_bands": [list(band) for band in STRUCTURE_LENGTH_BANDS],
            "samples_per_band": STRUCTURE_SAMPLES_PER_BAND,
            "comparison_rule": STRUCTURE_COMPARISON_RULE,
        },
        "inference": {
            "group_floor": GROUP_FLOOR,
            "unit": "wild-type family group",
            "draw_seed": DRAW_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "vanishing_point_rule": VANISHING_POINT_RULE,
            "primary_panel": PRIMARY_PANEL,
            "secondary_panel": SECONDARY_PANEL,
            "rows_per_forward": SCORING_ROWS_PER_FORWARD,
            "repeat_tolerance_nats": REPEAT_TOLERANCE_NATS,
            "batch_extent_probe_rows": BATCH_EXTENT_PROBE_ROWS,
            "numerics_reason": (
                "the endpoint is a difference of two scored states, so a batch-extent "
                "dependent shift lands on it directly; one row per forward removes the "
                "dependence at source rather than widening a tolerance to accommodate it"
            ),
            "reading_referent": READING_REFERENT,
            "reading_referent_reason": (
                "a contrast against the matched-unrelated context alone cannot separate "
                "'homologous context helps less than unrelated context' from 'any prefix "
                "hurts'; the empty context is therefore the referent the curve is read on, "
                "and the matched-unrelated condition's own contrast against it prices the "
                "general prefix cost"
            ),
            "bin_admission": (
                f"a bin is admitted only if its own retrieval support reaches "
                f"{GROUP_FLOOR} independent family groups, decided before any model "
                "is loaded"
            ),
        },
        "corpus_distance_bands": {
            "edges": list(STRATUM_EDGES),
            "names": list(STRATUM_NAMES),
            "source": "imported from src.capability.context.homology, frozen before this experiment",
        },
        "limitations": list(LIMITATIONS),
    }


#: Which sections of the declaration each kind of input depends on.
#:
#: A single whole-declaration digest is the obvious design and the wrong one: it
#: makes a stage refuse an input over a parameter that input never read, so
#: changing the structural sample count would invalidate a 30-minute corpus scan
#: and put an operator under pressure to bypass the check. Each scope therefore
#: names exactly the sections that govern that artefact, and a stage demands the
#: scope it actually depends on.
#: Each scope names the sections the corresponding artefact's content depends on,
#: and nothing more: a hit table depends on the search parameters, a binned
#: artefact additionally on the bins and the screens, and a set of built
#: conditions additionally on the conditions, the budget and the product rules.
DIGEST_SCOPES: dict[str, tuple[str, ...] | None] = {
    "search": ("schema_version", "search"),
    "retrieval": (
        "schema_version",
        "search",
        "identity_bins",
        "curve_order",
        "bin_candidate_cap",
        "screens",
    ),
    "context": (
        "schema_version",
        "search",
        "identity_bins",
        "curve_order",
        "bin_candidate_cap",
        "screens",
        "conditions",
        "ceiling_condition",
        "primary_referent",
        "secondary_referent",
        "budget",
        "copying",
        "structure",
    ),
    # Everything, for provenance.
    "full": None,
}

#: The record key each scope's digest is published under.
DIGEST_KEYS: dict[str, str] = {
    "search": "search_sha256",
    "retrieval": "retrieval_sha256",
    "context": "context_sha256",
    "full": "declaration_sha256",
}


def _digest(payload: Mapping[str, Any], sections: tuple[str, ...] | None) -> str:
    material = payload if sections is None else {key: payload[key] for key in sections}
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def scope_digest(scope: str = "full") -> str:
    """Digest of the sections of the declaration one kind of input depends on."""

    if scope not in DIGEST_SCOPES:
        raise KeyError(f"unknown declaration scope {scope!r}; scopes are {sorted(DIGEST_SCOPES)}")
    return _digest(declaration(), DIGEST_SCOPES[scope])


def declaration_digest() -> str:
    """Digest of the whole frozen declaration, written into every artefact."""

    return scope_digest("full")


def declaration_digests() -> dict[str, str]:
    """Every scope's digest, as an artefact publishes them."""

    return {DIGEST_KEYS[scope]: scope_digest(scope) for scope in DIGEST_SCOPES}


def require_declaration(record: Mapping[str, Any], *, scope: str = "full") -> None:
    """Refuse an input whose governing declaration is not the one this code declares.

    A stage reading an artefact built under different thresholds would produce
    numbers that no declaration describes, which is the failure this whole module
    exists to prevent, so it is refused rather than reconciled. When the artefact
    does not publish the scope's digest -- it was written by code that predates the
    scope -- the digest is recomputed from the declaration the artefact itself
    carries, which is stronger than trusting a key either way.
    """

    if scope not in DIGEST_SCOPES:
        raise KeyError(f"unknown declaration scope {scope!r}; scopes are {sorted(DIGEST_SCOPES)}")
    expected = scope_digest(scope)
    observed = record.get(DIGEST_KEYS[scope])
    source = DIGEST_KEYS[scope]
    if observed is None:
        embedded = record.get("declaration")
        if isinstance(embedded, dict):
            try:
                observed = _digest(embedded, DIGEST_SCOPES[scope])
                source = "recomputed from the declaration the input carries"
            except KeyError as error:
                raise ValueError(
                    f"the input's own declaration has no {error} section, so its {scope} "
                    "scope cannot be checked"
                ) from error
    if observed != expected:
        raise ValueError(
            f"input was built under {scope} declaration {observed!r} ({source}) but this "
            f"code declares {expected!r}; rebuild the input or check out the code that "
            "produced it rather than mixing two designs"
        )


# ------------------------------------------------------------------- the bins


def assign_identity_bin(identity: float) -> str:
    """Band a percent identity over the target, using the edges frozen above."""

    value = float(identity)
    if not math.isfinite(value):
        raise ValueError("percent identity is not finite")
    if not 0.0 <= value <= 100.0:
        raise ValueError(f"percent identity {value} is outside [0, 100]")
    for name, low, high in IDENTITY_BINS:
        if low <= value < high:
            return name
    raise RuntimeError(f"percent identity {value} fell through every identity bin")


def bin_edges(name: str) -> tuple[float, float]:
    """``(low, high)`` of one bin, for reporting the curve's x axis."""

    for entry, low, high in IDENTITY_BINS:
        if entry == name:
            return low, high
    raise KeyError(f"unknown identity bin {name!r}; bins are {list(BIN_NAMES)}")


def hit_coverage(hit: Hit) -> float:
    """Fraction of the target the alignment spans."""

    if hit.qlen < 1:
        raise ValueError("a hit against a zero-length target has no coverage")
    span = hit.qend - hit.qstart + 1
    if span < 1 or span > hit.qlen:
        raise ValueError(f"alignment spans {span} of a {hit.qlen}-residue target")
    return span / hit.qlen


def hit_segment(hit: Hit) -> str:
    """The aligned subject segment, gaps removed and upper-cased."""

    if hit.sseq_gapped is None:
        raise ValueError(
            "this hit carries no subject alignment; search with ALIGNMENT_FIELDS so "
            "context items can be taken from the aligned segment"
        )
    return hit.sseq_gapped.replace("-", "").upper()


def admissible_segment(segment: str) -> bool:
    """Canonical residues only: a context item is a protein, not an annotation."""

    return bool(segment) and set(segment) <= set(AA20)


def bin_candidates(
    hits: Sequence[Hit],
    *,
    wildtype: str,
    cap: int = BIN_CANDIDATE_CAP,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Per-bin candidate items for one target, and why hits were refused.

    The retention order is descending bitscore then subject accession -- a
    function of the alignment alone. Duplicate segments collapse to their
    best-scoring occurrence so that one corpus entry present under several
    accessions cannot fill a bin by itself.

    The target's own verbatim record is removed from every bin and returned by
    :func:`self_copy_candidate` instead: a copy of the target is not a homologue
    of it, and leaving it in the top bin would let the copying mechanism masquerade
    as high-identity homology.
    """

    if cap < 1:
        raise ValueError("candidate cap must be positive")
    refusals = {
        "coverage_below_floor": 0,
        "non_canonical_segment": 0,
        "verbatim_target": 0,
        "duplicate_segment": 0,
    }
    buckets: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in BIN_NAMES}
    ordered = sorted(hits, key=lambda hit: (-hit.bitscore, hit.subject, hit.qstart))
    for hit in ordered:
        coverage = hit_coverage(hit)
        if coverage < COVERAGE_FLOOR:
            refusals["coverage_below_floor"] += 1
            continue
        segment = hit_segment(hit)
        if not admissible_segment(segment):
            refusals["non_canonical_segment"] += 1
            continue
        if segment == wildtype:
            refusals["verbatim_target"] += 1
            continue
        identity = hit.identity_over_query
        name = assign_identity_bin(identity)
        bucket = buckets[name]
        if segment in bucket:
            refusals["duplicate_segment"] += 1
            continue
        bucket[segment] = {
            "subject": hit.subject,
            "sequence": segment,
            "identity": identity,
            "pident": hit.pident,
            "coverage": coverage,
            "bitscore": hit.bitscore,
            "evalue": hit.evalue,
        }
    return {name: list(bucket.values())[:cap] for name, bucket in buckets.items()}, refusals


def self_copy_candidate(hits: Sequence[Hit], *, wildtype: str) -> dict[str, Any] | None:
    """The target's own verbatim record in the corpus, if the search found one."""

    for hit in sorted(hits, key=lambda hit: (-hit.bitscore, hit.subject)):
        if hit.sseq_gapped is None:
            continue
        if hit_segment(hit) == wildtype:
            return {
                "subject": hit.subject,
                "sequence": wildtype,
                "identity": hit.identity_over_query,
                "pident": hit.pident,
                "coverage": hit_coverage(hit),
                "bitscore": hit.bitscore,
                "evalue": hit.evalue,
            }
    return None


def max_identity_over_query(hits: Sequence[Hit]) -> float:
    """Highest identity over the target among hits clearing the coverage floor.

    Zero when nothing clears it: an e-value-filtered miss is a measured absence
    of detectable homology at this coverage, and is banded as such.
    """

    values = [hit.identity_over_query for hit in hits if hit_coverage(hit) >= COVERAGE_FLOOR]
    return max(values) if values else 0.0


# ------------------------------------------------------- composition and overlap


def composition_vector(sequence: str) -> np.ndarray:
    """AA20 frequency vector of a sequence."""

    if not sequence:
        raise ValueError("an empty sequence has no composition")
    counts = np.array([sequence.count(residue) for residue in AA20], dtype=float)
    return counts / len(sequence)


def composition_distance(left: str, right: str) -> float:
    """Squared Euclidean distance between two AA20 frequency vectors."""

    difference = composition_vector(left) - composition_vector(right)
    return float(np.square(difference).sum())


def max_lcs(target: str, candidates: Sequence[str]) -> list[int]:
    """Longest common substring of the target with each candidate, in residues.

    Delegated to :func:`~.context_homologue.longest_common_substrings` rather than
    reimplemented, and imported inside the call because that module loads the
    panel's model interfaces while this one is read on CPU-only paths.
    """

    if not candidates:
        return []
    from .context_homologue import longest_common_substrings

    return [int(value) for value in longest_common_substrings(target, list(candidates))]


def kmer_containment(query: str, reference: str, *, k: int = OVERLAP_KMER) -> float:
    """Fraction of the query's k-mers that also occur in the reference.

    A containment rather than a Jaccard: the question for a generated sequence is
    how much of *it* came from the context, which a symmetric statistic dilutes
    when the two lengths differ.
    """

    if k < 1:
        raise ValueError("k must be positive")
    if len(query) < k:
        return 0.0
    query_kmers = {query[index : index + k] for index in range(len(query) - k + 1)}
    reference_kmers = {reference[index : index + k] for index in range(max(0, len(reference) - k + 1))}
    return len(query_kmers & reference_kmers) / len(query_kmers)


def donor_pool(
    target: str,
    *,
    candidates: Iterable[Mapping[str, Any]],
    excluded_subjects: Iterable[str],
    cap: int = DONOR_POOL_CAP,
) -> list[dict[str, Any]]:
    """Unrelated-control donors for one target, ordered by composition distance.

    A donor is refused if it is outside the target's residue-length band
    (:data:`DONOR_LENGTH_TOLERANCE`), if it is in the target's own hit list, if it
    repeats a segment already retained, or if it shares a verbatim run of
    :data:`UNRELATED_MAX_LCS` residues with the target. The length filter runs
    *before* the cap, for the reason recorded on
    :data:`DONOR_LENGTH_TOLERANCE`. The ordering is composition distance then
    accession: a function of the target and the donor, with no model quantity in
    it.
    """

    blocked = set(excluded_subjects)
    span = max(1, math.floor(DONOR_LENGTH_TOLERANCE * len(target)))
    low, high = len(target) - span, len(target) + span
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        subject = str(candidate["subject"])
        segment = str(candidate["sequence"])
        if subject in blocked or segment in unique or segment == target:
            continue
        if not low <= len(segment) <= high:
            continue
        if not admissible_segment(segment):
            continue
        unique[segment] = {"subject": subject, "sequence": segment}
    retained: list[dict[str, Any]] = []
    segments = list(unique)
    if segments:
        overlaps = max_lcs(target, segments)
        for segment, overlap in zip(segments, overlaps):
            if overlap >= UNRELATED_MAX_LCS:
                continue
            row = dict(unique[segment])
            row["max_lcs_to_target"] = overlap
            row["composition_distance"] = composition_distance(segment, target)
            retained.append(row)
    retained.sort(key=lambda row: (row["composition_distance"], row["subject"]))
    return retained[:cap]


# -------------------------------------------------------------- context planning


def item_token_window(referent: int) -> tuple[int, int]:
    """Admissible item token lengths around the target's own rendered length."""

    if referent < 1:
        raise ValueError("the token referent must be positive")
    slack = max(ITEM_TOKEN_FLOOR, math.floor(ITEM_TOKEN_TOLERANCE * referent))
    return referent - slack, referent + slack


def total_token_window(*, referent: int, count: int) -> tuple[int, int]:
    """Admissible total context token counts for ``count`` items.

    This is the quantity the conditions of one target are matched on, so every
    admitted condition of that target lands within
    :data:`TOTAL_TOKEN_TOLERANCE` of the same number.
    """

    if referent < 1 or count < 1:
        raise ValueError("the token referent and the item count must be positive")
    centre = referent * count
    slack = max(ITEM_TOKEN_FLOOR, math.floor(TOTAL_TOKEN_TOLERANCE * centre))
    return centre - slack, centre + slack


def matched_subset(
    pool: Sequence[Mapping[str, Any]], *, count: int, referent: int
) -> tuple[list[Mapping[str, Any]], int]:
    """The ``count`` items whose total token length lands closest to the referent.

    Exhaustive over the pool, so the result is the best available match rather
    than a greedy approximation, and deterministic: ties are broken by the
    position of the items in the pool, which the caller has already ordered by the
    condition's own frozen rule. Nothing here reads a model quantity or a label.
    """

    import itertools

    if count < 1:
        raise ValueError("the item count must be positive")
    if len(pool) < count:
        return [], 0
    centre = referent * count
    best: tuple[tuple[int, tuple[int, ...]], tuple[int, ...]] | None = None
    for combination in itertools.combinations(range(len(pool)), count):
        total = sum(int(pool[index]["tokens"]) for index in combination)
        key = (abs(total - centre), combination)
        if best is None or key < best[0]:
            best = (key, combination)
    assert best is not None
    chosen = [pool[index] for index in best[1]]
    return chosen, sum(int(item["tokens"]) for item in chosen)


def item_count(*, room: int, referent: int) -> int:
    """How many items of the referent length fit the target's own budget.

    ``k`` is decided by the target's length against :data:`POSITION_BUDGET` and by
    nothing else, so it is identical across that target's conditions and cannot
    vary with a bin's supply or with any measured quantity.
    """

    if room < 1:
        return 0
    upper = item_token_window(referent)[1]
    if upper < 1:
        raise ValueError("the item token window is empty")
    return min(CONTEXT_ITEMS_MAX, room // upper)


@dataclass(frozen=True)
class ConditionPlan:
    """One condition of one target: its items, or why it is absent."""

    condition: str
    status: str
    items: tuple[dict[str, Any], ...] = ()
    reason: str | None = None

    def record(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "status": self.status,
            "reason": self.reason,
            "n_items": len(self.items),
            "context_tokens": sum(int(item["tokens"]) for item in self.items),
            "identities": [float(item["identity"]) for item in self.items if "identity" in item],
            "max_lcs_to_target": [int(item["max_lcs_to_target"]) for item in self.items],
            "subjects": [str(item["subject"]) for item in self.items],
        }


PLAN_STATUS: tuple[str, ...] = ("present", "absent")


def plan_conditions(
    *,
    wildtype: str,
    bins: Mapping[str, Sequence[Mapping[str, Any]]],
    donors: Sequence[Mapping[str, Any]],
    self_copy: Mapping[str, Any] | None,
    token_length: Callable[[str], int],
    referent: int,
    room: int,
) -> tuple[dict[str, ConditionPlan], dict[str, Any]]:
    """Build every condition of one target at one fixed item count and length.

    Returns the per-condition plans and the target-level budget record. Absence is
    a first-class outcome: a bin with no item inside the token window is returned
    with ``status="absent"`` and a reason, and the caller keeps the target.
    """

    k = item_count(room=room, referent=referent)
    low, high = item_token_window(referent)
    total_low, total_high = (
        total_token_window(referent=referent, count=k) if k >= CONTEXT_ITEMS_MIN else (0, 0)
    )
    budget = {
        "item_count": k,
        "item_token_referent": referent,
        "item_token_window": [low, high],
        "total_token_window": [total_low, total_high],
        "room": room,
        "position_budget": POSITION_BUDGET,
    }
    plans: dict[str, ConditionPlan] = {
        NO_CONTEXT: ConditionPlan(condition=NO_CONTEXT, status="present"),
    }
    if k < CONTEXT_ITEMS_MIN:
        reason = f"target leaves room for {k} items of {low}-{high} tokens"
        for condition in CONDITIONS[1:]:
            plans[condition] = ConditionPlan(condition=condition, status="absent", reason=reason)
        plans[SELF_COPY] = ConditionPlan(condition=SELF_COPY, status="absent", reason=reason)
        return plans, budget

    def choose(
        candidates: Sequence[Mapping[str, Any]], *, condition: str, order: Callable[[Mapping[str, Any]], Any]
    ) -> ConditionPlan:
        eligible = []
        for candidate in candidates:
            tokens = int(token_length(str(candidate["sequence"])))
            if low <= tokens <= high:
                row = dict(candidate)
                row["tokens"] = tokens
                eligible.append(row)
        if len(eligible) < k:
            return ConditionPlan(
                condition=condition,
                status="absent",
                reason=(
                    f"{len(eligible)} of {k} needed items inside the {low}-{high} token "
                    f"window (from {len(candidates)} candidates)"
                ),
            )
        eligible.sort(key=order)
        chosen, total = matched_subset(eligible[:MATCH_POOL], count=k, referent=referent)
        if not (total_low <= total <= total_high):
            return ConditionPlan(
                condition=condition,
                status="absent",
                reason=(
                    f"the closest {k}-item total is {total} tokens, outside the "
                    f"{total_low}-{total_high} window this target's conditions are "
                    f"matched on (from {len(eligible)} eligible candidates)"
                ),
            )
        sequences = [str(row["sequence"]) for row in chosen]
        overlaps = max_lcs(wildtype, sequences)
        for row, overlap in zip(chosen, overlaps):
            row["max_lcs_to_target"] = overlap
        # An unrelated donor carries no identity to the target: it has no
        # alignment to it by construction, and recording a placeholder identity
        # would put a number where a measurement does not exist.
        return ConditionPlan(condition=condition, status="present", items=tuple(chosen))

    for name in CURVE_ORDER:
        plans[name] = choose(
            list(bins.get(name, ())),
            condition=name,
            order=lambda row: (-float(row["bitscore"]), str(row["subject"])),
        )
    plans[UNRELATED] = choose(
        list(donors),
        condition=UNRELATED,
        order=lambda row: (float(row["composition_distance"]), str(row["subject"])),
    )
    if self_copy is None:
        plans[SELF_COPY] = ConditionPlan(
            condition=SELF_COPY, status="absent", reason="the target is not verbatim in the corpus"
        )
    else:
        tokens = int(token_length(str(self_copy["sequence"])))
        if tokens > room:
            plans[SELF_COPY] = ConditionPlan(
                condition=SELF_COPY,
                status="absent",
                reason=f"the verbatim record needs {tokens} tokens of {room}",
            )
        else:
            row = dict(self_copy)
            row["tokens"] = tokens
            row["max_lcs_to_target"] = len(wildtype)
            plans[SELF_COPY] = ConditionPlan(
                condition=SELF_COPY, status="present", items=(row,)
            )
    return plans, budget


def admitted_bins(cluster_support: Mapping[str, int]) -> tuple[str, ...]:
    """Bins whose own retrieval support reaches the independence floor.

    Decided by retrieval alone, before any checkpoint is loaded, so admission
    cannot be a function of a gain. A refused bin is reported as unsupported; it
    is never merged into a neighbour, because merging after the fact is how a
    boundary becomes a degree of freedom.
    """

    missing = [name for name in BIN_NAMES if name not in cluster_support]
    if missing:
        raise ValueError(f"no cluster support recorded for bins {missing}")
    return tuple(name for name in CURVE_ORDER if int(cluster_support[name]) >= GROUP_FLOOR)


def balanced_targets(
    plans: Mapping[str, Mapping[str, str]], *, bins: Sequence[str] | None = None
) -> list[str]:
    """Targets supplying both controls and every bin of ``bins``.

    The complete-case panel, and the primary one: its group support is identical
    across bins by construction, so the bins carry comparable power and the curve
    cannot move because its bins rest on different proteins. ``bins`` defaults to
    every curve bin; a caller reading an admitted subset passes that subset.
    """

    required = set(CONTROL_CONDITIONS) | set(CURVE_ORDER if bins is None else bins)
    unknown = required - set(CONDITIONS)
    if unknown:
        raise ValueError(f"{sorted(unknown)} are not declared conditions")
    return sorted(
        target
        for target, statuses in plans.items()
        if all(statuses.get(condition) == "present" for condition in required)
    )


def required_groups(point: float, standard_error: float, *, groups: int, critical: float) -> int | None:
    """Independent groups this contrast would need to resolve its own estimate.

    The statistic that separates "no effect" from "no power": at a fixed effect
    size the standard error falls as the square root of the group count, so a
    contrast whose interval spans zero needs ``(critical * se / point) ** 2``
    times its current support. ``None`` when the point estimate is zero or the
    standard error is not positive, where the question has no answer.
    """

    if not math.isfinite(point) or not math.isfinite(standard_error):
        return None
    if point == 0.0 or standard_error <= 0.0 or groups < 1:
        return None
    ratio = (critical * standard_error) / abs(point)
    return int(math.ceil(groups * ratio * ratio))


def power_record(
    *,
    point: float,
    standard_error: float,
    groups: int,
    interval: Sequence[float] | None,
    critical: float,
) -> dict[str, Any]:
    """One estimate with its power, and one word for what it establishes.

    The verdict vocabulary is :func:`~.retrieval_strata.resolution`'s, imported so
    that thin support reads ``unresolved_thin_support`` here exactly as it does
    everywhere else in this package, and an unresolved cell is never reported as a
    cell where the effect is zero.
    """

    from .retrieval_strata import direction, resolution

    excludes = None if interval is None else (float(interval[0]) > 0.0 or float(interval[1]) < 0.0)
    return {
        "point": float(point),
        "standard_error": float(standard_error),
        "groups": int(groups),
        "interval": None if interval is None else [float(interval[0]), float(interval[1])],
        "resolution": resolution(groups, excludes),
        "direction": direction(interval),
        "groups_required_to_resolve": required_groups(
            point, standard_error, groups=groups, critical=critical
        ),
    }


def bin_power_record(
    *,
    bin_name: str,
    point: float,
    standard_error: float,
    groups: int,
    interval: Sequence[float] | None,
    critical: float,
) -> dict[str, Any]:
    """:func:`power_record` for one identity bin, carrying that bin's edges."""

    low, high = bin_edges(bin_name)
    return {
        "bin": bin_name,
        "identity_low": low,
        "identity_high": min(high, 100.0),
        **power_record(
            point=point,
            standard_error=standard_error,
            groups=groups,
            interval=interval,
            critical=critical,
        ),
    }


def vanishing_point(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply :data:`VANISHING_POINT_RULE` to one arm's per-bin records.

    Scans from the highest identity bin downward over the records given, which the
    caller has already restricted to the panel the rule names.
    """

    order = {name: index for index, name in enumerate(BIN_NAMES)}
    scanned = sorted(records, key=lambda record: order[record["bin"]])
    resolved: list[Mapping[str, Any]] = []
    for record in scanned:
        if record["resolution"] == "resolved" and record["direction"] == "above_zero":
            resolved.append(record)
            continue
        break
    last = resolved[-1] if resolved else None
    stopped = scanned[len(resolved)] if len(resolved) < len(scanned) else None
    if stopped is None:
        reason = "every scanned bin resolved above zero"
    elif stopped["resolution"] == "resolved":
        # Resolved on the other side of zero: the gain did not fade here, it
        # reversed, and calling that "unresolved" would hide a finding.
        reason = f"resolved {stopped['direction']}"
    else:
        reason = stopped["resolution"]
    return {
        "rule": VANISHING_POINT_RULE,
        "bins_scanned": [record["bin"] for record in scanned],
        "resolved_positive_bins": [record["bin"] for record in resolved],
        "lowest_resolved_positive_bin": None if last is None else last["bin"],
        "vanishing_identity_percent": None if last is None else last["identity_low"],
        "scan_stopped_at_bin": None if stopped is None else stopped["bin"],
        "scan_stopped_because": reason,
        "no_bin_resolved_positive": last is None,
        "resolved_negative_bins": [
            record["bin"]
            for record in scanned
            if record["resolution"] == "resolved" and record["direction"] == "below_zero"
        ],
    }


# ------------------------------------------------------- copying endpoints (E11)


def summarise_structure_cell(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One (condition, stratum) cell of a structural comparison, descriptively.

    Length and completeness travel with every confidence number here because
    predicted confidence depends on both, and a cell whose products are truncated
    at the generation budget is a cell of incomplete products whatever its mean
    pLDDT says.
    """

    rows = list(selected)
    residues = [int(row["residues"]) for row in rows]

    def mean_of(field: str) -> float | None:
        values = [row.get(field) for row in rows]
        if not values or any(value is None for value in values):
            return None
        return float(np.mean([float(value) for value in values]))

    return {
        "folded": len(rows),
        "targets": len({row.get("target_id") for row in rows}),
        "native_terminal": sum(1 for row in rows if row.get("stop_status") == "native_terminal"),
        "budget_censored": sum(1 for row in rows if row.get("stop_status") == "budget_censored"),
        "mean_plddt": mean_of("plddt"),
        "mean_ptm": mean_of("ptm"),
        "mean_pae": mean_of("pae"),
        "mean_residues": float(np.mean(residues)) if residues else None,
        "median_residues": float(np.median(residues)) if residues else None,
        "residue_range": [min(residues), max(residues)] if residues else None,
    }


def structure_comparison(
    products: Sequence[Mapping[str, Any]], *, conditions: Sequence[str]
) -> dict[str, Any]:
    """Predicted confidence by condition, **within** a length band.

    Each band reports its per-condition mean confidence, the mean length it was
    measured at, and whether the conditions carry an equal number of folded
    products in that band. A band whose counts differ is marked
    ``not comparable`` rather than silently averaged: predicted confidence rises
    with length and completeness, so an unequal or length-skewed band is exactly
    where a spurious improvement would appear. The same comparison is repeated
    with copies excluded, because a copy of a real relative folds like a real
    relative.
    """

    out: dict[str, Any] = {}
    for low, high in STRUCTURE_LENGTH_BANDS:
        band = f"len_{low}_{high}"
        cell: dict[str, Any] = {}
        for condition in conditions:
            rows = [
                row
                for row in products
                if row.get("condition") == condition
                and row.get("plddt") is not None
                and structure_length_band(int(row["residues"])) == band
            ]
            clean = [row for row in rows if not row.get("copy_verdict", {}).get("is_copy")]

            cell[condition] = {
                **summarise_structure_cell(rows),
                "non_copy": summarise_structure_cell(clean),
            }
        counts = {value["folded"] for value in cell.values()}
        comparable = len(counts) == 1 and counts != {0}
        out[band] = {
            "residue_range": [low, high],
            "per_condition": cell,
            "equal_count_across_conditions": comparable,
            "status": "comparable" if comparable else "not comparable",
            "reason": None
            if comparable
            else "the conditions do not carry an equal number of folded products in this band",
        }
    return out


def copy_statistics(generated: str, context: Sequence[str]) -> dict[str, Any]:
    """Local copying statistics of one generated sequence against its context."""

    if not context:
        return {
            "max_lcs_to_context": 0,
            "lcs_fraction_of_product": 0.0,
            "max_kmer_containment": 0.0,
        }
    overlaps = max_lcs(generated, list(context)) if generated else [0] * len(context)
    containment = (
        [kmer_containment(generated, item) for item in context] if generated else [0.0]
    )
    longest = max(overlaps) if overlaps else 0
    return {
        "max_lcs_to_context": int(longest),
        "lcs_fraction_of_product": (longest / len(generated)) if generated else 0.0,
        "max_kmer_containment": float(max(containment)),
    }


def structure_length_band(residues: int) -> str | None:
    """Which declared length band a product falls in, or ``None`` if outside them."""

    for low, high in STRUCTURE_LENGTH_BANDS:
        if low <= int(residues) <= high:
            return f"len_{low}_{high}"
    return None


def select_structure_products(
    attempts: Sequence[Mapping[str, Any]],
    *,
    conditions: Sequence[str],
    samples_per_band: int = STRUCTURE_SAMPLES_PER_BAND,
    seed: int = DRAW_SEED,
) -> tuple[set[str], dict[str, Any]]:
    """A length-matched, equal-count selection of products to predict structure for.

    Returns the selected attempt identifiers and the record of how the selection
    was made, so the balance can be checked rather than believed. The draw is a
    seeded permutation of each (band, condition) cell, which depends on nothing
    the model produced except the product's own length.

    ``conditions`` is an **order**, not a set: the cells consume draws from one
    generator as they are visited, so passing the same conditions in a different
    order gives a different, equally valid draw. Callers reproducing a frozen
    selection must pass :data:`GENERATION_CONDITIONS` in its declared order;
    alphabetical order reproduced 11 of 128 attempts on the E11 product set.
    """

    if samples_per_band < 1:
        raise ValueError("samples per band must be positive")
    cells: dict[tuple[str, str], list[str]] = {}
    for attempt in attempts:
        if int(attempt["residues"]) < MIN_PRODUCT_RESIDUES:
            continue
        band = structure_length_band(int(attempt["residues"]))
        if band is None:
            continue
        cells.setdefault((band, str(attempt["condition"])), []).append(str(attempt["attempt_id"]))

    rng = np.random.default_rng(seed)
    selected: set[str] = set()
    record: list[dict[str, Any]] = []
    for low, high in STRUCTURE_LENGTH_BANDS:
        band = f"len_{low}_{high}"
        available = {
            condition: sorted(cells.get((band, condition), ())) for condition in conditions
        }
        count = min((len(value) for value in available.values()), default=0)
        drawn = min(count, samples_per_band)
        for condition, identifiers in available.items():
            if drawn:
                order = rng.permutation(len(identifiers))[:drawn]
                selected.update(identifiers[index] for index in sorted(order))
        record.append(
            {
                "band": band,
                "residue_range": [low, high],
                "available_per_condition": {
                    condition: len(value) for condition, value in available.items()
                },
                "drawn_per_condition": drawn,
                "status": "used" if drawn else "unused",
                "reason": None
                if drawn
                else "at least one condition supplies no product in this band",
            }
        )
    return selected, {
        "rule": STRUCTURE_COMPARISON_RULE,
        "seed": seed,
        "samples_per_band": samples_per_band,
        "min_product_residues": MIN_PRODUCT_RESIDUES,
        "bands": record,
        "selected": len(selected),
        "selected_per_condition": {
            condition: sum(
                1
                for attempt in attempts
                if str(attempt["attempt_id"]) in selected
                and str(attempt["condition"]) == condition
            )
            for condition in conditions
        },
    }


def copy_verdict(statistics: Mapping[str, Any], *, identity_percent: float | None = None) -> dict[str, Any]:
    """Is this product a copy of its context? Each rule reported separately.

    ``identity_percent`` is the alignment identity of the product over itself
    against any context item, which only an aligner can supply; ``None`` means
    the annotation has not run yet and that rule abstains rather than passing.
    """

    rules = {
        "long_verbatim_run": int(statistics["max_lcs_to_context"]) >= COPY_LCS_RESIDUES,
        "kmer_containment": float(statistics["max_kmer_containment"]) >= COPY_KMER_CONTAINMENT,
        "alignment_identity": (
            None if identity_percent is None else float(identity_percent) >= COPY_IDENTITY_PERCENT
        ),
    }
    fired = [name for name, value in rules.items() if value is True]
    return {
        "rules": rules,
        "is_copy": bool(fired),
        "fired": fired,
        "identity_rule_evaluated": rules["alignment_identity"] is not None,
    }


def select_unmatched_structure_products(
    attempts: Sequence[Mapping[str, Any]],
    *,
    conditions: Sequence[str],
    samples: int = STRUCTURE_UNMATCHED_SAMPLES,
    seed: int = DRAW_SEED,
) -> tuple[set[str], dict[str, Any]]:
    """An equal-count draw from each condition's **own** length distribution.

    The counterpart to :func:`select_structure_products`, and deliberately not a
    replacement for it: this draw keeps the length difference between the
    conditions, which the length-matched draw removes. Both are needed, because
    "at equal length homologous context helps" and "homologous context yields
    better products" are different claims and this design can separate them.

    Returns the selected attempt identifiers and the record of how the draw was
    made, with each condition's own length distribution so the confound the draw
    leaves in is visible in the artefact rather than only in this docstring.
    """

    if samples < 1:
        raise ValueError("samples must be positive")
    pools: dict[str, list[str]] = {}
    lengths: dict[str, list[int]] = {}
    for attempt in attempts:
        if int(attempt["residues"]) < MIN_PRODUCT_RESIDUES:
            continue
        condition = str(attempt["condition"])
        if condition not in conditions:
            continue
        pools.setdefault(condition, []).append(str(attempt["attempt_id"]))
        lengths.setdefault(condition, []).append(int(attempt["residues"]))
    available = {condition: sorted(pools.get(condition, ())) for condition in conditions}
    drawn = min(min((len(value) for value in available.values()), default=0), samples)

    rng = np.random.default_rng(seed)
    selected: set[str] = set()
    for condition in conditions:
        identifiers = available[condition]
        if not drawn:
            continue
        order = rng.permutation(len(identifiers))[:drawn]
        selected.update(identifiers[index] for index in sorted(order))
    return selected, {
        "rule": STRUCTURE_UNMATCHED_RULE,
        "seed": seed,
        "samples": samples,
        "min_product_residues": MIN_PRODUCT_RESIDUES,
        "available_per_condition": {
            condition: len(value) for condition, value in available.items()
        },
        "drawn_per_condition": drawn,
        "status": "used" if drawn else "unused",
        "reason": None if drawn else "at least one condition supplies no product",
        "length_distribution_per_condition": {
            condition: {
                "n": len(values),
                "median_residues": float(np.median(values)) if values else None,
                "mean_residues": float(np.mean(values)) if values else None,
                "residue_range": [min(values), max(values)] if values else None,
            }
            for condition, values in sorted(lengths.items())
        },
        "selected": len(selected),
    }


def context_identity_distribution(
    products: Sequence[Mapping[str, Any]], *, conditions: Sequence[str]
) -> dict[str, Any]:
    """How much of each product is its own conditioning context, as a distribution.

    The copying endpoint is not a verdict count. A condition could leave every
    product under the 50% copy threshold and still be reproducing half of its
    prompt, so the quantity reported here is the identity of each product to the
    context it was generated under, summarised as a distribution over the whole
    product set, beside the two substring statistics that answer differently.

    A product whose context the aligner finds no alignment to at all is recorded
    as ``unaligned``: the distribution is reported both over the aligned products
    and with the unaligned counted at zero identity, because those are different
    quantities and the second is the one the copy rule is read on. The empty
    context has no counterpart to be identical to, so its cell is
    ``not applicable`` rather than zero -- an undefined contrast is named, not
    manufactured.
    """

    quantiles = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)

    def spread(values: Sequence[float]) -> dict[str, Any]:
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            return {"n": 0, "quantiles": None, "mean": None}
        return {
            "n": int(array.size),
            "mean": float(array.mean()),
            "quantiles": {
                f"q{int(level * 100):02d}": float(np.quantile(array, level))
                for level in quantiles
            },
        }

    out: dict[str, Any] = {}
    for condition in conditions:
        rows = [
            row
            for row in products
            if str(row.get("condition")) == condition
            and int(row["residues"]) >= MIN_PRODUCT_RESIDUES
        ]
        with_context = [row for row in rows if int(row.get("context_items") or 0) > 0]
        identities = [
            row.get("context_alignment_identity")
            for row in with_context
            if row.get("context_alignment_identity") is not None
        ]
        evaluated = any("context_alignment_identity" in row for row in with_context)
        lcs = [int(row["copy_statistics"]["max_lcs_to_context"]) for row in rows]
        containment = [float(row["copy_statistics"]["max_kmer_containment"]) for row in rows]
        cell: dict[str, Any] = {
            "products": len(rows),
            "products_with_context": len(with_context),
            "max_lcs_to_context": spread(lcs),
            "max_kmer_containment": spread(containment),
            "copy_rules_fired": {
                rule: sum(
                    1
                    for row in rows
                    if (row.get("copy_verdict") or {}).get("rules", {}).get(rule) is True
                )
                for rule in ("long_verbatim_run", "kmer_containment", "alignment_identity")
            },
            "copies": sum(1 for row in rows if (row.get("copy_verdict") or {}).get("is_copy")),
        }
        if not with_context:
            cell["context_alignment_identity"] = {
                "status": "not applicable",
                "reason": (
                    "this condition supplies no context, so there is nothing for its "
                    "products to be identical to"
                ),
            }
        elif not evaluated:
            cell["context_alignment_identity"] = {
                "status": "not evaluated",
                "reason": "no aligner has annotated these products against their context",
            }
        else:
            unaligned = len(with_context) - len(identities)
            with_zeros = list(identities) + [0.0] * unaligned
            cell["context_alignment_identity"] = {
                "status": "evaluated",
                "aligned": len(identities),
                "unaligned": unaligned,
                "unaligned_meaning": (
                    f"DIAMOND at e <= {EVALUE:g} found no alignment between the product "
                    "and any item of its own context"
                ),
                "among_aligned_percent": spread(identities),
                "unaligned_as_zero_percent": spread(with_zeros),
                "at_or_over_copy_threshold": sum(
                    1 for value in identities if float(value) >= COPY_IDENTITY_PERCENT
                ),
                "copy_threshold_percent": COPY_IDENTITY_PERCENT,
            }
        out[condition] = cell
    return out


def paired_contrast_panel(
    cells: Mapping[tuple[str, str], Mapping[str, float]],
    counts: Mapping[tuple[str, str], Mapping[str, int]],
    *,
    strata: Sequence[str],
    contrasts: Sequence[Sequence[str]],
    field: str,
    higher_is_better: bool,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """A contrast panel paired inside the family group, with simultaneous bounds.

    ``cells[(stratum, condition)]`` maps a family group to that group's value of
    one endpoint -- a mean predicted confidence, a yield, anything with one number
    per group -- and ``counts`` carries how many observations stand behind each of
    those numbers. A column of the panel is one (stratum, contrast) pair and a row
    is one family group, so the cell is a within-group difference and a group that
    happens to score high cannot move any contrast. A group that supplies no
    observation to one side is **missing**, never zero, which is what restricts
    every contrast to the groups both conditions actually populate.

    Intervals come from the shared family bootstrap the context lane already uses
    for its per-bin panel, so the simultaneous statement spans this whole panel
    rather than each column separately. The pointwise interval is reported beside
    it and is explicitly the weaker reading. Declared here once because the
    structural comparison and the yield comparison are the same inference on
    different endpoints, and only the endpoint should differ between them.
    """

    # Imported here, not at module scope: this module is the context lane's
    # declaration and must stay importable without the extensions package.
    from ..extensions.phenotype_strata import shared_bootstrap

    groups = sorted({group for cell in cells.values() for group in cell})
    columns: list[dict[str, Any]] = []
    matrix_columns: list[list[float]] = []
    for stratum in strata:
        for left, right in contrasts:
            left_cell = cells.get((stratum, left), {})
            right_cell = cells.get((stratum, right), {})
            shared = sorted(set(left_cell) & set(right_cell))
            matrix_columns.append(
                [
                    float(left_cell[group]) - float(right_cell[group])
                    if group in left_cell and group in right_cell
                    else np.nan
                    for group in groups
                ]
            )
            columns.append(
                {
                    "stratum": stratum,
                    "contrast": f"{left}_minus_{right}",
                    "left": left,
                    "right": right,
                    "field": field,
                    "higher_is_better": bool(higher_is_better),
                    "paired_groups": len(shared),
                    "left_products": sum(counts.get((stratum, left), {}).values()),
                    "right_products": sum(counts.get((stratum, right), {}).values()),
                    **bootstrap_unit_floor(len(shared), minimum_units=GROUP_FLOOR),
                }
            )
    matrix = np.asarray(matrix_columns, dtype=float).T if matrix_columns else np.zeros((0, 0))
    keep = [
        index
        for index in range(matrix.shape[1])
        if int(np.isfinite(matrix[:, index]).sum()) >= 2
    ]
    dropped = [
        {
            **columns[index],
            "status": "not estimable",
            "reason": "fewer than two family groups supply both sides of this contrast",
        }
        for index in range(matrix.shape[1])
        if index not in keep
    ]
    if not keep or matrix.shape[0] < 2:
        return {
            "field": field,
            "unit": "wild-type family group, paired within the group",
            "groups": len(groups),
            "status": "no estimable contrast",
            "columns": dropped,
            "bootstrap": None,
        }
    statistics, _ = shared_bootstrap(matrix[:, keep], draws=draws, seed=seed)
    estimated = [
        {
            **columns[index],
            "status": "estimated",
            "point": float(statistics["point"][position]),
            "standard_error": float(statistics["se"][position]),
            "pointwise_interval": list(statistics["pointwise_interval"][position]),
            "simultaneous_interval": list(statistics["simultaneous_interval"][position]),
            "bootstrap_groups": int(statistics["supported_families"][position]),
        }
        for position, index in enumerate(keep)
    ]
    return {
        "field": field,
        "unit": "wild-type family group, paired within the group",
        "groups": len(groups),
        "status": "estimated",
        "family_size": len(keep),
        "critical_value": float(statistics["critical_value"]),
        "interval_reading": (
            "the simultaneous interval is the panel-wide statement over every "
            "estimated column of this field; the pointwise interval is marginal and "
            "does not control the family"
        ),
        "columns": [*estimated, *dropped],
        "bootstrap": {
            "draws": int(statistics["draws"]),
            "seed": int(statistics["seed"]),
            "method": statistics["method"],
            "missing_category_policy": statistics["missing_category_policy"],
            "jointly_rejected_draws": int(statistics["jointly_rejected_draws"]),
        },
    }


def structure_contrast_panel(
    strata: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    contrasts: Sequence[Sequence[str]] = STRUCTURE_CONTRASTS,
    field: str = STRUCTURE_PRIMARY_CONFIDENCE,
    group_key: str = "target_id",
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """One confidence field's contrast panel over the folded products.

    ``strata`` maps a stratum name -- a length band, or the single stratum of the
    unmatched draw -- to the folded products in it. A family group's value is the
    mean of that field over its own products in that stratum; the rest of the
    inference is :func:`paired_contrast_panel`.
    """

    if field not in STRUCTURE_CONFIDENCE_FIELDS:
        raise ValueError(f"{field} is not a declared confidence field")
    cells: dict[tuple[str, str], dict[str, float]] = {}
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for stratum, rows in strata.items():
        collected: dict[tuple[str, str], dict[str, list[float]]] = {}
        for row in rows:
            if row.get(field) is None:
                continue
            key = (stratum, str(row["condition"]))
            collected.setdefault(key, {}).setdefault(str(row[group_key]), []).append(
                float(row[field])
            )
        for key, by_group in collected.items():
            cells[key] = {group: float(np.mean(values)) for group, values in by_group.items()}
            counts[key] = {group: len(values) for group, values in by_group.items()}
    return paired_contrast_panel(
        cells,
        counts,
        strata=list(strata),
        contrasts=contrasts,
        field=field,
        higher_is_better=STRUCTURE_CONFIDENCE_HIGHER_IS_BETTER[field],
        draws=draws,
        seed=seed,
    )


def yield_contrast_panel(
    products: Sequence[Mapping[str, Any]],
    *,
    contrasts: Sequence[Sequence[str]] = STRUCTURE_CONTRASTS,
    group_key: str = "target_id",
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """The non-copy complete-product yield, contrasted and paired inside the group.

    The first half of E11's question -- does conditioning on homologous sequence
    change *what the model produces* -- is answered by this, not by a confidence
    score: a product counts when the decoder reached a native terminator at or
    above :data:`MIN_PRODUCT_RESIDUES` and the product is not a copy of its own
    context. The denominator is every attempt made under that condition for that
    target, so a condition that mostly runs into the generation budget is penalised
    for exactly that.

    Paired inside the family group and bounded by the same shared bootstrap as the
    structural panel, over the same declared contrast family. It is reported in one
    stratum because a yield is a property of the attempts, not of the lengths the
    attempts happened to reach: stratifying it by the length of the products that
    survived would condition the denominator on the outcome.
    """

    cells: dict[tuple[str, str], dict[str, float]] = {}
    counts: dict[tuple[str, str], dict[str, int]] = {}
    attempts: dict[tuple[str, str], dict[str, int]] = {}
    complete: dict[tuple[str, str], dict[str, int]] = {}
    for row in products:
        key = ("all_attempts", str(row["condition"]))
        group = str(row[group_key])
        attempts.setdefault(key, {}).setdefault(group, 0)
        attempts[key][group] += 1
        clean = (
            row.get("stop_status") == "native_terminal"
            and int(row["residues"]) >= MIN_PRODUCT_RESIDUES
            and not (row.get("copy_verdict") or {}).get("is_copy")
        )
        complete.setdefault(key, {}).setdefault(group, 0)
        complete[key][group] += int(bool(clean))
    for key, by_group in attempts.items():
        cells[key] = {
            group: complete[key][group] / total for group, total in by_group.items() if total
        }
        counts[key] = dict(by_group)
    panel = paired_contrast_panel(
        cells,
        counts,
        strata=["all_attempts"],
        contrasts=contrasts,
        field="non_copy_native_terminal_yield",
        higher_is_better=True,
        draws=draws,
        seed=seed,
    )
    panel["numerator"] = (
        "attempts reaching a native terminator at or above MIN_PRODUCT_RESIDUES "
        "residues and not a copy of their own context"
    )
    panel["denominator"] = "every attempt made under that condition for that family group"
    panel["per_condition"] = {
        condition: {
            "attempts": sum(attempts[("all_attempts", condition)].values()),
            "complete_non_copy": sum(complete[("all_attempts", condition)].values()),
            "yield": sum(complete[("all_attempts", condition)].values())
            / sum(attempts[("all_attempts", condition)].values()),
            "groups": len(attempts[("all_attempts", condition)]),
        }
        for _, condition in sorted(attempts)
    }
    return panel


def condition_level_summary(
    strata: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    conditions: Sequence[str],
    field: str = STRUCTURE_PRIMARY_CONFIDENCE,
    group_key: str = "target_id",
) -> dict[str, Any]:
    """Each condition's own confidence level, over family groups rather than products.

    The contrast panel reports differences; this reports the levels those
    differences are taken between, as a t-interval over the family-group means so
    that the unit matches the panel's. It is descriptive: the levels are not
    paired, so their intervals must not be read against each other.
    """

    out: dict[str, Any] = {}
    for stratum, rows in strata.items():
        cell: dict[str, Any] = {}
        for condition in conditions:
            per_group: dict[str, list[float]] = {}
            for row in rows:
                if str(row["condition"]) != condition or row.get(field) is None:
                    continue
                per_group.setdefault(str(row[group_key]), []).append(float(row[field]))
            means = [float(np.mean(values)) for values in per_group.values()]
            if len(means) < 2:
                cell[condition] = {
                    "groups": len(means),
                    "status": "no interval",
                    "reason": "fewer than two family groups",
                    "mean": float(means[0]) if means else None,
                }
                continue
            cell[condition] = {
                "groups": len(means),
                "status": "estimated",
                **mean_interval(means),
                **bootstrap_unit_floor(len(means), minimum_units=GROUP_FLOOR),
            }
        out[stratum] = cell
    return {
        "field": field,
        "unit": "wild-type family group",
        "note": (
            "unpaired levels; read the contrast panel, not the overlap of these "
            "intervals, for any difference between conditions"
        ),
        "per_stratum": out,
    }
