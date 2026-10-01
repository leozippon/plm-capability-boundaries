"""Phenotype prediction on proteins the corpus does not contain.

Why this module exists
======================

The retrieval-bound measurement bounds a protein decoder's function prediction
*relative to a lookup of its own corpus*, and it cannot do better than bound it:
every natural protein has
homologues in UniRef50, so over 187 ProteinGym wild types the least homologous is
still 55.5% identical to a cluster the model was trained on (EXP-R2-189).
Retrieval could be estimated and subtracted; it could never be excluded.

EXP-R2-190 staged a referent where it is excluded by construction. 132 of the 148
de novo designs in Tsuboyama et al. 2023 return **no** DIAMOND hit against the
60.3 M-cluster staged UniRef50 at the gate the homology control runs, against a
within-cohort natural control from the same file, the same assay and matched
length that hits at 328 of 330. This module is the measurement that referent was
staged for: the same zero-shot phenotype estimand the retrieval bound uses, run on
designs where retrieval is excluded and on the natural domains beside them where
it is not.

What the estimand is, and what it is not
========================================

Per wild type, the Spearman correlation between an arm's summed log-likelihood of
a variant and that variant's measured stability (``ddG_ML``, the column
ProteinGym itself publishes as ``DMS_score`` for its 64 Tsuboyama assays -- the
two agree to 9e-16 on every shared variant). Per-wild-type correlations are
averaged inside a unit and the unit is resampled.

This is a **sequence-likelihood** estimand, so L31 does not bind it: a likelihood
needs no token alignment between the variant and the wild type, and the
multi-residue-BPE selection effect that makes position-level interventions
undefined on part of a ProtGPT2 cohort has no purchase here.

**A homologue-free referent is not an information-free one.** DIAMOND finding no
alignment says nothing about whether a design is built from *fragments* the
corpus is full of, and a decoder does not need to have seen a sequence to score
it above chance. So the baseline that replaces the profile LOOKUP -- empty by
construction on a disjoint referent -- has to be fragment-level, and it is built
here from the corrected corpus k-mer background as a proper conditional sequence
model rather than as a heuristic.

Identification is not uniform across the arms
=============================================

ProtGPT2's declared pretraining corpus **is** the staged UniRef50, so on that arm
alone the exclusion is *identified*: absence from the searched snapshot implies
absence from the 17%-smaller 2021_04 release it was trained on. Every ProGen2
rung saw UniRef90 and BFD30, and **BFD30 is not staged and was not searched**, so
a design absent from UniRef50 may still be in the corpus those arms read. That
asymmetry runs in the direction that flatters the model and is recorded in
:data:`ARM_IDENTIFICATION` beside every arm it applies to, so a ProGen2 number
cannot borrow ProtGPT2's certificate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from ..context import profiles as P
from ..core.arms import REPO, env_path
from ..models.fitness import BLOSUM62, parse_mutant
from ..core.io import write_json
from ..context.kmer_background import ALPHABET, KmerBackground
from ..core.statistics import bootstrap_unit_floor

SCHEMA_VERSION = "r2_designed_referent_v1"

# ------------------------------------------------------------------- locations

MEGASCALE_DIR = env_path(
    "TRANSFER_MEGASCALE_DIR", REPO / "data/megascale_tsuboyama2023"
)

#: EXP-R2-190's disjointness certificate. The design/natural label, the design
#: series and the list of designs that hit the corpus all come from here rather
#: than being re-derived, so this stage cannot disagree with the certificate
#: about which sequences the referent contains.
CERTIFICATE_DIR = env_path(
    "TRANSFER_MEGASCALE_CERTIFICATE_DIR", REPO / "results/shared/megascale_disjointness_20260813"
)

#: The corrected, record-local k-mer background (EXP-R2-190 section 4).
KMER_BACKGROUND_DIR = env_path(
    "TRANSFER_KMER_BACKGROUND_DIR", REPO / "data/kmer_background/uniref50"
)

#: The retrieval-bound artefact, read for one thing: the corpus residue
#: background that the ``background_composition`` free baseline is defined
#: against. Taking it from there rather than recomputing it makes this stage's
#: composition channel the same channel the retrieval bound reports
#: (Appendix B rule 12).
RETRIEVAL_BOUND_DIR = env_path(
    "TRANSFER_RETRIEVAL_BOUND_DIR", REPO / "results/R5/retrieval_bound_20260807"
)

# ------------------------------------------------------- the background gate

#: Totals of the **corrected** record-local pass. The first pass over this corpus
#: treated the newline as an invalid symbol, which silently discarded every
#: window spanning the 60-column FASTA line wrapping -- 3.09% of 3-mers and 4.71%
#: of 4-mers. As a *distribution* the error was small (total-variation 0.00028 at
#: k=3), but one 4-mer's frequency moves by 42%, and a baseline that scores
#: fragments individually reads exactly that. This module refuses to run against
#: the superseded counts.
CORRECTED_KMER_TOTALS: Mapping[int, int] = {3: 17_154_643_378, 4: 17_093_475_529}

#: The superseded totals, named so the refusal can say which artefact it found.
SUPERSEDED_KMER_TOTALS: Mapping[int, int] = {3: 16_640_807_917, 4: 16_324_342_706}

# ------------------------------------------------------------------ the cohort

#: The phenotype column. ProteinGym publishes exactly this column as
#: ``DMS_score`` for its 64 Tsuboyama assays; higher is more stable.
PHENOTYPE_COLUMN = "ddG_ML"

#: Wild types carrying fewer eligible variants than this are not scored. The
#: distribution is bimodal -- a MegaScale wild type carries either none or more
#: than a hundred -- so the constant is reported over
#: :data:`VARIANT_FLOOR_SWEEP` and the cohort is identical at every value in it.
MIN_VARIANTS = 30
VARIANT_FLOOR_SWEEP: tuple[int, ...] = (10, 30, 50, 100)

#: Free baselines: computable from the mutation string alone, before any model
#: and before any corpus. The four after BLOSUM62 are
#: :func:`~.profiles.free_baselines`' own family, imported rather than respelled.
FREE_BASELINES: tuple[str, ...] = (
    "blosum62",
    "position_index",
    "wt_hydropathy",
    "hydropathy_change",
    "background_composition",
)

#: Fragment-level retrieval baselines: what the corpus's own k-mer statistics
#: predict, with no model. These are what stands in for the profile LOOKUP,
#: which is empty by construction on a referent with no homologues.
FRAGMENT_BASELINES: tuple[str, ...] = ("fragment_markov_k3", "fragment_markov_k4")

BASELINES: tuple[str, ...] = FREE_BASELINES + FRAGMENT_BASELINES

#: Resamples and floor, both the values every other clustered interval in this
#: line of work is taken at (``profiles.cluster_bootstrap``,
#: ``statistics.MINIMUM_BOOTSTRAP_UNITS``).
BOOTSTRAP_RESAMPLES = 2000

#: The classes :data:`ARM_IDENTIFICATION` states. Declared once so the whole
#: vocabulary is visible in one place and a new class is an explicit addition
#: rather than a string spelled fresh beside an arm.
#:
#: ``exact``
#:     absence from the searched snapshot implies absence from the trained-on
#:     corpus, so the exclusion is identified.
#: ``unbounded_in_the_model_favouring_direction``
#:     an unsearched part of the corpus may contain the design. The sign is known
#:     and the size is not: the gap runs in the direction that flatters the model.
#: ``undeclared_corpus_no_exclusion_possible``
#:     no corpus is declared, so the searched snapshot's relation to the training
#:     data is not established and the residual cannot even be signed.
#: ``declared_family_relation_unestablished``
#:     a corpus *family* is declared -- not an identity -- and neither containment
#:     direction between it and the searched snapshot is evidenced, so the
#:     residual cannot be signed either.
IDENTIFICATION_CLASSES: tuple[str, ...] = (
    "exact",
    "unbounded_in_the_model_favouring_direction",
    "undeclared_corpus_no_exclusion_possible",
    "declared_family_relation_unestablished",
)

#: What each arm's corpus lets the certificate say. Not a caveat block: the
#: certificate was run against UniRef50 alone, so it *identifies* the exclusion
#: for ProtGPT2 and does not for any other arm, and the direction of the gap is
#: recorded because it is not symmetric.
ARM_IDENTIFICATION: Mapping[str, Mapping[str, str]] = {
    "protgpt2": {
        "identification": "exact",
        "note": (
            "ProtGPT2's declared pretraining corpus is UniRef50, which is the "
            "database the certificate searched. UniRef50 2021_04, the release it "
            "saw, holds 49.9 M clusters against the searched snapshot's 60.3 M, "
            "so absence from the searched snapshot implies absence from the "
            "trained-on one and the bias runs safe. This is the only arm on "
            "which the exclusion is identified rather than bounded."
        ),
    },
    "progen2-small": {
        "identification": "unbounded_in_the_model_favouring_direction",
        "note": (
            "UniRef90 + BFD30. BFD30 is not staged and was not searched, so a "
            "design absent from UniRef50 may still be present in this arm's "
            "corpus. The gap runs in the direction that flatters the model."
        ),
    },
    "progen2-base": {
        "identification": "unbounded_in_the_model_favouring_direction",
        "note": (
            "ProGen2-base's mixture is broader still than the UniRef90 + BFD30 "
            "rungs and none of its non-UniRef50 part was searched. Same "
            "direction, at least as wide."
        ),
    },
    "progen2-medium": {
        "identification": "unbounded_in_the_model_favouring_direction",
        "note": (
            "UniRef90 + BFD30, as progen2-small; BFD30 is not staged and was not "
            "searched."
        ),
    },
    "progen2-large": {
        "identification": "unbounded_in_the_model_favouring_direction",
        "note": (
            "UniRef90 + BFD30, as progen2-medium; BFD30 is not staged and was not "
            "fully searched. The bound runs in the model-favouring direction."
        ),
    },
    "progen2-xlarge": {
        "identification": "unbounded_in_the_model_favouring_direction",
        "note": (
            "UniRef90 + BFD30, as progen2-medium; BFD30 is not staged and was not "
            "fully searched. The bound runs in the model-favouring direction."
        ),
    },
    "proteinglm-7b-clm": {
        "identification": "unbounded_in_the_model_favouring_direction",
        "note": (
            "UniRef50/S + UniRef90 + ColabFoldDB, from the checkpoint's own card. "
            "The certificate searched UniRef50 alone and the searched snapshot is "
            "not identified as that mix, so a design absent from it may still be "
            "present in this arm's corpus: the UniRef90 and ColabFoldDB parts were "
            "not searched and are not staged. The gap runs in the direction that "
            "flatters the model. Scoring is the served continuation -- residues "
            "2..L under the native <gmask><sop><eos> prefix, no tail EOS -- not a "
            "full-protein left-to-right sum over residue 1 and the prefix."
        ),
    },
    "protgpt3-1.3b": {
        "identification": "undeclared_corpus_no_exclusion_possible",
        "note": (
            "the released card identifies no training corpus, so no corpus can be "
            "attributed to this checkpoint and the certificate's UniRef50 search "
            "cannot be shown to contain or to be contained in its training data. "
            "A design's absence from UniRef50 therefore implies nothing at all "
            "about this arm and the residual cannot be signed. Scored in the "
            "format the card declares -- BOS, then the N-to-C direction token, "
            "then the sequence -- rather than the bare residue string its "
            "published tokenizer config happens to produce unconfigured."
        ),
    },
    "rita-xl": {
        "identification": "declared_family_relation_unestablished",
        "note": (
            "RITA's released documentation names UniRef-100 as the pretraining "
            "corpus family. A family is not an identity: the release and the "
            "clustering the checkpoint saw are not evidenced, and neither "
            "containment direction between that corpus and the searched UniRef50 "
            "snapshot is established. UniRef100 is also a finer clustering, so a "
            "design with no UniRef50 hit may still be near-identical to a "
            "UniRef100 entry whose cluster representative is not a hit -- which is "
            "a model-favouring gap this repository cannot measure. The residual is "
            "therefore not signed in either direction. Scored in the checkpoint's "
            "own document rendering -- the boundary <EOS>, then the sequence, "
            "with the tokenizer's terminal <EOS> still appended -- with both "
            "marker positions excluded from the scored targets, so the scored "
            "targets are the sequence's residues."
        ),
    },
    "progen3-112m": {
        "identification": "undeclared_corpus_no_exclusion_possible",
        "note": (
            "the released ProGen3-112M card states no training corpus, so no "
            "corpus can be attributed to this rung and the certificate's UniRef50 "
            "search cannot be shown to contain or to be contained in its training "
            "data. A design's absence from UniRef50 therefore implies nothing at "
            "all about this rung and the residual cannot be signed: the "
            "certificate on this arm licenses no statement about retrieval, in "
            "either direction or of any size. This measurement scores the rung "
            "left-to-right on the N-to-C direction by its own decision, which is "
            "recorded at :data:`EXCLUDED_ARMS`; the reading is on the N-to-C "
            "stratum and is never compared with this lineage's bidirectional "
            "reading or with a comparison formed on that stratum."
        ),
    },
    "progen3-3b": {
        "identification": "unbounded_in_the_model_favouring_direction",
        "note": (
            "the rung's card declares the Profluent Protein Atlas v1 as its "
            "pretraining corpus. That corpus was not searched and is not staged, "
            "so a design absent from the searched UniRef50 snapshot may still be "
            "present in this rung's corpus and the gap runs in the direction that "
            "flatters the model. The bound is wider than the ProGen2 ladder's, "
            "which is already only a bound: that ladder's searched snapshot at "
            "least overlaps the UniRef90 part of its declared mixture, and nothing "
            "in this rung's corpus overlaps this certificate at all. This "
            "measurement scores the rung left-to-right on the N-to-C direction by "
            "its own decision, which is recorded at :data:`EXCLUDED_ARMS`; the "
            "reading is on the N-to-C stratum and is never compared with this "
            "lineage's bidirectional reading or with a comparison formed on that "
            "stratum."
        ),
    },
}

#: What EXP-R2-226's ProLLaMA rungs let the certificate say. A separate table
#: from :data:`ARM_IDENTIFICATION` because those are panel arms and staged scale
#: rungs, resolved through ``arms.arm_spec``, and these are joint checkpoints
#: reached by path -- ``joint_mode_qualification.py``'s rule keeps them out of
#: ``arms.py`` entirely. The two tables answer the same question and are kept
#: apart so that neither door can be opened onto the other's wall.
#:
#: The question is the certificate's: EXP-R2-190 ran it against UniRef50 alone,
#: so what a zero-hit design licenses about a checkpoint depends on what that
#: checkpoint's corpus is known to be.
JOINT_LINEAGE_IDENTIFICATION: Mapping[str, Mapping[str, str]] = {
    "llama-2-7b": {
        "identification": "undeclared_corpus_no_exclusion_possible",
        "note": (
            "Llama-2's documentation describes its pretraining data only as "
            "publicly available online sources and declares no corpus listing and "
            "no protein content. The certificate searched UniRef50, which cannot "
            "be shown to contain or to be contained in this checkpoint's training "
            "data, so a design's absence from it implies nothing at all about this "
            "rung and the residual cannot even be signed. This rung enters the "
            "ladder as a declared floor -- its measured directional-reversal cost "
            "is -0.0013 nats per scored token -- and a correlation from it is "
            "never a protein capability."
        ),
    },
    "prollama-stage-1": {
        "identification": "unbounded_in_the_model_favouring_direction",
        "note": (
            "the corpus FAMILY is declared by the model's own training script and "
            "is UniRef50 representative ids (EXP-R2-152's correction). A family is "
            "not an identity: the searched snapshot is not evidenced as that "
            "release or that clustering, so a design absent from it may still be "
            "present in this rung's corpus. The gap runs in the direction that "
            "flatters the model."
        ),
    },
    "prollama": {
        "identification": "unbounded_in_the_model_favouring_direction",
        "note": (
            "as prollama-stage-1, and no narrower: stage 2's instruction split is "
            "derived independently from the same UniRef50 representative ids and "
            "carries no disjointness guarantee against the stage-1 corpus. Same "
            "direction, at least as wide."
        ),
    },
}

#: What the Galactica and InstructProtein protein-mode doors let the certificate
#: say. A third table, not a widening of :data:`ARM_IDENTIFICATION` or
#: :data:`JOINT_LINEAGE_IDENTIFICATION`: those two resolve an ``ArmSpec`` or a
#: ProLLaMA rung, and these five names are reached the same way ProteinGym
#: reaches them -- ``galactica_fitness`` / ``instructprotein_fitness``, not
#: ``arms.arm_spec``. The question is still the certificate's.
JOINT_RENDERING_IDENTIFICATION: Mapping[str, Mapping[str, str]] = {
    "galactica-125m": {
        "identification": "undeclared_corpus_no_exclusion_possible",
        "note": (
            "Galactica's released documentation describes a scientific corpus of "
            "papers, code, and knowledge bases. The certificate searched UniRef50, "
            "which cannot be shown to contain or to be contained in that set, so a "
            "design's absence from it implies nothing at all about this rung and "
            "the residual cannot be signed. Protein-mode context-information on "
            "this rung is unidentified; that is a recorded interface status and "
            "does not withhold this score. Scored in the declared "
            "[START_AMINO]...[END_AMINO] rendering at float32, protein-content "
            "positions only, the same door ProteinGym uses."
        ),
    },
    "galactica-1.3b": {
        "identification": "undeclared_corpus_no_exclusion_possible",
        "note": (
            "as galactica-125m: the scientific corpus is declared as a family of "
            "sources, not as an identity with the searched UniRef50 snapshot, so "
            "the residual cannot be signed. Scored in the declared "
            "[START_AMINO]...[END_AMINO] rendering at float32, protein-content "
            "positions only, the same door ProteinGym uses."
        ),
    },
    "galactica-6.7b": {
        "identification": "undeclared_corpus_no_exclusion_possible",
        "note": (
            "as galactica-1.3b. Same unsigned residual; same ProteinGym rendering "
            "and float32 arithmetic."
        ),
    },
    "galactica-30b": {
        "identification": "undeclared_corpus_no_exclusion_possible",
        "note": (
            "as galactica-1.3b. Same unsigned residual; same ProteinGym rendering "
            "and float32 arithmetic."
        ),
    },
    "instructprotein": {
        "identification": "declared_family_relation_unestablished",
        "note": (
            "InstructProtein's released documentation names UniRef100 continued "
            "pretraining and instruction tuning. A family is not an identity: the "
            "release and the clustering the checkpoint saw are not evidenced, and "
            "neither containment direction between that corpus and the searched "
            "UniRef50 snapshot is established. UniRef100 is also a finer "
            "clustering, so a design with no UniRef50 hit may still be "
            "near-identical to a UniRef100 entry -- a model-favouring gap this "
            "repository cannot measure. The residual is therefore not signed. "
            "Scored in the declared <protein>ƤA..ƤY</protein> rendering, "
            "protein-content positions only, the same door ProteinGym uses."
        ),
    },
}

#: Every identification class either table states is one of
#: :data:`IDENTIFICATION_CLASSES`. Checked at import so that a class spelled fresh
#: beside an arm cannot become a second vocabulary a reader has to reconcile.
_unknown_classes = sorted(
    {
        entry["identification"]
        for entry in list(ARM_IDENTIFICATION.values())
        + list(JOINT_LINEAGE_IDENTIFICATION.values())
        + list(JOINT_RENDERING_IDENTIFICATION.values())
        if entry["identification"] not in IDENTIFICATION_CLASSES
    }
)
if _unknown_classes:
    raise AssertionError(
        f"identification classes outside {list(IDENTIFICATION_CLASSES)}: "
        f"{_unknown_classes}"
    )

#: Arms excluded from this estimand, with the reason, so that a narrower panel is
#: a declared decision rather than a default nobody checked (L18).
#:
#: **ProGen3 is not in this table, and the ground that put it here is gone rather
#: than overridden.** It was refused because its published scoring convention is
#: the mean of the N-to-C and C-to-N summed log-likelihoods, "which is a different
#: estimand from the summed left-to-right log-likelihood every arm here is read
#: under". That ground rested on reading ProGen3 in the N-to-C arms' *reported*
#: comparison while its own number was taken on another functional. This
#: measurement scores ProGen3 on the single N-to-C direction, by decision, and
#: records it under :data:`~src.capability.models.scale_comparison.STRATUM_N_TO_C` -- so one
#: name means one functional, the deviation from the lineage's own convention is
#: declared in the payload, and the comparison the ground was protecting against
#: no longer arises. The claim this licenses is the opposite of a claim that the
#: lineage's bidirectional convention is wrong: that convention is what
#: ``retrieval_bound.py`` scores and what its ProteinGym reading is on, and the
#: two ProGen3 readings now live on two functionals that are never pooled, ranked
#: together or converted into one another.
EXCLUDED_ARMS: Mapping[str, str] = {
    "zymctrl": (
        "its rendering is EC-conditioned and a de novo design carries no EC "
        "number, so the estimand is not defined on this referent for this arm"
    ),
}

_SUBSTITUTION_TOKEN = re.compile(r"^[A-Z]\d+[A-Z]$")

_ENCODE = np.full(256, -1, dtype=np.int64)
for _index, _symbol in enumerate(ALPHABET):
    _ENCODE[ord(_symbol)] = _index


# ------------------------------------------------------------ variant grammar


def eligible_substitutions(mut_type: str) -> tuple[tuple[str, int, str], ...] | None:
    """The substitutions of one MegaScale ``mut_type``, or ``None`` if ineligible.

    A filter rather than a parser: ``dataset2`` carries wild-type rows, insertions
    and deletions alongside substitutions, and a token whose two residues are
    equal (``T38T``) is a no-op the file spells as a mutation. The grammar itself
    is :func:`~.fitness.parse_mutant`'s, imported rather than respelled, so a
    variant cannot mean one thing here and another in the retrieval-bound cohort.

    Verified against the benchmark: on the 64 ProteinGym Tsuboyama assays this
    rule selects a **strict subset** of ProteinGym's own variant set -- zero
    variants selected here are absent from ProteinGym, on all 64 -- and
    reproduces it exactly on 27 of them. The 536 variants ProteinGym has and this
    rule does not carry ``ddG_ML == "-"`` in ``dataset2``.
    """

    tokens = mut_type.split(":")
    if not all(_SUBSTITUTION_TOKEN.match(token) for token in tokens):
        return None
    parsed = parse_mutant(mut_type)
    if any(wild == mutated for wild, _, mutated in parsed):
        return None
    return parsed


def apply_substitutions(
    sequence: str, substitutions: Sequence[tuple[str, int, str]]
) -> str:
    """The variant a mutation string names, checked against the wild type.

    Raises when the wild type does not carry the residue the mutation string
    claims, which is the check that lets a cohort be transmitted as wild types
    plus mutation strings rather than as half a million sequences.
    """

    out = list(sequence)
    for wild, position, mutated in substitutions:
        if position < 1 or position > len(out):
            raise ValueError(
                f"position {position} is outside a wild type of {len(out)} residues"
            )
        if out[position - 1] != wild:
            raise ValueError(
                f"the wild type carries {out[position - 1]!r} at position "
                f"{position}, but the mutation string says {wild!r}"
            )
        out[position - 1] = mutated
    return "".join(out)


# ------------------------------------------------------------------ the cohort


@dataclass(frozen=True)
class WildType:
    """One MegaScale wild type and every eligible variant measured on it."""

    name: str
    kind: str
    series: str
    cluster: str
    zero_hit: bool
    series_zero_hit: bool
    sequence: str
    mutants: tuple[str, ...]
    phenotype: np.ndarray
    replicates: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.kind not in ("design", "natural"):
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")
        if len(self.mutants) != self.phenotype.size:
            raise ValueError(f"{self.name}: mutants and phenotype disagree in length")
        if len(self.mutants) != len(self.replicates):
            raise ValueError(f"{self.name}: mutants and replicates disagree in length")
        if not np.isfinite(self.phenotype).all():
            raise ValueError(f"{self.name}: a non-finite phenotype reached the cohort")

    @property
    def unit(self) -> str:
        """The resampling unit this wild type belongs to.

        A design's unit is its **design series** -- topology x design round,
        run-family x run, or hallucination round -- which is the unit EXP-R2-190
        enumerated and the one this experiment is pre-registered on. A natural
        domain's unit is the dataset's own ``WT_cluster``, which is the analogue
        of the 50%-identity family and is not a threshold invented here.
        """

        return f"design:{self.series}" if self.kind == "design" else f"natural:{self.cluster}"

    @property
    def substitutions(self) -> list[tuple[tuple[str, int, str], ...]]:
        return [parse_mutant(mutant) for mutant in self.mutants]

    def sequences(self) -> list[str]:
        return [
            apply_substitutions(self.sequence, entry) for entry in self.substitutions
        ]

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "series": self.series,
            "cluster": self.cluster,
            "zero_hit": self.zero_hit,
            "series_zero_hit": self.series_zero_hit,
            "sequence": self.sequence,
            "mutants": list(self.mutants),
            "phenotype": [float(value) for value in self.phenotype],
            "replicates": list(self.replicates),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "WildType":
        return cls(
            name=str(record["name"]),
            kind=str(record["kind"]),
            series=str(record["series"]),
            cluster=str(record["cluster"]),
            zero_hit=bool(record["zero_hit"]),
            series_zero_hit=bool(record["series_zero_hit"]),
            sequence=str(record["sequence"]),
            mutants=tuple(str(value) for value in record["mutants"]),
            phenotype=np.asarray(record["phenotype"], dtype=np.float64),
            replicates=tuple(int(value) for value in record["replicates"]),
        )


@dataclass(frozen=True)
class Referent:
    """The frozen cohort: every scored wild type, and how it was assembled."""

    wildtypes: tuple[WildType, ...]
    provenance: dict[str, Any]

    def side(self, kind: str, *, zero_hit_only: bool = True) -> tuple[WildType, ...]:
        """The design side or the natural control.

        ``zero_hit_only`` is what excludes the 16 designs that hit the corpus.
        It has no meaning on the natural side, where hitting the corpus is the
        expected state and the control's whole point.
        """

        if kind == "design":
            return tuple(
                wt
                for wt in self.wildtypes
                if wt.kind == "design" and (wt.zero_hit or not zero_hit_only)
            )
        return tuple(wt for wt in self.wildtypes if wt.kind == kind)

    def record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "provenance": dict(self.provenance),
            "wildtypes": [wt.record() for wt in self.wildtypes],
        }


def build_referent(
    *,
    megascale_dir: Path | None = None,
    certificate_dir: Path | None = None,
    min_variants: int = MIN_VARIANTS,
) -> Referent:
    """Assemble the referent from the staged parquet and EXP-R2-190's certificate.

    A **census**, not a draw: every wild type the certificate searched and every
    eligible variant of it. There is nothing to sample and therefore no seeded
    permutation and no skip-offset sensitivity to report (Appendix B rule 1's
    hazard is absent rather than answered).

    Duplicate measurements of one variant -- 4,071 of 531,275 rows, spread over
    53 wild types -- are averaged rather than dropped or double-weighted, and the
    number averaged into each point is carried into the artefact.
    """

    import pyarrow.parquet as pq

    megascale = Path(megascale_dir) if megascale_dir is not None else MEGASCALE_DIR
    certificates = Path(certificate_dir) if certificate_dir is not None else CERTIFICATE_DIR
    shards = sorted((megascale / "dataset2").rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no dataset2 parquet shards under {megascale}")

    certificate = json.loads((certificates / "certificate.json").read_text(encoding="utf-8"))
    index = json.loads((certificates / "query_index.json").read_text(encoding="utf-8"))
    if certificate.get("schema_version") != "megascale_disjointness_certificate_v1":
        raise ValueError(
            f"{certificates} carries schema {certificate.get('schema_version')!r}, "
            "not megascale_disjointness_certificate"
        )
    hit_designs = {entry["WT_name"] for entry in certificate["design_hits_at_gate"]}
    series_zero_hit = {
        name: bool(entry["zero_hit"]) for name, entry in certificate["design_series"].items()
    }
    catalogue = {entry["WT_name"]: entry for entry in index}

    measurements: dict[str, dict[str, list[float]]] = {}
    clusters: dict[str, str] = {}
    rows_read = 0
    for shard in shards:
        table = pq.read_table(
            shard, columns=["mut_type", "WT_name", "WT_cluster", "ddG_ML", "aa_seq"]
        ).to_pydict()
        rows_read += len(table["WT_name"])
        for mut_type, name, cluster, value, aa_seq in zip(
            table["mut_type"],
            table["WT_name"],
            table["WT_cluster"],
            table["ddG_ML"],
            table["aa_seq"],
        ):
            entry = catalogue.get(name)
            if entry is None:
                continue
            clusters.setdefault(name, str(cluster))
            if value is None or value == "-":
                continue
            substitutions = eligible_substitutions(mut_type)
            if substitutions is None:
                continue
            # The file's own variant sequence is the check that a cohort
            # transmitted as wild type + mutation string reconstructs to the
            # sequence that was actually measured.
            if apply_substitutions(entry["sequence"], substitutions) != aa_seq:
                raise ValueError(
                    f"{name}/{mut_type}: the reconstructed variant differs from "
                    "the sequence the file records"
                )
            measurements.setdefault(name, {}).setdefault(mut_type, []).append(float(value))

    wildtypes: list[WildType] = []
    below_floor: list[dict[str, Any]] = []
    for name in sorted(catalogue):
        entry = catalogue[name]
        variants = measurements.get(name, {})
        if len(variants) < min_variants:
            below_floor.append({"name": name, "kind": entry["kind"], "n": len(variants)})
            continue
        mutants = tuple(sorted(variants))
        design = entry["kind"] == "design"
        series = f"{entry['group']}/{entry['series']}" if design else "-"
        wildtypes.append(
            WildType(
                name=name,
                kind=entry["kind"],
                series=series,
                cluster=clusters[name],
                zero_hit=(name not in hit_designs) if design else False,
                series_zero_hit=series_zero_hit[series] if design else False,
                sequence=entry["sequence"],
                mutants=mutants,
                phenotype=np.array(
                    [float(np.mean(variants[mutant])) for mutant in mutants],
                    dtype=np.float64,
                ),
                replicates=tuple(len(variants[mutant]) for mutant in mutants),
            )
        )

    provenance = {
        "megascale_dir": str(megascale),
        "certificate_dir": str(certificates),
        "certificate_gate_evalue": certificate["gate"]["evalue"],
        "phenotype": PHENOTYPE_COLUMN,
        "phenotype_orientation": "higher ddG_ML is more stable",
        "sampling": {
            "mode": "census",
            "note": (
                "every wild type the certificate searched and every eligible "
                "variant of it; nothing is sampled, so there is no draw to seed"
            ),
        },
        "min_variants": int(min_variants),
        "rows_read": rows_read,
        "duplicate_measurements_averaged": sum(
            sum(count - 1 for count in wt.replicates) for wt in wildtypes
        ),
        "counts": cohort_counts(wildtypes),
        "below_build_floor": below_floor,
    }
    return Referent(wildtypes=tuple(wildtypes), provenance=provenance)


def cohort_counts(
    wildtypes: Sequence[WildType], *, min_variants: int = 0
) -> dict[str, int]:
    """What the cohort contains, at one variant floor.

    One declaration, so the headline cohort and the floor sweep cannot count
    themselves differently.
    """

    kept = [wt for wt in wildtypes if len(wt.mutants) >= min_variants]
    designs = [wt for wt in kept if wt.kind == "design"]
    zero_hit = [wt for wt in designs if wt.zero_hit]
    naturals = [wt for wt in kept if wt.kind == "natural"]
    return {
        "designs_scored": len(designs),
        "designs_zero_hit_scored": len(zero_hit),
        "designs_hit_excluded_from_a": len(designs) - len(zero_hit),
        "design_series_scored": len({wt.series for wt in zero_hit}),
        "design_series_entirely_zero_hit_scored": len(
            {wt.series for wt in zero_hit if wt.series_zero_hit}
        ),
        "naturals_scored": len(naturals),
        "natural_clusters_scored": len({wt.cluster for wt in naturals}),
        "design_variants": sum(len(wt.mutants) for wt in zero_hit),
        "natural_variants": sum(len(wt.mutants) for wt in naturals),
    }


def save_referent(referent: Referent, path: Path) -> None:
    write_json(Path(path), referent.record())


def load_referent(path: Path) -> Referent:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} carries schema {payload.get('schema_version')!r}, not {SCHEMA_VERSION}"
        )
    return Referent(
        wildtypes=tuple(WildType.from_record(record) for record in payload["wildtypes"]),
        provenance=dict(payload["provenance"]),
    )


# ---------------------------------------------------------------- baselines


def require_corrected_background(background: KmerBackground) -> dict[str, Any]:
    """Refuse anything but the corrected, record-local k-mer count.

    Checked by total rather than by directory name, because the superseded counts
    were retained on disk beside the corrected ones and a path is not evidence of
    what a file holds.
    """

    totals = {k: int(background.counts[k].sum()) for k in background.ks}
    for k, expected in CORRECTED_KMER_TOTALS.items():
        if k not in totals:
            raise ValueError(f"the background carries no k = {k} counts")
        if totals[k] == SUPERSEDED_KMER_TOTALS[k]:
            raise ValueError(
                f"k = {k} totals {totals[k]}, which is the superseded line-local "
                "count that dropped every window spanning the corpus's 60-column "
                "wrapping; use data/kmer_background/uniref50"
            )
        if totals[k] != expected:
            raise ValueError(
                f"k = {k} totals {totals[k]}, not the corrected {expected}"
            )
        observed, possible = background.coverage(k)
        if observed != possible:
            raise ValueError(
                f"k = {k} observes {observed} of {possible} k-mers; a conditional "
                "model over this background would divide by zero"
            )
    return {
        "source": str(background.source),
        "residues": int(background.residues),
        "records": int(background.records),
        "totals": totals,
        "verified": "corrected record-local counts (EXP-R2-190 section 4)",
    }


def conditional_log_probabilities(background: KmerBackground, k: int) -> np.ndarray:
    """``log P(x_t | x_{t-k+1..t-1})`` for every k-mer, from the corpus counts.

    The (k-1)-mer denominator is the k-mer table's own row sum, so the
    conditional is normalised over exactly the windows the background counted and
    no second pass over the corpus is needed. Every k-mer is observed, so no
    smoothing rule is required and none is invented.
    """

    width = len(ALPHABET)
    table = background.counts[k].astype(np.float64).reshape(width ** (k - 1), width)
    prefix = table.sum(axis=1)
    if (table <= 0).any() or (prefix <= 0).any():
        raise ValueError(f"k = {k} carries an unobserved k-mer; the conditional is undefined")
    return (np.log(table) - np.log(prefix)[:, None]).reshape(-1)


def encode_sequences(sequences: Sequence[str]) -> np.ndarray:
    """Equal-length sequences as an ``(n, length)`` array of alphabet codes."""

    if not sequences:
        raise ValueError("no sequences to encode")
    length = len(sequences[0])
    if any(len(sequence) != length for sequence in sequences):
        raise ValueError("every sequence must have the same length")
    raw = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8)
    encoded = _ENCODE[raw]
    if (encoded < 0).any():
        raise ValueError("a sequence carries a residue outside the canonical alphabet")
    return encoded.reshape(len(sequences), length)


def fragment_log_likelihood(
    sequences: Sequence[str], log_conditional: np.ndarray, k: int
) -> np.ndarray:
    """The corpus k-mer model's log-likelihood of each sequence.

    The first ``k - 1`` residues carry no emission term. Every variant of one wild
    type has the same length and the same convention, so the omission is a
    constant offset within the unit the correlation is taken over.
    """

    encoded = encode_sequences(sequences)
    width = len(ALPHABET)
    n, length = encoded.shape
    if length < k:
        raise ValueError(f"a sequence of {length} residues carries no {k}-mer")
    index = np.zeros((n, length - k + 1), dtype=np.int64)
    for offset in range(k):
        index *= width
        index += encoded[:, offset : length - k + 1 + offset]
    return log_conditional[index].sum(axis=1)


# ------------------------------------------- the higher-order fragment channel
#
# What survives from the earlier audit is that ProtGPT2's margin over *corpus statistics*
# survives the exclusion of retrieval, and it rests entirely on the two channels
# above: maximum-likelihood conditionals at k = 3 and k = 4. A 3-mer or 4-mer
# Markov model is a weak model of corpus statistics, so that half is only worth
# what the strongest fragment channel the corpus supports is worth. Everything
# below builds that channel.
#
# Two things have to be right for it to be a fair test rather than a rigged one.
# A higher-order conditional cannot be maximum-likelihood: 20 letters give 3.2 M
# possible 5-mers and 64 M possible 6-mers, unobserved ones exist at those orders
# even in a 17 G-residue corpus, and an unobserved k-mer under maximum likelihood
# is a log-likelihood of minus infinity. So the channel needs a smoothing scheme,
# and the scheme changes its strength. It is *declared* rather than tuned: two
# parameter-free schemes, both reported. And past some order the conditional
# stops being estimated and starts being a lookup of the corpus itself, which
# would make the model look good for the wrong reason; where that happens is a
# measurement (held-out cross-entropy on natural sequence held out of the count)
# and not a matter of taste.

#: The two declared smoothing schemes. Both are parameter-free given the corpus,
#: which is what makes "declared, not tuned" enforceable: neither has a knob that
#: could be moved after a result is seen.
#:
#: ``witten_bell``  interpolate order j with order j-1 at weight
#:                  ``D(c) / (N(c) + D(c))``, ``D(c)`` the number of distinct
#:                  continuations of the context and ``N(c)`` its total count.
#: ``kneser_ney``   interpolated Kneser-Ney with the Chen-Goodman discount
#:                  ``D = n1 / (n1 + 2 n2)`` taken from each table's own
#:                  count-of-counts, and lower orders scored on continuation
#:                  counts rather than raw counts.
FRAGMENT_SMOOTHING: tuple[str, ...] = ("kneser_ney", "witten_bell")


def fragment_channel_name(order: int, scheme: str) -> str:
    """The baseline name an order-``k`` smoothed fragment channel is filed under."""

    if scheme not in FRAGMENT_SMOOTHING:
        raise ValueError(f"{scheme!r} is not one of {FRAGMENT_SMOOTHING}")
    return f"fragment_interp_k{int(order)}_{scheme}"


#: Rows per block in the reductions below. The k = 7 table is 1.28 G cells and
#: 10.24 GB; every reduction over it is written blocked so that the table is read
#: once, contiguously, with a bounded temporary. The column-at-a-time form is the
#: obvious one and costs twenty strided passes for the same answer.
_BLOCK_ROWS: int = 1 << 22


def _row_totals_and_types(table: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-context total count and number of distinct continuations."""

    view = table.reshape(-1, width)
    totals = np.empty(view.shape[0], dtype=np.int64)
    types = np.empty(view.shape[0], dtype=np.int64)
    for start in range(0, view.shape[0], _BLOCK_ROWS):
        block = view[start : start + _BLOCK_ROWS]
        totals[start : start + _BLOCK_ROWS] = block.sum(axis=1)
        types[start : start + _BLOCK_ROWS] = (block > 0).sum(axis=1)
    return totals, types


def _continuation_counts(table: np.ndarray, width: int) -> np.ndarray:
    """``N1+(. w)`` for every ``(j)``-gram ``w``, from the ``(j+1)``-gram table.

    The count vector indexes a k-mer as its base-20 value with the *first* symbol
    most significant, so reshaping to ``(width, -1)`` splits on that first symbol
    and the number of non-zero rows above a suffix is the number of distinct
    symbols that precede it. This is the quantity Kneser-Ney's lower orders are
    defined on, and taking it from the counts already in memory is why no second
    pass over the corpus is needed.
    """

    view = table.reshape(width, -1)
    out = np.zeros(view.shape[1], dtype=np.int64)
    for row in range(width):
        out += view[row] > 0
    return out


def _chen_goodman_discount(table: np.ndarray) -> float:
    """``n1 / (n1 + 2 n2)`` from a table's own count-of-counts.

    Parameter-free by construction. Zero when the table has no singletons, which
    happens only on tables small enough that every cell is heavily occupied (the
    unigram table, and toy corpora in the tests); there the level contributes its
    own maximum-likelihood estimate and no interpolation weight, which is correct
    rather than degenerate because every cell is then non-zero.
    """

    flat = table.reshape(-1)
    ones = 0
    twos = 0
    step = _BLOCK_ROWS * 20
    for start in range(0, flat.size, step):
        block = flat[start : start + step]
        ones += int((block == 1).sum())
        twos += int((block == 2).sum())
    if ones + 2 * twos == 0:
        return 0.0
    return float(ones / (ones + 2 * twos))


@dataclass(frozen=True)
class _Level:
    """One order of the interpolation: its table and the two context statistics."""

    table: np.ndarray
    totals: np.ndarray
    types: np.ndarray
    discount: float


class InterpolatedFragmentModel:
    """An interpolated Markov model over the corpus k-mer counts.

    The order is passed per call rather than fixed at construction, because every
    order shares the same tables and the question this class exists to answer is
    how the channel behaves *as the order rises*. A call at order ``k`` scores
    position ``t`` against the ``min(t, k-1)`` residues before it, so the model is
    a proper normalised distribution over whole sequences and no leading residue
    is dropped. That is a strengthening of :func:`fragment_log_likelihood`, which
    omits the first ``k - 1`` emissions; the two are reported side by side so the
    convention change and the order change are never confounded.
    """

    def __init__(self, counts: Mapping[int, np.ndarray], max_order: int, scheme: str) -> None:
        if scheme not in FRAGMENT_SMOOTHING:
            raise ValueError(f"{scheme!r} is not one of {FRAGMENT_SMOOTHING}")
        if max_order < 1:
            raise ValueError(f"the order must be positive, got {max_order}")
        width = len(ALPHABET)
        missing = [k for k in range(1, max_order + 1) if k not in counts]
        if missing:
            raise ValueError(
                f"an order-{max_order} channel needs count vectors for k = "
                f"{list(range(1, max_order + 1))}; missing {missing}"
            )
        self.width = width
        self.scheme = scheme
        self.max_order = int(max_order)
        self._raw: dict[int, _Level] = {}
        self._continuation: dict[int, _Level] = {}
        for order in range(1, max_order + 1):
            table = np.asarray(counts[order])
            if table.shape != (width**order,):
                raise ValueError(f"k = {order} carries {table.shape}, not {(width**order,)}")
            totals, types = _row_totals_and_types(table, width)
            self._raw[order] = _Level(table, totals, types, _chen_goodman_discount(table))
        if scheme == "kneser_ney":
            for order in range(1, max_order):
                table = _continuation_counts(np.asarray(counts[order + 1]), width)
                totals, types = _row_totals_and_types(table, width)
                self._continuation[order] = _Level(
                    table, totals, types, _chen_goodman_discount(table)
                )

    # -- the recursion ------------------------------------------------------

    def _level(self, order: int, *, top: bool) -> _Level:
        if top or self.scheme == "witten_bell":
            return self._raw[order]
        return self._continuation[order]

    def _step(self, level: _Level, context: np.ndarray, symbol: np.ndarray, lower: np.ndarray) -> np.ndarray:
        totals = level.totals[context].astype(np.float64)
        types = level.types[context].astype(np.float64)
        value = level.table[context * self.width + symbol].astype(np.float64)
        seen = totals > 0.0
        safe = np.where(seen, totals, 1.0)
        if self.scheme == "witten_bell":
            estimate = (value + types * lower) / (safe + types)
        else:
            weight = level.discount * types / safe
            estimate = np.maximum(value - level.discount, 0.0) / safe + weight * lower
        return np.where(seen, estimate, lower)

    def log_probability(self, context: np.ndarray, symbol: np.ndarray, order: int) -> np.ndarray:
        """``log P(symbol | context)`` under an order-``order`` interpolation.

        ``context`` carries the preceding residues as a base-20 value with the
        most recent residue least significant, so the order-``j`` context is
        ``context % width**(j-1)`` and one array serves every level.
        """

        if not 1 <= order <= self.max_order:
            raise ValueError(f"order {order} is outside 1..{self.max_order}")
        probability = np.full(np.shape(symbol), 1.0 / self.width, dtype=np.float64)
        for level_order in range(1, order + 1):
            modulus = self.width ** (level_order - 1)
            reduced = context % modulus if modulus > 1 else np.zeros_like(context)
            probability = self._step(
                self._level(level_order, top=level_order == order),
                reduced,
                symbol,
                probability,
            )
        if not (probability > 0.0).all():
            raise RuntimeError(
                f"the order-{order} {self.scheme} channel assigned zero probability; "
                "an interpolated model that can do that is not usable as a likelihood"
            )
        return np.log(probability)

    # -- scoring ------------------------------------------------------------

    def evaluate(self, sequences: Sequence[str], order: int) -> dict[str, Any]:
        """Score once and report the likelihood and the support together.

        One pass, because the sparsity diagnostics have to sit beside every number
        they qualify: a channel that wins on positions the corpus never saw has
        backed off to a lower order, and a channel that wins on positions the
        corpus saw verbatim is a lookup. Neither is visible in the likelihood.
        """

        stream = _SequenceStream(sequences, order, self.width)
        positional = self._positional(stream, order)
        total = float(positional.sum())
        positions = int(positional.size)
        return {
            "log_likelihood": stream.reduce(positional),
            "cross_entropy_nats": -total / positions,
            "perplexity": float(np.exp(-total / positions)),
            **self._support(stream, order),
        }

    def sequence_log_likelihood(self, sequences: Sequence[str], order: int) -> np.ndarray:
        """Whole-sequence log-likelihood of each sequence, every position scored."""

        stream = _SequenceStream(sequences, order, self.width)
        return stream.reduce(self._positional(stream, order))

    def support(self, sequences: Sequence[str], order: int) -> dict[str, Any]:
        """How much of this evaluation the corpus actually saw, at full order.

        Reported beside every result rather than used as a gate. A channel whose
        contexts are all unseen has backed off to a lower order and is not the
        order it is labelled; a channel whose k-mers are nearly all seen on one
        side and nearly none on the other is measuring two different things on
        the two sides.
        """

        return self._support(_SequenceStream(sequences, order, self.width), order)

    def _support(self, stream: "_SequenceStream", order: int) -> dict[str, Any]:
        at_order = stream.top == order
        context = (
            stream.context[at_order]
            if order > 1
            else np.zeros(int(at_order.sum()), dtype=np.int64)
        )
        level = self._raw[order]
        unseen_context = int((level.totals[context] == 0).sum())
        index = context * self.width + stream.symbol[at_order]
        unseen_kmer = int((level.table[index] == 0).sum())
        positions = int(at_order.sum())
        return {
            "positions": int(stream.symbol.size),
            "positions_at_full_order": positions,
            "unseen_context_positions": unseen_context,
            "unseen_kmer_positions": unseen_kmer,
            "unseen_context_fraction": (unseen_context / positions) if positions else None,
            "unseen_kmer_fraction": (unseen_kmer / positions) if positions else None,
        }

    def cross_entropy(self, sequences: Sequence[str], order: int) -> dict[str, Any]:
        """Per-residue cross-entropy in nats, over every scored position."""

        record = self.evaluate(sequences, order)
        record.pop("log_likelihood")
        return record

    def _positional(self, stream: "_SequenceStream", order: int) -> np.ndarray:
        values = np.empty(stream.symbol.size, dtype=np.float64)
        for top in range(1, order + 1):
            selected = stream.top == top
            if not selected.any():
                continue
            values[selected] = self.log_probability(
                stream.context[selected], stream.symbol[selected], top
            )
        return values


class _SequenceStream:
    """Every scored position of a list of sequences, as flat arrays.

    Sequences differ in length and a context may not reach across a boundary, so
    the position of each residue within its own sequence is carried and every lag
    that would reach past a sequence start contributes nothing. Building this
    once and evaluating it grouped by available context length is what keeps the
    corpus held-out pass -- millions of positions over variable-length records --
    a vectorised operation.
    """

    def __init__(self, sequences: Sequence[str], order: int, width: int) -> None:
        if not sequences:
            raise ValueError("no sequences to score")
        lengths = np.array([len(sequence) for sequence in sequences], dtype=np.int64)
        if (lengths < 1).any():
            raise ValueError("an empty sequence carries no scored position")
        raw = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8)
        symbol = _ENCODE[raw]
        if (symbol < 0).any():
            raise ValueError("a sequence carries a residue outside the canonical alphabet")
        self.lengths = lengths
        self.starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        self.symbol = symbol
        within = np.arange(symbol.size, dtype=np.int64) - np.repeat(self.starts, lengths)
        context = np.zeros(symbol.size, dtype=np.int64)
        for lag in range(1, order):
            shifted = np.zeros(symbol.size, dtype=np.int64)
            shifted[lag:] = symbol[:-lag]
            context += np.where(within >= lag, shifted, 0) * (width ** (lag - 1))
        self.context = context
        self.top = np.minimum(within + 1, order)

    def reduce(self, values: np.ndarray) -> np.ndarray:
        return np.add.reduceat(values, self.starts)


def corpus_residue_background(path: Path | None = None) -> np.ndarray:
    """The corpus residue frequencies the composition baseline is defined on."""

    root = Path(path) if path is not None else RETRIEVAL_BOUND_DIR
    payload = json.loads((root / "wildtypes.json").read_text(encoding="utf-8"))
    background = payload["corpus"]["background"]
    vector = np.array([float(background[residue]) for residue in P.AA20], dtype=np.float64)
    if not np.isfinite(vector).all() or (vector <= 0).any():
        raise ValueError(f"{root}/wildtypes.json carries an unusable residue background")
    return vector / vector.sum()


def baseline_scores(
    wildtype: WildType,
    *,
    residue_background: np.ndarray,
    log_conditional: Mapping[int, np.ndarray],
) -> dict[str, np.ndarray]:
    """Every baseline's prediction for every variant of one wild type."""

    substitutions = wildtype.substitutions
    scores: dict[str, np.ndarray] = {
        "blosum62": np.array(
            [
                sum(BLOSUM62[(wild, mutated)] for wild, _, mutated in entry)
                for entry in substitutions
            ],
            dtype=np.float64,
        )
    }
    scores.update(
        P.free_baselines(
            substitutions, residue_background, wildtype_length=len(wildtype.sequence)
        )
    )
    sequences = wildtype.sequences()
    for k, table in sorted(log_conditional.items()):
        scores[f"fragment_markov_k{k}"] = fragment_log_likelihood(sequences, table, k)
    missing = set(BASELINES) - set(scores)
    if missing:
        raise ValueError(f"{wildtype.name}: baselines {sorted(missing)} were not computed")
    return {name: scores[name] for name in BASELINES}


# --------------------------------------------------------------- statistics


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Rank correlation, or ``None`` when one side is constant.

    Returned rather than raised: a channel that cannot be ranked on one wild type
    is a fact about that wild type, and dropping it silently or writing a NaN
    into the artefact are the two ways this goes wrong later.
    """

    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.size < 3 or left.size != right.size:
        return None
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        return None
    # Checked before the call rather than after: scipy warns and returns NaN on a
    # constant input, and a warning is not a value a stage can branch on.
    if left.min() == left.max() or right.min() == right.max():
        return None
    value = float(stats.spearmanr(left, right).statistic)
    return None if not np.isfinite(value) else value


def unit_bootstrap(
    values: Sequence[float], units: Sequence[str], *, resamples: int, seed: int
) -> dict[str, Any]:
    """Percentile interval on the unit-mean average, resampling units.

    ``profiles.cluster_bootstrap`` performs the resample and carries
    ``statistics.bootstrap_unit_floor``'s refusal record; this wrapper only turns
    the unit names into the integer labels it expects, so the floor, the
    replicate count and the degeneracy record all have one declaration.
    """

    order = {name: position for position, name in enumerate(sorted(set(units)))}
    record = P.cluster_bootstrap(
        values,
        [order[name] for name in units],
        resamples=resamples,
        seed=seed,
    )
    record["unit"] = "design series / natural WT_cluster"
    record["unit_names"] = sorted(order)
    return record


def channel_comparison(
    model: Mapping[str, float],
    baseline: Mapping[str, float],
    units: Mapping[str, str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int,
) -> dict[str, Any]:
    """MODEL - BASELINE over the wild types both channels could be read on."""

    shared = sorted(set(model) & set(baseline))
    differences = [model[name] - baseline[name] for name in shared]
    record = unit_bootstrap(differences, [units[name] for name in shared], resamples=resamples, seed=seed)
    record["n_wildtypes"] = len(shared)
    record["n_wildtypes_dropped"] = len(set(model) | set(baseline)) - len(shared)
    record["beats_baseline"] = bool(
        record["interval"] is not None and record["interval"][0] > 0.0
    )
    return record


#: What EXP-R2-189 section 6 read off the retrieval-bound artefacts for the 64 ProteinGym
#: Tsuboyama assays. Quoted here so the cross-check compares against a recorded
#: number rather than against a memory of one.
PROTEINGYM_TSUBOYAMA_REFERENCE: Mapping[str, float] = {
    "protgpt2": 0.367,
    "progen2-medium": 0.365,
}


def design_length_bands(
    designs: Sequence[WildType],
) -> tuple[tuple[int, int], ...]:
    """Length bands for a post-hoc length-matched control, taken from the designs.

    Derived from the design lengths' own quantiles rather than chosen, and
    reported as a sweep, because the question a length-matched control answers is
    whether the design-side reading is about *designed* sequence or about *short*
    sequence, and a single hand-picked band would answer it at one width only.
    """

    lengths = np.array([len(wt.sequence) for wt in designs], dtype=np.int64)
    if lengths.size == 0:
        raise ValueError("no designs to take a length band from")
    low = int(lengths.min())
    return tuple(
        (low, int(np.percentile(lengths, q))) for q in (50.0, 75.0, 100.0)
    )


def arm_verdict(design: Mapping[str, bool], control: Mapping[str, bool]) -> dict[str, Any]:
    """The pre-registered rule, applied.

    A positive needs the arm to beat **every** free and fragment-level baseline on
    the corpus-disjoint designs *and* the same conjunction to be attainable on the
    natural control run under the identical procedure. A null on the designs with
    the control passing is a clean negative. A null on the designs with the
    control **failing** says nothing about the model: the instrument could not
    clear its own bar where retrieval is available, so the design-side null is an
    instrument bound and is reported as one.
    """

    missing = (set(BASELINES) - set(design)) | (set(BASELINES) - set(control))
    if missing:
        raise ValueError(f"the verdict needs every baseline; missing {sorted(missing)}")
    attainable = all(control[name] for name in BASELINES)
    beats_all = all(design[name] for name in BASELINES)
    if not attainable:
        verdict = "uninterpretable_instrument_bound"
    elif beats_all:
        verdict = "positive"
    else:
        verdict = "negative"
    return {
        "verdict": verdict,
        "attainable_on_control": attainable,
        "beats_every_baseline_on_designs": beats_all,
        "control_failures": sorted(name for name in BASELINES if not control[name]),
        "design_failures": sorted(name for name in BASELINES if not design[name]),
    }


# --------------------------------------------------- the design/natural interaction
#
# EXP-R2-192 reported MODEL - hydropathy_change on the designs and, in a control
# computed after the pinned scoring, on the natural domains inside the designs'
# own length span. The two have opposite signs. Two contrasts reported side by
# side are not an interaction: the quantity that carries the claim is their
# **difference**, with one joint interval, and until EXP-R2-193 nothing here
# computed it. Everything below is that statistic and the axes it is read on.

#: Which design family a series belongs to. EXP-R2-190 enumerated the series as
#: topology x round, run-family x run, and hallucination round; the three
#: families are what a held-out reading is split on. Unknown groups raise rather
#: than falling into a default, because a family that silently absorbs an
#: unrecognised series would change the unit counts a floor is checked against.
DESIGN_FAMILY_OF_GROUP: Mapping[str, str] = {
    "EEHEE": "topology",
    "EHEE": "topology",
    "HEEH": "topology",
    "HEEH_KT": "topology",
    "HHH": "topology",
    "EA": "run_keyed",
    "GG": "run_keyed",
    "XX": "run_keyed",
    "TrROS_Hall": "hallucination",
}

DESIGN_FAMILIES: tuple[str, ...] = ("topology", "run_keyed", "hallucination")

#: Residue padding applied to a design subset's own length span when the natural
#: side is matched to it. ``0`` is primary; the sweep is the invariance check
#: Appendix B rule 8 requires wherever a threshold cannot be avoided.
LENGTH_PADS: tuple[int, ...] = (0, 2, 5)

#: The magnitude an interaction must reach to count as confirming. Roughly a
#: third of the -0.47 EXP-R2-192's two existing ProtGPT2 numbers imply, fixed in
#: the pre-registration before any interval existed.
INTERACTION_MAGNITUDE = 0.15

#: The channel the interaction is declared on. The other six are computed too, as
#: the pre-registered specificity control, but only this one carries the verdict.
INTERACTION_CHANNEL = "hydropathy_change"

#: The arm the observation was made on. Its pooled reading is where the effect was
#: first seen, so it is *not* a confirmatory axis; the families and the other arms
#: are. Named rather than spelled at each use so the two cannot drift apart.
ORIGIN_ARM = "protgpt2"


def design_family(series: str) -> str:
    """The family of one design series, e.g. ``EEHEE/rd3`` -> ``topology``."""

    group = series.split("/", 1)[0]
    try:
        return DESIGN_FAMILY_OF_GROUP[group]
    except KeyError:
        raise ValueError(
            f"{series!r} belongs to no declared design family; add it to "
            "DESIGN_FAMILY_OF_GROUP rather than letting it fall into a default"
        ) from None


def _unit_means(values: Sequence[float], units: Sequence[str]) -> np.ndarray:
    """Per-unit averages, through ``profiles.cluster_means``."""

    order = {name: position for position, name in enumerate(sorted(set(units)))}
    means, _ = P.cluster_means(values, [order[name] for name in units])
    return means


def unit_mean_average(values: Sequence[float], units: Sequence[str]) -> float:
    """Average within unit, then across units -- the quantity the interaction differences.

    Exposed so that a caller reporting the four halves of an interaction reports
    them on the same construction the interaction itself is built from.
    """

    return float(_unit_means(values, units).mean())


def interaction_bootstrap(
    design_values: Sequence[float],
    design_units: Sequence[str],
    natural_values: Sequence[float],
    natural_units: Sequence[str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """The design-minus-natural difference of unit-mean averages, with its interval.

    The resample is **stratified and joint**: one draw takes the design units with
    replacement and the natural units with replacement and recomputes the
    difference inside the draw. An interval built from two separately resampled
    means is not an interval on their difference, and reporting the two contrasts
    beside each other -- which is what EXP-R2-192 did -- is not reporting the
    interaction at all.

    The floor is :func:`~.statistics.bootstrap_unit_floor` applied to **each side
    separately**, because a difference is no better bounded than its worse-bounded
    half. Refused rather than raised: too few units on one side is a fact about
    that stratum and belongs in the artefact beside the point estimate.
    """

    if resamples < 1 or not 0 < alpha < 1:
        raise ValueError("invalid bootstrap parameters")
    left = _unit_means(design_values, design_units)
    right = _unit_means(natural_values, natural_units)
    design_floor = bootstrap_unit_floor(int(left.size))
    natural_floor = bootstrap_unit_floor(int(right.size))
    degenerate = bool(design_floor["degenerate"] or natural_floor["degenerate"])
    record: dict[str, Any] = {
        "point": float(left.mean() - right.mean()),
        "design_side": float(left.mean()),
        "natural_side": float(right.mean()),
        "n_design_units": int(left.size),
        "n_natural_units": int(right.size),
        "n_design_wildtypes": int(np.asarray(design_values).size),
        "n_natural_wildtypes": int(np.asarray(natural_values).size),
        "resamples": int(resamples),
        "alpha": float(alpha),
        "degenerate": degenerate,
        "degenerate_reason": design_floor["degenerate_reason"]
        or natural_floor["degenerate_reason"],
        "minimum_units": int(design_floor["minimum_units"]),
        "interval": None,
        "excludes_zero": None,
    }
    if degenerate:
        return record
    rng = np.random.default_rng(seed)
    design_draws = rng.integers(0, left.size, size=(resamples, left.size))
    natural_draws = rng.integers(0, right.size, size=(resamples, right.size))
    statistic = left[design_draws].mean(axis=1) - right[natural_draws].mean(axis=1)
    low, high = np.percentile(statistic, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    record["interval"] = [float(low), float(high)]
    record["excludes_zero"] = bool(low > 0.0 or high < 0.0)
    return record


def interaction_outcome(
    record: Mapping[str, Any], *, magnitude: float = INTERACTION_MAGNITUDE
) -> str:
    """The pre-registered four-way reading of one interaction axis.

    ``confirms``    the interval lies wholly below zero and the point estimate
                    reaches the declared magnitude.
    ``attenuated``  the interval lies wholly below zero but the point estimate
                    does not: a real interaction, smaller than declared.
    ``refutes``     the interval does not exclude zero downward AND excludes an
                    interaction of the declared magnitude.
    ``unresolved``  everything else, including every floor refusal. This is
                    underpower and must never be read as absence.
    """

    if magnitude <= 0:
        raise ValueError("the confirming magnitude must be positive")
    interval = record.get("interval")
    if interval is None:
        return "unresolved"
    low, high = float(interval[0]), float(interval[1])
    if high < 0.0:
        return "confirms" if float(record["point"]) <= -magnitude else "attenuated"
    return "refutes" if low > -magnitude else "unresolved"


def _weighted_least_squares(
    design: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> np.ndarray | None:
    root = np.sqrt(weights)[:, None]
    solution, _, rank, _ = np.linalg.lstsq(design * root, target * np.sqrt(weights), rcond=None)
    return None if rank < design.shape[1] else solution


def adjusted_interaction_bootstrap(
    values: Sequence[float],
    units: Sequence[str],
    designed: Sequence[bool],
    covariates: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """The interaction adjusted for covariates, on the same units and the same draw.

    A length *band* is a restriction, not a match: the natural domains inside the
    designs' span are still longer on average than the designs are. This is the
    complement that needs no band -- the coefficient on ``designed`` in a
    wild-type-level weighted least squares -- and it is deliberately weighted so
    that with **no** covariates it reduces exactly to
    :func:`interaction_bootstrap`'s point estimate: each wild type carries weight
    ``1 / (wild types in its unit)``, so every unit contributes weight one and the
    two estimators cannot silently disagree about what a unit is.

    Its own weakness is the one a restriction does not have: a linear adjustment
    extrapolates where the two sides barely overlap. It complements the restricted
    readings rather than replacing them.
    """

    if resamples < 1 or not 0 < alpha < 1:
        raise ValueError("invalid bootstrap parameters")
    target = np.asarray(values, dtype=np.float64)
    flag = np.asarray(designed, dtype=bool)
    extra = np.asarray(covariates, dtype=np.float64).reshape(target.size, -1)
    labels = np.asarray(units)
    if flag.shape != target.shape or labels.shape != target.shape:
        raise ValueError("values, units and the designed flag must be aligned")
    if not np.isfinite(target).all() or not np.isfinite(extra).all():
        raise ValueError("the adjusted interaction requires finite inputs")

    unit_names, unit_index = np.unique(labels, return_inverse=True)
    sizes = np.bincount(unit_index, minlength=unit_names.size)
    weights = 1.0 / sizes[unit_index]
    if extra.shape[1]:
        centred = extra - extra.mean(axis=0)
        scale = centred.std(axis=0)
        scale[scale == 0.0] = 1.0
        extra = centred / scale
    matrix = np.column_stack([np.ones(target.size), flag.astype(np.float64), extra])

    # Which side each unit sits on. A unit that mixed designs and naturals would
    # make the stratified draw ill-defined, so it is refused rather than assigned.
    unit_is_design = np.zeros(unit_names.size, dtype=bool)
    for position in range(unit_names.size):
        rows = flag[unit_index == position]
        if rows.any() and not rows.all():
            raise ValueError(f"unit {unit_names[position]!r} mixes designs and naturals")
        unit_is_design[position] = bool(rows.all())
    design_units = np.flatnonzero(unit_is_design)
    natural_units = np.flatnonzero(~unit_is_design)
    rows_of_unit = [np.flatnonzero(unit_index == position) for position in range(unit_names.size)]

    solution = _weighted_least_squares(matrix, target, weights)
    design_floor = bootstrap_unit_floor(int(design_units.size))
    natural_floor = bootstrap_unit_floor(int(natural_units.size))
    degenerate = bool(design_floor["degenerate"] or natural_floor["degenerate"])
    record: dict[str, Any] = {
        "point": None if solution is None else float(solution[1]),
        "n_design_units": int(design_units.size),
        "n_natural_units": int(natural_units.size),
        "n_wildtypes": int(target.size),
        "n_covariates": int(extra.shape[1]),
        "resamples": int(resamples),
        "alpha": float(alpha),
        "degenerate": degenerate,
        "degenerate_reason": design_floor["degenerate_reason"]
        or natural_floor["degenerate_reason"],
        "interval": None,
        "excludes_zero": None,
        "rank_deficient_resamples": 0,
    }
    if degenerate or solution is None:
        return record

    rng = np.random.default_rng(seed)
    design_draws = rng.integers(0, design_units.size, size=(resamples, design_units.size))
    natural_draws = rng.integers(0, natural_units.size, size=(resamples, natural_units.size))
    statistic: list[float] = []
    refused = 0
    for replicate in range(resamples):
        drawn = np.concatenate(
            [design_units[design_draws[replicate]], natural_units[natural_draws[replicate]]]
        )
        rows = np.concatenate([rows_of_unit[position] for position in drawn])
        fit = _weighted_least_squares(matrix[rows], target[rows], weights[rows])
        if fit is None:
            refused += 1
            continue
        statistic.append(float(fit[1]))
    record["rank_deficient_resamples"] = int(refused)
    if len(statistic) < resamples // 2:
        record["degenerate"] = True
        record["degenerate_reason"] = (
            f"{refused} of {resamples} resamples were rank deficient; the adjusted "
            "coefficient is not identified on this subcohort"
        )
        return record
    low, high = np.percentile(statistic, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    record["interval"] = [float(low), float(high)]
    record["excludes_zero"] = bool(low > 0.0 or high < 0.0)
    return record


def standardised_mean_differences(
    design: np.ndarray, natural: np.ndarray, names: Sequence[str]
) -> dict[str, Any]:
    """Covariate balance between the two sides, as SMDs and their maximum.

    Restricting a cohort to a propensity common support removes the units that
    could never be matched; it does **not** make the survivors balanced. Reported
    rather than assumed, so a composition control that failed to balance cannot be
    read as one that succeeded.
    """

    left = np.asarray(design, dtype=np.float64).reshape(-1, len(names))
    right = np.asarray(natural, dtype=np.float64).reshape(-1, len(names))
    pooled = np.sqrt((left.var(axis=0, ddof=1) + right.var(axis=0, ddof=1)) / 2.0)
    pooled[pooled == 0.0] = np.nan
    smd = (left.mean(axis=0) - right.mean(axis=0)) / pooled
    values = {name: (None if not np.isfinite(v) else float(v)) for name, v in zip(names, smd)}
    finite = [abs(v) for v in values.values() if v is not None]
    return {
        "smd": values,
        "max_abs_smd": float(max(finite)) if finite else None,
        "balanced": bool(finite and max(finite) <= 0.25),
        "threshold": 0.25,
    }
