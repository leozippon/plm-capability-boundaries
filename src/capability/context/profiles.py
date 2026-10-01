"""MODEL - LOOKUP: what a protein decoder adds to a lookup of its own corpus.

**The subtraction this module supplies, and why the existing one is not enough.**
EXP-R2-134 measured MODEL - FREE over all 217 ProteinGym substitution assays:
ProGen3-112M's zero-shot fitness beats a BLOSUM62 substitution-matrix lookup by
+0.0647 Spearman [+0.0386, +0.0909]. That is the whole of this programme's
evidence that a protein decoder predicts function at all, and it cannot
separate the two accounts that matter. BLOSUM62 is free of the corpus, so a model
that had merely *stored* its pretraining data would beat it exactly as a model
that had *learned* anything would. The missing comparison is MODEL - LOOKUP,
where LOOKUP is a site-independent position-specific profile built by aligning
each assay's wild type against the arm's own pretraining corpus. Whatever a
column-frequency lookup of the corpus already supplies is retrieved, not acquired;
what the model adds on top of it is the part that needs an explanation.

The estimand is measured against **wet-lab phenotype** -- ProteinGym's DMS scores
-- rather than against a predicted structure or a sequence-inferred label, which
is the property the audit's survey says the second half lacks an
instrument for.

**Nothing here is fitted and no DMS label touches the channel.** The profile is a
weighted column frequency of DIAMOND hits; the pseudocount weight ``alpha`` is
declared in :data:`PSEUDOCOUNT_ALPHA` before any run and swept afterwards. In
particular the channel does not route through a trained probe: audit section 7
rejects probe-derived directions on arrival, and a probe fitted on DMS labels
would make the "lookup" arm a supervised predictor and the subtraction
meaningless.

**Three hazards this module is built around.**

*Repeat masking.* DIAMOND's default query masking truncated the alignments of
records that were byte-identical corpus members and pushed them out of the
near-duplicate stratum (audit 0.05, EXP-R2-061); every error ran in the direction
that defeated the hypothesis under test. :func:`~.homology.run_diamond_blastp`
now passes ``--masking 0`` and :func:`~.homology.assign_homology` refuses a
truncated alignment. This module adds the measured anchor those two lacked:
:func:`verbatim_anchor_check` reads the corpus directly for byte-identical wild
types and refuses the run unless every one of them lands in the top identity
stratum. On the 2026-08-07 corpus that anchor set is 73 of 187 wild types in
UniRef50 and 78 in Swiss-Prot union UniRef50, so it is a real constraint rather
than a formality.

*The unit of independence.* 217 assays carry 187 distinct wild types and many of
those are homologous; L1's cohort made the analogous error one level down, by
treating millions of mutant rows over one wild type as independent units. Wild
types are clustered at 50% identity (:func:`cluster_by_identity`), assay-level
differences are averaged within cluster, and every interval resamples clusters
under the floor in :data:`~.statistics.MINIMUM_BOOTSTRAP_UNITS`.

*A verbatim-presence split cannot be the axis.* 78 of 187 wild types are present
byte-identically, so "in the corpus" against "not in the corpus" is a 39/61 cut
that answers a different, cruder question. The stratifier is graded instead: max
identity to the corpus, and log10 Neff of homologues, both declared in residues,
both reported over a sweep of bin edges with the ordering required invariant
(Appendix B rule 17) and beside a threshold-free Kendall tau.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from ..core.amino_acids import AA20 as AA20
from .homology import Hit
from ..core.statistics import MINIMUM_BOOTSTRAP_UNITS, bootstrap_unit_floor, make_group_splits

SCHEMA_VERSION = "r2_transfer_retrieval_bound_v1"

#: Column code for "this corpus sequence has no residue aligned to this query
#: position". One past the alphabet, so a gap can never compare equal to a
#: residue and never contributes to a column's weight.
GAP_CODE = len(AA20)

# --------------------------------------------------------------- the channel

#: Pseudocount mass on the corpus background, declared before the run.
#:
#: The score is ``sum_p log[f(m,p) + alpha*b(m)] - log[f(w,p) + alpha*b(w)]``
#: with ``f`` a column frequency normalised over the column's own weight, so the
#: per-column normaliser cancels between the two terms and ``alpha`` is
#: interpretable directly: 0.05 gives the background one twentieth of the weight
#: a fully supported column carries. It is small enough that a well-supported
#: column dominates and large enough that an unobserved residue is finite rather
#: than minus infinity -- which is the whole reason a pseudocount is here, since
#: most columns of a real alignment observe two or three residues out of twenty.
#: Swept afterwards over :data:`ALPHA_SWEEP`; the headline is the declared value.
PSEUDOCOUNT_ALPHA = 0.05

#: Reported beside the headline so that no reader has to take the declared value
#: on trust. Appendix B rule 17 asks for a sweep wherever a constant is
#: unavoidable, and the ordering of MODEL against LOOKUP has to survive it.
ALPHA_SWEEP: tuple[float, ...] = (0.01, 0.05, 0.20)

#: Percent of the **query's residues** an alignment must cover before its subject
#: may contribute a column. Declared in residues rather than in HSP columns for
#: the reason ``homology.Hit.identity_over_query`` records: a corpus entry that
#: aligns perfectly to 40% of the query is not a homologue of the query.
PROFILE_COVERAGE_FLOOR = 80.0

#: Percent identity over the query at or above which a hit counts toward Neff.
NEFF_IDENTITY_FLOOR = 30.0

#: Percent identity at or above which two aligned corpus sequences are one
#: effective sequence. The classical 80% reweighting: each sequence's weight is
#: one over the number of sequences within this identity of it, so a family that
#: is a thousand near-copies of one protein contributes about one unit of
#: evidence rather than a thousand.
REWEIGHT_IDENTITY_FLOOR = 80.0

#: The 50%-identity / 80%-coverage rule wild types are clustered under. It is
#: UniRef50's own clustering definition, which is what makes "one cluster" the
#: same object as "one corpus cluster" rather than a threshold invented here.
FAMILY_IDENTITY = 50.0
FAMILY_COVERAGE = 80.0

#: Kyte-Doolittle hydropathy, for the free baselines. A fixed published scale, so
#: a predictor built on it is computable from the mutation string alone and is
#: free of the model, of the corpus and of the labels (Appendix B rule 28).
KYTE_DOOLITTLE: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# ------------------------------------------------------- pre-registered gates

#: What EXP-R2-134 measured for BLOSUM62 over the same 217 assays, frozen. The
#: positive control is declared against it rather than against a published
#: ProteinGym baseline: no ProteinGym reference table is staged on this host and
#: the workstation has no route to the model hub, so an external number could not
#: be verified before the run and must not gate one. This number was measured in
#: this repository, on this cohort, by this repository's own scoring code.
FROZEN_BLOSUM62_MEAN_SPEARMAN = 0.2098

#: EXP-R2-134's paired MODEL - BLOSUM62 advantage for ProGen3-112M, frozen. Used
#: only to give the absolute equivalence bound a scale a reader can check; the
#: operative bound is per arm and is derived from that arm's own measurement.
FROZEN_MODEL_MINUS_BLOSUM = 0.0647

#: Fraction of an arm's own measured MODEL - BLOSUM62 advantage inside which a
#: MODEL - LOOKUP interval has to fall before the null may be read as a positive
#: statement. At one half, containment says that at least half of everything the
#: model has over a free baseline is also available from a site-independent
#: lookup of its corpus.
EQUIVALENCE_FRACTION = 0.5

#: Bin-edge partitions of maximum percent identity to the corpus. The first is
#: ``homology.STRATUM_EDGES`` restated as a partition of the same axis, so this
#: stage's coarsest cut is the one the induction control already publishes.
IDENTITY_EDGE_SWEEP: tuple[tuple[float, ...], ...] = (
    (0.0, 95.0, 100.000001),
    (0.0, 30.0, 70.0, 95.0, 100.000001),
    (0.0, 50.0, 80.0, 95.0, 99.0, 100.000001),
    (0.0, 40.0, 60.0, 80.0, 90.0, 99.0, 100.000001),
)

#: Bin-edge partitions of log10 Neff.
NEFF_EDGE_SWEEP: tuple[tuple[float, ...], ...] = (
    (0.0, 1.0, 10.0),
    (0.0, 0.5, 1.5, 10.0),
    (0.0, 0.5, 1.0, 1.5, 2.0, 10.0),
)

#: How far a mismatched-profile donor may sit from its recipient before the
#: match is reported as not achieved. The donor is chosen by a declared distance
#: rather than filtered by these, so a wild type always has a control and the
#: quality of the match is a reported quantity rather than a silent exclusion.
MISMATCH_LENGTH_TOLERANCE = 0.20
MISMATCH_LOG10_NEFF_TOLERANCE = 0.3


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


# ------------------------------------------------------------- the corpus scan


@dataclass(frozen=True)
class CorpusScan:
    """One pass over the searched corpus: background, size, and verbatim members.

    The verbatim set is the anchor. DIAMOND's identity is an *inference* about a
    query's relation to the corpus; whether the query's exact bytes appear in the
    corpus is a fact, and reading it directly is the only way to check the
    aligner rather than trust it. That check is what EXP-R2-061 did not have.
    """

    fasta: Path
    records: int
    residues: int
    counts: dict[str, int]
    verbatim: frozenset[str]

    def __post_init__(self) -> None:
        if self.records < 1:
            raise ValueError("a corpus scan must see at least one record")
        if sum(self.counts.values()) < 1:
            raise ValueError("a corpus scan must see at least one alphabet residue")

    @property
    def background(self) -> dict[str, float]:
        """Corpus-wide residue frequencies over :data:`AA20`, summing to one."""

        total = sum(self.counts.values())
        return {residue: self.counts[residue] / total for residue in AA20}

    def background_vector(self) -> np.ndarray:
        background = self.background
        return np.array([background[residue] for residue in AA20], dtype=np.float64)

    def record(self) -> dict[str, Any]:
        return {
            "fasta": str(self.fasta),
            "records": int(self.records),
            "alphabet_residues": int(sum(self.counts.values())),
            "non_alphabet_residues": int(self.residues - sum(self.counts.values())),
            "background": {r: _finite(v, f"background {r}") for r, v in self.background.items()},
            "verbatim_targets": len(self.verbatim),
        }


def scan_corpus(
    fasta: Path, targets: Iterable[str], *, chunk: int = 1 << 26
) -> CorpusScan:
    """Background frequencies, record count and verbatim membership, in one pass.

    The corpus is tens of gigabytes and is read three times over by the rest of
    this stage's dependencies already (``makedb``, the search, the count check),
    so the two things this module needs from it are taken together. Records are
    split on ``\\n>`` and unwrapped with one C-level ``replace`` each, which is
    what keeps a 24 GB pass inside a minute.

    A target is *verbatim* when its exact residue bytes are some record's exact
    residue bytes. Not "highly similar", not "the same protein": the anchor has
    to be a fact the aligner cannot argue with.
    """

    fasta = Path(fasta)
    if not fasta.is_file():
        raise FileNotFoundError(f"{fasta} does not exist")
    wanted = {sequence.encode(): sequence for sequence in targets}
    lengths = {len(key) for key in wanted}
    codes = np.zeros(256, dtype=np.int64)
    alphabet = np.frombuffer(AA20.encode(), dtype=np.uint8)
    found: set[str] = set()
    records = 0
    residues = 0

    def consume(block: bytes) -> None:
        nonlocal records, residues
        index = block.find(b"\n")
        if index < 0:
            return
        sequence = block[index + 1 :].replace(b"\n", b"").replace(b"\r", b"")
        records += 1
        residues += len(sequence)
        if len(sequence) in lengths and sequence in wanted:
            found.add(wanted[sequence])
        pending.append(sequence)

    def flush() -> None:
        if not pending:
            return
        joined = b"".join(pending)
        pending.clear()
        codes[:] += np.bincount(
            np.frombuffer(joined, dtype=np.uint8), minlength=256
        ).astype(np.int64)

    pending: list[bytes] = []
    with fasta.open("rb") as handle:
        tail = b""
        while True:
            block = handle.read(chunk)
            if not block:
                break
            parts = (tail + block).split(b"\n>")
            tail = parts.pop()
            for part in parts:
                consume(part)
            flush()
        if tail:
            consume(tail)
            flush()
    if records < 1:
        raise RuntimeError(f"{fasta} contains no FASTA records")
    counts = {residue: int(codes[code]) for residue, code in zip(AA20, alphabet)}
    return CorpusScan(
        fasta=fasta,
        records=records,
        residues=residues,
        counts=counts,
        verbatim=frozenset(found),
    )


# ----------------------------------------------------------------- the profile


def _encode(sequence: str) -> np.ndarray:
    table = np.full(256, GAP_CODE, dtype=np.uint8)
    for index, residue in enumerate(AA20):
        table[ord(residue)] = index
    return table[np.frombuffer(sequence.encode(), dtype=np.uint8)]


def sequence_weights(
    rows: np.ndarray, *, identity: float = REWEIGHT_IDENTITY_FLOOR, budget: int = 1 << 26
) -> np.ndarray:
    """One over the number of aligned sequences within ``identity`` of each row.

    Identity between two aligned corpus sequences is counted over the query
    columns where **both** carry a residue, which is the only definition that
    does not make two sequences look identical because they are both absent.

    Computed in row blocks sized by ``budget`` so that a 3423-residue query with
    thousands of homologues does not allocate a comparison tensor the host cannot
    hold. The block size changes the arithmetic order and nothing else.
    """

    if rows.ndim != 2 or rows.shape[0] < 1:
        raise ValueError("sequence weights need a non-empty (sequences, columns) matrix")
    if not 0.0 < identity <= 100.0:
        raise ValueError("the reweighting identity must lie in (0, 100]")
    n_rows, n_columns = rows.shape
    present = rows != GAP_CODE
    block = max(1, int(budget // max(1, n_rows * n_columns)))
    neighbours = np.zeros(n_rows, dtype=np.float64)
    for start in range(0, n_rows, block):
        stop = min(start + block, n_rows)
        both = present[start:stop, None, :] & present[None, :, :]
        same = (rows[start:stop, None, :] == rows[None, :, :]) & both
        denominator = both.sum(axis=-1)
        numerator = same.sum(axis=-1)
        fraction = np.where(
            denominator > 0, numerator / np.maximum(denominator, 1), 0.0
        )
        neighbours[start:stop] = (fraction >= identity / 100.0).sum(axis=1)
    # A row is always within identity of itself, so the count is at least one and
    # the reciprocal is defined. Asserted rather than clamped: a zero would mean
    # the self-comparison failed, which is a defect and not a data property.
    if not np.all(neighbours >= 1):
        raise RuntimeError("a sequence is not within the reweighting identity of itself")
    return 1.0 / neighbours


@dataclass(frozen=True)
class Profile:
    """A site-independent position-specific profile of one wild type.

    ``frequencies`` is ``(length, 20)``, each row normalised over its own column
    weight, so a column supported by three effective sequences and one supported
    by three hundred are on the same scale and the pseudocount means the same
    thing at both. ``column_weight`` carries the support that normalisation
    divides out, because a frequency of 1.0 from one effective sequence and from
    three hundred are not the same evidence and the artefact must show which.
    """

    query_id: str
    wildtype: str
    n_hits: int
    n_sequences: int
    saturated: bool
    frequencies: np.ndarray
    column_weight: np.ndarray
    neff: float
    max_identity_over_query: float

    def __post_init__(self) -> None:
        length = len(self.wildtype)
        if self.frequencies.shape != (length, len(AA20)):
            raise ValueError("profile frequencies are not (length, 20)")
        if self.column_weight.shape != (length,):
            raise ValueError("profile column weights are not one per residue")
        if self.n_sequences < 0 or self.n_hits < 0:
            raise ValueError("profile sequence counts are non-negative")

    @property
    def length(self) -> int:
        return len(self.wildtype)

    @property
    def log10_neff(self) -> float:
        """Base-ten log of Neff, with the empty profile at zero rather than -inf.

        A wild type with no qualifying corpus support has Neff 0. ``log10(0)``
        is not a stratifier value, and mapping it to 0.0 is exact rather than a
        fudge: ``log10`` of the one effective sequence such a profile would have
        if its own query were counted is 0.0, and the stratum it lands in is the
        bottom one either way.
        """

        return float(np.log10(self.neff)) if self.neff >= 1.0 else 0.0

    def record(self) -> dict[str, Any]:
        supported = int((self.column_weight > 0).sum())
        return {
            "query_id": self.query_id,
            "length": self.length,
            "n_hits": int(self.n_hits),
            "n_profile_sequences": int(self.n_sequences),
            "hit_list_saturated": bool(self.saturated),
            "neff": _finite(self.neff, "neff"),
            "log10_neff": _finite(self.log10_neff, "log10 neff"),
            "max_identity_over_query": _finite(
                self.max_identity_over_query, "max identity"
            ),
            "supported_columns": supported,
            "supported_column_fraction": _finite(
                supported / self.length, "supported column fraction"
            ),
        }


def build_profile(
    wildtype: str,
    query_id: str,
    hits: Sequence[Hit],
    *,
    max_sequences: int,
    coverage_floor: float = PROFILE_COVERAGE_FLOOR,
    reweight_identity: float = REWEIGHT_IDENTITY_FLOOR,
    neff_identity_floor: float = NEFF_IDENTITY_FLOOR,
) -> Profile:
    """Weighted column frequencies over one wild type's corpus alignments.

    One corpus sequence contributes one row: where DIAMOND reports several HSPs
    against the same subject only the highest-scoring one is kept, because a
    subject counted twice is a subject that halves its own reweighted neighbours
    and doubles its column mass.

    Hits are ordered by bitscore and cut at ``max_sequences``. The cut is
    recorded rather than hidden: it right-censors Neff, which compresses the top
    of the stratifier, and a stage that does not report it invites the reader to
    treat a censored axis as a measured one.
    """

    if not wildtype:
        raise ValueError("a profile needs a non-empty wild type")
    if max_sequences < 1:
        raise ValueError("max_sequences must be positive")
    length = len(wildtype)
    best: dict[str, Hit] = {}
    for hit in hits:
        if hit.qlen != length:
            raise ValueError(
                f"{query_id}: DIAMOND reports qlen {hit.qlen} for a "
                f"{length}-residue wild type"
            )
        if hit.qseq_gapped is None or hit.sseq_gapped is None:
            raise ValueError(
                f"{query_id}: hit against {hit.subject} carries no aligned "
                "sequences; the search must request homology.ALIGNMENT_FIELDS"
            )
        aligned = hit.qend - hit.qstart + 1
        if 100.0 * aligned / length < coverage_floor:
            continue
        previous = best.get(hit.subject)
        if previous is None or hit.bitscore > previous.bitscore:
            best[hit.subject] = hit

    ordered = sorted(best.values(), key=lambda hit: (-hit.bitscore, hit.subject))
    saturated = len(ordered) > max_sequences
    ordered = ordered[:max_sequences]

    rows = np.full((len(ordered), length), GAP_CODE, dtype=np.uint8)
    for index, hit in enumerate(ordered):
        query_aligned = hit.qseq_gapped
        subject_aligned = hit.sseq_gapped
        if len(query_aligned) != len(subject_aligned):
            # DIAMOND's ungapped ``qseq``/``sseq`` differ in length on any
            # alignment carrying an indel, and walking them together would shift
            # every column after the first gap. Refused rather than trimmed.
            raise ValueError(
                f"{query_id}: alignment against {hit.subject} has {len(query_aligned)} "
                f"query columns and {len(subject_aligned)} subject columns; the search "
                "must request qseq_gapped/sseq_gapped (homology.ALIGNMENT_FIELDS)"
            )
        position = hit.qstart - 1
        codes = _encode(subject_aligned)
        for column, residue in enumerate(query_aligned):
            if residue == "-":
                continue
            if position >= length:
                raise ValueError(
                    f"{query_id}: alignment against {hit.subject} runs past the "
                    "wild type's last residue"
                )
            rows[index, position] = codes[column]
            position += 1
        if position != hit.qend:
            raise ValueError(
                f"{query_id}: alignment against {hit.subject} covers query residues "
                f"{hit.qstart}-{position} where DIAMOND reports {hit.qstart}-{hit.qend}"
            )

    frequencies = np.zeros((length, len(AA20)), dtype=np.float64)
    column_weight = np.zeros(length, dtype=np.float64)
    neff = 0.0
    if len(ordered):
        weights = sequence_weights(rows, identity=reweight_identity)
        for index in range(len(AA20)):
            frequencies[:, index] = (weights[:, None] * (rows == index)).sum(axis=0)
        column_weight = frequencies.sum(axis=1)
        positive = column_weight > 0
        frequencies[positive] /= column_weight[positive, None]
        neff = float(
            sum(
                weight
                for weight, hit in zip(weights, ordered)
                if hit.identity_over_query >= neff_identity_floor
            )
        )
    return Profile(
        query_id=query_id,
        wildtype=wildtype,
        n_hits=len(hits),
        n_sequences=len(ordered),
        saturated=saturated,
        frequencies=frequencies,
        column_weight=column_weight,
        neff=neff,
        max_identity_over_query=(
            max((hit.identity_over_query for hit in hits), default=0.0)
        ),
    )


def substitution_codes(substitutions: Sequence[tuple[str, int, str]]) -> np.ndarray:
    """``(n, 3)`` of ``(position - 1, wild code, mutant code)``, refusing anything else."""

    if not substitutions:
        raise ValueError("a variant must carry at least one substitution")
    index = {residue: code for code, residue in enumerate(AA20)}
    rows = []
    for wild, position, mutated in substitutions:
        if wild not in index or mutated not in index:
            raise ValueError(f"substitution {wild}{position}{mutated} is outside the alphabet")
        if position < 1:
            raise ValueError(f"substitution {wild}{position}{mutated} has a non-positive position")
        rows.append((position - 1, index[wild], index[mutated]))
    return np.asarray(rows, dtype=np.int64)


def lookup_score(
    frequencies: np.ndarray,
    background: np.ndarray,
    codes: np.ndarray,
    *,
    alpha: float = PSEUDOCOUNT_ALPHA,
) -> float:
    """``sum_p log[f(m,p) + alpha*b(m)] - log[f(w,p) + alpha*b(w)]``.

    Both terms are taken at the same column, so the per-column normaliser
    cancels and the score does not depend on how much evidence the column
    carries -- only on what that evidence says. The support itself is reported
    separately (``Profile.column_weight``) rather than folded in here, because
    folding it in would make the channel a confidence-weighted predictor and no
    longer the site-independent lookup it is being compared against.
    """

    if alpha <= 0:
        raise ValueError("the pseudocount weight must be positive")
    if frequencies.ndim != 2 or frequencies.shape[1] != len(AA20):
        raise ValueError("frequencies must be (length, 20)")
    positions = codes[:, 0]
    if positions.min() < 0 or positions.max() >= frequencies.shape[0]:
        raise IndexError(
            f"substitution position outside a {frequencies.shape[0]}-column profile"
        )
    wild = codes[:, 1]
    mutated = codes[:, 2]
    numerator = frequencies[positions, mutated] + alpha * background[mutated]
    denominator = frequencies[positions, wild] + alpha * background[wild]
    return _finite(float(np.log(numerator).sum() - np.log(denominator).sum()), "lookup score")


def profile_scores(
    profile: Profile,
    background: np.ndarray,
    variants: Sequence[Sequence[tuple[str, int, str]]],
    *,
    alpha: float = PSEUDOCOUNT_ALPHA,
    check_wildtype: bool = True,
) -> np.ndarray:
    """LOOKUP scores for a list of variants of this profile's own wild type.

    ``check_wildtype`` verifies that the mutation string's wild-type residue is
    the residue the profile's wild type actually carries at that position. It is
    on for the real channel and off for the mismatched-profile control, which
    deliberately scores a variant against a different protein's columns.
    """

    scores = np.empty(len(variants), dtype=np.float64)
    for index, substitutions in enumerate(variants):
        codes = substitution_codes(substitutions)
        if check_wildtype:
            for (wild, position, _), code in zip(substitutions, codes):
                if profile.wildtype[position - 1] != wild:
                    raise ValueError(
                        f"{profile.query_id}: mutation string says {wild!r} at position "
                        f"{position}, wild type carries {profile.wildtype[position - 1]!r}"
                    )
        scores[index] = lookup_score(
            profile.frequencies, background, codes, alpha=alpha
        )
    return scores


# --------------------------------------------------------------- free baselines


def free_baselines(
    variants: Sequence[Sequence[tuple[str, int, str]]],
    background: np.ndarray,
    *,
    wildtype_length: int,
) -> dict[str, np.ndarray]:
    """Predictors computable from the mutation string alone (rule 28).

    None of these sees the model, the corpus alignment or the labels.
    ``background_composition`` is the one that matters most: it is exactly what
    :func:`lookup_score` degenerates to when the profile supports no column, so
    a LOOKUP channel that does not beat it has contributed no position-specific
    information at all and the whole subtraction would be about composition.
    """

    if wildtype_length < 1:
        raise ValueError("wild-type length must be positive")
    index = {residue: code for code, residue in enumerate(AA20)}
    position_index = np.empty(len(variants), dtype=np.float64)
    wt_hydropathy = np.empty(len(variants), dtype=np.float64)
    hydropathy_change = np.empty(len(variants), dtype=np.float64)
    composition = np.empty(len(variants), dtype=np.float64)
    for row, substitutions in enumerate(variants):
        if not substitutions:
            raise ValueError("a variant must carry at least one substitution")
        position_index[row] = float(
            np.mean([position for _, position, _ in substitutions]) / wildtype_length
        )
        wt_hydropathy[row] = sum(KYTE_DOOLITTLE[wild] for wild, _, _ in substitutions)
        hydropathy_change[row] = sum(
            KYTE_DOOLITTLE[mutated] - KYTE_DOOLITTLE[wild]
            for wild, _, mutated in substitutions
        )
        composition[row] = sum(
            math.log(background[index[mutated]]) - math.log(background[index[wild]])
            for wild, _, mutated in substitutions
        )
    return {
        "position_index": position_index,
        "wt_hydropathy": wt_hydropathy,
        "hydropathy_change": hydropathy_change,
        "background_composition": composition,
    }


# ------------------------------------------------------------------ clustering


def cluster_by_identity(
    identifiers: Sequence[str],
    lengths: Mapping[str, int],
    hits: Sequence[Hit],
    *,
    identity: float = FAMILY_IDENTITY,
    coverage: float = FAMILY_COVERAGE,
) -> dict[str, int]:
    """Single-linkage clusters of the wild types, under UniRef50's own rule.

    Identity and coverage are both taken over the **shorter** of the two
    sequences, which is the clustering definition UniRef uses; taking them over
    the query would make the relation asymmetric and the components would depend
    on which of a pair DIAMOND happened to report first.

    Single linkage, not complete: the unit has to be a set of wild types that
    cannot be treated as independent, and transitivity through an intermediate
    homologue is exactly the dependence a bootstrap over families must absorb.
    """

    order = {identifier: position for position, identifier in enumerate(identifiers)}
    parent = list(range(len(identifiers)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for hit in hits:
        if hit.query not in order or hit.subject not in order:
            raise ValueError(f"hit {hit.query}->{hit.subject} is not between wild types")
        if hit.query == hit.subject:
            continue
        shorter = min(lengths[hit.query], lengths[hit.subject])
        if shorter < 1:
            raise ValueError("a wild type of zero length cannot be clustered")
        if 100.0 * hit.nident / shorter < identity:
            continue
        if 100.0 * (hit.qend - hit.qstart + 1) / shorter < coverage:
            continue
        left, right = find(order[hit.query]), find(order[hit.subject])
        if left != right:
            parent[left] = right
    roots: dict[int, int] = {}
    assignment: dict[str, int] = {}
    for identifier in identifiers:
        root = find(order[identifier])
        if root not in roots:
            roots[root] = len(roots)
        assignment[identifier] = roots[root]
    return assignment


def mismatched_donors(
    identifiers: Sequence[str],
    lengths: Mapping[str, int],
    log10_neff: Mapping[str, float],
    clusters: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    """For each wild type, a length- and Neff-matched profile from another family.

    The control asks whether the LOOKUP channel is reading *this* protein's
    columns or generic protein composition. A donor is therefore required to be
    outside the recipient's 50%-identity cluster, and to be **at least as long**
    as the recipient so that every substituted position has a column: filling a
    missing column from the background would make the control partly the
    background-composition baseline it is supposed to be separable from.

    Among eligible donors the nearest is taken under a declared distance in the
    two matching coordinates. The tolerances in
    :data:`MISMATCH_LENGTH_TOLERANCE` and :data:`MISMATCH_LOG10_NEFF_TOLERANCE`
    are then *reported* as achieved or not, rather than used as a filter that
    silently drops the wild types hardest to match -- which are the long, deeply
    supported ones the estimand cares most about.
    """

    result: dict[str, dict[str, Any]] = {}
    for identifier in identifiers:
        own_length = lengths[identifier]
        own_neff = log10_neff[identifier]
        best: tuple[float, str] | None = None
        for candidate in identifiers:
            if candidate == identifier or clusters[candidate] == clusters[identifier]:
                continue
            if lengths[candidate] < own_length:
                continue
            length_gap = abs(math.log(lengths[candidate] / own_length))
            neff_gap = abs(log10_neff[candidate] - own_neff)
            distance = (
                length_gap / math.log(1.0 + MISMATCH_LENGTH_TOLERANCE)
                + neff_gap / MISMATCH_LOG10_NEFF_TOLERANCE
            )
            if best is None or distance < best[0] or (
                distance == best[0] and candidate < best[1]
            ):
                best = (distance, candidate)
        if best is None:
            result[identifier] = {"donor": None, "reason": "no longer wild type outside its cluster"}
            continue
        donor = best[1]
        length_ratio = lengths[donor] / own_length - 1.0
        neff_gap = abs(log10_neff[donor] - own_neff)
        result[identifier] = {
            "donor": donor,
            "distance": _finite(best[0], "donor distance"),
            "length_excess": _finite(length_ratio, "donor length excess"),
            "log10_neff_gap": _finite(neff_gap, "donor neff gap"),
            "length_matched": bool(length_ratio <= MISMATCH_LENGTH_TOLERANCE),
            "neff_matched": bool(neff_gap <= MISMATCH_LOG10_NEFF_TOLERANCE),
        }
    return result


# ------------------------------------------------------------------ statistics


def cluster_means(
    values: Sequence[float], clusters: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Average within cluster; return the per-cluster values and their labels.

    This is the step that makes the resampling unit a family. 217 assays carry
    187 wild types and those fall into far fewer families, so an interval taken
    over assays would be an interval over a population whose members are copies
    of each other.
    """

    array = np.asarray(values, dtype=np.float64)
    labels = np.asarray(clusters)
    if array.ndim != 1 or labels.shape != array.shape:
        raise ValueError("values and cluster labels must be aligned one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError("cluster means require finite values")
    unique = np.unique(labels)
    means = np.array([array[labels == label].mean() for label in unique], dtype=np.float64)
    return means, unique


def cluster_bootstrap(
    values: Sequence[float],
    clusters: Sequence[int],
    *,
    resamples: int,
    seed: int,
    alpha: float = 0.05,
    minimum_units: int = MINIMUM_BOOTSTRAP_UNITS,
) -> dict[str, Any]:
    """Percentile interval on the cluster-mean average, resampling clusters.

    Refused below the unit floor, with the reason carried into the artefact, in
    the shape :func:`~.statistics.bootstrap_unit_floor` declares once for the
    whole package.
    """

    if resamples < 1 or not 0 < alpha < 1:
        raise ValueError("invalid bootstrap parameters")
    means, labels = cluster_means(values, clusters)
    floor = bootstrap_unit_floor(int(means.size), minimum_units=minimum_units)
    record: dict[str, Any] = {
        "point": _finite(float(means.mean()), "cluster-mean point estimate"),
        "resamples": int(resamples),
        "alpha": float(alpha),
        "unit": "wild-type family at 50% identity",
        **floor,
        "interval": None,
        "excludes_zero": None,
    }
    if floor["degenerate"]:
        return record
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, means.size, size=(resamples, means.size))
    statistic = means[draws].mean(axis=1)
    low, high = np.percentile(statistic, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    record["interval"] = [_finite(float(low), "interval low"), _finite(float(high), "interval high")]
    record["excludes_zero"] = bool(low > 0.0 or high < 0.0)
    record["n_assays"] = int(np.asarray(values).size)
    record["cluster_sizes"] = [
        int((np.asarray(clusters) == label).sum()) for label in labels
    ]
    return record


def share_bootstrap(
    numerator: Sequence[float],
    denominator: Sequence[float],
    clusters: Sequence[int],
    *,
    resamples: int,
    seed: int,
    alpha: float = 0.05,
    minimum_units: int = MINIMUM_BOOTSTRAP_UNITS,
    zero_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """The retrieval share, resampled as one ratio rather than as two means.

    The headline the whole stage exists to produce is *what fraction of the
    model's advantage over a free baseline a lookup of its corpus already has*.
    Both halves are measured on the same assays, so the ratio has to be taken
    inside the resample -- one draw of clusters scoring both -- or the interval
    is the interval of two independent quantities and means nothing about their
    quotient.

    Withheld, with the reason, when the denominator crosses zero on more than
    the tail :data:`~.statistics.MINIMUM_FINITE_DRAW_FRACTION` allows: a ratio
    whose denominator can vanish is unbounded, and a percentile interval over
    the draws that happened to survive is conditioned on survival rather than
    being the requested distribution.
    """

    from ..core.statistics import MINIMUM_FINITE_DRAW_FRACTION

    if resamples < 1 or not 0 < alpha < 1:
        raise ValueError("invalid bootstrap parameters")
    top, labels = cluster_means(numerator, clusters)
    bottom, other = cluster_means(denominator, clusters)
    if not np.array_equal(labels, other):
        raise ValueError("the two channels are not aligned on the same clusters")
    floor = bootstrap_unit_floor(int(top.size), minimum_units=minimum_units)
    record: dict[str, Any] = {
        "numerator": _finite(float(top.mean()), "share numerator"),
        "denominator": _finite(float(bottom.mean()), "share denominator"),
        "resamples": int(resamples),
        "unit": "wild-type family at 50% identity",
        **floor,
        "share": None,
        "interval": None,
    }
    if floor["degenerate"]:
        return record
    if abs(float(bottom.mean())) <= zero_tolerance:
        record["withheld_reason"] = "the denominator is zero on the full sample"
        return record
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, top.size, size=(resamples, top.size))
    top_draws = top[draws].mean(axis=1)
    bottom_draws = bottom[draws].mean(axis=1)
    usable = np.abs(bottom_draws) > zero_tolerance
    if usable.mean() < MINIMUM_FINITE_DRAW_FRACTION:
        record["withheld_reason"] = (
            f"only {usable.mean():.1%} of draws have a non-zero denominator, below "
            f"the {MINIMUM_FINITE_DRAW_FRACTION:.0%} floor; the share is not bounded "
            "on this cohort"
        )
        return record
    ratios = top_draws[usable] / bottom_draws[usable]
    low, high = np.percentile(ratios, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    record["share"] = _finite(float(top.mean() / bottom.mean()), "retrieval share")
    record["interval"] = [_finite(float(low), "share low"), _finite(float(high), "share high")]
    record["n_usable_draws"] = int(usable.sum())
    return record


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Threshold-free association between a stratifier and a per-unit quantity."""

    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size < 3:
        raise ValueError("Kendall tau needs two aligned vectors of at least three units")
    result = stats.kendalltau(left, right)
    tau = float(result.statistic)
    if not math.isfinite(tau):
        # Every value tied on one side. Reported as undefined rather than as
        # zero: no association and no variance are different findings.
        return {"tau": None, "p_value": None, "n": int(left.size), "undefined": True}
    return {
        "tau": tau,
        "p_value": _finite(float(result.pvalue), "kendall p"),
        "n": int(left.size),
        "undefined": False,
    }


def bin_sweep(
    stratifier: Sequence[float],
    values: Sequence[float],
    edge_sweep: Sequence[Sequence[float]],
    *,
    minimum_units: int = MINIMUM_BOOTSTRAP_UNITS,
) -> dict[str, Any]:
    """Per-bin means over several partitions, and whether their ordering is invariant.

    Appendix B rule 17: a result stated as an ordering must hold across the
    sweep, or it is a statement about the cut. The ordering compared is the sign
    of the difference between the lowest and highest populated bin, taken over
    bins that clear the unit floor -- a bin below it carries a mean that must not
    be compared against another population's, which is what
    :data:`~.statistics.MINIMUM_BOOTSTRAP_UNITS` was measured for.
    """

    axis = np.asarray(stratifier, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if axis.shape != array.shape or axis.ndim != 1:
        raise ValueError("stratifier and values must be aligned one-dimensional")
    partitions: list[dict[str, Any]] = []
    directions: list[int] = []
    for edges in edge_sweep:
        edge = [float(value) for value in edges]
        if len(edge) < 3 or any(b <= a for a, b in zip(edge, edge[1:])):
            raise ValueError(f"bin edges {edge} are not strictly increasing with two bins")
        bins: list[dict[str, Any]] = []
        for low, high in zip(edge, edge[1:]):
            mask = (axis >= low) & (axis < high)
            count = int(mask.sum())
            bins.append(
                {
                    "low": low,
                    "high": high,
                    "n_units": count,
                    "mean": _finite(float(array[mask].mean()), "bin mean") if count else None,
                    "below_unit_floor": bool(count < minimum_units),
                }
            )
        usable = [b for b in bins if b["mean"] is not None and not b["below_unit_floor"]]
        direction = 0
        if len(usable) >= 2:
            difference = usable[-1]["mean"] - usable[0]["mean"]
            direction = int(np.sign(difference))
        partitions.append({"edges": edge, "bins": bins, "top_minus_bottom_sign": direction})
        directions.append(direction)
    usable_directions = [value for value in directions if value != 0]
    return {
        "partitions": partitions,
        "ordering_invariant": bool(
            len(usable_directions) >= 2 and len(set(usable_directions)) == 1
        ),
        "signs": directions,
        "note": (
            "ordering_invariant is false when fewer than two partitions yield a "
            "comparable pair of bins above the unit floor; that is a statement "
            "about power, not about the ordering"
        ),
    }


def out_of_fold_difficulty_residual(
    values: Sequence[float],
    covariates: np.ndarray,
    clusters: Sequence[int],
    *,
    n_splits: int,
    seed: int,
) -> dict[str, Any]:
    """Subtract a difficulty prediction fitted on cluster-disjoint training folds.

    **Not a partial correlation.** This repository has retracted a partial
    correlation twice: it removes a monotone trend from both variables at once
    over the whole sample, so the covariate is fitted on the very units it then
    adjusts, and a covariate that is a function of the outcome leaves a residual
    correlation of one between two vectors carrying nothing. Here the difficulty
    model is fitted on training folds that share no wild-type family with the
    fold it predicts, so a covariate that only memorises its training units
    predicts nothing out of fold and subtracts nothing.

    Ordinary least squares with an intercept over a handful of declared
    covariates. Returns the residuals in the input order, plus the out-of-fold
    coefficient of determination, which is the honest measure of how much
    difficulty there was to remove: a negative value means the difficulty model
    is worse than the training mean, and the residual is then noise added rather
    than a confound removed.
    """

    array = np.asarray(values, dtype=np.float64)
    design = np.asarray(covariates, dtype=np.float64)
    if array.ndim != 1 or design.ndim != 2 or design.shape[0] != array.size:
        raise ValueError("covariates must be (units, features) aligned with the values")
    if not np.isfinite(array).all() or not np.isfinite(design).all():
        raise ValueError("difficulty control requires finite values and covariates")
    splits = make_group_splits(
        array, np.asarray(clusters), n_splits=n_splits, seed=seed, task_type="regression"
    )
    prediction = np.full(array.size, np.nan, dtype=np.float64)
    for train, test in splits:
        matrix = np.column_stack([np.ones(train.size), design[train]])
        coefficients, *_ = np.linalg.lstsq(matrix, array[train], rcond=None)
        prediction[test] = np.column_stack(
            [np.ones(test.size), design[test]]
        ) @ coefficients
    if not np.isfinite(prediction).all():
        raise RuntimeError("the difficulty control left a unit unpredicted")
    residual = array - prediction
    total = float(((array - array.mean()) ** 2).sum())
    if total <= 0:
        raise RuntimeError("the values to be adjusted have no variance")
    return {
        "residual": residual,
        "out_of_fold_r2": _finite(1.0 - float((residual**2).sum()) / total, "out-of-fold R2"),
        "n_splits": int(n_splits),
        "n_features": int(design.shape[1]),
        "mean_prediction": _finite(float(prediction.mean()), "mean difficulty prediction"),
    }


# ----------------------------------------------------------------- the verdict


def equivalence_verdict(
    delta_lookup: Mapping[str, Any], delta_blosum: Mapping[str, Any]
) -> dict[str, Any]:
    """The pre-registered reading of the interval, including what a null says.

    ``acquired``
        the MODEL - LOOKUP interval lies wholly above zero. The model carries
        fitness information a site-independent lookup of its own corpus does not.

    ``retrieval_bounded``
        the interval lies wholly inside plus or minus
        :data:`EQUIVALENCE_FRACTION` of the arm's own measured MODEL - BLOSUM62
        advantage. This is the null that establishes something: at least half of
        everything the model has over a free baseline is also available from the
        corpus lookup, so the acquired component is bounded and small.

    ``retrieval_dominated``
        the interval lies wholly below minus that bound. The lookup beats the
        model by more than the equivalence bound, so the question is resolved
        *against* the model.

        **Added after the fact, and that is recorded rather than hidden.** The
        first three outcomes were declared before any number existed and did not
        cover this case; ProGen3-112M produced it (-0.0808 [-0.1139, -0.0481]
        against a bound of 0.0340) and fell through to ``indeterminate``, whose
        stated reason -- "the interval spans zero" -- was false of that interval
        and whose documented meaning is that the cohort could not resolve the
        question, when it had resolved it decisively. A category that can only
        ever make an arm look worse cannot be a category chosen to favour a
        hypothesis, which is why adding it after seeing the data is admissible
        here; the three original readings are unchanged.

    ``indeterminate``
        the interval spans zero and leaves the equivalence bound, or the arm's
        own MODEL - BLOSUM62 advantage is not separable from zero so there is no
        advantage to partition. Not a null: a statement that this cohort cannot
        resolve the question, reported with the cluster count that could not
        resolve it.

    The bound is a fraction of the arm's *own* advantage rather than a fixed
    Spearman, because the arms differ in how much they beat a free baseline at
    all and an absolute bound would be a different question on each one. Where
    that advantage is not itself separable from zero there is nothing to
    partition, and the equivalence reading is withheld.
    """

    interval = delta_lookup.get("interval")
    base = delta_blosum.get("point")
    base_interval = delta_blosum.get("interval")
    if interval is None or base is None:
        return {
            "verdict": "indeterminate",
            "reason": delta_lookup.get("degenerate_reason")
            or "no interval was computable for MODEL - LOOKUP",
            "equivalence_bound": None,
        }
    bound = EQUIVALENCE_FRACTION * abs(float(base))
    base_separable = bool(
        base_interval is not None and (base_interval[0] > 0.0 or base_interval[1] < 0.0)
    )
    if interval[0] > 0.0:
        verdict = "acquired"
        reason = "the MODEL - LOOKUP interval lies wholly above zero"
    elif base_separable and -bound <= interval[0] and interval[1] <= bound:
        verdict = "retrieval_bounded"
        reason = (
            f"the MODEL - LOOKUP interval is contained in +/-{bound:.4f}, which is "
            f"{EQUIVALENCE_FRACTION:.0%} of this arm's own MODEL - BLOSUM62 advantage"
        )
    elif base_separable and interval[1] < -bound:
        verdict = "retrieval_dominated"
        reason = (
            f"the MODEL - LOOKUP interval lies wholly below -{bound:.4f}: the corpus "
            "lookup beats this arm by more than the equivalence bound, so the "
            "question is resolved against the model rather than left unresolved"
        )
    elif not base_separable:
        verdict = "indeterminate"
        reason = (
            "this arm's MODEL - BLOSUM62 advantage is not itself separable from "
            "zero, so there is no advantage to partition and no equivalence bound "
            "is defined"
        )
    else:
        verdict = "indeterminate"
        reason = (
            "the interval spans zero and leaves the equivalence bound"
            if interval[0] <= 0.0 <= interval[1]
            else f"the interval is wholly below zero but not wholly below "
            f"-{bound:.4f}, so it is neither an equivalence nor a resolved "
            f"advantage for the lookup"
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "equivalence_bound": _finite(bound, "equivalence bound"),
        "equivalence_fraction": EQUIVALENCE_FRACTION,
        "model_minus_blosum_point": _finite(float(base), "model minus blosum"),
        "model_minus_blosum_separable_from_zero": base_separable,
        "frozen_reference": {
            "source": "EXP-R2-134, ProGen3-112M over all 217 substitution assays",
            "model_minus_blosum62": FROZEN_MODEL_MINUS_BLOSUM,
            "blosum62_mean_spearman": FROZEN_BLOSUM62_MEAN_SPEARMAN,
        },
    }


def verbatim_anchor_check(
    verbatim: Iterable[str],
    wildtypes: Mapping[str, str],
    identities: Mapping[str, float],
    *,
    floor: float = 99.0,
) -> dict[str, Any]:
    """Every byte-identical corpus member must read as a near-duplicate of itself.

    This is EXP-R2-061's lesson made executable. There, DIAMOND's default repeat
    masking truncated the alignments of verbatim corpus members and binned them
    as diverged relatives -- an error whose every instance ran in the direction
    that defeated the hypothesis under test, and which left no trace in any
    artefact. A run whose aligner disagrees with the corpus's own bytes is not a
    weaker measurement, it is a different one, so the caller raises on a failure
    rather than recording a flag.
    """

    anchors = sorted(set(verbatim))
    by_sequence = {sequence: identifier for identifier, sequence in wildtypes.items()}
    failures: list[dict[str, Any]] = []
    checked = 0
    for sequence in anchors:
        identifier = by_sequence.get(sequence)
        if identifier is None:
            raise KeyError("a verbatim corpus member is not among the wild types searched")
        observed = float(identities[identifier])
        checked += 1
        if observed < floor:
            failures.append(
                {
                    "query_id": identifier,
                    "length": len(sequence),
                    "max_identity_over_query": observed,
                }
            )
    return {
        "anchors": checked,
        "identity_floor": float(floor),
        "failures": failures,
        "passes": not failures,
        "message": (
            "every wild type that is byte-identical to a corpus record reads at or "
            f"above {floor:.0f}% identity over the query"
            if not failures
            else f"{len(failures)} byte-identical corpus members read below "
            f"{floor:.0f}% identity over the query. That is the EXP-R2-061 failure: "
            "DIAMOND masking truncates the alignment of exactly the records that "
            "are stored verbatim, and every such error under-states retrieval. "
            "Re-run the search with --masking 0"
        ),
    }
