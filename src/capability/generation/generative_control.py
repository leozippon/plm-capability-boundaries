"""Matched statistical generators for the generation-and-control gate.

Why this module exists
======================

The generation gate asks whether a frozen decoder's *samples* satisfy a
biological requirement. Every existing generation endpoint in this programme
compares a generated sequence either against its own composition shuffle (the
structure contrast) or against another condition of the same arm (the requested
-minus-mismatched profile contrast). Neither answers the question a reader asks
first: **a generator that knows only the corpus's residue statistics would also
produce sequences a curated profile recognises -- how much of the endpoint is
that?**

Answering it needs generators, not scores. A baseline that ranks mutations
cannot be compared against a sampler, so the controls here are *generators*: each
one emits one sequence per attempt of the arm's own ledger, at that attempt's
exact length, carrying a declared and verifiable amount of sequence statistics
and nothing more. The declared amounts form one axis:

===============  ==========================================================
``shuffle``      the attempt's own residue multiset, order exact, order > 0
                 destroyed
``markov_k``     the corpus's ``(k+1)``-mer statistics, for k = 0, 2, 4
``fragment``     a contiguous substring of one corpus record -- every corpus
                 k-mer order at once, the limit of the ``markov_k`` axis
``hydropathy``   the corpus composition tilted to the attempt's own mean
                 Kyte-Doolittle hydropathy, and nothing else
``natural``      a whole corpus record in the attempt's length band, the
                 reference for what satisfying the requirement looks like
``profile``      sequences emitted from the oracle's own profiles: an
                 oracle-access ceiling, declared as such and never a control
                 the model is asked to beat
===============  ==========================================================

Competence before comparison
============================

The capability map binds every control with one rule: a control enters a
comparison only if it is competent in its own right. For a *predictor* that
means predicting held-out groups; for a *generator* it means reproducing the
order statistics it declares. :func:`qualify_cohort` measures that and returns a
receipt, and a cohort whose receipt does not pass licenses nothing -- the same
disposition the local-context gate's failed control receives.

The measurement here is deliberately not a model quantity. Which sequences a
profile recognises is an oracle output, so this module produces sequences and
receipts and leaves the oracle to the stage that runs it.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.amino_acids import AA20
from ..context.profiles import KYTE_DOOLITTLE
from ..core.statistics import MINIMUM_BOOTSTRAP_UNITS

#: This gate's own draw seed. Every generator, every pool draw and every
#: bootstrap in the gate reads it, so one value reproduces the gate.
GATE_SEED: int = 20260924

#: Bootstrap resamples, matching the generation-biology analysis so that a
#: contrast here and a structure contrast there carry comparable uncertainty.
RESAMPLES: int = 4000

#: Conditioning orders on the corpus k-mer axis. Order ``k`` emits from
#: ``P(residue | previous k residues)`` and therefore needs ``(k + 1)``-mer
#: counts. The staged UniRef50 background observes every 5-mer, so order 4 has
#: no unseen context and needs no smoothing; ``fragment`` supplies the limit of
#: the axis rather than a higher order, because a contiguous corpus substring
#: carries every order exactly.
MARKOV_ORDERS: tuple[int, ...] = (0, 2, 4)

#: Tolerance in nats per residue for "this cohort reproduces the order
#: statistics it declares", declared before any cohort was generated.
#:
#: **The premise this value was chosen against was wrong, and the record keeps
#: both.** It was declared on the belief that the corpus's own gain between
#: adjacent orders runs from 0.1 nats upward, which would have put 0.02 nats
#: well inside one order step. Measured on the length-matched corpus reference
#: of this gate, the adjacent-order gain is **0.0062 nats from order 0 to 1,
#: 0.0094 from 2 to 3 and 0.0246 from 4 to 5**. An absolute 0.02-nat margin is
#: therefore *larger* than one order step below order 4, so the second
#: condition -- sitting at least this far below the reference one order up --
#: is unreachable at orders 0 and 2 however faithful the sampler is.
#:
#: The value is not changed, because it was pre-declared and the outcome under
#: it is the primary one. :func:`qualify_markov` additionally measures the
#: scale-free form of the same condition, as a named post-hoc amendment, and
#: the gate reports its verdict under both sets.
ORDER_STATISTIC_TOLERANCE: float = 0.02

#: Tolerance in Kyte-Doolittle units for the hydropathy-matched generator's
#: realized per-sequence mean against its target. A length-L i.i.d. draw from a
#: tilted background has standard error ~2.0/sqrt(L), which is 0.14 at L = 200,
#: so this is a per-cohort mean-absolute-deviation condition rather than a
#: per-sequence one.
HYDROPATHY_TOLERANCE: float = 0.25

#: Fraction of a Pfam profile's model length the best domain alignment must
#: cover for the assignment to count as a complete domain rather than a
#: fragment. Declared before any coverage value was read.
COMPLETE_DOMAIN_COVERAGE: float = 0.80

_KD = np.array([KYTE_DOOLITTLE[residue] for residue in AA20], dtype=float)
_INDEX = {residue: position for position, residue in enumerate(AA20)}


def mean_hydropathy(sequence: str) -> float:
    """Mean Kyte-Doolittle hydropathy, in KD units per residue."""

    if not sequence:
        raise ValueError("an empty sequence has no mean hydropathy")
    return float(np.mean([KYTE_DOOLITTLE[residue] for residue in sequence]))


def encode(sequence: str) -> np.ndarray:
    """Residue indices into :data:`AA20`; raises on any non-canonical symbol."""

    try:
        return np.fromiter((_INDEX[residue] for residue in sequence), dtype=np.int64,
                           count=len(sequence))
    except KeyError as error:  # pragma: no cover - guarded by the caller's filter
        raise ValueError(f"{error.args[0]!r} is not in {AA20}") from error


def decode(indices: np.ndarray) -> str:
    return "".join(AA20[int(index)] for index in indices)


# --------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------


def composition_shuffle(rng: np.random.Generator, sequence: str) -> str:
    """A uniform permutation of the sequence's own residues.

    Length and residue multiset are exact by construction, which is what makes
    this the tightest control on the axis: everything it removes is residue
    order. A sequence whose permutation happens to equal itself is returned
    unchanged rather than redrawn, because redrawing until the control differs
    would select on the control's outcome.
    """

    if not sequence:
        return ""
    residues = np.array(list(sequence))
    rng.shuffle(residues)
    return "".join(residues.tolist())


@dataclass(frozen=True)
class MarkovSampler:
    """An order-``k`` residue Markov chain read off staged corpus k-mer counts.

    ``conditional`` is ``P(residue | previous k residues)`` with contexts in
    :func:`~.kmer_background.kmer_index` order, and ``context_marginal`` is the
    context distribution used to start a sequence. Both come from counts of the
    corpus, so the sampler is the corpus's own ``(k + 1)``-mer statistics and
    carries no arm-specific information at all.
    """

    order: int
    conditional: np.ndarray
    context_marginal: np.ndarray

    def emit(self, rng: np.random.Generator, length: int) -> str:
        if length <= 0:
            return ""
        if self.order == 0:
            draw = rng.choice(len(AA20), size=length, p=self.conditional[0])
            return decode(draw)
        if length <= self.order:
            # Too short to carry a full context: emit from the context marginal
            # collapsed onto single residues rather than refusing, so that the
            # control cohort keeps the arm's own length distribution.
            residue_marginal = self.conditional.T @ self.context_marginal
            residue_marginal = residue_marginal / residue_marginal.sum()
            return decode(rng.choice(len(AA20), size=length, p=residue_marginal))
        context = int(rng.choice(len(self.context_marginal), p=self.context_marginal))
        out: list[int] = []
        width = len(AA20)
        divisor = width ** (self.order - 1)
        for position in range(self.order):
            out.append((context // (width ** (self.order - 1 - position))) % width)
        for _ in range(length - self.order):
            residue = int(rng.choice(width, p=self.conditional[context]))
            out.append(residue)
            context = (context % divisor) * width + residue
        return decode(np.array(out, dtype=np.int64))


def markov_from_counts(counts: Mapping[int, np.ndarray], order: int) -> MarkovSampler:
    """Build an order-``k`` sampler from ``(k + 1)``-mer counts of a corpus.

    Refuses an unseen context rather than smoothing one away: at orders 0 to 4
    the staged UniRef50 background observes every context, so a zero row here
    means the counts are not the ones this sampler was declared against.
    """

    width = len(AA20)
    vector = np.asarray(counts[order + 1], dtype=np.float64)
    if vector.shape != (width ** (order + 1),):
        raise ValueError(f"order {order} needs {width ** (order + 1)} counts of "
                         f"{order + 1}-mers, got {vector.shape}")
    table = vector.reshape(width ** order, width)
    context_totals = table.sum(axis=1)
    if not (context_totals > 0).all():
        raise ValueError(f"order {order} has {int((context_totals == 0).sum())} unseen "
                         "contexts; this sampler does not smooth an unseen context away")
    conditional = table / context_totals[:, None]
    marginal = context_totals / context_totals.sum()
    return MarkovSampler(order=order, conditional=conditional, context_marginal=marginal)


def hydropathy_tilt(background: np.ndarray, target: float) -> tuple[np.ndarray, float, bool]:
    """Exponentially tilt a residue background to a target mean hydropathy.

    Returns the tilted distribution, the tilt parameter, and whether the target
    was reachable. The tilted family ``p_a ∝ q_a exp(λ h_a)`` spans the open
    interval between the alphabet's extreme Kyte-Doolittle values, so a target
    outside it is clamped and the caller is told, rather than silently matched
    to a different number.
    """

    background = np.asarray(background, dtype=float)
    background = background / background.sum()
    low, high = float(_KD.min()), float(_KD.max())
    reachable = low < target < high
    clamped = min(max(target, low + 1e-6), high - 1e-6)

    def mean_at(lam: float) -> float:
        weights = background * np.exp(lam * _KD)
        return float((weights * _KD).sum() / weights.sum())

    left, right = -50.0, 50.0
    for _ in range(200):
        middle = 0.5 * (left + right)
        if mean_at(middle) < clamped:
            left = middle
        else:
            right = middle
    lam = 0.5 * (left + right)
    weights = background * np.exp(lam * _KD)
    return weights / weights.sum(), lam, reachable


def hydropathy_matched(rng: np.random.Generator, background: np.ndarray,
                       target: float, length: int) -> tuple[str, float, bool]:
    """One sequence carrying the corpus composition tilted to ``target``."""

    if length <= 0:
        return "", 0.0, True
    tilted, lam, reachable = hydropathy_tilt(background, target)
    draw = rng.choice(len(AA20), size=length, p=tilted)
    return decode(draw), lam, reachable


def corpus_fragment(rng: np.random.Generator, record: str, length: int) -> str:
    """A contiguous substring of one corpus record, at an exact length."""

    if length <= 0:
        return ""
    if len(record) < length:
        raise ValueError(f"a {len(record)}-residue record cannot supply a "
                         f"{length}-residue fragment")
    start = int(rng.integers(0, len(record) - length + 1))
    return record[start:start + length]


# --------------------------------------------------------------------------
# qualification
# --------------------------------------------------------------------------


def load_kmer_counts(directory: Any, ks: Sequence[int]) -> dict[int, np.ndarray]:
    """Only the count vectors this gate reads, under the background's own digest rule.

    :func:`~.kmer_background.load` reads every ``k`` a directory holds, and the
    staged high-order background holds ``k = 7`` as a 10 GB vector of 1.28e9
    cells. This gate reads ``k = 1`` through ``k = 6``, so loading and hashing
    the seventh would cost more memory than the whole measurement. The digest
    check is the same one that loader applies -- a vector whose bytes have moved
    away from the manifest is refused, not used.
    """

    import json
    from pathlib import Path

    from ..context.kmer_background import ALPHABET, sha256_file

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("alphabet") != ALPHABET:
        raise ValueError(f"{directory} was counted over {manifest.get('alphabet')!r}, "
                         f"not {ALPHABET}")
    counts: dict[int, np.ndarray] = {}
    for k in sorted(set(int(value) for value in ks)):
        if k not in [int(value) for value in manifest["k"]]:
            raise KeyError(f"{directory} holds no k = {k} counts")
        path = directory / f"kmer_counts_k{k}.npy"
        observed = sha256_file(path)
        expected = manifest["sha256"][str(k)]
        if observed != expected:
            raise RuntimeError(f"{path} hashes {observed}, manifest says {expected}")
        vector = np.load(path)
        if vector.shape != (len(ALPHABET) ** k,):
            raise ValueError(f"k = {k} needs {len(ALPHABET) ** k} counts, got {vector.shape}")
        counts[k] = vector
    return counts


def conditional_table(counts: Mapping[int, np.ndarray], order: int,
                      *, smoothing: float = 1.0) -> np.ndarray:
    """``P(residue | previous `order` residues)`` for *evaluating* a cohort.

    Evaluation smooths and sampling does not, and the asymmetry is deliberate.
    A sampler must never emit from a context the corpus has not shown, so
    :func:`markov_from_counts` refuses one. An evaluator must be able to score a
    cohort that emits a k-mer the corpus never contained -- 48,115 of the
    64,000,000 6-mers are unobserved in the staged background -- and assigning
    those minus infinity would make one unseen fragment decide a cohort's whole
    order profile. Additive smoothing is recorded with the number rather than
    left implicit.
    """

    width = len(AA20)
    vector = np.asarray(counts[order + 1], dtype=np.float64) + float(smoothing)
    table = vector.reshape(width ** order, width)
    return table / table.sum(axis=1, keepdims=True)


def conditional_log_likelihood(sequences: Sequence[str], table: np.ndarray,
                               order: int) -> tuple[float, int]:
    """Mean ``log P(residue | previous `order`)`` in nats per residue, and the n.

    This is the statistic that places a cohort on the corpus's order axis. A
    generator that carries the corpus's statistics up to order k scores like a
    corpus draw at every order up to k and below it above k, which is what makes
    "this control is an order-k generator" a measured claim rather than a
    description of the code that produced it.
    """

    width = len(AA20)
    total = 0.0
    scored = 0
    logs = np.log(table)
    for sequence in sequences:
        if len(sequence) <= order:
            continue
        indices = encode(sequence)
        n = len(indices) - order
        context = np.zeros(n, dtype=np.int64)
        for offset in range(order):
            context = context * width + indices[offset:offset + n]
        total += float(logs[context, indices[order:]].sum())
        scored += n
    if scored == 0:
        return float("nan"), 0
    return total / scored, scored


def qualify_shuffle(parents: Sequence[str], controls: Sequence[str]) -> dict[str, Any]:
    """Exact length and exact residue multiset, on every pair."""

    length_failures = sum(1 for a, b in zip(parents, controls) if len(a) != len(b))
    composition_failures = sum(1 for a, b in zip(parents, controls)
                               if Counter(a) != Counter(b))
    passed = length_failures == 0 and composition_failures == 0
    return {"cohort": "shuffle", "n_pairs": len(parents),
            "declared_statistic": "exact_length_and_exact_residue_multiset",
            "length_failures": length_failures,
            "composition_failures": composition_failures,
            "qualified": passed}


def qualify_markov(order_profile: Mapping[int, float],
                   reference_profile: Mapping[int, float], order: int, *,
                   tolerance: float = ORDER_STATISTIC_TOLERANCE) -> dict[str, Any]:
    """An order-``k`` sampler scores like a corpus draw at ``k`` and below it at ``k + 1``.

    Both profiles are :func:`conditional_log_likelihood` in nats per residue on
    the same evaluation tables. The reference is a cohort drawn from the same
    corpus at the *same lengths* -- the fragment cohort -- rather than the
    corpus's own entropy, and that is what makes the condition noise-matched: a
    cohort scoring a few million positions at order 5 samples 64,000,000 cells
    of that conditional, so any absolute distance there is dominated by sparsity
    and only a like-for-like reference separates the sampler from the sample
    size.

    The two conditions are pre-declared. At the declared order the cohort must
    sit within ``tolerance`` nats of the reference; one order above it must sit
    at least ``tolerance`` nats *below* it. A sampler that passes the first and
    fails the second is carrying more than it declares and is not a point on
    this axis.
    """

    at_order = float(order_profile[order])
    reference_at = float(reference_profile[order])
    record: dict[str, Any] = {
        "cohort": f"markov_{order}", "order": order,
        "declared_statistic": f"corpus_conditional_distribution_at_order_{order}",
        "tolerance_nats_per_residue": tolerance,
        "nats_at_declared_order": at_order,
        "reference_nats_at_declared_order": reference_at,
        "gap_at_declared_order": at_order - reference_at,
    }
    above = order + 1
    if above in order_profile and above in reference_profile:
        gap_above = float(order_profile[above]) - float(reference_profile[above])
        record["nats_one_order_above"] = float(order_profile[above])
        record["reference_nats_one_order_above"] = float(reference_profile[above])
        record["gap_one_order_above"] = gap_above
        record["is_not_a_higher_order_sampler"] = gap_above <= -tolerance
        # The scale-free form of the same condition, measured because the
        # pre-declared absolute margin turned out to exceed one order step of
        # this corpus below order 4 (see ORDER_STATISTIC_TOLERANCE). Going up
        # one order the reference gains something; an order-k sampler should
        # gain none of it. The share it captures is the quantity, and a
        # non-positive share is the amendment's condition.
        reference_step = float(reference_profile[above]) - reference_at
        cohort_step = float(order_profile[above]) - at_order
        record["reference_order_step_nats"] = reference_step
        record["cohort_order_step_nats"] = cohort_step
        record["order_step_share_captured"] = (
            cohort_step / reference_step if reference_step > 0 else None)
        record["qualified_under_the_scale_free_amendment"] = bool(
            abs(record["gap_at_declared_order"]) <= tolerance
            and reference_step > 0 and cohort_step <= 0)
    else:
        record["nats_one_order_above"] = None
        record["gap_one_order_above"] = None
        record["is_not_a_higher_order_sampler"] = None
        record["reference_order_step_nats"] = None
        record["cohort_order_step_nats"] = None
        record["order_step_share_captured"] = None
        record["qualified_under_the_scale_free_amendment"] = False
    record["qualified"] = bool(abs(record["gap_at_declared_order"]) <= tolerance
                               and record["is_not_a_higher_order_sampler"] is True)
    record["amendment_note"] = (
        "qualified is the pre-declared condition and decides the primary verdict; "
        "qualified_under_the_scale_free_amendment is post-hoc and decides the "
        "secondary verdict reported beside it")
    return record


def qualify_fragment(controls: Sequence[str], sources: Sequence[str],
                     lengths: Sequence[int]) -> dict[str, Any]:
    """Every emitted fragment is a contiguous substring of its named record."""

    containment_failures = sum(1 for control, record in zip(controls, sources)
                               if control and control not in record)
    length_failures = sum(1 for control, length in zip(controls, lengths)
                          if len(control) != length)
    return {"cohort": "fragment", "n_sequences": len(controls),
            "declared_statistic": "contiguous_substring_of_a_corpus_record_at_exact_length",
            "containment_failures": containment_failures,
            "length_failures": length_failures,
            "qualified": containment_failures == 0 and length_failures == 0}


def qualify_hydropathy(controls: Sequence[str], targets: Sequence[float],
                       lengths: Sequence[int], *,
                       tolerance: float = HYDROPATHY_TOLERANCE) -> dict[str, Any]:
    """Realized mean hydropathy tracks the target it was tilted to."""

    deviations = [abs(mean_hydropathy(control) - target)
                  for control, target in zip(controls, targets) if control]
    length_failures = sum(1 for control, length in zip(controls, lengths)
                          if len(control) != length)
    mean_absolute = float(np.mean(deviations)) if deviations else None
    return {"cohort": "hydropathy", "n_sequences": len(controls),
            "declared_statistic": "corpus_composition_tilted_to_the_attempt_mean_kyte_doolittle",
            "tolerance_kd_units": tolerance,
            "mean_absolute_deviation_kd": mean_absolute,
            "median_absolute_deviation_kd":
                float(np.median(deviations)) if deviations else None,
            "length_failures": length_failures,
            "qualified": bool(length_failures == 0 and mean_absolute is not None
                              and mean_absolute <= tolerance)}


def qualify_natural(controls: Sequence[str], strata: Sequence[str],
                    stratum_bounds: Mapping[str, tuple[int, int]]) -> dict[str, Any]:
    """Every drawn record is a whole record inside the attempt's length band."""

    band_failures = 0
    for control, stratum in zip(controls, strata):
        if not control:
            continue
        low, high = stratum_bounds[stratum]
        if not low <= len(control) <= high:
            band_failures += 1
    return {"cohort": "natural", "n_sequences": len(controls),
            "declared_statistic": "whole_corpus_record_inside_the_attempt_length_stratum",
            "band_failures": band_failures,
            "qualified": band_failures == 0}


# --------------------------------------------------------------------------
# reading the oracle
# --------------------------------------------------------------------------


def parse_domain_table(path: Any) -> dict[str, list[dict[str, Any]]]:
    """Per-sequence Pfam domain rows from one ``hmmscan --domtblout`` table.

    The same oracle call the programme's profile endpoint already uses, read at
    a finer granularity: the domain table carries the profile's own model length
    and the alignment's coordinates *on the profile*, which is what separates a
    complete domain from a fragment of one. Nothing here changes the threshold,
    the database or the call; the family set this parser yields must equal the
    one the sequence table yields, and the stage checks that it does.
    """

    from pathlib import Path

    hits: dict[str, list[dict[str, Any]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=22)
        if len(fields) < 23:
            continue
        accession = fields[1]
        record = {
            "accession": accession,
            "accession_unversioned": accession.split(".", 1)[0],
            "target_name": fields[0],
            "target_length": int(fields[2]),
            "query_length": int(fields[5]),
            "domain_score": float(fields[13]),
            "hmm_from": int(fields[15]),
            "hmm_to": int(fields[16]),
            "ali_from": int(fields[17]),
            "ali_to": int(fields[18]),
        }
        record["profile_coverage"] = (
            (record["hmm_to"] - record["hmm_from"] + 1) / record["target_length"]
            if record["target_length"] > 0 else 0.0)
        hits.setdefault(fields[3], []).append(record)
    return hits


def best_profile_coverage(domains: Sequence[Mapping[str, Any]]) -> float | None:
    """The largest fraction of any assigned profile's model length that is aligned.

    The maximum is taken over domain *instances*, not over their union on one
    profile: the requirement being measured is that the sequence carries one
    complete copy of a curated domain, and two half-alignments to the same
    profile are not one complete domain.
    """

    if not domains:
        return None
    return max(float(domain["profile_coverage"]) for domain in domains)


def collect_oracle(workspace: Any) -> dict[str, dict]:
    """Per-query family set and best profile coverage, from one oracle call's tables.

    The sequence table decides which families a query carries, exactly as the
    existing profile endpoint does, and the domain table supplies the
    profile-side coverage the complete-domain endpoint reads. The two tables are
    not the same set and are not treated as one: HMMER reports a family on the
    sequence table when the full-sequence score clears the release's GA1 cut,
    and writes a domain row only when a single domain also clears GA2, so a
    family can be present on the sequence table with no domain row at all.
    That is a fact about the oracle rather than a disagreement, and it is
    counted per query. What must always hold is containment -- every family with
    a domain row is a family the sequence table carries -- and a query that
    breaks it is refused, because then the finer read would not be the same
    measurement as the coarser one.
    """

    import json
    from pathlib import Path

    from ..core.protein_annotations import parse_hmmscan_table

    workspace = Path(workspace)
    names = json.loads((workspace / "query_names.json").read_text(encoding="utf-8"))
    reverse = {value: key for key, value in names.items()}
    manifest = json.loads((workspace / "build_manifest.json").read_text(encoding="utf-8"))
    tables: list[Path] = []
    incomplete: list[str] = []
    for shard in manifest["shards"]:
        stem = Path(shard["path"]).stem
        tbl = workspace / "oracle" / f"{stem}.tbl"
        marker = Path(str(tbl) + ".done")
        if not (tbl.is_file() and tbl.with_suffix(".domtbl").is_file() and marker.is_file()):
            incomplete.append(stem)
            continue
        if json.loads(marker.read_text(encoding="utf-8")).get("fasta_sha256") != shard["sha256"]:
            incomplete.append(stem)
            continue
        tables.append(tbl)
    if incomplete:
        # A shard whose oracle call has not terminated leaves a truncated table
        # on disk, and reading one would turn an unfinished scan into a low
        # recognition rate. The endpoint is refused until every declared shard
        # binds its own query file.
        raise RuntimeError(
            f"{len(incomplete)} of {len(manifest['shards'])} oracle shards have no "
            f"terminal receipt binding their query file, first: {incomplete[:3]}")
    out: dict[str, dict] = {}
    uncontained: list[str] = []
    for tbl in tables:
        sequence_hits = parse_hmmscan_table(tbl)
        domain_hits = parse_domain_table(tbl.with_suffix(".domtbl"))
        for query, entries in sequence_hits.items():
            families = sorted({entry["accession_unversioned"] for entry in entries})
            domains = domain_hits.get(query, [])
            domain_families = sorted({domain["accession_unversioned"] for domain in domains})
            if not set(domain_families) <= set(families):
                uncontained.append(query)
            out[reverse[query]] = {
                "families": families,
                "families_with_a_domain_row": domain_families,
                "n_families_without_a_domain_row": len(families) - len(domain_families),
                "best_profile_coverage": best_profile_coverage(domains),
            }
    if uncontained:
        raise RuntimeError(
            f"{len(uncontained)} queries carry a domain-table family the sequence "
            f"table does not, which cannot happen for one oracle call, first: "
            f"{uncontained[:3]}")
    return out


# --------------------------------------------------------------------------
# estimation
# --------------------------------------------------------------------------


def paired_rate_contrast(model: Sequence[bool], control: Sequence[bool],
                         groups: Sequence[str], *, seed: int = GATE_SEED,
                         resamples: int = RESAMPLES) -> dict[str, Any]:
    """Model-minus-control recognition rate, resampling the attempt's own group.

    The unit is the ledger's frozen near-duplicate sequence group, so a batch of
    near-identical attempts contributes once rather than as independent
    biological replicates -- the same safeguard the conditioned-generation
    campaign applied, reused rather than redeclared. The pair is the attempt:
    every attempt of the ledger carries one model outcome and one control
    outcome, including the attempts whose output no oracle could search, which
    are ``False`` on both sides and stay in the denominator.

    The interval describes variation across the 800 attempts of one arm at one
    decoding configuration and one batch-seed stream. It does not describe
    uncertainty over the decoding configuration, over the seed stream as a
    whole, or over retraining.
    """

    if not (len(model) == len(control) == len(groups)):
        raise ValueError("model, control and group vectors must have equal length")
    if not model:
        raise ValueError("a contrast needs at least one attempt")
    order: dict[str, int] = {}
    for group in groups:
        order.setdefault(group, len(order))
    totals = np.zeros((len(order), 3), dtype=float)
    for hit, control_hit, group in zip(model, control, groups):
        row = totals[order[group]]
        row[0] += float(bool(hit))
        row[1] += float(bool(control_hit))
        row[2] += 1.0
    result: dict[str, Any] = {
        "n_attempts": int(totals[:, 2].sum()),
        "n_clusters": len(order),
        "model_rate": float(totals[:, 0].sum() / totals[:, 2].sum()),
        "control_rate": float(totals[:, 1].sum() / totals[:, 2].sum()),
        "unit": "frozen_near_duplicate_sequence_group_of_the_attempt_ledger",
        "seed": seed, "resamples": resamples,
        "scope": ("one decoding configuration and one batch-seed stream; not "
                  "decoding-configuration, seed-stream or retraining uncertainty"),
    }
    result["difference"] = result["model_rate"] - result["control_rate"]
    # The shared floor, imported rather than restated, so that a change to it
    # reaches this endpoint. The status string is a frozen artefact key and
    # stays spelled out.
    if len(order) < MINIMUM_BOOTSTRAP_UNITS:
        result["ci95"] = None
        result["ci97_5"] = None
        result["interval_status"] = "fewer_than_eight_sequence_groups"
        return result
    rng = np.random.default_rng(seed)
    draws = totals[rng.integers(0, len(order), (resamples, len(order)))].sum(axis=1)
    differences = (draws[:, 0] - draws[:, 1]) / draws[:, 2]
    result["ci95"] = np.quantile(differences, [0.025, 0.975]).tolist()
    result["ci97_5"] = np.quantile(differences, [0.0125, 0.9875]).tolist()
    result["interval_status"] = "attempt_cluster_bootstrap"
    return result


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054
                    ) -> tuple[float, float]:
    """Wilson 95% interval on a proportion, for an all-attempt denominator."""

    if trials <= 0:
        raise ValueError("a proportion needs a positive denominator")
    phat = successes / trials
    denominator = 1 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)
