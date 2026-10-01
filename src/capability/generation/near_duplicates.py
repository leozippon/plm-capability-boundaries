"""The unit of independence in a corpus is not the record.

Why this module exists
======================

A stage that fits a map on one split and reports it on another needs the two
splits to be *independent*, and it usually enforces that by checking that no
record appears on both sides. On a web corpus that check is the whole of the
property. On a protein corpus it is a small part of it, and the gap was measured
rather than argued.

Swiss-Prot is non-redundant at the level of the *entry* -- one entry per protein
per organism -- and not at the level of the *sequence*. In the 10,240-record pool
``../d2/scripts/capability/stages/model_diffing_baselines.py`` draws in band 32-507 at skip
0, 1,289 records are byte-identical to another record of the same pool: the
acyl carrier protein ``acpP`` appears 41 times, once for each sequenced
enterobacterial strain, byte for byte. The exact-string check therefore fires,
and the stage refuses -- which is correct, and which no skip offset repairs
(distinct fraction 0.874, 0.865, 0.879, 0.854, 0.905 at skips 0, 4,096, 10,240,
20,480, 40,960).

The number that decides the design is the next one. Deduplicating the pool would
let the check pass while leaving most of the leakage in place: on the same pool,
searched with DIAMOND at ``--masking 0 --very-sensitive``, **41.4% of the
held-out records have a relative in the training split at 95% or higher identity
over the shorter sequence, and only 17.4% are exact**. Removing exact duplicates
removes at most the second number. A map fitted on the training positions would
still be reported on positions whose sequence it has effectively already seen,
and the bias is not symmetric between the two pairings the stage reports: the
shuffled-pairing null destroys position correspondence, so homology leakage
inflates the true-pairing fit alone and widens the gap that decides whether a
Crosscoder is warranted.

Drawing from a corpus whose redundancy is already controlled helps and does not
finish the job. The same measurement on a UniRef50 pool of the same size and
band gives zero exact duplicates at all five skips and a median maximum identity
of 10.3% -- but still 36 of 2,048 held-out records at 95% or higher and 4 at
100%, because UniRef50's clustering bounds the distance from a member to its own
representative and not the distance between two representatives. The corpus
choice moves the leakage by a factor of twenty; only splitting at the level of
the near-duplicate group removes it.

What is grouped, and what is deliberately not
=============================================

The relation is **near-duplication**, not homology. Two records are joined when
one's shingles are largely contained in the other's; groups are the connected
components. Remote and even close homologues stay free to fall on opposite sides
of the split, and the residual is measured and reported rather than gated.

That boundary is where it is because of this programme's own rule that no gate is
applied to a protein arm until it is shown attainable on the text control under
the same procedure. A near-duplicate gate is attainable on text: on the
10,240-document OpenWebText pool the completed text-mode cell used, no document
pair reaches containment 0.5, the largest word-13-gram containment of any
held-out document in the whole training side is 0.336, and 92.7% of held-out
documents share not one 13-gram with it. A *homology* gate has no text analogue
at all -- there is no such thing as a remote homologue of a web page -- so gating
remote homology would make the two modes incomparable in the opposite direction,
by holding the protein side to a criterion the text side is not even defined
under.

The instrument, and how its threshold was calibrated
====================================================

Containment of shingle sets, ``|A & B| / min(|A|, |B|)``, and not Jaccard.
Jaccard was tried first because :func:`src.capability.relational.kmer_set` and
:func:`src.capability.core.families.boundary_leakage` already use it, and it fails here:
it penalises length asymmetry, which is exactly the shape a short protein
contained in a longer one has. Against DIAMOND identity over the shorter
sequence on 60,000 aligned pairs from the two pools above, no Jaccard threshold
separates the classes -- at k=5 the best operating point still misses 1.3% of the
protein pool's near-duplicate pairs and 11.3% of UniRef50's, and the miss rate
rises to 55% at threshold 0.7.

Containment separates them completely. At :data:`RESIDUE_SHINGLE` = 5 the
*minimum* containment over every DIAMOND pair at 95% identity or above is 0.732
on the Swiss-Prot pool and 0.742 on the UniRef50 pool, while the 99th percentile
over all pairs below 95% is 0.754 and 0.438. Every threshold in 0.3-0.7
therefore misses none of them on either corpus, which is what
:data:`NEAR_DUPLICATE_CONTAINMENT` sits in the middle of: the ordering is
invariant across the sweep, so the threshold is a declared choice inside a flat
region rather than a tuned one. What it costs is over-merging, which is the safe
direction -- at 0.5, 4.9% of the sub-95% pairs are joined as well, so some close
homologues are kept together on one side that need not have been.

The calibration is complete for the class it covers: a pair DIAMOND does not
align at ``--evalue 1e-3`` is not a 95%-identity pair, so it cannot be a miss
this instrument is responsible for.

Alignment-free on purpose. The grouping runs inside a stage, on a compute node
where no aligner is staged and where nothing may be installed, so it is built
from shingle sets and an inverted index alone; DIAMOND appears here only as the
external standard the threshold was calibrated against, once, offline.

Not to be confused with :mod:`.families`, which partitions a cohort by
*externally curated* family labels so that a method can be said to generalise to
an unseen family. That is a different and stricter property, it needs a label
table keyed on accession, and it is not available for a raw corpus stream whose
records arrive without accessions. This module answers the narrower question a
fit/held-out split has to answer first.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.sequence_groups import homology_disjoint_split

#: Residue shingle length, and the one the calibration above was run at.
RESIDUE_SHINGLE = 5

#: Word shingle length for a text record. Five words of English carry rather more
#: than five residues of protein, so the two are not the same criterion and are
#: not claimed to be; each is its own modality's near-duplicate relation, and what
#: has to hold in common is that the gate is attainable on both (module docstring).
WORD_SHINGLE = 5

#: Containment at or above which two records are one unit. The middle of the
#: 0.3-0.7 region over which the calibration misses nothing.
NEAR_DUPLICATE_CONTAINMENT = 0.5

#: The symbol units :func:`shingles` accepts, keyed by the name a corpus already
#: declares for what it is made of, so a caller passes the fact it already has
#: rather than a second name for it.
SHINGLE_UNITS: dict[str, int] = {"residues": RESIDUE_SHINGLE, "characters": WORD_SHINGLE}

_WORD = re.compile(r"\w+")


def shingles(record: str, *, unit: str, length: int | None = None) -> frozenset[str]:
    """The shingle set of one record, in the unit its corpus is made of.

    ``residues`` shingles the string itself, which is what a protein record is.
    ``characters`` -- the unit a text corpus declares -- shingles *words*, because
    a character shingle of English measures spelling: two unrelated documents
    share nearly every 5-character run, so their containment is near one and every
    record joins one group.
    """

    if unit not in SHINGLE_UNITS:
        raise ValueError(f"unknown symbol unit {unit!r}; declared: {sorted(SHINGLE_UNITS)}")
    size = SHINGLE_UNITS[unit] if length is None else int(length)
    if size < 1:
        raise ValueError("shingle length must be positive")
    tokens: Sequence[str] = record if unit == "residues" else _WORD.findall(record.lower())
    if len(tokens) < size:
        return frozenset()
    if unit == "residues":
        return frozenset(record[i : i + size] for i in range(len(record) - size + 1))
    return frozenset(
        " ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)
    )


def near_duplicate_groups(
    records: Sequence[str],
    *,
    unit: str,
    containment: float = NEAR_DUPLICATE_CONTAINMENT,
    shingle: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Connected components under "one record's shingles are contained in another's".

    Candidate pairs come from an inverted index over the shingles, so the cost is
    the number of co-occurrences rather than the ``n^2`` pairs: at 10,240 protein
    records that is 20.1M increments on the Swiss-Prot pool and 0.19M on the
    UniRef50 pool, seconds either way, against 52M set intersections for the
    exhaustive form.

    A record too short to carry a shingle is its own group. It is not silently
    joined to everything and it is not dropped: dropping would change the
    population the caller declared, and the count is reported instead.

    Component ids are allocated in record order, so the grouping is a function of
    the records alone -- the same property :func:`.families._connected_groups`
    holds, and for the same reason.
    """

    if not records:
        raise ValueError("a grouping needs at least one record")
    if not 0.0 < float(containment) <= 1.0:
        raise ValueError("containment must lie in (0, 1]")

    sets = [shingles(record, unit=unit, length=shingle) for record in records]
    index: dict[str, list[int]] = {}
    for position, entry in enumerate(sets):
        for gram in entry:
            index.setdefault(gram, []).append(position)

    parent = list(range(len(records)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    shared: dict[tuple[int, int], int] = {}
    for postings in index.values():
        if len(postings) < 2:
            continue
        for position, left in enumerate(postings):
            for right in postings[position + 1 :]:
                key = (left, right)
                shared[key] = shared.get(key, 0) + 1

    joined = 0
    for (left, right), count in shared.items():
        smaller = min(len(sets[left]), len(sets[right]))
        if smaller and count / smaller >= containment:
            if find(left) != find(right):
                joined += 1
            union(left, right)

    roots: dict[int, int] = {}
    groups = np.empty(len(records), dtype=np.int64)
    for position in range(len(records)):
        root = find(position)
        if root not in roots:
            roots[root] = len(roots)
        groups[position] = roots[root]

    _, sizes = np.unique(groups, return_counts=True)
    summary = {
        "unit": unit,
        "shingle_length": SHINGLE_UNITS[unit] if shingle is None else int(shingle),
        "containment_threshold": float(containment),
        "n_records": len(records),
        "n_groups": int(sizes.size),
        "n_singleton_groups": int((sizes == 1).sum()),
        "largest_group_size": int(sizes.max()),
        "largest_group_share": float(sizes.max() / len(records)),
        "n_records_without_shingles": int(sum(1 for entry in sets if not entry)),
        "candidate_pairs": len(shared),
        "edges_joining_groups": joined,
        "relation": (
            "records joined when |A & B| / min(|A|, |B|) of their shingle sets "
            "reaches the threshold, then connected components. Calibrated against "
            "DIAMOND identity over the shorter sequence at the 95% boundary; see "
            "src.capability.generation.near_duplicates"
        ),
    }
    return groups, summary


def group_disjoint_split(
    groups: np.ndarray,
    *,
    n_train: int,
    seed: int,
    fraction_tolerance: float = 0.02,
) -> tuple[np.ndarray, dict[str, Any]]:
    """A training mask that never divides a near-duplicate group, or a refusal.

    The mask itself is :func:`src.capability.relational.homology_disjoint_split`,
    which is this package's one implementation of "a mask that never splits a
    group" and which :func:`.families.family_disjoint_split` also builds on.
    What is added is the same pair of checks that function adds, for the same
    reason: the disjointness is re-verified on the returned mask rather than
    trusted from the construction, and the achieved fraction has to be near the
    requested one.

    The second check is what makes a refusal possible at all. Whole groups are
    indivisible, so a pool dominated by one near-duplicate group cannot be split
    at the requested fraction, and the honest outcome is to say so rather than to
    report a split that is not the one asked for. Widening the tolerance to make
    a run start reports a fraction it did not achieve.
    """

    groups = np.asarray(groups)
    total = int(groups.size)
    if not 0 < int(n_train) < total:
        raise ValueError(
            f"n_train {n_train} must be strictly between 0 and the pool size {total}"
        )
    if fraction_tolerance <= 0.0:
        raise ValueError("fraction_tolerance must be positive")

    requested = int(n_train) / total
    train = homology_disjoint_split(
        groups, train_fraction=requested, seed=int(seed), min_side=1
    )
    shared = np.intersect1d(np.unique(groups[train]), np.unique(groups[~train]))
    if shared.size:
        raise RuntimeError(
            f"{shared.size} near-duplicate groups appear on both sides of a split "
            "that must not divide one; the mask is not group-disjoint"
        )
    achieved = float(train.sum() / total)
    if abs(achieved - requested) > fraction_tolerance:
        largest = int(np.bincount(groups).max())
        raise RuntimeError(
            f"a near-duplicate-disjoint split put {achieved:.4f} of the pool on the "
            f"training side against a requested {requested:.4f}, outside the "
            f"{fraction_tolerance:.4f} tolerance; the largest near-duplicate group "
            f"holds {largest} of {total} records, so this pool cannot be partitioned "
            "at the requested fraction without dividing one. Draw a larger pool, or "
            "draw from a corpus whose redundancy is controlled -- do not widen the "
            "tolerance, which reports a split that was not achieved"
        )
    summary = {
        "verdict": "GROUP_DISJOINT",
        "seed": int(seed),
        "requested_train_fraction": requested,
        "achieved_train_fraction": achieved,
        "fraction_tolerance": float(fraction_tolerance),
        "n_train_records": int(train.sum()),
        "n_eval_records": int((~train).sum()),
        "n_train_groups": int(np.unique(groups[train]).size),
        "n_eval_groups": int(np.unique(groups[~train]).size),
        "split_mask_source": (
            "src.capability.relational.homology_disjoint_split, the one "
            "implementation of a mask that never divides a group"
        ),
    }
    return train, summary


def boundary_containment(
    records: Sequence[str],
    train: np.ndarray,
    *,
    unit: str,
    shingle: int | None = None,
) -> dict[str, Any]:
    """Measure, rather than assert, how similar the two sides still are.

    The grouping is a construction and this is the reading taken off the result,
    which is the distinction :func:`.families.boundary_leakage` draws for a
    curated split and the one Appendix B rule 24 asks for generally. Reported per
    held-out record as its maximum containment in any training record, so the
    statistic is threshold-free and a reader can put the cut wherever they like.

    Exhaustive over held-out records but not over pairs: the same inverted index
    that generated the grouping's candidates supplies every training record that
    shares a shingle, and a training record sharing none has containment zero.
    """

    train = np.asarray(train, dtype=bool)
    if train.shape != (len(records),):
        raise ValueError("mask must be one boolean per record")
    sets = [shingles(record, unit=unit, length=shingle) for record in records]
    index: dict[str, list[int]] = {}
    for position in np.flatnonzero(train):
        for gram in sets[int(position)]:
            index.setdefault(gram, []).append(int(position))

    best = np.zeros(int((~train).sum()), dtype=np.float64)
    for slot, position in enumerate(np.flatnonzero(~train)):
        entry = sets[int(position)]
        counts: dict[int, int] = {}
        for gram in entry:
            for other in index.get(gram, ()):
                counts[other] = counts.get(other, 0) + 1
        for other, count in counts.items():
            smaller = min(len(entry), len(sets[other]))
            if smaller:
                best[slot] = max(best[slot], count / smaller)
    return {
        "statistic": (
            "per held-out record, the maximum containment of its shingle set in "
            "any single training record; threshold-free"
        ),
        "n_held_out": int(best.size),
        "max": float(best.max()) if best.size else 0.0,
        "q99": float(np.quantile(best, 0.99)) if best.size else 0.0,
        "median": float(np.median(best)) if best.size else 0.0,
        "n_above_half": int((best >= 0.5).sum()),
        "n_above_threshold": int((best >= NEAR_DUPLICATE_CONTAINMENT).sum()),
        "n_with_no_shared_shingle": int((best == 0.0).sum()),
    }


def screen_against_training_stream(
    candidates: Sequence[str],
    training: Iterable[str],
    *,
    unit: str,
    containment: float = NEAR_DUPLICATE_CONTAINMENT,
    shingle: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Which held-out candidates no training record is a near-duplicate of.

    The same relation as :func:`near_duplicate_groups` and the same threshold,
    asked of a stream rather than of a pool. A stage that draws its held-out set
    by *skipping past* the training budget -- ``train_transcoder.py`` does --
    has no pool to group: the training side is hundreds of thousands of records
    read once, and materialising it to group it costs more memory than the run
    it protects. So the index is built over the **candidates**, which number in
    the hundreds, and the training records are streamed against it. The
    arithmetic is identical to :func:`boundary_containment`'s; only which side is
    resident differs.

    Returns a keep-mask over ``candidates`` in draw order and a summary carrying
    the threshold-free per-candidate maximum containment, so the cut is
    inspectable rather than only applied.

    ``training`` is consumed exactly once. A candidate too short to carry a
    shingle can never reach the threshold and is kept, which is the same
    treatment :func:`near_duplicate_groups` gives it -- its own group -- and the
    count is reported rather than left to be inferred.
    """

    if not 0.0 < float(containment) <= 1.0:
        raise ValueError("containment must lie in (0, 1]")
    if not candidates:
        raise ValueError("a screen needs at least one candidate")

    sets = [shingles(record, unit=unit, length=shingle) for record in candidates]
    index: dict[str, list[int]] = {}
    for position, entry in enumerate(sets):
        for gram in entry:
            index.setdefault(gram, []).append(position)

    best = np.zeros(len(candidates), dtype=np.float64)
    n_training = 0
    for record in training:
        n_training += 1
        entry = shingles(record, unit=unit, length=shingle)
        if not entry:
            continue
        counts: dict[int, int] = {}
        for gram in entry:
            for position in index.get(gram, ()):
                counts[position] = counts.get(position, 0) + 1
        for position, count in counts.items():
            smaller = min(len(entry), len(sets[position]))
            if smaller:
                best[position] = max(best[position], count / smaller)

    keep = best < float(containment)
    return keep, {
        "verdict": "SCREENED_AGAINST_TRAINING_STREAM",
        "relation": (
            "a candidate is dropped when any single training record reaches "
            "|A & B| / min(|A|, |B|) of the containment threshold with it -- the "
            "relation near_duplicate_groups joins on, applied one-sided because "
            "the training side is a stream and not a pool"
        ),
        "unit": unit,
        "shingle_length": SHINGLE_UNITS[unit] if shingle is None else int(shingle),
        "containment_threshold": float(containment),
        "n_candidates": len(candidates),
        "n_kept": int(keep.sum()),
        "n_dropped": int((~keep).sum()),
        "n_training_records_screened": n_training,
        "n_candidates_without_shingles": int(sum(1 for entry in sets if not entry)),
        "max_containment": float(best.max()),
        "q99_containment": float(np.quantile(best, 0.99)),
        "median_containment": float(np.median(best)),
        "n_with_no_shared_shingle": int((best == 0.0).sum()),
    }


#: Sequential fill that walks one ordered stream and assigns each accepted
#: record to the first still-open slot that has no exact or near-duplicate edge
#: to an earlier slot. Within a slot, near-duplicates are kept: they are the
#: bootstrap groups, not a second population.
GROUP_DISJOINT_FILL_ALGORITHM = "sequential_fill_against_earlier_slots"
GROUP_DISJOINT_FILL_VERSION = "group_disjoint_fill"
ELIGIBLE_CORPUS_EXHAUSTED = "ELIGIBLE_CORPUS_EXHAUSTED"


class EligibleCorpusExhausted(RuntimeError):
    """The ordered stream ended before every requested slot was full."""

    def __init__(self, message: str, *, detail: dict[str, Any]) -> None:
        super().__init__(message)
        self.detail = detail
        self.reason = ELIGIBLE_CORPUS_EXHAUSTED


@dataclass(frozen=True)
class GroupDisjointSlot:
    """One filled slot of a sequential group-disjoint draw."""

    name: str
    records: tuple[str, ...]
    source_positions: tuple[int, ...]
    labels: tuple[Any, ...] | None
    rejected_exact: int
    rejected_near: int

    def record(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "n_records": len(self.records),
            "source_positions": list(self.source_positions),
            "rejected_exact": int(self.rejected_exact),
            "rejected_near": int(self.rejected_near),
        }
        if self.labels is not None:
            payload["n_labels"] = len(self.labels)
        return payload


@dataclass(frozen=True)
class GroupDisjointFill:
    """The complete sequential fill: slots, rejections, and the algorithm identity."""

    slots: tuple[GroupDisjointSlot, ...]
    n_eligible: int
    n_scanned: int
    rejected_exact: int
    rejected_near: int
    algorithm: str
    version: str
    containment_threshold: float
    shingle_length: int
    unit: str

    def record(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "algorithm_version": self.version,
            "containment_threshold": float(self.containment_threshold),
            "shingle_length": int(self.shingle_length),
            "unit": self.unit,
            "n_eligible": int(self.n_eligible),
            "n_scanned": int(self.n_scanned),
            "rejected_exact": int(self.rejected_exact),
            "rejected_near": int(self.rejected_near),
            "slots": [slot.record() for slot in self.slots],
        }


class _EarlierCohortIndex:
    """Exact strings plus an inverted 5-mer index of every earlier-slot record.

    A later candidate is rejected when it is byte-identical to an earlier
    record or when any earlier record reaches shingle containment at the
    declared threshold. Cost is the candidate's shingles times the posting-list
    length, not the product of corpus size and accepted-record count.
    """

    def __init__(self, *, unit: str, containment: float, shingle: int | None) -> None:
        self.unit = unit
        self.containment = float(containment)
        self.shingle = shingle
        self.exact: set[str] = set()
        self.sets: list[frozenset[str]] = []
        self.index: dict[str, list[int]] = {}

    def reject_reason(self, record: str) -> str | None:
        if record in self.exact:
            return "exact"
        grams = shingles(record, unit=self.unit, length=self.shingle)
        if not grams:
            return None
        counts: dict[int, int] = {}
        for gram in grams:
            for position in self.index.get(gram, ()):
                counts[position] = counts.get(position, 0) + 1
        for position, shared in counts.items():
            smaller = min(len(grams), len(self.sets[position]))
            if smaller and shared / smaller >= self.containment:
                return "near"
        return None

    def add(self, record: str) -> None:
        self.exact.add(record)
        grams = shingles(record, unit=self.unit, length=self.shingle)
        position = len(self.sets)
        self.sets.append(grams)
        for gram in grams:
            self.index.setdefault(gram, []).append(position)


def fill_group_disjoint_slots(
    records: Sequence[str],
    *,
    slot_sizes: Sequence[int],
    slot_names: Sequence[str],
    source_positions: Sequence[int],
    labels: Sequence[Any] | None = None,
    containment: float = NEAR_DUPLICATE_CONTAINMENT,
    shingle: int | None = None,
    unit: str = "residues",
    algorithm: str = GROUP_DISJOINT_FILL_ALGORITHM,
    version: str = GROUP_DISJOINT_FILL_VERSION,
) -> GroupDisjointFill:
    """Fill named slots in order from one already-permuted stream.

    The first slot accepts every record until it is full. Each later slot
    rejects a record that is exactly identical to, or a 5-mer near-duplicate of,
    any record already accepted into an earlier slot. Near-duplicates inside a
    slot are retained. The stream is not reshuffled and no replacement seed is
    tried: if it ends before every slot is full the call raises
    :class:`EligibleCorpusExhausted`.
    """

    if len(slot_sizes) != len(slot_names):
        raise ValueError("slot_sizes and slot_names must have the same length")
    if not slot_sizes:
        raise ValueError("at least one slot is required")
    if any(int(size) < 1 for size in slot_sizes):
        raise ValueError("each slot must request at least one record")
    if len(source_positions) != len(records):
        raise ValueError("source_positions must align with the ordered records")
    if labels is not None and len(labels) != len(records):
        raise ValueError("labels must align with the ordered records")
    if not 0.0 < float(containment) <= 1.0:
        raise ValueError("containment must lie in (0, 1]")

    sizes = [int(size) for size in slot_sizes]
    names = [str(name) for name in slot_names]
    earlier = _EarlierCohortIndex(unit=unit, containment=containment, shingle=shingle)
    filled: list[GroupDisjointSlot] = []
    current_records: list[str] = []
    current_positions: list[int] = []
    current_labels: list[Any] = []
    rejected_exact = 0
    rejected_near = 0
    slot_exact = 0
    slot_near = 0
    slot_index = 0
    scanned = 0

    def close_slot() -> None:
        nonlocal slot_index, slot_exact, slot_near
        label_tuple = tuple(current_labels) if labels is not None else None
        filled.append(
            GroupDisjointSlot(
                name=names[slot_index],
                records=tuple(current_records),
                source_positions=tuple(current_positions),
                labels=label_tuple,
                rejected_exact=slot_exact,
                rejected_near=slot_near,
            )
        )
        for accepted in current_records:
            earlier.add(accepted)
        current_records.clear()
        current_positions.clear()
        current_labels.clear()
        slot_exact = 0
        slot_near = 0
        slot_index += 1

    for offset, record in enumerate(records):
        if slot_index >= len(sizes):
            break
        scanned += 1
        reason = earlier.reject_reason(record) if slot_index else None
        if reason == "exact":
            rejected_exact += 1
            slot_exact += 1
            continue
        if reason == "near":
            rejected_near += 1
            slot_near += 1
            continue
        current_records.append(record)
        current_positions.append(int(source_positions[offset]))
        if labels is not None:
            current_labels.append(labels[offset])
        if len(current_records) == sizes[slot_index]:
            close_slot()

    if slot_index < len(sizes):
        raise EligibleCorpusExhausted(
            f"eligible corpus cannot fill slot {names[slot_index]!r}: "
            f"need {sizes[slot_index]}, accepted {len(current_records)}, "
            f"scanned {scanned} of {len(records)} "
            f"({ELIGIBLE_CORPUS_EXHAUSTED})",
            detail={
                "reason": ELIGIBLE_CORPUS_EXHAUSTED,
                "failed_slot": names[slot_index],
                "needed": sizes[slot_index],
                "accepted_in_slot": len(current_records),
                "filled_slots": [slot.name for slot in filled],
                "n_scanned": scanned,
                "n_eligible": len(records),
                "rejected_exact": rejected_exact,
                "rejected_near": rejected_near,
            },
        )

    return GroupDisjointFill(
        slots=tuple(filled),
        n_eligible=len(records),
        n_scanned=scanned,
        rejected_exact=rejected_exact,
        rejected_near=rejected_near,
        algorithm=str(algorithm),
        version=str(version),
        containment_threshold=float(containment),
        shingle_length=SHINGLE_UNITS[unit] if shingle is None else int(shingle),
        unit=unit,
    )
