"""Family-disjoint splits over externally curated protein families.

Why this is one component and not a line in each stage
=====================================================

Every pending part-3 item ends in the same sentence: *and it reproduces on an
unseen protein family*. This programme has already recorded what happens when
that sentence is asserted rather than built. ``ConceptSpec.family_disjoint`` was
declared ``True`` for the fitness concept while ProteinGym ships four
``BLAT_ECOLX`` assays of one 286-residue protein, so the field said
"family-disjoint" and the data said "forty mutants of one protein" (§0.05 of the
audit). The defect was not the value of a flag; it was that no code computed the
property the flag claimed. Here the property is computed, is refused when it
does not hold, and is *measured* rather than asserted.

What a split from this module does and does not control
=======================================================

It controls leakage **between our own fitting side and our own evaluation
side**: no curated family is represented on both. That is the split under which
"the method generalises beyond the families it was tuned on" is a statement
about the method.

It does **not** control leakage into the *model's* pretraining corpus, and
nothing that can be built from public protein data would. ProtGPT2 was trained
on UniRef50 and Swiss-Prot lies inside it by construction, so every evaluation
protein this repository can reach is a candidate training member whatever family
it belongs to. That question has its own instruments -- :mod:`.homology` for the
identity stratification and :mod:`.profiles` for the retrieval bound -- and a
family-disjoint split neither replaces nor weakens them. A result quoted from
this module's split must say which of the two leaks it has closed.

The two label sources, and why neither is the other
===================================================

*Pfam family* (``pfam``), from the residue-span table :mod:`.channels` already
reads. Cheap, keyed on UniProt accession, and covers 523,433 Swiss-Prot entries.
Two proteins are in one group when they share a Pfam domain family, so the unit
is *domain-family* membership.

*CATH superfamily* (``cath_superfamily``), from the ``G3DSA:`` Gene3D signatures
in InterPro's ``protein2ipr`` dump. Gene3D assigns CATH superfamilies, so the
unit is *structural* superfamily membership, which groups remote homologues that
Pfam separates and is therefore the stricter of the two. It has to be extracted
once; see :func:`extract_cath_superfamilies`.

Running a result under both is the split-definition analogue of Appendix B rule
17: the grouping unit is a free choice, so the ordering has to survive changing
it.

``MODEL_BASE_DIR/datasets/cath-s40`` cannot supply the second source
==================================================================

Recorded here because the resource inventory that motivated this module said it
could. The staged dump holds three files -- a 34,653-line list of CATH **domain
identifiers**, the matching ATOM-record FASTA, and the domain coordinate
tarball. A CATH domain identifier (``12asA00``) is a PDB code, a chain and a
domain index; **it does not encode the C.A.T.H superfamily**, and no
``CathDomainList`` is staged beside it. The dump also carries no UniProt
accession, so it does not join to Swiss-Prot, AlphaFold, Pfam, PhosphoSite or
ProteinGym, all of which this programme keys on accession.

What it *is* good for is a guarantee this module cannot provide: its records are
pairwise below 40% sequence identity, so any split of it is identity-disjoint at
40% without a grouping at all. Making it superfamily-disjoint needs a ~4 MB
``CathDomainList`` fetch that has not been made; until then ``cath_superfamily``
here comes from InterPro and is keyed on accession.

Composition, for a three-way split
==================================

The role a lens needs is usually three-way -- fit the translator, choose the
layer, report -- and this module deliberately exposes only a two-way split.
Split once, then call :meth:`FamilyAssignment.select` on the training side and
split that. Because no group is ever divided, all three sides are pairwise
family-disjoint; ``tests/test_family_splits.py`` holds that as an executable
property rather than as this paragraph.

Not to be confused with :func:`statistics.make_group_splits`, which builds
grouped *cross-validation folds* for a supervised probe and needs labels. This
module builds one unlabelled fit/held-out partition for a method that is fitted
on one population and reported on another.
"""

from __future__ import annotations

import gzip
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .arms import REPO, SWISSPROT_FASTA, env_path, iter_fasta, require_input_path
from .sequence_metadata import PFAM_RESIDUE_TSV, load_pfam_spans
from .io import sha256_file
from .sequence_metadata import swissprot_accession
from .sequence_groups import homology_disjoint_split, kmer_set
from .statistics import MINIMUM_BOOTSTRAP_UNITS

#: InterPro's protein-to-entry dump. Every InterPro member signature for every
#: UniProt accession, which is where the Gene3D/CATH assignments live.
PROTEIN2IPR_DAT = env_path(
    "TRANSFER_PROTEIN2IPR", REPO / "data/interpro/protein2ipr.dat.gz"
)

#: The accession-to-CATH-superfamily table :func:`extract_cath_superfamilies`
#: derives from it. Small enough to load per stage; the dump is not.
CATH_SUPERFAMILY_TSV = env_path(
    "TRANSFER_CATH_SUPERFAMILY_TSV", REPO / "data/interpro/cath_superfamily.tsv"
)

CATH_TSV_HEADER = ("uniprot", "cath_superfamily")

#: The InterPro member-database prefix that carries CATH superfamily codes.
G3DSA_PREFIX = "G3DSA:"

#: ``protein2ipr`` is headerless with six tab-separated columns.
PROTEIN2IPR_COLUMNS = 6

FAMILY_SOURCES = ("pfam", "cath_superfamily")

#: What to do with a unit carrying more than one curated family. Both settings
#: are leak-free; they differ in the population they cover. ``merge`` keeps every
#: unit and joins two units whenever they share any family, so a hub domain can
#: chain a cohort into one group -- which the split then refuses rather than
#: hides. ``drop`` keeps only single-family units, which cannot chain but skews
#: the cohort towards small, single-domain, well-annotated proteins.
MULTI_LABEL_POLICIES = ("merge", "drop")

#: What to do with a unit carrying no curated family. ``refuse`` is the default
#: because the alternative reading -- "unannotated, therefore unrelated to
#: everything" -- is an assertion of exactly the kind this module exists to stop.
#: ``drop`` is available and records the count, because dropping selects for
#: annotated and therefore well-studied proteins and that bias has to be visible.
UNLABELLED_POLICIES = ("refuse", "drop")

#: Default k for the boundary-similarity audit, matching
#: ``relational.homology_clusters``' own default so the two report on one scale.
LEAKAGE_KMER = 3

#: How many cross-boundary pairs the audit samples before it stops being
#: exhaustive. 200,000 pairs of k-mer sets is seconds; the full product of a
#: thousand-unit split is half a million and of a ten-thousand-unit split is
#: twenty-five million.
LEAKAGE_PAIR_SAMPLE = 200_000


# ------------------------------------------------------------ label sources


def swissprot_accessions(path: Path = SWISSPROT_FASTA) -> set[str]:
    """Every Swiss-Prot accession, as the universe a label table is built over."""

    return {swissprot_accession(header) for header, _ in iter_fasta(path)}


def load_pfam_families(
    accessions: set[str] | None = None, path: Path = PFAM_RESIDUE_TSV
) -> dict[str, frozenset[str]]:
    """Pfam family sets per accession, from the residue-span table."""

    spans = load_pfam_spans(path, accessions=accessions)
    return {
        accession: frozenset(family for _, _, family in entries)
        for accession, entries in spans.items()
    }


def load_cath_superfamilies(
    accessions: set[str] | None = None, path: Path = CATH_SUPERFAMILY_TSV
) -> dict[str, frozenset[str]]:
    """CATH superfamily sets per accession, from the extracted table.

    Raises rather than falling back to Pfam when the table has not been
    extracted: the two sources group different things, and a stage that silently
    received the weaker one would report a superfamily-disjoint result it never
    measured.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; it is derived from {PROTEIN2IPR_DAT} by "
            "src.capability.core.families.extract_cath_superfamilies, which has to be "
            "run once, or relocated with TRANSFER_CATH_SUPERFAMILY_TSV"
        )
    families: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as handle:
        header = tuple(next(handle).rstrip("\n").split("\t"))
        if header != CATH_TSV_HEADER:
            raise ValueError(f"{path}: expected columns {CATH_TSV_HEADER}, found {header}")
        for number, line in enumerate(handle, 2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(CATH_TSV_HEADER):
                raise ValueError(
                    f"{path}:{number}: expected {len(CATH_TSV_HEADER)} columns, "
                    f"found {len(fields)}"
                )
            accession, superfamily = fields
            if accessions is not None and accession not in accessions:
                continue
            families.setdefault(accession, set()).add(superfamily)
    if not families:
        raise RuntimeError(f"{path}: no CATH superfamilies matched the requested accessions")
    return {accession: frozenset(entries) for accession, entries in families.items()}


def extract_cath_superfamilies(
    *,
    accessions: set[str],
    source: Path = PROTEIN2IPR_DAT,
    destination: Path = CATH_SUPERFAMILY_TSV,
) -> dict[str, Any]:
    """Derive the accession-to-CATH-superfamily table from InterPro's dump.

    One pass over a 17 GB gzip stream, restricted to ``accessions`` so the result
    is a file a stage can load rather than a second copy of the dump. A
    malformed row raises with its line number: this is a curated release, so a
    row that does not parse means the format moved, and a scan that skipped it
    would write a table missing an unknown number of assignments.
    """

    require_input_path(Path(source), "TRANSFER_PROTEIN2IPR")
    if not accessions:
        raise ValueError("an accession universe is required; scanning to an unfiltered table produces hundreds of millions of rows")
    pairs: set[tuple[str, str]] = set()
    rows = 0
    g3dsa_rows = 0
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != PROTEIN2IPR_COLUMNS:
                raise ValueError(
                    f"{source}:{number}: expected {PROTEIN2IPR_COLUMNS} columns, "
                    f"found {len(fields)}"
                )
            rows += 1
            signature = fields[3]
            if not signature.startswith(G3DSA_PREFIX):
                continue
            g3dsa_rows += 1
            accession = fields[0]
            if accession in accessions:
                pairs.add((accession, signature[len(G3DSA_PREFIX) :]))
    if not pairs:
        raise RuntimeError(
            f"{source}: no {G3DSA_PREFIX} signature matched any of the "
            f"{len(accessions)} requested accessions"
        )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(CATH_TSV_HEADER) + "\n")
        for accession, superfamily in sorted(pairs):
            handle.write(f"{accession}\t{superfamily}\n")
    return {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "destination": str(destination),
        "rows_read": rows,
        "g3dsa_rows": g3dsa_rows,
        "accessions_requested": len(accessions),
        "accessions_with_superfamily": len({accession for accession, _ in pairs}),
        "distinct_superfamilies": len({superfamily for _, superfamily in pairs}),
        "pairs_written": len(pairs),
    }


# ------------------------------------------------------------- assignment


def _connected_groups(labels: Sequence[frozenset[str]]) -> np.ndarray:
    """Contiguous component ids under "shares at least one label".

    Union-find over the label incidence rather than over the pairs, so the cost
    is linear in the number of (unit, label) incidences and a family with ten
    thousand members costs ten thousand unions rather than fifty million
    comparisons. Component ids are allocated in unit order, so the result is a
    function of the input alone.
    """

    parent = list(range(len(labels)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    representative: dict[str, int] = {}
    for index, label_set in enumerate(labels):
        for label in label_set:
            if label in representative:
                union(representative[label], index)
            else:
                representative[label] = index
    roots: dict[int, int] = {}
    groups = np.empty(len(labels), dtype=np.int64)
    for index in range(len(labels)):
        root = find(index)
        if root not in roots:
            roots[root] = len(roots)
        groups[index] = roots[root]
    return groups


@dataclass(frozen=True)
class FamilyAssignment:
    """Units carrying curated family labels, grouped so no group can be split."""

    source: str
    multi_label: str
    unit_ids: tuple[str, ...]
    labels: tuple[frozenset[str], ...]
    group_ids: np.ndarray
    summary: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.unit_ids:
            raise ValueError("a family assignment needs at least one unit")
        if len(self.unit_ids) != len(set(self.unit_ids)):
            raise ValueError("unit identifiers must be unique")
        if not len(self.unit_ids) == len(self.labels) == self.group_ids.size:
            raise ValueError("unit identifiers, labels and group ids must be aligned")
        if any(not entry for entry in self.labels):
            raise ValueError("every retained unit must carry at least one family label")

    @property
    def n_groups(self) -> int:
        return int(np.unique(self.group_ids).size)

    def select(self, mask: np.ndarray) -> FamilyAssignment:
        """The sub-assignment on ``mask``, for splitting one side again.

        Groups are recomputed from the retained units' own labels rather than
        renumbered, which is the same thing here -- a mask produced by
        :func:`family_disjoint_split` never divides a group -- and stays correct
        if a caller ever masks on something else.
        """

        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(self.unit_ids),):
            raise ValueError("mask must be one boolean per unit")
        if not mask.any():
            raise ValueError("mask selects no units")
        unit_ids = tuple(
            unit for unit, keep in zip(self.unit_ids, mask, strict=True) if keep
        )
        labels = tuple(
            label for label, keep in zip(self.labels, mask, strict=True) if keep
        )
        return _assemble(
            source=self.source,
            multi_label=self.multi_label,
            unit_ids=unit_ids,
            labels=labels,
            extra={"selected_from": self.summary},
        )


def _assemble(
    *,
    source: str,
    multi_label: str,
    unit_ids: tuple[str, ...],
    labels: tuple[frozenset[str], ...],
    extra: Mapping[str, Any],
) -> FamilyAssignment:
    group_ids = _connected_groups(labels)
    _, sizes = np.unique(group_ids, return_counts=True)
    distinct = {label for entry in labels for label in entry}
    summary: dict[str, Any] = {
        "source": source,
        "multi_label": multi_label,
        "n_units": len(unit_ids),
        "n_groups": int(sizes.size),
        "n_distinct_labels": len(distinct),
        "largest_group_size": int(sizes.max()),
        "largest_group_share": float(sizes.max() / len(unit_ids)),
        "n_singleton_groups": int((sizes == 1).sum()),
        "n_multi_label_units": int(sum(1 for entry in labels if len(entry) > 1)),
    }
    summary.update(extra)
    return FamilyAssignment(
        source=source,
        multi_label=multi_label,
        unit_ids=unit_ids,
        labels=labels,
        group_ids=group_ids,
        summary=summary,
    )


def family_assignment(
    unit_ids: Sequence[str],
    families: Mapping[str, frozenset[str]],
    *,
    source: str,
    multi_label: str = "merge",
    unlabelled: str = "refuse",
) -> FamilyAssignment:
    """Group units so that no curated family straddles a later split.

    ``families`` is the label table, keyed on the same identifier as
    ``unit_ids``. A unit absent from it, or present with an empty label set, is
    unlabelled and is handled by ``unlabelled``.
    """

    if source not in FAMILY_SOURCES:
        raise ValueError(f"unknown family source {source!r}; sources are {list(FAMILY_SOURCES)}")
    if multi_label not in MULTI_LABEL_POLICIES:
        raise ValueError(
            f"unknown multi-label policy {multi_label!r}; policies are {list(MULTI_LABEL_POLICIES)}"
        )
    if unlabelled not in UNLABELLED_POLICIES:
        raise ValueError(
            f"unknown unlabelled policy {unlabelled!r}; policies are {list(UNLABELLED_POLICIES)}"
        )
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("unit identifiers must be unique")

    missing = [unit for unit in unit_ids if not families.get(unit)]
    if missing and unlabelled == "refuse":
        raise RuntimeError(
            f"{len(missing)} of {len(unit_ids)} units carry no {source} family "
            f"(first: {missing[:5]}); a family-disjoint split over them would assert "
            "that unannotated proteins are unrelated. Pass unlabelled='drop' to "
            "exclude them and record the count, or build the cohort from annotated "
            "units in the first place"
        )
    kept = [unit for unit in unit_ids if families.get(unit)]
    labels = [frozenset(families[unit]) for unit in kept]

    dropped_multi = 0
    if multi_label == "drop":
        single = [
            (unit, label) for unit, label in zip(kept, labels, strict=True) if len(label) == 1
        ]
        dropped_multi = len(kept) - len(single)
        kept = [unit for unit, _ in single]
        labels = [label for _, label in single]
    if not kept:
        raise RuntimeError(
            f"no unit survived the {source} label policies "
            f"(multi_label={multi_label!r}, unlabelled={unlabelled!r})"
        )
    return _assemble(
        source=source,
        multi_label=multi_label,
        unit_ids=tuple(kept),
        labels=tuple(labels),
        extra={
            "unlabelled_policy": unlabelled,
            "n_units_requested": len(unit_ids),
            "n_units_unlabelled": len(missing),
            "n_units_dropped_multi_label": dropped_multi,
        },
    )


# ------------------------------------------------------------------ split


@dataclass(frozen=True)
class FamilySplit:
    """A fit/held-out partition that no curated family crosses."""

    assignment: FamilyAssignment
    train: np.ndarray
    seed: int
    summary: dict[str, Any]

    @property
    def test(self) -> np.ndarray:
        return ~self.train

    def unit_ids(self, side: str) -> tuple[str, ...]:
        if side not in ("train", "test"):
            raise ValueError("side must be 'train' or 'test'")
        mask = self.train if side == "train" else self.test
        return tuple(
            unit for unit, keep in zip(self.assignment.unit_ids, mask, strict=True) if keep
        )


def family_disjoint_split(
    assignment: FamilyAssignment,
    *,
    seed: int,
    train_fraction: float = 0.5,
    min_units_per_side: int = MINIMUM_BOOTSTRAP_UNITS,
    min_groups_per_side: int = 2,
    fraction_tolerance: float = 0.15,
) -> FamilySplit:
    """Partition an assignment by whole family groups, or refuse.

    The mask itself comes from :func:`relational.homology_disjoint_split`, which
    is this package's one implementation of "a mask that never divides a group".
    What is added here is what makes the result publishable: the achieved
    fraction has to be near the requested one, both sides have to carry enough
    groups to be resampled over, and the disjointness is re-checked on the
    returned mask rather than trusted from the construction. That last check is
    Appendix B rule 24 -- a precondition a comment states is a precondition the
    next run breaks.

    A refusal is the expected outcome on a cohort dominated by one family, and it
    is a finding about the cohort. Widening ``fraction_tolerance`` to make it pass
    reports a 50/50 split that is not one.
    """

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")
    if fraction_tolerance <= 0.0:
        raise ValueError("fraction_tolerance must be positive")
    if min_groups_per_side < 1:
        raise ValueError("min_groups_per_side must be positive")

    train = homology_disjoint_split(
        assignment.group_ids,
        train_fraction=train_fraction,
        seed=seed,
        min_side=min_units_per_side,
    )
    test = ~train

    train_groups = np.unique(assignment.group_ids[train])
    test_groups = np.unique(assignment.group_ids[test])
    shared = np.intersect1d(train_groups, test_groups)
    if shared.size:
        raise RuntimeError(
            f"{shared.size} family groups appear on both sides of a split that "
            "must not divide one; the mask is not family-disjoint"
        )

    # Group count before achieved fraction, and the order is deliberate. A cohort
    # with three families fails both, and told about the fraction an operator
    # widens the tolerance and gets a two-against-one split; told about the group
    # count they go and find more families, which is the actual repair.
    if train_groups.size < min_groups_per_side or test_groups.size < min_groups_per_side:
        raise RuntimeError(
            f"family-disjoint split gave {train_groups.size} training and "
            f"{test_groups.size} held-out family groups against a minimum of "
            f"{min_groups_per_side}; a side carrying one family cannot support a "
            "claim about unseen families"
        )
    achieved = float(train.sum() / train.size)
    if abs(achieved - train_fraction) > fraction_tolerance:
        largest = int(np.bincount(assignment.group_ids).max())
        raise RuntimeError(
            f"family-disjoint split put {achieved:.3f} of units on the training "
            f"side against a requested {train_fraction:.3f}, outside the "
            f"{fraction_tolerance:.3f} tolerance; the largest family group holds "
            f"{largest} of {train.size} units, so this cohort cannot be partitioned "
            "at the requested fraction by whole families"
        )

    def side_summary(mask: np.ndarray) -> dict[str, Any]:
        groups = assignment.group_ids[mask]
        _, sizes = np.unique(groups, return_counts=True)
        labels = {
            label
            for label, keep in zip(assignment.labels, mask, strict=True)
            if keep
            for label in label
        }
        return {
            "n_units": int(mask.sum()),
            "n_groups": int(sizes.size),
            "n_distinct_labels": len(labels),
            "largest_group_size": int(sizes.max()),
            "largest_group_share": float(sizes.max() / mask.sum()),
        }

    summary = {
        "source": assignment.source,
        "seed": int(seed),
        "requested_train_fraction": float(train_fraction),
        "achieved_train_fraction": achieved,
        "fraction_tolerance": float(fraction_tolerance),
        "min_units_per_side": int(min_units_per_side),
        "min_groups_per_side": int(min_groups_per_side),
        "train": side_summary(train),
        "test": side_summary(test),
        "assignment": assignment.summary,
    }
    return FamilySplit(assignment=assignment, train=train, seed=int(seed), summary=summary)


# --------------------------------------------------------- leakage audit


def boundary_leakage(
    split: FamilySplit,
    sequences: Mapping[str, str],
    *,
    kmer: int = LEAKAGE_KMER,
    sample_pairs: int = LEAKAGE_PAIR_SAMPLE,
    seed: int,
) -> dict[str, Any]:
    """Measure, rather than assert, that the two sides are unrelated.

    A curated family label is incomplete. Two proteins can be close homologues
    and carry no family in common -- an unannotated region, a family split
    between releases, a domain the signature missed -- and then the split is
    family-disjoint and not homology-disjoint, which is the failure the audit
    already records once.

    The measurement is k-mer Jaccard, the same similarity
    ``relational.homology_clusters`` groups on. Cross-boundary pairs are compared
    against **same-group** pairs, because the scale that matters is "as similar
    as two proteins we have already agreed are relatives".

    **``cross_max`` is the alarm; ``n_cross_above_same_group_median`` is only
    interpretable when the grouping is a sequence family, and this was measured
    rather than assumed.** On 400 Swiss-Prot proteins in the 64-246 band, the
    same-group median is 0.061-0.073 under Pfam grouping and 0.026-0.034 under
    CATH superfamily grouping, because superfamily members are remote homologues
    with little sequence similarity left -- which is the property that makes CATH
    the stricter split in the first place. The count above that median therefore
    reads 0-31 of ~35,000 pairs under Pfam and 917-3,335 of ~15,000 under CATH on
    the *same* cohort, and the second number is measuring how low the reference
    is, not how leaky the split is. ``cross_max`` moved over the identical range
    (0.072-0.218) under both. Read the maximum; read the count only against a
    sequence-family grouping.

    On a cohort whose every group is a singleton the reference does not exist and
    is reported as ``None``, not as a number. Above ``sample_pairs`` both
    populations are subsampled -- cross-boundary pairs with replacement, same-group
    pairs by walking a seeded permutation of the groups rather than the first few,
    which is Appendix B rule 1 applied to a control instead of to a cohort.
    """

    if kmer < 1:
        raise ValueError("kmer must be positive")
    if sample_pairs < 1:
        raise ValueError("sample_pairs must be positive")
    missing = [unit for unit in split.assignment.unit_ids if unit not in sequences]
    if missing:
        raise KeyError(
            f"{len(missing)} split units have no sequence (first: {missing[:5]}); "
            "the audit cannot report on a subset without saying so"
        )

    units = split.assignment.unit_ids
    kmers = [kmer_set(sequences[unit], kmer) for unit in units]
    generator = np.random.default_rng(seed)

    def jaccard(left: int, right: int) -> float:
        union = len(kmers[left] | kmers[right])
        return len(kmers[left] & kmers[right]) / union if union else 0.0

    train_index = np.flatnonzero(split.train)
    test_index = np.flatnonzero(split.test)
    total_cross = int(train_index.size) * int(test_index.size)
    exhaustive = total_cross <= sample_pairs
    if exhaustive:
        cross = np.asarray(
            [jaccard(int(a), int(b)) for a in train_index for b in test_index],
            dtype=np.float64,
        )
    else:
        left = generator.choice(train_index, size=sample_pairs)
        right = generator.choice(test_index, size=sample_pairs)
        cross = np.asarray(
            [jaccard(int(a), int(b)) for a, b in zip(left, right, strict=True)],
            dtype=np.float64,
        )

    groups = split.assignment.group_ids
    same: list[float] = []
    for group in generator.permutation(np.unique(groups)):
        members = np.flatnonzero(groups == group)
        if members.size < 2:
            continue
        for position, left_index in enumerate(members):
            for right_index in members[position + 1 :]:
                same.append(jaccard(int(left_index), int(right_index)))
        if len(same) >= sample_pairs:
            break
    same_median = float(np.median(same)) if same else None

    return {
        "kmer": int(kmer),
        "seed": int(seed),
        "cross_pairs_total": total_cross,
        "cross_pairs_scored": int(cross.size),
        "cross_exhaustive": bool(exhaustive),
        "cross_max": float(cross.max()),
        "cross_q99": float(np.quantile(cross, 0.99)),
        "cross_median": float(np.median(cross)),
        "same_group_pairs_scored": len(same),
        "same_group_median": same_median,
        "n_cross_above_same_group_median": (
            None if same_median is None else int((cross > same_median).sum())
        ),
        "fraction_cross_above_same_group_median": (
            None if same_median is None else float((cross > same_median).mean())
        ),
    }
