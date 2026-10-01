"""Ways a family-disjoint split could be wrong and say it was right.

Each test corresponds to a failure this component exists to make impossible
rather than to a line of its implementation:

* a family straddling the split -- the ``family_disjoint=True`` that was a field
  rather than a property (§0.05 of the audit);
* a multi-domain protein splitting a family through its second domain, which a
  "one dominant family per protein" rule cannot see;
* an unannotated protein being treated as unrelated to everything, which turns a
  gap in the label table into a licence to split anywhere;
* a cohort dominated by one family reporting a 50/50 split that is 95/5;
* a side carrying one family supporting a claim about unseen families;
* the label table's format moving under a reader that skips what it cannot parse;
* a similarity audit whose reference population is whatever came first in the
  file, which is Appendix B rule 1 arriving inside a control.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.families import (  # noqa: E402
    CATH_TSV_HEADER,
    boundary_leakage,
    extract_cath_superfamilies,
    family_assignment,
    family_disjoint_split,
    load_cath_superfamilies,
)


def _assignment(families: dict[str, frozenset[str]], **kwargs):
    return family_assignment(
        sorted(families), families, source="pfam", **kwargs
    )


def _many_families(n_families: int, per_family: int) -> dict[str, frozenset[str]]:
    return {
        f"P{family:03d}_{member:02d}": frozenset({f"PF{family:05d}"})
        for family in range(n_families)
        for member in range(per_family)
    }


# ------------------------------------------------------- the core invariant


@pytest.mark.parametrize("seed", range(12))
def test_no_family_ever_straddles_the_split(seed: int) -> None:
    """The property that must always hold, over many draws and uneven families."""

    families = {
        f"P{family:02d}_{member:02d}": frozenset({f"PF{family:05d}"})
        for family in range(14)
        for member in range(1 + family % 5)
    }
    split = family_disjoint_split(_assignment(families), seed=seed)
    train = {label for unit in split.unit_ids("train") for label in families[unit]}
    test = {label for unit in split.unit_ids("test") for label in families[unit]}
    assert not train & test
    assert set(split.unit_ids("train")) | set(split.unit_ids("test")) == set(families)
    assert not set(split.unit_ids("train")) & set(split.unit_ids("test"))


def test_a_shared_second_domain_joins_two_proteins() -> None:
    """The leak a dominant-family rule cannot see.

    ``A`` and ``B`` share nothing obvious -- their first families differ -- but
    both carry ``PF00002``. A split that keyed on one family per protein could
    put them on opposite sides and call the result family-disjoint.
    """

    families = {
        "A": frozenset({"PF00001", "PF00002"}),
        "B": frozenset({"PF00003", "PF00002"}),
        "C": frozenset({"PF00009"}),
        "D": frozenset({"PF00010"}),
    }
    assignment = _assignment(families)
    groups = dict(zip(assignment.unit_ids, assignment.group_ids, strict=True))
    assert groups["A"] == groups["B"]
    assert groups["C"] != groups["A"] and groups["D"] != groups["A"]
    assert assignment.summary["n_multi_label_units"] == 2


def test_transitive_chaining_through_a_hub_domain_is_visible_not_hidden() -> None:
    """A hub domain collapses the cohort, and the collapse is refused, not split."""

    families = {
        f"P{index:02d}": frozenset({f"PF{index:05d}", "PF00000"}) for index in range(20)
    }
    assignment = _assignment(families)
    assert assignment.n_groups == 1
    assert assignment.summary["largest_group_share"] == pytest.approx(1.0)
    with pytest.raises(RuntimeError, match="one homology cluster|does not exist"):
        family_disjoint_split(assignment, seed=1)


# --------------------------------------------------------------- refusals


def test_unlabelled_units_are_refused_by_default() -> None:
    families = {"A": frozenset({"PF1"}), "B": frozenset({"PF2"}), "C": frozenset()}
    with pytest.raises(RuntimeError, match="carry no pfam family"):
        family_assignment(["A", "B", "C"], families, source="pfam")


def test_unlabelled_units_may_be_dropped_and_the_count_is_recorded() -> None:
    families = {"A": frozenset({"PF1"}), "B": frozenset({"PF2"})}
    assignment = family_assignment(
        ["A", "B", "C", "D"], families, source="pfam", unlabelled="drop"
    )
    assert assignment.unit_ids == ("A", "B")
    assert assignment.summary["n_units_unlabelled"] == 2
    assert assignment.summary["n_units_requested"] == 4


def test_dropping_multi_label_units_keeps_the_split_leak_free_and_records_the_bias() -> None:
    families = {
        "A": frozenset({"PF1", "PF2"}),
        "B": frozenset({"PF2"}),
        "C": frozenset({"PF3"}),
    }
    assignment = family_assignment(
        sorted(families), families, source="pfam", multi_label="drop"
    )
    assert assignment.unit_ids == ("B", "C")
    assert assignment.summary["n_units_dropped_multi_label"] == 1
    assert assignment.summary["n_multi_label_units"] == 0


def test_a_dominant_family_refuses_rather_than_reporting_the_wrong_fraction() -> None:
    """One family holding 90% of the cohort cannot be partitioned 50/50."""

    families = {f"big_{index:02d}": frozenset({"PF00001"}) for index in range(90)}
    families.update({f"small_{index:02d}": frozenset({f"PF{index:05d}"}) for index in range(1, 11)})
    assignment = _assignment(families)
    with pytest.raises(RuntimeError, match="outside the .* tolerance|too homology-collapsed"):
        family_disjoint_split(assignment, seed=0, train_fraction=0.5)


def test_a_side_with_one_family_is_refused() -> None:
    families = _many_families(n_families=3, per_family=10)
    with pytest.raises(RuntimeError, match="family groups against a minimum"):
        family_disjoint_split(
            _assignment(families), seed=0, train_fraction=0.5, min_groups_per_side=2
        )


def test_a_side_below_the_bootstrap_floor_is_refused() -> None:
    families = _many_families(n_families=6, per_family=1)
    with pytest.raises(RuntimeError, match="too homology-collapsed|train and"):
        family_disjoint_split(_assignment(families), seed=0, min_units_per_side=8)


def test_unknown_policies_and_sources_raise() -> None:
    families = {"A": frozenset({"PF1"}), "B": frozenset({"PF2"})}
    with pytest.raises(ValueError, match="unknown family source"):
        family_assignment(["A", "B"], families, source="scop")
    with pytest.raises(ValueError, match="unknown multi-label policy"):
        family_assignment(["A", "B"], families, source="pfam", multi_label="dominant")
    with pytest.raises(ValueError, match="unknown unlabelled policy"):
        family_assignment(["A", "B"], families, source="pfam", unlabelled="singleton")
    with pytest.raises(ValueError, match="unique"):
        family_assignment(["A", "A"], families, source="pfam")


# ------------------------------------------------- determinism and reuse


def test_the_split_is_a_function_of_its_seed() -> None:
    families = _many_families(n_families=24, per_family=3)
    assignment = _assignment(families)
    first = family_disjoint_split(assignment, seed=7)
    again = family_disjoint_split(assignment, seed=7)
    other = family_disjoint_split(assignment, seed=8)
    assert np.array_equal(first.train, again.train)
    assert not np.array_equal(first.train, other.train)
    assert first.summary["achieved_train_fraction"] == pytest.approx(
        float(first.train.mean())
    )


def test_a_three_way_split_by_composition_is_pairwise_family_disjoint() -> None:
    """The documented recipe for fit / select / report, held as a property."""

    families = _many_families(n_families=30, per_family=4)
    assignment = _assignment(families)
    outer = family_disjoint_split(assignment, seed=3, train_fraction=0.6)
    inner = family_disjoint_split(
        assignment.select(outer.train), seed=4, train_fraction=0.5
    )
    sides = {
        "report": set(outer.unit_ids("test")),
        "fit": set(inner.unit_ids("train")),
        "select": set(inner.unit_ids("test")),
    }
    labels = {
        name: {label for unit in units for label in families[unit]}
        for name, units in sides.items()
    }
    for left in sides:
        for right in sides:
            if left < right:
                assert not sides[left] & sides[right]
                assert not labels[left] & labels[right]
    assert set().union(*sides.values()) == set(families)


def test_select_rejects_a_mask_of_the_wrong_shape_or_an_empty_one() -> None:
    assignment = _assignment(_many_families(n_families=6, per_family=3))
    with pytest.raises(ValueError, match="one boolean per unit"):
        assignment.select(np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="selects no units"):
        assignment.select(np.zeros(len(assignment.unit_ids), dtype=bool))


# ------------------------------------------------------- the label table


def test_cath_table_reader_enforces_its_header_and_column_count(tmp_path: Path) -> None:
    good = tmp_path / "good.tsv"
    good.write_text("\t".join(CATH_TSV_HEADER) + "\nP1\t3.40.50.300\nP1\t1.10.10.10\nP2\t3.40.50.300\n")
    families = load_cath_superfamilies(path=good)
    assert families["P1"] == frozenset({"3.40.50.300", "1.10.10.10"})
    assert families["P2"] == frozenset({"3.40.50.300"})

    wrong_header = tmp_path / "wrong_header.tsv"
    wrong_header.write_text("uniprot\tcath\nP1\t3.40.50.300\n")
    with pytest.raises(ValueError, match="expected columns"):
        load_cath_superfamilies(path=wrong_header)

    ragged = tmp_path / "ragged.tsv"
    ragged.write_text("\t".join(CATH_TSV_HEADER) + "\nP1\t3.40.50.300\textra\n")
    with pytest.raises(ValueError, match="expected 2 columns"):
        load_cath_superfamilies(path=ragged)

    with pytest.raises(FileNotFoundError, match="extract_cath_superfamilies"):
        load_cath_superfamilies(path=tmp_path / "absent.tsv")

    with pytest.raises(RuntimeError, match="no CATH superfamilies matched"):
        load_cath_superfamilies({"Q9"}, path=good)


def test_extraction_keeps_only_gene3d_signatures_and_raises_on_a_moved_format(
    tmp_path: Path,
) -> None:
    import gzip

    source = tmp_path / "protein2ipr.dat.gz"
    rows = [
        "P1\tIPR004839\tAminotransferase\tPF00155\t41\t381",
        "P1\tIPR015421\tPLP transferase\tG3DSA:3.40.640.10\t48\t288",
        "P1\tIPR015421\tPLP transferase\tG3DSA:3.40.640.10\t300\t360",
        "P2\tIPR027417\tP-loop\tG3DSA:3.40.50.300\t341\t573",
        "P2\tIPR027417\tP-loop\tSSF52540\t341\t573",
        "P9\tIPR027417\tP-loop\tG3DSA:9.99.99.99\t1\t10",
    ]
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")

    destination = tmp_path / "cath.tsv"
    report = extract_cath_superfamilies(
        accessions={"P1", "P2"}, source=source, destination=destination
    )
    assert report["rows_read"] == 6
    assert report["g3dsa_rows"] == 4
    assert report["pairs_written"] == 2  # the duplicate span collapses
    assert report["accessions_with_superfamily"] == 2
    families = load_cath_superfamilies(path=destination)
    assert families == {
        "P1": frozenset({"3.40.640.10"}),
        "P2": frozenset({"3.40.50.300"}),
    }

    moved = tmp_path / "moved.dat.gz"
    with gzip.open(moved, "wt", encoding="utf-8") as handle:
        handle.write("P1\tIPR004839\tG3DSA:3.40.640.10\t41\n")
    with pytest.raises(ValueError, match="expected 6 columns"):
        extract_cath_superfamilies(
            accessions={"P1"}, source=moved, destination=tmp_path / "out.tsv"
        )

    with pytest.raises(ValueError, match="accession universe is required"):
        extract_cath_superfamilies(
            accessions=set(), source=source, destination=tmp_path / "out2.tsv"
        )

    with pytest.raises(RuntimeError, match="no G3DSA: signature matched"):
        extract_cath_superfamilies(
            accessions={"ZZZ"}, source=source, destination=tmp_path / "out3.tsv"
        )


# ------------------------------------------------------- the leakage audit


def test_the_audit_detects_homologues_the_labels_missed() -> None:
    """Two near-identical proteins carrying different families is the failure."""

    rng = np.random.default_rng(0)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"

    def random_sequence(length: int = 300) -> str:
        return "".join(rng.choice(list(alphabet), size=length))

    families = _many_families(n_families=16, per_family=3)
    sequences = {unit: random_sequence() for unit in families}
    # Each family's members are near-copies of the first, so a same-group pair is
    # the "definitely related" reference the audit compares against.
    for family in range(16):
        base = sequences[f"P{family:03d}_00"]
        for member in (1, 2):
            sequences[f"P{family:03d}_{member:02d}"] = base[:280] + random_sequence(20)

    assignment = _assignment(families)
    split = family_disjoint_split(assignment, seed=5)
    clean = boundary_leakage(split, sequences, seed=0)
    assert clean["same_group_median"] is not None
    assert clean["n_cross_above_same_group_median"] == 0
    assert clean["cross_exhaustive"] is True

    # Now plant the leak: one held-out protein is a near-copy of a training one
    # while carrying a family of its own, which is exactly what an incomplete
    # label table produces.
    donor = split.unit_ids("train")[0]
    recipient = split.unit_ids("test")[0]
    leaked = dict(sequences)
    leaked[recipient] = sequences[donor][:290] + random_sequence(10)
    dirty = boundary_leakage(split, leaked, seed=0)
    assert dirty["cross_max"] > clean["cross_max"]
    assert dirty["n_cross_above_same_group_median"] > 0


def test_the_audit_refuses_a_unit_with_no_sequence_and_reports_a_null_reference() -> None:
    families = _many_families(n_families=20, per_family=1)
    sequences = {unit: "ACDEFGHIKLMNPQRSTVWY" * 5 for unit in families}
    assignment = _assignment(families)
    split = family_disjoint_split(assignment, seed=2)

    with pytest.raises(KeyError, match="have no sequence"):
        boundary_leakage(split, {k: v for k, v in list(sequences.items())[:5]}, seed=0)

    # Every group is a singleton here, so "as similar as two known relatives" has
    # no referent and must be reported as absent rather than as a number.
    report = boundary_leakage(split, sequences, seed=0)
    assert report["same_group_pairs_scored"] == 0
    assert report["same_group_median"] is None
    assert report["n_cross_above_same_group_median"] is None
    assert report["fraction_cross_above_same_group_median"] is None


def test_the_audit_subsamples_and_says_so() -> None:
    families = _many_families(n_families=40, per_family=2)
    sequences = {
        unit: "".join(np.random.default_rng(index).choice(list("ACDEFGHIKLMNPQRSTVWY"), 120))
        for index, unit in enumerate(sorted(families))
    }
    split = family_disjoint_split(_assignment(families), seed=1)
    report = boundary_leakage(split, sequences, seed=0, sample_pairs=50)
    assert report["cross_exhaustive"] is False
    assert report["cross_pairs_scored"] == 50
    assert report["cross_pairs_total"] > 50
