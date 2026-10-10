"""The anticipation stage's input contract, which is not the estimator's.

E02's statistic is formed over the residue alphabet an arm resolves
unambiguously, intersected across that arm's own assays so one conditional
column means one residue everywhere. Two things follow for a byte-pair interface
and both have taken a whole campaign cell down: the intersection can be empty,
and a non-empty intersection can miss a particular wild type entirely. Neither
is a defect in the measurement -- they are facts about that arm's rendering --
so the stage records them per assay and keeps measuring the rest of the panel.
These tests hold the stage to that, and leave the estimator alone.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

from src.capability.position.anticipation import composition, resolved_alphabet
from src.capability.position.contact_response import (
    CONTACT_ANGSTROM,
    MIN_SEQUENCE_SEPARATION,
    SiteGeometry,
    contact_pairs,
)
from src.capability.position.position_likelihood import CAUSAL

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "scripts/capability/position/measure_contact_anticipation.py"


def load_stage():
    spec = importlib.util.spec_from_file_location("_anticipation_stage", STAGE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def geometry_on_a_line(wildtype, *, step=2, spacing=0.6):
    """Alternate residues on a straight line, so every distance is exact."""

    return {
        position: SiteGeometry(
            j=position, residue=wildtype[position], atom="CB",
            xyz=(position * spacing, 0.0, 0.0), rsa=0.3,
        )
        for position in range(0, len(wildtype), step)
    }


def write_archive(path, *, residues, positions):
    """Only the fields the stage reads, plus the metadata the reader requires."""

    rng = np.random.default_rng(len(residues) + len(positions))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        wt_conditional_residues=np.asarray(list(residues), dtype=object).astype("U"),
        wt_conditional_positions=np.asarray(positions, dtype=np.int64),
        wt_conditional_logprobs=rng.normal(
            -3.0, 0.5, (len(positions), len(residues))
        ).astype(np.float32),
        metadata=json.dumps({"assay": path.stem}),
    )


def build(tmp_path, specification):
    """A completion record, its archives and a cohort for the named assays."""

    directory = tmp_path / "extraction"
    completion = {"identity": {"arm": "probe", "paradigm": CAUSAL}, "assays": []}
    cohort_rows = {}
    for assay, (wildtype, residues) in specification.items():
        write_archive(directory / "archives" / f"{assay}.npz",
                      residues=residues,
                      positions=sorted(geometry_on_a_line(wildtype)))
        completion["assays"].append({"assay": assay, "file": f"{assay}.npz"})
        cohort_rows[assay] = {"wildtype": wildtype, "cluster": assay, "mutants": [],
                              "sequences": []}

    def geometry_source(assay, row):
        geometry = geometry_on_a_line(row["wildtype"])
        return geometry, contact_pairs(
            geometry, cutoff=CONTACT_ANGSTROM, min_separation=MIN_SEQUENCE_SEPARATION
        )

    return completion, directory, cohort_rows, geometry_source


def test_an_unused_shared_alphabet_drops_one_assay_and_keeps_the_arm():
    # The library-level precondition the stage now handles rather than hits.
    assert composition("WYWY", ("W", "Y")).tolist() == [0.5, 0.5]
    assert resolved_alphabet([]) == ()


def test_the_stage_records_an_assay_whose_wild_type_misses_the_alphabet(tmp_path):
    stage = load_stage()
    completion, directory, cohort_rows, geometry_source = build(tmp_path, {
        # Resolved alphabets intersect in W alone; the second wild type has no W.
        "used": ("WYWYWYWYWYWYWYWYWYWYWYWYWYWYWYWY", ("W", "Y")),
        "unused": ("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", ("W", "F")),
    })
    identity, rows, shared, conditionals, compositions, per_assay = stage.arm_design(
        completion, directory, cohort_rows, geometry_source, forward_only=True,
    )
    assert identity["arm"] == "probe"
    assert shared == ("W",)
    assert set(compositions) == {"used"} and set(conditionals)
    assert rows and {row["assay"] for row in rows} == {"used"}
    status = {item["assay"]: item["status"] for item in per_assay}
    assert status["used"] == "designed"
    assert "composition control is undefined" in status["unused"]


def test_the_stage_records_an_arm_that_resolves_no_shared_alphabet(tmp_path):
    stage = load_stage()
    completion, directory, cohort_rows, geometry_source = build(tmp_path, {
        "first": ("WYWYWYWYWYWYWYWYWYWYWYWYWYWYWYWY", ("W", "Y")),
        "second": ("FGFGFGFGFGFGFGFGFGFGFGFGFGFGFGFG", ("F", "G")),
    })
    identity, rows, shared, conditionals, compositions, per_assay = stage.arm_design(
        completion, directory, cohort_rows, geometry_source, forward_only=True,
    )
    assert shared == () and rows == [] and not conditionals and not compositions
    reasons = {item["status"] for item in per_assay}
    assert len(reasons) == 1
    assert "no single-residue token alphabet shared" in reasons.pop()
