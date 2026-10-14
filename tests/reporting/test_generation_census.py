"""The attempt census counts each contributing checkpoint once.

The census is reported in two files whose checkpoint sets overlap: the 3B
campaign has its own file and is also carried in the main one. A count taken by
adding the two files' lengths therefore stated one procedure more than the
census contains. The condition that must hold is an identity, not a number: the
count of contributing procedures is the distinct population, and that is the
same population the per-arm rows are emitted for.
"""
import json

import pytest

from scripts.capability.reporting import build_derived_numbers as builder
from src.capability.core.evidence_ledger import Artifacts, Ledger

#: One census block per arm, carrying every field the census family reads.
BLOCK = {field: 800 for field in builder.CENSUS_FIELDS}


@pytest.fixture
def overlapping_sources(tmp_path):
    """Two census files sharing one arm, as the retained pair does."""
    arms = (("alpha", "beta", "shared"), ("shared",))
    for path, names in zip(builder.CENSUS_SOURCES, arms):
        source = tmp_path / path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps({
            "shingle_length": 5,
            "checkpoints": {name: dict(BLOCK, arm=name) for name in names},
        }))
    return tmp_path


def test_the_census_population_is_the_union_of_its_sources(overlapping_sources):
    ledger = Ledger(Artifacts(overlapping_sources))
    census = builder.census_checkpoints(ledger)
    sources = [ledger.artifacts.json(path)["checkpoints"] for path in builder.CENSUS_SOURCES]

    assert set(census) == set().union(*(set(block) for block in sources))
    # The defect this pins: the sources overlap, so their lengths do not add.
    assert len(census) < sum(len(block) for block in sources)
    # The later source is authoritative for the arm both carry.
    assert census["shared"][1] == builder.CENSUS_SOURCES[-1]


def test_every_distinct_arm_is_emitted_once(overlapping_sources):
    ledger = Ledger(Artifacts(overlapping_sources))
    builder.family_generation_census(ledger)
    rows = [quantity for quantity in ledger.quantities if quantity.family == "generation_census"]

    assert len(rows) == len({row.id for row in rows})
    assert {row.id.split("/")[1] for row in rows} == set(builder.census_checkpoints(ledger))


def test_the_emitted_procedure_count_is_that_population():
    """The conventions row and the per-arm rows read one population.

    This runs the two emitters the way ``build`` does, on the retained
    artifacts, because the defect was a second count computed beside the
    population rather than from it.
    """
    admission = f"{builder.READOUT_ROOT}/final_admission.json"
    artifacts = Artifacts()
    if not artifacts.exists(admission) or not builder.readout_reports():
        pytest.skip("the readout admission and reports are not staged in this checkout")
    ledger = Ledger(artifacts)
    ledger.declare_support("anchor_128", builder.ANCHOR_SUPPORT)
    ledger.declare_support("uncond_census_800", builder.CENSUS_SUPPORT)
    strata = {arm: block["stratum"]
              for arm, block in artifacts.json(admission)["interfaces"].items()}
    builder.family_generation_census(ledger)
    builder.family_roster_and_conventions(ledger, strata)
    rows = {quantity.id: quantity for quantity in ledger.quantities}

    population = builder.census_checkpoints(ledger)
    assert rows["conventions/census_procedures"].value == float(len(population))
    assert {row.id.split("/")[1] for row in ledger.quantities
            if row.family == "generation_census"} == set(population)
