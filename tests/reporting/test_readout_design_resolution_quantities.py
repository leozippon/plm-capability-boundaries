"""The readout-design arm and release-family counts, and the refusals that hold them.

The counts the Supplementary Information needs turn on a criterion no artifact
declares, so the first group of tests fixes the criterion itself: a design
resolves for an arm only when its increment excludes zero and is positive in
every cell the pair carries, and an arm that resolves at one seed and not at
another has not resolved.

A release-family count is a join across a gate artifact and the canonical
released-checkpoint-family mapping, and the second group fixes what that join may
and may not do. An arm that resolves a design and carries no release family is a
refusal: a count that silently dropped it would be one smaller than the evidence
and a count that invented a family for it would be larger. An arm a gate runs and
the mapping does not cover is a recorded gap, because it can enter no family
count whatever it resolves. A mapping that has diverged between the two places it
is persisted is a refusal, because the count turns on it.

The last tests run on the retained artifacts and assert the three pairs of counts
the Supplementary sentence rests on, so a change to either gate artifact or to the
mapping fails here rather than moving a number in the paper.
"""
import json

import pytest

from scripts.capability.reporting import readout_design_resolution_quantities as resolution
from src.capability.core.evidence_ledger import Artifacts, Ledger, LedgerError

SEEDS = (20260923, 20260924, 20260925)
#: A two-family mapping over three released checkpoints, enough to tell a family
#: count from an arm count.
FAMILIES = {"Alpha": ("alpha-small", "alpha-large"), "Beta": ("beta-one",)}
LIMIT = "Two release families are not two independent training trials."
BASIS = "released checkpoint families, independent of outcomes"


def summary(point: float, *, excludes_zero: bool) -> dict:
    return {"point": point, "excludes_zero": excludes_zero, "resamples": 2000,
            "unit": "wild-type family at 50% identity", "minimum_units": 8}


def class_cell(arm: str, seed: int, *, panel: str = "anchor201",
               designs: dict[str, dict] | None = None) -> dict:
    return {"path": f"{arm}/{seed}.json", "sha256": "0" * 64, "block": {
        "arm": arm, "panel": panel, "stratum": "native_sequence", "fold_seed": seed,
        "n_assays": 201, "n_families": 163, "n_variants": 25728,
        "schema_version": resolution.CELL_SCHEMA, "status": "complete",
        "primary_contrast": "per class, B_R minus B Spearman",
        "summaries": {"raw_M_spearman": {"point": 0.3}, **{
            f"{design}/{resolution.INCREMENT}": block
            for design, block in (designs or {}).items()}},
    }}


def class_payload(cells: list[dict], *, schema: str = resolution.CLASS_SCHEMA,
                  count: int | None = None) -> dict:
    return {"schema": schema, "rule": "copied verbatim",
            "counts": {"class": len(cells) if count is None else count, "depth": 0},
            "cells": {"class": cells, "depth": []}}


def every_design(arm: str, seed: int, **points: float) -> dict:
    """One cell carrying all nine declared designs, resolving only where named."""
    return class_cell(arm, seed, designs={
        design: summary(points.get(design, -0.01), excludes_zero=design in points)
        for design in resolution.DESIGNS})


def depth_payload(blocks: list[dict], *, roster: tuple[str, ...] = (),
                  schema: str = resolution.DEPTH_SCHEMA, status: str = "complete") -> dict:
    return {"schema_version": schema, "status": status, "n_cells": len(blocks),
            "arms": list(roster), "panels": ["anchor201"],
            "seed_consistent": blocks}


def depth_block(arm: str, *, minimum: float = 0.01, seeds: int = 3, axis: str = "depth") -> dict:
    return {"arm": arm, "panel": "anchor201", "stratum": "native_sequence", "axis": axis,
            "depth": 12, "seeds": seeds, "minimum": minimum, "maximum": minimum + 0.01}


def mapping_payloads(families: dict | None = None, *,
                     record_families: dict | None = None) -> tuple[dict, dict]:
    families = FAMILIES if families is None else families
    declaration = {
        "families": {name: ["yes", list(members)] for name, members in families.items()},
        "mapping_basis": BASIS,
    }
    records = families if record_families is None else record_families
    robustness = {
        "schema": resolution.LINEAGE_SCHEMA,
        "declaration_sha256": None,
        "family_records": {name: {"checkpoints": list(members)}
                           for name, members in records.items()},
        "training_ancestry_limit": LIMIT,
    }
    return declaration, robustness


def stage(tmp_path, *, class_cells: list[dict] | None = None,
          depth_blocks: list[dict] | None = None, depth_roster: tuple[str, ...] | None = None,
          families: dict | None = None, record_families: dict | None = None,
          declaration_digest: str | None = None, class_overrides: dict | None = None,
          depth_overrides: dict | None = None) -> Ledger:
    cells = class_cells if class_cells is not None else [
        every_design("alpha-small", seed, C1_compressed_linear=0.01) for seed in SEEDS]
    blocks = depth_blocks if depth_blocks is not None else [depth_block("beta-one")]
    roster = depth_roster if depth_roster is not None else tuple(
        arm for members in (families or FAMILIES).values() for arm in members)
    declaration, robustness = mapping_payloads(families, record_families=record_families)
    write(tmp_path, resolution.LINEAGE_DECLARATION, declaration)
    robustness["declaration_sha256"] = (
        declaration_digest if declaration_digest is not None
        else Artifacts(tmp_path).sha256(resolution.LINEAGE_DECLARATION))
    write(tmp_path, resolution.LINEAGE_ROBUSTNESS, robustness)
    write(tmp_path, resolution.CLASS_SWEEP,
          {**class_payload(cells), **(class_overrides or {})})
    write(tmp_path, resolution.DEPTH_PANEL,
          {**depth_payload(blocks, roster=roster), **(depth_overrides or {})})
    ledger = Ledger(Artifacts(tmp_path))
    resolution.add_readout_design_resolution_quantities(ledger)
    return ledger


def write(root, relative: str, payload: dict) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def rows(ledger: Ledger) -> dict:
    return {quantity.id: quantity for quantity in ledger.quantities}


def value(ledger: Ledger, identifier: str) -> float:
    return rows(ledger)[f"readout_design_resolution/{identifier}"].value


# --------------------------------------------------------------------------- #
# The criterion.
# --------------------------------------------------------------------------- #

def test_a_design_resolves_only_when_every_cell_it_carries_is_positive(tmp_path):
    # Three seeds, all resolving: the arm resolves.
    ledger = stage(tmp_path, class_cells=[
        every_design("alpha-small", seed, C1_compressed_linear=0.01) for seed in SEEDS])
    assert value(ledger, "class_axis/arms_resolving") == 1.0
    # One seed where the increment does not exclude zero: it has not resolved.
    ledger = stage(tmp_path, class_cells=[
        every_design("alpha-small", SEEDS[0], C1_compressed_linear=0.01),
        every_design("alpha-small", SEEDS[1], C1_compressed_linear=0.01),
        every_design("alpha-small", SEEDS[2])])
    assert value(ledger, "class_axis/arms_resolving") == 0.0
    # One seed where it excludes zero on the wrong side: also not resolved.
    ledger = stage(tmp_path, class_cells=[
        every_design("alpha-small", SEEDS[0], C1_compressed_linear=0.01),
        class_cell("alpha-small", SEEDS[1], designs={
            design: summary(-0.02 if design == "C1_compressed_linear" else -0.01,
                            excludes_zero=design == "C1_compressed_linear")
            for design in resolution.DESIGNS})])
    assert value(ledger, "class_axis/arms_resolving") == 0.0


def test_an_arm_resolving_any_one_design_resolves(tmp_path):
    ledger = stage(tmp_path, class_cells=[
        every_design("alpha-small", SEEDS[0], D_final_mean=0.01)])
    assert value(ledger, "class_axis/arms_resolving") == 1.0
    assert value(ledger, "class_axis/arms_resolving_only_at_one_split_seed") == 1.0


def test_an_arm_resolving_a_design_the_gate_ran_at_every_seed_is_not_counted_as_single_seed(
        tmp_path):
    ledger = stage(tmp_path, class_cells=[
        every_design("alpha-small", seed, C1_compressed_linear=0.01) for seed in SEEDS])
    assert value(ledger, "class_axis/arms_resolving") == 1.0
    assert value(ledger, "class_axis/arms_resolving_only_at_one_split_seed") == 0.0


def test_a_depth_block_that_is_not_positive_at_its_worst_seed_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, depth_blocks=[depth_block("beta-one", minimum=-0.001)])
    assert "not positive at every split seed" in str(refusal.value)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, depth_blocks=[depth_block("beta-one", seeds=0)])
    assert "rests on no split seed at all" in str(refusal.value)


# --------------------------------------------------------------------------- #
# The join, and what it may not do.
# --------------------------------------------------------------------------- #

def test_counts_separate_arms_from_release_families(tmp_path):
    ledger = stage(tmp_path, class_cells=[
        every_design(arm, seed, C1_compressed_linear=0.01)
        for arm in ("alpha-small", "alpha-large") for seed in SEEDS])
    # Two arms of one family in the class gate, one arm of another in the depth
    # panel: the union is three arms over two families.
    assert value(ledger, "class_axis/arms_resolving") == 2.0
    assert value(ledger, "class_axis/release_families_resolving") == 1.0
    assert value(ledger, "depth_panel/arms_resolving") == 1.0
    assert value(ledger, "depth_panel/release_families_resolving") == 1.0
    assert value(ledger, "union/arms_resolving") == 3.0
    assert value(ledger, "union/release_families_resolving") == 2.0


def test_an_unmapped_arm_that_resolves_a_design_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, class_cells=[
            every_design("gamma-unmapped", seed, C1_compressed_linear=0.01) for seed in SEEDS],
            depth_roster=("alpha-small", "alpha-large", "beta-one", "gamma-unmapped"))
    message = str(refusal.value)
    assert "gamma-unmapped" in message
    assert "drop them silently or invent a family for them" in message
    # The same arm resolving in the depth panel is refused too, so neither gate
    # can slip an unmapped arm into a family count.
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, depth_blocks=[depth_block("gamma-unmapped")],
              depth_roster=("alpha-small", "alpha-large", "beta-one", "gamma-unmapped"))
    assert "gamma-unmapped" in str(refusal.value)


def test_an_unmapped_arm_a_gate_merely_runs_is_recorded_as_a_gap(tmp_path):
    ledger = stage(tmp_path, class_cells=[
        every_design("alpha-small", seed, C1_compressed_linear=0.01) for seed in SEEDS]
        + [every_design("gamma-unmapped", seed) for seed in SEEDS],
        depth_roster=("alpha-small", "alpha-large", "beta-one", "gamma-unmapped"))
    gaps = {gap.id: gap for gap in ledger.gaps}
    for segment in ("class_axis", "depth_panel"):
        gap = gaps[f"readout_design_resolution/{segment}/unmapped_arms"]
        assert "gamma-unmapped" in gap.claim and "gamma-unmapped" in gap.reason
        assert gap.reachability == "not_recorded" and gap.searched
    # It is a gap and not a silent drop: the counts stand on the mapped arms.
    assert value(ledger, "class_axis/arms_resolving") == 1.0
    assert value(ledger, "class_axis/release_families_resolving") == 1.0


def test_the_join_itself_is_recorded_as_a_gap(tmp_path):
    ledger = stage(tmp_path)
    gap = {entry.id: entry for entry in ledger.gaps}[
        "readout_design_resolution/joined_counts"]
    assert "no retained artifact applies the all-seeds criterion" in gap.reason
    assert "one digest cannot fix a number two or three artifacts fix" in gap.reason
    assert gap.reachability == "not_recorded"


def test_every_artifact_of_a_join_is_named_with_its_digest_on_the_support(tmp_path):
    ledger = stage(tmp_path)
    artifacts = Artifacts(tmp_path)
    expected = {
        "readout_design_resolution/class_axis": (
            resolution.CLASS_SWEEP, resolution.LINEAGE_ROBUSTNESS, resolution.LINEAGE_DECLARATION),
        "readout_design_resolution/depth_panel": (
            resolution.DEPTH_PANEL, resolution.LINEAGE_ROBUSTNESS, resolution.LINEAGE_DECLARATION),
        "readout_design_resolution/union": (
            resolution.CLASS_SWEEP, resolution.DEPTH_PANEL, resolution.LINEAGE_ROBUSTNESS,
            resolution.LINEAGE_DECLARATION),
    }
    for support_id, paths in expected.items():
        description = ledger.supports[support_id].description
        for path in paths:
            assert f"{path} (sha256 {artifacts.sha256(path)})" in description
        assert "fixed jointly by" in description
    # Each row's own digest is the artifact its pointer reads, and a family count
    # points at the mapping.
    emitted = rows(ledger)
    assert emitted["readout_design_resolution/class_axis/arms_resolving"].source_path == \
        resolution.CLASS_SWEEP
    assert emitted["readout_design_resolution/depth_panel/arms_resolving"].source_path == \
        resolution.DEPTH_PANEL
    for segment in ("class_axis", "depth_panel", "union"):
        row = emitted[f"readout_design_resolution/{segment}/release_families_resolving"]
        assert row.source_path == resolution.LINEAGE_ROBUSTNESS
        assert row.source_sha256 == artifacts.sha256(resolution.LINEAGE_ROBUSTNESS)


def test_the_criterion_and_the_ancestry_limit_travel_on_every_support(tmp_path):
    ledger = stage(tmp_path)
    for support in ledger.supports.values():
        assert "excludes zero and is positive in every admitted cell" in support.description
        assert f"'{BASIS}'" in support.description
        assert LIMIT in support.description
        assert "release families and never independent lineages" in support.description


def test_a_mapping_that_has_diverged_between_its_two_records_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, record_families={"Alpha": ("alpha-small",), "Beta": ("beta-one",)})
    assert "no one mapping a family count can be taken under" in str(refusal.value)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, declaration_digest="f" * 64)
    assert "no longer state one mapping" in str(refusal.value)


def test_a_checkpoint_in_two_release_families_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, families={"Alpha": ("alpha-small",), "Beta": ("alpha-small",)})
    assert "has no one release family" in str(refusal.value)


# --------------------------------------------------------------------------- #
# What the gate artifacts have to be.
# --------------------------------------------------------------------------- #

def test_another_design_roster_or_an_incomplete_excerpt_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, class_cells=[class_cell("alpha-small", SEEDS[0], designs={
            design: summary(0.01, excludes_zero=True) for design in resolution.DESIGNS[:5]})])
    assert "a reduction over a different roster is a different count" in str(refusal.value)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, class_cells=[every_design("alpha-small", SEEDS[0], **{
            "C1_compressed_linear": 0.01})], class_overrides={"counts": {"class": 7, "depth": 0}})
    assert "one of the two is incomplete" in str(refusal.value)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, class_overrides={"schema": "readout_sweep_cell_excerpt_v2"})
    assert "not a readout_sweep_cell_excerpt excerpt" in str(refusal.value)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, depth_overrides={"status": "partial"})
    assert "is not a complete" in str(refusal.value)


def test_two_interval_conventions_or_two_contrasts_behind_one_criterion_refuse(tmp_path):
    other = every_design("alpha-small", SEEDS[1], C1_compressed_linear=0.01)
    other["block"]["summaries"][f"C1_compressed_linear/{resolution.INCREMENT}"]["resamples"] = 4000
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, class_cells=[
            every_design("alpha-small", SEEDS[0], C1_compressed_linear=0.01), other])
    assert "no one interval convention behind the criterion" in str(refusal.value)
    moved = every_design("alpha-small", SEEDS[1], C1_compressed_linear=0.01)
    moved["block"]["primary_contrast"] = "per class, something else"
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, class_cells=[
            every_design("alpha-small", SEEDS[0], C1_compressed_linear=0.01), moved])
    assert "is not one quantity" in str(refusal.value)


def test_an_unfitted_class_cell_refuses(tmp_path):
    stale = every_design("alpha-small", SEEDS[0], C1_compressed_linear=0.01)
    stale["block"]["status"] = "running"
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, class_cells=[stale])
    assert "not an admitted fit the criterion can be applied to" in str(refusal.value)


# --------------------------------------------------------------------------- #
# The retained artifacts.
# --------------------------------------------------------------------------- #

def test_the_retained_gates_carry_the_counts_the_supplement_quotes():
    artifacts = Artifacts()
    for path in (resolution.CLASS_SWEEP, resolution.DEPTH_PANEL,
                 resolution.LINEAGE_ROBUSTNESS, resolution.LINEAGE_DECLARATION):
        if not artifacts.exists(path):
            pytest.skip(f"{path} is not staged in this checkout")
    ledger = Ledger(artifacts)
    resolution.add_readout_design_resolution_quantities(ledger)
    # Nine arms over four release families on the class axis, nine over six in
    # the depth panel, and eleven over six for the union the Supplementary
    # sentence describes, because it names both design groups.
    assert value(ledger, "class_axis/arms_resolving") == 9.0
    assert value(ledger, "class_axis/release_families_resolving") == 4.0
    assert value(ledger, "depth_panel/arms_resolving") == 9.0
    assert value(ledger, "depth_panel/release_families_resolving") == 6.0
    assert value(ledger, "union/arms_resolving") == 11.0
    assert value(ledger, "union/release_families_resolving") == 6.0
    # Six of the class gate's nine arms resolve only through a pooled-depth
    # design that gate ran at a single split seed, which the support says.
    assert value(ledger, "class_axis/arms_resolving_only_at_one_split_seed") == 6.0
    assert "is 3 seeds for a class design and one for a pooled-depth design" in \
        ledger.supports["readout_design_resolution/class_axis"].description


def test_zymctrl_is_the_arm_the_retained_mapping_does_not_carry():
    artifacts = Artifacts()
    if not artifacts.exists(resolution.LINEAGE_ROBUSTNESS):
        pytest.skip("the release-family record is not staged in this checkout")
    ledger = Ledger(artifacts)
    resolution.add_readout_design_resolution_quantities(ledger)
    gaps = {gap.id: gap for gap in ledger.gaps}
    for segment in ("class_axis", "depth_panel"):
        gap = gaps[f"readout_design_resolution/{segment}/unmapped_arms"]
        assert "zymctrl" in gap.claim
        assert "none of them resolves any design" in gap.reason
