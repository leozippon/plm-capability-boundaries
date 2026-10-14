"""The sequential-control progression's panel-wide counts come from its blocks.

The main text claims a count over the whole panel: adding the evolutionary
profile to model likelihood, and then the local-window features on top of both,
yields a positive increment for every checkpoint, above zero under the
predeclared simultaneous interval. A count over the whole panel is only
statable when the blocks cover the whole panel, so a contrast missing an arm and
a group that has not finished are refusals rather than smaller counts.
"""
import json

import pytest

from scripts.capability.reporting import build_derived_numbers as builder
from src.capability.core.evidence_ledger import Artifacts, Ledger, LedgerError

ARMS = 4
PREFIX = f"information_progression/{builder.PROGRESSION_GROUP}"


def payload(contrasts: list[dict], *, status: str = "complete") -> dict:
    return {
        "panel": "full", "status": "complete",
        "families": [f"cluster-{index}" for index in range(163)],
        "arms": [f"arm-{index}" for index in range(ARMS)],
        "seeds": [20260923, 20260924, 20260925], "bootstrap": 2000,
        "aggregation": "mean seeds within assay, equal assays within family, equal families",
        "simultaneous_method": "joint max absolute centered bootstrap deviation",
        "groups": {builder.PROGRESSION_GROUP: {
            "status": status, "hypothesis_count": len(contrasts),
            "joint_critical_max_absolute_deviation": 0.05, "contrasts": contrasts,
        }},
    }


def blocks(*, negative: int = 0, unresolved: int = 0) -> list[dict]:
    """One block per arm per claimed contrast, with the first arms weakened."""
    rows = []
    for contrast in builder.PROGRESSION_CONTRASTS:
        for index in range(ARMS):
            point = -0.4 if index < negative else 0.4
            low = -0.05 if index < max(negative, unresolved) else 0.35
            rows.append({"arm": f"arm-{index}", "metric": "spearman", "contrast": contrast,
                         "point": point, "pointwise": [low, 0.43], "simultaneous": [low, 0.45]})
    return rows


def build(tmp_path, contrasts: list[dict], **over) -> Ledger:
    source = tmp_path / builder.PROGRESSION
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps(payload(contrasts, **over)))
    ledger = Ledger(Artifacts(tmp_path))
    builder.family_information_progression(ledger)
    return ledger


def emit(tmp_path, contrasts: list[dict], **over) -> dict:
    return {quantity.id: quantity for quantity in build(tmp_path, contrasts, **over).quantities}


def test_the_support_counts_are_the_panel_the_artifact_declares(tmp_path):
    rows = emit(tmp_path, blocks())
    assert rows["information_progression/arms"].value == float(ARMS)
    assert rows["information_progression/families"].value == 163.0
    assert all(row.seed_set == (20260923, 20260924, 20260925) for row in rows.values())


def test_the_resampling_procedure_stays_in_the_support_rather_than_a_row(tmp_path):
    """No row restates the draw count or the simultaneous family.

    No claim quotes either, and both collide with numbers the manuscript writes
    for other things -- the interval convention's 2,000 draws and the
    remote-homology close stratum's 66 groups -- so a row for them could only
    dilute the attribution of a number the manuscript does quote.
    """
    ledger = build(tmp_path, blocks())
    rows = {quantity.id: quantity for quantity in ledger.quantities}
    assert not [identifier for identifier in rows
                if identifier.endswith(("bootstrap_draws", "split_seeds", "hypotheses",
                                        "joint_critical_deviation"))]
    description = ledger.supports["information_progression_anchor"].description
    assert "2000 draws over a" in description
    assert f"{2 * ARMS}-contrast predeclared family" in description


def test_a_whole_panel_claim_counts_every_arm(tmp_path):
    rows = emit(tmp_path, blocks())
    for contrast in builder.PROGRESSION_CONTRASTS:
        assert rows[f"{PREFIX}/{contrast}/arms_with_a_positive_increment"].value == float(ARMS)
        resolving = rows[f"{PREFIX}/{contrast}/arms_resolving_above_zero"]
        assert resolving.value == float(ARMS)
        assert resolving.verdict == "supported"


def test_a_negative_arm_and_an_unresolved_arm_are_not_counted(tmp_path):
    rows = emit(tmp_path, blocks(negative=1, unresolved=2))
    for contrast in builder.PROGRESSION_CONTRASTS:
        assert rows[f"{PREFIX}/{contrast}/arms_with_a_positive_increment"].value == float(ARMS - 1)
        assert rows[f"{PREFIX}/{contrast}/arms_resolving_above_zero"].value == float(ARMS - 2)


def test_no_arm_resolving_is_not_detected_rather_than_supported(tmp_path):
    rows = emit(tmp_path, blocks(unresolved=ARMS))
    for contrast in builder.PROGRESSION_CONTRASTS:
        resolving = rows[f"{PREFIX}/{contrast}/arms_resolving_above_zero"]
        assert resolving.value == 0.0
        assert resolving.verdict == "not_detected"
        # The sign of a point is not a resolution, so it carries no verdict.
        assert rows[f"{PREFIX}/{contrast}/arms_with_a_positive_increment"].verdict is None


def test_a_contrast_missing_an_arm_refuses_rather_than_counting_fewer(tmp_path):
    incomplete = [row for row in blocks()
                  if not (row["arm"] == "arm-0" and row["contrast"] == "P|M")]
    with pytest.raises(LedgerError) as refusal:
        emit(tmp_path, incomplete)
    assert "whole panel" in str(refusal.value)


def test_an_unfinished_group_refuses(tmp_path):
    with pytest.raises(LedgerError):
        emit(tmp_path, blocks(), status="running")


def test_the_retained_panel_claims_every_arm_on_both_contrasts():
    artifacts = Artifacts()
    if not artifacts.exists(builder.PROGRESSION):
        pytest.skip("the information progression is not staged in this checkout")
    ledger = Ledger(artifacts)
    builder.family_information_progression(ledger)
    rows = {quantity.id: quantity for quantity in ledger.quantities}
    arms = rows["information_progression/arms"].value
    for contrast in builder.PROGRESSION_CONTRASTS:
        assert rows[f"{PREFIX}/{contrast}/arms_with_a_positive_increment"].value == arms
        assert rows[f"{PREFIX}/{contrast}/arms_resolving_above_zero"].value == arms
