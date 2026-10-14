"""The depth contrast's cross-cohort counts run over every staged cohort.

``DEPTH_CONTRAST_FILES`` is not only a list of files to read: the cohort count,
the comparison population and the "exceeds on all cohorts" count are reductions
over it. A consolidation that is staged and not named there is therefore worse
than absent -- it silently narrows the population those counts are reported
over, and the narrowed count still reads as a statement about the whole
analysis. The external-confirmation cohort was staged after the other three and
left out, which is exactly that.
"""
from pathlib import Path

import pytest

from scripts.capability.reporting import build_derived_numbers as builder
from src.capability.core.evidence_ledger import Artifacts, Ledger

REPO_ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = REPO_ROOT / builder.CONSOLIDATION_DIR / "depth_contrast"


def test_every_staged_consolidation_is_a_named_cohort():
    if not DIRECTORY.is_dir():
        pytest.skip("the depth-contrast consolidations are not staged in this checkout")
    staged = {f"depth_contrast/{path.name}"
              for path in DIRECTORY.glob("consolidation_*.json")}
    assert staged == {name for name, _ in builder.DEPTH_CONTRAST_FILES}


def test_each_named_cohort_reports_its_own_stated_population():
    """The cross-cohort counts agree with each artifact's own populations block.

    The per-gate count is computed here from the arms' differences, while the
    consolidation states its own figure; where it states one, the two have to
    be the same number or the reduction is not reading the cohort it names.
    """
    artifacts = Artifacts()
    staged = [(name, gate) for name, gate in builder.DEPTH_CONTRAST_FILES
              if artifacts.exists(f"{builder.CONSOLIDATION_DIR}/{name}")]
    if len(staged) < 2:
        pytest.skip("fewer than two depth-contrast consolidations are staged")
    ledger = Ledger(artifacts)
    builder.family_depth_contrast(ledger)
    builder.family_depth_contrast_population(ledger)
    rows = {quantity.id: quantity for quantity in ledger.quantities}

    assert rows["depth_contrast/comparison/cohorts"].value == float(len(staged))
    checked = 0
    for name, gate in staged:
        stated = (artifacts.json(f"{builder.CONSOLIDATION_DIR}/{name}").get("populations")
                  or {}).get("selected_exceeds_both_admitted")
        if stated is None:
            continue
        assert rows[f"depth_contrast/comparison/{gate}/exceeding_both_admitted"].value == \
            float(stated), gate
        checked += 1
    assert checked, "no consolidation states a population to check the reduction against"
