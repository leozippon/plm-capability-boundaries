"""The realised coverage at the resampling floor, and what its row may claim.

The number is prose in every text that carries it, so the tests fix the three
things an anchor-bound row has to get right. The anchor has to match exactly once
in each text, because a number bound to an anchor matching twice is bound to
neither occurrence. Every text has to give the same value, because the campaigns
cite the resampling module by name and a disagreement means they no longer do.
And the claim has to say that a percentile interval was simulated, not a
bias-corrected accelerated one, because the manuscript quotes the figure beside
BCa bounds and a row that said otherwise would licence the stronger sentence.

Anything the anchor cannot pin down produces a gap and no row, and the absence of
a machine-readable record is a gap of its own even on the happy path.
"""
import json

import pytest

from scripts.capability.reporting import interval_coverage_quantities as coverage
from src.capability.core.evidence_ledger import Artifacts, Ledger, LedgerError
from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS

NOTE = ("BCa over 8 domains: realised coverage of a nominal 95% percentile interval is about "
        "{value} at eight units (src.capability.core.statistics); the interval is "
        "anti-conservative, so read the bound as optimistic, not exact.")
MODULE = ("#: Measured here on 1200 normal samples per point, the realised coverage of a nominal\n"
          "#: 95% percentile interval over the cluster mean is 0.50 at two units, 0.74 at three,\n"
          "#: {value} at eight, 0.92 at sixteen and 0.94 at four hundred.\n"
          "MINIMUM_BOOTSTRAP_UNITS = 8\n")


def stage(tmp_path, *, module: str | None = None, notes: tuple[str, ...] = ("0.89", "0.89"),
          floors: tuple[int, int] = (MINIMUM_BOOTSTRAP_UNITS, MINIMUM_BOOTSTRAP_UNITS)) -> Ledger:
    write(tmp_path, coverage.STATISTICS,
          MODULE.format(value="0.89") if module is None else module)
    confirmatory, robustness = (entry[0] for entry in coverage.CAMPAIGNS)
    write(tmp_path, confirmatory, json.dumps({
        "frozen": {"interval_coverage_note": NOTE.format(value=notes[0])},
        "minimum_bootstrap_units": floors[0]}))
    write(tmp_path, robustness, json.dumps({
        "discipline": {"interval_coverage_note": NOTE.format(value=notes[1]),
                       "minimum_bootstrap_units": floors[1]}}))
    ledger = Ledger(Artifacts(tmp_path))
    coverage.add_interval_coverage_quantities(ledger)
    return ledger


def write(root, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def rows(ledger: Ledger) -> dict:
    return {quantity.id: quantity for quantity in ledger.quantities}


ROW = "interval_coverage/percentile_95/realised_at_the_eight_unit_floor"


def test_the_coverage_is_read_once_from_the_texts_that_agree(tmp_path):
    ledger = stage(tmp_path)
    row = rows(ledger)[ROW]
    assert row.value == 0.89
    assert row.unit == coverage.COVERAGE and row.kind == "artifact_leaf"
    assert row.interval is None and row.resampling_unit is None and row.seed_set == ()
    # The row reads the frozen confirmatory analysis, by the anchor it is bound to.
    assert row.source_path == coverage.CAMPAIGNS[0][0]
    assert row.source_pointer == ("frozen", "interval_coverage_note", "<anchor>",
                                  coverage.NOTE_ANCHOR.pattern)
    assert row.source_sha256 == Artifacts(tmp_path).sha256(coverage.CAMPAIGNS[0][0])
    # One row only: the coverage at every other unit count is not quotable.
    assert len(ledger.quantities) == 1


def test_the_claim_and_support_say_a_percentile_interval_was_simulated(tmp_path):
    ledger = stage(tmp_path)
    claim = rows(ledger)[ROW].claim
    assert "nominal 95% percentile interval over the cluster mean" in claim
    assert "not a measured coverage of a BCa interval" in claim
    assert "no campaign resamples it" in claim
    description = ledger.supports["interval_coverage/eight_unit_floor"].description
    assert "1,200 normal samples per point" in description
    assert "not a measured coverage of those bounds" in description
    assert "Nothing is resampled here" in description
    # Every text the figure is anchored in is named with its digest.
    artifacts = Artifacts(tmp_path)
    for path in (coverage.STATISTICS, *(entry[0] for entry in coverage.CAMPAIGNS)):
        assert f"{path} (sha256 {artifacts.sha256(path)})" in description


def test_the_missing_machine_readable_record_is_a_gap(tmp_path):
    gaps = {gap.id: gap for gap in stage(tmp_path).gaps}
    gap = gaps["interval_coverage/percentile_95/report"]
    assert "no retained artifact and no module constant records the coverage table" in gap.reason
    assert gap.reachability == "not_recorded" and gap.searched


def test_an_anchor_that_does_not_match_exactly_once_gives_a_gap_and_no_row(tmp_path):
    twice = MODULE.format(value="0.89") + "#: and again 0.89 at eight, which is a second anchor.\n"
    ledger = stage(tmp_path, module=twice)
    assert ROW not in rows(ledger)
    gap = {entry.id: entry for entry in ledger.gaps}[ROW]
    assert "does not match exactly once" in gap.reason and coverage.STATISTICS in gap.reason
    # A module that no longer records the coverage beside the floor is the same
    # failure, not a quieter number.
    ledger = stage(tmp_path, module="MINIMUM_BOOTSTRAP_UNITS = 8\n")
    assert ROW not in rows(ledger)
    assert "does not match exactly once" in {entry.id: entry for entry in ledger.gaps}[ROW].reason


def test_texts_that_give_different_figures_give_a_gap_and_no_row(tmp_path):
    ledger = stage(tmp_path, notes=("0.89", "0.91"))
    assert ROW not in rows(ledger)
    gap = {entry.id: entry for entry in ledger.gaps}[ROW]
    assert "there is no one coverage to quote" in gap.reason
    assert "0.89" in gap.reason and "0.91" in gap.reason


def test_a_campaign_publishing_above_another_floor_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, floors=(MINIMUM_BOOTSTRAP_UNITS, 6))
    message = str(refusal.value)
    assert "is not one statement about one floor" in message
    assert str(MINIMUM_BOOTSTRAP_UNITS) in message


def test_the_retained_texts_carry_the_figure_the_caption_quotes():
    artifacts = Artifacts()
    for path in (coverage.STATISTICS, *(entry[0] for entry in coverage.CAMPAIGNS)):
        if not artifacts.exists(path):
            pytest.skip(f"{path} is not present in this checkout")
    ledger = Ledger(artifacts)
    coverage.add_interval_coverage_quantities(ledger)
    # "whose realised coverage at eight units is about 0.89, so they read optimistic".
    assert rows(ledger)[ROW].value == 0.89
    assert MINIMUM_BOOTSTRAP_UNITS == 8
