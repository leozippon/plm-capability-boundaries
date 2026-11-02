"""The guidance and selection tables' cells, and the refusals that hold them.

The defect these tests are written against was not a wrong number in the
manuscript: every printed value was reproduced exactly from a retained artifact,
and forty-eight of the fifty-eight estimate cells still had no ledger row
carrying the (point, interval) pair they print. The first test is therefore the
one that would have caught it --- both committed tables are parsed, every printed
pair is looked up in the ledger at the precision it is printed to, and each has
to resolve to exactly one row whose digest and pointer still answer for it.

The second group fixes what the module may not do. It emits only pointers no row
read before, so a rebuild cannot move a row another module owns; it reads the
per-selector mean and never the per-domain field of the same name; it leaves an
unestimable cell alone, because such a cell prints an em dash and there is no
number to trace; and it refuses rather than guessing when a carried row has
drifted from its own source, when a declared cell does not resolve, when a record
is neither estimated nor unestimable, when the two campaigns disagree on the
declared draw count, when a condition is missing a reference the mean is taken
over, or when the module runs before the modules whose supports its rows sit on.

The last group asserts the readings the audit named one by one, so a change to
either analysis artifact fails here rather than moving a number in the paper.
"""
import json
import re

import pytest

from scripts.capability.reporting import build_derived_numbers as builder
from scripts.capability.reporting import r6_guidance_table_quantities as r6
from scripts.capability.reporting.coupling_guidance_quantities import (
    add_coupling_guidance_quantities)
from scripts.capability.reporting.progen3_robustness_quantities import (
    add_progen3_robustness_quantities)
from src.capability.core.evidence_ledger import Artifacts, Ledger, LedgerError, resolve

TABLES = ("manuscript/submission/tables/supplementary-guidance-selection.tex",
          "manuscript/submission/tables/table-1-guidance-selection.tex")
#: How many estimate cells each committed table prints. The supplement's three
#: tables print fifty-eight and the main table twenty-three; a table that grows
#: or loses a cell fails here, because the roster this module declares would no
#: longer be the set of cells the manuscript prints.
PRINTED = {TABLES[0]: 58, TABLES[1]: 23}
#: One printed estimate: a point and the two ends of its interval, in either
#: table's notation once the typesetting is normalised away.
CELL = re.compile(r"(?<![\d.])(-?\d+\.\d+)\s*\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]")
TYPESETTING = ((r"\textminus{}", "-"), (r"\;", " "), (r"\ ", " "), ("$", " "), ("&", " "))


# --------------------------------------------------------------------------- #
# The ledger the tables are checked against.
# --------------------------------------------------------------------------- #

def r6_ledger(*, with_owners: bool = True) -> Ledger:
    """The R6 rows of the real table, built from the retained artifacts.

    Only the modules whose rows this one reads or sits beside are run, which is
    what keeps the test cheap; the full table is built by ``build_derived_numbers``
    and this module runs last there for the same reason.
    """
    ledger = Ledger(Artifacts())
    if with_owners:
        builder.family_generative_control(ledger)
        add_coupling_guidance_quantities(ledger)
        add_progen3_robustness_quantities(ledger)
    return ledger


@pytest.fixture(scope="module")
def ledger() -> Ledger:
    built = r6_ledger()
    r6.add_r6_guidance_table_quantities(built)
    return built


def printed_cells(path: str) -> list[tuple[str, str, str]]:
    text = (Artifacts().root / path).read_text()
    for pattern, replacement in TYPESETTING:
        text = text.replace(pattern, replacement)
    return CELL.findall(text)


def places(token: str) -> int:
    return len(token.split(".")[1])


def matching(ledger: Ledger, point: str, low: str, high: str) -> list:
    return [quantity for quantity in ledger.quantities
            if quantity.interval is not None
            and round(quantity.value, places(point)) == float(point)
            and round(quantity.interval[0], places(low)) == float(low)
            and round(quantity.interval[1], places(high)) == float(high)]


def pointers(ledger: Ledger) -> set[tuple[str, tuple[str, ...]]]:
    return {(quantity.source_path, quantity.source_pointer) for quantity in ledger.quantities}


def rows(ledger: Ledger) -> dict:
    return {quantity.id: quantity for quantity in ledger.quantities}


# --------------------------------------------------------------------------- #
# Every printed cell resolves to one row.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("path", TABLES)
def test_every_printed_estimate_cell_resolves_to_exactly_one_row(ledger, path):
    cells = printed_cells(path)
    assert len(cells) == PRINTED[path]
    for point, low, high in cells:
        found = matching(ledger, point, low, high)
        assert len(found) == 1, (
            f"{path}: the cell {point} [{low}, {high}] resolves to "
            f"{[quantity.id for quantity in found]}")


def test_each_resolving_row_still_answers_for_its_own_source(ledger):
    artifacts = Artifacts()
    seen = set()
    for path in TABLES:
        for cell in printed_cells(path):
            quantity, = matching(ledger, *cell)
            seen.add(quantity.id)
            assert quantity.source_sha256 == artifacts.sha256(quantity.source_path)
            assert quantity.resampling_unit and quantity.resampling_draws
            pointer = quantity.source_pointer
            if any(segment.startswith("<") for segment in pointer):
                continue
            assert resolve(artifacts.json(quantity.source_path), pointer) == quantity.value
    assert len(seen) == 58


def test_the_module_is_wired_into_the_table_builder():
    source = (Artifacts().root
              / "scripts/capability/reporting/build_derived_numbers.py").read_text()
    assert source.count("add_r6_guidance_table_quantities") == 2


# --------------------------------------------------------------------------- #
# What the module may not do.
# --------------------------------------------------------------------------- #

def test_it_emits_only_pointers_no_row_read_before():
    before = r6_ledger()
    existing = {quantity.id: quantity for quantity in before.quantities}
    read_before = pointers(before)
    r6.add_r6_guidance_table_quantities(before)
    added = [quantity for quantity in before.quantities if quantity.id not in existing]
    assert len(added) == 54
    assert not {(quantity.source_path, quantity.source_pointer)
                for quantity in added} & read_before
    for identifier, quantity in existing.items():
        assert rows(before)[identifier] is quantity


def test_no_two_rows_read_one_pointer(ledger):
    assert len(pointers(ledger)) == len(ledger.quantities)


def test_an_unestimable_cell_prints_an_em_dash_and_gets_no_row(ledger):
    payload = Artifacts().json(r6.CAMPAIGNS[0].path)
    table = rows(ledger)
    unestimable = 0
    for stratum in r6.CAMPAIGNS[0].strata:
        for contrast in r6.CAMPAIGNS[0].contrasts:
            record = payload["primary_tm"][stratum]["contrasts"][contrast]
            if record["status"] == "estimated":
                continue
            unestimable += 1
            assert f"coupling_guidance/primary_tm/{stratum}/{contrast}/difference" not in table
    assert unestimable == 6


def test_a_record_that_is_neither_estimated_nor_unestimable_refuses():
    campaign = r6.CAMPAIGNS[0]
    payload = {"domains": list("ABCDEFGH"),
               "primary_tm": {"pooled": {"contrasts": {"X": {
                   "status": "estimated", "interval95_bca": None, "reason": "", "mean": 0.1,
                   "n_domains": 8, "missing_domains": {}}}}}}
    with pytest.raises(LedgerError, match="neither estimated nor unestimable"):
        r6._record(campaign, payload, "primary_tm", "pooled", "X")


def test_an_unestimable_record_carrying_an_interval_refuses():
    campaign = r6.CAMPAIGNS[0]
    payload = {"domains": list("ABCDEFGH"),
               "primary_tm": {"pooled": {"contrasts": {"X": {
                   "status": "unestimable", "interval95_bca": [0.0, 0.2], "reason": "short",
                   "mean": 0.1, "n_domains": 7, "missing_domains": {"A": "short"}}}}}}
    with pytest.raises(LedgerError, match="ambiguous"):
        r6._record(campaign, payload, "primary_tm", "pooled", "X")


def test_a_cell_short_of_the_domain_set_refuses():
    campaign = r6.CAMPAIGNS[0]
    payload = {"domains": list("ABCDEFGH"),
               "primary_tm": {"pooled": {"contrasts": {"X": {
                   "status": "estimated", "interval95_bca": [0.0, 0.2], "reason": None,
                   "mean": 0.1, "n_domains": 7, "missing_domains": {"A": "short"}}}}}}
    with pytest.raises(LedgerError, match="of the 8 sealed domains"):
        r6._record(campaign, payload, "primary_tm", "pooled", "X")


def test_a_declared_cell_the_artifact_does_not_record_refuses():
    campaign = r6.CAMPAIGNS[0]
    payload = {"domains": list("ABCDEFGH"), "primary_tm": {"pooled": {"contrasts": {}}}}
    with pytest.raises(LedgerError, match="does not resolve"):
        r6._record(campaign, payload, "primary_tm", "pooled", "X")


def test_a_carried_row_that_has_drifted_from_its_source_refuses(ledger):
    carried = rows(ledger)["coupling_guidance/feasibility/pooled/couplings_minus_unguided"]
    r6._agrees_with_source(carried, carried.value, carried.interval)
    with pytest.raises(LedgerError, match="no longer a reading of its own source"):
        r6._agrees_with_source(carried, carried.value + 1.0, carried.interval)
    with pytest.raises(LedgerError, match="no longer a reading of its own source"):
        r6._agrees_with_source(carried, carried.value, None)


def test_it_refuses_when_the_modules_owning_its_supports_have_not_run():
    with pytest.raises(LedgerError, match="is declared by"):
        r6.add_r6_guidance_table_quantities(r6_ledger(with_owners=False))


# --------------------------------------------------------------------------- #
# The selection table: the mean, and not the per-domain field of the same name.
# --------------------------------------------------------------------------- #

def test_selector_rows_read_the_mean_over_the_domains(ledger):
    payload = Artifacts().json(r6.CAMPAIGNS[0].path)
    arms = payload[r6.CAPTURE]["arms"]
    table = rows(ledger)
    for arm, _, segment in r6.ARMS:
        for selector, _ in r6.SELECTORS:
            block = arms[arm]["selectors"][selector]
            identifier = f"coupling_selection/{segment}/{selector}/captured_fraction"
            if identifier not in table:
                # The one per-selector cell another module already binds.
                identifier = "coupling_selection/profile_guided_pool/" \
                             "native_likelihood_captured_fraction"
            quantity = table[identifier]
            assert quantity.value == block[r6.CAPTURED]
            assert tuple(quantity.interval) == tuple(block["interval95_bca"])
            per_domain = {domain["selectors"][selector][r6.PER_DOMAIN_ONLY]
                          for domain in arms[arm]["per_domain"].values()}
            assert quantity.value not in per_domain


def test_a_selector_block_carrying_the_per_domain_field_refuses(tmp_path):
    path = r6.CAMPAIGNS[0].path
    payload = json.loads((Artifacts().root / path).read_bytes())
    block = payload[r6.CAPTURE]["arms"]["N"]["selectors"]["potts_full"]
    block[r6.PER_DOMAIN_ONLY] = block[r6.CAPTURED]
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload))
    with pytest.raises(LedgerError, match=r6.PER_DOMAIN_ONLY):
        r6._selector(Artifacts(tmp_path).json(path), "N", "potts_full")


def test_the_range_bound_over_selectors_and_the_potts_full_estimate_are_distinct(ledger):
    table = rows(ledger)
    bound = table["coupling_selection/profile_guided_pool/"
                  "best_admissible_selector_captured_fraction"]
    estimate = table["coupling_selection/profile_guided_pool/potts_full/captured_fraction"]
    assert bound.value == estimate.value
    assert bound.kind == "range_bound" and bound.interval is None
    assert estimate.kind == "estimate"
    assert [round(end, 3) for end in estimate.interval] == [0.149, 0.284]
    assert bound.source_pointer != estimate.source_pointer


def test_the_unguided_pool_potts_full_fraction_is_the_cell_the_tables_print(ledger):
    quantity = rows(ledger)["coupling_selection/unguided_pool/potts_full/captured_fraction"]
    assert round(quantity.value, 3) == 0.143
    assert [round(end, 3) for end in quantity.interval] == [0.051, 0.223]
    assert quantity.source_pointer == (
        r6.CAPTURE, "arms", "N", "selectors", "potts_full", r6.CAPTURED)


# --------------------------------------------------------------------------- #
# The declared draw count the captions quote.
# --------------------------------------------------------------------------- #

def test_the_draw_count_is_a_quantity_and_not_only_another_row_s_field(ledger):
    quantity = rows(ledger)["coupling_guidance/bca_resampling_draws"]
    assert quantity.value == 9999.0
    assert quantity.kind == "constant" and quantity.unit == "bootstrap draws"
    assert any(quantity.resampling_draws == 9999
               for quantity in ledger.quantities if quantity.interval is not None)


def _statement_root(tmp_path, confirmatory: str, robustness: str):
    for path, statement in ((r6.INTERVAL_STATEMENTS[0][0], confirmatory),
                            (r6.INTERVAL_STATEMENTS[1][0], robustness)):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"frozen": {"interval": statement},
                                      "discipline": {"interval": statement}}))
    ledger = Ledger(Artifacts(tmp_path))
    ledger.declare_support(f"e20_primary_tm/{r6.e20.POOLED}", "a stand-in population")
    return ledger


STATEMENT = "95% BCa bootstrap over domains, 9999 resamples, seed 20261013"


def test_campaigns_that_disagree_on_the_declared_draw_count_refuse(tmp_path):
    ledger = _statement_root(
        tmp_path, STATEMENT, "95% BCa bootstrap over domains, 4000 resamples, seed 20261013")
    with pytest.raises(LedgerError, match="different bias-corrected accelerated draw counts"):
        r6._resampling_draws(ledger, lambda *args, **kwargs: None)


def test_a_statement_the_anchor_does_not_bind_refuses(tmp_path):
    ledger = _statement_root(tmp_path, STATEMENT, "95% percentile bootstrap over domains")
    with pytest.raises(LedgerError, match="does not match exactly once"):
        r6._resampling_draws(ledger, lambda *args, **kwargs: None)


# --------------------------------------------------------------------------- #
# The prose readings the same gap hid.
# --------------------------------------------------------------------------- #

PROSE = {
    "coupling_guidance/primary_tm/pooled/F-M/difference": 0.011066,
    "coupling_guidance/primary_tm/protgpt2/F-M/difference": 0.018314,
    "coupling_guidance/primary_tm/progen2-medium/F-M/difference": 0.003819,
    "coupling_guidance/primary_tm/pooled/F-M/p_holm_signflip": 0.3281,
    "coupling_guidance/primary_tm/pooled/F-M/p_holm_ttest": 0.5651,
    "likelihood_anti_selection/protgpt2/composition_entropy/top_twelve_by_likelihood": 1.480,
    "likelihood_anti_selection/protgpt2/composition_entropy/"
    "size_matched_random_reference": 2.257,
    "generative_control/complete_domain/whole_record_over_fragment_reference": 0.063875,
}


def test_each_prose_reading_rounds_onto_the_numeral_the_supplement_writes(ledger):
    table = rows(ledger)
    text = (Artifacts().root / "manuscript/submission/supplementary-information.tex").read_text()
    for identifier, printed in PROSE.items():
        quantity = table[identifier]
        assert round(quantity.value, places(f"{printed}")) == printed
        assert f"{printed}" in text or f"${printed}$" in text


def test_the_entropy_reference_carries_the_convention_its_artifact_records(ledger):
    quantity = rows(ledger)[
        "likelihood_anti_selection/protgpt2/composition_entropy/size_matched_random_reference"]
    assert quantity.interval_kind == "95% Student-t"
    assert quantity.resampling_unit == "size-matched random draw"
    assert quantity.resampling_draws == 16
    assert quantity.unit == "nats"


def test_a_condition_missing_a_reference_refuses(tmp_path):
    payload = json.loads((Artifacts().root / r6.GATE).read_bytes())
    del payload["cells"][3]["endpoints"][r6.GATE_ENDPOINT]["controls"]["natural"]
    target = tmp_path / r6.GATE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload))
    ledger = Ledger(Artifacts(tmp_path))
    ledger.declare_support(r6.GATE_SUPPORT, "a stand-in population")
    with pytest.raises(LedgerError, match="carries no natural reference"):
        r6._reference_rate_sensitivity(ledger, lambda *args, **kwargs: None)


def test_the_reference_rate_mean_is_the_difference_of_the_two_recorded_means(ledger):
    payload = Artifacts().json(r6.GATE)
    cells = payload["cells"]
    whole, fragment = (
        sum(cell["endpoints"][r6.GATE_ENDPOINT]["controls"][name]["control_rate"]
            for cell in cells) / len(cells)
        for name in r6.GATE_REFERENCES)
    quantity = rows(ledger)[
        "generative_control/complete_domain/whole_record_over_fragment_reference"]
    assert quantity.value == whole - fragment
    assert quantity.interval is None and quantity.no_interval_reason

# --------------------------------------------------------------------------- #
# The cross-support exception: narrow, live, and the same in both places.
# --------------------------------------------------------------------------- #

#: Which contrast rows of which table each anchor is allowed to name, as
#: (table, contrast label, how many rows of that table carry it). Pinning the
#: counts is what makes an edit to a row label or a merged row break here rather
#: than silently widen or retire the exception.
PINNED = {
    "Profile $-$ unguided &": {TABLES[0]: 7, TABLES[1]: 2},
    "Couplings $-$ unguided &": {TABLES[0]: 3, TABLES[1]: 1},
    "Couplings $-$ profile &": {TABLES[0]: 7, TABLES[1]: 2},
    "Fields + couplings $-$ fields &": {TABLES[0]: 3, TABLES[1]: 1},
    "Couplings $-$ permuted &": {TABLES[0]: 7, TABLES[1]: 2},
}


def table_rows(path: str) -> list[str]:
    text = (Artifacts().root / path).read_text()
    return [row for row in text.split(r"\\") if "&" in row]


def test_each_anchor_is_pinned_to_the_rows_it_names():
    assert set(r6.CROSS_SUPPORT_ANCHORS) == set(PINNED)
    for anchor, expected in PINNED.items():
        for path, count in expected.items():
            text = (Artifacts().root / path).read_text()
            holding = [row for row in table_rows(path) if anchor in row]
            assert len(holding) == count, f"{path}: {anchor!r} names {len(holding)} rows"
            # Once per row it names, and nowhere in the file but those rows.
            assert text.count(anchor) == count


def test_no_anchor_names_anything_outside_the_two_guidance_tables():
    submission = Artifacts().root / "manuscript/submission"
    for anchor in r6.CROSS_SUPPORT_ANCHORS:
        elsewhere = sorted(
            path.relative_to(Artifacts().root).as_posix()
            for path in submission.rglob("*.tex")
            if anchor in path.read_text())
        assert elsewhere == sorted(TABLES), f"{anchor!r} also names {elsewhere}"


def test_no_row_binds_two_anchors():
    for path in TABLES:
        for row in table_rows(path):
            held = [anchor for anchor in r6.CROSS_SUPPORT_ANCHORS if anchor in row]
            assert len(held) <= 1, f"{path}: one row binds {held}"


def test_a_dead_or_doubled_anchor_refuses(tmp_path):
    """An anchor that names no row, or one row twice, is a refusal at build time."""
    def stage(replacement: str) -> Ledger:
        for path in r6.GUIDANCE_TABLES:
            text = (Artifacts().root / path).read_text()
            target = tmp_path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text.replace(r6.CROSS_SUPPORT_ANCHORS[0], replacement))
        return Ledger(Artifacts(tmp_path))

    with pytest.raises(LedgerError, match="binds no row"):
        r6._cross_support_anchors(stage("Profile over unguided &"))
    doubled = r6.CROSS_SUPPORT_ANCHORS[0] + " " + r6.CROSS_SUPPORT_ANCHORS[0]
    with pytest.raises(LedgerError, match="binds a row more than once"):
        r6._cross_support_anchors(stage(doubled))


def test_the_committed_policy_carries_exactly_the_declared_exceptions():
    policy = json.loads(
        (Artifacts().root
         / "results/shared/manuscript_evidence_20261004/checker-policy.json").read_text())
    assert policy["cross_support_exceptions"] == r6.cross_support_exceptions()
    # The exception covers that one condition and widens nothing else.
    for name in ("bare_estimate_exceptions", "effective_count_exceptions",
                 "provisional_exceptions", "untraced_accepted"):
        assert policy.get(name, []) == []


# --------------------------------------------------------------------------- #
# The prose the band was added to.
# --------------------------------------------------------------------------- #

def test_the_random_reference_band_is_quoted_with_its_own_convention(ledger):
    quantity = rows(ledger)[
        "likelihood_anti_selection/protgpt2/composition_entropy/size_matched_random_reference"]
    low, high = (round(end, 3) for end in quantity.interval)
    text = (Artifacts().root / "manuscript/submission/supplementary-information.tex").read_text()
    # The paragraph, not a split on "." --- every number in it carries a decimal point.
    sentence, = [line for line in text.split("\n") if "composition entropy" in line]
    assert f"2.257 [{low}, {high}] nats" in sentence
    # The band is a Student-t interval over the sixteen size-matched draws, not
    # the ten-thousand-draw bootstrap the same record holds for other fields.
    assert "Student-$t$" in sentence and "sixteen" in sentence
    assert "bootstrap" in sentence
