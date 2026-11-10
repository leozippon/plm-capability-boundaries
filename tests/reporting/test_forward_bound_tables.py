"""The forward-bound tables, and the refusals that keep them honest.

The tables are the authoritative statement of the bounds the summary's claim rests on, so the
conditions that must always hold are the ones under which a table may be written at all. A
panel's columns must still be arms and directions, still line up with the artifact's own
per-column record, still carry the statistic the table header names, and still leave the native
half-width wider than every forward point estimate, because the argument that the native band
cannot see an increment of the observed size is that last fact. When any of them stops holding
the emitter has to refuse, not render a table that no longer says what its header says.

The refusals are tested the way the emitter applies them: against a copy of a retained artifact
with one thing changed, read through the same reader the emitter uses. The artifacts are the
retained ones, because the layout the guards are written against is the artifacts' own and a
hand-built stand-in would only test itself, so those tests are skipped where the checkout does
not carry them. The marker and format tests need nothing retained and always run.
"""
import json
import shutil

import pytest

from scripts.capability.reporting import forward_bound_tables as tables

REPO = tables.REPO_ROOT
FORWARD, REVERSE = tables.FORWARD, tables.REVERSE
FIRST, SECOND = "bygpt5-base-en", "bygpt5-medium-en"

#: Where each layout keeps its panel, its two bands and the record that pins a column to its value.
LAYOUTS = {
    "stability": (tables.STABILITY, lambda d: d["panels"]["S|ranking"], "simultaneous"),
    "abundance": (tables.ABUNDANCE, lambda d: d["panel"]["ranking"], "simultaneous"),
    "interaction": (tables.INTERACTION, lambda d: d, "simultaneous_bootstrap"),
}
STUDENTISED = "standardized_maxT_sensitivity"
#: The layouts that keep only the point of each column in their own record, so a native interval
#: can be changed consistently without the record objecting first.
POINT_RECORDS = ("stability", "abundance")


def retained_artifacts_present() -> bool:
    return all((REPO / path).is_file() for path in tables.ARTIFACTS)


@pytest.fixture(scope="module")
def retained():
    if not retained_artifacts_present():
        pytest.skip("the retained campaign panels are not staged in this checkout")
    return tables.render(tables.reader(REPO))


def amended(path, change):
    """A reader that hands the emitter a fresh copy of ``path`` with ``change`` applied to it."""
    base = tables.reader(REPO)

    def read(relative):
        document = base(relative)
        if relative == path:
            change(document)
        return document
    return read


def refused(path, change, match, error=tables.PanelLayoutError):
    with pytest.raises(error, match=match):
        tables.render(amended(path, change))


def parts(layout):
    path, locate, native = LAYOUTS[layout]
    return path, locate, native


def cells(block):
    return [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in block.split("\n")[2:]]


# --- the rendered form ------------------------------------------------------------------------

def test_each_table_has_the_rows_and_columns_its_header_names(retained):
    for name, rows in ((tables.FORWARD_BOUND, 9), (tables.RESOLUTION_COUNTS, 9),
                       (tables.BASELINE_STRUCTURE, 3)):
        lines = retained[name].split("\n")
        width = lines[0].count("|")
        assert len(lines) == 2 + rows and all(line.count("|") == width for line in lines)


def test_a_native_bound_is_its_largest_estimate_plus_the_one_half_width(retained):
    for row in cells(retained[tables.FORWARD_BOUND]):
        estimate = float(row[2].split(",")[0])
        half_width = float(row[3])
        bound = float(row[4].split(",")[0])
        assert estimate < half_width
        assert bound == pytest.approx(estimate + half_width, abs=2e-4), row[0]


def test_a_row_agrees_with_the_panel_it_was_read_from(retained):
    read = tables.reader(REPO)
    panel = read(tables.STABILITY)["panels"]["S2|ranking"]
    forward = [i for i, c in enumerate(panel["columns"]) if c.endswith("|" + FORWARD)]
    level = next(v for c, v in zip(panel["absolute"]["columns"], panel["absolute"]["points"])
                 if c.endswith("|S2"))
    row = {r[0]: r for r in cells(retained[tables.FORWARD_BOUND])}["Stability, ranking (Spearman), S2"]
    bound = max(panel["simultaneous"]["interval"][i][1] for i in forward)
    assert row[1] == f"{level:.4f}"
    assert row[3] == f"{panel['simultaneous']['critical_value']:.4f}"
    assert row[4] == f"{bound:.4f}, {100 * bound / level:.1f}%"
    counts = {r[0]: r for r in cells(retained[tables.RESOLUTION_COUNTS])}["Stability ranking, S2"]
    studentised_low = [panel[STUDENTISED]["interval"][i][0] for i in forward]
    studentised_high = [panel[STUDENTISED]["interval"][i][1] for i in forward]
    assert counts[1] == f"0 / {sum(low > 0 for low in studentised_low)}"
    assert counts[2] == f"0 / {sum(high < 0 for high in studentised_high)}"


def test_the_grid_rows_are_read_from_the_stability_grid_and_agree_with_the_campaign(retained):
    document = tables.reader(REPO)(tables.STABILITY_GRID)["families"]
    row = {r[0]: r for r in cells(retained[tables.BASELINE_STRUCTURE])}["Ranking (Spearman)"]
    matched, profile = document["matched_ranking"], document["profile_ranking"]
    assert row[1] == f"{len(matched['resolved_above_zero'])} → {len(profile['resolved_above_zero'])}"
    assert row[2] == f"{len(matched['resolved_below_zero'])} → {len(profile['resolved_below_zero'])}"


def test_the_rendered_form_does_not_depend_on_the_order_of_a_panels_columns(retained):
    for layout in ("stability", "interaction"):
        path, locate, native = parts(layout)

        def reverse_order(document, locate=locate, native=native):
            panel = locate(document)
            order = list(range(len(panel["columns"])))[::-1]
            panel["columns"] = [panel["columns"][i] for i in order]
            for band in (native, STUDENTISED):
                panel[band]["point"] = [panel[band]["point"][i] for i in order]
                panel[band]["interval"] = [panel[band]["interval"][i] for i in order]
        assert tables.render(amended(path, reverse_order)) == retained, layout


def test_the_documents_carry_what_the_panels_render(retained):
    if not all((REPO / path).is_file() for path in tables.DOCUMENTS):
        pytest.skip("the summary and the run note are held locally and are not in this checkout")
    assert tables.main(["--check"]) == 0


# --- refusals: the columns stop supporting the form ---------------------------------------------

@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_column_that_is_not_an_arm_and_a_direction_is_refused(retained, layout):
    path, locate, _ = parts(layout)
    refused(path, lambda d: locate(d)["columns"].__setitem__(0, f"{FIRST}|sideways"), "unknown direction")
    refused(path, lambda d: locate(d)["columns"].__setitem__(0, FIRST), "'arm\\|direction' labels")
    refused(path, lambda d: locate(d)["columns"].__setitem__(1, locate(d)["columns"][0]), "listed twice")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_an_arm_that_loses_a_column_or_a_whole_arm_is_refused(retained, layout):
    path, locate, native = parts(layout)

    def drop(document, labels):
        panel = locate(document)
        keep = [i for i, c in enumerate(panel["columns"]) if c not in labels]
        panel["columns"] = [panel["columns"][i] for i in keep]
        for band in (native, STUDENTISED):
            panel[band]["point"] = [panel[band]["point"][i] for i in keep]
            panel[band]["interval"] = [panel[band]["interval"][i] for i in keep]
            panel[band]["family_size"] = len(keep)
    refused(path, lambda d: drop(d, {f"{FIRST}|{REVERSE}"}), "lacks one of its two directions")
    refused(path, lambda d: drop(d, {f"{FIRST}|{REVERSE}", f"{FIRST}|{FORWARD}"}), "32 arms, not 33")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_band_without_one_entry_per_column_is_refused(retained, layout):
    path, locate, native = parts(layout)
    refused(path, lambda d: locate(d)[native].__setitem__("family_size", 65), "one entry per column")
    refused(path, lambda d: locate(d)[STUDENTISED]["point"].pop(), "one entry per column")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_columns_that_no_longer_line_up_with_their_values_are_refused(retained, layout):
    """Two arms' labels exchanged while the values stay put: the artifact's own record gives it away."""
    path, locate, _ = parts(layout)

    def exchange(document):
        columns = locate(document)["columns"]
        a, b = columns.index(f"{FIRST}|{FORWARD}"), columns.index(f"{SECOND}|{FORWARD}")
        columns[a], columns[b] = columns[b], columns[a]
    refused(path, exchange, "disagrees with the artifact's own record")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_native_interval_that_is_not_the_one_half_width_is_refused(retained, layout):
    path, locate, native = parts(layout)
    refused(path, lambda d: locate(d)[native]["interval"][3].__setitem__(1, locate(d)[native]["interval"][3][1] + 1e-3),
            "one half-width")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_band_whose_method_stops_naming_its_statistic_is_refused(retained, layout):
    path, locate, native = parts(layout)
    refused(path, lambda d: locate(d)[native].__setitem__("method", "paired group bootstrap"),
            "no longer names the native statistic")
    refused(path, lambda d: locate(d)[STUDENTISED].__setitem__("method", locate(d)[native]["method"]),
            "no longer names the studentised statistic")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_stored_verdict_counts_that_disagree_with_the_intervals_are_refused(retained, layout):
    path, locate, _ = parts(layout)

    def tamper(document):
        verdicts = document["verdict_counts"] if layout == "interaction" else locate(document)["verdicts"]
        counts = verdicts[FORWARD] if layout == "interaction" else verdicts[FORWARD]["counts"]
        counts["positive"] += 1
    refused(path, tamper, "verdict counts")


def test_a_control_level_that_is_missing_or_not_positive_is_refused(retained):
    stability = LAYOUTS["stability"]

    def absent(document):
        stability[1](document)["absolute"]["columns"][0] = f"{FIRST}|M"
    refused(stability[0], absent, "control level is not recorded")

    def zero(document):
        panel = stability[1](document)
        panel["absolute"]["points"][panel["absolute"]["columns"].index(f"{FIRST}|S")] = 0.0
    refused(stability[0], zero, "not positive")
    refused(tables.INTERACTION, lambda d: d["arms"][FIRST]["absolute_mse_kcal2"].pop("U_C"), "no 'U_C'")


# --- refusals: the width claim -------------------------------------------------------------------

def narrow(half_width):
    def change(document):
        band = LAYOUTS["stability"][1](document)["simultaneous"]
        band["critical_value"] = half_width
        band["interval"] = [[p - half_width, p + half_width] for p in band["point"]]
    return change


@pytest.mark.parametrize("layout", POINT_RECORDS)
def test_a_native_half_width_that_no_longer_exceeds_a_forward_estimate_is_not_rendered(retained, layout):
    path, locate, native = parts(layout)

    def change(document):
        band = locate(document)[native]
        band["critical_value"] = 0.001
        band["interval"] = [[p - 0.001, p + 0.001] for p in band["point"]]
    refused(path, change, "width argument has changed", tables.WidthClaimError)


def test_the_width_claim_is_about_the_magnitude_of_an_estimate_not_its_sign(retained):
    path, locate, native = parts("stability")
    label = f"{FIRST}|{FORWARD}"

    def change(document):
        panel = locate(document)
        i = panel["columns"].index(label)
        value = -1.5 * panel[native]["critical_value"]
        panel[native]["point"][i] = panel[STUDENTISED]["point"][i] = value
        panel[native]["interval"][i] = [value - panel[native]["critical_value"], value + panel[native]["critical_value"]]
        panel[STUDENTISED]["interval"][i] = [value - 1e-4, value + 1e-4]
        panel["per_seed_points"][label] = {seed: value for seed in panel["per_seed_points"][label]}
    refused(path, change, "width argument has changed", tables.WidthClaimError)


# --- refusals: the stability grid ---------------------------------------------------------------

def grid(change, match):
    refused(tables.STABILITY_GRID, change, match)


def test_a_grid_baseline_that_no_longer_lists_what_the_table_says_it_lists_is_refused(retained):
    grid(lambda d: d["families"]["profile_ranking"].__setitem__("baseline_blocks", "ident + geom + chem + G"),
         "no longer S2")
    grid(lambda d: d["families"]["matched_ranking"].__setitem__("design", "S2"), "no longer S")


def test_stored_resolved_arms_that_disagree_with_the_grid_intervals_are_refused(retained):
    grid(lambda d: d["families"]["matched_ranking"]["resolved_above_zero"].pop(), "resolved arms disagree")
    grid(lambda d: d["families"]["profile_ranking"]["resolved_below_zero"].pop(), "resolved arms disagree")


def test_a_grid_estimate_that_is_not_the_campaigns_is_refused(retained):
    grid(lambda d: d["families"]["matched_ranking"]["bands"]["point"].__setitem__(0, 0.5), "same fits")


def test_a_grid_band_that_no_longer_names_the_studentised_statistic_is_refused(retained):
    grid(lambda d: d["families"]["matched_ranking"]["bands"].__setitem__("method", "native units"),
         "no longer names the studentised statistic")


def test_a_readout_with_no_arm_resolved_against_s_has_no_mean_to_render(retained):
    def change(document):
        family = document["families"]["matched_transfer_error"]
        family["bands"]["interval"] = [[min(iv[0], -1e-9), iv[1]] for iv in family["bands"]["interval"]]
        family["resolved_above_zero"] = []
    grid(change, "mean estimate of those arms is undefined")


# --- the markers and the formats -----------------------------------------------------------------

def block(name, body):
    return f"{tables.OPENING.format(name=name)}\n\n{body}\n\n{tables.CLOSING.format(name=name)}"


def test_a_block_is_replaced_between_its_markers_and_nothing_else_moves():
    text = f"before\n\n{block('one', 'old one')}\n\nmiddle\n\n{block('two', 'old two')}\n\nafter\n"
    spliced = tables.splice(text, "one", "new one")
    assert spliced == text.replace("old one", "new one")
    assert tables.splice(spliced, "one", "new one") == spliced


@pytest.mark.parametrize("damage", ["no opening", "no closing", "opening twice", "closing before opening"])
def test_a_missing_repeated_or_inverted_marker_is_refused(damage):
    opening, closing = tables.OPENING.format(name="one"), tables.CLOSING.format(name="one")
    text = {"no opening": f"x\n{closing}\n",
            "no closing": f"x\n{opening}\n",
            "opening twice": f"{opening}\nx\n{closing}\n{opening}\n",
            "closing before opening": f"{closing}\nx\n{opening}\n"}[damage]
    with pytest.raises(tables.MarkerError):
        tables.splice(text, "one", "body")


def test_the_formats_that_pin_the_rendered_form():
    assert tables.sig2(0.0002) == "0.00020" and tables.sig2(0.0072) == "0.0072"
    assert tables.sig2(0.0000098) == "0.0000098" and tables.sig2(0) == "0.0000"
    assert tables.percent(0.0344) == "3.4%" and tables.signed(-0.004) == "−0.0040"


def stage_root(root, body="stale"):
    for path in tables.ARTIFACTS:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / path, root / path)
    for relative, names in tables.DOCUMENTS.items():
        text = "intro\n\n" + "\n\n".join(block(name, body) for name in names) + "\n\noutro\n"
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text, encoding="utf-8")


def test_check_names_a_stale_document_and_write_repairs_it_in_place(retained, tmp_path, capsys):
    stage_root(tmp_path)
    before = {r: (tmp_path / r).read_text(encoding="utf-8") for r in tables.DOCUMENTS}
    assert tables.main(["--root", str(tmp_path), "--check"]) == 1
    assert "stale" in capsys.readouterr().err
    assert tables.main(["--root", str(tmp_path), "--write"]) == 0
    assert tables.main(["--root", str(tmp_path), "--check"]) == 0
    for relative, names in tables.DOCUMENTS.items():
        text = (tmp_path / relative).read_text(encoding="utf-8")
        assert text.startswith("intro\n\n") and text.endswith("\n\noutro\n")
        assert all(retained[name] in text for name in names)
        assert text != before[relative]


def test_nothing_is_written_when_a_panel_is_refused(retained, tmp_path, capsys):
    stage_root(tmp_path)
    path = tmp_path / tables.STABILITY
    document = json.loads(path.read_text(encoding="utf-8"))
    band = document["panels"]["S|ranking"]["simultaneous"]
    band["critical_value"] = 0.01
    band["interval"] = [[p - 0.01, p + 0.01] for p in band["point"]]
    path.write_text(json.dumps(document), encoding="utf-8")
    before = {r: (tmp_path / r).read_bytes() for r in tables.DOCUMENTS}
    assert tables.main(["--root", str(tmp_path), "--write"]) == 2
    assert "width argument has changed" in capsys.readouterr().err
    assert {r: (tmp_path / r).read_bytes() for r in tables.DOCUMENTS} == before


@pytest.mark.parametrize("damaged", list(tables.DOCUMENTS))
def test_nothing_is_written_when_a_marker_is_missing_from_either_document(retained, tmp_path, capsys, damaged):
    """The stale document is spliced first or second depending on which one is damaged, and neither is written."""
    stage_root(tmp_path)
    path = tmp_path / damaged
    closing = tables.CLOSING.format(name=tables.DOCUMENTS[damaged][0])
    path.write_text(path.read_text(encoding="utf-8").replace(closing, ""), encoding="utf-8")
    before = {r: (tmp_path / r).read_bytes() for r in tables.DOCUMENTS}
    assert tables.main(["--root", str(tmp_path), "--write"]) == 2
    assert "occurs 0 times" in capsys.readouterr().err
    assert {r: (tmp_path / r).read_bytes() for r in tables.DOCUMENTS} == before
