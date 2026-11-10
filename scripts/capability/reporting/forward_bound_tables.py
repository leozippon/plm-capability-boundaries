#!/usr/bin/env python3
"""The forward bound, the resolution counts and the baseline structure behind the reading of
three archived campaigns, rendered from their stored panels.

The summary says the project's claim rests on the forward bound -- the most a model can add to
its controls, as a fraction of what the controls achieve -- and not on whether a forward
increment resolves. The thirteenth pre-registration amendment makes the generated table the
authoritative statement of every such quantity, and a table is authoritative only when
something in the repository regenerates it from the artifacts it names. This module is that
something. It reads the stored panels of the 2026-11-06 stability, abundance and interaction
bidirectional campaigns and the stability grid that stood behind the manuscript's stability
reading, and renders three blocks:

    forward-bound                 summary: per panel, what the controls reach, the largest forward
                                  point estimate, the native half-width and bound, and the
                                  studentised half-widths and bound
    stability-baseline-structure  summary: the stability grid's counts and estimates against the
                                  baseline that lists no profile (S) and the one that lists
                                  composition and the bounded profile block (S2)
    resolution-counts             run note: per panel, how many of the 33 arms resolve positive,
                                  negative and, in the reverse direction, positive under each band,
                                  and a last row that sums the panels

Each block lives between its own pair of markers in the document that carries it, and
regenerating replaces exactly what lies between them. The prose around a block is written by
hand and points at it; the block is the statement of the numbers.

Definitions. A forward column is the gain of the joint arm over the control set alone, a reverse
column its gain over the model scalar alone. A column resolves positive when its interval's
lower limit lies above zero and negative when its upper limit lies below it. The forward bound
of a panel is the largest upper limit over its forward columns, in native units and as a
fraction of what the control set reaches (the error it leaves, for an error endpoint). The
native band has one half-width for the whole family; the studentised band has one per column.

The width claim. The native half-width is set by the widest column of a family that holds both
directions, and the argument that the native band cannot see a forward increment of the
observed size rests on it exceeding every forward point estimate in magnitude. That is asserted
here for every panel, not sampled: if one ever fails, the width argument has changed and the
module refuses to render (WidthClaimError) rather than render tables that no longer support it.

Refusals. The module also refuses (PanelLayoutError) when a panel's columns stop supporting
the rendered form: a column that is not an arm and a direction, a repeated column, an unknown
direction, an arm missing a direction, the wrong number of arms, a band without one entry per
column, a method string that no longer names the statistic the table header gives the band, a
native interval that is not its point plus and minus its one critical value, a point that
disagrees with the artifact's own per-column record of it, stored verdict counts that disagree
with the intervals, a control level that is absent or not positive, and, for the stability grid,
a baseline whose declared blocks are not the ones the table names, stored resolved lists that
disagree with its intervals, or point estimates that differ from the campaign's. A marker that
is missing, repeated or out of order is refused as well (MarkerError). Nothing is written when
anything is refused.

Run from the validated ct environment, at the repository root:

    python scripts/capability/reporting/forward_bound_tables.py            print the blocks
    python scripts/capability/reporting/forward_bound_tables.py --check    exit 1 if a block is stale
    python scripts/capability/reporting/forward_bound_tables.py --write    regenerate the blocks in place
"""
from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

REPO_ROOT = Path(__file__).resolve().parents[3]
EMITTER = "scripts/capability/reporting/forward_bound_tables.py"

CAMPAIGNS = "results/extensions/bidirectional_ablation_20261106/"
STABILITY = CAMPAIGNS + "r2/stability-native-bands/panel.json"
ABUNDANCE = CAMPAIGNS + "r2/abundance-bidirectional/panel.json"
INTERACTION = CAMPAIGNS + "r3-bidirectional/results.json"
STABILITY_GRID = ("results/external_baseline/20261008095614_7e8ae4b18a4d/"
                  "stability_ranking_error_grid_cpu/panel.json")
ARTIFACTS = (STABILITY, ABUNDANCE, INTERACTION, STABILITY_GRID)

SUMMARY = "docs/summary.md"
RUN_NOTE = "results/extensions/explicit_statistics_lanes_20261110/run-note.md"

FORWARD_BOUND = "forward-bound"
BASELINE_STRUCTURE = "stability-baseline-structure"
RESOLUTION_COUNTS = "resolution-counts"
#: Which blocks each document carries.
DOCUMENTS = {SUMMARY: (FORWARD_BOUND, BASELINE_STRUCTURE), RUN_NOTE: (RESOLUTION_COUNTS,)}

OPENING = "<!-- generated: {name}; " + EMITTER + " -->"
CLOSING = "<!-- end generated: {name} -->"

FORWARD = "model_given_controls"
REVERSE = "controls_given_model"
ARMS = 33
CONFIDENCE = 0.95
#: Two stored copies of one number agree to this, or the columns no longer line up.
TOLERANCE = 1e-9
#: The words each band's own method string has to carry for the table to name it as it does.
NATIVE_WORDS = ("native units", "no SE division")
STUDENTISED_WORDS = ("fixed group SE",)

STABILITY_READOUTS = (
    ("ranking", "Stability, ranking (Spearman)", "Stability ranking"),
    ("transfer_error", "Stability, transfer error (kcal²/mol²)", "Stability transfer error"),
    ("calibrated_error", "Stability, calibrated error (kcal²/mol²)", "Stability calibrated error"),
)
ABUNDANCE_READOUTS = (
    ("ranking", "Abundance, ranking (Spearman), archived set", "Abundance ranking, archived set"),
    ("squared_error", "Abundance, squared normalised fitness, archived set",
     "Abundance squared error, archived set"),
)
INTERACTION_LABELS = ("Interaction, cycle squared error (kcal²/mol²), archived set",
                      "Interaction cycle error, archived set")

BOUND_ORDER = tuple(f"stability {base} {readout}" for base in ("S", "S2")
                    for readout, _, _ in STABILITY_READOUTS) + (
    "abundance ranking", "abundance squared_error", "interaction")
COUNT_ORDER = ("abundance ranking", "abundance squared_error") + tuple(
    f"stability {base} {readout}" for base in ("S", "S2") for readout, _, _ in STABILITY_READOUTS) + (
    "interaction",)

#: The stability grid's two baselines, by the family prefix and the blocks the grid declares.
GRID_BASELINES = (("matched", "S", "ident + geom + chem + G"),
                  ("profile", "S2", "ident + geom + comp + chem + prof2 + G"))
GRID_READOUTS = (("ranking", "Ranking (Spearman)"),
                 ("calibrated_error", "Calibrated error (kcal²/mol²)"),
                 ("transfer_error", "Transfer error (kcal²/mol²)"))

BOUND_HEADER = (
    "| Panel and control set | Controls reach | Largest forward point estimate | Native half-width | "
    "Native bound (declared primary) | Studentised half-widths, narrowest to widest | "
    "Studentised bound (declared sensitivity) |",
    "| --- | --- | --- | --- | --- | --- | --- |")
COUNT_HEADER = (
    "| Panel and control set | Forward positive, against the claim (native / studentised) | "
    "Forward negative, for the claim (native / studentised) | "
    "Reverse resolved, for the claim (native / studentised) |",
    "| --- | --- | --- | --- |")
STRUCTURE_HEADER = (
    "| Stability readout | Arms resolved above zero, S → S2 | Arms resolved below zero, S → S2 | "
    "Mean forward estimate of the arms resolved against S, under S → under S2 | "
    "Largest forward estimate, against S → against S2 |",
    "| --- | --- | --- | --- | --- |")


class PanelLayoutError(ValueError):
    """A stored panel's columns no longer support the rendered form."""


class WidthClaimError(PanelLayoutError):
    """The native half-width no longer exceeds every forward point estimate in magnitude."""


class MarkerError(ValueError):
    """A generated block's markers are missing, repeated or out of order."""


@dataclass(frozen=True)
class Column:
    point: float
    native: tuple[float, float]
    studentised: tuple[float, float]


@dataclass(frozen=True)
class Panel:
    name: str
    bound_label: str
    count_label: str
    half_width: float
    level: dict[str, float]
    forward: dict[str, Column]
    reverse: dict[str, Column]


def reader(root: Path):
    """Read a stored artifact, by path from ``root``, afresh each time it is asked for."""
    def read(relative: str):
        return json.loads((root / relative).read_text(encoding="utf-8"))
    return read


def _number(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PanelLayoutError(f"{what} is not a finite number")
    return float(value)


def _interval(value, what: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise PanelLayoutError(f"{what} is not a lower and an upper limit")
    low, high = _number(value[0], what), _number(value[1], what)
    if low > high:
        raise PanelLayoutError(f"{what} has its lower limit above its upper limit")
    return low, high


def _get(mapping, key, name: str):
    if not isinstance(mapping, dict) or key not in mapping:
        raise PanelLayoutError(f"{name}: no {key!r}")
    return mapping[key]


def _panel(name, bound_label, count_label, columns, native, studentised, level, record, stored):
    """Validate one stored panel against everything the rendered form relies on."""
    def refuse(why):
        return PanelLayoutError(f"{name}: {why}")

    if not isinstance(columns, list) or not all(isinstance(c, str) and c.count("|") == 1 for c in columns):
        raise refuse("the columns are not 'arm|direction' labels")
    if len(set(columns)) != len(columns):
        raise refuse("a column is listed twice")
    pairs = [tuple(c.split("|")) for c in columns]
    unknown = sorted({d for _, d in pairs} - {FORWARD, REVERSE})
    if unknown:
        raise refuse(f"unknown direction {unknown}")
    arms = tuple(dict.fromkeys(arm for arm, _ in pairs))
    if len(arms) != ARMS:
        raise refuse(f"{len(arms)} arms, not {ARMS}")
    if set(columns) != {f"{arm}|{d}" for arm in arms for d in (FORWARD, REVERSE)}:
        raise refuse("an arm lacks one of its two directions")

    for band, block, words in (("native", native, NATIVE_WORDS), ("studentised", studentised, STUDENTISED_WORDS)):
        for key in ("method", "family_size", "confidence", "critical_value", "point", "interval"):
            _get(block, key, f"{name}: the {band} block")
        if not all(word in block["method"] for word in words):
            raise refuse(f"the {band} block's method string no longer names the {band} statistic: "
                         f"{block['method']!r}")
        if not (block["family_size"] == len(block["point"]) == len(block["interval"]) == len(columns)):
            raise refuse(f"the {band} block does not carry one entry per column ({len(columns)})")
        if block["confidence"] != CONFIDENCE:
            raise refuse(f"the {band} block is not a {CONFIDENCE:.0%} band")

    half_width = _number(native["critical_value"], f"{name}: the native critical value")
    if half_width <= 0:
        raise refuse("the native critical value is not positive")
    if not isinstance(level, dict) or set(level) != set(arms):
        raise refuse("the control level is not recorded for exactly the panel's arms")
    for arm, value in level.items():
        if _number(value, f"{name}: the control level of {arm}") <= 0:
            raise refuse(f"the control level of {arm} is not positive")

    parsed: dict[str, Column] = {}
    for i, label in enumerate(columns):
        point = _number(native["point"][i], f"{name}: the point of {label}")
        if abs(_number(studentised["point"][i], f"{name}: the point of {label}") - point) > TOLERANCE:
            raise refuse(f"{label}: the two bands carry different points")
        n_low, n_high = _interval(native["interval"][i], f"{name}: the native interval of {label}")
        s_low, s_high = _interval(studentised["interval"][i], f"{name}: the studentised interval of {label}")
        if abs(n_low - (point - half_width)) > TOLERANCE or abs(n_high - (point + half_width)) > TOLERANCE:
            raise refuse(f"{label}: the native interval is not its point plus and minus the one half-width")
        if not s_low <= point <= s_high:
            raise refuse(f"{label}: the studentised interval does not contain its point")
        if label not in record:
            raise refuse(f"the artifact keeps no per-column record of {label}")
        recorded_point, recorded_native, recorded_studentised = record[label]
        if abs(_number(recorded_point, f"{name}: the recorded point of {label}") - point) > TOLERANCE:
            raise refuse(f"{label}: the point disagrees with the artifact's own record of it, so the "
                         "columns and their values no longer line up")
        for band, mine, theirs in (("native", (n_low, n_high), recorded_native),
                                   ("studentised", (s_low, s_high), recorded_studentised)):
            if theirs is not None and max(abs(mine[0] - theirs[0]), abs(mine[1] - theirs[1])) > TOLERANCE:
                raise refuse(f"{label}: the {band} interval disagrees with the artifact's own record of it")
        parsed[label] = Column(point, (n_low, n_high), (s_low, s_high))

    forward = {arm: parsed[f"{arm}|{FORWARD}"] for arm in arms}
    reverse = {arm: parsed[f"{arm}|{REVERSE}"] for arm in arms}
    for arm, column in forward.items():
        if abs(column.point) >= half_width:
            raise WidthClaimError(
                f"{name}: the native half-width {half_width:.6f} does not exceed the forward point estimate "
                f"{column.point:.6f} of {arm}, so the width argument has changed and nothing is rendered")
    for direction, group in ((FORWARD, forward), (REVERSE, reverse)):
        counts = {"positive": sum(c.native[0] > 0 for c in group.values()),
                  "negative": sum(c.native[1] < 0 for c in group.values())}
        counts["unresolved"] = len(group) - counts["positive"] - counts["negative"]
        if _get(stored, direction, name) != counts:
            raise refuse(f"the stored {direction} verdict counts {stored[direction]} disagree with the "
                         f"native intervals, which give {counts}")

    return Panel(name, bound_label, count_label, half_width, dict(level), forward, reverse)


def _levels(absolute, label: str, name: str) -> dict[str, float]:
    columns, points = _get(absolute, "columns", name), _get(absolute, "points", name)
    if not (isinstance(columns, list) and isinstance(points, list) and len(columns) == len(points)):
        raise PanelLayoutError(f"{name}: the absolute block does not carry one point per column")
    level: dict[str, float] = {}
    for column, value in zip(columns, points):
        arm, _, quantity = str(column).partition("|")
        if quantity == label:
            if arm in level:
                raise PanelLayoutError(f"{name}: the control level of {arm} is listed twice")
            level[arm] = value
    return level


def _seed_record(panel, name: str) -> dict[str, tuple]:
    per_seed = _get(panel, "per_seed_points", name)
    record = {}
    for column, seeds in per_seed.items():
        if not isinstance(seeds, dict) or not seeds:
            raise PanelLayoutError(f"{name}: the per-seed points of {column} are not recorded")
        record[column] = (fmean(_number(v, f"{name}: a per-seed point of {column}") for v in seeds.values()),
                          None, None)
    return record


def _stored_counts(verdicts, name: str) -> dict:
    return {direction: _get(_get(verdicts, direction, name), "counts", name) for direction in (FORWARD, REVERSE)}


def _stability_panels(read) -> dict[str, Panel]:
    document = read(STABILITY)
    panels = _get(document, "panels", STABILITY)
    out = {}
    for base in ("S", "S2"):
        for readout, bound_name, count_name in STABILITY_READOUTS:
            name = f"{STABILITY} [{base}|{readout}]"
            raw = _get(panels, f"{base}|{readout}", STABILITY)
            out[f"stability {base} {readout}"] = _panel(
                name, f"{bound_name}, {base}", f"{count_name}, {base}", _get(raw, "columns", name),
                _get(raw, "simultaneous", name), _get(raw, "standardized_maxT_sensitivity", name),
                _levels(_get(raw, "absolute", name), base, name), _seed_record(raw, name),
                _stored_counts(_get(raw, "verdicts", name), name))
    return out


def _abundance_panels(read) -> dict[str, Panel]:
    document = read(ABUNDANCE)
    panels = _get(document, "panel", ABUNDANCE)
    out = {}
    for readout, bound_label, count_label in ABUNDANCE_READOUTS:
        name = f"{ABUNDANCE} [{readout}]"
        raw = _get(panels, readout, ABUNDANCE)
        out[f"abundance {readout}"] = _panel(
            name, bound_label, count_label, _get(raw, "columns", name), _get(raw, "simultaneous", name),
            _get(raw, "standardized_maxT_sensitivity", name), _levels(_get(raw, "absolute", name), "C", name),
            _seed_record(raw, name), _stored_counts(_get(raw, "verdicts", name), name))
    return out


def _interaction_panel(read) -> dict[str, Panel]:
    document = read(INTERACTION)
    name = INTERACTION
    arms = _get(document, "arms", name)
    level, record = {}, {}
    for arm, entry in arms.items():
        level[arm] = _get(_get(_get(entry, "absolute_mse_kcal2", name), "U_C", name), "point", name)
        for direction in (FORWARD, REVERSE):
            contrast = _get(_get(entry, "contrasts", name), direction, name)
            record[f"{arm}|{direction}"] = (
                _get(contrast, "mse_reduction", name),
                _interval(_get(contrast, "simultaneous95_ci", name), f"{name}: the recorded native interval"),
                _interval(_get(contrast, "standardized_maxT_sensitivity95_ci", name),
                          f"{name}: the recorded studentised interval"))
    return {"interaction": _panel(
        name, *INTERACTION_LABELS, _get(document, "columns", name), _get(document, "simultaneous_bootstrap", name),
        _get(document, "standardized_maxT_sensitivity", name), level, record,
        _get(document, "verdict_counts", name))}


def load_panels(read) -> dict[str, Panel]:
    """Every campaign panel, validated; any layout or width failure raises before anything is rendered."""
    panels = {}
    for loader in (_stability_panels, _abundance_panels, _interaction_panel):
        panels.update(loader(read))
    return panels


def sig2(value: float) -> str:
    """Fixed notation with at least four decimals and two significant figures."""
    if value == 0:
        return "0.0000"
    decimals = max(4, -int(math.floor(math.log10(abs(value)))) + 1)
    return f"{value:.{decimals}f}"


def percent(fraction: float) -> str:
    return f"{100 * fraction:.1f}%"


def signed(value: float) -> str:
    return f"{value:.4f}".replace("-", "−")


def _bound_row(panel: Panel) -> str:
    level, forward = panel.level, panel.forward
    top = max(forward, key=lambda arm: forward[arm].point)
    native_arm = max(forward, key=lambda arm: forward[arm].native[1] / level[arm])
    studentised_arm = max(forward, key=lambda arm: forward[arm].studentised[1] / level[arm])
    half_widths = [(c.studentised[1] - c.studentised[0]) / 2 for c in forward.values()]
    low, high = min(level.values()), max(level.values())
    reach = f"{low:.4f}" if f"{low:.4f}" == f"{high:.4f}" else f"{low:.4f} to {high:.4f}"
    native, studentised = forward[native_arm].native[1], forward[studentised_arm].studentised[1]
    cells = [panel.bound_label, reach,
             f"{forward[top].point:.4f}, {percent(forward[top].point / level[top])}",
             f"{panel.half_width:.4f}",
             f"{native:.4f}, {percent(native / level[native_arm])}",
             f"{sig2(min(half_widths))} to {sig2(max(half_widths))}",
             f"{studentised:.4f}, {percent(studentised / level[studentised_arm])}"]
    return "| " + " | ".join(cells) + " |"


def _counts(panel: Panel) -> tuple[int, ...]:
    """Forward positive, forward negative and reverse positive, each native then studentised."""
    forward, reverse = list(panel.forward.values()), list(panel.reverse.values())
    return (sum(c.native[0] > 0 for c in forward), sum(c.studentised[0] > 0 for c in forward),
            sum(c.native[1] < 0 for c in forward), sum(c.studentised[1] < 0 for c in forward),
            sum(c.native[0] > 0 for c in reverse), sum(c.studentised[0] > 0 for c in reverse))


def _count_row(label: str, counts: tuple[int, ...]) -> str:
    cells = [label, f"{counts[0]} / {counts[1]}", f"{counts[2]} / {counts[3]}", f"{counts[4]} / {counts[5]}"]
    return "| " + " | ".join(cells) + " |"


def render_bound_table(panels: dict[str, Panel]) -> str:
    return "\n".join(BOUND_HEADER + tuple(_bound_row(panels[key]) for key in BOUND_ORDER))


def render_count_table(panels: dict[str, Panel]) -> str:
    """One row per panel, then one row that sums them, so no total has to be typed beside the table."""
    rows = [_count_row(panels[key].count_label, _counts(panels[key])) for key in COUNT_ORDER]
    total = tuple(sum(column) for column in zip(*(_counts(panels[key]) for key in COUNT_ORDER)))
    label = f"All {len(COUNT_ORDER)} panels, summed ({len(COUNT_ORDER) * ARMS} columns per cell)"
    return "\n".join(COUNT_HEADER + tuple(rows) + (_count_row(label, total),))


def _grid_family(document, prefix: str, readout: str, design: str, blocks: str, campaign: Panel):
    key = f"{prefix}_{readout}"
    name = f"{STABILITY_GRID} [{key}]"
    family = _get(_get(document, "families", STABILITY_GRID), key, STABILITY_GRID)
    if family.get("design") != design or family.get("baseline_blocks") != blocks:
        raise PanelLayoutError(f"{name}: the baseline is no longer {design}, which lists {blocks}")
    bands = _get(family, "bands", name)
    columns = _get(family, "columns", name)
    if not (isinstance(columns, list) and len(set(columns)) == len(columns) == ARMS):
        raise PanelLayoutError(f"{name}: the columns are not {ARMS} distinct arms")
    if not (bands.get("family_size") == len(bands.get("point", ())) == len(bands.get("interval", ())) == ARMS):
        raise PanelLayoutError(f"{name}: the band does not carry one entry per arm")
    if "fixed group SE" not in str(bands.get("method")):
        raise PanelLayoutError(f"{name}: the band's method string no longer names the studentised statistic")
    if bands.get("confidence") != CONFIDENCE:
        raise PanelLayoutError(f"{name}: the band is not a {CONFIDENCE:.0%} band")
    points = dict(zip(columns, (_number(p, f"{name}: a point") for p in bands["point"])))
    intervals = dict(zip(columns, (_interval(iv, f"{name}: an interval") for iv in bands["interval"])))
    above = {arm for arm, iv in intervals.items() if iv[0] > 0}
    below = {arm for arm, iv in intervals.items() if iv[1] < 0}
    if above != set(_get(family, "resolved_above_zero", name)) or below != set(_get(family, "resolved_below_zero", name)):
        raise PanelLayoutError(f"{name}: the stored resolved arms disagree with the stored intervals")
    for arm in columns:
        if arm not in campaign.forward or abs(points[arm] - campaign.forward[arm].point) > TOLERANCE:
            raise PanelLayoutError(f"{name}: the point estimate of {arm} is not the campaign's, so the two "
                                   "artifacts no longer carry the same fits")
    return points, above, below


def render_structure(read, panels: dict[str, Panel]) -> str:
    """The stability grid's counts and estimates against S and against S2, per readout."""
    document = read(STABILITY_GRID)
    rows = []
    for readout, label in GRID_READOUTS:
        found = {}
        for prefix, base, blocks in GRID_BASELINES:
            found[base] = _grid_family(document, prefix, readout, base, blocks, panels[f"stability {base} {readout}"])
        (points_s, above_s, below_s), (points_s2, above_s2, below_s2) = found["S"], found["S2"]
        if not above_s:
            raise PanelLayoutError(f"{STABILITY_GRID} [{readout}]: no arm resolves above zero against S, so the "
                                   "mean estimate of those arms is undefined")
        cells = [label, f"{len(above_s)} → {len(above_s2)}", f"{len(below_s)} → {len(below_s2)}",
                 f"{signed(fmean(points_s[a] for a in above_s))} → {signed(fmean(points_s2[a] for a in above_s))}",
                 f"{signed(max(points_s.values()))} → {signed(max(points_s2.values()))}"]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(STRUCTURE_HEADER + tuple(rows))


def render(read) -> dict[str, str]:
    """Every block, by name. Raises before returning anything if any panel is refused."""
    panels = load_panels(read)
    return {FORWARD_BOUND: render_bound_table(panels),
            BASELINE_STRUCTURE: render_structure(read, panels),
            RESOLUTION_COUNTS: render_count_table(panels)}


def splice(text: str, name: str, body: str) -> str:
    """Replace the block ``name`` between its markers; refuse unless each marker occurs exactly once."""
    opening, closing = OPENING.format(name=name), CLOSING.format(name=name)
    for marker in (opening, closing):
        if text.count(marker) != 1:
            raise MarkerError(f"{marker!r} occurs {text.count(marker)} times, not once")
    start, end = text.index(opening), text.index(closing)
    if end < start:
        raise MarkerError(f"the closing marker of {name!r} comes before its opening marker")
    return text[:start] + f"{opening}\n\n{body}\n\n{closing}" + text[end + len(closing):]


def _write_atomically(path: Path, text: str) -> None:
    """Replace ``path`` in one step, so a reader never sees half a document."""
    mode = stat.S_IMODE(path.stat().st_mode)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
    try:
        os.chmod(handle.name, mode)
        os.replace(handle.name, path)
    except OSError:
        os.unlink(handle.name)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render the forward-bound tables from the stored panels.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="exit 1 if a document's block is stale")
    mode.add_argument("--write", action="store_true", help="regenerate the blocks in place")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root (default: this one)")
    args = parser.parse_args(argv)
    try:
        blocks = render(reader(args.root))
        if not (args.check or args.write):
            for relative, names in DOCUMENTS.items():
                for name in names:
                    print(f"== {name} ({relative}) ==\n{blocks[name]}\n")
            return 0
        # Every document is spliced before any is written, so one refusal leaves all of them as they were.
        updates = {}
        for relative, names in DOCUMENTS.items():
            path = args.root / relative
            text = updated = path.read_text(encoding="utf-8")
            for name in names:
                updated = splice(updated, name, blocks[name])
            if updated != text:
                updates[relative] = updated
        if args.write:
            for relative, updated in updates.items():
                _write_atomically(args.root / relative, updated)
    except (PanelLayoutError, MarkerError, OSError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if updates:
        print(("regenerated: " if args.write else "stale: ") + ", ".join(updates), file=sys.stderr)
    return 1 if (updates and args.check) else 0


if __name__ == "__main__":
    sys.exit(main())
