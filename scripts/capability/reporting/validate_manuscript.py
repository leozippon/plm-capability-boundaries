"""Validate the local Nature Computational Science Analysis and record evidence scope.

Run after compiling both PDFs. --source-only leaves PDF verification pending.
This checks format/provenance, not scientific completeness or author approval.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "manuscript"
NCS = HERE / "figure_data" / "ncs"
MAIN_ART = HERE / "figure_data" / "main"


# Labels must not collide with one another and must not leave the canvas.  Ink
# boxes come from the metrics matplotlib laid the text out with, over the three
# forms it writes with svg.fonttype="none": positioned text with a rotation,
# translated text, and mathtext as tspan runs.  Overlaps below TOUCH_TOLERANCE
# are within the rounding of those metrics; the densest two panels, the 34-arm
# matrix and the nine frozen strata, sit there and are recorded in the log.
TOUCH_TOLERANCE = 0.25

# The text block of sn-jnl with the sn-nature option, in PDF points. Artwork is
# scaled to these at insertion, so the type a reader sees is the type in the
# figure times that scale, not the type the figure was drawn with.
TEXT_WIDTH_PT = 31 * 12 * 72 / 72.27
TEXT_HEIGHT_PT = 552.69
POSITIONED = re.compile(r'<text style="([^"]*)"\s+x="([0-9.-]+)" y="([0-9.-]+)"\s+'
                        r'transform="rotate\((-?[0-9.]+) [0-9.-]+ [0-9.-]+\)">(.*?)</text>', re.S)
TRANSLATED = re.compile(r'<text style="([^"]*)"\s+transform="translate\(([0-9.-]+) ([0-9.-]+)\)'
                        r'( rotate\((-?[0-9.]+)\))?">([^<]*?)</text>', re.S)
MATHTEXT = re.compile(r'<g transform="translate\(([0-9.-]+) ([0-9.-]+)\)">\s*<text>(.*?)</text>', re.S)
RUN = re.compile(r'<tspan x="([0-9.-]+)" y="([0-9.-]+)" style="([^"]*)">(.*?)</tspan>', re.S)


def ink_box(text, size, oblique):
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    extents = TextPath((0, 0), text, size=size,
                       prop=FontProperties(family="DejaVu Sans",
                                           style="oblique" if oblique else "normal")).get_extents()
    return extents.x0, extents.x1, -extents.y1, -extents.y0


def label_boxes(content):
    """Every drawn label of one SVG as (x0, y0, x1, y1, text) in points."""
    def plain(raw):
        return re.sub(r"<[^>]+>", "", raw).replace("&#8722;", "\u2212").replace("&amp;", "&").strip()

    def size_of(style):
        found = re.search(r"font-size: ([0-9.]+)px", style)
        return float(found.group(1)) if found else 8.0

    boxes = []
    for match in POSITIONED.finditer(content):
        text = plain(match.group(5))
        if not text:
            continue
        size, x, y = size_of(match.group(1)), float(match.group(2)), float(match.group(3))
        found = re.search(r"text-anchor: (\w+)", match.group(1))
        anchor = found.group(1) if found else "start"
        left, right, top, bottom = ink_box(text, size, False)
        span = right - left
        if abs(float(match.group(4))) < 1:
            shift = {"start": 0.0, "middle": -span / 2 - left, "end": -span - left}[anchor]
            boxes.append((x + left + shift, y + top, x + right + shift, y + bottom, text))
        else:
            shift = {"start": 0.0, "middle": span / 2 + left, "end": span + left}[anchor]
            boxes.append((x + top, y - right + shift, x + bottom, y - left + shift, text))
    for match in TRANSLATED.finditer(content):
        text = plain(match.group(6))
        if not text:
            continue
        size, x, y = size_of(match.group(1)), float(match.group(2)), float(match.group(3))
        left, right, top, bottom = ink_box(text, size, False)
        if match.group(5) is None:
            boxes.append((x + left, y + top, x + right, y + bottom, text))
        else:
            boxes.append((x + top, y - right, x + bottom, y - left, text))
    for match in MATHTEXT.finditer(content):
        x, y = float(match.group(1)), float(match.group(2))
        for run in RUN.finditer(match.group(3)):
            text, style = plain(run.group(4)), run.group(3)
            if not text:
                continue
            left, right, top, bottom = ink_box(text, size_of(style), "oblique" in style)
            dx, dy = float(run.group(1)), float(run.group(2))
            boxes.append((x + dx + left, y + dy + top, x + dx + right, y + dy + bottom, text))
    return boxes


def layout_faults(content, width, height):
    """Colliding label pairs and labels outside the canvas, as descriptions."""
    boxes = label_boxes(content)
    faults = []
    for index, one in enumerate(boxes):
        for other in boxes[index + 1:]:
            across = min(one[2], other[2]) - max(one[0], other[0])
            down = min(one[3], other[3]) - max(one[1], other[1])
            if across > TOUCH_TOLERANCE and down > TOUCH_TOLERANCE:
                faults.append(f"{one[4]!r} overlaps {other[4]!r} by {across:.1f}x{down:.1f}pt")
        if one[0] < -0.3 or one[2] > width + 0.3 or one[1] < -0.3 or one[3] > height + 0.3:
            faults.append(f"{one[4]!r} outside the canvas: "
                          f"x[{one[0]:.1f},{one[2]:.1f}] y[{one[1]:.1f},{one[3]:.1f}] of {width:.0f}x{height:.0f}")
    return len(boxes), faults


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arguments(text, command):
    values = []
    for match in re.finditer(r"\\" + re.escape(command) + r"\{", text):
        position, depth = match.end(), 1
        begin = position
        while depth and position < len(text):
            char = text[position]
            if char in "{}" and (position == 0 or text[position - 1] != "\\"):
                depth += 1 if char == "{" else -1
            position += 1
        if depth:
            raise ValueError(f"Unclosed {command} argument")
        values.append(text[begin:position - 1])
    return values


def words(text):
    text = re.sub(r"\\(?:sub)?section\*?\{[^}]*\}|\\(?:cite|ref|label)\{[^}]*\}", "", text)
    text = text.replace("~", " ")
    text = re.sub(r"\\([%&])", r"\1", text)
    return len(text.replace("$", "").replace("{", "").replace("}", "").split())


def figure_options(text):
    """Map each included figure to the option list it is inserted with."""
    return {m.group(2): (m.group(1) or "")
            for m in re.finditer(r"\\includegraphics(?:\[([^]]*)\])?\{([^}]+)\}", text)}


def placed_scale(options, source_width, source_height):
    """The factor the class applies to one figure at insertion.

    A width option alone scales by width. A height cap scales by height as
    well, and with keepaspectratio the smaller factor is the one applied, so
    measuring width alone certifies type that the page then shrinks further.
    Any option this does not model is refused rather than silently ignored.
    """
    modelled = {"width", "height", "keepaspectratio"}
    for option in (part.strip() for part in options.split(",") if part.strip()):
        if option.split("=")[0].strip() not in modelled:
            raise ValueError(f"unmodelled \\includegraphics option: {option}")
    scale = TEXT_WIDTH_PT / source_width
    cap = re.search(r"height\s*=\s*([0-9.]+)\s*\\textheight", options)
    if cap:
        scale = min(scale, float(cap.group(1)) * TEXT_HEIGHT_PT / source_height)
    return scale


def figures(text):
    return re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)


def validate(source_only=False):
    failures = []

    def require(condition, message):
        if not condition:
            failures.append(message)

    main = (ROOT / "main.tex").read_text()
    supp = (ROOT / "supplementary-information.tex").read_text()
    bib = (ROOT / "references.bib").read_text()
    bibkeys = re.findall(r"^@\w+\{([^,]+),", bib, re.M)
    require(len(bibkeys) == len(set(bibkeys)), "Duplicate bibliography keys")
    allkeys = set()
    for name, text in [("main", main), ("supplement", supp)]:
        keys = {key.strip() for group in arguments(text, "cite") for key in group.split(",")}
        allkeys |= keys
        labels = arguments(text, "label")
        require(len(labels) == len(set(labels)), f"Duplicate labels in {name}")
        require(set(arguments(text, "ref")) <= set(labels), f"Undefined local references in {name}")
        require(keys <= set(bibkeys), f"Missing bibliography entries in {name}: {sorted(keys-set(bibkeys))}")
    require(allkeys == set(bibkeys), "Uncited bibliography entries remain")
    sections = arguments(main, "section")
    require(sections == ["Results", "Discussion", "Methods"],
            "Expected unheaded introduction, Results, Discussion, Methods")
    body = main.split(r"\maketitle", 1)[1].split(r"\section{Methods}", 1)[0]
    body = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", "", body, flags=re.S)
    body_words = words(body)
    abstracts = arguments(main, "abstract")
    require(len(abstracts) == 1, "Expected one abstract")
    abstract_words = words(abstracts[0]) if abstracts else 0
    require(100 <= abstract_words <= 150, f"Analysis abstract has {abstract_words} words; allowed 100–150")
    require(body_words <= 3500, f"Main text has {body_words} words; maximum 3500")
    main_figures = figures(main)
    supp_figures = figures(supp)
    insertion = figure_options(main) | figure_options(supp)
    main_tables = len(re.findall(r"\\begin\{table\}", main))
    require(len(main_figures) + main_tables <= 6, "More than six main displays")
    subsections = {}
    for index, section in enumerate(sections):
        chunk = main.split(r"\section{" + section + "}", 1)[1]
        chunk = chunk.split(r"\section", 1)[0]
        subsections[section] = len(arguments(chunk, "subsection"))
    require(subsections.get("Discussion") == 0, "Discussion contains subheadings")
    require(subsections.get("Results", 0) > 0 and subsections.get("Methods", 0) > 0, "Results and Methods need topical subheadings")
    require(not re.search(r"\\cite\{", abstracts[0]), "Abstract contains references")
    max_caption_words = max(words(c) for c in arguments(main, "caption"))
    require(max_caption_words <= 350, "Main caption exceeds 350 words")

    provenance_path = NCS / "source-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    for path, expected in provenance["source_sha256"].items():
        source = REPO / path
        require(source.is_file(), f"Missing provenance source: {path}")
        if source.is_file():
            require(digest(source) == expected, f"Changed provenance source: {path}")
    require(digest(HERE / "build_ncs_figures.py") == provenance["script_sha256"], "Figure extraction script differs from provenance")
    source_table_hashes = {}
    for name in provenance["source_tables"]:
        path = NCS / name
        require(path.is_file(), f"Missing source table: {name}")
        if not path.is_file():
            continue
        source_table_hashes[name] = digest(path)
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        require(bool(rows), f"Empty source table: {name}")
        for row in rows:
            if row.get("source_path"):
                require(row.get("source_sha256") == provenance["source_sha256"].get(row["source_path"]), f"CSV/provenance source mismatch: {name}")

    render_path = NCS / "render-provenance.json"
    render = json.loads(render_path.read_text())
    require(digest(HERE / "render_ncs_figures.py") == render["renderer_sha256"], "Renderer differs from render provenance")
    for name, expected in render["csv_sha256"].items():
        require(source_table_hashes.get(name) == expected, f"Plotted CSV differs from render provenance: {name}")

    font_rows = []
    for name in main_figures + supp_figures:
        pdf = ROOT / name
        svg_name = Path(name).with_suffix(".svg").name
        svg = (NCS if name.startswith("figures/ncs/") else MAIN_ART) / svg_name
        require(pdf.is_file() and svg.is_file(), f"Missing figure PDF/SVG: {name}")
        if not pdf.is_file() or not svg.is_file():
            continue
        content = svg.read_text()
        tree = ET.fromstring(content)
        require(tree.attrib["width"].endswith("pt"), f"SVG width not in points: {name}")
        source_width = float(tree.attrib["width"].removesuffix("pt"))
        sizes = [float(v) for v in re.findall(r"font-size:\s*([0-9.]+)px", content)]
        require(bool(sizes), f"No live SVG text sizes: {name}")
        if not sizes:
            continue
        source_height = float(tree.attrib["height"].removesuffix("pt"))
        scale = placed_scale(insertion.get(name, ""), source_width, source_height)
        width = TEXT_WIDTH_PT * min(1.0, scale * source_width / TEXT_WIDTH_PT)
        effective = min(sizes) * scale
        require(effective >= 7.9, f"Artwork below 8pt target at insertion: {name}: {effective:.2f}pt")
        labels, faults = layout_faults(content, source_width, source_height)
        for fault in faults:
            require(False, f"Label layout fault: {name}: {fault}")
        font_rows.append({"figure": name, "source_width_pdf_points": source_width,
                          "insertion_width_pdf_points": width, "minimum_source_font_points": min(sizes),
                          "minimum_effective_font_points": effective, "labels_measured": labels,
                          "label_layout_faults": faults,
                          "svg_sha256": digest(svg), "pdf_sha256": digest(pdf)})
    typography = {"status": "pass" if len(font_rows) == len(main_figures)+len(supp_figures) and all(r["minimum_effective_font_points"] >= 7.9 and not r["label_layout_faults"] for r in font_rows) else "fail",
                  "method": "Live SVG font size scaled by exact 31-pc PDF insertion width; 8pt target with 0.1pt numeric tolerance. Label ink boxes measured with the layout metrics and required not to overlap by more than 0.25pt or to leave the canvas. Visual inspection remains necessary.",
                  "figures": font_rows}
    (HERE / "figure_data").mkdir(parents=True, exist_ok=True)
    (HERE / "figure_data" / "figure-typography.json").write_text(json.dumps(typography, indent=2)+"\n")

    pdf_records = {}
    for name, source, images in [("main", "main.tex", main_figures), ("supplementary-information", "supplementary-information.tex", supp_figures)]:
        pdf = ROOT / f"{name}.pdf"
        if source_only:
            pdf_records[name] = {"status": "not_checked_source_only"}
            continue
        require(pdf.is_file(), f"Missing compiled PDF: {name}")
        if not pdf.is_file():
            continue
        inputs = [ROOT / source, ROOT / "references.bib", ROOT / "sn-jnl.cls", ROOT / "sn-nature.bst"] + [ROOT / x for x in images]
        require(all(pdf.stat().st_mtime >= x.stat().st_mtime for x in inputs if x.is_file()), f"PDF older than its inputs: {name}")
        info = subprocess.run(["pdfinfo", str(pdf)], check=True, text=True, capture_output=True).stdout
        pdf_records[name] = {"pages": int(re.search(r"^Pages:\s+(\d+)$", info, re.M)[1]), "sha256": digest(pdf)}
    result = {"date": datetime.now(timezone.utc).isoformat(), "journal": "Nature Computational Science", "article_type": "Analysis",
              "status": "fail" if failures else ("source_checks_pass_pdf_pending" if source_only else "format_and_provenance_checks_pass"),
              "main_word_count": body_words, "abstract_word_count": abstract_words,
              "count_convention": "Whitespace-separated prose after removing display environments, section headings and citation/reference commands; excludes abstract, Methods, references and legends. Hyphenated tokens count once; tildes and inline-math delimiters are normalized.",
              "main_figures": len(main_figures), "main_tables": main_tables, "main_display_count": len(main_figures)+main_tables,
              "supplementary_figures": len(supp_figures), "subsections_by_section": subsections, "references": len(bibkeys),
              "maximum_main_legend_words": max_caption_words, "pdfs": pdf_records, "failures": failures,
              "files_sha256": {name: digest(ROOT/name) for name in ["main.tex", "supplementary-information.tex", "references.bib", "sn-jnl.cls", "sn-nature.bst"]} | {"validate_manuscript.py": digest(HERE / "validate_manuscript.py")},
              "source_tables_sha256": source_table_hashes, "source_provenance_sha256": digest(provenance_path),
              "scientific_completeness_certified": False,
              "evidence_boundary": "Structural predictor-confidence evidence is supplementary and calibration-limited; aggregate provenance is distinct from full raw-run replay.",
              "author_and_access_requirements_remain": True, "visual_pdf_review_required": True}
    (HERE / "figure_data" / "validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ["status", "main_word_count", "abstract_word_count", "main_display_count", "references", "failures"]}, indent=2))
    return not failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true", help="Do not certify existing PDFs; leave their validation pending")
    options = parser.parse_args()
    raise SystemExit(0 if validate(options.source_only) else 1)
