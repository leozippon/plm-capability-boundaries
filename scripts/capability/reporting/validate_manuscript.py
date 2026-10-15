"""Validate the local Nature Computational Science Analysis and record evidence scope.

Run after compiling both PDFs. --source-only leaves PDF verification pending.
This checks format/provenance, not scientific completeness or author approval.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

if __package__:
    from . import publication_paths
else:
    import publication_paths

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / publication_paths.SUBMISSION
NCS = HERE / "figure_data" / "ncs"
MAIN_ART = HERE / "figure_data" / "main"


# Labels must not collide with one another and must not leave the canvas.  Ink
# boxes come from the metrics matplotlib laid the text out with, over the three
# forms it writes with svg.fonttype="none": positioned text with a rotation,
# translated text, and mathtext as tspan runs.  Overlaps below TOUCH_TOLERANCE
# are within the rounding of those metrics; the densest two panels, the 34-arm
# matrix and the nine frozen strata, sit there and are recorded in the log.
TOUCH_TOLERANCE = 0.25
FONT_TOLERANCE = 0.1
EXPECTED_SECTIONS = ["Results", "Discussion", "Limitations", "Methods"]

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


def ink_box(text, size, oblique, bold=False, family='DejaVu Sans'):
    from matplotlib.font_manager import FontProperties, findfont
    from matplotlib.textpath import TextPath
    properties = FontProperties(family=family, style='oblique' if oblique else 'normal',
                                weight='bold' if bold else 'normal')
    findfont(properties, fallback_to_default=False)
    extents = TextPath((0, 0), text, size=size, prop=properties).get_extents()
    return extents.x0, extents.x1, -extents.y1, -extents.y0


def label_boxes(content):
    """Every drawn label of one SVG as (x0, y0, x1, y1, text) in points."""
    def plain(raw):
        return re.sub(r"<[^>]+>", "", raw).replace("&#8722;", "\u2212").replace("&amp;", "&").strip()

    def size_of(style):
        found = re.search(r"font-size: ([0-9.]+)px", style)
        if not found:
            raise ValueError(f'Unmeasured SVG font style: {style}')
        return float(found.group(1))

    def metrics(text, style):
        family = re.search(r"font-family:\s*'([^']+)'", style)
        return ink_box(text, size_of(style), 'oblique' in style,
                       bool(re.search(r'font-weight:\s*(?:700|bold)', style)),
                       family.group(1) if family else 'DejaVu Sans')

    boxes = []
    for match in POSITIONED.finditer(content):
        text = plain(match.group(5))
        if not text:
            continue
        x, y = float(match.group(2)), float(match.group(3))
        found = re.search(r"text-anchor: (\w+)", match.group(1))
        anchor = found.group(1) if found else "start"
        left, right, top, bottom = metrics(text, match.group(1))
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
        x, y = float(match.group(2)), float(match.group(3))
        left, right, top, bottom = metrics(text, match.group(1))
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
            left, right, top, bottom = metrics(text, style)
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


def without_comments(text):
    """Drop LaTeX comments. A comment is not body text and is not typeset."""
    return re.sub(r'(?m)(?<!\\)%.*$', '', text)


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
    if not all(math.isfinite(value) and value > 0 for value in [source_width, source_height]):
        raise ValueError('Invalid artwork dimensions')
    dimensions = {}
    keep = False
    for option in (part.strip() for part in options.split(',') if part.strip()):
        if option == 'keepaspectratio' and not keep:
            keep = True
            continue
        match = re.fullmatch(r'(width|height)\s*=\s*([0-9.]+)?\s*\\(textwidth|textheight)', option)
        if not match or match[1] in dimensions or match[3] != ('textwidth' if match[1] == 'width' else 'textheight'):
            raise ValueError(f'unmodelled \\includegraphics option: {option}')
        fraction = float(match[2] or 1)
        if not math.isfinite(fraction) or fraction <= 0:
            raise ValueError(f'Invalid insertion dimension: {option}')
        dimensions[match[1]] = fraction * (TEXT_WIDTH_PT / source_width if match[1] == 'width' else TEXT_HEIGHT_PT / source_height)
    if len(dimensions) == 2 and not keep:
        raise ValueError('Width and height require keepaspectratio; distorted type is not certified')
    return min(dimensions.values()) if dimensions else 1.0


def figures(text):
    return re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)


def repository_path(relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Not a repository-relative path: {relative}')
    resolved = (REPO / path).resolve()
    if not resolved.is_relative_to(REPO.resolve()):
        raise ValueError(f'Path escapes repository: {relative}')
    return resolved


def framework_panel_labels(check):
    """Bind the assembler's explicit 8pt text telemetry to unique live SVG IDs.

    Only the six declared a–f headings qualify. One-letter 7pt axis/token text
    is never classified as a panel. SVG size and bold weight are rechecked below.
    """
    letters = [row['text'] for row in check['text_extents'] if row['font_points'] == 8]
    if letters != list('abcdef'):
        raise ValueError('Framework must declare six 8pt a–f panel headings')
    tree = ET.fromstring(repository_path(check['svg_path']).read_text())
    parents = {child: parent for parent in tree.iter() for child in parent}
    labels = []
    for letter in letters:
        matches = [node for node in tree.iter()
                   if node.tag.rsplit('}', 1)[-1] == 'text' and (node.text or '').strip() == letter
                   and re.search(r'font-size:\s*8(?:\.0+)?px(?:;|$)', node.get('style', ''))
                   and re.search(r'font-weight:\s*(?:700|bold)(?:;|$)', node.get('style', ''))]
        if len(matches) != 1:
            raise ValueError(f'Missing or ambiguous framework panel heading: {letter}')
        parent = parents[matches[0]]
        if not parent.get('id'):
            raise ValueError(f'Framework panel heading has no SVG group ID: {letter}')
        labels.append({'svg_id': parent.get('id'), 'text': letter})
    return labels


def supplied_raster_checks(check, sources):
    """Certify the untouched raster embedding, not its burned-in typography."""
    from PIL import Image

    if __package__:
        from . import render_framework as assembler
    else:
        import render_framework as assembler
    raster = check['raster']
    original = repository_path(raster['source_path'])
    expected_original = str(assembler.REFERENCE.relative_to(assembler.ROOT))
    if sources.get(expected_original) != digest(original):
        raise ValueError('Supplied PNG differs from rendered source digest')
    if raster['source_path'] != expected_original:
        raise ValueError('Raster is not the supplied original artwork')
    assembler.verify_embedding(repository_path(check['svg_path']), original)
    with Image.open(original) as image:
        native = list(image.size)
    width = check['width_inches'] * 72
    height = width * native[1] / native[0]
    expected_placement = [0, 0, width, height]
    if raster['native_dimensions'] != native or any(abs(a - b) > .001 for a, b in zip(raster['placement_points'], expected_placement)) or len(raster['placement_points']) != 4:
        raise ValueError('Raster native dimensions or placement changed')
    dpi = native[0] / check['width_inches']
    if abs(raster['effective_dpi'] - dpi) > .001 or type(raster['native_below_450dpi']) is not bool or raster['native_below_450dpi'] != (dpi < 450):
        raise ValueError('False native raster DPI declaration')
    if any(raster.get(flag) is not True for flag in ['original_png_bytes_embedded', 'pixel_identity']) or raster.get('interpolation') != 'none':
        raise ValueError('Undeclared raster pixel identity/interpolation')
    if check.get('not_certified_scope') != assembler.LIMITATION:
        raise ValueError('Original raster text must be explicitly outside font certification')
    tree = ET.fromstring(repository_path(check['svg_path']).read_text())
    live = [n.text for n in tree.iter() if n.tag.rsplit('}', 1)[-1] == 'text']
    extents = check['text_extents']
    if live != [row['text'] for row in extents] or check.get('measured_text_count') != len(live) or any(row['font_points'] not in (7, 8) for row in extents):
        raise ValueError('Unverified generated summary text telemetry')
    parents = {child: parent for parent in tree.iter() for child in parent}
    image = next(n for n in tree.iter() if n.tag.rsplit('}', 1)[-1] == 'image')
    matrix = re.fullmatch(r'matrix\(([^)]+)\)', image.get('transform', ''))
    if not matrix:
        raise ValueError('Unmeasured raster transform')
    a, b, c, d, x, y = map(float, matrix[1].split())
    if b != 0 or c != 0 or a <= 0 or d <= 0 or [float(image.get(k, '0')) for k in ['width', 'height']] != native:
        raise ValueError('Raster resampled, rotated or distorted')
    actual = [x, y, a * native[0], d * native[1]]
    if any(abs(v - target) > .02 for v, target in zip(actual, expected_placement)):
        raise ValueError('SVG raster placement differs from receipt')
    node = parents.get(image)
    while node is not None:
        if node.get('transform'):
            raise ValueError('Unmeasured ancestor raster transform')
        node = parents.get(node)
    listed = subprocess.run(['pdfimages', '-list', str(repository_path(check['pdf_path']))],
                            check=True, capture_output=True, text=True).stdout
    pdf_images = [line.split() for line in listed.splitlines() if len(line.split()) > 4 and line.split()[2] == 'image']
    if len(pdf_images) != 1 or [int(pdf_images[0][i]) for i in (3, 4)] != native:
        raise ValueError('PDF must retain the single native-resolution artwork raster')
    if check.get('svg_text_editable') != 'generated summary only':
        raise ValueError('Raster text cannot be claimed editable')
    selection = publication_paths.STRUCTURE_SELECTION.as_posix()
    if check.get('separate_structure_selection_path') != selection or selection not in sources:
        raise ValueError('Missing separate structure provenance binding')
    check['interval_check_status'] = 'not_applicable_summary_counts'


def load_render_manifest(path, failures):
    """Normalize quantitative, live-framework and supplied-composite receipts.

    Quantitative v1 uses renderer hash maps and explicit panel IDs. Framework
    and supplied-raster-composite v1 bind renderer hashes and 8pt telemetry;
    the latter certifies only generated text, with native raster identity and
    placement checked separately. Neither certifies summary-count intervals.
    """
    if not path.is_file():
        failures.append(f'Missing render receipt: {path.relative_to(REPO)}')
        return []
    try:
        report = json.loads(path.read_text())
        composite = report['schema'] == 'supplied-raster-composite-validation-v1'
        framework = report['schema'] == 'framework-assembly-validation-v1' or composite
        if framework:
            report['renderer_sha256'] = {report['renderer_path']: report['renderer_sha256']}
        elif report['schema'] != 'figure_render_validation_v1':
            raise ValueError('Unknown render receipt schema')
        for field in ['source_sha256', 'renderer_sha256', 'extraction_source_sha256', 'auxiliary_outputs_sha256']:
            if field in ['extraction_source_sha256', 'auxiliary_outputs_sha256'] and field not in report:
                continue
            if not isinstance(report[field], dict) or not report[field]:
                raise ValueError(f'{field} must be a nonempty path-to-digest map')
            for relative, expected in report[field].items():
                source = repository_path(relative)
                if not source.is_file() or digest(source) != expected:
                    failures.append(f'Missing or changed {field} input: {relative}')
        checks = report['checks']
        if not isinstance(checks, list) or not checks or any(not isinstance(check, dict) for check in checks):
            raise ValueError('Figure checks must be a nonempty list of objects')
        framework_pdf = publication_paths.figure_pdf('Fig1_framework').as_posix()
        if any(check.get('pdf_path') == framework_pdf for check in checks):
            for source in [publication_paths.FRAMEWORK_SOURCE.as_posix(), publication_paths.STRUCTURE_SELECTION.as_posix()]:
                if source not in report['source_sha256']:
                    failures.append(f'Missing declared framework source: {source}')
        if framework and (len(checks) != 1 or checks[0]['pdf_path'] != framework_pdf):
            raise ValueError('Framework receipt must cover only figure-1-framework.pdf')
        for check in checks:
            for kind in (['pdf', 'svg', 'png', 'plotted_source'] if framework else ['pdf', 'svg', 'plotted_source']):
                output = repository_path(check[f'{kind}_path'])
                if not output.is_file() or digest(output) != check[f'{kind}_sha256']:
                    failures.append(f'Missing or changed rendered {kind}: {check["figure"]}')
            if framework:
                if check.get('scope') != ('supplied-raster-composite' if composite else 'framework') or type(check.get('inferential_intervals')) is not bool or check['inferential_intervals']:
                    raise ValueError('Framework receipt must explicitly declare no inferential intervals')
                check['interval_check_status'] = 'not_applicable_conceptual_framework'
                if composite:
                    supplied_raster_checks(check, report['source_sha256'])
                check['panel_labels'] = framework_panel_labels(check)
                flags = ['all_text_inside_canvas', 'all_text_inside_panel']
                if type(check.get('text_text_overlaps')) is not int or check['text_text_overlaps'] != 0:
                    failures.append(f'Unverified framework text collisions: {check["figure"]}')
            else:
                flags = ['all_retained_intervals_inside_axes', 'all_text_inside_canvas']
            for flag in flags:
                if type(check.get(flag)) is not bool or not check[flag]:
                    failures.append(f'Unverified {flag}: {check["figure"]}')
            if check['minimum_font_points'] != 7 or check['panel_letter_points'] != 8:
                failures.append(f'Undeclared 7/8pt typography: {check["figure"]}')
            if not check['panel_labels']:
                failures.append(f'No explicit panel typography: {check["figure"]}')
        return checks
    except (OSError, KeyError, TypeError, ValueError, ET.ParseError, subprocess.SubprocessError) as error:
        failures.append(f'Invalid render receipt {path.relative_to(REPO)}: {error}')
        return []


def svg_typography(content, check, scale):
    """Measure live text; panel roles are explicit SVG IDs, never letter heuristics.

    Supports the positioned/translated text and tspan styles emitted by Matplotlib.
    Unmeasured fonts or scaling transforms fail; visual review remains necessary.
    """
    tree = ET.fromstring(content)
    panel_ids = {panel['svg_id']: panel['text'] for panel in check['panel_labels']}
    if len(panel_ids) != len(check['panel_labels']):
        raise ValueError('Duplicate panel SVG IDs')
    found_panels = {}
    measured = []

    def visit(node, inherited_style='', panel=None, unmeasured_transform=''):
        style = inherited_style + ';' + node.get('style', '')
        transform = node.get('transform', '')
        if re.search(r'\b(?:scale|matrix|skewX|skewY)\s*\(', transform):
            unmeasured_transform = transform
        panel = node.get('id') if node.get('id') in panel_ids else panel
        if node.tag.rsplit('}', 1)[-1] in {'text', 'tspan'} and (node.text or '').strip():
            if unmeasured_transform:
                raise ValueError(f'Unmeasured SVG text transform: {unmeasured_transform}')
            text = node.text.strip()
            sizes = re.findall(r'font-size:\s*([0-9.]+)px', style)
            if not sizes:
                raise ValueError(f'No explicit live SVG font size: {text}')
            size = float(sizes[-1])
            target = 8 if panel else 7
            if abs(size * scale - target) > FONT_TOLERANCE:
                raise ValueError(f'Artwork outside {target}pt target at insertion: {text}: {size * scale:.2f}pt')
            if panel:
                if text != panel_ids[panel] or not re.fullmatch('[a-z]', text) or not re.search(r'font-weight:\s*(700|bold)', style):
                    raise ValueError(f'Invalid bold lowercase panel letter: {panel}')
                found_panels[panel] = found_panels.get(panel, 0) + 1
            measured.append(size)
        for child in node:
            visit(child, style, panel, unmeasured_transform)

    visit(tree)
    if not measured or found_panels != dict.fromkeys(panel_ids, 1):
        raise ValueError('Missing live text or explicit panel letters')
    if re.search(r'id="(?:DejaVuSans|Arial|Helvetica)[^-" ]*-', content):
        raise ValueError('Outlined font glyphs are not editable SVG text')
    return measured


def pdf_footer_faults(content):
    """Check compiled text against the numbered sn-nature page footers."""
    pages = [node for node in ET.fromstring(content).iter()
             if node.tag.rsplit('}', 1)[-1] == 'page']
    if not pages:
        raise ValueError('No pages in PDF text geometry')
    faults = []
    for number, page in enumerate(pages, 1):
        words_on_page = [node for node in page.iter()
                         if node.tag.rsplit('}', 1)[-1] == 'word']
        width, height = float(page.attrib['width']), float(page.attrib['height'])
        footers = [word for word in words_on_page
                   if word.text == str(number) and float(word.attrib['yMin']) > .7 * height
                   and abs((float(word.attrib['xMin']) + float(word.attrib['xMax'])) / 2 - width / 2) < 40]
        if len(footers) != 1:
            faults.append(f'Cannot identify unique numbered footer on page {number}')
            continue
        footer = footers[0]
        if any(float(word.attrib['yMax']) >= float(footer.attrib['yMin'])
               for word in words_on_page if word is not footer):
            faults.append(f'Body text reaches page-number area on page {number}')
    return faults


def tex_inputs(text, root, active=()):
    """Read literal manuscript-local inputs, preserving repeated inclusions."""
    dependencies = []
    uncommented = without_comments(text)
    for name in arguments(uncommented, 'input'):
        path = root / name
        if not path.suffix:
            path = path.with_suffix('.tex')
        path = path.resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f'Nonlocal TeX input: {name}')
        if path in active:
            raise ValueError(f'Cyclic TeX input: {name}')
        content = path.read_text()
        dependencies.append((path, content))
        dependencies.extend(tex_inputs(content, root, (*active, path)))
    return dependencies


def float_placement_faults(text):
    """Allow ordinary text-page floats and float pages, including class defaults."""
    uncommented = without_comments(text)
    faults = []
    for match in re.finditer(r'\\begin\{(figure\*?|table\*?)\}\s*(?:\[([^]]*)\])?', uncommented):
        placement = match[2] if match[2] is not None else 'tbp'
        if 'p' not in placement or not set(placement) & set('htb'):
            faults.append(f'{match[1]} placement [{placement}] must allow text-page placement (h/t/b) and float pages (p)')
    return faults


def section_faults(sections, subsections):
    faults = []
    if sections != EXPECTED_SECTIONS:
        faults.append('Expected unheaded introduction, Results, Discussion, Limitations, Methods')
    if subsections.get('Results') != 7:
        faults.append('Results must contain the seven deliberate result titles')
    if not subsections.get('Methods', 0):
        faults.append('Methods needs topical subheadings')
    for section in ['Discussion', 'Limitations']:
        if subsections.get(section, 0):
            faults.append(f'{section} contains subheadings')
    return faults


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
    included = {}
    for name, text in [("main", main), ("supplement", supp)]:
        try:
            included[name] = tex_inputs(text, ROOT)
        except (OSError, ValueError) as exc:
            require(False, f'Invalid TeX inputs in {name}: {exc}')
            included[name] = []
        text += '\n' + '\n'.join(content for _, content in included[name])
        failures.extend(f'{name}: {fault}' for fault in float_placement_faults(text))
        keys = {key.strip() for group in arguments(text, "cite") for key in group.split(",")}
        allkeys |= keys
        labels = arguments(text, "label")
        require(len(labels) == len(set(labels)), f"Duplicate labels in {name}")
        require(set(arguments(text, "ref")) <= set(labels), f"Undefined local references in {name}")
        require(keys <= set(bibkeys), f"Missing bibliography entries in {name}: {sorted(keys-set(bibkeys))}")
    require(allkeys == set(bibkeys), "Uncited bibliography entries remain")
    sections = arguments(main, "section")
    body = main.split(r"\maketitle", 1)[1].split(r"\section{Methods}", 1)[0]
    body = without_comments(body)
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
    require(len(main_figures) == 5 and len(supp_figures) == 4, 'Expected five main and four supplementary figures')
    subsections = {}
    for index, section in enumerate(sections):
        chunk = main.split(r"\section{" + section + "}", 1)[1]
        chunk = chunk.split(r"\section", 1)[0]
        subsections[section] = len(arguments(chunk, "subsection"))
    failures.extend(section_faults(sections, subsections))
    require(not re.search(r"\\cite\{", abstracts[0]), "Abstract contains references")
    # A caption-less main text used to satisfy the length limit on a sentinel of
    # zero, so the one state in which no caption can be measured was the one
    # state that passed. Having nothing to read is its own failure.
    caption_words = [words(caption) for caption in arguments(main, "caption")]
    require(caption_words, "Main text declares no captions to check the length of")
    max_caption_words = max(caption_words, default=0)
    require(max_caption_words <= 350, "Main caption exceeds 350 words")

    provenance_path = NCS / 'source-provenance.json'
    source_table_hashes = {}
    try:
        provenance = json.loads(provenance_path.read_text())
        for path, expected in provenance['source_sha256'].items():
            source = repository_path(path)
            require(source.is_file(), f'Missing provenance source: {path}')
            if source.is_file():
                require(digest(source) == expected, f'Changed provenance source: {path}')
        require(digest(HERE / 'build_ncs_figures.py') == provenance['script_sha256'], 'Figure extraction script differs from provenance')
        for name in provenance['source_tables']:
            path = NCS / name
            require(path.is_file(), f'Missing source table: {name}')
            if not path.is_file():
                continue
            source_table_hashes[name] = digest(path)
            with path.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            require(bool(rows), f'Empty source table: {name}')
            for row in rows:
                if row.get('source_path'):
                    require(row.get('source_sha256') == provenance['source_sha256'].get(row['source_path']), f'CSV/provenance source mismatch: {name}')
    except (OSError, ValueError, KeyError, TypeError) as error:
        require(False, f'Invalid or missing retained extraction provenance: {error}')

    render_checks = load_render_manifest(MAIN_ART / 'figure_validation.json', failures)
    if (Path('figures') / publication_paths.PDF_NAMES['Fig1_framework']).as_posix() in main_figures:
        render_checks += load_render_manifest(MAIN_ART / 'framework_validation.json', failures)
    receipts = {}
    for check in render_checks:
        path = check['pdf_path']
        require(path not in receipts, f'Duplicate rendered figure receipt: {path}')
        receipts[path] = check
    expected_figures = {str((ROOT / name).relative_to(REPO)) for name in main_figures + supp_figures}
    require(set(receipts) <= expected_figures, 'Render receipt contains figures not included in the manuscript')

    font_rows = []
    for name in main_figures + supp_figures:
        relative = str((ROOT / name).relative_to(REPO))
        check = receipts.get(relative)
        require(check is not None, f'No current render receipt for figure: {name}')
        if check is None:
            continue
        pdf = repository_path(check['pdf_path'])
        svg = repository_path(check['svg_path'])
        require(pdf.is_file() and svg.is_file(), f'Missing figure PDF/SVG: {name}')
        if not pdf.is_file() or not svg.is_file():
            continue
        try:
            content = svg.read_text()
            tree = ET.fromstring(content)
            if not all(tree.attrib[key].endswith('pt') for key in ['width', 'height']):
                raise ValueError('SVG dimensions must be in points')
            source_width = float(tree.attrib['width'].removesuffix('pt'))
            source_height = float(tree.attrib['height'].removesuffix('pt'))
            require(abs(source_width - 72 * check['width_inches']) <= .02 and
                    abs(source_height - 72 * check['height_inches']) <= .02, f'Receipt/SVG dimension mismatch: {name}')
            scale = placed_scale(insertion.get(name, ''), source_width, source_height)
            require(source_width * scale <= TEXT_WIDTH_PT + .1 and source_height * scale <= TEXT_HEIGHT_PT + .1,
                    f'Artwork exceeds manuscript text area: {name}')
            sizes = svg_typography(content, check, scale)
            labels, faults = layout_faults(content, source_width, source_height)
            require(labels == len(sizes), f'Unmeasured live SVG label geometry: {name}')
            for fault in faults:
                require(False, f'Label layout fault: {name}: {fault}')
            font_rows.append({'figure': name, 'source_width_pdf_points': source_width,
                              'insertion_width_pdf_points': source_width * scale,
                              'minimum_source_font_points': min(sizes),
                              'minimum_effective_font_points': min(sizes) * scale,
                              'labels_measured': labels, 'label_layout_faults': faults,
                              'not_certified_scope': check.get('not_certified_scope'),
                              'native_raster_effective_dpi': check.get('raster', {}).get('effective_dpi'),
                              'svg_sha256': digest(svg), 'pdf_sha256': digest(pdf)})
        except (KeyError, TypeError, ValueError, ET.ParseError) as error:
            require(False, f'Artwork validation failed: {name}: {error}')
    uncertified = [check['not_certified_scope'] for check in render_checks if check.get('not_certified_scope')]
    typography = {'status': ('qualified_pass' if uncertified else 'pass') if len(font_rows) == len(main_figures)+len(supp_figures) and not failures else 'fail',
                  'not_certified_scope': uncertified,
                  'method': 'Explicit panel SVG IDs distinguish bold lowercase 8pt letters from 7pt body/axis text; framework-assembly v1 IDs are bound to its six declared 8pt a–f text-bound records and uniquely matched bold live SVG headings. Every live text run is checked at actual insertion scale, with 0.1pt lower/upper tolerance; unmodelled insertion options or text transforms fail. Positioned/translated Matplotlib text and tspan ink boxes use the declared single-quoted font family, weight and oblique style (DejaVu Sans if absent); unavailable fonts fail. Boxes must not overlap by more than 0.25pt or leave the canvas. Other SVG layouts are not certified. Visual inspection remains necessary.',
                  'figures': font_rows}
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
        inputs.extend(path for path, _ in included['main' if name == 'main' else 'supplement'])
        require(all(pdf.stat().st_mtime >= x.stat().st_mtime for x in inputs if x.is_file()), f"PDF older than its inputs: {name}")
        info = subprocess.run(["pdfinfo", str(pdf)], check=True, text=True, capture_output=True).stdout
        pages = re.search(r'^Pages:\s+(\d+)$', info, re.M)
        require(pages is not None, f'Cannot determine PDF page count: {name}')
        if pages:
            pdf_records[name] = {'pages': int(pages[1]), 'sha256': digest(pdf)}
        try:
            geometry = subprocess.run(['pdftotext', '-bbox', str(pdf), '-'],
                                      check=True, text=True, capture_output=True).stdout
            footer_faults = pdf_footer_faults(geometry)
            pdf_records.setdefault(name, {})['footer_faults'] = footer_faults
            for fault in footer_faults:
                require(False, f'{name}: {fault}')
        except (ET.ParseError, KeyError, TypeError, ValueError) as exc:
            require(False, f'Cannot verify PDF footer clearance in {name}: {exc}')
    result = {"date": datetime.now(timezone.utc).isoformat(), "journal": "Nature Computational Science", "article_type": "Analysis",
              "status": "fail" if failures else ("source_checks_pass_pdf_pending" if source_only else ("format_and_provenance_checks_qualified_pass" if uncertified else "format_and_provenance_checks_pass")),
              "not_certified_scope": uncertified,
              "warnings": [f"Supplied raster: native {check['raster']['native_dimensions']} at {check['raster']['effective_dpi']:.2f} DPI; burned-in text not certified; 450-DPI preview does not raise native resolution." for check in render_checks if check.get('raster')],
              "main_word_count": body_words, "abstract_word_count": abstract_words,
              "count_convention": "Whitespace-separated prose after removing display environments, section headings and citation/reference commands; excludes abstract, Methods, references and legends. Hyphenated tokens count once; tildes and inline-math delimiters are normalized.",
              "main_figures": len(main_figures), "main_tables": main_tables, "main_display_count": len(main_figures)+main_tables,
              "supplementary_figures": len(supp_figures), "subsections_by_section": subsections, "references": len(bibkeys),
              "maximum_main_legend_words": max_caption_words, "pdfs": pdf_records, "failures": failures,
              "files_sha256": {name: digest(ROOT/name) for name in ["main.tex", "supplementary-information.tex", "references.bib", "sn-jnl.cls", "sn-nature.bst"]} | {"validate_manuscript.py": digest(HERE / "validate_manuscript.py")},
              "source_tables_sha256": source_table_hashes,
              "source_provenance_sha256": digest(provenance_path) if provenance_path.is_file() else None,
              "render_receipts_sha256": {str(path.relative_to(REPO)): digest(path)
                                        for path in [MAIN_ART / 'figure_validation.json', MAIN_ART / 'framework_validation.json'] if path.is_file()},
              "scientific_completeness_certified": False,
              "evidence_boundary": "Structural predictor-confidence evidence is supplementary and calibration-limited; aggregate provenance is distinct from full raw-run replay.",
              "author_and_access_requirements_remain": True, "visual_pdf_review_required": True}
    (HERE / "figure_data" / "validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ["status", "main_word_count", "abstract_word_count", "main_display_count", "references", "failures", "warnings", "not_certified_scope"]}, indent=2))
    return not failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true", help="Do not certify existing PDFs; leave their validation pending")
    options = parser.parse_args()
    raise SystemExit(0 if validate(options.source_only) else 1)
