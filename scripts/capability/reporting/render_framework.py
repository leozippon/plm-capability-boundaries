#!/usr/bin/env python3
"""Embed the immutable author-supplied artwork above a verified vector summary.

No structure selection, prediction, scoring, fitting or resampling is performed.
Only the generated summary is editable and certified at 7/8 pt. Original raster
text and colors are retained as supplied, not font-certified or recolored.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle
import pandas as pd
from PIL import Image

if __package__:
    from . import palette, publication_paths
else:
    import palette
    import publication_paths

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / publication_paths.ASSETS
OUT = ROOT / "scripts/capability/reporting/figure_data/main"
REFERENCE = ROOT / publication_paths.FRAMEWORK_SOURCE
WIDTH = 372 / 72.27
BODY, PANEL = 7, 8
LIMITATION = "Original supplied-raster text is not editable or font-certified; original colors are not recolored. Preview DPI is not native artwork DPI."


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_embedding(svg: Path, source: Path) -> None:
    """Require one native-resolution image with the original PNG bytes."""
    images = [node for node in ET.fromstring(svg.read_text()).iter()
              if node.tag.rsplit('}', 1)[-1] == 'image']
    if len(images) != 1:
        raise ValueError("Composite must embed exactly one supplied raster")
    href = images[0].get('{http://www.w3.org/1999/xlink}href', '')
    embedded = base64.b64decode(href.split(',', 1)[1], validate=True)
    if embedded != source.read_bytes():
        raise ValueError("Embedded raster differs from original PNG bytes")
    with Image.open(io.BytesIO(embedded)) as actual, Image.open(source) as original:
        if actual.size != original.size or actual.convert('RGBA').tobytes() != original.convert('RGBA').tobytes():
            raise ValueError("Embedded raster pixel identity failed")


def summary_rows() -> tuple[list[dict], list[Path]]:
    """Count checkpoint all-split positivity, never family-mean positivity.

    Each row binds the retained table/ledger and its upstream source hashes and
    pointers. Intervals are used only to count resolved comparisons, not drawn.
    """
    rows, sources = [], []

    def table(name):
        path = OUT / name
        sources.append(path)
        frame = pd.read_csv(path)
        for relative, expected in frame[['source_path', 'source_sha256']].drop_duplicates().itertuples(index=False, name=None):
            artifact = ROOT / relative
            if sha(artifact) != expected:
                raise ValueError(f"Summary source digest mismatch: {relative}")
            sources.append(artifact)
        return frame

    def record(panel, text, frame, name, rule, value):
        rows.append(dict(panel=panel, displayed_value=text, count=value,
                         source_path=str((OUT / name).relative_to(ROOT)), source_sha256=sha(OUT / name),
                         source_pointer=rule,
                         upstream_sources=json.dumps(frame[['source_path', 'source_sha256']].drop_duplicates().to_dict('records'), sort_keys=True),
                         upstream_pointers=json.dumps(frame.get('source_pointer', pd.Series(dtype=str)).drop_duplicates().tolist())))

    def all_splits(frame):
        if frame.groupby('arm').split_seed.nunique().ne(3).any():
            raise ValueError("All-split count requires exactly three splits per checkpoint")
        return int(frame.groupby('arm').ci_low.min().gt(0).sum())

    name = 'local-increments-source.csv'
    frame = table(name).query("endpoint == 'increment_M_C_P_wall'")
    n = all_splits(frame)
    if (n, frame.arm.nunique()) != (14, 33):
        raise ValueError("Mutation summary count changed")
    record('a', 'Mutation ranking\n14/33 positive at all 3 splits\nBeyond local/profile controls', frame, name,
           "endpoint=increment_M_C_P_wall; count arms with ci_low>0 at all three split_seed values", n)

    ledger = ROOT / 'results/shared/manuscript_evidence_20261004/derived-numbers.csv'
    sources.append(ledger)
    data = pd.read_csv(ledger)

    def ledger_rows(panel, text, identifiers, expected):
        selected = data.loc[data['id'].isin(identifiers)]
        values = selected.set_index('id').value.to_dict()
        if values != dict(zip(identifiers, expected)):
            raise ValueError(f"Summary ledger count changed: {panel}")
        for relative, digest in selected[['source_path', 'source_sha256']].drop_duplicates().itertuples(index=False, name=None):
            path = ROOT / relative
            if sha(path) != digest:
                raise ValueError(f"Summary ledger artifact changed: {relative}")
            sources.append(path)
        rows.append(dict(panel=panel, displayed_value=text, count=json.dumps(expected),
                         source_path=str(ledger.relative_to(ROOT)), source_sha256=sha(ledger),
                         source_pointer='exact ledger id/value lookup: ' + '; '.join(identifiers),
                         upstream_sources=json.dumps(selected[['source_path', 'source_sha256']].drop_duplicates().to_dict('records'), sort_keys=True),
                         upstream_pointers=json.dumps(selected.source_pointer.tolist())))

    ledger_rows('b', 'Likelihood terms\n6 site + 6 downstream gains\nNo resolved joint-over-full gain',
                ['position_terms/wall/resolved/unique_own', 'position_terms/wall/resolved/unique_downstream',
                 'position_terms/wall/resolved/joint_over_full'], [6, 6, 0])
    name = 'external-abundance-source.csv'
    frame = table(name).query("endpoint == 'primary_likelihood_spearman'")
    n = all_splits(frame)
    stability = data.set_index('id').loc['stability_replay/resolved_positive', 'value']
    if (n, frame.arm.nunique(), stability) != (14, 33, 2):
        raise ValueError("Phenotype summary count changed")
    record('c', 'Phenotype extension\nAbundance: 14/33 all-split gains\nStability: 2/33 simultaneous positives', frame, name,
           'primary_likelihood_spearman; arms with ci_low>0 at all three splits', n)
    ledger_rows('c', '', ['stability_replay/resolved_positive', 'stability_replay/panel/family_size'], [2, 33])
    name = 'clean-interaction-source.csv'
    frame = table(name)
    pos, neg = int(frame.ci_low.gt(0).sum()), int(frame.ci_high.lt(0).sum())
    if (pos, neg, len(frame)) != (0, 1, 33):
        raise ValueError("Interaction summary count changed")
    record('d', 'Adjusted interactions\n0/33 positive; 1 negative\nClean two-site comparisons', frame, name,
           'count ci_low>0 (positive), ci_high<0 (negative); 33 rows', f'{pos};{neg};33')
    name = 'context-ranking-source.csv'
    frame = table(name)
    n = int(frame.groupby('arm').ci_low.min().gt(0).sum())
    if (n, frame.arm.nunique()) != (2, 4):
        raise ValueError("Context summary count changed")
    record('e', 'Context / generalization\n2/4 improve over both references\nRemote generalization not established', frame, name,
           'count arms with ci_low>0 for both delta_spearman and homolog_minus_no_context_spearman', n)
    name = 'generation-streams-replot.csv'
    frame = table(name).query("endpoint == 'complete_domain'")
    pos, neg = int(frame.ci_low.gt(0).sum()), int(frame.ci_high.lt(0).sum())
    if (pos, neg, len(frame), set(frame.n_streams)) != (3, 13, 20, {3}):
        raise ValueError("Generation summary count changed")
    record('f', 'Complete-domain generation\n3 positive / 13 negative / 4 unresolved\n20 cells; 3 streams; exploratory t', frame, name,
           'model-minus-fragments; ci_low>0, ci_high<0, otherwise unresolved; exploratory t intervals, not FWER', '3;13;4;20;3')
    return rows, list(dict.fromkeys(sources))


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()  # --help performs no reads/writes
    # Deliberate rerenders bind the author's current canonical bytes; subsequent
    # source edits invalidate the receipt rather than enforcing an obsolete hash.
    source_sha = sha(REFERENCE)
    rows, sources = summary_rows()
    selection = ASSETS / 'selection.json'
    if json.loads(selection.read_text())['schema'] != 'framework-pinned-structures-v1':
        raise ValueError('Missing separately supplied structure provenance')
    with Image.open(REFERENCE) as image:
        native = image.copy()
    w = WIDTH * 72
    raster_h = w * native.height / native.width
    height = raster_h + 178
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': BODY, 'pdf.fonttype': 42,
                               'svg.fonttype': 'none', 'svg.hashsalt': 'supplied-composite-v1', 'text.color': palette.INK})
    fig = plt.figure(figsize=(WIDTH, height / 72), dpi=150)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, w), ylim=(height, 0))
    ax.set_axis_off()
    ax.imshow(native, extent=(0, w, raster_h, 0), interpolation='none')
    texts, boxes = [], []
    for i, row in enumerate(r for r in rows if r['displayed_value']):
        x, y = 3 + (i % 2) * (w / 2), raster_h + 5 + (i // 2) * 57
        bw, bh = w / 2 - 6, 53
        boxes.append((x, y, x + bw, y + bh))
        ax.add_patch(Rectangle((x, y), bw, bh, facecolor=palette.NEUTRAL, edgecolor=palette.BORDER, linewidth=.4))
        texts.append(ax.text(x + 6, y + 11, row['panel'], fontsize=PANEL, fontweight='bold', va='center'))
        lines = row['displayed_value'].splitlines()
        for j, line in enumerate(lines):
            texts.append(ax.text(x + (20 if j == 0 else 6), y + 11 + 12 * j, line,
                                 fontsize=BODY, va='center', fontweight='normal'))
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    extents, measured = [], []
    for text in texts:
        bb = text.get_window_extent(renderer)
        data = ax.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        px, py = text.get_position()
        box = next(b for b in boxes if b[0] <= px <= b[2] and b[1] <= py <= b[3])
        if data[0, 0] < box[0] or data[1, 0] > box[2] or data[1, 1] < box[1] or data[0, 1] > box[3]:
            raise ValueError(f'Summary text leaves its card: {text.get_text()}')
        if any(bb.overlaps(other) for other in measured):
            raise ValueError(f'Summary text collision: {text.get_text()}')
        measured.append(bb)
        extents.append(dict(text=text.get_text(), font_points=text.get_fontsize(),
                            bbox_points=[float(v) / fig.dpi * 72 for v in bb.extents]))
    pdf = ROOT / publication_paths.figure_pdf('Fig1_framework')
    pdf.parent.mkdir(parents=True, exist_ok=True)
    svg, png, plotted = [OUT / ('Fig1_framework' + suffix) for suffix in ['.svg', '.png', '-plotted.csv']]
    fig.savefig(pdf, metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(svg, metadata={'Date': None})
    # Preserve PNG bytes, including its native pixel resolution, in the SVG.
    content = svg.read_text()
    import re
    content = re.sub(r'(xlink:href=")data:image/png;base64,[^"]+("\s*)',
                     lambda m: m[1] + 'data:image/png;base64,' + base64.b64encode(REFERENCE.read_bytes()).decode() + m[2], content)
    svg.write_text('\n'.join(line.rstrip() for line in content.splitlines()) + '\n')
    verify_embedding(svg, REFERENCE)
    fig.savefig(png, dpi=450, facecolor='white')
    plt.close(fig)
    with plotted.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    def rel(path):
        return str(path.relative_to(ROOT))
    check = dict(figure='Fig1_framework', scope='supplied-raster-composite', inferential_intervals=False,
                 width_inches=WIDTH, height_inches=height / 72, minimum_font_points=BODY, panel_letter_points=PANEL,
                 all_text_inside_canvas=True, all_text_inside_panel=True, text_text_overlaps=0,
                 text_extents=extents, measured_text_count=len(texts), svg_text_editable='generated summary only',
                 text_bounds_method='Agg get_window_extent; strict card containment and pairwise collision checks',
                 preview_dpi=450, pdf_fonttype=42,
                 not_certified_scope=LIMITATION,
                 raster=dict(source_path=rel(REFERENCE), native_dimensions=list(native.size),
                             placement_points=[0, 0, w, raster_h], effective_dpi=native.width / WIDTH,
                             original_png_bytes_embedded=True, pixel_identity=True, interpolation='none',
                             native_below_450dpi=native.width / WIDTH < 450),
                 separate_structure_selection_path=rel(selection))
    for kind, path in [('pdf', pdf), ('svg', svg), ('png', png), ('plotted_source', plotted)]:
        check[kind + '_path'], check[kind + '_sha256'] = rel(path), sha(path)
    script = Path(__file__).resolve()
    receipt = dict(schema='supplied-raster-composite-validation-v1', scope='generated summary only; no inferential intervals',
                   source_sha256={rel(p): sha(p) for p in [REFERENCE, palette.PATH, publication_paths.PATH, selection, *sources]},
                   renderer_path=rel(script), renderer_sha256=sha(script), checks=[check])
    if sha(REFERENCE) != source_sha:
        raise RuntimeError('Canonical artwork changed during rendering')
    (OUT / 'framework_validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(pdf=rel(pdf), native_dimensions=list(native.size), effective_dpi=native.width / WIDTH,
                          checked_generated_texts=len(texts), not_certified_scope=LIMITATION)))


if __name__ == '__main__':
    main()
