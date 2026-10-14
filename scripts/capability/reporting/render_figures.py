#!/usr/bin/env python3
"""Draw the six-result manuscript from retained estimates, without new inference.

Fig. 1 is assembled separately from authenticated illustration assets. This renderer
never opens or overwrites the author's framework-original.png. It draws Figs. 2–5 and
four supplementary figures. Editable PDF/SVG text uses embedded TrueType fonts.
The 7 pt body / 8 pt panel policy follows Nature's official figure specifications;
this explicitly supersedes the previous project-only 8 pt minimum for all text.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

if __package__:
    from . import palette, checkpoint_labels, publication_paths
else:
    import palette
    import checkpoint_labels
    import publication_paths

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = HERE / 'figure_data' / 'main'
SUPPLEMENT_DATA = HERE / 'figure_data' / 'ncs'
OUT = REPO / publication_paths.PDF_DIRECTORY
WIDTH = 5.14
BASE = 7.0
LETTER = 8.0
BLUE, ORANGE = palette.BINARY_FIRST, palette.BINARY_SECOND
INK = palette.INK
CMAP = palette.SIGNED
GEN = ['protgpt2', 'progen2-small', 'progen2-base', 'progen2-medium', 'progen2-large',
       'progen2-xlarge', 'progen3-112m', 'progen3-3b', 'protgpt3-1.3b', 'rita-xl',
       'galactica-125m', 'galactica-1.3b', 'galactica-6.7b', 'galactica-30b',
       'llama-2-7b', 'prollama-stage-1', 'prollama', 'proteinglm-7b-clm', 'instructprotein']
TXT = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'dialogpt-small', 'bygpt5-small-en',
       'bygpt5-base-en', 'bygpt5-medium-en', 'qwen2.5-0.5b', 'qwen2.5-0.5b-instruct',
       'qwen2.5-7b', 'qwen2.5-32b', 'qwen3-8b-base', 'llama-3.2-3b']
ORDER = GEN + TXT
PRIMARY = 20260923
HASHES, CHECKS, CONTRACTS = {}, [], []
EXPECTED_HASHES = {}
EXTRACTION_HASHES = {}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': BASE, 'axes.titlesize': BASE,
                     'axes.labelsize': BASE, 'xtick.labelsize': BASE, 'ytick.labelsize': BASE,
                     'legend.fontsize': BASE, 'axes.linewidth': .55, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'svg.fonttype': 'none', 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'xtick.major.size': 2, 'ytick.major.size': 2, 'axes.grid': False})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name) -> pd.DataFrame:
    """Read public source data, refusing drift from the retained render receipt."""
    path = DATA / (name + '.csv')
    relative = str(path.relative_to(REPO))
    actual = digest(path)
    if relative not in EXPECTED_HASHES:
        raise ValueError(f'No retained source-table digest: {relative}')
    if actual != EXPECTED_HASHES[relative]:
        raise ValueError(f'Source-table digest mismatch: {relative}')
    HASHES[relative] = actual
    table = pd.read_csv(path)
    for prefix in ['', 'baseline_']:
        path_key, hash_key = prefix + 'source_path', prefix + 'source_sha256'
        if {path_key, hash_key} <= set(table.columns):
            for source, expected in table[[path_key, hash_key]].dropna().itertuples(index=False, name=None):
                if source in EXTRACTION_HASHES and EXTRACTION_HASHES[source] != expected:
                    raise ValueError(f'Conflicting retained source digests: {source}')
                EXTRACTION_HASHES[source] = expected
    if table.empty:
        raise ValueError(f'Empty source table: {relative}')
    return table


def rec(data, **conditions):
    for key, value in conditions.items():
        data = data[data[key] == value]
    if len(data) != 1:
        raise ValueError((conditions, len(data)))
    return data.iloc[0]


def label(arm: str) -> str:
    return checkpoint_labels.label(arm)


def figure(height):
    return plt.figure(figsize=(WIDTH, height / 72))


def axis(fig, rect, labels=None, xlabel=''):
    width, height = fig.get_size_inches() * 72
    x, y, w, h = rect
    a = fig.add_axes([x / width, y / height, w / width, h / height])
    a.spines[['top', 'right']].set_visible(False)
    a.set_xlabel(xlabel, labelpad=2)
    if labels is not None:
        a.set_yticks(range(len(labels)), labels)
        a.set_ylim(len(labels) - .6, -.6)
        a.tick_params(axis='y', length=0, pad=3)
    a.axvline(0, color=palette.CONTROL, lw=.55, ls=':', zorder=0)
    return a


def heading(fig, x, y, letter, title):
    width, height = fig.get_size_inches() * 72
    fig.text(x / width, y / height, letter, weight='bold', fontsize=LETTER, gid=f'panel-{letter}')
    fig.text((x + 13) / width, y / height, title, va='baseline')


def interval(a, y, row, color=BLUE, offset: float = 0):
    point, low, high = [float(row[k]) for k in ['estimate', 'ci_low', 'ci_high']]
    if not np.isfinite([point, low, high]).all() or not low <= point <= high:
        raise ValueError(('invalid interval', point, low, high))
    a.plot([low, high], [y + offset] * 2, color=color, lw=.7)
    a.plot(point, y + offset, 'o', ms=3, mec=color, mew=.7,
           mfc=color if low > 0 or high < 0 else 'white')
    a._intervals = getattr(a, '_intervals', []) + [(low, high)]


def save(fig, name, frames, unit, uncertainty, supplement=False):
    destination = OUT
    previews = SUPPLEMENT_DATA if supplement else DATA
    destination.mkdir(parents=True, exist_ok=True)
    previews.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    for a in fig.axes:
        lo, hi = sorted(a.get_xlim())
        for low, high in getattr(a, '_intervals', []):
            if not lo <= low <= high <= hi:
                raise ValueError(('clipped interval', name, low, high, lo, hi))
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    clipped = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < bounds.x0 - 1 or box.x1 > bounds.x1 + 1 or box.y0 < bounds.y0 - 1 or box.y1 > bounds.y1 + 1:
            clipped.append(text.get_text())
    if clipped:
        raise ValueError(('text outside figure', name, clipped))
    paths = {'pdf': destination / publication_paths.PDF_NAMES[name], 'svg': previews / f'{name}.svg',
             'plotted_source': previews / f'{name}-plotted.csv'}
    fig.savefig(paths['pdf'])
    fig.savefig(paths['svg'])
    paths['svg'].write_text('\n'.join(line.rstrip() for line in paths['svg'].read_text().splitlines()) + '\n')
    fig.savefig(previews / f'{name}.png', dpi=450)
    pd.concat(frames, ignore_index=True).to_csv(paths['plotted_source'], index=False, lineterminator='\n')
    texts = [text for text in fig.findobj(Text)
             if text.get_visible() and text.get_text()]
    CHECKS.append({'figure': name, 'width_inches': float(fig.get_figwidth()), 'height_inches': float(fig.get_figheight()),
                   'minimum_font_points': min(text.get_fontsize() for text in texts), 'panel_letter_points': LETTER,
                   'panel_labels': [{'svg_id': text.get_gid(), 'text': text.get_text()}
                                    for text in texts if (text.get_gid() or '').startswith('panel-')],
                   'all_retained_intervals_inside_axes': True, 'all_text_inside_canvas': True} |
                  {f'{kind}_{field}': str(path.relative_to(REPO)) if field == 'path' else digest(path)
                   for kind, path in paths.items() for field in ['path', 'sha256']})
    CONTRACTS.append({'figure': name, 'source': str(paths['plotted_source'].relative_to(REPO)),
                      'statistical_unit': unit, 'uncertainty': uncertainty})
    plt.close(fig)


def clean_interactions(refresh=False):
    if not refresh:
        table = read('clean-interaction-source')
        validate_source_rows(table, 33, ['estimate', 'ci_low', 'ci_high', 'groups', 'cycles', 'site_pairs'])
        values = table[['estimate', 'ci_low', 'ci_high']].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not ((values[:, 1] <= values[:, 0]) & (values[:, 0] <= values[:, 2])).all():
            raise ValueError('Invalid clean-interaction intervals')
        return table
    path = REPO / 'results/R3/pairwise_residual_full_panel_indel_excluded_20260928/panel.json'
    source_digest = digest(path)
    EXTRACTION_HASHES[str(path.relative_to(REPO))] = source_digest
    data = json.loads(path.read_text())
    rows = []
    for arm, row in data['arms'].items():
        low, high = row['simultaneous95_interval_kcal2']
        rows.append(dict(arm=arm, estimate=row['seed_mean_mse_reduction_kcal2'], ci_low=low, ci_high=high,
                         groups=64, cycles=8189, site_pairs=214, source_path=str(path.relative_to(REPO)),
                         source_sha256=source_digest, source_pointer=f'arms.{arm}'))
    result = pd.DataFrame(rows)
    result.to_csv(DATA / 'clean-interaction-source.csv', index=False)
    EXPECTED_HASHES[str((DATA / 'clean-interaction-source.csv').relative_to(REPO))] = digest(DATA / 'clean-interaction-source.csv')
    return clean_interactions()


def validate_source_rows(table, count, columns):
    required = {'arm', 'source_path', 'source_sha256', 'source_pointer', *columns}
    if not required <= set(table.columns) or len(table) != count or table.arm.nunique() != count:
        raise ValueError(f'Invalid retained {count}-arm source table')
    if table[list(required)].isna().any().any() or not table.source_sha256.map(lambda value: bool(re.fullmatch('[0-9a-f]{64}', str(value)))).all():
        raise ValueError('Missing source-row provenance')
    for column in set(columns) - {'estimate', 'ci_low', 'ci_high', 'support_rule'}:
        values = table[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not ((values > 0) & (values == np.floor(values))).all():
            raise ValueError(f'Invalid concrete support counts: {column}')


def position_support(refresh=False):
    """Read retained counts; only explicit refresh opens the hash-bound fit files."""
    if not refresh:
        table = read('position-support-source')
        validate_source_rows(table, 32, ['assays', 'variants', 'families', 'support_rule'])
        return table
    panel_path = REPO / 'results/R1/position_terms_20260926/position_simultaneous.json'
    panel = json.loads(panel_path.read_text())
    rows = []
    for source, expected in panel['source_sha256'].items():
        path = REPO / source
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Position fit digest mismatch: {source}')
        EXTRACTION_HASHES[source] = actual
        fit = json.loads(path.read_text())
        if fit['fold_seed'] == PRIMARY:
            rows.append({key: fit[key] for key in ['arm', 'assays', 'variants', 'families', 'support_rule']} |
                        dict(source_path=source, source_sha256=actual, source_pointer='assays;variants;families;support_rule'))
    table = pd.DataFrame(rows)
    if set(table.arm) != set(panel['arms']) or len(table) != 32:
        raise ValueError('Position support does not match the current 32-arm panel')
    table.to_csv(DATA / 'position-support-source.csv', index=False)
    EXTRACTION_HASHES[str(panel_path.relative_to(REPO))] = digest(panel_path)
    EXPECTED_HASHES[str((DATA / 'position-support-source.csv').relative_to(REPO))] = digest(DATA / 'position-support-source.csv')
    return position_support()


def schematic_axes(fig, rect):
    """A point-coordinate canvas for editable, explicitly non-data diagrams."""
    width, height = fig.get_size_inches() * 72
    x, y, w, h = rect
    a = fig.add_axes([x / width, y / height, w / width, h / height])
    a.set_xlim(0, w); a.set_ylim(0, h); a.set_axis_off()
    return a


def structure_insets():
    """Authenticate the distributed bundle; selection provenance is documentary only."""
    directory = DATA.parent / 'structure-insets'
    manifest_path = directory / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError('Missing structure-inset bundle manifest')
    manifest = json.loads(manifest_path.read_text())
    keys = ['dvl1-abundance', '1ufm-stability', '1ufm-pair']
    selection = manifest.get('source_selection', {})
    if (manifest.get('schema') != 'manuscript-structure-insets-v1'
            or not isinstance(manifest.get('images'), dict)
            or selection.get('path') != publication_paths.STRUCTURE_SELECTION.as_posix()
            or not re.fullmatch('[0-9a-f]{64}', str(selection.get('sha256', '')))):
        raise ValueError('Invalid structure-inset bundle manifest')
    images = {}
    for key in keys:
        entry = manifest['images'].get(key, {})
        filename = entry.get('file', '')
        if (not filename or Path(filename).name != filename or not filename.endswith('.png')
                or not entry.get('record_id')
                or not re.fullmatch('[0-9a-f]{64}', str(entry.get('sha256', '')))):
            raise ValueError(f'Invalid structure-inset entry: {key}')
        path = directory / filename
        if not path.is_file():
            raise ValueError(f'Missing structure-inset image: {key}')
        actual = digest(path)
        if actual != entry['sha256']:
            raise ValueError(f'Structure-inset digest mismatch: {key}')
        HASHES[str(path.relative_to(REPO))] = actual
        images[key] = plt.imread(path)
    HASHES[str(manifest_path.relative_to(REPO))] = digest(manifest_path)
    return images


def draw_structure(inset, image):
    """Trim white export margins without changing the reference geometry."""
    visible = np.any(image[..., :3] < .98, axis=2)
    if image.shape[2] == 4:
        visible &= image[..., 3] > .01
    ys, xs = np.where(visible)
    if not len(xs):
        raise ValueError('Empty structure-inset image')
    cropped = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height, width = cropped.shape[:2]
    scale = min(84 / width, 50 / height)
    w, h = width * scale, height * scale
    inset.imshow(cropped, extent=(44 - w / 2, 44 + w / 2, 36 - h / 2, 36 + h / 2))


def generation_verdict_counts(streams):
    """Descriptive count of retained cell intervals; no additional confidence claim."""
    return (int((streams.ci_low > 0).sum()), int((streams.ci_high < 0).sum()),
            int(((streams.ci_low <= 0) & (streams.ci_high >= 0)).sum()))


def main_figures():
    images = structure_insets()  # Fail before writing figures if the bundle is absent or altered.
    position_support()
    family, rout, gaps = read('lineage-robustness-source'), read('readout-contrasts-source'), read('controlled-gap-source')
    fam = pd.concat([family.loc[family['protein_pretraining'].eq(group)].sort_values('release_family') for group in ['yes', 'no']]).reset_index(drop=True)
    fig = figure(425)
    heading(fig, 3, 413, 'a', 'Release-family likelihood increments')
    a = axis(fig, (61, 196, 103, 206), fam.release_family.tolist(), 'Change in Spearman ρ')
    for i, row in fam.iterrows():
        interval(a, i, row, BLUE if row.protein_pretraining == 'yes' else ORANGE)
    a.set_xlim(-.003, .036); a.set_xticks([0, .01, .02, .03])
    fig.text(.015, .35, 'Blue: specialized or adapted\nOrange: general text / scientific joint', linespacing=1.4)
    heading(fig, 176, 413, 'b', 'Readout comparisons')
    arms = [arm for arm in ORDER if arm in set(rout.arm)]
    b = axis(fig, (230, 137, 136, 265), [label(arm) for arm in arms], 'Change in Spearman ρ')
    for i, arm in enumerate(arms):
        row = rec(rout, arm=arm)
        for key, off, color in [('readout_minus_likelihood', -.18, palette.READOUT), ('representation_increment', .18, BLUE)]:
            interval(b, i, dict(estimate=row[key], ci_low=row[key + '_ci_low'], ci_high=row[key + '_ci_high']), color, off)
    b.set_xlim(-.095, .225); b.set_xticks([0, .1, .2])
    diagram = schematic_axes(fig, (6, 114, 175, 53))
    diagram.text(0, 28, 'Native score', color=palette.READOUT)
    diagram.annotate('', (85, 30), (58, 30),
                     arrowprops=dict(arrowstyle='->', color=palette.READOUT, lw=.7))
    diagram.text(90, 28, 'readout − native', color=palette.READOUT)
    diagram.text(0, 10, 'Matched baseline', color=BLUE)
    diagram.annotate('', (85, 12), (65, 12),
                     arrowprops=dict(arrowstyle='->', color=BLUE, lw=.7))
    diagram.text(90, 10, '+ representation', color=BLUE)
    heading(fig, 3, 106, 'c', 'Representation − likelihood over one matched baseline')
    matrix = np.array([[rec(gaps, arm=arm, split_seed=seed).estimate for arm in arms] for seed in sorted(gaps.split_seed.unique())])
    c = fig.add_axes((43 / 370.08, 77 / 425, 291 / 370.08, 24 / 425))
    im = c.imshow(matrix, cmap=CMAP, norm=TwoSlopeNorm(vmin=-.022, vcenter=0, vmax=.022), aspect='auto', interpolation='none')
    c.set_yticks(range(3), ['Split 1', 'Split 2', 'Split 3'])
    c.set_xticks(range(len(arms)), [label(arm) for arm in arms], rotation=90)
    c.tick_params(length=0, pad=2)
    for i, seed in enumerate(sorted(gaps.split_seed.unique())):
        for j, arm in enumerate(arms):
            row = rec(gaps, arm=arm, split_seed=seed)
            if row.ci_low > 0 or row.ci_high < 0:
                c.plot(j, i, '.', ms=2, color=palette.CONTROL)
    for spine in c.spines.values(): spine.set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_axes((342 / 370.08, 77 / 425, 5 / 370.08, 24 / 425)), ticks=[-.02, 0, .02])
    cb.ax.tick_params(length=1, pad=1)
    save(fig, 'Fig2_mutation', [fam.assign(panel='a'), rout.assign(panel='b'), gaps.assign(panel='c')],
         '163 biological groups', 'a: simultaneous 95%, 10000 draws; b/c: pointwise 95%, 2000 draws')

    pos = read('position-terms-replot'); selected = pos[pos.control == 'wall']
    arms = [arm for arm in ORDER if arm in set(selected.arm)]
    fig = figure(418)
    sequence = schematic_axes(fig, (72, 367, 288, 46))
    sequence.text(0, 36, 'Likelihood partition (schematic residues, not token boundaries)')
    for i, residue in enumerate('ACDEFGHIK'):
        color = palette.SITE if i == 3 else palette.DOWNSTREAM if i > 3 else palette.CONTROL
        sequence.add_patch(Rectangle((i * 15, 13), 13, 14, facecolor='white', edgecolor=color, lw=.8))
        sequence.text(i * 15 + 6.5, 20, residue, ha='center', va='center', color=color)
    sequence.text(48, 0, 'site', color=palette.SITE)
    sequence.text(86, 0, 'downstream', color=palette.DOWNSTREAM)
    sequence.text(170, 20, 'Full score = all terms', va='center')
    for j, (key, title, color) in enumerate([('unique_own', 'Site |\ndownstream', palette.SITE), ('unique_downstream', 'Downstream |\nsite', palette.DOWNSTREAM),
                                           ('joint_over_full', 'Both |\nfull score', palette.JOINT_TERM), ('own_minus_downstream', 'Site −\ndownstream', palette.TERM_CONTRAST)]):
        x = 72 + 74 * j
        heading(fig, x - 4, 350, chr(97 + j), title)
        a = axis(fig, (x, 39, 65, 298), [label(arm) for arm in arms] if j == 0 else [''] * len(arms), 'Δ Spearman ρ')
        for i, arm in enumerate(arms): interval(a, i, rec(selected, arm=arm, endpoint=key), color)
        a.set_xlim(-.027, .045); a.set_xticks([-.02, 0, .04], ['−.02', '0', '.04'])
    fig.text(.24, .016, '32 aligned subsets; simultaneous 95% intervals over 128 contrasts')
    save(fig, 'Fig3_position', [selected], '153–156 available families per checkpoint', 'simultaneous 95%; 10000 shared family draws')

    abundance, stability, clean = read('external-abundance-source'), read('stability-panel-source'), clean_interactions()
    abundance = abundance[(abundance.endpoint == 'primary_likelihood_spearman') & (abundance.split_seed == PRIMARY)]
    arms = [arm for arm in ORDER if arm in set(stability.arm)]
    fig = figure(445)
    for j, (key, caption) in enumerate([('dvl1-abundance', 'DVL1 PDZ'),
                                        ('1ufm-stability', 'Mouse PCI / 1UFM'),
                                        ('1ufm-pair', 'PCI / two sites')]):
        inset = schematic_axes(fig, (76 + 96 * j, 357, 88, 63))
        draw_structure(inset, images[key])
        inset.text(44, 0, caption, ha='center')
    fig.text(.21, 344 / 445, 'WT references; marked sites only, not mutant structures')
    for j, (data, title, xlabel, color) in enumerate([(abundance, 'Abundance ranking', 'Δ Spearman ρ', BLUE),
                                                    (stability, 'Stability singles', 'MSE reduction\n(kcal²/mol²)', BLUE),
                                                    (clean, 'Adjusted interactions', 'MSE reduction\n(kcal²/mol²)', ORANGE)]):
        x = 76 + 96 * j
        heading(fig, x - 5, 432, chr(97 + j), title)
        a = axis(fig, (x, 43, 87, 297), [label(arm) for arm in arms] if j == 0 else [''] * len(arms), xlabel)
        for i, arm in enumerate(arms): interval(a, i, rec(data, arm=arm), color)
        if j == 0: a.set_xlim(-.002, .013); a.set_xticks([0, .005, .01])
        if j == 1: a.set_xlim(-.008, .034); a.set_xticks([0, .01, .02, .03])
        if j == 2:
            values = clean[['ci_low', 'ci_high']].to_numpy()
            low, high = values.min(), values.max(); margin = (high - low) * .08
            a.set_xlim(low - margin, high + margin)
            a.set_xticks([-.004, -.002, 0], ['−.004', '−.002', '0'])
    save(fig, 'Fig4_phenotypes_and_interactions', [abundance.assign(panel='a'), stability.assign(panel='b'), clean.assign(panel='c')],
         'a: 96; b: 101; c: 64 biological groups', 'a: pointwise 95%; b/c: simultaneous 95% across 33 checkpoints; different targets/scales')

    context, streams = read('context-ranking-source'), read('generation-streams-replot')
    streams = streams[streams.endpoint == 'complete_domain']
    cells = sorted(streams.cell, key=lambda cell: ((GEN + ['zymctrl']).index(cell.split('__')[0]), cell))
    fig = figure(440)
    heading(fig, 3, 427, 'a', 'Homologous context: recovery versus net benefit')
    arms = ['progen3-3b', 'proteinglm-7b-clm', 'prollama-stage-1', 'prollama']
    a = axis(fig, (86, 353, 275, 60), [label(arm) for arm in arms], 'Mutation-ranking change in Spearman ρ')
    for i, arm in enumerate(arms):
        for key, offset, color in [('delta_spearman', -.15, BLUE), ('homolog_minus_no_context_spearman', .15, ORANGE)]:
            interval(a, i, rec(context, arm=arm, endpoint=key), color, offset)
    a.set_xlim(-.17, .22); a.set_xticks([-.1, 0, .1, .2])
    a.legend(handles=[Line2D([], [], color=BLUE, label='Homolog − unrelated'), Line2D([], [], color=ORANGE, label='Homolog − no context')],
             frameon=False, loc='lower left', bbox_to_anchor=(0, 1), ncol=2, borderpad=0, handlelength=1)
    context_diagram = schematic_axes(fig, (18, 290, 343, 36))
    context_diagram.text(0, 26, 'Prefix (schematic)')
    for x, text, color in [(105, 'Homolog', BLUE), (190, 'Unrelated', palette.CONTROL),
                            (279, 'None', ORANGE)]:
        context_diagram.text(x, 26, text, ha='center', color=color)
        context_diagram.annotate('', (x, 11), (x, 22),
                                 arrowprops=dict(arrowstyle='->', lw=.7, color=color))
        context_diagram.text(x, 0, 'target', ha='center')
    counts = generation_verdict_counts(streams)
    strip = schematic_axes(fig, (86, 239, 275, 25))
    strip.text(0, 17, '20 cell intervals versus zero (exploratory):')
    x = 0
    for count, verdict in zip(counts, ['above', 'below', 'overlap']):
        width = 275 * count / len(streams)
        strip.add_patch(Rectangle((x, 0), width, 10, facecolor='white',
                                  edgecolor=palette.COMPLETE_DOMAIN, lw=.7))
        strip.text(x + width / 2, 5, f'{count} {verdict}', ha='center', va='center')
        x += width
    heading(fig, 3, 267, 'b', 'Complete-domain recognition relative to natural fragments')
    b = axis(fig, (86, 34, 275, 189), [label(cell) for cell in cells], 'Model − fragment recognition rate')
    for i, cell in enumerate(cells): interval(b, i, rec(streams, cell=cell), palette.COMPLETE_DOMAIN)
    b.set_xlim(-.40, .68); b.set_xticks([-.4, -.2, 0, .2, .4, .6])
    fig.text(.015, .015, '† Requested condition     Three streams × 800 attempts per cell; fixed checkpoints')
    save(fig, 'Fig5_context_and_generation', [context.assign(panel='a'), streams.assign(panel='b')],
         'a: 134 biological groups; b: three generation streams', 'a: pointwise 95%; b: 95% Student-t, df=2, not panel simultaneous')


def supplementary_figures():
    native, text = read('native-fitness-source'), read('text-aa-fitness-source')
    fig = figure(250)
    for j, (data, ordering, title) in enumerate([(native, GEN, 'Native protein interfaces'), (text, TXT, 'Literal amino-acid strings')]):
        data = data[data.endpoint == 'model_minus_lookup']
        arms = [arm for arm in ordering if arm in set(data.arm)]
        heading(fig, 3 + 185 * j, 235, chr(97 + j), title)
        a = axis(fig, (85 + 185 * j, 37, 92, 182), [label(arm) for arm in arms], 'Native − profile (Δρ)')
        for i, arm in enumerate(arms): interval(a, i, rec(data, arm=arm), BLUE)
        a.set_xlim(-.42, .13); a.set_xticks([-.4, -.2, 0])
    save(fig, 'native-comparators', [native, text], 'native: checkpoint-specific groups; text: 169 groups', 'pointwise 95%', True)

    cond = read('conditional-class-yield-source')
    fig = figure(355)
    for j, arm in enumerate(['zymctrl', 'prollama']):
        selected = cond.loc[cond['arm'].eq(arm) & cond['condition'].eq('requested')].sort_values('class_key')
        labels = [row.class_key + ('' if row.primary_class else ' *') for _, row in selected.iterrows()]
        heading(fig, 3 + 185 * j, 341, chr(97 + j), label(arm) + (' EC classes' if j == 0 else ' superfamilies'))
        a = axis(fig, (58 + 185 * j, 50, 113, 278), labels, 'Fraction of 200 attempts')
        for i, (_, row) in enumerate(selected.iterrows()):
            mismatch = rec(cond, arm=arm, class_key=row.class_key, condition='mismatched')
            color = BLUE if row.primary_class else palette.CONTROL
            a.plot([mismatch.target_profile_rate, row.target_profile_rate], [i, i], color=color, lw=.6)
            a.plot(row.target_profile_rate, i, 'o', color=color, ms=3)
            a.plot(mismatch.target_profile_rate, i, 'o', mec=color, mfc='white', ms=3)
            a.plot(row.distinct_target_groups_per_attempt, i + .17, 's', color=ORANGE, ms=2.6)
        a.set_xlim(-.03, 1.03); a.set_xticks([0, .5, 1])
    fig.text(.04, .035, 'Filled circle: requested   Open circle: mismatched   Square: distinct target groups')
    fig.text(.04, .008, '* Excluded by natural-reference qualification; exact saved-batch rates')
    save(fig, 'conditional-class-yield', [cond], '200 attempts per class and condition', 'descriptive census; no sampling interval', True)

    census = read('generation-yield-source'); arms = [arm for arm in GEN if arm in set(census.arm)]
    fig = figure(280)
    for j, (keys, title) in enumerate([(['nonempty_yield', 'exact_unique_yield'], 'Sequence output'), (['profile_yield', 'profile_group_yield'], 'Profile recognition')]):
        heading(fig, 83 + 144 * j, 267, chr(97 + j), title)
        a = axis(fig, (89 + 142 * j, 53, 130, 202), [label(arm) for arm in arms] if j == 0 else [''] * len(arms), 'Fraction of 800 attempts')
        for i, arm in enumerate(arms):
            row = rec(census, arm=arm)
            for key, off, marker, color in zip(keys, [-.16, .16], ['o', 's'], [BLUE, ORANGE]):
                if pd.notna(row[key]): a.plot(row[key], i + off, marker, color=color, ms=3)
        a.set_xlim(-.03, 1.03); a.set_xticks([0, .5, 1])
    fig.text(.24, .042, 'Circle: nonempty / any profile hit')
    fig.text(.24, .013, 'Square: exactly unique / distinct recognized groups')
    save(fig, 'generation-yield', [census], '800 attempts per procedure', 'descriptive census; no sampling interval', True)

    gen = read('matched-generator-source')
    cells = sorted(gen[(gen.endpoint == 'any_family') & (gen.cohort == 'fragment')].cell,
                   key=lambda cell: ((GEN + ['zymctrl']).index(cell.split('__')[0]), cell))
    cohorts = ['shuffle', 'hydropathy', 'markov_0', 'markov_2', 'markov_4', 'fragment', 'natural']
    fig = figure(310)
    heading(fig, 3, 296, 'a', 'Recognition over every attempt')
    a = axis(fig, (99, 200, 255, 81), xlabel='')
    a.lines[0].remove()
    for ep, color, marker in [('any_family', BLUE, 'o'), ('complete_domain', palette.COMPLETE_DOMAIN, 's')]:
        vals = [rec(gen, cell=cell, endpoint=ep, cohort='fragment').model_rate for cell in cells]
        a.plot(range(20), vals, marker, color=color, ms=3)
        a.axhline(rec(gen, endpoint=ep, cohort='ceiling').model_rate, color=palette.CONTROL, lw=.7, ls=':')
    a.set_xlim(-.5, 19.5); a.set_ylim(-.03, 1.03); a.set_xticks([]); a.set_yticks([0, .5, 1]); a.set_ylabel('Recognition rate')
    a.legend(handles=[Line2D([], [], marker='o', ls='', color=BLUE, label='Any family'), Line2D([], [], marker='s', ls='', color=palette.COMPLETE_DOMAIN, label='Complete domain')],
             loc='lower left', bbox_to_anchor=(0, 1), ncol=2, frameon=False, borderpad=0)
    heading(fig, 3, 179, 'b', 'Any-family model − cohort rate')
    b = fig.add_axes((99 / 370.08, 92 / 310, 255 / 370.08, 78 / 310))
    matrix = np.array([[rec(gen, cell=cell, endpoint='any_family', cohort=cohort).difference for cell in cells] for cohort in cohorts])
    im = b.imshow(matrix, cmap=CMAP, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), aspect='auto', interpolation='none')
    b.set_yticks(range(7), ['Shuffle', 'Hydropathy', 'Order 0', 'Order 2', 'Order 4', 'Fragment', 'Whole record'])
    b.set_xticks(range(20), [label(cell) for cell in cells], rotation=90)
    b.tick_params(length=0, pad=2)
    for i, cohort in enumerate(cohorts):
        for j, cell in enumerate(cells):
            row = rec(gen, cell=cell, endpoint='any_family', cohort=cohort)
            if row.ci_low > 0 or row.ci_high < 0: b.plot(j, i, '.', color=palette.CONTROL, ms=2)
            if not bool(row.qualified): b.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch='///', ec=palette.CONTROL, lw=0))
    b.axhline(5.5, color=palette.CONTROL, lw=.6)
    for spine in b.spines.values(): spine.set_visible(False)
    fig.colorbar(im, cax=fig.add_axes((14 / 370.08, 31 / 310, 55 / 370.08, 5 / 310)), orientation='horizontal', ticks=[-1, 0, 1])
    save(fig, 'matched-generators', [gen], '800 attempts per cell; frozen near-duplicate groups', '97.5% paired percentile intervals, 4000 draws; dots show either-sign exclusion', True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supplement-only', action='store_true')
    parser.add_argument('--refresh-source-tables', action='store_true',
                        help='Explicitly re-extract clean interactions and position counts from retained raw receipts')
    args = parser.parse_args()
    HASHES.clear(); CHECKS.clear(); CONTRACTS.clear(); EXPECTED_HASHES.clear(); EXTRACTION_HASHES.clear()
    previous = json.loads((DATA / 'figure_validation.json').read_text())
    EXPECTED_HASHES.update(previous['source_sha256'])
    EXTRACTION_HASHES.update(previous.get('extraction_source_sha256', {}))
    EXTRACTION_HASHES.update({path: value for path, value in previous['source_sha256'].items()
                              if not path.endswith('.csv')})
    renderer_hashes = {str(path.relative_to(REPO)): digest(path)
                       for path in [Path(__file__).resolve(), palette.PATH, checkpoint_labels.PATH, publication_paths.PATH]}
    if args.supplement_only and previous.get('renderer_sha256') != renderer_hashes:
        raise ValueError('Run the full renderer first: main figures were made by a different renderer')
    DATA.mkdir(parents=True, exist_ok=True)
    if args.refresh_source_tables and args.supplement_only:
        raise ValueError('Source-table refresh requires a full render')
    if args.refresh_source_tables:
        clean_interactions(refresh=True)
        position_support(refresh=True)
    if not args.supplement_only: main_figures()
    supplementary_figures()
    receipt_data = SUPPLEMENT_DATA if args.supplement_only else DATA
    pd.DataFrame(CONTRACTS).to_csv(receipt_data / ('supplement_figure_data_contract.csv' if args.supplement_only else 'figure_data_contract.csv'), index=False, lineterminator='\n')
    current_checks = CHECKS
    current_hashes = HASHES
    if args.supplement_only:
        updated_paths = {check['pdf_path'] for check in CHECKS}
        current_checks = [check for check in previous['checks'] if check['pdf_path'] not in updated_paths] + CHECKS
        current_hashes = previous['source_sha256'] | HASHES
    auxiliary = {str(path.relative_to(REPO)): digest(path) for path in checkpoint_labels.write_key(REPO)}
    # Authenticated illustration inputs belong to this render, not to the frozen
    # experimental extraction. Do not carry their previous hashes in both roles.
    inset_directory = (DATA.parent / 'structure-insets').relative_to(REPO)
    illustration_inputs = {path for path in current_hashes if Path(path).parent == inset_directory}
    extraction_hashes = {path: value for path, value in EXTRACTION_HASHES.items() if path not in illustration_inputs}
    report = dict(schema='figure_render_validation_v1', checks=current_checks, source_sha256=current_hashes,
                  auxiliary_outputs_sha256=auxiliary,
                  extraction_source_sha256=extraction_hashes,
                  renderer_sha256=renderer_hashes,
                  insertion_width_inches=WIDTH,
                  minimum_font_points=BASE, panel_letter_points=LETTER,
                  typography_policy='Nature official specifications: 5–7 pt non-panel text, 8 pt bold lower-case panels; adopted 7/8 pt',
                  typography_source='https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications',
                  scope='Retained estimates only; no inference, model scoring, fitting or resampling. Figure 1 is external.')
    (DATA / 'figure_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Rendered {len(CHECKS)} quantitative figures; Fig. 1 was not modified.')


if __name__ == '__main__':
    main()
