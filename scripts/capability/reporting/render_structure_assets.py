#!/usr/bin/env python3
"""Supporting structure illustrations for the generation section, drawn from already-measured candidates.

What these are
==============

Orientation material for the author, and nothing else. Every picture here is a
C-alpha trace of an **ESMFold2 computational prediction**. None of it is measured
folding, measured thermodynamic stability or measured function, and no generated
sequence in this study carries an experimental endpoint. These files are not
manuscript figures, are not evidence, and must never stand in for either.

Where the candidates come from
==============================

The primary E20 confirmatory campaign (progen2-medium, protgpt2) retained its
per-candidate measurements and **no structure coordinates**. Every generated fold
that can be drawn at all therefore comes from the ProGen3 cross-model robustness
campaign -- ``progen3-3b`` and ``progen3-112m``, on the same eight sealed
evaluation domains, at the same arms. These illustrations say nothing about the
primary models and are labelled accordingly in every caption.

The residue correspondence
==========================

Superposition uses the **identity correspondence**: residue *i* of the candidate
against residue *i* of the wild-type reference. That is the correspondence the
manuscript's TM-score endpoint is defined on, and it is available because every
candidate is decoded at its domain's length. It is *not* a structural alignment
that searched for the best correspondence, and no such search happens anywhere in
this file.

Two superpositions exist at that one correspondence and the difference matters.
``fold_geometry.tm_score`` reports the maximum over rigid-body superpositions,
found by the standard iterative-extension search; that maximum is the endpoint the
manuscript quotes. The single fit on all residues scores far lower when the two
folds disagree (0.03 against an endpoint of 0.14 is typical in these pools). A
panel annotated with the endpoint has to *show* the superposition that attains it,
so :func:`tm_superposition` repeats the frozen module's search in order to return
its transform, which the frozen public interface does not expose. It then asserts
its own maximum equals the retained ``tm_score`` exactly; if the two ever diverge
the renderer fails rather than drawing a superposition that is not the endpoint's.

Honesty constraints that are enforced, not just documented
==========================================================

* A structure-derived quantity may never rank or choose a candidate.
  ``plddt_mean``, ``ptm`` and ``tm_score`` are refused as selectors by
  :func:`selector_ranking`, which reuses the frozen registry in
  ``analyse_e20_confirmatory``. The TM oracle is the single declared exception,
  is computed outside that registry, and is labelled a ceiling rather than a
  method wherever it appears.
* Every coordinate file opened is checked against the digest its record declares,
  and every annotated TM-score is recomputed from those coordinates and must
  match the retained value.
* Matplotlib stamps a wall-clock date into PDF and SVG output and salts SVG
  element ids at random, which has already cost this project a set of
  irreproducible figures. The salt is fixed below and the dates are dropped on
  save, so a re-render is byte-identical; ``--verify`` renders everything twice
  and compares digests.

Usage
=====

    python scripts/capability/reporting/render_structure_assets.py --verify

Outputs land in ``manuscript/assets/structures`` beside a ``manifest.json`` that
maps every image to the candidate ids, domains, arms, measurements and source
digests behind each panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts' / 'capability' / 'generation'))

import analyse_e20_confirmatory as e20  # noqa: E402
from src.capability.context.family_potts import load_family_model  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.ladder import fold_geometry as fg  # noqa: E402

if __package__:
    from . import palette
else:
    import palette  # noqa: E402

SCHEMA = 'manuscript-structure-assets-v1'
ENTRY_POINT = 'scripts/capability/reporting/render_structure_assets.py'
COMMAND = f'python {ENTRY_POINT} --verify'
OUT_DIRECTORY = 'manuscript/assets/structures'

# ---------------------------------------------------------------- retained inputs

WILD_TYPE = 'results/R6/e20_wild_type_references_20261013/freeze/wild_type_references.json'
WILD_TYPE_FOLD = 'results/R6/e20_wild_type_references_20261013/fold'
CAMPAIGN = 'results/R6/progen3_robustness_20261015'
ANALYSIS = 'results/R6/progen3_robustness_analysis_20261015/robustness.json'
FAMILY_REFERENCES = 'data/pairwise_assets/family_references_jointh_20261013'
#: The permutation seed the retained analysis was run under. Only the permuted-coupling
#: selector reads it; it is carried so the reused registry is called exactly as the
#: analysis called it.
PERMUTATION_SEED = 20261013

#: Precision is part of the native contract, so the campaign is two measure lines.
MEASURE_LINES = ('bfloat16', 'float16')
SEALED = e20.SEALED_DOMAINS
DEVELOPMENT = ('1K1V', '1UFM', '2KWH', '3L1X')

#: Reference scales, all of them the project's own declared numbers.
SAME_FOLD_TM = 0.5
MARGIN_TM = e20.MARGIN
ADMISSIBLE_IDENTITY = e20.ADMISSIBLE_IDENTITY
SELECTION_K = e20.SELECTION_K
SELECTION_POOL_ARM = e20.SELECTION_POOL_ARM

#: The two admissible scorers drawn in the selection asset, and the ceiling.
#: Both scorers come from the frozen registry, which cannot see a structure-derived
#: field. Native likelihood is the one a reader is most likely to assume works.
SCORERS = (('native_loglik', 'Native likelihood'), ('potts_full', 'Potts full'))
ORACLE = 'TM oracle'

ARM_LABELS = {'N': 'unguided', 'Pr': 'profile guidance', 'M': 'fields-only guidance',
              'C': 'coupling guidance', 'Cx': 'permuted-coupling control'}
#: The same arms, short enough to sit under a narrow panel without colliding with its neighbour.
ARM_SHORT = {'N': 'unguided', 'Pr': 'profile', 'M': 'fields only', 'C': 'coupling',
             'Cx': 'permuted control'}

STATEMENT = (
    'Supporting illustrations, not figures and not evidence. Every structure shown is an ESMFold2 '
    'computational prediction, including the wild-type references. Nothing here is measured folding, '
    'measured stability or measured function, and no generated sequence in this study has an '
    'experimental endpoint. The candidates are from the ProGen3 cross-model robustness campaign; the '
    'primary E20 confirmatory campaign retained per-candidate measurements and no coordinates, so no '
    'picture of a primary-model candidate exists and none is implied here.')

CORRESPONDENCE = (
    'Identity residue correspondence: residue i of the candidate against residue i of the wild-type '
    'reference, which is the correspondence the TM-score endpoint is defined on and is available '
    'because every candidate is decoded at its domain length. Not a structural alignment, and no '
    'search over correspondences happens. Within that fixed correspondence the drawn rigid-body '
    'superposition is the one that attains the retained TM-score (the endpoint maximises over '
    'superpositions), recomputed from the coordinates and required to match the retained value.')

ORACLE_STATEMENT = e20.CAPTURE_ORACLE_STATEMENT

# ---------------------------------------------------------------- typography and palette

WIDTH = 5.14  # inches; the manuscript figure width, so a panel can be lifted straight into a figure
CANVAS = WIDTH * 72  # points
MARGIN = 7.0
TEXT_WIDTH = CANVAS - 2 * MARGIN
BASE = 7.0
LETTER = 8.0
INK, SLATE, NEUTRAL, BORDER = palette.INK, palette.SLATE, palette.NEUTRAL, palette.BORDER
#: The reference is drawn as a wide, light backdrop rather than a second foreground trace, so
#: the deviation ramp -- whose low end is the palette's anchor blue -- stays readable on top of it.
REFERENCE_COLOUR = BORDER
#: Sequential ramps mixed from the publication palette anchors, the way palette.tint does.
#: Deviation runs anchor-blue (agrees) to anchor-red (disagrees); predicted confidence runs
#: the same ramp reversed, so high confidence is blue as in the folding literature.
DEVIATION_CMAP = LinearSegmentedColormap.from_list(
    'ca_deviation', [palette.BINARY_FIRST, palette.BINARY_SECOND, palette.NEGATIVE], N=256)
CONFIDENCE_CMAP = LinearSegmentedColormap.from_list(
    'predicted_confidence', [palette.NEGATIVE, palette.BINARY_SECOND, palette.BINARY_FIRST], N=256)
DEVIATION_MAX = 10.0  # angstrom; deviations above this are drawn at the top colour
CONFIDENCE_RANGE = (50.0, 90.0)  # pLDDT; the published confidence bands
DEVIATION_LABEL = f'Cα deviation after superposition (Å, clipped at {DEVIATION_MAX:g})'
CONFIDENCE_LABEL = 'per-residue predicted confidence (ESMFold2 pLDDT)'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': BASE, 'axes.titlesize': BASE,
                     'axes.labelsize': BASE, 'xtick.labelsize': BASE, 'ytick.labelsize': BASE,
                     'legend.fontsize': BASE, 'axes.linewidth': .55, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'svg.fonttype': 'none', 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'xtick.major.size': 2, 'ytick.major.size': 2, 'axes.grid': False,
                     # Matplotlib otherwise stamps a wall-clock date into the PDF and salts SVG
                     # element ids at random, so two renders of identical data differ in bytes.
                     'svg.hashsalt': 'plm-capability-boundaries/render_structure_assets'})


# ---------------------------------------------------------------- retained-input loading


#: Points two label boxes may overlap before the labels are unreadable. The project's own
#: typography check uses the same tolerance.
TEXT_TOLERANCE = 0.25


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def declared_sums(path: Path) -> dict[str, str]:
    """A ``sha256sum``-format checksum file, as {relative path: digest}."""

    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            value, name = line.split(None, 1)
            out[name.strip()] = value
    return out


def require_digest(path: Path, expected: str, what: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f'{what}: digest of {path} is {actual}, the retained record declares {expected}')
    return actual


class Inputs:
    """Every retained artefact the assets read, each checked against its declared digest."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.paths: dict[str, dict[str, str]] = {}
        wild_type = repo / WILD_TYPE
        declared, _ = Path(str(wild_type) + '.sha256').read_text().split(None, 1)
        self.paths['wild_type_references'] = {
            'path': WILD_TYPE, 'sha256': require_digest(wild_type, declared, 'wild-type reference manifest')}
        self.wild_type = json.loads(wild_type.read_text())['domains']
        missing = sorted(set(SEALED + DEVELOPMENT) - set(self.wild_type))
        if missing:
            raise ValueError(f'the wild-type reference manifest lacks {missing}')

        sums = declared_sums(repo / CAMPAIGN / 'SHA256SUMS.txt')
        self.records: list[dict] = []
        for line in MEASURE_LINES:
            relative = f'pg3rb_{line}_measure_cpu/per_sequence.jsonl'
            path = repo / CAMPAIGN / relative
            if relative not in sums:
                raise ValueError(f'{relative} carries no digest in the campaign checksum file')
            self.paths[f'records_{line}'] = {
                'path': f'{CAMPAIGN}/{relative}',
                'sha256': require_digest(path, sums[relative], f'{line} per-candidate records')}
            for raw in path.read_text().splitlines():
                record = json.loads(raw)
                record['measure_line'] = line
                self.records.append(record)
        if not self.records:
            raise ValueError('the campaign carries no per-candidate record')

        analysis = repo / ANALYSIS
        self.paths['analysis'] = {'path': ANALYSIS, 'sha256': sha256_file(analysis)}
        self.analysis = json.loads(analysis.read_text())
        if self.analysis.get('schema') != 'progen3_robustness_bounds_v1':
            raise ValueError(f'{ANALYSIS}: unexpected schema {self.analysis.get("schema")!r}')
        self.paths['family_references'] = {'path': FAMILY_REFERENCES,
                                           'sha256': 'per-domain sidecars; each verified on load'}
        self._references: dict[str, np.ndarray] = {}
        self._confidence: dict[str, np.ndarray] = {}
        self._families: dict[str, object] = {}
        self._traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # -- wild-type references -------------------------------------------------

    def reference(self, domain: str) -> np.ndarray:
        """The CA trace of a wild-type reference, from the frozen, digest-checked PDB."""

        if domain not in self._references:
            entry = self.wild_type[domain]
            path = self.repo / Path(WILD_TYPE).parent / entry['pdb']
            require_digest(path, entry['pdb_sha256'], f'{domain} wild-type reference')
            trace = fg.parse_ca_trace(path.read_text())
            if trace.shape[0] != entry['length']:
                raise ValueError(f'{domain}: {trace.shape[0]} CA atoms against a declared length '
                                 f'{entry["length"]}')
            self._references[domain] = trace
        return self._references[domain]

    def reference_confidence(self, domain: str) -> np.ndarray:
        """Per-residue pLDDT of a wild-type reference, from the prediction object it was frozen from.

        The frozen PDB carries pLDDT only as a two-decimal B-factor, so the per-residue
        values come from the object's ``.npz``; the object's PDB is required to be the
        byte-identical source of the frozen copy before its confidences are used.
        """

        if domain not in self._confidence:
            entry = self.wild_type[domain]
            obj = self.repo / WILD_TYPE_FOLD / entry['source_object']
            require_digest(obj / 'prediction.pdb', entry['pdb_sha256'],
                           f'{domain} reference prediction object')
            with np.load(obj / 'prediction.npz') as data:
                values = np.asarray(data['ca_plddt_0_100'], dtype=np.float64)
            if values.shape[0] != entry['length']:
                raise ValueError(f'{domain}: {values.shape[0]} confidences against length {entry["length"]}')
            self._confidence[domain] = values
        return self._confidence[domain]

    # -- generated candidates -------------------------------------------------

    def candidate(self, record: dict) -> tuple[np.ndarray, np.ndarray]:
        """A candidate's CA trace and per-residue pLDDT, digest-checked on first read."""

        key = record['id']
        if key not in self._traces:
            obj = (self.repo / CAMPAIGN / f'pg3rb_{record["measure_line"]}_structures_cpu'
                   / record['structure_object'])
            pdb = obj / 'prediction.pdb'
            require_digest(pdb, record['structure_pdb_sha256'], f'{key} prediction')
            trace = fg.parse_ca_trace(pdb.read_text())
            with np.load(obj / 'prediction.npz') as data:
                confidence = np.asarray(data['ca_plddt_0_100'], dtype=np.float64)
            if trace.shape[0] != record['length'] or confidence.shape[0] != record['length']:
                raise ValueError(f'{key}: coordinates or confidences disagree with the declared length')
            self._traces[key] = (trace, confidence)
        return self._traces[key]

    def family(self, domain: str):
        if domain not in self._families:
            self._families[domain] = load_family_model(domain, self.repo / FAMILY_REFERENCES)
        return self._families[domain]

    def object_path(self, record: dict) -> str:
        return (f'{CAMPAIGN}/pg3rb_{record["measure_line"]}_structures_cpu/'
                f'{record["structure_object"]}')

    def cell(self, domain: str, model: str, arm: str) -> list[dict]:
        return [r for r in self.records
                if r['domain'] == domain and r['model'] == model and r['arm'] == arm]

    def admissible(self, rows) -> list[dict]:
        return [r for r in rows if r['identity_to_wild_type'] < ADMISSIBLE_IDENTITY]

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(sorted({r['model'] for r in self.records}))

    def achieved_bound(self, contrast: str, stratum: str) -> float:
        return float(self.analysis['achieved_upper_bounds']['primary_tm'][contrast][stratum])


# ---------------------------------------------------------------- geometry


def tm_superposition(mobile: np.ndarray, reference: np.ndarray) -> dict:
    """The rigid-body superposition that attains the TM-score, at the identity correspondence.

    A deliberate repetition of the search in ``fold_geometry.tm_score``, which returns the
    score but not the transform that produced it. Drawing one superposition while quoting
    another's score would misrepresent the endpoint, so the transform is recovered here and
    the caller is required to check the score against the retained measurement. Every
    constant and the iteration order are the frozen module's, read from it rather than
    copied, so the two cannot drift apart silently.
    """

    mobile = np.asarray(mobile, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if mobile.shape != reference.shape:
        raise ValueError(f'the identity correspondence needs two traces of one length; got '
                         f'{mobile.shape[0]} and {reference.shape[0]}')
    length = mobile.shape[0]
    d0 = fg.tm_d0(length)

    def fit(subset: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        rotation, translation = fg._superpose(mobile[subset], reference[subset])
        moved = fg._apply(rotation, translation, mobile)
        distances = np.linalg.norm(moved - reference, axis=1)
        return float(np.mean(1.0 / (1.0 + (distances / d0) ** 2))), distances, moved, subset

    best = fit(np.arange(length))
    seeds = []
    for divisor in fg.TM_SEED_DIVISORS:
        fragment = max(fg.TM_MIN_SEED, length // divisor)
        if fragment > length:
            continue
        stride = max(1, fragment // 2)
        for start in range(0, length - fragment + 1, stride):
            seeds.append(np.arange(start, start + fragment))
    for seed in seeds:
        subset = seed
        for _ in range(fg.TM_MAX_REFINEMENTS):
            attempt = fit(subset)
            if attempt[0] > best[0]:
                best = attempt
            cutoff = max(d0, 3.0)
            grown = np.flatnonzero(attempt[1] < cutoff)
            while grown.size < 3:
                cutoff += 0.5
                grown = np.flatnonzero(attempt[1] < cutoff)
            if grown.size == subset.size and np.array_equal(grown, subset):
                break
            subset = grown
    score, distances, moved, subset = best
    return {'tm_score': min(1.0, score), 'deviation_angstrom': distances, 'moved': moved,
            'd0_angstrom': d0, 'fitted_residues': int(subset.size)}


def superposed(inputs: Inputs, record: dict) -> dict:
    """A candidate placed on its reference, with the retained TM-score re-derived and checked."""

    reference = inputs.reference(record['domain'])
    trace, confidence = inputs.candidate(record)
    placed = tm_superposition(trace, reference)
    if placed['tm_score'] != record['tm_score']:
        raise ValueError(f'{record["id"]}: superposition scores {placed["tm_score"]!r} where the '
                         f'retained measurement is {record["tm_score"]!r}; the drawn superposition '
                         f'would not be the endpoint it is annotated with')
    placed['confidence'] = confidence
    placed['reference'] = reference
    return placed


def view_frame(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic right-handed viewing frame from the principal axes of a trace.

    The projection has to be reproducible and must not depend on the arbitrary orientation
    the folding run happened to emit, so it is the trace's own principal axes with the two
    sign ambiguities resolved by the chain itself: the first axis runs N to C, and the
    second is signed by the residue furthest from the first.
    """

    coordinates = np.asarray(coordinates, dtype=np.float64)
    centre = coordinates.mean(axis=0)
    _, _, basis = np.linalg.svd(coordinates - centre, full_matrices=False)
    first, second = basis[0], basis[1]
    if float((coordinates[-1] - coordinates[0]) @ first) < 0:
        first = -first
    projected = (coordinates - centre) @ second
    if projected[int(np.argmax(np.abs(projected)))] < 0:
        second = -second
    return centre, np.stack([first, second, np.cross(first, second)])


def project(coordinates: np.ndarray, frame: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    centre, basis = frame
    return (np.asarray(coordinates, dtype=np.float64) - centre) @ basis.T


# ---------------------------------------------------------------- selection


def selector_ranking(inputs: Inputs, rows, name: str) -> list[dict]:
    """The top ``SELECTION_K`` of an admissible pool under one scorer, worst-to-best order last.

    The scores come from the frozen selector registry, which refuses any structure-derived
    field, so this cannot become a structure-ranked selection by accident.
    """

    scores = e20.selector_scores([name], e20.selector_view(rows), inputs.family(rows[0]['domain']),
                                 PERMUTATION_SEED)[name]
    return [rows[i] for i in np.argsort(-scores, kind='stable')[:SELECTION_K]]


def oracle_ranking(rows) -> list[dict]:
    """The ceiling: the top ``SELECTION_K`` by the realised endpoint, which no method may read."""

    order = sorted(range(len(rows)), key=lambda i: (-rows[i]['tm_score'], rows[i]['id']))
    return [rows[i] for i in order[:SELECTION_K]]


def nearest(rows, target: float, field: str = 'tm_score') -> dict:
    """The candidate nearest a target value, ties broken by id so the pick is reproducible."""

    return min(rows, key=lambda r: (abs(r[field] - target), r['id']))


def best(rows, field: str = 'tm_score') -> dict:
    return max(rows, key=lambda r: (r[field], r['id']))


def measurements(record: dict) -> dict:
    """The retained per-candidate measurements a caption or the manifest may quote."""

    return {'id': record['id'], 'domain': record['domain'], 'model': record['model'],
            'measure_line': record['measure_line'], 'arm': record['arm'],
            'arm_label': ARM_LABELS[record['arm']], 'length': record['length'],
            'tm_score': record['tm_score'],
            'identity_to_wild_type_percent': record['identity_to_wild_type'],
            'admissible': record['identity_to_wild_type'] < ADMISSIBLE_IDENTITY,
            'predicted_confidence_plddt_mean': record['plddt_mean'], 'predicted_ptm': record['ptm'],
            'native_loglik': record['native_loglik'],
            'structure_pdb_sha256': record['structure_pdb_sha256']}


# ---------------------------------------------------------------- drawing primitives


def figure(height: float) -> plt.Figure:
    """A figure of the manuscript width and a height given in points."""

    return plt.figure(figsize=(WIDTH, height / 72))


def points(fig: plt.Figure) -> tuple[float, float]:
    width, height = fig.get_size_inches() * 72
    return float(width), float(height)


#: Panel shape. The first principal axis is the longest by construction, so a projected trace
#: is always wider than it is tall and a square panel would be a third whitespace.
PANEL_ASPECT = 1.35


def panel(fig: plt.Figure, x: float, y: float, width_pt: float) -> plt.Axes:
    """A frameless axes of the standard panel shape, placed by its bottom-left corner."""

    canvas_width, canvas_height = points(fig)
    height_pt = width_pt / PANEL_ASPECT
    axes = fig.add_axes([x / canvas_width, y / canvas_height,
                         width_pt / canvas_width, height_pt / canvas_height])
    axes.set_axis_off()
    axes.set_aspect('equal')
    return axes


def text(fig: plt.Figure, x: float, y: float, body: str, **kwargs) -> Text:
    width, height = points(fig)
    return fig.text(x / width, y / height, body, **kwargs)


def frame_limits(axes: plt.Axes, *projected: np.ndarray) -> None:
    """A data window of the panel shape around everything drawn: nothing cropped, nothing distorted."""

    stacked = np.concatenate([p[:, :2] for p in projected])
    low, high = stacked.min(axis=0), stacked.max(axis=0)
    centre, span = (high + low) / 2, high - low
    half_x = max(span[0], span[1] * PANEL_ASPECT) / 2 * 1.10 + 1.0
    half_y = half_x / PANEL_ASPECT
    axes.set_xlim(centre[0] - half_x, centre[0] + half_x)
    axes.set_ylim(centre[1] - half_y, centre[1] + half_y)


def trace(axes: plt.Axes, projected: np.ndarray, *, colour=None, values=None, cmap=None,
          norm=None, lw: float = 1.0, alpha: float = 1.0, zorder: float = 2) -> None:
    """One CA trace as a depth-sorted polyline, optionally coloured per residue.

    Segments are emitted far-to-near along the viewing axis so the near side of the fold
    draws on top, which is the only depth cue a flat projection has.
    """

    segments = np.stack([projected[:-1, :2], projected[1:, :2]], axis=1)
    depth = (projected[:-1, 2] + projected[1:, 2]) / 2
    order = np.argsort(depth, kind='stable')
    collection = LineCollection(segments[order], linewidths=lw, alpha=alpha, zorder=zorder,
                                capstyle='round', joinstyle='round')
    if values is not None:
        middles = (np.asarray(values)[:-1] + np.asarray(values)[1:]) / 2
        collection.set_array(middles[order])
        collection.set_cmap(cmap)
        collection.set_norm(norm)
    else:
        collection.set_color(colour)
    axes.add_collection(collection)


def ramp(fig: plt.Figure, x: float, y: float, width_pt: float, cmap, norm, label: str,
         ticks) -> None:
    """A thin horizontal colour ramp with its own caption, in point space."""

    width, height = points(fig)
    axes = fig.add_axes([x / width, y / height, width_pt / width, 4.5 / height])
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=axes, orientation='horizontal')
    bar.outline.set_linewidth(.4)
    bar.outline.set_edgecolor(BORDER)
    bar.set_ticks(list(ticks))
    axes.xaxis.set_ticks_position('top')
    axes.tick_params(labelsize=BASE, length=1.6, pad=1.0)
    text(fig, x + width_pt + 5, y + .5, label, va='baseline', fontsize=BASE)


def panel_label(fig: plt.Figure, x: float, y: float, lines, side: float) -> float:
    """Two or three short lines under a panel; returns the baseline of the lowest line."""

    for index, (body, weight) in enumerate(lines):
        text(fig, x, y - index * 8.6, body, va='baseline', fontsize=BASE, weight=weight)
    return y - (len(lines) - 1) * 8.6


def heading(fig: plt.Figure, x: float, y: float, letter: str, title: str) -> None:
    text(fig, x, y, letter, weight='bold', fontsize=LETTER, va='baseline', gid=f'panel-{letter}')
    text(fig, x + 11, y, title, va='baseline', fontsize=BASE)


LINE = 8.4  # point leading of wrapped caption text


def wrap(body: str, width_pt: float) -> list[str]:
    """Break a caption into lines that fit, so the breaks are part of the file and the
    height a caption needs is known before the figure is sized."""

    lines, current = [], ''
    for word in body.split():
        trial = f'{current} {word}'.strip()
        if len(trial) * BASE * .515 > width_pt and current:
            lines.append(current)
            current = word
        else:
            current = trial
    lines.append(current)
    return lines


def caption(fig: plt.Figure, x: float, y: float, lines, colour: str = SLATE) -> None:
    """Draw pre-wrapped caption lines downward from a top baseline."""

    for index, line in enumerate(lines):
        text(fig, x, y - index * LINE, line, va='baseline', fontsize=BASE, color=colour)


def footer_height(lines, ramp_label: bool = True) -> float:
    """Points the bottom block needs, so a figure can be sized before anything is drawn."""

    return (38.0 if ramp_label else 16.0) + (len(lines) - 1) * LINE


def footer(fig: plt.Figure, lines, cmap=None, norm=None, label: str = '', ticks=()) -> None:
    """The colour ramp and the caption, stacked at the bottom of the canvas."""

    top = 6.0 + (len(lines) - 1) * LINE
    caption(fig, MARGIN, top, lines)
    if cmap is not None:
        ramp(fig, MARGIN + 8.0, top + 14.0, 96.0, cmap, norm, label, ticks)


def reference_trace(axes: plt.Axes, projected: np.ndarray, lw: float = 2.6) -> None:
    trace(axes, projected, colour=REFERENCE_COLOUR, lw=lw, zorder=1)


# ---------------------------------------------------------------- assets


def render_reference_gallery(inputs: Inputs) -> dict:
    """The twelve wild-type references as a small multiple, for orientation."""

    order = tuple(SEALED) + tuple(DEVELOPMENT)
    side, gap, left = 85.0, 6.0, MARGIN
    label_block, group_gap = 20.0, 13.0
    rows = [order[0:4], order[4:8], order[8:12]]
    norm = Normalize(*CONFIDENCE_RANGE)
    lines = wrap(
        'Predicted structures of the wild-type sequences, not experimental structures and not '
        'measured folding. Colour is predicted confidence, which is not quality and not validation. '
        'Each view is the trace’s own principal axes, N to C left to right.', TEXT_WIDTH)
    height = 26 + len(rows) * (side / PANEL_ASPECT + label_block) + group_gap + 32 + footer_height(lines)
    fig = figure(height)
    top = height - 14
    heading(fig, left, top, 'a', 'Wild-type references, ESMFold2 predictions of the 12 domain sequences')
    cursor = top - 12
    panels = []
    for index, row in enumerate(rows):
        if index == 0:
            text(fig, left, cursor - 8, f'Sealed evaluation domains ({len(SEALED)})', va='baseline',
                 fontsize=BASE, color=SLATE)
            cursor -= 16
        if index == 2:
            cursor -= group_gap
            text(fig, left, cursor - 8, f'Development domains ({len(DEVELOPMENT)}); this campaign '
                 'generated no candidates for these', va='baseline', fontsize=BASE, color=SLATE)
            cursor -= 16
        for column, domain in enumerate(row):
            entry = inputs.wild_type[domain]
            coordinates = inputs.reference(domain)
            confidence = inputs.reference_confidence(domain)
            x = left + column * (side + gap)
            axes = panel(fig, x, cursor - side / PANEL_ASPECT, side)
            projected = project(coordinates, view_frame(coordinates))
            frame_limits(axes, projected)
            trace(axes, projected, values=confidence, cmap=CONFIDENCE_CMAP, norm=norm, lw=1.3)
            panel_label(fig, x, cursor - side / PANEL_ASPECT - 9,
                        [(f'{domain}  {entry["length"]} aa', 'bold'),
                         (f'pLDDT {entry["plddt_mean"]:.0f} · pTM {entry["ptm"]:.2f}', 'normal')],
                        side)
            panels.append({'panel': domain, 'domain': domain, 'length': entry['length'],
                           'set': 'sealed evaluation' if domain in SEALED else 'development',
                           'predicted_confidence_plddt_mean': entry['plddt_mean'],
                           'predicted_ptm': entry['ptm'], 'pdb': f'{Path(WILD_TYPE).parent}/{entry["pdb"]}',
                           'pdb_sha256': entry['pdb_sha256']})
        cursor -= side / PANEL_ASPECT + label_block
    footer(fig, lines, CONFIDENCE_CMAP, norm, CONFIDENCE_LABEL, [50, 70, 90])
    return {'figure': fig, 'panels': panels,
            'selection_rule': 'all 12 wild-type references, in sealed-then-development order; no selection',
            'colour': CONFIDENCE_LABEL}


def render_tm_ladder(inputs: Inputs) -> dict:
    """A TM-score ladder within one domain, so only the candidate changes between panels."""

    spans = {d: max(r['tm_score'] for r in inputs.admissible(
        [r for r in inputs.records if r['domain'] == d])) for d in SEALED}
    domain = max(spans, key=lambda d: (spans[d], d))
    bracketing = sorted(d for d in spans if spans[d] >= SAME_FOLD_TM)
    pool = [r for r in inputs.records if r['domain'] == domain]
    admissible = inputs.admissible(pool)
    targets = (0.15, 0.25, 0.35, 0.45, SAME_FOLD_TM)
    rungs = [('admissible', nearest(admissible, t)) for t in targets]
    rungs.append(('campaign maximum', best(pool)))

    columns, side, gap, left = 3, 112.0, 10.0, MARGIN
    label_block = 21.0
    lines = wrap(
        f'Reference in grey. {domain} is the sealed domain whose admissible pool reaches the highest '
        f'TM-score, so this is the widest range available, not a typical one: only {len(bracketing)} '
        f'of {len(SEALED)} sealed domains have any admissible candidate at or above the '
        f'{SAME_FOLD_TM:g} same-fold threshold. The last panel is the campaign maximum over all arms '
        f'and is a near-copy of the wild-type sequence, which the identity restriction excludes from '
        f'the endpoint.', TEXT_WIDTH)
    height = 28 + 2 * (side / PANEL_ASPECT + label_block) + footer_height(lines)
    fig = figure(height)
    norm = Normalize(0, DEVIATION_MAX)
    top = height - 14
    heading(fig, left, top, 'b', f'TM-score ladder on {domain}, candidate against wild-type reference')
    cursor = top - 14
    panels = []
    for index, (rung, record) in enumerate(rungs):
        placed = superposed(inputs, record)
        frame = view_frame(placed['reference'])
        reference = project(placed['reference'], frame)
        candidate = project(placed['moved'], frame)
        x = left + (index % columns) * (side + gap)
        y = cursor - (index // columns) * (side / PANEL_ASPECT + label_block) - side / PANEL_ASPECT
        axes = panel(fig, x, y, side)
        frame_limits(axes, reference, candidate)
        reference_trace(axes, reference)
        trace(axes, candidate, values=placed['deviation_angstrom'], cmap=DEVIATION_CMAP, norm=norm,
              lw=1.4, zorder=3)
        tag = ('identity %.0f%%, admissible' % record['identity_to_wild_type'] if rung == 'admissible'
               else 'identity %.0f%%, NOT admissible' % record['identity_to_wild_type'])
        panel_label(fig, x, y - 9,
                    [(f'TM {record["tm_score"]:.3f}', 'bold'),
                     (tag, 'normal')], side)
        panels.append({'panel': f'tm-{record["tm_score"]:.3f}', 'rung': rung,
                       **measurements(record),
                       'source_object': inputs.object_path(record)})
    footer(fig, lines, DEVIATION_CMAP, norm, DEVIATION_LABEL, [0, 5, 10])
    return {'figure': fig, 'panels': panels,
            'selection_rule': (f'domain = the sealed evaluation domain with the highest TM-score in its '
                               f'admissible pool ({domain}; domains at or above {SAME_FOLD_TM:g}: '
                               f'{", ".join(bracketing)}). Rungs 1-5 = the admissible candidate nearest '
                               f'each target TM {targets}, ties by id. Rung 6 = the highest TM-score in '
                               f'the whole campaign, drawn to show what reaching the threshold costs in '
                               f'novelty; it is not admissible.'),
            'colour': DEVIATION_LABEL}


def render_tm_effect_scale(inputs: Inputs) -> dict:
    """What the manuscript's TM effects look like: the declared margin and the achieved bound."""

    admissible = inputs.admissible(inputs.records)
    pooled_median = float(np.median([r['tm_score'] for r in admissible]))
    medians = {d: float(np.median([r['tm_score'] for r in admissible if r['domain'] == d]))
               for d in SEALED}
    domain = min(medians, key=lambda d: (abs(medians[d] - pooled_median), d))
    pool = [r for r in admissible if r['domain'] == domain]
    bound = inputs.achieved_bound('C-Pr', 'pooled_bfloat16')
    base = nearest(pool, medians[domain])
    steps = [('pool median', 0.0, base),
             ('achieved bound on C−Pr', bound, nearest(pool, base['tm_score'] + bound)),
             ('declared equivalence margin', MARGIN_TM, nearest(pool, base['tm_score'] + MARGIN_TM))]

    side, gap, left = 112.0, 10.0, MARGIN
    label_block = 30.0
    lines = wrap(
        f'{domain} is the sealed domain whose admissible median TM-score sits closest to the pooled '
        f'admissible median, so it is the most ordinary domain available. The campaign’s achieved '
        f'upper bound on the headline coupling-over-profile contrast is {bound:.4f} TM and every '
        f'other arm contrast is bounded below {MARGIN_TM:g} TM. Panels 2 and 3 are those two '
        f'differences drawn against panel 1: the effects the manuscript reports are smaller than '
        f'anything a structure picture can resolve, which is the point of showing them.', TEXT_WIDTH)
    height = 28 + side / PANEL_ASPECT + label_block + footer_height(lines)
    fig = figure(height)
    norm = Normalize(0, DEVIATION_MAX)
    top = height - 14
    heading(fig, left, top, 'c', f'The scale of the reported TM effects, on {domain}')
    cursor = top - 14
    panels = []
    for index, (what, delta, record) in enumerate(steps):
        placed = superposed(inputs, record)
        frame = view_frame(placed['reference'])
        reference = project(placed['reference'], frame)
        candidate = project(placed['moved'], frame)
        x = left + index * (side + gap)
        y = cursor - side / PANEL_ASPECT
        axes = panel(fig, x, y, side)
        frame_limits(axes, reference, candidate)
        reference_trace(axes, reference)
        trace(axes, candidate, values=placed['deviation_angstrom'], cmap=DEVIATION_CMAP, norm=norm,
              lw=1.4, zorder=3)
        panel_label(fig, x, y - 9,
                    [(f'TM {record["tm_score"]:.4f}', 'bold'),
                     (what, 'normal'),
                     ('' if index == 0 else f'Δ TM {delta:+.4f} on panel 1', 'normal')], side)
        panels.append({'panel': what, 'delta_tm_from_first_panel': delta, **measurements(record),
                       'source_object': inputs.object_path(record)})
    footer(fig, lines, DEVIATION_CMAP, norm, DEVIATION_LABEL, [0, 5, 10])
    return {'figure': fig, 'panels': panels,
            'selection_rule': (f'domain = the sealed evaluation domain whose admissible median TM-score is '
                               f'nearest the pooled admissible median ({domain}, median '
                               f'{medians[domain]:.4f} against {pooled_median:.4f}). Panel 1 = the '
                               f'admissible candidate nearest that median; panels 2 and 3 = the '
                               f'admissible candidates nearest panel 1 plus the achieved upper bound on '
                               f'C-Pr ({bound:.4f}) and plus the declared margin ({MARGIN_TM:g}). Ties by id.'),
            'colour': DEVIATION_LABEL}


def render_admissible_ceiling(inputs: Inputs) -> dict:
    """The best novel candidate found for each sealed domain: the honest per-domain ceiling."""

    chosen = {domain: best(inputs.admissible([r for r in inputs.records if r['domain'] == domain]))
              for domain in SEALED}
    below = sum(record['tm_score'] < SAME_FOLD_TM for record in chosen.values())

    columns, side, gap, left = 4, 85.0, 6.0, MARGIN
    label_block = 21.0
    lines = wrap(
        f'The single highest-TM candidate with identity below {ADMISSIBLE_IDENTITY:g}% in each '
        f'domain’s whole retained pool, over every arm and both measure lines: a best case by '
        f'construction, and the arm label is where the best one happened to fall, not a result. '
        f'{below} of {len(SEALED)} sit below the {SAME_FOLD_TM:g} same-fold threshold.', TEXT_WIDTH)
    height = 28 + 2 * (side / PANEL_ASPECT + label_block) + footer_height(lines)
    fig = figure(height)
    norm = Normalize(0, DEVIATION_MAX)
    top = height - 14
    heading(fig, left, top, 'd', 'Best admissible candidate per sealed domain')
    cursor = top - 14
    panels = []
    for index, domain in enumerate(SEALED):
        record = chosen[domain]
        placed = superposed(inputs, record)
        frame = view_frame(placed['reference'])
        reference = project(placed['reference'], frame)
        candidate = project(placed['moved'], frame)
        x = left + (index % columns) * (side + gap)
        y = cursor - (index // columns) * (side / PANEL_ASPECT + label_block) - side / PANEL_ASPECT
        axes = panel(fig, x, y, side)
        frame_limits(axes, reference, candidate)
        reference_trace(axes, reference)
        trace(axes, candidate, values=placed['deviation_angstrom'], cmap=DEVIATION_CMAP, norm=norm,
              lw=1.3, zorder=3)
        panel_label(fig, x, y - 9,
                    [(f'{domain}  TM {record["tm_score"]:.3f}', 'bold'),
                     (ARM_SHORT[record['arm']], 'normal')], side)
        panels.append({'panel': domain, **measurements(record), 'pool': 'all arms, all measure lines',
                       'source_object': inputs.object_path(record)})
    footer(fig, lines, DEVIATION_CMAP, norm, DEVIATION_LABEL, [0, 5, 10])
    return {'figure': fig, 'panels': panels,
            'selection_rule': (f'per sealed domain, the candidate with the highest TM-score among those '
                               f'with identity below {ADMISSIBLE_IDENTITY:g}%, over all arms, models and '
                               f'measure lines; ties by id. A declared extremum, so a best case and not '
                               f'a representative one.'),
            'colour': DEVIATION_LABEL}


def selection_cells(inputs: Inputs) -> tuple[list[dict], dict]:
    """The two arm-N cells where native likelihood does best and worst against the ceiling.

    Showing only the failure would be a chosen anecdote, so the rule takes both extremes of
    the same declared sweep and the figure says which is which.
    """

    sweep = []
    for domain in SEALED:
        for model in inputs.models:
            rows = inputs.admissible(inputs.cell(domain, model, SELECTION_POOL_ARM))
            if len(rows) < e20.MIN_ADMISSIBLE:
                continue
            headroom = e20.tm_oracle_gain(rows)
            if headroom <= 0:
                continue
            pool_mean = float(np.mean([r['tm_score'] for r in rows]))
            picks = {name: selector_ranking(inputs, rows, name) for name, _ in SCORERS}
            gains = {name: float(np.mean([r['tm_score'] for r in picks[name]])) - pool_mean
                     for name in picks}
            sweep.append({'domain': domain, 'model': model, 'rows': rows, 'picks': picks,
                          'pool_mean': pool_mean, 'headroom': headroom, 'gains': gains,
                          'captured': {n: gains[n] / headroom for n in gains}})
    if len(sweep) < 2:
        raise ValueError('fewer than two arm-N cells carry an admissible pool and positive headroom')
    key = 'native_loglik'
    ordered = sorted(sweep, key=lambda c: (c['captured'][key], c['domain'], c['model']))
    return [ordered[0], ordered[-1]], {'n_cells': len(sweep), 'ranked_on': key}


def render_selection_failure(inputs: Inputs) -> dict:
    """What an admissible scorer retrieves at k = 10, against what the pool actually contains."""

    cells, sweep = selection_cells(inputs)
    roles = [(name, label, colour) for (name, label), colour in
             zip(SCORERS, (palette.BINARY_SECOND, palette.BINARY_FIRST))]
    roles.append((None, ORACLE, palette.CONTROL))

    side, gap, left = 112.0, 10.0, MARGIN
    strip_height, block_label, cell_gap = 40.0, 30.0, 18.0
    lines = wrap(
        f'Each structure panel overlays the {SELECTION_K} candidates one ranking retrieves from the '
        f'same pool, on the reference in grey. The oracle ranks by the realised TM-score, which no '
        f'method may read; it measures the ceiling and is not a proposed selector. The pools contain '
        f'candidates neither admissible scorer retrieves, and in the upper cell native likelihood '
        f'retrieves candidates worse than the pool average.', TEXT_WIDTH)
    block = 13 + side / PANEL_ASPECT + block_label + strip_height + 20
    height = 26 + 2 * block + cell_gap + footer_height(lines, ramp_label=False)
    fig = figure(height)
    top = height - 14
    heading(fig, left, top, 'e', f'Top {SELECTION_K} of an unguided pool: available scorers against the ceiling')
    cursor = top - 16
    panels = []
    for order, cell in enumerate(cells):
        extreme = 'worst' if order == 0 else 'best'
        rows, pool_mean, headroom = cell['rows'], cell['pool_mean'], cell['headroom']
        selections = {label: (cell['picks'][name] if name else oracle_ranking(rows))
                      for name, label, _ in roles}
        text(fig, left, cursor - 8,
             f'{cell["domain"]} · {cell["model"]} · {ARM_LABELS[SELECTION_POOL_ARM]} pool, '
             f'n = {len(rows)} admissible · {extreme} of {sweep["n_cells"]} cells',
             va='baseline', fontsize=BASE, weight='bold')
        cursor -= 13
        for index, (name, label, colour) in enumerate(roles):
            picked = selections[label]
            x = left + index * (side + gap)
            y = cursor - side / PANEL_ASPECT
            axes = panel(fig, x, y, side)
            reference = inputs.reference(cell['domain'])
            frame = view_frame(reference)
            projected_reference = project(reference, frame)
            placed = [project(superposed(inputs, record)['moved'], frame) for record in picked]
            frame_limits(axes, projected_reference, *placed)
            reference_trace(axes, projected_reference, lw=3.6)
            for drawn in placed:
                trace(axes, drawn, colour=colour, lw=.45, alpha=.6, zorder=3)
            mean_tm = float(np.mean([r['tm_score'] for r in picked]))
            gain = mean_tm - pool_mean
            panel_label(fig, x, y - 9,
                        [(label if name else f'{label} (ceiling)', 'bold'),
                         (f'mean TM {mean_tm:.3f} · {gain:+.3f} on pool', 'normal'),
                         ('' if name is None else f'{gain / headroom:+.2f} of the ceiling', 'normal')],
                        side)
            panels.append({'panel': f'{cell["domain"]}/{cell["model"]}/{label}',
                           'domain': cell['domain'], 'model': cell['model'],
                           'measure_line': picked[0]['measure_line'], 'arm': SELECTION_POOL_ARM,
                           'cell_extreme': extreme, 'scorer': label,
                           'scorer_is_admissible': name is not None,
                           'k': SELECTION_K, 'pool_size_admissible': len(rows),
                           'pool_mean_tm': pool_mean, 'selected_mean_tm': mean_tm,
                           'gain_on_pool_mean_tm': gain, 'oracle_headroom_tm': headroom,
                           'captured_fraction_of_oracle_headroom': None if name is None else gain / headroom,
                           'candidates': [measurements(r) for r in picked],
                           'source_objects': [inputs.object_path(r) for r in picked]})
        cursor -= side / PANEL_ASPECT + block_label

        # The strip is inset from the left so its row labels fit on the canvas, and stops
        # short of the right margin so the last tick label does, too.
        inset, trail = 62.0, 12.0
        width_pt = 3 * side + 2 * gap - inset - trail
        canvas_width, canvas_height = points(fig)
        axes = fig.add_axes([(left + inset) / canvas_width, (cursor - strip_height) / canvas_height,
                             width_pt / canvas_width, strip_height / canvas_height])
        values = np.array([r['tm_score'] for r in rows])
        axes.plot(values, np.full(values.shape, len(roles)), '|', color=BORDER, ms=4, mew=.7,
                  zorder=1)
        for index, (name, label, colour) in enumerate(roles):
            picked = selections[label]
            axes.plot([r['tm_score'] for r in picked], np.full(len(picked), len(roles) - 1 - index),
                      '|', color=colour, ms=5, mew=.9, zorder=3)
        axes.axvline(pool_mean, color=palette.CONTROL, lw=.55, ls=':', zorder=2)
        # Set after plotting: every plot call re-enables autoscaling, which would widen the
        # view back to the data edge and push the last tick label off the canvas.
        span = float(values.max() - values.min())
        axes.set_xlim(values.min() - .06 * span, values.max() + .06 * span)
        axes.set_yticks(range(len(roles) + 1), [r[1] for r in reversed(roles)] + ['pool'])
        axes.set_ylim(-.7, len(roles) + .7)
        axes.tick_params(axis='y', length=0, pad=2)
        axes.tick_params(axis='x', length=1.6, pad=1.4)
        axes.spines[['top', 'right', 'left']].set_visible(False)
        axes.spines['bottom'].set_color(BORDER)
        axes.set_xlabel('TM-score of the candidate (dotted line: pool mean)', labelpad=1.5)
        cursor -= strip_height + 20 + (cell_gap if order == 0 else 0)
    footer(fig, lines)
    return {'figure': fig, 'panels': panels,
            'selection_rule': (f'over the {sweep["n_cells"]} (domain, checkpoint) cells of the '
                               f'{ARM_LABELS[SELECTION_POOL_ARM]} arm with an admissible pool of at least '
                               f'{e20.MIN_ADMISSIBLE} and positive oracle headroom, the two cells where '
                               f'native likelihood captures the smallest and the largest fraction of that '
                               f'headroom at k = {SELECTION_K}. Both extremes are drawn so the failure is '
                               f'not a chosen anecdote. Within a cell the selection is the frozen '
                               f'registry’s top k, and the oracle is the top k by realised TM-score.'),
            'colour': 'one colour per ranking; the reference is grey'}


ASSETS = (('reference-gallery', render_reference_gallery),
          ('tm-ladder', render_tm_ladder),
          ('tm-effect-scale', render_tm_effect_scale),
          ('admissible-ceiling', render_admissible_ceiling),
          ('selection-failure', render_selection_failure))


# ---------------------------------------------------------------- saving


def save(fig: plt.Figure, name: str, out: Path) -> dict:
    """Write the PNG and PDF of one asset, refusing text that falls outside the canvas."""

    out.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    drawn = [(t.get_text(), t.get_window_extent(renderer)) for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip()]
    clipped = [body for body, box in drawn
               if not (bounds.x0 - 1 <= box.x0 and box.x1 <= bounds.x1 + 1
                       and bounds.y0 - 1 <= box.y0 and box.y1 <= bounds.y1 + 1)]
    if clipped:
        raise ValueError(f'{name}: text outside the canvas: {clipped}')
    collisions = [(a, b) for index, (a, first) in enumerate(drawn) for b, second in drawn[index + 1:]
                  if (min(first.x1, second.x1) - max(first.x0, second.x0) > TEXT_TOLERANCE
                      and min(first.y1, second.y1) - max(first.y0, second.y0) > TEXT_TOLERANCE)]
    if collisions:
        raise ValueError(f'{name}: overlapping labels: {collisions}')
    files = {}
    for kind, metadata, options in (('png', None, {'dpi': 400}),
                                    ('pdf', {'CreationDate': None}, {})):
        path = out / f'{name}.{kind}'
        fig.savefig(path, metadata=metadata, **options)
        files[kind] = {'file': path.name, 'sha256': digest(path), 'bytes': path.stat().st_size}
    plt.close(fig)
    return files


def render(out: Path, repo: Path) -> dict:
    """Render every asset into ``out`` and return the manifest."""

    inputs = Inputs(repo)
    assets = {}
    for name, function in ASSETS:
        drawn = function(inputs)
        figure_object = drawn.pop('figure')
        assets[name] = {'files': save(figure_object, name, out), **drawn}
    bounds = inputs.analysis['achieved_upper_bounds']['primary_tm']
    return {
        'schema': SCHEMA,
        'statement': STATEMENT,
        'generated_by': ENTRY_POINT,
        'command': COMMAND,
        'campaign': {
            'candidates_from': 'ProGen3 cross-model robustness campaign',
            'path': CAMPAIGN,
            'checkpoints': list(inputs.models),
            'domains_with_candidates': list(SEALED),
            'primary_campaign_coordinates': (
                'The primary E20 confirmatory campaign (progen2-medium, protgpt2) retained '
                'per-candidate measurements only and no structure coordinates. No candidate of a '
                'primary model is or can be drawn here.'),
        },
        'structures_are': ('ESMFold2 computational predictions, references included. Not measured '
                           'folding, stability or function; no generated sequence in this study has an '
                           'experimental endpoint.'),
        'correspondence': CORRESPONDENCE,
        'oracle_statement': ORACLE_STATEMENT,
        'forbidden_selector_fields': sorted(e20.FORBIDDEN_SELECTOR_FIELDS),
        'reference_scales': {
            'same_fold_tm': SAME_FOLD_TM,
            'declared_equivalence_margin_tm': MARGIN_TM,
            'admissible_identity_percent_strictly_below': ADMISSIBLE_IDENTITY,
            'selection_budget_k': SELECTION_K,
            'achieved_upper_bounds_primary_tm_pooled_bfloat16': {
                contrast: bounds[contrast]['pooled_bfloat16'] for contrast in sorted(bounds)},
        },
        'inputs': inputs.paths,
        'assets': assets,
    }


def verify(out: Path, repo: Path, manifest: dict) -> dict:
    """Re-render into a scratch directory and require every file to be byte-identical."""

    scratch = Path(tempfile.mkdtemp(prefix='structure-assets-verify-'))
    try:
        again = render(scratch, repo)
        differing = []
        for name, asset in manifest['assets'].items():
            for kind, entry in asset['files'].items():
                other = again['assets'][name]['files'][kind]
                if entry['sha256'] != other['sha256']:
                    differing.append(f'{name}.{kind}')
        if differing:
            raise ValueError(f'the render is not reproducible; these files differ between two renders '
                             f'of the same inputs: {differing}')
        return {'checked': sorted(f'{n}.{k}' for n, a in manifest['assets'].items()
                                  for k in a['files']),
                'byte_identical': True}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=REPO / OUT_DIRECTORY,
                        help=f'output directory (default {OUT_DIRECTORY})')
    parser.add_argument('--verify', action='store_true',
                        help='render a second time into a scratch directory and require identical bytes')
    args = parser.parse_args(argv)
    manifest = render(args.out, REPO)
    if args.verify:
        manifest['determinism_check'] = verify(args.out, REPO, manifest)
    write_json(args.out / 'manifest.json', manifest)
    for name, asset in manifest['assets'].items():
        sizes = ', '.join(f'{kind} {entry["bytes"] / 1024:.0f} kB' for kind, entry in asset['files'].items())
        print(f'{name}: {len(asset["panels"])} panels, {sizes}')
    if args.verify:
        print(f'determinism: {len(manifest["determinism_check"]["checked"])} files byte-identical '
              f'across two renders')
    print(f'manifest: {args.out / "manifest.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
