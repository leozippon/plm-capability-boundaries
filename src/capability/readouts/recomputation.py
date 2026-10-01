"""Refit a gate's representation cells at a selected readout class and depth.

Every representation-level verdict in the capability map is provisional until the
readout reassessment completes, after which the representation analyses at every
gate are revisited and recomputed. This module is that recomputation: given the
reassessment's selected readout class and depth per arm, it refits a published
cell on **identical** support, folds, seeds, weighting and label budget, and
reports the recomputed increment beside the published one with both intervals
and the resampling unit.

Three properties make the refit a recomputation of the published quantity rather
than a new measurement, and each is enforced rather than assumed.

**Identity, or a refusal.** :func:`refuse_on_drift` compares the realised cell
against the published one on the ordered support, the unit and variant counts,
the split seed, every outer fold's held-unit membership, every inner fold's
membership, the resampling contract and the label budget. Any difference raises:
an increment measured on a support the published one did not use is not a
recomputation of it, and reporting the two side by side would invite exactly
that reading.

**A pinned reduction order.** The compressed representation block is a float32
matrix product, and a float32 product is not invariant to how the BLAS library
splits its reduction. The admitted analysis formed it under four threads; at
forty-eight the held-out predictions moved by up to 9.085e-07 against a 1e-8
gate while the endpoints stayed bit-identical. The thread count is therefore part
of pipeline identity, is pinned through
:func:`~src.capability.readouts.readout_class_sweep.projected_block`, and a process with no
discoverable BLAS pool is refused rather than run unpinned.

**Rows blocked by assay.** The admitted analysis forms that product one assay at
a time. Forming it over a whole panel departs by up to 7.629e-06 in absolute
value at every thread count, which reached 3.483e-08 in the held-out predictions,
above the same gate. The product is therefore blocked by assay through
:func:`~src.capability.readouts.readout_depth.admitted_design`, and
:func:`product_blocking_departure` measures the departure on the arrays at hand
so the constraint is a number in the report rather than a claim in a comment.

The module introduces no resampler: every interval it reports comes from the
resampler the published cell used, at that cell's own draws, seed and unit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from ..core.gate_retention import GATE_RETENTION, SELECTION_KINDS
from .readout_class_sweep import (CLASSES, DEPTH_CLASSES, PROJECTION_BLAS_THREADS,
                                  REPRODUCTION_CLASS, blas_thread_counts)
from .readout_depth import SUMMARY_NAMES, admitted_design, gaussian_projection

#: Declared depth axes of the readout reassessment. ``admitted`` is the pair of
#: blocks the admitted extraction hooked, which every gate's own extraction also
#: hooked; the others exist only where a depth-resolved extraction was written.
DEPTH_AXES = ('admitted', 'depth', 'wide', 'union', 'pos', 'full')

#: Axes that name one block of the stack and therefore require a depth index.
INDEXED_DEPTH_AXES = ('depth', 'wide', 'pos', 'full')

#: Schema tag of a recomputation plan and of a cell report.
PLAN_SCHEMA = 'd1_recomputation_plan_v1'
CELL_SCHEMA = 'd1_recomputation_cell_v1'

#: The admitted five-arm admission receipt, whose ``baseline_tolerance`` is the
#: tolerance a pipeline-identity replay of the admitted selection is bound at.
#: Read from the receipt rather than restated, so the number cannot drift from
#: the artefact that established it.
ADMITTED_RECEIPT = Path('results/R2/readout_20260923/final_admission.json')


class RecomputationRefused(Exception):
    """A refusal: the recomputation cannot be a recomputation of the published cell."""


def admitted_prediction_tolerance(receipt: Path | str = ADMITTED_RECEIPT) -> float:
    """The absolute tolerance the admitted final-admission receipt bound its baseline at.

    Refused rather than defaulted when the receipt is unreadable. A replay that
    invented its own tolerance would certify nothing, and a tolerance is exactly
    the quantity a silent fallback corrupts.
    """
    path = Path(receipt)
    if not path.is_file():
        raise RecomputationRefused(
            f'the admitted admission receipt is not at {path}; the pipeline-identity tolerance '
            'is a property of that receipt and is not supplied from anywhere else')
    record = json.loads(path.read_text())
    try:
        atol = float(record['baseline_tolerance']['atol'])
    except (KeyError, TypeError, ValueError) as error:
        raise RecomputationRefused(
            f'{path} carries no baseline_tolerance.atol: {error}') from error
    if not np.isfinite(atol) or atol <= 0:
        raise RecomputationRefused(f'{path} declares a non-positive tolerance {atol!r}')
    return atol


@dataclass(frozen=True)
class ReadoutSelection:
    """One cell's selected readout class and depth, as the reassessment returns it.

    Keyed by arm **and panel**, because the published selection is: ProGen3-3B
    selects block 11 on the anchor panel and block 23 on the EC-conditioned one,
    and ProtGPT2 selects block 21 on two panels. A selection keyed by arm alone
    could not express that, and would have silently applied one panel's depth to
    another's cells.

    ``depth_axis`` is ``admitted`` when the selection stays at the two blocks the
    admitted extraction hooked, and one of the depth sweep's own axes otherwise.
    :attr:`selection_kind` maps that onto the retention requirement the inventory
    declares, which is what decides whether a gate can serve the selection at all.
    """

    arm: str
    panel: str = ''
    class_name: str = REPRODUCTION_CLASS
    depth_axis: str = 'admitted'
    depth_index: int | None = None
    coordinates: int | None = None
    projection_seed: int | None = None
    blas_threads: int = PROJECTION_BLAS_THREADS

    def __post_init__(self):
        if not self.arm:
            raise ValueError('a selection names one arm')
        if self.class_name not in CLASSES and self.class_name not in DEPTH_CLASSES:
            raise ValueError(f'{self.class_name!r} is not a declared readout class')
        if self.depth_axis not in DEPTH_AXES:
            raise ValueError(f'{self.depth_axis!r} is not a declared depth axis')
        indexed = self.depth_axis in INDEXED_DEPTH_AXES
        if indexed and self.depth_index is None:
            raise ValueError(f'the {self.depth_axis} axis names one block and needs a depth index')
        if not indexed and self.depth_index is not None:
            raise ValueError(f'the {self.depth_axis} axis takes no depth index')
        if self.depth_index is not None and self.depth_index < 0:
            raise ValueError('a zero-based depth index is non-negative')
        if self.blas_threads < 1:
            raise ValueError('a pinned BLAS thread count is positive')

    @property
    def is_admitted_pipeline(self) -> bool:
        """True when this selection is the admitted class at the admitted depths.

        A cell at this selection is the pipeline-identity control rather than a
        recomputation: it must reproduce the published held-out predictions inside
        the admitted tolerance, and a cell that does not raises.
        """
        return self.depth_axis == 'admitted' and self.class_name == REPRODUCTION_CLASS

    @property
    def selection_kind(self) -> str:
        return 'readout_class' if self.depth_axis == 'admitted' else 'extraction_depth'

    @property
    def key(self) -> tuple[str, str]:
        return (self.arm, self.panel)

    def as_dict(self) -> dict:
        return dict(arm=self.arm, panel=self.panel, class_name=self.class_name,
                    depth_axis=self.depth_axis,
                    depth_index=self.depth_index, coordinates=self.coordinates,
                    projection_seed=self.projection_seed, blas_threads=self.blas_threads,
                    selection_kind=self.selection_kind,
                    is_admitted_pipeline=self.is_admitted_pipeline)


def load_selection(path: Path | str) -> dict[tuple[str, str], ReadoutSelection]:
    """The reassessment's per-cell selection, read from its declaration.

    Refused when the declaration is absent, because the readout reassessment's
    results are what the selection comes from: a recomputation run against a
    default selection would report a recomputed increment that no reassessment
    chose. Cells are keyed ``arm|panel``, and a bare arm key is refused rather
    than spread over that arm's panels, because the published selection differs
    by panel for at least one arm and guessing which panel a bare key meant is
    exactly the substitution this driver exists to prevent.
    """
    path = Path(path)
    if not path.is_file():
        raise RecomputationRefused(
            f'no readout selection at {path}; the recomputation runs only against the '
            'reassessment\'s own declared class and depth per cell')
    record = json.loads(path.read_text())
    entries = record.get('cells')
    if not isinstance(entries, dict) or not entries:
        raise RecomputationRefused(f'{path} declares no per-cell selection under "cells"')
    selection = {}
    for key, fields in entries.items():
        if '|' not in key:
            raise RecomputationRefused(
                f'{path}: selection key {key!r} names no panel. The published selection differs '
                'by panel for at least one arm, so a key is "arm|panel"')
        arm, panel = key.split('|', 1)
        if not isinstance(fields, dict):
            raise RecomputationRefused(f'{path}: {key} does not declare a selection record')
        try:
            selection[(arm, panel)] = ReadoutSelection(arm=arm, panel=panel, **fields)
        except (TypeError, ValueError) as error:
            raise RecomputationRefused(f'{path}: {key} declares an unusable selection: {error}') \
                from error
    return selection


# --------------------------------------------------------------- cell identity


@dataclass(frozen=True)
class CellIdentity:
    """Everything a recomputation must hold identical to the published cell.

    The support is ordered because the fitted rows are concatenated in assay
    order, so two runs on the same set in a different order are not the same
    rows. Fold membership is carried per outer fold and per inner fold, because
    the seed alone does not certify the partition: a different unit set under the
    same seed yields a different partition.
    """

    support: tuple[str, ...]
    units: int
    variants: int
    split_seed: int
    outer_folds: tuple[tuple[int, ...], ...]
    inner_folds: tuple[tuple[tuple[int, ...], ...], ...]
    unit_label: str
    resample_draws: int
    resample_seed: int

    def digest(self) -> str:
        payload = dict(support=list(self.support), units=self.units, variants=self.variants,
                       split_seed=self.split_seed,
                       outer_folds=[list(fold) for fold in self.outer_folds],
                       inner_folds=[[list(inner) for inner in folds]
                                    for folds in self.inner_folds],
                       unit_label=self.unit_label, resample_draws=self.resample_draws,
                       resample_seed=self.resample_seed)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def as_dict(self) -> dict:
        return dict(support_assays=len(self.support), units=self.units, variants=self.variants,
                    split_seed=self.split_seed, unit=self.unit_label,
                    resample_draws=self.resample_draws, resample_seed=self.resample_seed,
                    outer_folds=len(self.outer_folds),
                    inner_folds_per_outer=[len(folds) for folds in self.inner_folds],
                    identity_sha256=self.digest())


def _plain(value):
    """A NumPy scalar rendered as the Python value it holds.

    An identity digest must not depend on which NumPy version rendered its
    inputs. ``repr(np.float64(0.5))`` is ``0.5`` under NumPy 1.26 and
    ``np.float64(0.5)`` under NumPy 2, and ``json.dumps`` refuses a NumPy integer
    outright, so a fold label or an assay identifier that arrives as a NumPy
    scalar would either hash differently across interpreters or fail to serialise
    at all. The same defect was measured on the pairwise row-identity digest,
    where the same 8,192 rows hashed to two different values across the two
    NumPy versions in use.
    """
    return value.item() if isinstance(value, np.generic) else value


def _folds(records, key: str) -> tuple[tuple, tuple]:
    outer, inner = [], []
    for record in records:
        outer.append(tuple(sorted(_plain(label) for label in record[key])))
        inner.append(tuple(sorted(tuple(sorted(_plain(label)
                                               for label in fold['validation_families']))
                                  for fold in record['inner_folds'])))
    return tuple(outer), tuple(inner)


def identity_from_readout_report(report: dict, *, design: str = 'B',
                                 metric: str = 'delta_spearman') -> CellIdentity:
    """The identity of a readout-family cell, read from its own report.

    Works on the admitted analysis reports and on the reports this module writes,
    because both carry the same fold, support, seed and summary structure. The
    resampling contract is read from the summary of one metric rather than
    restated: draws, alpha and the unit are properties of the published interval.
    """
    support = report.get('support', {}).get('assay_ids')
    if support is None:
        support = [row['assay'] for row in report['assays']]
    summary = report['summaries'][metric]
    outer, inner = _folds(report['folds'][design], 'held_families')
    support = tuple(str(_plain(assay)) for assay in support)
    return CellIdentity(support=support, units=int(report['n_families']),
                        variants=int(report['n_variants']), split_seed=int(report['fold_seed']),
                        outer_folds=outer, inner_folds=inner,
                        unit_label=str(summary['unit']),
                        resample_draws=int(summary['resamples']),
                        resample_seed=int(report['bootstrap_seed']))


def refuse_on_drift(published: CellIdentity, realised: CellIdentity) -> dict:
    """Refuse unless the realised cell is the published cell's own measurement.

    Every difference is named, not just the first, because a recomputation that
    drifted on two axes at once should be reported as such rather than repaired
    one axis at a time.
    """
    differences: list[str] = []
    if published.support != realised.support:
        missing = [a for a in published.support if a not in set(realised.support)]
        added = [a for a in realised.support if a not in set(published.support)]
        if missing or added:
            differences.append(
                f'support differs: {len(missing)} published assays absent '
                f'(e.g. {missing[:3]}), {len(added)} new (e.g. {added[:3]})')
        else:
            differences.append('support holds the same assays in a different order, so the '
                               'concatenated fitted rows are not the published rows')
    for field in ('units', 'variants', 'split_seed', 'unit_label', 'resample_draws',
                  'resample_seed'):
        left, right = getattr(published, field), getattr(realised, field)
        if left != right:
            differences.append(f'{field} differs: published {left!r}, realised {right!r}')
    if published.outer_folds != realised.outer_folds:
        moved = [index for index, (left, right)
                 in enumerate(zip(published.outer_folds, realised.outer_folds)) if left != right]
        differences.append(
            f'outer fold membership differs at folds {moved}'
            if len(published.outer_folds) == len(realised.outer_folds)
            else f'outer fold count differs: published {len(published.outer_folds)}, '
                 f'realised {len(realised.outer_folds)}')
    if published.inner_folds != realised.inner_folds:
        differences.append('inner fold membership differs from the published cell')
    if differences:
        raise RecomputationRefused(
            'the recomputation is not on the published cell\'s own support, folds, seeds, '
            'weighting and label budget:\n  - ' + '\n  - '.join(differences))
    return dict(identical=True, **published.as_dict())


# ------------------------------------------------- representation construction


def require_blas_pool() -> list[int]:
    """The thread counts of this process's BLAS pools, or a refusal.

    A process with no discoverable pool cannot pin the admitted reduction order,
    so it cannot reproduce the admitted float32 product and is refused rather
    than run unpinned.
    """
    counts = blas_thread_counts()
    if not counts:
        raise RecomputationRefused(
            'no BLAS thread pool is discoverable in this process, so the admitted float32 '
            'reduction order cannot be pinned; the recomputation is refused rather than run '
            'at whatever order the host happens to give it')
    return counts


def admitted_representation(blocks: np.ndarray, variant_counts) -> np.ndarray:
    """The admitted 1,024-coordinate block: per-assay rows, pinned reduction order.

    Delegates to the depth sweep's own construction, which blocks the rows by
    assay and pins the thread count inside the class sweep's projection contract.
    Both properties are part of pipeline identity and neither is restated here;
    the delegate's own refusals are re-raised under this driver's name, with their
    message unchanged, so a caller has one exception to handle.
    """
    require_blas_pool()
    try:
        return admitted_design(blocks, variant_counts)
    except ValueError as refusal:
        raise RecomputationRefused(str(refusal)) from refusal


def depth_representation(blocks: dict[str, np.ndarray], summaries, *, depth_index: int,
                         coordinates: int, base_seed: int, variant_counts) -> np.ndarray:
    """One depth's named summaries, projected under the declared rule, blocked by assay.

    The seed of each block is the declared base seed plus eight times the depth
    index plus the summary's index in the declared summary order, which is the
    depth sweep's own rule and depends on no label, observed vector or fitted
    outcome. The rows are blocked by assay for the same reason the admitted
    design is, so a cell is reproducible on its own terms.
    """
    require_blas_pool()
    missing = [name for name in summaries if name not in blocks]
    if missing:
        raise RecomputationRefused(
            f'the retained states hold no {missing} summary at depth {depth_index}; a selection '
            'that names it requires a depth-resolved extraction of this cohort')
    counts = [int(count) for count in variant_counts]
    rows = len(next(iter(blocks.values())))
    if sum(counts) != rows:
        raise RecomputationRefused(f'{sum(counts)} assay rows do not cover {rows} state rows')
    width = int(blocks[summaries[0]].shape[1])
    matrices = {name: gaussian_projection(base_seed + 8 * depth_index + SUMMARY_NAMES.index(name),
                                          width, coordinates)
                for name in summaries}
    pieces, start = [], 0
    for count in counts:
        stop = start + count
        pieces.append(np.concatenate([blocks[name][start:stop] @ matrices[name]
                                      for name in summaries], axis=1))
        start = stop
    return np.concatenate(pieces, axis=0)


def product_blocking_departure(blocks: np.ndarray, variant_counts) -> dict:
    """How far a whole-panel float32 product sits from the per-assay one, measured.

    The admitted analysis forms the compressed block one assay at a time, and the
    two blockings are equally valid float32 evaluations in different reduction
    orders. This reports the departure on the arrays at hand, together with the
    largest absolute entry it is read against and the BLAS pool it was measured
    under, so the constraint that forces per-assay blocking is a number in the
    cell report rather than an assertion.
    """
    counts = [int(count) for count in variant_counts]
    per_assay = admitted_representation(blocks, counts)
    whole = admitted_representation(blocks, [sum(counts)])
    difference = np.abs(per_assay - whole)
    return dict(max_absolute_departure=float(difference.max()),
                largest_absolute_entry=float(np.abs(per_assay).max()),
                identical=bool(np.array_equal(per_assay, whole)),
                rows=int(len(per_assay)), coordinates=int(per_assay.shape[1]),
                assay_blocks=len(counts), blas_thread_counts=require_blas_pool(),
                pinned_blas_threads=PROJECTION_BLAS_THREADS)


def representation(selection: ReadoutSelection, *, admitted_blocks=None, depth_blocks=None,
                   variant_counts=None) -> tuple[np.ndarray, dict]:
    """The representation block this selection declares, with its own provenance.

    The admitted axis reads the four pooled blocks every gate's extraction wrote;
    every other axis reads a depth-resolved extraction, and a gate whose cohort
    has none is refused here rather than silently served the admitted depths.
    """
    if variant_counts is None:
        raise RecomputationRefused('a representation block is formed per assay and needs the '
                                   'per-assay variant counts')
    if selection.depth_axis == 'admitted':
        if admitted_blocks is None:
            raise RecomputationRefused('the admitted axis needs the four admitted pooled blocks')
        spec = CLASSES.get(selection.class_name) or DEPTH_CLASSES[selection.class_name]
        names = {name for design in spec['designs']
                 for name in design.linear + design.mapped} - {'B'}
        if names == {'R_proj'}:
            block = admitted_representation(admitted_blocks, variant_counts)
            form = 'admitted four-block 1,024-coordinate fixed Gaussian projection'
        elif names == {'R_full'}:
            block = np.asarray(admitted_blocks).reshape(len(admitted_blocks), -1)
            form = 'the four admitted blocks uncompressed, at four times the hidden width'
        elif names & {'R_proj', 'R_full'} == {'R_proj', 'R_full'}:
            block = np.hstack([admitted_representation(admitted_blocks, variant_counts),
                               np.asarray(admitted_blocks).reshape(len(admitted_blocks), -1)])
            form = 'the compressed block beside the uncompressed one'
        elif len(names) == 1:
            name = next(iter(names))
            index = ('middle_mean', 'middle_last', 'final_mean', 'final_last').index(name)
            block = np.ascontiguousarray(np.asarray(admitted_blocks)[:, index, :])
            form = f'the {name} block alone, at hidden width'
        else:
            raise RecomputationRefused(
                f'{selection.class_name} names representation blocks {sorted(names)}, which this '
                'driver does not know how to form from the admitted four blocks')
        provenance = dict(form=form, coordinates=int(block.shape[1]),
                          pinned_blas_threads=PROJECTION_BLAS_THREADS,
                          blocked_by_assay=True, assay_blocks=len(list(variant_counts)))
        return block, provenance
    if depth_blocks is None:
        raise RecomputationRefused(
            f'the {selection.depth_axis} axis reads a depth-resolved extraction, and none was '
            'supplied; on a cohort without one the selection requires a fresh extraction')
    if selection.coordinates is None or selection.projection_seed is None:
        raise RecomputationRefused(
            f'the {selection.depth_axis} axis needs its declared coordinate count and projection '
            'seed, which the reassessment\'s selection carries')
    summaries = tuple(name for name in SUMMARY_NAMES if name in depth_blocks)
    if selection.depth_axis in ('depth', 'wide'):
        summaries = tuple(name for name in ('mean', 'last') if name in depth_blocks)
    elif selection.depth_axis == 'pos':
        summaries = tuple(name for name in ('mut', 'suffix') if name in depth_blocks)
    block = depth_representation(depth_blocks, summaries, depth_index=selection.depth_index,
                                 coordinates=selection.coordinates,
                                 base_seed=selection.projection_seed,
                                 variant_counts=variant_counts)
    return block, dict(form=f'{selection.depth_axis} axis at depth {selection.depth_index}: '
                            f'{list(summaries)} at {selection.coordinates} coordinates each',
                       coordinates=int(block.shape[1]),
                       pinned_blas_threads=PROJECTION_BLAS_THREADS, blocked_by_assay=True,
                       assay_blocks=len(list(variant_counts)))


# ------------------------------------------------------------------ comparison


def code_digest_differences(published: dict) -> dict:
    """Files whose bytes differ from the ones the published cell recorded.

    Reported, not refused. A published cell records the digest of every file its
    fit read, and some of those files have since moved -- the resampler among
    them, for every published cell of both gate families. The published bytes are
    not recoverable from the repository, so a file-level comparison can say that
    something changed and cannot say whether it changed a reported number. The
    binding that can is :func:`published_intervals_reproduce`.
    """
    recorded = published.get('analysis_code_sha256') or {}
    root = Path(__file__).resolve().parents[3]
    changed, absent = [], []
    for name, digest in sorted(recorded.items()):
        path = root / name
        if not path.is_file():
            absent.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            changed.append(name)
    return dict(recorded_files=len(recorded), changed=changed, absent=absent,
                reading=('a changed file is reported rather than refused, because the published '
                         'bytes cannot be recovered to diff against; whether the change reached a '
                         'reported number is what the resampler agreement measures'))


def published_intervals_reproduce(published: dict, metrics) -> dict:
    """Refuse unless the current resampler reproduces the published intervals exactly.

    This draws nothing itself. It invokes the declared resampler
    ``profile_increment.summarize`` on an input the published report holds, so it
    is a check on that resampler rather than a second one, and its name says so.

    A published cell carries the per-assay rows every summary was formed from, so
    the resampler is a deterministic function of an input the report itself holds.
    Re-running it at the published draw count and seed and requiring the published
    point, interval and unit back is a behavioural check on the code that forms
    every interval. It is stricter than comparing the file's digest, because it
    fails only when a change reached a reported number, and stronger, because it
    fails then even if no digest was ever recorded.
    """
    from ..context.profile_increment import summarize
    rows = published.get('assays')
    if not rows:
        raise RecomputationRefused(
            'the published cell carries no per-assay rows, so the resampler that formed its '
            'intervals cannot be checked against it; a recomputed interval would be compared '
            'against a published one with no evidence that the same resampler formed both')
    if 'bootstrap_seed' not in published:
        raise RecomputationRefused('the published cell declares no bootstrap seed, so its '
                                   'intervals cannot be re-formed from its own rows')
    seed = int(published['bootstrap_seed'])
    checked, differences, worst = [], [], 0.0
    for metric in metrics:
        summary = (published.get('summaries') or {}).get(metric)
        if summary is None:
            continue
        realised = summarize(rows, metric, bootstrap=int(summary['resamples']), seed=seed)
        checked.append(metric)
        if realised.get('unit') != summary.get('unit'):
            differences.append(f"{metric}: unit {realised.get('unit')!r} against the published "
                               f"{summary.get('unit')!r}")
            continue
        if realised.get('point') != summary.get('point') or \
                realised.get('interval') != summary.get('interval'):
            differences.append(f"{metric}: {realised.get('point')} {realised.get('interval')} "
                               f"against the published {summary.get('point')} "
                               f"{summary.get('interval')}")
            continue
        if summary.get('point') is not None and summary.get('interval') is not None:
            worst = max([worst, abs(realised['point'] - summary['point'])]
                        + [abs(a - b) for a, b in zip(realised['interval'], summary['interval'])])
    if differences:
        raise RecomputationRefused(
            'the resampler no longer reproduces the published intervals from the published cell\'s '
            'own rows, so a recomputed interval is not comparable with a published one:\n  - '
            + '\n  - '.join(differences))
    if not checked:
        raise RecomputationRefused('none of this gate\'s declared metrics is present in the '
                                   'published cell, so the resampler cannot be checked against it')
    return dict(metrics_checked=len(checked), metrics=checked, resample_seed=seed,
                max_absolute_difference=worst, exact=worst == 0.0,
                control=('the current resampler re-formed each published summary from the '
                         'published cell\'s own per-assay rows at its own draw count and seed'))


def compare_increments(published: dict, recomputed: dict, metrics) -> list[dict]:
    """Each metric's recomputed increment beside the published one.

    Both intervals and the resampling unit travel with every row, and the unit is
    compared rather than assumed: an interval over a different unit is not
    comparable with the published one whatever its width.
    """
    rows = []
    for metric in metrics:
        left = published['summaries'].get(metric)
        right = recomputed['summaries'].get(metric)
        if left is None or right is None:
            rows.append(dict(metric=metric, published=None, recomputed=None,
                             comparable=False,
                             reason=('the published report carries no such metric'
                                     if left is None else
                                     'the recomputed report carries no such metric')))
            continue
        if left['unit'] != right['unit']:
            rows.append(dict(metric=metric, published=left, recomputed=right, comparable=False,
                             reason=f"resampling unit differs: published {left['unit']!r}, "
                                    f"recomputed {right['unit']!r}"))
            continue
        moved = (None if left['point'] is None or right['point'] is None
                 else float(right['point'] - left['point']))
        rows.append(dict(
            metric=metric, comparable=True,
            unit=left['unit'], resample_draws=left['resamples'],
            published=dict(point=left['point'], interval=left['interval'],
                           n_units=left.get('n_units'), excludes_zero=left.get('excludes_zero')),
            recomputed=dict(point=right['point'], interval=right['interval'],
                            n_units=right.get('n_units'),
                            excludes_zero=right.get('excludes_zero')),
            point_change=moved,
            resolved_sign_changed=_sign(left) != _sign(right)))
    return rows


def _sign(summary: dict) -> str:
    """Whether a summary's interval resolves above zero, below it, or not at all."""
    interval = summary.get('interval')
    if not summary.get('excludes_zero') or interval is None:
        return 'unresolved'
    return 'above_zero' if interval[0] > 0 else 'below_zero'


def prediction_agreement(published: dict, recomputed: dict, designs) -> dict:
    """Worst absolute per-variant held-out prediction departure, per design.

    Reported for every selection and *gated* only for the admitted one: at a new
    class or depth the predictions are supposed to move, and gating them there
    would refuse the recomputation for doing its job.
    """
    left = {row['assay']: row for row in published['predictions']}
    worst = {}
    for design in designs:
        deviation = 0.0
        for row in recomputed['predictions']:
            reference = left.get(row['assay'])
            if reference is None or design not in reference or design not in row:
                continue
            a = np.asarray(reference[design], dtype=np.float64)
            b = np.asarray(row[design], dtype=np.float64)
            if a.shape != b.shape:
                raise RecomputationRefused(
                    f"{row['assay']}: published {design} holds {a.shape[0]} predictions against "
                    f'{b.shape[0]} recomputed')
            deviation = max(deviation, float(np.max(np.abs(a - b))))
        worst[design] = deviation
    return worst


def pipeline_agreement(worst: dict, tolerance: float) -> dict:
    """Whether an admitted-selection replay reproduced the published predictions.

    This decides and does not raise, so that a cell which fails it is still
    carried through to its increments and intervals and the refusal arrives with
    its evidence rather than in place of it. The question a reader has when a
    replay departs is whether any *published quantity* moved, and suppressing the
    comparison in order to raise earlier is the one way to make that
    unanswerable. The refusal belongs to the caller: :func:`cell_report` marks the
    verdict and the runner exits non-zero on it, so no cell that failed this gate
    is reported as having passed it.
    """
    offenders = {design: value for design, value in sorted(worst.items()) if value > tolerance}
    return dict(tolerance=tolerance, max_absolute_prediction_deviation=worst,
                passed=not offenders, offenders=offenders,
                refusal=(None if not offenders else
                         'the admitted-selection replay does not reproduce the published held-out '
                         f'predictions inside {tolerance}: '
                         + ', '.join(f'{design} at {value:.6g}'
                                     for design, value in offenders.items())))


def refuse_on_pipeline_drift(worst: dict, tolerance: float) -> dict:
    """The raising form, for a caller with nothing further to report."""
    record = pipeline_agreement(worst, tolerance)
    if not record['passed']:
        raise RecomputationRefused(record['refusal'])
    return record


def summary_agreement(published: dict, recomputed: dict) -> dict:
    """Whether any published quantity moved, over every summary the two share.

    The declared metrics are what a gate reports; this is every summary its cell
    carries, because whether a recomputation is faithful is a question about the
    estimand and not about the four numbers a record happens to headline. A point,
    both interval endpoints, the resampling unit, the unit count and whether the
    interval excludes zero are each compared, and a resolved sign that changes is
    named individually.
    """
    left = published.get('summaries') or {}
    right = recomputed.get('summaries') or {}
    shared = sorted(set(left) & set(right))
    moved, sign_changes = [], []
    worst_point = worst_interval = 0.0
    exact = 0
    for metric in shared:
        a, b = left[metric], right[metric]
        differences = []
        if a.get('unit') != b.get('unit'):
            differences.append(f"unit {b.get('unit')!r} against {a.get('unit')!r}")
        if a.get('n_units') != b.get('n_units'):
            differences.append(f"n_units {b.get('n_units')} against {a.get('n_units')}")
        if a.get('point') != b.get('point'):
            differences.append(f"point {b.get('point')} against {a.get('point')}")
            if a.get('point') is not None and b.get('point') is not None:
                worst_point = max(worst_point, abs(b['point'] - a['point']))
        if a.get('interval') != b.get('interval'):
            differences.append(f"interval {b.get('interval')} against {a.get('interval')}")
            if a.get('interval') is not None and b.get('interval') is not None:
                worst_interval = max([worst_interval]
                                     + [abs(x - y) for x, y in zip(b['interval'], a['interval'])])
        if _sign(a) != _sign(b):
            sign_changes.append(f'{metric}: {_sign(a)} to {_sign(b)}')
        if differences:
            moved.append(dict(metric=metric, differences=differences))
        else:
            exact += 1
    return dict(shared_metrics=len(shared), recovered_exactly=exact, moved=moved,
                resolved_sign_changes=sign_changes,
                max_absolute_point_change=worst_point,
                max_absolute_interval_endpoint_change=worst_interval,
                every_published_quantity_bit_identical=not moved,
                reading=('every summary both reports carry, compared on its point, both interval '
                         'endpoints, its resampling unit, its unit count and whether it excludes '
                         'zero. This is the estimand; the prediction departure is the arithmetic '
                         'path to it, and the two can differ. Bit-identity is reported and is not '
                         'the verdict: the floor is last-bit float64 arithmetic, reached even by '
                         'quantities that carry no representation at all, so the two maxima and '
                         'the resolved-sign list are what a reader should weigh against the '
                         'precision the record reports'))


#: Schema of the per-gate aggregation a batch writes beside its cell records.
BATCH_SCHEMA = 'representation_recomputation_batch_v1'


def aggregate_cells(records, *, failures=(), planned_refusals=()) -> dict:
    """Per-gate aggregation over recomputed cells, with the verdict question first.

    Two counts that mean opposite things are kept strictly apart. A **resolved
    sign change** is a published quantity crossing zero-exclusion: it is a
    scientific result. A **gate refusal with no sign change** is the
    pipeline-identity control firing on an arithmetic departure that reached no
    published number: it is a provenance fact. Collapsing them would report the
    second as if it were the first.

    A gate's verdict is read off its own declared metrics rather than a chosen
    primary: if no declared metric's resolved sign moves in any cell, the verdict
    stands; if one moves, the aggregation names the cell, the metric, the two
    signs and the two intervals, and says nothing about what the gate should now
    record. That assignment belongs to the gate's own record.
    """
    gates: dict[str, dict] = {}
    for record in records:
        gate = record['gate']
        entry = gates.setdefault(gate, dict(
            gate=gate, cells_attempted=0, cells_completed=0,
            cells_whose_pipeline_gate_refused=0,
            cells_refused_with_no_resolved_sign_change=0,
            max_absolute_point_change=0.0, max_absolute_interval_endpoint_change=0.0,
            resolved_sign_changes=[], summaries_compared=0, declared_metrics={},
            selection_kinds=set(), cells=[]))
        entry['cells_attempted'] += 1
        entry['cells_completed'] += 1
        entry['selection_kinds'].add(record['selection']['depth_axis'])
        quantities = record.get('published_quantities') or {}
        entry['summaries_compared'] = max(entry['summaries_compared'],
                                         int(quantities.get('shared_metrics') or 0))
        entry['max_absolute_point_change'] = max(
            entry['max_absolute_point_change'],
            float(quantities.get('max_absolute_point_change') or 0.0))
        entry['max_absolute_interval_endpoint_change'] = max(
            entry['max_absolute_interval_endpoint_change'],
            float(quantities.get('max_absolute_interval_endpoint_change') or 0.0))
        cell = f"{record['arm']}/{record['panel']}/{record['split_seed']}"
        refused = record.get('verdict') == 'refused'
        changes = []
        for row in record.get('increments') or ():
            if not row.get('comparable'):
                continue
            metric = row['metric']
            seen = entry['declared_metrics'].setdefault(metric, dict(cells=0, sign_changes=0))
            seen['cells'] += 1
            if row.get('resolved_sign_changed'):
                seen['sign_changes'] += 1
                changes.append(dict(cell=cell, metric=metric,
                                    published=row.get('published'),
                                    recomputed=row.get('recomputed'),
                                    point_change=row.get('point_change'),
                                    unit=row.get('unit')))
        entry['resolved_sign_changes'].extend(changes)
        if refused:
            entry['cells_whose_pipeline_gate_refused'] += 1
            if not changes:
                entry['cells_refused_with_no_resolved_sign_change'] += 1
        entry['cells'].append(dict(cell=cell, verdict=record.get('verdict'),
                                   resolved_sign_changes=len(changes)))
    for entry in gates.values():
        entry['selection_kinds'] = sorted(entry['selection_kinds'])
        entry['verdict'] = 'changes' if entry['resolved_sign_changes'] else 'stands'
    for failure in failures:
        entry = gates.setdefault(failure['gate'], dict(
            gate=failure['gate'], cells_attempted=0, cells_completed=0,
            cells_whose_pipeline_gate_refused=0,
            cells_refused_with_no_resolved_sign_change=0,
            max_absolute_point_change=0.0, max_absolute_interval_endpoint_change=0.0,
            resolved_sign_changes=[], summaries_compared=0, declared_metrics={},
            selection_kinds=[], cells=[], verdict='incomplete'))
        entry['cells_attempted'] += 1
        entry.setdefault('failures', []).append(failure)
        entry['verdict'] = 'incomplete'
    return dict(
        schema=BATCH_SCHEMA, gates=dict(sorted(gates.items())),
        cells_completed=sum(g['cells_completed'] for g in gates.values()),
        cells_failed=len(list(failures)),
        any_cell_refused=any(g['cells_whose_pipeline_gate_refused'] for g in gates.values()),
        any_verdict_changes=any(g['verdict'] == 'changes' for g in gates.values()),
        gates_whose_verdict_changes=sorted(name for name, g in gates.items()
                                           if g['verdict'] == 'changes'),
        left_for_the_waves=[dict(gate=c['gate'], arm=c['arm'], panel=c['panel'],
                                 refusal=c['refusal']) for c in planned_refusals],
        reading=('a resolved sign change is a published quantity crossing zero-exclusion and is a '
                 'scientific result; a pipeline-gate refusal with no sign change is an arithmetic '
                 'departure that reached no published number. The two counts are reported '
                 'separately because they mean opposite things'))


# ----------------------------------------------------------------------- plans


@dataclass(frozen=True)
class CellRequest:
    """One cell to recompute: a gate, an arm, a panel, a split seed and a selection."""

    gate: str
    arm: str
    panel: str
    split_seed: int
    selection: ReadoutSelection
    published: Path

    def __post_init__(self):
        if self.gate not in GATE_RETENTION:
            raise ValueError(f'{self.gate!r} is not a declared gate')
        if self.selection.arm != self.arm:
            raise ValueError(f'{self.gate}/{self.arm}: the selection names arm '
                             f'{self.selection.arm!r}')
        if self.selection.panel and self.selection.panel != self.panel:
            raise ValueError(f'{self.gate}/{self.arm}: the selection is for panel '
                             f'{self.selection.panel!r}, this cell is on {self.panel!r}')

    def as_dict(self) -> dict:
        return dict(gate=self.gate, arm=self.arm, panel=self.panel, split_seed=self.split_seed,
                    selection=self.selection.as_dict(), published=str(self.published))


#: Gates whose representation cells this driver can load, build and refit, with
#: the entry point it calls unchanged. A gate absent from this mapping is refused
#: by name in :func:`plan`, with what its adapter still needs, rather than being
#: run through a loader written against a schema nobody has read.
IMPLEMENTED_ADAPTERS: dict[str, str] = {
    'readout_panel': 'src.capability.readouts.readout_analysis.evaluate_readouts, on rows loaded from the '
                     'admitted readout extraction manifests',
    'crossed_controls': 'src.capability.readouts.recomputation_adapters.crossed_controls_cell, which calls '
                        'the crossed-controls stage\'s own profile-store binding, block builders '
                        'and src.capability.context.crossed_controls.evaluate_crossed_controls unchanged',
    'local_context': 'src.capability.readouts.recomputation_adapters.local_context_cell, which calls the '
                     'local-context stage\'s own block builders and '
                     'src.capability.context.local_context.evaluate_local_context unchanged, over the '
                     'carried-forward local blocks the published cell names',
}

#: What each remaining adapter needs before it can be written and tested. Each is
#: a retained artefact schema that is only readable on the remote allocation, so
#: writing the loader now would be writing it against an unread layout.
PENDING_ADAPTERS: dict[str, tuple[str, ...]] = {
    'folding_stability': (
        'one loader serves this gate, remote_homology and external_confirmation: all three declare '
        'their designs in src.capability.stability.stability_gate and retain the same two archive kinds per '
        'background, so writing three would be writing one three times',
        'both archive kinds, measured in-pod: full_<arm>_<background>.npz holds a single `features` '
        'array of (rows, 4, hidden width) float32, and <arm>_<background>.npz holds `projected` at '
        '(rows, 4, 256) beside the `likelihood` column and the token descriptors. A refit at a new '
        'class needs the first for the representation and the second for the likelihood, so a '
        'loader reading one kind is silently wrong',
        'the two kinds sit in one directory for the confirmation gates and in two separate runs for '
        'this gate, so the layout is a per-gate fact rather than a family one',
        'the per-arm fit record\'s nuisance block, for the fold map and the purged training groups'),
    'remote_homology': (
        'the stability-family loader above; this gate needs no extraction of any kind for the class '
        'axis, because a refit at a new readout class reads the already-hooked depths and runs no '
        'forward pass, so its 99 class cells wait on code alone',
        'its own 179 family groups and 6,291 substitutions, not the anchor panel: the support, the '
        'fold map and the resampling unit are this cohort\'s',
        '20,130 archives of both kinds over 33 arms, 8.51 GiB measured in-pod'),
    'external_confirmation': (
        'the same stability-family loader, and the same class-axis freedom: no forward pass, so its '
        '99 class cells wait on code alone. Its representation panel currently resolves 0 of 33 '
        'arms at the two hooked depths, which is the ambiguity L53 describes, and this endpoint is '
        'the paper\'s external confirmation, so a null there must be distinguishable from a '
        'breadth limit',
        '28,248 archives of both kinds over 33 arm directories, 139.37 GiB measured in-pod, so a '
        'whole-panel class pass reads that volume once',
        'its tokenisation descriptors qualified in 23 of 33 arms, unlike either sibling, so the '
        'control set a loader must reproduce is not the siblings\' control set'),
    'residue_interactions': (
        'the per-background array schema of results/pairwise_epistasis_20260924/extraction/<arm>/',
        'the per-arm fit record\'s row-identity and fold-identity digests, which the refit must '
        'reproduce through src.capability.interactions.pairwise_epistasis.row_identity'),
    'evolution_adaptation': (
        'nothing to write: this gate needs no adapter at all. Its own analysis stage, '
        '`scripts/capability/context/analyse_retrieval_strata.py`, already takes the crossed-control cell '
        'reports by directory through a repeatable --crossed-reports, re-aggregates their per-assay '
        'increments over the frozen strata declaration it binds by digest, and checks each cell '
        'against the cohort digest rather than against the published run. So the recomputed strata '
        'are that stage pointed at the recomputed cells: `--crossed-reports <recomputed directory>` '
        'with the same --strata-declaration and --expect-declaration-sha256. What it waits on is '
        'the crossed-control recomputation, not code',),
}


def plan(requests, *, roots=None) -> dict:
    """A recomputation plan: the cells that can run, and the cells that cannot.

    A cell is refused here, before any array is read, when its gate fitted no
    representation design, when the gate's retained artefacts cannot serve the
    selection kind, or when no adapter exists for the gate. Each refusal names
    what is missing, because the point of planning before running is to learn
    that from the declaration rather than from a half-finished campaign.
    """
    from ..core.gate_retention import recomputability
    runnable, refused = [], []
    for request in requests:
        verdict = recomputability(request.gate, request.selection.selection_kind)
        entry = request.as_dict()
        entry['retention'] = verdict
        if verdict['verdict'] != 'recomputable':
            entry['refusal'] = (verdict['no_cells_reason'] or '; '.join(verdict['missing'])
                                or f"retention verdict {verdict['verdict']}")
            refused.append(entry)
            continue
        if request.gate not in IMPLEMENTED_ADAPTERS:
            entry['refusal'] = ('no adapter loads and refits this gate\'s cells yet; it needs '
                                + '; '.join(PENDING_ADAPTERS.get(request.gate, ('an adapter',))))
            refused.append(entry)
            continue
        if not request.published.is_file():
            entry['refusal'] = f'the published cell report is not at {request.published}'
            refused.append(entry)
            continue
        entry['adapter'] = IMPLEMENTED_ADAPTERS[request.gate]
        runnable.append(entry)
    from ..core.gate_retention import declaration_digest
    return dict(schema=PLAN_SCHEMA, retention_declaration_sha256=declaration_digest(),
                selection_kinds=list(SELECTION_KINDS),
                roots={} if roots is None else {k: str(v) for k, v in sorted(roots.items())},
                cells=len(runnable) + len(refused), runnable=runnable, refused=refused,
                adapters=dict(implemented=dict(sorted(IMPLEMENTED_ADAPTERS.items())),
                              pending={k: list(v) for k, v in sorted(PENDING_ADAPTERS.items())}))


def cell_report(request: CellRequest, *, identity: dict, provenance: dict,
                comparison: list[dict], predictions: dict, blocking: dict,
                extra: dict | None = None) -> dict:
    """One recomputed cell, with the published increment beside the recomputed one.

    ``verdict`` is ``refused`` when a gate this cell had to pass did not, and the
    record is still complete: a refusal that carries its comparison is what lets a
    reader see whether any published quantity moved, and a caller treating this
    record as an accepted cell is contradicting a field it can read.
    """
    failed = predictions.get('passed') is False
    record = dict(schema=CELL_SCHEMA, **request.as_dict(),
                  verdict='refused' if failed else 'recomputed',
                  refusal=predictions.get('refusal') if failed else None, identity=identity,
                  representation=provenance, product_blocking=blocking,
                  prediction_agreement=predictions, increments=comparison,
                  reading=('the recomputed increment is this selection\'s; the published one is '
                           'the admitted compressed linear readout at the two admitted depths. '
                           'Both are conditional on their own readout class, and neither '
                           'establishes what the states contain.'))
    if extra:
        record.update(extra)
    return record


# --------------------------------------------------- the readout-family adapter

#: Held-out prediction vectors a readout-family replay compares design by design.
READOUT_DESIGNS = ('B', 'R', 'B_R', 'permuted_B', 'permuted_B_R')


def readout_rows(cohort_path, manifest_path, arm: str, assay_ids):
    """Rows and the four admitted pooled blocks for one arm on one declared support.

    Every bound identity -- the cohort digest, the manifest's arm and block order,
    each archive's checksum and metadata, the exact mutation order, the measured
    effects and the profile scores -- is checked by the admitted loader rather
    than by a second implementation, so a recomputation reads the arrays the
    published fit read or refuses.
    """
    from .readout_class_sweep import load_panel
    rows, provenance = load_panel(cohort_path, manifest_path, arm, assay_ids)
    counts = [len(row['mutants']) for row in rows]
    blocks = np.concatenate([row['full'] for row in rows])
    for row in rows:
        row.pop('full', None)
        row.pop('projected', None)
    return rows, blocks, counts, provenance


def readout_cell(request: CellRequest, rows: list[dict], published: dict, *,
                 admitted_blocks=None, depth_blocks=None, variant_counts=None,
                 metrics=None, device: str = 'cpu', receipt=ADMITTED_RECEIPT,
                 progress=None) -> dict:
    """Recompute one readout-family cell and report it beside its published one.

    The fitting procedure is the admitted :func:`evaluate_readouts`, called
    unchanged, so the folds, the targets, the weighting, the penalty grid, the tie
    rule, the label-permutation control and the resampling contract are the
    published cell's own. What this function contributes is the representation
    block at the selected class and depth, the identity refusal, and the
    comparison.
    """
    from .readout_analysis import evaluate_readouts
    if variant_counts is None:
        variant_counts = [len(row['mutants']) for row in rows]
    block, provenance = representation(request.selection, admitted_blocks=admitted_blocks,
                                       depth_blocks=depth_blocks, variant_counts=variant_counts)
    if len(block) != sum(variant_counts):
        raise RecomputationRefused(f'{len(block)} representation rows do not cover '
                                   f'{sum(variant_counts)} variant rows')
    start = 0
    for row, count in zip(rows, variant_counts):
        row['R'] = block[start:start + count]
        start += count
    # The resampling contract is the published cell's own: its draws and its seed
    # are read off its report rather than defaulted, so an interval this function
    # reports beside a published one was formed by the same resampler over the
    # same number of draws of the same unit.
    contract = identity_from_readout_report(published)
    recomputed = evaluate_readouts(rows, device=device,
                                   seed=contract.resample_seed,
                                   fold_seed=request.split_seed,
                                   bootstrap=contract.resample_draws, progress=progress)
    identity = refuse_on_drift(contract, identity_from_readout_report(recomputed))
    worst = prediction_agreement(published, recomputed, READOUT_DESIGNS)
    if request.selection.is_admitted_pipeline:
        agreement = pipeline_agreement(worst, admitted_prediction_tolerance(receipt))
        agreement['control'] = ('pipeline identity: the admitted selection must reproduce the '
                                'published held-out predictions inside the admitted tolerance')
    else:
        agreement = dict(tolerance=None, max_absolute_prediction_deviation=worst, passed=None,
                         control=('not gated: at a new readout class or depth the held-out '
                                  'predictions are expected to move, and the deviations are '
                                  'reported as the size of that move'))
    if metrics is None:
        metrics = GATE_RETENTION[request.gate].published_metrics
    blocking = (product_blocking_departure(admitted_blocks, variant_counts)
                if admitted_blocks is not None else
                dict(measured=False,
                     reason='a depth-axis selection has no admitted four-block array to measure '
                            'the two blockings against; its own product is blocked by assay'))
    return cell_report(request, identity=identity, provenance=provenance,
                       comparison=compare_increments(published, recomputed, metrics),
                       predictions=agreement, blocking=blocking,
                       extra=dict(resampler=published_intervals_reproduce(published, metrics),
                                  fitting_code=code_digest_differences(published),
                                  published_quantities=summary_agreement(published, recomputed),
                                  recomputed=dict(
                           n_assays=recomputed['n_assays'], n_families=recomputed['n_families'],
                           n_variants=recomputed['n_variants'],
                           feature_dimensions=recomputed['feature_dimensions'],
                           selected_alphas={name: [fold['alpha'] for fold in folds]
                                            for name, folds in recomputed['folds'].items()})))
