"""Anchor-panel gate adapters: refit one published cell at a selected readout class.

The crossed-controls and local-context gates are fitted by stage *scripts*, not by
library modules. Their panel loader, profile-store binding, anchor-support
refusal, packed-token budget and control-block builders live in
``scripts/capability/context/analyse_crossed_controls.py`` and
``scripts/capability/context/analyse_local_context_gate.py``. This module loads those
scripts by file location and calls them unchanged, so a recomputation reads the
same arrays, builds the same control blocks and fits the same designs as the
published cell did. What it contributes is the representation block at the
selected class, the identity refusals, and the comparison.

It sits beside :mod:`src.capability.readouts.recomputation` rather than inside it because
loading a production *script* by file location is a different dependency class
from anything the driver itself needs, and the driver should not acquire a
dependency on the scripts directory to plan a cell it is not running.

Both gates fit the same anchor panel -- 201 assays, 163 wild-type families at 50%
identity, 25,728 variants -- through the same loader, so one preparation serves
both and the two entry points differ only in the block builder and the evaluator
they call.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from ..core.gate_retention import GATE_RETENTION
from .readout_class_sweep import PROJECTION_BLAS_THREADS
from .recomputation import (ADMITTED_RECEIPT, CellIdentity, CellRequest, RecomputationRefused,
                            _folds, _plain, admitted_prediction_tolerance, cell_report,
                            code_digest_differences, compare_increments,
                            pipeline_agreement, product_blocking_departure,
                            published_intervals_reproduce, readout_rows, refuse_on_drift,
                            representation, summary_agreement)

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class AnchorStage:
    """One anchor-panel gate's stage script and the published schema it writes.

    ``identity_metric`` names the summary whose resampling contract -- draws,
    alpha and unit -- is read as the cell's own rather than defaulted. Both gates
    fit the bare composition control ``C``, so ``increment_R_C`` exists in every
    cell of either gate and the contract is read from a metric that is present by
    construction rather than by convention.
    """

    gate: str
    script: str
    schema_version: str
    identity_metric: str = 'increment_R_C'


ANCHOR_STAGES: dict[str, AnchorStage] = {
    'crossed_controls': AnchorStage(
        gate='crossed_controls', script='scripts/capability/context/analyse_crossed_controls.py',
        schema_version='d1_crossed_controls_v1'),
    'local_context': AnchorStage(
        gate='local_context', script='scripts/capability/context/analyse_local_context_gate.py',
        schema_version='d1_local_context_v1'),
}

_STAGES: dict[str, object] = {}


def load_stage(gate: str):
    """The gate's production stage script, loaded as a module and called unchanged."""
    if gate not in ANCHOR_STAGES:
        raise RecomputationRefused(f'{gate!r} is not an anchor-panel gate; this module fits '
                                   f'{sorted(ANCHOR_STAGES)}')
    if gate in _STAGES:
        return _STAGES[gate]
    path = ROOT / ANCHOR_STAGES[gate].script
    if not path.is_file():
        raise RecomputationRefused(f'the {gate} stage script is not at {path}')
    for entry in (str(ROOT), str(ROOT / 'scripts/capability')):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location(f'{gate}_stage', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _STAGES[gate] = module
    return module


# ------------------------------------------------------------------- identities


def anchor_identity(report: dict, metric: str) -> CellIdentity:
    """The identity of an anchor-panel cell, read from its own report.

    An anchor-panel cell states one fold map for every design it fits, as a flat
    list of outer folds, rather than one map per design as the readout family
    does; a report carrying anything else is refused rather than read through a
    guess. The support comes from the declared assay order when the report has
    been written out and from the fitted rows when it has not, which is what lets
    the same function read a published cell and a freshly returned one.
    """
    folds = report.get('folds')
    if not isinstance(folds, list):
        raise RecomputationRefused(
            'an anchor-panel cell states one fold map as a list of outer folds; this report '
            f'carries {type(folds).__name__}, so its fold membership cannot be read')
    summaries = report.get('summaries') or {}
    if metric not in summaries:
        raise RecomputationRefused(f'the report carries no {metric} summary, so the resampling '
                                   'contract of the interval it reports cannot be read')
    support = (report.get('support') or {}).get('assay_ids')
    if support is None:
        support = [row['assay'] for row in report['assays']]
    outer, inner = _folds(folds, 'held_families')
    summary = summaries[metric]
    return CellIdentity(support=tuple(str(_plain(assay)) for assay in support),
                        units=int(report['n_families']), variants=int(report['n_variants']),
                        split_seed=int(report['fold_seed']), outer_folds=outer, inner_folds=inner,
                        unit_label=str(summary['unit']),
                        resample_draws=int(summary['resamples']),
                        resample_seed=int(report['bootstrap_seed']))


def _by_name(mapping: dict) -> dict:
    """A source digest map keyed by file name rather than by absolute path.

    The published cells record absolute paths on the allocation they were fitted
    on, so comparing paths would report every source as changed whenever the
    repository sits at a different root. File names are unique across a published
    cell's sources -- one cohort, the extraction manifests, one archive per assay
    -- and a collision is refused rather than resolved, because a name matched to
    the wrong file would compare the wrong bytes.
    """
    named: dict[str, str] = {}
    for key, digest in mapping.items():
        name = Path(key).name
        if name in named and named[name] != digest:
            raise RecomputationRefused(
                f'two source files named {name} carry different digests, so the published '
                'sources cannot be matched to the recomputation\'s by name')
        named[name] = digest
    return named


def refuse_on_source_drift(published: dict, realised: dict, *, exhaustive: bool = True) -> dict:
    """Refuse unless the recomputation read byte-identical sources.

    Support, folds and seeds being identical does not make two fits the same
    measurement if one of them read a different extraction archive. Every
    difference is named rather than the first, for the same reason the fold
    refusal names all of its own.

    ``exhaustive`` is false for a depth-axis recomputation, where the
    representation comes from a depth-resolved extraction of the same cohort that
    the published cell never read. Two source sets then differ by construction --
    it did not read the depth archives and this does not read the admitted ones --
    so only the files **both** records name must match, and the two disjoint sets
    are recorded rather than refused. The depth archives are not left unbound by
    that: the depth panel verifies each one's checksum and metadata against its
    own manifest, and the manifest against the cohort digest. What the published
    source list still binds is the cohort and every manifest the two share.
    """
    left, right = _by_name(published), _by_name(realised)
    missing = sorted(set(left) - set(right))
    added = sorted(set(right) - set(left))
    changed = sorted(name for name in set(left) & set(right) if left[name] != right[name])
    differences = []
    if missing and exhaustive:
        differences.append(f'{len(missing)} published sources were not read (e.g. {missing[:3]})')
    if added and exhaustive:
        differences.append(f'{len(added)} sources were read that the published cell did not '
                           f'(e.g. {added[:3]})')
    if changed:
        differences.append(f'{len(changed)} sources changed bytes since the published cell read '
                           f'them (e.g. {changed[:3]})')
    if differences:
        raise RecomputationRefused('the recomputation did not read the published cell\'s own '
                                   'sources:\n  - ' + '\n  - '.join(differences))
    record = dict(identical=True, exhaustive=exhaustive, files=len(right),
                  published_files=len(left),
                  shared_files=len(set(left) & set(right)),
                  extraction_archives=sum(1 for name in right if name.endswith('.npz')),
                  manifests=sum(1 for name in right if name.startswith('manifest')))
    if not exhaustive:
        record.update(published_sources_not_read=len(missing), sources_not_published=len(added),
                      binding=('every file both records name matched bytes; the two disjoint sets '
                               'are the admitted archives this cell did not read and the '
                               'depth archives the published cell did not, each bound by its own '
                               'manifest checksums rather than by the other record'))
    return record


def published_prediction_arrays(cell_path: Path, report: dict) -> dict[str, np.ndarray]:
    """The published held-out predictions, bound to the digest their report declares."""
    arrays = Path(cell_path).with_suffix('.npz')
    if not arrays.is_file():
        raise RecomputationRefused(f'the published held-out predictions are not at {arrays}, so '
                                   'no prediction comparison is possible for this cell')
    digest = hashlib.sha256(arrays.read_bytes()).hexdigest()
    declared = report.get('prediction_array_sha256')
    if declared and digest != declared:
        raise RecomputationRefused(f'{arrays} does not match the digest its own report declares, '
                                   'so it is not the array that cell was admitted on')
    with np.load(arrays, allow_pickle=False) as data:
        return {name: np.asarray(data[name], dtype=np.float64) for name in data.files}


def flat_prediction_agreement(published: dict, recomputed: dict) -> dict:
    """Worst absolute held-out prediction departure per design, over the whole panel.

    An anchor-panel cell stores one concatenated prediction vector per design in
    panel row order, so the departure is read design by design rather than assay
    by assay. A recomputation that fitted a different design set is refused: it
    is not the published cell's measurement whatever its increments say.
    """
    if set(published) != set(recomputed):
        missing = sorted(set(published) - set(recomputed))
        added = sorted(set(recomputed) - set(published))
        raise RecomputationRefused(
            'the recomputation fitted a different design set than the published cell: '
            f'missing {missing}, extra {added}')
    worst = {}
    for design, left in sorted(published.items()):
        right = np.asarray(recomputed[design], dtype=np.float64)
        if left.shape != right.shape:
            raise RecomputationRefused(
                f'published {design} holds {left.shape} predictions against {right.shape} '
                'recomputed')
        worst[design] = float(np.max(np.abs(left - right)))
    return worst


def resolve_device(report: dict, device: str | None) -> tuple[str, bool]:
    """The device the recomputation fits on, defaulting to the published cell's own.

    A ridge solve on a different device is a different reduction order, so a
    replay on another device would confound a class change with a device change.
    The default is therefore the published cell's device, and a process that
    cannot provide it is refused rather than quietly moved to the host. An
    explicit override is allowed and recorded as a declared departure.
    """
    published = str((report.get('runtime') or {}).get('device', 'cpu'))
    chosen = published if device is None else str(device)
    if chosen.startswith('cuda'):
        import torch
        if not torch.cuda.is_available():
            raise RecomputationRefused(
                f'the published cell was fitted on {published} and this process has no CUDA '
                'device; a host replay is a different reduction order, so it is refused rather '
                'than compared against the published cell as though it were the same fit')
    return chosen, chosen == published


# ------------------------------------------------------------------ preparation


def _manifest_for(manifests, arm: str) -> Path:
    for path in manifests:
        source = json.loads(Path(path).read_bytes())
        if source.get('identity', {}).get('arm') == arm:
            return Path(path)
    raise RecomputationRefused(f'none of the {len(list(manifests))} supplied manifests declares '
                               f'arm {arm!r}')


def published_cell(request: CellRequest) -> dict:
    """The published cell report, checked to be the one this request names."""
    stage = ANCHOR_STAGES.get(request.gate)
    if stage is None:
        raise RecomputationRefused(f'{request.gate!r} is not an anchor-panel gate')
    if not request.published.is_file():
        raise RecomputationRefused(f'the published cell report is not at {request.published}')
    report = json.loads(request.published.read_bytes())
    if report.get('schema_version') != stage.schema_version:
        raise RecomputationRefused(
            f'{request.published} carries schema {report.get("schema_version")!r}, not the '
            f'{stage.schema_version!r} this adapter reads')
    if report.get('status') != 'complete':
        raise RecomputationRefused(f'{request.published} is a {report.get("status")!r} cell, not '
                                   'an admitted one')
    if report.get('arm') != request.arm:
        raise RecomputationRefused(f'{request.published} is arm {report.get("arm")!r}, not '
                                   f'{request.arm!r}')
    if int(report['fold_seed']) != int(request.split_seed):
        raise RecomputationRefused(f'{request.published} is split seed {report["fold_seed"]}, not '
                                   f'{request.split_seed}')
    return report


@dataclass
class PreparedCell:
    """One anchor-panel cell's rows, states and bindings, ready to fit.

    ``blocks`` holds the admitted four pooled blocks on the admitted axis and is
    ``None`` on a depth axis, where ``depth_blocks`` holds that depth's retained
    summaries instead. Exactly one of the two is populated, because the axis
    decides which extraction the representation comes from.
    """

    stage: object
    binding: object
    rows: list
    blocks: object
    counts: list
    provenance: dict
    cohort_record: dict
    store: dict
    sources: dict
    depth_blocks: dict | None = None


def prepare_cell(request: CellRequest, published: dict, *, cohort, manifests, profile_store,
                 anchor_arm: str | None = None, depth_manifest=None) -> PreparedCell:
    """Rows, states and bindings for one anchor-panel cell, or a refusal.

    Every binding is checked against the published cell rather than assumed: the
    cohort bytes, the extraction manifest digests, the profile-store digests the
    cohort itself declares, and the set of manifests whose intersection fixed the
    panel. The support is the published cell's declared assay order, so the
    concatenated rows are the rows it fitted.

    On a depth axis the representation comes from a depth-resolved extraction of
    the **same cohort** -- the anchor cohort has one, so these cells wait on no
    new extraction -- and the rows come from that extraction's own verified panel.
    Its archives are not among the published cell's sources by construction, so
    the source check is scoped rather than dropped: the cohort bytes and every
    manifest still have to match, and the depth archives' digests are recorded.
    """
    stage = load_stage(request.gate)
    # The profile-store binding, the anchor-support refusal and the packed-token
    # budget have one source: the crossed-controls stage. The local-context stage
    # imports them from it rather than keeping a second copy, so this adapter
    # reads them from the same place rather than from whichever stage it is
    # fitting -- asking the local-context module for them would find nothing.
    binding = load_stage('crossed_controls')
    paths = [Path(path) for path in manifests]
    if not paths:
        raise RecomputationRefused('an anchor-panel cell needs the extraction manifests whose '
                                   'intersection fixed its panel')
    # Resolved before anything is read. The anchor arm is not in every published
    # cell -- the stage gained the field after the first cells were fitted -- so it
    # is read when present and named by the caller otherwise, and never defaulted,
    # because a different anchor arm is a different required panel.
    anchor = anchor_arm or published.get('anchor_arm')
    if not anchor:
        raise RecomputationRefused(
            'this published cell records no anchor arm, so the arm whose manifest fixes the '
            'required panel cannot be read from it; name it explicitly rather than letting the '
            'stage default apply, because a different anchor arm is a different panel')
    arm_manifest = _manifest_for(paths, request.arm)
    support = list(published['support']['assay_ids'])
    axis = request.selection.depth_axis
    if axis == 'admitted':
        if depth_manifest is not None:
            raise RecomputationRefused(
                'the admitted axis reads the admitted extraction, so a depth manifest would not '
                'be used; it is refused rather than accepted and ignored')
        rows, blocks, counts, provenance = readout_rows(cohort, arm_manifest, request.arm, support)
        depth_summaries = None
    else:
        if depth_manifest is None:
            raise RecomputationRefused(
                f'the {axis} axis reads a depth-resolved extraction of this cohort; name its '
                'manifest rather than letting the admitted two depths stand in for it')
        rows, blocks, counts, provenance, depth_summaries = _depth_panel(
            request, cohort, depth_manifest, support)
    if provenance['source_sha256'][str(Path(cohort))] != published['cohort_sha256']:
        raise RecomputationRefused('the cohort bytes are not the published cell\'s own')
    if [row['assay'] for row in rows] != support:
        raise RecomputationRefused('the loaded rows are not the published support in its own '
                                   'order, so the concatenated rows are not the published rows')
    sources = dict(provenance['source_sha256'])
    for path in paths:
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    source_check = refuse_on_source_drift(published['source_sha256'], sources,
                                          exhaustive=axis == 'admitted')
    if axis != 'admitted':
        source_check['representation_source'] = (
            'the representation comes from this cohort\'s depth-resolved extraction, whose '
            'archives the published cell never read; its digests are recorded here and the '
            'cohort, the support and every manifest are still required to match')
    cohort_record = json.loads(Path(cohort).read_bytes())
    store, store_hashes = binding.load_profile_store(Path(profile_store), cohort_record)
    if store_hashes != published['profile_store_sha256']:
        raise RecomputationRefused(
            'the profile store does not hold the arrays the published cell read: '
            + ', '.join(sorted(name for name, digest in store_hashes.items()
                               if published['profile_store_sha256'].get(name) != digest)))
    manifest_arms = binding.attach_manifest_tokens(
        rows, paths, request.arm, anchor,
        {str(path): sources[str(path)] for path in paths}, support)
    if sorted(manifest_arms) != list(published['manifest_arms']):
        raise RecomputationRefused(
            f'the panel was fixed by {sorted(manifest_arms)}, not by the '
            f'{list(published["manifest_arms"])} the published cell records')
    return PreparedCell(stage=stage, binding=binding, rows=rows, blocks=blocks, counts=counts,
                        provenance=provenance, cohort_record=cohort_record, store=store,
                        sources=source_check, depth_blocks=depth_summaries)


def _depth_panel(request: CellRequest, cohort, depth_manifest, support: list[str]):
    """One arm's depth-resolved panel at the selected block, or a refusal.

    The panel is the depth sweep's own :class:`~src.capability.readouts.readout_depth.DepthPanel`,
    which checks the manifest's completion, schema and arm, the cohort digest,
    each assay's cluster, wild-type identity and mutation digest, each archive's
    checksum and metadata, the exact ordered mutation identifiers and the measured
    effects and profile scores. Nothing of that is restated here; its refusals are
    re-raised under this driver's name with their message intact. A selected block
    outside the retained range is refused rather than clamped: a nearer depth is
    not the selected one.
    """
    from .readout_depth import DepthPanel, depth_blocks
    index = request.selection.depth_index
    if index is None:
        raise RecomputationRefused(f'the {request.selection.depth_axis} axis names one block and '
                                   'the selection carries no depth index')
    try:
        panel = DepthPanel(cohort, depth_manifest, request.arm, support)
    except (ValueError, KeyError) as refusal:
        raise RecomputationRefused(
            f'the depth-resolved extraction does not serve this cell: {refusal}') from refusal
    try:
        if not 0 <= index < panel.depth:
            raise RecomputationRefused(
                f'the selection names block {index} and this extraction retains {panel.depth} '
                f'blocks, 0 to {panel.depth - 1}; a nearer depth is not the selected one')
        summaries = depth_blocks(panel, index)
        provenance = dict(arm=panel.arm, hidden_width=panel.width, n_assays=len(panel.rows),
                          n_clusters=len(panel.clusters), n_variants=panel.n_variants,
                          source_sha256=dict(panel.hashes), retained_depths=panel.depth,
                          selected_depth=index, position_resolved=panel.position_resolved,
                          retained_summaries=sorted(summaries),
                          prefix_check=panel.prefix_check(),
                          projection_blas_threads=PROJECTION_BLAS_THREADS)
        rows = [dict(row) for row in panel.rows]
        counts = [len(row['mutants']) for row in rows]
    finally:
        panel.close()
    return rows, None, counts, provenance, summaries


def fit_and_compare(request: CellRequest, published: dict, rows: list[dict], blocks, counts,
                    evaluate, *, device: str, receipt=ADMITTED_RECEIPT, depth_blocks=None,
                    sources: dict, extra: dict) -> dict:
    """Substitute the selected representation block, refit, and compare.

    The fitting procedure is the gate's own evaluator, called unchanged, so the
    control sets, folds, targets, weighting, penalty grid and resampling contract
    are the published cell's. The resampling draws and seed are read off the
    published summary rather than defaulted, so an interval reported beside a
    published one was formed by the same resampler over the same number of draws
    of the same unit.
    """
    metric = ANCHOR_STAGES[request.gate].identity_metric
    metrics = GATE_RETENTION[request.gate].published_metrics
    # Before the refit, because it is milliseconds and the refit is hours: the
    # resampler that formed the published intervals must still reproduce them
    # from the published cell's own rows, or a recomputed interval reported beside
    # a published one was formed by different code.
    resampler = published_intervals_reproduce(published, metrics)
    block, provenance = representation(request.selection, admitted_blocks=blocks,
                                       depth_blocks=depth_blocks, variant_counts=counts)
    if len(block) != sum(counts):
        raise RecomputationRefused(f'{len(block)} representation rows do not cover '
                                   f'{sum(counts)} variant rows')
    start = 0
    for row, count in zip(rows, counts):
        row['R'] = block[start:start + count]
        start += count
    contract = anchor_identity(published, metric)
    report, predictions = evaluate(rows, seed=contract.resample_seed,
                                  fold_seed=request.split_seed,
                                  bootstrap=contract.resample_draws, device=device)
    identity = refuse_on_drift(contract, anchor_identity(report, metric))
    worst = flat_prediction_agreement(published_prediction_arrays(request.published, published),
                                      predictions)
    if request.selection.is_admitted_pipeline:
        agreement = pipeline_agreement(worst, admitted_prediction_tolerance(receipt))
        agreement['control'] = ('pipeline identity: the admitted selection must reproduce the '
                                'published held-out predictions inside the admitted tolerance')
    else:
        agreement = dict(tolerance=None, max_absolute_prediction_deviation=worst, passed=None,
                         control=('not gated: at a new readout class or depth the held-out '
                                  'predictions are expected to move, and the deviations are '
                                  'reported as the size of that move'))
    blocking = (product_blocking_departure(blocks, counts)
                if request.selection.depth_axis == 'admitted' else
                dict(measured=False,
                     reason='a depth-axis selection has no admitted four-block array to measure '
                            'the two blockings against; its own product is blocked by assay'))
    return cell_report(
        request, identity=identity, provenance=provenance,
        comparison=compare_increments(published, report, metrics),
        predictions=agreement, blocking=blocking,
        extra=dict(sources=sources, device=device, resampler=resampler,
                   fitting_code=code_digest_differences(published),
                   published_quantities=summary_agreement(published, report),
                   recomputed=dict(n_assays=report['n_assays'], n_families=report['n_families'],
                                   n_variants=report['n_variants'],
                                   feature_dimensions=report['feature_dimensions'],
                                   block_dimensions=report['block_dimensions'],
                                   selected_alphas=report['selected_alphas']),
                   **extra))


def _states(provenance: dict) -> dict:
    """The retained states a cell was built from, on either axis.

    The depth axis carries four more facts a reader needs to interpret the cell:
    how many blocks the extraction retains, which one was selected, whether it
    resolves positions, and the causal prefix control that comes with that.
    """
    record = {key: provenance[key] for key in
              ('arm', 'hidden_width', 'n_assays', 'n_clusters', 'n_variants')}
    for key in ('retained_depths', 'selected_depth', 'position_resolved', 'retained_summaries',
                'prefix_check'):
        if key in provenance:
            record[key] = provenance[key]
    return record


# ------------------------------------------------------------- the two adapters


def crossed_controls_cell(request: CellRequest, *, cohort, manifests, profile_store,
                          device: str | None = None, receipt=ADMITTED_RECEIPT,
                          depth_manifest=None, anchor_arm: str | None = None) -> dict:
    """Recompute one crossed-controls cell at the selected readout class.

    The local-pattern, mutation-local-profile and tokenisation-interface blocks
    are built by the stage's own builder, which re-derives the profile block from
    the retained column-frequency arrays and refuses if the recomputed profile
    scores leave the cohort's own by more than the stage's tolerance, and which
    re-derives the packed-token maxima and refuses if they leave the extraction
    manifest's declared values at all.
    """
    if request.gate != 'crossed_controls':
        raise RecomputationRefused(f'this adapter fits crossed_controls, not {request.gate!r}')
    published = published_cell(request)
    cell = prepare_cell(request, published, cohort=cohort, manifests=manifests,
                        profile_store=profile_store, anchor_arm=anchor_arm,
                        depth_manifest=depth_manifest)
    chosen, matches = resolve_device(published, device)
    declared = published['projection']
    projection = cell.stage.local_projection(int(declared['tripeptide_dim']),
                                             int(declared['tripeptide_seed']))
    interface = cell.stage.build_blocks(cell.rows, cell.cohort_record, cell.store, request.arm,
                                       projection)
    from ..context.crossed_controls import evaluate_crossed_controls

    def evaluate(assays, **kwargs):
        return evaluate_crossed_controls(assays, **kwargs)

    return fit_and_compare(
        request, published, cell.rows, cell.blocks, cell.counts, evaluate, device=chosen,
        receipt=receipt, depth_blocks=cell.depth_blocks, sources=cell.sources,
        extra=dict(device_matches_published=matches, states=_states(cell.provenance),
                   tokenisation_interface={key: value for key, value in interface.items()
                                           if key != 'assays'},
                   tripeptide_projection=dict(dim=int(declared['tripeptide_dim']),
                                              seed=int(declared['tripeptide_seed']))))


def local_context_cell(request: CellRequest, *, cohort, manifests, profile_store,
                       device: str | None = None, receipt=ADMITTED_RECEIPT,
                       depth_manifest=None, anchor_arm: str | None = None) -> dict:
    """Recompute one local-context measurement cell at the selected readout class.

    The carried-forward local blocks and the addition set are read from the
    published cell rather than redeclared, and the frozen candidate declaration
    is bound by digest: a recomputation under a changed declaration would be
    measuring over a different local control, and the qualification receipt that
    makes this gate outcome-blind would no longer cover it. A qualification cell
    is refused, because it fits no representation block at all.
    """
    if request.gate != 'local_context':
        raise RecomputationRefused(f'this adapter fits local_context, not {request.gate!r}')
    published = published_cell(request)
    if published.get('stage') != 'measure':
        raise RecomputationRefused(
            f'{request.published} is a {published.get("stage")!r} cell; a qualification cell fits '
            'no representation block and there is nothing to recompute at a readout class')
    from ..context.local_context import declaration_sha256, evaluate_local_context
    if published.get('declaration_sha256') != declaration_sha256():
        raise RecomputationRefused(
            'the frozen local-context declaration has changed since this cell was fitted: '
            f'published {published.get("declaration_sha256")}, current {declaration_sha256()}. '
            'A recomputation under a changed declaration measures over a different local '
            'control, and the qualification receipt does not cover it')
    cell = prepare_cell(request, published, cohort=cohort, manifests=manifests,
                        profile_store=profile_store, anchor_arm=anchor_arm,
                        depth_manifest=depth_manifest)
    chosen, matches = resolve_device(published, device)
    local_blocks = list(published['local_blocks'])
    additions = {name: tuple(columns) for name, columns in published['additions'].items()}
    deviation = cell.stage.build_blocks(cell.rows, cell.cohort_record, cell.store, local_blocks)
    interface = cell.stage.build_interface(cell.rows, cell.cohort_record, request.arm)

    def evaluate(assays, **kwargs):
        return evaluate_local_context(assays, local_blocks=local_blocks, additions=additions,
                                      **kwargs)

    return fit_and_compare(
        request, published, cell.rows, cell.blocks, cell.counts, evaluate, device=chosen,
        receipt=receipt, depth_blocks=cell.depth_blocks, sources=cell.sources,
        extra=dict(device_matches_published=matches, states=_states(cell.provenance),
                   local_blocks=local_blocks,
                   declaration_sha256=declaration_sha256(),
                   maximum_profile_score_deviation=deviation,
                   tokenisation_interface={key: value for key, value in interface.items()
                                           if key != 'assays'}))


#: The two adapters, by gate. The driver's registry names them; this mapping is
#: what a runner dispatches on, so a gate gains an adapter in exactly one place.
ANCHOR_ADAPTERS = {'crossed_controls': crossed_controls_cell,
                   'local_context': local_context_cell}
