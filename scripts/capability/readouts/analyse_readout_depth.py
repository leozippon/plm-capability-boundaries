#!/usr/bin/env python3
"""Fit the frozen readout at every depth of one admitted (arm, panel) cell.

One invocation is one arm on one admitted panel, at every split seed that panel
declares. It reproduces the admitted two-depth fit twice before reporting
anything new -- once from the admitted retained states, which binds this
fitting code to the admitted report, and once from the depth-resolved
re-extraction, which binds the new extraction to it -- and then fits the
declared depth, capacity, union and position axes on identical rows, supports,
cluster folds, weights, targets and resampling contract.

It performs no model inference. The panel, the assay support, the cohort and
the admitted comparison reports are read from the frozen expansion roster,
which is bound by SHA-256 so a cell cannot silently be fitted against a
different panel definition.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.capability.readouts import readout_depth as rd
from src.capability.readouts.recomputation import admitted_prediction_tolerance  # noqa: E402

#: The admitted full-available-panel roster this sweep is bound to. Its digest is
#: the one the passed final admission receipt records for `expected.json`.
ROSTER_SHA256 = '1bf26a6ae8d71e3c0f790f12c92da7d3dd4a9add7be6f7ef807836297e12dd0c'
#: The admitted final admission receipt, whose ``baseline_tolerance.atol`` is the
#: tolerance the pipeline-identity control is gated on. Read from the receipt
#: rather than transcribed: a copied tolerance can drift from the artefact that
#: established it, and a tolerance is exactly the constant whose drift corrupts
#: silently. Resolved lazily through
#: :func:`~src.capability.readouts.recomputation.admitted_prediction_tolerance`, which
#: refuses rather than defaulting when the receipt cannot be read -- a replay that
#: invented its own tolerance would certify nothing. The default is the
#: project-relative path; a run whose code is a frozen snapshot passes
#: ``--admission-receipt`` with the project's own absolute path.
ADMISSION_RECEIPT = ROOT / 'results/R2/readout_20260923/final_admission.json'
CODE_FILES = ('scripts/capability/readouts/analyse_readout_depth.py',
              'src/capability/readouts/readout_depth.py',
              'src/capability/readouts/readout_analysis.py',
              'src/capability/context/profile_increment.py',
              'src/capability/context/profiles.py')
ADMITTED_DESIGN_MAP = {'B': 'baseline/B', 'R': 'admitted/R', 'B_R': 'admitted/B_R'}


def digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_cell(roster_path: Path, arm: str, panel: str):
    observed = digest(roster_path)
    if observed != ROSTER_SHA256:
        raise ValueError(f'roster digest {observed} is not the admitted roster {ROSTER_SHA256}')
    roster = json.loads(Path(roster_path).read_text())
    if panel not in roster['panels']:
        raise ValueError(f'panel {panel} is not in the admitted roster')
    declared = roster['panels'][panel]
    if arm not in declared['arms']:
        raise ValueError(f'{arm} is not an arm of panel {panel}')
    manifests = {entry['arm']: entry['manifest'] for entry in roster['extractions']}
    strata = {entry['arm']: entry['stratum'] for entry in roster['extractions']}
    if arm not in manifests:
        raise ValueError(f'{arm} has no extraction manifest in the roster')
    reports = {}
    for entry in roster['analyses']:
        if entry['arm'] == arm and entry['panel'] == panel:
            if entry['seed'] in reports:
                raise ValueError(f'duplicate admitted analysis for {arm}/{panel}/{entry["seed"]}')
            reports[entry['seed']] = Path(entry['report'])
    seeds = sorted(declared['seeds'])
    if sorted(reports) != seeds:
        raise ValueError(f'{arm}/{panel}: admitted reports {sorted(reports)} do not match '
                         f'declared seeds {seeds}')
    return dict(cohort=Path(roster['cohort']['path']), cohort_sha256=roster['cohort']['sha256'],
                admitted_manifest=Path(manifests[arm]), assay_ids=list(declared['assay_ids']),
                seeds=seeds, reports=reports, stratum=strata[arm],
                counts=roster['panel_counts'][panel])


#: Relative positions of the declared capacity grid, on which the two axes that
#: exist to answer a capacity objection are fitted rather than at every block.
#: Resolving the increment at every block is the primary axis's job; the capacity
#: axes only have to show that the profile's shape is not an artifact of the
#: coordinate budget, and fitting them at every block would roughly double a
#: cell's cost for no additional depth resolution.
CAPACITY_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def capacity_depths(depth: int) -> tuple[int, ...]:
    """The declared capacity grid: the two admitted blocks plus five relative depths."""
    grid = {round(fraction * (depth - 1)) for fraction in CAPACITY_FRACTIONS}
    grid.update(admitted_block_indices_of(depth))
    return tuple(sorted(grid))


def admitted_block_indices_of(depth: int) -> tuple[int, int]:
    return rd.admitted_block_indices(depth)


def declared_axes(depth: int, position_resolved: bool, primary: bool) -> dict[str, dict]:
    """Every design this cell fits, declared before any fit is inspected.

    ``depth`` runs the two admitted pooling rules at the admitted per-block
    compression, one depth at a time, at every declared split seed. ``wide``
    repeats it at the admitted total representation width, so a null at one
    depth cannot be read as a consequence of halving the coordinate budget.
    ``union`` supplies every depth at once at a reduced per-block width, so
    information spread across depths is not missed by resolving one depth at a
    time. ``position`` and ``full`` add the two position-resolved summaries
    where the rendering defines them. The three capacity axes run at the primary
    split seed only, matching the convention the admitted study uses for its
    sensitivities.
    """
    axes: dict[str, dict] = {}
    for index in range(depth):
        axes[f'depth{index:03d}'] = dict(
            requests=[(index, 'mean'), (index, 'last')],
            dim=rd.DEPTH_PROJECTION_DIM, seed=rd.DEPTH_PROJECTION_SEED)
    if primary:
        for index in capacity_depths(depth):
            axes[f'wide{index:03d}'] = dict(
                requests=[(index, 'mean'), (index, 'last')],
                dim=rd.WIDE_PROJECTION_DIM, seed=rd.WIDE_PROJECTION_SEED)
        axes['union'] = dict(
            requests=[(index, summary) for index in range(depth) for summary in rd.POOLED_SUMMARIES],
            dim=rd.UNION_PROJECTION_DIM, seed=rd.UNION_PROJECTION_SEED)
    if position_resolved:
        for index in range(depth):
            axes[f'pos{index:03d}'] = dict(
                requests=[(index, 'mut'), (index, 'suffix')],
                dim=rd.DEPTH_PROJECTION_DIM, seed=rd.DEPTH_PROJECTION_SEED)
        if primary:
            for index in capacity_depths(depth):
                axes[f'full{index:03d}'] = dict(
                    requests=[(index, summary) for summary in rd.SUMMARY_NAMES],
                    dim=rd.DEPTH_PROJECTION_DIM, seed=rd.DEPTH_PROJECTION_SEED)
    return axes


def fit_cell(panel: rd.DepthPanel, rows: list[dict], admitted_blocks: np.ndarray, reports: dict, *,
             seeds, device: str, bootstrap: int, shuffle: bool, progress,
             receipt=ADMISSION_RECEIPT) -> dict:
    """Every declared design, at every declared seed, on one shared set of rows.

    ``rows`` carries the admitted likelihood, profile score, sequence
    descriptors and measured effects, so the matched supervised baseline is the
    admitted one byte for byte and every depth increment is measured against
    exactly the baseline the admitted panel reports.
    """
    if rd.PRIMARY_SPLIT_SEED not in seeds:
        raise ValueError('the primary split seed must be among a panel\'s declared seeds')
    measured, assays, clusters = rd.row_labels(rows)
    baseline = rd.baseline_design(rows)
    variant_counts = [len(row['mutants']) for row in rows]
    admitted_representation = rd.admitted_design(rd.admitted_blocks_from_depth(panel),
                                                 variant_counts)
    admitted_source_representation = rd.admitted_design(admitted_blocks, variant_counts)
    state = {seed: dict(predictions=dict(raw_P=baseline[:, 0].copy(), raw_M=baseline[:, 1].copy()),
                        folds={}, contrasts={}, dimensions={}) for seed in seeds}

    def fit(fold_seed, label, design, targets=measured):
        progress(f'{fold_seed}/{label}')
        cell = state[fold_seed]
        cell['dimensions'][label] = int(design.shape[1])
        cell['predictions'][label], cell['folds'][label] = rd.fit_design(
            design, targets, assays, clusters, fold_seed=fold_seed, device=device)

    def fit_axis(label, representation, fold_seeds):
        combined = np.column_stack([baseline, representation])
        for fold_seed in fold_seeds:
            fit(fold_seed, f'{label}/R', representation)
            fit(fold_seed, f'{label}/B_R', combined)
            state[fold_seed]['contrasts'][label] = (f'{label}/B_R', 'baseline/B')

    primary = (rd.PRIMARY_SPLIT_SEED,)
    for fold_seed in seeds:
        fit(fold_seed, 'baseline/B', baseline)
    fit_axis('admitted', admitted_representation, seeds)
    fit_axis('admitted_source', admitted_source_representation, primary)

    # One pass over depth: each block's retained states are read once and every
    # design that uses them is built and fitted before they are released.
    union_parts = []
    capacity = set(capacity_depths(panel.depth))
    for index in range(panel.depth):
        blocks = rd.depth_blocks(panel, index)

        def project(names, dim, base_seed, depth_index=index):
            return rd.project_summaries(blocks, names, width=panel.width,  # noqa: F821  (closure over `blocks`; `del blocks` follows every call)
                                        depth_index=depth_index, dim=dim, base_seed=base_seed)

        pooled = project(rd.POOLED_SUMMARIES, rd.DEPTH_PROJECTION_DIM, rd.DEPTH_PROJECTION_SEED)
        fit_axis(f'depth{index:03d}', pooled, seeds)
        if index in capacity:
            fit_axis(f'wide{index:03d}',
                     project(rd.POOLED_SUMMARIES, rd.WIDE_PROJECTION_DIM,
                             rd.WIDE_PROJECTION_SEED), primary)
        union_parts.append(project(rd.POOLED_SUMMARIES, rd.UNION_PROJECTION_DIM,
                                   rd.UNION_PROJECTION_SEED))
        if panel.position_resolved:
            position = project(rd.POSITION_SUMMARIES, rd.DEPTH_PROJECTION_DIM,
                               rd.DEPTH_PROJECTION_SEED)
            fit_axis(f'pos{index:03d}', position, seeds)
            if index in capacity:
                fit_axis(f'full{index:03d}', np.concatenate([pooled, position], axis=1), primary)
            del position
        del blocks, pooled
    fit_axis('union', np.concatenate(union_parts, axis=1), primary)
    del union_parts
    if shuffle:
        shuffled = rd.permuted_labels(measured, assays, rd.BOOTSTRAP_SEED)
        fit(rd.PRIMARY_SPLIT_SEED, 'permuted/B', baseline, shuffled)
        fit(rd.PRIMARY_SPLIT_SEED, 'permuted/admitted_B_R',
            np.column_stack([baseline, admitted_representation]), shuffled)

    out: dict[str, dict] = {}
    for fold_seed in seeds:
        predictions = state[fold_seed]['predictions']
        folds = state[fold_seed]['folds']
        contrasts = state[fold_seed]['contrasts']
        dimensions = state[fold_seed]['dimensions']
        primary_seed = fold_seed == rd.PRIMARY_SPLIT_SEED
        records = rd.per_assay_metrics(rows, assays, measured, predictions, contrasts)
        summaries, cluster_sizes = rd.summarise_metrics(records, bootstrap=bootstrap,
                                                        seed=rd.BOOTSTRAP_SEED)
        admitted_records = [{key: row[key] for key in row
                             if key in ('assay', 'cluster', 'n_variants')
                             or key.startswith(('raw_', 'baseline/', 'admitted/', 'admitted_'))}
                            for row in records]
        agreement = rd.verify_against_admitted_report(
            json.loads(reports[fold_seed].read_text()), panel.assay_ids, rows, records,
            predictions, folds, ADMITTED_DESIGN_MAP)
        agreement['admitted_report'] = dict(path=str(reports[fold_seed]),
                                            sha256=digest(reports[fold_seed]))
        if primary_seed:
            source_map = {'B': 'baseline/B', 'R': 'admitted_source/R', 'B_R': 'admitted_source/B_R'}
            source = rd.verify_against_admitted_report(
                json.loads(reports[fold_seed].read_text()), panel.assay_ids, rows, records,
                predictions, folds, source_map)
            worst = max(source['max_absolute_deviation'][f'{name}_prediction']
                        for name in ('B', 'R', 'B_R'))
            tolerance = admitted_prediction_tolerance(receipt)
            source['tolerance'] = tolerance
            source['tolerance_source'] = str(receipt)
            source['worst_prediction_deviation'] = worst
            source['passed'] = bool(worst <= tolerance)
            agreement['pipeline_identity'] = source
            if not source['passed']:
                raise ValueError(
                    'pipeline-identity control failed: refitting the admitted class from the '
                    f'admitted retained states deviates by {worst} against a tolerance of '
                    f'{tolerance}')
        out[str(fold_seed)] = dict(
            fold_seed=fold_seed, feature_dimensions=dimensions,
            selected_alphas={label: [record['alpha'] for record in value]
                             for label, value in folds.items()},
            baseline_agreement=agreement, summaries=summaries, cluster_sizes=cluster_sizes,
            assays=admitted_records)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roster', required=True, type=Path,
                        help='Frozen readout expansion roster, bound by SHA-256')
    parser.add_argument('--depth-manifest', required=True, type=Path,
                        help='Canonical depth-resolved extraction manifest for this arm')
    parser.add_argument('--arm', required=True)
    parser.add_argument('--panel', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--bootstrap', type=int, default=rd.BOOTSTRAP_DRAWS)
    parser.add_argument('--admission-receipt', type=Path, default=ADMISSION_RECEIPT,
                        help='the admitted final admission receipt whose baseline_tolerance.atol '
                             'gates the pipeline-identity control; refused if unreadable')
    parser.add_argument('--no-shuffle-control', action='store_true')
    args = parser.parse_args()
    if args.device != 'cpu':
        raise ValueError('this stage is a fitting workload and runs on CPU only')
    if args.threads < 1:
        raise ValueError('positive thread count required')
    torch.set_num_threads(args.threads)

    cell = resolve_cell(args.roster, args.arm, args.panel)
    began = time.monotonic()
    panel = rd.DepthPanel(cell['cohort'], args.depth_manifest, args.arm, cell['assay_ids'])
    if panel.hashes[str(cell['cohort'])] != cell['cohort_sha256']:
        raise ValueError('cohort digest differs from the roster')
    for key, expected in (('n_assays', 'assays'), ('n_variants', 'variants')):
        observed = len(panel.rows) if key == 'n_assays' else panel.n_variants
        if observed != cell['counts'][expected]:
            raise ValueError(f'{key} {observed} differs from the roster count '
                             f'{cell["counts"][expected]}')
    if len(panel.clusters) != cell['counts']['clusters']:
        raise ValueError('cluster count differs from the roster count')
    admitted_rows, admitted_blocks, admitted_hashes, admitted_identity = rd.load_admitted_arrays(
        cell['cohort'], cell['admitted_manifest'], args.arm, cell['assay_ids'])
    if [row['assay'] for row in admitted_rows] != panel.assay_ids:
        raise ValueError('admitted and depth supports differ')
    for key in ('measured', 'P'):
        for left, right in zip(admitted_rows, panel.rows, strict=True):
            if not np.array_equal(left[key], right[key]):
                raise ValueError(f'{left["assay"]}: admitted and depth {key} differ')
    likelihood_difference = max(float(np.max(np.abs(left['M'] - right['M'])))
                                for left, right in zip(admitted_rows, panel.rows, strict=True))
    extraction_agreement = rd.block_relative_drift(rd.admitted_blocks_from_depth(panel),
                                                   admitted_blocks)
    extraction_agreement['gate'] = admitted_identity['max_feature_drift']
    extraction_agreement['within_gate'] = bool(
        extraction_agreement['gate'] is not None
        and extraction_agreement['max_relative_l2'] <= admitted_identity['max_feature_drift'])
    extraction_agreement['max_absolute_likelihood_difference_nats'] = likelihood_difference
    extraction_agreement['likelihood_gate'] = admitted_identity['max_score_drift']
    extraction_agreement['likelihood_within_gate'] = bool(
        admitted_identity['max_score_drift'] is not None
        and likelihood_difference <= admitted_identity['max_score_drift'])
    if not extraction_agreement['likelihood_within_gate']:
        raise ValueError('the depth re-extraction native likelihood deviates from the admitted '
                         f'likelihood by {likelihood_difference} nats, above the declared gate')
    print(json.dumps(dict(arm=args.arm, panel=args.panel, assays=len(panel.rows),
                          clusters=len(panel.clusters), variants=panel.n_variants,
                          depth=panel.depth, width=panel.width,
                          position_resolved=panel.position_resolved,
                          seeds=cell['seeds'],
                          extraction_agreement=extraction_agreement,
                          prefix_control=panel.prefix_check())), flush=True)

    seeds = fit_cell(panel, admitted_rows, admitted_blocks, cell['reports'],
                     receipt=args.admission_receipt, seeds=cell['seeds'],
                     device=args.device, bootstrap=args.bootstrap,
                     shuffle=not args.no_shuffle_control,
                     progress=lambda name: print(json.dumps(dict(
                         arm=args.arm, panel=args.panel, fitting=name,
                         elapsed_seconds=round(time.monotonic() - began, 1))), flush=True))
    panel.close()
    report = dict(
        schema_version=rd.ANALYSIS_SCHEMA, status='complete', arm=args.arm, panel=args.panel,
        stratum=cell['stratum'], created_utc=datetime.now(timezone.utc).isoformat(),
        roster=dict(path=str(args.roster), sha256=ROSTER_SHA256),
        support=dict(panel=args.panel, assay_ids=panel.assay_ids, n_assays=len(panel.rows),
                     n_clusters=len(panel.clusters), n_variants=panel.n_variants),
        extraction=dict(manifest=str(args.depth_manifest), identity=panel.identity,
                        block_count=panel.depth, hidden_width=panel.width,
                        position_resolved=panel.position_resolved),
        admitted_extraction=dict(manifest=str(cell['admitted_manifest']), identity=admitted_identity),
        extraction_agreement=extraction_agreement, prefix_control=panel.prefix_check(),
        declared_axes=sorted(declared_axes(panel.depth, panel.position_resolved, True)),
        capacity_depths=list(capacity_depths(panel.depth)),
        capacity_fractions=list(CAPACITY_FRACTIONS),
        projection=dict(depth=dict(dim=rd.DEPTH_PROJECTION_DIM, base_seed=rd.DEPTH_PROJECTION_SEED),
                        wide=dict(dim=rd.WIDE_PROJECTION_DIM, base_seed=rd.WIDE_PROJECTION_SEED),
                        union=dict(dim=rd.UNION_PROJECTION_DIM, base_seed=rd.UNION_PROJECTION_SEED),
                        admitted=dict(dim=rd.ADMITTED_PROJECTION_DIM,
                                      base_seed=rd.ADMITTED_PROJECTION_SEED),
                        rule='Gaussian sd=1/sqrt(dim); seed = base + 8*depth + summary index',
                        summary_order=list(rd.SUMMARY_NAMES)),
        alpha_grid=list(rd.ALPHAS), bootstrap=dict(draws=args.bootstrap, seed=rd.BOOTSTRAP_SEED,
                                                   alpha=0.05,
                                                   unit='wild-type family at 50% identity'),
        seeds=seeds, source_sha256={**panel.hashes, **admitted_hashes},
        code_sha256={name: digest(ROOT / name) for name in CODE_FILES},
        runtime=dict(torch=torch.__version__, numpy=np.__version__, device=args.device,
                     threads=args.threads),
        primary_contrast='per axis, B_R minus B Spearman; B minus B_R rank MSE',
        uncertainty=('95% paired cluster bootstrap conditional on the fitted cross-validation '
                     'predictions; training, tuning and split variation are not refitted, and '
                     'intervals are not adjusted across arms, depths, endpoints or seeds'),
        limitation=('Resolves depth at every transformer block under the two admitted pooling '
                    'rules, and position only where a rendering assigns one token per residue. '
                    'A null bounds this compressed linear class, these pooling and position '
                    'rules, this label budget and this cohort; it does not establish absent '
                    'information.'),
        elapsed_seconds=round(time.monotonic() - began, 1))
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / 'readout_depth.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(destination)
    headline = {}
    for seed, payload in seeds.items():
        best = None
        for label in payload['summaries']:
            if not label.startswith('depth') or not label.endswith('_delta_spearman'):
                continue
            point = payload['summaries'][label]['point']
            if point is not None and (best is None or point > best[1]):
                best = (label, point)
        headline[seed] = dict(
            admitted=payload['summaries']['admitted_delta_spearman']['point'],
            best_depth=None if best is None else best[0],
            best_depth_point=None if best is None else best[1])
    print(json.dumps(dict(arm=args.arm, panel=args.panel, headline=headline,
                          extraction_agreement=extraction_agreement,
                          elapsed_seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
