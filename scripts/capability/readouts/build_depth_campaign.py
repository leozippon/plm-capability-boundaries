#!/usr/bin/env python3
"""Generate the depth-sweep campaign manifests from the admitted roster.

Extraction covers every arm of the admitted roster -- the whole declared
extraction roster, with no selection at all -- at that arm's own admitted
precision setting and batch size. Fitting covers every (arm, panel) pair the
roster declares, at every split seed that panel declares, so the sweep composes
with the admitted panel cell for cell.

Balancing uses a declared cost model: for extraction, the admitted manifest's
own measured wall-clock seconds, scaled by the ratio of hooked blocks to the two
the admitted run hooked; for fitting, the fitted design dimensions and panel row
counts. Both are execution-scheduling quantities computed from retained
metadata, never from a measured effect or a fitted outcome. No pod name is
written here: a lane is an index and an in-pod GPU index only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.capability.readouts import readout_depth as rd

#: Observed float64 throughput on the allocation, in floating-point operations
#: per second: one dense product and one symmetric eigendecomposition. Load
#: balancing only.
PRODUCT_RATE = 2.6e12
EIGEN_RATE = 3.9e10
#: Pooling every block instead of two adds a per-block reduction and one
#: device-to-host copy to each forward. Measured against the admitted wall clock
#: this is the declared overhead factor applied per hooked block beyond two.
BLOCK_OVERHEAD = 0.004
EXTRACT_KEY = 'depth'
EXTRACT_STAGE = 'extract_depth_readout.py'
MERGE_STAGE = 'merge_depth_shards.py'
FIT_STAGE = 'analyse_readout_depth.py'
#: The campaign queue substitutes this with the run directory of the cell it is
#: launching, so extraction and fitting under one frozen snapshot share a
#: directory without this generator needing to know the run identifier.
RUN_DIR = '${TRANSFER_RESULTS_RUN_DIR}'
#: Staged validated interpreter, relative to the in-pod project root. The root is
#: host-local, so --python defaults to <TRANSFER_PROJECT_ROOT>/<this suffix> rather
#: than to one cluster's absolute path; h200/h200_env.sh exports
#: TRANSFER_PROJECT_ROOT. The runner expands nothing in an env column, so the
#: resolved path is written into each cell in full.
STAGED_PYTHON_SUFFIX = 'runtimes/ct-20260905/bin/python'


def label(text: str) -> str:
    return re.sub(r'_+', '_', re.sub(r'[^a-z0-9]+', '_', text.lower())).strip('_')


def extraction_cost(seconds: float, blocks: int) -> float:
    """Admitted wall clock plus the declared per-block pooling overhead."""
    return seconds * (1.0 + BLOCK_OVERHEAD * max(blocks - 2, 0))


def fit_cost(rows: int, columns: int, penalties: int) -> float:
    train, test = 0.8 * rows, 0.2 * rows
    inner_train, inner_test = 0.75 * train, 0.25 * train

    def one(fit_rows, predict_rows):
        return (2 * fit_rows * columns ** 2 / PRODUCT_RATE
                + columns ** 3 / EIGEN_RATE
                + 2 * predict_rows * columns * penalties / PRODUCT_RATE)

    return 5 * (4 * one(inner_train, inner_test) + one(train, test))


def cell_cost(rows: int, width: int, depth: int, position: bool, seeds: int) -> float:
    baseline, penalties = 446, len(rd.ALPHAS)
    pooled = 2 * rd.DEPTH_PROJECTION_DIM
    wide = 2 * rd.WIDE_PROJECTION_DIM
    union = 2 * depth * rd.UNION_PROJECTION_DIM
    full = 4 * rd.DEPTH_PROJECTION_DIM
    total = seeds * sum(fit_cost(rows, columns, penalties)
                        for columns in (baseline, 4 * rd.ADMITTED_PROJECTION_DIM,
                                        baseline + 4 * rd.ADMITTED_PROJECTION_DIM))
    total += seeds * depth * sum(fit_cost(rows, columns, penalties)
                                 for columns in (pooled, baseline + pooled))
    total += depth * sum(fit_cost(rows, columns, penalties) for columns in (wide, baseline + wide))
    total += sum(fit_cost(rows, columns, penalties) for columns in (union, baseline + union))
    if position:
        total += seeds * depth * sum(fit_cost(rows, columns, penalties)
                                     for columns in (pooled, baseline + pooled))
        total += depth * sum(fit_cost(rows, columns, penalties)
                             for columns in (full, baseline + full))
    # Reading every depth of the retained states, at the observed shared-filesystem rate.
    total += depth * 2 * rows * width * 4 / 1.0e9
    return total


def assign(entries: list[dict], lanes: int) -> list[list[dict]]:
    loads = [0.0] * lanes
    buckets: list[list[dict]] = [[] for _ in range(lanes)]
    for entry in sorted(entries, key=lambda e: (-e['cost'], e['label'])):
        index = min(range(lanes), key=lambda i: (loads[i], i))
        buckets[index].append(entry)
        loads[index] += entry['cost']
    return buckets


def write_manifest(path: Path, header: list[str], rows: list[tuple]) -> None:
    lines = [*header, '# slot key gpu stage label env expect args']
    for slot, row in enumerate(rows, start=1):
        lines.append('\t'.join((str(slot), *(str(value) for value in row))))
    path.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roster', required=True, type=Path, help='Local copy of the frozen roster')
    parser.add_argument('--roster-path', required=True, help='Roster path on the shared filesystem')
    parser.add_argument('--cohort-path', required=True, help='Cohort path on the shared filesystem')
    parser.add_argument('--conditioning-path', required=True,
                        help='Frozen genuine-EC sidecar path on the shared filesystem')
    parser.add_argument('--depths', required=True, type=Path,
                        help='JSON mapping of arm to retained block count, hidden width and '
                             'admitted wall-clock seconds')
    parser.add_argument('--shards', type=Path,
                        help='JSON mapping of arm to shard count, for arms split across GPUs')
    parser.add_argument('--lane-gpus', required=True,
                        help='Comma-separated in-pod GPU index per extraction lane')
    parser.add_argument('--fit-lanes', type=int, default=6)
    parser.add_argument('--fit-threads', type=int, default=24)
    parser.add_argument('--extract-threads', type=int, default=4)
    parser.add_argument('--extract-run-dir', default=RUN_DIR,
                        help='Run directory holding the extraction outputs. The default is the '
                             'campaign queue placeholder, which resolves to the run directory of '
                             'the snapshot a cell is launched from; pass an absolute path when '
                             'fitting runs from a later snapshot than extraction did')
    parser.add_argument('--python', default=None,
                        help='Validated interpreter path on the shared filesystem; defaults to '
                             f'<TRANSFER_PROJECT_ROOT>/{STAGED_PYTHON_SUFFIX}')
    parser.add_argument('--prefix', default='campaign_depth')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    if args.python is None:
        project_root = os.environ.get('TRANSFER_PROJECT_ROOT')
        if not project_root:
            parser.error('--python is not set and TRANSFER_PROJECT_ROOT is not in the '
                         'environment, so the validated interpreter cannot default to '
                         f'<TRANSFER_PROJECT_ROOT>/{STAGED_PYTHON_SUFFIX}; pass --python, or '
                         'source h200/h200_env.sh')
        args.python = f'{project_root.rstrip("/")}/{STAGED_PYTHON_SUFFIX}'

    roster = json.loads(args.roster.read_text())
    plan = json.loads(args.depths.read_text())
    shards = json.loads(args.shards.read_text()) if args.shards else {}
    arms = {entry['arm']: entry for entry in roster['extractions']}
    missing = set(arms) - set(plan)
    if missing:
        raise ValueError(f'no declared depth or cost for {sorted(missing)}')
    lane_gpus = [int(value) for value in args.lane_gpus.split(',')]
    if not lane_gpus:
        raise ValueError('at least one extraction lane required')

    extract_env = (f'TRANSFER_PYTHON={args.python} PYTHONDONTWRITEBYTECODE=1'
                   + ''.join(f' {name}={args.extract_threads}' for name in
                             ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')))
    fit_env = (f'TRANSFER_PYTHON={args.python} PYTHONDONTWRITEBYTECODE=1'
               + ''.join(f' {name}={args.fit_threads}' for name in
                         ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                          'NUMEXPR_NUM_THREADS')))

    entries, merges = [], []
    for arm in sorted(arms):
        identity = arms[arm]['identity']
        spec = plan[arm]
        count = int(shards.get(arm, 1))
        common = [f'--arm {arm}', f'--dtype {identity["dtype"]}',
                  f'--batch-size {identity["batch_size"]}', '--budget 1024',
                  '--max-score-drift 0.001', '--max-feature-drift 0.001',
                  f'--cohort {args.cohort_path}']
        if arm == 'zymctrl':
            common.append(f'--conditioning-map {args.conditioning_path}')
        cost = extraction_cost(spec['seconds'], spec['blocks']) / count
        for index in range(count):
            name = label(f'depth_extract_{arm}' + (f'_s{index}' if count > 1 else ''))
            argv = list(common)
            if count > 1:
                argv += [f'--shard-count {count}', f'--shard-index {index}']
            expect = (f'shard_depth_{arm}_{index}.json' if count > 1
                      else f'manifest_depth_{arm}.json')
            entries.append(dict(label=name, arm=arm, cost=cost,
                                row=(EXTRACT_KEY, 'GPU', EXTRACT_STAGE, name, extract_env,
                                     expect, ' '.join(argv))))
        if count > 1:
            sources = ' '.join(f'{args.extract_run_dir}/{label(f"depth_extract_{arm}_s{index}")}/'
                               f'shard_depth_{arm}_{index}.json' for index in range(count))
            merges.append((EXTRACT_KEY, 'cpu', MERGE_STAGE, label(f'depth_merge_{arm}'),
                           fit_env, f'manifest_depth_{arm}.json',
                           f'--cohort {args.cohort_path} --shards {sources}'))

    buckets = assign(entries, len(lane_gpus))
    for index, (bucket, gpu) in enumerate(zip(buckets, lane_gpus)):
        rows = [(entry['row'][0], gpu, *entry['row'][2:]) for entry in bucket]
        write_manifest(args.out / f'{args.prefix}_extract_g{index}.tsv', [
            '# Depth-resolved readout extraction: every transformer block, one forward per sequence.',
            '# One cell per arm or shard, at that arm\'s own admitted precision setting and batch size.',
            '# Cells are ordered longest-first by the admitted wall clock and the hooked block count;',
            '# that order carries no measured effect and no fitted outcome.'], rows)
    if merges:
        write_manifest(args.out / f'{args.prefix}_merge.tsv', [
            '# CPU-only merge of disjoint depth-extraction shard receipts into canonical manifests.',
            '# Dispatch only after every worker of the merged arm has completed.'], merges)

    fits = []
    for panel, declared in sorted(roster['panels'].items()):
        rows = roster['panel_counts'][panel]['variants']
        for arm in sorted(declared['arms']):
            spec = plan[arm]
            name = label(f'depth_fit_{arm}_{panel}')
            argv = [f'--roster {args.roster_path}', f'--arm {arm}', f'--panel {panel}',
                    f'--depth-manifest {args.extract_run_dir}/'
                    f'{label("depth_extract_" + arm) if int(shards.get(arm, 1)) == 1 else label("depth_merge_" + arm)}'
                    f'/manifest_depth_{arm}.json',
                    f'--threads {args.fit_threads}']
            fits.append(dict(label=name,
                             cost=cell_cost(rows, spec['width'], spec['blocks'],
                                            bool(spec['position_resolved']),
                                            len(declared['seeds'])),
                             row=(EXTRACT_KEY, 'cpu', FIT_STAGE, name, fit_env,
                                  'readout_depth.json', ' '.join(argv))))
    for index, bucket in enumerate(assign(fits, args.fit_lanes), start=1):
        write_manifest(args.out / f'{args.prefix}_fit_p{index}.tsv', [
            '# Depth-resolved readout fitting: one cell per admitted (arm, panel) pair.',
            '# CPU-only refitting over the retained depth-resolved states; no model inference.',
            '# Each cell reproduces the admitted two-depth fit before reporting anything new.'],
            [entry['row'] for entry in bucket])

    print(json.dumps(dict(
        extraction_cells=len(entries), extraction_lanes=len(lane_gpus),
        extraction_lane_cells=[len(bucket) for bucket in buckets],
        extraction_lane_hours=[round(sum(entry['cost'] for entry in bucket) / 3600, 2)
                               for bucket in buckets],
        merge_cells=len(merges), fit_cells=len(fits), fit_lanes=args.fit_lanes,
        fit_lane_hours=[round(sum(entry['cost'] for entry in bucket) / 3600, 2)
                        for bucket in assign(fits, args.fit_lanes)],
        prefix=str(args.out / args.prefix))))


if __name__ == '__main__':
    main()
