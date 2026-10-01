#!/usr/bin/env python3
"""Generate the readout-class-sweep campaign manifests from the admitted roster.

One cell per (arm, panel, split seed) of the admitted readout panel, so the sweep
covers exactly the 132 admitted fit cells and nothing else. Cells are balanced
across lanes by a declared cost model over the fitted design dimensions -- an
execution-scheduling quantity only, computed from the retained hidden widths and
the panel row counts, never from any measured effect or any fitted outcome.

The manifests this writes are consumed by h200/h200_campaign_queue.sh
and reach a pod only inside a frozen code snapshot. No pod name is written here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.capability.readouts.readout_class_sweep import (BANDWIDTHS, EXTENDED_ALPHAS, RANDOM_FEATURES)
from src.capability.readouts.readout_analysis import ALPHAS as BASELINE_ALPHAS

#: Observed float64 throughput on the allocation, in floating-point operations per
#: second: one dense product and one Cholesky factorization. Used for load
#: balancing only.
PRODUCT_RATE = 2.6e12
FACTOR_RATE = 1.0e12
PRIMARY_SEED = 20260923
KEY = 'sweep'
EXPECT = 'readout_class_sweep.json'


def fit_cost(train: float, test: float, columns: int, penalties: int,
             mapped_columns: int = 0) -> float:
    """Seconds for one ridge fit: covariance, factorizations, and any random map."""
    seconds = (2 * train * columns ** 2 / PRODUCT_RATE
               + penalties * columns ** 3 / 3 / FACTOR_RATE
               + 2 * test * columns * penalties / PRODUCT_RATE)
    if mapped_columns:
        seconds += 2 * (2 * train + test) * mapped_columns * RANDOM_FEATURES / PRODUCT_RATE
    return seconds


def design_cost(rows: int, columns: int, penalties: int, capacities: int = 1,
                mapped_columns: int = 0) -> float:
    """Five outer folds, each four inner folds per capacity plus one refit."""
    outer_train, outer_test = 0.8 * rows, 0.2 * rows
    inner_train, inner_test = 0.75 * outer_train, 0.25 * outer_train
    return 5 * (4 * capacities * fit_cost(inner_train, inner_test, columns, penalties, mapped_columns)
                + fit_cost(outer_train, outer_test, columns, penalties, mapped_columns))


def cell_cost(rows: int, width: int, *, depth: bool, shuffle: bool) -> float:
    baseline, projected, combined = 446, 1024, 1470
    full = 4 * width
    short, long = len(BASELINE_ALPHAS), len(EXTENDED_ALPHAS)
    capacities, mapped = len(BANDWIDTHS), RANDOM_FEATURES
    total = sum(design_cost(rows, columns, short)
                for columns in (baseline, projected, combined))
    total += sum(design_cost(rows, columns, long)
                 for columns in (baseline, projected, combined))
    total += design_cost(rows, full, long) + design_cost(rows, full + baseline, long)
    total += design_cost(rows, baseline + mapped, long, capacities, baseline)
    total += (design_cost(rows, projected + mapped, long, capacities, projected)
              + design_cost(rows, combined + mapped, long, capacities, combined))
    total += (design_cost(rows, projected + mapped, long, capacities, full)
              + design_cost(rows, combined + mapped, long, capacities, full + baseline))
    if depth:
        total += 4 * (design_cost(rows, width, long) + design_cost(rows, baseline + width, long))
    if shuffle:
        total += 2 * design_cost(rows, baseline, long)
        total += design_cost(rows, combined, long) + design_cost(rows, full + baseline, long)
        total += design_cost(rows, baseline + mapped, long, capacities, baseline)
        total += (design_cost(rows, combined + mapped, long, capacities, combined)
                  + design_cost(rows, combined + mapped, long, capacities, full + baseline))
    return total


def label(arm: str, panel: str, seed: int) -> str:
    stem = re.sub(r'[^a-z0-9]+', '_', f'class_sweep_{arm}_{panel}_{seed}'.lower())
    return re.sub(r'_+', '_', stem).strip('_')


def cells(roster: dict, widths: dict[str, int], threads: int, roster_path: str) -> list[dict]:
    out = []
    for panel, declared in sorted(roster['panels'].items()):
        rows = roster['panel_counts'][panel]['variants']
        for arm in sorted(declared['arms']):
            for seed in sorted(declared['seeds']):
                primary = seed == PRIMARY_SEED
                argv = ['--roster', roster_path, '--arm', arm, '--panel', panel,
                        '--fold-seed', str(seed), '--threads', str(threads)]
                if primary:
                    argv += ['--depth-resolution', '--shuffle-control']
                out.append(dict(label=label(arm, panel, seed), arm=arm, panel=panel, seed=seed,
                                cost=cell_cost(rows, widths[arm], depth=primary, shuffle=primary),
                                args=' '.join(argv)))
    return out


def assign(entries: list[dict], lanes: int) -> list[list[dict]]:
    """Deterministic longest-first balancing; ties break on the label."""
    loads = [0.0] * lanes
    buckets: list[list[dict]] = [[] for _ in range(lanes)]
    for entry in sorted(entries, key=lambda e: (-e['cost'], e['label'])):
        index = min(range(lanes), key=lambda i: (loads[i], i))
        buckets[index].append(entry)
        loads[index] += entry['cost']
    return buckets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roster', required=True, type=Path, help='Local copy of the frozen roster')
    parser.add_argument('--roster-path', required=True,
                        help='Absolute path the roster will have on the shared filesystem')
    parser.add_argument('--widths', required=True, type=Path,
                        help='JSON mapping of arm to retained hidden width')
    parser.add_argument('--lanes', type=int, default=6)
    parser.add_argument('--threads', type=int, default=48)
    parser.add_argument('--prefix', default='campaign_class_sweep')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    roster = json.loads(args.roster.read_text())
    widths = json.loads(args.widths.read_text())
    missing = {e['arm'] for e in roster['extractions']} - set(widths)
    if missing:
        raise ValueError(f'no retained hidden width for {sorted(missing)}')
    entries = cells(roster, widths, args.threads, args.roster_path)
    buckets = assign(entries, args.lanes)
    runtime = (f'TRANSFER_PYTHON={Path(args.roster_path).parents[2]}/runtimes/ct-20260905/bin/python'
               ' PYTHONDONTWRITEBYTECODE=1'
               + ''.join(f' {name}={args.threads}' for name in
                         ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                          'NUMEXPR_NUM_THREADS')))
    for index, bucket in enumerate(buckets, start=1):
        path = args.out / f'{args.prefix}_p{index}.tsv'
        lines = [
            '# Readout-class sweep: one cell per admitted (arm, panel, split seed) fit cell.',
            '# CPU-only refitting over retained frozen states; no model inference and no GPU.',
            '# Cells are ordered longest-first by a scheduling cost model over design',
            '# dimensions; that order carries no measured effect and no fitted outcome.',
            '# slot key gpu stage label env expect args',
        ]
        for slot, entry in enumerate(bucket, start=1):
            lines.append('\t'.join((str(slot), KEY, 'cpu', 'analyse_readout_class_sweep.py',
                                    entry['label'], runtime, EXPECT, entry['args'])))
        path.write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(cells=len(entries), lanes=args.lanes,
                          lane_cells=[len(b) for b in buckets],
                          lane_hours=[round(sum(e['cost'] for e in b) / 3600, 2) for b in buckets],
                          prefix=str(args.out / args.prefix))))


if __name__ == '__main__':
    main()
