#!/usr/bin/env python3
"""Summarise the readout-class sweep into long-form endpoint tables.

Reads the per-cell reports and copies numbers out of them. It performs no fitting,
no resampling, no threshold selection and no model ranking: every value it writes
is read from a report or is a count of interval directions over those reports.

Two conditions are enforced rather than reported. Each report must carry a passing
reproduction of its own admitted baseline cell -- identical support, identical
outer and inner cluster folds, identical selected penalties, and per-variant
held-out predictions agreeing to the admitted tolerance -- and every interval
inside one panel must rest on the same resampling unit, because intervals on
different units are not comparable.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path

#: The tolerance the admitted final admission receipt used to compare matched
#: baseline predictions, in absolute and relative terms.
BASELINE_TOLERANCE = 1e-8
DECLARED_RESAMPLES = 2000
DECLARED_ALPHA = 0.05
DECLARED_BOOTSTRAP_SEED = 20260923
#: Endpoints carried into the tables, per class.
ENDPOINTS = ('delta_spearman', 'delta_rank_mse', 'R_minus_raw_M_spearman',
             'B_R_minus_raw_M_spearman', 'B_spearman', 'R_spearman', 'B_R_spearman',
             'permuted_delta_spearman', 'permuted_B_spearman', 'permuted_B_R_spearman')
SHARED = ('raw_M_spearman', 'raw_P_spearman')


def need(test, message):
    if not test:
        raise ValueError(message)


def state(summary) -> str:
    if summary.get('interval') is None:
        return 'degenerate'
    if not summary['excludes_zero']:
        return 'unresolved'
    return 'resolved_positive' if summary['interval'][0] > 0 else 'resolved_negative'


def rows_from(report: dict, path: Path) -> list[dict]:
    need(report.get('schema_version') == 'd1_readout_class_sweep_v1', f'{path}: wrong schema')
    need(report.get('status') == 'complete', f'{path}: incomplete cell')
    need(report['bootstrap_seed'] == DECLARED_BOOTSTRAP_SEED, f'{path}: undeclared bootstrap seed')
    agreement = report['baseline_agreement']['max_absolute_deviation']
    worst = max(agreement.values())
    need(worst <= BASELINE_TOLERANCE,
         f'{path}: reproduction of the admitted baseline deviates by {worst}')
    classes = sorted(report['alpha_grids'])
    out = []
    for metric in SHARED:
        summary = report['summaries'][metric]
        out.append(dict(stratum=report['stratum'], arm=report['arm'], panel=report['panel'],
                        seed=report['fold_seed'], readout_class='shared', metric=metric,
                        point=repr(summary['point']),
                        lower='' if summary['interval'] is None else repr(summary['interval'][0]),
                        upper='' if summary['interval'] is None else repr(summary['interval'][1]),
                        interval_state=state(summary), assays=summary['n_assays'],
                        excluded_assays=summary['excluded_assays'],
                        resampling_unit=summary['unit'], resampling_units=summary['n_units'],
                        minimum_units=summary['minimum_units']))
    for name in classes:
        for endpoint in ENDPOINTS:
            key = f'{name}/{endpoint}'
            if key not in report['summaries']:
                continue
            summary = report['summaries'][key]
            need(summary['resamples'] == DECLARED_RESAMPLES and summary['alpha'] == DECLARED_ALPHA,
                 f'{path}/{key}: undeclared resampling contract')
            out.append(dict(stratum=report['stratum'], arm=report['arm'], panel=report['panel'],
                            seed=report['fold_seed'], readout_class=name, metric=endpoint,
                            point=repr(summary['point']),
                            lower='' if summary['interval'] is None else repr(summary['interval'][0]),
                            upper='' if summary['interval'] is None else repr(summary['interval'][1]),
                            interval_state=state(summary), assays=summary['n_assays'],
                            excluded_assays=summary['excluded_assays'],
                            resampling_unit=summary['unit'], resampling_units=summary['n_units'],
                            minimum_units=summary['minimum_units']))
    return out


def capacity_rows(report: dict) -> list[dict]:
    out = []
    for label, folds in sorted(report['folds'].items()):
        name, design = label.split('/', 1)
        grid = report['alpha_grids'][name]
        for fold in folds:
            out.append(dict(arm=report['arm'], panel=report['panel'], seed=report['fold_seed'],
                            readout_class=name, design=design, fold=fold['fold'],
                            columns=report['feature_dimensions'][label],
                            selected_alpha=repr(fold['alpha']),
                            selected_bandwidth='' if fold['bandwidth'] is None else repr(fold['bandwidth']),
                            at_grid_maximum=str(fold['alpha'] == max(grid)).lower(),
                            at_grid_minimum=str(fold['alpha'] == min(grid)).lower(),
                            training_clusters=fold['n_training_families']))
    return out


def write_tsv(path: Path, rows: list[dict]):
    need(rows, f'{path}: nothing to write')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', required=True, type=Path,
                        help='Directory searched recursively for readout_class_sweep.json')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--expected-cells', type=int, default=132)
    args = parser.parse_args()
    paths = sorted(args.reports.rglob('readout_class_sweep.json'))
    need(paths, f'no cell reports under {args.reports}')
    increments, capacities, cells, units = [], [], [], {}
    for path in paths:
        raw = path.read_bytes()
        report = json.loads(raw)
        increments.extend(rows_from(report, path))
        capacities.extend(capacity_rows(report))
        observed = (report['summaries']['raw_M_spearman']['unit'],
                    report['summaries']['raw_M_spearman']['n_units'])
        need(units.setdefault(report['panel'], observed) == observed,
             f"{report['panel']} mixes resampling units across cells")
        cells.append(dict(arm=report['arm'], panel=report['panel'], seed=report['fold_seed'],
                          stratum=report['stratum'], hidden_width=report['extraction']['hidden_width'],
                          block_indices=report['extraction']['block_indices'],
                          depth_resolution=report['depth_resolution'],
                          shuffle_control=report['shuffle_control'],
                          elapsed_seconds=report['elapsed_seconds'],
                          report_sha256=hashlib.sha256(raw).hexdigest(),
                          baseline_max_deviation=max(
                              report['baseline_agreement']['max_absolute_deviation'].values())))
    seen = {(c['arm'], c['panel'], c['seed']) for c in cells}
    need(len(seen) == len(cells), 'duplicate cells among the reports')
    counts = defaultdict(lambda: defaultdict(int))
    for row in increments:
        key = '/'.join((row['stratum'], row['panel'], str(row['seed']), row['readout_class'],
                        row['metric']))
        counts[key][row['interval_state']] += 1
    args.out.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out / 'class_sweep_increments.tsv', increments)
    write_tsv(args.out / 'class_sweep_capacity.tsv', capacities)
    write_tsv(args.out / 'class_sweep_cells.tsv', sorted(
        cells, key=lambda c: (c['panel'], c['arm'], c['seed'])))
    summary = dict(cells=len(cells), expected_cells=args.expected_cells,
                   complete=len(cells) == args.expected_cells,
                   panels={panel: dict(unit=unit, units=count)
                           for panel, (unit, count) in sorted(units.items())},
                   worst_baseline_deviation=max(c['baseline_max_deviation'] for c in cells),
                   baseline_tolerance=BASELINE_TOLERANCE,
                   interval_direction_counts={k: dict(sorted(v.items())) for k, v in sorted(counts.items())})
    (args.out / 'class_sweep_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(cells=len(cells), endpoint_rows=len(increments),
                          capacity_rows=len(capacities), complete=summary['complete'],
                          worst_baseline_deviation=summary['worst_baseline_deviation'],
                          output=str(args.out))))


if __name__ == '__main__':
    main()
