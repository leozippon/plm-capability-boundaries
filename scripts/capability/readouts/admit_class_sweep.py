#!/usr/bin/env python3
"""Emit the readout-class sweep's admission receipt.

The sweep was executed twice. The first execution is not admitted: its cells
reproduced the admitted support, cluster folds and selected penalties exactly and
its endpoint summaries matched the admitted report, but its held-out predictions
departed from that report by up to 9.085e-07 because the admitted float32
projection was formed under a different BLAS thread count, above the 1e-8 the
admitted admission receipt binds predictions at. The second execution pins that
thread count inside the loader and passes the gate in every cell.

Both executions left cell reports on the shared filesystem, and some cells appear
in both. A reader comparing two files for one cell cannot tell which produced a
published number, so this receipt names, per cell, the snapshot admitted and the
digest of the admitted report. A duplicate resolves the same way every time: the
admitted snapshot wins, and a cell present only in a non-admitted snapshot is
recorded as not admitted rather than silently used.

It verifies rather than asserts. Every admitted report must carry the declared
schema, a complete status, the declared bootstrap contract and a C0 reproduction
inside the tolerance; a report that does not is refused rather than admitted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

#: The tolerance the admitted final admission receipt binds predictions at.
BASELINE_TOLERANCE = 1e-8
DECLARED_RESAMPLES = 2000
DECLARED_ALPHA = 0.05
DECLARED_BOOTSTRAP_SEED = 20260923
SCHEMA = 'd1_readout_class_sweep_v1'
EXPECTED_CELLS = 132


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_cells(root: Path) -> dict[tuple[str, str, int], dict]:
    """Every cell report under one snapshot's result tree, keyed by cell identity."""
    out = {}
    for path in sorted(root.rglob('readout_class_sweep.json')):
        report = json.loads(path.read_text())
        key = (report['arm'], report['panel'], report['fold_seed'])
        if key in out:
            raise ValueError(f'{root.name} holds two reports for {key}')
        out[key] = dict(path=path, report=report)
    return out


def verify(entry: dict) -> dict:
    """Refuse a report that does not meet the conditions admission rests on."""
    report, path = entry['report'], entry['path']
    if report.get('schema_version') != SCHEMA:
        raise ValueError(f'{path}: schema {report.get("schema_version")} is not {SCHEMA}')
    if report.get('status') != 'complete':
        raise ValueError(f'{path}: status {report.get("status")} is not complete')
    if report.get('bootstrap_seed') != DECLARED_BOOTSTRAP_SEED:
        raise ValueError(f'{path}: undeclared bootstrap seed {report.get("bootstrap_seed")}')
    deviations = report['baseline_agreement']['max_absolute_deviation']
    worst = max(deviations.values())
    for metric, summary in report['summaries'].items():
        if summary.get('interval') is None:
            continue
        if summary['resamples'] != DECLARED_RESAMPLES or summary['alpha'] != DECLARED_ALPHA:
            raise ValueError(f'{path}/{metric}: undeclared resampling contract')
    return dict(worst_prediction_deviation=worst,
                within_tolerance=bool(worst <= BASELINE_TOLERANCE),
                selected_alphas=report['baseline_agreement']['reproduced_selected_alphas'],
                support_assays=report['baseline_agreement']['support_assays'],
                admitted_report=report['baseline_agreement']['admitted_report'],
                projection_blas_threads=report['runtime'].get('projection_blas_threads'),
                threads=report['runtime'].get('threads'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admitted-snapshot', required=True,
                        help='Run id of the snapshot whose cells are admitted')
    parser.add_argument('--admitted-root', required=True, type=Path,
                        help="That snapshot's result tree")
    parser.add_argument('--superseded-snapshot', action='append', default=[],
                        help='Run id of a snapshot whose cells are not admitted; repeat')
    parser.add_argument('--superseded-root', action='append', default=[], type=Path,
                        help='That snapshot\'s result tree, in the same order')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if len(args.superseded_snapshot) != len(args.superseded_root):
        raise ValueError('each --superseded-snapshot needs one --superseded-root')

    admitted = read_cells(args.admitted_root)
    superseded = {name: read_cells(root) for name, root
                  in zip(args.superseded_snapshot, args.superseded_root)}

    cells, duplicates = [], []
    for key in sorted(admitted):
        arm, panel, seed = key
        entry = admitted[key]
        checked = verify(entry)
        if not checked['within_tolerance']:
            raise ValueError(f'{entry["path"]}: C0 reproduction deviates by '
                             f'{checked["worst_prediction_deviation"]}, above {BASELINE_TOLERANCE}')
        elsewhere = [name for name, found in superseded.items() if key in found]
        record = dict(arm=arm, panel=panel, fold_seed=seed,
                      admitted_snapshot=args.admitted_snapshot,
                      admitted_report_sha256=digest(entry['path']),
                      admitted_report_path=str(entry['path']),
                      also_present_in=elsewhere, **checked)
        cells.append(record)
        for name in elsewhere:
            other = superseded[name][key]
            duplicates.append(dict(
                arm=arm, panel=panel, fold_seed=seed, superseded_snapshot=name,
                superseded_report_sha256=digest(other['path']),
                superseded_report_path=str(other['path']),
                admitted_snapshot=args.admitted_snapshot,
                admitted_report_sha256=record['admitted_report_sha256'],
                admitted_report_path=record['admitted_report_path'],
                superseded_worst_prediction_deviation=max(
                    other['report']['baseline_agreement']['max_absolute_deviation'].values())))

    not_admitted = sorted({key for found in superseded.values() for key in found}
                          - set(admitted))
    payload = dict(
        schema_version='d1_readout_class_sweep_admission_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        admitted_snapshot=args.admitted_snapshot,
        superseded_snapshots=list(superseded),
        baseline_tolerance=BASELINE_TOLERANCE,
        expected_cells=EXPECTED_CELLS, admitted_cells=len(cells),
        complete=len(cells) == EXPECTED_CELLS,
        worst_prediction_deviation=max(c['worst_prediction_deviation'] for c in cells),
        projection_blas_threads=sorted({c['projection_blas_threads'] for c in cells}),
        duplicate_cells=len(duplicates),
        cells_only_in_superseded=[dict(arm=a, panel=p, fold_seed=s) for a, p, s in not_admitted],
        resolution=('A cell present in more than one snapshot is taken from the admitted snapshot. '
                    'A cell present only in a superseded snapshot is not admitted and no number '
                    'from it is published.'),
        cells=cells, duplicates=duplicates,
        receipt_code_sha256=digest(Path(__file__)))
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / 'class_sweep_admission.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + '\n')
    temporary.replace(destination)
    print(json.dumps(dict(admitted_cells=len(cells), expected=EXPECTED_CELLS,
                          complete=payload['complete'],
                          worst_prediction_deviation=payload['worst_prediction_deviation'],
                          projection_blas_threads=payload['projection_blas_threads'],
                          duplicate_cells=len(duplicates),
                          cells_only_in_superseded=len(not_admitted),
                          output=str(destination))))


if __name__ == '__main__':
    main()
