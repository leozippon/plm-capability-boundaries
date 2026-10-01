#!/usr/bin/env python3
"""Release gate: does a retained position-term cell align with its archive?

A full lane is worth running only if the recomputed forward reproduces the
archived scalar it claims to decompose. This entry point answers that on one
cheap cell -- a single background, or the archived three-assay smoke draw -- by
comparing every retained file against the archived file of the same name.

It reports two quantities and classifies the second. T1 is the worst residual
between each retained per-token vector, re-reduced, and the scalar the unmodified
expression produced in the same forward; the extractor refuses to write a cell
whose T1 is nonzero, so a value here is a read-back of that gate. T2 is the worst
absolute difference between the recomputed per-state scalar and the archived one,
in nats, classified against tolerances fixed before the run in
``analyse_position_terms``: tier 1 at exactly zero, tier 2 within 1.0e-4 nats and
within three times the arm's own archived repeat maximum, tier 3 otherwise.

No tolerance is chosen here and none is widened here.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.core.io import write_json

sys.path.insert(0, str(ROOT / 'scripts/capability'))
from importlib import util as _util

_spec = _util.spec_from_file_location('apt', ROOT / 'scripts/capability/position/analyse_position_terms.py')
_apt = _util.module_from_spec(_spec)
_spec.loader.exec_module(_apt)


def absolute(data) -> np.ndarray:
    if 'wt_likelihood' in data.files:
        wild = float(data['wt_likelihood'])
        return np.concatenate(([wild], wild + np.asarray(data['likelihood'], dtype=np.float64)))
    return np.asarray(data['likelihood'], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path, action='append')
    parser.add_argument('--arm', required=True)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--repeat-max-nats', type=float, default=0.0,
                        help="the arm's own archived per-background repeat maximum")
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()

    index = {}
    for root in args.archive:
        for path in Path(root).rglob('*.npz'):
            index.setdefault(path.name, path)
    files = sorted(p for p in args.cell.glob('*.npz') if not p.name.startswith('full_'))
    if not files:
        raise SystemExit(f'no retained files under {args.cell}')
    t1, t2, closure, missing, states = [], [], [], [], 0
    for path in files:
        with np.load(path, allow_pickle=False) as new:
            retained = _apt.Retained(new)
            t1.append(retained.sum_check)
            totals = retained.totals()
            closure.append(float(np.max(np.abs(-totals - absolute(new)))))
            states += len(totals)
            twin = index.get(path.name)
            if twin is None:
                missing.append(path.name)
                continue
            with np.load(twin, allow_pickle=False) as old:
                new_absolute, old_absolute = absolute(new), absolute(old)
                if new_absolute.shape != old_absolute.shape:
                    raise SystemExit(f'{path.name}: state count differs from the archive')
                t2.append(np.abs(new_absolute - old_absolute))
    if missing:
        raise SystemExit(f'no archived counterpart for {len(missing)} retained files: {missing[:3]}')
    t1_worst = float(np.max(np.abs(np.asarray(t1, dtype=np.float64))))
    values = np.concatenate(t2)
    t2_worst = float(values.max())
    realised = _apt.tier(t2_worst, args.repeat_max_nats)
    record = {'schema': 'position_terms_gate_v1', 'arm': args.arm, 'cohort': args.cohort,
              'cell': str(args.cell), 'archives': [str(p) for p in args.archive],
              'files': len(files), 'states': states,
              'declared_tolerances': {'T1_nats': 0.0, 'T2_tier2_nats': _apt.TIER2_NATS,
                                      'T2_tier2_repeat_multiple': _apt.TIER2_REPEAT_MULTIPLE},
              'T1_worst_abs_nats': t1_worst,
              'T2_worst_abs_nats': t2_worst,
              'T2_states_nonzero': int((values > 0).sum()),
              'T2_states_above_tier2_nats': int((values > _apt.TIER2_NATS).sum()),
              'archived_repeat_max_nats': float(args.repeat_max_nats),
              'tier': realised,
              'vector_to_scalar_worst_abs_nats': float(np.max(closure)),
              'passed': bool(t1_worst == 0.0 and realised in (1, 2)),
              'created_utc': datetime.now(timezone.utc).isoformat()}
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / f'gate_{args.cohort}_{args.arm}.json', record)
    print(json.dumps({k: record[k] for k in ('arm', 'cohort', 'files', 'states',
                                             'T1_worst_abs_nats', 'T2_worst_abs_nats',
                                             'T2_states_nonzero', 'tier', 'passed')}), flush=True)


if __name__ == '__main__':
    main()
