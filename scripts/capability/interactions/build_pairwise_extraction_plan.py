#!/usr/bin/env python3
"""Freeze the label-free extraction plan and the model roster before any scoring.

Reads the frozen cohort, verifies its digest against the value declared on the
command line, and writes an extraction plan that carries sequences and cycle state
indices only. No measured stability value, cycle epsilon or measurement row
reaches the plan, so the GPU extraction this plan drives is label-blind by
construction rather than by convention.

The roster is the full pre-declared unconditioned Readout panel. It is written
here, before extraction, and is not a function of any Readout or pairwise outcome.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.capability.interactions.pairwise_epistasis import (
    ARM_DTYPE, PRODUCTION_BATCH_SIZE, ROSTER, TOKENISATION_STRATUM,
    extraction_plan, plan_digest)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b''):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--expect-cohort-sha256', required=True)
    parser.add_argument('--baseline-q', type=Path, required=True,
                        help='fitted pairwise baseline report, read only for its admissible group list')
    parser.add_argument('--expect-baseline-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    cohort_sha256 = digest(args.cohort)
    if cohort_sha256 != args.expect_cohort_sha256:
        raise SystemExit(f'cohort digest {cohort_sha256} does not match the declared value')
    baseline_sha256 = digest(args.baseline_q)
    if baseline_sha256 != args.expect_baseline_sha256:
        raise SystemExit(f'baseline digest {baseline_sha256} does not match the declared value')

    cohort = json.loads(args.cohort.read_text())
    plan = extraction_plan(cohort, cohort_sha256)
    baseline = json.loads(args.baseline_q.read_text())
    admissible = sorted(row['name'] for row in baseline['backgrounds'] if row['status'] == 'fitted')
    names = {row['name'] for row in plan['backgrounds']}
    if not set(admissible) <= names:
        raise SystemExit('baseline Q names a background outside the frozen cohort')
    q_groups = sorted(row['group'] for row in plan['backgrounds'] if row['name'] in set(admissible))
    plan['supports'] = {
        'all_groups': {'groups': sorted(row['group'] for row in plan['backgrounds']),
                       'backgrounds': sorted(names)},
        'q_groups': {'groups': q_groups, 'backgrounds': admissible},
    }
    plan['provenance'] = {
        'cohort': {'path': str(args.cohort), 'sha256': cohort_sha256},
        'baseline_q': {'path': str(args.baseline_q), 'sha256': baseline_sha256},
        'roster_size': len(ROSTER),
        'arm_dtype': {arm: ARM_DTYPE.get(arm, 'float32') for arm in ROSTER},
        'tokenisation_stratum': {arm: TOKENISATION_STRATUM[arm] for arm in ROSTER},
        'production_batch_size': {arm: PRODUCTION_BATCH_SIZE for arm in ROSTER},
        'batch_size_rationale': (
            'Whole-panel singleton extraction. Batch composition is the only '
            'numerical failure the Readout expansion actually measured, and batch '
            'size one removes it for every arm. The 0.001 nat and 0.001 relative '
            'L2 gates are consequently inapplicable at this setting: the repeat '
            'check re-runs the identical single-row computation, so a zero is '
            'structural and certifies nothing about accuracy.'),
    }
    if len(TOKENISATION_STRATUM) != len(ROSTER) or set(TOKENISATION_STRATUM) != set(ROSTER):
        raise SystemExit('tokenisation stratum map does not cover the roster exactly')
    payload = json.dumps(plan, sort_keys=False, separators=(',', ':'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload + '\n')
    print(json.dumps({'plan_sha256': plan_digest(plan),
                      'file_sha256': digest(args.out),
                      'roster': len(ROSTER),
                      'q_groups': len(q_groups),
                      'summary': plan['summary']}, indent=1))


if __name__ == '__main__':
    main()
