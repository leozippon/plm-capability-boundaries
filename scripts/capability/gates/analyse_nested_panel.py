#!/usr/bin/env python3
"""Assemble one nested gate's per-arm fits into its panel.

Reads every per-arm fit record in a directory, checks that they were all fitted
on the same cohort, the same control qualification and the same panel rows, and
reports the likelihood and representation increments per arm with their
intervals, the interface strata, and -- where the cohort declares identity
strata -- which of the three declared outcomes each arm's close-and-remote pair
selects.

Two reporting rules are enforced rather than left to a reader.

* An arm's verdict on a cell is **above zero** only when all three per-seed
  intervals exclude zero above it, **below zero** only when all three exclude it
  below, and **unresolved** otherwise. That is a stability requirement over three
  correlated resamplings of one support, not three independent tests, so it
  carries no nominal level.
* The representation cells are reported and carry **no verdict sentence**: they
  are conditional on the compressed linear readout and the readout-class
  reassessment is still running. The field is labelled provisional in the
  artefact.

Independent units are reported as family groups and effective (Kish) counts with
the weighting convention named, never as variant counts.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.mutation.external_confirmation import SPLIT_SEEDS, require_blas_threads
from src.capability.core.io import sha256_file, write_json
from src.capability.interactions.pairwise_epistasis import ROSTER, TOKENISATION_STRATUM

SCHEMA = 'nested_gate_panel_v1'

#: The cells this panel reports, with whether a verdict sentence is licensed.
CELLS = (
    ('primary_likelihood', 'squared-error increment over the primary control set', True),
    ('primary_likelihood_spearman', 'rank increment over the primary control set', True),
    ('secondary_likelihood', 'squared-error increment over the secondary set', True),
    ('secondary_likelihood_spearman', 'rank increment over the secondary set', True),
    ('primary_representation', 'squared-error increment over the primary control set', False),
    ('primary_representation_spearman', 'rank increment over the primary control set', False),
    ('secondary_representation', 'squared-error increment over the secondary set', False),
    ('secondary_representation_spearman', 'rank increment over the secondary set', False),
)


def verdict(records: list[dict]) -> str:
    """Three-seed stability verdict on one cell."""

    if any(record.get('point') is None or record.get('interval') is None
           for record in records):
        return 'undefined'
    if all(record['excludes_zero'] and record['point'] > 0 for record in records):
        return 'above zero'
    if all(record['excludes_zero'] and record['point'] < 0 for record in records):
        return 'below zero'
    return 'unresolved'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fits', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    numeric = require_blas_threads()
    paths = sorted(args.fits.glob('fit_*.json'))
    if not paths:
        raise SystemExit(f'no fit record under {args.fits}')
    fits = {}
    for path in paths:
        record = json.loads(path.read_bytes())
        if record.get('schema') != 'nested_gate_fit_v1':
            raise SystemExit(f'{path}: unexpected fit schema')
        fits[record['arm']] = record
    unknown = sorted(set(fits) - set(ROSTER))
    if unknown:
        raise SystemExit(f'fits for arms outside the frozen roster: {unknown}')

    shared = {}
    for key in ('gate', 'endpoint', 'unit', 'cohort_sha256', 'endpoint_sha256',
                'controls_sha256', 'profile_sha256', 'row_identity_sha256',
                'qualified_control_set', 'secondary_control_set', 'rows'):
        values = {json.dumps(record[key], sort_keys=True) for record in fits.values()}
        if len(values) != 1:
            raise SystemExit(f'the fits disagree on {key}; they are not one panel')
        shared[key] = json.loads(values.pop())

    baselines = {role: sorted({fits[arm]['per_seed'][str(seed)][f'{role}_baseline_mse']['point']
                               for arm in fits for seed in SPLIT_SEEDS})
                 for role in ('primary', 'secondary')}
    tokenisation = {arm: fits[arm]['tokenisation_verdict']['qualified'] for arm in fits}

    arms = {}
    for arm, record in sorted(fits.items()):
        entry = {'interface': TOKENISATION_STRATUM[arm],
                 'dtype': record['dtype'],
                 'tokenisation_descriptors_qualified': record['tokenisation_verdict']['qualified'],
                 'matched_baseline': record['matched_baseline'],
                 'repeat_likelihood_nats_max': record['repeat_likelihood_nats_max'],
                 'repeat_feature_relative_l2_max': record['repeat_feature_relative_l2_max'],
                 'cells': {}}
        for cell, description, licensed in CELLS:
            per_seed = [record['per_seed'][str(seed)][cell] for seed in SPLIT_SEEDS]
            entry['cells'][cell] = {
                'description': description,
                'licensed_for_a_verdict': licensed,
                'per_seed': {str(seed): value for seed, value in zip(SPLIT_SEEDS, per_seed)},
                'three_seed_mean': (None if any(v['point'] is None for v in per_seed)
                                    else float(np.mean([v['point'] for v in per_seed]))),
                'verdict': verdict(per_seed) if licensed else None,
                'status': None if licensed else 'provisional: pending the readout-class '
                                                'reassessment; no verdict is drawn',
            }
        if record.get('stratified_outcome'):
            entry['stratified'] = {}
            for name in ('close', 'remote', 'mixed'):
                key = f'stratum_{name}'
                if key not in record['per_seed'][str(SPLIT_SEEDS[0])]:
                    continue
                block = record['per_seed'][str(SPLIT_SEEDS[0])][key]
                entry['stratified'][name] = {
                    'groups': block['groups'], 'rows': block['rows'],
                    'clears_the_unit_floor': block['clears_the_unit_floor'],
                    'cells': {}}
                for cell in ('primary_likelihood', 'secondary_likelihood',
                             'primary_representation', 'secondary_representation'):
                    per_seed = [record['per_seed'][str(seed)][key][cell]
                                for seed in SPLIT_SEEDS]
                    entry['stratified'][name]['cells'][cell] = {
                        'per_seed': {str(s): v for s, v in zip(SPLIT_SEEDS, per_seed)},
                        'verdict': verdict(per_seed),
                    }
            entry['outcome'] = record['stratified_outcome']['outcomes']
        arms[arm] = entry

    strata_summary = {}
    for cell, _, licensed in CELLS:
        by_interface = {}
        for interface in sorted(set(TOKENISATION_STRATUM.values())):
            members = [arm for arm in arms if arms[arm]['interface'] == interface]
            means = [arms[arm]['cells'][cell]['three_seed_mean'] for arm in members]
            means = [value for value in means if value is not None]
            verdicts = [arms[arm]['cells'][cell]['verdict'] for arm in members]
            by_interface[interface] = {
                'arms': len(members),
                'median_three_seed_mean': float(np.median(means)) if means else None,
                'positive': int(sum(1 for value in means if value > 0)),
                'above_zero': int(sum(1 for value in verdicts if value == 'above zero')),
                'below_zero': int(sum(1 for value in verdicts if value == 'below zero')),
            }
        all_means = [arms[arm]['cells'][cell]['three_seed_mean'] for arm in arms]
        all_means = [value for value in all_means if value is not None]
        strata_summary[cell] = {
            'licensed_for_a_verdict': licensed,
            'panel_median_three_seed_mean': float(np.median(all_means)) if all_means else None,
            'by_interface': by_interface,
        }

    report = {
        'schema': SCHEMA,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        **shared,
        'numeric_environment': numeric,
        'roster_arms': len(ROSTER),
        'arms_fitted': len(fits),
        'arms_missing': sorted(set(ROSTER) - set(fits)),
        'roster_complete': sorted(fits) == sorted(ROSTER),
        'effective_units': next(iter(fits.values()))['effective_units'],
        'split_seeds': list(SPLIT_SEEDS),
        'verdict_rule': ('above zero only when all three per-seed intervals exclude zero '
                         'above it, below zero only when all three exclude it below, '
                         'unresolved otherwise; a conjunction over three correlated '
                         'resamplings of one support, carrying no nominal level'),
        'multiplicity': ('declared and unadjusted across arms, split seeds, control sets, '
                         'model quantities, metrics and identity strata'),
        'representation_status': ('computed and retained, labelled provisional: conditional '
                                  'on the compressed linear readout of four pooled block '
                                  'summaries, pending the readout-class reassessment'),
        'tokenisation_descriptors_qualified': int(sum(1 for value in tokenisation.values()
                                                      if value)),
        'baseline_mse_values': baselines,
        'baseline_between_arm_spread': {
            role: (0.0 if len(values) <= len(SPLIT_SEEDS) else float(max(values) - min(values)))
            for role, values in baselines.items()},
        'arms': arms,
        'panel': strata_summary,
        'fit_digests': {arm: sha256_file(args.fits / f'fit_{arm}.json') for arm in sorted(fits)},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'panel.json', report)

    print(f"{shared['gate']}: {len(fits)} of {len(ROSTER)} arms, "
          f"complete={report['roster_complete']}")
    print(f"tokenisation descriptors qualified: {report['tokenisation_descriptors_qualified']} "
          f"of {len(fits)}")
    header = f"{'arm':24s} {'iface':9s} " + ' '.join(f'{s:>11d}' for s in SPLIT_SEEDS)
    print('\n-- likelihood, squared-error increment over the primary control set --')
    print(header + '  verdict' + ('  outcome' if 'outcome' in next(iter(arms.values())) else ''))
    for arm in sorted(arms):
        cell = arms[arm]['cells']['primary_likelihood']
        points = ' '.join(f"{cell['per_seed'][str(s)]['point']:+11.5f}" for s in SPLIT_SEEDS)
        line = f"{arm:24s} {arms[arm]['interface']:9s} {points}  {cell['verdict']}"
        if 'outcome' in arms[arm]:
            close = arms[arm]['outcome']['primary']['close_resolved']
            remote = arms[arm]['outcome']['primary']['remote_resolved']
            line += f"  close={'y' if close else 'n'} remote={'y' if remote else 'n'}"
        print(line)
    print('\n-- by interface, licensed likelihood cell --')
    for interface, value in strata_summary['primary_likelihood']['by_interface'].items():
        print(f"{interface:9s} arms={value['arms']:2d} median={value['median_three_seed_mean']} "
              f"positive={value['positive']} above={value['above_zero']} "
              f"below={value['below_zero']}")


if __name__ == '__main__':
    main()
