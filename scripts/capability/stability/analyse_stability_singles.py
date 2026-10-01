#!/usr/bin/env python3
"""Aggregate the per-arm single-mutant stability fits into the panel report.

It reads records that already exist and performs no fit and no draw of its own.
Two conditions are checked across arms rather than asserted: every arm must
carry the identical cohort digest, endpoint digest, control-qualification digest,
row count, group count and site count; and the qualified control set's own
held-out mean squared error, which contains no arm-specific column, must be
identical across every arm at one split seed. A spread above zero there means the
arms were not compared on one support.

An arm counts as resolved only when all three per-seed intervals exclude zero in
the same direction. No interval is adjusted for the panel, the seeds or the two
contrasts, and the counts below are not a model-population prevalence.
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
from src.capability.interactions.pairwise_epistasis import ROSTER, TOKENISATION_STRATUM
from src.capability.stability.stability_gate import ENDPOINT, SPLIT_SEEDS

IDENTITY_FIELDS = ('cohort_sha256', 'endpoint_sha256', 'controls_sha256', 'profile_sha256',
                   'qualified_control_set', 'secondary_control_set', 'rows', 'groups', 'sites',
                   'split_seeds')

#: Every reported contrast: the model quantity, the control set it is read over and
#: the metric. A rule licenses the cell on its own metric: the mean-squared-error
#: rule qualified the primary set, and the correlation rule qualified the secondary
#: one, so the primary set's squared-error cells and the secondary set's correlation
#: cells are the licensed ones and the other two are sensitivities.
CONTRASTS = tuple(f'{role}_{quantity}{metric}'
                  for role in ('primary', 'secondary')
                  for quantity in ('likelihood', 'representation')
                  for metric in ('', '_spearman'))
LICENSED = ('primary_likelihood', 'primary_representation',
            'secondary_likelihood_spearman', 'secondary_representation_spearman')


def resolved(records: list[dict]) -> str:
    """Direction in which all three per-seed intervals exclude zero, or 'unresolved'."""

    if all(r['interval'] is not None and r['interval'][0] > 0 for r in records):
        return 'above_zero'
    if all(r['interval'] is not None and r['interval'][1] < 0 for r in records):
        return 'below_zero'
    return 'unresolved'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fits', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    records = {}
    for arm in ROSTER:
        path = args.fits / f'fit_{arm}.json'
        if path.exists():
            records[arm] = json.loads(path.read_bytes())
    if not records:
        raise SystemExit('no per-arm fit record was found')
    missing = [arm for arm in ROSTER if arm not in records]

    identity = {field: {json.dumps(record[field], sort_keys=True) for record in records.values()}
                for field in IDENTITY_FIELDS}
    disagreeing = {field: sorted(values) for field, values in identity.items()
                   if len(values) != 1}
    if disagreeing:
        raise SystemExit(f'arms do not share one support: {disagreeing}')

    baseline_spread = {}
    for seed in SPLIT_SEEDS:
        entry = {'arms_with_tokenisation_in_baseline': sum(
            1 for r in records.values() if r['tokenisation_verdict']['qualified'])}
        for role in ('primary', 'secondary'):
            values = [record['primary'][str(seed)][f'{role}_baseline_mse_kcal2_mol2']['point']
                      for record in records.values()
                      if not record['tokenisation_verdict']['qualified']]
            entry[role] = {
                'arms_without_arm_specific_columns': len(values),
                'mse_spread_kcal2_mol2': float(max(values) - min(values)) if values else None,
                'mse_kcal2_mol2': float(values[0]) if values else None}
        baseline_spread[str(seed)] = entry

    panel = {}
    for contrast in CONTRASTS:
        rows = {}
        for arm, record in records.items():
            per_seed = [record['primary'][str(seed)][contrast] for seed in SPLIT_SEEDS]
            rows[arm] = {
                'stratum': TOKENISATION_STRATUM[arm],
                'matched_baseline': record['matched_baseline'][
                    'primary' if contrast.startswith('primary') else 'secondary'],
                'per_seed_point': [r['point'] for r in per_seed],
                'per_seed_interval': [r['interval'] for r in per_seed],
                'seed_mean': float(np.mean([r['point'] for r in per_seed])),
                'verdict': resolved(per_seed),
                'remote': {
                    label: {
                        'per_seed_point': [
                            record['remote_stratification'][label][str(seed)][contrast]['point']
                            for seed in SPLIT_SEEDS],
                        'verdict': resolved([
                            record['remote_stratification'][label][str(seed)][contrast]
                            for seed in SPLIT_SEEDS]),
                        'purged_training_groups_per_fold': record['remote_stratification'][
                            label][str(SPLIT_SEEDS[0])]['purged_training_groups_per_fold'],
                    }
                    for label in record['remote_stratification']},
            }
        strata = {}
        for stratum in sorted({TOKENISATION_STRATUM[arm] for arm in rows}):
            members = [row for row in rows.values() if row['stratum'] == stratum]
            strata[stratum] = {
                'arms': len(members),
                'median_seed_mean': float(np.median(
                    [row['seed_mean'] for row in members])),
                'arms_with_positive_seed_mean': sum(
                    1 for row in members if row['seed_mean'] > 0),
                'arms_resolved_above_zero': sum(
                    1 for row in members if row['verdict'] == 'above_zero'),
                'arms_resolved_below_zero': sum(
                    1 for row in members if row['verdict'] == 'below_zero'),
            }
        panel[contrast] = {
            'licensed': contrast in LICENSED,
            'metric': 'group-equal mean squared error in kcal2/mol2' if not contrast.endswith(
                '_spearman') else 'within-background Spearman, dimensionless',
            'arms': rows,
            'panel_median_seed_mean_kcal2_mol2': float(np.median(
                [row['seed_mean'] for row in rows.values()])),
            'resolved_above_zero': sorted(a for a, r in rows.items() if r['verdict'] == 'above_zero'),
            'resolved_below_zero': sorted(a for a, r in rows.items() if r['verdict'] == 'below_zero'),
            'strata': strata,
        }

    sample = next(iter(records.values()))
    report = {
        'schema': 'stability_singles_panel_v1',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'endpoint': ENDPOINT,
        'arms_reported': sorted(records),
        'arms_missing': missing,
        'support': {field: sample[field] for field in IDENTITY_FIELDS},
        'discarded_controls': sample['discarded_controls'],
        'no_effect_null_mse_kcal2_mol2': {
            str(seed): sample['primary'][str(seed)]['no_effect_null_mse_kcal2_mol2']
            for seed in SPLIT_SEEDS},
        'secondary_control_derivation': sample['secondary_control_derivation'],
        'licensed_contrasts': list(LICENSED),
        'baseline_identity': baseline_spread,
        'tokenisation_verdicts': {arm: record['tokenisation_verdict']['qualified']
                                  for arm, record in sorted(records.items())},
        'repeat_reproducibility': {
            'max_likelihood_nats': max(r['repeat_likelihood_nats_max'] for r in records.values()),
            'max_feature_relative_l2': max(r['repeat_feature_relative_l2_max']
                                           for r in records.values()),
            'reading': sample['precision_gate'],
        },
        'permutation_check': {arm: record['permutation_check']
                              for arm, record in sorted(records.items())
                              if record['permutation_check']},
        'panel': panel,
        'multiplicity': ('intervals condition on the fitted cross-validation predictions, omit '
                         'training and split variation, and are not adjusted for 33 arms, three '
                         'split seeds, two control sets, two model quantities, two metrics or '
                         'two remote thresholds; the licensed cells alone are 33 arms times two '
                         'model quantities, and the pattern across arms rather than any single '
                         "arm's interval is the defensible reading"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'panel.json', report)
    print(json.dumps({
        'arms_reported': len(records), 'arms_missing': missing,
        'discarded_controls': report['discarded_controls'],
        'baseline_identity': baseline_spread,
        'resolved': {c: {'above_zero': panel[c]['resolved_above_zero'],
                         'below_zero': panel[c]['resolved_below_zero'],
                         'licensed': panel[c]['licensed']} for c in CONTRASTS},
        'panel_medians': {c: panel[c]['panel_median_seed_mean_kcal2_mol2'] for c in CONTRASTS},
        'strata': {c: panel[c]['strata'] for c in LICENSED},
    }, indent=1))


if __name__ == '__main__':
    main()
