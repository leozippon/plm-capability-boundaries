#!/usr/bin/env python3
"""Aggregate the per-arm pairwise-epistasis fits into the panel report.

Reads every completed per-arm record, checks the conditions that must hold across
the panel, and emits the cross-arm tables: the control ladder, the likelihood and
representation increments over each control set, the same increments over the
first-order controls, the Q-inclusive comparison on its own support, and the
tokenisation strata.

Two panel-level checks run here rather than inside a single arm's fit. The
sequence/profile control C and the nonlinear-additive control C+G contain no
arm-specific column, so their group-equal squared error must be identical for
every arm at one seed; a spread beyond floating-point reproducibility means the
supports or folds were not shared. And the cohort's independent-site double
contrast must read as its construction zero in every record.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.interactions.pairwise_epistasis import ROSTER, SPLIT_SEEDS, TOKENISATION_STRATUM

#: Contrasts carried into the panel tables, with the question each one answers.
HEADLINE = {
    'all': (
        ('C|ADDITIVE_NULL', 'sequence/profile control over the zero-interaction additive null'),
        ('C_G|C', 'nonlinear-additive nuisance over the sequence/profile control'),
        ('C_G_T|C_G', 'tokenisation interface descriptors over the nuisance control'),
        ('C_G_T+M|C_G_T', 'adding model likelihood to the matched baseline'),
        ('C_G_T+R|C_G_T', 'adding the model representation contrast to the matched baseline'),
        ('C_G_T+MR|C_G_T', 'adding both'),
        ('C_G_T_M1|C_G_T', 'first-order likelihood differences alone'),
        ('C_G_T_M1+M|C_G_T_M1', 'likelihood interaction over its own first-order control'),
        ('C_G_T_R1|C_G_T', 'first-order representation differences alone'),
        ('C_G_T_R1+R|C_G_T_R1', 'representation interaction over its own first-order control'),
    ),
    'q': (
        ('C_G_T_Q|C_G_T', 'pairwise sequence baseline Q over the matched baseline'),
        ('C_G_T+M|C_G_T', 'likelihood before Q, on the Q support'),
        ('C_G_T+R|C_G_T', 'representation before Q, on the Q support'),
        ('C_G_T_Q+M|C_G_T_Q', 'likelihood after Q'),
        ('C_G_T_Q+R|C_G_T_Q', 'representation after Q'),
        ('C_G_T_Q+MR|C_G_T_Q', 'both after Q'),
    ),
}
REPRODUCIBILITY_TOLERANCE = 1e-6


def summarise_seeds(record: dict, support: str, contrast: str, *, restricted: bool = False) -> dict:
    """Point estimate and conditional interval per seed, plus the across-seed range.

    With ``restricted``, the increments are read from the evaluation that drops the
    cycles resting on an insertion or deletion construct. The fits, folds, penalty
    selections and held-out predictions behind both readings are the same.
    """

    entry = record['supports'][support]
    per_seed = {}
    for seed in SPLIT_SEEDS:
        source = entry['seeds'][str(seed)]
        if restricted:
            source = source.get('restricted_evaluation')
            if source is None:
                raise SystemExit(f"{record['arm']}: no restricted evaluation in this record")
        block = source['increments'].get(contrast)
        if block is None:
            return {}
        mse = block['mse_reduction_kcal2']
        per_seed[str(seed)] = {
            'mse_reduction_kcal2': mse['point'], 'interval': mse['interval'],
            'excludes_zero': mse['excludes_zero'],
            'spearman_increment': block['spearman_increment'].get('point'),
            'spearman_interval': block['spearman_increment'].get('interval'),
            'distant_stratum': block['strata'].get('10+', {}).get('mse_reduction_kcal2', {}),
        }
    points = [value['mse_reduction_kcal2'] for value in per_seed.values()]
    return {'per_seed': per_seed, 'point_range': [min(points), max(points)],
            'seeds_with_interval_above_zero': sum(
                1 for value in per_seed.values()
                if value['excludes_zero'] and value['mse_reduction_kcal2'] > 0),
            'seeds_with_interval_below_zero': sum(
                1 for value in per_seed.values()
                if value['excludes_zero'] and value['mse_reduction_kcal2'] < 0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fits', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--use-restricted-evaluation', action='store_true',
                        help='read the evaluation that drops cycles resting on an indel construct')
    args = parser.parse_args()

    records = {}
    for path in sorted(args.fits.glob('fit_*.json')):
        record = json.loads(path.read_text())
        records[record['arm']] = record
    if not records:
        raise SystemExit(f'no per-arm record under {args.fits}')
    missing = sorted(set(ROSTER) - set(records))

    supports, checks = {}, {}
    reference = next(iter(records.values()))
    for support in ('all', 'q'):
        entry = reference['supports'][support]
        identities = {arm: r['supports'][support]['row_identity_sha256'] for arm, r in records.items()}
        folds = {arm: {seed: r['supports'][support]['seeds'][seed]['fold_identity_sha256']
                       for seed in r['supports'][support]['seeds']} for arm, r in records.items()}
        supports[support] = {
            'groups': entry['groups'], 'cycles': entry['cycles'],
            'site_pairs': entry['site_pairs'], 'states': entry['states'],
            'row_identity_sha256': entry['row_identity_sha256'],
            'independent_site_double_contrast_max_absolute':
                entry['independent_site_double_contrast_max_absolute'],
            'block_dimensions': entry['block_dimensions'],
            'design_dimensions': entry['design_dimensions'],
        }
        checks[support] = {
            'row_identity_shared_by_every_arm': len(set(identities.values())) == 1,
            'fold_identity_shared_by_every_arm': all(
                len({folds[arm][seed] for arm in folds}) == 1 for seed in folds[reference['arm']]),
        }
        for design in ('C', 'C_G'):
            spread = {}
            for seed in SPLIT_SEEDS:
                def designs_of(record):
                    node = record['supports'][support]['seeds'][str(seed)]
                    return (node['restricted_evaluation'] if args.use_restricted_evaluation
                            else node)['designs']
                values = [designs_of(r)[design]['group_equal_mse_kcal2']['point']
                          for r in records.values() if design in designs_of(r)]
                if values:
                    spread[str(seed)] = {'point': values[0],
                                         'max_absolute_spread_across_arms': max(values) - min(values)}
            if spread:
                checks[support][f'arm_independent_design_{design}'] = spread

    panel = {}
    for support, contrasts in HEADLINE.items():
        panel[support] = {}
        for contrast, question in contrasts:
            rows = {}
            for arm, record in sorted(records.items()):
                if support not in record['supports']:
                    continue
                summary = summarise_seeds(record, support, contrast,
                                          restricted=args.use_restricted_evaluation)
                if summary:
                    rows[arm] = dict(summary, stratum=TOKENISATION_STRATUM[arm])
            if not rows:
                continue
            points = {arm: float(np.mean([v['mse_reduction_kcal2']
                                          for v in row['per_seed'].values()]))
                      for arm, row in rows.items()}
            ordered = sorted(points, key=lambda arm: -points[arm])
            by_stratum = {}
            for stratum in ('amino_acid', 'byte', 'bpe'):
                members = [points[arm] for arm in points if TOKENISATION_STRATUM[arm] == stratum]
                if members:
                    by_stratum[stratum] = {
                        'arms': len(members), 'median': float(np.median(members)),
                        'min': min(members), 'max': max(members),
                        'arms_positive': sum(1 for value in members if value > 0)}
            panel[support][contrast] = {
                'question': question, 'arms': rows,
                'seed_mean_ranking': [{'arm': arm, 'seed_mean_mse_reduction_kcal2': points[arm],
                                       'stratum': TOKENISATION_STRATUM[arm]} for arm in ordered],
                'arms_with_every_seed_interval_above_zero': [
                    arm for arm, row in rows.items() if row['seeds_with_interval_above_zero'] == 3],
                'arms_with_every_seed_interval_below_zero': [
                    arm for arm, row in rows.items() if row['seeds_with_interval_below_zero'] == 3],
                'by_tokenisation_stratum': by_stratum}

    nuisance = {}
    for support in ('all', 'q'):
        values = [fold['held_out_weighted_r2'] for seed in reference['supports'][support]['seeds'].values()
                  for fold in seed['nuisance'] if fold['held_out_weighted_r2'] is not None]
        spearman = [fold['held_out_spearman'] for seed in reference['supports'][support]['seeds'].values()
                    for fold in seed['nuisance'] if fold['held_out_spearman'] is not None]
        response = [fold['calibration_response_range_kcal_mol']
                    for seed in reference['supports'][support]['seeds'].values()
                    for fold in seed['nuisance']]
        uncalibrated = [fold['uncalibrated_additive_cycle_max_absolute']
                        for seed in reference['supports'][support]['seeds'].values()
                        for fold in seed['nuisance']]
        nuisance[support] = {
            'folds': len(response),
            'first_stage_held_out_weighted_r2': [min(values), float(np.median(values)), max(values)],
            'first_stage_held_out_spearman': [min(spearman), float(np.median(spearman)), max(spearman)],
            'calibration_response_range_kcal_mol': [min(response), float(np.median(response)), max(response)],
            'uncalibrated_additive_cycle_max_absolute': max(uncalibrated),
        }

    repeat = {arm: record['extraction_repeat_maxima'] for arm, record in sorted(records.items())}
    report = {
        'schema': 'pairwise_epistasis_panel_v1',
        'evaluation': ('cycles resting on an insertion or deletion construct excluded'
                       if args.use_restricted_evaluation else 'every admitted cycle'),
        'restricted_evaluation_support': ({
            support: {key: reference['supports'][support]['seeds'][str(SPLIT_SEEDS[0])]
                      ['restricted_evaluation'][key]
                      for key in ('groups', 'cycles', 'site_pairs', 'dropped_cycles', 'dropped_groups')}
            for support in ('all', 'q')} if args.use_restricted_evaluation else None),
        'excluded_evaluation_cycles': reference.get('excluded_evaluation_cycles'),
        'arms_reported': sorted(records), 'arms_missing': missing,
        'roster_size': len(ROSTER),
        'tokenisation_strata': {stratum: sorted(
            arm for arm in records if TOKENISATION_STRATUM[arm] == stratum)
            for stratum in ('amino_acid', 'byte', 'bpe')},
        'supports': supports, 'panel_checks': checks, 'nuisance_diagnostics': nuisance,
        'extraction_repeat_maxima': {
            'likelihood_nats': max(v['likelihood_nats'] for v in repeat.values()),
            'projected_feature_relative_l2': max(
                v['projected_feature_relative_l2'] for v in repeat.values()),
            'per_arm': repeat},
        'recipe': reference['recipe'],
        'multiplicity': ('conditional pointwise 95% group-bootstrap intervals, not adjusted for '
                         f'{len(records)} arms, two supports, three split seeds or the reported contrasts'),
        'panel': panel,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({'arms_reported': len(records), 'arms_missing': missing,
                      'checks': checks, 'nuisance': nuisance}, indent=1))


if __name__ == '__main__':
    main()
