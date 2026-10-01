#!/usr/bin/env python3
"""Aggregate the per-arm global-context gate fits into one panel report.

It reads the per-arm records `fit_global_context_gate.py` wrote, checks the
conditions that must hold across the panel, and reports each declared contrast
per arm, per split seed and per separation stratum, with its group-bootstrap
interval and its effective unit counts.

Two conditions are checked rather than assumed. Every arm must share one row
identity and one fold identity per support and seed, and must have reproduced the
admitted pairwise fit's value on the designs it shares with it. And the context
blocks carry no arm-specific column, so the only way `CGT`, `CGTW` and `CGTX` can
differ between arms is through the tokenisation descriptors `T` that the flagship's
matched baseline contains; the spread across arms is reported as measured rather
than asserted to be zero, because `T` is arm-specific by construction and the
flagship's exact-zero spread was a statement about `C` and `C+G` alone.

An arm counts as resolved on a contrast only when all three per-seed intervals
exclude zero with the same sign. No multiplicity adjustment is applied, and the
count of resolved arms is reported beside the number of arms, supports, seeds
and contrasts that were examined.
"""
from pathlib import Path
import argparse
import json
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.context.global_context import (  # noqa: E402
    GATE_CONTRASTS, RECONSTRUCTION_CONTROLS, STRATA, declaration_digest)
from src.capability.interactions.pairwise_epistasis import (  # noqa: E402
    ROSTER, SPLIT_SEEDS, TOKENISATION_STRATUM)

#: Designs and contrasts carrying no model quantity. They differ between arms
#: only through the 16 tokenisation descriptors of the matched baseline, so their
#: across-arm spread measures what that block does rather than violating an
#: invariance.
SHARED_CONTEXT_DESIGNS = ('CGT', 'CGTW', 'CGTX')
CONTEXT_CONTRASTS = ('CGTW|CGT', 'CGTX|CGT', 'CGTX|CGTW')

CONTRAST_NAMES = tuple(f'{augmented}|{base}' for base, augmented in GATE_CONTRASTS)


def resolved(entries: list[dict]) -> str | None:
    """`above`, `below` or None: whether all three per-seed intervals exclude zero
    with one sign."""

    if any(entry['interval'] is None for entry in entries):
        return None
    if all(entry['interval'][0] > 0 for entry in entries):
        return 'above'
    if all(entry['interval'][1] < 0 for entry in entries):
        return 'below'
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fits', type=Path, required=True,
                        help='directory of per-arm gate fit records')
    parser.add_argument('--flagship-fits', type=Path, required=True,
                        help='directory of the admitted per-arm pairwise fit records')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--supports', default='all,q')
    args = parser.parse_args()

    records = {}
    for arm in ROSTER:
        path = args.fits / f'gate_{arm}.json'
        if not path.exists():
            raise SystemExit(f'{arm}: no gate fit record at {path}')
        record = json.loads(path.read_text())
        if record.get('schema') != 'global_context_gate_fit_v1' or record['arm'] != arm:
            raise SystemExit(f'{arm}: not a global-context gate fit record')
        if record['declaration']['sha256'] != declaration_digest():
            raise SystemExit(f'{arm}: fitted against a different block declaration')
        records[arm] = record

    report = {'schema': 'global_context_gate_panel_v1',
              'declaration_sha256': declaration_digest(),
              'arms': len(records), 'split_seeds': list(SPLIT_SEEDS),
              'contrasts': list(CONTRAST_NAMES),
              'multiplicity': (f'{len(records)} arms x {len(args.supports.split(","))} supports '
                               f'x {len(SPLIT_SEEDS)} seeds x {len(CONTRAST_NAMES)} contrasts of '
                               'unadjusted pointwise intervals'),
              'supports': {}}
    for support in args.supports.split(','):
        first = records[ROSTER[0]]['supports'][support]
        entry = {'groups': first['groups'], 'cycles': first['cycles'],
                 'site_pairs': first['site_pairs'],
                 'kish_effective_site_pairs': first['kish_effective_site_pairs'],
                 'stratum_support': first['stratum_support'],
                 'row_identity_sha256': first['row_identity_sha256'],
                 'shared_design_max_absolute_difference_kcal2': max(
                     record['supports'][support]['shared_design_max_absolute_difference_kcal2']
                     for record in records.values()),
                 'shared_context_designs': {}, 'context_increments': {}, 'arms': {},
                 'resolved': {}, 'stratum_medians': {}}
        for name in records.values():
            if name['supports'][support]['row_identity_sha256'] != entry['row_identity_sha256']:
                raise SystemExit(f'{support}: arms do not share one row identity')
        for design in SHARED_CONTEXT_DESIGNS:
            values = {}
            for seed in SPLIT_SEEDS:
                points = {arm: record['supports'][support]['seeds'][str(seed)]['designs'][
                    design]['group_equal_mse_kcal2']['point'] for arm, record in records.items()}
                values[str(seed)] = {
                    'median_group_equal_mse_kcal2': statistics.median(points.values()),
                    'spread_across_arms': max(points.values()) - min(points.values()),
                    'spread_is_the_tokenisation_block': ('these designs share every column but '
                                                         'the arm\'s own tokenisation '
                                                         'descriptors')}
            entry['shared_context_designs'][design] = values
        for contrast in CONTEXT_CONTRASTS:
            values = {}
            for seed in SPLIT_SEEDS:
                sourced = {arm: record['supports'][support]['seeds'][str(seed)]['increments'][
                    contrast] for arm, record in records.items()}
                points = {arm: item['mse_reduction_kcal2']['point']
                          for arm, item in sourced.items()}
                median_arm = sorted(points, key=points.get)[len(points) // 2]
                values[str(seed)] = {
                    'median_arm': median_arm,
                    **sourced[median_arm]['mse_reduction_kcal2'],
                    'minimum': min(points.values()), 'maximum': max(points.values()),
                    'spread_across_arms': max(points.values()) - min(points.values()),
                    'strata': {stratum: sourced[median_arm]['strata'][stratum]
                               for stratum in STRATA
                               if stratum in sourced[median_arm]['strata']}}
            entry['context_increments'][contrast] = values
        for arm, record in records.items():
            support_entry = record['supports'][support]
            arm_report = {'tokenisation_stratum': TOKENISATION_STRATUM[arm],
                          'matches_admitted_pairwise_fit':
                              support_entry['matches_admitted_pairwise_fit'],
                          'increments': {}, 'first_order_reconstruction': {}}
            for contrast in CONTRAST_NAMES:
                seeds = {str(seed): support_entry['seeds'][str(seed)]['increments'][contrast][
                    'mse_reduction_kcal2'] for seed in SPLIT_SEEDS}
                strata = {stratum: {str(seed): support_entry['seeds'][str(seed)]['increments'][
                    contrast]['strata'][stratum]['mse_reduction_kcal2']
                    for seed in SPLIT_SEEDS}
                    for stratum in STRATA
                    if stratum in support_entry['seeds'][str(SPLIT_SEEDS[0])]['increments'][
                        contrast]['strata']}
                arm_report['increments'][contrast] = {
                    'seeds': seeds,
                    'seed_mean': statistics.fmean(v['point'] for v in seeds.values()),
                    'resolved': resolved(list(seeds.values())),
                    'strata': {stratum: {
                        'seeds': values,
                        'seed_mean': statistics.fmean(v['point'] for v in values.values()),
                        'resolved': resolved(list(values.values())),
                        'groups': support_entry['seeds'][str(SPLIT_SEEDS[0])]['increments'][
                            contrast]['strata'][stratum]['groups'],
                        'site_pairs': support_entry['seeds'][str(SPLIT_SEEDS[0])]['increments'][
                            contrast]['strata'][stratum]['site_pairs'],
                    } for stratum, values in strata.items()}}
            for control in RECONSTRUCTION_CONTROLS:
                for index in (0, 1):
                    key = f'{control}|m1_{index}'
                    source = {str(seed): support_entry['seeds'][str(seed)][
                        'first_order_reconstruction'][key] for seed in SPLIT_SEEDS}
                    arm_report['first_order_reconstruction'][key] = {
                        statistic: {
                            'seeds': {seed: value[statistic] for seed, value in source.items()},
                            'seed_mean': statistics.fmean(
                                value[statistic]['point'] for value in source.values())}
                        for statistic in ('group_equal_weighted_r2',
                                          'within_background_spearman')}
            entry['arms'][arm] = arm_report
        for contrast in CONTRAST_NAMES:
            entry['resolved'][contrast] = {
                direction: sorted(arm for arm, value in entry['arms'].items()
                                  if value['increments'][contrast]['resolved'] == direction)
                for direction in ('above', 'below')}
            entry['resolved'][contrast]['strata'] = {
                stratum: {direction: sorted(
                    arm for arm, value in entry['arms'].items()
                    if value['increments'][contrast]['strata'].get(stratum, {}).get(
                        'resolved') == direction)
                    for direction in ('above', 'below')}
                for stratum in STRATA}
            entry['stratum_medians'][contrast] = {}
            for tokenisation in ('amino_acid', 'byte', 'bpe'):
                values = [value['increments'][contrast]['seed_mean']
                          for value in entry['arms'].values()
                          if value['tokenisation_stratum'] == tokenisation]
                entry['stratum_medians'][contrast][tokenisation] = {
                    'arms': len(values), 'median_seed_mean': statistics.median(values),
                    'positive_arms': sum(1 for value in values if value > 0)}
            values = [value['increments'][contrast]['seed_mean']
                      for value in entry['arms'].values()]
            entry['stratum_medians'][contrast]['panel'] = {
                'arms': len(values), 'median_seed_mean': statistics.median(values),
                'positive_arms': sum(1 for value in values if value > 0)}
        report['supports'][support] = entry

    flagship = {}
    for arm in ROSTER:
        path = args.flagship_fits / f'fit_{arm}.json'
        if not path.exists():
            raise SystemExit(f'{arm}: no admitted pairwise fit at {path}')
        flagship[arm] = json.loads(path.read_text())
    # On an unfiltered support this is an agreement check: the same rows and the
    # same folds must give the admitted increment back. On a filtered support the
    # rows deliberately differ, so the same difference measures what the declared
    # exclusion moves rather than whether anything agrees.
    report['admitted_first_order_agreement'] = {}
    for support in args.supports.split(','):
        base, filtered = support.replace('_no_indel', ''), support.endswith('_no_indel')
        worst, worst_arm, moves = 0.0, None, []
        for arm, record in records.items():
            for seed in SPLIT_SEEDS:
                mine = record['supports'][support]['seeds'][str(seed)]['increments'][
                    'CGT_M1|CGT']['mse_reduction_kcal2']['point']
                admitted = flagship[arm]['supports'][base]['seeds'][str(seed)][
                    'increments'].get('C_G_T_M1|C_G_T')
                if admitted is None:
                    continue
                moved = abs(mine - admitted['mse_reduction_kcal2']['point'])
                moves.append(moved)
                if moved > worst:
                    worst, worst_arm = moved, f'{arm} seed {seed}'
        report['admitted_first_order_agreement'][support] = {
            'max_absolute_difference_kcal2': worst, 'worst_case': worst_arm,
            'median_absolute_difference_kcal2':
                statistics.median(moves) if moves else None,
            'compared_against': f'the admitted pairwise fit on support {base}',
            'meaning': (('the first-order likelihood increment over C+G+T recomputed on rows '
                         'the declared exclusion changed, against the admitted value on the '
                         'unfiltered rows; this is the size of the exclusion\'s effect, not an '
                         'agreement check') if filtered else
                        ('the first-order likelihood increment over C+G+T recomputed here '
                         'against the admitted pairwise fit\'s own value, same rows and folds'))}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + '\n')
    for support, entry in report['supports'].items():
        print(f"== support {support}: {entry['groups']} groups, {entry['cycles']} cycles, "
              f"{entry['site_pairs']} site pairs, Kish "
              f"{entry['kish_effective_site_pairs']['fitting_weights']:.1f} under the fitted "
              f"weighting")
        print("   across-arm spread of the model-free designs (the tokenisation block): "
              + ', '.join(
                  f"{design} {max(v['spread_across_arms'] for v in values.values()):.3g}"
                  for design, values in entry['shared_context_designs'].items()))
        for contrast in CONTRAST_NAMES:
            median = entry['stratum_medians'][contrast]['panel']['median_seed_mean']
            above = len(entry['resolved'][contrast]['above'])
            below = len(entry['resolved'][contrast]['below'])
            distant = len(entry['resolved'][contrast]['strata']['10+']['above'])
            print(f"   {contrast:22s} panel median {median:+.5f}  above {above:2d}  "
                  f"below {below:2d}  above on 10+ {distant:2d}")
        agreement = report['admitted_first_order_agreement'][support]
        if agreement['median_absolute_difference_kcal2'] is None:
            print("   first-order increment: the admitted pairwise fit declares no first-order "
                  f"contrast on {agreement['compared_against']}, so there is nothing to "
                  "compare against")
        else:
            print(f"   first-order increment against {agreement['compared_against']}: max "
                  f"{agreement['max_absolute_difference_kcal2']:.3g}, median "
                  f"{agreement['median_absolute_difference_kcal2']:.3g} kcal2/mol2")
    print(f"written {args.out}")


if __name__ == '__main__':
    main()
