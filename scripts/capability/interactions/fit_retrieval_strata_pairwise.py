#!/usr/bin/env python3
"""Refit the measured-stability control ladder and retain its per-group errors.

The admitted pairwise fit records report group-equal squared error and its paired
reductions on the whole support and on the three sequence-separation strata, but
not the per-group values a retrieval stratification needs. This entry point runs
the same fit again and keeps them.

Nothing about the fit changes. The panel assembly, the nested comparison, the
weighting, the five outer and four inner held-group folds, the three split seeds,
the ridge grid and its tie rule and the 2,000-draw group bootstrap are all
imported from the admitted modules; this file adds no feature, no design and no
tuning choice. The run is refused unless the realised row identity, fold
identity, support counts, design widths, selected ridge penalties and
whole-support increments agree with the reference fit record, so a per-group
value can only be read as a decomposition of the number already published.

The strata declaration is bound by digest and checked for covering exactly the
fitted groups, which records that the stratification was frozen before this fit
rather than after it.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.capability.interactions.fit_pairwise_epistasis import CONTRASTS, digest, load_arm, load_profiles, build_panel
from src.capability.interactions.pairwise_epistasis import (
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, ROSTER, SPLIT_SEEDS, TOKENISATION_STRATUM, design_blocks,
    design_names, group_errors, group_spearman, interval, nested_compare)

SCHEMA = 'retrieval_strata_pairwise_refit_v1'
SUPPORT = 'all'


def basename(arm: str) -> str:
    """The one artefact name this cell writes into its output directory."""

    return f'refit_{arm}.json'


def retained_rows(plan: dict, groups: set, exclusion: dict, panel: dict) -> np.ndarray:
    """Row mask of the declared indel exclusion, aligned by replaying the panel's order.

    ``build_panel`` walks the plan's backgrounds in order and each background's
    cycles in order, so the same walk names every row. The realised background
    labels are compared elementwise rather than trusted, because a silently
    shifted mask would move an exclusion onto the wrong cycles.
    """

    if 'excluded_cycles' not in exclusion:
        raise SystemExit('the indel exclusion must be the cycle-level shape (key '
                         f"'excluded_cycles'); found keys {sorted(exclusion)}")
    mask, labels = [], []
    for background in plan['backgrounds']:
        if background['group'] not in groups:
            continue
        dropped = set(exclusion['excluded_cycles'].get(background['name'], []))
        if dropped - set(range(len(background['cycles']))):
            raise SystemExit(f'{background["name"]}: the exclusion names a cycle the plan lacks')
        for index in range(len(background['cycles'])):
            mask.append(index not in dropped)
            labels.append(background['name'])
    if labels != list(panel['background']):
        raise SystemExit('the exclusion mask is not aligned to the panel rows')
    mask = np.asarray(mask, dtype=bool)
    expected = exclusion['summary']['excluded_cycles']
    if int((~mask).sum()) != expected:
        raise SystemExit(f'the mask drops {int((~mask).sum())} rows against {expected} declared')
    return mask


def reference_checks(entry: dict, reference: dict, seeds: dict, tolerance: float) -> dict:
    """Refuse a refit that is not the reference fit's own computation."""

    support = reference['supports'][SUPPORT]
    for key in ('row_identity_sha256', 'groups', 'cycles', 'site_pairs', 'states'):
        if entry[key] != support[key]:
            raise SystemExit(f'{key} is {entry[key]!r} against {support[key]!r} in the reference fit')
    if entry['design_dimensions'] != support['design_dimensions']:
        raise SystemExit('design widths differ from the reference fit')
    deviations = []
    for seed, record in seeds.items():
        expected = support['seeds'][seed]
        if record['fold_identity_sha256'] != expected['fold_identity_sha256']:
            raise SystemExit(f'seed {seed}: held-group folds differ from the reference fit')
        realised = {fold['fold']: fold['alpha'] for fold in record['folds']}
        for fold in expected['folds']:
            if realised[fold['fold']] != fold['alpha']:
                raise SystemExit(f'seed {seed} fold {fold["fold"]}: selected ridge penalties differ')
        for name, published in expected['increments'].items():
            point = record['increments'][name]['mse_reduction_kcal2']['point']
            deviations.append((abs(point - published['mse_reduction_kcal2']['point']), seed, name))
    worst = max(deviations)
    if worst[0] > tolerance:
        raise SystemExit(f'{worst[2]} at seed {worst[1]} deviates by {worst[0]:.3e} kcal2/mol2 '
                         f'from the reference fit, above the {tolerance:.1e} tolerance')
    return {'contrasts_checked': len(deviations),
            'max_absolute_increment_deviation_kcal2': worst[0],
            'tolerance_kcal2': tolerance,
            'worst_contrast': {'seed': worst[1], 'contrast': worst[2]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--expect-plan-sha256', required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--baseline-q', type=Path, required=True)
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--extraction', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--reference-fit', type=Path, required=True)
    parser.add_argument('--strata-declaration', type=Path, required=True)
    parser.add_argument('--expect-declaration-sha256', required=True)
    parser.add_argument('--indel-exclusion', type=Path, required=True)
    parser.add_argument('--expect-indel-exclusion-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True,
                        help='output directory; the refit record is written into it by name')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--bootstrap', type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument('--reference-tolerance', type=float, default=1e-6)
    args = parser.parse_args()

    if args.arm not in ROSTER:
        raise SystemExit(f'{args.arm} is not on the frozen roster')
    from src.capability.interactions.pairwise_epistasis import plan_digest
    plan = json.loads(args.plan.read_text())
    if plan_digest(plan) != args.expect_plan_sha256:
        raise SystemExit('extraction plan digest does not match the declared value')
    cohort = json.loads(args.cohort.read_text())
    if digest(args.cohort) != plan['cohort_sha256']:
        raise SystemExit('cohort digest does not match the frozen plan')
    baseline = json.loads(args.baseline_q.read_text())
    if digest(args.baseline_q) != plan['provenance']['baseline_q']['sha256']:
        raise SystemExit('baseline Q digest does not match the frozen plan')
    declaration = json.loads(args.strata_declaration.read_text())
    if declaration['declaration_sha256'] != args.expect_declaration_sha256:
        raise SystemExit('the strata declaration digest does not match the declared value')
    exclusion = json.loads(args.indel_exclusion.read_text())
    if exclusion['exclusion_sha256'] != args.expect_indel_exclusion_sha256:
        raise SystemExit('the indel exclusion digest does not match the declared value')
    if exclusion['inputs']['cohort']['sha256'] != plan['cohort_sha256']:
        raise SystemExit('the indel exclusion was declared against a different cohort')
    reference = json.loads(args.reference_fit.read_text())
    if reference['arm'] != args.arm:
        raise SystemExit(f'the reference fit record is for {reference["arm"]}')

    profiles, profile_meta = load_profiles(args.profiles, plan)
    if profile_meta['plan_sha256'] != digest(args.plan):
        raise SystemExit('profile features were not built from this plan file')
    arm_data, manifest = load_arm(args.extraction / args.arm, args.arm, plan)
    if manifest['identity'] != reference['extraction_identity']:
        raise SystemExit('the extraction identity differs from the reference fit')

    groups = set(plan['supports']['all_groups']['groups'])
    declared = set(declaration['stability']['assignment'])
    if declared != groups:
        raise SystemExit(f'the declaration covers {len(declared)} groups against {len(groups)} fitted')
    panel = build_panel(plan, cohort, profiles, arm_data, baseline, groups)
    row_identity = hashlib.sha256('\n'.join(
        f'{g}|{p}|{e!r}' for g, p, e in zip(panel['group'], panel['site_pair'],
                                            panel['epsilon'])).encode()).hexdigest()
    epsilon, unit, pairs = panel['epsilon'], panel['group'], panel['site_pair']
    mask = retained_rows(plan, groups, exclusion, panel)

    def support_table(rows: np.ndarray) -> dict:
        """Site pairs, cycles and cycles per site pair of every group on a row subset."""

        table = {}
        for group in sorted(set(unit[rows])):
            selected = pairs[rows][unit[rows] == group]
            table[group] = {
                'site_pairs': len(set(selected)),
                'cycles': int(selected.size),
                'cycles_per_site_pair': sorted(int((selected == pair).sum())
                                               for pair in sorted(set(selected)))}
        return table

    every_row = np.ones(len(epsilon), dtype=bool)
    unfiltered_support = support_table(every_row)
    filtered_support = support_table(mask)
    filtered_groups = sorted(filtered_support)

    seeds = {}
    for seed in SPLIT_SEEDS:
        outcome = nested_compare(panel, SUPPORT, seed=seed, device=args.device)
        per_group, spearman, per_group_filtered = {}, {}, {}
        for name, prediction in outcome['predictions'].items():
            labels, values = group_errors(epsilon, prediction, unit, pairs)
            per_group[name] = dict(zip(labels.tolist(), values.tolist()))
            _, correlations = group_spearman(epsilon, prediction, unit)
            spearman[name] = dict(zip(labels.tolist(), correlations))
            kept, filtered_values = group_errors(epsilon[mask], prediction[mask],
                                                 unit[mask], pairs[mask])
            per_group_filtered[name] = dict(zip(kept.tolist(), filtered_values.tolist()))
        labels = sorted(set(unit))
        increments = {}
        for base, augmented in CONTRASTS[SUPPORT]:
            difference = [per_group[base][label] - per_group[augmented][label] for label in labels]
            filtered_difference = [per_group_filtered[base][label]
                                   - per_group_filtered[augmented][label]
                                   for label in filtered_groups]
            increments[f'{augmented}|{base}'] = {
                'per_group_mse_reduction_kcal2': dict(zip(labels, difference)),
                'mse_reduction_kcal2': interval(difference, draws=args.bootstrap,
                                                seed=BOOTSTRAP_SEED),
                'per_group_mse_reduction_kcal2_indel_filtered': dict(
                    zip(filtered_groups, filtered_difference)),
                'mse_reduction_kcal2_indel_filtered': interval(
                    filtered_difference, draws=args.bootstrap, seed=BOOTSTRAP_SEED)}
        seeds[str(seed)] = {
            'fold_identity_sha256': hashlib.sha256(json.dumps(
                [fold['held_groups'] for fold in outcome['folds']], sort_keys=True).encode()
            ).hexdigest(),
            'folds': [{key: fold[key] for key in ('fold', 'held_groups', 'alpha')}
                      for fold in outcome['folds']],
            'nuisance': [{key: record[key] for key in (
                'fold', 'selected_alpha', 'held_out_weighted_r2', 'held_out_spearman',
                'uncalibrated_additive_cycle_max_absolute')} for record in outcome['nuisance']],
            'per_group_mse_kcal2': per_group,
            'per_group_mse_kcal2_indel_filtered': per_group_filtered,
            'per_group_spearman': spearman,
            'increments': increments,
        }

    entry = {
        'row_identity_sha256': row_identity, 'groups': len(groups),
        'cycles': int(len(epsilon)), 'site_pairs': len(set(pairs)),
        'states': int(len(panel['states']['y'])),
        'design_dimensions': {
            name: int(sum(panel['blocks'][block].shape[1] for block in design_blocks(name)
                          if block != 'G') + (5 if 'G' in design_blocks(name) else 0))
            for name in design_names(SUPPORT)},
    }
    entry_support = {'per_group_support': unfiltered_support}
    record = {
        'schema': SCHEMA, 'arm': args.arm, 'support': SUPPORT,
        'tokenisation_stratum': TOKENISATION_STRATUM[args.arm],
        'qualified_batch_size': plan['provenance']['production_batch_size'][args.arm],
        'batch_size_caveat': ('the qualified batch size is recorded, not a numerical validation: '
                              'at batch size one the reference forward and the production forward '
                              'are the same computation, so the precision gate is structurally '
                              'vacuous for this arm'),
        'inputs': {
            'plan_content_sha256': args.expect_plan_sha256,
            'cohort_sha256': plan['cohort_sha256'],
            'baseline_q_sha256': plan['provenance']['baseline_q']['sha256'],
            'profiles_sha256': digest(args.profiles),
            'reference_fit_sha256': digest(args.reference_fit),
            'strata_declaration_sha256': args.expect_declaration_sha256,
            'indel_exclusion_sha256': args.expect_indel_exclusion_sha256,
        },
        'indel_exclusion': {
            'order': ('the reproduction gate runs first and on the unfiltered support, because '
                      'that is what proves this is the admitted estimator; the exclusion is then '
                      'applied on top, as an evaluation filter over the same fitted predictions, '
                      'and never before the gate'),
            'scope': exclusion['scope'],
            'excluded_cycles': exclusion['summary']['excluded_cycles'],
            'groups_lost_entirely': exclusion['summary']['groups_lost_entirely'],
            'retained_support': {
                'groups': len(filtered_groups),
                'site_pairs': sum(entry['site_pairs'] for entry in filtered_support.values()),
                'cycles': sum(entry['cycles'] for entry in filtered_support.values())},
            'per_group_support': filtered_support,
        },
        'arm_selection': (
            'this arm was selected because it carries the finding under test: the five refitted '
            'arms are exactly the five whose first-order likelihood increment resolved above zero '
            'at all three split seeds on the whole support. The stratified result therefore bounds '
            'an established positive and is not a fresh panel-wide estimate, and it cannot say '
            'whether a stratum-restricted gain appears in an arm that showed no whole-support gain'),
        'code_sha256': {str(path.relative_to(ROOT)): digest(path) for path in (
            Path(__file__), ROOT / 'scripts/capability/interactions/fit_pairwise_epistasis.py',
            ROOT / 'src/capability/interactions/pairwise_epistasis.py',
            ROOT / 'src/capability/readouts/readout_analysis.py', ROOT / 'src/capability/context/profiles.py')},
        'recipe': {'split_seeds': list(SPLIT_SEEDS), 'device': args.device,
                   'bootstrap': {'draws': args.bootstrap, 'seed': BOOTSTRAP_SEED,
                                 'unit': 'held group'},
                   'weighting': ('group equal; site pairs equal within a group; cycles equal '
                                 'within a site pair')},
        **entry, **entry_support, 'seeds': seeds,
    }
    record['reference_agreement'] = reference_checks(entry, reference, seeds,
                                                     args.reference_tolerance)
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / basename(args.arm)
    destination.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({'arm': args.arm, 'out': str(destination),
                      'groups': record['groups'], 'site_pairs': record['site_pairs'],
                      'reference_agreement': record['reference_agreement']}, indent=1))


if __name__ == '__main__':
    main()
