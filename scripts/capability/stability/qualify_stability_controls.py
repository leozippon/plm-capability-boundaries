#!/usr/bin/env python3
"""Qualify the single-mutant stability gate's controls before any model is read.

A control may enter a capability comparison only if it is a competent predictor
of held-out family groups in its own right. Each candidate is offered, in the
declared order, to the standing set that has already qualified, and it is kept
only if adding it raises — or at minimum does not lower — the standing set's
group-equal held-out mean squared error at every prespecified split seed. A
candidate that lowers it is discarded and reported, not carried forward.

No model likelihood, representation or score is read here, so the qualified set
cannot be a function of a model outcome. The four candidates are composition,
local chemistry in mutation-centred windows, mutation-local profile summaries and
a nonlinear-additive global response; every one of them is a competing
explanation for a stability change, never a mechanism.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.core.io import sha256_file, write_json
from src.capability.stability.stability_gate import (
    BASE_BLOCKS, CANDIDATE_BLOCKS, ENDPOINT, QUALIFICATION_SEEDS, build_panel,
    fold_predictions, group_errors, group_spearman, interval, load_profiles,
    paired_increment, qualify, spearman_increment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=0,
                        help='torch CPU thread count; 0 leaves the runtime default')
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    cohort = json.loads(args.cohort.read_bytes())
    if cohort.get('schema') != 'stability_singles_cohort_v1':
        raise SystemExit('unexpected cohort schema')
    wildtypes = {row['name']: row['wildtype'] for row in cohort['backgrounds']}
    profiles, profile_meta = load_profiles(args.profiles, wildtypes)
    panel = build_panel(cohort, profiles)

    ladder, standing = [], tuple(BASE_BLOCKS)
    baseline_reference = {}
    for candidate in CANDIDATE_BLOCKS:
        design = {'standing': standing, 'candidate': (*standing, candidate)}
        increments, spearman, folds, nuisance = {}, {}, {}, {}
        for seed in QUALIFICATION_SEEDS:
            outcome = fold_predictions(panel, panel['blocks'], design,
                                       seed=seed, device=args.device)
            record = paired_increment(panel, outcome['predictions'], 'candidate', 'standing')
            increments[seed] = record['point']
            spearman[seed] = spearman_increment(panel, outcome['predictions'],
                                                'candidate', 'standing')
            folds[seed] = {'alpha': [f['alpha'] for f in outcome['folds']],
                           'dimensions': outcome['folds'][0]['dimensions']}
            if candidate == 'G':
                nuisance[seed] = outcome['nuisance']
            baseline_reference[seed] = record['baseline_mse']
            if seed == QUALIFICATION_SEEDS[0]:
                interval_record = record
        verdict = qualify(increments)
        ladder.append({
            'candidate': candidate,
            'standing_set': list(standing),
            'per_seed_increment_kcal2_mol2': verdict['per_seed_increment_kcal2_mol2'],
            'min_increment_kcal2_mol2': verdict['min_increment_kcal2_mol2'],
            'first_seed_interval': interval_record,
            'spearman_increment': {str(s): spearman[s] for s in spearman},
            'folds': {str(s): folds[s] for s in folds},
            'nuisance_diagnostics': {str(s): nuisance[s] for s in nuisance},
            'qualified': verdict['qualified'],
            'disposition': 'kept in the qualified control set' if verdict['qualified']
            else 'discarded: it lowers the held-out mean squared error of the set it augments',
        })
        if verdict['qualified']:
            standing = (*standing, candidate)

    final = {}
    for seed in QUALIFICATION_SEEDS:
        outcome = fold_predictions(panel, panel['blocks'], {'S': standing},
                                   seed=seed, device=args.device)
        labels, errors = group_errors(panel['target'], outcome['predictions']['S'],
                                      panel['group'], panel['site'])
        _, null = group_errors(panel['target'], outcome['predictions']['NO_EFFECT_NULL'],
                               panel['group'], panel['site'])
        _, ranks = group_spearman(panel['target'], outcome['predictions']['S'], panel['group'])
        final[str(seed)] = {
            'qualified_set_mse_kcal2_mol2': interval(errors),
            'no_effect_null_mse_kcal2_mol2': interval(null),
            'reduction_over_no_effect_null_kcal2_mol2': interval(null - errors),
            'within_background_spearman': interval(ranks),
            'dimensions': outcome['folds'][0]['dimensions']['S'],
            'alpha': [f['alpha']['S'] for f in outcome['folds']],
        }

    report = {
        'schema': 'stability_control_qualification_v1',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'endpoint': ENDPOINT,
        'cohort_sha256': sha256_file(args.cohort),
        'endpoint_sha256': cohort['endpoint_sha256'],
        'profile_sha256': sha256_file(args.profiles),
        'profile_backgrounds_present': sum(
            1 for r in profile_meta['backgrounds'] if r['status'] == 'present'
            and r['name'] in wildtypes),
        'profile_backgrounds_absent': sorted(
            r['name'] for r in profile_meta['backgrounds']
            if r['status'] != 'present' and r['name'] in wildtypes),
        'rows': int(len(panel['target'])),
        'groups': int(len(set(panel['group']))),
        'sites': int(len(set(panel['site']))),
        'first_stage_states': int(len(panel['states']['y'])),
        'qualification_rule': (
            'a candidate is kept only if its paired reduction in group-equal held-out mean '
            'squared error over the standing set is positive at every prespecified split seed'),
        'split_seeds': list(QUALIFICATION_SEEDS),
        'base_blocks': list(BASE_BLOCKS),
        'candidate_order': list(CANDIDATE_BLOCKS),
        'ladder': ladder,
        'qualified_control_set': list(standing),
        'discarded': [row['candidate'] for row in ladder if not row['qualified']],
        'qualified_set_performance': final,
        'note': ('the nonlinear-additive response fits its own first stage on absolute '
                 'stability over the declared state descriptors, whose composition is a '
                 'separate declaration from this target-level qualification'),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'controls_qualification.json', report)
    print(json.dumps({'qualified_control_set': report['qualified_control_set'],
                      'discarded': report['discarded'],
                      'ladder': [{k: row[k] for k in
                                  ('candidate', 'per_seed_increment_kcal2_mol2', 'qualified')}
                                 for row in ladder],
                      'qualified_set_performance': {
                          s: {'mse': v['qualified_set_mse_kcal2_mol2']['point'],
                              'null': v['no_effect_null_mse_kcal2_mol2']['point'],
                              'spearman': v['within_background_spearman']['point']}
                          for s, v in final.items()}}, indent=1))


if __name__ == '__main__':
    main()
