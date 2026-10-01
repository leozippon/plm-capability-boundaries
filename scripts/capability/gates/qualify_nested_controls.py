#!/usr/bin/env python3
"""Qualify one nested gate's controls before any model quantity is read.

This entry point serves both cohorts the nested machinery fits -- the Domainome
external confirmation and the MGnify remote-homology gate -- because the
weighting, the folds, the ridge recipe, the candidate order and the qualification
rule are one declaration shared by the two, and only the endpoint's unit and its
strata differ. The cohort's own schema names which it is.

A control may enter a capability comparison only if it is a competent predictor
of held-out family groups in its own right. Each candidate is offered, in the
declared order, to the standing set that has already qualified, and it is kept
only if its paired reduction in group-equal held-out mean squared error over that
standing set is positive at every prespecified split seed. A candidate that
lowers it is discarded and reported, never carried forward.

**Each candidate's own contribution is reported with an interval, and the
qualified set's own contribution over the no-effect null is reported with one
too.** That is not decoration: a control whose own contribution is
indistinguishable from zero makes any increment measured over it uninformative,
which is the failure the limitations catalogue records as L48 with four
instances. A block kept by the three-seed sign rule whose own interval contains
zero is reported as exactly that.

No model likelihood, representation or score is read here, so the qualified set
cannot be a function of a model outcome.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.mutation.external_confirmation import (
    BASE_BLOCKS, CANDIDATE_BLOCKS, G_FEATURE_ORDER, SPLIT_SEEDS, build_panel,
    fold_predictions, group_errors, interval, kish_units, load_cohort, paired_increment,
    qualify, raw_spearman, require_blas_threads, row_identity, spearman_increment)
from src.capability.core.io import sha256_file, write_json
from src.capability.stability.stability_gate import load_profiles

SCHEMA = 'nested_gate_control_qualification_v1'

#: The first-stage column set of the nonlinear-additive response G. It is the
#: standing set at the point G is offered, which on this endpoint is a one-estimate
#: level rather than a two-estimate difference: there is no second measured state,
#: so the isotonic response is applied to the predicted effect itself.
G_FIRST_STAGE_NOTE = (
    'the first stage is a ridge predictor of the endpoint over the standing columns, fitted '
    'on the outer training groups only, its penalty selected on the same inner held-group '
    'partition as the outer tuning and its training rows cross-fitted on that partition; the '
    'isotonic response is then applied to the predicted effect, because this endpoint has no '
    'second measured state for it to be applied to')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    numeric = require_blas_threads()
    import torch
    torch.set_num_threads(numeric['pinned_threads'])

    loaded = load_cohort(args.cohort)
    units = loaded['units']
    wildtypes = {row['name']: row['wildtype'] for row in units}
    profiles, profile_meta = load_profiles(args.profiles, wildtypes)
    panel = build_panel(units, profiles)
    identity = row_identity(panel['group'], panel['site'], panel['target'])

    ladder, standing = [], tuple(BASE_BLOCKS)
    for candidate in CANDIDATE_BLOCKS:
        design = {'standing': standing, 'candidate': (*standing, candidate)}
        increments, own, spearman, folds, nuisance = {}, {}, {}, {}, {}
        for seed in SPLIT_SEEDS:
            outcome = fold_predictions(panel, panel['blocks'], design, seed=seed,
                                       first_stage=standing, device=args.device)
            record = paired_increment(panel, outcome['predictions'], 'candidate', 'standing')
            increments[seed] = record['point']
            own[str(seed)] = record
            spearman[str(seed)] = spearman_increment(panel, outcome['predictions'],
                                                     'candidate', 'standing')
            folds[str(seed)] = {'alpha': [f['alpha'] for f in outcome['folds']],
                                'dimensions': outcome['folds'][0]['dimensions']}
            if candidate == 'G':
                nuisance[str(seed)] = outcome['nuisance']
        verdict = qualify(increments)
        first = own[str(SPLIT_SEEDS[0])]
        ladder.append({
            'candidate': candidate,
            'standing_set': list(standing),
            **verdict,
            'own_contribution_interval_per_seed': own,
            'own_contribution_resolves_away_from_zero_at_the_first_seed': bool(
                first['excludes_zero']),
            'spearman_increment': spearman,
            'folds': folds,
            'nuisance_diagnostics': nuisance,
            'disposition': ('kept in the qualified control set' if verdict['qualified']
                            else 'discarded: it lowers the held-out mean squared error of '
                                 'the set it augments at at least one seed'),
        })
        if verdict['qualified']:
            standing = (*standing, candidate)

    final = {}
    for seed in SPLIT_SEEDS:
        outcome = fold_predictions(panel, panel['blocks'], {'S': standing}, seed=seed,
                                   first_stage=tuple(b for b in standing if b != 'G'),
                                   device=args.device)
        labels, errors = group_errors(panel['target'], outcome['predictions']['S'],
                                      panel['group'], panel['domain'], panel['site'])
        _, null = group_errors(panel['target'], outcome['predictions']['NO_EFFECT_NULL'],
                               panel['group'], panel['domain'], panel['site'])
        final[str(seed)] = {
            'qualified_set_mse': interval(errors),
            'no_effect_null_mse': interval(null),
            'reduction_over_no_effect_null': interval(null - errors),
            'share_of_the_null_removed': float(1.0 - errors.mean() / null.mean()),
            'within_unit_spearman': raw_spearman(panel, outcome['predictions']['S']),
            'dimensions': outcome['folds'][0]['dimensions']['S'],
            'alpha': [f['alpha']['S'] for f in outcome['folds']],
            'evaluated_groups': int(len(labels)),
        }

    report = {
        'schema': SCHEMA,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'gate': loaded['schema'],
        'endpoint': loaded['endpoint'],
        'unit': loaded['unit_name'],
        'numeric_environment': numeric,
        'cohort_sha256': loaded['sha256'],
        'endpoint_sha256': loaded['endpoint_sha256'],
        'profile_sha256': sha256_file(args.profiles),
        'row_identity_sha256': identity,
        'profile_units_present': sum(1 for r in profile_meta['backgrounds']
                                     if r['status'] == 'present' and r['name'] in wildtypes),
        'profile_units_absent': sorted(r['name'] for r in profile_meta['backgrounds']
                                       if r['status'] != 'present' and r['name'] in wildtypes),
        'rows': int(len(panel['target'])),
        'effective_units': kish_units(panel['group'], panel['domain'], panel['site']),
        'qualification_rule': (
            'a candidate is kept only if its paired reduction in group-equal held-out mean '
            'squared error over the standing set is positive at every prespecified split seed'),
        'l48_rule': (
            'each candidate carries its own contribution with a group-bootstrap interval, and '
            'the qualified set carries its own contribution over the no-effect null with one: '
            'a control contributing indistinguishably from zero makes any increment over it '
            'uninformative, which the catalogue records as L48 with four instances'),
        'split_seeds': list(SPLIT_SEEDS),
        'base_blocks': list(BASE_BLOCKS),
        'candidate_order': list(CANDIDATE_BLOCKS),
        'g_feature_order': list(G_FEATURE_ORDER),
        'g_first_stage': G_FIRST_STAGE_NOTE,
        'ladder': ladder,
        'qualified_control_set': list(standing),
        'discarded': [row['candidate'] for row in ladder if not row['qualified']],
        'qualified_set_performance': final,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'controls_qualification.json', report)
    print(json.dumps({
        'gate': report['gate'],
        'qualified_control_set': report['qualified_control_set'],
        'discarded': report['discarded'],
        'ladder': [{'candidate': row['candidate'],
                    'per_seed_increment': row['per_seed_increment'],
                    'qualified': row['qualified'],
                    'own_first_seed': {k: row['own_contribution_interval_per_seed'][
                        str(SPLIT_SEEDS[0])][k]
                        for k in ('point', 'interval', 'excludes_zero')}}
                   for row in ladder],
        'qualified_set_performance': {
            seed: {'mse': value['qualified_set_mse']['point'],
                   'null': value['no_effect_null_mse']['point'],
                   'reduction': value['reduction_over_no_effect_null'],
                   'share_of_the_null_removed': value['share_of_the_null_removed'],
                   'spearman': value['within_unit_spearman']['point'],
                   'dimensions': value['dimensions']}
            for seed, value in final.items()},
    }, indent=1))


if __name__ == '__main__':
    main()
