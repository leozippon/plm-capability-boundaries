#!/usr/bin/env python3
"""Fit one arm's likelihood and representation increments on single-mutant stability.

The target is ddG = y_mut - y_WT on the combined MegaScale dG_ML scale in
kcal/mol. The control set is the one ``qualify_stability_controls.py`` froze
before any model quantity was read; this entry point refuses to run without that
artefact and never re-derives it.

Two model quantities are added separately, never only jointly: the likelihood
difference M_mut - M_WT in nats, and the projected representation difference
R_mut - R_WT over four pooled blocks. The arm's own tokenisation descriptors are
offered to the matched baseline under the same qualification rule the controls
faced, because a representation difference can be nonzero purely through
segmentation; an arm whose descriptors lower its own baseline is reported as
such and its increments are read over the qualified set without them.

Every design on one seed sees the identical rows, the identical nested
held-group partitions and the identical label budget; only its declared column
blocks differ. The remote stratification repeats the model comparisons with the
nearest-detected-neighbour training groups purged, on exactly the same held-out
rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.core.io import sha256_file, write_json
from src.capability.interactions.pairwise_epistasis import (
    ROSTER, TOKENISATION_STRATUM, require_projected_width)
from src.capability.stability.stability_gate import (
    ENDPOINT, QUALIFICATION_SEEDS, SPLIT_SEEDS, TOKENISATION_FEATURE_ORDER, build_panel,
    fold_predictions, group_errors, group_spearman, interval, load_profiles,
    paired_increment, qualify, secondary_control_set, spearman_increment, tokenisation_block)


def load_arm(directory: Path, arm: str, cohort: dict) -> tuple[dict, dict]:
    """Per-variant model blocks, checked against the cohort's own state indices."""

    manifest = json.loads((directory / f'manifest_{arm}.json').read_text())
    if manifest['identity'].get('schema') != 'stability_singles_extraction_v1':
        raise ValueError(f'{arm}: extraction artefact is not this gate\'s schema')
    if manifest['status'] != 'complete' or manifest['identity']['arm'] != arm:
        raise ValueError(f'{arm}: extraction manifest is not a complete record of this arm')
    if manifest['identity']['cohort_sha256'] != cohort['_sha256']:
        raise ValueError(f'{arm}: extraction was run against a different cohort')
    by_name = {row['background']: row for row in manifest['backgrounds']}
    likelihood, representation, tokens = [], [], []
    for row in cohort['backgrounds']:
        record = by_name.get(row['name'])
        if record is None:
            raise ValueError(f"{arm}: no extraction artefact for {row['name']}")
        with np.load(directory / record['file'], allow_pickle=False) as data:
            states = data['variant_states']
            positions = data['variant_positions']
            declared = np.asarray([v['state'] for v in row['variants']])
            if not np.array_equal(states, declared):
                raise ValueError(f"{arm}: {row['name']} state indices differ from the cohort")
            if not np.array_equal(positions,
                                  np.asarray([v['position'] for v in row['variants']])):
                raise ValueError(f"{arm}: {row['name']} positions differ from the cohort")
            scores = data['likelihood']
            projected = data['projected'].reshape(len(scores), -1)
            pooled = data['pooled_token_counts']
            offsets = data['token_offsets']
            ids = data['token_ids']
            token_ids = [ids[offsets[i]:offsets[i + 1]].tolist() for i in range(len(scores))]
            for state in states:
                likelihood.append(float(scores[state] - scores[0]))
                representation.append(projected[state] - projected[0])
                tokens.append(tokenisation_block(pooled, token_ids, 0, int(state),
                                                 row['length']))
    blocks = {'M': np.asarray(likelihood, dtype=float)[:, None],
              'R': np.asarray(representation, dtype=float),
              'T': np.asarray(tokens, dtype=float)}
    require_projected_width(blocks['R'].shape[1], arm=arm,
                            source=f'the assembled representation of {directory.name}')
    if blocks['T'].shape[1] != len(TOKENISATION_FEATURE_ORDER):
        raise ValueError(f'{arm}: tokenisation descriptor width changed')
    return blocks, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True,
                        help='the frozen control-qualification artefact')
    parser.add_argument('--extraction', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--permutation-check', action='store_true',
                        help='also verify that permuting held-out group labels moves no '
                             'held-out prediction of any design')
    args = parser.parse_args()

    if args.arm not in ROSTER:
        raise SystemExit(f'{args.arm} is not on the frozen roster')
    if args.threads:
        torch.set_num_threads(args.threads)
    cohort = json.loads(args.cohort.read_bytes())
    if cohort.get('schema') != 'stability_singles_cohort_v1':
        raise SystemExit('unexpected cohort schema')
    cohort['_sha256'] = sha256_file(args.cohort)
    controls = json.loads(args.controls.read_bytes())
    if controls.get('schema') != 'stability_control_qualification_v1':
        raise SystemExit('unexpected control-qualification schema')
    if controls['cohort_sha256'] != cohort['_sha256']:
        raise SystemExit('the control qualification was run against a different cohort')
    qualified = tuple(controls['qualified_control_set'])

    wildtypes = {row['name']: row['wildtype'] for row in cohort['backgrounds']}
    profiles, _ = load_profiles(args.profiles, wildtypes)
    panel = build_panel(cohort, profiles)
    model_blocks, manifest = load_arm(args.extraction, args.arm, cohort)
    for name, block in model_blocks.items():
        if len(block) != len(panel['target']):
            raise SystemExit(f'{args.arm}: block {name} has {len(block)} rows against '
                             f"{len(panel['target'])} cohort variants")
    blocks = dict(panel['blocks'], **model_blocks)

    secondary, secondary_record = secondary_control_set(controls)
    designs = {}
    for prefix, columns in (('S', qualified), ('S2', secondary)):
        designs[prefix] = tuple(columns)
        designs[f'{prefix}_T'] = (*columns, 'T')
        designs[f'{prefix}_M'] = (*columns, 'M')
        designs[f'{prefix}_R'] = (*columns, 'R')
        designs[f'{prefix}_T_M'] = (*columns, 'T', 'M')
        designs[f'{prefix}_T_R'] = (*columns, 'T', 'R')
    main_pass, permutation = {}, {}
    for seed in SPLIT_SEEDS:
        outcome = fold_predictions(panel, blocks, designs, seed=seed, device=args.device)
        main_pass[seed] = outcome
    tokenisation_verdict = qualify({
        seed: paired_increment(panel, main_pass[seed]['predictions'], 'S_T', 'S')['point']
        for seed in QUALIFICATION_SEEDS})
    suffix = '_T' if tokenisation_verdict['qualified'] else ''
    baselines = {'primary': f'S{suffix}', 'secondary': f'S2{suffix}'}

    def contrasts(outcome: dict) -> dict:
        out = {}
        for role, baseline in baselines.items():
            for quantity, block in (('likelihood', 'M'), ('representation', 'R')):
                augmented = f'{baseline}_{block}'
                out[f'{role}_{quantity}'] = paired_increment(
                    panel, outcome['predictions'], augmented, baseline)
                out[f'{role}_{quantity}_spearman'] = spearman_increment(
                    panel, outcome['predictions'], augmented, baseline)
        return out

    primary = {}
    for seed in SPLIT_SEEDS:
        outcome = main_pass[seed]
        _, null = group_errors(panel['target'], outcome['predictions']['NO_EFFECT_NULL'],
                               panel['group'], panel['site'])
        record = {'no_effect_null_mse_kcal2_mol2': interval(null),
                  'tokenisation_increment_kcal2_mol2': paired_increment(
                      panel, outcome['predictions'], 'S_T', 'S'),
                  **contrasts(outcome),
                  'alpha': [f['alpha'] for f in outcome['folds']],
                  'dimensions': outcome['folds'][0]['dimensions'],
                  'nuisance': outcome['nuisance']}
        for role, baseline in baselines.items():
            _, errors = group_errors(panel['target'], outcome['predictions'][baseline],
                                     panel['group'], panel['site'])
            _, ranks = group_spearman(panel['target'], outcome['predictions'][baseline],
                                      panel['group'])
            record[f'{role}_baseline_mse_kcal2_mol2'] = interval(errors)
            record[f'{role}_baseline_within_background_spearman'] = interval(ranks)
        primary[str(seed)] = record

    remote_designs = {name: designs[name] for baseline in baselines.values()
                      for name in (baseline, f'{baseline}_M', f'{baseline}_R')}
    remote = {}
    for label, purge in cohort['remote_purge'].items():
        remote[label] = {}
        for seed in SPLIT_SEEDS:
            outcome = fold_predictions(panel, blocks, remote_designs,
                                       seed=seed, device=args.device, purge=purge)
            remote[label][str(seed)] = {
                **contrasts(outcome),
                'purged_training_groups_per_fold': [
                    len(f['purged_training_groups']) for f in outcome['folds']],
                'training_groups_per_fold': [len(f['training_groups']) for f in outcome['folds']],
            }

    if args.permutation_check:
        rng = np.random.default_rng(20260925)
        held = main_pass[SPLIT_SEEDS[0]]['folds'][0]['held_groups']
        rows = np.flatnonzero(np.isin(panel['group'], held))
        permuted = dict(panel)
        target = panel['target'].copy()
        target[rows] = target[rng.permutation(rows)]
        permuted['target'] = target
        states = dict(panel['states'])
        state_rows = np.flatnonzero(np.isin(panel['states']['group'], held))
        y = panel['states']['y'].copy()
        y[state_rows] = y[rng.permutation(state_rows)]
        states['y'] = y
        permuted['states'] = states
        outcome = fold_predictions(permuted, blocks, designs,
                                   seed=SPLIT_SEEDS[0], device=args.device)
        permutation = {
            'permuted_target_rows': int(len(rows)),
            'permuted_state_rows': int(len(state_rows)),
            'target_changed': bool(not np.array_equal(target, panel['target'])),
            'max_absolute_held_out_prediction_change': {
                name: float(np.max(np.abs(
                    outcome['predictions'][name][rows]
                    - main_pass[SPLIT_SEEDS[0]]['predictions'][name][rows])))
                for name in designs},
        }

    report = {
        'schema': 'stability_singles_fit_v1',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'endpoint': ENDPOINT,
        'arm': args.arm,
        'tokenisation_stratum': TOKENISATION_STRATUM[args.arm],
        'dtype': manifest['identity']['dtype'],
        'batch_size': manifest['identity']['batch_size'],
        'precision_gate': manifest['identity']['precision_gate'],
        'repeat_likelihood_nats_max': max(r['repeat_likelihood_nats']
                                          for r in manifest['backgrounds']),
        'repeat_feature_relative_l2_max': max(r['repeat_feature_relative_l2']
                                              for r in manifest['backgrounds']),
        'cohort_sha256': cohort['_sha256'],
        'endpoint_sha256': cohort['endpoint_sha256'],
        'controls_sha256': sha256_file(args.controls),
        'profile_sha256': sha256_file(args.profiles),
        'qualified_control_set': list(qualified),
        'secondary_control_set': list(secondary),
        'secondary_control_derivation': secondary_record,
        'discarded_controls': controls['discarded'],
        'tokenisation_verdict': tokenisation_verdict,
        'matched_baseline': baselines,
        'rows': int(len(panel['target'])),
        'groups': int(len(set(panel['group']))),
        'sites': int(len(set(panel['site']))),
        'split_seeds': list(SPLIT_SEEDS),
        'primary': primary,
        'remote_stratification': remote,
        'permutation_check': permutation,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / f'fit_{args.arm}.json', report)
    print(json.dumps({
        'arm': args.arm, 'matched_baseline': baselines,
        'tokenisation_qualified': tokenisation_verdict['qualified'],
        'primary_likelihood_mse': {s: primary[s]['primary_likelihood']['point'] for s in primary},
        'primary_representation_mse': {
            s: primary[s]['primary_representation']['point'] for s in primary},
        'secondary_likelihood_spearman': {
            s: primary[s]['secondary_likelihood_spearman']['point'] for s in primary},
        'secondary_representation_spearman': {
            s: primary[s]['secondary_representation_spearman']['point'] for s in primary},
    }, indent=1))


if __name__ == '__main__':
    main()
