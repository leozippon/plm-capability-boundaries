#!/usr/bin/env python3
"""Fit one arm's likelihood and representation increments on one nested gate's cohort.

This entry point serves both cohorts the nested machinery fits. The control set
is the one ``qualify_nested_controls.py`` froze before any model quantity was
read; this entry point refuses to run without that artefact and never re-derives
it.

Two model quantities are added separately, never only jointly: the likelihood
difference M_mut - M_WT in nats, and the projected representation difference
R_mut - R_WT over four pooled blocks. The arm's own tokenisation descriptors are
offered to the matched baseline under the same qualification rule the controls
faced, because a representation difference can be nonzero purely through
segmentation; an arm whose descriptors lower its own baseline is reported as such
and its increments are read over the qualified set without them.

Every design on one seed sees the identical rows, the identical nested
held-group partitions and the identical label budget; only its declared column
blocks differ.

**The stratified readout is a readout restriction, not a refit.** When the cohort
declares identity strata, the same folds, the same training groups and the same
held-out predictions are read out again over the close groups alone and over the
remote groups alone, and the per-group nesting is renormalised inside the
retained rows. The close stratum is the positive control: it is where a model
quantity that depends on retrieved homology should resolve. The pair of readouts
selects one of the three declared outcomes, and this entry point records which.

The representation cells are computed and retained but carry no verdict: they are
conditional on the compressed linear readout of four pooled block summaries and
the readout-class reassessment is still running.
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

from src.capability.mutation.external_confirmation import (
    ARM_CANDIDATE_BLOCKS, MODEL_BLOCKS, SPLIT_SEEDS,
    TOKENISATION_FEATURE_ORDER, build_panel, fold_predictions, group_errors, interval,
    kish_units, load_cohort, paired_increment, qualify, raw_spearman,
    require_blas_threads, row_identity, secondary_control_set, spearman_increment,
    tokenisation_block)
from src.capability.core.io import sha256_file, write_json
from src.capability.interactions.pairwise_epistasis import (  # noqa: E402
    ROSTER, TOKENISATION_STRATUM, require_projected_width)
from src.capability.context.remote_homology import (
    OUTCOME_HOMOLOGY_DEPENDENT, OUTCOME_SURVIVES, OUTCOME_UNRESOLVED, STRATUM_UNIT_FLOOR,
    outcome_record, stratum_keep)
from src.capability.stability.stability_gate import load_profiles

SCHEMA = 'nested_gate_fit_v1'
EXTRACTION_SCHEMA = 'stability_singles_extraction_v1'


def load_arm(directory: Path, arm: str, units: list[dict], cohort_sha256: str):
    """Per-variant model blocks, checked against the cohort's own state indices."""

    manifest = json.loads((directory / f'manifest_{arm}.json').read_text())
    identity = manifest['identity']
    if identity.get('schema') != EXTRACTION_SCHEMA:
        raise ValueError(f'{arm}: extraction artefact is not the nested-gate plan schema')
    if manifest['status'] != 'complete' or identity['arm'] != arm:
        raise ValueError(f'{arm}: extraction manifest is not a complete record of this arm')
    if identity['cohort_sha256'] != cohort_sha256:
        raise ValueError(f'{arm}: extraction was run against a different cohort')
    by_name = {row['background']: row for row in manifest['backgrounds']}
    likelihood, representation, tokens = [], [], []
    for row in units:
        record = by_name.get(row['name'])
        if record is None:
            raise ValueError(f"{arm}: no extraction artefact for {row['name']}")
        with np.load(directory / record['file'], allow_pickle=False) as data:
            states = data['variant_states']
            # A cohort's state index is its variant's position in that unit's own
            # declared order, because that order is what the extraction plan turned
            # into `sequences = [wild type] + variant sequences`. One cohort schema
            # records the index and the other leaves it implicit; deriving it from
            # the order reads both without either having to be re-declared, and the
            # position check below binds the order itself.
            declared = np.asarray([variant.get('state', index + 1)
                                   for index, variant in enumerate(row['variants'])])
            if not np.array_equal(states, declared):
                raise ValueError(f"{arm}: {row['name']} state indices differ from the cohort")
            if not np.array_equal(data['variant_positions'],
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
                                                 len(row['wildtype'])))
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
    parser.add_argument('--permutation-check', action='store_true',
                        help='also verify that permuting held-out group labels moves no '
                             'held-out prediction of any design')
    args = parser.parse_args()

    if args.arm not in ROSTER:
        raise SystemExit(f'{args.arm} is not on the frozen roster')
    numeric = require_blas_threads()
    import torch
    torch.set_num_threads(numeric['pinned_threads'])

    loaded = load_cohort(args.cohort)
    units = loaded['units']
    controls = json.loads(args.controls.read_bytes())
    if controls.get('schema') != 'nested_gate_control_qualification_v1':
        raise SystemExit('unexpected control-qualification schema')
    if controls['cohort_sha256'] != loaded['sha256']:
        raise SystemExit('the control qualification was run against a different cohort')
    qualified = tuple(controls['qualified_control_set'])
    secondary, secondary_record = secondary_control_set(controls)

    wildtypes = {row['name']: row['wildtype'] for row in units}
    profiles, _ = load_profiles(args.profiles, wildtypes)
    panel = build_panel(units, profiles)
    identity = row_identity(panel['group'], panel['site'], panel['target'])
    if identity != controls['row_identity_sha256']:
        raise SystemExit('the panel rows differ from the ones the controls were qualified on')
    model_blocks, manifest = load_arm(args.extraction, args.arm, units, loaded['sha256'])
    for name, block in model_blocks.items():
        if len(block) != len(panel['target']):
            raise SystemExit(f'{args.arm}: block {name} has {len(block)} rows against '
                             f"{len(panel['target'])} cohort variants")
    blocks = dict(panel['blocks'], **model_blocks)

    designs = {}
    for prefix, columns in (('S', qualified), ('S2', secondary)):
        designs[prefix] = tuple(columns)
        for extra in ARM_CANDIDATE_BLOCKS:
            designs[f'{prefix}_{extra}'] = (*columns, extra)
        for block in MODEL_BLOCKS:
            designs[f'{prefix}_{block}'] = (*columns, block)
            for extra in ARM_CANDIDATE_BLOCKS:
                designs[f'{prefix}_{extra}_{block}'] = (*columns, extra, block)
    # One nonlinear-additive response per fold, fitted over the primary standing
    # columns, shared by every design that names G. The secondary set is read
    # over that same response rather than over a second one, so the two sets
    # differ only in their own declared columns.
    first_stage = tuple(block for block in qualified if block != 'G')

    main_pass = {}
    for seed in SPLIT_SEEDS:
        main_pass[seed] = fold_predictions(panel, blocks, designs, seed=seed,
                                           first_stage=first_stage, device=args.device)
    tokenisation_verdict = qualify({
        seed: paired_increment(panel, main_pass[seed]['predictions'], 'S_T', 'S')['point']
        for seed in SPLIT_SEEDS})
    suffix = '_T' if tokenisation_verdict['qualified'] else ''
    baselines = {'primary': f'S{suffix}', 'secondary': f'S2{suffix}'}

    strata = loaded['strata']
    stratum_names = sorted(set(strata.values())) if strata else []
    stratum_masks = {name: stratum_keep(panel['group'], strata, name)
                     for name in stratum_names}

    def contrasts(predictions: dict, keep=None) -> dict:
        out = {}
        for role, baseline in baselines.items():
            for quantity, block in (('likelihood', 'M'), ('representation', 'R')):
                augmented = f'{baseline}_{block}'
                out[f'{role}_{quantity}'] = paired_increment(
                    panel, predictions, augmented, baseline, keep=keep)
                if keep is None:
                    out[f'{role}_{quantity}_spearman'] = spearman_increment(
                        panel, predictions, augmented, baseline)
        return out

    per_seed = {}
    for seed in SPLIT_SEEDS:
        predictions = main_pass[seed]['predictions']
        _, null = group_errors(panel['target'], predictions['NO_EFFECT_NULL'],
                               panel['group'], panel['domain'], panel['site'])
        record = {
            'no_effect_null_mse': interval(null),
            'tokenisation_increment': paired_increment(panel, predictions, 'S_T', 'S'),
            **contrasts(predictions),
            'alpha': [f['alpha'] for f in main_pass[seed]['folds']],
            'dimensions': main_pass[seed]['folds'][0]['dimensions'],
            'held_groups_per_fold': [f['held_groups'] for f in main_pass[seed]['folds']],
            'nuisance': main_pass[seed]['nuisance'],
        }
        for role, baseline in baselines.items():
            _, errors = group_errors(panel['target'], predictions[baseline],
                                     panel['group'], panel['domain'], panel['site'])
            record[f'{role}_baseline_mse'] = interval(errors)
            record[f'{role}_baseline_within_unit_spearman'] = raw_spearman(
                panel, predictions[baseline])
        for name, keep in stratum_masks.items():
            record[f'stratum_{name}'] = {
                'groups': int(len(set(panel['group'][keep].tolist()))),
                'rows': int(keep.sum()),
                'clears_the_unit_floor': bool(
                    len(set(panel['group'][keep].tolist())) >= STRATUM_UNIT_FLOOR),
                **contrasts(predictions, keep=keep),
            }
        per_seed[str(seed)] = record

    stratified_outcome = None
    if {'close', 'remote'} <= set(stratum_masks):
        resolved = {}
        for name in ('close', 'remote'):
            for role in baselines:
                key = f'{role}_likelihood'
                points = [per_seed[str(seed)][f'stratum_{name}'][key] for seed in SPLIT_SEEDS]
                resolved[(name, role)] = bool(
                    all(p['excludes_zero'] and p['point'] > 0 for p in points))
        stratified_outcome = {
            'rule': ('an arm resolves on a stratum when all three per-seed intervals of its '
                     'paired increment over that stratum exclude zero above it'),
            'positive_control': ('the close stratum gates the reading: with this floor a '
                                 'stratum on which nothing resolves cannot tell an absent '
                                 'quantity from an unmeasurable one, so an arm resolving on '
                                 'the remote groups while its own close stratum did not fire '
                                 'is unresolved and not survival'),
            'outcomes': {role: outcome_record(resolved[('close', role)],
                                              resolved[('remote', role)])
                         for role in baselines},
            'declared_outcomes': [OUTCOME_SURVIVES, OUTCOME_HOMOLOGY_DEPENDENT,
                                  OUTCOME_UNRESOLVED],
        }

    permutation = {}
    if args.permutation_check:
        generator = np.random.default_rng(20260925)
        held = main_pass[SPLIT_SEEDS[0]]['folds'][0]['held_groups']
        rows = np.flatnonzero(np.isin(panel['group'], held))
        permuted = dict(panel)
        target = panel['target'].copy()
        target[rows] = target[generator.permutation(rows)]
        permuted['target'] = target
        replayed = fold_predictions(permuted, blocks, designs, seed=SPLIT_SEEDS[0],
                                    first_stage=first_stage, device=args.device)
        permutation = {
            'permuted_rows': int(len(rows)),
            'target_changed': bool(not np.array_equal(target, panel['target'])),
            'max_absolute_held_out_prediction_change': {
                name: float(np.max(np.abs(
                    replayed['predictions'][name][rows]
                    - main_pass[SPLIT_SEEDS[0]]['predictions'][name][rows])))
                for name in designs},
        }

    report = {
        'schema': SCHEMA,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'gate': loaded['schema'],
        'endpoint': loaded['endpoint'],
        'unit': loaded['unit_name'],
        'arm': args.arm,
        'tokenisation_stratum': TOKENISATION_STRATUM[args.arm],
        'numeric_environment': numeric,
        'dtype': manifest['identity']['dtype'],
        'batch_size': manifest['identity']['batch_size'],
        'precision_gate': manifest['identity']['precision_gate'],
        'repeat_likelihood_nats_max': max(r['repeat_likelihood_nats']
                                          for r in manifest['backgrounds']),
        'repeat_feature_relative_l2_max': max(r['repeat_feature_relative_l2']
                                              for r in manifest['backgrounds']),
        'cohort_sha256': loaded['sha256'],
        'endpoint_sha256': loaded['endpoint_sha256'],
        'controls_sha256': sha256_file(args.controls),
        'profile_sha256': sha256_file(args.profiles),
        'row_identity_sha256': identity,
        'qualified_control_set': list(qualified),
        'secondary_control_set': list(secondary),
        'secondary_control_derivation': secondary_record,
        'discarded_controls': controls['discarded'],
        'tokenisation_verdict': tokenisation_verdict,
        'matched_baseline': baselines,
        'rows': int(len(panel['target'])),
        'effective_units': kish_units(panel['group'], panel['domain'], panel['site']),
        'split_seeds': list(SPLIT_SEEDS),
        'per_seed': per_seed,
        'stratified_outcome': stratified_outcome,
        'representation_status': (
            'computed and retained, labelled provisional: the representation cells are '
            'conditional on the compressed linear readout of four pooled block summaries '
            'and the readout-class reassessment is still running; no verdict about '
            'representational content is drawn here'),
        'permutation_check': permutation,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / f'fit_{args.arm}.json', report)
    print(json.dumps({
        'arm': args.arm, 'gate': report['gate'], 'matched_baseline': baselines,
        'tokenisation_qualified': tokenisation_verdict['qualified'],
        'primary_likelihood': {seed: per_seed[seed]['primary_likelihood']['point']
                               for seed in per_seed},
        'primary_representation': {seed: per_seed[seed]['primary_representation']['point']
                                   for seed in per_seed},
        'stratified_outcome': (stratified_outcome['outcomes'] if stratified_outcome else None),
    }, indent=1))


if __name__ == '__main__':
    main()
