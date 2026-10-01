#!/usr/bin/env python3
"""Fit the control ladder and the model increments on measured double-mutant nonadditivity.

The target is the wild-type-centred cycle epsilon = y_AB - y_A - y_B + y_WT on the
combined MegaScale dG_ML scale in kcal/mol. Every design on one support and one
split seed sees the identical rows, the identical nested held-group partitions and
the identical label budget; only its declared column blocks differ.

Primary evaluation is the paired reduction in group-equal epsilon mean squared
error in squared kcal/mol, with the site pair as the weighted unit inside a group,
group-bootstrap percentile intervals, and within-background Spearman reported
beside it. The zero-interaction additive null has a defined squared error and an
undefined Spearman, which is reported as undefined.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.interactions.pairwise_epistasis import (
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, CONTROL_SETS, FEATURE_BLOCKS,
    PROJECTION_DIM, ROSTER, SPLIT_SEEDS, TOKENISATION_STRATUM, cycle_contrast,
    cycle_control_features, design_blocks, design_names, group_errors, group_spearman,
    fold_identity, interval, kish_effective_site_pairs, nested_compare, plan_digest,
    require_projected_width, row_identity, state_features, tokenisation_features)
from src.capability.context.profiles import Profile

#: Declared contrasts. Each pair is (baseline, augmented) on one support: the
#: control ladder first, then likelihood and representation added separately over
#: every control set, then the same additions over the first-order control that
#: already carries the constituent single-mutant model quantities.
CONTRASTS: dict[str, tuple[tuple[str, str], ...]] = {
    'all': (
        ('ADDITIVE_NULL', 'C'), ('C', 'C_G'), ('C_G', 'C_G_T'),
        ('C', 'C+M'), ('C_G', 'C_G+M'), ('C_G_T', 'C_G_T+M'),
        ('C', 'C+R'), ('C_G', 'C_G+R'), ('C_G_T', 'C_G_T+R'),
        ('C_G_T', 'C_G_T+MR'),
        ('C_G_T', 'C_G_T_M1'), ('C_G_T_M1', 'C_G_T_M1+M'),
        ('C_G_T', 'C_G_T_R1'), ('C_G_T_R1', 'C_G_T_R1+R'),
    ),
    'q': (
        ('C_G_T', 'C_G_T_Q'),
        ('C_G_T', 'C_G_T+M'), ('C_G_T', 'C_G_T+R'),
        ('C_G_T_Q', 'C_G_T_Q+M'), ('C_G_T_Q', 'C_G_T_Q+R'),
        ('C_G_T_Q', 'C_G_T_Q+MR'),
    ),
}
STRATA = ('1-2', '3-9', '10+')


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b''):
            sha.update(chunk)
    return sha.hexdigest()


def load_profiles(path: Path, plan: dict) -> tuple[dict, dict]:
    wildtypes = {row['name']: row['wildtype'] for row in plan['backgrounds']}
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        profiles = {}
        for record in meta['backgrounds']:
            name = record['name']
            if record['status'] != 'present':
                profiles[name] = None
                continue
            scalars = data[f'{name}|scalars']
            profiles[name] = Profile(
                query_id=name, wildtype=wildtypes[name], n_hits=int(scalars[4]),
                n_sequences=int(scalars[3]), saturated=False,
                frequencies=data[f'{name}|frequencies'].astype(np.float64),
                column_weight=data[f'{name}|column_weight'].astype(np.float64),
                neff=float(scalars[0]), max_identity_over_query=float(scalars[2]))
    return profiles, meta


def load_arm(directory: Path, arm: str, plan: dict) -> tuple[dict, dict]:
    manifest = json.loads((directory / f'manifest_{arm}.json').read_text())
    if manifest['status'] != 'complete' or manifest['identity']['arm'] != arm:
        raise ValueError(f'{arm}: extraction manifest is not a complete record of this arm')
    if manifest['identity']['plan_sha256'] != plan_digest(plan):
        raise ValueError(f'{arm}: extraction was not run against this plan')
    if len(manifest['backgrounds']) != len(plan['backgrounds']):
        raise ValueError(f'{arm}: extraction covers {len(manifest["backgrounds"])} backgrounds')
    data = {}
    for record in manifest['backgrounds']:
        path = directory / record['file']
        if digest(path) != record['sha256']:
            raise ValueError(f'{arm}: {record["file"]} does not match its manifest digest')
        with np.load(path, allow_pickle=False) as saved:
            offsets = saved['token_offsets']
            projected = saved['projected'].astype(np.float64)
            require_projected_width(projected.reshape(len(saved['likelihood']), -1).shape[1],
                                    arm=arm, source=record['file'])
            data[record['background']] = {
                'projected': projected,
                'likelihood': saved['likelihood'].astype(np.float64),
                'pooled_token_counts': saved['pooled_token_counts'],
                'token_ids': [saved['token_ids'][offsets[i]:offsets[i + 1]].tolist()
                              for i in range(len(offsets) - 1)],
                'cycle_states': saved['cycle_states'],
                'cycle_positions': saved['cycle_positions'],
            }
    return data, manifest


def build_panel(plan: dict, cohort: dict, profiles: dict, arm_data: dict,
                baseline: dict, groups: set) -> dict:
    """Assemble the cycle rows, the declared blocks and the absolute-stability states."""

    measurements = {row['name']: row['measurements'] for row in cohort['backgrounds']}
    epsilon_by_background = {row['name']: [c['epsilon'] for c in row['cycles']]
                             for row in cohort['backgrounds']}
    contrasts = {row['name']: row.get('contrasts') for row in baseline['backgrounds']}
    row_group, row_pair, row_background, row_separation, epsilon, row_cycle = [], [], [], [], [], []
    control, tokens, q_value, m_value, m_first = [], [], [], [], []
    representation, first_order, cycle_states = [], [], []
    state_features_rows, state_y, state_group, state_background = [], [], [], []
    independent_site = []
    for background in plan['backgrounds']:
        name = background['name']
        if background['group'] not in groups:
            continue
        arm = arm_data[name]
        wildtype, sequences = background['wildtype'], background['sequences']
        profile = profiles[name]
        base = len(state_y)
        for sequence in sequences:
            state_features_rows.append(state_features(wildtype, sequence, profile))
            state_y.append(measurements[name][sequence]['value'])
            state_group.append(background['group'])
            state_background.append(name)
        stored = arm['cycle_states']
        if stored.shape != (len(background['cycles']), 4):
            raise ValueError(f'{name}: extraction cycle table does not match the plan')
        pooled, ids = arm['pooled_token_counts'], arm['token_ids']
        for index, cycle in enumerate(background['cycles']):
            states = tuple(cycle['states'])
            if tuple(stored[index]) != states or list(arm['cycle_positions'][index]) != cycle['positions']:
                raise ValueError(f'{name}: extraction cycle {index} disagrees with the plan')
            positions = tuple(cycle['positions'])
            quartet = tuple(sequences[state] for state in states)
            control.append(cycle_control_features(wildtype, quartet, positions,
                                                  cycle['separation'], profile))
            tokens.append(tokenisation_features(pooled, ids, states, len(wildtype)))
            cycle_states.append([base + state for state in states])
            row_group.append(background['group'])
            row_pair.append(f'{name}:{positions[0]}-{positions[1]}')
            row_background.append(name)
            row_separation.append(cycle['separation'])
            row_cycle.append(index)
            epsilon.append(epsilon_by_background[name][index])
            if contrasts[name] is not None:
                entry = contrasts[name][index]
                if entry['positions'] != cycle['positions'] or entry['separation'] != cycle['separation']:
                    raise ValueError(f'{name}: baseline Q contrast {index} is not aligned to the plan')
                q_value.append(entry['pairwise_contrast'])
                independent_site.append(abs(entry['independent_site_contrast']))
        local_states = np.asarray([cycle['states'] for cycle in background['cycles']])
        likelihood = arm['likelihood']
        m_value.extend(cycle_contrast(likelihood, local_states).tolist())
        m_first.extend(np.column_stack([
            likelihood[local_states[:, 1]] - likelihood[local_states[:, 0]],
            likelihood[local_states[:, 2]] - likelihood[local_states[:, 0]]]).tolist())
        projected = arm['projected'].reshape(len(likelihood), -1)
        representation.append(cycle_contrast(projected, local_states))
        first_order.append(np.concatenate([
            projected[local_states[:, 1]] - projected[local_states[:, 0]],
            projected[local_states[:, 2]] - projected[local_states[:, 0]]], axis=1))
    blocks = {'ident': np.asarray(control)[:, :400], 'geom': np.asarray(control)[:, 400:410],
              'comp': np.asarray(control)[:, 410:450], 'local': np.asarray(control)[:, 450:850],
              'prof': np.asarray(control)[:, 850:], 'T': np.asarray(tokens),
              'M': np.asarray(m_value)[:, None], 'M1': np.asarray(m_first),
              'R': np.concatenate(representation), 'R1': np.concatenate(first_order)}
    if q_value:
        blocks['Q'] = np.asarray(q_value)[:, None]
    state_weight = None
    panel = {'group': np.asarray(row_group), 'site_pair': np.asarray(row_pair),
             'background': np.asarray(row_background), 'separation': np.asarray(row_separation),
             'cycle_index': np.asarray(row_cycle, dtype=np.int64),
             'epsilon': np.asarray(epsilon, dtype=float), 'blocks': blocks,
             'cycle_states': np.asarray(cycle_states),
             'independent_site_max_absolute': max(independent_site) if independent_site else None,
             'states': {'features': np.asarray(state_features_rows),
                        'y': np.asarray(state_y, dtype=float),
                        'group': np.asarray(state_group),
                        'background': np.asarray(state_background),
                        'weight': state_weight}}
    from src.capability.readouts.readout_analysis import row_weights
    panel['states']['weight'] = row_weights(panel['states']['background'], panel['states']['group'])
    return panel


def excluded_mask(panel: dict, exclusions: dict) -> np.ndarray:
    """Rows whose cohort cycle rests on an excluded state, by background and index."""

    mask = np.zeros(len(panel['epsilon']), dtype=bool)
    for name, indices in exclusions.items():
        wanted = set(int(index) for index in indices)
        mask |= (panel['background'] == name) & np.isin(panel['cycle_index'], sorted(wanted))
    return mask


def restrict(panel: dict, outcome: dict, keep: np.ndarray) -> tuple[dict, dict]:
    """An evaluation-only view: the fits, folds and predictions are untouched."""

    view = {key: panel[key][keep] for key in
            ('group', 'site_pair', 'background', 'separation', 'cycle_index', 'epsilon')}
    return view, {'predictions': {name: value[keep]
                                  for name, value in outcome['predictions'].items()}}


def evaluate(panel: dict, outcome: dict, support: str, *, draws: int, seed: int) -> dict:
    """Group-equal squared error, within-background Spearman and paired increments."""

    epsilon, groups, pairs = panel['epsilon'], panel['group'], panel['site_pair']
    per_group_mse, per_group_spearman, summary = {}, {}, {}
    for name, prediction in outcome['predictions'].items():
        labels, values = group_errors(epsilon, prediction, groups, pairs)
        per_group_mse[name] = dict(zip(labels.tolist(), values.tolist()))
        _, spearman = group_spearman(epsilon, prediction, groups)
        per_group_spearman[name] = spearman
        summary[name] = {
            'group_equal_mse_kcal2': interval(values, draws=draws, seed=seed),
            'within_background_spearman': interval(spearman, draws=draws, seed=seed),
        }
    labels = sorted(set(groups))
    increments = {}
    for base, augmented in CONTRASTS[support]:
        difference = [per_group_mse[base][label] - per_group_mse[augmented][label] for label in labels]
        record = {'mse_reduction_kcal2': interval(difference, draws=draws, seed=seed)}
        left, right = per_group_spearman[base], per_group_spearman[augmented]
        paired = [None if a is None or b is None else b - a for a, b in zip(left, right)]
        record['spearman_increment'] = (
            {'undefined': 'the baseline prediction is constant, so its rank ordering is undefined'}
            if all(value is None for value in paired)
            else interval(paired, draws=draws, seed=seed))
        record['strata'] = {}
        for stratum in STRATA:
            rows = np.flatnonzero(panel['separation'] == stratum)
            if not len(rows):
                continue
            stratum_labels, base_values = group_errors(
                epsilon[rows], outcome['predictions'][base][rows], groups[rows], pairs[rows])
            _, augmented_values = group_errors(
                epsilon[rows], outcome['predictions'][augmented][rows], groups[rows], pairs[rows])
            record['strata'][stratum] = {
                'groups': len(stratum_labels), 'cycles': int(len(rows)),
                'site_pairs': len(set(pairs[rows])),
                'mse_reduction_kcal2': interval(base_values - augmented_values,
                                                draws=draws, seed=seed)}
        increments[f'{augmented}|{base}'] = record
    return {'designs': summary, 'increments': increments,
            'per_group_mse': per_group_mse, 'per_group_spearman': per_group_spearman}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--expect-plan-sha256', required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--baseline-q', type=Path, required=True)
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--extraction', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--bootstrap', type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument('--supports', default='all,q')
    parser.add_argument('--exclude-evaluation-cycles', type=Path,
                        help='background to cohort-cycle-index map; these cycles are dropped from '
                             'the evaluation only, leaving the fits, folds and predictions untouched')
    args = parser.parse_args()

    if args.arm not in ROSTER:
        raise SystemExit(f'{args.arm} is not on the frozen roster')
    plan = json.loads(args.plan.read_text())
    if plan_digest(plan) != args.expect_plan_sha256:
        raise SystemExit('extraction plan digest does not match the declared value')
    cohort = json.loads(args.cohort.read_text())
    if digest(args.cohort) != plan['cohort_sha256']:
        raise SystemExit('cohort digest does not match the frozen plan')
    baseline = json.loads(args.baseline_q.read_text())
    if digest(args.baseline_q) != plan['provenance']['baseline_q']['sha256']:
        raise SystemExit('baseline Q digest does not match the frozen plan')
    profiles, profile_meta = load_profiles(args.profiles, plan)
    if profile_meta['plan_sha256'] != digest(args.plan):
        raise SystemExit('profile features were not built from this plan file')
    arm_data, manifest = load_arm(args.extraction / args.arm, args.arm, plan)
    exclusions, exclusion_provenance = {}, None
    if args.exclude_evaluation_cycles is not None:
        payload = json.loads(args.exclude_evaluation_cycles.read_text())
        exclusions = {name: entry['cycle_indices'] for name, entry in payload.items()}
        exclusion_provenance = {
            'path': str(args.exclude_evaluation_cycles),
            'sha256': digest(args.exclude_evaluation_cycles),
            'backgrounds': sorted(exclusions),
            'cycles': sum(len(indices) for indices in exclusions.values()),
            'scope': ('dropped from the evaluation only; the fits, the nested folds, the penalty '
                      'selection and the held-out predictions are identical to the unrestricted run, '
                      'so the excluded states still enter training')}

    record = {'schema': 'pairwise_epistasis_fit_v1', 'arm': args.arm,
              'tokenisation_stratum': TOKENISATION_STRATUM[args.arm],
              'extraction_identity': manifest['identity'],
              'extraction_repeat_maxima': {
                  'likelihood_nats': max(r['repeat_likelihood_nats'] for r in manifest['backgrounds']),
                  'projected_feature_relative_l2': max(
                      r['repeat_feature_relative_l2'] for r in manifest['backgrounds']),
                  'meaning': ('run-to-run reproducibility of the identical single-row computation; '
                              'the 0.001 batch gates are inapplicable at batch size one')},
              'code_sha256': {str(path.relative_to(ROOT)): digest(path) for path in (
                  Path(__file__), ROOT / 'src/capability/interactions/pairwise_epistasis.py',
                  ROOT / 'src/capability/readouts/readout_analysis.py',
                  ROOT / 'src/capability/context/profile_increment.py',
                  ROOT / 'src/capability/context/profiles.py')},
              'inputs': {'plan': {'path': str(args.plan), 'content_sha256': args.expect_plan_sha256,
                                  'file_sha256': digest(args.plan)},
                         'cohort_sha256': plan['cohort_sha256'],
                         'baseline_q_sha256': plan['provenance']['baseline_q']['sha256'],
                         'profiles_sha256': digest(args.profiles),
                         'profile_absent_backgrounds': [r['name'] for r in profile_meta['backgrounds']
                                                        if r['status'] != 'present']},
              'recipe': {'split_seeds': list(SPLIT_SEEDS), 'outer_splits': 5, 'inner_splits': 4,
                         'projection': {'seed': plan['projection']['seed'], 'dim': PROJECTION_DIM,
                                        'blocks': list(FEATURE_BLOCKS)},
                         'bootstrap': {'draws': args.bootstrap, 'seed': BOOTSTRAP_SEED,
                                       'unit': 'held group'},
                         'control_sets': {k: list(v) for k, v in CONTROL_SETS.items()},
                         'target': 'epsilon = y_AB - y_A - y_B + y_WT, combined dG_ML, kcal/mol',
                         'weighting': 'group equal; site pairs equal within a group; cycles equal within a site pair'},
              'excluded_evaluation_cycles': exclusion_provenance, 'supports': {}}
    for support in args.supports.split(','):
        groups = set(plan['supports']['all_groups' if support == 'all' else 'q_groups']['groups'])
        panel = build_panel(plan, cohort, profiles, arm_data, baseline, groups)
        identity = row_identity(panel['group'], panel['site_pair'], panel['epsilon'])
        entry = {'row_identity_sha256': identity, 'groups': len(groups),
                 'cycles': int(len(panel['epsilon'])),
                 'site_pairs': len(set(panel['site_pair'])),
                 'kish_effective_site_pairs': round(kish_effective_site_pairs(
                     panel['group'], panel['site_pair']), 1),
                 'kish_weighting': 'groups equal; site pairs equal inside a group, as the fits weight them',
                 'states': int(len(panel['states']['y'])),
                 'independent_site_double_contrast_max_absolute': panel['independent_site_max_absolute'],
                 'block_dimensions': {k: int(v.shape[1]) for k, v in panel['blocks'].items()},
                 'design_dimensions': {
                     name: int(sum(panel['blocks'][b].shape[1] for b in design_blocks(name)
                                   if b != 'G') + (5 if 'G' in design_blocks(name) else 0))
                     for name in design_names(support)},
                 'seeds': {}}
        for seed in SPLIT_SEEDS:
            outcome = nested_compare(panel, support, seed=seed, device=args.device)
            evaluation = evaluate(panel, outcome, support, draws=args.bootstrap, seed=BOOTSTRAP_SEED)
            fold_digest = fold_identity(outcome['folds'])
            restricted = None
            if exclusions:
                keep = ~excluded_mask(panel, exclusions)
                view, outcome_view = restrict(panel, outcome, keep)
                restricted = dict(
                    evaluate(view, outcome_view, support, draws=args.bootstrap, seed=BOOTSTRAP_SEED),
                    groups=len(set(view['group'])), cycles=int(keep.sum()),
                    site_pairs=len(set(view['site_pair'])),
                    row_identity_sha256=row_identity(view['group'], view['site_pair'], view['epsilon']),
                    kish_effective_site_pairs=round(kish_effective_site_pairs(
                        view['group'], view['site_pair']), 1),
                    dropped_cycles=int((~keep).sum()),
                    dropped_groups=sorted(set(panel['group']) - set(view['group'])))
                restricted.pop('per_group_mse'), restricted.pop('per_group_spearman')
            entry['seeds'][str(seed)] = {
                'fold_identity_sha256': fold_digest,
                'folds': [{k: f[k] for k in ('fold', 'held_groups', 'alpha', 'dimensions')}
                          for f in outcome['folds']],
                'nuisance': outcome['nuisance'],
                'designs': evaluation['designs'], 'increments': evaluation['increments'],
                'restricted_evaluation': restricted}
        entry['fold_identity_matches_across_designs'] = True
        record['supports'][support] = entry
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({'arm': args.arm, 'out': str(args.out),
                      'supports': {k: {'groups': v['groups'], 'cycles': v['cycles'],
                                       'site_pairs': v['site_pairs']}
                                   for k, v in record['supports'].items()}}, indent=1))


if __name__ == '__main__':
    main()
