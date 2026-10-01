#!/usr/bin/env python3
"""Refit the published likelihood interaction on residualized double-mutant epsilon.

The label residual subtracts an isotonic response of epsilon to the measured
additive, fitted separately inside every inner and outer training partition. The model
term remains the archived cycle contrast of likelihood. For each split seed the
control designs are refit on that seed's residual, and the reported contrast is
the published one, C_G_T+M over C_G_T.

The support comes from the panel declaration. ``all`` admits every frozen cycle.
``indel-excluded`` restricts the panel by the declared insertion/deletion-construct
exclusion before any fit sees it, so the contaminated rows and the cycles that
rest on them are absent from the fitting side and the evaluation side alike.
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
from src.capability.interactions import pairwise_epistasis as module
from scripts.capability.interactions import fit_pairwise_epistasis as fit


def residual_targets(epsilon, additive, train, query):
    """Fit one training-only response and apply it to both sides of a split."""
    curve, _ = module.measured_singles_predict(
        additive[train], epsilon[train], additive[np.r_[train, query]])
    return (epsilon[train] - curve[:len(train)],
            epsilon[query] - curve[len(train):])


def nested_residual_compare(panel, additive, *, seed, device='cpu'):
    """Nested residual sensitivity; no held labels enter any fitted quantity.

    Training residuals use the training-fitted curve, as in an ordinary fitted
    preprocessing pipeline. Inner validation refits both the label correction
    and G; a globally cross-fitted target must never be reused for this purpose.
    """
    groups, epsilon = panel['group'], panel['epsilon']
    weights = module.row_weights(panel['site_pair'], groups)
    names = ('C_G_T', 'C_G_T+M')
    predictions = {name: np.full(len(groups), np.nan) for name in names}
    target = np.full(len(groups), np.nan)
    records = []
    for outer, held in enumerate(module.family_folds(groups, module.OUTER_SPLITS, seed)):
        train = np.flatnonzero(~np.isin(groups, held))
        test = np.flatnonzero(np.isin(groups, held))
        inner = module.family_folds(groups[train], module.INNER_SPLITS, seed + 100 + outer)
        losses = {name: np.zeros(len(module.ALPHAS)) for name in names}
        for index, validation in enumerate(inner):
            fitting = train[~np.isin(groups[train], validation)]
            valid = train[np.isin(groups[train], validation)]
            yfit, yvalid = residual_targets(epsilon, additive, fitting, valid)
            excluded = np.asarray(list(held) + list(validation))
            calibration_folds = module.family_folds(
                groups[fitting], module.INNER_SPLITS, seed + 1000 + outer * 10 + index)
            g, _ = module.nuisance_cycle(panel['states'], panel['cycle_states'],
                                         excluded, calibration_folds, device=device)
            blocks = dict(panel['blocks'], G=g)
            for name in names:
                x = np.column_stack([blocks[key] for key in module.design_blocks(name)])
                pred = module.ridge_predict(x[fitting], yfit, weights[fitting],
                                            x[valid], module.ALPHAS, device)
                losses[name] += len(validation) * (
                    weights[valid][:, None] * (pred - yvalid[:, None]) ** 2).sum(0)
        ytrain, target[test] = residual_targets(epsilon, additive, train, test)
        g, _ = module.nuisance_cycle(panel['states'], panel['cycle_states'],
                                     np.asarray(held), inner, device=device)
        blocks = dict(panel['blocks'], G=g)
        record = {'fold': outer, 'held_groups': held, 'alpha': {}}
        for name in names:
            alpha = module.ALPHAS[module._select_alpha(losses[name])]
            x = np.column_stack([blocks[key] for key in module.design_blocks(name)])
            predictions[name][test] = module.ridge_predict(
                x[train], ytrain, weights[train], x[test], [alpha], device)[:, 0]
            record['alpha'][name] = alpha
        records.append(record)
    predictions['ADDITIVE_NULL'] = np.zeros(len(groups))
    if not np.isfinite(target).all() or any(not np.isfinite(v).all() for v in predictions.values()):
        raise ValueError('incomplete residual predictions')
    return target, predictions, records


def additive_table(cohort: dict) -> dict:
    table = {}
    for background in cohort['backgrounds']:
        measurements = background['measurements']
        for index, cycle in enumerate(background['cycles']):
            values = [float(measurements[sequence]['value']) for sequence in cycle['sequences']]
            additive = values[1] + values[2] - values[0]
            epsilon = values[3] - values[1] - values[2] + values[0]
            table[(background['name'], index)] = (additive, epsilon)
    return table


def align(panel: dict, table: dict) -> tuple[np.ndarray, np.ndarray]:
    additive, epsilon = [], []
    for background, index in zip(panel['background'], panel['cycle_index']):
        try:
            pair = table[(background, int(index))]
        except KeyError as exc:
            raise ValueError(f'{background} cycle {index} is not in the cohort') from exc
        additive.append(pair[0])
        epsilon.append(pair[1])
    return np.asarray(additive, dtype=float), np.asarray(epsilon, dtype=float)


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError('nonfinite result')
        return number
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--declaration', type=Path)
    parser.add_argument('--exclusion', type=Path,
                        help='state-level indel-exclusion declaration; required by the '
                             'indel-excluded support and rejected by any other')
    args = parser.parse_args()

    root = args.root
    plan_path = root / 'data/pairwise_epistasis/extraction_plan_20260924_e338420f.json'
    cohort_path = root / 'data/pairwise_epistasis/cohort_20260924_8133463e.json'
    baseline_path = root / 'data/pairwise_epistasis/baseline_q.json'
    profiles_path = root / 'data/pairwise_epistasis/profile_features.npz'
    extraction = root / 'results/pairwise_epistasis_20260924/extraction'
    panel_path = root / 'results/pairwise_epistasis_20260924/panel.json'
    input_hashes = {key: fit.digest(path) for key, path in {
        'plan': plan_path, 'cohort': cohort_path, 'baseline': baseline_path,
        'profiles': profiles_path, 'published': panel_path}.items()}
    if args.exclusion is not None:
        input_hashes['exclusion'] = fit.digest(args.exclusion)
    support, declaration_sha256 = module.SUPPORT_ALL, None
    if args.declaration is not None:
        declaration = json.loads(args.declaration.read_text())
        support = declaration['support']
        if args.arm not in declaration['arms'] or input_hashes != declaration['input_sha256']:
            raise SystemExit('arm or input files differ from frozen panel declaration')
        if (fit.digest(Path(__file__)) != declaration['fitter_sha256']
                or fit.digest(Path(module.__file__)) != declaration['module_sha256']):
            raise SystemExit('fitting code differs from frozen panel declaration')
        if fit.digest(extraction / args.arm / f'manifest_{args.arm}.json') != declaration['extractions'][args.arm]['manifest_sha256']:
            raise SystemExit('extraction manifest differs from frozen panel declaration')
        declaration_sha256 = fit.digest(args.declaration)
    plan = json.loads(plan_path.read_text())
    declared = plan['provenance']['baseline_q']['sha256']
    if fit.digest(baseline_path) != declared:
        raise SystemExit('baseline Q digest does not match the extraction plan')
    if fit.digest(cohort_path) != plan['cohort_sha256']:
        raise SystemExit('cohort digest does not match the extraction plan')
    cohort = json.loads(cohort_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    profiles, profile_meta = fit.load_profiles(profiles_path, plan)
    arm_data, manifest = fit.load_arm(extraction / args.arm, args.arm, plan)
    groups = set(plan['supports']['all_groups']['groups'])
    panel = fit.build_panel(plan, cohort, profiles, arm_data, baseline, groups)
    table = additive_table(cohort)
    additive, measured = align(panel, table)
    if not np.allclose(measured, panel['epsilon']):
        raise SystemExit('assembled epsilon does not match the measured cycle epsilon')
    exclusion_accounting = None
    if support == module.SUPPORT_INDEL_EXCLUDED:
        if args.exclusion is None:
            raise SystemExit('the indel-excluded support requires its declared exclusion')
        exclusion = json.loads(args.exclusion.read_text())
        if exclusion['cohort_sha256'] != input_hashes['cohort']:
            raise SystemExit('the exclusion was declared over a different cohort')
        panel, exclusion_accounting = module.indel_excluded_support(panel, plan, exclusion, groups)
        states = panel['states']['y'][panel['cycle_states']]
        additive = states[:, 1] + states[:, 2] - states[:, 0]
        measured = panel['epsilon']
    elif support != module.SUPPORT_ALL:
        raise SystemExit(f'unknown declared support {support!r}')
    elif args.exclusion is not None:
        raise SystemExit('an exclusion was supplied for a support that does not declare one')
    published = json.loads(panel_path.read_text())
    published_arm = published['panel']['all']['C_G_T+M|C_G_T']['arms'][args.arm]
    seeds = {}
    for seed in module.SPLIT_SEEDS:
        residual, predictions, folds = nested_residual_compare(
            panel, additive, seed=seed, device=args.device)
        errors = {name: module.group_errors(residual, prediction, panel['group'],
                   panel['site_pair'])[1] for name, prediction in predictions.items()}
        def summary(values):
            return module.interval(values, draws=module.BOOTSTRAP_DRAWS, seed=module.BOOTSTRAP_SEED)
        seeds[str(seed)] = {
            'residual_group_equal_mse_kcal2': summary(errors['ADDITIVE_NULL']),
            'control_group_equal_mse_kcal2': summary(errors['C_G_T']),
            'augmented_group_equal_mse_kcal2': summary(errors['C_G_T+M']),
            'mse_reduction_kcal2': summary(errors['C_G_T'] - errors['C_G_T+M']),
            'per_group_mse': {name: values.tolist() for name, values in errors.items()},
            'folds': folds,
            'published_mse_reduction_kcal2': published_arm['per_seed'][str(seed)]['mse_reduction_kcal2'],
        }
    report = {
        'schema': 'pairwise_residual_likelihood_interaction_v2',
        'input_sha256': input_hashes,
        'declaration_sha256': declaration_sha256,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'arm': args.arm,
        'script_sha256': fit.digest(Path(__file__)),
        'module_sha256': fit.digest(Path(module.__file__)),
        'cohort_sha256': fit.digest(cohort_path),
        'device': args.device,
        'groups_order': sorted(set(panel['group'])),
        'correction': 'isotonic fit inside every inner and outer training partition',
        'interpretation': 'conditional sensitivity, not purified biological interaction',
        'contrast': 'C_G_T+M|C_G_T',
        'support': support,
        'indel_exclusion': exclusion_accounting,
        'label': 'fold-local isotonic residual of epsilon on the measured additive',
        'likelihood_source': 'archived extraction, not the position-term recomputation',
        'cycles': int(len(measured)),
        'groups': len(set(panel['group'])),
        'site_pairs': len(set(panel['site_pair'])),
        'extraction_torch': manifest.get('torch_version'),
        'seeds': seeds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + '.tmp')
    temporary.write_text(json.dumps(jsonable(report), indent=1) + '\n')
    temporary.replace(args.out)
    print(json.dumps({
        'arm': args.arm,
        'out': str(args.out),
        'mse_reduction_kcal2': {seed: row['mse_reduction_kcal2']['point'] for seed, row in seeds.items()},
    }, indent=1))


if __name__ == '__main__':
    main()
