#!/usr/bin/env python3
"""Cross-fit epsilon on the measured single-mutant additive, by held-out family.

The curve at each outer fold is fit on that fold's training groups only. The
in-sample curve is computed afterwards as a comparison and is never substituted
when the recorded family partition does not reproduce. No model is loaded and no
held-out design prediction is invented.
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
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, MEASURED_SINGLES_BINS, OUTER_SPLITS, SPLIT_SEEDS,
    group_errors, interval, measured_singles_correction, measured_singles_predict,
    require_recorded_family_folds)

COHORT = ROOT / 'archive/logs/R3/pairwise_cohort_20260924/cohort.json'
FIT = ROOT / 'archive/logs/R3/pairwise_epistasis_20260924/panel/fit_progen3-3b.json'
PANEL = ROOT / 'archive/logs/R3/pairwise_epistasis_20260924/panel/panel.json'
OUT = ROOT / 'results/R3/pairwise_measured_singles_20260926/measured_singles_crossfit.json'

#: Identity of the admitted 64-group MegaScale cycle cohort. A different support
#: is a different experiment and must not be written under this result.
ADMITTED = {'groups': 64, 'cycles': 8192, 'site_pairs': 217,
            'wild_types': 64, 'singles': 4721, 'doubles': 8192}
PRIMARY_METHOD = 'isotonic'
COMPARISON_METHOD = 'quantile_bins'
PUBLISHED_INCREMENT_ORDER_KCAL2 = 0.001
RESOLVED_ARMS = ('progen3-3b', 'prollama', 'protgpt2')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_file(path: Path, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f'{role} is missing: {path}')


def load_cycle_table(cohort: dict) -> dict:
    """Measured additive and epsilon for every admitted cycle, checked against the cohort."""

    if 'backgrounds' not in cohort:
        raise ValueError('cohort has no backgrounds')
    groups, pairs, additive, epsilon = [], [], [], []
    wild_types = singles = doubles = 0
    for background in cohort['backgrounds']:
        name, group = background['name'], background['group']
        measurements = background['measurements']
        cycles = background['cycles']
        if not cycles:
            raise ValueError(f'{name} has no cycles')
        wild = cycles[0]['sequences'][0]
        for sequence, record in measurements.items():
            if len(sequence) != len(wild):
                raise ValueError(f'{name}: a measurement length differs from the wild type')
            differences = sum(left != right for left, right in zip(sequence, wild))
            if differences == 0:
                wild_types += 1
            elif differences == 1:
                singles += 1
            elif differences == 2:
                doubles += 1
            else:
                raise ValueError(f'{name}: a measurement is not a wild type, single or double')
            if not np.isfinite(record['value']):
                raise ValueError(f'{name}: a measurement is nonfinite')
        for cycle in cycles:
            sequences = cycle['sequences']
            if len(sequences) != 4 or sequences[0] != wild:
                raise ValueError(f'{name}: a cycle is not wild type, two singles and a double')
            try:
                values = [float(measurements[sequence]['value']) for sequence in sequences]
            except KeyError as exc:
                raise ValueError(f'{name}: a cycle sequence has no measurement') from exc
            recomputed = values[3] - values[1] - values[2] + values[0]
            if abs(recomputed - float(cycle['epsilon'])) > 1e-9:
                raise ValueError(f'{name}: stored epsilon disagrees with the four measurements')
            positions = cycle['positions']
            groups.append(group)
            pairs.append(f'{name}:{int(positions[0])}-{int(positions[1])}')
            additive.append(values[1] + values[2] - values[0])
            epsilon.append(recomputed)
    table = {
        'group': np.asarray(groups),
        'site_pair': np.asarray(pairs),
        'additive': np.asarray(additive, dtype=float),
        'epsilon': np.asarray(epsilon, dtype=float),
        'counts': {
            'groups': len(set(groups)), 'cycles': len(epsilon), 'site_pairs': len(set(pairs)),
            'wild_types': wild_types, 'singles': singles, 'doubles': doubles,
        },
    }
    if table['counts'] != ADMITTED:
        raise ValueError(f'cohort identity {table["counts"]} is not the admitted support {ADMITTED}')
    return table


def _squared_error(epsilon: np.ndarray, prediction: np.ndarray, groups, pairs) -> dict:
    """Group-equal site-pair-equal mean squared error, with the pairwise bootstrap."""

    _, values = group_errors(epsilon, prediction, groups, pairs)
    return interval(values, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)


def _label_summary(epsilon: np.ndarray, residual: np.ndarray, groups, pairs,
                   *, before_values: np.ndarray) -> dict:
    _, after_values = group_errors(residual, np.zeros(len(residual)), groups, pairs)
    before_variance = float(epsilon.var())
    after_variance = float(residual.var())
    if before_variance <= 0:
        raise ValueError('epsilon variance is zero, so an absorbed fraction is undefined')
    return {
        'group_equal_mse_kcal2_per_mol2': interval(after_values, draws=BOOTSTRAP_DRAWS,
                                                   seed=BOOTSTRAP_SEED),
        'group_equal_mse_reduction_kcal2_per_mol2': interval(
            before_values - after_values, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED),
        'unweighted_variance_kcal2_per_mol2': after_variance,
        'unweighted_sd_kcal_per_mol': float(residual.std(ddof=1)),
        'absorbed_unweighted_variance_fraction': float(1.0 - after_variance / before_variance),
    }


def _fold_record(outcome: dict) -> list[dict]:
    rows = []
    for fold in outcome['folds']:
        row = {
            'fold': fold['fold'],
            'training_cycles': fold['training_cycles'],
            'held_out_cycles': fold['held_out_cycles'],
        }
        for key in ('increasing', 'constant_training_response', 'empty_bins_filled'):
            if key in fold and fold[key] is not None:
                row[key] = fold[key]
        rows.append(row)
    return rows


def _published_likelihood(panel: dict) -> dict:
    """Copy the recorded likelihood-interaction summary. This does not refit it."""

    block = panel['panel']['all']['C_G_T+M|C_G_T']
    arms = {}
    for arm in RESOLVED_ARMS:
        per_seed = {}
        for seed, entry in block['arms'][arm]['per_seed'].items():
            per_seed[seed] = {
                'mse_reduction_kcal2_per_mol2': entry['mse_reduction_kcal2'],
                'interval': entry['interval'],
                'excludes_zero': entry['excludes_zero'],
            }
        arms[arm] = {'per_seed': per_seed, 'point_range': block['arms'][arm]['point_range'],
                     'seeds_with_interval_above_zero': block['arms'][arm]['seeds_with_interval_above_zero']}
    return {
        'role': ('scale reference copied from the published panel summary; these increments '
                 'were not recomputed on the corrected label'),
        'contrast': 'C_G_T+M|C_G_T',
        'support': 'all',
        'unit': 'kcal^2/mol^2',
        'arms_with_every_seed_interval_above_zero': block['arms_with_every_seed_interval_above_zero'],
        'arms_in_the_published_summary': len(block['arms']),
        'copied_arms': arms,
    }


def _model_score_gap() -> dict:
    """Where a per-cycle likelihood refit would have to read, and what is actually there."""

    missing = []
    production = ROOT / 'results/pairwise_epistasis_20260924'
    if not production.exists():
        missing.append('results/pairwise_epistasis_20260924/')
    panel_dir = ROOT / 'archive/logs/R3/pairwise_epistasis_20260924/panel'
    for arm in RESOLVED_ARMS:
        path = panel_dir / f'fit_{arm}.json'
        if not path.is_file():
            missing.append(str(path.relative_to(ROOT)))
    extraction = ROOT / 'archive/logs/R3/pairwise_epistasis_20260924/local_extraction'
    present = sorted(path.name for path in extraction.iterdir()) if extraction.is_dir() else []
    for arm in RESOLVED_ARMS:
        if arm not in present:
            missing.append(f'archive/logs/R3/pairwise_epistasis_20260924/local_extraction/{arm}/')
    return {
        'missing_files': missing,
        'present_without_per_cycle_predictions': [
            'archive/logs/R3/pairwise_epistasis_20260924/panel/fit_progen3-3b.json '
            'stores fold membership, selected penalties and summary increments',
            'archive/logs/R3/pairwise_epistasis_20260924/panel/panel.json '
            'stores the published summary copied below as a scale reference',
        ],
        'local_extraction_arms_present': present,
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError('result contains a nonfinite number')
        return number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f'cannot record a {type(value).__name__}')


def _scale(residual_points: list[float], published_points: list[float]) -> dict:
    largest_published = max(published_points)
    smallest_residual = min(residual_points)
    if all(point > 10.0 * PUBLISHED_INCREMENT_ORDER_KCAL2 for point in residual_points):
        judgement = 'large'
    elif all(point < PUBLISHED_INCREMENT_ORDER_KCAL2 for point in residual_points):
        judgement = 'small'
    else:
        judgement = 'comparable'
    return {
        'judgement': judgement,
        'rule': ('large when every primary-method seed has cross-fitted group-equal mean squared '
                 'error above 10 times 0.001 kcal^2/mol^2; small when every seed is below 0.001'),
        'reference_order_kcal2_per_mol2': PUBLISHED_INCREMENT_ORDER_KCAL2,
        'primary_residual_group_equal_mse_kcal2_per_mol2': residual_points,
        'smallest_residual_over_0.001': smallest_residual / PUBLISHED_INCREMENT_ORDER_KCAL2,
        'largest_copied_likelihood_increment_kcal2_per_mol2': largest_published,
        'smallest_residual_over_largest_copied_increment': smallest_residual / largest_published,
    }


def build_report(cohort_path: Path, fit_path: Path, panel_path: Path) -> dict:
    _require_file(cohort_path, 'cohort')
    _require_file(fit_path, 'recorded fold map')
    _require_file(panel_path, 'published panel summary')
    cohort = json.loads(cohort_path.read_text())
    fit = json.loads(fit_path.read_text())
    panel = json.loads(panel_path.read_text())
    try:
        support = fit['supports']['all']
    except KeyError as exc:
        raise ValueError('recorded fit has no support "all" fold map') from exc
    if support.get('groups') != ADMITTED['groups'] or support.get('cycles') != ADMITTED['cycles']:
        raise ValueError('recorded fit support does not match the admitted 64-group cohort')
    table = load_cycle_table(cohort)
    epsilon, additive = table['epsilon'], table['additive']
    groups, pairs = table['group'], table['site_pair']
    _, before_values = group_errors(epsilon, np.zeros(len(epsilon)), groups, pairs)
    additive_null = interval(before_values, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    recorded_null = support['seeds'][str(SPLIT_SEEDS[0])]['designs']['ADDITIVE_NULL']['group_equal_mse_kcal2']
    if additive_null['point'] != recorded_null['point'] or additive_null['interval'] != recorded_null['interval']:
        raise ValueError(
            'recomputed additive-null mean squared error does not match the frozen fit; '
            'no residual is written')

    partitions = {}
    for seed in SPLIT_SEEDS:
        seed_record = support['seeds'][str(seed)]
        recorded_held = [fold['held_groups'] for fold in seed_record['folds']]
        checked = require_recorded_family_folds(
            groups, recorded_held, n_splits=OUTER_SPLITS, seed=seed)
        if checked['fold_identity_sha256'] != seed_record['fold_identity_sha256']:
            raise ValueError(f'fold identity at seed {seed} does not match the recorded digest')
        partitions[seed] = checked

    cross_fit = {}
    for method in (PRIMARY_METHOD, COMPARISON_METHOD):
        per_seed = {}
        for seed, checked in partitions.items():
            outcome = measured_singles_correction(
                epsilon, additive, groups, checked['held_groups'],
                method=method, bins=MEASURED_SINGLES_BINS)
            per_seed[str(seed)] = {
                **_label_summary(epsilon, outcome['residual'], groups, pairs,
                                 before_values=before_values),
                'folds': _fold_record(outcome),
            }
        cross_fit[method] = per_seed

    in_sample = {}
    for method in (PRIMARY_METHOD, COMPARISON_METHOD):
        prediction, info = measured_singles_predict(
            additive, epsilon, additive, method=method, bins=MEASURED_SINGLES_BINS)
        in_sample[method] = {
            'fit_on': 'all 64 groups, including the cycles being scored',
            'role': 'comparison, not the primary residual',
            'fit_info': info,
            **_label_summary(epsilon, epsilon - prediction, groups, pairs, before_values=before_values),
        }

    published = _published_likelihood(panel)
    if published['arms_with_every_seed_interval_above_zero'] != list(RESOLVED_ARMS):
        raise ValueError('published resolved-arm list is not the three arms this comparison cites')
    published_points = [
        entry['mse_reduction_kcal2_per_mol2']
        for arm in published['copied_arms'].values()
        for entry in arm['per_seed'].values()]
    residual_points = [
        cross_fit[PRIMARY_METHOD][str(seed)]['group_equal_mse_kcal2_per_mol2']['point']
        for seed in SPLIT_SEEDS]
    before_variance = float(epsilon.var())
    return {
        'schema': 'pairwise_measured_singles_crossfit_v1',
        'primary_method': PRIMARY_METHOD,
        'comparison_method': COMPARISON_METHOD,
        'primary_method_reason': (
            'On this cohort epsilon decreases with the measured additive, so the primary '
            'curve is an isotonic regression whose direction is chosen on each fold\'s '
            'training groups. The twenty-bin mean is the label instrument\'s estimator; '
            'it is reported in-sample and cross-fitted as the comparison, not as a fallback.'),
        'target': 'epsilon = y_AB - y_A - y_B + y_WT',
        'regressor': 'measured additive = y_A + y_B - y_WT',
        'stability_source': 'measurements[sequence].value, the median combined dG_ML',
        'epsilon_unit': 'kcal/mol',
        'squared_error_unit': 'kcal^2/mol^2',
        'counts': table['counts'],
        'inputs': {
            'cohort': {'path': str(cohort_path.relative_to(ROOT)), 'sha256': _sha256(cohort_path)},
            'recorded_folds': {'path': str(fit_path.relative_to(ROOT)), 'sha256': _sha256(fit_path),
                               'support': 'all'},
            'published_panel_summary': {'path': str(panel_path.relative_to(ROOT)),
                                        'sha256': _sha256(panel_path)},
        },
        'fold_rule': {
            'splitter': 'readout_analysis.family_folds',
            'outer_splits': OUTER_SPLITS,
            'seeds': list(SPLIT_SEEDS),
            'group_universe': 'the 64 groups of the admitted cohort',
            'reproduction': ('each seed\'s held-group lists equal the lists recorded in the fit, '
                             'and the fold-identity digest equals the recorded digest'),
            'held_out_family_labels_enter_the_curve': False,
            'in_sample_curve_used_as_fallback': False,
            'seeds_checked': {
                str(seed): {'fold_identity_sha256': checked['fold_identity_sha256'],
                            'folds': len(checked['held_groups'])}
                for seed, checked in partitions.items()},
        },
        'curve_fit_weighting': (
            'each cycle equally. Every admitted group contributes 128 cycles, so the groups '
            'contribute equally to the curve'),
        'reported_mse_weighting': (
            'group equal; site pairs equal within a group; cycles equal within a site pair. '
            'This is group_errors, the weighting of the published pairwise mean squared error'),
        'bootstrap': {'draws': BOOTSTRAP_DRAWS, 'seed': BOOTSTRAP_SEED, 'unit': 'group'},
        'epsilon_before_correction': {
            'pearson_r_with_measured_additive': float(np.corrcoef(epsilon, additive)[0, 1]),
            'unweighted_variance_kcal2_per_mol2': before_variance,
            'unweighted_sd_kcal_per_mol': float(epsilon.std(ddof=1)),
            'group_equal_mse_kcal2_per_mol2': additive_null,
            'matches_frozen_additive_null': True,
        },
        'in_sample_comparison': in_sample,
        'cross_fit': cross_fit,
        'residual_scale_relative_to_published_likelihood_increment': _scale(
            residual_points, published_points),
        'model_likelihood_increments': {
            'recomputed': False,
            'cross_fitted_on_the_corrected_label': False,
            'updated_resolved_arm_count': None,
            **_model_score_gap(),
            'published_reference_not_refit': published,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, default=COHORT)
    parser.add_argument('--fit', type=Path, default=FIT)
    parser.add_argument('--panel', type=Path, default=PANEL)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    report = build_report(args.cohort, args.fit, args.panel)
    payload = json.dumps(_jsonable(report), indent=1) + '\n'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + '.tmp')
    temporary.write_text(payload)
    temporary.replace(args.out)
    primary = report['cross_fit'][PRIMARY_METHOD]
    print(json.dumps({
        'out': str(args.out),
        'primary_method': PRIMARY_METHOD,
        'residual_scale': report['residual_scale_relative_to_published_likelihood_increment']['judgement'],
        'group_equal_mse_kcal2_per_mol2': {
            seed: primary[seed]['group_equal_mse_kcal2_per_mol2']['point'] for seed in primary},
        'model_increments_recomputed': False,
    }, indent=1))


if __name__ == '__main__':
    main()
