#!/usr/bin/env python3
"""Reaggregate frozen stability fits; paired bidirectional inference, CPU only.

The per-arm fits and their out-of-fold predictions are frozen inputs: this stage
adds the missing likelihood-only readout where a fit is absent and otherwise
re-reads completed records, then forms the declared family of paired reductions.
The primary family is the joint maximum absolute centered bootstrap deviation in
each endpoint's native units, as the run contract states; the project's
fixed-standard-error maxT family is retained beside it as an explicitly labelled
sensitivity, because the two give different bands and different positive counts.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.capability.interactions.run_residual_panel import simultaneous_bands
from scripts.capability.stability.fit_stability_ranking_panel import locate_extractions
from src.capability.core.io import sha256_file
from src.capability.extensions import bidirectional
from src.capability.extensions.stability_followups import COHORT_SHA256, verify_scalar_archive
from src.capability.interactions.pairwise_epistasis import ROSTER
from src.capability.stability import stability_gate as gate, gate_inputs, ranking_error as metric

DEFAULT_SOURCE = ROOT / 'results/extensions/bidirectional_ablation_20261106/r2/stability'
DEFAULT_OUT = ROOT / 'results/extensions/bidirectional_ablation_20261106/r2/stability-native-bands'
OLD = ROOT / 'results/external_baseline/20261008095614_7e8ae4b18a4d/stability_ranking_error_grid_cpu'
PANEL_SCHEMA = 'bidirectional_stability'
CONTRACT_SCHEMA = 'bidirectional_stability_contract'
#: The panel labels this reader accepts: the stable name and the one already written
#: for this declaration. Any other label is refused rather than stripped of a suffix.
PANEL_SCHEMAS = (PANEL_SCHEMA, 'bidirectional_stability_v2')

#: Endpoint readings and the native units each contrasted quantity carries.
READINGS = {'ranking': 'within-background Spearman rho',
            'transfer_error': 'squared kcal/mol; group-equal, site-equal MSE',
            'calibrated_error': 'squared kcal/mol; site-calibrated group-equal MSE'}
DESIGNS = {'M': ['scalar_likelihood_difference'],
           'S': ['ident', 'geom', 'chem', 'G'],
           'S2': ['ident', 'geom', 'chem', 'G', 'comp', 'prof2']}

SOURCE = DEFAULT_SOURCE
OUT = DEFAULT_OUT
STATE = {}


def configure(*, source=None, out=None):
    """Resolve the read-only fit source and the result directory for this process."""

    global SOURCE, OUT
    SOURCE = Path(source or DEFAULT_SOURCE).resolve()
    OUT = Path(out or DEFAULT_OUT).resolve()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def initialise():
    torch.set_num_threads(1)
    paths = gate_inputs.resolve(ROOT / 'results/R4/gate_stability_20260924')
    cohort = json.loads(paths['cohort.json'].read_text())
    assert sha256_file(paths['cohort.json']) == COHORT_SHA256
    profiles, _ = gate.load_profiles(paths['profile_features.npz'],
                                     {b['name']: b['wildtype'] for b in cohort['backgrounds']})
    panel = gate.build_panel(cohort, profiles)
    assert len(panel['target']) == 25856 and len(set(panel['group'])) == 101
    STATE.update(cohort=cohort, panel=panel)


def run_arm(task):
    """Fit the likelihood-only readout on the archived folds, or re-read its record."""

    arm, directory = task
    start = time.monotonic()
    target = OUT / 'fits' / (arm + '.json')
    if target.exists():
        return json.loads(target.read_text())
    panel = STATE['panel']
    oldrec = json.loads((OLD / 'fits' / f'fit_{arm}.json').read_text())
    oldpath = OLD / 'oof' / f'oof_{arm}.npz'
    assert sha256_file(oldpath) == oldrec['held_out_predictions']['sha256']
    scalar, manifest = verify_scalar_archive(Path(directory), arm, STATE['cohort'], COHORT_SHA256)
    assert sha256_file(Path(directory) / f'manifest_{arm}.json') == oldrec['extraction_manifest_sha256']
    old = np.load(oldpath, allow_pickle=False)
    for key, value in [('target', panel['target']), ('group', panel['group']), ('sample_id', panel['site'])]:
        assert np.array_equal(old[key], value), key
    retained, seeds, checks = {}, {}, []
    for seed in gate.SPLIT_SEEDS:
        fit = gate.fold_predictions(panel, {'M': scalar}, {'M': ('M',)}, seed=seed, device='cpu')
        folds = [f['held_groups'] for f in fit['folds']]
        assert folds == oldrec['seeds'][str(seed)]['held_groups']
        retained[f'{seed}|M'] = fit['predictions']['M']
        predictions = {k: old[f'{seed}|{k}'] for k in ['S', 'S_M', 'S2', 'S2_M']}
        predictions['M'] = fit['predictions']['M']
        records = {}
        for name, pred in predictions.items():
            labels, rank = gate.group_spearman(panel['target'], pred, panel['group'])
            assert labels == sorted(set(panel['group']))
            _, mse = gate.group_errors(panel['target'], pred, panel['group'], panel['site'])
            calibrated, calibration = metric.family_calibrated(pred, panel['target'],
                                                              panel['group'], panel['site'])
            _, cmse = gate.group_errors(panel['target'], calibrated, panel['group'], panel['site'])
            records[name] = {'ranking': rank, 'transfer_error': mse.tolist(),
                             'calibrated_error': cmse.tolist()}
        for baseline, prefix in [('S', 'matched'), ('S2', 'profile')]:
            for reading in ['ranking', 'transfer_error', 'calibrated_error']:
                sign = 1 if reading == 'ranking' else -1
                value = sign * (np.array(records[baseline + '_M'][reading])
                                - np.array(records[baseline][reading]))
                frozen = oldrec['seeds'][str(seed)]['contrasts']['combined'][prefix + '_' + reading]
                difference = float(np.max(np.abs(value - np.asarray(frozen))))
                assert difference < 1e-10, (arm, reading, difference)
                checks.append(difference)
        seeds[str(seed)] = {'metrics': records, 'folds': fit['folds']}
    npz = OUT / 'oof' / f'{arm}.npz'
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, target=panel['target'], group=panel['group'],
                        sample_id=panel['site'], **retained)
    record = {'arm': arm, 'status': 'complete', 'seconds': time.monotonic() - start,
              'groups': sorted(set(panel['group'])), 'seeds': seeds,
              'old_oof_sha256': sha256_file(oldpath), 'new_oof_sha256': sha256_file(npz),
              'scalar_manifest_sha256': sha256_file(Path(directory) / f'manifest_{arm}.json'),
              'validation': {'historical_forward_max_absolute_difference': max(checks),
                             'same_rows_and_folds': True},
              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write(target, record)
    print(json.dumps({'finished': arm, 'seconds': record['seconds']}), flush=True)
    return record


def load_records(source, arms):
    """Completed per-arm fit records whose OOF archives still match their digests."""

    records, missing = [], []
    for arm in arms:
        fit = source / 'fits' / f'{arm}.json'
        oof = source / 'oof' / f'{arm}.npz'
        if not fit.exists() or not oof.exists():
            missing.append(arm)
            continue
        record = json.loads(fit.read_text())
        if record.get('status') != 'complete' or record.get('arm') != arm:
            raise ValueError(f'{arm}: {fit} is not a completed record of that arm')
        if sha256_file(oof) != record['new_oof_sha256']:
            raise ValueError(f'{arm}: OOF archive {oof} differs from its recorded digest')
        records.append(record)
    if missing:
        raise ValueError(f'{source} holds no completed fit for {len(missing)} arms ({missing}); '
                         'an implicit refit is refused, pass --fit only to build a fresh panel')
    return records


def interval_matrix(values, draws=bidirectional.DRAWS):
    """Pointwise percentile interval, one column each, from the same group draws."""

    values = np.asarray(values, dtype=float)
    assert np.isfinite(values).all()
    rng = np.random.default_rng(bidirectional.SEED)
    out = []
    for start in range(0, draws, bidirectional.DRAW_CHUNK):
        size = min(bidirectional.DRAW_CHUNK, draws - start)
        indices = rng.integers(len(values), size=(size, len(values)))
        out.append(values[indices].mean(axis=1))
    return np.quantile(np.concatenate(out), [.025, .975], axis=0).T.tolist()


def aggregate(records):
    """Form both directions for every control set and metric; write the declared family."""

    records = sorted(records, key=lambda d: d['arm'])
    arms = [d['arm'] for d in records]
    if arms != sorted(ROSTER) or len(records) != len(ROSTER):
        raise ValueError(f'aggregation needs the frozen {len(ROSTER)}-arm roster')
    groups = records[0]['groups']
    if any(r['groups'] != groups for r in records):
        raise ValueError('arms do not share one held-group support')
    panels, matrices = {}, {}
    for baseline in ('S', 'S2'):
        for reading, units in READINGS.items():
            # A paired reduction is positive when the joint arm lowers the loss, so a
            # rank correlation enters as its negative and a squared error as itself.
            to_loss = -1 if reading == 'ranking' else 1
            designs = {'model_given_controls': (baseline, baseline + '_M'),
                       'controls_given_model': ('M', baseline + '_M')}
            names = (baseline, 'M', baseline + '_M')
            columns, cells, absolute, abscols, perseeds = [], [], [], [], {}
            for record in records:
                metrics = {seed: record['seeds'][str(seed)]['metrics'] for seed in gate.SPLIT_SEEDS}
                means = {name: np.mean([np.asarray(metrics[seed][name][reading], dtype=float)
                                        for seed in gate.SPLIT_SEEDS], axis=0) for name in names}
                for name in names:
                    absolute.append(means[name])
                    abscols.append(record['arm'] + '|' + name)
                cells.append(bidirectional.directional_reductions(
                    {name: to_loss * means[name] for name in names}, designs))
                columns.extend(record['arm'] + '|' + direction for direction in designs)
                for index, direction in enumerate(designs):
                    label = record['arm'] + '|' + direction
                    perseeds[label] = {
                        str(seed): float(np.mean(bidirectional.directional_reductions(
                            {name: to_loss * np.asarray(metrics[seed][name][reading], dtype=float)
                             for name in names}, designs)[:, index]))
                        for seed in gate.SPLIT_SEEDS}
            matrix = np.column_stack(cells)
            absolutes = np.column_stack(absolute)
            key = baseline + '|' + reading
            matrices[key] = matrix
            primary = bidirectional.paired_bands(matrix)
            values = bidirectional.verdict(primary['interval'])
            verdicts = {}
            for direction in bidirectional.DIRECTIONS:
                cells_found = [(label.rsplit('|', 1)[0], value)
                               for label, value in zip(columns, values)
                               if label.rsplit('|', 1)[1] == direction]
                verdicts[direction] = {
                    name: [arm for arm, value in cells_found if value == name]
                    for name in ('positive', 'negative', 'unresolved')}
                verdicts[direction]['counts'] = bidirectional.counts(v for _, v in cells_found)
            panels[key] = {
                'units': units, 'columns': columns,
                'simultaneous': dict(primary, units=units),
                'standardized_maxT_sensitivity': dict(
                    simultaneous_bands(matrix, draws=bidirectional.DRAWS, seed=bidirectional.SEED),
                    units='multiples of the fixed column standard error; retained sensitivity only'),
                'pointwise_interval': interval_matrix(matrix),
                'per_seed_points': perseeds,
                'verdicts': verdicts,
                'absolute': {'columns': abscols, 'points': absolutes.mean(axis=0).tolist(),
                             'pointwise_interval': interval_matrix(absolutes)}}
    np.savez_compressed(OUT / 'group_contrasts.npz', groups=np.array(groups), **matrices)
    write(OUT / 'panel.json', {
        'schema': PANEL_SCHEMA, 'arms': arms, 'groups': groups,
        'contract_sha256': sha256_file(OUT / 'contract.json'),
        'method': {'primary': bidirectional.NATIVE_METHOD,
                   'sensitivity': bidirectional.STANDARDIZED_METHOD,
                   'family': f'{bidirectional.FAMILY_SIZE} contrasts = 2 directions x 33 arms, '
                             'separately per control set and metric',
                   'unit': '101 families; seeds averaged within family'},
        'panels': panels})
    within = [record['validation']['historical_forward_max_absolute_difference'] for record in records]
    write(OUT / 'completion.json', {
        'status': 'complete', 'models': len(arms), 'seeds': len(gate.SPLIT_SEEDS),
        'endpoint': 'stability', 'gpu_used': False,
        'primary_method': bidirectional.NATIVE_METHOD,
        'sensitivity_method': bidirectional.STANDARDIZED_METHOD,
        'panel_sha256': sha256_file(OUT / 'panel.json'),
        'validation_max_historical_difference': max(within),
        'positive_counts': {key: {direction: panel['verdicts'][direction]['counts']['positive']
                                  for direction in bidirectional.DIRECTIONS}
                            for key, panel in panels.items()}})


def contract(source=None, arms=None, workers=None):
    """One declaration of support, designs, family and units for this output."""

    reused = source is not None
    return {
        'schema': CONTRACT_SCHEMA,
        'support': {'rows': 25856, 'groups': 101},
        'designs': DESIGNS,
        'G': 'training-fold derived nonlinear additive response; belongs to C, not M',
        'seeds': list(gate.SPLIT_SEEDS),
        'bootstrap': {'draws': bidirectional.DRAWS, 'seed': bidirectional.SEED,
                      'method': bidirectional.NATIVE_METHOD, 'units': 'native',
                      'family': f'{bidirectional.FAMILY_SIZE} contrasts = 2 directions x 33 arms, '
                                'separately per control set and metric',
                      'unit': '101 families; seeds averaged within family'},
        'sensitivity': {'method': bidirectional.STANDARDIZED_METHOD,
                        'role': 'fixed-standard-error studentised maxT over the same draws; retained '
                                'sensitivity only, because its bands and its positive counts differ '
                                'from the declared native family'},
        'contrasts': {'model_given_controls': 'MSE(S) - MSE(S+M) or rho(S+M) - rho(S), '
                                              'and the same reading for S2',
                      'controls_given_model': 'MSE(M) - MSE(S+M) or rho(S+M) - rho(M), '
                                              'and the same reading for S2',
                      'verdict': 'positive when the simultaneous interval excludes zero above; '
                                 'negative when it excludes zero below; otherwise unresolved'},
        'calibration': '4-fold within-target-family site calibration; not cross-family zero-shot transfer',
        'historical_limitation': 'the archived fits let the outer training partition enter the inner '
                                 'validation partition when G is refitted; recorded, not silently changed',
        'reuse': ('completed per-arm fits and OOF archives were read from the frozen source and never '
                  'refitted' if reused else 'per-arm likelihood-only fits were computed in this output'),
        'source': None if source is None else str(Path(source).resolve().relative_to(ROOT)),
        'source_fit_sha256': {} if source is None else
        {arm: sha256_file(Path(source) / 'fits' / f'{arm}.json') for arm in arms},
        'resource': {'workers': workers, 'blas_threads_per_worker': 1, 'gpu': False},
        'arms_requested': arms}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE,
                        help='read-only directory holding the completed per-arm fits')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--arms', nargs='*')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--fit', action='store_true',
                        help='compute missing likelihood-only fits into --out instead of reusing --source')
    args = parser.parse_args()
    configure(source=args.source, out=args.out)
    arms = sorted(args.arms or ROSTER)
    if not args.fit:
        records = load_records(SOURCE, arms)
        write(OUT / 'contract.json', contract(SOURCE, arms, workers=None))
        if len(arms) == len(ROSTER):
            aggregate(records)
        else:
            write(OUT / 'smoke.json', {'models': len(records), 'source': str(SOURCE)})
        return
    located = locate_extractions([ROOT / 'results/external_baseline/20261008022745_762304146fdd'], arms)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'contract.json', contract(None, arms, workers=args.workers))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialise) as pool:
        records = list(pool.map(run_arm, [(arm, str(located[arm])) for arm in arms]))
    if len(arms) == len(ROSTER):
        aggregate(records)
    else:
        write(OUT / 'smoke.json', {'models': len(records),
                                   'seconds': [record['seconds'] for record in records]})


if __name__ == '__main__':
    main()