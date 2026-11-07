#!/usr/bin/env python3
"""Validate coverage, reused fits, native bands, label isolation and verdict counts."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.capability.extensions.bidirectional_ablation_20261106 import r2_stability as run
from src.capability.extensions import bidirectional

SLACK = 1e-12


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--source', type=Path, default=run.DEFAULT_SOURCE)
    parser.add_argument('--out', type=Path, default=run.DEFAULT_OUT)
    args = parser.parse_args()
    start = time.monotonic()
    run.configure(source=args.source, out=args.out)
    run.initialise()
    panel_source, out = run.SOURCE, run.OUT
    panel = run.STATE['panel']
    records = [json.loads((panel_source / 'fits' / f'{arm}.json').read_text())
               for arm in sorted(run.ROSTER)]
    report = json.loads((out / 'panel.json').read_text())
    completion = json.loads((out / 'completion.json').read_text())
    assert report['schema'] in run.PANEL_SCHEMAS
    assert completion['status'] == 'complete'
    assert completion['panel_sha256'] == run.sha256_file(out / 'panel.json')
    assert report['contract_sha256'] == run.sha256_file(out / 'contract.json')
    assert len(records) == 33 and all(record['status'] == 'complete' for record in records)

    # The frozen per-arm fits are reused unchanged: identical OOF rows and folds.
    for record in records:
        path = panel_source / 'oof' / (record['arm'] + '.npz')
        assert run.sha256_file(path) == record['new_oof_sha256']
        with np.load(path, allow_pickle=False) as archive:
            assert np.array_equal(archive['target'], panel['target'])
            assert np.array_equal(archive['group'], panel['group'])
            for seed in run.gate.SPLIT_SEEDS:
                prediction = archive[f'{seed}|M']
                assert prediction.shape == (25856,) and np.isfinite(prediction).all()

    # Every declared point is the seed-mean paired reduction of the saved metrics,
    # and every band is the declared native statistic, not a studentised one.
    contrast_difference = 0.
    for key, entry in report['panels'].items():
        assert entry['simultaneous']['method'] == bidirectional.NATIVE_METHOD
        assert entry['simultaneous']['family_size'] == bidirectional.FAMILY_SIZE
        assert entry['simultaneous']['groups'] == 101
        assert entry['standardized_maxT_sensitivity']['method'] == bidirectional.STANDARDIZED_METHOD
        assert len(entry['columns']) == bidirectional.FAMILY_SIZE
        for index, label in enumerate(entry['columns']):
            arm, direction = label.rsplit('|', 1)
            assert arm in run.ROSTER and direction in bidirectional.DIRECTIONS
            record = next(record for record in records if record['arm'] == arm)
            baseline = key.split('|')[0]
            designs = {'model_given_controls': (baseline, baseline + '_M'),
                       'controls_given_model': ('M', baseline + '_M')}
            reduced, joint = designs[direction]
            reading = key.split('|')[1]
            to_loss = -1 if reading == 'ranking' else 1
            losses = []
            for seed in run.gate.SPLIT_SEEDS:
                metrics = record['seeds'][str(seed)]['metrics']
                assert len(metrics[reduced][reading]) == 101
                losses.append(float(np.mean(to_loss * (
                    np.asarray(metrics[reduced][reading], dtype=float)
                    - np.asarray(metrics[joint][reading], dtype=float)))))
            difference = abs(float(np.mean(losses)) - entry['simultaneous']['point'][index])
            contrast_difference = max(contrast_difference, difference)
            assert difference < SLACK
            seed_points = entry['per_seed_points'][label]
            assert [seed_points[str(seed)] for seed in run.gate.SPLIT_SEEDS] == losses
        point = np.asarray(entry['simultaneous']['point'])
        critical = entry['simultaneous']['critical_value']
        interval = np.asarray(entry['simultaneous']['interval'])
        assert np.allclose(interval[:, 0], point - critical, rtol=0, atol=0)
        assert np.allclose(interval[:, 1], point + critical, rtol=0, atol=0)
        assert len(entry['pointwise_interval']) == len(entry['columns'])

    # The saved group-by-contrast matrices reproduce the declared bands exactly.
    with np.load(out / 'group_contrasts.npz', allow_pickle=False) as matrices:
        assert np.array_equal(matrices['groups'], np.asarray(report['groups']))
        for key, entry in report['panels'].items():
            matrix = matrices[key]
            assert matrix.shape == (101, bidirectional.FAMILY_SIZE)
            recomputed = bidirectional.paired_bands(matrix)
            assert np.allclose(recomputed['point'], entry['simultaneous']['point'], rtol=0, atol=0)
            assert recomputed['critical_value'] == entry['simultaneous']['critical_value']
            counts = bidirectional.counts(bidirectional.verdict(entry['simultaneous']['interval']))
            assert counts == {
                name: sum(entry['verdicts'][direction]['counts'][name]
                          for direction in bidirectional.DIRECTIONS)
                for name in counts}
            for direction in bidirectional.DIRECTIONS:
                selected = [index for index, label in enumerate(entry['columns'])
                            if label.rsplit('|', 1)[1] == direction]
                assert len(selected) == 33
                assert entry['verdicts'][direction]['counts'] == bidirectional.counts(
                    bidirectional.verdict(np.asarray(entry['simultaneous']['interval'])[selected]))

    # The retained studentised family must be the frozen one on the frozen matrix:
    # only the declared statistic changed between the first stability product and this
    # one, so the frozen intervals and the new sensitivity agree to rounding.
    source_panel = panel_source / 'panel.json'
    standardized_difference = None
    if source_panel.exists():
        frozen = json.loads(source_panel.read_text())['panels']
        standardized_difference = 0.
        for key, entry in report['panels'].items():
            previous = np.asarray(frozen[key]['simultaneous']['interval'], dtype=float)
            current = np.asarray(entry['standardized_maxT_sensitivity']['interval'], dtype=float)
            standardized_difference = max(standardized_difference,
                                          float(np.max(np.abs(previous - current))))
        assert standardized_difference < 1e-12

    # Real prediction test: labels of an outer held-out family cannot alter any
    # prediction in that outer fold, including alpha selection and scaling.
    arm = 'progen2-xlarge'
    seed = run.gate.SPLIT_SEEDS[0]
    located = run.locate_extractions([ROOT / 'results/external_baseline/20261008022745_762304146fdd'], [arm])
    scalar, _ = run.verify_scalar_archive(located[arm], arm, run.STATE['cohort'], run.COHORT_SHA256)
    record = next(record for record in records if record['arm'] == arm)
    held = record['seeds'][str(seed)]['folds'][0]['held_groups']
    rows = np.flatnonzero(np.isin(panel['group'], held))
    changed = dict(panel)
    changed['target'] = panel['target'].copy()
    changed['target'][rows] = changed['target'][rows][::-1] + 7.5
    new = run.gate.fold_predictions(changed, {'M': scalar}, {'M': ('M',)}, seed=seed, device='cpu')
    with np.load(panel_source / 'oof' / (arm + '.npz')) as archive:
        original = archive[f'{seed}|M']
    isolation = float(np.max(np.abs(new['predictions']['M'][rows] - original[rows])))
    assert isolation == 0.

    counts = {key: {direction: entry['verdicts'][direction]['counts']
                    for direction in bidirectional.DIRECTIONS}
              for key, entry in report['panels'].items()}
    standardized_counts = {
        key: {direction: bidirectional.counts(bidirectional.verdict(
            np.asarray(entry['standardized_maxT_sensitivity']['interval'])[
                [index for index, label in enumerate(entry['columns'])
                 if label.rsplit('|', 1)[1] == direction]]))
            for direction in bidirectional.DIRECTIONS}
        for key, entry in report['panels'].items()}
    run.write(out / 'validation.json', {
        'status': 'passed', 'models': 33, 'new_prediction_vectors': 99, 'rows_per_vector': 25856,
        'same_row_identity_and_folds': True,
        'mean_contrast_max_absolute_difference': contrast_difference,
        'historical_forward_max_absolute_difference': max(
            record['validation']['historical_forward_max_absolute_difference'] for record in records),
        'source_standardized_band_max_absolute_difference': standardized_difference,
        'held_out_target_perturbation_max_prediction_change': isolation,
        'held_out_rows_perturbed': len(rows),
        'primary_method': bidirectional.NATIVE_METHOD, 'primary_counts': counts,
        'standardized_maxT_sensitivity_counts': standardized_counts,
        'seconds': time.monotonic() - start})
    print(json.dumps(json.loads((out / 'validation.json').read_text())), flush=True)


if __name__ == '__main__':
    main()