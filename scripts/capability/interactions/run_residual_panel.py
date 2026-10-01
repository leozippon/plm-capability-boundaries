#!/usr/bin/env python3
"""Freeze, fit and summarize the complete archived-score residual panel on H200.

The frozen inputs are one pinned set of bytes. What varies between products is the
declared support: ``all`` admits every cycle of the frozen cohort, and
``indel-excluded`` additionally applies a state-level insertion/deletion-construct
exclusion before any fit sees the panel. A support declares its cohort digest and
its group, cycle and site-pair counts here, and the run is refused unless the
bytes on disk realise them, so a support can be added without loosening the check
that a product is bound to the support it names.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from queue import Queue
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.capability.interactions import fit_pairwise_epistasis as fit
from src.capability.interactions import pairwise_epistasis as model

DRAWS = 10000
#: Digest of the one frozen cohort every declared support is built on.
COHORT_SHA256 = '8133463ec30293013864685a426a99b69c51c1bac6774b37ebd036915670040f'

#: Declared supports and the counts their bytes must realise. ``all`` is the
#: published support. ``indel-excluded`` drops the three cycles whose states rest
#: only on insertion-construct rows and carries the corrected median for the two
#: states that keep a value; it draws no replacement cycle, so three backgrounds
#: retain 127 of the cohort's 128 cycles and the three site pairs those cycles
#: carried leave the support.
SUPPORTS = {
    'all': {'counts': {'groups': 64, 'cycles': 8192, 'site_pairs': 217},
            'requires_exclusion': False},
    'indel-excluded': {'counts': {'groups': 64, 'cycles': 8189, 'site_pairs': 214},
                       'requires_exclusion': True},
}
INPUTS = {
    'plan': 'data/pairwise_epistasis/extraction_plan_20260924_e338420f.json',
    'cohort': 'data/pairwise_epistasis/cohort_20260924_8133463e.json',
    'baseline': 'data/pairwise_epistasis/baseline_q.json',
    'profiles': 'data/pairwise_epistasis/profile_features.npz',
    'published': 'results/pairwise_epistasis_20260924/panel.json',
}


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def simultaneous_bands(values, *, draws=DRAWS, seed=model.BOOTSTRAP_SEED):
    """Paired group bootstrap of centered, fixed-SE statistics across a family.

    Rows are exchangeable held groups and columns are prespecified contrasts.
    These approximate simultaneous intervals condition on fitted predictions;
    split seeds are averaged within group, never treated as independent units.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError('expected a finite group-by-contrast matrix')
    points = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / np.sqrt(len(values))
    varying = se > 0
    rng = np.random.default_rng(seed)
    maxima = np.empty(draws)
    for start in range(0, draws, 100):
        stop = min(start + 100, draws)
        indices = rng.integers(0, len(values), size=(stop-start, len(values)))
        deviations = values[indices].mean(axis=1) - points
        maxima[start:stop] = (np.max(np.abs(deviations[:, varying] / se[varying]), axis=1)
                               if varying.any() else 0.0)
    critical = float(np.quantile(maxima, .95))
    return {'method': 'paired group bootstrap maximum absolute centered statistic / fixed group SE',
            'draws': draws, 'seed': seed, 'groups': len(values), 'family_size': values.shape[1],
            'critical_value': critical, 'confidence': .95, 'point': points.tolist(),
            'interval': np.column_stack([points-critical*se, points+critical*se]).tolist(),
            'conditional_on_fitted_predictions': True}


def support_counts(cohort, exclusion):
    """Groups, cycles and site pairs the cohort bytes realise under a support.

    A cycle leaves the support when any of its four states rests only on
    insertion-construct rows, which is the state the declaration reports without a
    remaining median. Counting here, from the cohort, is independent of the panel
    assembly that applies the same exclusion at fit time; ``summarize`` then holds
    each arm's realised counts against the numbers declared from these bytes.
    """
    if exclusion and 'states' not in exclusion:
        raise ValueError('the indel-exclusion declaration must be the state-level shape '
                         "(key 'states'); found keys " + str(sorted(exclusion)))
    absent = {(state['background'], state['sequence']) for state in exclusion['states']
              if state['value_without_indel_rows_kcal_mol'] is None} if exclusion else set()
    groups, cycles, site_pairs = set(), 0, 0
    for background in cohort['backgrounds']:
        kept = [cycle for cycle in background['cycles']
                if not any((background['name'], state) in absent for state in cycle['sequences'])]
        if not kept:
            continue
        groups.add(background['group'])
        cycles += len(kept)
        site_pairs += len({tuple(cycle['positions']) for cycle in kept})
    return {'groups': len(groups), 'cycles': cycles, 'site_pairs': site_pairs}


def declare(root, out, support, exclusion_path):
    paths = {key: root / relative for key, relative in INPUTS.items()}
    if exclusion_path is not None:
        paths['exclusion'] = exclusion_path
    plan = json.loads(paths['plan'].read_text())
    hashes = {key: fit.digest(path) for key, path in paths.items()}
    if hashes['cohort'] != COHORT_SHA256:
        raise ValueError(f'cohort does not match its declared digest: {hashes["cohort"]}')
    if hashes['cohort'] != plan['cohort_sha256'] or hashes['baseline'] != plan['provenance']['baseline_q']['sha256']:
        raise ValueError('cohort or baseline mismatch')
    _, profile_meta = fit.load_profiles(paths['profiles'], plan)
    if profile_meta['plan_sha256'] != hashes['plan']:
        raise ValueError('profile plan digest mismatch')
    expected_backgrounds = {row['name'] for row in plan['backgrounds']}
    cohort = json.loads(paths['cohort'].read_text())
    exclusion = None
    if exclusion_path is not None:
        exclusion = json.loads(exclusion_path.read_text())
        if exclusion.get('schema') != 'pairwise_indel_exclusion_v1':
            raise ValueError('unexpected exclusion declaration schema')
        if exclusion['cohort_sha256'] != hashes['cohort']:
            raise ValueError('the exclusion was declared over a different cohort')
    counts = support_counts(cohort, exclusion)
    if counts != SUPPORTS[support]['counts']:
        raise ValueError(f'support {support!r} does not realise its declared counts: {counts}')
    arms = list(model.ROSTER)
    extraction = root / 'results/pairwise_epistasis_20260924/extraction'
    receipts = {}
    for arm in arms:
        manifest_path = extraction / arm / f'manifest_{arm}.json'
        manifest = json.loads(manifest_path.read_text())
        if (manifest['status'] != 'complete' or manifest['identity']['arm'] != arm
                or manifest['identity']['plan_sha256'] != model.plan_digest(plan)):
            raise ValueError(f'{arm}: invalid extraction identity')
        if (len(manifest['backgrounds']) != len(expected_backgrounds)
                or {row['background'] for row in manifest['backgrounds']} != expected_backgrounds):
            raise ValueError(f'{arm}: incomplete or duplicate background coverage')
        for row in manifest['backgrounds']:
            if fit.digest(extraction / arm / row['file']) != row['sha256']:
                raise ValueError(f'{arm}: score archive digest mismatch: {row["file"]}')
        receipts[arm] = {'manifest_sha256': fit.digest(manifest_path),
                         'archives_checked': len(manifest['backgrounds'])}
    declaration = {
        'schema': 'pairwise_residual_full_panel_declaration_v1',
        'declared_utc': datetime.now(timezone.utc).isoformat(), 'arms': arms,
        'target': 'epsilon minus training-only isotonic response to measured additive',
        'contrast': 'C_G_T+M|C_G_T', 'support': support, 'support_counts': counts,
        'split_seeds': list(model.SPLIT_SEEDS), 'outer_folds': model.OUTER_SPLITS,
        'inner_folds': model.INNER_SPLITS, 'alphas': list(model.ALPHAS),
        'weighting': 'equal groups; equal site pairs within group; equal cycles within site pair',
        'correction_and_G': 'refitted inside every inner and outer training partition',
        'primary': ('seed-mean per-group paired squared-error reduction; simultaneous bands '
                    f'across {len(arms)} arms'),
        'split_diagnostic': ('simultaneous bands across '
                             f'{len(arms) * len(model.SPLIT_SEEDS)} arm-seed cells'),
        'units': 'kcal^2/mol^2', 'bootstrap_draws': DRAWS, 'bootstrap_seed': model.BOOTSTRAP_SEED,
        'interpretation': 'label-assisted conditional sensitivity, not equivalence or biological purification',
        'input_sha256': hashes, 'extractions': receipts,
        'fitter_sha256': fit.digest(ROOT / 'scripts/capability/interactions/refit_likelihood_on_residual.py'),
        'orchestrator_sha256': fit.digest(Path(__file__)),
        'module_sha256': fit.digest(Path(model.__file__)),
    }
    if len(arms) != 33:
        raise ValueError('the declared 33-arm roster changed')
    write(out / 'declaration.json', declaration)
    return declaration


def summarize(out, declaration):
    errors, groups, reference_folds = [], None, None
    receipts = {}
    for arm in declaration['arms']:
        path = out / f'residual_{arm}.json'
        record = json.loads(path.read_text())
        if record['arm'] != arm or record['script_sha256'] != declaration['fitter_sha256'] or record['module_sha256'] != declaration['module_sha256']:
            raise ValueError(f'{arm}: fitter identity mismatch')
        if record['declaration_sha256'] != fit.digest(out / 'declaration.json'):
            raise ValueError(f'{arm}: declaration mismatch')
        if record['support'] != declaration['support']:
            raise ValueError(f'{arm}: support differs from the declaration')
        if {key: record[key] for key in ('groups', 'cycles', 'site_pairs')} != declaration['support_counts']:
            raise ValueError(f'{arm}: support counts differ')
        if groups is None:
            groups = record['groups_order']
            reference_folds = {s: [f['held_groups'] for f in record['seeds'][str(s)]['folds']]
                               for s in declaration['split_seeds']}
        if record['groups_order'] != groups:
            raise ValueError(f'{arm}: group alignment mismatch')
        per_seed = []
        for seed in declaration['split_seeds']:
            row = record['seeds'][str(seed)]
            if [f['held_groups'] for f in row['folds']] != reference_folds[seed]:
                raise ValueError(f'{arm}: fold alignment mismatch')
            error = row['per_group_mse']
            delta = np.asarray(error['C_G_T']) - np.asarray(error['C_G_T+M'])
            if not np.isclose(delta.mean(), row['mse_reduction_kcal2']['point'], rtol=0, atol=1e-15):
                raise ValueError(f'{arm}: per-group contrast mismatch')
            per_seed.append(delta)
        errors.append(per_seed)
        receipts[arm] = fit.digest(path)
    values = np.asarray(errors)  # arm, seed, group
    primary = simultaneous_bands(values.mean(axis=1).T)
    diagnostic = simultaneous_bands(values.reshape(-1, len(groups)).T)
    records = {}
    for index, arm in enumerate(declaration['arms']):
        lo, hi = primary['interval'][index]
        records[arm] = {'seed_mean_mse_reduction_kcal2': primary['point'][index],
                       'simultaneous95_interval_kcal2': [lo, hi],
                       'resolved_positive': lo > 0, 'resolved_negative': hi < 0,
                       'per_seed_simultaneous95': {str(seed): diagnostic['interval'][index*3+j]
                                                  for j, seed in enumerate(declaration['split_seeds'])}}
    report = {'schema': 'pairwise_residual_full_panel_v1', 'declaration_sha256': fit.digest(out/'declaration.json'),
              'arms': records, 'receipts_sha256': receipts, 'groups_order': groups,
              'primary_bootstrap': primary, 'split_diagnostic_bootstrap': diagnostic,
              'resolved_positive_arms': sum(r['resolved_positive'] for r in records.values()),
              'resolved_negative_arms': sum(r['resolved_negative'] for r in records.values())}
    write(out / 'panel.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6')
    parser.add_argument('--support', default='all', choices=sorted(SUPPORTS))
    parser.add_argument('--exclusion', type=Path,
                        help='state-level indel-exclusion declaration, required by a '
                             'support that declares one')
    args = parser.parse_args()
    if SUPPORTS[args.support]['requires_exclusion'] != (args.exclusion is not None):
        raise SystemExit(f'support {args.support!r} declares '
                         f'{"an" if SUPPORTS[args.support]["requires_exclusion"] else "no"} exclusion')
    if args.out.exists():
        raise SystemExit('refuse existing output directory; preserve prior declarations and results')
    args.out.mkdir(parents=True)
    declaration = declare(args.root, args.out, args.support, args.exclusion)
    print(f'DECLARATION_VERIFIED_{len(declaration["arms"])}_ARMS', flush=True)
    devices = [device for device in args.gpus.split(',') if device]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError('this campaign needs at least one distinct GPU device')
    pending = Queue()
    for arm in declaration['arms']:
        pending.put(arm)
    def worker(device):
        from queue import Empty
        while True:
            try:
                arm = pending.get_nowait()
            except Empty:
                return
            command = [sys.executable, str(ROOT/'scripts/capability/interactions/refit_likelihood_on_residual.py'),
                       '--root', str(args.root), '--arm', arm, '--device', f'cuda:{device}',
                       '--out', str(args.out/f'residual_{arm}.json'),
                       '--declaration', str(args.out/'declaration.json')]
            if args.exclusion is not None:
                command += ['--exclusion', str(args.exclusion)]
            with (args.out/f'{arm}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        list(pool.map(worker, devices))
    report = summarize(args.out, declaration)
    print(json.dumps({'status':'complete', 'support':args.support, 'arms':len(report['arms']),
                      'positive':report['resolved_positive_arms'], 'negative':report['resolved_negative_arms']}), flush=True)


if __name__ == '__main__':
    main()
