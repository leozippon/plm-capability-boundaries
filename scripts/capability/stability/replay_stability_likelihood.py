#!/usr/bin/env python3
"""Recover group errors by replaying only two unchanged historical R4 designs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.capability.stability import fit_stability_singles as fit
from scripts.capability.interactions.run_residual_panel import simultaneous_bands
from src.capability.stability import stability_gate as gate
from src.capability.core.io import sha256_file


def write(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')


def replay(task):
    root, out, arm = task
    torch.set_num_threads(4)
    started = time.monotonic()
    directory = root/'results/gate_stability_20260924'
    historical_path = directory/'fits'/f'fit_{arm}.json'
    original = json.loads(historical_path.read_text())
    cohort_path = directory/'cohort.json'
    cohort = json.loads(cohort_path.read_text())
    cohort['_sha256'] = sha256_file(cohort_path)
    controls_path = directory/'controls_qualification.json'
    profiles_path = directory/'profile_features.npz'
    if (cohort['_sha256'] != original['cohort_sha256']
            or sha256_file(controls_path) != original['controls_sha256']
            or sha256_file(profiles_path) != original['profile_sha256']):
        raise ValueError(f'{arm}: historical input hash mismatch')
    controls = json.loads(controls_path.read_text())
    qualified = tuple(controls['qualified_control_set'])
    if list(qualified) != original['qualified_control_set']:
        raise ValueError('control set changed')
    profiles, _ = gate.load_profiles(profiles_path, {b['name']:b['wildtype'] for b in cohort['backgrounds']})
    panel = gate.build_panel(cohort, profiles)
    wave = '20260923232409_9ab5f150f2ac' if arm in ('galactica-30b','qwen2.5-32b') else '20260924000148_fbfa7ddc1f5f'
    extraction = root/'results/shared/external_baseline'/wave/arm
    manifest_path = extraction/f'manifest_{arm}.json'
    retained_manifest = json.loads(manifest_path.read_text())
    for archive in retained_manifest['backgrounds']:
        if sha256_file(extraction/archive['file']) != archive['sha256']:
            raise ValueError(f'{arm}: retained score archive digest mismatch')
    blocks, manifest = fit.load_arm(extraction, arm, cohort)
    # The original primary baseline choice is retained, not requalified post hoc.
    baseline = original['matched_baseline']['primary']
    if baseline not in ('S','S_T'):
        raise ValueError('unexpected historical baseline')
    columns = qualified+(('T',) if baseline=='S_T' else ())
    designs = {baseline:columns, baseline+'_M':columns+('M',)}
    records = {}
    for seed in gate.SPLIT_SEEDS:
        result = gate.fold_predictions(panel, dict(panel['blocks'], **blocks), designs, seed=seed, device='cpu')
        previous = original['primary'][str(seed)]
        for index, fold in enumerate(result['folds']):
            for name in designs:
                if fold['alpha'][name] != previous['alpha'][index][name]:
                    raise ValueError(f'{arm}/{seed}: selected penalty changed')
                if fold['dimensions'][name] != previous['dimensions'][name]:
                    raise ValueError(f'{arm}/{seed}: design width changed')
            if previous['nuisance'] and fold['held_groups'] != previous['nuisance'][index]['held_groups']:
                raise ValueError(f'{arm}/{seed}: historical fold differs')
        groups, base = gate.group_errors(panel['target'], result['predictions'][baseline], panel['group'], panel['site'])
        _, augmented = gate.group_errors(panel['target'], result['predictions'][baseline+'_M'], panel['group'], panel['site'])
        historical = previous['primary_likelihood']
        for actual, expected in [(float((base-augmented).mean()), historical['point']),
                                 (float(base.mean()), historical['baseline_mse'])]:
            if not np.isclose(actual, expected, atol=1e-9, rtol=1e-8):
                raise ValueError(f'{arm}/{seed}: historical point or baseline MSE changed: {actual} vs {expected}')
        if not np.allclose(gate.interval(base-augmented)['interval'], historical['interval'], atol=1e-9, rtol=1e-8):
            raise ValueError(f'{arm}/{seed}: recovered group errors do not reproduce historical interval')
        records[str(seed)] = {'groups':groups.tolist(), 'baseline_mse':base.tolist(),
                              'augmented_mse':augmented.tolist(), 'folds':result['folds']}
    report = {'arm':arm,'historical_sha256':sha256_file(historical_path),
              'cohort_sha256':cohort['_sha256'],'baseline':baseline,'seeds':records,
              'extraction_manifest_sha256':sha256_file(manifest_path),
              'script_sha256':sha256_file(Path(__file__)),
              'gate_module_sha256':sha256_file(Path(gate.__file__)),
              'seconds':time.monotonic()-started}
    write(out/f'{arm}.json', report)
    print(json.dumps({'arm':arm,'seconds':report['seconds'],'validated':True}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('refuse existing replay directory')
    args.out.mkdir(parents=True)
    arms = [args.arm] if args.arm else list(fit.ROSTER)
    if any(a not in fit.ROSTER for a in arms):
        raise ValueError('unknown arm')
    write(args.out/'declaration.json', {'arms':arms,'purpose':'post-outcome matched multiplicity sensitivity; unchanged fits',
          'statistic':'seed-mean paired group MSE reduction;10000 shared draws and33-arm maximum-statistic bands',
          'target':'unchanged combined ddG','fit':'historical selected primary baseline plus likelihood only',
          'acceptance':'same cohort,input hashes,selected penalties,point estimates,baselineMSE; no original overwritten'})
    tasks = [(args.root,args.out,arm) for arm in arms]
    if args.workers==1:
        records = [replay(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            records = list(pool.map(replay,tasks))
    if len(arms)==33:
        groups = records[0]['seeds'][str(gate.SPLIT_SEEDS[0])]['groups']
        matrix=[]
        for record in records:
            values=[]
            for seed in gate.SPLIT_SEEDS:
                r=record['seeds'][str(seed)]
                if r['groups']!=groups:
                    raise ValueError('group alignment differs')
                values.append(np.asarray(r['baseline_mse'])-np.asarray(r['augmented_mse']))
            matrix.append(np.mean(values,axis=0))
        bands=simultaneous_bands(np.array(matrix).T)
        write(args.out/'panel.json',{'arms':arms,'primary':bands,
              'receipts_sha256':{a:sha256_file(args.out/f'{a}.json') for a in arms},
              'resolved_positive':sum(v[0]>0 for v in bands['interval']),
              'resolved_negative':sum(v[1]<0 for v in bands['interval'])})
    print('STABILITY_REPLAY_EXIT=0',flush=True)


if __name__=='__main__':
    main()
