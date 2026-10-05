#!/usr/bin/env python3
"""Replay the admitted primary paired readouts from retained rank features on CPU.

No model inference, sequence generation, new split definition or bootstrap.
Outputs are local reproductions, never replacements for frozen H200 evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ANALYSIS_ID = 'mutation_primary_local_reproduction_20261005'
BASE = 'C_P_wall'
AUGMENTED = BASE + '+M'
SEEDS = (20260923, 20260924, 20260925)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def point_metrics(rows, keys):
    """Original point estimator: mean within family, then equal-family mean."""
    import numpy as np
    from src.capability.context.profiles import cluster_means
    result = {}
    for key in keys:
        selected = [r for r in rows if r[key] is not None]
        result[key] = (float(np.mean(cluster_means([r[key] for r in selected],
                       [r['cluster'] for r in selected])[0])) if selected else None)
    return result


def compare_points(points, reference):
    import numpy as np
    return {key: {'local': value, 'h200': reference[key]['point'],
            'delta': None if value is None or reference[key]['point'] is None else value-reference[key]['point'],
            'consistent': value is None and reference[key]['point'] is None if value is None or reference[key]['point'] is None
            else bool(np.isclose(value, reference[key]['point'], atol=1e-9, rtol=1e-8))}
            for key, value in points.items()}


def paired_points(assays, y, indices, baseline, augmented):
    """Original per-assay rank metrics and paired changes, without new intervals."""
    import numpy as np
    from scipy.stats import rankdata
    from src.capability.context.profile_increment import correlation, standardized_rank
    scores = []
    for assay in assays:
        ix = indices[assay['assay']]
        target = standardized_rank(y[ix])
        row = {'cluster': assay['cluster']}
        for name, pred in ((BASE, baseline), (AUGMENTED, augmented)):
            row[name+'_spearman'] = correlation(rankdata(pred[ix]), target)
            row[name+'_rank_mse'] = float(np.mean((target-pred[ix])**2))
        high, low = row[AUGMENTED+'_spearman'], row[BASE+'_spearman']
        row['increment_M_'+BASE] = None if high is None or low is None else high-low
        row['rank_mse_reduction_M_'+BASE] = row[BASE+'_rank_mse']-row[AUGMENTED+'_rank_mse']
        scores.append(row)
    return point_metrics(scores, [key for key in scores[0] if key != 'cluster'])


def retained_features(root, admission, assays):
    """Read only raw_M features; no fitted R2 vector enters the replay design."""
    import numpy as np
    from scripts.capability.reporting import export_prediction_details as d
    from src.capability.context.profile_increment import standardized_rank
    from src.capability.context.local_context import fold_membership
    bundle = root/'results/R2/readout_expansion_20260923/complete'
    registry_path = bundle/'admission/source_registry.json'
    registry = json.loads(registry_path.read_text())['files']
    features, cells, provenance, memberships = {}, set(), [], {}
    for path in sorted((root/d.READOUT).glob('**/readout_*common*.json')):
        report = json.loads(path.read_text())
        if not d.is_supporting_readout(report):
            continue
        arm, seed = report['arm'], report['fold_seed']
        d.require((arm, seed) not in cells, 'duplicate model/split')
        d.require(d.digest(path) == registry[str(path.relative_to(bundle))]['sha256'], 'registry report hash mismatch')
        records = report['predictions']
        d.require([r['assay'] for r in records] == [r['assay'] for r in assays], 'assay order mismatch')
        for saved, assay in zip(records, assays):
            d.require(saved['mutants'] == assay['mutants'], 'mutation row order mismatch')
            d.finite_vector(saved['raw_M'], len(assay['mutants']), 'raw_M')
            d.require(np.array_equal(standardized_rank(saved['raw_M']), saved['raw_M']), 'raw_M rank semantics mismatch')
        vector = np.concatenate([r['raw_M'] for r in records]).astype(np.float64)
        if arm in features:
            d.require(np.array_equal(vector, features[arm]), 'rank feature changed across seeds')
        else:
            features[arm] = vector
        d.require(admission['cohort_sha256'] in report['source_sha256'].values(), 'cohort lineage mismatch')
        manifests = {p: h for p, h in report['source_sha256'].items() if Path(p).name == f'manifest_{arm}.json'}
        d.require(len(manifests) == 1, 'ambiguous source manifest')
        for original, expected in manifests.items():
            relative = 'results/external_baseline/' + original.split('/results/external_baseline/', 1)[1]
            d.require(d.digest(bundle/relative) == expected == registry[relative]['sha256'], 'extraction manifest lineage mismatch')
        # Family membership is shared across original designs; retain and check it, not their coefficients.
        groups = {r['cluster'] for r in assays}
        for records in report['folds'].values():
            d.outer_mapping(records, groups, held='held_families', training='training_families')
        realised = [fold_membership(records) for records in report['folds'].values()]
        d.require(bool(realised), 'missing source fold records')
        d.require(all(m == realised[0] for m in realised), 'source design fold disagreement')
        if seed in memberships:
            d.require(memberships[seed] == realised[0], 'source arm fold disagreement')
        memberships[seed] = realised[0]
        provenance.append({'path': str(path.relative_to(root)), 'sha256': d.digest(path),
                           'arm': arm, 'seed': seed, 'feature': 'predictions[].raw_M',
                           'extraction_manifest_sha256': manifests})
        cells.add((arm, seed))
    d.require({f'{a}/{s}' for a, s in cells} == set(admission['cells']), 'incomplete original panel')
    return features, memberships, provenance, d.digest(registry_path)


def run(root, out):
    sys.path.insert(0, str(root))
    import numpy as np
    import scipy
    import torch
    from scripts.capability.context import analyse_crossed_controls as crossed
    from scripts.capability.context import analyse_local_context_gate as gate
    from scripts.capability.reporting import export_prediction_details as d
    from src.capability.context.local_context import assemble_designs, fold_membership
    from src.capability.readouts.readout_analysis import nested_predict, sequence_features

    d.require(not out.exists() or not any(out.iterdir()), 'output must be new or empty')
    out.mkdir(parents=True, exist_ok=True)
    threads = int(os.environ.get('OMP_NUM_THREADS', '4'))
    torch.set_num_threads(threads)
    d.require(shutil.disk_usage(out).free > 2_000_000_000, 'insufficient disk headroom')
    runtime = {'host': 'local L20 server', 'device': 'cpu', 'threads': threads,
               'python': platform.python_version(), 'numpy': np.__version__,
               'scipy': scipy.__version__, 'torch': str(torch.__version__),
               'disk_free_bytes': shutil.disk_usage(out).free,
               'memory_preflight': Path('/proc/meminfo').read_text()}
    write_json(out/'runtime.json', runtime)
    admission = json.loads((root/d.LOCAL).read_text())
    d.require(d.digest(root/d.MUTATION) == admission['cohort_sha256'], 'cohort hash mismatch')
    cohort = json.loads((root/d.MUTATION).read_text())
    cohort_map = {r['assay']: r for r in cohort['assays']}
    assays = [cohort_map[a] for a in sorted(admission['support']['assay_ids'])]
    features, original_folds, sources, registry_sha = retained_features(root, admission, assays)
    store, profile_hashes = crossed.load_profile_store(root/'results/R5/retrieval_bound_20260807', cohort)
    rows = [dict(assay=r['assay'], cluster=r['cluster'], mutants=r['mutants'], measured=r['measured'],
                 P=r['profile_scores'], S=sequence_features(r['wildtype'], r['mutants'])) for r in assays]
    deviation = gate.build_blocks(rows, cohort, store, ['wall'])
    blocks = {name: np.concatenate([r[key] for r in rows]) for name, key in [('S','S'),('P','P_block'),('wall','wall')]}
    baseline_x = np.column_stack([blocks[k] for k in ('S','P','wall')])
    y = np.concatenate([r['measured'] for r in rows])
    aids = np.concatenate([[r['assay']]*len(r['mutants']) for r in rows])
    groups = np.concatenate([[r['cluster']]*len(r['mutants']) for r in rows])
    d.require(baseline_x.shape == (25728,569), 'primary design shape mismatch')
    indices = {r['assay']: np.flatnonzero(aids == r['assay']) for r in rows}
    keys = [f'{n}_{metric}' for n in (BASE,AUGMENTED) for metric in ('spearman','rank_mse')]
    keys += ['increment_M_'+BASE, 'rank_mse_reduction_M_'+BASE]
    code_files = [Path(__file__), root/'scripts/capability/reporting/export_prediction_details.py',
                  root/'scripts/capability/context/analyse_local_context_gate.py',
                  root/'scripts/capability/context/analyse_crossed_controls.py',
                  root/'src/capability/context/local_context.py', root/'src/capability/context/crossed_controls.py',
                  root/'src/capability/context/profiles.py', root/'src/capability/context/profile_increment.py',
                  root/'src/capability/readouts/readout_analysis.py',
                  root/'scripts/capability/stages/retrieval_bound.py',
                  root/'src/capability/models/fitness.py', root/'src/capability/core/amino_acids.py',
                  root/'src/capability/core/protein_properties.py']
    code_hashes = {str(p.relative_to(root)): d.digest(p) for p in code_files}
    contract = {'analysis_id': ANALYSIS_ID,
                'cohort_sha256': admission['cohort_sha256'], 'reference_path': d.LOCAL,
                'reference_sha256': d.digest(root/d.LOCAL), 'code_sha256': code_hashes,
                'source_registry_sha256': registry_sha, 'rank_feature_sources': sources,
                'profile_sources': profile_hashes, 'max_profile_deviation': deviation,
                'dimensions': {BASE:569,AUGMENTED:570}, 'seeds': list(SEEDS), 'models': sorted(features),
                'outer_folds':5, 'inner_folds':4, 'alphas':[.01,.1,1.,10.,100.],
                'limitation': 'Feature-equivalent replay; absent raw archives cannot be independently recomputed. Current code hashes are recorded; frozen-code byte equivalence is not asserted. Agreement of summaries does not prove identical absent H200 per-sample predictions. No new intervals.'}
    write_json(out/'contract.json', contract)
    comparisons, failures, completed = {}, [], []
    write_json(out/'comparisons.json', comparisons)
    write_json(out/'progress.json', {'expected_cells':99,'completed':completed,'failed':failures})
    sample_rows = []
    for assay in assays:
        for j, mutation in enumerate(assay['mutants']):
            sample_rows.append({'sample_id':d.sample_id(admission['cohort_sha256'],assay['assay'],mutation),
                 'assay':assay['assay'],'variant_index':j,'mutation':mutation,'group_id':assay['cluster']})
    columns = ['analysis_id','cohort_sha256','sample_id','assay','variant_index','mutation','group_id','model_id','seed',
               'outer_fold','is_held_out','baseline_design','model_design','baseline_prediction','model_prediction','status']
    for seed in SEEDS:
        print(json.dumps({'phase':'baseline','seed':seed}),flush=True)
        try:
            baseline, bf = nested_predict(baseline_x,y,aids,groups,seed=seed,device='cpu')
            mapping = d.outer_mapping(bf,set(groups),held='held_families',training='training_families')
            d.require(fold_membership(bf) == original_folds[seed], 'replayed inner/outer membership differs')
            np.savez_compressed(out/f'baseline-{seed}.npz', prediction=baseline)
            write_json(out/f'baseline-{seed}-folds.json',bf)
        except Exception as error:
            for arm in sorted(features):
                failure = {'arm':arm,'seed':seed,'status':'failed','phase':'baseline',
                           'error':f'{type(error).__name__}: {error}'}
                failures.append(failure)
            print(json.dumps(failures[-1]),flush=True)
            write_json(out/'progress.json',{'expected_cells':99,'completed':completed,'failed':failures})
            continue
        for arm in sorted(features):
            started = time.monotonic();cell=f'{arm}/{seed}'
            try:
                x = assemble_designs({**blocks,'M':features[arm][:,None]}, {BASE:('S','P','wall')}, {'+M':('M',)})[AUGMENTED]
                augmented, folds = nested_predict(x,y,aids,groups,seed=seed,device='cpu')
                d.require(fold_membership(folds)==fold_membership(bf), 'inner/outer membership mismatch')
                comparison=compare_points(paired_points(assays,y,indices,baseline,augmented),
                                          admission['summaries'][cell])
                comparison['baseline_digest']={'local':hashlib.sha256(np.asarray(baseline,dtype=np.float64).tobytes()).hexdigest(),
                    'h200':admission['model_independent_prediction_digests'][f'{seed}/{BASE}']}
                comparison['baseline_digest']['exact_match'] = (
                    comparison['baseline_digest']['local'] == comparison['baseline_digest']['h200'])
                comparisons[cell]=comparison
                status='consistent_with_retained_summaries' if all(comparison[k]['consistent'] for k in keys) else 'summary_discrepancy'
                filename=f'{arm}-seed{seed}.csv.gz'
                with d.gzip_text(out/filename) as stream:
                    table=d.Table(stream,columns)
                    for i,sample in enumerate(sample_rows):
                        table.row(**sample,analysis_id=ANALYSIS_ID,cohort_sha256=admission['cohort_sha256'],
                                  model_id=arm,seed=seed,outer_fold=mapping[sample['group_id']],is_held_out=True,
                                  baseline_design=BASE,model_design=AUGMENTED,baseline_prediction=float(baseline[i]),
                                  model_prediction=float(augmented[i]),status=status)
                write_json(out/f'{arm}-seed{seed}-folds.json',folds)
                completed.append({'arm':arm,'seed':seed,'file':filename,'sha256':d.digest(out/filename),'rows':len(y),'status':status})
                print(json.dumps({'cell':cell,'status':status,'seconds':time.monotonic()-started}),flush=True)
            except Exception as error:
                failures.append({'arm':arm,'seed':seed,'status':'failed','error':f'{type(error).__name__}: {error}'})
                print(json.dumps(failures[-1]),flush=True)
            write_json(out/'comparisons.json',comparisons)
            write_json(out/'progress.json',{'expected_cells':99,'completed':completed,'failed':failures})
    d.require(all(d.digest(root/p)==h for p,h in code_hashes.items()),'numerical source changed during replay')
    write_json(out/'completion.json',{'expected_cells':99,'completed':completed,'failed':failures,
               'all_predictions_available':len(completed)==99,'comparisons_sha256':d.digest(out/'comparisons.json')})
    if failures:
        raise RuntimeError(f'{len(failures)} cells failed; details preserved in completion.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.root.resolve(),args.out.resolve())


if __name__=='__main__':
    main()
