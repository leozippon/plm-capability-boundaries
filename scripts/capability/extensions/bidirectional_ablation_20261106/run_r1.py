#!/usr/bin/env python3
"""Reaggregate existing R1 held-out predictions; never refit or infer a model."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

import numpy as np
from scipy.stats import rankdata
import scipy

DESIGNS = {'C': 'BPL', 'M': 'M', 'CM': 'BMPL'}
SEEDS = [20260923, 20260924, 20260925]
METRICS = ['spearman', 'rank_mse']
BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20261106


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def save(path, obj):
    p = Path(path)
    with p.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def correlation(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt(x @ x * (y @ y))
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        raise ValueError('Undefined assay correlation; no post-hoc exclusions allowed')
    return float(x @ y / d)


def run(root, source, out):
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}
    def read(p):
        inputs[str(p.relative_to(root))] = sha(p)
        return json.loads(p.read_text())
    old = read(source/'contract.json')
    complete = read(source/'completion.json')
    assert complete['status'] == 'complete'
    models = old['models']
    assert len(models) == 33 and len(set(models)) == 33 and old['seeds'] == SEEDS
    samples = read(source/'full/samples.json')
    assert len(samples) == 25728
    aids = np.asarray([s['assay'] for s in samples])
    fams = np.asarray([s['cluster'] for s in samples])
    sample_ids = np.asarray([s['sample_id'] for s in samples])
    assert len(set(sample_ids)) == len(samples)
    assays = sorted(set(aids)); groups = sorted(set(fams))
    assert len(assays) == 201 and len(groups) == 163
    assay_ix = [np.flatnonzero(aids == a) for a in assays]
    assay_groups = []
    for ix in assay_ix:
        assert len(set(fams[ix])) == 1
        assay_groups.append(fams[ix[0]])
    group_ix = [np.flatnonzero(np.asarray(assay_groups) == g) for g in groups]
    contract = dict(analysis='bidirectional_ablation_20261106_r1', source=str(source.relative_to(root)),
        source_contract_sha256=sha(source/'contract.json'), source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        code_sha256=sha(__file__), models=models, seeds=SEEDS,
        support=dict(rows=len(samples), assays=len(assays), groups=len(groups)),
        designs=DESIGNS, C='444 sequence/substitution/position/composition/length + 14 profile + 111 local context features',
        M='One within-assay-ranked scalar mutant-minus-WT likelihood, fitted using historical nested ridge',
        metrics=METRICS, hypothesis_families={m:66 for m in METRICS},
        contrasts={'M_given_C':'CM - C for Spearman; C - CM for rank MSE', 'C_given_M':'CM - M for Spearman; M - CM for rank MSE'},
        aggregation='Average seeds within assay; equal assays within family; equal families',
        bootstrap=dict(draws=BOOTSTRAP, seed=BOOTSTRAP_SEED, unit='family', paired=True,
            method='95th percentile of joint maximum absolute centered bootstrap deviation over 33 models x 2 directions, separately for each endpoint',
            absolute_arm_CI='pointwise percentile 95%', directional_pointwise_CI='pointwise percentile 95%'),
        uncertainty='Conditional on existing OOF fits; no model inference or tuning refits',
        policy='Frozen before new aggregation. No exclusion based on outcome. C-minus-M descriptive only. No information/knowledge partition inferred.',
        resource=dict(device='cpu', workers=1, blas_threads=1, python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
            available_memory_kib=next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable')),
            cpu_count=os.cpu_count()))
    save(out/'contract.json', contract)
    old_metrics = read(source/'full/assay-metrics.json')
    index = {(r['arm'], r['seed'], r['assay']):r for r in old_metrics}
    assert len(index) == len(models)*len(SEEDS)*len(assays) == len(old_metrics)
    values = np.empty((len(METRICS),len(models),len(SEEDS),len(assays),len(DESIGNS)))
    target_ref = None; ref_memberships = {}; max_difference = 0.; cells = []
    for im, model in enumerate(models):
        for iz, seed in enumerate(SEEDS):
            stem=source/'full'/f'{model}-seed{seed}'
            p=Path(str(stem)+'.npz'); checks=read(Path(str(stem)+'-checks.json'))
            inputs[str(p.relative_to(root))]=sha(p)
            assert sha(p) == checks['verification']['prediction_archive_sha256']
            with np.load(p,allow_pickle=False) as z:
                assert np.array_equal(z['sample_id'], sample_ids)
                assert np.array_equal(z['original_index'], np.arange(len(samples)))
                target=z['target']
                assert np.isfinite(target).all()
                if target_ref is None: target_ref=target.copy()
                assert np.array_equal(target,target_ref)
                memberships=[]
                for idesign,(name,original) in enumerate(DESIGNS.items()):
                    pred=z[original]
                    check=checks['designs'][original]
                    assert array_sha(pred)==check['prediction_sha256']
                    assert array_sha(target)==check['target_sha256']
                    assert check['support_sha256']==old['panel_support']['full']['support_sha256']
                    assert len(pred)==len(samples) and np.isfinite(pred).all()
                    memberships.append(check['membership_sha256'])
                    folds=read(Path(str(stem)+f'-{original}-folds.json'))
                    realized=[(f['fold'],tuple(f['held_families']),tuple(f['training_families']),
                        tuple(tuple(i['validation_families']) for i in f['inner_folds'])) for f in folds]
                    membership_sha=hashlib.sha256(json.dumps(realized,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                    assert membership_sha==check['membership_sha256']
                    assert len(folds)==5
                    held=[]
                    for f in folds:
                        h=set(f['held_families']); t=set(f['training_families'])
                        assert not h&t and h|t==set(map(int,groups))
                        inner=[v for i in f['inner_folds'] for v in i['validation_families']]
                        assert len(f['inner_folds'])==4 and len(set(inner))==len(inner) and set(inner)==t
                        held.extend(f['held_families'])
                    assert len(held)==len(groups) and len(set(held))==len(groups)
                    for ia, ix in enumerate(assay_ix):
                        v=[correlation(rankdata(pred[ix]),target[ix]), float(np.mean((pred[ix]-target[ix])**2))]
                        for k, metric in enumerate(METRICS):
                            previous=index[model,seed,assays[ia]][metric][original]
                            assert previous is not None and np.isfinite(previous)
                            delta=abs(v[k]-previous);max_difference=max(max_difference,delta)
                            assert delta < 1e-12
                            values[k,im,iz,ia,idesign]=v[k]
                assert len(set(memberships))==1
                assert ref_memberships.setdefault(seed,memberships[0])==memberships[0]
            cells.append(dict(model=model,seed=seed,seconds=time.monotonic()-start))
        print(json.dumps(dict(model=model,completed_models=im+1,elapsed_seconds=time.monotonic()-start)),flush=True)
    # Metric x model x seed x group x design, then seeds averaged before any resampling.
    by_group_seed=np.stack([values[:,:,:,ix,:].mean(axis=3) for ix in group_ix],axis=3)
    by_group=by_group_seed.mean(axis=2)
    assert np.isfinite(by_group).all()
    # C is the same fitted control for every model, not a newly retuned baseline.
    assert np.max(np.abs(by_group[:, :, :, 0] - by_group[:, :1, :, 0])) < 1e-12
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    counts=np.asarray([np.bincount(rng.integers(len(groups),size=len(groups)),minlength=len(groups)) for _ in range(BOOTSTRAP)],float)
    result={}; flat=[]
    for k,metric in enumerate(METRICS):
        matrix=by_group[k].transpose(1,0,2).reshape(len(groups),-1)
        point=matrix.mean(axis=0).reshape(len(models),3)
        draws=(counts @ matrix / len(groups)).reshape(BOOTSTRAP,len(models),3)
        absolute_ci=np.quantile(draws,[.025,.975],axis=0)
        sign=1 if metric=='spearman' else -1
        dpoint=sign*np.stack([point[:,2]-point[:,0],point[:,2]-point[:,1]],axis=1)
        ddraw=sign*np.stack([draws[:,:,2]-draws[:,:,0],draws[:,:,2]-draws[:,:,1]],axis=2)
        dci=np.quantile(ddraw,[.025,.975],axis=0)
        critical=float(np.quantile(np.max(np.abs(ddraw-dpoint[None,:,:]),axis=(1,2)),.95))
        arms=[]
        for i,model in enumerate(models):
            absolute={name:dict(point=float(point[i,j]),ci95=absolute_ci[:,i,j].tolist(),
                seed_points=by_group_seed[k,i,:,:,j].mean(axis=1).tolist()) for j,name in enumerate(DESIGNS)}
            contrasts={}
            for j,c in enumerate(['M_given_C','C_given_M']):
                sim=[float(dpoint[i,j]-critical),float(dpoint[i,j]+critical)]
                contrasts[c]=dict(point=float(dpoint[i,j]),pointwise_ci95=dci[:,i,j].tolist(),simultaneous_ci95=sim,
                    seed_points=(sign*(by_group_seed[k,i,:,:,2]-by_group_seed[k,i,:,:,j])).mean(axis=1).tolist(),
                    verdict='positive' if sim[0]>0 else 'negative' if sim[1]<0 else 'unresolved')
                flat.append(dict(metric=metric,model=model,contrast=c,**contrasts[c]))
            arms.append(dict(model=model,absolute=absolute,contrasts=contrasts,
                C_minus_M_descriptive=float(point[i,0]-point[i,1])))
        counts_verdict={c:{v:sum(a['contrasts'][c]['verdict']==v for a in arms) for v in ['positive','negative','unresolved']} for c in ['M_given_C','C_given_M']}
        result[metric]=dict(hypotheses=66,simultaneous_critical=critical,models=arms,verdict_counts=counts_verdict)
    # Paired/bootstrap algebra must preserve both direction identities exactly.
    assert np.allclose(dpoint[:,0]-dpoint[:,1],sign*(point[:,1]-point[:,0]),rtol=0,atol=1e-14)
    np.savez_compressed(out/'group-metrics.npz',values=by_group_seed,models=np.asarray(models),seeds=np.asarray(SEEDS),groups=np.asarray(groups),metrics=np.asarray(METRICS),designs=np.asarray(list(DESIGNS)))
    save(out/'results.json',dict(analysis=contract['analysis'],status='complete',metrics=result,
        bootstrap_counts_sha256=array_sha(counts),bootstrap=BOOTSTRAP,bootstrap_seed=BOOTSTRAP_SEED))
    save(out/'input-manifest.json',inputs)
    save(out/'validation.json',dict(status='passed',source_cells=len(cells),prediction_vectors=len(cells)*3,
        metric_values=int(values.size),max_recomputed_metric_difference=max_difference,
        sample_identity_verified=True,prediction_hashes_verified=True,source_fold_membership_hashes_matched=True,
        constant_C_across_models_verified=True,complete_seed_assay_support_verified=True,paired_contrast_identity_verified=True))
    save(out/'completion.json',dict(status='complete',seconds=time.monotonic()-start,
        cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        models=len(models),seeds=len(SEEDS),bootstrap=BOOTSTRAP,verdict_counts={m:result[m]['verdict_counts'] for m in METRICS},
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print((out/'completion.json').read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.root,args.root/'results/extensions/information_progression_20261006',args.out)
