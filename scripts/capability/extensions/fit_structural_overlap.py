#!/usr/bin/env python3
"""CPU structural overlap, never modifies frozen evidence or performs inference."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
SEEDS = (20260923,20260924,20260925)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run(root,sites_path,out,bootstrap=2000):
    import numpy as np
    import scipy
    import torch
    from scripts.capability.reporting import export_prediction_details as d
    from scripts.capability.reporting.replay_mutation_predictions import retained_features
    from scripts.capability.context import analyse_crossed_controls as crossed
    from scripts.capability.context import analyse_local_context_gate as gate
    from src.capability.extensions import overlap as o
    from src.capability.context.local_context import fold_membership, SCALE_ORDER
    from src.capability.readouts.readout_analysis import sequence_features, ALPHAS

    if out.exists() and any(out.iterdir()):
        raise ValueError('output must be new or empty')
    out.mkdir(parents=True,exist_ok=True)
    threads = int(os.environ.get('OMP_NUM_THREADS','2'))
    if not 1 <= threads <= 4:
        raise ValueError('require 1-4 CPU threads')
    torch.set_num_threads(threads)
    if shutil.disk_usage(out).free<2_000_000_000:
        raise ValueError('insufficient disk headroom')
    save(out/'runtime.json',dict(device='cpu',threads=threads,python=platform.python_version(),
         numpy=np.__version__,scipy=scipy.__version__,torch=str(torch.__version__),
         disk_free_bytes=shutil.disk_usage(out).free,memory_preflight=Path('/proc/meminfo').read_text(),
         blas_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}))
    if not sites_path.exists():
        save(out/'completion.json',dict(status='not_fitted',reason='structural site table not yet available',
             sites_path=str(sites_path),overlap_rows=None))
        return
    with (gzip.open(sites_path,'rt') if sites_path.suffix=='.gz' else sites_path.open()) as handle:
        sites = json.load(handle)
    if not isinstance(sites,list):
        raise ValueError('sites must be a flat JSON array')
    coverage_path = sites_path.parent/'coverage.json'
    coverage_sha = None
    structural_hashes = {s['source_path']:s['source_sha256'] for s in sites if s.get('status')=='admitted'}
    for path,sha in structural_hashes.items():
        if digest(path)!=sha:
            raise ValueError('structural source hash mismatch')
    if any(s.get('status')=='admitted' and not s.get('method') for s in sites):
        # Earlier adapter tables place experimental method in the paired source receipt.
        coverage = json.loads(coverage_path.read_text()); coverage_sha = digest(coverage_path)
        sources = {s['sha256']:s for s in coverage['sources']}
        for site in sites:
            if site.get('status')=='admitted' and not site.get('method'):
                source = sources[site['source_sha256']]
                if source['path']!=site['source_path']:
                    raise ValueError('structural source lineage mismatch')
                site['method'] = source['method']
    admission = json.loads((root/d.LOCAL).read_text())
    if digest(root/d.MUTATION)!=admission['cohort_sha256']:
        raise ValueError('cohort hash mismatch')
    cohort = json.loads((root/d.MUTATION).read_text()); mapping = {a['assay']:a for a in cohort['assays']}
    assays = [mapping[a] for a in sorted(admission['support']['assay_ids'])]
    rows,exclusions = o.select_rows(assays,sites)
    save(out/'rows.json',rows)
    contract: dict[str, Any] = dict(analysis='structural_overlap_20261006',sites_sha256=digest(sites_path),
        cohort_sha256=admission['cohort_sha256'],admission_sha256=digest(root/d.LOCAL),
        paired_coverage_sha256=coverage_sha,structural_source_sha256=structural_hashes,
        exclusions=exclusions,rows=len(rows),assays=len({r['assay'] for r in rows}),
        families=len({r['cluster'] for r in rows}),seeds=list(SEEDS),outer_folds=5,inner_folds=4,
        alphas=list(ALPHAS),structural_columns=['RSA']+['RSA_x_delta_'+s for s in SCALE_ORDER],
        rank_semantics='M and P column 0 re-ranked within restricted assay; assay standardized rank targets',
        protocol='historical partitions projected, never regenerated; separate alpha tuning; training-only weighted scaling',
        gate='each original split: positive pointwise 95% lower bound for original Spearman increment; no selection',
        degree_sensitivity='not run; primary cohort never narrowed by contact degree',
        interpretation='predictive overlap, not causal mediation or shares',
        selection_limitation='Retained structure-matched assays are all Tsuboyama stability assays; inference does not generalize to the full mutation panel.',
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),
            ROOT/'src/capability/extensions/overlap.py',ROOT/'src/capability/extensions/structure.py',
            ROOT/'scripts/capability/reporting/replay_mutation_predictions.py',
            ROOT/'scripts/capability/reporting/export_prediction_details.py',
            ROOT/'scripts/capability/context/analyse_crossed_controls.py',
            ROOT/'scripts/capability/context/analyse_local_context_gate.py',
            ROOT/'src/capability/readouts/readout_analysis.py',
            ROOT/'src/capability/context/local_context.py',ROOT/'src/capability/context/crossed_controls.py',
            ROOT/'src/capability/context/profile_increment.py')})
    save(out/'contract.json',contract)
    if not rows:
        save(out/'completion.json',dict(status='not_fitted',reason='zero strict-single exact experimental RSA overlap',
             overlap_rows=0,rows_sha256=digest(out/'rows.json')))
        return
    features,signatures,sources,registry = retained_features(root,admission,assays)
    if len(features)!=33 or set(signatures)!=set(SEEDS):
        raise ValueError('expected complete 33-model / three-seed panel')
    aids = np.asarray([r['assay'] for r in rows]); families = np.asarray([r['cluster'] for r in rows])
    try:
        projected = {seed:o.project_folds(signatures[seed],families) for seed in SEEDS}
    except o.UnsupportedProjection as error:
        save(out/'original-folds.json',signatures)
        save(out/'completion.json',dict(status='not_fitted',reason=str(error),overlap_rows=len(rows)))
        return
    save(out/'projected-memberships.json',projected)
    store,profile_hashes = crossed.load_profile_store(root/'results/R5/retrieval_bound_20260807',cohort)
    baseline_rows = [dict(assay=a['assay'],cluster=a['cluster'],mutants=a['mutants'],measured=a['measured'],
                          P=a['profile_scores'],S=sequence_features(a['wildtype'],a['mutants'])) for a in assays]
    deviation = gate.build_blocks(baseline_rows,cohort,store,['wall'])
    index = np.asarray([r['original_index'] for r in rows])
    blocks = {k:np.concatenate([a[key] for a in baseline_rows])[index].copy()
              for k,key in (('S','S'),('P','P_block'),('wall','wall'))}
    blocks['P'][:,0] = o.rerank(blocks['P'][:,0],aids)
    b = np.column_stack([blocks[k] for k in ('S','P','wall')]); x = o.structural_features(rows)
    if b.shape[1]!=569:
        raise ValueError('baseline must have 569 columns')
    y = np.concatenate([a['measured'] for a in assays])[index]
    contract.update(models=sorted(features),rank_sources=sources,registry_sha256=registry,
                    profile_sources=profile_hashes,max_profile_deviation=deviation,
                    dimensions={'B':569,'B+M':570,'B+X':576,'B+M+X':577})
    save(out/'contract.json',contract)
    scores = []; completed = []; started = time.monotonic()
    for seed in SEEDS:
        common = {}
        for name,design in (('B',b),('B+X',np.column_stack([b,x]))):
            common[name],folds = o.projected_predict(design,y,aids,families,signatures[seed])
            save(out/f'{name}-seed{seed}-folds.json',folds)
        for arm in sorted(features):
            m = o.rerank(features[arm][index],aids)[:,None]; predictions = dict(common)
            for name,design in (('B+M',np.column_stack([b,m])),('B+M+X',np.column_stack([b,m,x]))):
                predictions[name],folds = o.projected_predict(design,y,aids,families,signatures[seed])
                if fold_membership(folds)!=[(f,tuple(h),tuple(t),tuple(tuple(v) for v in inn)) for f,h,t,inn in projected[seed]]:
                    raise ValueError('design realized membership disagreement')
                save(out/f'{arm}-{name}-seed{seed}-folds.json',folds)
            np.savez_compressed(out/f'{arm}-seed{seed}-oof.npz',**predictions,target=o.rerank(y,aids),original_index=index)
            scores.extend(o.assay_metrics(predictions,y,aids,families,arm,seed))
            completed.append(f'{arm}/{seed}')
            save(out/'progress.json',dict(completed=completed,expected_cells=99,elapsed_seconds=time.monotonic()-started))
            print(json.dumps(dict(cell=completed[-1],seconds=time.monotonic()-started)),flush=True)
    save(out/'assay-metrics.json',scores)
    summaries = o.joint_bootstrap(scores,bootstrap=bootstrap)
    save(out/'contrasts.json',summaries)
    split_summaries = {str(s):o.joint_bootstrap(scores,bootstrap=bootstrap,split=s,seed=s+8000) for s in SEEDS}
    gates = {}
    for arm in sorted(features):
        checks = []
        for s in SEEDS:
            summary = split_summaries[str(s)]
            match = [r for r in summary.get('contrasts',[]) if r['arm']==arm and r['metric']=='spearman' and r['contrast']=='model_without_X']
            checks.append(bool(match) and match[0]['pointwise'][0]>0)
        gates[arm] = dict(supported_each_original_split=all(checks),split_checks=dict(zip(map(str,SEEDS),checks)),
                          affirmative_attenuation_allowed=all(checks),selection_performed=False)
    save(out/'original-increment-gate.json',dict(gates=gates,split_summaries=split_summaries))
    hashes = {p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()}
    save(out/'completion.json',dict(status='complete',completed_cells=len(completed),overlap_rows=len(rows),
         output_sha256=hashes,conditional_on_fits=True,elapsed_seconds=time.monotonic()-started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--sites',type=Path,default=ROOT/'results/extensions/mutation_structure_20261006/structure/annotation/sites.json.gz')
    parser.add_argument('--out',type=Path,default=ROOT/'results/extensions/mutation_structure_20261006/overlap')
    parser.add_argument('--bootstrap',type=int,default=2000)
    args = parser.parse_args();run(args.root.resolve(),args.sites.resolve(),args.out.resolve(),args.bootstrap)


if __name__=='__main__':
    main()
