#!/usr/bin/env python3
"""Analyze one model on exact common support declared by extraction manifests."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.capability.readouts.readout_analysis import evaluate_readouts, sequence_features


def load_panel(cohort_path, manifests, arm, support, seed, projection_dim):
    hashes = {}
    def read(path):
        raw = path.read_bytes(); hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    cohort = read(cohort_path)
    sources = [(path, read(path)) for path in manifests]
    if len({s['identity']['arm'] for _, s in sources}) != len(sources):
        raise ValueError('duplicate model manifests')
    cohort_hash = hashes[str(cohort_path)]
    for path, source in sources:
        if source['status'] != 'complete':
            raise ValueError(f'incomplete extraction: {path}')
        if source['identity']['feature_names'] != ['middle_mean','middle_last','final_mean','final_last']:
            raise ValueError('representation block order mismatch')
        if source['identity']['cohort_sha256'] != cohort_hash:
            raise ValueError('cohort hash mismatch')
    selected = [(p,s) for p,s in sources if s['identity']['arm'] == arm]
    if len(selected) != 1:
        raise ValueError('requested arm requires exactly one manifest')
    maps = [{r['assay']:r for r in s['assays']} for _,s in sources]
    if any(len(m) != len(s['assays']) for m,(_,s) in zip(maps,sources)):
        raise ValueError('duplicate extraction assays')
    common = set.intersection(*(set(m) for m in maps))
    selected_path, source = selected[0]
    selected_map = {r['assay']:r for r in source['assays']}
    ids = common if support == 'common' else set(selected_map)
    if not ids:
        raise ValueError('empty assay support')
    cohort_map = {r['assay']:r for r in cohort['assays']}
    if len(cohort_map) != len(cohort['assays']):
        raise ValueError('duplicate cohort assays')
    rows = []; projection = None
    for assay in sorted(ids):
        c = cohort_map[assay]
        # Cohort hashes already prove common variant/label identity; enforce
        # each manifest's declared assay identity too, including other arms.
        checked = maps if support == 'common' else [selected_map]
        for mapping in checked:
            for key in ('cluster','wildtype_id','mutant_digest'):
                if mapping[assay][key] != c[key]:
                    raise ValueError(f'{assay}: manifest {key} mismatch')
            if mapping[assay]['variants'] != len(c['mutants']):
                raise ValueError('variant count mismatch')
        record = selected_map[assay]
        path = selected_path.parent / record['file']
        raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
        if digest != record['sha256']:
            raise ValueError(f'{assay}: extraction checksum mismatch')
        hashes[str(path)] = digest
        with np.load(path, allow_pickle=False) as data:
            expected_metadata = dict(identity=source['identity'], assay=assay, mutant_digest=c['mutant_digest'])
            if json.loads(str(data['metadata'])) != expected_metadata:
                raise ValueError('extraction metadata mismatch')
            if data['mutants'].tolist() != c['mutants']:
                raise ValueError('exact mutation order mismatch')
            for key in ('measured','profile_scores'):
                if not np.array_equal(data[key], np.asarray(c[key])):
                    raise ValueError(f'{assay}: {key} mismatch')
            hidden = np.asarray(data['features'], dtype=np.float32)
            if hidden.ndim != 3 or hidden.shape[:2] != (len(c['mutants']),4):
                raise ValueError('expected four aligned hidden feature blocks')
            if projection is None:
                projection = [np.random.default_rng(seed+i).normal(0,1/np.sqrt(projection_dim),size=(hidden.shape[2],projection_dim)).astype(np.float32) for i in range(4)]
            if hidden.shape[2] != projection[0].shape[0]:
                raise ValueError('hidden width changed within model')
            projected = np.concatenate([hidden[:,i] @ projection[i] for i in range(4)],axis=1)
            rows.append(dict(assay=assay,cluster=c['cluster'],mutants=c['mutants'],measured=data['measured'].copy(),
                             P=data['profile_scores'].copy(),M=data['likelihood'].copy(),R=projected,
                             S=sequence_features(c['wildtype'],c['mutants'])))
    support_record = dict(definition=support,assay_ids=sorted(ids),common_assay_ids=sorted(common),
                          native_assay_counts={s['identity']['arm']:len(s['assays']) for _,s in sources})
    return rows, hashes, support_record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',required=True,type=Path)
    p.add_argument('--manifest',required=True,action='append',type=Path)
    p.add_argument('--arm',required=True)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--support',choices=['common','native'],default='common')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--seed',type=int,default=20260923,help='fixed projection, bootstrap and permutation seed')
    p.add_argument('--fold-seed',type=int,default=20260923,help='independent outer/inner family assignment seed')
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--projection-dim',type=int,default=256)
    p.add_argument('--skip-permutation-control',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(4)
    if a.device.startswith('cuda'):
        if not torch.cuda.is_available(): raise RuntimeError('requested CUDA is unavailable')
        print(json.dumps(dict(device=a.device,gpu=torch.cuda.get_device_name(a.device),free_total_bytes=torch.cuda.mem_get_info(a.device))),flush=True)
    rows, hashes, support = load_panel(a.cohort,a.manifest,a.arm,a.support,a.seed,a.projection_dim)
    report=evaluate_readouts(rows,device=a.device,seed=a.seed,fold_seed=a.fold_seed,bootstrap=a.bootstrap,
        permutation_control=not a.skip_permutation_control,
        progress=lambda name:print(json.dumps(dict(arm=a.arm,readout=name,status='fitting')),flush=True))
    report.update(schema_version='d1_readout_v1',status='complete',arm=a.arm,
                  created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashes,support=support,
                  seed=a.seed,projection_seed=a.seed,fold_seed=a.fold_seed,analysis_code_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__), ROOT/'src/capability/readouts/readout_analysis.py', ROOT/'src/capability/context/profile_increment.py', ROOT/'src/capability/context/profiles.py']},
                  runtime=dict(torch=torch.__version__,numpy=np.__version__,device=a.device),
                  sequence_feature_order='400 directed substitution counts in AA20 row-major order; relative position mean, std; mutation count; WT composition20; mutant composition20; length/1024',
                  projection=dict(dim_per_block=a.projection_dim,block_seeds=[a.seed+i for i in range(4)],distribution='Gaussian sd=1/sqrt(dim); mutant-minus-WT blocks'))
    a.out.mkdir(parents=True,exist_ok=True)
    suffix='' if a.fold_seed == 20260923 else f'_fold{a.fold_seed}'
    destination=a.out/f'readout_{a.arm}_{a.support}{suffix}.json'
    temporary=destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temporary.replace(destination)
    print(json.dumps(dict(arm=a.arm,delta_spearman=report['summaries']['delta_spearman'],delta_rank_mse=report['summaries']['delta_rank_mse'])),flush=True)


if __name__=='__main__':main()
