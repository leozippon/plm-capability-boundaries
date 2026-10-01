#!/usr/bin/env python3
"""Retain native ProGen3 terms with WT and each single mutant in one forward.

Paired scoring equalizes the MoE dispatch shape experienced by the two identical
prefixes. Each variant retains its own WT reference; these references must never
be collapsed to one shared WT scalar. This is a fresh numerical measurement,
not a bit-identical replay of separately forwarded historical scores.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.capability.core.io import write_json
from src.capability.position.position_terms import (ResidueCoverage, alignment, blas_pinning,
    target_nll_terms, retention_residual, state_parts, mutation_parts, partition_masks)
from src.capability.readouts.readout_extraction import load_readout_arm, pack_sequence, forward_readout_rows


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<22),b''):h.update(part)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--support',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    args=p.parse_args()
    if not 0<=args.shard<args.shards:raise ValueError('invalid shard')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
    blas=blas_pinning()
    cohort=json.loads(args.cohort.read_bytes());support=json.loads(args.support.read_bytes())
    if sha(args.cohort)!=support['cohort_sha256']:raise ValueError('cohort digest mismatch')
    by_assay={r['assay']:r for r in cohort['assays']}
    assays=[a for a in sorted(support['assays']) if any(':' not in m for m in by_assay[a]['mutants'])]
    selected=[a for i,a in enumerate(assays) if i%args.shards==args.shard]
    spec=importlib.util.spec_from_file_location('stage46',ROOT/'scripts/capability/stages/context_homologue.py')
    stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
    arm=load_readout_arm('progen3-3b',stage,device='cuda:0',dtype='bfloat16')
    coverage=ResidueCoverage(arm)
    files=['scripts/capability/position/extract_paired_position_terms.py','scripts/capability/stages/context_homologue.py',
           'src/capability/readouts/readout_extraction.py','src/capability/position/position_terms.py','src/capability/models/progen3.py']
    identity=dict(arm=arm.name,dtype='bfloat16',batch_size=2,reference='separate WT reference per variant in same forward',
                  cohort_sha256=sha(args.cohort),support_sha256=sha(args.support),blas=blas,
                  code_sha256={name:sha(ROOT/name) for name in files},
                  checkpoint_metadata_sha256={p.name:sha(p) for p in sorted(Path(arm.spec.path).glob('*.json'))})
    args.out.mkdir(parents=True,exist_ok=True)
    receipts=[]
    for assay in selected:
        row=by_assay[assay]
        indices=[i for i,m in enumerate(row['mutants']) if ':' not in m]
        metadata=dict(identity=identity,assay=assay,mutant_digest=row['mutant_digest'],variant_indices=indices)
        filename='paired_'+hashlib.sha256(assay.encode()).hexdigest()[:20]+'.npz'
        path=args.out/filename
        if path.exists():
            with np.load(path,allow_pickle=False) as saved:
                if json.loads(str(saved['metadata']))!=metadata:raise ValueError('resume identity mismatch')
        else:
            wild_ids,wild_span,_=pack_sequence(arm,row['wildtype'])
            wc,wo=coverage.counts(wild_ids,wild_span,row['wildtype'])
            wild_state=dict(ids=wild_ids,span=wild_span,counts=wc,offset=wo)
            if len(wild_ids)>1024:raise ValueError('paired row exceeds frozen token budget')
            records=[];wild_terms=[];mutant_terms=[];mutant_ids=[];nlls=[]
            for i in indices:
                mutation=row['mutants'][i];site=int(mutation[1:-1])-1
                sequence=row['sequences'][i]
                ids,span,_=pack_sequence(arm,sequence)
                counts,offset=coverage.counts(ids,span,sequence)
                mutant_state=dict(ids=ids,span=span,counts=counts,offset=offset)
                if not alignment(wild_state,mutant_state,site)['aligned']:
                    raise ValueError('ProGen3 paired variant is not strictly token aligned')
                logits,packed=forward_readout_rows(arm,[wild_ids,ids],stage)
                terms=[target_nll_terms(logits[j:j+1],packed[j:j+1],*s).detach().cpu().numpy()
                       for j,s in enumerate((wild_span,span))]
                nll=[stage._target_nll(logits[j:j+1],packed[j:j+1],*s)['nll_sum']
                     for j,s in enumerate((wild_span,span))]
                if any(retention_residual(t,n,arm.device)!=0.0 for t,n in zip(terms,nll)):
                    raise ValueError('paired same-forward retention failed')
                parts=mutation_parts(state_parts(terms[0],wc,wo,site),
                                     state_parts(terms[1],counts,offset,site),float(nll[0]-nll[1]))
                prefix=partition_masks(wc,wo,site)['upstream']
                if not np.array_equal(terms[0][prefix],terms[1][prefix]):
                    raise ValueError('paired identical-prefix likelihood terms differ')
                if abs(parts['closure_nats'])>parts['closure_bound_nats']:
                    raise ValueError('paired component closure exceeds arithmetic bound')
                records.append(parts);wild_terms.append(terms[0]);mutant_terms.append(terms[1]);mutant_ids.append(ids);nlls.append(nll)
            arrays={k:np.asarray([r[k] for r in records]) for k in records[0]}
            with path.with_suffix('.tmp').open('wb') as handle:
                np.savez_compressed(handle,**arrays,metadata=json.dumps(metadata),variant_indices=np.asarray(indices),
                                    wild_position_nats=np.stack(wild_terms),mutant_position_nats=np.stack(mutant_terms),
                                    wild_ids=np.asarray(wild_ids),mutant_ids=np.asarray(mutant_ids),
                                    residue_counts=wc,residue_offset=np.asarray(wo),scored_span=np.asarray(wild_span),
                                    nll=np.asarray(nlls),mutants=np.asarray([row['mutants'][i] for i in indices]))
            path.with_suffix('.tmp').replace(path)
        receipts.append(dict(assay=assay,file=filename,sha256=sha(path),variants=len(indices)))
        print(json.dumps(dict(assay=assay,variants=len(indices),status='retained')),flush=True)
    write_json(args.out/f'manifest_paired_shard{args.shard}.json',dict(status='complete',identity=identity,
               shard=args.shard,shards=args.shards,assays=receipts,retention_max_abs_nats=0.0,prefix_term_max_abs_nats=0.0,torch=torch.__version__,gpu=torch.cuda.get_device_name(0)))

if __name__=='__main__':main()
