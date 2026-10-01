#!/usr/bin/env python3
"""Check identical-prefix likelihood invariance across batches and precision.

A same-shape WT/mutant pair contrasts positions strictly before the substituted
residue. Singleton and paired identical-sequence controls distinguish packing,
causal dependence and batch/precision effects without endpoint-based selection.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.capability.readouts.readout_extraction import load_readout_arm, extract_batch
from src.capability.position.position_terms import ResidueCoverage, partition_masks, state_parts, mutation_parts
from src.capability.core.io import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--support',type=Path,required=True)
    p.add_argument('--arm',default='progen3-3b')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dtype',nargs='+',choices=('bfloat16','float16'),default=['bfloat16'])
    args=p.parse_args()
    raw=args.cohort.read_bytes();cohort=json.loads(raw)
    spec=importlib.util.spec_from_file_location('stage46',ROOT/'scripts/capability/stages/context_homologue.py')
    stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
    # Frozen source ordering, first three admissible assay examples; no labels read.
    cases=[]
    support=json.loads(args.support.read_bytes())
    if support['cohort_sha256'] != hashlib.sha256(raw).hexdigest():
        raise ValueError('diagnostic support cohort digest mismatch')
    for row in cohort['assays']:
        if row['assay'] not in support['assays']:continue
        for mutant,sequence in zip(row['mutants'],row['sequences']):
            if ':' not in mutant and 20<int(mutant[1:-1])<len(row['wildtype'])-10:
                cases.append((row['assay'],row['wildtype'],sequence,int(mutant[1:-1])-1));break
        if len(cases)==3:break
    records=[]
    args.out.mkdir(parents=True,exist_ok=True)
    for dtype in args.dtype:
        print(json.dumps(dict(arm=args.arm,dtype=dtype,status='loading')),flush=True)
        arm=load_readout_arm(args.arm,stage,device='cuda:0',dtype=dtype)
        coverage=ResidueCoverage(arm)
        for assay,wild,mutant,site in cases:
            outputs={}
            for label,seqs in [('wild_single',[wild]),('mutant_single',[mutant]),
                               ('duplicate',[wild,wild]),('paired',[wild,mutant]),
                               ('paired_reverse',[mutant,wild]),('repeated_pair',[wild,mutant,wild,mutant])]:
                bucket=[]
                _,scores=extract_batch(arm,seqs,stage,position_terms=bucket)
                outputs[label]=(bucket,scores)
            wt=outputs['wild_single'][0][0];mt=outputs['mutant_single'][0][0]
            counts,offset=coverage.counts(wt['ids'],wt['span'],wild)
            mask=partition_masks(counts,offset,site)['upstream']
            if wt['ids'][:site+1] != mt['ids'][:site+1]:
                raise ValueError('expected identical packed prefix')
            def difference(a,b):
                delta=a['terms'].astype(float)-b['terms'].astype(float)
                return dict(upstream_max_abs=float(np.abs(delta[mask]).max()),
                            upstream_signed_sum=float(delta[mask].sum()),
                            all_max_abs=float(np.abs(delta).max()))
            pair=outputs['paired'][0];dupe=outputs['duplicate'][0];rev=outputs['paired_reverse'][0]
            def components(label,wi=0,mi=1):
                bucket,scores=outputs[label]
                return mutation_parts(state_parts(bucket[wi]['terms'],counts,offset,site),
                                      state_parts(bucket[mi]['terms'],counts,offset,site),
                                      float(scores[mi]-scores[wi]))
            contrasts={label:components(label,wi,mi) for label,wi,mi in [('paired',0,1),('paired_reverse',1,0),('repeated_pair',0,1),('duplicate',0,1)]}
            records.append(dict(dtype=dtype,assay=assay,site=site,length=len(wild),
                                singleton_mutation=difference(wt,mt),
                                paired_mutation=difference(pair[0],pair[1]),
                                identical_duplicate=difference(dupe[0],dupe[1]),
                                wild_single_vs_batch=difference(wt,pair[0]),
                                mutant_single_vs_batch=difference(mt,pair[1]),
                                wild_batch_order=difference(pair[0],rev[1]),
                                mutant_batch_order=difference(pair[1],rev[0]),
                                components=contrasts))
        write_json(args.out/f'position_prefix_diagnostic_{dtype}.json',dict(arm=args.arm,dtype=dtype,cohort_sha256=hashlib.sha256(raw).hexdigest(),records=[r for r in records if r['dtype']==dtype],torch=torch.__version__,gpu=torch.cuda.get_device_name(0)))
        print(json.dumps(dict(arm=args.arm,dtype=dtype,status='complete')),flush=True)
        del arm
        import gc
        gc.collect();torch.cuda.empty_cache()
    args.out.mkdir(parents=True,exist_ok=True)
    write_json(args.out/'position_prefix_diagnostic.json',dict(arm=args.arm,cohort_sha256=hashlib.sha256(raw).hexdigest(),records=records,torch=torch.__version__,gpu=torch.cuda.get_device_name(0)))

if __name__=='__main__':main()
