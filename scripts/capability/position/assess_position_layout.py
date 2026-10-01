#!/usr/bin/env python3
"""Declared native ProGen3 layout sensitivity on label-blind assay strata."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/capability')]
from src.capability.core.io import write_json
from src.capability.position.position_terms import ResidueCoverage,partition_masks,state_parts,mutation_parts,target_nll_terms,retention_residual
from src.capability.readouts.readout_extraction import load_readout_arm,pack_sequence,forward_readout_rows
from src.capability.position.position_readout import evaluate,CONTRASTS
from src.capability.context.profile_increment import summarize,correlation
from src.capability.readouts.readout_analysis import sequence_features
from scripts.capability.context.analyse_local_context_gate import build_blocks,crossed,CODE_FILES

def source_hashes():
    names=set(CODE_FILES)|{'scripts/capability/position/assess_position_layout.py','src/capability/position/position_terms.py','src/capability/position/position_readout.py','src/capability/readouts/readout_extraction.py','src/capability/models/progen3.py'}
    return {name:sha(ROOT/name) for name in sorted(names)}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def selection(cohort,support):
    result=[]
    for row in sorted(cohort['assays'],key=lambda r:r['assay']):
        if row['assay'] not in support['assays']:continue
        singles=[i for i,m in enumerate(row['mutants']) if ':' not in m]
        if len(singles)<3:continue
        def key(i):return hashlib.sha256(('20260927/'+row['assay']+'/'+row['mutants'][i]).encode()).hexdigest()
        bins=[[] for _ in range(4)]
        for i in singles:bins[min(3,4*(int(row['mutants'][i][1:-1])-1)//len(row['wildtype']))].append(i)
        chosen=[i for group in bins for i in sorted(group,key=key)[:4]]
        chosen+=sorted(set(singles)-set(chosen),key=key)[:16-len(chosen)]
        result.append(dict(assay=row['assay'],cluster=row['cluster'],length=len(row['wildtype']),indices=sorted(chosen)))
    return result

def parts(terms,nll,counts,offset,site):
    p=mutation_parts(state_parts(terms[0],counts,offset,site),state_parts(terms[1],counts,offset,site),float(nll[0]-nll[1]))
    prefix=partition_masks(counts,offset,site)['upstream']
    if not np.array_equal(terms[0][prefix],terms[1][prefix]):raise ValueError('layout prefix differs')
    if abs(p['closure_nats'])>p['closure_bound_nats']:raise ValueError('layout arithmetic closure failed')
    return p

def extract(args,cohort,plan):
    selection_sha=sha(args.selection)
    spec=importlib.util.spec_from_file_location('stage46',ROOT/'scripts/capability/stages/context_homologue.py')
    stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
    arm=load_readout_arm('progen3-3b',stage,device='cuda:0',dtype='bfloat16');coverage=ResidueCoverage(arm)
    manifests=[json.loads(p.read_bytes()) for p in sorted(args.paired.glob('manifest_paired_shard*.json'))]
    if not manifests or len(manifests)!=manifests[0]['shards'] or {m['shard'] for m in manifests}!=set(range(len(manifests))):raise ValueError('paired campaign incomplete')
    if any(m['status']!='complete' or m['identity']['cohort_sha256']!=plan['cohort_sha256'] or m['identity']['support_sha256']!=plan['support_sha256'] for m in manifests):raise ValueError('paired source identity failed')
    reference={a['assay']:a for m in manifests for a in m['assays']}
    by_assay={r['assay']:r for r in cohort['assays']};receipts=[]
    for k,item in enumerate(plan['assays']):
        if k%args.shards!=args.shard:continue
        row=by_assay[item['assay']];ref=args.paired/reference[row['assay']]['file']
        reference_sha=sha(ref)
        if reference_sha!=reference[row['assay']]['sha256']:raise ValueError('paired reference checksum failed')
        old=np.load(ref,allow_pickle=False);lookup={int(i):j for j,i in enumerate(old['variant_indices'])}
        wid,span,_=pack_sequence(arm,row['wildtype']);counts,offset=coverage.counts(wid,span,row['wildtype'])
        terms={name:[] for name in ('paired2','paired4','duplicate')};nlls={name:[] for name in terms};summaries=[]
        for i in item['indices']:
            mid,mspan,_=pack_sequence(arm,row['sequences'][i]);site=int(row['mutants'][i][1:-1])-1
            if mspan!=span:raise ValueError('unexpected residue span change')
            record={}
            for name,ids in [('paired2',[wid,mid]),('paired4',[wid,mid,wid,mid]),('duplicate',[wid,wid])]:
                logits,packed=forward_readout_rows(arm,ids,stage)
                t=np.stack([target_nll_terms(logits[j:j+1],packed[j:j+1],*span).detach().cpu().numpy() for j in range(len(ids))])
                n=np.asarray([stage._target_nll(logits[j:j+1],packed[j:j+1],*span)['nll_sum'] for j in range(len(ids))])
                if any(retention_residual(a,b,arm.device)!=0 for a,b in zip(t,n)):raise ValueError('layout same-forward retention failed')
                terms[name].append(t);nlls[name].append(n);record[name]=parts(t,n,counts,offset,site)
            j=lookup[i]
            record['replay_terms_max_abs_nats']=float(max(np.max(np.abs(terms['paired2'][-1][0]-old['wild_position_nats'][j])),np.max(np.abs(terms['paired2'][-1][1]-old['mutant_position_nats'][j]))))
            record['replay_scalar_max_abs_nats']=float(np.max(np.abs(nlls['paired2'][-1]-old['nll'][j])))
            record['duplicate4_state_max_abs_nats']=float(np.max(np.abs(terms['paired4'][-1][:2]-terms['paired4'][-1][2:])))
            record['duplicate2_state_max_abs_nats']=float(np.max(np.abs(terms['duplicate'][-1][0]-terms['duplicate'][-1][1])))
            summaries.append(record)
        filename='layout_'+hashlib.sha256(row['assay'].encode()).hexdigest()[:20]+'.npz';path=args.out/filename
        np.savez_compressed(path,**{f'{k}_terms':np.stack(v) for k,v in terms.items()},**{f'{k}_nll':np.stack(v) for k,v in nlls.items()},counts=counts,offset=offset,indices=item['indices'],summaries=json.dumps(summaries),assay=row['assay'],selection_sha256=selection_sha)
        receipts.append(dict(assay=row['assay'],file=filename,sha256=sha(path),reference_sha256=reference_sha,variants=len(item['indices'])))
        old.close()
        print(json.dumps(dict(assay=row['assay'],variants=len(item['indices']),status='complete')),flush=True)
    write_json(args.out/f'layout_shard{args.shard}.json',dict(status='complete',shard=args.shard,shards=args.shards,selection_sha256=selection_sha,source_sha256=source_hashes(),assays=receipts,torch=torch.__version__,gpu=torch.cuda.get_device_name(0)))

def analyse(args,cohort,plan):
    selection_sha=sha(args.selection)
    manifests=[json.loads(p.read_bytes()) for p in sorted(args.out.glob('layout_shard*.json'))]
    if not manifests or {r['shard'] for r in manifests}!=set(range(manifests[0]['shards'])):raise ValueError('missing layout shards')
    if any(m['status']!='complete' or m['selection_sha256']!=selection_sha for m in manifests):raise ValueError('layout manifest identity mismatch')
    receipts={a['assay']:a for m in manifests for a in m['assays']}
    if set(receipts)!={r['assay'] for r in plan['assays']}:raise ValueError('layout assay coverage changed')
    source={r['assay']:r for r in cohort['assays']};rows=[];diagnostics=[]
    for item in plan['assays']:
        c=source[item['assay']];path=args.out/receipts[c['assay']]['file']
        if sha(path)!=receipts[c['assay']]['sha256']:raise ValueError('layout archive hash mismatch')
        with np.load(path,allow_pickle=False) as data:
            if str(data['selection_sha256'])!=selection_sha or data['indices'].tolist()!=item['indices']:raise ValueError('layout selection mismatch')
            values=json.loads(str(data['summaries']))
            for j,i in enumerate(item['indices']):
                site=int(c['mutants'][i][1:-1])-1
                for layout in ('paired2','paired4','duplicate'):
                    restored=parts(data[layout+'_terms'][j],data[layout+'_nll'][j],data['counts'],int(data['offset']),site)
                    if restored!=values[j][layout]:raise ValueError('layout component reconstruction failed')
            audit=dict(assay=c['assay'],cluster=c['cluster'],variants=len(values))
            for key in ('replay_terms_max_abs_nats','replay_scalar_max_abs_nats','duplicate4_state_max_abs_nats','duplicate2_state_max_abs_nats'):audit[key]=max(r[key] for r in values)
            for key in ('own','downstream','full'):
                a=np.asarray([r['paired2'][key] for r in values]);b=np.asarray([r['paired4'][key] for r in values]);delta=b-a
                audit[key+'_rank_correlation']=correlation(rankdata(a),rankdata(b))
                audit[key+'_mae_nats']=float(np.abs(delta).mean());audit[key+'_max_abs_nats']=float(np.abs(delta).max())
            for j,label in enumerate(('wild','mutant')):
                delta=data['paired4_nll'][:,j]-data['paired2_nll'][:,j]
                audit[label+'_state_mae_nats']=float(np.abs(delta).mean())
            diagnostics.append(audit)
            rows.append(dict(assay=c['assay'],cluster=c['cluster'],mutants=c['mutants'],measured=np.asarray(c['measured']),P=np.asarray(c['profile_scores']),S=sequence_features(c['wildtype'],c['mutants']),indices=item['indices'],values=values))
    store,store_sha=crossed.load_profile_store(args.profile_store,cohort);build_blocks(rows,cohort,store,['wall','rf3'])
    for row in rows:
        idx=row['indices'];row['mutants']=[row['mutants'][i] for i in idx]
        for key in ('measured','P','S','P_block','wall','rf3'):row[key]=row[key][idx]
    outputs={};contrasts={}
    for seed in (20260923,20260924,20260925):
        outputs[seed]={}
        for layout in ('paired2','paired4'):
            for row in rows:
                for key in ('own','downstream','full','upstream'):row[key]=np.asarray([r[layout][key] for r in row['values']])
            report,pred=evaluate(rows,device='cpu',fold_seed=seed)
            outputs[seed][layout]=report
            np.savez_compressed(args.out/f'layout_{layout}_fold{seed}.npz',**pred)
        contrasts[seed]={}
        for control in ('wall','rf3'):
            a=outputs[seed]['paired2']['controls'][control]['assays'];b=outputs[seed]['paired4']['controls'][control]['assays'];delta=[]
            for x,y in zip(a,b):
                if x['assay']!=y['assay']:raise ValueError('layout held rows differ')
                delta.append(dict(assay=x['assay'],cluster=x['cluster'],**{key:None if x[key] is None or y[key] is None else y[key]-x[key] for key in CONTRASTS}))
            contrasts[seed][control]={key:summarize(delta,key,bootstrap=2000,seed=20260923) for key in CONTRASTS}
    stability={key:summarize(diagnostics,key,bootstrap=2000,seed=20260923) for key in diagnostics[0] if key not in ('assay','cluster','variants')}
    write_json(args.out/'layout_assessment.json',dict(stability_summaries=stability,selection=plan,selection_sha256=selection_sha,source_sha256=source_hashes(),profile_store_sha256=store_sha,diagnostics=diagnostics,readouts=outputs,paired4_minus_paired2_contrasts=contrasts,interpretation='Conditional pointwise family intervals on fixed label-blind subset; no outcome-selected numerical tolerance or kernel change'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('declare','extract','analyse'))
    for name in ('cohort','support','selection','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--paired',type=Path);p.add_argument('--profile-store',type=Path);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    args=p.parse_args();torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
    cohort=json.loads(args.cohort.read_bytes());support=json.loads(args.support.read_bytes())
    if sha(args.cohort)!=support['cohort_sha256']:raise ValueError('cohort support identity failed')
    plan=dict(schema='position_layout_subset',cohort_sha256=sha(args.cohort),support_sha256=sha(args.support),assays=selection(cohort,support),rule='up to16 per assay, four position strata then fixed-hash fill; no measured phenotype used')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.mode=='declare':write_json(args.selection,plan);return
    if json.loads(args.selection.read_bytes())!=plan:raise ValueError('declared selection changed')
    if args.mode=='extract':extract(args,cohort,plan)
    else:analyse(args,cohort,plan)
if __name__=='__main__':main()
