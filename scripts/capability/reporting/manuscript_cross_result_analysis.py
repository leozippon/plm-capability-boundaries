"""Exploratory manuscript sensitivity; no model inference or checkpoint-population tests."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

def main():
 p=argparse.ArgumentParser(); p.add_argument('--input-root',required=True);p.add_argument('--out',required=True);p.add_argument('--device');a=p.parse_args();root=Path(a.input_root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 manifest={}
 def read(path):
  b=(root/path).read_bytes();manifest[path]=hashlib.sha256(b).hexdigest();return json.loads(b)
 sources={}
 for folder,arms in [('results/R1/progen3_stage_20260917/retrieval_bound',['progen3-112m','progen3-3b']),('results/R1/adaptation_lineage_20260813/retrieval_bound',['llama-2-7b','prollama-stage-1','prollama']),('results/R1/progen2_fitness_20260813/retrieval_bound',['progen2-medium','progen2-large','progen2-xlarge']),('results/R1/proteingym_gap_20260807/analyse_verify',['progen2-small','progen2-base']),('results/R5/retrieval_bound_20260807',['protgpt2'])]:
  for arm in arms:sources[arm]=read(folder+'/model_'+arm+'.json')
 lookup={r['assay']:r for r in read('results/R1/progen3_stage_20260917/retrieval_bound/lookup.json')['assays']}
 models={arm:{r['assay']:r for r in d['assays'] if r.get('spearman') is not None} for arm,d in sources.items()}
 common=sorted(set.intersection(set(lookup),*[set(x) for x in models.values()]))
 for arm,d in models.items():
  for assay in common:assert d[assay]['mutant_digest']==lookup[assay]['mutant_digest'],(arm,assay)
 census=read('results/R6/unconditional_diversity_20260919/unconditional_diversity.json')['checkpoints'];census.update(read('results/R6/unconditional_diversity_3b_20260919/unconditional_diversity.json')['checkpoints'])
 arms=list(models);families=sorted({str(lookup[k]['cluster']) for k in common});rng=np.random.default_rng(20260921);boot=rng.integers(len(families),size=(4000,len(families)))
 vals=np.array([[np.mean([models[arm][k]['spearman'] for k in common if str(lookup[k]['cluster'])==f]) for arm in arms] for f in families]);base=np.array([[np.mean([lookup[k]['spearman'][b] for k in common if str(lookup[k]['cluster'])==f]) for b in ['lookup','blosum62']] for f in families]);means=vals.mean(0);draws=vals[boot].mean(1)
 def lineage(arm):return 'progen2' if arm.startswith('progen2') else 'progen3' if arm.startswith('progen3') else 'llama2-prollama' if arm in ['llama-2-7b','prollama-stage-1','prollama'] else 'galactica' if arm.startswith('galactica') else arm
 rows=[]
 for i,arm in enumerate(arms):
  r=dict(arm=arm,lineage=lineage(arm),n_assays=len(common),n_families=len(families),raw_spearman=means[i],raw_ci_low=np.quantile(draws[:,i],.025),raw_ci_high=np.quantile(draws[:,i],.975),model_minus_profile=means[i]-base[:,0].mean(),model_minus_blosum62=means[i]-base[:,1].mean())
  if arm in census:
   c=census[arm];n=c['n_attempts'];r.update(n_attempts=n,n_nonempty=c['n_nonempty'],n_profile_hits=c['n_any_profile_hits'],nonempty_rate=c['n_nonempty']/n,recognition_rate=c['n_any_profile_hits']/n,conditional_recognition=c['n_any_profile_hits']/c['n_nonempty'],profile_group_yield=(c['n_any_profile_groups'] or 0)/n,profile_group_retention=c['n_any_profile_groups']/c['n_any_profile_hits'] if c['n_any_profile_hits'] else None)
  rows.append(r)
 associations=[]
 def corr(x,y,data,label):
  rr=[r for r in data if r.get(x) is not None and r.get(y) is not None]
  if len(rr)<3:return
  val=float(spearmanr([r[x] for r in rr],[r[y] for r in rr]).statistic)
  associations.append(dict(x=x,y=y,sensitivity=label,n_checkpoints=len(rr),spearman=val,arms=';'.join(r['arm'] for r in rr),ci_low=None,ci_high=None))
 for x,y in [('raw_spearman','recognition_rate'),('raw_spearman','profile_group_yield'),('model_minus_profile','recognition_rate'),('model_minus_blosum62','recognition_rate')]:
  corr(x,y,rows,'common_support')
  for fam in sorted({r['lineage'] for r in rows}):corr(x,y,[r for r in rows if r['lineage']!=fam],'leave_out:'+fam)
  for fam in sorted({r['lineage'] for r in rows}):corr(x,y,[r for r in rows if r['lineage']==fam],'within:'+fam)
  selected=[(i,r) for i,r in enumerate(rows) if y in r];bs=np.array([spearmanr(draws[b,[i for i,r in selected]],[r[y] for i,r in selected]).statistic for b in range(len(boot))]); associations.append(dict(x=x,y=y,sensitivity='DMS_family_only_uncertainty_fixed_generation',n_checkpoints=len(selected),spearman=float(spearmanr([r[x] for i,r in selected],[r[y] for i,r in selected]).statistic),ci_low=float(np.quantile(bs,.025)),ci_high=float(np.quantile(bs,.975)),arms=';'.join(r['arm'] for i,r in selected)))
 # Original full-support lineage differences are retained separately.
 contrasts=[]
 for arm1,arm2 in [('llama-2-7b','prollama-stage-1'),('prollama-stage-1','prollama'),('progen3-112m','progen3-3b'),('progen2-small','progen2-base'),('progen2-base','progen2-medium'),('progen2-medium','progen2-large'),('progen2-large','progen2-xlarge')]:
  i,j=arms.index(arm1),arms.index(arm2);z=draws[:,j]-draws[:,i];contrasts.append(dict(earlier=arm1,later=arm2,estimate=means[j]-means[i],ci_low=np.quantile(z,.025),ci_high=np.quantile(z,.975),n_assays=len(common),n_families=len(families),metric='paired_DMS_raw_delta_common_support'))
 # Homologue correlation uses a common biological endpoint rather than nats/token.
 h=read('results/R5/context_homologue_expansion_20260919/context_homologue.json');hr=[]
 for arm,v in h['arms'].items():
  if arm not in census:continue
  r=dict(arm=arm,lineage=lineage(arm),recognition_rate=census[arm]['n_any_profile_hits']/census[arm]['n_attempts'])
  for s in ['pooled','decisive_stratum']:
   z=v[s]['auroc'];r[s]=z['mean']
  hr.append(r)
 for x in ['pooled','decisive_stratum']:
  corr(x,'recognition_rate',hr,'homologue_fixed_panel')
  for fam in sorted({r['lineage'] for r in hr}):corr(x,'recognition_rate',[r for r in hr if r['lineage']!=fam],'homologue_leave_out:'+fam)
 def write(name,data):
  keys=list(dict.fromkeys(k for r in data for k in r));f=open(out/name,'w');w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data);f.close()
 write('common-support-checkpoints.csv',rows);write('association-sensitivities.csv',associations);write('paired-lineage-contrasts.csv',contrasts);write('homologue-recognition.csv',hr)
 result=dict(status='complete',scope='exploratory fixed-checkpoint panel',n_assays=len(common),n_families=len(families),bootstrap_resamples=4000,seed=20260921,source_sha256=manifest,common_assays=common,limitations=['No independent generation-seed uncertainty','No cross-tokenizer NLL association','DMS bootstrap conditional on fixed checkpoints and fixed generation outcomes','Current structural replay gap: structural endpoints excluded','No per-output length sensitivity in retained input package'])
 (out/'manuscript_cross_result_analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['common_assays','source_sha256']}))
if __name__=='__main__':main()
