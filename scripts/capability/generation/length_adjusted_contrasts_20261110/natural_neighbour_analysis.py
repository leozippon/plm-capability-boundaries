import numpy as np, pandas as pd
from scipy import stats
rng=np.random.default_rng(20261113); B=2000
t=pd.read_csv('per_structure_composition.tsv',sep='\t').sort_values('sequence_sha256').reset_index(drop=True)
t['q']=['q%05d'%i for i in range(len(t))]
h=pd.read_csv(__import__('os').environ.get('HITS','natural_neighbour/work/hits.tsv'),sep='\t',header=None,names='q subject pident alnlen nident qstart qend qlen slen evalue bitscore'.split())
h['id_over_q']=100*h.nident/h.qlen
best=h.sort_values(['q','id_over_q','bitscore'],ascending=[True,False,False]).drop_duplicates('q').set_index('q')
t=t.join(best[['id_over_q','pident','qstart','qend','qlen','slen','evalue','bitscore']],on='q')
t['had_hit']=t.id_over_q.notna()
t['identity_over_query']=t.id_over_q.fillna(0.0)
t['query_coverage']=((t.qend-t.qstart+1)/t.qlen).where(t.had_hit)
t['subject_longer_than_query']=(t.slen>t.L).where(t.had_hit)
assert (t.qlen.dropna()==t.L[t.qlen.notna()]).all()
t.drop(columns=['q','id_over_q']).to_csv('natural_neighbour/per_structure_nearest_neighbour.tsv',sep='\t',index=False,float_format='%.5g')
ci=lambda x:np.nanpercentile(x,[2.5,97.5])
def grp(df,a,cap): return df[(df.arm==a)&(df.capped==cap)]
# group summaries, all arms
rows=[]
for a in sorted(t.arm.unique()):
    for cap in (True,False):
        g=grp(t,a,cap)
        rows.append(dict(arm=a,capped=cap,n=len(g),frac_any_hit=g.had_hit.mean(),mean_identity=g.identity_over_query.mean(),median_identity=g.identity_over_query.median(),
          frac_ge70=(g.identity_over_query>=70).mean(),frac_ge95=(g.identity_over_query>=95).mean(),median_qcov_of_hits=g.query_coverage.median(),
          median_evalue_of_hits=g.evalue.median(),frac_hit_subject_longer=g.subject_longer_than_query.mean()))
pd.DataFrame(rows).to_csv('natural_neighbour/group_summaries.tsv',sep='\t',index=False,float_format='%.4g')
# between groups, all pairs of interest
def diff(x,y,stat=np.mean):
    bs=[stat(rng.choice(x,len(x)))-stat(rng.choice(y,len(y))) for _ in range(B)]
    return stat(x)-stat(y),*ci(bs),stats.mannwhitneyu(x,y).pvalue
rows=[]
for pair,b,tu in [('112m','protgpt3-112m','protgpt3-112m-dpo'),('1.3b','protgpt3-1.3b','protgpt3-1.3b-dpo'),('10b','protgpt3-10b','protgpt3-10b-dpo'),('mix7','progen2-small-hugohrban','progen2-small-mix7')]:
    for lab,x,y in (('tuned capped minus tuned uncapped',grp(t,tu,True),grp(t,tu,False)),('tuned capped minus base capped',grp(t,tu,True),grp(t,b,True)),('base capped minus base uncapped',grp(t,b,True),grp(t,b,False))):
        for col in ('identity_over_query',):
            d,lo,hi,p=diff(x[col].values,y[col].values); rows.append(dict(pair=pair,contrast=lab,metric='mean identity over query (%)',n_x=len(x),n_y=len(y),diff=d,lo=lo,hi=hi,mannwhitney_p=p))
        d,lo,hi,p=diff(x.had_hit.values.astype(float),y.had_hit.values.astype(float)); rows.append(dict(pair=pair,contrast=lab,metric='fraction with any hit',n_x=len(x),n_y=len(y),diff=d,lo=lo,hi=hi,mannwhitney_p=p))
pd.DataFrame(rows).to_csv('natural_neighbour/between_group.tsv',sep='\t',index=False,float_format='%.4g')
# within arm association
rows=[]
for a in sorted(t.arm.unique()):
    for cap in (True,False):
        g=grp(t,a,cap)
        for ep in ('plddt','ptm'):
            rho,p=stats.spearmanr(g.identity_over_query,g[ep]); bs=[]
            v=g[['identity_over_query',ep]].values
            for _ in range(B): s=v[rng.integers(0,len(v),len(v))]; bs.append(stats.spearmanr(s[:,0],s[:,1])[0])
            rows.append(dict(arm=a,capped=cap,endpoint=ep,n=len(g),spearman=rho,lo=ci(bs)[0],hi=ci(bs)[1]))
pd.DataFrame(rows).to_csv('natural_neighbour/within_arm_association.tsv',sep='\t',index=False,float_format='%.4f')
# arm coefficient among capped rows: composition only vs composition + identity
M=['top_frac','entropy_bits','max_run','zlib_ratio','distinct_4mer_frac']
def design(df,tu,cov,ident):
    X=[np.ones(len(df)),(df.arm==tu).values.astype(float)]
    cols=[]
    if cov: cols+=M
    if ident: cols+=['identity_over_query']
    for m in cols:
        v=np.log(df[m].values) if m=='max_run' else df[m].values.astype(float); X.append((v-v.mean())/v.std())
    return np.column_stack(X)
rows=[]; c=t[t.capped]
for ep in ('plddt','ptm'):
  for pair,b,tu in [('112m','protgpt3-112m','protgpt3-112m-dpo'),('1.3b','protgpt3-1.3b','protgpt3-1.3b-dpo'),('10b','protgpt3-10b','protgpt3-10b-dpo'),('mix7','progen2-small-hugohrban','progen2-small-mix7')]:
    df=c[c.arm.isin([b,tu])].reset_index(drop=True); y=df[ep].values; r=dict(endpoint=ep,pair=pair,n_base=(df.arm==b).sum(),n_tuned=(df.arm==tu).sum())
    specs={'arm_only':(False,False),'plus_composition':(True,False),'plus_identity':(False,True),'plus_composition_and_identity':(True,True)}
    for nm,(cv,idn) in specs.items(): r[nm]=np.linalg.lstsq(design(df,tu,cv,idn),y,rcond=None)[0][1]
    bs={nm:[] for nm in specs}
    for _ in range(B):
        d2=df.iloc[rng.integers(0,len(df),len(df))]
        for nm,(cv,idn) in specs.items(): bs[nm].append(np.linalg.lstsq(design(d2,tu,cv,idn),d2[ep].values,rcond=None)[0][1])
    for nm in specs: r[nm+'_lo'],r[nm+'_hi']=ci(bs[nm])
    rows.append(r)
pd.DataFrame(rows).to_csv('natural_neighbour/capped_arm_effect_with_identity.tsv',sep='\t',index=False,float_format='%.4f')
print('ok')
