import numpy as np
import pandas as pd
import json
import zlib
import hashlib
from collections import Counter
from scipy import stats
rng=np.random.default_rng(20261112); B=2000
t=pd.read_csv(__import__('os').environ.get('TABLE','input/checkpoint_structures_12of68.tsv'),sep='\t',comment='#').rename(columns={'length_residues':'L','mean_ca_plddt_0_100':'plddt','ptm_0_1':'ptm'})
seq={}
for line in open('input/fold_cohort.jsonl'):
    r=json.loads(line); seq[r['id']]=r['sequence']
assert all(hashlib.sha256(seq[k].encode()).hexdigest()==k for k in list(seq)[:200]), 'id is not sha256 of sequence'
t['seq']=t.sequence_sha256.map(seq); assert t.seq.notna().all(); assert (t.seq.str.len()==t.L).all()
def comp(s):
    c=Counter(s); n=len(s); p=np.array(list(c.values()))/n
    run=m=1
    for a,b in zip(s,s[1:]):
        run=run+1 if a==b else 1; m=max(m,run)
    k4=len({s[i:i+4] for i in range(n-3)})/max(n-3,1)
    return max(c.values())/n, -(p*np.log2(p)).sum(), m, len(zlib.compress(s.encode(),9))/n, k4
t[['top_frac','entropy_bits','max_run','zlib_ratio','distinct_4mer_frac']]=np.array([comp(s) for s in t.seq])
t=t.drop(columns='seq'); t['capped']=(t.L==400)
t.drop(columns=['fold_seed','fold_seed_rule','evaluation_signature']).to_csv('per_structure_composition.tsv',sep='\t',index=False,float_format='%.5f')
PAIRS=[('112m','protgpt3-112m','protgpt3-112m-dpo'),('1.3b','protgpt3-1.3b','protgpt3-1.3b-dpo'),('10b','protgpt3-10b','protgpt3-10b-dpo'),('mix7','progen2-small-hugohrban','progen2-small-mix7')]
def ci(x):
    return np.nanpercentile(x,[2.5,97.5])
# ---- 2. levels
rows=[]
for a,g in t.groupby('arm'):
    for cap,h in g.groupby('capped'):
        r=dict(arm=a,capped=cap,n=len(h))
        for ep in ['plddt','ptm']: r[ep+'_mean']=h[ep].mean(); r[ep+'_sd']=h[ep].std(); r[ep+'_se']=h[ep].std()/np.sqrt(len(h))
        r['meanL']=h.L.mean(); rows.append(r)
pd.DataFrame(rows).to_csv('levels_capped_vs_uncapped.tsv',sep='\t',index=False,float_format='%.4f')
# ---- 1. pooled uncapped contrast
def basis(L,kn): 
    x=L/100.; return np.column_stack([x,x**2,x**3]+[np.maximum(x-k,0)**3 for k in kn])
def reg(Lb,yb,Lt,yt,kn,flex):
    L=np.r_[Lb,Lt]; y=np.r_[yb,yt]; a=np.r_[np.zeros(len(Lb)),np.ones(len(Lt))]
    Z=basis(L,kn) if flex else (L/100.)[:,None]
    return np.linalg.lstsq(np.column_stack([np.ones(len(L)),a,Z]),y,rcond=None)[0][1]
def binm(Lb,yb,Lt,yt,w=16):
    bt,bb=(Lt//w).astype(int),(Lb//w).astype(int); num=den=0; kb=0
    for k in np.unique(bt):
        mb=bb==k
        if mb.any(): m=bt==k; num+=m.sum()*(yt[m].mean()-yb[mb].mean()); den+=m.sum(); kb+=mb.sum()
    return (num/den if den else np.nan),den,kb
out=[]
for ep in ['plddt','ptm']:
  for name,b,tu in PAIRS:
    u=t[~t.capped]; sb=u[u.arm==b]; st=u[u.arm==tu]
    Lb,yb,Lt,yt=sb.L.values.astype(float),sb[ep].values,st.L.values.astype(float),st[ep].values
    kn=np.percentile(np.r_[Lb,Lt]/100.,[20,40,60,80])
    def est(Lb, yb, Lt, yt):
        return (yt.mean()-yb.mean(),reg(Lb,yb,Lt,yt,kn,False),reg(Lb,yb,Lt,yt,kn,True),binm(Lb,yb,Lt,yt)[0])
    pt=est(Lb,yb,Lt,yt); _,kt,kb=binm(Lb,yb,Lt,yt)
    bs=np.array([est(Lb[i],yb[i],Lt[j],yt[j]) for i,j in ((rng.integers(0,len(Lb),len(Lb)),rng.integers(0,len(Lt),len(Lt))) for _ in range(B))])
    r=dict(endpoint=ep,pair=name,n_base=len(Lb),n_tuned=len(Lt),gap_L_base_minus_tuned=Lb.mean()-Lt.mean(),bin_tuned_kept=kt,bin_base_kept=kb)
    for k,nm in enumerate(['raw','lin','spline','bin']): r[nm]=pt[k]; r[nm+'_lo'],r[nm+'_hi']=ci(bs[:,k])
    out.append(r)
pd.DataFrame(out).to_csv('uncapped_pooled_contrasts.tsv',sep='\t',index=False,float_format='%.4f')
# ---- within-arm slope in uncapped rows, per arm: all uncapped lengths, and 257-399
sl=[]
def slope(L,y): X=np.c_[np.ones(len(L)),L-L.mean()]; return np.linalg.lstsq(X,y,rcond=None)[0][1]*100
for a,g in t.groupby('arm'):
  for lab,h in (('uncapped, 16-399',g[~g.capped]),('uncapped, 257-399',g[(~g.capped)&(g.L>=257)]),('capped (L=400): slope undefined',None)):
    if h is None: continue
    for ep in ['plddt','ptm']:
        L,y=h.L.values.astype(float),h[ep].values
        bs=[slope(L[i],y[i]) for i in (rng.integers(0,len(L),len(L)) for _ in range(B))]
        sl.append(dict(arm=a,subset=lab,endpoint=ep,n=len(h),slope_per100=slope(L,y),lo=ci(bs)[0],hi=ci(bs)[1]))
pd.DataFrame(sl).to_csv('uncapped_within_arm_slopes.tsv',sep='\t',index=False,float_format='%.4f')
# ---- 3. composition among capped rows
M=['top_frac','entropy_bits','max_run','zlib_ratio','distinct_4mer_frac']
c=t[t.capped]; rows=[]
for name,b,tu in PAIRS:
    for lab,df in (('capped',t[t.capped]),('uncapped',t[~t.capped])):
        for m in M:
            xb=df[df.arm==b][m].values; xt=df[df.arm==tu][m].values
            bs=[rng.choice(xt,len(xt)).mean()-rng.choice(xb,len(xb)).mean() for _ in range(B)]
            rows.append(dict(pair=name,subset=lab,metric=m,base_mean=xb.mean(),tuned_mean=xt.mean(),diff=xt.mean()-xb.mean(),lo=ci(bs)[0],hi=ci(bs)[1],n_base=len(xb),n_tuned=len(xt)))
pd.DataFrame(rows).to_csv('composition_contrasts.tsv',sep='\t',index=False,float_format='%.4f')
# within-arm association (capped rows): Spearman of endpoint vs each metric
rows=[]
for a,g in c.groupby('arm'):
    for ep in ['plddt','ptm']:
        for m in M: rho,p=stats.spearmanr(g[m],g[ep]); rows.append(dict(arm=a,endpoint=ep,metric=m,spearman=rho,n=len(g)))
pd.DataFrame(rows).to_csv('composition_vs_endpoint_within_arm_capped.tsv',sep='\t',index=False,float_format='%.4f')
# mediation: arm coefficient among capped rows without / with composition covariates (log max_run, standardised)
rows=[]
def design(df,cov):
    X=[np.ones(len(df)),(df.arm==tu).values.astype(float)]
    if cov:
        for m in M:
            v=np.log(df[m].values) if m=='max_run' else df[m].values; X.append((v-v.mean())/v.std())
    return np.column_stack(X)
for ep in ['plddt','ptm']:
  for name,b,tu in PAIRS:
    df=c[c.arm.isin([b,tu])].reset_index(drop=True); y=df[ep].values; r=dict(endpoint=ep,pair=name,n_base=(df.arm==b).sum(),n_tuned=(df.arm==tu).sum())
    for cov,nm in ((False,'arm_only'),(True,'arm_plus_composition')):
        r[nm]=np.linalg.lstsq(design(df,cov),y,rcond=None)[0][1]
    bs=[]
    for _ in range(B):
        i=rng.integers(0,len(df),len(df)); d2=df.iloc[i]; bs.append([np.linalg.lstsq(design(d2,cv),d2[ep].values,rcond=None)[0][1] for cv in (False,True)])
    bs=np.array(bs); r['arm_only_lo'],r['arm_only_hi']=ci(bs[:,0]); r['adj_lo'],r['adj_hi']=ci(bs[:,1])
    Xf=design(df,True); beta=np.linalg.lstsq(Xf,y,rcond=None)[0]; r['R2_composition_plus_arm']=1-((y-Xf@beta)**2).sum()/((y-y.mean())**2).sum()
    rows.append(r)
pd.DataFrame(rows).to_csv('capped_arm_effect_with_composition.tsv',sep='\t',index=False,float_format='%.4f')
print('ok')
