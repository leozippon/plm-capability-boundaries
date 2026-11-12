"""Split capped rows (L==400) into degenerate and non-degenerate by a criterion taken from the naturally terminated reference,
fixed from composition alone before any score is read. Primary: zlib compressed-bytes-per-residue below the minimum of finished
rows with L>=257 (length-comparable reference), one pooled threshold for all arms. Sensitivities: 1st percentile of the same
reference, and distinct-4-mer fraction below its finished-L>=257 minimum."""
import numpy as np, pandas as pd
rng=np.random.default_rng(20261116); B=4000
D=''
c0=pd.read_csv('per_structure_composition.tsv',sep='\t'); nn=pd.read_csv('natural_neighbour/per_structure_nearest_neighbour.tsv',sep='\t')[['sequence_sha256','identity_over_query']]
t=c0.merge(nn,on='sequence_sha256'); assert len(t)==len(c0)
ref=t[(~t.capped)&(t.L>=257)]
CRIT={'primary: zlib below finished-L>=257 minimum':('zlib_ratio',ref.zlib_ratio.min()),
      'sensitivity: zlib below finished-L>=257 1st percentile':('zlib_ratio',ref.zlib_ratio.quantile(.01)),
      'sensitivity: distinct-4mer below finished-L>=257 minimum':('distinct_4mer_frac',ref.distinct_4mer_frac.min())}
PAIRS=[('112m','protgpt3-112m','protgpt3-112m-dpo'),('1.3b','protgpt3-1.3b','protgpt3-1.3b-dpo'),('10b','protgpt3-10b','protgpt3-10b-dpo'),('mix7','progen2-small-hugohrban','progen2-small-mix7')]
THIN=30; cap=t[t.capped].copy(); ci=lambda x:np.percentile(x,[2.5,97.5])
M=['top_frac','entropy_bits','max_run','zlib_ratio','distinct_4mer_frac']
def design(df,tu,cov,ident):
    X=[np.ones(len(df)),(df.arm==tu).values.astype(float)]
    for m in (M if cov else [])+(['identity_over_query'] if ident else []):
        v=np.log(df[m].values) if m=='max_run' else df[m].values.astype(float); sd=v.std(); X.append((v-v.mean())/(sd if sd>0 else 1))
    return np.column_stack(X)
counts=[];res=[]
for name,(col,thr) in CRIT.items():
    cap['degenerate']=cap[col]<thr
    for a,g in cap.groupby('arm'): counts.append(dict(criterion=name,threshold=thr,arm=a,n_capped=len(g),n_degenerate=int(g.degenerate.sum()),frac_degenerate=g.degenerate.mean()))
    for pair,b,tu in PAIRS:
      for cls,flag in (('degenerate',True),('non-degenerate',False)):
        g=cap[cap.degenerate==flag]; xb,xt=g[g.arm==b],g[g.arm==tu]
        r=dict(criterion=name,pair=pair,cls=cls,n_base=len(xb),n_tuned=len(xt),thin=min(len(xb),len(xt))<THIN)
        if min(len(xb),len(xt))<2: r['note']='not estimable'; res.append(r); continue
        for ep in ('plddt','ptm'):
            yb,yt=xb[ep].values,xt[ep].values; d=yt.mean()-yb.mean()
            bs=[yt[rng.integers(0,len(yt),len(yt))].mean()-yb[rng.integers(0,len(yb),len(yb))].mean() for _ in range(B)]
            r.update({ep+'_base':yb.mean(),ep+'_tuned':yt.mean(),ep+'_diff':d,ep+'_lo':ci(bs)[0],ep+'_hi':ci(bs)[1]})
        r.update(identity_base=xb.identity_over_query.mean(),identity_tuned=xt.identity_over_query.mean())
        if not r['thin']:
            df=g[g.arm.isin([b,tu])].reset_index(drop=True)
            for ep in ('plddt','ptm'):
                y=df[ep].values
                for nm,(cv,idn) in {'adj_identity':(False,True),'adj_composition_identity':(True,True)}.items():
                    r[f'{ep}_{nm}']=np.linalg.lstsq(design(df,tu,cv,idn),y,rcond=None)[0][1]
                    bb=[]
                    for _ in range(1000):
                        d2=df.iloc[rng.integers(0,len(df),len(df))]; bb.append(np.linalg.lstsq(design(d2,tu,cv,idn),d2[ep].values,rcond=None)[0][1])
                    r[f'{ep}_{nm}_lo'],r[f'{ep}_{nm}_hi']=ci(bb)
        res.append(r)
pd.DataFrame(counts).to_csv('capped_degeneracy_counts.tsv',sep='\t',index=False,float_format='%.4f')
pd.DataFrame(res).to_csv('capped_degeneracy_contrasts.tsv',sep='\t',index=False,float_format='%.4f')
print({k:round(v[1],4) for k,v in CRIT.items()}, 'reference n',len(ref))
