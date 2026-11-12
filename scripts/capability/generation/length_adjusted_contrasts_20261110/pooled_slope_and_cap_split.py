import numpy as np, pandas as pd
rng=np.random.default_rng(20261111)
d=pd.read_csv(__import__('os').environ.get('TABLE','input/checkpoint_structures_12of68.tsv'),sep='\t',comment='#').rename(columns={'length_residues':'L','mean_ca_plddt_0_100':'plddt','ptm_0_1':'ptm'})
PAIRS=[('112m','protgpt3-112m','protgpt3-112m-dpo'),('1.3b','protgpt3-1.3b','protgpt3-1.3b-dpo'),('10b','protgpt3-10b','protgpt3-10b-dpo'),('mix7','progen2-small-hugohrban','progen2-small-mix7')]
s=d[(d.L>=257)&(d.L<=512)]; B=2000; rows=[]
def pooled(s,ep):
    x=s.L.values/100.; y=s[ep].values; g=s.arm.values
    xc=x-pd.Series(x).groupby(g).transform('mean').values; yc=y-pd.Series(y).groupby(g).transform('mean').values
    return (xc*yc).sum()/(xc**2).sum()
for ep in ['plddt','ptm']:
    pt=pooled(s,ep); bs=[pooled(s.groupby('arm',group_keys=False).sample(frac=1,replace=True,random_state=int(rng.integers(1<<30))),ep) for _ in range(B)]
    rows.append(dict(kind='pooled_within_arm_slope_per100_all8arms',endpoint=ep,pair='all',est=pt,lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5),n_base=len(s),n_tuned=np.nan))
for ep in ['plddt','ptm']:
  for name,b,t in PAIRS:
    for lab,f in (('L==400 (budget cap; length exactly matched)',lambda z:z.L==400),('L<400 (ended before cap)',lambda z:z.L<400)):
        yb=s[(s.arm==b)&f(s)][ep].values; yt=s[(s.arm==t)&f(s)][ep].values
        bs=[rng.choice(yt,len(yt)).mean()-rng.choice(yb,len(yb)).mean() for _ in range(B)]
        rows.append(dict(kind=lab,endpoint=ep,pair=name,est=yt.mean()-yb.mean(),lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5),n_base=len(yb),n_tuned=len(yt)))
r=pd.DataFrame(rows); r.to_csv('pooled_slope_and_cap_split.tsv',sep='\t',index=False,float_format='%.4f'); print(r.round(3).to_string())
