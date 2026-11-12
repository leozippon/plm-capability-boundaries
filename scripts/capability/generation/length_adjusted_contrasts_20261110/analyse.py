import numpy as np, pandas as pd, json
rng=np.random.default_rng(20261110)
d=pd.read_csv(__import__('os').environ.get('TABLE','input/checkpoint_structures_12of68.tsv'),sep='\t',comment='#')
d=d.rename(columns={'length_residues':'L','mean_ca_plddt_0_100':'plddt','ptm_0_1':'ptm'})
PAIRS=[('112m','protgpt3-112m','protgpt3-112m-dpo'),('1.3b','protgpt3-1.3b','protgpt3-1.3b-dpo'),
       ('10b','protgpt3-10b','protgpt3-10b-dpo'),('mix7','progen2-small-hugohrban','progen2-small-mix7')]
STR={'16-128':(16,128),'129-256':(129,256),'257-512':(257,512)}
B=2000
def sub(a,lo,hi): return d[(d.arm==a)&(d.L>=lo)&(d.L<=hi)]
def ci(x): x=np.asarray(x); x=x[np.isfinite(x)]; return np.percentile(x,[2.5,97.5])
def slope(L,y):
    X=np.c_[np.ones(len(L)),L-L.mean()]; return np.linalg.lstsq(X,y,rcond=None)[0][1]
def basis(L,kn,flex):
    cols=[L/100.]
    if flex:
        cols+= [(L/100.)**2,(L/100.)**3]+[np.maximum(L/100.-k,0)**3 for k in kn]
    return np.column_stack(cols)
def reg(Lb,yb,Lt,yt,kn,flex):
    L=np.r_[Lb,Lt]; y=np.r_[yb,yt]; a=np.r_[np.zeros(len(Lb)),np.ones(len(Lt))]
    X=np.column_stack([np.ones(len(L)),a,basis(L,kn,flex)])
    return np.linalg.lstsq(X,y,rcond=None)[0][1]
def bin_match(Lb,yb,Lt,yt,w=16):
    bt=Lt//w; bb=Lb//w; num=0;den=0;nt=0;nb=0
    for k in np.unique(bt):
        mb=bb==k
        if mb.sum()==0: continue
        m=bt==k; num+=m.sum()*(yt[m].mean()-yb[mb].mean()); den+=m.sum(); nt+=m.sum(); nb+=mb.sum()
    return (num/den if den else np.nan), nt, nb
def nn_match(Lb,yb,Lt,yt,cal=5):
    # each tuned matched to nearest base length (with replacement); drop if gap>caliper
    o=np.argsort(Lb); Ls=Lb[o]; ys=yb[o]; diffs=[]
    for l,y in zip(Lt,yt):
        i=np.searchsorted(Ls,l); c=[j for j in (i-1,i) if 0<=j<len(Ls)]
        best=min(c,key=lambda j:abs(Ls[j]-l)); lo=np.searchsorted(Ls,Ls[best],'left'); hi=np.searchsorted(Ls,Ls[best],'right')
        if abs(Ls[best]-l)<=cal: diffs.append(y-ys[lo:hi].mean())
    return (np.mean(diffs) if diffs else np.nan), len(diffs)
out=[]; slopes=[]
def fmt(v,c): return f"{v:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]"
for ep in ['plddt','ptm']:
  # --- slopes in 257-512
  lo,hi=STR['257-512']
  for name,base,tun in PAIRS:
    for role,a in (('base',base),('tuned',tun)):
        s=sub(a,lo,hi); L=s.L.values.astype(float); y=s[ep].values
        bs=[]; 
        for _ in range(B):
            i=rng.integers(0,len(L),len(L)); bs.append(slope(L[i],y[i]))
        slopes.append(dict(endpoint=ep,pair=name,role=role,arm=a,n=len(L),per100=slope(L,y)*100,lo=ci(bs)[0]*100,hi=ci(bs)[1]*100,
            frac_at_400=float((s.L==400).mean()),meanL=L.mean()))
  # --- contrasts
  for sname,(lo,hi) in STR.items():
    for name,base,tun in PAIRS:
        sb=sub(base,lo,hi); st=sub(tun,lo,hi)
        Lb,yb=sb.L.values.astype(float),sb[ep].values; Lt,yt=st.L.values.astype(float),st[ep].values
        r=dict(endpoint=ep,stratum=sname,pair=name,n_base=len(Lb),n_tuned=len(Lt))
        if len(Lb)<2 or len(Lt)<2: out.append(r); continue
        r['gap_L_base_minus_tuned']=Lb.mean()-Lt.mean()
        kn=np.percentile(np.r_[Lb,Lt]/100.,[25,50,75])
        def est(Lb,yb,Lt,yt):
            e=dict(raw=yt.mean()-yb.mean(),lin=reg(Lb,yb,Lt,yt,kn,False),flex=reg(Lb,yb,Lt,yt,kn,True))
            e['bin'],_,_=bin_match(Lb,yb,Lt,yt); e['nn'],_=nn_match(Lb,yb,Lt,yt); return e
        pt=est(Lb,yb,Lt,yt); _,r['bin_tuned_kept'],r['bin_base_kept']=bin_match(Lb,yb,Lt,yt); _,r['nn_tuned_kept']=nn_match(Lb,yb,Lt,yt)
        r['slope_base_per100']=slope(Lb,yb)*100; r['slope_tuned_per100']=slope(Lt,yt)*100
        bs={k:[] for k in pt}; sd=[]
        for _ in range(B):
            ib=rng.integers(0,len(Lb),len(Lb)); it=rng.integers(0,len(Lt),len(Lt))
            e=est(Lb[ib],yb[ib],Lt[it],yt[it])
            for k in e: bs[k].append(e[k])
            sd.append(slope(Lt[it],yt[it])*100-slope(Lb[ib],yb[ib])*100)
        for k in pt: r[k]=pt[k]; r[k+'_lo'],r[k+'_hi']=ci(bs[k])
        r['slopediff_per100']=r['slope_tuned_per100']-r['slope_base_per100']; r['slopediff_lo'],r['slopediff_hi']=ci(sd)
        # adequacy of linear vs flexible: partial F on pooled (arm + L) vs (arm + cubic spline)
        L=np.r_[Lb,Lt]; y=np.r_[yb,yt]; a=np.r_[np.zeros(len(Lb)),np.ones(len(Lt))]
        def rss(X): b=np.linalg.lstsq(X,y,rcond=None)[0]; return ((y-X@b)**2).sum(),X.shape[1]
        r1,p1=rss(np.column_stack([np.ones(len(L)),a,basis(L,kn,False)])); r2,p2=rss(np.column_stack([np.ones(len(L)),a,basis(L,kn,True)]))
        from scipy import stats
        F=((r1-r2)/(p2-p1))/(r2/(len(L)-p2)); r['linear_vs_flex_F_p']=1-stats.f.cdf(F,p2-p1,len(L)-p2)
        # overlap: share of tuned rows whose length lies within base range of 5th-95th pct; frac at 400
        r['frac400_base']=float((Lb==400).mean()); r['frac400_tuned']=float((Lt==400).mean())
        r['overlap_tuned_in_base_p5_95']=float(((Lt>=np.percentile(Lb,5))&(Lt<=np.percentile(Lb,95))).mean())
        out.append(r)
pd.DataFrame(slopes).to_csv('slopes_257_512.tsv',sep='\t',index=False,float_format='%.4f')
pd.DataFrame(out).to_csv('contrasts_by_stratum.tsv',sep='\t',index=False,float_format='%.4f')
print(pd.DataFrame(slopes).round(2).to_string())
