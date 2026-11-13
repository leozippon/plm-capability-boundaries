"""Declared primary contrast: tuned-minus-base mean pLDDT within each identity stratum (docs/objective-alignment.md),
with the sixteenth amendment's termination split. Boundaries are the frozen ones (src/capability/context/homology.py):
[0,30) [30,70) [70,95) [95,100]; a query with no hit carries identity 0.0 as the search script assigns it."""
import numpy as np
import pandas as pd
from src.capability.context import homology as hm
B=10000; rng=np.random.default_rng(20261114)
D=__import__('os').environ.get('RESDIR','results/external_baseline_analysis/length_adjusted_contrasts_20261110/')
t=pd.read_csv(D+'natural_neighbour/per_structure_nearest_neighbour.tsv',sep='\t')
assert t.identity_over_query.between(0,100).all()
t['stratum']=t.identity_over_query.map(hm.assign_stratum)
ORDER=list(hm.STRATUM_NAMES)
PAIRS=[('112m','protgpt3-112m','protgpt3-112m-dpo'),('1.3b','protgpt3-1.3b','protgpt3-1.3b-dpo'),('10b','protgpt3-10b','protgpt3-10b-dpo'),('mix7','progen2-small-hugohrban','progen2-small-mix7')]
THIN=30   # convention of this analysis, not declared: either arm below this is flagged and excluded from the split-family band
rows=[]; devs={}
def boot(x,y):
    ix=rng.integers(0,len(x),(B,len(x))); iy=rng.integers(0,len(y),(B,len(y)))
    return x[ix].mean(1)-y[iy].mean(1)
for pair,b,tu in PAIRS:
  for s in ORDER:
    for term in ('pooled','finished','capped'):
        g=t[t.stratum==s]
        if term=='finished': g=g[~g.capped]
        if term=='capped': g=g[g.capped]
        xb=g[g.arm==b]; xt=g[g.arm==tu]
        r=dict(pair=pair,stratum=s,termination=term,n_base=len(xb),n_tuned=len(xt))
        if len(xb)<2 or len(xt)<2: r['status']='not estimable (an arm has <2 rows)'; rows.append(r); continue
        yb,yt=xb.plddt.values,xt.plddt.values; d=yt.mean()-yb.mean(); bs=boot(yt,yb)
        r.update(base_mean=yb.mean(),tuned_mean=yt.mean(),diff=d,marg_lo=np.percentile(bs,2.5),marg_hi=np.percentile(bs,97.5),
                 margin=max(2.0,0.25*max(0.0,70-yb.mean())),thin=min(len(xb),len(xt))<THIN,
                 ptm_diff=xt.ptm.mean()-xb.ptm.mean(),mean_identity_base=xb.identity_over_query.mean(),mean_identity_tuned=xt.identity_over_query.mean())
        devs[(pair,s,term)]=bs-d; rows.append(r)
R=pd.DataFrame(rows)
# native simultaneous bands: 0.95 quantile of the largest absolute centred deviation, native pLDDT units
def band(keys):
    m=np.max(np.abs(np.column_stack([devs[k] for k in keys])),axis=1); return np.quantile(m,0.95)
famA=[k for k in devs if k[2]=='pooled']                       # declared family: pairs x strata
famB=[k for k in devs if k[2]!='pooled' and not R[(R.pair==k[0])&(R.stratum==k[1])&(R.termination==k[2])].thin.iloc[0]]   # split cells with both arms >= THIN
hA,hB=band(famA),band(famB)
def fam(r):
    k=(r.pair,r.stratum,r.termination)
    if k in famA: return 'A_declared_pairs_x_strata'
    if k in famB: return 'B_split_cells_both_arms_ge_%d'%THIN
    return 'none'
R['family']=R.apply(lambda r:fam(r) if pd.notna(r.get('diff')) else 'none',axis=1)
R['band_half_width']=R.family.map({'A_declared_pairs_x_strata':hA,'B_split_cells_both_arms_ge_%d'%THIN:hB})
R['sim_lo']=R['diff']-R.band_half_width; R['sim_hi']=R['diff']+R.band_half_width
def verdict(r):
    if pd.isna(r.get('diff')): return 'not estimable'
    if pd.isna(r.band_half_width): return 'thin: no simultaneous reading'
    lo,hi,m=r.sim_lo,r.sim_hi,r.margin
    if lo>m: return 'exceeds margin (positive)'
    if hi<-m: return 'beyond margin (negative)'
    if lo>-m and hi<m: return 'bounded inside margin'
    return 'unresolved'
R['reading']=R.apply(verdict,axis=1)
R.to_csv(D+'identity_stratified_contrast.tsv',sep='\t',index=False,float_format='%.4f')
open(D+'identity_stratified_contrast_bands.txt','w').write('native simultaneous band half-widths (pLDDT, 0-100 scale), %d draws, seed 20261114 (not the declared per-pool seeds)\nfamily A (declared, pooled over termination, %d columns): %.3f\nfamily B (termination-split cells with both arms >= %d rows, %d columns): %.3f\n'%(B,len(famA),hA,THIN,len(famB),hB))
# secondary: identity shift, tuned - base
rows=[]
for pair,b,tu in PAIRS:
    for term in ('pooled','finished','capped'):
        g=t if term=='pooled' else t[t.capped==(term=='capped')]
        xb,xt=g[g.arm==b],g[g.arm==tu]
        bi=boot(xt.identity_over_query.values,xb.identity_over_query.values); bf=boot((xt.identity_over_query.values<30).astype(float),(xb.identity_over_query.values<30).astype(float))
        rows.append(dict(pair=pair,termination=term,n_base=len(xb),n_tuned=len(xt),mean_identity_shift=xt.identity_over_query.mean()-xb.identity_over_query.mean(),lo=np.percentile(bi,2.5),hi=np.percentile(bi,97.5),
            frac_below30_base=(xb.identity_over_query<30).mean(),frac_below30_tuned=(xt.identity_over_query<30).mean(),frac_below30_shift=(xt.identity_over_query<30).mean()-(xb.identity_over_query<30).mean(),f_lo=np.percentile(bf,2.5),f_hi=np.percentile(bf,97.5)))
pd.DataFrame(rows).to_csv(D+'identity_shift_secondary.tsv',sep='\t',index=False,float_format='%.4f')
# instrument: ESMFold confidence by identity stratum, pooled over all 8 arms, plus spearman by group
lv=t.groupby(['stratum','capped']).agg(n=('plddt','size'),plddt_mean=('plddt','mean'),plddt_sd=('plddt','std'),ptm_mean=('ptm','mean')).reset_index()
lv.to_csv(D+'identity_stratum_levels_all_arms.tsv',sep='\t',index=False,float_format='%.4f'); print(lv.round(3).to_string())
print(open(D+'identity_stratified_contrast_bands.txt').read())
