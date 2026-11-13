"""Sensitivity (labelled post hoc): the declared pairs-x-strata family with thin cells (either arm <30 rows) removed.
Needs identity_stratified_contrast.tsv and the nearest-neighbour table under RESDIR."""
import os
import numpy as np
import pandas as pd
from src.capability.context import homology as hm
D=os.environ.get('RESDIR','results/external_baseline_analysis/length_adjusted_contrasts_20261110/')
t=pd.read_csv(D+'natural_neighbour/per_structure_nearest_neighbour.tsv',sep='\t'); t['stratum']=t.identity_over_query.map(hm.assign_stratum)
r=pd.read_csv(D+'identity_stratified_contrast.tsv',sep='\t'); A=r[(r.termination=='pooled')&r.thin.eq(False)]
P={'112m':('protgpt3-112m','protgpt3-112m-dpo'),'1.3b':('protgpt3-1.3b','protgpt3-1.3b-dpo'),'10b':('protgpt3-10b','protgpt3-10b-dpo'),'mix7':('progen2-small-hugohrban','progen2-small-mix7')}
rng=np.random.default_rng(20261115); B=10000; cols=[]
for _,c in A.iterrows():
    b,tu=P[c.pair]; g=t[t.stratum==c.stratum]; x=g[g.arm==tu].plddt.values; y=g[g.arm==b].plddt.values
    cols.append(x[rng.integers(0,len(x),(B,len(x)))].mean(1)-y[rng.integers(0,len(y),(B,len(y)))].mean(1)-c['diff'])
h=np.quantile(np.abs(np.column_stack(cols)).max(1),0.95); print(len(A),round(h,3))
A=A.assign(lo=A['diff']-h,hi=A['diff']+h); print(A[['pair','stratum','n_base','n_tuned','diff','margin','lo','hi']].round(2).to_string())
open(D+'identity_stratified_contrast_bands.txt','a').write('sensitivity (labelled post hoc, thin cells removed): family A with both arms >= 30 rows, %d columns: %.3f (seed 20261115)\n'%(len(A),h))
A.to_csv(D+'identity_stratified_contrast_family_A_without_thin_cells.tsv',sep='\t',index=False,float_format='%.3f')
