"""Shift-share split of the pooled pLDDT gain via the project's own function, per pair and termination cell,
plus a sensitivity band for the declared family with thin cells (either arm <30) removed."""
import pandas as pd
from src.capability.context import homology as hm
from src.capability.generation import objective_tuned_analysis as ota
D=__import__('os').environ.get('RESDIR','results/external_baseline_analysis/length_adjusted_contrasts_20261110/')
t=pd.read_csv(D+'natural_neighbour/per_structure_nearest_neighbour.tsv',sep='\t'); t['stratum']=t.identity_over_query.map(hm.assign_stratum)
sec=pd.read_csv(D+'identity_shift_secondary.tsv',sep='\t').set_index(['pair','termination'])
PAIRS=[('112m','protgpt3-112m','protgpt3-112m-dpo'),('1.3b','protgpt3-1.3b','protgpt3-1.3b-dpo'),('10b','protgpt3-10b','protgpt3-10b-dpo'),('mix7','progen2-small-hugohrban','progen2-small-mix7')]
rows=[]
for pair,b,tu in PAIRS:
  for term in ('pooled','finished','capped'):
    g=t if term=='pooled' else t[t.capped==(term=='capped')]
    def tab(a):
        x=g[g.arm==a]; return {s:dict(share=(x.stratum==s).mean(),mean=x[x.stratum==s].plddt.mean()) for s in hm.STRATUM_NAMES if (x.stratum==s).sum()>0}
    ss=ota.shift_share(tab(b),tab(tu)); raw=g[g.arm==tu].plddt.mean()-g[g.arm==b].plddt.mean()
    s=sec.loc[(pair,term)]
    rows.append(dict(pair=pair,termination=term,raw_whole_gain=raw,pooled_gain=ss['pooled_gain'],within=ss['within_stratum_term'],composition=ss['composition_term'],
      strata_dropped=','.join(ss['strata_without_both_sides']) or '-',composition_over_half_gain=bool(ss['composition_term']>0.5*ss['pooled_gain']) if ss['pooled_gain']>0 else False,
      identity_shift_band_excludes_zero=bool(s.lo>0 or s.hi<0),identity_shift=s.mean_identity_shift))
pd.DataFrame(rows).to_csv(D+'identity_shift_share.tsv',sep='\t',index=False,float_format='%.3f'); print(pd.DataFrame(rows).round(2).to_string())
