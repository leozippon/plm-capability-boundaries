#!/usr/bin/env python3
"""Shared-family simultaneous inference for fixed localization contrasts."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.capability.core.io import write_json
from src.capability.interactions.pairwise_epistasis import ROSTER
KEYS=('unique_own','unique_downstream','joint_over_full','own_minus_downstream')
SEEDS=(20260923,20260924,20260925)

def bands(values,*,draws=10000,seed=20260923):
    values=np.asarray(values,dtype=float)
    if values.ndim!=2 or np.isinf(values).any():raise ValueError('expected finite-or-missing family matrix')
    available=np.isfinite(values);n=available.sum(axis=0)
    if (n<8).any():raise ValueError('fewer than8 available families')
    point=np.nanmean(values,axis=0);se=np.nanstd(values,axis=0,ddof=1)/np.sqrt(n);varying=se>0
    clean=np.nan_to_num(values,nan=0.0);rng=np.random.default_rng(seed);maximum=[]
    for start in range(0,draws,100):
        count=rng.multinomial(len(values),np.full(len(values),1/len(values)),size=min(100,draws-start))
        denominator=count@available.astype(float)
        if (denominator==0).any():raise ValueError('bootstrap draw has no available family for a contrast')
        deviation=count@clean/denominator-point
        maximum.extend(np.max(np.abs(deviation[:,varying]/se[varying]),axis=1) if varying.any() else np.zeros(len(count)))
    critical=float(np.quantile(maximum,.95))
    return dict(method='shared-universe family multinomial bootstrap; available-family ratio mean; maximum absolute centered statistic/fixed SE',draws=draws,seed=seed,critical_value=critical,family_size=values.shape[1],family_universe=len(values),available_families=n.tolist(),point=point.tolist(),interval=np.c_[point-critical*se,point+critical*se].tolist(),conditional_on_fitted_predictions=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--arms',nargs='+',default=[a for a in ROSTER if a!='progen3-3b'])
    args=p.parse_args();cells={};hashes={}
    for path in sorted(args.records.rglob('position_controlled_*_fold*.json')):
        r=json.loads(path.read_bytes())
        if r['arm'] not in args.arms:continue
        key=r['arm'],r['fold_seed']
        if key in cells:raise ValueError('duplicate inference cell')
        if r['retention_worst_nats']!=0 or r['historical_delta_max_abs_nats']!=0:raise ValueError('historical native score admission failed')
        cells[key]=r;hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    expected={(a,s) for a in args.arms for s in SEEDS}
    if set(cells)!=expected:raise ValueError('declared inference roster incomplete')
    families=sorted({str(row['cluster']) for r in cells.values() for row in r['controls']['wall']['assays']});columns=[(a,k) for a in args.arms for k in KEYS]
    results={}
    for control in ('wall','rf3'):
        values=np.full((len(families),len(columns)),np.nan)
        for j,(arm,key) in enumerate(columns):
            records=[cells[arm,s]['controls'][control]['assays'] for s in SEEDS]
            maps=[{row['assay']:row for row in rows} for rows in records]
            if not all(set(m)==set(maps[0]) for m in maps):raise ValueError('assay support changes across seeds')
            group={}
            for assay in maps[0]:
                rows=[m[assay] for m in maps]
                if any(row[key] is None for row in rows):raise ValueError('undefined localization contrast needs explicit support declaration')
                if len({str(row['cluster']) for row in rows})!=1:raise ValueError('family assignment changed')
                group.setdefault(str(rows[0]['cluster']),[]).append(np.mean([row[key] for row in rows]))
            for i,family in enumerate(families):
                if family in group:values[i,j]=np.mean(group[family])
        inference=bands(values)
        inference['contrasts']=[dict(arm=a,key=k,point=inference['point'][j],interval=inference['interval'][j],families=inference['available_families'][j]) for j,(a,k) in enumerate(columns)]
        del inference['point'],inference['interval'],inference['available_families']
        results[control]=inference
    args.out.mkdir(parents=True,exist_ok=True)
    write_json(args.out/'position_simultaneous.json',dict(schema='position_simultaneous',arms=args.arms,keys=KEYS,fold_seeds=SEEDS,families=families,controls=results,source_sha256=hashes,analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),declaration='Multiplicity follow-up declared after pointwise inspection; window control primary, rf3 separate sensitivity; ProGen3-3B excluded pending forward-layout stability',interpretation='Seeds averaged within biological families; shared draws preserve cross-arm dependence with each arm available-family denominator; no pooled effect across differing variant supports'))
if __name__=='__main__':main()
