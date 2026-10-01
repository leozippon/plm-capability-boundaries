#!/usr/bin/env python3
"""Reproduce the historical eighteen-checkpoint correlation by exact family draws."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.capability.reporting.audit_followup_support import FAMILIES
from src.capability.context import cross_measure_association as source
from src.capability.core.io import sha256_file


def compositions(total, parts):
    if parts==1:
        yield (total,)
    else:
        for n in range(total+1):
            for rest in compositions(total-n,parts-1):
                yield (n,)+rest


def exact(x,y,clusters,*,family_means=False):
    if family_means:
        x=np.array([x[c].mean() for c in clusters])
        y=np.array([y[c].mean() for c in clusters])
        clusters=[np.array([i]) for i in range(len(clusters))]
    n=len(clusters)
    values=[]; weights=[]; invalid_mass=0.; count=0
    for multiplicities in compositions(n,n):
        count+=1
        mass=math.factorial(n)/(n**n*math.prod(math.factorial(m) for m in multiplicities))
        indices=np.concatenate([np.tile(c,m) for c,m in zip(clusters,multiplicities)])
        value=source.spearman_rank(x[indices],y[indices])
        if value is None:
            invalid_mass+=mass
        else:
            values.append(value);weights.append(mass)
    order=np.argsort(values);values=np.asarray(values)[order];weights=np.asarray(weights)[order]
    cumulative=np.cumsum(weights)/sum(weights)
    interval=[float(values[min(np.searchsorted(cumulative,q),len(values)-1)]) for q in (.025,.975)]
    return {'point':source.spearman_rank(x,y),'interval':interval,'compositions':count,
            'clusters':n,'undefined_probability':invalid_mass,
            'quantile':'inverse CDF of multinomial-weighted exact compositions, conditional on finite correlation'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise SystemExit('refuse existing correlation receipt')
    path=args.root/'results/shared/cross_measure_20260918/cross_measure_association.json'
    published=json.loads(path.read_text())
    record=next(r for r in published['pooled'] if r['pair_id']=='P-S1-U2')
    arms,x,y,pending,dropped=source.pair_points(record['arms'],'S1','U2')
    if pending or dropped or len(arms)!=18:raise ValueError('historical support changed')
    x,y=np.asarray(x),np.asarray(y)
    if not np.isclose(source.spearman_rank(x,y),record['spearman'],rtol=0,atol=1e-15):
        raise ValueError('source rows fail to reproduce historical point')
    reverse={a:f for f,(_,aa) in FAMILIES.items() for a in aa}
    mapping={f:[a for a in arms if reverse[a]==f] for f in sorted({reverse[a] for a in arms})}
    previous={}
    for arm in arms:
        key=next((k for k,v in source.LINEAGES.items() if arm in v),arm)
        previous.setdefault(key,[]).append(arm)
    if {frozenset(v) for v in previous.values()}!={frozenset(v) for v in mapping.values()}:
        raise ValueError('historical and current release-family memberships differ')
    clusters=[np.array([arms.index(a) for a in aa]) for aa in mapping.values()]
    merged=dict(mapping)
    merged['Galactica']=merged['Galactica']+merged.pop('InstructProtein')
    result={'source_sha256':sha256_file(path),'source_module_sha256':sha256_file(Path(source.__file__)),
            'script_sha256':sha256_file(Path(__file__)),'source_pair':'P-S1-U2',
            'rows':[{'arm':a,'mutation_score':float(xx),'profile_rate':float(yy)} for a,xx,yy in zip(arms,x,y)],
            'release_family_mapping':mapping,'historical_group_membership_equal':True,
            'checkpoint_count':18,'release_family_count':9,
            'checkpoint_resampling_historical':record['interval'],
            'cluster_resampling':exact(x,y,clusters),
            'family_means':exact(x,y,clusters,family_means=True),
            'merge_galactica_instructprotein_sensitivity':exact(x,y,[np.array([arms.index(a) for a in aa]) for aa in merged.values()]),
            'limitations':'Fixed heterogeneous model/support panel; release families are not certified independent training ancestries; generation campaigns fixed.'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('LINEAGE_CORRELATION_EXIT=0',flush=True)


if __name__=='__main__':main()
