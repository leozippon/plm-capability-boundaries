#!/usr/bin/env python3
"""Replay family-level failure conditions and mutation-level profile increments."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.capability.context.profile_increment import common_vector_support, context_rescue, family_regression, mutation_support, profile_increment, summarize
from src.capability.core.amino_acids import AA20

RESULTS = ROOT / 'results'
RETRIEVAL = RESULTS / 'R5/retrieval_bound_20260807'
SOURCES = {
    **{a: RESULTS / f'R1/progen3_stage_20260917/retrieval_bound/model_{a}.json'
       for a in ('progen3-112m', 'progen3-3b')},
    **{a: RESULTS / f'R1/adaptation_lineage_20260813/retrieval_bound/model_{a}.json'
       for a in ('llama-2-7b', 'prollama-stage-1', 'prollama')},
    **{a: RESULTS / f'R1/progen2_fitness_20260813/retrieval_bound/model_{a}.json'
       for a in ('progen2-medium', 'progen2-large', 'progen2-xlarge')},
    **{a: RESULTS / f'R1/proteingym_gap_20260807/analyse_verify/model_{a}.json'
       for a in ('progen2-small', 'progen2-base')},
    'protgpt2': RETRIEVAL / 'model_protgpt2.json',
}
COVARIATES=['log10_neff','max_identity_over_query','profile_entropy','log10_length']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,required=True,
                   help='kept for the command interface; retrieval profiles are read from results/R5/retrieval_bound_20260807')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',action='append',default=[],help='arm=payload path; adds/replaces a source')
    p.add_argument('--vectors',action='append',default=[],type=Path)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20260923)
    p.add_argument('--device',default='cpu',help='accepted campaign interface; analysis is CPU-only')
    p.add_argument('--vectors-only',action='store_true')
    p.add_argument('--common-support',action='store_true',help='also analyze the exact shared assay/mutation panel across all --vectors arms')
    args=p.parse_args()
    manifest={}
    def read(path):
        raw=path.read_bytes();manifest[str(path)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    report=dict(schema_version='d1_profile_increment_v1',created_utc=datetime.now(timezone.utc).isoformat(),
                seed=args.seed,bootstrap=args.bootstrap,units='dimensionless Spearman; entropy nats/residue; identity percent; lengths residues')
    if not args.vectors_only:
        base=RETRIEVAL
        lookup=read(base/'lookup.json'); profiles=read(base/'profiles.json'); catalogue=read(base/'wildtypes.json')
        path=base/'profiles.npz';manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        stored=np.load(path)
        background=np.array([catalogue['corpus']['background'][aa] for aa in AA20])
        alpha=lookup['settings']['alpha']
        cov={}
        for identifier,r in profiles['profiles'].items():
            frequency=np.asarray(stored[identifier],float)+alpha*background
            probability=frequency/frequency.sum(1,keepdims=True)
            cov[identifier]=dict(log10_neff=r['log10_neff'],max_identity_over_query=r['max_identity_over_query'],
                profile_entropy=float(-(probability*np.log(probability)).sum(1).mean()),log10_length=float(np.log10(r['length'])),
                n_profile_sequences=r['n_profile_sequences'],supported_column_fraction=r['supported_column_fraction'],
                diamond_hit_list_saturated=r['diamond_hit_list_saturated'])
        cuts={k:np.quantile([r[k] for r in cov.values()],[1/3,2/3]).tolist() for k in COVARIATES}
        lookup_rows={r['assay']:r for r in lookup['assays']}
        paths=dict(SOURCES)
        for item in args.model:
            a,v=item.split('=',1);paths[a]=Path(v)
        models={}
        for arm,path in paths.items():
            source=read(path);rows=[]
            for model in source['assays']:
                if model.get('spearman') is None:continue
                row=lookup_rows[model['assay']]
                if model['mutant_digest'] != row['mutant_digest']:
                    raise ValueError(f'{arm}/{model["assay"]}: variant digest mismatch')
                if model['wildtype_id'] != row['wildtype_id']:
                    raise ValueError('wild-type identity mismatch')
                rows.append(dict(assay=model['assay'],cluster=row['cluster'],model_spearman=model['spearman'],
                    lookup_spearman=row['spearman']['lookup'],model_minus_lookup=model['spearman']-row['spearman']['lookup'],
                    **cov[row['wildtype_id']]))
            strata={}
            for k in COVARIATES:
                groups={}
                for index,label in enumerate(('low','middle','high')):
                    subset=[r for r in rows if np.searchsorted(cuts[k],r[k],side='right')==index]
                    groups[label]={metric:summarize(subset,metric,bootstrap=args.bootstrap,seed=args.seed) for metric in ('model_minus_lookup','model_spearman','lookup_spearman')}
                strata[k]=groups
            regression=family_regression(rows,COVARIATES,bootstrap=args.bootstrap,seed=args.seed)
            models[arm]=dict(n_assays=len(rows),n_families=len({r['cluster'] for r in rows}),
                n_saturated_assays=sum(r['diamond_hit_list_saturated'] for r in rows),
                overall=summarize(rows,'model_minus_lookup',bootstrap=args.bootstrap,seed=args.seed),
                strata=strata,regression=regression,assays=rows)
        report['failure_conditions']=dict(stratum_cutpoints=cuts,n_catalogue_wildtypes=len(cov),models=models)
    vector_sources=[]
    for path in args.vectors:
        source=read(path)
        if source.get('status') != 'complete':
            raise ValueError(f'{path}: mutation-vector scoring is not complete')
        if not source['assays']:
            raise ValueError(f'{path}: no mutation-vector assays')
        vector_sources.append(source)
    if args.common_support:
        common_panels,common_support=common_vector_support(vector_sources)
    def analyze(path,source,assays):
        conditions=sorted(set.intersection(*(set(r['scores']) for r in assays)))
        if not conditions:
            raise ValueError(f'{path}: no shared scoring condition')
        return dict(source=str(path),arm=source.get('arm'),support=mutation_support(assays),
            context_rescue=context_rescue(assays,bootstrap=args.bootstrap,seed=args.seed),
            conditions={c:profile_increment(assays,condition=c,bootstrap=args.bootstrap,seed=args.seed) for c in conditions})
    report['profile_increments']=[analyze(path,source,source['assays'])
                                 for path,source in zip(args.vectors,vector_sources)]
    if args.common_support:
        report['common_support']=dict(support=common_support,
            status='complete' if common_support['n_assays'] else 'undefined_empty_intersection',
            profile_increments=[analyze(path,source,assays)
                for path,source,assays in zip(args.vectors,vector_sources,common_panels)
                if assays])
    report['source_sha256']=manifest
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'profile_increment.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({a:r['overall'] for a,r in report.get('failure_conditions',{}).get('models',{}).items()},indent=2))


if __name__=='__main__':main()
