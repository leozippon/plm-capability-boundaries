#!/usr/bin/env python3
"""Collect complete three-seed matched-support position results without pooling arms."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.capability.core.io import write_json
from src.capability.interactions.pairwise_epistasis import ROSTER
from src.capability.position.position_readout import CONTRASTS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allow-incomplete',action='store_true')
    args=p.parse_args()
    cells={};hashes={}
    for path in sorted(args.records.rglob('position_controlled_*_fold*.json')):
        raw=path.read_bytes();r=json.loads(raw)
        key=(r['arm'],r['fold_seed'])
        if key in cells:raise ValueError(f'duplicate cell {key}')
        cells[key]=r;hashes[str(path)]=hashlib.sha256(raw).hexdigest()
    required={(arm,seed) for arm in ROSTER for seed in (20260923,20260924,20260925)}
    missing=sorted(required-set(cells))
    if missing and not args.allow_incomplete:raise ValueError(f'missing cells {missing}')
    arms=[]
    for arm in ROSTER:
        rows=[cells.get((arm,s)) for s in (20260923,20260924,20260925)]
        if any(r is None for r in rows):continue
        if any(r['support']!=rows[0]['support'] for r in rows[1:]):
            raise ValueError(f'{arm}: support changes across split seeds')
        contrasts={}
        for local in ('wall','rf3'):
            contrasts[local]={}
            extra=('historical_over_control','full_minus_historical') if rows[0].get('paired_measurement') else ()
            for key in (*CONTRASTS,'qualification_over_C','qualification_over_C_P',*extra):
                summaries=[r['controls'][local]['summaries'][key] for r in rows]
                contrasts[local][key]=dict(cells=summaries,positive_all_seeds=all(s['interval'] is not None and s['interval'][0]>0 for s in summaries),negative_all_seeds=all(s['interval'] is not None and s['interval'][1]<0 for s in summaries))
        row=rows[0]
        arms.append(dict(arm=arm,measurement_protocol='fresh paired native score' if row.get('paired_measurement') else 'historical native score',variants=row['variants'],assays=row['assays'],families=row['families'],
                         eligible_singles=sum(r['singles'] for r in row['support']),
                         aligned_fraction=row['variants']/sum(r['singles'] for r in row['support']),
                         historical_delta_max_abs_nats=row['historical_delta_max_abs_nats'],
                         aligned_upstream_max_abs_nats=row['aligned_upstream_max_abs_nats'],
                         controls=contrasts))
    args.out.mkdir(parents=True,exist_ok=True)
    report=dict(schema='position_controlled_panel_v1',arms=arms,missing_cells=missing,source_sha256=hashes,
                interpretation='Per-arm conditional support only; no pooled cross-arm effect and no extrapolation to excluded variants. Positive-all-seeds indicates repeated pointwise family-bootstrap intervals, not multiplicity-adjusted panel discovery.',
                status='complete' if not missing else 'incomplete')
    write_json(args.out/'position_controlled_panel.json',report)
    print(json.dumps(dict(status=report['status'],arms=len(arms),missing=len(missing))),flush=True)

if __name__=='__main__':main()
