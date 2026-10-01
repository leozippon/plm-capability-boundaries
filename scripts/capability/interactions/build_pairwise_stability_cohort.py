#!/usr/bin/env python3
"""Build a draft measured-cycle cohort only with an admitted final group map."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.capability.stability.pairwise_stability import build_cohort


def main():
    import pandas as pd
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parquet', type=Path, nargs='+', required=True)
    p.add_argument('--catalogue', type=Path, required=True)
    p.add_argument('--admitted-groups', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cap', type=int, default=128)
    p.add_argument('--seed', type=int, default=20260923)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    entries=json.loads(a.catalogue.read_text())
    catalogue={r['WT_name']:r for r in entries}
    if len(catalogue)!=len(entries): raise ValueError('duplicate catalogue backgrounds')
    frames=[pd.read_parquet(f) for f in a.parquet]
    out=build_cohort(pd.concat(frames,ignore_index=True),catalogue,json.loads(a.admitted_groups.read_text()),cap=a.cap,seed=a.seed)
    out['source_files']=[{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in [*a.parquet,a.catalogue,a.admitted_groups]]
    out['source_row_order']=[str(f) for f in a.parquet]
    out['code_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(__file__).resolve().parents[3]/'src/capability/stability/pairwise_stability.py']}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__': main()
