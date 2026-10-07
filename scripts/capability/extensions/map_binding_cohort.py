#!/usr/bin/env python3
"""Acquire bounded public CIF payloads and prepare exact operational PDB states."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.capability.extensions.binding_cohort import dump
from src.capability.extensions.binding_mapping import prepare_mapping


def resources():
    return dict(python=sys.version, executable=sys.executable, disk_free_bytes=shutil.disk_usage(ROOT).free,
                memory=[x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:', 'MemAvailable:'))],
                threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                command_argv=sys.argv,
                execution='CPU mapping only; no model inference')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phase1 = ROOT/'results/extensions/phenotype_followups_20261007/binding'
    parser.add_argument('--phase1', type=Path, default=phase1)
    parser.add_argument('--out', type=Path, default=phase1/'mapping')
    parser.add_argument('--cache', type=Path, default=ROOT/'data/phenotype_followups_20261007/binding-structures')
    parser.add_argument('--workers', type=int, choices=range(1,5), default=4)
    parser.add_argument('--cache-only', action='store_true', help='Prohibit downloads; only existing local/cached CIF payloads')
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f'output already exists; refusing overwrite: {args.out}')
    before = resources()
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out/'resource-receipt.json',dict(before=before,status='running'))
    try:
        counts = prepare_mapping(ROOT,args.phase1,args.out,args.cache,args.workers,cache_only=args.cache_only)
    except Exception as error:
        dump(args.out/'resource-receipt.json',dict(before=before,after=resources(),status='failed',error=str(error),wall_seconds=time.monotonic()-started))
        raise
    dump(args.out/'resource-receipt.json',dict(before=before,after=resources(),status='complete',wall_seconds=time.monotonic()-started))
    print(json.dumps(counts,indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
