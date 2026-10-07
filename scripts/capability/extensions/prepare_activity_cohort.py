#!/usr/bin/env python3
"""Acquire and qualify public Venus activity states on CPU; no fits or inference."""
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
from src.capability.extensions.activity_cohort import BASE, REVISION, dump, fetch_immutable, prepare

DATA = ROOT/'data/phenotype_followups_20261007/activity'
OUT = ROOT/'results/extensions/phenotype_followups_20261007/activity'


def resources() -> dict:
    return dict(executable=sys.executable, python=sys.version, disk_free_bytes=shutil.disk_usage(ROOT).free,
        memory=[line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:', 'MemAvailable:'))],
        cpus=os.cpu_count(), threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
        mode='standard-library CPU identity parsing; no CUDA/H200/model access')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--offline', action='store_true', help='Use hash-bound prior acquisition receipts, no requests')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('workers must be 1–4')
    if args.out.exists():
        parser.error('output exists; refusing overwrite; choose a fresh --out')
    if resources()['disk_free_bytes'] < 1024**3:
        parser.error('less than 1 GiB free disk; refusing acquisition')
    args.out.mkdir(parents=True)
    args.data.mkdir(parents=True, exist_ok=True)
    before, start = resources(), time.monotonic()
    dump(args.out/'resource-receipt.json', dict(before=before, status='running'))
    try:
        if not args.offline:
            # The advertised root doi.csv is absent; the live full repository tree
            # reveals the author map and WT summary nested under mutant/.
            source_receipts = []
            for source in ['mutant/doi.csv','mutant/dataset_summary.csv']:
                receipt = fetch_immutable(f'{BASE}/resolve/{REVISION}/{source}',
                    args.data/'source'/source.replace('/','__'))
                receipt['source_path'] = source
                source_receipts.append(receipt)
            dump(args.out/'source-acquisition.json', source_receipts)
            if any(r['status'] != 'ready' for r in source_receipts):
                raise ValueError('author DOI/WT summary acquisition failed; see source-acquisition.json')
        counts = prepare(ROOT, args.data, args.out, acquire=not args.offline, workers=args.workers)
        if not args.offline:
            frozen = args.data/'source/acquisition-frozen.json'
            payload = (args.out/'acquisition.json').read_bytes()
            if not frozen.exists():
                with frozen.open('xb') as handle:
                    handle.write(payload)
        failed = counts['failed_activity_files'] > 0
        dump(args.out/'resource-receipt.json', dict(before=before, after=resources(),
             status='partial_sources_failed' if failed else 'complete', wall_seconds=time.monotonic()-start))
        print(json.dumps(counts, indent=2, sort_keys=True), flush=True)
        if failed:
            raise SystemExit('Individual source failures recorded; qualification incomplete')
    except Exception as error:
        dump(args.out/'resource-receipt.json', dict(before=before, after=resources(), status='failed', error=str(error), wall_seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    main()
