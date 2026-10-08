#!/usr/bin/env python3
"""Bounded CPU/public acquisition and identity qualification; no models or fits."""
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
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
from src.capability.extensions.cellular_fitness_cohort import dump, prepare


def resources():
    return dict(executable=sys.executable, cpus=os.cpu_count(), disk_free_bytes=shutil.disk_usage(ROOT).free,
                memory=[line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:', 'MemTotal:'))],
                threads={v:os.environ[v] for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                execution='single-process CPU parsing / serial public HTTP only')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='replay cached public payloads into a fresh output directory')
    parser.add_argument('--out', type=Path, default=ROOT/'results/extensions/phenotype_followups_20261007/cellular-fitness')
    parser.add_argument('--cache', type=Path, default=ROOT/'data/phenotype_followups_20261007/cellular-fitness')
    args = parser.parse_args()
    out = args.out.resolve()
    allowed_out = ROOT/'results/extensions/phenotype_followups_20261007/cellular-fitness'
    allowed_cache = ROOT/'data/phenotype_followups_20261007/cellular-fitness'
    if not out.is_relative_to(allowed_out) or not args.cache.resolve().is_relative_to(allowed_cache):
        parser.error('output/cache must remain in assigned cellular-fitness scope')
    if out.exists():
        parser.error('refusing existing output; choose a fresh child directory for replay')
    out.mkdir(parents=True)
    started, before = time.monotonic(), resources()
    dump(out/'resource-receipt.json', dict(status='running', before=before))
    try:
        counts = prepare(ROOT, args.cache.resolve(), out, acquire=not args.offline)
    except Exception as error:
        dump(out/'resource-receipt.json', dict(status='failed', before=before, after=resources(), error=str(error)))
        raise
    dump(out/'resource-receipt.json', dict(status='complete', before=before, after=resources(), seconds=time.monotonic()-started))
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
