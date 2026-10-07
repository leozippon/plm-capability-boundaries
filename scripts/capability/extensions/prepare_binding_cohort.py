#!/usr/bin/env python3
"""CPU-only first-stage binding qualification. No downloads, fits or inference."""
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
from src.capability.extensions.binding_cohort import dump, prepare

OUT = ROOT / 'results/extensions/phenotype_followups_20261007/binding'


def resources() -> dict:
    return dict(python=sys.version, executable=sys.executable, cpus=os.cpu_count(),
                disk_free_bytes=shutil.disk_usage(ROOT).free,
                memory=[line for line in Path('/proc/meminfo').read_text().splitlines()
                        if line.startswith(('MemAvailable:', 'MemTotal:'))],
                threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                execution='standard-library CPU parsing only; single process, no GPU')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        parser.error(f'output already exists; refusing overwrite: {out}')
    started = time.monotonic()
    before = resources()
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'resource-receipt.json', dict(before=before, status='running'))
    try:
        counts = prepare(ROOT, out)
    except Exception as error:
        dump(out / 'resource-receipt.json', dict(before=before, after=resources(), status='failed', error=str(error)))
        raise
    dump(out / 'resource-receipt.json', dict(before=before, after=resources(), status='complete', wall_seconds=time.monotonic()-started))
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
