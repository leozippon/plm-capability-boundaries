#!/usr/bin/env python3
"""Freeze the membrane measurement/identity panel on CPU, without inference or fits."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import time

# Set limits before imports that may load numpy through shared mapping utilities.
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.capability.extensions.membrane_cohort import dump, prepare


def resources() -> dict:
    return dict(executable=sys.executable, python=sys.version,
                disk_free_bytes=shutil.disk_usage(ROOT).free,
                memory=[line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:', 'MemAvailable:'))],
                threads={key: os.environ[key] for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                execution='CPU-only identity/CSV parsing; no models, GPU, H200, fits or downloads')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/extensions/phenotype_followups_20261007/membrane/source-anchored-sgca')
    args = parser.parse_args()
    out = args.out.resolve()
    allowed = (ROOT / 'results/extensions/phenotype_followups_20261007/membrane').resolve()
    if allowed not in out.parents:
        parser.error('output must be a new subdirectory; preserve original membrane root outputs')
    if out.exists():
        parser.error(f'refusing to overwrite existing output: {out}')
    start = time.monotonic()
    before = resources()
    out.mkdir(parents=True)
    dump(out / 'resource-receipt.json', dict(before=before, status='running'))
    try:
        summary = prepare(ROOT, out)
    except Exception as error:
        dump(out / 'resource-receipt.json', dict(before=before, after=resources(), status='failed', error=str(error)))
        raise
    dump(out / 'resource-receipt.json', dict(before=before, after=resources(), status='complete', wall_seconds=time.monotonic() - start))
    print('Accepted row/state counts:', {key: (value['accepted_rows'], value['accepted_states']) for key, value in summary['channels'].items()})


if __name__ == '__main__':
    main()
