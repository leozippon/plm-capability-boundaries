#!/usr/bin/env python3
"""Freeze binding homology rules and run CPU pilot (default), or authorized full screen.

Only the adjudicated mapping with completed gate metadata is accepted. The
original pilot is pre-adjudication; new pilots use a separate output child.
The full stage requires an unchanged frozen contract produced by the pilot. No
model inference/fits, GPU, network or dependency installation. Outputs never
qualify source constructs/baselines or claim evolutionary independence.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

# Bound numerical pools before importing NumPy or the retained-source interface.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.capability.extensions.phenotype_homology import (
    ExactAligner, contract, dump, full_run, load_mapping, load_sources, pilot, sha,
    study_links, task_plan)


def resources():
    return dict(python=sys.version, executable=sys.executable,
        disk_free_bytes=shutil.disk_usage(ROOT).free,
        memory=[s for s in Path('/proc/meminfo').read_text().splitlines()
                if s.startswith(('MemTotal:', 'MemAvailable:'))],
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        thread_environment={k: os.environ[k] for k in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
        execution='CPU exact sequence alignment only; no GPU/network/model inference/fits')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / 'results/extensions/phenotype_followups_20261007/binding'
    parser.add_argument('--mapping-directory', '--mapping', dest='mapping', type=Path,
                        default=base / 'mapping/adjudicated',
                        help='qualified mapping directory; receipt must declare completed adjudication_gate')
    parser.add_argument('--out', type=Path, default=base / 'homology/adjudicated',
                        help='fresh pilot directory, separate from pre-adjudication records')
    parser.add_argument('--runtime', type=Path, default=ROOT / 'runtimes/phenotype-homology')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--mode', choices=('pilot', 'full'), default='pilot')
    args = parser.parse_args()
    if args.mode == 'pilot':
        if args.out.exists():
            parser.error(f'refusing to overwrite frozen pilot: {args.out}')
        args.out.mkdir(parents=True)
        destination = args.out
    else:
        if not (args.out / 'contract.json').exists() or not (args.out / 'pilot.json').exists():
            parser.error('full mode requires frozen pilot contract and timing')
        destination = args.out / 'full'
        if destination.exists():
            parser.error(f'refusing to overwrite full evidence: {destination}')
        destination.mkdir()
    before = resources()
    started = time.monotonic()
    dump(destination / 'resource-receipt.json', dict(status='running', before=before))
    try:
        rows, chains, inputs = load_mapping(args.mapping)
        sources, source_manifest = load_sources(ROOT)
        studies, study_manifest = study_links(args.mapping, rows)
        frozen = contract(inputs, source_manifest, study_manifest)
        if args.mode == 'pilot':
            dump(args.out / 'contract.json', frozen)
            dump(args.out / 'input-plan.json', task_plan(chains, sources))
            dump(args.out / 'study-parse.json', dict(manifest=study_manifest, complex_studies=studies))
        elif json.loads((args.out / 'contract.json').read_text()) != frozen:
            raise ValueError('contract/input mismatch since pilot; do not silently re-freeze full run')
        code_hashes = {p.name: sha(p) for p in
            (Path(__file__), ROOT / 'src/capability/extensions/phenotype_homology.py',
             ROOT / 'src/capability/extensions/phenotype_homology.c')}
        if args.mode == 'full':
            prior = json.loads((args.out / 'pilot.json').read_text())
            if prior['code_sha256'] != code_hashes:
                raise ValueError('implementation changed since pilot; a fresh pilot is required')
        aligner = ExactAligner(args.runtime)
        aligner.prepare([*chains.values(), *(s['sequence'] for s in sources)])
        result = (pilot(aligner, chains, sources, args.workers) if args.mode == 'pilot' else
                  full_run(aligner, rows, chains, sources, studies, destination, args.workers))
        result['contract_sha256'] = sha(args.out / 'contract.json')
        result['backend'] = aligner.provenance
        result['code_sha256'] = code_hashes
        dump(destination / ('pilot.json' if args.mode == 'pilot' else 'receipt.json'), result)
    except Exception as error:
        dump(destination / 'resource-receipt.json', dict(status='failed', before=before,
             after=resources(), wall_seconds=time.monotonic()-started,
             error=f'{type(error).__name__}: {error}'))
        raise
    dump(destination / 'resource-receipt.json', dict(status='complete', before=before,
         after=resources(), wall_seconds=time.monotonic()-started))
    summary = {k: v for k, v in result.items() if k not in ('records', 'backend', 'code_sha256')}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
