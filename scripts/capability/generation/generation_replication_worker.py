#!/usr/bin/env python3
"""Consume manifest cells once; late workers may join the same claim directory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def isolated_cuda_environment(device, environment):
    """Native generators use the default device; isolate the assigned physical card."""
    if not device.startswith('cuda:'):
        raise ValueError('generation workers require an explicit CUDA index')
    index = int(device.split(':', 1)[1])
    result = dict(environment)
    visible = result.get('CUDA_VISIBLE_DEVICES')
    result['CUDA_VISIBLE_DEVICES'] = visible.split(',')[index] if visible else str(index)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=(Path(__file__).resolve().parents[3] / 'configs/generation_replication_manifest.json'))
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--only-galactica-30b', action='store_true')
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    child_environment = isolated_cuda_environment(a.device, os.environ)
    a.workspace.mkdir(parents=True, exist_ok=True)
    claims = a.workspace / 'claims'
    claims.mkdir(exist_ok=True)
    completed = []
    failures = []
    # Both streams of priority cells precede the remainder, independent of outcomes.
    priority = set(manifest['priority_cells'])
    cells = sorted(manifest['cells'], key=lambda c: (c['cell'] not in priority,
                    manifest['priority_cells'].index(c['cell']) if c['cell'] in priority else 99))
    tasks = [(cell, campaign) for selected in (True, False)
             for campaign in manifest['campaigns'] for cell in cells
             if (cell['cell'] in priority) == selected
             and (cell['arm'] == 'galactica-30b') == a.only_galactica_30b]
    for cell, campaign in tasks:
        label = f'{campaign["id"]}__{cell["cell"]}'
        try:
            fd = os.open(claims / (label + '.json'), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        with os.fdopen(fd, 'w') as stream:
            json.dump({'cell': cell['cell'], 'campaign': campaign['id'],
                       'device': a.device, 'started_unix': time.time()}, stream)
        target = a.workspace / 'cells' / label
        command = [sys.executable, str(Path(__file__).with_name('replicate_generation.py')),
                   '--manifest', str(a.manifest), '--cell', cell['cell'],
                   '--campaign', campaign['id'], '--device', 'cuda:0', '--out', str(target)]
        log_dir = a.workspace / 'logs'
        log_dir.mkdir(exist_ok=True)
        with (log_dir / (label + '.log')).open('w') as log:
            result = subprocess.run(command, env=child_environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        receipt = {'label': label, 'returncode': result.returncode,
                   'finished_unix': time.time(), 'output': str(target)}
        (claims / (label + '.terminal.json')).write_text(json.dumps(receipt, indent=2) + '\n')
        (completed if result.returncode == 0 else failures).append(receipt)
        print(json.dumps(receipt), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'generation_worker.json').write_text(json.dumps({'completed': completed, 'failures': failures}, indent=2) + '\n')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
