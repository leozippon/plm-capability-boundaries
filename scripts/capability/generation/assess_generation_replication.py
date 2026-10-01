#!/usr/bin/env python3
"""Assess completed generation cells; final aggregation still requires the full census."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.capability.generation.generation_replication import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=(Path(__file__).resolve().parents[3] / 'configs/generation_replication_manifest.json'))
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads-per-cell', type=int, default=8)
    p.add_argument('--available-only', action='store_true')
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    selected, pending = [], []
    for cell in manifest['cells']:
        for campaign in manifest['campaigns']:
            label = f'{campaign["id"]}__{cell["cell"]}'
            directory = a.workspace / 'cells' / label
            summary_path = directory / 'generation_replication.json'
            if not summary_path.exists():
                pending.append(label)
                continue
            summary = json.loads(summary_path.read_text())
            if summary['manifest_sha256'] != sha256(a.manifest) or summary['attempts_sha256'] != sha256(directory / 'attempts.jsonl'):
                raise ValueError(f'generation receipt mismatch: {label}')
            selected.append(label)
    if pending and not a.available_only:
        raise ValueError(f'cannot assess incomplete generation census: {pending}')
    if not selected:
        raise ValueError('no completed generation cell is available')
    logs = a.workspace / 'profile_logs'
    logs.mkdir(exist_ok=True)

    def assess(label):
        target = a.workspace / 'profiles' / label
        path = target / 'generation_profiles.json'
        if path.exists():
            report = json.loads(path.read_text())
            if report['configuration']['manifest_sha256'] != sha256(a.manifest) or report['configuration']['attempts_sha256'] != sha256(a.workspace / 'cells' / label / 'attempts.jsonl'):
                raise ValueError(f'profile receipt mismatch: {label}')
            return {'label': label, 'returncode': 0, 'retained_complete': True}
        command = [sys.executable, str(Path(__file__).with_name('annotate_generation_replication.py')),
                   '--manifest', str(a.manifest), '--attempts', str(a.workspace / 'cells' / label / 'attempts.jsonl'),
                   '--pool', str(a.inputs / 'archive/logs/R6/gate_generative_control/pool/reservoir.json'),
                   '--hmmscan', str(a.inputs / 'hmmer/bin/hmmscan'), '--pfam', str(a.inputs / 'pfam/Pfam-A.hmm'),
                   '--out', str(target), '--workers', str(a.threads_per_cell)]
        with (logs / (label + '.log')).open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        receipt = {'label': label, 'returncode': result.returncode, 'retained_complete': False}
        print(json.dumps(receipt), flush=True)
        return receipt

    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        reports = list(executor.map(assess, selected))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'generation_assessment.json').write_text(json.dumps({'reports': reports, 'pending_generation': pending,
        'complete_census': not pending and all(r['returncode'] == 0 for r in reports)}, indent=2) + '\n')
    if any(r['returncode'] for r in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
