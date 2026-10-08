#!/usr/bin/env python3
"""Score several frozen checkpoints' native stability likelihoods in one cell.

This is a runner, not a second extractor.  Every forward pass, every archive and
every identity record is produced by ``extract_stability_singles.py
--likelihood-only``, which is invoked once per named arm with the identical plan
and the identical declared plan digest; nothing here touches a model, a
tokenizer or a score.

It exists because the campaign queue's slot is a global barrier: one arm per cell
would turn a 33-arm panel into seventeen barriers that every other lane in the
wave would have to wait behind, and the arms differ by more than an order of
magnitude in cost, so the faster card would idle at each one.  Grouping a
balanced set of arms into one cell keeps the allocation busy and the barrier
count small.

All of a cell's arms write into the one output directory the queue injects.  The
archives and manifests are arm-keyed by name, so they cannot collide, and the
fit stage discovers an arm by its manifest wherever the cell happened to land.
A failing arm aborts the cell with its own non-zero status and no completion
record is written, because a cell that scored some of its arms is not a cell that
can be read as done.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.core.io import sha256_file, write_json
from src.capability.interactions.pairwise_epistasis import ROSTER
from src.capability.stability import gate_inputs

#: The extractor this runner drives, and the archive schema it must produce.
EXTRACTOR = Path(__file__).resolve().parent / 'extract_stability_singles.py'
LIKELIHOOD_SCHEMA = 'stability_singles_likelihood_extraction_v1'

#: Written last, and only when every declared arm of the cell is complete.
COMPLETION = 'completion.json'


def declared_arms(text: str) -> list[str]:
    arms = [item.strip() for item in text.split(',') if item.strip()]
    if not arms:
        raise argparse.ArgumentTypeError('--arms needs at least one arm')
    unknown = [arm for arm in arms if arm not in ROSTER]
    if unknown:
        raise argparse.ArgumentTypeError(f'not on the frozen roster: {unknown}')
    if len(set(arms)) != len(arms):
        raise argparse.ArgumentTypeError('an arm is declared twice in one cell')
    return arms


def score_arm(arm: str, plan: Path, args) -> None:
    command = [sys.executable, str(EXTRACTOR),
               '--plan', str(plan),
               '--expect-plan-sha256', args.expect_plan_sha256,
               '--arm', arm, '--likelihood-only',
               '--device', args.device, '--out', str(args.out)]
    print(f'[wave] {arm}: {" ".join(command)}', flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(f'{arm}: extraction exited {completed.returncode}; the cell '
                         'writes no completion record')


def accept_arm(arm: str, out: Path, plan_digest: str) -> dict:
    """Re-read the manifest the extractor published and bind it to this cell."""

    path = out / f'manifest_{arm}.json'
    if not path.exists():
        raise SystemExit(f'{arm}: extraction exited zero without publishing a manifest')
    manifest = json.loads(path.read_text())
    identity = manifest['identity']
    if (manifest['status'] != 'complete' or identity['arm'] != arm
            or identity['schema'] != LIKELIHOOD_SCHEMA
            or identity['plan_sha256'] != plan_digest):
        raise SystemExit(f'{arm}: manifest is not a complete likelihood-only record of this '
                         'plan; refusing to mark the cell done')
    archives = {row['background']: row['sha256'] for row in manifest['backgrounds']}
    for row in manifest['backgrounds']:
        archive = out / row['file']
        if sha256_file(archive) != row['sha256']:
            raise SystemExit(f"{arm}: archive {row['file']} does not match its manifest digest")
    return {'arm': arm, 'manifest': path.name, 'manifest_sha256': sha256_file(path),
            'cohort_sha256': identity['cohort_sha256'], 'dtype': identity['dtype'],
            'backgrounds': len(archives),
            'sequences_scored': manifest['sequences_scored'],
            'elapsed_seconds': manifest['elapsed_seconds'],
            'repeat_likelihood_nats_max': max(row['repeat_likelihood_nats']
                                              for row in manifest['backgrounds']),
            'checkpoint_path': identity['checkpoint_path'],
            'gpu': manifest['gpu']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate-dir', type=Path, action='append', required=True,
                        help='a candidate frozen stability gate directory holding the '
                             'extraction plan; repeatable, because two directory layouts '
                             'of that measurement are in use')
    parser.add_argument('--expect-plan-sha256', required=True,
                        help='the declared plan digest, passed through unchanged')
    parser.add_argument('--arms', type=declared_arms, required=True,
                        help='comma-separated roster arms, scored in the order given')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    if not EXTRACTOR.exists():
        raise SystemExit(f'extractor not found at {EXTRACTOR}')
    plan = gate_inputs.resolve(args.gate_dir, ('extraction_plan.json',))['extraction_plan.json']
    args.out.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    accepted: list[dict] = []
    for arm in args.arms:
        if (args.out / f'manifest_{arm}.json').exists():
            print(f'[wave] {arm}: manifest already present, re-reading rather than rescoring',
                  flush=True)
        else:
            score_arm(arm, plan, args)
        accepted.append(accept_arm(arm, args.out, args.expect_plan_sha256))
        print(f'[wave] {len(accepted)}/{len(args.arms)} accepted '
              f'({time.monotonic() - began:.1f}s)', flush=True)

    digests = {record['cohort_sha256'] for record in accepted}
    if len(digests) != 1:
        raise SystemExit(f'the cell mixes cohorts: {sorted(digests)}')
    write_json(args.out / COMPLETION, {
        'schema': 'stability_likelihood_wave_v1',
        'status': 'complete',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'plan': str(plan),
        'plan_sha256': args.expect_plan_sha256,
        'cohort_sha256': digests.pop(),
        'device': args.device,
        'arms': list(args.arms),
        'extractor_sha256': sha256_file(EXTRACTOR),
        'runner_sha256': sha256_file(Path(__file__)),
        'records': accepted,
        'elapsed_seconds': time.monotonic() - began,
        'semantics': ('native packed residue-span summed log likelihood per state, nats; '
                      'no representations, projections or position terms'),
    })
    print(json.dumps({'arms': list(args.arms), 'accepted': len(accepted),
                      'seconds': round(time.monotonic() - began, 1)}), flush=True)


if __name__ == '__main__':
    main()
