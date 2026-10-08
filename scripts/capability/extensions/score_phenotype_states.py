#!/usr/bin/env python3
"""Score the planned phenotype states with one or more frozen checkpoints.

The measurement is the admitted one, imported unchanged: each state is packed in
its checkpoint's own native rendering, one forward per state at batch size one,
and the native next-token negative log likelihood is reduced over the interface's
scored span. That is the same code path the stability likelihood-only extraction
uses, so a likelihood produced here and one produced there are the same quantity.

What this stage does *not* do matters as much. It reads no phenotype label: the
plan carries deduplicated sequences and nothing else, and a plan that carries a
measurement field is refused. It scores a sequence once however many cohorts and
however many rows reference it, because a likelihood is a property of the state.
It never writes a partial arm: a nonfinite score, a state over the packed budget
or an unknown arm stops that arm, and the completion record names only the arms
that finished.

Each arm also re-scores a few states in an independent forward. At batch size one
that repeat is the identical single-row computation, so its agreement measures
run-to-run reproducibility of that computation and nothing stronger; it is
reported as exactly that.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.extensions.phenotype_breadth import PLAN_SCHEMA  # noqa: E402
from src.capability.interactions.pairwise_epistasis import (  # noqa: E402
    ARM_DTYPE, PRODUCTION_BATCH_SIZE, ROSTER)
from src.capability.readouts.readout_extraction import (  # noqa: E402
    forward_readout_rows, load_readout_arm, pack_sequence)

COMPLETION = 'phenotype_scoring.json'
SCHEMA = 'phenotype_state_likelihood_v1'
#: States re-scored per arm in an independent forward (see the module docstring).
REPEAT_STATES = 3
#: A plan carrying any of these keys is a label table, not a state plan.
FORBIDDEN_PLAN_FIELDS = ('label', 'labels', 'score', 'scores', 'measurement_value')


def resources(device: str) -> dict:
    record = {'executable': sys.executable, 'device': device, 'cpus': os.cpu_count(),
              'disk_free_bytes': shutil.disk_usage(ROOT).free,
              'memory': [line for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith(('MemTotal:', 'MemAvailable:'))]}
    try:
        import torch
        if torch.cuda.is_available() and device.startswith('cuda'):
            index = int(device.split(':')[1]) if ':' in device else 0
            free, total = torch.cuda.mem_get_info(index)
            record['gpu'] = {'name': torch.cuda.get_device_name(index),
                             'free_bytes': int(free), 'total_bytes': int(total),
                             'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
        record['torch'] = torch.__version__
    except Exception as error:  # a receipt must never be the reason a run dies
        record['gpu_probe_error'] = repr(error)
    return record


def load_stage46():
    """The admitted native scoring doors, loaded from the stage that owns them."""
    spec = importlib.util.spec_from_file_location(
        'stage46', ROOT / 'scripts/capability/stages/context_homologue.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_plan(path: Path, expected: str | None) -> dict:
    plan = json.loads(Path(path).read_text(encoding='utf-8'))
    if plan.get('schema') != PLAN_SCHEMA:
        raise SystemExit(f'{path}: not a {PLAN_SCHEMA} state plan')
    digest = sha256_file(Path(path))
    if expected is not None and digest != expected:
        raise SystemExit(f'{path}: plan sha256 {digest} does not match the expected {expected}')
    for state in plan['states']:
        if any(field in state for field in FORBIDDEN_PLAN_FIELDS):
            raise SystemExit(f'{path}: a state carries a measurement field; this stage is label-blind')
        if hashlib.sha256(state['sequence'].encode('utf-8')).hexdigest() != state['sequence_sha256']:
            raise SystemExit(f"{path}: state {state['index']} hash disagrees with its sequence")
    indices = [state['index'] for state in plan['states']]
    if indices != list(range(len(indices))):
        raise SystemExit(f'{path}: state indices are not a dense ordered range')
    plan['plan_sha256'] = digest
    return plan


def likelihood(arm, sequence: str, stage46, budget: int) -> tuple[float, int]:
    """The native forward and reducer, with no representation hooks."""
    ids, scored, _ = pack_sequence(arm, sequence)
    if len(ids) > budget:
        raise ValueError(f'packed state of {len(ids)} tokens exceeds the {budget}-position budget')
    logits, packed = forward_readout_rows(arm, [ids], stage46)
    value = -stage46._target_nll(logits, packed, *scored)['nll_sum']
    if not np.isfinite(value):
        raise ValueError('nonfinite native likelihood')
    return float(value), len(ids)


def score_arm(name: str, states: list[dict], stage46, args) -> dict:
    """Every planned state under one loaded checkpoint, in plan order."""
    dtype = args.dtype or ARM_DTYPE.get(name, 'float32')
    began = time.monotonic()
    arm = load_readout_arm(name, stage46, device=args.device, dtype=dtype)
    values = np.empty(len(states), dtype=np.float64)
    tokens = np.empty(len(states), dtype=np.int64)
    for position, state in enumerate(states):
        values[position], tokens[position] = likelihood(
            arm, state['sequence'], stage46, args.budget)
        if args.progress and position and position % args.progress == 0:
            print(f'{name}: {position}/{len(states)} states '
                  f'{time.monotonic() - began:.0f}s', flush=True)
    repeats = []
    for position, state in enumerate(states[:REPEAT_STATES]):
        again, _ = likelihood(arm, state['sequence'], stage46, args.budget)
        repeats.append({'state': state['index'],
                        'absolute_difference_nats': abs(again - values[position])})
    hashes = np.asarray([state['sequence_sha256'] for state in states], dtype='<U64')
    archive = args.out / f'{name}-states.npz'
    with archive.open('wb') as handle:
        np.savez_compressed(handle, state_index=np.asarray([s['index'] for s in states]),
                            sequence_sha256=hashes, likelihood_nats=values,
                            packed_tokens=tokens)
    return {'arm': name, 'dtype': dtype, 'states': len(states),
            'seconds': time.monotonic() - began,
            'archive': archive.name, 'archive_sha256': sha256_file(archive),
            'likelihood_nats': {'min': float(values.min()), 'max': float(values.max()),
                                'mean': float(values.mean())},
            'packed_tokens': {'min': int(tokens.min()), 'max': int(tokens.max())},
            'repeat_forward': {'states': len(repeats), 'records': repeats,
                               'max_absolute_difference_nats': max(
                                   (r['absolute_difference_nats'] for r in repeats), default=0.0),
                               'scope': ('run-to-run reproducibility of the identical single-row '
                                         'computation; not batch-composition invariance and not '
                                         'accuracy against a higher precision')},
            'serving_provenance': sorted((getattr(arm, 'serving_provenance', None) or {}))}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--plan', type=Path, required=True, help='state plan to score')
    parser.add_argument('--expect-plan-sha256', default=None,
                        help='refuse a plan whose bytes are not these')
    parser.add_argument('--arms', required=True,
                        help='comma-separated checkpoint names, scored in the order given')
    parser.add_argument('--out', type=Path, required=True, help='fresh output directory')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', default=None,
                        help='override the per-arm declared dtype; normally omitted')
    parser.add_argument('--batch-size', type=int, default=PRODUCTION_BATCH_SIZE,
                        help='panel setting is one forward per state')
    parser.add_argument('--budget', type=int, default=1024)
    parser.add_argument('--state-limit', type=int, default=0,
                        help='score only the first N planned states; for interface checks only')
    parser.add_argument('--progress', type=int, default=500,
                        help='print a progress line every N states; 0 silences it')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.batch_size != PRODUCTION_BATCH_SIZE:
        raise SystemExit('this measurement is declared at batch size one; no other size is admitted')
    arms = [name.strip() for name in args.arms.split(',') if name.strip()]
    unknown = [name for name in arms if name not in ROSTER]
    if unknown:
        raise SystemExit(f'not panel checkpoints: {unknown}')
    if len(set(arms)) != len(arms):
        raise SystemExit('duplicate arm requested')
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f'refusing an existing output directory: {out}')
    out.mkdir(parents=True)
    args.out = out
    plan = read_plan(args.plan, args.expect_plan_sha256)
    states = plan['states'][:args.state_limit] if args.state_limit else plan['states']
    started = time.monotonic()
    write_json(out / 'resource-start.json',
               {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'resources': resources(args.device), 'arms': arms, 'states': len(states),
                'plan_sha256': plan['plan_sha256'], 'argv': sys.argv[1:]})
    stage46 = load_stage46()
    completed = []
    for name in arms:
        record = score_arm(name, states, stage46, args)
        completed.append(record)
        write_json(out / f'{name}-states.json', record)
        print(json.dumps({'arm': name, 'states': record['states'],
                          'seconds': round(record['seconds'], 1),
                          'repeat_max_abs_nats': record['repeat_forward'][
                              'max_absolute_difference_nats']}), flush=True)
    write_json(out / 'resource-end.json',
               {'status': 'complete', 'resources': resources(args.device),
                'seconds': time.monotonic() - started})
    write_json(out / COMPLETION, {
        'schema': SCHEMA, 'stage': 'score_phenotype_states', 'status': 'complete',
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'seconds': time.monotonic() - started,
        'plan': str(args.plan), 'plan_sha256': plan['plan_sha256'],
        'states_scored': len(states), 'states_planned': len(plan['states']),
        'state_limit': args.state_limit, 'partial_plan': bool(args.state_limit),
        'batch_size': args.batch_size, 'budget': args.budget, 'device': args.device,
        'measurement': plan['measurement'], 'arms': completed,
        'resources': resources(args.device)})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
