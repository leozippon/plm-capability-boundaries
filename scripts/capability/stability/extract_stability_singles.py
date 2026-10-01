#!/usr/bin/env python3
"""Score the frozen single-mutant stability cohort with one frozen checkpoint.

One forward per sequence at batch size one. Every sequence of a background is
scored under one loaded checkpoint and the cohort's sequences are already
deduplicated inside each background, so a wild type shared by all of its mutants
is scored once.

The measurement identity is the admitted Readout one, imported unchanged from
``src.capability.readouts.readout_extraction``: the native packing and scoring span of each
interface, the block output of the middle and final transformer block before any
separately applied final normalisation, and the arithmetic mean over
residue-bearing tokens together with the last residue-bearing token state.

Nothing here reads a measured stability value: the extraction plan carries
sequences, mutated positions and state indices only, and this entry point refuses
a plan that carries a measurement field.

``--keep-full-features`` additionally writes the unprojected per-state block
outputs, so a different projection or a different readout class can be refitted
from the frozen states without a second forward pass. The projected 256-coordinate
blocks the fit consumes are written either way.

Batch size one is the panel setting. The 0.001-nat and 0.001-relative-L2 gates
the Readout recipe fixes are therefore INAPPLICABLE rather than passed: at batch
size one the repeat check re-runs the identical single-row computation, so its
zero is structural. What the repeat column measures is run-to-run reproducibility
of that identical computation, which is a weaker quantity and is reported as such.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.context import context_homologue as ch
from src.capability.core.io import write_json
from src.capability.interactions.pairwise_epistasis import (
    ARM_DTYPE, FEATURE_BLOCKS, PRODUCTION_BATCH_SIZE, PROJECTION_DIM, PROJECTION_SEED,
    ROSTER, projection_matrices)
from src.capability.stability.stability_gate import validate_plan_background
from src.capability.position.position_terms import (
    PARTITION_SEMANTICS, ResidueCoverage, SEMANTICS as POSITION_SEMANTICS, blas_pinning,
    retention_residual)
from src.capability.readouts.readout_extraction import (
    extract_batch, feature_block_names, hooked_block_indices, load_readout_arm, pack_sequence,
    representation_blocks,
    representation_positions, text_boundary)

SCHEMA = 'stability_singles_extraction_v1'
#: Sequences per background re-extracted in an independent forward. At batch size
#: one this measures run-to-run reproducibility of the same computation, not
#: batch-composition invariance and not accuracy against a higher precision.
REPEAT_SEQUENCES = 3
MEASUREMENT_FIELDS = ('ddg', 'kcal', 'trypsin', 'chymotrypsin', 'combined')


def sha(path: Path) -> str:
    sha256 = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def plan_digest(plan: dict) -> str:
    return hashlib.sha256(json.dumps(
        {k: plan[k] for k in ('schema', 'cohort_sha256', 'projection', 'backgrounds')},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def token_record(arm, sequence: str, budget: int) -> dict:
    ids, scored, pooled = pack_sequence(arm, sequence)
    if len(ids) > budget:
        raise ValueError(f'packed target of {len(ids)} tokens exceeds the {budget}-position budget')
    selected = representation_positions(arm, [sequence], [(ids, scored, pooled)])
    positions = list(range(pooled[0], pooled[1])) if selected is None else list(selected[0])
    return {'ids': [int(value) for value in ids], 'scored_span': [int(scored[0]), int(scored[1])],
            'pooled_positions': [int(value) for value in positions]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--expect-plan-sha256', required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=PRODUCTION_BATCH_SIZE)
    parser.add_argument('--budget', type=int, default=1024)
    parser.add_argument('--background-limit', type=int, default=0,
                        help='throughput qualification only; a limited run never publishes a manifest')
    parser.add_argument('--extra-block-index', type=int, action='append',
                        help='hook a further zero-based transformer block, repeatable. The two '
                             'admitted blocks are always hooked and stay at feature positions 0 '
                             'to 3, so an archive with extra depths still presents the admitted '
                             'four where a reader of the admitted layout expects them. An index '
                             'outside the stack is refused rather than clamped, and duplicates '
                             'are dropped')
    parser.add_argument('--keep-position-terms', action='store_true',
                        help='also retain the per-scored-token log-likelihood vector whose sum '
                             'is the archived per-state scalar, with the residue coverage needed '
                             'to attribute it, so a mutation score can be decomposed over '
                             'positions without a second forward pass')
    parser.add_argument('--blas-threads', type=int,
                        help='the declared BLAS thread pool; the run is refused unless the '
                             'environment and torch both report it')
    parser.add_argument('--keep-full-features', action='store_true',
                        help='also write the full-width per-state block outputs to a separate '
                             'file, so a different projection or readout class can be refitted '
                             'without model inference')
    args = parser.parse_args()

    if args.arm not in ROSTER:
        raise SystemExit(f'{args.arm} is not on the frozen roster')
    if args.batch_size != PRODUCTION_BATCH_SIZE:
        raise SystemExit('the panel setting is singleton extraction; batch size one only')
    dtype = ARM_DTYPE.get(args.arm, 'float32')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    plan = json.loads(args.plan.read_bytes())
    if plan.get('schema') != SCHEMA:
        raise SystemExit('unexpected extraction plan schema')
    if plan_digest(plan) != args.expect_plan_sha256:
        raise SystemExit('extraction plan digest does not match the declared value')
    serialised = json.dumps(plan['backgrounds'])
    if any(field in serialised for field in MEASUREMENT_FIELDS):
        raise SystemExit('extraction plan carries measurement fields; refusing to score')
    for row in plan['backgrounds']:
        validate_plan_background(row)

    spec = importlib.util.spec_from_file_location(
        'stage46', ROOT / 'scripts/capability/stages/context_homologue.py')
    stage46 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage46)

    code_paths = [Path(__file__), ROOT / 'src/capability/stability/stability_gate.py',
                  ROOT / 'src/capability/interactions/pairwise_epistasis.py',
                  ROOT / 'src/capability/readouts/readout_extraction.py',
                  ROOT / 'src/capability/context/context_homologue.py',
                  ROOT / 'scripts/capability/stages/context_homologue.py',
                  ROOT / 'src/capability/core/arms.py']
    handle = load_readout_arm(args.arm, stage46, dtype=dtype)
    checkpoint = handle.spec.path
    if text_boundary(handle) is not None:
        code_paths.extend(ROOT / name for name in (
            'src/capability/models/text_aa_fitness.py', 'src/capability/models/text_aa_cohort.py',
            'src/capability/text_aa_boundaries.json', 'src/capability/core/precision_policy.py'))
    identity = {
        'schema': SCHEMA, 'arm': args.arm, 'dtype': dtype, 'batch_size': args.batch_size,
        'budget': args.budget, 'plan_sha256': args.expect_plan_sha256,
        'cohort_sha256': plan['cohort_sha256'],
        'projection': {'seed': PROJECTION_SEED, 'dim': PROJECTION_DIM, 'blocks': list(FEATURE_BLOCKS)},
        'checkpoint_path': str(checkpoint.resolve()),
        'checkpoint_tensor_files': [{'name': x.name, 'bytes': x.stat().st_size}
                                    for x in sorted(checkpoint.iterdir())
                                    if x.is_file() and x.suffix in ('.safetensors', '.bin', '.pt')],
        'checkpoint_metadata_sha256': {
            str(x.relative_to(checkpoint)): sha(x) for x in sorted(checkpoint.iterdir())
            if x.is_file() and (x.suffix == '.json' or x.name in ('tokenizer.model', 'tokenizer.json'))},
        'code_sha256': {str(x.relative_to(ROOT)): sha(x) for x in code_paths},
        'block_semantics': 'zero-based transformer block output, before final output normalization',
        'representation_semantics': ('absolute per-state pooled block outputs; residue-bearing token '
                                     'mean and last residue-bearing token; no EOS, BOS, context or padding'),
        'likelihood_semantics': 'native packed residue-span summed log likelihood, nats, per state',
        'precision_gate': ('inapplicable at batch size one: the repeat forward is the identical '
                           'single-row computation, so a zero difference is structural'),
    }
    pinning = blas_pinning(args.blas_threads) if args.keep_position_terms else None
    if args.keep_position_terms:
        identity['position_terms'] = {'semantics': POSITION_SEMANTICS,
                                      'partition': PARTITION_SEMANTICS,
                                      'retention_gate_nats': 0.0,
                                      'blas': pinning}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / f'manifest_{args.arm}.json'
    progress_path = args.out / f'progress_{args.arm}.json'
    resume_path = manifest_path if manifest_path.exists() else progress_path
    if resume_path.exists():
        prior = json.loads(resume_path.read_text())
        if prior['identity'] != identity:
            raise SystemExit('resume identity differs; use a fresh output directory')

    arm = load_readout_arm(args.arm, stage46, device=args.device, dtype=dtype)
    arm.model.eval().requires_grad_(False)
    if text_boundary(arm) is None:
        ch.require_position_budget(arm.model.config, arm=args.arm)
    blocks = representation_blocks(arm)
    block_indices = hooked_block_indices(len(blocks), args.extra_block_index)
    coverage = ResidueCoverage(arm) if args.keep_position_terms else None

    selected = plan['backgrounds']
    if args.background_limit:
        selected = selected[:args.background_limit]
    completed, projection, began = [], None, time.monotonic()
    for row in selected:
        name = row['name']
        filename = f"{args.arm}_{hashlib.sha256(name.encode()).hexdigest()[:20]}.npz"
        path = args.out / filename
        row_identity = {'identity': identity, 'background': name, 'group': row['group'],
                        'sequences': len(row['sequences']), 'variants': len(row['variants'])}
        if not path.exists():
            tokens = [token_record(arm, sequence, args.budget) for sequence in row['sequences']]
            features, likelihood = [], []
            retained, residuals = [], []
            for sequence in row['sequences']:
                bucket = [] if args.keep_position_terms else None
                block, score = extract_batch(arm, [sequence], stage46, block_indices,
                                             position_terms=bucket)
                features.append(block[0])
                likelihood.append(score[0])
                if bucket is not None:
                    record_terms = bucket[0]
                    counts, offset = coverage.counts(
                        record_terms['ids'], record_terms['span'], sequence)
                    residuals.append(retention_residual(
                        record_terms['terms'], -float(score[0]), arm.device))
                    retained.append((record_terms['terms'], counts, offset))
            features = np.stack(features).astype(np.float32)
            likelihood = np.asarray(likelihood, dtype=np.float64)
            if features.ndim != 3 or features.shape[1] != 2 * len(block_indices):
                raise ValueError(f'expected {2 * len(block_indices)} aligned block summaries '
                                 'per state, two per hooked block')
            if features.shape[1] < len(FEATURE_BLOCKS):
                raise ValueError('the admitted four block summaries must be present')
            if projection is None:
                projection = projection_matrices(features.shape[2])
            if features.shape[2] != projection[0].shape[0]:
                raise ValueError('hidden width changed within one checkpoint')
            # The projected archive stays exactly what it is today: the projection is
            # formed over the admitted four blocks alone, under the admitted seeds
            # 20260923-20260926, so a published cell reads a byte-identical array
            # whether or not extra depths were hooked. Extra depths are retained at
            # full width only, which is what a depth-resolved refit reads anyway --
            # it forms its own projection under its own declared seed rule -- and
            # inventing projection seeds for them here would declare a contract
            # nothing has declared.
            projected = np.stack([features[:, i] @ projection[i]
                                  for i in range(len(FEATURE_BLOCKS))], axis=1)
            repeat_likelihood, repeat_projected = [], []
            for sequence in row['sequences'][:REPEAT_SEQUENCES]:
                block, score = extract_batch(arm, [sequence], stage46, block_indices)
                repeat_likelihood.append(score[0])
                repeat_projected.append(np.stack(
                    [block[0, i].astype(np.float32) @ projection[i]
                     for i in range(len(FEATURE_BLOCKS))]))
            repeat = min(REPEAT_SEQUENCES, len(row['sequences']))
            likelihood_repeat_nats = float(np.max(np.abs(
                np.asarray(repeat_likelihood) - likelihood[:repeat])))
            reference_norm = float(np.linalg.norm(np.stack(repeat_projected)))
            feature_repeat_relative_l2 = float(
                np.linalg.norm(np.stack(repeat_projected) - projected[:repeat])
                / reference_norm) if reference_norm > 0 else 0.0
            payload = {
                'projected': projected.astype(np.float32),
                'likelihood': likelihood,
                'pooled_token_counts': np.asarray([len(t['pooled_positions']) for t in tokens],
                                                  dtype=np.int64),
                'scored_token_counts': np.asarray(
                    [t['scored_span'][1] - t['scored_span'][0] for t in tokens], dtype=np.int64),
                'packed_token_counts': np.asarray([len(t['ids']) for t in tokens], dtype=np.int64),
                'token_ids': np.asarray([i for t in tokens for i in t['ids']], dtype=np.int64),
                'token_offsets': np.cumsum([0] + [len(t['ids']) for t in tokens]).astype(np.int64),
                'variant_states': np.asarray([v['state'] for v in row['variants']], dtype=np.int64),
                'variant_positions': np.asarray([v['position'] for v in row['variants']],
                                                dtype=np.int64),
                'hidden_width': np.asarray(features.shape[2], dtype=np.int64),
                'repeat_likelihood_nats': np.asarray(likelihood_repeat_nats),
                'repeat_feature_relative_l2': np.asarray(feature_repeat_relative_l2),
                'metadata': json.dumps(row_identity),
            }
            if args.keep_position_terms:
                worst = float(np.max(np.abs(np.asarray(residuals, dtype=np.float64))))
                if worst != 0.0:
                    raise ValueError(
                        f'{name}: the retained per-position vector does not re-reduce to its '
                        f'own archived scalar (max |sum - likelihood| = {worst!r} nats); this '
                        'is a retention defect and the cell writes no manifest'
                    )
                payload.update({
                    'position_nats': np.concatenate([t for t, _, _ in retained]).astype(np.float32),
                    'position_offsets': np.cumsum(
                        [0] + [len(t) for t, _, _ in retained]).astype(np.int64),
                    'position_residue_counts': np.concatenate(
                        [c for _, c, _ in retained]).astype(np.int16),
                    'position_residue_offsets': np.asarray(
                        [o for _, _, o in retained], dtype=np.int64),
                    'position_sum_check_nats': np.asarray(worst),
                })
            temp = path.with_suffix('.tmp')
            with temp.open('wb') as stream:
                np.savez_compressed(stream, **payload)
            temp.replace(path)
            if args.keep_full_features:
                full = args.out / f'full_{filename}'
                temp = full.with_suffix('.tmp')
                # One array per hooked block whenever more than the admitted pair is
                # hooked, and the single admitted array otherwise. The layout is
                # load-bearing rather than incidental: an every-block sweep of these
                # cohorts is about 720 GiB of states, and `np.load` on an npz reads a
                # named array whole, so a fit over them can stream one depth at a time
                # only if each depth is its own array. The default -- no extra depths --
                # writes exactly the array it always wrote, byte for byte, because 33
                # arms of completed extraction and every published cell read that key.
                if len(block_indices) > len(FEATURE_BLOCKS) // 2:
                    arrays = {f'features_d{index:03d}': features[:, 2 * position:2 * position + 2]
                              for position, index in enumerate(block_indices)}
                else:
                    arrays = {'features': features}
                with temp.open('wb') as stream:
                    np.savez(stream, **arrays, metadata=json.dumps(row_identity))
                temp.replace(full)
            print(f'{args.arm} {name}: repeat_delta_M_nats={likelihood_repeat_nats}, '
                  f'repeat_relative_l2={feature_repeat_relative_l2}', flush=True)
        with np.load(path, allow_pickle=False) as saved:
            if json.loads(str(saved['metadata'])) != row_identity:
                raise SystemExit(f'resume artifact identity mismatch: {path}')
            width = int(saved['hidden_width'])
            repeats = (float(saved['repeat_likelihood_nats']),
                       float(saved['repeat_feature_relative_l2']))
        completed.append({'background': name, 'group': row['group'], 'file': filename,
                          'sha256': sha(path), 'sequences': len(row['sequences']),
                          'variants': len(row['variants']), 'hidden_width': width,
                          'repeat_likelihood_nats': repeats[0],
                          'repeat_feature_relative_l2': repeats[1]})
        record = {'identity': identity, 'block_indices': list(block_indices),
                  'feature_blocks': feature_block_names(block_indices), 'backgrounds': completed,
                  'status': 'complete' if len(completed) == len(plan['backgrounds']) else 'running',
                  'background_limit': args.background_limit,
                  'updated_utc': datetime.now(timezone.utc).isoformat(),
                  'elapsed_seconds': time.monotonic() - began,
                  'sequences_scored': sum(r['sequences'] for r in completed),
                  'gpu': (torch.cuda.get_device_name(arm.device)
                          if str(arm.device).startswith('cuda') else 'cpu'),
                  'peak_allocated_bytes': (torch.cuda.max_memory_allocated(arm.device)
                                           if str(arm.device).startswith('cuda') else 0),
                  'torch_version': torch.__version__,
                  'blas': pinning}
        write_json(progress_path, record)
        if record['status'] == 'complete':
            write_json(manifest_path, record)
        print(f"{args.arm}: {len(completed)}/{len(selected)} {name} "
              f"({record['sequences_scored']} sequences, {record['elapsed_seconds']:.1f}s)",
              flush=True)


if __name__ == '__main__':
    main()
