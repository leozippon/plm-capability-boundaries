#!/usr/bin/env python3
"""Extract four frozen block summaries and native likelihood on a fixed DMS draw."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.capability.context import context_homologue as ch
from src.capability.core.io import write_json
from src.capability.position.position_terms import (PARTITION_SEMANTICS, ResidueCoverage,
    SEMANTICS as POSITION_SEMANTICS, blas_pinning, retention_residual)
from src.capability.readouts.readout_extraction import FEATURE_NAMES, extract_batch, pack_sequence, representation_blocks, mutation_relative_drift, load_readout_arm, text_boundary, text_ids, TEXT_EXTRA_ARM, validate_ec_conditioning, bind_readout_ec


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eligible_max_tokens(arm, sequences, budget):
    """Count native packed tokens; reject as soon as any full target cannot fit."""
    boundary = text_boundary(arm)
    prefix_length = 0 if boundary is not None or getattr(arm, 'name', None) == 'zymctrl' else len(ch.row_prefix_ids(arm))
    maximum = 0
    for sequence in sequences:
        if getattr(arm, 'name', None) == 'zymctrl':
            length = len(pack_sequence(arm, sequence)[0])
        else:
            length = len(text_ids(arm, sequence)) if boundary is not None else prefix_length + len(ch.item_ids(arm, sequence, modality='protein'))
        if length > budget:
            return None
        maximum = max(maximum, length)
    return maximum


def publish_progress(manifest_path, progress_path, payload, *, expected_assays):
    """The queue completion artifact exists only after every selected assay."""
    count = len(payload['assays'])
    if expected_assays < 1 or count > expected_assays:
        raise ValueError('Invalid extraction completion count')
    payload = dict(payload, status='complete' if count == expected_assays else 'running')
    write_json(progress_path, payload)
    if count == expected_assays:
        write_json(manifest_path, payload)


def partition_assays(universe, shard_count):
    """Deterministic longest-first balancing by packed-token work, without labels."""
    if not 1 <= shard_count <= len(universe) or len({r['assay'] for r in universe}) != len(universe):
        raise ValueError('Invalid shard count or duplicate assay universe')
    loads = [0] * shard_count
    assignment = {}
    for row in sorted(universe, key=lambda r: (-r['max_packed_tokens'] * (r['variants'] + 4), r['assay'])):
        index = min(range(shard_count), key=lambda i: (loads[i], i))
        assignment[row['assay']] = index
        loads[index] += row['max_packed_tokens'] * (row['variants'] + 4)
    return [[r['assay'] for r in universe if assignment[r['assay']] == i] for i in range(shard_count)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--conditioning-map', type=Path, help='Hash-bound exact-WT EC annotation sidecar, required only for ZymCTRL')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--dtype', default='float32')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--budget', type=int, default=1024)
    p.add_argument('--assay-limit', type=int, default=0)
    p.add_argument('--smoke-variants', type=int, default=0)
    p.add_argument('--smoke-length-strata', action='store_true')
    p.add_argument('--smoke-assay', action='append', default=[], help='Explicit eligible assay for a targeted smoke; repeat as needed')
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--keep-position-terms', action='store_true',
                   help='also retain the per-scored-token log-likelihood vector whose sum is the '
                        'archived per-state scalar, with the residue coverage needed to attribute it')
    p.add_argument('--blas-threads', type=int,
                   help='the declared BLAS thread pool; the run is refused unless the environment '
                        'and torch both report it')
    p.add_argument('--max-score-drift', type=float)
    p.add_argument('--max-feature-drift', type=float)
    args = p.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Invalid shard index/count')
    if args.batch_size < 1 or args.budget != 1024:
        raise ValueError('Positive batch size and frozen 1024-position budget required')
    if not args.smoke_variants and (args.max_score_drift is None or args.max_feature_drift is None):
        raise ValueError('Full extraction requires explicit precision limits after smoke qualification')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = importlib.util.spec_from_file_location('stage46', ROOT/'scripts/capability/stages/context_homologue.py')
    stage46 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage46)
    cohort = json.loads(args.cohort.read_text())
    if (args.arm == 'zymctrl') != (args.conditioning_map is not None):
        raise ValueError('--conditioning-map is required exactly for ZymCTRL')
    ec_by_assay = {}
    if args.conditioning_map is not None:
        conditioning_bytes = args.conditioning_map.read_bytes()
        conditioning_sha256 = hashlib.sha256(conditioning_bytes).hexdigest()
        conditioning_payload = json.loads(conditioning_bytes)
        ec_by_assay = validate_ec_conditioning(conditioning_payload, cohort, sha(args.cohort))
    full_sequences = {row['assay']: [row['wildtype']]+list(row['sequences']) for row in cohort['assays']}
    if args.smoke_variants:
        if not args.assay_limit or args.smoke_variants < 2:
            raise ValueError('Smoke variants requires assay limit and at least two variants')
        for row in cohort['assays']:
            for key in ('sequences', 'mutants', 'measured', 'profile_scores'):
                row[key] = row[key][:args.smoke_variants]
            row['mutant_digest'] = hashlib.sha256('\n'.join(row['mutants']).encode()).hexdigest()
    handle = load_readout_arm(args.arm, stage46, dtype=args.dtype)
    selected, skipped, token_lengths = [], [], {}
    for row in cohort['assays']:
        if args.arm == 'zymctrl':
            if row['assay'] not in ec_by_assay:
                skipped.append(dict(assay=row['assay'], reason='no exact-WT full EC annotation'))
                continue
            bind_readout_ec(handle, ec_by_assay[row['assay']])
        maximum = eligible_max_tokens(handle, full_sequences[row['assay']], args.budget)
        if maximum is None:
            skipped.append(dict(assay=row['assay'], reason='full target exceeds 1024 positions'))
        else:
            selected.append(row)
            token_lengths[row['assay']] = maximum
    if args.smoke_length_strata:
        if not args.smoke_variants or args.assay_limit != 3 or len(selected) < 3:
            raise ValueError('Length-strata smoke requires smoke variants, assay limit3, and3 eligible assays')
        ordered = sorted(selected, key=lambda row: (token_lengths[row['assay']], row['assay']))
        selected = [ordered[0], ordered[len(ordered)//2], ordered[-1]]
    if args.smoke_assay:
        if (not args.smoke_variants or args.smoke_length_strata or args.assay_limit != len(args.smoke_assay)
            or len(set(args.smoke_assay)) != len(args.smoke_assay)):
            raise ValueError('Targeted smoke requires unique assays, matching assay limit, and smoke variants')
        eligible = {r['assay']: r for r in selected}
        if not set(args.smoke_assay) <= set(eligible):
            raise ValueError('Requested smoke assay is not eligible')
        selected = [r for r in selected if r['assay'] in set(args.smoke_assay)]
    if args.assay_limit:
        selected = selected[:args.assay_limit]
    if not selected:
        raise ValueError('No eligible assays')
    partition = None
    if args.shard_count > 1:
        universe = [dict(assay=r['assay'], max_packed_tokens=token_lengths[r['assay']], variants=len(r['mutants'])) for r in selected]
        assigned = partition_assays(universe, args.shard_count)[args.shard_index]
        partition = dict(method='greedy_packed_token_work_v1', count=args.shard_count,
                         index=args.shard_index, universe=universe, assigned_assays=assigned)
        selected = [r for r in selected if r['assay'] in set(assigned)]
    args.out.mkdir(parents=True, exist_ok=True)
    code_paths = [Path(__file__), ROOT/'src/capability/readouts/readout_extraction.py', ROOT/'src/capability/context/context_homologue.py', ROOT/'scripts/capability/stages/context_homologue.py', ROOT/'src/capability/core/arms.py', ROOT/'src/capability/models/proteinglm.py', ROOT/'src/capability/models/progen3.py', ROOT/'src/capability/models/joint_lineage.py']
    if text_boundary(handle) is not None:
        code_paths.extend(ROOT/name for name in ('src/capability/models/text_aa_fitness.py', 'src/capability/models/text_aa_cohort.py', 'src/capability/text_aa_boundaries.json', 'src/capability/core/precision_policy.py'))
    checkpoint = handle.spec.path
    checkpoint_metadata = {str(x.relative_to(checkpoint)): sha(x) for x in sorted(checkpoint.iterdir()) if x.is_file() and (x.suffix == '.json' or x.name in ('tokenizer.model', 'tokenizer.json'))}
    identity = dict(schema_version='frozen_readout_v1', arm=args.arm, dtype=args.dtype, batch_size=args.batch_size,
        checkpoint_path=str(checkpoint.resolve()),
        checkpoint_tensor_files=[dict(name=x.name, bytes=x.stat().st_size) for x in sorted(checkpoint.iterdir()) if x.is_file() and x.suffix in ('.safetensors', '.bin', '.pt')],
        checkpoint_metadata_sha256=checkpoint_metadata, cohort_sha256=sha(args.cohort), code_sha256={str(x.relative_to(ROOT)):sha(x) for x in code_paths},
        budget=args.budget, smoke_variants=args.smoke_variants, smoke_length_strata=args.smoke_length_strata, max_score_drift=args.max_score_drift, max_feature_drift=args.max_feature_drift, feature_names=list(FEATURE_NAMES), assay_limit=args.assay_limit,
        block_semantics='zero-based transformer block output, before final output normalization',
        representation_semantics='mutant-minus-WT; residue-bearing token mean and last token; no EOS or padding',
        scoring_stratum='target_only_native_packed_residue_span')
    pinning = blas_pinning(args.blas_threads) if args.keep_position_terms else None
    if args.keep_position_terms:
        identity['position_terms'] = dict(semantics=POSITION_SEMANTICS, partition=PARTITION_SEMANTICS,
                                         retention_gate_nats=0.0, blas=pinning)
    if args.smoke_assay:
        identity['smoke_assays'] = args.smoke_assay
    if text_boundary(handle) is not None:
        identity['scoring_stratum'] = 'literal_AA_native_text_token_sum'
        identity['text_aa_boundary'] = json.loads(json.dumps(asdict(text_boundary(handle))))
        identity['scoring_target_rule'] = 'tokens2..L; first byte is unscored context' if text_boundary(handle).conditioning_id is None else 'all AA-bearing tokens after the document prefix'
        identity['representation_semantics'] = 'mutant-minus-WT; mean and last literal AA-bearing tokens, excluding document prefix; byte-zero included for no-prefix ByGPT5; BPE token means are not per-residue means'
        identity['research_role'] = 'unprompted instruction-tuned literal-AA string readout; not a chat-template evaluation' if args.arm == TEXT_EXTRA_ARM else 'literal-AA text string readout'
    if args.conditioning_map is not None:
        identity['conditioning_map_sha256'] = conditioning_sha256
        identity['conditioning_annotation_source'] = conditioning_payload['annotation_source']
        identity['conditioning_selection_rule'] = conditioning_payload['selection_rule']
        identity['scoring_stratum'] = 'exact_WT_EC_conditioned_native_residue_span'
        identity['representation_semantics'] = 'mutant-minus-WT; mean and last residue states strictly between native start/end markers; same genuine WT EC for every variant'
    if args.arm == 'protgpt2':
        identity['scoring_stratum'] = 'native_fasta_target_tokens_including_formatting_newlines'
        identity['representation_semantics'] = 'mutant-minus-WT; mean and last residue-bearing native FASTA tokens; exclude pure newline tokens; retain mixed residue/newline BPE tokens'
    manifest_path = args.out/f'manifest_{args.arm}.json'
    progress_path = args.out/f'progress_{args.arm}.json'
    if partition is not None:
        manifest_path = args.out/f'shard_{args.arm}_{args.shard_index}.json'
        progress_path = args.out/f'progress_{args.arm}_shard_{args.shard_index}.json'
    resume_path = manifest_path if manifest_path.exists() else progress_path
    if resume_path.exists():
        prior = json.loads(resume_path.read_text())
        if resume_path == manifest_path and prior.get('status') != 'complete':
            raise ValueError('Legacy incomplete completion manifest; use a fresh output directory')
        if prior.get('execution_partition') != partition:
            raise ValueError('Resume execution partition differs; use a fresh output directory')
        if prior['identity'] != identity:
            raise ValueError('Resume identity differs; use a fresh output directory')
    arm = load_readout_arm(args.arm, stage46, device=args.device, dtype=args.dtype)
    arm.model.eval().requires_grad_(False)
    if text_boundary(arm) is None:
        ch.require_position_budget(arm.model.config, arm=args.arm)
    block_count = len(representation_blocks(arm))
    coverage = ResidueCoverage(arm) if args.keep_position_terms else None
    completed = []
    began = time.monotonic()
    for row in selected:
        if args.arm == 'zymctrl':
            bind_readout_ec(arm, ec_by_assay[row['assay']])
        filename = f"{args.arm}_{hashlib.sha256(row['assay'].encode()).hexdigest()[:20]}.npz"
        path = args.out/filename
        row_identity = dict(identity=identity, assay=row['assay'], mutant_digest=row['mutant_digest'])
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if json.loads(str(saved['metadata'])) != row_identity:
                    raise ValueError(f'Resume artifact identity mismatch: {path}')
        else:
            strings = [row['wildtype']]+row['sequences']
            features, scores = [], []
            bucket = [] if args.keep_position_terms else None
            for start in range(0, len(strings), args.batch_size):
                f,s = extract_batch(arm, strings[start:start+args.batch_size], stage46,
                                    position_terms=bucket)
                features.append(f)
                scores.append(s)
            f, s = np.concatenate(features), np.concatenate(scores)
            # Independent single-row repeat checks batch-induced likelihood and representation drift.
            check_f, check_s = [], []
            for sequence in strings[:3]:
                a,b = extract_batch(arm, [sequence], stage46)
                check_f.append(a[0]); check_s.append(b[0])
            check_f, check_s = np.stack(check_f), np.array(check_s)
            delta_error = np.max(np.abs((s[1:3]-s[0])-(check_s[1:]-check_s[0])))
            feature_error = float(np.linalg.norm(f[:3]-check_f)/max(np.linalg.norm(check_f), 1e-12))
            mutation_feature_error = mutation_relative_drift(f[:3], check_f)
            print(f'precision {row["assay"]}: delta_M_nats={delta_error}, state_relative_l2={feature_error}, mutation_relative_l2={mutation_feature_error}', flush=True)
            if ((args.max_score_drift is not None and delta_error > args.max_score_drift) or
                (args.max_feature_drift is not None and mutation_feature_error > args.max_feature_drift)):
                raise RuntimeError('Declared precision limit exceeded; see preceding metrics')
            retention = {}
            if bucket is not None:
                if len(bucket) != len(strings):
                    raise ValueError('retained position vectors do not cover every state once')
                retained, residuals = [], []
                for index, (sequence, entry) in enumerate(zip(strings, bucket)):
                    counts, offset = coverage.counts(entry['ids'], entry['span'], sequence)
                    residuals.append(retention_residual(entry['terms'], -float(s[index]), arm.device))
                    retained.append((entry['terms'], counts, offset))
                worst = float(np.max(np.abs(np.asarray(residuals, dtype=np.float64))))
                if worst != 0.0:
                    raise RuntimeError(
                        f"{row['assay']}: the retained per-position vector does not re-reduce to "
                        f'its own archived scalar (max |sum - likelihood| = {worst!r} nats); this '
                        'is a retention defect and the cell writes no manifest')
                retention = dict(
                    position_nats=np.concatenate([v for v, _, _ in retained]).astype(np.float32),
                    position_offsets=np.cumsum([0]+[len(v) for v, _, _ in retained]).astype(np.int64),
                    position_residue_counts=np.concatenate([c for _, c, _ in retained]).astype(np.int16),
                    position_residue_offsets=np.asarray([o for _, _, o in retained], dtype=np.int64),
                    position_sum_check_nats=np.asarray(worst))
            temp = path.with_suffix('.tmp')
            with temp.open('wb') as stream:
                np.savez_compressed(stream, **retention, features=(f[1:]-f[0]).astype(np.float32),
                    likelihood=s[1:]-s[0], wt_features=f[0], wt_likelihood=s[0],
                    mutants=np.asarray(row['mutants']), measured=np.asarray(row['measured']),
                    profile_scores=np.asarray(row['profile_scores']), metadata=json.dumps(row_identity),
                    batch_check_likelihood_delta_nats=delta_error, batch_check_feature_relative_l2=feature_error,
                    batch_check_mutation_feature_relative_l2=mutation_feature_error)
            temp.replace(path)
        completed.append(dict(assay=row['assay'], file=filename, sha256=sha(path), cluster=row['cluster'],
            max_packed_tokens=token_lengths[row['assay']], wildtype_id=row['wildtype_id'], mutant_digest=row['mutant_digest'], variants=len(row['mutants'])))
        if args.arm == 'zymctrl':
            completed[-1]['conditioning_ec'] = ec_by_assay[row['assay']]
        publish_progress(manifest_path, progress_path, dict(identity=identity, execution_partition=partition,
            block_indices=[(block_count-1)//2,block_count-1], assays=completed, skipped=skipped,
            batch_size=args.batch_size, updated_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-began, gpu=torch.cuda.get_device_name(arm.device), blas=pinning,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(arm.device), torch_version=torch.__version__), expected_assays=len(selected))
        print(f'{args.arm}: {len(completed)}/{len(selected)} {row["assay"]}', flush=True)


if __name__ == '__main__':
    main()
