#!/usr/bin/env python3
"""Capture every transformer block's frozen readout summaries in one forward pass.

The admitted readout extraction hooked two blocks per arm. This stage hooks
every block of the same frozen checkpoint, on the same cohort, with the same
native rendering, scoring span, marker exclusions, precision settings and batch
composition, and retains the same two pooling rules at each depth. Where the
rendering assigns exactly one token per residue, so that a wild type and each of
its mutants occupy the same token grid, it additionally retains two
position-resolved summaries localized on the substituted residues and the
causal prefix control that must be exactly zero.

Capturing every block costs the same single forward pass per sequence as
capturing two; the additional cost is storage and fitting. Nothing about the
measurement identity changes except the set of hooked blocks and the retained
summaries, both declared in the manifest identity.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.capability.readouts.extract_frozen_readout import eligible_max_tokens, partition_assays, publish_progress, sha
from src.capability.readouts import readout_depth as rd
from src.capability.readouts.readout_extraction import (TEXT_EXTRA_ARM, bind_readout_ec, load_readout_arm,
                                             pack_sequence, representation_blocks, text_boundary,
                                             validate_ec_conditioning)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--conditioning-map', type=Path,
                        help='Hash-bound exact-WT EC annotation sidecar, required only for ZymCTRL')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--dtype', default='float32')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--budget', type=int, default=1024)
    parser.add_argument('--assay-limit', type=int, default=0)
    parser.add_argument('--smoke-variants', type=int, default=0)
    parser.add_argument('--smoke-length-strata', action='store_true')
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--max-score-drift', type=float)
    parser.add_argument('--max-feature-drift', type=float)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Invalid shard index/count')
    if args.batch_size < 1 or args.budget != 1024:
        raise ValueError('Positive batch size and frozen 1024-position budget required')
    if not args.smoke_variants and (args.max_score_drift is None or args.max_feature_drift is None):
        raise ValueError('Full extraction requires explicit precision limits after qualification')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = importlib.util.spec_from_file_location('stage46', ROOT / 'scripts/capability/stages/context_homologue.py')
    stage46 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage46)

    cohort = json.loads(args.cohort.read_text())
    if (args.arm == 'zymctrl') != (args.conditioning_map is not None):
        raise ValueError('--conditioning-map is required exactly for ZymCTRL')
    ec_by_assay, conditioning_sha256, conditioning_payload = {}, None, None
    if args.conditioning_map is not None:
        raw = args.conditioning_map.read_bytes()
        conditioning_sha256 = hashlib.sha256(raw).hexdigest()
        conditioning_payload = json.loads(raw)
        ec_by_assay = validate_ec_conditioning(conditioning_payload, cohort, sha(args.cohort))
    full_sequences = {row['assay']: [row['wildtype']] + list(row['sequences'])
                      for row in cohort['assays']}
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
            raise ValueError('Length-strata smoke requires smoke variants, assay limit 3 and 3 eligible assays')
        ordered = sorted(selected, key=lambda row: (token_lengths[row['assay']], row['assay']))
        selected = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    if args.assay_limit:
        selected = selected[:args.assay_limit]
    if not selected:
        raise ValueError('No eligible assays')

    # Position resolution is decided by the rendering alone, on the first
    # eligible assay's wild type and first mutant, and then enforced on every
    # row: a later row that is not one token per residue is a refusal, never a
    # silent fallback to the pooled summaries.
    probe = selected[0]
    if args.arm == 'zymctrl':
        bind_readout_ec(handle, ec_by_assay[probe['assay']])
    probe_sequences = [probe['wildtype'], probe['sequences'][0]]
    position_resolved = rd.per_residue_grid(
        handle, probe_sequences, [pack_sequence(handle, s) for s in probe_sequences])

    partition = None
    if args.shard_count > 1:
        universe = [dict(assay=row['assay'], max_packed_tokens=token_lengths[row['assay']],
                         variants=len(row['mutants'])) for row in selected]
        assigned = partition_assays(universe, args.shard_count)[args.shard_index]
        partition = dict(method='greedy_packed_token_work_v1', count=args.shard_count,
                         index=args.shard_index, universe=universe, assigned_assays=assigned)
        selected = [row for row in selected if row['assay'] in set(assigned)]
    args.out.mkdir(parents=True, exist_ok=True)

    code_paths = [Path(__file__), ROOT / 'src/capability/readouts/readout_depth.py',
                  ROOT / 'src/capability/readouts/readout_extraction.py',
                  ROOT / 'src/capability/context/context_homologue.py',
                  ROOT / 'scripts/capability/stages/context_homologue.py',
                  ROOT / 'scripts/capability/readouts/extract_frozen_readout.py',
                  ROOT / 'src/capability/core/arms.py', ROOT / 'src/capability/models/proteinglm.py',
                  ROOT / 'src/capability/models/progen3.py', ROOT / 'src/capability/models/joint_lineage.py']
    if text_boundary(handle) is not None:
        code_paths.extend(ROOT / name for name in
                          ('src/capability/models/text_aa_fitness.py', 'src/capability/models/text_aa_cohort.py',
                           'src/capability/text_aa_boundaries.json', 'src/capability/core/precision_policy.py'))
    checkpoint = handle.spec.path
    checkpoint_metadata = {str(x.relative_to(checkpoint)): sha(x) for x in sorted(checkpoint.iterdir())
                           if x.is_file() and (x.suffix == '.json'
                                               or x.name in ('tokenizer.model', 'tokenizer.json'))}
    identity = dict(
        schema_version=rd.EXTRACTION_SCHEMA, arm=args.arm, dtype=args.dtype,
        batch_size=args.batch_size, checkpoint_path=str(checkpoint.resolve()),
        checkpoint_tensor_files=[dict(name=x.name, bytes=x.stat().st_size)
                                 for x in sorted(checkpoint.iterdir())
                                 if x.is_file() and x.suffix in ('.safetensors', '.bin', '.pt')],
        checkpoint_metadata_sha256=checkpoint_metadata, cohort_sha256=sha(args.cohort),
        code_sha256={str(x.relative_to(ROOT)): sha(x) for x in code_paths},
        budget=args.budget, smoke_variants=args.smoke_variants,
        smoke_length_strata=args.smoke_length_strata, max_score_drift=args.max_score_drift,
        max_feature_drift=args.max_feature_drift, assay_limit=args.assay_limit,
        summary_names=list(rd.POOLED_SUMMARIES), position_summary_names=list(rd.POSITION_SUMMARIES),
        position_resolved=position_resolved,
        block_semantics='every zero-based transformer block output, before final output normalization',
        representation_semantics=('mutant-minus-WT; per block the residue-bearing token mean and '
                                  'the last residue-bearing token state; no EOS, BOS, context or '
                                  'padding'),
        position_semantics=('mutant-minus-WT mean over the substituted residues own tokens and over '
                            'every residue token after the last substituted one; defined only on a '
                            'one-token-per-residue grid'
                            if position_resolved else 'not defined for this rendering'),
        scoring_stratum='target_only_native_packed_residue_span')
    if text_boundary(handle) is not None:
        identity['scoring_stratum'] = 'literal_AA_native_text_token_sum'
        identity['text_aa_boundary'] = json.loads(json.dumps(asdict(text_boundary(handle))))
        identity['scoring_target_rule'] = ('tokens2..L; first byte is unscored context'
                                           if text_boundary(handle).conditioning_id is None
                                           else 'all AA-bearing tokens after the document prefix')
        identity['representation_semantics'] = (
            'mutant-minus-WT; per block the mean and last literal AA-bearing token states, '
            'excluding the document prefix; byte-zero included for no-prefix ByGPT5; BPE token '
            'means are not per-residue means')
        identity['research_role'] = (
            'unprompted instruction-tuned literal-AA string readout; not a chat-template evaluation'
            if args.arm == TEXT_EXTRA_ARM else 'literal-AA text string readout')
    if args.conditioning_map is not None:
        identity['conditioning_map_sha256'] = conditioning_sha256
        identity['conditioning_annotation_source'] = conditioning_payload['annotation_source']
        identity['conditioning_selection_rule'] = conditioning_payload['selection_rule']
        identity['scoring_stratum'] = 'exact_WT_EC_conditioned_native_residue_span'
        identity['representation_semantics'] = (
            'mutant-minus-WT; per block the mean and last residue states strictly between native '
            'start/end markers; same genuine WT EC for every variant')
    if args.arm == 'protgpt2':
        identity['scoring_stratum'] = 'native_fasta_target_tokens_including_formatting_newlines'
        identity['representation_semantics'] = (
            'mutant-minus-WT; per block the mean and last residue-bearing native FASTA token '
            'states; exclude pure newline tokens; retain mixed residue/newline BPE tokens')

    manifest_path = args.out / f'manifest_depth_{args.arm}.json'
    progress_path = args.out / f'progress_depth_{args.arm}.json'
    if partition is not None:
        manifest_path = args.out / f'shard_depth_{args.arm}_{args.shard_index}.json'
        progress_path = args.out / f'progress_depth_{args.arm}_shard_{args.shard_index}.json'
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
        from src.capability.context import context_homologue as ch
        ch.require_position_budget(arm.model.config, arm=args.arm)
    block_count = len(representation_blocks(arm))
    completed = []
    began = time.monotonic()
    for row in selected:
        if args.arm == 'zymctrl':
            bind_readout_ec(arm, ec_by_assay[row['assay']])
        filename = f"depth_{args.arm}_{hashlib.sha256(row['assay'].encode()).hexdigest()[:20]}.npz"
        path = args.out / filename
        row_identity = dict(identity=identity, assay=row['assay'], mutant_digest=row['mutant_digest'])
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if json.loads(str(saved['metadata'])) != row_identity:
                    raise ValueError(f'Resume artifact identity mismatch: {path}')
        else:
            extraction = rd.extract_assay(
                arm, stage46, row['wildtype'], list(row['sequences']), list(row['mutants']),
                batch_size=args.batch_size, position_resolved=position_resolved,
                max_score_drift=args.max_score_drift if args.max_score_drift is not None else np.inf,
                max_feature_drift=args.max_feature_drift if args.max_feature_drift is not None else np.inf)
            if extraction.pooled.shape[1] != block_count:
                raise ValueError('Retained block count differs from the hooked stack')
            print(f"precision {row['assay']}: delta_M_nats={extraction.drift['likelihood_delta_nats']}, "
                  f"state_relative_l2={extraction.drift['state_relative_l2']}, "
                  f"mutation_relative_l2={extraction.drift['mutation_relative_l2']}", flush=True)
            payload = rd.assay_arrays(extraction, mutants=row['mutants'], measured=row['measured'],
                                      profile_scores=row['profile_scores'],
                                      metadata=json.dumps(row_identity))
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream, **payload)
                # The digest below is read back from the shared filesystem, so the
                # bytes must have settled there first. Without this, two archives
                # of 6,834 written at about 101 MB each recorded a digest over a
                # read-back that differed from the settled file: identical length,
                # all 84 member CRC32 checks intact and identity metadata binding,
                # but a SHA-256 the fit stage then refused.
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        record = dict(assay=row['assay'], file=filename, sha256=sha(path), cluster=row['cluster'],
                      max_packed_tokens=token_lengths[row['assay']], wildtype_id=row['wildtype_id'],
                      mutant_digest=row['mutant_digest'], variants=len(row['mutants']),
                      bytes=path.stat().st_size)
        if args.arm == 'zymctrl':
            record['conditioning_ec'] = ec_by_assay[row['assay']]
        completed.append(record)
        publish_progress(manifest_path, progress_path, dict(
            identity=identity, execution_partition=partition, block_count=block_count,
            admitted_block_indices=list(rd.admitted_block_indices(block_count)),
            assays=completed, skipped=skipped, batch_size=args.batch_size,
            retained_bytes=sum(r['bytes'] for r in completed),
            updated_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic() - began, gpu=torch.cuda.get_device_name(arm.device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(arm.device),
            torch_version=torch.__version__), expected_assays=len(selected))
        print(f"{args.arm}: {len(completed)}/{len(selected)} {row['assay']} "
              f"blocks={block_count} position_resolved={position_resolved}", flush=True)


if __name__ == '__main__':
    main()
