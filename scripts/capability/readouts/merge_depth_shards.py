#!/usr/bin/env python3
"""Publish one canonical depth-extraction manifest from disjoint shard receipts.

A shard is execution metadata, not measurement identity: every worker runs the
same frozen code, cohort, checkpoint, precision setting, hooked blocks and
retained summaries, and differs only in which assays of the one deterministic
partition it was assigned. This merger refuses anything else. It verifies
identical identities, identical block counts and skip lists, the deterministic
disjoint assignment, complete coverage of the partition universe, every NPZ
checksum and metadata identity, the ordered mutation identifiers, the labels
and profile values, and both declared precision gates, then writes the
canonical manifest with relative paths to the unchanged worker artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.capability.readouts.extract_frozen_readout import partition_assays, sha
from src.capability.readouts import readout_depth as rd
from src.capability.core.io import write_json


def merge_shards(cohort_path, shard_paths, out, *, allow_smoke=False):
    paths = [Path(p).resolve() for p in shard_paths]
    if len(set(paths)) != len(paths) or len(paths) < 2:
        raise ValueError('At least two distinct shard manifests required')
    sources = [json.loads(path.read_text()) for path in paths]
    first = sources[0]
    identity = first['identity']
    if identity['schema_version'] != rd.EXTRACTION_SCHEMA:
        raise ValueError('Shards are not depth-resolved extractions')
    if identity['cohort_sha256'] != sha(cohort_path):
        raise ValueError('Cohort checksum mismatch')
    if list(identity['summary_names']) != list(rd.POOLED_SUMMARIES):
        raise ValueError('Pooled summary order differs')
    smoke = bool(identity['smoke_variants'] or identity['assay_limit'])
    if smoke and not allow_smoke:
        raise ValueError('Smoke output requires explicit --allow-smoke')
    for key in ('max_score_drift', 'max_feature_drift'):
        value = identity.get(key)
        if value is None or not np.isfinite(value) or value < 0:
            raise ValueError('Finite declared precision gates required for merging')
    cohort = json.loads(Path(cohort_path).read_text())
    cohort_rows = {row['assay']: row for row in cohort['assays']}
    if len(cohort_rows) != len(cohort['assays']):
        raise ValueError('Duplicate cohort assays')
    partition = first['execution_partition']
    if partition is None or partition['method'] != 'greedy_packed_token_work_v1':
        raise ValueError('Incomplete or unsupported partition')
    count, universe = partition['count'], partition['universe']
    if count != len(paths):
        raise ValueError('Shard receipt count differs from the declared partition count')
    assignments = partition_assays(universe, count)
    universe_ids = [row['assay'] for row in universe]
    skipped_ids = [row['assay'] for row in first['skipped']]
    if (len(set(skipped_ids)) != len(skipped_ids) or set(skipped_ids) & set(universe_ids)
            or not (set(skipped_ids) | set(universe_ids)) <= set(cohort_rows)
            or (not smoke and set(skipped_ids) | set(universe_ids) != set(cohort_rows))):
        raise ValueError('Eligible and skipped sets do not partition the cohort')
    out = Path(out).resolve()
    records, seen, receipts, shape = {}, set(), [], None
    for path, source in zip(paths, sources):
        part = source.get('execution_partition') or {}
        index = part.get('index', -1)
        stable = {k: v for k, v in part.items() if k not in ('index', 'assigned_assays')}
        expected_stable = {k: v for k, v in partition.items()
                           if k not in ('index', 'assigned_assays')}
        if (source.get('status') != 'complete' or source['identity'] != identity
                or source['block_count'] != first['block_count']
                or source['skipped'] != first['skipped']
                or source.get('batch_size') != identity['batch_size']
                or stable != expected_stable or not 0 <= index < count or index in seen):
            raise ValueError('Incomplete, duplicate or inconsistent shard identity/partition')
        seen.add(index)
        assigned = assignments[index]
        if part['assigned_assays'] != assigned or [r['assay'] for r in source['assays']] != assigned:
            raise ValueError('Shard assay coverage differs from the deterministic assignment')
        for record in source['assays']:
            assay = record['assay']
            row = cohort_rows[assay]
            expected_variants = identity['smoke_variants'] or len(row['mutants'])
            mutants = row['mutants'][:expected_variants]
            planned = next(r for r in universe if r['assay'] == assay)
            if (assay in records or any(record[k] != row[k] for k in ('cluster', 'wildtype_id'))
                    or record['variants'] != len(mutants)
                    or record['variants'] != planned['variants']
                    or record['max_packed_tokens'] != planned['max_packed_tokens']
                    or not 0 < record['max_packed_tokens'] <= identity['budget']):
                raise ValueError(f'{assay}: assay alignment or planned length mismatch')
            artifact = (path.parent / record['file']).resolve()
            if sha(artifact) != record['sha256']:
                raise ValueError(f'{assay}: NPZ checksum mismatch')
            with np.load(artifact, allow_pickle=False) as saved:
                if json.loads(str(saved['metadata'])) != dict(
                        identity=identity, assay=assay, mutant_digest=record['mutant_digest']):
                    raise ValueError(f'{assay}: NPZ measurement identity mismatch')
                if saved['mutants'].tolist() != mutants:
                    raise ValueError(f'{assay}: NPZ mutation order mismatch')
                for key in ('measured', 'profile_scores'):
                    if not np.array_equal(saved[key], np.asarray(row[key][:len(mutants)])):
                        raise ValueError(f'{assay}: NPZ labels or profile values mismatch')
                blocks = first['block_count']
                observed = tuple(saved[f'pooled_d{i:03d}'].shape for i in range(blocks))
                if any(s[:2] != (len(mutants), len(rd.POOLED_SUMMARIES)) for s in observed):
                    raise ValueError(f'{assay}: unexpected pooled block shape')
                if shape is not None and observed[-1][1:] != shape:
                    raise ValueError('Hidden feature dimensions differ across shards')
                shape = observed[-1][1:]
                if identity['position_resolved']:
                    for i in range(blocks):
                        if saved[f'position_d{i:03d}'].shape != observed[i]:
                            raise ValueError(f'{assay}: position block shape differs from pooled')
                for key in (*(f'pooled_d{i:03d}' for i in range(blocks)), 'likelihood',
                            'wt_likelihood'):
                    if not np.isfinite(saved[key]).all():
                        raise ValueError(f'{assay}: nonfinite retained values')
                for key, gate in (('batch_check_likelihood_delta_nats', 'max_score_drift'),
                                  ('batch_check_mutation_feature_relative_l2', 'max_feature_drift')):
                    value = float(saved[key])
                    if not np.isfinite(value) or value < 0 or value > identity[gate]:
                        raise ValueError(f'{assay}: precision gate failed at {value}')
            records[assay] = dict(record, file=os.path.relpath(artifact, out))
        receipts.append(dict(file=os.path.relpath(path, out), sha256=sha(path), index=index))
    if set(records) != set(universe_ids):
        raise ValueError('Incomplete merged assay coverage')
    payload = dict(
        identity=identity, status='complete', block_count=first['block_count'],
        admitted_block_indices=first['admitted_block_indices'],
        assays=[records[assay] for assay in universe_ids], skipped=first['skipped'],
        batch_size=identity['batch_size'],
        retained_bytes=sum(records[assay]['bytes'] for assay in universe_ids),
        updated_utc=datetime.now(timezone.utc).isoformat(),
        execution_merge=dict(method=partition['method'], count=count, universe=universe,
                             sources=sorted(receipts, key=lambda r: r['index']),
                             merger_sha256=sha(__file__)))
    target = out / f"manifest_depth_{identity['arm']}.json"
    if target.exists():
        raise ValueError('Refusing to overwrite an existing canonical manifest')
    out.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.tmp')
    write_json(temporary, payload)
    temporary.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--shards', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    parser.add_argument('--device', default='cpu', choices=['cpu'])
    args = parser.parse_args()
    print(merge_shards(args.cohort, args.shards, args.out, allow_smoke=args.allow_smoke))


if __name__ == '__main__':
    main()
