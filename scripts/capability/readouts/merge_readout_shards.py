#!/usr/bin/env python3
"""Validate disjoint fresh extraction shards and publish one canonical manifest (CPU only)."""
from __future__ import annotations
import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.capability.readouts.extract_frozen_readout import partition_assays, sha, FEATURE_NAMES
from src.capability.core.io import write_json


def merge_shards(cohort_path, shard_paths, out, *, allow_smoke=False):
    paths = [Path(p).resolve() for p in shard_paths]
    if len(set(paths)) != len(paths) or len(paths) < 2:
        raise ValueError('At least two distinct shard manifests required')
    sources = [json.loads(p.read_text()) for p in paths]
    first = sources[0]
    identity = first['identity']
    if identity['cohort_sha256'] != sha(cohort_path):
        raise ValueError('Cohort checksum mismatch')
    if identity['feature_names'] != list(FEATURE_NAMES):
        raise ValueError('Feature definitions differ')
    smoke = bool(identity['smoke_variants'] or identity['assay_limit'])
    if smoke and not allow_smoke:
        raise ValueError('Smoke output requires explicit --allow-smoke')
    if any(identity.get(k) is None or not np.isfinite(identity[k]) or identity[k] < 0
           for k in ('max_score_drift', 'max_feature_drift')):
        raise ValueError('Finite declared precision gates required for merging')
    cohort = json.loads(Path(cohort_path).read_text())
    rows = {r['assay']: r for r in cohort['assays']}
    if len(rows) != len(cohort['assays']):
        raise ValueError('Duplicate cohort assays')
    partition = first['execution_partition']
    count, universe = partition['count'], partition['universe']
    if count != len(paths) or partition['method'] != 'greedy_packed_token_work_v1':
        raise ValueError('Incomplete or unsupported partition')
    assignments = partition_assays(universe, count)
    universe_ids = [r['assay'] for r in universe]
    skipped_ids = [r['assay'] for r in first['skipped']]
    if (len(set(skipped_ids)) != len(skipped_ids) or set(skipped_ids) & set(universe_ids)
        or not (set(skipped_ids) | set(universe_ids)) <= set(rows)
        or (not smoke and set(skipped_ids) | set(universe_ids) != set(rows))):
        raise ValueError('Eligible and skipped sets do not partition cohort')
    expected_order = [r['assay'] for r in cohort['assays'] if r['assay'] in set(universe_ids)]
    if not identity['smoke_length_strata'] and universe_ids != expected_order:
        raise ValueError('Universe order differs from cohort')
    out = Path(out).resolve()
    records, seen_indices, receipts, feature_shape = {}, set(), [], None
    for path, source in zip(paths, sources):
        part = source.get('execution_partition', {})
        index = part.get('index', -1)
        if (source.get('status') != 'complete' or source['identity'] != identity
            or source['block_indices'] != first['block_indices'] or source['skipped'] != first['skipped']
            or source.get('batch_size') != identity['batch_size']
            or {k:v for k,v in part.items() if k not in ('index', 'assigned_assays')}
               != {k:v for k,v in partition.items() if k not in ('index', 'assigned_assays')}
            or not 0 <= index < count or index in seen_indices):
            raise ValueError('Incomplete, duplicate, or inconsistent shard identity/partition')
        seen_indices.add(index)
        assigned = assignments[index]
        if part['assigned_assays'] != assigned or [r['assay'] for r in source['assays']] != assigned:
            raise ValueError('Shard assay coverage differs from deterministic assignment')
        for record in source['assays']:
            assay = record['assay']; row = rows[assay]
            expected_n = identity['smoke_variants'] or len(row['mutants'])
            mutants = row['mutants'][:expected_n]
            digest = hashlib.sha256('\n'.join(mutants).encode()).hexdigest() if identity['smoke_variants'] else row['mutant_digest']
            planned = next(r for r in universe if r['assay'] == assay)
            if (assay in records or any(record[k] != row[k] for k in ('cluster','wildtype_id'))
                or record['mutant_digest'] != digest or record['variants'] != len(mutants)
                or record['variants'] != planned['variants']
                or record['max_packed_tokens'] != planned['max_packed_tokens']
                or not 0 < record['max_packed_tokens'] <= identity['budget']):
                raise ValueError('Assay alignment or planned length mismatch')
            artifact = (path.parent / record['file']).resolve()
            if sha(artifact) != record['sha256']:
                raise ValueError('NPZ checksum mismatch')
            with np.load(artifact, allow_pickle=False) as saved:
                if json.loads(str(saved['metadata'])) != dict(identity=identity, assay=assay, mutant_digest=digest):
                    raise ValueError('NPZ measurement identity mismatch; old/new artifacts cannot mix')
                if saved['mutants'].tolist() != mutants:
                    raise ValueError('NPZ mutation order mismatch')
                for key in ('measured','profile_scores'):
                    if not np.array_equal(saved[key], np.asarray(row[key][:len(mutants)])):
                        raise ValueError('NPZ labels/profile mismatch')
                f = saved['features']
                if f.ndim != 3 or f.shape[:2] != (len(mutants),4) or f.shape[2] < 1:
                    raise ValueError('Invalid feature shape')
                if feature_shape is not None and f.shape[1:] != feature_shape:
                    raise ValueError('Hidden feature dimensions differ across shards')
                feature_shape = f.shape[1:]
                if saved['likelihood'].shape != (len(mutants),) or saved['wt_features'].shape != feature_shape:
                    raise ValueError('Invalid likelihood or WT feature shape')
                for key in ('features','likelihood','wt_features','wt_likelihood','batch_check_feature_relative_l2'):
                    if not np.isfinite(saved[key]).all():
                        raise ValueError('Nonfinite NPZ values')
                for key, gate in (('batch_check_likelihood_delta_nats','max_score_drift'),
                                  ('batch_check_mutation_feature_relative_l2','max_feature_drift')):
                    value = float(saved[key])
                    if not np.isfinite(value) or value < 0 or value > identity[gate]:
                        raise ValueError('Precision gate failed')
            records[assay] = dict(record, file=os.path.relpath(artifact, out))
        receipts.append(dict(file=os.path.relpath(path, out), sha256=sha(path), index=index))
    if set(records) != set(universe_ids):
        raise ValueError('Incomplete merged assay coverage')
    payload = dict(identity=identity, status='complete', block_indices=first['block_indices'],
                   assays=[records[a] for a in universe_ids], skipped=first['skipped'],
                   batch_size=identity['batch_size'], updated_utc=datetime.now(timezone.utc).isoformat(),
                   execution_merge=dict(method=partition['method'], count=count, universe=universe,
                                        sources=sorted(receipts,key=lambda r:r['index']), merger_sha256=sha(__file__)))
    target = out / f"manifest_{identity['arm']}.json"
    if target.exists():
        raise ValueError('Refusing to overwrite an existing canonical manifest')
    out.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.tmp')
    write_json(temporary, payload)
    temporary.replace(target)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--shards', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--allow-smoke', action='store_true')
    p.add_argument('--device', default='cpu', choices=['cpu'])
    args = p.parse_args()
    print(merge_shards(args.cohort,args.shards,args.out,allow_smoke=args.allow_smoke))


if __name__ == '__main__':
    main()
