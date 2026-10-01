#!/usr/bin/env python3
"""Compare ProGen3 ProteinGym scores at batch size 1 with the published batch size 16.

The per-assay Spearman is the one already stored in each model file. The profile
channel is the lookup Spearman in the published retrieval file, matched on assay
name, wild-type identity and mutant digest. The reported difference is batch size
1 minus batch size 16, summarised as a cluster-mean over wild-type families.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('TRANSFER_PACKAGE_ROOT', Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT))

from src.capability.context.profiles import cluster_bootstrap

BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260923
ARMS = ('progen3-3b', 'progen3-112m')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path: Path) -> dict:
    payload = json.loads(path.read_text())
    rows = {}
    for row in payload['assays']:
        score = row['spearman']
        if isinstance(score, dict) or not np.isfinite(score):
            raise ValueError(f'{path}: {row["assay"]} has no finite model Spearman')
        if row['assay'] in rows:
            raise ValueError(f'{path}: duplicate assay {row["assay"]}')
        rows[row['assay']] = row
    return {'settings': payload['settings'], 'arm': payload['arm'], 'rows': rows,
            'sha256': sha256(path), 'path': str(path)}


def load_lookup(path: Path) -> dict:
    payload = json.loads(path.read_text())
    rows = {}
    for row in payload['assays']:
        if row['assay'] in rows:
            raise ValueError(f'duplicate lookup assay {row["assay"]}')
        rows[row['assay']] = row
    return rows


def summarise(values: list[float], clusters: list) -> dict:
    record = cluster_bootstrap(values, clusters, resamples=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    record['n_assays'] = len(values)
    record['n_clusters'] = len(set(clusters))
    return record


def compare_arm(batch1: dict, batch16: dict, lookup: dict) -> dict:
    if batch1['arm'] != batch16['arm']:
        raise ValueError('the two model files name different arms')
    names = sorted(set(batch1['rows']) & set(batch16['rows']) & set(lookup))
    if names != sorted(batch1['rows']) or names != sorted(batch16['rows']):
        raise ValueError(
            f'{batch1["arm"]}: assay sets differ '
            f'({len(batch1["rows"])}, {len(batch16["rows"])}, {len(lookup)})')
    if batch1['settings'].get('batch_size') != 1:
        raise ValueError(f'{batch1["arm"]}: batch-size-1 file records batch {batch1["settings"].get("batch_size")}')
    if batch16['settings'].get('batch_size') != 16:
        raise ValueError(f'{batch1["arm"]}: published file records batch {batch16["settings"].get("batch_size")}')
    rows = []
    for name in names:
        left, right, profile = batch1['rows'][name], batch16['rows'][name], lookup[name]
        if left['mutant_digest'] != right['mutant_digest'] or left['mutant_digest'] != profile['mutant_digest']:
            raise ValueError(f'{name}: mutant digest mismatch')
        if left['wildtype_id'] != right['wildtype_id'] or left['wildtype_id'] != profile['wildtype_id']:
            raise ValueError(f'{name}: wild-type identity mismatch')
        lookup_spearman = float(profile['spearman']['lookup'])
        model1 = float(left['spearman'])
        model16 = float(right['spearman'])
        rows.append({
            'assay': name,
            'cluster': profile['cluster'],
            'model_spearman_batch1': model1,
            'model_spearman_batch16': model16,
            'model_spearman_batch1_minus_batch16': model1 - model16,
            'lookup_spearman': lookup_spearman,
            'model_minus_lookup_batch1': model1 - lookup_spearman,
            'model_minus_lookup_batch16': model16 - lookup_spearman,
            'model_minus_lookup_batch1_minus_batch16': (model1 - lookup_spearman) - (model16 - lookup_spearman),
        })
    clusters = [row['cluster'] for row in rows]

    def column(key: str) -> dict:
        return summarise([row[key] for row in rows], clusters)

    return {
        'arm': batch1['arm'],
        'n_assays': len(rows),
        'n_clusters': len(set(clusters)),
        'batch1_settings': {key: batch1['settings'].get(key) for key in ('batch_size', 'dtype', 'seed', 'variants')},
        'batch16_settings': {key: batch16['settings'].get(key) for key in ('batch_size', 'dtype', 'seed', 'variants')},
        'model_spearman_batch1': column('model_spearman_batch1'),
        'model_spearman_batch16': column('model_spearman_batch16'),
        'model_spearman_batch1_minus_batch16': column('model_spearman_batch1_minus_batch16'),
        'model_minus_lookup_batch1': column('model_minus_lookup_batch1'),
        'model_minus_lookup_batch16': column('model_minus_lookup_batch16'),
        'model_minus_lookup_batch1_minus_batch16': column('model_minus_lookup_batch1_minus_batch16'),
        'assays': rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch1-3b', type=Path, required=True)
    parser.add_argument('--batch1-112m', type=Path, required=True)
    parser.add_argument('--batch16-3b', type=Path, required=True)
    parser.add_argument('--batch16-112m', type=Path, required=True)
    parser.add_argument('--lookup', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    lookup = load_lookup(args.lookup)
    files = {
        'progen3-3b': (args.batch1_3b, args.batch16_3b),
        'progen3-112m': (args.batch1_112m, args.batch16_112m),
    }
    arms = {}
    for arm in ARMS:
        batch1 = load_model(files[arm][0])
        batch16 = load_model(files[arm][1])
        if batch1['arm'] != arm or batch16['arm'] != arm:
            raise SystemExit(f'{arm}: file arm field does not match')
        arms[arm] = compare_arm(batch1, batch16, lookup)
    report = {
        'schema': 'progen3_batch1_vs_batch16_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'difference': 'batch size 1 minus batch size 16',
        'profile_channel': 'published lookup Spearman, external profile, not recomputed',
        'summary_unit': 'cluster mean over wild-type families at 50% identity',
        'bootstrap': {'draws': BOOTSTRAP_DRAWS, 'seed': BOOTSTRAP_SEED, 'interval': '95% percentile'},
        'lookup_sha256': sha256(args.lookup),
        'arms': arms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=1) + '\n')
    temporary.replace(args.out)
    brief = {
        arm: {
            'model_spearman_batch1_minus_batch16': record['model_spearman_batch1_minus_batch16']['point'],
            'model_minus_lookup_batch1_minus_batch16': record['model_minus_lookup_batch1_minus_batch16']['point'],
            'interval_model': record['model_spearman_batch1_minus_batch16']['interval'],
            'interval_minus_lookup': record['model_minus_lookup_batch1_minus_batch16']['interval'],
        }
        for arm, record in arms.items()}
    print(json.dumps({'out': str(args.out), 'arms': brief}, indent=1))


if __name__ == '__main__':
    main()
