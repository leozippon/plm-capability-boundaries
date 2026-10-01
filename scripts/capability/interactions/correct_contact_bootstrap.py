#!/usr/bin/env python3
"""Correct group-equal contact intervals while retaining the historical receipt.

Rebuild site-pair outcomes from pinned inputs, verify unchanged point estimates,
and recompute every affected interval. Unaffected site-pair-equal estimates and
structure controls are copied from the historical receipt, explicitly recorded
as inherited rather than claimed as rerun.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/capability'))

from scripts.capability.interactions import measure_contact_epsilon_enrichment as enrichment
from scripts.capability.interactions import reaudit_contact_endpoint as reaudit
from src.capability.interactions.contact_enrichment import two_stage_bootstrap
from src.capability.core.io import sha256_file, write_json


def recompute(task):
    support, definition, endpoint, rows, values, treated, boundary, draws, seed = task
    record = two_stage_bootstrap(rows, values, treated, rsa_boundary=boundary,
                                 group_equal=True, draws=draws, seed=seed)
    return support, definition, endpoint, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--published', type=Path, default=reaudit.PUBLISHED)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.out.resolve() == args.published.resolve():
        raise SystemExit('the correction must not overwrite the historical receipt')
    historical = json.loads(args.published.read_text())
    annotation = json.loads(reaudit.ANNOTATION.read_text())
    cohort = json.loads(reaudit.COHORT.read_text())
    if historical['schema'] != enrichment.SCHEMA:
        raise SystemExit('unexpected historical schema')
    if sha256_file(reaudit.ANNOTATION) != historical['inputs']['annotation']['sha256']:
        raise SystemExit('annotation digest mismatch')
    cohort_sha = sha256_file(reaudit.COHORT)
    if cohort_sha != historical['inputs']['cohort_sha256'] or cohort_sha != annotation['inputs']['cohort']['sha256']:
        raise SystemExit('cohort digest mismatch')
    declared = {Path(row['path']).name: row['sha256'] for row in cohort['source_files']}
    parquet = [ROOT / relative for relative in cohort['source_row_order']]
    for path in parquet:
        found = sha256_file(path)
        if found != declared[path.name] or found != historical['inputs']['parquet_sha256'][path.name]:
            raise SystemExit(f'parquet digest mismatch: {path.name}')
    sequences = {row['name']: set(row['sequences']) for row in cohort['backgrounds']}
    channels = enrichment.channel_states(parquet, sequences)
    tasks = []
    bootstrap = historical['declaration']['bootstrap']
    for support, exclude in (('indel_excluded', True), ('all_cycles', False)):
        known = enrichment.site_pair_statistics(cohort, channels, exclude_indel=exclude)['site_pairs']
        rows, boundary = reaudit.support_rows(annotation, known)
        if boundary != historical['declaration']['rsa_boundary']:
            raise SystemExit('matching boundary changed')
        for definition in enrichment.DEFINITIONS:
            treated = enrichment.treated_mask(rows, definition)
            for endpoint in enrichment.ENDPOINTS:
                available = np.array([known[row['site_pair']][endpoint] is not None for row in rows])
                subset = [row for row, ok in zip(rows, available) if ok]
                values = np.array([known[row['site_pair']][endpoint] for row in subset])
                tasks.append((support, definition, endpoint, subset, values, treated[available],
                              boundary, bootstrap['draws'], bootstrap['seed']))
    corrected = copy.deepcopy(historical)
    changes = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for support, definition, endpoint, record in pool.map(recompute, tasks):
            old = historical['supports'][support]['definitions'][definition]['endpoints'][endpoint]['group_equal']
            if abs(record['point'] - old['difference']) > 1e-12:
                raise SystemExit(f'point estimate changed: {support}/{definition}/{endpoint}')
            new = corrected['supports'][support]['definitions'][definition]['endpoints'][endpoint]['group_equal']
            for key in ('interval', 'excludes_zero', 'skipped_draws'):
                new[key] = record[key]
            new['bootstrap_draws'] = record['draws']
            changes.append({'support': support, 'definition': definition, 'endpoint': endpoint,
                            'old_interval': old['interval'], 'interval': record['interval'],
                            'old_excludes_zero': old['excludes_zero'], 'excludes_zero': record['excludes_zero']})
            print(json.dumps(changes[-1]), flush=True)
    corrected['bootstrap_correction'] = {
        'historical_path': str(args.published), 'historical_sha256': sha256_file(args.published),
        'reason': 'Preserve distinct sampled group occurrences during group-equal normalization.',
        'recomputed': changes,
        'inherited': 'All point estimates, site-pair-equal intervals, annotations and structure controls.',
        'code_sha256': {str(path.relative_to(ROOT)): sha256_file(path) for path in (
            Path(__file__), ROOT / 'src/capability/interactions/contact_enrichment.py',
            ROOT / 'scripts/capability/interactions/measure_contact_epsilon_enrichment.py')},
    }
    write_json(args.out, corrected)


if __name__ == '__main__':
    main()
