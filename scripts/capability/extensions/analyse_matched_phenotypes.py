#!/usr/bin/env python3
"""Analyse the matched multi-phenotype support: is the information shared?

This is the E06 analysis. It takes the matched pairs the qualification stage
admitted -- one exact wild-type sequence, one substitution, two assays of
different adjudicated phenotype classes -- and asks whether a checkpoint's
predictive information about a mutation is a property of the mutation or of the
phenotype being measured.

The likelihood difference of a substitution is one number whatever phenotype is
measured on it, so a model cannot carry more information about one phenotype
through a different score. Three quantities on identical rows separate the cases:

* the rank correlation of the two phenotype labels, which is the biology and owes
  nothing to the model;
* the ranking increment over the qualified control for each phenotype, and the
  signed difference between them, which says whether the one score aligns with
  one phenotype better than the other;
* the rank correlation of the two phenotypes' residuals with the likelihood in
  the design and without it. A negative change means the likelihood explained
  part of what the two phenotypes miss *in common*, which is shared information;
  an unchanged shared residual beside two positive increments means the
  likelihood added phenotype-specific information instead.

Counts are reported per phenotype pair before any of this, because no single pair
reaches the independent-group floor: the pooled protein-unit band is the only
inference and every per-pair number is marked descriptive.

``--device`` is accepted because the campaign queue injects it; the analysis is CPU.
"""
from __future__ import annotations

import argparse
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
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '4')

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.core.statistics import mean_interval  # noqa: E402
from src.capability.extensions import matched_phenotypes as matched  # noqa: E402
from src.capability.extensions import phenotype_breadth as breadth  # noqa: E402
from src.capability.stability.stability_gate import load_profiles  # noqa: E402

COMPLETION = 'matched_phenotype_analysis.json'
MATCHED_KEY = 'matched_proteingym'


def emit(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def resources(device: str) -> dict:
    return {'executable': sys.executable, 'device': device, 'cpus': os.cpu_count(),
            'disk_free_bytes': shutil.disk_usage(ROOT).free,
            'memory': [line for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith(('MemTotal:', 'MemAvailable:'))],
            'threads': {v: os.environ[v] for v in
                        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--scores', type=Path, action='append', required=True)
    parser.add_argument('--profiles', type=Path, default=None)
    parser.add_argument('--proteingym-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--strata-metadata', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu', help='accepted for queue compatibility')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f'refusing an existing output directory: {out}')
    out.mkdir(parents=True)
    admission = json.loads(
        (args.qualification / 'cohorts' / f'{MATCHED_KEY}.json').read_text(encoding='utf-8'))
    if admission['status'] != 'admitted':
        raise SystemExit(f'the matched cohort was {admission["status"]}: '
                         + '; '.join(admission['blockers']))
    started = time.monotonic()
    write_json(out / 'resource-start.json',
               {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'resources': resources(args.device), 'argv': sys.argv[1:]})
    try:
        scores = breadth.read_arm_scores(args.scores)
    except ValueError as error:
        raise SystemExit(str(error))
    reference = matched.read_reference(args.reference)
    strata = matched.read_strata_metadata(args.strata_metadata)
    pairs, census = matched.discover_pairs(args.proteingym_dir, reference, strata)
    if census != admission['discovery_census']:
        raise SystemExit('re-derived matched discovery disagrees with the admitted census')
    counts = matched.pair_counts(pairs)
    emit(out / 'matched-pair-counts.json', counts)

    profiles = None
    profile_record = {'supplied': False}
    if args.profiles is not None:
        plan = json.loads((args.qualification / 'profile-plan.json').read_text(encoding='utf-8'))
        wildtypes = {row['name']: row['wildtype'] for row in plan['backgrounds']}
        profiles, meta = load_profiles(args.profiles, wildtypes)
        present = sum(1 for r in meta['backgrounds'] if r['status'] == 'present')
        profile_record = {'supplied': True, 'path': str(args.profiles),
                          'sha256': sha256_file(args.profiles),
                          'backgrounds': len(meta['backgrounds']), 'present': present,
                          'absent': len(meta['backgrounds']) - present}
    candidates = (breadth.CANDIDATE_BLOCKS if profiles is not None else
                  tuple(b for b in breadth.CANDIDATE_BLOCKS if not b.startswith('prof')))

    censuses: list[dict] = []

    def blocks_for(rows):
        pruned, census = breadth.prune_constant_columns(breadth.control_blocks(rows, profiles))
        censuses.append(census)
        return pruned

    # Control qualification runs once, on the pooled matched support with a
    # protein-held-out partition, before any likelihood column exists.
    pooled_rows, pooled_sides = [], []
    for pair in pairs:
        rows_a, rows_b, _ = matched.matched_rows(pair, args.proteingym_dir, cohort=MATCHED_KEY)
        pooled_rows.extend(rows_a)
        pooled_sides.extend(rows_b)
    pooled_blocks = blocks_for(pooled_rows)
    emit(out / 'control-column-census.json', censuses[-1])
    backgrounds = np.asarray([row.background for row in pooled_rows])
    groups = np.asarray([row.unit for row in pooled_rows])
    labels = np.asarray([row.oriented_label() for row in pooled_rows], dtype=np.float64)
    control_qualification = breadth.qualify_controls(
        pooled_blocks, breadth.rerank(labels, backgrounds), backgrounds, groups,
        candidates=candidates)
    control_qualification.update({
        'cohort': MATCHED_KEY, 'endpoint': 'ranking',
        'support': 'pooled matched rows of the first-listed assay of every pair',
        'partition': 'protein-held-out outer folds; the independent group is the protein',
        'profile_control': 'present' if profiles is not None else 'absent',
        'profile_note': (None if profiles is not None else
                         'without the evolutionary-profile blocks an evolutionary-statistics '
                         'account of any matched increment is not excluded')})
    emit(out / 'control-qualification.json', control_qualification)
    qualified = control_qualification['qualified']

    sharing = matched.label_sharing(pairs, args.proteingym_dir)
    emit(out / 'label-sharing.json', sharing)

    cells, skipped = [], []
    for pair in pairs:
        try:
            cells.extend(matched.pair_increments(pair, args.proteingym_dir, blocks_for,
                                                 scores, qualified))
        except ValueError as error:
            skipped.append({**pair.record(), 'reason': repr(error)})
    if skipped:
        emit(out / 'skipped-pairs.json', skipped)
    if not cells:
        raise SystemExit('no matched pair could be fitted')
    emit(out / 'pair-cells.json', cells)
    pooled = matched.pooled_inference(cells, sorted(scores))
    emit(out / 'pooled-inference.json', pooled)

    label_interval = mean_interval([row['label_spearman'] for row in sharing
                                    if row['label_spearman'] is not None])
    by_pair = {}
    for row in sharing:
        by_pair.setdefault(tuple(row['phenotype_pair']), []).append(row['label_spearman'])
    headline = []
    for arm in sorted(scores):
        selected = [r for r in pooled['results'] if r['arm'] == arm]
        record = {'arm': arm}
        for item in selected:
            record[item['contrast']] = {
                'point': item['point'], 'simultaneous': item['simultaneous'],
                'resolved_positive': item['resolved_positive'],
                'resolved_negative': item['resolved_negative']}
        headline.append(record)
    write_json(out / 'resource-end.json',
               {'status': 'complete', 'resources': resources(args.device),
                'seconds': time.monotonic() - started})
    write_json(out / COMPLETION, {
        'schema': matched.SCHEMA, 'stage': 'analyse_matched_phenotypes', 'status': 'complete',
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'seconds': time.monotonic() - started,
        'qualification': str(args.qualification),
        'qualification_sha256': sha256_file(args.qualification / 'cohort_qualification.json'),
        'arms': sorted(scores), 'profiles': profile_record,
        'orientation': matched.ORIENTATION,
        'matched_pair_counts': counts,
        'label_sharing': {'per_pair_spearman': [
            {'phenotype_pair': list(key), 'values': values} for key, values in sorted(by_pair.items())],
            'protein_mean_interval': label_interval,
            'reading': ('a label correlation is the biology of the two phenotypes and is computed '
                        'without any model quantity')},
        'control_qualification': {k: v for k, v in control_qualification.items() if k != 'offers'},
        'within_protein_partition': ('mutated-position-held-out folds inside each protein; the '
                                     'increment generalises across sites of a known protein and is '
                                     'not the cross-protein increment the anchor panel reports'),
        'pooled_inference': pooled,
        'headline': headline,
        'skipped_pairs': skipped,
        'split_seeds': list(breadth.SPLIT_SEEDS),
        'bootstrap': {'draws': breadth.BOOTSTRAP_DRAWS, 'seed': breadth.BOOTSTRAP_SEED},
        'limitations': matched.LIMITATIONS,
        'resources': resources(args.device)})
    print(json.dumps({'out': str(out), 'arms': sorted(scores),
                      'pairs': len(pairs), 'cells': len(cells),
                      'pooled_groups': len(pooled['groups']),
                      'label_mean_spearman': label_interval['mean']}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
