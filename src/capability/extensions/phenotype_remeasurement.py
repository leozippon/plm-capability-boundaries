"""One-protein external remeasurement of frozen fitted OOF predictions.

Only evaluation ranks change. No fitting, inference, noise replication, population
resampling, checkpoint independence assumption, or quantitative-scale loss.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ASSAY = 'OPSD_HUMAN_Wan_2019'
CHANNELS = ('RHO_method1', 'RHO_method2')
LIMITATIONS = [
    'External remeasurement of the same RHO protein/family, not an independent new protein.',
    'BPL and BMPL are frozen fitted OOF predictions on old standardized-rank labels, not raw likelihoods.',
    'BMPL minus BPL is fixed-readout sensitivity, not a newly fitted likelihood increment or new-phenotype baseline qualification.',
    'One protein/family: no family bootstrap, checkpoint-iid tests, hypothesis significance or population generalization.',
    'Seeds and checkpoints are descriptive repeated readouts, not independent experimental units; no winner selection.',
    'No physical-scale MSE; existing measurement SE is retained descriptively, without invented normal replicates or noise resampling.',
    'Full native protein-state equality does not equate assay constructs or genotypes; reporter/antibody and membrane-proximity contexts differ.',
    'Source discordance is not filtered beyond each method accepted QC; paired common support is the primary denominator.',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def identity_sha(value):
    return text_sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False))


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float64).tobytes()).hexdigest()


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def read_rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle]


def project_path(root, supplied):
    """One entry-point policy: relative CLI paths are relative to the project root."""
    path = Path(supplied)
    return (path if path.is_absolute() else Path(root) / path).resolve()


def measurement_source_path(root, source_path):
    """Resolve declared source paths independently of the chosen output directory."""
    relative = Path(source_path)
    require(not relative.is_absolute() and bool(relative.parts) and '..' not in relative.parts,
            'unsafe measurement source path')
    base = Path(root) if relative.parts[0] in ('results', 'data') else (
        Path(root) / 'results/extensions/phenotype_followups_20261007/discovery')
    return base / relative


def measurement_replication(manifest):
    selected = [row for row in manifest if row.get('protein') == 'RHO']
    require(len(selected) == len(CHANNELS) and {row['channel'] for row in selected} == set(CHANNELS),
            'RHO method metadata coverage differs')
    return {row['channel']: {'study_replicates': row.get('study_replicates'),
                            'per_variant_replicates': row.get('per_variant_replicates')}
            for row in selected}


def state(wt, mutation):
    require(bool(wt) and set(wt) <= set('ACDEFGHIKLMNPQRSTVWY'), 'invalid full WT')
    match = re.fullmatch(r'([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])', mutation)
    if match is None:
        raise ValueError('original state must be an explicit single substitution')
    before, pos, after = match.groups()
    pos = int(pos)
    require(1 <= pos <= len(wt) and wt[pos - 1] == before and before != after,
            'mutation disagrees with original WT')
    mutant = wt[:pos - 1] + after + wt[pos:]
    return {'mutation': mutation, 'wt_sha256': text_sha(wt), 'mutant_sha256': text_sha(mutant),
            'state_id': text_sha(wt) + ':' + text_sha(mutant), 'mutated_sequence': mutant}


def join_measurements(wt, external_wt, original, canonical, rows):
    """Exact full-state join, with both accepted methods on one denominator."""
    require(wt == external_wt, 'external WT does not exactly match original cohort WT')
    old = {}
    for sample in original:
        bound = state(wt, sample['mutation'])
        require(bound['state_id'] not in old, 'duplicate original protein state')
        old[bound['state_id']] = {**sample, **bound}
    require(bool(old), 'empty original support')
    states = {}
    for saved in canonical:
        if saved['protein'] != 'RHO':
            continue
        bound = state(wt, saved['mutation'])
        require(all(saved[k] == bound[k] for k in bound), 'canonical full protein-state mismatch')
        require(saved['state_id'] not in states, 'duplicate canonical state')
        states[saved['state_id']] = saved
    per_method = {channel: {} for channel in CHANNELS}
    rejected = {channel: {} for channel in CHANNELS}
    for row in rows:
        if row['protein'] != 'RHO':
            continue
        require(row['channel'] in CHANNELS, 'unknown RHO method')
        if not row['accepted']:
            if row.get('mutation'):
                rejected[row['channel']][row['mutation']] = row
            continue
        bound = state(wt, row['mutation'])
        require(row['state_id'] == bound['state_id'] and row['state_id'] in states,
                'accepted row canonical state mismatch')
        require(np.isfinite(row['score']) and row['uncertainty'] is not None
                and np.isfinite(row['uncertainty']) and row['uncertainty'] >= 0,
                'accepted measurement missing score/SE QC')
        channel_rows = per_method[row['channel']]
        require(row['state_id'] not in channel_rows, 'duplicate accepted method state')
        channel_rows[row['state_id']] = row
    common = set(old).intersection(*(set(per_method[c]) for c in CHANNELS))
    require(len(common) >= 3, 'blocked: empty or fewer than three BOTH-method original-state matches')
    registry = []
    for sid, sample in old.items():
        methods = {}
        for channel in CHANNELS:
            row = per_method[channel].get(sid)
            rejected_row = rejected[channel].get(sample['mutation'])
            methods[channel] = {'accepted': row is not None, 'row': row,
                                'missing_QC': None if row else (
                                    rejected_row['exclusions'] if rejected_row else ['no_source_row']),
                                'rejected_row': rejected_row}
        registry.append({**sample, 'primary_matched': sid in common, 'methods': methods})
    coverage = {c: {'accepted_external_states': len(per_method[c]),
                    'original_accepted_matches': len(set(old) & set(per_method[c])),
                    'original_missing_or_rejected': len(set(old) - set(per_method[c]))}
                for c in CHANNELS}
    return registry, [r for r in registry if r['primary_matched']], coverage


def rho(x, y):
    """Evaluation-only average-tie Spearman; never mutate prediction features."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    require(x.ndim == y.ndim == 1 and x.shape == y.shape and len(x) >= 3
            and np.isfinite(x).all() and np.isfinite(y).all(), 'unaligned/nonfinite evaluation')
    a, b = rankdata(x), rankdata(y)
    require(np.ptp(a) > 0 and np.ptp(b) > 0, 'undefined constant-vector Spearman')
    return float(np.corrcoef(a, b)[0, 1])


def paired_contrasts(bpl, bmpl, method1, method2):
    scores = []
    for method in (method1, method2):
        baseline, augmented = rho(bpl, method), rho(bmpl, method)
        scores.append({'rho_BPL': baseline, 'rho_BMPL': augmented,
                       'difference_BMPL_minus_BPL': augmented - baseline})
    return {'methods': dict(zip(CHANNELS, scores)),
            'method_contrast_difference_method2_minus_method1':
                scores[1]['difference_BMPL_minus_BPL'] - scores[0]['difference_BMPL_minus_BPL']}


def descriptive_summary(cells):
    require(bool(cells), 'empty prediction cells')
    models = sorted({c['model'] for c in cells})
    summaries = []
    for model in models:
        selected = [c for c in cells if c['model'] == model]
        require(len(selected) == 3 and len({c['seed'] for c in selected}) == 3,
                'each model requires three distinct original seeds')
        methods = {channel: {key: float(np.mean([c['methods'][channel][key] for c in selected]))
                             for key in ('rho_BPL', 'rho_BMPL', 'difference_BMPL_minus_BPL')}
                   for channel in CHANNELS}
        summaries.append({'model': model, 'seeds': sorted(c['seed'] for c in selected),
                          'mean_seed_methods': methods,
                          'mean_seed_method_contrast_difference_method2_minus_method1': float(np.mean([
                              c['method_contrast_difference_method2_minus_method1'] for c in selected]))})
    return {'experimental_proteins': 1, 'experimental_families': 1,
            'inferential_tests': [], 'population_intervals': [], 'winner_selection': False,
            'models': summaries,
            'sign_counts_mean_seed': {channel: {
                sign: sum((r['mean_seed_methods'][channel]['difference_BMPL_minus_BPL'] > 0 if sign == 'positive'
                           else r['mean_seed_methods'][channel]['difference_BMPL_minus_BPL'] < 0 if sign == 'negative'
                           else r['mean_seed_methods'][channel]['difference_BMPL_minus_BPL'] == 0)
                          for r in summaries) for sign in ('positive', 'negative', 'zero')}
                for channel in CHANNELS}}


def verify_fold_records(records, projected, family):
    actual = [[r['fold'], r['held_families'], r['training_families'],
               [i['validation_families'] for i in r['inner_folds']]] for r in records]
    require(actual == projected, 'original held-family fold mismatch')
    held = [r['fold'] for r in records if family in r['held_families']]
    require(len(held) == 1 and all(family not in r['training_families'] for r in records
                                 if r['fold'] == held[0]), 'family leaked into held-fold training')
    return held[0]
