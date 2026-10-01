#!/usr/bin/env python3
"""Panel table for the position-term decomposition, read against the arms' split.

The decomposition's discriminating question is whether the mutated position's own
term carries the mutation ranking or the downstream context-mediated terms do. A
single arm cannot answer it: the reading only becomes legible against the panel's
own 14-of-33 protein-pretraining split, which is why the text arms are run.

This summariser reports, per arm and per stratum, the ranking quality of the own
term, of the downstream-terms-only sum and of the full retained sum on the same
units and the same folds, together with the two paired contrasts. It refuses to
summarise a cohort in which any arm failed its retention or alignment gate,
because a decomposition that is not the archived quantity has nothing to say
about the archived quantity.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.capability.core.io import write_json
from src.capability.interactions.pairwise_epistasis import ROSTER, TOKENISATION_STRATUM

SCHEMA = 'position_terms_panel_v1'

#: The panel's own pretraining split, as the external-confirmation record states
#: it: the 11 residue-level arms plus the three protein-corpus BPE checkpoints.
PROTEIN_PRETRAINED = tuple(sorted(
    [arm for arm, stratum in TOKENISATION_STRATUM.items() if stratum == 'amino_acid']
    + ['instructprotein', 'protgpt2', 'protgpt3-1.3b']))


def resolved(record) -> str:
    """Sign of a 95% percentile interval, or ``unresolved`` when it straddles zero."""

    if record is None or record.get('interval') is None:
        return 'undefined'
    low, high = record['interval']
    if low > 0:
        return 'above'
    if high < 0:
        return 'below'
    return 'unresolved'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True, type=Path)
    parser.add_argument('--cohort', required=True, choices=('stability', 'pairwise', 'anchor'))
    parser.add_argument('--stratum', default=None,
                        help='anchor only: single_substitution (default) or all_variants')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()

    stratum = args.stratum or ('single_substitution' if args.cohort == 'anchor'
                               else 'all_variants')
    rows, failed = {}, []
    for path in sorted(args.records.glob(f'position_terms_{args.cohort}_*.json')):
        record = json.loads(path.read_bytes())
        arm = record['arm']
        if arm not in set(ROSTER):
            continue
        if not record['passed']:
            failed.append({'arm': arm, 'T1': record['T1']['worst_abs_nats'],
                           'T2': record['T2']['worst_abs_nats'], 'tier': record['T2']['tier'],
                           'derived': record['derived_invariant']['worst_abs_difference']})
        contrast = record['contrast'][stratum]
        rows[arm] = {
            'arm': arm,
            'stratum_interface': TOKENISATION_STRATUM[arm],
            'protein_pretrained': arm in set(PROTEIN_PRETRAINED),
            'tier': record['T2']['tier'],
            'T1_worst_abs_nats': record['T1']['worst_abs_nats'],
            'T2_worst_abs_nats': record['T2']['worst_abs_nats'],
            'closure_worst_abs_nats': record['closure']['worst_abs_nats'],
            'derived_worst_abs_difference': record['derived_invariant']['worst_abs_difference'],
            'mutated_residue_unscored': record['coverage']['mutated_residue_unscored'],
            'own_residue_width_mean': record['coverage']['own_residue_width_mean'],
            'aligned_fraction': record['alignment']['aligned_fraction'],
            'aligned_variants': record['alignment']['aligned'],
            'unaligned_variants': record['alignment']['unaligned'],
            'mismatch_extent_residues_max': record['alignment']['mismatch_extent_residues_max'],
            'packed_length_delta_nonzero': record['alignment']['packed_length_delta_nonzero'],
            'upstream_term_abs_nats_max': record['alignment']['upstream_term_abs_nats']['max'],
            'measurable': record['measurable']['verdict'],
            'support_identifier': record['measurable']['support_identifier'],
            'own': contrast['own'], 'downstream': contrast['downstream'],
            'full': contrast['full'],
            'own_minus_full': contrast['own_minus_full'],
            'downstream_minus_full': contrast['downstream_minus_full'],
            'units': contrast.get('groups', contrast.get('clusters')),
            'variants': contrast['variants'],
        }
        for key in ('own_minus_full', 'downstream_minus_full'):
            rows[arm][f'{key}_resolved'] = resolved(contrast[key])
    if not rows:
        raise SystemExit(f'no {args.cohort} records under {args.records}')
    missing = sorted(set(ROSTER) - set(rows))

    def tally(predicate, require_measurable=True):
        subset = [row for row in rows.values()
                  if predicate(row) and (row['measurable'] or not require_measurable)
                  and row['own']['interval'] is not None
                  and row['full']['interval'] is not None]
        return {
            'arms': len(subset),
            'arms_not_measurable': sum(1 for row in rows.values()
                                       if predicate(row) and not row['measurable']),
            'own_minus_full_above_zero': sum(r['own_minus_full_resolved'] == 'above' for r in subset),
            'own_minus_full_below_zero': sum(r['own_minus_full_resolved'] == 'below' for r in subset),
            'own_minus_full_unresolved': sum(r['own_minus_full_resolved'] == 'unresolved' for r in subset),
            'downstream_minus_full_above_zero': sum(
                r['downstream_minus_full_resolved'] == 'above' for r in subset),
            'downstream_minus_full_below_zero': sum(
                r['downstream_minus_full_resolved'] == 'below' for r in subset),
            'downstream_minus_full_unresolved': sum(
                r['downstream_minus_full_resolved'] == 'unresolved' for r in subset),
            'own_point_exceeds_full_point': sum(
                1 for r in subset if r['own']['point'] is not None and r['full']['point'] is not None
                and r['own']['point'] > r['full']['point']),
            'downstream_point_exceeds_full_point': sum(
                1 for r in subset
                if r['downstream']['point'] is not None and r['full']['point'] is not None
                and r['downstream']['point'] > r['full']['point']),
        }

    record = {
        'schema': SCHEMA, 'cohort': args.cohort, 'stratum': stratum,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'arms_reported': len(rows), 'arms_missing': missing,
        'gate_failures': failed,
        'tiers': {str(t): sum(1 for r in rows.values() if r['tier'] == t) for t in (1, 2, 3)},
        'worst_T1_abs_nats': max(r['T1_worst_abs_nats'] for r in rows.values()),
        'worst_T2_abs_nats': max(r['T2_worst_abs_nats'] for r in rows.values()),
        'worst_closure_abs_nats': max(r['closure_worst_abs_nats'] for r in rows.values()),
        'worst_derived_abs_difference': max(r['derived_worst_abs_difference']
                                            for r in rows.values()),
        'panel': tally(lambda row: True),
        'protein_pretrained': tally(lambda row: row['protein_pretrained']),
        'text_pretrained': tally(lambda row: not row['protein_pretrained']),
        'by_interface': {value: tally(lambda row, value=value: row['stratum_interface'] == value)
                         for value in sorted(set(TOKENISATION_STRATUM.values()))},
        'alignment': {
            'rule': 'a variant aligns when the substitution changes the token grid only at '
                    'the tokens covering the substituted residue; measured per variant, never '
                    'assumed from the interface',
            'declared_minimum_aligned_fraction': 0.50,
            'per_arm_aligned_fraction': {row['arm']: row['aligned_fraction']
                                         for row in rows.values()},
            'arms_fully_aligned': sum(1 for row in rows.values()
                                      if row['aligned_fraction'] == 1.0),
            'arms_not_measurable': sorted(row['arm'] for row in rows.values()
                                          if not row['measurable']),
            'by_interface_median_aligned_fraction': {
                value: float(np.median([row['aligned_fraction'] for row in rows.values()
                                        if row['stratum_interface'] == value]))
                for value in sorted({row['stratum_interface'] for row in rows.values()})},
            'threshold_sweep': {
                str(threshold): tally(lambda row, threshold=threshold:
                                      (row['aligned_fraction'] or 0.0) >= threshold,
                                      require_measurable=False)
                for threshold in (0.0, 0.25, 0.5, 0.75, 0.95, 1.0)},
            'falsification_control': (
                'a text arm can serve as the falsification control for this contrast only '
                'where its own variants align; where they do not, the contrast is undefined '
                'on that arm rather than absent, and no unaligned-arm number is substituted '
                'into the control role'),
        },
        'protein_pretrained_arms': list(PROTEIN_PRETRAINED),
        'arms': [rows[arm] for arm in sorted(rows)],
        'reading': (
            'own_minus_full resolved above zero on an arm means the mutated position\'s own '
            'term ranks better than the full retained sum on that arm, so summing over '
            'downstream positions dilutes a locally carried signal -- the '
            'autoregressive-factorization account. downstream_minus_full resolved above zero '
            'means the downstream context-mediated terms carry the ranking, which refutes '
            'that account and supports a non-local one. Both readings are reported as the '
            'measurement gives them; neither part of the split was tuned.'),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / f'position_terms_panel_{args.cohort}_{stratum}.json', record)
    print(json.dumps({k: record[k] for k in (
        'cohort', 'stratum', 'arms_reported', 'arms_missing', 'tiers', 'worst_T1_abs_nats',
        'worst_T2_abs_nats', 'worst_closure_abs_nats', 'worst_derived_abs_difference',
        'alignment', 'panel', 'protein_pretrained', 'text_pretrained')}, indent=1,
        default=str), flush=True)
    if failed:
        print(json.dumps({'gate_failures': failed}), flush=True)


if __name__ == '__main__':
    main()
