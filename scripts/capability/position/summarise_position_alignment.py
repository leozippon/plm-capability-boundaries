#!/usr/bin/env python3
"""Panel table for whether a point substitution preserves an arm's token positions.

The per-arm records this reads are the measurement; this is the table they are
read as. It is deliberately separate from the decomposition summary and is
reported before it, because the alignment rate decides which arms can carry a
position-resolved reading at all, and it is a property of the interface rather
than of any model.

Two groupings are reported and neither is assumed. The repository's own
tokenisation stratum is one of them, and the table shows where that label and the
measurement disagree; the panel's protein-pretraining split is the other, and the
table shows that the arms which preserve position are not the arms that split.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.core.io import write_json
from src.capability.interactions.pairwise_epistasis import ROSTER, TOKENISATION_STRATUM

SCHEMA = 'position_alignment_panel_v1'
PROTEIN_PRETRAINED = tuple(sorted(
    [arm for arm, stratum in TOKENISATION_STRATUM.items() if stratum == 'amino_acid']
    + ['instructprotein', 'protgpt2', 'protgpt3-1.3b']))
COHORTS = ('stability', 'pairwise', 'anchor')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()

    rows: dict[str, dict] = {}
    for path in sorted(args.records.glob('alignment_*.json')):
        record = json.loads(path.read_bytes())
        rows.setdefault(record['arm'], {})[record['cohort']] = record
    missing = sorted(set(ROSTER) - set(rows))
    if missing:
        raise SystemExit(f'the alignment records do not cover the roster; missing {missing}')

    arms = []
    for arm in sorted(rows):
        entry = {'arm': arm, 'declared_interface': TOKENISATION_STRATUM[arm],
                 'protein_pretrained': arm in set(PROTEIN_PRETRAINED)}
        for cohort in COHORTS:
            record = rows[arm][cohort]
            entry[cohort] = {
                'variants': record['variants'],
                'aligned': record['aligned'],
                'aligned_fraction': record['aligned_fraction'],
                'aligned_fraction_single_substitution':
                    record['aligned_fraction_single_substitution'],
                'unaligned_with_unscored_substitution':
                    record['unaligned_with_unscored_substitution'],
                'packed_length_delta_nonzero': record['packed_length_delta_nonzero'],
                'mismatch_extent_residues_max': record['mismatch_extent_residues_max'],
                'unaligned_reasons': record['unaligned_reasons'],
            }
        entry['fully_aligned_all_cohorts'] = all(
            entry[cohort]['aligned_fraction'] == 1.0 for cohort in COHORTS)
        entry['position_preserving_grid'] = all(
            entry[cohort]['mismatch_extent_residues_max'] == 0
            and entry[cohort]['packed_length_delta_nonzero'] == 0 for cohort in COHORTS)
        arms.append(entry)

    def fractions(cohort, predicate=lambda entry: True):
        return [entry[cohort]['aligned_fraction'] for entry in arms if predicate(entry)]

    panel = {}
    for cohort in COHORTS:
        values = fractions(cohort)
        panel[cohort] = {
            'arms': len(values),
            'median_aligned_fraction': float(np.median(values)),
            'range_aligned_fraction': [float(min(values)), float(max(values))],
            'arms_fully_aligned': sum(1 for value in values if value == 1.0),
            'arms_at_or_above_0.99': sum(1 for value in values if value >= 0.99),
            'arms_at_or_above_0.50': sum(1 for value in values if value >= 0.50),
            'arms_below_0.50': sorted(entry['arm'] for entry in arms
                                      if entry[cohort]['aligned_fraction'] < 0.50),
            'by_declared_interface': {
                stratum: {
                    'arms': len(fractions(cohort, lambda e, s=stratum: e['declared_interface'] == s)),
                    'median': float(np.median(fractions(
                        cohort, lambda e, s=stratum: e['declared_interface'] == s))),
                    'range': [float(min(fractions(
                        cohort, lambda e, s=stratum: e['declared_interface'] == s))),
                              float(max(fractions(
                                  cohort, lambda e, s=stratum: e['declared_interface'] == s)))],
                    'fully_aligned': sum(1 for e in arms if e['declared_interface'] == stratum
                                         and e[cohort]['aligned_fraction'] == 1.0)}
                for stratum in sorted(set(TOKENISATION_STRATUM.values()))},
            'by_pretraining': {
                key: {'arms': len(fractions(cohort, lambda e, k=key: e['protein_pretrained'] == k)),
                      'median': float(np.median(fractions(
                          cohort, lambda e, k=key: e['protein_pretrained'] == k))),
                      'fully_aligned': sum(1 for e in arms if e['protein_pretrained'] == key
                                           and e[cohort]['aligned_fraction'] == 1.0)}
                for key in (True, False)},
        }

    fully = sorted(entry['arm'] for entry in arms if entry['fully_aligned_all_cohorts'])
    preserving = sorted(entry['arm'] for entry in arms if entry['position_preserving_grid'])
    label_disagreement = sorted(
        entry['arm'] for entry in arms
        if entry['declared_interface'] == 'amino_acid' and not entry['position_preserving_grid'])
    record = {
        'schema': SCHEMA, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'arms_reported': len(arms), 'cohorts': list(COHORTS),
        'rule': rows[arms[0]['arm']]['stability']['rule'],
        'fully_aligned_all_cohorts': fully,
        'position_preserving_grid': preserving,
        'declared_amino_acid_but_not_position_preserving': label_disagreement,
        'panel': panel,
        'usable_as_falsification_control': sorted(
            entry['arm'] for entry in arms
            if not entry['protein_pretrained'] and entry['position_preserving_grid']),
        'reading': (
            'A variant aligns when the substitution changes the token grid only at the tokens '
            'covering the substituted residue. Where it does not, no unique correspondence '
            'between a mutant position and a wild-type position exists, so the '
            'site-against-downstream split has no referent for that variant and is excluded '
            'from the contrast with the exclusion counted. Two distinctions the table keeps '
            'apart: a token grid that never moves but leaves the substituted residue outside '
            'the scored span, which is a scoring-window limit, and a grid that moves away from '
            'the substitution, which is a retokenisation limit.'),
        'arms': arms,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'position_alignment_panel.json', record)
    print(json.dumps({k: record[k] for k in (
        'arms_reported', 'fully_aligned_all_cohorts', 'position_preserving_grid',
        'declared_amino_acid_but_not_position_preserving',
        'usable_as_falsification_control', 'panel')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
