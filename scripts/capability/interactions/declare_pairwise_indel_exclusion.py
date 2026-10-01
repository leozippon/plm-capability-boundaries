#!/usr/bin/env python3
"""Declare the insertion/deletion-construct exclusion over the frozen pairwise cohort.

An indel construct's `aa_seq` is truncated to the wild type's length in the
pinned MegaScale dataset2 bytes, so it is length-matched to the substitution
states and entered the frozen cohort `8133463e...` as though it were a
substitution. `src.capability.stability.pairwise_stability.indel_construct` is the rule that
identifies such a row, and the cohort builder now applies it; the frozen cohort
predates that fix and is not re-frozen, because several gates reference its
digest as their declared support.

This stage therefore writes the exclusion as a declared filter over that frozen
cohort rather than a new cohort. For every state of every cohort background it
reports which contributing rows are indel constructs, the state's median value
over the remaining rows, and `null` where no row remains. A consumer applies it
as a sensitivity beside the unfiltered support and reports both.

This declaration is state-level and is consumed by a refit: it carries each
affected state's median over its remaining accepted rows, so a consumer can
correct the two states that keep a value, drop the three that keep none, and
refit on the corrected rows. `declare_indel_exclusion.py` declares the
complementary cycle-level list, which filters the evaluation of the already
fitted admitted predictions without changing a target value. The two answer
different questions and neither replaces the other; both read the same rule from
`src.capability.stability.pairwise_stability.indel_construct`.

It reads measured values, so it is not label blind and must not be run on a
support chosen after seeing a model outcome. It does no model inference, draws
nothing and selects nothing: every state of the frozen cohort is inspected.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.stability.pairwise_stability import (  # noqa: E402
    COHORT_SCHEMA, INDEL_MUT_TYPE_PREFIXES, QC_WIDTH_KCAL_MOL, indel_construct)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b''):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--expect-cohort-sha256', required=True)
    parser.add_argument('--parquet-manifest', type=Path,
                        default=ROOT / 'data/megascale_tsuboyama2023/manifest.json')
    parser.add_argument('--parquet-root', type=Path,
                        default=ROOT / 'data/megascale_tsuboyama2023')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    import pandas as pd

    if digest(args.cohort) != args.expect_cohort_sha256:
        raise SystemExit('cohort digest does not match the declared value')
    manifest = json.loads(args.parquet_manifest.read_text())
    files = []
    for entry in manifest['verification']['files']:
        path = args.parquet_root / entry['path']
        found = digest(path)
        if found != entry['sha256']:
            raise SystemExit(f'{path} does not match its pinned digest')
        files.append({'path': entry['path'], 'sha256': found})
    frame = pd.concat([pd.read_parquet(args.parquet_root / entry['path']) for entry in files],
                      ignore_index=True)
    indel = indel_construct(frame['mut_type'])
    mut_type = frame['mut_type'].astype(str).to_numpy()

    cohort = json.loads(args.cohort.read_text())
    if cohort.get('schema') != COHORT_SCHEMA:
        raise SystemExit('unexpected cohort schema')
    states, inspected = [], 0
    for background in cohort['backgrounds']:
        for sequence, measurement in background['measurements'].items():
            inspected += 1
            rows = measurement['rows']
            if max(row['source_row'] for row in rows) >= len(frame):
                raise SystemExit('a cohort row index lies outside the pinned files')
            flags = [bool(indel[row['source_row']]) for row in rows]
            if not any(flags):
                continue
            kept = [row['value'] for row, flag in zip(rows, flags) if not flag]
            states.append({
                'background': background['name'], 'group': background['group'],
                'sequence': sequence, 'rows': len(rows), 'indel_rows': int(sum(flags)),
                'mut_types': sorted({mut_type[row['source_row']]
                                     for row, flag in zip(rows, flags) if flag}),
                'value_kcal_mol': measurement['value'],
                'value_without_indel_rows_kcal_mol':
                    float(np.median(kept)) if kept else None,
            })
    record = {
        'schema': 'pairwise_indel_exclusion_v1',
        'cohort_sha256': args.expect_cohort_sha256,
        'parquet': files,
        'rule': ('a contributing row whose mut_type begins with one of '
                 f'{list(INDEL_MUT_TYPE_PREFIXES)} names an insertion or deletion construct '
                 'whose aa_seq is truncated to the wild type length in the pinned files, so it '
                 'is not a measured substitution and is excluded; a state keeps the median of '
                 'its remaining accepted rows, and a state with no remaining row is absent'),
        'aggregation': ('median over the accepted rows the frozen cohort retained for the '
                        f'state, on the same admission rule of width at most {QC_WIDTH_KCAL_MOL} '
                        'kcal/mol per channel'),
        'indel_rows_in_pinned_files': int(indel.sum()),
        'states_inspected': inspected,
        'states': states,
        'accounting': {
            'states_with_an_indel_row': len(states),
            'states_absent_after_exclusion': sum(
                1 for state in states if state['value_without_indel_rows_kcal_mol'] is None),
            'states_whose_value_moves': sum(
                1 for state in states if state['value_without_indel_rows_kcal_mol'] is not None
                and state['value_without_indel_rows_kcal_mol'] != state['value_kcal_mol']),
            'largest_state_value_move_kcal_mol': max(
                [abs(state['value_without_indel_rows_kcal_mol'] - state['value_kcal_mol'])
                 for state in states if state['value_without_indel_rows_kcal_mol'] is not None],
                default=0.0),
            'backgrounds_touched': sorted({state['background'] for state in states}),
        },
        'not_label_blind': ('this record reads measured stabilities; it is a measurement '
                            'correction declared over a frozen support, not a label-blind step'),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({'out': str(args.out), 'sha256': digest(args.out),
                      **record['accounting']}, indent=1))


if __name__ == '__main__':
    main()
