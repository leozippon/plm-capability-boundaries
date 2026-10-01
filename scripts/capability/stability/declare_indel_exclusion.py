#!/usr/bin/env python3
"""Declare which frozen stability cycles rest on an insertion or deletion construct.

The frozen cohort was admitted before this defect was found. An indel construct's
source ``aa_seq`` is truncated to the wild type's length, so such a row matches a
four-state substitution cycle on sequence alone and was admitted as one; the
cohort builder now refuses those rows, but the admitted fit records were produced
on the support that still holds them.

This entry point names them once, from the pinned measurement files and the frozen
cohort's own retained source-row indices, and writes a digest-bound exclusion the
stratified refit binds to. It reads the ``mut_type`` construct annotation only: no
stability value, confidence width or fitted quantity enters the decision, so the
exclusion is a property of the construct rather than of any outcome.

The exclusion is a filter on the evaluation of an already fitted prediction, not a
rebuilt cohort. Rebuilding the cohort with indel rows removed before the per-state
medians would change the target values and the admitted support, which is a
different cohort and not what this declares.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.stability.pairwise_stability import INDEL_MUT_TYPE_PREFIXES, indel_construct

SCHEMA = 'pairwise_indel_exclusion_v1'
EXCLUSION_BASENAME = 'indel_exclusion.json'


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b''):
            sha.update(chunk)
    return sha.hexdigest()


def content_digest(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def indel_source_rows(cohort: dict, root: Path) -> tuple[set[int], dict[int, str], list[dict]]:
    """Global source-row indices whose construct is an insertion or deletion."""

    import pandas as pd

    rows: set[int] = set()
    names: dict[int, str] = {}
    files = []
    offset = 0
    for relative in cohort['source_row_order']:
        path = root / relative
        declared = next(record for record in cohort['source_files']
                        if record['path'] == relative)
        found = digest(path)
        if found != declared['sha256']:
            raise SystemExit(f'{relative} does not match the digest the cohort was built from')
        frame = pd.read_parquet(path, columns=['mut_type'])
        flags = indel_construct(frame['mut_type'])
        for local, flag in enumerate(flags):
            if flag:
                rows.add(offset + local)
                names[offset + local] = str(frame['mut_type'].iloc[local])
        files.append({'path': relative, 'sha256': found, 'rows': int(len(frame))})
        offset += len(frame)
    return rows, names, files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--expect-cohort-sha256', required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, required=True,
                        help='output directory; the exclusion is written into it by name')
    args = parser.parse_args()

    cohort = json.loads(args.cohort.read_text())
    if digest(args.cohort) != args.expect_cohort_sha256:
        raise SystemExit('the cohort digest does not match the declared value')
    rows, names, files = indel_source_rows(cohort, args.data_root)

    excluded: dict[str, list[int]] = {}
    evidence: dict[str, list[dict]] = {}
    retained_cycles = retained_pairs = 0
    retained_groups: set[str] = set()
    for background in cohort['backgrounds']:
        by_state = {sequence: record['rows']
                    for sequence, record in background['measurements'].items()}
        kept_pairs = set()
        for index, cycle in enumerate(background['cycles']):
            carriers = [
                {'state': position, 'source_row': int(record['source_row']),
                 'row_name': record['name'], 'mut_type': names[int(record['source_row'])]}
                for position, state in enumerate(cycle['sequences'])
                for record in by_state[state] if int(record['source_row']) in rows]
            if carriers:
                excluded.setdefault(background['name'], []).append(index)
                evidence.setdefault(background['name'], []).append(
                    {'cycle': index, 'positions': list(cycle['positions']),
                     'carriers': carriers[:4], 'carrier_rows': len(carriers)})
                continue
            retained_cycles += 1
            retained_groups.add(background['group'])
            kept_pairs.add(tuple(cycle['positions']))
        retained_pairs += len(kept_pairs)

    record = {
        'schema': SCHEMA,
        'rule': (f'a cycle is excluded when any source row contributing to any of its four '
                 f'states carries a mut_type beginning with one of {list(INDEL_MUT_TYPE_PREFIXES)}; '
                 'the annotation is the construct identity and no measured value enters it'),
        'scope': ('an evaluation filter over already fitted held-out predictions on identical '
                  'folds, seeds and weighting: a retained cycle keeps its target and its weight, '
                  'and an excluded cycle leaves the group-equal average. Excluded rows remain in '
                  'the training side of the cross-validated fit, so this bounds the contribution '
                  'of indel-sourced cycles to the measured increment and not their contribution '
                  'to the fitted coefficients'),
        'inputs': {'cohort': {'path': str(args.cohort), 'sha256': args.expect_cohort_sha256},
                   'measurement_files': files},
        'code_sha256': {str(path.relative_to(ROOT)): digest(path) for path in (
            Path(__file__), ROOT / 'src/capability/stability/pairwise_stability.py')},
        'indel_source_rows': len(rows),
        'excluded_cycles': {name: sorted(indices) for name, indices in sorted(excluded.items())},
        'evidence': {name: entries for name, entries in sorted(evidence.items())},
        'summary': {
            'excluded_cycles': sum(len(indices) for indices in excluded.values()),
            'affected_backgrounds': len(excluded),
            'cohort_cycles': cohort['summary']['cycles'],
            'retained_cycles': retained_cycles,
            'retained_groups': len(retained_groups),
            'retained_site_pairs': retained_pairs,
            'groups_lost_entirely': sorted(
                background['group'] for background in cohort['backgrounds']
                if len(excluded.get(background['name'], [])) == len(background['cycles'])),
        },
    }
    record['exclusion_sha256'] = content_digest(
        {key: record[key] for key in ('schema', 'rule', 'excluded_cycles', 'summary')})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / EXCLUSION_BASENAME).write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'out': str(args.out / EXCLUSION_BASENAME),
                      'exclusion_sha256': record['exclusion_sha256'],
                      'summary': record['summary']}, indent=1))


if __name__ == '__main__':
    main()
