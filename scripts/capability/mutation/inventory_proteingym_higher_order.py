#!/usr/bin/env python3
"""Mutation-order census and complete-cube support across the local ProteinGym assays.

Stage ``census`` reads only the ``mutant`` column of every substitution CSV under
the given directory: the order census, and for every variant of order three or
above whether each of its lower orders is completely measured. No ``DMS_score``
value is read, so the support this stage reports cannot be a function of the
labels that support would carry.

The empty corner of a cube is the assay's unmeasured wild type. It enters a cycle
contrast of order k with coefficient ``(-1) ** k``, one value per assay, so it
cancels from within-assay ranks or centred residuals and nowhere else. This stage
therefore reports completeness over the non-empty subsets and names the constant
rather than imputing it.

No model score, likelihood or representation enters any quantity here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.mutation.proteingym_higher_order import AssayStates, parse_mutant  # noqa: E402

#: Minimum complete cubes on one site triple for that triple to be counted as a
#: populated unit, matching the >=10-double rule the pairwise inventory used for
#: site pairs so the two support statements are comparable.
POPULATED_TRIPLE_MINIMUM = 10


def _assay_census(path: Path) -> dict:
    """Order census and higher-order completeness for one assay, keys only."""

    keys: set[tuple] = set()
    wild: dict[int, str] = {}
    rows = 0
    with open(path, newline='', encoding='utf-8') as handle:
        header = handle.readline().rstrip('\n').split(',')
        if header[0] != 'mutant':
            raise ValueError(f'{path.name}: first column is {header[0]!r}, not "mutant"')
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            key = parse_mutant(line.split(',', 1)[0])
            for (position, _), wild_residue in zip(key.substitutions, key.wild):
                seen = wild.setdefault(position, wild_residue)
                if seen != wild_residue:
                    raise ValueError(
                        f'{path.name}: positions {position + 1} declared wild as both '
                        f'{seen} and {wild_residue}')
            if key.substitutions in keys:
                raise ValueError(f'{path.name}: repeats variant {line.split(",", 1)[0]!r}')
            keys.add(key.substitutions)

    assay = AssayStates.of(keys)
    orders = dict(assay.by_order)
    high = sorted(key for key in keys if len(key) >= 3)
    # Completeness is reported two ways: fully complete cubes, and the deepest
    # order at which support is still complete, so a variant that has all its
    # singles but not all its pairs is visible rather than merely incomplete.
    complete_to: Counter[int] = Counter()
    per_order_complete: Counter[int] = Counter()
    triples: Counter[tuple[int, ...]] = Counter()
    for key in high:
        depth = assay.support_depth(key)
        complete_to[depth] += 1
        if depth == len(key) - 1:
            per_order_complete[len(key)] += 1
            if len(key) == 3:
                triples[tuple(position for position, _ in key)] += 1

    return {
        'assay': path.stem,
        'rows': rows,
        'variants': len(keys),
        'wild_positions': len(wild),
        'order_census': {str(order): orders[order] for order in sorted(orders)},
        'max_order': max(orders) if orders else 0,
        'order_ge3': len(high),
        'complete_cubes_by_order': {
            str(order): per_order_complete[order] for order in sorted(per_order_complete)},
        'complete_to_depth': {str(depth): complete_to[depth] for depth in sorted(complete_to)},
        'complete_triples': sum(triples.values()),
        'site_triples_with_complete_cube': len(triples),
        'site_triples_populated': sum(
            1 for count in triples.values() if count >= POPULATED_TRIPLE_MINIMUM),
    }


def run_census(directory: Path, out: Path) -> dict:
    paths = sorted(directory.glob('*.csv'))
    if not paths:
        raise ValueError(f'no substitution CSVs under {directory}')
    assays = [_assay_census(path) for path in paths]

    totals: Counter[int] = Counter()
    complete: Counter[int] = Counter()
    for entry in assays:
        for order, count in entry['order_census'].items():
            totals[int(order)] += count
        for order, count in entry['complete_cubes_by_order'].items():
            complete[int(order)] += count

    report = {
        'schema': 'proteingym_higher_order_census/1',
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'directory': str(directory),
        'populated_triple_minimum': POPULATED_TRIPLE_MINIMUM,
        'wild_type_rows': 0,
        'summary': {
            'assays': len(assays),
            'rows': sum(entry['rows'] for entry in assays),
            'variants': sum(entry['variants'] for entry in assays),
            'order_census': {str(order): totals[order] for order in sorted(totals)},
            'complete_cubes_by_order': {
                str(order): complete[order] for order in sorted(complete)},
            'assays_with_order_ge3': sum(1 for entry in assays if entry['order_ge3']),
            'assays_with_complete_cube_ge3': sum(
                1 for entry in assays if entry['complete_cubes_by_order']),
            'assays_with_populated_site_triple': sum(
                1 for entry in assays if entry['site_triples_populated']),
            'site_triples_with_complete_cube': sum(
                entry['site_triples_with_complete_cube'] for entry in assays),
        },
        'assays': assays,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True)
    out.write_text(payload, encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))
    print(f'digest sha256 {hashlib.sha256(payload.encode()).hexdigest()}')
    print(f'written {out}')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    census_parser = sub.add_parser('census', help='order census and cube completeness')
    census_parser.add_argument(
        '--directory', type=Path,
        default=ROOT / 'data' / 'proteingym' / 'DMS_ProteinGym_substitutions')
    census_parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'census':
        run_census(args.directory, args.out)


if __name__ == '__main__':
    main()
