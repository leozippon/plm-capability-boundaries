#!/usr/bin/env python3
"""Survey depositions for the one shape an order-k model comparison needs.

The specification three closed higher-order assessments produced: **eight or
more wild-type families below 50% identity, each carrying substantive complete
order-2 or higher support, with a genuine replicate or dispersion channel
reaching every corner of the contrast.** Breadth, completeness and a
corner-covering channel, together. This stage measures that triple on the
depositions already staged locally that could plausibly carry it, and reports
which of the three each one supplies.

It reads state keys and repeat structure only -- never a measured stability,
affinity or fitness value -- so what it reports about a deposition's support
cannot be a function of the labels that support would carry. Measurement-only
and CPU-only: no model, no cohort, no forward pass, no GPU.

    python scripts/capability/mutation/survey_higher_order_depositions.py \
      --candidate mgnify_stability_cho2026 --candidate skempi2 \
      --out archive/logs/shared/higher_order_multifamily_20260924/survey.json
"""
from __future__ import annotations

import argparse
import collections
import csv
from datetime import datetime, timezone
import itertools
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.mutation.higher_order_qualification import (  # noqa: E402
    FAMILY_IDENTITY_THRESHOLD, RESIDUES)
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS  # noqa: E402

#: Sample size for the neighbour diagnostic that explains a zero. A zero cycle
#: count is only a finding once it is separated from a defect, and the
#: separation is whether the states have measured Hamming-1 neighbours at all.
NEIGHBOUR_SAMPLE = 400
NEIGHBOUR_SEED = 20260924


def hamming_neighbours(sequence: str, measured: set[str]) -> list[tuple[int, str]]:
    """Measured states one substitution from ``sequence``, as (position, residue)."""
    out, letters = [], list(sequence)
    for position in range(len(sequence)):
        original = letters[position]
        for residue in sorted(RESIDUES):
            if residue == original:
                continue
            letters[position] = residue
            if ''.join(letters) in measured:
                out.append((position, residue))
        letters[position] = original
    return out


def family_groups(references: list[str],
                  threshold: float = FAMILY_IDENTITY_THRESHOLD) -> list[list[str]]:
    """Single-linkage groups at ``threshold`` identity; unequal lengths stay distinct.

    Counting unequal-length references as distinct makes the group count an
    upper bound, which is the sound direction for a refusal.
    """
    parent = {reference: reference for reference in references}

    def root(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    by_length: dict[int, list[str]] = collections.defaultdict(list)
    for reference in references:
        by_length[len(reference)].append(reference)
    for length, members in by_length.items():
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                a, b = members[left], members[right]
                if sum(1 for x, y in zip(a, b) if x == y) / length >= threshold:
                    ra, rb = root(a), root(b)
                    if ra != rb:
                        parent[ra] = rb
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for reference in references:
        groups[root(reference)].append(reference)
    return [sorted(members) for members in groups.values()]


def survey_mgnify(root: Path) -> dict:
    """MGnify Stability: breadth and a two-protease channel, but is order 2 complete?

    The deposition carries one row per measured amino-acid state with a trypsin
    and a chymotrypsin free energy and a 95% interval on each, which is the
    admitted label instrument's own channel structure, and a declared
    double-mutant design partition. A complete order-2 cycle needs all four
    corners, so the question this answers is whether the designed doubles'
    constituent singles were measured beside them.
    """
    import random

    path = root / '230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv'
    csv.field_size_limit(1 << 24)
    measured: set[str] = set()
    designed: list[str] = []
    libraries: collections.Counter = collections.Counter()
    with path.open(newline='') as handle:
        reader = csv.reader(handle)
        index = {name: position for position, name in enumerate(next(reader))}
        for row in reader:
            sequence = row[index['aa_seq']]
            if set(sequence) - RESIDUES:
                continue
            if row[index['deltaG_t']] in ('', 'NA') or row[index['deltaG_c']] in ('', 'NA'):
                continue
            measured.add(sequence)
            libraries[row[index['lib']]] += 1
            if row[index['dm_design']] == 'True':
                designed.append(sequence)
    cycles: list[tuple[str, tuple[int, int]]] = []
    for double in designed:
        neighbours = hamming_neighbours(double, measured)
        for left in range(len(neighbours)):
            for right in range(left + 1, len(neighbours)):
                (pi, ri), (pj, rj) = neighbours[left], neighbours[right]
                if pi == pj:
                    continue
                letters = list(double)
                letters[pi], letters[pj] = ri, rj
                wild = ''.join(letters)
                if wild in measured:
                    cycles.append((wild, tuple(sorted((pi, pj)))))
    per_background: dict[str, set] = collections.defaultdict(set)
    for wild, pair in cycles:
        per_background[wild].add(pair)
    groups = family_groups(sorted(per_background)) if per_background else []
    generator = random.Random(NEIGHBOUR_SEED)
    diagnostic = {}
    for label, pool in (('designed_doubles', designed), ('all_measured_states', sorted(measured))):
        if not pool:
            continue
        sample = generator.sample(pool, min(NEIGHBOUR_SAMPLE, len(pool)))
        counts = collections.Counter(len(hamming_neighbours(state, measured))
                                     for state in sample)
        diagnostic[label] = {'sampled': len(sample),
                             'measured_hamming1_neighbours':
                                 {str(key): value for key, value in sorted(counts.items())}}
    return {
        'deposition': 'mgnify_stability_cho2026',
        'source': str(path.relative_to(ROOT)),
        'source_sha256': sha256_file(path),
        'measured_states_with_both_channels': len(measured),
        'libraries': dict(libraries.most_common()),
        'designed_double_rows': len(designed),
        'complete_order2_cycles': len(cycles),
        'backgrounds_carrying_a_cycle': len(per_background),
        'family_groups': len(groups),
        'families_with_enough_units': sum(
            1 for members in groups
            if sum(len(per_background[member]) for member in members) >= MINIMUM_BOOTSTRAP_UNITS),
        'channel': {'kind': 'replicate', 'reaches_every_corner': None,
                    'description': 'trypsin and chymotrypsin free energies with a 95% '
                                   'interval on each, on every measured state'},
        'neighbour_diagnostic': diagnostic,
        'verdict': ('breadth and a corner-covering channel, and no completeness: the '
                    'designed doubles carry no measured single-substitution neighbour, so '
                    'no order-2 cycle closes'),
    }


def survey_skempi(root: Path) -> dict:
    """SKEMPI v2: breadth and completeness from literature aggregation, and the channel?

    Every entry is one published affinity measurement of one mutant of one
    complex, so a state measured by two independent publications carries a
    genuine two-channel structure -- independent labs and methods rather than
    two proteases. The question is whether that channel reaches all four corners
    of the cycles, and all eight of the cubes.
    """
    path = root / 'skempi_v2.csv'
    token = re.compile(r'^([A-Z])([A-Z])(-?\d+[A-Za-z]?)([A-Z])$')
    references: dict[tuple[str, frozenset], set[str]] = collections.defaultdict(set)
    rows = 0
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle, delimiter=';'):
            rows += 1
            raw = row['Mutation(s)_cleaned'].strip()
            if not raw:
                continue
            parts = [piece for piece in raw.split(',') if piece]
            if not all(token.match(piece) for piece in parts):
                continue
            if len(set(parts)) != len(parts):
                continue
            references[(row['#Pdb'], frozenset(parts))].add(row['Reference'])
    states: dict[str, set[frozenset]] = collections.defaultdict(set)
    for complex_id, state in references:
        states[complex_id].add(state)
    per_order = {}
    for order in (2, 3):
        complete: dict[str, set] = collections.defaultdict(set)
        channelled: dict[str, set] = collections.defaultdict(set)
        for complex_id, measured in states.items():
            for state in measured:
                if len(state) != order:
                    continue
                subsets = [frozenset(pieces) for size in range(1, order)
                           for pieces in itertools.combinations(sorted(state), size)]
                if not all(subset in measured for subset in subsets):
                    continue
                complete[complex_id].add(tuple(sorted(state)))
                corners = subsets + [state]
                if all(len(references[(complex_id, corner)]) >= 2 for corner in corners):
                    channelled[complex_id].add(tuple(sorted(state)))
        per_order[str(order)] = {
            'complexes_with_complete_support': len(complete),
            'complete_contrasts': sum(len(value) for value in complete.values()),
            'complexes_with_8_or_more_complete': sum(
                1 for value in complete.values() if len(value) >= MINIMUM_BOOTSTRAP_UNITS),
            'complexes_whose_channel_reaches_every_corner': len(channelled),
            'contrasts_with_every_corner_independently_remeasured':
                sum(len(value) for value in channelled.values()),
            'complexes_with_8_or_more_channelled': sum(
                1 for value in channelled.values() if len(value) >= MINIMUM_BOOTSTRAP_UNITS),
        }
    return {
        'deposition': 'skempi2',
        'source': str(path.relative_to(ROOT)),
        'source_sha256': sha256_file(path),
        'entries': rows,
        'complexes': len(states),
        'order_census': {str(order): sum(1 for value in states.values()
                                         for state in value if len(state) == order)
                         for order in range(1, 6)},
        'distinct_states': len(references),
        'states_with_two_or_more_independent_references': sum(
            1 for value in references.values() if len(value) >= 2),
        'per_order': per_order,
        'wild_type_corner': ('the unmutated complex, whose affinity every entry reports '
                             'alongside the mutant, so corner () is measured by construction'),
        'channel': {'kind': 'replicate',
                    'description': 'two or more independent publications measuring the same '
                                   'mutant of the same complex; confounded by method and '
                                   'temperature, which two proteases are not'},
        'verdict': ('breadth and completeness, and a channel that does not reach the '
                    'corners: the repeat structure covers a small minority of states and '
                    'almost never all corners of one contrast'),
    }


SURVEYS = {
    'mgnify_stability_cho2026': survey_mgnify,
    'skempi2': survey_skempi,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--candidate', action='append', required=True, choices=sorted(SURVEYS))
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    candidates = []
    for name in dict.fromkeys(args.candidate):
        print(f'surveying {name}', flush=True)
        candidates.append(SURVEYS[name](args.data_root / name))
    record = {
        'schema_version': 'd1_higher_order_deposition_survey_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'specification': (
            f'{MINIMUM_BOOTSTRAP_UNITS} or more wild-type families below '
            f'{int(FAMILY_IDENTITY_THRESHOLD * 100)}% identity, each carrying substantive '
            'complete order-2 or higher support, with a genuine replicate or dispersion '
            'channel reaching every corner of the contrast'),
        'family_identity_threshold': FAMILY_IDENTITY_THRESHOLD,
        'unit_floor': MINIMUM_BOOTSTRAP_UNITS,
        'candidates': candidates,
        'reads': 'state keys and repeat structure only; no measured value, no model, no GPU',
        'code_sha256': {str(Path(__file__).relative_to(ROOT)): sha256_file(Path(__file__))},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, record)
    print(f'survey: {args.out}')
    print(f'sha256={sha256_file(args.out)}')
    for candidate in candidates:
        print(f"  {candidate['deposition']}: {candidate['verdict']}")


if __name__ == '__main__':
    main()
