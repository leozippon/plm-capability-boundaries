#!/usr/bin/env python3
"""Attainability check for the per-position residue attribution, without a GPU.

The decomposition rests on being able to say which residues each scored token
covers. That is a property of one arm's tokeniser and packing, not of its
weights, so it is checkable on a tokeniser alone -- and it must be, because a
coverage failure discovered inside a full extraction wastes the whole cell.

This entry point loads one arm's tokeniser, packs real cohort sequences through
the same ``pack_sequence`` the extraction uses, and requires
:class:`~src.capability.position.position_terms.ResidueCoverage` to reconstruct each
sequence's scored residue suffix twice over. It then reports the properties the
decomposition will be read against: how many residues each interface leaves
unscored, how wide the token covering a mutated residue is, how often a
substitution changes the token grid, and how often the mutated residue falls
outside the scored span altogether.

Sequences are drawn under a seeded permutation of the source's own units and the
draw is repeated at a second skip offset, because taking the first records of a
biological cohort has manufactured an effect in this repository three times.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.core.io import write_json
from src.capability.interactions.pairwise_epistasis import ARM_DTYPE, ROSTER
from src.capability.position.position_terms import ResidueCoverage, partition_masks
from src.capability.readouts.readout_extraction import load_readout_arm, pack_sequence

SCHEMA = 'position_coverage_validation_v1'
DRAW_SEED = 20260926


def _load_stage46():
    spec = importlib.util.spec_from_file_location(
        'stage46', ROOT / 'scripts/capability/stages/context_homologue.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _substitution_sites(wildtype: str, mutant: str) -> list[int]:
    sites = []
    for part in mutant.split(':'):
        match = re.fullmatch(r'([A-Z])(\d+)([A-Z])', part)
        if match is None:
            raise ValueError(f'unparsable substitution {part!r}')
        before, position, _ = match.groups()
        index = int(position) - 1
        if not 0 <= index < len(wildtype) or wildtype[index] != before:
            raise ValueError(f'substitution {part!r} disagrees with its own wild type')
        sites.append(index)
    return sorted(set(sites))


def units(kind: str, source: Path) -> list[dict]:
    """One unit per background or assay: a wild type and its mutated states."""

    payload = json.loads(source.read_bytes())
    rows = []
    if kind in {'stability', 'pairwise'}:
        for row in payload['backgrounds']:
            sequences = row['sequences']
            wildtype = sequences[0]
            states = []
            if kind == 'stability':
                for variant in row['variants']:
                    states.append((int(variant['state']), [int(variant['position']) - 1]))
            else:
                for cycle in row['cycles']:
                    low, high = (int(v) - 1 for v in cycle['positions'])
                    wild, single_low, single_high, double = (int(v) for v in cycle['states'])
                    states.extend([(single_low, [low]), (single_high, [high]),
                                   (double, [low, high])])
            rows.append({'unit': row['name'], 'wildtype': wildtype, 'sequences': sequences,
                         'states': states})
    elif kind == 'anchor':
        for row in payload['assays']:
            wildtype = row['wildtype']
            states = [(index + 1, _substitution_sites(wildtype, mutant))
                      for index, mutant in enumerate(row['mutants'])]
            rows.append({'unit': row['assay'], 'wildtype': wildtype,
                         'sequences': [wildtype] + list(row['sequences']), 'states': states})
    else:
        raise ValueError(f'unknown cohort kind {kind!r}')
    return rows


def check_unit(arm, coverage, unit: dict, *, states_per_unit: int, rng) -> dict:
    """Pack the wild type and a drawn set of its mutated states, and attribute both."""

    wildtype = unit['wildtype']
    sequences = unit['sequences']
    wild_ids, wild_span, _ = pack_sequence(arm, wildtype)
    wild_counts, wild_offset = coverage.counts(wild_ids, wild_span, wildtype)
    order = rng.permutation(len(unit['states']))[:states_per_unit]
    record = {'unit': unit['unit'], 'length': len(wildtype),
              'wild_scored_tokens': int(len(wild_counts)),
              'wild_unscored_residues': int(wild_offset), 'states': []}
    for index in order:
        state, sites = unit['states'][int(index)]
        sequence = sequences[state]
        if len(sequence) != len(wildtype):
            raise ValueError(f"{unit['unit']}: state {state} differs in length from its wild type")
        ids, span, _ = pack_sequence(arm, sequence)
        counts, offset = coverage.counts(ids, span, sequence)
        masks = partition_masks(counts, offset, sites)
        wild_masks = partition_masks(wild_counts, wild_offset, sites)
        record['states'].append({
            'state': int(state), 'sites': [int(s) for s in sites],
            'scored_tokens': int(len(counts)),
            'unscored_residues': int(offset),
            'grid_changed': bool(len(ids) != len(wild_ids)
                                 or list(ids) != list(wild_ids)),
            'token_count_changed': bool(len(ids) != len(wild_ids)),
            'own_tokens': int(masks['own'].sum()),
            'own_residue_width': int(np.asarray(counts, dtype=np.int64)[masks['own']].sum()),
            'own_tokens_wild': int(wild_masks['own'].sum()),
            'mutated_residue_scored': bool(masks['own'].any() and wild_masks['own'].any()),
            'downstream_tokens': int(masks['downstream'].sum()),
            'upstream_tokens': int(masks['upstream'].sum()),
        })
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--cohort', required=True, choices=('stability', 'pairwise', 'anchor'))
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--units', type=int, default=8)
    parser.add_argument('--states-per-unit', type=int, default=6)
    parser.add_argument('--skip-offsets', type=int, nargs='+', default=(0, 17))
    parser.add_argument('--device', default='cpu',
                        help='accepted and ignored: this check loads no weights')
    args = parser.parse_args()

    if args.arm not in ROSTER:
        raise SystemExit(f'{args.arm} is not on the frozen roster')
    stage46 = _load_stage46()
    dtype = ARM_DTYPE.get(args.arm, 'float32')
    arm = load_readout_arm(args.arm, stage46, dtype=dtype)
    coverage = ResidueCoverage(arm)
    rows = units(args.cohort, args.source)
    order = np.random.default_rng([DRAW_SEED, len(rows)]).permutation(len(rows))

    draws = {}
    for skip in args.skip_offsets:
        picked = [rows[int(i)] for i in np.roll(order, -int(skip))[:args.units]]
        rng = np.random.default_rng([DRAW_SEED, int(skip)])
        draws[str(skip)] = [check_unit(arm, coverage, unit, states_per_unit=args.states_per_unit,
                                       rng=rng) for unit in picked]

    flat = [state for unit in draws[str(args.skip_offsets[0])] for state in unit['states']]
    summary = {
        'states_checked': sum(len(u['states']) for d in draws.values() for u in d),
        'unscored_residues': sorted({u['wild_unscored_residues']
                                     for d in draws.values() for u in d}),
        'own_residue_width_max': max((s['own_residue_width'] for s in flat), default=0),
        'own_residue_width_mean': (float(np.mean([s['own_residue_width'] for s in flat]))
                                   if flat else None),
        'grid_changed_fraction': (float(np.mean([s['grid_changed'] for s in flat]))
                                  if flat else None),
        'token_count_changed_fraction': (float(np.mean([s['token_count_changed'] for s in flat]))
                                         if flat else None),
        'mutated_residue_unscored_fraction': (
            float(np.mean([not s['mutated_residue_scored'] for s in flat])) if flat else None),
    }
    record = {'schema': SCHEMA, 'arm': args.arm, 'cohort': args.cohort,
              'source': str(args.source), 'source_sha256': hashlib.sha256(
                  args.source.read_bytes()).hexdigest(),
              'draw_seed': DRAW_SEED, 'skip_offsets': list(args.skip_offsets),
              'units_per_draw': args.units, 'states_per_unit': args.states_per_unit,
              'summary': summary, 'draws': draws, 'passed': True,
              'created_utc': datetime.now(timezone.utc).isoformat()}
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / f'coverage_{args.cohort}_{args.arm}.json', record)
    print(json.dumps({'arm': args.arm, 'cohort': args.cohort, **summary}), flush=True)


if __name__ == '__main__':
    main()
