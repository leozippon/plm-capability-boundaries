#!/usr/bin/env python3
"""Does a point substitution move an arm's token grid only where the residue changed?

A per-position reading of an autoregressive mutation score needs the mutant's
token positions to correspond to the wild type's. A byte-pair tokeniser need not
give that: one substituted residue can move a merge boundary, change the packed
token count, and alter tokens far from the substitution. Where that happens the
site-against-downstream split has no unique referent, so the alignment rate is a
precondition of the decomposition and a property of the interface in its own
right.

The measurement needs no weights and no likelihood: it is a comparison of two
token sequences. Every variant of every declared state is classified, not a
sample, so the reported rate is the cohort's rate rather than a draw from it.
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
from src.capability.interactions.pairwise_epistasis import ARM_DTYPE, ROSTER, TOKENISATION_STRATUM
from src.capability.position.position_terms import ALIGNMENT, ResidueCoverage, alignment
from src.capability.readouts.readout_extraction import load_readout_arm, pack_sequence

SCHEMA = 'position_alignment_v1'


def substitution_sites(wildtype: str, mutant: str) -> list[int]:
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


def variants(kind: str, payload: dict):
    """Yield ``(unit, wild-type sequence, mutant sequence, sites)`` for every state."""

    if kind in {'stability', 'pairwise'}:
        for row in payload['backgrounds']:
            sequences = row['sequences']
            wildtype = sequences[0]
            if kind == 'stability':
                for variant in row['variants']:
                    yield (row['name'], wildtype, sequences[int(variant['state'])],
                           [int(variant['position']) - 1])
            else:
                seen = set()
                for cycle in row['cycles']:
                    wild, low, high, double = (int(v) for v in cycle['states'])
                    sites = sorted(int(v) - 1 for v in cycle['positions'])
                    for state, site in ((low, sites[0]), (high, sites[1])):
                        if state in seen:
                            continue
                        seen.add(state)
                        yield (row['name'], sequences[wild], sequences[state], [site])
    elif kind == 'anchor':
        for row in payload['assays']:
            wildtype = row['wildtype']
            for mutant, sequence in zip(row['mutants'], row['sequences']):
                yield (row['assay'], wildtype, sequence,
                       substitution_sites(wildtype, mutant))
    else:
        raise ValueError(f'unknown cohort kind {kind!r}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--cohort', required=True, choices=('stability', 'pairwise', 'anchor'))
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--support', type=Path,
                        help='anchor only: restrict to the frozen panel assay identifiers')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', default='cpu', help='accepted and ignored: no weights load')
    args = parser.parse_args()

    if args.arm not in ROSTER:
        raise SystemExit(f'{args.arm} is not on the frozen roster')
    spec = importlib.util.spec_from_file_location(
        'stage46', ROOT / 'scripts/capability/stages/context_homologue.py')
    stage46 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage46)
    arm = load_readout_arm(args.arm, stage46, dtype=ARM_DTYPE.get(args.arm, 'float32'))
    coverage = ResidueCoverage(arm)
    payload = json.loads(args.source.read_bytes())
    keep = None
    if args.support is not None:
        keep = set(json.loads(args.support.read_bytes())['assays'])
        payload = {'assays': [row for row in payload['assays'] if row['assay'] in keep]}

    def state(sequence: str) -> dict:
        ids, span, _ = pack_sequence(arm, sequence)
        counts, offset = coverage.counts(ids, span, sequence)
        return {'ids': [int(v) for v in ids], 'span': (int(span[0]), int(span[1])),
                'counts': counts, 'offset': int(offset)}

    # Only the wild types repeat -- each mutant state is visited once -- so a cache
    # over wild types alone is the whole saving, and caching mutants would hold a
    # cohort's packings for no reuse.
    wild_cache: dict[str, dict] = {}

    aligned, extents, lengths, reasons, singles, multi = [], [], [], {}, [], []
    outside_span = 0
    per_unit: dict[str, list[bool]] = {}
    for unit, wildtype, sequence, sites in variants(args.cohort, payload):
        if wildtype not in wild_cache:
            wild_cache[wildtype] = state(wildtype)
        fit = alignment(wild_cache[wildtype], state(sequence), sites)
        flag = bool(fit['aligned'])
        aligned.append(flag)
        per_unit.setdefault(unit, []).append(flag)
        lengths.append(int(fit['packed_length_delta']))
        if not flag:
            reasons[str(fit['reason'])] = reasons.get(str(fit['reason']), 0) + 1
            if fit['mismatch_extent_residues'] is not None:
                extents.append(int(fit['mismatch_extent_residues']))
            if fit.get('differing_outside_scored_span'):
                outside_span += 1
        (singles if len(sites) == 1 else multi).append(flag)

    percentiles = (50, 90, 99, 100)
    record = {
        'schema': SCHEMA, 'arm': args.arm, 'cohort': args.cohort,
        'interface': TOKENISATION_STRATUM[args.arm], 'rule': ALIGNMENT,
        'source': str(args.source),
        'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'support_restricted': keep is not None,
        'variants': len(aligned),
        'aligned': int(sum(aligned)),
        'unaligned': int(len(aligned) - sum(aligned)),
        'aligned_fraction': float(np.mean(aligned)) if aligned else None,
        'aligned_fraction_single_substitution': float(np.mean(singles)) if singles else None,
        'aligned_fraction_multi_substitution': float(np.mean(multi)) if multi else None,
        'single_substitution_variants': len(singles),
        'multi_substitution_variants': len(multi),
        'unaligned_reasons': reasons,
        'unaligned_with_unscored_substitution': int(outside_span),
        'packed_length_delta_nonzero': int(sum(bool(v) for v in lengths)),
        'packed_length_delta_range': [int(min(lengths)), int(max(lengths))] if lengths else None,
        'mismatch_extent_residues': (
            {str(q): float(np.percentile(extents, q)) for q in percentiles} if extents else None),
        'mismatch_extent_residues_max': int(max(extents)) if extents else 0,
        'units': len(per_unit),
        'units_fully_aligned': int(sum(all(v) for v in per_unit.values())),
        'units_with_no_aligned_variant': int(sum(not any(v) for v in per_unit.values())),
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / f'alignment_{args.cohort}_{args.arm}.json', record)
    print(json.dumps({k: record[k] for k in (
        'arm', 'cohort', 'interface', 'variants', 'aligned_fraction',
        'aligned_fraction_single_substitution', 'packed_length_delta_nonzero',
        'unaligned_with_unscored_substitution', 'mismatch_extent_residues_max',
        'units_with_no_aligned_variant')}), flush=True)


if __name__ == '__main__':
    main()
