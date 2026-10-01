#!/usr/bin/env python3
"""Precompute the independent-site profile of every cohort background.

The sequence/profile control C carries mutation-local profile summaries. Those
summaries come from the same frozen homolog rows, the same coverage floor and the
same reweighting threshold the admitted pairwise baseline used, so the profile
declaration cannot drift between the two. This step reads the retained DIAMOND
hit table once on CPU and writes a compact array file, which is what the fit
consumes; no phenotype, stability value or model output is read here.

A background that retrieves no qualifying homolog row gets no profile. It is
recorded as absent and the fit carries a zero availability indicator for it,
rather than borrowing a pooled or nearest-background profile.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.capability.context.homology import ALIGNMENT_FIELDS, parse_hits
from src.capability.context.profiles import (
    NEFF_IDENTITY_FLOOR, PROFILE_COVERAGE_FLOOR, REWEIGHT_IDENTITY_FLOOR, build_profile)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b''):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--hits', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-sequences', type=int, default=20000)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    wanted = {row['name']: row['wildtype'] for row in plan['backgrounds']}
    grouped: dict[str, list] = {name: [] for name in wanted}
    for hit in parse_hits(args.hits, fields=ALIGNMENT_FIELDS):
        if hit.query in grouped:
            grouped[hit.query].append(hit)

    arrays, records = {}, []
    for name, wildtype in sorted(wanted.items()):
        hits = grouped[name]
        record = {'name': name, 'length': len(wildtype), 'hits_retrieved': len(hits)}
        profile = build_profile(wildtype, name, hits, max_sequences=args.max_sequences,
                               coverage_floor=PROFILE_COVERAGE_FLOOR,
                               reweight_identity=REWEIGHT_IDENTITY_FLOOR,
                               neff_identity_floor=NEFF_IDENTITY_FLOOR) if hits else None
        if profile is None or profile.n_sequences == 0:
            record['status'] = 'absent'
            records.append(record)
            continue
        arrays[f'{name}|frequencies'] = profile.frequencies.astype(np.float32)
        arrays[f'{name}|column_weight'] = profile.column_weight.astype(np.float32)
        arrays[f'{name}|scalars'] = np.array(
            [profile.neff, profile.log10_neff, profile.max_identity_over_query,
             profile.n_sequences, profile.n_hits], dtype=np.float64)
        record.update(status='present', n_profile_sequences=int(profile.n_sequences),
                      neff=round(float(profile.neff), 4),
                      supported_columns=int((profile.column_weight > 0).sum()))
        records.append(record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {'schema': 'pairwise_profile_features_v1',
            'plan_sha256': digest(args.plan), 'hits_sha256': digest(args.hits),
            'max_sequences': args.max_sequences,
            'coverage_floor': PROFILE_COVERAGE_FLOOR,
            'reweight_identity': REWEIGHT_IDENTITY_FLOOR,
            'neff_identity_floor': NEFF_IDENTITY_FLOOR,
            'backgrounds': records}
    with args.out.open('wb') as stream:
        np.savez_compressed(stream, metadata=json.dumps(meta), **arrays)
    present = sum(1 for r in records if r['status'] == 'present')
    print(json.dumps({'out': str(args.out), 'sha256': digest(args.out),
                      'backgrounds': len(records), 'present': present,
                      'absent': [r['name'] for r in records if r['status'] != 'present']}, indent=1))


if __name__ == '__main__':
    main()
