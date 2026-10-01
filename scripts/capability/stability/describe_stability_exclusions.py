#!/usr/bin/env python3
"""Profile what the single-mutant stability cohort's exclusions removed.

Dropping a natural background that carries no accepted wild-type row conditions
the cohort on wild-type resolvability under the confidence filter, and a domain
whose wild-type free energy could not be resolved to within 0.5 kcal/mol is
plausibly not an arbitrary one: a marginally stable or poorly behaved domain is
exactly the case whose wild-type estimate fails. This stage measures the
difference rather than asserting it is innocent.

Two families of quantity are reported and kept apart. Label-independent
descriptors — length, composition, family-group size and retrieved homolog depth
— could in principle have entered a predictor and did not. Channel descriptors —
the share of a background's own rows that are censored or that exceed the
confidence width — are derived from the measurement and are reported here only as
a property of the support; they are never features, never enter a fit and never
reach the endpoint.

It reads artefacts and pinned bytes that already exist, fits nothing and draws
nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.io import sha256_file, write_json
from src.capability.context.profiles import KYTE_DOOLITTLE
from src.capability.stability.stability_gate import (
    CENSORED_DG_ML, ENDPOINT, QC_WIDTH_KCAL_MOL, load_source_frame, substitution_rows)

CHARGED = set('DEKR')


def descriptors(sequence: str) -> dict:
    return {
        'length': len(sequence),
        'hydropathy_mean': float(np.mean([KYTE_DOOLITTLE[a] for a in sequence])),
        'charged_fraction': sum(1 for a in sequence if a in CHARGED) / len(sequence),
    }


def quantiles(values: list[float]) -> dict | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {'n': int(array.size), 'min': float(array.min()),
            'q1': float(np.percentile(array, 25)), 'median': float(np.median(array)),
            'q3': float(np.percentile(array, 75)), 'max': float(array.max()),
            'mean': float(array.mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path,
                        default=ROOT / 'data/megascale_tsuboyama2023/dataset2/data')
    parser.add_argument('--catalogue', type=Path,
                        default=ROOT / 'results/shared/megascale_disjointness_20260813/query_index.json')
    parser.add_argument('--groups', type=Path,
                        default=ROOT / 'data/pairwise_assets/megascale_family_groups_20260924.json')
    parser.add_argument('--hits', type=Path, required=True,
                        help='the retained DIAMOND hit table, read for retrieval depth only')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    cohort = json.loads(args.cohort.read_bytes())
    if cohort.get('schema') != 'stability_singles_cohort_v1':
        raise SystemExit('unexpected cohort schema')
    retained = {row['name'] for row in cohort['backgrounds']}
    reasons = {row['name']: row['reason'] for row in cohort['excluded']}

    assignments = json.loads(args.groups.read_bytes())['assignments']
    group_size = Counter(assignments.values())
    catalogue = {row['WT_name']: row for row in json.loads(args.catalogue.read_bytes())}

    hits_per_query: Counter = Counter()
    with args.hits.open() as handle:
        for line in handle:
            query, _, _ = line.partition('\t')
            hits_per_query[query] += 1

    frame = load_source_frame(args.data_dir)
    numeric = pd.to_numeric(frame.dG_ML, errors='coerce')
    substitution = substitution_rows(frame.mut_type.to_numpy())
    censored = frame.dG_ML.isin(CENSORED_DG_ML).to_numpy()
    wide = ((~frame.deltaG_95CI.between(0, QC_WIDTH_KCAL_MOL))
            | (~frame.deltaG_t_95CI.between(0, QC_WIDTH_KCAL_MOL))
            | (~frame.deltaG_c_95CI.between(0, QC_WIDTH_KCAL_MOL))).to_numpy()
    finite = np.isfinite(numeric).to_numpy()

    per_background = {}
    for name, index in frame.groupby('WT_name', sort=True).indices.items():
        rows = np.asarray(index)
        substitution_rows_here = rows[substitution[rows]]
        total = len(substitution_rows_here)
        if not total:
            continue
        per_background[name] = {
            'substitution_rows': total,
            'censored_row_share': float(censored[substitution_rows_here].mean()),
            'wide_interval_row_share': float(
                (finite[substitution_rows_here] & wide[substitution_rows_here]).mean()),
        }

    strata: dict[str, list[str]] = {'retained': sorted(retained)}
    for name, reason in reasons.items():
        strata.setdefault(reason, []).append(name)
    report_strata = {}
    for label, names in strata.items():
        present = [n for n in names if n in catalogue]
        report_strata[label] = {
            'backgrounds': len(names),
            'label_independent': {
                key: quantiles([descriptors(catalogue[n]['sequence'])[key] for n in present])
                for key in ('length', 'hydropathy_mean', 'charged_fraction')},
            'family_group_size': quantiles([float(group_size[assignments[n]])
                                            for n in present if n in assignments]),
            'retrieved_homolog_hits': quantiles([float(hits_per_query.get(n, 0)) for n in present]),
            'backgrounds_with_no_retrieved_homolog': sum(
                1 for n in present if hits_per_query.get(n, 0) == 0),
            'channel_descriptors': {
                key: quantiles([per_background[n][key] for n in present if n in per_background])
                for key in ('substitution_rows', 'censored_row_share', 'wide_interval_row_share')},
        }

    report = {
        'schema': 'stability_exclusion_profile_v1',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'endpoint': ENDPOINT,
        'cohort_sha256': sha256_file(args.cohort),
        'strata': report_strata,
        'reading': (
            'the cohort is conditioned on wild-type resolvability under the confidence '
            'filter and on resolvable mutant states, so destabilised material is '
            'systematically absent; the channel descriptors say how far the excluded '
            'strata differ on that axis and are never used as features'),
        'label_independence': (
            'length, composition, family-group size and retrieved homolog depth are '
            'label-independent and could have entered a predictor; the censored and '
            'wide-interval shares are measurement-derived and enter nothing'),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'exclusion_profile.json', report)
    print(json.dumps({label: {
        'backgrounds': value['backgrounds'],
        'length_median': (value['label_independent']['length'] or {}).get('median'),
        'hydropathy_median': (value['label_independent']['hydropathy_mean'] or {}).get('median'),
        'charged_median': (value['label_independent']['charged_fraction'] or {}).get('median'),
        'group_size_median': (value['family_group_size'] or {}).get('median'),
        'hits_median': (value['retrieved_homolog_hits'] or {}).get('median'),
        'no_hit_backgrounds': value['backgrounds_with_no_retrieved_homolog'],
        'substitution_rows_median': (value['channel_descriptors']['substitution_rows'] or {}).get('median'),
        'censored_share_median': (value['channel_descriptors']['censored_row_share'] or {}).get('median'),
        'wide_share_median': (value['channel_descriptors']['wide_interval_row_share'] or {}).get('median'),
    } for label, value in report_strata.items()}, indent=1))


if __name__ == '__main__':
    main()
