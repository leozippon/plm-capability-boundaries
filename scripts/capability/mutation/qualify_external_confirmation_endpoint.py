#!/usr/bin/env python3
"""Measurement-only qualification of the Domainome external-confirmation endpoint.

The endpoint is qualified on **its own support**, which is the 494 domains that
remain once the 28 matching a development target by sequence are removed, and not
on all 522 the source file carries. Those 28 are named by the frozen endpoint
qualification and are re-derived here from it rather than retyped; no identifier
in either source names any of them, so the exclusion rests on sequence matching
and an identifier-only screen would have left every one of them in.

Two floors are reported and they are on two different scales.

* The **replicate floor** is the source's own three biological replicates. Each
  variant's log2 output-over-input ratio is read per replicate, and replicate 1
  against replicate 2 is decomposed the same way the folding-and-stability gate
  decomposes its two protease channels: a shared-component standard deviation
  that is an **upper** bound on reproducible signal, and a per-replicate
  discordance standard deviation that is a **lower** bound on per-replicate
  measurement noise. It is in log2-enrichment units, which is the scale the
  replicates are measured on and not the scale of the endpoint.
* The **endpoint-scale floor** is the source's own reported
  ``normalized_fitness_sigma``, summarised as a distribution rather than a single
  number because its tail is what bounds a small effect.

Relating the two is the source's own normalisation and is not attempted here: no
oracle ceiling is constructed, and no replicate variance is added to any other.

No model score, likelihood or representation enters any quantity here.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.mutation.external_confirmation import (
    ENDPOINT, MEASURED_QUANTITY, REPLICATES, SOURCE_ARCHIVE, SOURCE_ARCHIVE_SHA256,
    SOURCE_DOI, kish_units, load_cohort, require_blas_threads)
from src.capability.core.io import sha256_file, write_json
from src.capability.context.remote_homology import channel_interval, channel_moments

SCHEMA = 'external_confirmation_endpoint_qualification_v1'

#: The two channels the replicate floor is decomposed over, and their unit. They
#: are the source's first two biological replicates and they are in log2
#: enrichment, which is not the endpoint's scale, so the decomposition's keys
#: carry that unit and not the stability gate's kcal/mol.
REPLICATE_CHANNELS = ('replicate_1', 'replicate_2')
REPLICATE_UNIT = 'log2_enrichment'

#: The counts the clean support must reproduce. They are the frozen support
#: census's own numbers minus the 28 overlapping domains, asserted here so that a
#: change in the exclusion set or in the parse fails at this step rather than
#: moving a number in a document.
CLEAN_SUPPORT = {'domains': 494, 'uniprot_accessions': 413, 'pfam_accessions': 120,
                 'substitution_rows': 541202}


def spread(values) -> dict:
    values = np.asarray(sorted(values), dtype=float)
    if values.size == 0:
        return {'n': 0}
    return {'n': int(values.size), 'min': float(values[0]),
            'q1': float(np.percentile(values, 25)), 'median': float(np.median(values)),
            'q3': float(np.percentile(values, 75)), 'p90': float(np.percentile(values, 90)),
            'p95': float(np.percentile(values, 95)), 'max': float(values[-1]),
            'mean': float(values.mean()), 'sd': float(values.std(ddof=1)),
            'rms': float(np.sqrt((values ** 2).mean()))}


def overlap_domains(qualification: dict) -> tuple[set[str], dict]:
    """The development-overlapping domains, re-derived from the frozen record."""

    support = qualification['endpoints']['domainome_beltran2025']['measured_support']
    per_source, union = {}, set()
    for key in ('overlap_with_proteingym', 'overlap_with_tsuboyama2023'):
        block = support[key]
        names = set()
        for field in ('exact_sequence_matches', 'containment_matches'):
            names.update(str(pair[0]) for pair in block[field])
        per_source[key] = sorted(names)
        union |= names
    both = sorted(set(per_source['overlap_with_proteingym'])
                  & set(per_source['overlap_with_tsuboyama2023']))
    return union, {
        'matching_a_proteingym_target': len(per_source['overlap_with_proteingym']),
        'matching_a_tsuboyama_background': len(per_source['overlap_with_tsuboyama2023']),
        'matching_both': len(both),
        'excluded': sorted(union),
        'excluded_count': len(union),
        'named_by_an_identifier_in_either_source': 0,
        'reading': ('no ProteinGym entry and no Tsuboyama background names Beltran or the '
                    'domainome, so the exclusion rests on sequence matching; the union is '
                    'smaller than the two counts added because some domains match both'),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/domainome_beltran2025')
    parser.add_argument('--qualification', type=Path,
                        default=ROOT / 'data/endpoint_qualification_20260924.json')
    parser.add_argument('--registry', type=Path, default=ROOT / 'data/dataset_registry.json')
    parser.add_argument('--cohort', type=Path,
                        help='the declared cohort, for the floor on the drawn variants')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260923)
    args = parser.parse_args()

    numeric = require_blas_threads()
    archive = args.data_dir / SOURCE_ARCHIVE
    archive_sha = sha256_file(archive)
    if archive_sha != SOURCE_ARCHIVE_SHA256:
        raise SystemExit(f'{archive}: digest {archive_sha} is not the registered '
                         f'{SOURCE_ARCHIVE_SHA256}')
    qualification = json.loads(args.qualification.read_bytes())
    scope = qualification['endpoints']['domainome_beltran2025']['scope']
    excluded, overlap = overlap_domains(qualification)

    cohort_units: dict[str, str] | None = None
    cohort_record = None
    if args.cohort is not None:
        loaded = load_cohort(args.cohort)
        cohort_units = {row['name']: row['group'] for row in loaded['units']}
        cohort_record = {'cohort_sha256': loaded['sha256'],
                         'endpoint_sha256': loaded['endpoint_sha256'],
                         'units': len(cohort_units),
                         'family_groups': len(set(cohort_units.values()))}

    archive_handle = zipfile.ZipFile(archive)
    member = next(name for name in archive_handle.namelist()
                  if name.endswith('.txt') and not name.startswith('__MACOSX'))
    accessions: set[str] = set()
    pfam: set[str] = set()
    rows = substitutions = stops = 0
    sigma: list[float] = []
    replicate_sd: list[float] = []
    replicate_pairs: list[tuple[float, float, str, str, int]] = []
    domains: set[str] = set()
    cohort_sigma: list[float] = []
    cohort_pairs: list[tuple[float, float, str, str, int]] = []
    with archive_handle.open(member) as stream:
        reader = csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8'), delimiter='\t')
        for row in reader:
            rows += 1
            domain = row['domain_ID']
            if domain in excluded:
                continue
            domains.add(domain)
            accessions.add(row['uniprot_ID'])
            fields = [field for field in domain.split('_') if field.startswith('PF')]
            if len(fields) == 1:
                pfam.add(fields[0])
            if row['STOP'].strip().upper() == 'TRUE':
                stops += 1
                continue
            substitutions += 1
            reported = row['normalized_fitness_sigma']
            if reported not in ('', 'NA'):
                sigma.append(float(reported))
            try:
                inputs = [float(row[f'input_count_rep{k}']) for k in REPLICATES]
                outputs = [float(row[f'output_count_rep{k}']) for k in REPLICATES]
            except (KeyError, ValueError):
                continue
            if min(inputs) <= 0 or min(outputs) <= 0:
                continue
            ratios = [math.log2(out / inp) for inp, out in zip(inputs, outputs)]
            centre = sum(ratios) / len(ratios)
            replicate_sd.append(
                math.sqrt(sum((value - centre) ** 2 for value in ratios) / (len(ratios) - 1)))
            position = row['position']
            site = int(float(position)) if position not in ('', 'NA') else 0
            replicate_pairs.append((ratios[0], ratios[1], domain, domain, site))
            if cohort_units is not None and domain in cohort_units:
                cohort_pairs.append((ratios[0], ratios[1], cohort_units[domain], domain, site))
                if reported not in ('', 'NA'):
                    cohort_sigma.append(float(reported))

    measured = {'domains': len(domains), 'uniprot_accessions': len(accessions),
                'pfam_accessions': len(pfam), 'substitution_rows': substitutions}
    wrong = {key: (measured[key], value) for key, value in CLEAN_SUPPORT.items()
             if measured[key] != value}
    if wrong:
        raise SystemExit(f'the clean support does not reproduce its declared counts: {wrong}')

    def decompose(pairs, label: str) -> dict:
        """Replicate-channel decomposition, nested variants < sites < domains < units."""

        if not pairs:
            return {'unit': label, 'units': 0}
        nested: dict[str, dict[str, dict[int, list[tuple[float, float]]]]] = {}
        for first, second, unit, domain, site in pairs:
            nested.setdefault(unit, {}).setdefault(domain, {}).setdefault(
                site, []).append((first, second))
        vectors = []
        for unit in sorted(nested):
            per_domain = []
            for domain in sorted(nested[unit]):
                per_site = [channel_moments(np.asarray([a for a, _ in cell]),
                                            np.asarray([b for _, b in cell]))
                            for _, cell in sorted(nested[unit][domain].items())]
                per_domain.append(np.mean(per_site, axis=0))
            vectors.append(np.mean(per_domain, axis=0))
        record = channel_interval(vectors, channels=REPLICATE_CHANNELS,
                                  unit=REPLICATE_UNIT, draws=args.bootstrap, seed=args.seed)
        record.update(unit=label, variants=len(pairs),
                      domains=len({row[3] for row in pairs}))
        return record

    report = {
        'schema': SCHEMA,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'endpoint': ENDPOINT,
        'measured_quantity': MEASURED_QUANTITY,
        'licenses': scope['licenses'],
        'does_not_license': scope['does_not_license'],
        'numeric_environment': numeric,
        'inputs': {'source_doi': SOURCE_DOI, 'source_archive': SOURCE_ARCHIVE,
                   'source_archive_sha256': archive_sha, 'member': member,
                   'dataset_registry_sha256': sha256_file(args.registry),
                   'endpoint_qualification_sha256': sha256_file(args.qualification)},
        'development_overlap': overlap,
        'clean_support': {
            'rule': ('the source file minus the domains matching a development target by '
                     'sequence, which is the support this endpoint is qualified on'),
            'rows_in_file': rows,
            'declared': CLEAN_SUPPORT,
            'measured': measured,
            'stop_rows': stops,
        },
        'endpoint_scale_floor': {
            'quantity': "the source's own reported normalized_fitness_sigma",
            'unit': 'dimensionless normalised fitness',
            'clean_support': spread(sigma),
            'declared_cohort': spread(cohort_sigma) if cohort_sigma else None,
        },
        'replicate_floor': {
            'quantity': ('log2 output-over-input of replicate 1 against replicate 2, over '
                         'the variants whose three input and three output counts are all '
                         'positive'),
            'unit': 'log2 enrichment, which is not the endpoint scale',
            'between_replicate_sd_over_three_replicates': spread(replicate_sd),
            'clean_support': decompose(replicate_pairs, 'domain'),
            'declared_cohort': decompose(cohort_pairs, 'family group'),
            'reading': ('the shared-component standard deviation is an upper bound on '
                        'reproducible signal because error common to both replicates '
                        'contributes to it, and the per-replicate discordance standard '
                        'deviation is a lower bound on per-replicate noise because that '
                        'common error cancels; neither is combined with the endpoint-scale '
                        'floor and no oracle ceiling is constructed'),
        },
        'declared_cohort': cohort_record,
        'bootstrap': {'draws': args.bootstrap, 'seed': args.seed},
    }
    if cohort_units is not None:
        units = np.asarray([row[2] for row in cohort_pairs])
        report['declared_cohort']['effective_units_over_replicate_rows'] = kish_units(
            units, np.asarray([row[3] for row in cohort_pairs]),
            np.asarray([f'{row[3]}:{row[4]}' for row in cohort_pairs]))
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'endpoint_qualification.json', report)
    print(json.dumps({k: report[k] for k in
                      ('development_overlap', 'clean_support', 'endpoint_scale_floor',
                       'declared_cohort')}, indent=1)[:5000])
    print(json.dumps(report['replicate_floor']['declared_cohort'], indent=1)[:3000])


if __name__ == '__main__':
    main()
