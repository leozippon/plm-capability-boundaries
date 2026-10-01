#!/usr/bin/env python3
"""Measurement-only qualification of the remote-homology gate's stability endpoint.

Builds e = deltaG(variant) - deltaG(background) separately from the trypsin and
the chymotrypsin channel of the pinned Cho and Tsuboyama 2026 bytes, on one
identical admitted support, and reports between-channel agreement on two scales.

The two scales are the point of this step. On the **level** scale the channels
disagree at a mean offset of about -0.48 and a root-mean-square of about 0.82
kcal/mol against a combined spread of about 1.25, which is a floor two thirds of
the endpoint's own variation and is systematic rather than symmetric. That offset
is common to a variant and to its own background, so on the **effect** scale this
endpoint takes it cancels, and the floor that bounds this gate is the effect-scale
per-channel discordance measured here. Both are reported and every later
statement names which one it rests on.

Agreement is also reported inside each identity stratum, because a remote null is
only interpretable against the floor on the stratum it was measured on.

No model score, likelihood or representation enters any quantity here, so the
admission decision cannot be tuned to a model outcome. The two proteases are two
channels of one assay over one library and one stability inference, not
independent ground truth, and no oracle ceiling is constructed from them.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.mutation.external_confirmation import require_blas_threads
from src.capability.core.io import sha256_file, write_json
from src.capability.context.remote_homology import (
    BACKGROUND_DRAW, BACKGROUND_DRAW_SEED, BANDS, CHANNELS, ENDPOINT, MEASURED_QUANTITY,
    MIN_VARIANTS, QC_SOURCE, QC_WIDTH_KCAL_MOL, SOURCE_COLUMNS, SOURCE_DOI, SOURCE_FILE,
    SOURCE_SHA256, SUBSTITUTION_TOKEN, admitted, background_map, best_identity,
    channel_decomposition, channel_interval, channel_moments, declared_draw,
    endpoint_digest, family_groups, group_strata, kish_units, query_bands, read_source,
    substitution_at)

SCHEMA = 'remote_homology_endpoint_qualification_v1'

#: The census count this parse must reproduce: MGnify-flagged backgrounds that
#: carry a variant series and a wild-type row, as the frozen endpoint
#: qualification's own support census measured it. The background rule is
#: declared in one place and this is the check that the two readings agree, not a
#: second declaration of the rule.
CENSUS_MGNIFY_BACKGROUNDS_WITH_A_SERIES_AND_A_WILD_TYPE_ROW = 6051


def spread(values: np.ndarray) -> dict:
    """Quantiles, centre and spread of a sample, in the endpoint's own unit."""

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {'n': 0}
    return {'n': int(values.size), 'min': float(values.min()),
            'q1': float(np.percentile(values, 25)), 'median': float(np.median(values)),
            'q3': float(np.percentile(values, 75)), 'p90': float(np.percentile(values, 90)),
            'p95': float(np.percentile(values, 95)), 'max': float(values.max()),
            'mean': float(values.mean()), 'sd': float(values.std(ddof=1)),
            'rms': float(np.sqrt((values ** 2).mean()))}


def nested_unit_vectors(records: list[dict], key: str) -> list[np.ndarray]:
    """One moment vector per resampling unit, nested variants < sites < backgrounds.

    Every variant of a background is differenced against that background's single
    wild-type measurement, so resampling whole backgrounds or whole family groups
    keeps that shared measurement's signed contribution to all of its variants
    intact while variant-level resampling would break exactly that dependence.
    """

    by_unit: dict[str, dict[str, dict[int, list[dict]]]] = {}
    for row in records:
        by_unit.setdefault(row[key], {}).setdefault(
            row['background'], {}).setdefault(row['position'], []).append(row)
    vectors = []
    for unit in sorted(by_unit):
        per_background = []
        for name in sorted(by_unit[unit]):
            per_site = [channel_moments(
                np.asarray([r['trypsin'] for r in rows]),
                np.asarray([r['chymotrypsin'] for r in rows]))
                for _, rows in sorted(by_unit[unit][name].items())]
            per_background.append(np.mean(per_site, axis=0))
        vectors.append(np.mean(per_background, axis=0))
    return vectors


def agreement(records: list[dict], key: str, *, draws: int, seed: int) -> dict:
    """Effect-scale between-channel agreement over one set of admitted variants."""

    vectors = nested_unit_vectors(records, key)
    report = {'unit': key, 'units': len(vectors),
              'variants': len(records),
              'backgrounds': len({row['background'] for row in records}),
              'sites': len({(row['background'], row['position']) for row in records}),
              'sd_combined_effect_kcal_mol': float(
                  np.std([row['target'] for row in records], ddof=1)),
              'mean_combined_effect_kcal_mol': float(
                  np.mean([row['target'] for row in records]))}
    # The shared unit floor decides whether this stratum carries an interval;
    # a thin stratum keeps its point estimates and reports the refusal.
    report['decomposition'] = channel_interval(vectors, draws=draws, seed=seed)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', type=Path,
                        default=ROOT / 'data/mgnify_stability_cho2026' / SOURCE_FILE)
    parser.add_argument('--registry', type=Path, default=ROOT / 'data/dataset_registry.json')
    parser.add_argument('--qualification', type=Path,
                        default=ROOT / 'data/endpoint_qualification_20260924.json')
    parser.add_argument('--hits', type=Path,
                        default=ROOT / 'results/R5/remoteness_20260924/remoteness_hits.tsv',
                        help='the frozen identity search this gate reads its bands from')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260923)
    args = parser.parse_args()

    numeric = require_blas_threads()
    table_sha = sha256_file(args.table)
    if table_sha != SOURCE_SHA256:
        raise SystemExit(f'{args.table}: digest {table_sha} is not the registered '
                         f'{SOURCE_SHA256}')
    qualification = json.loads(args.qualification.read_bytes())
    scope = qualification['endpoints']['mgnify_stability_cho2026']['scope']

    rows = read_source(args.table)
    names = {row[0] for row in rows}
    roots = background_map(names)
    print(f'{len(rows)} rows, {len(names)} distinct names', flush=True)

    # Reproduce the census's own background accounting before anything is drawn
    # from it, so a divergence in the background rule fails here rather than
    # silently changing which sequences this gate scores.
    wildtype_sequence: dict[str, str] = {}
    wildtype_row: dict[str, tuple] = {}
    variant_names: dict[str, set[str]] = {}
    attributes: dict[str, tuple[str, str, str]] = {}
    for record in rows:
        name, library, sequence, mgnify, design = record[0], record[1], record[2], record[3], record[4]
        root = roots[name]
        attributes.setdefault(root, (mgnify, design, library))
        if name == root:
            wildtype_sequence.setdefault(root, sequence)
            wildtype_row.setdefault(root, record)
        else:
            variant_names.setdefault(root, set()).add(name)
    series = sorted(root for root in variant_names
                    if attributes[root][0] == 'True' and attributes[root][1] == 'False')
    with_wildtype = sorted(root for root in series if root in wildtype_sequence)
    if len(with_wildtype) != CENSUS_MGNIFY_BACKGROUNDS_WITH_A_SERIES_AND_A_WILD_TYPE_ROW:
        raise SystemExit(
            f'background rule reproduces {len(with_wildtype)} MGnify backgrounds with a '
            f'series and a wild-type row, not the census\'s '
            f'{CENSUS_MGNIFY_BACKGROUNDS_WITH_A_SERIES_AND_A_WILD_TYPE_ROW}')

    draw = declared_draw(with_wildtype, size=BACKGROUND_DRAW, seed=BACKGROUND_DRAW_SEED)
    print(f'{len(with_wildtype)} MGnify backgrounds with a series and a wild-type row; '
          f'declared draw {len(draw)}', flush=True)

    # Admission, and the accounting of everything the rule removes.
    counts = Counter()
    background_values: dict[str, tuple[float, float, float]] = {}
    for root in draw:
        record = wildtype_row[root]
        counts['drawn_backgrounds'] += 1
        if not admitted((record[6], record[8], record[10])):
            counts['backgrounds_without_an_admitted_wild_type_row'] += 1
            continue
        background_values[root] = (float(record[5]), float(record[7]), float(record[9]))

    records: list[dict] = []
    for record in rows:
        name = record[0]
        root = roots[name]
        if root not in background_values or name == root:
            continue
        token = name[len(root) + 1:]
        if SUBSTITUTION_TOKEN.match(token) is None:
            counts['non_substitution_variant_rows'] += 1
            continue
        counts['substitution_variant_rows'] += 1
        if not admitted((record[6], record[8], record[10])):
            counts['substitution_rows_outside_the_admission_rule'] += 1
            continue
        try:
            index = substitution_at(wildtype_sequence[root], record[2], token)
        except ValueError:
            counts['substitution_rows_disagreeing_with_their_own_sequence'] += 1
            continue
        wild_combined, wild_trypsin, wild_chymotrypsin = background_values[root]
        records.append({
            'background': root, 'position': index + 1, 'wild': token[0], 'mutant': token[-1],
            'sequence': record[2], 'library': record[1],
            'target': float(record[5]) - wild_combined,
            'trypsin': float(record[7]) - wild_trypsin,
            'chymotrypsin': float(record[9]) - wild_chymotrypsin,
            'width': float(record[6]),
            'background_combined_kcal_mol': wild_combined,
        })
    counts['admitted_variant_rows'] = len(records)

    # One admitted effect per (background, mutant sequence): the table repeats
    # names, so a repeated row is a repeated construct and is aggregated by its
    # median rather than counted twice.
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in records:
        grouped.setdefault((row['background'], row['sequence']), []).append(row)
    aggregated = []
    for (root, sequence), rows_here in sorted(grouped.items()):
        first = dict(rows_here[0])
        for field in ('target', 'trypsin', 'chymotrypsin', 'width'):
            first[field] = float(np.median([r[field] for r in rows_here]))
        first['rows'] = len(rows_here)
        aggregated.append(first)
    counts['admitted_states_after_median_aggregation'] = len(aggregated)
    counts['states_carrying_more_than_one_row'] = sum(1 for r in aggregated if r['rows'] > 1)

    per_background = Counter(row['background'] for row in aggregated)
    retained_names = sorted(name for name, count in per_background.items()
                            if count >= MIN_VARIANTS)
    below = sorted(name for name, count in per_background.items() if count < MIN_VARIANTS)
    retained = [row for row in aggregated if row['background'] in retained_names]
    print(f'{len(retained_names)} backgrounds at or above {MIN_VARIANTS} admitted '
          f'substitutions, {len(retained)} variants', flush=True)

    wildtypes = {name: wildtype_sequence[name] for name in retained_names}
    labels, grouping = family_groups(wildtypes)
    bands = query_bands(best_identity(args.hits, retained_names))
    assignment, strata = group_strata(labels, bands)
    for row in retained:
        row['group'] = labels[row['background']]
        row['band'] = bands[row['background']]
        row['stratum'] = assignment[labels[row['background']]]

    # The level-scale disagreement, over every admitted MGnify row of the draw,
    # reported so the effect-scale floor below is read against it.
    level = channel_decomposition(channel_moments(
        np.asarray([row['trypsin'] + background_values[row['background']][1]
                    for row in retained]),
        np.asarray([row['chymotrypsin'] + background_values[row['background']][2]
                    for row in retained])))

    report = {
        'schema': SCHEMA,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'endpoint': ENDPOINT,
        'measured_quantity': MEASURED_QUANTITY,
        'does_not_license': scope['does_not_license'],
        'licenses': scope['licenses'],
        'numeric_environment': numeric,
        'inputs': {
            'source_doi': SOURCE_DOI, 'source_file': SOURCE_FILE,
            'source_sha256': table_sha,
            'dataset_registry_sha256': sha256_file(args.registry),
            'endpoint_qualification_sha256': sha256_file(args.qualification),
            'identity_search_hits_sha256': sha256_file(args.hits),
            'columns_read': list(SOURCE_COLUMNS),
        },
        'admission_rule': (f'every one of the three reported 95% interval widths in '
                           f'[0, {QC_WIDTH_KCAL_MOL}] kcal/mol on the variant row and on its '
                           f'background\'s wild-type row; single amino-acid substitution '
                           f'constructs only, verified against both sequences'),
        'qc_source': QC_SOURCE,
        'aggregation': 'median over admitted rows per (background, mutant sequence)',
        'background_rule': ('a name\'s background is its longest proper underscore-prefix '
                            'that is itself a name in the table'),
        'background_rule_check': {
            'census_backgrounds_with_a_series_and_a_wild_type_row':
                CENSUS_MGNIFY_BACKGROUNDS_WITH_A_SERIES_AND_A_WILD_TYPE_ROW,
            'reproduced': len(with_wildtype),
        },
        'draw': {'size': BACKGROUND_DRAW, 'seed': BACKGROUND_DRAW_SEED,
                 'universe': len(with_wildtype),
                 'rule': ('a seeded sample of the sorted name universe; it reads names only '
                          'and reproduces the draw the frozen endpoint qualification banded')},
        'row_accounting': dict(sorted(counts.items())),
        'backgrounds_below_the_minimum': below,
        'minimum_variants': MIN_VARIANTS,
        'variant_cap': None,
        'variant_cap_note': ('no cap is declared: the largest admitted count on this support '
                             f'is {max(per_background.values())}, so a cap would not bind'),
        'support': {
            'backgrounds': len(retained_names),
            'family_groups': grouping['groups'],
            'variants': len(retained),
            'sites': len({(row['background'], row['position']) for row in retained}),
            'length_range': [min(map(len, wildtypes.values())), max(map(len, wildtypes.values()))],
            'libraries': dict(Counter(row['library'] for row in retained)),
            **kish_units(np.asarray([row['group'] for row in retained]),
                         np.asarray([row['background'] for row in retained]),
                         np.asarray([f"{row['background']}:{row['position']}"
                                     for row in retained])),
        },
        'grouping': grouping,
        'identity_bands': {'definition': '100 * nident / qlen over the best hit',
                           'bands': list(BANDS),
                           'backgrounds_per_band': dict(Counter(bands[n] for n in retained_names)),
                           'backgrounds_with_no_reported_hit': sum(
                               1 for value in best_identity(args.hits, retained_names).values()
                               if value is None)},
        'strata': strata,
        'level_scale_channel_disagreement': level,
        'level_scale_reading': (
            'the level-scale offset is systematic and is what makes these two channels '
            'non-interchangeable rather than merely noisy; it is common to a variant and '
            'to its own background, so it cancels in the effect this endpoint measures'),
        'effect_scale_agreement': {
            'all': agreement(retained, 'group', draws=args.bootstrap, seed=args.seed),
            'by_background': agreement(retained, 'background', draws=args.bootstrap,
                                       seed=args.seed),
            **{f'stratum_{name}': agreement(
                [row for row in retained if row['stratum'] == name], 'group',
                draws=args.bootstrap, seed=args.seed)
               for name in ('close', 'remote', 'mixed')
               if any(row['stratum'] == name for row in retained)},
        },
        'effect_spread_kcal_mol': spread(np.asarray([row['target'] for row in retained])),
        'interval_width_kcal_mol': spread(np.asarray([row['width'] for row in retained])),
        'channels': list(CHANNELS),
        'bootstrap': {'draws': args.bootstrap, 'seed': args.seed,
                      'unit': 'family group, with the background as a secondary unit'},
        'endpoint_digest': endpoint_digest(retained),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'endpoint_qualification.json', report)
    write_json(args.out / 'endpoint_records.json', {
        'schema': 'remote_homology_endpoint_records_v1',
        'endpoint': ENDPOINT,
        'source_sha256': table_sha,
        'endpoint_digest': report['endpoint_digest'],
        'minimum_variants': MIN_VARIANTS,
        'wildtypes': wildtypes,
        'background_channel_values': {name: list(background_values[name])
                                      for name in retained_names},
        'bands': {name: bands[name] for name in retained_names},
        'records': retained,
    })
    print(json.dumps({k: report[k] for k in
                      ('row_accounting', 'support', 'grouping', 'strata',
                       'level_scale_channel_disagreement', 'endpoint_digest')},
                     indent=1)[:6000])
    print(json.dumps(report['effect_scale_agreement']['all'], indent=1)[:3000])


if __name__ == '__main__':
    main()
