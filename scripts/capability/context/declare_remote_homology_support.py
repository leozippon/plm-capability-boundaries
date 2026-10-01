#!/usr/bin/env python3
"""Declare the label-blind support, groups and identity strata of the remote-homology gate.

Everything this entry point writes is fixed before any predictor is fitted and
before any model is scored, and nothing it decides reads a measured stability, an
interval width's value beyond the declared admission rule, or a model quantity.

Four rules do the selecting, and all four are declared in
``src.capability.context.remote_homology`` rather than here:

* **The background draw** is a seeded sample of the MGnify-flagged backgrounds
  that carry a variant series and a wild-type row. It reads names only, and it
  reproduces the draw whose wild types the frozen endpoint qualification already
  searched and banded, so this cohort's identity bands are the canonical ones.
* **The admission rule** is the tightened per-channel one, applied to the variant
  row and to its background's wild-type row because the endpoint is their
  difference; single amino-acid substitutions only, verified against both
  sequences.
* **A minimum of admitted substitutions per background**, so a retained
  background contributes a site-resolved panel. No variant cap is declared
  because the largest admitted count on this support does not reach one.
* **Family grouping** is the frozen contract's own alignment edge rule. These
  names carry no accession and no Pfam label, so alignment and exact sequence
  identity are the only union sources available.

It writes the cohort with its measurements, a label-free extraction plan carrying
sequences and state indices only, and the support declaration with every digest,
every exclusion count and the close, remote and mixed group strata the
self-calibrating readout is defined on.
"""
from __future__ import annotations

import argparse
import hashlib
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
    BACKGROUND_DRAW, BACKGROUND_DRAW_SEED, CLOSE_BANDS, ENDPOINT, FEATURE_BLOCKS,
    MEASURED_QUANTITY, MIN_VARIANTS, PROJECTION_DIM, PROJECTION_SEED, QC_SOURCE,
    QC_WIDTH_KCAL_MOL, REMOTE_BANDS, SOURCE_DOI, SOURCE_FILE, STRATUM_UNIT_FLOOR,
    declaration_digest, endpoint_digest, family_groups, group_strata, kish_units)

PLAN_SCHEMA = 'stability_singles_extraction_v1'
COHORT_SCHEMA = 'remote_homology_cohort_v1'
DECLARATION_SCHEMA = 'remote_homology_support_v1'

#: Strings the extraction plan must not contain. The plan is what reaches a GPU,
#: so it carries sequences and state indices only and an entry point that finds a
#: measurement field in it refuses to score.
MEASUREMENT_FIELDS = ('deltaG', 'kcal', 'trypsin', 'chymotrypsin', 'target', 'band')


def plan_digest(plan: dict) -> str:
    return hashlib.sha256(json.dumps(
        {k: plan[k] for k in ('schema', 'cohort_sha256', 'projection', 'backgrounds')},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', type=Path, required=True,
                        help='the endpoint qualification this support is declared against')
    parser.add_argument('--records', type=Path, required=True,
                        help='the admitted endpoint records that qualification wrote')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    numeric = require_blas_threads()
    qualification = json.loads(args.qualification.read_bytes())
    if qualification.get('schema') != 'remote_homology_endpoint_qualification_v1':
        raise SystemExit('the endpoint qualification artefact is not the expected schema')
    records = json.loads(args.records.read_bytes())
    if records.get('schema') != 'remote_homology_endpoint_records_v1':
        raise SystemExit('the endpoint records artefact is not the expected schema')
    if records['endpoint_digest'] != qualification['endpoint_digest']:
        raise SystemExit('the records and the qualification carry different endpoint digests')
    if records['minimum_variants'] != MIN_VARIANTS:
        raise SystemExit('the records were admitted under a different minimum-variant rule')

    wildtypes = dict(records['wildtypes'])
    bands = dict(records['bands'])
    labels, grouping = family_groups(wildtypes)
    assignment, strata = group_strata(labels, bands)

    per_background: dict[str, list[dict]] = {}
    for row in records['records']:
        per_background.setdefault(row['background'], []).append(row)
    if sorted(per_background) != sorted(wildtypes):
        raise SystemExit('the records cover a different background set from their wild types')

    backgrounds = []
    for name in sorted(per_background):
        wildtype = wildtypes[name]
        variants = sorted(per_background[name], key=lambda r: (r['position'], r['mutant']))
        sequences = [wildtype] + [row['sequence'] for row in variants]
        if len(set(sequences)) != len(sequences):
            raise SystemExit(f'{name}: a repeated state survived median aggregation')
        backgrounds.append({
            'name': name, 'group': labels[name], 'band': bands[name],
            'stratum': assignment[labels[name]],
            'wildtype': wildtype, 'length': len(wildtype),
            'library': variants[0]['library'],
            'background_combined_kcal_mol': variants[0]['background_combined_kcal_mol'],
            'admitted_variants': len(variants),
            'sites': sorted({row['position'] for row in variants}),
            'variants': [{'position': row['position'], 'mutant': row['mutant'],
                          'sequence': row['sequence'], 'state': index + 1,
                          'target': row['target'], 'uncertainty': row['width'],
                          'target_trypsin': row['trypsin'],
                          'target_chymotrypsin': row['chymotrypsin'],
                          'rows': row['rows']}
                         for index, row in enumerate(variants)],
        })

    flat = [{'background': row['name'], 'group': row['group'], **variant}
            for row in backgrounds for variant in row['variants']]
    group_labels = np.asarray([row['group'] for row in flat])
    background_labels = np.asarray([row['background'] for row in flat])
    site_labels = np.asarray([f"{row['background']}:{row['position']}" for row in flat])
    units = kish_units(group_labels, background_labels, site_labels)
    sites_per_background = np.asarray([len(row['sites']) for row in backgrounds], dtype=float)
    variants_per_site = np.asarray(list(Counter(site_labels.tolist()).values()), dtype=float)

    stratum_units = {}
    for name in ('close', 'remote', 'mixed'):
        rows = [row for row in flat if assignment[row['group']] == name]
        if not rows:
            stratum_units[name] = {'groups': 0, 'backgrounds': 0, 'variants': 0,
                                   'clears_the_unit_floor': False}
            continue
        keep = np.asarray([assignment[label] == name for label in group_labels])
        stratum_units[name] = {
            **kish_units(group_labels[keep], background_labels[keep], site_labels[keep]),
            'clears_the_unit_floor': bool(
                len({row['group'] for row in rows}) >= STRATUM_UNIT_FLOOR),
            'bands': dict(Counter(bands[row['background']] for row in rows)),
        }

    cohort = {
        'schema': COHORT_SCHEMA,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'endpoint': ENDPOINT,
        'measured_quantity': MEASURED_QUANTITY,
        'does_not_license': qualification['does_not_license'],
        'numeric_environment': numeric,
        'source': {'doi': SOURCE_DOI, 'file': SOURCE_FILE,
                   'sha256': records['source_sha256']},
        'endpoint_qualification_sha256': sha256_file(args.qualification),
        'admission_rule': qualification['admission_rule'],
        'qc_source': QC_SOURCE,
        'qc_width_kcal_mol': QC_WIDTH_KCAL_MOL,
        'uncertainty_source': ("the source's own combined `deltaG_95CI`, the width of the "
                               'reported 95% interval in kcal/mol; it is carried as reported '
                               'and is never weighted, shrunk or thresholded on beyond the '
                               'declared admission rule'),
        'draw': {'size': BACKGROUND_DRAW, 'seed': BACKGROUND_DRAW_SEED,
                 'minimum_variants': MIN_VARIANTS, 'variant_cap': None},
        'weighting': units['weighting'],
        'summary': {
            'backgrounds': len(backgrounds),
            'family_groups': grouping['groups'],
            'variants': len(flat),
            'sites': int(len(set(site_labels.tolist()))),
            'distinct_sequences': len({s for row in backgrounds
                                       for s in [row['wildtype']]
                                       + [v['sequence'] for v in row['variants']]}),
            'sequences_per_arm': sum(1 + len(row['variants']) for row in backgrounds),
            'residues_per_arm': sum((1 + len(row['variants'])) * row['length']
                                    for row in backgrounds),
            'length_range': [min(row['length'] for row in backgrounds),
                             max(row['length'] for row in backgrounds)],
            'sites_per_background': [int(sites_per_background.min()),
                                     int(np.median(sites_per_background)),
                                     int(sites_per_background.max())],
            'variants_per_site': [int(variants_per_site.min()),
                                  int(np.median(variants_per_site)),
                                  int(variants_per_site.max())],
            'libraries': dict(Counter(row['library'] for row in backgrounds)),
            **{k: v for k, v in units.items() if k != 'weighting'},
        },
        'grouping': grouping,
        'strata': strata,
        'stratum_support': stratum_units,
        'close_bands': list(CLOSE_BANDS),
        'remote_bands': list(REMOTE_BANDS),
        'backgrounds': backgrounds,
    }
    cohort['endpoint_sha256'] = endpoint_digest(flat)
    if cohort['endpoint_sha256'] != records['endpoint_digest']:
        raise SystemExit('the cohort endpoint digest differs from the qualified records')

    args.out.mkdir(parents=True, exist_ok=True)
    cohort_path = args.out / 'cohort.json'
    write_json(cohort_path, cohort)
    cohort_sha = sha256_file(cohort_path)

    plan = {
        'schema': PLAN_SCHEMA, 'cohort_sha256': cohort_sha,
        'projection': {'seed': PROJECTION_SEED, 'dim': PROJECTION_DIM,
                       'blocks': list(FEATURE_BLOCKS)},
        'backgrounds': [{
            'name': row['name'], 'group': row['group'], 'length': row['length'],
            'wildtype': row['wildtype'],
            'sequences': [row['wildtype']] + [v['sequence'] for v in row['variants']],
            'variants': [{'position': v['position'], 'mutant': v['mutant'],
                          'state': v['state']} for v in row['variants']],
        } for row in backgrounds],
        'summary': cohort['summary'],
    }
    serialised = json.dumps(plan['backgrounds'])
    present = [field for field in MEASUREMENT_FIELDS if field in serialised]
    if present:
        raise SystemExit(f'the extraction plan carries measurement fields {present}; '
                         'refusing to write it')
    plan_path = args.out / 'extraction_plan.json'
    write_json(plan_path, plan)

    declaration = {
        'schema': DECLARATION_SCHEMA,
        'declared_utc': datetime.now(timezone.utc).isoformat(),
        'endpoint': ENDPOINT,
        'measured_quantity': MEASURED_QUANTITY,
        'does_not_license': qualification['does_not_license'],
        'licenses': qualification['licenses'],
        'numeric_environment': numeric,
        'inputs': qualification['inputs'],
        'row_accounting': qualification['row_accounting'],
        'backgrounds_below_the_minimum': qualification['backgrounds_below_the_minimum'],
        'support': cohort['summary'],
        'grouping': grouping,
        'identity_bands': qualification['identity_bands'],
        'strata': strata,
        'stratum_support': stratum_units,
        'self_calibration': (
            'the same fit, the same folds and the same held-out predictions are read out '
            'separately on the close and on the remote groups; the close stratum is the '
            'positive control and a remote readout is never reported without naming which '
            'of the three declared outcomes the pair selects'),
        'floor': {
            'level_scale': qualification['level_scale_channel_disagreement'],
            'effect_scale': qualification['effect_scale_agreement']['all']['decomposition'],
            'reading': qualification['level_scale_reading'],
        },
        'endpoint_digest': cohort['endpoint_sha256'],
        'cohort_sha256': cohort_sha,
        'plan_content_sha256': plan_digest(plan),
        'plan_sha256': sha256_file(plan_path),
    }
    declaration['declaration_digest'] = declaration_digest(
        {k: declaration[k] for k in ('schema', 'endpoint', 'cohort_sha256',
                                     'plan_content_sha256', 'endpoint_digest')})
    write_json(args.out / 'support_declaration.json', declaration)
    print(json.dumps({'support': declaration['support'], 'strata': strata,
                      'stratum_support': {k: {kk: vv for kk, vv in v.items()
                                              if kk != 'weighting'}
                                          for k, v in stratum_units.items()},
                      'cohort_sha256': cohort_sha,
                      'plan_content_sha256': declaration['plan_content_sha256'],
                      'endpoint_digest': declaration['endpoint_digest']},
                     indent=1))


if __name__ == '__main__':
    main()
