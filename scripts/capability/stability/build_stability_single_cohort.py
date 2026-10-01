#!/usr/bin/env python3
"""Declare the bounded, label-blind support of the single-mutant stability gate.

The support is fixed before any predictor is fitted and before any model is
scored. Three rules do the selecting and none of them reads a measurement:

* one natural-labelled background per final family group, chosen by alphabetical
  ``WT_name`` among the backgrounds of that group that carry an accepted wild-type
  row and at least the declared cap of eligible single mutants;
* at most :data:`~src.capability.stability.stability_gate.VARIANT_CAP` single mutants per
  background, drawn by a stable hash of the seed, the background name and the
  mutant sequence;
* substitution constructs only, with insertion and deletion rows excluded because
  their truncated ``aa_seq`` would otherwise enter a substitution support.

It writes three artefacts: the cohort with its measurements, a label-free
extraction plan carrying sequences and state indices only, and the remote
stratification's purge map, which lists for each background the other cohort
backgrounds whose local alignment reaches the declared sub-contract similarity
thresholds. The purge map is a function of sequence alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.family_grouping import batch_align, encode
from src.capability.core.io import sha256_file, write_json
from src.capability.stability.stability_gate import (
    DRAW_SEED, ENDPOINT, PROJECTION_DIM, PROJECTION_SEED, REMOTE_THRESHOLDS,
    VARIANT_CAP, accept_rows, aggregate_states, build_endpoint, draw_order,
    endpoint_digest, load_source_frame)
from src.capability.interactions.pairwise_epistasis import FEATURE_BLOCKS

PLAN_SCHEMA = 'stability_singles_extraction_v1'
COHORT_SCHEMA = 'stability_singles_cohort_v1'
ALIGNMENT_GAP_OPEN = 11.0
ALIGNMENT_GAP_EXTEND = 1.0


def plan_digest(plan: dict) -> str:
    return hashlib.sha256(json.dumps(
        {k: plan[k] for k in ('schema', 'cohort_sha256', 'projection', 'backgrounds')},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def remote_purge_map(backgrounds: list[dict], *, null_draws: int = 5,
                     null_seed: int = 20260925) -> tuple[dict, dict]:
    """Nearest-detected-neighbour purge maps, with a composition-preserving null.

    The frozen grouping contract already merged every pair at 30% identity and 80%
    mutual coverage, so an edge here sits strictly below the contract's own rule.
    For each declared threshold this returns, per held-out group, the other cohort
    groups purged from that fold's training set. The held-out rows are untouched,
    so a purged fit is paired with an unpurged one on exactly the same evaluation
    rows.

    The composition-preserving within-sequence shuffle null is computed at every
    threshold from the same wild types, the same pair set and the same rule. Where
    the observed count sits inside the null's range there is no alignment-detectable
    residual family structure to stratify on, and the purge bounds dependence on
    nearest-detected training neighbours rather than certifying remote-family
    disjointness. Homology below these thresholds is undetected either way.
    """

    sequences = [row['wildtype'] for row in backgrounds]
    codes, lengths = encode(sequences)
    pairs = np.column_stack(np.triu_indices(len(sequences), k=1))
    statistics = batch_align(codes, lengths, pairs, gap_open=ALIGNMENT_GAP_OPEN,
                             gap_extend=ALIGNMENT_GAP_EXTEND)
    rng = np.random.default_rng(null_seed)
    null_statistics = []
    for _ in range(null_draws):
        shuffled = [''.join(rng.permutation(list(s))) for s in sequences]
        shuffled_codes, shuffled_lengths = encode(shuffled)
        null_statistics.append(batch_align(shuffled_codes, shuffled_lengths, pairs,
                                           gap_open=ALIGNMENT_GAP_OPEN,
                                           gap_extend=ALIGNMENT_GAP_EXTEND))

    purges, report = {}, {
        'gap_open': ALIGNMENT_GAP_OPEN, 'gap_extend': ALIGNMENT_GAP_EXTEND,
        'pairs_scored': int(len(pairs)), 'null_draws': null_draws, 'null_seed': null_seed,
        'thresholds': {},
        'reading': ('the frozen contract merged every pair at 30% identity and 80% mutual '
                    'coverage, so these edges sit strictly below its own rule; where the '
                    'observed count lies inside the composition-preserving shuffle null the '
                    'purge removes training groups whose similarity is at chance level, so it '
                    'bounds dependence on nearest-detected neighbours and certifies no '
                    'remote-family disjointness'),
    }
    for label, identity_floor, coverage_floor in REMOTE_THRESHOLDS:
        accepted = statistics.edges(identity_floor=identity_floor,
                                    coverage_floor=coverage_floor)
        purge: dict[str, set[str]] = {row['group']: set() for row in backgrounds}
        edges = []
        for row in np.flatnonzero(accepted):
            left, right = backgrounds[pairs[row, 0]], backgrounds[pairs[row, 1]]
            purge[left['group']].add(right['group'])
            purge[right['group']].add(left['group'])
            edges.append({'left': left['name'], 'right': right['name'],
                          'percent_identity': round(float(statistics.percent_identity[row]), 3),
                          'coverage_left': round(float(statistics.coverage_a[row]), 3),
                          'coverage_right': round(float(statistics.coverage_b[row]), 3)})
        degrees = np.array([len(v) for v in purge.values()], float)
        report['thresholds'][label] = {
            'identity_floor_percent': identity_floor,
            'coverage_floor_percent': coverage_floor,
            'edges_observed': int(accepted.sum()),
            'edges_shuffle_null': [int(s.edges(identity_floor=identity_floor,
                                               coverage_floor=coverage_floor).sum())
                                   for s in null_statistics],
            'groups_with_at_least_one_edge': int((degrees > 0).sum()),
            'max_purged_groups_for_one_group': int(degrees.max()),
            'mean_purged_groups_over_touched': float(
                degrees[degrees > 0].mean()) if (degrees > 0).any() else 0.0,
            'edges': edges,
        }
        purges[label] = {k: sorted(v) for k, v in purge.items()}
    return purges, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=ROOT / 'data/megascale_tsuboyama2023/dataset2/data')
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'data/megascale_tsuboyama2023/manifest.json')
    parser.add_argument('--catalogue', type=Path,
                        default=ROOT / 'results/shared/megascale_disjointness_20260813/query_index.json')
    parser.add_argument('--groups', type=Path,
                        default=ROOT / 'data/pairwise_assets/megascale_family_groups_20260924.json')
    parser.add_argument('--qualification', type=Path, required=True,
                        help='the endpoint qualification this support is declared against')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cap', type=int, default=VARIANT_CAP)
    parser.add_argument('--seed', type=int, default=DRAW_SEED)
    args = parser.parse_args()

    qualification = json.loads(args.qualification.read_bytes())
    if qualification.get('schema') != 'stability_single_endpoint_qualification_v1':
        raise SystemExit('the endpoint qualification artefact is not the expected schema')
    if args.cap < 2:
        raise SystemExit('the cap must admit at least two variants per background')

    manifest = json.loads(args.manifest.read_bytes())
    expected = {Path(f['path']).name: f['sha256'] for f in manifest['verification']['files']}
    measured = {p.name: sha256_file(p) for p in sorted(args.data_dir.glob('*.parquet'))}
    if measured != expected:
        raise SystemExit('staged parquet bytes differ from the pinned manifest digests')

    grouping = json.loads(args.groups.read_bytes())
    if grouping.get('status') != 'admitted' or not grouping.get('provenance'):
        raise SystemExit('the family grouping map is not admitted')
    assignments = grouping['assignments']

    frame = load_source_frame(args.data_dir)
    wt = frame[frame.mut_type == 'wt'].groupby('WT_name').aa_seq.agg(['nunique', 'first'])
    clusters = frame.groupby('WT_name').WT_cluster.agg(['nunique', 'first'])
    catalogue = {}
    for row in json.loads(args.catalogue.read_bytes()):
        name = row['WT_name']
        if name not in wt.index or name not in assignments:
            continue
        if row['sequence'] != wt.loc[name, 'first']:
            raise SystemExit(f'catalogue sequence for {name} differs from its wild-type row')
        catalogue[name] = {'kind': row['kind'], 'sequence': row['sequence'],
                           'cluster': str(clusters.loc[name, 'first']),
                           'group': assignments[name]}

    mask, accounting = accept_rows(frame)
    values = aggregate_states(frame, mask, statistic='median')
    endpoint, support = build_endpoint(values, catalogue)
    natural = endpoint[endpoint.kind == 'natural']

    natural_names = {n for n, row in catalogue.items() if row['kind'] == 'natural'}
    eligible = {name: len(part) for name, part in natural.groupby('WT_name')}
    candidates, excluded = [], []
    for name in sorted(natural_names):
        count = eligible.get(name, 0)
        if count < args.cap:
            excluded.append({'name': name, 'group': assignments[name],
                             'reason': 'no_accepted_wildtype_row' if count == 0 else 'below_cap',
                             'eligible_variants': int(count)})
            continue
        candidates.append({'name': name, 'group': assignments[name], 'eligible': int(count)})

    chosen: dict[str, dict] = {}
    for row in candidates:
        if row['group'] in chosen:
            excluded.append({'name': row['name'], 'group': row['group'],
                             'reason': 'group_already_represented',
                             'eligible_variants': row['eligible']})
        else:
            chosen[row['group']] = row

    backgrounds = []
    for group in sorted(chosen):
        name = chosen[group]['name']
        part = natural[natural.WT_name == name]
        wildtype = catalogue[name]['sequence']
        keep = set(draw_order(name, part.sequence.tolist(), seed=args.seed)[:args.cap])
        part = part[part.sequence.isin(keep)].sort_values('sequence')
        order = [wildtype] + sorted(keep)
        index = {sequence: position for position, sequence in enumerate(order)}
        variants = [{'sequence': r.sequence, 'position': int(r.position), 'mutant': r.mutant,
                     'state': index[r.sequence], 'ddg': float(r.combined),
                     'ddg_trypsin': float(r.trypsin), 'ddg_chymotrypsin': float(r.chymotrypsin),
                     'rows': int(r.n_rows), 'distinct_dna': int(r.n_dna)}
                    for r in part.itertuples()]
        backgrounds.append({
            'name': name, 'group': group, 'cluster': catalogue[name]['cluster'],
            'length': len(wildtype), 'wildtype': wildtype,
            'wildtype_combined_kcal_mol': float(part.wildtype_combined.iloc[0]),
            'eligible_variants': chosen[group]['eligible'], 'sequences': order,
            'sites': sorted({v['position'] for v in variants}), 'variants': variants})

    purge, alignment = remote_purge_map(backgrounds)
    sequences = {s for row in backgrounds for s in row['sequences']}
    sites = [(row['name'], v['position']) for row in backgrounds for v in row['variants']]
    per_group_counts = np.array([len(row['variants']) for row in backgrounds], float)
    site_sizes = np.array([len(row['sites']) for row in backgrounds], float)
    # Analysis weighting: groups equal, sites equal inside a group, variants equal
    # inside a site. The effective site count is the Kish count of that weight
    # vector at the site level, which is what bounds the power of this support.
    site_weights = np.concatenate([
        np.full(int(count), 1.0 / (len(backgrounds) * count)) for count in site_sizes])
    effective_sites = float(site_weights.sum() ** 2 / (site_weights ** 2).sum())
    cohort = {
        'schema': COHORT_SCHEMA,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'endpoint': ENDPOINT,
        'endpoint_qualification_sha256': sha256_file(args.qualification),
        'grouping_contract_sha256': grouping['provenance']['contract']['sha256'],
        'cap': args.cap, 'draw_seed': args.seed,
        'draw_rule': 'stable sha256 of seed, background name and mutant sequence; no measurement read',
        'background_rule': 'one natural-labelled background per final group, alphabetical WT_name',
        'row_accounting': accounting,
        'endpoint_support_accounting': support,
        'available_support': {
            'natural_backgrounds_in_catalogue': len(natural_names),
            'natural_backgrounds_with_variants': int(natural.WT_name.nunique()),
            'natural_groups_in_map': len({assignments[n] for n in natural_names}),
            'natural_variants_available': int(len(natural)),
            'design_backgrounds_not_eligible': sum(
                1 for row in catalogue.values() if row['kind'] != 'natural'),
        },
        'summary': {
            'backgrounds': len(backgrounds),
            'groups_covered': len({row['group'] for row in backgrounds}),
            'variants': sum(len(row['variants']) for row in backgrounds),
            'distinct_sequences': len(sequences),
            'residues': sum(len(s) for s in sequences),
            'length_range': [min(map(len, sequences)), max(map(len, sequences))],
            'sites': len(set(sites)),
            'sites_per_background': [int(site_sizes.min()), int(np.median(site_sizes)),
                                     int(site_sizes.max())],
            'variants_per_site': [int(np.min(np.bincount(np.unique(
                [f'{n}:{p}' for n, p in sites], return_inverse=True)[1]))),
                int(np.median(np.bincount(np.unique(
                    [f'{n}:{p}' for n, p in sites], return_inverse=True)[1]))),
                int(np.max(np.bincount(np.unique(
                    [f'{n}:{p}' for n, p in sites], return_inverse=True)[1])))],
            'effective_groups_kish': float(
                per_group_counts.sum() ** 2 / (per_group_counts ** 2).sum()),
            'effective_sites_kish': effective_sites,
            'weighting': 'groups equal, sites equal inside a group, variants equal inside a site',
        },
        'remote_stratification': alignment,
        'remote_purge': purge,
        'excluded': excluded,
        'backgrounds': backgrounds,
    }
    cohort['endpoint_sha256'] = endpoint_digest(
        [{'background': row['name'], 'sequence': v['sequence'], 'position': v['position'],
          'ddg': v['ddg']} for row in backgrounds for v in row['variants']])

    args.out.mkdir(parents=True, exist_ok=True)
    cohort_path = args.out / 'cohort.json'
    write_json(cohort_path, cohort)
    cohort_sha = sha256_file(cohort_path)

    plan = {
        'schema': PLAN_SCHEMA, 'cohort_sha256': cohort_sha,
        'projection': {'seed': PROJECTION_SEED, 'dim': PROJECTION_DIM,
                       'blocks': list(FEATURE_BLOCKS)},
        'backgrounds': [{'name': row['name'], 'group': row['group'], 'length': row['length'],
                         'wildtype': row['wildtype'], 'sequences': row['sequences'],
                         'variants': [{'position': v['position'], 'mutant': v['mutant'],
                                       'state': v['state']} for v in row['variants']]}
                        for row in backgrounds],
        'summary': cohort['summary'],
    }
    if any(field in json.dumps(plan) for field in ('ddg', 'kcal', 'trypsin')):
        raise SystemExit('the extraction plan carries a measurement field; refusing to write it')
    write_json(args.out / 'extraction_plan.json', plan)
    print(json.dumps({'cohort_sha256': cohort_sha, 'plan_sha256': plan_digest(plan),
                      'endpoint_sha256': cohort['endpoint_sha256'],
                      'summary': cohort['summary'],
                      'excluded': {r: sum(1 for e in excluded if e['reason'] == r)
                                   for r in sorted({e['reason'] for e in excluded})},
                      'remote_stratification': {
                          k: {f: v[f] for f in ('edges_observed', 'edges_shuffle_null',
                                                'groups_with_at_least_one_edge',
                                                'mean_purged_groups_over_touched')}
                          for k, v in alignment['thresholds'].items()}},
                     indent=1))


if __name__ == '__main__':
    main()
