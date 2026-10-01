#!/usr/bin/env python3
"""Declare and freeze the retrieval and memorization strata, before any stratified fit.

Two cohorts carry the gains this gate reads: the frozen 64-background measured
double-mutant stability cohort, whose resampling unit is the site pair inside an
equally weighted held group, and the frozen 201-assay Readout anchor, whose
resampling unit is the wild-type family at 50% identity. This entry point writes
one declaration covering both, with every external resource bound to a source
hash and every band edge taken from an already-frozen declaration.

Nothing measured is read. The stability unit table is built from the label-free
extraction plan rather than from the cohort file, and the Readout unit table
reads only the assay identifier, its cluster and its wild-type query id. A
measured stability, a DMS effect and a fitted prediction therefore cannot reach a
stratum boundary even in principle.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.context.homology import (
    ALIGNMENT_FIELDS, assign_stratum, parse_hits, potential_identity_over_query,
    truncated_alignment)
from src.capability.interactions.pairwise_epistasis import plan_digest
from src.capability.context.retrieval_strata import (
    SOURCE_EXCLUSION, SOURCE_TOKEN, STRATIFICATIONS, assign_unit, bands_of, kish_row_counts,
    kish_subunits, verify_partition)
from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS

SCHEMA = 'retrieval_memorization_strata_declaration_v1'

#: Written into the declared output directory, which is what the campaign runner
#: injects as ``--out``.
DECLARATION_BASENAME = 'strata_declaration.json'
COHORT_HITS_BASENAME = 'cohort_hits.tsv'

#: The profile settings both stores must share for one depth notion to span both
#: cohorts. A divergence is fatal rather than footnoted.
REQUIRED_PROFILE_SETTINGS = {'coverage_floor': 80.0, 'reweight_identity': 80.0,
                             'neff_identity_floor': 30.0}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b''):
            sha.update(chunk)
    return sha.hexdigest()


def content_digest(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def source(path: Path, **extra) -> dict:
    """Record one input by repository-relative path where it has one, and by digest."""

    resolved = Path(path)
    try:
        named = resolved.relative_to(ROOT)
    except ValueError:
        named = resolved
    return {'path': str(named), 'bytes': resolved.stat().st_size, 'sha256': digest(resolved),
            **extra}


def filter_hits(hits: Path, queries: set[str], destination: Path) -> tuple[Path, int, int]:
    """Keep the hit lines of the declared queries, so the frozen parser can read them.

    The retained table holds 618,403 lines over 478 queries; the parser builds one
    record per line with both aligned strings, so the cohort's own lines are split
    out first rather than materialising the whole table.
    """

    kept = total = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with hits.open() as handle, destination.open('w') as out:
        for line in handle:
            total += 1
            if line.split('\t', 1)[0] in queries:
                out.write(line)
                kept += 1
    return destination, kept, total


def nearest_homolog(hit_rows) -> dict[str, dict]:
    """The closest retrieved relative of every query, by identity over the query.

    ``identity_over_query`` is ``nident / qlen``, not the aligner's ``pident``:
    a corpus entry identical over a fragment of the query is not a stored copy of
    it, and the frozen homology module says so at its own definition.
    """

    best: dict[str, dict] = {}
    counts: dict[str, int] = defaultdict(int)
    truncated: dict[str, int] = defaultdict(int)
    potential: dict[str, float] = defaultdict(float)
    for hit in hit_rows:
        counts[hit.query] += 1
        observed = hit.identity_over_query
        if truncated_alignment(hit):
            truncated[hit.query] += 1
            potential[hit.query] = max(potential[hit.query],
                                       potential_identity_over_query(hit))
        current = best.get(hit.query)
        if current is None or (observed, hit.bitscore) > (current['max_identity_over_query'],
                                                          current['best_bitscore']):
            best[hit.query] = {'max_identity_over_query': float(observed),
                               'max_pident': float(hit.pident),
                               'best_subject': hit.subject,
                               'best_bitscore': float(hit.bitscore),
                               'query_length': int(hit.qlen)}
    for query, record in best.items():
        record['n_hits'] = int(counts[query])
        record['truncated_alignments'] = int(truncated[query])
        record['max_potential_identity_over_query'] = float(potential[query]) or None
    return best


def stability_units(args) -> dict:
    """Unit table of the stability cohort: one held group, one background."""

    plan = json.loads(args.pairwise_plan.read_text())
    if plan_digest(plan) != args.expect_pairwise_plan_sha256:
        raise SystemExit('the extraction plan content digest does not match the declared value')
    manifest = json.loads(args.pairwise_search_manifest.read_text())
    hits_sha = digest(args.pairwise_hits)
    if manifest['hits']['sha256'] != hits_sha:
        raise SystemExit('the hit table does not match its own search manifest digest')
    if tuple(manifest['fields']) != ALIGNMENT_FIELDS:
        raise SystemExit('the hit table does not carry the frozen alignment field list')
    with np.load(args.pairwise_profiles, allow_pickle=False) as data:
        profile_meta = json.loads(str(data['metadata']))
    if profile_meta['hits_sha256'] != hits_sha:
        raise SystemExit('the profile features were not built from this hit table')
    settings = {key: float(profile_meta[key]) for key in REQUIRED_PROFILE_SETTINGS}
    if settings != REQUIRED_PROFILE_SETTINGS:
        raise SystemExit(f'stability profile settings {settings} are not the declared ones')
    groups = json.loads(args.family_groups.read_text())
    if groups['provenance']['contract']['sha256'] != args.expect_grouping_contract_sha256:
        raise SystemExit('the family groups were not built from the declared grouping contract')
    parquet = groups['provenance']['parquet']
    if not all(SOURCE_TOKEN.replace('_', '').lower() in record['path'].lower()
               for record in parquet):
        raise SystemExit('the pinned measurement files do not carry the declared source')
    members: dict[str, int] = defaultdict(int)
    for group in groups['assignments'].values():
        members[group] += 1

    names = {row['name'] for row in plan['backgrounds']}
    filtered, kept, total = filter_hits(args.pairwise_hits, names,
                                        args.out / COHORT_HITS_BASENAME)
    nearest = nearest_homolog(parse_hits(filtered, fields=ALIGNMENT_FIELDS))
    depth = {record['name']: (float(record['neff']) if record['status'] == 'present' else None)
             for record in profile_meta['backgrounds']}
    retrieved = {record['name']: int(record['hits_retrieved'])
                 for record in profile_meta['backgrounds']}

    units, assignment, site_pairs, cycles_per_pair = {}, {}, {}, {}
    for row in plan['backgrounds']:
        name, group = row['name'], row['group']
        if group in units:
            raise SystemExit(f'{group} carries more than one cohort background')
        hit = nearest.get(name)
        identity = hit['max_identity_over_query'] if hit else 0.0
        if hit is not None and hit['n_hits'] != retrieved[name]:
            raise SystemExit(f'{name}: {hit["n_hits"]} retrieved hits against '
                             f'{retrieved[name]} in the profile record')
        pairs: dict[tuple, int] = {}
        for cycle in row['cycles']:
            key = tuple(cycle['positions'])
            pairs[key] = pairs.get(key, 0) + 1
        site_pairs[group] = len(pairs)
        cycles_per_pair[group] = sorted(pairs.values())
        units[group] = {
            'background': name, 'group': group, 'length': row['length'],
            'group_members_under_contract': int(members[group]),
            'max_identity_over_query': float(identity),
            'best_subject': hit['best_subject'] if hit else None,
            'retrieved_hits': int(retrieved[name]),
            'hit_list_saturated': bool(retrieved[name] >= int(manifest['command'][
                manifest['command'].index('--max-target-seqs') + 1])),
            'truncated_alignments': int(hit['truncated_alignments']) if hit else 0,
            'profile_neff': depth[name],
            'site_pairs': len(pairs),
            'cycles': len(row['cycles']),
        }
        assignment[group] = assign_unit(max_identity_over_query=identity, neff=depth[name],
                                        group_members=int(members[group]),
                                        assay_identifiers=[SOURCE_TOKEN])
    partition = verify_partition(assignment)
    return {
        'unit': 'held family group of the frozen grouping contract, one background each',
        'weighted_subunit': 'site pair inside an equally weighted held group',
        'units': units, 'assignment': assignment, 'partition': partition,
        'support': {'groups': len(units), 'site_pairs': sum(site_pairs.values()),
                    'cycles': sum(row['cycles'] for row in units.values()),
                    'kish_effective_site_pairs_estimator_weights': kish_subunits(site_pairs),
                    'kish_effective_site_pairs_cycle_counts': kish_row_counts(
                        [count for counts in cycles_per_pair.values() for count in counts])},
        'band_support': {
            name: {
                band: {
                    'groups': len(members_of_band),
                    'site_pairs': sum(site_pairs[group] for group in members_of_band),
                    'cycles': sum(units[group]['cycles'] for group in members_of_band),
                    'kish_effective_site_pairs_estimator_weights': kish_subunits(
                        {group: site_pairs[group] for group in members_of_band}),
                    'kish_effective_site_pairs_cycle_counts': kish_row_counts(
                        [count for group in members_of_band for count in cycles_per_pair[group]]),
                    'below_interval_unit_floor': len(members_of_band) < MINIMUM_BOOTSTRAP_UNITS,
                }
                for band, members_of_band in bands_of(assignment, name).items()
            } for name in STRATIFICATIONS
        },
        'source_provenance_evidence': {
            'token': SOURCE_TOKEN,
            'statement': ('every background of this cohort is drawn from the pinned MegaScale '
                          'release below, so the whole support sits in one provenance band and '
                          'the band carries no contrast here; it is declared to make the shared '
                          'source with the Readout anchor explicit rather than to stratify'),
            'pinned_measurement_files': parquet,
        },
        'sources': {
            'extraction_plan': source(args.pairwise_plan,
                                      content_sha256=args.expect_pairwise_plan_sha256),
            'hit_table': source(args.pairwise_hits, queries=int(manifest['query']['records']),
                                cohort_lines=kept, total_lines=total,
                                max_target_seqs=manifest['command'][
                                    manifest['command'].index('--max-target-seqs') + 1]),
            'search_manifest': source(args.pairwise_search_manifest),
            'diamond_binary_sha256': manifest['tool']['binary_sha256'],
            'corpus': {'path': manifest['database']['path'],
                       'sha256': manifest['database']['sha256'],
                       'bytes': manifest['database']['bytes'],
                       'indexed_sequences': manifest['database']['indexed_sequences'],
                       'indexed_letters': manifest['database']['indexed_letters']},
            'profile_features': source(args.pairwise_profiles, settings=settings),
            'family_groups': source(args.family_groups),
            'grouping_contract_sha256': args.expect_grouping_contract_sha256,
        },
    }


def readout_units(args) -> dict:
    """Unit table of the Readout anchor: one wild-type family at 50% identity."""

    cohort = json.loads(args.readout_cohort.read_text())
    cohort_sha = digest(args.readout_cohort)
    if cohort_sha != args.expect_readout_cohort_sha256:
        raise SystemExit('the Readout cohort digest does not match the declared value')
    wildtype = {row['assay']: row['wildtype_id'] for row in cohort['assays']}
    cohort_cluster = {row['assay']: str(row['cluster']) for row in cohort['assays']}
    universe_clusters = defaultdict(set)
    for row in cohort['assays']:
        universe_clusters[str(row['cluster'])].add(row['wildtype_id'])

    reports = sorted(args.crossed_reports.glob('*/crossed_controls_*_fold*.json'))
    if not reports:
        raise SystemExit(f'no crossed-control cell report under {args.crossed_reports}')
    anchor, clusters, variants, cells = None, None, None, []
    for path in reports:
        cell = json.loads(path.read_text())
        if cell['cohort_sha256'] != cohort_sha:
            raise SystemExit(f'{path.name} was fitted on different cohort bytes')
        assays = tuple(row['assay'] for row in cell['assays'])
        mapping = {row['assay']: str(row['cluster']) for row in cell['assays']}
        counts = {row['assay']: int(row['n_variants']) for row in cell['assays']}
        if anchor is None:
            anchor, clusters, variants = assays, mapping, counts
        elif assays != anchor or mapping != clusters or counts != variants:
            raise SystemExit(f'{path.name} does not share the anchor support and cluster map')
        cells.append(source(path, arm=cell['arm'], fold_seed=cell['fold_seed'],
                            n_assays=cell['n_assays'], n_families=cell['n_families']))
    for assay, cluster in clusters.items():
        if cohort_cluster.get(assay) != cluster:
            raise SystemExit(f'{assay}: the fitted cluster label is not the cohort label')

    search = json.loads(args.readout_search.read_text())
    profiles = json.loads(args.readout_profiles.read_text())
    settings = {key: float(profiles['settings'][key]) for key in REQUIRED_PROFILE_SETTINGS}
    if settings != REQUIRED_PROFILE_SETTINGS:
        raise SystemExit(f'Readout profile settings {settings} are not the declared ones')
    assignments = {record['query_id']: record for record in search['assignments']}
    store = profiles['profiles']

    per_cluster = defaultdict(list)
    for assay in anchor:
        per_cluster[clusters[assay]].append(assay)
    units, assignment, heterogeneous, mixed_source = {}, {}, {}, {}
    for cluster, members in sorted(per_cluster.items()):
        queries = sorted({wildtype[assay] for assay in members})
        sourced = [assay for assay in sorted(members) if SOURCE_TOKEN in assay]
        if sourced and len(sourced) != len(members):
            mixed_source[cluster] = sourced
        identity = max(float(assignments[query]['max_identity_over_query']) for query in queries)
        neff = max(float(store[query]['neff']) for query in queries)
        for query in queries:
            if assign_stratum(float(assignments[query]['max_identity_over_query'])) != \
                    assign_stratum(identity):
                heterogeneous.setdefault(cluster, []).append(query)
        members_under_contract = len(universe_clusters[cluster])
        units[cluster] = {
            'cluster': cluster, 'assays': sorted(members), 'n_assays': len(members),
            'wildtype_ids': queries,
            'group_members_under_contract': members_under_contract,
            'max_identity_over_query': identity,
            'profile_neff': neff,
            'retrieved_hits': max(int(assignments[query]['n_hits']) for query in queries),
            'hit_list_saturated': any(bool(assignments[query]['hit_list_saturated'])
                                      for query in queries),
            'truncated_best_hit': any(bool(assignments[query]['best_hit_looks_truncated'])
                                      for query in queries),
            'source_token_assays': sourced,
        }
        assignment[cluster] = assign_unit(max_identity_over_query=identity, neff=neff,
                                          group_members=members_under_contract,
                                          assay_identifiers=sorted(members))
    partition = verify_partition(assignment)
    assays_per_cluster = {cluster: record['n_assays'] for cluster, record in units.items()}
    return {
        'unit': 'wild-type family at 50% identity and 80% coverage of the shorter sequence',
        'weighted_subunit': 'assay inside an equally weighted wild-type family',
        'units': units, 'assignment': assignment, 'partition': partition,
        'cluster_level_rule': ('a cluster carries the maximum identity and the maximum depth over '
                              'its member wild types, which is the direction that cannot '
                              'understate retrieval support; the count of clusters whose members '
                              'fall in different identity bands under a per-wild-type rule is '
                              'reported as heterogeneous_clusters'),
        'heterogeneous_clusters': heterogeneous,
        'source_provenance_evidence': {
            'token': SOURCE_TOKEN,
            'statement': ('a cluster holding any assay with this token sits in the sourced band, '
                          'so excluding that band removes whole resampling units and leaves every '
                          'retained unit\'s internal weighting untouched; the cohort-wide assay '
                          'count with the token is reported beside the anchor count because the '
                          'anchor is a subset of the cohort'),
            'anchor_assays_with_token': sum(1 for assay in anchor if SOURCE_TOKEN in assay),
            'anchor_variants_with_token': sum(variants[assay] for assay in anchor
                                              if SOURCE_TOKEN in assay),
            'cohort_assays_with_token': sum(1 for row in cohort['assays']
                                            if SOURCE_TOKEN in row['assay']),
            'cohort_assays': len(cohort['assays']),
            'mixed_source_clusters': mixed_source,
            'exclusion_rule': SOURCE_EXCLUSION,
            'unshared_support_under_primary_rule': {
                'clusters': sum(1 for cluster in units
                                if assignment[cluster]['source_provenance'] == 'source_other'),
                'assays': sum(assays_per_cluster[cluster] for cluster in units
                              if assignment[cluster]['source_provenance'] == 'source_other'),
            },
            'unshared_support_under_sensitivity_rule': {
                'clusters': sum(1 for cluster in units
                                if assignment[cluster]['source_provenance'] == 'source_other'
                                or cluster in mixed_source),
                'assays': sum(
                    len([assay for assay in units[cluster]['assays']
                         if SOURCE_TOKEN not in assay])
                    for cluster in units
                    if assignment[cluster]['source_provenance'] == 'source_other'
                    or cluster in mixed_source),
            },
        },
        'support': {'clusters': len(units), 'assays': len(anchor),
                    'variants': sum(variants.values()),
                    'kish_effective_assays_estimator_weights': kish_subunits(assays_per_cluster),
                    'kish_effective_assays_variant_counts': kish_row_counts(
                        [variants[assay] for assay in anchor])},
        'band_support': {
            name: {
                band: {
                    'clusters': len(members_of_band),
                    'assays': sum(assays_per_cluster[cluster] for cluster in members_of_band),
                    'variants': sum(variants[assay] for cluster in members_of_band
                                    for assay in units[cluster]['assays']),
                    'kish_effective_assays_estimator_weights': kish_subunits(
                        {cluster: assays_per_cluster[cluster] for cluster in members_of_band}),
                    'kish_effective_assays_variant_counts': kish_row_counts(
                        [variants[assay] for cluster in members_of_band
                         for assay in units[cluster]['assays']]),
                    'below_interval_unit_floor': len(members_of_band) < MINIMUM_BOOTSTRAP_UNITS,
                }
                for band, members_of_band in bands_of(assignment, name).items()
            } for name in STRATIFICATIONS
        },
        'sources': {
            'cohort': source(args.readout_cohort),
            'search': source(args.readout_search,
                             max_target_seqs=search['max_target_seqs'],
                             hits_sha256=search['hits_sha256'],
                             diamond_binary_sha256=search['diamond']['binary_sha256'],
                             corpus={'path': search['database']['database_path'],
                                     'indexed_sequences': search['database']['indexed_sequences'],
                                     'indexed_letters': search['database']['indexed_letters']}),
            'profiles': source(args.readout_profiles, settings=settings),
            'crossed_control_cells': cells,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairwise-plan', type=Path, required=True)
    parser.add_argument('--expect-pairwise-plan-sha256', required=True)
    parser.add_argument('--pairwise-hits', type=Path, required=True)
    parser.add_argument('--pairwise-search-manifest', type=Path, required=True)
    parser.add_argument('--pairwise-profiles', type=Path, required=True)
    parser.add_argument('--family-groups', type=Path, required=True)
    parser.add_argument('--expect-grouping-contract-sha256', required=True)
    parser.add_argument('--readout-cohort', type=Path, required=True)
    parser.add_argument('--expect-readout-cohort-sha256', required=True)
    parser.add_argument('--readout-search', type=Path, required=True)
    parser.add_argument('--readout-profiles', type=Path, required=True)
    parser.add_argument('--crossed-reports', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True,
                        help='output directory; the declaration is written into it by name')
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    declaration = {
        'schema': SCHEMA,
        'stratifications': {name: {key: (list(value) if isinstance(value, tuple) else value)
                                   for key, value in spec.items()}
                            for name, spec in STRATIFICATIONS.items()},
        'interval_unit_floor': MINIMUM_BOOTSTRAP_UNITS,
        'label_independence': ('every band is a function of retrieved-homolog statistics and '
                               'frozen group membership only; no measured stability, DMS effect '
                               'or fitted prediction reaches a band boundary'),
        'code_sha256': {str(path.relative_to(ROOT)): digest(path) for path in (
            Path(__file__), ROOT / 'src/capability/context/retrieval_strata.py',
            ROOT / 'src/capability/context/homology.py')},
        'stability': stability_units(args),
        'readout_anchor': readout_units(args),
    }
    declaration['declaration_sha256'] = content_digest(
        {key: declaration[key] for key in ('schema', 'stratifications', 'stability',
                                           'readout_anchor')})
    destination = args.out / DECLARATION_BASENAME
    destination.write_text(json.dumps(declaration, indent=1, sort_keys=True) + '\n')
    print(json.dumps({
        'out': str(destination), 'declaration_sha256': declaration['declaration_sha256'],
        'stability': {'support': declaration['stability']['support'],
                      'bands': {name: {band: entry['groups'] for band, entry in bands.items()}
                                for name, bands in declaration['stability']['band_support'].items()}},
        'readout_anchor': {'support': declaration['readout_anchor']['support'],
                           'bands': {name: {band: entry['clusters'] for band, entry in bands.items()}
                                     for name, bands in
                                     declaration['readout_anchor']['band_support'].items()}},
    }, indent=1))


if __name__ == '__main__':
    main()
