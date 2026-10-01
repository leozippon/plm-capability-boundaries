#!/usr/bin/env python3
"""Declare the label-blind support of the external confirmation on Domainome 1.0.

Everything this entry point writes is fixed before any predictor is fitted and
before any model is scored, and nothing it decides reads a measured fitness, a
standard error or a model quantity.

Four rules do the selecting.

* **Development overlap is removed by sequence.** The 28 domains the endpoint
  qualification found to match a ProteinGym target or a Tsuboyama background
  exactly or by containment are dropped by name. An identifier check would have
  found none of them, because no ProteinGym entry names Beltran or the
  domainome, so an identifier-only screen would have silently contaminated this
  confirmation with development data.
* **A declared alignment screen** then refuses any remaining wild type that
  reaches 50% identity over 80% of its own length against any of the 217
  ProteinGym targets or the 478 Tsuboyama backgrounds. That is the readout
  cohort's own wild-type-family rule, so the screen removes a domain that would
  sit in a development target's family under the development panel's own
  definition. The frozen grouping contract's weaker 30% edge is reported as a
  declared stratum rather than applied.
* **Family grouping** unions exact sequence identity, shared UniProt accession,
  shared Pfam accession and the frozen contract's own 30%/80% alignment edge.
* **A variant cap** of at most :data:`VARIANT_CAP` substitutions per domain,
  drawn by a stable SHA-256 of the draw seed, the domain identifier and the
  mutant sequence, over domains carrying at least :data:`MIN_VARIANTS`.

It writes the cohort with its measurements, a label-free extraction plan
carrying sequences and state indices only, and the support declaration with
every digest and every exclusion count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.mutation.external_confirmation import (
    DEVELOPMENT_COVERAGE, DEVELOPMENT_IDENTITY, DEVELOPMENT_STRATUM_IDENTITY,
    DRAW_SEED, ENDPOINT, FEATURE_BLOCKS, MEASURED_QUANTITY, MIN_VARIANTS, POSITION_CONVENTION, PROJECTION_DIM,
    PROJECTION_SEED, SOURCE_ARCHIVE, SOURCE_ARCHIVE_SHA256, SOURCE_DOI,
    SOURCE_MEMBER, UNION_SOURCES, VARIANT_CAP, accept_rows, alignment_statistics,
    declaration_digest, draw_order, endpoint_digest, family_groups,
    pfam_accession, source_frame, substitution_records, wildtype_rows)
from src.capability.core.io import sha256_file, write_json

PLAN_SCHEMA = 'stability_singles_extraction_v1'
COHORT_SCHEMA = 'external_confirmation_cohort_v1'
DECLARATION_SCHEMA = 'external_confirmation_support_v1'

#: The tabular fields the screen reads, in DIAMOND's own order.
SCREEN_FIELDS = ('qseqid', 'sseqid', 'pident', 'length', 'nident',
                 'qstart', 'qend', 'qlen', 'slen', 'evalue', 'bitscore')


def plan_digest(plan: dict) -> str:
    return hashlib.sha256(json.dumps(
        {k: plan[k] for k in ('schema', 'cohort_sha256', 'projection', 'backgrounds')},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def proteingym_targets(directory: Path) -> dict[str, str]:
    """Reconstruct each ProteinGym substitution assay's target sequence.

    The reference CSVs carry mutant sequences rather than the target, so the
    target is recovered by reverting the first row's substitutions. Every
    reverted position is checked against the mutant string, so a file whose
    coordinates disagree with its own sequence is refused.
    """

    import pandas as pd

    targets = {}
    for path in sorted(directory.glob('*.csv')):
        frame = pd.read_csv(path, nrows=1)
        mutant, sequence = str(frame['mutant'].iloc[0]), list(str(frame['mutated_sequence'].iloc[0]))
        for token in mutant.split(':'):
            wild, position, substituted = token[0], int(token[1:-1]), token[-1]
            if sequence[position - 1] != substituted:
                raise ValueError(f'{path.name}: {token} disagrees with its mutant sequence')
            sequence[position - 1] = wild
        targets[path.stem] = ''.join(sequence)
    return targets


def tsuboyama_backgrounds(path: Path) -> dict[str, str]:
    """Wild-type sequence of every catalogue background of the pinned index."""

    records = json.loads(path.read_text())
    return {str(row['WT_name']): str(row['sequence']) for row in records}


def write_fasta(path: Path, sequences: dict[str, str]) -> str:
    path.write_text(''.join(f'>{name}\n{sequence}\n' for name, sequence
                            in sorted(sequences.items())))
    return sha256_file(path)


def diamond_screen(diamond: Path, queries: dict[str, str], subjects: dict[str, str],
                   work: Path, *, threads: int) -> tuple[list[dict], dict]:
    """Search the retained wild types against the development targets.

    ``--very-sensitive`` with ``--masking 0`` is the repository's setting wherever
    the claim that matters is a negative one: a fast or masked search cannot
    support a statement that a sequence has no close development relative.
    """

    work.mkdir(parents=True, exist_ok=True)
    query_fasta, subject_fasta = work / 'retained.faa', work / 'development.faa'
    query_sha = write_fasta(query_fasta, queries)
    subject_sha = write_fasta(subject_fasta, subjects)
    database = work / 'development.dmnd'
    makedb = [str(diamond), 'makedb', '--in', str(subject_fasta), '--db', str(database),
              '--threads', str(threads)]
    subprocess.run(makedb, check=True, capture_output=True, text=True)
    hits_path = work / 'screen_hits.tsv'
    command = [str(diamond), 'blastp', '--db', str(database), '--query', str(query_fasta),
               '--out', str(hits_path), '--outfmt', '6', *SCREEN_FIELDS,
               '--evalue', '0.001', '--max-target-seqs', '5000',
               '--threads', str(threads), '--masking', '0', '--very-sensitive']
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    hits = []
    for line in hits_path.read_text().splitlines():
        fields = line.split('\t')
        row = dict(zip(SCREEN_FIELDS, fields))
        qlen = int(row['qlen'])
        hits.append({
            'query': row['qseqid'], 'subject': row['sseqid'],
            'pident': float(row['pident']), 'columns': int(row['length']),
            'nident': int(row['nident']), 'qlen': qlen,
            'query_coverage': 100.0 * (int(row['qend']) - int(row['qstart']) + 1) / qlen,
            'identity_over_query': 100.0 * int(row['nident']) / qlen,
            'evalue': float(row['evalue']), 'bitscore': float(row['bitscore']),
        })
    manifest = {
        'tool': {'executable': str(diamond), 'binary_sha256': sha256_file(diamond),
                 'version': subprocess.run([str(diamond), '--version'], check=True,
                                           capture_output=True, text=True).stdout.strip()},
        'query': {'records': len(queries), 'sha256': query_sha},
        'subjects': {'records': len(subjects), 'sha256': subject_sha},
        'command': command, 'makedb_command': makedb,
        'fields': list(SCREEN_FIELDS),
        'log_tail': completed.stderr.strip().splitlines()[-3:],
        'hits': {'rows': len(hits), 'sha256': sha256_file(hits_path)},
    }
    return hits, manifest


def exact_screen(queries: dict[str, str], subjects: dict[str, str]) -> list[dict]:
    """Exact Smith-Waterman screen against the short development backgrounds.

    Every Tsuboyama background is under 128 residues, as is every retained
    domain, so this side of the screen rests on exact dynamic programming under
    the frozen contract's scoring rather than on a seeded heuristic. The DIAMOND
    screen covers the ProteinGym targets, which are whole proteins.
    """

    query_names, subject_names = sorted(queries), sorted(subjects)
    sequences = [queries[name] for name in query_names] + [subjects[name] for name in subject_names]
    offset = len(query_names)
    pairs = np.array([[i, offset + j] for i in range(offset)
                      for j in range(len(subject_names))], dtype=np.int64)
    statistics = alignment_statistics(sequences, pairs)
    identity = np.asarray(statistics.percent_identity, dtype=float)
    coverage_a = np.asarray(statistics.coverage_a, dtype=float)
    coverage_b = np.asarray(statistics.coverage_b, dtype=float)
    out = []
    for row in range(len(pairs)):
        if identity[row] < DEVELOPMENT_STRATUM_IDENTITY:
            continue
        out.append({'query': query_names[int(pairs[row, 0])],
                    'subject': subject_names[int(pairs[row, 1]) - offset],
                    'pident': float(identity[row]),
                    'query_coverage': float(coverage_a[row]),
                    'subject_coverage': float(coverage_b[row])})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/domainome_beltran2025')
    parser.add_argument('--registry', type=Path, default=ROOT / 'data/dataset_registry.json')
    parser.add_argument('--qualification', type=Path,
                        default=ROOT / 'data/endpoint_qualification_20260924.json')
    parser.add_argument('--proteingym', type=Path,
                        default=ROOT / 'data/proteingym/DMS_ProteinGym_substitutions')
    parser.add_argument('--megascale-index', type=Path,
                        default=ROOT / 'results/shared/megascale_disjointness_20260813/query_index.json')
    parser.add_argument('--diamond', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive = args.data_dir / SOURCE_ARCHIVE
    archive_sha = sha256_file(archive)
    if archive_sha != SOURCE_ARCHIVE_SHA256:
        raise SystemExit(f'{archive}: digest {archive_sha} is not the registered '
                         f'{SOURCE_ARCHIVE_SHA256}')
    registry_sha = sha256_file(args.registry)
    qualification_sha = sha256_file(args.qualification)
    qualification = json.loads(args.qualification.read_text())
    endpoint = qualification['endpoints']['domainome_beltran2025']

    overlap = set()
    for key in ('overlap_with_proteingym', 'overlap_with_tsuboyama2023'):
        block = endpoint['measured_support'][key]
        for field in ('exact_sequence_matches', 'containment_matches'):
            overlap.update(str(pair[0]) for pair in block[field])
    print(f'development overlap by sequence: {len(overlap)} domains', flush=True)

    frame = source_frame(archive)
    wild_rows = wildtype_rows(frame)
    wildtypes = {str(name): str(sequence) for name, sequence
                 in zip(wild_rows['domain_ID'], wild_rows['aa_seq'])}
    ranks = {str(name): int(rank) for name, rank in zip(wild_rows['domain_ID'],
                                                        wild_rows['quality_rank'])}
    mask, accounting = accept_rows(frame, wildtypes)
    records = substitution_records(frame, mask, wildtypes)
    accessions = {row['domain']: row['accession'] for row in records}
    print(f"accepted {accounting['accepted_rows']} substitution rows over "
          f'{len(set(r["domain"] for r in records))} domains', flush=True)

    # Scale anchors, measured rather than assumed: the wild-type reference and
    # the nonsense level that fix the endpoint's two normalisation points.
    stop = (frame['STOP'].astype('object') == True).to_numpy()  # noqa: E712
    finite = np.isfinite(frame['normalized_fitness'].to_numpy(dtype=float))
    nonsense = frame.loc[stop & finite, ['domain_ID', 'normalized_fitness']]
    nonsense_median = nonsense.groupby('domain_ID')['normalized_fitness'].median()
    anchors = {
        'wildtype_normalized_fitness_max_absolute': float(
            np.max(np.abs(wild_rows['normalized_fitness'].to_numpy(dtype=float)))),
        'wildtype_rows_exactly_zero': int(
            (wild_rows['normalized_fitness'].to_numpy(dtype=float) == 0.0).sum()),
        'wildtype_rows': int(len(wild_rows)),
        'per_domain_median_nonsense_normalized_fitness': {
            'domains': int(len(nonsense_median)),
            'median': float(nonsense_median.median()),
            'q1': float(nonsense_median.quantile(0.25)),
            'q3': float(nonsense_median.quantile(0.75)),
        },
    }

    retained = {name: sequence for name, sequence in wildtypes.items()
                if name not in overlap}
    gym = proteingym_targets(args.proteingym)
    megascale = tsuboyama_backgrounds(args.megascale_index)
    development = {f'gym|{k}': v for k, v in gym.items()}
    development.update({f'megascale|{k}': v for k, v in megascale.items()})
    hits, screen_manifest = diamond_screen(args.diamond, retained, development,
                                           args.out_dir / 'screen', threads=args.threads)
    crossing = sorted({hit['query'] for hit in hits
                       if hit['pident'] >= DEVELOPMENT_IDENTITY
                       and hit['query_coverage'] >= DEVELOPMENT_COVERAGE})
    stratum = sorted({hit['query'] for hit in hits
                      if hit['pident'] >= DEVELOPMENT_STRATUM_IDENTITY
                      and hit['query_coverage'] >= DEVELOPMENT_COVERAGE})
    exact = exact_screen(retained, megascale)
    exact_crossing = sorted({row['query'] for row in exact
                             if row['pident'] >= DEVELOPMENT_IDENTITY
                             and row['query_coverage'] >= DEVELOPMENT_COVERAGE})
    screened = {name: sequence for name, sequence in retained.items()
                if name not in crossing and name not in exact_crossing}
    print(f'alignment screen: {len(crossing)} domains cross the declared rule '
          f'({len(exact_crossing)} of them under the exact screen), '
          f'{len(stratum)} cross the reported 30% stratum', flush=True)

    per_domain: dict[str, list[dict]] = {}
    for row in records:
        if row['domain'] in screened:
            per_domain.setdefault(row['domain'], []).append(row)
    eligible = {name: rows for name, rows in per_domain.items() if len(rows) >= MIN_VARIANTS}
    below = sorted(set(per_domain) - set(eligible))
    labels, grouping = family_groups(sorted(eligible), screened, accessions)

    domains = []
    for name in sorted(eligible):
        rows = {row['sequence']: row for row in eligible[name]}
        order = draw_order(name, sorted(rows), seed=DRAW_SEED)[:VARIANT_CAP]
        variants = [rows[sequence] for sequence in order]
        variants.sort(key=lambda r: (r['position'], r['mutant']))
        domains.append({
            'name': name, 'group': labels[name], 'accession': accessions[name],
            'pfam': pfam_accession(name), 'quality_rank': ranks[name],
            'wildtype': screened[name], 'length': len(screened[name]),
            'eligible_variants': len(rows),
            'variants': [{'position': row['position'], 'mutant': row['mutant'],
                          'sequence': row['sequence'], 'target': row['target'],
                          'uncertainty': row['sigma']} for row in variants],
        })
    drawn = [{'domain': entry['name'], **row}
             for entry in domains for row in entry['variants']]
    sites = {f"{entry['name']}:{variant['position']}"
             for entry in domains for variant in entry['variants']}
    cohort = {
        'schema': COHORT_SCHEMA, 'endpoint': ENDPOINT,
        'measured_quantity': MEASURED_QUANTITY,
        'source': {'doi': SOURCE_DOI, 'archive': SOURCE_ARCHIVE,
                   'archive_sha256': archive_sha, 'member': SOURCE_MEMBER},
        'draw': {'seed': DRAW_SEED, 'variant_cap': VARIANT_CAP,
                 'minimum_variants': MIN_VARIANTS},
        'uncertainty_source': ('the source\'s own `normalized_fitness_sigma`, the standard '
                               'error of the row\'s normalised fitness; it is carried as '
                               'reported and is never weighted, shrunk or thresholded on'),
        'domains': domains,
    }
    # Named as the stability and remote-homology cohorts name it, so one loader
    # reads all three: the digest over this cohort's own endpoint values.
    cohort['endpoint_sha256'] = endpoint_digest(drawn) if drawn else None
    cohort_path = args.out_dir / 'cohort.json'
    write_json(cohort_path, cohort)
    cohort_sha = sha256_file(cohort_path)

    plan = {
        'schema': PLAN_SCHEMA, 'cohort_sha256': cohort_sha,
        'projection': {'seed': PROJECTION_SEED, 'dim': PROJECTION_DIM,
                       'blocks': list(FEATURE_BLOCKS)},
        'backgrounds': [{
            'name': entry['name'], 'group': entry['group'],
            'wildtype': entry['wildtype'],
            'sequences': [entry['wildtype']] + [v['sequence'] for v in entry['variants']],
            'variants': [{'state': index + 1, 'position': v['position'],
                          'mutant': v['mutant']}
                         for index, v in enumerate(entry['variants'])],
        } for entry in domains],
    }
    plan_path = args.out_dir / 'extraction_plan.json'
    write_json(plan_path, plan)

    declaration = {
        'schema': DECLARATION_SCHEMA,
        'declared_utc': datetime.now(timezone.utc).isoformat(),
        'endpoint': ENDPOINT, 'measured_quantity': MEASURED_QUANTITY,
        'not_a_replication_of': (
            'proteolysis-derived stability (Tsuboyama 2023) or ProteinGym DMS fitness; '
            'abundance by complementation in yeast is a different measured quantity, so '
            'agreement is cross-quantity corroboration and not a repeat of either estimand'),
        'inputs': {
            'source_archive_sha256': archive_sha,
            'dataset_registry_sha256': registry_sha,
            'endpoint_qualification_sha256': qualification_sha,
            'proteingym_reference_assays': len(gym),
            'megascale_catalogue_backgrounds': len(megascale),
        },
        'position_convention': POSITION_CONVENTION,
        'row_accounting': accounting,
        'scale_anchors': anchors,
        'development_overlap': {
            'excluded_by_sequence': sorted(overlap),
            'excluded_by_sequence_count': len(overlap),
            'identifier_check_would_have_found': 0,
            'note': ('no ProteinGym entry names Beltran or the domainome, so the exclusion '
                     'rests on sequence matching and an identifier-only screen would have '
                     'left every one of these domains in the confirmation support'),
        },
        'alignment_screen': {
            'rule': (f'refuse a retained wild type reaching percent identity >= '
                     f'{DEVELOPMENT_IDENTITY} with query coverage >= {DEVELOPMENT_COVERAGE} '
                     f'against any development target; this is the readout cohort own '
                     f'wild-type-family rule'),
            'diamond': screen_manifest,
            'crossing_domains': crossing,
            'crossing_domains_exact_screen': exact_crossing,
            'reported_stratum_identity': DEVELOPMENT_STRATUM_IDENTITY,
            'crossing_domains_reported_stratum': stratum,
            'exact_screen_rows_at_or_above_stratum': len(exact),
            'maximum_identity_over_query_any_hit': (
                max((hit['identity_over_query'] for hit in hits), default=0.0)),
        },
        'support': {
            'domains_in_source': len(wildtypes),
            'domains_after_overlap_exclusion': len(retained),
            'domains_after_alignment_screen': len(screened),
            'domains_below_minimum_variants': below,
            'domains_retained': len(domains),
            'family_groups': grouping['groups'],
            'accessions_retained': len(sorted({entry['accession'] for entry in domains})),
            'pfam_accessions_retained': len(sorted({entry['pfam'] for entry in domains})),
            'variants_eligible': sum(entry['eligible_variants'] for entry in domains),
            'variants_drawn': len(drawn),
            'mutated_sites': len(sites),
            'length_range': [min(entry['length'] for entry in domains),
                             max(entry['length'] for entry in domains)],
            'quality_rank_range': [min(entry['quality_rank'] for entry in domains),
                                   max(entry['quality_rank'] for entry in domains)],
            'quality_rank_median': float(np.median([entry['quality_rank'] for entry in domains])),
            'sequences_per_arm': sum(1 + len(entry['variants']) for entry in domains),
            'residues_per_arm': sum((1 + len(entry['variants'])) * entry['length']
                                    for entry in domains),
        },
        'grouping': grouping,
        'union_sources': list(UNION_SOURCES),
        'grouping_difference_from_the_frozen_contract': (
            'the frozen contract cannot express two of these sources for this dataset. Two '
            'domains cut from one UniProt accession are unioned, and the Pfam label is the '
            'source library design annotation carried in the domain identifier rather than an '
            'hmmscan call, so every domain carries exactly one accession and there is no '
            'unlabelled stratum; the contract had no Pfam label for 179 of its 478 '
            'backgrounds. The alignment edge rule, its scoring and its thresholds are the '
            'contract as delivered, and the design stratum and source-cluster sources do '
            'not apply because every retained domain is a natural human domain'),
        'endpoint_digest': cohort['endpoint_sha256'],
        'cohort_sha256': cohort_sha,
        'plan_content_sha256': plan_digest(plan),
        'plan_sha256': sha256_file(plan_path),
    }
    declaration['declaration_digest'] = declaration_digest(
        {k: declaration[k] for k in ('schema', 'endpoint', 'position_convention',
                                     'union_sources', 'cohort_sha256',
                                     'plan_content_sha256', 'endpoint_digest')})
    write_json(args.out_dir / 'support_declaration.json', declaration)
    print(json.dumps(declaration['support'], indent=1, sort_keys=True), flush=True)
    print(f"plan content digest {declaration['plan_content_sha256']}", flush=True)


if __name__ == '__main__':
    main()
