#!/usr/bin/env python
"""Audit the Zenodo 17477631 small-domain aggregation release as a candidate endpoint.

The release is Martell et al. 2025, *Global Analysis of Aggregation Determinants in
Small Protein Domains*, deposited at 10.5281/zenodo.17477631 under CC-BY-4.0. Its
endpoint is

    log2 fold change in soluble abundance = log2( mean normalised reporter-ion
    intensity after stress / mean normalised reporter-ion intensity in the paired
    unstressed control ),

dimensionless, more negative for more aggregation. It never sits beside a free
energy or a growth fitness, so nothing measured here may be pooled with the
kcal/mol stability gates or with a growth readout.

What this script measures, from the deposited files and never from the article:

* the library and replicate structure -- which reporter-ion channel carries which
  condition and which replicate, recovered by exact reproduction of the deposited
  per-condition means and standard deviations, and whether the deposited endpoint
  follows from those channels;
* the acidic-channel correction the release applies to Metagenomic 1, recovered as
  a slope and an intercept from the deposited corrected and uncorrected columns and
  refitted on the shared proteins, because what the correction was calibrated on
  decides whether the corrected channel's agreement is an independent measurement;
* the between-library channel decomposition on the proteins quantified in more than
  one library, with family groups as the resampling unit, both on the release's full
  shared set and on the support that survives exclusion of the sequences already
  held in the staged MGnify stability cohort;
* the quality-control support: the cysteine rule, the peptide counts, the
  quantification rates and the completeness of the reporter-ion channels.

The ratio of the shared-component standard deviation to the per-channel discordance
standard deviation is read against two comparators: the admitted folding-stability
endpoint's 3.86 [3.61, 4.13] and the disqualified MGnify stability endpoint's
1.566 [1.301, 1.847]. The ratio is dimensionless, so it compares across endpoints
whose units differ; the standard deviations themselves do not.

Every decomposition, interval, grouping contract and unit floor is imported from
``src.capability`` rather than restated here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.context.homology import STRATUM_NAMES  # noqa: E402
from src.capability.context.remote_homology import (  # noqa: E402
    STRATUM_UNIT_FLOOR, best_identity, channel_interval, channel_moments, family_groups,
    query_bands)

#: The deposited files this audit reads, with the record's own MD5 digests. A digest
#: mismatch is fatal: a measurement on a file that is not the deposited one is not a
#: measurement of the release.
STAGED_DIGESTS = {
    'Ordered_Libraries.csv': '9f5c0fa31183147ddbb4de14e5004c5c',
    'Overlap_Metagenomic_1_Metagenomic_2.csv': '8d1d764a138197d5b92388bbd3c6a634',
    'IndividualProt_LT_HT_Comparison.csv': '768a4ec43725f497e09fec091331a124',
    'MS_Intensity_AggregationCalcs_Metagenomic_1.csv': '70633154c1e49ee39af90c5d22955028',
    'MS_Intensity_AggregationCalcs_Metagenomic_2.csv': 'df2d938bf81482d9f15254188185fe8f',
    'MS_Intensity_AggregationCalcs_Protein_Fam.csv': '55679c2028184ce94676ed5f4026e637',
    'Metagenomic_dG.csv': '1a5b5e88a41887b211f0fdd5fc4e46fb',
}

#: Library name, its assay table and the label the ordering table gives it.
LIBRARIES = {
    'metagenomic_1': ('MS_Intensity_AggregationCalcs_Metagenomic_1.csv', 'Metagenomic_1'),
    'metagenomic_2': ('MS_Intensity_AggregationCalcs_Metagenomic_2.csv', 'Metagenomic_2'),
    'protein_family': ('MS_Intensity_AggregationCalcs_Protein_Fam.csv', 'Protein_Fam'),
}

#: The five assayed conditions as the deposited column suffixes name them: two
#: temperature stresses against a room-temperature control, one acidic stress
#: against a pH 7 control.
CONDITIONS = ('25', '50', '75', '7', '4')
STRESS_PAIRS = (('50', '25'), ('75', '25'), ('4', '7'))

#: The endpoint's unit, carried into every decomposition key so a ratio of soluble
#: abundance is never published under a free energy's unit.
UNIT = 'log2_fold_change'

#: The two comparators, as the frozen records report them.
COMPARATORS = {
    'admitted_folding_stability': {'point': 3.86, 'ci95': [3.61, 4.13]},
    'disqualified_mgnify_stability': {'point': 1.566, 'ci95': [1.301, 1.847]},
}

#: Metagenomic 1 is the only library whose acidic column the release rescales, so it
#: is the only one with two acidic readings; the others have one, under both names.
ACIDIC_COLUMNS = {
    'metagenomic_1': {'uncorrected': 'log2_fold_change_4_not_corrected_clip_meta_1',
                      'as_released': 'log2_fold_change_4_corrected_clip'},
    'metagenomic_2': {'uncorrected': 'log2_fold_change_4_clip',
                      'as_released': 'log2_fold_change_4_clip'},
    'protein_family': {'uncorrected': 'log2_fold_change_4_clip',
                       'as_released': 'log2_fold_change_4_clip'},
}


def digest(path: Path) -> str:
    """MD5 of one staged file, in the form the Zenodo record publishes."""

    hasher = hashlib.md5()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def verify_digests(root: Path) -> dict:
    """Refuse the audit unless every staged file matches the record's digest."""

    measured = {name: digest(root / name) for name in STAGED_DIGESTS}
    mismatched = sorted(n for n, d in measured.items() if d != STAGED_DIGESTS[n])
    if mismatched:
        raise SystemExit(f'staged files do not match the Zenodo record: {mismatched}')
    return {'source_doi': '10.5281/zenodo.17477631', 'licence': 'CC-BY-4.0',
            'published': '2025-10-29', 'md5': measured}


def read_table(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding='utf-8')))


def channel_columns(rows: list[dict]) -> list[str]:
    return [c for c in rows[0] if c.startswith('norm_total_')]


def channel_assignment(rows: list[dict]) -> dict:
    """Recover which reporter-ion channels carry which condition, from the file.

    The deposited ``avg_norm_total_<condition>`` and ``std_norm_total_<condition>``
    columns are reproduced exactly by one triple of reporter-ion channels, so the
    condition-to-channel map is measured rather than assumed, and so is the spread
    convention: the sample standard deviation over three replicates, which is also
    the convention the release's own individual-protein table uses.
    """

    names = channel_columns(rows)
    matrix = np.array([[float(r[c]) for c in names] for r in rows])
    assignment, unmatched = {}, []
    for condition in CONDITIONS:
        target = np.array([float(r[f'avg_norm_total_{condition}']) for r in rows])
        spread = np.array([float(r[f'std_norm_total_{condition}']) for r in rows])
        tolerance = 1e-9 * max(1.0, float(np.abs(target).max()))
        for triple in itertools.combinations(range(len(names)), 3):
            block = matrix[:, list(triple)]
            if np.max(np.abs(block.mean(axis=1) - target)) > tolerance:
                continue
            assignment[condition] = {
                'channels': [names[i] for i in triple],
                'max_absolute_mean_deviation':
                    float(np.max(np.abs(block.mean(axis=1) - target))),
                'max_absolute_sd_deviation_ddof1':
                    float(np.max(np.abs(block.std(axis=1, ddof=1) - spread))),
            }
            break
        else:
            unmatched.append(condition)
    if unmatched:
        raise SystemExit(f'no channel triple reproduces conditions {unmatched}')
    used = {c for record in assignment.values() for c in record['channels']}
    return {'per_condition': assignment, 'channels_total': len(names),
            'channels_carrying_no_condition': sorted(set(names) - used),
            'replicates_per_condition': 3,
            'spread_convention': 'sample standard deviation, ddof=1, over 3 replicates'}


def endpoint_reproduction(rows: list[dict], assignment: dict) -> dict:
    """Check that the deposited log2 fold changes follow from the deposited channels."""

    def condition_mean(condition: str) -> np.ndarray:
        channels = assignment['per_condition'][condition]['channels']
        return np.array([[float(r[c]) for c in channels] for r in rows]).mean(axis=1)

    report = {}
    for stress, control in STRESS_PAIRS:
        column = (f'log2_avg_norm_total_{stress}_div_avg_norm_total_{control}'
                  '_not_center_not_clip')
        deposited = np.array([float(r[column]) for r in rows])
        derived = np.log2(condition_mean(stress) / condition_mean(control))
        report[f'{stress}_over_{control}'] = float(np.max(np.abs(derived - deposited)))
    return report


def acidic_correction(rows: list[dict], shared: list[dict]) -> dict:
    """Recover the linear correction the release applies to Metagenomic 1's pH 4 values."""

    raw = np.array([float(r[ACIDIC_COLUMNS['metagenomic_1']['uncorrected']]) for r in rows])
    corrected = np.array([float(r[ACIDIC_COLUMNS['metagenomic_1']['as_released']]) for r in rows])
    unclipped = (raw < raw.max() - 1e-12) & (corrected < corrected.max() - 1e-12)
    slope, intercept = np.polyfit(raw[unclipped], corrected[unclipped], 1)
    residual = corrected[unclipped] - (slope * raw[unclipped] + intercept)
    shared_raw = np.array([float(r['log2_fold_change_4_not_corrected_clip_meta_1'])
                           for r in shared])
    shared_other = np.array([float(r['log2_fold_change_4_clip_meta2']) for r in shared])
    fit_slope, fit_intercept = np.polyfit(shared_raw, shared_other, 1)
    return {
        'deployed_transform': {'slope': float(slope), 'intercept': float(intercept),
                               'rows_used': int(unclipped.sum()),
                               'max_absolute_residual': float(np.abs(residual).max())},
        'refit_on_the_shared_proteins': {'slope': float(fit_slope),
                                         'intercept': float(fit_intercept),
                                         'proteins': len(shared)},
        'calibration_set': ('the shared proteins, which are also the only population '
                            'carrying a between-library channel, so the corrected acidic '
                            'column is a fitted alignment of the two channels rather than '
                            'an independent measurement of their agreement'),
    }


def grouped(names: list[str], sequences: dict[str, str]) -> tuple[dict, dict]:
    """Family groups of one population under the frozen 30%/80% contract."""

    return family_groups({name: sequences[name] for name in names})


def decompose(left_table: dict, left_library: str, right_table: dict, right_library: str,
              names: list[str], labels: dict[str, str]) -> dict:
    """Group-bootstrapped channel decomposition over one library pair.

    The resampling unit is the domain's family group, so a draw resamples whole
    groups and the unit count beside every interval is a group count and never a
    protein count.
    """

    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(labels[name], []).append(name)
    cases = {'temperature_50C': ('log2_fold_change_50_clip', 'log2_fold_change_50_clip'),
             'temperature_75C': ('log2_fold_change_75_clip', 'log2_fold_change_75_clip')}
    for variant in ('uncorrected', 'as_released'):
        cases[f'acidic_pH4_{variant}'] = (ACIDIC_COLUMNS[left_library][variant],
                                          ACIDIC_COLUMNS[right_library][variant])
    if cases['acidic_pH4_uncorrected'] == cases['acidic_pH4_as_released']:
        cases['acidic_pH4'] = cases.pop('acidic_pH4_uncorrected')
        cases.pop('acidic_pH4_as_released')
    report = {}
    for case, (left, right) in cases.items():
        vectors = [channel_moments(
            np.array([float(left_table[n][left]) for n in members]),
            np.array([float(right_table[n][right]) for n in members]))
            for members in groups.values()]
        report[case] = channel_interval(vectors, channels=(left_library, right_library),
                                        unit=UNIT)
    return report


def tryptic_fragments(sequence: str) -> list[str]:
    """Fully tryptic fragments: cleave after K or R except before P.

    Used only to bound how many distinct peptides a domain of at most 71 residues
    can yield, which is what shows the deposited count columns are not a
    distinct-peptide count.
    """

    return [part for part in re.split(r'(?<=[KR])(?!P)', sequence) if part]


def quality_control(root: Path, ordered: list[dict], tables: dict) -> dict:
    """What the release filtered, what it left in, and what is recoverable."""

    by_library: dict[str, list[dict]] = {}
    for row in ordered:
        by_library.setdefault(row['Library'], []).append(row)
    lengths = [len(r['protein_sequence']) for r in ordered]
    placements: dict[str, set[str]] = {}
    for row in ordered:
        placements.setdefault(row['protein_sequence'], set()).add(row['Library'])
    report = {
        'ordered_constructs': len(ordered),
        'ordered_by_library': {k: len(v) for k, v in by_library.items()},
        'sequences_ordered_into_more_than_one_library':
            sum(1 for libs in placements.values() if len(libs) > 1),
        'cysteine_containing_ordered_constructs':
            sum(1 for r in ordered if 'C' in r['protein_sequence']),
        'construct_length_aa': {'min': min(lengths), 'median': float(np.median(lengths)),
                                'max': max(lengths)},
        'per_library': {},
    }
    for library, (filename, label) in LIBRARIES.items():
        rows = read_table(root / filename)
        ordered_names = {r['name'] for r in by_library[label]}
        counts = np.array([float(r['SEQ_COUNT']) for r in rows])
        fragments = np.array([len(tryptic_fragments(r['protein_sequence'])) for r in rows])
        detectable = np.array([sum(1 for p in tryptic_fragments(r['protein_sequence'])
                                   if 7 <= len(p) <= 35) for r in rows])
        matrix = np.array([[float(r[c]) for c in channel_columns(rows)] for r in rows])
        report['per_library'][library] = {
            'ordered': len(ordered_names),
            'quantified': len(rows),
            'quantification_rate': float(len(rows) / len(ordered_names)),
            'quantified_but_not_ordered': len(set(tables[library]) - ordered_names),
            'seq_count_median': float(np.median(counts)),
            'seq_count_max': float(counts.max()),
            'rows_with_seq_count_one': int((counts == 1).sum()),
            'rows_where_spec_count_equals_seq_count':
                sum(1 for r in rows if r['SPEC_COUNT'] == r['SEQ_COUNT']),
            'rows_where_seq_count_exceeds_tryptic_fragments':
                int((counts > fragments).sum()),
            'tryptic_fragments_median': float(np.median(fragments)),
            'rows_with_fewer_than_two_detectable_tryptic_fragments':
                int((detectable < 2).sum()),
            'cysteine_containing_quantified':
                sum(1 for r in rows if 'C' in r['protein_sequence']),
            'rows_with_a_nonpositive_reporter_channel': int((matrix <= 0).any(axis=1).sum()),
        }
    report['peptide_support_reading'] = (
        'SPEC_COUNT and SEQ_COUNT hold the same value in every row of every library, and in '
        'the majority of rows -- the per-library counts are above -- that value exceeds the '
        'number of fully tryptic fragments the domain can yield, so the pair records one '
        'spectral-support quantity and the release deposits no distinct-peptide count. The '
        'selection criterion of at least two unique tryptic peptides of 7 to 35 residues is '
        'a design criterion on predicted peptides and cannot be re-verified per quantified '
        'protein from the deposited files.')
    return report


#: A de novo designed miniproteome name carries its topology and a design round,
#: so it is recognisable without a sequence lookup. Such a construct has no
#: evolutionary family, which is what makes its identity band uninformative.
DESIGNED_NAME = re.compile(r'^(EEHEE|EHEE|HEEH|HHH)[_0-9]')


def uniref50_bands(hits: Path, populations: dict[str, list[str]]) -> dict:
    """Query-level identity bands against the searched UniRef50 snapshot.

    Identity is ``100 * nident / qlen`` and the bands are the frozen declaration's,
    imported rather than restated. These are **query-level** counts: the independent
    unit of any remoteness claim is the family group, and a group count is not
    derivable from a query-level distribution. For a metagenomic cohort the band
    also runs in the inflating direction, because six roster arms declare corpora
    containing metagenome-derived sequence -- progen2-small, progen2-medium,
    progen2-large and progen2-xlarge on UniRef90 plus BFD30, proteinglm-7b-clm on
    UniRef50s plus UniRef90 plus ColabFoldDB, and progen3-3b on the Profluent
    protein atlas -- so a low identity to UniRef50 is weaker evidence of remoteness
    for those six arms than for a cohort of curated domains.
    """

    every = sorted({name for names in populations.values() for name in names})
    best = best_identity(hits, every)
    bands = query_bands(best)
    report = {'definition': '100 * nident / qlen against uniref50_full.dmnd',
              'level': 'query, not family group',
              'populations': {}}
    for tag, names in populations.items():
        split = {'all': names,
                 'de_novo_designed': [n for n in names if DESIGNED_NAME.match(n)],
                 'natural_accession': [n for n in names if not DESIGNED_NAME.match(n)]}
        report['populations'][tag] = {
            part: {'queries': len(members),
                   'bands': {band: int(Counter(bands[n] for n in members).get(band, 0))
                             for band in STRATUM_NAMES},
                   'no_reported_hit': sum(1 for n in members if best[n] is None)}
            for part, members in split.items() if members}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', required=True, type=Path,
                        help='directory holding the deposited files')
    parser.add_argument('--excluded-names', required=True, type=Path,
                        help='JSON carrying excluded_aggregation_names: the sequences '
                             'already held in the staged MGnify stability cohort')
    parser.add_argument('--uniref50-hits', type=Path,
                        help='DIAMOND tabular hits of every domain against '
                             'uniref50_full.dmnd; when given, the query-level identity '
                             'band distribution is reported')
    parser.add_argument('--out', required=True, type=Path)
    arguments = parser.parse_args()

    record = verify_digests(arguments.staged)
    tables = {library: {r['name']: r for r in read_table(arguments.staged / filename)}
              for library, (filename, _) in LIBRARIES.items()}
    shared_table = read_table(arguments.staged / 'Overlap_Metagenomic_1_Metagenomic_2.csv')
    ordered = read_table(arguments.staged / 'Ordered_Libraries.csv')
    excluded = set(json.loads(arguments.excluded_names.read_text())['excluded_aggregation_names'])
    sequences = {name: row['protein_sequence']
                 for table in tables.values() for name, row in table.items()}

    metagenomic_rows = read_table(arguments.staged / LIBRARIES['metagenomic_1'][0])
    assignment = channel_assignment(metagenomic_rows)

    shared = sorted(set(tables['metagenomic_1']) & set(tables['metagenomic_2']))
    surviving = sorted(n for n in shared if n not in excluded)
    three_library = sorted(set(shared) & set(tables['protein_family']))

    labels_shared, groups_shared = grouped(shared, sequences)
    labels_surviving, groups_surviving = grouped(surviving, sequences)
    labels_three, groups_three = grouped(three_library, sequences)

    result = {
        'schema_version': 'aggregation_endpoint_audit',
        'endpoint': ('log2 fold change in soluble abundance, dimensionless; never beside '
                     'a free energy or a growth fitness'),
        'record': record,
        'library_and_replicate_structure': {
            'reporter_ion_channels': assignment,
            'endpoint_reproducible_from_channels': endpoint_reproduction(
                metagenomic_rows, assignment),
            'quantified_in_both_metagenomic_libraries': len(shared),
            'quantified_in_all_three_libraries': len(three_library),
        },
        'acidic_correction': acidic_correction(metagenomic_rows, shared_table),
        'family_groups': {
            'contract': groups_shared['edge_rule'],
            'unit_floor': STRATUM_UNIT_FLOOR,
            'shared_set': groups_shared,
            'surviving_set': groups_surviving,
            'three_library_set': groups_three,
        },
        'between_library_channel': {
            'comparators': COMPARATORS,
            'resampling_unit': 'family group under the frozen 30%/80% single-linkage contract',
            'shared_set': decompose(tables['metagenomic_1'], 'metagenomic_1',
                                    tables['metagenomic_2'], 'metagenomic_2',
                                    shared, labels_shared),
            'surviving_set': decompose(tables['metagenomic_1'], 'metagenomic_1',
                                       tables['metagenomic_2'], 'metagenomic_2',
                                       surviving, labels_surviving),
        },
        'third_library_channel': {
            'population': len(three_library),
            'metagenomic_1_against_protein_family': decompose(
                tables['metagenomic_1'], 'metagenomic_1',
                tables['protein_family'], 'protein_family', three_library, labels_three),
            'metagenomic_2_against_protein_family': decompose(
                tables['metagenomic_2'], 'metagenomic_2',
                tables['protein_family'], 'protein_family', three_library, labels_three),
        },
        'quality_control': quality_control(arguments.staged, ordered, tables),
        'overlap_exclusion': {
            'excluded_sequences': len(excluded),
            'shared_pairs_before': len(shared),
            'shared_pairs_after': len(surviving),
        },
    }
    if arguments.uniref50_hits is not None:
        every = sorted(sequences)
        result['uniref50_bands'] = uniref50_bands(
            arguments.uniref50_hits,
            {'every_quantified_domain': every,
             'support_surviving_the_overlap_exclusion':
                 sorted(n for n in every if n not in excluded)})
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(result, indent=1, sort_keys=True) + '\n')
    print(f'wrote {arguments.out}')


if __name__ == '__main__':
    main()
