#!/usr/bin/env python3
"""Support declaration and measurement-noise floor for the higher-order gate on ProteinGym.

Two stages, run in that order, because the endpoint has to be chosen on support
and power rather than on a label outcome.

``support`` reads the pinned ProteinGym raw archive, reconciles each raw assay
against the processed benchmark CSV by variant identifier and count, and declares
the complete-cube support: which orders carry cubes whose every non-empty subset
and whose wild-type corner are measured, and how many *site tuples* those cubes
rest on. Site tuples and assays are the independent units; a cube count is not a
sample size. No replicate value is read into the admission rule.

``noise-floor`` re-derives that support, refuses to proceed unless the declared
digest still matches, and then measures the order-k cycle separately in each
replicate channel. It reports the between-channel decomposition and, independently,
the floor propagated from the measured per-state channel dispersion: an order-k
cycle sums ``2 ** k`` corner estimates with unit weights, so if corner errors are
independent with standard deviation ``sigma`` the channel-difference standard
deviation is ``sqrt(2 ** (k + 1)) * sigma`` and the per-channel discordance is
``sqrt(2 ** k) * sigma``. Measured and propagated are both reported, because their
agreement is what establishes that the corner errors behave independently.

Scale and estimand. The channel value is ``log10(rep_i)``, the study's own
per-replicate brightness normalised by its block's wild-type mean, so the endpoint
is an order-k cycle in log10 brightness units. The raw file carries the wild-type
state that the processed CSV drops, so the cycle is wild-type-centred and absolute
on that scale rather than a within-assay centred residual. Nonadditivity on a
declared transformed assay scale is not by itself a molecular interaction, and the
floor is measured on the same scale as the endpoint so that the comparison is
internally consistent.

No model score, likelihood or representation enters any quantity here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.mutation.higher_order_cycle import (  # noqa: E402
    cycle_contrast, kish_effective_units)
from src.capability.core.io import sha256_file  # noqa: E402
from src.capability.core.statistics import bootstrap_unit_floor  # noqa: E402
from src.capability.mutation.proteingym_higher_order import (  # noqa: E402
    RESIDUES, WILD_TYPE_STATE, AssayStates, channel_decomposition_bootstrap,
    complete_cubes, cube_addresses, cube_contrast, genotype_rank, parse_mutant,
    split_channels,
)

#: The pinned raw archive and the processed benchmark, with the digests the
#: declaration is bound to. ``substitutions_raw_DMS.zip`` is ProteinGym v1.3's
#: pre-preprocessing bundle; the processed CSVs are the local benchmark copy.
RAW_ARCHIVE = Path('data/proteingym_raw/substitutions_raw_DMS.zip')
RAW_ARCHIVE_SHA256 = '6d83b16585de2b71b67ae1985193b9eec2e01804784286c515ff276b5372e412'
REFERENCE_FILE = Path('data/proteingym_raw/DMS_substitutions.csv')
REFERENCE_SHA256 = 'a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308'
RAW_DIR = Path('data/proteingym_raw/raw_selected')
PROCESSED_DIR = Path('data/proteingym/DMS_ProteinGym_substitutions')

#: The one assay in the local panel that carries both substantial complete-cube
#: support and a genuine replicate channel structure. ``rep1..rep3`` are three
#: independent FACS replicates and the raw file carries the wild-type state.
ASSAY = 'PHOT_CHLRE_Chen_2023'
CHANNELS = ('rep1', 'rep2', 'rep3')

#: ``rep_i`` is normalised by its measurement block's wild-type replicate mean, so
#: ``log_rep_i - log10(rep_i)`` recovers that block's wild-type log brightness and
#: identifies the block of every row. Two blocks exist. A cycle's corners must come
#: from one block: a per-block offset cancels from the contrast only when every
#: corner carries the same one.
BLOCK_KEY_DECIMALS = 4

#: Orders the census and the floor report. Order 2 is included as the reference
#: point, because it is the order the admitted pairwise endpoint measures.
ORDERS = (2, 3, 4, 5, 6, 7, 8, 9)

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260923


# ------------------------------------------------------------------- raw reading

def _verify_digests() -> dict:
    """Bind the declaration to the retrieved bytes before anything is read."""
    out = {}
    for name, path, expected in (
            ('raw_archive', RAW_ARCHIVE, RAW_ARCHIVE_SHA256),
            ('reference_file', REFERENCE_FILE, REFERENCE_SHA256)):
        digest = sha256_file(ROOT / path)
        if digest != expected:
            raise ValueError(f'{path} digest {digest} differs from the declared {expected}')
        out[name] = {'path': str(path), 'sha256': digest,
                     'bytes': (ROOT / path).stat().st_size}
    return out


def read_phot_blocks() -> tuple[dict[float, dict[tuple, tuple[float, float, float]]], dict]:
    """Per-block measured states of the raw PHOT assay, in log10 brightness.

    One exclusion, declared and counted rather than silent: the raw file carries
    nonsense variants whose token names a stop codon as ``*``, which the processed
    benchmark CSV excludes. A truncated chain is not a substitution state and its
    brightness is not a corner of a substitution cube, so those rows are dropped
    and their number reported. Any other unparsable label is an error.

    Otherwise this refuses rather than repairs. A row whose three channels disagree
    on the block offset, a non-positive replicate, a repeated state inside one block
    and a block without its wild-type state are all errors: each would make a cycle
    contrast read something other than one block's own measurements.
    """
    path = ROOT / RAW_DIR / f'{ASSAY}.csv'
    blocks: dict[float, dict[tuple, tuple[float, float, float]]] = {}
    excluded = Counter()
    with open(path, newline='', encoding='utf-8') as handle:
        for record in csv.DictReader(handle):
            label = record['mutant']
            if label.upper() == 'WT':
                state = WILD_TYPE_STATE
            elif '*' in label:
                excluded['nonsense'] += 1
                continue
            else:
                state = parse_mutant(label).substitutions
            values = [float(record[channel]) for channel in CHANNELS]
            if min(values) <= 0:
                raise ValueError(f'{ASSAY}: non-positive replicate in row {record}')
            logs = [math.log10(value) for value in values]
            offsets = {round(float(record[f'log_{channel}']) - log, BLOCK_KEY_DECIMALS)
                       for channel, log in zip(CHANNELS, logs)}
            if len(offsets) != 1:
                raise ValueError(f'{ASSAY}: row {label} carries {len(offsets)} '
                                 'block offsets across its three channels')
            states = blocks.setdefault(offsets.pop(), {})
            if state in states:
                raise ValueError(f'{ASSAY}: a block repeats state {label}')
            states[state] = tuple(logs)
    for block, states in blocks.items():
        if WILD_TYPE_STATE not in states:
            raise ValueError(f'{ASSAY}: block {block} carries no wild-type state')
    return blocks, dict(excluded)


def read_processed_keys(assay: str) -> set[tuple]:
    """Variant keys of the processed benchmark CSV, for raw/processed reconciliation."""
    keys = set()
    with open(ROOT / PROCESSED_DIR / f'{assay}.csv', newline='', encoding='utf-8') as handle:
        for record in csv.DictReader(handle):
            keys.add(parse_mutant(record['mutant']).substitutions)
    return keys


# ------------------------------------------------------------------ support stage

def _tuple_counts(cubes: list[tuple]) -> tuple[int, float, Counter]:
    """Site tuples the cubes rest on, their Kish effective count and the histogram."""
    tuples = Counter(tuple(position for position, _ in cube) for cube in cubes)
    return len(tuples), kish_effective_units(tuples.values()), tuples


def run_support(out: Path) -> dict:
    digests = _verify_digests()
    blocks, excluded = read_phot_blocks()
    processed = read_processed_keys(ASSAY)

    block_records = []
    for block, states in sorted(blocks.items(), key=lambda item: -len(item[1])):
        non_wt = {key for key in states if key != WILD_TYPE_STATE}
        block_records.append({
            'block_offset_log10_brightness': block, 'states': len(states),
            'non_wild_type_states': len(non_wt),
            'reconciled_to_processed': len(non_wt & processed),
            'absent_from_processed': len(non_wt - processed),
        })
    primary = max(blocks.items(), key=lambda item: len(item[1]))
    block, states = primary
    assay_states = AssayStates.of(states)

    census = []
    for order in ORDERS:
        cubes = complete_cubes(assay_states, order)
        # Every corner of an admitted cube must also exist in the processed CSV,
        # because the processed CSV is where the model-side sequences come from.
        # A cube whose corner is raw-only would put a floor from one variant set
        # onto a contrast built from another.
        reconciled = [cube for cube in cubes
                      if all(state in processed or state == WILD_TYPE_STATE
                             for state in cube_addresses(cube).values())]
        n_tuples, kish, histogram = _tuple_counts(reconciled)
        census.append({
            'order': order, 'measured_states': assay_states.by_order.get(order, 0),
            'complete_cubes': len(cubes), 'cubes_reconciled_to_processed': len(reconciled),
            'site_tuples': n_tuples, 'effective_site_tuples_kish': kish,
            'max_cubes_on_one_site_tuple': max(histogram.values()) if histogram else 0,
            'corners_per_cube': 2 ** order,
        })

    report = {
        'schema': 'proteingym_higher_order_support/1',
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'sources': digests,
        'assay': ASSAY, 'channels': list(CHANNELS),
        'scale': 'log10 brightness, log10(rep_i)',
        'estimand': ('wild-type-centred order-k cycle in log10 brightness units; the '
                     'raw file carries the wild-type state the processed CSV drops, so '
                     'no per-assay centring is needed and the contrast is invariant to '
                     'an assay-wide additive offset by construction'),
        'excluded_raw_rows': excluded,
        'blocks': block_records,
        'primary_block_offset': block,
        'primary_block_states': len(states),
        'order_census': census,
        'independent_unit': 'site tuple within assay; the assay is the family-level unit',
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True)
    out.write_text(payload, encoding='utf-8')
    print(json.dumps({key: report[key] for key in
                      ('blocks', 'primary_block_states', 'order_census')}, indent=2))
    print(f'declaration digest sha256 {hashlib.sha256(payload.encode()).hexdigest()}')
    return report


# --------------------------------------------------------------- noise-floor stage

def _state_dispersion(states: dict, keys: set[tuple]) -> dict:
    """Per-state, per-channel measurement standard deviation across the channels.

    The pooled within-state variance across the three replicate channels, with the
    ``n - 1`` correction, is an estimate of one channel's measurement variance on
    one state. It is the quantity the propagated floor is built from.
    """
    variances = []
    for key in sorted(keys):
        values = np.array(states[key], dtype=np.float64)
        variances.append(float(values.var(ddof=1)))
    return {'states': len(variances),
            'per_channel_sd': float(np.sqrt(np.mean(variances))),
            'median_per_channel_sd': float(np.sqrt(np.median(variances)))}


def run_noise_floor(declaration: Path, declaration_digest: str, out: Path) -> dict:
    payload = declaration.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != declaration_digest:
        raise ValueError(f'declaration digest {digest} differs from the given '
                         f'{declaration_digest}; the support was not declared before this fit')
    declared = json.loads(payload)
    _verify_digests()
    blocks, _ = read_phot_blocks()
    processed = read_processed_keys(ASSAY)
    states = blocks[declared['primary_block_offset']]
    assay_states = AssayStates.of(states)

    orders = []
    for entry in declared['order_census']:
        order = entry['order']
        cubes = [cube for cube in complete_cubes(assay_states, order)
                 if all(state in processed or state == WILD_TYPE_STATE
                        for state in cube_addresses(cube).values())]
        if len(cubes) != entry['cubes_reconciled_to_processed']:
            raise ValueError(f'order {order} re-derives {len(cubes)} cubes against the '
                             f'declared {entry["cubes_reconciled_to_processed"]}')
        if not cubes:
            continue
        corner_states = {state for cube in cubes
                         for state in cube_addresses(cube).values()}
        dispersion = _state_dispersion(states, corner_states)
        omegas = np.array([
            [cube_contrast(cube, {state: states[state][index]
                                  for state in cube_addresses(cube).values()})
             for index in range(len(CHANNELS))]
            for cube in cubes])
        groups = [tuple(position for position, _ in cube) for cube in cubes]
        pairs = {}
        for left in range(len(CHANNELS)):
            for right in range(left + 1, len(CHANNELS)):
                pairs[f'{CHANNELS[left]}_vs_{CHANNELS[right]}'] = (
                    channel_decomposition_bootstrap(
                        omegas[:, left], omegas[:, right], groups,
                        seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES))
        measured = [record['estimates']['per_channel_discordance_sd']['point']
                    for record in pairs.values()]
        shared = [record['estimates']['shared_component_sd']['point']
                  for record in pairs.values()]
        propagated = math.sqrt(2 ** order) * dispersion['per_channel_sd']
        orders.append({
            'order': order, 'cubes': len(cubes), 'site_tuples': len(set(groups)),
            'effective_site_tuples_kish': kish_effective_units(Counter(groups).values()),
            'corner_states': dispersion['states'],
            'per_state_per_channel_sd_log10': dispersion['per_channel_sd'],
            'propagated_discordance_sd_log10': propagated,
            'mean_measured_discordance_sd_log10': float(np.mean(measured)),
            'mean_shared_component_sd_log10': float(np.mean(shared)),
            'mean_shared_to_discordance_ratio': float(np.mean(shared) / np.mean(measured)),
            'propagation_agreement_measured_over_propagated':
                float(np.mean(measured) / propagated),
            'omega_sd_per_channel_log10': [float(omegas[:, i].std(ddof=1))
                                           for i in range(len(CHANNELS))],
            'channel_pairs': pairs,
        })

    report = {
        'schema': 'proteingym_higher_order_noise_floor/1',
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'declaration': {'path': str(declaration), 'sha256': digest},
        'assay': ASSAY, 'channels': list(CHANNELS),
        'scale': 'log10 brightness',
        'resampling_unit': 'site tuple',
        'orders': orders,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    for entry in orders:
        print(f"order {entry['order']}: cubes {entry['cubes']}, site tuples "
              f"{entry['site_tuples']} (Kish {entry['effective_site_tuples_kish']:.1f}), "
              f"sigma {entry['per_state_per_channel_sd_log10']:.4f}, propagated floor "
              f"{entry['propagated_discordance_sd_log10']:.4f}, measured discordance "
              f"{entry['mean_measured_discordance_sd_log10']:.4f}, shared "
              f"{entry['mean_shared_component_sd_log10']:.4f}, ratio "
              f"{entry['mean_shared_to_discordance_ratio']:.2f}")
    print(f'written {out}')
    return report


# ------------------------------------------------------- count-derived floor stage

#: The four-site combinatorial GB1 library, the assay with the largest cube count
#: in the local panel. Its raw file carries the wild-type combination at Hamming
#: distance zero and per-variant input and selected read counts, but no replicate
#: channel, so the only noise model available for it is multinomial counting noise.
#: That is a *lower bound* on total measurement noise: it captures sequencing and
#: library sampling variance and says nothing about assay-level systematic error,
#: batch structure or the selection's own nonlinearity. An endpoint that merely
#: clears it is not qualified to the standard a replicate channel sets.
COUNT_ASSAY = 'SPG1_STRSG_Wu_2016'
#: A zero selected count has Poisson variance of order one rather than zero, so it
#: is floored at one read. Without this a dead variant would enter with a noise
#: estimate of exactly zero and inflate every ratio it appears in.
MINIMUM_SELECTED_READS = 1.0


def read_count_assay() -> tuple[dict[tuple, float], dict[tuple, float]]:
    """Enrichment ratio and its counting variance per measured state.

    The ratio is ``selected / input``, which is the study's ``Fitness`` up to the
    single wild-type normalising constant; a constant scale factor cancels from
    every ratio reported below. Variance is the delta-method multinomial estimate
    ``r ** 2 * (1 / selected + 1 / input)``.
    """
    path = ROOT / RAW_DIR / f'{COUNT_ASSAY}.csv'
    ratios: dict[tuple, float] = {}
    variances: dict[tuple, float] = {}
    with open(path, newline='', encoding='utf-8') as handle:
        for record in csv.DictReader(handle):
            label = record['mutant']
            state = WILD_TYPE_STATE if not label else parse_mutant(label).substitutions
            if int(record['HD']) != len(state):
                raise ValueError(f'{COUNT_ASSAY}: {label!r} carries HD {record["HD"]} '
                                 f'against {len(state)} substitutions')
            selected = max(float(record['Count selected']), MINIMUM_SELECTED_READS)
            supplied = float(record['Count input'])
            if supplied <= 0:
                raise ValueError(f'{COUNT_ASSAY}: {label!r} has no input reads')
            ratio = selected / supplied
            if state in ratios:
                raise ValueError(f'{COUNT_ASSAY}: repeats state {label!r}')
            ratios[state] = ratio
            variances[state] = ratio ** 2 * (1.0 / selected + 1.0 / supplied)
    if WILD_TYPE_STATE not in ratios:
        raise ValueError(f'{COUNT_ASSAY}: no wild-type state at Hamming distance zero')
    return ratios, variances


def run_count_floor(out: Path) -> dict:
    digests = _verify_digests()
    ratios, variances = read_count_assay()
    assay_states = AssayStates.of(ratios)
    processed = read_processed_keys(COUNT_ASSAY)
    orders = []
    for order in (2, 3, 4):
        cubes = [cube for cube in complete_cubes(assay_states, order)
                 if all(state in processed or state == WILD_TYPE_STATE
                        for state in cube_addresses(cube).values())]
        if not cubes:
            continue
        omega, sigma, groups = [], [], []
        for cube in cubes:
            corners = cube_addresses(cube).values()
            omega.append(cube_contrast(cube, ratios))
            sigma.append(math.sqrt(sum(variances[state] for state in corners)))
            groups.append(tuple(position for position, _ in cube))
        omega_array = np.array(omega)
        sigma_array = np.array(sigma)
        ratio = np.abs(omega_array) / sigma_array
        orders.append({
            'order': order, 'cubes': len(cubes), 'site_tuples': len(set(groups)),
            'effective_site_tuples_kish': kish_effective_units(Counter(groups).values()),
            'omega_sd_enrichment_ratio': float(omega_array.std(ddof=1)),
            'counting_noise_rms_enrichment_ratio': float(np.sqrt((sigma_array ** 2).mean())),
            'omega_sd_over_counting_noise_rms': float(
                omega_array.std(ddof=1) / np.sqrt((sigma_array ** 2).mean())),
            'median_abs_omega_over_counting_sd': float(np.median(ratio)),
            'fraction_abs_omega_above_2_counting_sd': float((ratio > 2).mean()),
            **{f'unit_floor_{key}': value for key, value in
               bootstrap_unit_floor(len(set(groups))).items()},
        })
    report = {
        'schema': 'proteingym_higher_order_count_floor/1',
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'sources': digests, 'assay': COUNT_ASSAY,
        'scale': 'selected/input enrichment ratio, linear',
        'noise_model': ('multinomial counting noise only, a lower bound on total '
                        'measurement noise; no replicate channel exists for this assay'),
        'resampling_unit': 'site tuple',
        'orders': orders,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    for entry in orders:
        print(f"order {entry['order']}: cubes {entry['cubes']}, site tuples "
              f"{entry['site_tuples']}, SD(omega) "
              f"{entry['omega_sd_enrichment_ratio']:.5f}, counting-noise RMS "
              f"{entry['counting_noise_rms_enrichment_ratio']:.5f}, ratio "
              f"{entry['omega_sd_over_counting_noise_rms']:.2f}, median |omega|/sd "
              f"{entry['median_abs_omega_over_counting_sd']:.2f}, above 2 sd "
              f"{entry['fraction_abs_omega_above_2_counting_sd']:.3f}")
    print(f'written {out}')
    return report


# ------------------------------------------------ synonymous-channel stages (HIS3)

#: The HIS3 interspecies-epistasis assay, whose source deposition carries what
#: neither the processed benchmark nor ProteinGym's raw bundle does: one fitness
#: value per *nucleotide* genotype. Synonymous genotypes of one amino-acid state
#: are independent library members grown and sequenced independently, so they are
#: a genuine replicate channel rather than a counting-noise model.
SYNONYMOUS_ASSAY = 'HIS7_YEAST_Pokusaeva_2019'
SYNONYMOUS_FILE = Path('data/his3_pokusaeva_2019/synonymous_variants_rescaled_data.tab.xz')
SYNONYMOUS_SHA256 = 'b362f273d61866b8c09d8a17a1bf3c701738ff67f8efe5a12634b7fc570aa4e7'
#: The assay is twelve overlapping segment libraries. Each segment's window is
#: located in the benchmark's own target sequence by minimising consensus
#: mismatch, and the located windows must tile the reference file's declared
#: mutated region; both are checked rather than assumed.
SYNONYMOUS_REGION = (6, 211)
#: A state needs at least this many synonymous genotypes to carry two channels.
MINIMUM_GENOTYPES = 2
SYNONYMOUS_ORDERS = (2, 3, 4, 5)


def read_synonymous_states() -> tuple[dict[tuple, list[float]], dict]:
    """Per-amino-acid-state synonymous fitness values, keyed on the benchmark's numbering.

    Returns the states and a provenance record. Non-modal-length windows and
    windows naming a residue outside the canonical twenty -- the deposition writes
    a stop or gap as ``_`` -- are excluded and counted, in the same spirit as the
    MegaScale indel exclusion: a truncated chain is not a substitution state.
    """
    import lzma

    digest = sha256_file(ROOT / SYNONYMOUS_FILE)
    if digest != SYNONYMOUS_SHA256:
        raise ValueError(f'{SYNONYMOUS_FILE} digest {digest} differs from the declared '
                         f'{SYNONYMOUS_SHA256}')
    target = None
    with open(ROOT / REFERENCE_FILE, newline='', encoding='utf-8') as handle:
        for record in csv.DictReader(handle):
            if record['DMS_id'] == SYNONYMOUS_ASSAY:
                target = record['target_seq']
    if target is None:
        raise ValueError(f'{SYNONYMOUS_ASSAY} is absent from the reference file')

    segments: dict[str, dict[str, list[tuple[bytes, float]]]] = {}
    with lzma.open(ROOT / SYNONYMOUS_FILE, 'rt') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            segments.setdefault(row['SegN'], {}).setdefault(row['aa_seq'], []).append(
                (genotype_rank(row['seq']), float(row['s'])))

    excluded = Counter()
    states: dict[tuple, list[tuple[bytes, float]]] = {}
    windows = {}
    for segment in sorted(segments, key=int):
        lengths = Counter(len(key) for key in segments[segment])
        width = lengths.most_common(1)[0][0]
        columns = [Counter() for _ in range(width)]
        for key in segments[segment]:
            if len(key) == width:
                for index, residue in enumerate(key):
                    columns[index][residue] += 1
        consensus = ''.join(column.most_common(1)[0][0] for column in columns)
        scored = sorted(
            (sum(a != b for a, b in zip(consensus, target[offset:offset + width])), offset)
            for offset in range(len(target) - width + 1))
        (best, offset), (runner, _) = scored[0], scored[1]
        if best >= runner:
            raise ValueError(f'segment {segment} has no unique placement in the target')
        windows[segment] = {'offset_1based': offset + 1, 'width': width,
                            'consensus_mismatches': best, 'next_best_mismatches': runner}
        window = target[offset:offset + width]
        for key, values in segments[segment].items():
            if len(key) != width:
                excluded['non_modal_length'] += 1
                continue
            if any(residue not in RESIDUES for residue in key):
                excluded['noncanonical_residue'] += 1
                continue
            address = tuple((offset + index, key[index])
                            for index in range(width) if key[index] != window[index])
            states.setdefault(address, []).extend(values)

    covered = {position for segment in windows.values()
               for position in range(segment['offset_1based'],
                                     segment['offset_1based'] + segment['width'])}
    if (min(covered), max(covered)) != SYNONYMOUS_REGION:
        raise ValueError(f'located windows cover {min(covered)}-{max(covered)}, not the '
                         f'reference file\'s declared {SYNONYMOUS_REGION}')
    if WILD_TYPE_STATE not in states:
        raise ValueError('no segment measured its own wild-type window')
    ordered = {key: [value for _, value in sorted(entries)]
               for key, entries in states.items()}
    return ordered, {'source': {'path': str(SYNONYMOUS_FILE), 'sha256': digest},
                     'segments': windows, 'excluded_states': dict(excluded),
                     'mapped_states': len(ordered),
                     'wild_type_genotypes': len(ordered[WILD_TYPE_STATE])}


def run_synonymous(out: Path, declaration: Path | None, declaration_digest: str | None) -> dict:
    """Support and floor in one stage, because the support is not value-independent here.

    A state enters only if it carries at least two synonymous genotypes, which is
    a property of the library rather than of the fitness value, so the admission
    rule still reads no label. The two stages are nonetheless fused: splitting them
    would declare a support that the same pass then measures, and the declaration
    digest would certify nothing it does not already certify. When a declaration is
    supplied it is verified instead.
    """
    if declaration is not None:
        digest = hashlib.sha256(declaration.read_bytes()).hexdigest()
        if digest != declaration_digest:
            raise ValueError(f'declaration digest {digest} differs from the given '
                             f'{declaration_digest}')
    states, provenance = read_synonymous_states()
    processed = read_processed_keys(SYNONYMOUS_ASSAY)
    replicated = {key: values for key, values in states.items()
                  if len(values) >= MINIMUM_GENOTYPES}
    reconciliation = {
        'processed_variants': len(processed),
        'processed_variants_covered': sum(1 for key in processed if key in states),
        'processed_variants_with_two_genotypes': sum(
            1 for key in processed if key in replicated),
        'states_with_two_genotypes': len(replicated),
    }
    if reconciliation['processed_variants_covered'] != len(processed):
        raise ValueError('the source deposition does not cover every processed variant')

    assay_states = AssayStates.of(replicated)
    channels = {key: split_channels(values) for key, values in replicated.items()}
    orders = []
    for order in SYNONYMOUS_ORDERS:
        cubes = [cube for cube in complete_cubes(assay_states, order)
                 if all(state in replicated for state in cube_addresses(cube).values())
                 and all(state in processed or state == WILD_TYPE_STATE
                         for state in cube_addresses(cube).values())]
        if not cubes:
            continue
        left, right, groups, propagated = [], [], [], []
        for cube in cubes:
            corners = list(cube_addresses(cube).items())
            left.append(cycle_contrast({corner: channels[state][0]
                                        for corner, state in corners}))
            right.append(cycle_contrast({corner: channels[state][1]
                                         for corner, state in corners}))
            variance = sum(channels[state][2] / channels[state][3]
                           + channels[state][2] / channels[state][4]
                           for _, state in corners)
            propagated.append(variance / 2.0)
            groups.append(tuple(position for position, _ in cube))
        record = channel_decomposition_bootstrap(
            left, right, groups, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES)
        measured = record['estimates']['per_channel_discordance_sd']
        shared = record['estimates']['shared_component_sd']
        orders.append({
            'order': order, 'cubes': len(cubes), 'site_tuples': len(set(groups)),
            'effective_site_tuples_kish': kish_effective_units(Counter(groups).values()),
            'positions': len({position for group in groups for position in group}),
            'substitutions': len({sub for cube in cubes for sub in cube}),
            'propagated_discordance_sd': float(np.sqrt(np.mean(propagated))),
            'measured_discordance_sd': measured, 'shared_component_sd': shared,
            'shared_to_discordance_ratio': record['estimates'][
                'shared_to_discordance_ratio'],
            'decomposition': record,
        })

    report = {
        'schema': 'his3_synonymous_higher_order/1',
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'assay': SYNONYMOUS_ASSAY,
        'channels': 'two alternating-rank means over synonymous nucleotide genotypes',
        'scale': 'scaled fitness s, 0 at nonsense and 1.0 at wild type',
        'estimand': ('wild-type-centred order-k cycle in scaled fitness units; the '
                     'source deposition measures the wild-type window in all twelve '
                     'segment libraries, so no per-assay centring is needed'),
        'provenance': provenance, 'reconciliation': reconciliation,
        'resampling_unit': 'site tuple',
        'minimum_genotypes_per_state': MINIMUM_GENOTYPES,
        'orders': orders,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True)
    out.write_text(payload, encoding='utf-8')
    for entry in orders:
        print(f"order {entry['order']}: cubes {entry['cubes']}, site tuples "
              f"{entry['site_tuples']} (Kish {entry['effective_site_tuples_kish']:.1f}), "
              f"{entry['positions']} positions, propagated floor "
              f"{entry['propagated_discordance_sd']:.4f}, measured discordance "
              f"{entry['measured_discordance_sd']['point']:.4f} "
              f"{entry['measured_discordance_sd'].get('ci95')}, shared "
              f"{entry['shared_component_sd']['point']:.4f} "
              f"{entry['shared_component_sd'].get('ci95')}, ratio "
              f"{entry['shared_to_discordance_ratio']['point']:.3f} "
              f"{entry['shared_to_discordance_ratio'].get('ci95')}")
    print(f'digest sha256 {hashlib.sha256(payload.encode()).hexdigest()}')
    print(f'written {out}')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    support = sub.add_parser('support', help='declare the complete-cube support')
    support.add_argument('--out', type=Path, required=True)
    floor = sub.add_parser('noise-floor', help='measure the endpoint noise floor')
    floor.add_argument('--declaration', type=Path, required=True)
    floor.add_argument('--declaration-digest', required=True)
    floor.add_argument('--out', type=Path, required=True)
    counts = sub.add_parser('count-floor', help='count-derived lower-bound floor on GB1')
    counts.add_argument('--out', type=Path, required=True)
    syn = sub.add_parser('synonymous', help='HIS3 support and floor on synonymous channels')
    syn.add_argument('--out', type=Path, required=True)
    syn.add_argument('--declaration', type=Path)
    syn.add_argument('--declaration-digest')
    args = parser.parse_args()
    if args.stage == 'support':
        run_support(args.out)
    elif args.stage == 'count-floor':
        run_count_floor(args.out)
    elif args.stage == 'synonymous':
        run_synonymous(args.out, args.declaration, args.declaration_digest)
    else:
        run_noise_floor(args.declaration, args.declaration_digest, args.out)


if __name__ == '__main__':
    main()
