#!/usr/bin/env python3
"""Measurement-only qualification of the MegaScale double-mutant label instrument.

Builds the WT-centered stability cycle epsilon = y_AB - y_A - y_B + y_WT
separately from the trypsin and the chymotrypsin channel of the pinned
Tsuboyama 2023 dataset2 bytes, then reports between-channel agreement,
disagreement magnitude in kcal/mol, support counts, effective group count,
construct/codon replicate spread and a source-labelled positive control.

No model score, likelihood or representation enters any quantity computed here,
so the resulting admission decision cannot be tuned to a model outcome. The two
proteases are two assay channels over shared sequences and a shared stability
inference, not independent ground-truth instruments; nothing here adds four
independent error variances to manufacture an oracle ceiling.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.io import sha256_file, write_json
from src.capability.stability.pairwise_stability import QC_SOURCE, QC_WIDTH_KCAL_MOL, validate_cycle

AA20 = 'ACDEFGHIKLMNPQRSTVWY'
CHANNELS = {'trypsin': 'deltaG_t', 'chymotrypsin': 'deltaG_c'}
CHANNEL_FIT_BOUND_KCAL_MOL = 15.0
CENSORED_DG_ML = ('<-1', '>5', '-')
SEPARATION_STRATA = (('1-2', 1, 2), ('3-9', 3, 9), ('10+', 10, 1 << 30))
PAIR_TABLE_MUT_TYPE = re.compile(r'^([A-Z])(\d+)([A-Z]):([A-Z])(\d+)([A-Z])$')
MIN_CYCLES_FOR_SPEARMAN = 20
COLUMNS = [
    'name', 'dna_seq', 'mut_type', 'WT_name', 'WT_cluster', 'aa_seq', 'dG_ML', 'ddG_ML', 'pair_name',
    'deltaG', 'deltaG_95CI', 'deltaG_95CI_low', 'deltaG_95CI_high',
    'deltaG_t', 'deltaG_t_95CI', 'deltaG_t_95CI_low', 'deltaG_t_95CI_high',
    'deltaG_c', 'deltaG_c_95CI', 'deltaG_c_95CI_low', 'deltaG_c_95CI_high',
]
VARIANTS = (
    ('admission_as_proposed', False, 'median'),
    ('both_channel_width', True, 'median'),
    ('both_channel_width_single_row', True, 'first'),
)


# ---------------------------------------------------------------- pinned input

def verify_pinned_bytes(data_dir: Path, manifest_path: Path) -> dict:
    """Bind the analysis to the pinned dataset2 bytes; mismatch is fatal."""
    manifest = json.loads(manifest_path.read_bytes())
    expected = {Path(f['path']).name: f['sha256'] for f in manifest['verification']['files']}
    measured = {p.name: sha256_file(p) for p in sorted(data_dir.glob('*.parquet'))}
    if set(measured) != set(expected):
        raise ValueError(f'staged parquet set {sorted(measured)} differs from manifest {sorted(expected)}')
    wrong = {n: measured[n] for n in measured if measured[n] != expected[n]}
    if wrong:
        raise ValueError(f'staged bytes differ from the pinned manifest digests: {wrong}')
    return {
        'corpus_id': manifest['corpus_id'],
        'config': manifest['config'],
        'revision': manifest['canonical_source']['revision'],
        'files': measured,
    }


def load_frame(data_dir: Path) -> pd.DataFrame:
    parts = sorted(data_dir.glob('*.parquet'))
    if not parts:
        raise ValueError(f'no parquet files under {data_dir}')
    frame = pd.concat([pd.read_parquet(p, columns=COLUMNS) for p in parts], ignore_index=True)
    for prefix in ('deltaG', 'deltaG_t', 'deltaG_c'):
        width = frame[f'{prefix}_95CI']
        span = frame[f'{prefix}_95CI_high'] - frame[f'{prefix}_95CI_low']
        finite = np.isfinite(width) & np.isfinite(span)
        if not np.allclose(width[finite], span[finite], rtol=1e-6, atol=1e-8):
            raise ValueError(f'{prefix}_95CI differs from its high minus low bound')
    return frame


def load_catalogue(path: Path, frame: pd.DataFrame) -> dict:
    """Source-derived natural/design labels; WT sequences must match the wt rows."""
    rows = json.loads(path.read_bytes())
    wt = frame[frame.mut_type == 'wt'].groupby('WT_name').aa_seq.agg(['nunique', 'first'])
    if int((wt['nunique'] > 1).sum()):
        raise ValueError('a background carries more than one distinct wt sequence')
    clusters = frame.groupby('WT_name').WT_cluster.agg(['nunique', 'first'])
    if int((clusters['nunique'] > 1).sum()):
        raise ValueError('a background carries more than one source WT_cluster')
    catalogue = {}
    for row in rows:
        name = row['WT_name']
        if name not in wt.index:
            continue
        if row['sequence'] != wt.loc[name, 'first']:
            raise ValueError(f'catalogue sequence for {name} differs from its wt row')
        if any(a not in AA20 for a in row['sequence']):
            raise ValueError(f'catalogue sequence for {name} leaves the canonical alphabet')
        catalogue[name] = {
            'kind': row['kind'],
            'sequence': row['sequence'],
            'cluster': str(clusters.loc[name, 'first']),
        }
    return catalogue


# ------------------------------------------------------------ accepted support

def accept(frame: pd.DataFrame, *, per_channel_width: bool) -> tuple[np.ndarray, dict]:
    """Identical acceptance rule applied to the rows that feed both channels."""
    numeric = pd.to_numeric(frame.dG_ML, errors='coerce')
    finite_numeric = np.isfinite(numeric)
    deviation = float(np.abs(numeric[finite_numeric] - frame.deltaG[finite_numeric]).max())
    if deviation > 1e-6:
        raise ValueError(f'numeric dG_ML departs from deltaG by {deviation} kcal/mol')
    combined = np.isfinite(frame.deltaG_95CI) & frame.deltaG_95CI.between(0, QC_WIDTH_KCAL_MOL)
    mask = (finite_numeric & combined).to_numpy()
    accounting = {
        'rows_total': int(len(frame)),
        'rows_censored_dG_ML': {v: int((frame.dG_ML == v).sum()) for v in CENSORED_DG_ML},
        'rows_finite_numeric_dG_ML': int(finite_numeric.sum()),
        'rows_after_combined_width_rule': int(mask.sum()),
    }
    if per_channel_width:
        for column in CHANNELS.values():
            mask &= frame[f'{column}_95CI'].between(0, QC_WIDTH_KCAL_MOL).to_numpy()
        accounting['rows_after_both_channel_width_rule'] = int(mask.sum())
        accounting['rows_excluded_by_both_channel_width_rule'] = (
            accounting['rows_after_combined_width_rule'] - int(mask.sum()))
    accounting['accepted_rows'] = int(mask.sum())
    accounting['accepted_rows_at_channel_fit_bound'] = {
        channel: int((frame.loc[mask, column].abs() == CHANNEL_FIT_BOUND_KCAL_MOL).sum())
        for channel, column in CHANNELS.items()}
    accounting['accepted_channel_range_kcal_mol'] = {
        channel: [float(frame.loc[mask, column].min()), float(frame.loc[mask, column].max())]
        for channel, column in CHANNELS.items()}
    return mask, accounting


def aggregate(frame: pd.DataFrame, mask: np.ndarray, *, statistic: str) -> pd.DataFrame:
    """One value per exact (WT_name, aa_seq) per channel, same rows for both."""
    if statistic not in ('median', 'first'):
        raise ValueError('statistic must be median or first')
    rows = frame.loc[mask, ['WT_name', 'aa_seq', 'name', 'dna_seq', *CHANNELS.values()]]
    grouped = rows.groupby(['WT_name', 'aa_seq'], sort=True)
    values = grouped[list(CHANNELS.values())].median() if statistic == 'median' else \
        grouped[list(CHANNELS.values())].first()
    values['n_rows'] = grouped.size()
    values['n_constructs'] = grouped['name'].nunique()
    values['n_dna'] = grouped['dna_seq'].nunique()
    return values.reset_index()


def replicate_spread(frame: pd.DataFrame, mask: np.ndarray) -> dict:
    """Spread across repeated observations of one exact sequence, per channel.

    Repeated rows are distinct codon constructs of the same amino-acid sequence
    read in the same assay, not independent biological replicates.
    """
    rows = frame.loc[mask, ['WT_name', 'aa_seq', 'name', 'dna_seq', *CHANNELS.values()]]
    grouped = rows.groupby(['WT_name', 'aa_seq'], sort=False)
    counts = grouped.size()
    report = {
        'aggregated_sequences': int(len(counts)),
        'rows_per_sequence_histogram': {str(k): int(v) for k, v in
                                        counts.value_counts().sort_index().items()},
        'sequences_with_repeat_rows': int((counts > 1).sum()),
        'sequences_with_distinct_codon_constructs': int((grouped['dna_seq'].nunique() > 1).sum()),
        'sequences_with_distinct_construct_names': int((grouped['name'].nunique() > 1).sum()),
    }
    repeats = rows[rows.set_index(['WT_name', 'aa_seq']).index.isin(counts[counts > 1].index)]
    spread = {}
    for channel, column in CHANNELS.items():
        parts = repeats.groupby(['WT_name', 'aa_seq'], sort=False)[column]
        deviation = (repeats[column] - parts.transform('median')).abs()
        spread[channel] = {
            'median_abs_deviation_from_sequence_median_kcal_mol': float(deviation.median()),
            'p95_abs_deviation_from_sequence_median_kcal_mol': float(np.percentile(deviation, 95)),
            'max_abs_deviation_from_sequence_median_kcal_mol': float(deviation.max()),
            'rms_within_sequence_range_kcal_mol': float(np.sqrt(np.mean((parts.max() - parts.min()) ** 2))),
        }
    report['repeat_row_spread'] = spread
    return report


# ------------------------------------------------------------------- cycle set

def background_states(values: pd.DataFrame, wildtype: str) -> dict | None:
    """Exact WT/single/double states of one background from accepted sequences."""
    length = len(wildtype)
    usable = values[values.aa_seq.str.len() == length]
    sequences = usable.aa_seq.to_numpy()
    if wildtype not in set(sequences):
        return None
    letters = np.frombuffer(''.join(sequences).encode('ascii'), np.uint8).reshape(len(sequences), length)
    reference = np.frombuffer(wildtype.encode('ascii'), np.uint8)
    differs = letters != reference
    counts = differs.sum(1)
    y = {channel: usable[column].to_numpy(float) for channel, column in CHANNELS.items()}
    wt_index = int(np.flatnonzero(sequences == wildtype)[0])
    singles: dict[tuple[int, str], int] = {}
    for index in np.flatnonzero(counts == 1):
        position = int(np.argmax(differs[index]))
        singles[(position, sequences[index][position])] = int(index)
    doubles = []
    for index in np.flatnonzero(counts == 2):
        first, second = (int(p) for p in np.flatnonzero(differs[index]))
        doubles.append((first, sequences[index][first], second, sequences[index][second], int(index)))
    return {'sequences': sequences, 'y': y, 'wt_index': wt_index, 'singles': singles,
            'doubles': doubles, 'wildtype': wildtype,
            'n_rows': usable['n_rows'].to_numpy(int), 'n_dna': usable['n_dna'].to_numpy(int)}


def build_cycles(values: pd.DataFrame, catalogue: dict) -> tuple[pd.DataFrame, dict, dict]:
    """Complete four-state cycles per background, identical support per channel."""
    frames, states_by_background, accounting = [], {}, {
        'backgrounds_in_catalogue': len(catalogue), 'backgrounds_without_accepted_wt': 0,
        'backgrounds_with_cycles': 0, 'validated_cycle_samples': 0}
    for name, part in values.groupby('WT_name', sort=True):
        entry = catalogue.get(name)
        if entry is None:
            continue
        states = background_states(part, entry['sequence'])
        if states is None:
            accounting['backgrounds_without_accepted_wt'] += 1
            continue
        rows, wildtype = [], states['wildtype']
        for first, letter_a, second, letter_b, index in states['doubles']:
            left = states['singles'].get((first, letter_a))
            right = states['singles'].get((second, letter_b))
            if left is None or right is None:
                continue
            rows.append((first + 1, second + 1, index, left, right))
        if not rows:
            continue
        sites = np.array([[r[0], r[1]] for r in rows], dtype=np.int32)
        double_ix = np.array([r[2] for r in rows]); left_ix = np.array([r[3] for r in rows])
        right_ix = np.array([r[4] for r in rows]); wt_ix = states['wt_index']
        record = {'WT_name': name, 'kind': entry['kind'], 'cluster': entry['cluster'],
                  'site_i': sites[:, 0], 'site_j': sites[:, 1],
                  'separation': sites[:, 1] - sites[:, 0]}
        for channel in CHANNELS:
            y = states['y'][channel]
            record[f'additive_{channel}'] = y[left_ix] + y[right_ix] - y[wt_ix]
            record[channel] = y[double_ix] - record[f'additive_{channel}']
        record['double_repeat_rows'] = states['n_rows'][double_ix]
        record['cycle_repeat_rows'] = (states['n_rows'][double_ix] + states['n_rows'][left_ix]
                                       + states['n_rows'][right_ix] + states['n_rows'][wt_ix])
        frames.append(pd.DataFrame(record))
        states_by_background[name] = states
        accounting['backgrounds_with_cycles'] += 1
        for first, letter_a, second, letter_b, index in states['doubles'][:2]:
            if (first, letter_a) in states['singles'] and (second, letter_b) in states['singles']:
                single_a = wildtype[:first] + letter_a + wildtype[first + 1:]
                single_b = wildtype[:second] + letter_b + wildtype[second + 1:]
                validate_cycle(wildtype, single_a, single_b, states['sequences'][index])
                accounting['validated_cycle_samples'] += 1
    if not frames:
        raise ValueError('no complete four-state cycle survived the accepted support')
    cycles = pd.concat(frames, ignore_index=True)
    return cycles, states_by_background, accounting


# --------------------------------------------------------- agreement estimates

MOMENTS = 7


def _moments(t: np.ndarray, c: np.ndarray) -> np.ndarray:
    d = t - c
    return np.array([t.sum(), c.sum(), (t * t).sum(), (c * c).sum(), (t * c).sum(),
                     (d * d).sum(), np.abs(d).sum()]) / t.size


def _derive(v: np.ndarray) -> dict:
    et, ec, ett, ecc, etc, ed2, eabsd = v
    var_t = max(float(ett - et * et), 0.0)
    var_c = max(float(ecc - ec * ec), 0.0)
    cov = float(etc - et * ec)
    bias = float(et - ec)
    var_d = max(float(ed2 - bias * bias), 0.0)
    shared = float(np.sqrt(max(cov, 0.0)))
    discordance = float(np.sqrt(var_d / 2.0))
    return {
        'mean_epsilon_trypsin_kcal_mol': float(et),
        'mean_epsilon_chymotrypsin_kcal_mol': float(ec),
        'sd_epsilon_trypsin_kcal_mol': float(np.sqrt(var_t)),
        'sd_epsilon_chymotrypsin_kcal_mol': float(np.sqrt(var_c)),
        'covariance_kcal2_mol2': cov,
        'pearson_r': float(cov / np.sqrt(var_t * var_c)) if var_t > 0 and var_c > 0 else None,
        'shared_component_sd_kcal_mol': shared,
        'channel_difference_mean_kcal_mol': bias,
        'channel_difference_sd_kcal_mol': float(np.sqrt(var_d)),
        'channel_difference_rms_kcal_mol': float(np.sqrt(ed2)),
        'channel_difference_mean_abs_kcal_mol': float(eabsd),
        'per_channel_discordance_sd_kcal_mol': discordance,
        'shared_to_discordance_ratio': float(shared / discordance) if discordance > 0 else None,
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return float(values[min(int(np.searchsorted(cumulative, q, side='left')), values.size - 1)])


def _units(cycles: pd.DataFrame, unit: str) -> tuple[list[np.ndarray], list[int], dict]:
    """Moment vectors for the resampling unit, preserving WT/single reuse.

    Every cycle of a background shares that background's WT and single
    measurements, so resampling whole backgrounds (or whole source clusters of
    backgrounds) keeps each shared measurement's signed contribution intact.
    Cycle-level resampling would break exactly that reuse.
    """
    per_background, counts, spearman = {}, {}, {}
    for name, part in cycles.groupby('WT_name', sort=True):
        t = part.trypsin.to_numpy(float)
        c = part.chymotrypsin.to_numpy(float)
        per_background[name] = _moments(t, c)
        counts[name] = int(t.size)
        if t.size >= MIN_CYCLES_FOR_SPEARMAN and t.std() > 0 and c.std() > 0:
            rt, rc = rankdata(t), rankdata(c)
            spearman[name] = float(np.corrcoef(rt, rc)[0, 1])
    cluster_of = cycles.groupby('WT_name', sort=True).cluster.first().to_dict()
    if unit == 'background':
        keys = sorted(per_background)
        vectors = [per_background[k] for k in keys]
        sizes = [counts[k] for k in keys]
        ranks = [spearman[k] for k in keys if k in spearman]
    elif unit == 'source_cluster':
        members: dict[str, list[str]] = {}
        for name in sorted(per_background):
            members.setdefault(cluster_of[name], []).append(name)
        keys = sorted(members)
        vectors = [np.mean([per_background[n] for n in members[k]], axis=0) for k in keys]
        sizes = [sum(counts[n] for n in members[k]) for k in keys]
        ranks = [float(np.mean([spearman[n] for n in members[k] if n in spearman]))
                 for k in keys if any(n in spearman for n in members[k])]
    else:
        raise ValueError('unit must be background or source_cluster')
    sizes_array = np.array(sizes, float)
    support = {
        'unit': unit,
        'units': len(keys),
        'effective_units_kish': float(sizes_array.sum() ** 2 / (sizes_array ** 2).sum()),
        'units_with_within_unit_spearman': len(ranks),
        'min_cycles_for_within_unit_spearman': MIN_CYCLES_FOR_SPEARMAN,
    }
    return vectors, ranks, support


def _bootstrap(vectors: list[np.ndarray], ranks: list[float], bootstrap: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    stack = np.array(vectors)
    draws = rng.integers(0, len(stack), size=(bootstrap, len(stack)))
    samples = [_derive(stack[row].mean(axis=0)) for row in draws]
    intervals = {}
    for key in samples[0]:
        column = np.array([s[key] for s in samples if s[key] is not None], float)
        column = column[np.isfinite(column)]
        intervals[key] = {'ci95': [float(np.percentile(column, 2.5)), float(np.percentile(column, 97.5))],
                          'finite_draws': int(column.size)} if column.size else {'ci95': None, 'finite_draws': 0}
    point = _derive(stack.mean(axis=0))
    report = {key: {'point': point[key], **intervals[key]} for key in point}
    if ranks:
        array = np.array(ranks, float)
        resampled = array[rng.integers(0, array.size, size=(bootstrap, array.size))].mean(axis=1)
        report['within_unit_spearman'] = {
            'point': float(array.mean()),
            'ci95': [float(np.percentile(resampled, 2.5)), float(np.percentile(resampled, 97.5))],
            'finite_draws': int(resampled.size)}
    return report


def summarise(cycles: pd.DataFrame, *, bootstrap: int, seed: int) -> dict:
    """Per-stratum support, between-channel agreement and disagreement scale."""
    out = {}
    for label, low, high in (('all', 1, 1 << 30),) + SEPARATION_STRATA:
        part = cycles[(cycles.separation >= low) & (cycles.separation <= high)]
        entry = {
            'separation_residues': [low, None if high > 1 << 20 else high],
            'cycles': int(len(part)),
            'backgrounds': int(part.WT_name.nunique()),
            'source_clusters': int(part.cluster.nunique()),
            'site_pairs': int(len(part.groupby(['WT_name', 'site_i', 'site_j']))),
        }
        if entry['cycles'] >= MIN_CYCLES_FOR_SPEARMAN:
            cycles_per_background = part.groupby('WT_name')['trypsin'].transform('size')
            backgrounds_per_cluster = part.groupby('cluster')['WT_name'].transform('nunique')
            clusters = float(part.cluster.nunique())
            weight = (1.0 / (clusters * backgrounds_per_cluster * cycles_per_background)).to_numpy(float)
            difference = (part.trypsin - part.chymotrypsin).abs().to_numpy(float)
            entry['group_equal_weighted_quantiles_kcal_mol'] = {
                'median_abs_channel_difference': _weighted_quantile(difference, weight, 0.5),
                'p95_abs_channel_difference': _weighted_quantile(difference, weight, 0.95),
                'median_abs_epsilon_trypsin': _weighted_quantile(
                    part.trypsin.abs().to_numpy(float), weight, 0.5),
                'median_abs_epsilon_chymotrypsin': _weighted_quantile(
                    part.chymotrypsin.abs().to_numpy(float), weight, 0.5),
            }
            for unit in ('source_cluster', 'background'):
                vectors, ranks, support = _units(part, unit)
                entry[unit] = {**support, 'agreement': _bootstrap(vectors, ranks, bootstrap, seed)}
        out[label] = entry
    return out


# ------------------------------------------------------------- positive control

def documented_pairs(frame: pd.DataFrame, catalogue: dict) -> dict:
    """Site pairs the source study named and scanned as coupling pair tables."""
    rows = frame[frame.pair_name.notna()]
    parsed = rows.mut_type.str.extract(PAIR_TABLE_MUT_TYPE)
    if int(parsed[0].isna().sum()):
        raise ValueError('a pair-table mut_type does not parse as two substitutions')
    series = rows.pair_name.str.extract(r'\.pdb_([A-Za-z0-9]+)')[0]
    table: dict[tuple[str, int, int], str] = {}
    for name, first_wt, first_pos, second_wt, second_pos, kind in zip(
            rows.WT_name, parsed[0], parsed[1].astype(int), parsed[3], parsed[4].astype(int), series):
        entry = catalogue.get(name)
        if entry is None:
            continue
        sequence = entry['sequence']
        if sequence[first_pos - 1] != first_wt or sequence[second_pos - 1] != second_wt:
            raise ValueError(f'pair-table numbering for {name} does not match its wt sequence')
        i, j = sorted((int(first_pos), int(second_pos)))
        table[(name, i, j)] = 'hnet' if str(kind).startswith('hnet') else str(kind)
    return table


def additive_surface_residual(states: dict, i: int, j: int, channel: str) -> dict | None:
    """Least-squares additive fit over one pair table, our declared approximation.

    The source study fits a Bayesian additive model per pair table with its
    prediction clipped to [-1, 5] kcal/mol and reports the residual as a
    thermodynamic coupling. This unclipped least-squares fit on the accepted
    states of the same table is a different estimand from both that residual
    and from the WT-centered cycle; the returned quantities are only used to
    measure how far the estimands sit apart. Double states whose two single
    states are not both accepted carry no WT-centered cycle and are therefore
    excluded from the fit as well, so both estimands share one support.
    """
    y = states['y'][channel]
    wildtype = states['wildtype']
    letters_i, letters_j, response, is_double = [], [], [], []
    letters_i.append(wildtype[i - 1]); letters_j.append(wildtype[j - 1])
    response.append(y[states['wt_index']]); is_double.append(False)
    for (position, letter), index in states['singles'].items():
        if position == i - 1:
            letters_i.append(letter); letters_j.append(wildtype[j - 1])
        elif position == j - 1:
            letters_i.append(wildtype[i - 1]); letters_j.append(letter)
        else:
            continue
        response.append(y[index]); is_double.append(False)
    doubles = []
    for first, letter_a, second, letter_b, index in states['doubles']:
        if (first + 1, second + 1) != (i, j):
            continue
        if (first, letter_a) not in states['singles'] or (second, letter_b) not in states['singles']:
            continue
        letters_i.append(letter_a); letters_j.append(letter_b)
        response.append(y[index]); is_double.append(True)
        doubles.append((letter_a, letter_b, index))
    if len(doubles) < 4:
        return None
    level_i = sorted(set(letters_i)); level_j = sorted(set(letters_j))
    design = np.zeros((len(response), 1 + len(level_i) - 1 + len(level_j) - 1))
    design[:, 0] = 1.0
    for row, (a, b) in enumerate(zip(letters_i, letters_j)):
        if a != level_i[0]:
            design[row, 1 + level_i.index(a) - 1] = 1.0
        if b != level_j[0]:
            design[row, len(level_i) + level_j.index(b) - 1] = 1.0
    response = np.array(response, float)
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ coefficients
    mask = np.array(is_double)
    cycle = np.array([y[index] - y[states['singles'][(i - 1, a)]]
                      - y[states['singles'][(j - 1, b)]] + y[states['wt_index']]
                      for a, b, index in doubles], float)
    return {'residual': residual[mask], 'cycle': cycle, 'states': int(len(response))}


def pair_table_estimands(states_by_background: dict, pairs: dict, catalogue: dict) -> pd.DataFrame:
    """Both estimands of every documented pair table, on one shared support.

    The refitted additive surface of a pair table absorbs a table-level additive
    baseline that WT-centering leaves in the cycle, so the two columns answer
    different questions about the same measured double states.
    """
    parts = []
    for (name, i, j) in sorted(pairs):
        states = states_by_background.get(name)
        if states is None:
            continue
        fits = {channel: additive_surface_residual(states, i, j, channel) for channel in CHANNELS}
        if any(fit is None for fit in fits.values()):
            continue
        sizes = {fit['cycle'].size for fit in fits.values()}
        if len(sizes) != 1:
            raise ValueError(f'channel supports differ for pair table {name} {i} {j}')
        size = sizes.pop()
        part = {'WT_name': [name] * size, 'cluster': [catalogue[name]['cluster']] * size,
                'kind': [catalogue[name]['kind']] * size,
                'site_i': np.full(size, i), 'site_j': np.full(size, j),
                'separation': np.full(size, j - i),
                'fitted_states': np.full(size, fits['trypsin']['states'])}
        for channel in CHANNELS:
            part[channel] = fits[channel]['residual']
            part[f'cycle_{channel}'] = fits[channel]['cycle']
        parts.append(pd.DataFrame(part))
    if not parts:
        raise ValueError('no documented pair table reached the minimum fitted support')
    return pd.concat(parts, ignore_index=True)


def global_response_adjusted(cycles: pd.DataFrame, *, bins: int = 20) -> tuple[pd.DataFrame, dict]:
    """Subtract a binned global response to the additive prediction, per channel.

    This is the measurement-side counterpart of the declared nonlinear-additive
    nuisance class: a single monotone-free response curve in the additive
    prediction, estimated in sample over the whole panel. Fitting it in sample
    removes more than a training-only calibration could, and the additive
    prediction shares measurement error with the cycle within a channel, so the
    reported absorbed fraction is an upper bound. Between-channel agreement of
    the adjusted cycle is not inflated by that shared error, because the error
    it regresses on is channel specific.
    """
    adjusted = cycles.copy()
    report = {'bins': bins, 'channels': {}}
    for channel in CHANNELS:
        prediction = cycles[f'additive_{channel}'].to_numpy(float)
        epsilon = cycles[channel].to_numpy(float)
        edges = np.quantile(prediction, np.linspace(0.0, 1.0, bins + 1))
        index = np.clip(np.searchsorted(edges, prediction, side='right') - 1, 0, bins - 1)
        response = np.zeros(bins)
        for b in range(bins):
            selected = index == b
            if selected.any():
                response[b] = epsilon[selected].mean()
        residual = epsilon - response[index]
        adjusted[channel] = residual
        report['channels'][channel] = {
            'pearson_r_epsilon_against_additive_prediction': float(
                np.corrcoef(epsilon, prediction)[0, 1]),
            'sd_epsilon_kcal_mol': float(epsilon.std(ddof=1)),
            'sd_adjusted_epsilon_kcal_mol': float(residual.std(ddof=1)),
            'absorbed_variance_fraction': float(1.0 - residual.var() / epsilon.var()),
            'response_range_kcal_mol': [float(response.min()), float(response.max())],
        }
    return adjusted, report


def positive_control(cycles: pd.DataFrame, states_by_background: dict, pairs: dict,
                     catalogue: dict, frame: pd.DataFrame, mask: np.ndarray,
                     *, bootstrap: int, seed: int) -> dict:
    """Source-labelled coupling pair tables; no example chosen by any model result."""
    series = pd.Series([pairs.get((str(name), int(i), int(j))) for name, i, j in
                        zip(cycles.WT_name, cycles.site_i, cycles.site_j)], index=cycles.index)
    report = {
        'documented_pair_tables_in_source': len(pairs),
        'selection_rule': 'source pair_name column and its mut_type site numbering; no model quantity',
        'strata': {},
    }
    for label, subset in (('documented_hnet', series == 'hnet'),
                          ('documented_dmutv5', series == 'dmutv5'),
                          ('documented_any', series.notna()),
                          ('not_documented', series.isna())):
        part = cycles[subset.to_numpy()]
        entry = {'cycles': int(len(part)), 'backgrounds': int(part.WT_name.nunique()),
                 'site_pairs': int(len(part.groupby(['WT_name', 'site_i', 'site_j']))) if len(part) else 0,
                 'median_separation_residues': float(part.separation.median()) if len(part) else None}
        if len(part) >= MIN_CYCLES_FOR_SPEARMAN:
            vectors, ranks, support = _units(part, 'source_cluster')
            entry.update({'source_cluster': support,
                          'agreement': _bootstrap(vectors, ranks, bootstrap, seed)})
            for channel in CHANNELS:
                values = part[channel].to_numpy(float)
                entry[f'{channel}_epsilon'] = {
                    'mean_kcal_mol': float(values.mean()),
                    'median_kcal_mol': float(np.median(values)),
                    'sd_kcal_mol': float(values.std(ddof=1)),
                    'fraction_above_0p5_kcal_mol': float(np.mean(np.abs(values) > 0.5)),
                    'fraction_positive': float(np.mean(values > 0)),
                }
        report['strata'][label] = entry
    tables = pair_table_estimands(states_by_background, pairs, catalogue)
    tables = tables[tables.kind == 'natural']
    estimands = {}
    for channel in CHANNELS:
        residual = tables[channel].to_numpy(float)
        cycle = tables[f'cycle_{channel}'].to_numpy(float)
        estimands[channel] = {
            'sd_additive_surface_residual_kcal_mol': float(residual.std(ddof=1)),
            'sd_wt_centered_cycle_kcal_mol': float(cycle.std(ddof=1)),
            'mean_additive_surface_residual_kcal_mol': float(residual.mean()),
            'mean_wt_centered_cycle_kcal_mol': float(cycle.mean()),
            'pearson_r': float(np.corrcoef(residual, cycle)[0, 1]),
            'spearman_rho': float(np.corrcoef(rankdata(residual), rankdata(cycle))[0, 1]),
            'rms_difference_kcal_mol': float(np.sqrt(np.mean((residual - cycle) ** 2))),
            'median_abs_difference_kcal_mol': float(np.median(np.abs(residual - cycle))),
        }
    report['estimand_separation'] = {
        'note': 'the source pair-table additive residual and the WT-centered cycle are distinct '
                'estimands and are not equated numerically',
        'pair_tables_fitted': int(len(tables.groupby(['WT_name', 'site_i', 'site_j']))),
        'double_states': int(len(tables)),
        'fitted_states': int(tables.groupby(['WT_name', 'site_i', 'site_j']).fitted_states.first().sum()),
        'channels': estimands,
    }
    report['pair_specific_residual_agreement'] = summarise(
        tables, bootstrap=bootstrap, seed=seed)
    report['sign_and_scale'] = sign_and_scale(frame, mask)
    return report


def sign_and_scale(frame: pd.DataFrame, mask: np.ndarray) -> dict:
    """Confirm the source ddG convention our cycle sign inherits."""
    rows = frame.loc[mask]
    singles = rows[rows.mut_type.str.fullmatch(r'[A-Z]\d+[A-Z]', na=False)]
    reported = pd.to_numeric(singles.ddG_ML, errors='coerce')
    wildtype = rows[rows.mut_type == 'wt'].groupby('WT_name').deltaG.median()
    usable = singles.assign(reported=reported).dropna(subset=['reported'])
    usable = usable[usable.WT_name.isin(wildtype.index)]
    difference = usable.deltaG.to_numpy(float) - wildtype.loc[usable.WT_name].to_numpy(float)
    reported = usable.reported.to_numpy(float)
    return {
        'single_mutant_rows': int(len(usable)),
        'mean_reported_ddG_ML_kcal_mol': float(reported.mean()),
        'mean_measured_mutant_minus_wt_kcal_mol': float(difference.mean()),
        'rms_residual_against_mutant_minus_wt_kcal_mol': float(np.sqrt(np.mean((reported - difference) ** 2))),
        'rms_residual_against_wt_minus_mutant_kcal_mol': float(np.sqrt(np.mean((reported + difference) ** 2))),
        'convention': 'ddG_ML = deltaG(mutant) - deltaG(WT); positive deltaG is more stable, so a '
                      'favourable WT interaction lost by either single substitution gives epsilon > 0',
    }


# ------------------------------------------------------------------------ main

def run(args: argparse.Namespace) -> dict:
    provenance = verify_pinned_bytes(args.data_dir, args.manifest)
    frame = load_frame(args.data_dir)
    catalogue = load_catalogue(args.catalogue, frame)
    pairs = documented_pairs(frame, catalogue)
    report = {
        'schema_version': 'd1_pairwise_label_instrument_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'seed': args.seed, 'bootstrap': args.bootstrap,
        'units': 'stability and all epsilon quantities in kcal/mol; covariance in kcal^2/mol^2; '
                 'correlations dimensionless; separations in residues',
        'qc_width_kcal_mol': QC_WIDTH_KCAL_MOL, 'qc_source': QC_SOURCE,
        'provenance': provenance,
        'inputs': {str(args.manifest): sha256_file(args.manifest),
                   str(args.catalogue): sha256_file(args.catalogue)},
        'model_outputs_used': False,
        'variants': {},
    }
    backgrounds: dict[str, dict] = {}
    for label, per_channel_width, statistic in VARIANTS:
        mask, accounting = accept(frame, per_channel_width=per_channel_width)
        values = aggregate(frame, mask, statistic=statistic)
        cycles, states, cycle_accounting = build_cycles(values, catalogue)
        kinds = {k: cycles[cycles.kind == k] for k in sorted(set(cycles.kind))}
        entry = {
            'acceptance': {'per_channel_width_rule': per_channel_width,
                           'aggregation': statistic, **accounting},
            'cycle_construction': cycle_accounting,
            'replicates': replicate_spread(frame, mask),
            'kinds': {},
        }
        for kind, part in kinds.items():
            entry['kinds'][kind] = summarise(part, bootstrap=args.bootstrap, seed=args.seed)
        if label == 'both_channel_width':
            natural = kinds.get('natural')
            if natural is None or natural.empty:
                raise ValueError('no natural-labelled cycle survived; the qualification has no support')
            entry['positive_control'] = positive_control(
                natural, states, pairs, catalogue, frame, mask,
                bootstrap=args.bootstrap, seed=args.seed)
            adjusted, response = global_response_adjusted(natural)
            entry['global_response'] = response
            entry['global_response_adjusted_agreement'] = summarise(
                adjusted, bootstrap=args.bootstrap, seed=args.seed)
            backgrounds = {
                name: {'cluster': part.cluster.iloc[0], 'kind': part.kind.iloc[0],
                       'cycles': int(len(part)),
                       'site_pairs': int(len(part.groupby(['site_i', 'site_j']))),
                       'sd_epsilon_trypsin_kcal_mol': (
                           float(part.trypsin.std(ddof=1)) if len(part) > 1 else None),
                       'sd_epsilon_chymotrypsin_kcal_mol': (
                           float(part.chymotrypsin.std(ddof=1)) if len(part) > 1 else None),
                       'rms_channel_difference_kcal_mol': float(
                           np.sqrt(np.mean((part.trypsin - part.chymotrypsin) ** 2))),
                       'cycles_by_separation': {
                           name_: int(((part.separation >= low) & (part.separation <= high)).sum())
                           for name_, low, high in SEPARATION_STRATA}}
                for name, part in cycles.groupby('WT_name', sort=True)}
        report['variants'][label] = entry
        del cycles, states, values
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'qualification.json', report)
    write_json(args.out / 'backgrounds.json', backgrounds)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path,
                        default=ROOT / 'data/megascale_tsuboyama2023/dataset2/data')
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'data/megascale_tsuboyama2023/manifest.json')
    parser.add_argument('--catalogue', type=Path,
                        default=ROOT / 'results/shared/megascale_disjointness_20260813/query_index.json')
    parser.add_argument('--out', type=Path, required=True, help='output directory under ignored logs/')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260923)
    args = parser.parse_args()
    report = run(args)
    for label, entry in report['variants'].items():
        natural = entry['kinds'].get('natural', {}).get('all', {})
        agreement = natural.get('source_cluster', {}).get('agreement', {})
        pearson = agreement.get('pearson_r', {})
        print(f"{label}: accepted rows {entry['acceptance']['accepted_rows']}, "
              f"natural cycles {natural.get('cycles')}, "
              f"source clusters {natural.get('source_clusters')}, "
              f"site pairs {natural.get('site_pairs')}, "
              f"pearson r {pearson.get('point')} {pearson.get('ci95')}")
        for key in ('global_response_adjusted_agreement',):
            block = entry.get(key, {}).get('all', {}).get('source_cluster', {})
            if block:
                r = block['agreement']['pearson_r']
                print(f"  {key}: pearson r {r['point']} {r['ci95']}")
        residual = entry.get('positive_control', {}).get('pair_specific_residual_agreement', {})
        block = residual.get('all', {}).get('source_cluster', {})
        if block:
            r = block['agreement']['pearson_r']
            print(f"  pair_specific_residual_agreement: pearson r {r['point']} {r['ci95']}")


if __name__ == '__main__':
    main()
