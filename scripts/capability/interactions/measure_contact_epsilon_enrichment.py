#!/usr/bin/env python3
"""Part 1 of the structure-contact gate: does measured nonadditivity concentrate at contacts?

This is the biological-validity check on the admitted label instrument, and it
runs before any model-side comparison. The endpoint is the frozen cohort's
wild-type-centred cycle epsilon on the combined MegaScale ``dG_ML`` scale in
kcal/mol, summarised per site pair with cycles weighted equally, and the contrast
is contact site pairs against non-contact site pairs matched by coarsened exact
matching on sequence separation, wild-type residue class and relative solvent
accessibility.

One declared filter narrows the support without touching the frozen cohort digest.
``pairwise_stability.build_cohort`` keys a state on ``aa_seq`` alone, and an
insertion or deletion construct whose ``aa_seq`` is truncated to the wild type's
length enters the substitution support as though it were a substitution. Those
states are excluded here by their source ``mut_type``, the excluded cycles are
counted per contact stratum, and the primary estimate is reported on both
supports so the filter's effect is readable rather than assumed.

Two quantities bound the reading. The per-channel half-difference of epsilon,
rebuilt here from the pinned trypsin and chymotrypsin columns on this exact
support, is a pure-noise endpoint: the same estimator applied to it measures how
much apparent enrichment the label instrument's own discordance can produce. And
the source study selected these site pairs as coupling tables, so the
``pair_name`` composition of each arm is reported beside every estimate.

The contact calls and the matching weights come from the label-free annotation
and never from the endpoint; this entry point is the first place in the gate
where a measurement is read.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.interactions.contact_enrichment import (  # noqa: E402
    ADDITIVE_RESPONSE_BINS, BALANCE_COVARIATES, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED,
    CB_CONTACT_ANGSTROM, HEAVY_ATOM_CONTACT_ANGSTROM, HYDROPHOBIC_CELLS, RSA_CELL_RULE,
    SEPARATION_CELLS, balance_table, cem_design, fit_binned_response, group_equal_weight,
    kish, two_stage_bootstrap, weighted_difference)
from src.capability.core.io import write_json  # noqa: E402
from src.capability.stability.pairwise_stability import QC_WIDTH_COLUMNS, QC_WIDTH_KCAL_MOL  # noqa: E402

SCHEMA = 'contact_epsilon_enrichment_v1'

#: Contact definitions compared. The primary is the heavy-atom one; the other two
#: are definition sensitivities on the same support and the same matching.
DEFINITIONS = ('heavy_atom', 'cb', 'ensemble_majority')

#: Experimental-method strata. 53 of the 64 backgrounds are solution NMR, where
#: the first deposited model carries no special status and the ensemble spread is
#: conformational uncertainty rather than noise, so the method strata and the
#: two contact definitions are reported against each other rather than pooled
#: behind one arbitrarily indexed model.
METHOD_STRATA = ('pooled', 'SOLUTION NMR', 'X-RAY DIFFRACTION')
STRATUM_DEFINITIONS = ('heavy_atom', 'ensemble_majority')
STRATUM_ENDPOINTS = ('mean_abs_epsilon', 'mean_abs_epsilon_adjusted')

#: Endpoints, all in kcal/mol at the site-pair level. The half channel difference
#: is the pure-noise negative control, not an interaction endpoint.
ENDPOINTS = ('mean_abs_epsilon', 'mean_epsilon', 'mean_abs_epsilon_adjusted',
             'mean_epsilon_adjusted', 'mean_abs_half_channel_difference')

WEIGHTINGS = ('site_pair_equal', 'group_equal')

#: Bins of the measured additive prediction used to remove the global response the
#: label instrument measured as roughly half of epsilon's variance. Fitted in
#: sample over the whole retained support, exactly as that qualification did, and
#: reported as a secondary endpoint rather than as the primary one. The bin count
#: lives with the residualizer so the cross-fit sensitivity uses the same width.
RESPONSE_BINS = ADDITIVE_RESPONSE_BINS

#: Source ``mut_type`` prefixes of the insertion and deletion constructs whose
#: length-truncated ``aa_seq`` collides with the substitution support. A cycle any
#: of whose four states carries such a row is excluded from the declared support;
#: the cohort's own medians are not recomputed, because its digest is the frozen
#: support three gates declare.
INDEL_PREFIXES = ('ins', 'del')

#: Backgrounds the family-grouping step identified as designed mini-proteins that
#: a four-character PDB-name screen mislabelled as natural. They stay in the
#: support carrying that flag, and the primary estimate is repeated without them
#: rather than reclassifying them here.
FLAGGED_DESIGN_BACKGROUNDS = ('5UP5.pdb', '5UYO.pdb')


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b''):
            sha.update(chunk)
    return sha.hexdigest()


def channel_states(parquet_paths: list[Path], sequences: dict[str, set[str]]) -> dict:
    """Per-channel median stability of every cohort state, on the admitted rule.

    The admission rule is the cohort's own: finite numeric ``dG_ML`` and all three
    confidence widths inside [0, 0.5] kcal/mol. Both protease channels are read
    from the same accepted rows, so the channel difference below is a difference
    of two readings of one row set and not of two differently filtered supports.
    """

    import pandas as pd
    import pyarrow.parquet as pq

    wanted = {'WT_name', 'aa_seq', 'mut_type', 'dG_ML', 'deltaG_t', 'deltaG_c', 'pair_name',
              *QC_WIDTH_COLUMNS,
              *(f'{column}_{bound}' for column in QC_WIDTH_COLUMNS
                for bound in ('low', 'high'))}
    frames = []
    for path in parquet_paths:
        frame = pq.read_table(path, columns=sorted(wanted)).to_pandas()
        frame = frame[frame.WT_name.isin(sequences)]
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    accepted = np.isfinite(pd.to_numeric(data.dG_ML, errors='coerce'))
    for column in QC_WIDTH_COLUMNS:
        width = data[f'{column}_high'] - data[f'{column}_low']
        finite = np.isfinite(data[column]) & np.isfinite(width)
        if not np.allclose(data.loc[finite, column], width[finite], rtol=1e-6, atol=1e-8):
            raise ValueError(f'{column} differs from its upper minus lower bound')
        accepted &= finite & data[column].between(0, QC_WIDTH_KCAL_MOL)
    data = data[accepted]
    keep = [name in sequences and sequence in sequences[name]
            for name, sequence in zip(data.WT_name, data.aa_seq)]
    data = data[keep]
    trypsin, chymotrypsin, tables, indel = {}, {}, {}, {}
    for (name, sequence), part in data.groupby(['WT_name', 'aa_seq'], sort=True):
        values = pd.to_numeric(part.deltaG_t, errors='coerce')
        if np.isfinite(values).any():
            trypsin[(name, sequence)] = float(np.median(values[np.isfinite(values)]))
        values = pd.to_numeric(part.deltaG_c, errors='coerce')
        if np.isfinite(values).any():
            chymotrypsin[(name, sequence)] = float(np.median(values[np.isfinite(values)]))
        named = sorted({str(value) for value in part.pair_name.tolist()
                        if isinstance(value, str) and value})
        if named:
            tables[(name, sequence)] = named
        kinds = sorted({str(value) for value in part.mut_type.tolist()
                        if isinstance(value, str) and value.lower().startswith(INDEL_PREFIXES)})
        if kinds:
            indel[(name, sequence)] = kinds
    return {'trypsin': trypsin, 'chymotrypsin': chymotrypsin, 'pair_names': tables,
            'indel_states': indel, 'rows_accepted': int(len(data))}


def support_cycles(cohort: dict, channels: dict, *, exclude_indel: bool) -> dict:
    """Cycles on one indel support, before any additive-response adjustment.

    ``exclude_indel`` drops every cycle any of whose four states carries an
    insertion or deletion source row. The additive column is ``y_A + y_B - y_WT``
    on the cohort's own state medians. No response curve is fit here.
    """

    indel_states = channels['indel_states']
    cycles, excluded = [], []
    for row in cohort['backgrounds']:
        measurements = row['measurements']
        for cycle in row['cycles']:
            wild, low, high, double = cycle['sequences']
            states = [measurements[sequence]['value'] for sequence in (wild, low, high, double)]
            additive = states[1] + states[2] - states[0]
            key = tuple(sorted(cycle['positions']))
            affected = any((row['name'], sequence) in indel_states
                           for sequence in (wild, low, high, double))
            if affected:
                excluded.append({'background': row['name'],
                                 'site_pair': f"{row['name']}:{key[0]}-{key[1]}"})
                if exclude_indel:
                    continue
            entry = {'background': row['name'], 'group': row['group'],
                     'site_pair': f"{row['name']}:{key[0]}-{key[1]}",
                     'epsilon': float(cycle['epsilon']), 'additive': float(additive)}
            half = []
            for channel in ('trypsin', 'chymotrypsin'):
                table = channels[channel]
                keys = [(row['name'], sequence) for sequence in (wild, low, high, double)]
                if all(item in table for item in keys):
                    values = [table[item] for item in keys]
                    half.append(values[3] - values[1] - values[2] + values[0])
            entry['half_channel_difference'] = (
                float((half[0] - half[1]) / 2) if len(half) == 2 else None)
            entry['pair_names'] = channels['pair_names'].get((row['name'], double), [])
            cycles.append(entry)
    return {'cycles': cycles, 'excluded': excluded, 'indel_states': indel_states}


def site_pair_statistics(cohort: dict, channels: dict, *, exclude_indel: bool) -> dict:
    """Per-site-pair endpoint values, cycles weighted equally inside a site pair.

    ``exclude_indel`` drops every cycle any of whose four states carries an
    insertion or deletion source row. The global response is refitted inside the
    retained support, so the adjusted endpoint is not carried over from the wider
    one. That refit is in-sample: every retained cycle enters the curve that is
    then subtracted from it.
    """

    supported = support_cycles(cohort, channels, exclude_indel=exclude_indel)
    cycles = supported['cycles']
    excluded = supported['excluded']
    indel_states = supported['indel_states']
    additive = np.asarray([cycle['additive'] for cycle in cycles])
    epsilon = np.asarray([cycle['epsilon'] for cycle in cycles])
    curve = fit_binned_response(additive, epsilon, RESPONSE_BINS)
    response = curve['response']
    for cycle, index in zip(cycles, curve['bins']):
        cycle['epsilon_adjusted'] = cycle['epsilon'] - float(response[index])
    statistics: dict[str, dict] = {}
    for cycle in cycles:
        entry = statistics.setdefault(cycle['site_pair'], {
            'background': cycle['background'], 'group': cycle['group'],
            'cycles': 0, 'epsilon': [], 'epsilon_adjusted': [], 'half': [],
            'pair_names': set()})
        entry['cycles'] += 1
        entry['epsilon'].append(cycle['epsilon'])
        entry['epsilon_adjusted'].append(cycle['epsilon_adjusted'])
        if cycle['half_channel_difference'] is not None:
            entry['half'].append(cycle['half_channel_difference'])
        entry['pair_names'].update(cycle['pair_names'])
    out = {}
    for pair, entry in statistics.items():
        values = np.asarray(entry['epsilon'])
        adjusted = np.asarray(entry['epsilon_adjusted'])
        half = np.asarray(entry['half']) if entry['half'] else None
        out[pair] = {
            'background': entry['background'], 'group': entry['group'],
            'cycles': entry['cycles'], 'channel_cycles': 0 if half is None else len(half),
            'mean_abs_epsilon': float(np.abs(values).mean()),
            'mean_epsilon': float(values.mean()),
            'mean_abs_epsilon_adjusted': float(np.abs(adjusted).mean()),
            'mean_epsilon_adjusted': float(adjusted.mean()),
            'mean_abs_half_channel_difference': None if half is None else float(
                np.abs(half).mean()),
            'mean_half_channel_difference': None if half is None else float(half.mean()),
            'pair_names': sorted(entry['pair_names']),
        }
    floor = {'excluded_indel_cycles': len(excluded),
             'excluded_indel_states': len(indel_states),
             'excluded_indel_backgrounds': sorted({row['background'] for row in excluded}),
             'excluded_indel_site_pairs': sorted({row['site_pair'] for row in excluded}),
             'indel_exclusion_applied': exclude_indel,
             'response_range_kcal_mol': float(response.max() - response.min()),
             'response_bins': RESPONSE_BINS,
             'cycles_with_both_channels': int(sum(
                 1 for cycle in cycles if cycle['half_channel_difference'] is not None)),
             'cycles': len(cycles)}
    resolved = np.asarray([cycle['half_channel_difference'] for cycle in cycles
                           if cycle['half_channel_difference'] is not None])
    if len(resolved):
        floor['per_cycle_discordance_sd_kcal_mol'] = float(resolved.std(ddof=1))
        floor['per_cycle_mean_absolute_kcal_mol'] = float(np.abs(resolved).mean())
    return {'site_pairs': out, 'floor': floor}


def treated_mask(rows: list[dict], definition: str) -> np.ndarray:
    if definition == 'heavy_atom':
        return np.asarray([row['contact'] for row in rows], dtype=bool)
    if definition == 'cb':
        return np.asarray([row['contact_cb'] for row in rows], dtype=bool)
    if definition == 'ensemble_majority':
        return np.asarray([row['contact_model_fraction'] >= 0.5 for row in rows], dtype=bool)
    raise ValueError(f'unknown contact definition {definition!r}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--parquet', type=Path, nargs='+', required=True)
    parser.add_argument('--parquet-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()

    annotation = json.loads(args.annotation.read_text())
    if annotation.get('schema') != 'contact_annotation_v1':
        raise SystemExit('unexpected annotation schema')
    cohort_sha = digest(args.cohort)
    if cohort_sha != annotation['inputs']['cohort']['sha256']:
        raise SystemExit('cohort digest does not match the annotation it is read with')
    manifest = json.loads(args.parquet_manifest.read_text())
    declared = {Path(entry['path']).name: entry['sha256']
                for entry in manifest['verification']['files']}
    parquet_digests = {}
    for path in args.parquet:
        value = digest(path)
        if declared.get(path.name) != value:
            raise SystemExit(f'{path.name} does not match the staged corpus manifest digest')
        parquet_digests[path.name] = value

    cohort = json.loads(args.cohort.read_text())
    sequences = {row['name']: set(row['sequences']) for row in cohort['backgrounds']}
    channels = channel_states(list(args.parquet), sequences)

    annotated = [dict(pair, method=record['structure']['method'])
                 for record in annotation['backgrounds'] for pair in record['site_pairs']]
    boundary = float(np.median([np.mean(row['rsa']) for row in annotated]))
    supports = {}
    for support, exclude in (('indel_excluded', True), ('all_cycles', False)):
        measured = site_pair_statistics(cohort, channels, exclude_indel=exclude)
        known = measured['site_pairs']
        unexpected = [pair for pair in known if pair not in {row['site_pair']
                                                             for row in annotated}]
        if unexpected:
            raise SystemExit(f'measured site pairs absent from the annotation: {unexpected[:5]}')
        rows = [row for row in annotated if row['site_pair'] in known]
        supports[support] = evaluate_support(rows, known, boundary, measured, annotated,
                                             draws=args.bootstrap)
        supports[support]['structure_positive_control'] = structure_positive_control(
            cohort, annotated, channels, exclude_indel=exclude, boundary=boundary,
            draws=args.bootstrap)

    report = build_report(args, annotation, cohort_sha, parquet_digests, channels,
                          annotated, boundary, supports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(json.dumps({
        'support': {name: entry['support'] for name, entry in supports.items()},
        'floor': supports['indel_excluded']['label_instrument_floor'],
        'positive_control': supports['indel_excluded']['structure_positive_control'],
        'primary': {endpoint: supports['indel_excluded']['definitions']['heavy_atom'][
            'endpoints'][endpoint]['site_pair_equal'] for endpoint in ENDPOINTS}}, indent=1))


def evaluate_support(rows: list[dict], known: dict, boundary: float, measured: dict,
                     annotated: list[dict], *, draws: int) -> dict:
    """The full definition x endpoint x weighting grid on one cycle support."""

    results = {}
    for definition in DEFINITIONS:
        treated = treated_mask(rows, definition)
        design = cem_design(rows, treated, rsa_boundary=boundary)
        entry = {
            'contacts': int(treated.sum()), 'non_contacts': int((~treated).sum()),
            'pruned_ineligible_separation': design['pruned_ineligible_separation'],
            'pruned_contacts_without_control': design['pruned_contacts_without_control'],
            'pruned_controls_without_contact': design['pruned_controls_without_contact'],
            'retained_cells': [list(cell) for cell in design['retained_cells']],
            'cell_occupancy': design['cell_occupancy'],
            'balance': balance_table(rows, design['weight'], treated,
                                     np.asarray([cell is not None for cell in design['cells']])),
            'endpoints': {},
        }
        for endpoint in ENDPOINTS:
            available = np.asarray([known[row['site_pair']][endpoint] is not None
                                    for row in rows])
            values = np.asarray([known[row['site_pair']][endpoint] or 0.0 for row in rows])
            subset = [row for row, ok in zip(rows, available) if ok]
            subtreated = treated[available]
            for weighting in WEIGHTINGS:
                weight = cem_design(subset, subtreated, rsa_boundary=boundary)['weight']
                if weighting == 'group_equal':
                    weight = group_equal_weight(
                        weight, np.asarray([row['group'] for row in subset]), subtreated)
                point = weighted_difference(values[available], weight, subtreated)
                interval = two_stage_bootstrap(
                    subset, values[available], subtreated, rsa_boundary=boundary,
                    group_equal=(weighting == 'group_equal'),
                    draws=draws, seed=BOOTSTRAP_SEED)
                groups = np.asarray([row['group'] for row in subset])
                entry['endpoints'].setdefault(endpoint, {})[weighting] = {
                    'unit': 'kcal/mol',
                    'contact_mean': point['contact'], 'control_mean': point['control'],
                    'difference': point['difference'], 'interval': interval['interval'],
                    'excludes_zero': interval['excludes_zero'],
                    'bootstrap_draws': interval['draws'],
                    'skipped_draws': interval['skipped_draws'],
                    'contact_site_pairs': point['contact_site_pairs'],
                    'control_site_pairs': point['control_site_pairs'],
                    'effective_contact_site_pairs': point['effective_contact'],
                    'effective_control_site_pairs': point['effective_control'],
                    'contact_groups': len({group for group, ok in zip(
                        groups, subtreated & (weight > 0)) if ok}),
                    'control_groups': len({group for group, ok in zip(
                        groups, ~subtreated & (weight > 0)) if ok}),
                    'cycles_contact': int(sum(known[row['site_pair']]['cycles']
                                              for row, ok in zip(subset, subtreated & (weight > 0))
                                              if ok)),
                    'cycles_control': int(sum(known[row['site_pair']]['cycles']
                                              for row, ok in zip(subset, ~subtreated & (weight > 0))
                                              if ok)),
                    'site_pairs_dropped_for_missing_endpoint': int((~available).sum()),
                }
        results[definition] = entry

    primary = treated_mask(rows, 'heavy_atom')
    tables: dict[str, dict] = {}
    for row, contact in zip(rows, primary):
        for name in known[row['site_pair']]['pair_names'] or ['(none)']:
            family = ('hbond_network' if '_hnet' in name.lower()
                      else 'dmutv5' if 'dmut' in name.lower() else 'other')
            entry = tables.setdefault(family, {'contact': 0, 'non_contact': 0})
            entry['contact' if contact else 'non_contact'] += 1
    flagged = {}
    present = sorted({row['background'] for row in rows
                      if row['background'] in FLAGGED_DESIGN_BACKGROUNDS})
    if present:
        kept = [row for row in rows if row['background'] not in present]
        treated = treated_mask(kept, 'heavy_atom')
        for endpoint in ('mean_abs_epsilon', 'mean_abs_epsilon_adjusted'):
            values = np.asarray([known[row['site_pair']][endpoint] for row in kept])
            weight = cem_design(kept, treated, rsa_boundary=boundary)['weight']
            point = weighted_difference(values, weight, treated)
            record = two_stage_bootstrap(kept, values, treated, rsa_boundary=boundary,
                                         draws=draws, seed=BOOTSTRAP_SEED)
            flagged[endpoint] = {
                'unit': 'kcal/mol', 'difference': point['difference'],
                'interval': record['interval'], 'excludes_zero': record['excludes_zero'],
                'contact_site_pairs': point['contact_site_pairs'],
                'control_site_pairs': point['control_site_pairs']}
    strata = {}
    for stratum in METHOD_STRATA:
        inside = [row for row in rows
                  if stratum == 'pooled' or row['method'] == stratum]
        first = treated_mask(inside, 'heavy_atom')
        majority = treated_mask(inside, 'ensemble_majority')
        entry = {
            'site_pairs': len(inside),
            'groups': len({row['group'] for row in inside}),
            'contacts_first_model': int(first.sum()),
            'contacts_majority_of_models': int(majority.sum()),
            'site_pairs_changing_status': int((first != majority).sum()),
            'contact_only_in_first_model': int((first & ~majority).sum()),
            'contact_only_in_the_majority': int((~first & majority).sum()),
            'model_fraction_of_first_model_contacts_median': (
                float(np.median([row['contact_model_fraction']
                                 for row, ok in zip(inside, first) if ok])) if first.any()
                else None),
            'estimates': {},
        }
        for definition in STRATUM_DEFINITIONS:
            treated = treated_mask(inside, definition)
            design = cem_design(inside, treated, rsa_boundary=boundary)
            for endpoint in STRATUM_ENDPOINTS:
                values = np.asarray([known[row['site_pair']][endpoint] for row in inside])
                point = weighted_difference(values, design['weight'], treated)
                record = two_stage_bootstrap(inside, values, treated,
                                             rsa_boundary=boundary, draws=draws,
                                             seed=BOOTSTRAP_SEED)
                entry['estimates'].setdefault(definition, {})[endpoint] = {
                    'unit': 'kcal/mol', 'difference': point['difference'],
                    'contact_mean': point['contact'], 'control_mean': point['control'],
                    'interval': record['interval'],
                    'excludes_zero': record['excludes_zero'],
                    'degenerate': record.get('degenerate'),
                    'degenerate_reason': record.get('degenerate_reason'),
                    'contact_site_pairs': point['contact_site_pairs'],
                    'control_site_pairs': point['control_site_pairs'],
                    'effective_control_site_pairs': point['effective_control'],
                    'bootstrap_groups': record.get('n_units')}
        strata[stratum] = entry
    dropped = [row['site_pair'] for row in annotated if row['site_pair'] not in known]
    excluded_by_stratum = {'contact': 0, 'non_contact': 0}
    contact_of = {row['site_pair']: bool(row['contact']) for row in annotated}
    for pair in measured['floor']['excluded_indel_site_pairs']:
        excluded_by_stratum['contact' if contact_of[pair] else 'non_contact'] += 1
    return {
        'support': {
            'site_pairs': len(rows), 'groups': len({row['group'] for row in rows}),
            'cycles': sum(known[row['site_pair']]['cycles'] for row in rows),
            'site_pairs_absent_from_this_support': dropped,
            'indel_excluded_site_pairs_by_stratum': excluded_by_stratum,
        },
        'label_instrument_floor': measured['floor'],
        'source_table_composition': tables,
        'method_strata': strata,
        'flagged_design_backgrounds_removed': {
            'backgrounds': present, 'primary_definition': 'heavy_atom',
            'weighting': 'site_pair_equal', 'estimates': flagged},
        'definitions': results,
    }


def structure_positive_control(cohort: dict, annotated: list[dict], channels: dict,
                               *, exclude_indel: bool, boundary: float,
                               draws: int) -> dict:
    """Does burial predict single-mutant destabilization on this exact support?

    This qualifies the structure instrument itself by a route independent of the
    contact contrast. Relative solvent accessibility at a site-pair position is a
    structural quantity read through the same alignment and the same chain as the
    contact calls, and the mean measured single-substitution ddG at that position
    is a label the annotation never saw. A positive association is evidence that
    the numbering, the chain choice and the accessibility calculation carry real
    information about this assay; its absence would mean a flat contact result
    could not be attributed to the endpoint rather than to the structures.
    """

    from scipy.stats import rankdata

    from src.capability.context.profile_increment import correlation
    from src.capability.context.profiles import cluster_bootstrap

    indel_states = channels['indel_states']
    accessibility = {}
    for row in annotated:
        for position, value in zip(row['positions'], row['rsa']):
            accessibility[(row['background'], position)] = value
    effects: dict[tuple, dict[str, float]] = {}
    for row in cohort['backgrounds']:
        measurements = row['measurements']
        for cycle in row['cycles']:
            wild, low_state, high_state, double = cycle['sequences']
            if exclude_indel and any((row['name'], sequence) in indel_states
                                     for sequence in cycle['sequences']):
                continue
            reference = measurements[wild]['value']
            for position, state in zip(sorted(cycle['positions']), (low_state, high_state)):
                key = (row['name'], position)
                if key not in accessibility:
                    continue
                effects.setdefault(key, {})[state] = (
                    measurements[state]['value'] - reference)
    positions = []
    for (background, position), values in sorted(effects.items()):
        series = np.asarray(list(values.values()))
        positions.append({'background': background, 'position': position,
                          'singles': len(series),
                          'rsa': accessibility[(background, position)],
                          'mean_ddg': float(series.mean()),
                          'mean_abs_ddg': float(np.abs(series).mean())})
    spearman, labels = [], []
    for background in sorted({row['background'] for row in positions}):
        rows = [row for row in positions if row['background'] == background]
        if len(rows) < 4 or len({row['rsa'] for row in rows}) < 2:
            continue
        spearman.append(correlation(
            rankdata([row['rsa'] for row in rows]),
            rankdata([row['mean_ddg'] for row in rows])))
        labels.append(background)
    buried = [row['mean_ddg'] for row in positions if row['rsa'] <= boundary]
    exposed = [row['mean_ddg'] for row in positions if row['rsa'] > boundary]
    return {
        'positions': len(positions),
        'backgrounds_with_at_least_four_positions': len(labels),
        'singles_per_position_median': float(np.median([row['singles']
                                                        for row in positions])),
        'within_background_spearman_rsa_vs_mean_ddg': (
            cluster_bootstrap(spearman, list(range(len(spearman))),
                              resamples=draws, seed=BOOTSTRAP_SEED)
            if len(spearman) >= 8 else {'point': None, 'interval': None,
                                        'degenerate': True, 'n_units': len(spearman)}),
        'mean_ddg_buried_kcal_mol': float(np.mean(buried)) if buried else None,
        'mean_ddg_exposed_kcal_mol': float(np.mean(exposed)) if exposed else None,
        'buried_positions': len(buried), 'exposed_positions': len(exposed),
        'rsa_boundary': boundary,
        'sign_convention': ('positive dG_ML is more stable, so a destabilizing '
                            'substitution has negative ddG and a positive rank '
                            'correlation means buried positions are more destabilized'),
    }


def build_report(args, annotation: dict, cohort_sha: str, parquet_digests: dict,
                 channels: dict, annotated: list[dict], boundary: float,
                 supports: dict) -> dict:
    return {
        'schema': SCHEMA,
        'inputs': {'annotation': {'path': str(args.annotation), 'sha256': digest(args.annotation)},
                   'cohort_sha256': cohort_sha, 'parquet_sha256': parquet_digests},
        'declaration': {
            'contact_primary': f'minimum heavy-atom distance <= {HEAVY_ATOM_CONTACT_ANGSTROM} A',
            'contact_secondary': f'CB-CB distance <= {CB_CONTACT_ANGSTROM} A',
            'contact_sensitivity': 'contact in at least half of the deposited models',
            'separation_cells': [list(cell) for cell in SEPARATION_CELLS],
            'hydrophobic_cells': [list(cell) for cell in HYDROPHOBIC_CELLS],
            'rsa_cell_rule': RSA_CELL_RULE, 'rsa_boundary': boundary,
            'balance_covariates': list(BALANCE_COVARIATES),
            'weighting_primary': 'site pairs equal inside the contact arm, controls cell-matched',
            'weighting_sensitivity': 'groups equal inside each arm',
            'bootstrap': {'draws': args.bootstrap, 'seed': BOOTSTRAP_SEED,
                          'unit': 'group, then site pair inside the drawn group'},
            'endpoint_unit': 'kcal/mol per site pair, cycles weighted equally inside a site pair',
        },
        'annotated_site_pairs': len(annotated),
        'annotated_groups': len({row['group'] for row in annotated}),
        'kish_site_pairs_unmatched': kish(np.ones(len(annotated))),
        'channel_rows_accepted': channels['rows_accepted'],
        'primary_support': 'indel_excluded',
        'supports': supports,
        'code_sha256': {
            'scripts/capability/interactions/measure_contact_epsilon_enrichment.py': digest(Path(__file__)),
            'src/capability/interactions/contact_enrichment.py': digest(ROOT / 'src/capability/interactions/contact_enrichment.py'),
        },
    }


if __name__ == '__main__':
    main()
