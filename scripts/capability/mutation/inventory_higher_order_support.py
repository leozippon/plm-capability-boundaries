#!/usr/bin/env python3
"""Availability inventory and measurement-noise floor for the higher-order gate.

Two stages, run in that order, because the endpoint has to be chosen on support
and power rather than on a label outcome.

``inventory`` reads only the *keys* of the admitted measured states: the
substitution-order census of the pinned MegaScale dataset2 bytes, and the
mutation pairs whose complete pairwise cycle is measured in two related
backgrounds. It computes no epsilon and no omega, so the support it reports
cannot be a function of the values that support will carry.

``noise-floor`` re-derives that same support, refuses to proceed unless the
inventory digest it is given still matches, and then measures the third-order
cycle omega separately in the trypsin and the chymotrypsin channel. It reports
the between-channel decomposition the label instrument declared, and it
propagates that instrument's pairwise floor to order three: the four reference
background corners and the four variant background corners are disjoint sets of
measurements, so an order-3 channel discordance variance is the sum of the two
constituent order-2 variances and its standard deviation is sqrt(2) times larger.

No model score, likelihood or representation enters any quantity here.
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

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.mutation.higher_order_cycle import (  # noqa: E402
    BackgroundContrast, cycle_contrast, enumerate_contrasts, kish_effective_units)
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.stability.pairwise_stability import QC_WIDTH_KCAL_MOL, separation_stratum  # noqa: E402
# The admitted label instrument owns the pinned-byte check, the tightened
# per-channel admission rule, the median state aggregation and the between-channel
# decomposition. They are imported rather than restated so that this gate cannot
# drift from the instrument it is judged against.
from scripts.capability.stability.qualify_megascale_label_instrument import (  # noqa: E402
    CHANNELS, accept, aggregate, load_catalogue, load_frame, verify_pinned_bytes,
    _bootstrap, _derive, _moments)

#: A length-changing construct's ``aa_seq`` is truncated to the wild type's
#: length, so an insertion near the N terminus reads as a run of substitutions
#: and an insertion at position 1 or 2 reads as one or two of them. Counting
#: ``aa_seq`` differences alone therefore admits 1,486 length-matched insertion
#: and deletion states into the substitution support, 29 of them at order 1 or 2
#: and 23 more pooled by median with a genuine substitution state. The
#: ``mut_type`` vocabulary is exactly ``wt``, colon-joined uppercase
#: substitutions, ``ins<residue><position>`` and ``del<residue><position>``, so
#: this lowercase prefix identifies every length-changing row without a
#: value-dependent filter.
INDEL_MUT_TYPE = re.compile(r'^(ins|del)')
#: Background pair relatedness cuts the inventory tabulates. One is the exact
#: third-order cube; the wider cuts show how fast the support grows and at what
#: cost in interpretability, because a background pair differing at 20 of 60
#: positions is not a background contrast of one mutation pair.
DISTANCE_CUTS = (1, 2, 3, 5, 10)
#: Below this many independent units a percentile group bootstrap is not an
#: interval: with n units the probability that a draw is a single unit repeated is
#: n ** (1 - n), which is 0.5 at n = 2 and 4.8e-7 at n = 8, so the reported
#: endpoints would be individual unit means rather than a sampling interval.
MIN_UNITS_FOR_INTERVAL = 8
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260923
GROUPING = 'data/pairwise_assets/megascale_family_groups_20260924.json'
CATALOGUE = 'results/shared/megascale_disjointness_20260813/query_index.json'
#: The admitted label instrument's retained qualification, whose order-2
#: per-channel discordance is the only channel-noise estimate on this endpoint
#: that carries a usable interval: 92 source clusters and 25.1 Kish effective
#: clusters, against the 2 site pairs the exact third-order support offers.
INSTRUMENT = 'archive/logs/R3/pairwise_label_instrument_20260923/qualification.json'
INSTRUMENT_SHA256 = 'e33d614a93f6f47f15f2c093925928b3b43bac9e5e6a90b439a8a603742c3e07'
#: Order-3 channel discordance variance is the sum of the two constituent order-2
#: variances, because the four reference-background corners and the four
#: variant-background corners are disjoint measurements.
ORDER_THREE_PROPAGATION = 2.0 ** 0.5


# ----------------------------------------------------------------- admitted keys

def admitted_states(data_dir: Path, manifest: Path, catalogue_path: Path) -> tuple[dict, dict, dict]:
    """Wild types, per-background state keys and per-state channel values.

    The keys drive the inventory; the values are returned but are read only by the
    noise-floor stage. The admitted support is the label instrument's tightened
    per-channel rule restricted to substitution-labelled rows, because a
    length-changing construct's truncated ``aa_seq`` is not the substitution state
    its residue differences would make it look like.
    """
    provenance = verify_pinned_bytes(data_dir, manifest)
    frame = load_frame(data_dir)
    catalogue = load_catalogue(catalogue_path, frame)
    mask, accounting = accept(frame, per_channel_width=True)
    indel = frame.mut_type.astype(str).str.match(INDEL_MUT_TYPE).to_numpy()
    accounting['indel_rows_excluded'] = int((mask & indel).sum())
    mask = mask & ~indel
    accounting['accepted_substitution_rows'] = int(mask.sum())
    values = aggregate(frame, mask, statistic='median')
    order_census = substitution_order_census(frame, mask, catalogue)
    wildtypes, states, measured = {}, {}, {}
    for name, part in values.groupby('WT_name', sort=True):
        entry = catalogue.get(name)
        if entry is None:
            continue
        wildtype = entry['sequence']
        keys = part.aa_seq.to_numpy()
        if wildtype not in set(keys):
            continue
        by_key: dict[tuple, dict] = {}
        for sequence, trypsin, chymotrypsin in zip(
                keys, part[CHANNELS['trypsin']].to_numpy(float),
                part[CHANNELS['chymotrypsin']].to_numpy(float)):
            if len(sequence) != len(wildtype):
                continue
            substitutions = tuple((i, sequence[i]) for i in range(len(wildtype))
                                  if sequence[i] != wildtype[i])
            if len(substitutions) > 2:
                continue
            by_key[substitutions] = {'trypsin': float(trypsin),
                                     'chymotrypsin': float(chymotrypsin)}
        wildtypes[name] = wildtype
        states[name] = set(by_key)
        measured[name] = by_key
    return ({'provenance': provenance, 'admission': accounting,
             'substitution_order_census': order_census,
             'backgrounds_with_admitted_wild_type': len(wildtypes)},
            {'wildtypes': wildtypes, 'states': states, 'measured': measured},
            {'catalogue': catalogue, 'frame': frame})


def indel_state_contamination(frame: pd.DataFrame, catalogue: dict,
                              cohort_path: Path | None) -> dict:
    """What the ``mut_type`` exclusion removes, and what it would have admitted.

    Reported because the pairwise cohort builder keys states on ``aa_seq`` alone
    and therefore does admit these rows. The frozen-cohort counts are the evidence
    for that; this gate excludes them and reports the exclusion rather than
    carrying the conflation forward.
    """
    wildtype = {name: entry['sequence'] for name, entry in catalogue.items()}
    mask, _ = accept(frame, per_channel_width=True)
    rows = frame.loc[mask, ['WT_name', 'aa_seq', 'mut_type']].copy()
    rows['indel'] = rows.mut_type.astype(str).str.match(INDEL_MUT_TYPE)
    rows = rows[[name in wildtype and len(sequence) == len(wildtype[name])
                 for name, sequence in zip(rows.WT_name, rows.aa_seq)]]
    tally = rows.groupby(['WT_name', 'aa_seq']).indel.agg(['sum', 'count'])
    carried = tally[tally['sum'] > 0]
    orders = [sum(1 for i in range(len(wildtype[name])) if sequence[i] != wildtype[name][i])
              for name, sequence in carried.index]
    out = {
        'length_matched_admitted_states': int(len(tally)),
        'states_carried_by_an_indel_row': int(len(carried)),
        'states_carried_only_by_indel_rows': int((carried['sum'] == carried['count']).sum()),
        'states_pooling_an_indel_row_with_substitution_rows':
            int((carried['sum'] < carried['count']).sum()),
        'states_at_cycle_eligible_order': {
            f'order_{order}': sum(1 for value in orders if value == order) for order in (0, 1, 2)},
        'frozen_pairwise_cohort': None,
    }
    if cohort_path is not None and cohort_path.exists():
        cohort = json.loads(cohort_path.read_bytes())
        tainted = set(carried.index)
        sequences = [(background['name'], sequence) for background in cohort['backgrounds']
                     for sequence in background['sequences']]
        cycles = [(background['name'], cycle) for background in cohort['backgrounds']
                  for cycle in background['cycles']]
        out['frozen_pairwise_cohort'] = {
            'path': str(Path(cohort_path).resolve().relative_to(ROOT)),
            'sha256': sha256_file(cohort_path),
            'sequences': len(sequences),
            'sequences_carried_by_an_indel_row': sum(1 for key in sequences if key in tainted),
            'cycles': len(cycles),
            'cycles_touching_such_a_state': sum(
                1 for name, cycle in cycles
                if any((name, sequence) in tainted for sequence in cycle['sequences'])),
            'backgrounds_affected': sorted({name for name, sequence in sequences
                                            if (name, sequence) in tainted}),
        }
    return out


def substitution_order_census(frame: pd.DataFrame, mask: np.ndarray, catalogue: dict) -> dict:
    """How many substitutions each variant carries relative to its own wild type.

    This is the triple-mutant endpoint's availability: an order-3 cycle inside one
    background needs a variant with three substitutions and all seven of its
    lower-order states. ``mask`` already excludes length-changing constructs, so a
    nonzero order-3 count here would be a genuine triple substitution.
    """
    wildtype = {name: entry['sequence'] for name, entry in catalogue.items()}
    out = {}
    for label, rows in (('all_rows', frame), ('admitted_rows', frame[mask])):
        tally: dict[str, int] = {}
        for name, part in rows.groupby('WT_name', sort=True):
            reference = wildtype.get(name)
            if reference is None:
                tally['background_outside_catalogue'] = (
                    tally.get('background_outside_catalogue', 0) + len(part))
                continue
            length = len(reference)
            same = part.aa_seq.str.len() == length
            for sequence in part.aa_seq[~same]:
                key = 'length_change_insertion' if len(sequence) > length else 'length_change_deletion'
                tally[key] = tally.get(key, 0) + 1
            for sequence in part.aa_seq[same]:
                order = sum(1 for i in range(length) if sequence[i] != reference[i])
                key = f'substitutions_{order}' if order < 3 else 'substitutions_3_or_more'
                tally[key] = tally.get(key, 0) + 1
        out[label] = dict(sorted(tally.items()))
    return out


# -------------------------------------------------------------------- inventory

def tabulate(contrasts: list[BackgroundContrast], groups: dict, catalogue: dict) -> dict:
    """Support counts at one relatedness cut, with the site pair as the unit."""
    backgrounds = sorted({name for c in contrasts for name in (c.reference, c.variant)})
    per_site_pair: dict[tuple[int, int], int] = {}
    for contrast in contrasts:
        per_site_pair[contrast.site_pair] = per_site_pair.get(contrast.site_pair, 0) + 1
    strata: dict[str, int] = {}
    for contrast in contrasts:
        first, second = contrast.site_pair
        key = separation_stratum(first + 1, second + 1)
        strata[key] = strata.get(key, 0) + 1
    return {
        'contrasts': len(contrasts),
        'distinct_mutation_pairs': len({(c.first, c.second) for c in contrasts}),
        'distinct_site_pairs': len(per_site_pair),
        'background_site_pair_cells': len(
            {(name, c.site_pair) for c in contrasts for name in (c.reference, c.variant)}),
        'background_pairs': len({(c.reference, c.variant) for c in contrasts}),
        'backgrounds': len(backgrounds),
        'family_groups': len({groups[n] for n in backgrounds if n in groups}),
        'kinds': {kind: sum(1 for n in backgrounds if catalogue[n]['kind'] == kind)
                  for kind in sorted({catalogue[n]['kind'] for n in backgrounds})},
        'effective_site_pairs_kish': kish_effective_units(per_site_pair.values()),
        'contrasts_per_site_pair': dict(sorted(
            ((f'{i + 1}-{j + 1}', n) for (i, j), n in per_site_pair.items()), key=lambda r: -r[1])),
        'separation_strata': dict(sorted(strata.items())),
        'orders': dict(sorted({f'order_{c.order}': sum(1 for x in contrasts if x.order == c.order)
                               for c in contrasts}.items())),
    }


def run_inventory(args: argparse.Namespace) -> dict:
    header, keys, extra = admitted_states(args.data, args.manifest, ROOT / CATALOGUE)
    groups = json.loads((ROOT / GROUPING).read_bytes())
    if groups.get('status') != 'admitted':
        raise ValueError('family grouping is not admitted')
    catalogue = extra['catalogue']
    cuts = {}
    for cut in DISTANCE_CUTS:
        contrasts = enumerate_contrasts(keys['wildtypes'], keys['states'],
                                        max_background_distance=cut)
        cuts[f'background_distance_le_{cut}'] = tabulate(contrasts, groups['assignments'], catalogue)
    exact = enumerate_contrasts(keys['wildtypes'], keys['states'], max_background_distance=1)
    return {
        'schema': 'higher_order_support_inventory_v1',
        'stage': 'inventory',
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'qc_width_kcal_mol': QC_WIDTH_KCAL_MOL,
        'grouping': {'path': GROUPING, 'sha256': sha256_file(ROOT / GROUPING),
                     'final_groups': len(set(groups['assignments'].values()))},
        **header,
        'indel_state_contamination': indel_state_contamination(
            extra['frame'], catalogue, args.pairwise_cohort),
        'triple_mutant_endpoint': {
            'variants_with_three_or_more_substitutions_in_own_background':
                header['substitution_order_census']['all_rows'].get('substitutions_3_or_more', 0),
            'admitted_variants_with_three_or_more_substitutions':
                header['substitution_order_census']['admitted_rows'].get('substitutions_3_or_more', 0),
        },
        'background_dependence_endpoint': cuts,
        'exact_third_order_cubes': [
            {'reference': c.reference, 'variant': c.variant,
             'background_site': c.background_sites[0] + 1,
             'site_pair': [c.first[0] + 1, c.second[0] + 1],
             'substitutions': [f'{c.first[0] + 1}{c.first[1]}', f'{c.second[0] + 1}{c.second[1]}']}
            for c in exact],
    }


# ------------------------------------------------------------------ noise floor

def channel_vectors(contrasts: list[BackgroundContrast], measured: dict) -> pd.DataFrame:
    """Per-contrast omega in both channels, with its constituent epsilons."""
    rows = []
    for contrast in contrasts:
        cube = {channel: {corner: measured[name][applied][channel]
                          for corner, (name, applied) in contrast.addresses().items()}
                for channel in CHANNELS}
        row = {'reference': contrast.reference, 'variant': contrast.variant,
               'site_pair': contrast.site_pair, 'order': contrast.order,
               'mutation_pair': (contrast.first, contrast.second)}
        for channel, corners in cube.items():
            row[f'omega_{channel}'] = cycle_contrast(corners)
            for tag, bit in (('reference', 0), ('variant', 1)):
                row[f'epsilon_{tag}_{channel}'] = cycle_contrast(
                    {(a, b): corners[(a, b, bit)] for a in (0, 1) for b in (0, 1)})
        rows.append(row)
    return pd.DataFrame(rows)


def decompose(table: pd.DataFrame, prefix: str, unit: str) -> dict:
    """Between-channel decomposition of one quantity, resampled over ``unit``.

    ``_moments`` and ``_derive`` are the label instrument's own estimators, so the
    shared component remains an upper bound on reproducible signal and the
    per-channel discordance a lower bound on measurement noise.
    """
    trypsin = table[f'{prefix}_trypsin'].to_numpy(float)
    chymotrypsin = table[f'{prefix}_chymotrypsin'].to_numpy(float)
    grouped = table.groupby(unit, sort=True).indices
    vectors = [_moments(trypsin[idx], chymotrypsin[idx]) for idx in grouped.values()]
    sizes = [len(idx) for idx in grouped.values()]
    support = {'quantity': prefix, 'unit': unit, 'observations': int(len(table)),
               'units': len(vectors), 'effective_units_kish': kish_effective_units(sizes)}
    if len(vectors) < MIN_UNITS_FOR_INTERVAL:
        point = _derive(np.array(vectors).mean(axis=0))
        return {**support,
                'interval': None,
                'interval_withheld_reason':
                    f'{len(vectors)} resampling units is below the declared floor of '
                    f'{MIN_UNITS_FOR_INTERVAL}; a percentile interval over so few units '
                    'reports individual unit means, not a sampling interval',
                'estimates': {key: {'point': value} for key, value in point.items()}}
    return {**support, 'interval': {'resamples': BOOTSTRAP_RESAMPLES, 'seed': BOOTSTRAP_SEED},
            'estimates': _bootstrap(vectors, [], BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)}


def instrument_floor() -> dict:
    """The order-3 floor propagated from the instrument's order-2 measurement.

    The propagation is monotone in the discordance standard deviation, so the
    interval endpoints propagate with the point estimate. This is the floor that
    carries an interval; the directly measured order-3 discordance does not,
    because its support is two site pairs.
    """
    path = ROOT / INSTRUMENT
    digest = sha256_file(path)
    if digest != INSTRUMENT_SHA256:
        raise ValueError(f'label instrument qualification digest {digest} differs from the '
                         f'documented {INSTRUMENT_SHA256}')
    admitted = json.loads(path.read_bytes())['variants']['both_channel_width']['kinds']['natural']
    out = {'path': INSTRUMENT, 'sha256': digest,
           'propagation_factor': ORDER_THREE_PROPAGATION, 'strata': {}}
    for stratum in ('all', '10+'):
        entry = admitted[stratum]['source_cluster']
        measured = entry['agreement']['per_channel_discordance_sd_kcal_mol']
        out['strata'][stratum] = {
            'source_clusters': entry['units'],
            'effective_clusters_kish': entry['effective_units_kish'],
            'order_two_discordance_sd_kcal_mol': {
                'point': measured['point'], 'ci95': measured['ci95']},
            'predicted_order_three_discordance_sd_kcal_mol': {
                'point': ORDER_THREE_PROPAGATION * measured['point'],
                'ci95': [ORDER_THREE_PROPAGATION * bound for bound in measured['ci95']]},
        }
    return out


def run_noise_floor(args: argparse.Namespace) -> dict:
    declaration = json.loads(args.inventory.read_bytes())
    digest = sha256_file(args.inventory)
    if declaration.get('stage') != 'inventory':
        raise ValueError('--inventory must name an inventory-stage output')
    if args.declaration_digest and args.declaration_digest != digest:
        raise ValueError(f'inventory digest {digest} differs from the declared '
                         f'{args.declaration_digest}')
    _, keys, _ = admitted_states(args.data, args.manifest, ROOT / CATALOGUE)
    groups = json.loads((ROOT / GROUPING).read_bytes())['assignments']
    out = {
        'schema': 'higher_order_noise_floor_v1',
        'stage': 'noise-floor',
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'declaration': {'path': str(args.inventory.resolve().relative_to(ROOT)),
                        'sha256': digest},
        'min_units_for_interval': MIN_UNITS_FOR_INTERVAL,
        'propagated_instrument_floor': instrument_floor(),
        'cuts': {},
    }
    for cut in DISTANCE_CUTS:
        contrasts = enumerate_contrasts(keys['wildtypes'], keys['states'],
                                        max_background_distance=cut)
        declared = declaration['background_dependence_endpoint'][f'background_distance_le_{cut}']
        if declared['contrasts'] != len(contrasts):
            raise ValueError(f'support at cut {cut} differs from the declared inventory')
        table = channel_vectors(contrasts, keys['measured'])
        table['site_pair_key'] = table.site_pair.map(lambda p: f'{p[0] + 1}-{p[1] + 1}')
        table['group_key'] = table.reference.map(lambda n: groups.get(n, n))
        entry = {'declared_support': declared, 'omega': {}, 'epsilon': {}}
        for unit in ('site_pair_key', 'group_key'):
            entry['omega'][unit] = decompose(table, 'omega', unit)
            for tag in ('reference', 'variant'):
                entry['epsilon'].setdefault(unit, {})[tag] = decompose(
                    table, f'epsilon_{tag}', unit)
        # The order-3 floor predicted from the order-2 floor measured on this same
        # support: the two constituent cycles share no measurement, so the channel
        # difference variances add.
        pairwise = [entry['epsilon']['site_pair_key'][tag]['estimates']
                    ['per_channel_discordance_sd_kcal_mol']['point'] for tag in
                    ('reference', 'variant')]
        entry['propagated_floor'] = {
            'constituent_epsilon_discordance_sd_kcal_mol': pairwise,
            'predicted_omega_discordance_sd_kcal_mol': float(
                np.sqrt(sum(value ** 2 for value in pairwise))),
            'measured_omega_discordance_sd_kcal_mol': entry['omega']['site_pair_key']['estimates'][
                'per_channel_discordance_sd_kcal_mol']['point'],
            'measured_omega_shared_component_sd_kcal_mol': entry['omega']['site_pair_key'][
                'estimates']['shared_component_sd_kcal_mol']['point'],
            'measured_omega_shared_to_discordance_ratio': entry['omega']['site_pair_key'][
                'estimates']['shared_to_discordance_ratio']['point'],
        }
        out['cuts'][f'background_distance_le_{cut}'] = entry
    return out


# ------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('inventory', 'noise-floor'))
    parser.add_argument('--data', type=Path,
                        default=ROOT / 'data/megascale_tsuboyama2023/dataset2/data')
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'data/megascale_tsuboyama2023/manifest.json')
    parser.add_argument('--inventory', type=Path,
                        help='inventory-stage output, required by noise-floor')
    parser.add_argument('--pairwise-cohort', type=Path,
                        default=ROOT / 'archive/logs/R3/pairwise_cohort_20260924/cohort.json',
                        help='frozen pairwise cohort, read only to report how many of '
                             'its cycles rest on a length-changing construct')
    parser.add_argument('--declaration-digest',
                        help='sha256 the inventory output must still carry')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'noise-floor' and args.inventory is None:
        parser.error('noise-floor requires --inventory')
    report = run_inventory(args) if args.stage == 'inventory' else run_noise_floor(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(json.dumps({'stage': args.stage, 'out': str(args.out), 'sha256': sha256_file(args.out)},
                     indent=1))


if __name__ == '__main__':
    main()
