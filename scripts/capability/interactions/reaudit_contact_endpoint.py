#!/usr/bin/env python3
"""Recompute the published contact endpoint, then test two readings of its null.

The published contrast matches contact site pairs to non-contacts on separation,
burial and wild-type hydrophobicity, and a 20-bin in-sample curve of epsilon on
the measured additive prediction then moves the contrast from about +0.192
kcal/mol to about 0. This script recomputes that published estimate from the
annotation and the frozen cohort. If the recomputation disagrees with the
published artefact it stops. It does not rewrite that artefact.

Two sensitivities then ask why the contrast moved. The cross-fit fits the same
20-bin curve on the other family groups only, so a held-out family cannot train
the curve subtracted from its own cycles. The buried short-separation contacts
that the match deletes, because that cell has no non-contact control, are
reported as their own stratum against the retained contacts. Neither sensitivity
is a model-side analysis.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.interactions.contact_enrichment import (  # noqa: E402
    ADDITIVE_RESPONSE_BINS, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, SEPARATION_CELLS,
    cem_design, cross_fit_binned_residuals, group_equal_weight, resampled_arm_difference,
    training_curve, two_stage_bootstrap, weighted_difference)
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.stability.pairwise_stability import COHORT_SCHEMA  # noqa: E402
from scripts.capability.interactions import measure_contact_epsilon_enrichment as enrichment  # noqa: E402

SCHEMA = 'contact_endpoint_reaudit_v1'
TOLERANCE_KCAL_MOL = 1e-9
ANNOTATION = ROOT / 'archive/logs/shared/gate_structure_contact_20260924/contact_annotation.json'
COHORT = ROOT / 'archive/logs/R3/pairwise_cohort_20260924/cohort.json'
PUBLISHED = ROOT / 'archive/logs/shared/gate_structure_contact_20260924/epsilon_enrichment.json'
OUT = ROOT / 'results/R3/contact_endpoint_reaudit_20260926/endpoint_reaudit.json'

MATCHED = 'matched contact minus matched non-contact'
DROPPED = 'dropped buried short-separation contacts minus retained contacts'
CONTACT_ARM = ('contact', 'non_contact')
DROPPED_ARM = ('dropped_contacts', 'retained_contacts')


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f'missing required input: {path}')


def close(found, expected, name: str, failures: list[str]) -> None:
    if isinstance(expected, bool) or expected is None or isinstance(found, bool):
        if found != expected:
            failures.append(f'{name}: recomputed {found!r} != published {expected!r}')
        return
    if isinstance(expected, int) and not isinstance(expected, bool) and float(found) == float(expected):
        return
    if abs(float(found) - float(expected)) > TOLERANCE_KCAL_MOL:
        failures.append(
            f'{name}: recomputed {float(found)!r} != published {float(expected)!r} '
            f'(delta {float(found) - float(expected):+.3e})')


def buried_short(row: dict, boundary: float) -> bool:
    low, high = SEPARATION_CELLS[0]
    return low <= row['separation'] <= high and not (float(np.mean(row['rsa'])) > boundary)


def annotation_confirmation(annotation: dict, boundary: float) -> dict:
    identities = [row['structure']['identity_over_wildtype'] for row in annotation['backgrounds']]
    offsets = [row['structure']['wildtype_to_entity_offset'] for row in annotation['backgrounds']]
    pure = [offset for offset in offsets if isinstance(offset, int) and not isinstance(offset, bool)]
    record = {
        'backgrounds_annotated': len(annotation['backgrounds']),
        'backgrounds_excluded': len(annotation['excluded']),
        'identity_over_wildtype_min': float(min(identities)),
        'identity_over_wildtype_max': float(max(identities)),
        'pure_shift_offsets': len(pure),
        'offsets_that_are_not_a_pure_shift': len(offsets) - len(pure),
        'rsa_boundary': boundary,
        'contact_primary': annotation['definitions']['contact_primary'],
        'contact_secondary': annotation['definitions']['contact_secondary'],
    }
    if record['backgrounds_annotated'] != 64 or record['backgrounds_excluded'] != 0:
        raise SystemExit(f"annotation admission is {record['backgrounds_annotated']} annotated, "
                         f"{record['backgrounds_excluded']} excluded")
    if record['identity_over_wildtype_min'] < 1.0 or record['offsets_that_are_not_a_pure_shift']:
        raise SystemExit('annotation numbering is not whole-wild-type identity with a pure shift: '
                         + json.dumps(record))
    return record


def support_rows(annotation: dict, known: dict) -> tuple[list[dict], float]:
    annotated = [dict(pair, method=record['structure']['method'])
                 for record in annotation['backgrounds'] for pair in record['site_pairs']]
    boundary = float(np.median([np.mean(row['rsa']) for row in annotated]))
    unexpected = [pair for pair in known if pair not in {row['site_pair'] for row in annotated}]
    if unexpected:
        raise SystemExit(f'measured site pairs absent from the annotation: {unexpected[:5]}')
    return [row for row in annotated if row['site_pair'] in known], boundary


def matched_contrast(rows: list[dict], values: np.ndarray, treated: np.ndarray,
                     boundary: float, *, group_equal: bool) -> tuple[dict, dict, np.ndarray]:
    weight = cem_design(rows, treated, rsa_boundary=boundary)['weight']
    if group_equal:
        weight = group_equal_weight(weight, np.asarray([row['group'] for row in rows]), treated)
    point = weighted_difference(values, weight, treated)
    interval = two_stage_bootstrap(
        rows, values, treated, rsa_boundary=boundary, group_equal=group_equal,
        draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    return point, interval, weight


def published_shaped(point: dict, interval: dict, rows: list[dict], known: dict,
                     treated: np.ndarray, weight: np.ndarray) -> dict:
    groups = np.asarray([row['group'] for row in rows])
    contact = treated & (weight > 0)
    control = ~np.asarray(treated, dtype=bool) & (weight > 0)
    return {
        'difference': point['difference'],
        'interval': interval['interval'],
        'excludes_zero': interval['excludes_zero'],
        'contact_mean': point['contact'],
        'control_mean': point['control'],
        'contact_site_pairs': point['contact_site_pairs'],
        'control_site_pairs': point['control_site_pairs'],
        'effective_contact_site_pairs': point['effective_contact'],
        'effective_control_site_pairs': point['effective_control'],
        'contact_groups': len({group for group, ok in zip(groups, contact) if ok}),
        'control_groups': len({group for group, ok in zip(groups, control) if ok}),
        'cycles_contact': int(sum(known[row['site_pair']]['cycles']
                                  for row, ok in zip(rows, contact) if ok)),
        'cycles_control': int(sum(known[row['site_pair']]['cycles']
                                  for row, ok in zip(rows, control) if ok)),
        'bootstrap_draws': interval['draws'],
        'skipped_draws': interval['skipped_draws'],
    }


def compare_endpoint(name: str, found: dict, published: dict, failures: list[str]) -> float:
    largest = 0.0
    for key in ('difference', 'contact_mean', 'control_mean',
                'effective_contact_site_pairs', 'effective_control_site_pairs'):
        close(found[key], published[key], f'{name}.{key}', failures)
        largest = max(largest, abs(float(found[key]) - float(published[key])))
    for index, edge in enumerate(('low', 'high')):
        close(found['interval'][index], published['interval'][index],
              f'{name}.interval_{edge}', failures)
        largest = max(largest, abs(float(found['interval'][index]) - float(published['interval'][index])))
    for key in ('excludes_zero', 'contact_site_pairs', 'control_site_pairs',
                'contact_groups', 'control_groups', 'cycles_contact', 'cycles_control',
                'bootstrap_draws', 'skipped_draws'):
        close(found[key], published[key], f'{name}.{key}', failures)
    return largest


def estimate_row(name: str, role: str, estimand: str, weighting: str, endpoint: str,
                 arm_names: tuple[str, str], point: dict, interval: dict,
                 groups: np.ndarray, weight: np.ndarray, arm: np.ndarray) -> dict:
    arm = np.asarray(arm, dtype=bool)
    weight = np.asarray(weight, dtype=float)
    groups = np.asarray(groups)
    left = arm & (weight > 0)
    right = ~arm & (weight > 0)
    active = weight > 0
    return {
        'name': name,
        'role': role,
        'estimand': estimand,
        'weighting': weighting,
        'endpoint': endpoint,
        'difference_kcal_mol': float(point['difference']),
        'interval_95_kcal_mol': [float(bound) for bound in interval['interval']],
        'excludes_zero': bool(interval['excludes_zero']),
        'n_groups': len(set(groups[active].tolist())),
        'n_pairs': int(active.sum()),
        'n_groups_by_arm': {
            arm_names[0]: len(set(groups[left].tolist())),
            arm_names[1]: len(set(groups[right].tolist())),
        },
        'n_pairs_by_arm': {
            arm_names[0]: int(left.sum()),
            arm_names[1]: int(right.sum()),
        },
        'kish_effective_controls': float(point['effective_control']),
        'means_kcal_mol': {
            arm_names[0]: float(point['contact']),
            arm_names[1]: float(point['control']),
        },
        'bootstrap_groups': int(interval['n_units']),
        'bootstrap_draws': int(interval['draws']),
        'skipped_draws': int(interval['skipped_draws']),
        'bootstrap_unit': interval['unit'],
    }


def values_for(rows: list[dict], known: dict, endpoint: str) -> np.ndarray:
    missing = [row['site_pair'] for row in rows if known[row['site_pair']][endpoint] is None]
    if missing:
        raise SystemExit(f'{endpoint} is missing for {missing[:3]}')
    return np.asarray([known[row['site_pair']][endpoint] for row in rows], dtype=float)


def site_pair_means(cycles: list[dict], residuals: np.ndarray) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for cycle, residual in zip(cycles, residuals):
        buckets.setdefault(cycle['site_pair'], []).append(abs(float(residual)))
    return {pair: float(np.mean(values)) for pair, values in buckets.items()}


def cross_fit_audit(additive: np.ndarray, epsilon: np.ndarray, groups: np.ndarray) -> dict:
    """Every held-out group is absent from the curve applied to it."""

    labels = sorted(set(groups.tolist()))
    if len(labels) < 2:
        raise SystemExit('cross-fit needs at least two family groups')
    train_counts = []
    for label in labels:
        held = groups == label
        curve = training_curve(additive, epsilon, groups, label, ADDITIVE_RESPONSE_BINS)
        if curve['n_train'] != int(len(epsilon) - held.sum()) or curve['n_train'] == len(epsilon):
            raise SystemExit(f'group {label} entered the curve that adjusts it')
        train_counts.append(curve['n_train'])
    return {
        'bins': ADDITIVE_RESPONSE_BINS,
        'training': 'all family groups except the held-out group',
        'held_out_groups': len(labels),
        'full_sample_cycles': int(len(epsilon)),
        'min_training_cycles': int(min(train_counts)),
        'max_training_cycles': int(max(train_counts)),
        'held_out_excluded_from_its_curve': True,
        'curve_refit_inside_bootstrap': False,
        'interval_note': (
            'The 95% interval resamples family groups and then site pairs and re-derives '
            'matching or arm weights. The additive-response curve is the leave-one-group-out '
            'curve of the observed sample and is not refit inside the draws, which is how the '
            'published in-sample interval treats its own curve.'),
    }


def dropped_partition(rows: list[dict], treated: np.ndarray, boundary: float,
                      known: dict) -> tuple[list[dict], np.ndarray, np.ndarray]:
    design = cem_design(rows, treated, rsa_boundary=boundary)
    kept_rows, arm = [], []
    for row, is_contact, cell, weight in zip(rows, treated, design['cells'], design['weight']):
        if not is_contact or cell is None:
            continue
        kept_rows.append(row)
        arm.append(not (weight > 0))
    arm_array = np.asarray(arm, dtype=bool)
    dropped = {row['site_pair'] for row, flag in zip(kept_rows, arm_array) if flag}
    buried = {row['site_pair'] for row, is_contact in zip(rows, treated)
              if is_contact and buried_short(row, boundary)}
    if dropped != buried:
        raise SystemExit(
            'the contacts deleted for lack of a control are not the buried short-separation '
            f'stratum ({len(dropped)} deleted, {len(buried)} buried short, '
            f'{len(dropped - buried)} deleted outside that stratum)')
    if len(dropped) != design['pruned_contacts_without_control']:
        raise SystemExit('pruned-contact count disagrees with the buried short-separation stratum')
    values = np.asarray([known[row['site_pair']]['mean_abs_epsilon'] for row in kept_rows])
    return kept_rows, values, arm_array


def arm_contrast(values: np.ndarray, groups: np.ndarray, arm: np.ndarray,
                 *, group_equal: bool) -> tuple[dict, dict, np.ndarray]:
    weight = np.ones(len(arm), dtype=float)
    if group_equal:
        weight = group_equal_weight(weight, groups, arm)
    point = weighted_difference(values, weight, arm)
    interval = resampled_arm_difference(
        values, groups, arm, group_equal=group_equal,
        draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    if interval['interval'] is None:
        raise SystemExit('the dropped-contact stratum is below the bootstrap unit floor')
    if abs(float(interval['point']) - float(point['difference'])) > TOLERANCE_KCAL_MOL:
        raise SystemExit('dropped-stratum point disagrees with the point inside its bootstrap')
    return point, interval, weight


def fmt(row: dict) -> str:
    low, high = row['interval_95_kcal_mol']
    return f"{row['difference_kcal_mol']:+.4f} [{low:+.4f}, {high:+.4f}] kcal/mol"


def build_verdict(estimates: list[dict]) -> str:
    by_name = {row['name']: row for row in estimates}
    raw = by_name['published_matched_site_pair_equal']
    ins = by_name['in_sample_20bin_site_pair_equal']
    cross = by_name['cross_fit_20bin_site_pair_equal']
    raw_g = by_name['published_matched_group_equal']
    ins_g = by_name['in_sample_20bin_group_equal']
    cross_g = by_name['cross_fit_20bin_group_equal']
    dropped = by_name['dropped_buried_short_versus_retained_site_pair_equal']
    dropped_g = by_name['dropped_buried_short_versus_retained_group_equal']
    low, high = ins['interval_95_kcal_mol']
    gap_in_sample = cross['difference_kcal_mol'] - ins['difference_kcal_mol']
    gap_raw = cross['difference_kcal_mol'] - raw['difference_kcal_mol']
    if low <= cross['difference_kcal_mol'] <= high:
        move = (
            f"The cross-fit point differs from the in-sample point by {gap_in_sample:+.4f} kcal/mol "
            f"and from the unadjusted point by {gap_raw:+.4f} kcal/mol, and it lies inside the "
            f"in-sample interval, so the move from {raw['difference_kcal_mol']:+.4f} kcal/mol to about 0 "
            "is reproduced when each family's 20-bin curve is fit on the other families only."
        )
    else:
        move = (
            f"The cross-fit point differs from the in-sample point by {gap_in_sample:+.4f} kcal/mol "
            f"and from the unadjusted point by {gap_raw:+.4f} kcal/mol, and it lies outside the "
            f"in-sample interval [{low:+.4f}, {high:+.4f}], so the move from "
            f"{raw['difference_kcal_mol']:+.4f} kcal/mol to about 0 is produced by the in-sample curve "
            "and is not reproduced by the cross-fit."
        )
    matched = [row for row in estimates if row['estimand'] == MATCHED]
    excluding = [row['name'] for row in matched if row['excludes_zero']]
    if excluding:
        qualify = ("Matched contact-versus-non-contact intervals that exclude zero: "
                   + ", ".join(excluding) + ".")
    else:
        qualify = (
            "No matched contact-versus-non-contact sensitivity has a 95% interval that excludes zero, "
            "so the endpoint still does not qualify for a model-side run."
        )
    dropped_groups = dropped['n_groups_by_arm']
    dropped_means = dropped['means_kcal_mol']
    opening = (
        f"On the indel-excluded support the published site-pair-equal contrast of mean absolute epsilon "
        f"is {fmt(raw)}, the in-sample 20-bin adjustment is {fmt(ins)}, and the family-cross-fitted "
        f"20-bin adjustment is {fmt(cross)}. Under group-equal weights the same three contrasts are "
        f"{fmt(raw_g)}, {fmt(ins_g)} and {fmt(cross_g)}."
    )
    stratum = (
        f"The {dropped['n_pairs_by_arm']['dropped_contacts']} buried short-separation contacts deleted "
        f"by the match, in {dropped_groups['dropped_contacts']} groups, have mean absolute epsilon "
        f"{dropped_means['dropped_contacts']:+.4f} kcal/mol against {dropped_means['retained_contacts']:+.4f} "
        f"for the {dropped['n_pairs_by_arm']['retained_contacts']} retained contacts in "
        f"{dropped_groups['retained_contacts']} groups; their difference is {fmt(dropped)} under "
        f"site-pair-equal weights and {fmt(dropped_g)} under group-equal weights, and that comparison "
        "has no non-contact control."
    )
    return " ".join((opening, move, stratum, qualify))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--published', type=Path, default=PUBLISHED)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    published_path = args.published
    for path in (ANNOTATION, COHORT, published_path):
        require_file(path)
    annotation = json.loads(ANNOTATION.read_text())
    cohort = json.loads(COHORT.read_text())
    published = json.loads(published_path.read_text())
    if annotation.get('schema') != 'contact_annotation_v1':
        raise SystemExit('unexpected annotation schema')
    if cohort.get('schema') != COHORT_SCHEMA:
        raise SystemExit('unexpected cohort schema')
    if published.get('schema') != 'contact_epsilon_enrichment_v1':
        raise SystemExit('unexpected published enrichment schema')
    cohort_sha = sha256_file(COHORT)
    if cohort_sha != annotation['inputs']['cohort']['sha256']:
        raise SystemExit('cohort digest does not match the annotation it is read with')
    if cohort_sha != published['inputs']['cohort_sha256']:
        raise SystemExit('cohort digest does not match the published enrichment')
    annotation_sha = sha256_file(ANNOTATION)
    if annotation_sha != published['inputs']['annotation']['sha256']:
        raise SystemExit('annotation digest does not match the published enrichment')

    parquet_paths = []
    declared = {Path(entry['path']).name: entry['sha256'] for entry in cohort['source_files']}
    for relative in cohort['source_row_order']:
        path = ROOT / relative
        require_file(path)
        found = sha256_file(path)
        if declared.get(path.name) != found or published['inputs']['parquet_sha256'].get(path.name) != found:
            raise SystemExit(f'{path.name} does not match the cohort or the published enrichment digest')
        parquet_paths.append(path)

    sequences = {row['name']: set(row['sequences']) for row in cohort['backgrounds']}
    print('reading pinned mut_type rows for the indel filter', file=sys.stderr)
    channels = enrichment.channel_states(parquet_paths, sequences)
    measured = enrichment.site_pair_statistics(cohort, channels, exclude_indel=True)
    known = measured['site_pairs']
    rows, boundary = support_rows(annotation, known)
    confirmation = annotation_confirmation(annotation, boundary)
    if abs(boundary - float(published['declaration']['rsa_boundary'])) > TOLERANCE_KCAL_MOL:
        raise SystemExit(f'RSA boundary {boundary} != published {published["declaration"]["rsa_boundary"]}')

    published_support = published['supports']['indel_excluded']
    heavy = published_support['definitions']['heavy_atom']
    floor = measured['floor']
    failures: list[str] = []
    close(len({row['group'] for row in rows}), published_support['support']['groups'],
          'support.groups', failures)
    close(len(rows), published_support['support']['site_pairs'], 'support.site_pairs', failures)
    close(sum(known[row['site_pair']]['cycles'] for row in rows),
          published_support['support']['cycles'], 'support.cycles', failures)
    for key in ('cycles', 'excluded_indel_cycles', 'excluded_indel_states', 'response_bins'):
        close(floor[key], published_support['label_instrument_floor'][key], f'floor.{key}', failures)
    close(floor['response_range_kcal_mol'],
          published_support['label_instrument_floor']['response_range_kcal_mol'],
          'floor.response_range_kcal_mol', failures)
    if floor['excluded_indel_site_pairs'] != published_support['label_instrument_floor']['excluded_indel_site_pairs']:
        failures.append('excluded indel site pairs disagree with the published floor')
    treated = enrichment.treated_mask(rows, 'heavy_atom')
    design = cem_design(rows, treated, rsa_boundary=boundary)
    close(int(treated.sum()), heavy['contacts'], 'contacts', failures)
    close(int((~treated).sum()), heavy['non_contacts'], 'non_contacts', failures)
    close(design['pruned_contacts_without_control'], heavy['pruned_contacts_without_control'],
          'pruned_contacts_without_control', failures)
    close(design['pruned_ineligible_separation'], heavy['pruned_ineligible_separation'],
          'pruned_ineligible_separation', failures)
    if failures:
        raise SystemExit('published endpoint did not match before the interval check:\n'
                         + '\n'.join(failures))

    verification_plan = (
        ('mean_abs_epsilon', 'site_pair_equal', False, 'published_matched_site_pair_equal'),
        ('mean_abs_epsilon', 'group_equal', True, 'published_matched_group_equal'),
        ('mean_abs_epsilon_adjusted', 'site_pair_equal', False, 'in_sample_20bin_site_pair_equal'),
        ('mean_abs_epsilon_adjusted', 'group_equal', True, 'in_sample_20bin_group_equal'),
        ('mean_abs_half_channel_difference', 'site_pair_equal', False, 'noise_half_difference_site_pair_equal'),
    )
    verified_endpoints = []
    reproduced: dict[str, dict] = {}
    groups = np.asarray([row['group'] for row in rows])
    prepared = []
    for endpoint, weighting, group_equal, name in verification_plan:
        values = values_for(rows, known, endpoint)
        weight = cem_design(rows, treated, rsa_boundary=boundary)['weight']
        if group_equal:
            weight = group_equal_weight(weight, groups, treated)
        point = weighted_difference(values, weight, treated)
        published_endpoint = heavy['endpoints'][endpoint][weighting]
        for key, found_key in (
                ('difference', 'difference'), ('contact_mean', 'contact'),
                ('control_mean', 'control'),
                ('effective_contact_site_pairs', 'effective_contact'),
                ('effective_control_site_pairs', 'effective_control'),
                ('contact_site_pairs', 'contact_site_pairs'),
                ('control_site_pairs', 'control_site_pairs')):
            close(point[found_key], published_endpoint[key], f'{name}.{key}', failures)
        prepared.append((endpoint, weighting, group_equal, name, values, weight, point,
                         published_endpoint))
    if failures:
        raise SystemExit('published point estimate did not match:\n' + '\n'.join(failures))
    for endpoint, weighting, group_equal, name, values, weight, point, published_endpoint in prepared:
        print(f'bootstrap {name}', file=sys.stderr)
        _point, interval, weight = matched_contrast(
            rows, values, treated, boundary, group_equal=group_equal)
        if abs(float(_point['difference']) - float(point['difference'])) > TOLERANCE_KCAL_MOL:
            raise SystemExit(f'{name}: bootstrap point disagrees with the pre-interval point')
        found = published_shaped(point, interval, rows, known, treated, weight)
        delta = compare_endpoint(name, found, published_endpoint, failures)
        verified_endpoints.append({
            'name': name, 'published_endpoint': endpoint, 'weighting': weighting,
            'max_abs_delta': delta, 'recomputed': found,
            'published': {key: published_endpoint[key] for key in found},
        })
        if endpoint != 'mean_abs_half_channel_difference':
            reproduced[name] = estimate_row(
                name, 'reproduction', MATCHED, weighting, endpoint, CONTACT_ARM,
                point, interval, groups, weight, treated)
    if failures:
        raise SystemExit('published endpoint did not match:\n' + '\n'.join(failures))

    supported = enrichment.support_cycles(cohort, channels, exclude_indel=True)
    cycles = supported['cycles']
    if len(cycles) != floor['cycles']:
        raise SystemExit('cycle table length disagrees with the site-pair floor')
    additive = np.asarray([cycle['additive'] for cycle in cycles], dtype=float)
    epsilon = np.asarray([cycle['epsilon'] for cycle in cycles], dtype=float)
    cycle_groups = np.asarray([cycle['group'] for cycle in cycles])
    audit = cross_fit_audit(additive, epsilon, cycle_groups)
    print('cross-fit residualization', file=sys.stderr)
    residuals = cross_fit_binned_residuals(
        additive, epsilon, cycle_groups, ADDITIVE_RESPONSE_BINS)
    cross_means = site_pair_means(cycles, residuals)
    missing = [row['site_pair'] for row in rows if row['site_pair'] not in cross_means]
    if missing:
        raise SystemExit(f'cross-fit means missing for {missing[:3]}')
    cross_values = np.asarray([cross_means[row['site_pair']] for row in rows], dtype=float)

    estimates = [
        reproduced['published_matched_site_pair_equal'],
        reproduced['published_matched_group_equal'],
        reproduced['in_sample_20bin_site_pair_equal'],
        reproduced['in_sample_20bin_group_equal'],
    ]
    for weighting, group_equal, name in (
            ('site_pair_equal', False, 'cross_fit_20bin_site_pair_equal'),
            ('group_equal', True, 'cross_fit_20bin_group_equal')):
        print(f'bootstrap {name}', file=sys.stderr)
        point, interval, weight = matched_contrast(
            rows, cross_values, treated, boundary, group_equal=group_equal)
        estimates.append(estimate_row(
            name, 'sensitivity', MATCHED, weighting, 'mean_abs_cross_fit_epsilon',
            CONTACT_ARM, point, interval, groups, weight, treated))

    dropped_rows, dropped_values, dropped_arm = dropped_partition(rows, treated, boundary, known)
    dropped_groups = np.asarray([row['group'] for row in dropped_rows])
    if int(dropped_arm.sum()) != 25 or int((~dropped_arm).sum()) != 111:
        raise SystemExit(
            f"dropped stratum is {int(dropped_arm.sum())} contacts against "
            f"{int((~dropped_arm).sum())} retained, not 25 against 111")
    for weighting, group_equal, name in (
            ('site_pair_equal', False, 'dropped_buried_short_versus_retained_site_pair_equal'),
            ('group_equal', True, 'dropped_buried_short_versus_retained_group_equal')):
        print(f'bootstrap {name}', file=sys.stderr)
        point, interval, weight = arm_contrast(
            dropped_values, dropped_groups, dropped_arm, group_equal=group_equal)
        estimates.append(estimate_row(
            name, 'sensitivity', DROPPED, weighting, 'mean_abs_epsilon',
            DROPPED_ARM, point, interval, dropped_groups, weight, dropped_arm))

    report = {
        'schema': SCHEMA,
        'question': (
            'Does any contact-versus-non-contact sensitivity of the published R4 endpoint '
            'have a 95% interval that excludes zero, and is the move from the unadjusted '
            'contrast to about 0 reproduced by a family-cross-fitted curve or only by the '
            'in-sample curve?'),
        'verified_recomputation': {
            'matches_published': True,
            'comparison_tolerance_kcal_mol': TOLERANCE_KCAL_MOL,
            'annotation_sha256': annotation_sha,
            'cohort_sha256': cohort_sha,
            'published_sha256': sha256_file(published_path),
            'published_path': str(published_path),
            'annotation_confirmation': confirmation,
            'support': {
                'groups': len({row['group'] for row in rows}),
                'site_pairs': len(rows),
                'cycles': int(sum(known[row['site_pair']]['cycles'] for row in rows)),
                'contacts': int(treated.sum()),
                'non_contacts': int((~treated).sum()),
                'pruned_contacts_without_control': int(design['pruned_contacts_without_control']),
                'retained_contact_site_pairs': int((treated & (design['weight'] > 0)).sum()),
                'retained_noncontact_site_pairs': int((~treated & (design['weight'] > 0)).sum()),
                'rsa_boundary': boundary,
            },
            'endpoints': verified_endpoints,
            'note': (
                'These fields were recomputed from the annotation, the cohort epsilon and '
                'additive columns, and the pinned mut_type filter, then compared with '
                'epsilon_enrichment.json. The published file was not modified.'),
        },
        'procedure': audit,
        'estimates': estimates,
        'verdict': build_verdict(estimates),
    }
    write_json(args.out, report)
    print(json.dumps({'out': str(args.out), 'verdict': report['verdict']}, indent=1))


if __name__ == '__main__':
    main()
