#!/usr/bin/env python3
"""Collect completed depth-sweep cell reports into one panel summary.

It copies numbers out of the cell reports and counts interval directions over
them. It fits nothing, resamples nothing and loads no model: a count of resolved
intervals here is a count, not a corrected significance statement and not a
prevalence in any model population.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.capability.readouts import readout_depth as rd

AXES = ('depth', 'pos', 'wide', 'full')


def digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def direction(record) -> str:
    interval = record.get('interval')
    if interval is None:
        return 'degenerate'
    low, high = interval
    if low > 0:
        return 'resolved_positive'
    if high < 0:
        return 'resolved_negative'
    return 'unresolved'


def axis_profile(summaries: dict, axis: str, depth: int) -> list[dict]:
    """One axis's increment over the matched baseline, depth by depth."""
    profile = []
    for index in range(depth):
        key = f'{axis}{index:03d}_delta_spearman'
        if key not in summaries:
            continue
        record = summaries[key]
        representation = summaries.get(f'{axis}{index:03d}/R_minus_raw_M_spearman', {})
        profile.append(dict(
            depth=index, relative_depth=round(index / max(depth - 1, 1), 4),
            delta_spearman=record['point'], interval=record.get('interval'),
            direction=direction(record),
            delta_rank_mse=summaries[f'{axis}{index:03d}_delta_rank_mse']['point'],
            delta_rank_mse_interval=summaries[f'{axis}{index:03d}_delta_rank_mse'].get('interval'),
            r_spearman=summaries[f'{axis}{index:03d}/R_spearman']['point'],
            b_r_spearman=summaries[f'{axis}{index:03d}/B_R_spearman']['point'],
            r_minus_raw_m=representation.get('point'),
            r_minus_raw_m_interval=representation.get('interval')))
    return profile


def collect(paths: list[Path]) -> dict:
    cells = []
    for path in sorted(paths):
        report = json.loads(path.read_text())
        if report.get('schema_version') != rd.ANALYSIS_SCHEMA or report['status'] != 'complete':
            raise ValueError(f'{path} is not a complete depth-sweep cell report')
        depth = report['extraction']['block_count']
        middle, final = rd.admitted_block_indices(depth)
        cell = dict(path=str(path), sha256=digest(path), arm=report['arm'],
                    panel=report['panel'], stratum=report['stratum'], depth=depth,
                    hidden_width=report['extraction']['hidden_width'],
                    position_resolved=report['extraction']['position_resolved'],
                    admitted_block_indices=[middle, final],
                    support=report['support'],
                    extraction_agreement=report['extraction_agreement'],
                    prefix_control=report['prefix_control'], seeds={})
        for seed, payload in report['seeds'].items():
            summaries = payload['summaries']
            entry = dict(
                reproduced_admitted_delta_spearman=summaries['admitted_delta_spearman']['point'],
                reproduced_admitted_interval=summaries['admitted_delta_spearman'].get('interval'),
                reproduced_admitted_direction=direction(summaries['admitted_delta_spearman']),
                raw_M_spearman=summaries['raw_M_spearman']['point'],
                raw_P_spearman=summaries['raw_P_spearman']['point'],
                baseline_spearman=summaries['baseline/B_spearman']['point'],
                baseline_agreement=payload['baseline_agreement'],
                axes={})
            for axis in AXES:
                profile = axis_profile(summaries, axis, depth)
                if not profile:
                    continue
                resolved = [row for row in profile if row['direction'] == 'resolved_positive']
                best = max(profile, key=lambda row: (row['delta_spearman'] is not None,
                                                     row['delta_spearman']
                                                     if row['delta_spearman'] is not None
                                                     else float('-inf')))
                entry['axes'][axis] = dict(
                    profile=profile, depths=len(profile),
                    resolved_positive=len(resolved),
                    resolved_negative=sum(1 for row in profile
                                          if row['direction'] == 'resolved_negative'),
                    unresolved=sum(1 for row in profile if row['direction'] == 'unresolved'),
                    resolved_positive_depths=[row['depth'] for row in resolved],
                    max_delta_spearman=best['delta_spearman'],
                    max_delta_spearman_depth=best['depth'],
                    max_delta_spearman_interval=best['interval'],
                    max_delta_spearman_direction=best['direction'])
            for extra in ('union',):
                key = f'{extra}_delta_spearman'
                if key in summaries:
                    entry[extra] = dict(delta_spearman=summaries[key]['point'],
                                        interval=summaries[key].get('interval'),
                                        direction=direction(summaries[key]),
                                        r_spearman=summaries[f'{extra}/R_spearman']['point'],
                                        b_r_spearman=summaries[f'{extra}/B_R_spearman']['point'])
            for label in ('permuted/B_spearman', 'permuted/admitted_B_R_spearman'):
                if label in summaries:
                    entry.setdefault('label_shuffle', {})[label] = dict(
                        point=summaries[label]['point'], interval=summaries[label].get('interval'),
                        direction=direction(summaries[label]))
            cell['seeds'][seed] = entry
        cells.append(cell)
    return cells


def panel_counts(cells: list[dict]) -> dict:
    """Interval-direction counts per stratum and per axis, never pooling strata."""
    counts: dict[str, dict] = {}
    for cell in cells:
        for seed, entry in cell['seeds'].items():
            for axis, payload in entry['axes'].items():
                key = f"{cell['panel']}/{cell['stratum']}/{axis}"
                bucket = counts.setdefault(key, dict(
                    panel=cell['panel'], stratum=cell['stratum'], axis=axis, arms=set(),
                    arm_seed_cells=0, depth_cells=0, resolved_positive=0, resolved_negative=0,
                    unresolved=0, arms_with_any_resolved_positive=set(),
                    max_delta_spearman=None, max_delta_spearman_arm=None,
                    max_delta_spearman_depth=None))
                bucket['arms'].add(cell['arm'])
                bucket['arm_seed_cells'] += 1
                bucket['depth_cells'] += payload['depths']
                for name in ('resolved_positive', 'resolved_negative', 'unresolved'):
                    bucket[name] += payload[name]
                if payload['resolved_positive']:
                    bucket['arms_with_any_resolved_positive'].add(cell['arm'])
                point = payload['max_delta_spearman']
                if point is not None and (bucket['max_delta_spearman'] is None
                                          or point > bucket['max_delta_spearman']):
                    bucket.update(max_delta_spearman=point, max_delta_spearman_arm=cell['arm'],
                                  max_delta_spearman_depth=payload['max_delta_spearman_depth'],
                                  max_delta_spearman_seed=seed)
    for bucket in counts.values():
        bucket['arms'] = sorted(bucket['arms'])
        bucket['arms_with_any_resolved_positive'] = sorted(bucket['arms_with_any_resolved_positive'])
    return counts


def refuse_duplicate_cells(cells: list[dict]) -> None:
    """Refuse a report set that covers one (arm, panel) cell twice.

    A cell can be computed twice — a killed lane's child may finish after its
    runner, and a relocated lane may race the original — and two reports of one
    cell would double every count that sums over cells.
    """
    seen: dict[tuple[str, str], int] = {}
    for cell in cells:
        key = (cell['arm'], cell['panel'])
        seen[key] = seen.get(key, 0) + 1
    repeated = sorted(key for key, count in seen.items() if count > 1)
    if repeated:
        listed = ', '.join(f'{arm}/{panel}' for arm, panel in repeated)
        raise ValueError(f'{len(cells)} reports cover {len(seen)} distinct cells; '
                         f'pass one report per cell rather than two for {listed}')


def stratum_tally(cells: list[dict]) -> dict:
    """Per stratum, the tally of depth cells resolved above and below zero.

    This is the quantity that licenses reading the rest of the panel, and it is a
    cross-cell aggregate, so no per-cell report can hold it. It is written here as
    an explicit field rather than left for a reader to recompute from the counts.
    """
    tally: dict[str, dict] = {}
    for cell in cells:
        for seed, entry in cell['seeds'].items():
            for axis, payload in entry['axes'].items():
                bucket = tally.setdefault(cell['stratum'], dict(
                    stratum=cell['stratum'], depth_cells=0, resolved_positive=0,
                    resolved_negative=0, unresolved=0, degenerate=0, arms=set(), panels=set(),
                    axes=set(), best_point=None, best_cell=None))
                bucket['arms'].add(cell['arm'])
                bucket['panels'].add(cell['panel'])
                bucket['axes'].add(axis)
                for row in payload['profile']:
                    bucket['depth_cells'] += 1
                    bucket[row['direction']] = bucket.get(row['direction'], 0) + 1
                    point = row['delta_spearman']
                    if point is not None and (bucket['best_point'] is None
                                              or point > bucket['best_point']):
                        bucket.update(best_point=point, best_cell=dict(
                            arm=cell['arm'], panel=cell['panel'], seed=seed, axis=axis,
                            depth=row['depth'], direction=row['direction'],
                            interval=row['interval']))
    for bucket in tally.values():
        for key in ('arms', 'panels', 'axes'):
            bucket[key] = sorted(bucket[key])
    return tally


def seed_consistent(cells: list[dict]) -> list[dict]:
    """Blocks resolved above zero at every declared seed, per arm, panel and axis.

    Reported because an unadjusted count over 1,004 blocks invites a tail artifact,
    while a block that resolves at every split seed does not.
    """
    out = []
    for cell in cells:
        seeds = sorted(cell['seeds'])
        for axis in AXES:
            per_seed, points = [], {}
            for seed in seeds:
                payload = cell['seeds'][seed]['axes'].get(axis)
                if payload is None:
                    per_seed = None
                    break
                per_seed.append({row['depth'] for row in payload['profile']
                                 if row['direction'] == 'resolved_positive'})
                for row in payload['profile']:
                    points.setdefault(row['depth'], []).append(row['delta_spearman'])
            if not per_seed or len(per_seed) != len(seeds):
                continue
            for depth in sorted(set.intersection(*per_seed)):
                values = points[depth]
                out.append(dict(arm=cell['arm'], panel=cell['panel'], stratum=cell['stratum'],
                                axis=axis, depth=depth, blocks=cell['depth'],
                                relative_depth=round(depth / max(cell['depth'] - 1, 1), 4),
                                admitted_block_indices=cell['admitted_block_indices'],
                                at_admitted_depth=depth in cell['admitted_block_indices'],
                                seeds=len(seeds), minimum=min(values), maximum=max(values)))
    return sorted(out, key=lambda row: -row['minimum'])


def breadth(cells: list[dict], consistent: list[dict]) -> list[dict]:
    """Per arm and panel, the admitted two-depth increment against the selected block."""
    pooled = {}
    for row in consistent:
        if row['axis'] != 'depth':
            continue
        key = (row['arm'], row['panel'])
        if key not in pooled or row['minimum'] > pooled[key]['minimum']:
            pooled[key] = row
    out = []
    for cell in cells:
        seeds = sorted(cell['seeds'])
        admitted = [cell['seeds'][s]['reproduced_admitted_delta_spearman'] for s in seeds]
        resolved = sum(1 for s in seeds
                       if cell['seeds'][s]['reproduced_admitted_direction'] == 'resolved_positive')
        row = pooled.get((cell['arm'], cell['panel']))
        out.append(dict(arm=cell['arm'], panel=cell['panel'], stratum=cell['stratum'],
                        seeds=len(seeds), admitted_minimum=min(admitted),
                        admitted_maximum=max(admitted), admitted_seeds_resolved=resolved,
                        selected_block=None if row is None else row['depth'],
                        selected_minimum=None if row is None else row['minimum'],
                        selected_maximum=None if row is None else row['maximum'],
                        selected_at_admitted_depth=None if row is None else row['at_admitted_depth'],
                        reading=('no seed-consistent block' if row is None else
                                 'boundary lifted' if resolved == 0 else 'already resolved')))
    return sorted(out, key=lambda r: (r['selected_block'] is None, r['arm'], r['panel']))


def ceilings(cells: list[dict], consistent: list[dict]) -> dict:
    """Largest increment per panel and stratum, seed-consistent and single-cell alike."""
    out: dict[str, dict] = {}
    for cell in cells:
        for seed, entry in cell['seeds'].items():
            for axis, payload in entry['axes'].items():
                key = f"{cell['panel']}/{cell['stratum']}"
                bucket = out.setdefault(key, dict(panel=cell['panel'], stratum=cell['stratum'],
                                                  single_cell_maximum=None, single_cell=None,
                                                  seed_consistent_maximum=None,
                                                  seed_consistent=None))
                for row in payload['profile']:
                    point = row['delta_spearman']
                    if point is None or row['direction'] != 'resolved_positive':
                        continue
                    if (bucket['single_cell_maximum'] is None
                            or point > bucket['single_cell_maximum']):
                        bucket.update(single_cell_maximum=point, single_cell=dict(
                            arm=cell['arm'], seed=seed, axis=axis, depth=row['depth'],
                            interval=row['interval']))
    for row in consistent:
        key = f"{row['panel']}/{row['stratum']}"
        bucket = out.setdefault(key, dict(panel=row['panel'], stratum=row['stratum'],
                                          single_cell_maximum=None, single_cell=None,
                                          seed_consistent_maximum=None, seed_consistent=None))
        if (bucket['seed_consistent_maximum'] is None
                or row['maximum'] > bucket['seed_consistent_maximum']):
            bucket.update(seed_consistent_maximum=row['maximum'], seed_consistent=dict(
                arm=row['arm'], axis=row['axis'], depth=row['depth'], minimum=row['minimum']))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', action='append', type=Path, required=True,
                        help='One completed depth-sweep cell report; repeat as needed')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', default='cpu', choices=['cpu'])
    args = parser.parse_args()
    cells = collect(args.report)
    refuse_duplicate_cells(cells)
    consistent = seed_consistent(cells)
    reproduction = [dict(arm=cell['arm'], panel=cell['panel'], seed=seed,
                         extraction_max_relative_l2=cell['extraction_agreement']['max_relative_l2'],
                         extraction_exactly_equal=cell['extraction_agreement']['exactly_equal'],
                         likelihood_difference_nats=cell['extraction_agreement'][
                             'max_absolute_likelihood_difference_nats'],
                         worst_admitted_prediction_deviation=max(
                             value for key, value in
                             entry['baseline_agreement']['max_absolute_deviation'].items()
                             if key.endswith('_prediction')),
                         pipeline_identity=entry['baseline_agreement']
                         .get('pipeline_identity', {}).get('worst_prediction_deviation'))
                    for cell in cells for seed, entry in cell['seeds'].items()]
    payload = dict(schema_version='d1_readout_depth_panel_v1', status='complete',
                   created_utc=datetime.now(timezone.utc).isoformat(),
                   n_cells=len(cells), arms=sorted({cell['arm'] for cell in cells}),
                   panels=sorted({cell['panel'] for cell in cells}),
                   strata=sorted({cell['stratum'] for cell in cells}),
                   reproduction=reproduction, counts=panel_counts(cells),
                   falsification=stratum_tally(cells),
                   seed_consistent=consistent, breadth=breadth(cells, consistent),
                   ceilings=ceilings(cells, consistent), cells=cells,
                   cell_sha256={f"{cell['arm']}/{cell['panel']}": cell['sha256'] for cell in cells},
                   summariser_sha256=digest(__file__))
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / 'readout_depth_panel.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + '\n')
    temporary.replace(destination)
    print(json.dumps(dict(
        cells=len(cells), arms=len(payload['arms']), output=str(destination),
        falsification={key: {name: bucket[name] for name in
                             ('depth_cells', 'resolved_positive', 'resolved_negative',
                              'unresolved', 'best_point')}
                       for key, bucket in payload['falsification'].items()},
        seed_consistent_blocks=len(consistent),
        breadth={reading: sum(1 for row in payload['breadth'] if row['reading'] == reading)
                 for reading in ('boundary lifted', 'already resolved', 'no seed-consistent block')},
        counts={key: {name: bucket[name] for name in
                      ('arm_seed_cells', 'depth_cells', 'resolved_positive',
                       'resolved_negative', 'unresolved', 'max_delta_spearman',
                       'max_delta_spearman_arm', 'max_delta_spearman_depth')}
                for key, bucket in payload['counts'].items()}), indent=1))


if __name__ == '__main__':
    main()
