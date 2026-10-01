#!/usr/bin/env python3
"""Re-estimate both model-side gains inside every declared retrieval stratum.

Two gains are read. The measured-stability gain is the paired reduction in
group-equal cycle mean squared error in squared kcal/mol from adding the two
constituent single-mutant likelihood differences to the matched baseline, over
held family groups. The mutation-ranking gain is the paired within-assay Spearman
increment from adding a frozen representation to a mutation-local profile
control, over held wild-type families.

Neither gain is refitted here. Each stratum's estimate is the same paired
statistic over the same fitted held-out predictions, taken over the subset of
resampling units the stratum holds, with the same 2,000-draw bootstrap at the
same seed and the same estimator machinery the original studies used. Every
retained unit therefore keeps its internal weighting, its fold and its label
budget, and the whole-support estimate is the unit-count-weighted mean of its
strata, which is computed and reported as a residual rather than assumed.

A stratum below the package's eight-unit percentile floor carries no verdict at
all and is reported as unresolved thin support, never as a stratum a gain failed
on.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.interactions.pairwise_epistasis import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, SPLIT_SEEDS, interval
from src.capability.context.profile_increment import summarize
from src.capability.context.retrieval_strata import (
    SOURCE_BANDS, SOURCE_EXCLUSION, SOURCE_TOKEN, STRATIFICATIONS, bands_of,
    decomposition_residual, direction, kish_row_counts, kish_subunits, resolution)

SCHEMA = 'retrieval_memorization_gate_report_v1'
REPORT_BASENAME = 'retrieval_strata_report.json'

#: The stability contrast this gate is about, and two companions that put it in
#: context: what the sequence/profile control itself removes on the same stratum,
#: and the likelihood interaction over the same matched baseline.
STABILITY_PRIMARY = 'C_G_T_M1|C_G_T'
STABILITY_CONTRASTS = (STABILITY_PRIMARY, 'C|ADDITIVE_NULL', 'C_G_T+M|C_G_T')

#: The mutation-ranking contrast this gate is about, and the companions the
#: crossed-control study reports beside it. ``C_P`` is the binding control by
#: that study's own verdict, so its representation increment is primary.
READOUT_PRIMARY = 'increment_R_C_P'
READOUT_CONTRASTS = ('increment_R_C_P', 'increment_R_C_L_P', 'increment_R_C_L_P_T',
                     'increment_R_after_M_C_P', 'increment_M_C_P')


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b''):
            sha.update(chunk)
    return sha.hexdigest()


def width(record: dict) -> float | None:
    bounds = record.get('interval')
    return None if bounds is None else float(bounds[1] - bounds[0])


def band_counts(units: list[str], per_group: dict) -> dict:
    """Support counts of one stability stratum, from the realised per-group table."""

    cycles_per_pair = [count for unit in units
                       for count in per_group[unit]['cycles_per_site_pair']]
    return {
        'groups': len(units),
        'site_pairs': sum(per_group[unit]['site_pairs'] for unit in units),
        'cycles': sum(per_group[unit]['cycles'] for unit in units),
        'kish_effective_site_pairs_estimator_weights': kish_subunits(
            {unit: per_group[unit]['site_pairs'] for unit in units}),
        'kish_effective_site_pairs_cycle_counts': kish_row_counts(cycles_per_pair),
    }


def stability_band(values: dict[str, float], units: list[str], per_group: dict,
                   draws: int) -> dict:
    """One stability stratum: the same group-bootstrap interval over its groups."""

    selected = [values[unit] for unit in units]
    record = interval(selected, draws=draws, seed=BOOTSTRAP_SEED) if selected else {
        'point': None, 'interval': None, 'groups': 0, 'excludes_zero': None}
    return {**record, 'interval_width_kcal2': width(record), **band_counts(units, per_group),
            'resolution': resolution(len(units), record.get('excludes_zero')),
            'direction': direction(record.get('interval'))}


def stability_view(record: dict, contrast: str, seed: int, assignment: dict, per_group: dict,
                   draws: int, *, filtered: bool) -> dict:
    """One arm, contrast and seed on one of the two declared stability supports."""

    suffix = '_indel_filtered' if filtered else ''
    increment = record['seeds'][str(seed)]['increments'][contrast]
    values = increment[f'per_group_mse_reduction_kcal2{suffix}']
    available = set(per_group)
    if set(values) != available:
        raise SystemExit(f'{record["arm"]} seed {seed}: per-group values cover {len(values)} '
                         f'groups against {len(available)} in the support table')
    if not available <= set(assignment):
        raise SystemExit(f'{record["arm"]}: the support holds a group the declaration lacks')
    full = increment[f'mse_reduction_kcal2{suffix}']
    entry = {'full_support': {**full, 'interval_width_kcal2': width(full),
                              **band_counts(sorted(available), per_group),
                              'resolution': resolution(len(available),
                                                        full.get('excludes_zero')),
                              'direction': direction(full.get('interval'))},
             'strata': {}}
    for name in STRATIFICATIONS:
        bands = {band: [unit for unit in units if unit in available]
                 for band, units in bands_of(assignment, name).items()}
        per_band = {band: stability_band(values, units, per_group, draws)
                    for band, units in bands.items()}
        entry['strata'][name] = {
            'bands': per_band,
            'decomposition_residual_kcal2': decomposition_residual(
                full['point'], {band: value['point'] for band, value in per_band.items()},
                {band: len(units) for band, units in bands.items()}),
        }
    return entry


def stability_gains(refits: dict, declaration: dict, draws: int) -> dict:
    """Every stability contrast, arm, seed and stratum, plus the decomposition residual."""

    stability = declaration['stability']
    assignment = stability['assignment']
    report: dict = {}
    for arm, record in sorted(refits.items()):
        unfiltered = record['per_group_support']
        filtered = record['indel_exclusion']['per_group_support']
        declared = stability['band_support']
        for name in STRATIFICATIONS:
            for band, units in bands_of(assignment, name).items():
                realised = band_counts(units, unfiltered)
                for key, value in declared[name][band].items():
                    if key in realised and abs(realised[key] - value) > 1e-9:
                        raise SystemExit(f'{arm}: {name}/{band} {key} is {realised[key]} against '
                                         f'{value} in the declaration')
        arm_entry: dict = {
            'reference_agreement': record['reference_agreement'],
            'tokenisation_stratum': record['tokenisation_stratum'],
            'qualified_batch_size': record['qualified_batch_size'],
            'batch_size_caveat': record['batch_size_caveat'],
            'arm_selection': record['arm_selection'],
            'indel_exclusion': {key: record['indel_exclusion'][key] for key in
                                ('order', 'scope', 'excluded_cycles', 'groups_lost_entirely',
                                 'retained_support')},
            'contrasts': {}, 'contrasts_indel_filtered': {},
        }
        for contrast in STABILITY_CONTRASTS:
            arm_entry['contrasts'][contrast] = {
                str(seed): stability_view(record, contrast, seed, assignment, unfiltered, draws,
                                          filtered=False) for seed in SPLIT_SEEDS}
            arm_entry['contrasts_indel_filtered'][contrast] = {
                str(seed): stability_view(record, contrast, seed, assignment, filtered, draws,
                                          filtered=True) for seed in SPLIT_SEEDS}
        report[arm] = arm_entry
    return report


def readout_band(rows: list[dict], key: str, support: dict, draws: int) -> dict:
    """One mutation-ranking stratum: the same cluster bootstrap over its clusters."""

    record = summarize(rows, key, bootstrap=draws, seed=BOOTSTRAP_SEED) if rows else {
        'point': None, 'interval': None, 'n_units': 0, 'excludes_zero': None}
    units = int(record.get('n_units') or 0)
    return {
        **record, 'interval_width': width(record), 'direction': direction(record.get('interval')),
        'clusters': units, 'assays': support['assays'], 'variants': support['variants'],
        'kish_effective_assays_estimator_weights':
            support['kish_effective_assays_estimator_weights'],
        'kish_effective_assays_variant_counts':
            support['kish_effective_assays_variant_counts'],
        'resolution': resolution(units, record.get('excludes_zero')),
    }


def readout_support(declaration: dict) -> tuple[tuple[str, ...], dict[str, str], dict]:
    """The declared anchor support: its assays, its cluster map and its counts."""

    anchor = declaration['readout_anchor']
    cluster_of = {assay: cluster for cluster, unit in anchor['units'].items()
                  for assay in unit['assays']}
    return tuple(sorted(cluster_of)), cluster_of, anchor['support']


def readout_gains(cells: dict, declaration: dict, draws: int) -> dict:
    """Every mutation-ranking contrast, arm, seed and stratum, re-aggregated exactly.

    A cell report is admitted only when its fitted support is the declared anchor
    exactly: the same cohort bytes, the same assays, the same cluster map, the same
    assay, family and variant counts, and the same per-assay variant counts as
    every other admitted cell. The declaration fixes that support once; a panel
    cell is checked against it rather than against the cells that established it.
    """

    anchor = declaration['readout_anchor']
    assignment = anchor['assignment']
    mixed = set(anchor['source_provenance_evidence']['mixed_source_clusters'])
    declared_assays, cluster_of, counts = readout_support(declaration)
    cohort_sha = anchor['sources']['cohort']['sha256']
    variants: dict[str, int] | None = None
    report: dict = {}
    for (arm, seed), cell in sorted(cells.items()):
        rows = cell['assays']
        if cell['cohort_sha256'] != cohort_sha:
            raise SystemExit(f'{arm} seed {seed}: fitted on different cohort bytes')
        if tuple(sorted(row['assay'] for row in rows)) != declared_assays:
            raise SystemExit(f'{arm} seed {seed}: the fitted assays are not the declared anchor')
        if {row['assay']: str(row['cluster']) for row in rows} != cluster_of:
            raise SystemExit(f'{arm} seed {seed}: the fitted cluster map is not the declared one')
        realised = {row['assay']: int(row['n_variants']) for row in rows}
        if variants is None:
            variants = realised
        elif realised != variants:
            raise SystemExit(f'{arm} seed {seed}: per-assay variant counts differ across cells')
        if (cell['n_assays'], cell['n_families'], cell['n_variants']) != (
                counts['assays'], counts['clusters'], counts['variants']):
            raise SystemExit(f'{arm} seed {seed}: the fitted counts are not the declared anchor')
        by_cluster: dict[str, list[dict]] = {}
        for row in rows:
            by_cluster.setdefault(str(row['cluster']), []).append(row)
        if set(by_cluster) != set(assignment):
            raise SystemExit(f'{arm} seed {seed}: the fitted clusters are not the declared units')
        entry = report.setdefault(arm, {'contrasts': {}})
        for key in READOUT_CONTRASTS:
            published = cell['summaries'][key]
            reaggregated = summarize(rows, key, bootstrap=draws, seed=BOOTSTRAP_SEED)
            if reaggregated['point'] != published['point']:
                raise SystemExit(f'{arm} seed {seed} {key}: re-aggregation gives '
                                 f'{reaggregated["point"]!r} against the published '
                                 f'{published["point"]!r}')
            support = anchor['support']
            record = {'full_support': {
                **reaggregated, 'interval_width': width(reaggregated),
                'clusters': support['clusters'], 'assays': support['assays'],
                'variants': support['variants'],
                'kish_effective_assays_estimator_weights':
                    support['kish_effective_assays_estimator_weights'],
                'kish_effective_assays_variant_counts':
                    support['kish_effective_assays_variant_counts'],
                'resolution': resolution(support['clusters'],
                                         reaggregated['excludes_zero']),
                'direction': direction(reaggregated.get('interval'))},
                'published_point': published['point'], 'strata': {}}
            for name in STRATIFICATIONS:
                bands = bands_of(assignment, name)
                per_band = {}
                for band, clusters in bands.items():
                    selected = [row for cluster in clusters for row in by_cluster[cluster]]
                    per_band[band] = readout_band(
                        selected, key, anchor['band_support'][name][band], draws)
                record['strata'][name] = {
                    'bands': per_band,
                    'decomposition_residual': decomposition_residual(
                        reaggregated['point'],
                        {band: value['point'] for band, value in per_band.items()},
                        {band: len(clusters) for band, clusters in bands.items()}),
                }
            unshared = [row for cluster, members in by_cluster.items() for row in members
                        if (assignment[cluster]['source_provenance'] == SOURCE_BANDS[0]
                            or cluster in mixed) and SOURCE_TOKEN not in row['assay']]
            sensitivity = summarize(unshared, key, bootstrap=draws, seed=BOOTSTRAP_SEED)
            clusters = {str(row['cluster']) for row in unshared}
            record['source_exclusion_sensitivity'] = {
                'rule': SOURCE_EXCLUSION['sensitivity'],
                **sensitivity, 'interval_width': width(sensitivity),
                'clusters': len(clusters), 'assays': len(unshared),
                'kish_effective_assays_estimator_weights': kish_subunits(
                    {cluster: sum(1 for row in unshared if str(row['cluster']) == cluster)
                     for cluster in clusters}),
                'kish_effective_assays_variant_counts': kish_row_counts(
                    [int(row['n_variants']) for row in unshared]),
                'resolution': resolution(len(clusters), sensitivity['excludes_zero']),
                'direction': direction(sensitivity.get('interval')),
            }
            entry['contrasts'].setdefault(key, {})[str(seed)] = record
    return report


def fold_identity(cells: dict) -> dict:
    """Held-family membership has to be one map per split seed across every arm."""

    per_seed: dict[str, dict] = {}
    for (arm, seed), cell in sorted(cells.items()):
        membership = [sorted(map(str, fold['held_families'])) for fold in cell['folds']]
        realised = hashlib.sha256(json.dumps(membership, sort_keys=True).encode()).hexdigest()
        held = per_seed.setdefault(str(seed), {'fold_identity_sha256': realised, 'arms': [],
                                               'held_family_counts': [len(f) for f in membership]})
        if held['fold_identity_sha256'] != realised:
            raise SystemExit(f'{arm} at seed {seed} does not share the held-family map')
        held['arms'].append(arm)
    return per_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strata-declaration', type=Path, required=True)
    parser.add_argument('--expect-declaration-sha256', required=True)
    parser.add_argument('--pairwise-refits', type=Path, action='append', required=True,
                        help='directory of stability refit records; repeatable, one per wave')
    parser.add_argument('--crossed-reports', type=Path, action='append', required=True,
                        help='directory of crossed-control cell reports; repeatable')
    parser.add_argument('--out', type=Path, required=True,
                        help='output directory; the gate report is written into it by name')
    parser.add_argument('--bootstrap', type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument('--cross-dispatch-tolerance', type=float, default=1e-9,
                        help='largest per-group squared-error difference admitted between two '
                             'dispatches of the same stability fit')
    parser.add_argument('--cell-agreement-tolerance', type=float, default=1e-5,
                        help='largest summary-point difference admitted between two dispatches '
                             'of the same crossed-control cell')
    args = parser.parse_args()

    declaration = json.loads(args.strata_declaration.read_text())
    if declaration['declaration_sha256'] != args.expect_declaration_sha256:
        raise SystemExit('the strata declaration digest does not match the declared value')

    refits: dict[str, dict] = {}
    cross_dispatch: dict[str, list[dict]] = {}
    for directory in args.pairwise_refits:
        for path in sorted(directory.glob('refit_*.json')):
            record = json.loads(path.read_text())
            if record['inputs']['strata_declaration_sha256'] != args.expect_declaration_sha256:
                raise SystemExit(f'{path} was not fitted against this declaration')
            arm = record['arm']
            if 'indel_exclusion' in record:
                if arm in refits:
                    raise SystemExit(f'{arm} carries two filtered refit records')
                refits[arm] = record
            else:
                cross_dispatch.setdefault(arm, []).append(
                    {'path': str(path), 'record': record})
    if not refits:
        raise SystemExit('no stability refit record carries the declared indel exclusion')
    reproducibility = {}
    for arm, earlier in sorted(cross_dispatch.items()):
        if arm not in refits:
            raise SystemExit(f'{arm} has a pre-exclusion record and no filtered one')
        worst = 0.0
        for entry in earlier:
            for seed in SPLIT_SEEDS:
                left = entry['record']['seeds'][str(seed)]['per_group_mse_kcal2']
                right = refits[arm]['seeds'][str(seed)]['per_group_mse_kcal2']
                if set(left) != set(right) or sorted(left['C_G_T']) != sorted(right['C_G_T']):
                    raise SystemExit(f'{arm} seed {seed}: two dispatches disagree on the designs '
                                     'or the groups they cover')
                worst = max(worst, max(abs(left[design][group] - right[design][group])
                                       for design in left for group in left[design]))
        reproducibility[arm] = {
            'earlier_dispatches': [entry['path'] for entry in earlier],
            'max_absolute_per_group_mse_difference_kcal2': worst,
            'tolerance_kcal2': args.cross_dispatch_tolerance}
        if worst > args.cross_dispatch_tolerance:
            raise SystemExit(f'{arm}: two dispatches of the same fit differ by {worst:.3e} '
                             f'kcal2/mol2 per group, above the '
                             f'{args.cross_dispatch_tolerance:.1e} tolerance')

    cells: dict[tuple[str, int], dict] = {}
    duplicates: dict[str, float] = {}
    for directory in args.crossed_reports:
        for path in sorted(directory.glob('*/crossed_controls_*_fold*.json')):
            cell = json.loads(path.read_text())
            key = (cell['arm'], int(cell['fold_seed']))
            if key not in cells:
                cells[key] = cell
                continue
            # Directories are given in preference order, so the first reading of an
            # arm and seed stands. A second reading of the same cell is a separate
            # dispatch of the same fitted design on the same rows: it must agree, and
            # how closely it agrees is measured rather than assumed.
            worst = max(
                abs(cells[key]['summaries'][name]['point'] - record['point'])
                for name, record in cell['summaries'].items()
                if record['point'] is not None
                and cells[key]['summaries'].get(name, {}).get('point') is not None)
            duplicates[f'{key[0]}/{key[1]}'] = worst
            if worst > args.cell_agreement_tolerance:
                raise SystemExit(f'{key} appears twice and its summary points differ by '
                                 f'{worst:.3e}, above the '
                                 f'{args.cell_agreement_tolerance:.1e} tolerance')
    if not cells:
        raise SystemExit('no crossed-control cell report under the given directories')

    report = {
        'schema': SCHEMA,
        'inputs': {
            'strata_declaration': {'path': str(args.strata_declaration),
                                   'declaration_sha256': args.expect_declaration_sha256,
                                   'file_sha256': digest(args.strata_declaration)},
            'stability_refits': {arm: refits[arm]['inputs'] for arm in sorted(refits)},
            'stability_cross_dispatch_reproducibility': reproducibility,
            'crossed_control_cells': sorted(f'{arm}/{seed}' for arm, seed in cells),
            'crossed_control_cell_directories': [str(path) for path in args.crossed_reports],
            'crossed_control_duplicate_agreement': {
                'tolerance': args.cell_agreement_tolerance,
                'max_summary_point_difference': (max(duplicates.values()) if duplicates
                                                 else None),
                'cells': duplicates},
        },
        'estimators': {
            'stability': ('paired reduction in group-equal cycle mean squared error, squared '
                          'kcal/mol; site pairs weighted equally inside an equally weighted held '
                          'group; 2,000-draw group-bootstrap percentile interval at seed '
                          f'{BOOTSTRAP_SEED}; the resampling unit is the held group and the '
                          'weighted unit inside it is the site pair, never the cycle'),
            'readout': ('paired within-assay Spearman increment averaged inside a wild-type family '
                        'at 50% identity and then equally across families; 2,000-draw family '
                        f'bootstrap percentile interval at seed {BOOTSTRAP_SEED}'),
            'stratum_rule': ('a stratum estimate is the same statistic over the units the stratum '
                             'holds; nothing is refitted, rescaled or retuned, and no unit\'s '
                             'internal weighting changes'),
            'stability_supports': ('two supports are reported for every stability stratum. The '
                                   'unfiltered one is the admitted cohort exactly as fitted, and '
                                   'it is the support the reproduction gate runs on. The '
                                   'indel-filtered one drops the declared insertion and deletion '
                                   'cycles from the evaluation of the same fitted predictions, '
                                   'which removes one group entirely, and is reported beside it '
                                   'rather than in place of it'),
        },
        'primary_contrasts': {'stability': STABILITY_PRIMARY, 'readout': READOUT_PRIMARY},
        'code_sha256': {str(path.relative_to(ROOT)): digest(path) for path in (
            Path(__file__), ROOT / 'src/capability/context/retrieval_strata.py',
            ROOT / 'src/capability/context/profile_increment.py', ROOT / 'src/capability/context/profiles.py',
            ROOT / 'src/capability/interactions/pairwise_epistasis.py')},
        'readout_fold_identity': fold_identity(cells),
        'stability': stability_gains(refits, declaration, args.bootstrap),
        'readout_anchor': readout_gains(cells, declaration, args.bootstrap),
    }
    reported = primary = 0
    for cohort, keys, primary_contrast in (
            ('stability', ('contrasts', 'contrasts_indel_filtered'), STABILITY_PRIMARY),
            ('readout_anchor', ('contrasts',), READOUT_PRIMARY)):
        for arm in report[cohort].values():
            for key in keys:
                for name, contrast in arm[key].items():
                    for entry in contrast.values():
                        count = 1 + sum(len(stratum['bands'])
                                        for stratum in entry['strata'].values())
                        count += 1 if 'source_exclusion_sensitivity' in entry else 0
                        reported += count
                        primary += count if name == primary_contrast else 0
    report['multiplicity'] = {
        'reported_intervals': reported,
        'primary_contrast_intervals': primary,
        'stability_arms': len(report['stability']),
        'readout_arms': len(report['readout_anchor']),
        'stratifications': len(STRATIFICATIONS),
        'split_seeds': len(SPLIT_SEEDS),
        'statement': ('every interval here is a pointwise 95% percentile interval conditional on '
                      'the fitted cross-validation predictions, unadjusted for the arms, the '
                      'stratifications, the bands, the contrasts, the two stability supports or '
                      'the three split seeds. At this count of reported intervals a handful of '
                      'bands clearing zero in one band and one seed is expected under no effect, '
                      'so a stratum verdict rests on holding at all three seeds and on the '
                      'whole-support finding it decomposes, not on an isolated interval'),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / REPORT_BASENAME
    destination.write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    residuals = [abs(entry['strata'][name]['decomposition_residual_kcal2'])
                 for arm in report['stability'].values()
                 for key in ('contrasts', 'contrasts_indel_filtered')
                 for contrast in arm[key].values()
                 for entry in contrast.values() for name in entry['strata']]
    residuals += [abs(entry['strata'][name]['decomposition_residual'])
                  for arm in report['readout_anchor'].values()
                  for contrast in arm['contrasts'].values()
                  for entry in contrast.values() for name in entry['strata']]
    print(json.dumps({'out': str(destination),
                      'stability_arms': sorted(report['stability']),
                      'readout_arms': sorted(report['readout_anchor']),
                      'max_absolute_decomposition_residual': max(residuals)}, indent=1))


if __name__ == '__main__':
    main()
