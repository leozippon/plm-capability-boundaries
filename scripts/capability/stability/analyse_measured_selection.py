#!/usr/bin/env python3
"""Retrospective measured-label selection over the frozen stability panel.

The panel at ``results/external_baseline/*/stability_ranking_error_grid_cpu``
already retains one held-out prediction vector per arm, split seed and design.
This stage asks a different question of those same bytes: inside each wild type's
own measured candidate library, do the likelihood-augmented predictions find the
measured stabilising variants better than the evolutionary-information baseline
does? Nothing is refitted and no model is run; the held-out predictions are read
back and re-ranked.

What the run refuses to do is as important as what it reports.

* It refuses to select over the whole cohort. An explicit, reviewed benign
  background allowlist is mandatory, is validated against the frozen cohort, has
  its source and provenance files re-hashed, and is recorded as the supplied
  review protocol. No metadata field in it is read as an automated safety verdict,
  and the 89 backgrounds the review did not admit are never ranked.
* It refuses a reordered or substituted held-out archive. The join is positional
  against the exact frozen panel row order, checks ``sample_id``, family group and
  measured ddG row by row, and never builds a dict keyed by ``sample_id`` -- that
  key is ``background:position`` and repeats across substitutions at one position.
* It refuses to infer the comparison arms from a name pattern. The four roles
  ``prollama_stage1``, ``prollama_stage2``, ``protgpt2`` and ``progen2_xlarge``
  are mapped to concrete roster arms by a CLI role sheet that is hashed before any
  outcome is read.
* It refuses to publish an interval below the project's eight-group floor. Fewer
  groups is a descriptive result with the reason recorded, never a narrow interval.

Selection is always inside one wild type: the candidate library is that wild
type's own measured variants, the random expectation is that library's mean, and
the split seeds are averaged inside the wild type before the family-group
bootstrap. Primary is ``S2_M`` minus ``S2`` (likelihood over the profile-inclusive
baseline); secondary is ``S_M`` minus ``S`` (over the matched baseline). ``S_T``
is the tokenisation-descriptor control and is not used as a selector. A native
absolute log-likelihood and the delta log-likelihood differ by a per-wild-type
constant, so within a fixed wild type they induce the identical ranking and are
not two experiments. A positive measured ddG is stabilising, and every selector
ranks by descending held-out prediction.

Relative paths inside the allowlist resolve against the repository root.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.core.io import sha256_file, write_json
from src.capability.evaluation.generated_phenotype import require_fresh_out
from src.capability.interactions.pairwise_epistasis import ROSTER, SPLIT_SEEDS
from src.capability.stability.ranking_error import CONTRASTS
from src.capability.stability import measured_selection as selection
from src.capability.stability.measured_selection import (
    EXCLUDED, FRACTIONS, LABEL_ORIENTATION, PRIMARY, SCHEMA,
    SECONDARY, SELECTION_DESIGNS)

#: The completion record the frozen 33-arm panel writes, and the statuses that
#: mean it is over the full published support.
COMPLETION_SCHEMA = 'stability_ranking_error_completion_v2'
COMPLETION_STATUSES = ('complete', 'complete_with_blocked_contrasts')

#: The frozen per-background cap this stage counts. The cohort's separate
#: ``eligible_variants`` field is broader raw eligibility and is never used for a
#: selected-panel count.
FROZEN_BACKGROUND_CAP = 256

#: Artefacts the completion record binds by digest.
COMPLETION_ARTIFACTS = ('panel.json', 'channels.json', 'extraction_audit.json')
OUTPUT_FILENAME = 'measured_selection.json'


def _refuse(message: str) -> None:
    raise SystemExit(message)


def _hash_matches(path: Path, declared, *, label: str) -> dict:
    if not path.is_file():
        _refuse(f'{label}: {path} is absent')
    actual = sha256_file(path)
    if declared is None:
        _refuse(f'{label}: no declared digest is present in the frozen record')
    if actual != str(declared):
        _refuse(f'{label}: {path} hashes to {actual}, not the frozen {declared}')
    return {'path': str(path), 'sha256': actual, 'declared_sha256': str(declared),
            'verified': True}


def load_completion(panel_dir: Path, protocol: dict) -> tuple[dict, dict]:
    """Read the frozen completion record and re-check every binding it declares."""

    completion_path = panel_dir / 'completion.json'
    if not completion_path.is_file():
        _refuse(f'the panel directory {panel_dir} carries no completion.json')
    _hash_matches(completion_path, protocol['completion_sha256'], label='protocol completion')
    completion = json.loads(completion_path.read_bytes())
    if completion.get('schema') != COMPLETION_SCHEMA:
        _refuse(f"unexpected completion schema {completion.get('schema')!r}")
    if completion.get('status') not in COMPLETION_STATUSES:
        _refuse(f"the completion record is not complete: {completion.get('status')!r}")
    arms = list(completion.get('arms') or ())
    if arms != list(ROSTER):
        _refuse('the completion record must carry the exact completed 33-arm roster')
    if completion['status'] != 'complete' or completion.get('blocked_contrasts'):
        _refuse('the final panel must have all six contrasts complete')
    for field in ('per_arm_fits', 'held_out_predictions'):
        if set(completion.get(field) or {}) != set(ROSTER):
            _refuse(f'the completion record must retain all 33 arms in {field}')
    bindings = {'completion': {'path': str(completion_path),
                               'sha256': sha256_file(completion_path),
                               'schema': completion['schema'], 'status': completion['status'],
                               'arms': arms}}
    for name in COMPLETION_ARTIFACTS:
        bindings.setdefault('artifacts', {})[name] = _hash_matches(
            panel_dir / name, (completion.get('artifacts') or {}).get(name),
            label=f'completion artifact {name}')
    _hash_matches(panel_dir / 'panel.json', protocol['panel_sha256'], label='protocol panel')
    published = json.loads((panel_dir / 'panel.json').read_bytes())
    if (published.get('schema') != 'stability_ranking_error_panel_v2'
            or published.get('arms') != list(ROSTER)
            or published.get('contrast_definitions') != CONTRASTS
            or set(published.get('families') or {}) != set(CONTRASTS)
            or published.get('blocked_contrasts')):
        _refuse('the panel does not carry the authoritative 33-arm/six-contrast contract')
    for name, family in published['families'].items():
        if family.get('status') != 'complete' or family.get('columns') != list(ROSTER):
            _refuse(f'{name}: the final contrast must be complete over the exact 33-arm roster')
    # Hash every retained archive without loading or ranking the other 29 arms.
    bindings['panel_contract'] = {'arms': list(ROSTER), 'contrasts': list(CONTRASTS),
                                  'retained_oof': {}}
    for arm in ROSTER:
        _hash_matches(panel_dir / 'fits' / f'fit_{arm}.json', completion['per_arm_fits'][arm],
                      label=f'{arm} fit')
        bindings['panel_contract']['retained_oof'][arm] = _hash_matches(
            panel_dir / 'oof' / f'oof_{arm}.npz', completion['held_out_predictions'][arm],
            label=f'{arm} held-out predictions')
    return completion, bindings


def load_arm(panel_dir: Path, completion: dict, arm: str, flattened: dict,
             bindings: dict) -> dict:
    """One arm's held-out archive, hash-checked and joined positionally to the cohort."""

    fit_path = panel_dir / 'fits' / f'fit_{arm}.json'
    oof_path = panel_dir / 'oof' / f'oof_{arm}.npz'
    fit_ref = _hash_matches(fit_path, (completion.get('per_arm_fits') or {}).get(arm),
                            label=f'{arm} fit')
    oof_ref = _hash_matches(oof_path, (completion.get('held_out_predictions') or {}).get(arm),
                            label=f'{arm} held-out predictions')
    fit = json.loads(fit_path.read_bytes())
    if fit.get('arm') != arm:
        _refuse(f'{fit_path} declares arm {fit.get("arm")!r}')
    retained = fit.get('held_out_predictions') or {}
    if retained.get('file') != oof_path.name:
        _refuse(f"{arm}: the fit names {retained.get('file')!r} rather than {oof_path.name}")
    if retained.get('sha256') != oof_ref['sha256']:
        _refuse(f'{arm}: the fit records a different held-out archive digest')
    seeds = tuple(str(seed) for seed in fit.get('seeds') or ())
    if set(seeds) != {str(seed) for seed in SPLIT_SEEDS}:
        _refuse(f'{arm}: the fit carries split seeds {sorted(seeds)} rather than the frozen '
                f'{list(SPLIT_SEEDS)}')
    designs = list(retained.get('designs') or ())
    missing = [design for design in SELECTION_DESIGNS if design not in designs]
    if missing:
        _refuse(f'{arm}: the retained designs are missing {missing}')
    predictions: dict[str, dict[str, np.ndarray]] = {}
    with np.load(oof_path, allow_pickle=False) as data:
        required = {'sample_id', 'group', 'target'}
        absent = sorted(required - set(data.files))
        if absent:
            _refuse(f'{arm}: the held-out archive lacks {absent}')
        selection.verify_oof_join(flattened, data['sample_id'], data['group'], data['target'],
                                  arm=arm)
        for seed in SPLIT_SEEDS:
            predictions[str(seed)] = {}
            for design in SELECTION_DESIGNS:
                key = f'{seed}|{design}'
                if key not in data.files:
                    _refuse(f'{arm}: the held-out archive lacks {key}')
                vector = data[key].astype(np.float64)
                if not np.isfinite(vector).all():
                    _refuse(f'{arm}: {key} carries a nonfinite prediction')
                predictions[str(seed)][design] = vector
    bindings.setdefault('per_arm', {})[arm] = {
        'fit': fit_ref, 'held_out_predictions': oof_ref,
        'seeds': [str(seed) for seed in SPLIT_SEEDS], 'designs': designs,
        'matched_baseline': fit.get('matched_baseline'),
        'tokenisation_stratum': fit.get('tokenisation_stratum'),
    }
    return {'fit': fit, 'predictions': predictions}


def _column_record(name, column_values, point, draws, *, simultaneous=None) -> dict:
    """One group-level column as a point, a marginal interval and a verdict."""

    record = {'column': name, 'groups': int(len(column_values)),
              'point': float(point)}
    if draws is None:
        record.update(marginal_interval=None, verdict=None)
    else:
        interval = selection.percentile_interval(draws)
        record.update(marginal_interval=interval, verdict=selection.verdict(interval))
    if simultaneous is not None:
        record['simultaneous_interval'] = simultaneous['interval']
        record['simultaneous_verdict'] = selection.verdict(simultaneous['interval'])
    return record


def build_analysis(cohort: dict, allowlist: dict, roles: dict, arm_data: dict,
                   bindings: dict, *, draws: int, seed: int) -> dict:
    """Assemble the selection record: gains, paired contrasts, oracle and inference."""

    began = time.monotonic()
    flattened = selection.flatten_cohort(cohort)
    allowed = [entry['name'] for entry in allowlist['backgrounds']]
    allowed_set = set(allowed)
    keep = np.asarray([name in allowed_set for name in flattened['backgrounds']])
    if not keep.any():
        _refuse('no frozen cohort variant belongs to an allowlisted background')
    measured = flattened['measured'][keep]
    background = flattened['backgrounds'][keep]
    groups = flattened['groups'][keep]
    state_keys = [key for key, kept in zip(flattened['state_keys'], keep) if kept]
    wt_labels = sorted(set(background.tolist()))
    group_of = {name: group for name, group in zip(background.tolist(), groups.tolist())}
    wt_groups = [group_of[name] for name in wt_labels]
    group_labels, _ = selection.aggregate_groups(np.zeros(len(wt_labels)), wt_labels, wt_groups)

    variants_per_background = {name: int(np.sum(background == name)) for name in wt_labels}
    uniform = sorted(set(variants_per_background.values()))
    frozen_panel_count = (FROZEN_BACKGROUND_CAP * len(wt_labels)
                          if uniform == [FROZEN_BACKGROUND_CAP] else None)
    floor = selection.support_floor(len(group_labels))

    # Per-wild-type gains. Seeds are averaged inside the wild type, so the three
    # split readings never enter the bootstrap as separate units.
    columns: dict[str, np.ndarray] = {}
    per_arm: dict[str, dict] = {}
    oracle_by_fraction: dict[str, np.ndarray] = {}
    for fraction in FRACTIONS:
        ceiling = selection.oracle_ceiling(measured, background, state_keys, fraction)
        _, oracle_by_fraction[str(fraction)] = selection.aggregate_groups(
            ceiling['oracle_gain'], wt_labels, wt_groups)
    for arm in roles['arms']:
        predictions = arm_data[arm]['predictions']
        primary_by_fraction, secondary_by_fraction = {}, {}
        selectors: dict[str, dict] = {design: {} for design in SELECTION_DESIGNS}
        for fraction in FRACTIONS:
            design_gain = {}
            for design in SELECTION_DESIGNS:
                per_seed = [selection.library_gains(measured, predictions[str(s)][design][keep],
                                                    background, state_keys, fraction)['gain']
                            for s in SPLIT_SEEDS]
                design_gain[design] = selection.average_seeds(per_seed)
            primary = design_gain[PRIMARY['augmented']] - design_gain[PRIMARY['baseline']]
            secondary = design_gain[SECONDARY['augmented']] - design_gain[SECONDARY['baseline']]
            primary_by_fraction[str(fraction)] = primary
            secondary_by_fraction[str(fraction)] = secondary
            for design in SELECTION_DESIGNS:
                selectors[design][str(fraction)] = design_gain[design]
            _, columns[f'primary|{arm}|{fraction}'] = selection.aggregate_groups(
                primary, wt_labels, wt_groups)
            _, columns[f'secondary|{arm}|{fraction}'] = selection.aggregate_groups(
                secondary, wt_labels, wt_groups)
            for design in SELECTION_DESIGNS:
                _, columns[f'selector|{arm}|{design}|{fraction}'] = selection.aggregate_groups(
                    design_gain[design], wt_labels, wt_groups)
        per_arm[arm] = {'primary': primary_by_fraction, 'secondary': secondary_by_fraction,
                        'selectors': selectors}
    for fraction in FRACTIONS:
        columns[f'oracle|{fraction}'] = oracle_by_fraction[str(fraction)]

    column_names = list(columns)
    matrix = np.column_stack([columns[name] for name in column_names])
    if not np.isfinite(matrix).all():
        _refuse('a group-level selection value is not finite')

    bootstrap_record = (None if floor['degenerate']
                        else selection.joint_group_bootstrap(matrix, draws=draws, seed=seed))
    point_values = matrix.mean(axis=0)
    standard_error = np.zeros(matrix.shape[1])
    if bootstrap_record is not None:
        point_values = bootstrap_record['point']
        standard_error = bootstrap_record['standard_error']
    draws_matrix = None if bootstrap_record is None else bootstrap_record['draws_matrix']
    index = {name: position for position, name in enumerate(column_names)}

    simultaneous: dict[str, dict] = {}
    for label in ('primary', 'secondary'):
        family = [f'{label}|{arm}|{fraction}' for arm in roles['arms'] for fraction in FRACTIONS]
        if draws_matrix is None:
            simultaneous[label] = {'status': 'descriptive', 'columns': family, 'bands': None,
                                   'reason': floor['degenerate_reason']}
            continue
        positions = [index[name] for name in family]
        bands = selection.simultaneous_band(draws_matrix[:, positions],
                                            point_values[positions],
                                            standard_error[positions])
        simultaneous[label] = {
            'status': 'complete', 'columns': family, 'bands': bands,
            'intervals': {name: bands['interval'][position]
                          for position, name in enumerate(family)},
            'verdicts': {name: selection.verdict(bands['interval'][position])
                         for position, name in enumerate(family)},
            'resolved_above_zero': [name for position, name in enumerate(family)
                                    if selection.verdict(bands['interval'][position])
                                    == 'above_zero'],
        }

    def marginal(name):
        return None if draws_matrix is None else draws_matrix[:, index[name]]

    selection_record: dict[str, dict] = {}
    for arm in roles['arms']:
        primary_blocks, secondary_blocks = {}, {}
        for fraction in FRACTIONS:
            key = str(fraction)
            primary_blocks[key] = _column_record(
                f'primary|{arm}|{fraction}', columns[f'primary|{arm}|{fraction}'],
                point_values[index[f'primary|{arm}|{fraction}']],
                marginal(f'primary|{arm}|{fraction}'),
                simultaneous=({'interval': simultaneous['primary']['intervals'][
                    f'primary|{arm}|{fraction}']}
                    if simultaneous['primary']['status'] == 'complete' else None))
            secondary_blocks[key] = _column_record(
                f'secondary|{arm}|{fraction}', columns[f'secondary|{arm}|{fraction}'],
                point_values[index[f'secondary|{arm}|{fraction}']],
                marginal(f'secondary|{arm}|{fraction}'),
                simultaneous=({'interval': simultaneous['secondary']['intervals'][
                    f'secondary|{arm}|{fraction}']}
                    if simultaneous['secondary']['status'] == 'complete' else None))
        selectors = {}
        for design in SELECTION_DESIGNS:
            selectors[design] = {}
            for fraction in FRACTIONS:
                name = f'selector|{arm}|{design}|{fraction}'
                selectors[design][str(fraction)] = _column_record(
                    name, columns[name], point_values[index[name]], marginal(name))
        selection_record[arm] = {'primary': primary_blocks, 'secondary': secondary_blocks,
                                 'selectors': selectors}

    oracle_record = {}
    for fraction in FRACTIONS:
        name = f'oracle|{fraction}'
        oracle_record[str(fraction)] = _column_record(
            name, columns[name], point_values[index[name]], marginal(name))

    # Selector gain as a share of the descriptive best-k ceiling, per arm and
    # budget, computed from the group-level points and withheld when the ceiling
    # is not positive.
    share_of_oracle = {}
    for arm in roles['arms']:
        share_of_oracle[arm] = {}
        for fraction in FRACTIONS:
            ceiling_point = float(point_values[index[f'oracle|{fraction}']])
            gain_point = float(point_values[index[
                f'selector|{arm}|{PRIMARY["augmented"]}|{fraction}']])
            share_of_oracle[arm][str(fraction)] = (
                gain_point / ceiling_point if ceiling_point > 0 else None)

    withheld = None
    if floor['degenerate']:
        withheld = {
            'reason': floor['degenerate_reason'],
            'groups': len(group_labels),
            'minimum_units': floor['minimum_units'],
            'status': 'descriptive',
            'reading': ('the group-level points are reported, and no interval or simultaneous '
                        'band is published, because a percentile interval over this many '
                        'groups under-covers and can be narrower than one over hundreds of '
                        'units'),
        }

    support = {
        'cohort_backgrounds': len(cohort['backgrounds']),
        'cohort_groups': len({row['group'] for row in cohort['backgrounds']}),
        'cohort_variants': int(len(flattened['measured'])),
        'allowlisted_backgrounds': len(wt_labels),
        'allowlisted_groups': len(group_labels),
        'allowlisted_variants': int(len(measured)),
        'variants_per_background': variants_per_background,
        'frozen_background_cap': FROZEN_BACKGROUND_CAP,
        'frozen_panel_count': frozen_panel_count,
        'frozen_panel_count_basis': ('the per-background variant counts are the frozen '
                                     'cap-256 panel membership, counted from the cohort\'s '
                                     '"variants" list; the separate "eligible_variants" field '
                                     'is broader raw eligibility and is not used'),
        'fractions': list(FRACTIONS),
        'minimum_bootstrap_units': floor['minimum_units'],
        'support_floor': floor,
        'group_semantics': ('the group labels are the frozen cohort homology groups; the '
                            'allowlist review did not re-verify evolutionary independence, '
                            'and no such claim is made here'),
        'unit': 'family group',
    }

    return {
        'schema': SCHEMA,
        'status': 'complete' if not floor['degenerate'] else 'descriptive',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'question': ('inside a reviewed benign wild-type library, does the held-out likelihood '
                     'difference, added to a baseline that already carries evolutionary '
                     'information, find the measured stabilising variants better than that '
                     'baseline does'),
        'label_orientation': LABEL_ORIENTATION,
        'native_versus_delta_likelihood': {
            'statement': ('an absolute native log-likelihood and the delta log-likelihood '
                          'M_mut - M_WT differ by a per-wild-type constant, so within a fixed '
                          'wild type they induce the identical ranking'),
            'not_a_separate_experiment': True,
            'consequence': ('only one likelihood ranking is analysed per wild type; a native '
                            'versus delta contrast is not computed anywhere here'),
        },
        'contrasts': {
            'primary': {**PRIMARY, 'candidate_library': 'the same wild type, held out',
                        'selection': 'top-k by descending held-out prediction'},
            'secondary': {**SECONDARY, 'candidate_library': 'the same wild type, held out',
                          'selection': 'top-k by descending held-out prediction'},
            'excluded': EXCLUDED,
        },
        'support': support,
        'fractions': list(FRACTIONS),
        'budgets_by_background': {
            name: {str(fraction): selection.selection_count(
                      int(np.sum(background == name)), fraction)
                   for fraction in FRACTIONS}
            for name in wt_labels},
        'allowlist': allowlist,
        'roles': roles,
        'input_bindings': bindings,
        'bootstrap': {
            'status': 'withheld' if floor['degenerate'] else 'complete',
            'method': ('group bootstrap of fixed held-out group-level selection outcomes, '
                       'with one common set of group indices for every selector, arm and '
                       'budget'),
            'unit': 'family group',
            'common_draws': True,
            'selection_resampled_within_draw': False,
            'conditional_on_fitted_predictions': True,
            'n_groups': len(group_labels),
            'n_draws': None if bootstrap_record is None else draws,
            'seed': None if bootstrap_record is None else seed,
            'group_order': group_labels,
            'group_aggregation': ('each group is the mean over its wild types, equally '
                                  'weighted, so a multi-wild-type group does not dominate'),
        },
        'selection': selection_record,
        'oracle_ceiling': oracle_record,
        'share_of_oracle_gap': share_of_oracle,
        'inference': {
            'marginal': ('per-column percentile interval from the common draws; marginal for '
                         'one column, not adjusted for any family'),
            'simultaneous': simultaneous,
            'multiplicity': (
                'two declared maximum-statistic families over the same resampled family '
                'groups: primary is the S2_M-minus-S2 gain over arms and budgets, secondary '
                'is S_M-minus-S. They are separate families and neither covers a comparison '
                'between them; absolute selector gains and the oracle ceiling carry marginal '
                'intervals only'),
        },
        'method_limitations': [
            'outcomes are conditional on the already-fitted held-out predictions; selection '
            'is not redone inside a bootstrap draw and no retraining or split uncertainty is '
            'claimed',
            'the analysis is retrospective and diagnostic; it is not a causal effect and it '
            'is not a pre-registered prediction',
            'eligibility is the supplied allowlist review protocol, not an automated safety '
            'classifier; nothing here certifies that a background is safe to deploy or '
            'optimise against, and the non-admitted cohort backgrounds are never ranked',
            'the family groups are the frozen cohort homology groups; evolutionary '
            'independence was not re-verified by the allowlist review',
            'absolute selector gains and the descriptive best-k oracle are marginal readings; '
            'only the two paired contrast families carry simultaneous bands',
            'the oracle ranks by the measured endpoint it would predict, so it is the realised '
            'best-k of the frozen library and not a usable method',
            'no across-family potency claim, no design claim and no de novo validation is '
            'available from these cohorts',
            'the primary and secondary families are not commensurable with each other as a '
            'single tested difference; each is reported under its own band',
        ],
        'withheld_intervals': withheld,
        'elapsed_seconds': time.monotonic() - began,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel-dir', type=Path, required=True,
                        help='the completed stability_ranking_error_grid_cpu directory')
    parser.add_argument('--cohort', type=Path, required=True,
                        help='the frozen stability_singles_cohort_v1 cohort.json')
    parser.add_argument('--allowlist', type=Path, required=True,
                        help='the reviewed benign-background allowlist (mandatory; there is '
                             'no default and the whole cohort is never selected over)')
    parser.add_argument('--roles', type=Path, required=True,
                        help='the frozen role-to-arm sheet mapping the four intended roles '
                             'onto roster arms')
    parser.add_argument('--device', choices=('cpu',), default='cpu',
                        help='CPU arithmetic on H200 via h200/run_transfer_h200.sh')
    parser.add_argument('--out', type=Path, required=True,
                        help=f'a fresh output directory; writes {OUTPUT_FILENAME}')
    parser.add_argument('--provenance-root', type=Path, default=ROOT,
                        help='asset root for relative paths in the hash-pinned review; '
                             'on H200 this is the GPFS project root, not the code snapshot')
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261011)
    args = parser.parse_args()

    if args.draws < 1:
        _refuse('--draws must be positive')
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        _refuse('--out must be a fresh directory or an empty launcher-created directory; '
                'refusing to overwrite existing work')
    began = time.monotonic()

    roles = selection.load_roles(args.roles)
    protocol = roles['protocol']
    _hash_matches(args.allowlist, protocol['allowlist_sha256'], label='protocol allowlist')
    _hash_matches(args.cohort, protocol['cohort_sha256'], label='protocol cohort')
    cohort = json.loads(args.cohort.read_bytes())
    if cohort.get('schema') != 'stability_singles_cohort_v1':
        _refuse(f"unexpected cohort schema {cohort.get('schema')!r}")
    cohort_sha256 = sha256_file(args.cohort)
    completion, bindings = load_completion(args.panel_dir, protocol)
    panel_path = args.panel_dir / 'panel.json'
    published = json.loads(panel_path.read_bytes())
    declared_cohort = (published.get('support') or {}).get('cohort_sha256')
    if declared_cohort != cohort_sha256:
        _refuse(f'the published panel binds cohort {declared_cohort} while {args.cohort} is '
                f'{cohort_sha256}')
    endpoint = cohort.get('endpoint_sha256')
    if (published.get('support') or {}).get('endpoint_sha256') != endpoint:
        _refuse('the published panel and the cohort disagree on the endpoint digest')
    bindings['cohort'] = {'path': str(args.cohort), 'sha256': cohort_sha256,
                          'panel_declared_sha256': declared_cohort, 'endpoint_sha256': endpoint}

    flattened = selection.flatten_cohort(cohort)
    allowlist = selection.load_allowlist(args.allowlist, cohort, root=args.provenance_root)
    bindings['provenance_root'] = str(args.provenance_root.resolve())
    for arm in roles['arms']:
        if arm not in completion['arms']:
            _refuse(f'the role sheet names {arm}, which the frozen completion does not carry')
    arm_data = {}
    for arm in roles['arms']:
        arm_data[arm] = load_arm(args.panel_dir, completion, arm, flattened, bindings)
    bindings['code'] = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in (
            Path(__file__), ROOT / 'src/capability/stability/measured_selection.py')}

    record = build_analysis(cohort, allowlist, roles, arm_data, bindings, draws=args.draws,
                            seed=args.seed)
    # The cohort is restricted to allowlisted backgrounds; the completion record
    # names the frozen panel, which the support block reports against.
    record['input_bindings']['allowlist'] = {'path': allowlist['path'],
                                             'sha256': allowlist['sha256']}
    record['input_bindings']['roles'] = {'path': roles['path'], 'sha256': roles['sha256']}
    # The campaign launcher pre-creates empty cell directories. Recheck for prior
    # work after computation; a failed write never prints a terminal success.
    require_fresh_out(args.out, OUTPUT_FILENAME)
    write_json(args.out / OUTPUT_FILENAME, record)
    print(json.dumps({
        'status': record['status'],
        'allowlisted_backgrounds': record['support']['allowlisted_backgrounds'],
        'allowlisted_groups': record['support']['allowlisted_groups'],
        'frozen_panel_count': record['support']['frozen_panel_count'],
        'withheld_intervals': None if record['withheld_intervals'] is None
        else record['withheld_intervals']['reason'],
        'seconds': round(time.monotonic() - began, 1)}, indent=1), flush=True)


if __name__ == '__main__':
    main()