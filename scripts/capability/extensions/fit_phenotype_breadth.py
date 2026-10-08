#!/usr/bin/env python3
"""Fit the admitted phenotype cohorts: qualified controls, then the two endpoints.

This is the E05 panel. For each admitted cohort it qualifies the control blocks
on that cohort's own endpoint, then measures what the native likelihood adds over
the qualified set, separately for ranking and for quantitative prediction, under
one simultaneous band per cohort and endpoint.

Three orderings are enforced rather than trusted. The cohort support and its
independent groups come from the qualification stage and are re-derived and
checked against its recorded digest, so a fit cannot quietly run on a support
the admission never saw. Control qualification runs before any likelihood column
is built, so the qualified set cannot be a function of a model outcome. The
quantitative endpoint runs only where the qualification licensed it.

The evolutionary-profile control is an input, not an option. Without
``--profiles`` the profile blocks are unavailable, the result is recorded as
``local_controls_only`` and the record says in terms that an
evolutionary-statistics explanation of the increment has not been excluded. The
profile file is the one the repository's own pipeline produces:
``search_pairwise_homologs.py`` over the qualification's ``profile-catalogue.json``
and ``build_pairwise_profile_features.py`` over its ``profile-plan.json``.

``--device`` is accepted because the campaign queue injects it; the fits are CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '4')

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.extensions import phenotype_breadth as breadth  # noqa: E402
from src.capability.extensions import phenotype_cohorts as sources  # noqa: E402
from src.capability.stability.stability_gate import load_profiles  # noqa: E402

COMPLETION = 'phenotype_breadth_panel.json'


def emit(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def resources(device: str) -> dict:
    return {'executable': sys.executable, 'device': device, 'cpus': os.cpu_count(),
            'disk_free_bytes': shutil.disk_usage(ROOT).free,
            'memory': [line for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith(('MemTotal:', 'MemAvailable:'))],
            'threads': {v: os.environ[v] for v in
                        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}


def rebuild_cohort(args, key: str, admission: dict) -> tuple[list, dict[str, str]]:
    """Re-derive the admitted support and refuse any drift from the admission."""
    cohort = sources.READERS[key](args.root)
    record, rows, _ = breadth.qualify(cohort, args.runtime, floor=args.floor)
    if record['rows_sha256'] != admission['rows_sha256']:
        raise SystemExit(f'{key}: re-derived support does not match the admitted digest')
    groups = json.loads((args.qualification / 'groups' / f'{key}.json').read_text())['assignment']
    missing = sorted({row.unit for row in rows} - set(groups))
    if missing:
        raise SystemExit(f'{key}: {len(missing)} units carry no admitted group')
    return rows, groups


def fit_one(args, key: str, admission: dict, scores, profiles) -> list[dict]:
    rows, groups = rebuild_cohort(args, key, admission)
    blocks, census = breadth.prune_constant_columns(breadth.control_blocks(rows, profiles))
    emit(args.out / 'controls' / f'{key}-column-census.json', census)
    backgrounds = np.asarray([row.background for row in rows])
    group_vector = np.asarray([groups[row.unit] for row in rows])
    candidates = (breadth.CANDIDATE_BLOCKS if profiles is not None else
                  tuple(b for b in breadth.CANDIDATE_BLOCKS if not b.startswith('prof')))
    endpoints = ['ranking'] + (['quantitative'] if admission['quantitative_licensed'] else [])
    produced = []
    for endpoint in endpoints:
        labels = np.asarray([row.oriented_label() for row in rows], dtype=np.float64)
        target = (breadth.rerank(labels, backgrounds) if endpoint == 'ranking' else labels)
        qualification = breadth.qualify_controls(
            blocks, target, backgrounds, group_vector, candidates=candidates)
        qualification.update({
            'cohort': key, 'endpoint': endpoint,
            'profile_control': 'present' if profiles is not None else 'absent',
            'profile_note': (None if profiles is not None else
                             'the evolutionary-profile blocks were not supplied, so an '
                             'evolutionary-statistics explanation of any increment on this cohort '
                             'is not excluded by this fit'),
            'target': ('within-background standardized rank of the oriented label'
                       if endpoint == 'ranking' else
                       f'the oriented label in its own unit: {admission["quantitative_unit"]}'),
            'calibration': ('the predictor is fitted on the endpoint\'s own units using training '
                            'groups only, so no post-hoc rescaling is applied and no per-group '
                            'calibration is fitted on held-out rows'
                            if endpoint == 'quantitative' else
                            'rank target; no physical calibration is claimed and rank-target error '
                            'is never reported as phenotype error')})
        emit(args.out / 'controls' / f'{key}-{endpoint}.json', qualification)
        panel = breadth.fit_cohort(rows, blocks, scores, unit_groups=groups,
                                   qualified=qualification['qualified'], endpoint=endpoint)
        panel.update({
            'cohort': key, 'phenotype': admission['phenotype'],
            'endpoint_definition': admission['endpoint'],
            'status': ('admitted' if profiles is not None else 'local_controls_only'),
            'support': admission['support'],
            'independence_floor': admission['independence_floor'],
            'cohort_limitations': admission.get('cohort_limitations', []),
            'control_qualification': {k: v for k, v in qualification.items()
                                      if k != 'offers'},
            'control_column_census': census,
            'limitations': breadth.LIMITATIONS})
        emit(args.out / 'panels' / f'{key}-{endpoint}.json', panel)
        produced.append({'cohort': key, 'endpoint': endpoint, 'status': panel['status'],
                         'metric': panel['metric'], 'qualified_controls': qualification['qualified'],
                         'profile_blocks_qualified': qualification['profile_blocks_qualified'],
                         'arms': panel['arms'], 'groups': len(panel['groups']),
                         'resolved_positive': panel['resolved_positive'],
                         'resolved_negative': panel['resolved_negative'],
                         'arm_results': panel['arm_results']})
    return produced


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--qualification', type=Path, required=True,
                        help='completed qualify_phenotype_cohorts output directory')
    parser.add_argument('--scores', type=Path, action='append', required=True,
                        help='completed score_phenotype_states output directory; repeatable')
    parser.add_argument('--profiles', type=Path, default=None,
                        help='profile_features npz over the qualification profile plan')
    parser.add_argument('--cohort', action='append', default=[],
                        help='restrict to these admitted cohorts; repeatable')
    parser.add_argument('--out', type=Path, required=True,
                        help='output directory; an empty one is accepted, one holding a previous '
                             'run is refused')
    parser.add_argument('--device', default='cpu', help='accepted for queue compatibility')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--runtime', type=Path, default=None)
    parser.add_argument('--floor', type=int, default=breadth.INDEPENDENCE_FLOOR)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        out = breadth.prepare_output(args.out, COMPLETION)
    except ValueError as error:
        raise SystemExit(str(error))
    args.out = out
    args.runtime = (args.runtime or out / 'runtime').resolve()
    qualification = json.loads(
        (args.qualification / 'cohort_qualification.json').read_text(encoding='utf-8'))
    if qualification.get('status') != 'complete':
        raise SystemExit('the qualification run is not complete')
    started = time.monotonic()
    write_json(out / 'resource-start.json',
               {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'resources': resources(args.device), 'argv': sys.argv[1:]})
    try:
        scores = breadth.read_arm_scores(args.scores)
    except ValueError as error:
        raise SystemExit(str(error))
    keys = [k for k in qualification['admitted_cohorts'] if k in sources.READERS]
    if args.cohort:
        unknown = [k for k in args.cohort if k not in keys]
        if unknown:
            raise SystemExit(f'not admitted independent cohorts: {unknown}')
        keys = [k for k in keys if k in args.cohort]
    if not keys:
        raise SystemExit('no admitted independent cohort to fit')
    profiles = None
    profile_record = {'supplied': False}
    if args.profiles is not None:
        plan = json.loads((args.qualification / 'profile-plan.json').read_text(encoding='utf-8'))
        wildtypes = {row['name']: row['wildtype'] for row in plan['backgrounds']}
        profiles, meta = load_profiles(args.profiles, wildtypes)
        present = sum(1 for r in meta['backgrounds'] if r['status'] == 'present')
        profile_record = {'supplied': True, 'path': str(args.profiles),
                          'sha256': sha256_file(args.profiles),
                          'backgrounds': len(meta['backgrounds']), 'present': present,
                          'absent': len(meta['backgrounds']) - present,
                          'note': ('a background that retrieved no qualifying homolog carries a '
                                   'zero availability indicator and never a pooled profile')}
    summaries = []
    for key in keys:
        admission = json.loads(
            (args.qualification / 'cohorts' / f'{key}.json').read_text(encoding='utf-8'))
        summaries.extend(fit_one(args, key, admission, scores, profiles))
    write_json(out / 'resource-end.json',
               {'status': 'complete', 'resources': resources(args.device),
                'seconds': time.monotonic() - started})
    write_json(out / COMPLETION, {
        'schema': breadth.SCHEMA, 'stage': 'fit_phenotype_breadth', 'status': 'complete',
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'seconds': time.monotonic() - started,
        'qualification': str(args.qualification),
        'qualification_sha256': sha256_file(args.qualification / 'cohort_qualification.json'),
        'score_directories': [str(p) for p in args.scores],
        'arms': sorted(scores), 'profiles': profile_record,
        'split_seeds': list(breadth.SPLIT_SEEDS),
        'bootstrap': {'draws': breadth.BOOTSTRAP_DRAWS, 'seed': breadth.BOOTSTRAP_SEED},
        'endpoints_separated': ('ranking and quantitative prediction are fitted, aggregated and '
                               'reported independently; neither is derived from the other'),
        'results': summaries, 'limitations': breadth.LIMITATIONS,
        'resources': resources(args.device)})
    print(json.dumps({'out': str(out), 'arms': sorted(scores),
                      'results': [{k: s[k] for k in ('cohort', 'endpoint', 'status',
                                                     'resolved_positive', 'resolved_negative')}
                                  for s in summaries]}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
