#!/usr/bin/env python3
"""Qualify candidate phenotype cohorts before any model cost is incurred.

This stage decides, for every candidate independent phenotype cohort and for the
matched multi-phenotype support, what may be measured at all. It reads prepared
identity artefacts and public labels only: no checkpoint, no GPU, no network and
no fit. Its three products are the admission record of every cohort, the state
plan of the cohorts that were admitted, and the query catalogue the committed
evolutionary-profile pipeline needs.

The order matters and is enforced by construction. Support screens and the
independent-group measurement come first; the independence floor verdict comes
next; only then is a state plan written, and only for cohorts that cleared it. A
cohort below the floor, or whose endpoint direction or construct identity is not
established, is written out as ``refused`` with its blockers named. Nothing about
a refused cohort reaches the scoring plan.

``--device`` is accepted because the campaign queue injects it; this stage is
CPU-only and uses it for nothing but the receipt.
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '2')

from src.capability.core.io import write_json  # noqa: E402
from src.capability.extensions import matched_phenotypes as matched  # noqa: E402
from src.capability.extensions import phenotype_breadth as breadth  # noqa: E402
from src.capability.extensions import phenotype_cohorts as sources  # noqa: E402

COMPLETION = 'cohort_qualification.json'
MATCHED_KEY = 'matched_proteingym'
SUBDIRECTORIES = ('cohorts', 'rows', 'groups')


def emit(path: Path, value) -> None:
    """``write_json`` writes atomically but does not create directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def resources(device: str) -> dict:
    return {'executable': sys.executable, 'device': device, 'cpus': os.cpu_count(),
            'disk_free_bytes': shutil.disk_usage(ROOT).free,
            'memory': [line for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith(('MemTotal:', 'MemAvailable:'))],
            'threads': {v: os.environ[v] for v in
                        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
            'execution': 'single-process CPU identity, homology and support qualification'}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True,
                        help='output directory; it may already exist while it is empty, and a '
                             'directory holding a previous run is refused')
    parser.add_argument('--device', default='cpu', help='accepted for queue compatibility')
    parser.add_argument('--root', type=Path, default=ROOT,
                        help='project root holding data/ and results/')
    parser.add_argument('--runtime', type=Path, default=None,
                        help='git-ignored directory for the exact-alignment backend build')
    parser.add_argument('--floor', type=int, default=breadth.INDEPENDENCE_FLOOR,
                        help='independent-group floor; the default is the project floor')
    parser.add_argument('--cohort', action='append', default=[],
                        help='restrict to these candidate cohorts; repeatable')
    parser.add_argument('--proteingym-dir', type=Path, default=None,
                        help='directory of processed ProteinGym substitution tables')
    parser.add_argument('--reference', type=Path, default=None,
                        help='release reference CSV naming every assay and its target sequence')
    parser.add_argument('--strata-metadata', type=Path, default=None,
                        help='assay-to-phenotype-class and frozen-cluster table')
    parser.add_argument('--skip-matched', action='store_true',
                        help='qualify the independent cohorts only')
    return parser.parse_args(argv)


def qualify_candidates(args, out: Path) -> tuple[dict, dict, dict]:
    """Admission record, retained rows and group assignment of every candidate."""
    records, retained, assignments = {}, {}, {}
    keys = args.cohort or list(sources.READERS)
    unknown = [key for key in keys if key not in sources.READERS]
    if unknown:
        raise SystemExit(f'unknown candidate cohorts: {unknown}')
    for key in keys:
        cohort = sources.READERS[key](args.root)
        record, rows, grouping = breadth.qualify(cohort, args.runtime, floor=args.floor)
        records[key] = record
        retained[key] = rows
        assignments[key] = grouping.get('assignment', {})
        emit(out / 'cohorts' / f'{key}.json', record)
        with (out / 'rows' / f'{key}.jsonl').open('w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps({**row.identity(),
                                         'label': row.label, 'direction': row.direction,
                                         'oriented_label': row.oriented_label()},
                                        sort_keys=True) + '\n')
        emit(out / 'groups' / f'{key}.json',
                   {'rule': grouping.get('rule'), 'assignment': grouping.get('assignment', {}),
                    'edges': grouping.get('edges', [])})
    return records, retained, assignments


def qualify_matched(args, out: Path) -> tuple[dict, dict, dict]:
    """The matched multi-phenotype support, with its per-pair counts."""
    missing = [name for name, value in (('--proteingym-dir', args.proteingym_dir),
                                        ('--reference', args.reference),
                                        ('--strata-metadata', args.strata_metadata))
               if value is None]
    if missing:
        raise SystemExit('the matched support requires ' + ', '.join(missing))
    reference = matched.read_reference(args.reference)
    strata = matched.read_strata_metadata(args.strata_metadata)
    pairs, census = matched.discover_pairs(args.proteingym_dir, reference, strata)
    counts = matched.pair_counts(pairs)
    rows: list[breadth.PhenotypeRow] = []
    per_pair = []
    for pair in pairs:
        rows_a, rows_b, draw = matched.matched_rows(pair, args.proteingym_dir, cohort=MATCHED_KEY)
        per_pair.append({**pair.record(), **draw})
        rows.extend(rows_a)
        rows.extend(rows_b)
    for row in rows:
        breadth.validate_row(row)
    units = {row.unit: row.wildtype for row in rows}
    grouping = breadth.independence_groups(units, args.runtime)
    floor_record = breadth.bootstrap_unit_floor(grouping['groups'], minimum_units=args.floor)
    record = {
        'cohort': MATCHED_KEY, 'phenotype': 'matched multi-phenotype support',
        'endpoint': matched.ORIENTATION, 'source': 'processed ProteinGym substitution tables',
        'status': 'admitted' if not floor_record['degenerate'] else 'refused',
        'evidence_level': ('matched multi-phenotype support on anchor proteins; not an '
                           'independent task cohort'),
        'blockers': ([] if not floor_record['degenerate'] else
                     [f"{grouping['groups']} independent groups is below the {args.floor}-group floor"]),
        'support': {'rows': len(rows), 'assay_pairs': len(pairs),
                    'proteins': counts['pooled']['proteins'],
                    'independent_groups': grouping['groups'],
                    'matched_substitution_rows': counts['pooled']['matched_substitution_rows']},
        'independence': {k: v for k, v in grouping.items() if k != 'assignment'},
        'independence_floor': floor_record,
        'pair_counts': counts, 'discovery_census': census, 'pairs': per_pair,
        'ranking_licensed': not floor_record['degenerate'],
        'quantitative_licensed': False,
        'quantitative_refusal': ('two phenotypes are measured in different units, so no squared '
                                 'error may be pooled across them'),
        'limitations': matched.LIMITATIONS,
    }
    emit(out / 'cohorts' / f'{MATCHED_KEY}.json', record)
    emit(out / 'groups' / f'{MATCHED_KEY}.json',
               {'rule': grouping['rule'], 'assignment': grouping['assignment'],
                'edges': grouping['edges']})
    with (out / 'rows' / f'{MATCHED_KEY}.jsonl').open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps({**row.identity(), 'label': row.label,
                                     'direction': row.direction}, sort_keys=True) + '\n')
    return record, {MATCHED_KEY: rows}, {MATCHED_KEY: grouping['assignment']}


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        out = breadth.prepare_output(args.out, COMPLETION)
    except ValueError as error:
        raise SystemExit(str(error))
    for name in SUBDIRECTORIES:
        (out / name).mkdir()
    args.runtime = (args.runtime or out / 'runtime').resolve()
    started = time.monotonic()
    emit(out / 'resource-start.json',
               {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'resources': resources(args.device), 'argv': sys.argv[1:]})
    records, retained, assignments = qualify_candidates(args, out)
    if not args.skip_matched:
        record, rows, assignment = qualify_matched(args, out)
        records[MATCHED_KEY] = record
        retained.update(rows)
        assignments.update(assignment)
    emit(out / 'declared-refusals.json', list(sources.DECLARED_REFUSALS))
    admitted = {key: retained[key] for key, record in records.items()
                if record['status'] == 'admitted' and retained.get(key)}
    plan = breadth.state_plan(admitted)
    emit(out / 'scoring-plan.json', plan)
    catalogue, profile_plan = breadth.profile_catalogue(admitted)
    emit(out / 'profile-catalogue.json', catalogue)
    emit(out / 'profile-plan.json', profile_plan)
    completion = {
        'schema': breadth.SCHEMA, 'stage': 'qualify_phenotype_cohorts', 'status': 'complete',
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'seconds': time.monotonic() - started,
        'independence_floor': args.floor,
        'candidates': [{'cohort': key, 'status': records[key]['status'],
                        'independent_groups': records[key]['support']['independent_groups'],
                        'rows': records[key]['support']['rows'],
                        'ranking_licensed': records[key]['ranking_licensed'],
                        'quantitative_licensed': records[key]['quantitative_licensed'],
                        'blockers': records[key]['blockers']}
                       for key in sorted(records)],
        'declared_refusals': [r['cohort'] for r in sources.DECLARED_REFUSALS],
        'admitted_cohorts': sorted(admitted),
        'states_planned': len(plan['states']),
        'states_sha256': plan['states_sha256'],
        'profile_queries': len(catalogue),
        'resources': resources(args.device),
        'limitations': breadth.LIMITATIONS,
        'next': ('score the planned states with score_phenotype_states.py, build the '
                 'evolutionary-profile control with search_pairwise_homologs.py and '
                 'build_pairwise_profile_features.py over profile-catalogue.json and '
                 'profile-plan.json, then fit with fit_phenotype_breadth.py'),
    }
    emit(out / 'resource-end.json',
               {'status': 'complete', 'resources': resources(args.device),
                'seconds': time.monotonic() - started})
    emit(out / COMPLETION, completion)
    print(json.dumps({'out': str(out), 'admitted': sorted(admitted),
                      'candidates': {k: records[k]['status'] for k in sorted(records)},
                      'states': len(plan['states'])}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
