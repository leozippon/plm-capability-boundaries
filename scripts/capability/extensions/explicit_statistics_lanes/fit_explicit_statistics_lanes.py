#!/usr/bin/env python3
"""Fit one lane's explicit-statistics hierarchy and write its declared families.

The declaration lives in :mod:`src.capability.extensions.explicit_statistics_lanes`:
the role vocabulary, the five-rung hierarchy, the two directions, the matched-support
rule, the per-lane families and the four finding states. This script is the执行 half.
It owns four things and no estimand:

**The wave gate.** The coupling and GEMME features arrive per cohort as their own
wave. A rung that needs a block whose table has not landed is not fitted and is named
as pending; the nuisance rung needs neither block, so the full-support sensitivity can
be fitted and reported while the wave is still running. Nothing is imputed and no
partial table is read as a support: the declaration's reader refuses an incomplete
upstream document outright.

**Resume.** One fitted cell is one file under ``cells/``, bound to the digests of
everything it was fitted from. A rerun re-reads a cell whose digests still match and
refits nothing; a changed input invalidates exactly the cells it reaches.

**The pilot.** Before any new panel is written, the lane's archived control and joint
arms are refitted on the full support with the archived qualified control set, and the
per-background values and the selected penalties must reproduce the published panel.
This is the only use the archived fits can be put to: the complete case drops the rows
whose reference carries no coupling or GEMME value, so an archived fit sits on a
different support and is *not* reusable inside the complete-case family, however
exactly its columns match. It is a reproduction gate, not saved compute.

**The adapter boundary.** Each lane keeps its own fitter, its own panel assembly and
its own archive. An adapter supplies the lane's rows, blocks, target, groups, a call
into that fitter and the published values its pilot reproduces; everything above is
shared and lives here. The adapters themselves live in
:mod:`explicit_statistics_lane_adapters` beside this script. A lane without one is
refused with the list of what it must supply, because a plausible-looking stub that
fits the wrong rows is worse than an absence.

**The qualified control set is read per arm.** Two lanes offered a tokenisation
nuisance to each arm separately, so an arm's control set is a property of that arm's
own archive and ``qualified(stratum, arm)`` is asked once per cell. The published
values a pilot reproduces are likewise per arm, and the aggregation that turns a
lane's own records into per-group values is the adapter's, not this script's: a
comparison made one step short of the published aggregation passes trivially
wherever a group holds a single measured background.

Modes::

    plan    --lane phenotype ...            # contract, fit plan, wave status, support
    pilot   --lane phenotype ...            # archived reproduction gate, full support
    fit     --lane phenotype ...            # fit the cells the plan names, resumably
    panel   --lane phenotype ...            # assemble the cached cells and infer
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_variable, '1')

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.capability.extensions.explicit_statistics_lanes.explicit_statistics_lane_adapters \
    import ADAPTERS  # noqa: E402
from src.capability.core.io import sha256_file  # noqa: E402
from src.capability.extensions import bidirectional  # noqa: E402
from src.capability.extensions import explicit_statistics_lanes as lanes  # noqa: E402

#: Stable schema labels of what this script writes.
CELL_SCHEMA = 'explicit_statistics_lane_cell'
PLAN_SCHEMA = 'explicit_statistics_lane_plan'
PILOT_SCHEMA = 'explicit_statistics_lane_pilot'

#: Largest admissible disagreement between a refitted archived arm and the published
#: per-group value, at the published aggregation. The refit is the same arithmetic on
#: the same rows in the same order, so anything above this is a different fit, not a
#: rounding difference.
PILOT_TOLERANCE = 1e-9

#: What the reproduction gate measures, named so it cannot be mistaken for inference.
REPRODUCTION_CRITERION = (
    'the maximum absolute difference between a refitted archived arm and the published '
    'per-group value at the published aggregation, against a stated tolerance, together '
    'with the number of arms that met it. It is not an interval and no band is formed on '
    'this path: a reproduction that succeeds is a matrix of near-exact zeros, which has no '
    'spread, and a band over it would assert the narrowest possible statement for the least '
    'informative possible input rather than measure anything')

#: Prefix of an input a cell records but is not bound to.
#:
#: A cached cell is refused when the inputs that can change its numbers have moved, and
#: reused when they have not. Those are not the same set as "everything worth recording":
#: a declaration's prose, its limitations and an instrument bound are worth recording
#: beside a fit and cannot alter it, so binding a cell to them makes writing one cost a
#: refit. An adapter marks such an input with this prefix; it is written into the cell
#: and left out of the digest. Nothing that decides a number may carry it.
RECORDED_PREFIX = 'note:'


def binding_inputs(inputs: dict) -> dict:
    """The inputs a fit is bound to: everything not marked as recorded-only."""

    return {key: value for key, value in inputs.items()
            if not key.startswith(RECORDED_PREFIX)}


def cell_identity(lane_name: str, support: str, inputs: dict) -> str:
    """The digest one cell is bound to: its binding inputs and its declared contract.

    The declared half is conditioned on the support, because no full-support rung carries
    an added block and so no full-support cell can be moved by the added-block
    vocabulary, its rendering or its admission rules. The declaration asserts that
    property at import rather than this script assuming it.
    """

    return digest_text(json.dumps(
        {'inputs': binding_inputs(inputs),
         'declared': lanes.fit_identity(lane_name, support)},
        sort_keys=True))


def write(path: Path, value) -> None:
    """Atomic JSON with no NaN, so a partial record is never left behind."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, sort_keys=False,
                                    default=bidirectional.json_default) + '\n')
    temporary.replace(path)


def digest_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def resolve(path):
    """Every path is read against the repository root, so a record names one thing."""

    if path is None:
        return None
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


# --------------------------------------------------------------------------- #
# The wave: the two added-block tables of one cohort.
# --------------------------------------------------------------------------- #

def read_wave(lane_name: str, *, couplings: Path | None, gemme: Path | None) -> dict:
    """Which added blocks have landed for this lane, and why the others have not.

    A role this lane declares non-identifiable is not part of its wave at all. A role
    whose table is absent, incomplete or unreadable is pending with the reader's own
    reason; a table that reads is landed. The reason is recorded verbatim, because
    "pending" and "refused" are different findings and the difference is the message.
    """

    declared = lanes.lane(lane_name)
    paths = {'couplings': couplings, 'gemme': gemme}
    tables, status = {}, {}
    for role in ('couplings', 'gemme'):
        if role not in declared.required_blocks():
            status[role] = {'state': 'not_required',
                            'reason': declared.non_identifiable.get(role, 'not part of this lane')}
            continue
        if paths[role] is None:
            status[role] = {'state': 'pending', 'reason': f'no --{role} table was named'}
            continue
        try:
            tables[role] = lanes.read_scores(paths[role], role)
        except (OSError, ValueError) as error:
            status[role] = {'state': 'pending', 'path': str(paths[role]), 'reason': str(error)}
            continue
        status[role] = {'state': 'landed', 'path': str(paths[role]),
                        'sha256': sha256_file(paths[role]),
                        'rows': len(tables[role]['samples'])}
    landed = sorted(tables)
    return {'required': list(declared.required_blocks()), 'landed': landed,
            'pending': sorted(role for role, entry in status.items()
                              if entry['state'] == 'pending'),
            'complete': set(landed) == set(declared.required_blocks()),
            'status': status, 'tables': tables}


def runnable_supports(wave: dict) -> dict[str, dict]:
    """Which of the plan's two supports can be fitted under this wave.

    The full-support sensitivity carries the nuisance rung only and needs no added
    block, so it runs while the wave is still out. The complete case needs every
    required block, because every control set in it is fitted on one row set.
    """

    return {
        'full': {'runnable': True,
                 'reason': 'the nuisance rung needs no added block'},
        'complete_case': {'runnable': bool(wave['complete']),
                          'reason': ('every required block has landed' if wave['complete'] else
                                     'pending blocks ' + ', '.join(wave['pending']) +
                                     '; a control set is never fitted on rows another control '
                                     'set does not have')},
    }


def adapter(args):
    """This lane's adapter, or an explicit refusal naming what one must supply."""

    if args.lane not in ADAPTERS:
        raise SystemExit(
            f'{args.lane}: no adapter. A lane adapter supplies, for each of its strata: the '
            'admitted rows on the lane\'s own support, the archived qualified control set '
            '(never re-qualified), the control blocks, the target, the group vector, the '
            'per-arm model block, a call into the lane\'s own fitter returning per-group values '
            'for the three arms, and the published per-background values its pilot reproduces. '
            f'Implemented adapters: {sorted(ADAPTERS)}. A stub that fits the wrong rows is worse '
            'than this refusal')
    return ADAPTERS[args.lane](args)


# --------------------------------------------------------------------------- #
# The added blocks, joined onto the lane's own row keys.
# --------------------------------------------------------------------------- #

def added_values(wave: dict, support: dict) -> dict:
    """The retained value of each added block, and what the export covers at all.

    The retained identities come from :func:`~.explicit_statistics_lanes.complete_case`,
    so a refused reference is absent here rather than carrying a substitute. The covered
    set is every identity the export names, which is what lets a lane distinguish "this
    row left the complete case" from "the export never had this row".
    """

    retained = set(support['sample_ids'])
    values, covered = {}, set()
    for role, table in wave['tables'].items():
        column = lanes.BLOCK_SOURCE[role]['column']
        identities = {str(row['sample_id']) for row in table['samples']}
        covered = identities if not covered else covered & identities
        values[role] = {str(row['sample_id']): float(row[column])
                        for row in table['samples'] if str(row['sample_id']) in retained}
        if len(values[role]) != len(retained):
            raise SystemExit(f'{role}: the table carries {len(values[role])} of the '
                             f'{len(retained)} retained rows')
    return {'values': values, 'covered': covered,
            'digest': digest_text('\n'.join(sorted(retained)))}


# --------------------------------------------------------------------------- #
# Cells, and the resume rule.
# --------------------------------------------------------------------------- #

def cell_path(out: Path, stratum: str, channel: str, support: str, rung: str, arm: str) -> Path:
    return out / 'cells' / stratum / channel / support / rung / f'{arm}.json'


def fit_cell(lane_name, instance, state, *, stratum, channel, support, rung, arm, qualified,
             inputs_sha256, out, refit=False) -> dict:
    """Fit one cell, or re-read the cached one whose inputs have not changed."""

    path = cell_path(out, stratum, channel, support, rung, arm)
    if path.is_file() and not refit:
        cached = json.loads(path.read_text(encoding='utf-8'))
        if cached.get('schema') != CELL_SCHEMA:
            raise SystemExit(f'{path}: schema {cached.get("schema")!r} is not {CELL_SCHEMA!r}')
        if cached.get('inputs_sha256') == inputs_sha256 and \
                cached.get('support_sha256') == state['support_sha256']:
            cached['reused'] = True
            return cached
        raise SystemExit(
            f'{path} was fitted from different inputs ({cached.get("inputs_sha256", "")[:12]} '
            f'against {inputs_sha256[:12]}). Fit into a fresh output directory rather than '
            'mixing two input sets in one family, or pass --refit to replace this cell')
    designs = lanes.arm_designs(lane_name, rung, qualified, channel=channel)
    started = time.monotonic()
    fitted = instance.fit(state, arm, designs)
    record = {'schema': CELL_SCHEMA, 'lane': lane_name, 'stratum': stratum, 'channel': channel,
              'support': support, 'rung': rung, 'arm': arm,
              'designs': {name: list(columns) for name, columns in designs.items()},
              'qualified': list(qualified), 'groups': state['group_order'],
              'support_sha256': state['support_sha256'], 'inputs_sha256': inputs_sha256,
              'seconds': time.monotonic() - started, 'reused': False, **fitted}
    write(path, record)
    return record


def cached_cells(out: Path, stratum: str, channel: str, support: str, rungs, arms) -> tuple:
    """Every cell of one family that is on disk, and the columns each rung was fitted on.

    The columns travel with the cells because the panel labels every number it reports
    with the rung it is an increment over, and a rung is only identified by its columns.
    They are read from the cells themselves rather than re-derived, so the label names
    what was actually fitted.
    """

    cells: dict = {}
    columns: dict = {}
    for rung in rungs:
        for arm in arms:
            path = cell_path(out, stratum, channel, support, rung, arm)
            if not path.is_file():
                continue
            record = json.loads(path.read_text(encoding='utf-8'))
            cells[rung, arm] = {**record['values'], 'support_sha256': record['support_sha256']}
            columns.setdefault(rung, {})[arm] = list(record['designs'][lanes.CONTROL_ARM])
    return cells, columns


# --------------------------------------------------------------------------- #
# Modes.
# --------------------------------------------------------------------------- #

def export_refusals(args):
    """The cohort export's own refusals document, or an explicit refusal to proceed.

    The export is the authority on the references it refused to score, so the document
    is required rather than optional: without it a reference refused for both blocks at
    once would have to be accepted on a block table's word alone, and a lane that lost a
    reference nobody enumerates would be refused for the wrong reason.
    """

    path = args.export / lanes.REFUSALS_FILE
    if not path.is_file():
        raise SystemExit(
            f'{path} is absent. The cohort export enumerates the references it refused to '
            'score, and the complete case reads its losses from there rather than taking a '
            "block table's word for a refusal it did not make")
    return lanes.read_refusals(path)


def plan(args, instance, wave) -> dict:
    """The contract, the fit plan, the wave status and, where possible, the support."""

    declared = lanes.lane(args.lane)
    supports = runnable_supports(wave)
    record = {'schema': PLAN_SCHEMA, 'lane': args.lane,
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'contract': lanes.contract(args.lane),
              'wave': {role: entry for role, entry in wave['status'].items()},
              'supports': supports, 'arms_scored': instance.arms(),
              'inputs_sha256': instance.inputs(), 'strata': {}}
    if sorted(instance.arms()) != sorted(declared.arms):
        record['arm_drift'] = {
            'declared': list(declared.arms), 'scored': sorted(instance.arms()),
            'note': ('the declaration states the realised arm set; a changed set is a changed '
                     'family and the declaration is amended before the panel is written')}
    for stratum in instance.strata():
        qualified = instance.qualified(stratum)
        entry = {'qualified_control_set': list(qualified),
                 'fit_plan': lanes.fit_plan(args.lane, qualified),
                 'families': {channel: lanes.declared_family(args.lane, channel)
                              for channel in declared.channels},
                 'fragile': lanes.stratum(args.lane, stratum).fragile}
        # An arm whose own archive qualified a further nuisance carries it, so the
        # realised control set is stated per arm beside the plan rather than left to
        # be inferred from the fit plan's arm-invariant columns.
        per_arm = {arm: list(instance.qualified(stratum, arm)) for arm in instance.arms()}
        differing = {arm: columns for arm, columns in per_arm.items()
                     if columns != list(qualified)}
        entry['control_set_per_arm'] = per_arm
        if differing:
            entry['arm_conditional_control_set'] = {
                'arms': sorted(differing), 'count': len(differing),
                'note': ('these arms qualified a further nuisance in their own archive and carry '
                         'it in their control set; the forward increment of such an arm is "the '
                         'model beyond this arm\'s own controls"')}
        record['strata'][stratum] = entry
    if wave['complete']:
        rows = lanes.read_rows(args.export / lanes.ROWS_FILE)
        record['complete_case'] = lanes.complete_case(args.lane, rows, wave['tables'],
                                                      refusals=export_refusals(args))
    else:
        record['complete_case'] = {
            'status': 'pending', 'pending_blocks': wave['pending'],
            'reason': supports['complete_case']['reason']}
    support_path = args.export / lanes.SUPPORT_FILE
    if support_path.is_file():
        record['gemme_risk_band'] = lanes.gemme_risk_band(json.loads(support_path.read_text()))
    return record


def pilot(args, instance) -> dict:
    """Refit the archived arms on the full support and require the published values.

    The archived fit is the only check available on the whole fitting path before any
    new feature exists: identical rows, identical columns, identical folds. It is not
    reused inside the complete-case family, because that family drops rows the archive
    kept.

    The comparison is made against the per-group values the lane's adapter reads back
    *at the published aggregation*, never one step short of it. The difference
    matters: on a cohort whose every group holds one measured background, comparing
    per-background values against group-aggregated ones agrees by construction and
    the gate would pass without having tested anything. Each stratum's record
    therefore states whether its support exercises the aggregation chain or only the
    fit and the per-group statistic.

    **The criterion is a maximum absolute difference against a stated tolerance, and
    nothing else** (:data:`REPRODUCTION_CRITERION`). It is not an interval and no band
    is formed anywhere on this path. A successful reproduction is a matrix of
    near-exact zeros, which carries no spread at all, so a band over it would not be a
    narrow measurement of anything -- it would be the narrowest possible statement
    asserted for the least informative possible input. The band machinery is reached
    only from :func:`panels`, over per-group arm performances that do vary.
    """

    declared = lanes.lane(args.lane)
    inputs = instance.inputs()
    strata = args.stratum or instance.strata()
    report = {'schema': PILOT_SCHEMA, 'lane': args.lane, 'status': 'passed',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'tolerance': PILOT_TOLERANCE, 'criterion': REPRODUCTION_CRITERION,
              'inputs_sha256': inputs, 'strata': {}}
    for stratum in strata:
        archived = instance.archived(stratum)
        state = instance.state(stratum, support='full', added={})
        arms = [arm for arm in (args.arm or declared.arms) if arm in archived['arms']]
        if not arms:
            raise SystemExit(f'{stratum}: the published panel covers none of the named arms')
        reproduced, worst = {}, 0.
        for arm in arms:
            qualified = instance.qualified(stratum, arm)
            published_columns = list(archived['qualified'][arm])
            if published_columns != list(qualified):
                raise SystemExit(f'{stratum}/{arm}: the published panel was fitted over '
                                 f'{published_columns}, not {list(qualified)}')
            # The archived qualified set is not the archived rung's realised design. Where a
            # lane declares a role its own forward qualification dropped, the rung carries
            # that block anyway -- which is the hierarchy's whole point -- and the refit then
            # sits on columns the published fit never had. Reproduction is not merely unlikely
            # there, it is undefined, so it is refused here with that reason and never
            # attempted: a declared structural impossibility is not a numerical failure and
            # must not be reported as one.
            realised = list(lanes.arm_designs(args.lane, lanes.ARCHIVED_RUNG, qualified,
                                              channel='likelihood')[lanes.CONTROL_ARM])
            if realised != published_columns:
                raise SystemExit(
                    f'{stratum}: this stratum has no reachable reproduction gate. The '
                    f'{lanes.ARCHIVED_RUNG!r} rung of {arm} is realised over {realised} and the '
                    f'published panel was fitted over {published_columns}; the declaration '
                    f'carries {sorted(set(realised) - set(published_columns))} at this rung '
                    'although this stratum\'s archive did not qualify it. A fit over other '
                    'columns is a different fit, so there is nothing here for a reproduction to '
                    'reproduce and none was attempted -- this is the declaration\'s own '
                    'structure, not a difference between the published numbers and these. The '
                    'gate is not widened to admit it and this stratum is not panelled; the '
                    'lane\'s notes record why')
            cell = fit_cell(args.lane, instance, state, stratum=stratum, channel='likelihood',
                            support='full', rung=lanes.ARCHIVED_RUNG, arm=arm,
                            qualified=qualified,
                            inputs_sha256=cell_identity(args.lane, 'full', inputs),
                            out=args.out, refit=args.refit)
            differences = {}
            for name in (lanes.CONTROL_ARM, lanes.JOINT_ARM):
                expected = np.asarray(archived['values'][arm][name], dtype=float)
                values = np.asarray(cell['values'][name], dtype=float)
                if values.shape != expected.shape or not np.isfinite(expected).all():
                    raise SystemExit(
                        f'{stratum}/{arm}/{name}: the published reading is {expected.shape} and '
                        f'the refit is {values.shape} over {len(state["group_order"])} groups')
                difference = float(np.max(np.abs(values - expected)))
                differences[name] = difference
                worst = max(worst, difference)
            penalties = archived['penalties'][arm]
            if sorted(penalties) != sorted(cell['alphas']):
                raise SystemExit(f'{stratum}/{arm}: the published penalties cover split seeds '
                                 f'{sorted(penalties)} and the refit {sorted(cell["alphas"])}')
            matched = all(list(cell['alphas'][seed][name]) == list(penalties[seed][name])
                          for seed in penalties
                          for name in (lanes.CONTROL_ARM, lanes.JOINT_ARM))
            reproduced[arm] = {'per_group_max_absolute_difference': differences,
                               'penalties_reproduced': matched, 'alphas': cell['alphas'],
                               'control_set': published_columns}
            if not matched or max(differences.values()) > PILOT_TOLERANCE:
                report['status'] = 'failed'
        within = [arm for arm, entry in reproduced.items()
                  if entry['penalties_reproduced']
                  and max(entry['per_group_max_absolute_difference'].values()) <= PILOT_TOLERANCE]
        report['strata'][stratum] = {
            'published': archived['path'], 'published_sha256': archived['sha256'],
            'criterion': REPRODUCTION_CRITERION, 'tolerance': PILOT_TOLERANCE,
            'arms_within_tolerance': len(within),
            'arms_outside_tolerance': sorted(set(reproduced) - set(within)),
            'arms': arms, 'arms_published': len(archived['arms']),
            'arms_pending': sorted(set(archived['arms']) - set(arms)),
            'groups': len(state['group_order']),
            'support_sha256': state['support_sha256'],
            'aggregation': archived['aggregation'],
            'support_shape': archived.get('support_shape'),
            'worst_per_group_difference': worst, 'arms_reproduced': reproduced}
    if report['status'] != 'passed':
        report['reason'] = ('a refitted archived arm did not reproduce the published per-group '
                            f'values within {PILOT_TOLERANCE:g} or chose a different penalty; the '
                            'fitting path is not the published one and no panel may be written')
    return report


def fit(args, instance, wave) -> dict:
    """Fit every cell the plan names on every support this wave allows."""

    declared = lanes.lane(args.lane)
    inputs = instance.inputs()
    supports = runnable_supports(wave)
    report = {'lane': args.lane, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'inputs_sha256': inputs, 'supports': supports, 'fitted': [], 'skipped': []}
    added: dict = {}
    if wave['complete']:
        rows = lanes.read_rows(args.export / lanes.ROWS_FILE)
        support_record = lanes.complete_case(args.lane, rows, wave['tables'],
                                             refusals=export_refusals(args))
        added = added_values(wave, support_record)
        write(args.out / 'support.json', support_record)
    for stratum in (args.stratum or instance.strata()):
        qualified = instance.qualified(stratum)
        for entry in lanes.fit_plan(args.lane, qualified):
            support = entry['support']
            if args.support and support not in args.support:
                continue
            if entry['role'] == 'pilot':
                # The pilot rung is fitted and compared by the pilot mode, which owns the
                # archived comparison; it is not a family and is not fitted here.
                continue
            if not supports[support]['runnable']:
                report['skipped'].append({'stratum': stratum, 'support': support,
                                          'rung': entry['rung'],
                                          'reason': supports[support]['reason']})
                continue
            state = instance.state(stratum, support=support,
                                   added=added if support == 'complete_case' else {})
            for arm in (args.arm or declared.arms):
                cell = fit_cell(args.lane, instance, state, stratum=stratum,
                                channel=entry['channel'], support=support, rung=entry['rung'],
                                arm=arm, qualified=instance.qualified(stratum, arm),
                                inputs_sha256=cell_identity(args.lane, support, inputs),
                                out=args.out, refit=args.refit)
                report['fitted'].append({'stratum': stratum, 'support': support,
                                         'rung': entry['rung'], 'arm': arm,
                                         'reused': cell['reused'], 'seconds': cell['seconds']})
                print(json.dumps({'event': 'cell', 'stratum': stratum, 'support': support,
                                  'rung': entry['rung'], 'arm': arm, 'reused': cell['reused'],
                                  'seconds': round(cell['seconds'], 2)}), flush=True)
    report['cells_fitted'] = sum(1 for cell in report['fitted'] if not cell['reused'])
    report['cells_reused'] = sum(1 for cell in report['fitted'] if cell['reused'])
    return report


def require_pilot(args, instance, stratum: str, arms) -> dict:
    """The recorded pilot of one stratum, or a refusal naming what must be run.

    The gate is a precondition of the panel rather than an option beside it. The
    archived control set is the only thing in this hierarchy that has a published
    counterpart, so a family written without a passing reproduction of it on every arm
    it covers rests on a fitting path nothing has ever checked. Reusing an archived
    *fit* is separately forbidden -- the complete case sits on other rows -- which is
    exactly why the reproduction is the whole of the archive's value here.
    """

    path = args.out / 'pilot.json'
    if not path.is_file():
        raise SystemExit(f'{args.lane}/{stratum}: no pilot at {path}. The archived control set is '
                         'refitted on the full support and required to reproduce the published '
                         'per-group values before any family is written; run the pilot mode first')
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('schema') != PILOT_SCHEMA or record.get('lane') != args.lane:
        raise SystemExit(f'{path} is not a {PILOT_SCHEMA} record of {args.lane}')
    if record.get('tolerance') != PILOT_TOLERANCE:
        raise SystemExit(f'{path} was run at tolerance {record.get("tolerance")!r}, not the '
                         f'declared {PILOT_TOLERANCE:g}; the gate tolerance is never loosened')
    recorded = record.get('inputs_sha256')
    if not isinstance(recorded, dict) or \
            binding_inputs(recorded) != binding_inputs(instance.inputs()):
        raise SystemExit(f'{path} was run against different inputs than this panel reads; rerun '
                         'the pilot rather than writing a family it did not cover')
    if record.get('status') != 'passed':
        raise SystemExit(f'{path} records status {record.get("status")!r}: '
                         f'{record.get("reason", "the reproduction did not pass")}')
    entry = (record.get('strata') or {}).get(stratum)
    if not entry:
        raise SystemExit(f'{path} covers {sorted(record.get("strata") or {})} and not {stratum!r}; '
                         'each stratum is its own support and its own reproduction')
    missing = [arm for arm in arms if arm not in (entry.get('arms_reproduced') or {})]
    if missing:
        raise SystemExit(
            f'{args.lane}/{stratum}: the pilot reproduced {len(entry["arms_reproduced"])} arms and '
            f'this family covers {len(arms)}; {len(missing)} are unpiloted, first {missing[0]!r}. '
            'An arm enters a family only once its own archived control set has been reproduced')
    return {'path': str(path.relative_to(args.out)),
            'worst_per_group_difference': entry['worst_per_group_difference'],
            'tolerance': PILOT_TOLERANCE, 'arms': len(entry['arms_reproduced']),
            'aggregation': entry['aggregation'], 'support_shape': entry.get('support_shape')}


def panels(args, instance, wave) -> dict:
    """Assemble whichever declared families are complete on disk, and name the rest."""

    declared = lanes.lane(args.lane)
    written, pending = {}, []
    for stratum in (args.stratum or instance.strata()):
        gate = require_pilot(args, instance, stratum, declared.arms)
        for channel in declared.channels:
            for support, role in (('complete_case', 'primary'), ('full', 'sensitivity')):
                if args.support and support not in args.support:
                    continue
                rungs = declared.rungs if role == 'primary' else ('nuisance',)
                cells, columns = cached_cells(args.out, stratum, channel, support, rungs,
                                              declared.arms)
                expected = len(rungs) * len(declared.arms)
                if len(cells) != expected:
                    pending.append({'stratum': stratum, 'channel': channel, 'support': support,
                                    'role': role, 'cells': len(cells), 'expected': expected,
                                    'reason': ('not every cell of this family is on disk; fit '
                                               'the rest or name why it cannot be fitted')})
                    continue
                digests = {cell['support_sha256'] for cell in cells.values()}
                if len(digests) != 1:
                    raise SystemExit(f'{stratum}/{channel}/{support}: cached cells carry '
                                     f'{len(digests)} different supports')
                groups = json.loads(cell_path(args.out, stratum, channel, support, rungs[0],
                                              declared.arms[0]).read_text())['groups']
                record = lanes.panel(args.lane, stratum, channel, cells, groups=groups,
                                     support_sha256=digests.pop(), role=role,
                                     control_columns=columns)
                record['wave'] = wave['status']
                record['pilot'] = gate
                path = args.out / 'panels' / f'{stratum}-{channel}-{support}.json'
                write(path, record)
                written[str(path.relative_to(args.out))] = {
                    'status': record['status'], 'family': record['family']['size'],
                    'states': record['state_counts']}
    return {'lane': args.lane, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'written': written, 'pending': pending,
            'reading': lanes.FORWARD_BOUND_READING}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['plan', 'pilot', 'fit', 'panel'])
    parser.add_argument('--lane', required=True, choices=sorted(lanes.LANES))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--export', type=Path,
                        help="the cohort's explicit-statistics export directory")
    parser.add_argument('--couplings', type=Path, help='the coupling score table of this cohort')
    parser.add_argument('--gemme', type=Path, help='the GEMME score table of this cohort')
    parser.add_argument('--qualification', type=Path, help='the frozen qualification run')
    parser.add_argument('--published', type=Path, help='the published panel run the pilot reads')
    parser.add_argument('--profiles', type=Path, help='the profile feature store')
    parser.add_argument('--scores', action='append', default=[],
                        help='a completed model-score or extraction directory; repeatable')
    parser.add_argument('--cache', type=Path,
                        help="a lane's prepared per-fold nuisance and target cache, where its "
                             'own fitter builds one')
    parser.add_argument('--stratum', action='append', default=[])
    parser.add_argument('--arm', action='append', default=[])
    parser.add_argument('--support', action='append', default=[],
                        choices=['complete_case', 'full'])
    parser.add_argument('--refit', action='store_true',
                        help='replace a cached cell instead of refusing a changed input')
    args = parser.parse_args(argv)
    for name in ('out', 'export', 'couplings', 'gemme', 'qualification', 'published',
                 'profiles', 'cache'):
        setattr(args, name, resolve(getattr(args, name)))
    args.scores = [resolve(path) for path in args.scores]
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.mode in ('plan', 'fit') and args.export is None:
        raise SystemExit(f'{args.mode} needs --export')
    instance = adapter(args)
    wave = read_wave(args.lane, couplings=args.couplings, gemme=args.gemme)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == 'plan':
        record = plan(args, instance, wave)
        write(args.out / 'plan.json', record)
    elif args.mode == 'pilot':
        record = pilot(args, instance)
        write(args.out / 'pilot.json', record)
        if record['status'] != 'passed':
            print(json.dumps({'pilot': record['status'], 'reason': record['reason']}), flush=True)
            return 1
    elif args.mode == 'fit':
        record = fit(args, instance, wave)
        write(args.out / 'fit.json', record)
    else:
        record = panels(args, instance, wave)
        write(args.out / 'panels.json', record)
    print(json.dumps({'mode': args.mode, 'lane': args.lane,
                      'out': str(args.out),
                      'summary': {key: record[key] for key in
                                  ('status', 'cells_fitted', 'cells_reused', 'written', 'pending',
                                   'supports')
                                  if key in record}}, default=str), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
