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
its own archive. An adapter supplies the lane's rows, blocks, target, groups and a
call into that fitter; everything above is shared. Only the added-phenotype lane has
an adapter here. A lane without one is refused with the list of what it must supply,
because a plausible-looking stub that fits the wrong rows is worse than an absence.

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

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.baselines.cohorts import SAMPLE_SEPARATOR  # noqa: E402
from src.capability.core.io import sha256_file  # noqa: E402
from src.capability.extensions import bidirectional  # noqa: E402
from src.capability.extensions import explicit_statistics_lanes as lanes  # noqa: E402
from src.capability.extensions import phenotype_breadth as breadth  # noqa: E402

#: Stable schema labels of what this script writes.
CELL_SCHEMA = 'explicit_statistics_lane_cell'
PLAN_SCHEMA = 'explicit_statistics_lane_plan'
PILOT_SCHEMA = 'explicit_statistics_lane_pilot'

#: Largest admissible disagreement between a refitted archived arm and the published
#: per-background value. The refit is the same arithmetic on the same rows in the same
#: order, so anything above this is a different fit, not a rounding difference.
PILOT_TOLERANCE = 1e-9


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


# --------------------------------------------------------------------------- #
# The added-phenotype adapter.
# --------------------------------------------------------------------------- #

class PhenotypeAdapter:
    """The added-phenotype lane: four cohort/endpoint strata on disjoint rows.

    The admitted rows are read back from the qualification run's own row files and the
    wild-type sequences from its profile catalogue, each verified against the digest
    the row carries, so the support is the admitted one and is never re-derived into a
    different one. The group assignment, the qualified control set and the published
    per-background values are read from the frozen runs and never re-qualified.
    """

    lane = 'phenotype'

    #: Stratum name -> the cohort and endpoint it is, and the metric that reads it.
    STRATA = {
        'activity_venus-ranking': ('activity_venus', 'ranking'),
        'binding_skempi-ranking': ('binding_skempi', 'ranking'),
        'binding_skempi-quantitative': ('binding_skempi', 'quantitative'),
        'cellular_fitness_sge-ranking': ('cellular_fitness_sge', 'ranking'),
    }

    def __init__(self, args):
        self.qualification = Path(args.qualification)
        self.published = Path(args.published)
        self.profiles_path = Path(args.profiles)
        self.score_directories = [Path(path) for path in args.scores]
        self._state: dict = {}
        self._scores: dict | None = None

    # ------------------------------------------------------------------ inputs

    def inputs(self) -> dict[str, str]:
        record = {str(path.relative_to(ROOT)): sha256_file(path)
                  for path in sorted(self.qualification.glob('rows/*.jsonl'))}
        for path in (self.qualification / 'profile-catalogue.json', self.profiles_path):
            record[str(path.relative_to(ROOT))] = sha256_file(path)
        for path in sorted(self.qualification.glob('groups/*.json')):
            record[str(path.relative_to(ROOT))] = sha256_file(path)
        for path in sorted(self.published.glob('controls/*.json')):
            record[str(path.relative_to(ROOT))] = sha256_file(path)
        record['code:phenotype_breadth'] = sha256_file(Path(breadth.__file__))
        record['code:explicit_statistics_lanes'] = sha256_file(Path(lanes.__file__))
        record['code:runner'] = sha256_file(Path(__file__))
        return record

    def scores(self) -> dict[str, dict[str, float]]:
        if self._scores is None:
            self._scores = breadth.read_arm_scores(self.score_directories)
        return self._scores

    def arms(self) -> list[str]:
        return sorted(self.scores())

    def strata(self) -> list[str]:
        return list(self.STRATA)

    def qualified(self, stratum: str) -> tuple[str, ...]:
        cohort, endpoint = self.STRATA[stratum]
        path = self.published / 'controls' / f'{cohort}-{endpoint}.json'
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('cohort') != cohort or record.get('endpoint') != endpoint:
            raise SystemExit(f'{path} does not record {cohort}/{endpoint}')
        return tuple(record['qualified'])

    def _rows(self, cohort: str) -> list[breadth.PhenotypeRow]:
        """The admitted rows of one cohort, rebuilt and digest-verified."""

        catalogue = json.loads(
            (self.qualification / 'profile-catalogue.json').read_text(encoding='utf-8'))
        sequences = {entry['WT_name']: entry['sequence'] for entry in catalogue}
        rows = []
        path = self.qualification / 'rows' / f'{cohort}.jsonl'
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            name = f"{record['cohort']}:{record['wildtype_sha256'][:16]}"
            wildtype = sequences.get(name)
            if wildtype is None:
                raise SystemExit(f"{record['row_id']}: {name} is not in the profile catalogue, so "
                                 'its wild-type sequence cannot be recovered')
            if breadth.sha_text(wildtype) != record['wildtype_sha256']:
                raise SystemExit(f"{record['row_id']}: the catalogue sequence is not the one the "
                                 'admitted row was measured on')
            index = int(record['position']) - 1
            if wildtype[index] != record['wt_aa']:
                raise SystemExit(f"{record['row_id']}: the wild type carries "
                                 f'{wildtype[index]} at {record["position"]}')
            mutant = wildtype[:index] + record['mt_aa'] + wildtype[index + 1:]
            if breadth.sha_text(mutant) != record['mutant_sha256']:
                raise SystemExit(f"{record['row_id']}: the rebuilt mutant is not the measured one")
            rows.append(breadth.PhenotypeRow(
                cohort=record['cohort'], row_id=record['row_id'], background=record['background'],
                unit=record['unit'], wildtype=wildtype, mutant_sequence=mutant,
                position=int(record['position']), wt_aa=record['wt_aa'], mt_aa=record['mt_aa'],
                label=float(record['label']), direction=int(record['direction']),
                label_unit=record.get('label_unit'), source_sha256=record['source_sha256']))
        if not rows:
            raise SystemExit(f'{path} holds no admitted row')
        return rows

    def sample_key(self, row: breadth.PhenotypeRow) -> str:
        """The export's own ``sample_id`` for one admitted row.

        The cohort export names a row by its background and its mutation string, and it
        carries one row per distinct pair, so several admitted rows of one pair read the
        same added-block value. That is a join, not a duplication: the block is a
        property of the (background, mutation) pair.
        """

        return f'{row.background}{SAMPLE_SEPARATOR}{row.wt_aa}{row.position}{row.mt_aa}'

    # ------------------------------------------------------------------- state

    def state(self, stratum: str, *, support: str, added: dict) -> dict:
        """Everything one stratum's fits read, restricted to the declared support.

        ``added`` is what :func:`added_values` returns: the retained value of each added
        block, and the sample identities the cohort export covers at all. An admitted
        row outside the retained set has left the complete case and is dropped here; an
        admitted row the export does not cover at all is a gap in the export and is
        refused, because silently dropping it would shrink the support unrecorded.
        """

        key = (stratum, support, added.get('digest', ''))
        if key in self._state:
            return self._state[key]
        cohort, endpoint = self.STRATA[stratum]
        rows = self._rows(cohort)
        groups = json.loads((self.qualification / 'groups' / f'{cohort}.json'
                             ).read_text(encoding='utf-8'))['assignment']
        missing = sorted({row.unit for row in rows} - set(groups))
        if missing:
            raise SystemExit(f'{cohort}: {len(missing)} units carry no admitted group')
        values: dict[str, dict[str, float]] = {}
        if support == 'complete_case':
            if not added:
                raise SystemExit('the complete case needs the added-block values')
            values = added['values']
            uncovered = [row.row_id for row in rows
                         if self.sample_key(row) not in added['covered']]
            if uncovered:
                raise SystemExit(
                    f'{cohort}: {len(uncovered)} admitted rows are not covered by the cohort '
                    f'export at all, first {uncovered[0]!r}; the export must cover the rows this '
                    'lane is fitted on before any of them can be dropped as a loss')
            rows = [row for row in rows
                    if all(self.sample_key(row) in values[role] for role in values)]
            if not rows:
                raise SystemExit(f'{cohort}: the complete case retains no row')
        profiles, _ = self._profiles()
        blocks, census = breadth.prune_constant_columns(breadth.control_blocks(rows, profiles))
        backgrounds = np.asarray([row.background for row in rows])
        group_vector = np.asarray([groups[row.unit] for row in rows])
        labels = np.asarray([row.oriented_label() for row in rows], dtype=np.float64)
        target = breadth.rerank(labels, backgrounds) if endpoint == 'ranking' else labels
        for role, retained in values.items():
            column = np.asarray([retained[self.sample_key(row)] for row in rows], dtype=float)
            blocks[lanes.NEW_BLOCKS[role]] = lanes.prepare_scalar(column, backgrounds)
        state = {'rows': rows, 'blocks': blocks, 'census': census, 'target': target,
                 'backgrounds': backgrounds, 'groups': group_vector,
                 'group_order': sorted(set(group_vector.tolist())),
                 'metric': 'spearman' if endpoint == 'ranking' else 'error',
                 'model_column': 'M_rank' if endpoint == 'ranking' else 'M_nats',
                 'endpoint': endpoint, 'cohort': cohort,
                 'support_sha256': lanes.identity_digest([row.row_id for row in rows])}
        self._state[key] = state
        return state

    def _profiles(self):
        from src.capability.stability.stability_gate import load_profiles
        plan = json.loads((self.qualification / 'profile-plan.json').read_text(encoding='utf-8'))
        wildtypes = {row['name']: row['wildtype'] for row in plan['backgrounds']}
        return load_profiles(self.profiles_path, wildtypes)

    # --------------------------------------------------------------------- fit

    def fit(self, state: dict, arm: str, designs: dict[str, tuple[str, ...]]) -> dict:
        """One cell: per-group values of the three arms, seeds averaged within background.

        The three arms see the identical rows, the identical seeded partitions and the
        identical label budget; only their columns differ. The aggregation is the lane's
        own: seeds within a background, backgrounds within their group, groups equal.
        """

        model = breadth.model_block(state['rows'], self.scores()[arm], state['backgrounds'])
        blocks = {**state['blocks'], 'M': model[state['model_column']]}
        per_background: dict[str, dict[str, list[float]]] = {}
        alphas: dict[str, dict[str, list[float]]] = {}
        for seed in breadth.SPLIT_SEEDS:
            signature = breadth.outer_signature(state['groups'], seed)
            predictions, folds = {}, {}
            for name, columns in designs.items():
                matrix = breadth.design(blocks, list(columns))
                predictions[name], records = breadth.fold_predictions(
                    matrix, state['target'], state['backgrounds'], state['groups'], signature)
                folds[name] = [record['alpha'] for record in records]
            alphas[str(seed)] = folds
            for row in breadth.background_metrics(predictions, state['target'],
                                                  state['backgrounds'], state['groups'],
                                                  metric=state['metric']):
                for name, value in row['scores'].items():
                    if value is None:
                        raise SystemExit(f"{arm}: undefined {state['metric']} in background "
                                         f"{row['background']}")
                    per_background.setdefault(name, {}).setdefault(
                        row['background'], []).append(float(value))
        values = {}
        for name, by_background in per_background.items():
            rows = [{'group': str(state['groups'][state['backgrounds'] == background][0]),
                     'value': float(np.mean(seeds))}
                    for background, seeds in sorted(by_background.items())]
            values[name] = breadth.aggregate(rows, 'value', state['group_order']).tolist()
        return {'values': values, 'alphas': alphas,
                'dimensions': {name: int(breadth.design(blocks, list(columns)).shape[1])
                               for name, columns in designs.items()}}

    # ------------------------------------------------------------------- pilot

    def archived(self, stratum: str) -> dict:
        """The published per-background control and joint values of one stratum."""

        cohort, endpoint = self.STRATA[stratum]
        path = self.published / 'panels' / f'{cohort}-{endpoint}.json'
        panel = json.loads(path.read_text(encoding='utf-8'))
        if panel.get('cohort') != cohort or panel.get('endpoint') != endpoint:
            raise SystemExit(f'{path} does not record {cohort}/{endpoint}')
        cells: dict[tuple[str, int, str], dict[str, float]] = {}
        for row in panel['per_background']:
            cells[row['arm'], int(row['seed']), row['background']] = {
                lanes.CONTROL_ARM: float(row['control']),
                lanes.JOINT_ARM: float(row['control_plus_model'])}
        return {'path': str(path.relative_to(ROOT)), 'sha256': sha256_file(path),
                'arms': list(panel['arms']), 'cells': cells,
                'alphas': panel['fits'], 'qualified': list(panel['qualified_controls'])}


ADAPTERS = {PhenotypeAdapter.lane: PhenotypeAdapter}


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


def cached_cells(out: Path, stratum: str, channel: str, support: str, rungs, arms) -> dict:
    """Every cell of one family that is on disk, keyed as the panel reads them."""

    cells = {}
    for rung in rungs:
        for arm in arms:
            path = cell_path(out, stratum, channel, support, rung, arm)
            if not path.is_file():
                continue
            record = json.loads(path.read_text(encoding='utf-8'))
            cells[rung, arm] = {**record['values'], 'support_sha256': record['support_sha256']}
    return cells


# --------------------------------------------------------------------------- #
# Modes.
# --------------------------------------------------------------------------- #

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
        record['strata'][stratum] = entry
    if wave['complete']:
        rows = lanes.read_rows(args.export / lanes.ROWS_FILE)
        record['complete_case'] = lanes.complete_case(args.lane, rows, wave['tables'])
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
    """

    declared = lanes.lane(args.lane)
    inputs = instance.inputs()
    inputs_sha256 = digest_text(json.dumps(inputs, sort_keys=True))
    strata = args.stratum or instance.strata()
    report = {'schema': PILOT_SCHEMA, 'lane': args.lane, 'status': 'passed',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'tolerance': PILOT_TOLERANCE, 'inputs_sha256': inputs, 'strata': {}}
    for stratum in strata:
        archived = instance.archived(stratum)
        qualified = instance.qualified(stratum)
        if archived['qualified'] != list(qualified):
            raise SystemExit(f'{stratum}: the published panel was fitted over '
                             f'{archived["qualified"]}, not {list(qualified)}')
        state = instance.state(stratum, support='full', added={})
        arms = [arm for arm in (args.arm or declared.arms) if arm in archived['arms']]
        if not arms:
            raise SystemExit(f'{stratum}: the published panel covers none of the named arms')
        reproduced, worst = {}, 0.
        for arm in arms:
            cell = fit_cell(args.lane, instance, state, stratum=stratum, channel='likelihood',
                            support='full', rung=lanes.ARCHIVED_RUNG, arm=arm,
                            qualified=qualified, inputs_sha256=inputs_sha256, out=args.out,
                            refit=args.refit)
            # The published record is per background; the refit is aggregated to groups, so
            # the comparison is made on the published cells' own aggregation.
            published = {}
            for (named, seed, background), values in archived['cells'].items():
                if named != arm:
                    continue
                for name, value in values.items():
                    published.setdefault(name, {}).setdefault(background, []).append(value)
            differences = {}
            for name in (lanes.CONTROL_ARM, lanes.JOINT_ARM):
                rows = [{'group': str(state['groups'][state['backgrounds'] == background][0]),
                         'value': float(np.mean(seeds))}
                        for background, seeds in sorted(published[name].items())]
                expected = breadth.aggregate(rows, 'value', state['group_order'])
                difference = float(np.max(np.abs(np.asarray(cell['values'][name]) - expected)))
                differences[name] = difference
                worst = max(worst, difference)
            matched = all(
                cell['alphas'][str(seed)][lanes.CONTROL_ARM]
                == archived['alphas'][arm][str(seed)]['alphas_control']
                and cell['alphas'][str(seed)][lanes.JOINT_ARM]
                == archived['alphas'][arm][str(seed)]['alphas_augmented']
                for seed in breadth.SPLIT_SEEDS)
            reproduced[arm] = {'per_group_max_absolute_difference': differences,
                               'penalties_reproduced': matched, 'alphas': cell['alphas']}
            if not matched or max(differences.values()) > PILOT_TOLERANCE:
                report['status'] = 'failed'
        report['strata'][stratum] = {
            'published': archived['path'], 'published_sha256': archived['sha256'],
            'arms': arms, 'groups': len(state['group_order']),
            'support_sha256': state['support_sha256'],
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
    inputs_sha256 = digest_text(json.dumps(inputs, sort_keys=True))
    supports = runnable_supports(wave)
    report = {'lane': args.lane, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'inputs_sha256': inputs, 'supports': supports, 'fitted': [], 'skipped': []}
    added: dict = {}
    if wave['complete']:
        rows = lanes.read_rows(args.export / lanes.ROWS_FILE)
        support_record = lanes.complete_case(args.lane, rows, wave['tables'])
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
                                arm=arm, qualified=qualified, inputs_sha256=inputs_sha256,
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


def panels(args, instance, wave) -> dict:
    """Assemble whichever declared families are complete on disk, and name the rest."""

    declared = lanes.lane(args.lane)
    written, pending = {}, []
    for stratum in (args.stratum or instance.strata()):
        for channel in declared.channels:
            for support, role in (('complete_case', 'primary'), ('full', 'sensitivity')):
                if args.support and support not in args.support:
                    continue
                rungs = declared.rungs if role == 'primary' else ('nuisance',)
                cells = cached_cells(args.out, stratum, channel, support, rungs, declared.arms)
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
                                     support_sha256=digests.pop(), role=role)
                record['wave'] = wave['status']
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
                        help='a completed model-score directory; repeatable')
    parser.add_argument('--stratum', action='append', default=[])
    parser.add_argument('--arm', action='append', default=[])
    parser.add_argument('--support', action='append', default=[],
                        choices=['complete_case', 'full'])
    parser.add_argument('--refit', action='store_true',
                        help='replace a cached cell instead of refusing a changed input')
    args = parser.parse_args(argv)
    for name in ('out', 'export', 'couplings', 'gemme', 'qualification', 'published', 'profiles'):
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
