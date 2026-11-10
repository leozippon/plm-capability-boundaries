#!/usr/bin/env python3
"""One adapter per lane: the lane's own rows, blocks, fitter and archive.

:mod:`fit_explicit_statistics_lanes` owns the wave gate, the resume rule, the
pilot comparison and the panel assembly; the declaration in
:mod:`src.capability.extensions.explicit_statistics_lanes` owns the hierarchy,
the directions and the families. Neither owns an estimand. What is left over is
exactly what differs between lanes, and it lives here:

* ``inputs``    -- every frozen byte the lane reads, as a name -> digest map.
* ``arms``      -- the arms this lane can actually fit, read off its own archives.
* ``strata``    -- the lane's declared endpoints.
* ``qualified`` -- the archived qualified control set, per arm, never re-qualified.
* ``state``     -- rows, blocks, target, groups and support digest of one stratum.
* ``fit``       -- one cell: per-group values of the three arms, seeds averaged.
* ``archived``  -- the published per-group values and penalties the pilot repeats.

**Nothing here qualifies a control.** Each lane's archive already recorded which
blocks its forward qualification kept, and the hierarchy carries profile from the
profile rung upwards whether or not that qualification kept it. A lane reads its
archive and refuses a shape it does not recognise.

**The qualified control set is a function of the arm.** The abundance archive
offered the tokenisation block ``T`` to each arm separately and 23 of 33 arms
qualified it, so those arms' control sets carry it and the other ten do not. That
is the archive's own shape and it is the correct one: a tokenisation nuisance has
to absorb *that arm's* segmentation artefacts, and a control held common across
arms would credit segmentation to the model. The stability archive offered ``T``
under the same rule and no arm qualified it, so that lane's realised control set
is arm-invariant although its mechanism is not. The interaction archive carries
``T`` in every arm's control set by declaration rather than per arm.

**Several strata share one fit.** Abundance reads a squared error and a rank
correlation, and stability reads three errors, off the *identical* out-of-fold
predictions; only the per-group statistic differs. Each adapter therefore computes
every reading of a cell at once and serves the others from
:attr:`_readings`, so a second stratum costs nothing. The cache is keyed by the
support digest, the arm and the control columns, which is the full identity of a
fit.

**The declaration's bytes are recorded, not binding.** Each adapter reports the
declaration module's digest under a ``note:`` key, which the runner writes into a cell
and leaves out of the digest the cell is bound to. What the cell *is* bound to is
``fit_identity``, the declared contract for its own support. The reason is that this
module's declaration is mostly prose by design -- it is where every limitation and
refusal reason is recorded -- so binding a fit to its bytes would make writing a
limitation cost a refit. The fitters' own bytes stay binding, because almost any edit to
them can move a number.

**Two identities per row, never one.** A lane's own row identity -- the frozen
registry's ``sample_id`` for the two single-substitution lanes, the background and
cycle index for the interaction lane -- is what the support digest is taken over and
what binds a cached cell to the rows it was fitted on. The *export* identity is
separate: the explicit-statistics export names a row ``<assay>::<mutation>``, which is
the only key the coupling and GEMME tables can be joined on. Conflating the two reads
a sha256 against a human-readable key and joins nothing.

**The abundance export is keyed by domain and refused by reference.** Two of the 428
Domainome domains are byte-identical in sequence, so the export carries 427 references
and 256 of its rows name an ``assay`` that is not their ``wildtype_id``. The join is on
the assay, because that is what the export's ``sample_id`` is built from; the losses are
enumerated per reference, so a refusal of that shared reference drops both domains
together. Neither is a defect: one is the row identity and the other is the model's.

**The interaction lane's added block is a cycle contrast, not a row value.** The export
scores mutants, and this lane's row is a four-corner cycle, so the coupling value of a
cycle is ``v(AB) - v(A) - v(B)`` with ``v(WT) = 0`` by construction -- which is exactly
the contrast the declaration calls identically zero for GEMME and non-zero for a full
sequence Potts energy difference. A cycle therefore enters the complete case only when
all three of its scored corners do, and the corner keys are recovered by diffing each
retained measured state against its own wild type.

**The nuisance rung is not profile-free, and is not claimed to be.** ``G`` is a
frozen block whose first stage reads the lane's standing columns, which include
the profile blocks in all three lanes. Refitting ``G`` without them would make the
nuisance rung something other than the archived control set minus profile, and
would stop the profile rung reproducing the archive column for column. ``G`` is
therefore kept exactly as the archive defined it and the leak is recorded in
:data:`NUISANCE_RUNG_LIMITATION` rather than removed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_variable, '1')

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.baselines.cohorts import SAMPLE_SEPARATOR  # noqa: E402
from src.capability.core.io import sha256_file  # noqa: E402
from src.capability.extensions import explicit_statistics_lanes as lanes  # noqa: E402
from src.capability.extensions import phenotype_breadth as breadth  # noqa: E402

#: This module and the runner beside it, hashed by path so an adapter's inputs
#: record binds the code that read them without importing the runner back.
ADAPTER_SOURCE = Path(__file__)
RUNNER_SOURCE = Path(__file__).with_name('fit_explicit_statistics_lanes.py')

#: Why the nuisance rung carries profile information through ``G``. The statement is the
#: declaration's, read from there so one wording serves every lane and every record.
NUISANCE_RUNG_LIMITATION = lanes.NUISANCE_RUNG_LIMITATION

#: What a pilot comparison does and does not exercise when every group holds one
#: measured background: the fit and the per-group statistic, but not the averaging
#: of backgrounds inside a group.
ONE_BACKGROUND_PER_GROUP = (
    'every group holds exactly one measured background, so this comparison exercises the fitting '
    'path and the per-group statistic but not the background-to-group averaging, which is the '
    'identity on this support')
MANY_BACKGROUNDS_PER_GROUP = (
    'groups hold up to {most} measured backgrounds, so this comparison exercises the whole '
    'published aggregation: seeds within a background, backgrounds within their group, groups '
    'equal')


def _one(paths, flag: str, lane: str) -> Path:
    """Exactly one path from a repeatable argument, or an explicit refusal."""

    values = [path for path in (paths or []) if path is not None]
    if len(values) != 1:
        raise SystemExit(f'{lane}: --{flag} names {len(values)} directories; this lane reads '
                         'exactly one')
    return Path(values[0])


def _require(path, lane: str, what: str) -> Path:
    if path is None:
        raise SystemExit(f'{lane}: this lane needs {what}')
    return Path(path)


def _backgrounds_per_group(backgrounds, groups) -> dict:
    """How many measured backgrounds each group holds, and what that means."""

    counts: dict[str, set] = {}
    for background, group in zip(np.asarray(backgrounds).tolist(), np.asarray(groups).tolist()):
        counts.setdefault(str(group), set()).add(str(background))
    most = max((len(value) for value in counts.values()), default=0)
    return {'groups': len(counts), 'backgrounds': len({b for v in counts.values() for b in v}),
            'most_backgrounds_in_one_group': most,
            'exercises_aggregation': most > 1,
            'reading': (MANY_BACKGROUNDS_PER_GROUP.format(most=most) if most > 1
                        else ONE_BACKGROUND_PER_GROUP)}


class _Readings:
    """Per-cell cache of every reading one fit supports.

    One fit of one (support, arm, control columns) triple serves every stratum of
    the lane, because the strata differ only in the per-group statistic they take
    of the same out-of-fold predictions. ``fit`` computes them all and the other
    strata read them back here.
    """

    def __init__(self):
        self._cells: dict = {}

    def key(self, state: dict, arm: str, designs: dict):
        return (state['support_sha256'], arm, tuple(designs[lanes.CONTROL_ARM]))

    def get(self, state, arm, designs, reading):
        cell = self._cells.get(self.key(state, arm, designs))
        if cell is None:
            return None
        return {'values': cell['readings'][reading], 'alphas': cell['alphas'],
                'dimensions': cell['dimensions'], 'readings_available': sorted(cell['readings'])}

    def put(self, state, arm, designs, *, readings, alphas, dimensions):
        self._cells[self.key(state, arm, designs)] = {
            'readings': readings, 'alphas': alphas, 'dimensions': dimensions}


# --------------------------------------------------------------------------- #
# The added-phenotype lane.
# --------------------------------------------------------------------------- #

class PhenotypeAdapter:
    """The added-phenotype lane: four cohort/endpoint strata on disjoint rows.

    The admitted rows are read back from the qualification run's own row files and
    the wild-type sequences from its profile catalogue, each verified against the
    digest the row carries, so the support is the admitted one and is never
    re-derived into a different one. The group assignment, the qualified control
    set and the published per-background values are read from the frozen runs and
    never re-qualified.
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
        self.qualification = _require(args.qualification, self.lane, 'the frozen qualification run')
        self.published = _require(args.published, self.lane, 'the published panel run')
        self.profiles_path = _require(args.profiles, self.lane, 'the profile feature store')
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
        record['note:code:explicit_statistics_lanes'] = sha256_file(Path(lanes.__file__))
        record['code:adapters'] = sha256_file(ADAPTER_SOURCE)
        record['code:runner'] = sha256_file(RUNNER_SOURCE)
        return record

    def scores(self) -> dict[str, dict[str, float]]:
        if self._scores is None:
            self._scores = breadth.read_arm_scores(self.score_directories)
        return self._scores

    def arms(self) -> list[str]:
        return sorted(self.scores())

    def strata(self) -> list[str]:
        return list(self.STRATA)

    def qualified(self, stratum: str, arm: str | None = None) -> tuple[str, ...]:
        """The archived qualified control set. This lane qualified it per stratum,
        not per arm, so the arm is accepted and makes no difference."""

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

        The cohort export names a row by its background and its mutation string,
        and it carries one row per distinct pair, so several admitted rows of one
        pair read the same added-block value. That is a join, not a duplication:
        the block is a property of the (background, mutation) pair.
        """

        return f'{row.background}{SAMPLE_SEPARATOR}{row.wt_aa}{row.position}{row.mt_aa}'

    # ------------------------------------------------------------------- state

    def state(self, stratum: str, *, support: str, added: dict) -> dict:
        """Everything one stratum's fits read, restricted to the declared support.

        ``added`` is what the runner's ``added_values`` returns: the retained value
        of each added block, and the sample identities the cohort export covers at
        all. An admitted row outside the retained set has left the complete case
        and is dropped here; an admitted row the export does not cover at all is a
        gap in the export and is refused, because silently dropping it would
        shrink the support unrecorded.
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
                 'endpoint': endpoint, 'cohort': cohort, 'stratum': stratum, 'support': support,
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

        The three arms see the identical rows, the identical seeded partitions and
        the identical label budget; only their columns differ. The aggregation is
        the lane's own: seeds within a background, backgrounds within their group,
        groups equal.
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
        """The published values of one stratum, at the published aggregation.

        The panel records one value per (arm, seed, background), so the comparison
        is made at the aggregation the published panel declares and not one step
        short of it: seeds are averaged within a background, backgrounds within
        their group, and groups enter equally.
        """

        cohort, endpoint = self.STRATA[stratum]
        path = self.published / 'panels' / f'{cohort}-{endpoint}.json'
        panel = json.loads(path.read_text(encoding='utf-8'))
        if panel.get('cohort') != cohort or panel.get('endpoint') != endpoint:
            raise SystemExit(f'{path} does not record {cohort}/{endpoint}')
        state = self.state(stratum, support='full', added={})
        group_of = {str(background): str(state['groups'][state['backgrounds'] == background][0])
                    for background in sorted(set(state['backgrounds'].tolist()))}
        seeds: dict[str, dict[str, dict[str, list[float]]]] = {}
        for row in panel['per_background']:
            background = str(row['background'])
            if group_of.get(background) != str(row['group']):
                raise SystemExit(f'{path}: background {background!r} is published in group '
                                 f'{row["group"]!r} and the admitted rows put it in '
                                 f'{group_of.get(background)!r}')
            cell = seeds.setdefault(str(row['arm']), {})
            cell.setdefault(lanes.CONTROL_ARM, {}).setdefault(background, []).append(
                float(row['control']))
            cell.setdefault(lanes.JOINT_ARM, {}).setdefault(background, []).append(
                float(row['control_plus_model']))
        values = {}
        for arm, by_arm in seeds.items():
            values[arm] = {}
            for name, by_background in by_arm.items():
                rows = [{'group': group_of[background], 'value': float(np.mean(drawn))}
                        for background, drawn in sorted(by_background.items())]
                values[arm][name] = breadth.aggregate(
                    rows, 'value', state['group_order']).tolist()
        penalties = {arm: {str(seed): {lanes.CONTROL_ARM: list(entry['alphas_control']),
                                       lanes.JOINT_ARM: list(entry['alphas_augmented'])}
                           for seed, entry in panel['fits'][arm].items()}
                     for arm in panel['arms']}
        shape = _backgrounds_per_group(state['backgrounds'], state['groups'])
        return {'path': str(path.relative_to(ROOT)), 'sha256': sha256_file(path),
                'arms': list(panel['arms']),
                'qualified': {arm: list(panel['qualified_controls']) for arm in panel['arms']},
                'values': values, 'penalties': penalties,
                'aggregation': ('seeds averaged within a background, backgrounds averaged within '
                                'their group, groups equal'),
                'support_shape': shape}



# --------------------------------------------------------------------------- #
# The abundance lane.
# --------------------------------------------------------------------------- #

class AbundanceAdapter:
    """Domainome abundance: two readings of one set of fits, 96 family groups.

    The frozen cohort, its profile store and its control qualification are read
    through the published replay stage, which hashes every one of the 428 restored
    per-domain archives against the arm's own extraction manifest and refuses a
    compacted or partially restored pool. The arm's scalar likelihood difference
    ``M`` and its tokenisation block ``T`` come out of that archive exactly as the
    original fit read them; nothing is re-scored.

    Both declared strata -- the group-equal squared error and the grouped
    within-domain Spearman -- are two statistics of the same out-of-fold
    predictions, so one fit serves both.
    """

    lane = 'abundance'
    #: Stratum name -> the per-group reading of the shared predictions.
    STRATA = {'squared_error': 'mse', 'ranking': 'rank'}

    def __init__(self, args):
        self.source = _require(args.qualification, self.lane,
                               'the frozen cohort, profile store and control qualification '
                               '(--qualification)')
        self.published = _require(args.published, self.lane,
                                  'the published bidirectional panel (--published)')
        self.extraction = _one(args.scores, 'scores', self.lane)
        self._readings = _Readings()
        self._state: dict = {}
        self._archived: dict = {}

    # ------------------------------------------------------------------ inputs

    def _module(self):
        """The published replay stage, imported lazily: it loads torch."""

        from scripts.capability.extensions.bidirectional_ablation_20261106 import r2_abundance
        return r2_abundance

    def _shared(self) -> dict:
        """The frozen cohort, controls, profiles and panel, validated once."""

        module = self._module()
        if self.source.resolve() != (ROOT / module.SOURCE).resolve():
            raise SystemExit(
                f'{self.lane}: --qualification names {self.source}, and the published replay stage '
                f'reads {ROOT / module.SOURCE}. The frozen cohort is not relocated here')
        return module.initialise(ROOT, self.extraction)

    def inputs(self) -> dict[str, str]:
        module = self._module()
        state = self._shared()
        record = {str(Path(path).relative_to(ROOT)): state['hashes'][key]
                  for key, path in state['paths'].items()}
        for arm in self.arms():
            path = self.published / 'fits' / f'{arm}.json'
            record[str(path.relative_to(ROOT))] = sha256_file(path)
            reference = ROOT / module.SOURCE / 'fits' / f'fit_{arm}.json'
            record[str(reference.relative_to(ROOT))] = sha256_file(reference)
        for name in ('panel.json', 'contract.json'):
            path = self.published / name
            record[str(path.relative_to(ROOT))] = sha256_file(path)
        record.update({f'code:{key}': value for key, value in module.code_hashes().items()})
        record['note:code:explicit_statistics_lanes'] = sha256_file(Path(lanes.__file__))
        record['code:adapters'] = sha256_file(ADAPTER_SOURCE)
        record['code:runner'] = sha256_file(RUNNER_SOURCE)
        return record

    def arms(self) -> list[str]:
        """The arms whose restored extraction and published fit are both present."""

        module = self._module()
        state = self._shared()
        present = []
        for arm in sorted(lanes.PANEL_ARMS):
            if not (self.published / 'fits' / f'{arm}.json').is_file():
                continue
            try:
                module.locate_extraction(state['extraction'], arm)
            except Exception:  # noqa: BLE001 - an absent archive is an absent arm
                continue
            present.append(arm)
        if not present:
            raise SystemExit(f'{self.lane}: no arm has both a restored extraction under '
                             f'{self.extraction} and a published fit under {self.published}')
        return present

    def strata(self) -> list[str]:
        return list(self.STRATA)

    # -------------------------------------------------------------- qualified

    def qualified(self, stratum: str, arm: str | None = None) -> tuple[str, ...]:
        """The archived qualified control set, with this arm's own ``T`` where it
        qualified one.

        ``arm`` of ``None`` returns the arm-invariant part, which is what the fit
        plan is stated over; a named arm returns the columns that arm's archive was
        actually fitted on, in the archived order.
        """

        module = self._module()
        frozen = tuple(self._shared()['controls']['qualified_control_set'])
        if arm is None:
            return frozen
        return module.control_columns(self._baseline(arm), frozen)

    def _baseline(self, arm: str) -> str:
        module = self._module()
        reference = module.replay.read_json(
            module.replay.input_paths(ROOT, 'abundance', arm)['reference'])
        return module.archived_baseline(reference, arm)

    # ------------------------------------------------------------------- state

    def state(self, stratum: str, *, support: str, added: dict) -> dict:
        key = (support, added.get('digest', ''))
        if key not in self._state:
            self._state[key] = self._support(support, added)
        return dict(self._state[key], stratum=stratum,
                    reading=self.STRATA[stratum], support=support)

    def _support(self, support: str, added: dict) -> dict:
        state = self._shared()
        panel, records = state['panel'], state['records']
        # Two identities: the frozen registry's sample_id is this lane's own row
        # identity and the support digest is taken over it; the export names the same
        # row by its assay and mutation, which is the only key its tables carry.
        identifiers = [row['sample_id'] for row in records]
        export = [f"{row['unit']}{SAMPLE_SEPARATOR}{row['mutation']}" for row in records]
        keep = np.ones(len(identifiers), dtype=bool)
        if support == 'complete_case':
            if not added:
                raise SystemExit('the complete case needs the added-block values')
            covered = added['covered']
            uncovered = [name for name in export if name not in covered]
            if uncovered:
                raise SystemExit(
                    f'{self.lane}: {len(uncovered)} cohort rows are not covered by the explicit '
                    f'statistics export at all, first {uncovered[0]!r}; the export must cover the '
                    'rows this lane is fitted on before any of them can be dropped as a loss')
            retained = set.intersection(*(set(values) for values in added['values'].values()))
            keep = np.asarray([name in retained for name in export])
            if not keep.any():
                raise SystemExit(f'{self.lane}: the complete case retains no row')
        working = self._restrict(panel, keep)
        added_blocks = {}
        for role, values in (added.get('values') or {}).items():
            column = np.asarray([values[name] for name, held in zip(export, keep) if held],
                                dtype=float)
            added_blocks[lanes.NEW_BLOCKS[role]] = lanes.prepare_scalar(column, working['domain'])
        retained = [name for name, held in zip(identifiers, keep) if held]
        return {'panel': working, 'added_blocks': added_blocks, 'row_mask': keep,
                'sample_ids': retained,
                'export_keys': [name for name, held in zip(export, keep) if held],
                'backgrounds': working['domain'], 'groups': working['group'],
                'group_order': sorted(set(working['group'].tolist())),
                'first_stage': tuple(block for block in self.qualified('squared_error')
                                     if block != 'G'),
                'support_sha256': lanes.identity_digest(retained),
                'rows_offered': len(identifiers), 'rows_retained': int(keep.sum()),
                'nuisance_rung_limitation': NUISANCE_RUNG_LIMITATION}

    @staticmethod
    def _restrict(panel: dict, keep: np.ndarray) -> dict:
        """The panel on a subset of its rows, with the nested weights refitted.

        The weights are domain-then-site-then-variant shares of their group, so a
        dropped row changes the shares of the rows that remain. Recomputing them
        is what keeps the retained support group-equal rather than group-equal on
        a support that no longer exists.
        """

        if keep.all():
            return panel
        from src.capability.mutation import external_confirmation as abundance
        working = {key: value[keep] for key, value in panel.items()
                   if key not in ('blocks', 'weights')}
        working['blocks'] = {name: block[keep] for name, block in panel['blocks'].items()}
        working['weights'] = abundance.nested_weights(working['group'], working['domain'],
                                                      working['site'])
        return working

    # --------------------------------------------------------------------- fit

    def fit(self, state: dict, arm: str, designs: dict[str, tuple[str, ...]]) -> dict:
        """One cell: both readings of one set of fits, three split seeds averaged.

        The three arms see identical rows, identical outer and inner group
        partitions and the identical label budget, and ``G`` is regenerated inside
        every outer training partition from the archived first-stage columns. Both
        declared readings are taken from the same out-of-fold predictions, so the
        squared-error and the ranking strata are never fitted twice.
        """

        cached = self._readings.get(state, arm, designs, state['reading'])
        if cached is not None:
            return cached
        from src.capability.mutation import external_confirmation as abundance
        module = self._module()
        panel = state['panel']
        # The archived model blocks are exported on the full cohort, so they are
        # restricted with the same mask the panel was, never re-derived.
        prepared = module.prepare(arm)
        mask = state['row_mask']
        model = {name: block if mask.all() else block[mask]
                 for name, block in prepared['blocks'].items()}
        blocks = {**panel['blocks'], **model, **state['added_blocks']}
        missing = sorted({name for columns in designs.values() for name in columns}
                         - set(blocks) - {'G'})
        if missing:
            raise SystemExit(f'{arm}: the design names feature blocks the panel does not carry: '
                             f'{missing}')
        readings = {name: {} for name in self.STRATA.values()}
        alphas: dict[str, dict[str, list[float]]] = {}
        dimensions: dict[str, int] = {}
        for seed in module.SEEDS:
            result = abundance.fold_predictions(
                panel, blocks, {name: tuple(columns) for name, columns in designs.items()},
                seed=seed, first_stage=state['first_stage'], device='cpu')
            alphas[str(seed)] = {name: [fold['alpha'][name] for fold in result['folds']]
                                 for name in designs}
            dimensions = {name: int(result['folds'][0]['dimensions'][name]) for name in designs}
            for name in designs:
                prediction = result['predictions'][name]
                labels, errors = abundance.group_errors(
                    panel['target'], prediction, panel['group'], panel['domain'], panel['site'])
                if labels.tolist() != state['group_order']:
                    raise SystemExit(f'{arm} seed {seed}: the group support of {name} differs')
                readings['mse'].setdefault(name, []).append(np.asarray(errors, dtype=float))
                ranks = module.group_ranks(panel, prediction)
                if any(value is None for value in ranks):
                    raise SystemExit(f'{arm} seed {seed}: {name} leaves a group without a defined '
                                     'within-domain rank correlation; an undefined group is a '
                                     'support finding, not a value to average over')
                readings['rank'].setdefault(name, []).append(np.asarray(ranks, dtype=float))
        collapsed = {reading: {name: np.mean(values, axis=0).tolist()
                               for name, values in by_name.items()}
                     for reading, by_name in readings.items()}
        self._readings.put(state, arm, designs, readings=collapsed, alphas=alphas,
                           dimensions=dimensions)
        return self._readings.get(state, arm, designs, state['reading'])

    # ------------------------------------------------------------------- pilot

    def archived(self, stratum: str) -> dict:
        """The published per-group values and penalties of the bidirectional panel.

        The panel's per-arm records carry the per-group reading of each of the
        three arms for each of the three split seeds, so the comparison is made at
        the published aggregation: the seeds are averaged inside the group, and the
        96 groups each already carry the published domain-then-site-then-variant
        weighting of their own rows.
        """

        if stratum in self._archived:
            return self._archived[stratum]
        field = self.STRATA[stratum]
        role_of = {lanes.CONTROL_ARM: 'C', lanes.JOINT_ARM: 'CM'}
        state = self.state(stratum, support='full', added={})
        values, penalties, qualified, arms = {}, {}, {}, []
        for arm in self.arms():
            path = self.published / 'fits' / f'{arm}.json'
            record = json.loads(path.read_text(encoding='utf-8'))
            if record.get('schema') not in self._module().FIT_SCHEMAS or record.get('arm') != arm:
                raise SystemExit(f'{path} is not a published fit record of {arm}')
            if record.get('status') != 'complete':
                raise SystemExit(f'{path} records status {record.get("status")!r}; a failed '
                                 'archived arm is not a reproduction target')
            if record['groups'] != state['group_order']:
                raise SystemExit(f'{path}: the published group order differs from the realised '
                                 'support')
            seeds = {str(seed) for seed in self._module().SEEDS}
            if set(record['seeds']) != seeds:
                raise SystemExit(f'{path} covers split seeds {sorted(record["seeds"])}, not '
                                 f'{sorted(seeds)}')
            values[arm] = {}
            for name, role in role_of.items():
                drawn = [np.asarray([np.nan if value is None else float(value)
                                     for value in record['seeds'][seed][field][role]], dtype=float)
                         for seed in sorted(seeds)]
                mean = np.mean(drawn, axis=0)
                if mean.shape != (len(state['group_order']),) or not np.isfinite(mean).all():
                    raise SystemExit(f'{path}: the published {field}/{role} reading is not one '
                                     f'finite value for each of the {len(state["group_order"])} '
                                     'groups')
                values[arm][name] = mean.tolist()
            penalties[arm] = {
                seed: {name: [fold['alpha'][role] for fold in record['seeds'][seed]['folds']]
                       for name, role in role_of.items()}
                for seed in sorted(seeds)}
            qualified[arm] = list(record['control_columns'])
            arms.append(arm)
        panel = json.loads((self.published / 'panel.json').read_text(encoding='utf-8'))
        shape = _backgrounds_per_group(state['backgrounds'], state['groups'])
        self._archived[stratum] = {
            'path': str((self.published / 'fits').relative_to(ROOT)),
            'sha256': sha256_file(self.published / 'panel.json'),
            'arms': arms, 'qualified': qualified, 'values': values, 'penalties': penalties,
            'aggregation': ('the published per-group reading of each arm, split seeds averaged '
                            'inside the group; the 96 groups carry the published '
                            'domain-then-site-then-variant weighting of their own rows'),
            'support_shape': shape,
            'panel_absolute': {'columns': panel['panel'][stratum]['absolute']['columns'],
                               'points': panel['panel'][stratum]['absolute']['points']},
        }
        return self._archived[stratum]


# --------------------------------------------------------------------------- #
# The stability lane.
# --------------------------------------------------------------------------- #

class StabilityAdapter:
    """Measured single-substitution stability: three readings of one set of fits.

    The archived primary control set of this lane is ``S2`` -- ``ident``, ``geom``,
    ``chem``, ``G``, ``comp``, ``prof2`` in the archived order -- so the profile
    rung reproduces it column for column and the nuisance rung is that set without
    ``prof2``. The archive's own ``S`` baseline drops composition as well as
    profile and is therefore *not* this hierarchy's nuisance rung.

    The tokenisation block was offered to each arm under the same qualification
    rule the abundance lane used, and the archive records that no arm qualified
    it: every one of the 33 published records carries ``matched_baseline: "S"``
    and ``tokenisation_verdict.qualified: false``. The archived ``S_T`` design was
    the qualification probe, and no ``S_T_M`` joint arm was ever fitted. So this
    lane's realised control set carries no tokenisation column, which is read off
    the archive rather than assumed, and :meth:`qualified` refuses an archive that
    says otherwise.

    The three declared strata -- the within-background rank correlation, the
    transfer error and the site-calibrated error -- are three statistics of the
    same out-of-fold predictions, so one fit serves all three.
    """

    lane = 'stability'
    #: Stratum -> (published control-arm key, published contrast key, the sign that
    #: turns the published contrast back into the joint arm's own value).
    STRATA = {
        'ranking': ('baseline_within_family_ranking', 'profile_ranking', 1.),
        'transfer_error': ('baseline_transfer_error', 'profile_transfer_error', -1.),
        'calibrated_error': ('baseline_calibrated_error', 'profile_calibrated_error', -1.),
    }
    #: The archived qualified control set, in the archived column order.
    ARCHIVED_CONTROL_SET = ('ident', 'geom', 'chem', 'G', 'comp', 'prof2')
    #: The archived design name of each arm of the hierarchy's profile rung.
    ARCHIVED_DESIGN = {lanes.CONTROL_ARM: 'S2', lanes.JOINT_ARM: 'S2_M'}
    SUPPORT = {'rows': 25856, 'groups': 101}

    def __init__(self, args):
        self.gate = _require(args.qualification, self.lane,
                             'the frozen stability gate directory (--qualification)')
        self.published = _require(args.published, self.lane,
                                  'the published ranking-error grid (--published)')
        self.extraction = _one(args.scores, 'scores', self.lane)
        self._shared_state: dict | None = None
        self._readings = _Readings()
        self._state: dict = {}
        self._archived: dict = {}
        self._located: dict | None = None

    # ------------------------------------------------------------------ inputs

    def _shared(self) -> dict:
        if self._shared_state is not None:
            return self._shared_state
        from src.capability.extensions.stability_followups import COHORT_SHA256
        from src.capability.stability import gate_inputs, stability_gate as gate
        paths = gate_inputs.resolve(self.gate)
        if sha256_file(paths['cohort.json']) != COHORT_SHA256:
            raise SystemExit(f'{paths["cohort.json"]} is not the frozen stability cohort '
                             f'{COHORT_SHA256[:12]}; this lane never fits a substitute cohort')
        cohort = json.loads(paths['cohort.json'].read_text())
        profiles, _ = gate.load_profiles(
            paths['profile_features.npz'],
            {row['name']: row['wildtype'] for row in cohort['backgrounds']})
        panel = gate.build_panel(cohort, profiles)
        if (len(panel['target']), len(set(panel['group'].tolist()))) != (
                self.SUPPORT['rows'], self.SUPPORT['groups']):
            raise SystemExit(
                f'{self.lane}: the rebuilt panel holds {len(panel["target"])} rows over '
                f'{len(set(panel["group"].tolist()))} groups, not the frozen {self.SUPPORT}')
        self._shared_state = {'paths': paths, 'cohort': cohort, 'profiles': profiles,
                              'panel': panel, 'cohort_sha256': COHORT_SHA256}
        return self._shared_state

    def _extractions(self) -> dict:
        if self._located is None:
            from scripts.capability.stability.fit_stability_ranking_panel import locate_extractions
            self._located = locate_extractions([self.extraction], list(lanes.PANEL_ARMS))
        return self._located

    def inputs(self) -> dict[str, str]:
        shared = self._shared()
        record = {str(path.relative_to(ROOT)): sha256_file(path)
                  for path in sorted(shared['paths'].values())}
        for arm in self.arms():
            path = self.published / 'fits' / f'fit_{arm}.json'
            record[str(path.relative_to(ROOT))] = sha256_file(path)
            manifest = self._extractions()[arm] / f'manifest_{arm}.json'
            record[str(manifest.relative_to(ROOT))] = sha256_file(manifest)
        from src.capability.extensions import stability_followups
        from src.capability.stability import ranking_error, stability_gate
        for module in (stability_gate, ranking_error, stability_followups):
            record[f'code:{Path(module.__file__).stem}'] = sha256_file(Path(module.__file__))
        record['note:code:explicit_statistics_lanes'] = sha256_file(Path(lanes.__file__))
        record['code:adapters'] = sha256_file(ADAPTER_SOURCE)
        record['code:runner'] = sha256_file(RUNNER_SOURCE)
        return record

    def arms(self) -> list[str]:
        located = self._extractions()
        present = [arm for arm in sorted(lanes.PANEL_ARMS)
                   if arm in located and (self.published / 'fits' / f'fit_{arm}.json').is_file()]
        if not present:
            raise SystemExit(f'{self.lane}: no arm has both a likelihood extraction under '
                             f'{self.extraction} and a published grid record under {self.published}')
        return present

    def strata(self) -> list[str]:
        return list(self.STRATA)

    # -------------------------------------------------------------- qualified

    def qualified(self, stratum: str, arm: str | None = None) -> tuple[str, ...]:
        """The archived primary control set, verified against the arm's own archive.

        The archive's ``matched_baseline`` is a bare string in this lane, not the
        nested record the abundance archive writes, and a reader that indexed it
        like the abundance one would silently take its first character. It is read
        as a string here and an ``S_T`` arm is refused rather than carried,
        because the archive fitted no ``S_T_M`` joint arm for such an arm to be
        compared against.
        """

        if arm is None:
            return self.ARCHIVED_CONTROL_SET
        record = json.loads(
            (self.published / 'fits' / f'fit_{arm}.json').read_text(encoding='utf-8'))
        baseline = record.get('matched_baseline')
        if not isinstance(baseline, str):
            raise SystemExit(f'{arm}: the archived matched_baseline is '
                             f'{type(baseline).__name__}, not the bare string this lane records')
        verdict = (record.get('tokenisation_verdict') or {}).get('qualified')
        if baseline != 'S' or verdict is not False:
            raise SystemExit(
                f'{arm}: the archive records matched_baseline={baseline!r} and '
                f'tokenisation_verdict.qualified={verdict!r}. Every published arm of this lane '
                'qualified no tokenisation block, and the archive fitted no S_T_M joint arm, so an '
                'arm that qualified one has no published joint arm to reproduce and is refused '
                'rather than fitted against a design the archive never held')
        designs = (record.get('held_out_predictions') or {}).get('designs') or []
        missing = [name for name in self.ARCHIVED_DESIGN.values() if name not in designs]
        if missing:
            raise SystemExit(f'{arm}: the archived out-of-fold store holds {sorted(designs)} and '
                             f'not {missing}')
        return self.ARCHIVED_CONTROL_SET

    # ------------------------------------------------------------------- state

    def state(self, stratum: str, *, support: str, added: dict) -> dict:
        key = (support, added.get('digest', ''))
        if key not in self._state:
            self._state[key] = self._support(support, added)
        return dict(self._state[key], stratum=stratum, support=support)

    def _support(self, support: str, added: dict) -> dict:
        from src.capability.stability import stability_gate as gate
        shared = self._shared()
        cohort, panel = shared['cohort'], shared['panel']
        registry = self._registry()
        rows = [row['sample_id'] for row in registry]
        export = [f"{row['unit']}{SAMPLE_SEPARATOR}{row['mutation']}" for row in registry]
        backgrounds = [row['unit'] for row in registry]
        lost: set[str] = set()
        if support == 'complete_case':
            if not added:
                raise SystemExit('the complete case needs the added-block values')
            covered = added['covered']
            uncovered = [name for name in export if name not in covered]
            if uncovered:
                raise SystemExit(
                    f'{self.lane}: {len(uncovered)} cohort rows are not covered by the explicit '
                    f'statistics export at all, first {uncovered[0]!r}')
            retained = set.intersection(*(set(values) for values in added['values'].values()))
            lost = {background for background, name in zip(backgrounds, export)
                    if name not in retained}
            kept = {background for background, name in zip(backgrounds, export)
                    if name in retained}
            partial = sorted(lost & kept)
            if partial:
                raise SystemExit(
                    f'{self.lane}: {len(partial)} backgrounds lose some rows and keep others '
                    f'(first {partial[0]!r}). Every declared loss of this lane is a refusal of a '
                    'whole reference, and the state table of this panel is indexed per background, '
                    'so a part-background loss is refused rather than rebuilt into a panel whose '
                    'wild-type reference row no longer belongs to its own mutants')
            if not kept:
                raise SystemExit(f'{self.lane}: the complete case retains no background')
            cohort = dict(cohort, backgrounds=[row for row in cohort['backgrounds']
                                               if row['name'] in kept])
            panel = gate.build_panel(cohort, shared['profiles'])
        added_blocks = {}
        mask = np.asarray([background not in lost for background in backgrounds])
        for role, values in (added.get('values') or {}).items():
            column = np.asarray([values[name] for name, held in zip(export, mask) if held],
                                dtype=float)
            added_blocks[lanes.NEW_BLOCKS[role]] = lanes.prepare_scalar(column, panel['group'])
        retained_rows = [name for name, held in zip(rows, mask) if held]
        if len(retained_rows) != len(panel['target']):
            raise SystemExit(f'{self.lane}: the restricted cohort holds {len(panel["target"])} '
                             f'rows and the retained identities are {len(retained_rows)}')
        return {'panel': panel, 'cohort': cohort, 'added_blocks': added_blocks,
                'sample_ids': retained_rows, 'lost_backgrounds': sorted(lost),
                'export_keys': [name for name, held in zip(export, mask) if held],
                'backgrounds': panel['group'], 'groups': panel['group'],
                'group_order': sorted(set(panel['group'].tolist())),
                'support_sha256': lanes.identity_digest(retained_rows),
                'rows_offered': len(rows), 'rows_retained': len(retained_rows),
                'nuisance_rung_limitation': NUISANCE_RUNG_LIMITATION}

    def _registry(self) -> list[dict]:
        """The frozen row registry of the cohort, in the panel's own row order.

        It carries both identities a row needs: the frozen ``sample_id`` this lane's
        archives are keyed by, and the unit and mutation the explicit-statistics
        export names the same row by.
        """

        from scripts.capability.reporting import replay_phenotype_predictions as replay
        shared = self._shared()
        return replay.registry(shared['cohort'], 'stability', shared['cohort_sha256'])

    # --------------------------------------------------------------------- fit

    def fit(self, state: dict, arm: str, designs: dict[str, tuple[str, ...]]) -> dict:
        """One cell: all three readings of one set of fits, three split seeds averaged."""

        cached = self._readings.get(state, arm, designs, state['stratum'])
        if cached is not None:
            return cached
        from src.capability.extensions.stability_followups import verify_scalar_archive
        from src.capability.stability import ranking_error as metric, stability_gate as gate
        panel = state['panel']
        scalar, _ = verify_scalar_archive(self._extractions()[arm], arm,
                                          self._shared()['cohort'], self._shared()['cohort_sha256'])
        mask = np.asarray([row['unit'] not in set(state['lost_backgrounds'])
                           for row in self._registry()])
        if not mask.all():
            scalar = scalar[mask]
        blocks = {**panel['blocks'], 'M': scalar, **state['added_blocks']}
        missing = sorted({name for columns in designs.values() for name in columns}
                         - set(blocks) - {'G'})
        if missing:
            raise SystemExit(f'{arm}: the design names feature blocks the panel does not carry: '
                             f'{missing}')
        readings: dict[str, dict[str, list]] = {name: {} for name in self.STRATA}
        alphas: dict[str, dict[str, list[float]]] = {}
        dimensions: dict[str, int] = {}
        for seed in gate.SPLIT_SEEDS:
            result = gate.fold_predictions(
                panel, blocks, {name: tuple(columns) for name, columns in designs.items()},
                seed=seed, device='cpu')
            alphas[str(seed)] = {name: [fold['alpha'][name] for fold in result['folds']]
                                 for name in designs}
            dimensions = {name: int(result['folds'][0]['dimensions'][name]) for name in designs}
            for name in designs:
                prediction = result['predictions'][name]
                labels, rank = gate.group_spearman(panel['target'], prediction, panel['group'])
                if list(labels) != state['group_order']:
                    raise SystemExit(f'{arm} seed {seed}: the group support of {name} differs')
                if any(value is None or not np.isfinite(value) for value in rank):
                    raise SystemExit(f'{arm} seed {seed}: {name} leaves a group without a defined '
                                     'within-background rank correlation')
                _, transfer = gate.group_errors(panel['target'], prediction, panel['group'],
                                                panel['site'])
                calibrated, _ = metric.family_calibrated(prediction, panel['target'],
                                                         panel['group'], panel['site'])
                _, calibrated_error = gate.group_errors(panel['target'], calibrated,
                                                        panel['group'], panel['site'])
                for stratum, values in (('ranking', np.asarray(rank, dtype=float)),
                                        ('transfer_error', np.asarray(transfer, dtype=float)),
                                        ('calibrated_error', np.asarray(calibrated_error,
                                                                        dtype=float))):
                    readings[stratum].setdefault(name, []).append(values)
        collapsed = {stratum: {name: np.mean(values, axis=0).tolist()
                               for name, values in by_name.items()}
                     for stratum, by_name in readings.items()}
        self._readings.put(state, arm, designs, readings=collapsed, alphas=alphas,
                           dimensions=dimensions)
        return self._readings.get(state, arm, designs, state['stratum'])

    # ------------------------------------------------------------------- pilot

    def archived(self, stratum: str) -> dict:
        """The published per-group values and penalties of the archived grid.

        The grid publishes the control arm's own per-group value and the per-group
        contrast of the joint arm against it, both at the panel's own aggregation
        of three split seeds inside a group. The joint arm's published value is the
        control value plus that signed contrast, which is the only form the grid
        records it in.
        """

        if stratum in self._archived:
            return self._archived[stratum]
        baseline_key, contrast_key, sign = self.STRATA[stratum]
        from src.capability.stability import stability_gate as gate
        state = self.state(stratum, support='full', added={})
        seeds = [str(seed) for seed in gate.SPLIT_SEEDS]
        values, penalties, qualified, arms = {}, {}, {}, []
        for arm in self.arms():
            path = self.published / 'fits' / f'fit_{arm}.json'
            record = json.loads(path.read_text(encoding='utf-8'))
            if record.get('arm') != arm:
                raise SystemExit(f'{path} is not the published record of {arm}')
            if sorted(record['seeds']) != sorted(seeds):
                raise SystemExit(f'{path} covers split seeds {sorted(record["seeds"])}, not '
                                 f'{sorted(seeds)}')
            control, joint = [], []
            for seed in seeds:
                combined = record['seeds'][seed]['contrasts']['combined']
                if list(combined['groups']) != state['group_order']:
                    raise SystemExit(f'{path} seed {seed}: the published group order differs from '
                                     'the realised support')
                undefined = (combined.get('undefined_ranking_groups') or {}).get('profile') or []
                if undefined:
                    raise SystemExit(f'{path} seed {seed}: the published panel leaves '
                                     f'{len(undefined)} groups without a rank correlation')
                base = np.asarray(combined[baseline_key]['profile'], dtype=float)
                contrast = np.asarray(combined[contrast_key], dtype=float)
                control.append(base)
                joint.append(base + sign * contrast)
            values[arm] = {lanes.CONTROL_ARM: np.mean(control, axis=0).tolist(),
                           lanes.JOINT_ARM: np.mean(joint, axis=0).tolist()}
            penalties[arm] = {
                seed: {name: [fold[design] for fold in record['seeds'][seed]['alpha']]
                       for name, design in self.ARCHIVED_DESIGN.items()}
                for seed in seeds}
            qualified[arm] = list(self.qualified(stratum, arm))
            arms.append(arm)
        shape = _backgrounds_per_group(state['backgrounds'], state['groups'])
        self._archived[stratum] = {
            'path': str((self.published / 'fits').relative_to(ROOT)),
            'sha256': sha256_file(self.published / 'panel.json'),
            'arms': arms, 'qualified': qualified, 'values': values, 'penalties': penalties,
            'aggregation': ('the published per-group control value and the published per-group '
                            'contrast of the joint arm against it, split seeds averaged inside '
                            'the group; each group carries the published site-equal weighting of '
                            'its own rows'),
            'joint_arm_reading': (f'the joint arm\'s published per-group value is the control '
                                  f'value plus {sign:+g} times the published contrast, which is '
                                  'the only form the archived grid records it in'),
            'support_shape': shape}
        return self._archived[stratum]


# --------------------------------------------------------------------------- #
# The double-mutant interaction lane.
# --------------------------------------------------------------------------- #

class InteractionAdapter:
    """Measured-single-assisted double-mutant cycle residual, one stratum.

    The frozen plan, cohort, homolog baseline, profile store and indel-exclusion
    declaration are read through the published stage's own declaration rather than
    renamed here, so this lane has one source for what its inputs are. The support
    is the published indel-excluded one: the exclusion is applied before any fit
    sees the panel and the realised counts are asserted against the archived
    declaration rather than assumed.

    **GEMME cannot restrict this lane.** The official multi-mutant score is the sum
    of its per-site scores, so a double-mutant cycle contrast of GEMME scores is
    identically zero; the declaration records the role as non-identifiable and the
    runner's wave gate never asks for that table here. The complete case is
    therefore the coupling support alone, and it is a *different* support from the
    stability lane's although both lanes read one cohort of reference sequences.

    **The per-fold cache is an input, not a product.** ``G`` and the
    measured-single-assisted residual target depend on the fold partition and on
    the measured states, never on an arm's scores, so the published stage built
    them once from one arm's local extraction and every arm reproduces the shared
    input hashes against the written record. This adapter reads that cache for the
    published support and builds its own, under its own support digest, for any
    other -- because the folds, the nuisance and the residual target all move when
    the group universe moves.
    """

    lane = 'interaction'
    STRATUM = 'cycle_squared_error'
    #: The archived design name of each arm of the hierarchy's profile rung.
    ARCHIVED_DESIGN = {lanes.CONTROL_ARM: 'C_G_T', lanes.JOINT_ARM: 'C_G_T+M'}

    def __init__(self, args):
        self.published = _require(args.published, self.lane,
                                  'the published residual panel (--published)')
        self.extraction = _one(args.scores, 'scores', self.lane)
        self.cache = _require(args.cache, self.lane,
                              'the prepared per-fold nuisance and residual cache (--cache)')
        self.out = Path(args.out)
        self._module_cache = None
        self._state: dict = {}
        self._archived: dict = {}
        self._panels: dict = {}

    # ------------------------------------------------------------------ inputs

    def _module(self):
        """The published stage, configured onto this repository and this cache."""

        if self._module_cache is None:
            from scripts.capability.extensions.bidirectional_ablation_20261106.r3 import run_r3
            run_r3.configure(ROOT, self.cache)
            if self.published.resolve() != (ROOT / run_r3.RELATIVE_OLD).resolve():
                raise SystemExit(
                    f'{self.lane}: --published names {self.published} and the published stage\'s '
                    f'declaration names {ROOT / run_r3.RELATIVE_OLD}. The archived panel is not '
                    'relocated here, and the indel-excluded and all-groups supports are never '
                    'substituted for one another')
            if self.extraction.resolve() != (ROOT / 'results/pairwise_epistasis_20260924'
                                             / 'extraction').resolve():
                raise SystemExit(
                    f'{self.lane}: --scores names {self.extraction} and the published stage reads '
                    'its per-arm archives from results/pairwise_epistasis_20260924/extraction')
            self._module_cache = run_r3
        return self._module_cache

    def inputs(self) -> dict[str, str]:
        from scripts.capability.interactions import fit_pairwise_epistasis as fit
        from src.capability.interactions import pairwise_epistasis as mod
        module = self._module()
        record = {str(Path(path).relative_to(ROOT)): fit.digest(path)
                  for path in module.INPUTS.values()}
        record[str((self.published / 'declaration.json').relative_to(ROOT))] = fit.digest(
            self.published / 'declaration.json')
        for arm in self.arms():
            path = self.published / f'residual_{arm}.json'
            record[str(path.relative_to(ROOT))] = fit.digest(path)
            manifest = self.extraction / arm / f'manifest_{arm}.json'
            record[str(manifest.relative_to(ROOT))] = fit.digest(manifest)
        shared = self.cache / 'shared-inputs.json'
        record[str(shared.relative_to(ROOT))] = fit.digest(shared)
        for path in sorted((self.cache / 'cache').glob('*.npz')):
            record[str(path.relative_to(ROOT))] = fit.digest(path)
        for source in (Path(module.__file__), Path(mod.__file__), Path(fit.__file__)):
            record[f'code:{source.stem}'] = sha256_file(source)
        record['note:code:explicit_statistics_lanes'] = sha256_file(Path(lanes.__file__))
        record['code:adapters'] = sha256_file(ADAPTER_SOURCE)
        record['code:runner'] = sha256_file(RUNNER_SOURCE)
        return record

    def arms(self) -> list[str]:
        module = self._module()
        present = [arm for arm in sorted(module.DECL['arms'])
                   if (self.extraction / arm / f'manifest_{arm}.json').is_file()
                   and (self.published / f'residual_{arm}.json').is_file()]
        if not present:
            raise SystemExit(f'{self.lane}: no arm has both an extraction under {self.extraction} '
                             f'and a published residual record under {self.published}')
        return present

    def strata(self) -> list[str]:
        return [self.STRATUM]

    # -------------------------------------------------------------- qualified

    def qualified(self, stratum: str, arm: str | None = None) -> tuple[str, ...]:
        """The archived full control set: the C blocks, ``G`` and ``T``.

        ``T`` is carried by every arm's control set in this lane by declaration
        rather than qualified per arm, so the set does not depend on the arm, and
        the published panel's own design list is what it is read from.
        """

        from src.capability.interactions import pairwise_epistasis as mod
        module = self._module()
        columns = tuple(module.DESIGNS['U_C'])
        if columns != (*mod.C_BLOCKS, 'G', 'T'):
            raise SystemExit(f'{self.lane}: the published control arm is {columns}, not the C '
                             'blocks with G and T')
        if tuple(module.DESIGNS['U_C_M']) != (*columns, 'M'):
            raise SystemExit(f'{self.lane}: the published joint arm is not the control arm plus M')
        return columns

    # ------------------------------------------------------------------- state

    def _load(self, arm: str, groups: set | None):
        """One arm's indel-excluded panel on a declared group universe."""

        from scripts.capability.interactions import fit_pairwise_epistasis as fit
        from src.capability.interactions import pairwise_epistasis as mod
        module = self._module()
        key = (arm, None if groups is None else tuple(sorted(groups)))
        if key in self._panels:
            return self._panels[key]
        plan = json.loads(module.INPUTS['plan'].read_text())
        cohort = json.loads(module.INPUTS['cohort'].read_text())
        profiles, _ = fit.load_profiles(module.INPUTS['profiles'], plan)
        directory = self.extraction / arm
        if fit.digest(directory / f'manifest_{arm}.json') != \
                module.DECL['extractions'][arm]['manifest_sha256']:
            raise SystemExit(f'{arm}: the extraction manifest differs from the archived declaration')
        data, _ = fit.load_arm(directory, arm, plan)
        universe = set(plan['supports']['all_groups']['groups'])
        selected = universe if groups is None else set(groups)
        unknown = sorted(selected - universe)
        if unknown:
            raise SystemExit(f'{self.lane}: {len(unknown)} requested groups are outside the '
                             f'archived group universe, first {unknown[0]!r}')
        panel = fit.build_panel(plan, cohort, profiles, data,
                                json.loads(module.INPUTS['baseline'].read_text()), selected)
        panel, accounting = mod.indel_excluded_support(
            panel, plan, json.loads(module.EXCLUSION.read_text()), selected)
        counts = {'groups': len(set(panel['group'].tolist())), 'cycles': len(panel['group']),
                  'site_pairs': len(set(panel['site_pair'].tolist()))}
        if groups is None and counts != module.DECL['support_counts']:
            raise SystemExit(f'{self.lane}: the realised published support {counts} differs from '
                             f'the archived declaration {module.DECL["support_counts"]}')
        states = panel['states']['y'][panel['cycle_states']]
        if not np.allclose(states[:, 3] - states[:, 1] - states[:, 2] + states[:, 0],
                           panel['epsilon']):
            raise SystemExit(f'{self.lane}: the panel target is not the realised four-state cycle')
        additive = states[:, 1] + states[:, 2] - states[:, 0]
        self._panels[key] = (panel, additive, accounting, counts)
        return self._panels[key]

    def state(self, stratum: str, *, support: str, added: dict) -> dict:
        key = (support, added.get('digest', ''))
        if key not in self._state:
            self._state[key] = self._support(support, added)
        return dict(self._state[key], stratum=stratum, support=support)

    def _support(self, support: str, added: dict) -> dict:
        from src.capability.interactions import pairwise_epistasis as mod
        module = self._module()
        reference_arm = self.arms()[0]
        panel, additive, accounting, counts = self._load(reference_arm, None)
        cycle_key = self._cycle_keys(panel)
        corners = self._corner_keys(panel, None)
        groups = None
        lost: list[str] = []
        if support == 'complete_case':
            if not added:
                raise SystemExit('the complete case needs the added-block values')
            covered = added['covered']
            uncovered = sorted({key for triple in corners for key in triple} - set(covered))
            if uncovered:
                raise SystemExit(
                    f'{self.lane}: {len(uncovered)} scored cycle corners are not covered by the '
                    f'explicit statistics export at all, first {uncovered[0]!r}; the export must '
                    'cover the corners this lane is fitted on before any cycle can be dropped as '
                    'a loss')
            retained = set.intersection(*(set(values) for values in added['values'].values()))
            keep = np.asarray([all(key in retained for key in triple) for triple in corners])
            kept_groups = set(panel['group'][keep].tolist())
            lost = sorted(set(panel['group'].tolist()) - kept_groups)
            partial = sorted({group for group in kept_groups
                              if not keep[panel['group'] == group].all()})
            if partial:
                raise SystemExit(
                    f'{self.lane}: {len(partial)} groups lose some cycles and keep others (first '
                    f'{partial[0]!r}). Every declared loss of this lane is a refusal of a whole '
                    'reference and every group of this plan is one reference, so a part-group loss '
                    'is refused rather than fitted on a cycle set whose corners are not all scored')
            if not kept_groups:
                raise SystemExit(f'{self.lane}: the complete case retains no group')
            groups = kept_groups
            panel, additive, accounting, counts = self._load(reference_arm, groups)
            cycle_key = self._cycle_keys(panel)
            corners = self._corner_keys(panel, groups)
        added_blocks = {}
        for role, values in (added.get('values') or {}).items():
            # The export scores mutants and a row here is a four-corner cycle, so the
            # added block is the cycle contrast of the scored corners, with the wild
            # type's own score zero by construction.
            column = np.asarray([values[double] - values[low] - values[high]
                                 for double, low, high in corners], dtype=float)
            added_blocks[lanes.NEW_BLOCKS[role]] = lanes.prepare_scalar(column,
                                                                        panel['background'])
        return {'groups_selected': None if groups is None else sorted(groups),
                'lost_groups': lost, 'added_blocks': added_blocks,
                'corner_keys': corners,
                'weights': mod.row_weights(panel['site_pair'], panel['group']),
                'additive': additive, 'accounting': accounting, 'counts': counts,
                'cycle_keys': cycle_key,
                'backgrounds': panel['background'], 'groups': panel['group'],
                'group_order': sorted(set(panel['group'].tolist())),
                'cache': self._fold_cache(panel, additive, groups),
                'support_sha256': lanes.identity_digest(cycle_key),
                'rows_offered': module.DECL['support_counts']['cycles'],
                'rows_retained': counts['cycles'],
                'nuisance_rung_limitation': NUISANCE_RUNG_LIMITATION}

    @staticmethod
    def _cycle_keys(panel: dict) -> list[str]:
        """The export's identity of each cycle: its background and its index.

        A cycle is four measured corners rather than one mutation, so the
        coupling table is joined on the cycle the export enumerates, and the
        identity has to name the background as well as the index.
        """

        return [f'{background}|{index}' for background, index
                in zip(panel['background'].tolist(), panel['cycle_index'].tolist())]

    def _corner_keys(self, panel: dict, groups: set | None) -> list[tuple[str, str, str]]:
        """The export's identity of each cycle's three scored corners.

        The export names a row ``<assay>::<mutation>``, so a corner is recovered by
        diffing its own retained measured state against its wild type. The state table
        the panel carries has already dropped the states the indel exclusion removed,
        and the mask is recomputed here from the same declaration and then *checked*
        against the panel's own state table, so a divergence between the two readings
        fails rather than silently misaligning every corner by one row.
        """

        from src.capability.interactions import pairwise_epistasis as mod
        module = self._module()
        plan = json.loads(module.INPUTS['plan'].read_text())
        declaration = json.loads(module.EXCLUSION.read_text())
        selected = set(plan['supports']['all_groups']['groups']) if groups is None else set(groups)
        absent = {(entry['background'], entry['sequence']) for entry in declaration['states']
                  if entry['value_without_indel_rows_kcal_mol'] is None}
        retained = [key for key in mod.panel_state_keys(plan, selected) if key not in absent]
        if [name for name, _ in retained] != panel['states']['background'].tolist():
            raise SystemExit(
                f'{self.lane}: the retained measured states this adapter reads do not line up '
                f'with the {len(panel["states"]["background"])} the panel carries, so no corner '
                'of any cycle could be named reliably')
        wildtype_of = {row['name']: row['wildtype'] for row in plan['backgrounds']}
        mutation = []
        for name, sequence in retained:
            wild = wildtype_of[name]
            if len(sequence) != len(wild):
                raise SystemExit(f'{self.lane}: {name} carries a state of a different length from '
                                 'its wild type, which is not a substitution state')
            mutation.append(':'.join(
                f'{wild[index]}{index + 1}{sequence[index]}'
                for index in range(len(wild)) if wild[index] != sequence[index]))
        keys = []
        for row in panel['cycle_states']:
            wild, low, high, double = (int(index) for index in row)
            if mutation[wild]:
                raise SystemExit(f'{self.lane}: a cycle\'s wild-type corner carries '
                                 f'{mutation[wild]!r}; the reference corner is unmutated by '
                                 'construction and its own score is zero')
            background = str(panel['states']['background'][double])
            keys.append(tuple(f'{background}{SAMPLE_SEPARATOR}{mutation[index]}'
                              for index in (double, low, high)))
        return keys

    def _fold_cache(self, panel: dict, additive, groups) -> dict:
        """The per-fold nuisance and residual cache of this support.

        The published support reads the cache the published stage prepared, with
        its shared-input hashes verified against this panel, so the folds, the
        nuisance and the residual target are the archived ones. Any other support
        gets its own cache under its own support digest, because all three move
        when the group universe moves.
        """

        from scripts.capability.interactions import fit_pairwise_epistasis as fit
        module = self._module()
        if groups is None:
            directory = self.cache / 'cache'
            shared = json.loads((self.cache / 'shared-inputs.json').read_text())
            if module.common_hashes(panel) != shared:
                raise SystemExit(
                    f'{self.lane}: the prepared cache under {self.cache} was built from different '
                    'shared inputs than this panel carries; the nuisance and the residual target '
                    'are not the archived ones and no fit may read them')
            role = 'the published stage\'s own prepared cache, hash-bound to this panel'
        else:
            directory = self.out / 'cache' / lanes.identity_digest(sorted(groups))
            self._prepare_cache(panel, additive, directory)
            role = 'prepared for this support, because the folds and the nuisance move with it'
        entries = {}
        for seed in module.mod.SPLIT_SEEDS:
            for outer, held in enumerate(module.mod.family_folds(
                    panel['group'], module.mod.OUTER_SPLITS, seed)):
                path = directory / f'{seed}-{outer}.npz'
                meta = json.loads(path.with_suffix('.json').read_text())
                if meta['arrays_sha256'] != fit.digest(path):
                    raise SystemExit(f'{path}: the cached fold arrays differ from their digest')
                if list(meta['held_groups']) != list(held):
                    raise SystemExit(f'{path}: the cached outer fold membership is not the one '
                                     'this support produces')
                entries[seed, outer] = {'path': path, 'meta': meta,
                                        'sha256': meta['arrays_sha256']}
        return {'directory': str(directory), 'role': role, 'entries': entries}

    def _prepare_cache(self, panel: dict, additive, directory: Path) -> None:
        """Build the per-fold nuisance and residual cache of a new support.

        This is the published ``prepare`` step on a different group universe: the
        same outer and inner partitions, the same calibration partition, the same
        cross-fitted nuisance and the same training-only residual target. No arm
        score reaches any of it.
        """

        from scripts.capability.interactions import fit_pairwise_epistasis as fit
        from scripts.capability.interactions import refit_likelihood_on_residual as oldfit
        module = self._module()
        mod = module.mod
        groups = panel['group']
        directory.mkdir(parents=True, exist_ok=True)
        shared = directory / 'shared-inputs.json'
        if not shared.is_file():
            shared.write_text(json.dumps(module.common_hashes(panel), indent=2, sort_keys=True)
                              + '\n')
        for seed in mod.SPLIT_SEEDS:
            for outer, held in enumerate(mod.family_folds(groups, mod.OUTER_SPLITS, seed)):
                destination = directory / f'{seed}-{outer}.npz'
                if destination.exists() and destination.with_suffix('.json').is_file():
                    continue
                train = np.flatnonzero(~np.isin(groups, held))
                test = np.flatnonzero(np.isin(groups, held))
                inner = mod.family_folds(groups[train], mod.INNER_SPLITS, seed + 100 + outer)
                arrays = {}
                meta = {'seed': seed, 'outer': outer, 'held_groups': held, 'inner': []}
                for index, validation in enumerate(inner):
                    fitting = train[~np.isin(groups[train], validation)]
                    valid = train[np.isin(groups[train], validation)]
                    calibration = mod.family_folds(groups[fitting], mod.INNER_SPLITS,
                                                   seed + 1000 + outer * 10 + index)
                    excluded = np.asarray(list(held) + list(validation))
                    block, diagnostic = mod.nuisance_cycle(panel['states'], panel['cycle_states'],
                                                           excluded, calibration, device='cpu')
                    fit_y, valid_y = oldfit.residual_targets(panel['epsilon'], additive,
                                                             fitting, valid)
                    arrays.update({f'G_{index}': block, f'fit_{index}': fitting,
                                   f'valid_{index}': valid, f'yfit_{index}': fit_y,
                                   f'yvalid_{index}': valid_y})
                    meta['inner'].append({'validation_groups': validation,
                                          'calibration_folds': calibration,
                                          'G_diagnostic': diagnostic})
                block, diagnostic = mod.nuisance_cycle(panel['states'], panel['cycle_states'],
                                                       np.asarray(held), inner, device='cpu')
                train_y, test_y = oldfit.residual_targets(panel['epsilon'], additive, train, test)
                arrays.update(G=block, train=train, test=test, ytrain=train_y, ytest=test_y)
                np.savez_compressed(destination, **arrays)
                meta.update(G_diagnostic=diagnostic, arrays_sha256=fit.digest(destination),
                            common_inputs_sha256=fit.digest(shared))
                destination.with_suffix('.json').write_text(
                    json.dumps(meta, indent=2, default=lambda value: np.asarray(value).tolist())
                    + '\n')

    # --------------------------------------------------------------------- fit

    def fit(self, state: dict, arm: str, designs: dict[str, tuple[str, ...]]) -> dict:
        """One cell: per-group cycle squared error of the three arms, seeds averaged.

        The loop is the published stage's own: the cached outer and inner
        partitions, the cached cross-fitted nuisance, the cached training-only
        residual target, the same penalty grid and the same tie rule. Only the
        declared column set of each arm changes.
        """

        module = self._module()
        mod = module.mod
        panel, additive, _, _ = self._load(arm, state['groups_selected'] and
                                           set(state['groups_selected']))
        groups, weights = panel['group'], state['weights']
        if list(self._cycle_keys(panel)) != list(state['cycle_keys']):
            raise SystemExit(f'{arm}: the realised cycle support differs from the state it is '
                             'fitted on')
        blocks_base = {**panel['blocks'], **state['added_blocks']}
        missing = sorted({name for columns in designs.values() for name in columns}
                         - set(blocks_base) - {'G'})
        if missing:
            raise SystemExit(f'{arm}: the design names feature blocks the panel does not carry: '
                             f'{missing}')
        names = list(designs)
        per_seed, alphas, dimensions = [], {}, {}
        for seed in mod.SPLIT_SEEDS:
            predictions = {name: np.full(len(groups), np.nan) for name in names}
            target = np.full(len(groups), np.nan)
            folds: dict[str, list[float]] = {name: [] for name in names}
            for outer, held in enumerate(mod.family_folds(groups, mod.OUTER_SPLITS, seed)):
                entry = state['cache']['entries'][seed, outer]
                with np.load(entry['path'], allow_pickle=False) as cached:
                    losses = {name: np.zeros(len(mod.ALPHAS)) for name in names}
                    for index, inner in enumerate(entry['meta']['inner']):
                        fitting, valid = cached[f'fit_{index}'], cached[f'valid_{index}']
                        blocks = dict(blocks_base, G=cached[f'G_{index}'])
                        if np.isin(groups[fitting],
                                   list(held) + list(inner['validation_groups'])).any():
                            raise SystemExit('an outer held group enters an inner training '
                                             'partition')
                        for name in names:
                            matrix = np.column_stack([blocks[key] for key in designs[name]])
                            predicted = mod.ridge_predict(matrix[fitting], cached[f'yfit_{index}'],
                                                          weights[fitting], matrix[valid],
                                                          mod.ALPHAS, 'cpu')
                            losses[name] += len(inner['validation_groups']) * (
                                weights[valid][:, None]
                                * (predicted - cached[f'yvalid_{index}'][:, None]) ** 2).sum(0)
                    train, test = cached['train'], cached['test']
                    target[test] = cached['ytest']
                    blocks = dict(blocks_base, G=cached['G'])
                    for name in names:
                        alpha = mod.ALPHAS[mod._select_alpha(losses[name])]
                        matrix = np.column_stack([blocks[key] for key in designs[name]])
                        predictions[name][test] = mod.ridge_predict(
                            matrix[train], cached['ytrain'], weights[train], matrix[test],
                            [alpha], 'cpu')[:, 0]
                        folds[name].append(alpha)
                        dimensions[name] = int(matrix.shape[1])
            if not np.isfinite(target).all() or not all(np.isfinite(value).all()
                                                        for value in predictions.values()):
                raise SystemExit(f'{arm} seed {seed}: incomplete out-of-fold predictions')
            alphas[str(seed)] = folds
            errors = {}
            for name in names:
                labels, value = mod.group_errors(target, predictions[name], groups,
                                                 panel['site_pair'])
                if list(labels) != state['group_order']:
                    raise SystemExit(f'{arm} seed {seed}: the group support of {name} differs')
                errors[name] = np.asarray(value, dtype=float)
            per_seed.append(errors)
        values = {name: np.mean([entry[name] for entry in per_seed], axis=0).tolist()
                  for name in names}
        return {'values': values, 'alphas': alphas, 'dimensions': dimensions}

    # ------------------------------------------------------------------- pilot

    def archived(self, stratum: str) -> dict:
        """The published per-group cycle errors and penalties of the residual panel."""

        if stratum in self._archived:
            return self._archived[stratum]
        from scripts.capability.interactions import fit_pairwise_epistasis as fit
        module = self._module()
        seeds = [str(seed) for seed in module.mod.SPLIT_SEEDS]
        state = self.state(stratum, support='full', added={})
        values, penalties, qualified, arms = {}, {}, {}, []
        for arm in self.arms():
            path = self.published / f'residual_{arm}.json'
            record = json.loads(path.read_text(encoding='utf-8'))
            if list(record['groups_order']) != state['group_order']:
                raise SystemExit(f'{path}: the published group order differs from the realised '
                                 'support')
            if record['declaration_sha256'] != fit.digest(self.published / 'declaration.json'):
                raise SystemExit(f'{path} is not bound to the declaration beside it')
            if sorted(record['seeds']) != sorted(seeds):
                raise SystemExit(f'{path} covers split seeds {sorted(record["seeds"])}, not '
                                 f'{sorted(seeds)}')
            values[arm] = {}
            for name, design in self.ARCHIVED_DESIGN.items():
                drawn = [np.asarray(record['seeds'][seed]['per_group_mse'][design], dtype=float)
                         for seed in seeds]
                values[arm][name] = np.mean(drawn, axis=0).tolist()
            penalties[arm] = {
                seed: {name: [fold['alpha'][design]
                              for fold in record['seeds'][seed]['folds']]
                       for name, design in self.ARCHIVED_DESIGN.items()}
                for seed in seeds}
            qualified[arm] = list(self.qualified(stratum, arm))
            arms.append(arm)
        shape = _backgrounds_per_group(state['backgrounds'], state['groups'])
        self._archived[stratum] = {
            'path': str(self.published.relative_to(ROOT)),
            'sha256': fit.digest(self.published / 'declaration.json'),
            'arms': arms, 'qualified': qualified, 'values': values, 'penalties': penalties,
            'aggregation': ('the published per-group cycle squared error of each arm, split seeds '
                            'averaged inside the group; each group carries the published '
                            'site-pair-equal weighting of its own cycles'),
            'support_shape': shape}
        return self._archived[stratum]


#: Every implemented adapter, by the lane it adapts.
ADAPTERS = {adapter.lane: adapter for adapter in
            (PhenotypeAdapter, AbundanceAdapter, StabilityAdapter, InteractionAdapter)}
