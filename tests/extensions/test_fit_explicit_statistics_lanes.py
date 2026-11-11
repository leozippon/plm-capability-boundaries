"""The lane runner: the wave gate, the resume rule, the join, the pilot and the panel.

The adapter is stubbed here, because what this file tests is the machinery around a
lane's fitter rather than any lane's fitter. One test exercises the real phenotype
adapter's join key against the cohort export's own rule, and one checks the refusal a
lane without an adapter gets.

The pilot tests matter more than their size suggests. The gate is a precondition of
every panel, and it is worth nothing unless it compares at the published aggregation
and fails when the refit disagrees, so both halves are exercised against a stub whose
archive is under the test's control.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.capability.extensions.explicit_statistics_lanes import (
    explicit_statistics_lane_adapters as adapters)
from scripts.capability.extensions.explicit_statistics_lanes import (
    fit_explicit_statistics_lanes as runner)
from src.capability.extensions import explicit_statistics_lanes as lanes

STRATUM = 'activity_venus-ranking'
QUALIFIED = ('ident', 'geom', 'prof', 'prof2')
GROUPS = [f'g{index:02d}' for index in range(12)]


def cohort_rows(references=('r1', 'r2'), per_reference=3):
    rows = []
    for reference in references:
        for index in range(per_reference):
            rows.append({'sample_id': f'{reference}::A{index + 1}C', 'assay': reference,
                         'wildtype_id': reference, 'mutation': f'A{index + 1}C',
                         'original_index': len(rows), 'n_substitutions': 1})
    return rows


def score_table(role, rows):
    source = lanes.BLOCK_SOURCE[role]
    status = 'fitted' if role == 'couplings' else 'scored'
    return {'schema': source['schema'], 'counts': {'complete': True, 'samples': len(rows)},
            'refused': {}, 'samples': [
                {'sample_id': row['sample_id'], 'assay': row['assay'],
                 'wildtype_id': row['wildtype_id'], 'mutation': row['mutation'],
                 'status': status, source['column']: float(index) + 0.5}
                for index, row in enumerate(rows)]}


class StubAdapter:
    """A lane adapter that fits nothing: it returns declared synthetic per-group values."""

    lane = 'phenotype'
    calls: list = []

    def __init__(self, args):
        self.args = args
        self.digest = Path(args.qualification).read_text() if args.qualification else 'fixed'

    #: What this stub records beside a fit without being bound to it.
    note: str = 'first'

    def inputs(self):
        return {'stub': runner.digest_text(self.digest),
                f'{runner.RECORDED_PREFIX}limitation': StubAdapter.note}

    def arms(self):
        return list(lanes.lane('phenotype').arms)

    def strata(self):
        return [STRATUM]

    #: Arms whose archive qualified a further nuisance, as the abundance archive does.
    tokenised: tuple = ()
    #: What this stub's archive disagrees with its own fit by, and whether its
    #: published penalties drift. Both default to agreement.
    perturb: float = 0.
    penalty_drift: bool = False

    def qualified(self, stratum, arm=None):
        assert stratum == STRATUM
        if arm is not None and arm in StubAdapter.tokenised:
            return (*QUALIFIED, 'T')
        return QUALIFIED

    def state(self, stratum, *, support, added):
        return {'group_order': list(GROUPS), 'support_sha256': f'{support}-digest',
                'added': sorted(added.get('values', {})), 'support': support}

    def fit(self, state, arm, designs):
        StubAdapter.calls.append((state['support'], arm, tuple(sorted(designs))))
        # Each arm carries its own per-group deviation, so every paired contrast varies
        # across the resampling units as a real per-group error does. A stub whose
        # contrasts are exactly constant has no studentised statistic at all, and the
        # shared band helper refuses such a family rather than reporting it resolved.
        rng = np.random.default_rng(sum(arm.encode()))
        groups = len(state['group_order'])
        joint = rng.normal(0.5, 0.01, size=groups)
        values = {lanes.CONTROL_ARM: joint + rng.normal(-0.05, 0.004, size=groups),
                  lanes.MODEL_ARM: joint + rng.normal(-0.4, 0.01, size=groups),
                  lanes.JOINT_ARM: joint}
        return {'values': {name: value.tolist() for name, value in values.items()},
                'alphas': {'1': {name: [1.0] for name in designs}},
                'dimensions': {name: len(columns) for name, columns in designs.items()}}

    def archived(self, stratum):
        """A published archive that agrees with this stub's own fit, or does not."""

        state = self.state(stratum, support='full', added={})
        values, penalties, qualified = {}, {}, {}
        for arm in self.arms():
            columns = self.qualified(stratum, arm)
            designs = lanes.arm_designs('phenotype', lanes.ARCHIVED_RUNG, columns)
            fitted = self.fit(state, arm, designs)
            values[arm] = {name: [value + StubAdapter.perturb
                                  for value in fitted['values'][name]]
                           for name in (lanes.CONTROL_ARM, lanes.JOINT_ARM)}
            penalties[arm] = {'1': {name: [2.0 if StubAdapter.penalty_drift else 1.0]
                                    for name in (lanes.CONTROL_ARM, lanes.JOINT_ARM)}}
            qualified[arm] = list(columns)
        StubAdapter.calls = []
        return {'path': 'stub/panel.json', 'sha256': 'f' * 64, 'arms': self.arms(),
                'qualified': qualified, 'values': values, 'penalties': penalties,
                'aggregation': 'stub: one value per group',
                'support_shape': {'groups': len(GROUPS), 'backgrounds': len(GROUPS),
                                  'most_backgrounds_in_one_group': 1,
                                  'exercises_aggregation': False, 'reading': 'stub support'}}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setitem(runner.ADAPTERS, 'phenotype', StubAdapter)
    StubAdapter.calls = []
    StubAdapter.tokenised = ()
    StubAdapter.perturb = 0.
    StubAdapter.penalty_drift = False
    StubAdapter.note = 'first'
    rows = cohort_rows()
    export = tmp_path / 'export'
    (export / 'inputs').mkdir(parents=True)
    (export / lanes.ROWS_FILE).write_text(json.dumps({'schema': lanes.ROWS_SCHEMA,
                                                      'samples': rows}))
    (export / lanes.REFUSALS_FILE).write_text(json.dumps(
        {'schema': lanes.REFUSALS_SCHEMA, 'count': 0, 'refused': {}}))
    (tmp_path / 'couplings.json').write_text(json.dumps(score_table('couplings', rows)))
    (tmp_path / 'gemme.json').write_text(json.dumps(score_table('gemme', rows)))
    (tmp_path / 'qualification').write_text('first')
    return {'root': tmp_path, 'export': export, 'rows': rows,
            'out': tmp_path / 'out',
            'base': ['--lane', 'phenotype', '--out', str(tmp_path / 'out'),
                     '--export', str(export), '--qualification', str(tmp_path / 'qualification')],
            'wave': ['--couplings', str(tmp_path / 'couplings.json'),
                     '--gemme', str(tmp_path / 'gemme.json')]}


# --------------------------------------------------------------------------- #
# The wave gate.
# --------------------------------------------------------------------------- #

def test_an_absent_wave_leaves_the_full_support_runnable(workspace):
    wave = runner.read_wave('phenotype', couplings=None, gemme=None)
    assert wave['complete'] is False and wave['pending'] == ['couplings', 'gemme']
    supports = runner.runnable_supports(wave)
    assert supports['full']['runnable'] is True
    assert supports['complete_case']['runnable'] is False
    assert 'pending blocks couplings, gemme' in supports['complete_case']['reason']


def test_a_landed_wave_opens_the_complete_case(workspace):
    wave = runner.read_wave('phenotype', couplings=workspace['root'] / 'couplings.json',
                            gemme=workspace['root'] / 'gemme.json')
    assert wave['complete'] is True and wave['landed'] == ['couplings', 'gemme']
    assert runner.runnable_supports(wave)['complete_case']['runnable'] is True
    assert all(entry['state'] == 'landed' for entry in wave['status'].values())


def test_an_incomplete_table_is_pending_with_the_readers_own_reason(workspace, tmp_path):
    table = score_table('couplings', workspace['rows'])
    table['counts']['complete'] = False
    path = tmp_path / 'partial.json'
    path.write_text(json.dumps(table))
    wave = runner.read_wave('phenotype', couplings=path, gemme=None)
    assert wave['status']['couplings']['state'] == 'pending'
    assert 'counts.complete=False' in wave['status']['couplings']['reason']


def test_a_block_the_lane_cannot_identify_is_not_part_of_its_wave():
    wave = runner.read_wave('interaction', couplings=None, gemme=None)
    assert wave['status']['gemme']['state'] == 'not_required'
    assert 'sum of its per-site scores' in wave['status']['gemme']['reason']
    assert wave['required'] == ['couplings'] and wave['pending'] == ['couplings']


# --------------------------------------------------------------------------- #
# The join.
# --------------------------------------------------------------------------- #

def test_added_values_separates_a_loss_from_an_absent_row(workspace):
    wave = runner.read_wave('phenotype', couplings=workspace['root'] / 'couplings.json',
                            gemme=workspace['root'] / 'gemme.json')
    support = lanes.complete_case('phenotype', workspace['rows'], wave['tables'])
    added = runner.added_values(wave, support)
    assert sorted(added['values']) == ['couplings', 'gemme']
    assert set(added['values']['couplings']) == set(support['sample_ids'])
    assert added['covered'] == {row['sample_id'] for row in workspace['rows']}
    assert added['digest'] == runner.digest_text('\n'.join(sorted(support['sample_ids'])))
    short = dict(wave, tables={role: dict(table, samples=table['samples'][:-1])
                               for role, table in wave['tables'].items()})
    with pytest.raises(SystemExit, match='of the 6 retained rows'):
        runner.added_values(short, support)


def test_the_phenotype_join_key_is_the_cohort_exports_own_rule():
    from src.capability.baselines import cohorts
    from src.capability.extensions import phenotype_breadth as breadth

    wildtype = 'MKVAA'
    row = breadth.PhenotypeRow(
        cohort='activity_venus', row_id='x#1', background='urn:bg-1', unit='u' * 64,
        wildtype=wildtype, mutant_sequence='MKAAA', position=3, wt_aa='V', mt_aa='A',
        label=1.0, direction=1, label_unit=None, source_sha256='s' * 64)
    reference = cohorts.Reference(wildtype_id='activity_venus:abc', sequence=wildtype,
                                 query_id='activity_venus:abc', aliases=('activity_venus:abc',))
    exported = cohorts.rows_of([reference], {'activity_venus:abc': [('urn:bg-1', 'V3A')]})
    instance = adapters.PhenotypeAdapter.__new__(adapters.PhenotypeAdapter)
    assert instance.sample_key(row) == exported[0]['sample_id']


def test_a_lane_without_an_adapter_is_refused_with_what_one_must_supply():
    class Args:
        lane = 'a-lane-with-no-adapter'

    with pytest.raises(SystemExit) as error:
        runner.adapter(Args())
    message = str(error.value)
    assert 'no adapter' in message and 'never re-qualified' in message
    assert 'per-group values for the three arms' in message
    assert str(sorted(runner.ADAPTERS)) in message


def test_every_declared_lane_has_an_adapter():
    assert sorted(runner.ADAPTERS) == sorted(lanes.LANES)


# --------------------------------------------------------------------------- #
# Fitting, resume and the panel.
# --------------------------------------------------------------------------- #

def test_the_runner_fits_the_declared_family_and_resumes_without_refitting(workspace):
    arguments = workspace['base'] + workspace['wave'] + ['--support', 'complete_case']
    assert runner.main(['fit'] + arguments) == 0
    report = json.loads((workspace['out'] / 'fit.json').read_text())
    declared = lanes.lane('phenotype')
    assert report['cells_fitted'] == len(declared.rungs) * len(declared.arms) == 50
    assert report['cells_reused'] == 0
    assert len(StubAdapter.calls) == 50
    # The support record of the complete case is written beside the cells.
    support = json.loads((workspace['out'] / 'support.json').read_text())
    assert support['rows_retained'] == 6 and support['schema'] == lanes.SUPPORT_SCHEMA
    # A rerun refits nothing.
    StubAdapter.calls = []
    assert runner.main(['fit'] + arguments) == 0
    report = json.loads((workspace['out'] / 'fit.json').read_text())
    assert report['cells_reused'] == 50 and report['cells_fitted'] == 0
    assert StubAdapter.calls == []


def test_a_changed_input_is_refused_rather_than_mixed_into_one_family(workspace):
    arguments = workspace['base'] + workspace['wave'] + ['--support', 'complete_case']
    assert runner.main(['fit'] + arguments) == 0
    (workspace['root'] / 'qualification').write_text('second')
    with pytest.raises(SystemExit, match='fitted from different inputs'):
        runner.main(['fit'] + arguments)
    StubAdapter.calls = []
    assert runner.main(['fit'] + arguments + ['--refit']) == 0
    assert len(StubAdapter.calls) == 50


def test_the_pending_part_of_a_partial_wave_is_named_and_the_rest_is_fitted(workspace):
    assert runner.main(['fit'] + workspace['base']) == 0
    report = json.loads((workspace['out'] / 'fit.json').read_text())
    # The full-support sensitivity is fitted; the complete-case rungs are named pending.
    assert {entry['support'] for entry in report['fitted']} == {'full'}
    assert report['cells_fitted'] == len(lanes.lane('phenotype').arms) == 10
    skipped = {entry['rung'] for entry in report['skipped']}
    assert skipped == set(lanes.lane('phenotype').rungs)
    assert all('pending blocks' in entry['reason'] for entry in report['skipped'])
    assert not (workspace['out'] / 'support.json').exists()


def test_the_panel_refuses_a_family_whose_stratum_has_no_pilot(workspace):
    arguments = workspace['base'] + workspace['wave']
    assert runner.main(['fit'] + arguments + ['--support', 'complete_case']) == 0
    with pytest.raises(SystemExit, match='no pilot at'):
        runner.main(['panel'] + arguments)


def test_the_panel_refuses_a_family_the_pilot_did_not_cover_on_every_arm(workspace):
    arguments = workspace['base'] + workspace['wave']
    assert runner.main(['fit'] + arguments + ['--support', 'complete_case']) == 0
    arms = list(lanes.lane('phenotype').arms)
    assert runner.main(['pilot'] + workspace['base'] + ['--arm', arms[0]]) == 0
    with pytest.raises(SystemExit, match='are unpiloted'):
        runner.main(['panel'] + arguments)


def test_the_panel_refuses_a_pilot_that_did_not_pass(workspace):
    arguments = workspace['base'] + workspace['wave']
    assert runner.main(['fit'] + arguments + ['--support', 'complete_case']) == 0
    StubAdapter.perturb = 1e-6
    assert runner.main(['pilot'] + workspace['base']) == 1
    StubAdapter.perturb = 0.
    with pytest.raises(SystemExit, match='records status'):
        runner.main(['panel'] + arguments)


def test_the_panel_assembles_a_complete_family_and_names_an_incomplete_one(workspace):
    arguments = workspace['base'] + workspace['wave']
    assert runner.main(['fit'] + arguments + ['--support', 'complete_case']) == 0
    assert runner.main(['pilot'] + workspace['base']) == 0
    assert runner.main(['panel'] + arguments) == 0
    report = json.loads((workspace['out'] / 'panels.json').read_text())
    written = report['written']
    assert list(written) == ['panels/activity_venus-ranking-likelihood-complete_case.json']
    assert written[list(written)[0]]['family'] == 100
    # The full-support sensitivity family has no cells yet, and is named rather than
    # written as a partial family.
    assert [entry['role'] for entry in report['pending']] == ['sensitivity']
    assert report['pending'][0]['cells'] == 0 and report['pending'][0]['expected'] == 10
    panel = json.loads((workspace['out'] / list(written)[0]).read_text())
    assert panel['schema'] == lanes.PANEL_SCHEMA and panel['status'] == 'complete'
    # The family records the gate it was written under, with its measured difference.
    assert panel['pilot']['worst_per_group_difference'] == 0.
    assert panel['pilot']['tolerance'] == runner.PILOT_TOLERANCE
    assert panel['pilot']['arms'] == len(lanes.lane('phenotype').arms)
    assert panel['support'] == 'complete_case'
    assert panel['support_sha256'] == 'complete_case-digest'
    assert panel['forward_bound']['columns']
    assert panel['wave']['couplings']['state'] == 'landed'
    assert sum(panel['state_counts'][name] for name in lanes.STATES) == 100


def test_the_panel_refuses_cells_from_two_supports(workspace):
    arguments = workspace['base'] + workspace['wave'] + ['--support', 'complete_case']
    assert runner.main(['fit'] + arguments) == 0
    assert runner.main(['pilot'] + workspace['base']) == 0
    path = runner.cell_path(workspace['out'], STRATUM, 'likelihood', 'complete_case',
                            'nuisance', lanes.lane('phenotype').arms[0])
    record = json.loads(path.read_text())
    record['support_sha256'] = 'another-support'
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match='different supports'):
        runner.main(['panel'] + arguments)


def test_a_cell_file_of_another_schema_is_refused(workspace):
    arguments = workspace['base'] + workspace['wave'] + ['--support', 'complete_case']
    assert runner.main(['fit'] + arguments) == 0
    path = runner.cell_path(workspace['out'], STRATUM, 'likelihood', 'complete_case',
                            'nuisance', lanes.lane('phenotype').arms[0])
    record = json.loads(path.read_text())
    record['schema'] = 'explicit_statistics_lane_cell_v2'
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match='is not'):
        runner.main(['fit'] + arguments)


# --------------------------------------------------------------------------- #
# The pilot: the precondition of every panel.
# --------------------------------------------------------------------------- #

def test_the_pilot_reproduces_the_published_values_and_states_what_it_exercised(workspace):
    assert runner.main(['pilot'] + workspace['base']) == 0
    record = json.loads((workspace['out'] / 'pilot.json').read_text())
    assert record['status'] == 'passed' and record['tolerance'] == runner.PILOT_TOLERANCE
    stratum = record['strata'][STRATUM]
    assert stratum['worst_per_group_difference'] == 0.
    assert stratum['groups'] == len(GROUPS)
    assert stratum['aggregation'] == 'stub: one value per group'
    # A support whose every group holds one background cannot test the aggregation,
    # and the record says so rather than letting a reader assume it did.
    assert stratum['support_shape']['exercises_aggregation'] is False
    arm = stratum['arms'][0]
    assert stratum['arms_reproduced'][arm]['penalties_reproduced'] is True
    assert stratum['arms_reproduced'][arm]['control_set'] == list(QUALIFIED)


def test_the_pilot_fails_on_a_published_value_it_cannot_reproduce(workspace):
    StubAdapter.perturb = 1e-6
    assert runner.main(['pilot'] + workspace['base']) == 1
    record = json.loads((workspace['out'] / 'pilot.json').read_text())
    assert record['status'] == 'failed'
    assert 'the fitting path is not the published one' in record['reason']
    stratum = record['strata'][STRATUM]
    assert stratum['worst_per_group_difference'] == pytest.approx(1e-6)


def test_a_difference_just_inside_the_tolerance_still_passes(workspace):
    StubAdapter.perturb = runner.PILOT_TOLERANCE / 2
    assert runner.main(['pilot'] + workspace['base']) == 0
    record = json.loads((workspace['out'] / 'pilot.json').read_text())
    assert record['status'] == 'passed'


def test_the_pilot_fails_on_a_penalty_the_refit_did_not_choose(workspace):
    StubAdapter.penalty_drift = True
    assert runner.main(['pilot'] + workspace['base']) == 1
    record = json.loads((workspace['out'] / 'pilot.json').read_text())
    assert record['status'] == 'failed'
    stratum = record['strata'][STRATUM]
    assert stratum['worst_per_group_difference'] == 0.
    assert all(not entry['penalties_reproduced']
               for entry in stratum['arms_reproduced'].values())


def test_the_pilot_refuses_a_published_control_set_that_is_not_the_arms_own(workspace,
                                                                           monkeypatch):
    # The archive says one thing about an arm's columns and the adapter another:
    # refitting the wrong design would reproduce nothing and must not be attempted.
    original = StubAdapter.archived

    def drifted(self, stratum):
        record = original(self, stratum)
        record['qualified'][self.arms()[0]] = ['ident']
        return record

    monkeypatch.setattr(StubAdapter, 'archived', drifted)
    with pytest.raises(SystemExit) as error:
        runner.main(['pilot'] + workspace['base'])
    assert 'was fitted over' in str(error.value)


# --------------------------------------------------------------------------- #
# The arm-conditional control set.
# --------------------------------------------------------------------------- #

def test_each_arm_is_fitted_on_its_own_archived_control_set(workspace):
    arms = list(lanes.lane('phenotype').arms)
    StubAdapter.tokenised = tuple(arms[:3])
    # The pilot fits the archived rung on the full support; the sensitivity rung is
    # fitted beside it, and an arm's own nuisance column has to reach both.
    assert runner.main(['pilot'] + workspace['base']) == 0
    assert runner.main(['fit'] + workspace['base'] + ['--support', 'full']) == 0
    for arm in arms:
        tokenised = arm in StubAdapter.tokenised
        archived = list(QUALIFIED) + (['T'] if tokenised else [])
        nuisance = ['ident', 'geom'] + (['T'] if tokenised else [])
        for rung, expected in ((lanes.ARCHIVED_RUNG, archived), ('nuisance', nuisance)):
            cell = json.loads(runner.cell_path(workspace['out'], STRATUM, 'likelihood', 'full',
                                               rung, arm).read_text())
            assert cell['qualified'] == archived
            assert cell['designs'][lanes.CONTROL_ARM] == expected
            assert cell['designs'][lanes.JOINT_ARM] == expected + ['M']
            assert cell['designs'][lanes.MODEL_ARM] == ['M']


def test_the_plan_records_the_arms_whose_control_set_differs(workspace):
    arms = list(lanes.lane('phenotype').arms)
    StubAdapter.tokenised = tuple(arms[:3])
    assert runner.main(['plan'] + workspace['base']) == 0
    record = json.loads((workspace['out'] / 'plan.json').read_text())
    entry = record['strata'][STRATUM]
    assert entry['qualified_control_set'] == list(QUALIFIED)
    assert sorted(entry['arm_conditional_control_set']['arms']) == sorted(arms[:3])
    assert entry['arm_conditional_control_set']['count'] == 3
    assert entry['control_set_per_arm'][arms[0]] == list(QUALIFIED) + ['T']
    assert entry['control_set_per_arm'][arms[-1]] == list(QUALIFIED)


# --------------------------------------------------------------------------- #
# What a cell is bound to, and what it merely records.
# --------------------------------------------------------------------------- #

def test_a_recorded_only_input_does_not_invalidate_a_fit(workspace):
    arguments = workspace['base'] + workspace['wave'] + ['--support', 'complete_case']
    assert runner.main(['fit'] + arguments) == 0
    assert json.loads((workspace['out'] / 'fit.json').read_text())['cells_fitted'] == 50
    # Recording a different limitation cannot move a number, so it costs no refit.
    StubAdapter.note = 'a limitation recorded after the fact'
    StubAdapter.calls = []
    assert runner.main(['fit'] + arguments) == 0
    report = json.loads((workspace['out'] / 'fit.json').read_text())
    assert report['cells_reused'] == 50 and report['cells_fitted'] == 0
    assert StubAdapter.calls == []
    # The new wording is nonetheless recorded in the run's own inputs.
    assert report['inputs_sha256'][f'{runner.RECORDED_PREFIX}limitation'] == StubAdapter.note


def test_a_binding_input_still_invalidates_a_fit(workspace):
    arguments = workspace['base'] + workspace['wave'] + ['--support', 'complete_case']
    assert runner.main(['fit'] + arguments) == 0
    (workspace['root'] / 'qualification').write_text('second')
    with pytest.raises(SystemExit, match='fitted from different inputs'):
        runner.main(['fit'] + arguments)


def test_a_cell_is_bound_to_the_declared_contract_of_its_own_support(workspace, monkeypatch):
    arguments = workspace['base'] + workspace['wave']
    assert runner.main(['fit'] + arguments + ['--support', 'complete_case']) == 0
    assert runner.main(['fit'] + arguments + ['--support', 'full']) == 0
    # The added-block admission vocabulary decides which rows a complete-case fit sees
    # and cannot reach a full-support one, so it invalidates exactly one of the two.
    status = dict(lanes.ROW_STATUS)
    status['gemme'] = {**status['gemme'], 'some_new_upstream_status': lanes.REFUSED}
    monkeypatch.setattr(lanes, 'ROW_STATUS', status)
    StubAdapter.calls = []
    assert runner.main(['fit'] + arguments + ['--support', 'full']) == 0
    assert json.loads((workspace['out'] / 'fit.json').read_text())['cells_reused'] == 10
    assert StubAdapter.calls == []
    with pytest.raises(SystemExit, match='fitted from different inputs'):
        runner.main(['fit'] + arguments + ['--support', 'complete_case'])


def test_the_panel_gate_also_ignores_a_recorded_only_input(workspace):
    arguments = workspace['base'] + workspace['wave']
    assert runner.main(['fit'] + arguments + ['--support', 'complete_case']) == 0
    assert runner.main(['pilot'] + workspace['base']) == 0
    StubAdapter.note = 'landed after the pilot ran'
    assert runner.main(['panel'] + arguments) == 0
    report = json.loads((workspace['out'] / 'panels.json').read_text())
    assert 'panels/activity_venus-ranking-likelihood-complete_case.json' in report['written']


def test_the_complete_case_needs_the_exports_refusals_document(workspace):
    (workspace['export'] / lanes.REFUSALS_FILE).unlink()
    arguments = workspace['base'] + workspace['wave'] + ['--support', 'complete_case']
    with pytest.raises(SystemExit, match='enumerates the references it refused'):
        runner.main(['fit'] + arguments)
