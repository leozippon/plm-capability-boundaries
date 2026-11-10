"""The lane runner: the wave gate, the resume rule, the join and the panel assembly.

The adapter is stubbed here, because what this file tests is the machinery around a
lane's fitter rather than any lane's fitter. One test exercises the real phenotype
adapter's join key against the cohort export's own rule, and one checks the refusal a
lane without an adapter gets.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.capability.extensions import fit_explicit_statistics_lanes as runner
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

    def inputs(self):
        return {'stub': runner.digest_text(self.digest)}

    def arms(self):
        return list(lanes.lane('phenotype').arms)

    def strata(self):
        return [STRATUM]

    def qualified(self, stratum):
        assert stratum == STRATUM
        return QUALIFIED

    def state(self, stratum, *, support, added):
        return {'group_order': list(GROUPS), 'support_sha256': f'{support}-digest',
                'added': sorted(added.get('values', {})), 'support': support}

    def fit(self, state, arm, designs):
        StubAdapter.calls.append((state['support'], arm, tuple(sorted(designs))))
        rng = np.random.default_rng(abs(hash(arm)) % 2 ** 32)
        joint = rng.normal(0.5, 0.01, size=len(state['group_order']))
        values = {lanes.CONTROL_ARM: joint - 0.05, lanes.MODEL_ARM: joint - 0.4,
                  lanes.JOINT_ARM: joint}
        return {'values': {name: value.tolist() for name, value in values.items()},
                'alphas': {}, 'dimensions': {name: len(columns)
                                             for name, columns in designs.items()}}

    def archived(self, stratum):
        raise AssertionError('the stub carries no archive')


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setitem(runner.ADAPTERS, 'phenotype', StubAdapter)
    StubAdapter.calls = []
    rows = cohort_rows()
    export = tmp_path / 'export'
    (export / 'inputs').mkdir(parents=True)
    (export / lanes.ROWS_FILE).write_text(json.dumps({'schema': lanes.ROWS_SCHEMA,
                                                      'samples': rows}))
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
    instance = StubAdapter.__new__(runner.PhenotypeAdapter)
    assert instance.sample_key(row) == exported[0]['sample_id']


def test_a_lane_without_an_adapter_is_refused_with_what_one_must_supply():
    class Args:
        lane = 'interaction'

    with pytest.raises(SystemExit) as error:
        runner.adapter(Args())
    message = str(error.value)
    assert 'no adapter' in message and 'never re-qualified' in message
    assert 'per-group values for the three arms' in message
    assert "['phenotype']" in message


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


def test_the_panel_assembles_a_complete_family_and_names_an_incomplete_one(workspace):
    arguments = workspace['base'] + workspace['wave']
    assert runner.main(['fit'] + arguments + ['--support', 'complete_case']) == 0
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
    assert panel['support'] == 'complete_case'
    assert panel['support_sha256'] == 'complete_case-digest'
    assert panel['forward_bound']['columns']
    assert panel['wave']['couplings']['state'] == 'landed'
    assert sum(panel['state_counts'][name] for name in lanes.STATES) == 100


def test_the_panel_refuses_cells_from_two_supports(workspace):
    arguments = workspace['base'] + workspace['wave'] + ['--support', 'complete_case']
    assert runner.main(['fit'] + arguments) == 0
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
