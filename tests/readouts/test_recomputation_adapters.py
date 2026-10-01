"""The anchor-panel gate adapters: what they refuse, and what they read.

Every refusal here is reachable without the retained extraction archives, which
live on the allocation, so this file tests the conditions that must hold before
any array is read: that a published cell is the one the request names, that its
sources are byte-identical, that its fold map is read from the shape an
anchor-panel cell actually writes, and that the declared metric names address
summaries the published cells really carry. The end-to-end refit is gated by the
pipeline-identity replay instead, which needs the archives and therefore runs
where they are.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.readouts import recomputation as RC  # noqa: E402
from src.capability.readouts import recomputation_adapters as AD  # noqa: E402
from src.capability.context.crossed_controls import CONTROL_SETS  # noqa: E402
from src.capability.core import gate_retention as GR  # noqa: E402
from src.capability.core.gate_retention import GATE_RETENTION  # noqa: E402
from src.capability.context.local_context import control_sets  # noqa: E402

#: A published crossed-controls cell, staged into the ignored log tree. Absent on
#: a checkout that has not pulled it, and the tests that read it say so rather
#: than asserting on a fixture. The staged copy predates the schema rename and is
#: a historical record, so its schema_version is normalised to what the adapter
#: reads now instead of the record being edited.
PUBLISHED_CROSSED = (ROOT / 'archive/logs/shared/crossed_controls_20260923/reports'
                     / 'crossed_controls_progen3_3b_20260923'
                     / 'crossed_controls_progen3-3b_fold20260923.json')
PUBLISHED_LOCAL_ADMISSION = (ROOT / 'results/R5/local_context_20260923/20260923223301_305478e124be'
                             / 'lcgm_admission/local_context_measurement.json')


def _published_crossed() -> dict:
    if not PUBLISHED_CROSSED.is_file():
        pytest.skip(f'no published crossed-controls cell at {PUBLISHED_CROSSED}')
    payload = json.loads(PUBLISHED_CROSSED.read_text())
    expected = AD.ANCHOR_STAGES['crossed_controls'].schema_version
    if payload.get('schema_version') != expected:
        payload['schema_version'] = expected
    return payload


def _request(gate: str, *, published: Path, arm='progen3-3b', panel='anchor201',
             split_seed=20260923, selection=None) -> RC.CellRequest:
    return RC.CellRequest(gate=gate, arm=arm, panel=panel, split_seed=split_seed,
                          selection=selection or RC.ReadoutSelection(arm=arm, panel=panel),
                          published=published)


# ------------------------------------------------------- the declared addressing


def test_every_declared_crossed_metric_is_a_name_the_evaluator_actually_writes():
    """The defect this pins: four declared names that addressed no summary at all.

    The crossed-controls evaluator writes ``increment_<label>_<control set>`` and
    ``rank_mse_reduction_<label>_<control set>``, so a declared metric is checked
    against that rule rather than against prose.
    """
    labels = ('M', 'R', 'R_after_M')
    writable = {f'{prefix}_{label}_{control}'
                for prefix in ('increment', 'rank_mse_reduction')
                for label in labels for control in CONTROL_SETS}
    declared = set(GATE_RETENTION['crossed_controls'].published_metrics)
    assert declared <= writable, sorted(declared - writable)
    assert declared, 'the gate fits representation cells and must declare a metric'


def test_every_declared_local_context_metric_is_a_name_the_evaluator_actually_writes():
    labels = ('M', 'R', 'R_after_M')
    sets = control_sets(('wall', 'rf3'))
    writable = {f'{prefix}_{label}_{name}'
                for prefix in ('increment', 'rank_mse_reduction')
                for label in labels for name in sets}
    declared = set(GATE_RETENTION['local_context'].published_metrics)
    assert declared <= writable, sorted(declared - writable)


def test_the_declared_metrics_address_summaries_the_published_cells_carry():
    published = _published_crossed()
    for metric in GATE_RETENTION['crossed_controls'].published_metrics:
        assert metric in published['summaries'], metric
    if not PUBLISHED_LOCAL_ADMISSION.is_file():
        pytest.skip(f'no local-context admission at {PUBLISHED_LOCAL_ADMISSION}')
    admission = json.loads(PUBLISHED_LOCAL_ADMISSION.read_text())
    summaries = admission['summaries'][admission['cells'][0]]
    for metric in GATE_RETENTION['local_context'].published_metrics:
        assert metric in summaries, metric


def test_the_identity_metric_exists_in_every_control_set_declaration_of_both_gates():
    """``increment_R_C`` is present by construction, not by convention."""
    assert 'C' in CONTROL_SETS
    assert 'C' in control_sets(('wall', 'rf3'))
    for stage in AD.ANCHOR_STAGES.values():
        assert stage.identity_metric == 'increment_R_C'


def test_the_adapters_registered_in_the_driver_are_exactly_the_ones_that_exist():
    assert set(AD.ANCHOR_ADAPTERS) == set(RC.IMPLEMENTED_ADAPTERS) - {'readout_panel'}
    for gate in AD.ANCHOR_ADAPTERS:
        assert gate not in RC.PENDING_ADAPTERS
        assert gate in GATE_RETENTION


# ----------------------------------------------------------- the stage it calls


def test_each_stage_script_loads_and_exposes_the_functions_the_adapter_calls():
    crossed = AD.load_stage('crossed_controls')
    for name in ('load_profile_store', 'attach_manifest_tokens', 'build_blocks',
                 'local_projection', 'PROFILE_SCORE_TOLERANCE'):
        assert hasattr(crossed, name), name
    local = AD.load_stage('local_context')
    for name in ('build_blocks', 'build_interface', 'crossed'):
        assert hasattr(local, name), name
    # The local-context stage keeps no second copy of the profile-store binding,
    # the anchor-support refusal or the packed-token budget: it imports the
    # crossed-controls stage for them, which is why the adapter reads those two
    # from that stage for both gates rather than from the gate it is fitting.
    assert not hasattr(local, 'load_profile_store')
    assert not hasattr(local, 'attach_manifest_tokens')
    # Loading both stages puts two module objects of the crossed-controls script
    # in one process -- the local-context stage loads it under its own name -- so
    # the adapter takes the binding from its own instance and the check here is
    # that both instances are the same code rather than the same object.
    assert Path(local.crossed.__file__) == Path(crossed.__file__)
    assert local.crossed.PROFILE_SCORE_TOLERANCE == crossed.PROFILE_SCORE_TOLERANCE
    assert AD.load_stage('crossed_controls') is crossed


def test_loading_a_stage_for_a_gate_that_is_not_an_anchor_panel_gate_is_refused():
    with pytest.raises(RC.RecomputationRefused, match='not an anchor-panel gate'):
        AD.load_stage('folding_stability')


# --------------------------------------------------------------------- identity


def test_the_identity_of_a_real_published_cell_is_read_from_its_own_report():
    published = _published_crossed()
    identity = AD.anchor_identity(published, 'increment_R_C')
    assert len(identity.support) == 201
    assert identity.units == 163
    assert identity.variants == 25728
    assert identity.split_seed == 20260923
    assert identity.resample_draws == 2000
    assert identity.resample_seed == 20260923
    assert identity.unit_label == 'wild-type family at 50% identity'
    assert len(identity.outer_folds) == 5
    assert all(len(inner) == 4 for inner in identity.inner_folds)
    # Every family appears in exactly one outer fold, so the partition the digest
    # binds is a partition of the published units.
    held = [family for fold in identity.outer_folds for family in fold]
    assert len(held) == len(set(held)) == identity.units
    assert AD.anchor_identity(published, 'increment_R_C').digest() == identity.digest()


def test_a_report_whose_folds_are_keyed_by_design_is_refused_rather_than_guessed():
    published = _published_crossed()
    readout_shaped = dict(published, folds={'B': published['folds']})
    with pytest.raises(RC.RecomputationRefused, match='list of outer folds'):
        AD.anchor_identity(readout_shaped, 'increment_R_C')


def test_a_report_without_the_identity_metric_is_refused_rather_than_defaulted():
    published = _published_crossed()
    stripped = dict(published, summaries={k: v for k, v in published['summaries'].items()
                                          if k != 'increment_R_C'})
    with pytest.raises(RC.RecomputationRefused, match='resampling contract'):
        AD.anchor_identity(stripped, 'increment_R_C')


def test_the_support_falls_back_to_the_fitted_rows_when_no_support_block_exists():
    """A freshly returned report carries rows but no support block yet."""
    published = _published_crossed()
    fresh = {k: v for k, v in published.items() if k != 'support'}
    identity = AD.anchor_identity(fresh, 'increment_R_C')
    assert identity.support == tuple(published['support']['assay_ids'])


# ---------------------------------------------------------------- source drift


def test_sources_are_matched_by_file_name_so_a_relocated_root_is_not_reported_as_drift():
    published = {'/gpfs/project/data/cohort.json': 'aa', '/gpfs/project/x/q0.npz': 'bb'}
    realised = {'/home/other/data/cohort.json': 'aa', '/home/other/x/q0.npz': 'bb'}
    record = AD.refuse_on_source_drift(published, realised)
    assert record == dict(identical=True, exhaustive=True, files=2, published_files=2,
                          shared_files=2, extraction_archives=1, manifests=0)


def test_a_depth_axis_recomputation_is_not_refused_for_reading_a_different_extraction():
    """Its archives are not in the published record by construction.

    The scoped check still refuses a manifest whose bytes moved and a source the
    published cell never held, so what it drops is only the requirement that
    every published archive be read again.
    """
    published = {'/a/cohort.json': 'aa', '/a/manifest_x.json': 'bb', '/a/q0.npz': 'cc'}
    depth = {'/b/cohort.json': 'aa', '/b/manifest_x.json': 'bb', '/b/pooled_q0.npz': 'dd'}
    with pytest.raises(RC.RecomputationRefused, match='were not read'):
        AD.refuse_on_source_drift(published, depth)
    record = AD.refuse_on_source_drift(published, depth, exhaustive=False)
    assert record['identical'] is True and record['exhaustive'] is False
    assert record['shared_files'] == 2
    assert record['published_sources_not_read'] == 1 and record['sources_not_published'] == 1
    # What the scoped check still refuses: a file both records name whose bytes moved.
    moved = {'/b/cohort.json': 'aa', '/b/manifest_x.json': 'CHANGED', '/b/pooled_q0.npz': 'dd'}
    with pytest.raises(RC.RecomputationRefused, match='changed bytes'):
        AD.refuse_on_source_drift(published, moved, exhaustive=False)


def test_a_changed_a_missing_and_an_extra_source_are_all_named_in_one_refusal():
    published = {'/a/cohort.json': 'aa', '/a/q0.npz': 'bb', '/a/q1.npz': 'cc'}
    realised = {'/b/cohort.json': 'aa', '/b/q0.npz': 'CHANGED', '/b/q2.npz': 'dd'}
    with pytest.raises(RC.RecomputationRefused) as refusal:
        AD.refuse_on_source_drift(published, realised)
    message = str(refusal.value)
    assert 'q1.npz' in message and 'q2.npz' in message and 'q0.npz' in message
    assert 'were not read' in message and 'changed bytes' in message


def test_a_basename_collision_is_refused_rather_than_resolved():
    with pytest.raises(RC.RecomputationRefused, match='cannot be matched'):
        AD._by_name({'/a/features.npz': 'aa', '/b/features.npz': 'bb'})
    # The same file listed twice under one name is not a collision.
    assert AD._by_name({'/a/features.npz': 'aa', '/b/features.npz': 'aa'}) == {'features.npz': 'aa'}


def test_a_real_published_cell_matches_its_own_source_map_relocated():
    published = _published_crossed()['source_sha256']
    relocated = {f'/somewhere/else/{Path(key).name}': digest for key, digest in published.items()}
    record = AD.refuse_on_source_drift(published, relocated)
    assert record['files'] == len(published) == 207
    assert record['extraction_archives'] == 201
    assert record['manifests'] == 5


# ------------------------------------------------------------------ predictions


def test_the_worst_prediction_departure_is_reported_per_design():
    published = {'C': np.zeros(4), 'C+R': np.array([0.0, 1.0, 2.0, 3.0])}
    recomputed = {'C': np.zeros(4), 'C+R': np.array([0.0, 1.0, 2.0, 3.5])}
    worst = AD.flat_prediction_agreement(published, recomputed)
    assert worst == {'C': 0.0, 'C+R': 0.5}


def test_a_recomputation_that_fitted_a_different_design_set_is_refused():
    with pytest.raises(RC.RecomputationRefused, match='different design set'):
        AD.flat_prediction_agreement({'C': np.zeros(2)}, {'C': np.zeros(2), 'C+R': np.zeros(2)})


def test_a_design_of_a_different_length_is_refused_rather_than_compared_on_a_prefix():
    with pytest.raises(RC.RecomputationRefused, match='predictions against'):
        AD.flat_prediction_agreement({'C': np.zeros(3)}, {'C': np.zeros(2)})


def test_published_predictions_are_bound_to_the_digest_their_own_report_declares(tmp_path):
    arrays = tmp_path / 'cell.npz'
    np.savez_compressed(arrays, C=np.arange(3, dtype=np.float64))
    digest = hashlib.sha256(arrays.read_bytes()).hexdigest()
    loaded = AD.published_prediction_arrays(tmp_path / 'cell.json',
                                            {'prediction_array_sha256': digest})
    assert set(loaded) == {'C'}
    with pytest.raises(RC.RecomputationRefused, match='does not match the digest'):
        AD.published_prediction_arrays(tmp_path / 'cell.json',
                                       {'prediction_array_sha256': '0' * 64})
    with pytest.raises(RC.RecomputationRefused, match='not at'):
        AD.published_prediction_arrays(tmp_path / 'absent.json', {})


# ---------------------------------------------------------------------- devices


def test_the_default_device_is_the_one_the_published_cell_was_fitted_on():
    assert AD.resolve_device({'runtime': {'device': 'cpu'}}, None) == ('cpu', True)
    assert AD.resolve_device({'runtime': {'device': 'cpu'}}, 'cpu') == ('cpu', True)


def test_a_host_replay_of_a_cuda_cell_is_refused_unless_the_caller_asks_for_it():
    import torch
    report = {'runtime': {'device': 'cuda:0'}}
    if torch.cuda.is_available():
        assert AD.resolve_device(report, None) == ('cuda:0', True)
    else:
        with pytest.raises(RC.RecomputationRefused, match='different reduction order'):
            AD.resolve_device(report, None)
    assert AD.resolve_device(report, 'cpu') == ('cpu', False)


# ----------------------------------------------- refusals before any array read


def test_a_published_cell_of_the_wrong_schema_status_arm_or_seed_is_refused(tmp_path):
    published = _published_crossed()
    for mutation, pattern in (
            (dict(schema_version='something_else'), 'carries schema'),
            (dict(status='smoke'), "not\n?\\s*an admitted one|is a 'smoke' cell"),
            (dict(arm='gpt2'), 'is arm'),
            (dict(fold_seed=20260925), 'is split seed')):
        path = tmp_path / 'cell.json'
        path.write_text(json.dumps(dict(published, **mutation)))
        with pytest.raises(RC.RecomputationRefused, match=pattern):
            AD.published_cell(_request('crossed_controls', published=path))


def test_an_absent_published_cell_is_refused_by_path(tmp_path):
    with pytest.raises(RC.RecomputationRefused, match='not at'):
        AD.published_cell(_request('crossed_controls', published=tmp_path / 'absent.json'))


def test_each_adapter_refuses_a_request_for_another_gate(tmp_path):
    path = tmp_path / 'cell.json'
    path.write_text('{}')
    with pytest.raises(RC.RecomputationRefused, match='fits crossed_controls'):
        AD.crossed_controls_cell(_request('local_context', published=path), cohort=path,
                                 manifests=[path], profile_store=tmp_path)
    with pytest.raises(RC.RecomputationRefused, match='fits local_context'):
        AD.local_context_cell(_request('crossed_controls', published=path), cohort=path,
                              manifests=[path], profile_store=tmp_path)


def test_a_local_context_qualification_cell_has_nothing_to_recompute(tmp_path):
    from src.capability.context.local_context import declaration_sha256
    path = tmp_path / 'cell.json'
    path.write_text(json.dumps(dict(schema_version='d1_local_context_v1', status='complete',
                                    arm='progen3-3b', fold_seed=20260923, stage='qualify',
                                    declaration_sha256=declaration_sha256())))
    with pytest.raises(RC.RecomputationRefused, match='fits no representation block'):
        AD.local_context_cell(_request('local_context', published=path), cohort=path,
                              manifests=[path], profile_store=tmp_path)


def test_a_changed_local_context_declaration_is_refused_before_any_array_is_read(tmp_path):
    path = tmp_path / 'cell.json'
    path.write_text(json.dumps(dict(schema_version='d1_local_context_v1', status='complete',
                                    arm='progen3-3b', fold_seed=20260923, stage='measure',
                                    declaration_sha256='0' * 64)))
    with pytest.raises(RC.RecomputationRefused, match='qualification receipt does not cover it'):
        AD.local_context_cell(_request('local_context', published=path), cohort=path,
                              manifests=[path], profile_store=tmp_path)


def test_a_manifest_set_that_declares_no_such_arm_is_refused(tmp_path):
    manifest = tmp_path / 'manifest_other.json'
    manifest.write_text(json.dumps({'identity': {'arm': 'gpt2'}}))
    with pytest.raises(RC.RecomputationRefused, match='declares arm'):
        AD._manifest_for([manifest], 'progen3-3b')


def test_an_anchor_panel_cell_without_its_manifests_is_refused(tmp_path):
    published = _published_crossed()
    path = tmp_path / 'cell.json'
    path.write_text(json.dumps(published))
    with pytest.raises(RC.RecomputationRefused, match='needs the extraction manifests'):
        AD.prepare_cell(_request('crossed_controls', published=path), published,
                        cohort=tmp_path / 'cohort.json', manifests=[], profile_store=tmp_path)


# ------------------------------------------- the resampler that formed the interval


def test_the_current_resampler_reproduces_a_real_published_cell_exactly():
    """The finding this pins: the resampler's file moved, its behaviour did not.

    Every published cell of both gate families recorded a digest for
    `src/capability/context/profile_increment.py` that differs from the file's current bytes,
    and the published bytes are not recoverable from the repository. What can be
    measured is whether the change reached a reported number, and it did not: the
    current resampler re-forms each published summary from the published cell's own
    rows at its own draw count and seed.
    """
    published = _published_crossed()
    record = RC.published_intervals_reproduce(published,
                                    GATE_RETENTION['crossed_controls'].published_metrics)
    assert record['exact'] is True
    assert record['max_absolute_difference'] == 0.0
    assert record['metrics_checked'] == 4
    assert record['resample_seed'] == published['bootstrap_seed']


def test_a_moved_published_interval_is_refused_rather_than_differenced():
    published = _published_crossed()
    summaries = dict(published['summaries'])
    moved = dict(summaries['increment_R_C'])
    moved['point'] = moved['point'] + 1e-9
    summaries['increment_R_C'] = moved
    with pytest.raises(RC.RecomputationRefused, match='no longer reproduces the published'):
        RC.published_intervals_reproduce(dict(published, summaries=summaries), ('increment_R_C',))


def test_a_published_cell_with_no_rows_or_no_seed_cannot_be_checked_and_is_refused():
    published = _published_crossed()
    with pytest.raises(RC.RecomputationRefused, match='no per-assay rows'):
        RC.published_intervals_reproduce(dict(published, assays=[]), ('increment_R_C',))
    stripped = {k: v for k, v in published.items() if k != 'bootstrap_seed'}
    with pytest.raises(RC.RecomputationRefused, match='no bootstrap seed'):
        RC.published_intervals_reproduce(stripped, ('increment_R_C',))
    with pytest.raises(RC.RecomputationRefused, match='none of this gate'):
        RC.published_intervals_reproduce(published, ('a_metric_no_gate_writes',))


def test_changed_and_absent_fitting_code_is_reported_and_not_refused():
    record = RC.code_digest_differences({'analysis_code_sha256': {
        'src/capability/readouts/recomputation.py': '0' * 64,
        'src/capability/no_such_file.py': '1' * 64}})
    assert record['recorded_files'] == 2
    assert record['changed'] == ['src/capability/readouts/recomputation.py']
    assert record['absent'] == ['src/capability/no_such_file.py']
    assert RC.code_digest_differences({})['recorded_files'] == 0


def test_a_real_published_cell_records_fitting_code_that_has_since_moved():
    """Stated as a measurement, not an assumption: which files moved is data."""
    published = _published_crossed()
    record = RC.code_digest_differences(published)
    assert record['recorded_files'] == 7
    # The cell recorded the pre-rename ``transfer`` package paths, and the
    # restructure moved all seven, so none of them resolves in the tree today.
    # The resampler ``profile_increment`` is among them, which is why the
    # behavioural check above exists rather than a digest comparison.
    assert set(record['absent']) == {'scripts/transfer/analyse_crossed_controls.py',
                                     'scripts/transfer/analyse_readout.py',
                                     'src/transfer/crossed_controls.py',
                                     'src/transfer/profile_increment.py',
                                     'src/transfer/profiles.py',
                                     'src/transfer/readout_analysis.py',
                                     'src/transfer/readout_extraction.py'}
    assert record['changed'] == []


def test_a_published_cell_without_an_anchor_arm_must_have_one_named(tmp_path):
    """Measured on the real cells: the stage gained `anchor_arm` after they were fitted."""
    published = _published_crossed()
    assert 'anchor_arm' not in published
    path = tmp_path / 'cell.json'
    path.write_text(json.dumps(published))
    request = _request('crossed_controls', published=path)
    with pytest.raises(RC.RecomputationRefused, match='records no anchor arm'):
        AD.prepare_cell(request, published, cohort=tmp_path / 'cohort.json',
                        manifests=[tmp_path / 'manifest_x.json'], profile_store=tmp_path)


# ------------------------------------------------- the runner's selection lookup


def _runner():
    import importlib.util
    path = ROOT / 'scripts/capability/readouts/recompute_representation_cells.py'
    spec = importlib.util.spec_from_file_location('recompute_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_roster_cell_is_matched_to_its_own_arm_and_panel_selection():
    """The defect this pins: the runner looked up a bare arm in a keyed selection.

    The published selection is keyed by arm and panel -- ProGen3-3B selects one
    block on the anchor panel and another on the EC-conditioned one -- so a bare
    arm lookup matched nothing and every roster was refused. Serving the arm's
    other panel instead would have been the worse failure.
    """
    runner = _runner()
    anchor = RC.ReadoutSelection(arm='progen3-3b', panel='anchor201', depth_axis='depth',
                                 depth_index=11, coordinates=256, projection_seed=20261123)
    ec = RC.ReadoutSelection(arm='progen3-3b', panel='EC_conditioned40', depth_axis='depth',
                             depth_index=23, coordinates=256, projection_seed=20261123)
    selection = {anchor.key: anchor, ec.key: ec}
    cells = [dict(gate='crossed_controls', arm='progen3-3b', panel='anchor201',
                  split_seed=20260923, published='x.json')]
    requests = runner._requests(cells, selection)
    assert len(requests) == 1
    assert requests[0].selection.depth_index == 11
    cells[0]['panel'] = 'EC_conditioned40'
    assert runner._requests(cells, selection)[0].selection.depth_index == 23


def test_a_roster_cell_on_an_unselected_panel_is_refused_and_the_selected_panels_named():
    runner = _runner()
    anchor = RC.ReadoutSelection(arm='progen3-3b', panel='anchor201')
    cells = [dict(gate='crossed_controls', arm='progen3-3b', panel='native_progen3',
                  split_seed=20260923, published='x.json')]
    with pytest.raises(SystemExit, match="selected only on \\['anchor201'\\]"):
        runner._requests(cells, {anchor.key: anchor})
    other = [dict(gate='crossed_controls', arm='gpt2', panel='anchor201',
                  split_seed=20260923, published='x.json')]
    with pytest.raises(SystemExit, match='not selected on any panel'):
        runner._requests(other, {anchor.key: anchor})


def test_the_roster_requires_each_gate_its_own_adapter_inputs(tmp_path):
    runner = _runner()
    path = tmp_path / 'roster.json'
    path.write_text(json.dumps({'cells': [dict(
        gate='crossed_controls', arm='progen3-3b', panel='anchor201', split_seed=20260923,
        published='x.json', cohort='c.json')]}))
    with pytest.raises(SystemExit, match="crossed_controls cell and is missing"):
        runner._roster(path)
    path.write_text(json.dumps({'cells': [dict(
        gate='folding_stability', arm='progen3-3b', panel='anchor201', split_seed=20260923,
        published='x.json', cohort='c.json')]}))
    with pytest.raises(SystemExit, match='no adapter in this runner dispatches'):
        runner._roster(path)


def test_the_plan_entry_point_runs_end_to_end_on_a_real_selection(tmp_path):
    """The gap this closes: both defects so far were in the entry point, not the functions.

    The roster resolver was broken for every roster while the function tests
    passed, because they built requests directly and nothing drove the command
    line over them. This runs `main` itself, on a selection and roster written
    here, and asserts the plan it writes.
    """
    runner = _runner()
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'cells': {
        'progen3-3b|anchor201': {'class_name': 'C4_full_random_feature', 'depth_axis': 'depth',
                                 'depth_index': 11, 'coordinates': 256,
                                 'projection_seed': 20261123}}}))
    roster = tmp_path / 'roster.json'
    roster.write_text(json.dumps({'cells': [
        dict(gate='crossed_controls', arm='progen3-3b', panel='anchor201', split_seed=20260923,
             published=str(tmp_path / 'absent.json'), cohort='c.json',
             manifests=['m.json'], profile_store='p'),
        dict(gate='readout_panel', arm='progen3-3b', panel='anchor201', split_seed=20260923,
             published=str(tmp_path / 'absent.json'), cohort='c.json', manifest='m.json')]}))
    out = tmp_path / 'plan.json'
    assert runner.main(['plan', '--selection', str(selection), '--cells', str(roster),
                        '--out', str(out)]) == 0
    plan = json.loads(out.read_text())
    assert plan['cells'] == 2
    # Both refuse on the absent published report rather than on the selection,
    # which is what proves the roster resolved and the adapters were reached.
    assert len(plan['runnable']) == 0
    assert {c['gate'] for c in plan['refused']} == {'crossed_controls', 'readout_panel'}
    for cell in plan['refused']:
        assert 'published cell report is not at' in cell['refusal']
        assert cell['selection']['depth_index'] == 11


def test_the_plan_entry_point_refuses_a_roster_whose_panel_is_unselected(tmp_path):
    runner = _runner()
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'cells': {
        'progen3-3b|anchor201': {'class_name': 'C4_full_random_feature', 'depth_axis': 'depth',
                                 'depth_index': 11, 'coordinates': 256,
                                 'projection_seed': 20261123}}}))
    roster = tmp_path / 'roster.json'
    roster.write_text(json.dumps({'cells': [
        dict(gate='crossed_controls', arm='progen3-3b', panel='EC_conditioned40',
             split_seed=20260923, published='x.json', cohort='c.json',
             manifests=['m.json'], profile_store='p')]}))
    with pytest.raises(SystemExit, match="selected only on \\['anchor201'\\]"):
        runner.main(['plan', '--selection', str(selection), '--cells', str(roster),
                     '--out', str(tmp_path / 'plan.json')])


def test_a_refused_cell_is_written_before_the_runner_fails(tmp_path):
    """The evidence must survive the refusal, or the refusal answers nothing."""
    runner = _runner()
    path = tmp_path / 'cell.json'
    record = dict(verdict='refused', refusal='predictions departed by 2.9e-06',
                  published_quantities=dict(shared_metrics=72, recovered_exactly=72, moved=[],
                                            resolved_sign_changes=[]))
    with pytest.raises(SystemExit) as refusal:
        runner._refuse_on_verdict(record, path)
    message = str(refusal.value)
    assert 'predictions departed by 2.9e-06' in message
    assert '72 of 72 published summaries recovered exactly' in message
    assert '0 moved' in message and '0 resolved signs changed' in message
    assert runner._refuse_on_verdict(dict(verdict='recomputed'), path) is None


# --------------------------------------------------------------- batch aggregation


def _cell(gate, arm, seed, *, verdict='recomputed', point=0.0, edge=0.0, signs=(),
          metrics=(('increment_R_C', False),)):
    return dict(schema='representation_recomputation_cell', gate=gate, arm=arm,
                panel='anchor201', split_seed=seed, verdict=verdict,
                selection=dict(depth_axis='depth'),
                published_quantities=dict(shared_metrics=72,
                                          max_absolute_point_change=point,
                                          max_absolute_interval_endpoint_change=edge,
                                          resolved_sign_changes=list(signs)),
                increments=[dict(metric=name, comparable=True, resolved_sign_changed=changed,
                                 unit='wild-type family at 50% identity',
                                 published=dict(point=-0.04, interval=[-0.06, -0.02]),
                                 recomputed=dict(point=0.01, interval=[0.001, 0.02]),
                                 point_change=0.05)
                            for name, changed in metrics])


def test_the_two_counts_that_mean_opposite_things_are_kept_apart():
    """A refusal with no sign change is provenance; a sign change is a result."""
    records = [
        _cell('crossed_controls', 'a', 20260923, verdict='refused', point=2.1e-9, edge=9.1e-9),
        _cell('crossed_controls', 'b', 20260923, verdict='refused', point=1e-9),
        _cell('crossed_controls', 'c', 20260923, point=3e-9),
    ]
    summary = RC.aggregate_cells(records)
    gate = summary['gates']['crossed_controls']
    assert gate['cells_attempted'] == gate['cells_completed'] == 3
    assert gate['cells_whose_pipeline_gate_refused'] == 2
    assert gate['cells_refused_with_no_resolved_sign_change'] == 2
    assert gate['resolved_sign_changes'] == []
    assert gate['verdict'] == 'stands'
    assert gate['max_absolute_point_change'] == 3e-9
    assert gate['max_absolute_interval_endpoint_change'] == 9.1e-9
    assert summary['any_cell_refused'] is True
    assert summary['any_verdict_changes'] is False


def test_a_verdict_change_is_named_with_its_cell_metric_and_intervals():
    records = [_cell('local_context', 'gpt2', 20260924,
                     metrics=(('increment_R_C', True), ('increment_R_C_P', False)))]
    summary = RC.aggregate_cells(records)
    gate = summary['gates']['local_context']
    assert gate['verdict'] == 'changes'
    assert summary['any_verdict_changes'] is True
    assert summary['gates_whose_verdict_changes'] == ['local_context']
    change = gate['resolved_sign_changes'][0]
    assert change['cell'] == 'gpt2/anchor201/20260924'
    assert change['metric'] == 'increment_R_C'
    assert change['published']['interval'] == [-0.06, -0.02]
    assert change['recomputed']['interval'] == [0.001, 0.02]
    assert change['unit'] == 'wild-type family at 50% identity'
    assert gate['declared_metrics']['increment_R_C']['sign_changes'] == 1
    assert gate['declared_metrics']['increment_R_C_P']['sign_changes'] == 0


def test_a_failed_cell_makes_its_gate_incomplete_rather_than_silently_absent():
    summary = RC.aggregate_cells(
        [_cell('crossed_controls', 'a', 20260923)],
        failures=[dict(gate='crossed_controls', arm='b', panel='anchor201',
                       split_seed=20260923, error='a retained archive is not present')],
        planned_refusals=[dict(gate='folding_stability', arm='c', panel='stability101',
                               refusal='no depth-resolved extraction exists')])
    gate = summary['gates']['crossed_controls']
    assert gate['verdict'] == 'incomplete'
    assert gate['cells_attempted'] == 2 and gate['cells_completed'] == 1
    assert summary['cells_failed'] == 1
    assert summary['left_for_the_waves'][0]['gate'] == 'folding_stability'


def test_a_refusal_that_also_changed_a_sign_is_not_counted_as_provenance_only():
    records = [_cell('crossed_controls', 'a', 20260923, verdict='refused',
                     metrics=(('increment_R_C', True),))]
    gate = RC.aggregate_cells(records)['gates']['crossed_controls']
    assert gate['cells_whose_pipeline_gate_refused'] == 1
    assert gate['cells_refused_with_no_resolved_sign_change'] == 0
    assert gate['verdict'] == 'changes'


# ------------------------------------------- the width a stage is allowed to fit


def test_a_projected_representation_of_the_wrong_width_is_refused():
    """The defect this closes: one stage refused a wrong width and its sibling did not.

    `fit_stability_singles.load_arm` has always refused a representation that is
    not the admitted four-block design. `fit_pairwise_epistasis.load_arm` did not,
    so the same substitution failed loudly on three gates and silently on the
    fourth. A narrower array is not a narrower version of this design: one depth's
    two summaries are a differently constructed measurement at a different
    capacity, so a stage handed one must refuse rather than fit it and report the
    result beside a published number.
    """
    from src.capability.interactions.pairwise_epistasis import (FEATURE_BLOCKS, PROJECTION_DIM,
                                                 require_projected_width)
    expected = len(FEATURE_BLOCKS) * PROJECTION_DIM
    assert expected == 1024
    assert require_projected_width(expected, arm='gpt2-large', source='x.npz') == expected
    # The width a depth selection would hand it: two summaries at 256 each.
    with pytest.raises(ValueError, match='512-coordinate projected representation'):
        require_projected_width(512, arm='gpt2-large', source='x.npz')
    with pytest.raises(ValueError, match='different design at a different capacity'):
        require_projected_width(2048, arm='gpt2-large', source='x.npz')
    # The refusal names the arm and the file, because a campaign refusing on one
    # background of one arm should not require a search to locate it.
    with pytest.raises(ValueError, match='progen3-3b: bg7.npz'):
        require_projected_width(512, arm='progen3-3b', source='bg7.npz')


def test_every_fit_stage_calls_the_shared_width_condition():
    """One condition, three stages, no inline copies.

    Two of the three asserted the width inline with their own copy of the message
    and the third did not assert it at all. Adding a shared helper beside two
    copies would have made three expressions of one condition, so both copies now
    call it and this asserts that none came back.
    """
    stages = ('fit_stability_singles.py', 'fit_nested_singles.py', 'fit_pairwise_epistasis.py')
    for stage in stages:
        from scripts.capability.entrypoints import stage_path
        source = stage_path(ROOT, stage).read_text()
        assert 'require_projected_width(' in source, stage
        assert source.count('require_projected_width') >= 2, stage  # imported and called
        assert 'representation width is not four projected blocks' not in source, stage
    for path in (ROOT / 'src/capability').glob('*.py'):
        if path.name != 'pairwise_epistasis.py':
            assert 'representation width is not four projected blocks' not in path.read_text()


# --------------------------------------- the inventory checking its own declarations


def test_a_declared_pattern_with_an_absent_prefix_is_reported_absent(tmp_path):
    """The hole six declared locations went through.

    `probe_artifacts` exempted any location containing a placeholder as
    `pattern_not_probed`, so a pattern whose directory does not exist made no
    presence claim at all and reached a driver hours later instead of a build.
    A pattern's fixed prefix carries no placeholder and is a concrete path, so it
    is claimed; the per-arm expansion still is not, because that needs the roster.
    """
    gate = 'remote_homology'
    records = {r['role']: r for r in GR.probe_artifacts(gate, {'cluster': tmp_path,
                                                               'repository': tmp_path})}
    fold = records['fold_map']
    assert '<arm>' in fold['location']
    assert fold['status'] == 'absent', fold
    assert 'fixed prefix' in fold['detail'] and 'does not exist' in fold['detail']

    # With the prefix present, the pattern is reported as prefix-present and the
    # expansion is explicitly left to the driver.
    (tmp_path / 'results/R5/remote_homology_20260924/fits').mkdir(parents=True)
    records = {r['role']: r for r in GR.probe_artifacts(gate, {'cluster': tmp_path,
                                                               'repository': tmp_path})}
    fold = records['fold_map']
    assert fold['status'] == 'pattern_prefix_present', fold
    assert "driver's to check" in fold['detail']


def test_a_store_with_no_root_is_unreachable_and_not_absent(tmp_path):
    """A clean run proves nothing about a store whose root was not given."""
    records = GR.probe_artifacts('remote_homology', {'repository': tmp_path})
    cluster = [r for r in records if r['store'] == 'cluster']
    assert cluster, 'this gate declares cluster locations'
    assert {r['status'] for r in cluster} == {'unreachable_from_this_host'}
    assert all('no root given' in r['detail'] for r in cluster)


def test_the_inventory_has_exactly_one_location_checker():
    """The three-copies lesson applied to this checker itself.

    A standalone `verify_locations` was added beside `probe_artifacts`, which would
    have made two checkers for one condition. It was folded in rather than shipped.
    """
    import src.capability.core.gate_retention as module
    assert not hasattr(module, 'verify_locations')
    assert not hasattr(module, 'refuse_on_missing_locations')
    assert hasattr(module, 'probe_artifacts')
    assert hasattr(module.RetainedArtifact, 'fixed_prefix')
