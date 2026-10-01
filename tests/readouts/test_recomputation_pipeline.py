"""Conditions the representation recomputation must always hold.

The recomputation exists to replace one number by another on the same support,
so the conditions worth testing are the ones under which that replacement would
be a different measurement wearing the published one's name. Three of them have
already cost this programme a run, and each is a test below rather than a comment
in the driver:

* a float32 matrix product's reduction order follows the BLAS thread count, so
  the admitted projection is pinned and a process with no pool is refused;
* the same product is not invariant to how its rows are blocked, so it is blocked
  by assay as the admitted analysis blocks it, and the departure is measured;
* an identity digest over NumPy scalars hashes the same data to two values across
  NumPy versions, so every label is rendered as a Python value first.

The tests are a mixture of synthetic and real. The real ones read artefacts this
host actually holds -- the admitted 201-assay proteinglm analysis report, the
admitted admission receipt, one retained three-assay extraction at hidden width
4,096 -- and they are marked in their names, because a pipeline verified only on
synthetic arrays is not verified against the artefacts it will be run on.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core import gate_contract as GC  # noqa: E402
from src.capability.core import gate_retention as GR  # noqa: E402
from src.capability.readouts import recomputation as RC  # noqa: E402
from src.capability.readouts import targeted_depth as TD  # noqa: E402
from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.readouts.readout_analysis import evaluate_readouts, sequence_features  # noqa: E402
from src.capability.readouts.readout_class_sweep import PROJECTION_BLAS_THREADS, projection_matrices  # noqa: E402

#: Real artefacts this host holds. A test that needs one and cannot find it skips
#: with the path named, rather than passing on the synthetic half alone.
ADMITTED_REPORT = (REPO_ROOT / 'results/R2/readout_20260923/complete/results/'
                   'external_baseline/20260923105347_9e1188dbfcda/'
                   'readout_analysis_proteinglm_common/readout_proteinglm-7b-clm_common.json')
ADMITTED_REPORT_SEED2 = (REPO_ROOT / 'results/R2/readout_20260923/complete/results/'
                         'external_baseline/20260923105347_9e1188dbfcda/'
                         'readout_analysis_proteinglm_common_fold20260924/'
                         'readout_proteinglm-7b-clm_common_fold20260924.json')
RETAINED_SMOKE = (REPO_ROOT / 'results/R2/readout_20260923/smoke/'
                  'readout_proteinglm_smoke')


from scripts.capability.entrypoints import stage_path

#: The retained admission receipt and the archived gate records are results and
#: records, not distributed with the code repository.
needs_admission_receipt = pytest.mark.skipif(
    not RC.ADMITTED_RECEIPT.is_file(),
    reason=f'{RC.ADMITTED_RECEIPT} is absent: the admitted admission receipt is a retained '
           'result, not distributed with the code repository')
needs_archived_records = pytest.mark.skipif(
    not (REPO_ROOT / 'archive').is_dir(),
    reason='archive/ is absent: the archived gate records and logs are not distributed '
           'with the code repository')

def _real(path: Path) -> dict:
    if not path.is_file():
        pytest.skip(f'the real artefact {path} is not on this host')
    return json.loads(path.read_text())


# ------------------------------------------------------------ the retention inventory


def test_every_gate_with_representation_cells_declares_every_required_role():
    """A cell cannot be refitted on identical terms from an incomplete declaration.

    The required roles are the cohort, the fold map, the penalty grid, the design
    declaration and the published increment. A gate that declares cells and omits
    one of them is not recomputable, whatever else it retained, and the inventory
    says so rather than implying readiness.
    """
    for gate, entry in GR.GATE_RETENTION.items():
        if entry.representation_cells:
            assert entry.role_gaps() == (), f'{gate} declares cells and lacks {entry.role_gaps()}'
        for kind in GR.SELECTION_KINDS:
            verdict = GR.recomputability(gate, kind)
            assert verdict['verdict'] in ('recomputable', 'requires_re_extraction',
                                          'not_applicable')
            if verdict['verdict'] != 'recomputable':
                assert verdict['missing'] or verdict['no_cells_reason'], \
                    f'{gate}/{kind} is not recomputable and names nothing missing'


def test_a_gate_with_no_representation_cells_is_not_applicable_rather_than_blocked():
    """Four states, never merged: recomputable, needing re-extraction, and no cells at all."""
    for gate in ('higher_order', 'higher_order_extended', 'structural_constraints',
                 'generation_control', 'global_additive_context'):
        for kind in GR.SELECTION_KINDS:
            verdict = GR.recomputability(gate, kind)
            assert verdict['verdict'] == 'not_applicable', gate
            assert verdict['representation_cells'] == 0
            assert verdict['no_cells_reason']


def test_a_depth_selection_is_served_only_where_the_archives_are_named():
    """The central property, asserted so it cannot quietly change.

    A gate fitted on a cohort with no depth-resolved extraction can serve a
    different readout class from its retained full-width states and cannot serve
    a different depth from anything, because no other block's state was written.
    A gate whose wave has landed names where those states are: claiming the
    selection without naming the archives would be the inventory describing what
    a gate wrote rather than what a recomputation reads.
    """
    # Four gates were blocked this way when the inventory was written. All four
    # cohorts were extracted between 2026-09-24 and 2026-09-26 and every wave is
    # reconciled against disk, so none is blocked now.
    blocked = {gate for gate in GR.GATE_RETENTION
               if GR.recomputability(gate, 'extraction_depth')['verdict']
               == 'requires_re_extraction'}
    assert blocked == set(), blocked
    for gate in ('folding_stability', 'residue_interactions', 'remote_homology',
                 'external_confirmation'):
        assert GR.recomputability(gate, 'extraction_depth')['verdict'] == 'recomputable'
        states = [a for a in GR.GATE_RETENTION[gate].artifacts if a.role == 'depth_states']
        assert len(states) == 1, gate
        assert 'byte-identical' in states[0].note
        assert 'reconciled against disk' in states[0].note, gate
    # The negative path, on a constructed entry rather than on a roster the
    # campaigns keep changing: cells, no depth-resolved archives, and the reason
    # carried rather than implied by the verdict alone.
    unextracted = GR.GateRetention(
        gate='x', kind='mechanism_gate', question='q', record='r', cohort='c', unit='u',
        representation_cells=9, cell_shape='s', published_metrics=('m',),
        artifacts=tuple(GR.RetainedArtifact(role, 'loc', 'repository')
                        for role in GR.REQUIRED_ROLES),
        supported_selections=frozenset({'readout_class'}),
        missing_for={'extraction_depth': ('no depth-resolved extraction exists on this gate\'s '
                                          'own cohort',)})
    GR.GATE_RETENTION['x'] = unextracted
    try:
        assert GR.recomputability('x', 'readout_class')['verdict'] == 'recomputable'
        refused = GR.recomputability('x', 'extraction_depth')
        assert refused['verdict'] == 'requires_re_extraction'
        assert any('no depth-resolved extraction' in reason for reason in refused['missing'])
        assert 'x' in GR.inventory({'repository': REPO_ROOT,
                                    'cluster': None})['requires_re_extraction']['extraction_depth']
    finally:
        del GR.GATE_RETENTION['x']


def test_a_declaration_that_accounts_for_neither_selection_kind_is_refused():
    common = dict(gate='x', kind='mechanism_gate', question='q', record='r', cohort='c',
                  unit='u', cell_shape='s', published_metrics=('m',),
                  artifacts=tuple(GR.RetainedArtifact(role, 'loc', 'repository')
                                  for role in GR.REQUIRED_ROLES))
    with pytest.raises(ValueError, match='neither supported nor accounted for'):
        GR.GateRetention(representation_cells=1, supported_selections=frozenset(), **common)
    with pytest.raises(ValueError, match='both supported and missing'):
        GR.GateRetention(representation_cells=1,
                         supported_selections=frozenset(GR.SELECTION_KINDS),
                         missing_for={'readout_class': ('x',)}, **common)
    with pytest.raises(ValueError, match='must say why'):
        GR.GateRetention(representation_cells=0, published_metrics=(),
                         supported_selections=frozenset(),
                         missing_for={k: ('x',) for k in GR.SELECTION_KINDS},
                         **{k: v for k, v in common.items() if k != 'published_metrics'})
    with pytest.raises(ValueError, match='undeclared retention role'):
        GR.RetainedArtifact('not_a_role', 'loc', 'repository')
    with pytest.raises(ValueError, match='undeclared store'):
        GR.RetainedArtifact('cohort', 'loc', 'somewhere_else')


@needs_archived_records
def test_an_unreachable_store_is_not_reported_as_an_absent_artefact(tmp_path):
    """Off the allocation the cluster locations are unreachable, not missing.

    Reporting them absent from the workstation would turn every substantive
    retained array into a defect and bury the ones that really are missing.
    """
    without = GR.inventory({'repository': REPO_ROOT, 'cluster': None})
    statuses = {artifact['status'] for gate in without['gates']
                for artifact in gate['artifacts'] if artifact['store'] == 'cluster'}
    assert statuses <= {'unreachable_from_this_host'}
    assert not [artifact for gate in without['gates'] for artifact in gate['artifacts']
                if artifact['status'] == 'absent']
    # With a root that holds none of them, the same locations are absent, which is
    # the finding the probe exists to produce when it is run where they should be.
    with_empty = GR.inventory({'repository': REPO_ROOT, 'cluster': tmp_path})
    absent = [artifact for gate in with_empty['gates'] for artifact in gate['artifacts']
              if artifact['status'] == 'absent']
    assert absent, 'an empty cluster root must make the declared cluster locations absent'


def test_the_retention_declaration_digest_follows_the_declaration():
    before = GR.declaration_digest()
    assert len(before) == 64 and before == GR.declaration_digest()
    entry = GR.GATE_RETENTION['local_context']
    GR.GATE_RETENTION['local_context'] = GR.GateRetention(
        **{**entry.__dict__, 'cell_shape': entry.cell_shape + ' (perturbed)'})
    try:
        assert GR.declaration_digest() != before
    finally:
        GR.GATE_RETENTION['local_context'] = entry
    assert GR.declaration_digest() == before


# --------------------------------------------------------------------- the selection


def test_a_selection_is_refused_unless_it_names_a_declared_class_and_depth():
    with pytest.raises(ValueError, match='not a declared readout class'):
        RC.ReadoutSelection(arm='a', class_name='C9_invented')
    with pytest.raises(ValueError, match='not a declared depth axis'):
        RC.ReadoutSelection(arm='a', depth_axis='sideways')
    with pytest.raises(ValueError, match='needs a depth index'):
        RC.ReadoutSelection(arm='a', depth_axis='wide')
    with pytest.raises(ValueError, match='takes no depth index'):
        RC.ReadoutSelection(arm='a', depth_axis='union', depth_index=3)
    with pytest.raises(ValueError, match='thread count is positive'):
        RC.ReadoutSelection(arm='a', blas_threads=0)
    admitted = RC.ReadoutSelection(arm='a')
    assert admitted.is_admitted_pipeline and admitted.selection_kind == 'readout_class'
    assert admitted.blas_threads == PROJECTION_BLAS_THREADS
    wider = RC.ReadoutSelection(arm='a', class_name='C2_full_linear')
    assert wider.selection_kind == 'readout_class' and not wider.is_admitted_pipeline
    deeper = RC.ReadoutSelection(arm='a', depth_axis='depth', depth_index=28,
                                 coordinates=256, projection_seed=20261123)
    assert deeper.selection_kind == 'extraction_depth' and not deeper.is_admitted_pipeline


def test_a_recomputation_without_the_reassessments_selection_is_refused(tmp_path):
    with pytest.raises(RC.RecomputationRefused, match='no readout selection at'):
        RC.load_selection(tmp_path / 'absent.json')
    empty = tmp_path / 'empty.json'
    empty.write_text(json.dumps({'cells': {}}))
    with pytest.raises(RC.RecomputationRefused, match='declares no per-cell selection'):
        RC.load_selection(empty)
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps({'cells': {'progen2-base|anchor201': {'class_name': 'C9_invented'}}}))
    with pytest.raises(RC.RecomputationRefused, match='unusable selection'):
        RC.load_selection(bad)
    # A key naming no panel is refused rather than spread over that arm's panels:
    # the published selection differs by panel for ProGen3-3B, so guessing which
    # panel a bare key meant is the substitution this driver exists to prevent.
    bare = tmp_path / 'bare.json'
    bare.write_text(json.dumps({'cells': {'progen2-base': {'depth_axis': 'admitted'}}}))
    with pytest.raises(RC.RecomputationRefused, match='names no panel'):
        RC.load_selection(bare)
    good = tmp_path / 'good.json'
    good.write_text(json.dumps({'cells': {'progen2-base|anchor201': {
        'depth_axis': 'depth', 'depth_index': 25, 'coordinates': 256,
        'projection_seed': 20261123}}}))
    selection = RC.load_selection(good)
    assert selection[('progen2-base', 'anchor201')].selection_kind == 'extraction_depth'


def test_the_published_selection_loads_and_differs_by_panel_for_one_arm():
    """The real artefact, not a fixture: ten cells, one class, and two arms whose
    selected block depends on which panel the cell is on."""
    path = REPO_ROOT / 'archive/logs/shared/recomputation_20260924/readout_selection.json'
    if not path.is_file():
        pytest.skip(f'{path} is not on this host')
    selection = RC.load_selection(path)
    assert len(selection) == 10
    assert {choice.class_name for choice in selection.values()} == {'C4_full_random_feature'}
    assert selection[('progen3-3b', 'anchor201')].depth_index == 11
    assert selection[('progen3-3b', 'EC_conditioned40')].depth_index == 23
    assert selection[('protgpt2', 'anchor201')].depth_index == 21
    assert selection[('protgpt2', 'native_protgpt2')].depth_index == 21
    assert all(choice.selection_kind == 'extraction_depth' for choice in selection.values())


def test_the_tolerance_comes_from_the_admitted_receipt_and_is_never_defaulted(tmp_path):
    """A replay that invented its own tolerance would certify nothing."""
    receipt = REPO_ROOT / 'results/R2/readout_20260923/final_admission.json'
    if receipt.is_file():
        assert RC.admitted_prediction_tolerance(receipt) == 1e-8
    with pytest.raises(RC.RecomputationRefused, match='is not at'):
        RC.admitted_prediction_tolerance(tmp_path / 'absent.json')
    broken = tmp_path / 'broken.json'
    broken.write_text(json.dumps({'baseline_tolerance': {}}))
    with pytest.raises(RC.RecomputationRefused, match='no baseline_tolerance'):
        RC.admitted_prediction_tolerance(broken)
    negative = tmp_path / 'negative.json'
    negative.write_text(json.dumps({'baseline_tolerance': {'atol': 0.0}}))
    with pytest.raises(RC.RecomputationRefused, match='non-positive tolerance'):
        RC.admitted_prediction_tolerance(negative)


# ------------------------------------------------------------------- cell identity


def _identity(**overrides) -> RC.CellIdentity:
    fields = dict(support=('a', 'b', 'c'), units=3, variants=30, split_seed=20260923,
                  outer_folds=((1,), (2,), (3,)),
                  inner_folds=(((2,), (3,)), ((1,), (3,)), ((1,), (2,))),
                  unit_label='wild-type family at 50% identity',
                  resample_draws=2000, resample_seed=20260923)
    fields.update(overrides)
    return RC.CellIdentity(**fields)


@pytest.mark.parametrize('overrides, expected', [
    ({'support': ('a', 'b', 'd')}, 'support differs'),
    ({'support': ('c', 'b', 'a')}, 'different order'),
    ({'units': 4}, 'units differs'),
    ({'variants': 31}, 'variants differs'),
    ({'split_seed': 20260924}, 'split_seed differs'),
    ({'unit_label': 'assay'}, 'unit_label differs'),
    ({'resample_draws': 500}, 'resample_draws differs'),
    ({'resample_seed': 1}, 'resample_seed differs'),
    ({'outer_folds': ((1,), (3,), (2,))}, 'outer fold membership differs'),
    ({'outer_folds': ((1,), (2,))}, 'outer fold count differs'),
    ({'inner_folds': (((3,), (2,)), ((1,), (3,)), ((1,), (2,)))},
     'inner fold membership differs'),
])
def test_every_axis_of_identity_drift_is_refused_and_named(overrides, expected):
    published = _identity()
    assert RC.refuse_on_drift(published, _identity())['identical'] is True
    with pytest.raises(RC.RecomputationRefused, match=expected):
        RC.refuse_on_drift(published, _identity(**overrides))


def test_two_drifts_at_once_are_both_named():
    with pytest.raises(RC.RecomputationRefused) as refusal:
        RC.refuse_on_drift(_identity(), _identity(units=4, split_seed=20260925))
    message = str(refusal.value)
    assert 'units differs' in message and 'split_seed differs' in message


def test_an_identity_digest_does_not_depend_on_numpy_scalar_rendering():
    """The defect that cost the global-context gate a run, in this driver's own form.

    ``repr(np.float64(0.5))`` differs between NumPy 1.26 and NumPy 2, and
    ``json.dumps`` refuses a NumPy integer outright, so a fold label arriving as a
    NumPy scalar would either hash differently across interpreters or fail to
    serialise. Both inputs must give one digest.
    """
    report = dict(n_families=3, n_variants=30, fold_seed=20260923, bootstrap_seed=20260923,
                  support={'assay_ids': ['a', 'b', 'c']},
                  summaries={'delta_spearman': {'unit': 'wild-type family at 50% identity',
                                                'resamples': 2000}},
                  folds={'B': [dict(held_families=[1], inner_folds=[dict(validation_families=[2]),
                                                                    dict(validation_families=[3])]),
                               dict(held_families=[2], inner_folds=[dict(validation_families=[1]),
                                                                    dict(validation_families=[3])]),
                               dict(held_families=[3], inner_folds=[dict(validation_families=[1]),
                                                                    dict(validation_families=[2])])]})
    numpyish = json.loads(json.dumps(report))
    for fold in numpyish['folds']['B']:
        fold['held_families'] = [np.int64(v) for v in fold['held_families']]
        for inner in fold['inner_folds']:
            inner['validation_families'] = [np.int64(v) for v in inner['validation_families']]
    numpyish['support']['assay_ids'] = [np.str_(a) for a in numpyish['support']['assay_ids']]
    plain = RC.identity_from_readout_report(report)
    scalars = RC.identity_from_readout_report(numpyish)
    assert plain.digest() == scalars.digest()
    assert RC.refuse_on_drift(plain, scalars)['identical'] is True


def test_the_real_admitted_report_is_read_and_a_real_cross_seed_swap_is_refused():
    """Read on the admitted 201-assay proteinglm cells this host holds."""
    published = RC.identity_from_readout_report(_real(ADMITTED_REPORT))
    assert (len(published.support), published.units, published.variants) == (201, 163, 25728)
    assert published.split_seed == 20260923 and published.resample_draws == 2000
    assert published.unit_label == 'wild-type family at 50% identity'
    other = RC.identity_from_readout_report(_real(ADMITTED_REPORT_SEED2))
    with pytest.raises(RC.RecomputationRefused) as refusal:
        RC.refuse_on_drift(published, other)
    message = str(refusal.value)
    assert 'split_seed differs: published 20260923, realised 20260924' in message
    assert 'outer fold membership differs' in message


# ----------------------------------------------------- the representation construction


def _blocks(assays=4, per=5, width=32, seed=11):
    rng = np.random.default_rng(seed)
    counts = [per] * assays
    return (rng.standard_normal((assays * per, 4, width)) * 3.0).astype(np.float32), counts


def test_a_process_with_no_blas_pool_is_refused_rather_than_run_unpinned(monkeypatch):
    assert RC.require_blas_pool()
    monkeypatch.setattr(RC, 'blas_thread_counts', lambda: [])
    with pytest.raises(RC.RecomputationRefused, match='no BLAS thread pool'):
        RC.require_blas_pool()
    with pytest.raises(RC.RecomputationRefused, match='no BLAS thread pool'):
        RC.admitted_representation(*_blocks())


def test_the_admitted_product_is_blocked_by_assay_and_the_departure_is_measured():
    """The repair the depth sweep's pipeline-identity control forced, as a property.

    The per-assay product must equal an independently assembled per-assay product
    exactly, and must differ from the whole-panel one, which is what makes the
    blocking a choice with a consequence rather than a stylistic preference.
    """
    blocks, counts = _blocks(assays=6, per=8, width=256)
    per_assay = RC.admitted_representation(blocks, counts)
    projection = projection_matrices(blocks.shape[2])
    pieces, start = [], 0
    for count in counts:
        pieces.append(np.concatenate([blocks[start:start + count, i] @ projection[i]
                                      for i in range(4)], axis=1))
        start += count
    assert np.array_equal(per_assay, np.concatenate(pieces, axis=0))
    departure = RC.product_blocking_departure(blocks, counts)
    assert departure['pinned_blas_threads'] == PROJECTION_BLAS_THREADS
    assert departure['assay_blocks'] == len(counts)
    assert departure['coordinates'] == 1024
    assert departure['max_absolute_departure'] > 0.0, (
        'the two blockings agreed exactly, so this array cannot demonstrate the '
        'constraint that forces per-assay blocking')
    assert departure['max_absolute_departure'] < 1e-3 * departure['largest_absolute_entry']


def test_the_real_retained_states_reproduce_the_pinned_per_assay_product():
    """Measured on one retained three-assay extraction at hidden width 4,096."""
    manifest_path = RETAINED_SMOKE / 'manifest_proteinglm-7b-clm.json'
    manifest = _real(manifest_path)
    blocks, counts = [], []
    for record in manifest['assays']:
        with np.load(RETAINED_SMOKE / record['file'], allow_pickle=False) as data:
            features = np.asarray(data['features'], dtype=np.float32)
        blocks.append(features)
        counts.append(len(features))
    blocks = np.concatenate(blocks)
    assert blocks.shape[1] == 4 and blocks.shape[2] == 4096
    projection = projection_matrices(blocks.shape[2])
    pieces, start = [], 0
    for count in counts:
        pieces.append(np.concatenate([blocks[start:start + count, i] @ projection[i]
                                      for i in range(4)], axis=1))
        start += count
    assert np.array_equal(RC.admitted_representation(blocks, counts),
                          np.concatenate(pieces, axis=0))
    departure = RC.product_blocking_departure(blocks, counts)
    assert departure['max_absolute_departure'] > 0.0
    assert departure['identical'] is False


def test_a_depth_selection_is_refused_where_no_depth_resolved_state_exists():
    blocks, counts = _blocks()
    deeper = RC.ReadoutSelection(arm='a', depth_axis='depth', depth_index=3,
                                 coordinates=64, projection_seed=20261123)
    with pytest.raises(RC.RecomputationRefused, match='reads a depth-resolved extraction'):
        RC.representation(deeper, admitted_blocks=blocks, variant_counts=counts)
    rng = np.random.default_rng(5)
    pooled = {name: rng.standard_normal((sum(counts), 16)).astype(np.float32)
              for name in ('mean', 'last')}
    with pytest.raises(RC.RecomputationRefused, match='needs its declared coordinate count'):
        RC.representation(RC.ReadoutSelection(arm='a', depth_axis='depth', depth_index=3),
                          depth_blocks=pooled, variant_counts=counts)
    with pytest.raises(RC.RecomputationRefused, match='hold no'):
        RC.depth_representation(pooled, ('mut', 'suffix'), depth_index=3, coordinates=8,
                                base_seed=20261123, variant_counts=counts)
    with pytest.raises(RC.RecomputationRefused, match='do not cover'):
        RC.depth_representation(pooled, ('mean', 'last'), depth_index=3, coordinates=8,
                                base_seed=20261123, variant_counts=[1, 1])
    block, provenance = RC.representation(deeper, depth_blocks=pooled, variant_counts=counts)
    assert block.shape == (sum(counts), 128) and provenance['blocked_by_assay'] is True


def test_the_depth_projection_seed_follows_the_declared_rule_and_nothing_else():
    """Base seed plus eight times the depth index plus the summary's own index.

    Asserted against an independent construction, because a projection that
    depended on anything else -- a label, an observed vector, a fitted outcome --
    would make a depth cell's design a function of its own answer.
    """
    from src.capability.readouts.readout_depth import SUMMARY_NAMES, gaussian_projection
    counts = [4, 6]
    rng = np.random.default_rng(3)
    pooled = {name: rng.standard_normal((sum(counts), 12)).astype(np.float32)
              for name in ('mean', 'last')}
    built = RC.depth_representation(pooled, ('mean', 'last'), depth_index=7, coordinates=5,
                                    base_seed=20261123, variant_counts=counts)
    expected, start = [], 0
    for count in counts:
        expected.append(np.concatenate(
            [pooled[name][start:start + count]
             @ gaussian_projection(20261123 + 8 * 7 + SUMMARY_NAMES.index(name), 12, 5)
             for name in ('mean', 'last')], axis=1))
        start += count
    assert np.array_equal(built, np.concatenate(expected, axis=0))


def test_the_uncompressed_and_single_block_classes_form_the_blocks_they_name():
    blocks, counts = _blocks(width=24)
    full, provenance = RC.representation(RC.ReadoutSelection(arm='a',
                                                             class_name='C2_full_linear'),
                                         admitted_blocks=blocks, variant_counts=counts)
    assert full.shape == (sum(counts), 4 * 24) and 'uncompressed' in provenance['form']
    one, provenance = RC.representation(RC.ReadoutSelection(arm='a', class_name='D_final_mean'),
                                        admitted_blocks=blocks, variant_counts=counts)
    assert one.shape == (sum(counts), 24) and np.array_equal(one, blocks[:, 2, :])
    assert 'final_mean' in provenance['form']
    with pytest.raises(RC.RecomputationRefused, match='needs the per-assay variant counts'):
        RC.representation(RC.ReadoutSelection(arm='a'), admitted_blocks=blocks)


# -------------------------------------------------------------------- the comparison


def _summary(point, low, high, unit='wild-type family at 50% identity', units=163):
    return dict(point=point, interval=[low, high], unit=unit, resamples=2000, n_units=units,
                excludes_zero=bool(low > 0 or high < 0))


def test_a_recomputed_increment_is_reported_with_both_intervals_and_the_unit():
    published = dict(summaries={'delta_spearman': _summary(0.0197, 0.00942, 0.02971)})
    recomputed = dict(summaries={'delta_spearman': _summary(0.0530, 0.02916, 0.07792)})
    row, = RC.compare_increments(published, recomputed, ('delta_spearman',))
    assert row['comparable'] and row['unit'] == 'wild-type family at 50% identity'
    assert row['published']['interval'] == [0.00942, 0.02971]
    assert row['recomputed']['interval'] == [0.02916, 0.07792]
    assert row['point_change'] == pytest.approx(0.0333, abs=1e-9)
    assert row['resample_draws'] == 2000
    assert row['resolved_sign_changed'] is False


def test_a_sign_change_and_a_unit_mismatch_are_both_reported_rather_than_averaged():
    published = dict(summaries={'m': _summary(-0.0176, -0.02935, -0.00599)})
    recomputed = dict(summaries={'m': _summary(0.0530, 0.02916, 0.07792)})
    row, = RC.compare_increments(published, recomputed, ('m',))
    assert row['resolved_sign_changed'] is True
    mismatched = dict(summaries={'m': _summary(0.05, 0.02, 0.07, unit='assay')})
    row, = RC.compare_increments(published, mismatched, ('m',))
    assert row['comparable'] is False and 'resampling unit differs' in row['reason']
    row, = RC.compare_increments(published, dict(summaries={}), ('m',))
    assert row['comparable'] is False and 'recomputed report carries no such metric' in row['reason']


def test_the_pipeline_identity_gate_refuses_a_replay_outside_the_admitted_tolerance():
    published = dict(predictions=[dict(assay='a', B=[0.0, 1.0], R=[0.0, 1.0])])
    identical = dict(predictions=[dict(assay='a', B=[0.0, 1.0], R=[0.0, 1.0])])
    worst = RC.prediction_agreement(published, identical, ('B', 'R'))
    assert worst == {'B': 0.0, 'R': 0.0}
    assert RC.refuse_on_pipeline_drift(worst, 1e-8)['passed'] is True
    assert RC.pipeline_agreement(worst, 1e-8) == dict(
        tolerance=1e-8, max_absolute_prediction_deviation=worst, passed=True, offenders={},
        refusal=None)
    moved = dict(predictions=[dict(assay='a', B=[0.0, 1.0], R=[9.085e-7, 1.0])])
    worst = RC.prediction_agreement(published, moved, ('B', 'R'))
    assert worst['R'] == pytest.approx(9.085e-7)
    with pytest.raises(RC.RecomputationRefused, match='R at 9.085e-07'):
        RC.refuse_on_pipeline_drift(worst, 1e-8)
    # The deciding form does not raise, so a failing cell can still be carried
    # through to its increments; the verdict is what refuses it.
    record = RC.pipeline_agreement(worst, 1e-8)
    assert record['passed'] is False and list(record['offenders']) == ['R']
    assert 'R at 9.085e-07' in record['refusal']
    short = dict(predictions=[dict(assay='a', B=[0.0], R=[0.0])])
    with pytest.raises(RC.RecomputationRefused, match='predictions against'):
        RC.prediction_agreement(published, short, ('B',))


# --------------------------------------------------------------------------- planning


def _request(gate, selection=None, published=None):
    return RC.CellRequest(gate=gate, arm='progen2-base', panel='anchor201',
                          split_seed=20260923,
                          selection=selection or RC.ReadoutSelection(arm='progen2-base'),
                          published=published or REPO_ROOT / 'results/R2/readout_20260923/'
                                                             'final_admission.json')


@needs_admission_receipt
def test_a_plan_refuses_before_reading_an_array_and_names_what_is_missing(tmp_path):
    deeper = RC.ReadoutSelection(arm='progen2-base', depth_axis='depth', depth_index=25,
                                 coordinates=256, projection_seed=20261123)
    schedule = RC.plan([
        _request('readout_panel'),
        _request('higher_order'),
        _request('folding_stability', selection=deeper),
        _request('residue_interactions'),
        _request('readout_panel', published=tmp_path / 'absent.json'),
    ])
    assert schedule['cells'] == 5
    assert len(schedule['runnable']) == 1
    assert schedule['runnable'][0]['gate'] == 'readout_panel'
    refusals = {cell['gate']: cell['refusal'] for cell in schedule['refused']}
    assert 'no model quantity was read at all' in refusals['higher_order']
    # Its cohort's depth archives have landed, so what blocks it now is the
    # loader rather than the extraction, and the refusal says so.
    assert 'no adapter' in refusals['folding_stability']
    assert 'no adapter' in refusals['residue_interactions']
    assert schedule['retention_declaration_sha256'] == GR.declaration_digest()
    assert set(schedule['adapters']['pending']) >= {'folding_stability', 'residue_interactions',
                                                    'evolution_adaptation'}
    assert set(schedule['adapters']['implemented']) == {'readout_panel', 'crossed_controls',
                                                        'local_context'}


def test_a_cell_request_is_refused_for_an_undeclared_gate_or_a_mismatched_arm():
    with pytest.raises(ValueError, match='not a declared gate'):
        _request('not_a_gate')
    with pytest.raises(ValueError, match='the selection names arm'):
        RC.CellRequest(gate='readout_panel', arm='progen2-base', panel='anchor201',
                       split_seed=20260923, selection=RC.ReadoutSelection(arm='other'),
                       published=Path('x'))


# -------------------------------------------------------------- end to end, synthetic


def _cohort(assays=10, per=8, width=24, seed=7):
    rng = np.random.default_rng(seed)
    wildtype = ''.join(rng.choice(list(AA20), 40))
    rows, blocks = [], []
    for index in range(assays):
        positions = rng.choice(np.arange(1, 41), size=per, replace=False)
        mutants = []
        for position in positions:
            before = wildtype[position - 1]
            after = rng.choice([residue for residue in AA20 if residue != before])
            mutants.append(f'{before}{position}{after}')
        features = (rng.standard_normal((per, 4, width)) * 0.5).astype(np.float32)
        measured = rng.standard_normal(per) + 0.6 * features[:, 0, :3].sum(1)
        rows.append(dict(assay=f'assay{index:02d}', cluster=index, mutants=mutants,
                         measured=measured, P=rng.standard_normal(per),
                         M=rng.standard_normal(per) + 0.3 * measured,
                         S=sequence_features(wildtype, mutants)))
        blocks.append(features)
    return rows, np.concatenate(blocks), [per] * assays


@pytest.fixture(scope='module')
def _published_cell():
    """One published cell, fitted by the admitted procedure at the admitted selection.

    Module-scoped because the admitted design is 1,470 columns before the
    intercept and the fit is the cost of the test; the thread count is reduced for
    that reason alone and restored afterwards, and it touches the float64 solve
    rather than the float32 product the pinned count governs.
    """
    threads = torch.get_num_threads()
    torch.set_num_threads(min(8, threads))
    try:
        rows, blocks, counts = _cohort()
        admitted, _ = RC.representation(RC.ReadoutSelection(arm='synthetic'),
                                        admitted_blocks=blocks, variant_counts=counts)
        fitted = [dict(row) for row in rows]
        start = 0
        for row, count in zip(fitted, counts):
            row['R'] = admitted[start:start + count]
            start += count
        report = evaluate_readouts(fitted, fold_seed=20260923, seed=20260923, bootstrap=2000)
        report['support'] = {'assay_ids': [row['assay'] for row in rows]}
        yield rows, blocks, counts, report
    finally:
        torch.set_num_threads(threads)


@needs_admission_receipt
def test_the_admitted_selection_replay_recovers_the_published_cell(_published_cell):
    """The pipeline-identity control, end to end through the driver.

    Every design's held-out predictions and every published increment must come
    back, on an identity that is the published cell's own. This is the control a
    recomputed cell has to pass before its own number is read.
    """
    rows, blocks, counts, published = _published_cell
    threads = torch.get_num_threads()
    torch.set_num_threads(min(8, threads))
    try:
        request = _request('readout_panel', selection=RC.ReadoutSelection(arm='progen2-base'))
        record = RC.readout_cell(request, [dict(row) for row in rows], published,
                                 admitted_blocks=blocks, variant_counts=counts)
    finally:
        torch.set_num_threads(threads)
    assert record['identity']['identical'] is True
    agreement = record['prediction_agreement']
    assert agreement['passed'] is True and agreement['tolerance'] == 1e-8
    assert set(agreement['max_absolute_prediction_deviation']) == set(RC.READOUT_DESIGNS)
    assert max(agreement['max_absolute_prediction_deviation'].values()) <= 1e-8
    for row in record['increments']:
        assert row['comparable']
        assert row['point_change'] == pytest.approx(0.0, abs=1e-12)
        assert row['published']['interval'] == row['recomputed']['interval']
    assert record['product_blocking']['pinned_blas_threads'] == PROJECTION_BLAS_THREADS


def test_a_new_readout_class_keeps_the_identity_moves_the_increment_and_is_not_gated(
        _published_cell):
    """At a new class the predictions are supposed to move; gating them would refuse
    the recomputation for doing its job. What must not move is the support, the
    folds, the seeds, the weighting or the label budget."""
    rows, blocks, counts, published = _published_cell
    threads = torch.get_num_threads()
    torch.set_num_threads(min(8, threads))
    try:
        request = _request('readout_panel',
                           selection=RC.ReadoutSelection(arm='progen2-base',
                                                         class_name='C2_full_linear'))
        record = RC.readout_cell(request, [dict(row) for row in rows], published,
                                 admitted_blocks=blocks, variant_counts=counts)
    finally:
        torch.set_num_threads(threads)
    assert record['identity']['identical'] is True
    assert record['prediction_agreement']['passed'] is None
    assert record['prediction_agreement']['tolerance'] is None
    assert max(record['prediction_agreement']['max_absolute_prediction_deviation'].values()) > 1e-8
    assert record['representation']['coordinates'] == 4 * blocks.shape[2]
    moved = [row['point_change'] for row in record['increments']]
    assert any(abs(value) > 0 for value in moved)
    assert all(row['unit'] == 'wild-type family at 50% identity' for row in record['increments'])


def test_a_representation_of_the_wrong_length_is_refused(_published_cell):
    rows, blocks, counts, published = _published_cell
    request = _request('readout_panel')
    with pytest.raises(RC.RecomputationRefused, match='do not cover'):
        RC.readout_cell(request, [dict(row) for row in rows], published,
                        admitted_blocks=blocks, variant_counts=[counts[0]] * (len(counts) - 1))


# ------------------------------------------------------------------- the gate contract


def test_every_declared_gate_states_every_contract_step_exactly_once():
    for gate, declaration in GC.GATE_DECLARATIONS.items():
        steps = [element.step for element in declaration.elements]
        assert sorted(steps) == sorted(GC.CONTRACT_STEPS), gate
        assert len(steps) == len(set(steps)), gate


def test_a_representation_verdict_is_refused_while_the_readout_reassessment_is_open():
    """The programme-level sequencing rule, expressed once and checked.

    A gate that reported its representation step as settled while the class and
    depth sweeps are open would be asserting exactly what they are testing.
    """
    assert GC.READOUT_REASSESSMENT_OPEN is True
    for gate, declaration in GC.GATE_DECLARATIONS.items():
        status = declaration.by_step()['representation_readout'].status
        assert status in ('provisional', 'not_applicable', 'absent'), f'{gate} claims {status}'
    offending = GC.GateDeclaration(
        gate='x', record='archive/docs/R1/gate-local-context.md',
        elements=tuple(
            GC.GateElement(step, 'reported', quantities=('q',), bound='b',
                           remote='r' if step == 'generalization' else '',
                           contributions=(GC.ControlContribution('c', 'q [0, 1]', '[0, 1]', True),)
                           if step == 'control_qualification' else ())
            for step in GC.CONTRACT_STEPS))
    GC.GATE_DECLARATIONS['x'] = offending
    try:
        assert any('readout reassessment is' in message
                   for message in GC.conform('x')['divergences'])
    finally:
        del GC.GATE_DECLARATIONS['x']


def test_a_step_that_asserts_no_measurement_must_give_a_reason_and_name_no_quantity():
    GC.GATE_DECLARATIONS['y'] = GC.GateDeclaration(
        gate='y', record='archive/docs/R1/gate-local-context.md',
        elements=(
            GC.GateElement('endpoint_qualification', 'reported'),
            GC.GateElement('control_qualification', 'reported', quantities=('q',), bound='b'),
            GC.GateElement('native_likelihood', 'absent', quantities=('q',)),
            GC.GateElement('representation_readout', 'provisional', quantities=('q',), bound='b'),
            GC.GateElement('generalization', 'reported', quantities=('q',), bound='b'),
            GC.GateElement('causal_validation', 'reported', bound='b'),
        ))
    try:
        divergences = GC.conform('y')['divergences']
    finally:
        del GC.GATE_DECLARATIONS['y']
    joined = '\n'.join(divergences)
    assert 'endpoint_qualification is reported and names no quantity' in joined
    assert 'states no bound' in joined
    assert 'native_likelihood is absent and gives no reason' in joined
    assert 'native_likelihood is absent yet names quantities' in joined
    assert 'declares no control contribution' in joined
    assert 'says nothing about remote-family generalization' in joined
    assert 'without naming the resolved and generalising increment' in joined


def test_a_control_contribution_is_well_formed_in_its_declared_form_or_refused():
    """Three declared forms, and each is complete in its own terms or refused.

    An interval-form contribution must publish its interval and say whether it
    resolves away from zero. An exactness or order-statistic form publishes
    neither -- a census has no sampling variation and an axis has no zero to
    resolve against -- and must instead say what it establishes, so that the
    non-interval form is a declared difference in kind rather than an omission.
    """
    GC.ControlContribution('c', '+0.1 [0.0, 0.2]', '[0.0, 0.2]', True)
    GC.ControlContribution('c', '0 failures of 14,467', form='exactness',
                           establishes='that it holds on every row')
    GC.ControlContribution('c', '-0.0007 nats/residue', form='order_statistic',
                           establishes='its position on the corpus order axis')
    with pytest.raises(ValueError, match='not a declared qualification form'):
        GC.ControlContribution('c', 'q', form='vibes')
    with pytest.raises(ValueError, match='publishes its interval'):
        GC.ControlContribution('c', '+0.1', resolved_away_from_zero=True)
    with pytest.raises(ValueError, match='says whether it'):
        GC.ControlContribution('c', '+0.1 [0.0, 0.2]', '[0.0, 0.2]')
    with pytest.raises(ValueError, match='must be part of the'):
        GC.ControlContribution('c', '+0.1 [0.0, 0.2]', '[9.9, 9.9]', True)
    with pytest.raises(ValueError, match='publishes no interval'):
        GC.ControlContribution('c', '0 failures', '[0.0, 0.2]', True, form='exactness')
    with pytest.raises(ValueError, match='states what its qualification establishes'):
        GC.ControlContribution('c', '0 failures', form='exactness')


def test_an_interval_form_contribution_with_no_interval_is_a_divergence():
    """The rule the generation gate's three generator controls diverged on until
    their form was declared: a missing interval is a divergence, a declared
    non-interval form is not."""
    GC.GATE_DECLARATIONS['z'] = GC.GateDeclaration(
        gate='z', record='archive/docs/R1/gate-local-context.md',
        elements=tuple(
            GC.GateElement(step, 'reported', quantities=('q',), bound='b',
                           remote='r' if step == 'generalization' else '',
                           contributions=(
                               GC.ControlContribution.__new__(GC.ControlContribution),)
                           if step == 'control_qualification' else ())
            if step != 'representation_readout' else
            GC.GateElement(step, 'provisional', quantities=('q',), bound='b')
            for step in GC.CONTRACT_STEPS))
    # Bypass the constructor to simulate a declaration that lost its interval,
    # which is what the divergence rule has to catch when the form still says
    # 'interval'.
    broken = GC.GATE_DECLARATIONS['z'].by_step()['control_qualification'].contributions[0]
    object.__setattr__(broken, 'name', 'a control')
    object.__setattr__(broken, 'contribution', '+0.1')
    object.__setattr__(broken, 'interval', '')
    object.__setattr__(broken, 'resolved_away_from_zero', None)
    object.__setattr__(broken, 'form', 'interval')
    object.__setattr__(broken, 'establishes', '')
    try:
        divergences = GC.conform('z')['divergences']
    finally:
        del GC.GATE_DECLARATIONS['z']
    assert any('publishes no interval' in message and 'no other qualification form' in message
               for message in divergences)


@needs_archived_records
def test_every_declared_quantity_occurs_verbatim_in_the_record_it_points_at():
    """The read-only conformance check: a transcription cannot drift unnoticed.

    A gate every one of whose steps is absent names no quantity, and checking
    zero of them is the right outcome there rather than a vacuous pass; a gate
    that asserts a measurement anywhere must have something checked.
    """
    report = GC.conformance(root=REPO_ROOT)
    for gate, entry in report['gates'].items():
        transcription = entry['transcription']
        assert transcription['readable'], f'{gate}: {transcription["record"]} is unreadable'
        assert transcription['missing'] == [], f'{gate}: {transcription["missing"]}'
        measured = any(step['status'] in GC.MEASURED_STATUSES
                       for step in entry['steps'].values())
        assert (transcription['checked'] > 0) == measured, gate
    assert report['contract_sha256'] == GC.contract_digest()


def test_every_retention_entry_is_declared_on_the_contract_and_conforms():
    """No entry is left undeclared, and the unlaunched gates are a recorded state.

    An entry with no record to transcribe from would be the one hole the
    transcription check cannot see, so the three unlaunched gates point at the
    capability map that states their admission conditions and declare every step
    absent with its reason.
    """
    report = GC.conformance(root=REPO_ROOT)
    assert report['not_declared']['gates'] == []
    assert set(report['gates']) == set(GR.GATE_RETENTION)
    assert report['diverging'] == {}
    assert sorted(report['conforming']) == sorted(GR.GATE_RETENTION)
    for gate in ('conformational_dynamics', 'molecular_recognition', 'cellular_regulation'):
        statuses = {step: entry['status']
                    for step, entry in report['gates'][gate]['steps'].items()}
        assert set(statuses.values()) == {'absent'}, gate


def test_the_generation_gates_controls_are_qualified_in_a_declared_non_interval_form():
    """What was a divergence is now a declared difference in kind.

    The three generator controls are qualified by exactness and by an order
    statistic, each of which is tighter than an interval on the quantity it
    establishes, and each says what it establishes.
    """
    report = GC.conformance(root=REPO_ROOT)
    assert 'generation_control' in report['conforming']
    contributions = report['gates']['generation_control']['steps'][
        'control_qualification']['contributions']
    assert len(contributions) == 3
    forms = {row['form'] for row in contributions}
    assert forms == {'exactness', 'order_statistic'}
    for row in contributions:
        assert row['interval'] is None and row['resolved_away_from_zero'] is None
        assert row['establishes']


# ------------------------------------------------- the targeted depth extraction


def test_the_targeted_depth_set_is_the_selected_depth_and_the_two_admitted():
    """Three depths, or two when the selection already coincides with an admitted one.

    Writing a block twice would inflate the storage estimate and produce a lane
    that hooks the same state under two names, so the set is deduplicated and the
    plan reports how many depths it actually holds.
    """
    cost = TD.ARM_COSTS['progen2-xlarge']
    assert cost.admitted_depths == (15, 31) and cost.blocks == 32
    # The order is the extractor's: the admitted pair first, so an archive keeps the
    # admitted four feature blocks at positions 0 to 3 whatever else it holds.
    assert TD.depths_for('progen2-xlarge', 28) == (15, 31, 28)
    assert TD.depths_for('progen2-xlarge', 31) == (15, 31)
    assert TD.depths_for('progen2-xlarge', 15) == (15, 31)
    assert TD.depths_for('progen2-xlarge', [28, 4]) == (15, 31, 4, 28)
    every = TD.depths_for('progen2-xlarge', TD.every_block('progen2-xlarge'))
    assert every[:2] == (15, 31) and len(every) == 32
    with pytest.raises(ValueError, match='is outside 0..31'):
        TD.depths_for('progen2-xlarge', 32)
    with pytest.raises(ValueError, match='is outside'):
        TD.depths_for('progen2-xlarge', -1)


def test_the_gpu_estimate_is_the_measured_wall_clock_and_does_not_follow_depth_count():
    """The property that makes a targeted extraction affordable, asserted.

    Every hooked block is read from the same forward pass, so the GPU cost is the
    measured wall clock of the extraction that already ran; only the retained
    storage grows, and it grows exactly in proportion to the depths hooked.
    """
    for gate in TD.MEASURED_COHORTS:
        two = TD.arm_estimate(gate, 'proteinglm-7b-clm', (17, 35))
        three = TD.arm_estimate(gate, 'proteinglm-7b-clm', (17, 30, 35))
        assert two['gpu_hours'] == three['gpu_hours']
        assert two['measured_seconds'] == TD.ARM_COSTS['proteinglm-7b-clm'].seconds(gate)
        assert three['retained_bytes'] * 2 == two['retained_bytes'] * 3
        assert TD.cohort_gpu_hours(gate)['gpu_hours'] == pytest.approx(
            sum(cost.seconds(gate) for cost in TD.ARM_COSTS.values()) / 3600.0, abs=5e-4)
    for gate in TD.COHORTS:
        whole = TD.estimate(gate)
        assert whole['arms'] == len(TD.ARM_COSTS) == 33
        assert whole['gpu_cost']['gpu_hours'] == whole['total_gpu_hours']
        # Storage follows the depth count exactly; GPU time does not follow it at all.
        assert (sum(TD.retained_bytes(gate, arm, (0, 1, 2)) for arm in TD.ARM_COSTS) * 2
                == sum(TD.retained_bytes(gate, arm, (0, 1)) for arm in TD.ARM_COSTS) * 3)


def test_a_cohort_cost_is_measured_or_labelled_scaled_and_never_silently_either():
    """Two of the four cohorts carry per-arm wall clock, three carry a measured
    total, and the fourth is in flight. A scaled figure must say so, name what it
    scaled from, and carry the calibration the one scaled-and-measured cohort
    supplies -- which shows the state-count ratio understating by about 1.42, so
    the plain scaling is a lower bound rather than the number to plan against."""
    assert set(TD.MEASURED_COHORT_SECONDS) == set(TD.COHORTS) - {'external_confirmation'}
    for gate in TD.MEASURED_COHORT_SECONDS:
        record = TD.cohort_gpu_hours(gate)
        assert record['basis'] == 'measured' and 'gpu_hours_calibrated' not in record
        assert record['per_arm_measured'] is (gate in TD.MEASURED_COHORTS)
    scaled = TD.cohort_gpu_hours('external_confirmation')
    assert scaled['basis'] == 'scaled' and scaled['scaled_from'] == 'folding_stability'
    assert scaled['calibration_factor'] == pytest.approx(1.42, abs=0.02)
    assert scaled['gpu_hours_calibrated'] > scaled['gpu_hours']
    assert 'lower bound' in scaled['note']
    # The calibration is the measured cohort's own ratio, not a chosen number.
    predicted = (TD.MEASURED_COHORT_SECONDS['folding_stability']
                 * TD.COHORTS['remote_homology'].states / TD.COHORTS['folding_stability'].states)
    assert scaled['calibration_factor'] == pytest.approx(
        TD.MEASURED_COHORT_SECONDS['remote_homology'] / predicted, abs=5e-4)
    with pytest.raises(ValueError, match='no per-arm measured wall clock'):
        TD.ARM_COSTS['progen3-3b'].seconds('external_confirmation')


def test_the_storage_model_reproduces_the_measured_two_depth_footprint():
    """Validated against the footprint the completed retention pass actually wrote.

    Summed over the 33 arms of the stability cohort, the archives of the
    two-depth full-width retention pass occupy 29.79 GiB on the allocation. The
    model -- one float32 coordinate per state per summary per depth -- gives
    29.75 GiB over the same arms, so the estimate rests on a measured footprint
    rather than on an assumed layout.
    """
    modelled = sum(TD.retained_bytes('folding_stability', arm, (0, 1))
                   for arm in TD.ARM_COSTS) / 2 ** 30
    assert modelled == pytest.approx(29.79, abs=0.1)
    one = TD.retained_bytes('folding_stability', 'progen3-3b', (0, 1))
    assert one == 25_957 * 2 * 2 * 1280 * 4


def test_a_lane_naming_an_option_its_stage_does_not_define_is_not_queueable(tmp_path):
    """The defect that cost the first retention dispatch three of its four cells.

    A manifest that names an option its stage does not define is refused by
    argparse with exit 2, so the builder reads the stage's source and marks the
    campaign blocked rather than emitting a lane that cannot parse. Both real
    extractors now accept the option, so the property is checked on a stage that
    does not: the rule has to be the source, not the date it was written.
    """
    absent = tmp_path / 'stage_without_the_option.py'
    absent.write_text("parser.add_argument('--arm', required=True)")
    assert TD.extractor_accepts_depth(absent) is False
    # A source that cannot be read is a different answer from a source that does
    # not take the option, and is refused rather than reported as the latter.
    with pytest.raises(FileNotFoundError):
        TD.extractor_accepts_depth(tmp_path / 'no_such_stage.py')
    present = tmp_path / 'present.py'
    present.write_text(f"parser.add_argument('{TD.REQUIRED_EXTRACTOR_OPTION}', type=int)")
    assert TD.extractor_accepts_depth(present) is True


def test_every_arm_gets_one_lane_longest_first_naming_only_its_new_depths():
    selection = {arm: cost.middle_block for arm, cost in TD.ARM_COSTS.items()}
    selection['progen2-xlarge'] = 28
    rows = TD.campaign_rows('folding_stability', selection,
                            project_root='/root', runtime='ENV=1')
    assert len(rows) == len(TD.ARM_COSTS)
    assert [row['label'] for row in rows][0] == 'depth3-fs-galactica-30b'
    assert all(rows[i]['measured_seconds'] >= rows[i + 1]['measured_seconds']
               for i in range(len(rows) - 1))
    by_label = {row['label']: row for row in rows}
    moved = by_label['depth3-fs-progen2-xlarge']
    assert moved['extra_depths'] == [28]
    assert f'{TD.REQUIRED_EXTRACTOR_OPTION} 28' in moved['args']
    for label, row in by_label.items():
        if label != 'depth3-fs-progen2-xlarge':
            assert row['extra_depths'] == [] and TD.REQUIRED_EXTRACTOR_OPTION not in row['args']
    rendered = TD.render_campaign(rows, 'a header line')
    assert rendered.startswith('# a header line')
    assert '# slot\tkey\tgpu\tstage\tlabel\tenv\texpect\targs' in rendered
    assert len([line for line in rendered.splitlines() if not line.startswith('#')]) == len(rows)


def test_the_targeted_depth_declaration_digest_follows_the_measured_table():
    before = TD.declaration_digest()
    assert len(before) == 64 and before == TD.declaration_digest()
    entry = TD.ARM_COSTS['progen3-3b']
    TD.ARM_COSTS['progen3-3b'] = TD.ArmCost(**{**entry.__dict__,
                                               'stability_seconds': entry.stability_seconds + 1})
    try:
        assert TD.declaration_digest() != before
    finally:
        TD.ARM_COSTS['progen3-3b'] = entry
    assert TD.declaration_digest() == before


# ------------------------------------------------- the two collapsed duplications


def test_the_shared_fold_and_row_identity_digests_reproduce_the_expressions_they_replace():
    """Two fits are the same fit only if both sides hash the partition the same way.

    The global-context gate refuses to proceed when its fold digest differs from
    the admitted pairwise fit's, and until now each side computed the expression
    itself. Both are now one function, and both reproduce the expression they
    replace exactly, so every digest already recorded under them still verifies.
    """
    import hashlib
    from src.capability.interactions.pairwise_epistasis import fold_identity, row_identity
    folds = [dict(fold=0, held_groups=['nat-007', 'nat-030'], alpha=1.0),
             dict(fold=1, held_groups=['nat-001'], alpha=10.0)]
    expected = hashlib.sha256(json.dumps([f['held_groups'] for f in folds],
                                         sort_keys=True).encode()).hexdigest()
    assert fold_identity(folds) == expected
    groups, pairs, targets = ['a', 'a', 'b'], [(1, 2), (1, 2), (3, 4)], [0.5, -1.25, 2.0]
    row_expected = hashlib.sha256('\n'.join(
        f'{g}|{p}|{float(e)!r}' for g, p, e in zip(groups, pairs, targets)).encode()).hexdigest()
    assert row_identity(groups, pairs, targets) == row_expected
    # The rendering hazard both functions exist for: a NumPy target must hash as
    # the Python float does, and a NumPy fold label must refuse rather than hash
    # to a second value.
    assert row_identity(groups, pairs, np.asarray(targets)) == row_expected
    with pytest.raises(TypeError):
        fold_identity([dict(held_groups=[np.int64(1)])])


# ------------------------------------------- the depth option on the two extractors


EXTRACTORS = ('extract_stability_singles.py', 'extract_pairwise_epistasis.py')


def test_the_default_hooked_blocks_are_unchanged_in_both_real_extractors():
    """33 arms of completed extraction and every published cell rest on this default.

    Checked two ways against the real extractors rather than a stand-in: the
    shared rule reproduces the expression both of them used, for every block count
    in the panel and across the whole plausible range; and neither extractor
    carries that expression any more, so there is no second place for the default
    to drift to.
    """
    from src.capability.readouts.readout_extraction import feature_block_names, hooked_block_indices
    for count in list(range(1, 80)) + [cost.blocks for cost in TD.ARM_COSTS.values()]:
        assert hooked_block_indices(count) == ((count - 1) // 2, count - 1), count
    for arm, cost in TD.ARM_COSTS.items():
        assert hooked_block_indices(cost.blocks) == cost.admitted_depths, arm
    for name in EXTRACTORS:
        source = (stage_path(REPO_ROOT, name)).read_text(encoding='utf-8')
        assert 'hooked_block_indices(len(blocks), args.extra_block_index)' in source, name
        assert '(len(blocks) - 1) // 2' not in source, (
            f'{name} still computes the default itself, so it can drift from the shared rule')
        assert "'--extra-block-index'" in source, name
        assert 'feature_block_names(block_indices)' in source, name
    from src.capability.readouts.readout_extraction import FEATURE_NAMES
    assert feature_block_names(hooked_block_indices(32)) == list(FEATURE_NAMES)


def test_an_extra_depth_is_appended_deduplicated_and_ordered():
    """The admitted pair keeps feature positions 0 to 3, so an archive with extra
    depths still presents the admitted four where a reader of the admitted layout
    expects them; extras follow in ascending order and never repeat."""
    from src.capability.readouts.readout_extraction import feature_block_names, hooked_block_indices
    assert hooked_block_indices(32, [28]) == (15, 31, 28)
    assert hooked_block_indices(36, [34, 11, 12]) == (17, 35, 11, 12, 34)
    assert hooked_block_indices(32, [31, 15]) == (15, 31)
    assert hooked_block_indices(32, [28, 28, 28]) == (15, 31, 28)
    assert hooked_block_indices(32, []) == hooked_block_indices(32)
    names = feature_block_names(hooked_block_indices(32, [28]))
    assert names == ['middle_mean', 'middle_last', 'final_mean', 'final_last',
                     'block028_mean', 'block028_last']


def test_a_requested_depth_outside_the_stack_is_refused_rather_than_clamped():
    from src.capability.readouts.readout_extraction import hooked_block_indices
    for bad in (32, 64, -1, -99):
        with pytest.raises(ValueError, match='outside 0..31'):
            hooked_block_indices(32, [bad])
    with pytest.raises(ValueError, match='at least one block'):
        hooked_block_indices(0)


def test_the_campaign_is_queueable_once_both_extractors_accept_the_option():
    """The marking follows the extractors rather than a standing assumption."""
    for cohort in TD.COHORTS.values():
        stage = stage_path(REPO_ROOT, cohort.extractor)
        assert TD.extractor_accepts_depth(stage) is True, (
            f'{cohort.extractor} no longer accepts {TD.REQUIRED_EXTRACTOR_OPTION}')


# ------------------------------------------- the tolerance the depth stage reads


def test_the_depth_stage_reads_its_tolerance_from_the_receipt_and_keeps_its_call_shape():
    """The substitution, checked on the real stage rather than on its diff.

    Two things the edit could have broken, and one it had to remove. The stage's
    ``fit_cell`` must still accept exactly the arguments its existing callers
    pass, with the receipt optional, because a required parameter there would
    break the depth sweep's own harness. The tolerance must come from the
    admitted receipt and equal the 1e-8 that receipt declares. And no transcribed
    copy of that number may survive in the stage, or the single source is not one.

    Written as a test rather than left as a console check because the suite that
    exercises this stage end to end takes tens of minutes on a contended host,
    and the property this driver is responsible for is checkable in a second.
    """
    import importlib.util
    import inspect
    stage_path = REPO_ROOT / 'scripts/capability/readouts/analyse_readout_depth.py'
    if not stage_path.is_file():
        pytest.skip(f'{stage_path} is not on this host')
    for entry in (str(REPO_ROOT), str(REPO_ROOT / 'scripts/capability')):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location('_depth_stage_under_test', stage_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    signature = inspect.signature(module.fit_cell)
    assert signature.parameters['receipt'].default is module.ADMISSION_RECEIPT
    # The call shape every existing caller uses, including that stage's own test.
    signature.bind(object(), [], object(), {}, seeds=(1,), device='cpu', bootstrap=16,
                   shuffle=True, progress=None)

    source = stage_path.read_text(encoding='utf-8')
    assert 'PIPELINE_TOLERANCE' not in source, (
        'the transcribed tolerance is back, so the receipt is no longer its single source')

    if module.ADMISSION_RECEIPT.is_file():
        assert module.admitted_prediction_tolerance(module.ADMISSION_RECEIPT) == 1e-8
    with pytest.raises(RC.RecomputationRefused, match='is not at'):
        module.admitted_prediction_tolerance(REPO_ROOT / 'no-such-receipt.json')


# ------------------------------------------- the gate instrument that renders the verdict


def _write_archives(directory, name, features, per_depth=None):
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {'features': features} if per_depth is None else per_depth
    np.savez(directory / f'full_{name}.npz', **arrays,
             metadata=json.dumps({'assay': name}))


def test_the_identity_verifier_passes_a_faithful_archive_and_fails_a_permuted_one(tmp_path):
    """The instrument that decides whether a 720 GiB sweep may be dispatched.

    It has to pass an archive whose admitted four blocks are the retained ones and
    fail one whose hook order moved, which is the only failure mode that matters:
    a permuted order produces a well-formed archive of the right shape holding the
    right numbers in the wrong places.
    """
    import importlib.util
    script = REPO_ROOT / 'scripts/capability/readouts/verify_depth_archive_identity.py'
    spec = importlib.util.spec_from_file_location('_identity_verifier', script)
    verifier = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = verifier
    spec.loader.exec_module(verifier)

    rng = np.random.default_rng(19)
    blocks, width, rows = 36, 8, 5
    admitted = (17, 35)                      # (blocks - 1) // 2 and blocks - 1
    retained = rng.standard_normal((rows, 4, width)).astype(np.float32)
    depth_arrays = {f'features_d{index:03d}': rng.standard_normal((rows, 2, width)).astype(np.float32)
                    for index in range(blocks)}
    # A faithful depth archive holds the retained slices under the admitted names.
    depth_arrays[f'features_d{admitted[0]:03d}'] = retained[:, 0:2]
    depth_arrays[f'features_d{admitted[1]:03d}'] = retained[:, 2:4]

    admitted_dir, depth_dir = tmp_path / 'admitted', tmp_path / 'depth'
    _write_archives(admitted_dir, 'bg', retained)
    _write_archives(depth_dir, 'bg', None, per_depth=depth_arrays)
    out = tmp_path / 'faithful.json'
    assert verifier.main(['--arm', 'a', '--blocks', str(blocks), '--depth-dir', str(depth_dir),
                          '--admitted-dir', str(admitted_dir), '--out', str(out)]) == 0
    record = json.loads(out.read_text())
    assert record['identical'] is True and record['compared'] == 1
    assert record['admitted_block_indices'] == list(admitted)
    assert record['archives'][0]['byte_identical'] is True

    # The failure that matters: the two admitted blocks swapped, so every number is
    # present and every one is in the wrong place.
    swapped = dict(depth_arrays)
    swapped[f'features_d{admitted[0]:03d}'] = retained[:, 2:4]
    swapped[f'features_d{admitted[1]:03d}'] = retained[:, 0:2]
    swapped_dir = tmp_path / 'swapped'
    _write_archives(swapped_dir, 'bg', None, per_depth=swapped)
    out = tmp_path / 'swapped.json'
    assert verifier.main(['--arm', 'a', '--blocks', str(blocks), '--depth-dir', str(swapped_dir),
                          '--admitted-dir', str(admitted_dir), '--out', str(out)]) == 1
    record = json.loads(out.read_text())
    assert record['identical'] is False and record['non_identical'] == ['bg.npz']
    assert record['max_absolute_difference'] > 0.0

    # And the two refusals: an archive that is not depth-resolved, and no overlap.
    plain_dir = tmp_path / 'plain'
    _write_archives(plain_dir, 'bg', retained)
    with pytest.raises(SystemExit, match='not a depth-resolved archive'):
        verifier.main(['--arm', 'a', '--blocks', str(blocks), '--depth-dir', str(plain_dir),
                       '--admitted-dir', str(admitted_dir), '--out', str(tmp_path / 'x.json')])
    with pytest.raises(SystemExit, match='nothing to compare'):
        verifier.main(['--arm', 'a', '--blocks', str(blocks), '--depth-dir', str(depth_dir),
                       '--admitted-dir', str(tmp_path / 'empty'), '--out', str(tmp_path / 'y.json')])


def test_two_cohorts_cells_for_one_arm_never_share_an_output_directory():
    """The defect the gate run surfaced, and it was not in the hooking rule.

    The campaign queue derives a cell's output directory from the run id and the
    lane label alone. Both cohorts' gate manifests labelled their cell
    ``depth3-gpt2-large``, so the second cell found the first's manifest at that
    path and was reported ``skipped-complete`` -- writing nothing while looking
    complete. On the 32-arm wave that would have produced no archives for a whole
    cohort under a clean status file. Every label now carries the cohort's own tag.
    """
    labels = {}
    selection = {'gpt2-large': 0}
    for gate in TD.COHORTS:
        row, = TD.campaign_rows(gate, selection, project_root='/root', runtime='ENV=1')
        labels[gate] = row['label']
        assert row['label'].startswith(f'depth3-{TD.COHORTS[gate].tag}-')
        assert row['label'].endswith('-gpt2-large')
    assert len(set(labels.values())) == len(TD.COHORTS), labels
    assert len({cohort.tag for cohort in TD.COHORTS.values()}) == len(TD.COHORTS)


def test_a_wave_spreads_over_its_cards_with_one_cell_per_card_per_slot():
    """The queue refuses two cells on one card in one slot, so the spread must
    satisfy that by construction rather than by luck. With one card the manifest
    is the serial one a gate cell wants."""
    selection = {arm: cost.middle_block for arm, cost in TD.ARM_COSTS.items()}
    rows = TD.campaign_rows('folding_stability', selection, project_root='/root',
                            runtime='ENV=1', gpus=[0, 1, 2, 3])
    assert len(rows) == len(TD.ARM_COSTS)
    seen = set()
    for row in rows:
        key = (row['slot'], row['gpu'])
        assert key not in seen, f'slot {row["slot"]} names card {row["gpu"]} twice'
        seen.add(key)
    assert {row['gpu'] for row in rows} == {0, 1, 2, 3}
    assert max(row['slot'] for row in rows) == -(-len(rows) // 4)
    serial = TD.campaign_rows('folding_stability', selection, project_root='/root',
                              runtime='ENV=1', gpu=2)
    assert {row['gpu'] for row in serial} == {2}
    assert [row['slot'] for row in serial] == list(range(1, len(serial) + 1))
    with pytest.raises(ValueError, match='distinct and non-empty'):
        TD.campaign_rows('folding_stability', selection, project_root='/root',
                         runtime='ENV=1', gpus=[0, 0])


def test_a_failed_pipeline_gate_still_produces_a_complete_record_marked_refused():
    """A refusal that carries its comparison is what answers the real question.

    The prediction departure is the arithmetic path; whether any published
    quantity moved is the estimand. A gate that raised before the increments were
    computed made the second question unanswerable from the artefact.
    """
    request = _request('readout_panel')
    failed = RC.pipeline_agreement({'B_R': 2.9e-6}, 1e-8)
    record = RC.cell_report(request, identity={'identical': True}, provenance={},
                            comparison=[], predictions=failed, blocking={})
    assert record['verdict'] == 'refused'
    assert 'B_R at 2.9e-06' in record['refusal']
    passed = RC.pipeline_agreement({'B_R': 0.0}, 1e-8)
    assert RC.cell_report(request, identity={}, provenance={}, comparison=[],
                          predictions=passed, blocking={})['verdict'] == 'recomputed'


def test_summary_agreement_reports_every_shared_quantity_and_names_a_sign_change():
    unit = 'wild-type family at 50% identity'
    published = dict(summaries={
        'a': dict(point=0.5, interval=[0.1, 0.9], unit=unit, n_units=163, excludes_zero=True,
                  resamples=2000),
        'b': dict(point=-0.2, interval=[-0.4, -0.05], unit=unit, n_units=163, excludes_zero=True,
                  resamples=2000),
        'c': dict(point=0.0, interval=[-0.1, 0.1], unit=unit, n_units=163, excludes_zero=False,
                  resamples=2000)})
    identical = dict(summaries={k: dict(v) for k, v in published['summaries'].items()})
    exact = RC.summary_agreement(published, identical)
    assert exact['every_published_quantity_bit_identical'] is True
    assert exact['shared_metrics'] == exact['recovered_exactly'] == 3
    assert exact['max_absolute_point_change'] == 0.0
    assert exact['max_absolute_interval_endpoint_change'] == 0.0
    assert exact['resolved_sign_changes'] == []
    moved = dict(summaries={k: dict(v) for k, v in published['summaries'].items()})
    moved['summaries']['b'] = dict(moved['summaries']['b'], point=0.3, interval=[0.05, 0.6])
    changed = RC.summary_agreement(published, moved)
    assert changed['every_published_quantity_bit_identical'] is False
    assert changed['recovered_exactly'] == 2
    assert changed['resolved_sign_changes'] == ['b: below_zero to above_zero']
    assert changed['max_absolute_point_change'] == pytest.approx(0.5)
    assert changed['max_absolute_interval_endpoint_change'] == pytest.approx(0.65)
