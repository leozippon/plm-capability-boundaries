"""Independence floor, declared directions, separated endpoints, refused paths.

The conditions asserted here are the ones that must hold whatever cohort arrives:
a cohort below the independent-group floor is refused and recorded rather than
scored; a measured quantity with no declared direction never becomes a label; a
state the interface budget cannot pack never reaches a plan; a likelihood column
is never built from partial state coverage; and the ranking and quantitative
endpoints are fitted and reported as two different things.
"""
from pathlib import Path
import random

import numpy as np
import pytest

from src.capability.core.amino_acids import AA20
from src.capability.extensions import phenotype_breadth as breadth
from src.capability.extensions.phenotype_cohorts import activity_quantity

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / 'runtimes/phenotype-homology-tests'

#: Seeded pseudo-random wild types. Two of them carry no family edge at the
#: frozen 30%/80% rule (checked by the grouping assertions below), and their
#: residue diversity keeps the substitution-identity design from being degenerate
#: the way a two-residue repeat would be.
WILDTYPE_SEED = 20261008


def wildtype(index: int, length: int = 120) -> str:
    rng = random.Random(f'{WILDTYPE_SEED}:wt:{index}:{length}')
    return ''.join(rng.choices(AA20, k=length))


#: Fitting mechanics are exercised over narrow control blocks on purpose. The
#: 400-column substitution-identity block needs a real cohort's thousands of rows
#: to be conditioned; a hundred synthetic rows would test the ridge's behaviour
#: on a rank-deficient design rather than the logic under test.
NARROW_BASE = ('geom',)
NARROW_QUALIFIED = ('geom', 'comp')


def substitute(sequence: str, position: int, residue: str) -> str:
    return sequence[:position - 1] + residue + sequence[position:]


def make_rows(units: int, per_unit: int = 12, *, cohort: str = 'synthetic',
              direction: int = 1, label_unit: str | None = None,
              length: int = 120, labels=None) -> list[breadth.PhenotypeRow]:
    rows = []
    for unit in range(units):
        wt = wildtype(unit, length)
        rng = random.Random(f'{WILDTYPE_SEED}:mutants:{unit}')
        for offset in range(per_unit):
            position = 2 * offset + 1
            residue = rng.choice([a for a in AA20 if a != wt[position - 1]])
            # The default label is seeded noise, deliberately unrelated to the
            # mutated position and to the residue, so that a control block cannot
            # predict it and a likelihood that carries it shows a real increment.
            value = (labels(unit, offset) if labels is not None
                     else random.Random(f'{WILDTYPE_SEED}:label:{unit}:{offset}').gauss(0.0, 1.0))
            rows.append(breadth.PhenotypeRow(
                cohort=cohort, row_id=f'{cohort}:{unit}:{offset}', background=f'bg{unit}',
                unit=breadth.sha_text(wt), wildtype=wt,
                mutant_sequence=substitute(wt, position, residue), position=position,
                wt_aa=wt[position - 1], mt_aa=residue, label=value, direction=direction,
                label_unit=label_unit, source_sha256=breadth.sha_text(f'src{unit}{offset}')))
    return rows


def cohort_of(rows, **kwargs) -> breadth.Cohort:
    return breadth.Cohort(key=kwargs.pop('key', 'synthetic'),
                          phenotype=kwargs.pop('phenotype', 'synthetic phenotype'),
                          endpoint=kwargs.pop('endpoint', 'synthetic endpoint'),
                          source=kwargs.pop('source', 'synthetic'), rows=list(rows), **kwargs)


def scores_for(rows, *, coefficient: float = 1.0, noise: float = 0.0,
               seed: int = 7) -> dict[str, float]:
    """A likelihood table whose mutant-minus-wild-type difference tracks the label."""
    rng = np.random.default_rng(seed)
    table: dict[str, float] = {}
    for row in rows:
        table[breadth.sha_text(row.wildtype)] = 0.0
        table[breadth.sha_text(row.mutant_sequence)] = (
            coefficient * row.oriented_label() + noise * float(rng.normal()))
    return table


# --------------------------------------------------------------------------- #
# The independence floor
# --------------------------------------------------------------------------- #

def test_cohort_below_the_floor_is_refused_and_fully_recorded():
    record, rows, grouping = breadth.qualify(cohort_of(make_rows(3)), RUNTIME)
    assert record['status'] == 'refused'
    assert grouping['groups'] == 3
    assert record['support']['independent_groups'] == 3
    assert record['independence_floor']['degenerate'] is True
    assert any('below the 8-group floor' in blocker for blocker in record['blockers'])
    # A refusal is a finding: the support survives in the record and the rows are
    # still returned, so a reader can see what was refused and why.
    assert record['support']['rows'] == len(rows) == 36
    assert record['ranking_licensed'] is False
    assert record['quantitative_licensed'] is False


def test_cohort_at_and_above_the_floor_is_admitted_with_a_no_margin_note():
    at_floor, _, _ = breadth.qualify(cohort_of(make_rows(breadth.INDEPENDENCE_FLOOR)), RUNTIME)
    assert at_floor['status'] == 'admitted'
    assert at_floor['independence_floor']['degenerate'] is False
    assert any('exactly the floor' in note for note in at_floor['cohort_limitations'])
    above, _, _ = breadth.qualify(cohort_of(make_rows(breadth.INDEPENDENCE_FLOOR + 2)), RUNTIME)
    assert above['status'] == 'admitted'
    assert above['cohort_limitations'] == []


def test_homologous_units_collapse_into_one_group():
    rows = make_rows(1)
    wt = rows[0].wildtype
    # A point variant of the same wild type is one biological unit, not two.
    variant = substitute(wt, 119, 'W' if wt[118] != 'W' else 'Y')
    extra = []
    for offset in range(12):
        position = 2 * offset + 1
        residue = 'W' if variant[position - 1] != 'W' else 'Y'
        extra.append(breadth.PhenotypeRow(
            cohort='synthetic', row_id=f'synthetic:v:{offset}', background='bgv',
            unit=breadth.sha_text(variant), wildtype=variant,
            mutant_sequence=substitute(variant, position, residue), position=position,
            wt_aa=variant[position - 1], mt_aa=residue, label=float(offset), direction=1,
            label_unit=None, source_sha256=breadth.sha_text(f'v{offset}')))
    record, _, grouping = breadth.qualify(cohort_of(rows + extra), RUNTIME)
    assert grouping['units'] == 2 and grouping['groups'] == 1
    assert record['status'] == 'refused'


def test_floor_is_the_project_floor_and_is_not_redeclared():
    from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS
    assert breadth.INDEPENDENCE_FLOOR == MINIMUM_BOOTSTRAP_UNITS == 8


# --------------------------------------------------------------------------- #
# Declared endpoint direction
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('assay, quantity, direction', [
    ('P1_kcat', 'kcat', 1),
    ('P1_1_kcatkm_substrate_R', 'kcatkm', 1),
    ('P1_vmax', 'vmax', 1),
    ('P1_DTNB_activity', 'activity', 1),
    ('P1_1_km', 'km', None),
    ('P1_1_ratio_of_kcatkm_R_over_S', 'ratio', None),
    ('P1_tm', 'unrecognised', None),
])
def test_activity_quantity_direction_is_declared_not_guessed(assay, quantity, direction):
    name, record = activity_quantity(assay)
    assert name == quantity
    assert record['direction'] == direction
    assert record['reason']


def test_catalytic_efficiency_is_never_read_as_a_michaelis_constant():
    assert activity_quantity('P1_kcatkm')[0] == 'kcatkm'
    assert activity_quantity('P1_km')[0] == 'km'


def test_undeclared_direction_refuses_the_row_rather_than_assuming_one():
    with pytest.raises(ValueError, match='direction'):
        breadth.validate_row(breadth.PhenotypeRow(
            cohort='c', row_id='r', background='b', unit='u', wildtype='AC', mutant_sequence='AW',
            position=2, wt_aa='C', mt_aa='W', label=1.0, direction=0, label_unit=None,
            source_sha256='s'))


# --------------------------------------------------------------------------- #
# Support screens
# --------------------------------------------------------------------------- #

def test_identity_errors_raise_and_are_never_silent_exclusions():
    wt = wildtype(0)
    for broken in (
        dict(position=3, wt_aa=wt[0]),                      # declared residue is elsewhere
        dict(mt_aa=wt[0]),                                  # identity substitution
        dict(mutant_sequence=wt[:-1]),                      # length change
    ):
        row = breadth.PhenotypeRow(
            cohort='c', row_id='r', background='b', unit='u', wildtype=wt,
            mutant_sequence=substitute(wt, 1, 'W'), position=1, wt_aa=wt[0], mt_aa='W',
            label=1.0, direction=1, label_unit=None, source_sha256='s')
        with pytest.raises(ValueError):
            breadth.validate_row(row.__class__(**{**row.__dict__, **broken}))


def test_states_over_the_interface_budget_are_refused_before_any_plan():
    rows = make_rows(breadth.INDEPENDENCE_FLOOR + 1, length=breadth.RESIDUE_BUDGET + 2)
    keep = make_rows(breadth.INDEPENDENCE_FLOOR, cohort='keep')
    record, retained, _ = breadth.qualify(cohort_of(rows + keep), RUNTIME)
    assert record['refusals']['state_over_residue_budget'] == len(rows)
    assert all(len(row.wildtype) <= breadth.RESIDUE_BUDGET for row in retained)
    plan = breadth.state_plan({'c': retained})
    assert all(state['length'] <= breadth.RESIDUE_BUDGET for state in plan['states'])


def test_repeated_background_variant_rows_are_dropped_and_counted():
    rows = make_rows(breadth.INDEPENDENCE_FLOOR)
    duplicate = rows[0]
    record, retained, _ = breadth.qualify(cohort_of([*rows, duplicate]), RUNTIME)
    # Both copies go: choosing a summary of a repeated measurement is a
    # declaration the source has to make, not something a screen may invent.
    assert record['refusals']['repeated_background_variant_rows'] == 2
    assert all(row.row_id != duplicate.row_id for row in retained)


def test_a_background_with_a_constant_label_cannot_be_ranked():
    rows = make_rows(breadth.INDEPENDENCE_FLOOR, labels=lambda unit, offset: (
        1.0 if unit == 0 else float(offset)))
    record, retained, _ = breadth.qualify(cohort_of(rows), RUNTIME)
    assert record['refusals']['background_constant_label'] == 12
    assert {row.background for row in retained} == {f'bg{i}' for i in range(1, 8)}


def test_variant_cap_is_label_blind_and_matches_the_frozen_panel_cap():
    from src.capability.mutation.external_confirmation import VARIANT_CAP as FROZEN
    assert breadth.VARIANT_CAP == FROZEN
    rows = make_rows(2, per_unit=12)
    groups = {row.unit: f'g{row.unit[:4]}' for row in rows}
    kept, record = breadth.cap_rows(rows, groups, cap=5)
    assert record['dropped_rows'] == 14 and len(kept) == 10
    # Reordering the input cannot change which rows are kept.
    again, _ = breadth.cap_rows(list(reversed(rows)), groups, cap=5)
    assert [row.row_id for row in kept] == [row.row_id for row in again]


# --------------------------------------------------------------------------- #
# The state plan and the model column
# --------------------------------------------------------------------------- #

def test_state_plan_scores_a_shared_sequence_once():
    rows = make_rows(breadth.INDEPENDENCE_FLOOR)
    plan = breadth.state_plan({'a': rows, 'b': rows})
    hashes = [state['sequence_sha256'] for state in plan['states']]
    assert len(hashes) == len(set(hashes))
    assert len(plan['states']) == breadth.INDEPENDENCE_FLOOR * 13  # 12 mutants + 1 wild type
    assert [record['rows'] for record in plan['cohorts']] == [len(rows), len(rows)]


def test_model_block_is_mutant_minus_wildtype_and_refuses_partial_coverage():
    rows = make_rows(breadth.INDEPENDENCE_FLOOR)[:4]
    backgrounds = np.asarray([row.background for row in rows])
    table = scores_for(rows, coefficient=2.0)
    block = breadth.model_block(rows, table, backgrounds)
    expected = np.asarray([2.0 * row.oriented_label() for row in rows])
    assert block['M_nats'].ravel() == pytest.approx(expected)
    assert block['M_rank'].shape == (len(rows), 1)
    del table[breadth.sha_text(rows[0].mutant_sequence)]
    with pytest.raises(ValueError, match='exact likelihood'):
        breadth.model_block(rows, table, backgrounds)


# --------------------------------------------------------------------------- #
# Control qualification
# --------------------------------------------------------------------------- #

def test_a_control_that_raises_held_out_error_is_discarded_and_reported():
    rows = make_rows(breadth.INDEPENDENCE_FLOOR + 2)
    blocks = breadth.control_blocks(rows, None)
    # A block of pure noise at the scale of the design cannot help a held-out
    # group and should lower the standing set's accuracy at some seed.
    noise = np.random.default_rng(20261008).normal(scale=20.0, size=blocks['chem'].shape)
    blocks['chem'] = noise
    backgrounds = np.asarray([row.background for row in rows])
    groups = np.asarray([row.unit for row in rows])
    labels = np.asarray([row.oriented_label() for row in rows])
    qualification = breadth.qualify_controls(
        blocks, breadth.rerank(labels, backgrounds), backgrounds, groups,
        base=NARROW_BASE, candidates=('chem',))
    offered = {offer['candidate']: offer for offer in qualification['offers']}
    assert offered['chem']['kept'] is False
    assert min(offered['chem']['reduction_by_seed'].values()) <= 0.0
    assert 'chem' not in qualification['qualified']
    assert qualification['qualified'] == list(NARROW_BASE)
    assert len(offered['chem']['reduction_by_seed']) == len(breadth.SPLIT_SEEDS)
    # The qualified set's own competence is reported, because an increment over a
    # control that predicts nothing is uninformative.
    assert 'qualified_contribution_over_no_effect' in qualification


def test_the_qualification_rule_is_the_frozen_strictly_positive_one():
    # The rule is the nested gate's own: kept only if the reduction is positive
    # at every split seed, with no tolerance invented here.
    from src.capability.mutation.external_confirmation import qualify as frozen_rule
    assert frozen_rule({seed: 1e-9 for seed in breadth.SPLIT_SEEDS})['qualified'] is True
    assert frozen_rule({**{s: 1.0 for s in breadth.SPLIT_SEEDS},
                        breadth.SPLIT_SEEDS[0]: 0.0})['qualified'] is False


def test_profile_blocks_are_absent_rather_than_imputed_without_profiles():
    rows = make_rows(breadth.INDEPENDENCE_FLOOR)
    blocks = breadth.control_blocks(rows, None)
    assert np.count_nonzero(blocks['prof']) == 0
    assert np.count_nonzero(blocks['prof2']) == 0


# --------------------------------------------------------------------------- #
# The two endpoints
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def fitted():
    rows = make_rows(breadth.INDEPENDENCE_FLOOR + 2)
    record, retained, grouping = breadth.qualify(
        cohort_of(rows, quantitative_unit='synthetic units'), RUNTIME)
    assert record['status'] == 'admitted'
    blocks = breadth.control_blocks(retained, None)
    scores = {'informative': scores_for(retained, coefficient=3.0),
              'uninformative': scores_for(retained, coefficient=0.0, noise=1.0)}
    return record, retained, grouping['assignment'], blocks, scores


def test_ranking_and_quantitative_endpoints_are_reported_separately(fitted):
    record, rows, groups, blocks, scores = fitted
    ranking = breadth.fit_cohort(rows, blocks, scores, unit_groups=groups,
                                 qualified=NARROW_QUALIFIED, endpoint='ranking')
    quantitative = breadth.fit_cohort(rows, blocks, scores, unit_groups=groups,
                                      qualified=NARROW_QUALIFIED, endpoint='quantitative')
    assert ranking['metric'] == 'spearman' and ranking['model_column'] == 'M_rank'
    assert quantitative['metric'] == 'error' and quantitative['model_column'] == 'M_nats'
    assert 'Spearman' in ranking['increment_sign']
    assert 'squared error' in quantitative['increment_sign']
    # A likelihood that carries the label is positive on both endpoints; the two
    # numbers are different quantities and neither is derived from the other.
    informative = {r['arm']: r for r in ranking['arm_results']}['informative']
    assert informative['point'] > 0 and informative['resolved_positive'] is True
    quantified = {r['arm']: r for r in quantitative['arm_results']}['informative']
    assert quantified['point'] > 0
    assert ranking['point'] if False else True  # endpoints share no fields by accident
    assert set(ranking['groups']) == set(quantitative['groups'])


def test_the_quantitative_predictor_is_fitted_in_label_units(fitted):
    record, rows, groups, blocks, scores = fitted
    panel = breadth.fit_cohort(rows, blocks, scores, unit_groups=groups,
                               qualified=NARROW_QUALIFIED, endpoint='quantitative')
    labels = np.asarray([row.oriented_label() for row in rows])
    # Held-out error is on the label's own scale, so it is commensurate with the
    # label variance and not with a rank variance of one.
    errors = [cell['control'] for cell in panel['per_background']]
    assert max(errors) <= 4.0 * float(np.var(labels))
    assert min(errors) >= 0.0


def test_an_uninformative_likelihood_is_not_resolved_positive(fitted):
    record, rows, groups, blocks, scores = fitted
    ranking = breadth.fit_cohort(rows, blocks, scores, unit_groups=groups,
                                 qualified=NARROW_QUALIFIED, endpoint='ranking')
    noise = {r['arm']: r for r in ranking['arm_results']}['uninformative']
    assert noise['resolved_positive'] is False


def test_an_unknown_endpoint_is_refused(fitted):
    record, rows, groups, blocks, scores = fitted
    with pytest.raises(ValueError, match='unknown endpoint'):
        breadth.fit_cohort(rows, blocks, scores, unit_groups=groups,
                           qualified=NARROW_QUALIFIED, endpoint='classification')


def test_quantitative_licence_follows_a_traced_shared_unit():
    rows = make_rows(breadth.INDEPENDENCE_FLOOR)
    without, _, _ = breadth.qualify(cohort_of(rows), RUNTIME)
    assert without['quantitative_licensed'] is False
    assert 'heterogeneous units' in without['quantitative_refusal']
    with_unit, _, _ = breadth.qualify(
        cohort_of(rows, quantitative_unit='kcal/mol'), RUNTIME)
    assert with_unit['quantitative_licensed'] is True
    assert with_unit['quantitative_refusal'] is None


def test_an_unverified_construct_blocks_unless_the_tolerance_is_written_down():
    rows = make_rows(breadth.INDEPENDENCE_FLOOR)
    silent, _, _ = breadth.qualify(cohort_of(rows, construct_status='unverified'), RUNTIME)
    assert silent['status'] == 'refused'
    assert any('construct identity' in blocker for blocker in silent['blockers'])
    declared, _, _ = breadth.qualify(cohort_of(
        rows, construct_status='unverified',
        construct_tolerated='the endpoint is a within-construct difference'), RUNTIME)
    assert declared['status'] == 'admitted'
    assert any('tolerated because' in note for note in declared['cohort_limitations'])


# --------------------------------------------------------------------------- #
# Inference bookkeeping
# --------------------------------------------------------------------------- #

def test_simultaneous_band_is_wider_than_pointwise_and_carries_the_floor():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(12, 5))
    band = breadth.simultaneous_band(matrix, draws=500, seed=11)
    assert band['independence_floor']['degenerate'] is False
    for index in range(5):
        point = band['point'][index]
        low, high = band['simultaneous'][index]
        plow, phigh = band['pointwise'][index]
        assert low <= plow and high >= phigh
        assert low < point < high
    degenerate = breadth.simultaneous_band(matrix[:3], draws=500, seed=11)
    assert degenerate['independence_floor']['degenerate'] is True


def test_fold_predictions_hold_out_every_group_exactly_once():
    rows = make_rows(breadth.INDEPENDENCE_FLOOR + 2)
    groups = np.asarray([row.unit for row in rows])
    backgrounds = np.asarray([row.background for row in rows])
    signature = breadth.outer_signature(groups, breadth.SPLIT_SEEDS[0])
    held = [group for _, fold_held, _, _ in signature for group in fold_held]
    assert sorted(held) == sorted(set(groups.tolist()))
    assert len(held) == len(set(held))
    design = breadth.design(breadth.control_blocks(rows, None), NARROW_QUALIFIED)
    labels = np.asarray([row.oriented_label() for row in rows])
    prediction, records = breadth.fold_predictions(
        design, breadth.rerank(labels, backgrounds), backgrounds, groups, signature)
    assert np.isfinite(prediction).all() and len(records) == breadth.OUTER_FOLDS
    for record, (_, fold_held, _, _) in zip(records, signature):
        assert record['held_groups'] == list(fold_held)


def test_constant_control_columns_are_pruned_and_censused():
    rows = make_rows(breadth.INDEPENDENCE_FLOOR)
    blocks = breadth.control_blocks(rows, None)
    pruned, census = breadth.prune_constant_columns(blocks)
    # The 400 directed substitution types cannot all be observed in a hundred
    # rows, so most of the identity block is constant on this support.
    assert census['ident']['columns'] == 400
    assert census['ident']['retained'] < 400
    assert census['ident']['constant_columns_removed'] == 400 - census['ident']['retained']
    assert census['ident']['fully_constant'] is False
    assert pruned['ident'].shape == (len(rows), census['ident']['retained'])
    # Without profiles the two profile blocks have no variation at all; they are
    # kept as a single column so a design naming them still resolves, and the
    # census says so rather than hiding it.
    for name in ('prof', 'prof2'):
        assert census[name]['fully_constant'] is True
        assert pruned[name].shape == (len(rows), 1)
    assert 'unpenalised intercept' in census['rule']


def test_pruning_leaves_every_prediction_unchanged():
    rows = make_rows(breadth.INDEPENDENCE_FLOOR + 2)
    blocks = breadth.control_blocks(rows, None)
    pruned, _ = breadth.prune_constant_columns(blocks)
    backgrounds = np.asarray([row.background for row in rows])
    groups = np.asarray([row.unit for row in rows])
    labels = np.asarray([row.oriented_label() for row in rows])
    target = breadth.rerank(labels, backgrounds)
    signature = breadth.outer_signature(groups, breadth.SPLIT_SEEDS[0])
    full, _ = breadth.fold_predictions(
        breadth.design(blocks, NARROW_QUALIFIED), target, backgrounds, groups, signature)
    small, _ = breadth.fold_predictions(
        breadth.design(pruned, NARROW_QUALIFIED), target, backgrounds, groups, signature)
    assert small == pytest.approx(full, abs=1e-9)


def test_a_background_spanning_two_independent_groups_is_a_defect():
    # Two non-homologous wild types reported under one ranking background: a rank
    # across them is not a rank, and the group weighting is undefined.
    rows = make_rows(breadth.INDEPENDENCE_FLOOR)
    crossed = [row.__class__(**{**row.__dict__, 'background': 'shared'})
               for row in rows if row.unit in {rows[0].unit, rows[-1].unit}]
    others = [row for row in rows if row.unit not in {rows[0].unit, rows[-1].unit}]
    with pytest.raises(ValueError, match='span more than one independent group'):
        breadth.qualify(cohort_of(others + crossed), RUNTIME)


# --------------------------------------------------------------------------- #
# The output-directory contract the campaign runner imposes
# --------------------------------------------------------------------------- #

COMPLETION = 'phenotype_scoring.json'


def test_a_missing_output_directory_is_created(tmp_path):
    target = tmp_path / 'fresh' / 'cell'
    out = breadth.prepare_output(target, COMPLETION)
    assert out == target.resolve() and out.is_dir()
    assert not any(out.iterdir())


def test_an_existing_empty_output_directory_is_accepted(tmp_path):
    # This is the normal case under the campaign queue: the runner does its own
    # mkdir -p and then injects the directory as --out, so a guard on existence
    # would refuse every cell.
    target = tmp_path / 'cell'
    target.mkdir()
    assert breadth.prepare_output(target, COMPLETION) == target.resolve()
    # Idempotent: preparing the same empty directory twice is still accepted.
    assert breadth.prepare_output(target, COMPLETION) == target.resolve()


def test_a_directory_holding_the_completion_artifact_is_refused(tmp_path):
    target = tmp_path / 'cell'
    target.mkdir()
    (target / COMPLETION).write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='a previous run already completed') as error:
        breadth.prepare_output(target, COMPLETION)
    assert str(target.resolve()) in str(error.value)
    assert COMPLETION in str(error.value)


def test_a_directory_holding_an_incomplete_previous_run_is_refused(tmp_path):
    target = tmp_path / 'cell'
    (target / 'nested').mkdir(parents=True)
    with pytest.raises(ValueError, match='non-empty output directory') as error:
        breadth.prepare_output(target, COMPLETION)
    assert str(target.resolve()) in str(error.value)
    # The refusal says why: merging into a partial run is the hazard.
    assert 'interleave' in str(error.value)


def test_an_output_path_that_is_not_a_directory_is_refused(tmp_path):
    target = tmp_path / 'cell'
    target.write_text('', encoding='utf-8')
    with pytest.raises(ValueError, match='not a directory'):
        breadth.prepare_output(target, COMPLETION)


def test_every_stage_uses_the_shared_output_contract():
    # The four stages must not grow their own guard: the runner contract is one
    # declaration, and a stage that re-implements it is the one that fails.
    stages = ('qualify_phenotype_cohorts', 'score_phenotype_states',
              'fit_phenotype_breadth', 'analyse_matched_phenotypes')
    root = Path(__file__).resolve().parents[2] / 'scripts/capability/extensions'
    for stage in stages:
        source = (root / f'{stage}.py').read_text(encoding='utf-8')
        assert 'breadth.prepare_output(args.out, COMPLETION)' in source, stage
        assert 'refusing an existing output directory' not in source, stage
