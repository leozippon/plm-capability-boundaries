"""The prediction side-effect's declaration, its estimand, its matched support and its band.

Fixtures are synthetic ladder rows, so the invariants are tested without a scored arm. One
test reads the real checkpoint of the smallest tuned arm where it is present, because the
packing it is scored through is the one thing a synthetic fixture cannot witness.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.capability.extensions import bidirectional
from src.capability.extensions import explicit_statistics_lanes as lanes
from src.capability.extensions import objective_tuned_prediction as endpoint
from src.capability.extensions import statistical_ladder as ladder
from src.capability.generation import objective_tuned_arms as tuned_arms

ROOT = Path(__file__).resolve().parents[2]


def ladder_rows(*, forward=None, reverse=None, groups=12, seeds=ladder.SEEDS):
    """Per-assay ladder rows whose increments are known: controls 0.40, joint = 0.40 + f."""
    forward, reverse = forward or {}, reverse or {}
    rows = []
    for pair in endpoint.admitted_pairs():
        for arm in pair.arms:
            f, r = forward.get(arm, 0.02), reverse.get(arm, 0.30)
            for seed in seeds:
                for group in range(groups):
                    wobble = 0.001 * ((group * 7 + seed) % 5)
                    peg = 0.40 + wobble
                    rows.append(dict(assay=f'a{group}', cluster=str(group), model=arm, seed=seed,
                                     spearman={'PEG': peg, 'MPEG': peg + f, 'M': peg + f - r},
                                     rank_mse={'PEG': 1., 'MPEG': 1., 'M': 1.}))
    return rows


CONTROLS = 0.40


def supports(digest='d' * 64):
    arms = {name for p in endpoint.admitted_pairs() for name in p.arms}
    return {name: digest for name in arms | {'controls'}}


# ------------------------------------------------------------ the declaration


def test_the_family_is_the_admitted_pairs_crossed_with_the_two_directions():
    assert endpoint.family_size() == 2 * len(endpoint.ADMITTED) == len(endpoint.labels())
    assert tuple(endpoint.DIRECTIONS.values()) == bidirectional.DIRECTIONS == tuple(ladder.DIRECTIONS)


def test_admitted_and_not_measured_partition_the_declared_pairs():
    declared = {p.tuned for p in tuned_arms.PAIRS}
    assert set(endpoint.ADMITTED) | set(endpoint.NOT_MEASURED_REASONS) == declared
    assert not set(endpoint.ADMITTED) & set(endpoint.NOT_MEASURED_REASONS)
    assert all(endpoint.NOT_MEASURED_REASONS.values())


def test_the_control_set_is_profile_potts_and_gemme_and_nothing_else():
    letters = {x for role in endpoint.CONTROL_ROLES for x in lanes.VARIANT_EFFECT_ROLES[role]}
    declared = ladder.CONTROL_SETS[endpoint.CONTROL_SET]
    assert set(ladder.DESIGNS[declared['control']]) == letters
    assert set(ladder.DESIGNS[declared['joint']]) == letters | {'M'}


def test_the_disposition_maps_onto_exactly_the_four_states():
    assert set(endpoint.DISPOSITION) == set(endpoint.STATES) and len(endpoint.STATES) == 4
    assert endpoint.NOT_MEASURED not in endpoint.STATES


def test_no_tuned_arm_joins_the_published_roster_and_panel_bases_reuse_their_archive():
    assert not {p.tuned for p in endpoint.admitted_pairs()} & set(lanes.PANEL_ARMS)
    for p in endpoint.admitted_pairs():
        assert endpoint.reuses_archive(p.base) == (p.base in lanes.PANEL_ARMS)


def test_the_published_roster_is_the_ladders_own_thirty_three_arms():
    path = ROOT / 'results/extensions/statistical_ladder_20261106/full/contract.json'
    if not path.is_file():
        pytest.skip('published ladder contract absent')
    assert set(json.loads(path.read_text())['scope']['models']) == set(lanes.PANEL_ARMS)


def test_the_instrument_floor_is_a_measured_correlation_not_a_score_bound():
    assert 0 < endpoint.INSTRUMENT_FLOOR < 0.05
    assert endpoint.DECLARATION['instrument_floor']['basis']


def test_the_list_refuses_to_load_against_a_changed_cost_estimate(tmp_path):
    declaration = json.loads(endpoint.DECLARATION_PATH.read_text())
    declaration['cost_estimate']['sha256'] = hashlib.sha256(b'other').hexdigest()
    path = tmp_path / 'declaration.json'
    path.write_text(json.dumps(declaration))
    with pytest.raises(RuntimeError, match='no longer matches the digest'):
        endpoint.load_declaration(path)


def test_a_pair_cannot_be_dropped_without_its_reason(monkeypatch):
    monkeypatch.setattr(endpoint, 'ADMITTED', endpoint.ADMITTED[:-1])
    with pytest.raises(RuntimeError, match='partition'):
        endpoint._check_declarations()


def test_a_tuned_arm_on_the_published_roster_is_refused(monkeypatch):
    monkeypatch.setattr(lanes, 'PANEL_ARMS', (*lanes.PANEL_ARMS, endpoint.ADMITTED[0]))
    with pytest.raises(RuntimeError, match='published ladder'):
        endpoint._check_declarations()


# --------------------------------------------------------------- the estimand


def test_the_difference_is_the_tuned_increment_minus_the_base_increment():
    pair = endpoint.admitted_pairs()[0]
    rows = ladder_rows(forward={pair.base: 0.01, pair.tuned: 0.05},
                       reverse={pair.base: 0.30, pair.tuned: 0.20})
    out = endpoint.difference_matrix(rows)
    forward = out['matrix'][:, out['labels'].index((pair.tuned, endpoint.FORWARD))]
    reverse = out['matrix'][:, out['labels'].index((pair.tuned, endpoint.REVERSE))]
    assert np.allclose(forward, 0.04) and np.allclose(reverse, -0.10)


def test_an_unrelated_pair_does_not_move_a_pairs_difference():
    first, second = endpoint.admitted_pairs()[:2]
    out = endpoint.difference_matrix(ladder_rows(forward={second.tuned: 0.09}))
    assert np.allclose(out['matrix'][:, out['labels'].index((first.tuned, endpoint.FORWARD))], 0)


def test_an_incomplete_cell_raises_instead_of_dropping_a_group():
    rows = ladder_rows()
    with pytest.raises(ValueError, match='metric support'):
        endpoint.difference_matrix(rows[:-1])


def test_an_undefined_correlation_raises():
    rows = ladder_rows()
    rows[0]['spearman']['M'] = float('nan')
    with pytest.raises(ValueError, match='undefined|nonfinite'):
        endpoint.difference_matrix(rows)


# -------------------------------------------------------------------- support


def test_matched_support_is_the_intersection_and_names_what_each_arm_lost():
    controls = np.array([1, 1, 1, 1, 1, 0], bool)
    out = endpoint.matched_support({'a': np.array([1, 1, 0, 1, 1, 1], bool),
                                    'b': np.array([1, 1, 1, 0, 1, 1], bool)}, controls)
    assert out['mask'].tolist() == [True, True, False, False, True, False]
    assert out['rows'] == 3 and out['lost'] == {'a': 1, 'b': 1} and out['lost_to_controls'] == 1


def test_an_empty_or_misshapen_support_is_refused():
    with pytest.raises(ValueError, match='empty'):
        endpoint.matched_support({'a': np.zeros(4, bool)}, np.ones(4, bool))
    with pytest.raises(ValueError, match='same rows'):
        endpoint.matched_support({'a': np.ones(3, bool)}, np.ones(4, bool))


# ----------------------------------------------------------------------- band


def band_of(rows):
    out = endpoint.difference_matrix(rows)
    return endpoint.band(out['matrix'], out['labels'], supports(), controls_performance=CONTROLS)


def test_the_band_refuses_unmatched_support():
    out = endpoint.difference_matrix(ladder_rows())
    mixed = supports()
    mixed[endpoint.admitted_pairs()[0].tuned] = 'e' * 64
    with pytest.raises(ValueError, match='unmatched support'):
        endpoint.band(out['matrix'], out['labels'], mixed, controls_performance=CONTROLS)
    controls_off = dict(supports(), controls='f' * 64)
    with pytest.raises(ValueError, match='unmatched support'):
        endpoint.band(out['matrix'], out['labels'], controls_off, controls_performance=CONTROLS)


def test_the_band_needs_a_digest_for_every_arm_and_the_declared_columns():
    out = endpoint.difference_matrix(ladder_rows())
    partial = supports()
    partial.pop('controls')
    with pytest.raises(ValueError, match='support digest'):
        endpoint.band(out['matrix'], out['labels'], partial, controls_performance=CONTROLS)
    with pytest.raises(ValueError, match='declared'):
        endpoint.band(out['matrix'][:, :-1], out['labels'][:-1], supports(), controls_performance=CONTROLS)
    with pytest.raises(ValueError, match='declared'):
        endpoint.band(out['matrix'], out['labels'][::-1], supports(), controls_performance=CONTROLS)


def test_no_difference_is_bounded_narrowing_with_the_fraction_reported():
    result = band_of(ladder_rows())
    assert result['status'] == 'complete' and result['family_size'] == endpoint.family_size()
    assert {c['state'] for c in result['columns']} == {endpoint.BOUNDED_NEGATIVE}
    assert {p['reading'] for p in result['pairs']} == {endpoint.NARROWING_BOUNDED}
    for c in result['columns']:
        low, high = c['interval']
        assert c['extent_fraction'] == pytest.approx(max(abs(low), abs(high)) / CONTROLS)
        assert c['interval_fraction'] == pytest.approx([low / CONTROLS, high / CONTROLS])


def test_an_interval_too_wide_to_exclude_the_controls_own_performance_is_unresolved():
    wide = (-CONTROLS * 1.5, CONTROLS * 1.5)
    got = endpoint.column_state(0.0, wide, CONTROLS)
    assert got['state'] == endpoint.UNRESOLVED and got['extent_fraction'] == pytest.approx(1.5)
    narrow = endpoint.column_state(0.0, (-0.02, 0.02), CONTROLS)
    assert narrow['state'] == endpoint.BOUNDED_NEGATIVE
    forward, reverse = got, narrow
    assert endpoint.reading(forward, reverse) == endpoint.READING_UNRESOLVED


def test_a_resolved_forward_rise_is_added_information_and_a_fall_is_narrowing_with_cost():
    first, second = endpoint.admitted_pairs()[:2]
    rows = ladder_rows(forward={first.tuned: 0.20, second.tuned: -0.20})
    readings = {p['tuned']: p['reading'] for p in band_of(rows)['pairs']}
    assert readings[first.tuned] == endpoint.ADDED_INFORMATION
    assert readings[second.tuned] == endpoint.NARROWING_WITH_COST


def test_a_reverse_only_rise_is_the_strongest_narrowing_not_the_bounded_case():
    pair = endpoint.admitted_pairs()[0]
    rows = ladder_rows(reverse={pair.tuned: 0.60})
    got = {p['tuned']: p for p in band_of(rows)['pairs']}[pair.tuned]
    assert got['forward'] == endpoint.BOUNDED_NEGATIVE and got['reverse'] == endpoint.SUPPORTED
    assert got['reading'] == endpoint.NARROWING_LOSS_OF_OVERLAP != endpoint.NARROWING_BOUNDED


def test_a_reverse_fall_with_a_bounded_forward_has_no_declared_reading():
    pair = endpoint.admitted_pairs()[0]
    rows = ladder_rows(reverse={pair.tuned: 0.05})
    got = {p['tuned']: p for p in band_of(rows)['pairs']}[pair.tuned]
    assert got['reading'] == endpoint.READING_UNRESOLVED


def test_there_are_five_readings():
    assert len(endpoint.READINGS) == 5


def test_nothing_under_the_instrument_floor_is_resolved():
    tiny = endpoint.INSTRUMENT_FLOOR / 2
    assert not endpoint.column_state(tiny, (tiny / 10, tiny * 2), CONTROLS)['rises']
    assert not endpoint.column_state(-tiny, (-tiny * 2, -tiny / 10), CONTROLS)['falls']
    big = endpoint.INSTRUMENT_FLOOR * 4
    assert endpoint.column_state(big, (big / 2, big * 2), CONTROLS)['state'] == endpoint.SUPPORTED
    assert endpoint.column_state(-big, (-big * 2, -big / 2), CONTROLS)['falls']


def test_the_controls_must_perform_above_zero():
    with pytest.raises(ValueError, match='above zero'):
        endpoint.column_state(0.0, (-0.1, 0.1), 0.0)


def test_a_stratum_under_the_bootstrap_unit_floor_is_unresolved_without_an_interval():
    result = band_of(ladder_rows(groups=3))
    assert result['status'] == endpoint.UNRESOLVED and 'columns' not in result


def test_the_band_is_the_shared_native_construction_over_the_declared_family():
    out = endpoint.difference_matrix(ladder_rows(forward={endpoint.ADMITTED[0]: 0.03}))
    shared = bidirectional.paired_bands(out['matrix'])
    result = endpoint.band(out['matrix'], out['labels'], supports(), controls_performance=CONTROLS)
    assert result['resolution'] == shared['critical_value']
    assert result['draws'] == bidirectional.DRAWS and result['seed'] == bidirectional.SEED
    assert [c['point'] for c in result['columns']] == shared['point']
    assert result['controls_performance'] == CONTROLS


# ----------------------------------------------------------- the scoring path


def test_every_admitted_arm_has_a_packing_family_the_context_module_knows():
    from scripts.capability.extensions import score_objective_tuned_prediction as script
    from src.capability.context import context_homologue as ch
    for pair in endpoint.admitted_pairs():
        for arm in pair.arms:
            family = script.PACKING_REPRESENTATIVE[tuned_arms.spec_for(arm).input_format]
            ch.packing_of(family)


def test_the_smallest_tuned_arm_scores_as_a_direct_forward_pass_does():
    torch = pytest.importorskip('torch')
    name = 'protgpt3-112m-dpo'
    if not tuned_arms.weights_complete(name):
        pytest.skip('checkpoint not staged')
    from scripts.capability.extensions import score_objective_tuned_prediction as script
    scorer = script.Scorer(name, 'cpu')
    sequence = 'MKTAYIAKQRQISFVKSHFSRQ'
    got = scorer.log_likelihood([sequence])[0]
    tokenizer, model = scorer.arm.tokenizer, scorer.arm.model
    ids = [tokenizer.bos_token_id, tokenizer.convert_tokens_to_ids('1'),
           *tokenizer(sequence, add_special_tokens=False)['input_ids']]
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0].float()
    logp = torch.log_softmax(logits[1:-1], -1)
    want = float(logp.gather(-1, torch.tensor(ids[2:])[:, None]).sum())
    assert got == pytest.approx(want, abs=1e-3)
    assert scorer.log_likelihood([sequence])[0] == got


def test_family_mean_reads_one_arms_rows_out_of_the_whole_run():
    """``run`` hands it every arm's rows; the ladder's cell check refuses them unfiltered."""
    from scripts.capability.extensions import score_objective_tuned_prediction as script
    arms = [name for p in endpoint.admitted_pairs() for name in p.arms]
    rows = ladder_rows(forward={arms[0]: 0.05})
    assert len({r['model'] for r in rows}) == len(arms) > 1
    got = script.family_mean(rows, arms[0], 'MPEG')
    want = float(np.mean([r['spearman']['MPEG'] for r in rows if r['model'] == arms[0]]))
    assert got == pytest.approx(want)
    assert script.family_mean(rows, arms[1], 'MPEG') != pytest.approx(got)
