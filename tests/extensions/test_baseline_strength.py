"""Strength of the declared new controls: direction, one copy, honest refusals.

Every fixture here is synthetic, because no complete full-stage ladder exists yet.
The tests therefore build the saved family-level absolute matrix themselves, with
member columns repeated verbatim across models exactly as the ladder saves them.
"""
import json
from typing import Any

import numpy as np
import pytest

from src.capability.extensions import statistical_ladder as sl
from scripts.capability.extensions import summarize_baseline_strength as strength

FAMILIES = 163
MODELS = [f'arm{index:02d}' for index in range(33)]
DESIGNS = sl.stage_designs('full')
REUSED = ('M', 'MP', 'BPL', 'BMPL')
FITTED = tuple(name for name in DESIGNS if name not in REUSED)
SHARED = tuple(name for name in FITTED if 'M' not in sl.DESIGNS[name])
HYPOTHESES = len(MODELS)*len(sl.CONTROL_SETS)*len(sl.DIRECTIONS)


def write_stage(tmp_path, *, spearman, rank_mse, status='complete', stage='full',
                families=FAMILIES, drop_design=None, split=None) -> dict[str, Any]:
    """Write one synthetic ladder directory; ``split`` perturbs one repeated column."""
    directory = tmp_path/'stage'
    directory.mkdir(parents=True)
    rng = np.random.default_rng(11)
    designs = [name for name in DESIGNS if name != drop_design]
    labels = [(model, name) for model in MODELS for name in designs]
    levels, absolute = {}, {}
    for metric, means, sign in (('spearman', spearman, 1.), ('rank_mse', rank_mse, -1.)):
        level = {'P': .35 + rng.normal(0, .06, families),
                 'BPL': .45 + rng.normal(0, .06, families)}
        for name, reference, mean in (('PE', 'P', means['PE']), ('PG', 'P', means['PG']),
                                      ('PEG', 'P', means['PEG']),
                                      ('BPLEG', 'BPL', means['BPLEG'])):
            level[name] = level[reference] + sign*mean + rng.normal(0, .01, families)
        for name in ('M', 'MP', 'MPE', 'MPG', 'MPEG', 'BMPL', 'BMPLEG'):
            level[name] = rng.normal(.3, .05, families)
        levels[metric] = level
        absolute[metric] = np.column_stack([level[name] for _ in MODELS for name in designs])
        if split is not None and split[1] in designs:
            model, name, row, delta = split
            absolute[metric][row, MODELS.index(model)*len(designs) + designs.index(name)] += delta
    contrasts = np.zeros((families, HYPOTHESES))
    hypotheses = [(model, name, direction) for model in MODELS
                  for name in sl.CONTROL_SETS for direction in sl.DIRECTIONS]
    for metric in ('spearman', 'rank_mse'):
        np.savez_compressed(
            directory/f'aggregate-{metric}.npz',
            contrasts=contrasts, absolute=absolute[metric],
            families=np.asarray([str(index) for index in range(families)]),
            contrast_labels=np.asarray([json.dumps(list(label)) for label in hypotheses]),
            design_labels=np.asarray([json.dumps(list(label)) for label in labels]))
    fit = status == 'complete' and stage == 'full'
    (directory/'contract.json').write_text(json.dumps(dict(
        analysis='statistical_baseline_ladder_20261106', stage=stage, mode='declared_full',
        ladder_complete=fit,
        scope=dict(models=MODELS, seeds=list(sl.SEEDS), control_sets=list(sl.CONTROL_SETS),
                   designs=dict(fitted=list(FITTED), reused=list(REUSED),
                                shared_across_models=list(SHARED))),
        support=dict(rows=25728, assays=201, families=families, wildtypes=174,
                     support_sha256='a'*64),
        registry=sl.registry(), metrics=list(sl.METRICS),
        hypothesis_family=dict(declared=396, realized=len(MODELS)*len(sl.CONTROL_SETS)*2))))
    (directory/'contrasts.json').write_text(json.dumps(dict(
        analysis='statistical_baseline_ladder_20261106', stage=stage, ladder_complete=fit,
        hypotheses_per_metric=HYPOTHESES, declared_hypotheses_per_metric=396)))
    for model in MODELS:
        for seed in sl.SEEDS:
            cell = directory/f'cells/{model}-seed{seed}-fitted.npz'
            cell.parent.mkdir(exist_ok=True)
            cell.write_bytes(b'fitted')
    files = [directory/name for name in strength.SOURCES] + sorted(
        (directory/'cells').glob('*.npz'))
    recorded = {str(path.relative_to(directory)): sl.digest(path) for path in files}
    (directory/'completion.json').write_text(json.dumps(dict(
        status=status, ladder_complete=fit, stage=stage, models=MODELS, seeds=list(sl.SEEDS),
        control_sets=list(sl.CONTROL_SETS), completed_cells=len(MODELS)*len(sl.SEEDS),
        hypotheses_per_metric=HYPOTHESES, declared_hypotheses_per_metric=396,
        output_sha256=recorded)))
    return dict(directory=directory, levels=levels, designs=designs)


STRONG = dict(PE=.04, PG=.03, PEG=.05, BPLEG=.02)
WEAK = dict(PE=.0, PG=.0, PEG=.0, BPLEG=.0)
WORSE = dict(PE=-.05, PG=-.04, PEG=-.06, BPLEG=-.03)


def test_effect_signs_cover_both_metrics_and_qualification_is_native_only(tmp_path):
    fixture = write_stage(tmp_path, spearman=STRONG, rank_mse=STRONG)
    summary = strength.summarise(fixture['directory'])
    assert summary['schema'] == strength.SCHEMA
    assert summary['support']['families'] == FAMILIES
    assert summary['support']['designs'] == list(DESIGNS)
    assert summary['bootstrap'] == dict(draws=10000, seed=20261106, unit='biological family',
                                        paired=True, confidence=.95)
    assert set(summary['source']['files']) >= set(strength.SOURCES)
    for metric, sign in (('spearman', 1.), ('rank_mse', -1.)):
        designs = fixture['levels'][metric]
        result = summary['metrics'][metric]
        assert result['families'] == FAMILIES
        assert result['hypothesis_family']['per_metric'] == 4
        assert result['tests']['standardized_maxT']['role'] == 'sensitivity'
        assert result['tests']['native']['family_size'] == 4
        assert result['qualified'] == [f'{enhanced} - {baseline}'
                                       for enhanced, baseline in strength.PAIRS]
        for record in result['pairs']:
            enhanced = designs[record['enhanced_design']]
            baseline = designs[record['baseline_design']]
            assert record['point'] == pytest.approx(sign*float(np.mean(enhanced - baseline)))
            assert record['native']['lower'] <= record['point'] <= record['native']['upper']
            assert record['pointwise'][0] <= record['pointwise'][1]
            sensitivity = record['standardized_maxT']
            assert sensitivity['lower'] <= sensitivity['upper']
            assert record['native']['verdict'] == 'positive'
        assert result['verdict_counts'] == {'positive': 4, 'negative': 0, 'unresolved': 0}


def test_a_degraded_control_is_reported_negative_and_never_qualified(tmp_path):
    summary = strength.summarise(write_stage(tmp_path, spearman=WORSE, rank_mse=WORSE)['directory'])
    for metric in ('spearman', 'rank_mse'):
        result = summary['metrics'][metric]
        assert result['qualified'] == []
        assert result['verdict_counts'] == {'positive': 0, 'negative': 4, 'unresolved': 0}
        assert all(record['native']['upper'] < 0 for record in result['pairs'])


def test_model_repeats_of_a_control_are_one_copy_not_more_samples(tmp_path):
    fixture = write_stage(tmp_path, spearman=STRONG, rank_mse=STRONG)
    absolute = np.load(fixture['directory']/'aggregate-spearman.npz')['absolute']
    labels = [(model, name) for model in MODELS for name in fixture['designs']]
    copies = strength.shared_controls(absolute, labels, dict(
        models=MODELS, reused=list(REUSED), shared=list(SHARED)))
    assert sorted(copies) == sorted(['BPL', 'P', 'PE', 'PEG', 'PG', 'BPLEG'])
    assert all(values.shape == (FAMILIES,) for values in copies.values())
    summary = strength.summarise(fixture['directory'])
    native = summary['metrics']['spearman']['tests']['native']
    assert native['groups'] == FAMILIES and native['family_size'] == 4
    assert summary['shared_columns']['designs'] == ['PE', 'PG', 'PEG', 'BPLEG', 'P', 'BPL']
    assert list(copies) == summary['shared_columns']['designs']


def test_a_repeated_control_column_that_disagrees_is_refused(tmp_path):
    fixture = write_stage(tmp_path, spearman=STRONG, rank_mse=STRONG,
                          split=('arm07', 'PEG', 4, 1e-9))
    with pytest.raises(ValueError, match='disagrees; repeats are verified'):
        strength.summarise(fixture['directory'])
    absolute = np.load(fixture['directory']/'aggregate-spearman.npz')['absolute']
    labels = [(model, name) for model in MODELS for name in fixture['designs']]
    with pytest.raises(ValueError, match='PEG is declared model-independent'):
        strength.shared_controls(absolute, labels, dict(
            models=MODELS, reused=list(REUSED), shared=list(SHARED)))


def test_a_partial_pilot_is_refused_before_any_column_is_read(tmp_path):
    fixture = write_stage(tmp_path, spearman=STRONG, rank_mse=STRONG,
                          status='partial_profile_only', stage='profile_only')
    with pytest.raises(ValueError, match='is not a complete full stage'):
        strength.summarise(fixture['directory'])


def test_incomplete_designs_support_or_cells_are_refused(tmp_path):
    fixture = write_stage(tmp_path, spearman=STRONG, rank_mse=STRONG, drop_design='BPLEG')
    with pytest.raises(ValueError, match='model/design columns'):
        strength.summarise(fixture['directory'])
    fixture = write_stage(tmp_path/'short', spearman=STRONG, rank_mse=STRONG, families=162)
    with pytest.raises(ValueError, match='not the frozen'):
        strength.summarise(fixture['directory'])
    fixture = write_stage(tmp_path/'cells', spearman=STRONG, rank_mse=STRONG)
    (fixture['directory']/f'cells/arm05-seed{sl.SEEDS[1]}-fitted.npz').unlink()
    with pytest.raises(ValueError, match='is missing'):
        strength.summarise(fixture['directory'])
    fixture = write_stage(tmp_path/'undefined', spearman=STRONG, rank_mse=STRONG,
                          split=('arm03', 'P', 7, np.nan))
    with pytest.raises(ValueError, match='undefined family cells'):
        strength.summarise(fixture['directory'])


def test_an_effect_that_crosses_zero_is_unresolved_not_qualified(tmp_path):
    summary = strength.summarise(write_stage(tmp_path, spearman=WEAK, rank_mse=WEAK)['directory'])
    result = summary['metrics']['spearman']
    assert result['qualified'] == []
    assert result['verdict_counts']['positive'] == 0
    assert result['verdict_counts']['unresolved'] == 4


def test_the_summary_never_overwrites_an_existing_file(tmp_path, monkeypatch):
    fixture = write_stage(tmp_path, spearman=STRONG, rank_mse=STRONG)
    out = tmp_path/'strength.json'
    out.write_text('{}')
    monkeypatch.setattr('sys.argv', ['summarize_baseline_strength.py', '--source',
                                     str(fixture['directory']), '--out', str(out)])
    with pytest.raises(ValueError, match='never overwritten'):
        strength.main()
    assert out.read_text() == '{}'
