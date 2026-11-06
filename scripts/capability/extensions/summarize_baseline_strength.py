#!/usr/bin/env python3
"""Are the control sets that carry Potts/GEMME stronger than the controls they extend?

``fit_statistical_ladder.py`` publishes absolute family-level performance and the
396-hypothesis bidirectional family, but neither answers the question declared in
``docs/statistical-baseline-ladder.md`` before any new Potts/GEMME phenotype
comparison was read: P+E-P, P+G-P, P+E+G-P and C0+E+G-C0, one paired four-column
family per metric, Spearman as enhanced minus baseline and rank MSE as baseline
minus enhanced so that a positive effect is an improvement. This post-processor
reads one *complete full-stage* directory; nothing is refitted, no label is read and
no held-out prediction is recomputed.

The compared designs carry no model score: the ladder fits them once and reuses them
for all 33 models, so the saved matrix repeats each column 33 times. The repeats are
verified identical and read once, never averaged into 33 observations. Qualification
uses only the native maximum-absolute-centered-deviation lower bound
(``bidirectional.paired_bands``, 10000 draws, seed 20261106); the historical
standardized maxT helper and the pointwise percentiles of the same draws are reported
beside it as labelled evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.core.io import write_json  # noqa: E402
from src.capability.extensions import bidirectional, responses  # noqa: E402
from src.capability.extensions import statistical_ladder as sl  # noqa: E402

SCHEMA = 'baseline_strength_summary'
ANALYSIS = 'statistical_baseline_strength_20261106'
#: The four declared qualification pairs: the enhanced control set extends the baseline.
PAIRS = (('P+E', 'P'), ('P+G', 'P'), ('P+E+G', 'P'), ('C0+E+G', 'C0'))
#: The control designs those pairs compare, enhanced before baseline, each once.
COMPARED = tuple(dict.fromkeys([sl.CONTROL_SETS[enhanced]['control'] for enhanced, _ in PAIRS]
                               + [sl.CONTROL_SETS[baseline]['control'] for _, baseline in PAIRS]))
#: Sign that turns the enhanced-minus-baseline difference into an improvement, per metric.
SIGNS = {'spearman': 1., 'rank_mse': -1.}
UNITS = {
    'spearman': 'differences of assay-internal Spearman correlations of the held-out prediction '
                'against the within-assay standardized measured rank',
    'rank_mse': 'reductions in mean squared prediction error for within-assay standardized '
                'measured ranks; fitted predictions are not themselves re-ranked for this metric',
}
#: The frozen support every complete stage must declare.
SUPPORT = dict(rows=25728, assays=201, families=163, wildtypes=174)
MODELS = 33
SEEDS = sl.SEEDS
DESIGNS = sl.stage_designs('full')
#: Files whose recorded digests are verified; the cell archives prove the 33x3 grid.
#: ``completion.json`` writes that receipt, so it cannot record its own digest.
SOURCES = ('contract.json', 'contrasts.json',
           'aggregate-spearman.npz', 'aggregate-rank_mse.npz')
DECLARED = dict(
    locus='docs/statistical-baseline-ladder.md',
    statement='Four control-set qualification contrasts were declared before any new '
              'Potts/GEMME phenotype comparison was read: P+E-P, P+G-P, P+E+G-P, C0+E+G-C0, '
              'one paired four-column bootstrap family per metric; only a native lower bound '
              'above zero qualifies the control set as stronger on that metric.')
RELATION = (
    'These four columns per metric are neither independent repetitions of the 396 model contrasts '
    'nor an extension of that family. They compare control sets with and without the new scalars, so '
    'they are model-invariant differences read from the same 163 families, the same three split seeds '
    'and the same folds as the model contrasts, whose arm is fitted on those identical rows. A '
    'qualification here says nothing about any per-model increment, and a failed qualification never '
    'removes a control set from the ladder or licenses selecting favourable model conclusions.')
LIMITATIONS = [
    'Intervals condition on the fitted held-out predictions; they do not resample refits or training history.',
    'Four columns share the 163 families, so multiplicity is controlled within one metric only; no across-metric joint claim follows.',
    'An unresolved native interval is neither equivalence nor absence of information; no equivalence margin is declared after the fact.',
    'Potts and GEMME enter each control set as single assay-reranked scalars on the shared homolog support, so no effect estimates either method at an ideal MSA.',
    'Spearman strength is not rank-MSE strength: the metrics are separate families and never combine into one ranking of control sets.',
]


def read_json(path) -> Any:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'{path} is missing')
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f'{path} is not valid JSON') from error


def admit_stage(source) -> dict[str, Any]:
    """Admit one complete full-stage directory; a declared partial one is refused."""
    source = Path(source)
    completion = read_json(source/'completion.json')
    contract = read_json(source/'contract.json')
    if (completion.get('status'), completion.get('stage'),
            completion.get('ladder_complete')) != ('complete', 'full', True):
        raise ValueError(f"{source} is not a complete full stage (status="
                         f"{completion.get('status')!r}, stage={completion.get('stage')!r}); a "
                         'partial pilot is never summarised as the strength ladder')
    if contract.get('stage') != 'full' or contract.get('ladder_complete') is not True:
        raise ValueError('the source contract does not declare the complete six-set stage')
    scope = contract.get('scope') or {}
    models, seeds = list(scope.get('models') or []), list(scope.get('seeds') or [])
    if len(set(models)) != MODELS:
        raise ValueError(f'the strength ladder needs the original {MODELS} models, not {len(models)}')
    if sorted(set(seeds)) != list(SEEDS):
        raise ValueError(f'the strength ladder needs the original three split seeds, not {seeds}')
    if list(scope.get('control_sets') or []) != list(sl.CONTROL_SETS):
        raise ValueError('the source scope does not declare the six control sets of the full ladder')
    declared = {key: (contract.get('support') or {}).get(key) for key in SUPPORT}
    if declared != SUPPORT or not (contract['support'].get('support_sha256')):
        raise ValueError(f'the declared support is {declared}, not the frozen {SUPPORT} with a '
                         'recorded support digest')
    if list(completion.get('models') or []) != models:
        raise ValueError('the completion receipt does not cover the declared model roster')
    if set(completion.get('seeds') or []) != set(seeds) or \
            completion.get('completed_cells') != MODELS*len(SEEDS):
        raise ValueError('the completion receipt does not cover all 99 model/seed cells')
    if completion.get('hypotheses_per_metric') != MODELS*len(sl.CONTROL_SETS)*len(sl.DIRECTIONS):
        raise ValueError('the completion receipt does not declare the realized 396-hypothesis family')
    designs = scope.get('designs') or {}
    return dict(directory=str(source), models=models, seeds=seeds,
                reused=list(designs.get('reused') or []),
                shared=list(designs.get('shared_across_models') or []),
                support={**SUPPORT, 'support_sha256': contract['support']['support_sha256']},
                recorded=completion.get('output_sha256') or {})


def verify_source_files(stage) -> dict[str, str]:
    """Bind every consumed byte, and every cell archive, to the completion receipt."""
    root = Path(stage['directory'])
    paths = [root/name for name in SOURCES] + [
        root/f'cells/{model}-seed{seed}-fitted.npz'
        for model in stage['models'] for seed in stage['seeds']]
    hashes = {}
    for path in paths:
        name = str(path.relative_to(root))
        if not path.is_file():
            raise ValueError(f'{path} is missing')
        if name not in stage['recorded']:
            raise ValueError(f'the completion receipt does not record {name}')
        if (sha := sl.digest(path)) != stage['recorded'][name]:
            raise ValueError(f'{name} disagrees with its completion receipt digest')
        hashes[name] = sha
    return hashes


def _label(value, name, length):
    try:
        label = tuple(json.loads(str(value)))
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f'{name} carries a malformed column label') from error
    if len(label) != length or not all(isinstance(part, str) for part in label):
        raise ValueError(f'{name} carries a malformed column label')
    return label


def read_aggregate(source, metric, stage) -> dict[str, Any]:
    """Read one saved family-level absolute matrix and its declared columns."""
    path = Path(source)/f'aggregate-{metric}.npz'
    if not path.is_file():
        raise ValueError(f'{path} is missing')
    with np.load(path, allow_pickle=False) as saved:
        missing = {'contrasts', 'absolute', 'families', 'contrast_labels',
                   'design_labels'} - set(saved.files)
        if missing:
            raise ValueError(f'{path.name} lacks {sorted(missing)}')
        absolute = np.asarray(saved['absolute'], dtype=float)
        families = [str(value) for value in saved['families']]
        labels = [_label(value, path.name, 2) for value in saved['design_labels']]
        contrast = np.asarray(saved['contrasts'])
        hypotheses = [_label(value, path.name, 3) for value in saved['contrast_labels']]
    expected = [(model, design) for model in stage['models'] for design in DESIGNS]
    if len(labels) != len(expected) or set(labels) != set(expected):
        raise ValueError(f'{path.name} does not hold the full-stage {len(expected)} model/design '
                         'columns')
    if len(families) != SUPPORT['families'] or len(set(families)) != len(families):
        raise ValueError(f'{path.name} is not the {SUPPORT["families"]}-family absolute matrix')
    if absolute.shape != (len(families), len(labels)):
        raise ValueError(f'{path.name} is not a families-by-columns absolute matrix')
    undefined = int((~np.isfinite(absolute)).sum())
    if undefined:
        raise ValueError(f'{path.name} holds {undefined} undefined family cells; an undefined held-out '
                         'correlation leaves this metric without a strength estimate and is never '
                         'dropped or imputed')
    declared = [(model, name, direction) for model in stage['models']
                for name in sl.CONTROL_SETS for direction in sl.DIRECTIONS]
    if set(hypotheses) != set(declared) or contrast.shape != (len(families), len(declared)):
        raise ValueError(f'{path.name} does not hold the declared 396-hypothesis contrast family')
    return dict(labels=labels, absolute=absolute, families=len(families))


def shared_controls(absolute, labels, stage) -> dict[str, np.ndarray]:
    """One verified copy of every model-independent control the four pairs compare."""
    position = {label: index for index, label in enumerate(labels)}
    copies = {}
    for design in COMPARED:
        if 'M' in sl.DESIGNS[design]:
            raise ValueError(f'{design} carries the model score; strength compares controls only')
        if design not in stage['reused'] and design not in stage['shared']:
            raise ValueError(f'{design} is not declared shared across models in the source scope')
        columns = [absolute[:, position[(model, design)]] for model in stage['models']]
        for model, column in zip(stage['models'][1:], columns[1:]):
            if not np.array_equal(column, columns[0]):
                raise ValueError(f'{design} is declared model-independent, but the column saved for '
                                 f'{model} disagrees; repeats are verified, never averaged or repaired')
        copies[design] = columns[0]
    return copies


def infer_strength(matrix) -> dict[str, Any]:
    """Native family band decides qualification; the other bands are reported beside it."""
    matrix = np.asarray(matrix, dtype=float)
    native = bidirectional.paired_bands(matrix, draws=sl.BOOTSTRAP_DRAWS, seed=sl.BOOTSTRAP_SEED)
    _, pointwise = sl.pointwise_interval(matrix, draws=sl.BOOTSTRAP_DRAWS, seed=sl.BOOTSTRAP_SEED)
    maximal = responses.simultaneous_bands(matrix, draws=sl.BOOTSTRAP_DRAWS,
                                           seed=sl.BOOTSTRAP_SEED)
    interval = np.asarray(native['interval'], dtype=float)
    studentised = np.asarray(maximal['interval'], dtype=float)
    verdicts = bidirectional.verdict(interval)
    columns = [dict(
        point=float(native['point'][position]), pointwise=pointwise[:, position].tolist(),
        native=dict(lower=float(interval[position, 0]), upper=float(interval[position, 1]),
                    verdict=verdicts[position]),
        standardized_maxT=dict(
            lower=float(studentised[position, 0]), upper=float(studentised[position, 1]),
            verdict=(bidirectional.verdict(studentised[position:position+1])[0]
                     if studentised[position, 1] > studentised[position, 0] else None)))
        for position in range(len(verdicts))]
    return dict(
        native=dict(method=native['method'], draws=native['draws'], seed=native['seed'],
                    confidence=native['confidence'], family_size=native['family_size'],
                    groups=native['groups'],
                    joint_critical_max_absolute_deviation=native['critical_value'],
                    unit='native metric units; no standard-error division'),
        standardized_maxT=dict(
            method=maximal['method'], draws=maximal['draws'], seed=maximal['seed'],
            family_size=maximal['family_size'],
            joint_critical_standardized=maximal['critical_value'], role='sensitivity',
            note='the historical standardized maxT helper, reported beside the native band; it '
                 'never decides qualification and never replaces the primary method'),
        columns=columns, verdicts=verdicts)


def summarise(source) -> dict[str, Any]:
    """Build the strength summary of one complete full-stage ladder directory."""
    source = Path(source)
    stage = admit_stage(source)
    hashes = verify_source_files(stage)
    metrics = {}
    for metric in sl.METRICS:
        aggregate = read_aggregate(source, metric, stage)
        copies = shared_controls(aggregate['absolute'], aggregate['labels'], stage)
        effects = np.column_stack([SIGNS[metric]*(copies[sl.CONTROL_SETS[enhanced]['control']]
                                                  - copies[sl.CONTROL_SETS[baseline]['control']])
                                   for enhanced, baseline in PAIRS])
        inference = infer_strength(effects)
        records = [{'pair': f'{enhanced} - {baseline}', 'control_set': enhanced,
                    'baseline_control_set': baseline,
                    'enhanced_design': sl.CONTROL_SETS[enhanced]['control'],
                    'baseline_design': sl.CONTROL_SETS[baseline]['control'],
                    'sign': ('enhanced minus baseline' if SIGNS[metric] > 0
                             else 'baseline minus enhanced'), **column}
                   for (enhanced, baseline), column in zip(PAIRS, inference['columns'])]
        metrics[metric] = dict(
            unit=UNITS[metric], families=aggregate['families'],
            hypothesis_family=dict(
                per_metric=len(PAIRS), columns=[record['pair'] for record in records],
                qualification='native simultaneous lower bound above zero on this metric only; '
                              'negative and unresolved results are reported as they are and never '
                              'remove a control set from the ladder'),
            tests=dict(native=inference['native'], standardized_maxT=inference['standardized_maxT']),
            pairs=records,
            qualified=[record['pair'] for record, verdict in zip(records, inference['verdicts'])
                       if verdict == 'positive'],
            verdict_counts=bidirectional.counts(inference['verdicts']))
    return dict(
        schema=SCHEMA, analysis=ANALYSIS,
        role='post-processed control-set strength; no refit, no label read, no new prediction',
        declaration=DECLARED,
        source=dict(directory=stage['directory'], files=hashes,
                    stage=dict(models=len(stage['models']), seeds=list(stage['seeds']),
                               completed_cells=MODELS*len(SEEDS),
                               hypotheses_per_metric=MODELS*len(sl.CONTROL_SETS)*len(sl.DIRECTIONS))),
        support={**stage['support'], 'designs': list(DESIGNS),
                 'shared_across_models': stage['shared'], 'reused': stage['reused']},
        aggregation='the saved matrix already averages seeds within assay, assays within family '
                    'and then weights the 163 families equally',
        shared_columns=dict(
            designs=list(COMPARED),
            verified='every model repeat of a model-independent design is bit-identical',
            reading='one copy per design over 163 families; the 33 repeats are neither 33 '
                    'observations nor averaged into a larger sample'),
        bootstrap=dict(draws=sl.BOOTSTRAP_DRAWS, seed=sl.BOOTSTRAP_SEED, unit='biological family',
                       paired=True, confidence=bidirectional.CONFIDENCE),
        metrics=metrics, relation_to_the_396_model_family=RELATION, limitations=LIMITATIONS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='one complete full-stage ladder directory')
    parser.add_argument('--out', type=Path, required=True, help='new JSON file to write')
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f'{args.out} already exists; a strength summary is never overwritten')
    document = summarise(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, document)
    print(json.dumps(dict(out=str(args.out), schema=SCHEMA,
                          families=document['support']['families'],
                          qualified={metric: value['qualified'] for metric, value
                                     in document['metrics'].items()}), indent=1))


if __name__ == '__main__':
    main()