"""Frozen R1 statistical-baseline ladder: control sets, strict joins, inference.

This module holds the definitions and pure procedures for the predeclared
statistical-baseline ladder in ``docs/statistical-baseline-ladder.md``: the six
control sets, the paired direction definitions, the strict Potts/GEMME sample
join, the equal-family aggregation and the declared uncertainty families. It
fits nothing itself; the runner supplies retained blocks, one archived or newly
fitted held-out prediction vector per design, and the same original partitions
for every design.

Only the superposed adversary is new. ``M`` is the retained scalar likelihood
readout, ``P`` the 14 historical profile features, ``C0`` the historical
``B+P+L`` control, ``E`` the Potts energy difference and ``G`` the official
GEMME score. Both new scalars enter exactly like ``M``: one column, reranked
within each retained assay on the frozen support, with the raw values kept only
for the descriptive zero-shot layer.

Uncertainty is not defined here twice. The primary family is the shared
``bidirectional`` native maximum-absolute-centered-deviation bootstrap, whose
draw mechanism is reused for pointwise quantiles; the standardized maxT family
is the project's existing fixed-standard-error helper, retained as an explicitly
labelled sensitivity.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy.stats import rankdata

from . import overlap
from .progression import array_digest, identity_digest
from ..context.local_context import fold_membership
from ..context.profile_increment import correlation
from ..readouts.readout_analysis import row_weights

#: The original three split seeds; never regenerated.
SEEDS = (20260923, 20260924, 20260925)
#: Declared joint family: max centered absolute deviation, 10,000 shared draws.
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261106
METRICS = ('spearman', 'rank_mse')

#: Canonical column order. Every design concatenates its blocks in this order,
#: so the reused designs reproduce the archived design matrices exactly.
BLOCK_ORDER = ('B', 'M', 'P', 'L', 'E', 'G')
WIDTHS = {'B': 444, 'M': 1, 'P': 14, 'L': 111, 'E': 1, 'G': 1}
#: Blocks whose first column is reranked within each assay on the frozen support.
RERANKED = ('M', 'P', 'E', 'G')

#: Design names are their blocks concatenated in :data:`BLOCK_ORDER`, so a name
#: is also the historical design name whenever the design already existed.
DESIGNS = {
    'M': ('M',),
    'MP': ('M', 'P'),
    'MPE': ('M', 'P', 'E'),
    'MPG': ('M', 'P', 'G'),
    'MPEG': ('M', 'P', 'E', 'G'),
    'P': ('P',),
    'PE': ('P', 'E'),
    'PG': ('P', 'G'),
    'PEG': ('P', 'E', 'G'),
    'BPL': ('B', 'P', 'L'),
    'BMPL': ('B', 'M', 'P', 'L'),
    'BPLEG': ('B', 'P', 'L', 'E', 'G'),
    'BMPLEG': ('B', 'M', 'P', 'L', 'E', 'G'),
}
#: The six declared control sets, each with its model-only and joint arm.
CONTROL_SETS = {
    'P': {'control': 'P', 'model': 'M', 'joint': 'MP'},
    'P+E': {'control': 'PE', 'model': 'M', 'joint': 'MPE'},
    'P+G': {'control': 'PG', 'model': 'M', 'joint': 'MPG'},
    'P+E+G': {'control': 'PEG', 'model': 'M', 'joint': 'MPEG'},
    'C0': {'control': 'BPL', 'model': 'M', 'joint': 'BMPL'},
    'C0+E+G': {'control': 'BPLEG', 'model': 'M', 'joint': 'BMPLEG'},
}
#: Direction name -> the role of the reduced arm inside a control set. The joint
#: arm is compared with the traditional control and with the model-only readout;
#: both are paired reductions of the same joint arm.
DIRECTIONS = {'model_given_controls': 'control', 'controls_given_model': 'model'}
#: Stages: the pilot exercises P/M/P+M and the original C0 control only.
STAGE_CONTROL_SETS = {'profile_only': ('P', 'C0'), 'full': tuple(CONTROL_SETS)}
#: The pilot refits these archives from scratch and requires exact reproduction;
#: they are the only reused designs that depend on the model.
REPRODUCTION_DESIGNS = ('M', 'MP')

#: Column names the ladder reads from the two upstream baseline tables.
POTTS_COLUMN = 'potts_delta'
GEMME_COLUMN = 'gemme_score'
PROFILE_COLUMN = 'profile_delta'
#: Top-level keys under which an upstream document may list its per-sample records.
RECORD_CONTAINERS = ('samples', 'records', 'variants', 'rows')

PILOT_SCHEMA = 'statistical_ladder_reuse_pilot_v1'
_HEX64 = re.compile(r'^[0-9a-f]{64}$')

#: The completion contract of each upstream baseline document this ladder reads,
#: as the producing exporters declare it: ``potts`` by
#: ``scripts/capability/baselines/r1_msa_baselines.py``, ``gemme`` by the three-run
#: official aggregate. A finite score from an unconverged background or a partly
#: scored wild type is not a baseline result, so the counts, the per-row status and
#: the declared run multiplicity are all required, not just numeric finiteness.
BASELINE_CONTRACT = {
    'potts': {
        'counts': {'complete': True, 'samples': 25728, 'samples_with_potts': 25728,
                   'backgrounds_expected': 174, 'backgrounds_resolved': 174,
                   'backgrounds_fitted': 174, 'backgrounds_not_converged': 0,
                   'backgrounds_failed': 0, 'backgrounds_pending': 0},
        'row_status': 'fitted',
    },
    'gemme': {
        'counts': {'complete': True, 'samples': 25728, 'samples_expected': 25728,
                   'samples_scored': 25728, 'wildtypes_expected': 174,
                   'wildtypes_complete': 174, 'wildtypes_incomplete': 0},
        'row_status': 'scored',
        'empty_lists': ('incomplete',),
        'repeats': 3,
    },
}

LIMITATIONS = [
    'Potts enters as the full-sequence energy difference of the retained plmc fit and GEMME as '
    'the mean of the declared three official runs per wild type, both reranked within assay; '
    'the official multi-mutant GEMME value is the same run\'s per-site sum, so neither scalar is '
    'a non-additive epistasis score.',
    'GEMME run-to-run spread is tool stochasticity (JET2 samples cluster partitions without a '
    'seed), not biological replication, so the three runs are averaged and never resampled.',
    'Increments are predictive contrasts under the historical blocks, training-only scaling and tuned ridge; they are not exclusive biological knowledge, mutual information or a causal decomposition.',
    'C0+E+G retains all of C0, which already contains P; set-specific increments overlap and are never summed into additive shares.',
    'Potts and GEMME enter as single assay-reranked scalars; per-position and per-mutation structure stays outside this ladder.',
    'The same aligned homolog support underlies every method, so these contrasts do not estimate any method at an ideal MSA.',
    'Unresolved simultaneous intervals are not equivalence; no post-hoc equivalence margin is declared.',
    'Intervals condition on fitted held-out predictions; they exclude refitting and training-history uncertainty.',
    'Reused M and C0 fits are admitted by archive digest, sample identity, fold membership and metric recomputation; the reproduction pilot covers exactly the models it names.',
    'A profile-only run is partial by construction and cannot be reported as the complete six-set ladder.',
]


def registry() -> dict[str, Any]:
    """The frozen design registry, echoed into every run contract."""
    for name, columns in DESIGNS.items():
        if ''.join(columns) != name or tuple(sorted(columns, key=BLOCK_ORDER.index)) != columns:
            raise ValueError(f'design {name!r} is not its blocks in canonical order')
    for name, entry in CONTROL_SETS.items():
        for role in ('control', 'model', 'joint'):
            if entry[role] not in DESIGNS:
                raise ValueError(f'control set {name} names unknown design {entry[role]}')
        if set(DESIGNS[entry['joint']]) - set(DESIGNS[entry['control']]) - set(DESIGNS[entry['model']]):
            raise ValueError(f'control set {name} joint arm carries undeclared blocks')
    return dict(blocks={name: WIDTHS[name] for name in BLOCK_ORDER},
                reranked_blocks=list(RERANKED),
                designs={name: list(columns) for name, columns in DESIGNS.items()},
                control_sets={name: dict(CONTROL_SETS[name]) for name in CONTROL_SETS},
                control_set_order=list(CONTROL_SETS),
                directions=dict(DIRECTIONS),
                stages={name: list(sets) for name, sets in STAGE_CONTROL_SETS.items()},
                contrasts='joint minus reduced arm for Spearman; reduced minus joint for rank MSE, so positive is always improvement')


def stage_designs(stage: str) -> tuple[str, ...]:
    """Every design a stage needs, in registry order."""
    names = {'M'}
    for control_set in STAGE_CONTROL_SETS[stage]:
        entry = CONTROL_SETS[control_set]
        names.update((entry['control'], entry['model'], entry['joint']))
    return tuple(name for name in DESIGNS if name in names)


def reused_designs(stage: str) -> tuple[str, ...]:
    """Designs taken verbatim from the frozen source archives when available."""
    return tuple(name for name in stage_designs(stage) if name in ('M', 'MP', 'BPL', 'BMPL'))


def prepare_block(name: str, values, aids) -> np.ndarray:
    """Validate one block and rerank its first column once, within assay."""
    if name not in WIDTHS:
        raise ValueError(f'unknown block {name!r}')
    aids = np.asarray(aids)
    x = np.asarray(values, float)
    if x.shape != (len(aids), WIDTHS[name]) or not np.isfinite(x).all():
        raise ValueError(f'invalid {name} block')
    x = x.copy()
    if name in RERANKED:
        x[:, 0] = overlap.rerank(x[:, 0], aids)
    return x


def design_matrix(prepared: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    """Concatenate one design's blocks in the canonical order."""
    if name not in DESIGNS:
        raise ValueError(f'unknown design {name!r}')
    missing = [block for block in DESIGNS[name] if block not in prepared]
    if missing:
        raise ValueError(f'design {name} needs blocks {missing}')
    rows = {len(prepared[block]) for block in DESIGNS[name]}
    if len(rows) != 1:
        raise ValueError(f'design {name} mixes block row counts')
    return np.column_stack([prepared[block] for block in DESIGNS[name]])


def projected_membership(signature, families) -> list[tuple]:
    """The original seeded outer/inner membership projected onto these families."""
    return [(fold, tuple(held), tuple(training), tuple(tuple(v) for v in inner))
            for fold, held, training, inner in overlap.project_folds(signature, families)]


def realized_membership(records) -> list[tuple]:
    """Realized outer/inner membership of one fitted design."""
    return fold_membership(records)


def fit_design(prepared, measured, aids, families, signature, name):
    """Fit one design with the original nested ridge on the original partitions."""
    x = design_matrix(prepared, name)
    prediction, records = overlap.projected_predict(x, measured, aids, families, signature)
    expected = projected_membership(signature, families)
    if realized_membership(records) != expected:
        raise ValueError(f'{name}: realized folds differ from the projected originals')
    return prediction, records, array_digest(x)


def assay_rows(predictions, measured, aids, families, *, model, seed, designs) -> list[dict]:
    """Per-assay held-out Spearman and rank MSE for every design of one cell."""
    aids = np.asarray(aids); families = np.asarray(families)
    measured = np.asarray(measured, float)
    row_weights(aids, families)
    target = overlap.rerank(measured, aids)
    for name in designs:
        if name not in predictions:
            raise ValueError(f'missing {name} prediction')
        prediction = np.asarray(predictions[name], float)
        if prediction.shape != target.shape or not np.isfinite(prediction).all():
            raise ValueError(f'invalid {name} prediction')
    rows = []
    for assay in sorted({str(a) for a in aids}):
        index = aids == assay
        spearman, rank_mse = {}, {}
        for name in designs:
            prediction = np.asarray(predictions[name], float)
            spearman[name] = correlation(rankdata(prediction[index]), target[index])
            rank_mse[name] = float(np.mean((prediction[index] - target[index]) ** 2))
        rows.append(dict(assay=assay, cluster=str(families[index][0]), model=model, seed=seed,
                         spearman=spearman, rank_mse=rank_mse))
    return rows


def _indexed_rows(rows, models, seeds):
    """Check the complete model/seed/assay support and group assays in families."""
    index, assay_family = {}, {}
    for row in rows:
        key = (row['model'], row['seed'], row['assay'])
        if key in index:
            raise ValueError('duplicate model/seed/assay metric row')
        index[key] = row
        if assay_family.setdefault(row['assay'], row['cluster']) != row['cluster']:
            raise ValueError('assay spans biological families')
    if not index:
        raise ValueError('empty metric support')
    expected = {(model, seed, assay) for model in models for seed in seeds for assay in assay_family}
    if set(index) != expected:
        raise ValueError(f'metric support is {len(index)} cells, not the declared {len(expected)}')
    families = sorted({str(family) for family in assay_family.values()})
    groups = {family: [assay for assay in sorted(assay_family) if assay_family[assay] == family]
              for family in families}
    return index, families, groups


def _assemble(index, groups, families, labels, values, *, seeds):
    """Seeds within assay, then equal assays within family, then equal families."""
    matrix = np.full((len(families), len(labels)), np.nan)
    undefined = []
    for column, label in enumerate(labels):
        for row, family in enumerate(families):
            per_assay = []
            for assay in groups[family]:
                per_seed = [values(index[label[0], seed, assay], label) for seed in seeds]
                if any(value is None for value in per_seed):
                    undefined.append(dict(assay=assay, label=label)); per_assay = None; break
                if not np.isfinite(per_seed).all():
                    raise ValueError('nonfinite inference value')
                per_assay.append(float(np.mean(per_seed)))
            if per_assay is None:
                break
            matrix[row, column] = float(np.mean(per_assay))
    return dict(families=families, labels=labels, matrix=matrix, undefined=undefined)


def ladder_matrix(rows, metric, *, control_sets=CONTROL_SETS, models, seeds=SEEDS) -> dict[str, Any]:
    """Paired-direction family matrix, one column per model x set x direction."""
    if metric not in METRICS:
        raise ValueError('unknown metric')
    sign = 1 if metric == 'spearman' else -1
    index, families, groups = _indexed_rows(rows, models, seeds)
    labels = [(model, name, direction) for model in models
              for name in control_sets for direction in DIRECTIONS]

    def value(row, label):
        scores = row[metric]
        joint = control_sets[label[1]]['joint']
        reduced = control_sets[label[1]][DIRECTIONS[label[2]]]
        if scores[joint] is None or scores[reduced] is None:
            return None
        return sign * (scores[joint] - scores[reduced])

    return _assemble(index, groups, families, labels, value, seeds=seeds)


def absolute_matrix(rows, metric, *, designs, models, seeds=SEEDS) -> dict[str, Any]:
    """Absolute three-arm family matrix, one column per model x design."""
    if metric not in METRICS:
        raise ValueError('unknown metric')
    index, families, groups = _indexed_rows(rows, models, seeds)
    labels = [(model, name) for model in models for name in designs]

    def value(row, label):
        return row[metric][label[1]]

    return _assemble(index, groups, families, labels, value, seeds=seeds)


def _shared_helper():
    """The sibling declarations this module reuses instead of restating."""
    from . import bidirectional
    missing = [name for name in ('paired_bands', 'resample_means', 'verdict')
               if not hasattr(bidirectional, name)]
    if missing:
        raise RuntimeError(f'the shared bidirectional inference helper lacks {missing}; '
                           'the ladder will not define a second bootstrap')
    return bidirectional


def pointwise_interval(matrix, *, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Percentile interval of the declared shared family draws."""
    helper = _shared_helper()
    draws_ = helper.resample_means(np.asarray(matrix, float), draws=draws, seed=seed)
    return draws_, np.quantile(draws_, [.025, .975], axis=0)


def infer(matrix, labels, *, metric, role='primary', bootstrap=BOOTSTRAP_DRAWS,
          seed=BOOTSTRAP_SEED) -> dict[str, Any]:
    """Native joint band as primary, standardized maxT as labelled sensitivity."""
    if metric not in METRICS:
        raise ValueError('unknown metric')
    matrix = np.asarray(matrix, float)
    if matrix.ndim != 2 or matrix.shape[1] != len(labels) or len(matrix) < 2:
        raise ValueError('inference needs a groups-by-labelled-columns matrix with at least two groups')
    if not np.isfinite(matrix).all():
        return dict(status='not_estimable', role=role, metric=metric, hypothesis_count=len(labels),
                    reason='undefined per-assay correlations; no outcome-driven exclusions')
    helper = _shared_helper()
    native = helper.paired_bands(matrix, draws=bootstrap, seed=seed)
    _, pointwise = pointwise_interval(matrix, draws=bootstrap, seed=seed)
    standardized = _standardized_bands(matrix, draws=bootstrap, seed=seed)
    verdicts = helper.verdict(np.asarray(native['interval'], float))
    hypotheses = []
    for position, label in enumerate(labels):
        hypotheses.append(dict(model=label[0], arm=label[1], direction=label[2], metric=metric,
                               point=native['point'][position],
                               pointwise=pointwise[:, position].tolist(),
                               simultaneous=native['interval'][position],
                               verdict=verdicts[position]))
    sensitivity = dict(status='complete', role='sensitivity', metric=metric,
                       method=standardized['method'], draws=standardized['draws'],
                       seed=standardized['seed'], hypothesis_count=int(standardized['family_size']),
                       joint_critical_standardized=standardized['critical_value'],
                       note='declared sensitivity only; the native-unit family remains primary')
    return dict(status='complete', role=role, metric=metric, hypothesis_count=len(labels),
                native=dict(method=native['method'], draws=native['draws'], seed=native['seed'],
                            confidence=native['confidence'], family_size=native['family_size'],
                            joint_critical_max_absolute_deviation=native['critical_value']),
                standardized_maxT=sensitivity, hypotheses=hypotheses)


def _standardized_bands(matrix, *, draws, seed):
    """The project's fixed-standard-error simultaneous helper, not a new bootstrap."""
    from . import responses
    return responses.simultaneous_bands(np.asarray(matrix, float), draws=draws, seed=seed)


def verdict_counts(hypotheses) -> dict[str, int]:
    values = [row['verdict'] for row in hypotheses]
    unknown = sorted(set(values) - {'positive', 'negative', 'unresolved'})
    if unknown:
        raise ValueError(f'unknown verdict values {unknown}')
    return {name: values.count(name) for name in ('positive', 'negative', 'unresolved')}


def zero_shot(raw: Mapping[str, np.ndarray], measured, aids, families, *,
              draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED) -> dict[str, Any]:
    """Descriptive assay-internal ranking of raw, unsupervised scores."""
    aids = np.asarray(aids); families = np.asarray(families)
    measured = np.asarray(measured, float)
    if aids.shape != measured.shape or families.shape != measured.shape:
        raise ValueError('the descriptive layer needs row-aligned scores, assay ids and families')
    if not raw:
        raise ValueError('the descriptive layer needs at least one raw score')
    for name, values in raw.items():
        values = np.asarray(values, float)
        if values.shape != measured.shape or not np.isfinite(values).all():
            raise ValueError(f'invalid raw {name} score')
    assays = sorted({str(a) for a in aids})
    target = overlap.rerank(measured, aids)
    assay_family = {}
    for assay, family in zip(aids, families):
        if assay_family.setdefault(str(assay), str(family)) != str(family):
            raise ValueError('assay assigned to multiple families')
    columns = {}
    for name, values in raw.items():
        values = np.asarray(values, float)
        columns[name] = np.asarray([correlation(rankdata(values[aids == assay]), target[aids == assay])
                                    for assay in assays], dtype=float)
    groups = {}
    for position, assay in enumerate(assays):
        groups.setdefault(assay_family[assay], []).append(position)
    result = {}
    for name, values in columns.items():
        undefined = [assays[position] for position in np.flatnonzero(~np.isfinite(values))]
        if undefined:
            result[name] = dict(status='not_estimable', undefined_assays=undefined)
            continue
        matrix = np.asarray([values[groups[family]].mean() for family in sorted(groups)], float)[:, None]
        _, pointwise = pointwise_interval(matrix, draws=draws, seed=seed)
        result[name] = dict(status='complete', point=float(matrix.mean()),
                            pointwise=pointwise[:, 0].tolist(),
                            families=len(matrix), assays=len(assays))
    return dict(status='complete' if all(v['status'] == 'complete' for v in result.values())
                else 'partial', role='descriptive_zero_shot', draws=draws, seed=seed,
                aggregation='equal assays within family, equal families', columns=result)


def read_predictions(path, designs, *, sample_ids, target) -> dict[str, np.ndarray]:
    """Load one saved held-out prediction archive and check its identities.

    A requested design must be present with a finite vector; extra design arrays
    the archive happens to hold (the frozen source run saved all eight) are
    ignored, because the exact bytes are bound by the recorded digest instead.
    """
    path = Path(path)
    target = np.asarray(target, float)
    sample_ids = np.asarray(sample_ids)
    with np.load(path, allow_pickle=False) as saved:
        identity = {'original_index', 'target', 'sample_id'}
        missing = set(designs) - set(saved.files)
        if missing or not identity <= set(saved.files):
            raise ValueError(f'{path.name}: archive keys disagree with the declared designs: '
                             f'missing {sorted(missing)}')
        if not np.array_equal(saved['sample_id'], sample_ids):
            raise ValueError(f'{path.name}: saved sample identity disagreement')
        if not np.array_equal(saved['original_index'], np.arange(len(target))):
            raise ValueError(f'{path.name}: saved row order disagreement')
        if not np.array_equal(saved['target'], target):
            raise ValueError(f'{path.name}: saved target disagreement')
        predictions = {}
        for name in designs:
            values = np.asarray(saved[name], float)
            if values.shape != target.shape or not np.isfinite(values).all():
                raise ValueError(f'{path.name}: invalid {name} prediction')
            predictions[name] = values
    return predictions


def verify_checks(checks, predictions, *, designs, support_sha256, membership, target,
                  fold_records) -> dict[str, Any]:
    """Check declared digests, support identity and realized folds of one cell."""
    declared = checks.get('designs')
    if not isinstance(declared, dict) or set(designs) - set(declared):
        raise ValueError('archive checks do not record every design')
    membership_sha = identity_digest(membership)
    target_sha = array_digest(target)
    expected = [(fold, tuple(held), tuple(training), tuple(tuple(v) for v in inner))
                for fold, held, training, inner in membership]
    for name in designs:
        entry = declared[name]
        if list(entry.get('design', [])) != list(DESIGNS[name]):
            raise ValueError(f'{name}: archived design blocks disagree with the registry')
        if entry.get('support_sha256') != support_sha256:
            raise ValueError(f'{name}: archived support digest disagreement')
        if entry.get('membership_sha256') != membership_sha:
            raise ValueError(f'{name}: archived fold membership digest disagreement')
        if entry.get('target_sha256') != target_sha:
            raise ValueError(f'{name}: archived target digest disagreement')
        if int(entry.get('rows', -1)) != len(target):
            raise ValueError(f'{name}: archived row count disagreement')
        if array_digest(predictions[name]) != entry.get('prediction_sha256'):
            raise ValueError(f'{name}: archived prediction digest disagreement')
        if realized_membership(fold_records[name]) != expected:
            raise ValueError(f'{name}: saved folds disagree with the projected originals')
    return dict(designs=list(designs), rows=len(target), support_sha256=support_sha256,
                membership_sha256=membership_sha, target_sha256=target_sha,
                prediction_sha256={name: array_digest(predictions[name]) for name in designs})


def digest(path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_score_table(path, *, metadata_required=False) -> dict[str, Any]:
    """Read one upstream baseline table into a sample-keyed table.

    The file is either an ``.npz`` holding ``sample_id`` plus one array per
    column, or a ``.json`` document. A JSON document may be an array of records,
    an object listing them under ``samples``/``records``/``variants``/``rows``,
    or an object keyed by ``sample_id``. Whatever else the document declares is
    kept as provenance and bound by digest; the sample rows themselves never are.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'score table {path} is missing')
    if path.suffix == '.npz':
        identifiers, columns, annotations = _npz_columns(path)
        metadata = {}
    elif path.suffix == '.json':
        try:
            document = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise ValueError(f'score table {path} is not valid JSON') from error
        records, metadata = _document_records(document)
        if not records:
            raise ValueError(f'{path} holds no sample records')
        identifiers, columns, annotations = _record_columns(records, path)
    else:
        raise ValueError('score table must be .npz or .json')
    if metadata_required and not metadata:
        raise ValueError(f'{path} declares no upstream provenance metadata; an unbound score '
                         'dump cannot stand in for an official baseline table')
    return dict(path=str(path), sample_id=identifiers, columns=columns, annotations=annotations,
                metadata=metadata, metadata_sha256=_metadata_digests(metadata), sha256=digest(path))


def _split_columns(named):
    """Numeric fields are scores; textual ones stay as annotations, never scores."""
    columns, annotations = {}, {}
    for name, values in named.items():
        values = np.asarray(values)
        if values.dtype.kind in 'iufb':
            columns[name] = values
        else:
            annotations[name] = values
    return columns, annotations


def _npz_columns(path):
    with np.load(path, allow_pickle=False) as data:
        if 'sample_id' not in data.files:
            raise ValueError(f'{path} has no sample_id array')
        identifiers = [str(value) for value in data['sample_id']]
        named = {name: np.asarray(data[name]) for name in data.files if name != 'sample_id'}
    columns, annotations = _split_columns(named)
    return identifiers, columns, annotations


def _document_records(document):
    if isinstance(document, list):
        return document, {}
    if not isinstance(document, dict) or not document:
        raise ValueError('a JSON score table is an array of records or a JSON object')
    for key in RECORD_CONTAINERS:
        if isinstance(document.get(key), list):
            return document[key], {name: value for name, value in document.items() if name != key}
    values = list(document.values())
    if all(isinstance(value, dict) for value in values):
        return [dict(record, sample_id=record.get('sample_id', key))
                for key, record in document.items()], {}
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return [dict(sample_id=key, score=value) for key, value in document.items()], {}
    raise ValueError('unrecognised JSON score table; expected records carrying sample_id or a '
                     f'mapping keyed by sample_id, found keys {sorted(document)[:8]}')


def _record_columns(records, path):
    if any(not isinstance(record, dict) or not record.get('sample_id') for record in records):
        raise ValueError(f'{path}: every score record needs a non-empty sample_id')
    identifiers = [str(record['sample_id']) for record in records]
    names = sorted({name for record in records for name in record} - {'sample_id'})
    named = {}
    for name in names:
        if any(name not in record for record in records):
            raise ValueError(f'{path}: only some rows carry {name}')
        try:
            named[name] = np.asarray([record[name] for record in records], dtype=np.float64)
        except (TypeError, ValueError):
            named[name] = np.asarray([record[name] for record in records])
    columns, annotations = _split_columns(named)
    return identifiers, columns, annotations


def _metadata_digests(metadata) -> dict[str, str]:
    return {name: hashlib.sha256(json.dumps(value, sort_keys=True, default=repr)
                                 .encode()).hexdigest()
            for name, value in metadata.items()}


def join_scores(sample_ids, table, *, columns) -> dict[str, np.ndarray]:
    """Join one table's score columns onto the frozen order by unique ``sample_id``.

    Duplicate keys, missing or extra rows, unaligned lengths and non-finite
    values fail; nothing is imputed, truncated or reordered silently.
    """
    expected = [str(value) for value in sample_ids]
    if len(set(expected)) != len(expected):
        raise ValueError('frozen support carries duplicate sample_id values')
    identifiers = [str(value) for value in table['sample_id']]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{table['path']} carries duplicate sample_id values")
    missing = sorted(set(expected) - set(identifiers))
    extra = sorted(set(identifiers) - set(expected))
    if missing or extra:
        raise ValueError(f"{table['path']} does not cover the frozen support: "
                         f'{len(missing)} missing, {len(extra)} extra sample_id values')
    position = {identifier: index for index, identifier in enumerate(identifiers)}
    order = np.asarray([position[identifier] for identifier in expected])
    joined = {}
    for name in columns:
        if name not in table['columns']:
            raise ValueError(f"{table['path']} lacks required column {name}; numeric columns "
                             f"are {sorted(table['columns'])}")
        values = np.asarray(table['columns'][name])
        if values.shape != (len(identifiers),):
            raise ValueError(f'{name} is not aligned with the score rows')
        try:
            values = values.astype(np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f'{name} holds non-numeric values') from error
        undefined = ~np.isfinite(values)
        if undefined.any():
            status = table.get('annotations', {}).get('status')
            detail = ''
            if status is not None and len(status) == len(values):
                counts = {}
                for state in np.asarray(status)[undefined].tolist():
                    counts[str(state)] = counts.get(str(state), 0) + 1
                detail = f' with upstream statuses {counts}'
            raise ValueError(f'{name} holds {int(undefined.sum())} non-finite scores{detail}; '
                             'zeros are never substituted')
        joined[name] = values[order].copy()
    return joined


def require_complete_baseline(name, table) -> dict[str, Any]:
    """Refuse an upstream baseline table that is not a complete official run.

    ``table`` is what :func:`read_score_table` returns. The declared counts, the
    per-row status and, for GEMME, the three-run aggregate are checked against
    :data:`BASELINE_CONTRACT`; numeric finiteness alone is never evidence of
    completion, because a fitted-but-unconverged background or a partly scored
    wild type still produces finite values. The returned summary is echoed into
    the run contract as the evidence that admission was verified.
    """
    contract = BASELINE_CONTRACT.get(name)
    if contract is None:
        raise ValueError(f'unknown baseline {name!r}')
    metadata = table.get('metadata') or {}
    counts = metadata.get('counts')
    if not isinstance(counts, dict):
        raise ValueError(f'{name}: the upstream document declares no counts, so its completion '
                         'cannot be verified')
    if counts.get('complete') is not True:
        raise ValueError(f'{name}: upstream counts.complete is {counts.get("complete")!r}; an '
                         'incomplete baseline is never admitted as a control')
    rows = len(table['sample_id'])
    required = contract['counts']
    wrong = {key: dict(observed=counts.get(key), required=value)
             for key, value in required.items() if key != 'complete' and counts.get(key) != value}
    if rows != required['samples']:
        wrong['rows'] = dict(observed=rows, required=required['samples'])
    if wrong:
        raise ValueError(f'{name}: upstream counts disagree with the frozen baseline contract: {wrong}')
    for key in contract.get('empty_lists', ()):
        listed = metadata.get(key)
        if not isinstance(listed, list) or listed:
            raise ValueError(f'{name}: the upstream document lists {key}={listed!r}; a complete '
                             'baseline declares it empty')
    repeats = contract.get('repeats')
    if repeats is not None and metadata.get('repeats') != repeats:
        raise ValueError(f'{name}: upstream repeats is {metadata.get("repeats")!r}, not {repeats}')
    status = _row_field(table, 'status')
    if status is None:
        raise ValueError(f'{name}: rows carry no status, so completion cannot be verified')
    states = Counter(str(value) for value in np.asarray(status).tolist())
    unexpected = {state: count for state, count in states.items()
                  if state != contract['row_status']}
    if unexpected:
        raise ValueError(f'{name}: {sum(unexpected.values())} rows are not '
                         f'{contract["row_status"]}: {unexpected}')
    if repeats is not None:
        runs = _row_field(table, 'n_runs')
        if runs is None:
            raise ValueError(f'{name}: rows carry no n_runs, so the {repeats}-run aggregate cannot '
                             'be verified')
        values = np.asarray(runs)
        if values.shape != (rows,) or not (values == repeats).all():
            raise ValueError(f'{name}: rows do not each aggregate exactly {repeats} official runs')
    return dict(name=name, rows=rows, row_status=contract['row_status'],
                repeats=repeats, counts={key: counts[key] for key in required})


def _row_field(table, name):
    for source in (table.get('annotations'), table.get('columns')):
        if isinstance(source, dict) and name in source:
            return source[name]
    return None


def validate_pilot(pilot, *, source_contract_sha256, source_completion_sha256, models,
                   threads, designs=REPRODUCTION_DESIGNS) -> dict[str, Any]:
    """Refuse to reuse archived fits without a bit-exact reproduction pilot."""
    if not isinstance(pilot, dict):
        raise ValueError('a reuse pilot is a record object')
    if pilot.get('schema') != PILOT_SCHEMA:
        raise ValueError(f"reuse pilot schema is not {PILOT_SCHEMA}")
    if pilot.get('status') != 'passed':
        raise ValueError(f"reuse pilot status is {pilot.get('status')!r}, not passed")
    if pilot.get('source_contract_sha256') != source_contract_sha256:
        raise ValueError('reuse pilot was not run against the current source contract')
    if pilot.get('source_completion_sha256') != source_completion_sha256:
        raise ValueError('reuse pilot was not run against the current source completion receipt')
    if pilot.get('threads') != threads:
        raise ValueError('reuse pilot used a different BLAS thread count; reuse would not be exact')
    for name in ('script_sha256', 'module_sha256'):
        if not isinstance(pilot.get(name), str) or not _HEX64.match(pilot[name]):
            raise ValueError(f'reuse pilot {name} is not a recorded digest')
    if list(pilot.get('designs', [])) != list(designs):
        raise ValueError('reuse pilot did not reproduce the designs being reused')
    reproduction = pilot.get('reproduction')
    if not isinstance(reproduction, dict) or not reproduction:
        raise ValueError('reuse pilot carries no per-model reproduction records')
    uncovered = sorted(set(models) - set(reproduction))
    if uncovered:
        raise ValueError(f'reuse pilot does not reproduce models {uncovered}')
    for model in models:
        cells = reproduction[model]
        if sorted(cells) != sorted(str(seed) for seed in SEEDS):
            raise ValueError(f'reuse pilot model {model} does not cover all three seeds')
        for seed in SEEDS:
            cell = cells[str(seed)]
            for name in designs:
                entry = cell.get(name)
                if not isinstance(entry, dict):
                    raise ValueError(f'reuse pilot lacks {model}/{seed}/{name}')
                if entry.get('prediction_identical') is not True:
                    raise ValueError(f'reuse pilot did not reproduce {model}/{seed}/{name} exactly')
                if entry.get('alpha_reproduced') is not True:
                    raise ValueError('reuse pilot did not reproduce the archived alphas')
                if entry.get('fold_membership_matched') is not True:
                    raise ValueError('reuse pilot did not reproduce the archived folds')
                if float(entry.get('max_absolute_difference', np.inf)) != 0.:
                    raise ValueError(f'reuse pilot {model}/{seed}/{name} differs above zero')
    return dict(models=len(models), designs=list(designs), seeds=len(SEEDS), threads=threads)
