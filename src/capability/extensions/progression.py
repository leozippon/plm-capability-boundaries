"""Frozen sequential predictive blocks; not exclusive biological information."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any

import numpy as np
from scipy.stats import rankdata

from . import overlap
from ..context.crossed_controls import PROFILE_FEATURE_ORDER
from ..context.local_context import SCALE_ORDER, candidate_feature_names, fold_membership
from ..context.profile_increment import correlation
from ..readouts.readout_analysis import row_weights

SEEDS = (20260923, 20260924, 20260925)
WIDTHS = {'B': 444, 'M': 1, 'P': 14, 'L': 111, 'S': 7}
DESIGNS = {
    'full': {'M': ('M',), 'MP': ('M', 'P'), 'MPL': ('M', 'P', 'L'),
             'BM': ('B', 'M'), 'BMP': ('B', 'M', 'P'), 'BMPL': ('B', 'M', 'P', 'L'),
             'BPL': ('B', 'P', 'L'), 'BML': ('B', 'M', 'L')},
    'structural': {'M': ('M',), 'MP': ('M', 'P'), 'MPL': ('M', 'P', 'L'),
                   'MPLS': ('M', 'P', 'L', 'S'), 'BM': ('B', 'M'),
                   'BMP': ('B', 'M', 'P'), 'BMPL': ('B', 'M', 'P', 'L'),
                   'BMPLS': ('B', 'M', 'P', 'L', 'S'), 'BPLS': ('B', 'P', 'L', 'S'),
                   'BMLS': ('B', 'M', 'L', 'S'), 'BMPS': ('B', 'M', 'P', 'S')},
}
# Each entry is (augmented, reduced, multiplicity class, scientific roles).
# Identical ladder/drop pairs appear once, not as duplicate hypotheses.
CONTRASTS = {
    'full': {
        'P|M': ('MP', 'M', 'primary', ('sequential',)),
        'L|MP': ('MPL', 'MP', 'primary', ('sequential',)),
        'P|BM': ('BMP', 'BM', 'supplementary', ('adjusted_ladder',)),
        'L|BMP': ('BMPL', 'BMP', 'supplementary', ('adjusted_ladder', 'drop_L')),
        'M|BPL': ('BMPL', 'BPL', 'supplementary', ('drop_M',)),
        'P|BML': ('BMPL', 'BML', 'supplementary', ('drop_P',)),
        'B|MPL': ('BMPL', 'MPL', 'supplementary', ('final_B_increment',)),
    },
    'structural': {
        'P|M': ('MP', 'M', 'primary', ('sequential',)),
        'L|MP': ('MPL', 'MP', 'primary', ('sequential',)),
        'S|MPL': ('MPLS', 'MPL', 'primary', ('sequential',)),
        'P|BM': ('BMP', 'BM', 'supplementary', ('adjusted_ladder',)),
        'L|BMP': ('BMPL', 'BMP', 'supplementary', ('adjusted_ladder',)),
        'S|BMPL': ('BMPLS', 'BMPL', 'supplementary', ('adjusted_ladder', 'drop_S')),
        'M|BPLS': ('BMPLS', 'BPLS', 'supplementary', ('drop_M',)),
        'P|BMLS': ('BMPLS', 'BMLS', 'supplementary', ('drop_P',)),
        'L|BMPS': ('BMPLS', 'BMPS', 'supplementary', ('drop_L',)),
        'B|MPLS': ('BMPLS', 'MPLS', 'supplementary', ('final_B_increment',)),
    },
}
LIMITATIONS = [
    'Predictive accessibility under historical blocks, training-only scaling and tuned ridge; basis/penalty-dependent, not exclusive biological knowledge identification.',
    'Historical definitions retained without pruning or orthogonalization; overlapping and linearly dependent columns change the ridge penalty geometry.',
    'B402 equals sum(B0:400); WT and MT compositions each sum to one.',
    'On strict singles P1=P4, P2=P3, P5=P6=P7; P0 is rank of summed log odds, P9 is their mean (not their rank).',
    'On singles L30:36 substitution deltas lie in the linear span of B substitution counts; each local class-fraction vector sums to one; each BLOSUM difference is the exact mutant-minus-WT definition (floating rounding possible).',
    'S combines RSA with six RSA-times-chemistry interactions using the same scales as L, not seven pure structural facts; BLOSUM carries evolutionary statistics.',
    'Structural support is exact-mapped strict-single Tsuboyama stability assays, not representative of the full mixed-mutation panel.',
    'Intervals condition on fitted OOF predictions, without training/tuning refits; seeds are averaged, never independent bootstrap units.',
    'Retained scalar raw_M inputs cannot independently reconstruct absent model archives; no model/token inference is performed.',
]


def registry(panel):
    if panel not in DESIGNS:
        raise ValueError('unknown panel')
    return {'designs': DESIGNS[panel], 'contrasts': {
        c: dict(augmented=a, reduced=b, family=f, roles=roles,
                spearman='augmented - reduced', rank_mse='reduced - augmented')
        for c, (a, b, f, roles) in CONTRASTS[panel].items()}}


def column_inventory():
    return {'B': {'width': 444, 'substitution_counts': [0, 400],
                  'relative_position_mean_std': [400, 402], 'mutation_count': 402,
                  'WT_composition': [403, 423], 'MT_composition': [423, 443],
                  'length_over_1024': 443},
            'P': list(PROFILE_FEATURE_ORDER), 'L': list(candidate_feature_names()['wall']),
            'L_parts': {'wcomp': [0, 30], 'wchem': [30, 96], 'wsub': [96, 111]},
            'S': ['RSA'] + ['RSA_x_delta_' + s for s in SCALE_ORDER],
            'M': ['retained_raw_M_within_panel_assay_rank'],
            'name_translation': 'old S -> public B sequence444; old overlap B -> public B+P+L fullcontrols569; public S is RSA7'}


def array_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def identity_digest(samples):
    return hashlib.sha256(json.dumps(samples, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def census(x):
    """Label-blind exact column census, linear in table size; no giant SVD."""
    x = np.asarray(x, float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('invalid census matrix')
    buckets = {}; duplicates = []
    for j in range(x.shape[1]):
        column = x[:, j].copy(); column[column == 0] = 0  # canonical signed zero
        key = array_digest(column)
        candidates = buckets.setdefault(key, [])
        match = next((k for k in candidates if np.array_equal(x[:, k], column)), None)
        if match is None:
            candidates.append(j)
        else:
            duplicates.append([match, j])
    return dict(rows=len(x), columns=x.shape[1],
                zero_columns=np.flatnonzero(np.all(x == 0, axis=0)).tolist(),
                constant_columns=np.flatnonzero(np.all(x == x[0], axis=0)).tolist(),
                exact_duplicate_columns=duplicates,
                scope='within block, exact equality only; not a rank or near-collinearity estimate; no pruning')


def validate_samples(samples, measured, aids, families):
    n = len(samples)
    if not n or any(len(a) != n for a in (measured, aids, families)):
        raise ValueError('empty or unaligned panel')
    keys = [(r['assay'], r['mutation']) for r in samples]
    if len(set(keys)) != n or len({r['original_index'] for r in samples}) != n:
        raise ValueError('duplicate sample identity')
    if any(r['assay'] != a or r['cluster'] != f or r['original_index'] < 0
           or r['variant_index'] < 0 for r, a, f in zip(samples, aids, families)):
        raise ValueError('sample identity mismatch or negative index')
    if not np.isfinite(measured).all() or np.asarray(measured).ndim != 1:
        raise ValueError('invalid measured vector')
    if min(Counter(aids).values()) < 3:
        raise ValueError('each assay requires three rows')
    row_weights(np.asarray(aids), np.asarray(families))


def prepare_blocks(blocks, aids, panel):
    """Subset first, then rerank M and P0 without changing any other column."""
    required = {b for columns in DESIGNS[panel].values() for b in columns}
    result = {}
    for b in required:
        x = np.asarray(blocks[b], float).copy()
        if x.shape != (len(aids), WIDTHS[b]) or not np.isfinite(x).all():
            raise ValueError(f'invalid {b} block')
        if b in ('M', 'P'):
            x[:, 0] = overlap.rerank(x[:, 0], np.asarray(aids))
        result[b] = x
    return result


def fit_cell(blocks, measured, aids, families, signature, panel, samples):
    """One arm/seed, all frozen designs, same ordered support and partitions."""
    aids = np.asarray(aids); families = np.asarray(families)
    validate_samples(samples, measured, aids, families)
    blocks = prepare_blocks(blocks, aids, panel)
    projected = overlap.project_folds(signature, families)
    membership = [(f, tuple(h), tuple(t), tuple(tuple(v) for v in inn))
                  for f, h, t, inn in projected]
    predictions = {}; records = {}; checks = {}
    for name, columns in DESIGNS[panel].items():
        x = np.column_stack([blocks[b] for b in columns])
        predictions[name], records[name] = overlap.projected_predict(x, measured, aids, families, signature)
        if fold_membership(records[name]) != membership:
            raise ValueError('realized design fold disagreement')
        checks[name] = dict(design=columns, design_sha256=array_digest(x),
                            target_sha256=array_digest(overlap.rerank(measured, aids)),
                            support_sha256=identity_digest(samples),
                            membership_sha256=identity_digest(projected), rows=len(aids),
                            prediction_sha256=array_digest(predictions[name]))
    return predictions, records, checks


def contrast_values(scores, metric, panel):
    if metric not in ('spearman', 'rank_mse'):
        raise ValueError('unknown metric')
    sign = 1 if metric == 'spearman' else -1
    return {c: None if scores[a] is None or scores[b] is None else sign * (scores[a] - scores[b])
            for c, (a, b, _, _) in CONTRASTS[panel].items()}


def assay_metrics(predictions, measured, aids, families, arm, seed, panel):
    aids = np.asarray(aids); families = np.asarray(families)
    if set(predictions) != set(DESIGNS[panel]):
        raise ValueError('incomplete design predictions')
    if any(np.asarray(p).shape != (len(aids),) or not np.isfinite(p).all()
           for p in predictions.values()):
        raise ValueError('invalid predictions')
    row_weights(aids, families)
    target = overlap.rerank(measured, aids); rows = []
    for assay in sorted(set(aids)):
        ix = aids == assay
        row: dict[str, Any] = dict(assay=str(assay), cluster=str(families[ix][0]), arm=arm, seed=seed)
        for metric in ('spearman', 'rank_mse'):
            scores = {name: correlation(rankdata(pred[ix]), target[ix]) if metric == 'spearman'
                      else float(np.mean((pred[ix] - target[ix]) ** 2))
                      for name, pred in predictions.items()}
            row[metric] = scores
            row[metric + '_contrasts'] = contrast_values(scores, metric, panel)
        rows.append(row)
    return rows


def joint_bootstrap(rows, panel, *, bootstrap=2000, seed=20261006,
                    expected_seeds=SEEDS, expected_arms=None) -> dict[str, Any]:
    """Shared family draws; seeds -> assay -> family -> equal families.

    Three distinct within-panel multiplicity families. Undefined correlations
    render the affected family nonestimable; neither assays nor seeds are dropped.
    """
    if bootstrap < 100:
        raise ValueError('at least 100 bootstrap draws required')
    arms = sorted(expected_arms if expected_arms is not None else {r['arm'] for r in rows})
    if (not rows or not arms or not expected_seeds or len(set(arms)) != len(arms)
            or len(set(expected_seeds)) != len(expected_seeds)):
        raise ValueError('empty or duplicate expected inference cells')
    indexed = {}; assay_family = {}
    for r in rows:
        key = (r['arm'], r['seed'], r['assay'])
        if key in indexed:
            raise ValueError('duplicate arm/seed/assay')
        indexed[key] = r
        if assay_family.setdefault(r['assay'], r['cluster']) != r['cluster']:
            raise ValueError('assay spans families')
    assays = sorted(assay_family)
    expected = {(a, s, assay) for a in arms for s in expected_seeds for assay in assays}
    if set(indexed) != expected:
        raise ValueError('incomplete or extra arm/seed/assay support')
    families = sorted(set(assay_family.values()))
    if len(families) < 2:
        raise ValueError('at least two bootstrap families required')
    group_assays = [[a for a in assays if assay_family[a] == f] for f in families]
    # Counts encode the same actual family resamples for every multiplicity family.
    rng = np.random.default_rng(seed)
    counts = np.asarray([np.bincount(rng.integers(len(families), size=len(families)),
                                   minlength=len(families)) for _ in range(bootstrap)], float)
    groups = {}
    for label, metric, kind in (('primary_spearman', 'spearman', 'primary'),
                                ('supplementary_spearman', 'spearman', 'supplementary'),
                                ('secondary_rank_mse', 'rank_mse', None)):
        keys = [(a, c) for a in arms for c, entry in CONTRASTS[panel].items()
                if kind is None or entry[2] == kind]
        undefined = []; matrix = []
        for local_assays in group_assays:
            cells = []
            for arm, c in keys:
                values = []
                for assay in local_assays:
                    rs = [indexed[arm, s, assay] for s in expected_seeds]
                    vs = [r[metric + '_contrasts'][c] for r in rs]
                    if any(v is None for v in vs):
                        undefined.append(dict(arm=arm, assay=assay, contrast=c))
                        values.append(np.nan)
                    else:
                        if not np.isfinite(vs).all():
                            raise ValueError('nonfinite inference contrast')
                        values.append(float(np.mean(vs)))
                cells.append(float(np.mean(values)))
            matrix.append(cells)
        if undefined:
            groups[label] = dict(status='not_estimable', hypothesis_count=len(keys),
                                reason='undefined assay correlations; no outcome-driven exclusions',
                                undefined=undefined)
            continue
        matrix = np.asarray(matrix); point = matrix.mean(0)
        draws = counts @ matrix / len(families)
        lo, hi = np.quantile(draws, [.025, .975], axis=0)
        critical = float(np.quantile(np.max(np.abs(draws - point), axis=1), .95))
        groups[label] = dict(status='complete', hypothesis_count=len(keys),
                            joint_critical_max_absolute_deviation=critical,
                            contrasts=[dict(arm=a, metric=metric, contrast=c, point=float(point[j]),
                                            pointwise=[float(lo[j]), float(hi[j])],
                                            simultaneous=[float(point[j]-critical), float(point[j]+critical)])
                                       for j, (a, c) in enumerate(keys)])
    return dict(panel=panel, status='complete' if all(g['status']=='complete' for g in groups.values())
                else 'not_estimable', families=families, arms=arms, seeds=list(expected_seeds),
                bootstrap=bootstrap, bootstrap_seed=seed,
                shared_family_draws_sha256=array_digest(counts),
                aggregation='mean seeds within assay, equal assays within family, equal families',
                simultaneous_method='95th percentile joint max absolute centered bootstrap deviation, separately in each declared within-panel family; no cross-panel pooling',
                uncertainty='conditional on fitted OOF predictions; no refitting', groups=groups)
