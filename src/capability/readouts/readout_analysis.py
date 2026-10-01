"""Nested family-held-out readout comparison; labels never enter held-fold fits."""
from __future__ import annotations

from collections import Counter
import re

import numpy as np
import torch
from scipy.stats import rankdata

from ..core.amino_acids import AA20
from ..context.profile_increment import correlation, standardized_rank, summarize

ALPHAS = (.01, .1, 1., 10., 100.)


def sequence_features(wildtype: str, mutants: list[str]) -> np.ndarray:
    """400 substitution counts, relative position mean/std, count, compositions, L/1024."""
    if not wildtype or set(wildtype) - set(AA20):
        raise ValueError('noncanonical or empty wildtype')
    length = len(wildtype)
    wtcomp = np.array([wildtype.count(a) / length for a in AA20])
    rows = []
    for mutant in mutants:
        counts = np.zeros((20, 20)); positions = []; sequence = list(wildtype)
        for mutation in mutant.split(':'):
            match = re.fullmatch(r'([A-Z])(\d+)([A-Z])', mutation)
            if match is None:
                raise ValueError(f'invalid substitution {mutation}')
            before, pos, after = match.groups(); pos = int(pos) - 1
            if before not in AA20 or after not in AA20 or not 0 <= pos < length or wildtype[pos] != before or pos in positions:
                raise ValueError(f'inconsistent substitution {mutation}')
            positions.append(pos); sequence[pos] = after
            counts[AA20.index(before), AA20.index(after)] += 1
        relative = (np.asarray(positions) + 1) / length
        comp = np.array([sequence.count(a) / length for a in AA20])
        rows.append(np.r_[counts.ravel(), relative.mean(), relative.std(), len(positions), wtcomp, comp, length / 1024])
    return np.asarray(rows, dtype=np.float64)


def family_folds(families, n_splits: int, seed: int) -> list[list]:
    unique = [v.item() if isinstance(v, np.generic) else v for v in sorted(set(families))]
    if len(unique) < n_splits:
        raise ValueError(f'{n_splits} folds require at least {n_splits} families')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    return [[unique[int(i)] for i in chunk] for chunk in (order[i::n_splits] for i in range(n_splits))]


def row_weights(assays: np.ndarray, families: np.ndarray) -> np.ndarray:
    """Sum-one weights: each family equal, each assay equal within family."""
    assay_families = {}
    for assay, family in zip(assays, families):
        if assay in assay_families and assay_families[assay] != family:
            raise ValueError('assay assigned to multiple families')
        assay_families[assay] = family
    family_counts = Counter(assay_families.values()); assay_counts = Counter(assays)
    return np.asarray([1 / (len(family_counts) * family_counts[f] * assay_counts[a]) for a, f in zip(assays, families)])


def ridge_predict(x_train, y_train, weights, x_test, alphas=ALPHAS, device='cpu'):
    """Weighted mean squared error + alpha ||beta||², unpenalized intercept.

    Weighted centering/scaling is fitted exclusively on training rows. One
    symmetric eigendecomposition gives the entire strictly-positive alpha grid.
    Float64 accumulation prevents cancellation in correlated feature blocks.
    """
    if any(a <= 0 for a in alphas):
        raise ValueError('ridge alpha must be positive')
    x = torch.as_tensor(x_train, device=device, dtype=torch.float64)
    y = torch.as_tensor(y_train, device=device, dtype=torch.float64)
    w = torch.as_tensor(weights, device=device, dtype=torch.float64)
    w = w / w.sum()
    mean = (w[:, None] * x).sum(0)
    xc = x - mean
    scale = (w[:, None] * xc.square()).sum(0).sqrt()
    scale = torch.where(scale < 1e-12, torch.ones_like(scale), scale)
    z = xc / scale
    ym = (w * y).sum()
    covariance = z.T @ (w[:, None] * z)
    rhs = z.T @ (w * (y - ym))
    values, vectors = torch.linalg.eigh(covariance)
    alpha = torch.tensor(alphas, device=device, dtype=torch.float64)
    beta = vectors @ ((vectors.T @ rhs)[:, None] / (values.clamp_min(0)[:, None] + alpha))
    test = torch.as_tensor(x_test, device=device, dtype=torch.float64)
    return (((test - mean) / scale) @ beta + ym).cpu().numpy()


def nested_predict(x, measured, assay_ids, families, *, seed=20260923,
                   outer_splits=5, inner_splits=4, alphas=ALPHAS, device='cpu'):
    """Generate predictions and a full family split/tuning audit trail."""
    x = np.asarray(x, float); measured = np.asarray(measured, float)
    assay_ids = np.asarray(assay_ids); families = np.asarray(families)
    if x.ndim != 2 or len(x) != len(measured) or len(x) != len(assay_ids) or len(x) != len(families):
        raise ValueError('unaligned readout arrays')
    if not np.isfinite(x).all() or not np.isfinite(measured).all():
        raise ValueError('nonfinite readout arrays')
    row_weights(assay_ids, families)  # validates that assays cannot cross folds
    prediction = np.full(len(x), np.nan); records = []
    def targets(indices):
        result = np.empty(len(indices))
        for assay in np.unique(assay_ids[indices]):
            local = assay_ids[indices] == assay
            result[local] = standardized_rank(measured[indices[local]])
        return result
    for outer, held in enumerate(family_folds(families, outer_splits, seed)):
        train = np.flatnonzero(~np.isin(families, held)); test = np.flatnonzero(np.isin(families, held))
        losses = np.zeros(len(alphas)); inner_records = []
        for validation in family_folds(families[train], inner_splits, seed + 100 + outer):
            fit = train[~np.isin(families[train], validation)]
            valid = train[np.isin(families[train], validation)]
            pred = ridge_predict(x[fit], targets(fit), row_weights(assay_ids[fit], families[fit]), x[valid], alphas, device)
            # Weight by number of validation families, so unequal fold sizes
            # still yield an equal-family average across the full inner panel.
            loss = (row_weights(assay_ids[valid], families[valid])[:, None] * (pred - targets(valid)[:, None]) ** 2).sum(0)
            losses += len(validation) * loss
            inner_records.append(dict(validation_families=list(validation), rank_mse=loss.tolist()))
        losses /= len(set(families[train]))
        best = len(losses) - 1 - int(np.argmin(losses[::-1])); alpha = alphas[best]
        prediction[test] = ridge_predict(x[train], targets(train), row_weights(assay_ids[train], families[train]), x[test], [alpha], device)[:, 0]
        records.append(dict(fold=outer, held_families=list(held), training_families=[v.item() if isinstance(v, np.generic) else v for v in sorted(set(families[train]))],
                            alpha=float(alpha), inner_family_rank_mse=losses.tolist(), inner_folds=inner_records))
    if not np.isfinite(prediction).all():
        raise ValueError('incomplete or nonfinite held-out predictions')
    return prediction, records


def evaluate_readouts(assays: list[dict], *, device='cpu', seed=20260923, fold_seed=20260923,
                      bootstrap=2000, permutation_control=True, progress=None) -> dict:
    """Rows require assay, cluster, mutants, measured, P, M, S, R."""
    if len({r['assay'] for r in assays}) != len(assays):
        raise ValueError('duplicate assays')
    for r in assays:
        n = len(r['mutants'])
        if n < 3 or len(set(r['mutants'])) != n:
            raise ValueError('too few or duplicate mutants')
        for key in ('measured', 'P', 'M', 'S', 'R'):
            if len(r[key]) != n or not np.isfinite(r[key]).all():
                raise ValueError(f'unaligned/nonfinite {key}')
    y = np.concatenate([r['measured'] for r in assays])
    aid = np.concatenate([[r['assay']] * len(r['mutants']) for r in assays])
    family = np.concatenate([[r['cluster']] * len(r['mutants']) for r in assays])
    p, m, s, representation = [np.concatenate([r[k] for r in assays]) for k in ('P', 'M', 'S', 'R')]
    p = np.concatenate([standardized_rank(r['P']) for r in assays])
    m = np.concatenate([standardized_rank(r['M']) for r in assays])
    baseline = np.column_stack([p, m, s])
    designs = dict(P=p[:, None], M=m[:, None], P_M=np.column_stack([p, m]), B=baseline, B_R=np.column_stack([baseline, representation]), R=representation,
                   sequence=s, profile_sequence=np.column_stack([p, s]))
    predictions = dict(raw_M=m, raw_P=p); folds = {}
    for name, x in designs.items():
        if progress: progress(name)
        predictions[name], folds[name] = nested_predict(x, y, aid, family, seed=fold_seed, device=device)
    if permutation_control:
        # A single seeded negative-control realization, not a permutation p-value.
        rng = np.random.default_rng(seed + 9000); shuffled = y.copy()
        for assay in np.unique(aid):
            idx = np.flatnonzero(aid == assay); shuffled[idx] = rng.permutation(y[idx])
        for name in ('B', 'B_R'):
            if progress: progress('permuted_' + name)
            predictions['permuted_' + name], folds['permuted_' + name] = nested_predict(designs[name], shuffled, aid, family, seed=fold_seed, device=device)
    rows = []
    for r in assays:
        idx = np.flatnonzero(aid == r['assay']); target = standardized_rank(y[idx])
        row = dict(assay=r['assay'], cluster=r['cluster'], n_variants=len(idx))
        for name, pred in predictions.items():
            row[name + '_spearman'] = correlation(rankdata(pred[idx]), target)
            if name not in ('raw_M', 'raw_P'):
                row[name + '_rank_mse'] = float(np.mean((target - pred[idx]) ** 2))
        a, b = row['B_R_spearman'], row['B_spearman']
        row['delta_spearman'] = None if a is None or b is None else a - b
        row['delta_rank_mse'] = row['B_rank_mse'] - row['B_R_rank_mse']
        for name in ('R', 'B_R', 'B'):
            left, right = row[name + '_spearman'], row['raw_M_spearman']
            row[name + '_minus_raw_M_spearman'] = None if left is None or right is None else left - right
        rows.append(row)
    metrics = [k for k in rows[0] if k not in ('assay', 'cluster', 'n_variants')]
    return dict(n_assays=len(rows), n_families=len(set(family)), n_variants=len(y),
                fold_seed=fold_seed, bootstrap_seed=seed, permutation_seed=seed+9000 if permutation_control else None,
                feature_dimensions={k:v.shape[1] for k,v in designs.items()},
                summaries={k:summarize(rows,k,bootstrap=bootstrap,seed=seed) for k in metrics},
                assays=rows, folds=folds,
                predictions=[dict(assay=r['assay'], mutants=r['mutants'], **{k:v[aid == r['assay']].tolist() for k,v in predictions.items()}) for r in assays],
                protocol='5 outer / 4 inner family-grouped folds; train-only weighted feature scaling; training assay standardized rank targets; weighted rank MSE alpha tuning; equal family and within-family assay weights; unpenalized intercept',
                ridge_objective='weighted mean squared error + alpha * squared coefficient norm; weights sum to one',
                alpha_grid=list(ALPHAS), primary_contrast='B_R minus B Spearman; B minus B_R rank MSE',
                uncertainty='95% family bootstrap conditional on fitted cross-validation predictions; training uncertainty is not refitted',
                permutation_control='one within-assay label permutation used only for fitting, evaluated against original held-out labels; diagnostic, not a permutation p-value' if permutation_control else None,
                limitation='Tests accessibility with fixed projected summaries and linear readouts; a negative result does not establish absence of information. Model training provenance can overlap evaluation families.')
