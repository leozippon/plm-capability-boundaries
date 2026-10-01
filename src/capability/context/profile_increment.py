"""Family-aware descriptive profile increments and held-family prediction.

Partial rank correlation conditions on a scalar profile score, not all family
information. Neither it nor held-family prediction identifies a mechanism.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from scipy.stats import rankdata

from .profiles import cluster_bootstrap
from ..core.statistics import (
    MINIMUM_BOOTSTRAP_UNITS,
    MINIMUM_FINITE_DRAW_FRACTION,
    bootstrap_unit_floor,
)


def mutation_support(assays: list[dict]) -> dict:
    """Describe the actual mutation panel independently of defined endpoints."""
    return dict(n_assays=len(assays), n_families=len({r['cluster'] for r in assays}),
                n_variants=sum(len(r['mutants']) for r in assays),
                assay_ids=sorted(r['assay'] for r in assays))


def common_vector_support(sources: list[dict]) -> tuple[list[list[dict]], dict]:
    """Intersect assays while refusing inconsistent frozen mutation draws.

    Scores and selected contexts remain model-specific. No vectors are reordered
    or trimmed, and profile/effect labels must identify exactly the same rows.
    """
    if len(sources) < 2:
        raise ValueError('common support requires at least two vector sources')
    arms = [s['arm'] for s in sources]
    if len(set(arms)) != len(arms):
        raise ValueError('common support requires distinct model arms')
    panels = []
    for source in sources:
        panel = {r['assay']: r for r in source['assays']}
        if len(panel) != len(source['assays']):
            raise ValueError('duplicate assays')
        panels.append(panel)
    shared = sorted(set.intersection(*(set(p) for p in panels)))
    for assay in shared:
        reference = panels[0][assay]
        for panel in panels[1:]:
            row = panel[assay]
            if row['mutants'] != reference['mutants']:
                raise ValueError(f'{assay}: exact mutation alignment mismatch')
            for key in ('wildtype_id', 'cluster', 'mutant_digest', 'original_mutant_digest',
                        'profile_scores', 'measured'):
                if row[key] != reference[key]:
                    raise ValueError(f'{assay}: common-support {key} mismatch')
    subsets = [[panel[a] for a in shared] for panel in panels]
    support = mutation_support(subsets[0])
    support.update(arms=arms, definition='intersection of assays with identical ordered mutations, WT, family, draw digests, profile scores and measured effects; model-specific contexts retained',
                   per_arm={s['arm']: dict(native=mutation_support(s['assays']),
                              excluded_assay_ids=sorted(set(panel)-set(shared)))
                            for s, panel in zip(sources, panels)})
    return subsets, support


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 3 or not np.isfinite([x, y]).all():
        raise ValueError("correlation requires aligned finite vectors with at least three rows")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def standardized_rank(x: np.ndarray) -> np.ndarray:
    rank = rankdata(x).astype(float)
    sd = rank.std()
    return (rank - rank.mean()) / sd if sd > 0 else rank * 0


def partial_rank(model: np.ndarray, profile: np.ndarray, measured: np.ndarray) -> dict[str, Any]:
    """Correlate rank residuals, with both outcome and model residualized."""
    m, p, y = [standardized_rank(np.asarray(x, float)) for x in (model, profile, measured)]
    design = np.column_stack([np.ones(len(p)), p])
    mr = m - design @ np.linalg.lstsq(design, m, rcond=None)[0]
    yr = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return {"model_profile_spearman": correlation(m, p),
            "model_measured_spearman": correlation(m, y),
            "profile_measured_spearman": correlation(p, y),
            "partial_rank_correlation": correlation(mr, yr)}


def summarize(rows: list[dict], key: str, *, bootstrap: int, seed: int) -> dict:
    included = [r for r in rows if r[key] is not None]
    if not included:
        return {"point": None, "interval": None, "n_assays": 0, "excluded_assays": len(rows)}
    record = cluster_bootstrap([r[key] for r in included], [r["cluster"] for r in included],
                               resamples=bootstrap, seed=seed)
    record.update(n_assays=len(included), excluded_assays=len(rows)-len(included))
    return record


def profile_increment(assays: list[dict], *, condition: str, bootstrap: int = 2000,
                      seed: int = 20260923) -> dict:
    """Leave one family out, with equal-family/equal-assay training weights.

    Inputs and targets become within-assay standardized ranks. Predictors thus
    operate on an assay panel (transductive score normalization, without labels).
    The target ranks of held-out families enter evaluation only. Coefficients
    use fixed unregularized OLS: no tuning occurs on test labels.
    """
    if len({r['cluster'] for r in assays}) < 3:
        raise ValueError("held-family prediction requires at least three families")
    if len({r['assay'] for r in assays}) != len(assays):
        raise ValueError("duplicate assays")
    rows, arrays = [], []
    for assay in assays:
        m, p, y = [np.asarray(x, float) for x in
                   (assay['scores'][condition], assay['profile_scores'], assay['measured'])]
        if not (len(m) == len(p) == len(y) == len(assay['mutants'])):
            raise ValueError("unaligned mutation vectors")
        if len(set(assay['mutants'])) != len(m):
            raise ValueError("duplicate mutation identifiers")
        if not np.isfinite([m, p, y]).all():
            raise ValueError("nonfinite mutation vectors")
        row = dict(assay=assay['assay'], cluster=assay['cluster'], n_variants=len(m),
                   **partial_rank(m, p, y))
        rows.append(row)
        arrays.append((np.column_stack([np.ones(len(m)), standardized_rank(p), standardized_rank(m)]),
                       standardized_rank(y)))
    family_counts = Counter(r['cluster'] for r in rows)
    # Sufficient statistics avoid refitting over all mutation rows per fold.
    stats = {}
    for row, (x, y) in zip(rows, arrays):
        family = row['cluster']
        weight = 1 / (family_counts[family] * len(y))
        xx, xy = stats.setdefault(family, (np.zeros((3, 3)), np.zeros(3)))
        xx += weight * x.T @ x
        xy += weight * x.T @ y
    total_xx = sum(v[0] for v in stats.values())
    total_xy = sum(v[1] for v in stats.values())
    folds = {}
    for family, (xx, xy) in stats.items():
        train_xx, train_xy = total_xx-xx, total_xy-xy
        coefficients = np.linalg.lstsq(train_xx, train_xy, rcond=None)[0]
        baseline = np.linalg.lstsq(train_xx[:2, :2], train_xy[:2], rcond=None)[0]
        folds[family] = (baseline, coefficients)
    for row, (x, y) in zip(rows, arrays):
        baseline, coefficients = folds[row['cluster']]
        p_only, p_m = x[:, :2] @ baseline, x @ coefficients
        rp = correlation(rankdata(p_only), y)
        rpm = correlation(rankdata(p_m), y)
        row.update(cv_profile_spearman=rp, cv_profile_model_spearman=rpm,
                   cv_delta_spearman=None if rp is None or rpm is None else rpm-rp,
                   cv_delta_rank_mse=float(np.mean((y-p_only)**2)-np.mean((y-p_m)**2)))
    metrics = ['model_profile_spearman', 'model_measured_spearman', 'profile_measured_spearman',
               'partial_rank_correlation', 'cv_profile_spearman', 'cv_profile_model_spearman',
               'cv_delta_spearman', 'cv_delta_rank_mse']
    return dict(condition=condition, n_assays=len(rows), n_families=len(stats),
                summaries={k:summarize(rows,k,bootstrap=bootstrap,seed=seed+i) for i,k in enumerate(metrics)},
                assays=rows, folds=[dict(held_family=k, profile_coefficients=v[0].tolist(),
                        profile_model_coefficients=v[1].tolist()) for k,v in sorted(folds.items())],
                protocol='leave-one-50%-identity-family-out; within-assay standardized ranks; equal family weights; equal assay weights within family; fixed OLS',
                uncertainty='95% family-bootstrap intervals conditional on fitted cross-validation predictions; training-set uncertainty is not re-fitted',
                limitation='Incremental prediction conditional on the scalar profile; not a test of all evolutionary information or a causal mechanism.')


def family_regression(rows: list[dict], covariates: list[str], *, bootstrap: int,
                      seed: int) -> dict:
    """Family-mean OLS with a 95% family-bootstrap interval on every coefficient.

    Two publishability conditions, both refusals rather than flags, because
    ``analyse_profile_increment.py`` writes every coefficient interval straight
    into its artefact and a degenerate marker beside one would be optional to
    notice.

    The first is the shared unit floor: the resampling unit is the 50%-identity
    family, so :data:`~.statistics.MINIMUM_BOOTSTRAP_UNITS` governs this
    interval as it governs every other percentile interval in the package. The
    function published one at any family count until 2026-09-24. Replayed on
    the generic non-collinear covariates of
    ``tests.test_profile_increment.family_rows`` at row seed 7, with 2,000
    family-bootstrap draws at seed 20260923, the pre-fix depth coefficient reads
    +0.194 [-0.251, +5.560] at seven families over the 286 draws it could fit,
    5.81 wide, against 18.00 wide at eight families and 7.27 at twelve. The
    published width therefore does not order with the support it rests on, and
    the least identified of the three designs reads as the most precise -- the
    non-monotone pinching the floor's own derivation is written about, with the
    illegal design reading as the precise one.

    The second is specific to a regression: a draw whose resampled design is
    rank deficient cannot be fitted and was silently discarded, so the
    percentile was taken over however many draws happened to be identifiable.
    At eight families and this six-parameter design that is 819 of 2,000 draws,
    and an interval over them is conditioned on identifiability rather than
    being the requested bootstrap distribution --
    :data:`~.statistics.MINIMUM_FINITE_DRAW_FRACTION` is the fraction below
    which the two must not be published under one name. Eight families is
    therefore necessary and not sufficient here; twelve retain 1,966 of 2,000
    and sixteen retain all of them.
    """

    families = sorted({r['cluster'] for r in rows})
    floor = bootstrap_unit_floor(len(families))
    if floor['degenerate']:
        raise ValueError(f'family regression refused: {floor["degenerate_reason"]}')
    # Family means define both predictor and response units consistently.
    x = np.array([[np.mean([r[k] for r in rows if r['cluster']==f]) for k in covariates]
                  for f in families])
    y = np.array([np.mean([r['model_minus_lookup'] for r in rows if r['cluster']==f]) for f in families])
    means, scales = x.mean(0), x.std(0)
    if np.any(scales == 0):
        raise ValueError('constant regression covariate')
    z = (x-means)/scales
    design = np.column_stack([np.ones(len(x)), z, z[:,0]*z[:,2]])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError('rank deficient family regression')
    coef = np.linalg.lstsq(design,y,rcond=None)[0]
    rng = np.random.default_rng(seed)
    draws=[]
    for _ in range(bootstrap):
        pick=rng.integers(0,len(y),len(y))
        if np.linalg.matrix_rank(design[pick]) == design.shape[1]:
            draws.append(np.linalg.lstsq(design[pick],y[pick],rcond=None)[0])
    minimum_draws=int(np.ceil(MINIMUM_FINITE_DRAW_FRACTION*bootstrap))
    if len(draws) < minimum_draws:
        raise ValueError(
            f'family regression refused: {len(draws)} of {bootstrap} family-bootstrap '
            f'draws had an identifiable design, below the {MINIMUM_FINITE_DRAW_FRACTION:.0%} '
            f'floor; a percentile over the identifiable draws is conditioned on '
            f'identifiability and is not the requested bootstrap distribution. '
            f'{len(families)} families do not resolve a {design.shape[1]}-parameter '
            'design under resampling with replacement')
    low,high=np.percentile(draws,[2.5,97.5],axis=0)
    names=['intercept',*covariates,'log10_neff_x_profile_entropy']
    return dict(n_families=len(families),minimum_families=MINIMUM_BOOTSTRAP_UNITS,
                covariate_means=means.tolist(),covariate_scales=scales.tolist(),
                design_condition_number=float(np.linalg.cond(design)),
                covariate_correlations=np.corrcoef(x.T).tolist(),
                coefficients={k:dict(point=float(c),interval=[float(lo),float(hi)]) for k,c,lo,hi in zip(names,coef,low,high)},
                bootstrap_success=len(draws), bootstrap_requested=bootstrap,
                protocol='family-mean OLS; covariates standardized using the fixed observed cohort; 95% family-bootstrap intervals refused below the shared unit floor and below the identifiable-draw fraction; exploratory unadjusted intervals')


def context_rescue(assays: list[dict], *, bootstrap: int = 2000,
                   seed: int = 20260923) -> dict:
    """Context contrasts on identical variants, with family-equal aggregation."""
    rows=[]
    for assay in assays:
        y=rankdata(assay['measured'])
        correlations={k:correlation(rankdata(v),y) for k,v in assay['scores'].items()}
        profile=correlation(rankdata(assay['profile_scores']),y)
        h,u,n=correlations['homolog'],correlations['unrelated'],correlations['no_context']
        tokens=assay['wt_scored_tokens']
        if tokens<=0:
            raise ValueError('wild-type likelihood contrast needs a positive scored token count')
        rows.append(dict(assay=assay['assay'],cluster=assay['cluster'],
            homolog_identity=assay['homolog_identity'],max_lcs=assay['max_lcs'],
            delta_spearman=None if h is None or u is None else h-u,
            homolog_minus_no_context_spearman=None if h is None or n is None else h-n,
            unrelated_minus_no_context_spearman=None if u is None or n is None else u-n,
            wt_delta_loglikelihood_per_token=(assay['wt_log_likelihood']['homolog']-assay['wt_log_likelihood']['unrelated'])/tokens,
            **{f'{k}_minus_lookup':None if v is None or profile is None else v-profile for k,v in correlations.items()}))
    strata={'pooled':rows,'identity_below_50_percent':[r for r in rows if r['homolog_identity']<50],
            'lcs_below_10_residues':[r for r in rows if r['max_lcs']<10],
            'identity_below_50_and_lcs_below_10':[r for r in rows if r['homolog_identity']<50 and r['max_lcs']<10]}
    # Append secondary contrasts to preserve existing metrics' bootstrap seeds.
    metrics=['delta_spearman','wt_delta_loglikelihood_per_token',*[f'{k}_minus_lookup' for k in ('no_context','unrelated','homolog')],
             'homolog_minus_no_context_spearman','unrelated_minus_no_context_spearman']
    return dict(strata={name:dict(n_assays=len(subset),n_families=len({r['cluster'] for r in subset}),
                    metrics={k:summarize(subset,k,bootstrap=bootstrap,seed=seed+i) for i,k in enumerate(metrics)})
                        for name,subset in strata.items()},assays=rows,
                primary_contrast='delta_spearman: homolog minus unrelated',
                secondary_contrasts=['homolog_minus_no_context_spearman','unrelated_minus_no_context_spearman'],
                units='Spearman dimensionless; WT likelihood contrast nats/scored token; identity query-normalized percent; LCS residues',
                limitation='Context-condition effect for one fixed homolog and matched unrelated sequence per target; not evidence that homolog information was absent from model parameters; token units do not permit cross-model likelihood magnitudes.')
