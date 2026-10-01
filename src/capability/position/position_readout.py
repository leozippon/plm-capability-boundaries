"""Matched-support conditional prediction from additive likelihood components.

This localizes predictive information in score summands, not its causal neural
mechanism. Each component uses its within-assay standardized rank, matching R1.
The joint design can reweight components and is therefore distinct from their
fixed-coefficient sum. All contrasts share outer/inner held-family assignments.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from ..context.local_context import fold_membership
from ..context.profile_increment import correlation, standardized_rank, summarize
from ..readouts.readout_analysis import nested_predict

CONTRASTS = {
    'own_over_control': ('own', 'control'),
    'downstream_over_control': ('downstream', 'control'),
    'joint_over_control': ('joint', 'control'),
    'full_over_control': ('full', 'control'),
    'unique_own': ('joint', 'downstream'),
    'unique_downstream': ('joint', 'own'),
    'joint_over_full': ('joint', 'full'),
    'own_minus_downstream': ('own', 'downstream'),
}


def evaluate(rows: list[dict], *, device='cuda:0', fold_seed=20260923,
             bootstrap=2000, bootstrap_seed=20260923,
             numerical_sensitivity=False) -> tuple[dict, dict]:
    """Fit the two previously qualified local controls on identical aligned rows."""
    aid = np.concatenate([[r['assay']] * len(r['mutants']) for r in rows])
    families = np.concatenate([[r['cluster']] * len(r['mutants']) for r in rows])
    measured = np.concatenate([r['measured'] for r in rows])
    controls = {key: np.concatenate([np.asarray(r[key], dtype=np.float64) for r in rows])
                for key in ('S', 'P_block', 'wall', 'rf3')}
    parts = {key: np.concatenate([standardized_rank(r[key]) for r in rows])[:, None]
             for key in ('own', 'downstream', 'full')}
    contrasts = dict(CONTRASTS)
    if numerical_sensitivity:
        # Separate removal of the upstream forward discrepancy from replacement
        # of the native scalar by its float64 own-plus-downstream term sum.
        values = {
            'upstream_free': [np.asarray(r['full'])-np.asarray(r['upstream']) for r in rows],
            'term_sum': [np.asarray(r['own'])+np.asarray(r['downstream']) for r in rows],
        }
        for name, arrays in values.items():
            parts[name] = np.concatenate([standardized_rank(a) for a in arrays])[:,None]
            contrasts[name+'_over_control'] = (name,'control')
            contrasts[name+'_minus_full'] = (name,'full')
            contrasts['joint_over_'+name] = ('joint',name)
    if all('historical_full' in row for row in rows):
        parts['historical_full'] = np.concatenate([standardized_rank(r['historical_full']) for r in rows])[:,None]
        contrasts['full_minus_historical'] = ('full','historical_full')
        contrasts['historical_over_control'] = ('historical_full','control')
    records, predictions, dimensions, alphas = {}, {}, {}, {}
    membership = None
    for local in ('wall', 'rf3'):
        base = np.concatenate([controls[k] for k in ('S', 'P_block', local)], axis=1)
        designs = {'C': controls['S'], 'C_P': np.c_[controls['S'],controls['P_block']],
                   'local': controls[local], 'C_local': np.c_[controls['S'],controls[local]],
                   'control': base, **{key: np.c_[base, value] for key, value in parts.items()},
                   'joint': np.c_[base, parts['own'], parts['downstream']]}
        for name, design in designs.items():
            key = f'{local}_{name}'
            predictions[key], audit = nested_predict(design, measured, aid, families,
                                                      seed=fold_seed, device=device)
            realized = fold_membership(audit)
            if membership is None:
                membership, folds = realized, audit
            elif membership != realized:
                raise ValueError('position-term designs do not share family folds')
            dimensions[key] = int(design.shape[1])
            alphas[key] = [fold['alpha'] for fold in audit]
        assays = []
        for row in rows:
            index = np.flatnonzero(aid == row['assay'])
            target = standardized_rank(measured[index])
            entry = dict(assay=row['assay'], cluster=row['cluster'], n_variants=len(index))
            for name in designs:
                pred = predictions[f'{local}_{name}'][index]
                entry[name] = correlation(rankdata(pred), target)
                entry[name + '_rank_mse'] = float(np.mean((target - pred) ** 2))
            for name, (high, low) in dict(contrasts, qualification_over_C=('C_local','C'),
                                         qualification_over_C_P=('control','C_P')).items():
                entry[name] = (None if entry[high] is None or entry[low] is None
                               else entry[high] - entry[low])
                entry[name + '_rank_mse_reduction'] = (entry[low + '_rank_mse']
                                                       - entry[high + '_rank_mse'])
            assays.append(entry)
        metrics = [key for key in assays[0] if key not in ('assay', 'cluster', 'n_variants')]
        records[local] = {'assays': assays, 'summaries': {
            key: summarize(assays, key, bootstrap=bootstrap, seed=bootstrap_seed)
            for key in metrics}}
    return dict(controls=records, fold_seed=fold_seed, bootstrap_seed=bootstrap_seed,
                bootstrap_draws=bootstrap, families=len(set(families)), assays=len(rows),
                variants=len(measured), folds=folds, selected_alphas=alphas,
                feature_dimensions=dimensions, folds_identical=True,
                estimand='family-held predictive rank increment on scored, grid-aligned single substitutions; conditional on the retained support and fitted predictions',
                interpretation='own denotes the token covering the mutation, which can cover multiple residues; downstream denotes scored tokens after that token; neither component is a context ablation',
                multiplicity='95% pointwise family-bootstrap intervals, unadjusted across arms, controls, contrasts and split seeds'), predictions
