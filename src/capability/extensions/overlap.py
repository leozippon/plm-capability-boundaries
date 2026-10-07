"""Strict structural overlap; projected historical folds, no causal interpretation."""
from __future__ import annotations

from collections import Counter
import hashlib
import re
from typing import Any

import numpy as np
from scipy.stats import rankdata

from ..core.amino_acids import AA20
from ..context.local_context import SCALE_ORDER, _scale_matrix
from ..context.profile_increment import correlation, standardized_rank
from ..readouts.readout_analysis import ALPHAS, ridge_predict, row_weights
from .structure import EXPERIMENTAL_METHODS

DESIGNS = ('B', 'B+M', 'B+X', 'B+M+X')
CONTRASTS = ('model_without_X', 'structure_without_M', 'model_with_X', 'attenuation')


class UnsupportedProjection(ValueError):
    """Original partitions cannot support the restricted cohort."""


def _identity(site, primary, legacy):
    if primary in site and legacy in site and site[primary] != site[legacy]:
        raise ValueError(f'conflicting {primary}/{legacy}')
    return site.get(primary, site.get(legacy))


def select_rows(assays, sites):
    """Label-blind strict single substitutions; identity errors are never exclusions."""
    assay_map = {a['assay']: a for a in assays}
    if len(assay_map) != len(assays):
        raise ValueError('duplicate assay')
    indexed = {}
    for site in sites:
        aid, pos = site['assay_id'], site['wt_position']
        if aid not in assay_map:
            raise ValueError('unknown structural assay')
        key = (aid, pos)
        if key in indexed:
            raise ValueError('duplicate structural site')
        assay = assay_map[aid]; wt = assay['wildtype']
        if (not isinstance(pos, int) or isinstance(pos, bool) or not 1 <= pos <= len(wt)
                or site['cluster'] != assay['cluster']
                or _identity(site, 'wt_sequence', 'wildtype') != wt
                or _identity(site, 'wt_sha256', 'wildtype_sha256') != hashlib.sha256(wt.encode()).hexdigest()
                or site['residue'] != wt[pos-1]):
            raise ValueError('structural identity mismatch')
        indexed[key] = site
    rows, excluded = [], Counter()
    offset = 0
    for assay in assays:
        mutants = assay['mutants']; wt = assay['wildtype']
        if len(set(mutants)) != len(mutants):
            raise ValueError('duplicate mutation')
        for j, mutation in enumerate(mutants):
            tokens = mutation.split(':'); positions = []
            for token in tokens:
                match = re.fullmatch(r'([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])', token)
                if match is None:
                    raise ValueError('invalid mutation')
                before, text, after = match.groups(); pos = int(text)
                if not 1 <= pos <= len(wt) or wt[pos-1] != before or before == after or pos in positions:
                    raise ValueError('mutation identity mismatch')
                positions.append(pos)
            if len(tokens) != 1:
                excluded['not_strict_single'] += 1; continue
            site = indexed.get((assay['assay'], positions[0]))
            if site is None:
                excluded['no_site'] += 1; continue
            if (site['status'] != 'admitted' or site.get('mapping_kind') not in ('full_entity', 'full_wt_fragment')
                    or site.get('method', '').upper() not in EXPERIMENTAL_METHODS):
                excluded['not_exact_experimental'] += 1; continue
            if not site['coordinate_present'] or site['rsa'] is None:
                excluded['missing_rsa'] += 1; continue
            if not np.isfinite(site['rsa']) or site['rsa'] < 0:
                raise ValueError('invalid RSA')
            if (not site.get('source_sha256') or not site.get('source_path') or not site.get('entry_id')
                    or site.get('entity_position') is None or site.get('model') is None or not site.get('chain')):
                raise ValueError('incomplete exact mapping provenance')
            rows.append(dict(assay=assay['assay'], cluster=assay['cluster'], mutation=mutation,
                             variant_index=j, original_index=offset+j, **{'site': site}))
        offset += len(mutants)
    # Ranking requires at least three retained substitutions per assay, label-blind.
    counts = Counter(r['assay'] for r in rows)
    kept = [r for r in rows if counts[r['assay']] >= 3]
    excluded['assay_below_three_rows'] += len(rows)-len(kept)
    return kept, dict(excluded)


def structural_features(rows):
    scales = _scale_matrix(); result = []
    for row in rows:
        mutation = row['mutation']; rsa = float(row['site']['rsa'])
        change = scales[:, AA20.index(mutation[-1])] - scales[:, AA20.index(mutation[0])]
        result.append(np.r_[rsa, rsa*change])
    return np.asarray(result, float).reshape((-1, 1+len(SCALE_ORDER)))


def rerank(values, assay_ids):
    result = np.empty(len(values), float)
    for assay in np.unique(assay_ids):
        mask = assay_ids == assay
        result[mask] = standardized_rank(np.asarray(values)[mask])
    return result


def project_folds(signature, families):
    def native(values):
        return {v.item() if isinstance(v, np.generic) else v for v in values}
    present = native(families)
    projected = []; seen = set()
    if len(signature) != 5:
        raise ValueError('expected five original outer folds')
    for fold, held, training, inner in signature:
        held = sorted(native(held) & present); training = sorted(native(training) & present)
        if set(held) & set(training) or set(held) | set(training) != present or seen & set(held):
            raise ValueError('invalid original outer membership')
        if not held or not training:
            raise UnsupportedProjection('empty projected outer test/train fold')
        seen.update(held)
        if len(inner) != 4:
            raise ValueError('expected four original inner folds')
        validations = [sorted(native(v) & set(training)) for v in inner]
        covered = set()
        for validation in validations:
            if not validation or not set(training)-set(validation):
                raise UnsupportedProjection('empty projected inner validation/train fold')
            if covered & set(validation) or not set(validation) <= set(training):
                raise ValueError('invalid inner partition')
            covered.update(validation)
        if covered != set(training):
            raise ValueError('incomplete inner partition')
        projected.append((fold, held, training, validations))
    if seen != present:
        raise ValueError('incomplete outer partition')
    return projected


def projected_predict(x, measured, aids, families, signature):
    x = np.asarray(x, float); measured = np.asarray(measured, float)
    aids = np.asarray(aids); families = np.asarray(families)
    if x.ndim != 2 or not len(x) or any(len(a) != len(x) for a in (measured,aids,families)):
        raise ValueError('unaligned or empty fitting arrays')
    if not np.isfinite(x).all() or not np.isfinite(measured).all():
        raise ValueError('nonfinite fitting arrays')
    row_weights(aids, families)
    target = rerank(measured, aids); prediction = np.full(len(x), np.nan); records = []
    for fold, held, training, validations in project_folds(signature, families):
        train = np.flatnonzero(np.isin(families, training)); test = np.flatnonzero(np.isin(families, held))
        losses = np.zeros(len(ALPHAS)); inner = []
        for validation in validations:
            valid = np.flatnonzero(np.isin(families, validation))
            fit = train[~np.isin(families[train], validation)]
            pred = ridge_predict(x[fit], target[fit], row_weights(aids[fit],families[fit]), x[valid], ALPHAS, 'cpu')
            loss = (row_weights(aids[valid],families[valid])[:,None]*(pred-target[valid,None])**2).sum(0)
            losses += len(validation)*loss
            inner.append(dict(validation_families=validation, rank_mse=loss.tolist()))
        losses /= len(training)
        best = len(losses)-1-int(np.argmin(losses[::-1])); alpha = ALPHAS[best]
        prediction[test] = ridge_predict(x[train],target[train],row_weights(aids[train],families[train]),x[test],[alpha],'cpu')[:,0]
        records.append(dict(fold=fold,held_families=held,training_families=training,alpha=alpha,
                            inner_family_rank_mse=losses.tolist(),inner_folds=inner))
    if not np.isfinite(prediction).all():
        raise ValueError('incomplete OOF')
    return prediction, records


def contrast_values(scores, metric):
    """Positive means incremental benefit; attenuation = unconditional - conditional."""
    sign = 1 if metric == 'spearman' else -1
    original = sign*(scores['B+M']-scores['B'])
    structure = sign*(scores['B+X']-scores['B'])
    conditional = sign*(scores['B+M+X']-scores['B+X'])
    return dict(zip(CONTRASTS,(original,structure,conditional,original-conditional)))


def assay_metrics(predictions, measured, aids, families, arm, seed):
    target = rerank(measured,aids); rows = []
    for assay in sorted(set(aids)):
        ix = aids == assay
        row: dict[str, Any] = dict(assay=assay,cluster=str(families[ix][0]),arm=arm,seed=seed)
        for metric in ('spearman','rank_mse'):
            scores = {name: (correlation(rankdata(pred[ix]),target[ix]) if metric == 'spearman'
                            else float(np.mean((pred[ix]-target[ix])**2))) for name,pred in predictions.items()}
            row[metric] = scores
            row[metric+'_contrasts'] = (None if any(v is None for v in scores.values())
                                        else contrast_values(scores,metric))
        rows.append(row)
    return rows


def joint_bootstrap(rows, *, bootstrap=2000, seed=20261006, split=None) -> dict[str, Any]:
    """Joint complete-family bootstrap; seeds -> assay -> family, no refitting."""
    selected = [r for r in rows if split is None or r['seed'] == split]
    keys = [(arm,metric,c) for arm in sorted({r['arm'] for r in selected})
            for metric in ('spearman','rank_mse') for c in CONTRASTS]
    families = sorted({r['cluster'] for r in selected}); matrix = []
    for family in families:
        cells = []
        for arm,metric,c in keys:
            assays = sorted({r['assay'] for r in selected if r['cluster']==family and r['arm']==arm})
            values = []
            for assay in assays:
                rs = [r for r in selected if r['arm']==arm and r['assay']==assay]
                if any(r[metric+'_contrasts'] is None for r in rs):
                    return {'status':'not_estimable','reason':'undefined per-assay correlation; no outcome-based dropping'}
                values.append(np.mean([r[metric+'_contrasts'][c] for r in rs]))
            cells.append(np.mean(values))
        matrix.append(cells)
    if len(families)<2 or bootstrap<100:
        return {'status':'not_estimable','reason':'insufficient families or bootstrap draws'}
    matrix = np.asarray(matrix); point = matrix.mean(0); rng = np.random.default_rng(seed)
    draws = np.asarray([matrix[rng.integers(len(matrix),size=len(matrix))].mean(0) for _ in range(bootstrap)])
    lo,hi = np.quantile(draws,[.025,.975],axis=0)
    # Single max-deviation band over all arms, metrics and primary contrasts.
    critical = float(np.quantile(np.max(np.abs(draws-point),axis=1),.95))
    return {'status':'complete','families':families,'bootstrap':bootstrap,'seed':seed,
            'multiplicity':'all arms x 2 metrics x 4 primary contrasts',
            'uncertainty':'conditional on fitted OOF; no refitting or training-history uncertainty',
            'contrasts':[dict(arm=a,metric=m,contrast=c,point=float(point[j]),
                        pointwise=[float(lo[j]),float(hi[j])],
                        simultaneous=[float(point[j]-critical),float(point[j]+critical)])
                         for j,(a,m,c) in enumerate(keys)]}
