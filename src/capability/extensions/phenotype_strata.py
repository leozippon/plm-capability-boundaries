"""Retrospective fixed-OOF metadata strata; no fitting or checkpoint bootstrap."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

CATEGORIES = ('Activity', 'Binding', 'Expression', 'OrganismalFitness', 'Stability')
SEEDS = (20260923, 20260924, 20260925)


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity_digest(samples) -> str:
    return hashlib.sha256(json.dumps(samples, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_digest(values) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def validate_archive(saved, samples, target):
    ids = np.asarray([r['sample_id'] for r in samples])
    if len(set(ids)) != len(ids) or len({(r['assay'], r['mutation']) for r in samples}) != len(ids):
        raise ValueError('duplicate sample IDs')
    for key, expected in [('sample_id', ids), ('original_index', [r['original_index'] for r in samples]), ('target', target)]:
        if not np.array_equal(saved[key], expected):
            raise ValueError(f'misaligned OOF {key}')
    for key in ('BPL', 'BMPL'):
        if saved[key].shape != (len(ids),) or not np.isfinite(saved[key]).all():
            raise ValueError('invalid fitted OOF predictions')


def audit_export_groups(samples, exported):
    """Verify the export against canonical fitting groups, never reported counts."""
    canonical = pd.DataFrame(samples)
    if exported.sample_id.duplicated().any():
        raise ValueError('duplicate exported sample identity')
    flags = exported.headline_common_member.astype(str).str.lower()
    if not set(flags) <= {'true', 'false'}:
        raise ValueError('invalid exported support flag')
    selected = exported.loc[flags.eq('true')]
    if set(selected.sample_id) != set(canonical.sample_id):
        raise ValueError('export common support differs from production')
    joined = canonical.merge(selected[['sample_id', 'unit_id', 'mutation', 'group_id']],
                             on='sample_id', validate='one_to_one', suffixes=('', '_export'))
    if (not joined.assay.eq(joined.unit_id).all()
            or not joined.mutation.eq(joined.mutation_export).all()
            or not pd.to_numeric(joined.group_id, errors='raise').eq(joined.cluster).all()):
        raise ValueError('export assay/mutation/group differs from production')
    return dict(authoritative='production full/samples.json cluster and contract',
                status='exact_export_identity_and_group_agreement', rows=len(joined),
                biological_families=int(joined.cluster.nunique()))


def assay_scores(saved, samples, arm, seed):
    frame = pd.DataFrame(samples)
    rows = []
    for assay, local in frame.groupby('assay', sort=True):
        ix = local.index.to_numpy()
        y, b, a = (saved[k][ix] for k in ('target', 'BPL', 'BMPL'))
        if min(len(np.unique(v)) for v in (y, b, a)) < 2:
            raise ValueError(f'nonestimable assay; no outcome-based deletion: {assay}')
        base, aug = (float(np.corrcoef(rankdata(y), rankdata(v))[0, 1]) for v in (b, a))
        rows.append(dict(arm=arm, seed=seed, assay=assay, cluster=int(local.cluster.iloc[0]),
                         baseline_spearman=base, augmented_spearman=aug, contrast=aug-base))
    return rows


def family_matrix(scores, metadata, arms, families, seeds=SEEDS):
    """Average seeds within assay, then equal assays within category/family."""
    frame = pd.DataFrame(scores)
    if frame.duplicated(['arm', 'seed', 'assay']).any():
        raise ValueError('duplicate metric cell')
    expected = {(a, s, x) for a in arms for s in seeds for x in metadata.assay}
    if set(frame[['arm', 'seed', 'assay']].itertuples(index=False, name=None)) != expected:
        raise ValueError('incomplete model/seed/assay support')
    averaged = frame.groupby(['arm', 'assay', 'cluster'], as_index=False)[['baseline_spearman', 'augmented_spearman', 'contrast']].mean()
    averaged = averaged.merge(metadata[['assay', 'category']], on='assay', validate='many_to_one')
    grouped = averaged.groupby(['cluster', 'arm', 'category'])[['baseline_spearman', 'augmented_spearman', 'contrast']].mean()
    keys = [(a, c) for a in arms for c in CATEGORIES]
    matrices = {}
    index = pd.MultiIndex.from_tuples([(f, a, c) for f in families for a, c in keys], names=['cluster', 'arm', 'category'])
    for metric in ('baseline_spearman', 'augmented_spearman', 'contrast'):
        lookup = grouped.reset_index().set_index(['cluster', 'arm', 'category']).to_dict('index')
        matrices[metric] = np.asarray([lookup.get(tuple(key), {}).get(metric, np.nan) for key in index]).reshape(len(families), len(keys))
    return keys, matrices, averaged


def shared_bootstrap(values, *, draws=10000, seed=20261007):
    """Global family draws, category-specific ratio means, paired max fixed-SE.

    Structural missingness is never zero-valued evidence. Entire-category-empty
    draws are rejected JOINTLY for every contrast and redrawn, conditioning the
    bootstrap on representation of every supported category; rejected draws are
    counted explicitly. No checkpoint/seed resampling and no fitting occurs.
    """
    values = np.asarray(values, float)
    if values.ndim != 2 or len(values) < 2 or np.isinf(values).any():
        raise ValueError('invalid family contrast matrix')
    mask = np.isfinite(values)
    n = mask.sum(axis=0)
    if (n < 2).any():
        raise ValueError('each supported contrast needs at least two families')
    point = np.nanmean(values, axis=0)
    se = np.nanstd(values, axis=0, ddof=1) / np.sqrt(n)
    rng = np.random.default_rng(seed)
    output, accepted, rejected, attempts = [], 0, np.zeros(values.shape[1], int), 0
    while accepted < draws:
        size = min(100, draws-accepted)
        counts = np.asarray([np.bincount(rng.integers(len(values), size=len(values)), minlength=len(values)) for _ in range(size)])
        denominator = counts @ mask.astype(float)
        empty = denominator == 0
        rejected += empty.sum(axis=0)
        keep = ~empty.any(axis=1)
        estimates = (counts[keep] @ np.nan_to_num(values)) / denominator[keep]
        output.append(estimates)
        accepted += len(estimates)
        attempts += size
        if attempts > draws * 100:
            raise ValueError('excessive empty-category draws; inference blocked')
    boot = np.concatenate(output)
    varying = se > 0
    maxima = np.max(np.abs((boot[:, varying]-point[varying])/se[varying]), axis=1) if varying.any() else np.zeros(draws)
    critical = float(np.quantile(maxima, .95))
    return dict(point=point.tolist(), se=se.tolist(), pointwise_interval=np.quantile(boot, [.025, .975], axis=0).T.tolist(),
                simultaneous_interval=np.column_stack((point-critical*se, point+critical*se)).tolist(),
                confidence=.95, critical_value=critical, family_size=values.shape[1], original_families=len(values),
                supported_families=n.tolist(), draws=draws, seed=seed, attempted_draws=attempts,
                jointly_rejected_draws=attempts-draws, empty_draws_by_contrast=rejected.tolist(),
                missing_category_policy='joint reject/redraw; conditional on every category represented; never zero impute',
                method='paired shared original-family bootstrap; category ratio means; max absolute centered / fixed category SE',
                conditional_on_fitted_predictions=True), boot


def load_production(root: Path):
    """Verify completed producer bytes, exact cohort identity/target and folds."""
    base = root/'results/extensions/information_progression_20261006'
    hashes = {}
    def load(path):
        hashes[str(path.relative_to(root))] = digest(path)
        return json.loads(path.read_text())
    completion = load(base/'completion.json')
    if completion['status'] != 'complete' or not completion['tests_passed'] or not completion['full_panel_fitted']:
        raise ValueError('production not complete')
    for relative, sha in completion['output_sha256'].items():
        if relative == 'contract.json' or relative.startswith('full/'):
            path = base/relative
            if digest(path) != sha:
                raise ValueError(f'producer artifact hash mismatch: {relative}')
            hashes[str(path.relative_to(root))] = sha
    contract = load(base/'contract.json')
    if contract['mode'] != 'production' or len(contract['models']) != 33 or contract['seeds'] != list(SEEDS):
        raise ValueError('not full production model panel')
    from scripts.capability.reporting.export_prediction_details import MUTATION, LOCAL, sample_id
    for relative, key in [(MUTATION, 'cohort_sha256'), (LOCAL, 'admission_sha256')]:
        if digest(root/relative) != contract['provenance'][key]:
            raise ValueError('cohort/admission hash mismatch')
    cohort, admission = load(root/MUTATION), load(root/LOCAL)
    selected = sorted(admission['support']['assay_ids'])
    mapping = {a['assay']: a for a in cohort['assays']}
    samples = load(base/'full/samples.json')
    expected, target = [], []
    for assay in selected:
        a = mapping[assay]
        ranks = rankdata(a['measured'], method='average'); ranks = (ranks-ranks.mean())/ranks.std()
        target.extend(ranks)
        for j, mutation in enumerate(a['mutants']):
            expected.append(dict(assay=assay, cluster=a['cluster'], mutation=mutation, variant_index=j,
                                 original_index=len(expected), sample_id=sample_id(contract['provenance']['cohort_sha256'], assay, mutation)))
    if samples != expected or identity_digest(samples) != contract['panel_support']['full']['support_sha256']:
        raise ValueError('exact production sample registry differs from source cohort')
    if (len(samples), len(selected), len({r['cluster'] for r in samples})) != (25728, 201, 163):
        raise ValueError('wrong exact all-model score coverage')
    projected = load(base/'full/projected-memberships.json')
    scores = []
    cells = [c for c in completion['completed_cells'] if c['panel'] == 'full']
    if len(cells) != 99 or {(c['arm'], c['seed']) for c in cells} != {(a,s) for a in contract['models'] for s in SEEDS}:
        raise ValueError('missing or duplicate completed OOF cells')
    families = {r['cluster'] for r in samples}
    for cell in cells:
        stem = f"{cell['arm']}-seed{cell['seed']}"
        path = base/'full'/f'{stem}.npz'
        if digest(path) != cell['prediction_archive_sha256'] or not cell['identity_verified'] or not cell['folds_verified']:
            raise ValueError('completion cell mismatch')
        checks = load(base/'full'/f'{stem}-checks.json')['designs']
        with np.load(path, allow_pickle=False) as saved:
            validate_archive(saved, samples, np.asarray(target))
            for design in ('BPL', 'BMPL'):
                check = checks[design]
                if (check['prediction_sha256'] != array_digest(saved[design]) or check['target_sha256'] != array_digest(target)
                        or check['support_sha256'] != identity_digest(samples)):
                    raise ValueError('OOF numerical check differs')
                folds = load(base/'full'/f'{stem}-{design}-folds.json')
                membership = [[f['fold'], f['held_families'], f['training_families'], [i['validation_families'] for i in f['inner_folds']]] for f in folds]
                if membership != projected[str(cell['seed'])]:
                    raise ValueError('same-fold support differs')
                held = [f for fold in folds for f in fold['held_families']]
                if len(held) != len(set(held)) or set(held) != families:
                    raise ValueError('OOF held-family coverage differs')
                if any(set(f['held_families']) & set(f['training_families']) or set(f['held_families']) | set(f['training_families']) != families for f in folds):
                    raise ValueError('held/training family leakage')
            scores.extend(assay_scores(saved, samples, cell['arm'], cell['seed']))
    return contract, samples, scores, hashes
