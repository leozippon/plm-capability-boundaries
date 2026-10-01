"""Readout-class sweep over the frozen readout states of the admitted panel.

The representation is held fixed. What varies is the class of function fitted to
it: the penalty range, whether the four block summaries enter compressed or at
full width, whether the head is linear or a random-feature approximation of a
radial-basis kernel, and which depth and pooling block is supplied. Everything
else -- cohort, assay support, panel, split seeds, outer and inner fold maps,
equal cluster/assay/variant weights, within-assay standardized-rank targets,
training-only feature scaling, the ridge objective and tie rule, the 2,000-draw
paired cluster bootstrap at alpha 0.05 and its unit -- is imported unchanged from
:mod:`src.capability.readouts.readout_analysis` and :mod:`src.capability.context.profile_increment`,
so every estimate is directly comparable to its admitted baseline.

A resolved increment in a class states that a function of that class carries
mutation-effect information the matched supervised baseline does not; it does not
locate where that information came from. An unresolved or negative increment
states that no function this class reached carries such information, and is bound
by the classes actually fitted, the label budget and the two extraction depths the
retained artifacts hold.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_info, threadpool_limits
import torch

from ..context.profile_increment import correlation, standardized_rank, summarize
from .readout_analysis import (ALPHAS as BASELINE_ALPHAS, family_folds, row_weights, sequence_features)

#: Extended ridge grid. It contains the admitted five-point grid and reaches
#: eleven orders of magnitude, because a design of 28,672 standardized columns
#: needs a penalty range the five-point grid cannot express. The upper end is
#: chosen so that the largest penalty is already the near-constant predictor: the
#: weighted correlation matrix of d standardized columns has largest eigenvalue at
#: most d, at most 28,672 here, so 1e6 shrinks even that direction by a factor
#: below 0.03. Selection stays in inner folds, and every selected penalty is
#: reported so a boundary selection is visible rather than inferred.
EXTENDED_ALPHAS = (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
#: Random-feature bandwidth multipliers, the nonlinear head's capacity knob,
#: selected jointly with the penalty in inner folds. Each input group is divided
#: by the square root of its own width before the map, so the squared distance
#: between two rows is about twice the number of groups and these multipliers span
#: a near-nearest-neighbour kernel through a near-linear one.
BANDWIDTHS = (0.5, 1.0, 2.0, 4.0)
#: Random-feature count and draw seed. The count is fixed rather than tuned: a
#: larger count approximates the kernel more closely, and 2,048 is a declared bound.
RANDOM_FEATURES = 2048
RANDOM_FEATURE_SEED = 20260927
#: Fixed projection contract of the admitted study, reproduced exactly.
PROJECTION_DIM = 256
PROJECTION_SEED = 20260923
#: BLAS thread count the admitted analysis formed the compressed block under, read
#: from the admitted analysis campaign manifests (`OMP_NUM_THREADS=4`,
#: `MKL_NUM_THREADS=4`). It belongs to the projection contract because a float32
#: matrix product is not thread-order invariant: over one 128-row assay of hidden
#: width 3,584, the four-thread and forty-eight-thread products of the admitted
#: projection differ by up to 1.073e-06 in absolute value, against a largest
#: absolute entry of 2.528 and a float32 rounding of the same product of 7.128e-07
#: against a float64 reference. The admitted per-variant predictions are bound at
#: 1e-8, below that floor, so this count is pinned inside the loader rather than
#: left to the caller's environment.
PROJECTION_BLAS_THREADS = 4
FEATURE_NAMES = ('middle_mean', 'middle_last', 'final_mean', 'final_last')
#: Fixed label-permutation diagnostic offset of the admitted study.
PERMUTATION_OFFSET = 9000
BOOTSTRAP_DRAWS = 2000
#: Row block for float64 accumulation over full-width states, in rows.
CHUNK_ROWS = 4096


def weighted_ridge_predict(x_train, y_train, weights, x_test, alphas=EXTENDED_ALPHAS,
                           device='cpu'):
    """Weighted mean squared error + alpha ||beta||^2, unpenalized intercept.

    The objective, the weighting, the training-only weighted centering and scaling,
    the constant-column rule and the returned prediction are those of
    :func:`readout_analysis.ridge_predict`. Only the linear algebra differs: one
    Cholesky factorization per penalty rather than one symmetric eigendecomposition
    for the whole grid, because on this host a 16,384-column eigendecomposition
    costs about seventy Cholesky factorizations of the same matrix and the
    full-width designs reach 29,118 columns. Accumulation runs over row blocks in
    float64 so a full-width design never needs a second float64 copy of itself.
    Agreement with the eigendecomposition path is asserted by test and, on real
    designs, by reproducing the admitted baseline predictions.
    """
    if any(a <= 0 for a in alphas):
        raise ValueError('ridge alpha must be positive')
    rows, columns = x_train.shape
    if len(y_train) != rows or len(weights) != rows:
        raise ValueError('unaligned ridge inputs')

    def block(source, start):
        # An owning float64 copy: the accumulators below centre and scale in place,
        # and a shared view would rewrite the caller's design matrix.
        stop = min(start + CHUNK_ROWS, len(source))
        return torch.tensor(source[start:stop], dtype=torch.float64, device=device), stop

    w = torch.as_tensor(weights, dtype=torch.float64, device=device)
    w = w / w.sum()
    y = torch.as_tensor(y_train, dtype=torch.float64, device=device)
    centre = torch.zeros(columns, dtype=torch.float64, device=device)
    for start in range(0, rows, CHUNK_ROWS):
        chunk, stop = block(x_train, start)
        centre += w[start:stop] @ chunk
    spread = torch.zeros(columns, dtype=torch.float64, device=device)
    for start in range(0, rows, CHUNK_ROWS):
        chunk, stop = block(x_train, start)
        chunk -= centre
        spread += w[start:stop] @ chunk.square()
    scale = spread.sqrt()
    scale = torch.where(scale < 1e-12, torch.ones_like(scale), scale)
    offset = (w * y).sum()
    covariance = torch.zeros(columns, columns, dtype=torch.float64, device=device)
    rhs = torch.zeros(columns, dtype=torch.float64, device=device)
    for start in range(0, rows, CHUNK_ROWS):
        chunk, stop = block(x_train, start)
        chunk -= centre
        chunk /= scale
        weight = w[start:stop]
        covariance += chunk.T @ (weight[:, None] * chunk)
        rhs += chunk.T @ (weight * (y[start:stop] - offset))
    work = torch.empty_like(covariance)
    beta = torch.empty(columns, len(alphas), dtype=torch.float64, device=device)
    for index, alpha in enumerate(alphas):
        work.copy_(covariance)
        work.diagonal().add_(float(alpha))
        factor, info = torch.linalg.cholesky_ex(work)
        if int(info) != 0:
            raise ValueError(f'ridge solve failed at alpha {alpha}')
        beta[:, index] = torch.cholesky_solve(rhs[:, None], factor)[:, 0]
    del covariance, work, factor
    predicted = torch.empty(len(x_test), len(alphas), dtype=torch.float64, device=device)
    for start in range(0, len(x_test), CHUNK_ROWS):
        chunk, stop = block(x_test, start)
        chunk -= centre
        chunk /= scale
        predicted[start:stop] = chunk @ beta + offset
    return predicted.cpu().numpy()


@dataclass(frozen=True)
class Design:
    """One fitted predictor: named raw blocks entering linearly and through the map."""

    name: str
    linear: tuple[str, ...]
    mapped: tuple[str, ...] = ()

    def key(self, alphas, bandwidths):
        return (self.linear, self.mapped, tuple(alphas), tuple(bandwidths))


def _designs(representation, mapped_representation=None):
    """Baseline, representation-only and baseline-plus-representation designs."""
    if mapped_representation is None:
        return (Design('B', ('B',)), Design('R', (representation,)),
                Design('B_R', ('B', representation)))
    return (Design('B', ('B',), ('B',)),
            Design('R', (representation,), (mapped_representation,)),
            Design('B_R', ('B', representation), ('B', mapped_representation)))


#: The declared class set. C0 reproduces the admitted class and its penalty grid
#: exactly and exists to prove the pipeline is matched; C1 to C4 form the
#: {linear, random-feature} x {compressed, full-width} factorial. The
#: random-feature classes keep the compressed coordinates in their linear block,
#: so each contains the compressed linear class of the same designs; neither
#: contains the full-width linear class.
CLASSES: dict[str, dict] = {
    'C0_compressed_linear_baseline_grid':
        dict(alphas=BASELINE_ALPHAS, bandwidths=(None,), designs=_designs('R_proj')),
    'C1_compressed_linear':
        dict(alphas=EXTENDED_ALPHAS, bandwidths=(None,), designs=_designs('R_proj')),
    'C2_full_linear':
        dict(alphas=EXTENDED_ALPHAS, bandwidths=(None,), designs=_designs('R_full')),
    'C3_compressed_random_feature':
        dict(alphas=EXTENDED_ALPHAS, bandwidths=BANDWIDTHS, designs=_designs('R_proj', 'R_proj')),
    'C4_full_random_feature':
        dict(alphas=EXTENDED_ALPHAS, bandwidths=BANDWIDTHS, designs=_designs('R_proj', 'R_full')),
}

#: Depth and pooling resolution at the resolution the retained artifacts hold: two
#: extraction depths crossed with two pooling rules. The retained states hold no
#: other layer and no per-position state, so neither a per-layer sweep nor a
#: position-resolved readout can be fitted from them.
DEPTH_CLASSES: dict[str, dict] = {
    f'D_{block}': dict(alphas=EXTENDED_ALPHAS, bandwidths=(None,), designs=_designs(block))
    for block in FEATURE_NAMES
}

REPRODUCTION_CLASS = 'C0_compressed_linear_baseline_grid'


def projection_matrices(width: int) -> list[np.ndarray]:
    """The admitted fixed Gaussian projection: four blocks, 256 coordinates each."""
    return [np.random.default_rng(PROJECTION_SEED + i)
            .normal(0, 1 / np.sqrt(PROJECTION_DIM), size=(width, PROJECTION_DIM)).astype(np.float32)
            for i in range(4)]


def blas_thread_counts() -> list[int]:
    """Thread counts of the BLAS pools this process dispatches float32 products to."""
    return [int(pool['num_threads']) for pool in threadpool_info()
            if pool.get('user_api') == 'blas']


def projected_block(hidden: np.ndarray, projection: list[np.ndarray]) -> np.ndarray:
    """The admitted compressed block, formed under the admitted BLAS reduction order.

    See :data:`PROJECTION_BLAS_THREADS`: the reduction order of this float32
    product, and therefore its value in the seventh significant digit, depends on
    the BLAS thread count, so the count is part of the reproduction and is pinned
    here. A process with no discoverable BLAS pool cannot reproduce the admitted
    order and is refused rather than run unpinned.
    """
    if not blas_thread_counts():
        raise ValueError('no BLAS thread pool to pin the admitted projection to')
    with threadpool_limits(limits=PROJECTION_BLAS_THREADS, user_api='blas'):
        observed = blas_thread_counts()
        if any(count != PROJECTION_BLAS_THREADS for count in observed):
            raise ValueError(f'BLAS pools stand at {observed} threads, not the admitted '
                             f'{PROJECTION_BLAS_THREADS}')
        return np.concatenate([hidden[:, i] @ projection[i] for i in range(4)], axis=1)


def load_panel(cohort_path, manifest_path, arm: str, assay_ids) -> tuple[list[dict], dict]:
    """Load one arm on one declared assay support, verifying every bound identity."""
    cohort_path, manifest_path = Path(cohort_path), Path(manifest_path)
    hashes: dict[str, str] = {}

    def read(path):
        raw = Path(path).read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    cohort = read(cohort_path)
    manifest = read(manifest_path)
    if manifest['status'] != 'complete':
        raise ValueError(f'incomplete extraction: {manifest_path}')
    if manifest['identity']['arm'] != arm:
        raise ValueError('manifest arm mismatch')
    if list(manifest['identity']['feature_names']) != list(FEATURE_NAMES):
        raise ValueError('representation block order mismatch')
    if manifest['identity']['cohort_sha256'] != hashes[str(cohort_path)]:
        raise ValueError('cohort hash mismatch')
    cohort_map = {r['assay']: r for r in cohort['assays']}
    if len(cohort_map) != len(cohort['assays']):
        raise ValueError('duplicate cohort assays')
    records = {r['assay']: r for r in manifest['assays']}
    if len(records) != len(manifest['assays']):
        raise ValueError('duplicate extraction assays')
    requested = sorted(assay_ids)
    if len(set(requested)) != len(requested):
        raise ValueError('duplicate requested assays')
    missing = [a for a in requested if a not in records]
    if missing:
        raise ValueError(f'{arm} does not cover {len(missing)} requested assays, e.g. {missing[:3]}')
    rows, projection, width = [], None, None
    for assay in requested:
        cohort_row, record = cohort_map[assay], records[assay]
        for key in ('cluster', 'wildtype_id', 'mutant_digest'):
            if record[key] != cohort_row[key]:
                raise ValueError(f'{assay}: manifest {key} mismatch')
        if record['variants'] != len(cohort_row['mutants']):
            raise ValueError(f'{assay}: variant count mismatch')
        path = manifest_path.parent / record['file']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record['sha256']:
            raise ValueError(f'{assay}: extraction checksum mismatch')
        hashes[str(path)] = digest
        with np.load(path, allow_pickle=False) as data:
            expected = dict(identity=manifest['identity'], assay=assay,
                            mutant_digest=cohort_row['mutant_digest'])
            if json.loads(str(data['metadata'])) != expected:
                raise ValueError(f'{assay}: extraction metadata mismatch')
            if data['mutants'].tolist() != cohort_row['mutants']:
                raise ValueError(f'{assay}: exact mutation order mismatch')
            for key in ('measured', 'profile_scores'):
                if not np.array_equal(data[key], np.asarray(cohort_row[key])):
                    raise ValueError(f'{assay}: {key} mismatch')
            hidden = np.asarray(data['features'], dtype=np.float32)
            if hidden.ndim != 3 or hidden.shape[:2] != (len(cohort_row['mutants']), 4):
                raise ValueError(f'{assay}: expected four aligned hidden feature blocks')
            if width is None:
                width = int(hidden.shape[2])
                projection = projection_matrices(width)
            if hidden.shape[2] != width:
                raise ValueError('hidden width changed within arm')
            rows.append(dict(assay=assay, cluster=cohort_row['cluster'],
                             mutants=list(cohort_row['mutants']),
                             measured=np.asarray(data['measured'], dtype=np.float64),
                             P=np.asarray(data['profile_scores'], dtype=np.float64),
                             M=np.asarray(data['likelihood'], dtype=np.float64),
                             full=np.ascontiguousarray(hidden),
                             projected=projected_block(hidden, projection),
                             S=sequence_features(cohort_row['wildtype'], cohort_row['mutants'])))
    provenance = dict(arm=arm, hidden_width=width, source_sha256=hashes,
                      projection_blas_threads=PROJECTION_BLAS_THREADS,
                      assay_ids=requested, n_assays=len(rows),
                      n_clusters=len({r['cluster'] for r in rows}),
                      n_variants=int(sum(len(r['mutants']) for r in rows)),
                      extraction_identity=manifest['identity'],
                      extraction_block_indices=manifest.get('block_indices'))
    return rows, provenance


def raw_blocks(rows: list[dict], needed) -> dict[str, np.ndarray]:
    """Named raw feature blocks over the concatenated variant rows.

    The four block summaries stay in float32 as they were retained; the baseline
    block stays in float64, so every design this builds has the same dtype as the
    admitted study's design of the same columns.
    """
    needed = set(needed)
    unknown = needed - {'B', 'R_proj', 'R_full', *FEATURE_NAMES}
    if unknown:
        raise ValueError(f'unknown raw blocks {sorted(unknown)}')
    blocks: dict[str, np.ndarray] = {}
    blocks['B'] = np.column_stack([np.concatenate([standardized_rank(r['P']) for r in rows]),
                                   np.concatenate([standardized_rank(r['M']) for r in rows]),
                                   np.concatenate([r['S'] for r in rows])])
    if 'R_proj' in needed:
        blocks['R_proj'] = np.concatenate([r['projected'] for r in rows])
    for r in rows:
        r.pop('projected', None)
    if needed & {'R_full', *FEATURE_NAMES}:
        full = np.concatenate([r['full'] for r in rows])
        for r in rows:
            r.pop('full', None)
        if 'R_full' in needed:
            blocks['R_full'] = full.reshape(len(full), -1)
        for index, name in enumerate(FEATURE_NAMES):
            if name in needed:
                blocks[name] = np.ascontiguousarray(full[:, index, :])
    for r in rows:
        r.pop('full', None)
    for name, value in blocks.items():
        if not np.isfinite(value).all():
            raise ValueError(f'nonfinite raw block {name}')
    return blocks


def weighted_moments(block: np.ndarray, rows: np.ndarray, weights: np.ndarray):
    """Weighted column mean and scale over selected rows; constant columns get scale one."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    total = np.zeros(block.shape[1])
    square = np.zeros(block.shape[1])
    for start in range(0, len(rows), CHUNK_ROWS):
        index = rows[start:start + CHUNK_ROWS]
        chunk = np.asarray(block[index], dtype=np.float64)
        weight = w[start:start + CHUNK_ROWS]
        total += weight @ chunk
        square += weight @ (chunk * chunk)
    scale = np.sqrt(np.maximum(square - total * total, 0.0))
    return total, np.where(scale < 1e-12, 1.0, scale)


def random_feature_map(widths: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Fixed Gaussian frequencies and phases for one group layout."""
    rng = np.random.default_rng(RANDOM_FEATURE_SEED)
    total = int(sum(widths))
    frequencies = rng.standard_normal((total, RANDOM_FEATURES))
    phases = rng.uniform(0.0, 2.0 * np.pi, RANDOM_FEATURES)
    offset = 0
    for width in widths:
        frequencies[offset:offset + width] /= np.sqrt(width)
        offset += width
    return frequencies, phases


class DesignBuilder:
    """Assemble a design matrix from raw blocks, using fitting rows only for scaling.

    A linear block passes through unchanged: :func:`ridge_predict` applies its own
    training-weighted standardization, and ridge on standardized columns is
    invariant to a per-column affine change of its input. A mapped block is
    standardized here on the fitting rows alone, divided by the square root of its
    own width, and passed through the fixed random-feature map. No held-out row's
    label enters either step, and no held-out row enters the scaling statistics.
    """

    def __init__(self, blocks: dict[str, np.ndarray]):
        self.blocks = blocks
        self._linear: dict[tuple[str, ...], np.ndarray] = {}
        self._maps: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}

    def linear(self, names: tuple[str, ...]) -> np.ndarray:
        if names not in self._linear:
            parts = [self.blocks[name] for name in names]
            self._linear[names] = parts[0] if len(parts) == 1 else np.hstack(parts)
        return self._linear[names]

    def widths(self, names: tuple[str, ...]) -> tuple[int, ...]:
        return tuple(int(self.blocks[name].shape[1]) for name in names)

    def build(self, design: Design, fit_rows: np.ndarray, fit_weights: np.ndarray,
              bandwidth) -> np.ndarray:
        linear = self.linear(design.linear)
        if not design.mapped:
            if bandwidth is not None:
                raise ValueError('a linear design takes no bandwidth')
            return linear
        if bandwidth is None:
            raise ValueError('a mapped design requires a bandwidth')
        widths = self.widths(design.mapped)
        if widths not in self._maps:
            self._maps[widths] = random_feature_map(widths)
        frequencies, phases = self._maps[widths]
        rows = len(linear)
        argument = np.tile(phases, (rows, 1))
        offset = 0
        for name, width in zip(design.mapped, widths):
            block = self.blocks[name]
            mean, scale = weighted_moments(block, fit_rows, fit_weights)
            scaled = frequencies[offset:offset + width] / (scale[:, None] * bandwidth)
            argument -= (mean / (scale * bandwidth)) @ frequencies[offset:offset + width]
            for start in range(0, rows, CHUNK_ROWS):
                stop = min(start + CHUNK_ROWS, rows)
                argument[start:stop] += np.asarray(block[start:stop], dtype=np.float64) @ scaled
            offset += width
        mapped = np.cos(argument, out=argument)
        mapped *= np.sqrt(2.0 / RANDOM_FEATURES)
        return np.hstack([np.asarray(linear, dtype=np.float64), mapped])


def nested_predict(builder: DesignBuilder, design: Design, target: np.ndarray, assay_ids, families, *,
                   fold_seed: int, alphas, bandwidths, outer_splits=5, inner_splits=4,
                   device='cpu') -> tuple[np.ndarray, list[dict]]:
    """Held-cluster predictions with the capacity grid selected inside inner folds only.

    ``target`` holds within-assay standardized ranks. Because every assay of a
    cluster stays together at both fold levels, a held-out cluster's labels never
    reach a fit, its feature scaling, its capacity selection or its calibration.
    """
    assay_ids, families = np.asarray(assay_ids), np.asarray(families)
    row_weights(assay_ids, families)  # refuses an assay that would cross folds
    target = np.asarray(target, dtype=np.float64)
    prediction = np.full(len(assay_ids), np.nan)
    records = []
    for outer, held in enumerate(family_folds(families, outer_splits, fold_seed)):
        train = np.flatnonzero(~np.isin(families, held))
        test = np.flatnonzero(np.isin(families, held))
        losses = np.zeros((len(bandwidths), len(alphas)))
        inner_records = []
        for validation in family_folds(families[train], inner_splits, fold_seed + 100 + outer):
            fit = train[~np.isin(families[train], validation)]
            valid = train[np.isin(families[train], validation)]
            fit_weights = row_weights(assay_ids[fit], families[fit])
            valid_weights = row_weights(assay_ids[valid], families[valid])
            fold_losses = np.zeros((len(bandwidths), len(alphas)))
            for index, bandwidth in enumerate(bandwidths):
                x = builder.build(design, fit, fit_weights, bandwidth)
                predicted = weighted_ridge_predict(x[fit], target[fit], fit_weights, x[valid],
                                                   alphas, device)
                fold_losses[index] = (valid_weights[:, None]
                                      * (predicted - target[valid][:, None]) ** 2).sum(0)
                del x
            losses += len(validation) * fold_losses
            inner_records.append(dict(validation_families=[int(v) for v in validation],
                                      rank_mse=fold_losses.tolist()))
        losses /= len(set(families[train]))
        # Exact ties break toward the stronger penalty, then the smoother bandwidth.
        _, _, _, best_band, best_alpha = min((losses[b, a], -alphas[a], -(bandwidths[b] or 0.0), b, a)
                                             for b in range(len(bandwidths))
                                             for a in range(len(alphas)))
        bandwidth, alpha = bandwidths[best_band], alphas[best_alpha]
        train_weights = row_weights(assay_ids[train], families[train])
        x = builder.build(design, train, train_weights, bandwidth)
        prediction[test] = weighted_ridge_predict(x[train], target[train], train_weights,
                                                  x[test], [alpha], device)[:, 0]
        del x
        records.append(dict(fold=outer, held_families=[int(v) for v in held],
                            n_training_families=len(set(families[train])),
                            alpha=float(alpha),
                            bandwidth=None if bandwidth is None else float(bandwidth),
                            inner_family_rank_mse=losses.tolist(), inner_folds=inner_records))
    if not np.isfinite(prediction).all():
        raise ValueError('incomplete or nonfinite held-out predictions')
    return prediction, records


def target_vector(values: np.ndarray, assay_ids: np.ndarray) -> np.ndarray:
    """Within-assay standardized average ranks. Assays never cross folds, so this is
    the same vector the admitted study computes inside each fold."""
    out = np.empty(len(values))
    for assay in np.unique(assay_ids):
        index = np.flatnonzero(assay_ids == assay)
        out[index] = standardized_rank(values[index])
    return out


def required_blocks(class_specs: dict[str, dict]) -> set[str]:
    names = {'B'}
    for spec in class_specs.values():
        for design in spec['designs']:
            names.update(design.linear)
            names.update(design.mapped)
    return names


def evaluate_class_sweep(rows: list[dict], *, class_specs: dict[str, dict], fold_seed: int,
                         device: str = 'cpu', bootstrap: int = BOOTSTRAP_DRAWS,
                         bootstrap_seed: int = PROJECTION_SEED,
                         shuffle_classes: tuple[str, ...] = (), progress=None) -> dict:
    """Fit every declared class on identical rows, folds, weights and targets."""
    if len({r['assay'] for r in rows}) != len(rows):
        raise ValueError('duplicate assays')
    for r in rows:
        n = len(r['mutants'])
        if n < 3 or len(set(r['mutants'])) != n:
            raise ValueError('too few or duplicate mutants')
        for key in ('measured', 'P', 'M', 'S'):
            if len(r[key]) != n or not np.isfinite(r[key]).all():
                raise ValueError(f'unaligned or nonfinite {key}')
    unknown = set(shuffle_classes) - set(class_specs)
    if unknown:
        raise ValueError(f'shuffle requested for undeclared classes {sorted(unknown)}')
    y = np.concatenate([r['measured'] for r in rows])
    aid = np.concatenate([[r['assay']] * len(r['mutants']) for r in rows])
    family = np.concatenate([[r['cluster']] * len(r['mutants']) for r in rows])
    blocks = raw_blocks(rows, required_blocks(class_specs))
    builder = DesignBuilder(blocks)
    measured = target_vector(y, aid)
    rng = np.random.default_rng(bootstrap_seed + PERMUTATION_OFFSET)
    shuffled = y.copy()
    for assay in np.unique(aid):
        index = np.flatnonzero(aid == assay)
        shuffled[index] = rng.permutation(y[index])
    targets = {'measured': measured, 'permuted': target_vector(shuffled, aid)}

    cache: dict[tuple, tuple[np.ndarray, list[dict]]] = {}
    predictions = {'raw_P': blocks['B'][:, 0].copy(), 'raw_M': blocks['B'][:, 1].copy()}
    folds: dict[str, list[dict]] = {}
    dimensions: dict[str, int] = {}
    for class_name, spec in class_specs.items():
        alphas, bandwidths = tuple(spec['alphas']), tuple(spec['bandwidths'])
        wanted = [('measured', design) for design in spec['designs']]
        if class_name in shuffle_classes:
            wanted += [('permuted', design) for design in spec['designs']
                       if design.name in ('B', 'B_R')]
        for kind, design in wanted:
            key = design.key(alphas, bandwidths) + (kind,)
            label = f'{class_name}/' + ('permuted_' if kind == 'permuted' else '') + design.name
            if key not in cache:
                if progress:
                    progress(label)
                cache[key] = nested_predict(builder, design, targets[kind], aid, family,
                                            fold_seed=fold_seed, alphas=alphas,
                                            bandwidths=bandwidths, device=device)
            predictions[label], folds[label] = cache[key]
            dimensions[label] = (sum(builder.widths(design.linear))
                                 + (RANDOM_FEATURES if design.mapped else 0))

    per_assay = []
    for r in rows:
        index = np.flatnonzero(aid == r['assay'])
        target = measured[index]
        record = dict(assay=r['assay'], cluster=int(r['cluster']), n_variants=len(index))
        for label, vector in predictions.items():
            record[f'{label}_spearman'] = correlation(rankdata(vector[index]), target)
            if not label.startswith('raw_'):
                record[f'{label}_rank_mse'] = float(np.mean((target - vector[index]) ** 2))
        for class_name in class_specs:
            prefix = f'{class_name}/'
            for stem in ('', 'permuted_'):
                if f'{prefix}{stem}B_spearman' not in record or f'{prefix}{stem}B_R_spearman' not in record:
                    continue
                low, high = record[f'{prefix}{stem}B_spearman'], record[f'{prefix}{stem}B_R_spearman']
                record[f'{prefix}{stem}delta_spearman'] = None if low is None or high is None else high - low
                record[f'{prefix}{stem}delta_rank_mse'] = (record[f'{prefix}{stem}B_rank_mse']
                                                           - record[f'{prefix}{stem}B_R_rank_mse'])
            for name in ('R', 'B_R', 'B'):
                if f'{prefix}{name}_spearman' not in record:
                    continue
                left, right = record[f'{prefix}{name}_spearman'], record['raw_M_spearman']
                record[f'{prefix}{name}_minus_raw_M_spearman'] = (None if left is None or right is None
                                                                  else left - right)
        per_assay.append(record)
    metrics = [k for k in per_assay[0] if k not in ('assay', 'cluster', 'n_variants')]
    retained = {k: v.tolist() for k, v in predictions.items()
                if k.startswith('raw_') or k.startswith(REPRODUCTION_CLASS + '/')}
    return dict(n_assays=len(per_assay), n_families=len(set(family)), n_variants=len(y),
                fold_seed=fold_seed, bootstrap_seed=bootstrap_seed,
                permutation_seed=bootstrap_seed + PERMUTATION_OFFSET,
                shuffle_classes=list(shuffle_classes), feature_dimensions=dimensions,
                alpha_grids={name: list(spec['alphas']) for name, spec in class_specs.items()},
                bandwidth_grids={name: list(spec['bandwidths']) for name, spec in class_specs.items()},
                summaries={k: summarize(per_assay, k, bootstrap=bootstrap, seed=bootstrap_seed)
                           for k in metrics},
                assays=per_assay, folds=folds, predictions=retained,
                prediction_assay_order=[r['assay'] for r in rows],
                protocol=('5 outer / 4 inner cluster-grouped folds; train-only weighted feature '
                          'scaling; within-assay standardized-rank targets; weighted rank-MSE '
                          'capacity tuning inside inner folds only; equal cluster and within-cluster '
                          'assay weights; unpenalized intercept'),
                ridge_objective=('weighted mean squared error + alpha * squared coefficient norm; '
                                 'weights sum to one'),
                random_feature_map=dict(count=RANDOM_FEATURES, seed=RANDOM_FEATURE_SEED,
                                        bandwidths=list(BANDWIDTHS),
                                        form=('sqrt(2/m) cos(u W / bandwidth + phase); u is each group '
                                              'standardized on fitting rows and divided by sqrt(group width)')),
                primary_contrast='per class, B_R minus B Spearman; B minus B_R rank MSE',
                uncertainty=('95% paired cluster bootstrap conditional on fitted cross-validation '
                             'predictions; training, tuning and split variation are not refitted, and '
                             'intervals are not adjusted across arms, classes, endpoints or seeds'),
                limitation=('Varies the readout class on one fixed pair of extraction depths and two '
                            'pooling rules. The retained states hold no other layer and no per-position '
                            'state, so neither a per-layer depth sweep nor a position-resolved readout '
                            'can be fitted from them. A null bounds these classes, this label budget '
                            'and these depths; it does not establish absent information.'))


def verify_against_admitted(admitted: dict, provenance: dict, report: dict) -> dict:
    """Bind this run to its admitted baseline: identical support, folds and refit.

    Every check is an equality or a maximum absolute deviation over quantities the
    admitted artifact already carries. A mismatch raises rather than being recorded
    as a tolerated difference.
    """
    if list(admitted['support']['assay_ids']) != list(provenance['assay_ids']):
        raise ValueError('support assay identifiers differ from the admitted report')
    for key in ('n_assays', 'n_families', 'n_variants', 'fold_seed', 'bootstrap_seed'):
        if admitted[key] != report[key]:
            raise ValueError(f'{key} differs from the admitted report')
    reference, mine = admitted['folds']['B'], report['folds'][f'{REPRODUCTION_CLASS}/B']
    if len(reference) != len(mine):
        raise ValueError('outer fold count differs from the admitted report')
    for left, right in zip(reference, mine):
        if sorted(left['held_families']) != sorted(right['held_families']):
            raise ValueError(f"outer fold {left['fold']} membership differs from the admitted report")
        if ([sorted(f['validation_families']) for f in left['inner_folds']]
                != [sorted(f['validation_families']) for f in right['inner_folds']]):
            raise ValueError(f"inner folds of outer fold {left['fold']} differ from the admitted report")
    admitted_rows = {r['assay']: r for r in admitted['assays']}
    deviations = {metric: max(abs(r[metric] - admitted_rows[r['assay']][metric])
                              for r in report['assays'])
                  for metric in ('raw_M_spearman', 'raw_P_spearman')}
    admitted_predictions = {r['assay']: r for r in admitted['predictions']}
    order = [r['assay'] for r in report['assays']]
    if order != report['prediction_assay_order']:
        raise ValueError('prediction row order is inconsistent')
    offsets, start = {}, 0
    for row in report['assays']:
        offsets[row['assay']] = (start, start + row['n_variants'])
        start += row['n_variants']
    for design in ('B', 'R', 'B_R', 'permuted_B', 'permuted_B_R'):
        label = f'{REPRODUCTION_CLASS}/{design}'
        if label not in report['predictions'] or design not in admitted_predictions[order[0]]:
            continue
        vector = np.asarray(report['predictions'][label])
        worst = 0.0
        for assay, (low, high) in offsets.items():
            expected = np.asarray(admitted_predictions[assay][design], dtype=np.float64)
            if len(expected) != high - low:
                raise ValueError(f'{assay}: admitted prediction length differs')
            worst = max(worst, float(np.max(np.abs(vector[low:high] - expected))))
        deviations[f'{design}_prediction'] = worst
    admitted_alphas = [f['alpha'] for f in admitted['folds']['B']]
    if admitted_alphas != [f['alpha'] for f in mine]:
        raise ValueError('the reproduced baseline selected different ridge penalties')
    return dict(support_assays=len(provenance['assay_ids']), outer_folds=len(mine),
                inner_folds_per_outer=len(mine[0]['inner_folds']),
                admitted_alpha_grid=admitted['alpha_grid'],
                reproduced_selected_alphas=admitted_alphas,
                max_absolute_deviation=deviations)
