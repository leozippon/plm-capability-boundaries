"""Depth-resolved frozen readout over every transformer block of an admitted arm.

The admitted readout panel hooked exactly two blocks per arm -- the block after
half the stack and the last block -- and retained the target-residue-token mean
and the last residue-token state at each. Its panel-wide near-null is therefore
bounded by two depths and two pooling rules. This module holds the
representation family fixed in every other respect and resolves depth: one
forward pass per sequence captures every block's output under the same two
pooling rules, and, where a rendering assigns exactly one token per residue so
that a wild type and a mutant occupy the same token grid, also two
position-resolved summaries localized on the substituted residues.

Everything the admitted study fixed is imported from its own modules rather
than restated: the cohort, the assay supports, the wild-type-family folds and
their seeds, the within-assay standardized-rank targets, the equal
cluster/assay/variant weights, the training-only weighted feature scaling, the
ridge objective, penalty grid and tie rule, and the 2,000-draw paired cluster
bootstrap at alpha 0.05 over the unit *wild-type family at 50% identity*.

A resolved positive increment at some depth states that a compressed linear
function of that depth's summaries carries mutation-effect information the
matched supervised baseline does not; it does not locate where that information
came from. An unresolved or negative increment states that no function this
class reached at that depth carries such information, and is bound by the
compression, the pooling and position rules, the label budget and the cohort.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.stats import rankdata
import torch

from ..core.amino_acids import AA20
from ..context.profile_increment import correlation, standardized_rank, summarize
from .readout_analysis import ALPHAS, nested_predict, sequence_features
from .readout_class_sweep import projected_block
from .readout_extraction import (pool_hidden, representation_blocks, representation_positions,
                                 pack_sequence, forward_readout_rows)

#: The two admitted pooling rules, retained at every block.
POOLED_SUMMARIES = ('mean', 'last')
#: The two position-resolved summaries, retained only where a wild type and each
#: of its mutants render to the same one-token-per-residue grid. ``mut`` is the
#: mean mutant-minus-wild-type state over the substituted residues' own tokens;
#: ``suffix`` is that mean over every residue token after the last substituted
#: one. The complementary prefix difference is provably zero for a causal model
#: and is retained as a per-variant maximum-absolute check rather than a feature.
POSITION_SUMMARIES = ('mut', 'suffix')
SUMMARY_NAMES = POOLED_SUMMARIES + POSITION_SUMMARIES
#: The admitted four-block feature order, reproduced from the two admitted depths.
ADMITTED_FEATURE_NAMES = ('middle_mean', 'middle_last', 'final_mean', 'final_last')
ADMITTED_PROJECTION_SEED = 20260923
ADMITTED_PROJECTION_DIM = 256
#: Declared projection seeds and widths of this experiment's own designs. Each
#: depends only on the depth index, the summary index and the declared width;
#: never on a label, an observed vector or a fitted outcome.
DEPTH_PROJECTION_SEED = 20261123
DEPTH_PROJECTION_DIM = 256
WIDE_PROJECTION_SEED = 20271123
WIDE_PROJECTION_DIM = 512
UNION_PROJECTION_SEED = 20281123
UNION_PROJECTION_DIM = 32
#: The admitted resampling contract and split seeds, unchanged.
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260923
PRIMARY_SPLIT_SEED = 20260923
SPLIT_SEEDS = (20260923, 20260924, 20260925)
PERMUTATION_OFFSET = 9000
#: Schema tag of the per-assay depth artifacts and of a cell report.
EXTRACTION_SCHEMA = 'frozen_readout_depth_v1'
ANALYSIS_SCHEMA = 'd1_readout_depth_v1'


def admitted_block_indices(depth: int) -> tuple[int, int]:
    """The two zero-based blocks the admitted extraction hooked."""
    if depth < 1:
        raise ValueError('a transformer stack has at least one block')
    return ((depth - 1) // 2, depth - 1)


def mutated_residue_indices(wildtype: str, mutant: str) -> list[int]:
    """Zero-based residue indices a mutation identifier substitutes.

    Validated against the wild type with the same rule the admitted sequence
    descriptors use, so an inconsistent identifier is refused rather than
    silently producing a position-resolved feature at the wrong residue.
    """
    positions: list[int] = []
    for mutation in mutant.split(':'):
        match = re.fullmatch(r'([A-Z])(\d+)([A-Z])', mutation)
        if match is None:
            raise ValueError(f'invalid substitution {mutation}')
        before, position, after = match.group(1), int(match.group(2)) - 1, match.group(3)
        if (before not in AA20 or after not in AA20 or not 0 <= position < len(wildtype)
                or wildtype[position] != before or position in positions):
            raise ValueError(f'inconsistent substitution {mutation}')
        positions.append(position)
    return sorted(positions)


def verify_mutant_sequence(wildtype: str, mutant: str, sequence: str) -> list[int]:
    """Check the cohort's stored mutant against its identifier and return its residues.

    The cohort carries both the mutation identifier and the rendered mutant
    sequence. A position-resolved difference is meaningful only if the two agree
    exactly, so this refuses any row where the stored sequence differs from the
    wild type anywhere other than the substituted residues.
    """
    residues = mutated_residue_indices(wildtype, mutant)
    if len(sequence) != len(wildtype):
        raise ValueError(f'{mutant}: stored mutant length differs from the wild type')
    differing = [i for i, (a, b) in enumerate(zip(wildtype, sequence)) if a != b]
    if differing != residues:
        raise ValueError(f'{mutant}: stored mutant differs at {differing}, identifier says {residues}')
    return residues


def per_residue_grid(arm, sequences, packed) -> bool:
    """Does every row place exactly one residue-bearing token per residue?

    This is a property of the tokenizer and the declared rendering alone. It is
    the condition under which a wild type and a mutant of equal length occupy
    the same token grid, so a difference of states at one grid position compares
    the same residue index in both. Where it fails -- byte-pair renderings whose
    tokens carry several residues, and ProtGPT2's FASTA wrapping -- no
    position-resolved difference is defined and none is substituted.
    """
    spans = [row[2] for row in packed]
    return (representation_positions(arm, sequences, packed) is None
            and all(right - left == len(sequence)
                    for sequence, (left, right) in zip(sequences, spans, strict=True)))


def position_summaries(delta: torch.Tensor, residues: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
    """The two position-resolved summaries and the causal prefix control.

    ``delta`` is one row's mutant-minus-wild-type state at every residue-bearing
    token of a shared one-token-per-residue grid. ``mut`` is the mean over the
    substituted residues' own tokens and ``suffix`` the mean over every token
    after the last substituted one, which is the zero vector when the last
    substitution is at the final residue. The prefix control is the maximum
    absolute difference before the first substitution, which a causal model
    leaves at exactly zero because no earlier position attends to a later one.
    """
    if delta.ndim != 2:
        raise ValueError('position summaries need one row of residue-token states')
    if not residues or residues != sorted(set(residues)) or not 0 <= residues[0] <= residues[-1] < delta.shape[0]:
        raise ValueError('substituted residue indices are unordered or outside the target')
    width = delta.shape[1]
    mut = delta[residues].mean(0).cpu().numpy()
    tail = residues[-1] + 1
    suffix = (delta[tail:].mean(0).cpu().numpy() if tail < delta.shape[0]
              else np.zeros(width, np.float32))
    prefix = float(delta[:residues[0]].abs().max()) if residues[0] > 0 else 0.0
    return mut, suffix, prefix


@dataclass
class AssayExtraction:
    """One assay's depth-resolved summaries, as mutant-minus-wild-type deltas."""

    pooled: np.ndarray                 # (n_variants, depth, 2, width)
    wt_pooled: np.ndarray              # (depth, 2, width)
    likelihood: np.ndarray             # (n_variants,)
    wt_likelihood: float
    position: np.ndarray | None        # (n_variants, depth, 2, width)
    prefix_abs_max: np.ndarray | None  # (n_variants, depth)
    empty_suffix: np.ndarray | None    # (n_variants,)
    n_mutated: np.ndarray | None       # (n_variants,)
    drift: dict


def _forward_depth_batch(arm, sequences, stage46, *, position_state, mutated_rows):
    """Pooled summaries at every block, the native likelihood, and position deltas.

    One forward pass per batch. ``position_state`` is a mutable list, one entry
    per block, that receives the wild-type row's residue-token states the first
    time this is called for an assay; ``None`` disables position resolution.
    """
    packed = [pack_sequence(arm, sequence) for sequence in sequences]
    blocks = representation_blocks(arm)
    positions = representation_positions(arm, sequences, packed)
    spans = [row[2] for row in packed]
    time_first = arm.name == 'proteinglm-7b-clm'
    pooled: dict[int, np.ndarray] = {}
    mut: dict[int, np.ndarray] = {}
    suffix: dict[int, np.ndarray] = {}
    prefix: dict[int, np.ndarray] = {}
    if position_state is not None and len({tuple(span) for span in spans}) != 1:
        raise ValueError('position resolution requires one shared residue-token span per assay')

    def capture(index):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if time_first:
                hidden = hidden.transpose(0, 1)
            pooled[index] = pool_hidden(hidden, spans, positions=positions)
            if position_state is None:
                return
            left, right = spans[0]
            states = hidden[:, left:right].float()
            if position_state[index] is None:
                position_state[index] = states[0].clone()
            delta = states - position_state[index]
            width = delta.shape[2]
            mut[index] = np.zeros((len(sequences), width), np.float32)
            suffix[index] = np.zeros((len(sequences), width), np.float32)
            prefix[index] = np.zeros(len(sequences), np.float32)
            for row, residues in enumerate(mutated_rows):
                if residues is None:
                    continue
                values = position_summaries(delta[row], residues)
                mut[index][row], suffix[index][row], prefix[index][row] = values
        return hook

    handles = [blocks[index].register_forward_hook(capture(index)) for index in range(len(blocks))]
    try:
        logits, ids = forward_readout_rows(arm, [row[0] for row in packed], stage46)
        likelihood = np.array([-stage46._target_nll(logits[i:i + 1], ids[i:i + 1], *row[1])['nll_sum']
                               for i, row in enumerate(packed)])
    finally:
        for handle in handles:
            handle.remove()
    if set(pooled) != set(range(len(blocks))) or not np.isfinite(likelihood).all():
        raise ValueError('incomplete or nonfinite forward')
    stacked = np.stack([pooled[index] for index in range(len(blocks))], axis=1)
    if position_state is None:
        return stacked, likelihood, None, None
    position = np.stack([np.stack([mut[index], suffix[index]], axis=1)
                         for index in range(len(blocks))], axis=1)
    prefix_max = np.stack([prefix[index] for index in range(len(blocks))], axis=1)
    return stacked, likelihood, position, prefix_max


def extract_assay(arm, stage46, wildtype: str, sequences: list[str], mutants: list[str], *,
                  batch_size: int, position_resolved: bool, max_score_drift: float,
                  max_feature_drift: float) -> AssayExtraction:
    """Every block's pooled summaries for one assay's wild type and variants.

    The wild type is row zero of the first batch, exactly as the admitted
    extraction composes its batches, so a batched arm's pooled summaries are
    computed from the same batch composition as the admitted artifacts.
    """
    from .readout_extraction import mutation_relative_drift

    strings = [wildtype] + list(sequences)
    residues: list[list[int] | None] = [None]
    for mutant, sequence in zip(mutants, sequences, strict=True):
        residues.append(verify_mutant_sequence(wildtype, mutant, sequence)
                        if position_resolved else [])
    blocks = len(representation_blocks(arm))
    position_state: list | None = [None] * blocks if position_resolved else None
    pooled_parts, likelihood_parts, position_parts, prefix_parts = [], [], [], []
    for start in range(0, len(strings), batch_size):
        stop = start + batch_size
        pooled, likelihood, position, prefix = _forward_depth_batch(
            arm, strings[start:stop], stage46, position_state=position_state,
            mutated_rows=residues[start:stop])
        pooled_parts.append(pooled)
        likelihood_parts.append(likelihood)
        if position_resolved:
            position_parts.append(position)
            prefix_parts.append(prefix)
    pooled = np.concatenate(pooled_parts)
    likelihood = np.concatenate(likelihood_parts)

    # An independent single-row repeat of the first three rows. At batch size one
    # this reference forward and the production forward are the same single-row
    # computation, so an exactly-zero drift is true by construction and certifies
    # nothing; the informative case is a batched arm.
    check_pooled, check_likelihood = [], []
    for sequence in strings[:3]:
        one_pooled, one_likelihood, _, _ = _forward_depth_batch(
            arm, [sequence], stage46, position_state=None, mutated_rows=[None])
        check_pooled.append(one_pooled[0])
        check_likelihood.append(one_likelihood[0])
    check_pooled = np.stack(check_pooled)
    check_likelihood = np.asarray(check_likelihood)
    flat = pooled[:3].reshape(min(3, len(pooled)), -1)
    flat_check = check_pooled.reshape(len(check_pooled), -1)
    delta_error = float(np.max(np.abs((likelihood[1:3] - likelihood[0])
                                      - (check_likelihood[1:] - check_likelihood[0]))))
    state_error = float(np.linalg.norm(flat - flat_check) / max(np.linalg.norm(flat_check), 1e-12))
    mutation_error = mutation_relative_drift(flat, flat_check)
    if delta_error > max_score_drift or mutation_error > max_feature_drift:
        raise RuntimeError(f'declared precision limit exceeded: delta_M_nats={delta_error}, '
                           f'mutation_relative_l2={mutation_error}')

    position = prefix_max = empty_suffix = n_mutated = None
    if position_resolved:
        position = np.concatenate(position_parts)[1:]
        prefix_max = np.concatenate(prefix_parts)[1:]
        empty_suffix = np.asarray([row[-1] + 1 >= len(wildtype) for row in residues[1:]])
        n_mutated = np.asarray([len(row) for row in residues[1:]], np.int32)
    return AssayExtraction(
        pooled=(pooled[1:] - pooled[0]).astype(np.float32), wt_pooled=pooled[0].astype(np.float32),
        likelihood=likelihood[1:] - likelihood[0], wt_likelihood=float(likelihood[0]),
        position=None if position is None else position.astype(np.float32),
        prefix_abs_max=None if prefix_max is None else prefix_max.astype(np.float32),
        empty_suffix=empty_suffix, n_mutated=n_mutated,
        drift=dict(likelihood_delta_nats=delta_error, state_relative_l2=state_error,
                   mutation_relative_l2=mutation_error))


def assay_arrays(extraction: AssayExtraction, *, mutants, measured, profile_scores,
                 metadata: str) -> dict[str, np.ndarray]:
    """One assay's NPZ payload, one array per depth so a fit can stream depths."""
    payload: dict[str, np.ndarray] = dict(
        likelihood=extraction.likelihood, wt_likelihood=np.asarray(extraction.wt_likelihood),
        mutants=np.asarray(mutants), measured=np.asarray(measured),
        profile_scores=np.asarray(profile_scores), metadata=np.asarray(metadata),
        batch_check_likelihood_delta_nats=np.asarray(extraction.drift['likelihood_delta_nats']),
        batch_check_feature_relative_l2=np.asarray(extraction.drift['state_relative_l2']),
        batch_check_mutation_feature_relative_l2=np.asarray(extraction.drift['mutation_relative_l2']))
    for index in range(extraction.pooled.shape[1]):
        payload[f'pooled_d{index:03d}'] = extraction.pooled[:, index]
        payload[f'wt_pooled_d{index:03d}'] = extraction.wt_pooled[index]
        if extraction.position is not None:
            payload[f'position_d{index:03d}'] = extraction.position[:, index]
    if extraction.position is not None:
        payload['prefix_abs_max'] = extraction.prefix_abs_max
        payload['empty_suffix'] = extraction.empty_suffix
        payload['n_mutated'] = extraction.n_mutated
    return payload


# ------------------------------------------------------------------- projections


def gaussian_projection(seed: int, width: int, dim: int) -> np.ndarray:
    """The admitted projection form: Gaussian entries of standard deviation 1/sqrt(dim)."""
    return (np.random.default_rng(seed)
            .normal(0, 1 / np.sqrt(dim), size=(width, dim)).astype(np.float32))


def admitted_projection(width: int) -> list[np.ndarray]:
    """The admitted four-block projection, in the admitted block order and seeds."""
    return [gaussian_projection(ADMITTED_PROJECTION_SEED + i, width, ADMITTED_PROJECTION_DIM)
            for i in range(4)]


# --------------------------------------------------------------- panel loading


class DepthPanel:
    """Verified handles onto one arm's depth-resolved extraction on one support.

    Every bound identity the admitted analysis checks is checked here against the
    same cohort: the manifest's completion and arm, the cohort digest, each
    assay's cluster, wild-type identity and mutation digest, each NPZ checksum
    and metadata identity, the exact ordered mutation identifiers, and the
    measured effects and profile scores. A mismatch raises.
    """

    def __init__(self, cohort_path, manifest_path, arm: str, assay_ids):
        cohort_path, manifest_path = Path(cohort_path), Path(manifest_path)
        self.hashes: dict[str, str] = {}
        cohort = self._read(cohort_path)
        manifest = self._read(manifest_path)
        if manifest['status'] != 'complete':
            raise ValueError(f'incomplete extraction: {manifest_path}')
        identity = manifest['identity']
        if identity['arm'] != arm:
            raise ValueError('manifest arm mismatch')
        if identity.get('schema_version') != EXTRACTION_SCHEMA:
            raise ValueError('manifest is not a depth-resolved extraction')
        if list(identity['summary_names']) != list(POOLED_SUMMARIES):
            raise ValueError('pooled summary order mismatch')
        if identity['cohort_sha256'] != self.hashes[str(cohort_path)]:
            raise ValueError('cohort hash mismatch')
        cohort_rows = {row['assay']: row for row in cohort['assays']}
        if len(cohort_rows) != len(cohort['assays']):
            raise ValueError('duplicate cohort assays')
        records = {row['assay']: row for row in manifest['assays']}
        if len(records) != len(manifest['assays']):
            raise ValueError('duplicate extraction assays')
        requested = sorted(assay_ids)
        if len(set(requested)) != len(requested):
            raise ValueError('duplicate requested assays')
        missing = [assay for assay in requested if assay not in records]
        if missing:
            raise ValueError(f'{arm} does not cover {len(missing)} requested assays, '
                             f'e.g. {missing[:3]}')
        self.arm = arm
        self.identity = identity
        self.depth = int(manifest['block_count'])
        self.position_resolved = bool(identity['position_resolved'])
        self.assay_ids = requested
        self.handles = []
        self.rows = []
        width = None
        for assay in requested:
            cohort_row, record = cohort_rows[assay], records[assay]
            for key in ('cluster', 'wildtype_id', 'mutant_digest'):
                if record[key] != cohort_row[key]:
                    raise ValueError(f'{assay}: manifest {key} mismatch')
            if record['variants'] != len(cohort_row['mutants']):
                raise ValueError(f'{assay}: variant count mismatch')
            path = manifest_path.parent / record['file']
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != record['sha256']:
                raise ValueError(f'{assay}: extraction checksum mismatch')
            self.hashes[str(path)] = digest
            data = np.load(path, allow_pickle=False)
            expected = dict(identity=identity, assay=assay,
                            mutant_digest=cohort_row['mutant_digest'])
            if json.loads(str(data['metadata'])) != expected:
                raise ValueError(f'{assay}: extraction metadata mismatch')
            if data['mutants'].tolist() != cohort_row['mutants']:
                raise ValueError(f'{assay}: exact mutation order mismatch')
            for key in ('measured', 'profile_scores'):
                if not np.array_equal(data[key], np.asarray(cohort_row[key])):
                    raise ValueError(f'{assay}: {key} mismatch')
            shape = data[f'pooled_d{self.depth - 1:03d}'].shape
            if shape[:2] != (len(cohort_row['mutants']), len(POOLED_SUMMARIES)):
                raise ValueError(f'{assay}: unexpected pooled block shape {shape}')
            if width is None:
                width = int(shape[2])
            if int(shape[2]) != width:
                raise ValueError('hidden width changed within arm')
            self.handles.append(data)
            self.rows.append(dict(assay=assay, cluster=cohort_row['cluster'],
                                  mutants=list(cohort_row['mutants']),
                                  measured=np.asarray(data['measured'], float),
                                  P=np.asarray(data['profile_scores'], float),
                                  M=np.asarray(data['likelihood'], float),
                                  S=sequence_features(cohort_row['wildtype'],
                                                      cohort_row['mutants'])))
        self.width = width
        self.n_variants = int(sum(len(row['mutants']) for row in self.rows))
        self.clusters = sorted({row['cluster'] for row in self.rows})

    def _read(self, path):
        raw = Path(path).read_bytes()
        self.hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def close(self):
        for handle in self.handles:
            handle.close()
        self.handles = []

    def block(self, depth_index: int, summary: str) -> np.ndarray:
        """One (depth, summary) block over the concatenated variant rows."""
        if not 0 <= depth_index < self.depth:
            raise ValueError(f'depth {depth_index} is outside 0..{self.depth - 1}')
        if summary in POOLED_SUMMARIES:
            key, column = f'pooled_d{depth_index:03d}', POOLED_SUMMARIES.index(summary)
        elif summary in POSITION_SUMMARIES:
            if not self.position_resolved:
                raise ValueError(f'{self.arm} retains no position-resolved summaries')
            key, column = f'position_d{depth_index:03d}', POSITION_SUMMARIES.index(summary)
        else:
            raise ValueError(f'unknown summary {summary}')
        return np.concatenate([handle[key][:, column] for handle in self.handles])

    def prefix_check(self) -> dict:
        """The causal prefix control: the wild-type difference before the first
        substituted residue must be exactly zero at every depth."""
        if not self.position_resolved:
            return dict(available=False)
        worst = 0.0
        rows = 0
        for handle in self.handles:
            values = handle['prefix_abs_max']
            worst = max(worst, float(np.max(values)) if values.size else 0.0)
            rows += int(values.shape[0])
        empty = int(sum(int(handle['empty_suffix'].sum()) for handle in self.handles))
        return dict(available=True, max_absolute_prefix_difference=worst,
                    variants=rows, empty_suffix_variants=empty)


def load_admitted_arrays(cohort_path, manifest_path, arm: str, assay_ids):
    """The admitted four-block arrays on one declared support, identities checked.

    This reads the admitted extraction the readout panel was fitted from, so the
    same fitting code can be run against the admitted states and against the
    depth-resolved re-extraction. Every bound identity the admitted analysis
    checks is checked here against the same cohort.
    """
    cohort_path, manifest_path = Path(cohort_path), Path(manifest_path)
    hashes: dict[str, str] = {}

    def read(path):
        raw = Path(path).read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    cohort = read(cohort_path)
    manifest = read(manifest_path)
    if manifest['status'] != 'complete' or manifest['identity']['arm'] != arm:
        raise ValueError('incomplete or mismatched admitted manifest')
    if list(manifest['identity']['feature_names']) != list(ADMITTED_FEATURE_NAMES):
        raise ValueError('admitted representation block order mismatch')
    if manifest['identity']['cohort_sha256'] != hashes[str(cohort_path)]:
        raise ValueError('cohort hash mismatch')
    cohort_rows = {row['assay']: row for row in cohort['assays']}
    records = {row['assay']: row for row in manifest['assays']}
    rows, blocks = [], []
    for assay in sorted(assay_ids):
        cohort_row, record = cohort_rows[assay], records[assay]
        for key in ('cluster', 'wildtype_id', 'mutant_digest'):
            if record[key] != cohort_row[key]:
                raise ValueError(f'{assay}: admitted manifest {key} mismatch')
        path = manifest_path.parent / record['file']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record['sha256']:
            raise ValueError(f'{assay}: admitted extraction checksum mismatch')
        hashes[str(path)] = digest
        with np.load(path, allow_pickle=False) as data:
            expected = dict(identity=manifest['identity'], assay=assay,
                            mutant_digest=cohort_row['mutant_digest'])
            if json.loads(str(data['metadata'])) != expected:
                raise ValueError(f'{assay}: admitted extraction metadata mismatch')
            if data['mutants'].tolist() != cohort_row['mutants']:
                raise ValueError(f'{assay}: admitted mutation order mismatch')
            for key in ('measured', 'profile_scores'):
                if not np.array_equal(data[key], np.asarray(cohort_row[key])):
                    raise ValueError(f'{assay}: admitted {key} mismatch')
            hidden = np.asarray(data['features'], dtype=np.float32)
            if hidden.ndim != 3 or hidden.shape[:2] != (len(cohort_row['mutants']), 4):
                raise ValueError(f'{assay}: expected four aligned admitted feature blocks')
            blocks.append(hidden)
            rows.append(dict(assay=assay, cluster=cohort_row['cluster'],
                             mutants=list(cohort_row['mutants']),
                             measured=np.asarray(data['measured'], float),
                             P=np.asarray(data['profile_scores'], float),
                             M=np.asarray(data['likelihood'], float),
                             S=sequence_features(cohort_row['wildtype'], cohort_row['mutants'])))
    return rows, np.concatenate(blocks), hashes, manifest['identity']


def admitted_design(blocks: np.ndarray, variant_counts) -> np.ndarray:
    """The admitted 1,024-coordinate representation design from four-block arrays.

    The admitted analysis forms this float32 product one assay at a time, and a
    float32 matrix product is not invariant to how its rows are blocked. On the
    212-assay native Qwen2.5-32B panel, 27,071 rows at hidden width 5,120, forming
    the product over all rows at once departs from the per-assay product by up to
    7.629e-06 in absolute value against a largest absolute entry of 723.245 -- at
    every BLAS thread count of 1, 4 and 16 -- and that reached 3.483e-08 in the
    held-out predictions, above the 1e-8 the pipeline-identity control is bound at.
    The rows are therefore blocked by assay exactly as the admitted analysis blocks
    them, through the admitted projection contract in
    :func:`readout_class_sweep.projected_block`, which also pins the BLAS reduction
    order. The per-assay product is measured identical at 1, 4 and 16 threads.
    """
    if blocks.ndim != 3 or blocks.shape[1] != 4:
        raise ValueError('admitted design requires four aligned blocks')
    counts = [int(count) for count in variant_counts]
    if sum(counts) != len(blocks):
        raise ValueError(f'{sum(counts)} assay rows do not cover {len(blocks)} block rows')
    projection = admitted_projection(int(blocks.shape[2]))
    pieces, start = [], 0
    for count in counts:
        pieces.append(projected_block(blocks[start:start + count], projection))
        start += count
    return np.concatenate(pieces, axis=0)


def admitted_blocks_from_depth(panel: 'DepthPanel') -> np.ndarray:
    """Assemble the admitted four blocks, in the admitted order, from every depth."""
    middle, final = admitted_block_indices(panel.depth)
    return np.stack([panel.block(middle, 'mean'), panel.block(middle, 'last'),
                     panel.block(final, 'mean'), panel.block(final, 'last')], axis=1)


def block_relative_drift(observed: np.ndarray, reference: np.ndarray) -> dict:
    """Agreement of two four-block mutation-vector sets, in the admitted gate's form."""
    if observed.shape != reference.shape:
        raise ValueError('mutation-vector sets have different shapes')
    flat_observed = observed.reshape(len(observed), -1)
    flat_reference = reference.reshape(len(reference), -1)
    norms = np.linalg.norm(flat_reference, axis=1)
    difference = np.linalg.norm(flat_observed - flat_reference, axis=1)
    zero = norms == 0
    if np.any(difference[zero] != 0):
        raise ValueError('nonzero difference against a zero reference mutation vector')
    relative = np.zeros_like(difference)
    relative[~zero] = difference[~zero] / norms[~zero]
    return dict(max_relative_l2=float(relative.max()),
                mean_relative_l2=float(relative.mean()),
                max_absolute_difference=float(np.max(np.abs(flat_observed - flat_reference))),
                exactly_equal=bool(np.array_equal(flat_observed, flat_reference)),
                variants=int(len(observed)),
                zero_reference_variants=int(zero.sum()))


# ------------------------------------------------------------------- fitting


def baseline_design(rows: list[dict]) -> np.ndarray:
    """The admitted 446-feature matched baseline: rank(P), rank(M) and 444 descriptors."""
    return np.column_stack([
        np.concatenate([standardized_rank(row['P']) for row in rows]),
        np.concatenate([standardized_rank(row['M']) for row in rows]),
        np.concatenate([row['S'] for row in rows])])


def row_labels(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    measured = np.concatenate([row['measured'] for row in rows])
    assays = np.concatenate([[row['assay']] * len(row['mutants']) for row in rows])
    clusters = np.concatenate([[row['cluster']] * len(row['mutants']) for row in rows])
    return measured, assays, clusters


def permuted_labels(measured: np.ndarray, assays: np.ndarray, seed: int) -> np.ndarray:
    """The admitted fixed within-assay permutation, used only for fitting and tuning."""
    rng = np.random.default_rng(seed + PERMUTATION_OFFSET)
    shuffled = measured.copy()
    for assay in np.unique(assays):
        index = np.flatnonzero(assays == assay)
        shuffled[index] = rng.permutation(measured[index])
    return shuffled


def depth_blocks(panel: DepthPanel, depth_index: int) -> dict[str, np.ndarray]:
    """Every retained summary at one depth, read once so a cell streams depths."""
    names = POOLED_SUMMARIES + (POSITION_SUMMARIES if panel.position_resolved else ())
    return {name: panel.block(depth_index, name) for name in names}


def project_summaries(blocks: dict[str, np.ndarray], names, *, width: int, depth_index: int,
                      dim: int, base_seed: int) -> np.ndarray:
    """Concatenate the named summaries of one depth under the declared projection rule.

    The seed of each block is the declared base seed plus eight times the depth
    index plus the summary's index in the declared summary order, so it depends
    only on the declaration and never on a label, an observed vector or a fitted
    outcome.
    """
    return np.concatenate(
        [blocks[name] @ gaussian_projection(
            base_seed + 8 * depth_index + SUMMARY_NAMES.index(name), width, dim)
         for name in names], axis=1)


def fit_design(design: np.ndarray, targets: np.ndarray, assays, clusters, *, fold_seed: int,
               device: str) -> tuple[np.ndarray, list[dict]]:
    """Held-cluster predictions from the admitted nested fitting procedure, unchanged."""
    return nested_predict(design, targets, assays, clusters, seed=fold_seed,
                          alphas=ALPHAS, device=device)


def per_assay_metrics(rows: list[dict], assays: np.ndarray, measured: np.ndarray,
                      predictions: dict[str, np.ndarray], contrasts) -> list[dict]:
    """Within-assay Spearman, rank MSE and the declared paired contrasts."""
    records = []
    for row in rows:
        index = np.flatnonzero(assays == row['assay'])
        target = standardized_rank(measured[index])
        record = dict(assay=row['assay'], cluster=int(row['cluster']), n_variants=len(index))
        for name, vector in predictions.items():
            record[f'{name}_spearman'] = correlation(rankdata(vector[index]), target)
            if not name.startswith('raw_'):
                record[f'{name}_rank_mse'] = float(np.mean((target - vector[index]) ** 2))
        for label, (representation, baseline) in contrasts.items():
            high, low = record[f'{representation}_spearman'], record[f'{baseline}_spearman']
            record[f'{label}_delta_spearman'] = None if high is None or low is None else high - low
            record[f'{label}_delta_rank_mse'] = (record[f'{baseline}_rank_mse']
                                                 - record[f'{representation}_rank_mse'])
        for name in list(predictions):
            if name.startswith('raw_'):
                continue
            left, right = record[f'{name}_spearman'], record['raw_M_spearman']
            record[f'{name}_minus_raw_M_spearman'] = (None if left is None or right is None
                                                      else left - right)
        records.append(record)
    return records


def summarise_metrics(records: list[dict], *, bootstrap: int = BOOTSTRAP_DRAWS,
                      seed: int = BOOTSTRAP_SEED) -> tuple[dict, list[int] | None]:
    """Every metric's cluster-equal point estimate and paired bootstrap interval.

    The resampled cluster sizes are identical across metrics defined on the whole
    support, so they are returned once rather than repeated in every record: a
    cell fits on the order of a thousand designs and the repetition would
    dominate its report.
    """
    keys = [key for key in records[0] if key not in ('assay', 'cluster', 'n_variants')]
    summaries, sizes = {}, None
    for key in keys:
        record = summarize(records, key, bootstrap=bootstrap, seed=seed)
        observed = record.pop('cluster_sizes', None)
        if observed is not None and record['n_assays'] == len(records):
            sizes = observed
        summaries[key] = record
    return summaries, sizes


def verify_against_admitted_report(admitted: dict, support_ids, rows: list[dict],
                                   records: list[dict], predictions: dict[str, np.ndarray],
                                   folds: dict[str, list[dict]], design_map: dict[str, str]) -> dict:
    """Bind a refit to the admitted report: same support, same folds, same numbers.

    Every check is an equality or a maximum absolute deviation over quantities
    the admitted artifact already carries. A structural mismatch raises rather
    than being recorded as a tolerated difference; the numerical deviations are
    returned so the caller can gate on them.
    """
    if list(admitted['support']['assay_ids']) != list(support_ids):
        raise ValueError('support assay identifiers differ from the admitted report')
    for key, value in (('n_assays', len(rows)),
                       ('n_families', len({row['cluster'] for row in rows})),
                       ('n_variants', int(sum(len(row['mutants']) for row in rows)))):
        if admitted[key] != value:
            raise ValueError(f'{key} differs from the admitted report')
    deviations: dict[str, float] = {}
    admitted_rows = {row['assay']: row for row in admitted['assays']}
    for metric in ('raw_M_spearman', 'raw_P_spearman'):
        deviations[metric] = max(abs(row[metric] - admitted_rows[row['assay']][metric])
                                 for row in records)
    admitted_predictions = {row['assay']: row for row in admitted['predictions']}
    offsets, start = {}, 0
    for row in records:
        offsets[row['assay']] = (start, start + row['n_variants'])
        start += row['n_variants']
    for admitted_name, label in design_map.items():
        if label not in folds:
            continue
        reference, mine = admitted['folds'][admitted_name], folds[label]
        if len(reference) != len(mine):
            raise ValueError(f'{admitted_name}: outer fold count differs from the admitted report')
        for left, right in zip(reference, mine):
            if sorted(left['held_families']) != sorted(right['held_families']):
                raise ValueError(f"{admitted_name}: outer fold {left['fold']} membership differs")
            if ([sorted(f['validation_families']) for f in left['inner_folds']]
                    != [sorted(f['validation_families']) for f in right['inner_folds']]):
                raise ValueError(f"{admitted_name}: inner folds of outer fold {left['fold']} differ")
        if [f['alpha'] for f in reference] != [f['alpha'] for f in mine]:
            raise ValueError(f'{admitted_name}: refit selected different ridge penalties')
        worst = 0.0
        vector = predictions[label]
        for assay, (low, high) in offsets.items():
            expected = np.asarray(admitted_predictions[assay][admitted_name], dtype=np.float64)
            if len(expected) != high - low:
                raise ValueError(f'{assay}: admitted prediction length differs')
            worst = max(worst, float(np.max(np.abs(vector[low:high] - expected))))
        deviations[f'{admitted_name}_prediction'] = worst
    return dict(admitted_report_designs=sorted(design_map),
                outer_folds=len(next(iter(folds.values()))),
                selected_alphas={name: [f['alpha'] for f in admitted['folds'][name]]
                                 for name in design_map},
                max_absolute_deviation=deviations)
