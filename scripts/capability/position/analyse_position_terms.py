#!/usr/bin/env python3
"""Check a position-term recomputation against its archive, then decompose it.

Three things happen here, in this order, and the third is not reported unless
the first two pass.

**T1, retention.** The per-token vector each cell retained must re-reduce to the
scalar the unmodified expression produced in the same forward. The extractor
already refuses to write a background whose worst residual is not identically
zero; this entry point reads the recorded worst residual back and reports it, so
the gate is visible in the analysis record rather than only in a run log.

**T2, alignment against the archive.** The recomputed per-state scalar is
compared to the archived one. Tier 1 is every state at exactly 0.0 nats. Tier 2
is a worst absolute difference at or below 1.0e-4 nats *and* at or below three
times that arm's own archived per-background repeat maximum, which for most arms
is 0.0 and therefore admits tier 2 only where the archive itself recorded
run-to-run movement. Tier 3 is anything larger: the forward changed, and the
cell's decomposition may not be attributed to the published score. The tiers are
fixed in :data:`TIER2_NATS` before any residual is seen and are not widened here.

**The decomposition.** Every scored token is assigned to the mutated residue's
own token, to the tokens upstream of it, or to the tokens downstream of it, on
the residue axis. The three signed class differences reconstruct the mutation
score exactly, and the record carries that closure residual alongside the
ranking quality of each part against the cohort's own endpoint, on the frozen
family-held groups and under the frozen group-bootstrap contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.core.io import write_json
from importlib import util as _util

from src.capability.interactions.pairwise_epistasis import (
    ARM_DTYPE, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, ROSTER, cycle_contrast, group_spearman,
    interval)
from src.capability.position.position_terms import (
    ALIGNMENT, PARTITION, alignment, mutation_parts, state_parts, retokenized_span_masks, masked_state_parts)
from src.capability.readouts.readout_extraction import load_readout_arm, pack_sequence
from src.capability.position.position_terms import ResidueCoverage
from src.capability.context.profile_increment import correlation, standardized_rank, summarize

SCHEMA = 'position_terms_analysis_v1'

#: Declared before any residual was read. Tier 2 additionally requires the arm's
#: own archived repeat maximum to admit it, so an arm whose archive recorded a
#: bit-identical repeat has no tier-2 room at all.
TIER2_NATS = 1.0e-4
TIER2_REPEAT_MULTIPLE = 3.0
#: Derived-quantity invariant: the published endpoint may not move at all at
#: tier 1, and may move by at most this dimensionless amount at tier 2.
DERIVED_TIER1 = 0.0
DERIVED_TIER2 = 1.0e-6
#: The four-state float64 identity the pairwise panel checks at 2.84e-14.
CYCLE_IDENTITY_NATS = 1.0e-12
#: Declared before any alignment rate was read: an arm carries the decomposition
#: contrast only if at least this fraction of its variants align, and only on the
#: aligned variants. Below it the arm is reported as not measurable for this
#: quantity rather than contributing a contrast over a biased remnant.
MEASURABLE_ALIGNED_FRACTION = 0.50

PART_KEYS = ('own', 'downstream', 'full')


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tier(worst: float, repeat_max: float) -> int:
    if worst == 0.0:
        return 1
    if worst <= TIER2_NATS and worst <= TIER2_REPEAT_MULTIPLE * float(repeat_max):
        return 2
    return 3


def substitution_sites(wildtype: str, mutant: str) -> list[int]:
    sites = []
    for part in mutant.split(':'):
        match = re.fullmatch(r'([A-Z])(\d+)([A-Z])', part)
        if match is None:
            raise ValueError(f'unparsable substitution {part!r}')
        before, position, _ = match.groups()
        index = int(position) - 1
        if not 0 <= index < len(wildtype) or wildtype[index] != before:
            raise ValueError(f'substitution {part!r} disagrees with its own wild type')
        sites.append(index)
    return sorted(set(sites))


class Retained:
    """One background's or assay's retained per-state vectors and coverage."""

    def __init__(self, data) -> None:
        for key in ('position_nats', 'position_offsets', 'position_residue_counts',
                    'position_residue_offsets', 'position_sum_check_nats'):
            if key not in data:
                raise ValueError(f'the recomputation archive lacks {key}; it was not retained')
        self.terms = np.asarray(data['position_nats'])
        self.offsets = np.asarray(data['position_offsets'])
        self.counts = np.asarray(data['position_residue_counts'])
        self.residue_offsets = np.asarray(data['position_residue_offsets'])
        self.sum_check = float(data['position_sum_check_nats'])
        if self.terms.dtype != np.float32:
            raise ValueError('retained position terms are not float32')
        if len(self.counts) != len(self.terms) or len(self.offsets) != len(self.residue_offsets) + 1:
            raise ValueError('retained position arrays are not aligned')
        if (self.terms.ndim != 1 or self.counts.ndim != 1 or self.offsets.ndim != 1
                or self.offsets[0] != 0 or self.offsets[-1] != len(self.terms)
                or np.any(np.diff(self.offsets) <= 0) or np.any(self.counts < 0)
                or np.any(self.residue_offsets < 0) or not np.isfinite(self.terms).all()):
            raise ValueError('invalid retained position values or ragged offsets')

    def parts(self, state: int, positions) -> dict[str, float]:
        low, high = int(self.offsets[state]), int(self.offsets[state + 1])
        return state_parts(self.terms[low:high], self.counts[low:high],
                           int(self.residue_offsets[state]), positions)

    def totals(self) -> np.ndarray:
        return np.asarray([float(np.asarray(self.terms[self.offsets[i]:self.offsets[i + 1]],
                                            dtype=np.float64).sum())
                           for i in range(len(self.residue_offsets))])


class Packer:
    """Re-derive one arm's packing at analysis time and check it against the archive.

    The alignment verdict needs the packed token sequence of both states, which
    the retained payload does not carry for every cohort. Re-deriving it through
    the same ``pack_sequence`` the extraction used is exact, and the per-token
    residue counts it produces are compared against the counts the extraction
    wrote, so a packing that has drifted between extraction and analysis fails
    here instead of moving a number.
    """

    def __init__(self, arm, *, partition_mode='strict') -> None:
        if partition_mode not in ('strict', 'span'):
            raise ValueError('unknown position partition mode')
        self.partition_mode = partition_mode
        self._arm = arm
        self._coverage = ResidueCoverage(arm)
        self._cache: dict[str, dict] = {}

    def state(self, sequence: str) -> dict:
        found = self._cache.get(sequence)
        if found is None:
            ids, span, _ = pack_sequence(self._arm, sequence)
            counts, offset = self._coverage.counts(ids, span, sequence)
            found = {'ids': [int(v) for v in ids], 'span': (int(span[0]), int(span[1])),
                     'counts': counts, 'offset': int(offset)}
            self._cache[sequence] = found
        return found

    def checked(self, sequence: str, retained, index: int) -> dict:
        state = self.state(sequence)
        low, high = int(retained.offsets[index]), int(retained.offsets[index + 1])
        if not np.array_equal(np.asarray(retained.counts[low:high]), state['counts']):
            raise ValueError('analysis-time packing disagrees with the retained residue counts')
        if int(retained.residue_offsets[index]) != state['offset']:
            raise ValueError('analysis-time packing disagrees with the retained residue offset')
        return state


def locate_archived_npz(archive: Path, recorded: str, index: dict[str, Path]) -> Path:
    """Resolve the npz an archived manifest recorded.

    A merged manifest stores either a bare filename in its own directory or a
    relative path into a sibling shard directory. Both are opened from that
    directory. A name that is not a file there is taken from the basename index
    built over the extra roots.
    """

    located = archive / recorded
    if located.is_file():
        return located
    found = index.get(Path(recorded).name)
    if found is None or not found.is_file():
        raise FileNotFoundError(
            f'{recorded} is not a file under {archive} and its name is absent '
            'from the archive index')
    return found


def index_files(roots: list[Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob('*.npz'):
            found.setdefault(path.name, path)
    return found


# --------------------------------------------------------------- per cohort


def load_stability(wave: Path, archive: Path, arm: str, source: Path, packer: Packer) -> dict:
    cohort = json.loads(source.read_bytes())
    manifest = json.loads((wave / f'manifest_{arm}.json').read_text())
    archived = json.loads((archive / f'manifest_{arm}.json').read_text())
    repeat_max = max(row['repeat_likelihood_nats'] for row in archived['backgrounds'])
    by_name = {row['background']: row for row in manifest['backgrounds']}
    archive_by_name = {row['background']: row for row in archived['backgrounds']}
    rows, t1, t2, closure, coverage = [], [], [], [], []
    for background in cohort['backgrounds']:
        name = background['name']
        new = np.load(wave / by_name[name]['file'], allow_pickle=False)
        old = np.load(archive / archive_by_name[name]['file'], allow_pickle=False)
        retained = Retained(new)
        t1.append(retained.sum_check)
        if not np.array_equal(new['variant_states'], old['variant_states']):
            raise ValueError(f'{name}: state indices differ from the archive')
        if not np.array_equal(new['variant_positions'], old['variant_positions']):
            raise ValueError(f'{name}: mutated positions differ from the archive')
        t2.append(np.abs(np.asarray(new['likelihood']) - np.asarray(old['likelihood'])))
        wild_state = packer.checked(background['sequences'][0], retained, 0)
        for variant in background['variants']:
            state, site = int(variant['state']), int(variant['position']) - 1
            mutant_state = packer.checked(background['sequences'][state], retained, state)
            fit = alignment(wild_state, mutant_state, [site])
            record = mutation_parts(retained.parts(0, [site]), retained.parts(state, [site]),
                                    float(new['likelihood'][state] - new['likelihood'][0]))
            archived_full = float(old['likelihood'][state] - old['likelihood'][0])
            rows.append({'group': background['group'], 'site': f'{name}:{site}',
                         'target': float(variant['ddg']),
                         'own': record['own'], 'downstream': record['downstream'],
                         'upstream': record['upstream'], 'full': record['full'],
                         'archived_full': archived_full,
                         'own_scored': record['own_scored'],
                         'own_residue_width': record['own_residue_width_mutant'],
                         'substitutions': 1, 'alignment': fit})
            closure.append((abs(record['closure_nats']), record['closure_bound_nats']))
            coverage.append({'own_scored': record['own_scored'],
                             'own_residue_width': record['own_residue_width_mutant'],
                             'token_count_changed': bool(
                                 int(new['scored_token_counts'][state])
                                 != int(new['scored_token_counts'][0]))})
        new.close(); old.close()
    return {'rows': rows, 't1': t1, 't2': np.concatenate(t2), 'closure': closure,
            'coverage': coverage, 'repeat_max': repeat_max,
            'wave_manifest': manifest, 'archive_manifest': archived,
            'unit': 'MegaScale family group', 'endpoint': 'ddG, kcal/mol'}


def keep_first_unexcluded(state: int, seen: set[int], excluded: bool) -> bool:
    """Keep one observation of a single-mutant state, from its first unexcluded cycle.

    The exclusion is decided before the state is marked seen. A state whose first
    cycle is an indel cycle must stay available to a later kept cycle; marking it
    seen first would drop that later measurement.
    """

    if excluded or state in seen:
        return False
    seen.add(state)
    return True


def load_pairwise(wave: Path, archive: Path, arm: str, source: Path,
                  plan: Path, excluded: Path | None, packer: Packer) -> dict:
    cohort = json.loads(source.read_bytes())
    planned = json.loads(plan.read_bytes())
    manifest = json.loads((wave / f'manifest_{arm}.json').read_text())
    archived = json.loads((archive / f'manifest_{arm}.json').read_text())
    repeat_max = max(row['repeat_likelihood_nats'] for row in archived['backgrounds'])
    by_name = {row['background']: row for row in manifest['backgrounds']}
    archive_by_name = {row['background']: row for row in archived['backgrounds']}
    measurements = {row['name']: row['measurements'] for row in cohort['backgrounds']}
    # The declared indel-affected cycles of the published panel, keyed by background
    # and cycle index exactly as that artefact records them. They are excluded and
    # counted, never silently dropped, and the count is reported in the record.
    drop: set[tuple[str, int]] = set()
    if excluded is not None and excluded.exists():
        payload = json.loads(excluded.read_bytes())
        for background, entry in payload.items():
            for index in entry['cycle_indices']:
                drop.add((background, int(index)))
    rows, t1, t2, closure, coverage, identity, grid = [], [], [], [], [], [], []
    cycle_rows = []
    for background in planned['backgrounds']:
        name = background['name']
        new = np.load(wave / by_name[name]['file'], allow_pickle=False)
        old = np.load(archive / archive_by_name[name]['file'], allow_pickle=False)
        retained = Retained(new)
        t1.append(retained.sum_check)
        if not np.array_equal(new['cycle_states'], old['cycle_states']):
            raise ValueError(f'{name}: cycle state indices differ from the archive')
        new_likelihood = np.asarray(new['likelihood'])
        old_likelihood = np.asarray(old['likelihood'])
        t2.append(np.abs(new_likelihood - old_likelihood))
        states = np.asarray(new['cycle_states'])
        identity.append(np.abs(cycle_contrast(new_likelihood, states)
                               - cycle_contrast(old_likelihood, states)))
        packed = np.asarray(new['packed_token_counts'])
        grid.extend(bool(len(set(packed[row].tolist())) != 1) for row in states)
        measured = measurements[name]
        sequences = background['sequences']
        seen = set()
        for cycle_index, (cycle, positions) in enumerate(
                zip(states, np.asarray(new['cycle_positions']))):
            wild, low, high, double = (int(v) for v in cycle)
            sites = sorted(int(v) - 1 for v in positions)
            cycle_parts = {}
            for key in PARTITION:
                values = np.asarray([
                    retained.parts(state, sites)[key] for state in (wild, low, high, double)])
                cycle_parts[key] = float(-(values[3] - values[1] - values[2] + values[0]))
            cycle_parts['full'] = float(cycle_contrast(new_likelihood, cycle[None, :])[0])
            cycle_parts['closure_nats'] = float(
                cycle_parts['full'] - sum(cycle_parts[key] for key in PARTITION))
            cycle_rows.append({'background': name, 'group': background['group'],
                               'positions': sites, 'excluded': (name, cycle_index) in drop,
                               **cycle_parts})
            for state, site in ((low, sites[0]), (high, sites[1])):
                if not keep_first_unexcluded(state, seen, (name, cycle_index) in drop):
                    continue
                record = mutation_parts(retained.parts(wild, [site]),
                                        retained.parts(state, [site]),
                                        float(new_likelihood[state] - new_likelihood[wild]))
                target = measured.get(sequences[state], {}).get('value')
                reference = measured.get(sequences[wild], {}).get('value')
                if target is None or reference is None:
                    continue
                fit = alignment(packer.checked(sequences[wild], retained, wild),
                                packer.checked(sequences[state], retained, state), [site])
                rows.append({'group': background['group'], 'site': f'{name}:{site}',
                             'target': float(target) - float(reference),
                             'own': record['own'], 'downstream': record['downstream'],
                             'upstream': record['upstream'], 'full': record['full'],
                             'archived_full': float(old_likelihood[state] - old_likelihood[wild]),
                             'own_scored': record['own_scored'],
                             'own_residue_width': record['own_residue_width_mutant'],
                             'substitutions': 1, 'alignment': fit})
                closure.append((abs(record['closure_nats']), record['closure_bound_nats']))
                coverage.append({'own_scored': record['own_scored'],
                                 'own_residue_width': record['own_residue_width_mutant'],
                                 'token_count_changed': bool(
                                     int(packed[state]) != int(packed[wild]))})
        new.close(); old.close()
    return {'rows': rows, 't1': t1, 't2': np.concatenate(t2), 'closure': closure,
            'coverage': coverage, 'repeat_max': repeat_max,
            'wave_manifest': manifest, 'archive_manifest': archived,
            'cycle_identity': np.concatenate(identity), 'cycle_rows': cycle_rows,
            'cycle_grid_unequal': grid, 'declared_exclusions': len(drop),
            'unit': 'MegaScale family group',
            'endpoint': 'first-order single-mutant ddG, kcal/mol'}


_ANCHOR_PACKER: Packer | None = None


def assays_for_shard(assays: list[str], shard: int, shards: int) -> list[str]:
    """The assays whose sorted index falls in this shard. Every assay is in one shard."""

    if shards < 1 or not 0 <= shard < shards:
        raise ValueError(f'assay shard {shard} of {shards} is outside the shard count')
    return [assay for index, assay in enumerate(assays) if index % shards == shard]


def _anchor_assay_job(job: tuple) -> dict:
    """Score one anchor assay. The packer is inherited from the parent process."""

    assay, row, wave_file, old_file = job
    packer = _ANCHOR_PACKER
    if packer is None:
        raise RuntimeError('anchor packing was not initialised')
    new = np.load(wave_file, allow_pickle=False)
    old = np.load(old_file, allow_pickle=False)
    try:
        retained = Retained(new)
        if new['mutants'].tolist() != old['mutants'].tolist():
            raise ValueError(f'{assay}: mutation order differs from the archive')
        new_absolute = np.concatenate(([float(new['wt_likelihood'])],
                                       float(new['wt_likelihood']) + np.asarray(new['likelihood'], dtype=np.float64)))
        old_absolute = np.concatenate(([float(old['wt_likelihood'])],
                                       float(old['wt_likelihood']) + np.asarray(old['likelihood'], dtype=np.float64)))
        wildtype = row['wildtype']
        sequences = [wildtype] + list(row['sequences'])
        wild_state = packer.checked(wildtype, retained, 0)
        entries, closure, coverage = [], [], []
        for position, mutant in enumerate(new['mutants'].tolist()):
            sites = substitution_sites(wildtype, mutant)
            record = mutation_parts(retained.parts(0, sites), retained.parts(position + 1, sites),
                                    float(new['likelihood'][position]))
            mutant_state = packer.checked(sequences[position + 1], retained, position + 1)
            fit = alignment(wild_state, mutant_state, sites)
            if packer.partition_mode == 'span' and len(sites) == 1:
                try:
                    masks = retokenized_span_masks(wild_state, mutant_state, sites[0])
                except ValueError as error:
                    fit.update(aligned=False, reason=str(error))
                else:
                    parts = []
                    for state_index, mask in zip((0, position + 1), masks):
                        low, high = retained.offsets[state_index:state_index+2]
                        parts.append(masked_state_parts(retained.terms[low:high],
                                                        retained.counts[low:high], mask))
                    record = mutation_parts(*parts, float(new['likelihood'][position]))
                    fit.update(aligned=True, reason='exact mutation-associated token span')
            entries.append({'own': record['own'], 'downstream': record['downstream'],
                            'upstream': record['upstream'], 'full': record['full'],
                            'archived_full': float(old['likelihood'][position]),
                            'substitutions': len(sites),
                            'own_scored': record['own_scored'],
                            'own_residue_width': record['own_residue_width_mutant'],
                            'measured': float(new['measured'][position]),
                            'alignment': fit})
            closure.append((abs(record['closure_nats']), record['closure_bound_nats']))
            coverage.append({'own_scored': record['own_scored'],
                             'own_residue_width': record['own_residue_width_mutant'],
                             'token_count_changed': None})
        return {'assay': assay, 't1': retained.sum_check,
                't2': np.maximum(np.abs(new_absolute - old_absolute),
                                 np.concatenate(([0.0], np.abs(
                                     np.asarray(new['likelihood'], dtype=np.float64)
                                     - np.asarray(old['likelihood'], dtype=np.float64))))),
                'closure': closure, 'coverage': coverage,
                'row': {'assay': assay, 'cluster': row['cluster'], 'entries': entries}}
    finally:
        new.close()
        old.close()


def _assemble_anchor(pieces: list[dict], manifest: dict, archived: dict) -> dict:
    if not pieces:
        raise ValueError('an anchor shard produced no assays')
    return {'rows': [piece['row'] for piece in pieces],
            't1': [piece['t1'] for piece in pieces],
            't2': np.concatenate([np.asarray(piece['t2'], dtype=np.float64) for piece in pieces]),
            'closure': [pair for piece in pieces for pair in piece['closure']],
            'coverage': [item for piece in pieces for item in piece['coverage']],
            'repeat_max': 0.0, 'wave_manifest': manifest, 'archive_manifest': archived,
            'unit': 'wild-type identity cluster', 'endpoint': 'measured DMS fitness, rank'}


def load_anchor(wave: Path, archive: Path, arm: str, source: Path,
                support: set[str] | None, extra: list[Path], packer: Packer, *,
                assay_shard: int | None = None, assay_shards: int | None = None,
                workers: int = 1) -> dict:
    cohort = json.loads(source.read_bytes())
    manifest = json.loads((wave / f'manifest_{arm}.json').read_text())
    archived = json.loads((archive / f'manifest_{arm}.json').read_text())
    index = index_files([archive, *extra])
    cohort_map = {row['assay']: row for row in cohort['assays']}
    wave_map = {row['assay']: row for row in manifest['assays']}
    archive_map = {row['assay']: row for row in archived['assays']}
    assays = sorted(set(wave_map) & set(archive_map))
    if support is not None:
        assays = sorted(set(assays) & support)
    if assay_shards is not None:
        if assay_shard is None:
            raise ValueError('an assay shard index is required when a shard count is set')
        assays = assays_for_shard(assays, assay_shard, assay_shards)
    elif assay_shard is not None:
        raise ValueError('an assay shard index requires a shard count')
    jobs = [(assay, cohort_map[assay], str(wave / wave_map[assay]['file']),
             str(locate_archived_npz(archive, archive_map[assay]['file'], index)))
            for assay in assays]
    global _ANCHOR_PACKER
    _ANCHOR_PACKER = packer
    if workers > 1 and len(jobs) > 1:
        with mp.get_context('fork').Pool(min(workers, len(jobs))) as pool:
            pieces = pool.map(_anchor_assay_job, jobs, chunksize=1)
    else:
        pieces = [_anchor_assay_job(job) for job in jobs]
    return _assemble_anchor(pieces, manifest, archived)


# ------------------------------------------------------------- the contrast


def alignment_table(entries: list[dict], *, rule: str = ALIGNMENT) -> dict:
    """Per-arm alignment rate and how far a mismatch reaches past the substitution."""

    fits = [entry['alignment'] for entry in entries]
    aligned = [bool(fit['aligned']) for fit in fits]
    extents = [int(fit['mismatch_extent_residues']) for fit in fits
               if fit['mismatch_extent_residues'] is not None and not fit['aligned']]
    unscored = int(sum(bool(fit.get('differing_outside_scored_span')) for fit in fits
                       if not fit['aligned']))
    upstream = np.abs(np.asarray([entry['upstream'] for entry in entries], dtype=np.float64))
    reasons: dict[str, int] = {}
    for fit in fits:
        if not fit['aligned']:
            reasons[str(fit['reason'])] = reasons.get(str(fit['reason']), 0) + 1
    percentiles = [50, 90, 99, 100]
    return {
        'rule': rule,
        'variants': len(fits),
        'aligned': int(sum(aligned)),
        'unaligned': int(len(aligned) - sum(aligned)),
        'aligned_fraction': float(np.mean(aligned)) if aligned else None,
        'unaligned_reasons': reasons,
        'unaligned_with_unscored_substitution': unscored,
        'packed_length_delta_nonzero': int(sum(bool(fit['packed_length_delta']) for fit in fits)),
        'mismatch_extent_residues': (
            {str(q): float(np.percentile(extents, q)) for q in percentiles} if extents else None),
        'mismatch_extent_residues_max': int(max(extents)) if extents else 0,
        'upstream_term_abs_nats': {
            'max': float(upstream.max()) if upstream.size else None,
            'nonzero': int((upstream > 0).sum()),
            'p99': float(np.percentile(upstream, 99)) if upstream.size else None},
        'upstream_term_abs_nats_aligned': {
            'max': float(np.max(upstream[np.asarray(aligned)])) if any(aligned) else None,
            'nonzero': int((upstream[np.asarray(aligned)] > 0).sum()) if any(aligned) else 0},
    }


def group_contrast(rows: list[dict], *, key_target='target') -> dict:
    """Within-group rank quality of each part, and the paired part-minus-full contrast."""

    groups = np.asarray([row['group'] for row in rows])
    target = np.asarray([row[key_target] for row in rows], dtype=float)
    values = {}
    per_group = {}
    for key in PART_KEYS:
        prediction = np.asarray([row[key] for row in rows], dtype=float)
        labels, spearman = group_spearman(target, prediction, groups)
        per_group[key] = spearman
        values[key] = interval(spearman, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    for key in ('own', 'downstream'):
        paired = [None if a is None or b is None else a - b
                  for a, b in zip(per_group[key], per_group['full'])]
        values[f'{key}_minus_full'] = interval(paired, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    values['groups'] = len(set(groups.tolist()))
    values['variants'] = len(rows)
    return values


def assay_contrast(rows: list[dict], *, only_single=False) -> dict:
    """Per-assay Spearman of each part against the measured endpoint, family-bootstrapped."""

    table = []
    for row in rows:
        entries = [e for e in row['entries'] if not only_single or e['substitutions'] == 1]
        if len(entries) < 3:
            continue
        target = standardized_rank(np.asarray([e['measured'] for e in entries], dtype=float))
        record = {'assay': row['assay'], 'cluster': row['cluster'], 'n_variants': len(entries)}
        for key in PART_KEYS:
            prediction = np.asarray([e[key] for e in entries], dtype=float)
            record[key] = correlation(rankdata(standardized_rank(prediction)), target)
        for key in ('own', 'downstream'):
            left, right = record[key], record['full']
            record[f'{key}_minus_full'] = None if left is None or right is None else left - right
        record['archived_full'] = correlation(
            rankdata(standardized_rank(np.asarray([e['archived_full'] for e in entries],
                                                  dtype=float))), target)
        table.append(record)
    keys = list(PART_KEYS) + ['own_minus_full', 'downstream_minus_full', 'archived_full']
    values = {key: summarize(table, key, bootstrap=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
              for key in keys}
    values['assays'] = len(table)
    values['clusters'] = len({r['cluster'] for r in table})
    values['variants'] = sum(r['n_variants'] for r in table)
    values['per_assay'] = table
    return values


def write_anchor_partial(path: Path, loaded: dict) -> None:
    """Write one shard's per-assay pieces so a later merge can rebuild the arm."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    payload = {
        'rows': loaded['rows'], 't1': loaded['t1'], 't2': loaded['t2'],
        'closure': loaded['closure'], 'coverage': loaded['coverage'],
        'repeat_max': loaded['repeat_max'], 'wave_manifest': loaded['wave_manifest'],
        'archive_manifest': loaded['archive_manifest'], 'unit': loaded['unit'],
        'endpoint': loaded['endpoint'], 'assays': [row['assay'] for row in loaded['rows']],
    }
    with temporary.open('wb') as handle:
        pickle.dump(payload, handle, protocol=4)
    temporary.replace(path)


def merge_anchor_partials(paths: list[Path]) -> dict:
    """Concatenate shard pieces in assay-name order, refusing a repeated assay."""

    pieces = []
    meta = None
    seen: set[str] = set()
    for path in paths:
        with Path(path).open('rb') as handle:
            payload = pickle.load(handle)
        if meta is None:
            meta = payload
        cursor = 0
        entry_cursor = 0
        t2 = np.asarray(payload['t2'], dtype=np.float64)
        for assay, t1, row in zip(payload['assays'], payload['t1'], payload['rows']):
            if assay in seen:
                raise ValueError(f'duplicate assay {assay}')
            seen.add(assay)
            count = len(row['entries'])
            width = count + 1
            pieces.append({
                'assay': assay, 't1': t1, 'row': row,
                't2': t2[cursor:cursor + width],
                'closure': payload['closure'][entry_cursor:entry_cursor + count],
                'coverage': payload['coverage'][entry_cursor:entry_cursor + count],
            })
            cursor += width
            entry_cursor += count
        if cursor != len(t2) or entry_cursor != len(payload['closure']):
            raise ValueError(f'{path} does not partition into its assays')
    if meta is None or not pieces:
        raise ValueError('no anchor partials to merge')
    pieces.sort(key=lambda piece: piece['assay'])
    return _assemble_anchor(pieces, meta['wave_manifest'], meta['archive_manifest'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', required=True, choices=('stability', 'pairwise', 'anchor'))
    parser.add_argument('--arm', required=True)
    parser.add_argument('--wave', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--plan', type=Path, help='pairwise only: the label-free extraction plan')
    parser.add_argument('--excluded-cycles', type=Path,
                        help='pairwise only: the declared indel-affected cycle exclusions')
    parser.add_argument('--support', type=Path,
                        help='anchor only: the frozen panel assay identifiers')
    parser.add_argument('--archive-extra', type=Path, action='append', default=[],
                        help='anchor only: further directories holding archived shard files')
    parser.add_argument('--assay-shard', type=int,
                        help='anchor only: this process keeps assays whose sorted index matches')
    parser.add_argument('--assay-shards', type=int,
                        help='anchor only: number of assay shards')
    parser.add_argument('--workers', type=int, default=1,
                        help='anchor only: parallel assay processes, forked after the tokenizer loads')
    parser.add_argument('--partial', type=Path,
                        help='anchor only: write this shard and skip the arm-level record')
    parser.add_argument('--merge-partials', type=Path, nargs='+',
                        help='anchor only: rebuild the arm record from these shard partials')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()

    if args.arm not in ROSTER:
        raise SystemExit(f'{args.arm} is not on the frozen roster')
    if args.merge_partials:
        if args.cohort != 'anchor':
            raise SystemExit('partial merge is only defined for the anchor cohort')
        loaded = merge_anchor_partials(args.merge_partials)
    else:
        spec = _util.spec_from_file_location('stage46', ROOT / 'scripts/capability/stages/context_homologue.py')
        stage46 = _util.module_from_spec(spec)
        spec.loader.exec_module(stage46)
        packer = Packer(load_readout_arm(args.arm, stage46,
                                        dtype=ARM_DTYPE.get(args.arm, 'float32')))
        if args.cohort == 'stability':
            loaded = load_stability(args.wave, args.archive, args.arm, args.source, packer)
        elif args.cohort == 'pairwise':
            if args.plan is None:
                raise SystemExit('the pairwise cohort needs its label-free plan')
            loaded = load_pairwise(args.wave, args.archive, args.arm, args.source, args.plan,
                                  args.excluded_cycles, packer)
        else:
            support = None
            if args.support is not None:
                support = set(json.loads(args.support.read_bytes())['assays'])
            loaded = load_anchor(args.wave, args.archive, args.arm, args.source, support,
                                list(args.archive_extra), packer,
                                assay_shard=args.assay_shard, assay_shards=args.assay_shards,
                                workers=args.workers)
            if args.partial is not None:
                write_anchor_partial(args.partial, loaded)
                print(json.dumps({'arm': args.arm, 'partial': str(args.partial),
                                  'assays': len(loaded['rows'])}), flush=True)
                return

    t1_worst = float(np.max(np.abs(np.asarray(loaded['t1'], dtype=np.float64))))
    t2_values = np.asarray(loaded['t2'], dtype=np.float64)
    t2_worst = float(t2_values.max())
    realised = tier(t2_worst, loaded['repeat_max'])
    closure_pairs = np.asarray(loaded['closure'], dtype=np.float64)
    closure_worst = float(closure_pairs[:, 0].max())
    closure_over_bound = int((closure_pairs[:, 0] > closure_pairs[:, 1]).sum())

    record = {
        'schema': SCHEMA, 'cohort': args.cohort, 'arm': args.arm,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'wave': str(args.wave), 'archive': str(args.archive),
        'source_sha256': sha(args.source),
        'declared_tolerances': {
            'T1_retention_nats': 0.0, 'T2_tier2_nats': TIER2_NATS,
            'T2_tier2_repeat_multiple': TIER2_REPEAT_MULTIPLE,
            'derived_tier1': DERIVED_TIER1, 'derived_tier2': DERIVED_TIER2,
            'cycle_identity_nats': CYCLE_IDENTITY_NATS,
            'declared': 'fixed in this file before any residual was read'},
        'runtime': {
            'wave_torch': loaded['wave_manifest'].get('torch_version'),
            'archive_torch': loaded['archive_manifest'].get('torch_version'),
            'wave_gpu': loaded['wave_manifest'].get('gpu'),
            'archive_gpu': loaded['archive_manifest'].get('gpu'),
            'wave_blas': loaded['wave_manifest'].get('blas'),
            'wave_batch_size': loaded['wave_manifest']['identity'].get('batch_size'),
            'archive_batch_size': loaded['archive_manifest']['identity'].get('batch_size'),
            'wave_dtype': loaded['wave_manifest']['identity'].get('dtype'),
            'archive_dtype': loaded['archive_manifest']['identity'].get('dtype'),
            'wave_elapsed_seconds': loaded['wave_manifest'].get('elapsed_seconds'),
            'archive_elapsed_seconds': loaded['archive_manifest'].get('elapsed_seconds')},
        'T1': {'worst_abs_nats': t1_worst, 'cells': len(loaded['t1']),
               'passed': t1_worst == 0.0},
        'T2': {'states': int(t2_values.size), 'worst_abs_nats': t2_worst,
               'states_nonzero': int((t2_values > 0).sum()),
               'states_above_tier2': int((t2_values > TIER2_NATS).sum()),
               'archived_repeat_max_nats': float(loaded['repeat_max']),
               'tier': realised},
        'closure': {
            'worst_abs_nats': closure_worst, 'variants': len(loaded['closure']),
            'variants_above_arithmetic_bound': closure_over_bound,
            'bound': ('the retained scalar is a float32 reduction and the class sums are a '
                      'float64 reduction of the same terms, so a residual of about '
                      'log2(n) * 2**-24 of each state summed magnitude is arithmetic; the '
                      'bound is per variant and declared from that, never from the data')},
        'coverage': {
            'variants': len(loaded['coverage']),
            'mutated_residue_unscored': int(sum(not c['own_scored'] for c in loaded['coverage'])),
            'own_residue_width_max': int(max(c['own_residue_width'] for c in loaded['coverage'])),
            'own_residue_width_mean': float(np.mean([c['own_residue_width']
                                                     for c in loaded['coverage']])),
            'token_count_changed': (
                None if loaded['coverage'][0]['token_count_changed'] is None
                else int(sum(bool(c['token_count_changed']) for c in loaded['coverage'])))},
        'unit': loaded['unit'], 'endpoint': loaded['endpoint'],
        'resampling': {'unit': loaded['unit'], 'draws': BOOTSTRAP_DRAWS,
                       'seed': BOOTSTRAP_SEED, 'interval': '95% percentile'},
    }

    if args.cohort == 'anchor':
        entries = [entry for row in loaded['rows'] for entry in row['entries']]
        record['alignment'] = alignment_table(entries)
        record['alignment']['single_substitution'] = alignment_table(
            [e for e in entries if e['substitutions'] == 1])
        aligned_rows = [dict(row, entries=[e for e in row['entries'] if e['alignment']['aligned']])
                        for row in loaded['rows']]
        record['contrast'] = {
            'single_substitution': assay_contrast(aligned_rows, only_single=True),
            'all_variants': assay_contrast(aligned_rows)}
        primary = record['contrast']['single_substitution']
        derived_rows = [dict(row, entries=[e for e in row['entries'] if e['substitutions'] == 1])
                        for row in loaded['rows']]
        unrestricted = assay_contrast(derived_rows, only_single=True)
        derived = [abs(r['full'] - r['archived_full']) for r in unrestricted['per_assay']
                   if r['full'] is not None and r['archived_full'] is not None]
        record['derived_invariant'] = {
            'statistic': ('per-assay raw_M Spearman against the measured endpoint, on the '
                          'panel support and the single-substitution stratum, over every '
                          'variant rather than only the aligned ones, because the published '
                          'statistic reads every variant'),
            'units': len(derived), 'worst_abs_difference': max(derived) if derived else 0.0}
    else:
        record['alignment'] = alignment_table(loaded['rows'])
        record['contrast'] = {'all_variants': group_contrast(
            [row for row in loaded['rows'] if row['alignment']['aligned']])}
        primary = record['contrast']['all_variants']
        groups = np.asarray([row['group'] for row in loaded['rows']])
        target = np.asarray([row['target'] for row in loaded['rows']], dtype=float)
        _, new_spearman = group_spearman(
            target, np.asarray([row['full'] for row in loaded['rows']]), groups)
        _, old_spearman = group_spearman(
            target, np.asarray([row['archived_full'] for row in loaded['rows']]), groups)
        derived = [abs(a - b) for a, b in zip(new_spearman, old_spearman)
                   if a is not None and b is not None]
        record['derived_invariant'] = {
            'statistic': 'per-group within-background Spearman of the retained mutation score',
            'units': len(derived), 'worst_abs_difference': max(derived) if derived else 0.0}

    if args.cohort == 'pairwise':
        cycle = np.asarray(loaded['cycle_identity'], dtype=np.float64)
        closures = np.asarray([abs(row['closure_nats']) for row in loaded['cycle_rows']])
        record['T3a'] = {'cycles': int(cycle.size), 'worst_abs_nats': float(cycle.max()),
                         'passed': bool(cycle.max() <= CYCLE_IDENTITY_NATS)}
        record['T3b'] = {'cycles': int(closures.size),
                         'worst_cycle_closure_nats': float(closures.max()),
                         'cycles_without_one_token_grid': int(sum(loaded['cycle_grid_unequal'])),
                         'declared_exclusions_applied': loaded['declared_exclusions'],
                         'interpretation': (
                             'the interaction-term decomposition is recorded and carries no '
                             'finding: that endpoint is unresolved on every arm, and a '
                             'decomposition of an unresolved quantity inherits its '
                             'non-resolution')}

    fraction = (record['alignment']['single_substitution']['aligned_fraction']
                if args.cohort == 'anchor' else record['alignment']['aligned_fraction'])
    units = primary.get('groups', primary.get('clusters'))
    record['measurable'] = {
        'declared_minimum_aligned_fraction': MEASURABLE_ALIGNED_FRACTION,
        'aligned_fraction': fraction,
        'units_on_aligned_support': units,
        'variants_on_aligned_support': primary['variants'],
        'verdict': bool(fraction is not None and fraction >= MEASURABLE_ALIGNED_FRACTION
                        and units is not None and units >= 3
                        and primary['own']['interval'] is not None
                        and primary['full']['interval'] is not None),
        'support_identifier': (f"{args.cohort}_aligned_only_"
                               f"{primary['variants']}variants_{units}units"),
        'note': ('the contrast is computed on aligned variants only; the restriction '
                 'changes the support, so the restricted quantity carries its own '
                 'identifier and is not the published cohort'),
    }
    limit = DERIVED_TIER1 if realised == 1 else DERIVED_TIER2
    record['passed'] = bool(record['T1']['passed'] and realised in (1, 2)
                            and closure_over_bound == 0
                            and record['derived_invariant']['worst_abs_difference'] <= limit
                            and (args.cohort != 'pairwise' or record['T3a']['passed']))
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / f'position_terms_{args.cohort}_{args.arm}.json', record)
    print(json.dumps({'arm': args.arm, 'cohort': args.cohort, 'T1': t1_worst,
                      'T2': t2_worst, 'tier': realised, 'closure': closure_worst,
                      'derived': record['derived_invariant']['worst_abs_difference'],
                      'aligned_fraction': record['alignment']['aligned_fraction'],
                      'measurable': record['measurable']['verdict'],
                      'passed': record['passed']}), flush=True)


if __name__ == '__main__':
    main()
