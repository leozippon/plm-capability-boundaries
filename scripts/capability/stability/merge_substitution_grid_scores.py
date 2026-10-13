#!/usr/bin/env python3
"""Merge legacy reused and new missing native likelihoods into full substitution grids.

The reviewed 12 human/mouse intracellular-domain backgrounds were only measured at
the exact wild type plus 256 frozen cohort single mutants each (3084 states per
model). A later forward wave scored the remaining 9810 substitutions per model
plus twelve wild-type repeat checks; its shard plans are in
``data/r6_constraint_response_20261013``. This script binds both sources and
assembles one complete native-likelihood grid per (arm, background): one wild
type and the 19 non-wild-type canonical substitutions at every site.

It is a merge and validation step only. It loads no model, needs no GPU, and
performs no inference. Every legacy score is preserved verbatim rather than
averaged: the new wave's wild-type state is a repeat check against the archived
wild type, and the merged wild-type score is the legacy value. The merged
per-state quantity is the native packed residue-span summed log likelihood in
nats (``FULLSUM``); the wild-type-referenced response is

    CFSCORE R_M,i(a) = S_native(s_i -> a) - S_native(s)

which is zero at the wild type by construction. Neither quantity is a
probability, and a token-segmentation difference between states or arms is not a
biological error. No per-token mean, no rank transform and no q-normalised
probability field is produced.

Every input's digest is recorded, and every input that declares one is verified
against it: the grid manifest verifies the allowlist and cohort digests, the parent
and shard plans are checked by file digest and logical plan digest, the two reuse
maps are digest-computed and cross-checked against the legacy manifests and
archive entries they name, every legacy NPZ is verified against its archive
digest, and every new shard manifest and NPZ is verified against its declared
digest. A mismatch, a duplicated or missing
background, a state outside the enumerated grid, a non-finite score, a wild-type
repeat whose token counts differ, or a wild-type repeat score drift beyond the
fixed 1e-3 nats tolerance refuses the merge rather than being silently
reinterpreted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.amino_acids import AA20
from src.capability.core.io import sha256_file, write_json
from src.capability.extensions.stability_followups import verify_scalar_archive
from src.capability.stability.stability_gate import validate_likelihood_token_support

#: The substitution-grid contract is imported, never restated, so the grid, the
#: logical plan digest and the frozen arithmetic cannot drift from preparation.
_SPEC = importlib.util.spec_from_file_location(
    'substitution_grid_contract',
    ROOT / 'scripts/capability/stability/build_substitution_grid_plan.py')
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError('the substitution-grid preparation contract is unavailable')
PLAN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(PLAN)

sha256_text = PLAN.sha256_text
substitution_grid = PLAN.substitution_grid
plan_digest = PLAN.plan_digest
BUDGET = PLAN.BUDGET

#: Written artefact schema and the fixed wild-type repeat tolerance. The tolerance
#: is the one the pre-merge calibration fixed; it is never relaxed after a failure.
MERGE_SCHEMA = 'stability_substitution_grid_merge_v1'
WT_TOLERANCE_NATS = 1e-3

DEFAULT_REFERENCE = ROOT / 'data/r6_constraint_response_20261013'
DEFAULT_ARCHIVE = (ROOT / 'results/external_baseline/20261008022745_762304146fdd/'
                          'stability_likelihood_small_dense_gpu3')
DEFAULT_RAW = ROOT / 'results/R6/r6_constraint_response_missing_20261013/raw'
DEFAULT_OUT = ROOT / 'results/R6/r6_constraint_response_missing_20261013/merged'

#: Every array a native likelihood-only archive must carry. A missing member is
#: corrupt data, reported as such rather than read as an empty state.
ARCHIVE_ARRAYS = ('likelihood', 'packed_token_counts', 'scored_token_counts',
                  'pooled_token_counts', 'token_ids', 'token_offsets',
                  'variant_states', 'variant_positions', 'state_sequence_sha256',
                  'metadata')

RESPONSE_SEMANTICS = ('CFSCORE R_M,i(a) = S_native(s_i -> a) - S_native(s); raw native '
                      'packed residue-span summed log likelihood in nats; exactly zero at '
                      'the wild type')
DISCLOSURE = ('each merged grid is a model-only induced full-native-sum response matrix, '
              'not a biological conditional probability; a token-segmentation difference '
              'is not a biological error and is never averaged away')


class NewIntakeUnavailable(RuntimeError):
    """The new forward-wave intake is not present yet; readiness is reported, not faked."""


# --------------------------------------------------------------------------- #
# Reading and hash binding
# --------------------------------------------------------------------------- #

def resolve_path(value: str) -> Path:
    """Resolve a manifest path: absolute as given, otherwise relative to the repo."""

    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_archive(path: Path) -> dict:
    """One native archive as a plain mapping, with corrupt data refused explicitly."""

    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
    except Exception as error:  # noqa: BLE001 - any unreadable archive is corrupt data
        raise ValueError(f'{path}: unreadable native archive: {error}') from error
    absent = [key for key in ARCHIVE_ARRAYS if key not in arrays]
    if absent:
        raise ValueError(f'{path}: native archive is missing {absent}')
    return arrays


def full_grid_map(wildtype: str) -> dict:
    """The enumerated full grid, indexed by exact sequence hash and by grid state."""

    states = substitution_grid(wildtype)
    by_sha, by_state = {}, {}
    for state in states:
        record = {'grid_state': state['state'], 'position': state['position'],
                  'mutant': state['mutant'], 'sha256': state['sha256']}
        if state['sha256'] in by_sha:
            raise ValueError('the enumerated substitution grid repeats a sequence')
        by_sha[state['sha256']] = record
        by_state[state['state']] = record
    wild_sha = sha256_text(wildtype)
    wild = {'grid_state': 0, 'position': 0, 'mutant': None, 'sha256': wild_sha}
    by_sha[wild_sha] = wild
    by_state[0] = wild
    if len(by_state) != 1 + len(states):
        raise ValueError('the enumerated full grid is not one state per (position, mutant)')
    return {'states': states, 'by_sha': by_sha, 'by_state': by_state,
            'wildtype_sha256': wild_sha, 'full_states': 1 + len(states)}


# --------------------------------------------------------------------------- #
# Reference binding
# --------------------------------------------------------------------------- #

def load_reference(reference_dir, archive_dir) -> dict:
    """Verify and bind the frozen grid manifest, its plans and both legacy arms."""

    reference_dir, archive_dir = Path(reference_dir), Path(archive_dir)
    grid_path = reference_dir / 'grid_manifest.json'
    grid = json.loads(grid_path.read_text())
    if grid.get('schema') != PLAN.GRID_SCHEMA:
        raise ValueError(f'{grid_path}: unexpected grid manifest schema')
    grid_sha = sha256_file(grid_path)

    cohort_path = resolve_path(grid['cohort']['path'])
    cohort_sha = sha256_file(cohort_path)
    if cohort_sha != grid['cohort']['sha256']:
        raise ValueError(f'{cohort_path}: cohort digest differs from the grid manifest')
    allowlist_path = resolve_path(grid['allowlist']['path'])
    allowlist_sha = sha256_file(allowlist_path)
    if allowlist_sha != grid['allowlist']['sha256']:
        raise ValueError(f'{allowlist_path}: allowlist digest differs from the grid manifest')
    cohort = json.loads(cohort_path.read_text())

    for key, value in grid['expected'].items():
        if grid['observed'].get(key) != value:
            raise ValueError(f'the grid manifest declares expected {key}={value} but observed '
                             f'{grid["observed"].get(key)}')

    parent_path = reference_dir / 'plan.json'
    parent = json.loads(parent_path.read_text())
    if parent.get('schema') != PLAN.SCHEMA:
        raise ValueError(f'{parent_path}: unexpected extraction plan schema')
    if plan_digest(parent) != grid['parent_plan']['plan_sha256']:
        raise ValueError(f'{parent_path}: logical plan digest differs from the grid manifest')

    shards = []
    for entry in grid['shards']['files']:
        path = reference_dir / entry['file']
        if sha256_file(path) != entry['sha256']:
            raise ValueError(f'{path}: shard plan bytes differ from the grid manifest')
        plan = json.loads(path.read_text())
        if plan.get('schema') != PLAN.SCHEMA:
            raise ValueError(f'{path}: unexpected shard plan schema')
        if plan_digest(plan) != entry['plan_sha256']:
            raise ValueError(f'{path}: logical shard plan digest differs from the grid manifest')
        shard = plan['shard']
        shards.append({'file': entry['file'], 'sha256': entry['sha256'],
                       'plan_sha256': entry['plan_sha256'], 'plan': plan,
                       'index': int(shard['index']), 'count': int(shard['count']),
                       'backgrounds': list(shard['backgrounds']),
                       'missing_states': int(shard['missing_states'])})
    if len({(shard['index'], shard['count']) for shard in shards}) != len(shards):
        raise ValueError('the grid manifest repeats a shard index')

    backgrounds = {row['name']: row for row in grid['backgrounds']}
    if len(backgrounds) != len(grid['backgrounds']):
        raise ValueError(f'{grid_path}: the grid manifest repeats a background')
    if len(backgrounds) != grid['observed']['backgrounds']:
        raise ValueError('the grid manifest background list disagrees with its observed count')

    arms = {}
    for arm in sorted(grid['provenance']['arms']):
        map_path = reference_dir / f'reuse_map_{arm}.json'
        reuse = json.loads(map_path.read_text())
        if reuse.get('schema') != PLAN.REUSE_SCHEMA or reuse.get('arm') != arm:
            raise ValueError(f'{map_path}: unexpected reuse map schema or arm')
        if reuse.get('cohort_sha256') != cohort_sha:
            raise ValueError(f'{map_path}: reuse map cohort differs from the grid manifest')
        if reuse.get('parent_plan_sha256') != grid['parent_plan']['plan_sha256']:
            raise ValueError(f'{map_path}: reuse map parent plan differs from the grid manifest')
        manifest_path = resolve_path(reuse['manifest']['path'])
        if manifest_path.resolve() != (archive_dir / f'manifest_{arm}.json').resolve():
            raise ValueError(f'{map_path}: the legacy manifest is not {archive_dir}/manifest_{arm}.json')
        manifest_sha = sha256_file(manifest_path)
        if manifest_sha != reuse['manifest']['sha256']:
            raise ValueError(f'{manifest_path}: legacy manifest bytes differ from the reuse map')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('status') != 'complete':
            raise ValueError(f'{manifest_path}: legacy manifest is not complete')
        if manifest.get('identity') != reuse['identity']:
            raise ValueError(f'{manifest_path}: embedded identity differs from the reuse map identity')
        rows = {row['background']: row for row in manifest['backgrounds']}
        if set(reuse['backgrounds']) != set(backgrounds):
            raise ValueError(f'{map_path}: reuse map does not cover exactly the frozen grid backgrounds')
        for name, binding in reuse['backgrounds'].items():
            row = rows.get(name)
            if row is None:
                raise ValueError(f'{map_path}: {name} has no legacy archive row')
            if (binding['file'], binding['sha256']) != (row['file'], row['sha256']):
                raise ValueError(f'{map_path}: {name} legacy archive differs from its manifest row')
        # The existing archive validator re-hashes and re-aligns every legacy state.
        verify_scalar_archive(archive_dir, arm, cohort, cohort_sha)
        arms[arm] = {'reuse': reuse, 'reuse_sha': sha256_file(map_path),
                     'manifest_path': str(manifest_path), 'manifest_sha256': manifest_sha,
                     'identity': reuse['identity'], 'manifest': manifest}

    return {'reference_dir': reference_dir, 'grid_path': str(grid_path), 'grid_sha256': grid_sha,
            'grid': grid, 'cohort': cohort, 'cohort_path': str(cohort_path),
            'cohort_sha256': cohort_sha, 'allowlist_path': str(allowlist_path),
            'allowlist_sha256': allowlist_sha, 'parent': parent, 'shards': shards,
            'backgrounds': backgrounds, 'archive_dir': archive_dir, 'arms': arms}


def raw_roots(reference: dict, raw_root, raw_maps=None) -> dict:
    """The per-arm producer root, overridable per arm by an explicit mapping."""

    root = Path(raw_root)
    maps = dict(raw_maps or {})
    roots = {}
    for arm in reference['arms']:
        roots[arm] = Path(maps[arm]) if arm in maps else root / arm
    return roots


def _shard_dir(root: Path, shard: dict) -> Path:
    return root / f"shard_{shard['index']:02d}of{shard['count']:02d}"


def probe_new_intake(reference: dict, roots: dict) -> dict:
    """Which new shard manifests are already on disk, without reading them."""

    present, missing = {}, []
    for arm in sorted(reference['arms']):
        for shard in reference['shards']:
            manifest = _shard_dir(roots[arm], shard) / f'manifest_{arm}.json'
            if manifest.is_file():
                present.setdefault(arm, []).append(str(manifest))
            else:
                missing.append(str(manifest))
    return {'ready': not missing, 'missing': missing, 'present': present}


# --------------------------------------------------------------------------- #
# New intake binding
# --------------------------------------------------------------------------- #

def verify_new_identity(identity, arm: str, shard: dict, reference: dict, legacy: dict) -> dict:
    """Bind one new manifest identity to arm, native protocol, plan and checkpoint."""

    contract = shard['plan']['contract']
    expected = {'schema': PLAN.EXTRACTOR.LIKELIHOOD_SCHEMA, 'arm': arm,
                'dtype': contract['dtype'][arm], 'batch_size': contract['batch_size'],
                'budget': contract['budget'], 'likelihood_semantics': contract['likelihood_semantics'],
                'plan_sha256': shard['plan_sha256'], 'cohort_sha256': reference['cohort_sha256']}
    if not isinstance(identity, dict):
        raise ValueError(f'{arm}: new shard {shard["index"]} manifest carries no identity block')
    wrong = {key: [identity.get(key), value] for key, value in expected.items()
             if identity.get(key) != value}
    if wrong:
        raise ValueError(f'{arm}: new shard {shard["index"]} is not the reusable native protocol: '
                         f'{wrong}')
    for key in ('checkpoint_metadata_sha256', 'checkpoint_tensor_files'):
        if identity.get(key) != legacy.get(key):
            raise ValueError(f'{arm}: new shard {shard["index"]} declares a different legacy '
                             f'checkpoint {key}; declared identity is not weight-byte proof')
    return {'checkpoint_path_declared': identity.get('checkpoint_path'),
            'checkpoint_path_agrees': identity.get('checkpoint_path') == legacy.get('checkpoint_path'),
            'checkpoint_metadata_and_tensor_sizes_match': True,
            'weight_byte_identity_verified': False}


def validate_new_archive(background: dict, plan_row: dict, data: dict, identity: dict) -> list[dict]:
    """Re-derive every new state against the shard plan and the full grid, or refuse.

    The wild-type state is validated and reported but never inserted twice and
    never averaged into the legacy wild-type score; the caller keeps the legacy
    value and uses the new repeat only as a drift and token-count check.
    """

    name = background['name']
    sequences = plan_row['sequences']
    states = len(sequences)
    if plan_row.get('name') != name or plan_row.get('group') != background['group']:
        raise ValueError(f'{name}: shard plan row names a different background or group')
    validate_likelihood_token_support(data, states, budget=BUDGET)
    metadata = json.loads(str(data['metadata']))
    if metadata.get('identity') != identity:
        raise ValueError(f'{name}: new archive embeds a different protocol identity')
    declared = (metadata.get('background'), metadata.get('group'), metadata.get('sequences'),
                metadata.get('variants'))
    if declared != (name, background['group'], states, len(plan_row['variants'])):
        raise ValueError(f'{name}: new archive metadata disagrees with the shard plan')
    observed = np.asarray(data['state_sequence_sha256'])
    expected = np.asarray([sha256_text(sequence) for sequence in sequences], dtype='<U64')
    if observed.dtype != np.dtype('<U64') or not np.array_equal(observed, expected):
        raise ValueError(f'{name}: new archive state hashes do not match the shard plan order')
    if sequences[0] != background['wildtype'] or str(observed[0]) != background['wildtype_sha256']:
        raise ValueError(f'{name}: new archive state zero is not the declared wild type')

    grid = full_grid_map(background['wildtype'])
    declared_states = np.asarray([int(v['state']) for v in plan_row['variants']])
    declared_positions = np.asarray([int(v['position']) for v in plan_row['variants']])
    if (not np.array_equal(np.asarray(data['variant_states']), declared_states)
            or not np.array_equal(np.asarray(data['variant_positions']), declared_positions)):
        raise ValueError(f'{name}: new archive variant states or positions disagree with the plan')
    if not np.array_equal(declared_states, np.arange(1, states)):
        raise ValueError(f'{name}: shard plan variants are not exactly the non-wild-type states')

    likelihood = np.asarray(data['likelihood'], dtype=float)
    if likelihood.shape != (states,) or not np.isfinite(likelihood).all():
        raise ValueError(f'{name}: new archive carries missing or nonfinite native scores')
    packed = np.asarray(data['packed_token_counts'])
    scored = np.asarray(data['scored_token_counts'])
    pooled = np.asarray(data['pooled_token_counts'])
    entries = [{'grid_state': 0, 'position': 0, 'mutant': None, 'sha256': str(observed[0]),
                'score': float(likelihood[0]), 'packed': int(packed[0]),
                'scored': int(scored[0]), 'pooled': int(pooled[0])}]
    for variant in plan_row['variants']:
        state = int(variant['state'])
        record = grid['by_sha'].get(sha256_text(sequences[state]))
        if record is None:
            raise ValueError(f'{name}: new state {state} is outside the enumerated full grid')
        if record['position'] != int(variant['position']) or record['mutant'] != variant['mutant']:
            raise ValueError(f'{name}: new state {state} maps to a different mutation than declared')
        entries.append({'grid_state': record['grid_state'], 'position': record['position'],
                        'mutant': record['mutant'], 'sha256': str(observed[state]),
                        'score': float(likelihood[state]), 'packed': int(packed[state]),
                        'scored': int(scored[state]), 'pooled': int(pooled[state])})
    return entries


def collect_new_cells(reference: dict, roots: dict) -> dict:
    """Verify every new shard manifest and archive, then check full grid coverage."""

    cells: dict[str, dict] = {}
    for arm in sorted(reference['arms']):
        legacy = reference['arms'][arm]['identity']
        per_background: dict[str, dict] = {}
        for shard in reference['shards']:
            directory = _shard_dir(roots[arm], shard)
            manifest_path = directory / f'manifest_{arm}.json'
            if not manifest_path.is_file():
                raise NewIntakeUnavailable(f'{manifest_path}: new intake shard is absent')
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('status') != 'complete':
                raise ValueError(f'{manifest_path}: new shard manifest is not complete')
            identity = manifest.get('identity')
            binding = verify_new_identity(identity, arm, shard, reference, legacy)
            manifest_rows = manifest['backgrounds']
            rows = {row['background']: row for row in manifest_rows}
            if len(rows) != len(manifest_rows):
                raise ValueError(f'{manifest_path}: new manifest repeats a background')
            if set(rows) != set(shard['backgrounds']):
                raise ValueError(f'{manifest_path}: new manifest does not cover exactly its shard plan')
            for plan_row in shard['plan']['backgrounds']:
                name = plan_row['name']
                row = rows[name]
                if (row.get('group') != plan_row['group']
                        or int(row.get('sequences', -1)) != len(plan_row['sequences'])
                        or int(row.get('variants', -1)) != len(plan_row['variants'])):
                    raise ValueError(f'{arm}/{name}: new manifest row disagrees with its shard plan')
                archive_path = directory / row['file']
                if not archive_path.is_file():
                    raise ValueError(f'{arm}/{name}: new archive {archive_path} is absent')
                if sha256_file(archive_path) != row['sha256']:
                    raise ValueError(f'{arm}/{name}: new archive bytes differ from the manifest digest')
                if name in per_background:
                    raise ValueError(f'{arm}/{name}: background is covered by more than one shard')
                data = _read_archive(archive_path)
                entries = validate_new_archive(reference['backgrounds'][name], plan_row, data,
                                               identity)
                per_background[name] = {
                    'arm': arm, 'shard': shard, 'shard_index': shard['index'],
                    'manifest_path': str(manifest_path), 'manifest_sha256': sha256_file(manifest_path),
                    'file': row['file'], 'sha256': row['sha256'], 'plan_row': plan_row,
                    'identity_binding': binding, 'entries': entries}
        expected = set(reference['backgrounds'])
        absent = expected - set(per_background)
        if absent:
            raise ValueError(f'{arm}: new intake does not cover {sorted(absent)}')
        if set(per_background) - expected:
            raise ValueError(f'{arm}: new intake covers unknown backgrounds')
        cells[arm] = per_background
    return cells


# --------------------------------------------------------------------------- #
# Legacy binding and assembly
# --------------------------------------------------------------------------- #

def validate_old_archive(background: dict, binding: dict, manifest_row: dict, identity: dict,
                         data: dict) -> list[dict]:
    """Re-derive every legacy state against the frozen cohort, the reuse map and the grid."""

    name = background['name']
    reused = binding['reused']
    states = len(reused)
    validate_likelihood_token_support(data, states, budget=BUDGET)
    metadata = json.loads(str(data['metadata']))
    if metadata.get('identity') != identity:
        raise ValueError(f'{name}: legacy archive embeds a different protocol identity')
    declared = (metadata.get('background'), metadata.get('group'), metadata.get('sequences'),
                metadata.get('variants'))
    if declared != (name, background['group'], states, states - 1):
        raise ValueError(f'{name}: legacy archive metadata disagrees with its cohort support')
    if (manifest_row.get('file'), manifest_row.get('sha256')) != (binding['file'], binding['sha256']):
        raise ValueError(f'{name}: legacy archive differs from its manifest row')
    if int(manifest_row.get('sequences', -1)) != states or int(manifest_row.get('variants', -1)) != states - 1:
        raise ValueError(f'{name}: legacy manifest support is not the frozen cohort support')

    observed = np.asarray(data['state_sequence_sha256'])
    expected = np.asarray([entry['sha256'] for entry in reused], dtype='<U64')
    if observed.dtype != np.dtype('<U64') or not np.array_equal(observed, expected):
        raise ValueError(f'{name}: legacy state hashes do not match the reuse map order')
    if str(observed[0]) != background['wildtype_sha256']:
        raise ValueError(f'{name}: legacy state zero is not the declared wild type')
    if (not np.array_equal(np.asarray(data['variant_states']), np.arange(1, states))
            or not np.array_equal(np.asarray(data['variant_positions']),
                                  [int(entry['position']) for entry in reused[1:]])):
        raise ValueError(f'{name}: legacy variant states or positions differ from the cohort')

    likelihood = np.asarray(data['likelihood'], dtype=float)
    if likelihood.shape != (states,) or not np.isfinite(likelihood).all():
        raise ValueError(f'{name}: legacy archive carries missing or nonfinite native scores')
    packed = np.asarray(data['packed_token_counts'])
    scored = np.asarray(data['scored_token_counts'])
    pooled = np.asarray(data['pooled_token_counts'])

    grid = full_grid_map(background['wildtype'])
    entries, seen = [], set()
    for index, entry in enumerate(reused):
        if index == 0:
            if entry['role'] != 'wildtype' or int(entry['grid_state']) != 0:
                raise ValueError(f'{name}: legacy reuse map does not start at the wild type')
            grid_state, position, mutant = 0, 0, None
        else:
            grid_state = int(entry['grid_state'])
            record = grid['by_state'].get(grid_state)
            if record is None:
                raise ValueError(f'{name}: legacy reuse state {grid_state} is outside the full grid')
            if (record['sha256'] != entry['sha256'] or record['position'] != int(entry['position'])
                    or record['mutant'] != entry['mutant']):
                raise ValueError(f'{name}: legacy state {index} maps to a different mutation than declared')
            if grid_state in seen:
                raise ValueError(f'{name}: the legacy reuse map repeats a grid state')
            seen.add(grid_state)
            position, mutant = record['position'], record['mutant']
        entries.append({'grid_state': grid_state, 'position': position, 'mutant': mutant,
                        'sha256': entry['sha256'], 'score': float(likelihood[index]),
                        'packed': int(packed[index]), 'scored': int(scored[index]),
                        'pooled': int(pooled[index])})
    return entries


def assemble_full_grid(background: dict, old_entries: list[dict], new_entries: list[dict],
                       *, wt_tolerance: float = WT_TOLERANCE_NATS) -> dict:
    """Place every state exactly once, preserving legacy scores and the wild type."""

    name = background['name']
    grid = full_grid_map(background['wildtype'])
    if len(grid['states']) != 19 * len(background['wildtype']):
        raise ValueError(f'{name}: the enumerated grid is not 19 states per site')
    total = grid['full_states']
    score = np.full(total, np.nan, dtype=np.float64)
    filled = np.zeros(total, dtype=bool)
    position = np.zeros(total, dtype=np.int32)
    mutant_index = np.full(total, -1, dtype=np.int8)
    sha = np.empty(total, dtype='<U64')
    packed = np.zeros(total, dtype=np.int64)
    scored = np.zeros(total, dtype=np.int64)
    pooled = np.zeros(total, dtype=np.int64)

    def place(entry: dict, source: str) -> None:
        grid_state = int(entry['grid_state'])
        if not 0 <= grid_state < total:
            raise ValueError(f'{name}: state {grid_state} is outside the full grid')
        if filled[grid_state]:
            raise ValueError(f'{name}: grid state {grid_state} is filled more than once ({source})')
        filled[grid_state] = True
        score[grid_state] = entry['score']
        position[grid_state] = entry['position']
        mutant_index[grid_state] = -1 if entry['mutant'] is None else AA20.index(entry['mutant'])
        sha[grid_state] = entry['sha256']
        packed[grid_state] = entry['packed']
        scored[grid_state] = entry['scored']
        pooled[grid_state] = entry['pooled']

    # Legacy first, including the wild type, which is preserved rather than averaged
    # with the new repeat measurement.
    place(old_entries[0], 'legacy wild type')
    for entry in old_entries[1:]:
        place(entry, 'legacy')
    new_wild = new_entries[0]
    if int(new_wild['grid_state']) != 0 or new_wild['sha256'] != sha[0]:
        raise ValueError(f'{name}: the new wild-type repeat does not match the legacy wild type')
    if new_wild['position'] != 0 or new_wild['mutant'] is not None:
        raise ValueError(f'{name}: the new wild-type repeat is not a wild-type state')
    for field in ('packed', 'scored', 'pooled'):
        if int(new_wild[field]) != int(old_entries[0][field]):
            raise ValueError(f'{name}: the new wild-type repeat {field} token count '
                             f'({new_wild[field]}) differs from the legacy wild type '
                             f'({old_entries[0][field]})')
    for entry in new_entries[1:]:
        place(entry, 'new')

    if not filled.all():
        missing = np.flatnonzero(~filled)
        raise ValueError(f'{name}: {len(missing)} full-grid states are unfilled, e.g. {missing[:8].tolist()}')
    if not np.isfinite(score).all():
        raise ValueError(f'{name}: merged grid carries a nonfinite native score')
    drift = abs(float(new_wild['score']) - float(score[0]))
    if not np.isfinite(drift):
        raise ValueError(f'{name}: wild-type repeat drift is not finite')
    if drift > wt_tolerance:
        raise ValueError(f'{name}: wild-type repeat drift {drift:.3e} nats exceeds the fixed '
                         f'{wt_tolerance:.0e} nats tolerance')
    wildtype_score = float(score[0])
    response = score - wildtype_score
    response[0] = 0.0
    return {'name': name, 'group': background['group'], 'length': int(background['length']),
            'wildtype': background['wildtype'], 'wildtype_sha256': background['wildtype_sha256'],
            'full_states': total, 'reused_states': len(old_entries),
            'missing_states': len(new_entries) - 1, 'score': score, 'response': response,
            'wildtype_score': wildtype_score, 'position': position, 'mutant_index': mutant_index,
            'state_sequence_sha256': sha, 'packed_token_counts': packed,
            'scored_token_counts': scored, 'pooled_token_counts': pooled,
            'wildtype_drift_nats': drift}


def merge_background(reference: dict, arm: str, name: str, cell: dict,
                     *, wt_tolerance: float = WT_TOLERANCE_NATS) -> dict:
    """Bind one legacy archive and its new shard cell into a complete grid."""

    background = reference['backgrounds'][name]
    reuse_binding = reference['arms'][arm]['reuse']['backgrounds'][name]
    manifest_row = {row['background']: row
                    for row in reference['arms'][arm]['manifest']['backgrounds']}[name]
    old_path = Path(reference['archive_dir']) / reuse_binding['file']
    old_entries = validate_old_archive(background, reuse_binding, manifest_row,
                                       reference['arms'][arm]['identity'], _read_archive(old_path))
    assembled = assemble_full_grid(background, old_entries, cell['entries'],
                                   wt_tolerance=wt_tolerance)
    if assembled['full_states'] != int(background['full_states']):
        raise ValueError(f'{arm}/{name}: merged grid has {assembled["full_states"]} states, not '
                         f'{background["full_states"]}')
    if assembled['reused_states'] != int(background['reused_states']):
        raise ValueError(f'{arm}/{name}: legacy support is {assembled["reused_states"]} states, not '
                         f'{background["reused_states"]}')
    if assembled['missing_states'] != int(background['missing_states']):
        raise ValueError(f'{arm}/{name}: new support is {assembled["missing_states"]} states, not '
                         f'{background["missing_states"]}')
    assembled.update(arm=arm, legacy_file=reuse_binding['file'], legacy_sha256=reuse_binding['sha256'],
                     new_file=cell['file'], new_sha256=cell['sha256'],
                     new_manifest_path=cell['manifest_path'],
                     new_manifest_sha256=cell['manifest_sha256'])
    return assembled


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def _background_metadata(assembled: dict, arm: str) -> str:
    return json.dumps({
        'schema': MERGE_SCHEMA, 'arm': arm, 'background': assembled['name'],
        'group': assembled['group'], 'length': assembled['length'],
        'wildtype': assembled['wildtype'], 'wildtype_sha256': assembled['wildtype_sha256'],
        'full_states': assembled['full_states'], 'reused_states': assembled['reused_states'],
        'missing_states': assembled['missing_states'], 'response_reduction': 'FULLSUM',
        'likelihood_semantics': PLAN.NATIVE_LIKELIHOOD_SEMANTICS,
        'response_semantics': RESPONSE_SEMANTICS,
        'wildtype_drift_nats': assembled['wildtype_drift_nats'],
        'wildtype_tolerance_nats': WT_TOLERANCE_NATS,
        'legacy': {'file': assembled['legacy_file'], 'sha256': assembled['legacy_sha256']},
        'new': {'file': assembled['new_file'], 'sha256': assembled['new_sha256'],
                'manifest': assembled['new_manifest_path'],
                'manifest_sha256': assembled['new_manifest_sha256']},
        'disclosure': DISCLOSURE,
    }, sort_keys=True)


def write_background(path: Path, assembled: dict, arm: str, *, overwrite: bool = False) -> None:
    """One merged background archive, written atomically and never overwriting silently."""

    if path.exists():
        if not overwrite:
            raise FileExistsError(f'{path}: refusing to overwrite an existing merged archive')
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'score': assembled['score'], 'wildtype_score': np.float64(assembled['wildtype_score']),
        'response': assembled['response'], 'position': assembled['position'],
        'mutant_index': assembled['mutant_index'],
        'state_sequence_sha256': assembled['state_sequence_sha256'],
        'packed_token_counts': assembled['packed_token_counts'],
        'scored_token_counts': assembled['scored_token_counts'],
        'pooled_token_counts': assembled['pooled_token_counts'],
        'metadata': _background_metadata(assembled, arm),
    }
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **payload)
    temp.replace(path)


def _build_manifest(reference: dict, cells: dict, merged: dict, out: Path,
                    wt_tolerance: float) -> dict:
    arms_report, source = {}, {'old_manifests': {}, 'reuse_maps': {}, 'old_archives': {},
                               'new_manifests': {}, 'new_archives': {}}
    totals = [0, 0, 0]
    drifts = {}
    for arm in sorted(reference['arms']):
        arms_report[arm] = {'dtype': reference['arms'][arm]['identity']['dtype'],
                            'batch_size': reference['arms'][arm]['identity']['batch_size'],
                            'budget': reference['arms'][arm]['identity']['budget'],
                            'response_reduction': 'FULLSUM', 'checkpoint': None,
                            'backgrounds': {}, 'validation': {}}
        source['old_manifests'][arm] = {'path': reference['arms'][arm]['manifest_path'],
                                        'sha256': reference['arms'][arm]['manifest_sha256']}
        source['reuse_maps'][arm] = {'sha256': reference['arms'][arm]['reuse_sha']}
        source['old_archives'][arm] = {}
        source['new_manifests'][arm], source['new_archives'][arm] = {}, {}
        arm_total = [0, 0, 0]
        finite = True
        for shard in reference['shards']:
            cell = next(c for c in cells[arm].values() if c['shard_index'] == shard['index'])
            source['new_manifests'][arm][str(shard['index'])] = {
                'path': cell['manifest_path'], 'sha256': cell['manifest_sha256']}
            if arms_report[arm]['checkpoint'] is None:
                arms_report[arm]['checkpoint'] = cell['identity_binding']
            elif arms_report[arm]['checkpoint'] != cell['identity_binding']:
                raise ValueError(f'{arm}: shards disagree on the declared checkpoint identity')
        for name in sorted(reference['backgrounds']):
            record = merged[arm][name]
            arm_total[0] += record['full_states']
            arm_total[1] += record['reused_states']
            arm_total[2] += record['missing_states']
            finite &= bool(np.isfinite(record['score']).all())
            drifts[f'{arm}/{name}'] = record['wildtype_drift_nats']
            source['old_archives'][arm][name] = {'file': record['legacy_file'],
                                                 'sha256': record['legacy_sha256']}
            source['new_archives'][arm][name] = {'file': record['new_file'],
                                                'sha256': record['new_sha256'],
                                                'manifest_sha256': record['new_manifest_sha256']}
            arms_report[arm]['backgrounds'][name] = {
                'group': record['group'], 'length': record['length'],
                'wildtype_sha256': record['wildtype_sha256'], 'full_states': record['full_states'],
                'reused_states': record['reused_states'], 'missing_states': record['missing_states'],
                'wildtype_score_nats': record['wildtype_score'],
                'wildtype_drift_nats': record['wildtype_drift_nats'],
                'sha256': sha256_file(Path(out) / arm / f'{name}.npz')}
        for index in range(3):
            totals[index] += arm_total[index]
        arms_report[arm]['validation'] = {
            'backgrounds_complete': len(reference['backgrounds']),
            'states': arm_total[0], 'reused_states': arm_total[1], 'missing_states': arm_total[2],
            'all_finite': finite,
            'wildtype_max_drift_nats': max(drifts[f'{arm}/{name}']
                                           for name in reference['backgrounds']),
            'wildtype_response_zero': True,
        }
    return {'schema': MERGE_SCHEMA, 'generated_utc': datetime.now(timezone.utc).isoformat(),
            'reference': {'grid_manifest': {'path': reference['grid_path'],
                                            'sha256': reference['grid_sha256']},
                          'grid_schema': reference['grid']['schema'],
                          'cohort': {'path': reference['cohort_path'],
                                     'sha256': reference['cohort_sha256']},
                          'allowlist': {'path': reference['allowlist_path'],
                                        'sha256': reference['allowlist_sha256']},
                          'parent_plan_sha256': reference['grid']['parent_plan']['plan_sha256'],
                          'shards': [{'file': s['file'], 'sha256': s['sha256'],
                                      'plan_sha256': s['plan_sha256'], 'index': s['index'],
                                      'count': s['count'], 'backgrounds': s['backgrounds'],
                                      'missing_states': s['missing_states']}
                                     for s in reference['shards']]},
            'response_reduction': 'FULLSUM',
            'likelihood_semantics': PLAN.NATIVE_LIKELIHOOD_SEMANTICS,
            'response_semantics': RESPONSE_SEMANTICS,
            'disclosure': DISCLOSURE,
            'source_binding': source,
            'arms': arms_report,
            'validation': {'complete': True, 'all_finite': True, 'arms': sorted(reference['arms']),
                           'model_count': len(reference['arms']),
                           'biological_backgrounds': len(reference['backgrounds']),
                           'background_model_records_total':
                               len(reference['arms']) * len(reference['backgrounds']),
                           'states_total': totals[0], 'reused_states_total': totals[1],
                           'missing_states_total': totals[2],
                           'wildtype_tolerance_nats': wt_tolerance,
                           'wildtype_max_drift_nats': max(drifts.values()),
                           'wildtype_drifts_nats': drifts}}


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #

def readiness_report(reference: dict, probe: dict, wt_tolerance: float) -> dict:
    """Explicit readiness: bound reference, missing intake, frozen arithmetic."""

    grid = reference['grid']
    return {'schema': MERGE_SCHEMA, 'status': 'awaiting-new-intake', 'ready': False,
            'generated_utc': datetime.now(timezone.utc).isoformat(),
            'grid_manifest': {'path': reference['grid_path'], 'sha256': reference['grid_sha256']},
            'reference_verified': True, 'arms': sorted(reference['arms']),
            'backgrounds': len(reference['backgrounds']), 'shards': len(reference['shards']),
            'expected': grid['expected'], 'observed': grid['observed'],
            'missing_new_intake': probe['missing'], 'present_new_intake': probe['present'],
            'wildtype_tolerance_nats': wt_tolerance,
            'message': ('the new forward-wave shard manifests are not all present; no merged grid '
                        'was written, so this is readiness, not a completed matrix')}


def run_merge(args) -> dict:
    """Bind the reference, probe readiness, and merge only when the intake exists."""

    wt_tolerance = float(args.wt_tolerance_nats)
    if not np.isfinite(wt_tolerance) or wt_tolerance != WT_TOLERANCE_NATS:
        raise ValueError(f'the wild-type tolerance is the fixed {WT_TOLERANCE_NATS:.0e} nats '
                         'production calibration and cannot be overridden')
    reference = load_reference(args.reference, args.archive_dir)
    roots = raw_roots(reference, args.raw_root, args.raw_maps)
    probe = probe_new_intake(reference, roots)
    if not probe['ready']:
        return readiness_report(reference, probe, wt_tolerance)

    cells = collect_new_cells(reference, roots)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    targets = {arm: {name: out / arm / f'{name}.npz' for name in reference['backgrounds']}
               for arm in reference['arms']}
    manifest_path = out / 'manifest.json'
    if not args.overwrite:
        existing = [str(path) for arm in targets for path in targets[arm].values() if path.exists()]
        if manifest_path.exists():
            existing.append(str(manifest_path))
        if existing:
            raise FileExistsError(f'refusing to overwrite existing merged outputs: {existing[:4]}')

    merged = {}
    for arm in sorted(reference['arms']):
        merged[arm] = {}
        for name in sorted(reference['backgrounds']):
            merged[arm][name] = merge_background(reference, arm, name, cells[arm][name],
                                                 wt_tolerance=wt_tolerance)
    for arm in sorted(merged):
        for name in sorted(merged[arm]):
            write_background(targets[arm][name], merged[arm][name], arm, overwrite=args.overwrite)

    manifest = _build_manifest(reference, cells, merged, out, wt_tolerance)
    declared = reference['grid']['observed']
    for arm, report in sorted(manifest['arms'].items()):
        counts = report['validation']
        if (counts['states'], counts['reused_states'], counts['missing_states'],
                counts['backgrounds_complete']) != (declared['states_per_model'],
                                                    declared['reused_per_model'],
                                                    declared['missing_per_model'],
                                                    declared['backgrounds']):
            raise ValueError(f'{arm}: the merged grid does not realise the frozen arithmetic '
                             f'declared in the grid manifest: {counts}')
    manifest['status'] = 'complete'
    write_json(manifest_path, manifest)
    manifest['output'] = {'directory': str(out), 'manifest': str(manifest_path),
                          'manifest_sha256': sha256_file(manifest_path)}
    return manifest


def _raw_map(text: str) -> tuple[str, str]:
    arm, separator, path = text.partition('=')
    if not separator or not arm or not path:
        raise argparse.ArgumentTypeError('a raw producer root is declared as ARM=PATH')
    return arm, path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--archive-dir', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--raw-root', type=Path, default=DEFAULT_RAW)
    parser.add_argument('--raw-map', type=_raw_map, action='append', default=[], dest='raw_maps',
                        metavar='ARM=PATH', help='override one arm producer root, repeatable')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--wt-tolerance-nats', type=float, default=WT_TOLERANCE_NATS)
    parser.add_argument('--overwrite', action='store_true',
                        help='replace existing merged archives and manifest')
    parser.add_argument('--check-readiness', action='store_true',
                        help='report readiness and exit zero when the new intake is unavailable')
    args = parser.parse_args(argv)
    args.raw_maps = dict(args.raw_maps)
    report = run_merge(args)
    print(json.dumps(report, indent=1, sort_keys=True))
    if report.get('status') != 'complete' and not args.check_readiness:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())