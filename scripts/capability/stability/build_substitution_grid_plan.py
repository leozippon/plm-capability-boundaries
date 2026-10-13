#!/usr/bin/env python3
"""Prepare a label-blind missing-state plan for the 12 reviewed substitution grids.

This is preparation, not measurement. It enumerates the exact wild type plus all
19 non-wild-type canonical residues at every position of the 12 human/mouse
intracellular-domain backgrounds the supplied allowlist admits, verifies that the
already-scored native-likelihood archives cover exactly the wild type and the 256
frozen cohort single mutants of each background, and writes a
``stability_singles_extraction_v1`` plan carrying only the still-missing states
plus the wild type the extraction contract requires.

Nothing here loads a model, touches a GPU or reads a measured stability value.
The plan carries sequences, mutated positions and state indices only, so the
forward pass it drives stays label-blind by construction. The plan is consumed
unchanged by ``scripts/capability/stability/extract_stability_singles.py
--likelihood-only --plan ... --expect-plan-sha256 ... --arm ... --device cuda:0
--out ...``.

The reuse identity is re-derived, never assumed from counts: the coalesced
cohort digest, the two arms on the frozen roster, the likelihood-only schema and
its ``native packed residue-span summed log likelihood, nats, per state``
semantics, singleton batch, the 1024-token budget, the float32 archive dtype and
the exact ``state_sequence_sha256`` -> mutation map of every reused state are all
checked against the archives the manifest names, and the plan builder fails to
the caller rather than reinterpreting a contradictory count.

Reusable likelihoods are FULLSUM native per-state sums. They are archived
verbatim and are not a probability; a segmentation difference or a disagreement
between quantities is not read here as a biological error.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.amino_acids import AA20
from src.capability.core.io import sha256_file, write_json
from src.capability.extensions.stability_followups import verify_scalar_archive
from src.capability.interactions.pairwise_epistasis import (
    ARM_DTYPE, FEATURE_BLOCKS, PRODUCTION_BATCH_SIZE, PROJECTION_DIM, PROJECTION_SEED,
    ROSTER)
from src.capability.stability.measured_selection import load_allowlist
from src.capability.stability.stability_gate import (
    validate_likelihood_token_support, validate_plan_background)


def _extractor_contract():
    """The extraction entry point, imported so the plan digest cannot drift."""

    path = ROOT / 'scripts/capability/stability/extract_stability_singles.py'
    spec = importlib.util.spec_from_file_location('stability_extractor_contract', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'the extraction contract is unavailable at {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXTRACTOR = _extractor_contract()
SCHEMA = EXTRACTOR.SCHEMA
plan_digest = EXTRACTOR.plan_digest

#: Written artefacts; each is a separate schema so a partial preparation cannot
#: be read as a different one.
GRID_SCHEMA = 'stability_substitution_grid_manifest_v1'
REUSE_SCHEMA = 'stability_substitution_grid_reuse_map_v1'
REPORT_SCHEMA = 'stability_substitution_grid_preparation_report_v1'

BUDGET = 1024
DEFAULT_SHARDS = 7

#: The reviewed panel and its verified full-grid arithmetic. These are the
#: values the preparation must reproduce; a contradiction is refused, not fitted.
EXPECTED_BACKGROUNDS = 12
EXPECTED_STATES_PER_MODEL = 12894
EXPECTED_REUSED_PER_MODEL = 3084
EXPECTED_MISSING_PER_MODEL = 9810

NATIVE_LIKELIHOOD_SEMANTICS = 'native packed residue-span summed log likelihood, nats, per state'


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# --------------------------------------------------------------------------- #
# Pure grid construction. One state per (position, mutant), wild type first.
# --------------------------------------------------------------------------- #

def substitution_grid(wildtype: str) -> list[dict]:
    """The exact wild type plus all 19 non-wild-type canonical residues per site.

    The order is fixed -- position ascending, then residue in :data:`AA20`
    alphabetical order skipping the wild-type residue -- so a state index is a
    stable function of the background and the mutation alone. ``state`` numbers
    the full grid with the wild type at zero; ``plan_row`` renumbers the surviving
    states separately.
    """

    grid = []
    for index, residue in enumerate(wildtype):
        for candidate in AA20:
            if candidate == residue:
                continue
            sequence = wildtype[:index] + candidate + wildtype[index + 1:]
            grid.append({'position': index + 1, 'mutant': candidate, 'sequence': sequence,
                         'sha256': sha256_text(sequence), 'state': len(grid) + 1})
    return grid


def grid_background(cohort_row: dict) -> dict:
    """The arm-independent full grid of one frozen cohort background."""

    wildtype = cohort_row['wildtype']
    if len(wildtype) != int(cohort_row['length']):
        raise ValueError(f"{cohort_row['name']}: declared length disagrees with the wild type")
    if set(wildtype) - set(AA20):
        raise ValueError(f"{cohort_row['name']}: wild type is not entirely canonical residues")
    return {'name': cohort_row['name'], 'group': cohort_row['group'],
            'length': len(wildtype), 'wildtype': wildtype,
            'wildtype_sha256': sha256_text(wildtype),
            'substitutions': substitution_grid(wildtype)}


def plan_row(background: dict, missing: list[dict]) -> dict:
    """One schema-valid background row carrying the wild type and missing states."""

    sequences = [background['wildtype']] + [state['sequence'] for state in missing]
    variants = [{'position': state['position'], 'mutant': state['mutant'], 'state': index + 1}
                for index, state in enumerate(missing)]
    row = {'name': background['name'], 'group': background['group'],
           'length': background['length'], 'wildtype': background['wildtype'],
           'sequences': sequences, 'variants': variants}
    validate_plan_background(row)
    return row


def shard_backgrounds(backgrounds: list[dict], shard_count: int, missing_counts: dict) -> list[list[dict]]:
    """Balanced, deterministic, family-whole shards of the plan backgrounds.

    Backgrounds are assigned largest missing-state load first, ties by name, each
    to the currently lightest shard. One background carries exactly one reviewed
    family, so no shard splits a family. The result is a function of the frozen
    grid alone.
    """

    if shard_count < 1 or shard_count > len(backgrounds):
        raise ValueError('shard count must be between one and the number of backgrounds')
    ordered = sorted(backgrounds, key=lambda row: (-missing_counts[row['name']], row['name']))
    shards = [[] for _ in range(shard_count)]
    totals = [0] * shard_count
    for background in ordered:
        index = min(range(shard_count), key=lambda candidate: (totals[candidate], candidate))
        shards[index].append(background)
        totals[index] += missing_counts[background['name']]
    return shards


def validate_grid_partition(backgrounds: list[dict], bindings: dict) -> None:
    """Reused and missing states must partition each grid with no shared state.

    The overlap check is global: a state hash appearing in two backgrounds would
    silently pool two families, so it is refused rather than counted.
    """

    seen: dict[str, str] = {}
    for background in backgrounds:
        name = background['name']
        full = {background['wildtype_sha256']} | {s['sha256'] for s in background['substitutions']}
        if len(full) != 1 + len(background['substitutions']):
            raise ValueError(f'{name}: the substitution grid repeats a state')
        binding = bindings[name]
        reused = {entry['sha256'] for entry in binding['reused']}
        missing = {state['sha256'] for state in binding['missing']}
        if reused & missing or reused | missing != full:
            raise ValueError(f'{name}: reused and missing states do not partition the full grid')
        for digest in full:
            if digest in seen:
                raise ValueError(f'state hash shared by {seen[digest]} and {name}')
            seen[digest] = name


# --------------------------------------------------------------------------- #
# Reuse verification. Identity, protocol and state binding are re-derived.
# --------------------------------------------------------------------------- #

def archive_identity_pin(manifest: dict, arm: str, cohort_sha256: str) -> dict:
    """Bind one manifest to the reusable native-likelihood protocol, or refuse."""

    identity = manifest.get('identity')
    if not isinstance(identity, dict):
        raise ValueError(f'{arm}: archive manifest carries no identity block')
    expected = {'schema': EXTRACTOR.LIKELIHOOD_SCHEMA, 'arm': arm,
                'dtype': ARM_DTYPE.get(arm, 'float32'),
                'batch_size': PRODUCTION_BATCH_SIZE, 'budget': BUDGET,
                'cohort_sha256': cohort_sha256,
                'likelihood_semantics': NATIVE_LIKELIHOOD_SEMANTICS}
    wrong = {key: [identity.get(key), value] for key, value in expected.items()
             if identity.get(key) != value}
    if manifest.get('status') != 'complete' or wrong or manifest.get('blas') is not None:
        raise ValueError(f'{arm}: archive is not the reusable native protocol: '
                         f'{wrong or "status/blas"}')
    return identity


def checkpoint_binding(identity: dict) -> dict:
    """The declared checkpoint identity, with what was actually verified disclosed.

    The checkpoint lives on the H200 filesystem, so this host can usually only
    re-read the declaration the archives embed. When the path is present the
    declared metadata file hashes and tensor file sizes are verified; tensor
    weights are never content-hashed here, so size equality is not weight-byte
    identity. When the path is absent that limit is recorded instead of being
    reported as a pass.
    """

    checkpoint = Path(identity['checkpoint_path'])
    binding = {'declared_path': str(checkpoint),
               'tensor_files': identity['checkpoint_tensor_files'],
               'metadata_sha256': identity['checkpoint_metadata_sha256'],
               'metadata_and_tensor_sizes_verified_on_host': False,
               'weight_byte_identity_verified': False,
               'verification': ('declared identity only; checkpoint path absent from this host, '
                                'so no metadata hash, tensor size or weight byte was checked')}
    if checkpoint.is_dir():
        for name, expected in identity['checkpoint_metadata_sha256'].items():
            path = checkpoint / name
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f'checkpoint metadata {name} is absent or differs from its hash')
        for item in identity['checkpoint_tensor_files']:
            path = checkpoint / item['name']
            if not path.is_file() or path.stat().st_size != item['bytes']:
                raise ValueError(f"checkpoint tensor {item['name']} is absent or its size differs")
        binding['metadata_and_tensor_sizes_verified_on_host'] = True
        binding['verification'] = ('declared metadata file hashes and tensor file sizes verified on '
                                   'this host; tensor weights were not content-hashed, so size '
                                   'equality does not establish weight-byte identity')
    return binding


def verify_and_bind_background(background: dict, cohort_row: dict, manifest_row: dict,
                               archive_path: Path, identity: dict, arm: str) -> dict:
    """Reuse every archived state of one background, keyed by exact sequence hash.

    The archive is checked to be the frozen cohort's own state order, to embed the
    manifest's protocol identity, to carry a valid ragged token layout and to map
    each archived state hash to the mutation the cohort declared at that state.
    Only then is a state admitted as reused; the remainder of the full grid is the
    missing set.
    """

    name = background['name']
    if manifest_row.get('background') != name or manifest_row.get('group') != background['group']:
        raise ValueError(f'{arm}/{name}: manifest row names a different background or group')
    expected_states = len(cohort_row['sequences'])
    expected_variants = len(cohort_row['variants'])
    if (manifest_row.get('sequences') != expected_states
            or manifest_row.get('variants') != expected_variants):
        raise ValueError(f'{arm}/{name}: manifest support is not the frozen cohort support')
    if sha256_file(archive_path) != manifest_row.get('sha256'):
        raise ValueError(f'{arm}/{name}: archive bytes differ from the manifest digest')
    substitution_by_digest = {state['sha256']: state for state in background['substitutions']}
    if len(substitution_by_digest) != len(background['substitutions']):
        raise ValueError(f'{name}: the substitution grid repeats a sequence')
    with np.load(archive_path, allow_pickle=False) as data:
        validate_likelihood_token_support(data, expected_states, budget=BUDGET)
        metadata = json.loads(str(data['metadata']))
        if metadata.get('identity') != identity:
            raise ValueError(f'{arm}/{name}: archive embeds a different protocol identity')
        if (metadata.get('background'), metadata.get('group'), metadata.get('sequences'),
                metadata.get('variants')) != (name, background['group'], expected_states,
                                              expected_variants):
            raise ValueError(f'{arm}/{name}: archive metadata names a different background')
        observed = data['state_sequence_sha256']
        expected = np.asarray([sha256_text(sequence) for sequence in cohort_row['sequences']],
                              dtype='<U64')
        if observed.dtype != np.dtype('<U64') or not np.array_equal(observed, expected):
            raise ValueError(f'{arm}/{name}: archive state hashes do not match the frozen cohort order')
        variant_states = np.asarray(data['variant_states'])
        variant_positions = np.asarray(data['variant_positions'])
        declared_states = np.asarray([int(v['state']) for v in cohort_row['variants']])
        declared_positions = np.asarray([int(v['position']) for v in cohort_row['variants']])
        if (not np.array_equal(variant_states, declared_states)
                or not np.array_equal(variant_positions, declared_positions)):
            raise ValueError(f'{arm}/{name}: archive variant states or positions differ from the cohort')
        if not np.array_equal(declared_states, np.arange(1, expected_states)):
            raise ValueError(f'{arm}/{name}: cohort variants are not exactly the non-wild-type states')
        declared_by_state = {int(v['state']): v for v in cohort_row['variants']}
        reused, seen = [], set()
        for archive_index, digest in enumerate(observed.tolist()):
            if digest in seen:
                raise ValueError(f'{arm}/{name}: archive repeats a state hash')
            seen.add(digest)
            if archive_index == 0:
                if digest != background['wildtype_sha256']:
                    raise ValueError(f'{arm}/{name}: archive state zero is not the wild type')
                reused.append({'sha256': digest, 'archive_index': 0, 'role': 'wildtype',
                               'position': 0, 'mutant': None, 'grid_state': 0})
                continue
            substitution = substitution_by_digest.get(digest)
            if substitution is None:
                raise ValueError(f'{arm}/{name}: archive state {archive_index} is outside the full grid')
            declared = declared_by_state.get(archive_index)
            if (declared is None or substitution['position'] != int(declared['position'])
                    or substitution['mutant'] != declared['mutant']):
                raise ValueError(f'{arm}/{name}: archive state {archive_index} maps to a different '
                                 'mutation than the cohort declared')
            reused.append({'sha256': digest, 'archive_index': archive_index,
                           'role': 'measured_single', 'position': substitution['position'],
                           'mutant': substitution['mutant'], 'grid_state': substitution['state']})
        reused_digests = {entry['sha256'] for entry in reused}
        missing = [state for state in background['substitutions']
                   if state['sha256'] not in reused_digests]
        if len(reused) + len(missing) != 1 + len(background['substitutions']):
            raise ValueError(f'{arm}/{name}: reuse and missing states do not partition the grid')
    return {'file': manifest_row['file'], 'sha256': manifest_row['sha256'],
            'reused': reused, 'missing': missing}


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #

def _arm_manifest(text: str) -> tuple[str, Path]:
    arm, separator, path = text.partition('=')
    if not separator or not arm or not path:
        raise argparse.ArgumentTypeError('an archive manifest is declared as ARM=PATH')
    if arm not in ROSTER:
        raise argparse.ArgumentTypeError(f'{arm} is not on the frozen roster')
    return arm, Path(path)


def _fingerprint(text: str) -> str:
    lowered = text.strip().lower()
    if len(lowered) != 64 or any(character not in '0123456789abcdef' for character in lowered):
        raise argparse.ArgumentTypeError('a declared digest is 64 hexadecimal characters')
    return lowered


def prepare(args) -> dict:
    """Verify the reviewed assets and write the plan, shard and reuse artefacts."""

    cohort_sha256 = sha256_file(args.cohort)
    if cohort_sha256 != args.expect_cohort_sha256:
        raise SystemExit(f'cohort digest {cohort_sha256} does not match the declared value')
    cohort = json.loads(args.cohort.read_bytes())
    if cohort.get('schema') != 'stability_singles_cohort_v1':
        raise SystemExit('the frozen cohort is not the expected schema')

    allowlist_sha256 = sha256_file(args.allowlist)
    if allowlist_sha256 != args.expect_allowlist_sha256:
        raise SystemExit(f'allowlist digest {allowlist_sha256} does not match the declared value')
    allowlist = load_allowlist(args.allowlist, cohort, root=ROOT)
    names = [row['name'] for row in allowlist['backgrounds']]
    if len(names) != EXPECTED_BACKGROUNDS:
        raise SystemExit(f'the reviewed allowlist names {len(names)} backgrounds, not '
                         f'{EXPECTED_BACKGROUNDS}')

    cohort_by_name = {row['name']: row for row in cohort['backgrounds']}
    backgrounds = []
    for name in names:
        if name not in cohort_by_name:
            raise SystemExit(f'{name}: the reviewed allowlist names a background outside the cohort')
        backgrounds.append(grid_background(cohort_by_name[name]))
    full_states = sum(1 + len(background['substitutions']) for background in backgrounds)
    if full_states != EXPECTED_STATES_PER_MODEL:
        raise SystemExit(f'the enumerated grid has {full_states} states per model, not '
                         f'{EXPECTED_STATES_PER_MODEL}')

    archive_dir = args.archive_dir.resolve()
    reuse, provenance, reused_total = {}, {}, None
    for arm, manifest_path in args.archive_manifests:
        if manifest_path.resolve() != (archive_dir / f'manifest_{arm}.json').resolve():
            raise SystemExit(f'{arm}: the manifest must be {archive_dir}/manifest_{arm}.json '
                             'for the existing archive validator to bind it')
        verify_scalar_archive(archive_dir, arm, cohort, cohort_sha256)
        manifest = json.loads(manifest_path.read_bytes())
        identity = archive_identity_pin(manifest, arm, cohort_sha256)
        by_name = {row['background']: row for row in manifest['backgrounds']}
        bindings = {}
        for background in backgrounds:
            row = by_name.get(background['name'])
            if row is None:
                raise SystemExit(f"{arm}: no archive for {background['name']}")
            bindings[background['name']] = verify_and_bind_background(
                background, cohort_by_name[background['name']], row,
                archive_dir / row['file'], identity, arm)
        counts = sum(len(binding['reused']) for binding in bindings.values())
        if reused_total is None:
            reused_total = counts
        elif counts != reused_total:
            raise SystemExit(f'{arm}: {counts} reused states against {reused_total} for another arm')
        reuse[arm] = bindings
        provenance[arm] = {
            'manifest': {'path': str(manifest_path), 'sha256': sha256_file(manifest_path)},
            'identity': identity,
            'checkpoint': checkpoint_binding(identity),
            'archives': {name: {'file': binding['file'], 'sha256': binding['sha256']}
                         for name, binding in bindings.items()},
        }

    if reused_total != EXPECTED_REUSED_PER_MODEL:
        raise SystemExit(f'the archives reuse {reused_total} states per model, not '
                         f'{EXPECTED_REUSED_PER_MODEL}')
    missing_total = full_states - reused_total
    if missing_total != EXPECTED_MISSING_PER_MODEL:
        raise SystemExit(f'the missing set is {missing_total} states per model, not '
                         f'{EXPECTED_MISSING_PER_MODEL}')

    reference = next(iter(reuse))
    for arm in reuse:
        for name in reuse[arm]:
            if [entry['sha256'] for entry in reuse[arm][name]['reused']] \
                    != [entry['sha256'] for entry in reuse[reference][name]['reused']]:
                raise SystemExit(f'{arm}/{name}: reused state order differs from {reference}')
    validate_grid_partition(backgrounds, reuse[reference])

    parent_rows = [plan_row(background, reuse[reference][background['name']]['missing'])
                   for background in backgrounds]
    projection = {'seed': PROJECTION_SEED, 'dim': PROJECTION_DIM, 'blocks': list(FEATURE_BLOCKS)}
    parent_plan = {'schema': SCHEMA, 'cohort_sha256': cohort_sha256, 'projection': projection,
                   'backgrounds': parent_rows,
                   'summary': {'backgrounds': len(parent_rows),
                               'states': sum(len(row['sequences']) for row in parent_rows),
                               'wildtype_repeat_states': len(parent_rows),
                               'genuinely_missing_states': missing_total}}
    parent_digest = plan_digest(parent_plan)

    missing_counts = {name: len(reuse[reference][name]['missing']) for name in reuse[reference]}
    shards = shard_backgrounds(backgrounds, args.shards, missing_counts)
    contract = {'alphabet': AA20,
                'substitution_rule': 'exact wild type plus all 19 non-wild-type canonical residues at every position',
                'dtype': {arm: ARM_DTYPE.get(arm, 'float32') for arm in reuse},
                'batch_size': PRODUCTION_BATCH_SIZE, 'budget': BUDGET,
                'native_format': EXTRACTOR.LIKELIHOOD_SCHEMA,
                'response_reduction': 'FULLSUM',
                'likelihood_semantics': NATIVE_LIKELIHOOD_SEMANTICS,
                'likelihood_semantics_binding': 'declared by the archive identity and producer protocol; not independently re-derived from count arrays',
                'token_closure': ('ragged layout and count bounds only: packed counts equal the '
                                  'token-offset widths, offsets are monotone and within the 1024 '
                                  'budget, and pooled <= scored <= packed; the native scored-span '
                                  'boundaries and FULLSUM reduction are bound by the archive '
                                  'identity, not re-derived from these counts'),
                'state_sequence_sha256_semantics': 'SHA256 of exact UTF-8 sequence bytes, <U64 vector in plan sequence order',
                'wildtype_contract': 'the exact wild type is the first state of every background; it is required by the extraction plan validator and is a repeat check, not a genuinely missing state'}
    shard_files, cells = [], []
    for index, shard in enumerate(shards, start=1):
        rows = [plan_row(background, reuse[reference][background['name']]['missing'])
                for background in shard]
        plan = {'schema': SCHEMA, 'cohort_sha256': cohort_sha256, 'projection': projection,
                'backgrounds': rows,
                'shard': {'parent_plan_sha256': parent_digest, 'index': index,
                          'count': len(shards),
                          'backgrounds': [background['name'] for background in shard],
                          'missing_states': sum(missing_counts[b['name']] for b in shard)},
                'contract': contract}
        path = args.out / f'plan_shard_{index:02d}of{len(shards):02d}.json'
        write_json(path, plan)
        digest = plan_digest(plan)
        shard_files.append({'file': path.name, 'sha256': sha256_file(path),
                            'plan_sha256': digest,
                            'backgrounds': [background['name'] for background in shard],
                            'missing_states': sum(missing_counts[b['name']] for b in shard)})
        for arm in reuse:
            cells.append({'arm': arm, 'shard': index, 'plan_file': path.name,
                          'plan_sha256': digest,
                          'backgrounds': [background['name'] for background in shard],
                          'missing_states': sum(missing_counts[b['name']] for b in shard),
                          'gpu_slot': index - 1,
                          'device_hint': '--device cuda:0 with CUDA_VISIBLE_DEVICES set to this gpu_slot',
                          'out_dir_hint': f'{arm}/shard_{index:02d}of{len(shards):02d}'})

    grid_backgrounds = []
    for background in backgrounds:
        binding = reuse[reference][background['name']]
        missing = binding['missing']
        grid_backgrounds.append({
            'name': background['name'], 'group': background['group'],
            'length': background['length'], 'wildtype': background['wildtype'],
            'wildtype_sha256': background['wildtype_sha256'],
            'full_states': 1 + len(background['substitutions']),
            'reused_states': len(binding['reused']),
            'missing_states': len(missing),
            'missing': [{'position': state['position'], 'mutant': state['mutant'],
                         'grid_state': state['state'], 'plan_state': index + 1,
                         'sha256': state['sha256']} for index, state in enumerate(missing)],
        })
    grid = {
        'schema': GRID_SCHEMA,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'cohort': {'path': str(args.cohort), 'sha256': cohort_sha256},
        'allowlist': {'path': str(args.allowlist), 'sha256': allowlist_sha256,
                      'backgrounds': len(names), 'schema': allowlist['schema'],
                      'reviewed_utc': allowlist['review'].get('reviewed_utc')},
        'alphabet': AA20,
        'substitution_rule': contract['substitution_rule'],
        'expected': {'backgrounds': EXPECTED_BACKGROUNDS,
                     'states_per_model': EXPECTED_STATES_PER_MODEL,
                     'reused_per_model': EXPECTED_REUSED_PER_MODEL,
                     'missing_per_model': EXPECTED_MISSING_PER_MODEL},
        'observed': {'backgrounds': len(backgrounds), 'states_per_model': full_states,
                     'reused_per_model': reused_total, 'missing_per_model': missing_total,
                     'wildtype_repeat_states_per_model': len(backgrounds)},
        'parent_plan': {'file': 'plan.json', 'plan_sha256': parent_digest},
        'shards': {'count': len(shards),
                   'policy': 'largest missing-state load first, ties by name, each family assigned to the lightest shard; one background is one reviewed family',
                   'files': shard_files},
        'cells': cells,
        'h200': {
            'allocation': 'one seven-GPU allocation; 14 cells in two waves of seven, one cell per GPU, each process pinned with CUDA_VISIBLE_DEVICES to its gpu_slot',
            'no_background_limit': 'each shard plan covers all of its own backgrounds, so --background-limit is never used as sharding',
            'extractor': 'scripts/capability/stability/extract_stability_singles.py',
        },
        'reuse': {'response_reduction': 'FULLSUM',
                  'semantics': 'each reused state keeps its native packed residue-span summed log likelihood in nats; it is never converted to a per-token mean',
                  'likelihood_semantics_binding': 'declared by the archive identity and producer protocol; not independently re-derived from count arrays',
                  'token_closure': 'ragged layout and count bounds only (packed == offset widths, pooled <= scored <= packed)',
                  'token_closure_validator': 'src.capability.stability.stability_gate.validate_likelihood_token_support',
                  'archives_immutable': 'archive bytes are re-hashed against the manifest digest before reuse and are never written by this preparation',
                  'disclosure': 'a score is not a probability and a BPE or category-path difference is not read here as a biological error'},
        'provenance': {'arms': provenance},
        'limitations': ['preparation only: no model inference, no GPU and no label-conditioned quantity is formed',
                        'the checkpoint declaration is re-read from the archives; when the declared path is present this host verifies metadata file hashes and tensor file sizes only, never weight content, so size equality is not weight-byte identity'],
        'backgrounds': grid_backgrounds,
    }
    write_json(args.out / 'plan.json', parent_plan)
    for arm, bindings in reuse.items():
        map_record = {
            'schema': REUSE_SCHEMA, 'arm': arm, 'cohort_sha256': cohort_sha256,
            'parent_plan_sha256': parent_digest,
            'manifest': provenance[arm]['manifest'],
            'identity': provenance[arm]['identity'],
            'response_reduction': 'FULLSUM',
            'likelihood_semantics': NATIVE_LIKELIHOOD_SEMANTICS,
            'likelihood_semantics_binding': 'declared by the archive identity and producer protocol; not independently re-derived from count arrays',
            'token_closure': ('ragged layout and count bounds only: packed counts equal the '
                              'token-offset widths, offsets are monotone and within the 1024 '
                              'budget, and pooled <= scored <= packed'),
            'backgrounds': {name: {'file': binding['file'], 'sha256': binding['sha256'],
                                   'reused': binding['reused']}
                            for name, binding in bindings.items()},
        }
        write_json(args.out / f'reuse_map_{arm}.json', map_record)
    write_json(args.out / 'grid_manifest.json', grid)
    report = {
        'schema': REPORT_SCHEMA, 'generated_utc': grid['generated_utc'],
        'cohort_sha256': cohort_sha256, 'allowlist_sha256': allowlist_sha256,
        'arms': sorted(reuse), 'backgrounds': len(backgrounds),
        'states_per_model': full_states, 'reused_per_model': reused_total,
        'missing_per_model': missing_total, 'shards': len(shards),
        'cells': len(cells), 'parent_plan_sha256': parent_digest,
        'shard_files': shard_files, 'out': str(args.out),
    }
    write_json(args.out / 'report.json', report)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--expect-cohort-sha256', type=_fingerprint, required=True)
    parser.add_argument('--allowlist', type=Path, required=True)
    parser.add_argument('--expect-allowlist-sha256', type=_fingerprint, required=True)
    parser.add_argument('--archive-dir', type=Path, required=True,
                        help='the one directory holding manifest_<arm>.json and the archives it names')
    parser.add_argument('--archive-manifest', type=_arm_manifest, action='append', required=True,
                        metavar='ARM=PATH', dest='archive_manifests',
                        help='one reusable arm manifest, repeatable')
    parser.add_argument('--out', type=Path, required=True,
                        help='an ignored preparation directory, never a measured results tree')
    parser.add_argument('--shards', type=int, default=DEFAULT_SHARDS,
                        help='GPU-family shards per model; the plan is never truncated with --background-limit')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    report = prepare(args)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()