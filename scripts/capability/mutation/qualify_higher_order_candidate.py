#!/usr/bin/env python3
"""Qualify one candidate deposition for an order-k interaction endpoint.

Two stages, so the admitted support is digest-bound before any measured value is
opened. ``declare`` reads the state keys of every declared table, reports the
complete lower-order support and the independent site tuples it rests on, and
writes a declaration whose SHA-256 it prints. ``qualify`` refuses a declaration
whose bytes no longer hash to that digest, re-derives the same cube count at
every order, and only then reads the replicate channels to measure the order-k
noise floor and decide whether the endpoint clears it.

Stop is the default. A candidate that does not clear reports a measurement
limitation and no model scoring is scheduled; the harness raises rather than
returning a permissive record. Measurement-only and CPU-only: no model score,
likelihood or representation is read, and no GPU is scheduled.

    python scripts/capability/mutation/qualify_higher_order_candidate.py declare \
      --candidate escobedo2025 \
      --out archive/logs/shared/higher_order_escobedo_20260924/declaration.json
    python scripts/capability/mutation/qualify_higher_order_candidate.py qualify \
      --candidate escobedo2025 \
      --declaration archive/logs/shared/higher_order_escobedo_20260924/declaration.json \
      --declaration-digest <sha256> \
      --out archive/logs/shared/higher_order_escobedo_20260924/qualification.json
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.mutation import higher_order_qualification as hoq  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.readouts.readout_class_sweep import PROJECTION_BLAS_THREADS  # noqa: E402

#: Declared orders. Two is included because the order-2 case of the same
#: contrast is the endpoint the admitted pairwise experiment measured, so a
#: candidate whose order-2 reading also fails its floor is a statement about the
#: assay and one whose order-2 reading resolves while order three does not is a
#: statement about interaction order -- the distinction the ProteinGym gate
#: turned on.
DECLARED_ORDERS = (2, 3, 4, 5)

#: Group-bootstrap contract, fixed here and carried into every artefact.
BOOTSTRAP_SEED = 20260924
BOOTSTRAP_RESAMPLES = 2000

#: The candidate depositions this harness has been pointed at. Each table names
#: its archive member, its delimiter, its per-replicate channel columns and the
#: channel kind those columns are. The per-replicate ``fitness<i>_uncorr``
#: columns of a DiMSum table are separately grown and separately sequenced
#: repeats of one amino-acid state, which is the ``replicate`` kind; the pooled
#: ``fitness`` column is the label a model comparison would be fitted to and is
#: never a channel.
CANDIDATES: dict[str, dict[str, object]] = {
    'escobedo2025': {
        'title': 'Escobedo, Voigt, Faure and Lehner 2025, combinatorial randomized-core libraries',
        'archive': 'data/randomized_cores_escobedo2025/'
                   'Escobedo_et_al_protein_randomization_data_files.zip',
        'doi': '10.5281/zenodo.16266645',
        'licence': 'CC-BY-4.0',
        'channel_kind': 'replicate',
        'tables': {
            'core_randomization': (
                'zenodo/Fig5/FYN_core_randomization_fitness_replicates.txt', '\t', 3),
            'surface_randomization': (
                'zenodo/Fig5/FYN_surface_randomization_fitness_replicates.txt', '\t', 3),
            'buried_and_exposed_randomization': (
                'zenodo/Fig5/FYN_buried_n_exposed_randomization_fitness_replicates.txt', '\t', 3),
            'sparse_dts_cores': (
                'zenodo/Fig2/Sparse_DTS_Cores_fitness_replicates.txt', '\t', 2),
            'fyn_sh3_cores': (
                'zenodo/Fig3/FYN-SH3/FYN_fitness_replicates.txt', ',', 2),
            'ci2a_cores': (
                'zenodo/FigS4/CI-2A/CI2A_fitness_replicates.txt', ',', 2),
            'fyn_suppressors': (
                'zenodo/Fig6/FYN_suppressor_fitness_replicates.txt', '\t', 3),
        },
    },
}


def channel_columns(n_channels: int) -> list[str]:
    """DiMSum's per-replicate fitness columns, in file order."""
    return [f'fitness{index + 1}_uncorr' for index in range(n_channels)]


def read_table(archive: Path, member: str, delimiter: str) -> list[dict[str, str]]:
    """One archive member as rows, read without extracting the whole archive."""
    with zipfile.ZipFile(archive) as handle:
        with handle.open(member) as raw:
            text = io.TextIOWrapper(raw, encoding='utf-8', newline='')
            return list(csv.DictReader(text, delimiter=delimiter))


def build_blocks(candidate: dict[str, object], archive: Path) -> dict[str, object]:
    """Every declared table's admitted measurement blocks, keyed by table and length."""
    blocks, summary, excluded, members = [], [], {}, {}
    for label, (member, delimiter, n_channels) in sorted(candidate['tables'].items()):
        columns = channel_columns(n_channels)
        rows = read_table(archive, member, delimiter)
        parsed = hoq.dimsum_blocks(rows, channel_columns=columns)
        members[label] = {'member': member, 'rows': len(rows), 'channel_columns': columns}
        for block in parsed['blocks']:
            blocks.append(hoq.CandidateBlock(
                block_id=f'{label}/{block.block_id}', reference=block.reference,
                channels=block.channels, values=block.values))
        for entry in parsed['block_summary']:
            summary.append({'table': label, **entry})
        for key, count in parsed['excluded_rows'].items():
            excluded[key] = excluded.get(key, 0) + count
    return {'blocks': blocks, 'block_summary': summary, 'excluded_rows': excluded,
            'members': members}


def provenance(candidate: dict[str, object], archive: Path) -> dict[str, object]:
    """Source binding and the host resources the stage ran against."""
    usage = shutil.disk_usage(ROOT)
    return {
        'title': candidate['title'],
        'doi': candidate['doi'],
        'licence': candidate['licence'],
        'archive': str(Path(candidate['archive'])),
        'archive_bytes': archive.stat().st_size,
        'archive_sha256': sha256_file(archive),
        'host': {
            'python': sys.version.split()[0],
            'cpu_count': os.cpu_count(),
            'threadpools': [{'user_api': pool.get('user_api'),
                             'internal_api': pool.get('internal_api'),
                             'num_threads': int(pool['num_threads'])}
                            for pool in threadpool_info()],
            'disk_free_gib': round(usage.free / 2 ** 30, 1),
            'disk_used_fraction': round(usage.used / usage.total, 4),
        },
        'reads': ('the aa_seq column and the per-replicate fitness columns of the declared '
                  'members; no fitted free energy, no model score and no GPU'),
    }


def stage_declare(args: argparse.Namespace) -> dict[str, object]:
    candidate = CANDIDATES[args.candidate]
    archive = ROOT / str(candidate['archive'])
    built = build_blocks(candidate, archive)
    declaration = hoq.declare_candidate(
        args.candidate, blocks=built['blocks'], orders=DECLARED_ORDERS,
        channel_kind_name=str(candidate['channel_kind']),
        provenance={**provenance(candidate, archive), 'members': built['members']},
        block_summary=built['block_summary'], excluded_rows=built['excluded_rows'])
    declaration['bootstrap'] = {'seed': BOOTSTRAP_SEED, 'resamples': BOOTSTRAP_RESAMPLES,
                                'unit': 'site tuple'}
    declaration['created_utc'] = datetime.now(timezone.utc).isoformat()
    return declaration


def stage_qualify(args: argparse.Namespace) -> dict[str, object]:
    observed = sha256_file(args.declaration)
    if observed != args.declaration_digest:
        raise SystemExit(
            f'declaration digest {observed} does not match the declared '
            f'{args.declaration_digest}; the support this qualification would be read '
            'against is not the support that was declared before any value was opened')
    declared = json.loads(args.declaration.read_bytes())
    if declared['candidate'] != args.candidate:
        raise SystemExit(f'declaration is for {declared["candidate"]!r}, not {args.candidate!r}')
    candidate = CANDIDATES[args.candidate]
    archive = ROOT / str(candidate['archive'])
    built = build_blocks(candidate, archive)
    orders = tuple(declared['declared_orders'])
    kind = declared['channel']['kind']
    cells, redeclared = [], {}
    for block in built['blocks']:
        states = block.states()
        variable = len({position for key in states.states for position, _ in key})
        inventory = hoq.support_inventory(states, orders=orders, variable_positions=variable)
        redeclared[block.block_id] = inventory
        expected = declared['blocks'].get(block.block_id)
        if expected is None:
            raise SystemExit(f'{block.block_id}: block absent from the declaration')
        for order in orders:
            left = inventory['per_order'][str(order)]['cubes']
            right = expected['per_order'][str(order)]['cubes']
            if left != right:
                raise SystemExit(
                    f'{block.block_id} at order {order}: {left} complete cubes re-derived '
                    f'against {right} declared')
        for order in orders:
            if inventory['per_order'][str(order)]['cubes'] < 1:
                continue
            floor = hoq.order_floor(block, order, seed=BOOTSTRAP_SEED,
                                    resamples=BOOTSTRAP_RESAMPLES)
            cells.append(hoq.clearance(floor, kind))
    if not cells:
        raise SystemExit('no declared order carries a complete cube in any admitted block; '
                         'there is no endpoint to qualify')
    verdict = hoq.candidate_verdict(
        args.candidate, cells, declaration_digest=args.declaration_digest,
        declared_orders=orders, blocks=built['blocks'],
        duplicate_groups=hoq.duplicate_sequence_blocks(built['blocks']),
        notes=[
            'Two pairs of admitted blocks carry identical measured sequences under '
            'different references, so they are one library each rather than two units of '
            'evidence. Where a block carries no wild-type-flagged row the reference is the '
            'modal residue per position, which is the designed library background rather '
            'than the protein\'s wild type: the cycle is then an exact order-k interaction '
            'contrast among substitutions away from that background, and the deposition '
            'does not certify that the block\'s fitness is on the wild-type-normalised '
            'scale.',
            'The fitted free energies this deposition also carries are inferred from these '
            'same fitness measurements by one global fit, so they supply no independent '
            'per-state channel: a cycle computed on them would report the fit\'s own '
            'structure and its uncertainty would be a fit uncertainty. An energy scale '
            'fixes the estimand and not the signal-to-noise ratio, so the floor is '
            'qualified on the measured fitness replicates the energies were fitted to.',
            'The endpoint is the absolute reference-centred order-k cycle in the '
            'deposition\'s own fitness units. Nonadditivity on a growth-rate-derived '
            'fitness scale is not by itself a molecular interaction, and the floor is '
            'measured on the same scale as the endpoint so the comparison is internally '
            'consistent.',
        ])
    verdict['created_utc'] = datetime.now(timezone.utc).isoformat()
    verdict['provenance'] = provenance(candidate, archive)
    verdict['bootstrap'] = declared['bootstrap']
    verdict['redeclared_support'] = redeclared
    verdict['block_summary'] = built['block_summary']
    verdict['shared_sequence_census'] = [
        pair for pair in hoq.shared_sequence_census(built['blocks']) if pair['shared_sequences']]
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=('declare', 'qualify'))
    parser.add_argument('--candidate', required=True, choices=sorted(CANDIDATES))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--declaration', type=Path)
    parser.add_argument('--declaration-digest')
    args = parser.parse_args()
    if (args.stage == 'qualify') != (args.declaration is not None
                                     and args.declaration_digest is not None):
        raise SystemExit('qualify requires --declaration and --declaration-digest, '
                         'and declare takes neither')
    # The BLAS thread count is pinned to the admitted projection's, because a
    # float32 reduction order depends on it and every quantity this programme
    # composes was formed at four threads.
    with threadpool_limits(limits=PROJECTION_BLAS_THREADS, user_api='blas'):
        record = stage_declare(args) if args.stage == 'declare' else stage_qualify(args)
        record['blas_threads'] = PROJECTION_BLAS_THREADS
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, record)
    print(f'{args.stage}: {args.out}')
    print(f'sha256={sha256_file(args.out)}')
    if args.stage == 'qualify':
        print(f'recommendation={record["recommendation"]} '
              f'schedule_model_scoring={record["schedule_model_scoring"]}')


if __name__ == '__main__':
    main()
