#!/usr/bin/env python3
"""The local-context gate: qualify a mutation-centred local control, then use it.

Two stages, in this order and never the other.

``--stage qualify`` fits the declared candidate local blocks against the two
baselines they augment on the frozen 201-assay Readout anchor. Every design in
that stage is built from sequence and alignment statistics alone, so no model
quantity reaches any fitted number and a candidate cannot be tuned against the
increment it is later meant to absorb. ``--collect --stage qualify`` then applies
the declared gate across the three split seeds and names which candidates are
carried forward and which are discarded.

``--stage measure`` fits the likelihood and representation increments over the
carried-forward local control, over composition, and over composition plus the
mutation-local profile block, for one arm at one split seed, on the identical
support, folds, weighting and label budget the crossed-controls study used.
``--collect --stage measure`` binds the measurement cells to one another and to
the qualification receipt.

The panel loader, the representation projection, the composition descriptors,
the mutation-local profile block, the profile-store binding, the anchor-support
refusal, the nested folds, the ridge recipe and the cluster-bootstrap intervals
are all imported from the admitted Readout analysis and from the crossed-controls
entry point, so these numbers compose with that study's tables directly.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/capability'))
from scripts.capability.readouts.analyse_readout import load_panel  # noqa: E402  the admitted panel loader, reused unchanged
from src.capability.context.crossed_controls import profile_features, tokenisation_features  # noqa: E402
from src.capability.models.fitness import parse_mutant  # noqa: E402
from src.capability.core.io import write_json  # noqa: E402
from src.capability.context.local_context import (  # noqa: E402
    ADDITIONS, CANDIDATE_BLOCKS, SPLIT_SEEDS, candidate_blocks, control_sets, declaration,
    declaration_sha256, evaluate_local_context, fold_membership, qualification_gate)
from src.capability.context.profiles import profile_scores as lookup_scores  # noqa: E402
from src.capability.readouts.readout_extraction import (  # noqa: E402
    load_readout_arm, pack_sequence, representation_positions)

#: The crossed-controls entry point, loaded as a module so that the profile-store
#: binding, the anchor-support refusal and the packed-token budget have one
#: source rather than a second copy. Its bytes are recorded in every report, so a
#: change to it is visible in the artifact rather than silent.
CROSSED_SCRIPT = ROOT / 'scripts/capability/context/analyse_crossed_controls.py'
_spec = importlib.util.spec_from_file_location('crossed_controls_cli', CROSSED_SCRIPT)
if _spec is None or _spec.loader is None:
    raise RuntimeError('the crossed-controls entry point could not be loaded')
crossed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crossed)

#: The arm whose complete extraction manifest defines the anchor panel, and the
#: arm the qualification stage reads its support and labels from. Its own model
#: quantities are loaded and discarded there: the qualification designs contain
#: no model column.
DEFAULT_ANCHOR_ARM = 'progen3-3b'

#: Every module whose bytes determine a fitted number. The measurement cells are
#: required to carry the same hashes as the qualification cells they are read
#: against: a gate qualified under one feature or fitting implementation cannot
#: license a measurement made under another, and several agents commit into this
#: checkout, so the condition has to be checked rather than assumed.
FITTED_CODE_FILES = ('scripts/capability/context/analyse_local_context_gate.py',
                     'scripts/capability/context/analyse_crossed_controls.py',
                     'scripts/capability/readouts/analyse_readout.py',
                     'src/capability/context/local_context.py', 'src/capability/context/crossed_controls.py',
                     'src/capability/readouts/readout_analysis.py', 'src/capability/context/profiles.py',
                     'src/capability/context/profile_increment.py', 'src/capability/core/protein_properties.py',
                     'src/capability/core/amino_acids.py')

#: The interface-descriptor module is recorded but not required to match: it
#: reaches the tokenisation stratum and no fitted column.
CODE_FILES = (*FITTED_CODE_FILES, 'src/capability/readouts/readout_extraction.py')


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def build_blocks(rows: list[dict], cohort: dict, store: dict, local_blocks: list[str]) -> float:
    """Attach the profile block and the named local blocks to the panel rows.

    Returns the largest deviation between the cohort's retained profile scores
    and the scores recomputed from the retained column-frequency arrays, which
    binds the profile block to the arrays the cohort's own scores were built
    from rather than to a separately estimated profile.
    """
    cohort_map = {row['assay']: row for row in cohort['assays']}
    deviations = []
    for row in rows:
        record = cohort_map[row['assay']]
        wildtype, sequences = record['wildtype'], list(record['sequences'])
        if row['mutants'] != record['mutants'] or len(sequences) != len(row['mutants']):
            raise ValueError(f"{row['assay']}: cohort mutation order or sequence count mismatch")
        substitutions = [parse_mutant(mutant) for mutant in row['mutants']]
        profile = store['profile_from_store'](store['stored'], record['wildtype_id'], wildtype,
                                              store['metadata'])
        recomputed = lookup_scores(profile, store['background'], substitutions, alpha=store['alpha'])
        deviations.append(float(np.abs(recomputed - np.asarray(row['P'], dtype=np.float64)).max()))
        row['P_block'] = profile_features(
            wildtype, row['mutants'], sequences, row['P'], profile.frequencies, store['background'],
            store['metadata']['profiles'][record['wildtype_id']], alpha=store['alpha'])
        blocks = candidate_blocks(wildtype, row['mutants'], sequences)
        for name in local_blocks:
            row[name] = blocks[name]
    deviation = max(deviations)
    if deviation > crossed.PROFILE_SCORE_TOLERANCE:
        raise ValueError(f'recomputed profile scores deviate by {deviation} from the cohort values')
    return deviation


def build_interface(rows: list[dict], cohort: dict, arm_name: str) -> dict:
    """Measure the arm's segmentation, and check it against the production packing.

    The declared registry label and the measured segmentation are reported
    separately, because a registry label of ``bpe`` can coexist with exactly one
    token per residue -- Galactica's protein-delimiter rendering does -- and it is
    the measured quantity a segmentation attribution turns on. The counted tokens
    are exactly the mean-pooling denominators the production extraction used, and
    the recomputed maximum packed-token count is required to equal the value the
    production extraction manifest declares.
    """
    cohort_map = {row['assay']: row for row in cohort['assays']}
    handle = load_readout_arm(arm_name, None, device=None)
    descriptors, leading, trailing = [], [], []
    for row in rows:
        record = cohort_map[row['assay']]
        wildtype, sequences = record['wildtype'], list(record['sequences'])
        states = [wildtype] + sequences
        packed = [pack_sequence(handle, sequence) for sequence in states]
        observed = max(len(entry[0]) for entry in packed)
        if observed != row['max_packed_tokens']:
            raise ValueError(f"{row['assay']}: recomputed packed maximum {observed} differs from "
                             f"the manifest value {row['max_packed_tokens']}")
        if observed > crossed.BUDGET:
            raise ValueError(f"{row['assay']}: recomputed packing exceeds the frozen budget")
        selected = representation_positions(handle, states, packed)
        if selected is None:
            spans = [entry[0][entry[2][0]:entry[2][1]] for entry in packed]
        else:
            spans = [[entry[0][position] for position in positions]
                     for entry, positions in zip(packed, selected)]
        leading.append(packed[0][2][0])
        trailing.append(len(packed[0][0]) - packed[0][2][1])
        features = tokenisation_features(spans[0], spans[1:], wildtype, sequences,
                                         budget=crossed.BUDGET)
        descriptors.append(dict(
            assay=row['assay'], max_packed_tokens=observed, wildtype_pooled_tokens=len(spans[0]),
            wildtype_residues=len(wildtype),
            token_count_cycle_minimum=float(features[:, 2].min()),
            token_count_cycle_maximum=float(features[:, 2].max()),
            segmentation_changed_tokens_maximum=float(features[:, 5].max())))
    tokenisation = handle.spec.tokenisation
    one_per_residue = all(
        entry['wildtype_pooled_tokens'] == entry['wildtype_residues']
        and entry['token_count_cycle_minimum'] == 0.0 and entry['token_count_cycle_maximum'] == 0.0
        for entry in descriptors)
    return dict(
        arm=arm_name, declared_tokenisation=tokenisation,
        stratum={'residue': 'amino-acid', 'byte': 'byte'}.get(tokenisation, 'bpe'),
        measured_segmentation='one_token_per_residue' if one_per_residue else 'multi_residue',
        leading_marker_tokens=[min(leading), max(leading)],
        trailing_marker_tokens=[min(trailing), max(trailing)],
        assays_with_nonzero_token_count_cycle=sum(
            1 for entry in descriptors
            if entry['token_count_cycle_minimum'] != 0.0 or entry['token_count_cycle_maximum'] != 0.0),
        token_count_cycle_minimum=min(entry['token_count_cycle_minimum'] for entry in descriptors),
        token_count_cycle_maximum=max(entry['token_count_cycle_maximum'] for entry in descriptors),
        segmentation_changed_tokens_maximum=max(
            entry['segmentation_changed_tokens_maximum'] for entry in descriptors),
        residues_per_token_minimum=min(entry['wildtype_residues'] / entry['wildtype_pooled_tokens']
                                       for entry in descriptors),
        residues_per_token_maximum=max(entry['wildtype_residues'] / entry['wildtype_pooled_tokens']
                                       for entry in descriptors),
        tokens_per_residue_minimum=min(entry['wildtype_pooled_tokens'] / entry['wildtype_residues']
                                       for entry in descriptors),
        tokens_per_residue_maximum=max(entry['wildtype_pooled_tokens'] / entry['wildtype_residues']
                                       for entry in descriptors),
        assays=descriptors)


def fit(args) -> None:
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('requested CUDA is unavailable')
        print(json.dumps(dict(device=args.device, gpu=torch.cuda.get_device_name(args.device))),
              flush=True)
    qualify = args.stage == 'qualify'
    local_blocks = list(CANDIDATE_BLOCKS) if qualify else list(args.local_block)
    additions = {'': ()} if qualify else dict(ADDITIONS)
    rows, hashes, support = load_panel(args.cohort, args.manifest, args.arm, 'common', args.seed,
                                       args.projection_dim)
    if args.assay_limit:
        rows = rows[:args.assay_limit]
        support = dict(support, definition='common_assay_limited',
                       assay_ids=[row['assay'] for row in rows], assay_limit=args.assay_limit)
    cohort_bytes = args.cohort.read_bytes()
    if sha256_bytes(cohort_bytes) != hashes[str(args.cohort)]:
        raise ValueError('cohort bytes changed between reads')
    cohort = json.loads(cohort_bytes)
    store, store_hashes = crossed.load_profile_store(args.profile_store, cohort)
    arms = crossed.attach_manifest_tokens(rows, args.manifest, args.arm, args.anchor_arm, hashes,
                                          None if args.assay_limit
                                          else [row['assay'] for row in rows])
    deviation = build_blocks(rows, cohort, store, local_blocks)
    interface = None if qualify else build_interface(rows, cohort, args.arm)
    report, predictions = evaluate_local_context(
        rows, local_blocks=local_blocks, additions=additions, device=args.device, seed=args.seed,
        fold_seed=args.fold_seed, bootstrap=args.bootstrap,
        progress=lambda name: print(json.dumps(dict(arm=args.arm, stage=args.stage, design=name,
                                                    status='fitting')), flush=True))
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f'local_context_{args.stage}_{args.arm}_fold{args.fold_seed}'
    if args.assay_limit:
        stem += f'_smoke{args.assay_limit}'
    arrays = args.out / f'{stem}.npz'
    np.savez_compressed(arrays, **{name: np.asarray(value, dtype=np.float64)
                                   for name, value in predictions.items()})
    report.update(
        schema_version='d1_local_context_v1', stage=args.stage,
        status='smoke' if args.assay_limit else 'complete',
        arm=args.arm, support_definition=support['definition'], manifest_arms=sorted(arms),
        anchor_arm=args.anchor_arm, cohort_sha256=hashes[str(args.cohort)],
        created_utc=datetime.now(timezone.utc).isoformat(), support=support, seed=args.seed,
        declaration=declaration(), declaration_sha256=declaration_sha256(),
        projection=dict(representation_dim_per_block=args.projection_dim,
                        representation_block_seeds=[args.seed + i for i in range(4)],
                        distribution='Gaussian sd=1/sqrt(dim)'),
        maximum_profile_score_deviation=deviation,
        tokenisation_interface=interface,
        profile_store_sha256=store_hashes, source_sha256=hashes,
        prediction_array_sha256=sha256_bytes(arrays.read_bytes()),
        analysis_code_sha256={name: sha256_bytes((ROOT / name).read_bytes()) for name in CODE_FILES},
        runtime=dict(torch=torch.__version__, numpy=np.__version__, device=args.device))
    write_json(args.out / f'{stem}.json', report)
    if qualify:
        printed = {f'{baseline}_{candidate}':
                   report['summaries'][f'qualification_delta_over_{baseline}_{candidate}']['point']
                   for candidate in CANDIDATE_BLOCKS for baseline in ('C', 'C_P')}
        printed.update({f'standalone_{candidate}':
                        report['summaries'][f'standalone_{candidate}']['point']
                        for candidate in CANDIDATE_BLOCKS})
    else:
        printed = {f'{label}_{name}': report['summaries'][f'increment_{label}_{name}']['point']
                   for name in control_sets(local_blocks)
                   for label in ('M', 'R', 'R_after_M')}
    print(json.dumps(dict(arm=args.arm, stage=args.stage, fold_seed=args.fold_seed,
                          points=printed)), flush=True)


def _load_reports(paths, stage: str) -> dict:
    reports = {}
    for path in sorted(paths):
        raw = path.read_bytes()
        payload = json.loads(raw)
        if payload.get('schema_version') != 'd1_local_context_v1':
            raise ValueError(f'{path}: not a local-context report')
        if payload.get('status') != 'complete' or payload.get('stage') != stage:
            raise ValueError(f'{path}: not a complete {stage} report')
        key = (payload['arm'], payload['fold_seed'])
        if key in reports:
            raise ValueError(f'{path}: duplicate arm and split seed')
        reports[key] = (path, sha256_bytes(raw), payload)
    if not reports:
        raise ValueError('no reports supplied')
    anchor = next(iter(reports.values()))[2]
    for path, _, payload in reports.values():
        for key in ('cohort_sha256', 'declaration_sha256', 'control_sets', 'additions',
                    'local_blocks', 'n_assays', 'n_families', 'n_variants', 'support_definition',
                    'anchor_arm'):
            if payload[key] != anchor[key]:
                raise ValueError(f'{path}: {key} differs from the shared panel')
        if payload['support']['assay_ids'] != anchor['support']['assay_ids']:
            raise ValueError(f'{path}: assay support differs from the shared panel')
        moved = [name for name in FITTED_CODE_FILES
                 if payload['analysis_code_sha256'][name] != anchor['analysis_code_sha256'][name]]
        if moved:
            raise ValueError(f'{path}: fitting code moved between cells: {moved}')
    return reports


def _shared_bindings(reports: dict, model_independent: list[str]) -> tuple[dict, dict]:
    folds, shared = {}, {}
    for (_, fold_seed), (path, _, payload) in sorted(reports.items()):
        realised = fold_membership(payload['folds'])
        if folds.setdefault(fold_seed, realised) != realised:
            raise ValueError(f'{path}: fold membership differs from another cell at the same '
                             'split seed')
        for design in model_independent:
            digest = payload['prediction_digests'][design]
            if shared.setdefault((fold_seed, design), digest) != digest:
                raise ValueError(f'{path}: model-independent design {design} differs across cells')
    return folds, shared


def collect_qualify(args) -> None:
    reports = _load_reports(args.collect, 'qualify')
    anchor = next(iter(reports.values()))[2]
    seeds = sorted({fold_seed for _, fold_seed in reports})
    if seeds != sorted(SPLIT_SEEDS):
        raise ValueError(f'the gate requires the three declared split seeds, received {seeds}')
    if len(reports) != len(seeds):
        raise ValueError('the qualification stage takes exactly one cell per split seed')
    if anchor['model_quantities_fitted']:
        raise ValueError('a qualification cell fitted a model quantity; the gate is refused')
    model_independent = [name + label for name in anchor['control_sets']
                         for label in anchor['additions']]
    _, shared = _shared_bindings(reports, model_independent)
    measured = {str(fold_seed): payload['summaries']
                for (_, fold_seed), (_, _, payload) in sorted(reports.items())}
    gate = qualification_gate(measured)
    admission = dict(
        schema_version='d1_local_context_qualification_v1', status='admitted', stage='qualify',
        created_utc=datetime.now(timezone.utc).isoformat(),
        cells=sorted(f'{arm}/{fold_seed}' for arm, fold_seed in reports),
        report_sha256={str(path): digest for path, digest, _ in reports.values()},
        cohort_sha256=anchor['cohort_sha256'], anchor_arm=anchor['anchor_arm'],
        declaration=anchor['declaration'], declaration_sha256=anchor['declaration_sha256'],
        support=dict(assay_ids=anchor['support']['assay_ids'], n_assays=anchor['n_assays'],
                     n_families=anchor['n_families'], n_variants=anchor['n_variants']),
        fold_seeds=seeds, control_sets=anchor['control_sets'],
        feature_dimensions=anchor['feature_dimensions'],
        analysis_code_sha256=anchor['analysis_code_sha256'],
        fitted_code_files=list(FITTED_CODE_FILES),
        model_quantities_fitted=anchor['model_quantities_fitted'],
        model_independent_designs=model_independent,
        model_independent_prediction_digests={f'{seed}/{design}': digest
                                              for (seed, design), digest in sorted(shared.items())},
        checks=['identical cohort bytes, assay support, counts and block declaration digest '
                'across every cell',
                'exactly one cell per declared split seed and no other split seed',
                'no model quantity fitted in any qualification design',
                'identical outer and inner group membership within each split seed'],
        gate=gate,
        summaries=measured)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'local_context_qualification.json', admission)
    print(json.dumps(dict(status='admitted', stage='qualify', fold_seeds=seeds,
                          qualified=gate['qualified'], discarded=gate['discarded'],
                          carried_forward=gate['carried_forward'])), flush=True)


def collect_measure(args) -> None:
    if not args.gate:
        raise SystemExit('--collect --stage measure requires --gate, the qualification receipt')
    gate_raw = args.gate.read_bytes()
    receipt = json.loads(gate_raw)
    if receipt.get('schema_version') != 'd1_local_context_qualification_v1':
        raise ValueError('the supplied gate file is not a qualification receipt')
    reports = _load_reports(args.collect, 'measure')
    anchor = next(iter(reports.values()))[2]
    if anchor['declaration_sha256'] != receipt['declaration_sha256']:
        raise ValueError('the measurement cells declare a different block set from the gate')
    if anchor['cohort_sha256'] != receipt['cohort_sha256']:
        raise ValueError('the measurement cells were fitted on a different cohort from the gate')
    if anchor['support']['assay_ids'] != receipt['support']['assay_ids']:
        raise ValueError('the measurement support differs from the qualified support')
    if sorted(anchor['local_blocks']) != sorted(receipt['gate']['carried_forward']):
        raise ValueError(f"measurement local blocks {anchor['local_blocks']} are not the "
                         f"carried-forward set {receipt['gate']['carried_forward']}")
    moved = [name for name in FITTED_CODE_FILES
             if anchor['analysis_code_sha256'][name] != receipt['analysis_code_sha256'][name]]
    if moved:
        raise ValueError(f'fitting code moved between the gate and the measurement: {moved}')
    model_independent = [name for name in anchor['control_sets']]
    folds, shared = _shared_bindings(reports, model_independent)
    admission = dict(
        schema_version='d1_local_context_measurement_v1', status='admitted', stage='measure',
        created_utc=datetime.now(timezone.utc).isoformat(),
        cells=sorted(f'{arm}/{fold_seed}' for arm, fold_seed in reports),
        report_sha256={str(path): digest for path, digest, _ in reports.values()},
        gate_sha256=sha256_bytes(gate_raw), gate_path=str(args.gate),
        local_blocks=anchor['local_blocks'],
        cohort_sha256=anchor['cohort_sha256'], anchor_arm=anchor['anchor_arm'],
        declaration_sha256=anchor['declaration_sha256'],
        support=dict(assay_ids=anchor['support']['assay_ids'], n_assays=anchor['n_assays'],
                     n_families=anchor['n_families'], n_variants=anchor['n_variants']),
        fold_seeds=sorted(folds), control_sets=anchor['control_sets'],
        additions=anchor['additions'],
        feature_dimensions={arm: payload['feature_dimensions']
                            for (arm, _), (_, _, payload) in sorted(reports.items())},
        model_independent_designs=model_independent,
        model_independent_prediction_digests={f'{seed}/{design}': digest
                                              for (seed, design), digest in sorted(shared.items())},
        manifest_arms={arm: payload['manifest_arms']
                       for (arm, _), (_, _, payload) in sorted(reports.items())},
        checks=['identical cohort bytes, assay support, counts and block declaration digest '
                'across every cell and against the qualification receipt',
                'the fitted local blocks are exactly the gate\'s carried-forward set',
                'identical outer and inner group membership across arms at each split seed',
                'bit-identical predictions for every model-independent control design'],
        tokenisation_stratum={arm: [payload['tokenisation_interface']['declared_tokenisation'],
                                    payload['tokenisation_interface']['stratum'],
                                    payload['tokenisation_interface']['measured_segmentation']]
                              for (arm, _), (_, _, payload) in sorted(reports.items())},
        tokenisation_interface={arm: {key: value for key, value
                                      in payload['tokenisation_interface'].items()
                                      if key != 'assays'}
                                for (arm, _), (_, _, payload) in sorted(reports.items())},
        summaries={f'{arm}/{fold_seed}': payload['summaries']
                   for (arm, fold_seed), (_, _, payload) in sorted(reports.items())})
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'local_context_measurement.json', admission)
    print(json.dumps(dict(status='admitted', stage='measure', cells=len(reports),
                          fold_seeds=sorted(folds), local_blocks=anchor['local_blocks'])),
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['qualify', 'measure'])
    parser.add_argument('--cohort', type=Path)
    parser.add_argument('--manifest', action='append', type=Path, default=[])
    parser.add_argument('--arm')
    parser.add_argument('--profile-store', type=Path)
    parser.add_argument('--local-block', action='append', default=[],
                        help='measurement only: a carried-forward local control, repeatable')
    parser.add_argument('--collect', nargs='+', type=Path)
    parser.add_argument('--gate', type=Path,
                        help='measurement collection only: the qualification receipt to bind to')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=20260923,
                        help='fixed representation-projection and bootstrap seed')
    parser.add_argument('--fold-seed', type=int, default=20260923,
                        help='outer/inner group assignment seed')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--projection-dim', type=int, default=256)
    parser.add_argument('--assay-limit', type=int, default=0,
                        help='interface check only: restrict to the first N assays')
    parser.add_argument('--anchor-arm', default=DEFAULT_ANCHOR_ARM)
    args = parser.parse_args()
    if args.collect:
        if args.cohort or args.manifest or args.arm or args.profile_store or args.local_block:
            raise SystemExit('--collect takes report paths only')
        (collect_qualify if args.stage == 'qualify' else collect_measure)(args)
        return
    missing = [name for name, value in (('--cohort', args.cohort), ('--arm', args.arm),
                                        ('--profile-store', args.profile_store)) if not value]
    if missing:
        raise SystemExit(f'a fitting cell requires {", ".join(missing)}')
    if args.stage == 'qualify':
        if args.local_block:
            raise SystemExit('the qualification stage fits every declared candidate; '
                             '--local-block is a measurement argument')
        if args.arm != args.anchor_arm:
            raise SystemExit('the qualification stage reads the anchor arm, whose model '
                             'quantities it discards; no other arm may stand in for it')
    elif not args.local_block:
        raise SystemExit('a measurement cell requires at least one --local-block')
    if len(args.manifest) < (1 if args.arm == args.anchor_arm else 2):
        raise SystemExit('a fitting cell requires its own manifest and the anchor manifest')
    fit(args)


if __name__ == '__main__':
    main()
