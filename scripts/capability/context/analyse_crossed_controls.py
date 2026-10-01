#!/usr/bin/env python3
"""Crossed C / C+L / C+P / C+L+P controls with separate likelihood and representation increments.

One fitting cell fits every nested control set for one admitted Readout arm at
one split seed, on the exact common support the admitted extraction manifests
declare. The panel loader, the representation projection, the composition
descriptors, the nested folds, the ridge recipe and the group-bootstrap
intervals are imported from the admitted Readout analysis; this entry point adds
the local-pattern, mutation-local-profile and tokenisation-interface blocks and
the crossed contrasts.

``--collect`` instead verifies a completed set of cell reports against one
another: identical cohort bytes, support, assay order and fold membership, and
bit-identical predictions for every model-independent control design.
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
from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.context.crossed_controls import (  # noqa: E402
    CONTROL_SETS, LOCAL_PROJECTION_DIM, LOCAL_PROJECTION_SEED, PROFILE_FEATURE_ORDER,
    TOKENISATION_FEATURE_ORDER, evaluate_crossed_controls, fold_membership, local_pattern_features,
    local_projection, profile_features, tokenisation_features)
from src.capability.models.fitness import parse_mutant  # noqa: E402
from src.capability.core.io import write_json  # noqa: E402
from src.capability.context.profiles import PSEUDOCOUNT_ALPHA, profile_scores as lookup_scores  # noqa: E402
from src.capability.readouts.readout_extraction import (  # noqa: E402
    load_readout_arm, pack_sequence, representation_positions)

BUDGET = 1024
#: The arm whose complete extraction manifest defines the anchor panel. The
#: readout protocol's scheduling rule allows a cell to load its own manifest plus
#: one complete manifest that fixes the intersection; this arm's native support is
#: exactly the 201-assay anchor, so the realised common support is required to
#: equal its assay set and a cell whose arm covers less of the anchor is refused
#: rather than silently fitted on a smaller panel.
DEFAULT_ANCHOR_ARM = 'progen3-3b'
#: Baseline files the mutation-local profile block is bound to. The searched-hit
#: table is not among them: this block reads the retained column frequencies, not
#: the alignment rows a pairwise fit would need.
PROFILE_STORE_FILES = ('profiles.npz', 'profiles.json', 'lookup.json', 'wildtypes.json')
CODE_FILES = ('scripts/capability/context/analyse_crossed_controls.py', 'scripts/capability/readouts/analyse_readout.py',
              'src/capability/context/crossed_controls.py', 'src/capability/readouts/readout_analysis.py',
              'src/capability/readouts/readout_extraction.py', 'src/capability/context/profiles.py',
              'src/capability/context/profile_increment.py')
#: Control designs built from no model output, and therefore required to be
#: bit-identical across arms on the shared panel at one split seed.
MODEL_INDEPENDENT = ('C', 'C_L', 'C_P', 'C_L_P')
#: Largest tolerated deviation between the retained cohort profile scores and
#: the scores recomputed from the retained float32 column-frequency arrays.
PROFILE_SCORE_TOLERANCE = 1e-4


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_profile_store(directory: Path, cohort: dict) -> tuple[dict, dict]:
    """Bind the retained profile arrays to the cohort's own declared baseline hashes."""
    declared = cohort['baseline_sha256']
    hashes = {}
    for name in PROFILE_STORE_FILES:
        if name not in declared:
            raise ValueError(f'{name} is not a declared cohort baseline file')
        digest = sha256_bytes((directory / name).read_bytes())
        if digest != declared[name]:
            raise ValueError(f'{name} does not match the cohort baseline hash')
        hashes[name] = digest
    spec = importlib.util.spec_from_file_location('stage20', ROOT / 'scripts/capability/stages/retrieval_bound.py')
    stage20 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage20)
    settings = json.loads((directory / 'lookup.json').read_text())['settings']
    alpha = float(settings['alpha'])
    if alpha != PSEUDOCOUNT_ALPHA:
        raise ValueError('retained profile alpha differs from the declared pseudocount weight')
    catalogue = json.loads((directory / 'wildtypes.json').read_text())
    background = np.array([catalogue['corpus']['background'][residue] for residue in AA20], dtype=np.float64)
    return dict(stored=np.load(directory / 'profiles.npz', allow_pickle=False),
                metadata=json.loads((directory / 'profiles.json').read_text()),
                background=background, alpha=alpha,
                profile_from_store=stage20._profile_from_store), hashes


def build_blocks(rows: list[dict], cohort: dict, store: dict, arm_name: str,
                 projection: np.ndarray) -> dict:
    """Attach L, P and T blocks to the admitted panel rows, verifying every binding."""
    cohort_map = {row['assay']: row for row in cohort['assays']}
    handle = load_readout_arm(arm_name, None, device=None)
    descriptors, deviations, leading, trailing = [], [], [], []
    for row in rows:
        record = cohort_map[row['assay']]
        wildtype, sequences = record['wildtype'], list(record['sequences'])
        if row['mutants'] != record['mutants'] or len(sequences) != len(row['mutants']):
            raise ValueError(f"{row['assay']}: cohort mutation order or sequence count mismatch")
        substitutions = [parse_mutant(mutant) for mutant in row['mutants']]
        row['L'] = local_pattern_features(wildtype, row['mutants'], sequences, projection)
        profile = store['profile_from_store'](store['stored'], record['wildtype_id'], wildtype,
                                              store['metadata'])
        recomputed = lookup_scores(profile, store['background'], substitutions, alpha=store['alpha'])
        deviations.append(float(np.abs(recomputed - np.asarray(row['P'], dtype=np.float64)).max()))
        row['P_block'] = profile_features(
            wildtype, row['mutants'], sequences, row['P'], profile.frequencies, store['background'],
            store['metadata']['profiles'][record['wildtype_id']], alpha=store['alpha'])
        states = [wildtype] + sequences
        packed = [pack_sequence(handle, sequence) for sequence in states]
        observed = max(len(entry[0]) for entry in packed)
        if observed != row['max_packed_tokens']:
            raise ValueError(f"{row['assay']}: recomputed packed maximum {observed} differs from "
                             f"the manifest value {row['max_packed_tokens']}")
        if observed > BUDGET:
            raise ValueError(f"{row['assay']}: recomputed packing exceeds the frozen budget")
        # ProtGPT2 pools residue-bearing tokens only, excluding pure FASTA
        # newlines; every other arm pools its whole residue-bearing span. Taking
        # the production selector rather than the span keeps the counted tokens
        # equal to the mean-pooling denominators in both cases.
        selected = representation_positions(handle, states, packed)
        if selected is None:
            spans = [entry[0][entry[2][0]:entry[2][1]] for entry in packed]
        else:
            spans = [[entry[0][position] for position in positions]
                     for entry, positions in zip(packed, selected)]
        leading.append(packed[0][2][0])
        trailing.append(len(packed[0][0]) - packed[0][2][1])
        row['T'] = tokenisation_features(spans[0], spans[1:], wildtype, sequences, budget=BUDGET)
        descriptors.append(dict(
            assay=row['assay'], max_packed_tokens=observed, wildtype_pooled_tokens=len(spans[0]),
            wildtype_residues=len(wildtype),
            mutant_pooled_tokens_minimum=int(min(len(span) for span in spans[1:])),
            mutant_pooled_tokens_maximum=int(max(len(span) for span in spans[1:])),
            token_count_cycle_minimum=float(row['T'][:, 2].min()),
            token_count_cycle_maximum=float(row['T'][:, 2].max()),
            segmentation_changed_tokens_mean=float(row['T'][:, 5].mean()),
            segmentation_changed_tokens_maximum=float(row['T'][:, 5].max())))
    deviation = max(deviations)
    if deviation > PROFILE_SCORE_TOLERANCE:
        raise ValueError(f'recomputed profile scores deviate by {deviation} from the cohort values')
    tokenisation = handle.spec.tokenisation
    # The measured stratum, not only the declared one. A registry label of "bpe"
    # can still render exactly one token per residue -- Galactica's protein
    # delimiter path does -- and then the interface has no segmentation degree of
    # freedom at all, which is the fact a segmentation attribution turns on.
    one_per_residue = all(
        entry['wildtype_pooled_tokens'] == entry['wildtype_residues']
        and entry['token_count_cycle_minimum'] == 0.0 and entry['token_count_cycle_maximum'] == 0.0
        for entry in descriptors)
    return dict(
        arm=arm_name, tokenisation=tokenisation,
        stratum={'residue': 'amino-acid', 'byte': 'byte'}.get(tokenisation, 'bpe'),
        measured_segmentation='one_token_per_residue' if one_per_residue else 'multi_residue',
        # Leading and trailing tokens of the packed wild-type row that carry no
        # residue, read off the production packing rather than a per-family rule.
        leading_marker_tokens=[min(leading), max(leading)],
        trailing_marker_tokens=[min(trailing), max(trailing)],
        maximum_profile_score_deviation=deviation,
        assays_with_nonzero_token_count_cycle=sum(
            1 for entry in descriptors
            if entry['token_count_cycle_minimum'] != 0.0 or entry['token_count_cycle_maximum'] != 0.0),
        token_count_cycle_minimum=min(entry['token_count_cycle_minimum'] for entry in descriptors),
        token_count_cycle_maximum=max(entry['token_count_cycle_maximum'] for entry in descriptors),
        segmentation_changed_tokens_maximum=max(
            entry['segmentation_changed_tokens_maximum'] for entry in descriptors),
        tokens_per_residue_minimum=min(entry['wildtype_pooled_tokens'] / entry['wildtype_residues']
                                       for entry in descriptors),
        tokens_per_residue_maximum=max(entry['wildtype_pooled_tokens'] / entry['wildtype_residues']
                                       for entry in descriptors),
        assays=descriptors)


def attach_manifest_tokens(rows: list[dict], manifests: list[Path], arm: str, anchor_arm: str,
                           hashes: dict, support_assay_ids: list[str] | None) -> list[str]:
    """Carry the declared packed-token maxima on, and bind the panel to the anchor.

    ``support_assay_ids`` is ``None`` only for an assay-limited interface check,
    whose artifact is marked ``smoke`` and cannot be collected.
    """
    arms, declared, anchor = [], None, None
    for path in manifests:
        raw = path.read_bytes()
        if sha256_bytes(raw) != hashes[str(path)]:
            raise ValueError(f'{path}: manifest bytes changed between reads')
        source = json.loads(raw)
        arms.append(source['identity']['arm'])
        if source['identity']['arm'] == arm:
            declared = {entry['assay']: entry['max_packed_tokens'] for entry in source['assays']}
        if source['identity']['arm'] == anchor_arm:
            anchor = sorted(entry['assay'] for entry in source['assays'])
    if declared is None:
        raise ValueError(f'no manifest declares arm {arm}')
    if anchor is None:
        raise ValueError(f'no manifest declares the anchor arm {anchor_arm}')
    if support_assay_ids is not None and support_assay_ids != anchor:
        raise ValueError(f'realised support of {len(support_assay_ids)} assays is not the '
                         f'{len(anchor)}-assay anchor that {anchor_arm} declares; this arm needs a '
                         'separately named exact intersection, which this entry point does not fit')
    for row in rows:
        if row['assay'] not in declared:
            raise ValueError(f"{row['assay']}: absent from the selected arm manifest")
        row['max_packed_tokens'] = declared[row['assay']]
    return arms


def fit(args) -> None:
    # The GPU route leaves this at the admitted four threads; the CPU route needs
    # to be told how many it may use, because a pod-CPU lane is sized by the
    # caller and the default would serialise the ridge solves.
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('requested CUDA is unavailable')
        print(json.dumps(dict(device=args.device, gpu=torch.cuda.get_device_name(args.device))), flush=True)
    rows, hashes, support = load_panel(args.cohort, args.manifest, args.arm, 'common', args.seed,
                                       args.projection_dim)
    if args.assay_limit:
        # An interface check, never an admitted result: the artifact is named and
        # marked so that it cannot be collected as a production cell.
        rows = rows[:args.assay_limit]
        support = dict(support, definition='common_assay_limited',
                       assay_ids=[row['assay'] for row in rows], assay_limit=args.assay_limit)
    cohort_bytes = args.cohort.read_bytes()
    if sha256_bytes(cohort_bytes) != hashes[str(args.cohort)]:
        raise ValueError('cohort bytes changed between reads')
    cohort = json.loads(cohort_bytes)
    store, store_hashes = load_profile_store(args.profile_store, cohort)
    arms = attach_manifest_tokens(rows, args.manifest, args.arm, args.anchor_arm, hashes,
                                  None if args.assay_limit else [row['assay'] for row in rows])
    projection = local_projection(args.local_projection_dim, args.local_projection_seed)
    interface = build_blocks(rows, cohort, store, args.arm, projection)
    report, predictions = evaluate_crossed_controls(
        rows, device=args.device, seed=args.seed, fold_seed=args.fold_seed, bootstrap=args.bootstrap,
        progress=lambda name: print(json.dumps(dict(arm=args.arm, design=name, status='fitting')), flush=True))
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f'crossed_controls_{args.arm}_fold{args.fold_seed}'
    if args.assay_limit:
        stem += f'_smoke{args.assay_limit}'
    arrays = args.out / f'{stem}.npz'
    np.savez_compressed(arrays, **{name: np.asarray(value, dtype=np.float64)
                                   for name, value in predictions.items()})
    report.update(
        schema_version='d1_crossed_controls_v1', status='smoke' if args.assay_limit else 'complete',
        arm=args.arm, support_definition=support['definition'], manifest_arms=sorted(arms),
        anchor_arm=args.anchor_arm, cohort_sha256=hashes[str(args.cohort)],
        created_utc=datetime.now(timezone.utc).isoformat(), support=support, seed=args.seed,
        projection=dict(representation_dim_per_block=args.projection_dim,
                        representation_block_seeds=[args.seed + i for i in range(4)],
                        tripeptide_dim=args.local_projection_dim,
                        tripeptide_seed=args.local_projection_seed,
                        distribution='Gaussian sd=1/sqrt(dim)'),
        feature_order=dict(composition='the admitted 444 Readout sequence descriptors, unchanged',
                           local='400 exact dipeptide count differences then the projected tripeptide block',
                           profile=list(PROFILE_FEATURE_ORDER),
                           tokenisation=list(TOKENISATION_FEATURE_ORDER)),
        tokenisation_interface=interface, profile_store_sha256=store_hashes, source_sha256=hashes,
        prediction_array_sha256=sha256_bytes(arrays.read_bytes()),
        analysis_code_sha256={name: sha256_bytes((ROOT / name).read_bytes()) for name in CODE_FILES},
        runtime=dict(torch=torch.__version__, numpy=np.__version__, device=args.device))
    write_json(args.out / f'{stem}.json', report)
    print(json.dumps(dict(arm=args.arm, fold_seed=args.fold_seed, increments={
        f'{label}_{control}': report['summaries'][f'increment_{label}_{control}']['point']
        for control in CONTROL_SETS for label in ('M', 'R', 'R_after_M')})), flush=True)


def collect(args) -> None:
    reports = {}
    for path in sorted(args.collect):
        raw = path.read_bytes()
        payload = json.loads(raw)
        if payload.get('schema_version') != 'd1_crossed_controls_v1' or payload.get('status') != 'complete':
            raise ValueError(f'{path}: not a complete crossed-controls report')
        key = (payload['arm'], payload['fold_seed'])
        if key in reports:
            raise ValueError(f'{path}: duplicate arm and split seed')
        reports[key] = (path, sha256_bytes(raw), payload)
    if not reports:
        raise ValueError('no reports supplied')
    anchor = next(iter(reports.values()))[2]
    for path, _, payload in reports.values():
        for key in ('cohort_sha256', 'control_sets', 'feature_order', 'n_assays', 'n_families',
                    'n_variants', 'support_definition', 'anchor_arm'):
            if payload[key] != anchor[key]:
                raise ValueError(f'{path}: {key} differs from the shared panel')
        if payload['support']['assay_ids'] != anchor['support']['assay_ids']:
            raise ValueError(f'{path}: assay support differs from the shared panel')
        if payload['feature_dimensions'] != anchor['feature_dimensions'] and payload['arm'] == anchor['arm']:
            raise ValueError(f'{path}: design widths differ within one arm')
    folds, shared = {}, {}
    for (arm, fold_seed), (path, _, payload) in sorted(reports.items()):
        realised = fold_membership(payload['folds'])
        if folds.setdefault(fold_seed, realised) != realised:
            raise ValueError(f'{path}: fold membership differs from another arm at the same split seed')
        for design in MODEL_INDEPENDENT:
            digest = payload['prediction_digests'][design]
            if shared.setdefault((fold_seed, design), digest) != digest:
                raise ValueError(f'{path}: model-independent design {design} differs across arms')
    admission = dict(
        schema_version='d1_crossed_controls_admission_v1', status='admitted',
        created_utc=datetime.now(timezone.utc).isoformat(),
        cells=sorted(f'{arm}/{fold_seed}' for arm, fold_seed in reports),
        report_sha256={str(path): digest for path, digest, _ in reports.values()},
        cohort_sha256=anchor['cohort_sha256'], anchor_arm=anchor['anchor_arm'],
        manifest_arms={arm: payload['manifest_arms']
                       for (arm, _), (_, _, payload) in sorted(reports.items())},
        support=dict(assay_ids=anchor['support']['assay_ids'], n_assays=anchor['n_assays'],
                     n_families=anchor['n_families'], n_variants=anchor['n_variants']),
        fold_seeds=sorted(folds), control_sets=anchor['control_sets'],
        feature_dimensions={arm: payload['feature_dimensions']
                            for (arm, _), (_, _, payload) in sorted(reports.items())},
        model_independent_designs=list(MODEL_INDEPENDENT),
        model_independent_prediction_digests={f'{seed}/{design}': digest
                                              for (seed, design), digest in sorted(shared.items())},
        checks=['identical cohort bytes, assay support and counts across every cell',
                'identical outer and inner group membership across arms at each split seed',
                'bit-identical predictions for every model-independent control design',
                'identical control-set and feature-order definitions'],
        summaries={f'{arm}/{fold_seed}': payload['summaries']
                   for (arm, fold_seed), (_, _, payload) in sorted(reports.items())},
        tokenisation_stratum={arm: [payload['tokenisation_interface']['stratum'],
                                    payload['tokenisation_interface']['measured_segmentation']]
                              for (arm, _), (_, _, payload) in sorted(reports.items())},
        tokenisation_interface={arm: {key: value for key, value
                                      in payload['tokenisation_interface'].items() if key != 'assays'}
                                for (arm, _), (_, _, payload) in sorted(reports.items())})
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'crossed_controls_admission.json', admission)
    print(json.dumps(dict(status='admitted', cells=len(reports), fold_seeds=sorted(folds))), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path)
    parser.add_argument('--manifest', action='append', type=Path, default=[])
    parser.add_argument('--arm')
    parser.add_argument('--profile-store', type=Path)
    parser.add_argument('--collect', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=20260923,
                        help='fixed representation-projection and bootstrap seed')
    parser.add_argument('--fold-seed', type=int, default=20260923, help='outer/inner group assignment seed')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--projection-dim', type=int, default=256)
    parser.add_argument('--assay-limit', type=int, default=0,
                        help='interface check only: restrict to the first N assays of the common support')
    parser.add_argument('--anchor-arm', default=DEFAULT_ANCHOR_ARM,
                        help='the arm whose complete manifest defines the required anchor panel')
    parser.add_argument('--local-projection-dim', type=int, default=LOCAL_PROJECTION_DIM)
    parser.add_argument('--local-projection-seed', type=int, default=LOCAL_PROJECTION_SEED)
    args = parser.parse_args()
    if args.collect:
        if args.cohort or args.manifest or args.arm or args.profile_store:
            raise SystemExit('--collect takes report paths only')
        collect(args)
        return
    missing = [name for name, value in (('--cohort', args.cohort), ('--arm', args.arm),
                                        ('--profile-store', args.profile_store)) if not value]
    if missing:
        raise SystemExit(f'a fitting cell requires {", ".join(missing)}')
    # The anchor arm's own manifest is the anchor manifest, so it needs no second
    # one; every other arm does, and a duplicate manifest is refused upstream.
    if len(args.manifest) < (1 if args.arm == args.anchor_arm else 2):
        raise SystemExit('a fitting cell requires its own manifest and the anchor manifest')
    fit(args)


if __name__ == '__main__':
    main()
