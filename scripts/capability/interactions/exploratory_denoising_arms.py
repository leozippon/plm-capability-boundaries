#!/usr/bin/env python3
"""Score the frozen cohorts with a denoising arm, on a roster of its own.

This entry point exists to qualify one claim and nothing else: that a
non-autoregressive generative model can enter this pipeline only through the
part of its score family that is not paradigm-specific. It therefore scores two
frozen cohorts with two DPLM rungs and the two ESM-2 checkpoints DPLM was
initialised from, and reports where each lands.

**The arms here are not panel arms.** :data:`EXPLORATORY_ROSTER` is separate from
``src.capability.interactions.pairwise_epistasis.ROSTER`` and nothing in this file touches it,
so the admitted panel stays at 33 arms and every panel count, verdict and figure
is unchanged by anything measured here. An exploratory row is read beside the
panel, never inside it.

The score
---------
DPLM is an absorbing-state discrete diffusion model whose denoiser is one
bidirectional forward pass over a partially masked sequence and which carries no
timestep input. Its native score family is therefore indexed by a corruption
level p: mask a set M of positions, and read the mean log-probability the
denoiser assigns to the true residues at those positions. The member used here
is the deterministic one at p = 1/L, with M holding the substituted position
alone:

    D(w, j, a) = log p_theta(a | w with position j masked)
               - log p_theta(w_j | w with position j masked)

Wild type and mutant are read off the *same* forward pass, so the corruption is
matched by construction rather than by agreement between two draws, and the
member carries no sampling at all. A multi-substitution variant is scored as the
sum of its per-position contrasts, each taken with only that position masked;
that is an additive convention over the same one-position member and is declared
rather than derived.

The choice of member is not neutral and was qualified before this run
(``archive/docs/shared/diffusion-arm-feasibility.md``): the full-sequence member, which is
the one that is actually about the diffusion paradigm, needs 331 to 2,694 noise
realisations per variant to bring its sampling noise to a tenth of the
between-variant spread, and is not run here.

The forward pass
----------------
The DPLM checkpoints load into ``transformers.EsmForMaskedLM`` with no missing,
unexpected or mismatched keys. Equivalence to the reference ``byprot`` loader
rests on that zero-mismatch load and on reading the reference wrapper, which
builds the stock embeddings including ``token_dropout``, pre-scales the query and
calls scaled-dot-product attention with unit scale, and uses the stock head. **The
reference loader was never executed, so no numerical A/B supports this**; the
residual risk is recorded with every number this entry point produces. The
reference's decoding path additionally suppresses the mask and unknown logits,
which cancels exactly in the contrast above because both terms share one
normalisation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.core.io import sha256_file, write_json
from src.capability.context.profile_increment import correlation, standardized_rank, summarize
from src.capability.context.profiles import cluster_bootstrap
from src.capability.stability.stability_gate import (
    SPLIT_SEEDS, build_panel, fold_predictions, group_errors, interval,
    load_profiles, paired_increment, secondary_control_set, spearman_increment)

SCHEMA = 'exploratory_denoising_arms_v1'

#: The exploratory roster, disjoint from the admitted panel by construction. Each
#: entry is the checkpoint directory name under ``--model-root`` and the paradigm
#: the arm is here to represent. The two ESM-2 rungs are not optional: DPLM-650M
#: and DPLM-3B were initialised from them, so without the pair a DPLM row cannot
#: be read as evidence about discrete diffusion rather than about bidirectional
#: masked scoring.
EXPLORATORY_ROSTER: dict[str, dict[str, str]] = {
    'dplm-650m': {'directory': 'dplm_650m', 'paradigm': 'absorbing_state_discrete_diffusion',
                  'initialised_from': 'esm2-650m', 'source': 'airkingbd/dplm_650m'},
    'dplm-3b': {'directory': 'dplm_3b', 'paradigm': 'absorbing_state_discrete_diffusion',
                'initialised_from': 'esm2-3b', 'source': 'airkingbd/dplm_3b'},
    'esm2-650m': {'directory': 'esm2_650m', 'paradigm': 'masked_language_model',
                  'initialised_from': '', 'source': 'facebook/esm2_t33_650M_UR50D'},
    'esm2-3b': {'directory': 'esm2_t36_3B_UR50D', 'paradigm': 'masked_language_model',
                'initialised_from': '', 'source': 'facebook/esm2_t36_3B_UR50D'},
}

#: Declared residue capacity of the ESM-2 architecture these arms share: the
#: configuration states 1,026 positions and two are the terminal tokens. An assay
#: whose wild type exceeds it is outside the arm's declared interface and is
#: excluded with its identity recorded, exactly as other arms carry their own
#: context-limited support subsets.
RESIDUE_CAPACITY = 1024

#: Sequences re-scored in an independent second pass to record run-to-run
#: reproducibility of the identical computation.
REPEAT_BACKGROUNDS = 3

BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260926


def load_arm(model_root: Path, arm: str, device: str):
    from transformers import AutoTokenizer, EsmForMaskedLM

    directory = model_root / EXPLORATORY_ROSTER[arm]['directory']
    if not directory.exists():
        raise SystemExit(f'{arm}: {directory} does not exist on this host')
    tokenizer = AutoTokenizer.from_pretrained(directory)
    # ``torch_dtype`` rather than ``dtype``: the pod runtime carries Transformers
    # 4.52.4, whose ``from_pretrained`` does not accept ``dtype``, while 4.57.3
    # accepts both and only deprecates this spelling. One call has to load in both.
    model, info = EsmForMaskedLM.from_pretrained(
        directory, torch_dtype=torch.float32, output_loading_info=True)
    unmatched = {key: info[key] for key in ('missing_keys', 'mismatched_keys')
                 if info[key]}
    if unmatched:
        raise SystemExit(f'{arm}: checkpoint does not fit the graph: {unmatched}')
    model.eval().to(device)
    identity = {'arm': arm, **EXPLORATORY_ROSTER[arm],
                'parameters': int(sum(p.numel() for p in model.parameters())),
                'unexpected_keys': list(info['unexpected_keys']),
                'reference_loader_executed': False}
    return tokenizer, model, identity


@torch.no_grad()
def position_logprobs(model, tokenizer, wildtype: str, positions, device: str,
                      batch: int) -> dict[int, np.ndarray]:
    """Log-probabilities at each requested position, that position alone masked."""

    encoded = tokenizer(wildtype, return_tensors='pt')['input_ids']
    ordered = sorted(positions)
    out: dict[int, np.ndarray] = {}
    for start in range(0, len(ordered), batch):
        chunk = ordered[start:start + batch]
        rows = encoded.repeat(len(chunk), 1)
        for index, position in enumerate(chunk):
            rows[index, position + 1] = tokenizer.mask_token_id
        logits = model(input_ids=rows.to(device)).logits.float()
        logprobs = torch.log_softmax(logits, -1).cpu().numpy()
        for index, position in enumerate(chunk):
            out[position] = logprobs[index, position + 1]
    return out


def substitutions(identifier: str, wildtype: str) -> list[tuple[int, str, str]]:
    """Parse one mutation identifier into zero-based (position, wild, mutant) triples."""

    parsed = []
    for token in identifier.split(':'):
        position = int(token[1:-1]) - 1
        if not 0 <= position < len(wildtype) or wildtype[position] != token[0]:
            raise ValueError(f'{identifier}: {token} disagrees with the cohort wild type')
        parsed.append((position, token[0], token[-1]))
    return parsed


def contrasts(model, tokenizer, wildtype: str, identifiers, device: str,
              batch: int) -> np.ndarray:
    parsed = [substitutions(identifier, wildtype) for identifier in identifiers]
    needed = {position for entry in parsed for position, _, _ in entry}
    table = position_logprobs(model, tokenizer, wildtype, needed, device, batch)
    values = []
    for entry in parsed:
        total = 0.0
        for position, wild, mutant in entry:
            row = table[position]
            total += float(row[tokenizer.convert_tokens_to_ids(mutant)]
                           - row[tokenizer.convert_tokens_to_ids(wild)])
        values.append(total)
    return np.asarray(values, dtype=float)


# --------------------------------------------------------------------------- #
# Endpoint 1: the frozen single-mutant stability cohort
# --------------------------------------------------------------------------- #

def run_stability(args) -> None:
    cohort = json.loads(args.cohort.read_bytes())
    if cohort.get('schema') != 'stability_singles_cohort_v1':
        raise SystemExit('unexpected cohort schema')
    cohort_sha = sha256_file(args.cohort)
    controls = json.loads(args.controls.read_bytes())
    if controls.get('schema') != 'stability_control_qualification_v1':
        raise SystemExit('unexpected control-qualification schema')
    if controls['cohort_sha256'] != cohort_sha:
        raise SystemExit('the control qualification was run against a different cohort')
    qualified = tuple(controls['qualified_control_set'])
    secondary, secondary_record = secondary_control_set(controls)

    tokenizer, model, identity = load_arm(args.model_root, args.arm, args.device)
    started = time.time()
    scores, repeat_difference, forwards = [], 0.0, 0
    for index, row in enumerate(cohort['backgrounds']):
        wildtype = row['wildtype']
        identifiers = [f"{wildtype[v['position'] - 1]}{v['position']}{v['mutant']}"
                       for v in row['variants']]
        values = contrasts(model, tokenizer, wildtype, identifiers, args.device, args.batch)
        forwards += len({v['position'] - 1 for v in row['variants']})
        if index < REPEAT_BACKGROUNDS:
            again = contrasts(model, tokenizer, wildtype, identifiers, args.device, args.batch)
            repeat_difference = max(repeat_difference, float(np.abs(values - again).max()))
        scores.append(values)
    elapsed = time.time() - started
    model_column = np.concatenate(scores)
    del model
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()

    wildtypes = {row['name']: row['wildtype'] for row in cohort['backgrounds']}
    profiles, _ = load_profiles(args.profiles, wildtypes)
    panel = build_panel(cohort, profiles)
    if len(model_column) != len(panel['target']):
        raise SystemExit(f'{args.arm}: {len(model_column)} scores against '
                         f"{len(panel['target'])} cohort variants")
    blocks = dict(panel['blocks'], M=model_column[:, None])

    designs = {'S': qualified, 'S_M': (*qualified, 'M'),
               'S2': secondary, 'S2_M': (*secondary, 'M')}
    seeds, group_error = {}, {}
    for seed in SPLIT_SEEDS:
        outcome = fold_predictions(panel, blocks, designs, seed=seed, device=args.fit_device)
        for design in designs:
            labels, errors = group_errors(panel['target'], outcome['predictions'][design],
                                          panel['group'], panel['site'])
            group_error[f'{design}@{seed}'] = errors
            group_error['labels'] = np.asarray(labels)
        _, null = group_errors(panel['target'], outcome['predictions']['NO_EFFECT_NULL'],
                               panel['group'], panel['site'])
        record = {'no_effect_null_mse_kcal2_mol2': interval(null),
                  'alpha': [fold['alpha'] for fold in outcome['folds']],
                  'dimensions': outcome['folds'][0]['dimensions']}
        for role, baseline in (('primary', 'S'), ('secondary', 'S2')):
            record[f'{role}_score_increment_kcal2_mol2'] = paired_increment(
                panel, outcome['predictions'], f'{baseline}_M', baseline)
            record[f'{role}_score_increment_spearman'] = spearman_increment(
                panel, outcome['predictions'], f'{baseline}_M', baseline)
            _, errors = group_errors(panel['target'], outcome['predictions'][baseline],
                                     panel['group'], panel['site'])
            record[f'{role}_baseline_mse_kcal2_mol2'] = interval(errors)
        seeds[str(seed)] = record

    write_json(args.out / f'stability_{args.arm}.json', {
        'schema': SCHEMA, 'endpoint': 'megascale_single_mutant_stability',
        'roster': 'exploratory', 'identity': identity,
        'cohort_sha256': cohort_sha,
        'controls_sha256': sha256_file(args.controls),
        'qualified_control_set': list(qualified),
        'secondary_control_set': list(secondary),
        'secondary_control_record': secondary_record,
        'baseline_omits_tokenisation_block': True,
        'baseline_omits_tokenisation_block_reason':
            'no panel arm qualified its tokenisation descriptors (0 of 33), so the panel '
            'baseline carries no arm-dependent column and each seed has one number; using '
            'that identical baseline keeps the exploratory increment on the panel scale, '
            'and offering these arms a descriptor block would change the baseline instead',
        'score': {'member': 'deterministic_p_equals_one_over_L',
                  'multi_substitution_rule': 'sum of per-position contrasts',
                  'repeat_max_abs_difference_nats': repeat_difference,
                  'backgrounds_rescored': min(REPEAT_BACKGROUNDS, len(cohort['backgrounds']))},
        'cost': {'forward_passes': forwards, 'seconds': elapsed,
                 'device': torch.cuda.get_device_name(0)
                 if args.device.startswith('cuda') else args.device},
        'variants': int(len(model_column)),
        'backgrounds': len(cohort['backgrounds']),
        'seeds': seeds,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
    })
    np.savez_compressed(args.out / f'stability_scores_{args.arm}.npz',
                        score=model_column, group=panel['group'], site=panel['site'],
                        **group_error)


# --------------------------------------------------------------------------- #
# Endpoint 2: the frozen mutation-ranking anchor cohort
# --------------------------------------------------------------------------- #

def run_anchor(args) -> None:
    cohort = json.loads(args.cohort.read_bytes())
    if 'assays' not in cohort:
        raise SystemExit('unexpected anchor cohort schema')
    cohort_sha = sha256_file(args.cohort)
    tokenizer, model, identity = load_arm(args.model_root, args.arm, args.device)

    rows, excluded, per_variant = [], [], {}
    started, forwards, repeat_difference = time.time(), 0, 0.0
    for index, assay in enumerate(cohort['assays']):
        wildtype = assay['wildtype']
        if len(wildtype) > RESIDUE_CAPACITY:
            excluded.append({'assay': assay['assay'], 'wildtype_residues': len(wildtype),
                             'reason': 'wild type exceeds the declared residue capacity'})
            continue
        values = contrasts(model, tokenizer, wildtype, assay['mutants'], args.device, args.batch)
        forwards += len({position for identifier in assay['mutants']
                         for position, _, _ in substitutions(identifier, wildtype)})
        if index < REPEAT_BACKGROUNDS:
            again = contrasts(model, tokenizer, wildtype, assay['mutants'],
                              args.device, args.batch)
            repeat_difference = max(repeat_difference, float(np.abs(values - again).max()))
        measured = np.asarray(assay['measured'], dtype=float)
        lookup = np.asarray(assay['profile_scores'], dtype=float)
        model_rank = standardized_rank(values)
        model_spearman = correlation(model_rank, standardized_rank(measured))
        lookup_spearman = correlation(standardized_rank(lookup), standardized_rank(measured))
        rows.append({
            'assay': assay['assay'], 'cluster': assay['cluster'],
            'n_variants': len(values), 'wildtype_residues': len(wildtype),
            'model_spearman': model_spearman, 'lookup_spearman': lookup_spearman,
            'model_minus_lookup': None if model_spearman is None or lookup_spearman is None
            else model_spearman - lookup_spearman,
            'model_lookup_spearman': correlation(model_rank, standardized_rank(lookup)),
        })
        per_variant[assay['assay']] = values
    elapsed = time.time() - started

    summaries = {metric: summarize(rows, metric, bootstrap=BOOTSTRAP_DRAWS,
                                   seed=BOOTSTRAP_SEED + offset)
                 for offset, metric in enumerate(
                     ('model_spearman', 'lookup_spearman', 'model_minus_lookup',
                      'model_lookup_spearman'))}
    write_json(args.out / f'anchor_{args.arm}.json', {
        'schema': SCHEMA, 'endpoint': 'proteingym_mutation_ranking_common_support',
        'roster': 'exploratory', 'identity': identity, 'cohort_sha256': cohort_sha,
        'score': {'member': 'deterministic_p_equals_one_over_L',
                  'multi_substitution_rule': 'sum of per-position contrasts',
                  'repeat_max_abs_difference_nats': repeat_difference,
                  'assays_rescored': min(REPEAT_BACKGROUNDS, len(rows))},
        'support': {'assays': len(rows),
                    'families': len({row['cluster'] for row in rows}),
                    'variants': int(sum(row['n_variants'] for row in rows)),
                    'assays_in_cohort': len(cohort['assays']),
                    'excluded_over_capacity': excluded,
                    'residue_capacity': RESIDUE_CAPACITY},
        'cost': {'forward_passes': forwards, 'seconds': elapsed,
                 'device': torch.cuda.get_device_name(0)
                 if args.device.startswith('cuda') else args.device},
        'summaries': summaries, 'assays': rows,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
    })
    np.savez_compressed(args.out / f'anchor_scores_{args.arm}.npz',
                        **{name: value for name, value in per_variant.items()})


# --------------------------------------------------------------------------- #
# The comparison the roster exists for
# --------------------------------------------------------------------------- #

def run_agreement(args) -> None:
    """How far each diffusion arm's score sits from its own masked initialisation."""

    def load(endpoint: str, arm: str):
        path = args.out / f'{endpoint}_scores_{arm}.npz'
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    record = {'schema': SCHEMA, 'comparison': 'diffusion_arm_against_its_initialisation',
              'endpoints': {}}
    for endpoint, unit_key in (('stability', 'group'), ('anchor', None)):
        pairs = {}
        for arm, spec in EXPLORATORY_ROSTER.items():
            if not spec['initialised_from']:
                continue
            try:
                left, right = load(endpoint, arm), load(endpoint, spec['initialised_from'])
            except FileNotFoundError:
                continue
            if unit_key:
                units = left.pop(unit_key)
                # Both arms enter the identical folds through the identical
                # arm-independent baseline, so the two error reductions are paired
                # per held group and their difference is a paired contrast on one
                # support rather than two intervals read against each other.
                paired = {}
                for seed in SPLIT_SEEDS:
                    if f'S_M@{seed}' not in left or f'S_M@{seed}' not in right:
                        continue
                    baseline_gap = float(np.abs(left[f'S@{seed}']
                                                - right[f'S@{seed}']).max())
                    difference = right[f'S_M@{seed}'] - left[f'S_M@{seed}']
                    paired[str(seed)] = {
                        'increment_difference_kcal2_mol2': interval(difference),
                        'shared_baseline_max_abs_group_error_difference': baseline_gap}
                left.pop('site', None)
                right.pop(unit_key, None)
                right.pop('site', None)
                for key in [k for k in list(left) if k != 'score']:
                    left.pop(key)
                for key in [k for k in list(right) if k != 'score']:
                    right.pop(key)
                a, b = left['score'], right['score']
                per_unit = [correlation(standardized_rank(a[units == u]),
                                        standardized_rank(b[units == u]))
                            for u in sorted(set(units.tolist()))
                            if (units == u).sum() >= 3]
                labels = list(range(len(per_unit)))
            else:
                paired = {}
                rows = {name: {row['assay']: row for row in json.loads(
                    (args.out / f'anchor_{name}.json').read_bytes())['assays']}
                    for name in (arm, spec['initialised_from'])}
                shared = sorted(set(rows[arm]) & set(rows[spec['initialised_from']]))
                for metric in ('model_spearman', 'model_minus_lookup'):
                    difference = [
                        {'cluster': rows[arm][assay]['cluster'],
                         'value': rows[arm][assay][metric]
                         - rows[spec['initialised_from']][assay][metric]}
                        for assay in shared
                        if rows[arm][assay][metric] is not None
                        and rows[spec['initialised_from']][assay][metric] is not None]
                    paired[f'{metric}_difference'] = summarize(
                        difference, 'value', bootstrap=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
                clusters = {row['assay']: row['cluster'] for row in json.loads(
                    (args.out / f'anchor_{arm}.json').read_bytes())['assays']}
                keys = [k for k in sorted(set(left) & set(right)) if len(left[k]) >= 3]
                per_unit = [correlation(standardized_rank(left[k]), standardized_rank(right[k]))
                            for k in keys]
                labels = [clusters[k] for k in keys]
            finite = [(value, label) for value, label in zip(per_unit, labels)
                      if value is not None]
            pairs[f'{arm}_vs_{spec["initialised_from"]}'] = {
                'units': len(finite),
                'resampling_unit': 'stability group' if unit_key else '50% identity family',
                'within_unit_spearman': cluster_bootstrap(
                    [value for value, _ in finite], [label for _, label in finite],
                    resamples=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED),
                'paired_difference': paired,
            }
        record['endpoints'][endpoint] = pairs
    record['generated_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(args.out / 'agreement_with_initialisation.json', record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('endpoint', choices=('stability', 'anchor', 'agreement'))
    parser.add_argument('--arm', default='')
    parser.add_argument('--cohort', type=Path)
    parser.add_argument('--controls', type=Path)
    parser.add_argument('--profiles', type=Path)
    parser.add_argument('--model-root', type=Path, default=Path('models'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fit-device', default='cpu')
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.endpoint == 'agreement':
        run_agreement(args)
        return
    if args.arm not in EXPLORATORY_ROSTER:
        raise SystemExit(f'{args.arm} is not on the exploratory roster')
    if args.endpoint == 'stability':
        run_stability(args)
    else:
        run_anchor(args)


if __name__ == '__main__':
    main()
