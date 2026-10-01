#!/usr/bin/env python3
"""Frozen ProteinGym homolog-context mutation rescue diagnostic (128 variants)."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.capability.context import context_homologue as ch
from src.capability.context import profiles as P
from src.capability.models.fitness import load_assay, PROTEINGYM_ROOT
from src.capability.context.homology import parse_hits, ALIGNMENT_FIELDS
from src.capability.core.io import write_json


from scripts.capability.entrypoints import stage_path

def module(filename):
    spec = importlib.util.spec_from_file_location(filename, stage_path(ROOT, filename))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(values):
    return hashlib.sha256('\n'.join(values).encode()).hexdigest()


def rho(x, y):
    result = float(spearmanr(x, y).statistic)
    return result if np.isfinite(result) else None


def prepare(args):
    base = args.baseline
    catalogue = json.loads((base / 'wildtypes.json').read_text())
    lookup = json.loads((base / 'lookup.json').read_text())
    metadata = json.loads((base / 'profiles.json').read_text())
    stored = np.load(base / 'profiles.npz')
    hits = parse_hits(base / 'corpus_hits.tsv', fields=ALIGNMENT_FIELDS)
    candidates = {key: [] for key in catalogue['wildtypes']}
    all_subjects = {key: set() for key in candidates}
    for hit in hits:
        all_subjects[hit.query].add(hit.subject)
        if (hit.qend-hit.qstart+1) / hit.qlen < .8 or not 30 <= hit.identity_over_query < 80:
            continue
        sequence = hit.sseq_gapped.replace('-', '').upper()
        if not set(sequence) <= set(P.AA20):
            continue
        candidates[hit.query].append(dict(subject=hit.subject, sequence=sequence,
            identity=hit.identity_over_query, bitscore=hit.bitscore))
    for key, rows in candidates.items():
        unique = {}
        for row in sorted(rows, key=lambda x: (-x['bitscore'], x['subject'])):
            unique.setdefault(row['sequence'], row)
        candidates[key] = list(unique.values())[:32]
    # Donors are separate WT families and absent from the target's detected hit list.
    pool = [(key, row) for key, rows in candidates.items() for row in rows]
    stage20 = module('retrieval_bound.py')
    background = np.array([catalogue['corpus']['background'][aa] for aa in P.AA20])
    rows = []
    for index, old in enumerate(lookup['assays']):
        assay = load_assay(old['assay'], n=lookup['settings']['variants'],
            seed=lookup['settings']['seed'] + index, directory=args.proteingym_dir)
        if digest(assay.mutants) != old['mutant_digest']:
            raise ValueError(f"Original mutation draw drift: {assay.name}")
        chosen = np.sort(np.random.default_rng(20260923 + index).permutation(len(assay.mutants))[:args.variants])
        key = old['wildtype_id']
        profile = stage20._profile_from_store(stored, key, assay.wildtype, metadata)
        scores = P.profile_scores(profile, background, assay.substitutions, alpha=lookup['settings']['alpha'])
        hom = candidates[key]
        donor = [r for k, r in pool if catalogue['wildtypes'][k]['cluster'] != old['cluster']
                 and r['subject'] not in all_subjects[key]]
        # Bound the pool deterministically, by residue-length and composition proximity.
        if hom:
            length = len(hom[0]['sequence'])
            donor = sorted(donor, key=lambda r: (abs(len(r['sequence'])-length), r['subject']))[:256]
        else:
            donor = []
        rows.append(dict(assay=assay.name, wildtype_id=key, cluster=old['cluster'],
            wildtype=assay.wildtype, mutants=[assay.mutants[i] for i in chosen],
            sequences=[assay.sequences[i] for i in chosen], measured=assay.scores[chosen].tolist(),
            profile_scores=scores[chosen].tolist(), original_mutant_digest=old['mutant_digest'],
            mutant_digest=digest([assay.mutants[i] for i in chosen]),
            homolog_candidates=hom, unrelated_candidates=donor))
    payload = dict(schema_version='context_mutation_rescue_v1', seed=20260923,
        variants=args.variants, baseline_sha256={n: sha(base/n) for n in
        ['lookup.json','wildtypes.json','profiles.json','profiles.npz','corpus_hits.tsv']}, assays=rows)
    write_json(args.out, payload)
    print(f"Prepared {len(rows)} assays; {sum(bool(r['homolog_candidates']) for r in rows)} have eligible homologs", flush=True)


def plan_row(arm, row, budget):
    target_ids = [ch.item_ids(arm, s, modality='protein') for s in [row['wildtype']] + row['sequences']]
    prefix = ch.row_prefix_ids(arm)
    room = budget - len(prefix) - max(map(len, target_ids))
    if room <= 0:
        return None, 'full target exceeds context budget'
    donors = []
    for candidate in row['unrelated_candidates']:
        ids = ch.item_ids(arm, candidate['sequence'], modality='protein')
        if len(ids) <= room:
            donors.append((candidate, ids))
    for homolog in row['homolog_candidates']:
        ids = ch.item_ids(arm, homolog['sequence'], modality='protein')
        if len(ids) > room:
            continue
        eligible = [(d, d_ids) for d, d_ids in donors if len(d_ids) == len(ids)]
        if not eligible:
            continue
        hcomp = np.array([homolog['sequence'].count(aa)/len(homolog['sequence']) for aa in P.AA20])
        eligible.sort(key=lambda pair: (float(np.square(np.array([pair[0]['sequence'].count(aa)/len(pair[0]['sequence']) for aa in P.AA20])-hcomp).sum()), pair[0]['subject']))
        for donor, donor_ids in eligible:
            if ch.longest_common_substrings(row['wildtype'], [donor['sequence']])[0] >= 10:
                continue
            return dict(homolog=homolog, unrelated=donor, context_tokens=len(ids),
                max_lcs=int(ch.longest_common_substrings(row['wildtype'], [homolog['sequence']])[0]),
                context_ids={'no_context': [], 'unrelated': donor_ids, 'homolog': ids}), None
    return None, 'no eligible homolog and exact-token-matched unrelated context within budget'


def score(args):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.budget != 1024:
        raise ValueError('The frozen diagnostic requires a 1024-position budget')
    cohort = json.loads(args.cohort.read_text())
    handle = ch.tokenizer_arm(args.arm)
    selected, skipped = [], []
    for row in cohort['assays']:
        plan, reason = plan_row(handle, row, args.budget)
        if plan is None:
            skipped.append(dict(assay=row['assay'], reason=reason))
        else:
            selected.append((row, plan))
    if args.assay_limit:
        selected = selected[:args.assay_limit]
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / f'plan_{args.arm}.json', dict(cohort_sha256=sha(args.cohort),
        selected=[dict(assay=r['assay'], **p) for r,p in selected], skipped=skipped))
    print(f'{args.arm}: planned {len(selected)} assays, excluded {len(skipped)}', flush=True)
    if not selected:
        raise RuntimeError('No assay has matched contexts')
    if args.phase == 'plan':
        return
    stage46 = module('context_homologue.py')
    arm = stage46.load_scorable_arm(args.arm, device=args.device, dtype=args.dtype)
    ch.require_position_budget(arm.model.config, arm=args.arm)
    results = []
    checks = {}
    for row, plan in selected:
        target_strings = [row['wildtype']] + row['sequences']
        targets = [ch.item_ids(arm, s, modality='protein') for s in target_strings]
        spans = [ch.target_span(arm, ids, record=s) for ids,s in zip(targets,target_strings)]
        scores, wt = {}, {}
        for condition, context in plan['context_ids'].items():
            prefix = ch.row_prefix_ids(arm) + context
            packed = [prefix + ids for ids in targets]
            sums = []
            for start in range(0, len(packed), args.batch_size):
                batch = packed[start:start+args.batch_size]
                logits, ids = stage46._forward_rows(arm, batch)
                for j in range(len(batch)):
                    left, right = spans[start+j]
                    result = stage46._target_nll(logits[j:j+1], ids[j:j+1], len(prefix)+left, len(prefix)+right)
                    sums.append(-result['nll_sum'])
                del logits, ids
            if not results:
                single = []
                for j in range(min(3, len(packed))):
                    logits, ids = stage46._forward_rows(arm, [packed[j]])
                    left, right = spans[j]
                    check = stage46._target_nll(logits, ids, len(prefix)+left, len(prefix)+right)
                    single.append(-check['nll_sum'])
                    del logits, ids
                gaps = [abs(sums[j]-single[j])/(spans[j][1]-spans[j][0]) for j in range(len(single))]
                checks[condition] = dict(max_batch_single_nats_per_token=max(gaps),
                    mutant_delta_difference_nats=[(sums[j]-sums[0])-(single[j]-single[0]) for j in range(1,len(single))])
                if max(gaps) > .02 or max(abs(x) for x in checks[condition]['mutant_delta_difference_nats']) > .001:
                    raise RuntimeError(f'Batch/single target NLL check failed: {checks[condition]}')
            if not np.isfinite(sums).all():
                raise RuntimeError(f"Nonfinite likelihood {row['assay']} {condition}")
            wt[condition] = sums[0]
            scores[condition] = (np.array(sums[1:])-sums[0]).tolist()
        result = {k: row[k] for k in ['assay','wildtype_id','cluster','mutants','measured','profile_scores','mutant_digest','original_mutant_digest']}
        result.update(scores=scores, wt_log_likelihood=wt,
            wt_scored_tokens=spans[0][1]-spans[0][0], context_tokens=plan['context_tokens'],
            homolog_identity=plan['homolog']['identity'], max_lcs=plan['max_lcs'],
            spearman={c:rho(s,row['measured']) for c,s in scores.items()}, lookup_spearman=rho(row['profile_scores'], row['measured']))
        results.append(result)
        write_json(args.out / f'scores_{args.arm}.json', dict(schema_version='context_mutation_rescue_v1',
            arm=args.arm, dtype=args.dtype, caveats=ch.CAVEATS.get(args.arm), scoring_stratum='target_only_native_packed_residue_span',
            cohort_sha256=sha(args.cohort), baseline_sha256=cohort['baseline_sha256'],
            budget=args.budget, batch_size=args.batch_size, batch_single_checks=checks, status='complete' if len(results)==len(selected) else 'running',
            assays=results, skipped=skipped, assay_limit=args.assay_limit))
        print(f"{len(results)}/{len(selected)} {row['assay']} {result['spearman']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare','plan','score'])
    parser.add_argument('--baseline', type=Path, default=ROOT/'results/R5/retrieval_bound_20260807')
    parser.add_argument('--proteingym-dir', type=Path, default=PROTEINGYM_ROOT)
    parser.add_argument('--cohort', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm')
    parser.add_argument('--variants', type=int, default=128)
    parser.add_argument('--budget', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--dtype', default='float32')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--assay-limit', type=int, default=0)
    args = parser.parse_args()
    (prepare if args.phase=='prepare' else score)(args)

if __name__ == '__main__':
    main()
