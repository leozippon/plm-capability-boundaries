#!/usr/bin/env python3
"""Score the objective-tuned prediction side-effect on the R1 cohort.

Declared in ``docs/objective-alignment.md`` (third amendment) and
``src/capability/extensions/objective_tuned_prediction.py``. Three subcommands:

``probe``  measure the per-arm scoring cost on a small fixed slice of the variant table and
           extrapolate it to the full table, stating the basis. Scores nothing that is kept.
``floor``  propagate the coupling fitter's run-to-run spread into correlation units.
``run``    score the admitted arms, fit the ladder's own designs on the matched support and
           read the family under its simultaneous band. ``--dry-run`` proves that whole path on
           a small slice, against a scratch output, and claims no result.

The scalar is the one the 33-arm panel's ``M`` is: the summed native log-likelihood of the
packed target for a variant minus that of the wild type, float32, one sequence per forward.
Packing is the context-homologue module's, reached through an arm whose name is the
packing-representative of its family, and checked against the rendering the arm declares.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.core.arms import (INPUT_FORMAT_BOS_DIRECTION_SEQ, load_arm_spec,  # noqa: E402
                                      rendering_marker_ids)
from src.capability.generation import objective_tuned_arms as tuned_arms  # noqa: E402

COHORT_FILE = 'results/R5/context_mutation_rescue_20260923/cohort.json'
LOCAL_FILE = ('results/R5/local_context_20260923/20260923233257_e429ce7f31e4/'
              'lcgp_admission/local_context_measurement.json')
SOURCE = ROOT / 'results/extensions/information_progression_20261006'
PUBLISHED = ROOT / 'results/extensions/statistical_ladder_20261106'
DESCRIPTORS = ROOT / 'configs/explicit_statistics_cohorts.json'
DTYPE = 'float32'
RUN_DIR = ROOT / 'logs/operations/objective-tuned-prediction-20261110'
BUDGET = 1024
#: The archived extraction's own limit on a repeated single-sequence score.
REPEAT_LIMIT_NATS = 1e-3
#: The fixed cost-probe slice: assays at these quantiles of wild-type length, and this many
#: leading variants of each. Fixed here so the measurement is repeatable.
PROBE_ASSAYS, PROBE_VARIANTS = 12, 16
#: The dry-run slice: this many assays spread over the sorted table, this many variants each.
DRY_ASSAYS, DRY_VARIANTS = 40, 12
#: The packing family each rendering belongs to, as an arm name the packing module knows.
PACKING_REPRESENTATIVE = {INPUT_FORMAT_BOS_DIRECTION_SEQ: 'protgpt3-1.3b',
                          tuned_arms.INPUT_FORMAT_N_TO_C_CONTROL: 'progen2-small'}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_table():
    """The admitted variant table: per assay, wild type and variant sequences in sample order."""
    cohort = json.loads((ROOT / COHORT_FILE).read_text())
    admission = json.loads((ROOT / LOCAL_FILE).read_text())
    if sha256(ROOT / COHORT_FILE) != admission['cohort_sha256']:
        raise ValueError('cohort hash mismatch')
    by_name = {row['assay']: row for row in cohort['assays']}
    assays = [by_name[name] for name in sorted(admission['support']['assay_ids'])]
    return [dict(assay=a['assay'], cluster=a['cluster'], wildtype=a['wildtype'],
                 mutants=list(a['mutants']), sequences=list(a['sequences'])) for a in assays]


# ------------------------------------------------------------------ scoring


def stage46():
    spec = importlib.util.spec_from_file_location(
        'stage46', ROOT / 'scripts/capability/stages/context_homologue.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Scorer:
    """One loaded arm scored through the packing family its rendering belongs to."""

    def __init__(self, name: str, device: str):
        import torch
        from src.capability.context import context_homologue as ch
        from src.capability.readouts import readout_extraction as extraction
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        spec = tuned_arms.spec_for(name)
        if spec.input_format not in PACKING_REPRESENTATIVE:
            raise ValueError(f'{name}: no packing family for rendering {spec.input_format!r}')
        began = time.monotonic()
        loaded = load_arm_spec(spec, device=device, dtype=DTYPE)
        loaded.model.eval().requires_grad_(False)
        self.load_seconds = time.monotonic() - began
        # Packing is keyed by arm name; the representative carries the family's packing and
        # nothing else of it. The real tokenizer, model and rendering stay the arm's own.
        representative = PACKING_REPRESENTATIVE[spec.input_format]
        self.arm = dataclasses.replace(
            loaded, spec=dataclasses.replace(spec, name=representative))
        self.name, self.device, self._stage = name, device, stage46()
        self._pack, self._ch = extraction.pack_sequence, ch
        ch.require_position_budget(loaded.model.config, arm=name)
        self.contract = self.check_packing(loaded, spec)

    def check_packing(self, loaded, spec) -> dict:
        """The packed ids must open with exactly the marker ids the arm's rendering declares."""
        markers = tuple(rendering_marker_ids(loaded))
        ids, span, _ = self._pack(self.arm, 'ACDEFGHIKLMNPQRSTVWY')
        if tuple(ids[:len(markers)]) != markers or span[0] != len(markers) or span[1] != len(ids):
            raise ValueError(f'{self.name}: packed ids {ids[:len(markers) + 2]} do not open with '
                             f'the declared {spec.input_format} markers {list(markers)}')
        return dict(input_format=spec.input_format, marker_ids=list(markers),
                    packing_family=self.arm.name)

    def lengths(self, strings) -> list[int]:
        return [len(self._pack(self.arm, s)[0]) for s in strings]

    def log_likelihood(self, strings) -> np.ndarray:
        """Summed native log-likelihood of each packed target, one sequence per forward."""
        import torch
        out = []
        for s in strings:
            ids, span, _ = self._pack(self.arm, s)
            if len(ids) > BUDGET:
                raise ValueError('a packed target exceeds the 1024-position budget')
            logits, packed = self._stage._forward_rows(self.arm, [ids])
            out.append(-self._stage._target_nll(logits[0:1], packed[0:1], *span)['nll_sum'])
        if self.device.startswith('cuda'):
            torch.cuda.synchronize(self.device)
        return np.asarray(out, np.float64)

    def score_assay(self, row, *, variants=None) -> dict:
        """M for one assay's variants, with the repeat check the archived extraction made."""
        sequences = row['sequences'][:variants]
        strings = [row['wildtype']] + sequences
        likelihood = self.log_likelihood(strings)
        repeat = self.log_likelihood(strings[:3])
        drift = float(np.max(np.abs(repeat - likelihood[:3])))
        if drift > REPEAT_LIMIT_NATS:
            raise RuntimeError(f"{row['assay']}: a repeated score moved by {drift:g} nats")
        return dict(M=likelihood[1:] - likelihood[0], repeat_drift=drift)


# ------------------------------------------------------------------- probe


def probe_slice(table):
    ordered = sorted(table, key=lambda r: (len(r['wildtype']), r['assay']))
    picks = np.unique(np.round(np.linspace(0, len(ordered) - 1, PROBE_ASSAYS)).astype(int))
    return [ordered[i] for i in picks]


def extrapolate(points, population):
    """Predicted seconds for the full table from (tokens, seconds per sequence) points.

    Primary basis: a straight line of seconds per sequence on packed length, fitted over the
    slice's assays and summed over every sequence of the full table by its own length.
    Alternative basis, for the spread: the slice's mean seconds per sequence times the number
    of sequences, which is the right reading when a forward is launch-bound and length-blind.
    """
    x = np.asarray([p[0] for p in points], float)
    y = np.asarray([p[1] for p in points], float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(((y - fitted) ** 2).sum()) / ss if ss else 1.0
    lengths = np.asarray(population, float)
    linear = float(np.maximum(intercept + slope * lengths, y.min() * 0.5).sum())
    per_sequence = float(y.mean()) * len(lengths)
    return dict(linear_seconds=linear, mean_per_sequence_seconds=per_sequence, slope=float(slope),
                intercept=float(intercept), r2=r2, sequences=int(len(lengths)),
                tokens=int(lengths.sum()))


def probe(args):
    import torch
    table = load_table()
    arms = args.arms or [n for p in tuned_arms.PAIRS for n in p.arms]
    selected = probe_slice(table)
    report = dict(schema='objective_tuned_prediction_cost_probe', device_name=None,
                  dtype=DTYPE, batch_size=1, cohort_sha256=sha256(ROOT / COHORT_FILE),
                  slice=dict(assays=[r['assay'] for r in selected], variants=PROBE_VARIANTS,
                             rule=f'{PROBE_ASSAYS} assays at evenly spaced ranks of wild-type '
                                  f'length, the first {PROBE_VARIANTS} variants of each'),
                  table=dict(assays=len(table), variants=sum(len(r['mutants']) for r in table)),
                  arms={})
    for name in arms:
        entry = dict(status='measured')
        try:
            scorer = Scorer(name, args.device)
        except torch.cuda.OutOfMemoryError as error:
            entry.update(status='does_not_fit', error=str(error).splitlines()[0])
            report['arms'][name] = entry
            torch.cuda.empty_cache()
            continue
        report['device_name'] = torch.cuda.get_device_name(args.device)
        scorer.log_likelihood([selected[0]['wildtype']] * 2)  # warm-up, untimed
        points, drifts = [], []
        for row in selected:
            strings = [row['wildtype']] + row['sequences'][:PROBE_VARIANTS]
            began = time.monotonic()
            scorer.log_likelihood(strings)
            seconds = time.monotonic() - began
            length = float(np.mean(scorer.lengths(strings)))
            points.append((length, seconds / len(strings)))
        for row in selected[:2]:
            drifts.append(scorer.score_assay(row, variants=PROBE_VARIANTS)['repeat_drift'])
        population = [length for r in table
                      for length in scorer.lengths([r['wildtype']] + r['sequences'])]
        entry.update(load_seconds=scorer.load_seconds, contract=scorer.contract,
                     peak_allocated_gib=torch.cuda.max_memory_allocated(args.device) / 2**30,
                     slice_seconds_per_sequence=[dict(tokens=t, seconds=s) for t, s in points],
                     repeat_drift_max=max(drifts), **extrapolate(points, population))
        report['arms'][name] = entry
        print(json.dumps({name: {k: entry[k] for k in
                                 ('linear_seconds', 'mean_per_sequence_seconds', 'r2', 'load_seconds',
                                  'peak_allocated_gib')}}), flush=True)
        del scorer
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


# ------------------------------------------------------------- the ladder


def controls():
    """The frozen support with Profile, Potts and GEMME, the very tables the published
    ladder read; a table that is not byte-identical to its receipt is refused."""
    from scripts.capability.extensions import fit_statistical_ladder as fl
    from src.capability.extensions import overlap
    from src.capability.extensions import statistical_ladder as sl
    cohort = json.loads(DESCRIPTORS.read_text())['cohorts']['r1']
    scores = ROOT / cohort['out'] / 'accepted-scores'
    receipt = json.loads((ROOT / cohort['out'] / 'full/scores.json').read_text())
    if Path(receipt['directory']).resolve() != scores.resolve():
        raise ValueError('the published ladder did not read the declared score directory')
    data = fl.load_support(ROOT, SOURCE)
    data['sample_ids'] = np.asarray([row['sample_id'] for row in data['samples']])
    for name, column in (('potts', sl.POTTS_COLUMN), ('gemme', sl.GEMME_COLUMN)):
        table = fl.read_baseline_table(scores, name, column)
        if table['sha256'] != receipt['tables'][name]['sha256']:
            raise ValueError(f'{name} scores differ from the table the published ladder read')
        data['blocks'][{'potts': 'E', 'gemme': 'G'}[name]] = sl.join_scores(
            data['sample_ids'], table, columns=(column,))[column][:, None]
    data['target'] = overlap.rerank(data['measured'], data['aids'])
    return data


def fit_rows(data, scalars, mask, *, blocks=None, seeds=None):
    """The ladder's own designs M, PEG and MPEG for each arm on one matched row set.

    Returns the per-assay metric rows in the published schema and, per arm, the digest of
    the rows it was fitted on.
    """
    from src.capability.extensions import statistical_ladder as sl
    blocks = blocks or data['blocks']
    aids, families = data['aids'][mask], data['families'][mask]
    measured = data['measured'][mask]
    prepared = {name: sl.prepare_block(name, blocks[name][mask], aids) for name in 'PEG'}
    digest = hashlib.sha256('\n'.join(data['sample_ids'][mask]).encode()).hexdigest()
    rows = []
    for seed in seeds or sl.SEEDS:
        signature = data['signatures'][seed]
        shared = sl.fit_design(prepared, measured, aids, families, signature, 'PEG')[0]
        for arm, scalar in scalars.items():
            block = dict(prepared, M=sl.prepare_block('M', np.asarray(scalar)[mask][:, None], aids))
            predictions = {'PEG': shared}
            for name in ('M', 'MPEG'):
                predictions[name] = sl.fit_design(block, measured, aids, families,
                                                  signature, name)[0]
            rows += sl.assay_rows(predictions, measured, aids, families, model=arm, seed=seed,
                                  designs=('M', 'PEG', 'MPEG'))
    return rows, {arm: digest for arm in scalars}


def verify_reuse(rows, models):
    """A reused base refitted on the full support must reproduce its published rows."""
    published = {(r['model'], r['seed'], r['assay']): r['spearman']
                 for r in json.loads((PUBLISHED / 'full/assay-metrics.json').read_text())}
    worst = 0.0
    for row in rows:
        if row['model'] in models:
            for name, value in row['spearman'].items():
                worst = max(worst, abs(value - published[(row['model'], row['seed'],
                                                          row['assay'])][name]))
    if worst > 1e-9:
        raise ValueError(f'a reused arm differs from its published rows by {worst:g}')
    return worst


def family_mean(rows, arm, design):
    """One arm's family-mean rho for a design, from that arm's rows alone."""
    from src.capability.extensions import statistical_ladder as sl
    rows = [r for r in rows if r['model'] == arm]
    matrix = sl.absolute_matrix(rows, 'spearman', designs=(design,), models=[arm])['matrix']
    return float(matrix.mean())


# ------------------------------------------------------------------- floor


def floor(args):
    """Propagate the coupling fitter's run-to-run spread into correlation units.

    The Potts column is perturbed by Gaussian noise of the declared per-variant spread, the
    controls and the reference arm are refitted, and the shift in the family-mean rho of the
    controls and of both increments is recorded. The floor is the largest shift at the
    nominal spread; the iteration-cap scenario is recorded for reference only, because a
    capped background is booked incomplete and never used.
    """
    from src.capability.extensions import objective_tuned_bounds as bounds
    data = controls()
    mask = np.ones(len(data['sample_ids']), bool)
    reference = args.reference
    scalar = {reference: data['features'][reference]}

    def measure(blocks):
        rows, _ = fit_rows(data, scalar, mask, blocks=blocks)
        means = {d: family_mean(rows, reference, d) for d in ('M', 'PEG', 'MPEG')}
        return dict(controls=means['PEG'], forward=means['MPEG'] - means['PEG'],
                    reverse=means['MPEG'] - means['M'])

    nominal = measure(data['blocks'])
    report = dict(schema='objective_tuned_prediction_instrument_floor', reference=reference,
                  nominal=nominal, scenarios={})
    for label, factor in (('run_spread', 1), ('iteration_cap', bounds.CAPPED_FIT_FACTOR)):
        shifts = []
        for draw in range(args.draws):
            noise = np.random.default_rng(20261110 + draw).normal(
                0, bounds.COUPLING_RUN_SPREAD * factor, len(mask))
            blocks = dict(data['blocks'], E=data['blocks']['E'] + noise[:, None])
            moved = measure(blocks)
            shifts.append({k: moved[k] - nominal[k] for k in nominal})
        report['scenarios'][label] = dict(
            spread=bounds.COUPLING_RUN_SPREAD * factor, draws=args.draws, shifts=shifts,
            max_abs={k: max(abs(s[k]) for s in shifts) for k in nominal})
    report['correlation_floor'] = max(report['scenarios']['run_spread']['max_abs'].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(correlation_floor=report['correlation_floor'])))


# ------------------------------------------------------------------- score


def score_arm(name, table, scores, device, *, variants=None, assays=None):
    """M for every scored assay of one arm, resumable per assay.

    ``scores`` is the directory holding one subdirectory per arm. The checkpoint is loaded
    only if an assay is still missing, so a finished arm is read back without its weights.
    """
    directory = scores / name
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(arm=name, dtype=DTYPE, cohort_sha256=sha256(ROOT / COHORT_FILE),
                    variants=variants)
    rows = [r for r in table if assays is None or r['assay'] in assays]
    paths = {r['assay']: directory / (hashlib.sha256(r['assay'].encode()).hexdigest()[:20]
                                      + '.npz') for r in rows}
    scorer = Scorer(name, device) if not all(p.exists() for p in paths.values()) else None
    scored = {}
    for row in rows:
        path = paths[row['assay']]
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if json.loads(str(saved['identity'])) != identity:
                    raise ValueError(f'{path}: stored scores belong to a different run')
                scored[row['assay']] = saved['M']
            continue
        result = scorer.score_assay(row, variants=variants)
        np.savez_compressed(path, M=result['M'], repeat_drift=result['repeat_drift'],
                            identity=json.dumps(identity), contract=json.dumps(scorer.contract))
        scored[row['assay']] = result['M']
    return scored


def score(args):
    """Score whole arms on the full table; one process per card, pairs kept on one card."""
    table = load_table()
    for name in args.arms:
        began = time.monotonic()
        score_arm(name, table, args.scores, args.device)
        print(json.dumps(dict(arm=name, seconds=time.monotonic() - began)), flush=True)


def equivalence(args):
    """Whether this scorer reproduces a panel arm's archived ``M`` on whole assays.

    A reused base sits beside tuned arms scored here, so the two must be one instrument:
    the scalar is rescored on a few whole assays and its within-assay standardized rank is
    compared with the archived one, which is what the ladder consumes.
    """
    from src.capability.context.profile_increment import standardized_rank
    table, data = load_table(), controls()
    picks = probe_slice(table)[::max(1, PROBE_ASSAYS // args.assays)][:args.assays]
    scored = score_arm(args.arm, table, args.out.parent / f'{args.out.stem}-scores', args.device,
                       assays={r['assay'] for r in picks})
    cells = []
    for row in picks:
        index = np.flatnonzero(data['aids'] == row['assay'])
        archived = data['features'][args.arm][index]
        ranked = standardized_rank(scored[row['assay']])
        cells.append(dict(assay=row['assay'], variants=len(index),
                          max_rank_difference=float(np.max(np.abs(ranked - archived))),
                          disagreeing_variants=int((np.abs(ranked - archived) > 1e-6).sum())))
    report = dict(schema='objective_tuned_prediction_reuse_equivalence', arm=args.arm,
                  dtype=DTYPE, cells=cells)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report))


def reuse_scope():
    """What the archived-reuse check covers, stated no wider than the check."""
    path = RUN_DIR / 'reuse-equivalence.json'
    report = json.loads(path.read_text())
    cells = report['cells']
    if any(c['disagreeing_variants'] for c in cells):
        raise ValueError('the reuse check found disagreeing variants; the base cannot be reused')
    variants = sum(c['variants'] for c in cells)
    return dict(arm=report['arm'], whole_assays=len(cells), variants=variants,
                evidence_sha256=sha256(path),
                statement=(f"the archived {report['arm']} base reproduced exactly on {len(cells)} "
                           f"whole assays ({variants} variants) of the 25,728-variant table; "
                           "'one instrument' is established on that check and no wider"))


def run(args):
    from src.capability.extensions import objective_tuned_prediction as endpoint
    if args.out.exists():
        raise ValueError('never overwrite prior outputs; choose a new output directory')
    args.out.mkdir(parents=True)
    table = load_table()
    scores = args.scores or args.out / 'scores'
    data = controls()
    names = [a['assay'] for a in table]
    if names != [a for a, _ in dict.fromkeys(zip(data['aids'].tolist(), data['families'].tolist()))]:
        raise ValueError('the variant table does not list the frozen assays in order')
    row_assay = data['aids']
    offsets = {}
    for assay in table:
        first = int(np.flatnonzero(row_assay == assay['assay'])[0])
        offsets[assay['assay']] = (first, len(assay['mutants']))
    variants = assays = None
    if args.dry_run:
        step = max(1, len(table) // DRY_ASSAYS)
        assays = {r['assay'] for r in table[::step][:DRY_ASSAYS]}
        variants = DRY_VARIANTS
    scalars, availability = {}, {}
    for pair in endpoint.admitted_pairs():
        for arm in pair.arms:
            vector = np.full(len(row_assay), np.nan)
            if endpoint.reuses_archive(arm):
                vector[:] = data['features'][arm]
            else:
                for assay, values in score_arm(arm, table, scores, args.device,
                                               variants=variants, assays=assays).items():
                    first, _ = offsets[assay]
                    vector[first:first + len(values)] = values
            scalars[arm], availability[arm] = vector, np.isfinite(vector)
    support = endpoint.matched_support(availability, np.ones(len(row_assay), bool))
    mask = support['mask']
    models = {arm for p in endpoint.admitted_pairs() for arm in p.arms}
    rows, digests = fit_rows(data, {arm: np.nan_to_num(v) for arm, v in scalars.items()
                                    if arm in models}, mask)
    reused = {p.base for p in endpoint.admitted_pairs() if endpoint.reuses_archive(p.base)}
    reproduced = verify_reuse(rows, reused) if mask.all() and reused else None
    difference = endpoint.difference_matrix(rows)
    # The controls are one fit shared by every arm, so their performance is one number.
    controls_performance = {family_mean(rows, arm, 'PEG') for arm in sorted(models)}
    if len(controls_performance) != 1:
        raise ValueError('the controls\' performance differs between arms on one support')
    result = endpoint.band(difference['matrix'], difference['labels'],
                           dict(digests, controls=endpoint.support_digest(
                               data['sample_ids'][mask])),
                           controls_performance=controls_performance.pop())
    record = dict(schema='objective_tuned_prediction_result', dry_run=bool(args.dry_run),
                  claims_result=not args.dry_run, contract=endpoint.contract(),
                  support={k: v for k, v in support.items() if k != 'mask'},
                  archived_reuse_max_difference=reproduced, reuse_scope=reuse_scope(),
                  band=result,
                  # Descriptive only: each arm's own family-mean rho per design, outside the
                  # declared family. Forward = MPEG - PEG, reverse = MPEG - M.
                  descriptive_family_means={arm: {d: family_mean(rows, arm, d)
                                                  for d in ('M', 'PEG', 'MPEG')}
                                            for arm in sorted(models)})
    if args.dry_run:
        record['note'] = ('plumbing proof on a slice of whole-table rows: nothing here is a '
                          'result and no state below is a finding')
    path = args.out / ('dry-run.json' if args.dry_run else 'result.json')
    path.write_text(json.dumps(record, indent=2, allow_nan=False, default=float) + '\n')
    print(json.dumps(dict(path=str(path), status=result['status'], dry_run=bool(args.dry_run))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('probe')
    p.add_argument('--device', default='cuda:7')
    p.add_argument('--arms', nargs='*')
    p.add_argument('--out', type=Path, required=True)
    p.set_defaults(handler=probe)
    p = sub.add_parser('floor')
    p.add_argument('--reference', default='protgpt3-1.3b')
    p.add_argument('--draws', type=int, default=3)
    p.add_argument('--out', type=Path, required=True)
    p.set_defaults(handler=floor)
    p = sub.add_parser('equivalence')
    p.add_argument('--arm', default='protgpt3-1.3b')
    p.add_argument('--assays', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', type=Path, required=True)
    p.set_defaults(handler=equivalence)
    p = sub.add_parser('score')
    p.add_argument('--arms', nargs='+', required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--scores', type=Path, required=True)
    p.set_defaults(handler=score)
    p = sub.add_parser('run')
    p.add_argument('--device')
    p.add_argument('--scores', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    p.set_defaults(handler=run)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == '__main__':
    main()
