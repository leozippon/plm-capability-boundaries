#!/usr/bin/env python3
"""Emit the position-term recomputation campaigns, ordered by measured cost.

One cell per arm per cohort, dispatched through the frozen campaign queue. Three
properties of the ordering are deliberate.

Lanes are filled longest-first and grouped into slots of one cell per available
card, because the queue holds a barrier at the end of every slot: the campaign's
wall clock is the sum of each slot's slowest cell, and descending order minimises
that sum. The per-arm cost is the ``elapsed_seconds`` the arm's own archived
extraction recorded on the same cohort and the same card type, not an estimate.

Each cohort is pinned to the interpreter its archive was produced under --
torch 2.7.1 for the stability and pairwise cohorts, torch 2.9.1 for the anchor --
because the recomputation has to reproduce an archived float32 reduction and a
different build is a different reduction. The anchor's per-arm batch size is
likewise the archived one, for the same reason: batching changes the reduction.

Every cell declares its BLAS pool, and the extractor refuses to run without one.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

PYTHON_27 = '/opt/ac2/bin/python3'
BLAS_THREADS = 4

#: Staged validated interpreter, relative to the in-pod project root. The archived
#: plans, the anchor cohort and that interpreter all sit under that root, which is
#: host-local: it is read from TRANSFER_PROJECT_ROOT (exported by h200/h200_env.sh)
#: rather than named as a literal, and every path below resolves on use, so
#: importing this module on a machine with no cluster root is not an error.
STAGED_PYTHON_SUFFIX = 'runtimes/ct-20260905/bin/python'

STABILITY_PLAN_DIGEST = '283b4503ea166a3d61052bf68818c8aef162692ed3eb436a7a9864759a48763d'
PAIRWISE_PLAN_DIGEST = 'e338420f5df70143ccfc8d16ec5479a67b330ec35d36ea7c5965ea31a59e179c'
ANCHOR_DRIFT = '0.001'


def project_root() -> str:
    """The in-pod project root the archived plans and cohort are named against."""

    root = os.environ.get('TRANSFER_PROJECT_ROOT')
    if not root:
        raise SystemExit('TRANSFER_PROJECT_ROOT is not set, so the archived plan and cohort '
                         'paths have no root to resolve against; source h200/h200_env.sh, or '
                         'set TRANSFER_PROJECT_ROOT to the in-pod project root')
    return root.rstrip('/')


def anchor_python() -> str:
    """The interpreter the anchor cohort's archived reduction was produced under."""

    return f'{project_root()}/{STAGED_PYTHON_SUFFIX}'


def stability_plan() -> str:
    return f'{project_root()}/results/gate_stability_20260924/extraction_plan.json'


def pairwise_plan() -> str:
    return f'{project_root()}/data/pairwise_epistasis/extraction_plan_20260924_e338420f.json'


def anchor_cohort() -> str:
    return f'{project_root()}/data/context_mutation_rescue/cohort.json'

#: Interface stratum representatives for the release gate, one per stratum, each
#: the cheapest archived smoke cell of its stratum.
GATE_ARMS = {'amino_acid': 'progen2-small', 'bpe': 'gpt2', 'byte': 'bygpt5-small-en'}


def env(python: str) -> str:
    return (f'TRANSFER_PYTHON={python} OMP_NUM_THREADS={BLAS_THREADS} '
            f'MKL_NUM_THREADS={BLAS_THREADS}')


def slots(costs: dict[str, float], cards: int) -> list[list[str]]:
    """Descending cost grouped into slots of one cell per card."""

    order = sorted(costs, key=lambda arm: (-costs[arm], arm))
    return [order[i:i + cards] for i in range(0, len(order), cards)]


def rows(cohort: str, costs: dict[str, float], cards: int, argline) -> list[str]:
    out = []
    for index, group in enumerate(slots(costs, cards), start=1):
        for card, arm in enumerate(group):
            out.append('\t'.join([str(index), 'pt', str(card), *argline(arm)]))
    return out


def stability_args(arm: str) -> tuple[str, ...]:
    return ('extract_stability_singles.py', f'pt-stability-{arm}', env(PYTHON_27),
            f'manifest_{arm}.json',
            f'--plan {stability_plan()} --expect-plan-sha256 {STABILITY_PLAN_DIGEST} '
            f'--arm {arm} --keep-position-terms --blas-threads {BLAS_THREADS}')


def pairwise_args(arm: str) -> tuple[str, ...]:
    return ('extract_pairwise_epistasis.py', f'pt-pairwise-{arm}', env(PYTHON_27),
            f'manifest_{arm}.json',
            f'--plan {pairwise_plan()} --expect-plan-sha256 {PAIRWISE_PLAN_DIGEST} '
            f'--arm {arm} --keep-position-terms --blas-threads {BLAS_THREADS}')


def anchor_args(settings: dict[str, dict]):
    def build(arm: str) -> tuple[str, ...]:
        spec = settings[arm]
        return ('extract_frozen_readout.py', f'pt-anchor-{arm}', env(anchor_python()),
                f'manifest_{arm}.json',
                f'--cohort {anchor_cohort()} --arm {arm} --dtype {spec["dtype"]} '
                f'--batch-size {spec["batch"]} --max-score-drift {ANCHOR_DRIFT} '
                f'--max-feature-drift {ANCHOR_DRIFT} --keep-position-terms '
                f'--blas-threads {BLAS_THREADS}')
    return build


def gate_rows(anchor: dict[str, dict]) -> list[str]:
    """Nine cells: three interface strata on all three cohorts, one card, cheapest first.

    Each cell is a single background or the archived three-assay smoke draw, so
    the whole gate is minutes of card time and every cell has an archived
    counterpart to align against.
    """

    out = []
    slot = 0
    for stratum, arm in sorted(GATE_ARMS.items()):
        for cohort in ('stability', 'pairwise', 'anchor'):
            slot += 1
            if cohort == 'stability':
                stage, label, environment, expect, argline = stability_args(arm)
                argline += ' --background-limit 1'
                expect = f'progress_{arm}.json'
            elif cohort == 'pairwise':
                stage, label, environment, expect, argline = pairwise_args(arm)
                argline += ' --background-limit 1'
                expect = f'progress_{arm}.json'
            else:
                stage, label, environment, expect, argline = anchor_args(anchor)(arm)
                argline += ' --smoke-variants 3 --assay-limit 3 --smoke-length-strata'
                expect = f'progress_{arm}.json'
            label = f'gate-{cohort}-{arm}'
            out.append('\t'.join([str(slot), 'pt', '0', stage, label, environment, expect, argline]))
    return out


HEADER = """# Position-term recomputation: the per-token log-likelihood vector whose sum is an
# archived mutation-score scalar, retained under --keep-position-terms beside the
# unchanged projected archive. Identity, packing, scoring span, cohort digest, plan
# digest, seed, dtype and batch size are the archived ones; the code hash necessarily
# differs, so this writes to its own wave directory and resumes into no published one.
#
# {title}
# Cards: {cards}. Order: descending archived elapsed_seconds, grouped one cell per card,
# because the queue holds a slot barrier and the wall clock is the sum of slot maxima.
#
# slot\tkey\tgpu\tstage\tlabel\tenv\texpect\targs"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--costs', type=Path, required=True,
                        help='archive probe carrying per-arm elapsed_seconds per cohort')
    parser.add_argument('--anchor-settings', type=Path, required=True,
                        help='archive probe carrying the per-arm anchor batch size and dtype')
    parser.add_argument('--out', type=Path, default=ROOT / 'logs/generated-campaigns')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    costs = json.loads(args.costs.read_text())
    anchor_probe = json.loads(args.anchor_settings.read_text())
    anchor = {key.split('|', 1)[1]: value for key, value in anchor_probe['anchor'].items()
              if key.startswith('full|')}
    stability = {arm: float(row['elapsed_s']) for arm, row in costs['stability'].items()}
    pairwise = {arm: float(row['elapsed_s']) for arm, row in costs['pairwise'].items()}
    roster = sorted(stability)
    if sorted(pairwise) != roster:
        raise SystemExit('the stability and pairwise archives disagree on the roster')
    anchor = {arm: spec for arm, spec in anchor.items() if arm in set(roster)}
    if sorted(anchor) != roster:
        raise SystemExit('the anchor archive does not cover the roster exactly')
    # A merged sharded manifest carries no elapsed time of its own; its cost is
    # taken from the same arm's stability elapsed, scaled by the ratio the arms
    # that have both measured. Ordering only, never a reported quantity.
    measured = {arm: spec['elapsed'] for arm, spec in anchor.items() if spec['elapsed']}
    ratio = (sum(measured.values())
             / sum(stability[arm] for arm in measured))
    anchor_costs = {arm: float(spec['elapsed']) if spec['elapsed'] else stability[arm] * ratio
                    for arm, spec in anchor.items()}

    plans = [
        ('campaign_position_terms_anchor.tsv', 'ProteinGym anchor cohort, 4 cards', 4,
         anchor_costs, anchor_args(anchor)),
        ('campaign_position_terms_stability.tsv', 'Folding-stability singles cohort, 4 cards', 4,
         stability, stability_args),
        ('campaign_position_terms_pairwise.tsv', 'Pairwise cycle cohort, 2 cards', 2,
         pairwise, pairwise_args),
    ]
    written = {}
    for name, title, cards, cost, argline in plans:
        body = rows(name, cost, cards, argline)
        text = HEADER.format(title=title, cards=cards) + '\n' + '\n'.join(body) + '\n'
        (args.out / name).write_text(text)
        written[name] = {'cells': len(body), 'slots': len(slots(cost, cards)),
                         'wall_clock_seconds': sum(max(cost[a] for a in g)
                                                   for g in slots(cost, cards)),
                         'gpu_seconds': sum(cost.values())}
    gate = gate_rows(anchor)
    text = (HEADER.format(title='Release gate: three interface strata, three cohorts, 1 card', cards=1)
            + '\n' + '\n'.join(gate) + '\n')
    (args.out / 'campaign_position_terms_gate.tsv').write_text(text)
    written['campaign_position_terms_gate.tsv'] = {'cells': len(gate), 'slots': len(gate)}
    print(json.dumps({'anchor_shard_cost_ratio': ratio, 'campaigns': written}, indent=1))


if __name__ == '__main__':
    main()
