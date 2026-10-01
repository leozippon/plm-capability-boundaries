#!/usr/bin/env python3
"""Write the extraction campaign manifests for the two nested gates.

Both gates score the same frozen 33-arm roster on their own cohort, one forward
per sequence at batch size one, through the admitted extraction entry point
``extract_stability_singles.py`` rather than a second copy of it: the two
cohorts' extraction plans carry the same schema, so the plan is the only thing
that differs between them.

The roster is the frozen one and every arm is dispatched. No arm is selected by
any outcome, and none is dropped for being slow or large.

**Waves.** A campaign runner sees only the allocation of the pod it runs in, so
one manifest is written per lane group rather than one for the whole cluster, and
each carries a distinct basename. Arms are laid out round-robin over the lanes
after sorting by descending declared parameter scale, so the largest checkpoints
land in different lane groups and each lane's slot sequence is close to
equal-cost. Lane counts are the argument, not a constant: pods are disposable and
the next allocation may not match.

``--keep-full-features`` is on every cell. The representation cells of both gates
are provisional pending the readout-class reassessment, and retaining the
unprojected per-state block outputs is what makes that reassessment a refit
rather than a second forward pass.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.mutation.external_confirmation import BLAS_THREADS, BLAS_THREAD_VARIABLES
from src.capability.interactions.pairwise_epistasis import ARM_DTYPE, ROSTER

STAGE = 'extract_stability_singles.py'

#: The staged interpreter every cell runs under, relative to the in-pod project
#: root. The pod environment file's own fallback is the pod image's ``python3``
#: -- Python 3.12.7 with numpy 2.1.2 -- which is not the validated runtime, so
#: the interpreter is named in the manifest's own environment column rather than
#: left to that fallback; h200_env.sh honours a pre-set value. The runner
#: expands nothing in that column, so the path is written in full. It is part of
#: the campaign's identity for the same reason the snapshot is.
STAGED_PYTHON_SUFFIX = 'runtimes/ct-20260905/bin/python'

#: Declared parameter scale of each roster arm, in millions, used only to order
#: the round-robin layout so that the largest checkpoints do not share a lane
#: group. It orders dispatch and enters no measurement.
ARM_SCALE_MILLIONS: dict[str, float] = {
    'qwen2.5-32b': 32500, 'galactica-30b': 30000, 'progen2-xlarge': 6400,
    'qwen3-8b-base': 8200, 'llama-2-7b': 6700, 'prollama': 6700,
    'prollama-stage-1': 6700, 'proteinglm-7b-clm': 7000, 'qwen2.5-7b': 7600,
    'galactica-6.7b': 6700, 'instructprotein': 6700, 'progen3-3b': 3000,
    'llama-3.2-3b': 3200, 'progen2-large': 2700, 'rita-xl': 1200,
    'gpt2-xl': 1560, 'protgpt3-1.3b': 1300, 'galactica-1.3b': 1300,
    'gpt2-large': 774, 'protgpt2': 774, 'progen2-base': 764, 'progen2-medium': 764,
    'qwen2.5-0.5b': 494, 'qwen2.5-0.5b-instruct': 494, 'gpt2-medium': 355,
    'bygpt5-base-en': 200, 'gpt2': 124, 'galactica-125m': 125, 'progen3-112m': 112,
    'progen2-small': 151, 'bygpt5-medium-en': 100, 'dialogpt-small': 117,
    'bygpt5-small-en': 50,
}

#: The gates this builder writes manifests for, with the results directory each
#: one's declared cohort and plan were written to.
GATES = {
    'ec': {'directory': 'external_confirmation_20260924',
           'what': 'Domainome 1.0 external confirmation, abundance by aPCA'},
    'rh': {'directory': 'remote_homology_20260924',
           'what': 'MGnify remote-homology gate, cDNA-display proteolysis stability'},
}


def header(gate: str, lane: str, plan: dict, declaration: dict, cells: list[dict],
           project_root: str) -> str:
    support = declaration['support']
    lines = [
        f"# Nested-gate extraction, {GATES[gate]['what']}.",
        f"# Lane group {lane}: {len(cells)} of the frozen {len(ROSTER)}-arm roster, "
        f"{len({c['gpu'] for c in cells})} cards, "
        f"{max(c['slot'] for c in cells)} slots.",
        f"# Cohort sha256 {declaration['cohort_sha256']}.",
        f"# Extraction plan content digest {declaration['plan_content_sha256']}.",
        f"# Endpoint digest {declaration['endpoint_digest']}.",
        f"# Support: {support.get('domains_retained', support.get('backgrounds'))} units, "
        f"{support.get('family_groups')} family groups, "
        f"{support.get('variants_drawn', support.get('variants'))} variants, "
        f"{plan and sum(1 + len(b['variants']) for b in plan['backgrounds'])} sequences "
        f"and {support.get('residues_per_arm')} residues per arm.",
        '#',
        '# Every cell carries --keep-full-features: the representation cells are '
        'provisional',
        '# pending the readout-class reassessment, and the unprojected per-state block',
        '# outputs are what make that reassessment a refit rather than a second forward.',
        '#',
        '# The plan carries sequences and state indices only. The entry point verifies the',
        '# plan digest, refuses a plan carrying a measurement field, and re-derives every',
        '# declared state against its own wild type before a forward pass.',
        '#',
        '# Every cell names the staged ct-20260905 interpreter and pins the BLAS thread',
        f'# count to {BLAS_THREADS}: a float32 matmul reduction order depends on the thread',
        '# count, and the admitted projection ran at that value.',
        '#',
        f'# Paths are resolved against {project_root}.',
        '#',
        '# slot\tkey\tgpu\tstage\tlabel\tenv\texpect\targs',
    ]
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', choices=sorted(GATES), required=True)
    parser.add_argument('--results-root', type=Path, default=ROOT / 'results')
    parser.add_argument('--project-root', default=os.environ.get('TRANSFER_PROJECT_ROOT'),
                        help='the in-pod project root every manifest path resolves against; '
                             'defaults to TRANSFER_PROJECT_ROOT, which h200/h200_env.sh exports')
    parser.add_argument('--lane', action='append', required=True, metavar='TAG=CARDS',
                        help='one lane group per pod allocation, as tag=comma-separated '
                             'card indices; repeatable')
    parser.add_argument('--key', default='cc', help='which --snapshot the cells come from')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'scripts/capability')
    args = parser.parse_args()

    if not args.project_root:
        parser.error('--project-root is not set and TRANSFER_PROJECT_ROOT is not in the '
                     'environment, so no manifest path has a root to resolve against; pass '
                     '--project-root, or source h200/h200_env.sh')

    directory = GATES[args.gate]['directory']
    declaration = json.loads((args.results_root / directory / 'support_declaration.json').read_bytes())
    plan = json.loads((args.results_root / directory / 'extraction_plan.json').read_bytes())
    plan_path = f'{args.project_root}/results/{directory}/extraction_plan.json'

    lanes = []
    for entry in args.lane:
        tag, _, cards = entry.partition('=')
        if not tag or not cards:
            raise SystemExit(f'--lane takes tag=cards, got {entry!r}')
        lanes.append((tag, [int(value) for value in cards.split(',')]))
    slots = [(tag, card) for tag, cards in lanes for card in cards]
    if not slots:
        raise SystemExit('no lane carries a card')

    # The numeric environment and the interpreter, declared in the manifest so a
    # cell cannot inherit the pod image's python or an unpinned BLAS thread pool.
    environment = ' '.join([
        f'TRANSFER_PYTHON={args.project_root}/{STAGED_PYTHON_SUFFIX}',
        *(f'{name}={BLAS_THREADS}' for name in BLAS_THREAD_VARIABLES)])

    order = sorted(ROSTER, key=lambda arm: (-ARM_SCALE_MILLIONS[arm], arm))
    missing = sorted(set(ROSTER) - set(ARM_SCALE_MILLIONS))
    if missing:
        raise SystemExit(f'no declared scale for {missing}')
    assignment: dict[tuple[str, int], list[str]] = {slot: [] for slot in slots}
    for index, arm in enumerate(order):
        assignment[slots[index % len(slots)]].append(arm)

    written = []
    for tag, cards in lanes:
        cells = []
        depth = max(len(assignment[(tag, card)]) for card in cards)
        for position in range(depth):
            for card in cards:
                arms = assignment[(tag, card)]
                if position >= len(arms):
                    continue
                arm = arms[position]
                cells.append({
                    'slot': position + 1, 'key': args.key, 'gpu': card, 'stage': STAGE,
                    'label': f'{args.gate}-{arm}', 'env': environment,
                    'expect': f'manifest_{arm}.json',
                    'args': ' '.join([
                        '--plan', plan_path,
                        '--expect-plan-sha256', declaration['plan_content_sha256'],
                        '--arm', arm, '--keep-full-features']),
                })
        path = args.out_dir / f'campaign_nested_{args.gate}_extract_{tag}.tsv'
        body = header(args.gate, tag, plan, declaration, cells, args.project_root)
        for cell in cells:
            body += '\t'.join(str(cell[field]) for field in
                              ('slot', 'key', 'gpu', 'stage', 'label', 'env', 'expect',
                               'args')) + '\n'
        path.write_text(body)
        written.append({'manifest': str(path.relative_to(ROOT)), 'lane': tag,
                        'cards': cards, 'cells': len(cells),
                        'slots': max(c['slot'] for c in cells),
                        'arms': [c['label'].split('-', 1)[1] for c in cells]})

    dispatched = sorted(arm for entry in written for arm in entry['arms'])
    if dispatched != sorted(ROSTER):
        raise SystemExit('the lane groups do not cover the frozen roster exactly')
    print(json.dumps({
        'gate': args.gate,
        'cohort_sha256': declaration['cohort_sha256'],
        'plan_content_sha256': declaration['plan_content_sha256'],
        'roster_arms': len(ROSTER),
        'bfloat16_arms': sorted(ARM_DTYPE),
        'sequences_per_arm': sum(1 + len(b['variants']) for b in plan['backgrounds']),
        'manifests': written,
    }, indent=1))


if __name__ == '__main__':
    main()
