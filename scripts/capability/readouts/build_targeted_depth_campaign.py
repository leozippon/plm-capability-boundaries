#!/usr/bin/env python
"""Plan and write the targeted three-depth extraction for the two depth-blocked gates.

Two subcommands.

``estimate`` needs no selection and answers the allocation question: what a
targeted extraction of each cohort would cost in GPU-hours and in retained-state
storage. The GPU figure does not depend on which depth is selected, because every
hooked block is read from the same forward pass, so it is the measured wall clock
of the extraction that already ran on that cohort. The storage figure is quoted
at three depths per arm, which is the upper bound.

``manifests`` needs the reassessment's own per-arm depth selection and refuses
without it. It writes one campaign lane per arm plus a plan record, and it marks
every lane **blocked** unless the extractor it names already accepts the depth
option: a lane naming an option its stage does not define is refused by argparse
with exit 2, which is how three of the four cells of the first full-width
retention dispatch were lost.

Nothing is dispatched here, and no model is loaded.

    python scripts/capability/readouts/build_targeted_depth_campaign.py estimate \
      --out archive/logs/shared/recomputation_20260924/targeted_depth_estimate.json

    python scripts/capability/readouts/build_targeted_depth_campaign.py manifests \
      --selection <reassessment selection>.json \
      --out archive/logs/shared/recomputation_20260924/targeted_depth
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import write_json  # noqa: E402
from src.capability.readouts.recomputation import RecomputationRefused, load_selection  # noqa: E402
from scripts.capability.entrypoints import stage_path  # noqa: E402
from src.capability.readouts.targeted_depth import (  # noqa: E402
    ARM_COSTS, COHORTS, REQUIRED_EXTRACTOR_OPTION, campaign_rows, declaration_digest,
    depth_agnostic_estimate, estimate, every_block, extractor_accepts_depth, render_campaign)

#: Staged validated interpreter the default runtime string names, relative to the
#: in-pod project root. The root itself is host-local and comes from
#: ``TRANSFER_PROJECT_ROOT`` (exported by ``h200/h200_env.sh``) rather than from a
#: literal here. Both it and the runtime string are overridable, because a lane is
#: only queueable against the allocation it was written for.
STAGED_PYTHON_SUFFIX = 'runtimes/ct-20260905/bin/python'

HEADER = """Targeted three-depth extraction for the {gate} gate, on {cohort}
({rows}). One cell per arm at that arm's own admitted precision setting and batch
size one, hooking the depth the readout reassessment selected for that arm
together with the two depths its own retained manifest records as admitted, so
the new cells are comparable with the published ones on the same arrays. An arm
whose selected depth is already one of those two hooks two rather than three,
and its lane names no extra depth.

Why a targeted set and not a stack: the representation extraction of this gate
hooked (L-1)//2 and L-1 only, so a depth selection other than those two cannot be
applied to it at all (audit L53). Capturing a third block costs the same single
forward pass per sequence, so the GPU cost is this cohort's own measured
extraction cost, {hours} GPU-hours over {arms} arms, and only the retained
storage grows, to {gib} GiB at the selected depths against 29.75 GiB at the two
admitted ones.

Cells are ordered longest-first by the measured wall clock of the same arm's
completed extraction, which reads a measured duration and no fitted outcome.

{blocked}"""

NO_EXTRA_NOTE = ("Queueable as written: every selected depth coincides with one this arm's "
                 "extraction already hooked, so no lane names {option} and {stage} needs no "
                 "change. This campaign then re-extracts nothing new and exists only to record "
                 "that the selection required nothing.")

BLOCKED_NOTE = ("NOT QUEUEABLE AS WRITTEN. {stage} computes its hooked blocks as "
                "[(len(blocks) - 1) // 2, len(blocks) - 1] and defines no depth option, so every "
                "lane below names {option}, which that stage would refuse with exit 2. The lanes "
                "are written now so the campaign is a dispatch rather than a derivation once the "
                "option exists; they must not be queued before it does.")

READY_NOTE = "{stage} accepts {option}, so these lanes are queueable as written."


def _estimate(args) -> int:
    record = dict(schema='d1_targeted_depth_estimate_v1',
                  declaration_sha256=declaration_digest(),
                  extractor_option=REQUIRED_EXTRACTOR_OPTION,
                  extractors={gate: dict(
                      stage=cohort.extractor,
                      accepts_depth_option=extractor_accepts_depth(
                          stage_path(REPO_ROOT, cohort.extractor)))
                      for gate, cohort in sorted(COHORTS.items())},
                  gates={gate: estimate(gate) for gate in sorted(COHORTS)},
                  depth_agnostic=depth_agnostic_estimate())
    combined_hours = sum(entry['total_gpu_hours'] for entry in record['gates'].values())
    combined_bytes = sum(entry['total_retained_bytes'] for entry in record['gates'].values())
    record['combined'] = dict(gpu_hours=round(combined_hours, 3),
                              retained_bytes=combined_bytes,
                              retained_gib=round(combined_bytes / 2 ** 30, 2),
                              arms=len(ARM_COSTS), cells=2 * len(ARM_COSTS))
    write_json(args.out, record)
    print(json.dumps(dict(out=str(args.out), declaration_sha256=record['declaration_sha256'],
                          combined=record['combined'],
                          per_gate={gate: dict(gpu_hours=entry['total_gpu_hours'],
                                               retained_gib=entry['total_retained_gib'],
                                               arms=entry['arms'],
                                               longest_arm=entry['longest_arm'])
                                    for gate, entry in record['gates'].items()},
                          extractors=record['extractors'],
                          depth_agnostic={label: entry['total_gib'] for label, entry
                                          in record['depth_agnostic']['variants'].items()}),
                     indent=1))
    return 0


def _manifests(args) -> int:
    if args.every_block:
        if args.selection is not None:
            raise SystemExit('--every-block hooks every block of every arm, so it takes no '
                             'selection; pass one or the other')
        depths = {arm: sorted(every_block(arm)) for arm in ARM_COSTS}
        return _write(args, depths, label='depth-agnostic')
    if args.selection is None:
        raise SystemExit('pass --selection with the reassessment\'s per-arm depths, or '
                         '--every-block for the depth-agnostic pass that needs no selection')
    selection = load_selection(args.selection)
    depths: dict[str, int] = {}
    for arm, choice in selection.items():
        if arm not in ARM_COSTS:
            continue
        if choice.depth_index is None:
            raise SystemExit(f'{arm}: the selection names depth axis {choice.depth_axis!r} with no '
                             'depth index, so there is no depth to extract at; a targeted '
                             'extraction is planned only for an indexed depth axis')
        depths[arm] = int(choice.depth_index)
    missing = sorted(set(ARM_COSTS) - set(depths))
    if missing:
        raise SystemExit(f'the selection declares no indexed depth for {len(missing)} arms of the '
                         f'panel, e.g. {missing[:3]}; a partial roster would make the two gates\' '
                         'cells incomparable across arms')
    return _write(args, depths, label='targeted')


def _project_root(args) -> str:
    """The in-pod project root the lanes' plan paths resolve against.

    The root is host-local, so it comes from ``TRANSFER_PROJECT_ROOT`` unless the
    caller passes ``--project-root``; nothing is resolved at import time.
    """

    if args.project_root:
        return args.project_root
    raise SystemExit('--project-root is not set and TRANSFER_PROJECT_ROOT is not in the '
                     'environment, so the lanes have no root to resolve their plan paths '
                     'against; pass --project-root, or source h200/h200_env.sh')


def _runtime(args, project_root: str) -> str:
    """The env column the lanes declare, pinned to the staged ct interpreter."""

    if args.runtime is not None:
        return args.runtime
    return (f'TRANSFER_PYTHON={project_root}/{STAGED_PYTHON_SUFFIX} PYTHONDONTWRITEBYTECODE=1 '
            'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4')


def _write(args, depths, *, label: str) -> int:
    wanted = [arm.strip() for arm in args.arms.split(',') if arm.strip()] if args.arms else None
    if wanted:
        unknown = [arm for arm in wanted if arm not in ARM_COSTS]
        if unknown:
            raise SystemExit(f'--arms names arms outside the panel: {unknown}')
        depths = {arm: value for arm, value in depths.items() if arm in wanted}
    project_root = _project_root(args)
    runtime = _runtime(args, project_root)
    args.out.mkdir(parents=True, exist_ok=True)
    plan_out = args.out if args.plan_out is None else args.plan_out
    plan_out.mkdir(parents=True, exist_ok=True)
    gates = [gate.strip() for gate in args.gates.split(',') if gate.strip()] or sorted(COHORTS)
    unknown = [gate for gate in gates if gate not in COHORTS]
    if unknown:
        raise SystemExit(f'--gates names cohorts outside the declaration: {unknown}')
    written, blocked = [], {}
    for gate, cohort in [(gate, COHORTS[gate]) for gate in sorted(gates)]:
        accepts = extractor_accepts_depth(stage_path(REPO_ROOT, cohort.extractor))
        entry = estimate(gate, depths)
        cards = [int(index) for index in args.gpus.split(',') if index.strip()] or None
        rows = campaign_rows(gate, depths, project_root=project_root, runtime=runtime,
                             gpu=args.gpu, gpus=cards)
        # A lane is blocked only if it actually names the option. A selection that
        # lands on an admitted depth for every arm needs no change to the stage,
        # and calling that campaign blocked would be a refusal with no subject.
        needs_option = any(row['extra_depths'] for row in rows)
        ready = accepts or not needs_option
        blocked[gate] = not ready
        template = (READY_NOTE if accepts else NO_EXTRA_NOTE if not needs_option
                    else BLOCKED_NOTE)
        note = template.format(stage=cohort.extractor, option=REQUIRED_EXTRACTOR_OPTION)
        header = HEADER.format(gate=gate, cohort=cohort.label, rows=cohort.rows,
                               hours=entry['total_gpu_hours'], arms=entry['arms'],
                               gib=entry['total_retained_gib'], blocked=note)
        path = args.out / f'campaign_{args.name}_{gate}.tsv'
        path.write_text(render_campaign(rows, header), encoding='utf-8')
        written.append(str(path))
        write_json(plan_out / f'plan_{args.name}_{gate}.json',
                   dict(schema='d1_targeted_depth_plan_v1',
                        declaration_sha256=declaration_digest(), gate=gate,
                        queueable=ready, lanes_naming_the_option=needs_option,
                        extractor_accepts_option=accepts,
                        blocked_reason=None if ready else note, variant=label,
                        selection=None if args.selection is None else str(args.selection),
                        estimate=entry,
                        lanes=[{k: v for k, v in row.items()} for row in rows]))
        written.append(str(plan_out / f'plan_{args.name}_{gate}.json'))
    print(json.dumps(dict(out=str(args.out), variant=label, written=written,
                          queueable={gate: not value for gate, value in blocked.items()},
                          arms=len(depths),
                          depths_per_arm={arm: len(set(value)) if isinstance(value, list) else 1
                                          for arm, value in list(depths.items())[:3]}), indent=1))
    return 1 if any(blocked.values()) else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    first = sub.add_parser('estimate', help='GPU-hours and storage, no selection needed')
    first.add_argument('--out', type=Path, required=True)
    first.set_defaults(handler=_estimate)

    second = sub.add_parser('manifests', help='campaign lanes at the selected per-arm depths')
    second.add_argument('--selection', type=Path,
                        help="the readout reassessment's own per-arm class and depth")
    second.add_argument('--every-block', action='store_true',
                        help='hook every block of every arm instead, which needs no selection '
                             'and costs the same GPU time; see the estimate subcommand for the '
                             'storage it takes')
    second.add_argument('--out', type=Path, required=True,
                        help='where the campaign manifests are written. They reach a pod only '
                             'inside a code snapshot, so this is scripts/capability for a campaign '
                             'meant to be dispatched')
    second.add_argument('--plan-out', type=Path, default=None,
                        help='where the plan records are written, defaulting beside the '
                             'manifests. A dispatched campaign puts its manifests in the tree and '
                             'its plans under ignored logs/, which is where reports belong')
    second.add_argument('--gates', default='',
                        help='comma-separated cohorts to write manifests for, defaulting to all '
                             'four. A deferred cohort is left out here rather than written and '
                             'not queued, so a manifest on disk is always one somebody decided '
                             'to run')
    second.add_argument('--arms', default='',
                        help='comma-separated arms to write lanes for, defaulting to all 33. A '
                             'gate cell and its remainder are two runs with complementary lists, '
                             'so the gate arm is never extracted twice')
    second.add_argument('--name', default='depth3',
                        help='basename stem for the campaign and plan files, so a gate manifest '
                             'and the wave that follows it do not collide')
    second.add_argument('--project-root', default=os.environ.get('TRANSFER_PROJECT_ROOT'),
                        help='the in-pod project root the lanes\' plan paths are written '
                             'against; defaults to TRANSFER_PROJECT_ROOT, which '
                             'h200/h200_env.sh exports')
    second.add_argument('--runtime', default=None,
                        help='the env column every lane declares; defaults to '
                             f'TRANSFER_PYTHON=<project root>/{STAGED_PYTHON_SUFFIX} '
                             'PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4')
    second.add_argument('--gpu', type=int, default=0,
                        help='the single card index every lane names, for a one-cell gate manifest')
    second.add_argument('--gpus', default='',
                        help='comma-separated card indices to spread the lanes over instead, for '
                             'a wave; the card a cell lands on is execution metadata outside the '
                             'measurement identity')
    second.set_defaults(handler=_manifests)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except RecomputationRefused as refusal:
        raise SystemExit(f'refused: {refusal}') from refusal


if __name__ == '__main__':
    raise SystemExit(main())
