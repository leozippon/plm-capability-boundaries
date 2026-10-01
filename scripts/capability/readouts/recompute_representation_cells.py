#!/usr/bin/env python
"""Refit published representation cells at a selected readout class and depth.

Three subcommands, in the order they are used.

``plan`` joins a cell roster with the readout reassessment's per-arm selection and
reports which cells can run and which cannot, before any array is read. A cell is
refused when its gate fitted no representation design, when the gate's retained
artefacts cannot serve the selection kind, when no adapter loads that gate's
cells, or when its published report is not where the roster says. Refusing at
plan time is the point: which gates need a fresh extraction is a property of the
retention declaration and should be learned from it rather than from a
half-finished campaign.

``replay`` is the pipeline-identity control and is the only subcommand that runs
before the reassessment lands. It refits one cell at the **admitted** selection
from the retained states and requires the published held-out predictions back
inside the tolerance the admitted admission receipt declares, on identical
support, folds, seeds, weighting and label budget. A cell that fails it raises.

``run`` is the recomputation. It refuses without the reassessment's own selection
declaration, because a recomputation against a default selection would report an
increment that no reassessment chose.

    python scripts/capability/readouts/recompute_representation_cells.py plan \
      --selection <reassessment selection>.json --cells <roster>.json \
      --out archive/logs/shared/recomputation_20260924/plan.json

    python scripts/capability/readouts/recompute_representation_cells.py replay \
      --cohort archive/logs/R2/readout_20260923/cohort.json \
      --manifest <extraction>/manifest_<arm>.json --arm <arm> --panel anchor201 \
      --split-seed 20260923 --published <admitted analysis>.json \
      --out archive/logs/shared/recomputation_20260924/replay_<arm>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import write_json  # noqa: E402
from src.capability.readouts.recomputation import (  # noqa: E402
    CellRequest, IMPLEMENTED_ADAPTERS, ReadoutSelection, RecomputationRefused, aggregate_cells,
    load_selection, plan, readout_cell, readout_rows)
from src.capability.readouts.recomputation_adapters import ANCHOR_ADAPTERS  # noqa: E402

#: Roster fields every cell record must carry. The support is not among them: it
#: is read from the published report, so a cell is fitted on the support its own
#: published increment was fitted on rather than on a separately declared one.
CELL_FIELDS = ('gate', 'arm', 'panel', 'split_seed', 'published', 'cohort')

#: What each gate's adapter needs beyond the common fields. An anchor-panel gate
#: takes every manifest whose intersection fixed its panel, because its adapter
#: refuses unless that set and every digest match the published cell's own
#: record, and the profile store its mutation-local block is rebuilt from.
GATE_FIELDS = {
    'readout_panel': ('manifest',),
    'crossed_controls': ('manifests', 'profile_store'),
    'local_context': ('manifests', 'profile_store'),
}

#: Fields a cell may carry and is not required to. ``depth_manifest`` is needed
#: only by a depth-axis selection and refused by the admitted axis, so requiring
#: it of every cell would refuse the admitted pipeline-identity control; the
#: adapter decides, because the adapter is what reads the selection.
OPTIONAL_CELL_FIELDS = ('depth_manifest', 'anchor_arm')


def _roster(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f'no cell roster at {path}')
    record = json.loads(path.read_text())
    cells = record.get('cells')
    if not isinstance(cells, list) or not cells:
        raise SystemExit(f'{path} lists no cells')
    for index, cell in enumerate(cells):
        missing = [field for field in CELL_FIELDS if field not in cell]
        if missing:
            raise SystemExit(f'{path} cell {index} is missing {missing}')
        gate = cell['gate']
        if gate not in GATE_FIELDS:
            raise SystemExit(f'{path} cell {index} names gate {gate!r}, which no adapter in this '
                             f'runner dispatches; it fits {sorted(GATE_FIELDS)}')
        missing = [field for field in GATE_FIELDS[gate] if field not in cell]
        if missing:
            raise SystemExit(f'{path} cell {index} is a {gate} cell and is missing {missing}')
    return cells


def _requests(cells: list[dict], selection: dict[tuple[str, str], ReadoutSelection]
              ) -> list[CellRequest]:
    """One request per roster cell, at that cell's own arm-and-panel selection.

    Keyed by arm **and** panel, because the published selection is: ProGen3-3B
    selects block 11 on the anchor panel and block 23 on the EC-conditioned one.
    A roster cell whose panel the reassessment did not select for is refused by
    name rather than served the arm's other panel, which is the substitution this
    whole driver exists to prevent.
    """
    requests = []
    for cell in cells:
        key = (cell['arm'], cell['panel'])
        if key not in selection:
            panels = sorted(panel for arm, panel in selection if arm == cell['arm'])
            raise SystemExit(
                f"the selection declares no class and depth for {cell['arm']} on panel "
                f"{cell['panel']}; a cell is not recomputed at a selection the reassessment did "
                'not make'
                + (f', and this arm is selected only on {panels}' if panels else
                   ', and this arm is not selected on any panel'))
        requests.append(CellRequest(gate=cell['gate'], arm=cell['arm'], panel=cell['panel'],
                                    split_seed=int(cell['split_seed']),
                                    selection=selection[key],
                                    published=Path(cell['published'])))
    return requests


def _published(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f'no published cell report at {path}')
    return json.loads(path.read_text())


def _refuse_on_verdict(record: dict, path) -> None:
    """Exit non-zero on a refused cell, after its record is on disk.

    The record is written first on purpose. A cell that fails the admitted
    pipeline-identity gate still carries its increments, its intervals and the
    comparison of every published quantity against the recomputed one, and those
    are exactly what a reader needs in order to see whether the departure reached
    anything that was published. Writing the evidence and then failing is not a
    softer refusal than failing without it.
    """
    if record.get('verdict') != 'refused':
        return
    moved = (record.get('published_quantities') or {})
    raise SystemExit(
        f"refused: {record['refusal']}\n  the record is at {path}, and it carries the comparison: "
        f"{moved.get('recovered_exactly')} of {moved.get('shared_metrics')} published summaries "
        f"recovered exactly, {len(moved.get('moved') or [])} moved, "
        f"{len(moved.get('resolved_sign_changes') or [])} resolved signs changed")


def _dispatch(request: CellRequest, cell: dict, args) -> dict:
    """One cell through its own gate's adapter, with a missing archive named.

    An archive the manifest lists and the filesystem does not hold is a retention
    finding -- the arrays were not staged where this host can read them -- rather
    than a fault in a loader, so it is reported as a refusal naming the file
    instead of as a traceback.
    """
    try:
        if request.gate == 'readout_panel':
            if request.selection.depth_axis != 'admitted':
                raise SystemExit(
                    f'refused: {request.arm}/{request.panel} is a readout-panel cell at the '
                    f'{request.selection.depth_axis} axis, and the depth sweep fitted its own '
                    'per-block cells on this cohort; this runner does not refit them. Read them '
                    'from the depth sweep\'s artefacts, or say they are absent if its retention '
                    'gap has not been closed')
            published = _published(request.published)
            support = published.get('support', {}).get('assay_ids')
            if support is None:
                raise SystemExit(f'{request.published} declares no support.assay_ids, so the '
                                 'recomputation has no published support to be fitted on')
            rows, blocks, counts, provenance = readout_rows(cell['cohort'], cell['manifest'],
                                                            request.arm, support)
            record = readout_cell(
                request, rows, published, admitted_blocks=blocks, variant_counts=counts,
                receipt=args.receipt, device=args.device or 'cpu',
                progress=(lambda name: print(f'  {name}', flush=True))
                if getattr(args, 'progress', False) else None)
            record['extraction_provenance'] = dict(
                arm=provenance['arm'], hidden_width=provenance['hidden_width'],
                n_assays=provenance['n_assays'], n_clusters=provenance['n_clusters'],
                n_variants=provenance['n_variants'],
                projection_blas_threads=provenance['projection_blas_threads'],
                extraction_block_indices=provenance['extraction_block_indices'],
                source_sha256=provenance['source_sha256'])
            return record
        adapter = ANCHOR_ADAPTERS.get(request.gate)
        if adapter is None:
            raise SystemExit(f'no adapter in this runner dispatches {request.gate!r}')
        # device stays None by default here, and each anchor adapter then fits on
        # the device its own published cell was fitted on: a ridge solve in a
        # different reduction order would confound a class change with a device
        # change, and the readout family's replay is the only one measured to
        # reproduce its published predictions inside the admitted tolerance on
        # the host.
        return adapter(request, cohort=cell['cohort'], manifests=cell['manifests'],
                       profile_store=cell['profile_store'], device=args.device,
                       receipt=args.receipt,
                       anchor_arm=cell.get('anchor_arm') or getattr(args, 'anchor_arm', None),
                       depth_manifest=(cell.get('depth_manifest')
                                       or getattr(args, 'depth_manifest', None)))
    except FileNotFoundError as missing:
        raise SystemExit(
            f'refused: {request.gate}/{request.arm} cannot be recomputed from this host because a '
            f'retained archive its manifest lists is not present: {missing.filename}. The '
            'admitted extraction archives live on the allocation\'s shared filesystem; the '
            'recomputation runs where they are.') from missing


def _replay(args) -> int:
    """The pipeline-identity control: one cell at the admitted selection."""
    request = CellRequest(gate=args.gate, arm=args.arm, panel=args.panel,
                          split_seed=args.split_seed,
                          selection=ReadoutSelection(arm=args.arm), published=args.published)
    if not args.manifest:
        raise SystemExit('a replay needs the extraction manifests its panel was fixed by')
    if args.gate == 'readout_panel' and len(args.manifest) != 1:
        raise SystemExit('a readout-panel replay takes exactly one --manifest')
    if 'profile_store' in GATE_FIELDS.get(args.gate, ()) and not args.profile_store:
        raise SystemExit(f'a {args.gate} replay requires --profile-store')
    cell = dict(cohort=args.cohort, manifest=args.manifest[0], manifests=list(args.manifest),
                profile_store=args.profile_store, anchor_arm=args.anchor_arm,
                depth_manifest=args.depth_manifest)
    record = _dispatch(request, cell, args)
    write_json(args.out, record)
    _refuse_on_verdict(record, args.out)
    print(json.dumps(dict(out=str(args.out), gate=args.gate, arm=args.arm, panel=args.panel,
                          split_seed=args.split_seed,
                          identity=record['identity'],
                          prediction_agreement=record['prediction_agreement'],
                          product_blocking=record['product_blocking'],
                          increments=[dict(metric=row['metric'],
                                           published=row.get('published'),
                                           recomputed=row.get('recomputed'),
                                           point_change=row.get('point_change'))
                                      for row in record['increments']]), indent=1))
    return 0


def _key(cell) -> tuple:
    if isinstance(cell, dict):
        return cell['gate'], cell['arm'], cell['panel'], int(cell['split_seed'])
    return cell.gate, cell.arm, cell.panel, cell.split_seed


def _run(args) -> int:
    selection = load_selection(args.selection)
    cells = _roster(args.cells)
    requests = _requests(cells, selection)
    schedule = plan(requests)
    if not schedule['runnable']:
        raise SystemExit('the plan admits no runnable cell:\n  '
                         + '\n  '.join(f"{cell['gate']}/{cell['arm']}/{cell['panel']}: "
                                       f"{cell['refusal']}" for cell in schedule['refused']))
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'plan.json', schedule)
    by_key = {_key(cell): cell for cell in cells}
    by_request = {_key(request): request for request in requests}
    written, records, failures = [], [], []
    for entry in schedule['runnable']:
        request = by_request[_key(entry)]
        cell = by_key[_key(entry)]
        # A batch must not stop at a refusal it expects. Each cell fails fast on
        # its own terms and is recorded as failed; the batch runs to the end and
        # the non-zero exit comes once, after every cell has been measured.
        # Halting here would leave the remaining cells unmeasured for no reason.
        try:
            record = _dispatch(request, cell, args)
        except (RecomputationRefused, SystemExit, ValueError, KeyError) as failure:
            failures.append(dict(gate=request.gate, arm=request.arm, panel=request.panel,
                                 split_seed=request.split_seed, error=str(failure)))
            print(json.dumps(dict(cell=f'{request.gate}/{request.arm}/{request.panel}/'
                                       f'{request.split_seed}', status='failed',
                                  error=str(failure)[:400])), flush=True)
            continue
        path = args.out / f'{request.gate}_{request.arm}_{request.panel}_{request.split_seed}.json'
        write_json(path, record)
        written.append(str(path))
        records.append(record)
        quantities = record.get('published_quantities') or {}
        print(json.dumps(dict(cell=f'{request.gate}/{request.arm}/{request.panel}/'
                                   f'{request.split_seed}', verdict=record.get('verdict'),
                              resolved_sign_changes=len(quantities.get('resolved_sign_changes')
                                                        or []),
                              max_point_change=quantities.get('max_absolute_point_change'))),
              flush=True)
    summary = aggregate_cells(records, failures=failures, planned_refusals=schedule['refused'])
    summary['written'] = written
    write_json(args.out / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'gates'}, indent=1))
    print(json.dumps({name: {k: v for k, v in gate.items() if k not in ('cells',)}
                      for name, gate in summary['gates'].items()}, indent=1))
    if failures or summary['any_cell_refused']:
        raise SystemExit(
            f"{len(failures)} cells failed and {sum(g['cells_whose_pipeline_gate_refused'] for g in summary['gates'].values())} "
            f'were refused by the pipeline-identity gate; every completed cell is written and the '
            f"aggregation is at {args.out / 'summary.json'}")
    return 0


def _aggregate(args) -> int:
    """Re-aggregate a directory of cell records without refitting anything."""
    records = [json.loads(path.read_text()) for path in sorted(args.cells.glob('*.json'))
               if path.name != 'summary.json' and path.name != 'plan.json']
    records = [r for r in records if r.get('schema') and 'cell' in str(r.get('schema'))]
    if not records:
        raise SystemExit(f'no cell records under {args.cells}')
    summary = aggregate_cells(records)
    write_json(args.out, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'gates'}, indent=1))
    return 0


def _plan(args) -> int:
    selection = load_selection(args.selection)
    schedule = plan(_requests(_roster(args.cells), selection))
    write_json(args.out, schedule)
    print(json.dumps(dict(out=str(args.out), cells=schedule['cells'],
                          runnable=len(schedule['runnable']), refused=len(schedule['refused']),
                          adapters=sorted(IMPLEMENTED_ADAPTERS),
                          refusals=[dict(gate=c['gate'], arm=c['arm'], panel=c['panel'],
                                         refusal=c['refusal']) for c in schedule['refused']]),
                     indent=1))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    for name, handler in (('plan', _plan), ('run', _run)):
        stage = sub.add_parser(name, help=handler.__doc__)
        stage.add_argument('--selection', type=Path, required=True,
                           help="the readout reassessment's own per-arm class and depth")
        stage.add_argument('--cells', type=Path, required=True, help='the cell roster')
        stage.add_argument('--out', type=Path, required=True,
                           help='report path for plan; output directory for run')
        stage.add_argument('--receipt', type=Path,
                           default=REPO_ROOT / 'results/R2/readout_20260923/'
                                               'final_admission.json',
                           help='the admitted admission receipt the tolerance is read from')
        stage.add_argument('--depth-manifest', type=Path, default=None,
                           help='a depth-resolved extraction manifest applied to every cell that '
                                'does not name its own')
        stage.add_argument('--device', default=None,
                           help='override the device each cell is fitted on; the default is the '
                                'device its own published cell was fitted on')
        stage.set_defaults(handler=handler)

    regroup = sub.add_parser('aggregate', help=_aggregate.__doc__)
    regroup.add_argument('--cells', required=True, type=Path,
                         help='a directory of cell records written by run')
    regroup.add_argument('--out', required=True, type=Path)
    regroup.set_defaults(handler=_aggregate)

    replay = sub.add_parser('replay', help='the pipeline-identity control at the admitted selection')
    replay.add_argument('--gate', default='readout_panel')
    replay.add_argument('--arm', required=True)
    replay.add_argument('--panel', required=True)
    replay.add_argument('--split-seed', type=int, required=True)
    replay.add_argument('--cohort', type=Path, required=True)
    replay.add_argument('--manifest', type=Path, action='append', required=True,
                        help='repeatable: every manifest whose intersection fixed the panel')
    replay.add_argument('--profile-store', type=Path,
                        help='the retained column-frequency store the anchor-panel gates rebuild '
                             'their mutation-local profile block from')
    replay.add_argument('--depth-manifest', type=Path,
                        help="the depth-resolved extraction manifest of this cell's own cohort, "
                             'required for a depth-axis selection and refused for the admitted '
                             'axis')
    replay.add_argument('--anchor-arm',
                        help='the arm whose manifest fixes the required anchor panel; needed for '
                             'a published cell fitted before the stage recorded it')
    replay.add_argument('--device', default=None)
    replay.add_argument('--published', type=Path, required=True)
    replay.add_argument('--out', type=Path, required=True)
    replay.add_argument('--progress', action='store_true')
    replay.add_argument('--receipt', type=Path,
                        default=REPO_ROOT / 'results/R2/readout_20260923/'
                                            'final_admission.json')
    replay.set_defaults(handler=_replay)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except RecomputationRefused as refusal:
        raise SystemExit(f'refused: {refusal}') from refusal


if __name__ == '__main__':
    raise SystemExit(main())
