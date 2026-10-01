#!/usr/bin/env python
"""Check that a depth-resolved archive holds the admitted four blocks unchanged.

The hooking loop in :func:`~src.capability.readouts.readout_extraction.extract_batch` has no
test in this repository, because exercising it needs a loaded checkpoint. Its
order is preserved by construction -- the admitted pair is hooked first and its
captures are concatenated first -- but construction-preserving arguments are what
this programme has learned not to trust at face value, and if that order moved
every archive in a depth sweep would be silently wrong while looking well formed.

So this compares a newly written depth-resolved archive against the already
retained two-depth archive of the same arm, background by background, and
requires the admitted four block summaries to be **byte-identical**. It reads
only the two archive sets and the arm's block count; it loads no model, fits
nothing and resamples nothing.

A depth-resolved archive writes one array per hooked block, ``features_d<index>``
of shape ``(rows, 2, width)`` holding that block's mean and last summaries. The
admitted archive writes a single ``features`` of shape ``(rows, 4, width)``, whose
four slices are the middle block's two summaries then the final block's two. So
the admitted view of a depth archive is the two arrays named for the arm's
admitted block indices, concatenated in that order, and that is what must match.

    python scripts/capability/readouts/verify_depth_archive_identity.py \
      --arm gpt2-large --blocks 36 \
      --depth-dir <wave>/depth3-gpt2-large \
      --admitted-dir <retained wave>/full-gpt2-large \
      --out archive/logs/shared/recomputation_20260924/gate_identity_gpt2-large.json

Exit status is 0 only if every compared background is byte-identical. Anything
else is a defect in the shared hooking rule and not a tolerance to widen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import write_json  # noqa: E402
from src.capability.readouts.readout_extraction import hooked_block_indices  # noqa: E402


def _admitted_view(archive: Path, admitted: tuple[int, ...]) -> np.ndarray:
    """The admitted four-block view of a depth-resolved archive, in admitted order."""
    with np.load(archive, allow_pickle=False) as data:
        names = [f'features_d{index:03d}' for index in admitted]
        missing = [name for name in names if name not in data.files]
        if missing:
            raise SystemExit(f'{archive} holds no {missing}; it is not a depth-resolved archive '
                             'of this arm, or the admitted indices are wrong')
        return np.concatenate([np.asarray(data[name]) for name in names], axis=1)


def _admitted_archive(archive: Path) -> np.ndarray:
    with np.load(archive, allow_pickle=False) as data:
        if 'features' not in data.files:
            raise SystemExit(f'{archive} holds no features array; it is not an admitted archive')
        return np.asarray(data['features'])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--blocks', type=int, required=True,
                        help="the arm's transformer block count, which fixes its admitted pair")
    parser.add_argument('--depth-dir', type=Path, required=True,
                        help='the directory of the depth-resolved archives')
    parser.add_argument('--admitted-dir', type=Path, required=True,
                        help='the directory of the already retained two-depth archives')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)

    admitted_pair = hooked_block_indices(args.blocks)
    depth_archives = {path.name.removeprefix('full_'): path
                      for path in sorted(args.depth_dir.glob('full_*.npz'))}
    known = {path.name.removeprefix('full_'): path
             for path in sorted(args.admitted_dir.glob('full_*.npz'))}
    shared = sorted(set(depth_archives) & set(known))
    if not shared:
        raise SystemExit(f'no archive name is shared between {args.depth_dir} and '
                         f'{args.admitted_dir}; there is nothing to compare')

    rows, worst = [], 0.0
    identical = True
    for name in shared:
        view = _admitted_view(depth_archives[name], admitted_pair)
        reference = _admitted_archive(known[name])
        if view.shape != reference.shape:
            rows.append(dict(archive=name, identical=False,
                             reason=f'shape {view.shape} against {reference.shape}'))
            identical = False
            continue
        same_bytes = view.tobytes() == reference.tobytes()
        equal = bool(np.array_equal(view, reference))
        difference = float(np.max(np.abs(view.astype(np.float64)
                                         - reference.astype(np.float64)))) if not equal else 0.0
        worst = max(worst, difference)
        identical = identical and same_bytes and equal
        rows.append(dict(archive=name, identical=bool(same_bytes and equal),
                         byte_identical=bool(same_bytes), array_equal=equal,
                         max_absolute_difference=difference,
                         rows=int(view.shape[0]), blocks=int(view.shape[1]),
                         width=int(view.shape[2])))

    record = dict(schema='d1_depth_archive_identity_v1', arm=args.arm, blocks=args.blocks,
                  admitted_block_indices=list(admitted_pair),
                  depth_dir=str(args.depth_dir), admitted_dir=str(args.admitted_dir),
                  compared=len(shared), depth_archives=len(depth_archives),
                  admitted_archives=len(known),
                  identical=identical, max_absolute_difference=worst,
                  non_identical=[row['archive'] for row in rows if not row['identical']],
                  condition=('the admitted four block summaries of a depth-resolved archive must '
                             'be byte-identical to the retained two-depth archive of the same '
                             'background; a difference is a defect in the shared hooking rule'),
                  archives=rows)
    write_json(args.out, record)
    print(json.dumps({k: v for k, v in record.items() if k != 'archives'}, indent=1))
    return 0 if identical else 1


if __name__ == '__main__':
    raise SystemExit(main())
