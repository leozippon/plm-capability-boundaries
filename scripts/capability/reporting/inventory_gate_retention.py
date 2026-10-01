#!/usr/bin/env python
"""Report what each gate retained and whether its representation cells can be refitted.

Reads the frozen retention declaration in :mod:`src.capability.core.gate_retention`,
probes each declared artefact under the roots it is given, and writes one record
per gate per selection kind. It reads nothing else: no cohort, no state array, no
fitted report, and no model.

The probe distinguishes three outcomes for a location, and the distinction is the
point of running it from two hosts rather than one. ``present`` is found under the
root given for its store. ``absent`` is declared by a gate record and not found,
which is a finding. ``unreachable_from_this_host`` is a location in a store this
invocation was given no root for -- the substantive arrays live on the remote
allocation's shared filesystem -- and is not a finding at all.

    python scripts/capability/reporting/inventory_gate_retention.py \
      --out archive/logs/shared/recomputation_20260924/gate_retention.json

    # in a pod, where the cluster store is reachable
    python scripts/capability/reporting/inventory_gate_retention.py \
      --cluster-root /path/to/shared/root \
      --out archive/logs/shared/recomputation_20260924/gate_retention_inpod.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.gate_retention import SELECTION_KINDS, inventory  # noqa: E402
from src.capability.core.io import write_json  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repository-root', type=Path, default=REPO_ROOT,
                        help='root the repository-store locations are relative to')
    parser.add_argument('--cluster-root', type=Path, default=None,
                        help='root the cluster-store locations are relative to; omit it off the '
                             'allocation, where those locations are unreachable rather than absent')
    parser.add_argument('--out', type=Path, required=True, help='report path')
    args = parser.parse_args(argv)

    record = inventory({'repository': args.repository_root, 'cluster': args.cluster_root})
    write_json(args.out, record)

    absent = [(gate['gate'], artifact['location'])
              for gate in record['gates'] for artifact in gate['artifacts']
              if artifact['status'] == 'absent']
    summary = dict(out=str(args.out), declaration_sha256=record['declaration_sha256'],
                   entries=record['counts']['entries'],
                   with_representation_cells=record['counts']['with_representation_cells'],
                   representation_cells=record['counts']['representation_cells'],
                   requires_re_extraction=record['requires_re_extraction'],
                   role_gaps={gate['gate']: gate['role_gaps'] for gate in record['gates']
                              if gate['role_gaps']},
                   absent_declared_artifacts=[f'{gate}: {location}' for gate, location in absent],
                   selection_kinds=list(SELECTION_KINDS))
    print(json.dumps(summary, indent=1))
    # A declared artefact that is absent under a root this invocation was given is
    # a finding about retention and is reported with a non-zero exit, so a
    # scheduled probe cannot pass silently over one.
    return 1 if absent else 0


if __name__ == '__main__':
    raise SystemExit(main())
