#!/usr/bin/env python
"""Check every gate's declared elements against the unified gate contract.

Read-only in the strong sense: it reads the contract, the per-gate element
declarations transcribed from each gate's record, and each record document's own
bytes. It writes one report and changes nothing. No gate is expressed on the
contract in its own code, and this check is what makes that safe: a quantity
named in a declaration must occur verbatim in the record it is transcribed from,
so a declaration that has drifted from its source is reported rather than
trusted.

``--gate`` names the gate the check is demonstrated on, and a divergence or a
transcription miss there fails the run. Every other declared gate is reported
with its own divergences and its own transcription result, so the report answers
which gates conform and where they do not without making one gate's divergence
block the others.

    python scripts/capability/reporting/check_gate_contract.py \
      --gate local_context \
      --out archive/logs/shared/recomputation_20260924/gate_contract.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.gate_contract import GATE_DECLARATIONS, conformance  # noqa: E402
from src.capability.core.io import write_json  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gate', default='local_context', choices=sorted(GATE_DECLARATIONS),
                        help='the gate the conformance check is demonstrated on')
    parser.add_argument('--root', type=Path, default=REPO_ROOT,
                        help='root the record documents are relative to')
    parser.add_argument('--out', type=Path, required=True, help='report path')
    args = parser.parse_args(argv)

    record = conformance(root=args.root)
    record['demonstrated_on'] = args.gate
    write_json(args.out, record)

    demonstrated = record['gates'][args.gate]
    transcription = demonstrated['transcription']
    summary = dict(out=str(args.out), contract_sha256=record['contract_sha256'],
                   demonstrated_on=args.gate,
                   demonstrated=dict(conforms=demonstrated['conforms'],
                                     divergences=demonstrated['divergences'],
                                     record=transcription['record'],
                                     quantities_checked=transcription['checked'],
                                     quantities_missing=transcription['missing']),
                   conforming=record['conforming'], diverging=record['diverging'],
                   transcription={gate: dict(checked=entry['transcription']['checked'],
                                             missing=len(entry['transcription']['missing']))
                                  for gate, entry in record['gates'].items()},
                   not_declared=record['not_declared']['gates'])
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    failed = (not demonstrated['conforms']) or bool(transcription['missing']) \
        or not transcription['readable']
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
