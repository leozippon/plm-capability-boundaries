#!/usr/bin/env python3
"""Measure E20 fold recapitulation, controls and arm contrasts. CPU only.

Inputs: generator records (e20_generated_sequence_v1), the evidence tree that
``run_structure_evidence.py`` produced from their cohort, and the frozen
wild-type reference manifest. ``--role`` selects development domains (choose
lambda) or evaluation domains (read once, everything frozen); the output records
the role it covers. Writes per-sequence measurements, per-cell controls and the
contrast report, and exits non-zero after writing if an evaluation-role interval
was refused below the eight-domain floor.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation import fold_recapitulation as fr  # noqa: E402
from src.capability.generation import fold_recapitulation_inference as fri  # noqa: E402

COMPLETION = "fold_recapitulation.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="accepted and unused; CPU-only")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True, help="wild_type_references.json")
    parser.add_argument("--wild-type-fasta", type=Path, default=REPO_ROOT / fr.DEFAULT_WT_FASTA,
                        help="query_12.faa; data/ is not in the pod snapshot, pass the staged path there")
    parser.add_argument("--msa-dir", type=Path, default=REPO_ROOT / fr.DEFAULT_MSA_DIR,
                        help="directory of <DOMAIN>.pdb.fasta MSAs (novelty and family composition)")
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / fr.DEFAULT_PROTOCOL,
                        help="diagnostic_protocol.json holding the development/evaluation roles")
    parser.add_argument("--component-scale", type=Path,
                        help="component_scale.json written by merge_e20_cells.py; required for --role development")
    parser.add_argument("--role", choices=("development", "evaluation"), required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()

    for path, option in ((args.records, "--records"), (args.references, "--references")):
        fr.require_asset(path, option)
    fr.require_asset(args.structure, "--structure", directory=True)
    fr.require_asset(args.msa_dir, "--msa-dir", directory=True)
    wild = fr.load_wild_types(args.wild_type_fasta)
    roles = fr.load_roles(args.protocol)
    records = [r for r in fr.load_records(args.records, wild) if roles[r["domain"]] == args.role]
    if args.role == "evaluation":
        fr.require_frozen_strength(records)
    if not records:
        raise SystemExit(f"no records in role {args.role}")
    msas = {d: fr.load_msa(args.msa_dir, d, len(wild[d])) for d in {r["domain"] for r in records}}
    measured = fr.measure_records(records, wild_types=wild, msas=msas, tree=args.structure,
                                  reference_manifest=args.references)
    if args.role == "evaluation":
        report = fri.analyse(measured, roles=roles, role=args.role, seed=args.seed)
    else:
        if args.component_scale is None:
            raise SystemExit("--role development needs --component-scale (from merge_e20_cells.py)")
        scale = json.loads(fr.require_asset(args.component_scale, "--component-scale").read_text())
        report = fri.analyse_development(measured, roles=roles, component_scale=scale)
        report["refusals"] = []
    args.out.mkdir(parents=True, exist_ok=True)
    per_sequence = args.out / "per_sequence.jsonl"
    per_sequence.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in measured))
    write_json(args.out / "cell_controls.json", fr.cell_controls(measured))
    write_json(args.out / "contrasts.json", report)
    write_json(args.out / COMPLETION, {
        "schema_version": "e20_fold_recapitulation_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "role": args.role, "domains": report["domains"], "n_sequences": len(measured),
        "refusals": report["refusals"],
        "inputs": {"records": sha256_file(args.records), "references": sha256_file(args.references)},
        "outputs": {n: sha256_file(args.out / n) for n in ("per_sequence.jsonl", "cell_controls.json", "contrasts.json")},
        "code_sha256": {Path(m.__file__).name: sha256_file(Path(m.__file__)) for m in (fr, fri)} | {Path(__file__).name: sha256_file(Path(__file__))},
        "interpretation": "Computational endpoints on predicted structures; no wet-lab validation.",
    })
    if args.role == "evaluation" and report["refusals"]:
        raise SystemExit(f"evaluation intervals refused: {report['refusals'][:3]}")


if __name__ == "__main__":
    main()
