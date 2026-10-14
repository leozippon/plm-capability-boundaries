#!/usr/bin/env python3
"""E20: one cell of a model-independent reference arm, ``P`` (Potts joint) or ``H`` (held-out natural).

Neither arm loads a language model. The cell writes the same files the guided-decoding cell does --
``sequences.jsonl`` in the ``e20_generated_sequence_v1`` contract and a ``guided_decoding_pool.json``
completion record (the file name ``merge_e20_cells.py`` discovers cells by) -- so the fold, merge and
measure stages take it unchanged. The model labels are ``potts-joint`` and ``natural``; the
language-model fields and the strength pair are null.

Protocol: the ``coupling_guided_generation_v1`` protocol of the guided cells (``domains``,
``seed_streams``) plus a ``reference_arms`` object, whose values are recorded in the completion record::

    "reference_arms": {"potts_joint": {"chains": 100, "sweeps": 200, "mixing_tolerance": 0.02},
                       "natural": {"max_rows": 100, "min_admissible": 25}}

``--stream-index`` picks the cell's seed; ``merge_e20_cells.py`` refuses two cells sharing a stream seed,
so each (arm, domain) cell needs its own stream, distinct from the guided cells'.

``P`` needs ``--family-reference-dir``. A domain whose chains fail the mixing check fails the cell: a
failure record with the measured distance, no sequences, a non-zero exit. ``H`` needs ``--msa-dir`` and
``--qualification`` (``family_references_jointh_*/qualification.json``): the holdout is recomputed with
the fit script's own ``split_rows`` and must reproduce the recorded split parameters and held-out row
count, else the cell fails. Fewer than ``min_admissible`` gapless held-out rows is recorded, not an error.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.capability.interactions import fit_family_references as ffr  # noqa: E402
from src.capability.context.family_potts import MixingError, load_family_model  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation import fold_recapitulation as fr  # noqa: E402
from src.capability.generation import reference_arms as ra  # noqa: E402

PROTOCOL_SCHEMA = "coupling_guided_generation_v1"
COMPLETION_SCHEMA = "e20_reference_arm_cell_v1"
FAILURE_SCHEMA = "e20_reference_arm_failure_v1"
COMPLETION_FILENAME = "guided_decoding_pool.json"  # merge_e20_cells.py discovers cells by this name
FAILURE_FILENAME = "failure_reference_arm_cell.json"
SEQUENCES_FILENAME = "sequences.jsonl"
SPLIT_KEYS = {"cluster_identity": ffr.CLUSTER_IDENTITY, "min_overlap_fraction": ffr.MIN_OVERLAP_FRACTION,
              "held_out_fraction": ffr.HELD_OUT_FRACTION,
              "max_held_out_cluster_fraction": ffr.MAX_HELD_OUT_CLUSTER_FRACTION, "seed": ffr.SPLIT_SEED}


def _int(value: Any, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}, got {value!r}")
    return value


def bind_protocol(protocol: Any, *, arm: str, domain: str, stream_index: int) -> dict[str, Any]:
    """Validate the protocol values this cell uses and bind them to the cell; nothing is defaulted."""

    if not isinstance(protocol, Mapping) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"protocol must be a {PROTOCOL_SCHEMA} object")
    entry = (protocol.get("domains") or {}).get(domain)
    if not isinstance(entry, Mapping) or not isinstance(entry.get("wt_sha256"), str):
        raise ValueError(f"domain {domain!r} is not declared by the protocol")
    streams = protocol.get("seed_streams")
    if not isinstance(streams, list) or len(set(streams)) != len(streams) or not 0 <= stream_index < len(streams):
        raise ValueError("protocol.seed_streams must be distinct and contain --stream-index")
    declared = (protocol.get("reference_arms") or {}).get({"P": "potts_joint", "H": "natural"}[arm])
    if not isinstance(declared, Mapping):
        raise ValueError(f"protocol.reference_arms declares no {arm} values")
    keys = {"P": ("chains", "sweeps", "mixing_tolerance"), "H": ("max_rows", "min_admissible")}[arm]
    if set(declared) != set(keys):
        raise ValueError(f"protocol.reference_arms for {arm} must hold exactly {list(keys)}")
    values = {k: _int(declared[k], k, 1) for k in keys if k != "mixing_tolerance"}
    if arm == "P":
        _int(declared["sweeps"], "sweeps", 2)
        tolerance = declared["mixing_tolerance"]
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not 0 < tolerance < 1:
            raise ValueError(f"mixing_tolerance must be in (0, 1), got {tolerance!r}")
        values["mixing_tolerance"] = float(tolerance)
    return {"arm": arm, "domain": domain, "length": _int(entry.get("L"), "domains.L", 1),
            "wt_sha256": entry["wt_sha256"], "stream_index": stream_index,
            "stream_seed": _int(streams[stream_index], "seed_streams[]", 0), "values": values}


def _binds_wild_type(domain: str, wildtype: str, bound: Mapping[str, Any]) -> None:
    if len(wildtype) != bound["length"] or sha256(wildtype.encode()).hexdigest() != bound["wt_sha256"]:
        raise ValueError(f"{domain}: the wild type does not match the protocol's L and wt_sha256")


def run_potts(bound: Mapping[str, Any], family_reference_dir: Path) -> tuple[list, dict]:
    model = load_family_model(bound["domain"], family_reference_dir)  # refuses an unqualified reference
    _binds_wild_type(bound["domain"], model.wildtype, bound)
    return ra.potts_joint_cell(model, domain=bound["domain"], seed=bound["stream_seed"], **bound["values"])


def run_natural(bound: Mapping[str, Any], msa_dir: Path, qualification: Path) -> tuple[list, dict]:
    domain, length = bound["domain"], bound["length"]
    record = json.loads(fr.require_asset(qualification, "--qualification").read_text())
    split = record["split"]
    drift = {k: (split.get(k), v) for k, v in SPLIT_KEYS.items() if split.get(k) != v}
    if drift:
        raise ValueError(f"{qualification} records split parameters {drift} that differ from the fit script's")
    fr.load_msa(msa_dir, domain, length)  # every row must be `length` columns, else refused
    msa = fr.read_fasta(Path(msa_dir) / f"{domain}.pdb.fasta")
    if msa[0][0] != "focus":
        raise ValueError(f"{domain}: the first MSA row is not the focus")
    _binds_wild_type(domain, msa[0][1], bound)
    held = ffr.split_rows(domain, ffr.encode_rows([s for _, s in msa]), ffr.CLUSTER_IDENTITY)["held"]
    recorded = record["domains"][f"{domain}.pdb"]["held_out_rows"]
    if int(held.sum()) != recorded:
        raise ValueError(f"{domain}: the recomputed holdout has {int(held.sum())} rows, qualification records {recorded}")
    rows, manifest = ra.natural_cell(msa, held, domain=domain, length=length, seed=bound["stream_seed"],
                                     **bound["values"])
    manifest["qualification_held_out_rows"] = recorded
    return rows, manifest


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for r in rows).encode()


def run_job(args: argparse.Namespace) -> dict[str, Any]:
    protocol_bytes = Path(args.protocol).read_bytes()
    bound = bind_protocol(json.loads(protocol_bytes), arm=args.arm, domain=args.domain, stream_index=args.stream_index)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"refusing to reuse the non-empty output directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    if args.arm == "P":
        rows, sampling = run_potts(bound, fr.require_asset(args.family_reference_dir, "--family-reference-dir", directory=True))
    else:
        rows, sampling = run_natural(bound, fr.require_asset(args.msa_dir, "--msa-dir", directory=True), args.qualification)
    wildtype = {args.domain: "A" * bound["length"]}  # lengths only; residues are the sequences' own
    if rows:
        fr.validate_records(rows, wildtype)
    payload = _jsonl(rows)
    (out / SEQUENCES_FILENAME).write_bytes(payload)
    digest = sha256(payload).hexdigest()
    (out / f"{SEQUENCES_FILENAME}.sha256").write_text(f"{digest}  {SEQUENCES_FILENAME}\n")
    completion = {
        "schema": COMPLETION_SCHEMA, "status": "complete",
        "cell": {"model_arm": fr.MODEL_FREE_ARMS[args.arm], "domain": args.domain, "guidance_arm": None,
                 "arm_code": args.arm, "lambda": 0.0, "lambda_fields": None, "lambda_coupling": None,
                 "stream_index": bound["stream_index"], "stream_seed": bound["stream_seed"],
                 "sequences_per_cell": len(rows), "attempt_budget": ra.ATTEMPTS},
        "reference_arm": {"protocol_values": bound["values"], "sampling": sampling},
        "protocol": {"path": str(args.protocol), "sha256": sha256(protocol_bytes).hexdigest()},
        "sampling": None, "native_contract": None, "component_scale": None,
        "artifacts": {"sequences": {"file": SEQUENCES_FILENAME, "sha256": digest, "records": len(rows),
                                    "schema": fr.SCHEMA_VERSION, "validated_against": "fold_recapitulation.validate_records"}},
        "source": {"script": "scripts/capability/generation/generate_reference_arms.py",
                   "code_sha256": {p: sha256_file(REPO_ROOT / p) for p in (
                       "scripts/capability/generation/generate_reference_arms.py",
                       "src/capability/generation/reference_arms.py", "src/capability/context/family_potts.py",
                       "scripts/capability/interactions/fit_family_references.py")}},
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / COMPLETION_FILENAME, completion)
    return completion


def build_parser() -> argparse.ArgumentParser:
    """Flat parser; the campaign queue injects ``--device`` and ``--out`` ahead of the cell arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="accepted and unused; CPU-only")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", required=True, choices=sorted(fr.MODEL_FREE_ARMS))
    parser.add_argument("--domain", required=True)
    parser.add_argument("--stream-index", type=int, required=True)
    parser.add_argument("--family-reference-dir", type=Path, help="arm P: staged family references")
    parser.add_argument("--msa-dir", type=Path, help="arm H: directory of <DOMAIN>.pdb.fasta")
    parser.add_argument("--qualification", type=Path, help="arm H: qualification.json recording the holdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        completion = run_job(args)
    except Exception as error:
        if Path(args.out).is_dir():
            failure = {"schema": FAILURE_SCHEMA, "status": "failed", "arm": args.arm, "domain": args.domain,
                       "stream_index": args.stream_index, "error_type": type(error).__name__, "error": str(error),
                       "updated_utc": datetime.now(timezone.utc).isoformat()}
            if isinstance(error, MixingError):
                failure["mixing"] = {"max_tv": error.mixing_max_tv, "tolerance": error.mixing_tolerance,
                                     "sweeps": error.sweeps, "check_sweeps": error.check_sweeps}
            write_json(Path(args.out) / FAILURE_FILENAME, failure)
        raise
    print(json.dumps({"cell": completion["cell"], "reference_arm": completion["reference_arm"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
