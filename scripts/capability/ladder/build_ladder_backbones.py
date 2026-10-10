#!/usr/bin/env python3
"""Freeze the ladder's backbone set from folds this project has already measured.

The ladder needs natural protein backbones that are foldable with high confidence
at full length, because a divergence between likelihood and structure can only be
read against a parent the structure instrument is sure about. Nothing is folded
here: the generation-evaluation experiment already folded 1,822 **whole**
Swiss-Prot entries with ESMFold2 -- not length-matched fragments, which is what
makes them usable -- and this stage selects from that measured pool and reuses
its parent folds by literal path.

Admission is :func:`src.capability.ladder.design.backbone_rejection`, declared
before anything is read, and every candidate's refusal reason is written out so
the set can be replayed from the pool. Two further conditions are applied here
because they are relations between admitted members rather than properties of
one: no two backbones may be near-duplicates of each other under this project's
own shingle relation, and no two may carry the same EC number, so the
EC-conditioned arm is not sixteen repetitions of one class request.

Every backbone carries a single EC number by construction, because the
EC-conditioned arm has to be expressible on all of them. The set is therefore
EC-annotated enzymes and nothing measured on it generalises to non-enzymes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.ladder import design  # noqa: E402

COMPLETION = "ladder_backbones.json"
RECORDS = "ladder_backbones.jsonl"
SCHEMA_VERSION = "ladder_backbones_v1"


def read_ec_labels(path: Path) -> dict[str, list[str]]:
    """Accession to EC numbers, from the EC-labelled Swiss-Prot FASTA headers."""

    labels: dict[str, list[str]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            parts = line[1:].strip().split("|")
            if len(parts) != 2:
                raise ValueError(f"unexpected EC header: {line.strip()!r}")
            labels.setdefault(parts[0], []).append(parts[1])
    if not labels:
        raise ValueError(f"{path} carries no EC-labelled header")
    return labels


def read_fold_index(paths: list[Path]) -> list[dict[str, Any]]:
    """Every row of one or more ``run_structure_evidence.py`` index shards."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = str(row["id"])
            if identifier in seen:
                raise SystemExit(f"duplicate row id {identifier} across fold indices")
            seen.add(identifier)
            rows.append(row)
    if not rows:
        raise SystemExit("the fold indices carry no row")
    return rows


def _order_key(row: dict[str, Any]) -> str:
    """A deterministic pseudo-random order derived from the campaign seed alone."""

    material = f"{design.CAMPAIGN}|{design.CAMPAIGN_SEED}|{row['id']}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _containment(left: str, right: str) -> float:
    from src.capability.generation.near_duplicates import shingles

    a = shingles(left, unit="residues")
    b = shingles(right, unit="residues")
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def select(
    rows: list[dict[str, Any]], ec_labels: dict[str, list[str]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from src.capability.generation.near_duplicates import NEAR_DUPLICATE_CONTAINMENT

    attrition: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    for row in rows:
        if "natural" not in (row.get("roles") or []):
            attrition["not_a_natural_record"] += 1
            continue
        reason = design.backbone_rejection(row, ec_labels=ec_labels)
        if reason is not None:
            attrition[reason] += 1
            continue
        eligible.append(row)
    eligible.sort(key=_order_key)

    admitted: list[dict[str, Any]] = []
    per_stratum: Counter[int] = Counter()
    used_ec: set[str] = set()
    for row in eligible:
        stratum = design.stratum_of(int(row["length"]))
        if per_stratum[stratum] >= design.BACKBONES_PER_STRATUM:
            attrition["stratum_full"] += 1
            continue
        label = ec_labels[str(row["accession"])][0]
        if label in used_ec:
            attrition["duplicate_ec_number"] += 1
            continue
        if any(
            _containment(row["sequence"], member["sequence"]) >= NEAR_DUPLICATE_CONTAINMENT
            for member in admitted
        ):
            attrition["near_duplicate_of_admitted"] += 1
            continue
        structure = row["structure"]
        admitted.append(
            {
                "backbone_id": f"lb_{row['accession']}",
                "accession": str(row["accession"]),
                "source_row_id": str(row["id"]),
                "sequence": str(row["sequence"]),
                "length": int(row["length"]),
                "ec_label": label,
                "stratum": stratum,
                "parent_sequence_sha256": str(structure["sequence_sha256"]),
                "parent_object_directory": str(structure["object_directory"]),
                "parent_mean_ca_plddt": float(structure["mean_ca_plddt"]),
                "parent_fraction_ca_plddt_ge70": float(structure["fraction_ca_plddt_ge70"]),
                "parent_ptm": float(structure["ptm"]),
                "parent_mean_pae_angstrom": float(structure["mean_pae_angstrom"]),
                "window_spans": {
                    f"k{extent}": list(design.window_span(int(row["length"]), extent))
                    for extent in design.WINDOW_EXTENTS
                },
                **design.sequence_descriptors(str(row["sequence"])),
            }
        )
        per_stratum[stratum] += 1
        used_ec.add(label)
        if len(admitted) == design.N_BACKBONES:
            break
    admitted.sort(key=lambda member: (member["stratum"], member["length"], member["backbone_id"]))
    return admitted, {
        "n_rows_read": len(rows),
        "n_eligible": len(eligible),
        "n_admitted": len(admitted),
        "attrition": dict(sorted(attrition.items())),
        "per_stratum_admitted": {str(key): value for key, value in sorted(per_stratum.items())},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    design.require_fresh_out(args.out, COMPLETION)
    rows = read_fold_index(args.fold_index)
    ec_labels = read_ec_labels(args.ec_fasta)
    admitted, census = select(rows, ec_labels)
    if len(admitted) != design.N_BACKBONES:
        raise SystemExit(
            f"the measured pool yields {len(admitted)} admissible backbones and the "
            f"ladder declares {design.N_BACKBONES}. Refusing to run a ladder on a "
            f"smaller backbone set than it declares. Census: {json.dumps(census)}"
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records_digest = design.write_jsonl(out / RECORDS, admitted)
    lengths = [member["length"] for member in admitted]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "declaration": design.declaration(),
        "census": census,
        "records": RECORDS,
        "records_sha256": records_digest,
        "n_backbones": len(admitted),
        "length_summary": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "per_stratum": {
                str(index): sorted(
                    member["length"] for member in admitted if member["stratum"] == index
                )
                for index in range(design.BACKBONE_STRATA)
            },
        },
        "parent_confidence_summary": {
            "min_mean_ca_plddt": min(member["parent_mean_ca_plddt"] for member in admitted),
            "min_ptm": min(member["parent_ptm"] for member in admitted),
        },
        "ec_numbers": sorted(member["ec_label"] for member in admitted),
        "sources": {
            "fold_index": [str(path) for path in args.fold_index],
            "fold_index_sha256": {
                str(path): sha256_file(path) for path in args.fold_index
            },
            "ec_fasta": str(args.ec_fasta),
            "parent_folds_are_reused": (
                "the parent fold of every backbone is the object this index already "
                "carries. Nothing is refolded and no second instrument is introduced"
            ),
        },
    }
    write_json(out / COMPLETION, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="accepted and unused; this stage is CPU-only")
    parser.add_argument(
        "--fold-index",
        type=Path,
        nargs="+",
        required=True,
        help="index-*.jsonl shards of a run_structure_evidence.py tree holding whole natural records",
    )
    parser.add_argument("--ec-fasta", type=Path, required=True, help="EC-labelled Swiss-Prot FASTA")
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"n_backbones": payload["n_backbones"], "census": payload["census"]}))


if __name__ == "__main__":
    main()
