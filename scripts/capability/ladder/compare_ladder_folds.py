#!/usr/bin/env python3
"""Join every ladder variant to its fold and compare it with its parent's fold.

Reads the fold objects two trees hold -- the ones this campaign produced for the
variants, and the staged tree the backbones' parent folds were already measured
in -- and derives, per variant, the independent structural evaluation the ladder
correlates against likelihood.

For a window rung the comparison is exact: length is preserved, so residue *i* of
the variant is residue *i* of the parent and the TM-score and the local measures
are computed at that known correspondence with no alignment inferred. For the
full-generation anchor there is no parent fold and no correspondence, so the
parent columns are **absent** from the row rather than filled with a number that
would not mean the same thing.

The pairwise confidence field is kept pairwise. ``window_flank_mean_pae_angstrom``
is the mean predicted aligned error of the window-against-flank block, which is
the part of the matrix that says whether the predictor knows where the new window
sits relative to the rest of the chain; reducing the matrix to one global mean
would throw exactly that away.

Nothing here is stability and nothing here is function.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.ladder import design, fold_geometry  # noqa: E402

COMPLETION = "ladder_structure.json"
RECORDS = "ladder_structure.jsonl"
SCHEMA_VERSION = "ladder_structure_v1"


def load_object(roots: list[Path], digest: str) -> dict[str, Any] | None:
    """One fold object, looked up by sequence digest across several trees.

    Returns ``None`` when no tree holds it -- a variant whose fold has not been
    computed yet is a pending row and not an error -- but raises when a tree
    claims the fold succeeded and its files are missing, because that is a tree
    that did not produce the result it records.
    """

    for root in roots:
        directory = Path(root) / "objects" / digest
        result_path = directory / "result.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "ok":
            return {"status": result.get("status"), "result": result, "directory": str(directory)}
        pdb = directory / "prediction.pdb"
        arrays = directory / "prediction.npz"
        if not (pdb.is_file() and arrays.is_file()):
            raise SystemExit(
                f"{directory} records status ok and is missing its prediction files; this "
                "tree did not produce the result it claims"
            )
        with np.load(arrays) as handle:
            stored = {name: handle[name] for name in handle.files}
        return {
            "status": "ok",
            "result": result,
            "directory": str(directory),
            "pdb_text": pdb.read_text(encoding="utf-8"),
            "arrays": stored,
        }
    return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    design.require_fresh_out(args.out, COMPLETION)
    backbones = {row["backbone_id"]: row for row in design.read_jsonl(args.backbones)}
    parents: dict[str, dict[str, Any]] = {}
    for backbone_id, backbone in backbones.items():
        found = load_object(list(args.parent_structure), backbone["parent_sequence_sha256"])
        if found is None or found["status"] != "ok":
            raise SystemExit(
                f"{backbone_id}: the staged parent fold "
                f"{backbone['parent_sequence_sha256']} is not available in "
                f"{[str(path) for path in args.parent_structure]}. The ladder's parent "
                "folds are reused rather than recomputed, so a missing one is a defect "
                "in the path, not an occasion to refold"
            )
        parents[backbone_id] = found

    rows: list[dict[str, Any]] = []
    census: Counter[str] = Counter()
    started = time.monotonic()
    for path in args.variants:
        for variant in design.read_jsonl(path):
            key = f"{variant['arm']}|{variant['rung']}"
            if variant["status"] != "filled" or not variant["sequence"]:
                census[f"{key}|unfilled"] += 1
                rows.append(
                    {
                        "id": variant["id"],
                        "arm": variant["arm"],
                        "condition": variant["condition"],
                        "rung": variant["rung"],
                        "backbone_id": variant["backbone_id"],
                        "draw": variant["draw"],
                        "fold_status": "no_sequence",
                    }
                )
                continue
            found = load_object(list(args.structure), variant["sequence_sha256"])
            if found is None:
                census[f"{key}|fold_missing"] += 1
                rows.append(
                    {
                        "id": variant["id"],
                        "arm": variant["arm"],
                        "condition": variant["condition"],
                        "rung": variant["rung"],
                        "backbone_id": variant["backbone_id"],
                        "draw": variant["draw"],
                        "fold_status": "missing",
                    }
                )
                continue
            if found["status"] != "ok":
                census[f"{key}|{found['status']}"] += 1
                rows.append(
                    {
                        "id": variant["id"],
                        "arm": variant["arm"],
                        "condition": variant["condition"],
                        "rung": variant["rung"],
                        "backbone_id": variant["backbone_id"],
                        "draw": variant["draw"],
                        "fold_status": str(found["status"]),
                    }
                )
                continue
            span = variant["window_span"]
            readouts = fold_geometry.confidence_readouts(
                found["arrays"],
                start=span[0] if span else None,
                stop=span[1] if span else None,
            )
            row: dict[str, Any] = {
                "id": variant["id"],
                "arm": variant["arm"],
                "condition": variant["condition"],
                "rung": variant["rung"],
                "nominal_extent": variant["nominal_extent"],
                "backbone_id": variant["backbone_id"],
                "draw": variant["draw"],
                "sequence_sha256": variant["sequence_sha256"],
                "fold_status": "ok",
                "fold_object_directory": found["directory"],
                "length": readouts["length"],
                **{key2: value for key2, value in readouts.items() if key2 != "length"},
            }
            if variant["rung"] != design.FULL_GENERATION and span is not None:
                row.update(
                    fold_geometry.compare_to_parent(
                        found["pdb_text"],
                        parents[variant["backbone_id"]]["pdb_text"],
                        start=span[0],
                        stop=span[1],
                    )
                )
                row["parent_mean_ca_plddt"] = float(
                    backbones[variant["backbone_id"]]["parent_mean_ca_plddt"]
                )
                row["delta_mean_ca_plddt"] = row["mean_ca_plddt"] - row["parent_mean_ca_plddt"]
            else:
                row["parent_comparison"] = "absent: the anchor has no parent fold and no correspondence"
            census[f"{key}|ok"] += 1
            rows.append(row)
        print(
            json.dumps({"file": str(path), "rows": len(rows),
                        "elapsed_seconds": round(time.monotonic() - started, 1)}),
            flush=True,
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    digest = design.write_jsonl(out / RECORDS, rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "records": RECORDS,
        "records_sha256": digest,
        "n_rows": len(rows),
        "census": dict(sorted(census.items())),
        "geometry": {
            "schema_version": fold_geometry.SCHEMA_VERSION,
            "primary": design.PRIMARY_STRUCTURE_READOUT,
            "local": design.LOCAL_STRUCTURE_READOUT,
            "lddt_inclusion_radius_angstrom": fold_geometry.LDDT_INCLUSION_RADIUS,
            "lddt_thresholds": list(fold_geometry.LDDT_THRESHOLDS),
            "correspondence": "identity; every window rung preserves length exactly",
        },
        "structure_roots": [str(path) for path in args.structure],
        "parent_structure_roots": [str(path) for path in args.parent_structure],
        "variants": {str(path): sha256_file(path) for path in args.variants},
        "backbones": {"path": str(args.backbones), "sha256": sha256_file(args.backbones)},
        "ceiling": {
            "confidence_is_not_stability": design.CEILING["confidence_is_not_stability"],
            "nothing_is_measured": design.CEILING["nothing_is_measured"],
            "length_is_the_first_confound": design.CEILING["length_is_the_first_confound"],
            "search_is_a_heuristic": (
                "the TM-score superposition search is the standard iterative-extension "
                "heuristic with subsampled seed starts, so it is a lower bound on the "
                "maximum. The all-residue superposition score is reported beside it"
            ),
        },
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    write_json(out / COMPLETION, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="accepted and unused; this stage is CPU-only")
    parser.add_argument("--backbones", type=Path, required=True)
    parser.add_argument("--variants", type=Path, nargs="+", required=True)
    parser.add_argument("--structure", type=Path, nargs="+", required=True,
                        help="run_structure_evidence.py trees holding the variant folds")
    parser.add_argument("--parent-structure", type=Path, nargs="+", required=True,
                        help="staged tree holding the backbones' own folds")
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"n_rows": payload["n_rows"], "census": payload["census"]}))


if __name__ == "__main__":
    main()
