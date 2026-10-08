#!/usr/bin/env python3
"""Apply the restored Pfam/HMMER oracle to a sequence ledger, and refuse a broken one.

The oracle this programme's generation endpoints are measured on is HMMER 3.4
scanning a pressed Pfam-A at the release's own gathering thresholds. The paths
``configs/generation_replication_manifest.json`` pins are the authoring host's
and no longer resolve, so the release has to be restored -- from
``results/R6/generation_inputs_20260905/oracle-inputs.tar.gz``, or by building
HMMER from ``external/tools/hmmer-3.4.tar.gz`` and pressing the Pfam HMMs in
``external/references/``. ``--oracle-root`` names the restored tree.

Nothing here is best-effort. All six pinned files must be present and must hash
to the pinned digests, and the declared recognition controls must reproduce the
family sets the frozen 2026-09-05 oracle run recorded -- and must still return
*no* family where that run recorded none -- before a single generated sequence is
scored. A silently broken oracle and a model that generated nothing recognisable
produce the same artefact, and only one of them is a result.

``--select`` filters the input ledger on declared record fields, so the one
oracle stage serves the fresh conditional cells, the retained natural references
that price the instrument, and the family-matched natural draw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import write_json  # noqa: E402
from src.capability.generation import family_oracle as fo  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "configs/generation_replication_manifest.json"
COMPLETION = "family_recognition.json"
SCHEMA_VERSION = "d1_family_recognition_v1"


def _parse_selection(values: list[str]) -> dict[str, set[str]]:
    selection: dict[str, set[str]] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--select takes field=value[,value]; got {item!r}")
        field, raw = item.split("=", 1)
        selection.setdefault(field.strip(), set()).update(
            part.strip() for part in raw.split(",") if part.strip()
        )
    return selection


def _selected(row: dict[str, Any], selection: dict[str, set[str]]) -> bool:
    return all(str(row.get(field)) in wanted for field, wanted in selection.items())


def _restore(archive: Path, root: Path) -> dict[str, Any]:
    """Extract a pinned oracle archive into its own empty directory.

    The archive is untrusted data: it is extracted into a directory of its own,
    nothing is imported from it, and the binary it carries is invoked only by
    absolute path. The digests are checked afterwards by ``load_oracle``, so a
    tampered or truncated archive cannot be scored with.
    """

    import tarfile

    if not archive.is_file():
        raise SystemExit(
            f"--oracle-archive {archive} does not exist, and --oracle-root carries no "
            "usable oracle. Recognition is refused rather than returning an empty set"
        )
    root.mkdir(parents=True, exist_ok=True)
    wanted = ("hmmer/bin/hmmscan", "pfam/")
    with tarfile.open(archive, "r:*") as handle:
        members = [
            member
            for member in handle.getmembers()
            if member.name.startswith(wanted) and not member.isdev()
        ]
        if not members:
            raise SystemExit(f"{archive} carries no hmmer/bin/hmmscan or pfam/ member")
        for member in members:
            resolved = (root / member.name).resolve()
            if not str(resolved).startswith(str(root.resolve())):
                raise SystemExit(f"{archive} member {member.name!r} escapes {root}")
        handle.extractall(root, members=members)
    return {"archive": str(archive), "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "extracted_to": str(root), "n_members": len(members)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    pinned = fo.pinned_digests(args.pinned_manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.work or (args.out / "work")
    work.mkdir(parents=True, exist_ok=True)

    restoration: dict[str, Any] | None = None
    root = args.oracle_root
    if root is None or not (root / fo.HMMSCAN_RELATIVE).is_file():
        if args.oracle_archive is None:
            raise SystemExit(
                "neither --oracle-root nor --oracle-archive supplies a restored "
                "Pfam/HMMER oracle. Recognition is refused rather than returning an "
                "empty recognition set, which would read as a true-negative yield of zero"
            )
        root = args.oracle_root or (work / "oracle_release")
        restoration = _restore(args.oracle_archive, root)
    oracle = fo.load_oracle(root, pinned=pinned)

    control = fo.run_positive_control(
        oracle,
        workspace=work,
        controls=None if args.controls is None else json.loads(args.controls.read_text())["controls"],
        threads=args.threads,
    )
    write_json(args.out / "oracle_control.json", control)

    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in args.sequences:
        block = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not block:
            raise SystemExit(f"{path} carries no record")
        sources.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "n_records": len(block),
            }
        )
        rows.extend(block)
    selection = _parse_selection(args.select)
    chosen = [row for row in rows if _selected(row, selection)]
    if not chosen:
        raise SystemExit(
            f"the selection {selection} matched none of the {len(rows)} records in "
            f"{[str(path) for path in args.sequences]}; an empty recognition set is "
            "refused rather than scored"
        )
    identifiers = [str(row["id"]) for row in chosen]
    if len(set(identifiers)) != len(identifiers):
        raise SystemExit(
            "the selected records carry duplicate identifiers; two ledgers describing "
            "the same attempt cannot be scored as one set"
        )

    sequences = {str(row["id"]): str(row.get("sequence") or "") for row in chosen}
    empty = sorted(name for name, value in sequences.items() if not value)
    hits, receipt = fo.recognise(
        oracle,
        sequences,
        workspace=work,
        shard_size=args.shard_size,
        workers=args.workers,
        threads=args.threads,
        label=args.label,
    )

    referents: dict[str, dict[str, Any]] = {}
    for path in args.anchors:
        payload = json.loads(path.read_text(encoding="utf-8"))
        arm = str(payload.get("arm") or path.stem)
        referents[arm] = fo.class_referents(payload)

    annotations: list[dict[str, Any]] = []
    for row in chosen:
        name = str(row["id"])
        block = hits.get(name, {"families": [], "any_family": False,
                               "best_profile_coverage": None, "complete_domain": False})
        record: dict[str, Any] = {
            "id": name,
            "sequence_sha256": row.get("sequence_sha256")
            or hashlib.sha256(sequences[name].encode()).hexdigest(),
            "arm": row.get("arm"),
            "class_key": row.get("class_key"),
            "condition": row.get("condition"),
            "role": row.get("role"),
            "stage": row.get("stage"),
            "stream": row.get("stream"),
            "length": row.get("length"),
            "pfam_families": block["families"],
            "any_profile_hit": block["any_family"],
            "best_profile_coverage": block["best_profile_coverage"],
            "complete_domain": block["complete_domain"],
            "profile_search_status": "empty_not_searched" if name in set(empty) else "searched",
            "target_profile_hit": None,
            "target_referent": None,
        }
        arm = str(row.get("arm") or "")
        class_key = str(row.get("class_key") or "")
        if arm in referents and class_key in referents[arm]:
            referent = referents[arm][class_key]["referent"]
            record["target_referent"] = list(referent)
            record["target_class_admitted"] = referents[arm][class_key]["admitted"]
            hit, empty = fo.target_hit_for_class(block["families"], referent)
            record["target_profile_hit"] = hit
            record["target_referent_empty"] = empty
        annotations.append(record)

    sidecar = args.out / "family_recognition.jsonl"
    sidecar.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in annotations), encoding="utf-8"
    )

    scored = [row for row in annotations if row["target_profile_hit"] is not None]
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": args.label,
        "sequences": sources,
        "oracle_restoration": restoration,
        "selection": {field: sorted(values) for field, values in selection.items()},
        "n_records_in_ledger": len(rows),
        "n_selected": len(chosen),
        "n_empty_not_searched": len(empty),
        "n_with_any_family": sum(1 for row in annotations if row["any_profile_hit"]),
        "n_with_complete_domain": sum(1 for row in annotations if row["complete_domain"]),
        "n_with_target_referent": len(scored),
        "n_target_profile_hits": sum(1 for row in scored if row["target_profile_hit"]),
        "anchors": [str(path) for path in args.anchors],
        "oracle": oracle.record(),
        "oracle_control": {
            "passed": control["passed"],
            "n_positive": control["n_positive"],
            "n_negative": control["n_negative"],
        },
        "scan_receipt": {key: value for key, value in receipt.items() if key != "shards"},
        "n_shards": len(receipt.get("shards", [])),
        "annotations_jsonl": str(sidecar),
        "annotations_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
    }
    write_json(args.out / COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, nargs="+", required=True,
                        help="one or more JSONL ledgers of records carrying id and sequence")
    parser.add_argument("--oracle-root", type=Path, default=None,
                        help="the restored tree holding hmmer/bin/hmmscan and pfam/Pfam-A.hmm*")
    parser.add_argument("--oracle-archive", type=Path, default=None,
                        help="a pinned oracle tar.gz to extract when --oracle-root is absent")
    parser.add_argument("--pinned-manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="the manifest whose oracle_inputs digests the restored release must match")
    parser.add_argument("--controls", type=Path, default=None,
                        help="override the declared recognition controls; the default set ships with the package")
    parser.add_argument("--anchors", type=Path, nargs="*", default=[],
                        help="anchors_<arm>.json files supplying the frozen class-to-Pfam referent map")
    parser.add_argument("--select", nargs="*", default=[],
                        help="field=value[,value] filters on the ledger, e.g. role=generation condition=mismatched")
    parser.add_argument("--label", default="recognition")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=200)
    parser.add_argument("--work", type=Path, default=None,
                        help="scan workspace; defaults to <out>/work and is never a data directory")
    parser.add_argument("--device", default="cpu", help="accepted because the campaign queue injects it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = run(args)
    print(json.dumps({key: record[key] for key in ("status", "label", "n_selected",
                                                   "n_with_any_family", "n_with_complete_domain",
                                                   "n_target_profile_hits")}, sort_keys=True))


if __name__ == "__main__":
    main()
