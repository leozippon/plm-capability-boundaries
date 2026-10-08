#!/usr/bin/env python3
"""The aggregated conditional endpoint: requested minus mismatched, resampled over classes.

``conditional/panel/requested_minus_mismatched`` is the one quantity the
manuscript's conditional panel needs and no run has ever computed. The per-cell
rates have existed since the 2026-09-05 evidence stream and the per-arm aggregate
under the campaign's pre-registered convention since 2026-08-26; what is absent
is the class-resampled paired difference on the support the panel quotes -- all
sixteen drawn classes at the attempt-level rate.

This stage computes it. Both declared conventions are reported side by side,
because they differ in support (all sixteen drawn classes versus the fourteen and
fifteen the per-class instrument gate admitted) and in estimator (one unit per
attempt versus one unit per near-duplicate group), and neither is a correction of
the other. The resampling unit is the class and the interval comes from the
package's one class-clustered resampler; no bootstrap is written here.

A **stream** is the set of ledgers that together carry both sides of the
difference for one sampling stream, and ``--ledger <stream>:<path>`` is how it is
declared. This matters because the generation cells are one-sided: the
replication campaign sampled only the requested condition, and the cells that
sampled the missing mismatched side are separate files. A stream label that
collects only one condition cannot form a within-class paired difference and is
refused, naming what is missing, rather than quietly reporting a one-sided rate.

Each stream yields a within-stream class-clustered interval; across streams the
summary is an equal-weight Student-t interval at the stream as the unit, which is
the replication campaign's own declared uncertainty rule. ZymCTRL and ProLLaMA
are reported side by side and are never pooled or differenced: they are asked for
different kinds of class through different oracles.

Ledgers that carry only an oracle family set and no ``target_profile_hit`` -- the
replication campaign's own annotated attempts, which predate any mismatched cell
-- have it derived from the frozen class-to-Pfam referent through the one shared
definition in ``family_oracle.target_hit_for_class``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import write_json  # noqa: E402
from src.capability.generation import conditional_contrast as cc  # noqa: E402
from src.capability.generation import conditioned_generation as cg  # noqa: E402
from src.capability.generation import family_oracle as fo  # noqa: E402

DEFAULT_QUEUE = REPO_ROOT / "results/R6/conditioned_generation_queue_20260826/class_queue.json"
COMPLETION = "conditional_contrast.json"


def _join_recognition(rows: list[dict[str, Any]], sidecars: list[Path]) -> list[dict[str, Any]]:
    """Attach recognition outcomes from the oracle stage to attempts that lack them."""

    if not sidecars:
        return rows
    recognition: dict[str, dict[str, Any]] = {}
    for path in sidecars:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            recognition[str(record["id"])] = record
    joined: list[dict[str, Any]] = []
    for row in rows:
        merged = dict(row)
        found = recognition.get(str(row["id"]))
        if found is not None:
            if found.get("sequence_sha256") and row.get("sequence_sha256") and found[
                "sequence_sha256"
            ] != row["sequence_sha256"]:
                raise SystemExit(
                    f"attempt {row['id']} and its recognition record describe different "
                    "sequences; the two artefacts are not the same run"
                )
            for field in ("pfam_families", "any_profile_hit", "best_profile_coverage",
                          "complete_domain", "target_profile_hit", "target_referent"):
                if field in found:
                    merged[field] = found[field]
        joined.append(merged)
    return joined


def _parse_ledger(value: str) -> tuple[str, Path]:
    stream, separator, path = value.partition(":")
    if not separator or not stream or not path:
        raise SystemExit(
            f"--ledger takes stream:path, got {value!r}; the stream label is explicit "
            "so a stream is never inferred from a path, and both the requested and the "
            "mismatched ledgers of one sampling stream carry the same label"
        )
    return stream, Path(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    queue = cg.load_queue(args.queue)
    referents: dict[str, dict[str, Any]] = {}
    for path in args.anchors:
        payload = json.loads(path.read_text(encoding="utf-8"))
        arm = str(payload.get("arm") or path.stem)
        referents[arm] = fo.class_referents(payload)

    by_stream: dict[str, list[Path]] = defaultdict(list)
    for value in args.ledger:
        stream, path = _parse_ledger(value)
        by_stream[stream].append(path)

    # The unconditioned floor arms were sampled once, under no class request, and the
    # campaign that declared this endpoint read one floor measurement from both
    # conditioned arms. Supplying it here lets a stream that contains only conditioned
    # cells still answer clause 2; the floor is shared across streams rather than
    # resampled per stream, and the artefact says so.
    floor_rows: list[dict[str, Any]] = []
    floor_inputs: list[dict[str, Any]] = []
    for path in args.floor_ledger:
        block = cc.read_attempts(path)
        floor_rows.extend(block)
        floor_inputs.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "n_records": len(block),
            }
        )

    streams: dict[str, dict[str, Any]] = {}
    inputs: list[dict[str, Any]] = []
    for stream, paths in sorted(by_stream.items()):
        rows: list[dict[str, Any]] = []
        for path in paths:
            block = cc.read_attempts(path)
            inputs.append(
                {
                    "stream": stream,
                    "path": str(path),
                    "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    "n_records": len(block),
                }
            )
            rows.extend(block)
        identifiers = [str(row["id"]) for row in rows]
        if len(set(identifiers)) != len(identifiers):
            raise SystemExit(
                f"stream {stream!r} collects ledgers that share attempt identifiers; "
                "the same attempt cannot enter one stream twice"
            )
        rows = _join_recognition(rows, args.recognition)
        per_arm: dict[str, Any] = {}
        for arm in sorted({str(row.get("arm")) for row in rows} & set(queue["arms"])):
            selected = [
                row
                for row in rows
                if row.get("arm") == arm and row.get("condition") in cc.CONDITIONS
            ]
            if not selected:
                continue
            selected = cc.derive_target_hits(selected, referents.get(arm, {}))
            per_arm[arm] = cc.arm_contrast(
                selected,
                arm=arm,
                pairing=cc.pairing_from_queue(queue, arm),
                referents=referents.get(arm),
                resamples=args.resamples,
                seed=args.bootstrap_seed,
                hit_field=args.hit_field,
            )
        if not per_arm:
            raise SystemExit(
                f"stream {stream!r} carries no requested or mismatched cell of a queued arm"
            )
        floors = {
            arm: cc.floor_rates(rows + floor_rows, referents=referents.get(arm))
            for arm in per_arm
        }
        for arm, contrast in per_arm.items():
            relevant = {
                name: block
                for name, block in floors[arm].items()
                if name in cg.FLOORS.get(arm, ())
            }
            contrast["verdict"] = cc.arm_verdict(contrast, floor=relevant)
        streams[stream] = {
            "arms": per_arm,
            "floors": floors,
            "ledgers": [str(path) for path in paths],
        }

    by_arm: dict[str, dict[str, Any]] = defaultdict(dict)
    for stream, payload in streams.items():
        for arm, contrast in payload["arms"].items():
            by_arm[arm][stream] = contrast

    panel = {
        arm: {
            "primary": {
                stream: {
                    "mean": contrast["primary"]["mean"],
                    "ci95": contrast["primary"]["ci95"],
                    "n_classes": contrast["primary"]["n_classes"],
                    "verdict": contrast["verdict"]["outcome"],
                    "grouping_source": sorted(
                        {
                            block["conditions"][cond]["grouping_source"]
                            for block in contrast["per_class"].values()
                            for cond in cc.CONDITIONS
                        }
                    ),
                }
                for stream, contrast in sorted(per_stream.items())
            },
            "across_streams": {
                key: cc.combine_streams(per_stream, key=key)
                for key in sorted(next(iter(per_stream.values()))["panels"])
            },
            "degeneracy": {
                stream: contrast["degeneracy"] for stream, contrast in sorted(per_stream.items())
            },
        }
        for arm, per_stream in sorted(by_arm.items())
    }

    args.out.mkdir(parents=True, exist_ok=True)
    detail = args.out / "conditional_contrast_detail.json"
    write_json(detail, {"streams": streams})
    record = {
        "schema_version": cc.SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endpoint": "requested_minus_mismatched",
        "endpoint_definition": (
            "per class c: the rate at which generations produced under the native "
            "request for c are assigned to c by the Pfam/HMMER oracle, minus the rate "
            "at which generations produced under the request for c's frozen donor "
            "class are assigned to c. Aggregated as an equal-weight mean over classes "
            "with the class as the resampling unit"
        ),
        "stream_definition": (
            "a stream is the set of ledgers that together carry both conditions of one "
            "sampling stream; the across-stream interval treats the stream as the unit"
        ),
        "primary_convention": {
            "support": cc.PRIMARY_SUPPORT,
            "estimator": cc.PRIMARY_ESTIMATOR,
            "reason": (
                "the manuscript's conditional panel reads all sixteen drawn classes at "
                "the attempt-level rate; the admitted/near-duplicate-collapsed "
                "convention of the 2026-08-26 campaign report is reported beside it and "
                "is not superseded"
            ),
        },
        "grouping_note": (
            "the collapsed estimator uses the near-duplicate group where a ledger "
            "declares one and the exact sequence digest otherwise; grouping_source "
            "records which, per stream"
        ),
        "hit_field": args.hit_field,
        "resamples": int(args.resamples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "resampling_unit": "the class",
        "queue_digest": queue["digest"],
        "streams": {stream: payload["ledgers"] for stream, payload in sorted(streams.items())},
        "inputs": inputs,
        "anchors": [str(path) for path in args.anchors],
        "floor_ledgers": floor_inputs,
        "floor_is_shared_across_streams": bool(floor_inputs),
        "floor_note": (
            "the unconditioned floor arms were sampled once under no class request; "
            "clause 2 reads that one measurement from every stream rather than "
            "resampling a floor per stream"
        ),
        "recognition_sidecars": [str(path) for path in args.recognition],
        "arms": sorted(by_arm),
        "panel": panel,
        "detail_file": str(detail),
        "ceiling": dict(cc.CEILING),
    }
    write_json(args.out / COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", nargs="+", required=True,
                        help="stream:path pairs; all ledgers of one sampling stream share its label")
    parser.add_argument("--floor-ledger", type=Path, nargs="*", default=[],
                        help="ledgers carrying the unconditioned floor arms, read by every stream")
    parser.add_argument("--recognition", type=Path, nargs="*", default=[],
                        help="family_recognition.jsonl sidecars for ledgers that carry no recognition")
    parser.add_argument("--anchors", type=Path, nargs="*", default=[],
                        help="anchors_<arm>.json, which supply the admitted support and the referents")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--hit-field", default="target_profile_hit")
    parser.add_argument("--resamples", type=int, default=cc.BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=cc.BOOTSTRAP_SEED)
    parser.add_argument("--device", default="cpu", help="accepted because the campaign queue injects it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = run(args)
    print(json.dumps(record["panel"], sort_keys=True, default=str)[:4000])


if __name__ == "__main__":
    main()
