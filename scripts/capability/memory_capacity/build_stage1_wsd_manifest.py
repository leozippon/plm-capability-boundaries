#!/usr/bin/env python3
"""Generate the canonical 70-run Stage-1 WSD calibration JSON manifest for Spot workers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.memory_capacity.spec import (  # noqa: E402
    SCALE_SPECS,
    assert_no_forbidden_infra_strings,
    build_stage1_wsd_calibration_runs,
)
from src.capability.memory_theory.io_utils import write_json_with_sha256  # noqa: E402

MANIFEST_SCHEMA = "plm_parametric_capacity_stage1_wsd_manifest_v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination path for the Stage-1 WSD manifest JSON",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=20261127,
        help="Deterministic synthetic dataset seed (default: 20261127)",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=20261128,
        help="Deterministic model initialization seed (default: 20261128)",
    )
    parser.add_argument(
        "--min-steps-per-epoch",
        type=int,
        default=16,
        help="Minimum optimizer steps per epoch for tiny-dataset batch capping (default: 16)",
    )
    args = parser.parse_args(argv)

    runs = build_stage1_wsd_calibration_runs(
        data_seed=args.data_seed,
        model_seed=args.model_seed,
        min_steps_per_epoch=args.min_steps_per_epoch,
    )
    n_single = sum(1 for r in runs if not r.ddp_preferred)
    n_ddp = sum(1 for r in runs if r.ddp_preferred)
    by_paradigm: dict[str, int] = {}
    for r in runs:
        by_paradigm[r.paradigm] = by_paradigm.get(r.paradigm, 0) + 1

    payload = {
        "schema": MANIFEST_SCHEMA,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_runs": len(runs),
        "n_single_gpu": n_single,
        "n_ddp_preferred": n_ddp,
        "runs_by_paradigm": by_paradigm,
        "total_estimated_tokens": sum(int(r.estimated_tokens) for r in runs),
        "total_estimated_param_tokens": sum(
            int(r.estimated_tokens) * int(SCALE_SPECS[r.scale_id].exact_non_embedding_params)
            for r in runs
        ),
        "runs": [r.to_dict() for r in runs],
    }
    assert_no_forbidden_infra_strings(payload, label=args.out.name)
    json.dumps(payload, sort_keys=True, allow_nan=False)
    digest = write_json_with_sha256(args.out, payload)
    print(
        f"Wrote {len(runs)} Stage-1 WSD runs (single_gpu={n_single}, ddp={n_ddp}, "
        f"by_paradigm={by_paradigm}) -> {args.out} (sha256={digest[:16]}...)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
