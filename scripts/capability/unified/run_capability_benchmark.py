#!/usr/bin/env python3
"""CLI entrypoint for evaluating external or builtin protein models on the capability benchmark.

Examples:
    python scripts/capability/unified/run_capability_benchmark.py --hf-mlm facebook/esm2_t6_8M_UR50D --mode smoke --suites all --out-dir /tmp/bench
    python scripts/capability/unified/run_capability_benchmark.py --hf-causal hugohrban/progen2-small --mode smoke --suites validate,r1_dms
    python scripts/capability/unified/run_capability_benchmark.py --adapter path/to/my_model.py:MyProteinAdapter --mode smoke --suites all
    python scripts/capability/unified/run_capability_benchmark.py --builtin dima-esm2-8m --mode smoke --suites all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.benchmark import (
    ALL_SUITES,
    HuggingFaceCausalPLMAdapter,
    HuggingFaceMaskedPLMAdapter,
    load_builtin_adapter,
    load_custom_adapter,
    run_benchmark,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the reusable protein capability benchmark on any model adapter.")
    src_group = p.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--hf-causal", type=str, help="HuggingFace AutoModelForCausalLM name or local path.")
    src_group.add_argument("--hf-mlm", type=str, help="HuggingFace AutoModelForMaskedLM name or local path.")
    src_group.add_argument("--adapter", type=str, help="Dynamic plugin spec 'path/to/file.py:Class' or 'pkg.mod:Class'.")
    src_group.add_argument("--builtin", type=str, help="First-party roster arm (e.g. dima-esm2-8m, dplm2-150m, proteinmpnn-v-020).")

    p.add_argument("--model-id", type=str, default=None, help="Optional override model_id for HuggingFace adapters.")
    p.add_argument("--mode", choices=("smoke", "full"), default="smoke", help="Evaluation cohort size (default: smoke).")
    p.add_argument(
        "--suites",
        type=str,
        default="all",
        help=f"Comma-separated suites or 'all'. Available: {','.join(ALL_SUITES)}",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Directory to write report JSON, .sha256, and summary MD.")
    p.add_argument("--unified-dir", type=Path, default=None, help="Optional directory containing sealed unified reference JSONs.")
    p.add_argument("--device", type=str, default="auto", help="Torch device (auto, cpu, cuda:0, ...).")
    p.add_argument("--dtype", type=str, default="float32", help="Precision (float32, bfloat16, float16).")
    p.add_argument("--seed", type=int, default=20261124, help="Deterministic evaluation seed.")
    p.add_argument("--local-files-only", action="store_true", help="Disable HuggingFace Hub network calls.")
    p.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to HuggingFace loader.")
    p.add_argument("--adapter-kwargs", type=str, default="{}", help="Optional JSON object of kwargs for --adapter.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]

    if args.hf_causal:
        adapter = HuggingFaceCausalPLMAdapter(
            args.hf_causal,
            model_id=args.model_id,
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.hf_mlm:
        adapter = HuggingFaceMaskedPLMAdapter(
            args.hf_mlm,
            model_id=args.model_id,
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.adapter:
        extra = json.loads(args.adapter_kwargs)
        if not isinstance(extra, dict):
            raise ValueError("--adapter-kwargs must decode to a JSON object")
        adapter = load_custom_adapter(args.adapter, **extra)
    else:
        adapter = load_builtin_adapter(args.builtin, device=args.device, dtype=args.dtype)

    report = run_benchmark(
        adapter,
        suites=suites,
        mode=args.mode,
        out_dir=args.out_dir,
        unified_dir=args.unified_dir,
        seed=args.seed,
    )
    print(json.dumps({"model_id": report["metadata"]["model_id"], "mode": report["mode"], "suites": {k: v.get("status") for k, v in report["suites"].items()}, "artifacts": report.get("artifacts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
