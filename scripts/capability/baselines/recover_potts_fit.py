#!/usr/bin/env python3
"""Independent q00059 final-fit producer with one bounded L-BFGS history reset.

Build patches a COPY of plmc-18c9e55; prepare reuses the frozen export functions
and all five original CV decisions. Run is a NEW cold 5000-attempt final fit,
not a float32 warm start. Only the first finite ROUNDING_ERROR may reset history;
the failed outer search counts against the shared cap. No score assembly occurs.

Typical explicit stages (run only after review and immutable H200 staging)::

    recover_potts_fit.py build --vendor .../debbiemarkslab-plmc-18c9e55 --build-dir ...
    recover_potts_fit.py prepare --snapshot ... --background .../q00059.json \
        --build-dir ... --out .../potts-numerical-recovery/q00059
    recover_potts_fit.py run --out .../potts-numerical-recovery/q00059

All new output must be outside the original snapshot. No invocation reuses an
existing run directory. The receipt records true stop status independently of
exit status; a successful receipt is NOT an accepted-score/merged-background
record. final.double and failure-episode-*.double contain unquantized x in
internal xHi/xEij layout: 'PLM64LE\\n', three little-endian int32 (L,q,n), then
n little-endian doubles. final.params is the separate vendor float32 export.
Historical compiler argv is unknown; the new build manifest makes no claim of
binary-equivalent production arithmetic.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PATCH = Path(__file__).with_name("assets") / "plmc-bounded-restart.patch"
MEMORY_LIMIT = 10 << 30


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def check_files(files: dict[str, str]) -> None:
    for name, expected in files.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"content changed: {name}")


def resources() -> dict:
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return {"available_memory_bytes": int(info["MemAvailable"].split()[0]) * 1024,
            "affinity_cpus": len(os.sched_getaffinity(0)),
            "python": sys.version.split()[0]}


def build(vendor: Path, out: Path, *, testing: bool = False) -> dict:
    """testing is a Python-test-only build option, not a production CLI switch."""
    vendor, out = vendor.resolve(), out.resolve()
    if out.is_relative_to(vendor):
        raise ValueError("build must not modify the vendor tree")
    original = {str(p): digest(p) for p in sorted((vendor / "src").rglob("*")) if p.is_file()}
    if not original or not (vendor / "src/inference.c").is_file():
        raise ValueError("complete plmc vendor source required")
    out.mkdir(parents=True, exist_ok=False)
    save_new(out / "resources.json", resources())
    shutil.copytree(vendor / "src", out / "src")
    patch_argv = ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(PATCH)]
    with (out / "patch.log").open("x") as log:
        subprocess.run(patch_argv, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
    gcc = shutil.which("gcc")
    if gcc is None:
        raise RuntimeError("gcc is required; no dependency installation is attempted")
    argv = [gcc, "-O3", "-std=c99", "-fopenmp", "-ffunction-sections", "-fdata-sections"]
    if testing:
        argv.append("-DPLMC_RECOVERY_TESTING")
    argv += [str(out / "src" / p) for p in
             ("main.c", "plm.c", "inference.c", "weights.c", "lib/lbfgs.c", "lib/twister.c")]
    argv += ["-Wl,--gc-sections", "-lm", "-o", str(out / "plmc-recovery")]
    with (out / "compile.log").open("x") as log:
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    check_files(original)
    manifest = {"producer": "plmc-bounded-history-restart", "testing": testing,
                "original_vendor_files": original, "patch_sha256": digest(PATCH),
                "driver_sha256": digest(Path(__file__)), "patch_argv": patch_argv,
                "compile_argv": argv, "compiler": subprocess.check_output(
                    [gcc, "--version"], text=True).splitlines()[0],
                "binary_sha256": digest(out / "plmc-recovery"),
                "patched_sources": {str(p.relative_to(out)): digest(p)
                                    for p in sorted((out / "src").rglob("*")) if p.is_file()},
                "historical_compile_argv": None}
    save_new(out / "build.json", manifest)
    return manifest


def verify_build(build_dir: Path) -> dict:
    m = json.loads((build_dir / "build.json").read_text())
    if m["testing"]:
        raise ValueError("test-injection build cannot be a scientific producer")
    if m["driver_sha256"] != digest(Path(__file__)) or m["patch_sha256"] != digest(PATCH):
        raise ValueError("driver/patch changed since build")
    check_files({str(build_dir / p): h for p, h in m["patched_sources"].items()})
    check_files({str(build_dir / "plmc-recovery"): m["binary_sha256"]})
    return m


def prepare(snapshot: Path, background: Path, build_dir: Path, out: Path) -> dict:
    snapshot, background, build_dir, out = (p.resolve() for p in (snapshot, background, build_dir, out))
    if out.is_relative_to(snapshot) or "potts-numerical-recovery" not in out.parts:
        raise ValueError("use an independent potts-numerical-recovery directory outside original snapshot")
    if not background.is_relative_to(snapshot):
        raise ValueError("background must belong to the original snapshot")
    producer = verify_build(build_dir)
    record = json.loads(background.read_text())
    if (record["wildtype_id"] != "q00059" or record["length"] != 490
            or record["status"] != "final_not_converged_after_caps"
            or record["selected_lambda_e"] != 80 or record["lambda_h"] != .01
            or record["threads"] != 4 or len(record["sweep"]) != 5
            or not all(e["converged"] for e in record["sweep"])
            or record["final"]["status_text"] != "ROUNDING_ERROR"
            or record["final"]["max_iter"] != 5000):
        raise ValueError("not the approved q00059 failure and five-qualified-CV selection")
    fasta, weights = (snapshot / record["support"][k] for k in ("msa", "weights"))
    old_binary = snapshot / record["plmc"]["path"]
    old_params = Path(record["params"]["path"])
    if not all(p.resolve().is_relative_to(snapshot) for p in (fasta, weights, old_binary, old_params)):
        raise ValueError("original inputs must be inside snapshot")
    original = {str(background): digest(background),
                str(fasta): record["alignment"]["msa_sha256"],
                str(weights): record["alignment"]["weights_sha256"],
                str(old_binary): record["plmc"]["sha256"],
                str(old_params): record["params"]["sha256"]}
    check_files(original)
    # No alternate FASTA/weight convention: use the exact frozen producer.
    sys.path.insert(0, str(snapshot))
    b = importlib.import_module("src.capability.baselines.msa_baselines")
    expected_module = snapshot / "src/capability/baselines/msa_baselines.py"
    if b.__file__ is None or Path(b.__file__).resolve() != expected_module:
        raise RuntimeError("a different msa_baselines module was already imported; use a fresh CLI process")
    original[str(expected_module)] = digest(expected_module)
    wildtype = b.read_fasta(fasta)[0][1]
    rows, row_weights = b.read_exported_alignment(fasta, weights, wildtype)
    out.mkdir(parents=True, exist_ok=False)
    save_new(out / "preparation-resources.json", resources())
    b.write_query_first_alignment(out / "full.fa", out / "full.weights", wildtype, rows, row_weights)
    if digest(out / "full.fa") != digest(fasta) or digest(out / "full.weights") != digest(weights):
        raise ValueError("reconstructed full inputs do not equal frozen export bytes")
    check_files(original)
    manifest = {"producer": producer, "build_dir": str(build_dir), "original_files": original,
                "original_status": record["status"], "original_final": record["final"],
                "reused_cv": record["sweep"], "selected_lambda_e": 80., "lambda_h": .01,
                "initialization": "original cold initializer; no parameter loading",
                "total_outer_attempt_cap": 5000, "max_history_resets": 1,
                "epsilon": .001, "threads": 4,
                "inputs": {"full.fa": digest(out / "full.fa"),
                           "full.weights": digest(out / "full.weights")},
                "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    save_new(out / "preparation.json", manifest)
    return manifest


def read_events(log: str) -> list[dict[str, str]]:
    return [dict(item.split("=", 1) for item in line.split()[1:])
            for line in log.splitlines() if line.startswith("RECOVERY ")]


def classify(returncode: int, events: list[dict[str, str]], cap: int) -> dict:
    finals = [e for e in events if e["event"] == "final"]
    final = finals[0] if len(finals) == 1 else None
    success = False
    if final is not None:
        values = [float(final[k]) for k in ("objective", "xnorm", "gnorm", "ratio")]
        success = (returncode == 0 and int(final["return_code"]) in (0, 2)
                   and final["accepted"] == "1" and final["unquantized"] == "1"
                   and all(math.isfinite(v) for v in values) and 0 <= values[3] <= .001
                   and 0 <= int(final["cumulative"]) <= cap
                   and sum(e["event"] == "restart" for e in events) <= 1)
    # Preserve textual scalar values (including nan/inf) without invalid JSON.
    return {"scientific_success": success,
            "status": "converged_independent_producer" if success else "failed",
            "process_returncode": returncode, "true_final_event": final,
            "episodes": [e for e in events if e["event"] == "episode_end"],
            "rounding_traces": [e for e in events if e["event"] == "rounding_trace"],
            "history_resets": sum(e["event"] == "restart" for e in events)}


def execute(out: Path) -> dict:
    out = out.resolve()
    m = json.loads((out / "preparation.json").read_text())
    build_dir = Path(m["build_dir"])
    producer = verify_build(build_dir)
    if producer != m["producer"]:
        raise ValueError("producer changed after preparation")
    check_files(m["original_files"])
    check_files({str(out / p): h for p, h in m["inputs"].items()})
    r = resources()
    if (r["available_memory_bytes"] < MEMORY_LIMIT + (2 << 30) or r["affinity_cpus"] < 4
            or shutil.disk_usage(out).free < (3 << 30)):
        raise RuntimeError("insufficient CPU/memory/disk headroom; nothing launched")
    argv = [str(build_dir / "plmc-recovery"), "-o", str(out / "final.params"),
            "-f", "focus", "-le", "80", "-lh", "0.01", "-m", "5000", "-g", "-n", "4",
            "-w", str(out / "full.weights"), str(out / "full.fa")]
    env = {k: v for k, v in os.environ.items() if not k.startswith("PLMC_TEST_")}
    env.update(OMP_NUM_THREADS="4", OMP_DYNAMIC="FALSE", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    save_new(out / "run-start.json", {"argv": argv, "resources": r,
                                     "memory_limit_bytes": MEMORY_LIMIT, "cpu_count": 4})
    affinity = sorted(os.sched_getaffinity(0))[:4]

    def limits() -> None:
        os.sched_setaffinity(0, affinity)
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))

    started = time.monotonic()
    receipt = {"status": "failed", "scientific_success": False, "producer": producer,
               "original_status": m["original_status"], "score_assembly": "not_performed"}
    try:
        with (out / "fit.log").open("x") as log:
            p = subprocess.run(argv, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                               preexec_fn=limits, start_new_session=True, check=False)
        events = read_events((out / "fit.log").read_text())
        receipt.update(classify(p.returncode, events, 5000))
        check_files(m["original_files"])
        verify_build(build_dir)
        receipt["original_bytes_unchanged"] = True
        receipt["artifacts"] = {p.name: digest(p) for p in out.iterdir()
                                if p.name in ("final.params", "final.double", "fit.log")
                                or p.name.startswith("failure-episode-")}
        if receipt["scientific_success"] and not {"final.params", "final.double"} <= receipt["artifacts"].keys():
            raise RuntimeError("success without required artifacts")
    except BaseException as exc:
        receipt.update(status="failed", scientific_success=False, error=str(exc))
        raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic() - started,
                       child_maxrss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        save_new(out / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("build")
    p.add_argument("--vendor", type=Path, required=True)
    p.add_argument("--build-dir", type=Path, required=True)
    p = sub.add_parser("prepare")
    for name in ("snapshot", "background", "build-dir", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "build":
        build(args.vendor, args.build_dir)
    elif args.action == "prepare":
        prepare(args.snapshot, args.background, args.build_dir, args.out)
    else:
        return 0 if execute(args.out)["scientific_success"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
