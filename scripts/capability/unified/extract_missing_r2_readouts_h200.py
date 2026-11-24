#!/usr/bin/env python3
"""Assemble & extract missing 1024-d readout archives for the unified 62-arm yardstick.

Modes
-----
``assemble-prereg33-stability``
    Pure-CPU GPFS assembly of the 33 pre-registered arms' MegaScale stability
    1024-d readout matrices ``R`` (shape ``(25856, 1024)``, ``float32``) directly
    from the frozen ``stability_singles_extraction_v1`` archives on GPFS, using
    the exact state-difference rule of ``fit_stability_singles.py::load_arm``
    (``projected[state] - projected[0]`` in exact cohort variant order).

``esm3-gpu0``
    Offline GPU extraction for ``esm3-1.4b`` (``esm3-sm-open-v1``, ``bfloat16``):
    1. ``r2_stability_R_esm3-1.4b.npz``: ``(25856, 1024)`` ``float32``
    2. ``r3_cycles_R_esm3-1.4b.npz``: ``R_cycle`` ``(8192, 1024)`` ``float32``
    3. ``r1_R_esm3-1.4b.npz``: ``(25728, 1024)`` ``float32``

``esm3-gpu1``
    Offline GPU extraction for ``esm3-1.4b`` (``esm3-sm-open-v1``, ``bfloat16``):
    1. ``r2_abundance_R_esm3-1.4b.npz``: ``(109568, 1024)`` ``float32``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.general_plm.eval_utils import project_four_blocks
from src.capability.interactions.pairwise_epistasis import (
    PROJECTION_DIM,
    PROJECTION_SEED,
    require_projected_width,
)
from src.capability.unified.roster import PREREGISTERED_33_ARMS

DEFAULT_GPFS_STABILITY_ROOTS: tuple[Path, ...] = (
    Path(
        "/gpfs/jiaotongdamoxing/zhk_zip/InterpretabilityTransfer/results/external_baseline/20260924000148_fbfa7ddc1f5f"
    ),
    Path(
        "/gpfs/jiaotongdamoxing/zhk_zip/InterpretabilityTransfer/results/external_baseline/20260923232409_9ab5f150f2ac"
    ),
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _apply_mutation(wt: str, mutation: str) -> str:
    chars = list(wt)
    for token in mutation.split(":"):
        token = token.strip()
        if not token:
            continue
        wt_aa = token[0]
        mt_aa = token[-1]
        pos_1based = int(token[1:-1])
        idx = pos_1based - 1
        if not (0 <= idx < len(chars)):
            raise ValueError(f"Mutation {token!r} out of range for length {len(chars)}")
        if chars[idx] != wt_aa:
            raise ValueError(
                f"WT mismatch for {token!r}: expected {wt_aa!r}, got {chars[idx]!r}"
            )
        chars[idx] = mt_aa
    return "".join(chars)


def _adaptive_batch_size(seq_len: int, base_batch_size: int) -> int:
    if seq_len <= 128:
        return base_batch_size
    if seq_len <= 256:
        return max(1, min(base_batch_size, 32))
    if seq_len <= 512:
        return max(1, min(base_batch_size, 16))
    if seq_len <= 1024:
        return max(1, min(base_batch_size, 8))
    return max(1, min(base_batch_size, 2))


def _assemble_one_prereg_stability_arm(
    args: tuple[str, str, str, str, str],
) -> dict[str, Any]:
    arm, src_dir_s, cohort_path_s, cohort_sha256, out_dir_s = args
    t0 = time.monotonic()
    src_dir = Path(src_dir_s)
    out_dir = Path(out_dir_s)
    cohort = json.loads(Path(cohort_path_s).read_text(encoding="utf-8"))

    manifest_path = src_dir / f"manifest_{arm}.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{arm}: missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ident = manifest.get("identity", {})
    if ident.get("schema") != "stability_singles_extraction_v1":
        raise ValueError(f"{arm}: unexpected manifest schema {ident.get('schema')!r}")
    if manifest.get("status") != "complete" or ident.get("arm") != arm:
        raise ValueError(f"{arm}: incomplete or mismatched manifest")
    if ident.get("cohort_sha256") != cohort_sha256:
        raise ValueError(
            f"{arm}: cohort_sha256 mismatch ({ident.get('cohort_sha256')} != {cohort_sha256})"
        )

    by_name = {row["background"]: row for row in manifest["backgrounds"]}
    chunks: list[np.ndarray] = []
    for bg in cohort["backgrounds"]:
        rec = by_name.get(bg["name"])
        if rec is None:
            raise ValueError(f"{arm}: missing extraction record for background {bg['name']!r}")
        npz_path = src_dir / rec["file"]
        with np.load(npz_path, allow_pickle=False) as data:
            states = data["variant_states"]
            positions = data["variant_positions"]
            declared_states = np.asarray([int(v["state"]) for v in bg["variants"]], dtype=np.int64)
            declared_positions = np.asarray(
                [int(v["position"]) for v in bg["variants"]], dtype=np.int64
            )
            if not np.array_equal(states, declared_states):
                raise ValueError(f"{arm}: {bg['name']} variant_states differ from cohort")
            if not np.array_equal(positions, declared_positions):
                raise ValueError(f"{arm}: {bg['name']} variant_positions differ from cohort")
            projected = np.asarray(data["projected"], dtype=np.float32)
            n_states = projected.shape[0]
            flat = projected.reshape(n_states, -1)
            require_projected_width(flat.shape[1], arm=arm, source=str(npz_path))
            diff = (flat[states] - flat[0:1]).astype(np.float32, copy=False)
            chunks.append(diff)

    R = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if R.shape != (25856, 1024):
        raise RuntimeError(f"{arm}: expected R shape (25856, 1024), got {R.shape}")
    if not np.isfinite(R).all():
        raise RuntimeError(f"{arm}: non-finite entries in assembled R")

    out_path = out_dir / f"r2_stability_R_{arm}.npz"
    tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(
        tmp_path,
        R=R,
        cohort_sha256=np.asarray(cohort_sha256),
        arm=np.asarray(arm),
        source_dir=np.asarray(str(src_dir)),
        dtype_realised=np.asarray(str(ident.get("dtype") or "")),
    )
    os.replace(tmp_path, out_path)
    return {
        "arm": arm,
        "path": str(out_path),
        "shape": list(R.shape),
        "bytes": int(out_path.stat().st_size),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }


def run_assemble_prereg33_stability(
    data_root: Path,
    out_dir: Path,
    *,
    roots: tuple[Path, ...] = DEFAULT_GPFS_STABILITY_ROOTS,
    workers: int = 8,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = data_root / "results/R4/gate_stability_20260924/cohort/cohort.json"
    cohort_sha256 = _sha256_file(cohort_path)

    arm_sources: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for arm in PREREGISTERED_33_ARMS:
            cand = root / arm
            if (cand / f"manifest_{arm}.json").is_file():
                arm_sources[arm] = cand

    missing = [arm for arm in PREREGISTERED_33_ARMS if arm not in arm_sources]
    if missing:
        raise FileNotFoundError(f"Missing GPFS MegaScale stability archives for arms: {missing}")

    jobs = [
        (arm, str(arm_sources[arm]), str(cohort_path), cohort_sha256, str(out_dir))
        for arm in PREREGISTERED_33_ARMS
    ]
    t0 = time.monotonic()
    results: list[dict[str, Any]] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for item in pool.map(_assemble_one_prereg_stability_arm, jobs):
                results.append(item)
                print(
                    f"[assemble-prereg33] {len(results)}/33 {item['arm']} "
                    f"shape={item['shape']} bytes={item['bytes']} ({item['elapsed_seconds']}s)",
                    flush=True,
                )
    else:
        for job in jobs:
            item = _assemble_one_prereg_stability_arm(job)
            results.append(item)
            print(
                f"[assemble-prereg33] {len(results)}/33 {item['arm']} "
                f"shape={item['shape']} bytes={item['bytes']} ({item['elapsed_seconds']}s)",
                flush=True,
            )

    summary = {
        "mode": "assemble-prereg33-stability",
        "n_arms": len(results),
        "cohort_sha256": cohort_sha256,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "arms": results,
    }
    return summary


def _extract_esm3_projected(handle: Any, sequences: list[str], *, batch_size: int) -> np.ndarray:
    from src.capability.esm3_ext.esm3_adapter import extract_hidden_states

    h = extract_hidden_states(handle, sequences, batch_size=batch_size)
    raw_blocks = np.stack(
        [h["middle_mean"], h["middle_cls"], h["final_mean"], h["final_cls"]],
        axis=1,
    )
    return project_four_blocks(raw_blocks, dim=PROJECTION_DIM, seed=PROJECTION_SEED)


def extract_esm3_stability(
    handle: Any, data_root: Path, out_dir: Path, *, batch_size: int = 128
) -> dict[str, Any]:
    t0 = time.monotonic()
    cohort_path = data_root / "results/R4/gate_stability_20260924/cohort/cohort.json"
    cohort_sha256 = _sha256_file(cohort_path)
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))

    chunks: list[np.ndarray] = []
    backgrounds = cohort["backgrounds"]
    for idx, bg in enumerate(backgrounds, 1):
        wt = bg["wildtype"]
        mut_seqs = [v["sequence"] for v in bg["variants"]]
        bs = _adaptive_batch_size(len(wt), batch_size)
        proj = _extract_esm3_projected(handle, [wt, *mut_seqs], batch_size=bs)
        diff = (proj[1:] - proj[0:1]).astype(np.float32, copy=False)
        chunks.append(diff)
        if idx % 20 == 0 or idx == len(backgrounds):
            print(
                f"[esm3-stability] {idx}/{len(backgrounds)} backgrounds "
                f"({time.monotonic() - t0:.1f}s)",
                flush=True,
            )

    R = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if R.shape != (25856, 1024) or not np.isfinite(R).all():
        raise RuntimeError(f"esm3-1.4b stability R invalid: shape={R.shape}")
    out_path = out_dir / "r2_stability_R_esm3-1.4b.npz"
    tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(
        tmp_path,
        R=R,
        cohort_sha256=np.asarray(cohort_sha256),
        arm=np.asarray("esm3-1.4b"),
        dtype_realised=np.asarray(str(handle.dtype)),
    )
    os.replace(tmp_path, out_path)
    return {
        "task": "r2_stability",
        "path": str(out_path),
        "shape": list(R.shape),
        "bytes": int(out_path.stat().st_size),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }


def extract_esm3_r3_cycles(
    handle: Any, data_root: Path, out_dir: Path, *, batch_size: int = 128
) -> dict[str, Any]:
    t0 = time.monotonic()
    plan_path = data_root / "archive/logs/R3/pairwise_epistasis_20260924/extraction_plan.json"
    plan_sha256 = _sha256_file(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    r_chunks: list[np.ndarray] = []
    cycle_groups: list[str] = []
    cycle_seps: list[str] = []
    backgrounds = plan["backgrounds"]
    for idx, bg in enumerate(backgrounds, 1):
        seqs: list[str] = bg["sequences"]
        bs = _adaptive_batch_size(max(len(s) for s in seqs), batch_size)
        proj_states = _extract_esm3_projected(handle, seqs, batch_size=bs)
        bg_cycles = []
        for cyc in bg["cycles"]:
            i_wt, i_a, i_b, i_ab = (int(x) for x in cyc["states"])
            bg_cycles.append(
                proj_states[i_ab] - proj_states[i_a] - proj_states[i_b] + proj_states[i_wt]
            )
            cycle_groups.append(str(bg["group"]))
            cycle_seps.append(str(cyc["separation"]))
        r_chunks.append(np.stack(bg_cycles, axis=0).astype(np.float32, copy=False))
        if idx % 16 == 0 or idx == len(backgrounds):
            print(
                f"[esm3-r3-cycles] {idx}/{len(backgrounds)} backgrounds "
                f"({time.monotonic() - t0:.1f}s)",
                flush=True,
            )

    R_cycle = np.concatenate(r_chunks, axis=0).astype(np.float32, copy=False)
    if R_cycle.shape != (8192, 1024) or not np.isfinite(R_cycle).all():
        raise RuntimeError(f"esm3-1.4b R_cycle invalid: shape={R_cycle.shape}")
    out_path = out_dir / "r3_cycles_R_esm3-1.4b.npz"
    tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(
        tmp_path,
        R_cycle=R_cycle,
        cycle_group=np.asarray(cycle_groups, dtype="U"),
        cycle_separation=np.asarray(cycle_seps, dtype="U"),
        plan_sha256=np.asarray(plan_sha256),
        arm=np.asarray("esm3-1.4b"),
        dtype_realised=np.asarray(str(handle.dtype)),
    )
    os.replace(tmp_path, out_path)
    return {
        "task": "r3_cycles",
        "path": str(out_path),
        "shape": list(R_cycle.shape),
        "bytes": int(out_path.stat().st_size),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }


def extract_esm3_r1_ladder(
    handle: Any, data_root: Path, out_dir: Path, *, batch_size: int = 64
) -> dict[str, Any]:
    t0 = time.monotonic()
    vt_path = data_root / "results/extensions/statistical_ladder_20261106/inputs/variant_table.json"
    cohort_path = data_root / "results/R5/context_mutation_rescue_20260923/cohort.json"
    vt_sha256 = _sha256_file(vt_path)
    cohort_sha256 = _sha256_file(cohort_path)

    vt = json.loads(vt_path.read_text(encoding="utf-8"))
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort_by_assay = {row["assay"]: row for row in cohort["assays"]}

    by_assay: dict[str, list[dict[str, Any]]] = {}
    for s in vt["samples"]:
        by_assay.setdefault(s["assay"], []).append(s)

    assay_names = sorted(by_assay.keys())
    chunks: list[np.ndarray] = []
    sample_ids: list[str] = []
    assays_arr: list[str] = []
    clusters_arr: list[int] = []

    for idx, assay_name in enumerate(assay_names, 1):
        wt = cohort_by_assay[assay_name]["wildtype"]
        samples = by_assay[assay_name]
        mut_seqs = [_apply_mutation(wt, s["mutation"]) for s in samples]
        bs = _adaptive_batch_size(len(wt), batch_size)
        proj = _extract_esm3_projected(handle, [wt, *mut_seqs], batch_size=bs)
        diff = (proj[1:] - proj[0:1]).astype(np.float32, copy=False)
        chunks.append(diff)
        sample_ids.extend(str(s["sample_id"]) for s in samples)
        assays_arr.extend([assay_name] * len(samples))
        clusters_arr.extend(int(s["cluster"]) for s in samples)
        if idx % 25 == 0 or idx == len(assay_names):
            print(
                f"[esm3-r1] {idx}/{len(assay_names)} assays ({time.monotonic() - t0:.1f}s)",
                flush=True,
            )

    R = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if R.shape != (25728, 1024) or not np.isfinite(R).all():
        raise RuntimeError(f"esm3-1.4b R1 R invalid: shape={R.shape}")
    out_path = out_dir / "r1_R_esm3-1.4b.npz"
    tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(
        tmp_path,
        R=R,
        sample_ids=np.asarray(sample_ids, dtype="U"),
        assay=np.asarray(assays_arr, dtype="U"),
        group=np.asarray(clusters_arr, dtype=np.int64),
        variant_table_sha256=np.asarray(vt_sha256),
        cohort_sha256=np.asarray(cohort_sha256),
        arm=np.asarray("esm3-1.4b"),
        dtype_realised=np.asarray(str(handle.dtype)),
    )
    os.replace(tmp_path, out_path)
    return {
        "task": "r1_ladder",
        "path": str(out_path),
        "shape": list(R.shape),
        "bytes": int(out_path.stat().st_size),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }


def extract_esm3_abundance(
    handle: Any, data_root: Path, out_dir: Path, *, batch_size: int = 128
) -> dict[str, Any]:
    t0 = time.monotonic()
    cohort_path = data_root / "results/R1/external_confirmation_20260924/cohort.json"
    cohort_sha256 = _sha256_file(cohort_path)
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))

    chunks: list[np.ndarray] = []
    domains = cohort["domains"]
    for idx, dom in enumerate(domains, 1):
        wt = dom["wildtype"]
        mut_seqs = [v["sequence"] for v in dom["variants"]]
        bs = _adaptive_batch_size(len(wt), batch_size)
        proj = _extract_esm3_projected(handle, [wt, *mut_seqs], batch_size=bs)
        diff = (proj[1:] - proj[0:1]).astype(np.float32, copy=False)
        chunks.append(diff)
        if idx % 50 == 0 or idx == len(domains):
            print(
                f"[esm3-abundance] {idx}/{len(domains)} domains "
                f"({time.monotonic() - t0:.1f}s)",
                flush=True,
            )

    R = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if R.shape != (109568, 1024) or not np.isfinite(R).all():
        raise RuntimeError(f"esm3-1.4b abundance R invalid: shape={R.shape}")
    out_path = out_dir / "r2_abundance_R_esm3-1.4b.npz"
    tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(
        tmp_path,
        R=R,
        cohort_sha256=np.asarray(cohort_sha256),
        arm=np.asarray("esm3-1.4b"),
        dtype_realised=np.asarray(str(handle.dtype)),
    )
    os.replace(tmp_path, out_path)
    return {
        "task": "r2_abundance",
        "path": str(out_path),
        "shape": list(R.shape),
        "bytes": int(out_path.stat().st_size),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("assemble-prereg33-stability", "esm3-gpu0", "esm3-gpu1"),
    )
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("TRANSFER_DATA_ROOT") or ROOT))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--esm3-model-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "assemble-prereg33-stability":
        summary = run_assemble_prereg33_stability(
            args.data_root, args.out, workers=args.workers
        )
        print(json.dumps(summary, indent=2), flush=True)
        return

    from src.capability.esm3_ext.esm3_adapter import load_esm3_offline

    handle = load_esm3_offline(
        model_dir=args.esm3_model_dir,
        device=args.device,
        dtype=args.dtype,
        eager_load_aux=False,
    )
    if args.mode == "esm3-gpu0":
        res = [
            extract_esm3_stability(handle, args.data_root, args.out, batch_size=args.batch_size),
            extract_esm3_r3_cycles(handle, args.data_root, args.out, batch_size=args.batch_size),
            extract_esm3_r1_ladder(
                handle, args.data_root, args.out, batch_size=min(args.batch_size, 64)
            ),
        ]
    else:
        res = [
            extract_esm3_abundance(handle, args.data_root, args.out, batch_size=args.batch_size),
        ]
    print(json.dumps({"mode": args.mode, "results": res}, indent=2), flush=True)


if __name__ == "__main__":
    main()
